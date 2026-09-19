"""
Model loading and inference.

Stage 1: will ANY fault occur in the next 12 hours?
Stage 2: if so, is it hardware (alert the team) or software (auto-remediate)?

Explanations use XGBoost's built-in `pred_contribs=True`, which returns exact
TreeSHAP contributions without needing the heavyweight `shap` package.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEPLOY_THRESHOLD,
    FEATURE_COLS,
    FEATURE_LABELS,
    HORIZON_HOURS,
    STAGE1_MODEL_PATH,
    STAGE2_MODEL_PATH,
)

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before models are available."""


class ModelService:
    def __init__(self) -> None:
        self.stage1 = None
        self.stage2 = None
        self.load_error: str | None = None
        # Feature names the loaded model actually expects, read from the model
        # file itself. The app adapts to this rather than assuming.
        self.expected_features: list[str] | None = None
        self.feature_warning: str | None = None
        self.load()

    # ------------------------------------------------------- feature contract
    @staticmethod
    def _model_feature_names(model) -> list[str] | None:
        """Feature names stored inside the model file, if it has them."""
        try:
            booster = model.get_booster()
            if booster.feature_names:
                return list(booster.feature_names)
        except Exception:  # noqa: BLE001
            pass
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return list(names)
        return None

    def _resolve_features(self) -> None:
        """
        Work out which features the loaded model wants, and whether this app can
        supply them. Sets expected_features, or load_error if it cannot comply.
        """
        from .features import build_feature_row

        names = self._model_feature_names(self.stage1)
        n_expected = getattr(self.stage1, "n_features_in_", None)

        # What this app is able to compute.
        available = set(build_feature_row([], "access", {}).columns)

        if names:
            missing = [n for n in names if n not in available]
            if missing:
                self.load_error = (
                    f"The model expects {len(names)} features, but this app cannot build "
                    f"{len(missing)} of them: {missing[:6]}"
                    + ("…" if len(missing) > 6 else "")
                    + ". Retrain with the app's feature set, or update FEATURE_COLS and "
                      "features.build_feature_row to match your training pipeline. "
                      "Run:  python -m backend.app.check_model  for a full report."
                )
                self.stage1 = self.stage2 = None
                return
            self.expected_features = names
            if list(names) != list(FEATURE_COLS):
                extra = [n for n in names if n not in FEATURE_COLS]
                absent = [n for n in FEATURE_COLS if n not in names]
                bits = []
                if extra:
                    bits.append(f"uses {extra}")
                if absent:
                    bits.append(f"does not use {absent}")
                if bits:
                    self.feature_warning = (
                        "Model feature set differs from the app default: "
                        + "; ".join(bits) + ". Adapting automatically."
                    )
                    logger.warning(self.feature_warning)
            return

        # No names stored — fall back to positional, but only if the count matches.
        if n_expected and int(n_expected) != len(FEATURE_COLS):
            self.load_error = (
                f"The model expects {int(n_expected)} features but this app builds "
                f"{len(FEATURE_COLS)}, and the model file stores no feature names, so "
                "they cannot be matched up safely. Re-save the model from a pandas "
                "DataFrame so feature names are preserved, or align FEATURE_COLS. "
                "Run:  python -m backend.app.check_model"
            )
            self.stage1 = self.stage2 = None
            return
        self.expected_features = list(FEATURE_COLS)

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and order columns exactly as the loaded model expects."""
        cols = self.expected_features or FEATURE_COLS
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ModelNotLoadedError(f"Cannot build required features: {missing}")
        return X[cols]

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            self.load_error = f"xgboost is not installed: {exc}"
            logger.error(self.load_error)
            return

        missing = [
            str(p) for p in (STAGE1_MODEL_PATH, STAGE2_MODEL_PATH) if not p.exists()
        ]
        if missing:
            self.load_error = (
                "Model files not found: "
                + ", ".join(missing)
                + ". Copy stage1_fault_model.json and stage2_type_model.json "
                "from your Kaggle /kaggle/working/ directory into backend/models/."
            )
            logger.warning(self.load_error)
            return

        try:
            self.stage1 = XGBClassifier()
            self.stage1.load_model(str(STAGE1_MODEL_PATH))
            self.stage2 = XGBClassifier()
            self.stage2.load_model(str(STAGE2_MODEL_PATH))
            self.load_error = None
            self.feature_warning = None
            self._resolve_features()
            if self.load_error:
                logger.error(self.load_error)
                return
            logger.info("Both models loaded successfully. Features: %d%s",
                        len(self.expected_features or []),
                        " (adapted)" if self.feature_warning else "")
        except Exception as exc:  # noqa: BLE001 - surface any load failure to the API
            self.stage1 = self.stage2 = None
            self.load_error = f"Failed to load models: {exc}"
            logger.exception(self.load_error)

    @property
    def ready(self) -> bool:
        return self.stage1 is not None and self.stage2 is not None

    def _require_ready(self) -> None:
        if not self.ready:
            raise ModelNotLoadedError(self.load_error or "Models are not loaded.")

    # ------------------------------------------------------------- predict
    @staticmethod
    def _verdict(fault_proba: float, thr: float) -> dict[str, Any]:
        """The Stage 1 half of a decision. One place, so wording never diverges."""
        return {
            "fault_probability": round(fault_proba, 4),
            "threshold": thr,
            "fault_predicted": fault_proba >= thr,
            "fault_type": None,
            "type_confidence": None,
            "action": "none",
            "action_detail": f"No fault predicted in the next {HORIZON_HOURS} hours.",
            "horizon_hours": HORIZON_HOURS,
        }

    @staticmethod
    def _type_verdict(type_pred: int, type_proba: float) -> dict[str, Any]:
        """The Stage 2 half: which remediation path this fault is routed down."""
        is_hw = type_pred == 1
        return {
            "fault_type": "hardware" if is_hw else "software",
            "type_confidence": round(type_proba, 4),
            "action": "alert_team" if is_hw else "auto_remediate",
            "action_detail": (
                "Hardware fault predicted — not auto-fixable. Alert the on-call team."
                if is_hw
                else "Software fault predicted — attempt scripted remediation."
            ),
        }

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> dict[str, Any]:
        """Run both stages on a single feature row and derive the action."""
        self._require_ready()
        thr = DEPLOY_THRESHOLD if threshold is None else float(threshold)

        Xa = self._align(X)
        fault_proba = float(self.stage1.predict_proba(Xa)[:, 1][0])
        result = self._verdict(fault_proba, thr)

        if fault_proba >= thr:
            type_pred = int(self.stage2.predict(Xa)[0])
            type_proba = float(self.stage2.predict_proba(Xa)[0][type_pred])
            result.update(self._type_verdict(type_pred, type_proba))

        return result

    def predict_many(self, X: pd.DataFrame,
                     threshold: float | None = None) -> list[dict[str, Any]]:
        """
        Run both stages over many rows in one pass.

        Scoring a fleet router by router costs roughly 80ms per call in wrapper
        overhead alone — DataFrame validation and DMatrix construction dominate,
        not the trees. At 34 routers that is nearly three seconds on every page
        load, and it grows linearly with the fleet. One batched call collapses
        that to a single pass. Stage 2 runs only on the rows Stage 1 flagged,
        which is exactly the same decision path as predict(), just vectorised.
        """
        self._require_ready()
        thr = DEPLOY_THRESHOLD if threshold is None else float(threshold)
        if X.empty:
            return []

        Xa = self._align(X)
        probas = self.stage1.predict_proba(Xa)[:, 1]
        out = [self._verdict(float(p), thr) for p in probas]

        hot = [i for i, p in enumerate(probas) if p >= thr]
        if hot:
            Xh = Xa.iloc[hot]
            types = self.stage2.predict(Xh)
            tprobs = self.stage2.predict_proba(Xh)
            for k, i in enumerate(hot):
                tp = int(types[k])
                out[i].update(self._type_verdict(tp, float(tprobs[k][tp])))
        return out

    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Stage 1 probabilities for many rows at once (fleet scoring)."""
        self._require_ready()
        return self.stage1.predict_proba(self._align(X))[:, 1]

    # --------------------------------------------------------- explanation
    def explain(self, X: pd.DataFrame, top_n: int = 8) -> dict[str, Any]:
        """
        Exact per-prediction TreeSHAP contributions for Stage 1, in log-odds.
        Positive values push toward 'fault', negative toward 'no fault'.
        """
        self._require_ready()
        import xgboost as xgb

        Xa = self._align(X)
        booster = self.stage1.get_booster()
        dm = xgb.DMatrix(Xa, feature_names=list(Xa.columns))
        contribs = booster.predict(dm, pred_contribs=True)[0]

        base_value = float(contribs[-1])
        values = contribs[:-1]

        rows = []
        for name, val in zip(list(Xa.columns), values):
            rows.append(
                {
                    "feature": name,
                    "label": FEATURE_LABELS.get(name, name),
                    "value": float(Xa.iloc[0][name]),
                    "contribution": round(float(val), 4),
                }
            )

        rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
        return {
            "base_value": round(base_value, 4),
            "contributions": rows[:top_n],
            "all_contributions": rows,
        }

    # -------------------------------------------------------------- status
    def info(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "error": self.load_error,
            "deploy_threshold": DEPLOY_THRESHOLD,
            "n_features": len(self.expected_features or FEATURE_COLS),
            "features": self.expected_features or FEATURE_COLS,
            "app_default_features": FEATURE_COLS,
            "feature_warning": self.feature_warning,
            "stage1_path": str(STAGE1_MODEL_PATH),
            "stage2_path": str(STAGE2_MODEL_PATH),
            "notes": {
                "excluded_feature": "days_since_last_fault_device",
                "exclusion_reason": (
                    "Encoded a simulator artifact (the 999 'never faulted' "
                    "sentinel) and inverted the chronic-vs-healthy risk ranking."
                ),
            },
        }


model_service = ModelService()
