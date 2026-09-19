"""
Model diagnostic.

Run this after copying your model files in. It tells you whether they will work
and, if not, exactly what to change.

    python -m backend.app.check_model
"""
from __future__ import annotations

import sys
from pathlib import Path

from .config import FEATURE_COLS, STAGE1_MODEL_PATH, STAGE2_MODEL_PATH

OK, BAD, WARN, INFO = "  [OK]  ", "  [!!]  ", "  [ ? ]  ", "        "


def main() -> int:
    print("\n" + "=" * 68)
    print("  RouterGuardian — model check")
    print("=" * 68 + "\n")

    problems: list[str] = []

    # 1. files present -----------------------------------------------------
    print("1. Model files")
    for label, p in (("Stage 1 (fault)", STAGE1_MODEL_PATH),
                     ("Stage 2 (type)", STAGE2_MODEL_PATH)):
        if p.exists():
            print(f"{OK}{label}: {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"{BAD}{label}: MISSING")
            print(f"{INFO}expected at: {p}")
            problems.append(f"Copy {p.name} into {p.parent}")
    if problems:
        _summary(problems)
        return 1

    # 2. dependencies ------------------------------------------------------
    print("\n2. Dependencies")
    try:
        import xgboost  # noqa: F401
        print(f"{OK}xgboost {xgboost.__version__}")
    except ImportError:
        print(f"{BAD}xgboost is not installed")
        problems.append("pip install -r backend/requirements.txt")
        _summary(problems)
        return 1
    try:
        import sklearn  # noqa: F401
        print(f"{OK}scikit-learn {sklearn.__version__}")
    except ImportError:
        print(f"{BAD}scikit-learn is not installed (xgboost's sklearn API needs it)")
        problems.append("pip install -r backend/requirements.txt")

    # 3. load --------------------------------------------------------------
    print("\n3. Loading")
    from .model_service import ModelService
    svc = ModelService()
    if not svc.ready:
        print(f"{BAD}{svc.load_error}")
        problems.append("See the message above.")
        _summary(problems)
        return 1
    print(f"{OK}both models loaded")

    # 4. feature contract --------------------------------------------------
    print("\n4. Feature contract")
    names = svc.expected_features or []
    print(f"{INFO}model expects : {len(names)} features")
    print(f"{INFO}app default   : {len(FEATURE_COLS)} features")

    extra = [n for n in names if n not in FEATURE_COLS]
    absent = [n for n in FEATURE_COLS if n not in names]
    if not extra and not absent and list(names) == list(FEATURE_COLS):
        print(f"{OK}exact match, same order")
    else:
        if extra:
            print(f"{WARN}model uses features the app does not list by default: {extra}")
        if absent:
            print(f"{WARN}app default features the model does not use: {absent}")
        if not extra and not absent:
            print(f"{WARN}same features, different order — handled automatically")
        print(f"{OK}the app adapts to the model's own feature list, so this is fine")
    if svc.feature_warning:
        print(f"{INFO}{svc.feature_warning}")

    # 5. live prediction ---------------------------------------------------
    print("\n5. Test prediction")
    from .features import build_feature_row
    X = build_feature_row(
        events=[
            {"hours_ago": 20, "alarm_type": "equipmentAlarm", "severity": "Warning", "raise_or_clear": "Raise"},
            {"hours_ago": 3,  "alarm_type": "equipmentAlarm", "severity": "Major",   "raise_or_clear": "Raise"},
            {"hours_ago": 1,  "alarm_type": "equipmentAlarm", "severity": "Critical","raise_or_clear": "Raise"},
        ],
        device_role="core",
        history={"n_prior_faults": 5, "n_prior_hw_faults": 4,
                 "n_prior_sw_faults": 1, "days_since_last_fault": 12.0,
                 "has_pending_fault": 0},
    )
    try:
        res = svc.predict(X)
        print(f"{OK}P(fault) = {res['fault_probability']:.4f}  "
              f"-> {'FAULT PREDICTED' if res['fault_predicted'] else 'no fault'}"
              f"{' (' + str(res['fault_type']) + ')' if res['fault_type'] else ''}")
        ex = svc.explain(X)
        top = ex["contributions"][0]
        print(f"{OK}explanation works — top driver: {top['label']} ({top['contribution']:+.3f})")
    except Exception as exc:  # noqa: BLE001
        print(f"{BAD}prediction failed: {exc}")
        problems.append(str(exc))

    # 6. threshold ---------------------------------------------------------
    print("\n6. Deployment threshold")
    from .config import DEPLOY_THRESHOLD
    print(f"{INFO}config.DEPLOY_THRESHOLD = {DEPLOY_THRESHOLD}")
    print(f"{WARN}make sure this matches the 'recall-prioritized' threshold your")
    print(f"{INFO}training cell printed. If you retrained, update it in")
    print(f"{INFO}backend/app/config.py — otherwise the alert boundary is wrong.")

    _summary(problems)
    return 1 if problems else 0


def _summary(problems: list[str]) -> None:
    print("\n" + "=" * 68)
    if problems:
        print("  RESULT: not ready\n")
        for i, p in enumerate(problems, 1):
            print(f"    {i}. {p}")
    else:
        print("  RESULT: ready — start the app with  docker compose up --build")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    sys.exit(main())
