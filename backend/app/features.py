"""
Feature engineering — mirrors window_and_label.py from the training pipeline.

Every function here must stay byte-for-byte consistent with how the model was
trained. If the training feature set changes, change it here too, or predictions
will be silently wrong.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Sequence

import pandas as pd

from .config import (
    ALARM_TYPES,
    FEATURE_COLS,
    HARDWARE_TYPES,
    LOOKBACK_HOURS,
    SEVERITY_ORDER,
)


def severity_to_score(sev: str) -> int:
    """Ordinal severity: Warning=0 .. Critical=3. Unknown severities score 0."""
    try:
        return SEVERITY_ORDER.index(sev)
    except ValueError:
        return 0


def build_feature_row(
    events: Sequence[dict],
    device_role: str,
    history: dict | None = None,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Turn a router's recent alarm events into the exact 26-column feature row
    the model expects.

    events: dicts with keys hours_ago, alarm_type, severity, raise_or_clear.
            'hours_ago' is measured back from reference_time.
    history: n_prior_faults, n_prior_hw_faults, n_prior_sw_faults,
             has_pending_fault. Supplied as known context about the device.
    """
    now = reference_time or datetime.utcnow()
    history = history or {}

    rows = []
    for ev in events:
        rows.append(
            {
                "timestamp": now - timedelta(hours=float(ev["hours_ago"])),
                "alarm_type": ev.get("alarm_type", ""),
                "severity": ev.get("severity", "Warning"),
                "raise_or_clear": ev.get("raise_or_clear", "Raise"),
            }
        )

    df = pd.DataFrame(rows)
    n = len(df)
    feats: dict[str, float] = {}

    feats["n_events_total"] = n
    feats["n_raise"] = int((df["raise_or_clear"] == "Raise").sum()) if n else 0
    feats["n_clear"] = int((df["raise_or_clear"] == "Clear").sum()) if n else 0
    # Each supplied event is treated as a distinct alarm name: the simplified
    # API input does not carry alarm_name. Matches the training-time predict cell.
    feats["n_unique_alarm_names"] = n

    for atype in ALARM_TYPES:
        feats[f"n_{atype}"] = int((df["alarm_type"] == atype).sum()) if n else 0

    for sev in SEVERITY_ORDER:
        feats[f"n_severity_{sev.lower()}"] = (
            int((df["severity"] == sev).sum()) if n else 0
        )

    if n:
        scores = df["severity"].apply(severity_to_score)
        feats["mean_severity_score"] = float(scores.mean())
        feats["max_severity_score"] = float(scores.max())
        feats["hours_since_last_event"] = (
            now - df["timestamp"].max()
        ).total_seconds() / 3600.0
    else:
        feats["mean_severity_score"] = 0.0
        feats["max_severity_score"] = 0.0
        feats["hours_since_last_event"] = float(LOOKBACK_HOURS)

    half_point = now - timedelta(hours=LOOKBACK_HOURS / 2)
    first_half = int((df["timestamp"] < half_point).sum()) if n else 0
    second_half = n - first_half
    feats["first_half_count"] = first_half
    feats["second_half_count"] = second_half
    # +1 smoothing so an empty first half does not divide by zero.
    feats["trend_ratio"] = (second_half + 1) / (first_half + 1)

    feats["days_since_last_fault_device"] = history.get("days_since_last_fault", 999.0)
    feats["n_prior_faults_device"] = history.get("n_prior_faults", 0)
    feats["n_prior_hw_faults_device"] = history.get("n_prior_hw_faults", 0)
    feats["n_prior_sw_faults_device"] = history.get("n_prior_sw_faults", 0)
    feats["has_pending_fault_device"] = int(history.get("has_pending_fault", 0))

    role = (device_role or "access").lower()
    feats["role_access"] = 1 if role == "access" else 0
    feats["role_core"] = 1 if role == "core" else 0
    feats["role_edge"] = 1 if role == "edge" else 0

    # Return every feature we know how to compute. ModelService selects and
    # orders the subset that the loaded model actually expects, so a model
    # trained with a slightly different feature set still works.
    row = pd.DataFrame([feats])
    ordered = FEATURE_COLS + [c for c in row.columns if c not in FEATURE_COLS]
    return row[ordered]


def events_from_dataframe(
    df: pd.DataFrame, window_end: datetime, lookback_hours: int = LOOKBACK_HOURS
) -> list[dict]:
    """Extract the events inside [window_end - lookback, window_end) as
    hours_ago dicts, so a CSV log can reuse build_feature_row."""
    start = window_end - timedelta(hours=lookback_hours)
    win = df[(df["timestamp"] >= start) & (df["timestamp"] < window_end)]
    out = []
    for _, r in win.iterrows():
        out.append(
            {
                "hours_ago": (window_end - r["timestamp"]).total_seconds() / 3600.0,
                "alarm_type": r.get("alarm_type", ""),
                "severity": r.get("severity", "Warning"),
                "raise_or_clear": r.get("raise_or_clear", "Raise"),
            }
        )
    return out


def derive_history_from_log(
    device_df: pd.DataFrame, window_end: datetime
) -> dict:
    """
    Reconstruct per-device history features from a raw alarm log, counting
    fault episodes that began strictly before window_end (never at or after,
    so no label leakage).

    A row counts as a fault episode if the log carries an `is_fault_episode`
    flag; otherwise Critical-severity Raise events are used as a proxy.
    """
    prior = device_df[device_df["timestamp"] < window_end]

    if "is_fault_episode" in prior.columns:
        faults = prior[
            prior["is_fault_episode"].astype(str).str.lower().isin(["true", "1"])
            & (prior["raise_or_clear"] == "Raise")
        ]
    else:
        faults = prior[
            (prior["severity"] == "Critical") & (prior["raise_or_clear"] == "Raise")
        ]

    n_total = len(faults)
    n_hw = int(faults["alarm_type"].isin(HARDWARE_TYPES).sum()) if n_total else 0

    # Only used if the loaded model was trained with this feature. 999 is the
    # "never faulted" sentinel the training pipeline used.
    if n_total:
        days_since = (window_end - faults["timestamp"].max()).total_seconds() / 86400.0
    else:
        days_since = 999.0

    # Unresolved if the most recent fault Raise has no later Clear.
    has_pending = 0
    if n_total:
        last_fault_ts = faults["timestamp"].max()
        cleared = prior[
            (prior["raise_or_clear"] == "Clear") & (prior["timestamp"] > last_fault_ts)
        ]
        has_pending = int(len(cleared) == 0)

    return {
        "n_prior_faults": n_total,
        "n_prior_hw_faults": n_hw,
        "n_prior_sw_faults": n_total - n_hw,
        "days_since_last_fault": round(days_since, 3),
        "has_pending_fault": has_pending,
    }
