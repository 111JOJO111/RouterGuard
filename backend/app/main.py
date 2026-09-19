"""
RouterGuardian API — router fault prediction from Huawei alarm logs.

The app is database-driven: routers and their alarm streams live in SQLite, so
predictions are made by selecting a router from the topology rather than by
typing alarms in by hand.

Stage 1 predicts whether a fault occurs in the next 12 hours; Stage 2 classifies
it hardware (alert the team) vs software (attempt automatic remediation).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import assistant
from . import database as db
from . import notifications
from .config import (
    DEPLOY_THRESHOLD,
    DISPLAY_METRICS,
    FEATURE_COLS,
    HARDWARE_TYPES,
    HORIZON_HOURS,
    LOOKBACK_HOURS,
    MEASURED_METRICS,
)
from .features import build_feature_row, derive_history_from_log, events_from_dataframe
from .model_service import ModelNotLoadedError, model_service
from .schemas import ChatRequest, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the schema at import time as well as on startup, so the API is safe
# even if it is mounted without lifespan events running (tests, some ASGI hosts).
try:
    db.init_db()
except Exception as _exc:  # noqa: BLE001
    logger.warning("Could not initialise database at import: %s", _exc)

app = FastAPI(
    title="RouterGuardian API",
    description="Predicts router faults from alarm logs and routes them to the right remediation path.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    try:
        info = db.autoseed_if_available()
        if info:
            logger.info("Auto-seeded database: %s", info)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-seed skipped: %s", exc)


def _risk_band(p: float) -> str:
    if p >= DEPLOY_THRESHOLD:
        return "critical"
    if p >= 0.40:
        return "elevated"
    if p >= 0.15:
        return "watch"
    return "normal"


def _parse_at(at: str | None) -> datetime | None:
    """Optional 'score the fleet as of this moment' timestamp from the query."""
    if not at:
        return None
    try:
        return datetime.fromisoformat(at.replace("Z", "").replace(" ", "T"))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"'at' must be an ISO timestamp such as 2026-08-11T04:00:00 (got {at!r}).",
        ) from exc


def _require_seeded() -> None:
    if not db.is_seeded():
        raise HTTPException(
            status_code=409,
            detail="No network data loaded. Upload an alarm-log CSV on the Data tab "
                   "(or place it at backend/data/alarm_log.csv) to build the topology.",
        )


def _score_device(device_id: str, at: datetime | None = None,
                  explain: bool = False) -> dict:
    """
    Score one router from its alarm history.

    If the model is not loaded this still returns the router's real alarm and
    history data with the risk fields set to None, so the whole app keeps
    working — only the predictions are missing.
    """
    dev = db.get_device(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail=f"Unknown device '{device_id}'.")

    when = at or db.reference_time() + timedelta(seconds=1)
    alarms = db.get_alarms(device_id, before=when)
    events = events_from_dataframe(alarms, when, LOOKBACK_HOURS) if not alarms.empty else []
    history = (
        derive_history_from_log(alarms, when)
        if not alarms.empty
        else {"n_prior_faults": 0, "n_prior_hw_faults": 0,
              "n_prior_sw_faults": 0, "has_pending_fault": 0}
    )

    out: dict = {
        "device_id": device_id,
        "device_role": dev["device_role"],
        "device_type": dev["device_type"],
        "site": dev["site"],
        "model": dev["model"],
        "is_monitored": bool(dev["is_monitored"]),
        "n_downstream": dev["n_downstream"],
        "n_downstream_endpoints": dev["n_downstream_endpoints"],
        "scored_at": when.isoformat(),
        "n_events_window": len(events),
        "history": history,
        "fault_probability": None,
        "threshold": DEPLOY_THRESHOLD,
        "fault_predicted": None,
        "fault_type": None,
        "type_confidence": None,
        "action": "unknown",
        "action_detail": "Model offline — no prediction available.",
        "risk_band": "unscored",
        "scored": False,
    }

    # Switches and end devices carry no alarms of their own; their status is
    # inherited from the router upstream of them.
    if not dev["is_monitored"]:
        out["action_detail"] = "Passive device — status follows its upstream router."
        return out

    if not model_service.ready:
        return out

    role = dev["device_role"] if dev["device_role"] in ("core", "edge", "access") else "access"
    X = build_feature_row(events, role, history)
    res = model_service.predict(X)
    out.update(res)
    out["scored"] = True
    out["risk_band"] = _risk_band(res["fault_probability"])
    out["features"] = {k: float(X.iloc[0][k]) for k in FEATURE_COLS}
    if explain:
        ex = model_service.explain(X)
        out["explanation"] = {"base_value": ex["base_value"],
                              "contributions": ex["contributions"]}
    return out


# ------------------------------------------------------------------ status
@app.get("/api/health", tags=["status"])
def health():
    return {
        "status": "ok" if model_service.ready else "degraded",
        "models_loaded": model_service.ready,
        "detail": model_service.load_error,
        "data_loaded": db.is_seeded(),
        "n_devices": int(db.get_meta("n_devices") or 0),
        "n_routers": int(db.get_meta("n_routers") or 0),
        "data_source": db.get_meta("source") or "csv",
        "n_alarms": int(db.get_meta("n_alarms") or 0),
        "reference_time": db.get_meta("reference_time"),
        "dictionary_loaded": db.dictionary_size() > 0,
        "n_dictionary": db.dictionary_size(),
        # The prediction contract, read from config so the UI never hardcodes it.
        "lookback_hours": LOOKBACK_HOURS,
        "horizon_hours": HORIZON_HOURS,
        "threshold": DEPLOY_THRESHOLD,
        "alerting": notifications.delivery_status(),
        "assistant": assistant.status(),
        "n_alerts": db.alert_stats()["n_alerts"],
    }


@app.get("/api/metrics", tags=["status"])
def metrics():
    """Metrics shown in the UI, alongside the values actually measured."""
    return {"display": DISPLAY_METRICS, "measured": MEASURED_METRICS,
            "deploy_threshold": DEPLOY_THRESHOLD,
            "lookback_hours": LOOKBACK_HOURS,
            "horizon_hours": HORIZON_HOURS}


@app.get("/api/model/info", tags=["status"])
def model_info():
    return model_service.info()


# -------------------------------------------------------------------- data
@app.post("/api/seed", tags=["data"])
async def seed(file: UploadFile = File(...)):
    """Load an alarm-log CSV and rebuild the topology database from it."""
    try:
        info = db.seed_from_csv_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not import CSV: {exc}") from exc
    return {"status": "ok", **info}


@app.post("/api/dictionary", tags=["dictionary"])
async def upload_dictionary(file: UploadFile = File(...)):
    """
    Load the Huawei alarm dictionary (huawei_alarm_dictionary_final.csv).

    Optional. With it loaded, alarms are shown with their official description,
    system impact, probable causes and Huawei's documented remediation steps.
    It is kept separately from the network data, so re-seeding the topology does
    not remove it.

    If the network currently loaded is the built-in demo, it is regenerated
    afterwards. The demo draws its alarm names from the dictionary, so a demo
    built before the dictionary arrived is full of alarms the dictionary has
    never heard of — which is exactly what produced rows reading "no definition".
    Regenerating means every alarm in the demo resolves to a real Huawei entry.
    """
    try:
        info = db.seed_dictionary_from_bytes(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not import dictionary: {exc}") from exc

    regenerated = False
    if db.get_meta("source") == "demo":
        try:
            db.seed_demo()
            regenerated = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not regenerate the demo network: %s", exc)

    return {"status": "ok", "demo_regenerated": regenerated, **info}


@app.get("/api/dictionary/status", tags=["dictionary"])
def dictionary_status():
    n = db.dictionary_size()
    return {"loaded": n > 0, "n_entries": n}


@app.get("/api/dictionary/search", tags=["dictionary"])
def dictionary_search(q: str = Query(..., min_length=2), limit: int = Query(40, ge=1, le=200)):
    if db.dictionary_size() == 0:
        raise HTTPException(status_code=409, detail="No alarm dictionary loaded.")
    return {"query": q, "results": db.search_dictionary(q, limit)}


@app.get("/api/devices/{device_id}/remediation", tags=["dictionary"])
def device_remediation(device_id: str, hours: float = Query(168.0, ge=1, le=8760),
                       limit: int = Query(20, ge=1, le=80)):
    """
    What actually went wrong on this router, and how to fix it.

    Returns two lists:
      * faults  — the alarms that were logged as real fault episodes, each with
                  its documented remediation. This is the list that matters.
      * other   — the remaining alarm activity in the window, for context.

    Both are enriched from the alarm dictionary when one is loaded.
    """
    dev = db.get_device(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail=f"Unknown device '{device_id}'.")

    when = db.reference_time() + timedelta(seconds=1)
    alarms = db.get_alarms(device_id, before=when, hours=hours)
    dict_loaded = db.dictionary_size() > 0

    def group(df) -> list[dict]:
        out: dict[tuple, dict] = {}
        for _, r in df.iterrows():
            key = (r.get("alarm_name"), r.get("raise_or_clear"))
            g = out.setdefault(key, {
                "alarm_name": r.get("alarm_name"), "alarm_id": r.get("alarm_id"),
                "raise_or_clear": r.get("raise_or_clear"), "severity": r.get("severity"),
                "alarm_type": r.get("alarm_type"), "count": 0,
                "first_seen": r["timestamp"], "last_seen": r["timestamp"],
            })
            g["count"] += 1
            g["first_seen"] = min(g["first_seen"], r["timestamp"])
            g["last_seen"] = max(g["last_seen"], r["timestamp"])
        return list(out.values())

    def enrich(items, is_fault) -> list[dict]:
        SEV = {"Critical": 0, "Major": 1, "Minor": 2, "Warning": 3}
        items = sorted(items, key=lambda g: (SEV.get(g["severity"], 4), -g["count"]))
        res = []
        for g in items:
            e = {**g,
                 "first_seen": g["first_seen"].isoformat(),
                 "last_seen": g["last_seen"].isoformat(),
                 "is_fault": is_fault,
                 "category": "hardware" if g["alarm_type"] in HARDWARE_TYPES else "software",
                 "definition": None}
            if dict_loaded:
                d = db.lookup_alarm(alarm_name=g["alarm_name"], alarm_id=g["alarm_id"],
                                    raise_or_clear=g["raise_or_clear"])
                if d:
                    e["definition"] = {k: d.get(k) for k in
                                       ("description", "impact_on_system", "possible_causes",
                                        "procedure", "mnemonic_code", "parameters")}
            res.append(e)
        return res

    if alarms.empty:
        faults, other, n_fault_events, fault_log = [], [], 0, []
    else:
        fmask = (alarms["is_fault_episode"] == 1) & (alarms["raise_or_clear"] == "Raise")
        n_fault_events = int(fmask.sum())
        # Faults are never truncated: if this router failed 11 times, all 11
        # distinct alarms come back with their fix. Only the routine background
        # activity is capped.
        faults = enrich(group(alarms[fmask]), True)
        other = enrich(group(alarms[~fmask]), False)[:limit]
        # The individual fault events too, in order, so the UI can list each
        # occurrence and not just the distinct alarm names.
        fault_log = [
            {"timestamp": r["timestamp"].isoformat(),
             "alarm_name": r.get("alarm_name"), "alarm_id": r.get("alarm_id"),
             "alarm_type": r.get("alarm_type"), "severity": r.get("severity"),
             "category": "hardware" if r.get("alarm_type") in HARDWARE_TYPES else "software"}
            for _, r in alarms[fmask].sort_values("timestamp", ascending=False).iterrows()
        ]

    all_items = faults + other
    n_matched = sum(1 for i in all_items if i["definition"])
    n_faults_matched = sum(1 for i in faults if i["definition"])
    return {
        "device_id": device_id,
        "window_hours": hours,
        "dictionary_loaded": dict_loaded,
        "n_alarms": int(len(alarms)),
        "n_fault_events": n_fault_events,
        "n_distinct_faults": len(faults),
        "n_distinct_other": len(other),
        "n_matched_to_dictionary": n_matched,
        "n_faults_matched_to_dictionary": n_faults_matched,
        "faults": faults,
        "fault_log": fault_log,
        "other": other,
        # kept so older clients keep working
        "alarms": all_items,
    }


@app.post("/api/seed/demo", tags=["data"])
def seed_demo_network(n_core: int = 4, n_edge: int = 8,
                      n_access: int = 22, days: int = 120,
                      n_active_incidents: int = 7):
    """
    Generate and load a demo network.

    n_active_incidents routers are left mid-cascade at the end of the data, so
    the model has something to predict and the fleet shows real alerts. If an
    alarm dictionary is loaded, alarm names are taken from it so the remediation
    panel resolves every alarm.
    """
    return {"status": "ok", **db.seed_demo(
        n_core=n_core, n_edge=n_edge, n_access=n_access, days=days,
        n_active_incidents=n_active_incidents)}


@app.get("/api/topology", tags=["data"])
def get_topology():
    _require_seeded()
    return db.topology()


@app.get("/api/devices", tags=["data"])
def get_devices(routers_only: bool = False):
    _require_seeded()
    return {"devices": db.list_devices(monitored_only=routers_only)}


@app.get("/api/devices/{device_id}", tags=["data"])
def get_device(device_id: str):
    dev = db.get_device(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail=f"Unknown router '{device_id}'.")
    return dev


@app.get("/api/devices/{device_id}/alarms", tags=["data"])
def get_device_alarms(device_id: str, hours: float | None = Query(None, ge=0),
                      limit: int = Query(500, ge=1, le=10000)):
    """Recent alarms for one router, newest first."""
    if not db.get_device(device_id):
        raise HTTPException(status_code=404, detail=f"Unknown router '{device_id}'.")
    ref = db.reference_time() + timedelta(seconds=1)
    df = db.get_alarms(device_id, before=ref, hours=hours)
    df = df.sort_values("timestamp", ascending=False).head(limit) if not df.empty else df
    records = [] if df.empty else [
        {**{k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in r.items()}} for r in df.to_dict("records")
    ]
    return {"device_id": device_id, "count": len(records), "alarms": records}


@app.get("/api/devices/{device_id}/alarms.csv", tags=["data"],
         response_class=PlainTextResponse)
def download_device_alarms(device_id: str):
    """Download this router's full alarm log as CSV."""
    if not db.get_device(device_id):
        raise HTTPException(status_code=404, detail=f"Unknown router '{device_id}'.")
    return PlainTextResponse(
        db.alarms_to_csv(device_id),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{device_id}_alarms.csv"'},
    )


@app.get("/api/alarms.csv", tags=["data"], response_class=PlainTextResponse)
def download_full_log():
    """Download the entire fleet's alarm log as CSV."""
    _require_seeded()
    return PlainTextResponse(
        db.full_log_to_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fleet_alarm_log.csv"'},
    )


# ----------------------------------------------------------------- predict
@app.get("/api/devices/{device_id}/predict", tags=["predict"])
def predict_device(device_id: str, explain: bool = True, at: str | None = None):
    """
    Predict fault risk for a router selected from the topology.

    `at` scores the router as it looked at that moment instead of at the newest
    alarm in the log.
    """
    _require_seeded()
    return _score_device(device_id, at=_parse_at(at), explain=explain)


@app.post("/api/predict", response_model=PredictResponse, tags=["predict"])
def predict_manual(req: PredictRequest):
    """Score an arbitrary alarm pattern. Kept for API/testing use."""
    X = build_feature_row(
        events=[e.model_dump() for e in req.events],
        device_role=req.device_role,
        history=req.device_history.model_dump(),
    )
    try:
        result = model_service.predict(X, threshold=req.threshold)
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    explanation = None
    if req.include_explanation:
        ex = model_service.explain(X)
        explanation = {"base_value": ex["base_value"], "contributions": ex["contributions"]}
    return PredictResponse(
        device_id=req.device_id,
        features={k: float(X.iloc[0][k]) for k in FEATURE_COLS},
        explanation=explanation,
        **result,
    )


# ------------------------------------------------------------------- fleet
@app.get("/api/moments", tags=["fleet"])
def moments(limit: int = Query(8, ge=1, le=40)):
    """
    Timestamps worth scoring the fleet at.

    An uploaded historical log almost never ends during an incident, so scoring
    only at its newest alarm makes every router look healthy even when the log
    contains hundreds of real failures. Each moment here is positioned one hour
    before a real fault episode began, which is inside the 12-hour horizon the
    model predicts over — so at that moment the correct answer is "fault coming".
    """
    _require_seeded()
    ref = db.reference_time()
    out = [{
        "label": "Now (newest alarm in the log)",
        "timestamp": ref.isoformat(),
        "device_id": None,
        "why": "The live view. Scores every router on the last 24 hours of the log.",
    }]

    seen: set[str] = set()
    for f in db.fault_onsets(limit=600):
        did = f["device_id"]
        if did in seen:
            continue
        try:
            onset = datetime.fromisoformat(f["timestamp"])
        except (TypeError, ValueError):
            continue
        # One hour before onset: the fault is still in the future, so this is a
        # genuine prediction rather than a reading of an alarm already raised.
        when = onset - timedelta(hours=1)
        if when <= db.reference_time() - timedelta(days=3650):
            continue
        seen.add(did)
        out.append({
            "label": f"1h before {did} failed — {when.strftime('%d %b %Y %H:%M')}",
            "timestamp": when.isoformat(),
            "device_id": did,
            "why": (f"{did} logged a real {f.get('severity') or ''} fault "
                    f"({f.get('alarm_name') or 'unnamed alarm'}) one hour after this "
                    "moment, so a correct model should already be flagging it."),
        })
        if len(out) > limit:
            break
    return {"reference_time": ref.isoformat(), "moments": out[:limit + 1]}


@app.get("/api/fleet", tags=["fleet"])
def fleet(at: str | None = None, notify: bool = True):
    """
    Score every router in the topology, ranked by risk.

    `at` scores the whole fleet as it looked at that moment. See /api/moments for
    timestamps where the log is known to contain a real failure just ahead.

    `notify=false` suppresses on-call alerting for this call — used by the CSV
    export and by historical replays, which should not page anyone.
    """
    _require_seeded()
    when = (_parse_at(at) or db.reference_time()) + timedelta(seconds=1)

    # Build every router's feature row first, then score them all in one pass.
    # Scoring device by device spends most of its time in per-call model
    # overhead rather than in the model itself, which made a page load wait
    # several seconds on a 34-router demo and would be far worse on a real fleet.
    rows: list[dict] = []
    feature_rows: list[pd.DataFrame] = []
    scored_at: list[int] = []

    by_device = db.get_alarms_by_device(before=when)
    empty = pd.DataFrame(columns=["timestamp", "alarm_type", "severity",
                                  "raise_or_clear", "is_fault_episode"])
    for dev in db.list_routers():
        alarms = by_device.get(dev["device_id"], empty)
        events = (events_from_dataframe(alarms, when, LOOKBACK_HOURS)
                  if not alarms.empty else [])
        history = (derive_history_from_log(alarms, when) if not alarms.empty
                   else {"n_prior_faults": 0, "n_prior_hw_faults": 0,
                         "n_prior_sw_faults": 0, "has_pending_fault": 0})
        rows.append({
            "device_id": dev["device_id"], "device_role": dev["device_role"],
            "device_type": dev["device_type"], "site": dev["site"],
            "model": dev["model"],
            "fault_probability": None, "fault_predicted": None,
            "fault_type": None, "action": "unknown",
            "risk_band": "unscored", "scored": False,
            "n_events_window": len(events),
            "n_downstream": dev["n_downstream"],
            "n_downstream_endpoints": dev["n_downstream_endpoints"],
            "n_prior_faults": history["n_prior_faults"],
        })
        if model_service.ready:
            role = (dev["device_role"]
                    if dev["device_role"] in ("core", "edge", "access") else "access")
            feature_rows.append(build_feature_row(events, role, history))
            scored_at.append(len(rows) - 1)

    if feature_rows:
        X = pd.concat(feature_rows, ignore_index=True)
        for i, res in zip(scored_at, model_service.predict_many(X)):
            rows[i].update({
                "fault_probability": res["fault_probability"],
                "fault_predicted": res["fault_predicted"],
                "fault_type": res["fault_type"],
                "action": res["action"],
                "risk_band": _risk_band(res["fault_probability"]),
                "scored": True,
            })

    rows.sort(key=lambda d: (d["fault_probability"] is None,
                             -(d["fault_probability"] or 0)))
    by_band: dict[str, int] = {}
    for r in rows:
        by_band[r["risk_band"]] = by_band.get(r["risk_band"], 0) + 1
    n_at_risk = sum(1 for r in rows if r["fault_predicted"])
    scored = [r["fault_probability"] for r in rows if r["fault_probability"] is not None]

    # ---- act on the predictions -----------------------------------------
    # A prediction nobody is told about is not worth much, so crossing the
    # threshold raises an alert here. raise_alert() enforces a per-router
    # cooldown, which matters because this endpoint runs on every page load and
    # every CSV download — without it one flagged router would page the on-call
    # team dozens of times an hour.
    alerts_raised = []
    if notify and model_service.ready:
        for r in rows:
            if not r["fault_predicted"]:
                continue
            # Check the cooldown BEFORE re-scoring. Building an alert means a
            # second scoring pass with explanations plus dictionary lookups, and
            # doing that only to discard it would make every page load pay for
            # alerts it is not going to send.
            if notifications.cooldown_remaining(r["device_id"]) > 0:
                continue
            try:
                full = _score_device(r["device_id"], at=when, explain=True)
                a = notifications.raise_alert(r["device_id"], full)
                if a:
                    alerts_raised.append({"id": a["id"], "device_id": a["device_id"],
                                          "kind": a["kind"], "status": a["status"]})
            except Exception as exc:  # noqa: BLE001 - alerting must never break scoring
                logger.warning("Could not raise an alert for %s: %s", r["device_id"], exc)

    # When nothing is flagged, say why. "0 predicted faults" on its own reads as
    # a broken app; it usually means the log simply ends on a quiet stretch.
    quiet_note = None
    if model_service.ready and n_at_risk == 0 and scored:
        top = max(scored)
        quiet_note = (
            f"No router is above the {DEPLOY_THRESHOLD:.0%} alert threshold at this "
            f"moment — the highest score in the fleet is {top:.1%}. That is the "
            f"expected result for a {LOOKBACK_HOURS}-hour window with little alarm "
            "activity in it. Pick a different moment above (each one sits one hour "
            "before a real failure in this log) to see the alert and "
            "auto-remediation paths fire."
        )

    return {
        "analyzed_at": when.isoformat(),
        "requested_at": at,
        "reference_time": db.reference_time().isoformat(),
        "n_devices": len(rows),
        "models_loaded": model_service.ready,
        "n_at_risk": n_at_risk,
        "n_hardware": sum(1 for r in rows if r["fault_type"] == "hardware"),
        "n_software": sum(1 for r in rows if r["fault_type"] == "software"),
        "max_probability": max(scored) if scored else None,
        "quiet_note": quiet_note,
        "alerts_raised": alerts_raised,
        "by_band": by_band,
        "threshold": DEPLOY_THRESHOLD,
        "horizon_hours": HORIZON_HOURS,
        "lookback_hours": LOOKBACK_HOURS,
        "devices": rows,
    }


@app.get("/api/fleet.csv", tags=["fleet"], response_class=PlainTextResponse)
def fleet_csv(at: str | None = None):
    """
    Download the fleet risk report as CSV.

    Every column is named so it can be read without the app open. In particular
    there is no "fault type" for a router with no predicted fault: Stage 2 only
    runs on routers Stage 1 flagged, so the cell is left blank and the
    `fault_predicted_next_12h` column carries the yes/no.
    """
    # notify=False: downloading a report is not an incident and must not page anyone.
    data = fleet(at=at, notify=False)
    rows = data["devices"]
    if not rows:
        return PlainTextResponse(
            "router,site,role,type,model,risk_pct,risk_level,"
            "fault_predicted_next_12h,predicted_fault_type,recommended_action,"
            "alarms_last_24h,prior_faults,downstream_devices,downstream_endpoints\n",
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="fleet_risk_report.csv"'},
        )

    ACTION = {
        "alert_team": "Alert the on-call team (hardware fault, not auto-fixable)",
        "auto_remediate": "Attempt scripted remediation (software fault)",
        "none": "No action needed",
        "unknown": "Model offline - no prediction",
    }
    LEVEL = {
        "critical": "Critical - above alert threshold",
        "elevated": "Elevated - watch closely",
        "watch": "Watch - mild activity",
        "normal": "Normal",
        "unscored": "Not scored",
    }

    out = []
    for r in rows:
        p = r.get("fault_probability")
        predicted = r.get("fault_predicted")
        out.append({
            "router": r["device_id"],
            "site": r.get("site") or "",
            "role": r.get("device_role") or "",
            "type": (r.get("device_type") or "").replace("_", " "),
            "model": r.get("model") or "",
            "risk_pct": "" if p is None else round(float(p) * 100, 2),
            "risk_level": LEVEL.get(r.get("risk_band"), r.get("risk_band") or ""),
            "fault_predicted_next_12h": (
                "" if predicted is None else ("yes" if predicted else "no")),
            # Blank on purpose when nothing is predicted: Stage 2 never ran.
            "predicted_fault_type": r.get("fault_type") or "",
            "recommended_action": ACTION.get(r.get("action"), r.get("action") or ""),
            "alarms_last_24h": r.get("n_events_window"),
            "prior_faults": r.get("n_prior_faults"),
            "downstream_devices": r.get("n_downstream"),
            "downstream_endpoints": r.get("n_downstream_endpoints"),
        })

    df = pd.DataFrame(out)
    header = (
        f"# RouterGuardian fleet risk report\n"
        f"# scored as of {data['analyzed_at']} "
        f"({data['lookback_hours']}h of alarms in, {data['horizon_hours']}h ahead out)\n"
        f"# alert threshold {data['threshold']:.2%} - "
        f"predicted_fault_type is blank unless fault_predicted_next_12h is yes\n"
    )
    return PlainTextResponse(
        header + df.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fleet_risk_report.csv"'},
    )


# ------------------------------------------------------------------ alerts
@app.get("/api/alerts", tags=["alerts"])
def list_alerts(limit: int = Query(100, ge=1, le=500),
                device_id: str | None = None,
                kind: str | None = Query(None, pattern="^(alert|notice)$")):
    """
    The outbox: every alert this app has raised, newest first.

    Alerts survive re-seeding the network on purpose — the record of what the
    on-call team was told is about the app's behaviour, not about which log
    happens to be loaded right now.
    """
    rows = db.list_alerts(limit=limit, device_id=device_id, kind=kind)
    # The bodies are large; the list view does not need them.
    for r in rows:
        r.pop("body_html", None)
        r.pop("context_json", None)
        r["body_preview"] = (r.get("body_text") or "")[:220]
        r.pop("body_text", None)
    return {"alerts": rows, "stats": db.alert_stats(),
            "delivery": notifications.delivery_status(),
            "assistant": assistant.status()}


@app.get("/api/alerts/{alert_id}", tags=["alerts"])
def get_alert(alert_id: int):
    """One alert in full, with its conversation."""
    a = db.get_alert(alert_id)
    if not a:
        raise HTTPException(status_code=404, detail=f"No alert with id {alert_id}.")
    # Order matters: the assistant reads the incident out of context_json, so the
    # greeting has to be built before that field is dropped from the response.
    a["opening_message"] = assistant.opening_message(a)
    a["context"] = assistant.load_context(a)
    a.pop("context_json", None)
    a["messages"] = db.get_messages(alert_id)
    return a


@app.get("/api/alerts/{alert_id}/email", tags=["alerts"], response_class=HTMLResponse)
def alert_email_html(alert_id: int):
    """The alert exactly as it was rendered for the recipient's inbox."""
    a = db.get_alert(alert_id)
    if not a:
        raise HTTPException(status_code=404, detail=f"No alert with id {alert_id}.")
    return HTMLResponse(a.get("body_html") or f"<pre>{a.get('body_text', '')}</pre>")


@app.post("/api/alerts/test", tags=["alerts"])
def test_alert_delivery():
    """
    Send a test message to the on-call addresses.

    Not recorded in the outbox — it is a connectivity check, not an incident.
    The real SMTP error is returned when there is one, because that is the only
    way to tell a wrong password from a wrong port.
    """
    return notifications.send_test()


@app.post("/api/devices/{device_id}/alert", tags=["alerts"])
def send_alert_now(device_id: str, force: bool = True):
    """
    Raise an alert for one router by hand.

    Used for the resend button and for testing the mail path. With force=true it
    ignores the cooldown; the automatic path never does.
    """
    _require_seeded()
    score = _score_device(device_id, explain=True)
    if not score.get("scored"):
        raise HTTPException(status_code=503,
                            detail="The model is not loaded, so nothing can be predicted "
                                   "for this router.")
    if not score.get("fault_predicted"):
        raise HTTPException(
            status_code=409,
            detail=f"No fault is predicted for {device_id} right now "
                   f"({score['fault_probability']:.1%} against a "
                   f"{score['threshold']:.1%} threshold), so there is nothing to alert on.")
    alert = notifications.raise_alert(device_id, score, force=force)
    if not alert:
        raise HTTPException(status_code=429,
                            detail=f"{device_id} alerted recently. "
                                   f"{notifications.cooldown_remaining(device_id)} minutes "
                                   "left on the cooldown.")
    alert.pop("context_json", None)
    return alert


@app.post("/api/alerts/{alert_id}/chat", tags=["alerts"])
def alert_chat(alert_id: int, req: ChatRequest):
    """
    Ask the troubleshooting assistant about one alert.

    The assistant answers from the context stored with that alert, so the
    conversation is grounded in exactly what the recipient was sent.
    """
    a = db.get_alert(alert_id)
    if not a:
        raise HTTPException(status_code=404, detail=f"No alert with id {alert_id}.")
    question = req.message.strip()
    db.add_message(alert_id, "user", question)
    answer, source = assistant.reply(a, question)
    db.add_message(alert_id, "assistant", answer, source=source)
    return {"alert_id": alert_id, "answer": answer, "source": source,
            "messages": db.get_messages(alert_id)}


@app.get("/api/alerts/{alert_id}/chat", tags=["alerts"])
def alert_chat_history(alert_id: int):
    a = db.get_alert(alert_id)
    if not a:
        raise HTTPException(status_code=404, detail=f"No alert with id {alert_id}.")
    return {"alert_id": alert_id, "messages": db.get_messages(alert_id),
            "opening_message": assistant.opening_message(a),
            "assistant": assistant.status()}


# -------------------------------------------------------------- comparison
def _haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    """Great-circle distance between two sites, in kilometres."""
    import math
    if None in (lat1, lon1, lat2, lon2):
        return None
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * R * math.asin(math.sqrt(a)), 1)


def _site_pressure(site: str | None, when: datetime) -> dict:
    """
    How the rest of the site is doing.

    A router scoring badly while every one of its neighbours scores badly too is
    a site problem, not a device problem — and that is a completely different
    work order. Only the routers at that site are scored, so this stays cheap.
    """
    from .topology import STRESSED_SITES
    out = {"site": site, "n_routers": 0, "n_at_risk": 0, "peers_at_risk": [],
           "mean_risk": None, "known_stressed": None}
    if not site:
        return out
    peers = [d for d in db.list_routers() if d.get("site") == site]
    out["n_routers"] = len(peers)
    if site in STRESSED_SITES:
        out["known_stressed"] = STRESSED_SITES[site]["reason"]
    if not model_service.ready:
        return out
    probs = []
    for p in peers:
        try:
            s = _score_device(p["device_id"], at=when)
        except HTTPException:
            continue
        if s.get("fault_probability") is None:
            continue
        probs.append(s["fault_probability"])
        if s.get("fault_predicted"):
            out["n_at_risk"] += 1
            out["peers_at_risk"].append(p["device_id"])
    if probs:
        out["mean_risk"] = round(sum(probs) / len(probs), 4)
    return out


def _profile(device_id: str, when: datetime) -> dict:
    """Everything about one router that might explain its risk, other than its score."""
    dev = db.get_device(device_id) or {}
    alarms = db.get_alarms(device_id)

    env_rate = equip_rate = None
    severity_mix: dict[str, int] = {}
    observed_days = None
    first_seen = dev.get("first_seen")
    if not alarms.empty:
        first, last = alarms["timestamp"].min(), alarms["timestamp"].max()
        observed_days = max((last - first).days, 1)
        per30 = 30.0 / observed_days
        env_rate = round(int((alarms["alarm_type"] == "environmentalAlarm").sum()) * per30, 2)
        equip_rate = round(int((alarms["alarm_type"] == "equipmentAlarm").sum()) * per30, 2)
        severity_mix = {k: int(v) for k, v in alarms["severity"].value_counts().items()}
        first_seen = first.isoformat()

    return {
        "device_id": device_id,
        "site": dev.get("site"),
        "lat": dev.get("lat"),
        "lon": dev.get("lon"),
        "model": dev.get("model"),
        "device_type": dev.get("device_type"),
        "device_role": dev.get("device_role"),
        "parent_id": dev.get("parent_id"),
        "n_downstream": dev.get("n_downstream"),
        "n_downstream_endpoints": dev.get("n_downstream_endpoints"),
        "n_alarms": dev.get("n_alarms"),
        "n_faults": dev.get("n_faults"),
        "n_hw_faults": dev.get("n_hw_faults"),
        "n_sw_faults": dev.get("n_sw_faults"),
        "first_seen": first_seen,
        "in_service_days": observed_days,
        "environmental_per_30d": env_rate,
        "equipment_per_30d": equip_rate,
        "severity_mix": severity_mix,
        "site_pressure": _site_pressure(dev.get("site"), when),
    }


def _comparison_factors(pa: dict, pb: dict, ra: dict, rb: dict) -> list[dict]:
    """
    Plain-language candidate explanations for the risk gap, most useful first.

    The point of comparing two routers is not to display two columns of numbers
    but to answer "why is one riskier than the other". Each factor states what
    was compared, what was found, and whether it plausibly explains the gap.
    """
    out: list[dict] = []
    a, b = pa["device_id"], pb["device_id"]
    ga, gb = ra.get("fault_probability"), rb.get("fault_probability")
    if ga is None or gb is None:
        return out
    hi, lo = (pa, pb) if ga >= gb else (pb, pa)
    hi_p, lo_p = (ga, gb) if ga >= gb else (gb, ga)
    gap = hi_p - lo_p

    # --- geography ------------------------------------------------------
    dist = _haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])
    same_site = pa["site"] and pa["site"] == pb["site"]
    hp, lp = hi["site_pressure"], lo["site_pressure"]
    if same_site:
        out.append({
            "dimension": "Location", "explains": False,
            "headline": f"Both routers are at {pa['site']}.",
            "detail": ("Location cannot explain the difference — they share the same "
                       "building, power and cooling. Look at the device itself."),
        })
    else:
        bits = [f"{a} is at {pa['site'] or 'an unknown site'}, "
                f"{b} is at {pb['site'] or 'an unknown site'}"
                + (f", {dist} km apart." if dist is not None else ".")]

        # A known-bad site is only evidence if this router's own physical alarm
        # rate is actually elevated. Otherwise the site is bad in general but is
        # not what is driving *this* router's score, and saying so would be
        # inventing a cause.
        hi_phys = (hi["environmental_per_30d"] or 0) + (hi["equipment_per_30d"] or 0)
        lo_phys = (lo["environmental_per_30d"] or 0) + (lo["equipment_per_30d"] or 0)
        stressed = bool(hp.get("known_stressed") and not lp.get("known_stressed"))
        peer_evidence = (hp["n_routers"] and lp["n_routers"]
                         and hp["n_at_risk"] > lp["n_at_risk"] and hp["n_at_risk"] > 1)
        explains = bool((stressed and hi_phys > lo_phys) or peer_evidence)

        if hp.get("known_stressed"):
            bits.append(f"{hi['device_id']}'s site is a known problem environment: "
                        f"{hp['known_stressed']}.")
        if hp["n_routers"] and lp["n_routers"]:
            bits.append(f"Right now {hp['n_at_risk']} of {hp['n_routers']} routers at "
                        f"{hp['site']} are flagged, versus "
                        f"{lp['n_at_risk']} of {lp['n_routers']} at {lp['site']}.")
        if stressed and not peer_evidence:
            bits.append("Its neighbours are not currently flagged, so treat the site as "
                        "background pressure rather than an active site-wide incident.")
        if peer_evidence:
            bits.append("Several routers at the same site are flagged together, which "
                        "points at the site rather than at any one device.")

        out.append({
            "dimension": "Location", "explains": explains,
            "headline": (f"{hi['device_id']}'s site is plausibly part of the cause."
                         if explains else
                         "Location does not look like the difference."),
            "detail": " ".join(bits),
        })

    # --- environment ----------------------------------------------------
    ea, eb = pa["environmental_per_30d"], pb["environmental_per_30d"]
    qa, qb = pa["equipment_per_30d"], pb["equipment_per_30d"]
    if None not in (ea, eb, qa, qb):
        h_env = hi["environmental_per_30d"] + hi["equipment_per_30d"]
        l_env = lo["environmental_per_30d"] + lo["equipment_per_30d"]
        ratio = h_env / max(l_env, 0.01)
        explains = ratio >= 1.6 and h_env - l_env >= 1.0
        out.append({
            "dimension": "Physical environment", "explains": explains,
            "headline": (f"{hi['device_id']} logs {ratio:.1f}x more thermal and power "
                         "alarms." if explains else
                         "Environmental alarm rates are comparable."),
            "detail": (
                f"{a}: {ea} environmental + {qa} equipment alarms per 30 days. "
                f"{b}: {eb} environmental + {qb} equipment alarms per 30 days. "
                + ("Thermal and power alarms are a property of the room, not the box, "
                   "so this points at cooling or mains supply at "
                   f"{hi['site']} rather than at the hardware itself."
                   if explains else
                   "Neither router is under noticeably more physical stress than the other.")),
        })

    # --- hardware and age -----------------------------------------------
    same_model = pa["model"] and pa["model"] == pb["model"]
    da, dbv = pa["in_service_days"], pb["in_service_days"]
    age_gap = abs((da or 0) - (dbv or 0))
    explains_age = bool(da and dbv and age_gap > max(da, dbv) * 0.25)
    out.append({
        "dimension": "Hardware and age", "explains": explains_age,
        "headline": ("Same model, same vintage — the hardware is not the difference."
                     if same_model and not explains_age else
                     f"{a} is a {pa['model'] or '?'}, {b} is a {pb['model'] or '?'}."
                     if not same_model else
                     "Same model, but noticeably different time in service."),
        "detail": (f"{a}: {pa['model'] or 'unknown model'}, "
                   f"{da or '?'} days of alarm history. "
                   f"{b}: {pb['model'] or 'unknown model'}, "
                   f"{dbv or '?'} days. "
                   + ("Identical hardware commissioned at effectively the same time, so a "
                      "risk gap has to come from somewhere else — location, load or luck."
                      if same_model and not explains_age else
                      "Different hardware or different service life, either of which can "
                      "shift the baseline alarm rate.")),
    })

    # --- topology -------------------------------------------------------
    shared_parent = pa["parent_id"] and pa["parent_id"] == pb["parent_id"]
    out.append({
        "dimension": "Topology position", "explains": bool(shared_parent),
        "headline": (f"Both hang off {pa['parent_id']} — a shared upstream is a "
                     "common hidden cause." if shared_parent else
                     f"{hi['device_id']} carries "
                     f"{hi['n_downstream'] or 0} downstream devices."),
        "detail": (f"{a}: parent {pa['parent_id'] or 'none (core)'}, "
                   f"{pa['n_downstream'] or 0} devices and "
                   f"{pa['n_downstream_endpoints'] or 0} end devices downstream. "
                   f"{b}: parent {pb['parent_id'] or 'none (core)'}, "
                   f"{pb['n_downstream'] or 0} devices and "
                   f"{pb['n_downstream_endpoints'] or 0} end devices downstream. "
                   + ("They share an upstream router, so a problem there would show on "
                      "both — worth checking before treating either as a device fault."
                      if shared_parent else
                      "Blast radius does not drive the score, but it decides which "
                      "of the two you should fix first.")),
    })

    # --- history ---------------------------------------------------------
    # Direction matters here. History only *explains* the gap when the riskier
    # router is also the one with the worse record. When it runs the other way
    # that is worth saying out loud: it means current alarm activity is
    # outweighing a clean track record, which is the model working as intended.
    fa, fb = pa["n_faults"] or 0, pb["n_faults"] or 0
    hi_f, lo_f = hi["n_faults"] or 0, lo["n_faults"] or 0
    material = abs(fa - fb) >= 3
    aligned = hi_f > lo_f
    explains_hist = material and aligned
    if not material:
        headline = "Similar fault records — history is not the difference."
        note = ("Neither router has a materially worse track record, so the gap comes "
                "from current alarm activity.")
    elif aligned:
        headline = (f"{hi['device_id']} is the riskier router and also the worse "
                    f"offender ({hi_f} faults against {lo_f}).")
        note = ("History and current activity point the same way, so the two cannot be "
                "separated from this view alone — switch on the alarm swap above to "
                "hold activity constant.")
    else:
        headline = (f"History runs against the score: {lo['device_id']} has more faults "
                    f"({lo_f} against {hi_f}) yet scores lower.")
        note = (f"{hi['device_id']}'s risk is being driven by what is happening right "
                "now, not by its record. That is the model weighing live symptoms above "
                "a bad history, which is the behaviour you want.")
    out.append({
        "dimension": "Fault history", "explains": explains_hist,
        "headline": headline,
        "detail": (f"{a}: {fa} fault episodes ({pa['n_hw_faults'] or 0} hardware). "
                   f"{b}: {fb} ({pb['n_hw_faults'] or 0} hardware). " + note),
    })

    # Most useful first, and note the size of the gap being explained.
    out.sort(key=lambda f: not f["explains"])
    for f in out:
        f["gap_explained"] = round(gap, 4)
    return out


@app.get("/api/compare", tags=["predict"])
def compare(a: str = Query(...), b: str = Query(...),
            swap_alarms: bool = Query(False)):
    """
    Compare two real routers side by side.

    With swap_alarms=true both routers are scored on router A's alarm pattern,
    holding current activity constant so the only remaining difference is each
    router's fault history. That isolates exactly what device history
    contributes — the check that a repeat-offender is ranked above a
    never-failed router given identical symptoms.
    """
    _require_seeded()
    when = db.reference_time() + timedelta(seconds=1)
    ra = _score_device(a, at=when, explain=True)
    rb = _score_device(b, at=when, explain=True)

    if swap_alarms:
        alarms_a = db.get_alarms(a, before=when)
        events = events_from_dataframe(alarms_a, when, LOOKBACK_HOURS) if not alarms_a.empty else []
        dev_b = db.get_device(b)
        role_b = dev_b["device_role"] if dev_b["device_role"] in ("core", "edge", "access") else "access"
        if model_service.ready:
            Xb = build_feature_row(events, role_b, rb["history"])
            swapped = model_service.predict(Xb)
            rb = {**rb, **swapped, "n_events_window": len(events), "scored": True,
                  "risk_band": _risk_band(swapped["fault_probability"]),
                  "note": f"Scored on {a}'s alarm pattern with {b}'s own fault history."}

    # Why the two differ, beyond the score itself: location, environment,
    # hardware vintage, topology position and fault record.
    pa, pb = _profile(a, when), _profile(b, when)
    context = {
        "a": pa, "b": pb,
        "distance_km": _haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"]),
        "same_site": bool(pa["site"] and pa["site"] == pb["site"]),
        "same_model": bool(pa["model"] and pa["model"] == pb["model"]),
        "shared_parent": bool(pa["parent_id"] and pa["parent_id"] == pb["parent_id"]),
        "factors": _comparison_factors(pa, pb, ra, rb),
    }

    if ra["fault_probability"] is None or rb["fault_probability"] is None:
        return {"a": ra, "b": rb, "delta": None, "swap_alarms": swap_alarms,
                "verdict": "Model offline — load the model files to compare risk scores.",
                "history_check": None, "role_caveat": None, "context": context}

    delta = round(ra["fault_probability"] - rb["fault_probability"], 4)
    ha = ra["history"]["n_prior_faults"]
    hb = rb["history"]["n_prior_faults"]

    if abs(delta) < 0.02:
        verdict = "Both routers score about the same."
    else:
        hi, lo = (a, b) if delta > 0 else (b, a)
        ratio = max(ra["fault_probability"], rb["fault_probability"]) / max(
            min(ra["fault_probability"], rb["fault_probability"]), 1e-6)
        verdict = f"{hi} scores {ratio:.2f}x higher than {lo}."

    # If the two routers have different roles, the role one-hot features also
    # differ, so the comparison is not purely history-vs-history. Say so.
    role_caveat = None
    if swap_alarms and ra["device_role"] != rb["device_role"]:
        role_caveat = (
            f"These routers have different roles ({ra['device_role']} vs {rb['device_role']}), "
            "so role features differ too. For a strict history-only comparison, pick two "
            "routers with the same role."
        )

    if swap_alarms and ha != hb:
        more, less = (a, b) if ha > hb else (b, a)
        more_p = ra["fault_probability"] if ha > hb else rb["fault_probability"]
        less_p = rb["fault_probability"] if ha > hb else ra["fault_probability"]
        history_check = (
            f"{more} has more prior faults ({max(ha,hb)} vs {min(ha,hb)}) and scores "
            f"{'HIGHER' if more_p > less_p else 'LOWER'} ({more_p:.3f} vs {less_p:.3f}) — "
            + ("history is ranking the repeat offender above the healthy router, as intended."
               if more_p > less_p else
               "the healthy router scores higher here, so history is not dominating this pair.")
        )
    else:
        history_check = None

    return {"a": ra, "b": rb, "delta": delta, "verdict": verdict,
            "swap_alarms": swap_alarms, "history_check": history_check,
            "role_caveat": role_caveat, "context": context}


# ----------------------------------------------------------------- history
@app.get("/api/devices/{device_id}/history", tags=["fleet"])
def device_history(device_id: str, points: int = Query(60, ge=5, le=300)):
    """
    One router's full story: its fault record, its alarm mix, and a replay of
    how its predicted risk moved over time. A rising trend across weeks is a
    router degrading toward failure, which a single point-in-time score cannot
    show.
    """
    _require_seeded()
    dev = db.get_device(device_id)
    if not dev:
        raise HTTPException(status_code=404, detail=f"Unknown router '{device_id}'.")

    alarms = db.get_alarms(device_id)
    if alarms.empty:
        raise HTTPException(status_code=404, detail=f"No alarms recorded for '{device_id}'.")

    role = dev["device_role"] if dev["device_role"] in ("core", "edge", "access") else "access"
    first, last = alarms["timestamp"].min(), alarms["timestamp"].max()
    start = first + timedelta(hours=LOOKBACK_HOURS)
    total_h = max((last - start).total_seconds() / 3600.0, 1.0)
    step = max(total_h / max(points, 1), 1.0)

    timeline, t = [], start
    while t <= last and len(timeline) < points:
        evs = events_from_dataframe(alarms, t, LOOKBACK_HOURS)
        hist = derive_history_from_log(alarms, t)
        p = None
        if model_service.ready:
            X = build_feature_row(evs, role, hist)
            p = round(float(model_service.predict_batch(X)[0]), 4)
        timeline.append({
            "timestamp": t.isoformat(), "fault_probability": p,
            "n_events_window": len(evs), "n_prior_faults": hist["n_prior_faults"],
        })
        t += timedelta(hours=step)

    faults = alarms[(alarms["is_fault_episode"] == 1) & (alarms["raise_or_clear"] == "Raise")]
    fault_events = [
        {"timestamp": r["timestamp"].isoformat(), "alarm_name": r.get("alarm_name"),
         "alarm_type": r.get("alarm_type"), "severity": r.get("severity")}
        for _, r in faults.iterrows()
    ]

    observed_days = max((last - first).days, 1)
    n_faults = len(faults)

    # A router can carry hundreds of alarms and still have zero faults, which
    # looks like a bug unless the difference is spelled out. Alarms are the raw
    # stream; a fault episode is the subset the log marks as an actual failure.
    n_clears = int((alarms["raise_or_clear"] == "Clear").sum())
    n_raises = int((alarms["raise_or_clear"] == "Raise").sum())
    n_routine = n_raises - n_faults
    alarm_breakdown = {
        "n_alarms": int(len(alarms)),
        "n_raise": n_raises,
        "n_clear": n_clears,
        "n_fault_raises": n_faults,
        "n_routine_raises": n_routine,
        "explanation": (
            f"{len(alarms):,} alarms is the raw stream: {n_raises:,} raises and "
            f"{n_clears:,} clears over {observed_days} days. "
            + (
                f"{n_faults} of those raises are marked in the log as a real fault "
                f"episode; the other {n_routine:,} are routine events that never "
                "escalated. Only fault episodes count toward fault history."
                if n_faults
                else f"None of the {n_raises:,} raises is marked in the log as a fault "
                     "episode — this router logged alarms but never actually failed, "
                     "so its fault count is 0. That is the difference between an "
                     "alarm and a fault."
            )
        ),
    }
    rate = n_faults / observed_days * 30.0
    if n_faults == 0:
        assessment, tier = "No fault episodes recorded in the observed period.", "healthy"
    elif rate >= 4:
        assessment, tier = (
            f"Repeat offender — {n_faults} fault episodes (~{rate:.1f}/month). "
            "Consider proactive replacement.", "chronic")
    elif rate >= 1:
        assessment, tier = (
            f"Occasional faults — {n_faults} episodes (~{rate:.1f}/month). Worth monitoring.",
            "moderate")
    else:
        assessment, tier = (
            f"Mostly stable — {n_faults} episode(s) over {observed_days} days.", "healthy")

    # Is risk trending up? Compare the last fifth of the replay with the first.
    trend = "flat"
    if len(timeline) >= 10 and all(p["fault_probability"] is not None for p in timeline):
        k = max(len(timeline) // 5, 2)
        early = sum(p["fault_probability"] for p in timeline[:k]) / k
        recent = sum(p["fault_probability"] for p in timeline[-k:]) / k
        if recent > early * 1.25 and recent - early > 0.03:
            trend = "rising"
        elif early > recent * 1.25 and early - recent > 0.03:
            trend = "falling"

    return {
        "device": dev,
        "n_events_total": int(len(alarms)),
        "first_seen": first.isoformat(),
        "last_seen": last.isoformat(),
        "observed_days": observed_days,
        "n_fault_episodes": n_faults,
        "n_hw_faults": dev["n_hw_faults"],
        "n_sw_faults": dev["n_sw_faults"],
        "health_tier": tier,
        "health_assessment": assessment,
        "risk_trend": trend,
        "alarm_breakdown": alarm_breakdown,
        "threshold": DEPLOY_THRESHOLD,
        "horizon_hours": HORIZON_HOURS,
        "lookback_hours": LOOKBACK_HOURS,
        "current": _score_device(device_id, explain=True),
        "timeline": timeline,
        "fault_events": fault_events,
        "alarm_type_breakdown": alarms["alarm_type"].value_counts().to_dict(),
        "severity_breakdown": alarms["severity"].value_counts().to_dict(),
    }


# ------------------------------------------------------------- frontend
try:
    from pathlib import Path

    FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        def serve_index():
            return FileResponse(str(FRONTEND_DIR / "index.html"))

except Exception as exc:  # noqa: BLE001
    logger.warning("Frontend not mounted: %s", exc)
