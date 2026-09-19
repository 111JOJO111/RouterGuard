"""
SQLite store for the network topology and its alarm history.

The app is database-driven: routers, their parent/child links and their alarm
streams all live here, so nothing has to be typed in by hand. The database is
seeded from an alarm-log CSV (the synthetic_alarm_log.csv produced by the
training pipeline works as-is).
"""
from __future__ import annotations

import csv
import io
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .config import HARDWARE_TYPES
from .topology import build_topology, blast_radius

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "routerguardian.db"
# If a CSV is dropped here it is imported automatically on first startup.
AUTOSEED_CSV = DATA_DIR / "alarm_log.csv"
# Optional: the Huawei alarm dictionary, for descriptions and remediation steps.
# Same idea — the pipeline's own export name is accepted as-is.
AUTOSEED_DICT_NAMES = ("alarm_dictionary.csv", "huawei_alarm_dictionary_final.csv")
AUTOSEED_DICT = DATA_DIR / AUTOSEED_DICT_NAMES[0]

# Bump this whenever the devices/alarms schema changes OR whenever the demo
# generator changes in a way that makes an existing database misleading. A
# database created by an older version is detected on startup and rebuilt, so
# upgrading never leaves a stale table behind (which would fail with "no such
# column: device_type") and never leaves stale *content* behind either.
#
# 4: the demo network moved to Tunisian sites and gained active incidents, so
#    databases built by version 3 had Moroccan site names and no router the
#    model would ever flag. Those had to be rebuilt.
# 5: the demo network gained a site-level environmental effect, so that
#    comparing two routers geographically reveals a real cause rather than
#    noise. Existing demo data predates it and has to be regenerated.
SCHEMA_VERSION = "5"

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id     TEXT PRIMARY KEY,
    device_role   TEXT NOT NULL,
    device_type   TEXT NOT NULL DEFAULT 'access_router',
    parent_id     TEXT,
    site          TEXT,
    lat           REAL,
    lon           REAL,
    model         TEXT,
    endpoint_kind TEXT,
    is_monitored  INTEGER DEFAULT 1,
    n_alarms      INTEGER DEFAULT 0,
    n_faults      INTEGER DEFAULT 0,
    n_hw_faults   INTEGER DEFAULT 0,
    n_sw_faults   INTEGER DEFAULT 0,
    n_downstream       INTEGER DEFAULT 0,
    n_downstream_endpoints INTEGER DEFAULT 0,
    first_seen    TEXT,
    last_seen     TEXT
);

CREATE TABLE IF NOT EXISTS alarms (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id        TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    alarm_id         TEXT,
    alarm_name       TEXT,
    alarm_type       TEXT,
    severity         TEXT,
    raise_or_clear   TEXT,
    is_fault_episode INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alarms_device_ts ON alarms(device_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_alarms_ts        ON alarms(timestamp);

CREATE TABLE IF NOT EXISTS alarm_dictionary (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    alarm_id         TEXT,
    alarm_name       TEXT,
    raise_or_clear   TEXT,
    severity         TEXT,
    alarm_type       TEXT,
    mnemonic_code    TEXT,
    description      TEXT,
    impact_on_system TEXT,
    possible_causes  TEXT,
    procedure        TEXT,
    parameters       TEXT
);

CREATE INDEX IF NOT EXISTS idx_dict_name ON alarm_dictionary(alarm_name);
CREATE INDEX IF NOT EXISTS idx_dict_id   ON alarm_dictionary(alarm_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Outbox. Every alert the app raises is recorded here whether or not it was
-- actually transmitted, so there is always a complete, inspectable record of
-- what the on-call team was told and when. Deliberately NOT dropped when the
-- network is re-seeded: the alert history is about the app's behaviour, not
-- about which log happens to be loaded.
CREATE TABLE IF NOT EXISTS alerts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT NOT NULL,
    device_id         TEXT NOT NULL,
    site              TEXT,
    kind              TEXT NOT NULL DEFAULT 'alert',   -- alert | notice
    fault_type        TEXT,                            -- hardware | software
    action            TEXT,
    fault_probability REAL,
    threshold         REAL,
    horizon_hours     INTEGER,
    subject           TEXT NOT NULL,
    body_text         TEXT NOT NULL,
    body_html         TEXT,
    recipients        TEXT,
    sender            TEXT,
    status            TEXT NOT NULL,                   -- sent | simulated | failed
    error             TEXT,
    scored_at         TEXT,
    context_json      TEXT
);

CREATE INDEX IF NOT EXISTS idx_alerts_device  ON alerts(device_id, created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);

-- The troubleshooting conversation attached to one alert.
CREATE TABLE IF NOT EXISTS alert_messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id   INTEGER NOT NULL,
    role       TEXT NOT NULL,        -- user | assistant
    content    TEXT NOT NULL,
    source     TEXT,                 -- llm | grounded
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_msgs_alert ON alert_messages(alert_id, id);
"""

@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _discard_database_file() -> bool:
    """
    Delete the database file and any leftover journal, so the next connection
    starts from an empty one.

    Returns True if something was actually removed. Everything in here is
    regenerated on startup — the topology from the alarm-log CSV or the demo
    generator, the dictionary from backend/data — so there is nothing to lose.
    """
    removed = False
    for path in (DB_PATH, Path(str(DB_PATH) + "-journal"), Path(str(DB_PATH) + "-wal"),
                 Path(str(DB_PATH) + "-shm")):
        try:
            if path.exists():
                path.unlink()
                removed = True
        except OSError as exc:  # pragma: no cover - depends on the filesystem
            logger.warning("Could not remove %s: %s", path.name, exc)
    return removed


def init_db() -> None:
    """
    Create the schema, recovering automatically from a damaged database file.

    A database can be left unusable by an interrupted write — a stale
    `-journal` beside the file, a half-finished transaction, a file copied off a
    network share mid-write. SQLite then fails every query with "database disk
    image is malformed" or "disk I/O error", which previously took the whole app
    down with it even though every byte in that file is reproducible. Rather
    than surface an error the user cannot act on, the file is discarded and
    rebuilt: on the next startup autoseed_if_available() repopulates it.
    """
    try:
        with get_conn() as conn:
            conn.executescript(SCHEMA)
            # Touch the tables as well. executescript can succeed on a file whose
            # pages are corrupt further in, so this is what actually proves the
            # database is readable.
            conn.execute("SELECT COUNT(*) FROM devices").fetchone()
            conn.execute("SELECT COUNT(*) FROM alarm_dictionary").fetchone()
            conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
        return
    except sqlite3.Error as exc:
        logger.warning("Database at %s is unusable (%s) — rebuilding it from scratch.",
                       DB_PATH, exc)

    if not _discard_database_file():
        # Could not remove it (read-only directory, file held open elsewhere).
        # Re-raise so the failure is visible rather than silently swallowed.
        with get_conn() as conn:
            conn.executescript(SCHEMA)
        return

    with get_conn() as conn:
        conn.executescript(SCHEMA)
    logger.info("A fresh database was created at %s.", DB_PATH)


def is_seeded() -> bool:
    """True only if the database has data AND matches the current schema."""
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM devices").fetchone()
            if not row or row["c"] == 0:
                return False
            # Verify the schema actually has the columns this version needs.
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(devices)")}
            required = {"device_type", "lat", "lon", "is_monitored", "n_downstream"}
            if not required.issubset(cols):
                return False
            ver = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            return bool(ver and ver["value"] == SCHEMA_VERSION)
    except sqlite3.Error:
        return False


def _reset_schema() -> None:
    """Drop everything and recreate. Used when an old database is detected."""
    try:
        with get_conn() as conn:
            for t in ("alarms", "devices", "meta"):
                conn.execute(f"DROP TABLE IF EXISTS {t}")
    except sqlite3.Error:
        pass
    init_db()


def seed_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Wipe and rebuild the database from an alarm-log dataframe."""
    required = {"device_id", "timestamp", "alarm_type", "severity", "raise_or_clear"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise ValueError("No rows with a valid timestamp.")

    if "is_fault_episode" in df.columns:
        df["is_fault_episode"] = (
            df["is_fault_episode"].astype(str).str.lower().isin(["true", "1"]).astype(int)
        )
    else:
        # Fall back to Critical raises as the fault-episode proxy.
        df["is_fault_episode"] = (
            (df["severity"] == "Critical") & (df["raise_or_clear"] == "Raise")
        ).astype(int)

    for col in ("alarm_id", "alarm_name", "device_role"):
        if col not in df.columns:
            df[col] = None
    df["device_role"] = df["device_role"].fillna("access")

    devices = []
    for device_id, ddf in df.groupby("device_id"):
        faults = ddf[(ddf["is_fault_episode"] == 1) & (ddf["raise_or_clear"] == "Raise")]
        n_hw = int(faults["alarm_type"].isin(HARDWARE_TYPES).sum()) if len(faults) else 0
        devices.append(
            {
                "device_id": str(device_id),
                "device_role": str(ddf["device_role"].iloc[0]),
                "parent_id": None,
                "site": None,
                "model": None,
                "n_alarms": int(len(ddf)),
                "n_faults": int(len(faults)),
                "n_hw_faults": n_hw,
                "n_sw_faults": int(len(faults)) - n_hw,
                "first_seen": ddf["timestamp"].min().isoformat(),
                "last_seen": ddf["timestamp"].max().isoformat(),
            }
        )
    # Build the full network: the routers from the log plus the switches and
    # end devices that hang off them.
    devices = build_topology(devices)
    radius = blast_radius(devices)
    for d in devices:
        r = radius.get(d["device_id"], {})
        d["n_downstream"] = r.get("devices", 0)
        d["n_downstream_endpoints"] = r.get("endpoints", 0)

    _reset_schema()
    with get_conn() as conn:
        conn.execute("DELETE FROM alarms")
        conn.execute("DELETE FROM devices")
        conn.execute("DELETE FROM meta")

        conn.executemany(
            """INSERT INTO devices (device_id, device_role, device_type, parent_id,
                   site, lat, lon, model, endpoint_kind, is_monitored,
                   n_alarms, n_faults, n_hw_faults, n_sw_faults,
                   n_downstream, n_downstream_endpoints, first_seen, last_seen)
               VALUES (:device_id, :device_role, :device_type, :parent_id,
                   :site, :lat, :lon, :model, :endpoint_kind, :is_monitored,
                   :n_alarms, :n_faults, :n_hw_faults, :n_sw_faults,
                   :n_downstream, :n_downstream_endpoints, :first_seen, :last_seen)""",
            devices,
        )

        conn.executemany(
            """INSERT INTO alarms (device_id, timestamp, alarm_id, alarm_name,
                   alarm_type, severity, raise_or_clear, is_fault_episode)
               VALUES (?,?,?,?,?,?,?,?)""",
            [
                (
                    str(r.device_id), r.timestamp.isoformat(),
                    None if pd.isna(r.alarm_id) else str(r.alarm_id),
                    None if pd.isna(r.alarm_name) else str(r.alarm_name),
                    str(r.alarm_type), str(r.severity), str(r.raise_or_clear),
                    int(r.is_fault_episode),
                )
                for r in df.itertuples(index=False)
            ],
        )

        # The log is historical, so "now" for the app is the newest timestamp
        # in the data, not the wall clock.
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
            [
                ("reference_time", df["timestamp"].max().isoformat()),
                ("seeded_at", datetime.utcnow().isoformat()),
                ("schema_version", SCHEMA_VERSION),
                ("n_devices", str(len(devices))),
                ("n_routers", str(sum(1 for d in devices if d["is_monitored"]))),
                ("n_alarms", str(len(df))),
            ],
        )

    return {
        "n_devices": len(devices),
        "n_routers": sum(1 for d in devices if d["is_monitored"]),
        "n_alarms": int(len(df)),
        "reference_time": df["timestamp"].max().isoformat(),
        "date_range": [df["timestamp"].min().isoformat(), df["timestamp"].max().isoformat()],
    }


def seed_from_csv_bytes(content: bytes) -> dict[str, Any]:
    return seed_from_dataframe(pd.read_csv(io.BytesIO(content)))


def autoseed_if_available() -> dict[str, Any] | None:
    """
    On startup, make sure the app has a network to show.

    Priority: a CSV the user dropped at backend/data/alarm_log.csv, otherwise a
    generated demo network — so the UI is never empty on first run.
    """
    init_db()
    if is_seeded():
        return None
    # Either empty, or built by an older version — rebuild from scratch.
    _reset_schema()
    # Load the alarm dictionary FIRST. The demo generator draws its alarm names
    # from the dictionary when one is present, which is what stops the
    # remediation panel showing "no definition" for half its alarms.
    if dictionary_size() == 0:
        for name in AUTOSEED_DICT_NAMES:
            path = DATA_DIR / name
            if not path.exists():
                continue
            try:
                seed_dictionary(pd.read_csv(path))
                break
            except Exception:  # noqa: BLE001
                continue
    if AUTOSEED_CSV.exists():
        info = seed_from_dataframe(pd.read_csv(AUTOSEED_CSV))
        info["source"] = "csv"
        return info
    return seed_demo()


def alarm_pools_from_dictionary() -> dict[str, list[tuple]] | None:
    """
    Real (alarm_id, alarm_name, alarm_type) triples from the loaded dictionary,
    split into hardware and software. Used so the demo network's alarms resolve
    to real definitions in the remediation panel instead of showing
    "no definition".
    """
    import hashlib

    from .config import HARDWARE_TYPES as HW

    # Only alarms that carry a real fix procedure are eligible. An alarm with a
    # row in the dictionary but an empty `procedure` still shows up as "nothing
    # to do" in the remediation panel, which is the complaint this avoids.
    query = (
        "SELECT DISTINCT alarm_id, alarm_name, alarm_type FROM alarm_dictionary "
        "WHERE raise_or_clear='Raise' AND alarm_name IS NOT NULL "
        "AND alarm_type IS NOT NULL {extra} ORDER BY alarm_name"
    )
    strict = (
        "AND procedure IS NOT NULL AND TRIM(procedure) != '' "
        "AND description IS NOT NULL AND TRIM(description) != ''"
    )
    try:
        with get_conn() as conn:
            rows = conn.execute(query.format(extra=strict)).fetchall()
            if len(rows) < 12:
                # Dictionary export without procedures — fall back to any named
                # alarm rather than giving up on real names entirely.
                rows = conn.execute(query.format(extra="")).fetchall()
    except sqlite3.Error:
        return None

    hw, sw = [], []
    for r in rows:
        t = (r["alarm_id"], r["alarm_name"], r["alarm_type"])
        (hw if r["alarm_type"] in HW else sw).append(t)
    if not hw or not sw:
        return None

    # Keep the pools small so alarms repeat and look like a real fleet. The sort
    # key must be stable across restarts, so md5 is used rather than hash():
    # Python randomises string hashing per process, which would silently reshuffle
    # the demo network's alarm names on every boot.
    def key(t: tuple) -> str:
        return hashlib.md5(str(t[1]).encode()).hexdigest()

    hw.sort(key=key)
    sw.sort(key=key)
    return {"hardware": hw[:14], "software": sw[:14]}


def seed_demo(**kwargs: Any) -> dict[str, Any]:
    """Generate and load the built-in demo network."""
    from .demo_data import generate_demo_log
    kwargs.setdefault("alarm_pools", alarm_pools_from_dictionary())
    info = seed_from_dataframe(generate_demo_log(**kwargs))
    info["used_dictionary_alarms"] = kwargs.get("alarm_pools") is not None
    info["source"] = "demo"
    with get_conn() as conn:
        conn.execute("INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                     ("source", "demo"))
    return info


# --------------------------------------------------------------- queries
DICT_COLS = ["alarm_id", "alarm_name", "raise_or_clear", "severity", "alarm_type",
             "mnemonic_code", "description", "impact_on_system", "possible_causes",
             "procedure", "parameters"]


def seed_dictionary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Load the Huawei alarm dictionary (huawei_alarm_dictionary_final.csv).

    Only alarm_name is strictly required; every other column is optional and
    filled with NULL if the export does not have it.
    """
    if "alarm_name" not in df.columns:
        raise ValueError("Dictionary CSV must contain an 'alarm_name' column. "
                         f"Found: {sorted(df.columns)[:12]}")
    df = df.copy()
    for c in DICT_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[DICT_COLS].where(pd.notna(df[DICT_COLS]), None)

    init_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM alarm_dictionary")
        conn.executemany(
            f"INSERT INTO alarm_dictionary ({','.join(DICT_COLS)}) "
            f"VALUES ({','.join('?' * len(DICT_COLS))})",
            [tuple(r) for r in df.itertuples(index=False, name=None)],
        )
        conn.execute("INSERT OR REPLACE INTO meta (key,value) VALUES (?,?)",
                     ("n_dictionary", str(len(df))))
    n_proc = int(df["procedure"].notna().sum())
    return {"n_entries": int(len(df)), "n_with_procedure": n_proc,
            "n_unique_alarms": int(df["alarm_name"].nunique())}


def seed_dictionary_from_bytes(content: bytes) -> dict[str, Any]:
    return seed_dictionary(pd.read_csv(io.BytesIO(content)))


def dictionary_size() -> int:
    try:
        with get_conn() as conn:
            r = conn.execute("SELECT COUNT(*) AS c FROM alarm_dictionary").fetchone()
            return int(r["c"]) if r else 0
    except sqlite3.Error:
        return 0


def lookup_alarm(alarm_name: str | None = None, alarm_id: str | None = None,
                 raise_or_clear: str | None = None) -> dict | None:
    """
    Find an alarm definition. Prefers an exact (alarm_id, raise_or_clear) match,
    which is the dictionary's own dedup key, then falls back to alarm_name.
    """
    try:
        with get_conn() as conn:
            if alarm_id:
                q = "SELECT * FROM alarm_dictionary WHERE alarm_id=?"
                params: list[Any] = [alarm_id]
                if raise_or_clear:
                    q += " AND raise_or_clear=?"
                    params.append(raise_or_clear)
                row = conn.execute(q + " LIMIT 1", params).fetchone()
                if row:
                    return dict(row)
            if alarm_name:
                row = conn.execute(
                    "SELECT * FROM alarm_dictionary WHERE alarm_name=? LIMIT 1",
                    (alarm_name,)).fetchone()
                if row:
                    return dict(row)
                # Clear events are often named <RaiseName>Resume / <RaiseName>Clear.
                base = alarm_name
                for suf in ("Resume", "Clear", "Recover"):
                    if base.endswith(suf):
                        base = base[: -len(suf)]
                        break
                if base != alarm_name:
                    row = conn.execute(
                        "SELECT * FROM alarm_dictionary WHERE alarm_name=? LIMIT 1",
                        (base,)).fetchone()
                    if row:
                        return dict(row)
    except sqlite3.Error:
        return None
    return None


def search_dictionary(q: str, limit: int = 40) -> list[dict]:
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM alarm_dictionary WHERE alarm_name LIKE ? OR alarm_id LIKE ? "
                "OR description LIKE ? LIMIT ?",
                (f"%{q}%", f"%{q}%", f"%{q}%", limit)).fetchall()
            return [dict(r) for r in rows]
    except sqlite3.Error:
        return []


def get_meta(key: str, default: str | None = None) -> str | None:
    """Read a meta value. Safe to call before the schema exists."""
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default
    except sqlite3.Error:
        return default


def reference_time() -> datetime:
    """The moment the app treats as 'now' — the newest alarm in the data."""
    val = get_meta("reference_time")
    return datetime.fromisoformat(val) if val else datetime.utcnow()


def fault_onsets(limit: int = 400) -> list[dict]:
    """
    Every fault episode in the log, newest first.

    Used to offer "score the fleet as it looked just before this real failure"
    moments. An uploaded historical log usually ends on a quiet night, so
    scoring only at the newest timestamp makes every router look healthy even
    though the log is full of failures.
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT device_id, timestamp, alarm_name, alarm_type, severity "
                "FROM alarms WHERE is_fault_episode = 1 AND raise_or_clear = 'Raise' "
                "ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
    except sqlite3.Error:
        return []
    return [dict(r) for r in rows]


# ----------------------------------------------------------------- outbox
ALERT_COLS = ("created_at", "device_id", "site", "kind", "fault_type", "action",
              "fault_probability", "threshold", "horizon_hours", "subject",
              "body_text", "body_html", "recipients", "sender", "status",
              "error", "scored_at", "context_json")


def record_alert(alert: dict[str, Any]) -> int:
    """Write one alert to the outbox and return its id."""
    row = {c: alert.get(c) for c in ALERT_COLS}
    row["created_at"] = row["created_at"] or datetime.utcnow().isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT INTO alerts ({','.join(ALERT_COLS)}) "
            f"VALUES ({','.join(':' + c for c in ALERT_COLS)})", row)
        return int(cur.lastrowid)


def update_alert(alert_id: int, fields: dict[str, Any]) -> None:
    """
    Fill in an alert after it has been created.

    The row is inserted before the message is rendered so the alert has a real
    id, which the email's "discuss this with the assistant" link needs to point
    at. The body and delivery outcome are written back here.
    """
    allowed = {k: v for k, v in fields.items() if k in ALERT_COLS}
    if not allowed:
        return
    sets = ", ".join(f"{k}=:{k}" for k in allowed)
    allowed["id"] = int(alert_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE alerts SET {sets} WHERE id=:id", allowed)


def last_alert_at(device_id: str, kind: str | None = None) -> datetime | None:
    """When this router last paged the team. Drives the cooldown."""
    q = "SELECT created_at FROM alerts WHERE device_id=?"
    params: list[Any] = [device_id]
    if kind:
        q += " AND kind=?"
        params.append(kind)
    try:
        with get_conn() as conn:
            row = conn.execute(q + " ORDER BY id DESC LIMIT 1", params).fetchone()
    except sqlite3.Error:
        return None
    if not row or not row["created_at"]:
        return None
    try:
        return datetime.fromisoformat(row["created_at"])
    except ValueError:
        return None


def list_alerts(limit: int = 100, device_id: str | None = None,
                kind: str | None = None) -> list[dict]:
    q = "SELECT * FROM alerts"
    where, params = [], []
    if device_id:
        where.append("device_id=?")
        params.append(device_id)
    if kind:
        where.append("kind=?")
        params.append(kind)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        with get_conn() as conn:
            rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    except sqlite3.Error:
        return []
    # Message counts, so the list can show which alerts have been discussed.
    if rows:
        ids = [r["id"] for r in rows]
        marks = ",".join("?" * len(ids))
        try:
            with get_conn() as conn:
                counts = {r["alert_id"]: r["n"] for r in conn.execute(
                    f"SELECT alert_id, COUNT(*) AS n FROM alert_messages "
                    f"WHERE alert_id IN ({marks}) GROUP BY alert_id", ids)}
        except sqlite3.Error:
            counts = {}
        for r in rows:
            r["n_messages"] = counts.get(r["id"], 0)
    return rows


def get_alert(alert_id: int) -> dict | None:
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
            return dict(row) if row else None
    except sqlite3.Error:
        return None


def alert_stats() -> dict[str, Any]:
    try:
        with get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM alerts").fetchone()["c"]
            by_status = {r["status"]: r["c"] for r in conn.execute(
                "SELECT status, COUNT(*) AS c FROM alerts GROUP BY status")}
            by_kind = {r["kind"]: r["c"] for r in conn.execute(
                "SELECT kind, COUNT(*) AS c FROM alerts GROUP BY kind")}
            last = conn.execute(
                "SELECT created_at FROM alerts ORDER BY id DESC LIMIT 1").fetchone()
    except sqlite3.Error:
        return {"n_alerts": 0, "by_status": {}, "by_kind": {}, "last_alert_at": None}
    return {"n_alerts": int(total), "by_status": by_status, "by_kind": by_kind,
            "last_alert_at": last["created_at"] if last else None}


def add_message(alert_id: int, role: str, content: str,
                source: str | None = None) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO alert_messages (alert_id, role, content, source, created_at) "
            "VALUES (?,?,?,?,?)",
            (int(alert_id), role, content, source, datetime.utcnow().isoformat()))
        return int(cur.lastrowid)


def get_messages(alert_id: int) -> list[dict]:
    try:
        with get_conn() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM alert_messages WHERE alert_id=? ORDER BY id", (alert_id,))]
    except sqlite3.Error:
        return []


LAYER_ORDER = ("core_router", "edge_router", "access_router", "switch", "endpoint")


def list_devices(monitored_only: bool = False) -> list[dict]:
    q = "SELECT * FROM devices"
    if monitored_only:
        q += " WHERE is_monitored = 1"
    try:
        with get_conn() as conn:
            rows = [dict(r) for r in conn.execute(q).fetchall()]
    except sqlite3.Error:
        return []
    rows.sort(key=lambda d: (LAYER_ORDER.index(d["device_type"])
                             if d["device_type"] in LAYER_ORDER else 9, d["device_id"]))
    return rows


def list_routers() -> list[dict]:
    """Only the devices that actually carry alarms and can be scored."""
    return list_devices(monitored_only=True)


def get_device(device_id: str) -> dict | None:
    try:
        with get_conn() as conn:
            row = conn.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()
            return dict(row) if row else None
    except sqlite3.Error:
        return None


def get_alarms(device_id: str, before: datetime | None = None,
               hours: float | None = None, limit: int | None = None) -> pd.DataFrame:
    q = "SELECT * FROM alarms WHERE device_id=?"
    params: list[Any] = [device_id]
    if before is not None:
        q += " AND timestamp < ?"
        params.append(before.isoformat())
        if hours is not None:
            from datetime import timedelta
            q += " AND timestamp >= ?"
            params.append((before - timedelta(hours=hours)).isoformat())
    q += " ORDER BY timestamp"
    if limit:
        q += f" LIMIT {int(limit)}"
    try:
        with get_conn() as conn:
            df = pd.read_sql_query(q, conn, params=params)
    except Exception:
        return pd.DataFrame(columns=["device_id", "timestamp", "alarm_type",
                                     "severity", "raise_or_clear", "is_fault_episode"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def get_alarms_by_device(before: datetime | None = None) -> dict[str, pd.DataFrame]:
    """
    Every device's alarms in one query, keyed by device_id.

    Fleet scoring needs the full history of all routers at once. Asking for them
    one device at a time costs a round trip and a DataFrame construction each,
    which dominates fleet scoring on anything larger than a demo network.
    """
    q = "SELECT * FROM alarms"
    params: list[Any] = []
    if before is not None:
        q += " WHERE timestamp < ?"
        params.append(before.isoformat())
    q += " ORDER BY device_id, timestamp"
    try:
        with get_conn() as conn:
            df = pd.read_sql_query(q, conn, params=params)
    except Exception:  # noqa: BLE001
        return {}
    if df.empty:
        return {}
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return {str(k): g for k, g in df.groupby("device_id", sort=False)}


def alarms_to_csv(device_id: str) -> str:
    df = get_alarms(device_id)
    buf = io.StringIO()
    cols = ["device_id", "timestamp", "alarm_id", "alarm_name", "alarm_type",
            "severity", "raise_or_clear", "is_fault_episode"]
    if df.empty:
        csv.writer(buf).writerow(cols)
        return buf.getvalue()
    df[[c for c in cols if c in df.columns]].to_csv(buf, index=False)
    return buf.getvalue()


def full_log_to_csv() -> str:
    try:
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT * FROM alarms ORDER BY timestamp", conn)
    except Exception:
        df = pd.DataFrame()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def topology() -> dict[str, Any]:
    devs = list_devices()
    links = [
        {"source": d["parent_id"], "target": d["device_id"]}
        for d in devs if d.get("parent_id")
    ]
    sites: dict[str, dict] = {}
    for d in devs:
        if not d.get("site"):
            continue
        s = sites.setdefault(d["site"], {
            "name": d["site"], "lat": d.get("lat"), "lon": d.get("lon"),
            "n_devices": 0, "n_routers": 0, "device_ids": []})
        s["n_devices"] += 1
        s["device_ids"].append(d["device_id"])
        if d.get("is_monitored"):
            s["n_routers"] += 1
    counts: dict[str, int] = {}
    for d in devs:
        counts[d["device_type"]] = counts.get(d["device_type"], 0) + 1
    return {"devices": devs, "links": links,
            "sites": sorted(sites.values(), key=lambda x: x["name"]),
            "counts": counts}
