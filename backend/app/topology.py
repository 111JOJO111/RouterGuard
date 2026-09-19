"""
Network topology construction.

The alarm log only tells us which routers exist and what role each plays. A real
network also contains the access-layer switches and the end devices hanging off
them, so those are synthesised deterministically from the router list: every
access router gets switches, every switch gets end devices.

Deterministic means the same log always produces the same network — the topology
does not shuffle between page loads.
"""
from __future__ import annotations

import hashlib
from typing import Any

# Device classes, ordered from the core of the network outward.
CORE_ROUTER = "core_router"
EDGE_ROUTER = "edge_router"
ACCESS_ROUTER = "access_router"
SWITCH = "switch"
ENDPOINT = "endpoint"

LAYER_OF = {
    CORE_ROUTER: 0,
    EDGE_ROUTER: 1,
    ACCESS_ROUTER: 2,
    SWITCH: 3,
    ENDPOINT: 4,
}

ROLE_TO_TYPE = {
    "core": CORE_ROUTER,
    "edge": EDGE_ROUTER,
    "access": ACCESS_ROUTER,
}

TYPE_LABEL = {
    CORE_ROUTER: "Core router",
    EDGE_ROUTER: "Edge router",
    ACCESS_ROUTER: "Access router",
    SWITCH: "Switch",
    ENDPOINT: "End device",
}

# Hardware models, purely for presentation.
TYPE_MODELS = {
    CORE_ROUTER: ["NE8000-M14", "NE8000-X8", "NE9000-8"],
    EDGE_ROUTER: ["NE40E-X8", "NE40E-X16", "NE20E-S2F"],
    ACCESS_ROUTER: ["ATN 910C", "ATN 950D", "NE05E-SQ"],
    SWITCH: ["S5732-H24", "S6730-H48", "CE6863-48S6CQ"],
}

ENDPOINT_KINDS = [
    ("server", "Server"),
    ("wifi_ap", "Wi-Fi AP"),
    ("camera", "IP camera"),
    ("voip", "VoIP gateway"),
    ("iot", "IoT gateway"),
]

# Tunisian sites with real coordinates, so the topology can be drawn on a map.
SITES = [
    {"name": "Tunis-Centre",    "lat": 36.8065, "lon": 10.1815},
    {"name": "Ariana-DC1",      "lat": 36.8625, "lon": 10.1956},
    {"name": "Sfax-Sud",        "lat": 34.7406, "lon": 10.7603},
    {"name": "Sousse-Nord",     "lat": 35.8256, "lon": 10.6369},
    {"name": "Bizerte-Port",    "lat": 37.2744, "lon": 9.8739},
    {"name": "Kairouan-Metro",  "lat": 35.6781, "lon": 10.0963},
    {"name": "Gabes-Est",       "lat": 33.8815, "lon": 10.0982},
    {"name": "Monastir-Ville",  "lat": 35.7643, "lon": 10.8113},
    {"name": "Nabeul-Cap",      "lat": 36.4513, "lon": 10.7357},
    {"name": "Gafsa-Ouest",     "lat": 34.4250, "lon": 8.7840},
]


# Sites where the physical environment is poor — inadequate cooling, unstable
# mains. Routers here log more thermal and power alarms than identical hardware
# elsewhere. Used by the demo generator so that "same model, same age, different
# risk" has a real cause sitting behind it, which is exactly what the router
# comparison is meant to surface.
STRESSED_SITES = {
    "Gafsa-Ouest": {"factor": 2.8, "reason": "undersized cooling in a desert climate"},
    "Gabes-Est": {"factor": 2.0, "reason": "unstable mains supply, frequent power events"},
}


def site_for(role: str, index: int) -> dict:
    """
    Which site the Nth router of a given role belongs to.

    Single source of truth: build_topology() uses it to place routers, and the
    demo generator uses it to know a router's site *before* the topology exists,
    so it can vary alarm behaviour by location. If these two ever disagreed, the
    environmental effect would land on the wrong routers.
    """
    return SITES[index % len(SITES)]


def site_stress(site_name: str) -> float:
    """Environmental alarm multiplier for a site. 1.0 means a normal site."""
    return STRESSED_SITES.get(site_name, {}).get("factor", 1.0)


def _seed(*parts: str) -> int:
    """Stable pseudo-random integer from strings, so layouts never shuffle."""
    return int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16)


def _pick(options: list, *key: str):
    return options[_seed(*key) % len(options)]


def build_topology(routers: list[dict]) -> list[dict]:
    """
    Take the routers found in the alarm log and return the full device list:
    the routers themselves, plus synthesised switches and end devices.

    Each router dict must have device_id and device_role. Everything else is
    filled in here.
    """
    core = [r for r in routers if r["device_role"] == "core"]
    edge = [r for r in routers if r["device_role"] == "edge"]
    access = [r for r in routers if r["device_role"] == "access"]

    devices: list[dict] = []

    # --- routers from the log -------------------------------------------
    for i, r in enumerate(core):
        site = site_for("core", i)
        devices.append({**r,
            "device_type": CORE_ROUTER, "parent_id": None,
            "site": site["name"], "lat": site["lat"], "lon": site["lon"],
            "model": _pick(TYPE_MODELS[CORE_ROUTER], r["device_id"]),
            "is_monitored": 1, "endpoint_kind": None})

    for i, r in enumerate(edge):
        parent = core[i % len(core)]["device_id"] if core else None
        site = site_for("edge", i)
        devices.append({**r,
            "device_type": EDGE_ROUTER, "parent_id": parent,
            "site": site["name"], "lat": site["lat"], "lon": site["lon"],
            "model": _pick(TYPE_MODELS[EDGE_ROUTER], r["device_id"]),
            "is_monitored": 1, "endpoint_kind": None})

    upstream = edge or core
    for i, r in enumerate(access):
        parent = upstream[i % len(upstream)]["device_id"] if upstream else None
        site = site_for("access", i)
        devices.append({**r,
            "device_type": ACCESS_ROUTER, "parent_id": parent,
            "site": site["name"], "lat": site["lat"], "lon": site["lon"],
            "model": _pick(TYPE_MODELS[ACCESS_ROUTER], r["device_id"]),
            "is_monitored": 1, "endpoint_kind": None})

    # --- synthesised access layer ---------------------------------------
    # Switches hang off access routers (or edge routers if the log has none).
    switch_parents = access or edge or core
    sw_n = 0
    switches: list[dict] = []
    for p in switch_parents:
        parent_dev = next(d for d in devices if d["device_id"] == p["device_id"])
        n_sw = 1 + (_seed("sw", p["device_id"]) % 2)  # 1-2 switches per router
        for k in range(n_sw):
            sw_n += 1
            sid = f"SW-{sw_n:03d}"
            switches.append({
                "device_id": sid, "device_role": "switch", "device_type": SWITCH,
                "parent_id": p["device_id"], "site": parent_dev["site"],
                "lat": parent_dev["lat"], "lon": parent_dev["lon"],
                "model": _pick(TYPE_MODELS[SWITCH], sid),
                "is_monitored": 0, "endpoint_kind": None,
                "n_alarms": 0, "n_faults": 0, "n_hw_faults": 0, "n_sw_faults": 0,
                "first_seen": None, "last_seen": None,
            })
    devices.extend(switches)

    # End devices hang off switches.
    ep_n = 0
    endpoints: list[dict] = []
    for sw in switches:
        n_ep = 2 + (_seed("ep", sw["device_id"]) % 3)  # 2-4 endpoints per switch
        for k in range(n_ep):
            ep_n += 1
            eid = f"EP-{ep_n:04d}"
            kind, _label = _pick(ENDPOINT_KINDS, eid)
            endpoints.append({
                "device_id": eid, "device_role": "endpoint", "device_type": ENDPOINT,
                "parent_id": sw["device_id"], "site": sw["site"],
                "lat": sw["lat"], "lon": sw["lon"],
                "model": None, "is_monitored": 0, "endpoint_kind": kind,
                "n_alarms": 0, "n_faults": 0, "n_hw_faults": 0, "n_sw_faults": 0,
                "first_seen": None, "last_seen": None,
            })
    devices.extend(endpoints)

    return devices


def blast_radius(devices: list[dict]) -> dict[str, dict[str, int]]:
    """
    For each device, how much of the network sits downstream of it.

    This is what makes a core router's predicted fault more urgent than an
    access router's: it answers "if this fails, how many devices lose service?"
    """
    children: dict[str, list[str]] = {}
    by_id = {d["device_id"]: d for d in devices}
    for d in devices:
        if d.get("parent_id"):
            children.setdefault(d["parent_id"], []).append(d["device_id"])

    memo: dict[str, dict[str, int]] = {}

    def walk(did: str) -> dict[str, int]:
        if did in memo:
            return memo[did]
        total = {"devices": 0, "endpoints": 0, "switches": 0, "routers": 0}
        for c in children.get(did, []):
            cd = by_id[c]
            total["devices"] += 1
            if cd["device_type"] == ENDPOINT:
                total["endpoints"] += 1
            elif cd["device_type"] == SWITCH:
                total["switches"] += 1
            else:
                total["routers"] += 1
            sub = walk(c)
            for k in total:
                total[k] += sub[k]
        memo[did] = total
        return total

    return {d["device_id"]: walk(d["device_id"]) for d in devices}
