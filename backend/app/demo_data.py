"""
Built-in demo network.

Two things matter for the demo to be useful:

1. **Some routers must be predicted to fault right now.** A fault episode placed
   randomly in the past is invisible to the model at the current moment — the
   model looks at the last 24 hours only. So a number of routers are given an
   *active incident*: an accelerating precursor cascade running right up to the
   end of the data, with the fault itself still in the future. Those are the
   routers the model should flag.

2. **The alarm names must exist in the loaded alarm dictionary**, otherwise the
   remediation panel has nothing to show. When a dictionary has been imported,
   alarm names are sampled from it instead of from the small built-in list.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Fallback pools, used only when no alarm dictionary has been loaded.
FALLBACK_HARDWARE = [
    ("0x08130055", "hwEntityInvalid", "equipmentAlarm"),
    ("0x0813001B", "hwPowerInvalid", "equipmentAlarm"),
    ("0x08130085", "hwPtimAlarm", "equipmentAlarm"),
    ("0x08130101", "hwBoardTempOver", "environmentalAlarm"),
    ("0x08130102", "hwFanFailed", "environmentalAlarm"),
    ("0x08130103", "hwOpticalPowerLow", "equipmentAlarm"),
]
FALLBACK_SOFTWARE = [
    ("0x00F10467", "hwMplsLdpSessionDown", "communicationsAlarm"),
    ("0x00F10123", "hwBgpPeerDown", "communicationsAlarm"),
    ("0x00F10456", "hwOspfNbrChange", "communicationsAlarm"),
    ("0x0801214C", "hwRateOfTrafficRising", "qualityOfServiceAlarm"),
    ("0x08012200", "hwCpuUtilizationHigh", "processingErrorAlarm"),
    ("0x08012201", "hwMemUsageHigh", "processingErrorAlarm"),
]

HARDWARE_TYPES = {"equipmentAlarm", "environmentalAlarm"}
BACKGROUND_RATE = {"core": 1.4, "edge": 1.0, "access": 0.6}

# Baseline share of background alarms that are hardware rather than software.
# Raised at sites with a poor physical environment — see topology.STRESSED_SITES.
HARDWARE_SHARE = 0.30


def generate_demo_log(
    n_core: int = 4,
    n_edge: int = 8,
    n_access: int = 22,
    days: int = 120,
    seed: int = 42,
    alarm_pools: dict[str, list[tuple]] | None = None,
    n_active_incidents: int = 7,
) -> pd.DataFrame:
    """
    Produce an alarm log shaped like synthetic_alarm_log.csv.

    alarm_pools: optional {"hardware": [(id,name,type)…], "software": […]}
                 taken from the loaded alarm dictionary, so every generated
                 alarm resolves to a real definition.
    n_active_incidents: how many routers are mid-cascade at the end of the data
                 (these are the ones the model should predict a fault for).
    """
    random.seed(seed)
    np.random.seed(seed)

    hw_pool = (alarm_pools or {}).get("hardware") or FALLBACK_HARDWARE
    sw_pool = (alarm_pools or {}).get("software") or FALLBACK_SOFTWARE

    end = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    # Each router's site is resolved here, from the same helper the topology
    # builder uses, so a router's environment is known before its alarms are
    # generated. Without this the "hot site" effect would land on the wrong
    # routers and the geographic comparison would be showing pure noise.
    from .topology import site_for, site_stress

    fleet: list[tuple[str, str, str, str]] = []
    for i in range(n_core):
        fleet.append((f"CORE-{i + 1:03d}", "core", _tier(0.35), site_for("core", i)["name"]))
    for i in range(n_edge):
        fleet.append((f"EDGE-{i + 1:03d}", "edge", _tier(0.30), site_for("edge", i)["name"]))
    for i in range(n_access):
        fleet.append((f"ACCESS-{i + 1:03d}", "access", _tier(0.22), site_for("access", i)["name"]))

    rows: list[dict] = []

    for device_id, role, tier, site in fleet:
        # A router in a badly cooled or power-unstable site raises more thermal
        # and power alarms than identical hardware elsewhere. This is the whole
        # point of comparing two routers geographically: the cause of the gap is
        # the building, not the box.
        stress = site_stress(site)
        hw_share = min(HARDWARE_SHARE * stress, 0.85)

        # ---- background noise, right up to the end -----------------------
        for d in range(days):
            for _ in range(np.random.poisson(BACKGROUND_RATE[role] * (1 + (stress - 1) * 0.35))):
                pool = hw_pool if random.random() < hw_share else sw_pool
                aid, name, atype = random.choice(pool)
                # Stressed sites do not just log more hardware alarms, they log
                # more severe ones.
                sev = (random.choice(["Minor", "Major", "Major", "Critical"])
                       if stress > 1.5 and pool is hw_pool
                       else random.choice(["Warning", "Minor", "Minor", "Major"]))
                rows.append(_row(device_id, role,
                                 start + timedelta(days=d, hours=random.uniform(0, 24)),
                                 aid, name, atype, sev, "Raise", False))

        # ---- historical fault episodes -----------------------------------
        # Every router gets at least one, so the inspector is never a blank
        # "0 faults over 120 days".
        n_faults = {"healthy": random.randint(1, 2),
                    "moderate": random.randint(3, 5),
                    "chronic": random.randint(6, 10)}[tier]

        for _ in range(n_faults):
            is_hw = random.random() < 0.5
            onset = start + timedelta(days=random.uniform(2, days - 2))
            _inject_episode(rows, device_id, role, onset, is_hw, hw_pool, sw_pool,
                            end, resolved=True)

    # ---- active incidents: cascades still building at "now" --------------
    # These give the model something to actually predict. The fault onset is in
    # the near future, so only the build-up is in the data.
    #
    # Hardware and software are ALTERNATED rather than coin-flipped, so the demo
    # always exercises both branches of the decision layer: at least one router
    # routed to "alert the team" and at least one to "attempt auto-remediation".
    # A coin flip can easily produce seven of the same kind and leave one of the
    # two KPI tiles reading zero.
    candidates = [f for f in fleet if f[2] in ("chronic", "moderate")] or fleet
    random.shuffle(candidates)
    # Bias toward routers with more downstream impact so the map has something
    # worth looking at: core and edge first, then access.
    rank = {"core": 0, "edge": 1, "access": 2}
    candidates.sort(key=lambda f: rank.get(f[1], 3))

    n_incidents = max(0, min(n_active_incidents, len(candidates)))
    for i, (device_id, role, _tier_name, site) in enumerate(candidates[:n_incidents]):
        # Hardware incidents are steered toward the environmentally stressed
        # sites, so the routers the model flags for the on-call team are the ones
        # whose location genuinely explains them.
        is_hw = (i % 2 == 0) or site_stress(site) > 1.5
        pool = hw_pool if is_hw else sw_pool
        # The whole cascade must sit inside the 24h lookback window, and its last
        # event must be recent enough that `hours_since_last_event` stays small —
        # that feature dominates the model, so a cascade that ends 8 hours ago
        # scores far lower than the same cascade ending 20 minutes ago.
        lead = random.uniform(9.0, 20.0)
        n_pre = random.randint(12, 20)
        # Beta(3,1) reversed -> most events land close to the end (accelerating).
        offsets = sorted((1 - np.random.beta(3.0, 1.0, n_pre)) * lead, reverse=True)
        # Pin the tail of the cascade to the last couple of hours.
        offsets = [float(o) for o in offsets]
        offsets[-1] = random.uniform(0.15, 0.6)
        offsets[-2] = random.uniform(0.7, 1.8)
        for k, off in enumerate(offsets):
            aid, name, atype = random.choice(pool)
            frac = k / max(len(offsets) - 1, 1)
            sev = "Critical" if frac > 0.8 else "Major" if frac > 0.4 else "Minor"
            ts = end - timedelta(hours=float(off))
            if ts <= end:
                rows.append(_row(device_id, role, ts, aid, name, atype, sev, "Raise", False))

    df = pd.DataFrame(rows)
    df = df[df["timestamp"] <= end].sort_values("timestamp").reset_index(drop=True)
    return df


def _inject_episode(rows, device_id, role, onset, is_hw, hw_pool, sw_pool, end,
                    resolved=True):
    """A precursor cascade, the fault itself, and usually a clearing event."""
    pool = hw_pool if is_hw else sw_pool
    lead = 36.0 if is_hw else 18.0
    for off in (1 - np.random.beta(3.0, 1.0, random.randint(8, 15))) * lead:
        aid, name, atype = random.choice(pool)
        rows.append(_row(device_id, role, onset - timedelta(hours=float(off)),
                         aid, name, atype,
                         "Major" if off < lead / 2 else "Minor", "Raise", False))

    aid, name, atype = random.choice(pool)
    rows.append(_row(device_id, role, onset, aid, name, atype,
                     "Critical", "Raise", True))

    if resolved and random.random() > 0.1:
        clear_at = onset + timedelta(hours=float(np.random.lognormal(1.1, 0.8)))
        if clear_at < end:
            rows.append(_row(device_id, role, clear_at, aid, name, atype,
                             "Warning", "Clear", False))


def _tier(chronic_p: float) -> str:
    r = random.random()
    if r < chronic_p:
        return "chronic"
    if r < chronic_p + 0.35:
        return "moderate"
    return "healthy"


def _row(device_id, role, ts, aid, name, atype, sev, rc, is_fault) -> dict:
    return {
        "device_id": device_id,
        "device_role": role,
        "timestamp": ts,
        "alarm_id": aid,
        "alarm_name": name,
        "alarm_type": atype,
        "severity": sev,
        "raise_or_clear": rc,
        "is_fault_episode": is_fault,
    }
