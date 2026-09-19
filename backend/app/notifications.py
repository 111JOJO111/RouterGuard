"""
On-call alerting.

When Stage 2 routes a predicted fault to "alert the team", that has to become an
actual message to actual people, otherwise the prediction dies inside the app.
This module turns a score into an email and delivers it.

Two design decisions worth knowing:

1. **Every alert is recorded, whether or not it is transmitted.** The outbox in
   the database is the source of truth. With SMTP configured the same message
   also leaves the building; without it the alert is stored with status
   "simulated". The app is therefore fully demonstrable on a laptop with no mail
   server, and an operator can always audit what the team was told.

2. **The context that goes into the email is stored alongside it.** The
   troubleshooting assistant later answers questions from that stored context
   rather than recomputing it, so the conversation is grounded in exactly the
   facts the recipient was sent — not in a fresh score taken minutes later that
   may have moved.
"""
from __future__ import annotations

import json
import logging
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Any

from . import config
from . import database as db
from .config import HARDWARE_TYPES, HORIZON_HOURS, LOOKBACK_HOURS

logger = logging.getLogger(__name__)

# How a log-odds contribution is described in prose. Mirrors the UI so the email
# and the screen never disagree about how strong a driver was.
_STRENGTH = ((1.0, "very strong"), (0.5, "strong"), (0.2, "moderate"), (0.0, "slight"))


def _strength(v: float) -> str:
    a = abs(v)
    for cut, label in _STRENGTH:
        if a >= cut:
            return label
    return "slight"


def _pct(p: float | None) -> str:
    return "—" if p is None else f"{p * 100:.1f}%"


# --------------------------------------------------------------- context
def build_incident_context(device_id: str, score: dict) -> dict[str, Any]:
    """
    Everything a human needs to act on this prediction, in one dict.

    Assembled once, emailed, stored, and later re-read by the assistant. Keeping
    it in a single structure is what stops the email, the UI and the chatbot
    drifting apart.
    """
    dev = db.get_device(device_id) or {}
    when = score.get("scored_at")
    try:
        when_dt = datetime.fromisoformat(when) if when else db.reference_time()
    except (TypeError, ValueError):
        when_dt = db.reference_time()

    window = db.get_alarms(device_id, before=when_dt, hours=LOOKBACK_HOURS)
    recent: list[dict] = []
    if not window.empty:
        for _, r in window.sort_values("timestamp", ascending=False).head(30).iterrows():
            recent.append({
                "timestamp": r["timestamp"].isoformat(),
                "alarm_name": r.get("alarm_name"),
                "alarm_id": r.get("alarm_id"),
                "alarm_type": r.get("alarm_type"),
                "severity": r.get("severity"),
                "raise_or_clear": r.get("raise_or_clear"),
                "category": "hardware" if r.get("alarm_type") in HARDWARE_TYPES else "software",
            })

    # Documented fixes for the distinct alarms seen in the window, worst first.
    SEV = {"Critical": 0, "Major": 1, "Minor": 2, "Warning": 3}
    seen: dict[str, dict] = {}
    for a in recent:
        key = a["alarm_name"] or a["alarm_id"] or "?"
        e = seen.setdefault(key, {**a, "count": 0})
        e["count"] += 1
        if SEV.get(a["severity"], 4) < SEV.get(e["severity"], 4):
            e["severity"] = a["severity"]
    remedies = []
    for a in sorted(seen.values(), key=lambda x: (SEV.get(x["severity"], 4), -x["count"])):
        d = db.lookup_alarm(alarm_name=a["alarm_name"], alarm_id=a["alarm_id"],
                            raise_or_clear="Raise")
        remedies.append({
            "alarm_name": a["alarm_name"], "alarm_id": a["alarm_id"],
            "alarm_type": a["alarm_type"], "severity": a["severity"],
            "count": a["count"], "category": a["category"],
            "description": (d or {}).get("description"),
            "impact_on_system": (d or {}).get("impact_on_system"),
            "possible_causes": (d or {}).get("possible_causes"),
            "procedure": (d or {}).get("procedure"),
        })

    drivers = []
    for c in (score.get("explanation") or {}).get("contributions", [])[:6]:
        drivers.append({
            "label": c["label"], "value": c["value"],
            "contribution": c["contribution"],
            "direction": "toward a fault" if c["contribution"] > 0 else "away from a fault",
            "strength": _strength(c["contribution"]),
        })

    hist = score.get("history") or {}
    return {
        "device_id": device_id,
        "site": dev.get("site"),
        "model": dev.get("model"),
        "device_role": dev.get("device_role"),
        "device_type": dev.get("device_type"),
        "scored_at": when_dt.isoformat(),
        "fault_probability": score.get("fault_probability"),
        "threshold": score.get("threshold"),
        "fault_type": score.get("fault_type"),
        "action": score.get("action"),
        "action_detail": score.get("action_detail"),
        "type_confidence": score.get("type_confidence"),
        "horizon_hours": HORIZON_HOURS,
        "lookback_hours": LOOKBACK_HOURS,
        "n_events_window": score.get("n_events_window"),
        "n_downstream": dev.get("n_downstream"),
        "n_downstream_endpoints": dev.get("n_downstream_endpoints"),
        "n_prior_faults": hist.get("n_prior_faults"),
        "n_prior_hw_faults": hist.get("n_prior_hw_faults"),
        "n_prior_sw_faults": hist.get("n_prior_sw_faults"),
        "has_pending_fault": hist.get("has_pending_fault"),
        "drivers": drivers,
        "recent_alarms": recent,
        "remediation": remedies,
        "dictionary_loaded": db.dictionary_size() > 0,
    }


# ---------------------------------------------------------------- render
def _subject(ctx: dict) -> str:
    kind = "ACTION REQUIRED" if ctx["action"] == "alert_team" else "FYI"
    what = (ctx.get("fault_type") or "fault").capitalize()
    where = f" ({ctx['site']})" if ctx.get("site") else ""
    return (f"[RouterGuardian] {kind}: {what} fault predicted on "
            f"{ctx['device_id']}{where} — {_pct(ctx['fault_probability'])} risk")


def _render_text(ctx: dict, link: str = "") -> str:
    L: list[str] = []
    add = L.append

    hardware = ctx.get("fault_type") == "hardware"
    add(f"{'HARDWARE' if hardware else 'SOFTWARE'} FAULT PREDICTED — {ctx['device_id']}")
    add("=" * 68)
    add("")
    if hardware:
        add("This fault cannot be fixed by a script. It needs someone on site or")
        add("hands on the device. You are being paged because of that.")
    else:
        add("This is a software fault. Scripted remediation is being attempted")
        add("automatically. No action is needed unless it fails — this message is")
        add("for your awareness.")
    add("")
    add("PREDICTION")
    add("-" * 68)
    add(f"  Router            {ctx['device_id']}")
    add(f"  Site              {ctx.get('site') or 'unknown'}")
    add(f"  Model / role      {ctx.get('model') or '?'} · {ctx.get('device_role') or '?'}")
    add(f"  Risk              {_pct(ctx['fault_probability'])} "
        f"(alert threshold {_pct(ctx.get('threshold'))})")
    add(f"  Window            last {ctx['lookback_hours']}h of alarms, "
        f"predicting {ctx['horizon_hours']}h ahead")
    add(f"  Scored at         {str(ctx['scored_at']).replace('T', ' ')[:19]}")
    if ctx.get("type_confidence") is not None:
        add(f"  Type confidence   {_pct(ctx['type_confidence'])} that this is "
            f"{ctx.get('fault_type')}")
    add("")
    add("IMPACT IF IT FAILS")
    add("-" * 68)
    nd, ne = ctx.get("n_downstream") or 0, ctx.get("n_downstream_endpoints") or 0
    if nd:
        add(f"  {nd} devices sit downstream of this router, including {ne} end devices")
        add("  (servers, access points, cameras, gateways). They lose service with it.")
    else:
        add("  No devices recorded downstream of this router.")
    add(f"  Prior fault episodes on this router: {ctx.get('n_prior_faults', 0)} "
        f"({ctx.get('n_prior_hw_faults', 0)} hardware, {ctx.get('n_prior_sw_faults', 0)} software)")
    add("")
    add("WHY THE MODEL FLAGGED IT")
    add("-" * 68)
    if ctx["drivers"]:
        for d in ctx["drivers"]:
            arrow = "+" if d["contribution"] > 0 else "-"
            add(f"  [{arrow}] {d['label']}: {d['value']:g}  "
                f"({d['strength']}, {d['direction']})")
        add("")
        add("  These are the model's own TreeSHAP contributions, ordered by how far")
        add("  each moved this router's score away from the fleet average.")
    else:
        add("  No per-feature explanation was available for this score.")
    add("")
    add(f"ALARMS IN THE LAST {ctx['lookback_hours']} HOURS ({ctx.get('n_events_window', 0)} total)")
    add("-" * 68)
    for a in ctx["recent_alarms"][:12]:
        add(f"  {str(a['timestamp']).replace('T', ' ')[:16]}  "
            f"{(a['severity'] or ''):<8} {(a['alarm_name'] or '?'):<32} "
            f"{a['raise_or_clear']}")
    if len(ctx["recent_alarms"]) > 12:
        add(f"  … and {len(ctx['recent_alarms']) - 12} more")
    add("")
    add("DOCUMENTED FIX PROCEDURES")
    add("-" * 68)
    withproc = [r for r in ctx["remediation"] if r.get("procedure")][:3]
    if not withproc:
        add("  No documented procedure matched these alarms.")
        if not ctx.get("dictionary_loaded"):
            add("  (The Huawei alarm dictionary is not loaded in RouterGuardian.)")
    for r in withproc:
        add("")
        add(f"  >> {r['alarm_name']}  [{r['severity']}, {r['category']}, "
            f"seen {r['count']}x]")
        if r.get("description"):
            add("     What it means:")
            add(_wrap(r["description"], 60, 7))
        if r.get("possible_causes"):
            add("     Likely causes:")
            add(_wrap(r["possible_causes"], 60, 7))
        add("     Procedure:")
        add(_wrap(r["procedure"], 60, 7))
    add("")
    add("=" * 68)
    add("TALK THIS THROUGH WITH THE ASSISTANT")
    add("")
    add("  " + (link or "(link unavailable)"))
    add("")
    add("  Opens this incident's troubleshooting conversation. The assistant")
    add("  already has everything above — the alarms, the score, the drivers")
    add("  and the procedures — so you can ask what to do first, why this")
    add("  router was flagged, or whether it can wait.")
    add("=" * 68)
    add("Raised automatically by RouterGuardian.")
    return "\n".join(L)


def _wrap(text: str, width: int, indent: int) -> str:
    """Wrap a block of documentation for a plain-text email."""
    import textwrap
    body = " ".join(str(text).split())
    # break_long_words=False: Huawei descriptions embed very long parameter
    # tokens like EntityPhysicalIndex=[EntityPhysicalIndex], and splitting those
    # across lines makes them unreadable.
    return "\n".join(textwrap.wrap(body, width=width,
                                   initial_indent=" " * indent,
                                   subsequent_indent=" " * indent,
                                   break_long_words=False,
                                   break_on_hyphens=False)) or ""


def _esc(s: Any) -> str:
    return (str(s if s is not None else "")
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _render_html(ctx: dict, link: str = "") -> str:
    hardware = ctx.get("fault_type") == "hardware"
    accent = "#7c3aed" if hardware else "#0891b2"
    band = "#be123c" if hardware else "#0e7490"
    rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:#64748b;white-space:nowrap'>{_esc(k)}</td>"
        f"<td style='padding:4px 0;color:#0f172a;font-weight:600'>{_esc(v)}</td></tr>"
        for k, v in [
            ("Router", ctx["device_id"]),
            ("Site", ctx.get("site") or "unknown"),
            ("Model", ctx.get("model") or "?"),
            ("Risk", f"{_pct(ctx['fault_probability'])} (threshold {_pct(ctx.get('threshold'))})"),
            ("Horizon", f"last {ctx['lookback_hours']}h in, {ctx['horizon_hours']}h ahead"),
            ("Scored at", str(ctx["scored_at"]).replace("T", " ")[:19]),
        ])

    def _driver_li(d: dict) -> str:
        up = d["contribution"] > 0
        return (f"<li style='margin:3px 0'>"
                f"<span style='color:{'#be123c' if up else '#0f766e'}'>"
                f"{'&#9650;' if up else '&#9660;'}</span> "
                f"<b>{_esc(d['label'])}</b>: {_esc(format(d['value'], 'g'))} "
                f"<span style='color:#64748b'>({_esc(d['strength'])}, "
                f"{_esc(d['direction'])})</span></li>")

    drivers = ("".join(_driver_li(d) for d in ctx["drivers"])
               or "<li style='color:#64748b'>No explanation available.</li>")

    alarms = "".join(
        f"<tr>"
        f"<td style='padding:3px 10px 3px 0;color:#64748b;white-space:nowrap;font-family:monospace;font-size:12px'>"
        f"{_esc(str(a['timestamp']).replace('T', ' ')[:16])}</td>"
        f"<td style='padding:3px 10px 3px 0'><span style='background:"
        f"{'#fecdd3' if a['severity'] == 'Critical' else '#e9d5ff' if a['severity'] == 'Major' else '#e0f2fe'};"
        f"padding:1px 6px;border-radius:4px;font-size:11px'>{_esc(a['severity'])}</span></td>"
        f"<td style='padding:3px 10px 3px 0;font-family:monospace;font-size:12px'>{_esc(a['alarm_name'])}</td>"
        f"<td style='padding:3px 0;color:#64748b;font-size:12px'>{_esc(a['raise_or_clear'])}</td>"
        f"</tr>"
        for a in ctx["recent_alarms"][:12])

    fixes = ""
    for r in [x for x in ctx["remediation"] if x.get("procedure")][:3]:
        fixes += (
            f"<div style='margin:14px 0;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden'>"
            f"<div style='background:#f8fafc;padding:8px 12px;border-bottom:1px solid #e2e8f0'>"
            f"<b style='font-family:monospace'>{_esc(r['alarm_name'])}</b> "
            f"<span style='color:#64748b;font-size:12px'>· {_esc(r['severity'])} · "
            f"{_esc(r['category'])} · seen {r['count']}x</span></div>"
            f"<div style='padding:10px 12px;font-size:13px;line-height:1.6'>"
            + (f"<p style='margin:0 0 8px'><b>What it means.</b> {_esc(r['description'])}</p>"
               if r.get("description") else "")
            + (f"<p style='margin:0 0 8px'><b>Likely causes.</b> {_esc(r['possible_causes'])}</p>"
               if r.get("possible_causes") else "")
            + f"<p style='margin:0 0 4px'><b>Procedure</b></p>"
            f"<pre style='margin:0;white-space:pre-wrap;font-family:inherit;"
            f"background:#f1f5f9;padding:10px;border-radius:6px'>{_esc(r['procedure'])}</pre>"
            f"</div></div>")
    if not fixes:
        fixes = ("<p style='color:#64748b'>No documented procedure matched these alarms."
                 + ("" if ctx.get("dictionary_loaded")
                    else " The Huawei alarm dictionary is not loaded in RouterGuardian.")
                 + "</p>")

    nd, ne = ctx.get("n_downstream") or 0, ctx.get("n_downstream_endpoints") or 0
    impact = (f"{nd} devices sit downstream of this router, including {ne} end devices. "
              "They lose service with it." if nd
              else "No devices are recorded downstream of this router.")

    lead = ("This fault cannot be fixed by a script. It needs someone on site or hands on "
            "the device — that is why you are being paged."
            if hardware else
            "This is a software fault. Scripted remediation is being attempted automatically; "
            "this message is for your awareness in case it fails.")

    cta = (f"""
    <div style="margin:22px 0 6px;padding:16px;border-radius:10px;
      background:#f8fafc;border:1px solid #e2e8f0;text-align:center">
      <div style="font-size:13px;color:#0f172a;margin-bottom:10px">
        Not sure where to start? Talk it through with the assistant — it already has
        the alarms, the score, the drivers and the procedures above.</div>
      <a href="{_esc(link)}" style="display:inline-block;background:{band};color:#fff;
        text-decoration:none;padding:10px 22px;border-radius:8px;font-weight:600;
        font-size:14px">Open the troubleshooting assistant</a>
      <div style="font-size:11px;color:#94a3b8;margin-top:10px;word-break:break-all">
        {_esc(link)}</div>
    </div>""" if link else "")

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f1f5f9;
 font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#0f172a">
<div style="max-width:680px;margin:0 auto;background:#ffffff">
  <div style="background:{band};color:#fff;padding:18px 24px">
    <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;opacity:.85">
      RouterGuardian · {'action required' if hardware else 'notification'}</div>
    <div style="font-size:20px;font-weight:700;margin-top:4px">
      {'Hardware' if hardware else 'Software'} fault predicted on {_esc(ctx['device_id'])}</div>
    <div style="font-size:13px;opacity:.9;margin-top:2px">
      {_pct(ctx['fault_probability'])} probability within {ctx['horizon_hours']} hours
      · {_esc(ctx.get('site') or 'unknown site')}</div>
  </div>
  <div style="padding:20px 24px">
    <p style="margin:0 0 16px;font-size:14px;line-height:1.6">{lead}</p>

    <h3 style="margin:20px 0 8px;font-size:13px;text-transform:uppercase;
      letter-spacing:.08em;color:{accent}">Prediction</h3>
    <table style="border-collapse:collapse;font-size:13px">{rows}</table>

    <h3 style="margin:22px 0 8px;font-size:13px;text-transform:uppercase;
      letter-spacing:.08em;color:{accent}">Impact if it fails</h3>
    <p style="margin:0;font-size:13px;line-height:1.6">{impact}<br>
      Prior fault episodes on this router: <b>{ctx.get('n_prior_faults', 0)}</b>
      ({ctx.get('n_prior_hw_faults', 0)} hardware, {ctx.get('n_prior_sw_faults', 0)} software).</p>

    <h3 style="margin:22px 0 8px;font-size:13px;text-transform:uppercase;
      letter-spacing:.08em;color:{accent}">Why the model flagged it</h3>
    <ul style="margin:0;padding-left:18px;font-size:13px;line-height:1.6">{drivers}</ul>

    <h3 style="margin:22px 0 8px;font-size:13px;text-transform:uppercase;
      letter-spacing:.08em;color:{accent}">Alarms in the last {ctx['lookback_hours']} hours</h3>
    <table style="border-collapse:collapse;width:100%">{alarms}</table>

    <h3 style="margin:22px 0 8px;font-size:13px;text-transform:uppercase;
      letter-spacing:.08em;color:{accent}">Documented fix procedures</h3>
    {fixes}
    {cta}

    <p style="margin:20px 0 0;padding-top:14px;border-top:1px solid #e2e8f0;
      font-size:12px;color:#64748b;line-height:1.6">
      Raised automatically by RouterGuardian.</p>
  </div>
</div></body></html>"""


# --------------------------------------------------------------- delivery
def _deliver(subject: str, text: str, html: str,
             recipients: list[str]) -> tuple[str, str | None]:
    """
    Try to actually send. Returns (status, error).

    status is "sent" when SMTP accepted it, "simulated" when no mail server is
    configured, and "failed" when a server was configured but rejected it. A
    failure never raises: an alert that could not be transmitted still has to be
    recorded, and the app must not fall over because a mail server is down.
    """
    if not config.smtp_configured():
        return "simulated", None

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(("RouterGuardian", config.ALERT_FROM))
    msg["To"] = ", ".join(recipients)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain="routerguardian.local")
    msg["X-RouterGuardian-Alert"] = "1"
    msg.set_content(text)
    msg.add_alternative(html, subtype="html")

    try:
        if config.SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT,
                                      timeout=config.SMTP_TIMEOUT)
        else:
            server = smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT,
                                  timeout=config.SMTP_TIMEOUT)
        with server:
            server.ehlo()
            if config.SMTP_USE_TLS and config.SMTP_PORT != 465:
                server.starttls()
                server.ehlo()
            if config.SMTP_USER:
                server.login(config.SMTP_USER, config.SMTP_PASSWORD)
            server.send_message(msg)
        return "sent", None
    except Exception as exc:  # noqa: BLE001 - any SMTP failure must be recorded
        logger.warning("Alert for %s could not be sent: %s", recipients, exc)
        return "failed", f"{type(exc).__name__}: {exc}"


# ------------------------------------------------------------------ api
def cooldown_remaining(device_id: str) -> int:
    """Minutes left before this router may page the team again. 0 means now."""
    last = db.last_alert_at(device_id)
    if not last:
        return 0
    elapsed = (datetime.utcnow() - last).total_seconds() / 60.0
    return max(0, int(round(config.ALERT_COOLDOWN_MINUTES - elapsed)))


def raise_alert(device_id: str, score: dict, force: bool = False) -> dict | None:
    """
    Send and record an alert for one scored router.

    Returns the stored alert, or None when nothing was raised — either because
    no fault is predicted or because this router alerted recently. Fleet scoring
    happens on every page load, so without the cooldown a single flagged router
    would page the team dozens of times an hour.
    """
    if not score.get("fault_predicted"):
        return None

    kind = "alert" if score.get("action") == "alert_team" else "notice"
    if kind == "notice" and not config.NOTIFY_SOFTWARE_FAULTS:
        return None

    if not force:
        remaining = cooldown_remaining(device_id)
        if remaining > 0:
            return None

    ctx = build_incident_context(device_id, score)
    subject = _subject(ctx)
    recipients = list(config.ALERT_TO)

    # The row is written first, unsent, purely to reserve an id. The email
    # carries a link straight to this incident's troubleshooting conversation,
    # and that link needs the id — so the message cannot be rendered until the
    # record exists. The body and the delivery outcome are written back below.
    alert_id = db.record_alert({
        "created_at": datetime.utcnow().isoformat(),
        "device_id": device_id,
        "site": ctx.get("site"),
        "kind": kind,
        "fault_type": ctx.get("fault_type"),
        "action": ctx.get("action"),
        "fault_probability": ctx.get("fault_probability"),
        "threshold": ctx.get("threshold"),
        "horizon_hours": ctx.get("horizon_hours"),
        "subject": subject,
        "body_text": "",
        "body_html": "",
        "recipients": ", ".join(recipients),
        "sender": config.ALERT_FROM,
        "status": "pending",
        "error": None,
        "scored_at": ctx.get("scored_at"),
        "context_json": json.dumps(ctx, default=str),
    })

    link = config.alert_link(alert_id)
    text = _render_text(ctx, link)
    html = _render_html(ctx, link)
    status, error = _deliver(subject, text, html, recipients)

    db.update_alert(alert_id, {"body_text": text, "body_html": html,
                               "status": status, "error": error})
    logger.info("Alert %s raised for %s (%s)", alert_id, device_id, status)
    return db.get_alert(alert_id)


def send_test() -> dict[str, Any]:
    """
    Prove the mail path works without waiting for a router to fail.

    Returns the real SMTP error when there is one — authentication failures are
    the normal case while setting this up, and hiding the reason makes it
    guesswork.
    """
    when = datetime.utcnow().isoformat(timespec="seconds")
    recipients = list(config.ALERT_TO)
    subject = "[RouterGuardian] Test message — alerting is configured correctly"
    text = (
        "RouterGuardian test message\n"
        "===========================\n\n"
        "If you are reading this, alert delivery works. Real alerts will arrive\n"
        "at this address whenever a router is predicted to fail, and each one\n"
        "carries the alarms behind the prediction, the reasons the model flagged\n"
        "it, and Huawei's documented fix procedure — plus a link to talk it\n"
        "through with the troubleshooting assistant.\n\n"
        f"Sent at {when} UTC\n"
        f"From:    {config.ALERT_FROM}\n"
        f"To:      {', '.join(recipients)}\n"
        f"Server:  {config.SMTP_HOST or '(none configured)'}:{config.SMTP_PORT}\n"
        f"App:     {config.PUBLIC_BASE_URL}\n"
    )
    html = f"""<!DOCTYPE html><html><body style="margin:0;background:#f1f5f9;
 font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">
<div style="max-width:600px;margin:0 auto;background:#fff">
 <div style="background:#0e7490;color:#fff;padding:18px 24px">
  <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;opacity:.85">
   RouterGuardian</div>
  <div style="font-size:20px;font-weight:700;margin-top:4px">Alerting is configured</div>
 </div>
 <div style="padding:20px 24px;font-size:14px;line-height:1.6;color:#0f172a">
  <p style="margin:0 0 14px">If you are reading this, alert delivery works.</p>
  <p style="margin:0 0 14px">Real alerts arrive here whenever a router is predicted to
   fail. Each one carries the alarms behind the prediction, the reasons the model flagged
   it, Huawei's documented fix procedure, and a link to talk it through with the
   troubleshooting assistant.</p>
  <table style="border-collapse:collapse;font-size:12px;color:#64748b">
   <tr><td style="padding:3px 12px 3px 0">Sent</td>
       <td style="color:#0f172a">{_esc(when)} UTC</td></tr>
   <tr><td style="padding:3px 12px 3px 0">From</td>
       <td style="color:#0f172a">{_esc(config.ALERT_FROM)}</td></tr>
   <tr><td style="padding:3px 12px 3px 0">To</td>
       <td style="color:#0f172a">{_esc(', '.join(recipients))}</td></tr>
   <tr><td style="padding:3px 12px 3px 0">Server</td>
       <td style="color:#0f172a">{_esc(config.SMTP_HOST or '(none)')}:{config.SMTP_PORT}</td></tr>
  </table>
  <p style="margin:16px 0 0"><a href="{_esc(config.PUBLIC_BASE_URL)}"
    style="color:#0e7490">Open RouterGuardian</a></p>
 </div></div></body></html>"""

    status, error = _deliver(subject, text, html, recipients)
    return {
        "status": status,
        "error": error,
        "recipients": recipients,
        "sender": config.ALERT_FROM,
        "smtp_host": config.SMTP_HOST or None,
        "smtp_port": config.SMTP_PORT if config.SMTP_HOST else None,
        "detail": {
            "sent": f"Delivered to {', '.join(recipients)}. Check the inbox.",
            "simulated": config.smtp_status()[1],
            "failed": f"The mail server rejected the message: {error}",
        }.get(status, ""),
    }


def delivery_status() -> dict[str, Any]:
    """What the app can currently do with an alert, for the UI to explain."""
    ready, why = config.smtp_status()
    return {
        "smtp_configured": ready,
        "smtp_host": config.SMTP_HOST or None,
        "smtp_port": config.SMTP_PORT if config.SMTP_HOST else None,
        "smtp_user": config.SMTP_USER or None,
        "sender": config.ALERT_FROM,
        "recipients": list(config.ALERT_TO),
        "cooldown_minutes": config.ALERT_COOLDOWN_MINUTES,
        "notify_software_faults": config.NOTIFY_SOFTWARE_FAULTS,
        "public_url": config.PUBLIC_BASE_URL,
        "mode": "live" if ready else "simulated",
        "blocked_reason": None if ready else why,
        "detail": (
            f"Alerts are emailed through {config.SMTP_HOST}:{config.SMTP_PORT} "
            f"as {config.ALERT_FROM}, and recorded in the outbox."
            if ready else
            why + " Until then alerts are still written in full to the outbox below — "
                   "they are simply not transmitted."
        ),
    }
