"""
Troubleshooting assistant.

When the on-call engineer receives an alert, the useful next step is a
conversation: "what do I actually do first", "why this router and not its
neighbour", "can I wait until morning". This module answers those questions
against the incident the alert was raised for.

The important property is **grounding**. Every answer is built from the context
stored with the alert — that router's alarms, its prediction, the features that
drove it, and Huawei's own documented procedures. Nothing is invented. Two
engines share that same context:

* With an API key present, an LLM phrases the answer. The context is injected as
  system material and the model is told to refuse anything it cannot support
  from it, so it cannot quietly invent a repair procedure.
* With no key, the same facts are answered by intent matching. Less fluent,
  identical facts, no network, no cost.

A wrong repair procedure is worse than no answer at all, which is why the
offline path exists rather than the feature simply being unavailable.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

from . import config
from . import database as db

logger = logging.getLogger(__name__)

MAX_HISTORY = 12          # conversation turns sent to the LLM
MAX_PROCEDURES = 4        # procedures included in the grounding block

# The most recent reason a language-model call failed. Kept so the UI can say
# "the key was rejected" instead of silently answering offline and leaving the
# user to wonder why the assistant never came online.
_LAST_ERROR: dict[str, str | None] = {"message": None}


def _record_failure(message: str) -> None:
    _LAST_ERROR["message"] = message
    logger.warning("Assistant fell back to offline answers: %s", message)


# --------------------------------------------------------------- context
def load_context(alert: dict) -> dict[str, Any]:
    """The incident context stored when the alert was raised."""
    raw = alert.get("context_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _pct(p: Any) -> str:
    try:
        return f"{float(p) * 100:.1f}%"
    except (TypeError, ValueError):
        return "unknown"


def context_block(ctx: dict) -> str:
    """
    The incident, flattened into text the model can read.

    This is the only source of facts the assistant is allowed to use.
    """
    L: list[str] = []
    add = L.append
    add("INCIDENT UNDER DISCUSSION")
    add(f"- Router: {ctx.get('device_id')} ({ctx.get('model') or 'unknown model'}, "
        f"{ctx.get('device_role') or 'unknown role'})")
    add(f"- Site: {ctx.get('site') or 'unknown'}")
    add(f"- Predicted fault type: {ctx.get('fault_type') or 'unknown'} "
        f"-> action: {ctx.get('action')}")
    add(f"- Probability: {_pct(ctx.get('fault_probability'))} against an alert "
        f"threshold of {_pct(ctx.get('threshold'))}")
    add(f"- Prediction window: the last {ctx.get('lookback_hours')}h of alarms, "
        f"predicting {ctx.get('horizon_hours')}h ahead")
    add(f"- Scored at: {ctx.get('scored_at')}")
    add(f"- Alarms in that window: {ctx.get('n_events_window')}")
    add(f"- Downstream impact: {ctx.get('n_downstream') or 0} devices, "
        f"{ctx.get('n_downstream_endpoints') or 0} end devices")
    add(f"- Prior fault episodes on this router: {ctx.get('n_prior_faults', 0)} "
        f"({ctx.get('n_prior_hw_faults', 0)} hardware, "
        f"{ctx.get('n_prior_sw_faults', 0)} software)")
    if ctx.get("has_pending_fault"):
        add("- There is an unresolved fault already open on this router.")

    add("")
    add("WHY THE MODEL FLAGGED IT (TreeSHAP contributions, log-odds)")
    for d in ctx.get("drivers", []):
        add(f"- {d['label']} = {d['value']:g} -> {d['contribution']:+.3f} "
            f"({d['strength']}, {d['direction']})")

    add("")
    add("ALARMS IN THE WINDOW (newest first)")
    for a in ctx.get("recent_alarms", [])[:15]:
        add(f"- {str(a.get('timestamp'))[:16].replace('T', ' ')} "
            f"{a.get('severity')} {a.get('alarm_name')} "
            f"({a.get('alarm_type')}, {a.get('category')}, {a.get('raise_or_clear')})")

    add("")
    add("DOCUMENTED HUAWEI PROCEDURES FOR THESE ALARMS")
    withproc = [r for r in ctx.get("remediation", []) if r.get("procedure")]
    if not withproc:
        add("- None of these alarms matched a documented procedure.")
    for r in withproc[:MAX_PROCEDURES]:
        add("")
        add(f"### {r['alarm_name']} ({r.get('severity')}, {r.get('category')}, "
            f"seen {r.get('count')}x)")
        if r.get("description"):
            add(f"Description: {r['description']}")
        if r.get("impact_on_system"):
            add(f"Impact: {r['impact_on_system']}")
        if r.get("possible_causes"):
            add(f"Possible causes: {r['possible_causes']}")
        add(f"Procedure: {r['procedure']}")
    return "\n".join(L)


SYSTEM_PROMPT = """You are the RouterGuardian troubleshooting assistant, talking \
to a network on-call engineer who has just been paged about a predicted router \
fault.

Ground every answer in the incident context provided below. That context is the \
only source of truth you have about this network.

Rules:
- Never invent a repair procedure, command, alarm name, or measurement. If the \
context does not contain it, say so plainly and suggest what the engineer could \
check on the device instead.
- When you give repair steps, take them from the documented Huawei procedure in \
the context and say which alarm they came from.
- Be concrete and ordered. An engineer at 3am wants "do this first, then this", \
not an essay.
- Distinguish clearly between what the model predicted (a probability, not a \
certainty) and what the log actually records (alarms that really fired).
- If asked whether it can wait, reason from the downstream impact and the \
severity of the alarms in the context, and be honest that the prediction is \
probabilistic.
- Keep answers under about 200 words unless asked to go deeper. Plain text, \
short paragraphs or numbered steps. No markdown headers.

{context}"""


# ------------------------------------------------------------------- llm
def _ask_llm(ctx: dict, history: list[dict], question: str) -> str | None:
    """Answer via the Anthropic API. Returns None if unavailable or it fails."""
    if not config.llm_configured():
        return None

    messages = []
    for m in history[-MAX_HISTORY:]:
        role = "assistant" if m["role"] == "assistant" else "user"
        messages.append({"role": role, "content": m["content"]})
    messages.append({"role": "user", "content": question})
    # The API rejects a leading assistant turn.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    payload = json.dumps({
        "model": config.LLM_MODEL,
        "max_tokens": config.LLM_MAX_TOKENS,
        "system": SYSTEM_PROMPT.format(context=context_block(ctx)),
        "messages": messages,
    }).encode()

    req = urllib.request.Request(
        config.LLM_BASE_URL, data=payload, method="POST",
        headers={"content-type": "application/json",
                 "x-api-key": config.LLM_API_KEY,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # Read the body: the API explains *why* it refused, and "invalid x-api-key"
        # versus "credit balance too low" need completely different fixes.
        detail = ""
        try:
            body = json.loads(exc.read().decode())
            detail = body.get("error", {}).get("message", "")
        except Exception:  # noqa: BLE001
            pass
        _record_failure(f"HTTP {exc.code}: {detail or exc.reason}")
        return None
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        _record_failure(f"{type(exc).__name__}: {exc}")
        return None

    _LAST_ERROR["message"] = None

    parts = [b.get("text", "") for b in data.get("content", [])
             if b.get("type") == "text"]
    text = "\n".join(p for p in parts if p).strip()
    return text or None


# -------------------------------------------------------------- grounded
def _fmt_procedure(r: dict) -> str:
    out = [f"{r['alarm_name']} ({r.get('severity')}, seen {r.get('count')}x)"]
    if r.get("description"):
        out.append(f"What it means: {' '.join(str(r['description']).split())}")
    if r.get("possible_causes"):
        out.append(f"Likely causes: {' '.join(str(r['possible_causes']).split())}")
    out.append("Documented procedure:")
    out.append(" ".join(str(r["procedure"]).split()))
    return "\n".join(out)


def _suggestions(ctx: dict) -> str:
    return ("You can ask me: what do I do first · why was this flagged · "
            "what alarms fired · how urgent is it · has this router failed before · "
            "can this be scripted")


def _grounded_answer(ctx: dict, question: str) -> str:
    """
    Answer from the incident context without an LLM.

    Intent matching over the same material the LLM would receive. Deliberately
    conservative: when it cannot tell what is being asked it says so and offers
    the questions it can answer, rather than guessing.
    """
    q = question.lower().strip()
    dev = ctx.get("device_id", "this router")
    hardware = ctx.get("fault_type") == "hardware"
    withproc = [r for r in ctx.get("remediation", []) if r.get("procedure")]

    def has(*words: str) -> bool:
        return any(re.search(rf"\b{w}", q) for w in words)

    # --- what do I do -----------------------------------------------------
    if has("what do i", "what should", "how do i fix", "fix", "repair", "procedure",
           "steps", "step", "resolve", "remediat", "solution", "solve"):
        if not withproc:
            return (f"I do not have a documented procedure for the alarms on {dev}. "
                    f"The alarms in the window were: "
                    f"{', '.join(sorted({a['alarm_name'] for a in ctx.get('recent_alarms', []) if a.get('alarm_name')})) or 'none recorded'}. "
                    "Without a matching dictionary entry I will not guess at repair "
                    "steps. Check the device's current alarm list and the physical "
                    "state of the flagged component first.")
        head = (f"Start with the most severe alarm on {dev}. "
                if len(withproc) > 1 else "")
        body = "\n\n".join(_fmt_procedure(r) for r in withproc[:2])
        tail = ("\n\nBecause this is a hardware fault, none of this can be scripted — "
                "it needs someone with access to the device."
                if hardware else
                "\n\nThis is a software fault, so these steps are scriptable. "
                "RouterGuardian attempts them automatically; you are seeing this in "
                "case the attempt fails.")
        return head + body + tail

    # --- why was it flagged ----------------------------------------------
    if has("why", "reason", "cause", "driver", "explain", "how did you", "what made"):
        drivers = ctx.get("drivers", [])
        if not drivers:
            return (f"{dev} scored {_pct(ctx.get('fault_probability'))} against a "
                    f"{_pct(ctx.get('threshold'))} threshold, but no per-feature "
                    "explanation was stored with this alert.")
        up = [d for d in drivers if d["contribution"] > 0][:3]
        down = [d for d in drivers if d["contribution"] < 0][:2]
        lines = [f"{dev} scored {_pct(ctx.get('fault_probability'))}, above the "
                 f"{_pct(ctx.get('threshold'))} alert threshold.", ""]
        if up:
            lines.append("Pushing it toward a fault:")
            lines += [f"  - {d['label']} = {d['value']:g} ({d['strength']})" for d in up]
        if down:
            lines.append("Pushing it away:")
            lines += [f"  - {d['label']} = {d['value']:g} ({d['strength']})" for d in down]
        lines.append("")
        lines.append("These are the model's own contributions for this router, "
                     "measured against the fleet average. It is a probability over "
                     f"the next {ctx.get('horizon_hours')} hours, not a certainty.")
        return "\n".join(lines)

    # --- urgency ----------------------------------------------------------
    if has("urgent", "wait", "priority", "how bad", "serious", "impact", "downstream",
           "morning", "tomorrow", "now", "blast"):
        nd = ctx.get("n_downstream") or 0
        ne = ctx.get("n_downstream_endpoints") or 0
        crit = [a for a in ctx.get("recent_alarms", []) if a.get("severity") == "Critical"]
        lines = [f"{dev} scored {_pct(ctx.get('fault_probability'))} for a fault within "
                 f"{ctx.get('horizon_hours')} hours."]
        lines.append(f"If it fails, {nd} downstream devices lose service, including "
                     f"{ne} end devices."
                     if nd else "No devices are recorded downstream of it.")
        if crit:
            lines.append(f"There {'is' if len(crit) == 1 else 'are'} {len(crit)} "
                         f"Critical alarm{'' if len(crit) == 1 else 's'} in the last "
                         f"{ctx.get('lookback_hours')} hours, which argues against waiting.")
        else:
            lines.append("No Critical alarms fired in the window — the score is driven "
                         "by the pattern and rate of lower-severity alarms.")
        if hardware:
            lines.append("It is a hardware fault, so it will not clear on its own and "
                         "no script will fix it.")
        lines.append("The honest answer on timing: this is a probability, not a "
                     "guarantee. Weigh it against the downstream count above.")
        return "\n".join(lines)

    # --- what alarms ------------------------------------------------------
    if has("alarm", "log", "event", "what happened", "what fired", "symptom"):
        al = ctx.get("recent_alarms", [])
        if not al:
            return f"No alarms were recorded for {dev} in the scoring window."
        lines = [f"{len(al)} alarms on {dev} in the last "
                 f"{ctx.get('lookback_hours')} hours, newest first:", ""]
        for a in al[:10]:
            lines.append(f"  {str(a.get('timestamp'))[:16].replace('T', ' ')}  "
                         f"{a.get('severity'):<8} {a.get('alarm_name')} "
                         f"({a.get('category')})")
        if len(al) > 10:
            lines.append(f"  … and {len(al) - 10} more.")
        return "\n".join(lines)

    # --- history ----------------------------------------------------------
    if has("history", "before", "previous", "past", "again", "repeat", "prior",
           "how many times"):
        n = ctx.get("n_prior_faults", 0)
        if not n:
            return (f"{dev} has no prior fault episodes on record, so this would be "
                    "its first. The score is coming from current alarm activity "
                    "rather than a bad track record.")
        return (f"{dev} has {n} prior fault episodes on record "
                f"({ctx.get('n_prior_hw_faults', 0)} hardware, "
                f"{ctx.get('n_prior_sw_faults', 0)} software). "
                + ("There is also an unresolved fault still open on it. "
                   if ctx.get("has_pending_fault") else "")
                + "Fault history is one of the features the model uses, so a repeat "
                  "offender is ranked above a never-failed router facing the same alarms.")

    # --- automation -------------------------------------------------------
    if has("script", "automat", "auto", "remediat", "self", "restart", "reboot"):
        if hardware:
            return ("No. Stage 2 classified this as a hardware fault, which is exactly "
                    "the case that cannot be scripted — a board, an optic, a fan or a "
                    "power feed is involved. That classification is why you were paged "
                    "instead of a remediation script being run. "
                    f"Confidence in the hardware classification: "
                    f"{_pct(ctx.get('type_confidence'))}.")
        return ("Yes. Stage 2 classified this as a software fault, so RouterGuardian "
                "attempts scripted remediation automatically — restarting a process, "
                "resetting a session. You were notified so you know it is happening. "
                f"Confidence in the software classification: "
                f"{_pct(ctx.get('type_confidence'))}.")

    # --- fallback ---------------------------------------------------------
    return (f"Here is what I hold on {dev}: a "
            f"{_pct(ctx.get('fault_probability'))} predicted "
            f"{ctx.get('fault_type') or ''} fault within "
            f"{ctx.get('horizon_hours')} hours, {ctx.get('n_events_window')} alarms in "
            f"the scoring window, {ctx.get('n_prior_faults', 0)} prior fault episodes, "
            f"and {len(withproc)} documented fix "
            f"procedure{'' if len(withproc) == 1 else 's'}.\n\n"
            + _suggestions(ctx))


# ------------------------------------------------------------------- api
def reply(alert: dict, question: str) -> tuple[str, str]:
    """
    Answer one question about an alert. Returns (answer, source).

    source is "llm" or "grounded", so the UI can be honest about which engine
    produced the text.
    """
    ctx = load_context(alert)
    if not ctx:
        return ("The context for this alert was not stored, so I cannot answer "
                "questions about it reliably. Re-run the alert from the Fleet tab "
                "to regenerate it.", "grounded")

    history = db.get_messages(alert["id"])
    answer = _ask_llm(ctx, history, question)
    if answer:
        return answer, "llm"
    return _grounded_answer(ctx, question), "grounded"


def opening_message(alert: dict) -> str:
    """The assistant's first turn, shown before the engineer types anything."""
    ctx = load_context(alert)
    if not ctx:
        return "I do not have the context for this alert."
    hardware = ctx.get("fault_type") == "hardware"
    n_proc = len([r for r in ctx.get("remediation", []) if r.get("procedure")])
    return (
        f"I have the full context of this alert on {ctx.get('device_id')} at "
        f"{ctx.get('site') or 'an unknown site'} — the "
        f"{_pct(ctx.get('fault_probability'))} score, the "
        f"{ctx.get('n_events_window')} alarms behind it, the features that drove it, "
        f"and {n_proc} documented Huawei procedure{'' if n_proc == 1 else 's'}.\n\n"
        + ("This one is hardware, so it needs hands on the device.\n\n" if hardware
           else "This one is software, so remediation is being attempted automatically.\n\n")
        + _suggestions(ctx))


def status() -> dict[str, Any]:
    ready, why = config.llm_status()
    last_error = _LAST_ERROR["message"]
    detail = (
        f"Answers are written by {config.LLM_MODEL}, grounded in the stored "
        "incident context. If the API call fails the assistant falls back to "
        "the offline engine rather than going quiet."
        if ready else
        why + " The offline engine uses exactly the same incident context, so the "
              "facts are identical either way."
    )
    if ready and last_error:
        detail = (f"The last call to {config.LLM_MODEL} failed, so answers are coming "
                  f"from the offline engine for now. The API said: {last_error}")
    return {
        "llm_configured": ready,
        "model": config.LLM_MODEL if ready else None,
        "engine": "llm" if ready and not last_error else "grounded",
        "key_present": bool(config.LLM_API_KEY),
        "blocked_reason": None if ready else why,
        "last_error": last_error,
        "detail": detail,
    }
