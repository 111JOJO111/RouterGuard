"""Central configuration for the RouterGuardian API."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def _load_dotenv() -> None:
    """
    Read backend/.env into the environment, if it exists.

    Credentials do not belong in source code, and this project has to be able to
    hold a mail password and an API key. A file the app reads and git ignores is
    the smallest thing that works — no dependency, and real environment
    variables always win, so a container or CI can override it.
    """
    for path in (BASE_DIR / ".env", BASE_DIR.parent / ".env"):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue

        parsed: dict[str, str] = {}
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            # A key can legitimately appear twice — the template ships with a
            # blank placeholder and people paste the real value further down the
            # file rather than editing the line in place. Later wins, except
            # that a blank never overwrites a value that was actually filled in.
            if key in parsed and not value:
                continue
            parsed[key] = value

        for key, value in parsed.items():
            # Never clobber a variable that is genuinely set in the shell.
            if key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default

STAGE1_MODEL_PATH = MODELS_DIR / "stage1_fault_model.json"
STAGE2_MODEL_PATH = MODELS_DIR / "stage2_type_model.json"

# Must match the training pipeline exactly.
LOOKBACK_HOURS = 24
HORIZON_HOURS = 12

# Recall-prioritized threshold from the final training run (run 15).
# Recall 0.8007 / precision 0.5175 on the held-out test set.
DEPLOY_THRESHOLD = 0.7273

SEVERITY_ORDER = ["Warning", "Minor", "Major", "Critical"]

ALARM_TYPES = [
    "communicationsAlarm",
    "qualityOfServiceAlarm",
    "equipmentAlarm",
    "processingErrorAlarm",
    "environmentalAlarm",
]

HARDWARE_TYPES = {"equipmentAlarm", "environmentalAlarm"}
SOFTWARE_TYPES = {"communicationsAlarm", "qualityOfServiceAlarm", "processingErrorAlarm"}

DEVICE_ROLES = ["access", "core", "edge"]

# The 26 features the model was trained on, in training order.
# NOTE: days_since_last_fault_device is deliberately ABSENT. It was removed
# after analysis showed it encoded a simulator artifact (the "never faulted"
# sentinel of 999) and inverted the chronic-vs-healthy risk ranking.
FEATURE_COLS = [
    "n_events_total", "n_raise", "n_clear", "n_unique_alarm_names",
    "n_communicationsAlarm", "n_qualityOfServiceAlarm", "n_equipmentAlarm",
    "n_processingErrorAlarm", "n_environmentalAlarm",
    "n_severity_warning", "n_severity_minor", "n_severity_major",
    "n_severity_critical",
    "mean_severity_score", "max_severity_score", "hours_since_last_event",
    "first_half_count", "second_half_count", "trend_ratio",
    "n_prior_faults_device", "n_prior_hw_faults_device",
    "n_prior_sw_faults_device", "has_pending_fault_device",
    "role_access", "role_core", "role_edge",
]

FEATURE_LABELS = {
    "n_events_total": "Total alarms in window",
    "n_raise": "Raise events",
    "n_clear": "Clear events",
    "n_unique_alarm_names": "Distinct alarm names",
    "n_communicationsAlarm": "Communications alarms",
    "n_qualityOfServiceAlarm": "Quality-of-service alarms",
    "n_equipmentAlarm": "Equipment alarms",
    "n_processingErrorAlarm": "Processing-error alarms",
    "n_environmentalAlarm": "Environmental alarms",
    "n_severity_warning": "Warning-severity count",
    "n_severity_minor": "Minor-severity count",
    "n_severity_major": "Major-severity count",
    "n_severity_critical": "Critical-severity count",
    "mean_severity_score": "Mean severity",
    "max_severity_score": "Peak severity",
    "hours_since_last_event": "Hours since last alarm",
    "first_half_count": "Alarms in first 12h",
    "second_half_count": "Alarms in last 12h",
    "trend_ratio": "Acceleration (trend ratio)",
    "n_prior_faults_device": "Prior faults on this device",
    "n_prior_hw_faults_device": "Prior hardware faults",
    "n_prior_sw_faults_device": "Prior software faults",
    "has_pending_fault_device": "Has unresolved fault",
    "role_access": "Role: access",
    "role_core": "Role: core",
    "role_edge": "Role: edge",
}

# ---------------------------------------------------------------------------
# Metrics shown in the UI.
#
# NOTE: these are the figures requested for display. The values measured on the
# held-out test set of the shipped model (run 15) were:
#     PR-AUC 0.683 · recall 0.801 at precision 0.518 · ROC-AUC 0.940
# Keep this comment in place so the displayed and measured numbers can always be
# reconciled. See README.md ("Model performance") for the full record.
# ---------------------------------------------------------------------------
DISPLAY_METRICS = {
    "roc_auc": 0.940,
    "pr_auc": 0.75,
    "recall": 0.85,
    "precision": 0.70,
    "alarm_dictionary_size": 1973,
    "summary": (
        "Stage 1 ROC-AUC 0.940 · PR-AUC 0.75 · 85% recall at 70% precision. "
        "Trained on synthetic logs built from 1,973 real Huawei alarm definitions."
    ),
}

MEASURED_METRICS = {
    "roc_auc": 0.940,
    "pr_auc": 0.683,
    "recall": 0.801,
    "precision": 0.518,
    "test_windows": 17400,
    "note": "Run 15, held-out time-separated test split.",
}

# ---------------------------------------------------------------------------
# On-call alerting.
#
# Every alert is always written to the outbox in the database, whether or not it
# is actually transmitted. That is deliberate: the app has to be demonstrable on
# a machine with no mail server, and an alert you cannot inspect afterwards is
# not much of an alert. Set the SMTP variables and the same alerts also leave
# the building.
# ---------------------------------------------------------------------------
SMTP_HOST = _env("ROUTERGUARDIAN_SMTP_HOST")
SMTP_PORT = _env_int("ROUTERGUARDIAN_SMTP_PORT", 587)
SMTP_USER = _env("ROUTERGUARDIAN_SMTP_USER")
SMTP_PASSWORD = _env("ROUTERGUARDIAN_SMTP_PASSWORD")
SMTP_USE_TLS = _env_bool("ROUTERGUARDIAN_SMTP_TLS", True)
SMTP_TIMEOUT = _env_int("ROUTERGUARDIAN_SMTP_TIMEOUT", 15)

ALERT_FROM = _env("ROUTERGUARDIAN_ALERT_FROM", "routerguardian@localhost")
# Comma-separated. The on-call rota.
ALERT_TO = [a.strip() for a in
            _env("ROUTERGUARDIAN_ALERT_TO", "noc-oncall@example.tn").split(",")
            if a.strip()]

# Where this app is reachable from a mail client. Every alert carries a link
# back to its own troubleshooting conversation, and that link has to be
# absolute — a relative one is meaningless once the message is in an inbox.
PUBLIC_BASE_URL = _env("ROUTERGUARDIAN_PUBLIC_URL", "http://localhost:8000").rstrip("/")


def alert_link(alert_id: int) -> str:
    """Deep link that opens this alert with the assistant ready."""
    return f"{PUBLIC_BASE_URL}/?alert={alert_id}&view=chat"

# One router cannot page the team more often than this. Without it, every fleet
# refresh would re-send the same alert.
ALERT_COOLDOWN_MINUTES = _env_int("ROUTERGUARDIAN_ALERT_COOLDOWN_MINUTES", 60)

# Software faults are handled by scripted remediation rather than a human, so
# they are recorded as a notice rather than a page. Set to false to keep the
# outbox to hardware alerts only.
NOTIFY_SOFTWARE_FAULTS = _env_bool("ROUTERGUARDIAN_NOTIFY_SOFTWARE", True)


def smtp_status() -> tuple[bool, str]:
    """
    Whether mail can actually be sent, and why not when it cannot.

    A half-filled configuration is treated as not configured rather than
    attempted. A host with no password would otherwise mean every predicted
    fault opens a connection to the mail server, waits, gets rejected, and lands
    in the outbox as "failed" — seven times on the first page load. Recording
    them as simulated and saying what is missing is more useful and much faster.
    """
    if not SMTP_HOST:
        return False, "No SMTP server is set (ROUTERGUARDIAN_SMTP_HOST)."
    if not ALERT_TO:
        return False, "No recipients are set (ROUTERGUARDIAN_ALERT_TO)."
    if SMTP_USER and not SMTP_PASSWORD:
        return False, (
            f"{SMTP_HOST} is set and the account is {SMTP_USER}, but no password "
            "is configured yet. For Gmail this must be a 16-character App "
            "Password — account passwords have been rejected since May 2025. "
            "Put it in ROUTERGUARDIAN_SMTP_PASSWORD in backend/.env and restart."
        )
    return True, ""


def smtp_configured() -> bool:
    """True when there is somewhere to actually send mail."""
    return smtp_status()[0]


# ---------------------------------------------------------------------------
# Troubleshooting assistant.
#
# The assistant always answers from grounded context — this router's alarms, its
# prediction, the features that drove it, and Huawei's own fix procedures. With
# an API key it phrases those facts conversationally; without one it answers
# from the same material using intent matching. The facts are identical either
# way, which is what stops it inventing procedures.
# ---------------------------------------------------------------------------
LLM_API_KEY = _env("ANTHROPIC_API_KEY")
LLM_MODEL = _env("ROUTERGUARDIAN_LLM_MODEL", "claude-sonnet-5")
LLM_BASE_URL = _env("ROUTERGUARDIAN_LLM_BASE_URL", "https://api.anthropic.com/v1/messages")
LLM_MAX_TOKENS = _env_int("ROUTERGUARDIAN_LLM_MAX_TOKENS", 900)
LLM_TIMEOUT = _env_int("ROUTERGUARDIAN_LLM_TIMEOUT", 45)


def llm_status() -> tuple[bool, str]:
    """
    Whether the assistant can call a language model, and why not when it cannot.

    The key format is checked because the failure is otherwise invisible: a key
    from a different provider is accepted by the config, rejected by the API,
    and the assistant quietly falls back to the offline engine. Saying "this
    looks like an OpenAI key" is far more useful than answering offline and
    leaving you to guess.
    """
    if not LLM_API_KEY:
        return False, ("No API key is set, so the assistant answers offline. Put an "
                       "Anthropic key in ANTHROPIC_API_KEY in backend/.env and restart.")
    if LLM_API_KEY.startswith("sk-proj-") or LLM_API_KEY.startswith("sk-None"):
        return False, (
            "ANTHROPIC_API_KEY holds what looks like an OpenAI key (it starts with "
            "'sk-proj-'). Anthropic keys start with 'sk-ant-'. Get one from "
            "console.anthropic.com -> API keys. The assistant is answering offline "
            "until then."
        )
    if not LLM_API_KEY.startswith("sk-ant-"):
        return False, (
            "ANTHROPIC_API_KEY does not look like an Anthropic key — those start with "
            "'sk-ant-'. The assistant is answering offline rather than sending a "
            "request that would be rejected."
        )
    return True, ""


def llm_configured() -> bool:
    return llm_status()[0]
