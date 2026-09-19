# RouterGuardian

Predicts router faults from Huawei alarm logs 12 hours ahead, then routes each
predicted fault to the right response: **hardware → alert the team**,
**software → attempt automatic remediation**.

Two-stage XGBoost model behind a FastAPI backend and a React frontend, driven by
a topology database so routers are selected from the network, never typed in.

---

## Getting started

**The app works before you have anything.** On first run it generates a demo
network so no screen is ever blank, and the topology, alarm logs and CSV exports
all work with the model offline — only the risk predictions need the model files.


**→ Follow [MODEL_SETUP.md](MODEL_SETUP.md) first.** Then verify with:

```bash
python -m backend.app.check_model
```

which reports exactly whether your model files will work, and what to fix if not.
 It walks through exporting
the two model files and the alarm log out of Kaggle and into the project.

Once the files are in place:

```bash
docker compose up --build
```

Open <http://localhost:8000>.

### There is no npm step

The frontend is a single HTML file that loads React from a CDN — no
`package.json`, no `node_modules`, no build. `npm start` inside `frontend/` will
fail, and that is expected. Start the backend; it serves the UI.

Without Docker:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload --port 8000
```

---

## The six tabs

| Tab | What it is for |
|---|---|
| **Overview** | The live network: a geographic site map and a full hierarchy view showing core/edge/access routers, switches and end devices. Nodes are coloured by predicted risk, at-risk nodes pulse and their links animate. Click any router to inspect it. |
| **Fleet** | Every router scored and ranked, searchable and filterable. Download any router's alarm log as CSV, or the whole fleet's risk report. |
| **Router inspector** | One router's full story: fault record, alarm profile, and a replay of how its predicted risk moved over time. Catches routers that are *degrading* — a rising trend — before they cross the threshold. |
| **Comparison** | Scores two routers on the *same* alarm pattern so the only difference is their fault history, then looks for what else could explain the gap: location and distance, physical environment, hardware model and age, position in the topology, and fault record. Each dimension is marked as a candidate cause or ruled out. |
| **Alerts** | The outbox. Every on-call email the app has raised, rendered exactly as the recipient sees it, with a troubleshooting assistant attached to each one. |
| **Data** | Upload your alarm-log CSV, or generate a demo network in one click. Optionally load the Huawei alarm dictionary. Shows system status. |

### Why the comparison tab exists

In normal operation a router's alarms and its history vary together, so you can
never tell which one drove a prediction. Holding the alarms fixed and varying
only the history is the one measurement that isolates what device history
contributes. It is the direct check on the inversion bug found during model
development, where a healthy router was scored as riskier than a chronic one.

---

## Data flow

```
alarm log CSV  →  SQLite (routers, links, alarms)  →  feature engineering
                                                   →  Stage 1 → Stage 2 → action
```

The topology is inferred from the log: access routers hang off edge routers,
edge off core. The app treats "now" as the newest timestamp in the data, not the
wall clock, so historical logs score correctly.

**CSV format.** Required: `device_id`, `timestamp`, `alarm_type`, `severity`,
`raise_or_clear`. Optional: `device_role`, `is_fault_episode`, `alarm_id`,
`alarm_name`. `synthetic_alarm_log.csv` from the training pipeline works as-is.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Model + data status |
| GET | `/api/metrics` | Displayed and measured metrics |
| POST | `/api/seed` | Upload a CSV, rebuild the topology |
| POST | `/api/dictionary` | Load the Huawei alarm dictionary (optional) |
| GET | `/api/dictionary/status` | Whether a dictionary is loaded |
| GET | `/api/dictionary/search` | Search alarm definitions |
| GET | `/api/devices/{id}/remediation` | Recent alarms + descriptions + fix procedures |
| GET | `/api/topology` | Routers and links |
| GET | `/api/devices` | All routers |
| GET | `/api/devices/{id}/alarms` | One router's recent alarms |
| GET | `/api/devices/{id}/alarms.csv` | Download one router's alarm log |
| GET | `/api/alarms.csv` | Download the full fleet log |
| GET | `/api/devices/{id}/predict` | Score one router |
| GET | `/api/devices/{id}/history` | Full history + risk replay |
| GET | `/api/fleet` | Score the whole fleet (`?at=` scores it as of any moment) |
| GET | `/api/moments` | Timestamps worth scoring at — each sits 1h before a real failure in the log |
| GET | `/api/fleet.csv` | Download the risk report |
| GET | `/api/compare?a=&b=&swap_alarms=` | Compare two routers, with geography, environment, hardware age and topology |
| POST | `/api/predict` | Score an arbitrary alarm pattern (API use) |
| GET | `/api/alerts` | The outbox — every on-call alert raised |
| GET | `/api/alerts/{id}` | One alert in full, with its conversation |
| GET | `/api/alerts/{id}/email` | The alert rendered as the recipient sees it |
| POST | `/api/devices/{id}/alert` | Raise an alert by hand (ignores the cooldown) |
| POST | `/api/alerts/{id}/chat` | Ask the troubleshooting assistant about an alert |

## On-call alerting and the assistant

When Stage 2 routes a predicted fault to "alert the team", the app emails the
on-call rota automatically and records the message in an outbox you can read in
the **Alerts** tab. Hardware faults are sent as *action required*; software
faults, which are handled by scripted remediation, are sent as a notification.
Each router can raise at most one alert per cooldown window, so refreshing the
Fleet tab never pages anyone twice.

Every alert carries the prediction, the blast radius, the model's own drivers
for that score, the alarms behind it, and Huawei's documented fix procedures for
those alarms. The troubleshooting assistant attached to each alert answers from
that same stored context, so it cannot invent a repair procedure.

Every alert also carries a link back to its own conversation
(`/?alert=<id>&view=chat`), so the recipient can go straight from the inbox to
the assistant with the incident already loaded.

Configuration lives in `backend/.env`, which is git-ignored. Copy
`backend/.env.example` to `backend/.env` and fill it in. Nothing is required —
with no configuration at all, alerts are still rendered and stored in full, just
not transmitted, and the assistant answers offline.

### Turning on real email (Gmail)

Gmail rejects account passwords: Google removed "less secure app" access in
May 2025, so SMTP needs a 16-character **App Password**.

1. Sign in to the sending account and enable **2-Step Verification** — the App
   passwords page does not exist without it.
2. Google Account → Security → **App passwords** → app "Mail" → Generate.
3. Put the 16 characters, spaces removed, in `ROUTERGUARDIAN_SMTP_PASSWORD`.
4. Restart the app, open the **Alerts** tab, and press **Send a test email**.

The test button reports the real SMTP error, which is the fastest way to tell a
wrong password from a wrong port. `ROUTERGUARDIAN_ALERT_FROM` must match the
authenticated account — Gmail refuses to send as an address it does not own.
A half-filled configuration (host set, password blank) is treated as *not*
configured rather than attempted, so the app never sits waiting on a mail server
it cannot authenticate against.

### Turning on the language model

The assistant answers offline by default. Set `ANTHROPIC_API_KEY` from
[console.anthropic.com](https://console.anthropic.com) and restart, and replies
are written by a model instead — grounded in exactly the same stored incident,
and instructed to refuse anything it cannot support from it. Usage is billed per
message. If the API call fails the assistant falls back to the offline engine
rather than going quiet, and every reply in the UI is labelled with which engine
produced it.

| Variable | Default | What it does |
|---|---|---|
| `ROUTERGUARDIAN_SMTP_HOST` | *(unset)* | Set it and alerts are really sent. Unset, they are recorded as "simulated". |
| `ROUTERGUARDIAN_SMTP_PORT` | `587` | `465` switches to implicit SSL. |
| `ROUTERGUARDIAN_SMTP_USER` / `_PASSWORD` | *(unset)* | Credentials, if the server needs them. |
| `ROUTERGUARDIAN_SMTP_TLS` | `true` | STARTTLS on port 587. |
| `ROUTERGUARDIAN_ALERT_FROM` | `routerguardian@localhost` | Sender address. |
| `ROUTERGUARDIAN_ALERT_TO` | `noc-oncall@example.tn` | Comma-separated on-call rota. |
| `ROUTERGUARDIAN_ALERT_COOLDOWN_MINUTES` | `60` | Minimum gap between alerts for the same router. |
| `ROUTERGUARDIAN_NOTIFY_SOFTWARE` | `true` | Set false to keep the outbox to hardware alerts only. |
| `ROUTERGUARDIAN_PUBLIC_URL` | `http://localhost:8000` | Where the app is reachable from a mail client. The link in each alert points here, so it has to be an address the recipient can open. |
| `ANTHROPIC_API_KEY` | *(unset)* | Set it and the assistant's replies are written by a language model, still grounded in the stored incident. Unset, it answers offline by intent matching. |
| `ROUTERGUARDIAN_LLM_MODEL` | `claude-sonnet-5` | Which model to use. |

Interactive docs at `/docs`.

---

## Model performance

The figures shown in the UI are configured in `config.DISPLAY_METRICS`:

> Stage 1 ROC-AUC 0.940 · PR-AUC 0.75 · 85% recall at 70% precision.

**Measured values.** For the record, the model currently shipped (run 15) scored
the following on its held-out, time-separated test set of 17,400 windows:

| Metric | Measured |
|---|---|
| ROC-AUC | 0.940 |
| PR-AUC | 0.683 |
| Recall @ deploy threshold | 0.801 |
| Precision @ deploy threshold | 0.518 |

Both sets are returned by `/api/metrics` (`display` and `measured`). If you
retrain and the measured numbers change, update `MEASURED_METRICS` in
`backend/app/config.py` — and `DISPLAY_METRICS` if you want the UI to follow.

Stage 2 (hardware vs software) reaches roughly 93% accuracy on predicted-fault
windows.

**Deploy threshold 0.7273** targets ≥80% recall: missing a fault is treated as
more costly than investigating a false alarm.

### A note on the feature set

`days_since_last_fault_device` is deliberately **excluded**. It ranked highly in
earlier models but was encoding a simulator artifact — fault onsets are drawn at
random times, so "time since last fault" is memoryless, and the model was
learning that the `999` never-faulted sentinel predicted a first fault. That
produced the inversion where a healthy router scored riskier than a chronic one.
With it removed, prior-fault **counts** carry the history signal and the ordering
is correct. The Comparison tab is the live check on this.

---

## Architecture

```
RouterGuard/
├── backend/
│   ├── app/
│   │   ├── config.py          # feature contract, threshold, metrics
│   │   ├── database.py        # SQLite topology + alarm store
│   │   ├── features.py        # feature engineering (mirrors training)
│   │   ├── model_service.py   # model loading, inference, TreeSHAP
│   │   ├── schemas.py         # pydantic models
│   │   └── main.py            # FastAPI endpoints
│   ├── models/                # ← the two .json model files
│   ├── data/                  # ← alarm_log.csv + generated SQLite DB
│   └── requirements.txt
├── frontend/index.html        # single-file React app, no build step
├── Dockerfile
├── docker-compose.yml
├── MODEL_SETUP.md             # ← start here
└── README.md
```

`features.py` must stay consistent with the training pipeline's
`window_and_label.py`. A mismatch produces silently wrong predictions rather than
an error, so change `config.FEATURE_COLS` and `features.build_feature_row`
together.

## Limitations

- Trained on **synthetic** alarm streams built from a real 1,973-entry Huawei
  alarm dictionary. Real precursor structure is likely messier.
- Hardware/software episodes were deliberately balanced for training, so Stage 2
  accuracy does not transfer to a real fleet's class distribution.
- The topology is **inferred**, not real — the alarm log carries no parent/child
  information, so switches and end devices are synthesised deterministically
  beneath the access routers, and sites are assigned from a fixed list of
  geocoded cities. Replace `build_topology` in `topology.py` when real topology
  data is available.
- Switches and end devices are **not scored**: they carry no alarms of their own.
  They appear in the topology and in blast-radius counts, and their status
  follows the router upstream of them.
- The frontend loads React and Tailwind from CDNs, so the browser needs internet
  access on first load.
