# Installing your trained model

Three files, three steps, one command to verify. About 5 minutes.

---

## Step 1 — Get the files out of Kaggle

Run your notebook through **Step 5 (training)**. It ends with:

```
Done. Models saved to /kaggle/working/.
```

Then in the right-hand panel open the **Output** tab (or **Data → Output**) and
download:

| File | What it is |
|---|---|
| `stage1_fault_model.json` | predicts *whether* a fault happens in the next 12h |
| `stage2_type_model.json` | predicts *what kind* — hardware vs software |
| `synthetic_alarm_log.csv` | the alarm history that becomes your network |
| `huawei_alarm_dictionary_final.csv` | **optional** — alarm descriptions and Huawei's remediation steps |

You do **not** need the other notebook outputs. `fault_episodes.csv`,
`train_windowed.csv`, `test_windowed.csv`, `windowed_dataset.csv` and
`shap_summary_stage1.png` are training intermediates; the app rebuilds features
from the raw alarm log and computes SHAP live per prediction.

Not seeing them? Add a cell and run `!ls -la /kaggle/working/` to confirm they
were written.

Also copy down the threshold line the training cell printed — you need it in
step 3:

```
Test (recall-prioritized, target>=80%) @ threshold=0.7273:
                                                  ^^^^^^ this number
```

## Step 2 — Drop them in

```
RouterGuard/
└── backend/
    ├── models/
    │   ├── stage1_fault_model.json   <- here
    │   └── stage2_type_model.json    <- here
    └── data/
        ├── alarm_log.csv             <- rename synthetic_alarm_log.csv to this
        └── alarm_dictionary.csv      <- optional: huawei_alarm_dictionary_final.csv
```

The CSV is optional here — you can also upload it from the **Data** tab in the
running app, or just use the built-in demo network. Putting it at
`backend/data/alarm_log.csv` means it loads automatically at startup.

## Step 3 — Set the threshold

Open `backend/app/config.py`:

```python
DEPLOY_THRESHOLD = 0.7273
```

Make it match the number from step 1. If it doesn't match, probabilities are
still correct but the alert / no-alert line is in the wrong place.

## Step 4 — Verify

```bash
cd RouterGuard
pip install -r backend/requirements.txt      # first time only
python -m backend.app.check_model
```

This checks everything and tells you exactly what to fix. A good result:

```
1. Model files
  [OK]  Stage 1 (fault): stage1_fault_model.json  (246 KB)
  [OK]  Stage 2 (type): stage2_type_model.json  (70 KB)
2. Dependencies
  [OK]  xgboost 3.2.0
  [OK]  scikit-learn 1.7.2
3. Loading
  [OK]  both models loaded
4. Feature contract
  [OK]  exact match, same order
5. Test prediction
  [OK]  P(fault) = 0.8134  -> FAULT PREDICTED (hardware)
  [OK]  explanation works — top driver: Alarms in last 12h (+0.940)

  RESULT: ready — start the app with  docker compose up --build
```

## Step 5 — Run it

```bash
docker compose up --build
```

Open <http://localhost:8000>. The header should read **model online** and the
Fleet tab should show real percentages instead of dashes.

---

## The alarm dictionary (optional)

Drop `huawei_alarm_dictionary_final.csv` at
`backend/data/alarm_dictionary.csv`, or upload it from the **Data** tab.

With it loaded, the Router inspector gains an **Alarm reference & remediation**
panel: each alarm the router raised is shown with its official description, its
impact on the system, the probable causes, and Huawei's documented fix
procedure — labelled *alert the team* for hardware alarms and *can be scripted*
for software ones, matching the two-stage decision the model makes.

It is stored separately from the network data, so re-seeding the topology or
uploading a new alarm log does not remove it.

## The app adapts to your model

You do **not** have to match the app's feature list exactly. On load, the app
reads the feature names stored inside your model file and adapts:

- **Different column order** — reordered automatically, nothing to do.
- **Your model uses `days_since_last_fault_device`** (i.e. you did not drop it)
  — the app computes that feature and feeds it in. It logs a note and carries on.
- **Your model needs a feature the app cannot compute** — it refuses to load and
  names the missing features, rather than producing silently wrong predictions.

`check_model` reports which case you are in under "Feature contract".

---

## Troubleshooting

**"Models not loaded" in the app**
Run `python -m backend.app.check_model` — it prints the precise reason.
Most often the filenames are wrong: they must be exactly
`stage1_fault_model.json` and `stage2_type_model.json`.

**`Feature shape mismatch, expected: 26`**
Your model wants a feature set the app can't build. `check_model` lists exactly
which features. Either retrain with the app's set, or add the missing feature to
`FEATURE_COLS` and `features.build_feature_row`.

**`sklearn needs to be installed in order to use this module`**
`pip install -r backend/requirements.txt`. XGBoost's classifier API needs
scikit-learn.

**`libgomp.so.1: cannot open shared object file`** (Linux, no Docker)
`sudo apt-get install libgomp1`

**Every router scores near zero**
Usually the reference time. The app treats "now" as the newest timestamp in your
data, not today's date. If your log ends long after its fault episodes, most
routers genuinely have quiet 24-hour windows. Check `reference_time` in
`/api/health`.

**You ran an earlier version of this app**
Nothing to do. A database built by an older version is detected and rebuilt
automatically at startup.

---

## Retraining later

Copy the two new `.json` files over the old ones, update `DEPLOY_THRESHOLD` from
the new training output, re-run `check_model`, restart. No image rebuild needed —
`docker-compose.yml` mounts the models directory from the host.
