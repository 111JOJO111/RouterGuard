# Model files go here

Copy these two files out of your Kaggle working directory into this folder:

- `stage1_fault_model.json` — Stage 1, fault vs no-fault
- `stage2_type_model.json`  — Stage 2, hardware vs software

In your Kaggle notebook they are written to `/kaggle/working/` by the training
cell. Download them from the notebook's **Output** tab.

The API starts without them but every prediction endpoint returns HTTP 503 and
the UI shows a warning banner until both files are present.

## Important: keep the threshold in sync

`backend/app/config.py` sets `DEPLOY_THRESHOLD = 0.7273`, the recall-prioritized
threshold from the run that produced these models. **If you retrain, the
training cell prints a new threshold and you must update that constant**, or the
alert/no-alert boundary will be wrong even though the probabilities are right.

## Feature contract

The models expect exactly 26 features in the order listed in
`config.FEATURE_COLS`. `days_since_last_fault_device` is deliberately excluded —
it encoded a simulator artifact and inverted the chronic-vs-healthy ranking. If
you retrain with a different feature set, update `FEATURE_COLS` to match.
