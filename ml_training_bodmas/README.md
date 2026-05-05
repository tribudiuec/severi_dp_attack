# BODMAS ML Training

This folder mirrors `backdoor_notebook/ml_training`, but uses the local BODMAS
`.npz` dataset instead of EMBER `.dat` files. It supports LightGBM, CatBoost,
XGBoost, and RandomForest-style baselines.

The default data directory from the repository root is:

```bash
../bodmat/datasets
```

The loader looks for `bodmas.npz` in that directory. You can also pass the exact
file path, for example `../bodmat/datasets/bodmas.npz`.

Train and save a model without Optuna:

```bash
python backdoor_notebook/ml_training_bodmas/train_ml.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/lgbm_bodmas.joblib \
  --algorithm lightgbm
```

Other algorithms:

```bash
python backdoor_notebook/ml_training_bodmas/train_ml.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/xgb_bodmas.joblib \
  --algorithm xgboost \
  --max-train-samples 50000

python backdoor_notebook/ml_training_bodmas/train_ml.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/rf_bodmas.joblib \
  --algorithm random_forest \
  --max-train-samples 50000

python backdoor_notebook/ml_training_bodmas/train_ml.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/catboost_bodmas.joblib \
  --algorithm catboost \
  --max-train-samples 50000
```

Tune and save a model with Optuna:

```bash
python backdoor_notebook/ml_training_bodmas/tune_ml_optuna.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/lgbm_bodmas.joblib \
  --algorithm lightgbm \
  --n-trials 30 \
  --max-train-samples 50000
```

Evaluate the saved model on the deterministic BODMAS test split:

```bash
python backdoor_notebook/ml_training_bodmas/eval_ml.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/lgbm_bodmas.joblib \
  --subset test
```

By default the split is `20%` test, then `10%` of the remaining data for
validation, using `--random-state 0`. Keep `--test-fraction`,
`--validation-fraction`, and `--random-state` the same between training and
evaluation when you want identical splits.

Each training run saves a joblib model artifact and a manifest:

- `*_bodmas.joblib`: Python joblib artifact with the fitted model, params,
  metrics, features, and class labels. This is the common format for all
  supported algorithms.
- `*_bodmas_manifest.json`: JSON pointer for the next step, including output
  paths and training metadata.

Native model exports are also saved where the library supports them:

- `lgbm_bodmas.txt`: native LightGBM booster file loadable with
  `lightgbm.Booster(model_file=...)`.
- `xgb_bodmas.json`: native XGBoost model file.
- `catboost_bodmas.cbm`: native CatBoost model file.
- RandomForest does not have a separate native export; use the `.joblib`.

You can also evaluate the native booster directly:

```bash
python backdoor_notebook/ml_training_bodmas/eval_ml.py \
  ../bodmat/datasets \
  backdoor_notebook/ml_training_bodmas/models/lgbm_bodmas.txt \
  --subset test
```

Notes:

- `optuna` is required for `tune_ml_optuna.py`.
- `catboost` must be installed before using `--algorithm catboost`.
- The local path in this checkout is `../bodmat/datasets/bodmas.npz`.
- The expected BODMAS archive contains `X` with shape `(n_samples, 2381)` and
  `y` with binary labels.
