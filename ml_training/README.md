# EMBER 2018 ML Training

This folder mirrors the EMBER 2024 example flow for the local EMBER 2018
dataset. It supports LightGBM, CatBoost, XGBoost, and RandomForest-style
baselines over the same vectorized `.dat` files.

The default data directory from the repository root is:

```bash
../MalwareBackdoors/ember2018
```

Train and save a model without Optuna:

```bash
python backdoor_notebook/ml_training/train_ml.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/lgbm_ember2018.joblib \
  --algorithm lightgbm
```

Other algorithms:

```bash
python backdoor_notebook/ml_training/train_ml.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/xgb_ember2018.joblib \
  --algorithm xgboost \
  --max-train-samples 50000

python backdoor_notebook/ml_training/train_ml.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/rf_ember2018.joblib \
  --algorithm random_forest \
  --max-train-samples 50000

python backdoor_notebook/ml_training/train_ml.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/catboost_ember2018.joblib \
  --algorithm catboost \
  --max-train-samples 50000
```

Tune and save a model with Optuna:

```bash
python backdoor_notebook/ml_training/tune_ml_optuna.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/lgbm_ember2018.joblib \
  --algorithm lightgbm \
  --n-trials 30 \
  --max-train-samples 50000
```

Evaluate the saved model on `X_test.dat`/`y_test.dat`:

```bash
python backdoor_notebook/ml_training/eval_ml.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/lgbm_ember2018.joblib \
  --subset test
```

Each training run saves a joblib model artifact and a manifest:

- `*_ember2018.joblib`: Python joblib artifact with the fitted model, params,
  metrics, features, and class labels. This is the common format for all
  supported algorithms.
- `*_ember2018_manifest.json`: JSON pointer for the next step, including output
  paths and training metadata.

Native model exports are also saved where the library supports them:

- `lgbm_ember2018.txt`: native LightGBM booster file loadable with
  `lightgbm.Booster(model_file=...)`.
- `xgb_ember2018.json`: native XGBoost model file.
- `catboost_ember2018.cbm`: native CatBoost model file.
- RandomForest does not have a separate native export; use the `.joblib`.

You can also evaluate the native booster directly:

```bash
python backdoor_notebook/ml_training/eval_ml.py \
  ../MalwareBackdoors/ember2018 \
  backdoor_notebook/ml_training/models/lgbm_ember2018.txt \
  --subset test
```

Notes:

- `optuna` is required for `tune_ml_optuna.py`.
- `catboost` must be installed before using `--algorithm catboost`.
- The local path in this checkout is `../MalwareBackdoors/ember2018`.
- The loader infers the EMBER feature dimension from the dat file sizes, filters
  unlabeled training rows with label `-1`, and handles the float32 labels used
  by these dat files.
