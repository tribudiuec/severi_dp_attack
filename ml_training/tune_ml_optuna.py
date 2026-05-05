"""
Tune ML classifiers on vectorized EMBER 2018 features with Optuna.

Examples:
    python backdoor_notebook/ml_training/tune_ml_optuna.py ../MalwareBackdoors/ember2018 backdoor_notebook/ml_training/models/lgbm_ember2018.joblib --n-trials 30
    python backdoor_notebook/ml_training/tune_ml_optuna.py ../MalwareBackdoors/ember2018 /tmp/xgb_ember2018.joblib --algorithm xgboost --max-train-samples 20000 --study-path /tmp/xgb_ember2018.db

The run saves a joblib artifact, a manifest JSON, and native model files where supported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from ml_train_utils import (
    SUPPORTED_ALGORITHMS,
    build_estimator,
    canonical_algorithm,
    dataset_summary,
    fit_estimator,
    load_labeled_training_data,
    prediction_labels,
    prediction_scores,
    save_model_artifact,
    split_train_validation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing vectorized EMBER 2018 files such as X_train.dat and y_train.dat.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path where the tuned joblib model artifact will be saved.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lightgbm",
        choices=SUPPORTED_ALGORITHMS,
        help="Classifier to tune.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        choices=("roc_auc", "average_precision", "macro_f1", "accuracy"),
        help="Optimization metric. Defaults to roc_auc for binary labels.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for the Optuna study.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of labeled training data reserved for validation during tuning.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50000,
        help="Cap on labeled training samples used during tuning and final fitting.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for subsampling, validation split, and model defaults.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optional Optuna study name.",
    )
    parser.add_argument(
        "--study-path",
        type=str,
        default=None,
        help="Optional SQLite path for resuming the Optuna study.",
    )
    parser.add_argument(
        "--best-params-path",
        type=str,
        default=None,
        help="Optional path to save the best parameter JSON summary.",
    )
    parser.add_argument(
        "--booster-path",
        type=str,
        default=None,
        help="Deprecated alias for --native-model-path. Kept for LightGBM compatibility.",
    )
    parser.add_argument(
        "--native-model-path",
        type=str,
        default=None,
        help="Optional native model path. Defaults to .txt for LightGBM, .cbm for CatBoost, .json for XGBoost.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=None,
        help="Optional JSON manifest path describing saved model outputs.",
    )
    return parser.parse_args()


def require_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is not installed in this environment. Install it before "
            "running tune_ml_optuna.py."
        ) from exc
    return optuna


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Not a directory: {args.data_dir}")
    if args.n_trials <= 0:
        raise ValueError("--n-trials must be positive.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max-train-samples must be positive.")

    algorithm = canonical_algorithm(args.algorithm)
    optuna = require_optuna()

    X, y = load_labeled_training_data(
        args.data_dir,
        max_train_samples=args.max_train_samples,
        random_state=args.random_state,
    )
    X_train, X_val, y_train, y_val = split_train_validation(
        X,
        y,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
    )
    metric = choose_metric(args.metric, y_train)

    study_kwargs: dict[str, Any] = {
        "direction": "maximize",
        "study_name": args.study_name or f"{algorithm}_{metric}_ember2018",
    }
    if args.study_path is not None:
        study_path = Path(args.study_path)
        study_path.parent.mkdir(parents=True, exist_ok=True)
        study_kwargs["storage"] = f"sqlite:///{study_path}"
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    objective = build_objective(
        algorithm=algorithm,
        metric=metric,
        random_state=args.random_state,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

    best_trial = study.best_trial
    best_params = dict(best_trial.user_attrs["model_params"])
    best_value = float(best_trial.value)

    run_summary = {
        "algorithm": algorithm,
        "metric": metric,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "train_samples": int(X_train.shape[0]),
        "validation_samples": int(X_val.shape[0]),
        "dataset": dataset_summary(args.data_dir),
    }
    print(json.dumps(run_summary, indent=2, sort_keys=True))

    final_estimator = build_estimator(
        algorithm,
        best_params,
        num_classes=np.unique(y).shape[0],
        random_state=args.random_state,
    )
    final_estimator = fit_estimator(final_estimator, X, y, None, None)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_outputs = save_model_artifact(
        final_estimator,
        model_path,
        algorithm=algorithm,
        params=best_params,
        metrics={f"best_{metric}": best_value},
        num_features=X.shape[1],
        classes=np.unique(y),
        booster_path=args.booster_path,
        native_model_path=args.native_model_path,
        manifest_path=args.manifest_path,
        extra_metadata={
            "best_value": best_value,
            "data_dir": str(Path(args.data_dir).resolve()),
            "max_train_samples": args.max_train_samples,
            "metric": metric,
            "n_trials": len(study.trials),
            "random_state": args.random_state,
            "study_name": study.study_name,
            "study_path": args.study_path,
            "validation_fraction": args.validation_fraction,
        },
    )
    print(f"Saved tuned model artifact to {model_path}")
    if "native_model_path" in model_outputs:
        print(f"Saved native model to {model_outputs['native_model_path']}")
    print(f"Saved model manifest to {model_outputs['manifest_path']}")

    best_params_path = resolve_best_params_path(args.best_params_path, model_path)
    params_summary = {
        "algorithm": algorithm,
        "metric": metric,
        "best_value": best_value,
        "best_params": best_params,
        "n_trials": len(study.trials),
        "study_name": study.study_name,
        "study_path": args.study_path,
    }
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    best_params_path.write_text(json.dumps(params_summary, indent=2, sort_keys=True) + "\n")
    print(f"Saved best params summary to {best_params_path}")


def build_objective(
    *,
    algorithm: str,
    metric: str,
    random_state: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
):
    def objective(trial) -> float:
        params = suggest_params(trial, algorithm)
        trial.set_user_attr("model_params", params)
        estimator = build_estimator(
            algorithm,
            params,
            num_classes=np.unique(y_train).shape[0],
            random_state=random_state,
        )
        estimator = fit_estimator(estimator, X_train, y_train, X_val, y_val)
        return score_estimator(estimator, X_val, y_val, metric)

    return objective


def choose_metric(requested_metric: str | None, y: np.ndarray) -> str:
    if requested_metric is not None:
        return requested_metric
    if np.unique(y).shape[0] == 2:
        return "roc_auc"
    return "macro_f1"


def score_estimator(model, X_val: np.ndarray, y_val: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_val, prediction_labels(model, X_val)))
    if metric == "macro_f1":
        return float(f1_score(y_val, prediction_labels(model, X_val), average="macro"))

    y_score = prediction_scores(model, X_val)
    if y_score is None:
        raise ValueError(f"Metric '{metric}' requires predict_proba or decision_function.")
    if y_score.ndim == 2:
        if y_score.shape[1] != 2:
            raise ValueError(f"Metric '{metric}' is only supported for binary classification.")
        y_score = y_score[:, 1]

    if metric == "roc_auc":
        return float(roc_auc_score(y_val, y_score))
    if metric == "average_precision":
        return float(average_precision_score(y_val, y_score))
    raise ValueError(f"Unsupported metric '{metric}'.")


def suggest_params(trial, algorithm: str) -> dict[str, Any]:
    algorithm = canonical_algorithm(algorithm)
    if algorithm == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=16),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, step=10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 5),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }

    if algorithm == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 150, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        }

    if algorithm == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 700, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
        }

    if algorithm == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 6, 32),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.2, 0.5, None]),
        }

    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def resolve_best_params_path(requested_path: str | None, model_path: Path) -> Path:
    if requested_path is not None:
        return Path(requested_path)
    return model_path.with_name(f"{model_path.stem}_best_params.json")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(str(exc))
