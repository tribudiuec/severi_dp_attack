"""
Train and save ML classifiers on vectorized EMBER 2018 features.

Examples:
    python backdoor_notebook/ml_training/train_ml.py ../MalwareBackdoors/ember2018 backdoor_notebook/ml_training/models/lgbm_ember2018.joblib
    python backdoor_notebook/ml_training/train_ml.py ../MalwareBackdoors/ember2018 /tmp/xgb_ember2018.joblib --algorithm xgboost --param n_estimators=700
    python backdoor_notebook/ml_training/train_ml.py ../MalwareBackdoors/ember2018 /tmp/rf_ember2018.joblib --algorithm random_forest --max-train-samples 50000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ml_train_utils import (
    SUPPORTED_ALGORITHMS,
    build_estimator,
    canonical_algorithm,
    dataset_summary,
    evaluate_classifier,
    fit_estimator,
    load_labeled_training_data,
    parse_cli_params,
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
        help="Path where the trained joblib model artifact will be saved.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="lightgbm",
        choices=SUPPORTED_ALGORITHMS,
        help="Classifier to train.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Estimator parameter override in key=value form. Values are parsed as JSON when possible.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Fraction of labeled training data reserved for validation.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on labeled training samples. Omit to use all labeled training rows.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for subsampling, validation split, and model defaults.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional path to save validation metrics as JSON.",
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


def main() -> None:
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise ValueError(f"Not a directory: {args.data_dir}")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        raise ValueError("--max-train-samples must be positive.")

    algorithm = canonical_algorithm(args.algorithm)
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    cli_params = parse_cli_params(args.param)
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

    classes = np.unique(y_train)
    estimator = build_estimator(
        algorithm,
        cli_params,
        num_classes=classes.shape[0],
        random_state=args.random_state,
    )

    print(
        json.dumps(
            {
                "algorithm": algorithm,
                "dataset": dataset_summary(args.data_dir),
                "num_classes": int(classes.shape[0]),
                "num_features": int(X_train.shape[1]),
                "params": cli_params,
                "train_samples": int(X_train.shape[0]),
                "validation_samples": int(X_val.shape[0]),
            },
            indent=2,
            sort_keys=True,
        )
    )

    estimator = fit_estimator(estimator, X_train, y_train, X_val, y_val)
    metrics = evaluate_classifier(estimator, X_val, y_val)
    print(json.dumps({"metrics": metrics}, indent=2, sort_keys=True))

    model_outputs = save_model_artifact(
        estimator,
        model_path,
        algorithm=algorithm,
        params=cli_params,
        metrics=metrics,
        num_features=X_train.shape[1],
        classes=np.unique(y),
        booster_path=args.booster_path,
        native_model_path=args.native_model_path,
        manifest_path=args.manifest_path,
        extra_metadata={
            "data_dir": str(Path(args.data_dir).resolve()),
            "max_train_samples": args.max_train_samples,
            "random_state": args.random_state,
            "validation_fraction": args.validation_fraction,
        },
    )
    print(f"Saved model artifact to {model_outputs['joblib_model_path']}")
    if "native_model_path" in model_outputs:
        print(f"Saved native model to {model_outputs['native_model_path']}")
    print(f"Saved model manifest to {model_outputs['manifest_path']}")

    if args.metrics_path is not None:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"Saved validation metrics to {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(str(exc))
