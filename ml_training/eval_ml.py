"""
Evaluate saved ML artifacts on vectorized EMBER 2018 features.

Examples:
    python backdoor_notebook/ml_training/eval_ml.py ../MalwareBackdoors/ember2018 backdoor_notebook/ml_training/models/lgbm_ember2018.joblib
    python backdoor_notebook/ml_training/eval_ml.py ../MalwareBackdoors/ember2018 /tmp/lgbm_ember2018.joblib --subset test --max-eval-samples 20000
    python backdoor_notebook/ml_training/eval_ml.py ../MalwareBackdoors/ember2018 backdoor_notebook/ml_training/models/lgbm_ember2018.txt --subset test
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "ember2018-mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ml_train_utils import (
    load_saved_model,
    open_labeled_subset,
    open_validation_split,
    prediction_labels,
    prediction_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing vectorized EMBER 2018 dat files.",
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to a saved joblib artifact or supported native model file.",
    )
    parser.add_argument(
        "--model-format",
        type=str,
        default="auto",
        choices=("auto", "joblib", "lightgbm", "catboost", "xgboost"),
        help="How to load model_path. auto detects joblib, LightGBM .txt, CatBoost .cbm, and XGBoost .json/.ubj.",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="test",
        choices=("train", "validation", "test"),
        help="Dataset subset to evaluate. Use validation for a held-out split of X_train.dat.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help="Held-out fraction used when --subset validation.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap on evaluated labeled samples for faster runs.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used for validation split and optional evaluation subsampling.",
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Optional output path for the ROC/AUC PDF.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Optional path to save metrics JSON.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Batch size used for prediction during evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, algorithm, _ = load_saved_model(args.model_path, model_format=args.model_format)

    X_memmap, labeled_indices, y_eval = open_eval_subset(
        args.data_dir,
        subset=args.subset,
        validation_fraction=args.validation_fraction,
        max_eval_samples=args.max_eval_samples,
        random_state=args.random_state,
    )
    y_pred, y_score = batched_predictions(
        model,
        X_memmap,
        labeled_indices,
        batch_size=args.batch_size,
    )

    metrics = {
        "algorithm": algorithm,
        "subset": args.subset,
        "samples": int(y_eval.shape[0]),
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        "macro_f1": float(f1_score(y_eval, y_pred, average="macro")),
    }

    if np.unique(y_eval).shape[0] == 2:
        if y_score is None:
            raise ValueError("Binary ROC/AUC evaluation requires predict_proba or decision_function.")
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        precision, recall, _ = precision_recall_curve(y_eval, y_score)
        metrics["roc_auc"] = float(roc_auc_score(y_eval, y_score))
        metrics["pr_auc"] = float(auc(recall, precision))

        pdf_path = Path(args.pdf_path) if args.pdf_path else default_pdf_path(args.subset, algorithm)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        export_roc_pdf(
            pdf_path,
            y_eval,
            y_score,
            title=f"ROC Curve ({algorithm}, {args.subset})",
        )
        metrics["roc_pdf"] = str(pdf_path)
        print(f"Saved ROC/AUC PDF to {pdf_path}")

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.metrics_path is not None:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"Saved metrics JSON to {metrics_path}")


def open_eval_subset(
    data_dir: str | Path,
    *,
    subset: str,
    validation_fraction: float,
    max_eval_samples: int | None,
    random_state: int,
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    if max_eval_samples is not None and max_eval_samples <= 0:
        raise ValueError("--max-eval-samples must be positive.")
    if subset == "validation":
        return open_validation_split(
            data_dir,
            validation_fraction=validation_fraction,
            max_eval_samples=max_eval_samples,
            random_state=random_state,
        )
    return open_labeled_subset(
        data_dir,
        subset=subset,
        max_samples=max_eval_samples,
        random_state=random_state,
    )


def default_pdf_path(subset: str, algorithm: str) -> Path:
    safe_algorithm = str(algorithm).replace(" ", "_")
    return Path(__file__).resolve().parent / f"Classifier_ROC_AUC_{subset}_{safe_algorithm}.pdf"


def batched_predictions(
    model,
    X_memmap: np.memmap,
    labeled_indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    pred_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    scores_available = True

    for start in range(0, labeled_indices.shape[0], batch_size):
        batch_indices = labeled_indices[start : start + batch_size]
        X_batch = np.asarray(X_memmap[batch_indices], dtype=np.float32)
        pred_chunks.append(prediction_labels(model, X_batch))
        batch_scores = prediction_scores(model, X_batch)
        if batch_scores is None:
            scores_available = False
        elif scores_available:
            score_chunks.append(np.asarray(batch_scores))

    y_pred = np.concatenate(pred_chunks)
    if not scores_available:
        return y_pred, None
    return y_pred, np.concatenate(score_chunks)


def export_roc_pdf(pdf_path: Path, y_true: np.ndarray, y_score: np.ndarray, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    fpr_target = 0.01
    index = np.argmin(np.abs(fpr - fpr_target))
    tpr_at_target = float(tpr[index])

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="black", label=f"AUC = {roc_auc:.4f}")
    plt.plot(
        [fpr_target, fpr_target, 0],
        [0, tpr_at_target, tpr_at_target],
        color="red",
        linestyle="--",
        label=f"TPR at 1% FPR = {tpr_at_target:.4f}",
    )
    plt.title(title)
    plt.xlabel("False Positive Rate (log scale)")
    plt.ylabel("True Positive Rate")
    plt.xscale("log")
    plt.xlim(0.00005, 1.0)
    plt.ylim(0.0, 1.02)
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.close()


if __name__ == "__main__":
    main()
