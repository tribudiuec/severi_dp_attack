from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any

MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "backdoor-notebook-defense-mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

import lightgbm as lgb

from mw_backdoor import model_utils


@dataclass
class AttackArtifacts:
    experiment_dir: Path
    X_train_watermarked: Any
    y_train_watermarked: np.ndarray
    X_test_watermarked_mw: np.ndarray
    wm_config: dict[str, Any]
    backdoor_model: Any | None = None


def load_notebook_attack_artifacts(
    experiment_dir: str | Path,
    dataset: str = "ember",
    model_id: str = "lightgbm",
) -> AttackArtifacts:
    """Load artifacts saved by backdoor_codex_20pct.ipynb with cfg['save'] set."""
    experiment_dir = Path(experiment_dir)
    required = [
        experiment_dir / "watermarked_X.npy",
        experiment_dir / "watermarked_y.npy",
        experiment_dir / "watermarked_X_test.npy",
        experiment_dir / "wm_config.npy",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing saved attack artifact(s): {}. Re-run the attack cell with "
            "cfg['save'] set before running defenses.".format(", ".join(missing))
        )

    X_train_watermarked = np.load(experiment_dir / "watermarked_X.npy", allow_pickle=True)
    y_train_watermarked = np.load(experiment_dir / "watermarked_y.npy", allow_pickle=True)
    X_test_watermarked_mw = stack_rows(
        np.load(experiment_dir / "watermarked_X_test.npy", allow_pickle=True)
    )
    wm_config = np.load(experiment_dir / "wm_config.npy", allow_pickle=True).item()

    model_path = experiment_dir / f"{dataset}_{model_id}_backdoored"
    backdoor_model = lgb.Booster(model_file=str(model_path)) if model_path.exists() else None

    return AttackArtifacts(
        experiment_dir=experiment_dir,
        X_train_watermarked=X_train_watermarked,
        y_train_watermarked=np.asarray(y_train_watermarked).astype(np.int32),
        X_test_watermarked_mw=X_test_watermarked_mw,
        wm_config=wm_config,
        backdoor_model=backdoor_model,
    )


def select_defense_feature_indices(
    wm_config: dict[str, Any],
    shap_values_df: pd.DataFrame | np.ndarray | None = None,
    top_k: int = 32,
    mode: str = "hybrid",
) -> np.ndarray:
    """Select feature columns used by the defense detector.

    mode='watermark' uses only watermark feature ids.
    mode='shap' uses top mean-absolute SHAP features.
    mode='hybrid' starts with watermark features and pads with top SHAP features.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if mode not in {"watermark", "shap", "hybrid"}:
        raise ValueError("mode must be one of: watermark, shap, hybrid.")

    watermark_ids = np.asarray(wm_config.get("wm_feat_ids", []), dtype=np.int64)
    if mode == "watermark":
        return unique_limited(watermark_ids, limit=top_k)

    if shap_values_df is None:
        if mode == "shap":
            raise ValueError("mode='shap' requires shap_values_df.")
        return unique_limited(watermark_ids, limit=top_k)

    shap_arr = shap_values_df.to_numpy(copy=False) if hasattr(shap_values_df, "to_numpy") else np.asarray(shap_values_df)
    shap_scores = np.nanmean(np.abs(shap_arr), axis=0)
    shap_ranked = np.argsort(shap_scores)[::-1]
    if mode == "shap":
        return unique_limited(shap_ranked, limit=top_k)
    return unique_limited(np.concatenate([watermark_ids, shap_ranked]), limit=top_k)


def unique_limited(values: np.ndarray, limit: int) -> np.ndarray:
    selected: list[int] = []
    seen: set[int] = set()
    for raw_value in values:
        value = int(raw_value)
        if value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= limit:
            break
    return np.asarray(selected, dtype=np.int64)


def run_isolation_forest_defense(
    artifacts: AttackArtifacts,
    feature_indices: np.ndarray,
    contamination: str | float = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    X_gw_selected, is_clean = selected_goodware_matrix(artifacts, feature_indices)
    model = IsolationForest(
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    predictions = model.fit_predict(X_gw_selected)
    remove_goodware_mask = predictions == -1
    return build_defense_result(
        name="isolation_forest",
        artifacts=artifacts,
        remove_goodware_mask=remove_goodware_mask,
        is_clean=is_clean,
        feature_indices=feature_indices,
        details={
            "contamination": contamination,
            "random_state": random_state,
            "suspect_scores": -model.score_samples(X_gw_selected),
        },
    )


def run_spectral_signature_defense(
    artifacts: AttackArtifacts,
    feature_indices: np.ndarray,
    remove_fraction: float | None = None,
) -> dict[str, Any]:
    X_gw_selected, is_clean = selected_goodware_matrix(artifacts, feature_indices)
    X_centered = X_gw_selected - np.mean(X_gw_selected, axis=0)
    _, _, vh = np.linalg.svd(X_centered, full_matrices=False)
    top_vector = vh[0:1]
    scores = np.linalg.norm(np.matmul(top_vector, X_centered.T), axis=0)

    if remove_fraction is None:
        poison_count = int(np.sum(is_clean == 0))
        remove_count = max(poison_count, 1)
    else:
        if not 0 < remove_fraction < 1:
            raise ValueError("remove_fraction must be in (0, 1).")
        remove_count = max(1, int(round(X_gw_selected.shape[0] * remove_fraction)))
    remove_count = min(remove_count, X_gw_selected.shape[0])

    remove_indices = np.argsort(scores)[-remove_count:]
    remove_goodware_mask = np.zeros(X_gw_selected.shape[0], dtype=bool)
    remove_goodware_mask[remove_indices] = True
    return build_defense_result(
        name="spectral_signature",
        artifacts=artifacts,
        remove_goodware_mask=remove_goodware_mask,
        is_clean=is_clean,
        feature_indices=feature_indices,
        details={
            "remove_count": int(remove_count),
            "remove_fraction": None if remove_fraction is None else float(remove_fraction),
            "spectral_scores": scores,
        },
    )


def run_hdbscan_defense(
    artifacts: AttackArtifacts,
    feature_indices: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
    remove_noise: bool = True,
    small_cluster_fraction: float = 0.02,
) -> dict[str, Any]:
    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError("hdbscan is not installed in this environment.") from exc

    X_gw_selected, is_clean = selected_goodware_matrix(artifacts, feature_indices)
    clusterer = hdbscan.HDBSCAN(
        metric="euclidean",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X_gw_selected)
    remove_goodware_mask = np.zeros(labels.shape[0], dtype=bool)
    if remove_noise:
        remove_goodware_mask |= labels == -1

    max_small_cluster_size = max(1, int(round(labels.shape[0] * small_cluster_fraction)))
    for label in set(labels.tolist()):
        if label == -1:
            continue
        label_mask = labels == label
        if int(np.sum(label_mask)) <= max_small_cluster_size:
            remove_goodware_mask |= label_mask

    return build_defense_result(
        name="hdbscan",
        artifacts=artifacts,
        remove_goodware_mask=remove_goodware_mask,
        is_clean=is_clean,
        feature_indices=feature_indices,
        details={
            "labels": labels,
            "min_cluster_size": int(min_cluster_size),
            "min_samples": None if min_samples is None else int(min_samples),
            "remove_noise": bool(remove_noise),
            "small_cluster_fraction": float(small_cluster_fraction),
        },
    )


def selected_goodware_matrix(
    artifacts: AttackArtifacts,
    feature_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = artifacts.y_train_watermarked
    X_gw = artifacts.X_train_watermarked[y == 0]
    X_gw_selected = selected_dense_matrix(X_gw, feature_indices)
    X_gw_selected = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X_gw_selected)
    return X_gw_selected, goodware_is_clean_bitmap(y, artifacts.wm_config)


def selected_dense_matrix(X: Any, feature_indices: np.ndarray) -> np.ndarray:
    if scipy.sparse.issparse(X):
        return X[:, feature_indices].toarray().astype(np.float32, copy=False)
    return np.asarray(X[:, feature_indices], dtype=np.float32)


def stack_rows(X: Any) -> np.ndarray:
    if scipy.sparse.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    arr = np.asarray(X)
    if arr.dtype != object:
        return arr.astype(np.float32, copy=False)
    if arr.shape == ():
        arr = arr.item()
    rows = [np.asarray(row, dtype=np.float32).reshape(-1) for row in arr]
    return np.vstack(rows)


def goodware_is_clean_bitmap(
    y_train_watermarked: np.ndarray,
    wm_config: dict[str, Any],
) -> np.ndarray:
    goodware_count = int(np.sum(y_train_watermarked == 0))
    poison_count = int(wm_config["num_gw_to_watermark"])
    if poison_count > goodware_count:
        raise ValueError("Poison count is larger than the goodware count.")
    is_clean = np.ones(goodware_count, dtype=np.int32)
    is_clean[-poison_count:] = 0
    return is_clean


def build_defense_result(
    *,
    name: str,
    artifacts: AttackArtifacts,
    remove_goodware_mask: np.ndarray,
    is_clean: np.ndarray,
    feature_indices: np.ndarray,
    details: dict[str, Any],
) -> dict[str, Any]:
    X_filtered, y_filtered = filter_watermarked_training_set(
        artifacts.X_train_watermarked,
        artifacts.y_train_watermarked,
        remove_goodware_mask,
    )
    removed_poison = int(np.sum(remove_goodware_mask & (is_clean == 0)))
    removed_clean = int(np.sum(remove_goodware_mask & (is_clean == 1)))
    poison_total = int(np.sum(is_clean == 0))
    clean_total = int(np.sum(is_clean == 1))
    return {
        "name": name,
        "X_filtered": X_filtered,
        "y_filtered": y_filtered,
        "remove_goodware_mask": remove_goodware_mask,
        "is_clean": is_clean,
        "feature_indices": np.asarray(feature_indices, dtype=np.int64),
        "removed_total": int(np.sum(remove_goodware_mask)),
        "removed_poison": removed_poison,
        "removed_clean": removed_clean,
        "poison_total": poison_total,
        "clean_total": clean_total,
        "poison_recall": safe_divide(removed_poison, poison_total),
        "clean_false_positive_rate": safe_divide(removed_clean, clean_total),
        "details": details,
    }


def filter_watermarked_training_set(
    X_train_watermarked: Any,
    y_train_watermarked: np.ndarray,
    remove_goodware_mask: np.ndarray,
) -> tuple[Any, np.ndarray]:
    y = np.asarray(y_train_watermarked)
    keep_mask = np.ones(y.shape[0], dtype=bool)
    goodware_positions = np.flatnonzero(y == 0)
    if remove_goodware_mask.shape[0] != goodware_positions.shape[0]:
        raise ValueError("remove_goodware_mask length must match goodware count.")
    keep_mask[goodware_positions[remove_goodware_mask]] = False
    return X_train_watermarked[keep_mask], y[keep_mask]


def train_and_evaluate_defense(
    defense_result: dict[str, Any],
    X_clean_test: Any,
    y_clean_test: np.ndarray,
    X_test_watermarked_mw: np.ndarray,
    model_id: str = "lightgbm",
) -> dict[str, Any]:
    defended_model = model_utils.train_model(
        model_id=model_id,
        x_train=defense_result["X_filtered"],
        y_train=defense_result["y_filtered"],
    )
    clean_pred = predict_binary(defended_model, X_clean_test)
    wm_pred = predict_binary(defended_model, X_test_watermarked_mw)

    clean_accuracy = float(np.mean(clean_pred == y_clean_test))
    cm = confusion_matrix(y_clean_test, clean_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    watermarked_detection_rate = float(np.mean(wm_pred == 1))
    metrics = {
        "defense": defense_result["name"],
        "removed_total": defense_result["removed_total"],
        "removed_poison": defense_result["removed_poison"],
        "removed_clean": defense_result["removed_clean"],
        "poison_total": defense_result["poison_total"],
        "poison_recall": defense_result["poison_recall"],
        "clean_false_positive_rate": defense_result["clean_false_positive_rate"],
        "filtered_train_rows": int(defense_result["y_filtered"].shape[0]),
        "clean_accuracy": clean_accuracy,
        "clean_fp_rate": safe_divide(fp, fp + tn),
        "clean_fn_rate": safe_divide(fn, fn + tp),
        "watermarked_malware_detection_rate": watermarked_detection_rate,
        "watermarked_malware_evasion_rate": 1.0 - watermarked_detection_rate,
    }
    return {
        "model": defended_model,
        "metrics": metrics,
        "clean_confusion_matrix": cm,
    }


def evaluate_original_or_backdoored_model(
    model: Any,
    X_clean_test: Any,
    y_clean_test: np.ndarray,
    X_test_watermarked_mw: np.ndarray,
    label: str,
) -> dict[str, Any]:
    clean_pred = predict_binary(model, X_clean_test)
    wm_pred = predict_binary(model, X_test_watermarked_mw)
    cm = confusion_matrix(y_clean_test, clean_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    watermarked_detection_rate = float(np.mean(wm_pred == 1))
    return {
        "defense": label,
        "removed_total": 0,
        "removed_poison": 0,
        "removed_clean": 0,
        "poison_total": np.nan,
        "poison_recall": np.nan,
        "clean_false_positive_rate": np.nan,
        "filtered_train_rows": np.nan,
        "clean_accuracy": float(np.mean(clean_pred == y_clean_test)),
        "clean_fp_rate": safe_divide(fp, fp + tn),
        "clean_fn_rate": safe_divide(fn, fn + tp),
        "watermarked_malware_detection_rate": watermarked_detection_rate,
        "watermarked_malware_evasion_rate": 1.0 - watermarked_detection_rate,
    }


def predict_binary(model: Any, X: Any, threshold: float = 0.5) -> np.ndarray:
    if scipy.sparse.issparse(X):
        scores = model.predict(X)
    else:
        scores = model.predict(np.asarray(X, dtype=np.float32))
    scores = np.asarray(scores)
    if scores.ndim == 2:
        scores = scores[:, 1]
    return (scores >= threshold).astype(np.int32)


def metrics_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "defense",
        "removed_total",
        "removed_poison",
        "removed_clean",
        "poison_total",
        "poison_recall",
        "clean_false_positive_rate",
        "filtered_train_rows",
        "clean_accuracy",
        "clean_fp_rate",
        "clean_fn_rate",
        "watermarked_malware_detection_rate",
        "watermarked_malware_evasion_rate",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else 0.0
