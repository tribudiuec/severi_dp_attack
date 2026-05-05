"""
Utilities for ML baseline training on vectorized EMBER 2018 datasets.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "ember2018-mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from lightgbm import (
        Booster,
        LGBMClassifier,
        early_stopping as lgb_early_stopping,
        log_evaluation as lgb_log_evaluation,
    )
except ImportError:
    Booster = None
    LGBMClassifier = None
    lgb_early_stopping = None
    lgb_log_evaluation = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


SUPPORTED_ALGORITHMS = ("lightgbm", "catboost", "xgboost", "random_forest", "randomforest")
EMBER2018_V2_FEATURE_DIM = 2381
EMBER_CATEGORICAL_FEATURES = [2, 3, 4, 5, 6, 701, 702]


def canonical_algorithm(algorithm: str) -> str:
    algorithm = algorithm.lower()
    if algorithm == "randomforest":
        return "random_forest"
    return algorithm


def parse_cli_params(param_args: list[str] | None) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for raw_param in param_args or []:
        if "=" not in raw_param:
            raise ValueError(
                f"Invalid --param value '{raw_param}'. Expected the form key=value."
            )
        key, raw_value = raw_param.split("=", 1)
        if not key:
            raise ValueError(f"Invalid --param value '{raw_param}'. Missing key.")
        params[key] = parse_value(raw_value)
    return params


def parse_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def dataset_summary(data_dir: str | Path) -> dict[str, Any]:
    data_path = Path(data_dir)
    info_path = data_path / "dataset_info.json"
    summary: dict[str, Any] = {}
    if info_path.is_file():
        summary.update(json.loads(info_path.read_text()))

    for subset in ("train", "test"):
        X_path = data_path / f"X_{subset}.dat"
        y_path = data_path / f"y_{subset}.dat"
        if X_path.is_file() and y_path.is_file():
            y = read_labels(y_path)
            feature_dim = infer_feature_dim(X_path, y.shape[0], summary)
            summary[f"{subset}_rows"] = int(y.shape[0])
            summary[f"{subset}_features"] = int(feature_dim)
            summary[f"{subset}_labels"] = {
                str(int(label)): int(count)
                for label, count in zip(*np.unique(y, return_counts=True))
            }
    return summary


def load_labeled_training_data(
    data_dir: str | Path,
    max_train_samples: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    X_memmap, labeled_indices, y = open_labeled_subset(
        data_dir,
        subset="train",
        max_samples=max_train_samples,
        random_state=random_state,
    )
    X = np.asarray(X_memmap[labeled_indices], dtype=np.float32)
    return X, y


def open_labeled_subset(
    data_dir: str | Path,
    subset: str = "train",
    max_samples: int | None = None,
    random_state: int = 0,
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    data_path = Path(data_dir)
    X_path, y_path = subset_paths(data_path, subset)
    dataset_info = load_dataset_info(data_path)
    y_all = read_labels(y_path)
    feature_dim = infer_feature_dim(X_path, y_all.shape[0], dataset_info)
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r").reshape(-1, feature_dim)

    if X_memmap.shape[0] != y_all.shape[0]:
        raise ValueError(
            f"Row mismatch for subset '{subset}': {X_path} has {X_memmap.shape[0]} "
            f"rows but {y_path} has {y_all.shape[0]} labels."
        )

    labeled_indices = np.flatnonzero(y_all != -1)
    y = y_all[labeled_indices].astype(np.int32, copy=False)
    if max_samples is not None:
        labeled_indices, y = subsample_indices(
            labeled_indices,
            y,
            max_samples=max_samples,
            random_state=random_state,
        )
    return X_memmap, labeled_indices, y


def open_validation_split(
    data_dir: str | Path,
    validation_fraction: float = 0.1,
    max_eval_samples: int | None = None,
    random_state: int = 0,
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    X_memmap, labeled_indices, y = open_labeled_subset(
        data_dir,
        subset="train",
        max_samples=None,
        random_state=random_state,
    )
    _, val_indices, _, y_val = split_indices_train_validation(
        labeled_indices,
        y,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    if max_eval_samples is not None:
        val_indices, y_val = subsample_indices(
            val_indices,
            y_val,
            max_samples=max_eval_samples,
            random_state=random_state,
        )
    return X_memmap, val_indices, y_val


def subset_paths(data_path: Path, subset: str) -> tuple[Path, Path]:
    X_path = data_path / f"X_{subset}.dat"
    y_path = data_path / f"y_{subset}.dat"
    missing = [str(path) for path in (X_path, y_path) if not path.is_file()]
    if missing:
        available = ", ".join(available_subsets(data_path)) or "none"
        raise ValueError(
            "Missing usable EMBER dat file(s): {}. Available usable subsets in {}: {}. "
            "Use --subset validation to evaluate from a held-out split of train.dat."
            .format(", ".join(missing), data_path, available)
        )
    return X_path, y_path


def available_subsets(data_dir: str | Path) -> list[str]:
    data_path = Path(data_dir)
    subsets: list[str] = []
    for X_path in sorted(data_path.glob("X_*.dat")):
        subset = X_path.stem.removeprefix("X_")
        y_path = data_path / f"y_{subset}.dat"
        if X_path.is_file() and y_path.is_file():
            subsets.append(subset)
    if "train" in subsets:
        subsets.append("validation")
    return subsets


def load_dataset_info(data_path: Path) -> dict[str, Any]:
    info_path = data_path / "dataset_info.json"
    if not info_path.is_file():
        return {}
    return json.loads(info_path.read_text())


def read_labels(y_path: str | Path) -> np.ndarray:
    path = Path(y_path)
    float_labels = np.memmap(path, dtype=np.float32, mode="r")
    if is_valid_label_array(float_labels):
        return np.asarray(float_labels, dtype=np.int32)

    int_labels = np.memmap(path, dtype=np.int32, mode="r")
    if is_valid_label_array(int_labels):
        return np.asarray(int_labels, dtype=np.int32)

    raise ValueError(
        f"Could not infer label dtype for {path}. Expected float32 or int32 "
        "labels with values such as -1, 0, and 1."
    )


def is_valid_label_array(labels: np.ndarray) -> bool:
    if labels.ndim != 1 or labels.shape[0] == 0:
        return False
    if not np.all(np.isfinite(labels)):
        return False
    unique = np.unique(np.asarray(labels[:]))
    if unique.shape[0] > 32:
        return False
    return bool(np.array_equal(unique, np.round(unique)))


def infer_feature_dim(
    X_path: str | Path,
    label_count: int,
    dataset_info: dict[str, Any] | None = None,
) -> int:
    path = Path(X_path)
    if label_count <= 0:
        raise ValueError("Cannot infer feature dimension with zero labels.")

    dataset_info = dataset_info or {}
    for key in ("train_dim", "feature_dim", "num_features"):
        value = dataset_info.get(key)
        if value:
            return int(value)

    byte_count = path.stat().st_size
    bytes_per_float32 = np.dtype(np.float32).itemsize
    row_bytes, remainder = divmod(byte_count, label_count)
    if remainder == 0 and row_bytes % bytes_per_float32 == 0:
        return int(row_bytes // bytes_per_float32)
    if byte_count % (EMBER2018_V2_FEATURE_DIM * bytes_per_float32) == 0:
        return EMBER2018_V2_FEATURE_DIM
    raise ValueError(
        f"Could not infer feature dimension for {path} from {byte_count} bytes "
        f"and {label_count} labels."
    )


def split_train_validation(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    return train_test_split(
        X,
        y,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=maybe_stratify(y),
    )


def split_indices_train_validation(
    indices: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.1,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    return train_test_split(
        indices,
        y,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=maybe_stratify(y),
    )


def subsample_indices(
    indices: np.ndarray,
    y: np.ndarray,
    max_samples: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if max_samples is None:
        return indices, y
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive.")
    if max_samples < np.unique(y).shape[0]:
        raise ValueError("--max-samples is smaller than the number of classes.")
    if max_samples < y.shape[0]:
        indices, _, y, _ = train_test_split(
            indices,
            y,
            train_size=max_samples,
            random_state=random_state,
            stratify=maybe_stratify(y),
        )
    return indices, y


def maybe_stratify(y: np.ndarray) -> np.ndarray | None:
    _, counts = np.unique(y, return_counts=True)
    if counts.min() < 2:
        return None
    return y


def build_estimator(
    algorithm: str,
    params: dict[str, Any],
    num_classes: int,
    random_state: int,
):
    algorithm = canonical_algorithm(algorithm)
    if algorithm not in {canonical_algorithm(item) for item in SUPPORTED_ALGORITHMS}:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Choose from: {', '.join(SUPPORTED_ALGORITHMS)}."
        )

    if algorithm == "lightgbm":
        if LGBMClassifier is None:
            raise ImportError(
                "lightgbm is not installed in this environment. Install it before "
                "using --algorithm lightgbm."
            )
        defaults: dict[str, Any] = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 50,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": random_state,
            "verbosity": -1,
        }
        if num_classes == 2:
            defaults.update({"objective": "binary", "class_weight": "balanced"})
        else:
            defaults.update(
                {
                    "objective": "multiclass",
                    "num_class": num_classes,
                    "class_weight": "balanced",
                }
            )
        defaults.update(params)
        return LGBMClassifier(**defaults)

    if algorithm == "catboost":
        if CatBoostClassifier is None:
            raise ImportError(
                "catboost is not installed in this environment. Install it before "
                "using --algorithm catboost."
            )
        defaults = {
            "iterations": 400,
            "learning_rate": 0.05,
            "depth": 8,
            "random_seed": random_state,
            "thread_count": -1,
            "verbose": False,
            "allow_writing_files": False,
        }
        if num_classes == 2:
            defaults.update(
                {
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "auto_class_weights": "Balanced",
                }
            )
        else:
            defaults.update(
                {
                    "loss_function": "MultiClass",
                    "eval_metric": "MultiClass",
                    "auto_class_weights": "Balanced",
                }
            )
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    if algorithm == "xgboost":
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed in this environment. Install it before "
                "using --algorithm xgboost."
            )
        defaults = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "n_jobs": -1,
            "random_state": random_state,
        }
        if num_classes == 2:
            defaults.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            defaults.update(
                {
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "num_class": num_classes,
                }
            )
        defaults.update(params)
        return XGBClassifier(**defaults)

    defaults = {
        "n_estimators": 400,
        "n_jobs": -1,
        "class_weight": "balanced_subsample",
        "random_state": random_state,
    }
    defaults.update(params)
    return RandomForestClassifier(**defaults)


def fit_estimator(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
):
    fit_kwargs: dict[str, Any] = {}
    estimator_module = estimator.__class__.__module__

    if estimator_module.startswith("lightgbm"):
        fit_kwargs["categorical_feature"] = EMBER_CATEGORICAL_FEATURES
    if X_val is not None and y_val is not None and estimator_module.startswith("lightgbm"):
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "auc" if np.unique(y_train).shape[0] == 2 else "multi_logloss"
        if lgb_early_stopping is not None and lgb_log_evaluation is not None:
            fit_kwargs["callbacks"] = [
                lgb_early_stopping(stopping_rounds=50, verbose=False),
                lgb_log_evaluation(period=0),
            ]

    if X_val is not None and y_val is not None and estimator_module.startswith("catboost"):
        fit_kwargs["eval_set"] = (X_val, y_val)
        fit_kwargs.setdefault("verbose", False)

    if X_val is not None and y_val is not None and estimator_module.startswith("xgboost"):
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs.setdefault("verbose", False)

    estimator.fit(X_train, y_train, **fit_kwargs)
    return estimator


def evaluate_classifier(model, X_val: np.ndarray, y_val: np.ndarray) -> dict[str, float]:
    y_pred = prediction_labels(model, X_val)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro")),
    }
    if np.unique(y_val).shape[0] == 2:
        y_score = prediction_scores(model, X_val)
        if y_score is not None:
            if y_score.ndim == 2:
                y_score = y_score[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_val, y_score))
            metrics["average_precision"] = float(average_precision_score(y_val, y_score))
    return metrics


def save_model_artifact(
    model,
    model_path: str | Path,
    *,
    algorithm: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    num_features: int,
    classes: np.ndarray,
    booster_path: str | Path | None = None,
    native_model_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    algorithm = canonical_algorithm(algorithm)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    class_list = np.asarray(classes).astype(int).tolist()
    artifact = {
        "algorithm": algorithm,
        "model": model,
        "params": params,
        "metrics": metrics,
        "num_features": int(num_features),
        "classes": class_list,
    }
    dump(artifact, model_path)

    outputs = {"joblib_model_path": str(model_path)}
    requested_native_path = native_model_path if native_model_path is not None else booster_path
    resolved_native_path = resolve_native_model_path(model_path, algorithm, requested_native_path)
    if resolved_native_path is not None:
        save_native_model(model, resolved_native_path, algorithm)
        outputs["native_model_path"] = str(resolved_native_path)
        if algorithm == "lightgbm":
            outputs["lightgbm_booster_path"] = str(resolved_native_path)

    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path
        else model_path.with_name(f"{model_path.stem}_manifest.json")
    )
    manifest = {
        "algorithm": algorithm,
        "classes": class_list,
        "formats": {
            "joblib_model_path": outputs["joblib_model_path"],
        },
        "metrics": metrics,
        "num_features": int(num_features),
        "params": params,
    }
    if "native_model_path" in outputs:
        manifest["formats"]["native_model_path"] = outputs["native_model_path"]
    if "lightgbm_booster_path" in outputs:
        manifest["formats"]["lightgbm_booster_path"] = outputs["lightgbm_booster_path"]
    if extra_metadata:
        manifest["training"] = extra_metadata
    write_json(resolved_manifest_path, manifest)
    outputs["manifest_path"] = str(resolved_manifest_path)
    return outputs


def resolve_native_model_path(
    model_path: Path,
    algorithm: str,
    requested_path: str | Path | None,
) -> Path | None:
    if requested_path:
        return Path(requested_path)
    if algorithm == "lightgbm":
        return model_path.with_suffix(".txt")
    if algorithm == "catboost":
        return model_path.with_suffix(".cbm")
    if algorithm == "xgboost":
        return model_path.with_suffix(".json")
    return None


def save_native_model(model, native_path: str | Path, algorithm: str) -> None:
    algorithm = canonical_algorithm(algorithm)
    if algorithm == "lightgbm":
        save_lightgbm_booster(model, native_path)
        return
    native_path = Path(native_path)
    native_path.parent.mkdir(parents=True, exist_ok=True)
    if algorithm in {"catboost", "xgboost"} and hasattr(model, "save_model"):
        model.save_model(str(native_path))
        return
    raise ValueError(f"Native model export is not supported for algorithm '{algorithm}'.")


def save_lightgbm_booster(model, booster_path: str | Path) -> None:
    booster_path = Path(booster_path)
    booster_path.parent.mkdir(parents=True, exist_ok=True)

    booster = getattr(model, "booster_", None)
    if booster is None and is_lightgbm_booster(model):
        booster = model
    if booster is None or not hasattr(booster, "save_model"):
        raise ValueError("Expected a fitted LightGBM model with a native booster.")
    booster.save_model(str(booster_path))


def load_saved_model(
    model_path: str | Path,
    model_format: str = "auto",
) -> tuple[Any, str, dict[str, Any]]:
    model_path = Path(model_path)
    if model_format not in {"auto", "joblib", "lightgbm", "catboost", "xgboost"}:
        raise ValueError("--model-format must be one of: auto, joblib, lightgbm, catboost, xgboost.")

    if model_format == "lightgbm" or (
        model_format == "auto" and model_path.suffix.lower() in {".txt", ".lgbm", ".model"}
    ):
        if Booster is None:
            raise ImportError("lightgbm is not installed, so native booster files cannot be loaded.")
        return Booster(model_file=str(model_path)), "lightgbm", {"format": "lightgbm_booster"}

    if model_format == "catboost" or (
        model_format == "auto" and model_path.suffix.lower() == ".cbm"
    ):
        if CatBoostClassifier is None:
            raise ImportError("catboost is not installed, so native CatBoost files cannot be loaded.")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        return model, "catboost", {"format": "catboost_native"}

    if model_format == "xgboost" or (
        model_format == "auto" and model_path.suffix.lower() in {".json", ".ubj"}
    ):
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed, so native XGBoost files cannot be loaded.")
        model = XGBClassifier()
        model.load_model(str(model_path))
        return model, "xgboost", {"format": "xgboost_native"}

    artifact = load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact["model"], artifact.get("algorithm", "model"), artifact
    return artifact, artifact.__class__.__name__, {"format": "joblib_model"}


def prediction_labels(model, X: np.ndarray) -> np.ndarray:
    if is_lightgbm_booster(model):
        scores = np.asarray(model.predict(X))
        if scores.ndim == 1:
            return (scores >= 0.5).astype(np.int32)
        return np.argmax(scores, axis=1).astype(np.int32)
    labels = np.asarray(model.predict(X))
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.ravel()
    return labels.astype(np.int32, copy=False)


def prediction_scores(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    if is_lightgbm_booster(model):
        return np.asarray(model.predict(X))
    return None


def is_lightgbm_booster(model) -> bool:
    return model.__class__.__module__.startswith("lightgbm") and model.__class__.__name__ == "Booster"


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n")


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
