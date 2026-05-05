"""
Utilities for ML baseline training on the vectorized BODMAS dataset.
"""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

MPLCONFIG_DIR = Path(tempfile.gettempdir()) / "bodmas-mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import numpy as np
from numpy.lib import format as np_format
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
BODMAS_DEFAULT_NPZ = "bodmas.npz"
BODMAS_FEATURE_DIM = 2381
BODMAS_CATEGORICAL_FEATURES: list[int] = []


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
    npz_path = resolve_npz_path(data_dir)
    X_shape, X_dtype = npz_array_info(npz_path, "X")
    y = load_bodmas_labels(npz_path)
    if len(X_shape) != 2 or X_shape[0] != y.shape[0]:
        raise ValueError(
            f"Invalid BODMAS archive {npz_path}: X shape {X_shape} does not match y shape {y.shape}."
        )
    summary: dict[str, Any] = {
        "dataset_path": str(npz_path),
        "rows": int(y.shape[0]),
        "features": int(X_shape[1]) if len(X_shape) == 2 else None,
        "X_dtype": str(X_dtype),
        "y_dtype": str(y.dtype),
        "labels": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(y, return_counts=True))
        },
    }
    return summary


def load_labeled_training_data(
    data_dir: str | Path,
    max_train_samples: int | None = None,
    random_state: int = 0,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    X_all, train_indices, y = open_labeled_subset(
        data_dir,
        subset="train",
        max_samples=max_train_samples,
        random_state=random_state,
        test_fraction=test_fraction,
    )
    X = np.asarray(X_all[train_indices], dtype=np.float32)
    return X, y


def open_labeled_subset(
    data_dir: str | Path,
    subset: str = "train",
    max_samples: int | None = None,
    random_state: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if subset not in {"train", "validation", "test", "all"}:
        raise ValueError("--subset must be one of: train, validation, test, all.")

    X_all, y_all, _ = load_bodmas_arrays(data_dir)
    all_indices = np.arange(y_all.shape[0])
    train_pool_indices, test_indices, y_train_pool, y_test = split_indices_train_test(
        all_indices,
        y_all,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    _, val_indices, _, y_val = split_indices_train_validation(
        train_pool_indices,
        y_train_pool,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )

    if subset == "all":
        selected_indices = all_indices
        y = y_all
    elif subset == "test":
        selected_indices = test_indices
        y = y_test
    elif subset == "validation":
        selected_indices = val_indices
        y = y_val
    else:
        selected_indices = train_pool_indices
        y = y_train_pool

    if max_samples is not None:
        selected_indices, y = subsample_indices(
            selected_indices,
            y,
            max_samples=max_samples,
            random_state=random_state,
        )
    return X_all, selected_indices, y


def open_validation_split(
    data_dir: str | Path,
    validation_fraction: float = 0.1,
    max_eval_samples: int | None = None,
    random_state: int = 0,
    test_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return open_labeled_subset(
        data_dir,
        subset="validation",
        max_samples=max_eval_samples,
        random_state=random_state,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )


def load_bodmas_arrays(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, Path]:
    npz_path = resolve_npz_path(data_dir)
    with np.load(npz_path, allow_pickle=False) as data:
        if "X" not in data.files or "y" not in data.files:
            raise ValueError(f"{npz_path} must contain arrays named 'X' and 'y'.")
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int32).reshape(-1)
    validate_bodmas_arrays(X, y, npz_path)
    return X, y, npz_path


def load_bodmas_labels(npz_path: str | Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "y" not in data.files:
            raise ValueError(f"{npz_path} must contain an array named 'y'.")
        return np.asarray(data["y"], dtype=np.int32).reshape(-1)


def validate_bodmas_arrays(X: np.ndarray, y: np.ndarray, npz_path: Path) -> None:
    if X.ndim != 2:
        raise ValueError(f"Expected X in {npz_path} to be 2D, got shape {X.shape}.")
    if y.ndim != 1:
        raise ValueError(f"Expected y in {npz_path} to be 1D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"Row mismatch in {npz_path}: X has {X.shape[0]} rows but y has {y.shape[0]} labels."
        )
    if X.shape[1] != BODMAS_FEATURE_DIM:
        raise ValueError(
            f"Expected {BODMAS_FEATURE_DIM} BODMAS features, got {X.shape[1]} in {npz_path}."
        )
    if not np.all(np.isfinite(y)):
        raise ValueError(f"Labels in {npz_path} contain non-finite values.")


def available_subsets(data_dir: str | Path) -> list[str]:
    resolve_npz_path(data_dir)
    return ["train", "validation", "test", "all"]


def resolve_npz_path(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if path.is_file():
        if path.suffix != ".npz":
            raise ValueError(f"Expected a .npz file, got {path}.")
        return path
    if not path.is_dir():
        raise ValueError(f"Not a directory or .npz file: {path}")

    default_path = path / BODMAS_DEFAULT_NPZ
    if default_path.is_file():
        return default_path

    npz_files = sorted(path.glob("*.npz"))
    if len(npz_files) == 1:
        return npz_files[0]
    if not npz_files:
        raise ValueError(f"No .npz files found in {path}.")
    raise ValueError(
        f"Multiple .npz files found in {path}; pass the exact file path. "
        f"Candidates: {', '.join(str(item) for item in npz_files)}"
    )


def npz_array_info(npz_path: str | Path, key: str) -> tuple[tuple[int, ...], np.dtype]:
    member_name = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as zf:
        if member_name not in zf.namelist():
            raise ValueError(f"{npz_path} must contain an array named '{key}'.")
        with zf.open(member_name) as fp:
            version = np_format.read_magic(fp)
            if version == (1, 0):
                shape, _, dtype = np_format.read_array_header_1_0(fp)
            elif version == (2, 0):
                shape, _, dtype = np_format.read_array_header_2_0(fp)
            else:
                shape, _, dtype = np_format._read_array_header(fp, version)
    return tuple(int(item) for item in shape), np.dtype(dtype)


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


def split_indices_train_test(
    indices: np.ndarray,
    y: np.ndarray,
    test_fraction: float = 0.2,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be between 0 and 1.")
    return train_test_split(
        indices,
        y,
        test_size=test_fraction,
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
        fit_kwargs["categorical_feature"] = BODMAS_CATEGORICAL_FEATURES
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
