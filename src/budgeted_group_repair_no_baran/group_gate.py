"""Leakage-safe cell-query uplift models.

The gate learns two binary events for every ``(cell_id, query_id)`` pair:

``helpful``
    The executable LLM repair is correct while the Baran fallback is wrong.
``harmful``
    The executable LLM repair is wrong while the Baran fallback is correct.

Only a parsed, cell- and base-aware validated ``propose`` item is label-eligible.
The schema-level helper in this module assumes that value-specific dirty/base
checks have already been applied by the caller.  Every other outcome is the
Baran fallback and therefore receives two neutral labels.
Point predictions come from a model fitted on all training families.  Model
uncertainty is the sample standard deviation (``ddof=1``) of net-gain
predictions from leave-one-family-out replicas.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version as package_version
import math
from numbers import Real
import os
from pathlib import Path
from statistics import median
import time
from typing import Any, Literal, Mapping, Sequence, TypeAlias
import warnings

import numpy as np
import pandas as pd

Backend: TypeAlias = Literal[
    "catboost", "lightgbm", "xgboost", "tabiclv2", "tabpfn3"
]
FeatureInput: TypeAlias = Any

_BACKEND_PACKAGES = {
    "catboost": "catboost",
    "lightgbm": "lightgbm",
    "tabiclv2": "tabicl",
    "tabpfn3": "tabpfn",
    "xgboost": "xgboost",
}

_FORBIDDEN_FEATURES = frozenset(
    {
        "annotation",
        "baran_correct",
        "clean_value",
        "correct_repair",
        "error_type",
        "harmful",
        "helpful",
        "llm_correct",
        "llm_correct_in_query",
        "missing_value",
        "right_value",
        "tuple_pairs",
    }
)


@dataclass(frozen=True)
class UpliftTargets:
    """Immutable helpful/harmful labels for a calibration table."""

    helpful: tuple[int, ...]
    harmful: tuple[int, ...]

    def as_dict(self) -> dict[str, list[int]]:
        return {"helpful": list(self.helpful), "harmful": list(self.harmful)}


@dataclass(frozen=True)
class GroupGatePrediction:
    """Prediction for one cell-query pair."""

    q_helpful: float
    q_harmful: float
    net_gain: float
    sigma: float
    conservative_uplift: float

    @property
    def ell(self) -> float:
        """Mathematical alias for the uncertainty-penalized routing score."""

        return self.conservative_uplift

    def as_dict(self) -> dict[str, float]:
        return {
            "q_helpful": float(self.q_helpful),
            "q_harmful": float(self.q_harmful),
            "net_gain": float(self.net_gain),
            "sigma": float(self.sigma),
            "conservative_uplift": float(self.conservative_uplift),
        }

    def to_dict(self) -> dict[str, float]:
        """Compatibility alias for JSON-oriented callers."""

        return self.as_dict()


@dataclass(frozen=True)
class _ColumnSpec:
    name: str
    kind: Literal["numeric", "categorical"]
    numeric_fill: float = 0.0
    categories: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "numeric_fill": float(self.numeric_fill),
            "categories": list(self.categories),
        }


class PairFeatureEncoder:
    """Deterministic train-only encoder for cell-query feature tables."""

    _UNKNOWN_CATEGORY = -1.0
    _MISSING_CATEGORY = -2.0

    def __init__(self) -> None:
        self._columns: tuple[_ColumnSpec, ...] = ()
        self._fitted = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return tuple(column.name for column in self._columns)

    def fit(self, features: FeatureInput) -> "PairFeatureEncoder":
        records, names = _as_records(features)
        if not records:
            raise ValueError("features must contain at least one row")
        _reject_forbidden_features(names)
        if not names:
            names = ["__bias__"]
            records = [{"__bias__": 0.0} for _ in records]

        columns: list[_ColumnSpec] = []
        for name in names:
            values = [record.get(name) for record in records]
            observed = [value for value in values if not _is_missing(value)]
            numeric = bool(observed) and all(_is_numeric(value) for value in observed)
            if numeric:
                finite = [float(value) for value in observed if _is_finite_number(value)]
                fill = float(median(finite)) if finite else 0.0
                columns.append(_ColumnSpec(name=name, kind="numeric", numeric_fill=fill))
            else:
                vocabulary = tuple(sorted({_category_token(value) for value in observed}))
                columns.append(
                    _ColumnSpec(name=name, kind="categorical", categories=vocabulary)
                )
        self._columns = tuple(columns)
        self._fitted = True
        return self

    def transform(self, features: FeatureInput) -> list[list[float]]:
        self._require_fitted()
        records, names = _as_records(features)
        _reject_forbidden_features(names)
        category_maps = {
            column.name: {value: index for index, value in enumerate(column.categories)}
            for column in self._columns
            if column.kind == "categorical"
        }
        matrix: list[list[float]] = []
        for record in records:
            row: list[float] = []
            for column in self._columns:
                value = record.get(column.name)
                if column.kind == "numeric":
                    row.append(
                        float(value)
                        if not _is_missing(value) and _is_finite_number(value)
                        else column.numeric_fill
                    )
                elif _is_missing(value):
                    row.append(self._MISSING_CATEGORY)
                else:
                    row.append(
                        float(
                            category_maps[column.name].get(
                                _category_token(value), self._UNKNOWN_CATEGORY
                            )
                        )
                    )
            matrix.append(row)
        return matrix

    def fit_transform(self, features: FeatureInput) -> list[list[float]]:
        return self.fit(features).transform(features)

    def as_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {"columns": [column.as_dict() for column in self._columns]}

    def to_dict(self) -> dict[str, object]:
        return self.as_dict()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("PairFeatureEncoder has not been fitted")


class CatBoostFeatureEncoder:
    """Train-only schema adapter that preserves native categorical values."""

    _MISSING_CATEGORY = "__BGR_MISSING_CATEGORY__"

    def __init__(self) -> None:
        self._columns: tuple[_ColumnSpec, ...] = ()
        self._fitted = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return tuple(column.name for column in self._columns)

    @property
    def categorical_feature_indices(self) -> tuple[int, ...]:
        self._require_fitted()
        return tuple(
            index
            for index, column in enumerate(self._columns)
            if column.kind == "categorical"
        )

    def fit(self, features: FeatureInput) -> "CatBoostFeatureEncoder":
        records, names = _as_records(features)
        if not records:
            raise ValueError("features must contain at least one row")
        _reject_forbidden_features(names)
        if not names:
            names = ["__bias__"]
            records = [{"__bias__": 0.0} for _ in records]

        columns: list[_ColumnSpec] = []
        for name in names:
            values = [record.get(name) for record in records]
            observed = [value for value in values if not _is_missing(value)]
            numeric = bool(observed) and all(_is_numeric(value) for value in observed)
            if numeric:
                finite = [float(value) for value in observed if _is_finite_number(value)]
                fill = float(median(finite)) if finite else 0.0
                columns.append(_ColumnSpec(name=name, kind="numeric", numeric_fill=fill))
            else:
                vocabulary = tuple(sorted({_category_token(value) for value in observed}))
                columns.append(
                    _ColumnSpec(name=name, kind="categorical", categories=vocabulary)
                )
        self._columns = tuple(columns)
        self._fitted = True
        return self

    def transform(self, features: FeatureInput) -> list[list[object]]:
        self._require_fitted()
        records, names = _as_records(features)
        _reject_forbidden_features(names)
        matrix: list[list[object]] = []
        for record in records:
            row: list[object] = []
            for column in self._columns:
                value = record.get(column.name)
                if column.kind == "numeric":
                    row.append(
                        float(value)
                        if not _is_missing(value) and _is_finite_number(value)
                        else column.numeric_fill
                    )
                elif _is_missing(value):
                    row.append(self._MISSING_CATEGORY)
                else:
                    row.append(_category_token(value))
            matrix.append(row)
        return matrix

    def fit_transform(self, features: FeatureInput) -> list[list[object]]:
        return self.fit(features).transform(features)

    def as_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "kind": "catboost_native_categorical",
            "missing_category": self._MISSING_CATEGORY,
            "categorical_feature_indices": list(self.categorical_feature_indices),
            "columns": [column.as_dict() for column in self._columns],
        }

    def to_dict(self) -> dict[str, object]:
        return self.as_dict()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("CatBoostFeatureEncoder has not been fitted")


class FoundationFeatureEncoder:
    """Train-only DataFrame adapter preserving native categorical semantics."""

    _MISSING_CATEGORY = "__BGR_MISSING_CATEGORY__"
    _UNKNOWN_CATEGORY = "__BGR_UNKNOWN_CATEGORY__"

    def __init__(self) -> None:
        self._columns: tuple[_ColumnSpec, ...] = ()
        self._fitted = False

    @property
    def feature_names(self) -> tuple[str, ...]:
        self._require_fitted()
        return tuple(column.name for column in self._columns)

    @property
    def categorical_feature_indices(self) -> tuple[int, ...]:
        self._require_fitted()
        return tuple(
            index
            for index, column in enumerate(self._columns)
            if column.kind == "categorical"
        )

    def fit(self, features: FeatureInput) -> "FoundationFeatureEncoder":
        records, names = _as_records(features)
        if not records:
            raise ValueError("features must contain at least one row")
        _reject_forbidden_features(names)
        if not names:
            names = ["__bias__"]
            records = [{"__bias__": 0.0} for _ in records]

        columns: list[_ColumnSpec] = []
        for name in names:
            values = [record.get(name) for record in records]
            observed = [value for value in values if not _is_missing(value)]
            numeric = bool(observed) and all(_is_numeric(value) for value in observed)
            if numeric:
                finite = [float(value) for value in observed if _is_finite_number(value)]
                fill = float(median(finite)) if finite else 0.0
                columns.append(_ColumnSpec(name=name, kind="numeric", numeric_fill=fill))
            else:
                vocabulary = tuple(sorted({_category_token(value) for value in observed}))
                columns.append(
                    _ColumnSpec(name=name, kind="categorical", categories=vocabulary)
                )
        self._columns = tuple(columns)
        self._fitted = True
        return self

    def transform(self, features: FeatureInput) -> pd.DataFrame:
        self._require_fitted()
        records, names = _as_records(features)
        _reject_forbidden_features(names)
        transformed: dict[str, object] = {}
        for column in self._columns:
            values = [record.get(column.name) for record in records]
            if column.kind == "numeric":
                transformed[column.name] = pd.Series(
                    [
                        float(value)
                        if not _is_missing(value) and _is_finite_number(value)
                        else column.numeric_fill
                        for value in values
                    ],
                    dtype="float64",
                )
                continue
            vocabulary = set(column.categories)
            category_values: list[str] = []
            for value in values:
                if _is_missing(value):
                    category_values.append(self._MISSING_CATEGORY)
                    continue
                token = _category_token(value)
                category_values.append(
                    token if token in vocabulary else self._UNKNOWN_CATEGORY
                )
            categories = (
                self._MISSING_CATEGORY,
                self._UNKNOWN_CATEGORY,
                *column.categories,
            )
            transformed[column.name] = pd.Series(
                pd.Categorical(category_values, categories=categories),
                dtype=pd.CategoricalDtype(categories=categories),
            )
        return pd.DataFrame(transformed, columns=list(self.feature_names))

    def fit_transform(self, features: FeatureInput) -> pd.DataFrame:
        return self.fit(features).transform(features)

    def as_dict(self) -> dict[str, object]:
        self._require_fitted()
        return {
            "kind": "foundation_native_categorical",
            "numeric_missing_strategy": "train_median",
            "missing_category": self._MISSING_CATEGORY,
            "unknown_category": self._UNKNOWN_CATEGORY,
            "categorical_feature_indices": list(self.categorical_feature_indices),
            "columns": [column.as_dict() for column in self._columns],
        }

    def to_dict(self) -> dict[str, object]:
        return self.as_dict()

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("FoundationFeatureEncoder has not been fitted")


class TabICLv2ClassifierAdapter:
    """Strict, provenance-rich wrapper around ``tabicl.TabICLClassifier``."""

    CHECKPOINT_FILENAME = "tabicl-classifier-v2-20260212.ckpt"
    _CONFIG_KEYS = frozenset(
        {
            "allow_auto_download",
            "batch_size",
            "checkpoint_filename",
            "checkpoint_path",
            "checkpoint_path_env",
            "checkpoint_sha256",
            "device",
            "kv_cache",
            "n_estimators",
            "offload_mode",
            "random_state",
            "use_amp",
            "use_fa3",
        }
    )

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        raw = dict(config or {})
        unknown = sorted(set(raw) - self._CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unsupported TabICLv2 configuration keys: {unknown}")
        checkpoint_filename = str(
            raw.get("checkpoint_filename", self.CHECKPOINT_FILENAME)
        )
        if checkpoint_filename != self.CHECKPOINT_FILENAME:
            raise ValueError(
                "TabICLv2 checkpoint_filename must be " + self.CHECKPOINT_FILENAME
            )
        path_value = raw.get("checkpoint_path")
        path_env = str(raw.get("checkpoint_path_env", "BGR_TABICL_MODEL_PATH"))
        if path_value in {None, ""}:
            path_value = os.environ.get(path_env, "")
        allow_auto_download = bool(raw.get("allow_auto_download", False))
        checkpoint_path = (
            Path(str(path_value)).expanduser().resolve() if path_value else None
        )
        if checkpoint_path is None and not allow_auto_download:
            raise FileNotFoundError(
                f"TabICLv2 checkpoint path is unset; set {path_env} or checkpoint_path"
            )
        if (
            checkpoint_path is not None
            and not checkpoint_path.is_file()
            and not allow_auto_download
        ):
            raise FileNotFoundError(f"TabICLv2 checkpoint is missing: {checkpoint_path}")
        if checkpoint_path is not None and checkpoint_path.name != checkpoint_filename:
            raise ValueError(
                f"TabICLv2 checkpoint basename must be {checkpoint_filename!r}"
            )
        actual_sha = (
            _sha256_file(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.is_file()
            else ""
        )
        expected_sha = str(raw.get("checkpoint_sha256", "")).strip().lower()
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"TabICLv2 checkpoint SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )

        self._config = {
            "n_estimators": int(raw.get("n_estimators", 8)),
            "batch_size": int(raw.get("batch_size", 8)),
            "kv_cache": raw.get("kv_cache", False),
            "model_path": None if checkpoint_path is None else str(checkpoint_path),
            "allow_auto_download": allow_auto_download,
            "checkpoint_version": checkpoint_filename,
            "device": str(raw.get("device", "cuda")),
            "use_amp": raw.get("use_amp", "auto"),
            "use_fa3": raw.get("use_fa3", "auto"),
            "offload_mode": raw.get("offload_mode", "auto"),
            "random_state": int(raw.get("random_state", random_state)),
        }
        if int(self._config["n_estimators"]) <= 0 or int(self._config["batch_size"]) <= 0:
            raise ValueError("TabICLv2 n_estimators and batch_size must be positive")
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha256 = actual_sha
        self._model: object | None = None
        self.classes_: np.ndarray = np.asarray([], dtype=int)
        self._fit_seconds = 0.0
        self._predict_seconds = 0.0
        self._peak_ram_bytes = 0
        self._peak_vram_bytes = 0
        self._runtime_environment: dict[str, object] = {}

    def fit(
        self, features: pd.DataFrame, labels: Sequence[int]
    ) -> "TabICLv2ClassifierAdapter":
        if not isinstance(features, pd.DataFrame):
            raise TypeError("TabICLv2 requires a pandas DataFrame feature matrix")
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "TabICLv2 backend requested; install tabicl==2.1.1 in the foundation environment"
            ) from exc
        started = time.perf_counter()
        self._reset_peak_vram()
        model = TabICLClassifier(**self._config)
        model.fit(features, list(int(value) for value in labels))
        self._fit_seconds = float(time.perf_counter() - started)
        self._model = model
        self.classes_ = np.asarray(model.classes_)
        self._capture_resource_peaks()
        self._runtime_environment = self._capture_runtime_environment(model)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabICLv2 adapter has not been fitted")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("TabICLv2 requires a pandas DataFrame feature matrix")
        started = time.perf_counter()
        self._reset_peak_vram()
        probabilities = np.asarray(self._model.predict_proba(features), dtype=float)
        self._predict_seconds += float(time.perf_counter() - started)
        self._capture_resource_peaks()
        if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
            raise RuntimeError("TabICLv2 predict_proba returned an invalid shape")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("TabICLv2 predict_proba returned non-finite values")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError("TabICLv2 predict_proba returned values outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("TabICLv2 probability rows do not sum to one")
        return probabilities

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "classifier",
            "class_name": type(self).__name__,
            "package": "tabicl",
            "package_version": package_version("tabicl"),
            "checkpoint_filename": self.CHECKPOINT_FILENAME,
            "checkpoint_path": (
                None if self._checkpoint_path is None else str(self._checkpoint_path)
            ),
            "checkpoint_sha256": self._checkpoint_sha256,
            "parameters": dict(self._config),
            "classes": [int(value) for value in self.classes_],
            "fit_seconds": self._fit_seconds,
            "predict_seconds": self._predict_seconds,
            "peak_ram_bytes": self._peak_ram_bytes,
            "peak_vram_bytes": self._peak_vram_bytes,
            "runtime_environment": dict(self._runtime_environment),
        }

    def _capture_runtime_environment(self, model: object) -> dict[str, object]:
        resolved_amp: object = getattr(
            model, "use_amp_", getattr(model, "use_amp", "unknown")
        )
        resolved_fa3: object = getattr(
            model, "use_fa3_", getattr(model, "use_fa3", "unknown")
        )
        resolver = getattr(model, "_resolve_amp_fa3", None)
        if callable(resolver):
            try:
                resolved_amp, resolved_fa3 = resolver()
            except Exception:
                pass
        result: dict[str, object] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "requested_device": self._config["device"],
            "requested_use_amp": self._config["use_amp"],
            "requested_use_fa3": self._config["use_fa3"],
            "effective_use_amp": resolved_amp,
            "effective_use_fa3": resolved_fa3,
        }
        try:
            import torch

            result["torch_version"] = str(torch.__version__)
            result["torch_cuda_version"] = str(torch.version.cuda)
            result["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                result["logical_cuda_device"] = int(torch.cuda.current_device())
                result["gpu_name"] = str(torch.cuda.get_device_name())
                result["gpu_capability"] = list(torch.cuda.get_device_capability())
                parameter_dtype = "unknown"
                fitted_model = getattr(model, "model_", None)
                if fitted_model is not None:
                    try:
                        parameter_dtype = str(next(fitted_model.parameters()).dtype)
                    except Exception:
                        pass
                result["model_parameter_dtype"] = parameter_dtype
                result["effective_dtype"] = (
                    str(torch.get_autocast_dtype("cuda"))
                    if resolved_amp is True
                    else parameter_dtype
                )
        except Exception:
            result["cuda_available"] = False
        return result

    @staticmethod
    def _reset_peak_vram() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    def _capture_resource_peaks(self) -> None:
        try:
            import psutil

            self._peak_ram_bytes = max(
                self._peak_ram_bytes,
                int(psutil.Process().memory_info().rss),
            )
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                self._peak_vram_bytes = max(
                    self._peak_vram_bytes,
                    int(torch.cuda.max_memory_allocated()),
                )
        except Exception:
            pass


class TabPFN3ClassifierAdapter:
    """Strict local-checkpoint wrapper around ``tabpfn.TabPFNClassifier``."""

    CHECKPOINT_FILENAME = "tabpfn-v3-classifier-v3_20260506_ood.ckpt"
    _CONFIG_KEYS = frozenset(
        {
            "allow_auto_download",
            "auto_scale_n_estimators",
            "average_before_softmax",
            "balance_probabilities",
            "checkpoint_filename",
            "checkpoint_path",
            "checkpoint_path_env",
            "checkpoint_sha256",
            "device",
            "fit_mode",
            "ignore_pretraining_limits",
            "inference_precision",
            "memory_saving_mode",
            "n_estimators",
            "n_preprocessing_jobs",
            "random_state",
            "show_progress_bar",
            "softmax_temperature",
        }
    )

    def __init__(
        self,
        config: Mapping[str, object] | None = None,
        *,
        random_state: int = 42,
    ) -> None:
        raw = dict(config or {})
        unknown = sorted(set(raw) - self._CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unsupported TabPFN-3 configuration keys: {unknown}")
        checkpoint_filename = str(
            raw.get("checkpoint_filename", self.CHECKPOINT_FILENAME)
        )
        if checkpoint_filename != self.CHECKPOINT_FILENAME:
            raise ValueError(
                "TabPFN-3 checkpoint_filename must be " + self.CHECKPOINT_FILENAME
            )
        if bool(raw.get("allow_auto_download", False)):
            raise ValueError("TabPFN-3 formal backend forbids automatic checkpoint download")
        path_value = raw.get("checkpoint_path")
        path_env = str(raw.get("checkpoint_path_env", "BGR_TABPFN_MODEL_PATH"))
        if path_value in {None, ""}:
            path_value = os.environ.get(path_env, "")
        if not path_value:
            raise FileNotFoundError(
                f"TabPFN-3 checkpoint path is unset; set {path_env} or checkpoint_path"
            )
        checkpoint_path = Path(str(path_value)).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"TabPFN-3 checkpoint is missing: {checkpoint_path}")
        if checkpoint_path.name != checkpoint_filename:
            raise ValueError(
                f"TabPFN-3 checkpoint basename must be {checkpoint_filename!r}"
            )
        actual_sha = _sha256_file(checkpoint_path)
        expected_sha = str(raw.get("checkpoint_sha256", "")).strip().lower()
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"TabPFN-3 checkpoint SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
            )

        self._constructor_config: dict[str, object] = {
            "model_path": str(checkpoint_path),
            "n_estimators": int(raw.get("n_estimators", 8)),
            "auto_scale_n_estimators": bool(
                raw.get("auto_scale_n_estimators", False)
            ),
            "softmax_temperature": float(raw.get("softmax_temperature", 0.9)),
            "balance_probabilities": bool(raw.get("balance_probabilities", False)),
            "average_before_softmax": bool(
                raw.get("average_before_softmax", False)
            ),
            "device": str(raw.get("device", "cuda")),
            "ignore_pretraining_limits": bool(
                raw.get("ignore_pretraining_limits", False)
            ),
            "inference_precision": raw.get("inference_precision", "auto"),
            "fit_mode": str(raw.get("fit_mode", "fit_preprocessors")),
            "memory_saving_mode": raw.get("memory_saving_mode", "auto"),
            "random_state": int(raw.get("random_state", random_state)),
            "n_preprocessing_jobs": int(raw.get("n_preprocessing_jobs", 1)),
            "show_progress_bar": bool(raw.get("show_progress_bar", False)),
        }
        if int(self._constructor_config["n_estimators"]) <= 0:
            raise ValueError("TabPFN-3 n_estimators must be positive")
        if bool(self._constructor_config["auto_scale_n_estimators"]):
            raise ValueError(
                "TabPFN-3 formal backend requires auto_scale_n_estimators=false"
            )
        if int(self._constructor_config["n_preprocessing_jobs"]) != 1:
            raise ValueError("TabPFN-3 formal backend requires n_preprocessing_jobs=1")
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha256 = actual_sha
        self._model: object | None = None
        self.classes_: np.ndarray = np.asarray([], dtype=int)
        self._categorical_feature_indices: tuple[int, ...] = ()
        self._fit_seconds = 0.0
        self._predict_seconds = 0.0
        self._peak_ram_bytes = 0
        self._peak_vram_bytes = 0
        self._runtime_environment: dict[str, object] = {}

    def fit(
        self, features: pd.DataFrame, labels: Sequence[int]
    ) -> "TabPFN3ClassifierAdapter":
        if not isinstance(features, pd.DataFrame):
            raise TypeError("TabPFN-3 requires a pandas DataFrame feature matrix")
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "TabPFN-3 backend requested; install tabpfn==8.1.0 in the foundation environment"
            ) from exc
        categorical_indices = tuple(
            index
            for index, dtype in enumerate(features.dtypes)
            if isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        self._categorical_feature_indices = categorical_indices
        constructor_config = dict(self._constructor_config)
        constructor_config["categorical_features_indices"] = list(
            categorical_indices
        )
        started = time.perf_counter()
        self._reset_peak_vram()
        model = TabPFNClassifier(**constructor_config)
        model.fit(features, list(int(value) for value in labels))
        self._fit_seconds = float(time.perf_counter() - started)
        self._model = model
        self.classes_ = np.asarray(model.classes_)
        self._capture_resource_peaks()
        self._runtime_environment = self._capture_runtime_environment(model)
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TabPFN-3 adapter has not been fitted")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("TabPFN-3 requires a pandas DataFrame feature matrix")
        started = time.perf_counter()
        self._reset_peak_vram()
        probabilities = np.asarray(self._model.predict_proba(features), dtype=float)
        self._predict_seconds += float(time.perf_counter() - started)
        self._capture_resource_peaks()
        if probabilities.ndim != 2 or probabilities.shape[0] != len(features):
            raise RuntimeError("TabPFN-3 predict_proba returned an invalid shape")
        if not np.isfinite(probabilities).all():
            raise RuntimeError("TabPFN-3 predict_proba returned non-finite values")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise RuntimeError("TabPFN-3 predict_proba returned values outside [0, 1]")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError("TabPFN-3 probability rows do not sum to one")
        return probabilities

    def metadata(self) -> dict[str, object]:
        parameters = dict(self._constructor_config)
        parameters.update(
            {
                "allow_auto_download": False,
                "categorical_features_indices": list(
                    self._categorical_feature_indices
                ),
            }
        )
        return {
            "kind": "classifier",
            "class_name": type(self).__name__,
            "package": "tabpfn",
            "package_version": package_version("tabpfn"),
            "checkpoint_filename": self.CHECKPOINT_FILENAME,
            "checkpoint_path": str(self._checkpoint_path),
            "checkpoint_sha256": self._checkpoint_sha256,
            "parameters": parameters,
            "classes": [int(value) for value in self.classes_],
            "fit_seconds": self._fit_seconds,
            "predict_seconds": self._predict_seconds,
            "peak_ram_bytes": self._peak_ram_bytes,
            "peak_vram_bytes": self._peak_vram_bytes,
            "runtime_environment": dict(self._runtime_environment),
        }

    def _capture_runtime_environment(self, model: object) -> dict[str, object]:
        use_autocast = bool(getattr(model, "use_autocast_", False))
        forced_dtype = getattr(model, "forced_inference_dtype_", None)
        result: dict[str, object] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "requested_device": self._constructor_config["device"],
            "requested_inference_precision": self._constructor_config[
                "inference_precision"
            ],
            "effective_use_autocast": use_autocast,
            "forced_inference_dtype": str(forced_dtype),
            "effective_inference_precision": (
                "autocast" if use_autocast else str(forced_dtype or "float32")
            ),
            "effective_n_estimators": int(
                getattr(
                    model,
                    "n_estimators_",
                    self._constructor_config["n_estimators"],
                )
            ),
        }
        try:
            import torch

            result["torch_version"] = str(torch.__version__)
            result["torch_cuda_version"] = str(torch.version.cuda)
            result["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                result["logical_cuda_device"] = int(torch.cuda.current_device())
                result["gpu_name"] = str(torch.cuda.get_device_name())
                result["gpu_capability"] = list(torch.cuda.get_device_capability())
                result["effective_dtype"] = (
                    str(torch.get_autocast_dtype("cuda"))
                    if use_autocast
                    else str(forced_dtype or torch.float32)
                )
        except Exception:
            result["cuda_available"] = False
        return result

    @staticmethod
    def _reset_peak_vram() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            return

    def _capture_resource_peaks(self) -> None:
        try:
            import psutil

            self._peak_ram_bytes = max(
                self._peak_ram_bytes,
                int(psutil.Process().memory_info().rss),
            )
        except Exception:
            pass
        try:
            import torch

            if torch.cuda.is_available():
                self._peak_vram_bytes = max(
                    self._peak_vram_bytes,
                    int(torch.cuda.max_memory_allocated()),
                )
        except Exception:
            pass


class _ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = _clip_probability(probability)

    def predict_positive(self, row_count: int) -> list[float]:
        return [self.probability] * row_count

    def as_dict(self) -> dict[str, object]:
        return {"kind": "constant", "probability": float(self.probability)}


@dataclass
class _FittedHeads:
    family_left_out: str | None
    rows: int
    encoder: PairFeatureEncoder | CatBoostFeatureEncoder | FoundationFeatureEncoder
    helpful_model: object
    harmful_model: object


def build_uplift_targets(
    baran_correct: Sequence[bool | int],
    llm_correct: Sequence[bool | int],
    executable_use_llm: Sequence[bool | int],
) -> UpliftTargets:
    """Construct executable-upgrade labels.

    ``executable_use_llm`` must already include parser and basic-validation
    validity.  A false flag means the online path falls back to Baran, so both
    labels are zero even if a non-executable raw string happened to match the
    reference value.
    """

    if not (
        len(baran_correct) == len(llm_correct) == len(executable_use_llm)
    ):
        raise ValueError(
            "baran_correct, llm_correct, and executable_use_llm must have equal length"
        )
    helpful: list[int] = []
    harmful: list[int] = []
    for baran, llm, executable in zip(
        baran_correct, llm_correct, executable_use_llm
    ):
        use_llm = bool(executable)
        baran_ok = bool(baran)
        llm_ok = bool(llm)
        helpful.append(int(use_llm and llm_ok and not baran_ok))
        harmful.append(int(use_llm and baran_ok and not llm_ok))
    return UpliftTargets(tuple(helpful), tuple(harmful))


def executable_use_llm(item: Mapping[str, object] | object | None) -> bool:
    """Return whether a parsed item passes the schema-level proposal check.

    Parsers may pass an immutable item (already schema-valid) or a ledger
    mapping.  A mapping can explicitly set ``base_valid``/``verifier_valid``
    false to turn the item into a neutral fallback before labels are built.
    This helper does not compare the repair with a cell's dirty or base value;
    production label construction performs those value-specific checks before
    calling :func:`build_uplift_targets`.
    """

    if item is None:
        return False
    decision = str(_field(item, "decision", "")).strip().lower()
    if decision != "propose":
        return False
    status = _field(item, "parse_status", None)
    if status is not None and not str(status).startswith("ok"):
        return False
    if _field(item, "base_valid", True) is False:
        return False
    if _field(item, "verifier_valid", True) is False:
        return False
    repair = _field(item, "repair", _field(item, "prediction", None))
    return repair is not None


class GroupUpliftGate:
    """Shared tabular helpful/harmful model for cell-query pairs."""

    def __init__(
        self,
        backend: Backend,
        *,
        rho: float = 1.0,
        gamma: float = 1.0,
        random_state: int = 42,
        backend_config: Mapping[str, object] | None = None,
    ) -> None:
        if backend not in {
            "catboost",
            "lightgbm",
            "xgboost",
            "tabiclv2",
            "tabpfn3",
        }:
            raise ValueError(
                "backend must be 'catboost', 'lightgbm', 'xgboost', "
                "'tabiclv2', or 'tabpfn3'"
            )
        _validate_penalty("rho", rho)
        _validate_penalty("gamma", gamma)
        self.backend = backend
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.random_state = int(random_state)
        self.backend_config = dict(backend_config or {})
        self._full: _FittedHeads | None = None
        self._replicas: tuple[_FittedHeads, ...] = ()
        self._training_summary: dict[str, object] = {}

    @staticmethod
    def targets(
        baran_correct: Sequence[bool | int],
        llm_correct: Sequence[bool | int],
        executable_use_llm_flags: Sequence[bool | int],
    ) -> dict[str, list[int]]:
        return build_uplift_targets(
            baran_correct, llm_correct, executable_use_llm_flags
        ).as_dict()

    def fit(
        self,
        features: FeatureInput,
        baran_correct: Sequence[bool | int],
        llm_correct: Sequence[bool | int],
        executable_use_llm_flags: Sequence[bool | int],
        families: Sequence[str],
    ) -> "GroupUpliftGate":
        """Fit the full point model and every non-empty LOFO replica."""

        records, names = _as_records(features)
        _reject_forbidden_features(names)
        targets = build_uplift_targets(
            baran_correct, llm_correct, executable_use_llm_flags
        )
        row_count = len(records)
        if row_count == 0:
            raise ValueError("calibration data must contain at least one row")
        if len(targets.helpful) != row_count or len(families) != row_count:
            raise ValueError("features, targets, and families must have equal length")
        normalized_families = tuple(str(family) for family in families)
        if any(not family for family in normalized_families):
            raise ValueError("families must contain non-empty identifiers")

        self._full = self._fit_heads(
            records,
            targets.helpful,
            targets.harmful,
            family_left_out=None,
            feature_names=names,
        )
        replicas: list[_FittedHeads] = []
        for family in sorted(set(normalized_families)):
            indices = [
                index
                for index, row_family in enumerate(normalized_families)
                if row_family != family
            ]
            if not indices:
                continue
            replicas.append(
                self._fit_heads(
                    [records[index] for index in indices],
                    tuple(targets.helpful[index] for index in indices),
                    tuple(targets.harmful[index] for index in indices),
                    family_left_out=family,
                    feature_names=names,
                )
            )
        self._replicas = tuple(replicas)
        self._training_summary = {
            "rows": row_count,
            "families": sorted(set(normalized_families)),
            "helpful_positive_rate": float(sum(targets.helpful) / row_count),
            "harmful_positive_rate": float(sum(targets.harmful) / row_count),
            "lofo_replicas": len(self._replicas),
            "uncertainty_ddof": 1,
        }
        return self

    def predict(self, features: FeatureInput) -> list[GroupGatePrediction]:
        self._require_fitted()
        assert self._full is not None
        full_helpful, full_harmful = self._predict_heads(self._full, features)
        replica_net: list[list[float]] = []
        for replica in self._replicas:
            helpful, harmful = self._predict_heads(replica, features)
            replica_net.append(
                [
                    float(q_helpful - self.rho * q_harmful)
                    for q_helpful, q_harmful in zip(helpful, harmful)
                ]
            )

        predictions: list[GroupGatePrediction] = []
        for index, (q_helpful, q_harmful) in enumerate(
            zip(full_helpful, full_harmful)
        ):
            net_gain = float(q_helpful - self.rho * q_harmful)
            replicate_values = [values[index] for values in replica_net]
            sigma = _sample_standard_deviation(replicate_values)
            ell = max(0.0, net_gain - self.gamma * sigma)
            predictions.append(
                GroupGatePrediction(
                    q_helpful=float(q_helpful),
                    q_harmful=float(q_harmful),
                    net_gain=net_gain,
                    sigma=sigma,
                    conservative_uplift=float(ell),
                )
            )
        return predictions

    def predict_dicts(self, features: FeatureInput) -> list[dict[str, float]]:
        return [prediction.as_dict() for prediction in self.predict(features)]

    def metadata(self) -> dict[str, object]:
        self._require_fitted()
        assert self._full is not None
        return {
            "backend": self.backend,
            "backend_version": package_version(_BACKEND_PACKAGES[self.backend]),
            "rho": self.rho,
            "gamma": self.gamma,
            "random_state": self.random_state,
            "training": dict(self._training_summary),
            "full": _heads_metadata(self._full),
            "lofo": [_heads_metadata(replica) for replica in self._replicas],
        }

    def to_metadata(self) -> dict[str, object]:
        return self.metadata()

    def _fit_heads(
        self,
        features: FeatureInput,
        helpful: Sequence[int],
        harmful: Sequence[int],
        *,
        family_left_out: str | None,
        feature_names: Sequence[str],
    ) -> _FittedHeads:
        encoder: PairFeatureEncoder | CatBoostFeatureEncoder | FoundationFeatureEncoder
        if self.backend == "catboost":
            encoder = CatBoostFeatureEncoder()
        elif self.backend in {"tabiclv2", "tabpfn3"}:
            encoder = FoundationFeatureEncoder()
        else:
            encoder = PairFeatureEncoder()
        encoder_input = (
            pd.DataFrame(features, columns=list(feature_names))
            if self.backend in {"tabiclv2", "tabpfn3"}
            else features
        )
        matrix = encoder.fit_transform(encoder_input)
        return _FittedHeads(
            family_left_out=family_left_out,
            rows=len(matrix),
            encoder=encoder,
            helpful_model=self._fit_binary_head(
                matrix, helpful, encoder.categorical_feature_indices
                if isinstance(encoder, CatBoostFeatureEncoder)
                else ()
            ),
            harmful_model=self._fit_binary_head(
                matrix, harmful, encoder.categorical_feature_indices
                if isinstance(encoder, CatBoostFeatureEncoder)
                else ()
            ),
        )

    def _fit_binary_head(
        self,
        matrix: FeatureInput,
        labels: Sequence[int],
        categorical_feature_indices: Sequence[int] = (),
    ) -> object:
        values = tuple(int(label) for label in labels)
        if not values or any(label not in {0, 1} for label in values):
            raise ValueError("binary labels must be a non-empty sequence of 0/1 values")
        if len(set(values)) == 1:
            return _ConstantProbabilityModel(float(values[0]))
        model = self._new_backend_model()
        if self.backend == "catboost":
            model.fit(
                matrix,
                list(values),
                cat_features=list(categorical_feature_indices),
            )
        else:
            model.fit(matrix, list(values))
        return model

    def _new_backend_model(self) -> object:
        if self.backend == "tabiclv2":
            return TabICLv2ClassifierAdapter(
                self.backend_config,
                random_state=self.random_state,
            )
        if self.backend == "tabpfn3":
            return TabPFN3ClassifierAdapter(
                self.backend_config,
                random_state=self.random_state,
            )
        if self.backend == "catboost":
            try:
                from catboost import CatBoostClassifier
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise ImportError(
                    "CatBoost backend requested; install the project's locked dependencies"
                ) from exc
            return CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="Logloss",
                iterations=200,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=self.random_state,
                task_type="CPU",
                thread_count=1,
                allow_writing_files=False,
                verbose=False,
            )

        if self.backend == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise ImportError(
                    "LightGBM backend requested; install the project's locked dependencies"
                ) from exc
            return LGBMClassifier(
                objective="binary",
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                subsample=1.0,
                colsample_bytree=1.0,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=1,
                deterministic=True,
                verbosity=-1,
            )

        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError(
                "XGBoost backend requested; install the project's locked dependencies"
            ) from exc
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    @staticmethod
    def _predict_heads(
        fitted: _FittedHeads, features: FeatureInput
    ) -> tuple[list[float], list[float]]:
        matrix = fitted.encoder.transform(features)
        return (
            _predict_positive(fitted.helpful_model, matrix),
            _predict_positive(fitted.harmful_model, matrix),
        )

    def _require_fitted(self) -> None:
        if self._full is None:
            raise RuntimeError("GroupUpliftGate has not been fitted")


# The earlier prototype used this name; keeping the alias makes runner
# integration painless without changing the new cell-query semantics.
RelativeGainGate = GroupUpliftGate
TabularPreprocessor = PairFeatureEncoder
GatePrediction = GroupGatePrediction


def _heads_metadata(fitted: _FittedHeads) -> dict[str, object]:
    return {
        "family_left_out": fitted.family_left_out,
        "rows": fitted.rows,
        "encoder": fitted.encoder.as_dict(),
        "helpful_head": _model_metadata(fitted.helpful_model),
        "harmful_head": _model_metadata(fitted.harmful_model),
    }


def _model_metadata(model: object) -> dict[str, object]:
    if isinstance(model, _ConstantProbabilityModel):
        return model.as_dict()
    if hasattr(model, "metadata") and callable(getattr(model, "metadata")):
        value = model.metadata()
        if isinstance(value, Mapping):
            return dict(value)
    metadata: dict[str, object] = {
        "kind": "classifier",
        "class_name": type(model).__name__,
    }
    if type(model).__name__ == "CatBoostClassifier" and hasattr(model, "get_params"):
        metadata["parameters"] = dict(model.get_params())
    return metadata


def _predict_positive(model: object, matrix: object) -> list[float]:
    if isinstance(model, _ConstantProbabilityModel):
        return model.predict_positive(len(matrix))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        probabilities = model.predict_proba(matrix)
    classes = [int(value) for value in model.classes_]
    if 1 not in classes:
        raise RuntimeError("binary classifier does not expose positive class 1")
    positive_index = classes.index(1)
    return [_clip_probability(float(row[positive_index])) for row in probabilities]


def _sample_standard_deviation(values: Sequence[float]) -> float:
    """Sample standard deviation, explicitly using ``ddof=1``."""

    if len(values) < 2:
        return 0.0
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return float(math.sqrt(max(0.0, variance)))


def _validate_penalty(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _reject_forbidden_features(names: Sequence[str]) -> None:
    forbidden = sorted(
        name for name in (str(value).strip().lower() for value in names) if name in _FORBIDDEN_FEATURES
    )
    if forbidden:
        raise ValueError(f"label or annotation fields cannot be gate features: {forbidden}")


def _as_records(features: FeatureInput) -> tuple[list[dict[str, object]], list[str]]:
    if features is None:
        raise ValueError("features cannot be None")
    if hasattr(features, "to_dict") and hasattr(features, "columns"):
        raw_records = features.to_dict(orient="records")
        names = [str(name) for name in list(features.columns)]
        return (
            [{str(key): value for key, value in row.items()} for row in raw_records],
            names,
        )
    if hasattr(features, "tolist") and not isinstance(features, (str, bytes)):
        features = features.tolist()
    if isinstance(features, Mapping):
        raise TypeError("features must contain rows, not a single mapping")
    if isinstance(features, (str, bytes)):
        raise TypeError("features must be a two-dimensional table")

    rows = list(features)
    if not rows:
        return [], []
    if all(isinstance(row, Mapping) for row in rows):
        names = sorted({str(key) for row in rows for key in row.keys()})
        return (
            [
                {str(key): value for key, value in row.items()}  # type: ignore[union-attr]
                for row in rows
            ],
            names,
        )

    normalized: list[list[object]] = []
    for row in rows:
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            normalized.append([row])
        else:
            normalized.append(list(row))
    width = len(normalized[0])
    if width == 0:
        return ([{} for _ in normalized], [])
    if any(len(row) != width for row in normalized):
        raise ValueError("all feature rows must have the same width")
    names = [f"f{index}" for index in range(width)]
    return [dict(zip(names, row)) for row in normalized], names


def _field(item: Mapping[str, object] | object, name: str, default: object) -> object:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _is_missing(value: object) -> bool:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _is_numeric(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_numeric(value) and math.isfinite(float(value))


def _category_token(value: object) -> str:
    return f"{type(value).__name__}:{value!s}"


def _clip_probability(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("probability must be finite")
    return float(min(1.0, max(0.0, float(value))))


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
