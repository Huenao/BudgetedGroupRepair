"""Leakage-safe cell-query uplift models.

The gate learns two binary events for every ``(cell_id, query_id)`` pair:

``helpful``
    The executable LLM repair is correct while the Baran fallback is wrong.
``harmful``
    The executable LLM repair is wrong while the Baran fallback is correct.

Only a parsed, basically validated ``propose`` item is executable.  Every
other outcome is the Baran fallback and therefore receives two neutral labels.
Point predictions come from a model fitted on all training families.  Model
uncertainty is the sample standard deviation (``ddof=1``) of net-gain
predictions from leave-one-family-out replicas.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from statistics import median
from typing import Any, Literal, Mapping, Sequence, TypeAlias


Backend: TypeAlias = Literal["lightgbm", "xgboost"]
FeatureInput: TypeAlias = Any

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
        """Mathematical alias for the conservative uplift."""

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
    encoder: PairFeatureEncoder
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
    """Return whether a parsed item denotes a basically valid LLM upgrade.

    Parsers may pass an immutable item (already schema-valid) or a ledger
    mapping.  A mapping can explicitly set ``base_valid``/``verifier_valid``
    false to turn the item into a neutral fallback before labels are built.
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
    """Shared LightGBM/XGBoost helpful/harmful model for cell-query pairs."""

    def __init__(
        self,
        backend: Backend,
        *,
        rho: float = 1.0,
        gamma: float = 1.0,
        random_state: int = 42,
    ) -> None:
        if backend not in {"lightgbm", "xgboost"}:
            raise ValueError("backend must be 'lightgbm' or 'xgboost'")
        _validate_penalty("rho", rho)
        _validate_penalty("gamma", gamma)
        self.backend = backend
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.random_state = int(random_state)
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
    ) -> _FittedHeads:
        encoder = PairFeatureEncoder()
        matrix = encoder.fit_transform(features)
        return _FittedHeads(
            family_left_out=family_left_out,
            rows=len(matrix),
            encoder=encoder,
            helpful_model=self._fit_binary_head(matrix, helpful),
            harmful_model=self._fit_binary_head(matrix, harmful),
        )

    def _fit_binary_head(
        self, matrix: list[list[float]], labels: Sequence[int]
    ) -> object:
        values = tuple(int(label) for label in labels)
        if not values or any(label not in {0, 1} for label in values):
            raise ValueError("binary labels must be a non-empty sequence of 0/1 values")
        if len(set(values)) == 1:
            return _ConstantProbabilityModel(float(values[0]))
        model = self._new_backend_model()
        model.fit(matrix, list(values))
        return model

    def _new_backend_model(self) -> object:
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
    return {"kind": "classifier", "class_name": type(model).__name__}


def _predict_positive(model: object, matrix: list[list[float]]) -> list[float]:
    if isinstance(model, _ConstantProbabilityModel):
        return model.predict_positive(len(matrix))
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
