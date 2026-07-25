"""Fresh Baran adapter with label-free online records.

Baran itself needs clean values to simulate its twenty-label protocol.  That
oracle access is contained inside :func:`run_baran`; the returned records are
checked recursively and never contain clean values or correctness labels.
Evaluation is an explicit, separate operation.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .data import LoadedDataset, OracleCell, normalize_for_match


FORBIDDEN_ONLINE_BARAN_KEYS = frozenset(
    {
        "clean",
        "clean_value",
        "right_value",
        "correct",
        "correct_repair",
        "baran_correct",
        "llm_correct",
        "helpful",
        "harmful",
        "error_type",
        "missing_value",
        "constraint",
        "tuple_pairs",
    }
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value).strip()


def _load_raha(raha_source_root: str | Path) -> tuple[Any, Any]:
    root = Path(raha_source_root).resolve()
    if not (root / "raha").is_dir():
        raise FileNotFoundError(f"configured RAHA source root has no raha package: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    try:
        dataset_module = importlib.import_module("raha.dataset")
        correction_module = importlib.import_module("raha.correction")
    except ImportError as error:
        raise ImportError(f"failed to import Baran from configured RAHA root {root}") from error
    for module in (dataset_module, correction_module):
        module_path = Path(str(getattr(module, "__file__", ""))).resolve()
        if root not in module_path.parents:
            raise RuntimeError(
                f"raha module was already imported from another checkout: {module_path}; "
                "start a fresh process"
            )
    return dataset_module.Dataset, correction_module.Correction


def _configure_raha_pool(start_method: str | None) -> str:
    """Select the process context used by Baran's repeatedly-created pools."""

    method = (
        str(start_method).strip()
        if start_method is not None
        else multiprocessing.get_start_method(allow_none=False)
    )
    available = tuple(multiprocessing.get_all_start_methods())
    if method not in available:
        raise ValueError(
            f"unsupported Baran multiprocessing start method {method!r}; "
            f"available methods are {available}"
        )
    correction_module = importlib.import_module("raha.correction")
    correction_module.Pool = multiprocessing.get_context(method).Pool
    return method


def _source_names(app: Any, column_count: int, feature_count: int) -> list[str]:
    encodings = [str(value) for value in getattr(app, "VALUE_ENCODINGS", ("identity", "unicode"))]
    result = [
        f"value_{model}_{encoding}"
        for model in ("remover", "adder", "replacer", "swapper")
        for encoding in encodings
    ]
    result.extend(f"vicinity_column_{index}" for index in range(column_count))
    result.append("domain")
    if len(result) < feature_count:
        result.extend(f"source_{index}" for index in range(len(result), feature_count))
    return result[:feature_count]


def _empty_diagnostics(status: str) -> dict[str, Any]:
    return {
        "candidate_count": 0,
        "top_support": 0.0,
        "second_support": 0.0,
        "support_margin": 0.0,
        "predicted_support": 0.0,
        "source_agreement": 0.0,
        "source_group_agreement": 0.0,
        "source_vote_count": 0,
        "source_model_count": 0,
        "predicted_is_top_support": False,
        "top_candidates": [],
        "diagnostic_status": status,
    }


def _candidate_diagnostics(
    app: Any,
    raha_dataset: Any,
    coordinate: tuple[int, int],
    prediction: str,
) -> dict[str, Any]:
    generator = getattr(app, "_feature_generator_process", None)
    if generator is None:
        return _empty_diagnostics("unavailable:no_feature_generator")
    try:
        _, pair_features, _ = generator([coordinate], dataset=raha_dataset)
        raw_candidates = pair_features.get(coordinate, {})
    except Exception as error:  # private upstream API varies
        return _empty_diagnostics(f"unavailable:{type(error).__name__}")
    if not raw_candidates:
        return _empty_diagnostics("ok_no_candidates")

    feature_count = max(len(np.asarray(vector).reshape(-1)) for vector in raw_candidates.values())
    source_names = _source_names(app, int(raha_dataset.dataframe.shape[1]), feature_count)
    candidates: list[dict[str, Any]] = []
    active_sources: set[int] = set()
    for raw_value, vector in raw_candidates.items():
        numeric = np.nan_to_num(
            np.asarray(vector, dtype=float).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0
        )
        indices = {index for index, score in enumerate(numeric) if score > 0.0}
        active_sources.update(indices)
        candidates.append(
            {
                "value": _text(raw_value),
                "raw_support": float(numeric.sum()),
                "source_indices": indices,
                "sources": [source_names[index] for index in sorted(indices) if index < len(source_names)],
            }
        )
    active_count = max(1, len(active_sources))
    for candidate in candidates:
        indices = candidate["source_indices"]
        candidate["support"] = float(candidate["raw_support"]) / active_count
        candidate["source_count"] = len(indices)
        candidate["source_agreement"] = len(indices) / active_count
    candidates.sort(key=lambda candidate: (-float(candidate["support"]), str(candidate["value"])))
    selected = next((candidate for candidate in candidates if candidate["value"] == prediction), candidates[0])
    top_support = float(candidates[0]["support"])
    second_support = float(candidates[1]["support"]) if len(candidates) > 1 else 0.0
    selected_groups = {str(source).split("_", 1)[0] for source in selected["sources"]}
    active_groups = {
        source_names[index].split("_", 1)[0]
        for index in active_sources
        if index < len(source_names)
    }
    top_candidates = [
        {
            "value": candidate["value"],
            "support": round(float(candidate["support"]), 8),
            "source_count": int(candidate["source_count"]),
            "sources": list(candidate["sources"]),
        }
        for candidate in candidates[:10]
    ]
    return {
        "candidate_count": len(candidates),
        "top_support": round(top_support, 8),
        "second_support": round(second_support, 8),
        "support_margin": round(max(0.0, top_support - second_support), 8),
        "predicted_support": round(float(selected["support"]), 8),
        "source_agreement": round(float(selected["source_agreement"]), 8),
        "source_group_agreement": round(len(selected_groups) / max(1, len(active_groups)), 8),
        "source_vote_count": int(selected["source_count"]),
        "source_model_count": len(active_sources),
        "predicted_is_top_support": selected["value"] == candidates[0]["value"],
        "top_candidates": top_candidates,
        "diagnostic_candidate": selected["value"],
        "diagnostic_status": "ok",
    }


def _walk_keys(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_keys(nested)


def assert_online_baran_record_safe(record: Mapping[str, object]) -> None:
    leaked = {key for key in _walk_keys(record) if key.lower() in FORBIDDEN_ONLINE_BARAN_KEYS}
    if leaked:
        raise AssertionError(f"online Baran record contains oracle/correctness fields: {sorted(leaked)}")


def run_baran(
    dataset: LoadedDataset,
    cells: Sequence[OracleCell],
    raha_source_root: str | Path,
    labeling_budget: int = 20,
    seed: int = 16,
    workers: int = 4,
    multiprocessing_start_method: str | None = None,
    *,
    verbose: bool = False,
    extract_diagnostics: bool = True,
) -> list[dict[str, Any]]:
    """Run fresh Baran and emit one label-free online record per oracle cell."""

    selected = list(cells)
    if not selected:
        return []
    if not all(isinstance(cell, OracleCell) for cell in selected):
        raise TypeError("run_baran requires OracleCell inputs at the isolated Baran boundary")
    if any(cell.suite != dataset.suite or cell.dataset != dataset.name for cell in selected):
        raise ValueError("all Baran cells must belong to the supplied dataset")
    if labeling_budget < 0:
        raise ValueError("labeling_budget must be non-negative")
    if workers <= 0:
        raise ValueError("workers must be positive")
    if not dataset.dirty_path.is_file() or not dataset.clean_path.is_file():
        raise FileNotFoundError("Baran requires existing dirty and clean CSV paths")

    DatasetClass, CorrectionClass = _load_raha(raha_source_root)
    resolved_start_method = _configure_raha_pool(multiprocessing_start_method)
    random.seed(int(seed))
    np.random.seed(int(seed))
    raha_dataset = DatasetClass(
        {
            "name": dataset.name,
            "path": str(dataset.dirty_path.resolve()),
            "clean_path": str(dataset.clean_path.resolve()),
        }
    )
    coordinates = [(cell.row, cell.col) for cell in selected]
    # This is the only online-pipeline boundary at which oracle repair values
    # are bound.  RAHA uses them solely for its simulated labeling budget.
    raha_dataset.detected_cells = {
        coordinate: cell.clean_value for coordinate, cell in zip(coordinates, selected)
    }

    app = CorrectionClass()
    app.LABELING_BUDGET = min(int(labeling_budget), int(raha_dataset.dataframe.shape[0]))
    app.SAVE_RESULTS = False
    app.VERBOSE = bool(verbose)
    app.NUM_WORKERS = max(1, int(workers))
    corrections = app.run(raha_dataset)

    records: list[dict[str, Any]] = []
    for cell, coordinate in zip(selected, coordinates):
        has_prediction = coordinate in corrections
        prediction = _text(corrections.get(coordinate, cell.dirty_value))
        diagnostics = (
            _candidate_diagnostics(app, raha_dataset, coordinate, prediction)
            if extract_diagnostics
            else _empty_diagnostics("disabled")
        )
        record: dict[str, Any] = {
            **cell.to_safe().as_dict(),
            "prediction": prediction,
            "method": "baran",
            "model": "baran",
            "parse_status": "ok_baran" if has_prediction else "no_prediction",
            "valid_prediction": bool(has_prediction),
            "labeling_budget": int(app.LABELING_BUDGET),
            "seed": int(seed),
            "workers": int(app.NUM_WORKERS),
            "multiprocessing_start_method": resolved_start_method,
            "diagnostics": diagnostics,
        }
        for key in (
            "candidate_count",
            "top_support",
            "support_margin",
            "predicted_support",
            "source_agreement",
            "source_group_agreement",
            "source_vote_count",
            "source_model_count",
            "predicted_is_top_support",
        ):
            record[key] = diagnostics[key]
        record["top_candidate_support"] = diagnostics["top_support"]
        record["second_candidate_support"] = diagnostics["second_support"]
        record["candidate_margin"] = diagnostics["support_margin"]
        record["no_candidate"] = not has_prediction
        record["candidate_generation_empty"] = diagnostics["candidate_count"] == 0
        record["top_candidates"] = json.dumps(
            diagnostics.get("top_candidates", []), ensure_ascii=False, sort_keys=True
        )
        assert_online_baran_record_safe(record)
        records.append(record)
    return records


def evaluate_baran_records(
    records: Sequence[Mapping[str, object]],
    oracle_cells: Sequence[OracleCell],
) -> list[dict[str, object]]:
    """Attach clean values/correctness only inside an explicit evaluation stage."""

    oracle = {str(cell.cell_id): cell for cell in oracle_cells}
    if len(oracle) != len(oracle_cells):
        raise ValueError("duplicate oracle cell IDs")
    evaluated: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_record in records:
        record = dict(raw_record)
        cell_id = str(record.get("cell_id", ""))
        if not cell_id or cell_id in seen or cell_id not in oracle:
            raise ValueError(f"invalid or duplicate Baran record cell_id: {cell_id!r}")
        seen.add(cell_id)
        cell = oracle[cell_id]
        prediction = normalize_for_match(record.get("prediction", cell.dirty_value))
        valid = bool(record.get("valid_prediction", False))
        evaluated.append(
            {
                **record,
                "clean_value": cell.clean_value,
                "correct_repair": valid and prediction == normalize_for_match(cell.clean_value),
                "evaluation_stage": True,
            }
        )
    if seen != set(oracle):
        missing = sorted(set(oracle) - seen)
        raise ValueError(f"Baran evaluation is missing {len(missing)} oracle cells")
    return evaluated


__all__ = [
    "FORBIDDEN_ONLINE_BARAN_KEYS",
    "assert_online_baran_record_safe",
    "evaluate_baran_records",
    "run_baran",
]
