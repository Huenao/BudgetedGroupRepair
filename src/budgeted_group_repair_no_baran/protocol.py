"""Strict family-holdout protocol and singleton-referenced token budgets."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

import pandas as pd

from .data import SOURCE_DATASETS, TABLEEG_DATASETS


PROTOCOL_NAME = "strict_base_family_target_zero_shot_no_validation"

BASE_FAMILY = {
    "flight_10": "flights",
    "flight_20": "flights",
    "flights": "flights",
    "restaurant_10": "restaurant",
    "restaurant_20": "restaurant",
    "restaurant_rule_only": "restaurant",
    "movie_metadata_10": "movies",
    "movies_1": "movies",
}

_TARGET_LABEL_COLUMNS = frozenset(
    {
        "clean_value",
        "right_value",
        "correct_repair",
        "baran_correct",
        "llm_correct",
        "helpful",
        "harmful",
        "uplift_label",
        "target_label",
    }
)
_TARGET_RESPONSE_COLUMNS = frozenset(
    {"response_text", "llm_response", "parsed_response", "llm_prediction"}
)


def base_family(dataset: object) -> str:
    name = str(dataset)
    return BASE_FAMILY.get(name, name)


@dataclass(frozen=True, slots=True)
class SplitAudit:
    protocol: str
    target_suite: str
    target_dataset: str
    train_datasets: tuple[str, ...]
    test_datasets: tuple[str, ...]
    train_cells: int
    test_cells: int
    train_test_cell_overlap: int
    target_in_train: bool
    target_base_family: str
    train_test_base_family_overlap: int
    train_test_row_identity_overlap: int
    train_test_query_overlap: int = 0
    train_test_group_signature_overlap: int = 0
    target_group_label_used: bool = False
    target_response_used_before_selection: bool = False
    validation_cells: int = 0

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["train_datasets"] = ",".join(self.train_datasets)
        row["test_datasets"] = ",".join(self.test_datasets)
        return row


def _meaningful_values(frame: pd.DataFrame, columns: Iterable[str]) -> bool:
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        present = series.notna() & series.astype(str).str.strip().ne("")
        if bool(present.any()):
            return True
    return False


def _membership_signature(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "\x1f".join(sorted(str(item) for item in value))
    text = str(value)
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return text
    if isinstance(parsed, list):
        return "\x1f".join(sorted(str(item) for item in parsed))
    return text


def _signatures(frame: pd.DataFrame) -> set[str]:
    if "group_signature" in frame.columns:
        return set(frame["group_signature"].astype(str))
    if "cell_ids" in frame.columns:
        memberships = frame["cell_ids"].map(_membership_signature)
        return set(frame["dataset"].astype(str) + "\x1e" + memberships)
    return set()


def split_for_target(
    feature_table: pd.DataFrame,
    target_suite: str,
    target_dataset: str,
    *,
    enforce_target_unlabeled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, SplitAudit]:
    """Create a zero-validation, whole-base-family holdout split.

    The input may be a cell table or an expanded cell-query table.  Audit cell
    counts are unique coordinate counts, while returned frames retain all pair
    rows.  Target labels/responses are rejected before selection by default.
    """

    required = {"cell_id", "suite", "dataset"}
    missing = required - set(feature_table.columns)
    if missing:
        raise ValueError(f"feature table is missing required columns: {sorted(missing)}")
    target_family = base_family(target_dataset)
    families = feature_table["dataset"].map(base_family)
    if target_suite == "tableeg":
        if target_dataset not in TABLEEG_DATASETS:
            raise ValueError(f"unknown TableEG target: {target_dataset}")
    elif target_suite == "source":
        if target_dataset not in SOURCE_DATASETS:
            raise ValueError(f"unknown source target: {target_dataset}")
    else:
        raise ValueError("target_suite must be 'source' or 'tableeg'")

    train_mask = (feature_table["suite"].astype(str) == "tableeg") & (families != target_family)
    test_mask = (feature_table["suite"].astype(str) == target_suite) & (
        feature_table["dataset"].astype(str) == target_dataset
    )
    train = feature_table.loc[train_mask].copy().reset_index(drop=True)
    test = feature_table.loc[test_mask].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError(f"empty training split for {target_suite}/{target_dataset}")
    if test.empty:
        raise ValueError(f"empty test split for {target_suite}/{target_dataset}")

    train_ids = set(train["cell_id"].astype(str))
    test_ids = set(test["cell_id"].astype(str))
    train_datasets = tuple(sorted(set(train["dataset"].astype(str))))
    train_families = set(train["dataset"].map(base_family))
    test_families = set(test["dataset"].map(base_family))
    row_identity_overlap = 0
    if {"row_id", "column"}.issubset(feature_table.columns):
        train_rows = set(
            zip(
                train["dataset"].map(base_family),
                train["row_id"].astype(str),
                train["column"].astype(str),
            )
        )
        test_rows = set(
            zip(
                test["dataset"].map(base_family),
                test["row_id"].astype(str),
                test["column"].astype(str),
            )
        )
        row_identity_overlap = len(train_rows & test_rows)
    train_query_ids = set(train["query_id"].astype(str)) if "query_id" in train else set()
    test_query_ids = set(test["query_id"].astype(str)) if "query_id" in test else set()
    target_group_label_used = _meaningful_values(test, _TARGET_LABEL_COLUMNS)
    target_response_used = _meaningful_values(test, _TARGET_RESPONSE_COLUMNS)
    audit = SplitAudit(
        protocol=PROTOCOL_NAME,
        target_suite=target_suite,
        target_dataset=target_dataset,
        train_datasets=train_datasets,
        test_datasets=(target_dataset,),
        train_cells=len(train_ids),
        test_cells=len(test_ids),
        train_test_cell_overlap=len(train_ids & test_ids),
        target_in_train=target_dataset in train_datasets,
        target_base_family=target_family,
        train_test_base_family_overlap=len(train_families & test_families),
        train_test_row_identity_overlap=row_identity_overlap,
        train_test_query_overlap=len(train_query_ids & test_query_ids),
        train_test_group_signature_overlap=len(_signatures(train) & _signatures(test)),
        target_group_label_used=target_group_label_used,
        target_response_used_before_selection=target_response_used,
    )
    structural_leak = (
        audit.train_test_cell_overlap
        or audit.target_in_train
        or audit.train_test_base_family_overlap
        or audit.train_test_row_identity_overlap
        or audit.train_test_query_overlap
        or audit.train_test_group_signature_overlap
    )
    label_leak = audit.target_group_label_used or audit.target_response_used_before_selection
    if structural_leak or (enforce_target_unlabeled and label_leak):
        raise AssertionError(f"leaky split: {audit}")
    return train, test, audit


def target_order() -> tuple[tuple[str, str], ...]:
    return tuple(("source", name) for name in SOURCE_DATASETS) + tuple(
        ("tableeg", name) for name in TABLEEG_DATASETS
    )


def singleton_reference_cost(
    queries: pd.DataFrame,
    *,
    cost_column: str = "estimated_total_tokens",
) -> int:
    """Return the cost of exactly one full singleton query per target cell."""

    if cost_column not in queries.columns:
        raise ValueError(f"missing singleton cost column: {cost_column}")
    singletons = queries
    if "group_size" in queries.columns:
        singletons = queries.loc[pd.to_numeric(queries["group_size"], errors="coerce") == 1]
    if "group_view" in singletons.columns:
        singletons = singletons.loc[singletons["group_view"].astype(str) == "singleton"]
    if singletons.empty:
        raise ValueError("no singleton queries found")
    if "cell_id" in singletons.columns:
        ids = singletons["cell_id"].astype(str)
    elif "cell_ids" in singletons.columns:
        ids = singletons["cell_ids"].map(_membership_signature)
    else:
        raise ValueError("singleton queries require cell_id or cell_ids")
    if ids.duplicated().any():
        raise ValueError("each target cell must have exactly one singleton query")
    costs = pd.to_numeric(singletons[cost_column], errors="raise")
    if bool((costs <= 0).any()):
        raise ValueError("singleton estimated token costs must be positive")
    return int(costs.sum())


def total_estimated_budget(
    singleton_queries: pd.DataFrame,
    budget_share: float,
    *,
    cost_column: str = "estimated_total_tokens",
) -> int:
    if not 0.0 <= float(budget_share) <= 1.0:
        raise ValueError("budget_share must be in [0, 1]")
    reference = singleton_reference_cost(singleton_queries, cost_column=cost_column)
    return int(round(reference * float(budget_share)))


__all__ = [
    "BASE_FAMILY",
    "PROTOCOL_NAME",
    "SplitAudit",
    "base_family",
    "singleton_reference_cost",
    "split_for_target",
    "target_order",
    "total_estimated_budget",
]
