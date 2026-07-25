"""Oracle-bound evaluation for complementarity and grouping effectiveness."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .data import normalize_for_match
from .group_generator import GroupQueryAction
from .statistics import cluster_bootstrap, exact_mcnemar, holm_adjust, two_way_cluster_bootstrap


def _actual_tokens(response: Mapping[str, Any]) -> int | None:
    observed = response.get("observed_total_tokens")
    if observed is not None and int(observed or 0) > 0:
        return int(observed)
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    for key in ("total_tokens",):
        if usage.get(key) is not None:
            return int(usage[key])
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt is None or completion is None:
        return None
    return int(prompt) + int(completion)


def _usage_component(response: Mapping[str, Any], *names: str) -> int | None:
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return None
    for name in names:
        if usage.get(name) is not None:
            return int(usage[name])
    return None


def materialize_arm_results(
    actions: Sequence[GroupQueryAction],
    responses: Sequence[Mapping[str, Any]],
    *,
    dirty_by_cell: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    response_by_query = {str(row.get("query_id")): row for row in responses}
    records: list[dict[str, Any]] = []
    for action in actions:
        response = response_by_query.get(action.query_id, {})
        items_raw = response.get("items")
        items = {
            str(item.get("cell_id")): item
            for item in items_raw
            if isinstance(item, Mapping) and item.get("cell_id") is not None
        } if isinstance(items_raw, list) else {}
        tokens = _actual_tokens(response)
        prompt_tokens = _usage_component(response, "prompt_tokens", "input_tokens")
        completion_tokens = _usage_component(response, "completion_tokens", "output_tokens")
        for cell_id in action.cell_ids:
            item = items.get(cell_id)
            decision = str(item.get("decision", "")) if item else ""
            prediction = str(item.get("repair", "")) if item else ""
            dirty = str(dirty_by_cell[cell_id])
            valid = bool(
                item
                and decision == "propose"
                and prediction.strip()
                and normalize_for_match(prediction) != normalize_for_match(dirty)
            )
            item_present = item is not None
            records.append(
                {
                    "suite": action.suite,
                    "dataset": action.dataset,
                    "cell_id": cell_id,
                    "arm": action.arm,
                    "query_id": action.query_id,
                    "group_view": action.group_view,
                    "group_size": action.group_size,
                    "prediction": prediction if valid else "",
                    "decision": decision or "missing",
                    "parse_status": str(response.get("parse_status", "missing_response")),
                    "item_present": item_present,
                    "parse_valid_item": bool(
                        item_present
                        and decision in {"propose", "abstain"}
                        and response.get("model_matches_request", True)
                    ),
                    "missing_item": not item_present,
                    "unchanged_dirty": bool(
                        item_present
                        and decision == "propose"
                        and normalize_for_match(prediction) == normalize_for_match(dirty)
                    ),
                    "valid_prediction": valid,
                    "query_usage_key": f"{action.query_id}:{action.prompt_hash}",
                    "actual_query_tokens": tokens,
                    "actual_prompt_tokens": prompt_tokens,
                    "actual_completion_tokens": completion_tokens,
                    "estimated_query_tokens": action.estimated_total_tokens,
                    "attempts": int(response.get("attempts", 0) or 0),
                    "unknown_usage_attempts": int(
                        response.get("unknown_usage_attempts", 0) or 0
                    ),
                    "latency_seconds": float(response.get("latency_seconds", 0.0) or 0.0),
                    "provider_request_hash": str(response.get("provider_request_hash", "")),
                    "cache_hit": bool(response.get("cache_hit")),
                    "checkpoint_hit": bool(response.get("checkpoint_hit")),
                }
            )
    if len(records) != sum(action.group_size for action in actions):
        raise AssertionError("arm result materialization lost query members")
    return tuple(records)


def bind_oracle_correctness(
    records: Sequence[Mapping[str, Any]],
    clean_by_cell: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    for record in records:
        cell_id = str(record["cell_id"])
        if cell_id not in clean_by_cell:
            raise KeyError(f"missing oracle value for {cell_id}")
        prediction = str(record.get("prediction", ""))
        output.append(
            {
                **dict(record),
                "correct": bool(
                    record.get("valid_prediction")
                    and normalize_for_match(prediction) == normalize_for_match(clean_by_cell[cell_id])
                ),
            }
        )
    return tuple(output)


def _proposal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    total = len(rows)
    valid = sum(bool(row.get("valid_prediction")) for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    precision = correct / valid if valid else math.nan
    recall = correct / total if total else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "valid_proposals": valid,
        "correct": correct,
        "accuracy": recall,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else math.nan


def _rate(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else math.nan


def _unique_query_tokens(rows: Sequence[Mapping[str, Any]]) -> int | None:
    queries = {str(row["query_id"]): row for row in rows}
    values = [row.get("actual_query_tokens") for row in queries.values()]
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def complementarity_metrics(
    singleton_records: Sequence[Mapping[str, Any]],
    *,
    baran_prediction_by_cell: Mapping[str, str],
    clean_by_cell: Mapping[str, str],
    row_id_by_cell: Mapping[str, str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float = 0.95,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in singleton_records:
        cell_id = str(record["cell_id"])
        clean = clean_by_cell[cell_id]
        baran_correct = normalize_for_match(baran_prediction_by_cell[cell_id]) == normalize_for_match(clean)
        llm_correct = bool(record.get("correct"))
        by_dataset[str(record["dataset"])].append(
            {
                **dict(record),
                "row_id": row_id_by_cell[cell_id],
                "baran_correct": baran_correct,
                "llm_correct": llm_correct,
            }
        )
    rows_out: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    all_cells: list[dict[str, Any]] = []
    for dataset, rows in sorted(by_dataset.items()):
        all_cells.extend(rows)
        counts = Counter((int(row["baran_correct"]), int(row["llm_correct"])) for row in rows)
        n11, n10, n01, n00 = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
        total = len(rows)
        baran_acc = (n11 + n10) / total
        llm_acc = (n11 + n01) / total
        oracle_ub = 1 - n00 / total
        def statistic(sample: Sequence[Mapping[str, Any]]) -> float:
            return sum(bool(row["llm_correct"]) for row in sample) / len(sample)
        llm_low, llm_high = cluster_bootstrap(
            rows,
            cluster_key="row_id",
            statistic=statistic,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence=confidence,
        )
        ub_low, ub_high = cluster_bootstrap(
            rows,
            cluster_key="row_id",
            statistic=lambda sample: sum(
                bool(row["baran_correct"] or row["llm_correct"]) for row in sample
            ) / len(sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
            confidence=confidence,
        )
        salvage_low, salvage_high = cluster_bootstrap(
            rows,
            cluster_key="row_id",
            statistic=lambda sample: (
                sum(bool(row["llm_correct"] and not row["baran_correct"]) for row in sample)
                / sum(not row["baran_correct"] for row in sample)
                if any(not row["baran_correct"] for row in sample)
                else math.nan
            ),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2,
            confidence=confidence,
        )
        proposal = _proposal_metrics(rows)
        p_values[dataset] = exact_mcnemar(n10, n01)
        rows_out.append(
            {
                "dataset": dataset,
                "N": total,
                "baran_accuracy": baran_acc,
                "singleton_accuracy": llm_acc,
                "singleton_accuracy_ci_low": llm_low,
                "singleton_accuracy_ci_high": llm_high,
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": n00,
                "oracle_upper_bound": oracle_ub,
                "oracle_upper_bound_ci_low": ub_low,
                "oracle_upper_bound_ci_high": ub_high,
                "upper_bound_minus_baran": n01 / total,
                "upper_bound_minus_best": min(n10, n01) / total,
                "llm_salvage_rate": n01 / (n01 + n00) if n01 + n00 else math.nan,
                "llm_salvage_rate_ci_low": salvage_low,
                "llm_salvage_rate_ci_high": salvage_high,
                "baran_rescue_rate": n10 / (n10 + n00) if n10 + n00 else math.nan,
                "mcnemar_p": p_values[dataset],
                **{f"singleton_{key}": value for key, value in proposal.items()},
            }
        )
    adjusted = holm_adjust(p_values)
    for row in rows_out:
        row["mcnemar_p_holm"] = adjusted[row["dataset"]]
    aggregate_counts = Counter(
        (int(row["baran_correct"]), int(row["llm_correct"])) for row in all_cells
    )
    n = len(all_cells)
    micro = {
        "N": n,
        "n11": aggregate_counts[(1, 1)],
        "n10": aggregate_counts[(1, 0)],
        "n01": aggregate_counts[(0, 1)],
        "n00": aggregate_counts[(0, 0)],
    }
    micro["baran_accuracy"] = (micro["n11"] + micro["n10"]) / n
    micro["singleton_accuracy"] = (micro["n11"] + micro["n01"]) / n
    micro["oracle_upper_bound"] = 1 - micro["n00"] / n
    micro["upper_bound_minus_baran"] = micro["n01"] / n
    micro["upper_bound_minus_best"] = min(micro["n10"], micro["n01"]) / n
    micro["llm_salvage_rate"] = (
        micro["n01"] / (micro["n01"] + micro["n00"])
        if micro["n01"] + micro["n00"]
        else math.nan
    )
    micro.update({f"singleton_{key}": value for key, value in _proposal_metrics(all_cells).items()})
    macro = {
        key: _mean([float(row[key]) for row in rows_out])
        for key in (
            "baran_accuracy",
            "singleton_accuracy",
            "oracle_upper_bound",
            "upper_bound_minus_baran",
            "upper_bound_minus_best",
            "llm_salvage_rate",
            "singleton_precision",
            "singleton_recall",
            "singleton_f1",
        )
    }
    return rows_out, {
        "micro": micro,
        "macro": macro,
        "worst_dataset": min(
            rows_out, key=lambda row: (float(row["upper_bound_minus_best"]), row["dataset"])
        )["dataset"],
        "worst_dataset_upper_bound_minus_best": min(
            float(row["upper_bound_minus_best"]) for row in rows_out
        ),
        "datasets_with_llm_salvage": sum(row["n01"] > 0 for row in rows_out),
        "datasets_with_bidirectional_complementarity": sum(
            row["n01"] > 0 and row["n10"] > 0 for row in rows_out
        ),
    }


def grouping_metrics(
    singleton: Sequence[Mapping[str, Any]],
    structured: Sequence[Mapping[str, Any]],
    random: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence: float,
    noninferiority_margin: float,
    minimum_token_saving: float,
    maximum_parse_validity_drop: float,
    maximum_missing_item_rate_increase: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    single_by_cell = {str(row["cell_id"]): row for row in singleton}
    structured_by_cell = {str(row["cell_id"]): row for row in structured}
    random_by_cell = {str(row["cell_id"]): row for row in random}
    if not (set(single_by_cell) == set(structured_by_cell) == set(random_by_cell)):
        raise ValueError("experiment-two arms must cover the same cell population")
    datasets = sorted({str(row["dataset"]) for row in structured})
    output: list[dict[str, Any]] = []
    for dataset in datasets:
        cell_ids = sorted(
            cell_id for cell_id, row in structured_by_cell.items() if str(row["dataset"]) == dataset
        )
        paired = [
            {
                "cell_id": cell_id,
                "structured_query_id": structured_by_cell[cell_id]["query_id"],
                "random_query_id": random_by_cell[cell_id]["query_id"],
                "single_correct": int(bool(single_by_cell[cell_id].get("correct"))),
                "structured_correct": int(bool(structured_by_cell[cell_id].get("correct"))),
                "random_correct": int(bool(random_by_cell[cell_id].get("correct"))),
            }
            for cell_id in cell_ids
        ]
        n = len(paired)
        single_acc = sum(row["single_correct"] for row in paired) / n
        structured_acc = sum(row["structured_correct"] for row in paired) / n
        random_acc = sum(row["random_correct"] for row in paired) / n
        for row in paired:
            row["structured_minus_single"] = row["structured_correct"] - row["single_correct"]
            row["structured_minus_random"] = row["structured_correct"] - row["random_correct"]
        delta_low, delta_high = cluster_bootstrap(
            paired,
            cluster_key="structured_query_id",
            statistic=lambda sample: sum(float(row["structured_minus_single"]) for row in sample) / len(sample),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
            confidence=confidence,
        )
        sr_low, sr_high = two_way_cluster_bootstrap(
            paired,
            first_cluster_key="structured_query_id",
            second_cluster_key="random_query_id",
            value_key="structured_minus_random",
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 1,
            confidence=confidence,
        )
        structured_queries = {str(structured_by_cell[cell_id]["query_id"]): structured_by_cell[cell_id] for cell_id in cell_ids}
        single_queries = {str(single_by_cell[cell_id]["query_id"]): single_by_cell[cell_id] for cell_id in cell_ids}
        structured_tokens = _unique_query_tokens(
            [structured_by_cell[cell_id] for cell_id in cell_ids]
        )
        single_tokens = _unique_query_tokens([single_by_cell[cell_id] for cell_id in cell_ids])
        token_saving = (
            1 - structured_tokens / single_tokens
            if structured_tokens is not None and single_tokens
            else math.nan
        )
        cost_pairs = []
        for query_id, group_row in structured_queries.items():
            members = [row for row in paired if row["structured_query_id"] == query_id]
            member_single_tokens = _unique_query_tokens(
                [single_by_cell[str(row["cell_id"])] for row in members]
            )
            cost_pairs.append(
                {
                    "query_id": query_id,
                    "group_tokens": group_row.get("actual_query_tokens"),
                    "single_tokens": member_single_tokens,
                }
            )
        token_low, token_high = cluster_bootstrap(
            cost_pairs,
            cluster_key="query_id",
            statistic=lambda sample: (
                1
                - sum(int(row["group_tokens"]) for row in sample)
                / sum(int(row["single_tokens"]) for row in sample)
                if sample
                and all(row["group_tokens"] is not None and row["single_tokens"] for row in sample)
                else math.nan
            ),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + 2,
            confidence=confidence,
        )
        correct_per_million_single = (
            sum(row["single_correct"] for row in paired) / single_tokens * 1e6
            if single_tokens
            else math.nan
        )
        correct_per_million_group = (
            sum(row["structured_correct"] for row in paired) / structured_tokens * 1e6
            if structured_tokens
            else math.nan
        )
        singleton_parse = _rate([single_by_cell[cell_id] for cell_id in cell_ids], "parse_valid_item")
        structured_parse = _rate([structured_by_cell[cell_id] for cell_id in cell_ids], "parse_valid_item")
        singleton_missing = _rate([single_by_cell[cell_id] for cell_id in cell_ids], "missing_item")
        structured_missing = _rate([structured_by_cell[cell_id] for cell_id in cell_ids], "missing_item")
        parse_drop = singleton_parse - structured_parse
        missing_increase = structured_missing - singleton_missing
        parse_acceptable = (
            parse_drop <= float(maximum_parse_validity_drop)
            and missing_increase <= float(maximum_missing_item_rate_increase)
        )
        if delta_low > 0 and parse_acceptable:
            decision = "A_quality_superiority"
        elif (
            delta_low > -float(noninferiority_margin)
            and token_saving >= float(minimum_token_saving)
            and token_low > 0
            and correct_per_million_group > correct_per_million_single
            and parse_acceptable
        ):
            decision = "B_noninferior_more_efficient"
        else:
            decision = "C_not_supported"
        output.append(
            {
                "dataset": dataset,
                "N_cells": n,
                "N_structured_groups": len(structured_queries),
                "singleton_accuracy": single_acc,
                "structured_accuracy": structured_acc,
                "delta_accuracy": structured_acc - single_acc,
                "delta_accuracy_ci_low": delta_low,
                "delta_accuracy_ci_high": delta_high,
                "random_accuracy": random_acc,
                "structured_minus_random": structured_acc - random_acc,
                "structured_minus_random_ci_low": sr_low,
                "structured_minus_random_ci_high": sr_high,
                "structured_tokens": structured_tokens,
                "singleton_tokens": single_tokens,
                "token_per_cell_saving": token_saving,
                "token_per_cell_saving_ci_low": token_low,
                "token_per_cell_saving_ci_high": token_high,
                "correct_per_million_tokens_singleton": correct_per_million_single,
                "correct_per_million_tokens_structured": correct_per_million_group,
                "singleton_propose_rate": _rate([single_by_cell[cell_id] for cell_id in cell_ids], "valid_prediction"),
                "structured_propose_rate": _rate([structured_by_cell[cell_id] for cell_id in cell_ids], "valid_prediction"),
                "random_propose_rate": _rate([random_by_cell[cell_id] for cell_id in cell_ids], "valid_prediction"),
                "singleton_parse_validity": singleton_parse,
                "structured_parse_validity": structured_parse,
                "parse_validity_drop": parse_drop,
                "singleton_missing_item_rate": singleton_missing,
                "structured_missing_item_rate": structured_missing,
                "missing_item_rate_increase": missing_increase,
                "structured_unchanged_dirty_rate": _rate([structured_by_cell[cell_id] for cell_id in cell_ids], "unchanged_dirty"),
                "logical_calls_singleton": len(single_queries),
                "logical_calls_structured": len(structured_queries),
                "decision": decision,
            }
        )
    macro = {
        key: _mean([float(row[key]) for row in output])
        for key in (
            "singleton_accuracy",
            "structured_accuracy",
            "delta_accuracy",
            "random_accuracy",
            "structured_minus_random",
            "token_per_cell_saving",
            "parse_validity_drop",
            "missing_item_rate_increase",
            "correct_per_million_tokens_singleton",
            "correct_per_million_tokens_structured",
        )
    }
    macro_delta_low, macro_delta_high = cluster_bootstrap(
        output,
        cluster_key="dataset",
        statistic=lambda sample: _mean([float(row["delta_accuracy"]) for row in sample]),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 10,
        confidence=confidence,
    )
    macro_token_low, macro_token_high = cluster_bootstrap(
        output,
        cluster_key="dataset",
        statistic=lambda sample: _mean(
            [float(row["token_per_cell_saving"]) for row in sample]
        ),
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 11,
        confidence=confidence,
    )
    macro["delta_accuracy_ci_low"] = macro_delta_low
    macro["delta_accuracy_ci_high"] = macro_delta_high
    macro["token_per_cell_saving_ci_low"] = macro_token_low
    macro["token_per_cell_saving_ci_high"] = macro_token_high
    macro_parse_acceptable = (
        macro["parse_validity_drop"] <= float(maximum_parse_validity_drop)
        and macro["missing_item_rate_increase"]
        <= float(maximum_missing_item_rate_increase)
    )
    if macro_delta_low > 0 and macro_parse_acceptable:
        macro_decision = "A_quality_superiority"
    elif (
        macro_delta_low > -float(noninferiority_margin)
        and macro["token_per_cell_saving"] >= float(minimum_token_saving)
        and macro_token_low > 0
        and macro["correct_per_million_tokens_structured"]
        > macro["correct_per_million_tokens_singleton"]
        and macro_parse_acceptable
    ):
        macro_decision = "B_noninferior_more_efficient"
    else:
        macro_decision = "C_not_supported"
    macro["decision"] = macro_decision
    all_cell_ids = sorted(structured_by_cell)
    all_structured_tokens = _unique_query_tokens([structured_by_cell[cell_id] for cell_id in all_cell_ids])
    all_single_tokens = _unique_query_tokens([single_by_cell[cell_id] for cell_id in all_cell_ids])
    micro = {
        "N_cells": len(all_cell_ids),
        "singleton_accuracy": sum(bool(single_by_cell[cell_id].get("correct")) for cell_id in all_cell_ids) / len(all_cell_ids),
        "structured_accuracy": sum(bool(structured_by_cell[cell_id].get("correct")) for cell_id in all_cell_ids) / len(all_cell_ids),
        "random_accuracy": sum(bool(random_by_cell[cell_id].get("correct")) for cell_id in all_cell_ids) / len(all_cell_ids),
        "singleton_tokens": all_single_tokens,
        "structured_tokens": all_structured_tokens,
        "token_per_cell_saving": (
            1 - all_structured_tokens / all_single_tokens
            if all_structured_tokens is not None and all_single_tokens
            else math.nan
        ),
    }
    micro["delta_accuracy"] = micro["structured_accuracy"] - micro["singleton_accuracy"]
    micro["structured_minus_random"] = micro["structured_accuracy"] - micro["random_accuracy"]
    worst = min(output, key=lambda row: (float(row["delta_accuracy"]), row["dataset"]))
    return output, {
        "macro": macro,
        "micro": micro,
        "decision": macro_decision,
        "worst_dataset": worst["dataset"],
        "worst_dataset_delta_accuracy": float(worst["delta_accuracy"]),
        "datasets_improved": sum(float(row["delta_accuracy"]) > 0 for row in output),
        "datasets_noninferior": sum(float(row["delta_accuracy_ci_low"]) > -noninferiority_margin for row in output),
        "datasets_failed": sum(row["decision"] == "C_not_supported" for row in output),
    }


__all__ = [
    "bind_oracle_correctness",
    "complementarity_metrics",
    "grouping_metrics",
    "materialize_arm_results",
]
