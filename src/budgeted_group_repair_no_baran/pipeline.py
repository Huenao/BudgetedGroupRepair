"""Phase 2.5 routeability and gated Phase 3 BGR execution."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .baran_reference import load_baran_records
from .data import load_dataset, normalize_for_match, read_jsonl, write_jsonl
from .experiment import ExperimentRunner, SafetyCapExceeded, _read_actions
from .group_gate import GroupUpliftGate
from .group_llm import GroupLLMJob, run_group_llm_batch
from .group_objective import build_group_objective
from .group_optimizer import select_queries
from .protocol import base_family
from .public_fd import fds_for_dataset, load_public_fds
from .run_state import read_json, write_json
from .sampling import SELECTED_DATASETS
from .verifier import GroupRepairVerifier, RankedRepairCandidate, VerifierConfig


FEATURE_FIELDS = (
    "suite",
    "dataset",
    "column",
    "dirty_length",
    "group_view",
    "group_size",
    "cohesion",
    "same_row",
    "same_column",
    "dirty_type_count",
    "baran_type_count",
    "baran_changed_share",
    "estimated_prompt_tokens",
    "estimated_total_tokens",
    "baran_candidate_count",
    "baran_top_support",
    "baran_support_margin",
    "baran_source_agreement",
)


def _feature_record(
    action: Any,
    cell: Any,
    baran: Mapping[str, Any],
) -> dict[str, Any]:
    group = action.group_features
    return {
        "suite": action.suite,
        "dataset": action.dataset,
        "column": str(cell.column),
        "dirty_length": len(str(cell.dirty_value)),
        "group_view": action.group_view,
        "group_size": action.group_size,
        "cohesion": float(group.get("cohesion", 1.0 if action.group_size == 1 else 0.0)),
        "same_row": int(group.get("same_row", action.group_size == 1)),
        "same_column": int(group.get("same_column", action.group_size == 1)),
        "dirty_type_count": int(group.get("dirty_type_count", 1)),
        "baran_type_count": int(group.get("baran_type_count", 1)),
        "baran_changed_share": float(group.get("baran_changed_share", 0.0)),
        "estimated_prompt_tokens": action.estimated_prompt_tokens,
        "estimated_total_tokens": action.estimated_total_tokens,
        "baran_candidate_count": int(baran.get("candidate_count", 0) or 0),
        "baran_top_support": float(baran.get("top_candidate_support", 0.0) or 0.0),
        "baran_support_margin": float(baran.get("support_margin", 0.0) or 0.0),
        "baran_source_agreement": float(baran.get("source_agreement", 0.0) or 0.0),
    }


def _calibration_rows(runner: ExperimentRunner) -> list[dict[str, Any]]:
    if not runner._state().stage_completed("experiment2"):
        raise RuntimeError("routeability requires completed preliminary experiments")
    experiment1 = read_jsonl(runner.paths.records / "experiment1_cells.jsonl")
    experiment2 = read_jsonl(runner.paths.records / "experiment2_primary_cells.jsonl")
    observed = [*experiment1, *(row for row in experiment2 if row.get("arm") == "structured")]
    action_paths = (
        runner.paths.queries / "singleton_actions.jsonl",
        runner.paths.queries / "structured_group_actions.jsonl",
    )
    actions = {action.query_id: action for path in action_paths for action in _read_actions(path)}
    cell_map: dict[str, Any] = {}
    clean: dict[str, str] = {}
    baran_map: dict[str, Mapping[str, Any]] = {}
    for key in SELECTED_DATASETS:
        loaded = load_dataset(*key, runner.data_root)
        cell_map.update({str(cell.cell_id): cell for cell in loaded.safe_cells()})
        clean.update({str(cell.cell_id): cell.clean_value for cell in loaded.oracle_cells(include_annotations=False)})
        baran_map.update(
            {
                str(row["cell_id"]): row
                for row in load_baran_records(runner.paths.run_dir / "baran_reference", *key)
            }
        )
    rows: list[dict[str, Any]] = []
    for record in observed:
        action = actions[str(record["query_id"])]
        cell_id = str(record["cell_id"])
        baran = baran_map[cell_id]
        baran_prediction = str(baran.get("prediction", cell_map[cell_id].dirty_value))
        rows.append(
            {
                "cell_id": cell_id,
                "query_id": action.query_id,
                "family": base_family(action.dataset),
                "baran_correct": normalize_for_match(baran_prediction) == normalize_for_match(clean[cell_id]),
                "llm_correct": bool(record.get("correct")),
                "executable": bool(record.get("valid_prediction")),
                "features": _feature_record(action, cell_map[cell_id], baran),
            }
        )
    return rows


def run_routeability(runner: ExperimentRunner) -> dict[str, Any]:
    runner.assert_binding_current()
    gates = read_json(runner.paths.metrics / "decision_gates.json")
    if gates.get("complementarity_supported") is not True or gates.get("grouping_supported") is not True:
        raise RuntimeError("routeability is gated on supported complementarity and grouping")
    calibration = _calibration_rows(runner)
    predictions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    backends = tuple(runner.config["full_pipeline"]["gate_backends"])
    for backend in backends:
        for target_suite, target_dataset in SELECTED_DATASETS:
            target_family = base_family(target_dataset)
            train = [
                row
                for row in calibration
                if row["features"]["suite"] == "tableeg" and row["family"] != target_family
            ]
            test = [
                row
                for row in calibration
                if row["features"]["suite"] == target_suite
                and row["features"]["dataset"] == target_dataset
            ]
            if not train or not test:
                continue
            gate = GroupUpliftGate(
                str(backend),
                rho=float(runner.config["full_pipeline"]["harm_penalty_rho"]),
                gamma=float(runner.config["full_pipeline"]["uncertainty_penalty_gamma"]),
                random_state=int(runner.config["sample_seed"]),
            ).fit(
                [row["features"] for row in train],
                [row["baran_correct"] for row in train],
                [row["llm_correct"] for row in train],
                [row["executable"] for row in train],
                [row["family"] for row in train],
            )
            predicted = gate.predict_dicts([row["features"] for row in test])
            helpful = [int(row["executable"] and row["llm_correct"] and not row["baran_correct"]) for row in test]
            harmful = [int(row["executable"] and row["baran_correct"] and not row["llm_correct"]) for row in test]
            helpful_probability = [row["q_helpful"] for row in predicted]
            harmful_probability = [row["q_harmful"] for row in predicted]
            helpful_prevalence = sum(helpful) / len(helpful)
            helpful_auprc = (
                float(average_precision_score(helpful, helpful_probability))
                if len(set(helpful)) > 1
                else helpful_prevalence
            )
            harmful_auprc = (
                float(average_precision_score(harmful, harmful_probability))
                if len(set(harmful)) > 1
                else sum(harmful) / len(harmful)
            )
            summaries.append(
                {
                    "backend": backend,
                    "target_suite": target_suite,
                    "target_dataset": target_dataset,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "helpful_prevalence": helpful_prevalence,
                    "helpful_auprc": helpful_auprc,
                    "harmful_prevalence": sum(harmful) / len(harmful),
                    "harmful_auprc": harmful_auprc,
                    "helpful_brier": float(brier_score_loss(helpful, helpful_probability)),
                    "harmful_brier": float(brier_score_loss(harmful, harmful_probability)),
                }
            )
            for source, result in zip(test, predicted):
                predictions.append(
                    {
                        "backend": backend,
                        "target_suite": target_suite,
                        "target_dataset": target_dataset,
                        "cell_id": source["cell_id"],
                        "query_id": source["query_id"],
                        **result,
                    }
                )
    pd.DataFrame(predictions).to_csv(runner.paths.metrics / "routeability_predictions.csv", index=False)
    pd.DataFrame(summaries).to_csv(runner.paths.metrics / "routeability_by_dataset.csv", index=False)
    by_backend: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_backend[str(row["backend"])].append(row)
    macro = {
        backend: {
            "helpful_auprc": sum(float(row["helpful_auprc"]) for row in rows) / len(rows),
            "helpful_prevalence": sum(float(row["helpful_prevalence"]) for row in rows) / len(rows),
            "harmful_auprc": sum(float(row["harmful_auprc"]) for row in rows) / len(rows),
            "harmful_prevalence": sum(float(row["harmful_prevalence"]) for row in rows) / len(rows),
        }
        for backend, rows in by_backend.items()
    }
    supported = any(
        values["helpful_auprc"] > values["helpful_prevalence"]
        and values["harmful_auprc"] >= values["harmful_prevalence"]
        for values in macro.values()
    )
    result = {"supported": supported, "macro": macro, "rows": len(predictions)}
    write_json(runner.paths.metrics / "routeability_summary.json", result)
    gates["routeability_supported"] = supported
    gates["phase3_allowed"] = all(
        gates.get(key) is True
        for key in ("complementarity_supported", "grouping_supported", "routeability_supported")
    )
    write_json(runner.paths.metrics / "decision_gates.json", gates)
    runner._state().update_stage("routeability", "complete", **result)
    return result


def _fit_target_gate(
    runner: ExperimentRunner,
    calibration: Sequence[Mapping[str, Any]],
    target_dataset: str,
    backend: str,
) -> GroupUpliftGate:
    target_family = base_family(target_dataset)
    train = [
        row
        for row in calibration
        if row["features"]["suite"] == "tableeg" and row["family"] != target_family
    ]
    if not train:
        raise ValueError(f"empty Phase 3 training split for {target_dataset}")
    return GroupUpliftGate(
        backend,
        rho=float(runner.config["full_pipeline"]["harm_penalty_rho"]),
        gamma=float(runner.config["full_pipeline"]["uncertainty_penalty_gamma"]),
        random_state=int(runner.config["sample_seed"]),
    ).fit(
        [row["features"] for row in train],
        [row["baran_correct"] for row in train],
        [row["llm_correct"] for row in train],
        [row["executable"] for row in train],
        [row["family"] for row in train],
    )


def plan_bgr(runner: ExperimentRunner) -> dict[str, Any]:
    runner.assert_binding_current()
    gates = read_json(runner.paths.metrics / "decision_gates.json")
    if gates.get("phase3_allowed") is not True:
        raise RuntimeError("Phase 3 is blocked until all preliminary decision gates pass")
    calibration = _calibration_rows(runner)
    all_actions = _read_actions(runner.paths.queries / "all_candidate_actions.jsonl")
    baran_map: dict[str, Mapping[str, Any]] = {}
    cell_map: dict[str, Any] = {}
    for key in SELECTED_DATASETS:
        cell_map.update({str(cell.cell_id): cell for cell in load_dataset(*key, runner.data_root).safe_cells()})
        baran_map.update(
            {str(row["cell_id"]): row for row in load_baran_records(runner.paths.run_dir / "baran_reference", *key)}
        )
    selections: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for backend in runner.config["full_pipeline"]["gate_backends"]:
        for suite, dataset in SELECTED_DATASETS:
            actions = [action for action in all_actions if action.suite == suite and action.dataset == dataset]
            gate = _fit_target_gate(runner, calibration, dataset, str(backend))
            pair_sources = [
                (action, cell_id)
                for action in actions
                for cell_id in action.cell_ids
            ]
            feature_rows = [
                _feature_record(action, cell_map[cell_id], baran_map[cell_id])
                for action, cell_id in pair_sources
            ]
            predicted = gate.predict_dicts(feature_rows)
            objective_rows = []
            for (action, cell_id), values in zip(pair_sources, predicted):
                row = {
                    "backend": backend,
                    "suite": suite,
                    "dataset": dataset,
                    "query_id": action.query_id,
                    "cell_id": cell_id,
                    **values,
                }
                predictions.append(row)
                objective_rows.append(row)
            objective = build_group_objective(objective_rows)
            costs = {action.query_id: float(action.estimated_total_tokens) for action in actions}
            reference = sum(
                action.estimated_total_tokens for action in actions if action.arm == "singleton"
            )
            budget = round(reference * float(runner.config["full_pipeline"]["primary_budget_share"]))
            selection = select_queries(objective, costs, budget)
            selections.append(
                {
                    "backend": backend,
                    "suite": suite,
                    "dataset": dataset,
                    "singleton_reference_tokens": reference,
                    **selection.as_dict(),
                }
            )
    pd.DataFrame(predictions).to_csv(runner.paths.metrics / "bgr_pair_predictions.csv", index=False)
    write_jsonl(runner.paths.run_dir / "selections" / "bgr_selections.jsonl", selections)
    result = {
        "selection_rows": len(selections),
        "unique_selected_queries": len(
            {query_id for row in selections for query_id in row["selected_query_ids"]}
        ),
    }
    write_json(runner.paths.run_dir / "selections" / "bgr_plan.json", result)
    runner._state().update_stage("bgr_plan", "complete", **result)
    return result


def run_bgr(runner: ExperimentRunner, token_cap: int | None) -> dict[str, Any]:
    if not runner._state().stage_completed("bgr_plan"):
        plan_bgr(runner)
    selections = read_jsonl(runner.paths.run_dir / "selections" / "bgr_selections.jsonl")
    selected_ids = {str(query_id) for row in selections for query_id in row["selected_query_ids"]}
    action_by_query = {
        action.query_id: action for action in _read_actions(runner.paths.queries / "all_candidate_actions.jsonl")
    }
    actions = [action_by_query[query_id] for query_id in sorted(selected_ids)]
    already = runner._existing_conservative_tokens()
    pending_actions = [
        action
        for action in actions
        if not runner.response_reusable(action, "bgr_selected_union")
    ]
    reservation = sum(action.estimated_total_tokens for action in pending_actions) * (
        int(runner.llm_config["max_retries"]) + 1
    )
    if token_cap is not None and already + reservation > int(token_cap):
        raise SafetyCapExceeded("Phase 3 selected-query reservation exceeds the total run token cap")
    runner.freeze_token_cap(token_cap)
    client = runner._client()
    jobs = [
        GroupLLMJob.from_action(
            action,
            metadata={
                "phase": "bgr_selected_union",
                "model_requested": str(runner.llm_config["model"]),
                "require_complete_response": False,
            },
        )
        for action in actions
    ]
    responses = run_group_llm_batch(client, jobs, runner.paths.llm / "shared")
    write_jsonl(runner.paths.llm / "bgr_responses.jsonl", responses)
    runner.write_cost_audit()
    response_by_query = {str(row["query_id"]): row for row in responses}
    pair_predictions = pd.read_csv(runner.paths.metrics / "bgr_pair_predictions.csv", keep_default_na=False)
    gain = {
        (str(row.backend), str(row.query_id), str(row.cell_id)): float(row.conservative_uplift)
        for row in pair_predictions.itertuples(index=False)
    }
    fd_registry = load_public_fds(runner.fd_path)
    final_records: list[dict[str, Any]] = []
    for selection in selections:
        backend = str(selection["backend"])
        key = (str(selection["suite"]), str(selection["dataset"]))
        loaded = load_dataset(*key, runner.data_root)
        safe_cells = loaded.safe_cells()
        safe_by_id = {str(cell.cell_id): cell for cell in safe_cells}
        baran = {
            str(row["cell_id"]): row
            for row in load_baran_records(runner.paths.run_dir / "baran_reference", *key)
        }
        verifier = GroupRepairVerifier(
            loaded.dirty,
            safe_cells,
            fds_for_dataset(fd_registry, *key),
            VerifierConfig(),
        )
        selected_query_ids = tuple(str(value) for value in selection["selected_query_ids"])
        for cell_id, cell in safe_by_id.items():
            candidates: list[RankedRepairCandidate] = []
            for query_id in selected_query_ids:
                action = action_by_query[query_id]
                if cell_id not in action.cell_ids:
                    continue
                response = response_by_query.get(query_id, {})
                items = response.get("items") if isinstance(response.get("items"), list) else []
                item = (
                    next(
                        (
                            value
                            for value in items
                            if str(value.get("cell_id")) == cell_id
                        ),
                        None,
                    )
                    if response.get("status") == "success"
                    and response.get("model_matches_request", True)
                    else None
                )
                if item is not None:
                    item = {**item, "parse_status": response.get("parse_status", "")}
                candidates.append(
                    RankedRepairCandidate(
                        query_id=query_id,
                        item=item or {},
                        conservative_uplift=gain.get((backend, query_id, cell_id), 0.0),
                        cost=action.estimated_total_tokens,
                        group_size=action.group_size,
                    )
                )
            arbitration = verifier.arbitrate(cell, baran[cell_id], candidates)
            final_records.append(
                {
                    "backend": backend,
                    "suite": key[0],
                    "dataset": key[1],
                    "cell_id": cell_id,
                    **arbitration.as_dict(),
                }
            )
    write_jsonl(runner.paths.records / "bgr_final_cells.jsonl", final_records)
    clean: dict[str, str] = {}
    dirty: dict[str, str] = {}
    for key in SELECTED_DATASETS:
        for cell in load_dataset(*key, runner.data_root).oracle_cells(include_annotations=False):
            clean[str(cell.cell_id)] = cell.clean_value
            dirty[str(cell.cell_id)] = cell.dirty_value
    evaluated: list[dict[str, Any]] = []
    for record in final_records:
        cell_id = str(record["cell_id"])
        prediction = str(record.get("final_prediction", ""))
        proposed = normalize_for_match(prediction) != normalize_for_match(dirty[cell_id])
        evaluated.append(
            {
                **record,
                "valid_proposal": proposed,
                "correct": proposed
                and normalize_for_match(prediction) == normalize_for_match(clean[cell_id]),
            }
        )
    write_jsonl(runner.paths.records / "bgr_final_evaluated_cells.jsonl", evaluated)
    metrics: list[dict[str, Any]] = []
    for backend in sorted({str(row["backend"]) for row in evaluated}):
        backend_rows = [row for row in evaluated if str(row["backend"]) == backend]
        for dataset in sorted({str(row["dataset"]) for row in backend_rows}):
            rows = [row for row in backend_rows if str(row["dataset"]) == dataset]
            proposed = sum(bool(row["valid_proposal"]) for row in rows)
            correct = sum(bool(row["correct"]) for row in rows)
            precision = correct / proposed if proposed else math.nan
            recall = correct / len(rows)
            metrics.append(
                {
                    "backend": backend,
                    "dataset": dataset,
                    "N": len(rows),
                    "valid_proposals": proposed,
                    "correct": correct,
                    "accuracy": recall,
                    "precision": precision,
                    "recall": recall,
                    "f1": (
                        2 * precision * recall / (precision + recall)
                        if precision + recall > 0
                        else 0.0
                    ),
                }
            )
    pd.DataFrame(metrics).to_csv(runner.paths.metrics / "bgr_by_dataset.csv", index=False)
    result = {
        "queries": len(actions),
        "final_records": len(final_records),
        "metric_rows": len(metrics),
    }
    runner._state().update_stage("bgr", "complete", **result)
    return result


__all__ = ["plan_bgr", "run_bgr", "run_routeability"]
