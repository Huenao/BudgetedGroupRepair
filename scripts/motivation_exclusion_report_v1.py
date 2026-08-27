#!/usr/bin/env python3
"""Build an offline sensitivity report excluding ``source/movies_1``.

The frozen nine-dataset run is an immutable input.  This script filters its
finalized cell ledgers and exact logical-to-physical mapping, recomputes every
statistic, confidence interval, Holm family, cost total, and figure for the
retained eight datasets, then atomically publishes a separate derived run.

No model client is imported or constructed and no network access is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "motivation-evidence-derived-exclusion-v1"
PARENT_RUN_ID = "motivation_evidence_deepseek_v4_flash_20260822_full"
DERIVED_RUN_ID = (
    "motivation_evidence_deepseek_v4_flash_20260822_full_excluding_source_movies_1"
)
EXCLUDED = ("source", "movies_1")
EXPECTED = {
    "datasets": 8,
    "complementarity_cells": 14_523,
    "group_cell_incidences": 86_060,
    "logical_queries": 64_979,
    "physical_queries": 64_768,
    "complementarity_metric_rows": 12,
    "group_metric_rows": 74,
    "group_cost_rows": 203,
    "group_transition_rows": 222,
    "statistical_test_rows": 230,
    "observed_input_tokens": 151_288_011,
    "observed_output_tokens": 18_897_581,
    "observed_total_tokens": 170_185_592,
    "estimated_total_tokens": 175_070_696,
    "attempts": 66_172,
    "retries": 1_404,
    "unknown_usage_attempts": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _source_files(parent: Path) -> dict[str, Path]:
    return {
        "run_manifest": parent / "run_manifest.json",
        "finalization_summary": parent / "metrics" / "finalization_summary.json",
        "plan_summary": parent / "evidence" / "plan_summary.json",
        "config": parent / "configs" / "motivation_evidence.json",
        "logical_queries": parent / "evidence" / "logical_queries.jsonl",
        "complementarity": parent / "records" / "complementarity_cell_outcomes.csv",
        "group": parent / "records" / "group_cell_outcomes.csv",
        "api_cost": parent / "metrics" / "api_cost_audit.csv",
        "parent_complementarity_metrics": (
            parent / "metrics" / "complementarity_by_dataset.csv"
        ),
        "parent_group_metrics": parent / "metrics" / "group_by_dataset_view_size.csv",
    }


def _verify_parent(parent: Path) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    files = _source_files(parent)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing parent artifacts: {missing}")
    manifest = _load_json(files["run_manifest"])
    finalization = _load_json(files["finalization_summary"])
    plan = _load_json(files["plan_summary"])
    config = _load_json(files["config"])
    if parent.name != PARENT_RUN_ID or manifest.get("run_id") != PARENT_RUN_ID:
        raise ValueError("unexpected parent run identity")
    if not manifest.get("stages", {}).get("finalize", {}).get("completed"):
        raise ValueError("parent run is not finalized")
    actual = {name: _sha256(path) for name, path in files.items()}
    recorded_final = finalization.get("artifact_sha256", {})
    for name in ("complementarity", "group"):
        if actual[name] != recorded_final.get(name):
            raise RuntimeError(f"parent finalized {name} hash drift")
    if actual["api_cost"] != recorded_final.get("api_cost_audit"):
        raise RuntimeError("parent finalized API cost hash drift")
    if actual["logical_queries"] != plan.get("artifact_sha256", {}).get("logical_queries"):
        raise RuntimeError("parent logical-query hash drift")
    datasets = [
        (str(row["suite"]), str(row["dataset"]))
        for row in config.get("formal_datasets", [])
    ]
    retained = [dataset for dataset in datasets if dataset != EXCLUDED]
    if len(datasets) != 9 or len(retained) != 8 or EXCLUDED not in datasets:
        raise ValueError("parent dataset registry does not match the exclusion protocol")
    return files, actual, {
        "manifest": manifest,
        "finalization": finalization,
        "plan": plan,
        "config": config,
        "datasets": datasets,
        "retained": retained,
    }


def _filter_inputs(
    files: Mapping[str, Path], stage: Path, retained: Sequence[tuple[str, str]]
) -> dict[str, Any]:
    retained_set = set(retained)
    complement_fields, complement_all = _read_csv(files["complementarity"])
    complement = [
        row
        for row in complement_all
        if (str(row["suite"]), str(row["dataset"])) in retained_set
    ]
    group_fields, group_all = _read_csv(files["group"])
    group = [
        row
        for row in group_all
        if (str(row["suite"]), str(row["dataset"])) in retained_set
    ]
    if len(complement) != EXPECTED["complementarity_cells"]:
        raise RuntimeError(f"unexpected retained complementarity rows: {len(complement)}")
    if len(group) != EXPECTED["group_cell_incidences"]:
        raise RuntimeError(f"unexpected retained group rows: {len(group)}")
    if any((row["suite"], row["dataset"]) == EXCLUDED for row in complement + group):
        raise RuntimeError("excluded dataset leaked into retained cell ledgers")
    complement_ids = {str(row["cell_id"]) for row in complement}
    if len(complement_ids) != len(complement):
        raise RuntimeError("retained complementarity grain is not unique")
    group_grain = {
        (
            str(row["suite"]),
            str(row["dataset"]),
            str(row["source_view"]),
            int(row["group_size"]),
            str(row["cell_id"]),
        )
        for row in group
    }
    if len(group_grain) != len(group):
        raise RuntimeError("retained group grain is not unique")
    conditions = {(suite, dataset, view, size) for suite, dataset, view, size, _ in group_grain}
    if len(conditions) != 48:
        raise RuntimeError(f"expected 48 retained dataset conditions, observed {len(conditions)}")

    logical_output = stage / "evidence" / "logical_queries.jsonl"
    logical_output.parent.mkdir(parents=True, exist_ok=True)
    logical_temporary = logical_output.with_name(f".{logical_output.name}.{os.getpid()}.tmp")
    retained_logical = 0
    physical_counts: Counter[str] = Counter()
    physical_owners: dict[str, set[tuple[str, str]]] = defaultdict(set)
    excluded_physical: set[str] = set()
    singleton_cells: set[str] = set()
    with files["logical_queries"].open("r", encoding="utf-8") as source, logical_temporary.open(
        "w", encoding="utf-8"
    ) as destination:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            owner = (str(row["suite"]), str(row["dataset"]))
            physical_id = str(row["physical_query_id"])
            physical_owners[physical_id].add(owner)
            if owner == EXCLUDED:
                excluded_physical.add(physical_id)
                continue
            if owner not in retained_set:
                raise RuntimeError(f"unexpected logical-query owner at line {line_number}: {owner}")
            retained_logical += 1
            physical_counts[physical_id] += 1
            if str(row["arm"]) == "singleton":
                cells = list(row["ordered_cell_ids"])
                if len(cells) != 1:
                    raise RuntimeError("singleton logical query does not contain one cell")
                singleton_cells.add(str(cells[0]))
            destination.write(line if line.endswith("\n") else line + "\n")
        destination.flush()
        os.fsync(destination.fileno())
    logical_temporary.replace(logical_output)
    if retained_logical != EXPECTED["logical_queries"]:
        raise RuntimeError(f"unexpected retained logical calls: {retained_logical}")
    if singleton_cells != complement_ids:
        raise RuntimeError("retained singleton logical universe differs from complementarity cells")
    mixed_ownership = {
        physical_id: sorted(owners)
        for physical_id, owners in physical_owners.items()
        if len(owners) != 1
    }
    if mixed_ownership:
        raise RuntimeError(f"physical requests span datasets: {len(mixed_ownership)}")
    retained_physical = set(physical_counts)
    overlap = retained_physical & excluded_physical
    if overlap:
        raise RuntimeError(f"retained and excluded physical requests overlap: {len(overlap)}")
    if len(retained_physical) != EXPECTED["physical_queries"]:
        raise RuntimeError(f"unexpected retained physical calls: {len(retained_physical)}")

    api_fields, api_all = _read_csv(files["api_cost"])
    api_by_id = {str(row["physical_query_id"]): row for row in api_all}
    if len(api_by_id) != len(api_all):
        raise RuntimeError("parent API audit has duplicate physical IDs")
    if not retained_physical <= set(api_by_id):
        raise RuntimeError("retained logical requests are missing from API cost audit")
    api = []
    for row in api_all:
        physical_id = str(row["physical_query_id"])
        if physical_id not in retained_physical:
            continue
        retained_row = dict(row)
        retained_row["logical_query_mappings"] = str(physical_counts[physical_id])
        api.append(retained_row)
    if len(api) != EXPECTED["physical_queries"]:
        raise RuntimeError(f"unexpected retained API rows: {len(api)}")
    if sum(int(row["logical_query_mappings"]) for row in api) != retained_logical:
        raise RuntimeError("retained API logical mapping count mismatch")

    observed_input = sum(int(row["observed_input_tokens"]) for row in api)
    observed_output = sum(int(row["observed_output_tokens"]) for row in api)
    observed_total = sum(int(row["observed_total_tokens"]) for row in api)
    estimated_total = sum(int(row["estimated_total_tokens"]) for row in api)
    attempts = sum(int(row["attempts"]) for row in api)
    retries = attempts - len(api)
    unknown_usage = sum(int(row["unknown_usage_attempts"]) for row in api)
    exact_totals = {
        "observed_input_tokens": observed_input,
        "observed_output_tokens": observed_output,
        "observed_total_tokens": observed_total,
        "estimated_total_tokens": estimated_total,
        "attempts": attempts,
        "retries": retries,
        "unknown_usage_attempts": unknown_usage,
    }
    for field, value in exact_totals.items():
        if value != EXPECTED[field]:
            raise RuntimeError(f"unexpected retained {field}: {value}")
    if observed_input + observed_output != observed_total:
        raise RuntimeError("retained API token identity failed")
    if any(str(row["model_returned"]) != "deepseek-v4-flash" for row in api):
        raise RuntimeError("provider model identity drift in retained API rows")
    if any(str(row["historical_imported_response"]).strip().lower() != "false" for row in api):
        raise RuntimeError("historical response detected in retained API rows")

    _write_csv(
        stage / "records" / "complementarity_cell_outcomes.csv",
        complement_fields,
        complement,
    )
    _write_csv(stage / "records" / "group_cell_outcomes.csv", group_fields, group)
    _write_csv(stage / "metrics" / "api_cost_audit.csv", api_fields, api)
    return {
        "complementarity_rows": len(complement),
        "group_rows": len(group),
        "conditions": len(conditions),
        "logical_queries": retained_logical,
        "physical_queries": len(retained_physical),
        "deduplicated_logical_queries": retained_logical - len(retained_physical),
        "mixed_dataset_physical_queries": 0,
        "retained_excluded_physical_overlap": 0,
        "exact_cost": exact_totals,
    }


def _dynamic_group_transitions(reporting: Any, metric_inputs: Sequence[Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for metric, rows in metric_inputs:
        for contrast in reporting.CONTRASTS:
            both, treatment_only, baseline_only, neither = reporting._transition_counts(
                rows, contrast
            )
            family = "descriptive"
            if metric["scope"] == "micro" and metric["source_view"] in reporting.PRIMARY_VIEWS:
                family = "group_primary_18"
            elif metric["scope"] == "dataset":
                family = "group_secondary_dataset_144"
            cluster_keys = {
                "structured_minus_singleton": "row_cluster×structured_physical_query_id",
                "random_minus_singleton": "row_cluster×random_physical_query_id",
                "structured_minus_random": (
                    "row_cluster×structured_physical_query_id×random_physical_query_id"
                ),
            }[contrast]
            if bool(metric["cell_cluster_in_ci"]):
                cluster_keys += "×cell_id"
            output.append(
                {
                    "experiment": "group",
                    "scope": metric["scope"],
                    "suite": metric["suite"],
                    "dataset": metric["dataset"],
                    "source_view": metric["source_view"],
                    "group_size": metric["group_size"],
                    "contrast": contrast,
                    "N": len(rows),
                    "both_correct": both,
                    "treatment_only_correct": treatment_only,
                    "baseline_only_correct": baseline_only,
                    "both_wrong": neither,
                    "effect": metric[contrast],
                    "effect_ci_low": metric[f"{contrast}_ci_low"],
                    "effect_ci_high": metric[f"{contrast}_ci_high"],
                    "mcnemar_p": (
                        reporting.exact_mcnemar(treatment_only, baseline_only)
                        if "macro" not in str(metric["scope"])
                        else math.nan
                    ),
                    "holm_adjusted_p": math.nan,
                    "test_family": family,
                    "cluster_keys": cluster_keys,
                }
            )
    for family, expected in (
        ("group_primary_18", 18),
        ("group_secondary_dataset_144", 144),
    ):
        members = [row for row in output if row["test_family"] == family]
        if len(members) != expected:
            raise RuntimeError(f"{family}: expected {expected}, observed {len(members)}")
        keyed = {
            (
                f"{row['suite']}/{row['dataset']}/{row['source_view']}/"
                f"{row['group_size']}/{row['contrast']}"
            ): float(row["mcnemar_p"])
            for row in members
        }
        adjusted = reporting.holm_adjust(keyed)
        for row in members:
            key = (
                f"{row['suite']}/{row['dataset']}/{row['source_view']}/"
                f"{row['group_size']}/{row['contrast']}"
            )
            row["holm_adjusted_p"] = adjusted[key]
    return output


def _dynamic_complementarity_tests(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    for row in metrics:
        if row["scope"] != "dataset":
            continue
        tests.append(
            {
                "experiment": "complementarity",
                "scope": "dataset",
                "suite": row["suite"],
                "dataset": row["dataset"],
                "source_view": "",
                "group_size": "",
                "contrast": "singleton_llm_minus_baran",
                "N": row["N"],
                "both_correct": row["n11"],
                "treatment_only_correct": row["n01"],
                "baseline_only_correct": row["n10"],
                "both_wrong": row["n00"],
                "effect": row["llm_minus_baran"],
                "effect_ci_low": row["llm_minus_baran_ci_low"],
                "effect_ci_high": row["llm_minus_baran_ci_high"],
                "mcnemar_p": row["mcnemar_p"],
                "holm_adjusted_p": row["mcnemar_p_holm"],
                "test_family": "complementarity_dataset_8",
                "cluster_keys": "row_cluster",
            }
        )
    if len(tests) != 8:
        raise RuntimeError(f"expected eight complementarity tests, observed {len(tests)}")
    return tests


def _replace_stale_labels(value: Any) -> Any:
    if isinstance(value, str):
        return (
            value.replace("the frozen nine datasets", "the retained eight datasets")
            .replace("all nine frozen datasets", "all eight retained datasets")
            .replace("all nine datasets", "all eight retained datasets")
            .replace("nine-dataset", "eight-dataset")
            .replace("nine dataset", "eight dataset")
        )
    if isinstance(value, list):
        return [_replace_stale_labels(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_stale_labels(item) for key, item in value.items()}
    return value


def _run_statistics(
    project_root: Path,
    stage: Path,
    retained: Sequence[tuple[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    src = project_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from budgeted_group_repair_no_baran import motivation_reporting as reporting

    original = {
        "datasets": reporting.SELECTED_DATASETS,
        "transitions": reporting._build_group_transitions,
        "complementarity_tests": reporting._complementarity_tests,
        "figure": reporting._figure,
        "markdown": reporting._report_markdown,
    }
    reporting.SELECTED_DATASETS = tuple(retained)
    reporting._build_group_transitions = lambda metric_inputs: _dynamic_group_transitions(
        reporting, metric_inputs
    )
    reporting._complementarity_tests = _dynamic_complementarity_tests
    # Publication-quality derived outputs are written below; prevent the formal
    # hard-coded nine-dataset presentation layer from emitting stale labels.
    reporting._figure = lambda *args, **kwargs: None
    reporting._report_markdown = lambda *args, **kwargs: "# Derived report pending\n"
    try:
        result = reporting.build_motivation_report(
            stage,
            bootstrap_replicates=int(config["bootstrap_replicates"]),
            bootstrap_seed=int(config["bootstrap_seed"]),
            confidence=float(config["confidence_level"]),
        )
    finally:
        reporting.SELECTED_DATASETS = original["datasets"]
        reporting._build_group_transitions = original["transitions"]
        reporting._complementarity_tests = original["complementarity_tests"]
        reporting._figure = original["figure"]
        reporting._report_markdown = original["markdown"]

    complement_path = stage / "metrics" / "complementarity_by_dataset.csv"
    fields, rows = _read_csv(complement_path)
    for row in rows:
        if row.get("scope") == "macro":
            row["quadrant_count_aggregation"] = (
                "pooled_cell_counts; rate fields use the unweighted eight-dataset macro"
            )
    _write_csv(complement_path, fields, rows)

    complement_summary_path = stage / "metrics" / "complementarity_summary.json"
    complement_summary = _replace_stale_labels(_load_json(complement_summary_path))
    complement_summary["schema_version"] = SCHEMA_VERSION
    complement_summary["aggregation"]["dataset_order"] = [list(item) for item in retained]
    complement_summary["aggregation"]["macro"] = (
        "unweighted mean over the retained eight datasets"
    )
    complement_summary["holm_family"] = "eight dataset-level paired McNemar tests"
    _atomic_json(complement_summary_path, complement_summary)

    group_summary_path = stage / "metrics" / "group_summary.json"
    group_summary = _replace_stale_labels(_load_json(group_summary_path))
    group_summary["schema_version"] = SCHEMA_VERSION
    group_summary["aggregation"]["dataset_order"] = [list(item) for item in retained]
    group_summary["aggregation"]["panel_b"] = (
        "unweighted macro over the retained eight datasets"
    )
    group_summary["secondary_dataset_holm_tests"] = 144
    _atomic_json(group_summary_path, group_summary)
    normalized_result = dict(result)
    normalized_result["schema_version"] = SCHEMA_VERSION
    normalized_result["run_dir"] = "."
    normalized_result["complementarity_cells"] = EXPECTED["complementarity_cells"]
    normalized_result["group_cell_incidences"] = EXPECTED["group_cell_incidences"]
    normalized_result["physical_union_calls"] = EXPECTED["physical_queries"]
    normalized_result["complementarity_metric_rows"] = EXPECTED[
        "complementarity_metric_rows"
    ]
    normalized_result["group_metric_rows"] = EXPECTED["group_metric_rows"]
    normalized_result["primary_holm_tests"] = 18
    normalized_result["secondary_dataset_holm_tests"] = 144
    normalized_result["outputs"] = {
        name: str(Path(path).resolve().relative_to(stage.resolve()))
        for name, path in dict(result["outputs"]).items()
    }
    return normalized_result


def _load_visualization_module(project_root: Path) -> Any:
    path = project_root / "scripts" / "motivation_visualization_amendment_v1.py"
    spec = importlib.util.spec_from_file_location("motivation_visualization_amendment_v1", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_figure(project_root: Path, stage: Path) -> tuple[float, float]:
    viz = _load_visualization_module(project_root)
    micro, macro_rows = viz._load_inputs(stage)
    viz.SAVING_LEGEND_VALUES = (0.19, 0.28, 0.34)
    # The retained-cohort R2 point is closer to the semantic x-axis boundary
    # than in the nine-dataset figure, so keep its direct label inside the axes.
    viz.LABEL_OFFSETS[("semantic", "random", 2)] = (7, 8)
    pdf_path = stage / "figures" / "introduction_motivation.pdf"
    svg_path = stage / "figures" / "introduction_motivation.svg"
    with viz.plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": SCHEMA_VERSION,
        }
    ):
        figure = viz.plt.figure(figsize=(17.2, 5.5), layout="constrained")
        outer = figure.add_gridspec(1, 3, width_ratios=(1.08, 1.95, 1.18))
        panel_a = figure.add_subplot(outer[0, 0])
        panel_b = figure.add_subfigure(outer[0, 1])
        panel_b.suptitle(
            "B  Group benefit–interference (8-dataset macro; Movies excluded)",
            x=0.0,
            ha="left",
            fontweight="bold",
        )
        middle = panel_b.add_gridspec(2, 2, height_ratios=(1.0, 0.24))
        axes = {
            "pattern": panel_b.add_subplot(middle[0, 0]),
            "semantic": panel_b.add_subplot(middle[0, 1]),
        }
        legend_axis = panel_b.add_subplot(middle[1, :])
        panel_c = figure.add_subplot(outer[0, 2])
        viz._panel_a(panel_a, micro)
        saving_range = viz._panel_b(axes, legend_axis, macro_rows)
        viz._panel_c(panel_c)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_tmp = pdf_path.with_name(f".{pdf_path.name}.{os.getpid()}.tmp")
        svg_tmp = svg_path.with_name(f".{svg_path.name}.{os.getpid()}.tmp")
        figure.savefig(
            pdf_tmp,
            format="pdf",
            bbox_inches="tight",
            metadata={"Creator": SCHEMA_VERSION, "CreationDate": None, "ModDate": None},
        )
        figure.savefig(
            svg_tmp,
            format="svg",
            bbox_inches="tight",
            metadata={"Creator": SCHEMA_VERSION, "Date": None},
        )
        viz.plt.close(figure)
        pdf_tmp.replace(pdf_path)
        svg_tmp.replace(svg_path)
    return float(saving_range[0]), float(saving_range[1])


def _float(row: Mapping[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def _p(value: str) -> str:
    number = float(value)
    if number < 0.0001:
        return "<0.0001"
    return f"{number:.4f}"


def _report_markdown(
    stage: Path,
    files: Mapping[str, Path],
    filter_summary: Mapping[str, Any],
    saving_range: tuple[float, float],
) -> str:
    _, comp = _read_csv(stage / "metrics" / "complementarity_by_dataset.csv")
    _, group = _read_csv(stage / "metrics" / "group_by_dataset_view_size.csv")
    _, transitions = _read_csv(stage / "metrics" / "group_paired_transitions.csv")
    _, costs = _read_csv(stage / "metrics" / "group_costs.csv")
    _, parent_comp = _read_csv(files["parent_complementarity_metrics"])
    _, parent_group = _read_csv(files["parent_group_metrics"])
    comp_micro = next(row for row in comp if row["scope"] == "micro")
    comp_macro = next(row for row in comp if row["scope"] == "macro")
    parent_micro = next(row for row in parent_comp if row["scope"] == "micro")
    parent_macro = next(row for row in parent_comp if row["scope"] == "macro")
    group_macro = [
        row
        for row in group
        if row["scope"] == "macro" and row["source_view"] in ("pattern", "semantic")
    ]
    group_micro = [
        row
        for row in group
        if row["scope"] == "micro" and row["source_view"] in ("pattern", "semantic")
    ]
    group_macro.sort(key=lambda row: (("pattern", "semantic").index(row["source_view"]), int(row["group_size"])))
    group_micro.sort(key=lambda row: (("pattern", "semantic").index(row["source_view"]), int(row["group_size"])))
    primary = [row for row in transitions if row["test_family"] == "group_primary_18"]
    run_cost = next(row for row in costs if row["scope"] == "run_physical_union")
    micro_costs = [
        row
        for row in costs
        if row["scope"] == "micro"
        and row["source_view"] in ("pattern", "semantic")
        and row["arm"] in ("structured", "random")
    ]
    micro_costs.sort(
        key=lambda row: (
            ("pattern", "semantic").index(row["source_view"]),
            int(row["group_size"]),
            ("structured", "random").index(row["arm"]),
        )
    )
    significant_primary = sum(float(row["holm_adjusted_p"]) < 0.05 for row in primary)
    complement_tests = [row for row in comp if row["scope"] == "dataset"]
    significant_complement = sum(float(row["mcnemar_p_holm"]) < 0.05 for row in complement_tests)

    lines = [
        "# Introduction Motivation Evidence — Sensitivity Analysis Without Movies",
        "",
        "This is a complete **offline derived analysis** of the frozen DeepSeek V4 Flash run, "
        "excluding exactly `source/movies_1`. It reuses finalized predictions and makes "
        "**zero new LLM/API/network calls**. It is a sensitivity analysis, not a replacement "
        "for the registered nine-dataset primary result.",
        "",
        "[Figure PDF](../figures/introduction_motivation.pdf) · "
        "[Figure SVG](../figures/introduction_motivation.svg) · "
        "[Derived provenance](../provenance/derived_manifest.json)",
        "",
        "## Main result",
        "",
        f"After removing Movies, the analysis retains **{int(comp_micro['N']):,} error cells**, "
        f"**{int(filter_summary['logical_queries']):,} logical queries**, and "
        f"**{int(filter_summary['physical_queries']):,} unique physical responses**. "
        f"Baran's micro accuracy is **{_float(comp_micro, 'baran_accuracy'):.2%}** and the "
        f"singleton LLM's is **{_float(comp_micro, 'llm_accuracy'):.2%}**. Their offline "
        f"oracle-union opportunity is **{_float(comp_micro, 'oracle_union_upper_bound'):.2%}**: "
        f"LLM-only salvages account for **{_float(comp_micro, 'llm_salvage_opportunity'):.2%}** "
        f"of evaluated cells. Overwrites—cells that Baran gets right but the LLM gets "
        f"wrong—account for **{_float(comp_micro, 'overwrite_risk'):.2%}** of all evaluated "
        "cells.",
        "",
        "The group result changes materially in this sensitivity slice. On the unweighted "
        "eight-dataset macro, structured grouping is close to the singleton baseline: "
        "G−S ranges from about −0.52 percentage points to +0.13 points. Structured groups "
        "still outperform matched random groups in every view/size condition, while saving "
        f"{saving_range[0]:.1%}–{saving_range[1]:.1%} of tokens at the dataset-macro level.",
        "",
        "## Sensitivity relative to the registered nine-dataset report",
        "",
        "| Estimand | Nine datasets | Without Movies | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for field, label in (
        ("baran_accuracy", "Micro Baran accuracy"),
        ("llm_accuracy", "Micro singleton-LLM accuracy"),
        ("oracle_union_upper_bound", "Micro oracle-union upper bound"),
        ("llm_salvage_opportunity", "Micro LLM-only salvage"),
        ("overwrite_risk", "Micro overwrite risk"),
    ):
        before = _float(parent_micro, field)
        after = _float(comp_micro, field)
        lines.append(f"| {label} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")
    for field, label in (
        ("baran_accuracy", "Macro Baran accuracy"),
        ("llm_accuracy", "Macro singleton-LLM accuracy"),
        ("oracle_union_upper_bound", "Macro oracle-union upper bound"),
    ):
        before = _float(parent_macro, field)
        after = _float(comp_macro, field)
        lines.append(f"| {label} | {before:.4f} | {after:.4f} | {after - before:+.4f} |")

    lines.extend(
        [
            "",
            "Direct interpretation: Movies was not the sole source of complementarity, but it "
            "was an important source of the apparent average harm from grouping. Removing it "
            "raises both micro accuracies and makes the macro structured-group effect roughly "
            "neutral instead of consistently negative.",
            "",
            "## Experiment 1 — Baran versus singleton LLM",
            "",
            "| Population | N | Baran acc. | LLM acc. | Oracle union | LLM-only salvage | Overwrite | Holm p |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in comp:
        label = (
            f"{row['suite']}/{row['dataset']}"
            if row["scope"] == "dataset"
            else f"{row['scope']}:{row['suite']}/{row['dataset']}"
        )
        holm = _p(row["mcnemar_p_holm"]) if row["scope"] == "dataset" else "—"
        lines.append(
            f"| {label} | {int(row['N']):,} | {_float(row, 'baran_accuracy'):.4f} | "
            f"{_float(row, 'llm_accuracy'):.4f} | "
            f"{_float(row, 'oracle_union_upper_bound'):.4f} | "
            f"{_float(row, 'llm_salvage_opportunity'):.4f} | "
            f"{_float(row, 'overwrite_risk'):.4f} | {holm} |"
        )
    lines.extend(
        [
            "",
            f"The four paired micro quadrants are n11={int(comp_micro['n11']):,}, "
            f"n10={int(comp_micro['n10']):,}, n01={int(comp_micro['n01']):,}, and "
            f"n00={int(comp_micro['n00']):,}. {significant_complement}/8 dataset-level exact "
            "McNemar tests remain significant after Holm correction.",
            "",
            "## Experiment 2 — Structured groups, random groups, and singleton control",
            "",
            "Panel B and the first table use the unweighted macro over the fixed eight retained "
            "datasets. The second table reports cell-weighted micro results.",
            "",
            "### Eight-dataset macro",
            "",
            "| Condition | Coverage | S acc. | G acc. | R acc. | G rescue | G interference | G−S | R−S | G−R | G token saving |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_macro:
        lines.append(
            f"| {row['source_view']}/k={row['group_size']} | {_float(row, 'coverage_rate'):.4f} | "
            f"{_float(row, 'singleton_accuracy'):.4f} | {_float(row, 'structured_accuracy'):.4f} | "
            f"{_float(row, 'random_accuracy'):.4f} | {_float(row, 'structured_rescue_rate'):.4f} | "
            f"{_float(row, 'structured_interference_rate'):.4f} | "
            f"{_float(row, 'structured_minus_singleton'):+.4f} | "
            f"{_float(row, 'random_minus_singleton'):+.4f} | "
            f"{_float(row, 'structured_minus_random'):+.4f} | "
            f"{_float(row, 'structured_token_saving'):.4f} |"
        )
    lines.extend(
        [
            "",
            "### Cell-weighted micro",
            "",
            "| Condition | N | S acc. | G acc. | R acc. | G rescue | G interference | G−S | R−S | G−R |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_micro:
        lines.append(
            f"| {row['source_view']}/k={row['group_size']} | {int(row['eligible_cell_incidences']):,} | "
            f"{_float(row, 'singleton_accuracy'):.4f} | {_float(row, 'structured_accuracy'):.4f} | "
            f"{_float(row, 'random_accuracy'):.4f} | {_float(row, 'structured_rescue_rate'):.4f} | "
            f"{_float(row, 'structured_interference_rate'):.4f} | "
            f"{_float(row, 'structured_minus_singleton'):+.4f} | "
            f"{_float(row, 'random_minus_singleton'):+.4f} | "
            f"{_float(row, 'structured_minus_random'):+.4f} |"
        )

    lines.extend(
        [
            "",
            "### Primary paired tests (18-test Holm family)",
            "",
            "| Condition | Contrast | Effect | Clustered 95% CI | Holm p |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in primary:
        lines.append(
            f"| {row['source_view']}/k={row['group_size']} | {row['contrast']} | "
            f"{_float(row, 'effect'):+.4f} | [{_float(row, 'effect_ci_low'):+.4f}, "
            f"{_float(row, 'effect_ci_high'):+.4f}] | {_p(row['holm_adjusted_p'])} |"
        )
    lines.extend(
        [
            "",
            f"{significant_primary}/18 micro contrasts are significant after Holm correction. "
            "The 144 dataset-level contrasts form a separate secondary family. Clustered "
            "percentile intervals account for dirty-row and physical-query dependence; the "
            "reported exact McNemar p-values and Holm corrections are cell-level and are not "
            "cluster-adjusted.",
            "",
            "## Observed query cost",
            "",
            "The following scoped rows are attribution views and must not be summed. The final "
            "physical-union row is the authoritative additive total for the retained queries. "
            "These are historical costs reused from the parent run, not new spend.",
            "",
            "| Condition/arm | Logical calls | Physical calls | Observed tokens | Token saving vs S | Request reduction | Retries |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in micro_costs:
        lines.append(
            f"| {row['source_view']}/k={row['group_size']}/{row['arm']} | "
            f"{int(row['logical_calls']):,} | {int(row['physical_calls']):,} | "
            f"{int(row['observed_total_tokens']):,} | "
            f"{_float(row, 'token_saving_vs_singleton'):.2%} | "
            f"{_float(row, 'request_reduction_vs_singleton'):.2%} | {int(row['retries']):,} |"
        )
    lines.append(
        f"| exact retained physical union | {int(run_cost['logical_calls']):,} | "
        f"{int(run_cost['physical_calls']):,} | {int(run_cost['observed_total_tokens']):,} | "
        f"— | — | {int(run_cost['retries']):,} |"
    )
    lines.extend(
        [
            "",
            "## Scope, definitions, and methods",
            "",
            "- Scope: the registered datasets except `source/movies_1`; both Source and "
            "TableEG suites remain represented. Group views are pattern and semantic, with "
            "k=2,4,8 and the registered g≥3 eligibility rule.",
            "- Correctness: invalid, missing, abstaining, empty, unchanged, and provider-failed "
            "repairs remain wrong in a fixed denominator.",
            "- Complementarity: `salvage` is LLM-correct/Baran-wrong; `overwrite` is "
            "Baran-correct/LLM-wrong. Oracle union is only an offline opportunity upper bound.",
            "- Group effects: rescue is G-correct/S-wrong; interference is S-correct/G-wrong; "
            "G−S equals rescue minus interference. R is matched random batching.",
            "- Inference: 2,000 bootstrap replicates, seed 45, 95% intervals. Complementarity "
            "uses dirty-row clusters. Group intervals use crossed dirty-row and physical-query "
            "multipliers, adding cell ID for across-condition summaries.",
            "- Multiplicity: 8 complementarity dataset tests, 18 primary micro group contrasts, "
            "and 144 secondary dataset group contrasts are separate Holm families.",
            "",
            "## Limitations and interpretation",
            "",
            "This is a post-run exclusion/sensitivity analysis. Removing a large dataset changes "
            "both the cell-weighted population and the equal-dataset macro estimand, so the result "
            "should be reported alongside—not instead of—the frozen nine-dataset analysis. The "
            "same predictions are reused, eliminating model drift and extra cost, but this does "
            "not create a new independent model replication. Exact McNemar p-values ignore "
            "within-query cell dependence; clustered confidence intervals are the preferred "
            "uncertainty summary.",
            "",
            "## Reproducibility and artifacts",
            "",
            "The derived run stores filtered paired ledgers, the retained logical-query mapping, "
            "the exact retained physical cost union, all tables, PDF/SVG, and this report. Raw "
            "responses are not duplicated; their immutable parent artifacts are identified by "
            "hash in `provenance/derived_manifest.json`. No API key was read and no provider, "
            "network, or LLM call was made.",
            "",
            "Questions for paper use: should this exclusion be presented as a sensitivity check "
            "in the appendix, and should the main text emphasize that Movies drives much of the "
            "negative group-vs-singleton macro effect?",
            "",
        ]
    )
    return "\n".join(lines)


def _tree_metadata(parent: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(parent)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(parent.rglob("*"))
        if path.is_file()
    }


def _validate_outputs(stage: Path, filter_summary: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "complementarity_ledger": stage / "records" / "complementarity_cell_outcomes.csv",
        "group_ledger": stage / "records" / "group_cell_outcomes.csv",
        "logical_queries": stage / "evidence" / "logical_queries.jsonl",
        "api_cost_audit": stage / "metrics" / "api_cost_audit.csv",
        "complementarity_metrics": stage / "metrics" / "complementarity_by_dataset.csv",
        "complementarity_summary": stage / "metrics" / "complementarity_summary.json",
        "group_metrics": stage / "metrics" / "group_by_dataset_view_size.csv",
        "group_summary": stage / "metrics" / "group_summary.json",
        "group_transitions": stage / "metrics" / "group_paired_transitions.csv",
        "group_costs": stage / "metrics" / "group_costs.csv",
        "statistical_tests": stage / "metrics" / "statistical_tests.csv",
        "figure_pdf": stage / "figures" / "introduction_motivation.pdf",
        "figure_svg": stage / "figures" / "introduction_motivation.svg",
        "report": stage / "report" / "report.md",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing derived outputs: {missing}")
    expected_rows = {
        "complementarity_metrics": EXPECTED["complementarity_metric_rows"],
        "group_metrics": EXPECTED["group_metric_rows"],
        "group_transitions": EXPECTED["group_transition_rows"],
        "group_costs": EXPECTED["group_cost_rows"],
        "statistical_tests": EXPECTED["statistical_test_rows"],
    }
    parsed: dict[str, list[dict[str, str]]] = {}
    for name, expected in expected_rows.items():
        _, rows = _read_csv(paths[name])
        parsed[name] = rows
        if len(rows) != expected:
            raise RuntimeError(f"{name}: expected {expected} rows, observed {len(rows)}")
    families = Counter(row["test_family"] for row in parsed["statistical_tests"])
    if families != Counter(
        {
            "complementarity_dataset_8": 8,
            "group_primary_18": 18,
            "group_secondary_dataset_144": 144,
            "descriptive": 60,
        }
    ):
        raise RuntimeError(f"unexpected statistical families: {families}")
    primary_ids = {
        (row["source_view"], row["group_size"], row["contrast"])
        for row in parsed["statistical_tests"]
        if row["test_family"] == "group_primary_18"
    }
    if len(primary_ids) != 18:
        raise RuntimeError("primary statistical test IDs are not unique")
    micro = next(
        row for row in parsed["complementarity_metrics"] if row["scope"] == "micro"
    )
    quadrants = {
        "n11": int(micro["n11"]),
        "n10": int(micro["n10"]),
        "n01": int(micro["n01"]),
        "n00": int(micro["n00"]),
    }
    if quadrants != {"n11": 8786, "n10": 3131, "n01": 1272, "n00": 1334}:
        raise RuntimeError(f"unexpected retained complementarity quadrants: {quadrants}")
    expected_micro = {
        "baran_accuracy": 0.820560490256834,
        "llm_accuracy": 0.6925566343042071,
        "oracle_union_upper_bound": 0.9081456999242581,
        "llm_salvage_opportunity": 0.08758520966742409,
        "overwrite_risk": 0.21558906562005095,
    }
    for field, expected in expected_micro.items():
        if not math.isclose(float(micro[field]), expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"unexpected retained micro {field}: {micro[field]}")
    run_union = [
        row for row in parsed["group_costs"] if row["scope"] == "run_physical_union"
    ]
    if len(run_union) != 1:
        raise RuntimeError("derived cost report must contain one physical-union row")
    union = run_union[0]
    for field, expected in (
        ("logical_calls", EXPECTED["logical_queries"]),
        ("physical_calls", EXPECTED["physical_queries"]),
        ("observed_total_tokens", EXPECTED["observed_total_tokens"]),
        ("attempts", EXPECTED["attempts"]),
        ("retries", EXPECTED["retries"]),
        ("unknown_usage_attempts", EXPECTED["unknown_usage_attempts"]),
    ):
        if int(float(union[field])) != expected:
            raise RuntimeError(f"run physical-union {field} mismatch: {union[field]}")
    report = paths["report"].read_text(encoding="utf-8")
    required_phrases = (
        "zero new LLM/API/network calls",
        "eight-dataset macro",
        "144 dataset-level contrasts",
        "sensitivity analysis",
    )
    if any(phrase not in report for phrase in required_phrases):
        raise RuntimeError("derived Markdown is missing required scope language")
    if paths["figure_pdf"].read_bytes()[:5] != b"%PDF-":
        raise RuntimeError("derived PDF is invalid")
    svg_head = paths["figure_svg"].read_text(encoding="utf-8")[:1000]
    if "<svg" not in svg_head:
        raise RuntimeError("derived SVG is invalid")
    if filter_summary["conditions"] != 48:
        raise RuntimeError("derived filter summary condition mismatch")
    return {
        "valid": True,
        "row_counts": expected_rows,
        "holm_family_counts": dict(sorted(families.items())),
        "quadrants": quadrants,
        "primary_test_ids": len(primary_ids),
        "artifact_sha256": {
            name: _sha256(path) for name, path in sorted(paths.items())
        },
    }


def build(parent: Path, output: Path, script_path: Path) -> dict[str, Any]:
    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    script_path = script_path.resolve()
    project_root = script_path.parents[1]
    expected_parent = project_root / "runs" / PARENT_RUN_ID
    if parent != expected_parent.resolve():
        raise ValueError(f"parent must be the frozen formal run: {expected_parent}")
    if output.parent != (project_root / "runs").resolve() or output.name != DERIVED_RUN_ID:
        raise ValueError("derived output path must use the registered derived run ID")
    if output == parent or output.is_relative_to(parent) or parent.is_relative_to(output):
        raise ValueError("source and derived output roots must be disjoint")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing derived run: {output}")

    files, parent_hashes_before, parent_state = _verify_parent(parent)
    parent_metadata_before = _tree_metadata(parent)
    retained = list(parent_state["retained"])
    config = parent_state["config"]
    stage = output.parent / f".{output.name}.tmp.{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=False)
    try:
        derived_config = {
            "schema_version": SCHEMA_VERSION,
            "analysis_type": "post-run dataset-exclusion sensitivity analysis",
            "parent_run_id": PARENT_RUN_ID,
            "derived_run_id": DERIVED_RUN_ID,
            "excluded_datasets": [list(EXCLUDED)],
            "retained_datasets": [list(item) for item in retained],
            "bootstrap_replicates": int(config["bootstrap_replicates"]),
            "bootstrap_seed": int(config["bootstrap_seed"]),
            "confidence_level": float(config["confidence_level"]),
            "group_sizes": list(config["group_sizes"]),
            "primary_views": list(config["primary_views"]),
            "execution_declarations": {
                "network_calls": 0,
                "provider_calls": 0,
                "llm_calls": 0,
                "api_calls": 0,
                "provider_credentials_read": False,
                "llm_responses_regenerated": False,
                "parent_artifacts_read_only": True,
                "raw_responses_copied": False,
            },
        }
        _atomic_json(stage / "configs" / "derived_analysis.json", derived_config)
        filter_summary = _filter_inputs(files, stage, retained)
        statistics_result = _run_statistics(project_root, stage, retained, config)
        saving_range = _write_figure(project_root, stage)
        _atomic_text(
            stage / "report" / "report.md",
            _report_markdown(stage, files, filter_summary, saving_range),
        )
        validation = _validate_outputs(stage, filter_summary)

        finalization_summary = {
            "schema_version": SCHEMA_VERSION,
            "parent_run_id": PARENT_RUN_ID,
            "derived_run_id": DERIVED_RUN_ID,
            "excluded_datasets": [list(EXCLUDED)],
            "retained_dataset_count": 8,
            "complementarity_cells": filter_summary["complementarity_rows"],
            "group_cell_condition_rows": filter_summary["group_rows"],
            "logical_queries": filter_summary["logical_queries"],
            "physical_cost_rows": filter_summary["physical_queries"],
            "quadrants": validation["quadrants"],
            "artifact_sha256": {
                "complementarity": validation["artifact_sha256"]["complementarity_ledger"],
                "group": validation["artifact_sha256"]["group_ledger"],
                "api_cost_audit": validation["artifact_sha256"]["api_cost_audit"],
            },
        }
        _atomic_json(stage / "metrics" / "finalization_summary.json", finalization_summary)

        generated_paths = [
            path
            for path in sorted(stage.rglob("*"))
            if path.is_file() and path.name not in {"run_manifest.json", "derived_manifest.json"}
        ]
        output_hashes = {
            str(path.relative_to(stage)): _sha256(path) for path in generated_paths
        }
        reporting_path = (
            project_root
            / "src"
            / "budgeted_group_repair_no_baran"
            / "motivation_reporting.py"
        )
        visualization_path = project_root / "scripts" / "motivation_visualization_amendment_v1.py"
        generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "analysis_type": "offline dataset-exclusion sensitivity analysis",
            "parent_run": {
                "run_id": PARENT_RUN_ID,
                "path": os.path.relpath(parent, output),
                "required_input_sha256": dict(sorted(parent_hashes_before.items())),
                "checkpoint_reference": {
                    "path": os.path.relpath(parent / "llm" / "query_checkpoint.jsonl", output),
                    "copied": False,
                    "rehash_performed": False,
                    "reason": "finalized paired ledgers and cost audit are the bound analysis inputs",
                },
            },
            "filter": {
                "excluded": [list(EXCLUDED)],
                "retained": [list(item) for item in retained],
                "rule": "retain rows and logical queries whose exact (suite,dataset) is not source/movies_1",
            },
            "execution_declarations": derived_config["execution_declarations"],
            "filter_audit": filter_summary,
            "statistics": {
                "bootstrap_replicates": int(config["bootstrap_replicates"]),
                "bootstrap_seed": int(config["bootstrap_seed"]),
                "confidence_level": float(config["confidence_level"]),
                "holm_families": {
                    "complementarity_dataset_8": 8,
                    "group_primary_18": 18,
                    "group_secondary_dataset_144": 144,
                },
                "mcnemar_note": (
                    "exact cell-level McNemar p-values; clustered multiplier/bootstrap applies to confidence intervals"
                ),
            },
            "implementation_sha256": {
                str(script_path.relative_to(project_root)): _sha256(script_path),
                str(reporting_path.relative_to(project_root)): _sha256(reporting_path),
                str(visualization_path.relative_to(project_root)): _sha256(visualization_path),
            },
            "validation": validation,
            "generated_artifact_sha256": output_hashes,
        }
        _atomic_json(stage / "provenance" / "derived_manifest.json", provenance)
        run_manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": DERIVED_RUN_ID,
            "mode": "offline-derived-sensitivity-analysis",
            "parent_run_id": PARENT_RUN_ID,
            "excluded_datasets": [list(EXCLUDED)],
            "retained_dataset_count": 8,
            "completed": True,
            "generated_at": generated_at,
            "network_calls": 0,
            "provider_calls": 0,
            "llm_calls": 0,
            "api_calls": 0,
            "llm_responses_regenerated": False,
            "counts": {
                **filter_summary,
                "complementarity_metric_rows": EXPECTED["complementarity_metric_rows"],
                "group_metric_rows": EXPECTED["group_metric_rows"],
                "statistical_test_rows": EXPECTED["statistical_test_rows"],
            },
            "reporting": statistics_result,
            "validation": {
                "valid": True,
                "holm_family_counts": validation["holm_family_counts"],
            },
            "provenance": "provenance/derived_manifest.json",
            "report": "report/report.md",
            "figure_pdf": "figures/introduction_motivation.pdf",
            "figure_svg": "figures/introduction_motivation.svg",
        }
        _atomic_json(stage / "run_manifest.json", run_manifest)

        parent_files_after, parent_hashes_after, _ = _verify_parent(parent)
        if set(parent_files_after) != set(files) or parent_hashes_after != parent_hashes_before:
            raise RuntimeError("parent required artifacts changed during derived analysis")
        if _tree_metadata(parent) != parent_metadata_before:
            raise RuntimeError("parent run tree changed during derived analysis")
        os.replace(stage, output)
        return {
            "valid": True,
            "run_id": DERIVED_RUN_ID,
            "run_dir": str(output),
            "report": str(output / "report" / "report.md"),
            "figure_pdf": str(output / "figures" / "introduction_motivation.pdf"),
            "figure_svg": str(output / "figures" / "introduction_motivation.svg"),
            "network_calls": 0,
            "api_calls": 0,
            "llm_calls": 0,
            "counts": filter_summary,
        }
    except BaseException:
        if stage.exists() and stage.parent == output.parent and stage.name.startswith(
            f".{output.name}.tmp."
        ):
            shutil.rmtree(stage)
        raise


def refresh_presentation(parent: Path, output: Path, script_path: Path) -> dict[str, Any]:
    """Safely refresh only the derived Markdown/figure and their provenance."""

    parent = parent.expanduser().resolve()
    output = output.expanduser().resolve()
    script_path = script_path.resolve()
    project_root = script_path.parents[1]
    if parent != (project_root / "runs" / PARENT_RUN_ID).resolve():
        raise ValueError("unexpected parent run")
    if output != (project_root / "runs" / DERIVED_RUN_ID).resolve():
        raise ValueError("unexpected derived run")
    run_manifest_path = output / "run_manifest.json"
    provenance_path = output / "provenance" / "derived_manifest.json"
    run_manifest = _load_json(run_manifest_path)
    provenance = _load_json(provenance_path)
    if run_manifest.get("run_id") != DERIVED_RUN_ID or not run_manifest.get("completed"):
        raise ValueError("derived run is not a completed registered output")
    if provenance.get("parent_run", {}).get("run_id") != PARENT_RUN_ID:
        raise ValueError("derived provenance parent mismatch")

    files, parent_hashes_before, _ = _verify_parent(parent)
    parent_metadata_before = _tree_metadata(parent)
    recorded = dict(provenance.get("generated_artifact_sha256", {}))
    presentation = {
        "figures/introduction_motivation.pdf",
        "figures/introduction_motivation.svg",
        "report/report.md",
    }
    for relative, expected_hash in recorded.items():
        path = output / relative
        if relative not in presentation and _sha256(path) != expected_hash:
            raise RuntimeError(f"non-presentation derived artifact drift: {relative}")

    filter_summary = dict(run_manifest["counts"])
    saving_range = _write_figure(project_root, output)
    _atomic_text(
        output / "report" / "report.md",
        _report_markdown(output, files, filter_summary, saving_range),
    )
    validation = _validate_outputs(output, filter_summary)
    generated_paths = [
        path
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name not in {"run_manifest.json", "derived_manifest.json"}
    ]
    output_hashes = {
        str(path.relative_to(output)): _sha256(path) for path in generated_paths
    }
    reporting_path = (
        project_root / "src" / "budgeted_group_repair_no_baran" / "motivation_reporting.py"
    )
    visualization_path = project_root / "scripts" / "motivation_visualization_amendment_v1.py"
    refreshed_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    provenance["implementation_sha256"] = {
        str(script_path.relative_to(project_root)): _sha256(script_path),
        str(reporting_path.relative_to(project_root)): _sha256(reporting_path),
        str(visualization_path.relative_to(project_root)): _sha256(visualization_path),
    }
    provenance["generated_artifact_sha256"] = output_hashes
    provenance["validation"] = validation
    provenance["presentation_refresh"] = {
        "refreshed_at": refreshed_at,
        "scope": "wording clarification and deterministic label placement only",
        "statistics_recomputed": False,
        "network_calls": 0,
        "provider_calls": 0,
        "llm_calls": 0,
    }
    _atomic_json(provenance_path, provenance)
    run_manifest["presentation_refreshed_at"] = refreshed_at
    run_manifest["presentation_refresh_statistics_recomputed"] = False
    run_manifest["validation"] = {
        "valid": True,
        "holm_family_counts": validation["holm_family_counts"],
    }
    _atomic_json(run_manifest_path, run_manifest)
    _, parent_hashes_after, _ = _verify_parent(parent)
    if parent_hashes_after != parent_hashes_before or _tree_metadata(parent) != parent_metadata_before:
        raise RuntimeError("parent run changed during presentation refresh")
    return {
        "valid": True,
        "run_id": DERIVED_RUN_ID,
        "run_dir": str(output),
        "presentation_refreshed": True,
        "statistics_recomputed": False,
        "network_calls": 0,
        "api_calls": 0,
        "llm_calls": 0,
        "report": str(output / "report" / "report.md"),
        "figure_pdf": str(output / "figures" / "introduction_motivation.pdf"),
        "figure_svg": str(output / "figures" / "introduction_motivation.svg"),
    }


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-run",
        type=Path,
        default=project_root / "runs" / PARENT_RUN_ID,
    )
    parser.add_argument(
        "--output-run",
        type=Path,
        default=project_root / "runs" / DERIVED_RUN_ID,
    )
    parser.add_argument(
        "--refresh-presentation",
        action="store_true",
        help="refresh only Markdown/figure presentation and bound provenance",
    )
    args = parser.parse_args()
    result = (
        refresh_presentation(args.parent_run, args.output_run, script_path)
        if args.refresh_presentation
        else build(args.parent_run, args.output_run, script_path)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
