"""Build the canonical report artifact for a completed No-Baran Router-v2 run.

The module deliberately keeps report authoring separate from HTML rendering.  It
first creates the canonical ``artifact.json`` payload (manifest, bounded
snapshot, and provenance), then optionally invokes the Data Analytics portable
builder.  No clean values or per-cell predictions are copied into the report.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


PRIMARY_BUDGET_SHARE = 0.2
DEFAULT_BUDGET_SHARES = (0.01, 0.05, 0.1, 0.2, 0.5)
DEFAULT_GROUP_SIZES = (1, 4)
EXPECTED_DATASET_COUNT = 9
EXPECTED_ERROR_CELL_COUNT = 22_198


class ReportInputError(ValueError):
    """Raised when formal-run metrics are incomplete or internally inconsistent."""


class PortableReportError(RuntimeError):
    """Raised when the canonical portable artifact builder rejects the report."""


@dataclass(frozen=True)
class ReportInputs:
    """Resolved, explicitly allow-listed inputs used by report generation."""

    run_dir: Path
    method_metrics: Path
    budget_curves: Path | None
    size_ablation: Path | None
    api_cost_audit: Path | None
    record_audit: Path
    split_audit: Path | None
    run_manifest: Path | None


@dataclass(frozen=True)
class ReportBuildResult:
    """Paths and the portable-builder receipt for one report build."""

    artifact_path: Path
    html_path: Path | None
    receipt: Mapping[str, Any] | None


def resolve_report_inputs(run_dir: str | Path) -> ReportInputs:
    """Resolve only known non-secret run artifacts and require core metrics."""

    root = Path(run_dir).expanduser().resolve()
    metrics = root / "metrics"
    method_metrics = metrics / "method_metrics.csv"
    record_audit = metrics / "record_audit.json"
    missing = [path for path in (method_metrics, record_audit) if not path.is_file()]
    if missing:
        names = ", ".join(str(path.relative_to(root)) for path in missing)
        raise ReportInputError(f"completed-run report inputs are missing: {names}")
    return ReportInputs(
        run_dir=root,
        method_metrics=method_metrics,
        budget_curves=_first_file(metrics / "budget_curves.csv"),
        size_ablation=_first_file(metrics / "size_ablation.csv"),
        api_cost_audit=_first_file(metrics / "api_cost_audit.csv"),
        record_audit=record_audit,
        split_audit=_first_file(root / "gates" / "split_audit.csv"),
        run_manifest=_first_file(root / "run_manifest.json"),
    )


def build_report_artifact(
    run_dir: str | Path,
    *,
    primary_budget_share: float = PRIMARY_BUDGET_SHARE,
    budget_shares: Sequence[float] = DEFAULT_BUDGET_SHARES,
    group_sizes: Sequence[int] = DEFAULT_GROUP_SIZES,
    expected_dataset_count: int = EXPECTED_DATASET_COUNT,
    expected_error_cell_count: int = EXPECTED_ERROR_CELL_COUNT,
    generated_at: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Create a canonical ``surface=report`` artifact from formal-run outputs.

    ``strict=True`` is the publication gate: it requires complete per-dataset
    primary results, both gate backends, all requested budget points and group
    sizes, an exact micro cell universe, and a passing record audit.
    """

    inputs = resolve_report_inputs(run_dir)
    method_rows = _read_csv(inputs.method_metrics)
    if not method_rows:
        raise ReportInputError("metrics/method_metrics.csv is empty")

    record_audit = _read_json_object(inputs.record_audit)
    run_manifest = (
        _read_json_object(inputs.run_manifest) if inputs.run_manifest else {}
    )
    derived_recovery = str(run_manifest.get("run_kind", "")) == "derived_recovery"
    timestamp = _generated_at(generated_at, inputs.run_manifest)
    primary = _primary_results(method_rows, primary_budget_share)
    budget_input = _read_csv(inputs.budget_curves) if inputs.budget_curves else method_rows
    budget_rows = _budget_results(budget_input, budget_shares)
    size_input = _read_csv(inputs.size_ablation) if inputs.size_ablation else method_rows
    size_rows = _size_results(size_input, primary_budget_share, group_sizes)
    cost_rows = _read_csv(inputs.api_cost_audit) if inputs.api_cost_audit else []
    cost_summary = _summarize_cost(cost_rows)
    record_rows = _record_audit_rows(record_audit)
    split_rows = _split_audit_rows(_read_csv(inputs.split_audit)) if inputs.split_audit else []

    _validate_report_inputs(
        primary,
        budget_rows,
        size_rows,
        record_audit,
        expected_dataset_count=expected_dataset_count,
        expected_error_cell_count=expected_error_cell_count,
        expected_budgets=budget_shares,
        expected_group_sizes=group_sizes,
        strict=strict,
    )

    primary_table, primary_long, aggregate_rows, headline = _primary_snapshot(primary)
    budget_snapshot = _budget_snapshot(budget_rows, headline["baran_f1"])
    size_snapshot = _size_snapshot(size_rows, headline["baran_f1"])
    run_id = _safe_run_id(inputs.run_dir.name)
    title = f"No-Baran Prompt Budgeted Group Repair 正式实验报告（{run_id}）"

    source_defs = _source_definitions(inputs)
    manifest_sources = [source["manifest"] for source in source_defs]
    canonical_sources = [source["canonical"] for source in source_defs]
    source_ids = {source["manifest"]["id"] for source in source_defs}
    budget_source = "budget_curves" if inputs.budget_curves else "method_metrics"
    size_source = "size_ablation" if inputs.size_ablation else "method_metrics"

    cards = _headline_cards()
    cost_cards = _cost_cards(cost_summary) if inputs.api_cost_audit else []
    cards.extend(cost_cards)

    charts = [
        _primary_chart(),
        _budget_chart(float(headline["baran_f1"]), budget_source),
        _size_chart(float(headline["baran_f1"]), size_source),
    ]
    tables = [
        _primary_table_spec(),
        _aggregate_table_spec(),
        _budget_table_spec(budget_source),
        _size_table_spec(size_source),
        _record_audit_table_spec(),
    ]
    if inputs.api_cost_audit:
        tables.append(_cost_table_spec())
    if inputs.split_audit:
        tables.append(_split_audit_table_spec())

    blocks = _report_blocks(
        title=title,
        headline=headline,
        cost_card_ids=[card["id"] for card in cost_cards],
        include_cost=inputs.api_cost_audit is not None,
        include_split_audit=inputs.split_audit is not None,
        split_rows=split_rows,
        derived_recovery=derived_recovery,
    )
    _assert_references_resolve(blocks, cards, charts, tables, source_ids)

    snapshot_datasets: dict[str, list[dict[str, Any]]] = {
        "headline_metrics": [headline],
        "primary_dataset_long": primary_long,
        "primary_dataset_table": primary_table,
        "aggregate_metrics": aggregate_rows,
        "budget_curve": budget_snapshot,
        "size_ablation": size_snapshot,
        "record_audit": record_rows,
    }
    if inputs.api_cost_audit:
        snapshot_datasets["cost_summary"] = [cost_summary]
    if inputs.split_audit:
        snapshot_datasets["split_audit"] = split_rows

    artifact: dict[str, Any] = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "9 个指定 CARE 数据集上的 Baran 与 No-Baran Prompt Budgeted Group Repair 正式实验、"
                "预算曲线、组大小消融及成本/泄漏审计。"
            ),
            "generatedAt": timestamp,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "ready",
            "datasets": snapshot_datasets,
        },
        "sources": canonical_sources,
        "package_info": {
            "originUrl": f"artifact://budgeted-group-repair/{run_id}",
            "controls": {"edit": False, "refresh": False},
        },
    }
    _assert_artifact_safe(artifact)
    return artifact


def write_report_artifact(
    run_dir: str | Path,
    output_path: str | Path | None = None,
    **build_options: Any,
) -> Path:
    """Write ``artifact.json`` atomically inside the run's report directory."""

    root = Path(run_dir).expanduser().resolve()
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "report" / "artifact.json"
    )
    artifact = build_report_artifact(root, **build_options)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def discover_data_analytics_plugin_root(explicit: str | Path | None = None) -> Path:
    """Locate an installed Data Analytics plugin without touching report data."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    configured = os.environ.get("DATA_ANALYTICS_PLUGIN_ROOT")
    if configured:
        candidates.append(Path(configured).expanduser())
    codex_home = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    cache = codex_home / "plugins" / "cache"
    for channel in ("openai-curated-remote", "openai-curated", "openai-bundled"):
        candidates.extend(sorted((cache / channel / "data-analytics").glob("*"), reverse=True))
    for root in candidates:
        builder = root / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs"
        if builder.is_file():
            return root.resolve()
    raise PortableReportError(
        "Data Analytics portable builder was not found; pass --plugin-root or set "
        "DATA_ANALYTICS_PLUGIN_ROOT"
    )


def discover_node_executable(explicit: str | Path | None = None) -> Path:
    """Find Node.js, preferring an explicit path and then Codex's local runtime."""

    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser())
    on_path = shutil.which("node")
    if on_path:
        candidates.append(Path(on_path))
    runtime_root = Path.home() / ".cache" / "codex-runtimes"
    candidates.append(runtime_root / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node")
    candidates.extend(
        sorted(runtime_root.glob("*/dependencies/node/bin/node"), reverse=True)
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise PortableReportError("Node.js was not found; pass --node with an executable path")


def deliver_portable_report(
    artifact_path: str | Path,
    output_path: str | Path,
    *,
    plugin_root: str | Path | None = None,
    node_executable: str | Path | None = None,
    timeout_seconds: float = 120.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Invoke the plugin's canonical one-pass package-and-verify command."""

    artifact = Path(artifact_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not artifact.is_file():
        raise PortableReportError(f"artifact input does not exist: {artifact}")
    root = discover_data_analytics_plugin_root(plugin_root)
    node = discover_node_executable(node_executable)
    builder = root / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(node), str(builder), "--input", str(artifact), "--output", str(output)]
    completed = runner(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    receipt = _parse_builder_receipt(completed.stdout)
    if completed.returncode != 0 or not receipt.get("ok", False):
        detail = receipt.get("error") or completed.stderr.strip() or completed.stdout.strip()
        raise PortableReportError(f"portable report delivery failed: {detail}")
    if not output.is_file():
        raise PortableReportError("portable builder reported success but produced no HTML file")
    return receipt


def generate_report(
    run_dir: str | Path,
    *,
    artifact_path: str | Path | None = None,
    html_path: str | Path | None = None,
    deliver: bool = True,
    plugin_root: str | Path | None = None,
    node_executable: str | Path | None = None,
    **build_options: Any,
) -> ReportBuildResult:
    """Generate the canonical artifact and, by default, its portable HTML."""

    root = Path(run_dir).expanduser().resolve()
    artifact = write_report_artifact(root, artifact_path, **build_options)
    if not deliver:
        return ReportBuildResult(artifact, None, None)
    html = (
        Path(html_path).expanduser().resolve()
        if html_path is not None
        else root / "report" / "report.html"
    )
    receipt = deliver_portable_report(
        artifact,
        html,
        plugin_root=plugin_root,
        node_executable=node_executable,
    )
    return ReportBuildResult(artifact, html, receipt)


def build_report(
    run_dir: str | Path,
    project_root: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """CLI-compatible one-call report build.

    ``output_path`` is the optional HTML destination; the canonical source
    artifact is always written to ``RUN_DIR/report/artifact.json`` so the final
    HTML remains reproducible and independently inspectable.
    """

    root = Path(run_dir).expanduser().resolve()
    project = Path(project_root).expanduser().resolve()
    if not project.is_dir():
        raise ReportInputError(f"project root does not exist: {project}")
    try:
        from .router_v2 import validate_run

        validation = validate_run(root, require_complete=True)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise ReportInputError(
            f"formal-run validation failed before report generation: {error}"
        ) from error
    result = generate_report(
        root,
        html_path=output_path,
        deliver=True,
    )
    receipt = dict(result.receipt or {})
    return {
        "ok": bool(receipt.get("ok", False)),
        "run_dir": root,
        "artifact": result.artifact_path,
        "html": result.html_path,
        "run_validation": validation,
        "verification": receipt.get("stages", {}).get("verification"),
        "delivery": receipt,
    }


def build_router_report(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """No-Baran CLI wrapper with the package root supplied automatically."""

    root = Path(run_dir).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json_object(manifest_path)
        experiment = manifest.get("experiment_config", {})
        if (
            isinstance(experiment, Mapping)
            and str(experiment.get("router_revision", ""))
            in {
                "router_v3_exact_size_conditioned",
                "router_v3_budget_sweep_exact_size_conditioned",
                "router_v3_catboost_exact_size_conditioned",
            }
        ):
            from .router_reporting_v3 import build_router_v3_report

            return build_router_v3_report(root, output_path=output_path)
    return build_report(
        root,
        project_root=Path(__file__).resolve().parents[2],
        output_path=output_path,
    )


def _first_file(path: Path) -> Path | None:
    return path if path.is_file() else None


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ReportInputError(f"{path.name} must contain a JSON object")
    return value


def _generated_at(explicit: str | None, run_manifest: Path | None) -> str:
    candidate = explicit
    if candidate is None and run_manifest:
        manifest = _read_json_object(run_manifest)
        for key in ("completed_at", "generated_at", "updated_at", "created_at"):
            if isinstance(manifest.get(key), str):
                candidate = str(manifest[key])
                break
    if candidate is None:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    text = candidate.strip()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReportInputError("generated_at must be an ISO-8601 timestamp") from error
    return text


def _primary_results(rows: Sequence[Mapping[str, Any]], budget: float) -> dict[str, Any]:
    dataset_rows = [row for row in rows if _scope(row) == "dataset"]
    micro_rows = [row for row in rows if _scope(row) == "micro"]
    macro_rows = [row for row in rows if _scope(row) == "macro"]
    baselines = _deduplicate_baseline(dataset_rows)
    micro_baseline = _choose_baseline(micro_rows, "MICRO")
    macro_baseline = _choose_baseline(macro_rows, "MACRO")
    methods: dict[str, dict[str, Mapping[str, Any]]] = {"lightgbm": {}, "xgboost": {}}
    micro_methods: dict[str, Mapping[str, Any]] = {}
    macro_methods: dict[str, Mapping[str, Any]] = {}
    for backend in methods:
        for row in dataset_rows:
            if (
                _backend(row) == backend
                and _is_primary_scenario(row)
                and _variant_is_all(row)
                and _same_number(_number(row, "budget_share", "budget", "beta"), budget)
            ):
                methods[backend][_dataset(row)] = row
        candidates = [
            row
            for row in micro_rows
            if _backend(row) == backend
            and _is_primary_scenario(row)
            and _variant_is_all(row)
            and _same_number(_number(row, "budget_share", "budget", "beta"), budget)
        ]
        if candidates:
            micro_methods[backend] = _choose_unique(candidates, f"{backend} primary micro")
        macro_candidates = [
            row
            for row in macro_rows
            if _backend(row) == backend
            and _is_primary_scenario(row)
            and _variant_is_all(row)
            and _same_number(_number(row, "budget_share", "budget", "beta"), budget)
        ]
        if macro_candidates:
            macro_methods[backend] = _choose_unique(
                macro_candidates, f"{backend} primary macro"
            )
    return {
        "baselines": baselines,
        "micro_baseline": micro_baseline,
        "macro_baseline": macro_baseline,
        "methods": methods,
        "micro_methods": micro_methods,
        "macro_methods": macro_methods,
    }


def _budget_results(
    rows: Sequence[Mapping[str, Any]], budgets: Sequence[float]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backend in ("lightgbm", "xgboost"):
        for budget in budgets:
            candidates = [
                row
                for row in rows
                if _backend(row) == backend
                and _scope(row) in {"micro", "unspecified"}
                and _is_primary_scenario(row)
                and _variant_is_all(row)
                and _same_number(_number(row, "budget_share", "budget", "beta"), budget)
            ]
            if candidates:
                row = _choose_unique(candidates, f"{backend} budget={budget}")
                normalized = _metric_row(row)
                normalized["backend"] = backend
                normalized["budget_share"] = float(budget)
                results.append(normalized)
    return results


def _size_results(
    rows: Sequence[Mapping[str, Any]], budget: float, group_sizes: Sequence[int]
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for backend in ("lightgbm", "xgboost"):
        for size in group_sizes:
            candidates = []
            for row in rows:
                row_size = _variant_size(row)
                row_budget = _number(row, "budget_share", "budget", "beta")
                if (
                    _backend(row) == backend
                    and _scope(row) in {"micro", "unspecified"}
                    and row_size == int(size)
                    and (row_budget is None or _same_number(row_budget, budget))
                ):
                    candidates.append(row)
            if candidates:
                row = _choose_ablation(candidates, backend, size)
                normalized = _metric_row(row)
                normalized["backend"] = backend
                normalized["group_size"] = int(size)
                normalized["budget_share"] = float(budget)
                results.append(normalized)
    return results


def _validate_report_inputs(
    primary: Mapping[str, Any],
    budget_rows: Sequence[Mapping[str, Any]],
    size_rows: Sequence[Mapping[str, Any]],
    record_audit: Mapping[str, Any],
    *,
    expected_dataset_count: int,
    expected_error_cell_count: int,
    expected_budgets: Sequence[float],
    expected_group_sizes: Sequence[int],
    strict: bool,
) -> None:
    baselines = primary["baselines"]
    if not baselines:
        raise ReportInputError("no per-dataset Baran rows were found")
    if primary["micro_baseline"] is None:
        raise ReportInputError("no Baran micro row was found")
    if primary["macro_baseline"] is None:
        raise ReportInputError("no Baran macro row was found")
    structural_problems: list[str] = []
    datasets = set(baselines)
    if len(datasets) != expected_dataset_count:
        structural_problems.append(f"datasets={len(datasets)} (expected {expected_dataset_count})")
    for backend in ("lightgbm", "xgboost"):
        observed = set(primary["methods"][backend])
        if observed != datasets:
            structural_problems.append(
                f"primary {backend} dataset coverage missing={len(datasets - observed)} "
                f"extra={len(observed - datasets)}"
            )
        if backend not in primary["micro_methods"]:
            structural_problems.append(f"primary {backend} micro row missing")
        if backend not in primary["macro_methods"]:
            structural_problems.append(f"primary {backend} macro row missing")
    micro_cells = _integer(primary["micro_baseline"], "true_error_cells", "error_cells", "cells")
    if micro_cells != expected_error_cell_count:
        structural_problems.append(f"micro error cells={micro_cells} (expected {expected_error_cell_count})")
    for backend in ("lightgbm", "xgboost"):
        observed = {
            round(float(row["budget_share"]), 12)
            for row in budget_rows
            if row["backend"] == backend
        }
        expected = {round(float(value), 12) for value in expected_budgets}
        if observed != expected:
            structural_problems.append(f"{backend} budget points={sorted(observed)} (expected {sorted(expected)})")
        observed_sizes = {
            int(row["group_size"])
            for row in size_rows
            if row["backend"] == backend
        }
        if observed_sizes != {int(value) for value in expected_group_sizes}:
            structural_problems.append(
                f"{backend} group sizes={sorted(observed_sizes)} "
                f"(expected {sorted(int(value) for value in expected_group_sizes)})"
            )
    if structural_problems:
        raise ReportInputError(
            "formal-run report structure is incomplete: " + "; ".join(structural_problems)
        )
    if strict and record_audit.get("ok") is not True:
        raise ReportInputError("formal-run report gate failed: record_audit.ok is not true")


def _primary_snapshot(
    primary: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    baseline_micro = _metric_row(primary["micro_baseline"])
    baseline_macro = _metric_row(primary["macro_baseline"])
    micro = {
        backend: _metric_row(primary["micro_methods"][backend])
        for backend in ("lightgbm", "xgboost")
    }
    macro = {
        backend: _metric_row(primary["macro_methods"][backend])
        for backend in ("lightgbm", "xgboost")
    }
    best_backend = max(micro, key=lambda key: (micro[key]["f1"], key))
    table_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    wins = {"lightgbm": 0, "xgboost": 0}
    for suite_dataset, baseline_source in sorted(primary["baselines"].items()):
        suite, dataset = suite_dataset.split("/", 1)
        baseline = _metric_row(baseline_source)
        method_metrics = {
            backend: _metric_row(primary["methods"][backend][suite_dataset])
            for backend in ("lightgbm", "xgboost")
        }
        for backend, values in method_metrics.items():
            if values["f1"] > baseline["f1"]:
                wins[backend] += 1
        table = {
            "suite": suite,
            "dataset": dataset,
            "dataset_label": f"{suite}/{dataset}",
            "true_error_cells": baseline["true_error_cells"],
            "baran_precision": baseline["precision"],
            "baran_recall": baseline["recall"],
            "baran_f1": baseline["f1"],
            "baran_correct_repairs": baseline["correct_repairs"],
        }
        for backend, values in method_metrics.items():
            prefix = "bgr_lightgbm" if backend == "lightgbm" else "bgr_xgboost"
            table.update(
                {
                    f"{prefix}_precision": values["precision"],
                    f"{prefix}_recall": values["recall"],
                    f"{prefix}_f1": values["f1"],
                    f"{prefix}_correct_repairs": values["correct_repairs"],
                    f"{prefix}_f1_delta": values["f1"] - baseline["f1"],
                }
            )
        table["best_f1_delta"] = max(
            table["bgr_lightgbm_f1_delta"], table["bgr_xgboost_f1_delta"]
        )
        table_rows.append(table)
        series = [
            ("Baran", "baran", baseline),
            ("BGR-LightGBM", "lightgbm", method_metrics["lightgbm"]),
            ("BGR-XGBoost", "xgboost", method_metrics["xgboost"]),
        ]
        for label, backend, values in series:
            long_rows.append(
                {
                    "suite": suite,
                    "dataset": dataset,
                    "dataset_label": table["dataset_label"],
                    "method": label,
                    "backend": backend,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "correct_repairs": values["correct_repairs"],
                    "true_error_cells": values["true_error_cells"],
                    "f1_delta_vs_baran": values["f1"] - baseline["f1"],
                    "budget_share": None if backend == "baran" else PRIMARY_BUDGET_SHARE,
                }
            )
    table_rows.sort(key=lambda row: (-float(row["best_f1_delta"]), str(row["dataset_label"])))
    rank = {row["dataset_label"]: index + 1 for index, row in enumerate(table_rows)}
    for row in long_rows:
        row["rank_by_best_delta"] = rank[row["dataset_label"]]
    long_rows.sort(key=lambda row: (row["rank_by_best_delta"], ("Baran", "BGR-LightGBM", "BGR-XGBoost").index(row["method"])))
    headline = {
        "row": "micro_20_percent",
        "true_error_cells": baseline_micro["true_error_cells"],
        "baran_precision": baseline_micro["precision"],
        "baran_recall": baseline_micro["recall"],
        "baran_f1": baseline_micro["f1"],
        "baran_correct_repairs": baseline_micro["correct_repairs"],
        "lightgbm_precision": micro["lightgbm"]["precision"],
        "lightgbm_recall": micro["lightgbm"]["recall"],
        "lightgbm_f1": micro["lightgbm"]["f1"],
        "lightgbm_f1_delta": micro["lightgbm"]["f1"] - baseline_micro["f1"],
        "lightgbm_correct_repairs": micro["lightgbm"]["correct_repairs"],
        "lightgbm_dataset_wins": wins["lightgbm"],
        "xgboost_precision": micro["xgboost"]["precision"],
        "xgboost_recall": micro["xgboost"]["recall"],
        "xgboost_f1": micro["xgboost"]["f1"],
        "xgboost_f1_delta": micro["xgboost"]["f1"] - baseline_micro["f1"],
        "xgboost_correct_repairs": micro["xgboost"]["correct_repairs"],
        "xgboost_dataset_wins": wins["xgboost"],
        "best_backend": best_backend,
        "best_bgr_f1": micro[best_backend]["f1"],
        "best_bgr_f1_delta": micro[best_backend]["f1"] - baseline_micro["f1"],
        "best_bgr_correct_repairs": micro[best_backend]["correct_repairs"],
        "dataset_count": len(table_rows),
        "primary_budget_share": PRIMARY_BUDGET_SHARE,
        "baran_macro_precision": baseline_macro["precision"],
        "baran_macro_recall": baseline_macro["recall"],
        "baran_macro_f1": baseline_macro["f1"],
        "lightgbm_macro_precision": macro["lightgbm"]["precision"],
        "lightgbm_macro_recall": macro["lightgbm"]["recall"],
        "lightgbm_macro_f1": macro["lightgbm"]["f1"],
        "xgboost_macro_precision": macro["xgboost"]["precision"],
        "xgboost_macro_recall": macro["xgboost"]["recall"],
        "xgboost_macro_f1": macro["xgboost"]["f1"],
    }
    aggregates: list[dict[str, Any]] = []
    for scope, values_by_method in (
        ("micro", {"Baran": baseline_micro, "BGR-LightGBM": micro["lightgbm"], "BGR-XGBoost": micro["xgboost"]}),
        ("macro", {"Baran": baseline_macro, "BGR-LightGBM": macro["lightgbm"], "BGR-XGBoost": macro["xgboost"]}),
    ):
        for method_label, values in values_by_method.items():
            aggregates.append(
                {
                    "scope": scope,
                    "method": method_label,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1"],
                    "correct_repairs": values["correct_repairs"],
                    "true_error_cells": values["true_error_cells"],
                }
            )
    return table_rows, long_rows, aggregates, headline


def _budget_snapshot(rows: Sequence[Mapping[str, Any]], baseline_f1: float) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda value: (value["backend"], value["budget_share"])):
        value = dict(row)
        value["backend_label"] = _backend_label(str(value["backend"]))
        value["budget_percent"] = float(value["budget_share"])
        value["budget_label"] = f"{float(value['budget_share']):.0%}"
        value["f1_delta_vs_baran"] = float(value["f1"]) - baseline_f1
        output.append(value)
    return output


def _size_snapshot(rows: Sequence[Mapping[str, Any]], baseline_f1: float) -> list[dict[str, Any]]:
    output = []
    for row in sorted(rows, key=lambda value: (value["backend"], value["group_size"])):
        value = dict(row)
        value["backend_label"] = _backend_label(str(value["backend"]))
        value["group_size_label"] = f"k={int(value['group_size'])}"
        value["f1_delta_vs_baran"] = float(value["f1"]) - baseline_f1
        output.append(value)
    return output


def _metric_row(row: Mapping[str, Any]) -> dict[str, Any]:
    precision = _required_number(row, "precision")
    recall = _required_number(row, "recall", "correction_accuracy")
    f1 = _required_number(row, "f1", "F1")
    true_cells = _required_integer(row, "true_error_cells", "error_cells", "cells")
    correct = _required_integer(row, "correct_repairs", "correct", "correct_cells")
    result: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_error_cells": true_cells,
        "correct_repairs": correct,
    }
    for canonical, aliases in {
        "estimated_tokens": ("estimated_tokens", "logical_estimated_tokens"),
        "actual_tokens": ("actual_tokens", "provider_total_tokens", "actual_total_tokens"),
        "logical_api_calls": ("logical_api_calls",),
        "physical_api_calls": ("physical_api_calls",),
    }.items():
        value = _number(row, *aliases)
        if value is not None:
            result[canonical] = value
    return result


def _summarize_cost(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = {
        "calibration_estimated_tokens": (
            "calibration_estimated_tokens",
            "offline_calibration_estimated_tokens",
        ),
        "calibration_provider_tokens": (
            "calibration_provider_tokens",
            "offline_calibration_tokens",
        ),
        "online_logical_estimated_tokens": (
            "online_logical_estimated_tokens",
            "logical_estimated_tokens",
        ),
        "provider_input_tokens": ("provider_input_tokens", "actual_input_tokens", "input_tokens"),
        "provider_output_tokens": ("provider_output_tokens", "actual_output_tokens", "output_tokens"),
        "provider_total_tokens": (
            "provider_total_tokens",
            "actual_tokens",
            "actual_total_tokens",
            "actual_tokens_if_available",
        ),
        "logical_api_calls": ("logical_api_calls",),
        "physical_api_calls": ("physical_api_calls",),
        "unknown_token_attempts": ("unknown_token_attempts", "unknown_usage_attempts"),
        "provider_failed_records": ("provider_failed_records", "failed_records"),
        "historical_failed_records": ("historical_failed_records",),
        "operational_fallback_records": ("operational_fallback_records",),
        "operational_fallback_cells": ("operational_fallback_cells",),
        "unresolved_operational_failures": ("unresolved_operational_failures",),
    }
    totals = [row for row in rows if _is_total_row(row)]
    selected = totals[:1] if totals else list(rows)
    summary: dict[str, Any] = {"scope": "formal_run"}
    for canonical, aliases in fields.items():
        values = [_number(row, *aliases) for row in selected]
        present = [value for value in values if value is not None]
        if present:
            summary[canonical] = sum(present) if not totals else present[0]
    if not totals:
        calibration_rows = [row for row in rows if _is_calibration_row(row)]
        online_rows = [row for row in rows if not _is_calibration_row(row)]
        if "calibration_estimated_tokens" not in summary:
            values = [_number(row, "estimated_tokens") for row in calibration_rows]
            present = [value for value in values if value is not None]
            if present:
                summary["calibration_estimated_tokens"] = sum(present)
        if "online_logical_estimated_tokens" not in summary:
            values = [_number(row, "estimated_tokens") for row in online_rows]
            present = [value for value in values if value is not None]
            if present:
                summary["online_logical_estimated_tokens"] = sum(present)
    if "provider_total_tokens" not in summary:
        input_tokens = summary.get("provider_input_tokens")
        output_tokens = summary.get("provider_output_tokens")
        if input_tokens is not None and output_tokens is not None:
            summary["provider_total_tokens"] = float(input_tokens) + float(output_tokens)
    return summary


def _record_audit_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = [
        ("记录审计总状态", audit.get("ok") is True, "ok", audit.get("ok")),
        ("重复记录", _count(audit.get("duplicate_records")) == 0, "duplicate_records", _count(audit.get("duplicate_records"))),
        ("覆盖错误", _count(audit.get("coverage_errors")) == 0, "coverage_errors", _count(audit.get("coverage_errors"))),
        ("缓存标签不一致", _count(audit.get("annotation_mismatches")) == 0, "annotation_mismatches", _count(audit.get("annotation_mismatches"))),
        ("必要字段缺失", _count(audit.get("missing_fields")) == 0, "missing_fields", _count(audit.get("missing_fields"))),
    ]
    rows = []
    for label, passed, field, value in checks:
        detail = str(value).lower() if isinstance(value, bool) else str(value)
        rows.append({"check": label, "status": "通过" if passed else "失败", "evidence_field": field, "value": detail})
    for field in ("records", "unique_records", "slices", "valid_repairs", "correct_repairs"):
        if field in audit:
            rows.append({"check": field, "status": "信息", "evidence_field": field, "value": str(audit[field])})
    return rows


def _split_audit_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    specifications = [
        ("Base-family 训练/测试重叠", ("train_test_base_family_overlap", "base_family_overlap"), "zero"),
        ("Cell 训练/测试重叠", ("train_test_cell_overlap", "cell_overlap"), "zero"),
        ("Row 训练/测试重叠", ("train_test_row_overlap", "row_overlap"), "zero"),
        ("Query 训练/测试重叠", ("train_test_query_overlap", "query_overlap"), "zero"),
        ("Target family 标签进入训练", ("target_group_label_used", "target_label_used"), "false"),
        ("选择前读取 target response", ("target_response_visible_before_selection", "target_response_used"), "false"),
    ]
    output = []
    for label, aliases, expectation in specifications:
        values = [_raw(row, *aliases) for row in rows]
        values = [value for value in values if value not in {None, ""}]
        if not values:
            output.append({"check": label, "status": "未报告", "targets": len(rows), "worst_value": "—"})
            continue
        if expectation == "zero":
            numeric = [_coerce_number(value) for value in values]
            passed = all(value == 0 for value in numeric if value is not None) and all(value is not None for value in numeric)
            worst = max(numeric) if numeric and all(value is not None for value in numeric) else "invalid"
        else:
            parsed = [_coerce_bool(value) for value in values]
            passed = all(value is False for value in parsed)
            worst = any(value is True for value in parsed)
        output.append({"check": label, "status": "通过" if passed else "失败", "targets": len(values), "worst_value": str(worst).lower()})
    return output


def _source_definitions(inputs: ReportInputs) -> list[dict[str, dict[str, Any]]]:
    entries: list[tuple[str, str, str, str, list[str]]] = [
        (
            "method_metrics",
            "正式实验方法指标",
            "metrics/method_metrics.csv",
            "由完整 per-cell 最终记录独立重算的逐数据集、micro 与 macro correction-only 指标。",
            [
                "Precision = correct repairs / parse-valid predicted repairs.",
                "Recall = correct repairs / oracle error cells.",
                "F1 = harmonic mean of Precision and Recall.",
                "Primary BGR rows use logical estimated-token budget share 0.20.",
            ],
        ),
        (
            "record_audit",
            "最终记录覆盖审计",
            "metrics/record_audit.json",
            "检查最终记录的身份唯一性、字段完整性、覆盖以及缓存 correctness 注释一致性。",
            [],
        ),
        (
            "method_spec",
            "BGR 方法与实验规范",
            "docs/03_实现与实验说明.md",
            "实验矩阵、指标定义、成本口径、泄漏边界与完成标准。",
            [],
        ),
    ]
    if inputs.run_manifest:
        entries.append(
            (
                "run_manifest",
                "运行绑定与派生来源",
                "run_manifest.json",
                "记录代码、配置、数据绑定，以及派生恢复运行的父运行与回退策略。",
                [],
            )
        )
    if inputs.budget_curves:
        entries.append(("budget_curves", "预算曲线指标", "metrics/budget_curves.csv", "BGR 两个 gate backend 的 micro 指标与逻辑预算点。", ["Budget share is relative to the full singleton estimated-token cost."]))
    if inputs.size_ablation:
        entries.append(("size_ablation", "组大小消融指标", "metrics/size_ablation.csv", "20% 预算下 singleton-only 与 singleton+exact-4 的 micro 指标。", ["k=1 is singleton-only; k=4 includes singletons plus exact-size-4 groups."]))
    if inputs.api_cost_audit:
        entries.append(("api_cost_audit", "API 与 token 成本审计", "metrics/api_cost_audit.csv", "区分离线校准、逻辑在线预算与物理 provider 调用的正式账本汇总。", []))
    if inputs.split_audit:
        entries.append(("split_audit", "跨 family 切分与泄漏审计", "gates/split_audit.csv", "逐 target family 检查训练/测试重叠、target 标签和选择前 response 可见性。", []))
    output = []
    for source_id, label, path, description, metric_definitions in entries:
        query: dict[str, Any] = {"description": description}
        if path.endswith(".csv"):
            query.update(
                {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": f"SELECT * FROM read_csv_auto('{path}', header = true);",
                    "tables_used": [path],
                }
            )
        elif path.endswith(".json"):
            query.update(
                {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": f"SELECT * FROM read_json_auto('{path}');",
                    "tables_used": [path],
                }
            )
        if metric_definitions:
            query["metric_definitions"] = metric_definitions
        output.append(
            {
                "manifest": {"id": source_id, "label": label, "path": path},
                "canonical": {"id": source_id, "label": label, "path": path, "query": query},
            }
        )
    return output


def _headline_cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "baran_micro_f1",
            "description": "Fresh Baran 在全部 oracle error cells 上的 correction-only micro F1。",
            "dataset": "headline_metrics",
            "sourceId": "method_metrics",
            "filter": {"row": "micro_20_percent"},
            "metrics": [{"label": "Baran micro F1", "field": "baran_f1", "format": "percent"}],
        },
        {
            "id": "lightgbm_micro_f1",
            "description": "20% 逻辑 token 预算下 BGR-LightGBM 的 micro F1。",
            "dataset": "headline_metrics",
            "sourceId": "method_metrics",
            "filter": {"row": "micro_20_percent"},
            "metrics": [
                {"label": "BGR-LightGBM F1", "field": "lightgbm_f1", "format": "percent"},
                {"label": "vs Baran", "field": "lightgbm_f1_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "xgboost_micro_f1",
            "description": "20% 逻辑 token 预算下 BGR-XGBoost 的 micro F1。",
            "dataset": "headline_metrics",
            "sourceId": "method_metrics",
            "filter": {"row": "micro_20_percent"},
            "metrics": [
                {"label": "BGR-XGBoost F1", "field": "xgboost_f1", "format": "percent"},
                {"label": "vs Baran", "field": "xgboost_f1_delta", "format": "percent", "signed": True},
            ],
        },
        {
            "id": "best_correct_repairs",
            "description": "20% 预算下 micro F1 更高的 BGR backend 所产生的正确修复数。",
            "dataset": "headline_metrics",
            "sourceId": "method_metrics",
            "filter": {"row": "micro_20_percent"},
            "metrics": [{"label": "Best BGR 正确修复", "field": "best_bgr_correct_repairs", "format": "number"}],
        },
    ]


def _cost_cards(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    specifications = [
        ("calibration_provider_tokens", "离线校准 provider tokens", "校准阶段实际 provider token；不计入任一在线预算。"),
        ("online_logical_estimated_tokens", "在线逻辑 estimated tokens", "选择场景按完整查询成本计的逻辑 token，不因缓存命中减免。"),
        ("provider_total_tokens", "物理 provider tokens", "去重后的 run-local 物理 API 调用实际 token。"),
        ("physical_api_calls", "物理 API 调用", "run-local 唯一 query/prompt 请求数。"),
    ]
    cards = []
    for field, label, description in specifications:
        if field not in summary:
            continue
        cards.append(
            {
                "id": f"cost_{field}",
                "description": description,
                "dataset": "cost_summary",
                "sourceId": "api_cost_audit",
                "filter": {"scope": "formal_run"},
                "metrics": [{"label": label, "field": field, "format": "number"}],
            }
        )
    return cards


def _primary_chart() -> dict[str, Any]:
    return {
        "id": "primary_f1_by_dataset",
        "title": "逐数据集 correction-only F1：Baran 与 20% 预算 BGR",
        "subtitle": "按每个数据集上最佳 BGR 相对 Baran 的 F1 差值排序；横轴越高越好。",
        "type": "horizontalBar",
        "intent": "comparison",
        "question": "20% 预算下，两个 BGR gate backend 在各数据集是否优于 Baran？",
        "rationale": "横向分组条形图便于同时读取 9 个数据集名称并比较三个方法。",
        "dataset": "primary_dataset_long",
        "sourceId": "method_metrics",
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "dataset_label", "type": "nominal", "label": "数据集"},
            "y": {"field": "f1", "type": "quantitative", "label": "F1", "format": "percent"},
            "color": {"field": "method", "type": "nominal", "label": "方法"},
            "tooltip": [
                {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                {"field": "correct_repairs", "type": "quantitative", "label": "正确修复", "format": "number"},
                {"field": "true_error_cells", "type": "quantitative", "label": "错误单元格", "format": "number"},
                {"field": "f1_delta_vs_baran", "type": "quantitative", "label": "F1 Δ vs Baran", "format": "percent"},
            ],
        },
        "legend": {"position": "bottom", "sort": "spec", "title": "方法"},
        "layout": "full",
    }


def _budget_chart(baseline_f1: float, source_id: str) -> dict[str, Any]:
    return {
        "id": "micro_f1_budget_curve",
        "title": "Micro F1 随逻辑 token 预算变化",
        "subtitle": "预算按全量 singleton estimated-token 成本的比例定义；虚线基准为 Baran。",
        "type": "line",
        "intent": "trend",
        "question": "BGR 的 micro F1 是否随可用逻辑 token 预算稳定改善？",
        "rationale": "预算份额具有有序数值含义，折线用于呈现两个 gate backend 的响应轨迹。",
        "dataset": "budget_curve",
        "sourceId": source_id,
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "budget_percent", "type": "quantitative", "label": "逻辑预算份额", "format": "percent"},
            "y": {"field": "f1", "type": "quantitative", "label": "Micro F1", "format": "percent"},
            "color": {"field": "backend_label", "type": "nominal", "label": "Gate backend"},
            "tooltip": [
                {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                {"field": "correct_repairs", "type": "quantitative", "label": "正确修复", "format": "number"},
                {"field": "f1_delta_vs_baran", "type": "quantitative", "label": "F1 Δ vs Baran", "format": "percent"},
            ],
        },
        "legend": {"position": "bottom", "sort": "spec", "title": "Gate backend"},
        "referenceLines": [{"axis": "y", "value": baseline_f1, "label": "Baran micro F1", "lineStyle": "dashed", "color": "neutral"}],
        "layout": "full",
    }


def _size_chart(baseline_f1: float, source_id: str) -> dict[str, Any]:
    return {
        "id": "micro_f1_size_ablation",
        "title": "20% 预算下组大小 1/2/4/8 消融",
        "subtitle": "k=1 为 singleton-only；k=2/4/8 为 singleton 加 exact-size-k 组。",
        "type": "bar",
        "intent": "comparison",
        "question": "允许哪一种精确非 singleton 组大小时，BGR 的 micro F1 最高？",
        "rationale": "四个离散候选池设置适合用分组条形图直接比较。",
        "dataset": "size_ablation",
        "sourceId": source_id,
        "valueFormat": "percent",
        "encodings": {
            "x": {"field": "group_size_label", "type": "nominal", "label": "组大小设置"},
            "y": {"field": "f1", "type": "quantitative", "label": "Micro F1", "format": "percent"},
            "color": {"field": "backend_label", "type": "nominal", "label": "Gate backend"},
            "tooltip": [
                {"field": "precision", "type": "quantitative", "label": "Precision", "format": "percent"},
                {"field": "recall", "type": "quantitative", "label": "Recall", "format": "percent"},
                {"field": "correct_repairs", "type": "quantitative", "label": "正确修复", "format": "number"},
                {"field": "f1_delta_vs_baran", "type": "quantitative", "label": "F1 Δ vs Baran", "format": "percent"},
            ],
        },
        "legend": {"position": "bottom", "sort": "spec", "title": "Gate backend"},
        "referenceLines": [{"axis": "y", "value": baseline_f1, "label": "Baran micro F1", "lineStyle": "dashed", "color": "neutral"}],
        "layout": "full",
    }


def _primary_table_spec() -> dict[str, Any]:
    columns = [
        {"field": "dataset_label", "label": "数据集", "type": "text"},
        {"field": "true_error_cells", "label": "错误 cells", "format": "number"},
    ]
    for prefix, label in (("baran", "Baran"), ("bgr_lightgbm", "BGR-LGBM"), ("bgr_xgboost", "BGR-XGB")):
        columns.extend(
            [
                {"field": f"{prefix}_precision", "label": f"{label} P", "format": "percent"},
                {"field": f"{prefix}_recall", "label": f"{label} R", "format": "percent"},
                {"field": f"{prefix}_f1", "label": f"{label} F1", "format": "percent"},
                {"field": f"{prefix}_correct_repairs", "label": f"{label} 正确数", "format": "number"},
            ]
        )
    columns.append(
        {
            "field": "best_f1_delta",
            "label": "Best BGR F1 Δ",
            "format": "percent",
            "movement": True,
        }
    )
    return {
        "id": "primary_dataset_metrics",
        "title": "逐数据集完整测试指标",
        "subtitle": "BGR 使用 20% 逻辑 estimated-token 预算；correct repairs 为规范化精确匹配数。",
        "dataset": "primary_dataset_table",
        "sourceId": "method_metrics",
        "defaultSort": {"field": "best_f1_delta", "direction": "desc"},
        "density": "dense",
        "columns": columns,
        "layout": "full",
    }


def _aggregate_table_spec() -> dict[str, Any]:
    return {
        "id": "primary_aggregate_metrics",
        "title": "Micro / Macro 汇总",
        "subtitle": "Micro 先合并全部 cell 计数；Macro 对逐数据集 Precision、Recall 与 F1 做等权平均。",
        "dataset": "aggregate_metrics",
        "sourceId": "method_metrics",
        "defaultSort": {"field": "scope", "direction": "asc"},
        "density": "dense",
        "columns": [
            {"field": "scope", "label": "聚合", "type": "text"},
            {"field": "method", "label": "方法", "type": "text"},
            {"field": "precision", "label": "Precision", "format": "percent"},
            {"field": "recall", "label": "Recall", "format": "percent"},
            {"field": "f1", "label": "F1", "format": "percent"},
            {"field": "correct_repairs", "label": "正确修复", "format": "number"},
            {"field": "true_error_cells", "label": "错误 cells", "format": "number"},
        ],
        "layout": "full",
    }


def _budget_table_spec(source_id: str) -> dict[str, Any]:
    return {
        "id": "budget_curve_metrics",
        "title": "五点预算曲线明细",
        "subtitle": "每一行均为覆盖全部测试 cell 的 micro 指标；逻辑成本不因物理缓存命中而减免。",
        "dataset": "budget_curve",
        "sourceId": source_id,
        "defaultSort": {"field": "budget_share", "direction": "asc"},
        "density": "dense",
        "columns": [
            {"field": "backend_label", "label": "Backend", "type": "text"},
            {"field": "budget_share", "label": "预算", "format": "percent"},
            {"field": "precision", "label": "Precision", "format": "percent"},
            {"field": "recall", "label": "Recall", "format": "percent"},
            {"field": "f1", "label": "F1", "format": "percent"},
            {"field": "f1_delta_vs_baran", "label": "F1 Δ vs Baran", "format": "percent", "movement": True},
            {"field": "correct_repairs", "label": "正确修复", "format": "number"},
        ],
        "layout": "full",
    }


def _size_table_spec(source_id: str) -> dict[str, Any]:
    return {
        "id": "size_ablation_metrics",
        "title": "组大小消融明细",
        "subtitle": "固定 20% 预算并仅改变候选池允许的精确组大小。",
        "dataset": "size_ablation",
        "sourceId": source_id,
        "defaultSort": {"field": "group_size", "direction": "asc"},
        "density": "dense",
        "columns": [
            {"field": "backend_label", "label": "Backend", "type": "text"},
            {"field": "group_size", "label": "k", "format": "number"},
            {"field": "precision", "label": "Precision", "format": "percent"},
            {"field": "recall", "label": "Recall", "format": "percent"},
            {"field": "f1", "label": "F1", "format": "percent"},
            {"field": "f1_delta_vs_baran", "label": "F1 Δ vs Baran", "format": "percent", "movement": True},
            {"field": "correct_repairs", "label": "正确修复", "format": "number"},
        ],
        "layout": "full",
    }


def _cost_table_spec() -> dict[str, Any]:
    return {
        "id": "cost_audit_summary",
        "title": "成本账本汇总",
        "subtitle": "离线校准、逻辑在线预算和物理 provider 成本采用不同口径，不相互替代。",
        "dataset": "cost_summary",
        "sourceId": "api_cost_audit",
        "defaultSort": {"field": "scope", "direction": "asc"},
        "columns": [
            {"field": "scope", "label": "范围", "type": "text"},
            {"field": "calibration_estimated_tokens", "label": "校准 estimated tokens", "format": "number"},
            {"field": "calibration_provider_tokens", "label": "校准 provider tokens", "format": "number"},
            {"field": "online_logical_estimated_tokens", "label": "在线逻辑 tokens", "format": "number"},
            {"field": "provider_input_tokens", "label": "Provider input", "format": "number"},
            {"field": "provider_output_tokens", "label": "Provider output", "format": "number"},
            {"field": "provider_total_tokens", "label": "Provider total", "format": "number"},
            {"field": "logical_api_calls", "label": "逻辑调用", "format": "number"},
            {"field": "physical_api_calls", "label": "物理调用", "format": "number"},
            {"field": "unknown_token_attempts", "label": "未知 usage 尝试", "format": "number"},
            {"field": "provider_failed_records", "label": "Provider 失败终态", "format": "number"},
            {"field": "historical_failed_records", "label": "历史失败批次", "format": "number"},
            {"field": "operational_fallback_records", "label": "Baran 回退查询", "format": "number"},
            {"field": "operational_fallback_cells", "label": "Baran 回退 cells", "format": "number"},
            {"field": "unresolved_operational_failures", "label": "未解决失败", "format": "number"},
        ],
        "layout": "full",
    }


def _record_audit_table_spec() -> dict[str, Any]:
    return {
        "id": "record_audit_checks",
        "title": "最终记录覆盖与重算审计",
        "subtitle": "失败项表示该 run 不满足正式完成标准。",
        "dataset": "record_audit",
        "sourceId": "record_audit",
        "defaultSort": {"field": "check", "direction": "asc"},
        "columns": [
            {"field": "check", "label": "检查", "type": "text"},
            {"field": "status", "label": "状态", "type": "text"},
            {"field": "evidence_field", "label": "证据字段", "type": "text"},
            {"field": "value", "label": "值", "type": "text"},
        ],
        "layout": "full",
    }


def _split_audit_table_spec() -> dict[str, Any]:
    return {
        "id": "split_leakage_checks",
        "title": "跨 family 切分与 target 泄漏审计",
        "subtitle": "每项按全部 target family 取最坏值；“未报告”不能解释为通过。",
        "dataset": "split_audit",
        "sourceId": "split_audit",
        "defaultSort": {"field": "check", "direction": "asc"},
        "columns": [
            {"field": "check", "label": "检查", "type": "text"},
            {"field": "status", "label": "状态", "type": "text"},
            {"field": "targets", "label": "已报告 targets", "format": "number"},
            {"field": "worst_value", "label": "最坏值", "type": "text"},
        ],
        "layout": "full",
    }


def _report_blocks(
    *,
    title: str,
    headline: Mapping[str, Any],
    cost_card_ids: Sequence[str],
    include_cost: bool,
    include_split_audit: bool,
    split_rows: Sequence[Mapping[str, Any]],
    derived_recovery: bool,
) -> list[dict[str, Any]]:
    best = _backend_label(str(headline["best_backend"]))
    summary = (
        f"## 技术摘要\n\n"
        f"在 20% 逻辑 estimated-token 预算下，{best} 的 micro F1 为 "
        f"{_percent(headline['best_bgr_f1'])}，Baran 为 {_percent(headline['baran_f1'])}，"
        f"绝对差值 {_signed_percent(headline['best_bgr_f1_delta'])}。"
        f"LightGBM 与 XGBoost 分别在 {headline['lightgbm_dataset_wins']}/"
        f"{headline['dataset_count']} 和 {headline['xgboost_dataset_wins']}/"
        f"{headline['dataset_count']} 个数据集上取得高于 Baran 的 F1。"
        f"Macro F1 分别为 Baran {_percent(headline['baran_macro_f1'])}、"
        f"BGR-LightGBM {_percent(headline['lightgbm_macro_f1'])} 与 "
        f"BGR-XGBoost {_percent(headline['xgboost_macro_f1'])}。"
        "下文同时给出逐数据集 Precision/Recall/F1/正确修复数、五点预算曲线、"
        "组大小消融和可复核审计。"
    )
    split_failures = sum(row.get("status") == "失败" for row in split_rows)
    split_unreported = sum(row.get("status") == "未报告" for row in split_rows)
    if include_split_audit:
        leakage_note = (
            f"切分审计中失败项 {split_failures} 个、未报告项 {split_unreported} 个；"
            "只有“通过”项才构成该 run 的直接证据。"
        )
    else:
        leakage_note = "该 run 未提供 gates/split_audit.csv，因此报告不会把切分/target 泄漏状态宣称为已通过。"
    blocks: list[dict[str, Any]] = [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        *(
            [
                {
                    "id": "derived_recovery_note",
                    "type": "markdown",
                    "body": (
                        "## 运行来源说明\n\n该结果来自带 lineage 的派生恢复运行。"
                        "父运行的原始模型失败、重试次数和 token 账本保持不变；"
                        "无效响应按预先声明的操作语义回退 fresh Baran，未被改写为模型成功。"
                    ),
                    "sourceId": "run_manifest",
                }
            ]
            if derived_recovery
            else []
        ),
        {"id": "technical_summary", "type": "markdown", "body": summary, "sourceId": "method_metrics"},
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["baran_micro_f1", "lightgbm_micro_f1", "xgboost_micro_f1", "best_correct_repairs"]},
        {"id": "primary_heading", "type": "markdown", "body": "## 主结果：逐数据集 Baran vs BGR（20% 预算）\n\n图表使用相同的 correction-only 指标口径；表格保留精确数值用于复核。"},
        {"id": "aggregate_table_block", "type": "table", "tableId": "primary_aggregate_metrics"},
        {"id": "primary_chart_block", "type": "chart", "chartId": "primary_f1_by_dataset"},
        {"id": "primary_table_block", "type": "table", "tableId": "primary_dataset_metrics"},
        {"id": "budget_heading", "type": "markdown", "body": "## 预算敏感性\n\n比较 1%、5%、10%、20%、50% 五个逻辑 estimated-token 预算点；物理缓存命中不会改变任一场景的逻辑成本。", "sourceId": "method_spec"},
        {"id": "budget_chart_block", "type": "chart", "chartId": "micro_f1_budget_curve"},
        {"id": "budget_table_block", "type": "table", "tableId": "budget_curve_metrics"},
        {"id": "size_heading", "type": "markdown", "body": "## 组大小 1/2/4/8 消融\n\n所有设置固定为 20% 预算。k=1 只允许 singleton；k=2/4/8 分别允许 singleton 与 exact-size-k 分组。", "sourceId": "method_spec"},
        {"id": "size_chart_block", "type": "chart", "chartId": "micro_f1_size_ablation"},
        {"id": "size_table_block", "type": "table", "tableId": "size_ablation_metrics"},
        {"id": "definitions", "type": "markdown", "body": "## 指标与方法定义\n\n- **Baran**：以固定 labeling budget fresh 运行的完整基线。\n- **BGR-LightGBM / BGR-XGBoost**：用相应 gate 估计组查询的保守 helpful-minus-harmful 收益，在逻辑 token 预算内选择重叠组；未选中、解析失败或 verifier 拒绝的 cell 回退 Baran。\n- **Precision**：正确修复数除以 parse-valid 预测数。\n- **Recall / correction accuracy**：正确修复数除以 oracle error cells。\n- **F1**：Precision 与 Recall 的调和平均。", "sourceId": "method_spec"},
        {"id": "methodology", "type": "markdown", "body": "## 方法学\n\n正式测试矩阵覆盖指定 9 个数据集、LightGBM/XGBoost 两个 gate backend 与五个预算点，20% 额外报告 singleton-only 和 singleton+exact-4。Router 对每个 target base family 使用其余 TableEG family 的离线标签训练；Baran 特征仅进入 router，LLM repair messages 只包含 dirty-side evidence。物理 LLM 查询按严格 request identity 去重，而每个逻辑场景仍按完整查询成本计费。", "sourceId": "method_spec"},
    ]
    if include_cost:
        blocks.extend(
            [
                {"id": "cost_heading", "type": "markdown", "body": "## 成本与 API 使用\n\n离线校准成本、在线逻辑预算与去重后的物理 provider 成本分开报告，不能用物理 cache saving 重新解释某个场景的预算约束。", "sourceId": "api_cost_audit"},
                *([{"id": "cost_metrics", "type": "metric-strip", "cardIds": list(cost_card_ids)}] if cost_card_ids else []),
                {"id": "cost_table_block", "type": "table", "tableId": "cost_audit_summary"},
            ]
        )
    else:
        blocks.append({"id": "cost_missing", "type": "markdown", "body": "## 成本与 API 使用\n\nmetrics/api_cost_audit.csv 未提供；本报告无法核实实际 provider token 与物理 API 调用。"})
    blocks.extend(
        [
            {"id": "audit_heading", "type": "markdown", "body": f"## 泄漏与覆盖审计\n\n{leakage_note}"},
            {"id": "record_audit_block", "type": "table", "tableId": "record_audit_checks"},
        ]
    )
    if include_split_audit:
        blocks.append({"id": "split_audit_block", "type": "table", "tableId": "split_leakage_checks"})
    blocks.extend(
        [
            {"id": "limitations", "type": "markdown", "body": "## 局限性与稳健性\n\n- 实验假设错误 cell 坐标已知，因此结论针对 correction 而非 error detection。\n- 组收益依赖有限校准样本与单次固定随机种子；应通过多种子重复与置信区间验证稳定性。\n- Estimated tokens 是选择约束，provider usage 是事后物理成本；两者不应混为同一变量。\n- 报告是正式 run 的静态快照，不会随 API、模型版本或数据变化自动刷新。\n- 本地 CARE 数据的再分发许可尚未由本实验确认。", "sourceId": "method_spec"},
            {"id": "next_steps", "type": "markdown", "body": "## 下一步\n\n1. 用多个 Baran/gate/selection 随机种子重复主实验并报告 bootstrap 置信区间。\n2. 对 F1 增益集中的数据集做 group-view、cohesion 与 verifier rejection 分层诊断。\n3. 将 provider token 与端到端延迟纳入同一成本—质量前沿，但保留逻辑预算口径。"},
            {"id": "further_questions", "type": "markdown", "body": "## 可进一步追问的问题\n\n- 增益主要来自更大的组，还是来自更好的重叠覆盖？\n- 两个 gate backend 的差异是否集中在特定 base family？\n- Verifier 避免了多少 harmful overwrite，又放弃了多少 helpful upgrade？\n- 若按实际 provider token 而非 estimated token 约束，方法排序是否改变？"},
        ]
    )
    return blocks


def _assert_references_resolve(
    blocks: Sequence[Mapping[str, Any]],
    cards: Sequence[Mapping[str, Any]],
    charts: Sequence[Mapping[str, Any]],
    tables: Sequence[Mapping[str, Any]],
    sources: set[str],
) -> None:
    card_ids = {str(item["id"]) for item in cards}
    chart_ids = {str(item["id"]) for item in charts}
    table_ids = {str(item["id"]) for item in tables}
    for item in [*cards, *charts, *tables]:
        source_id = item.get("sourceId")
        if source_id and source_id not in sources:
            raise AssertionError(f"unresolved sourceId {source_id!r}")
    for block in blocks:
        source_id = block.get("sourceId")
        if source_id and source_id not in sources:
            raise AssertionError(f"unresolved block sourceId {source_id!r}")
        if block.get("type") == "metric-strip" and not set(block.get("cardIds", ())) <= card_ids:
            raise AssertionError("unresolved metric card reference")
        if block.get("type") == "chart" and block.get("chartId") not in chart_ids:
            raise AssertionError("unresolved chart reference")
        if block.get("type") == "table" and block.get("tableId") not in table_ids:
            raise AssertionError("unresolved table reference")


def _assert_artifact_safe(artifact: Mapping[str, Any]) -> None:
    if artifact.get("surface") != "report" or artifact.get("manifest", {}).get("surface") != "report":
        raise AssertionError("report surface must be explicit at both levels")
    datasets = artifact.get("snapshot", {}).get("datasets", {})
    if not isinstance(datasets, dict) or not datasets:
        raise AssertionError("snapshot.datasets must be a non-empty mapping")
    for source in artifact.get("manifest", {}).get("sources", []):
        path = str(source.get("path", ""))
        posix = Path(path)
        if not path or posix.is_absolute() or ".." in posix.parts or "~" in posix.parts:
            raise AssertionError(f"unsafe report source path: {path!r}")
    for block in artifact.get("manifest", {}).get("blocks", []):
        if block.get("type") == "html":
            raise AssertionError("ordinary BGR reports must use native blocks only")
    serialized = json.dumps(artifact, ensure_ascii=False).lower()
    for forbidden in ("api_key", "access_token", "private_key", "password"):
        if forbidden in serialized:
            raise AssertionError(f"sensitive field name entered artifact: {forbidden}")


def _parse_builder_receipt(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        return {}
    for candidate in (text, *reversed(text.splitlines())):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"ok": False, "error": "portable builder returned a non-JSON receipt"}


def _deduplicate_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if _is_baran(row):
            grouped.setdefault(_dataset(row), []).append(row)
    return {key: _choose_baseline(group, key) for key, group in grouped.items()}


def _choose_baseline(rows: Sequence[Mapping[str, Any]], label: str) -> Mapping[str, Any] | None:
    candidates = [row for row in rows if _is_baran(row)]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda row: (
            0 if str(row.get("scenario") or "main").lower() == "main" else 1,
            0 if _variant_is_all(row) else 1,
            0 if _number(row, "budget_share", "budget", "beta") is None else 1,
            json.dumps(dict(row), sort_keys=True),
        ),
    )
    chosen = ranked[0]
    chosen_metrics = _metric_row(chosen)
    for other in ranked[1:]:
        if any(not _same_number(chosen_metrics[field], _metric_row(other)[field]) for field in ("precision", "recall", "f1", "correct_repairs", "true_error_cells")):
            raise ReportInputError(f"Baran rows disagree for {label}")
    return chosen


def _choose_unique(rows: Sequence[Mapping[str, Any]], label: str) -> Mapping[str, Any]:
    first = rows[0]
    signature = _metric_row(first)
    for row in rows[1:]:
        current = _metric_row(row)
        if any(not _same_number(signature[field], current[field]) for field in signature if field in current):
            raise ReportInputError(f"duplicate metric rows disagree for {label}")
    return first


def _choose_ablation(rows: Sequence[Mapping[str, Any]], backend: str, size: int) -> Mapping[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if "ablation" in str(row.get("scenario") or "").lower() else 1,
            0 if str(row.get("group_size_variant") or row.get("variant") or "") == str(size) else 1,
            json.dumps(dict(row), sort_keys=True),
        ),
    )
    chosen = ranked[0]
    chosen_metrics = _metric_row(chosen)
    same_rank = [row for row in ranked if ("ablation" in str(row.get("scenario") or "").lower()) == ("ablation" in str(chosen.get("scenario") or "").lower())]
    for other in same_rank[1:]:
        current = _metric_row(other)
        if any(not _same_number(chosen_metrics[field], current[field]) for field in ("precision", "recall", "f1", "correct_repairs")):
            raise ReportInputError(f"group-size rows disagree for {backend}, k={size}")
    return chosen


def _scope(row: Mapping[str, Any]) -> str:
    scope = str(row.get("scope") or "").strip().lower()
    if scope in {"dataset", "micro", "macro"}:
        return scope
    dataset = str(row.get("dataset") or "").strip().upper()
    if dataset == "MICRO":
        return "micro"
    if dataset == "MACRO":
        return "macro"
    if dataset and dataset != "ALL":
        return "dataset"
    return "unspecified"


def _dataset(row: Mapping[str, Any]) -> str:
    dataset = str(row.get("dataset") or "").strip()
    suite = str(row.get("suite") or "").strip()
    if not dataset:
        raise ReportInputError("metric row has no dataset")
    return f"{suite}/{dataset}" if suite and suite.upper() != "ALL" else dataset


def _backend(row: Mapping[str, Any]) -> str:
    backend = str(row.get("backend") or "").strip().lower()
    method = str(row.get("method") or row.get("experiment") or "").strip().lower()
    if "lightgbm" in backend or "lightgbm" in method or "lgbm" in backend or "lgbm" in method:
        return "lightgbm"
    if "xgboost" in backend or "xgboost" in method or backend == "xgb" or "_xgb" in method:
        return "xgboost"
    if method == "baran" or backend in {"", "none", "baran"} and "baran" in method:
        return "baran"
    return backend or method


def _backend_label(backend: str) -> str:
    return {"lightgbm": "BGR-LightGBM", "xgboost": "BGR-XGBoost", "baran": "Baran"}.get(backend, backend)


def _is_baran(row: Mapping[str, Any]) -> bool:
    method = str(row.get("method") or row.get("experiment") or "").strip().lower()
    return method == "baran" or (_backend(row) == "baran" and "budgeted" not in method)


def _is_primary_scenario(row: Mapping[str, Any]) -> bool:
    scenario = str(row.get("scenario") or "main").strip().lower()
    return scenario in {"main", "overlapping", "overlapping_group_router", "all"}


def _variant_is_all(row: Mapping[str, Any]) -> bool:
    variant = str(row.get("group_size_variant") or row.get("variant") or "all").strip().lower()
    return variant in {"", "all", "main", "all_sizes", "overlapping"}


def _variant_size(row: Mapping[str, Any]) -> int | None:
    raw = str(row.get("group_size_variant") or row.get("variant") or row.get("group_size") or "").strip().lower()
    if not raw or raw in {"all", "main", "all_sizes"}:
        return None
    if raw.isdigit():
        return int(raw)
    matches = re.findall(r"(?:^|[^0-9])(?:k|size|exact)[_=-]?(1|2|4|8)(?:[^0-9]|$)", raw)
    if matches:
        return int(matches[-1])
    return None


def _is_total_row(row: Mapping[str, Any]) -> bool:
    return any(
        str(row.get(field) or "").strip().lower() in {"total", "all", "micro", "formal_run"}
        for field in ("scope", "dataset", "stage", "row")
    )


def _is_calibration_row(row: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(row.get(field) or "").strip().lower()
        for field in ("scope", "stage", "scenario", "method", "phase")
    )
    return "calibrat" in text or "offline" in text


def _raw(row: Mapping[str, Any], *keys: str) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for key in keys:
        if key in row and row[key] not in {None, ""}:
            return row[key]
        value = lowered.get(key.lower())
        if value not in {None, ""}:
            return value
    return None


def _number(row: Mapping[str, Any], *keys: str) -> float | None:
    return _coerce_number(_raw(row, *keys))


def _integer(row: Mapping[str, Any], *keys: str) -> int | None:
    value = _number(row, *keys)
    return None if value is None else int(round(value))


def _required_number(row: Mapping[str, Any], *keys: str) -> float:
    value = _number(row, *keys)
    if value is None:
        raise ReportInputError(f"metric row is missing numeric field {keys[0]!r}")
    return value


def _required_integer(row: Mapping[str, Any], *keys: str) -> int:
    value = _integer(row, *keys)
    if value is None:
        raise ReportInputError(f"metric row is missing integer field {keys[0]!r}")
    return value


def _coerce_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if not text or text.lower() in {"none", "null", "nan", "unknown", "n/a", "—"}:
            return None
        if text.endswith("%"):
            try:
                number = float(text[:-1]) / 100.0
            except ValueError:
                return None
        else:
            try:
                number = float(text)
            except ValueError:
                return None
    return number if math.isfinite(number) else None


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _same_number(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def _count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set, dict)):
        return len(value)
    number = _coerce_number(value)
    return int(number) if number is not None else 1


def _percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _signed_percent(value: Any) -> str:
    return f"{float(value):+.2%}"


def _safe_run_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return cleaned[:80] or "formal-run"


__all__ = [
    "DEFAULT_BUDGET_SHARES",
    "DEFAULT_GROUP_SIZES",
    "EXPECTED_DATASET_COUNT",
    "EXPECTED_ERROR_CELL_COUNT",
    "PRIMARY_BUDGET_SHARE",
    "PortableReportError",
    "ReportBuildResult",
    "ReportInputError",
    "ReportInputs",
    "build_report",
    "build_router_report",
    "build_report_artifact",
    "deliver_portable_report",
    "discover_data_analytics_plugin_root",
    "discover_node_executable",
    "generate_report",
    "resolve_report_inputs",
    "write_report_artifact",
]
