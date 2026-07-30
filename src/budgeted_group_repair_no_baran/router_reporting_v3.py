"""Dataset-first Markdown, HTML, and JSON reporting for Router-v3."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VARIANTS = ("1", "2", "4", "8", "all")
BACKENDS = ("lightgbm", "xgboost")
SWEEP_VARIANTS = ("2", "4")
SWEEP_BUDGETS = (0.01, 0.05, 0.1, 0.2, 0.5)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Router-v3 report input is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Router-v3 report input is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Router-v3 report JSON must be an object: {path}")
    return value


def _calibration_provenance_path(root: Path) -> Path:
    current = root / "provenance" / "calibration.json"
    if current.is_file():
        return current
    frozen = root / "provenance" / "reuse_manifest.json"
    if frozen.is_file():
        return frozen
    raise ValueError("Router-v3 report calibration provenance is missing")


def _method_label(row: Mapping[str, object]) -> str:
    method = str(row.get("method", ""))
    if method == "baran":
        return "Baran-only"
    if method == "llm_only":
        return "LLM-only"
    backend = str(row.get("backend", ""))
    variant = str(row.get("group_size_variant", ""))
    return f"BGR-{backend} k={variant}"


def _number(value: object, digits: int = 4) -> str:
    if value in {None, ""}:
        return "—"
    return f"{float(value):.{digits}f}"


def _integer(value: object) -> str:
    if value in {None, ""}:
        return "—"
    return f"{int(float(value)):,}"


def _markdown_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            str(row.get(field, "")).replace("|", "\\|").replace("\n", " ")
            for field, _ in columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join((header, divider, *body))


def _html_table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(row.get(field, '')))}</td>"
            for field, _ in columns
        )
        + "</tr>"
        for row in rows
    )
    return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _dataset_name(row: Mapping[str, object]) -> str:
    suite = str(row.get("suite", ""))
    dataset = str(row.get("dataset", ""))
    return f"{suite}/{dataset}" if suite else dataset


def _prepare_f1_matrix(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for row in rows:
        item: dict[str, object] = {
            "dataset": _dataset_name(row),
            "baran": _number(row["baran_only_f1"]),
            "llm": _number(row["llm_only_f1"]),
        }
        for backend in BACKENDS:
            for variant in VARIANTS:
                item[f"{backend}_{variant}"] = _number(
                    row[f"bgr_{backend}_k{variant}_f1"]
                )
        prepared.append(item)
    return prepared


def _prepare_detailed(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    prepared: list[dict[str, object]] = []
    for row in rows:
        prepared.append(
            {
                "dataset": _dataset_name(row),
                "method": _method_label(row),
                "correct": _integer(row.get("correct_repairs")),
                "predicted": _integer(row.get("predicted_repairs")),
                "precision": _number(row.get("precision")),
                "recall": _number(row.get("correction_accuracy", row.get("recall"))),
                "f1": _number(row.get("f1")),
                "delta_baran": _number(row.get("delta_f1_vs_baran")),
                "delta_llm": _number(row.get("delta_f1_vs_llm_only")),
                "llm_cells": _integer(row.get("llm_upgraded_cells")),
                "logical_calls": _integer(row.get("logical_calls")),
                "estimated_tokens": _integer(row.get("logical_estimated_tokens")),
                "physical_calls": _integer(row.get("physical_calls_charged")),
                "observed_tokens": _integer(
                    row.get("logical_provider_tokens_observed")
                ),
            }
        )
    return prepared


def _prepare_aggregate(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "scope": str(row.get("scope", "")),
            "method": _method_label(row),
            "correct": _integer(row.get("correct_repairs")),
            "predicted": _integer(row.get("predicted_repairs")),
            "precision": _number(row.get("precision")),
            "recall": _number(row.get("correction_accuracy", row.get("recall"))),
            "f1": _number(row.get("f1")),
        }
        for row in rows
        if str(row.get("scope", "")) in {"micro", "macro"}
    ]


def _win_tie_loss(
    detailed: Sequence[Mapping[str, object]],
    *,
    backends: Sequence[str] = BACKENDS,
    variants: Sequence[str] = VARIANTS,
) -> list[dict[str, object]]:
    index = {
        (
            str(row.get("suite", "")),
            str(row.get("dataset", "")),
            str(row.get("method", "")),
            str(row.get("backend", "")),
            str(row.get("group_size_variant", "")),
        ): float(row.get("f1", 0.0))
        for row in detailed
    }
    result: list[dict[str, object]] = []
    datasets = sorted(
        {
            (str(row.get("suite", "")), str(row.get("dataset", "")))
            for row in detailed
        }
    )
    for backend in backends:
        for variant in variants:
            method_key = (f"budgeted_group_{backend}", backend, variant)
            for baseline, baseline_key in (
                ("Baran-only", ("baran", "none", "all")),
                ("LLM-only", ("llm_only", "none", "1")),
            ):
                deltas = [
                    index[(*dataset, *method_key)] - index[(*dataset, *baseline_key)]
                    for dataset in datasets
                ]
                result.append(
                    {
                        "method": f"BGR-{backend} k={variant}",
                        "baseline": baseline,
                        "win": sum(value > 1e-12 for value in deltas),
                        "tie": sum(abs(value) <= 1e-12 for value in deltas),
                        "loss": sum(value < -1e-12 for value in deltas),
                    }
                )
    return result


def _paired_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "dataset": _dataset_name(row),
            "method": f"BGR-{row['backend']} k={row['group_size_variant']}",
            "baseline": str(row["baseline"]),
            "delta_f1": _number(row["delta_f1"]),
            "ci": (
                f"[{_number(row['delta_f1_ci_low'])}, "
                f"{_number(row['delta_f1_ci_high'])}]"
            ),
            "holm_p": _number(row["holm_adjusted_p_value"]),
        }
        for row in rows
    ]


def _cost_summary(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "phase": str(row.get("phase", "")),
            "records": _integer(row.get("records")),
            "physical_requests": _integer(row.get("physical_requests")),
            "cache_hits": _integer(row.get("cache_hits")),
            "provider_tokens": _integer(row.get("total_tokens")),
            "failed": _integer(row.get("failed_records")),
        }
        for row in rows
    ]


def _source_rows(root: Path, paths: Iterable[Path]) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _budget_label(value: float) -> str:
    return f"{int(round(value * 100)):02d}pct"


def _sweep_method_label(row: Mapping[str, object]) -> str:
    base = _method_label(row)
    if str(row.get("method", "")).startswith("budgeted_group_"):
        return f"{base} · {int(round(float(row.get('budget_share', 0)) * 100))}%"
    return base


def _budget_curve_svg(
    rows: Sequence[Mapping[str, object]],
    *,
    scope: str,
) -> str:
    """Render an honest 0–1 F1 budget curve with non-color distinctions."""

    width, height = 920, 390
    left, right, top, bottom = 70, 28, 36, 58
    plot_width = width - left - right
    plot_height = height - top - bottom
    budgets = list(SWEEP_BUDGETS)
    by_variant = {
        variant: {
            round(float(row["budget_share"]), 12): float(row["f1"])
            for row in rows
            if str(row.get("scope")) == scope
            and str(row.get("method")) == "budgeted_group_lightgbm"
            and str(row.get("group_size_variant")) == variant
        }
        for variant in SWEEP_VARIANTS
    }
    baselines = {
        str(row.get("method")): float(row["f1"])
        for row in rows
        if str(row.get("scope")) == scope
        and str(row.get("method")) in {"baran", "llm_only"}
    }
    if any(set(values) != set(budgets) for values in by_variant.values()) or set(
        baselines
    ) != {"baran", "llm_only"}:
        raise ValueError(f"budget curve source is incomplete for {scope}")

    def x(index: int) -> float:
        return left + index * plot_width / (len(budgets) - 1)

    def y(value: float) -> float:
        return top + (1.0 - value) * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(scope)} F1 by budget">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="22" fill="#172033" font-size="17" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">{html.escape(scope.title())} F1 by logical budget</text>',
    ]
    for tick in range(0, 11, 2):
        value = tick / 10
        y_value = y(value)
        parts.append(
            f'<line x1="{left}" y1="{y_value:.2f}" x2="{width-right}" y2="{y_value:.2f}" stroke="#dce1ea" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y_value+4:.2f}" text-anchor="end" fill="#5d6678" font-size="12">{value:.1f}</text>'
        )
    for index, budget in enumerate(budgets):
        x_value = x(index)
        parts.append(
            f'<text x="{x_value:.2f}" y="{height-27}" text-anchor="middle" fill="#5d6678" font-size="12">{int(budget*100)}%</text>'
        )
    for method, dash, label in (
        ("baran", "7 5", "Baran-only"),
        ("llm_only", "2 5", "LLM-only"),
    ):
        y_value = y(baselines[method])
        parts.append(
            f'<line x1="{left}" y1="{y_value:.2f}" x2="{width-right}" y2="{y_value:.2f}" stroke="#697386" stroke-width="1.5" stroke-dasharray="{dash}"/>'
        )
        parts.append(
            f'<text x="{width-right-4}" y="{y_value-5:.2f}" text-anchor="end" fill="#5d6678" font-size="11">{label} {baselines[method]:.4f}</text>'
        )
    for variant, color, marker in (
        ("2", "#2656a8", "circle"),
        ("4", "#b47a00", "square"),
    ):
        points = [
            (x(index), y(by_variant[variant][budget]))
            for index, budget in enumerate(budgets)
        ]
        parts.append(
            '<polyline fill="none" '
            f'stroke="{color}" stroke-width="3" points="'
            + " ".join(f"{x_value:.2f},{y_value:.2f}" for x_value, y_value in points)
            + '"/>'
        )
        for x_value, y_value in points:
            if marker == "circle":
                parts.append(
                    f'<circle cx="{x_value:.2f}" cy="{y_value:.2f}" r="5" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>'
                )
            else:
                parts.append(
                    f'<rect x="{x_value-4.5:.2f}" y="{y_value-4.5:.2f}" width="9" height="9" fill="#ffffff" stroke="{color}" stroke-width="2.5"/>'
                )
        last_x, last_y = points[-1]
        parts.append(
            f'<text x="{last_x-8:.2f}" y="{last_y-10:.2f}" text-anchor="end" fill="{color}" font-size="12" font-weight="600">k={variant}</text>'
        )
    parts.append(
        f'<text x="{left+plot_width/2:.2f}" y="{height-7}" text-anchor="middle" fill="#5d6678" font-size="12">Budget share of full singleton estimated-token cost</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _build_budget_sweep_report(
    root: Path,
    validation: Mapping[str, object],
    output_path: str | Path | None,
) -> dict[str, Any]:
    metrics = root / "metrics"
    input_paths = (
        metrics / "per_dataset_f1_matrix.csv",
        metrics / "per_dataset_method_comparison.csv",
        metrics / "method_metrics.csv",
        metrics / "paired_statistics.csv",
        metrics / "api_cost_audit.csv",
        metrics / "selection_audit.csv",
        metrics / "aubc.csv",
        metrics / "budget_curves.csv",
        metrics / "record_audit.json",
        metrics / "formal_run_audit.json",
        _calibration_provenance_path(root),
        root / "provenance" / "router_artifact_reuse.json",
        root / "provenance" / "response_reuse.json",
        root / "llm" / "router_v3_budget_sweep_dry_plan.json",
        root / "run_manifest.json",
    )
    f1_rows = _read_csv(input_paths[0])
    detailed_rows = _read_csv(input_paths[1])
    metric_rows = _read_csv(input_paths[2])
    paired_rows = _read_csv(input_paths[3])
    cost_rows = _read_csv(input_paths[4])
    selection_rows = _read_csv(input_paths[5])
    aubc_rows = _read_csv(input_paths[6])
    budget_rows = _read_csv(input_paths[7])
    record_audit = _read_json(input_paths[8])
    formal_audit = _read_json(input_paths[9])
    reuse = _read_json(input_paths[10])
    gate_reuse = _read_json(input_paths[11])
    response_reuse = _read_json(input_paths[12])
    dry_plan = _read_json(input_paths[13])
    manifest = _read_json(input_paths[14])
    if (
        len(f1_rows) != 9
        or len(detailed_rows) != 108
        or len(metric_rows) != 132
        or len(paired_rows) != 180
        or len(selection_rows) != 90
        or len(aubc_rows) != 22
        or len(budget_rows) != 110
        or record_audit.get("records") != 266_376
        or formal_audit.get("ok") is not True
        or int(validation.get("split_rows", -1)) != 18
    ):
        raise ValueError("Router-v3 budget-sweep report inputs failed acceptance counts")

    matrix_by_variant: dict[str, list[dict[str, object]]] = {}
    for variant in SWEEP_VARIANTS:
        prepared: list[dict[str, object]] = []
        for row in f1_rows:
            item: dict[str, object] = {
                "dataset": _dataset_name(row),
                "baran_f1": _number(row["baran_only_f1"]),
                "llm_f1": _number(row["llm_only_f1"]),
                "llm_cells": _integer(row["llm_only_valid_llm_cells"]),
            }
            for budget in SWEEP_BUDGETS:
                prefix = f"bgr_lightgbm_k{variant}_{_budget_label(budget)}"
                label = str(int(round(budget * 100)))
                item[f"f1_{label}"] = _number(row[f"{prefix}_f1"])
                item[f"cells_{label}"] = _integer(row[f"{prefix}_llm_cells"])
            prepared.append(item)
        matrix_by_variant[variant] = prepared

    detailed = [
        {
            "dataset": _dataset_name(row),
            "method": _sweep_method_label(row),
            "correct": _integer(row.get("correct_repairs")),
            "predicted": _integer(row.get("predicted_repairs")),
            "precision": _number(row.get("precision")),
            "recall": _number(row.get("correction_accuracy", row.get("recall"))),
            "f1": _number(row.get("f1")),
            "llm_cells": _integer(row.get("llm_upgraded_cells")),
            "delta_baran": _number(row.get("delta_f1_vs_baran")),
            "delta_llm": _number(row.get("delta_f1_vs_llm_only")),
            "logical_calls": _integer(row.get("logical_calls")),
            "estimated_tokens": _integer(row.get("logical_estimated_tokens")),
            "physical_calls": _integer(row.get("physical_calls_charged")),
            "observed_tokens": _integer(row.get("logical_provider_tokens_observed")),
        }
        for row in detailed_rows
    ]
    aggregate = _prepare_aggregate(metric_rows)
    paired = [
        {
            "dataset": _dataset_name(row),
            "method": f"BGR-LightGBM k={row['group_size_variant']} · {int(round(float(row['budget_share'])*100))}%",
            "baseline": str(row["baseline"]),
            "delta_f1": _number(row["delta_f1"]),
            "ci": f"[{_number(row['delta_f1_ci_low'])}, {_number(row['delta_f1_ci_high'])}]",
            "holm_p": _number(row["holm_adjusted_p_value"]),
        }
        for row in paired_rows
    ]
    index = {
        (
            str(row["suite"]),
            str(row["dataset"]),
            str(row["method"]),
            str(row["group_size_variant"]),
            (
                None
                if str(row["method"]) in {"baran", "llm_only"}
                else round(float(row["budget_share"]), 12)
            ),
        ): float(row["f1"])
        for row in detailed_rows
    }
    datasets = sorted({(str(row["suite"]), str(row["dataset"])) for row in detailed_rows})
    wtl: list[dict[str, object]] = []
    for variant in SWEEP_VARIANTS:
        for budget in SWEEP_BUDGETS:
            for baseline, baseline_method, baseline_variant in (
                ("Baran-only", "baran", "all"),
                ("LLM-only", "llm_only", "1"),
            ):
                deltas = [
                    index[(*dataset, "budgeted_group_lightgbm", variant, budget)]
                    - index[(*dataset, baseline_method, baseline_variant, None)]
                    for dataset in datasets
                ]
                wtl.append(
                    {
                        "method": f"k={variant} · {int(budget*100)}%",
                        "baseline": baseline,
                        "win": sum(value > 1e-12 for value in deltas),
                        "tie": sum(abs(value) <= 1e-12 for value in deltas),
                        "loss": sum(value < -1e-12 for value in deltas),
                    }
                )
    aubc = [
        {
            "scope": str(row["scope"]),
            "dataset": _dataset_name(row),
            "k": str(row["group_size_variant"]),
            "baran_anchor": _number(row["baseline_value"]),
            "f1_aubc": _number(row["f1_aubc"]),
        }
        for row in aubc_rows
    ]
    costs = _cost_summary(cost_rows)
    macro_rows = [row for row in metric_rows if str(row.get("scope")) == "macro"]
    best_macro = max(
        (
            row
            for row in macro_rows
            if str(row.get("method")) == "budgeted_group_lightgbm"
        ),
        key=lambda row: float(row["f1"]),
    )
    title = "Router-v3：LightGBM k=2/4 多预算实验"
    generated_at = str(
        manifest.get("completed_at")
        or manifest.get("updated_at")
        or datetime.now(timezone.utc).isoformat()
    )
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    macro_svg_path = report_dir / "budget_curve_macro.svg"
    micro_svg_path = report_dir / "budget_curve_micro.svg"
    macro_svg_path.write_text(
        _budget_curve_svg(metric_rows, scope="macro"), encoding="utf-8"
    )
    micro_svg_path.write_text(
        _budget_curve_svg(metric_rows, scope="micro"), encoding="utf-8"
    )
    main_columns = [
        ("dataset", "Dataset"),
        ("baran_f1", "Baran F1"),
        ("llm_f1", "LLM-only F1"),
        ("llm_cells", "LLM valid cells"),
        *[
            item
            for budget in SWEEP_BUDGETS
            for item in (
                (f"f1_{int(budget*100)}", f"BGR {int(budget*100)}% F1"),
                (f"cells_{int(budget*100)}", f"BGR {int(budget*100)}% LLM cells"),
            )
        ],
    ]
    detailed_columns = (
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("correct", "Correct"),
        ("predicted", "Predicted"),
        ("precision", "Precision"),
        ("recall", "Recall/CA"),
        ("f1", "F1"),
        ("llm_cells", "LLM cells"),
        ("delta_baran", "ΔF1 vs Baran"),
        ("delta_llm", "ΔF1 vs LLM"),
        ("logical_calls", "Logical calls"),
        ("estimated_tokens", "Logical est. tokens"),
        ("physical_calls", "Physical calls"),
        ("observed_tokens", "Observed tokens"),
    )
    paired_columns = (
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("baseline", "Baseline"),
        ("delta_f1", "ΔF1"),
        ("ci", "95% row-cluster CI"),
        ("holm_p", "Holm p"),
    )
    aubc_columns = (
        ("scope", "Scope"),
        ("dataset", "Dataset"),
        ("k", "k"),
        ("baran_anchor", "Baran anchor"),
        ("f1_aubc", "F1 AUBC"),
    )
    wtl_columns = (
        ("method", "Method"),
        ("baseline", "Baseline"),
        ("win", "Win"),
        ("tie", "Tie"),
        ("loss", "Loss"),
    )
    aggregate_columns = (
        ("scope", "Scope"),
        ("method", "Method"),
        ("correct", "Correct"),
        ("predicted", "Predicted"),
        ("precision", "Precision"),
        ("recall", "Recall/CA"),
        ("f1", "F1"),
    )
    cost_columns = (
        ("phase", "Phase"),
        ("records", "Logical records"),
        ("physical_requests", "Physical requests"),
        ("cache_hits", "Cache hits"),
        ("provider_tokens", "Provider tokens"),
        ("failed", "Failed"),
    )
    technical_summary = (
        f"完整矩阵覆盖 9 个数据集、22,198 个 error cells。Dataset-Macro 最佳切片是 "
        f"k={best_macro['group_size_variant']}、{int(round(float(best_macro['budget_share'])*100))}% "
        f"预算，F1={float(best_macro['f1']):.4f}。20% 切片已逐 cell 与父 Router-v3 对账。"
    )
    markdown = "\n\n".join(
        (
            f"# {title}",
            "## 技术摘要\n\n" + technical_summary,
            "## k=2：逐数据集 F1 与升级为 LLM 修复的 cells\n\n"
            + _markdown_table(matrix_by_variant["2"], main_columns),
            "## k=4：逐数据集 F1 与升级为 LLM 修复的 cells\n\n"
            + _markdown_table(matrix_by_variant["4"], main_columns),
            "## 预算曲线\n\n"
            "预算轴是相对全量 singleton estimated-token 成本的逻辑比例；虚线为两条 baseline。\n\n"
            "![Dataset-Macro budget curve](budget_curve_macro.svg)\n\n"
            "![Micro budget curve](budget_curve_micro.svg)",
            "## 每个 k 的 AUBC\n\n" + _markdown_table(aubc, aubc_columns),
            "## 逐数据集详细长表\n\n" + _markdown_table(detailed, detailed_columns),
            "## Dirty-row cluster paired bootstrap\n\n"
            "每个 baseline × k × budget 的 9 数据集族内做 Holm 校正；2,000 次，seed=45。\n\n"
            + _markdown_table(paired, paired_columns),
            "## Win / Tie / Loss\n\n" + _markdown_table(wtl, wtl_columns),
            "## Micro 与 Dataset-Macro（补充）\n\n"
            + _markdown_table(aggregate, aggregate_columns),
            "## 成本与复用（补充）\n\n"
            + _markdown_table(costs, cost_columns)
            + "\n\n"
            + (
                f"Selection={len(selection_rows)}；model folds={validation['split_rows']}；"
                f"父响应复用 success={response_reuse.get('imported_success_rows', 0)}、"
                f"terminal failure={response_reuse.get('imported_terminal_failure_rows', 0)}；"
                f"dry-plan missing physical queries={dry_plan.get('online_physical_queries', 0)}。"
            ),
            "## 限制、稳健性与下一步\n\n"
            "预算曲线是同一冻结 Router 预测下的选择效果，不是重新训练五次；相邻预算的选中集合不强制嵌套。"
            "Bootstrap 衡量 paired 结果不确定性，但不建立因果结论。",
            f"生成时间：{generated_at}",
        )
    ) + "\n"
    markdown_path = report_dir / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else report_dir / "report.html"
    )
    sections = [
        f"<h1>{html.escape(title)}</h1>",
        f"<h2>技术摘要</h2><p>{html.escape(technical_summary)}</p>",
    ]
    for variant in SWEEP_VARIANTS:
        sections.extend(
            (
                f"<h2>k={variant}：逐数据集 F1 与升级为 LLM 修复的 cells</h2>",
                _html_table(matrix_by_variant[variant], main_columns),
            )
        )
    sections.extend(
        (
            "<h2>预算曲线</h2><p>预算轴为相对全量 singleton estimated-token 成本的逻辑比例；虚线为 baseline。</p>",
            macro_svg_path.read_text(encoding="utf-8"),
            micro_svg_path.read_text(encoding="utf-8"),
            "<h2>每个 k 的 AUBC</h2>",
            _html_table(aubc, aubc_columns),
            "<h2>逐数据集详细长表</h2>",
            _html_table(detailed, detailed_columns),
            "<h2>Dirty-row cluster paired bootstrap</h2><p>2,000 次，seed=45；在 baseline × k × budget 的 9 数据集族内做 Holm 校正。</p>",
            _html_table(paired, paired_columns),
            "<h2>Win / Tie / Loss</h2>",
            _html_table(wtl, wtl_columns),
            "<h2>Micro 与 Dataset-Macro（补充）</h2>",
            _html_table(aggregate, aggregate_columns),
            "<h2>成本与复用（补充）</h2>",
            _html_table(costs, cost_columns),
            "<h2>限制、稳健性与下一步</h2><p>五个预算共享同一冻结 Router 预测，仅选择独立；集合不强制嵌套。Bootstrap 不建立因果结论。</p>",
        )
    )
    html_document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light dark"><title>{html.escape(title)}</title><style>
:root{{--ink:#172033;--muted:#5d6678;--line:#dce1ea;--paper:#fff;--page:#f4f6fa;--accent:#2656a8}}@media(prefers-color-scheme:dark){{:root{{--ink:#e8edf7;--muted:#aeb7c8;--line:#39445a;--paper:#151b27;--page:#0f1420;--accent:#78a7ff}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1600px;margin:24px auto;padding:36px;background:var(--paper)}}h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:34px;border-bottom:2px solid var(--accent);padding-bottom:7px}}.table-wrap{{overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{position:sticky;top:0;background:color-mix(in srgb,var(--paper) 85%,var(--accent));color:var(--ink)}}svg{{display:block;width:100%;height:auto;margin:14px 0 24px;border:1px solid var(--line)}}
</style></head><body><main>{''.join(sections)}<p>{html.escape(generated_at)}</p></main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_document, encoding="utf-8")
    artifact = {
        "schema_version": "bgr-router-v3-budget-sweep-report-v1",
        "surface": "report",
        "title": title,
        "run_id": root.name,
        "generated_at": generated_at,
        "validation": validation,
        "dimensions": {
            "datasets": 9,
            "error_cells": 22_198,
            "method_slices": 12,
            "cell_records": 266_376,
            "selection_slices": 90,
            "model_folds": 18,
            "budget_shares": list(SWEEP_BUDGETS),
            "variants": list(SWEEP_VARIANTS),
            "backends": ["lightgbm"],
        },
        "tables": {
            "per_dataset_k2": matrix_by_variant["2"],
            "per_dataset_k4": matrix_by_variant["4"],
            "per_dataset_detailed": detailed,
            "aubc": aubc,
            "paired_statistics": paired,
            "win_tie_loss": wtl,
            "aggregate_supplement": aggregate,
            "cost_supplement": costs,
        },
        "charts": {
            "budget_curve_macro": {
                "family": "line",
                "x": "budget_share",
                "y": "f1",
                "series": "group_size_variant",
                "path": str(macro_svg_path.relative_to(root)),
                "y_domain": [0, 1],
            },
            "budget_curve_micro": {
                "family": "line",
                "x": "budget_share",
                "y": "f1",
                "series": "group_size_variant",
                "path": str(micro_svg_path.relative_to(root)),
                "y_domain": [0, 1],
            },
        },
        "audits": {
            "record": record_audit,
            "formal": formal_audit,
            "calibration_reuse": reuse,
            "gate_reuse": gate_reuse,
            "response_reuse": response_reuse,
            "dry_plan": dry_plan,
        },
        "sources": _source_rows(root, input_paths),
    }
    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = _read_json(artifact_path)
    if (
        persisted.get("schema_version")
        != "bgr-router-v3-budget-sweep-report-v1"
        or len(persisted.get("tables", {}).get("per_dataset_k2", [])) != 9
        or len(persisted.get("tables", {}).get("per_dataset_k4", [])) != 9
        or len(persisted.get("tables", {}).get("per_dataset_detailed", []))
        != 108
        or not all(
            path.is_file()
            for path in (
                markdown_path,
                html_path,
                macro_svg_path,
                micro_svg_path,
            )
        )
    ):
        raise ValueError("Router-v3 budget-sweep report artifact validation failed")
    return {
        "ok": True,
        "run_dir": root,
        "artifact": artifact_path,
        "markdown": markdown_path,
        "html": html_path,
        "charts": [macro_svg_path, micro_svg_path],
        "run_validation": validation,
        "verification": {
            "dataset_rows_per_k": 9,
            "detailed_rows": len(detailed),
            "paired_rows": len(paired),
            "selection_rows": len(selection_rows),
            "model_folds": validation["split_rows"],
        },
    }


def _build_catboost_report(
    root: Path,
    validation: Mapping[str, object],
    output_path: str | Path | None,
) -> dict[str, Any]:
    """Build the frozen 20% CatBoost dataset-first report."""

    metrics = root / "metrics"
    comparison_path = metrics / "per_dataset_router_comparison.csv"
    comparison_provenance_path = root / "provenance" / "comparison_reuse.json"
    comparison_enabled = (
        comparison_path.is_file() and comparison_provenance_path.is_file()
    )
    common_paths = (
        metrics / "per_dataset_f1_matrix.csv",
        metrics / "per_dataset_method_comparison.csv",
        metrics / "method_metrics.csv",
        metrics / "paired_statistics.csv",
        metrics / "api_cost_audit.csv",
        metrics / "selection_audit.csv",
        metrics / "record_audit.json",
        metrics / "formal_run_audit.json",
        _calibration_provenance_path(root),
        root / "provenance" / "response_reuse.json",
        root / "llm" / "router_v3_catboost_dry_plan.json",
        root / "run_manifest.json",
    )
    input_paths = (
        *common_paths,
        *((comparison_path, comparison_provenance_path) if comparison_enabled else ()),
    )
    f1_rows = _read_csv(common_paths[0])
    detailed_rows = _read_csv(common_paths[1])
    metric_rows = _read_csv(common_paths[2])
    paired_rows = _read_csv(common_paths[3])
    cost_rows = _read_csv(common_paths[4])
    selection_rows = _read_csv(common_paths[5])
    record_audit = _read_json(common_paths[6])
    formal_audit = _read_json(common_paths[7])
    reuse = _read_json(common_paths[8])
    response_reuse = _read_json(common_paths[9])
    dry_plan = _read_json(common_paths[10])
    manifest = _read_json(common_paths[11])
    comparison_rows = _read_csv(comparison_path) if comparison_enabled else []
    comparison_reuse = (
        _read_json(comparison_provenance_path)
        if comparison_enabled
        else {"enabled": False, "comparison_records": 0}
    )
    if (
        len(f1_rows) != 9
        or len(detailed_rows) != 63
        or len(comparison_rows) != (90 if comparison_enabled else 0)
        or len(metric_rows) != 77
        or len(paired_rows) != (180 if comparison_enabled else 90)
        or len(selection_rows) != 45
        or record_audit.get("records") != 155_386
        or formal_audit.get("ok") is not True
    ):
        raise ValueError("Router-v3 CatBoost report inputs failed acceptance counts")

    matrix: list[dict[str, object]] = []
    for row in f1_rows:
        item: dict[str, object] = {
            "dataset": _dataset_name(row),
            "baran": _number(row["baran_only_f1"]),
            "llm": _number(row["llm_only_f1"]),
            "llm_cells": _integer(row["llm_only_valid_llm_cells"]),
        }
        for variant in VARIANTS:
            item[f"catboost_{variant}_f1"] = _number(
                row[f"bgr_catboost_k{variant}_f1"]
            )
            item[f"catboost_{variant}_cells"] = _integer(
                row[f"bgr_catboost_k{variant}_llm_cells"]
            )
        matrix.append(item)

    detailed = _prepare_detailed(detailed_rows)
    aggregate = _prepare_aggregate(metric_rows)
    wtl = _win_tie_loss(
        detailed_rows, backends=("catboost",), variants=VARIANTS
    )
    paired = _paired_summary(paired_rows)
    costs = _cost_summary(cost_rows)
    comparison = [
        {
            "dataset": _dataset_name(row),
            "k": str(row["group_size_variant"]),
            "comparison": str(row["comparison_backend"]),
            "catboost_f1": _number(row["catboost_f1"]),
            "comparison_f1": _number(row["comparison_f1"]),
            "delta_f1": _number(row["delta_f1"]),
            "catboost_cells": _integer(row["catboost_llm_upgraded_cells"]),
            "comparison_cells": _integer(row["comparison_llm_upgraded_cells"]),
        }
        for row in comparison_rows
    ]

    matrix_columns = [
        ("dataset", "Dataset"),
        ("baran", "Baran F1"),
        ("llm", "LLM-only F1"),
        ("llm_cells", "LLM-only valid cells"),
        *[
            column
            for variant in VARIANTS
            for column in (
                (f"catboost_{variant}_f1", f"CatBoost k={variant} F1"),
                (
                    f"catboost_{variant}_cells",
                    f"CatBoost k={variant} LLM cells",
                ),
            )
        ],
    ]
    detailed_columns = (
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("correct", "Correct"),
        ("predicted", "Predicted"),
        ("precision", "Precision"),
        ("recall", "Recall/CA"),
        ("f1", "F1"),
        ("delta_baran", "ΔF1 vs Baran"),
        ("delta_llm", "ΔF1 vs LLM"),
        ("llm_cells", "LLM upgraded cells"),
        ("logical_calls", "Logical calls"),
        ("estimated_tokens", "Estimated tokens"),
        ("physical_calls", "Physical calls"),
        ("observed_tokens", "Observed tokens"),
    )
    comparison_columns = (
        ("dataset", "Dataset"),
        ("k", "k"),
        ("comparison", "Comparator"),
        ("catboost_f1", "CatBoost F1"),
        ("comparison_f1", "Comparator F1"),
        ("delta_f1", "ΔF1"),
        ("catboost_cells", "CatBoost LLM cells"),
        ("comparison_cells", "Comparator LLM cells"),
    )
    paired_columns = (
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("baseline", "Comparator"),
        ("delta_f1", "ΔF1"),
        ("ci", "95% row-cluster CI"),
        ("holm_p", "Holm p"),
    )
    wtl_columns = (
        ("method", "Method"),
        ("baseline", "Baseline"),
        ("win", "Win"),
        ("tie", "Tie"),
        ("loss", "Loss"),
    )
    aggregate_columns = (
        ("scope", "Scope"),
        ("method", "Method"),
        ("correct", "Correct"),
        ("predicted", "Predicted"),
        ("precision", "Precision"),
        ("recall", "Recall/CA"),
        ("f1", "F1"),
    )
    cost_columns = (
        ("phase", "Phase"),
        ("records", "Logical records"),
        ("physical_requests", "Physical requests"),
        ("cache_hits", "Cache hits"),
        ("provider_tokens", "Provider tokens"),
        ("failed", "Failed"),
    )
    title = "Router-v3：CatBoost 20% 预算全 k 实验"
    generated_at = str(
        manifest.get("completed_at")
        or manifest.get("updated_at")
        or datetime.now(timezone.utc).isoformat()
    )
    markdown = "\n\n".join(
        (
            f"# {title}",
            (
                f"Run: `{root.name}`。覆盖 9 个数据集、22,198 个 error cells；"
                "CatBoost 使用原生类别特征，BGR 预算固定为全量 singleton estimated-token 成本的 20%。"
            ),
            "## 逐数据集主表\n\n" + _markdown_table(matrix, matrix_columns),
            "## 逐数据集详细指标\n\n"
            + _markdown_table(detailed, detailed_columns),
            (
                "## 与 Router-v3 LightGBM / XGBoost 对齐比较\n\n"
                + _markdown_table(comparison, comparison_columns)
                if comparison_enabled
                else "## 外部 Router 比较\n\n本次运行未绑定比较 run；CatBoost 指标独立有效。"
            ),
            "## Dirty-row cluster paired bootstrap\n\n"
            "2,000 次重采样，seed=45；按 comparator × k 在九个数据集内做 Holm 校正。\n\n"
            + _markdown_table(paired, paired_columns),
            "## Win / Tie / Loss\n\n" + _markdown_table(wtl, wtl_columns),
            "## Micro 与 Dataset-Macro（补充）\n\n"
            + _markdown_table(aggregate, aggregate_columns),
            "## 成本审计（补充）\n\n" + _markdown_table(costs, cost_columns),
            (
                "## 完整性与复用\n\n"
                f"- Cell ledger: {record_audit['records']:,} records，audit={record_audit['ok']}。\n"
                f"- Selection: {len(selection_rows)} slices，model folds={validation['split_rows']}。\n"
                f"- Calibration ledger: {reuse.get('calibration_queries')} queries / "
                f"{reuse.get('calibration_pair_labels')} pair labels。\n"
                f"- Response reuse: success={response_reuse.get('imported_success_rows', 0)}，"
                f"terminal failure={response_reuse.get('imported_terminal_failure_rows', 0)}。\n"
                f"- External comparison enabled: {comparison_enabled}；rows="
                f"{comparison_reuse.get('comparison_records', 0)}；未复制进 CatBoost cell ledger。"
            ),
            f"生成时间：{generated_at}",
        )
    ) + "\n"

    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "report.md"
    html_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else report_dir / "report.html"
    )
    artifact_path = report_dir / "artifact.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_sections = "".join(
        (
            f"<h1>{html.escape(title)}</h1>",
            f"<p class=lede>Run: <code>{html.escape(root.name)}</code> · 9 datasets · 22,198 error cells · budget 20%</p>",
            "<h2>逐数据集主表</h2>",
            _html_table(matrix, matrix_columns),
            "<h2>逐数据集详细指标</h2>",
            _html_table(detailed, detailed_columns),
            (
                "<h2>与 Router-v3 LightGBM / XGBoost 对齐比较</h2>"
                + _html_table(comparison, comparison_columns)
                if comparison_enabled
                else "<h2>外部 Router 比较</h2><p>本次运行未绑定比较 run；CatBoost 指标独立有效。</p>"
            ),
            "<h2>Dirty-row cluster paired bootstrap</h2>",
            _html_table(paired, paired_columns),
            "<h2>Win / Tie / Loss</h2>",
            _html_table(wtl, wtl_columns),
            "<h2>Micro 与 Dataset-Macro（补充）</h2>",
            _html_table(aggregate, aggregate_columns),
            "<h2>成本审计（补充）</h2>",
            _html_table(costs, cost_columns),
            f"<p class=foot>Generated {html.escape(generated_at)}</p>",
        )
    )
    html_document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--ink:#172033;--muted:#5d6678;--line:#dce1ea;--paper:#fff;--accent:#7a3ea1}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6fa;color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1600px;margin:24px auto;padding:36px;background:var(--paper);box-shadow:0 8px 30px #23304a18}}
h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:34px;border-bottom:2px solid var(--accent);padding-bottom:7px}}.lede,.foot{{color:var(--muted)}}
.table-wrap{{overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{position:sticky;top:0;background:#f2ebf7;color:#4b2762}}tbody tr:nth-child(even){{background:#fafbfe}}code{{background:#edf0f5;padding:2px 5px;border-radius:4px}}
</style></head><body><main>{html_sections}</main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_document, encoding="utf-8")

    artifact = {
        "schema_version": "bgr-router-v3-catboost-report-v1",
        "surface": "report",
        "title": title,
        "run_id": root.name,
        "generated_at": generated_at,
        "validation": validation,
        "primary_narrative": "per_dataset_f1_and_llm_upgrades",
        "dimensions": {
            "datasets": 9,
            "error_cells": 22_198,
            "method_slices": 7,
            "cell_records": 155_386,
            "selection_slices": 45,
            "budget_share": 0.2,
            "variants": list(VARIANTS),
            "backends": ["catboost"],
        },
        "tables": {
            "per_dataset_f1": matrix,
            "per_dataset_detailed": detailed,
            "router_comparison": comparison,
            "paired_statistics": paired,
            "win_tie_loss": wtl,
            "aggregate_supplement": aggregate,
            "cost_supplement": costs,
        },
        "audits": {
            "record": record_audit,
            "formal": formal_audit,
            "reuse": reuse,
            "response_reuse": response_reuse,
            "comparison_reuse": comparison_reuse,
            "dry_plan": dry_plan,
        },
        "sources": _source_rows(root, input_paths),
    }
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = _read_json(artifact_path)
    persisted_detailed = persisted.get("tables", {}).get(
        "per_dataset_detailed", []
    )
    source_physical_calls = sum(
        int(float(row.get("physical_calls_charged", 0) or 0))
        for row in detailed_rows
    )
    expected_physical_calls = int(dry_plan.get("online_physical_queries", -1))
    reported_physical_calls = sum(
        int(str(row.get("physical_calls", "0")).replace(",", ""))
        for row in persisted_detailed
    )
    if (
        persisted.get("schema_version") != "bgr-router-v3-catboost-report-v1"
        or len(persisted.get("tables", {}).get("per_dataset_f1", [])) != 9
        or len(persisted_detailed) != 63
        or not all("physical_calls" in row for row in persisted_detailed)
        or source_physical_calls != expected_physical_calls
        or reported_physical_calls != source_physical_calls
        or len(persisted.get("tables", {}).get("router_comparison", []))
        != (90 if comparison_enabled else 0)
        or not markdown_path.is_file()
        or not html_path.is_file()
        or "Physical calls" not in markdown_path.read_text(encoding="utf-8")
        or "Physical calls" not in html_path.read_text(encoding="utf-8")
    ):
        raise ValueError("Router-v3 CatBoost report artifact validation failed")
    return {
        "ok": True,
        "run_dir": root,
        "artifact": artifact_path,
        "markdown": markdown_path,
        "html": html_path,
        "run_validation": validation,
        "verification": {
            "dataset_rows": len(matrix),
            "detailed_rows": len(detailed),
            "comparison_rows": len(comparison),
            "paired_rows": len(paired),
            "selection_rows": len(selection_rows),
        },
    }


def build_router_v3_report(
    run_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a completed run and write the three Router-v3 report formats."""

    from .router_v3 import (
        ROUTER_V3_BUDGET_SWEEP_REVISION,
        ROUTER_V3_CATBOOST_REVISION,
        ROUTER_V3_REVISION,
        validate_run,
    )

    root = Path(run_dir).expanduser().resolve()
    validation = validate_run(root, require_complete=True)
    if validation.get("router_revision") == ROUTER_V3_BUDGET_SWEEP_REVISION:
        return _build_budget_sweep_report(root, validation, output_path)
    if validation.get("router_revision") == ROUTER_V3_CATBOOST_REVISION:
        return _build_catboost_report(root, validation, output_path)
    if validation.get("router_revision") != ROUTER_V3_REVISION:
        raise ValueError("Router-v3 report received a non-v3 run")
    metrics = root / "metrics"
    input_paths = (
        metrics / "per_dataset_f1_matrix.csv",
        metrics / "per_dataset_method_comparison.csv",
        metrics / "method_metrics.csv",
        metrics / "paired_statistics.csv",
        metrics / "api_cost_audit.csv",
        metrics / "selection_audit.csv",
        metrics / "record_audit.json",
        metrics / "formal_run_audit.json",
        _calibration_provenance_path(root),
        root / "run_manifest.json",
    )
    f1_rows = _read_csv(input_paths[0])
    detailed_rows = _read_csv(input_paths[1])
    metric_rows = _read_csv(input_paths[2])
    paired_rows = _read_csv(input_paths[3])
    cost_rows = _read_csv(input_paths[4])
    selection_rows = _read_csv(input_paths[5])
    record_audit = _read_json(input_paths[6])
    formal_audit = _read_json(input_paths[7])
    reuse = _read_json(input_paths[8])
    manifest = _read_json(input_paths[9])
    if (
        len(f1_rows) != 9
        or len(detailed_rows) != 108
        or len(metric_rows) != 132
        or len(paired_rows) != 180
        or len(selection_rows) != 90
        or record_audit.get("records") != 266_376
        or formal_audit.get("ok") is not True
    ):
        raise ValueError("Router-v3 report artifact inputs failed acceptance counts")

    matrix = _prepare_f1_matrix(f1_rows)
    detailed = _prepare_detailed(detailed_rows)
    aggregate = _prepare_aggregate(metric_rows)
    wtl = _win_tie_loss(detailed_rows)
    paired = _paired_summary(paired_rows)
    costs = _cost_summary(cost_rows)
    matrix_columns = [
        ("dataset", "Dataset"),
        ("baran", "Baran-only"),
        ("llm", "LLM-only"),
        *[
            (f"{backend}_{variant}", f"{backend} k={variant}")
            for backend in BACKENDS
            for variant in VARIANTS
        ],
    ]
    detailed_columns = (
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("correct", "Correct"),
        ("predicted", "Predicted"),
        ("precision", "Precision"),
        ("recall", "Recall/CA"),
        ("f1", "F1"),
        ("delta_baran", "ΔF1 vs Baran"),
        ("delta_llm", "ΔF1 vs LLM"),
        ("logical_calls", "Logical calls"),
        ("estimated_tokens", "Estimated tokens"),
        ("physical_calls", "Physical calls"),
        ("observed_tokens", "Observed tokens"),
    )
    aggregate_columns = (
        ("scope", "Scope"),
        ("method", "Method"),
        ("correct", "Correct"),
        ("predicted", "Predicted"),
        ("precision", "Precision"),
        ("recall", "Recall/CA"),
        ("f1", "F1"),
    )
    wtl_columns = (
        ("method", "Method"),
        ("baseline", "Baseline"),
        ("win", "Win"),
        ("tie", "Tie"),
        ("loss", "Loss"),
    )
    paired_columns = (
        ("dataset", "Dataset"),
        ("method", "Method"),
        ("baseline", "Baseline"),
        ("delta_f1", "ΔF1"),
        ("ci", "95% row-cluster CI"),
        ("holm_p", "Holm p"),
    )
    cost_columns = (
        ("phase", "Phase"),
        ("records", "Logical records"),
        ("physical_requests", "Physical requests"),
        ("cache_hits", "Cache hits"),
        ("provider_tokens", "Provider tokens"),
        ("failed", "Failed"),
    )
    title = "Router-v3：按 Group Size 独立训练的 20% 预算实验"
    generated_at = str(
        manifest.get("completed_at")
        or manifest.get("updated_at")
        or datetime.now(timezone.utc).isoformat()
    )
    markdown = "\n\n".join(
        (
            f"# {title}",
            (
                f"Run: `{root.name}`。正式测试覆盖 9 个数据集、22,198 个 error cells；"
                "Baran 标注预算为 20，BGR 逻辑预算为全量 singleton estimated-token 成本的 20%。"
                "主表以逐数据集 F1 为中心；micro/macro 仅放在后部。"
            ),
            "## 逐数据集 F1 主表\n\n" + _markdown_table(matrix, matrix_columns),
            "## 逐数据集详细指标\n\n" + _markdown_table(detailed, detailed_columns),
            "## Dirty-row cluster paired bootstrap\n\n"
            "2,000 次重采样，seed=45；逐 backend × k × baseline 在 9 个数据集内做 Holm 校正。\n\n"
            + _markdown_table(paired, paired_columns),
            "## Win / Tie / Loss\n\n" + _markdown_table(wtl, wtl_columns),
            "## Micro 与 Dataset-Macro（补充）\n\n"
            + _markdown_table(aggregate, aggregate_columns),
            "## 成本审计（补充）\n\n" + _markdown_table(costs, cost_columns),
            (
                "## 完整性与复用\n\n"
                f"- Cell ledger: {record_audit['records']:,} records / "
                f"{record_audit['slices']} dataset slices，audit={record_audit['ok']}。\n"
                f"- Selection: {len(selection_rows)} slices，formal audit={formal_audit['ok']}。\n"
                f"- Calibration ledger: {reuse.get('calibration_queries')} queries / "
                f"{reuse.get('calibration_pair_labels')} pair labels；逻辑成本保持不变。"
            ),
            f"生成时间：{generated_at}",
        )
    ) + "\n"

    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = report_dir / "report.md"
    html_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else report_dir / "report.html"
    )
    artifact_path = report_dir / "artifact.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_sections = "".join(
        (
            f"<h1>{html.escape(title)}</h1>",
            f"<p class=lede>Run: <code>{html.escape(root.name)}</code> · 9 datasets · 22,198 error cells · budget 20%</p>",
            "<h2>逐数据集 F1 主表</h2>",
            _html_table(matrix, matrix_columns),
            "<h2>逐数据集详细指标</h2>",
            _html_table(detailed, detailed_columns),
            "<h2>Dirty-row cluster paired bootstrap</h2>",
            "<p>2,000 replicates, seed 45; Holm correction within backend × k × baseline.</p>",
            _html_table(paired, paired_columns),
            "<h2>Win / Tie / Loss</h2>",
            _html_table(wtl, wtl_columns),
            "<h2>Micro 与 Dataset-Macro（补充）</h2>",
            _html_table(aggregate, aggregate_columns),
            "<h2>成本审计（补充）</h2>",
            _html_table(costs, cost_columns),
            f"<p class=foot>Generated {html.escape(generated_at)}</p>",
        )
    )
    html_document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{--ink:#172033;--muted:#5d6678;--line:#dce1ea;--paper:#fff;--accent:#2656a8}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6fa;color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1500px;margin:24px auto;padding:36px;background:var(--paper);box-shadow:0 8px 30px #23304a18}}
h1{{font-size:30px;margin:0 0 8px}}h2{{margin-top:34px;border-bottom:2px solid var(--accent);padding-bottom:7px}}.lede,.foot{{color:var(--muted)}}
.table-wrap{{overflow:auto;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid var(--line);padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}th{{position:sticky;top:0;background:#edf2fb;color:#243b68}}tbody tr:nth-child(even){{background:#fafbfe}}code{{background:#edf0f5;padding:2px 5px;border-radius:4px}}
</style></head><body><main>{html_sections}</main></body></html>"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_document, encoding="utf-8")

    artifact = {
        "schema_version": "bgr-router-v3-report-v1",
        "surface": "report",
        "title": title,
        "run_id": root.name,
        "generated_at": generated_at,
        "validation": validation,
        "primary_narrative": "per_dataset_f1",
        "dimensions": {
            "datasets": 9,
            "error_cells": 22_198,
            "method_slices": 12,
            "cell_records": 266_376,
            "selection_slices": 90,
            "budget_share": 0.2,
            "variants": list(VARIANTS),
            "backends": list(BACKENDS),
        },
        "tables": {
            "per_dataset_f1": matrix,
            "per_dataset_detailed": detailed,
            "paired_statistics": paired,
            "win_tie_loss": wtl,
            "aggregate_supplement": aggregate,
            "cost_supplement": costs,
        },
        "audits": {
            "record": record_audit,
            "formal": formal_audit,
            "reuse": reuse,
        },
        "sources": _source_rows(root, input_paths),
    }
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = _read_json(artifact_path)
    if (
        persisted.get("schema_version") != "bgr-router-v3-report-v1"
        or len(persisted.get("tables", {}).get("per_dataset_f1", [])) != 9
        or len(persisted.get("tables", {}).get("per_dataset_detailed", [])) != 108
        or not markdown_path.is_file()
        or not html_path.is_file()
    ):
        raise ValueError("Router-v3 report artifact validation failed")
    return {
        "ok": True,
        "run_dir": root,
        "artifact": artifact_path,
        "markdown": markdown_path,
        "html": html_path,
        "run_validation": validation,
        "verification": {
            "dataset_rows": len(matrix),
            "detailed_rows": len(detailed),
            "paired_rows": len(paired),
            "selection_rows": len(selection_rows),
        },
    }


__all__ = ["build_router_v3_report"]
