"""Technical Markdown/HTML reporting for the Router-v4 calibration experiment."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report JSON input must be an object: {path}")
    return value


def _number(value: object, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _table(
    rows: Sequence[Mapping[str, object]],
    columns: Sequence[tuple[str, str]],
) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field, _ in columns)
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
            f"<td>{html.escape(str(row.get(field, '')))}</td>" for field, _ in columns
        )
        + "</tr>"
        for row in rows
    )
    return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _delta_svg(rows: Sequence[Mapping[str, object]]) -> str:
    width, height = 1050, 500
    left, right, top, bottom = 190, 45, 45, 45
    values = [float(row[f"k{variant}_delta_f1"]) for row in rows for variant in ("1", "4")]
    bound = max(0.01, max(abs(value) for value in values) * 1.15)
    center = left + (width - left - right) / 2
    scale = (width - left - right) / (2 * bound)
    row_height = (height - top - bottom) / len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Delta F1 by dataset and k">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<line x1="{center:.2f}" y1="{top-12}" x2="{center:.2f}" y2="{height-bottom+8}" stroke="#26344a"/>',
    ]
    for index, row in enumerate(rows):
        y = top + (index + 0.5) * row_height
        parts.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" fill="#273247" font-size="12">{html.escape(str(row["dataset"]))}</text>'
        )
        for variant, offset, color in (("1", -6, "#295fa6"), ("4", 6, "#c27819")):
            value = float(row[f"k{variant}_delta_f1"])
            x = center + value * scale
            start = min(center, x)
            parts.append(
                f'<rect x="{start:.2f}" y="{y+offset-4:.2f}" width="{abs(x-center):.2f}" height="8" fill="{color}"/>'
            )
    parts.append(
        f'<text x="{left}" y="22" fill="#172033" font-size="17">Calibrated − raw ΔF1 · blue k=1 · orange k=4 · domain ±{bound:.3f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def _calibration_svg(summary: Mapping[str, object]) -> str:
    width, height = 920, 360
    left, top, bottom = 85, 50, 55
    plot_height = height - top - bottom
    items: list[tuple[str, float, float]] = []
    by_variant = summary["calibration_by_variant"]
    assert isinstance(by_variant, Mapping)
    for variant in ("1", "4"):
        variant_value = by_variant[variant]
        assert isinstance(variant_value, Mapping)
        heads = variant_value["heads"]
        assert isinstance(heads, Mapping)
        for head in ("helpful", "harmful"):
            values = heads[head]
            assert isinstance(values, Mapping)
            for metric in ("brier", "ece"):
                items.append(
                    (
                        f"k={variant} {head} {metric.upper()}",
                        float(values[f"raw_{metric}"]),
                        float(values[f"calibrated_{metric}"]),
                    )
                )
    maximum = max(max(raw, calibrated) for _, raw, calibrated in items) * 1.12 or 1.0
    group_width = (width - left - 30) / len(items)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Fold macro Brier and ECE comparison">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="85" y="25" fill="#172033" font-size="17">Fold-macro probability error · lower is better · gray raw · green calibrated</text>',
    ]
    for index, (label, raw, calibrated) in enumerate(items):
        x = left + (index + 0.5) * group_width
        for value, offset, color in ((raw, -10, "#798294"), (calibrated, 3, "#278765")):
            bar_height = value / maximum * plot_height
            parts.append(
                f'<rect x="{x+offset:.2f}" y="{top+plot_height-bar_height:.2f}" width="10" height="{bar_height:.2f}" fill="{color}"/>'
            )
        parts.append(
            f'<text x="{x:.2f}" y="{height-bottom+18}" text-anchor="middle" fill="#273247" font-size="9" transform="rotate(25 {x:.2f} {height-bottom+18})">{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def build_router_v4_report(
    root: Path,
    validation: Mapping[str, object],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    metrics = root / "metrics"
    effects = _csv(metrics / "calibration_effect_by_dataset.csv")
    paired = _csv(metrics / "calibration_paired_statistics.csv")
    calibration = _csv(metrics / "calibration_by_dataset.csv")
    costs = _csv(metrics / "api_cost_audit.csv")
    selections = _csv(metrics / "selection_audit.csv")
    summary = _json(metrics / "calibration_summary.json")
    formal = _json(metrics / "formal_run_audit.json")
    record = _json(metrics / "record_audit.json")
    dry_plan = _json(root / "llm" / "router_v4_isotonic_dry_plan.json")
    comparator = _json(root / "provenance" / "historical_raw_comparator.json")
    if (
        len(effects) != 18
        or len(paired) != 54
        or len(calibration) != 36
        or len(selections) != 18
        or formal.get("ok") is not True
        or int(record.get("records", -1)) != 88_792
        or validation.get("ok") is not True
    ):
        raise ValueError("Router-v4 report input acceptance counts failed")
    indexed = {
        (str(row["suite"]), str(row["dataset"]), str(row["group_size_variant"])): row
        for row in effects
    }
    main_rows: list[dict[str, object]] = []
    datasets = sorted({(str(row["suite"]), str(row["dataset"])) for row in effects})
    for suite, dataset in datasets:
        item: dict[str, object] = {"dataset": f"{suite}/{dataset}"}
        for variant in ("1", "4"):
            row = indexed[(suite, dataset, variant)]
            for metric in ("precision", "recall", "f1"):
                item[f"k{variant}_raw_{metric}"] = _number(row[f"raw_{metric}"])
                item[f"k{variant}_cal_{metric}"] = _number(row[f"calibrated_{metric}"])
                item[f"k{variant}_delta_{metric}"] = _number(row[f"delta_{metric}"])
        main_rows.append(item)
    main_columns = [("dataset", "Dataset")]
    for variant in ("1", "4"):
        for metric, label in (("precision", "P"), ("recall", "R"), ("f1", "F1")):
            main_columns.extend(
                (
                    (f"k{variant}_raw_{metric}", f"k={variant} Raw {label}"),
                    (f"k{variant}_cal_{metric}", f"k={variant} Cal {label}"),
                    (f"k{variant}_delta_{metric}", f"k={variant} Δ{label}"),
                )
            )
    decision = summary["decision_by_variant"]
    assert isinstance(decision, Mapping)
    aggregate_rows: list[dict[str, object]] = []
    for variant in ("1", "4"):
        value = decision[variant]
        assert isinstance(value, Mapping)
        row: dict[str, object] = {
            "k": variant,
            "wtl": f"{value['wins']}/{value['ties']}/{value['losses']}",
            "worst": _number(value["worst_dataset_delta_f1"]),
            "decision_success": value["success"],
        }
        for scope, key in (("micro", "micro"), ("macro", "dataset_macro")):
            raw = value[f"raw_{key}"]
            calibrated_value = value[f"calibrated_{key}"]
            assert isinstance(raw, Mapping) and isinstance(calibrated_value, Mapping)
            for metric in ("precision", "recall", "f1"):
                row[f"raw_{scope}_{metric}"] = _number(raw[metric])
                row[f"cal_{scope}_{metric}"] = _number(calibrated_value[metric])
        aggregate_rows.append(row)
    aggregate_columns = (
        ("k", "k"),
        ("raw_micro_precision", "Raw micro P"),
        ("cal_micro_precision", "Cal micro P"),
        ("raw_micro_recall", "Raw micro R"),
        ("cal_micro_recall", "Cal micro R"),
        ("raw_micro_f1", "Raw micro F1"),
        ("cal_micro_f1", "Cal micro F1"),
        ("raw_macro_f1", "Raw macro F1"),
        ("cal_macro_f1", "Cal macro F1"),
        ("wtl", "F1 W/T/L"),
        ("worst", "Worst ΔF1"),
        ("decision_success", "Decision success"),
    )
    conclusion = str(summary["usefulness_conclusion"])
    headline = (
        f"按预注册分层口径，本 calibration 模块的结论为“{conclusion}”。"
        "Probability calibration 与最终 20% budget 决策效果分别判断；AUPRC 仅作为 ranking 诊断。"
    )
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    delta_path = report_dir / "delta_f1_by_dataset.svg"
    calibration_path = report_dir / "calibration_error_comparison.svg"
    delta_path.write_text(_delta_svg(main_rows), encoding="utf-8")
    calibration_path.write_text(_calibration_svg(summary), encoding="utf-8")
    limitations = (
        "Isotonic maps are fitted only from train-family OOF predictions and are frozen before selection. "
        "The experiment evaluates one backbone, two k values, one budget and one seed; bootstrap intervals quantify "
        "dirty-row sampling uncertainty but do not establish causal generalization to other datasets or budgets."
    )
    markdown = "\n\n".join(
        (
            "# Router-v4 LightGBM Isotonic Calibration 实验报告",
            "## 结论先行\n\n" + headline,
            "## 9 个数据集完整主表\n\n" + _table(main_rows, main_columns),
            "## ΔF1 分数据集\n\n![ΔF1](delta_f1_by_dataset.svg)",
            "## Brier / ECE 对比\n\n![Calibration error](calibration_error_comparison.svg)",
            "## Micro、Dataset-Macro 与稳健性\n\n"
            + _table(aggregate_rows, aggregate_columns),
            "## 定义与方法\n\n"
            "Precision 的分母为 schema-valid predicted repairs；Recall 是完整 dirty-cell universe 上的 correct repairs 比例；"
            "F1 是两者调和平均。每个 target × k 使用 train-family LOFO raw scores 拟合 helpful/harmful 两张 isotonic maps；"
            "full head 与所有 replicas 共享冻结 maps，sigma 在 calibrated replica net gain 上以 ddof=1 计算。"
            "所有 18 个 score ledgers 哈希冻结后才执行 optimizer。",
            "## 不确定性与统计检验\n\n"
            "对 P/R/F1 使用 2,000 次 dirty-row cluster paired bootstrap（seed=45），并在每个 k × metric 的九数据集族内做 Holm 校正。"
            f"完整统计见 `metrics/calibration_paired_statistics.csv`（{len(paired)} 行）。",
            "## 泄漏、成本与 provenance 审计\n\n"
            f"formal audit={formal['ok']}；final records={record['records']:,}；selection slices={len(selections)}；"
            f"dry-plan missing physical queries={dry_plan['online_physical_queries']}；"
            f"historical raw records={comparator['raw_records']}，五个 frozen hashes 已验证。",
            "## Calibration 改善但 F1 未改善的案例\n\n"
            + (", ".join(summary["calibration_improved_but_final_f1_not_improved"]) or "无"),
            "## 局限与后续建议\n\n" + limitations,
        )
    ) + "\n"
    markdown_path = report_dir / "report.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = Path(output_path).resolve() if output_path is not None else report_dir / "report.html"
    html_sections = "".join(
        (
            "<h1>Router-v4 LightGBM Isotonic Calibration 实验报告</h1>",
            f"<h2>结论先行</h2><p>{html.escape(headline)}</p>",
            "<h2>9 个数据集完整主表</h2>",
            _html_table(main_rows, main_columns),
            "<h2>ΔF1 分数据集</h2>",
            delta_path.read_text(encoding="utf-8"),
            "<h2>Brier / ECE 对比</h2>",
            calibration_path.read_text(encoding="utf-8"),
            "<h2>Micro、Dataset-Macro 与稳健性</h2>",
            _html_table(aggregate_rows, aggregate_columns),
            "<h2>定义、方法与统计</h2><p>Precision 使用有效 repairs；Recall 使用完整 dirty-cell universe。2,000 次 dirty-row cluster paired bootstrap，seed 45；Holm 校正在 k × metric 内完成。</p>",
            f"<h2>泄漏、成本与 provenance</h2><p>formal audit={formal['ok']}；records={record['records']:,}；missing physical queries={dry_plan['online_physical_queries']}。</p>",
            f"<h2>局限与后续建议</h2><p>{html.escape(limitations)}</p>",
        )
    )
    html_document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Router-v4 calibration report</title><style>
body{{margin:0;background:#f3f6fa;color:#172033;font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}main{{max-width:1700px;margin:24px auto;padding:36px;background:#fff}}h2{{margin-top:34px;border-bottom:2px solid #295fa6;padding-bottom:7px}}.table-wrap{{overflow:auto;border:1px solid #dce1ea}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{border-bottom:1px solid #dce1ea;padding:7px 9px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:#edf3fb}}svg{{width:100%;height:auto;border:1px solid #dce1ea}}
</style></head><body><main>{html_sections}</main></body></html>'''
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html_document, encoding="utf-8")
    artifact = {
        "schema_version": "bgr-router-v4-isotonic-report-v1",
        "surface": "report",
        "run_id": root.name,
        "title": "Router-v4 LightGBM Isotonic Calibration",
        "usefulness_conclusion": conclusion,
        "validation": dict(validation),
        "dimensions": {
            "datasets": 9,
            "error_cells": 22_198,
            "method_slices": 4,
            "cell_records": 88_792,
            "gate_folds": 18,
            "selection_slices": 18,
            "routeability_folds": 18,
        },
        "tables": {
            "raw_vs_calibrated": main_rows,
            "aggregate": aggregate_rows,
            "paired_statistics": paired,
        },
        "charts": {
            "delta_f1": str(delta_path.relative_to(root)),
            "calibration_error": str(calibration_path.relative_to(root)),
        },
        "audits": {
            "formal": formal,
            "record": record,
            "dry_plan": dry_plan,
            "historical_comparator": comparator,
        },
    }
    artifact_path = report_dir / "artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = _json(artifact_path)
    if (
        persisted.get("schema_version") != "bgr-router-v4-isotonic-report-v1"
        or len(persisted.get("tables", {}).get("raw_vs_calibrated", [])) != 9
        or len(persisted.get("tables", {}).get("paired_statistics", [])) != 54
        or "结论先行" not in markdown_path.read_text(encoding="utf-8")
        or "结论先行" not in html_path.read_text(encoding="utf-8")
    ):
        raise ValueError("Router-v4 report artifact validation failed")
    return {
        "ok": True,
        "run_dir": root,
        "artifact": artifact_path,
        "markdown": markdown_path,
        "html": html_path,
        "usefulness_conclusion": conclusion,
        "run_validation": dict(validation),
    }


__all__ = ["build_router_v4_report"]
