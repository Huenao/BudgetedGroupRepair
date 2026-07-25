"""Canonical artifact and portable HTML report for experiments one and two."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_state import read_json


class ReportInputError(ValueError):
    pass


class PortableReportError(RuntimeError):
    pass


def _number(value: str) -> Any:
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    return int(number) if number.is_integer() else number


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {str(key): _number(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _required(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ReportInputError(f"completed-run report input is missing: {relative}")
    return path


def _source(identifier: str, label: str, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = {"id": identifier, "label": label, "path": path}
    if path.endswith(".csv"):
        sql = f"SELECT * FROM read_csv_auto('{path}', header = true);"
    else:
        sql = f"SELECT * FROM read_json_auto('{path}');"
    canonical = {
        **manifest,
        "query": {
            "description": label,
            "engine": "duckdb",
            "language": "sql",
            "sql": sql,
            "tables_used": [path],
        },
    }
    return manifest, canonical


def build_report_artifact(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    run_manifest = read_json(_required(root, "run_manifest.json"))
    experiment_config = run_manifest.get("experiment_config")
    group_size = int(
        experiment_config.get("primary_group_size", 4)
        if isinstance(experiment_config, Mapping)
        else 4
    )
    exp1_path = _required(root, "metrics/experiment1_by_dataset.csv")
    exp2_path = _required(root, "metrics/experiment2_by_dataset.csv")
    exp1_summary = read_json(_required(root, "metrics/experiment1_summary.json"))
    exp2_summary = read_json(_required(root, "metrics/experiment2_summary.json"))
    partition_audit = read_json(
        _required(root, "manifests/partition_matching_audit.json")
    )
    audit = read_json(_required(root, "metrics/record_audit.json"))
    if audit.get("ok") is not True:
        raise ReportInputError("record audit has not passed")
    exp1 = _read_csv(exp1_path)
    exp2 = _read_csv(exp2_path)
    if len(exp1) != 9:
        raise ReportInputError("experiment one requires all nine per-dataset results")
    coverage = [
        {
            "dataset": str(dataset_key).split("/", 1)[-1],
            "covered_cells": int(dataset_audit["structured"]["covered_cell_count"]),
            "structured_groups": int(dataset_audit["structured"]["selected_group_count"]),
            "positive_coverage": int(dataset_audit["structured"]["covered_cell_count"])
            > 0,
        }
        for dataset_key, dataset_audit in sorted(partition_audit.items())
    ]
    positive_coverage = {
        str(row["dataset"]) for row in coverage if row["positive_coverage"]
    }
    if {str(row["dataset"]) for row in exp2} != positive_coverage:
        raise ReportInputError(
            "experiment-two results differ from the frozen positive-coverage partition"
        )
    cost_path = _required(root, "metrics/api_cost_audit.csv")
    cost = _read_csv(cost_path)
    exp1_long = [
        {
            "dataset": row["dataset"],
            "method": method,
            "accuracy": row[field],
        }
        for row in exp1
        for method, field in (
            ("Baran", "baran_accuracy"),
            ("No-Baran singleton", "singleton_accuracy"),
            ("Oracle UB", "oracle_upper_bound"),
        )
    ]
    exp2_delta = [
        {
            "dataset": row["dataset"],
            "comparison": comparison,
            "delta_accuracy": row[field],
        }
        for row in exp2
        for comparison, field in (
            ("Structured − singleton", "delta_accuracy"),
            ("Structured − random", "structured_minus_random"),
        )
    ]
    headline = {
        "row": "macro",
        "oracle_gain_vs_best": exp1_summary["macro"]["upper_bound_minus_best"],
        "structured_delta_accuracy": exp2_summary["macro"]["delta_accuracy"],
        "structured_token_saving": exp2_summary["macro"]["token_per_cell_saving"],
        "grouping_decision": exp2_summary["decision"],
        "covered_datasets": len(positive_coverage),
        "covered_cells": sum(int(row["covered_cells"]) for row in coverage),
    }
    source_pairs = [
        _source("experiment1", "实验一逐数据集指标", "metrics/experiment1_by_dataset.csv"),
        _source("experiment2", "实验二逐数据集指标", "metrics/experiment2_by_dataset.csv"),
        _source("cost", "唯一 query API 成本审计", "metrics/api_cost_audit.csv"),
        _source("record_audit", "完成与泄漏审计", "metrics/record_audit.json"),
        _source("partition", "冻结分组覆盖与随机匹配审计", "manifests/partition_matching_audit.json"),
    ]
    sources = [pair[0] for pair in source_pairs]
    canonical_sources = [pair[1] for pair in source_pairs]
    timestamp = datetime.now(timezone.utc).isoformat()
    title = f"No-Baran Prompt v1 k={group_size} 前置实验报告（{root.name}）"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Baran/No-Baran singleton 互补性与结构化分组有效性的预注册验证。",
            "generatedAt": timestamp,
            "cards": [
                {
                    "id": "oracle_gain",
                    "description": "实验一 macro Oracle UB 相对最佳单方法的绝对增益。",
                    "dataset": "headline",
                    "sourceId": "experiment1",
                    "filter": {"row": "macro"},
                    "metrics": [{"label": "Oracle gain", "field": "oracle_gain_vs_best", "format": "percent", "signed": True}],
                },
                {
                    "id": "group_delta",
                    "description": "实验二跨数据集等权 macro structured−singleton accuracy。",
                    "dataset": "headline",
                    "sourceId": "experiment2",
                    "filter": {"row": "macro"},
                    "metrics": [{"label": "Structured ΔAcc", "field": "structured_delta_accuracy", "format": "percent", "signed": True}],
                },
                {
                    "id": "token_saving",
                    "description": "实验二 structured 相对 singleton 的 macro provider token 节省率。",
                    "dataset": "headline",
                    "sourceId": "cost",
                    "filter": {"row": "macro"},
                    "metrics": [{"label": "Token saving", "field": "structured_token_saving", "format": "percent"}],
                },
            ],
            "charts": [
                {
                    "id": "experiment1_accuracy",
                    "title": "逐数据集：Baran、No-Baran singleton 与 Oracle UB",
                    "subtitle": "accuracy 分母包含 abstain、missing、invalid 与 provider failure。",
                    "type": "horizontalBar",
                    "intent": "comparison",
                    "question": "两个独立方法是否存在可利用的互补空间？",
                    "rationale": "分组横条便于比较九个数据集上的三种准确率。",
                    "dataset": "experiment1_long",
                    "sourceId": "experiment1",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "dataset", "type": "nominal", "label": "数据集"},
                        "y": {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"},
                        "color": {"field": "method", "type": "nominal", "label": "方法"},
                    },
                    "legend": {"position": "bottom", "sort": "spec", "title": "方法"},
                    "layout": "full",
                },
                {
                    "id": "experiment2_delta",
                    "title": "逐数据集：结构化 group 的 ΔAccuracy",
                    "subtitle": "同时对比 canonical singleton 与 matched random batching。",
                    "type": "horizontalBar",
                    "intent": "comparison",
                    "question": "增益来自结构化分组还是普通 batching？",
                    "rationale": "以零为中心的差值图直接展示帮助与 batch interference。",
                    "dataset": "experiment2_delta",
                    "sourceId": "experiment2",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "dataset", "type": "nominal", "label": "数据集"},
                        "y": {"field": "delta_accuracy", "type": "quantitative", "label": "ΔAccuracy", "format": "percent"},
                        "color": {"field": "comparison", "type": "nominal", "label": "比较"},
                    },
                    "legend": {"position": "bottom", "sort": "spec", "title": "比较"},
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "experiment1_table",
                    "title": "实验一精确四格表与置信区间",
                    "dataset": "experiment1",
                    "sourceId": "experiment1",
                    "density": "dense",
                    "columns": [
                        {"field": "dataset", "label": "Dataset", "type": "text"},
                        {"field": "N", "label": "N", "format": "number"},
                        {"field": "baran_accuracy", "label": "Baran Acc", "format": "percent"},
                        {"field": "singleton_accuracy", "label": "Singleton Acc", "format": "percent"},
                        {"field": "n11", "label": "n11", "format": "number"},
                        {"field": "n10", "label": "n10", "format": "number"},
                        {"field": "n01", "label": "n01", "format": "number"},
                        {"field": "n00", "label": "n00", "format": "number"},
                        {"field": "oracle_upper_bound", "label": "Oracle UB", "format": "percent"},
                        {"field": "upper_bound_minus_best", "label": "UB−Best", "format": "percent", "movement": True},
                    ],
                    "layout": "full",
                },
                {
                    "id": "experiment2_table",
                    "title": "实验二分组质量、成本与预注册判定",
                    "dataset": "experiment2",
                    "sourceId": "experiment2",
                    "density": "dense",
                    "columns": [
                        {"field": "dataset", "label": "Dataset", "type": "text"},
                        {"field": "N_cells", "label": "Cells", "format": "number"},
                        {"field": "singleton_accuracy", "label": "Singleton", "format": "percent"},
                        {"field": "structured_accuracy", "label": "Structured", "format": "percent"},
                        {"field": "delta_accuracy", "label": "ΔAcc", "format": "percent", "movement": True},
                        {"field": "random_accuracy", "label": "Random", "format": "percent"},
                        {"field": "structured_minus_random", "label": "S−R", "format": "percent", "movement": True},
                        {"field": "token_per_cell_saving", "label": "Token saving", "format": "percent"},
                        {"field": "decision", "label": "判定", "type": "text"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "coverage_table",
                    "title": "实验二冻结分组覆盖（零覆盖数据集不进入 paired accuracy 分母）",
                    "dataset": "coverage",
                    "sourceId": "partition",
                    "density": "dense",
                    "columns": [
                        {"field": "dataset", "label": "Dataset", "type": "text"},
                        {"field": "covered_cells", "label": "Covered cells", "format": "number"},
                        {"field": "structured_groups", "label": f"k={group_size} groups", "format": "number"},
                        {"field": "positive_coverage", "label": "Paired metric eligible", "type": "boolean"},
                    ],
                    "layout": "full",
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {"id": "summary", "type": "markdown", "body": f"## 技术摘要\n\n结果严格使用冻结阈值；clean value 仅在响应账本冻结后绑定。成本按共享 append-only ledger 中的物理请求去重。实验二只在 {len(positive_coverage)}/9 个具有完整 k={group_size} group 的数据集、共 {sum(int(row['covered_cells']) for row in coverage)} 个 cell 上计算 paired accuracy。"},
                {"id": "headline", "type": "metric-strip", "cardIds": ["oracle_gain", "group_delta", "token_saving"]},
                {"id": "exp1_finding", "type": "markdown", "sourceId": "experiment1", "body": "## Singleton 与 Baran 的互补空间\n\n下图在每个数据集上并列比较两个独立方法与 Oracle UB；Oracle UB 只表示二者正确集合的并集上限，不是可部署系统的实际准确率。"},
                {"id": "exp1_chart", "type": "chart", "chartId": "experiment1_accuracy"},
                {"id": "exp1_table_block", "type": "table", "tableId": "experiment1_table"},
                {"id": "exp2_finding", "type": "markdown", "sourceId": "experiment2", "body": f"## k={group_size} 结构化分组的质量与成本\n\n差值图同时回答 structured group 是否优于相同 cell 的 singleton，以及其效果是否超过 matched random batching；所有 abstain、missing、invalid 和 provider failure 都保留在分母中。"},
                {"id": "exp2_chart", "type": "chart", "chartId": "experiment2_delta"},
                {"id": "exp2_table_block", "type": "table", "tableId": "experiment2_table"},
                {"id": "coverage_table_block", "type": "table", "tableId": "coverage_table"},
                {"id": "limits", "type": "markdown", "body": f"## 限制、稳健性与下一步\n\n首轮每数据集仅 300 个 error cells；实验二只有 {len(positive_coverage)}/9 个数据集形成完整 k={group_size} group，不能把 paired 结果推广到零覆盖数据集。matched random 中还存在无法在冻结分层内打散的组，因此 structured−random 只提供受限的机制归因。Phase 2.5 与 Phase 3 只有在三项门禁全部通过后才允许执行。"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": timestamp,
            "status": "ready",
            "datasets": {
                "headline": [headline],
                "experiment1": exp1,
                "experiment1_long": exp1_long,
                "experiment2": exp2,
                "experiment2_delta": exp2_delta,
                "coverage": coverage,
                "cost": cost,
            },
        },
        "sources": canonical_sources,
        "package_info": {
            "originUrl": f"artifact://budgeted-group-repair-no-baran/{root.name}",
            "controls": {"edit": False, "refresh": False},
        },
    }
    serialized = json.dumps(artifact, ensure_ascii=False).lower()
    for forbidden in ("api_key", "access_token", "private_key", "password"):
        if forbidden in serialized:
            raise AssertionError(f"sensitive field name entered artifact: {forbidden}")
    return artifact


def write_report_artifact(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    output = root / "report" / "artifact.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(build_report_artifact(root), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def _plugin_root() -> Path:
    cache = Path.home() / ".codex" / "plugins" / "cache" / "openai-curated-remote" / "data-analytics"
    for root in sorted(cache.glob("*"), reverse=True):
        if (root / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs").is_file():
            return root.resolve()
    raise PortableReportError("Data Analytics portable builder was not found")


def _node() -> Path:
    candidates: Sequence[Path] = (
        Path(shutil.which("node") or ""),
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime" / "dependencies" / "node" / "bin" / "node",
    )
    for candidate in candidates:
        if str(candidate) and candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise PortableReportError("Codex Node runtime was not found")


def _receipt(stdout: str) -> dict[str, Any]:
    for candidate in (stdout.strip(), *reversed(stdout.splitlines())):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {"ok": False, "error": "portable builder returned no JSON receipt"}


def build_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    deliver: bool = True,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    artifact = write_report_artifact(root)
    if not deliver:
        return {"artifact": str(artifact), "html": None, "validated": False}
    output = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else root / "report" / "report.html"
    )
    builder = _plugin_root() / "skills" / "build-report" / "scripts" / "deliver_portable_artifact.mjs"
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [str(_node()), str(builder), "--input", str(artifact), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    receipt = _receipt(completed.stdout)
    if completed.returncode != 0 or receipt.get("ok") is not True or not output.is_file():
        detail = receipt.get("error") or completed.stderr.strip() or "portable report validation failed"
        raise PortableReportError(str(detail))
    return {"artifact": str(artifact), "html": str(output), "validated": True, "receipt": receipt}


__all__ = ["PortableReportError", "ReportInputError", "build_report", "build_report_artifact"]
