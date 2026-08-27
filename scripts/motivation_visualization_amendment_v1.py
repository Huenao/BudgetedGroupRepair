#!/usr/bin/env python3
"""Build the publication visualization amendment for the motivation evidence run.

This script is intentionally outside the frozen experiment code set.  It reads
only finalized, offline report artifacts; it never imports the evidence runner,
opens a network connection, or calls a model provider.  Canonical report files
are treated as immutable and are hash-checked before and after publication
outputs are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


AMENDMENT_SCHEMA_VERSION = "motivation-visualization-amendment-v1"
DEFAULT_RUN_ID = "motivation_evidence_deepseek_v4_flash_20260822_full"
VIEWS = ("pattern", "semantic")
GROUP_SIZES = (2, 4, 8)
COLORS = {"pattern": "#2c7fb8", "semantic": "#d95f0e"}
MARKERS = {2: "o", 4: "s", 8: "D"}
ARM_STYLE = {
    "structured": {"prefix": "structured", "short": "G", "filled": True},
    "random": {"prefix": "random", "short": "R", "filled": False},
}
FOCUSED_AXIS_LIMITS = (0.02, 0.10)
SAVING_LEGEND_VALUES = (0.18, 0.25, 0.31)

# Offsets are in display points and are deliberately frozen.  Faceting leaves
# six points per axis; these offsets keep labels clear without moving the data.
LABEL_OFFSETS = {
    ("pattern", "structured", 2): (-17, -15),
    ("pattern", "random", 2): (-22, 8),
    ("pattern", "structured", 4): (7, -15),
    ("pattern", "random", 4): (-24, 8),
    ("pattern", "structured", 8): (8, 2),
    ("pattern", "random", 8): (8, 5),
    ("semantic", "structured", 2): (7, -15),
    ("semantic", "random", 2): (-25, -5),
    ("semantic", "structured", 4): (7, -14),
    ("semantic", "random", 4): (-25, 8),
    ("semantic", "structured", 8): (8, 2),
    ("semantic", "random", 8): (-26, 6),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"empty amendment input: {path}")
    return rows


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _finite(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"missing or non-numeric {field!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field!r}")
    return value


def _load_inputs(run_dir: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    complementarity = _read_csv(run_dir / "metrics" / "complementarity_by_dataset.csv")
    group = _read_csv(run_dir / "metrics" / "group_by_dataset_view_size.csv")
    micro = [row for row in complementarity if row.get("scope") == "micro"]
    macro = [
        row
        for row in group
        if row.get("scope") == "macro" and row.get("source_view") in VIEWS
    ]
    if len(micro) != 1:
        raise ValueError(f"expected one complementarity micro row, observed {len(micro)}")
    identities = {(row["source_view"], int(row["group_size"])) for row in macro}
    expected = {(view, size) for view in VIEWS for size in GROUP_SIZES}
    if len(macro) != 6 or identities != expected:
        raise ValueError("publication Panel B requires the six frozen macro conditions")
    for row in macro:
        for arm in ARM_STYLE:
            prefix = ARM_STYLE[arm]["prefix"]
            for field in (
                f"{prefix}_rescue_rate",
                f"{prefix}_rescue_rate_ci_low",
                f"{prefix}_rescue_rate_ci_high",
                f"{prefix}_interference_rate",
                f"{prefix}_interference_rate_ci_low",
                f"{prefix}_interference_rate_ci_high",
                f"{arm}_token_saving",
            ):
                _finite(row, field)
    return micro[0], sorted(
        macro, key=lambda row: (VIEWS.index(row["source_view"]), int(row["group_size"]))
    )


def _bubble_area(saving: float) -> float:
    """Scatter area in points squared, proportional to absolute saving."""

    return 900.0 * abs(float(saving))


def _panel_a(axis: Any, micro: Mapping[str, str]) -> None:
    matrix = np.asarray(
        [
            [_finite(micro, "n11"), _finite(micro, "n10")],
            [_finite(micro, "n01"), _finite(micro, "n00")],
        ],
        dtype=float,
    )
    total = float(np.sum(matrix))
    if total <= 0.0:
        raise ValueError("empty complementarity population")
    axis.imshow(matrix / total, cmap="Blues", vmin=0.0, vmax=1.0)
    labels = (("n11", "n10"), ("n01", "n00"))
    for row_index in range(2):
        for column_index in range(2):
            count = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                f"{labels[row_index][column_index]}\n{count:,}\n({count / total:.1%})",
                ha="center",
                va="center",
                color="white" if count / total > 0.35 else "black",
                fontweight="semibold",
            )
    axis.set_xticks((0, 1), ("LLM correct", "LLM wrong"))
    axis.set_yticks((0, 1), ("Baran correct", "Baran wrong"))
    axis.set_title("A  Baran–singleton complementarity", loc="left", fontweight="bold")
    axis.set_xlabel(
        "Baran acc. "
        f"{_finite(micro, 'baran_accuracy'):.1%}  |  "
        "LLM acc. "
        f"{_finite(micro, 'llm_accuracy'):.1%}\n"
        "Offline opportunity upper bound "
        f"{_finite(micro, 'oracle_union_upper_bound'):.1%}"
    )


def _panel_b(
    axes: Mapping[str, Any],
    legend_axis: Any,
    macro_rows: Sequence[Mapping[str, str]],
) -> tuple[float, float]:
    savings: list[float] = []
    for view in VIEWS:
        axis = axes[view]
        color = COLORS[view]
        rows = [row for row in macro_rows if row["source_view"] == view]
        for row in rows:
            size = int(row["group_size"])
            for arm, style in ARM_STYLE.items():
                prefix = style["prefix"]
                x = _finite(row, f"{prefix}_rescue_rate")
                y = _finite(row, f"{prefix}_interference_rate")
                x_low = _finite(row, f"{prefix}_rescue_rate_ci_low")
                x_high = _finite(row, f"{prefix}_rescue_rate_ci_high")
                y_low = _finite(row, f"{prefix}_interference_rate_ci_low")
                y_high = _finite(row, f"{prefix}_interference_rate_ci_high")
                saving = _finite(row, f"{arm}_token_saving")
                savings.append(saving)
                axis.errorbar(
                    x,
                    y,
                    xerr=np.asarray([[max(0.0, x - x_low)], [max(0.0, x_high - x)]]),
                    yerr=np.asarray([[max(0.0, y - y_low)], [max(0.0, y_high - y)]]),
                    fmt="none",
                    ecolor=color,
                    alpha=0.55 if not style["filled"] else 0.8,
                    capsize=2,
                    linewidth=0.9,
                    zorder=1,
                )
                axis.scatter(
                    [x],
                    [y],
                    s=_bubble_area(saving),
                    marker=MARKERS[size],
                    facecolors=color if style["filled"] else "white",
                    edgecolors=color,
                    linewidths=1.5,
                    alpha=0.95,
                    zorder=3,
                )
                if saving < 0.0:
                    axis.scatter(
                        [x],
                        [y],
                        s=_bubble_area(saving) * 0.45,
                        marker="x",
                        color="0.15",
                        linewidths=1.2,
                        zorder=4,
                    )
                axis.annotate(
                    f"{style['short']}{size}",
                    (x, y),
                    xytext=LABEL_OFFSETS[(view, arm, size)],
                    textcoords="offset points",
                    fontsize=7.5,
                    fontweight="semibold",
                    color="0.12",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.82},
                    zorder=5,
                )
        low, high = FOCUSED_AXIS_LIMITS
        axis.plot((low, high), (low, high), linestyle="--", color="0.42", linewidth=1)
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks((0.02, 0.04, 0.06, 0.08, 0.10))
        axis.set_yticks((0.02, 0.04, 0.06, 0.08, 0.10))
        axis.set_xlabel("Rescue rate")
        axis.set_title(f"{view.capitalize()} view", fontsize=9.5, pad=5)
        axis.grid(alpha=0.18)
    axes["pattern"].set_ylabel("Interference rate")
    axes["semantic"].tick_params(labelleft=False)

    neutral = "#4d4d4d"
    arm_and_size_handles: list[Line2D] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor=neutral,
            markeredgecolor=neutral,
            markersize=7,
            label="G structured (filled)",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=neutral,
            markeredgewidth=1.4,
            markersize=7,
            label="R random (open)",
        ),
    ]
    arm_and_size_handles.extend(
        Line2D(
            [],
            [],
            marker=MARKERS[size],
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=neutral,
            markersize=7,
            label=f"k={size}",
        )
        for size in GROUP_SIZES
    )
    saving_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=neutral,
            markersize=math.sqrt(_bubble_area(value)),
            label=f"{value:.0%} token saving",
        )
        for value in SAVING_LEGEND_VALUES
    ]
    legend_axis.axis("off")
    arm_legend = legend_axis.legend(
        handles=arm_and_size_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=7.3,
        handletextpad=0.45,
        columnspacing=0.9,
        title="Fill = arm · shape = group size",
        title_fontsize=7.5,
    )
    legend_axis.add_artist(arm_legend)
    legend_axis.legend(
        handles=saving_handles,
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=7.3,
        handletextpad=0.5,
        columnspacing=1.15,
        title="Bubble area ∝ absolute token saving",
        title_fontsize=7.5,
    )
    legend_axis.text(
        0.5,
        0.02,
        "Shared focused 2–10% axes; dashed line = equal rescue and interference.",
        transform=legend_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.35",
    )
    return min(savings), max(savings)


def _panel_c(axis: Any) -> None:
    axis.axis("off")
    axis.set_title("C  Joint action-routing motivation", loc="left", fontweight="bold")
    actions = (
        "Baran fallback",
        "Singleton LLM",
        "Pattern groups",
        "Semantic groups",
        "Other structured groups",
    )
    y_positions = np.linspace(0.86, 0.34, len(actions))
    for label, y in zip(actions, y_positions):
        axis.text(
            0.08,
            y,
            label,
            transform=axis.transAxes,
            ha="left",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#f2f2f2", "ec": "#777777"},
        )
        axis.annotate(
            "",
            xy=(0.66, 0.18),
            xytext=(0.57, y),
            xycoords=axis.transAxes,
            arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 1.0},
        )
    axis.text(
        0.68,
        0.18,
        "Joint budgeted\nquery-action routing",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.55", "fc": "#d9edf7", "ec": "#2c7fb8"},
    )
    axis.text(
        0.5,
        0.04,
        "Concept only — no Router outcomes are used",
        transform=axis.transAxes,
        ha="center",
        color="0.35",
        fontsize=8,
    )


def _write_figure(
    micro: Mapping[str, str],
    macro_rows: Sequence[Mapping[str, str]],
    *,
    pdf_path: Path,
    svg_path: Path,
) -> tuple[float, float]:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.hashsalt": AMENDMENT_SCHEMA_VERSION,
        }
    ):
        figure = plt.figure(figsize=(17.2, 5.5), layout="constrained")
        outer = figure.add_gridspec(1, 3, width_ratios=(1.08, 1.95, 1.18))
        panel_a = figure.add_subplot(outer[0, 0])
        panel_b = figure.add_subfigure(outer[0, 1])
        panel_b.suptitle(
            "B  Group benefit–interference (9-dataset macro)",
            x=0.0,
            ha="left",
            fontweight="bold",
        )
        middle = panel_b.add_gridspec(2, 2, height_ratios=(1.0, 0.24))
        panel_b_axes = {
            "pattern": panel_b.add_subplot(middle[0, 0]),
            "semantic": panel_b.add_subplot(middle[0, 1]),
        }
        legend_axis = panel_b.add_subplot(middle[1, :])
        panel_c = figure.add_subplot(outer[0, 2])

        _panel_a(panel_a, micro)
        saving_range = _panel_b(panel_b_axes, legend_axis, macro_rows)
        _panel_c(panel_c)

        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_tmp = pdf_path.with_name(f".{pdf_path.name}.{os.getpid()}.tmp")
        svg_tmp = svg_path.with_name(f".{svg_path.name}.{os.getpid()}.tmp")
        figure.savefig(
            pdf_tmp,
            format="pdf",
            bbox_inches="tight",
            metadata={
                "Creator": AMENDMENT_SCHEMA_VERSION,
                "CreationDate": None,
                "ModDate": None,
            },
        )
        figure.savefig(
            svg_tmp,
            format="svg",
            bbox_inches="tight",
            metadata={"Creator": AMENDMENT_SCHEMA_VERSION, "Date": None},
        )
        plt.close(figure)
        pdf_tmp.replace(pdf_path)
        svg_tmp.replace(svg_path)
    return saving_range


def _markdown(
    macro_rows: Sequence[Mapping[str, str]],
    *,
    saving_range: tuple[float, float],
) -> str:
    table = [
        "| Condition | S acc. | G acc. | R acc. | G rescue | "
        "G interference | G−S | G token saving |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in macro_rows:
        table.append(
            "| "
            + " | ".join(
                (
                    f"{row['source_view']}/k={row['group_size']}",
                    f"{_finite(row, 'singleton_accuracy'):.4f}",
                    f"{_finite(row, 'structured_accuracy'):.4f}",
                    f"{_finite(row, 'random_accuracy'):.4f}",
                    f"{_finite(row, 'structured_rescue_rate'):.4f}",
                    f"{_finite(row, 'structured_interference_rate'):.4f}",
                    f"{_finite(row, 'structured_minus_singleton'):.4f}",
                    f"{_finite(row, 'structured_token_saving'):.4f}",
                )
            )
            + " |"
        )
    return "\n".join(
        (
            "# Publication Visualization Amendment",
            "",
            "This is a visualization-only, offline amendment to the frozen Introduction "
            "motivation evidence run. It does not change queries, responses, ledgers, "
            "estimands, confidence intervals, Holm families, or canonical report outputs.",
            "",
            "[Publication PDF](../figures/introduction_motivation_publication.pdf) · "
            "[Publication SVG](../figures/introduction_motivation_publication.svg) · "
            "[Canonical audit report](report.md)",
            "",
            "## Visualization change",
            "",
            "Panel B is faceted by pattern versus semantic view, uses shared focused 2–10% "
            "axes, deterministic direct labels (`G2`, `R2`, and so on), and places its "
            "legend outside the plotting areas. Bubble area is proportional to absolute "
            "token saving, with explicit 18%, 25%, and 31% reference sizes. The observed "
            f"macro range is {saving_range[0]:.1%}–{saving_range[1]:.1%}. Panels A and C "
            "retain the canonical semantics; Panel C remains conceptual and contains no "
            "Router outcome.",
            "",
            "The publication figure supersedes the canonical figure only for visual "
            "presentation. The canonical PDF/SVG and Markdown remain retained as the "
            "frozen audit outputs.",
            "",
            "## Frozen Panel B values",
            "",
            *table,
            "",
            "## Provenance",
            "",
            "See [`../provenance/visualization_amendment.json`]"
            "(../provenance/visualization_amendment.json) "
            "for input hashes, the frozen reporting-code hash, the amendment script hash, "
            "and the no-network/no-API declaration.",
            "",
        )
    )


def build(run_dir: Path, script_path: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_id = str(run_manifest.get("run_id", ""))
    if run_id != DEFAULT_RUN_ID or run_dir.name != DEFAULT_RUN_ID:
        raise ValueError(f"unexpected run identity: {run_id!r} at {run_dir}")

    project_root = script_path.resolve().parents[1]
    fingerprint_path = run_dir / "provenance" / "data_fingerprint.json"
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    reporting_relative = "src/budgeted_group_repair_no_baran/motivation_reporting.py"
    frozen_reporting_sha = str(fingerprint["code_sha256"][reporting_relative])
    reporting_path = project_root / reporting_relative
    current_reporting_sha = _sha256(reporting_path)
    if current_reporting_sha != frozen_reporting_sha:
        raise RuntimeError("frozen motivation_reporting.py fingerprint drift")

    consumed = {
        "complementarity_metrics": run_dir / "metrics" / "complementarity_by_dataset.csv",
        "group_metrics": run_dir / "metrics" / "group_by_dataset_view_size.csv",
    }
    upstream = {
        "complementarity_ledger": run_dir / "records" / "complementarity_cell_outcomes.csv",
        "group_ledger": run_dir / "records" / "group_cell_outcomes.csv",
        "api_cost_audit": run_dir / "metrics" / "api_cost_audit.csv",
    }
    canonical = {
        "figure_pdf": run_dir / "figures" / "introduction_motivation.pdf",
        "figure_svg": run_dir / "figures" / "introduction_motivation.svg",
        "report": run_dir / "report" / "report.md",
    }
    for path in (*consumed.values(), *upstream.values(), *canonical.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    canonical_before = {name: _sha256(path) for name, path in canonical.items()}

    micro, macro_rows = _load_inputs(run_dir)
    output_pdf = run_dir / "figures" / "introduction_motivation_publication.pdf"
    output_svg = run_dir / "figures" / "introduction_motivation_publication.svg"
    output_report = run_dir / "report" / "report_publication.md"
    saving_range = _write_figure(
        micro,
        macro_rows,
        pdf_path=output_pdf,
        svg_path=output_svg,
    )
    _atomic_text(output_report, _markdown(macro_rows, saving_range=saving_range))

    canonical_after = {name: _sha256(path) for name, path in canonical.items()}
    if canonical_after != canonical_before:
        raise RuntimeError("canonical report output changed during amendment generation")

    provenance_path = run_dir / "provenance" / "visualization_amendment.json"
    payload: dict[str, Any] = {
        "schema_version": AMENDMENT_SCHEMA_VERSION,
        "amendment_id": "publication-panel-b-layout-v1",
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "scope": "offline visualization and publication amendment only",
        "reason": (
            "Canonical Panel B had overlapping direct labels and an overlapping legend/note; "
            "the publication amendment facets the two views and moves an absolute saving legend "
            "outside the plotting areas."
        ),
        "execution_declarations": {
            "network_calls": 0,
            "api_calls": 0,
            "llm_responses_regenerated": False,
            "provider_credentials_read": False,
            "canonical_outputs_modified": False,
            "frozen_ledgers_modified": False,
            "statistical_results_modified": False,
        },
        "visualization_config": {
            "views": list(VIEWS),
            "group_sizes": list(GROUP_SIZES),
            "focused_shared_axis_limits": list(FOCUSED_AXIS_LIMITS),
            "direct_label_offsets_points": {
                "/".join((view, arm, str(size))): list(offset)
                for (view, arm, size), offset in sorted(LABEL_OFFSETS.items())
            },
            "bubble_area_formula": "900 * abs(token_saving)",
            "saving_reference_values": list(SAVING_LEGEND_VALUES),
            "observed_saving_range": list(saving_range),
            "negative_saving_encoding": "x overlay denotes token overhead",
        },
        "consumed_inputs_sha256": {
            str(path.relative_to(run_dir)): _sha256(path) for path in consumed.values()
        },
        "upstream_finalized_artifacts_sha256": {
            str(path.relative_to(run_dir)): _sha256(path) for path in upstream.values()
        },
        "frozen_reporting_code": {
            "path": reporting_relative,
            "sha256": frozen_reporting_sha,
            "matches_current_file": True,
            "fingerprint_source": str(fingerprint_path.relative_to(run_dir)),
        },
        "amendment_script": {
            "path": str(script_path.resolve().relative_to(project_root)),
            "sha256": _sha256(script_path.resolve()),
            "outside_frozen_code_set": True,
        },
        "canonical_outputs": {
            name: {
                "path": str(path.relative_to(run_dir)),
                "sha256_before": canonical_before[name],
                "sha256_after": canonical_after[name],
                "retained": True,
                "superseded_for_publication_visual_only": name.startswith("figure_"),
            }
            for name, path in canonical.items()
        },
        "publication_outputs_sha256": {
            str(path.relative_to(run_dir)): _sha256(path)
            for path in (output_pdf, output_svg, output_report)
        },
    }
    _atomic_json(provenance_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs" / DEFAULT_RUN_ID,
    )
    args = parser.parse_args()
    payload = build(args.run_dir, Path(__file__))
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
