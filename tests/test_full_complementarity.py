from __future__ import annotations

from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.cli import parse_args
from budgeted_group_repair_no_baran.full_complementarity import (
    BASELINE_MANIFEST_SCHEMA,
    DEFAULT_SOURCE_RUN,
    aggregate_outcomes,
    build_full_complementarity,
    pair_baseline_records,
    validate_baseline_manifest,
)
from budgeted_group_repair_no_baran.run_state import sha256_file, write_json


def _record(
    method: str,
    cell_id: str,
    prediction: str | None,
    *,
    clean: str = "clean",
) -> dict[str, object]:
    valid = prediction is not None
    row: dict[str, object] = {
        "method": method,
        "scenario": "baseline",
        "backend": "none",
        "budget_share": None,
        "group_size_variant": "all" if method == "baran" else "1",
        "suite": "source",
        "dataset": "toy",
        "cell_id": cell_id,
        "clean_value": clean,
        "prediction": prediction,
        "parse_status": "ok_baran" if method == "baran" else "ok_llm_only",
        "valid_prediction": valid,
        "correct_repair": prediction == clean,
        "final_source": "baran" if method == "baran" else "llm",
    }
    if method == "llm_only":
        row.update(
            {
                "baran_fallback_used": False,
                "selected_query_id": f"q-{cell_id}",
                "llm_decision": "propose",
            }
        )
    return row


def _four_cell_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cell_ids = [f"source:toy:{index}:0" for index in range(4)]
    baran = [
        _record("baran", cell_ids[0], "clean"),
        _record("baran", cell_ids[1], "clean"),
        _record("baran", cell_ids[2], "wrong"),
        _record("baran", cell_ids[3], "wrong"),
    ]
    llm = [
        _record("llm_only", cell_ids[0], "clean"),
        _record("llm_only", cell_ids[1], "wrong"),
        _record("llm_only", cell_ids[2], "clean"),
        _record("llm_only", cell_ids[3], "wrong"),
    ]
    return baran, llm


def test_full_complementarity_pairs_exact_four_cells() -> None:
    baran, llm = _four_cell_records()
    paired = pair_baseline_records(baran, llm, require_formal_population=False)
    summary = aggregate_outcomes(
        paired,
        scope="dataset",
        suite="source",
        dataset="toy",
        bootstrap_replicates=50,
        bootstrap_seed=45,
        confidence=0.95,
    )
    assert (summary["n11"], summary["n10"], summary["n01"], summary["n00"]) == (
        1,
        1,
        1,
        1,
    )
    assert summary["baran_accuracy"] == 0.5
    assert summary["llm_accuracy"] == 0.5
    assert summary["oracle_upper_bound"] == 0.75
    assert summary["upper_bound_minus_baran"] == 0.25
    assert summary["disagreement_rate"] == 0.5
    assert summary["mcnemar_p"] == 1.0


def test_full_complementarity_rejects_fallback_and_population_mismatch() -> None:
    baran, llm = _four_cell_records()
    llm[0]["baran_fallback_used"] = True
    with pytest.raises(ValueError, match="not pure No-Baran"):
        pair_baseline_records(baran, llm, require_formal_population=False)
    llm[0]["baran_fallback_used"] = False
    with pytest.raises(ValueError, match="cell universes differ"):
        pair_baseline_records(baran, llm[:-1], require_formal_population=False)


def test_baseline_manifest_hash_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    source.mkdir()
    bundle.mkdir()
    source_payload = source / "run_manifest.json"
    source_payload.write_text('{"status":"complete"}\n', encoding="utf-8")
    payload = bundle / "baran_only.jsonl"
    payload.write_text('{"cell_id":"c"}\n', encoding="utf-8")
    write_json(
        bundle / "manifest.json",
        {
            "schema_version": BASELINE_MANIFEST_SCHEMA,
            "source_run_resolved": str(source),
            "source_files": {
                source_payload.name: {
                    "bytes": source_payload.stat().st_size,
                    "sha256": sha256_file(source_payload),
                }
            },
            "outputs": {
                payload.name: {
                    "bytes": payload.stat().st_size,
                    "sha256": sha256_file(payload),
                }
            },
        },
    )
    assert validate_baseline_manifest(bundle)["ok"] is True
    payload.write_text('{"cell_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_baseline_manifest(bundle)


def test_full_complementarity_cli_is_offline_and_configurable() -> None:
    args = parse_args(
        [
            "analyze-full-complementarity",
            "--source-run",
            "runs/source",
            "--baseline-dir",
            "runs/baseline",
            "--output-dir",
            "runs/analysis",
            "--bootstrap-replicates",
            "100",
        ]
    )
    assert args.command == "analyze-full-complementarity"
    assert args.source_run == Path("runs/source")
    assert args.bootstrap_replicates == 100
    assert not hasattr(args, "token_cap")


@pytest.mark.skipif(
    not (DEFAULT_SOURCE_RUN / "final" / "all_methods.jsonl").is_file(),
    reason="frozen full-nine Router-v3 run is not available",
)
def test_canonical_full_complementarity_regression(tmp_path: Path) -> None:
    result = build_full_complementarity(
        DEFAULT_SOURCE_RUN,
        baseline_dir=tmp_path / "baseline",
        output_dir=tmp_path / "analysis",
        bootstrap_replicates=10,
    )
    assert result["network_calls"] == 0
    assert result["cells"] == 22_198
    assert result["quadrants"] == [12_335, 5_408, 1_555, 2_900]
    assert result["physical_calls_required_for_exact_reuse"] == 0
    validation = result["baseline_validation"]
    assert validation["output_files_checked"] == 3
    assert validation["source_files_checked"] == 5
