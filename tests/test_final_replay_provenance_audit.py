from __future__ import annotations

import json
from pathlib import Path

import pytest

from budgeted_group_repair_no_baran.final_replay_provenance_audit import (
    CORRECTION_RELATIVE_PATH,
    LEDGER_RELATIVE_PATH,
    plan_copy_on_write_correction,
    write_copy_on_write_corrections,
)
from budgeted_group_repair_no_baran.run_state import sha256_file


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _replay_fixture(tmp_path: Path, *, status: str = "complete") -> tuple[Path, Path]:
    source = tmp_path / "source" / LEDGER_RELATIVE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("kind,value\nsource,1\n", encoding="utf-8")

    run = tmp_path / "replay"
    local = run / LEDGER_RELATIVE_PATH
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("kind,value\nrecomputed,2\n", encoding="utf-8")
    stages = {
        name: {"status": "complete"}
        for name in ("final_records", "metrics", "audit")
    }
    _write_json(
        run / "run_manifest.json",
        {
            "run_id": "replay-test",
            "status": status,
            "stages": stages,
        },
    )
    _write_json(
        run / "provenance" / "final_replay.json",
        {
            "run_kind": "strict_offline_mgreedy_final_replay",
            "linked_read_only_inputs": [
                {
                    "relative_path": LEDGER_RELATIVE_PATH,
                    "source_path": str(source.resolve()),
                    "source_sha256": sha256_file(source),
                    "link_read_only_input": True,
                }
            ],
        },
    )
    return run, source


def test_correction_is_additive_and_records_post_metrics_state(tmp_path: Path) -> None:
    run, source = _replay_fixture(tmp_path)
    provenance = run / "provenance" / "final_replay.json"
    provenance_before = provenance.read_bytes()
    source_before = source.read_bytes()

    [created] = write_copy_on_write_corrections([run])

    assert created == run / CORRECTION_RELATIVE_PATH
    assert provenance.read_bytes() == provenance_before
    assert source.read_bytes() == source_before
    correction = json.loads(created.read_text(encoding="utf-8"))
    artifact = correction["corrected_artifact"]
    assert artifact["initial_materialisation"]["classification"] == (
        "read_only_symlink_input"
    )
    assert artifact["post_metrics_state"]["classification"] == (
        "run_local_recomputed_output"
    )
    assert artifact["post_metrics_state"]["is_symlink"] is False
    assert artifact["post_metrics_state"]["content_differs_from_source"] is True
    assert artifact["source_integrity_at_correction"][
        "matches_initial_recorded_sha256"
    ] is True
    assert correction["correction_semantics"]["network_or_api_calls"] is False

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_copy_on_write_corrections([run])
    assert provenance.read_bytes() == provenance_before


def test_incomplete_run_is_rejected_without_creating_artifact(tmp_path: Path) -> None:
    run, _ = _replay_fixture(tmp_path, status="running")

    with pytest.raises(ValueError, match="not complete"):
        write_copy_on_write_corrections([run])

    assert not (run / CORRECTION_RELATIVE_PATH).exists()


def test_still_linked_ledger_is_not_misclassified_as_copy_on_write(
    tmp_path: Path,
) -> None:
    run, source = _replay_fixture(tmp_path)
    local = run / LEDGER_RELATIVE_PATH
    local.unlink()
    local.symlink_to(source)

    with pytest.raises(ValueError, match="still a symlink"):
        plan_copy_on_write_correction(run)

    assert not (run / CORRECTION_RELATIVE_PATH).exists()


def test_batch_preflights_all_runs_before_writing(tmp_path: Path) -> None:
    complete, _ = _replay_fixture(tmp_path / "one")
    incomplete, _ = _replay_fixture(tmp_path / "two", status="running")

    with pytest.raises(ValueError, match="not complete"):
        write_copy_on_write_corrections([complete, incomplete])

    assert not (complete / CORRECTION_RELATIVE_PATH).exists()
    assert not (incomplete / CORRECTION_RELATIVE_PATH).exists()
