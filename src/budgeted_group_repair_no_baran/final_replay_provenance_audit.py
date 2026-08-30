"""Additive provenance correction for final-replay copy-on-write outputs.

``final_replay`` initially links ``metrics/logical_budget_ledger.csv`` from the
historical source run.  The metrics stage writes that path atomically, replacing
the symlink with a run-local regular file while leaving the source file intact.
The original provenance therefore describes the initial materialisation, not
the post-metrics state.  This offline utility records that lifecycle transition
in a new artifact without editing the original provenance or run manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .run_state import sha256_file, utc_now


LEDGER_RELATIVE_PATH = "metrics/logical_budget_ledger.csv"
CORRECTION_RELATIVE_PATH = Path(
    "provenance/final_replay_copy_on_write_correction.json"
)
_REQUIRED_COMPLETE_STAGES = ("final_records", "metrics", "audit")


@dataclass(frozen=True)
class _PlannedCorrection:
    destination: Path
    original_provenance: Path
    original_provenance_sha256: str
    payload: dict[str, Any]


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _linked_ledger_record(provenance: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    raw_records = provenance.get("linked_read_only_inputs")
    if not isinstance(raw_records, list):
        raise ValueError("final replay provenance has no linked_read_only_inputs list")
    matches = [
        (index, dict(record))
        for index, record in enumerate(raw_records)
        if isinstance(record, Mapping)
        and str(record.get("relative_path", "")) == LEDGER_RELATIVE_PATH
    ]
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one logical-budget-ledger provenance record, "
            f"observed={len(matches)}"
        )
    index, record = matches[0]
    if record.get("link_read_only_input") is not True:
        raise ValueError("logical-budget-ledger record is not an initial read-only link")
    return index, record


def plan_copy_on_write_correction(run_dir: str | Path) -> _PlannedCorrection:
    """Validate one completed replay and plan its additive correction.

    This function is read-only.  It intentionally rejects running or otherwise
    incomplete runs, so the correction can only be generated after all stages
    that may replace the ledger path have finished.
    """

    run = Path(run_dir).resolve()
    if not run.is_dir():
        raise FileNotFoundError(f"final replay run directory does not exist: {run}")

    manifest_path = run / "run_manifest.json"
    manifest = _load_json_object(manifest_path)
    if str(manifest.get("status", "")) != "complete":
        raise ValueError(
            f"final replay run is not complete: {run.name} "
            f"(status={manifest.get('status')!r})"
        )
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("completed run manifest has no stages object")
    stage_statuses = {
        stage: str(value.get("status", "")) if isinstance(value, Mapping) else ""
        for stage in _REQUIRED_COMPLETE_STAGES
        for value in (stages.get(stage),)
    }
    incomplete = [stage for stage, status in stage_statuses.items() if status != "complete"]
    if incomplete:
        raise ValueError(f"final replay stages are incomplete: {incomplete}")

    provenance_path = run / "provenance" / "final_replay.json"
    provenance = _load_json_object(provenance_path)
    if str(provenance.get("run_kind", "")) != "strict_offline_mgreedy_final_replay":
        raise ValueError("provenance is not a strict offline final replay")
    record_index, record = _linked_ledger_record(provenance)

    local_ledger = run / LEDGER_RELATIVE_PATH
    if not local_ledger.is_file():
        raise FileNotFoundError(f"recomputed logical budget ledger is missing: {local_ledger}")
    if local_ledger.is_symlink():
        raise ValueError(
            "logical budget ledger is still a symlink; no copy-on-write transition "
            "can be recorded"
        )

    source_text = str(record.get("source_path", ""))
    source_ledger = Path(source_text)
    if not source_text or not source_ledger.is_absolute():
        raise ValueError("logical-budget-ledger provenance has no absolute source_path")
    source_ledger = source_ledger.resolve()
    if not source_ledger.is_file():
        raise FileNotFoundError(f"source logical budget ledger is missing: {source_ledger}")
    if os.path.samefile(local_ledger, source_ledger):
        raise ValueError("run-local and source logical budget ledgers are the same file")

    recorded_source_sha256 = str(record.get("source_sha256", ""))
    current_source_sha256 = sha256_file(source_ledger)
    if current_source_sha256 != recorded_source_sha256:
        raise ValueError(
            "source logical budget ledger changed after initial replay materialisation"
        )
    local_sha256 = sha256_file(local_ledger)
    original_provenance_sha256 = sha256_file(provenance_path)
    destination = run / CORRECTION_RELATIVE_PATH
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite provenance correction: {destination}")

    run_id = str(manifest.get("run_id", ""))
    if not run_id:
        raise ValueError("completed run manifest has no run_id")
    payload: dict[str, Any] = {
        "schema_version": "final-replay-copy-on-write-correction-v1",
        "correction_kind": "additive_provenance_correction",
        "generated_at": utc_now(),
        "run_id": run_id,
        "run_manifest": {
            "relative_path": "run_manifest.json",
            "sha256": sha256_file(manifest_path),
            "status": "complete",
            "validated_stage_statuses": stage_statuses,
        },
        "original_provenance": {
            "relative_path": "provenance/final_replay.json",
            "sha256": original_provenance_sha256,
            "ledger_record_json_pointer": (
                f"/linked_read_only_inputs/{record_index}"
            ),
            "preserved_without_modification": True,
        },
        "corrected_artifact": {
            "relative_path": LEDGER_RELATIVE_PATH,
            "initial_materialisation": {
                "classification": "read_only_symlink_input",
                "source_path": str(source_ledger),
                "recorded_source_sha256": recorded_source_sha256,
            },
            "post_metrics_state": {
                "classification": "run_local_recomputed_output",
                "is_symlink": False,
                "is_same_file_as_source": False,
                "sha256": local_sha256,
                "content_differs_from_source": local_sha256 != current_source_sha256,
            },
            "source_integrity_at_correction": {
                "current_sha256": current_source_sha256,
                "matches_initial_recorded_sha256": True,
            },
            "observed_lifecycle": (
                "initial symlink recorded; metrics stage completed; run-local "
                "regular output observed"
            ),
        },
        "correction_semantics": {
            "clarification": (
                "The original linked_read_only_inputs entry describes initial "
                "materialisation only. After metrics, this path is a run-local "
                "recomputed output created through copy-on-write replacement."
            ),
            "original_provenance_is_not_overwritten": True,
            "network_or_api_calls": False,
        },
    }
    return _PlannedCorrection(
        destination=destination,
        original_provenance=provenance_path,
        original_provenance_sha256=original_provenance_sha256,
        payload=payload,
    )


def write_copy_on_write_corrections(
    run_dirs: Sequence[str | Path],
) -> list[Path]:
    """Exclusively create corrections after preflighting every supplied run."""

    if not run_dirs:
        raise ValueError("at least one completed final replay run is required")
    planned = [plan_copy_on_write_correction(run_dir) for run_dir in run_dirs]
    destinations = [item.destination for item in planned]
    if len(destinations) != len(set(destinations)):
        raise ValueError("duplicate final replay run directories were supplied")

    written: list[Path] = []
    for item in planned:
        encoded = json.dumps(
            item.payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        with item.destination.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
        if sha256_file(item.original_provenance) != item.original_provenance_sha256:
            raise RuntimeError("original final replay provenance changed during correction")
        written.append(item.destination)
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create additive copy-on-write provenance corrections for completed "
            "offline final-replay runs"
        )
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = write_copy_on_write_corrections(args.run_dirs)
    print(
        json.dumps(
            {"created": [str(path) for path in paths], "network_or_api_calls": False},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
