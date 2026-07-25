"""Validated import of immutable Baran predictions from a prior formal run."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import read_jsonl, sha256_file
from .run_state import canonical_json_sha256, write_json


@dataclass(frozen=True, slots=True)
class BaranReference:
    source_run: Path
    resolved_baran_dir: Path
    manifest: Mapping[str, object]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def import_baran_reference(
    source_run: str | Path,
    destination: str | Path,
    *,
    selected_datasets: Sequence[tuple[str, str]],
    data_manifest_path: str | Path,
    expected_seed: int,
    expected_labeling_budget: int,
    expected_raha_code_sha256: str,
) -> BaranReference:
    source = Path(source_run).expanduser().resolve()
    run_manifest_path = source / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    run_manifest = _read_object(run_manifest_path)
    expected_data_hash = str(run_manifest.get("input_data_manifest_sha256", ""))
    observed_data_hash = sha256_file(data_manifest_path)
    if not expected_data_hash or expected_data_hash != observed_data_hash:
        raise ValueError("Baran reference data manifest does not match the local snapshot")
    experiment = run_manifest.get("experiment_config")
    if not isinstance(experiment, Mapping):
        raise ValueError("source run lacks experiment_config")
    if int(experiment.get("baran_seed", -1)) != int(expected_seed):
        raise ValueError("Baran seed mismatch")
    if int(experiment.get("baran_labeling_budget", -1)) != int(expected_labeling_budget):
        raise ValueError("Baran labeling budget mismatch")
    source_raha_hash = str(run_manifest.get("raha_code_sha256", ""))
    if not source_raha_hash or source_raha_hash != str(expected_raha_code_sha256):
        raise ValueError("Baran reference RAHA/Baran code hash mismatch")
    baran_link = source / "baran"
    resolved_baran = baran_link.resolve(strict=True)
    target = Path(destination).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    total_records = 0
    for suite, dataset in selected_datasets:
        name = f"{suite}__{dataset}.jsonl"
        source_file = resolved_baran / name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        rows = read_jsonl(source_file)
        if not rows:
            raise ValueError(f"empty Baran reference: {source_file}")
        for row in rows:
            if str(row.get("suite")) != suite or str(row.get("dataset")) != dataset:
                raise ValueError(f"mixed dataset records in {source_file}")
            if int(row.get("seed", -1)) != int(expected_seed):
                raise ValueError(f"record-level Baran seed mismatch in {source_file}")
            if int(row.get("labeling_budget", -1)) != int(expected_labeling_budget):
                raise ValueError(f"record-level Baran budget mismatch in {source_file}")
        cell_ids = [str(row.get("cell_id", "")) for row in rows]
        if not all(cell_ids) or len(cell_ids) != len(set(cell_ids)):
            raise ValueError(f"Baran reference has missing or duplicate cell IDs: {source_file}")
        destination_file = target / name
        shutil.copy2(source_file, destination_file)
        digest = sha256_file(source_file)
        if sha256_file(destination_file) != digest:
            raise IOError(f"copied Baran reference hash mismatch: {name}")
        files.append(
            {
                "suite": suite,
                "dataset": dataset,
                "source_file": str(source_file),
                "copied_file": destination_file.name,
                "sha256": digest,
                "record_count": len(rows),
            }
        )
        total_records += len(rows)
    manifest: dict[str, object] = {
        "source_run_id": str(run_manifest.get("run_id", source.name)),
        "source_run_manifest_sha256": sha256_file(run_manifest_path),
        "source_run_binding_fingerprint": str(run_manifest.get("binding_fingerprint", "")),
        "data_manifest_sha256": observed_data_hash,
        "raha_code_sha256": source_raha_hash,
        "baran_seed": int(expected_seed),
        "baran_labeling_budget": int(expected_labeling_budget),
        "baran_link": str(baran_link),
        "resolved_baran_dir": str(resolved_baran),
        "files": files,
        "record_count": total_records,
    }
    manifest["reference_fingerprint"] = canonical_json_sha256(manifest)
    write_json(target / "baran_reference_manifest.json", manifest)
    return BaranReference(source, resolved_baran, manifest)


def load_baran_records(reference_dir: str | Path, suite: str, dataset: str) -> list[dict[str, object]]:
    return read_jsonl(Path(reference_dir) / f"{suite}__{dataset}.jsonl")


__all__ = ["BaranReference", "import_baran_reference", "load_baran_records"]
