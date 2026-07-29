"""Auditable, run-local lifecycle and fingerprint binding helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


BINDING_KEYS = (
    "run_id",
    "protocol",
    "experiment_config",
    "experiment_config_sha256",
    "llm_config",
    "llm_config_sha256",
    "data_root",
    "input_data_manifest_sha256",
    "data_content_fingerprint",
    "raha_source_root",
    "raha_code_sha256",
    "implementation_sha256",
    "model",
    "prompt_schema_version",
    "prompt_schema_sha256",
    "baran_source_run",
    "baran_source_manifest_sha256",
    "response_reuse_run",
    "response_reuse_manifest_sha256",
    "router_artifact_reuse_run",
    "router_artifact_reuse_manifest_sha256",
    "router_comparison_run",
    "router_comparison_manifest_sha256",
    "binding_fingerprint",
)
_SENSITIVE_KEY_PARTS = ("api_key", "access_token", "bearer", "secret", "password", "cookie")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _redact(value: object, parent_key: str = "") -> object:
    normalized = parent_key.lower()
    if any(token in normalized for token in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(key): _redact(nested, str(key)) for key, nested in value.items()}
    if isinstance(value, tuple):
        return tuple(_redact(item) for item in value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def redacted_config(config: Mapping[str, object]) -> dict[str, object]:
    value = _redact(config)
    assert isinstance(value, dict)
    return value


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if not source.exists():
        return {}
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {source}")
    return value


def content_fingerprint(paths: Sequence[str | Path], *, root: str | Path | None = None) -> str:
    """Hash explicit files with stable labels; directories are rejected."""

    base = Path(root).resolve() if root is not None else None
    records: list[tuple[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"fingerprint input must be a file: {path}")
        if base is not None:
            try:
                label = path.relative_to(base).as_posix()
            except ValueError as error:
                raise ValueError(f"fingerprint input is outside configured root: {path}") from error
        else:
            label = path.name
        records.append((label, sha256_file(path)))
    labels = [label for label, _ in records]
    if len(labels) != len(set(labels)):
        raise ValueError("fingerprint labels are not unique")
    return canonical_json_sha256(sorted(records))


def build_run_binding(
    *,
    run_id: str,
    protocol: str,
    experiment_config_path: str | Path,
    llm_config_path: str | Path,
    data_manifest_path: str | Path,
    implementation_sha256: str,
    raha_code_sha256: str,
    model: str,
    prompt_schema_version: str,
    prompt_schema_sha256: str,
) -> dict[str, object]:
    """Build the minimum immutable identity for run/checkpoint reuse."""

    binding = {
        "run_id": str(run_id),
        "protocol": str(protocol),
        "experiment_config_sha256": sha256_file(experiment_config_path),
        "llm_config_sha256": sha256_file(llm_config_path),
        "input_data_manifest_sha256": sha256_file(data_manifest_path),
        "implementation_sha256": str(implementation_sha256),
        "raha_code_sha256": str(raha_code_sha256),
        "model": str(model),
        "prompt_schema_version": str(prompt_schema_version),
        "prompt_schema_sha256": str(prompt_schema_sha256),
    }
    if any(not str(value) for value in binding.values()):
        raise ValueError("run binding fields must be non-empty")
    binding["binding_fingerprint"] = canonical_json_sha256(binding)
    return binding


def checkpoint_key(
    binding_fingerprint: str,
    query_id: str,
    prompt_hash: str,
) -> str:
    if not binding_fingerprint or not query_id or not prompt_hash:
        raise ValueError("checkpoint binding, query ID, and prompt hash are required")
    return canonical_json_sha256(
        {
            "binding_fingerprint": binding_fingerprint,
            "query_id": query_id,
            "prompt_hash": prompt_hash,
        }
    )


def validate_checkpoint_record(
    record: Mapping[str, object],
    *,
    binding_fingerprint: str,
    query_id: str,
    prompt_hash: str,
) -> None:
    expected = checkpoint_key(binding_fingerprint, query_id, prompt_hash)
    observed = str(record.get("checkpoint_key", ""))
    if observed != expected:
        raise ValueError("checkpoint record does not match run/query/prompt fingerprints")


@dataclass(slots=True)
class RunState:
    run_dir: Path
    metadata_path: Path

    @classmethod
    def create(
        cls,
        run_dir: str | Path,
        metadata: Mapping[str, object],
        *,
        resume: bool = False,
    ) -> "RunState":
        destination = Path(run_dir).resolve()
        metadata_path = destination / "run_manifest.json"
        requested = redacted_config(metadata)
        if destination.exists() and any(destination.iterdir()) and not resume:
            raise FileExistsError(
                f"run directory is not empty: {destination}; choose a new run ID or pass --resume"
            )
        destination.mkdir(parents=True, exist_ok=True)
        if resume:
            if not metadata_path.is_file():
                raise FileNotFoundError(f"cannot resume without run_manifest.json: {destination}")
            existing = read_json(metadata_path)
            drift: dict[str, dict[str, Any]] = {}
            for key in BINDING_KEYS:
                if key in existing or key in requested:
                    if existing.get(key) != requested.get(key):
                        drift[key] = {"existing": existing.get(key), "requested": requested.get(key)}
            if drift:
                raise ValueError(
                    "refusing resume after run-binding drift: "
                    + json.dumps(drift, ensure_ascii=False, sort_keys=True, default=str)
                )
            return cls(destination, metadata_path)

        manifest = {
            **requested,
            "created_at": utc_now(),
            "status": "running",
            "fresh_run": True,
            "external_prediction_inputs": [],
            "host": socket.gethostname(),
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "pid": os.getpid(),
            "stages": {},
        }
        write_json(metadata_path, manifest)
        return cls(destination, metadata_path)

    @property
    def manifest(self) -> dict[str, object]:
        return read_json(self.metadata_path)

    def stage_completed(self, stage: str) -> bool:
        stages = self.manifest.get("stages", {})
        return (
            isinstance(stages, Mapping)
            and isinstance(stages.get(stage), Mapping)
            and stages[stage].get("status") == "complete"
        )

    def update_stage(self, stage: str, status: str, **details: object) -> None:
        if not stage.strip() or not status.strip():
            raise ValueError("stage and status are required")
        manifest = self.manifest
        stages_raw = manifest.get("stages", {})
        stages = dict(stages_raw) if isinstance(stages_raw, Mapping) else {}
        prior_raw = stages.get(stage, {})
        prior = dict(prior_raw) if isinstance(prior_raw, Mapping) else {}
        stages[stage] = {
            **prior,
            **redacted_config(details),
            "status": status,
            "updated_at": utc_now(),
        }
        manifest["stages"] = stages
        manifest["updated_at"] = utc_now()
        write_json(self.metadata_path, manifest)

    def complete(
        self,
        *,
        required_stages: Sequence[str] = (),
        **details: object,
    ) -> None:
        missing = [stage for stage in required_stages if not self.stage_completed(stage)]
        if missing:
            raise RuntimeError(f"cannot complete run; incomplete required stages: {missing}")
        manifest = self.manifest
        manifest.update(redacted_config(details))
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
        write_json(self.metadata_path, manifest)


__all__ = [
    "BINDING_KEYS",
    "RunState",
    "build_run_binding",
    "canonical_json_sha256",
    "checkpoint_key",
    "content_fingerprint",
    "read_json",
    "redacted_config",
    "sha256_file",
    "utc_now",
    "validate_checkpoint_record",
    "write_json",
]
