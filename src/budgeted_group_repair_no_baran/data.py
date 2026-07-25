"""Portable CARE dataset I/O with a hard online/oracle boundary.

The experiment knows error *coordinates* at deployment time, but it must not
leak clean values or TableEG generation annotations into grouping, prompting,
selection, or verification.  ``SafeCell`` and ``SafeDataset`` are therefore
deliberately separate physical records: their dataclass fields cannot hold any
oracle or error-generation annotation.
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, NewType, Sequence

import pandas as pd


SOURCE_DATASETS = ("hospital", "flights", "beers", "rayyan", "movies_1")
TABLEEG_DATASETS = (
    "company",
    "flight_10",
    "flight_20",
    "marketing",
    "movie_metadata_10",
    "restaurant_10",
    "restaurant_20",
    "restaurant_rule_only",
    "soccer",
)
SUITES = ("source", "tableeg")

EXPECTED_ORACLE_ERRORS: Mapping[tuple[str, str], int] = {
    ("source", "beers"): 4362,
    ("source", "flights"): 4920,
    ("source", "hospital"): 509,
    ("source", "movies_1"): 7675,
    ("source", "rayyan"): 948,
    ("tableeg", "company"): 572,
    ("tableeg", "flight_10"): 477,
    ("tableeg", "flight_20"): 658,
    ("tableeg", "marketing"): 723,
    ("tableeg", "movie_metadata_10"): 244,
    ("tableeg", "restaurant_10"): 304,
    ("tableeg", "restaurant_20"): 606,
    ("tableeg", "restaurant_rule_only"): 76,
    ("tableeg", "soccer"): 1883,
}
EXPECTED_DATASET_COUNT = 14
EXPECTED_ORACLE_ERROR_COUNT = 23_957
_COPY_SUFFIXES = (".csv", ".jsonl", ".json")
_FORBIDDEN_SAFE_FIELDS = frozenset(
    {
        "clean",
        "clean_path",
        "clean_value",
        "right_value",
        "correct_repair",
        "baran_correct",
        "llm_correct",
        "error_type",
        "error_value",
        "missing_value",
        "constraint",
        "tuple_pairs",
    }
)

CellId = NewType("CellId", str)


def normalize_value(value: object) -> str:
    """Apply CARE/Raha normalization without pandas NA coercion."""

    if value is None:
        value = ""
    normalized = html.unescape(str(value))
    normalized = re.sub(r"[\t\n ]+", " ", normalized, flags=re.UNICODE)
    return normalized.strip("\t\n ")


def normalize_for_match(value: object) -> str:
    return normalize_value(value)


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(
        Path(path),
        sep=",",
        header="infer",
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    if hasattr(frame, "map"):
        return frame.map(normalize_value)
    return frame.applymap(normalize_value)  # pragma: no cover - pandas < 2.1


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: str | Path, record: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SafeCell:
    """An error coordinate safe for every online BGR stage.

    Do not add clean values, correctness labels, or error-log annotations to
    this record.  Tests intentionally inspect the physical dataclass fields.
    """

    suite: str
    dataset: str
    row: int
    col: int
    column: str
    row_id: str
    dirty_value: str

    @property
    def cell_id(self) -> CellId:
        return CellId(f"{self.suite}:{self.dataset}:{self.row}:{self.col}")

    def as_dict(self) -> dict[str, object]:
        return {"cell_id": str(self.cell_id), **asdict(self)}


@dataclass(frozen=True, slots=True)
class OracleCell:
    """Evaluation/Baran-only cell; never pass this record to online stages."""

    suite: str
    dataset: str
    row: int
    col: int
    column: str
    row_id: str
    dirty_value: str
    clean_value: str
    error_type: str = ""
    missing_value: str = ""
    constraint: str = ""
    tuple_pairs: str = ""

    @property
    def cell_id(self) -> CellId:
        return CellId(f"{self.suite}:{self.dataset}:{self.row}:{self.col}")

    def to_safe(self) -> SafeCell:
        return SafeCell(
            suite=self.suite,
            dataset=self.dataset,
            row=self.row,
            col=self.col,
            column=self.column,
            row_id=self.row_id,
            dirty_value=self.dirty_value,
        )

    def as_evaluation_dict(self) -> dict[str, object]:
        return {"cell_id": str(self.cell_id), **asdict(self)}


@dataclass(frozen=True, slots=True)
class SafeDataset:
    """Dirty-only dataset projection consumed by online BGR components."""

    suite: str
    name: str
    dirty_path: Path
    dirty: pd.DataFrame
    cells: tuple[SafeCell, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.dirty.columns)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dirty.shape

    def cell_by_id(self) -> dict[CellId, SafeCell]:
        return {cell.cell_id: cell for cell in self.cells}


@dataclass(slots=True)
class LoadedDataset:
    """Oracle-bound normalized dirty/clean pair used only at the boundary."""

    suite: str
    name: str
    root: Path
    dirty_path: Path
    clean_path: Path
    dirty: pd.DataFrame
    clean: pd.DataFrame

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.dirty.columns]

    @property
    def shape(self) -> tuple[int, int]:
        return self.dirty.shape

    @property
    def annotation_path(self) -> Path | None:
        if self.suite == "tableeg":
            candidate = self.root / "error_log.jsonl"
            return candidate if candidate.exists() else None
        candidates = sorted(self.root.glob("*annotation.jsonl"))
        return candidates[0] if candidates else None

    def row_id_for_index(self, row: int) -> str:
        for name in ("row_id", "tuple_id", "index", "id"):
            if name in self.dirty.columns:
                return normalize_value(self.dirty.at[row, name])
        return str(row)

    def _annotation_map(self) -> dict[tuple[int, str], dict[str, str]]:
        """Read annotations only while constructing OracleCell records."""

        path = self.annotation_path
        if path is None:
            return {}
        row_id_column = next(
            (name for name in ("row_id", "tuple_id", "index", "id") if name in self.dirty.columns),
            None,
        )
        row_id_to_index = (
            {
                normalize_for_match(value): int(index)
                for index, value in enumerate(self.dirty[row_id_column].tolist())
            }
            if row_id_column is not None
            else {}
        )
        annotations: dict[tuple[int, str], dict[str, str]] = {}
        for entry in read_jsonl(path):
            raw_row = normalize_for_match(entry.get("row_id", ""))
            row_index = row_id_to_index.get(raw_row)
            if row_index is None:
                try:
                    row_index = int(raw_row)
                except ValueError:
                    continue
            column = str(entry.get("column", ""))
            if row_index < 0 or row_index >= len(self.dirty) or column not in self.columns:
                continue
            annotations[(row_index, column)] = {
                "error_type": str(entry.get("error_type") or ""),
                "missing_value": str(entry.get("missing_value") or ""),
                "constraint": "" if entry.get("constraint") is None else str(entry.get("constraint")),
                "tuple_pairs": "" if entry.get("tuple_pairs") is None else str(entry.get("tuple_pairs")),
            }
        return annotations

    def oracle_cells(self, *, include_annotations: bool = True) -> list[OracleCell]:
        if self.dirty.shape != self.clean.shape:
            raise ValueError(
                f"{self.suite}/{self.name}: dirty and clean shapes differ: "
                f"{self.dirty.shape} vs {self.clean.shape}"
            )
        # CARE source tables sometimes use different header spellings in the
        # clean file (for example ``city`` versus ``City``) while retaining an
        # identical positional schema.  Repair coordinates are positional and
        # online column names always come from the dirty table.
        annotations = self._annotation_map() if include_annotations else {}
        dirty_values = self.dirty.astype(str).values
        clean_values = self.clean.astype(str).values
        row_indices, col_indices = (dirty_values != clean_values).nonzero()
        result: list[OracleCell] = []
        for raw_row, raw_col in zip(row_indices, col_indices):
            row = int(raw_row)
            col = int(raw_col)
            column = self.columns[col]
            annotation = annotations.get((row, column), {})
            result.append(
                OracleCell(
                    suite=self.suite,
                    dataset=self.name,
                    row=row,
                    col=col,
                    column=column,
                    row_id=self.row_id_for_index(row),
                    dirty_value=normalize_value(dirty_values[row, col]),
                    clean_value=normalize_value(clean_values[row, col]),
                    error_type=annotation.get("error_type", "unlabeled_diff"),
                    missing_value=annotation.get("missing_value", ""),
                    constraint=annotation.get("constraint", ""),
                    tuple_pairs=annotation.get("tuple_pairs", ""),
                )
            )
        return result

    def safe_cells(self) -> list[SafeCell]:
        """Build the online error-coordinate projection without reading annotations.

        Error coordinates are the positional dirty/clean differences allowed by
        the benchmark protocol.  In particular, this method deliberately does
        not call :meth:`_annotation_map`, so TableEG generation fields never
        enter (or need to be read by) grouping, prompting, gating, selection,
        or verification.
        """

        if self.dirty.shape != self.clean.shape:
            raise ValueError(
                f"{self.suite}/{self.name}: dirty and clean shapes differ: "
                f"{self.dirty.shape} vs {self.clean.shape}"
            )
        dirty_values = self.dirty.astype(str).values
        clean_values = self.clean.astype(str).values
        row_indices, col_indices = (dirty_values != clean_values).nonzero()
        return [
            SafeCell(
                suite=self.suite,
                dataset=self.name,
                row=int(raw_row),
                col=int(raw_col),
                column=self.columns[int(raw_col)],
                row_id=self.row_id_for_index(int(raw_row)),
                dirty_value=normalize_value(dirty_values[int(raw_row), int(raw_col)]),
            )
            for raw_row, raw_col in zip(row_indices, col_indices)
        ]

    def safe_view(self) -> SafeDataset:
        return SafeDataset(
            suite=self.suite,
            name=self.name,
            dirty_path=self.dirty_path,
            dirty=self.dirty.copy(deep=True),
            cells=tuple(self.safe_cells()),
        )


def assert_safe_schema() -> None:
    physical_fields = {field.name for field in fields(SafeCell)} | {
        field.name for field in fields(SafeDataset)
    }
    leaked = physical_fields & _FORBIDDEN_SAFE_FIELDS
    if leaked:
        raise AssertionError(f"safe data records contain forbidden fields: {sorted(leaked)}")


assert_safe_schema()


def suite_datasets(suite: str) -> tuple[str, ...]:
    if suite == "source":
        return SOURCE_DATASETS
    if suite == "tableeg":
        return TABLEEG_DATASETS
    raise ValueError(f"unknown suite {suite!r}; expected one of {SUITES}")


def dataset_dir(suite: str, name: str, data_root: str | Path) -> Path:
    if name not in suite_datasets(suite):
        raise ValueError(f"unknown dataset {suite}/{name}")
    return Path(data_root) / suite / name


def load_dataset(suite: str, name: str, data_root: str | Path) -> LoadedDataset:
    root = dataset_dir(suite, name, data_root)
    dirty_path = root / "dirty.csv"
    clean_path = root / "clean.csv"
    if not dirty_path.is_file() or not clean_path.is_file():
        raise FileNotFoundError(f"missing dirty/clean files for {suite}/{name} under {root}")
    return LoadedDataset(
        suite=suite,
        name=name,
        root=root,
        dirty_path=dirty_path,
        clean_path=clean_path,
        dirty=read_csv(dirty_path),
        clean=read_csv(clean_path),
    )


def load_safe_dataset(suite: str, name: str, data_root: str | Path) -> SafeDataset:
    """Bind oracle coordinates once, then return a physically dirty-only view."""

    return load_dataset(suite, name, data_root).safe_view()


def load_datasets(
    suite: str,
    names: Sequence[str] | None = None,
    data_root: str | Path = "data",
) -> dict[str, LoadedDataset]:
    selected = tuple(names or suite_datasets(suite))
    return {name: load_dataset(suite, name, data_root) for name in selected}


def iter_oracle_cells(datasets: Mapping[str, LoadedDataset]) -> Iterable[OracleCell]:
    for dataset in datasets.values():
        yield from dataset.oracle_cells()


def _dataset_file_manifest(root: Path) -> dict[str, dict[str, object]]:
    files: dict[str, dict[str, object]] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in _COPY_SUFFIXES:
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return files


def build_portable_manifest(data_root: str | Path) -> dict[str, object]:
    """Recompute a path-independent manifest from the local 14-dataset copy."""

    root = Path(data_root).resolve()
    manifest: dict[str, object] = {
        "schema_version": 2,
        "sha256_algorithm": "sha256",
        "data_root": ".",
        "normalization": "html-unescape+collapse-ascii-whitespace+strip",
        "benchmark": "CARE 5 source + 9 TableEG snapshot",
        "source": {},
        "tableeg": {},
        "unavailable": {},
    }
    for suite in SUITES:
        suite_manifest = manifest[suite]
        assert isinstance(suite_manifest, dict)
        for name in suite_datasets(suite):
            dataset = load_dataset(suite, name, root)
            annotation = dataset.annotation_path
            suite_manifest[name] = {
                "shape": list(dataset.shape),
                "oracle_errors": len(dataset.oracle_cells()),
                "annotation": (
                    annotation.relative_to(root).as_posix() if annotation is not None else ""
                ),
                "files": _dataset_file_manifest(dataset.root),
            }
    return manifest


def write_portable_manifest(
    data_root: str | Path,
    manifest_path: str | Path | None = None,
) -> dict[str, object]:
    root = Path(data_root)
    destination = Path(manifest_path) if manifest_path is not None else root / "manifest.json"
    payload = build_portable_manifest(root)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return payload


@dataclass(frozen=True, slots=True)
class ManifestAudit:
    dataset_count: int
    oracle_error_count: int
    file_count: int
    hashes_verified: bool
    portable: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _manifest_is_portable(manifest: Mapping[str, object]) -> bool:
    if manifest.get("data_root") not in (None, "."):
        return False
    for forbidden in ("source_root", "copied_from"):
        if forbidden in manifest:
            return False
    for suite in SUITES:
        raw_suite = manifest.get(suite, {})
        if not isinstance(raw_suite, Mapping):
            return False
        for raw_entry in raw_suite.values():
            if not isinstance(raw_entry, Mapping):
                return False
            annotation = str(raw_entry.get("annotation", ""))
            if annotation and Path(annotation).is_absolute():
                return False
            if "copied_from" in raw_entry:
                return False
    return True


def validate_manifest(
    data_root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    require_portable: bool = True,
) -> ManifestAudit:
    """Verify exact dataset coverage, 23,957 diffs, and every listed hash."""

    root = Path(data_root).resolve()
    source = Path(manifest_path) if manifest_path is not None else root / "manifest.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a JSON object")
    portable = _manifest_is_portable(raw)
    if require_portable and not portable:
        raise ValueError("manifest contains machine-specific paths; rebuild it with write_portable_manifest")
    if raw.get("unavailable") not in ({}, None):
        raise ValueError("manifest reports unavailable datasets")

    dataset_count = 0
    oracle_count = 0
    file_count = 0
    for suite in SUITES:
        raw_suite = raw.get(suite)
        if not isinstance(raw_suite, Mapping):
            raise ValueError(f"manifest suite {suite!r} must be an object")
        expected_names = set(suite_datasets(suite))
        if set(raw_suite) != expected_names:
            raise ValueError(
                f"manifest {suite} datasets differ: expected {sorted(expected_names)}, "
                f"found {sorted(raw_suite)}"
            )
        for name in suite_datasets(suite):
            entry = raw_suite[name]
            if not isinstance(entry, Mapping):
                raise ValueError(f"manifest entry {suite}/{name} must be an object")
            files = entry.get("files")
            if not isinstance(files, Mapping) or not {"dirty.csv", "clean.csv"}.issubset(files):
                raise ValueError(f"manifest entry {suite}/{name} lacks dirty.csv/clean.csv hashes")
            dataset_root = (root / suite / name).resolve()
            for file_name, file_record in files.items():
                if Path(str(file_name)).name != str(file_name):
                    raise ValueError(f"unsafe manifest file path: {file_name!r}")
                if not isinstance(file_record, Mapping):
                    raise ValueError(f"invalid file record for {suite}/{name}/{file_name}")
                path = dataset_root / str(file_name)
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected_hash = str(file_record.get("sha256", ""))
                actual_hash = sha256_file(path)
                if not expected_hash or actual_hash != expected_hash:
                    raise ValueError(
                        f"SHA-256 mismatch for {suite}/{name}/{file_name}: "
                        f"expected {expected_hash}, found {actual_hash}"
                    )
                expected_bytes = file_record.get("bytes")
                if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
                    raise ValueError(f"byte-size mismatch for {suite}/{name}/{file_name}")
                file_count += 1

            dataset = load_dataset(suite, name, root)
            shape = [int(value) for value in entry.get("shape", [])]
            if shape != list(dataset.shape):
                raise ValueError(f"shape mismatch for {suite}/{name}")
            actual_oracle = len(dataset.oracle_cells())
            expected_oracle = EXPECTED_ORACLE_ERRORS[(suite, name)]
            if int(entry.get("oracle_errors", -1)) != expected_oracle or actual_oracle != expected_oracle:
                raise ValueError(
                    f"oracle error mismatch for {suite}/{name}: "
                    f"expected {expected_oracle}, manifest {entry.get('oracle_errors')}, actual {actual_oracle}"
                )
            dataset_count += 1
            oracle_count += actual_oracle

    if dataset_count != EXPECTED_DATASET_COUNT or oracle_count != EXPECTED_ORACLE_ERROR_COUNT:
        raise ValueError(
            f"benchmark coverage mismatch: {dataset_count} datasets / {oracle_count} oracle cells"
        )
    return ManifestAudit(dataset_count, oracle_count, file_count, True, portable)


def _copy_dataset(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir(), key=lambda path: path.name):
        if not item.is_file() or item.suffix.lower() not in _COPY_SUFFIXES:
            continue
        target = destination / item.name
        if item.resolve() != target.resolve():
            shutil.copy2(item, target)


def prepare_data(
    source_root: str | Path,
    data_root: str | Path,
    *,
    strict: bool = True,
) -> dict[str, object]:
    """Copy the fixed CARE snapshot and rebuild its portable manifest."""

    source = Path(source_root)
    destination = Path(data_root)
    unavailable: list[str] = []
    for suite in SUITES:
        for name in suite_datasets(suite):
            source_dataset = source / suite / name
            if not source_dataset.is_dir():
                unavailable.append(f"{suite}/{name}")
                continue
            _copy_dataset(source_dataset, destination / suite / name)
    if strict and unavailable:
        raise FileNotFoundError(f"CARE source is incomplete: {', '.join(unavailable)}")
    if unavailable:
        return {"unavailable": unavailable}
    manifest = write_portable_manifest(destination)
    validate_manifest(destination)
    return manifest


__all__ = [
    "CellId",
    "EXPECTED_DATASET_COUNT",
    "EXPECTED_ORACLE_ERROR_COUNT",
    "EXPECTED_ORACLE_ERRORS",
    "LoadedDataset",
    "ManifestAudit",
    "OracleCell",
    "SOURCE_DATASETS",
    "SUITES",
    "SafeCell",
    "SafeDataset",
    "TABLEEG_DATASETS",
    "append_jsonl",
    "assert_safe_schema",
    "build_portable_manifest",
    "dataset_dir",
    "iter_oracle_cells",
    "load_dataset",
    "load_datasets",
    "load_safe_dataset",
    "normalize_for_match",
    "normalize_value",
    "prepare_data",
    "read_csv",
    "read_jsonl",
    "sha256_file",
    "suite_datasets",
    "validate_manifest",
    "write_jsonl",
    "write_portable_manifest",
]
