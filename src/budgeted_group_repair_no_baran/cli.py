"""Command line interface for the standalone no-Baran Prompt project."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROUTER_CONFIG = PROJECT_ROOT / "configs" / "experiment_router_v3.json"
DEFAULT_ROUTER_LLM_CONFIG = PROJECT_ROOT / "configs" / "deepseek_v4.json"
DEFAULT_MOTIVATION_CONFIG = PROJECT_ROOT / "configs" / "motivation_evidence.json"
DEFAULT_MOTIVATION_LLM_CONFIG = PROJECT_ROOT / "configs" / "deepseek_v4.json"

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "bearer",
    "secret",
    "password",
    "cookie",
    "authorization",
)


class EnvFileError(ValueError):
    """Raised for malformed dotenv input without exposing any value."""


def _parse_env_value(raw: str, *, path: Path, line_number: int) -> str:
    try:
        lexer = shlex.shlex(raw, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
    except ValueError as error:
        raise EnvFileError(
            f"malformed quoted value in environment file {path} at line {line_number}"
        ) from error
    if not tokens:
        return ""
    if len(tokens) != 1:
        raise EnvFileError(
            f"unquoted whitespace in environment file {path} at line {line_number}"
        )
    value = tokens[0]
    if any(character in value for character in ("\x00", "\n", "\r")):
        raise EnvFileError(
            f"invalid control character in environment file {path} at line {line_number}"
        )
    return value


def load_env_file(
    path: str | Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Load literal ``NAME=value`` records; values are never printed or evaluated."""

    source = Path(path).expanduser().resolve()
    target = os.environ if environ is None else environ
    try:
        contents = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise EnvFileError(f"cannot read environment file: {source}") from error
    if "\x00" in contents:
        raise EnvFileError(f"environment file contains a NUL byte: {source}")
    loaded: list[str] = []
    for line_number, original in enumerate(contents.splitlines(), start=1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvFileError(
                f"missing '=' in environment file {source} at line {line_number}"
            )
        name, raw = line.split("=", 1)
        name = name.strip()
        if not _ENV_NAME.fullmatch(name):
            raise EnvFileError(
                f"invalid variable name in environment file {source} at line {line_number}"
            )
        value = _parse_env_value(raw.strip(), path=source, line_number=line_number)
        if name not in target:
            target[name] = value
            loaded.append(name)
    return tuple(loaded)


def _run_id(value: str) -> str:
    if not _RUN_ID.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "run ID must be 1-128 characters using letters, digits, '.', '_' or '-'"
        )
    return value


def _add_run(parser: argparse.ArgumentParser, *, paid: bool = False) -> None:
    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--resume", action="store_true")
    if paid:
        budget = parser.add_mutually_exclusive_group(required=True)
        budget.add_argument(
            "--token-cap",
            type=int,
            help="hard conservative cap over observed plus reserved provider tokens",
        )
        budget.add_argument(
            "--no-token-cap",
            action="store_true",
            help="explicitly authorize uncapped provider usage for this run",
        )
        parser.add_argument(
            "--env-file",
            type=Path,
            help="literal dotenv file; only variable names are ever reported",
        )


def _add_router_run(parser: argparse.ArgumentParser, *, paid: bool = False) -> None:
    _add_run(parser, paid=paid)
    parser.add_argument(
        "--baran-source-run",
        type=Path,
        help="optional completed run supplying a verified Baran ledger",
    )
    parser.add_argument(
        "--response-reuse-run",
        type=Path,
        help="optional run supplying request-identical No-Baran responses",
    )
    parser.add_argument(
        "--calibration-source-run",
        type=Path,
        help="completed run supplying frozen calibration queries, executions, and labels",
    )
    parser.add_argument(
        "--router-artifact-reuse-run",
        type=Path,
        help="completed Router-v3 run supplying request-identical gate artifacts",
    )
    parser.add_argument(
        "--router-comparison-run",
        type=Path,
        help="optional base V3 run used only for cross-backend reporting",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_ROUTER_CONFIG,
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_ROUTER_LLM_CONFIG,
    )


def _add_motivation_run(parser: argparse.ArgumentParser, *, paid: bool = False) -> None:
    """Evidence commands intentionally expose no historical reuse inputs."""

    parser.add_argument("--run-id", type=_run_id, required=True)
    parser.add_argument("--resume", action="store_true")
    if paid:
        budget = parser.add_mutually_exclusive_group(required=True)
        budget.add_argument(
            "--token-cap",
            type=int,
            help="hard conservative cap over observed plus reserved provider tokens",
        )
        budget.add_argument(
            "--no-token-cap",
            action="store_true",
            help="explicitly authorize uncapped provider usage for this run",
        )
        parser.add_argument(
            "--env-file",
            type=Path,
            help="literal dotenv file; only variable names are ever reported",
        )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_MOTIVATION_CONFIG,
    )
    parser.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_MOTIVATION_LLM_CONFIG,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="budgeted-group-repair-no-baran",
        description="Run the Router-v3 and full No-Baran baseline workflows.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_data = commands.add_parser("validate-data")
    validate_data.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    validate_data.add_argument("--manifest", type=Path)

    complementarity = commands.add_parser(
        "analyze-full-complementarity",
        help="analyze complete Baran/LLM baseline slices without model calls",
    )
    complementarity.add_argument("--source-run", type=Path, required=True)
    complementarity.add_argument("--baseline-dir", type=Path)
    complementarity.add_argument("--output-dir", type=Path)
    complementarity.add_argument("--bootstrap-replicates", type=int, default=2_000)
    complementarity.add_argument("--bootstrap-seed", type=int, default=45)
    complementarity.add_argument("--confidence-level", type=float, default=0.95)

    for name, paid in (
        ("plan-router-run", False),
        ("run-router-calibration", True),
        ("train-router", False),
        ("plan-router-bgr", False),
        ("run-router-bgr", True),
        ("check-model", True),
    ):
        command = commands.add_parser(name)
        _add_router_run(command, paid=paid)

    baselines = commands.add_parser(
        "run-full-baselines",
        help="run/resume all formal Baran-only and singleton LLM-only cells",
    )
    _add_router_run(baselines, paid=True)
    baselines.add_argument("--baseline-dir", type=Path)
    baselines.add_argument("--output-dir", type=Path)
    baselines.add_argument("--bootstrap-replicates", type=int, default=2_000)
    baselines.add_argument("--bootstrap-seed", type=int, default=45)
    baselines.add_argument("--confidence-level", type=float, default=0.95)

    finalize = commands.add_parser("finalize-run")
    _add_run(finalize)

    validate = commands.add_parser("validate-run")
    _add_run(validate)
    validate.add_argument("--allow-incomplete", action="store_true")

    report = commands.add_parser("report")
    _add_run(report)
    report.add_argument("--output", type=Path)

    motivation_plan = commands.add_parser(
        "plan-motivation-evidence",
        help="fresh-Baran, zero-provider planning for the Introduction evidence run",
    )
    _add_motivation_run(motivation_plan)
    motivation_plan.add_argument("--mode", choices=("full",), default="full")

    motivation_run = commands.add_parser(
        "run-motivation-queries",
        help="execute/resume the frozen interleaved physical request schedule",
    )
    _add_motivation_run(motivation_run, paid=True)

    motivation_finalize = commands.add_parser("finalize-motivation-evidence")
    _add_motivation_run(motivation_finalize)

    motivation_validate = commands.add_parser("validate-motivation-evidence")
    _add_motivation_run(motivation_validate)
    motivation_validate.add_argument("--allow-incomplete", action="store_true")
    motivation_validate.add_argument("--allow-unfinalized", action="store_true")

    motivation_report = commands.add_parser("report-motivation-evidence")
    _add_motivation_run(motivation_report)
    return parser




def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if getattr(args, "token_cap", None) is not None and int(args.token_cap) <= 0:
        raise SystemExit("--token-cap must be positive")
    if (
        getattr(args, "bootstrap_replicates", None) is not None
        and int(args.bootstrap_replicates) <= 0
    ):
        raise SystemExit("--bootstrap-replicates must be positive")
    if getattr(args, "confidence_level", None) is not None and not (
        0.0 < float(args.confidence_level) < 1.0
    ):
        raise SystemExit("--confidence-level must be between zero and one")
    return args




def _router_runner(args: argparse.Namespace) -> Any:
    from .router_v3 import ExperimentRunner

    return ExperimentRunner.create(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        config_path=Path(args.experiment_config),
        llm_config_path=Path(args.llm_config),
        vendor_root=PROJECT_ROOT / "vendor" / "raha_source",
        runs_root=PROJECT_ROOT / "runs",
        run_id=args.run_id,
        resume=bool(args.resume),
        baran_source_run=(
            Path(args.baran_source_run)
            if args.baran_source_run is not None
            else None
        ),
        response_reuse_run=(
            Path(args.response_reuse_run)
            if args.response_reuse_run is not None
            else None
        ),
        calibration_source_run=(
            Path(args.calibration_source_run)
            if args.calibration_source_run is not None
            else None
        ),
        router_artifact_reuse_run=(
            Path(args.router_artifact_reuse_run)
            if args.router_artifact_reuse_run is not None
            else None
        ),
        router_comparison_run=(
            Path(args.router_comparison_run)
            if args.router_comparison_run is not None
            else None
        ),
        provider_token_cap=getattr(args, "token_cap", None),
        allow_uncapped_provider_usage=bool(
            getattr(args, "no_token_cap", False)
        ),
    )


def _motivation_runner(args: argparse.Namespace) -> Any:
    from .motivation_evidence import DEFAULT_RUN_ID, MotivationEvidenceRunner

    if str(args.run_id) != DEFAULT_RUN_ID:
        raise ValueError(
            f"the formal motivation evidence run ID is frozen as {DEFAULT_RUN_ID!r}"
        )
    bound_root = PROJECT_ROOT / "runs" / args.run_id / "configs"
    bound_experiment = bound_root / "motivation_evidence.json"
    bound_llm = bound_root / "deepseek_v4.json"
    bound_presence = (bound_experiment.is_file(), bound_llm.is_file())
    if any(bound_presence) and not all(bound_presence):
        raise FileNotFoundError(
            "same-run motivation resume has an incomplete bound configuration pair"
        )
    # Once a run has bound both configurations, every same-run command reads
    # those copies.  Project defaults may evolve without changing a frozen
    # plan or making a later execute/finalize/validate command appear to drift.
    experiment_config = (
        bound_experiment if all(bound_presence) else Path(args.experiment_config)
    )
    llm_config = bound_llm if all(bound_presence) else Path(args.llm_config)
    return MotivationEvidenceRunner.create(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        vendor_root=PROJECT_ROOT / "vendor" / "raha_source",
        runs_root=PROJECT_ROOT / "runs",
        run_id=args.run_id,
        config_path=experiment_config,
        llm_config_path=llm_config,
        resume=bool(args.resume),
        provider_token_cap=getattr(args, "token_cap", None),
        allow_uncapped_provider_usage=bool(getattr(args, "no_token_cap", False)),
    )




def _redact(value: object, parent_key: str = "") -> object:
    if any(part in parent_key.lower() for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if is_dataclass(value) and not isinstance(value, type):
        return _redact(asdict(value), parent_key)
    if hasattr(value, "as_dict") and callable(getattr(value, "as_dict")):
        return _redact(value.as_dict(), parent_key)
    if isinstance(value, Mapping):
        return {str(key): _redact(item, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item, parent_key) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _execute(args: argparse.Namespace) -> object:
    if args.command == "validate-data":
        from .data import validate_manifest

        root = Path(args.data_root).expanduser().resolve()
        manifest = (
            Path(args.manifest).expanduser().resolve()
            if args.manifest is not None
            else root / "manifest.json"
        )
        return validate_manifest(root, manifest)

    if args.command == "analyze-full-complementarity":
        from .full_complementarity import build_full_complementarity

        baseline_dir = (
            Path(args.baseline_dir)
            if args.baseline_dir is not None
            else PROJECT_ROOT / "runs" / "baselines" / Path(args.source_run).name
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else PROJECT_ROOT / "runs" / "analyses" / Path(args.source_run).name
        )
        return build_full_complementarity(
            args.source_run,
            baseline_dir=baseline_dir,
            output_dir=output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence_level,
        )

    if getattr(args, "env_file", None) is not None:
        load_env_file(args.env_file)
    if args.command == "validate-run":
        from .router_v3 import validate_run

        return validate_run(
            PROJECT_ROOT / "runs" / args.run_id,
            require_complete=not args.allow_incomplete,
        )
    if args.command == "finalize-run":
        from .router_v3 import finalize_existing_run

        return finalize_existing_run(PROJECT_ROOT / "runs" / args.run_id)
    if args.command == "report":
        from .router_reporting_v3 import build_router_v3_report

        return build_router_v3_report(
            PROJECT_ROOT / "runs" / args.run_id,
            output_path=args.output,
        )

    if args.command.startswith("plan-motivation-") or args.command in {
        "run-motivation-queries",
        "finalize-motivation-evidence",
        "validate-motivation-evidence",
        "report-motivation-evidence",
    }:
        runner = _motivation_runner(args)
        if args.command == "plan-motivation-evidence":
            return runner.plan()
        if args.command == "run-motivation-queries":
            return runner.run_queries()
        if args.command == "finalize-motivation-evidence":
            return runner.finalize()
        if args.command == "validate-motivation-evidence":
            return runner.validate(
                require_execution=not bool(args.allow_incomplete),
                require_finalized=(
                    not bool(args.allow_incomplete)
                    and not bool(args.allow_unfinalized)
                ),
            )
        if args.command == "report-motivation-evidence":
            return runner.report_results()
        raise AssertionError(f"unhandled motivation command: {args.command}")

    runner = _router_runner(args)
    if args.command == "plan-router-run":
        return runner.plan_run()
    if args.command == "check-model":
        return runner.check_model()
    if args.command == "run-router-calibration":
        if not runner.state.stage_completed("calibration_plan"):
            runner.plan_run()
        if not runner.state.stage_completed("model_preflight"):
            runner.check_model()
        return runner.run_calibration_stage()
    if args.command == "train-router":
        return runner.train_and_select_stage()
    if args.command == "plan-router-bgr":
        return runner.plan_selected_llm_stage()
    if args.command == "run-router-bgr":
        if runner.is_router_v3_foundation:
            runner.plan_selected_llm_stage()
            if not runner.state.stage_completed("model_preflight"):
                if not runner.reuse_model_preflight_stage():
                    runner.check_model()
        selected = runner.run_selected_llm_stage()
        final = runner.build_final_records_stage()
        metrics = runner.build_metrics_stage()
        audit = runner.build_audit_stage()
        return {
            "selected_llm": selected,
            "final": final,
            "metrics": metrics,
            "audit": audit,
        }
    if args.command == "run-full-baselines":
        baseline_dir = (
            Path(args.baseline_dir)
            if args.baseline_dir is not None
            else PROJECT_ROOT / "runs" / "baselines" / args.run_id
        )
        output_dir = (
            Path(args.output_dir)
            if args.output_dir is not None
            else PROJECT_ROOT / "runs" / "analyses" / args.run_id
        )
        return runner.run_full_baselines(
            baseline_dir=baseline_dir,
            output_dir=output_dir,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence_level,
        )
    raise AssertionError(f"unhandled command: {args.command}")




def main(argv: Sequence[str] | None = None) -> int:
    result = _execute(parse_args(argv))
    print(json.dumps(_redact(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = ["EnvFileError", "build_parser", "load_env_file", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(main())
