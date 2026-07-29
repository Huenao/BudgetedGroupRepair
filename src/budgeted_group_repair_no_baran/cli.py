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
DEFAULT_SOURCE_RUN = (
    PROJECT_ROOT.parent
    / "BudgetedGroupRepairProject"
    / "runs"
    / "bgr_deepseek_v4_20260720_final_v4_cap80m"
)
DEFAULT_RESPONSE_REUSE_RUN = (
    PROJECT_ROOT / "runs" / "no_baran_deepseek_v4_20260724_final"
)
DEFAULT_ROUTER_CONFIG = PROJECT_ROOT / "configs" / "experiment_router_v2.json"
DEFAULT_ROUTER_LLM_CONFIG = PROJECT_ROOT / "configs" / "deepseek_v4.json"

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
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
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
        "--response-reuse-run",
        type=Path,
        default=DEFAULT_RESPONSE_REUSE_RUN,
    )
    parser.add_argument(
        "--router-artifact-reuse-run",
        type=Path,
        help="completed Router-v3 run supplying request-identical gate artifacts",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="budgeted-group-repair-no-baran",
        description="Run the standalone no-Baran Prompt experiments and gated BGR pipeline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_data = commands.add_parser("validate-data")
    validate_data.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    validate_data.add_argument("--manifest", type=Path)

    plan = commands.add_parser("plan-run")
    _add_run(plan)
    plan.add_argument(
        "--experiment-config",
        type=Path,
        help="experiment config to freeze into a new run; resume uses the run-local copy",
    )

    router_plan = commands.add_parser("plan-router-run")
    _add_router_run(router_plan)

    router_calibration = commands.add_parser("run-router-calibration")
    _add_router_run(router_calibration, paid=True)

    router_train = commands.add_parser("train-router")
    _add_router_run(router_train)

    router_bgr = commands.add_parser("run-router-bgr")
    _add_router_run(router_bgr, paid=True)

    for name in ("check-model", "run-experiment1", "run-experiment2"):
        command = commands.add_parser(name)
        _add_run(command, paid=True)

    routeability = commands.add_parser("run-routeability")
    _add_run(routeability)

    bgr = commands.add_parser("run-bgr")
    _add_run(bgr, paid=True)
    bgr.add_argument("--router-v2", action="store_true")
    bgr.add_argument(
        "--response-reuse-run",
        type=Path,
        default=DEFAULT_RESPONSE_REUSE_RUN,
    )
    bgr.add_argument(
        "--experiment-config",
        type=Path,
        default=DEFAULT_ROUTER_CONFIG,
    )
    bgr.add_argument(
        "--llm-config",
        type=Path,
        default=DEFAULT_ROUTER_LLM_CONFIG,
    )

    for name in ("finalize-run", "validate-run", "report"):
        command = commands.add_parser(name)
        _add_run(command)
        if name == "validate-run":
            command.add_argument("--require-experiments", action="store_true")
            command.add_argument("--require-router", action="store_true")
        if name == "finalize-run":
            command.add_argument("--require-router", action="store_true")
        if name == "report":
            command.add_argument("--output", type=Path)
            command.add_argument("--artifact-only", action="store_true")
            command.add_argument("--require-router", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if getattr(args, "token_cap", None) is not None and int(args.token_cap) <= 0:
        raise SystemExit("--token-cap must be positive")
    return args


def _runner(args: argparse.Namespace) -> Any:
    from .experiment import ExperimentRunner

    return ExperimentRunner(
        project_root=PROJECT_ROOT,
        run_id=args.run_id,
        source_run=Path(args.source_run),
        resume=bool(args.resume),
        experiment_config=getattr(args, "experiment_config", None),
    )


def _router_runner(args: argparse.Namespace) -> Any:
    from .router_v2 import ExperimentRunner

    return ExperimentRunner.create(
        project_root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / "data",
        config_path=Path(args.experiment_config),
        llm_config_path=Path(args.llm_config),
        vendor_root=PROJECT_ROOT / "vendor" / "raha_source",
        runs_root=PROJECT_ROOT / "runs",
        run_id=args.run_id,
        resume=bool(args.resume),
        baran_source_run=Path(args.source_run),
        response_reuse_run=Path(args.response_reuse_run),
        router_artifact_reuse_run=(
            Path(args.router_artifact_reuse_run)
            if args.router_artifact_reuse_run is not None
            else None
        ),
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

    if getattr(args, "env_file", None) is not None:
        load_env_file(args.env_file)
    if args.command in {
        "plan-router-run",
        "run-router-calibration",
        "train-router",
        "run-router-bgr",
    } or (args.command == "run-bgr" and args.router_v2):
        runner = _router_runner(args)
        if args.command == "plan-router-run":
            return runner.plan_run()
        if args.command == "run-router-calibration":
            if not runner.state.stage_completed("calibration_plan"):
                runner.plan_run()
            if not runner.state.stage_completed("model_preflight"):
                runner.check_model()
            return runner.run_calibration_stage()
        if args.command == "train-router":
            return runner.train_and_select_stage()
        if args.command in {"run-router-bgr", "run-bgr"}:
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

    if args.command == "validate-run" and args.require_router:
        from .router_v2 import validate_run

        return validate_run(
            PROJECT_ROOT / "runs" / args.run_id,
            require_complete=False,
        )
    if args.command == "finalize-run" and args.require_router:
        from .router_v2 import finalize_existing_run

        return finalize_existing_run(PROJECT_ROOT / "runs" / args.run_id)
    if args.command == "report" and args.require_router:
        from .router_reporting import build_router_report

        return build_router_report(
            PROJECT_ROOT / "runs" / args.run_id,
            output_path=args.output,
        )

    runner = _runner(args)
    if args.command == "plan-run":
        return runner.plan_run()
    if args.command == "check-model":
        return runner.check_model(args.token_cap)
    if args.command == "run-experiment1":
        return runner.run_experiment1(args.token_cap)
    if args.command == "run-experiment2":
        return runner.run_experiment2(args.token_cap)
    if args.command == "run-routeability":
        from .pipeline import run_routeability

        return run_routeability(runner)
    if args.command == "run-bgr":
        from .pipeline import plan_bgr, run_bgr

        if not runner._state().stage_completed("bgr_plan"):
            plan_bgr(runner)
        return run_bgr(runner, args.token_cap)
    if args.command == "validate-run":
        return runner.validate_run(require_experiments=args.require_experiments)
    if args.command == "finalize-run":
        return runner.finalize_run()
    if args.command == "report":
        from .reporting import build_report

        return build_report(
            runner.paths.run_dir,
            output_path=args.output,
            deliver=not args.artifact_only,
        )
    raise AssertionError(f"unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    result = _execute(parse_args(argv))
    print(json.dumps(_redact(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


__all__ = ["EnvFileError", "build_parser", "load_env_file", "main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover - exercised through CLI smoke tests
    raise SystemExit(main())
