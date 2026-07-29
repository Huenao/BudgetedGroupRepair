from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from budgeted_group_repair_no_baran.cli import EnvFileError, load_env_file, parse_args
from budgeted_group_repair_no_baran.group_gate import (
    GroupUpliftGate,
    build_uplift_targets,
    executable_use_llm,
)
from budgeted_group_repair_no_baran.group_objective import GroupUpliftObjective
from budgeted_group_repair_no_baran.group_optimizer import exhaustive_optimum, lazy_gain_cost_greedy
from budgeted_group_repair_no_baran.pipeline import plan_bgr, run_routeability
from budgeted_group_repair_no_baran.run_state import write_json


def test_gate_labels_only_executable_proposals() -> None:
    targets = build_uplift_targets(
        [False, True, False], [True, False, True], [True, True, False]
    )
    assert targets.as_dict() == {"helpful": [1, 0, 0], "harmful": [0, 1, 0]}
    assert executable_use_llm({"decision": "propose", "repair": "x"})
    assert not executable_use_llm({"decision": "abstain", "repair": "x"})
    assert not executable_use_llm(
        {"decision": "propose", "repair": "x", "parse_status": "malformed"}
    )
    with pytest.raises(ValueError, match="cannot be gate features"):
        GroupUpliftGate("lightgbm").fit(
            [{"clean_value": "poison"}], [False], [False], [False], ["family"]
        )


def test_optimizer_matches_exact_small_solution_and_respects_budget() -> None:
    objective = GroupUpliftObjective(
        {"q1": {"c1": 0.4, "c2": 0.2}, "q2": {"c2": 0.5, "c3": 0.1}}
    )
    greedy = lazy_gain_cost_greedy(objective, {"q1": 2.0, "q2": 2.0}, 2.0)
    exact = exhaustive_optimum(objective, {"q1": 2.0, "q2": 2.0}, 2.0)
    assert greedy.selected_query_ids == exact.selected_query_ids == ("q1",)
    assert greedy.total_cost <= 2.0


def test_phase_25_and_phase_3_are_hard_gated(tmp_path) -> None:
    metrics = tmp_path / "metrics"
    write_json(
        metrics / "decision_gates.json",
        {
            "complementarity_supported": False,
            "grouping_supported": True,
            "routeability_supported": False,
            "phase3_allowed": False,
        },
    )
    runner = SimpleNamespace(
        paths=SimpleNamespace(metrics=metrics), assert_binding_current=lambda: None
    )
    with pytest.raises(RuntimeError, match="gated"):
        run_routeability(runner)
    with pytest.raises(RuntimeError, match="blocked"):
        plan_bgr(runner)


def test_cli_requires_paid_cap_and_env_parser_never_interpolates(tmp_path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["check-model", "--run-id", "r1"])
    args = parse_args(["run-experiment1", "--run-id", "r1", "--token-cap", "1000"])
    assert args.token_cap == 1000
    args = parse_args(["run-experiment1", "--run-id", "r1", "--no-token-cap"])
    assert args.token_cap is None and args.no_token_cap is True
    plan = parse_args(
        [
            "plan-run",
            "--run-id",
            "v1",
            "--experiment-config",
            "configs/experiment.json",
        ]
    )
    assert plan.experiment_config == Path("configs/experiment.json")
    router = parse_args(
        [
            "plan-router-run",
            "--run-id",
            "router-sweep",
            "--router-artifact-reuse-run",
            "runs/router-v3-parent",
        ]
    )
    assert router.router_artifact_reuse_run == Path("runs/router-v3-parent")
    env = tmp_path / "safe.env"
    env.write_text("DEEPSEEK_API_KEY='literal-$VALUE'\n", encoding="utf-8")
    target: dict[str, str] = {}
    assert load_env_file(env, environ=target) == ("DEEPSEEK_API_KEY",)
    assert target["DEEPSEEK_API_KEY"] == "literal-$VALUE"
    bad = tmp_path / "bad.env"
    bad.write_text("not an assignment\n", encoding="utf-8")
    with pytest.raises(EnvFileError):
        load_env_file(bad, environ={})
