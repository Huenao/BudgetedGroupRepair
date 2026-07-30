# Budgeted Group Repair — No-Baran Prompt

这是一个独立 Python 项目，用于研究有限 token/API 预算下的重叠组级数据修复。Baran 可用于 Prompt 外的分组特征、收益估计、验证和 fallback，但 Baran candidate、support 与 diagnostics 不会进入 LLM messages。

v1 初步实验和 Router-v2 正式结果保持冻结；Router-v3 新增按 group size 独立训练的 20% 预算流程。核心设置：

- 包名：`budgeted_group_repair_no_baran`
- 模型：`deepseek-v4-flash`
- Prompt schema：`bgr-no-baran-v1`
- Group views：`row / pattern / public_fd / semantic`
- Router-v2 group sizes：`1 / 2 / 4 / 8`
- Router-v3 variants：`k=1 → {1}`、`k=2 → {1,2}`、`k=4 → {1,4}`、`k=8 → {1,8}`、`all → {1,2,4,8}`
- 数据：14-dataset 本地快照；正式实验选择 9 个数据集
- Router calibration：全部 9 个 TableEG singleton（5,543）加每数据集最多 300 个分层 non-singleton
- 正式测试：指定 9 个数据集的全部 22,198 cells

Router-v2 中实验一、二和 routeability 只作为诊断；数据泄漏、预算、模型身份、Prompt 信息边界、request identity、run fingerprint 和最终覆盖仍是 hard gates。

## 快速验证

```bash
export PYTHONPATH=src
../BudgetedGroupRepairProject/.venv/bin/python -m pytest -q
../BudgetedGroupRepairProject/.venv/bin/python -m compileall -q src tests
../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli validate-data
```

## Dry run

```bash
export PYTHONPATH=src
../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  plan-run \
  --run-id <new_v1_run_id> \
  --source-run ../BudgetedGroupRepairProject/runs/bgr_deepseek_v4_20260720_final_v4_cap80m
```

`plan-run` 不调用付费 API，会完成数据/Baran 校验、抽样、全部 action、structured/random partition、Prompt 泄漏审计和 token 估算。

## 正式实验

```bash
../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  check-model --run-id <run_id> --resume --no-token-cap --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-experiment1 --run-id <run_id> --resume --no-token-cap --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-experiment2 --run-id <run_id> --resume --no-token-cap --env-file ../.deepseek_env
```

密钥只通过环境变量 `DEEPSEEK_API_KEY` 使用，不得打印或写入 artifact。

## 冻结 v1 结果

正式 run：`runs/no_baran_deepseek_v4_20260724_final/`

- 实验一：Baran 67.52%，singleton LLM 40.56%，Oracle UB 77.63%，互补性通过；
- 实验二：覆盖 2,280 cells / 570 个 k=4 groups；structured−singleton macro 为 −0.39 pp，95% CI [−1.47, +0.63] pp；macro token/cell 节省 27.46%；
- 总体判定：`C_not_supported`，因此 Phase 3 被门禁禁止。

## Router-v2 正式实验

```bash
export PYTHONPATH=src
RUN_ID=no_baran_router_v2_deepseek_v4_20260725_full9

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  plan-router-run --run-id "$RUN_ID" \
  --source-run ../BudgetedGroupRepairProject/runs/bgr_deepseek_v4_20260720_final_v4_cap80m

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-calibration --run-id "$RUN_ID" --resume --no-token-cap \
  --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  train-router --run-id "$RUN_ID" --resume

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-bgr --router-v2 --run-id "$RUN_ID" --resume --no-token-cap \
  --env-file ../.deepseek_env
```

密钥只由 env loader 注入客户端，不会打印、复制或写入 artifact。

## Router-v3：20% 预算、按 size 独立训练

固定 run：`runs/no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all/`。它复用 Router-v2 的候选、8,197 条 calibration queries、16,451 条 pair labels 和 strict request identity 完全一致的成功响应，但不覆盖父 run。物理 cache hit 不减少 logical budget。

```bash
export PYTHONPATH=src
RUN_ID=no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all
PARENT_RUN=runs/no_baran_router_v2_deepseek_v4_20260725_full9

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  plan-router-run --run-id "$RUN_ID" \
  --experiment-config configs/experiment_router_v3.json \
  --response-reuse-run "$PARENT_RUN" \
  --source-run ../BudgetedGroupRepairProject/runs/bgr_deepseek_v4_20260720_final_v4_cap80m

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-calibration --run-id "$RUN_ID" --resume --no-token-cap \
  --experiment-config configs/experiment_router_v3.json \
  --response-reuse-run "$PARENT_RUN" --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  train-router --run-id "$RUN_ID" --resume \
  --experiment-config configs/experiment_router_v3.json \
  --response-reuse-run "$PARENT_RUN"

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-bgr --run-id "$RUN_ID" --resume --no-token-cap \
  --experiment-config configs/experiment_router_v3.json \
  --response-reuse-run "$PARENT_RUN" --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  finalize-run --run-id "$RUN_ID" --require-router

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  report --run-id "$RUN_ID" --require-router
```

最终 ledger 固定为 2 个 baseline 加 2 个 backend × 5 个 size variant，共 12 个完整方法切片、266,376 条 cell records 和 90 个 selection slices。报告主表为 9 个数据集的 F1 矩阵；详细长表同时保留修复数、P/R/F1、相对两个 baseline 的差值与逻辑/物理成本。

## 完整 Baran / No-Baran Singleton 基线与互补性

Router-v3 固定 run 已包含正式九数据集全部 22,198 cells 的 Baran-only 和纯 LLM-only 结果。纯 LLM-only 切片不会在失败、abstain 或 unchanged dirty 时回退 Baran。以下命令只读取冻结运行，不调用模型，也不需要 API key：

```bash
export PYTHONPATH=src
../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  analyze-full-complementarity \
  --source-run runs/no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all
```

派生产物保存在：

- `runs/baselines/no_baran_singleton_deepseek_v4_full9/`：带来源哈希的 manifest、两个完整 baseline JSONL 和离线 Router labels；
- `runs/analyses/baran_llm_complementarity_full9/`：逐 cell 配对、分数据集/family/micro/macro 统计、bootstrap CI、McNemar/Holm 检验和独立 Markdown 报告。

未来 singleton-only Router 应显式使用上述固定 Router-v3 run 作为 `--response-reuse-run`，并在新 protocol revision 中冻结 success 与 terminal failure。`singleton_router_labels.csv` 含 clean-label 派生字段，只能进入离线训练标签，不能作为在线特征。

## Router-v3：LightGBM k=2/4 多预算

独立 run：`runs/no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm/`。配置为 `configs/experiment_router_v3_budget_sweep_k24_lightgbm.json`；预算点为 `1% / 5% / 10% / 20% / 50%`。每个 target × k 只复用或建立一套 LightGBM 预测，五个预算分别运行选择器。

```bash
export PYTHONPATH=src
RUN_ID=no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm
PARENT_RUN=runs/no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  plan-router-run --run-id "$RUN_ID" \
  --experiment-config configs/experiment_router_v3_budget_sweep_k24_lightgbm.json \
  --response-reuse-run "$PARENT_RUN" \
  --router-artifact-reuse-run "$PARENT_RUN"

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  train-router --run-id "$RUN_ID" --resume \
  --response-reuse-run "$PARENT_RUN" \
  --router-artifact-reuse-run "$PARENT_RUN"

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-calibration --run-id "$RUN_ID" --resume --no-token-cap \
  --response-reuse-run "$PARENT_RUN" \
  --router-artifact-reuse-run "$PARENT_RUN" --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-bgr --run-id "$RUN_ID" --resume --no-token-cap \
  --response-reuse-run "$PARENT_RUN" \
  --router-artifact-reuse-run "$PARENT_RUN" --env-file ../.deepseek_env
```

外部调用会发送 dataset-derived dirty-cell Prompt；执行者必须在知情后明确授权该数据传输。run 创建时自动保存 `bound_experiment_config.json` 和 `bound_llm_config.json`，resume 只使用这两个冻结快照。

## Router-v3：CatBoost 20% 全 k

独立 run：`runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/`。配置为 `configs/experiment_router_v3_catboost.json`；只替换 Router backend，使用 CatBoost 原生类别特征，预算、特征、标签、划分、优化器和 Verifier 均与 Router-v3 20% 实验保持一致。

```bash
export PYTHONPATH=src
RUN_ID=no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost
RESPONSE_PARENT=runs/no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  plan-router-run --run-id "$RUN_ID" \
  --experiment-config configs/experiment_router_v3_catboost.json \
  --response-reuse-run "$RESPONSE_PARENT"

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-calibration --run-id "$RUN_ID" --resume --no-token-cap \
  --response-reuse-run "$RESPONSE_PARENT" --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  train-router --run-id "$RUN_ID" --resume \
  --response-reuse-run "$RESPONSE_PARENT"

# 先检查 llm/router_v3_catboost_dry_plan.json，并取得针对精确缺失数量的授权。
../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  run-router-bgr --run-id "$RUN_ID" --resume --no-token-cap \
  --response-reuse-run "$RESPONSE_PARENT" --env-file ../.deepseek_env

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  finalize-run --run-id "$RUN_ID" --require-router

../BudgetedGroupRepairProject/.venv/bin/python -m budgeted_group_repair_no_baran.cli \
  report --run-id "$RUN_ID" --require-router
```

该 run 已完成：7 个方法切片、155,386 条唯一 cell records、45 个 selection、45 个模型 fold、180 条 paired-stat rows；完整结果见 `08_Router-v3-CatBoost-20pct全k实验.md` 及 run 内的 Markdown/HTML/artifact 报告。

文档：

- `01_研究背景动机与问题定义.md`
- `02_方法与理论.md`
- `03_实现与实验说明.md`
- `04_实验结果与分析.md`
- `05_后续消融与Router增强实验计划.md`
- `06_Router-v3按组大小独立训练实验.md`
- `07_Router-v3-LightGBM-k24多预算实验.md`
- `08_Router-v3-CatBoost-20pct全k实验.md`
