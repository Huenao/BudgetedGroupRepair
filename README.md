# Budgeted Group Repair — No-Baran Prompt

这是一个独立 Python 项目，用于研究有限 token/API 预算下的重叠组级数据修复。Baran 可用于 Prompt 外的分组特征、收益估计、验证和 fallback，但 Baran candidate、support 与 diagnostics 不会进入 LLM messages。

v1 初步实验保持冻结；当前新增 Router-v2 正式流程。核心设置：

- 包名：`budgeted_group_repair_no_baran`
- 模型：`deepseek-v4-flash`
- Prompt schema：`bgr-no-baran-v1`
- Group views：`row / pattern / public_fd / semantic`
- Router-v2 group sizes：`1 / 2 / 4 / 8`
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

文档：

- `01_研究背景动机与问题定义.md`
- `02_方法与理论.md`
- `03_实现与实验说明.md`
- `04_实验结果与分析.md`
