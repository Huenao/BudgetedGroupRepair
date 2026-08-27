# Budgeted Group Repair — Router V3 Baseline

这是 Budgeted Group Repair 的 V3-only 基准项目。它在 Prompt 中严格排除 Baran candidate、support 和 diagnostics，同时允许在 Prompt 外使用 Baran 特征、Router uplift 估计、group 选择、验证与 fallback。

当前保留五套正式 V3 revision：

- `router_v3_exact_size_conditioned`：LightGBM/XGBoost，20% 预算，`k=1/2/4/8/all`。
- `router_v3_budget_sweep_exact_size_conditioned`：LightGBM，`k=2/4`，预算为 `1%/5%/10%/20%/50%`。
- `router_v3_catboost_exact_size_conditioned`：CatBoost，20% 预算，`k=1/2/4/8/all`；外部 LightGBM/XGBoost comparison run 可选。
- `router_v3_tabiclv2_k14_exact_size_conditioned`：TabICLv2，20% 预算，`k=1/4`；与冻结的 LightGBM/XGBoost run 对齐比较。
- `router_v3_tabpfn3_k14_exact_size_conditioned`：TabPFN-3，20% 预算，`k=1/4`；与冻结的 LightGBM/XGBoost/TabICLv2 链对齐比较。

正式测试集固定为 9 个数据集、22,198 个错误单元格；训练与 calibration 使用严格 base-family holdout，target 数据的 clean labels 和响应不会在选择前进入 Router。

## 环境与数据检查

项目要求 Python 3.10，依赖版本记录在 `requirements-lock.txt`。安装项目已有依赖后：

```bash
export PYTHONPATH=src
python -m budgeted_group_repair_no_baran.cli validate-data
```

所有 run 写入 `runs/<run_id>/`，默认不会提交到 Git。

## 完整 Router V3 流程

以下示例使用基础 LightGBM/XGBoost 配置。`--baran-source-run` 与 `--response-reuse-run` 均为可选参数：不提供时分别现场运行 Baran、现场执行 LLM；提供时会校验来源 manifest，并只复用 request identity 完全一致的记录。

```bash
RUN_ID=router_v3_baseline

python -m budgeted_group_repair_no_baran.cli plan-router-run \
  --run-id "$RUN_ID"

python -m budgeted_group_repair_no_baran.cli run-router-calibration \
  --run-id "$RUN_ID" --resume --token-cap <token_cap> \
  --env-file <env_file>

python -m budgeted_group_repair_no_baran.cli train-router \
  --run-id "$RUN_ID" --resume

python -m budgeted_group_repair_no_baran.cli run-router-bgr \
  --run-id "$RUN_ID" --resume --token-cap <token_cap> \
  --env-file <env_file>

python -m budgeted_group_repair_no_baran.cli finalize-run \
  --run-id "$RUN_ID" --resume

python -m budgeted_group_repair_no_baran.cli report \
  --run-id "$RUN_ID" --resume
```

如需明确允许无 token 上限运行，可用 `--no-token-cap` 代替 `--token-cap`。CLI 不会读取或输出环境文件中的密钥值。

可选复用示例：

```bash
python -m budgeted_group_repair_no_baran.cli plan-router-run \
  --run-id "$RUN_ID" \
  --baran-source-run runs/<baran_run> \
  --response-reuse-run runs/<response_run>
```

多预算配置还要求通过 `--router-artifact-reuse-run` 指定基础 V3 run。CatBoost 可通过 `--router-comparison-run` 生成与 LightGBM/XGBoost 的额外对齐比较；不提供时 CatBoost 自身训练、测试和报告仍然完整。

TabICLv2/TabPFN-3 使用独立环境和本地 checkpoint；基础环境不会导入这两个可选包。Foundation revision 在任何 provider 调用前必须先执行零调用 dry plan，并将其给出的精确 retry-adjusted cap 用于正式命令：

```bash
python -m budgeted_group_repair_no_baran.cli plan-router-bgr \
  --run-id "$RUN_ID" --resume \
  --experiment-config configs/experiment_router_v3_tabiclv2_k14.json
```

两个 foundation revision 均要求 `--router-comparison-run`；后续实验范围和优先级统一见 [`../BudgetedGroupRepair/03_实现与实验说明.md`](../BudgetedGroupRepair/03_实现与实验说明.md)，原实施指南已归档至 [`../BudgetedGroupRepair_NoBaranPrompt_MarkdownBackup_20260820/09_Router-v3-TabICLv2与TabPFN-3实验实施指南.md`](../BudgetedGroupRepair_NoBaranPrompt_MarkdownBackup_20260820/09_Router-v3-TabICLv2与TabPFN-3实验实施指南.md)。

## 独立运行两条全量 baseline

`run-full-baselines` 不训练 Router，也不执行 group 选择。它只运行或复用正式 9 数据集的：

- Baran-only；
- 纯 No-Baran Singleton LLM-only。

LLM failure、abstain、空 repair 和 unchanged dirty 都按未修复处理，绝不回退到 Baran。

```bash
BASELINE_RUN=full_baselines_v3

python -m budgeted_group_repair_no_baran.cli run-full-baselines \
  --run-id "$BASELINE_RUN" --token-cap <token_cap> \
  --env-file <env_file>
```

该命令支持 `--resume`、`--baran-source-run` 和 `--response-reuse-run`，并生成：

- `runs/<run_id>/final/all_methods.jsonl`：两条完整 baseline cell ledger；
- `runs/baselines/<run_id>/`：可复用的 Baran/LLM baseline bundle；
- `runs/analyses/<run_id>/`：逐 cell 配对、数据集/family/micro/macro 汇总、bootstrap CI 与 McNemar/Holm 检验。

## 离线互补分析

任何已经完成、且包含两条完整 baseline 的 V3 run 都可以离线分析；该命令不调用模型：

```bash
python -m budgeted_group_repair_no_baran.cli analyze-full-complementarity \
  --source-run runs/<completed_v3_run>
```

## Introduction 动机证据实验

该实验使用独立 runner，不实例化 Router、optimizer、verifier 或 Baran fallback。正式 run ID 固定为 `motivation_evidence_deepseek_v4_flash_20260822_full`：

```bash
MOTIVATION_RUN=motivation_evidence_deepseek_v4_flash_20260822_full

python -m budgeted_group_repair_no_baran.cli plan-motivation-evidence \
  --run-id "$MOTIVATION_RUN"

python -m budgeted_group_repair_no_baran.cli run-motivation-queries \
  --run-id "$MOTIVATION_RUN" --resume --no-token-cap \
  --env-file .deepseek_env

python -m budgeted_group_repair_no_baran.cli finalize-motivation-evidence \
  --run-id "$MOTIVATION_RUN" --resume

python -m budgeted_group_repair_no_baran.cli report-motivation-evidence \
  --run-id "$MOTIVATION_RUN" --resume

python -m budgeted_group_repair_no_baran.cli validate-motivation-evidence \
  --run-id "$MOTIVATION_RUN" --resume
```

Plan 阶段现场执行 fresh Baran，但不调用 provider；它冻结 22,198 个 singleton 与 `g>=3` 的 structured/random 分组、交错 schedule 和精确 physical request union。付费阶段会先在 `/tmp` 中执行 excluded-dataset 的 singleton 与 ordered `k=2` pilot，确认返回模型身份、解析和 resume，然后删除 pilot 响应。正式 raw response、checkpoint、usage、paired ledgers、统计表、PDF/SVG 和 Markdown 报告全部保存在唯一的 `runs/<run_id>/` 中。

## 配置

- `configs/experiment_router_v3.json`：基础 LightGBM/XGBoost 20% 配置。
- `configs/experiment_router_v3_budget_sweep_k24_lightgbm.json`：LightGBM k2/k4 多预算配置。
- `configs/experiment_router_v3_catboost.json`：CatBoost 20% 全 k 配置。
- `configs/experiment_router_v3_tabiclv2_k14.json`：TabICLv2 20% k1/k4 配置。
- `configs/experiment_router_v3_tabpfn3_k14.json`：TabPFN-3 20% k1/k4 配置。
- `configs/deepseek_v4.json`：模型、重试、并发和 Prompt schema 配置。
- `configs/motivation_evidence.json`：Introduction 动机证据实验的冻结数据集、分组、随机种子、计数与审计配置。
- `configs/public_fds.json`：公开 FD 规则。

Foundation 模型分别冻结在 `requirements-tabiclv2-lock.txt` 与 `requirements-tabpfn3-lock.txt`；checkpoint 和 `runs/` 始终保持 Git ignored。

## 验证

```bash
export PYTHONPATH=src
python -m pytest
python -m compileall -q src tests
```

`validate-run` 会从冻结 artifact 重新核对 V3 revision、数据/模型/Prompt fingerprint、calibration coverage、family split、selection budget、最终 cell ledger、指标与成本审计。
