# Router-v3：LightGBM k=2/4 多预算实验

## 1. 实验问题

本实验固定 Router 的训练条件，考察 logical LLM budget 从 1% 增长到 50% 时的逐数据集修复效果和成本。只保留 LightGBM，并分别训练或复用两套精确 size-conditioned Router：

| Variant | 允许的 train/test group sizes |
|---|---:|
| `k=2` | `{1,2}` |
| `k=4` | `{1,4}` |

预算点为 `1% / 5% / 10% / 20% / 50%`，基准为每个数据集全部 singleton queries 的 estimated-token 总成本。`baran_labeling_budget=20`、seed 42、`rho=1`、`gamma=1` 和 verifier 参数与原 Router-v3 保持一致。

## 2. 独立 run 与冻结配置

```text
runs/no_baran_router_v3_deepseek_v4_20260726_budget_sweep_k24_lightgbm/
configs/experiment_router_v3_budget_sweep_k24_lightgbm.json
router_revision=router_v3_budget_sweep_exact_size_conditioned
```

新 run 不修改或覆盖原 20% Router-v3。创建 run 时自动复制并校验：

```text
bound_experiment_config.json
bound_llm_config.json
```

Resume 只读取 run-local 冻结快照；外部 config 路径不能改变已有 run 的绑定。

## 3. 模型与选择循环

执行单位为 `target × backend × k`。每个单位只产生一份 prediction CSV、一份 model metadata 和一行 split audit；五个预算只重复选择，不重复拟合或预测：

```text
gates/lightgbm/variant_<k>/<suite>__<dataset>.csv
gates/lightgbm/variant_<k>/<suite>__<dataset>.metadata.json

selections/lightgbm/size_conditioned/variant_<k>/<budget>/<suite>__<dataset>.json
```

五个预算的 selected sets 不要求嵌套。每个选择器单独受对应数据集的 singleton-cost budget 约束，且 `selected_estimated_tokens <= budget_estimated_tokens` 是 hard gate。

## 4. Router-v3 artifact 复用

`--router-artifact-reuse-run` 指向完整的 20% Router-v3。新 run 逐项校验并复制 k=2/k=4 LightGBM 的 18 个 prediction files 和 18 个 metadata files。复用 provenance 写入：

```text
provenance/router_artifact_reuse.json
```

Baran 可现场运行或通过 `--baran-source-run` 严格导入；candidate groups 与 memberships 在当前 run 中确定性生成。Calibration 可重新执行，也可以通过 request-identical response checkpoint 节省物理调用。Response reuse 使用 latest-row 语义，并要求以下 identity 完全一致：

```text
query_id / prompt_hash / provider_request_hash /
model / prompt_schema_version
```

若指定 response reuse run，成功响应和其中已冻结的 terminal provider failures 均可复用。失败响应在新 run 中保持失败，BGR 回退 Baran，LLM-only 仍记为未修复且不回退 Baran。Cache hit 只减少 physical calls，不改变各 slice 的 logical calls/tokens。

20% 的每个 selected-ID 列表必须与父 Router-v3 的 LightGBM k=2/k=4 完全一致。最终 validator 还逐 cell 比较 prediction、correctness、accepted-LLM、selected query 与 final source，从而同时保证 F1 和升级为 LLM 修复的 cell 数一致。

## 5. Dry plan 与外部调用

在任何新模型调用之前，90 个 selection 全部冻结，并生成：

```text
metrics/selection_audit.csv
metrics/logical_budget_ledger.csv
llm/selected_union_plan.json
llm/router_v3_budget_sweep_dry_plan.json
```

Dry plan 记录跨预算去重后的 physical union、父响应 success/failure 复用数、缺失 query 数与 estimated tokens。外部调用会把 dataset-derived dirty-cell Prompt 发送给 DeepSeek API，因此必须在知情后显式授权；`--no-token-cap` 只表示不设 token 上限，不替代数据传输授权。

## 6. 最终矩阵与报告

正式测试仍为 9 个数据集、22,198 个 error cells。最终矩阵为：

- Baran-only；
- LLM-only；
- LightGBM k=2 × 五预算；
- LightGBM k=4 × 五预算。

因此共有 12 个方法切片、108 个 dataset slices、266,376 条 cell records、90 个 selections 和 18 个 split/model folds。

报告以两张逐数据集主表为中心，每张表展示 Baran F1、LLM-only F1/有效 LLM cells，以及五个预算的 BGR F1/升级为 LLM 修复的 cells。详细长表补充 P/R/F1、correct/predicted repairs、相对两个 baseline 的差值和 logical/physical 成本。预算曲线和每个 k 的 F1 AUBC 使用 Baran 作为 β=0 anchor。

每个 BGR slice 分别对 Baran-only 和 LLM-only 运行 dirty-row cluster paired bootstrap：2,000 次、seed 45；Holm 校正在每个 `baseline × k × budget` 的 9 数据集族内执行。Micro、Dataset-Macro、Win/Tie/Loss 与成本汇总放在报告后部。

## 7. 完成门槛

只有以下条件全部通过才可标记 complete：

- 266,376 records 完整无重复；
- 90 selections 全部预算合规；
- 18 split/model folds 完整，family/cell/row/query/group-signature leakage 全为零；
- 20% selected IDs、逐 cell 结果、F1 和 LLM-upgraded cells 与父 run 完全一致；
- LLM-only 无 Baran fallback；
- response/artifact provenance 和 logical/physical 去重可独立重算；
- 108 个逐数据集方法指标、132 个含 micro/macro 指标、180 个 paired-stat rows 完整；
- pytest、compileall、数据校验、finalize、strict validation 与 Markdown/HTML/artifact validation 全部通过。
