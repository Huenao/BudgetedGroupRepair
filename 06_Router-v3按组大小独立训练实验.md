# Router-v3：按 Group Size 独立训练的 20% 预算实验

## 1. 实验目的

Router-v2 在全部 `1/2/4/8` group-size calibration pairs 上训练一个混合 Router，再在选择阶段限制候选 group size。这会把“训练条件”和“可选 action 条件”混在一起：即使 `k=1` 只选择 singleton，其分数仍来自见过 non-singleton 标签的混合模型。

Router-v3 将 group size 视为实验超参数。每个 target family、backend 和 variant 都先过滤 calibration/test pairs，再独立拟合、预测和选择：

| Variant | 训练和预测允许的 group sizes | 含义 |
|---|---:|---|
| `1` | `{1}` | singleton-only |
| `2` | `{1,2}` | singleton 加 exact size-2 groups |
| `4` | `{1,4}` | singleton 加 exact size-4 groups |
| `8` | `{1,8}` | singleton 加 exact size-8 groups |
| `all` | `{1,2,4,8}` | 完整混合模型，在 v3 内重新训练 |

所有条件使用相同的 20% logical estimated-token budget，预算基准是对应数据集全部 singleton queries 的 estimated-token 总成本。`baran_labeling_budget=20`、seed 42、`rho=1`、`gamma=1` 和 verifier 参数保持冻结。

## 2. 正式矩阵

固定 run ID：

```text
no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all
```

正式测试为 9 个数据集、22,198 个 error cells：

```text
beers / flights / hospital / movies_1 / rayyan /
company / marketing / restaurant_20 / soccer
```

最终方法矩阵包括：

- `baran_only`：复用并重新校验 fresh Baran 输出；
- `llm_only`：对每个 cell 使用 No-Baran singleton Prompt，失败、abstain、missing、invalid 或未改变 dirty value 时记为未修复，不使用 Baran fallback；
- `budgeted_group_lightgbm`：`k=1/2/4/8/all`；
- `budgeted_group_xgboost`：`k=1/2/4/8/all`。

因此共有 12 个完整方法切片、108 个 dataset slices、266,376 条 cell records，以及 2 backend × 5 variants × 9 datasets = 90 个 selection slices。每个 BGR dataset slice 必须覆盖全部 error cells；未选择、LLM 拒绝或 provider failure 均按协议回退 Baran。

## 3. 复用与成本语义

Router-v3 是独立的新 run，不修改 v1 或 Router-v2。它在本地重新生成并逐文件校验以下 identity：

- 输入 manifest；
- Baran cell ledger；
- cell features；
- candidate actions；
- memberships；
- calibration query plan。

只有 hash 完全一致时，才复用 Router-v2 的 8,197 条 calibration executions 和 16,451 条 pair labels。No-Baran response 的复用还要求以下字段全部一致：

```text
query_id / prompt_hash / provider_request_hash /
model / prompt_schema_version
```

复用来源和 hash 写入 `provenance/reuse_manifest.json` 与 `provenance/response_reuse.json`。Cache hit 只减少 physical request；selection 预算和报告中的 logical calls/tokens 不变。LLM-only 的全量 singleton queries 与十个 BGR slices 的选择结果在 selection 冻结后组成一个 physical request union，并按严格 request identity 去重。

## 4. Router 与 verifier 隔离

Router 产物按以下路径隔离：

```text
gates/<backend>/variant_<k>/<suite>__<dataset>.csv
gates/<backend>/variant_<k>/<suite>__<dataset>.metadata.json
```

每份 metadata 记录 target、variant、允许的 train/test group sizes、pair 数、特征列、训练摘要、正例信息和模型元数据。`split_audit.csv` 对 2 backend × 5 variants × 9 targets 分别记录 family/cell/row/query/group-signature overlap；所有 overlap 必须为零。

同一 `(cell, query)` 在不同 Router variant 下可能得到不同 conservative uplift，因此 verifier cache identity 包含 `variant`。一个 variant 的 verifier 结果不会被另一个 variant 复用。

## 5. 指标与统计

报告以逐数据集证据为主：

1. 9 行 × 12 方法列的 F1 主矩阵；
2. 108 行详细长表：correct repairs、predicted repairs、precision、recall/correction accuracy、F1、相对 Baran-only 与 LLM-only 的差值、logical calls 和 tokens；
3. BGR vs Baran-only、BGR vs LLM-only 的 dirty-row cluster paired bootstrap：2,000 次、seed 45；
4. 每个 backend × variant × baseline 内，对 9 个 dataset p-values 做 Holm 校正；
5. Micro、Dataset-Macro、Win/Tie/Loss 和成本审计放在报告后部作为补充。

机器可读结果包括：

```text
metrics/per_dataset_f1_matrix.csv
metrics/per_dataset_method_comparison.csv
metrics/paired_statistics.csv
metrics/method_metrics.csv
metrics/selection_audit.csv
metrics/api_cost_audit.csv
report/artifact.json
```

同时生成 `report/report.md` 与 `report/report.html`。

## 6. 完成门槛

Router-v3 只有同时满足以下条件才可标记为 `complete`：

- 266,376 条 final records 完整且 identity 无重复；
- 108 个 dataset slices 均精确覆盖自己的完整 cell universe；
- 90 个 selection slices 都不超过 20% estimated-token budget；
- `k=1/2/4/8/all` 的训练、预测和选择只含声明的 group sizes；
- 五个 variant 使用五套隔离的模型、预测和 diagnostics；
- family/cell/row/query/group-signature leakage 全为零；
- LLM-only 不读取或回退 Baran；
- response reuse、logical/physical cost 去重和复用 provenance 可独立核验；
- method metrics 可从 final cell ledger 独立重算；
- pytest、compileall、数据校验、run validation 和 report artifact validation 全部通过。

冻结的 `04_实验结果与分析.md` 和 Router-v2 report 不做修改。
