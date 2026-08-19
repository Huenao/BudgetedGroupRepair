# Router-v3：CatBoost 20% 预算全 k 实验

## 1. 实验状态

- Run：`runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/`
- Router revision：`router_v3_catboost_exact_size_conditioned`
- 数据：9 个正式测试数据集，22,1　98 个 error cells。
- 方法：Baran-only、LLM-only、CatBoost `k=1/2/4/8/all`，共 7 个完整切片。
- BGR 逻辑预算：每个数据集全量 singleton estimated-token 成本的 20%。
- 状态：`complete`；155,386 条 cell records 完整且无重复。

本实验是受控 Router 替换：仅将 LightGBM/XGBoost 换成 CatBoost。训练标签、输入特征、LOFO 划分、选择器、Verifier、`rho=1`、`gamma=1`、seed 42 和逻辑预算均保持不变。CatBoost 使用原生 categorical features；数值缺失值使用训练集 median，类别缺失值使用固定 sentinel。

当前 V3 基准实现不强制绑定外部 comparison run。未提供 `--router-comparison-run` 时，CatBoost 的训练、选择、完整指标与报告独立完成；提供时才增加 LightGBM/XGBoost 对齐比较。本页冻结 run 使用了 comparison run，因此包含后文 90 条额外比较。

## 2. Group-size 条件

| Variant | 训练和候选 group sizes |
| --- | --- |
| `k=1` | `{1}` |
| `k=2` | `{1,2}` |
| `k=4` | `{1,4}` |
| `k=8` | `{1,8}` |
| `all` | `{1,2,4,8}` |

45 个 target × k 模型和 45 个 selection 均独立冻结；每个 selection 都未超过 20% estimated-token 预算。family/cell/row/query/group-signature 泄漏检查均为 0。

## 3. 逐数据集主要结果

单元格格式为 `F1（升级为 LLM 修复的 cells）`。LLM-only 列中的 cells 是合法、非空且改变 dirty value 的 LLM repairs 数。

| Dataset | Baran F1 | LLM-only | CatBoost k=1 | CatBoost k=2 | CatBoost k=4 | CatBoost k=8 | CatBoost all |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hospital | 0.9024 | 0.9311（507） | 0.9316（36） | 0.9358（45） | 0.9369（44） | **0.9429（47）** | 0.9316（36） |
| flights | **1.0000** | 0.9230（4,850） | 0.9984（8） | 0.9990（5） | 0.9980（10） | 0.9984（8） | 0.9986（7） |
| beers | 0.8599 | 0.4915（3,899） | **0.8959（192）** | 0.8822（230） | 0.8673（269） | 0.8668（274） | 0.8673（265） |
| rayyan | **0.6420** | 0.0063（327） | 0.6264（32） | 0.6310（27） | 0.6326（23） | 0.6350（19） | 0.6234（46） |
| movies_1 | **0.8414** | 0.6131（5,764） | 0.8131（757） | 0.8153（621） | 0.8190（524） | 0.8171（466） | 0.8099（610） |
| company | 0.5689 | **0.6301（558）** | 0.6103（72） | 0.6077（83） | 0.6098（99） | 0.6103（84） | 0.6136（93） |
| marketing | 0.5687 | **0.6287（645）** | 0.6224（101） | 0.6217（102） | 0.6255（109） | 0.6208（100） | 0.6236（107） |
| restaurant_20 | 0.2056 | 0.2440（402） | 0.2484（53） | 0.2595（66） | 0.2806（57） | 0.3096（81） | **0.3111（90）** |
| soccer | 0.8562 | **0.9819（1,875）** | 0.9168（174） | 0.9170（176） | 0.9261（208） | 0.9161（164） | 0.9203（192） |

Dataset-macro F1 为：Baran 0.7161、LLM-only 0.6055、CatBoost k=1 0.7404、k=2 0.7410、k=4 0.7440、k=8 0.7463、all 0.7444。五个 CatBoost variant 相对 Baran 均为 6 win / 0 tie / 3 loss；最好的 dataset-macro variant 是 `k=8`，但不存在对所有数据集都最优的单一 k。

## 4. API、复用与失败策略

- Dry plan 冻结了 25,048 个 physical-union query identities，其中 23,025 条成功响应和 77 条终态失败由父 run 复用。
- 经授权发送的缺失 dataset-derived prompts 恰为 1,946 条，全部取得 success；1,919 条 `ok`、27 条 `partial`，共 1,950 provider attempts、7,203,336 observed tokens。
- 加上一次不含数据集内容的 model/schema preflight，本次 fresh physical calls 为 1,947，observed tokens 为 7,203,971。
- 缓存只降低物理调用，不改变任何 BGR 切片的逻辑预算。BGR 未选择、拒绝、partial 缺失 cell 或冻结 provider failure 均回退 Baran；LLM-only 不回退 Baran。

## 5. 验收

- 41 项 pytest 全部通过，`compileall` 通过。
- 数据 manifest：14 个数据集、46 个文件，hash 全部验证成功。
- 155,386 条 cell records、63 个逐数据集方法切片、77 条含 micro/macro 指标。
- 45 个 selection、45 个 split/model metadata、180 条 paired-stat rows、90 条 CatBoost vs LightGBM/XGBoost 对齐比较。
- 严格完成态 validator 从 cell ledger 独立重算通过；formal、record、budget 与 leakage audits 均为 `ok=true`。

完整产物：

- `runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/report/report.md`
- `runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/report/report.html`
- `runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/report/artifact.json`
- `runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/metrics/per_dataset_f1_matrix.csv`
- `runs/no_baran_router_v3_deepseek_v4_20260726_budget20_k1248_all_catboost/metrics/per_dataset_method_comparison.csv`
