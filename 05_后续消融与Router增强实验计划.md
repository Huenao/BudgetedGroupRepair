# Budgeted Group Repair（No-Baran Prompt）：后续消融与 Router 增强实验计划

> 本文用于指导 `no_baran_router_v2_deepseek_v4_20260725_full9` 之后的探索性实验、模型选择和最终确认实验。现有 v1 与 Router-v2 正式 run 均保持只读，不覆盖配置、响应、指标或报告。

## 1. 当前证据与后续实验目标

### 1.1 当前已经得到的结果

现有实验提供了三类证据：

1. **互补性得到支持。** 在实验一的 2,700 个固定样本 cells 上，Baran 与 No-Baran singleton LLM 修对的 cell 集合不同，Oracle UB 相比 Baran 提高 10.11 个百分点。
2. **Group 的成本摊销得到支持，但总体质量非劣没有得到支持。** 实验二的 structured group 相比 matched singleton 的 accuracy 点估计下降 0.39 个百分点，95% CI 为 `[−1.47, +0.63]` 个百分点，没有通过预设的 −1 个百分点非劣界；8 个 usage 完整数据集的 macro token/cell saving 为 27.46%。
3. **预算 Router 有可利用信号，但完整 group library 仍不稳定。** Router-v2 在 20% 预算下，两种 backend 都有 6/9 个数据集的 F1 高于 Baran，正确修复总数也增加；但 micro F1 略低于 Baran，而且 singleton-only 在 20% 预算下优于完整 `1/2/4/8` 候选池。

因此，后续实验不是重新证明实验一，而是回答以下问题：

- 哪些 group size、view 和 cohesion 区间真正有帮助？
- 性能下降来自 Router 排序、LLM batch interference，还是 verifier 错误接受？
- `rho`、`gamma`、接受阈值和预算如何影响 helpful/harmful 权衡？
- 完整 BGR 是否能在相同预算下超过 singleton-only Router？
- 更强或更合适的 Router 是否能在严格 family holdout 下稳定改善结果？
- 当前观察到的提升是否能跨 seed、跨数据集和跨 Prompt 条件复现？

### 1.2 后续实验的最终目标

后续实验应形成一条可审计的证据链：

```text
定位 group harm 来源
→ 筛选安全且有效的 group 子库
→ 调整风险与 verifier 参数
→ 完成组件消融和选择器基线
→ 比较更强 Router
→ 冻结最终配置
→ 在未参与模型选择的数据上做确认实验
```

最终需要区分两种可能结论：

- **结论 A：完整 BGR 得到支持。** Group actions 在相同预算下相对 singleton-only 形成稳定的质量提升，或在质量非劣时形成明确的成本优势。
- **结论 B：预算化 singleton Router 得到支持，但当前 group actions 未形成额外价值。** 该结论仍然有研究价值，但论文必须把 group 的结果作为限制或负结果报告。

## 2. 实验治理与公平性规则

### 2.1 冻结现有正式 run

以下 run 只能作为只读基线：

```text
runs/no_baran_deepseek_v4_20260724_final/
runs/no_baran_router_v2_deepseek_v4_20260725_full9/
```

任何后续实验必须：

- 使用新的 run ID；
- 保存独立配置和 run fingerprint；
- 记录父 run、复用文件 hash 和响应来源；
- 不修改现有 `calibration_pair_labels.csv`、候选 queries、selection 或 final records；
- 将 exploratory 与 confirmatory 结果物理隔离。

建议使用以下 run ID 前缀：

```text
no_baran_followup_e1_group_attribution_<date>
no_baran_followup_e2_risk_sweep_<date>
no_baran_followup_e3_component_ablation_<date>
no_baran_followup_e4_router_benchmark_<date>
no_baran_followup_e5_confirmatory_<date>
```

这些名称是实验阶段编号，不改变论文中的方法名称，也不自动将方法升级为新的理论版本。

### 2.2 探索集与确认集必须分离

当前 9 个正式测试数据集的结果已经被观察。后续可以使用它们进行：

- action-level 错误分析；
- group size/view 诊断；
- 生成探索性图表；
- 验证代码能否复算现有结果。

但不应在这些结果上反复挑选参数，然后仍将同一批数据称为“完全未见测试集”。后续模型选择应采用：

```text
外层：保持现有 target-base-family holdout，用于估计跨 family 泛化
内层：只在外层训练 families 中再次进行 family CV，用于选择模型和参数
```

严格禁止：

- 使用当前 target family 的 clean value 调参；
- 使用 target LLM response 决定 selection；
- 根据 9 个正式测试集的最终 F1 选择 `rho/gamma/backend`，再把同一结果作为无偏确认结果；
- 使用随机 row split 代替 family split 来报告主要 Router 性能。

最终确认实验推荐增加新的 base families。若暂时没有新数据集，必须将后续结果标为 exploratory，并使用 nested family CV 报告，而不是宣称外部确认。

### 2.3 预先冻结指标与比较口径

论文中的评价必须采用“**逐数据集完整结果为主体，跨数据集聚合为总结**”的层次。Micro、Dataset-Macro 和 Family-Macro 回答的问题不同，任何一个聚合数都不能替代逐数据集结果。

#### 2.3.1 完整评价集与配对比较

对每个数据集 $d$，先冻结完整评价 cell 集合 $E_d$。Baran、BGR 和所有消融方法必须在完全相同的 $E_d$ 上计算指标：

- 不能只评价被 Router 选中或被 LLM 成功返回的 cells；
- 未选择、abstain、missing、invalid、verifier reject 和终止性 provider failure 都必须按协议回退 Baran，并保留在分母中；
- 每个方法在每个数据集上的最终记录数必须恰好等于 `|E_d|`；
- BGR 与 Baran 的差值必须按同一个 cell 配对计算；
- 预算、Prompt、模型和失败处理必须一致，不能为某个数据集单独选择更有利的 operating point。

每个数据集至少报告：

| 字段 | 含义 |
|---|---|
| `dataset`、`base_family`、`n_eval_cells` | 数据身份、独立 family 和完整评价规模 |
| Baran P/R/F1/accuracy | Baseline 的完整数据集结果 |
| BGR P/R/F1/accuracy | 同一完整数据集上的 BGR 结果 |
| `delta_f1`、`delta_correct_repairs` | BGR 相对 Baran 的配对差值 |
| 95% CI、校正后 p-value | 按 dirty row 聚类的配对不确定性 |
| logical/provider tokens、token/cell | 相同预算口径下的成本 |
| Win/Tie/Loss | 按预先冻结的差值或非劣界判定 |

逐数据集结果是论文判断“方法在哪些数据上有效、在哪些数据上失败”的主要证据，不能只展示平均值。

#### 2.3.2 三种聚合口径

**Micro**：先汇总所有数据集的 cell-level confusion counts，再计算指标。例如：

\[
P_{micro}=\frac{\sum_d TP_d}{\sum_d(TP_d+FP_d)},\qquad
R_{micro}=\frac{\sum_d TP_d}{\sum_d(TP_d+FN_d)}
\]

\[
F1_{micro}=\frac{2P_{micro}R_{micro}}{P_{micro}+R_{micro}}
\]

它回答“在所有评价 cells 合并后，整体 cell-level 修复效果如何”，但会被 cell 数量大的数据集主导。

**Dataset-Macro**：先在每个数据集上计算指标，再让每个数据集等权：

\[
F1_{dataset\text{-}macro}=\frac{1}{D}\sum_{d=1}^{D}F1_d
\]

它回答“在名义上的不同数据集之间是否普遍有效”，但如果同一个 base family 有多个变体，该 family 会获得更多权重。

**Family-Macro**：先在同一个 base family 内平均，再对不同 base families 等权：

\[
F1_f=\frac{1}{|D_f|}\sum_{d\in D_f}F1_d,\qquad
F1_{family\text{-}macro}=\frac{1}{F}\sum_{f=1}^{F}F1_f
\]

当前存在同 family 数据集变体，因此 Family-Macro 更接近“跨独立数据来源的泛化表现”。`base_family` 映射必须来自冻结 manifest，不能在看结果后调整。

所有聚合口径都应同时报告绝对值和相对 Baran 的配对差值，例如：

\[
\Delta F1_{family\text{-}macro}
=\frac{1}{F}\sum_f\frac{1}{|D_f|}\sum_{d\in D_f}
\left(F1^{BGR}_d-F1^{Baran}_d\right)
\]

#### 2.3.3 论文中的指标层级

建议在最终确认实验前冻结为：

- **逐数据集主要证据：** 每个完整数据集上的 `delta_f1`、`delta_correct_repairs` 和配对 95% CI；
- **主要跨域汇总指标：** 预先指定 operating point 下的 Family-Macro `delta_f1`；
- **关键次要汇总指标：** Dataset-Macro F1、Micro F1、correct repairs/correction accuracy、Win/Tie/Loss；
- **成本—质量指标：** Micro、Dataset-Macro 和 Family-Macro F1 AUBC，以及 token/cell；
- **安全护栏：** Micro F1 非劣界、最差单数据集退化、Baran-correct cells 被错误覆盖的 harmful rate；
- **Group 独立证据：** group-enabled 相对 singleton-only 的质量差值或成本—质量 Pareto 优势。

如果论文的主张是“跨数据集/跨来源普遍有效”，Family-Macro 应优先于 Micro；如果主张是“在这 22,198 个 cells 上总共修得更好”，Micro 才是对应的总体指标。两者都应报告，但不能在看完结果后选择其中更有利的一项作为主要指标。

“整体优于 Baran”也不等于“每一个数据集都必须提高”。正式结论应结合：Family-Macro 的方向和置信区间、独立 families 的 Win/Tie/Loss、Micro 安全护栏以及最差数据集退化；详细门槛见 9.3。

#### 2.3.4 必报指标全集

每个实验必须同时报告以下指标：

| 类型 | 必报指标 |
|---|---|
| 修复质量 | correct repairs、correction accuracy/recall、precision、F1 |
| 汇总 | per-dataset、Micro、Dataset-Macro、Family-Macro、Win/Tie/Loss |
| 路由质量 | helpful AUPRC/Brier、harmful AUPRC/Brier、top-ranked observed uplift |
| Group 行为 | group size/view、selected groups、unique covered cells、overlap、batch interference |
| 风险 | helpful accepts、harmful overwrites、false accepts、false rejects、fallback |
| 成本 | logical calls、physical calls、estimated tokens、provider tokens、token/cell、AUBC |
| 审计 | family/cell/row/query/group-signature leakage、预算、Prompt、coverage |

由于理论目标直接对正确修复数建模，correct repairs 必须与 F1 同时报告；不能只报告其中更有利的一项。

### 2.4 可复用数据与 API 边界

后续实验优先复用当前正式 run 的冻结产物：

| 产物 | 路径 | 可复用范围 |
|---|---|---|
| 输入快照 | `input_data_manifest.json` | 数据不变时复用 |
| Baran 结果 | `baran/` | Baran 配置和数据不变时复用 |
| Cell 特征 | `cell_features/` | 特征定义不变时复用 |
| 候选 action | `groups/candidates/` | 数据、分组和 Prompt identity 不变时复用 |
| Cell-query 特征 | `groups/memberships/` | Router feature schema 不变时复用 |
| Calibration 计划与响应 | `llm/calibration_*.jsonl` | request identity 不变时复用 |
| Router 标签 | `llm/calibration_pair_labels.csv` | 标签语义、Baran reference 和响应不变时复用 |
| 已执行响应缓存 | `llm/group_response_cache.jsonl` | `query_id + prompt_hash + provider_request_hash + model + schema` 一致时复用 |
| 最终 cell records | `final/all_methods.jsonl` | 只用于复算当前已冻结场景 |

当前已有：

- 126,603 个候选 queries；
- 362,391 个 cell-query incidences；
- 8,197 个 calibration queries；
- 16,451 个 calibration cell-query 标签；
- 当前 selection union 的已执行响应和失败记录。

以下实验通常不需要新的 API 调用：

- group failure attribution；
- Router feature/model/seed 的离线比较；
- nested family CV；
- `rho/gamma` 和 calibration 参数扫描；
- 使用已有响应覆盖范围内的 verifier 与指标重算；
- 统计 bootstrap 和置信区间。

以下情况可能需要新的 API 调用：

- 新 Router 选中了当前 cache 中不存在的 query；
- 修改 group membership、Prompt、上下文、completion ceiling 或模型；
- 运行 Baran-informed Prompt 配对实验；
- 增加新的最终确认数据集。

响应复用只减少物理调用。每个逻辑场景仍必须按其完整 query cost 计算预算，不能把 cache hit 当作零成本 action。

## 3. 总体执行顺序

| 阶段 | 核心问题 | 默认 API | 主要输出 | 是否进入下一阶段 |
|---|---|---:|---|---|
| Phase 0 | 冻结协议、复用和指标 | 否 | follow-up manifest | 所有审计通过 |
| Phase 1 | Group 为什么失败 | 否 | action-level attribution | 找到可解释的 size/view 风险模式 |
| Phase 2 | 风险参数能否控制 harmful | 否 | nested-CV risk frontier | calibration-only 结果稳定 |
| Phase 3 | 哪些组件真正贡献 | 通常否 | 完整消融矩阵 | group 子库优于或 Pareto 支配 singleton-only |
| Phase 4 | 更强 Router 是否更好 | 否 | router benchmark | inner-family CV 稳定胜出 |
| Phase 5 | 结果是否稳健、可确认 | 可能需要 | confirmatory report | 预注册门槛通过 |

后续阶段不能因为前一阶段没有正结果而修改已冻结指标或删除负结果。允许停止并报告“当前 group 不受支持”。

## 4. Phase 0：准备与冻结

### Step 0.1：创建 follow-up 配置

以 `configs/experiment_router_v2.json` 为只读参考，新增 follow-up 配置。配置至少冻结：

- parent run ID 和 parent manifest hash；
- 使用的数据集与 base-family 映射；
- candidate/action library hash；
- calibration label hash；
- Prompt schema 和模型；
- Router seed 列表；
- 预算点；
- 主要/次要指标；
- 统计方法和 bootstrap seed；
- exploratory 或 confirmatory 标记。

### Step 0.2：验证复用身份

对所有复用产物检查：

```text
input manifest hash
Baran manifest/config
cell feature schema
candidate query ID uniqueness
group signature
prompt hash
provider request hash
model identity
calibration coverage
```

若任一身份漂移，只允许重建受影响的派生产物；不能静默继续复用。

### Step 0.3：冻结统计口径

推荐：

- Router seeds：`42, 43, 44, 45, 46`；
- bootstrap：至少 2,000 次，最终报告可提高到 10,000 次；
- per-dataset 多重比较：Holm 校正；
- cell 结果：按 dirty row 聚类 bootstrap；
- 每个数据集：预先冻结完整评价 cell 集合，所有方法使用相同分母和 fallback 规则；
- 汇总层级：同时报告 per-dataset、Micro、Dataset-Macro、Family-Macro 和 Win/Tie/Loss；
- 独立性单位：以 manifest 中冻结的 `base_family` 为准，同 family 的多个数据集变体不能当作完全独立样本；
- Macro 结果：Dataset-Macro 按 dataset 等权，Family-Macro 先在 family 内平均、再按 family 等权；
- 主要 operating point、主要指标、非劣界和最差数据集退化上限必须在确认结果可见前冻结；
- 预算曲线：固定 `1/5/10/20/50%`，同时报告 AUBC；
- 所有失败、abstain、missing、invalid 保留在完整分母。

### Phase 0 验收条件

- parent run 未被修改；
- 所有复用文件 hash 已记录；
- target label/response 在 selection 前不可见；
- exploratory/confirmatory 状态明确；
- 指标和参数搜索空间在运行前冻结。

## 5. Phase 1：Group 失败归因

### 5.1 实验目的

确定完整 group library 在 20% 下落后 singleton-only 的原因，分别量化：

1. LLM batch interference；
2. Router 选择错误；
3. group-level 收益聚合误差；
4. verifier false accept/false reject；
5. 特定 dataset/view/size 对总体结果的支配。

### 5.2 Step 1：建立同 cell 的 singleton–group 配对表

对 calibration 和已有正式响应构造如下粒度的数据：

```text
(suite, dataset, base_family, cell_id,
 singleton_query_id, group_query_id,
 group_view, group_size, cohesion_quartile,
 baran_correct, singleton_correct, group_correct,
 predicted_helpful, predicted_harmful,
 conservative_uplift, verifier_decision,
 selected, estimated_tokens, provider_tokens)
```

只比较同一个 cell 的 singleton 与 group 结果，避免把数据集组成差异误认为 group 效果。

### 5.3 Step 2：定义错误归因类别

每个 cell-query pair 至少分入以下类别：

| 类别 | 判定 |
|---|---|
| Group helpful | group 正确、Baran 错误 |
| Group harmful | Baran 正确、group 错误 |
| Positive group transfer | singleton 错误、group 正确 |
| Batch interference | singleton 正确、group 错误 |
| Both correct | singleton 与 group 都正确 |
| Both wrong | singleton 与 group 都错误 |
| Router false positive | 被选中但 realized uplift 小于 0 |
| Router false negative | 未选中但已有响应显示 realized uplift 大于 0，仅作离线诊断 |
| Verifier false accept | verifier 接受了错误 LLM candidate，而 Baran 正确 |
| Verifier false reject | verifier 拒绝了正确 LLM candidate，而 Baran 错误 |

`false negative`、`false accept/reject` 使用 clean label，只能进入离线诊断，不能成为在线特征。

### 5.4 Step 3：分层分析

按以下维度分别报告计数、比例、净 uplift 和 bootstrap CI：

- dataset/base family；
- `group_size ∈ {2,4,8}`；
- `group_view ∈ {row,pattern,public_fd,semantic}`；
- cohesion quartile；
- dirty type/format；
- Baran changed/support bucket；
- predicted helpful/harmful decile；
- selected budget；
- verifier decision；
- cell 在候选库中的 overlap degree。

重点检查：

- `movies_1`：为何 group 和高预算累积大量 harmful；
- `flights`：Baran 已接近/达到完美时，Router 为何仍允许升级；
- `rayyan`：LLM 几乎没有 helpful 时，错误调用和错误预测从哪里产生；
- size-8 semantic：是否存在明显 attention/batch interference；
- public-FD/pattern size-2：是否形成相对安全的 group 子库。

### 5.5 Step 4：生成可执行结论

输出以下表格：

1. `group_interference_by_dataset_view_size.csv`；
2. `router_error_attribution.csv`；
3. `verifier_error_attribution.csv`；
4. `safe_group_subsets.csv`；
5. `phase1_group_attribution.md/html`。

### Phase 1 验收与决策

Phase 1 不要求所有 group 都变好。需要得到以下至少一种结果：

- 找到一个预定义明确的 group 子库，在多个 training families 上表现为正 transfer 或质量非劣且成本更低；
- 明确识别某些 size/view 为系统性 harmful，并有充分理由从后续候选池中删除或增加惩罚；
- 若不存在任何稳定 group 子库，则停止把“group 提升质量”作为主要经验主张，后续以 budgeted singleton Router 为主要系统。

筛选规则必须在 calibration inner folds 上确定；不能仅根据正式测试集上最好的组合挑选。

## 6. Phase 2：Router 风险与阈值参数实验

### 6.1 实验目的

控制高预算下的 harmful accumulation，使更多预算不会因为错误覆盖 Baran-correct cells 而显著降低逐数据集、Family-Macro 或 Micro F1。

当前基线为：

```text
rho = 1
gamma = 1
verifier.minimum_llm_confidence = 0.55
verifier.minimum_net_gain = 0
verifier.acceptance_score = 0.55
```

### 6.2 Step 1：粗粒度 `rho/gamma` 扫描

建议先运行较小网格：

```text
rho   ∈ {1, 2, 4}
gamma ∈ {0, 0.5, 1, 2}
```

共 12 个组合。每个组合在 inner family CV 中生成：

- 五点预算曲线；
- micro/macro F1 AUBC；
- correct repairs；
- harmful overwrite rate；
- selected group size/view 分布；
- 预算使用率。

不得用 outer target 结果选择组合。

### 6.3 Step 2：净收益和 harmful 阈值

在 Step 1 的前 2–3 个组合上再扫描：

```text
minimum conservative uplift ∈ {0, 0.02, 0.05, 0.10}
maximum predicted harmful    ∈ {0.02, 0.05, 0.10, unrestricted}
```

如果概率 calibration 不可靠，阈值实验必须同时报告 reliability curve、Brier 和 expected calibration error，不能只看 AUPRC。

### 6.4 Step 3：Group-size 和 view 风险惩罚

根据 Phase 1 的 training-family 诊断，测试：

```text
adjusted_gain = conservative_uplift
                - size_penalty[group_size]
                - view_penalty[group_view]
```

惩罚值必须在 inner CV 中学习或从冻结的小网格选择。不能根据 target dataset 的最终错误率手工设定。

推荐至少比较：

- 无额外惩罚；
- 只惩罚 size-8；
- 对各 size 使用单调增加惩罚；
- 使用 Phase 1 识别的 view-specific penalty。

### 6.5 Step 4：Verifier 阈值扫描

固定 Router 后再测试：

```text
minimum_llm_confidence ∈ {0.55, 0.65, 0.75}
acceptance_score       ∈ {0.55, 0.65, 0.75}
require_comparative_signal ∈ {true, false}
```

为控制搜索规模，采用分阶段扫描，不运行全部笛卡尔积：

1. 固定 acceptance score，扫描 minimum confidence；
2. 固定最佳 inner-CV confidence，扫描 acceptance score；
3. 最后比较 comparative signal 开关。

### Phase 2 选择规则

建议采用以下 lexicographic 规则：

1. 先排除 inner folds 中出现明显 harmful 爆发或预算违规的配置；
2. 排除违反 Micro 非劣界或最差 family 退化护栏的配置；
3. 在剩余配置中最大化 inner-family 的 Family-Macro F1 AUBC；
4. 若差异小于预先冻结的容差，选择 harmful rate 更低的配置；
5. 仍相同时选择更简单、参数更接近当前基线的配置。

输出：

- `risk_sweep_all_configs.csv`；
- `risk_sweep_inner_fold_metrics.csv`；
- `risk_sweep_pareto_frontier.csv`；
- `selected_risk_config.json`；
- `phase2_risk_sweep.md/html`。

## 7. Phase 3：完整组件消融

### 7.1 实验目的

分别验证 group library、Router 特征、uncertainty、优化器和 verifier 是否真正贡献结果。

所有消融必须保持：

- 相同 target fold；
- 相同预算定义；
- 相同候选 population；
- 相同 Prompt/model；
- 相同失败与 fallback 规则；
- 相同指标分母。

一次只改变一个因素。

### 7.2 Step 1：Group size 消融

当前已有：

```text
[1]
[1,4]
[1,2,4,8]
```

需要补充：

```text
[1,2]
[1,8]
[1,2,4]
[1,2,8]
[1,4,8]
```

主要比较：

- 每增加一种 group size 后，F1/correct repairs 是否提高；
- token/cell 和 logical calls 是否下降；
- harmful overwrite 是否集中在 size-8；
- 结果是否跨 backend/seed 一致。

### 7.3 Step 2：Group view 消融

固定 size 集合后运行：

```text
singleton-only
singleton + row
singleton + pattern
singleton + public_fd
singleton + semantic
singleton + Phase-1-safe-views
singleton + all views
```

不得将 singleton 从对照中删除，因为正式系统允许 fallback 和点动作。

### 7.4 Step 3：Router feature 消融

至少比较：

| Variant | 保留特征 |
|---|---|
| Full | 当前完整 Project-compatible schema |
| No-Baran-router-features | 去掉 Baran type/format/support/changed 等 Router 特征 |
| Cell-only | 只保留 cell-level 特征 |
| Group-only | 只保留 group/view/size/cohesion/cost 特征以及最小身份字段 |
| No-interaction | 去掉 cell × group interaction |
| No-uncertainty | 使用净 uplift，不减 `gamma × sigma` |

`No-Baran-router-features` 与 No-Baran Prompt 是不同消融：前者禁止 Baran 进入 Router，后者只禁止 Baran 进入 LLM messages。

### 7.5 Step 4：选择器和优化器基线

在相同预算下比较：

1. Baran only；
2. random feasible actions；
3. singleton-only Router；
4. 逐 query 的 predicted uplift/cost 排序，不做重叠边际更新；
5. 逐 cell Router，只允许 singleton；
6. non-overlap group selection；
7. 当前 lazy submodular knapsack；
8. 小规模实例的 exhaustive optimum，仅用于实现正确性检查；
9. oracle selection，仅作为不可部署上界，不参与主要方法排名。

该步骤直接回答 RQ6：性能来自 Router、group 覆盖，还是次模优化器本身。

### 7.6 Step 5：Verifier 消融

比较：

```text
当前 verifier
无 verifier，直接采用最高 ranked candidate
只检查格式/约束的 verifier
只检查 comparative signal 的 verifier
oracle verifier（离线上界）
```

报告 verifier：

- 阻止了多少 harmful overwrite；
- 同时拒绝了多少 helpful repair；
- net saved corrections；
- 对 precision/recall/F1 的独立贡献。

### Phase 3 最小结论要求

对于“group actions 有额外价值”的主张，至少需要满足以下一种条件：

- 在相同逻辑 token 预算下，group-enabled 方法相对 singleton-only 的主要质量指标提高，且置信区间支持该方向；
- 质量通过预先冻结的非劣界，同时 token/cell 或 logical calls 达到预先冻结的实际节省门槛；
- 在多个 base families 上形成稳定的成本—质量 Pareto 优势，而不是只由一个小数据集驱动。

若未满足，应保留完整负结果，并将主要方法收缩为 budgeted singleton routing。

## 8. Phase 4：更强 Router 模型比较

### 8.1 实验目的

判断当前瓶颈是模型容量、概率 calibration、排序目标，还是训练数据本身。

### 8.2 Step 1：建立统一 Router 接口

所有 Router backend 必须输出：

```text
p_helpful
p_harmful
uncertainty
net_gain
conservative_uplift
```

统一输入 schema、fold、seed、sample weight 和 missing-value 规则。所有 backend 使用相同 optimizer 和 verifier，避免把后处理差异误认为模型差异。

### 8.3 Step 2：模型候选

优先比较无需新增依赖的模型：

- LightGBM；
- XGBoost；
- sklearn HistGradientBoosting；
- ExtraTrees/RandomForest；
- LightGBM + XGBoost calibrated ensemble。

第二阶段再考虑：

- CatBoost；
- ranking/LambdaMART 风格目标；
- helpful 与 harmful 使用不同 backend；
- cost-sensitive 或 focal-style harmful loss。

如果需要新增依赖，必须先单独确认，不应为了模型列表完整而直接安装。

### 8.4 Step 3：Nested family CV

对每个 outer target family：

1. 排除 outer target family；
2. 在剩余 calibration families 上做 inner leave-one-family-out；
3. 用 inner folds 选择模型、超参数和 calibration；
4. 使用全部 outer-training families 重新拟合；
5. 对 outer target 只预测一次；
6. selection 冻结后才允许读取 outer response/label 做评价。

不得使用随机 row-level CV 作为模型选择主结果。

### 8.5 Step 4：模型选择指标

Router 不能只按 AUPRC 排名。建议同时比较：

- helpful/harmful AUPRC；
- helpful/harmful Brier/ECE；
- top-ranked observed uplift；
- inner-fold Family-Macro budgeted F1 AUBC；
- inner-fold Micro F1 和最差 family 退化护栏；
- inner-fold correct repairs；
- harmful overwrite rate；
- 跨 family 最差结果；
- seed 方差。

最终模型优先选择预算后的实际 utility 稳定者，而不是单一分类指标最高者。

### 8.6 Step 5：模型保存与复现

每个被保留的模型应保存：

- backend 和版本；
- feature schema/order；
- hyperparameters；
- calibration 方法；
- inner-CV selection 记录；
- training-family 列表；
- seed；
- serialized model；
- model artifact hash。

### Phase 4 验收条件

更强 Router 只有在以下条件下才替换现有模型：

- inner-family CV 的预算后指标稳定提高；
- 不依赖单个 family；
- harmful rate 没有明显恶化；
- calibration 和 seed 方差可接受；
- outer target 没有被用于模型选择。

## 9. Phase 5：统计稳健性与最终确认

### 9.1 Step 1：多 seed 稳健性

至少运行 5 个 Router seeds。只改变 Router 训练、calibration 或确定性 tie-breaking 中声明允许变化的随机源，不改变：

- 数据快照；
- Prompt/model；
- query action identity；
- target fold；
- 预算；
- 指标定义。

报告 mean、standard deviation、median、min/max，以及每个 dataset 的符号一致性。

### 9.2 Step 2：配对置信区间

对最终冻结配置运行：

- BGR vs Baran 的 row-cluster paired bootstrap；
- BGR vs singleton-only 的 row-cluster paired bootstrap；
- per-dataset 95% CI；
- Dataset-Macro 的 dataset-level bootstrap；
- Family-Macro 的 hierarchical bootstrap：首先以 base family 为独立单位重采样，再在 family 内保留或重采样其数据集/dirty rows；
- 每个 aggregation level 的 `delta_f1`、`delta_correct_repairs` 和 Win/Tie/Loss；
- 多数据集/多预算比较的 Holm 校正；
- correct repairs、F1 和 harmful overwrite 的配对分析。

不能使用独立样本检验代替同 cell 的配对检验。

同一 base family 的多个数据集变体不能在统计检验中被当作完全独立的外部数据集；否则会低估不确定性并让拥有更多变体的 family 获得更高权重。

### 9.3 Step 3：确认性成功门槛

以下是建议门槛，具体数值必须在确认实验前冻结。成功判断分成三层，不能用其中一层替代另一层。

#### A. 评价完整性 hard-gate

1. 每个方法在每个数据集上覆盖完整且相同的冻结评价 cell 集；
2. 未选择和所有失败状态均按统一规则回退，并保留在分母中；
3. operating point、预算、主要指标和 `base_family` 映射在结果可见前冻结；
4. 所有预算、模型、Prompt、泄漏、provenance 和 coverage 审计通过。

任一 hard-gate 失败时，不计算或宣称“整体优于 Baran”。

#### B. “BGR 整体优于 Baran”主张

建议同时要求：

1. 主要指标 Family-Macro `delta_f1 > 0`；在新增锁定确认 families 上，其预先指定置信区间下界也大于 0；
2. 独立 base families 的 Win 数占多数，且所有 family/dataset 的负结果完整展示；
3. Micro F1 不低于预设非劣界，避免通过多个小数据集的收益掩盖大量 cells 上的损失；
4. 最差单数据集 `delta_f1` 不超过预设退化上限，避免隐藏灾难性失败；
5. correct repairs/correction accuracy 有净提升，且 harmful overwrite 不超过预设安全上限；
6. 实际成本不超过冻结预算。

如果只有 Family-Macro 点估计为正、但置信区间跨 0，只能表述为“观察到正向趋势”；如果只在 Micro 上提高，则只能表述为“在合并 cells 上提高”，不能推断为跨数据来源普遍有效。

#### C. “Group 组件提供额外价值”主张

必须单独比较 group-enabled 与 singleton-only，并至少满足以下之一：

1. 相同逻辑 token 预算下，group-enabled 的主要质量指标显著提高；
2. 相同质量下，group-enabled 的实际 token/cell 或调用成本显著降低，形成稳定的成本—质量 Pareto 优势。

BGR 优于 Baran 并不能自动证明 group action 有价值；如果收益主要来自 singleton Router，论文必须分别陈述这两个结论。

非劣 margin、最差数据集退化上限和实际 token saving threshold 应在看新确认集结果之前写入配置。实验二的 1 个百分点和 15% 可以作为候选口径，但不能在结果出来后修改。当前 9 个已观察数据集上的后续调参结果只能作为 exploratory；确认性优越结论需要新增锁定 base families。

### 9.4 Step 4：Baran-informed Prompt 配对对照

为识别“从 LLM Prompt 删除 Baran 信息”的因果影响，建立完全配对对照：

```text
No-Baran Prompt
vs
Baran-informed Prompt
```

必须保持：

- 相同 query cell groups；
- 相同 folds；
- 相同模型与生成参数；
- 相同预算；
- 相同 Router/optimizer/verifier 比较口径；
- 相同失败处理；
- 同 cell 配对统计。

由于 Prompt messages 和 request hash 改变，该实验通常需要新的 API 调用，不能复用 No-Baran 响应冒充对照。

### 9.5 Step 5：新增锁定确认数据

最终建议增加未参与以下过程的新数据集/base families：

- group 子库筛选；
- 风险参数选择；
- Router backend 选择；
- verifier 阈值选择；
- 论文主 operating point 选择。

新数据第一次运行前应冻结：

- 最终代码 hash；
- 最终配置；
- 数据 manifest；
- 模型 artifact；
- primary metric；
- success criteria；
- API/token policy。

## 10. 建议的代码实现顺序

下面是实验代码的推荐增量，每一步都应小范围实现和测试。

### Step A：只读归因分析器

功能：

- 读取当前 run 的 candidates、memberships、responses、predictions、selections 和 final records；
- 生成 singleton–group 配对表；
- 输出 Router/LLM/verifier 错误归因；
- 不修改 parent run。

测试：

- 同一 `(cell_id, query_id)` 不重复；
- singleton/group 匹配正确；
- 分层计数可以回加到总体；
- clean-derived 字段不会写入在线特征。

### Step B：Derived-run 与离线重算

功能：

- 绑定 parent run；
- 复用 `baran/cell_features/groups/calibration`；
- 为新的 risk/ablation config 重训、重选和重算；
- 缺失 response 时只生成待调用计划，不伪造结果。

测试：

- parent hashes 不变；
- cache identity 严格；
- logical cost 不因复用而减少；
- target label 在 selection 前不可见。

### Step C：参数扫描 runner

功能：

- 分阶段运行 `rho/gamma/threshold/verifier` 网格；
- 保存 inner-fold 结果；
- 根据冻结规则选择配置；
- 禁止读取 outer target 指标进行选择。

测试：

- 配置数和 fold 数准确；
- 每个 fold 无 base-family 重叠；
- 相同 seed 可重复；
- tie-breaking 稳定。

### Step D：消融矩阵 runner

功能：

- size/view/feature/optimizer/verifier variants；
- 统一预算与指标；
- 输出 paired comparison 和 AUBC。

测试：

- 每次只改变声明的因素；
- 候选 size/view 过滤准确；
- 每个方法切片覆盖完整 22,198 cells；
- fallback 和成本独立重算。

### Step E：Router backend registry

功能：

- 统一 two-head interface；
- nested family CV；
- probability calibration；
- serialized model 和 provenance。

测试：

- constant-head fallback；
- rare helpful/harmful class；
- feature order/schema；
- model identity；
- outer-label invisibility。

### Step F：确认实验与报告

功能：

- 多 seed；
- paired cluster bootstrap；
- multiple-comparison correction；
- matched Prompt 对照；
- 新数据集 manifest；
- Markdown/HTML 报告。

## 11. 每阶段必须保存的产物

建议目录结构：

```text
runs/<followup_run_id>/
├── configs/
├── provenance/
│   ├── parent_run.json
│   └── reuse_manifest.json
├── diagnostics/
│   ├── group_pair_table.csv
│   ├── router_error_attribution.csv
│   └── verifier_error_attribution.csv
├── inner_cv/
│   ├── folds.csv
│   ├── parameter_results.csv
│   └── selected_config.json
├── models/
├── selections/
├── llm/
│   ├── selected_union_plan.json
│   └── execution.jsonl
├── final/
├── metrics/
├── audits/
├── report/
└── run_manifest.json
```

每张指标表必须包含：

```text
run_id
parent_run_id
exploratory_or_confirmatory
dataset
base_family
n_eval_cells
aggregation_level
backend
seed
outer_target_family
inner_validation_family
scenario
budget_share
group_size_variant
group_view_variant
feature_variant
optimizer_variant
verifier_variant
baseline_metric
method_metric
paired_delta
confidence_interval
win_tie_loss
```

## 12. Go/No-Go 决策表

| Gate | Go 条件 | No-Go 后的结论 |
|---|---|---|
| G0 Evaluation integrity | 每个方法逐数据集完整同分母，fallback、coverage、预算和泄漏审计全部通过 | 不得进行总体优越性比较，先修复实验协议或产物 |
| G1 Group signal | 至少一个 training-family 稳定的安全 group 子库 | 将 group 作为负结果，转向 singleton Router |
| G2 Risk control | inner CV 中 harmful 降低且质量/AUBC 不恶化 | 保留当前保守参数，不扩大预算 |
| G3 Component value | group-enabled 相对 singleton-only 有质量或 Pareto 优势 | 不宣称 group 提供额外经验价值 |
| G4 Stronger Router | nested family CV 跨 seed/family 稳定胜出 | 保留 LightGBM/XGBoost |
| G5 Confirmation | 新锁定 families 的逐数据集结果、Family-Macro 主指标、Micro/最差数据集护栏和冻结 success criteria 均通过 | 结果只作为 exploratory 或 mixed-evidence 报告 |

No-Go 不表示理论错误。它表示当前数据、group generator、Router 或 verifier 没有提供足够的经验支持，必须缩小论文主张。

## 13. 推荐的实际执行批次

### 批次 1：不调用 API 的诊断

1. 实现只读 attribution；
2. 生成 size/view/cohesion batch-interference 表；
3. 完成 `movies_1/flights/rayyan` case analysis；
4. 给出 safe/unsafe group 子库候选。

### 批次 2：不调用 API 的 Router 调参与消融

1. nested family CV；
2. `rho/gamma` 粗网格；
3. size/view/feature 消融；
4. verifier/optimizer 消融；
5. 多 Router seeds；
6. 选出 1–2 个候选配置。

### 批次 3：更强 Router 离线比较

1. 统一 backend interface；
2. 比较现有和新增模型；
3. 概率 calibration；
4. 根据 inner-CV budgeted utility 冻结最终 Router。

### 批次 4：补充缺失 selected responses

1. 生成新配置的 selected-query union；
2. 与现有 response cache 做严格 identity 匹配；
3. 估算缺失 queries、tokens 和费用；
4. 经明确授权后只调用缺失请求；
5. 完成 verifier 和 final metrics。

### 批次 5：确认实验

1. 冻结最终配置与 success criteria；
2. 加入新 base-family 数据；
3. 运行 No-Baran 最终确认；
4. 运行完全配对的 Baran-informed Prompt 对照；
5. 生成统计、成本、泄漏和覆盖审计；
6. 生成最终论文表格与图。

## 14. 与研究问题的对应关系

| 研究问题 | 当前状态 | 后续实验 |
|---|---|---|
| RQ1 Baran–LLM 互补性 | 已支持 | 多数据/Prompt 对照复现 |
| RQ2 Group 相对 singleton 的质量 | 未总体支持 | Phase 1、size/view 消融、确认性非劣检验 |
| RQ3 Structured 相对 random | 只有方向性结果 | view-specific matched random、跨 seed 复现 |
| RQ4 Group 成本效率 | 已支持但有 unknown usage | actual token/cell、延迟和 Pareto frontier |
| RQ5 Routeability | 有诊断信号 | nested CV、风险参数、更强 Router |
| RQ6 次模预算选择 | 部分支持 | optimizer/heuristic/random/singleton 基线 |
| No-Baran 信息边界的因果作用 | 尚未识别 | 完全配对 Baran-informed Prompt 对照 |

## 15. 完成检查清单

每阶段结束前检查：

- [ ] parent run 和既有正式结果未修改；
- [ ] 新 run/config/fingerprint 已冻结；
- [ ] 所有复用产物有 hash 与 provenance；
- [ ] outer target family 未进入训练或调参；
- [ ] target label/response 在 selection 前不可见；
- [ ] 每个变量只改变一个实验因素；
- [ ] 所有预算使用相同 reference cost；
- [ ] cache 只减少 physical calls，不减少 logical cost；
- [ ] 每个方法切片覆盖完整目标 cell 集；
- [ ] invalid/missing/abstain/failure 均保留在分母；
- [ ] per-dataset、Micro、Dataset-Macro、Family-Macro 和 Win/Tie/Loss 同时报告；
- [ ] 每个数据集均展示 Baran、BGR、配对差值和 95% CI；
- [ ] 同 family 数据集变体未被当作完全独立的确认样本；
- [ ] “BGR vs Baran”与“group-enabled vs singleton-only”使用独立证据和结论；
- [ ] 正结果和负结果同时保存；
- [ ] exploratory 与 confirmatory 结论没有混用；
- [ ] 所有统计检验、CI 和多重比较规则已记录；
- [ ] API 调用前完成 token/call 估算和 Prompt/model preflight；
- [ ] 最终报告明确哪些主张得到支持、部分支持或未支持。

## 16. 推荐的下一步起点

下一步应从 **Phase 1 的只读 Group failure attribution** 开始，而不是立即更换模型或追加 API 调用。该阶段能够使用现有 response、membership、prediction、selection 和 final records，成本低、可复核，并直接决定后续是：

```text
保留完整 group library
缩小到安全 size/view 子库
还是将主要系统收缩为 budgeted singleton Router
```

只有在完成归因后，`rho/gamma`、verifier 和更强 Router 的搜索空间才有明确依据。
