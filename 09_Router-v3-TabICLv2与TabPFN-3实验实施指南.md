# Router V3：TabICLv2 与 TabPFN-3 Backbone 实验实施指南

## 1. 文档用途

本文档是一份可以直接交给 Codex 实施的任务规范。目标是在当前 Router V3 基准上增加两个新的表格基础模型 backbone：

- `tabiclv2`：TabICLv2 classifier；
- `tabpfn3`：TabPFN-3 classifier。

这是一项受控 backbone 替换实验。除 Router backbone 外，训练标签、输入特征、base-family holdout、LOFO 不确定性、group candidates、预算、优化器、Verifier、Prompt、Baran fallback、LLM model 和最终指标定义都必须保持不变。

现有源 run：

```text
runs/no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all
```

该目录必须视为只读输入，不得改写、裁剪或删除其中任何文件。新实验不得依赖该 run 的 LightGBM/XGBoost `gates/`、`selections/` 或模型比较结果。

## 2. 已有实验事实与预期矩阵

源 run 应当提供：

- 14 个数据集的 23,957 条 Baran records；
- 8,197 个 calibration queries；
- 8,197 条 calibration execution records；
- 16,451 条 `(cell_id, query_id)` calibration pair labels；
- 9 个正式测试数据集、22,198 个 error cells；
- 全量 singleton LLM-only 响应以及当前 V3 已执行 group queries 的 response checkpoint。

新 revision 建议命名为：

```text
router_v3_foundation_models_exact_size_conditioned
```

新配置建议命名为：

```text
configs/experiment_router_v3_foundation_models.json
```

正式配置只包含：

```json
"gate_backends": ["tabiclv2", "tabpfn3"]
```

仍然运行五个 group-size variant：

| Variant | 训练和预测允许的 group sizes |
| --- | --- |
| `1` | `{1}` |
| `2` | `{1, 2}` |
| `4` | `{1, 4}` |
| `8` | `{1, 8}` |
| `all` | `{1, 2, 4, 8}` |

正式预算仍为 full-singleton estimated tokens 的 20%。完整运行应包含：

- `baran_only`；
- `llm_only`，保持纯 No-Baran singleton 定义；
- `budgeted_group_tabiclv2` 的五个 variant；
- `budgeted_group_tabpfn3` 的五个 variant。

若沿用现有完整 V3 矩阵，应得到 12 个方法切片、108 个 dataset slices、266,376 条 final cell records 和 90 个 selection slices。

## 3. 不可违反的实验约束

Codex 实施时必须遵守以下约束：

1. 不修改三个已有正式 V3 revision 的行为、配置或冻结结果。
2. 不删除 LightGBM、XGBoost、CatBoost 支持；新模型通过独立 revision/config 接入。
3. 不读取源 run 的 `gates/lightgbm`、`gates/xgboost`、旧 selections 或旧 backbone predictions 作为新模型输入。
4. 不使用 target family 的 clean labels、LLM responses 或最终指标进行训练、调参、checkpoint 选择、阈值选择或概率校准。
5. 不改变现有 25 个 `MODEL_FEATURE_COLUMNS`。
6. 不对 helpful/harmful labels 做过采样、下采样或 target-aware class balancing。
7. 不因显存不足静默减少训练行、改变 `n_estimators` 或删除 LOFO replicas。任何资源降级都必须成为新的显式配置/revision。
8. 不在实现和测试阶段调用 DeepSeek 或其他付费 API。
9. 不使用 TabPFN hosted API；本实验只允许本地 checkpoint 推理。切换到云 API 必须单独获得用户明确授权。
10. 不提交 `runs/`、模型 checkpoint、cache、token、环境文件或其他 secrets。

## 4. 实施前的源 run 审计

在修改代码前，Codex 应只读检查以下文件：

```text
run_manifest.json
input_data_manifest.json
baran/*.jsonl
groups/candidates/*.jsonl
groups/memberships/*.csv
llm/calibration_queries.jsonl
llm/calibration_execution.jsonl
llm/calibration_pair_labels.csv
llm/group_query_checkpoint.jsonl
llm/group_response_cache.jsonl
final/all_methods.jsonl
```

至少验证：

- Baran 文件覆盖 14 个数据集，合计 23,957 个唯一 cell IDs；
- calibration query 和 execution 均恰好为 8,197，且 `(query_id, prompt_hash)` 一一对应；
- calibration labels 恰好为 16,451，且 `(cell_id, query_id)` 无重复；
- calibration execution 的成功记录满足 model、Prompt schema 和 request identity；
- memberships 包含完整 `MODEL_FEATURE_COLUMNS`；
- 正式测试 universe 为 9 个数据集、22,198 个唯一 error cells；
- checkpoint 文件存在且可解析，但审计过程不得打印 response 正文或任何环境变量值。

源 run 的 manifest 可能仍引用旧 Baran/Router V2 parent。若这些父目录在新机器上不存在，旧 run 自身的严格 `validate-run` 可能失败。不要因此修改源 run 的 manifest。新实验只应严格校验并复制源 run 内已经物化的 Baran、calibration 和 response artifacts，并在新 run 中写入新的本地 provenance。

## 5. 代码接入设计

### 5.1 Backend 类型和 revision

扩展 `group_gate.py` 中的 backend 类型：

```text
catboost / lightgbm / xgboost / tabiclv2 / tabpfn3
```

为 foundation-model revision 增加精确 backend 校验：

```text
("tabiclv2", "tabpfn3")
```

不要放宽已有 revision 的 backend 约束。validator、report、method naming、expected matrix 和 metadata 校验需要识别新 revision，但必须继续拒绝未声明或混合错误的 backend 组合。

### 5.2 保持双 head 和 LOFO 语义

当前 `GroupUpliftGate` 的语义必须保持：

```text
q_helpful = P(executable LLM correct, Baran wrong)
q_harmful = P(executable LLM wrong, Baran correct)
net_gain = q_helpful - rho * q_harmful
sigma = full-family leave-one-family-out net-gain sample std, ddof=1
conservative_uplift = max(0, net_gain - gamma * sigma)
```

每个 full/LOFO fold 仍分别拟合 helpful 和 harmful 两个二分类 head。若某个 fold 的 label 只有一个类别，继续使用现有 constant-probability model，不调用 foundation model。

### 5.3 Foundation feature encoder

不要把类别列编码为普通连续浮点编号后再交给模型。新增一个 train-only、确定性的 foundation feature adapter，输出列顺序固定的 pandas DataFrame：

- 数值列保持数值 dtype，缺失值使用 train-only median 或交由模型原生缺失处理；
- `dirty_type`、`dirty_format`、`baran_type`、`baran_format`、`group_view` 等类别列保持 string/category 语义；
- test 中未见类别必须映射到固定 unknown token 或使用模型官方 unknown-category 路径；
- metadata 记录 feature names、categorical indices、缺失值策略和训练类别词表摘要；
- 不引入 `skrub` 或额外语义编码作为首个正式实验，因为那会同时改变特征表示和 backbone。

TabICLv2 可以从 pandas DataFrame 自动检测 categorical columns。TabPFN-3 应在当前安装版本支持时显式传入 categorical feature indices，避免把类别编号误认为连续变量。

### 5.4 统一 classifier adapter

两个新 backend 都应通过统一的小型 adapter 满足：

```text
fit(X_train, y_train)
predict_proba(X_test)
classes_
metadata()
```

不要在 Router 主流程中加入模型特有分支。模型特有的构造、device、checkpoint 和概率输出规范化应封装在 adapter/factory 内。

需要修正 package-version 映射：

```text
tabiclv2 -> tabicl
tabpfn3  -> tabpfn
```

不能直接调用 `package_version("tabiclv2")` 或 `package_version("tabpfn3")`。

### 5.5 配置建议

配置中显式冻结以下内容，实际构造参数必须以安装版本的官方签名为准；如果参数已经更名，Codex 应先检查签名并在 metadata 中记录最终值，不能静默忽略：

```json
{
  "foundation_backends": {
    "tabiclv2": {
      "checkpoint_filename": "tabicl-classifier-v2-20260212.ckpt",
      "checkpoint_path_env": "BGR_TABICL_MODEL_PATH",
      "allow_auto_download": false,
      "n_estimators": 8,
      "batch_size": 8,
      "kv_cache": false,
      "random_state": 42,
      "device": "auto"
    },
    "tabpfn3": {
      "checkpoint_filename": "tabpfn-v3-classifier-v3_20260506_ood.ckpt",
      "checkpoint_path_env": "BGR_TABPFN3_MODEL_PATH",
      "allow_auto_download": false,
      "n_estimators": 8,
      "random_state": 42,
      "device": "auto"
    }
  }
}
```

这里将 TabPFN-3 OOD checkpoint 预注册为主实验 checkpoint，因为 Router 协议本身是 strict base-family target zero-shot。不得在观察正式 target 结果后改选 checkpoint。若资源允许，可把官方 default 或 binary-specialized checkpoint 作为预先声明的敏感性实验，但必须使用不同 revision/config/run ID，不能选取其中最好结果冒充主实验。

Apple Silicon 上，TabICLv2 的 `mps` 需要显式指定；CUDA 环境用 `cuda`；CPU 仅用于小规模 smoke。正式 metadata 必须记录：

- package version；
- checkpoint filename 和 SHA-256；
- device、dtype/AMP、ensemble size、batch size；
- categorical feature indices；
- model constructor 的最终有效参数；
- wall time 和可获得时的 peak RAM/VRAM。

### 5.6 Checkpoint 生命周期

在正式训练开始前一次性解析 checkpoint 路径并计算 SHA-256。之后给每个 estimator 传入已有本地文件并设置 `allow_auto_download=false`，避免在数百个 full/LOFO classifier 实例中反复触发联网。

不得在 helpful/harmful head 之间复用 fitted estimator，因为两个 head 的 labels 不同。可以复用官方提供的只读 pretrained weight cache，但不能共享下游 fitted context 或 labels。

TabPFN 新版提供 batched probability inference；它可作为通过正确性验证后的性能优化，但不得在第一版中为追求速度改变模型输出。优化前后必须用固定小数据断言概率在允许的数值误差内一致。

## 6. 依赖与环境策略

当前项目要求 Python 3.10。优先在独立实验虚拟环境中安装 TabICL/TabPFN，避免破坏冻结 Router V3 环境。Codex 在安装新依赖前应展示：

- 计划安装的直接依赖和版本；
- 现有 NumPy、pandas、scikit-learn、PyTorch 是否会被升级或降级；
- 对 `requirements-lock.txt` 的影响；
- 是否可以在独立 `.venv` 中完成。

模型权重下载已获用户授权，但不得自动接受许可证、读取或打印用户 token。若 TabICL 与 TabPFN 的依赖无法在同一 Python 3.10 环境共存，停止并报告冲突；不要强行大版本升级基础环境。可提出两个独立 backend worker 环境、通过标准 CSV prediction artifacts 交接的备选方案，但实施前需用户确认。

建议新增可选 dependency group 或独立 lock，例如：

```text
requirements-table-foundation-lock.txt
```

基础安装不应因为缺少 `tabicl`/`tabpfn` 而失败。新依赖只在对应 backend 被选择时 lazy import，并给出明确安装错误。

## 7. 模型下载与人工上传回退

### 7.1 官方来源

只允许使用官方来源，不接受随机网盘、第三方镜像或来源不明的 `.ckpt`。PyTorch `.ckpt` 可能包含 pickle 数据，加载非官方文件存在代码执行风险。

TabICLv2：

- 代码与说明：https://github.com/soda-inria/tabicl
- checkpoint repository：https://huggingface.co/jingang/TabICL
- 文件：`tabicl-classifier-v2-20260212.ckpt`

TabPFN-3：

- 代码与说明：https://github.com/PriorLabs/TabPFN
- checkpoint repository：https://huggingface.co/Prior-Labs/tabpfn_3
- 主实验文件：`tabpfn-v3-classifier-v3_20260506_ood.ckpt`
- 许可证/登录入口：https://ux.priorlabs.ai

TabPFN-3 权重允许研究和内部评估，但使用前必须由用户自行阅读并接受对应许可证。Codex 不得替用户点击接受。

### 7.2 方法 A：由 agent 自动下载

安装依赖且完成许可条件后，agent 可以优先调用模型官方 auto-download：

```python
from tabicl import TabICLClassifier

model = TabICLClassifier(
    checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
    allow_auto_download=True,
    device="cpu",
    n_estimators=1,
    random_state=42,
)
```

TabPFN-3 可以使用当前官方 `TabPFNClassifier` 默认 V3 或显式 `model_path` 文件名触发官方下载。若出现浏览器登录或许可证页面，agent 必须暂停，让用户完成许可操作；不得尝试绕过。

下载成功后，将文件复制或链接到项目外部/被 Git 忽略的模型目录，例如：

```text
models/tabiclv2/tabicl-classifier-v2-20260212.ckpt
models/tabpfn3/tabpfn-v3-classifier-v3_20260506_ood.ckpt
```

随后计算 SHA-256，设置 `allow_auto_download=false`，用显式路径完成正式运行。

### 7.3 方法 B：Hugging Face CLI 下载

若自动下载失败，可由用户或 agent 在允许联网的终端运行：

```bash
hf download jingang/TabICL \
  tabicl-classifier-v2-20260212.ckpt \
  --local-dir models/tabiclv2

hf download Prior-Labs/tabpfn_3 \
  tabpfn-v3-classifier-v3_20260506_ood.ckpt \
  --local-dir models/tabpfn3
```

如需 Hugging Face 登录，用户应在自己的终端交互执行：

```bash
hf auth login
```

不要把 Hugging Face token 写进命令历史、Markdown、聊天、`.env` 示例或日志。网络较慢时可由用户设置 Hugging Face 官方支持的 download timeout 后重试，但 agent 不应关闭 TLS 校验。

### 7.4 方法 C：浏览器人工下载后放入项目

用户可以在浏览器打开上述两个 Hugging Face repository：

1. 找到精确文件名；
2. 确认 repository owner 分别为 `jingang` 和 `Prior-Labs`；
3. 阅读并接受 TabPFN-3 license；
4. 下载文件；
5. 放入以下目录：

```text
models/tabiclv2/tabicl-classifier-v2-20260212.ckpt
models/tabpfn3/tabpfn-v3-classifier-v3_20260506_ood.ckpt
```

若通过 Codex Desktop 上传，可把 checkpoint 作为任务附件上传，并明确告诉 agent 目标文件名。Agent 应先检查附件的实际路径、文件大小和 SHA-256，再逐文件移动到目标目录；不得覆盖已经存在且 hash 不同的文件。

### 7.5 方法 D：从另一台机器上传到计算服务器

如果模型在本地浏览器下载，而实验在远程 GPU 服务器运行，可由用户使用 `scp` 或 `rsync`：

```bash
scp /local/path/tabicl-classifier-v2-20260212.ckpt \
  user@server:/absolute/project/models/tabiclv2/

rsync -avP /local/path/tabpfn-v3-classifier-v3_20260506_ood.ckpt \
  user@server:/absolute/project/models/tabpfn3/
```

上传完成后在下载端和服务器端分别计算 SHA-256，二者必须一致。Agent 只记录 hash，不记录任何认证信息。

### 7.6 Git 忽略规则

在任何下载前，将模型目录和 checkpoint 扩展名加入 `.gitignore`，例如：

```gitignore
# Local tabular foundation-model weights
models/**
!models/.gitkeep
*.ckpt
*.safetensors
```

提交前必须确认 `git status` 中没有 checkpoint、Hugging Face cache、token 或 run artifacts。

## 8. 离线 calibration 复用要求

新 backbone 训练不应重新调用 DeepSeek。Codex 应实现一个严格的 calibration import/materialization 路径，可以新增 `--calibration-source-run`，或在不破坏现有语义的情况下扩展 `--response-reuse-run`。

离线导入必须：

1. 先由新 run 用当前代码重新生成 input manifest、Baran/cell features、candidate actions、memberships 和 calibration query plan；
2. 验证新旧 calibration `(query_id, prompt_hash)` 集合完全一致；
3. 验证 model、Prompt schema、provider request hash 和 execution coverage；
4. 从源 execution/checkpoint 导入 request-identical responses；
5. 使用新 run 的本地 calibration data 重新 materialize labels，或严格校验复制 labels 的 cell/query identity 与 hash；
6. 写入新 run 自己的 `provenance/calibration.json`，包含 source path、source manifest hash、query/execution/label hashes 和计数；
7. 将 calibration stage 标记为 complete，使 `train-router` 能够完全离线执行；
8. 不要求旧 V2 parent 存在，也不重写源 run manifest。

源 run 中的 `model_preflight.json` 与 Router backbone 无关。纯 Router 训练不得因为缺少 provider preflight 而发起网络请求。完整 BGR 执行前，可以严格复用 request-identical preflight receipt；若不能复用，只允许在用户明确授权后执行一次新的 preflight。

## 9. 分阶段实施顺序

### Phase 0：只读审计

- 检查 Git branch/status；
- 审计源 run 计数与 hashes；
- 检查磁盘空间、Python 3.10、PyTorch 和 CPU/MPS/CUDA；
- 输出实现计划和预计修改文件；
- 不安装依赖、不下载模型、不调用 API，直到相应授权条件满足。

### Phase 1：接口和单元测试

- 扩展 backend 类型、revision 和配置解析；
- 实现 foundation feature adapter；
- 实现 TabICLv2/TabPFN-3 classifier adapter；
- 使用 fake classifier 测试，不要求真实 checkpoint；
- 保证旧 44 项或当前完整测试集继续通过。

### Phase 2：本地 checkpoint smoke

对每个真实模型只运行一个很小的二分类 smoke：

- 至少包含一个 numeric、一个 categorical、missing 和 unknown category；
- `fit` 成功；
- `predict_proba` shape 为 `(n_test, 2)`；
- `classes_` 包含 `0/1`；
- 所有概率有限且在 `[0, 1]`；
- 每行概率和接近 1；
- 相同 seed 重复运行在声明的容差内一致。

CPU smoke 可以使用 `n_estimators=1`，但正式配置仍为 8；smoke 参数不得写入正式实验 metadata。

### Phase 3：单 fold 离线 Router smoke

使用源 run calibration artifacts，选一个 target 和 `all` variant，分别运行 TabICLv2 和 TabPFN-3：

- 保持完整 train rows，不得抽样；
- 运行 full helpful/harmful heads；
- 至少运行并验证两个 LOFO replicas，随后再验证完整 LOFO 集合；
- 生成 `q_helpful`、`q_harmful`、`net_gain`、`sigma`、`conservative_uplift`；
- 检查 target family/cell/row/query/group-signature overlap 均为零；
- 记录时间和内存；
- 不执行 selected LLM。

### Phase 4：完整离线 Router 矩阵

在资源允许时运行：

```text
2 backends × 5 variants × 9 formal targets
```

并生成 90 个 prediction/selection slices。Router diagnostics 使用相同的新 backend，不得从旧 LightGBM/XGBoost diagnostics 复制数值。

优先比较：

- helpful AUPRC 和 Brier；
- harmful AUPRC 和 Brier；
- top-10% helpful/harmful prevalence；
- LOFO sigma 分布；
- conservative-uplift 为零的比例；
- selection 的 predicted gain、group-size 构成和预算利用率；
- wall time、RAM/VRAM。

普通 accuracy 不是主要指标，因为 Router 使用的是概率差和不确定性下界。

### Phase 5：完整 BGR 前的 response 缺口计划

训练和 selection 完成后，先生成只读 dry plan：

- 新两个 backend/五个 variant 的 selected query union；
- LLM-only singleton union；
- 源 `group_query_checkpoint.jsonl` 的严格 request-identity cache hits；
- missing requests；
- missing requests 的 estimated token upper bound；
- 是否存在 frozen terminal failures。

该步骤必须保证零 provider calls。若当前 CLI 无法做到，应新增明确的 `plan-router-bgr` 或等价 `--dry-run`，而不是调用会直接发送请求的 `run-router-bgr`。

如果 missing requests 大于零，Codex 必须停下并向用户报告数量、估计 tokens 和建议 token cap。只有用户明确批准后，才能使用环境文件执行缺失请求。环境文件中的密钥不得读取、打印或提交。

### Phase 6：完整 BGR、验证和报告

获得授权后才可运行缺失 LLM queries。建议命令骨架：

```bash
SOURCE_RUN=runs/no_baran_router_v3_deepseek_v4_20260725_budget20_k1248_all
RUN_ID=no_baran_router_v3_foundation_models_deepseek_v4_budget20_k1248_all

python -m budgeted_group_repair_no_baran.cli plan-router-run \
  --run-id "$RUN_ID" \
  --experiment-config configs/experiment_router_v3_foundation_models.json \
  --baran-source-run "$SOURCE_RUN" \
  --response-reuse-run "$SOURCE_RUN"

# 该阶段必须使用严格离线 calibration import，不调用 provider。
python -m budgeted_group_repair_no_baran.cli train-router \
  --run-id "$RUN_ID" \
  --resume \
  --experiment-config configs/experiment_router_v3_foundation_models.json

# 先执行新增加的 dry plan；命令名以最终实现为准。
python -m budgeted_group_repair_no_baran.cli plan-router-bgr \
  --run-id "$RUN_ID" \
  --resume \
  --experiment-config configs/experiment_router_v3_foundation_models.json
```

不要传入 `--router-artifact-reuse-run`，因为新 backbone 必须重新训练。正式 provider 命令、token cap 和环境文件应在 dry plan 后由用户单独批准。

最后运行：

```bash
python -m budgeted_group_repair_no_baran.cli finalize-run \
  --run-id "$RUN_ID" --resume

python -m budgeted_group_repair_no_baran.cli validate-run \
  --run-id "$RUN_ID"

python -m budgeted_group_repair_no_baran.cli report \
  --run-id "$RUN_ID" --resume
```

## 10. 测试要求

至少新增或更新以下测试：

1. 两个 backend 的 lazy import；基础环境缺包时旧测试仍能运行。
2. 模型路径不存在且禁止 auto-download 时给出明确错误。
3. checkpoint filename/hash 与配置不一致时拒绝运行。
4. numeric/categorical/missing/unknown-category 编码稳定。
5. 禁止特征仍被 `_FORBIDDEN_FEATURES` 拒绝。
6. 两个 backend 都返回合法 positive-class probability。
7. 单类别 fold 仍使用 constant model。
8. full + LOFO replicas 数量、family-left-out identity 和 `ddof=1` 不变。
9. 新 revision 只接受 `tabiclv2 + tabpfn3`。
10. 三个旧 revision 的 backend 约束和 validator 行为不变。
11. 离线 calibration import 的 query/hash/model/schema/label coverage。
12. 源 run 缺文件、计数错误、hash 错误时 fail closed。
13. 新 experiment 不读取旧 `gates/` 和 `selections/`。
14. dry plan 为零 provider calls，并准确报告 cache hits/misses/token upper bound。
15. 新方法命名、90 个 selection slices、预算和 leakage audit。
16. report/validator 能处理新 backend metadata。

验证命令至少包括：

```bash
export PYTHONPATH=src
python -m pytest
python -m compileall -q src tests
python -m budgeted_group_repair_no_baran.cli validate-data
python -m budgeted_group_repair_no_baran.cli --help
```

真实 checkpoint smoke 应做成显式 opt-in 测试，不应让普通 `pytest` 自动联网或下载权重。

## 11. 正式验收标准

代码实现完成必须同时满足：

- 源 run 完全未修改；
- 旧 Router V3 测试全部通过；
- 新 backbone 不读取旧 backbone artifacts；
- 两个真实模型 smoke 通过；
- calibration 完全离线复用，计数仍为 8,197 queries / 16,451 labels；
- 所有正式 split 的 base-family/cell/row/query/group-signature overlap 为零；
- 每个 selection 不超过 20% logical token budget；
- foundation checkpoint、package 和推理参数 provenance 完整；
- dry plan 在任何 LLM 调用前完成；
- 未经批准没有付费 API calls；
- 完整实验完成时 final records、metrics、report 能被 validator 独立重算；
- Git 中没有 run artifacts、checkpoint、cache 或 secrets。

## 12. Codex 最终交付格式

Codex 完成实现后应报告：

1. 修改和新增了哪些文件；
2. 两个模型的实际 package version、checkpoint filename、SHA-256 和下载方式；
3. 使用的硬件和实际有效推理参数；
4. 源 run 审计计数；
5. 单 fold 与完整离线 Router 结果；
6. response dry plan 的 cache hit、missing request 和 token upper bound；
7. 运行了哪些测试及结果；
8. 是否发生任何 provider call；
9. 尚未完成的正式实验步骤和原因；
10. `git status` 与 `git diff --stat`，等待用户确认后才能 commit/push。

不要只报告“模型训练成功”。必须同时报告概率质量、LOFO 不确定性、selection budget、leakage、资源消耗和可复现 provenance。

## 13. 官方参考

- TabICLv2 官方实现：https://github.com/soda-inria/tabicl
- TabICLv2 API：https://tabicl.readthedocs.io/en/latest/api.html
- TabICLv2 checkpoint：https://huggingface.co/jingang/TabICL
- TabPFN 官方实现：https://github.com/PriorLabs/TabPFN
- TabPFN-3 checkpoint：https://huggingface.co/Prior-Labs/tabpfn_3
- TabPFN-3 技术报告：https://priorlabs.ai/technical-reports/tabpfn-3
- Hugging Face CLI：https://huggingface.co/docs/huggingface_hub/guides/cli
