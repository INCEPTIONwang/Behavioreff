# RLinf Documentations

Welcome to the documentation for RLinf! This README provides detailed instructions on how to generate the project documentation locally using Sphinx. It covers the entire process, from setting up your environment to building and viewing the documentation. Additionally, it includes information on cleaning the build directory and an introduction to Sphinx and reStructuredText (RST).

---

## Setting Up Your Environment

### Step 1: Set Environment Variables

Every time you open a new terminal session to work on the documentation, run these commands to set the locale for Sphinx:

```bash
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

These ensure proper character encoding with the `C.UTF-8` locale.

**Note**: Repeat this step for every new terminal session before building the documentation.

### Step 2: Install Dependencies

```bash
cd <path_to_RLinf_project_root>  # Navigate to the RLinf project root directory
bash requirements/install.sh docs --venv .docs-venv
source .docs-venv/bin/activate
```

---

## Building the Documentation

With your environment ready, build the documentation using Sphinx. Source files are in the `source` directory, and output HTML files go to `build/html`.

You can simply run the following command to build the English docs and open a server for preview and live reloading:
```bash
bash autobuild.sh
```
To build the Chinese docs, run this:

```bash
bash autobuild.sh zh
```

To build without running the preview server, run this command:
```bash
sphinx-build source-en build/html # change to source-zh for Chinese docs
```

---

## Viewing the Documentation

After building, view the documentation in your browser.

### With `sphinx-build`

1. Go to the `build/html` directory.
2. Open `index.html` in your browser.

Or, serve it with a Python HTTP server:

```bash
cd build/html
python -m http.server 8000
```

Visit `http://localhost:8000` in your browser.

### With `sphinx-autobuild`

Running `sphinx-autobuild` automatically hosts the documentation at `http://localhost:8000`. Open this URL to view it with live reloading.

---

## Cleaning the Build Directory

To remove generated files and start fresh, clean the build directory:

```bash
make clean
```

This deletes the `build` directory and its contents.

---

## Writing reStructuredText (RST)

Sphinx uses reStructuredText (RST), a simple yet powerful markup language for documentation.

[RST grammer](https://zh-sphinx-doc.readthedocs.io/en/latest/rest.html)

---

## Rollout→训练（OpenPI 动态 Horizon 5/10/15）函数与数据流说明

本节是“可直接阅读的 Markdown 说明”，用于解释你这次在具身训练里引入的动态规划长度（`rollout.action_horizons_pattern: [5,10,15]`）之后，从 rollout 交互到 actor 训练的端到端数据流向、关键函数职责，以及近期相关改动点与原因。

### 配置与关键概念

以 [libero_spatial_grpo_openpi_pi05.yaml](file:///mnt/43t/wxh/RLinf/examples/embodiment/config/libero_spatial_grpo_openpi_pi05.yaml) 为例：

- `actor.model.num_action_chunks = 5`：每次与环境交互发送 5 步动作（chunk）。
- `rollout.action_horizons_pattern = [5, 10, 15]`：为 batch 内每个样本分配“规划长度”H（单位=动作步），按 pattern 循环分配。
- `algorithm.group_size = 9`：同一 prompt 的采样组大小（用于 GRPO / group-based advantage）。
- `plan_reward_coef/base_h`：只对成功轨迹给的额外 plan 奖励，按 `H/base_h` 缩放，并在 score 阶段累加到最终奖励（受 `use_plan_reward` 开关控制）。

### 端到端数据流（从 env 到训练）

#### 1) Rollout Worker：生成动作、与环境交互、缓存 forward_inputs

入口： [MultiStepRolloutWorker.generate](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L207-L370)

每个 chunk-step 的核心流程（按实现顺序）：

- 接收环境 batch：`env_output = await recv_env_output(...)`（含 `obs/states`、`rewards/dones/...`）
- 预处理观测：`extracted_obs = hf_model.preprocess_env_obs(env_output["obs"])`
- 分配每个样本的 horizon（H）：
  - `pattern.repeat(reps)[:bsz]` 得到 `horizons: [bsz]`
  - 代码参考：[huggingface_worker.py:L251-L260](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L251-L260)
- 规划缓存 plan_cache：
  - `offsets: [bsz]` 表示每个样本在 plan 内已经消费的步数
  - `need_new = offsets >= horizons` 触发重规划
  - 参考：[huggingface_worker.py:L257-L276](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L257-L276)
- 需要重规划时调用模型生成“完整 plan”（长度=各自 H）：
  - `predict_action_batch(..., action_horizons=horizons.tolist(), return_full_plan=True)`
  - 参考：[huggingface_worker.py:L261-L272](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L261-L272)
- 从完整 plan 中逐样本切出本次 chunk：
  - `s = offsets[i]`，`e = s + chunk_size`
  - 同时切出对应 `prev_logprobs`（并裁剪到 `action_env_dim`）
  - 参考：[huggingface_worker.py:L277-L305](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L277-L305)
- 发送动作前做 `output_transform`（需要 state 以做缩放）：
  - `state_for_out = env_output["obs"].get("states", None)` + fallback
  - 参考：[huggingface_worker.py:L288-L303](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L288-L303)
- 组织并写入 `forward_inputs`（训练重算 logprob 需要）：
  - `finputs["action"]`：chunk 动作展平后写入
  - `finputs["chunk_offset"]`：该样本 chunk 在完整 plan 中的起始偏移（逐样本）
  - `finputs["plan_horizon"]`：该样本的 H，用于 plan_reward 与分档成功率
  - 参考：[huggingface_worker.py:L306-L319](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L306-L319)

额外一步（bootstrap value）：

- rollout epoch 结束时会额外跑一次 `predict(extracted_obs)`，只为拿到 bootstrap 用的 `prev_values`，并把最后一步的 `dones/rewards` 等写入 buffer
- 参考：[huggingface_worker.py:L331-L367](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L331-L367)

#### 2) Rollout Buffer：ChunkStepResult / EmbodiedRolloutResult 的组织方式

数据结构定义：

- [ChunkStepResult](file:///mnt/43t/wxh/RLinf/rlinf/data/io_struct.py#L1245-L1270)
- [EmbodiedRolloutResult](file:///mnt/43t/wxh/RLinf/rlinf/data/io_struct.py#L1273-L1386)

关键点：

- `ChunkStepResult.__post_init__` 会把字段搬到 CPU 并 contiguous，保证跨 worker 传输稳定。
- `EmbodiedRolloutResult.to_dict()` 会 `stack` 出形状约为：
  - `prev_logprobs/rewards`: `[rollout_epoch*n_chunk_steps, bsz, chunk_size, ...]`
  - `dones/prev_values`: `[rollout_epoch*(n_chunk_steps+1), bsz, chunk_size, ...]`
- `forward_inputs` 会被 `stack_list_of_dict_tensor` 合并到顶层 batch dict（即 actor 侧收到的 batch 顶层直接包含 `forward_inputs` 的各个 key）。

#### 3) Actor Worker：接收 batch、形状重排、优势/回报、rollout 指标

入口：[FSDPActor.recv_rollout_batch](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L812-L832)

流程：

- 从 channel 收集分片，然后 `cat_list_of_dict_tensor(recv_list, dim=1)` 拼到 batch 维
- 参考：[cat_list_of_dict_tensor](file:///mnt/43t/wxh/RLinf/rlinf/utils/nested_dict_process.py#L92-L108)

预处理：[_process_received_rollout_batch](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L833-L920)

- `process_nested_dict_for_adv` 把 `[rollout_epoch*n_chunk_steps, bsz, ...]` 重排为 `[n_chunk_steps, rollout_epoch*bsz, ...]`
  - 参考：[process_nested_dict_for_adv](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L70-L88)
- 始终根据 `dones` 计算 `loss_mask/loss_mask_sum`（保证统计与优势计算稳定）
  - 参考：[fsdp_actor_worker.py:L843-L855](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L843-L855)
- 将 `forward_inputs.plan_horizon` 提升为顶层 `rollout_batch["plan_horizon"]` 并对齐形状
  - 参考：[fsdp_actor_worker.py:L856-L870](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L856-L870)
- 可选 reward filter：按 prompt 分组过滤轨迹，将掩码合入 `loss_mask`
  - 参考：[fsdp_actor_worker.py:L872-L919](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L872-L919)

优势与指标：

- 计算优势入口：[compute_advantages_and_returns](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L922-L987)
- 优势预处理/打分/后处理在：[preprocess_embodied_advantages_inputs](file:///mnt/43t/wxh/RLinf/rlinf/algorithms/utils.py#L67-L169)、[calculate_scores](file:///mnt/43t/wxh/RLinf/rlinf/algorithms/utils.py#L172-L355)、[postprocess_embodied_advantages_outputs](file:///mnt/43t/wxh/RLinf/rlinf/algorithms/utils.py#L358-L408)
- rollout 指标聚合在：[compute_rollout_metrics](file:///mnt/43t/wxh/RLinf/rlinf/utils/metric_utils.py#L155-L192)

plan_reward 与分档成功率产生位置（关键逻辑）：

- `plan_horizon` 必须存在：`preprocess_embodied_advantages_inputs` 会严格检查并构造 `plan_horizon_steps: [n_steps, bsz]`
- `calculate_scores` 在 score 阶段：
  - 成功轨迹才给 plan_reward：`success_mask = (rewards>0).any(dim=0)`
  - 在 EOS 位置取出 `h_at_eos`，按 `h_at_eos/base_h` 缩放得到 plan_term，并加到最终 `scores`
  - 产出 `plan_h5_success_count/plan_h5_total_count/...` 等计数，用于日志计算 `plan_success_rate_h5/h10/h15`

#### 4) 训练循环：shuffle → global/micro batch 拆分 → 重算 logprob → loss/optimizer

训练入口：[FSDPActor.run_training](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L988-L1160)

- `process_nested_dict_for_train` 把 `[num_chunk, B, ...]` flatten 到 `[N, ...]` 并 shuffle
  - 对 `dones/terminations/truncations/prev_values` 去掉最后一步（bootstrap 帧不参与训练）
  - 参考：[process_nested_dict_for_train](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L91-L113)
- `get_iterator_k_split` 做 global batch 与 micro batch 拆分
  - 只支持可拆分的 tensor/list，其它键会 warning 丢弃（你看到的 plan_* warning 来源于此）
  - 参考：[data_iter_utils.py:L177-L195](file:///mnt/43t/wxh/RLinf/rlinf/utils/data_iter_utils.py#L177-L195)
- 模型 forward 时把 `chunk_offset=data.get("chunk_offset", 0)` 传入，使训练侧能重算出与 rollout 同一个 chunk 的 logprob
  - 参考：[fsdp_actor_worker.py:L1083-L1091](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L1083-L1091)
  - OpenPI 训练 forward 为：[default_forward](file:///mnt/43t/wxh/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py#L271-L340)

### 近期改动点与原因（聚焦动态 horizon 相关）

#### Rollout 侧（动态规划与交互）

- 新增 per-sample `horizons` 分配与 `plan_cache`（actions/logprobs_full/offsets）
  - 解决：同一 batch 内不同样本需要不同规划长度；同时避免每个 chunk-step 都全量重规划
  - 代码： [huggingface_worker.py:L218-L319](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L218-L319)
- 仅向 env 发送 `action_env_dim`，避免把 32 维 full action 误传到只需要 7 维的环境
  - 代码： [huggingface_worker.py:L285-L289](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L285-L289)
- `output_transform` 明确传 `state` 并做 fallback，避免尺度错误和 `KeyError: 'observation/state'`
  - 代码： [huggingface_worker.py:L288-L303](file:///mnt/43t/wxh/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py#L288-L303)

#### 模型侧（动态 horizon 推理与训练重算）

- `predict_action_batch` 按 horizon 分组多次调用 `sample_actions`，再合并回 batch
  - 原因：一次 forward 只能生成一个固定长度序列，必须按 horizon 分组
  - 代码： [predict_action_batch](file:///mnt/43t/wxh/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py#L385-L492)
- `get_suffix_out` 对齐 suffix pad/att mask 长度并按 action_horizon 裁剪输出
  - 解决：attention mask 维度不一致导致的 RuntimeError
  - 代码： [get_suffix_out](file:///mnt/43t/wxh/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py#L731-L788)
- `default_forward` 支持逐样本 `chunk_offset` tensor，并按样本切片 logprob/entropy
  - 解决：`TypeError: only integer tensors of a single element can be converted to an index`
  - 代码： [default_forward](file:///mnt/43t/wxh/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py#L271-L340)

#### Actor/优势/日志侧（plan_reward 与分档成功率）

- `plan_reward` 在 `calculate_scores` 阶段累加到 `scores`（与 eff_reward 同层级影响 advantage），成功轨迹才给
  - 代码： [calculate_scores](file:///mnt/43t/wxh/RLinf/rlinf/algorithms/utils.py#L172-L237)
- 严格要求 `plan_horizon` 必须存在，并构造 `plan_horizon_steps`（EOS 取值用）
  - 代码： [preprocess_embodied_advantages_inputs](file:///mnt/43t/wxh/RLinf/rlinf/algorithms/utils.py#L67-L152)
- 增加 `plan_success_rate_h5/h10/h15` 的分布式聚合（由 success/total 计数归约）
  - 代码： [metric_utils.py:L181-L192](file:///mnt/43t/wxh/RLinf/rlinf/utils/metric_utils.py#L181-L192)

#### 为什么之前训练阶段也会出现 plan_* warning（以及如何修复）

- warning 来源：微批拆分器只支持拆分 batch 维 tensor/list，遇到标量统计键会丢弃并 warning
  - 代码： [data_iter_utils.py:L177-L195](file:///mnt/43t/wxh/RLinf/rlinf/utils/data_iter_utils.py#L177-L195)
- 本次修复：在 `compute_rollout_metrics` 聚合完日志后，从 `rollout_batch` 中移除 plan_* 这类纯日志键，避免进入训练 micro-batch 拆分
  - 代码： [fsdp_actor_worker.py:L963-L986](file:///mnt/43t/wxh/RLinf/rlinf/workers/actor/fsdp_actor_worker.py#L963-L986)
