---
title: "主流后训练框架对比总结（源码级）"
date: 2026-09-19
permalink: /posts/2026/09/post-training-frameworks/
excerpt: "对 veRL、TRL、DeepSpeed、OpenRLHF、slime、NeMo-RL 六个主流后训练框架做源码级横向对比：推理引擎选型、训练并行路线、同步与异步、权重同步难题、多轮 TITO 与 MoE 路由回放，并附 Miles v0.1 的谱系对照。"
tags:
  - Post-Training
  - RL
  - veRL
  - DeepSpeed
  - TRL
toc: true
toc_sticky: true
---
{% raw %}
> **本文范围**：对 6 个主流开源后训练框架做源码级横向对比，并以本工作区已有的 **Miles v0.1** 解析为参照系。
>
> **方法说明**：所有结论以**读源码**为准，不采信 README 宣传语。凡 README 与代码不符之处，以代码为准并单独标注。每份单框架解析文档都给出了 `文件路径::符号名` 级证据。
>
> **快照信息**（均为浅克隆，只含 1 个 commit，因此**无法做历史演化分析**，只能描述「当前形态」）：

| 框架 | 仓库 | commit | 版本 | 协议 |
| --- | --- | --- | --- | --- |
| **Miles**（参照） | `radixark/miles` | `327eefa0f` (2026-09-09) | 0.1.0 | Apache-2.0 |
| **slime** | `THUDM/slime` | `4c193f1` (2026-09-03) | 0.3.2 | Apache-2.0 |
| **veRL** | `volcengine/verl` | `10db40d` (2026-09-11) | 0.10.0.dev | Apache-2.0 |
| **OpenRLHF** | `OpenRLHF/OpenRLHF` | `54e3036` (2026-09-13) | 0.11.2 | Apache-2.0 |
| **TRL** | `huggingface/trl` | `cd2c528` (2026-09-12) | 1.14.0.dev0 | Apache-2.0 |
| **NeMo-RL** | `NVIDIA-NeMo/RL` | `88ee6c1` (2026-09-12) | 0.6.0 | Apache-2.0 |
| **DeepSpeed** | `deepspeedai/DeepSpeed` | `b5e000c` (2026-09-13) | 0.19.6 | Apache-2.0 |

**单框架解析文档**（同目录）：
`框架解析_01_veRL.md` · `框架解析_02_TRL.md` · `框架解析_03_DeepSpeed.md` · `框架解析_04_OpenRLHF.md` · `框架解析_05_slime.md` · `框架解析_06_NeMo-RL.md`

---

## 目录

- [1. 一句话结论](#1-一句话结论)
- [2. 总表](#2-总表)
- [3. 先分清阵营：这六个不是同类产品](#3-先分清阵营这六个不是同类产品)
- [4. 十个关键分歧点](#4-十个关键分歧点)
- [5. Miles 的谱系：与 slime 的逐条对照](#5-miles-的谱系与-slime-的逐条对照)
- [6. 选型建议](#6-选型建议)
- [7. 方法附录：本次调查如何做的（可复现）](#7-方法附录本次调查如何做的可复现)

---

## 1. 一句话结论

1. **这不是「六个同类产品比高低」，而是三个不同物种。** DeepSpeed 是训练加速**库**（不做 RL），TRL 是训练器**库**（不做编排与推理服务），另外四个（veRL / NeMo-RL / slime / Miles / OpenRLHF）才是**完整的 RL 系统**。把它们放在一张表里比"功能多少"会产生误导，必须先分层。
2. **完整 RL 系统内部的分水岭是「谁提供推理引擎」和「谁提供训练并行」。** 这两件事都不是 RL 框架自己发明的——它们全都建立在 SGLang / vLLM / Megatron / FSDP / DeepSpeed 之上，差异在于**接线方式**。
3. **「TITO / 多轮 token 保真」是这个赛道最有区分度的技术点**，因为它是 LLM RL 独有的难题，且没有标准答案。七个框架给出了**四种不同的回答**（§4.5），而没有一家做到 Miles 那种"服务端拥有 tokenization"的形态。
4. **异步（生成与训练真正并行）不是 Miles 独有的**，但它在这个赛道上的成熟度差异极大：NeMo-RL 有一套 20 万行级的工业级异步子系统；veRL 把它放在 `experimental/` 下；Miles 用约 19 KB 实现了核心机制；TRL / OpenRLHF 基本没有。
5. **Miles 是 slime 的演化分支，而不是独立发明。** slime 自己的 README 就写明 Miles「built on slime」，且 slime 已具备三阶段循环、R3、TIS/ice-pop、disk-delta、colocate-IPC、四旋钮恒等式等。Miles 的真实增量收敛为**四束**（§5）。而且 Miles **并非 slime 的严格超集**——它丢掉了 slime 的 `cispo`。

---

## 2. 总表

### 2.1 定位与依赖

| | **Miles** | **slime** | **veRL** | **OpenRLHF** | **TRL** | **NeMo-RL** | **DeepSpeed** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **物种** | 完整 RL 系统 | 完整 RL 系统 | 完整 RL 系统 | 完整 RL 系统 | **训练器库** | 完整 RL 系统 | **加速库** |
| **推理引擎** | SGLang（唯一） | SGLang（唯一） | vLLM / **SGLang** / TensorRT-LLM | vLLM（唯一） | `transformers.generate` / vLLM | vLLM / **SGLang**¹ / TRT-LLM / Megatron-Inf / Dynamo | ZeRO-Inference |
| **训练并行来源** | Megatron-LM / **FSDP2** | Megatron-LM（唯一） | Megatron / FSDP | **DeepSpeed ZeRO** | `accelerate` | Megatron-Core / FSDP2 | 自身 |
| **编排** | Ray | Ray | Ray（single-controller） | Ray（single-controller） | **无**（`accelerate`） | Ray | `torch.distributed` |
| **配置范式** | argparse flag | argparse flag | **Hydra YAML + dataclass** | argparse（**点号命名空间**） | `TrainerConfig` dataclass | **YAML + dataclass** | **JSON/hjson 配置文件** |

### 2.2 规模、成熟度与能力

| | **Miles** | **slime** | **veRL** | **OpenRLHF** | **TRL** | **NeMo-RL** | **DeepSpeed** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **核心包代码量** | 69.5 K 行 / 440 文件 | 35.9 K / 153 | **112.8 K / 366** | **12.0 K / 53** | 66.0 K / 154 | **161.9 K / 348** | 123.7 K / 647 |
| **唯一 CLI flag 数** | 358 | 340 | 107 | 180 | 136 | 239 | （JSON 配置，非 flag） |
| **Ray 引用文件数** | — (22 同级 slime) | 22 | 95 | 12 | **0** | **149** | 0 |
| **同步训练** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| **生成-训练重叠** | ✅ 一等公民 | ✅（rollout 函数） | ⚠️ `experimental/` | ✅（第二实现） | ⚠️ `experimental/` | ✅ 大型子系统 | ✗ |
| **colocate（共用 GPU）** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| **R3 路由回放** | ✅ | ✅ | ✅ | ✗ | ✗ | ✅ | ✗ |
| **TITO 多轮保真** | ✅ 服务端拥有 | ⚠️ 容错式 | ⚠️ server 级 | ⚠️ executor 级 | ✗ | ⚠️ 默认休眠 | ✗ |
| **权重同步模式数** | 4（含 **RDMA P2P**） | 4（无 P2P） | 分桶 + delta（ZMQ/IPC） | 2（IPC + NCCL） | 仅 `experimental/` | 独立模块（多种） | — |
| **LoRA** | ✅ + multi-LoRA | ✗ | ✅ | ✅ | ✅（PEFT） | ✅² | ✗ |
| **advantage estimator** | 5 | **6（含 cispo）** | **7+** | 3+ | — | 3+ | — |
| **算法族** | GRPO/GSPO/RF++/PPO | +cispo | +RLOO/ReMax/OPO/GAE | PPO/GRPO/GSPO/RLOO | SFT/DPO/GRPO/KTO/RLOO/RM/Distill（**PPO 已移除**） | GRPO/PPO/DPO/SFT/RM/OPD/MPO/Distill | ✗ |
| **前沿规模实证** | 744B（GLM-5.2） | 744B / 1T（GLM-5.2、Kimi-K2 脚本） | ~30B 级为主 | ≤70B | ≤8B 级 | **235B / DeepSeek-V3** | — |

> **¹ ² 两处脚注：NeMo-RL 的能力带前置条件，不能按粗粒度读。** `nemo_rl/models/policy/lm_policy.py` 里有一段相邻的约束：
>
> ```python
> assert (config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False) is False
>     ), "LoRA is not supported for DTensorPolicyWorker V1"
> if (config.get("generation") or {}).get("backend") == "sglang":
>     raise ValueError(
>         "policy.generation.backend='sglang' requires "
>         "policy.dtensor_cfg._v2=true or policy.megatron_cfg.enabled=true; "
>         "DTensorPolicyWorker V1 does not implement the SGLang refit path."
>     )
> ```
>
> 也就是说 **NeMo-RL 的 LoRA 与 SGLang 后端都要求走 DTensor V2 或 Megatron**，DTensor V1 两条都不支持。这类"支持但带前置条件"的情况在七个框架里普遍存在——**这是本文反复强调「粗粒度的有/没有结论几乎总是错的」的又一个实例**，而这一次粗粒度的是我自己初版的总表。

> 表中「—」表示该维度对 DeepSpeed 不适用（它不是 RL 框架）。**代码量仅供感受体量差异，不等于质量或能力**：OpenRLHF 用 1.2 万行实现了完整的四模式 RLHF，而 NeMo-RL 用 16 万行。

---

## 3. 先分清阵营：这六个不是同类产品

### 阵营 A：训练加速库 —— DeepSpeed

**它不做 RL 算法，但它比"纯底座"多一层。** 需要说清三件事：

1. **提供 RL 框架需要但不自己实现的那些东西**：ZeRO 分片、CPU/NVMe offload、混合并行、通信压缩。
2. **库内确实有 HybridEngine**：`deepspeed/runtime/hybrid_engine.py::DeepSpeedHybridEngine`（容易误以为它在 DeepSpeedExamples）。
3. ⚠️ **库内还意外地有一小套 rollout 子系统**：`deepspeed/runtime/rollout/`，含 `RolloutEngine`（ABC）与 `HybridEngineRollout`，另有 `continuous_batching.py`。但它的权重同步是**有文档的 no-op**：

   ```python
   def sync_weights(self, step: int) -> None:
       """Push updated weights into the rollout backend.

       No-op when the rollout engine is co-located with the training engine
       (e.g. hybrid engine shares weights directly).
       """
   ```

   **这句话与 TRL 的默认模式是同一个世界观**（§4.4）：不传输，直接共享。

**关键事实**：**DeepSpeed-Chat（真正的 RLHF 训练系统）不在本仓库**，它在 `DeepSpeedExamples/applications/DeepSpeed-Chat/`。证据是 DeepSpeed 仓库里仍挂着一条**跨仓库 CI**：`.github/workflows/nv-ds-chat.yml` 会 clone DeepSpeedExamples 并跑 stage1/stage2/stage3 的测试。

**推论**：拿 DeepSpeed 和 Miles 比"RL 功能"是范畴错误。它的正确对照物是 **Megatron-LM / FSDP2**——也就是 Miles 的**训练后端**。事实上 OpenRLHF 就是「DeepSpeed 当后端 + vLLM 当 rollout」的组合。

### 阵营 B：训练器库 —— TRL

**它有 RL 算法，但没有 RL 系统。** 核心事实：

- 所有训练器继承 `transformers.Trainer`（`trl/trainer/base_trainer.py::_BaseTrainer`），分布式完全交给 `accelerate`。
- **Ray 引用文件数 = 0**。没有独立服务进程、没有引擎编排。
- 公开 API 只有 7 个训练器（SFT / DPO / GRPO / RLOO / KTO / Reward / Distillation），其余移到 `trl/experimental/`。
- ⚠️ **本快照中 PPO 已被移除**：全仓库对 `PPOTrainer` / `PPOConfig` 的引用为零，`PPO` 字样只剩一个被 GRPO 复用的 fused 内核基类 `FusedLinearPPOBase`。

它的 generation 与 training **在同一进程**里完成（vLLM 可作可选后端，也有 `trl vllm-serve` 的 server 模式）。这带来了极低的使用门槛，也带来了明确的上限。

### 阵营 C：完整 RL 系统 —— veRL / NeMo-RL / OpenRLHF / slime / Miles

这五个才是可直接比较的对象。它们共享同一个骨架：

```
     ┌──────────────┐   prompts    ┌──────────────┐
     │  推理引擎     │ ←─────────── │   数据源      │
     │ SGLang/vLLM  │              └──────────────┘
     └──────┬───────┘
            │ trajectories + logprobs
            ▼
     ┌──────────────┐   rewards    ┌──────────────┐
     │   reward/验证 │ ←─────────── │   环境/工具   │
     └──────┬───────┘              └──────────────┘
            ▼
     ┌──────────────┐
     │   训练引擎    │  Megatron / FSDP / DeepSpeed
     └──────┬───────┘
            │  权重同步（← 这是各家差异最大的地方）
            └──────────────► 回到推理引擎
```

**差异全部集中在**：谁提供推理引擎、谁提供训练并行、怎么编排、怎么同步权重、怎么处理多轮保真、是同步还是异步。下面逐条展开。

---

## 4. 十个关键分歧点

### 4.1 推理引擎：单栈 vs 多栈

| 选择 | 框架 | 含义 |
| --- | --- | --- |
| **只绑 SGLang** | Miles、slime | 深度集成换性能。Miles 的 affinity 路由、R3、TITO 都依赖 SGLang 的特定能力（如 `X-SMG-Routing-Key`、`return_routed_experts`）。**代价是没有退路** |
| **只绑 vLLM** | OpenRLHF | 生态最广、最稳 |
| **多引擎可插拔** | veRL（vLLM / SGLang / TRT-LLM）、NeMo-RL（+Megatron-Inf / Dynamo） | 灵活，但要维护多套适配层 |

**为什么这很重要**：推理引擎决定了**能做什么保真机制**。例如

- `return_routed_experts`（R3 需要）是 SGLang 的能力，vLLM 侧对应能力不同 → 这直接解释了为什么 OpenRLHF **没有 R3**。
- Miles 的 slime 血统决定了它从一开始就是 SGLang-only。

### 4.2 训练并行：三条路线

| 路线 | 框架 | 特点 |
| --- | --- | --- |
| **Megatron 系** | slime、NeMo-RL、veRL、Miles | 并行维度最全（TP/PP/CP/EP/ETP），适合大 MoE。代价是需要 checkpoint 转换、模型定义要写 spec |
| **DeepSpeed ZeRO 系** | OpenRLHF（+ AutoTP + RingAttention） | 上手最容易，无需写模型定义。但缺少专家并行等维度，超大 MoE 吃力 |
| **FSDP 系** | TRL（经 accelerate）、veRL、NeMo-RL、**Miles** | 直接读 HF 目录，适合中小模型与"对照 HF 参考实现" |

**Miles 的位置**：它是**同时提供 Megatron 与 FSDP2 两家**的少数派之一（另一家是 veRL、NeMo-RL）。而**它的上游 slime 只有 Megatron**——这是 Miles 相对 slime 的明确增量之一。

### 4.3 同步 vs 异步：成熟度差异极大

先说清楚三种模式（这也是最容易被混为一谈的地方）：

| 模式 | 墙钟时间 | 数据新鲜度 |
| --- | --- | --- |
| 同步 | `rollout + train`，两侧轮流空转 | 最新 |
| 一步预取 | 训练当前批时生成下一批（深度 1） | 略旧 |
| 完全异步 | `max(rollout, train)`，生成永不停止 | 最旧，需要 staleness 控制 |

| 框架 | 异步的实现状态（源码判据） |
| --- | --- |
| **NeMo-RL** | **工业级子系统**。`nemo_rl/algorithms/async_utils/` 下有 `replay_buffer.py`（84.5 KB）、`trajectory_collector.py`（89 KB）、`staleness_sampler.py`（33 KB）；`nemo_rl/experience/` 下 `rollout_manager.py`（107 KB）、`rollouts.py`（127 KB）、`rollout_recovery.py`（50 KB）。配置里有 `async_engine: true` 与大量 `*-async-1off` recipe |
| **Miles** | **一等公民且显式**。`--fully-async` 是专门 flag，配套 8 条启动期约束；核心实现只有 `fully_async_rollout.py` + `fully_async_data_buffer.py` + `submission_scheduler.py` ≈ **19 KB** |
| **slime** | **有实现无 flag**。`slime/rollout/fully_async_rollout.py` 存在，但通过 `--rollout-function-path` 手动指定；⚠️ **无 staleness 控制**（全仓库 `staleness` 零命中） |
| **veRL** | ⚠️ **experimental**。`verl/verl/experimental/fully_async_policy/`（8 个文件），另有 `one_step_off_policy/`、`separation/` |
| **OpenRLHF** | 有第二实现 `PPOTrainerAsync` + `VLLM_Lock` 机制，但**无显式 staleness 控制** |
| **TRL** | ⚠️ `trl/experimental/async_grpo/`，**experimental** |
| **DeepSpeed** | 不适用 |

**结论**：Miles 的异步**不是最庞大的，但是最"产品化"的**——它有专门 flag、有启动期约束校验、有明确的 staleness 语义（`--max-weight-staleness`）与 7 个专用指标。NeMo-RL 的异步**规模大一个数量级**，覆盖了失败恢复、轨迹重组、staleness 采样抽样等 Miles 完全没有的工业场景。

> ⚠️ 一个反直觉细节：Miles 的 `--max-weight-staleness` **默认是 `None`，即默认不做 staleness 过滤**——只统计不丢弃。这一点极易被误读成"默认有保护"。

### 4.4 权重同步：这个赛道的核心工程难题

**为什么难**：推理引擎与训练引擎对权重的**内存布局要求完全不同**（分片方式、dtype、命名、量化格式），而且 1T 模型的完整同步能到分钟级。

| 框架 | 传输方式 | 关键特征 |
| --- | --- | --- |
| **Miles** | `broadcast`（NCCL）/ `p2p`（**RDMA，Mooncake TransferEngine**）/ `disk-delta`（共享文件系统）/ `cuda_ipc`（colocate） | 四条路径统一在一个 `WeightTransferProtocol` 抽象下，`--update-weight-transfer-mode` 一个枚举切换 |
| **slime** | `nccl` / `disk` × `full` / `delta`（**两个正交 flag**） | Miles 的四条路径里有三条在此；**无 RDMA P2P**（`mooncake` 零命中） |
| **veRL** | **6 个 CheckpointEngine**：`verl/checkpoint_engine/` 下的 `{naive→cuda ipc, nccl, hccl, nixl, mooncake, kimi, delta}_checkpoint_engine.py`；另有 bucketed 与 delta 两种传输模式 | 数量仅次于 NeMo-RL；`delta_weight_transfer.py` 的注释说明 payload 走 "existing ZMQ/CUDA-IPC channel"，最终调 vLLM 的 checkpoint patch loader |
| **OpenRLHF** | colocate → **CUDA IPC**；否则 `stateless_init_process_group` 的 **NCCL 广播** | ZeRO-3 下逐参数 `GatheredParameters` 流式 gather |
| **NeMo-RL** | **7 个同步器**（数量最多）：`nemo_rl/weight_sync/` 下的 `{ipc, collective, nccl_reshard, sglang, megatron, vllm_remote_sparse, checkpoint_engine}_weight_synchronizer.py`（另有 `xferdtensor.py` 作为 nccl_reshard 的传输辅助） | 载体多样性最高；⚠️ **但无 P2P/RDMA**。其中 `vllm_remote_sparse` 自述为 "Shared **S3/ZeroMQ** sparse synchronizer for remote non-colocated vLLM refit"——稀疏是**逐元素**的（与 MoE 专家稀疏无关） |
| **TRL** | ⚠️ **默认根本不做传输**。`vllm_mode` 默认已是 `"colocate"`，此时 `sync_weights()` 退化为**进程内直接传参数引用**（下述） | 与其余六家不是同一根轴上的答案 |
| **DeepSpeed** | 库内有 `runtime/rollout/` 子系统，但 `RolloutEngine.sync_weights` 是**有文档的 no-op**（"co-located … shares weights directly"） | 与 TRL 默认模式同一世界观：**不传输，直接共享** |

**⚠️ TRL 的答案是一个范式差异，必须单独讲。** 在默认的 `colocate` 模式下，`trl/generation/vllm_generation.py::sync_weights` 的实际行为是：

```python
elif self.mode == "colocate":
    for name, param in self._iter_named_params():
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights([(name, param)])
```

即**把 trainer 自己的参数对象直接交给同进程内的 vLLM engine**。没有序列化、没有分桶、没有格式转换、没有 NCCL/RDMA/磁盘。

所以这个轴上的答案不是"哪种传输更快"，而是**两种世界观**：

| | 做法 | 代表 | 代价 |
| --- | --- | --- | --- |
| **传输派** | **把权重运过去**：分桶、转换 HF 命名、NCCL/RDMA/共享存储传输 | Miles / slime / veRL / OpenRLHF / NeMo-RL | 需要维护传输协议栈与拓扑适配 |
| **共享派** | **不运，让两个引擎共享同一份物理权重** | **TRL（默认）**、DeepSpeed `HybridEngineRollout` | 见下 |

TRL 这个选择换来了一个**结构性保证**：既然训练和生成用的是同一份内存里的同一份权重，**on-policy 就不需要靠机制去保证**。但代价是明确的，而且源码把它写了出来——`trl/trainer/grpo_config.py::GRPOConfig.__post_init__` 直接拒绝序列维并行：

> "GRPOTrainer does not support sequence-dim parallelism (`parallelism_config.cp_size > 1` or `parallelism_config.sp_size > 1`) yet. **GRPO builds model inputs after generation inside the trainer**, so Transformers' context-parallel / Ulysses sequence-parallel input sharding cannot be applied to the raw generation batch."

**这是一条清晰的因果链**：共进程 → 输入在 trainer 内部构造 → CP/SP 的输入分片无法作用于原始生成批次 → 长序列场景受限。相比之下 Miles 走的是另一条路：把两者拆成独立进程/独立 GPU 池，因此可以用 CP（GLM-5.2 参考运行用了 **CP 4**），代价是必须维护一整套跨进程权重同步与 offload 机制。

**这张表有一个需要**当场撤回**的初版结论。** 初稿写过「**RDMA P2P 只有 Miles 一家有**」——**这句也是错的**。veRL 的 `verl/checkpoint_engine/mooncake_checkpoint_engine.py` 是：

```python
from mooncake.engine import TransferEngine
...
@CheckpointEngineRegistry.register("mooncake")
class MooncakeCheckpointEngine(CheckpointEngine):
    """Mooncake checkpoint engine with p2p communication using TransferEngine
```

而 Miles 的 `weight_update/protocols/p2p_transfer_utils.py` 里是同一行 `from mooncake.engine import TransferEngine`。**两者用的是同一个组件做 RDMA P2P 权重传输。**

准确的说法是：**Miles 的 `p2p` 是一个一等公民的传输模式（`--update-weight-transfer-mode p2p`），veRL 的等价能力藏在 CheckpointEngine 注册表里（`mooncake` / `nixl` 两个后端）**。功能存在，只是暴露方式与命名完全不同——而**只按 Miles 的词汇表去 grep 别人，必然得出"只有它有"的错误结论**。

不过要注意**这根本不是一根可以数数的轴**：Miles 的四种是按**拓扑**切分（本地 / NCCL / RDMA / 共享存储），而 veRL 的六种、NeMo-RL 的七种里有多个是按**引擎与权重格式**切分（SGLang 双路、Megatron、vLLM sparse、mooncake、nixl）。**"谁的方法多"在这个问题上是伪命题**——它们切的是不同的维度。

**关键差异：disk-delta 的载体不同。** slime / Miles 的 disk-delta 走**共享文件系统**（不需要 NCCL/RDMA fabric，这是它在跨网络部署下可用的原因）；veRL 的 delta 走 **ZMQ/IPC 通道**（仍需连通性）。同名的"delta"解决的是同一个问题（只传变化字节），但适用拓扑不同。

### 4.5 多轮 / TITO：四种不同的答案（最有区分度的一节）

**问题**：多轮 agent 的数据流是 `tokens → 文本 → 工具执行 → 重新渲染 → tokens`。每一步都可能改变 token 序列（重新序列化 tool call、省略 reasoning、模板差异），于是 **trainer 看到的 token 序列 ≠ 引擎实际采样的序列**。而 RL 的 loss 依赖 logprob，错位会让梯度建立在"从未发生过的轨迹"上。

**七家的回答：**

| 框架 | 策略 | 证据 |
| --- | --- | --- |
| **Miles** | **服务端拥有 tokenization**。session server 保存每轮原始 token ID，后续轮只对**追加后缀**做增量 tokenize，并校验旧前缀 | `--use-session-server` / `--tito-model`；`miles/rollout/session/`；`assert_pretokenized_prefix` |
| **slime** | **接受漂移存在，靠数据结构容忍**。`slime/agent/trajectory.py::TrajectoryManager` 把多轮数据建成 per-session 消息树，docstring 明说"**tolerating TITO re-tokenization drift via fork/replace**" | ⚠️ **无 session server**（`session_server` grep 零命中） |
| **OpenRLHF** | **执行器层保证**。README 主张 "unifies generation and training through token-in-token-out agent execution"；实现是 `AgentExecutorBase.execute()`——动作 token 位级保真、反馈增量 tokenize | ⚠️ **无独立 session/tokenizer server** |
| **veRL** | ⚠️ **客户端拥有 tokenization——与 Miles 恰成镜像**。**最硬的证据是服务端连 tokenizer 都不加载**：`verl/trainer/config/rollout/rollout.yaml` 里 `skip_tokenizer_init: True`，注释写着 "the rollout assume token in token out for generation"；`LLMServerClient.generate` 的入参是 `prompt_ids: list[int]`。多轮 token 由客户端 `verl/utils/tokenizer/continuous_token.py::ContinuousTokenBuilder` 维护（assistant token 纯拼接、tool/user 上下文走前缀 diff、不一致直接抛错），并有 `chat_template.py::initialize_turn_separator` 显式修补 turn 边界丢失的分隔符 | 代码注释直言「Continuous Token is the only rollout tokenization path for agent loops」；`rl_dataset.py::RLHFDataset.__getitem__` 的 docstring 也写明 "apply_chat_template has been moved to AgentLoop"——tokenization 所有权彻底移出 dataset 与服务端 |
| **NeMo-RL** | ⚠️ **有完整的四道修正栈（Gym 路径），但 native 多轮路径不做**。① **服务端 tokenization**：自身 vLLM 层暴露 `/tokenize` 端点（`generation_router.py`），设计文档明说 "Token IDs are extracted at the NeMo RL vLLM layer via the `/tokenize` endpoint … **No re-tokenization drift between generation and training**"；② `replace_prefix_tokens` 前缀拼接；③ token 回传 + **后缀增量**（只取 `prompt_token_ids[len(seen_token_ids):]`）；④ `TokenCaptureConfig` ledger——**默认休眠**（`enabled: bool = False`），且 token 数据只能走 TQ、禁止过 actor RPC（`_FORBIDDEN_RPC_KEYS`） | 与 Miles 一样**在服务端持有 tokenization**，但形态是"端点 + 四道事后修正"而非会话服务端；native 多轮的环境观测仍裸调 `tokenizer()`（源码留 TODO） |
| **TRL** | ⚠️ **主线 GRPO 有客户端增量 tokenize——与 Miles 用同一个技巧**。`trl/trainer/grpo_trainer.py::_get_tool_suffix_ids` 用 dummy 会话对齐 `prefix_ids` / `full_ids` 后取差集，并做同款前缀断言：`if full_ids[:len(prefix_ids)] != prefix_ids: raise ValueError(...)`。**但 async 模式下 prompt 以文本过 HTTP 往返，漂移会真的发生**，于是靠 `experimental/async_grpo/async_rollout_worker.py::DriftKind`（CLEAN / REALIGN / FORK）+ `fork_threshold_tokens`（默认 1024）兜底——承认不保真并 fork 新训练行 | ⚠️ **初版把这一格写成"不做"，是错的**（第 6 次同类错误，见下） |
| **DeepSpeed** | **不做** | — |

**这是一张很有信息量的表**：同一个问题，七家给出「服务端拥有 / 客户端拥有 / 数据结构容忍 / 执行器保证 / 默认休眠的 ledger / 不做」六种态度。

> 🔑 **最值得记住的对照是 Miles 与 veRL——它们把责任放在了相反的两端。**
>
> | | 谁拥有 tokenization | agent 发什么 | 漂移怎么处理 |
> | --- | --- | --- | --- |
> | **Miles** | **服务端**（session server） | 发**消息**，服务端决定如何变成 token | **预防**：复用已存 token 前缀，只对追加后缀增量 tokenize |
> | **veRL** | **客户端**（`ContinuousTokenBuilder`） | 发**token IDs**（`prompt_ids: list[int]`），服务端无状态 | **检测**：前缀 diff，不一致直接抛错 + 显式修补 turn 分隔符 |
> | **NeMo-RL** | **服务端**（vLLM 层的 `/tokenize` 端点）**+ 客户端**（Gym 侧） | Gym 走 HTTP 文本协议 | **事后修正**：端点保证 tokenizer 一致 → 前缀拼接 → 后缀增量 → ledger |
>
> ⚠️ **本文必须撤回一个初版结论，并借此说明一条贯穿全文的教训。** 初稿写过「**没有任何一家**复制了 Miles 那种把 tokenization 收归服务端的形态」——**这句是错的**。NeMo-RL 的 vLLM 层同样在服务端持有 tokenization（`/tokenize` 端点），设计文档 `docs/design-docs/nemo-gym-integration.md` 甚至用了几乎相同的措辞来声明它消除了漂移。>
> 准确的说法是：**Miles 与 NeMo-RL 都把 tokenization 放在服务端，但形态与适用范围不同**——Miles 是一个一等公民的会话服务端（`--use-session-server`，作用于主线多轮路径，靠**前缀复用预防**漂移）；NeMo-RL 是一个 `/tokenize` 端点加三道下游修正（作用于 **Gym 集成路径**，靠**端点一致性 + 事后校验**消除漂移，且 native 多轮路径不覆盖）。
>
> **这只是本文被推翻的"唯一性/否定性"断言之一。** 完整清单（**6 条**）：
>
> | # | 初版断言 | 实际情况 | 漏检原因 |
> | --- | --- | --- | --- |
> | 1 | 没有任何一家在服务端持有 tokenization | NeMo-RL 的 `/tokenize` 端点 | 只搜 TITO / session_server |
> | 2 | RDMA P2P 只有 Miles 有 | veRL 用**同一个** `mooncake.engine.TransferEngine` | 按 `p2p` 搜，别人叫 `MooncakeCheckpointEngine` |
> | 3 | 三个框架里只有 Miles 没有 CISPO | veRL / TRL 也有，共四家有 | 只查了 slime 与 NeMo-RL |
> | 4 | OpenRLHF 是唯一以 DeepSpeed 为后端的 RL 系统 | TRL 多个训练器也引用 DeepSpeed | 未检查 TRL |
> | 5 | （§3）"DeepSpeed 不做 RL" | 库内有 `runtime/rollout/` 子系统 | 未查该目录 |
> | 6 | **TRL 的多轮 token 保真"不做"** | 主线 GRPO 有客户端增量 tokenize，**且与 Miles 用同一个 dummy 前缀技巧** | 只搜 `tito` / `session_server` |
>
> **共同点：全部是"用 A 的词汇表去 grep B，然后宣布只有 A 有 / B 没有"。** 同一个能力在别人那里可能叫 `TokenCaptureConfig`、`use_cispo`、`MooncakeCheckpointEngine`、`RolloutEngine`、`_get_tool_suffix_ids`——**按关键字检索证明不了唯一性，也证明不了不存在**。这条教训已写入 §7.3 与方法附录。

> 🔬 **第 6 条还带来一个正面发现：TRL 与 Miles 在"多轮增量 tokenize"上独立收敛到了同一个实现。**
>
> | | TRL `grpo_trainer.py::_get_tool_suffix_ids` | Miles `tito_tokenizer.py` |
> | --- | --- | --- |
> | 技巧 | 造 `dummy_messages`，比较 `prefix_ids` 与 `full_ids` 取差集 | "renders the complete appended suffix … under a synthetic `[dummy_system, dummy_assistant]` prefix" |
> | 校验 | `if full_ids[:len(prefix_ids)] != prefix_ids: raise ValueError(...)` | `assert_pretokenized_prefix` |
> | 兜底 | EOS 位置裁剪（`eos_positions`） | 模型族特定的边界 quirk 覆盖 |
>
> 两个独立项目、同一手法、同一条断言。**这说明"利用 chat template 的前缀单调性做增量差分"已经是这个问题的标准解法**——而 Miles 相对它多出来的那一层，是**把已算好的前缀存下来跨轮复用（服务端会话）**，而不是每轮重算差分。
>
> **而且这个收敛不止两家**：veRL 的 `continuous_token.py` 里有
>
> ```python
> if prefix and prefix[-1] == self._im_end_id:
>     prefix.append(self._newline_id)
>     inserted_token_ids.append(self._newline_id)
> ```
>
> 与 Miles `Qwen3TITOTokenizer.merge_tokens` 修的**是同一个 bug**——Qwen 模板渲染 `<|im_end|>\n`，而生成停在 `<|im_end|>`，少一个换行 token。**三家独立发现并修了同一个边界 quirk**，这本身就说明：多轮 token 保真的难点不在主流程，而在这些模型族的边界特例上。

> ⚠️ **这一栏最容易被误判**，包括我在初稿里也写错过。教训是：**「某能力不存在」的结论必须用多种检索方式交叉验证**——NeMo-RL 的机制叫 `TokenCaptureConfig` 而不是 `tito`/`session_server`，只搜关键词必然漏掉。本文对每家都补做了"目录结构 + 配置类名 + docstring"三重检查。

### 4.6 R3（MoE 路由回放）

MoE 模型里 rollout 与 training 的 top-k 专家选择可能因数值差异翻转，导致"更新作用在从未参与计算的专家上"。

| 有 R3 | 无 R3 |
| --- | --- |
| **Miles**（`--use-rollout-routing-replay`）、**slime**（`slime/utils/routing_replay.py`）、**veRL**、**NeMo-RL** | **OpenRLHF**、**TRL**、**DeepSpeed** |

**注意**：R3 在 slime 里就已存在，**不是 Miles 的发明**。Miles 的增量在于把它与 session server 结合，使回放覆盖**整个多轮 episode**。

### 4.7 编排与并行模型

| 框架 | 编排 | 含义 |
| --- | --- | --- |
| **TRL** | 无（`accelerate`） | 单进程/单脚本。上手最快，扩展性最受限 |
| **OpenRLHF** | Ray 单控制器（12 个文件引用 Ray） | 轻量，好读好改 |
| **slime / Miles** | Ray（22 / 同级） | 中等抽象，**Miles 把 ray/ 拆成 `train/` `rollout/` `specs/` 子包** |
| **veRL** | Ray single-controller（95） | 最重的抽象层之一，配 Hydra 配置系统 |
| **NeMo-RL** | Ray（**149**，最高） | 最重的编排层，配合 `single_controller`、`data_plane`、`experience` 等自创抽象 |

**一个跨框架的收敛：colocate 都靠"分数 GPU + bundle 打包"，但实现不同。**

| | 机制 | 代码 |
| --- | --- | --- |
| **Miles** | 声明**分数 num_gpus** 抢 bundle，再用 `num_gpu_slots_per_worker` 精确占地 | `specs/train.py::_NUM_GPUS_PER_TRAINER_WORKER = 0.4`；engine 侧 `num_gpus_per_worker=0.2` |
| **veRL** | 用 `max_colocate_count` 控制**一个 bundle 里塞几个 worker group** | `single_controller/ray/base.py` 的 `bundle = {"CPU": self.max_colocate_count}`；注释：FSDP 用 `max_colocate_count=1` 合并 WorkerGroup，Megatron 用 `>1` 放不同模型 |

两者都在解决同一个问题——**让多个角色（trainer / rollout engine / critic / reward）共享同一批物理 GPU 并时间复用**——且都通过 Ray 的 bundle 打包实现，只是一个从"GPU 份额"切入、一个从"worker 打包数"切入。

> ⚠️ 一个术语陷阱值得单独提醒：veRL 的 `DisaggregationConfig` 指的是 **rollout 内部 Prefill-Decode 拆分**，**不是** train/rollout 拆分。看到 `disaggregation` 不要条件反射地理解成"训练与推理分离"。

### 4.8 配置范式：三种哲学

| 范式 | 框架 | 适用场景 |
| --- | --- | --- |
| **扁平 argparse flag** | slime（340）、Miles（**358**） | 命令行长、可 grep、可 git diff；缺点是 flag 爆炸 |
| **点号命名空间 argparse** | OpenRLHF（180） | `--actor.num_nodes`、`--ref.num_gpus_per_node`——用点号模拟层级 |
| **YAML + dataclass** | veRL（Hydra）、NeMo-RL（77 个 yaml） | 配置可版本化、可复现；缺点是要读 schema 才知道有什么 |
| **TrainerConfig dataclass** | TRL | 与 HF 生态一致 |
| **JSON 配置文件** | DeepSpeed | `deepspeed/runtime/config.py` 读 JSON/hjson |

> Miles 的 358 个 flag 是这张表里最多的。这既是它"什么都能调"的体现，也是它"参数面复杂"的来源——文档里那篇 §3.6 的架构前提（Miles 自己不定义大部分训练超参，那些来自 Megatron 的 parser）正是理解这 358 个 flag 的钥匙。

### 4.9 算法覆盖

| 框架 | advantage estimator / 算法 |
| --- | --- |
| **veRL** | 最丰富：`gae` / `grpo` / `gspo` / `rloo` / `remax` / `opo` / `reinforce_plus_plus` |
| **slime** | `grpo` / `gspo` / **`cispo`** / `reinforce_plus_plus` / `reinforce_plus_plus_baseline` / `ppo`（**6 个**） |
| **Miles** | `grpo` / `gspo` / `reinforce_plus_plus` / `reinforce_plus_plus_baseline` / `ppo`（**5 个**） |
| **NeMo-RL** | GRPO（sync + async）/ PPO / DPO / SFT / RM / OPD / MPO / distillation |
| **TRL** | SFT / DPO / GRPO / RLOO / KTO / Reward / Distillation——**PPO 已移除** |
| **OpenRLHF** | PPO / GRPO / GSPO / RLOO（`--advantage-estimator` 支持 `gae`/`gspo`/`rloo`） |
| **DeepSpeed** | 无 |

**两个值得注意的收敛与分化：**

- **收敛：train-rollout 不一致修正已成为共识工具。** 三家**独立地**各自实现了同类机制，但成熟度与命名不同：

  | 框架 | 实现 | 覆盖 |
  | --- | --- | --- |
  | **Miles** | `loss_hub/corrections.py::{vanilla_tis_function, icepop_function}` | 2 种（TIS + clip-or-pop） |
  | **slime** | 同 Miles（逐字同源） | 2 种 |
  | **OpenRLHF** | `loss.py::PolicyLoss` 的 `--vllm-is-correction-type` | **3 种**（`tis` / `icepop` / `seq-mask-tis`，后者是序列级几何均值过滤） |
  | **veRL** | **独立子系统** `trainer/ppo/rollout_corr_helper.py` + `algorithm.rollout_correction` | IS 权重 + **多种拒绝采样准则**（`token_k1`、`seq_sum_k3` …），并输出 ESS、rejection rate 等统计 |
  | **NeMo-RL** | loss 配置里的开关 | TIS / icepop / seq-mask-tis / **CISPO** |

  **代码级证据（Miles 与 OpenRLHF 两边语义完全一致）**——Miles `loss_hub/corrections.py::icepop_function`：

  ```python
  ice_weight = torch.where(
      (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
  )
  pg_loss = pg_loss * ice_weight
  ```

  OpenRLHF `openrlhf/models/loss.py::PolicyLoss`：

  ```python
  if self.vllm_is_correction_type == "icepop":
      # ICEPOP: token-level filtering (set coefficients outside the interval to 0)
      vllm_is = torch.exp(rollout_log_ratio).detach()
      mask = (vllm_is >= low_threshold) & (vllm_is <= high_threshold)
      vllm_is = torch.where(mask, vllm_is, 0.0)
      loss = vllm_is * loss
  ```

  ⚠️ **这一条本文做了专门核实**，因为单框架报告里出现了自相矛盾的表述：一处说"本快照中不存在 `clip-or-pop`"（该句在**策略损失类型**的语境下是对的——`--actor.policy_loss_type` 只有 `{ppo, gspo}`），另一处却说"只有 TIS 类，没有 clip-or-pop"（**这句错了**）。读源码可定论：OpenRLHF 的 `icepop` 分支与 Miles 的 `icepop_function` **是同一个算法**（区间内透传比值、区间外置零），且**同名**。教训是：**同一份报告内部可能自相矛盾，结论级句子必须回到代码复核**。

- **分化**：**Miles 丢掉了 slime 的 `cispo`**（`slime/utils/ppo_utils.py::compute_cispo_loss`）。所以 **Miles 不是 slime 的严格超集**——它是一个有取舍的分支。

  ⚠️ 而 `cispo` 的分布**推翻了我初版的一个判断**。我最初写"三个框架里只有 Miles 没有 CISPO"，按 Miles 的词汇表只查了 slime 与 NeMo-RL。补查全部七家后的真实分布是：

  | 框架 | CISPO 存在形式 |
  | --- | --- |
  | **slime** | `--advantage-estimator cispo` 枚举值 |
  | **NeMo-RL** | loss 配置里的 `use_cispo` 布尔（注释写明 "from MiniMax-M1"） |
  | **veRL** | `@register_policy_loss("cispo")` —— `trainer/ppo/core_algos.py::compute_policy_loss_cispo` |
  | **TRL** | `grpo_loss.py` 的 `loss_type == "cispo"`（与 `sapo` / `luspo` / `vespo` 并列） |
  | **OpenRLHF** | ✗ |
  | **Miles** | ✗ |
  | **DeepSpeed** | 不适用 |

  所以准确结论是：**CISPO 在七家里的四家都有，Miles 只是"没有它的两家之一"**，并非孤例。这与它移除 `--custom-advantage-function-path` 仍指向同一取向（Miles 在 advantage/loss 层选择了更小更固定的算法面），但**"只有它没有"这个强度是我不该下的**。

### 4.10 扩展哲学：两个方向的极端

这一节是本次调查中最出乎意料的对比——两家**都把"可扩展性"当成核心议题，却给出了相反的答案**。

| | **Miles / slime** | **TRL** |
| --- | --- | --- |
| 机制 | `--custom-*-path` 一族（Miles 17 个 `--custom-*` + 16 个其它 `*-path`） | `AGENTS.md` **明令禁止** |
| 官方表述 | `docs/developer/architecture.md`：「**If you find yourself patching the trainer to make something work, that's a sign we're missing a hook. Open an issue.**」 | `trl/AGENTS.md`：「**Do not add layers of indirection (registries, factory patterns, plugin systems). A contributor should be able to read a trainer top to bottom and understand the full flow.**」 |
| 代码组织 | 共享 `training_utils/`、`loss_hub/`、可替换组件 | 训练器之间**逐字复制**代码，且规约明确"**Consistency over correctness**" |
| 唯一"注册表" | `function_registry.load_function` + 各种 `--*-path` | CLI 的 `get_commands` |

**两边都不是随手做的选择**：Miles 面向的是"不同模型/硬件/环境的配方千差万别，核心不该被 fork"；TRL 面向的是"`Trainer` 子类应该能被一个人从上到下读懂，抽象层本身就是成本"。

**这把"可扩展性"这个词的相对性暴露得很清楚**：同一个词，一家理解为**提供钩子**，另一家理解为**不要抽象**。所以本文在总表里没有给任何框架打"扩展性"的分——**这个维度上没有共同标尺**。

---

## 5. Miles 的谱系：与 slime 的逐条对照

这一节是本文最有实证价值的部分，因为 **slime 的 README 自己就写明了关系**：

> [**Miles**](https://github.com/radixark/miles) is an RL post-training framework for large-scale models, **built on slime** by RadixArk. It stays closely aligned with slime's upstream development while extending it with enterprise-oriented features: deeper SGLang integration, operational tooling, deployment support, and optimizations for new models and hardware. **Miles also adds a growing set of production features, including LoRA, TITO, and low-precision training.**

这段一手陈述（来自 slime 侧，非 Miles 自述）与我们的源码比对**完全吻合**。

### 5.1 结构性证据

| 证据 | slime | Miles |
| --- | --- | --- |
| 入口脚本名 | `train.py` / `train_async.py` | `train.py` / `train_async.py` / +`train_multi_lora_async.py` |
| 包名 | `slime` / `slime_plugins` | `miles` / `miles_plugins` |
| 同名文件（逐字） | `utils/{types,arguments,disk_delta,routing_replay,tensor_backper,mask_utils,seqlen_balancing,dp_schedule,memory_utils,reloadable_process_group}.py` | 同名存在 |
| `--train-backend` | 存在，但 `choices=["megatron"]`（**只有一个取值**） | `choices=["megatron","fsdp"]` |
| 四旋钮恒等式 | 存在，**断言逐字同源** | 存在（含同一个校验缺口） |
| `--custom-*` 家族 | 16 个 | 17 个（**增 2 删 1**，见 §5.2） |
| 唯一 CLI flag | 340 | 358（**200 个共有**） |

> 「`--train-backend` 在 slime 里只有一个取值」是一个特别直观的继承证据：Miles 接过这个 flag，给它加了第二个后端。

**「逐字同源」不是修辞，是可以贴出来的。** slime 的 `slime/utils/arguments.py`：

```python
if args.num_steps_per_rollout is not None:
    global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
    if args.global_batch_size is not None:
        assert args.global_batch_size == global_batch_size, (
            f"global_batch_size {args.global_batch_size} is not equal to "
            f"rollout_batch_size {args.rollout_batch_size} * n_samples_per_prompt {args.n_samples_per_prompt} "
            f"// num_steps_per_rollout {args.num_steps_per_rollout}"
        )
    args.global_batch_size = global_batch_size
```

Miles 的同名校验与它**逐 token 相同，连错误消息字符串都一样**。

> ⚠️ **一个连带的重要推论**：我在 Miles 文档 §3.3 里指出的那个「**四旋钮恒等式存在校验缺口**」（不设 `--num-steps-per-rollout` 时不做一致性检查）——**这个缺口同样来自 slime，不是 Miles 引入的**。两边都是同一个 `if args.num_steps_per_rollout is not None:` 守卫。这提醒我们：**在演化关系里，"发现一个缺陷"不等于"这个缺陷是后来引入的"**。

### 5.2 已在 slime 中存在的 Miles 特征（**不是 Miles 的发明**）

三阶段循环 · `TrainRayActor` / `RayTrainGroup` / `placement_group` · `--custom-*-path` 家族 · 四旋钮恒等式 · **R3** · **TIS / ice-pop** · **disk-delta** · **colocate + CUDA IPC 权重同步** · advantage 家族（slime 甚至多一个 `cispo`）· `tensor_backper.py` 等工具文件

**两处需要精确到 flag 级的细分**（这两条很容易被"都有/都没有"的粗粒度结论盖掉）：

| 机制 | slime | Miles | 说明 |
| --- | --- | --- | --- |
| **R3（MoE 路由回放）** | ✅ `--use-rollout-routing-replay` | ✅ 同名 | **继承而来** |
| **Indexer replay（DSA indexer top-k）** | ✗ **零命中** | ✅ `--use-rollout-indexer-replay` / `--use-indexer-replay` | **Miles 新增**。所以 R3 是继承的，而它的 indexer 变体是 Miles 扩的 |
| `--custom-*` 家族 | 16 个 | 17 个 | **增 2 删 1**：新增 `--custom-async-data-buffer-path`、`--custom-megatron-post-save-hook-path`；**删除 `--custom-advantage-function-path`** |

> ⚠️ 最后那条「删 1」值得单独注意：Miles **移除了 slime 的一个扩展点**。这与它同时丢掉 `cispo` 是同一件事的两面——slime 允许用户自带 advantage 函数，而 Miles 把它收敛成固定的 5 个 estimator 枚举。**一个以"可扩展"为卖点的框架，在这个位置选择了收窄**，这是分析 Miles 设计取向时不该漏掉的一笔。

### 5.3 slime 中**不存在**的 Miles 特征（Miles 的真实增量）

用 flag 差集（158 个 Miles 独有 flag）+ 目录差集交叉验证，收敛为**四束**：

| # | 增量 | 证据 |
| --- | --- | --- |
| **①** | **TITO session server** | `--use-session-server` / `--tito-model` / `--session-message-matcher` / `--session-*-path`；`miles/rollout/session/`（含 v2 树形服务）；slime 侧 `session_server` grep 零命中 |
| **②** | **FSDP2 后端 + 独立 `training_utils/` + true-on-policy** | `miles/backends/fsdp_utils/`、`miles/backends/training_utils/`、`miles/true_on_policy/`；slime 侧 `fsdp` 与 `true_on_policy` 均零命中。相关 flag：`--true-on-policy-mode`、`--recompute-logprobs-via-prefill` |
| **③** | **DataBuffer 契约 + staleness 控制 + RDMA P2P + 自研 Router** | `--fully-async`（专门 flag + 8 条约束）、`--max-weight-staleness`、`--async-data-buffer-capacity-factor`、`--custom-async-data-buffer-path`、`--update-weight-transfer-mode`（含 `p2p`）、`miles/router/`；slime 侧 `mooncake` 零命中 |
| **④** | **角色再分层、LoRA/multi-LoRA、工程化设施** | `miles/ray/{train,rollout,specs,multi_lora}/` 子包拆分；`--lora-*` / `--multi-lora-*`（slime `lora` 仅 2 个文件）；`miles/dashboard/`、`miles/utils/{ft_utils,audit_utils,chat_template_utils,workers}/`；测试 **107 → 769** |

**同时 Miles 也丢了一些东西**：`cispo` advantage、`--custom-advantage-function-path`、`--megatron-deepgemm-*`、`--rollout-data-transport`、`--use-stateless-adam`、`--force-fp8-ue8m0-scale` 等 23 个 slime 独有 flag。

> 这 23 个 flag 的存在说明一件事：**演化分支会做减法**。"Miles 是 slime 的超集"是一个方便但错误的默认假设——它有明确的取舍，且在**扩展点**上（custom advantage）反而比祖先更保守。

### 5.4 一句话概括谱系

> **slime 提供了骨架与大部分机制；Miles 在其上做了三件事：把不可靠的部分（多轮 token 保真）从"容忍"变成"消灭"、把单后端扩成双后端、把一套研究框架做成可运维的生产系统。** 代价是代码量翻倍（35.9 K → 69.5 K 行）与 flag 面扩张。

---

## 6. 选型建议

| 你的场景 | 建议 | 理由 |
| --- | --- | --- |
| **要跑到前沿规模（数百 B ~ 1T MoE）** | **Miles** 或 **slime** 或 **NeMo-RL** | 三家都有 700B+ 级实证脚本；需要 Megatron 的并行维度 |
| **要 agentic / 长程多轮工具调用** | **Miles**（若要多轮前缀复用与 `--tito-model` 注册表）；**veRL**（若要多引擎与丰富算法）；**NeMo-RL**（若要厂商栈与 Gym 集成） | 三家都实现了 token 级可控，但形态不同：Miles 是服务端会话（前缀复用）、veRL 是客户端 token 构建、NeMo-RL 是 `/tokenize` 端点 + 四道修正（§4.5） |
| **要快速验证算法 / 做研究** | **TRL** 或 **OpenRLHF** | TRL 改一个 TrainerConfig 就能跑；OpenRLHF 只有 1.2 万行，改起来最快 |
| **要单机 / 中小模型 / 最低门槛** | **TRL** | 不引入 Ray、不引入独立引擎，`transformers` 生态直接可用 |
| **要 DeepSpeed 生态（已有 ZeRO 调优经验）** | **OpenRLHF** 或 **TRL**（经 `accelerate` 可接 DeepSpeed） | 实测引用 DeepSpeed 的完整 RL 系统主要是 OpenRLHF；TRL 的多个训练器也引用 DeepSpeed，但它属训练器库而非完整系统 |
| **要 NVIDIA 全家桶（Megatron-Core + TRT-LLM + ModelOpt）** | **NeMo-RL** | 厂商深度绑定，量化与推理侧集成最完整 |
| **只要训练加速，RL 自己写** | **DeepSpeed** | 它的正确对照物是 Megatron/FSDP，不是 RL 框架 |
| **要最丰富的算法集合** | **veRL** | RLOO / ReMax / OPO / GAE 等独有 |

**三条容易踩的坑：**

1. **别把 DeepSpeed 和 RL 框架对比**——它是底座。同理，TRL 与 veRL 也不是同类。
2. **"支持异步"这四个字水分很大**。veRL 与 TRL 的异步在 `experimental/` 下；slime 有实现但无 flag 也无 staleness 控制；Miles 有 flag 但 staleness **默认关闭**（只统计不丢弃）。**就本次调查所及**，异步子系统规模最大、recipe 最完整的是 NeMo-RL（20 个 `*-async-*` recipe + `algorithms/async_utils/`），但这是一条**成熟度比较**而非"别人没有"——按 §4.5 的教训，凡"只有"字样都应视为待验证。
3. **"支持 TITO"有四种以上含义**（§4.5）。如果你需要**严格**的 token 级一致（例如要做 true-on-policy 或精确的 logprob 对齐），要看的不是"有没有 TITO"，而是**它把 tokenization 放在哪一端、靠什么机制保证**——Miles 靠服务端前缀复用**预防**，veRL 靠客户端前缀 diff**检测**，NeMo-RL 靠 `/tokenize` 端点 + 事后校验**修正**。三种都能用，但对失败模式的处理不同。

4. ⚠️ **本文最贵的教训：「唯一性」断言不能用关键字检索来支持。** 本文初版产生过 **5 条**错误的唯一性断言（完整清单见 §4.5），全部在后续核实中被推翻——从"没有任何一家在服务端持有 tokenization"到"RDMA P2P 只有 Miles 有"。同一能力在别人那里可能叫 `TokenCaptureConfig`（不是 TITO）、`use_cispo`（不是 advantage estimator）、`MooncakeCheckpointEngine`（不是 p2p）、`RolloutEngine`（不是"不做 RL"）。
   **可操作的做法**：先确认该能力的**机制语义**（它解决什么问题、在数据流的哪一步），再按语义去找别人家的实现，而不是按名字找。本文对"存在"类结论都给了 flag 名/类名/逐字代码以便复核；对**凡本文未附证据的"独有/唯一"，请一律视为待验证断言**。

5. **"某框架不支持 X"的结论要看它的文档与代码是否一致。** 本次调查在 veRL 与 NeMo-RL 两处都发现过 README/文档声称的能力与实际代码不符（veRL 的 `use_remove_padding`/`use_dynamic_bsz` 类级默认不生效、VeOmni 激活卸载只有 FSDP 读而文档声称会 reject；NeMo-RL 的能力矩阵与大量 `NotImplementedError` 组合约束有差距）。**读文档只能生成假设，不能生成结论。**

---

## 7. 方法附录：本次调查如何做的（可复现）

### 7.1 克隆

```bash
mkdir -p frameworks && cd frameworks
for spec in \
  "verl https://github.com/volcengine/verl.git" \
  "trl https://github.com/huggingface/trl.git" \
  "DeepSpeed https://github.com/deepspeedai/DeepSpeed.git" \
  "OpenRLHF https://github.com/OpenRLHF/OpenRLHF.git" \
  "slime https://github.com/THUDM/slime.git" ; do
  set -- $spec; git clone --depth 1 "$2" "$1"
done
git clone --depth 1 https://github.com/NVIDIA-NeMo/RL.git NeMo-RL
```

> ⚠️ 全部使用 `--depth 1` 浅克隆，因此**无法做 git 历史分析**，本文所有结论都是"当前形态"而非"演化过程"。这一点影响了 §5 的措辞：只断言"当前 slime 中不存在 X"，而不断言"X 是 Miles 首先引入的"。

### 7.2 关键对比命令

```bash
# 代码规模（核心包 .py 行数）
find <pkg> -name '*.py' -exec cat {} + | wc -l

# 唯一 CLI flag 数（允许点号命名空间，以覆盖 OpenRLHF 的 --actor.num_nodes 风格）
grep -rhoE '"--[a-z][a-z0-9._-]*"' <repo> --include=*.py | sort -u | wc -l

# Ray 使用强度
grep -rl "import ray\|from ray" <repo> --include=*.py | wc -l

# flag 差集（本文 §5.3 的核心方法）
python3 -c "
import re
def flags(p):
    s=open(p,encoding='utf-8',errors='replace').read()
    return set(re.findall(r'\"(--[a-z0-9][a-z0-9\-]*)\"', s))
sl=flags('frameworks/slime/slime/utils/arguments.py')
mi=flags('upstream/miles/utils/arguments.py')
print('共有', len(sl&mi), '| 仅 Miles', len(mi-sl), '| 仅 slime', len(sl-mi))
"
```

### 7.3 方法学教训（本次调查中实际踩到的）

1. **`grep tito` 会产生假阳性**——`mul-ti-to-ken`（multi-token prediction，MTP）里正好含 "tito"。必须用 `\btito\b` 并人工核对命中行。本文初版因此把 veRL / NeMo-RL 误判为"有 TITO"。
2. **`fully_async` 这类命名差异会导致漏判**——NeMo-RL 不用 `fully_async` 这个词，但它有规模最大的异步子系统（`algorithms/async_utils/`）。**不能只靠一个关键词判断有无某能力**，要结合目录结构、配置键与 recipe 命名。
3. **`--colocate` 之类的 flag 名不可跨框架套用**——只有 slime/Miles 用这个 flag 名；veRL 与 NeMo-RL 用配置键（`colocated` 出现 374 次），OpenRLHF 用 `--colocate-all`。
4. **README 不可作为能力证据**——本文对每一条能力主张都回到源码；实际发现 README 与代码不符之处（如 TRL 的 PPO 已被移除但文档仍有残留印象）已在各单框架文档中标注。
5. **浅克隆的代价要写进结论**——不能从"当前不存在"推出"历史上从未有"，也不能从"两个仓库都有 X"推出"谁抄谁"。
6. ⚠️ **本次调查最大的教训：「唯一性」断言几乎全错。** 初版产生过 5 条错误的唯一性断言（清单见 §4.5），根因统一：**用 A 框架的词汇表去检索 B 框架**。`grep tito` 会命中 `mul-ti-to-ken`；`grep p2p` 找不到 `MooncakeCheckpointEngine`；`grep session_server` 找不到 `/tokenize`。
   **因此本文的结论按强度分三级**：
   - **强**：附 flag 名 / 类名 / 逐字代码，且已在本工作区的固定 commit 上验证 → 可直接引用。
   - **中**：附文件路径，但依赖子代理报告未逐行复核 → 建议复核后引用。
   - **弱**：任何形式的"只有 X 这样做 / 其他家都没有" → **本文已尽力清理，但请一律视为待验证**。

---

## 附：六个框架各自的「一句话记忆点」

| 框架 | 记住这一句 |
| --- | --- |
| **Miles** | slime 的生产化分支：**在服务端持有 tokenization 并靠前缀复用预防漂移**（TITO session server），双训练后端，四传输权重同步，异步做得最"产品化" |
| **slime** | 极简且诚实：只连 Megatron + SGLang，不做中间抽象，把 rollout 完全开放给用户 |
| **veRL** | 算法最全、引擎最多、配置最工程化，但异步与 agent loop 还在 `experimental/` |
| **OpenRLHF** | 1.2 万行实现完整 RLHF，DeepSpeed + vLLM 路线，最易读易改 |
| **TRL** | 训练器库而非 RL 系统：无 Ray、无独立引擎，门槛最低、上限也最明确 |
| **NeMo-RL** | 厂商重装路线：16 万行、Ray 149 文件、工业级异步子系统，深度绑定 Megatron-Core/TRT-LLM |
| **DeepSpeed** | 不是 RL 框架，是 RL 框架的**底座**；RLHF 部分（DeepSpeed-Chat）在另一个仓库 |

{% endraw %}
