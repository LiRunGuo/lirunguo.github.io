---
title: "作业汇总"
collection: course-notes
chapter: true
permalink: /course-notes/cs336-language-modeling-zh/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cs336-language-modeling-zh/) · [← l17](/course-notes/cs336-language-modeling-zh/l17)

{% raw %}
## 作业汇总

五次作业都在 **PyTorch 中、以极少脚手架**完成：仓库提供单元测试与 adapter 接口用于验证正确性，但所有实现都要你自己写。先在本地 CPU 上调正确性，再上 GPU（选课学生由 Modal 赞助算力）做训练与 benchmark。截止日期见课程日程表。通过 Gradescope 提交；部分作业设有排行榜（在固定预算下最小化困惑度）。**AI 政策**（2025–26）禁止用 AI 实现作业的任何部分——只可用于概念提问与 API 文档查询，并需把仓库的 `AGENTS.md` 粘贴到聊天对话开头。

---

### 作业 1：基础（分词、模型、训练）— `assignment1-basics`

**目标：** 实现训练一个标准 Transformer 语言模型所需的全部组件，并真正训练出一个最小模型（TinyStories，然后 OpenWebText）。

**主要实现任务（摘自 handout）：**
1. **BPE 分词器**（第 2 节）：在语料上学习 merge（词表初始化、用 GPT-2 风格 regex 做预分词、merge 计算、special token 如 `<|endoftext|>`、并行预分词、优化合并步骤）；保证 encode/decode 往返正确。
2. **Transformer 语言模型**（第 3 节）：token embedding、**RMSNorm**、**RoPE**、**因果多头自注意力**（QKV 投影、因果掩码、softmax）、逐位置 **SwiGLU 前馈**、pre-norm 块、LM head → 下一 token 概率；handout 明确了各维度约定（B、S、D 等）。
3. **交叉熵损失 + AdamW 优化器**（第 4 节）：标准 NLL 损失，以及作为 `torch.optim.Optimizer` 子类实现的 AdamW（一阶/二阶矩、权重衰减、学习率调度；handout 中给出了 SGD 的完整示例）。
4. **训练循环**（第 5 节）：checkpoint 保存/加载（序列化模型 + 优化器状态）、训练配置（batch size、LR 等）、解码支持（贪心/采样）、困惑度评估。
5. **资源核算**：给定配置下 Transformer 各组件的 FLOPs 与显存。
6. **运行**：先在 TinyStories 上训练小模型（handout 有样例输出），再换到 OpenWebText。
7. **排行榜**：在 B200 上 45 分钟内最小化 OpenWebText 困惑度（讲义给出了去年的排行榜链接）。

**最关键的讲座内容：** 第 1 讲（分词/BPE——直接蓝图）、第 2 讲（资源核算、Adam 系优化器、训练循环）、第 3 讲（现代架构变体：pre-norm、RMSNorm、RoPE、SwiGLU、无 bias）。

---

### 作业 2：系统（Profiling、Kernel、分布式）— `assignment2-systems`

**目标：** 用高级工具对作业 1 的模型做 profiling 与 benchmark；用自己的 Triton 实现 FlashAttention-2 优化 attention；构建显存高效、可扩展的分布式训练。

**主要实现任务（摘自 handout）：**
1. **Benchmark + profiling 框架**：测量各算子的运行时间与显存；使用 Nsight Compute 与 NVTX range；回答"前向+反向哪个 kernel 最耗时"这类问题；混合精度（autocast）数据类型相关问题。
2. **激活重计算**：为 TransformerBlock 包装重计算；用 `saved_tensors_hooks`（pack/unpack hook）验证显存节省。
3. **FlashAttention-2 Triton kernel**：前向（分块 QKᵀ、在线 softmax、带掩码写回）与反向（重算打分矩阵），用 `torch.autograd.Function` 承载；handout 附有 WeightedSum 的 autograd 教学示例与精度容差指南。
4. **分布式数据并行训练**：梯度 all-reduce（反向 hook、异步通信），与单卡对比 benchmark。
5. **优化器状态分片**：梯度 reduce-scatter、本地 AdamW 更新、参数 all-gather（ZeRO-1）。
6. **全分片数据并行（FSDP）**：连参数也分片，前向/反向按需 all-gather、梯度 reduce-scatter；验证显存扩展；训练一个原本装不下的模型。
7. **排行榜**：提交你的最佳结果。

**最关键的讲座内容：** 第 5 讲（GPU 模型、roofline）、第 6 讲（Triton kernel：分块、在线 softmax、融合——四类 kernel 模式）、第 7 讲（集合通信、DDP、带宽测量）、第 8 讲（ZeRO 1–3/FSDP 的显存与通信核算、重叠）。

---

### 作业 3：扩展（Scaling Laws）— `assignment3-scaling`

**目标：** 拟合扩展律，为一次大规模训练（48 B200 小时）预测算力最优配置，而实验预算只有 12 B200 小时。

**主要实现任务（摘自 handout）：**
1. 调用**训练 API**（HTTP 端点、`X-API-Key` 头）：提交带超参数（层数、embedding 大小、head 数、batch size、学习率、训练 token 数）与最长墙钟时间的实验；轮询状态；获取验证损失。
2. 设计搜索空间策略：IsoFLOPs 剖面（Chinchilla 方法 2）——对每个算力预算，以 D = C/6N 扫模型规模；或联合拟合（方法 3）、Kaplan 风格、muP 相关思路（handout 欢迎）。
3. 拟合幂律 N_opt(C) 与 D_opt(C)；**外推**到 48 B200 小时的预算；提交预测的算力最优超参数与预测最终验证损失。
4. 报告：完整方法论，需详细到可复现；部分成绩取决于你预测模型的**实测**性能。

**最关键的讲座内容：** 第 9 讲（扩展律基础、IsoFLOPs 流程、6ND 法则、幂律拟合、Chinchilla vs Kaplan）、第 11 讲（案例研究：用 WSD 重启让拟合变便宜、DeepSeek/MiniCPM 配方、muP、临界 batch/LR 的扩展）。

---

### 作业 4：数据（数据加工流水线）— `assignment4-data`

**目标：** 把原始 Common Crawl 转成可用的预训练数据，涵盖转换、过滤、去重；并衡量对模型质量的影响。

**主要实现任务（摘自 handout）：**
1. **Common Crawl HTML 转文本**（WARC/WET 文件）：用 trafilatura/resiliparse 解析；对比 WET 与 WARC；用 `concurrent.futures` 并行处理。
2. **过滤**：(a) 基于规则的过滤（Gopher/C4 标准——例如词数太少、字母内容比例过低、boilerplate、非英语的文档要删）；(b) **有害内容分类器**（提供的 NSFW 与仇恨言论分类器）；(c) **PII 移除**（邮箱、IP 地址）；(d) 可选的额外过滤。
3. **去重**：先精确哈希，再用 **MinHash** 做近重复检测（含 Jaccard 相似度与 LSH banding）；必要时用你作业 1 的 BPE 分词器做分词。
4. **训练**：构建 token 化数据集，在"过滤后 vs 未过滤"数据上训练小 LM（handout 提供了分词与用保存好的 token ID 训练的起始代码）。
5. **排行榜**：在给定 token 预算下最小化困惑度。

**最关键的讲座内容：** 第 13 讲（来源：Common Crawl 的 WARC/WET、版权、数据集谱系、质量分类器）、第 14 讲（完整流水线：转换；过滤框架——目标 vs 原始数据、KenLM/fastText 分类器、毒性过滤；去重——MurmurHash、MinHash、LSH 阈值与 (b, r) 数学；混合——epoch 陷阱、UniMax、模拟 epoch）、第 12 讲（以困惑度作为评测指标）。

---

### 作业 5：对齐与推理强化学习（GRPO）— `assignment5-alignment`

**目标：** 通过监督式提示基线 + 强化学习（GRPO）训练语言模型解数学题，并探索策略梯度估计器变体。*可选第二部分：*用 SFT + DPO 做指令跟随与安全对齐。

**主要实现任务（必修，摘自 handout）：**
1. **Zero-shot、few-shot 与思维链（CoT）提示**基线（在数学评测数据上）。
2. **GRPO 实现**：vLLM rollout 服务（`VLLMCompletion`/`VLLMServer`、停止字符串）、带 response mask 的分词、当前/旧/参考策略下的对数概率、组内归一化优势、含 KL 惩罚 + 长度归一化的裁剪比损失、梯度累积、带指标的完整训练循环（奖励、KL、回答长度）。
3. **策略梯度估计器变体**：baseline 选择（组均值 vs 其他）、重要性权重裁剪（不裁剪 vs 裁剪）、长度归一化重加权——分析方差/期望；handout 推导了"未裁剪的 token 级重加权"估计器及其裁剪版本。
4. 在小模型（如 OLMo-2-0425-1B）上运行，每个推理批次做约 32 步离策略训练；测量准确率提升。

**可选第二部分（补充——SFT + DPO）：**
1. 在 MMLU、GSM8K、AlpacaEval、SimpleSafetyTests、Anthropic HH 上做 zero-shot 基线。
2. 在指令-回答数据上对 Llama 3.1 8B 做**监督微调**。
3. 在成对偏好数据（Anthropic HH）上做**直接偏好优化（DPO）**，用 Llama 3.3 70B 作裁判评测 AlpacaEval 与安全性。

**最关键的讲座内容：** 第 15 讲（SFT 数据、RLHF 目标、PPO vs DPO、过度优化/模式崩塌）、第 16 讲（GRPO——你要实现的正是这个算法，及其偏差与变体；R1/Kimi/Qwen3 配方；RL 基础设施）、第 10 讲（推理基础设施——vLLM、KV cache、生成）、第 12 讲（补充部分使用的评测基准）。

---

### 作业与讲座对应表

| 作业 | 主要相关讲座 | 核心技能 |
|---|---|---|
| 1 — 基础 | 1、2、3 | BPE、Transformer 模块、AdamW、训练循环、资源核算 |
| 2 — 系统 | 5、6、7、8 | profiling、Triton kernel、FlashAttention-2、DDP/ZeRO/FSDP |
| 3 — 扩展 | 9、11 | 扩展律拟合、IsoFLOPs、外推与预测 |
| 4 — 数据 | 12、13、14 | HTML→文本、分类器过滤、MinHash/LSH 去重、混合、PPL 评测 |
| 5 — 对齐 | 10、15、16 | prompting、GRPO、策略梯度变体、（可选 DPO/SFT） |

---

## 术语表（Glossary）

- **Activation checkpointing（激活重计算）**——反向时重算前向激活而不保存；用算力换显存（亦称 gradient checkpointing / rematerialization）。
- **Adam / AdamW**——自适应优化器，维护逐参数的一阶/二阶矩（AdamW 加解耦权重衰减）；LLM 的标准优化器（fp32 下 8 字节/参数）。
- **All-reduce / reduce-scatter / all-gather（集合通信）**——跨设备求和并复制；求和并分片；把分片汇聚到所有设备。all-reduce = reduce-scatter + all-gather。
- **Arithmetic intensity（算术强度）**——每搬运一字节所做的 FLOPs；与硬件的加速器强度（峰值 FLOP/s ÷ 带宽）比较，判断是 memory-bound 还是 compute-bound。
- **AWQ（激活感知权重量化）**——训练后量化，依据激活幅度把少数（被大激活通道命中的）权重保留更高精度。
- **Bank conflict（bank 冲突）**——warp 内多线程访问同一 shared memory bank 导致的串行化；用 swizzling/padding 缓解。
- **bf16 / fp16 / fp8 / fp4**——浮点格式：bf16 是 2 字节但动态范围同 fp32；fp16 范围小（有下溢风险）；fp8（E4M3/E5M2）与 NVFP4 用于低精度计算/推理。
- **BPE（字节对编码）**——子词分词器：从字节出发，反复合并最高频相邻对直到达到目标词表大小。
- **Chinchilla 扩展律**——算力最优扩展：N_opt ∝ C^0.5、D_opt ∝ C^0.5，约每参数 20 token（对比 Kaplan 的 0.73/0.27）。
- **Collective operation（集合操作）**——所有设备参与的通信模式（broadcast、scatter、gather、reduce 及其 "all-" 变体）。
- **Compute-bound vs memory-bound**——计算时间还是访存时间占主导（由算术强度与加速器强度之比决定）。
- **Continuous batching（连续批处理）**——迭代级调度，在每个解码步加入/移除请求，而非静态批次。
- **DDP（分布式数据并行）**——复制参数、切分 batch、all-reduce 梯度。
- **DPO（直接偏好优化）**——无需奖励模型的 RLHF：用偏好对分类损失优化隐含奖励 β·log(π/π_ref)。
- **Deduplication（去重）**——去除精确与近重复文档（精确哈希；近重复用 MinHash + LSH）以省算力并避免记忆。
- **Fair use（合理使用）**——版权例外原则（四要素：目的、作品性质、使用量、市场影响）；部分 LLM 训练在具体案件中被判合法。
- **FlashAttention**——融合、分块的 attention，配在线（telescoping）softmax；不物化 S/P 矩阵；block 级显存 O(1)。
- **FLOPs 与 FLOP/s**——浮点运算次数（工作量）与每秒浮点运算次数（速度）；MFU = 实际/峰值。
- **FSDP / ZeRO-3**——分片参数、梯度与优化器状态；按需 all-gather 参数、reduce-scatter 梯度；通信约 3×#params。
- **GQA（分组查询注意力）**——KV 头少于 query 头以缩小 KV cache；MQA 即 1 个 KV 头。
- **GRPO（组相对策略优化）**——去掉价值模型的 PPO：优势 = (奖励 − 组均值)/组标准差（每提示 G 条 rollout）；已知有标准差归一化偏差与长度偏差。
- **IsoFLOPs**——扩展律方法：固定算力 C_i、扫模型规模、取最小损失；拟合 N_opt(C)、D_opt(C) 的幂律。
- **KV cache**——缓存 key/value 向量，使每 token 生成为 O(1) 成本（而非 O(T) 重算）；其大小决定生成阶段的访存受限程度。
- **Linear attention（线性注意力）**——核为恒等的 attention：Q(KᵀV) = (QKᵀ)V，序列长度线性；递推形式 S_t = S_{t−1} + k_t v_tᵀ（对偶性）。
- **LSH（局部敏感哈希）**——b 个 band、每 band r 个 min-hash；碰撞概率 1−(1−s^r)^b 是 S 形曲线，阈值在 (1/b)^(1/r)。
- **Memory coalescing（显存合并访问）**——warp 的访问合并为 128 字节事务；快速 HBM 读取的前提。
- **MFU（模型浮点利用率）**——实际 FLOP/s ÷ 峰值 FLOP/s；≥0.5 已算不错。
- **MinHash**——满足 Pr[h(A)=h(B)] = Jaccard(A,B) 的哈希方案；即集合在随机哈希下的最小哈希值。
- **MLA（多头隐层注意力）**——把 K/V 压成低维隐向量 c（DeepSeek v2），缩小 KV cache；需额外保留非旋转维度以兼容 RoPE。
- **MoE（混合专家）**——多个专家 FFN + 路由器；在每 token FLOPs 不变的前提下增加参数；用 top-k 路由 + 均衡损失训练。
- **muP（最大更新参数化）**——宽度感知的初始化 + LR 缩放，使最优超参数可跨规模迁移；会被 RMSNorm 增益/强 weight decay 破坏。
- **PagedAttention**——给 KV cache 做虚拟内存式分页（vLLM）：不连续 block、前缀共享、写时复制。
- **Perplexity（困惑度）**——exp(平均 NLL)，即 (1/p(D))^(1/|D|)；越低越好；随机猜测时约等于词表大小。
- **Prefill 与 decode**——推理的两个阶段：并行处理 prompt（compute-bound）与一次生成一个 token（memory-bound）。
- **PPO（近端策略优化）**——带裁剪重要性比、价值模型与 KL 控制的 RL 算法；经典 RLHF 优化器。
- **QK-norm / z-loss**——稳定性技巧：在 softmax 前归一化 Q、K；惩罚 log-sum-exp 以防止 logit 漂移。
- **RLHF**——基于人类（或 AI）成对反馈的强化学习，配合 KL 控制来优化偏好。
- **RLVR**——可验证奖励的 RL（精确答案、测试通过）：可扩展，避免奖励模型的过度优化。
- **RMSNorm / LayerNorm**——归一化层：RMSNorm 只按均方根缩放、不减均值也无 bias；LayerNorm 做中心化与缩放。
- **Roofline model**——以算术强度为横轴、可达 FLOP/s 为纵轴的图；拐点即加速器强度。
- **RoPE（旋转位置编码）**——按位置旋转 Q/K 的坐标对，使 attention 分数只依赖相对位置。
- **Scaling law（扩展律）**——损失与数据/模型/算力之间的幂律关系；支持小规模 → 大规模预测。
- **Speculative sampling（投机采样）**——草稿模型提议 token、目标模型并行打分；修正拒绝采样对目标模型是**精确**的。
- **SFT（监督微调）**——在指令-回答示范上训练；最擅长抽取预训练已具备的行为。
- **SwiGLU / GeGLU**——门控 FFN 激活（swish/高斯误差门 ⊙ 线性分支）；2023 年后模型的标准；FFN 维度约 8/3× 模型维度。
- **Tensor parallel（张量并行）**——把权重矩阵切分到多设备，逐层 all-gather 激活；需要 NVLink 级互联。
- **Tiling（分块）**——在共享内存 tile 上计算以减少 HBM 流量（矩阵乘、FlashAttention）；全局读取降低 T 倍。
- **Tokenizer（分词器）**——encode(文本)→token ID 与 decode(ID)→文本；标准做法是 BPE；必须往返正确并处理 UTF-8 与 special token。
- **Upcycling**——用预训练的稠密模型初始化 MoE。
- **WSD 学习率**——warmup–stable–decay 调度；可重启，便于廉价拟合扩展律。
- **ZeRO-1 / ZeRO-2**——仅分片优化器状态 / 再加上梯度；通信量与 DDP 相同而显存更优。

---

*本笔记整理自公开的 CS336 课程网站（Spring 2026 开设；并参考 Spring 2024/2025 存档以核对日程）、可执行讲义文件、讲义幻灯片 PDF 与作业 handout。视频录播（非公开 YouTube 播放列表）、Slack，以及第 7 讲并列的 PDF（私有 GitHub 仓库）属于受限资源，已在相应位置标注。*

{% endraw %}
