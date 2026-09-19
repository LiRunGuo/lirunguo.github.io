---
title: "附录 A：Exam 1 复习笔记（Lecture 1–13）"
collection: course-notes
chapter: true
permalink: /course-notes/cmu-15418-parallel-computer-architecture/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cmu-15418-parallel-computer-architecture/) · [← l26](/course-notes/cmu-15418-parallel-computer-architecture/l26)

{% raw %}
## 附录 A：Exam 1 复习笔记（Lecture 1–13）

> 依据官方公开讲义 `lectures/revision_exam1.pdf`（24 页，首页标注 Fall 2023/Fall 2024 版本，属讲义沿用）整理，并对照 `lectures/01`–`13` 的公开讲义原文补全。
> 考试形式：当堂闭卷，可用 **一张 A4 双面手写纸**，必须黑/蓝笔，无计算器/电子设备；题型以**简答**与**选择+解释**为主（解释的分值远高于选项）。

---

### A.1 官方复习幻灯片覆盖清单（逐页）

| 复习页 | 内容 | 对应讲次 |
|---|---|---|
| 1 | 考试细节 | — |
| 2–3 | ISPC `sinx`：**interleaved（交错）vs blocked（分块）** 分配 | L3、L4 |
| 4–5 | CUDA 的 grid/block/thread、`__global__`、bulk launch | L5、L6 |
| 6 | CUDA 同步：`__syncthreads()`、atomic、kernel 返回的隐式全局屏障 | L5、L6 |
| 7 | 共享地址空间 SPMD 求解器（lock + barrier + 收敛判定） | L7 |
| 8+ | 同步原语、一致性协议、互连网络等 | L10–13、L15 |

---

### A.2 核心公式一张表

| 公式 | 表达式 | 用途 |
|---|---|---|
| 加速比 | `Speedup(P) = T₁ / T_P` | 基本定义；基线必须是**最好的串行实现** |
| 效率 | `Efficiency(P) = Speedup(P) / P` | 判断"快但低效" |
| Amdahl | `Speedup ≤ 1 / (S + (1−S)/P)`，`S` = 串行比例 | 加速比硬上限 `1/S` |
| Amdahl 速度上限 | `lim_{P→∞} Speedup = 1/S` | 判断是否值得继续加核 |
| Work-Span 界 | `T_P ≤ T₁/P + T∞` | 并行时间下界 |
| 并行度 | `Parallelism = T₁ / T∞` | 达到线性加速所需的最大核数 |
| 算术强度 | `AI = FLOPs / Bytes` | Roofline 横轴 |
| Roofline | `P ≤ min(P_peak, AI × BW)` | 判断算力受限 or 带宽受限 |
| Ridge point | `AI* = P_peak / BW` | AI < AI* 则带宽受限 |
| 延迟-带宽 | `T(n) = α + n/β` | 消息传递/互连/通信代价 |
| 平均访存时间 | `AMAT = HitTime + MissRate × MissPenalty` | cache 性能 |
| 能耗缩放 | `P_dyn ∝ C·V²·f` | 功率墙的成因 |
| 加速比（KS 模型，含并行开销） | `S = 1 / (S + (1−S)/P + κ)` | 解释次线性加速 |

---

### A.3 高频考点一：ISPC / SPMD 与 SIMD

**官方复习题（slide 2–3 原文）**：给出 `sinx` 的 two versions，要求分析交错 vs 分块分配。

**交错（interleaved）版本**：
```text
gang 有 programCount 个实例（如 AVX2 下为 8）
  实例 0 处理 idx = 0, 8, 16, 24, ...
  实例 1 处理 idx = 1, 9, 17, 25, ...
  ...
  → 每次对 idx 的访问是一段连续的 8 个 float
  → 生成 packed load / packed store（单条 vmovups 搬 32 字节）
```
**分块（blocked）版本**：
```text
  实例 0 处理 idx = 0 .. N/8−1（连续 128 个元素）
  实例 1 处理 idx = N/8 .. 2N/8−1
  ...
  → 同一时刻 8 个实例访问的地址相距 N/8 个 float
  → 生成 gather / scatter（AVX2 的 vgatherdps 极慢，约比 packed load 慢一个数量级）
```
**结论**：**交错分配在 SIMD 上通常更优**，因为一个 gang 的并发访问是连续的，能够合并（coalesce）成整宽向量访存。这也解释了为什么 ISPC 的 `foreach` 默认采用交错分配。

**必答要点**：
- `programCount`：gang 中同时执行的实例数（uniform 值），由目标 ISA 的向量宽度决定（SSE 4、AVX2 8、AVX-512 16）。
- `programIndex`：当前实例在 gang 中的 id（**varying** 值）。
- `uniform`：类型修饰符，所有实例共享同一值，**纯粹是优化提示，不影响正确性**。
- gang 执行模型：所有实例执行同一条指令流（coherent execution），分支导致**发散（divergence）**，硬件用掩码串行化各分支路径。

**常考陷阱**：`uniform` 不是正确性要求（写成 varying 也正确，只是慢）；`programCount` 与 `programIndex` 不能用于标量代码。

---

### A.4 高频考点二：CUDA 线程层次与同步

**官方复习题（slide 4）**：`gridDim`、`blockIdx`、`blockDim`、`threadIdx` 各是什么？**为什么没有 `gridIdx` 和 `threadDim`？**

- 四个内建变量：
  - `gridDim`：整个 grid 的维度（block 数量）
  - `blockIdx`：当前 block 在 grid 中的索引
  - `blockDim`：一个 block 的维度（线程数量）
  - `threadIdx`：当前线程在 **block 内**的索引
- 全局线程 id：`int i = blockIdx.x * blockDim.x + threadIdx.x;`
- **为什么没有 `gridIdx`**：一个 kernel 只由一个 grid 启动（"launch a grid of CUDA thread blocks"），因此 grid 层之上没有"第几个 grid"的概念；grid 是启动单位而不是层级中的一层。
- **为什么没有 `threadDim`**：线程是层级的最底层，其下没有子层级，所以不需要"我的维度"这个概念；反之 block、grid 都有子元素，因此它们有 dim。

**同步三件套（slide 6）**：
1. `__syncthreads()`：**block 内屏障**，等待同一 block 的所有线程到达。**不能跨 block**。
2. 原子操作：`atomicAdd/atomicCAS/atomicExch/...`，可用于 **global memory 与 shared memory**。
3. **kernel 返回时的隐式全局屏障**：所有线程结束时 kernel 才算完成，主机侧 `cudaDeviceSynchronize()` 或拷贝回数据时等待。CUDA **不提供** kernel 内部的全局（跨 block）屏障——这正是"用 kernel 分解代替全局同步"这一设计模式的原因。

**经典错误**：依赖 block 间顺序（例如让 block 0 写、block 1 读，用标志位自旋等待）。两个 block 可能被调度到同一个 SM 并占满资源，导致"自旋的 block 等着永不被调度的 block"⇒ **死锁**。正确做法是拆成两个 kernel。

---

### A.5 高频考点三：共享地址空间求解器（SPMD）

**官方复习题（slide 7）**：给出二维网格求解器代码，问其中的错误/性能问题。逐项排查：

| 代码片段 | 问题 |
|---|---|
| `diff = 0.f;` 在 barrier **之后**、每个线程都写 | 应在线程**私有**变量上累加（`myDiff`），否则多个线程写同一地址 ⇒ 真共享写竞争；且 `diff = 0.f` 与后面的 `diff += myDiff` 存在数据竞争 |
| `lock(myLock); diff += myDiff; unlock(myLock);` | 每次迭代都加锁 ⇒ 若改成"每线程只加一次锁"，成本降一个数量级（讲义原文："Now only lock once per thread, not once per (i,j) loop iteration!"） |
| `A[i,j] = 0.2f * (A[i-1,j] + ...)` 原地更新 | **原位（in-place）更新**：读旧值写新值，且依赖邻居在同一轮中已被更新的值 ⇒ 结果依赖线程执行顺序（**竞态**）。正确做法用红黑着色或双缓冲 |
| `barrier(myBarrier, NUM_PROCESSORS)` 三次 | 屏障 1 可与其他同步合并（讲义演示用 `diff[3]` 轮转把三个屏障降为一个） |
| `if (diff/(n*n) < TOLERANCE) done = true;` | 每个线程都执行判断并写 `done`，且没有把判断结果广播给所有线程（需要 barrier 保证所有线程看到同一值）⇒ 可能"有些线程已退出、有些还在循环" |
| `float myDiff` 在外层声明又在循环内重复声明 | 遮蔽（shadowing）问题 |

**红黑着色解法**：把格点按 (i+j) 的奇偶分为两组。同色格点的 5 点 stencil 只依赖异色格点，因此**同色格点之间无依赖**，可完全并行；一次 sweep 只需两个相位 + 一次屏障。

---

### A.6 高频考点四：性能模型（每考必有）

**必须能在 30 秒内手算的量**：

**例 1（计算峰值）**：16 核 × 3.0 GHz × 8 宽 AVX2 × 2（FMA）= 768 GFLOP/s。
**例 2（带宽上限）**：某程序算术强度 4 FLOP/Byte，机器峰值 100 GFLOP/s、带宽 25 GB/s。`AI* = 100/25 = 4`，恰在拐点上，两者都可成为瓶颈；实测通常介于 `min(100, 4×25) = 100` 与实际带宽利用率之间。
**例 3（Amdahl）**：串行比例 S=5%，64 核最多加速 `1/0.05 = 20×`，实际 `1/(0.05 + 0.95/64) = 13.9×`。**若负载不均导致某核多干 20% 的活（等效串行比例约 5%），加速比同样被限制在约 14×**。
**例 4（work-span）**：一个算法的 T₁ = 1000 秒、T∞ = 10 秒 ⇒ 并行度 100；在 8 核上的时间下界 `1000/8 + 10 = 135` 秒 ⇒ 最高加速 7.4×，达不到 8×。

**常考判断题**：
- "我的代码在 8 核上加速 6.5×，所以 64 核上会加速 52×" ⇒ 错。要先看**并行度（T₁/T∞）够不够**，再看串行比例。
- "超线性加速不可能" ⇒ 错。当并行版本获得更好 cache 局部性（例如多个核的 cache 合起来能装下工作集）时可能出现 `Speedup > P`，此时应报告绝对时间与内存层次影响。
- "效率 100% 才算好" ⇒ 错。若 8 核用 4 倍功耗换 3 倍性能，从"性能/功耗"角度看是**变差**的；效率要与目标（时间还是能耗）一起谈。

---

### A.7 高频考点五：硬件结构

**流水线/ILP（L2）**：
- 数据冒险（RAW/WAR/WAW）、控制冒险、结构冒险三类；旁路转发（forwarding）解决大部分 RAW；WAR/WAW 靠**寄存器重命名**消除。
- OoO 核心 = 前端（取指/译码/重命名）+ 指令窗口/ROB（乱序发射、顺序提交）+ 后端（发射端口/执行单元）。
- 提取性能的两把尺子：**延迟界**（依赖链长度 × 延迟）与**吞吐界**（操作数 / 发射宽度）；实际性能 = 两者中较慢者。
- 为什么 ILP 撞墙：窗口 `O(W²)` 的复杂度、单程序内可用并行度有限、频率受 `C·V²·f` 限制。

**现代多核（L3）**：
- 四个概念：**多核**（线程级并行）、**SIMD**（数据级并行）、**内存延迟**、**内存带宽**。
- 跑满机器的条件：`核数 × 每核 IPC × 向量宽度 × SMT` = 需要的独立工作份数（例如 16 核 × 8 宽 × 4 线程 = 512 份）。
- 延迟可以**隐藏**（预取、硬件多线程、ILP），带宽**只能少访问**。
- Roofline 判断：`AI < P_peak/BW` ⇒ 带宽受限，优化方向是"减少字节"（分块、融合、压缩数据），而不是"减少指令"。

**互连网络（L10）**：
- 拓扑比较维度：**成本**（连线数/端口数）、**延迟**（直径、平均跳数）、**带宽**（二分带宽）、**可扩展性**、**路由复杂度**。
- 关键数字感：crossbar 成本 `O(N²)`（N 端口）；ring 直径 `N/2`；mesh 直径 `2(√N−1)`；fat tree 二分带宽最优但成本高；Omega/蝶形网络用 `log N` 级换低成本但有阻塞。
- 流控三级粒度：message → packet → flit；store-and-forward 延迟 ∝ 跳数 × 包长，cut-through/wormhole 大幅降低；虚通道（VC）解决队头阻塞（HOL）。
- 公式：`T = hops × (T_router + T_link)`，理想情况下 `T ≈ α + n/β`。

**缓存一致性（L11–13）**：
- 为什么需要：私有 cache 复制数据 ⇒ 共享内存语义被破坏。
- 三条要求：**写传播（write propagation）**、**写串行化（write serialization）**。等价不变量：**SWMR**（Single-Writer-Multiple-Reader）+ **Data-Value**。
- **监听（snooping）**：所有 cache 监听总线，state machine per cache line。
  - **MSI**：M（Modified，独占且脏）、S（Shared，多副本干净）、I（Invalid）。写 miss ⇒ BusRdX 使他人失效；读 miss ⇒ BusRd。
  - **MESI**：增加 **E（Exclusive clean）**：独占且与内存一致，可**静默**升级为 M 而不发总线事务 ⇒ 消除"先读后写私有数据"的 upgrade 开销。**这是 MESI 相对 MSI 的核心收益。**
  - **MOESI / MESIF**：O（Owned，脏但可共享，负责供给数据，免写回内存）；F（Forward，指定唯一响应者，减少重复响应）。
- **状态迁移必背表（MESI，本地请求）**：

| 当前状态 | 事件：PrRd | PrWr | BusRd（他核读） | BusRdX（他核写） |
|---|---|---|---|---|
| I | → S（发 BusRd） | → M（发 BusRdX） | — | — |
| S | 命中 | → M（发 BusRdX） | 保持 S | → I |
| E | 命中 | → M（**静默**，无总线事务） | → S（发 flush） | → I |
| M | 命中 | 命中 | → S（**发 flush** 并降级） | → I（**发 flush**） |

- **伪共享（false sharing）**：不同变量落在同一条 cache line，被不同核写 ⇒ 一致性流量与真共享相同。**这是"人为通信"，不是算法固有的。**修法：padding 对齐、每线程本地副本 + 最后归并。
- **目录协议（L12）**：每 cache line 一个目录项，含 `P` 个 presence bit + dirty bit；存储在 home node。读 miss 2 跳；写 miss 需要失效所有 sharer（`2 + 2k` 条消息、4 跳关键路径）。优化：**limited pointer**（只记录 k 个 sharer，溢出则回退广播）、**sparse directory**（只为在 cache 中的行保留项）、**intervention forwarding**（owner 直接回数据，降低关键路径）。
- **监听实现（L13）**：原子总线 vs **拆分事务总线**（请求/响应分离 + 请求表 + NACK 流控）；写回缓冲（write-back buffer）也要参与监听；**取数死锁（fetch deadlock）**、**活锁**、**饥饿**；**写提交（commit）≠ 写完成（complete）**；缓冲带来**缓冲死锁**，解法是请求队列与响应队列分离。

---

### A.8 高频考点六：并行编程基础与性能优化（L7–L9）

**四步法**：Decomposition（分解）→ Assignment（分配）→ Orchestration（编排）→ Mapping（映射）。

**工作分配**：
- 静态 blocked：每个 worker 拿连续一块。优点：简单、零运行时开销、访存局部性好。缺点：**负载不均时致命**（Amdahl 放大）。
- 静态 interleaved：每个 worker 拿 `i % P`。负载更均匀、cache 利用可能更好，但**访存不连续**（共享内存下无妨，消息传递下会产生大量人为通信）。
- 动态：共享计数器 + chunk / 工作队列。开销 = 每次取号的**原子操作与竞争**（约 20–500 ns）。粒度公式：`k ≥ c·P/t`（避免抢任务的成本超过做任务）与 `k ≤ M/(P)`（保证足够多的任务数）。
- 工作窃取（work stealing）：每 worker 一个 deque，本地从尾部 push/pop，窃取者从头部拿；**continuation stealing**（先执行 spawn 的子任务，父任务留给窃取者）保证串行语义 + 减少队列操作。

**通信代价**：`T(n) = T₀ + n/B`，`T₀` = overhead + occupancy + network delay。**重叠加计算（overlap）**可以把通信藏起来（如 MPI `Isend/Irecv` + `Waitall`）。
**四类优化（讲义"四个 C"的实践）**：
1. **Blocking（分块）**：提高重用距离内的时间局部性。
2. **Fusion（循环融合）**：把多遍遍历合成一遍，算术强度翻倍（例：三遍遍历 AI = 1/3 → 融合后 3/5）。
3. **Sharing（共享）**：把访问同一数据的任务放到同一执行单元（CUDA block 共享 L1/shared）。
4. **粒度（消息大小）**：cache line（64 B）造成的边界浪费与伪共享。

**竞争（contention）**：热点资源吞吐上限 = `1/t_crit`（`t_crit` = 该资源的关键路径延迟）。对策：复制资源（每线程私有 + 归并）、细粒度锁、分布式队列，或**把串行化改写为"数据并行 + 排序"**（讲义分桶例子）。

---

### A.9 一页纸速记（可抄到 A4 上）

```text
【三把尺子】
  Work T₁ / Span T∞ / 并行度 = T₁/T∞
  加速比 ≤ 1/(S + (1−S)/P)；上限 1/S
  AI = FLOP/Byte；P ≤ min(P_peak, AI×BW)；AI* = P_peak/BW

【三个墙】
  功耗墙 P ∝ C·V²·f  → 频率停涨 → 多核
  ILP 墙（窗口 O(W²)、程序并行度有限）→ 需要显式并行
  带宽墙（BW 增长 << 峰值算力）→ 大多数程序带宽受限

【硬件层次】
  寄存器 → L1(~4cyc) → L2(~12) → L3(~38) → DRAM(~200+)
  SIMD 宽度：SSE 4 / AVX2 8 / AVX-512 16 (float)
  GPU：SM = 4 sub-core × 32 lane = 128 lane；warp = 32 线程
  跑满机器所需独立工作 = 核数 × 每核IPC × 向量宽度 × SMT

【一致性】
  MSI：M(独占脏) S(共享净) I；MESI 加 E(独占净，可静默升级)
  读 miss → BusRd；写 miss → BusRdX；M 态被监听 → flush
  伪共享 = 不同变量同一条 line 被多核写 = 人为通信
  目录：presence bits + dirty bit；读 2 跳、写 2+2k 消息 4 跳
  limited pointer / sparse directory / intervention forwarding

【同步】
  锁性能看三段：acquire / waiting / release
  TAS → TTAS → ticket → 数组锁 / MCS 队列锁（O(1) 流量 + FIFO）
  屏障：集中式 sense-reversal（两次自旋变一次）/ 树形 lg P
  SC 太慢（写缓冲必须停顿）→ TSO/PSO/WO → 用 fence 与 acquire/release 补
```

---

### A.10 自测 12 题

1. **Q**：为什么 MESI 中的 E 态能提升性能？
   **A**：E 态表示"独占且干净"，处理器可以**不发任何总线事务**直接把 E 升级为 M（静默升级）。这消除了"先读后写一块私有数据"时的 BusRdX 事务；MSI 下同样的访问模式会产生读 miss（BusRd）+ 写 miss（BusRdX）两次事务。

2. **Q**：一个程序在 8 核上加速 7.2×，在 16 核上加速 7.4×。最可能的原因是什么？如何验证？
   **A**：串行比例或并行度不足。由 Amdahl 得 `S ≈ 1/7.4 − 小量 ≈ 13%`，与 8 核的 7.2× 一致（`1/(0.13+0.87/8) = 6.9`）。验证方法：用 work-span 分析算出 T∞，或做**强扩展/弱扩展曲线**、最小二乘拟合 Karp-Flatt 度量 `e = (1/S − 1/P)/(1 − 1/P)`（`e` 接近常数说明并行开销固定占比）。

3. **Q**：为什么"给数组每个元素加 1"在 8 核上只有约 2× 加速？
   **A**：算术强度极低（每 4 字节读 + 4 字节写只有 1 次加法 ⇒ 0.125 FLOP/Byte），远低于 ridge point，**带宽受限**；多核共享同一条内存总线，带宽不会随核数线性增长。实测受 STREAM 类上限约束。

4. **Q**：写出 ISPC 中交错分配与分块分配的差别，并说明哪个在 AVX2 上更快及原因。
   **A**：见 A.3。交错更快，因为同一 gang 的并发访存地址连续，可合并为 packed load（`vmovups` 32 字节）；分块导致地址相距很远，需要 `vgatherdps`。

5. **Q**：`__syncthreads()` 能不能用来同步两个不同的 block？
   **A**：不能。它只同步 block 内的线程。跨 block 同步必须用 kernel 分解（kernel 返回是全局隐式屏障）或原子操作 + 全局标志（但这有死锁风险，因为 block 不保证并发驻留）。

6. **Q**：目录协议的写 miss 需要多少条消息？关键路径多少跳？
   **A**：`2 + 2k` 条消息（k = sharer 数）：请求→home、home 失效 k 个 sharer、k 个 ack 回 home、home 授权请求者（或用 intervention forwarding 让 owner 直接转发数据，关键路径降为 3 跳）。关键路径 4 跳（请求→home→sharer→home→请求者）。

7. **Q**：为什么"动态分配"不总是优于"静态分配"？
   **A**：动态分配每次取任务都要做原子操作或加锁，有 20–500 ns 的竞争成本；若任务粒度太细（任务本身只有微秒级），调度开销会淹没收益。此外动态分配可能破坏局部性（任务与数据的亲和性丢失）。

8. **Q**：解释"延迟可以被隐藏，带宽不能"。
   **A**：一次访存的延迟可以用**其他独立工作**填充（ILP、硬件多线程、预取），所以延迟是"能否找到并行工作"的问题；带宽是单位时间能搬运的字节数上限，是**资源吞吐的物理上界**，除了减少访问量（分块/融合/降低精度）无法绕过。

9. **Q**：什么是伪共享？为什么 padding 能解决它？
   **A**：多个核写**不同变量**但落在**同一条 64 字节 cache line** 上，导致该 line 在所有写者之间反复迁移（ping-pong），一致性流量与真正共享同一变量时相同。padding 把每个线程的变量放进各自的 cache line（`alignas(64)`），使迁移消失。

10. **Q**：为什么"写提交（commit）"不等于"写完成（complete）"？
    **A**：提交是指令从流水线退休（架构状态已更新），而完成是指该写在一致性协议中取得所有权并实际生效。x86 的 store buffer 使得**本核看到的自己写的值**早于**其他核能看到的时间**；这也是 TSO 的成因之一。

11. **Q**：FP32 矩阵乘在 V100（峰值 ~15.7 TFLOP/s、带宽 900 GB/s，ridge ≈ 17.4）上，若分块宽度 L=32，能达到多少性能？
    **A**：分块后算术强度 `AI = L/4 = 8 FLOP/Byte < 17.4` ⇒ **带宽受限**，`P ≤ 8 × 900 GB/s = 7.2 TFLOP/s`（约为峰值的 46%）。若把 L 提到 64，AI = 16 ≈ 拐点；L = 128 时 AI = 32 > 17.4，转为算力受限。这也解释了为什么"分块尺寸"是 GPU GEMM 最重要的调优旋钮。

12. **Q**：8 核机器上某同步程序加速仅 3×，用 PMU 看到 `BusRdX` 事件随核数平方增长。最可能的两个原因？
    **A**：① 伪共享（不同线程写同一 cache line 的不同字节）；② 集中式同步原语（全局自旋锁的 TAS 每次尝试都发 BusRdX，争用流量 O(P²)）。修法：`alignas(64)` 隔离 + 改 TTAS/ticket/MCS 锁或每线程私有累加。

---

## 第三部分：速查表与附录

### 速查表 A：并行编程模型速查表

| 模型 | 代表 API | 地址空间 | 并行单位 | 通信方式 | 同步原语 | 最适合的问题 |
|---|---|---|---|---|---|---|
| **共享地址空间 / threads** | pthreads、C++ `std::thread` | 单一共享 | OS 线程 | 隐式（load/store） | mutex、condvar、barrier、原子 | 不规则、共享数据结构的通用并行 |
| **OpenMP** | `#pragma omp parallel for`、`omp task` | 单一共享 | 线程（fork-join） | 隐式 | `critical`、`atomic`、`barrier`、`reduction` | 循环级数据并行、任务图 |
| **ISPC（SPMD→SIMD）** | `foreach`、`programCount`、`programIndex` | 单一共享 | gang 实例（= SIMD lane） | 隐式 | `reduce_*`、`forall` | 规则数据并行，需要榨干 SIMD |
| **Cilk / TBB** | `cilk_spawn`、`cilk_sync`、`parallel_for` | 单一共享 | 线程 + 工作窃取 | 隐式 | `cilk_sync`、reducer | 递归分治、不规则任务图 |
| **CUDA** | `<<<grid, block>>>`、`__global__` | 主机 + 设备（分离） | warp/block/grid | 显式 `cudaMemcpy` + 共享内存 | `__syncthreads()`、atomic、kernel 边界 | 大数据并行、规则计算、矩阵/卷积/归约 |
| **MPI（消息传递）** | `MPI_Send/Recv/Isend/Irecv/Allreduce` | 每进程私有 | 进程（可跨节点） | 显式消息 | `MPI_Barrier`、集合通信 | 集群规模、分布式内存、可预测的规则通信 |
| **数据并行 / stream** | ISPC `foreach`、CUDA、Halide/Triton | 单一（逻辑） | 元素/瓦片 | 极少（仅 gather/scatter/边界） | 无（除归约） | 逐元素、map/reduce、算子融合 |

**选择决策树**：

```text
数据能否装进单节点内存？
├── 否 → MPI（+ 节点内 OpenMP/CUDA 混合）
└── 是 → 计算是否规则、数据量是否巨大（>10⁷ 独立元素）？
         ├── 是 → GPU（CUDA）；若只需 SIMD 加速 → ISPC
         └── 否 → 是循环级数据并行吗？
                  ├── 是 → OpenMP parallel for / reduction
                  └── 否 → 是递归分治或动态任务图吗？
                           ├── 是 → OpenMP task / Cilk / TBB（工作窃取）
                           └── 否 → 共享数据结构的细粒度操作？→ pthreads + 细粒度锁/无锁
```

**跨模型对照：同一个"向量加"的六种写法**

```c
// 1. pthreads：手动分块 + join
for (t = 0; t < P; t++) pthread_create(&th[t], NULL, worker, &arg[t]);
for (t = 0; t < P; t++) pthread_join(th[t], NULL);

// 2. OpenMP：一个 pragma
#pragma omp parallel for schedule(static)
for (i = 0; i < N; i++) c[i] = a[i] + b[i];

// 3. ISPC：SPMD gang，编译器生成 SIMD
foreach (i = 0 ... N) c[i] = a[i] + b[i];

// 4. CUDA：成千上万轻量线程
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
add<<<(n+255)/256, 256>>>(a, b, c, n);

// 5. MPI：每个进程处理自己的切片，末尾可能需边界交换
int lo = rank * (n/P), hi = lo + n/P;
for (i = lo; i < hi; i++) c[i] = a[i] + b[i];
// 若 c 的邻居需要 a 的 halo，则 MPI_Sendrecv

// 6. C++17 并行 STL：最高层抽象，实现可能是线程池 + SIMD
std::transform(std::execution::par_unseq, a, a+N, b, c,
               [](float x, float y){ return x + y; });
```

---

### 速查表 B：性能公式速查表

#### B.1 可扩展性

| 名称 | 公式 | 说明 / 陷阱 |
|---|---|---|
| 加速比 | `Speedup(P) = T₁ / T_P` | `T₁` 必须是**最优串行算法**的时间 |
| 效率 | `E(P) = Speedup(P) / P` | 与能耗效率区分开 |
| Amdahl（固定问题规模） | `Speedup(P) = 1 / (S + (1−S)/P)` | 上限 `1/S`；加核收益递减 |
| Gustafson（固定时间） | `ScaledSpeedup(P) = S + P(1−S)` | 问题规模随 P 增长时更乐观 |
| Karp-Flatt 度量 | `e = (1/Speedup − 1/P) / (1 − 1/P)` | `e` 随 P 上升 ⇒ 并行开销在增长；`e` 恒定 ⇒ 串行比例主导 |
| Work-Span 下界 | `T_P ≥ max(T₁/P, T∞)` | 松弛版：`T_P ≤ T₁/P + T∞` |
| 并行度 | `T₁ / T∞` | 达到线性加速所需的最大核数 |
| 贪婪调度 | `T_P ≤ T₁/P + T∞`（期望/上界） | 工作窃取达到此界 |
| 超线性加速 | `Speedup > P` | 由于 cache/内存层次效应；需测量绝对时间解释 |
| 固定问题陷阱 | 6/8/12/18 MP3 编码例 | 小问题加速一般，大问题反而接近线性 |

#### B.2 内存与带宽

| 名称 | 公式 | 说明 |
|---|---|---|
| AMAT | `AMAT = HitTime + MissRate × MissPenalty` | 或 `= HitTime_L1 + MR_L1 × (HitTime_L2 + MR_L2 × ...)` |
| 平均访存停顿 | `Stall = Misses/Instr × MissPenalty` | 与 CPI 相加得总 CPI |
| 算术强度 | `AI = FLOPs / Bytes transferred` | 分母是**实际搬到片上**的字节 |
| Roofline | `Attainable = min(P_peak, AI × BW_peak)` | 对数-对数图上的斜线 + 平台 |
| Ridge point | `AI* = P_peak / BW_peak` | 例：V100 `15.7e12/900e9 ≈ 17.4` |
| Little 定律 | `并发度 = 延迟 × 吞吐` | 例：要维持 8 次未完成访存 × 100 ns ⇒ 需 80 ns/次的吞吐能力 |
| 通信延迟-带宽 | `T(n) = α + n/β` | `α` 零负载延迟，`β` 渐近带宽 |
| 交叉点 | `n* = α·β` | `n > n*` 时带宽主导 |
| 带宽受限时间 | `T ≥ Bytes_total / BW` | 常用于给并行程序算下界 |
| 表面-体积比 | `Comm/Comp ∝ 1/n`（3D 分块） | 分块越大通信占比越低，但并行度下降 |
| TLB reach | `reach = entries × page_size` | 4 KB × 64 项 = 256 KB；2 MB × 64 = 128 MB |
| 页走代价 | `walks × levels × mem_latency` | x86-64 四级：4 次串行访存 |

#### B.3 能耗与功率

| 名称 | 公式/数值 | 说明 |
|---|---|---|
| 动态功率 | `P_dyn ∝ C·V²·f` | 电压下降是节能主因；Dennard 缩放终止于 ~2003 |
| 能量-延迟积 | `EDP = E × T` | 移动端的关键指标 |
| 整数运算能耗 | ~1 pJ（32 位加法） | 数量级参考 |
| 寄存器访问 | ~1–2 pJ | — |
| 片上 SRAM 访问 | ~26 pJ（8 KB 内） | 与 DRAM 相差 ~46× |
| DRAM 读写 | ~1200 pJ（64 bit LPDDR） | **数据搬运才是能耗大头** |
| 专用化能效收益 | 10×（GPU 核）/ 20×（DSP/域专用）/ 100–1000×（ASIC） | 解释 L18 的动机 |

#### B.4 计算峰值

```text
峰值 FLOPS = 核数 × 时钟频率 × 每周期 FMA 数 × 2 × SIMD 宽度
  例：16 核 × 3.0e9 × 1 FMA × 2 × 8 (AVX2) = 768 GFLOP/s
  例：H100 SXM：132 SM × 128 FP32 lane × 2 × ~1.98 GHz ≈ 67 TFLOP/s (FP32)
                Tensor Core FP16（稀疏）可达约 2000 TFLOP/s
```

---

### 速查表 C：硬件架构参数速查表

#### C.1 存储层次典型延迟与容量

| 层次 | 典型延迟（周期） | 典型延迟（ns @3GHz） | 典型容量 | 谁管理 |
|---|---|---|---|---|
| 寄存器 | 1 | 0.3 | ~64–256 × 8 B | 编译器 |
| L1 (per core) | 4 | 1.3 | 32–48 KB | 硬件 |
| L2 (per core) | 12 | 4 | 512 KB–2 MB | 硬件 |
| L3 (shared) | 38 | 12 | 8–64 MB | 硬件 |
| DRAM | 200–350 | 70–120 | GB | 硬件 + OS |
| 本地磁盘/SSD | 10⁵–10⁷ | — | TB | OS |
| 数据中心网络 | 10⁴–10⁵ | µs 级 | — | 系统 |

> 记忆要点：**L1→L2→L3→DRAM 的数量级是 4 / 12 / 38 / 200+ 周期**（约 3× 递增）。这些数字在考试里常用来做定量估算。

#### C.2 CPU vs GPU 结构对比

| 维度 | 多核 CPU | GPU |
|---|---|---|
| 核心 | 少量强核（4–64），OoO、大乱序窗口、深流水 | 大量弱核（256–132 SM × 128 lane），in-order、小窗口 |
| 控制开销 | 预测、重命名、乱序执行消耗大量晶体管 | 最小化控制，把面积留给 ALU |
| 延迟隐藏 | ILP + OoO + 有限 SMT（2–8 线程/核） | 大量硬件多线程（每 SM 最多 64 warp） |
| SIMD 宽度 | 4–16（AVX-512） | 32（warp 宽度） |
| 内存 | 大 cache（MB 级）、NUMA 感知 | 小 cache、**显式** shared memory + register 分块 |
| 编程模型 | 线程 + 共享内存 | SIMT：grid/block/warp/shared |
| 吞吐/延迟比 | 中等 | 极高（面向吞吐，牺牲单线程延迟） |
| 典型峰值 | ~1 TFLOP/s | ~10–1000 TFLOP/s |

#### C.3 SIMD 宽度

| ISA | 向量宽度 | float 数量 | 引入 |
|---|---|---|---|
| SSE | 128 bit | 4 | 1999 |
| AVX | 256 bit | 8 | 2011 |
| AVX2 | 256 bit（整数也 256） | 8 | 2013 |
| AVX-512 | 512 bit | 16 | 2016 |
| NEON (ARM) | 128 bit | 4 | — |
| SVE (ARM) | 128–2048 bit（可变） | 4–64 | 2016+ |

#### C.4 一致性协议状态速查

| 协议 | 状态 | 与前一版的差异 | 主要收益 |
|---|---|---|---|
| MSI | M, S, I | 基础 | 最小协议，但私有数据"先读后写"需 2 次事务 |
| **MESI** | + E | E = 独占且干净 | **静默升级**，消除 upgrade 事务 |
| MESIF | MESI + F | F = 唯一响应者 | 减少"多个 S 副本都响应"的重复流量（Intel） |
| MOESI | MSI + O | O = 脏且共享 | 共享脏数据由 O 态 cache 供给，**免写回内存**（AMD） |
| Dragon (update) | E, S, Sm, M | 更新而非失效 | 写少读多的共享数据可省失效流量 |

#### C.5 内存一致性模型

| 模型 | 允许的重排 | 代表 | 需要的 fence |
|---|---|---|---|
| SC | 无 | 理论模型 | 全部 |
| TSO | StoreLoad（WX→RY） | x86 | `MFENCE` / `lock` 指令 |
| PSO | 再加 StoreStore | SPARC | `MEMBAR` |
| WO / RC | 更松，仅同步操作有序 | ARM/POWER 早期 | `DMB`/`ISYNC`，现用 acquire/release |
| SC-for-DRF（C++11/C11/Java） | 有竞争则无保证 | 现代语言内存模型 | `atomic_thread_fence`，`memory_order_acquire/release` |

---

### 速查表 D：优化技巧速查表

#### D.1 决策表：先量什么，再改什么

| 症状 | 优先怀疑 | 诊断手段 | 修法 |
|---|---|---|---|
| 加核不加速（曲线平） | 串行段 / 并行度不足 | work-span 分析；Karp-Flatt `e` | 减少串行段、提高并行度、重组算法 |
| 加核**反而**变慢 | 竞争 / 伪共享 / 一致性流量 | `perf c2c`、PMU 的 `BusRdX`、`llc-miss` | padding、私有化 + 归并、细粒度无锁 |
| 核多了但性能饱和 | 内存带宽耗尽 | `perf stat` 的 DRAM 字节 / 带宽 | 分块（blocking）、循环融合、降低数据精度、压缩 |
| 时间随 N 呈台阶式跳变 | cache 容量/TLB 溢出 | 改变问题规模做扫描；大页 | 分块、循环重排、huge pages、数据布局重组 |
| 随机性大、重复测量差异大 | 调度抖动 / NUMA 放置 / 首次触碰 | 多次运行取中位数、`numactl`、first-touch | 并行初始化、绑核、`numactl --interleave` |
| 热点集中在某个函数但不知原因 | 未发现真正瓶颈（抽象陷阱） | 高水位实验、Roofline 定位 | 见 D.3 |

#### D.2 优化技巧清单（按影响量级排序）

**内存/带宽类（通常收益最大）**
1. **分块（blocking/tiling）**：把工作集切到能装进 cache 的瓦片，循环顺序按瓦片嵌套。
2. **循环融合（fusion）**：把多遍遍历合并成一遍，`AI` 直接翻倍。注意寄存器压力与可读性。
3. **重用（reuse）与数据布局**：结构体数组 AoS → 数组结构体 SoA（利于向量化与合并访存）。
4. **消除人为通信**：padding 对齐（`alignas(64)`）、每线程私有副本、减少 halo 交换频率。
5. **提高消息粒度**：小消息合并成大消息（`α` 摊薄），例：把每次 8 B 的交换改成每次 4 KB。
6. **大页 / TLB 优化**：`madvise(MADV_HUGEPAGE)`、`mmap` 2 MB 页；避免页步长（page-stride）访问大数组。

**并行结构类**
7. **提高并行度**：把串行段并行化；用树形归约代替线性归约（`span` 从 `O(P)` 降到 `O(log P)`）。
8. **负载均衡**：动态分配、工作窃取、长任务优先（LPT）、半静态自适应重划分。
9. **减少同步次数**：合并屏障（`diff[3]` 轮转）、批量提交、每线程一次锁而非每次迭代。
10. **overlap 通信与计算**：`MPI_Isend/Irecv` + `Waitall`；CUDA stream 与 `cudaMemcpyAsync`。
11. **避免伪共享**：写操作隔离到各自 cache line；读共享数据则无妨。

**指令/SIMD 类**
12. **向量化**：`-O3 -march=native`、`#pragma omp simd`、ISPC `foreach`；检查生成汇编是否有 `vfmadd*`。
13. **消除分支发散**：GPU 上让整个 warp 走同一路径；用算术或掩码代替分支（`cmov`）。
14. **合并访存（coalescing）**：让相邻线程访问相邻地址（CUDA 尤其重要）；避免 stride 访问。
15. **ILP 与展开**：多条独立累加链打破延迟界；`#pragma unroll`。
16. **对齐**：`_mm_malloc(..., 32)` / `aligned_alloc(64, ...)`，避免跨 cache line 的向量访问。

#### D.3 高水位实验（High Watermark）—— 判断"我的实现到底有多好"

> 方法：构造一个**只能更快**的"作弊版本"，测出物理上限，再看自己离它多远。

| 实验 | 做法 | 揭示 |
|---|---|---|
| 删除计算 | 保留访存、把计算换成 `A[0] = ...` | 计算的真实成本 |
| 删除访存（伪造） | 只用已加载值反复计算 | 访存/带宽的真实成本 |
| 全部改写成 `A[0]` | 破坏所有并发冲突 | 同步与一致性流量的成本 |
| 用 dummy 数据 | 让分支/数据依赖消失 | 分支预测与延迟界的成本 |
| 指定理想调度 | 手动静态最优分配 | 调度器/负载均衡的损失 |

#### D.4 测量方法论 Checklist

- [ ] 用**墙钟时间**（`omp_get_wtime`、`std::chrono::steady_clock`），不用 `clock()`（那是 CPU 时间，多线程会累加）。
- [ ] **warmup**：先跑几次让 cache、TLB、线程池、频率进入稳态；CPU 频率缩放会把第一次运行变慢。
- [ ] **重复多次取中位数/最小值**，报告分布而非单点。
- [ ] **防止死代码消除**：把结果写进一个 `volatile` 全局或打印校验和。
- [ ] **基线要正确**：串行版本必须是最优串行实现（不能用带锁的版本当基线）。
- [ ] **固定问题规模 vs 固定时间**：明确报告是强扩展还是弱扩展。
- [ ] **关闭 turbo/绑核/控制变量**：报告机器参数，避免跨机器混合比较。
- [ ] **用计数器而非感觉**：`perf stat -e cycles,instructions,cache-misses,LLC-load-misses`；GPU 用 Nsight Compute 的 `sm__throughput` / `dram__throughput` 判断算力受限还是带宽受限。

---

### 速查表 E：同步原语与一致性协议速查表

#### E.1 锁的对比

| 锁 | 获取操作 | 竞争者等待方式 | 每次释放的互连流量 | 公平性 | 备注 |
|---|---|---|---|---|---|
| 简单自旋 TAS | `test&set` | 反复写（RMW） | `O(P)`（每次都发 BusRdX） | 无 | 整体代价 `O(P²)`，最差 |
| TTAS | 先读后 `test&set` | 本地只读自旋，锁定才 RMW | `O(P)`（只有一次成功 RMW，但所有等待者同时探测） | 无 | 需指数退避 |
| Ticket 锁 | `fetch_add` + 读 `now_serving` | 只读自旋 | `O(P)`（读），但常数小 | **FIFO** | 需 `PAUSE` |
| 数组锁（Anderson） | `fetch_add` + 各自槽位 | 各在**独立槽位**自旋 | `O(1)`（只有后继探测） | FIFO | 需每线程一槽 |
| **MCS 队列锁** | 原子交换 + 写自己节点的 `next` | 各在**自己的节点**上自旋 | **`O(1)`** | FIFO | 需每锁一份节点；**可扩展性最好** |
| 无锁 CAS 循环 | `compare_exchange` 重试 | 无等待（乐观） | 争用时 `O(P)` 重试 | 无 | 需处理 ABA、内存回收 |

#### E.2 屏障的对比

| 屏障 | 单次代价 | 是否有感知翻转 | 说明 |
|---|---|---|---|
| 朴素计数器 + flag | 死锁（连续复用） | 无 | 计数器不清零 ⇒ 下一代立即通过，**错误** |
| 两计数器版本 | 2 次自旋 | 无 | 正确但慢（等所有人离开再清 flag） |
| **sense-reversal** | 2 次自旋 | **有** | 用翻转代替清零 ⇒ 结构性更简单 |
| 合并树（combining tree） | `O(log P)` 关键路径 | 有 | 只在点对点网络上有收益；总线上流量仍被串行化 |
| 传播式（dissemination） | `O(log P)` 轮 | 有 | 每轮与 `2^k` 距离的线程交换 |
| 广播/集中式（GPU 上用 atomic） | `O(1)` 硬件 | — | GPU 的 barrier 是硬件实现的 block 级同步 |

#### E.3 一致性协议状态迁移（MESI 核心表）

见 A.7 的状态迁移表。补充要点：

- **读未命中 S 态**：发 `BusRd`，内存/其他 cache 供给数据，自己不写回。
- **写未命中**：I → M 发 `BusRdX`；S → M 发 `BusRdX`（无效化其他副本）；E → M **静默**。
- **M 态被监听**：`BusRd` ⇒ flush 数据并降到 S（或 F/O）；`BusRdX` ⇒ flush 并降到 I。
- **换出（eviction）**：M 态换出必须写回内存（write-back）；O 态换出也需写回。

#### E.4 原子操作与硬件支持

| 原语 | 语义 | 硬件实现 | 用途 |
|---|---|---|---|
| `test&set` | 写 1，返回旧值 | 原子 RMW | 简单锁 |
| `compare&swap (CAS)` | 若等于期望则写入新值 | 原子 RMW | 无锁结构、乐观并发 |
| `fetch&add` | 返回旧值并加 | 原子 RMW | 取号（ticket 锁、工作队列） |
| `load-linked / store-conditional` | LL 标记 + SC 条件写 | 需监控地址 | 无锁，避免 ABA |
| `exchange (xchg)` | 交换 | 隐式 `lock` | 隐含全屏障（x86） |
| `atomicAdd`（CUDA） | 设备侧原子加 | 共享/全局内存均支持 | 直方图、归约 |

---

### 速查表 F：常见性能陷阱速查表

| # | 陷阱 | 症状 | 根因 | 对策 |
|---|---|---|---|---|
| 1 | **数据竞争** | 结果不确定、偶发错误 | 缺少同步/原子操作 | 用 `-fsanitize=thread` 检测；用原子或锁；正确放置 fence |
| 2 | **伪共享** | 加核反而变慢、`BusRdX` 暴涨 | 多个变量共享一条 64 B cache line | `alignas(64)` padding；每线程私有 + 归并 |
| 3 | **负载不均** | 加速比远低于核数、核空闲 | 任务代价差异大 + 静态分配 | 动态分配/chunk、工作窃取、LPT 长任务优先 |
| 4 | **忽略内存带宽** | 到达某个核数后不再提升 | 带宽是共享资源 | 算 `AI`，用 blocking/fusion 减少字节；不要只看 FLOPs |
| 5 | **同步开销淹没收益** | 小问题上并行更慢 | 屏障/锁的固定延迟（µs 级） | 增大粒度、减少同步次数、合并屏障 |
| 6 | **错误基线** | 加速比虚高 | 串行版本不是最优实现 | 优化串行基线；报告绝对时间 |
| 7 | **固定问题规模外推** | 声称"64 核能加速 52×" | Amdahl + 并行度不足 | 先算并行度与 `S`；给出强弱扩展两条曲线 |
| 8 | **抽象距离过大** | 热点"看起来"很慢但不知为何 | 编译器生成代码与源码差异巨大 | 看汇编、做高水位实验、用 PMU 计数器 |
| 9 | **NUMA 首触放置** | 核多后带宽反而下降 | 串行初始化导致内存都在一个 node | 并行 first-touch；`numactl --interleave` |
| 10 | **页步长访问** | 大数组上性能悬崖式下降 | TLB 覆盖不足（4 KB 页 reach 小） | 大页、改变遍历顺序、分块 |
| 11 | **GPU 分支发散** | warp 内部分歧导致串行化 | SIMT 用掩码串行执行各路径 | 重排数据让 warp 走同一路径；用算术代替分支 |
| 12 | **GPU 未合并访存** | 带宽利用率远低于峰值 | 相邻线程访问间隔大 | 让 `threadIdx` 映射到最内层连续维；用 shared memory 转置 |
| 13 | **GPU 跨 block 自旋等待** | 挂死 | block 不保证并发驻留 | 用 kernel 分解代替全局同步 |
| 14 | **共享内存 bank conflict** | shared memory 访问串行化 | 同 bank 不同地址 | padding 打散（如 `[32][33]`） |
| 15 | **忽略 kernel 启动/拷贝开销** | 小任务 GPU 更慢 | 启动 µs 级 + PCIe 拷贝 | 增大批量、用 pinned memory、异步 stream 重叠 |
| 16 | **`clock()` 当墙钟用** | 测得时间随线程数异常 | `clock()` 返回所有线程 CPU 时间之和 | 用 `omp_get_wtime` / `steady_clock` |
| 17 | **忘记 warmup** | 第一次运行特别慢 | cache/TLB 冷、频率尚未 boost | 先跑 warmup 迭代 |
| 18 | **ABA 问题** | 无锁结构偶发数据损坏 | CAS 只比较值，不比较"是否被改过" | 版本号/tagged pointer、DCAS、hazard pointer |
| 19 | **内存序假设错误** | x86 上"能跑"、ARM 上崩 | 依赖 TSO 的额外强度 | 用 `memory_order_acquire/release` 或 fence |
| 20 | **事务内存的伪冲突** | 事务中止率高、扩展性差 | 冲突检测粒度为 cache line；全局版本时钟争用 | 填充隔离元数据、减少事务内工作、切开长事务 |

---

### 附录 B：课程资源清单

#### B.1 官方链接

| 资源 | 链接 |
|---|---|
| 课程主页 | <https://www.cs.cmu.edu/~418/> |
| 日程表 | <https://www.cs.cmu.edu/~418/schedule.html> |
| 作业 | <https://www.cs.cmu.edu/~418/assignments.html> |
| 考试 | <https://www.cs.cmu.edu/~418/exams.html> |
| 项目 | <https://www.cs.cmu.edu/~418/projects.html> |
| 资源 | <https://www.cs.cmu.edu/~418/resources.html> |
| 教职员 | <https://www.cs.cmu.edu/~418/staff.html> |
| 讲义目录 | <https://www.cs.cmu.edu/~418/lectures/> |
| 课程大纲 PDF | <https://www.cs.cmu.edu/~418/syllabus/syllabus.pdf> |
| Assignment 1 说明 | <https://www.cs.cmu.edu/~418/doc/asst1_handout.pdf> |
| CUDA Recitation | <https://www.cs.cmu.edu/~418/doc/CUDA-recitation.pdf> |
| Ed 讨论区（需登录） | <https://edstem.org/us/courses/102588/discussion> |
| Autolab（需登录） | <https://autolab.andrew.cmu.edu/courses/15418-f26> |
| 教师主页 | <https://www.cs.cmu.edu/~bpr/>、<https://www.cs.cmu.edu/~dskarlat/> |

#### B.2 本地已下载材料（本工作目录）

```text
cmu15418_data/
├── *.html                     课程网站全部公开页面（8 个）
├── lectures_data.json         结构化的逐讲资料记录（含公开性状态）
├── syllabus.pdf               课程大纲
├── doc_asst1_handout.pdf      Assignment 1 说明（13 页）
├── doc_CUDA-recitation.pdf    CUDA Recitation（39 页）
├── lectures/                  Fall 2026 公开讲义 PDF（21 份）+ 讲义逐页文本
├── extracted/                 38 份讲义/试卷的逐页文本抽取（含 slide 标记）
├── cs149_supp/                Stanford CS149 Fall 2025 公开讲义文本（18 份，补充材料）
├── supp/exams/                Spring 2022 练习题（4 份）+ 文本抽取
├── past/                      归档讲义下载尝试（实为 CMU 登录页，已记录为未公开）
└── notes/                     26 份逐讲学习笔记（本笔记主体）
```

#### B.3 相关课程与扩展阅读

| 资源 | 说明 |
|---|---|
| Stanford CS149 "Parallel Computing"（Kayvon Fatahalian） | 与 15-418 同源、讲义公开，本笔记用于补全 L14/L17/L24 等未发布讲次 |
| CMU 15-213 "Introduction to Computer Systems" | 先修课；cache、汇编、虚拟内存基础 |
| CMU 18-447 "Introduction to Computer Architecture" | 微架构与流水线细节 |
| Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* | 第 5 章（线程级并行）、附录（互连网络、存储层次） |
| Mattson et al., *Patterns for Parallel Programming* | 分解/分配/编排模式的系统化 |
| Kirk & Hwu, *Programming Massively Parallel Processors* | CUDA 与 GPU 架构的系统教材 |
| Gropp, Lusk, Skjellum, *Using MPI* | MPI 实践 |
| Herlihy & Shavit, *The Art of Multiprocessor Programming* | 同步原语、无锁数据结构、一致性 |
| Williams, Waterman, Patterson, "Roofline" (CACM 2009) | Roofline 模型原始论文 |
| Fog, *Microarchitecture of Intel/AMD CPUs* | 指令延迟/吞吐的权威微基准手册 |

---

### 结语：这门课的"最后一句话"

> **并行编程的本质不是"用更多核"，而是"用有限的硬件资源，以最小的数据移动，完成尽可能多的工作"。**

- 硬件给了你：更多的执行单元（核 / SIMD lane / GPU lane）、多层次的存储、以及一套把并行性拼装起来的一致性机制。
- 软件要做的事：**分解**出足够多的独立工作、**分配**得足够均匀、**编排**得足够少的通信与同步、并**映射**到最合适的执行单元上。
- 判断标准永远只有三把尺子：**Work/Span（并行度）、Amdahl（串行比例）、算术强度与 Roofline（瓶颈类型）**。

祝复习顺利。

{% endraw %}
