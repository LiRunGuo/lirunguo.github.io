---
title: "核心术语表（Glossary）"
collection: course-notes
chapter: true
permalink: /course-notes/stanford-cs149-parallel-computing/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/stanford-cs149-parallel-computing/) · [← l18](/course-notes/stanford-cs149-parallel-computing/l18)

{% raw %}
## 核心术语表（Glossary）

> 按字母顺序排列，涵盖全部 18 讲中定义的关键术语。

- **`__syncthreads()`（块内屏障）**：block 内所有线程必须到达的屏障点，用于隔开 shared memory 的"写"与"读"阶段；只同步本 block。
- **ABA problem（ABA 问题）**：CAS 只比较值而无法区分"值未变"与"值变回原样"导致的错误，典型后果是无锁栈中栈顶地址 A→B→A 变化后 CAS 误成功而丢失节点。
- **Abstraction vs Implementation（抽象 vs 实现）**：程序"算什么"（semantics）与"在并行机上如何算"（scheduling：谁/何时/在哪个 ALU 或 lane 上）的区分；把二者混为一谈是并行编程最常见的困惑来源。
- **Acquire / Release（获取/释放语义）**：成对使用的单向内存序——release 保证其之前的操作在其之后对其他线程可见，acquire 保证其之后的操作能看到 release 之前发布的内容；锁、屏障等同步原语的内部分子。
- **Algorithm / Schedule 分离**：Halide 的核心设计——算法声明输出如何由输入计算（不规定迭代顺序），调度指定循环顺序、分块、向量化与并行化方式，二者独立可调。
- **All-to-All（全交换）**：每个节点把自己的数据分片分别发送给所有其他节点（如 rank 0 的 A0..A3 分别发给 rank 0..3）。
- **AllGather（全收集）**：每个节点把自己的分片广播汇总，最终所有节点都拥有完整数据。
- **AllReduce（全归约）**：所有节点对各自数据归约后得到相同完整结果；等价于 ReduceScatter + AllGather。
- **AMAT（Average Memory Access Time）**：平均访存时间 = Σ(访问频率 × 访问延迟)；多处理器下因一致性通信而增大（AMAT_Multiprocessor > AMAT_Uniprocessor）。
- **Amdahl's Law（Amdahl 定律）**：可并行部分占比 p、串行部分占比 1−p 时，加速比上限 `1 / ((1−p) + p/P)`，P→∞ 时收敛到 `1/(1−p)`——串行部分决定加速比天花板（本讲以三个课堂演示建立直觉，后续课程系统使用）。
- **Amdahl's Law（阿姆达尔定律）**：最大加速比受串行占比限制：设 S 为串行占比，则加速比 ≤ 1/S。
- **Arithmetic intensity / roofline（运算强度 / 屋顶线模型）**：单位字节数据搬运对应的运算量；决定程序是 compute-bound 还是 bandwidth-bound，是评估加速器匹配度的基本工具（贯穿第 9–12 讲）。
- **Arithmetic Intensity（算术强度）**：单位数据搬运对应的计算量；如逐元素向量乘法每元素 12 字节只换 1 次 MUL（强度极低），是带宽杀手（本讲未用该词，但思想实验即其雏形）。
- **Artifactual Communication（人为通信）**：源于系统实现细节（cache line 粒度、容量有限等）的额外通信，可用局部性优化减少。
- **ASIC（Application-Specific Integrated Circuit，专用集成电路）**：为单一用途定制、不可编程的电路；性能/面积与性能/功率可分别比 CPU 核高约 1000 倍与 100 倍（FFT 案例），但设计/验证/流片成本高达数千万至数亿美元。
- **Assignment（分配）**：把任务分配给 worker（线程/实例/向量通道），目标是负载均衡与低通信成本，可静态或动态执行。
- **Asynchronous (nonblocking) execution（异步/非阻塞执行）**：在较早操作完成前就启动较晚操作，重叠访存、通信与计算，避免等待；需要软件+硬件异步指令与同步原语（或硬件乱序执行）。
- **Atomic Operation（原子操作）**：不可被其他线程并发观察到的中间状态的读-改-写操作（如 `atomicAdd`）。
- **Atomicity violation（原子性违例）**：程序员把逻辑上原子的序列错误拆成多个原子块，使其他线程可插入破坏性操作导致的错误。
- **Atomicity（原子性，all or nothing）**：事务提交时所有写一次性生效、中止时所有写如同从未发生。
- **Attention（注意力）**：transformer 核心块——`S = QKᵀ`（N×N 分数）、`P = softmax(S)`（逐行）、`O = PV`；朴素实现需 N² 空间，长序列会内存爆炸。
- **Autoscheduler（自动调度器）**：把调度建模为序列化决策（每个节点的 compute_at 层级 + tile 尺寸），用贪婪/束搜索在调度空间中搜索，并用 MLP 代价模型（几十微秒/次）快速评估候选。
- **Bandwidth-Bound（带宽受限）**：程序执行时间由"需搬运的数据量/带宽"决定而非计算量决定的状态；数据并行方案多次遍历数据，容易带宽受限（"if you can avoid being bandwidth bound"）。
- **Bandwidth-Limited（带宽受限）**：处理器请求数据速率超过内存供给速率时的执行状态；稳态下核心利用率只取决于指令吞吐与内存吞吐之比，与延迟、未完成请求数无关。
- **Bank（DRAM 存储体）**：DRAM 芯片内可独立预充电/激活的存储分区；各 bank 共享数据引脚，但可流水操作（一个 bank 传输时另一个 bank 激活），实现高引脚利用率。
- **Barrier（屏障）**：让指定数量线程都到达后才能继续的同步原语，是表达依赖的保守方式，把计算划分为阶段。
- **Block Descriptor（块描述符）**：发生窃取时运行时为含 spawn 的代码块创建的数据结构，追踪该块已 spawn 与已完成的工作数，用于实现 sync。
- **Blocked Assignment（分块分配）**：实例 k 处理连续块 [k·count, (k+1)·count)；同一时刻各实例访问不连续地址，需昂贵的 gather 指令。
- **Blocking / Tiling（分块）**：把 GEMM 组织成 C 的子块计算，使 A、B 子块在缓存/shared memory 驻留期间被复用 BLOCKSIZE 次，显著提高算术强度；块须装得进存储层级（不是越大越好）。
- **Blocking algorithm（阻塞式算法）**：允许一个线程无限期阻止其他线程完成操作的算法，任何使用锁的算法（无论自旋或让出 CPU）都是阻塞式的。
- **Blocking Receive（阻塞接收）**：recv 在消息数据拷入本地址空间并发出确认后才返回的同步通信。
- **Blocking Send（阻塞发送）**：send 在收到接收方确认、消息已进入接收方地址空间后才返回的同步通信。
- **Bulk Launch（批量启动）**：一次 kernel launch 创建成千上万线程的机制，成本远低于逐个 `pthread_create`（无需每线程分配栈与 OS 控制块）。
- **Burst mode（突发传输模式）**：一条 DRAM 命令批量传输多个连续列的数据，摊销预充电/激活的固定开销。
- **BusRd / BusRdX / BusWB（总线事务）**：MSI 协议的三类总线消息——BusRd 获取共享副本、BusRdX 获取独占副本（写前必须发出，迫使他人失效）、BusWB 把脏行写回内存。
- **Cache Coherence（缓存一致性）**：对每个内存位置存在一个与所有处理器观察一致的假设串行顺序，且每处理器按程序序执行、读返回串行顺序中最后一次写的值；解决"多个私有缓存复制同一数据导致观察不一致"的问题。
- **Cache Hierarchy（缓存层级）**：L1 → L2 → L3 → DRAM 的多级实现；越近容量越小、延迟越低（Kaby Lake 示例：4 / 12 / 38 / ~248 周期）。
- **Cache Line（缓存行）**：cache 存取数据的粒度（如 4 字节/行）；装入一行会顺带预载相邻地址的数据。
- **Cache Miss（缓存缺失）**：目标地址不在 cache 中，必须从更低层级取数；包括 cold miss（首次访问）与 capacity miss（工作集超出容量、旧行被逐出后再访问）。
- **Cache（缓存）**：芯片上的存储，保存内存中一部分值的副本；只影响性能不影响程序输出；命中时访问远快于 DRAM。
- **Capacity Miss（容量缺失）**：因 cache 容量不足、数据在两次访问之间被逐出而导致的 cache miss。
- **Child Stealing（子任务窃取）**：先执行 continuation、把子任务入队的策略（run continuation first），广度优先、需 O(N) 空间存储已 spawn 工作。
- **cilk_spawn**：Cilk 关键字，调用函数但允许调用者与它异步并行继续执行。
- **cilk_sync**：Cilk 关键字，等待当前函数所有已 spawn 的调用完成；含 spawn 的函数末尾有隐式 sync。
- **Cilk（Cilk Plus）**：源自 MIT、现为开源标准（GCC/Intel ICC 支持）的 C++ 语言扩展，提供 fork-join 并行原语。
- **Coherent Execution（一致执行）**：同一段指令序列适用于大量数据元素的程序性质；是 SIMD 高效利用的必要条件（多核并行不需要）。
- **Combiner（合并函数）**：并行 fold 中把各子段结果合并成最终结果的二元函数。
- **Communication Overhead（通信开销）**：并行单元之间传递数据（如 partial sum）所花的时间；通信是限制最大加速比的首要因素（DEMO 1 结论）。
- **Compare-and-Swap（CAS，比较并交换）**：原子指令，若当前值等于期望值则写入新值并返回是否成功；构建无锁数据结构与高级同步的万能积木（`compare_exchange_strong`）。
- **Compare-and-swap（CAS，比较并交换）**：原子指令，若目标当前值等于期望值则写入新值并置成功标志、否则把当前值读回，是无锁算法的基础（x86 的 `lock cmpxchg`）。
- **Composability（可组合性）**：多个同步模块可安全组合成更大原子操作的性质，事务嵌套时最外层定义原子性边界。
- **Compute Bound（算力受限）**：执行时间由机器指令处理能力决定的状态，此时吞吐达到峰值算力。
- **Compute Mode（计算模式）**：NVIDIA Tesla（2007）提供的首个非图形专用 GPU 接口——分配 buffer、上传 kernel 二进制、`launch(myKernel, N)` 以 SPMD 方式运行 N 个实例。
- **Compute-communication overlap（计算-通信重叠）**：让通信（如 AllReduce）与计算（GEMM/权重加载）并行进行；RDU 上 AllReduce 走芯片间专用通路、不消耗 HBM 带宽，32 socket 时仍保持 70%+ 利用率。
- **Conflict detection（冲突检测）**：系统判定两个并发事务是否冲突以及何时判定的策略，分悲观（每次访存检查）与乐观（提交时检查）。
- **Contention manager（竞争管理器）**：冲突发生时仲裁谁暂停、谁中止、何时重试的策略组件，用于避免活锁等无进展局面。
- **Contention（争用）**：大量请求在短时间内涌向同一资源（热点）导致排队、整体操作时间变长的现象。
- **Continuation Stealing（延续窃取）**：先执行子任务、把 continuation 入队的策略（run child first），深度优先、空间有界、无窃取时执行顺序同串行。
- **Continuation（延续）**：cilk_spawn 之后调用者"剩余要执行的代码"，可被放入工作队列供其他线程窃取。
- **Convolutional Layer（卷积层）**：局部连接且同一层所有单元共享同一组参数（weights + bias）的神经网络层；滤波器可视为"模式检测器"，输出幅度为该滤波器对输入局部区域的响应。
- **Critical Section（临界区）**：需要互斥保护的、访问共享资源的代码段，其中的操作应整体"原子"执行。
- **Cross-Instance Operation（跨实例操作）**：gang 内实例间交换数据的标准库原语，如 reduce_add（求和）、reduce_min（取最小）、broadcast（广播某实例的值）、shift/rotate（把值沿实例编号平移/循环移动）。
- **Cross-Program-Instance Operation（跨实例操作）**：ISPC 标准库提供的实例间通信原语，如 `reduce_add`、`reduce_min`、`broadcast`、`rotate`。
- **Crossbar（交叉开关）**：将所有核心两两直连的片上互连结构（如 SUN Niagara 2），面积约等于一个核。
- **CSR（Compressed Sparse Row，压缩稀疏行）**：稀疏矩阵的存储格式——`values`（非零值）、`cols`（列号）、`row_starts`（每行起始下标）三个数组。
- **CUDA Thread（CUDA 线程）**：逻辑控制流单元，抽象上与 pthread 类似，但实现完全不同（硬件线程、无 OS 调度）；通过 `threadIdx` 等内置变量区分身份。
- **CUDA（Compute Unified Device Architecture）**：NVIDIA 于 2007 年随 Tesla 架构推出的"C 风格"GPU 编程语言与运行时，抽象贴近现代 GPU 硬件能力，设计目标为低抽象距离（low abstraction distance）。
- **cuDNN**：NVIDIA 提供的高性能 DNN 层库（卷积等关键层有多种算法可选）；其 backend 还能现场编译生成融合实现。
- **CUTLASS**：NVIDIA 的 CUDA 模板库，提供 shared-memory GEMM、warp 级 GEMM、块加载迭代器、张量归约等原语，用于编写自定义高性能 DNN 层。
- **CUTLASS / Cute-DSL**：NVIDIA 的 GEMM 模板库（CUTLASS）及其 Python 化表达（Cute-DSL），以 tiled tensor 与布局抽象实现 tensor core 高性能编程。
- **Data parallel patterns（数据并行模式）**：可组合的计算原语——MM、Map、Zip、Reduce、Gather、Scatter 等；程序=模式组合，编译器负责 Tiling/Parallelization/Metapipelining/Place&Route/Codegen。
- **Data Parallelism（DP，数据并行）**：多份模型副本并行处理不同 batch；通信模式为 ReduceScatter+AllGather 或 AllReduce。
- **Data Parallelism（数据并行）**：对大量数据元素执行同一序列操作的并行方式，典型表达为 `foreach`、`#pragma omp parallel for`、`map()`，强调"对每个元素独立地做同一件事"。
- **Data Race（数据竞争）**：多个线程无同步地并发访问同一内存位置且至少一个为写操作，导致结果不确定。
- **Data versioning policy（数据版本策略）**：TM 系统管理未提交（新）版本与已提交（旧）版本数据的方式，分 eager（undo-log）与 lazy（write-buffer）两种。
- **Data-Level Parallelism（DLP，数据级并行）**：同一序列的指令同时作用于大量不同数据；SIMD 利用的就是 DLP。
- **Data-Parallel Model（数据并行模型）**：把计算组织成对元素序列的操作（如对序列所有元素执行同一函数）；典型表达为 NumPy 的 `C = A + B`、`map()` 等。
- **Data-Value Invariant（数据值不变量 / 写串行化）**：一个 epoch 开始时某地址的值等于上一个写 epoch 结束时的值，保证写结果被串行化传播，是 coherence 的另一条不变量。
- **Dataflow execution（数据流执行）**：数据到达即计算、数据在计算单元间直接流动的执行方式，无指令流、无集中控制；AI 模型本身可看作数据流图（GEMM + Pool + SoftMax 等）。
- **DDR（Double Data Rate）**：每时钟传输两次数据的内存标准；如 DDR4 2400 = 64-bit × 1.2 GHz × 2 = 19.2 GB/s/通道，双通道 38.4 GB/s。
- **Deadlock（死锁）**：多线程相互等待对方释放资源/完成操作而永远无法继续的阻塞状态（如同步收发下全体先 send 后 recv）。
- **Declarative vs. Imperative abstraction（声明式 vs. 命令式抽象）**：前者只声明"做什么"（如 `atomic { }`），后者规定"怎么做"（如"拿锁-执行-放锁"）。
- **Decomposition / Assignment / Communication & Synchronization（分解 / 分配 / 通信与同步）**：并行思维三步骤——把问题拆成可安全并行的工作块、把工作分配给处理器、管理通信与同步使其不成为加速比瓶颈。
- **Decomposition（分解）**：把问题拆成可并行执行的子任务，关键挑战是识别依赖。
- **Dependency（依赖）**：一个计算必须等另一个计算完成后才能执行的关系；识别依赖是分解阶段的核心挑战。
- **Dequeue（双端队列）**：工作窃取中每个 worker 的工作队列实现，本地线程从底部 push/pop，窃取线程从顶部偷取。
- **Device（设备）**：执行 kernel 的 GPU 侧；device 代码即 kernel 与 `__device__` 辅助函数。
- **DGX SuperPOD**：NVIDIA 的模块化 AI 集群架构；1K GPU 集群 = 140 个 DGX A100 节点（1120 GPU）+ Lustre 存储 + Mellanox HDR 200 Gb/s InfiniBand 全胖树网络。
- **dim3 / threadIdx / blockIdx / blockDim**：CUDA 内置的多维索引机制——`threadIdx` 是线程在块内的坐标，`blockIdx` 是块在网格中的坐标，`blockDim` 是块维度；全局坐标由 `blockIdx * blockDim + threadIdx` 计算，最多三维。
- **DIMM（Dual Inline Memory Module）**：多个 DRAM 芯片组成的模块（如 8 芯片 × 8 bit = 64-bit 接口），对控制器表现为更高容量、更宽接口的内存，最小传输粒度 64 bit。
- **Directory-Based Coherence（目录式一致性）**：用集中目录记录每行在各缓存中的状态，通过点对点"按需告知"消息维护一致性（如 Intel Core i7 用 inclusive 的 L3 充当目录），避免广播以提升可扩展性。
- **Dirty Bit（脏位）**：缓存行中标记"该行已被修改、内存副本过期"的状态位；在写回一致性协议中，脏状态表示该缓存是行的唯一有效持有者（M 状态）。
- **Distributed Address Space（分布式地址空间）**：host 内存与 device global 内存是两个不同地址空间，数据搬运靠 `cudaMalloc`/`cudaMemcpy` 等原语（`cudaMemcpyHostToDevice` 等方向参数），host 不能直接解引用 device 指针。
- **Divergent Execution（分支发散）**：warp 内线程走了不同指令路径（如 `if` 两侧），硬件只能掩码执行、逐路径串行，性能受损；是 SIMT 执行的主要性能陷阱。
- **Divergent Execution（分歧执行）**：指令流缺乏一致性（如 SIMD 内 if/else 两路分支）；硬件顺序执行各分支并用 mask 掩蔽无用 ALU 输出，最坏只有 1/8（CPU）/1/32（GPU）峰值。
- **Domain-specific accelerator（领域专用加速器）**：在有限领域内可编程（通常用 DSL，如 DNN），能效约 20×（如 Google TPU），是"完全可编程"与"完全固定"之间的折中。
- **Domain-Specific Programming System**：围绕 DSL 构建的系统，通过提高抽象层次让程序员快速写出高性能、可移植的程序，并用领域知识自动选择算法与并行化策略。
- **Double Buffering（双缓冲）**：重叠数据搬运与计算的技术——同时持有"正在处理的数据"与"正在传输的数据"两份片上缓冲，上一块计算时下一块已在加载。
- **Double compare-and-swap（DCAS / doubleword CAS）**：同时比较交换两个相邻值（如栈顶与弹出计数）的原子操作，x86 用 `cmpxchg8b`/`cmpxchg16b` 实现，是 ABA 的计数器解法。
- **DRAM row buffer（DRAM 行缓冲）**：DRAM 阵列中存放当前激活行（约 2 Kbit）的缓冲；访问已激活行（row hit）可跳过 precharge 与 row activation，延迟大幅降低。
- **DRF（Data-Race-Free）**：无数据竞争程序；同步化的 DRF 程序在非 SC 系统上也得到 SC 结果（"SC for DRF"），这是语言内存模型承诺的基础。
- **DSL（Domain-Specific Language）**：针对特定领域、表达能力受限的编程语言，通常高层、声明式且确定性，用受限表达力换取系统级的自动优化能力。
- **DSL（Domain-Specific Language，领域专用语言）**：面向特定领域（如 DNN）的编程语言/抽象，限制可编程范围以换取易用性与性能；GPU AI kernel 场景的 DSL 包括 Mosaic GPU、Cute-DSL 等。
- **DSP（Digital Signal Processor，数字信号处理器）**：可编程但指令流控制更简单的处理器，用 SIMD/VLIW 等复杂指令摊销控制成本；例：Qualcomm Hexagon DSP 每周期执行 29 个 RISC 级操作。
- **Dual channel（双通道内存）**：两个内存控制器/通道并行工作，等效加宽总线、提升吞吐。
- **Dynamic Assignment（动态分配）**：运行时动态决定分配以保证负载均衡，用于任务代价或数量不可预测的场景（如共享计数器、work queue）。
- **Eager versioning（急切版本管理，undo-log based）**：写内存立即生效并记录撤销日志，commit 快、abort 慢、有中途崩溃容错问题。
- **Efficiency（效率）**：加速比除以处理器数，`efficiency = speedup / P = T(1) / (P·T(P))`，衡量硬件被有效利用的程度。
- **Embedded DSL（嵌入式 DSL）**：嵌入宿主语言（如 CUDA/C++）的库式 DSL，如 ThunderKittens——用模板类型封装布局、异步原语与协调模式，程序员写"布局声明 + producer/consumer 回调"。
- **Energy efficiency（能效）**：单位能量完成的运算量；由 Power = Ops/s × Joules/Op 可知，提高能效即提高每焦耳运算数或降低每运算能耗。
- **Execution Context（执行上下文）**：处理器为正在执行的指令流保存的状态（寄存器值 + PC），硬件多线程下每个硬件线程各有一份。
- **Expert Parallelism（EP，专家并行）**：Mixture-of-Experts 中把不同 expert 放在不同设备；通信模式为 All-to-All。
- **Explicit SIMD（显式 SIMD）**：向量化发生在编译期，二进制中可见向量指令（vloadps/vmulps 等）；来源包括 intrinsics、并行语言语义、编译器自动向量化。
- **Failure atomicity（失败原子性）**：异常（除程序员显式管理的）导致事务整体中止并撤销内存更新，失败恢复即 abort + restart。
- **False conflict（假冲突）**：检测粒度过粗（如对象级）时，两个访问不同字段的事务被误判冲突而损失并发。
- **False Sharing（伪共享）**：两个处理器写不同地址但这些地址落在同一条缓存行，导致缓存行乒乓的纯人为通信；与程序语义无关，修复方法是 padding/对齐使数据独占缓存行。
- **FAST ≠ EFFICIENT（快 ≠ 高效）**：程序跑得快不代表用好了硬件；判断标准是是否充分利用了机器提供的能力（如 10 处理器上 2x speedup 的效率只有 20%）。
- **Fat-tree（胖树拓扑）**：数据中心常用网络拓扑，逐层收敛带宽充足，支持自适应路由与网内计算卸载（如 SharpV2）。
- **Filter（过滤）**：删除序列中不满足谓词的元素，输出为输入的子序列。
- **Fine-grained locking（细粒度锁）**：把一把全局数据结构锁拆成多个小锁（如每节点一把）以提高操作并行度的同步策略。
- **FlashAttention / Fused Attention（融合注意力）**：把 softmax 分块计算（增量维护 running max/sum-of-exp）并融合 QKᵀ 与 PV 两个矩阵乘，从不物化 N² 矩阵、O 块常驻缓存；代价是每并入新块需重缩放旧累加值（额外计算）。
- **Fold / Reduce（折叠 / 归约）**：用二元操作 `f :: (b,a) -> b` 把序列折叠成一个值（串行 fold 需初始值）；并行 fold 还需 combiner 函数 `comb :: (b,b) -> b`，若 f 本身满足结合律则无需 combiner，初始值必须是单位元。
- **foreach**：ISPC 关键语言构造，声明循环迭代可并行，由系统负责把迭代分配给 gang 内的 program instance。
- **foreach（ISPC 循环结构）**：声明循环迭代互相独立、由整个 gang 完成；迭代如何分配给实例由 ISPC 实现决定，让程序员几乎像写串行程序一样表达数据并行。
- **Fork-Join Parallelism（分叉-汇合并行）**：用"创建新控制流（fork）→ 等待其完成（join）"表达分治算法中独立工作的并行模式。
- **FPGA（Field Programmable Gate Array，现场可编程门阵列）**：逻辑块 + 可编程互连构成的芯片，介于 ASIC 与处理器之间；用 LUT（查找表）实现逻辑，用硬件描述语言（如 Verilog）编程；现代 FPGA 含硬化的 SRAM 块、DSP 块与 CPU（ARM/RISC-V）。
- **FR-FCFS（First-Ready First-Come-First-Serve）**：常见内存调度策略——优先服务当前开放行的请求（最大化行局部性），其余按 FIFO；控制器还会合并小请求为大连续请求。
- **Gang（线程组）**：一次 ISPC 调用 spawn 的全部程序实例的集合；由 SIMD 指令实现（实例数 = SIMD 宽度或其倍数），运行在一个核的一个线程内。
- **Gang（组）**：一次调用 ISPC 函数时同时执行的一组 program instance 的总称。
- **Gather（收集指令）**：一条 SIMD 指令从多个不连续内存地址取数（如 vgatherdps）；比 packed load 复杂且昂贵。
- **Gather（聚集）**：`output[i] = input[index[i]]`——按索引序列取值，可并行且无冲突；CPU 有 AVX2 gather 指令，GPU 硬件支持但比连续加载贵。
- **Gauss-Seidel（高斯-赛德尔迭代）**：逐点用邻居当前值更新网格的迭代求解算法，本课程以 2D grid solver 为其并行化案例。
- **GEMM（General Matrix Multiply）**：稠密矩阵乘 `C += A×B`，是全连接层、卷积层、transformer 注意力块的共同内核（"the kernel for…"）。
- **Ghost Cell（幽灵单元）**：消息传递模型中从远端线程地址空间复制到本地数组边缘、归他人所有的数据副本（如网格边界行）。
- **Global Memory（全局内存）**：device 的 DRAM 主存，所有线程（所有 block）可读写，容量大但延迟高、带宽宝贵。
- **GPGPU（General-Purpose computation on GPU）**：2002-2003 年把 GPU 当作数据并行机器使用的"hack"（如把 512×512 图像映射为数组、用 fragment shader 逐像素计算），为 CUDA 的诞生铺路。
- **Greedy Join Scheduling（贪心汇合调度）**：线程在 sync/空闲时立即尝试窃取而非空等，只有系统中无工作可偷才空闲的调度策略。
- **Grid（网格）**：一次 kernel launch 启动的全部线程块集合；块之间被假设无依赖、可按任意顺序调度到任意数量核上。
- **GroupByKey（按键分组）**：`Seq (key, T) -> Seq (key, Seq T)`，把相同 key 的元素聚合成"序列的序列"。
- **Halide**：嵌入 C++ 的图像处理 DSL，用 Func/无副作用表达式声明式描述"做什么"（算法），用调度原语描述"怎么做"（调度），由编译器生成平台特定代码。
- **Hand-over-hand locking（手递手锁）**：遍历链表时先锁下一个节点再释放上一个节点的细粒度锁模式，所有线程按一致方向获取锁从而无死锁。
- **Happens-Before（先于关系）**：事件之间的必须先后关系；若某个期望输出导致 happens-before 图出现环（事件必须先于自身），则该输出不可能。
- **Hardware accelerator（硬件加速器）**：为特定计算任务（如 DNN 推理、FFT、视频编码）定制的专用硬件，通过消除通用处理器的指令流开销与减少数据搬运获得更高能效（perf/watt）。
- **Hardware Multi-Threading（硬件多线程）**：在同一个核上交错执行多个线程以隐藏停顿——当前线程无法推进就执行另一线程的指令。
- **Hardware transactional memory（HTM，硬件事务内存）**：版本管理放缓存（R/W 位记录读集/写集）、冲突检测融入 cache coherence 协议的事务内存。
- **Hazard pointer（危险指针）**：每线程声明"我正在访问、不可释放"的指针，被弹出节点进入 retire 列表、确认无任何危险指针指向后才真正释放，用于解决无锁内存回收。
- **HBM（High-Bandwidth Memory，高带宽内存）**：3D 堆叠 DRAM 芯片、用硅通孔（TSV）连接、经硅中介层与处理器互连的高带宽内存，接口 1024-bit/stack；优势为更多带宽、高能效、小尺寸（H100：6×HBM3=6144-bit、3.2 TB/s、80 GB）。
- **Hierarchical Blocking（层次化分块）**：按 L2 → L1 → 寄存器多级分块，逐级匹配内存层级以最大化各级复用。
- **High Watermark（性能上限水印）**：通过程序修改实验（如全部访问 A[0]、删除锁、删除数学运算）建立的"最好能做到多快"的上界，用于判断性能受何限制。
- **Hillis-Steele Scan（朴素并行 scan）**：每步跨距翻倍的并行 scan，Work = O(N log N)、Span = O(log N)——Work 高于串行算法，但 SIMD 通道利用率高，适合 warp 内 32 元素扫描。
- **Host（主机）**：运行普通 C/C++ 串行代码的 CPU 侧；host 代码负责分配内存、拷贝数据、启动 kernel。
- **Hot Spot（热点）**：被过多并发请求集中的共享资源。
- **im2col / Explicit GEMM（显式 GEMM）**：把卷积展开为矩阵乘——为每个输出位置物化一行"输入窗口拉平"向量（带 0-padding），再与滤波器矩阵做 GEMM；代价是存储开销 O(N) 且 DRAM 流量放大 R×S 倍。
- **Implicit GEMM（隐式 GEMM）**：不物化完整卷积矩阵，只在 GPU 片上 shared memory 里按块构造子矩阵并用调优子块 GEMM（如 CUTLASS）计算——无额外片外存储、不增加 DRAM 流量。
- **Implicit SIMD（隐式 SIMD）**：编译器生成标量指令，但硬件总是同时运行 N 份程序实例，并在运行时把相同指令合并到 SIMD ALU（GPU 模式，宽度 8–32）。
- **Inherent Communication（固有通信）**：并行算法中必须发生的通信，是算法根本属性（如发送 ghost rows），可用好的 assignment 减少。
- **Instruction Pipeline（指令流水线）**：指令分 IF/D/EX/WB 等阶段重叠执行；单条指令延迟 4 周期但吞吐 1 条/周期；"每时钟一条指令"指的是吞吐而非延迟。
- **Instruction Stream（指令流）**：程序编译后得到的一串处理器指令；处理器逐条取指、译码、执行。
- **Instruction-Level Parallelism（ILP，指令级并行）**：单条指令流中互不依赖的指令可并行执行；超标量处理器在硬件上自动发现并利用它。
- **Intel RTM（Restricted Transactional Memory）**：Intel Haswell 的 HTM 指令集（xbegin/xend/xabort），在 L1 跟踪读写集、不保证进展、abort 需跳转 fallback 地址。
- **Interleaved Assignment（交错分配）**：实例 k 处理元素 k, k+8, …；同一时刻各实例访问连续地址，可用一条 packed vector load 实现。
- **Interleaved Multi-Threading（交错多线程）**：每个时钟从多个线程中选一个执行其一条指令（时间上交错）。
- **Interposer（硅中介层）**：承载 DRAM 堆与处理器的硅基板，作为高带宽互连介质。
- **Invalidation（失效）**：一致性协议中通知其他缓存"某行即将/已被独占修改，你的副本作废"的机制；失效后其他处理器再读会触发缓存缺失。
- **Isolation（隔离性）**：事务提交前，其他处理器观察不到该事务的任何写。
- **ISPC Task（ISPC 任务）**：用于实现多核并行的第二种抽象；task 函数 + `launch[count]`（启动若干任务实例，各得 taskIndex/taskCount）+ `sync`（等待全部完成）。
- **ISPC（Intel SPMD Program Compiler）**：Intel 的 SPMD 编译器，把一份"看起来像标量"的代码编译成含 SIMD 指令的目标文件，供 C/C++ 链接调用。
- **Kernel fusion（kernel 融合）**：把多个计算步骤融合为一个 kernel，中间结果留在片上，提高数据局部性、消除启动与同步开销；RDU 把一个 Llama 3.1 8B decoder 融合成一次 kernel 调用（每 token 3 次 vs GPU ~800 次）。
- **Kernel launch / synchronization overhead（kernel 启动与同步开销）**：每次启动 kernel 与 kernel 间同步的固定代价；融合与持久 kernel 可将其消除。
- **Kernel loop / persistent kernel（kernel 循环 / 持久 kernel）**：把整个 decoder（甚至多个 decoder）的迭代放入一个 kernel 执行，消除每次迭代的启动开销（RDU 每 token 3 次调用 vs GPU ~800 次）。
- **KernelBench**：含数百个 PyTorch kernel 的基准测试集，用于评测 LLM 智能体自动生成快速且正确 CUDA kernel 的能力。
- **Kernel（内核）**：用 `__global__` 修饰、在 GPU（device）上以 SPMD 方式执行的函数；一次 kernel launch（`kernel<<<grid, block>>>(args)`）批量启动大量 CUDA 线程执行同一份代码。
- **Latency Hiding（延迟隐藏）**：用其他线程的工作填满访存停顿；不改变延迟本身，只消除利用率损失；算术/访存比越高，需要的线程越少（3 算术+12 周期 load 需 5 线程，6 算术只需 3 线程）。
- **Latency（延迟）**：完成一次操作所需的时间（如从 SF 开车到 Stanford 0.5 小时、一次访存 ~248 周期）。
- **launch / sync（启动 / 同步）**：ISPC 任务并行的两个语句——launch 创建并启动任务实例，sync 等待所有任务结束（相当于多线程的 spawn + join）。
- **Layer Fusion（层融合）**：把相邻层（如 Conv → Scale/Bias → MaxPool）融合进同一 kernel，中间结果不落 DRAM——如 scale/bias 在 conv 每算出一个元素后立即执行、max pool 在每算完 2×2 区域后立即取最大值，DRAM 流量减少约一个数量级。
- **Lazy versioning（懒惰版本管理，write-buffer based）**：写先进入写缓冲、提交时才更新内存，abort 快、无容错问题、commit 慢。
- **Livelock（活锁）**：系统在执行大量操作但没有任何线程取得有意义进展的状态，典型场景是操作不断 abort 并重试。
- **LLM Agent Optimization Loop（LLM 智能体优化循环）**：LLM 生成 kernel → 执行/profile（正确性、耗时、SM/DRAM 利用率、L2 命中率）→ 反思瓶颈 → 修改代码的迭代闭环（"trial and error via reflection"）。
- **Load Imbalance（负载不均）**：工作未平均分配导致部分处理器闲置、部分仍在忙；并行完成时间由最慢的处理器决定（DEMO 2 结论）。
- **Lock-free stack（无锁栈）**：用 CAS 循环实现 push/pop 的栈，核心思想是"只要没有其他线程改过栈顶，本线程的修改就可进行"。
- **Lock-free（无锁）**：非阻塞算法中保证至少有一个线程能推进（系统级进展）的性质，不保证单个线程不被饿死。
- **Lock（锁）**：提供互斥的同步原语，保证临界区同时只有一个线程进入。
- **Loop Fusion（循环融合）**：把多个遍历同一数组的循环合并为一个，提高算术强度、减少重复数据加载。
- **Loop Tiling/Blocking（循环分块）**：重排计算顺序使小邻域数据在 cache 中存活到被复用、减少 capacity miss 的优化。
- **Low Precision（低精度）**：用 16-bit/8-bit（正向 4-bit、极端 1-bit）表示 DNN 权重与激活，降低存储与带宽需求、配合张量核提速。
- **LUT（Lookup Table，查找表）**：FPGA 中实现组合逻辑的可编程真值表；如 LUT6 是 6 输入 1 输出、可视为 64 元素表，40 输入 AND 可由 8 个 LUT6 级联实现（延迟 3）。
- **Mapping（映射）**：把线程/worker 映射到硬件执行单元，可由操作系统、编译器或硬件完成。
- **Map（映射）**：把无副作用的一元函数 `f :: a -> b` 应用到序列每个元素、产生等长输出序列的高阶函数（Haskell `map`、C++ `std::transform`、JAX `vmap`）；因元素互不影响可任意并行。
- **mbarrier（内存屏障）**：CUDA 中用于异步数据搬运完成通知与同步的轻量原语（arrive/wait 语义）。
- **Memory Access Latency（访存延迟）**：内存系统把数据提供给处理器所需的时间（如 100 周期 / 100 nsec）。
- **Memory Bandwidth（内存带宽）**：内存系统向处理器提供数据的速率（如 20 GB/s）；与延迟正交——加带宽不降延迟。
- **Memory Consistency（内存一致性模型）**：定义不同地址上读写操作对其他处理器可见的顺序（硬件与编译器允许的重排范围），是硬件/编译器与应用软件之间的契约。
- **Memory controller（内存控制器）**：连接 LLC 与 DRAM 的访存请求调度器；负责物理地址到 bank/row/column 的映射，并在吞吐/延迟/能耗三个冲突目标间调度几十至几百个未完成请求。
- **Memory fence / memory ordering（内存栅栏 / 内存顺序）**：在弱一致性硬件上保证访存顺序可见性的机制，无锁代码与单读单写队列都依赖它（或 C++11 `atomic<>` 的顺序语义）。
- **Memory Fence（内存屏障）**：阻止重排的指令——fence 之前的所有内存操作完成后，之后的操作才能开始；昂贵但能恢复任意排序保证（如 x86 的 lfence/sfence/mfence）。
- **Memory Hierarchy（内存层次结构）**：寄存器、各级 cache、本地内存、远端内存按"低延迟高带宽小容量 → 高延迟低带宽大容量"排列的存储体系。
- **Memory transaction（内存事务）**：一段原子且隔离的内存访问序列，满足原子性、隔离性、可串行化三条语义。
- **MESI（协议）**：MSI 增加 E（Exclusive Clean）状态——行未修改但只有本缓存持有，E→M 升级无需总线事务，消除"无共享也要付两次事务"的低效（MESI，不是 Messi！）。
- **Message Passing（消息传递模型）**：每个线程在私有地址空间中运行、仅通过 send/receive 消息交换数据的编程模型，是集群与超算的模型。
- **Metapipelining（元流水线）**：层级化粗粒度流水线（"流水线的流水线"），把嵌套循环转成流式流水、重叠多个循环迭代、级间双缓冲；可改变访问模式（如转置），在 fusion 失效时仍有效。
- **Micro-batch（微批）**：把 mini-batch 进一步拆成的小批样本，用于细粒度流水并行：forward/backward 在 micro-batch 间错位流水，消除 pipeline bubble。
- **MMA / FMA / DP4（矩阵乘加 / 融合乘加 / 四元点积指令）**：复合指令类型，用一条复杂指令摊销指令流处理开销；半精度场景下可编程性开销分别约 27% / 2000% / 500%。
- **MSI（协议）**：基于失效的写回一致性协议，缓存行三态 Invalid/Shared/Modified；只有 M 状态可本地静默写，写非独占行必须先广播 BusRdX。
- **Multi-Core Processor（多核处理器）**：把晶体管用于增加核数而非让单条指令流更快；每个核独立取指/译码、执行完全不同的指令流。
- **Mutual exclusion / Hold and wait / No preemption / Circular wait（互斥 / 持有并等待 / 不可抢占 / 循环等待）**：死锁成立的四个必要条件，缺一即不可能死锁。
- **Mutual Exclusion（互斥）**：保证同一时刻只有一个线程进入临界区访问共享资源的机制，通常用 lock 实现。
- **NKI（AWS Neuron Kernel Interface）**：AWS Trainium 加速器上编写自定义 kernel 的底层库（Assignment 4 使用）。
- **Non-Blocking（Asynchronous）Send/Receive（非阻塞异步收发）**：send/recv 立即返回句柄、由 checksend/checkrecv 查询完成状态的通信，允许通信与计算重叠。
- **NUMA（Non-Uniform Memory Access，非均匀内存访问）**：不同核心访问同一内存地址延迟/带宽不同的体系结构特性（如多插槽系统、Intel ring interconnect、Sun Niagara crossbar）。
- **Numerical formats：BF16 / BF8（数值格式）**：低精度浮点格式降低每运算能耗与面积；BF16（1+8+7）与 FP32 范围相同但精度低；BF8 有 E4M3（最大 448）与 E5M2（最大 57344）两种。
- **Optimistic conflict detection（乐观冲突检测）**：只在提交时检测冲突且提交方优先，有前进保证但发现晚、有公平性问题。
- **Orchestration（编排）**：组织通信结构、加入同步、组织数据布局、调度任务，目标是降低通信/同步成本并保持局部性。
- **Out-of-Order Execution（乱序执行）**：不按程序顺序执行指令、但保证输出与顺序执行一致的调度方式；"respect program order"即结果必须与按序执行相同。
- **Overhead（开销）**：为获得并行性/均衡/低通信而付出的额外工作（调度、同步、簿记等），串行程序中不存在。
- **Parallel Computer（并行计算机）**：一组相互协作（cooperate）以快速求解问题的处理单元（processing elements）的集合——既要"多"，更要"协作"。
- **Parallel Slack（并行松弛）**：独立工作总量与机器并行执行能力的比值，实践中约 8 为宜。
- **Parallelism + hardware specialization（并行 + 硬件特化）**：课程结论——未来高性能硬件的来源是增加并行度与专用硬件（GPU、TPU、媒体 ASIC 等）的结合。
- **PCU / PMU / AGCU**：Pattern Compute Unit（模式计算单元，systolic+SIMD）、Pattern Memory Unit（模式存储单元，0.5 MB 高带宽）、Address Generator and Coalescing Unit（地址生成与合并单元，片外内存/IO 门户）。
- **Performance Counter（性能计数器）**：CPU 中统计指令数、时钟、cache miss、内存字节数等事件的寄存器（可用 PCM/VTune/PAPI 读取）。
- **Persistent Threads（常驻线程编程风格）**：程序员启动恰好填满 GPU 的 block 数，在 while 循环里用 `atomicInc` 自行领取工作的风格——依赖底层硬件参数（如 V100 的 `BLOCKS_PER_CHIP`），绕过硬件的线程块调度器。
- **Pessimistic conflict detection（悲观冲突检测）**：每次 load/store 后立即检查冲突，由 contention manager 决定暂停或中止，发现早但无前进保证。
- **PE（Processing Element，处理单元）**：脉动阵列中的基本计算单元，内部通常为"乘法器 + 加法器 + 局部寄存器"，每周期完成一次乘加并把数据传给邻居。
- **Pipeline bubble（流水气泡）**：流水并行中设备处于空闲的时间段（如流水线填充/排空期）；micro-batch 流水与合理 schedule 可显著缩小。
- **Pipeline Parallelism（PP，流水并行）**：按层切分模型，不同层在不同设备上，数据流水流过；通信模式为 Send-Receive。
- **Pipelining（流水线）**：把处理拆成多级、各级重叠执行以隐藏延迟/增加通信计算重叠的技术（硬件层面亦指指令/内存流水）。
- **Pooling（池化）**：在局部区域（如 2×2）取最大值（max pool）或平均值（avg pool）的下采样层，减小空间分辨率（如 W×H → W/2×H/2）。
- **Power Wall（功耗墙）**：动态功耗 ∝ 容性负载 × 电压² × 频率，静态功耗来自漏电；功耗=发热，限制了时钟频率无限提升，是单核性能停止增长的原因之一。
- **Prefetching（预取）**：硬件动态分析访存模式、预测未来地址并提前装入 cache 以减少停顿；预测错误会浪费带宽、污染 cache。
- **Prefix Sum（前缀和）**：当 ⊕ 为加法时的 scan，即 `[a0, a0+a1, a0+a1+a2, ...]`。
- **Private Memory（私有内存）**：kernel 内每线程独占的地址空间（通常即寄存器），其他线程不可见。
- **Producer-consumer pipeline（生产者-消费者流水线）**：把 kernel 组织为"生产者异步加载 tile → 消费者（tensor core）计算 → 写回"的流水，级间用多级缓冲与 mbarrier 协调，使计算与访存重叠。
- **Program Instance（程序实例）**：一次 ISPC 函数调用中并行执行的函数副本；每个实例有自己的一份局部（varying）变量。
- **Program Order（程序序）**：单个线程程序中 load/store 的原始执行顺序；SC 要求每线程操作按程序序提交。
- **programCount**：gang 中同时执行的程序实例个数（uniform 值）。
- **programIndex**：当前程序实例在 gang 中的编号（varying 值，每个实例不同）。
- **Programmability overhead（可编程性开销）**：现代处理器执行一条指令所需取指/译码/冒险检查/寄存器堆访问等环节消耗的能耗与时间；H.264 能耗分解显示功能单元占比很小，开销主要来自控制与数据通路。
- **PSO（Partial Store Order）**：进一步放松 W→W，允许写缓冲内重排写（如 `A=1; flag=1` 可能被其他处理器看到反序）。
- **R bit / W bit（读位 / 写位）**：cache line 上标记该行属于当前事务读集/写集的元数据位，commit/abort 时批量清零。
- **RDU（Reconfigurable Dataflow Unit）**：SambaNova SN40L 的数据流处理单元，含 1040 个 PCU+PMU、638 TFLOPS (bf16)、520 MB 片上 SRAM；PCU 做 systolic+SIMD 计算，PMU 做高带宽寻址存储，AGCU 访问片外内存。
- **Read set / Write set（读集 / 写集）**：事务执行期间读过的地址集合与写过的地址集合，冲突检测基于二者与别的事务集合的交集。
- **Read-write conflict（读写冲突）**：事务 A 读了地址 X，而事务 B 未提交地写了 X。
- **Reconfigurable Dataflow Architecture（RDA，可重构数据流架构）**：把计算按空间布局在芯片上的可配置架构（如 Plasticine、SambaNova SN40L），无指令 ⇒ 无取指/译码开销，极端异步。
- **Red-Black Coloring（红黑着色）**：把网格按棋盘格染色、先并行更新所有红格再更新所有黑格的 Gauss-Seidel 重排方法，使同色格互不依赖。
- **ReduceScatter（归约-散播）**：先做归约，再把结果按 rank 分片分发，每个节点得到最终结果的一部分。
- **Register checkpoint（寄存器检查点）**：事务 begin 时保存的寄存器状态，abort 时据此恢复执行上下文。
- **Relaxed Consistency（放松一致性）**：允许违反部分内存操作排序约束以换取性能（隐藏内存延迟）的模型族，如 TSO、PC、PSO、WO、RC，区别在于放弃哪几条排序。
- **ReLU（Rectified Linear Unit）**：`f(x) = max(0, x)` 的逐元素激活函数，常用于卷积后引入非线性。
- **Ring Interconnect（环形互连）**：Intel 处理器（Sandy Bridge 起）的片上互连结构，四环分别承载 request/snoop/ack/data 消息。
- **Roofline Model（屋顶模型）**：以算术强度为横轴、可达吞吐为纵轴的性能分析模型，水平区为计算受限、对角区为带宽受限。
- **Row hit / row miss（行命中 / 行缺失）**：访问是否落在当前已激活行；row miss 需 Precharge + Activate + Column Access 全流程（各约 10 ns），row hit 仅需列选+传输。
- **SC for DRF（无数据竞争即顺序一致性）**：C11/C++11 与 Java 5 的承诺——DRF 程序获得顺序一致性，编译器负责插入必要同步；有 race 则无任何保证。
- **Scale up / Scale out（纵向扩展 / 横向扩展）**：单节点做强（更快互连、更多内存）vs 标准节点堆集群（网络互连）；AI 数据中心两者并用。
- **Scan（扫描）**：给定结合律操作 ⊕，输出每个位置"从开头到该位置"的累计结果；inclusive scan 含当前元素，exclusive scan 不含（首元素为单位元 I）。
- **Scatter（散开）**：`output[index[i]] = input[i]`——按索引序列写入目标位置，可能冲突（多个 i 写同一位置），可用原子操作、排序+分段 scan 或 sort+gather 实现。
- **Scheduling Primitive（调度原语）**：描述 N 维域迭代方式的高层指令，如 tile（分块）、vectorize（SIMD 向量化）、parallel（多线程）、compute_at（指定中间量计算层级）。
- **Segmented Scan（分段扫描）**：scan 的推广，对输入序列的连续分区各自独立做 scan（如 `[[1,2],[6],[1,2,3,4]]` 的 exclusive 分段 scan 为 `[[0,1],[0],[0,1,3,6]]`）。
- **Semi-Static Assignment（半静态分配）**：周期性 profile 执行并重新调整、两次调整之间保持静态的分配，适合变化缓慢的自适应网格/粒子模拟。
- **Sequence / Context Parallelism（SP/CP，序列/上下文并行）**：按序列或上下文长度切分激活（长序列场景）；CP 常用于超长上下文的 attention 计算。
- **Sequence（序列）**：有序元素集合（C++ `Sequence<T>`、Scala `List[T]`、Pandas Dataframe、PyTorch/JAX Tensor）；与数组不同，只能通过特定操作访问元素，不能直接按下标访问。
- **Sequential Consistency（顺序一致性，SC）**：Lamport 1976 提出——所有内存操作存在一个全局串行顺序（如同操作单一内存），且每线程保持程序序；维持 W→R、R→R、R→W、W→W 全部四种排序。
- **Serializability（可串行化）**：所有事务看起来以某个单一串行顺序提交，但语义不保证确切提交顺序。
- **Shared Address Space（共享地址空间模型）**：线程通过读写共享内存变量进行通信（通信隐式于 load/store），并需用锁、屏障等同步原语协调访问的编程模型。
- **Shared Memory（共享内存，`__shared__`）**：按 block 分配、块内所有线程可读写的片上高速存储，用于把 DRAM 中复用的数据暂存片上、节省带宽。
- **SIMD Utilization（SIMD 利用率）**：SIMD 通道实际忙碌的比例；不规则负载（如行长度不同的稀疏矩阵按行并行）会导致 lane 空转，需用分段 scan 等方式转为规则并行。
- **SIMD（Single Instruction, Multiple Data，单指令多数据）**：一条指令广播给多个 ALU，各 ALU 对不同数据同时执行；取指/译码成本被摊薄（如 AVX2 的 8-wide float）。
- **SIMT（Single Instruction, Multiple Thread）**：GPU 的执行模式：硬件线程流只有标量指令，硬件检测到多个线程执行同一指令时用 SIMD ALU 同时执行，分歧线程被 mask。
- **Simultaneous Multi-Threading（SMT，同时多线程）**：每个时钟从多个线程同时选指令放到不同 ALU 上执行；Intel Hyper-Threading（超线程，每核 2 线程）是其代表。
- **Single-reader/single-writer queue（单读单写队列）**：只允许一个生产者与一个消费者访问的队列，因 head/tail 各由一方写而完全无需同步，空/满时直接返回失败。
- **SM（Streaming Multiprocessor，流式多处理器）**：GPU 上的一个多线程 SIMD 核；V100 有 80 个 SM，每 SM 含 4 个 sub-core、256 KB 寄存器堆、128 KB shared+L1，最多驻留 64 个 warp。
- **Snooping（总线嗅探）**：基于广播的一致性方案——缓存控制器监听总线上所有一致性相关操作并按协议响应，同时响应本地 LD/ST 与互连广播两个方向的事件。
- **Softmax**：`softmax(x)_i = e^{x_i - m(x)} / Σ_j e^{x_j - m(x)}`（m 为行最大值，保证数值稳定）的归一化函数，把分数向量变为概率分布。
- **Software transactional memory（STM，软件事务内存）**：由编译器插桩 + 运行时（锁、时间戳、数据拷贝等）实现版本管理与冲突检测的事务内存，无硬件支持。
- **Sort（排序）**：按 key 对整个序列排序；数据并行实现常与 scan/gather 组合构建直方图、粒子网格等结构。
- **Span（跨度）**：并行算法最长串行依赖链的长度（关键路径），决定理想并行执行时间。
- **Spatial execution（空间执行）**：通过在芯片上布局计算单元与数据通路来"调度"计算，而非逐条指令执行；数据流架构的核心执行方式。
- **Spatial Locality（空间局部性）**：装入一条 cache line 顺带预载同行相邻地址的数据，使后续不同地址的访问命中。
- **Specialization / fixed-function（专用化 / 固定功能）**：把"程序控制"变成"固定数据通路"的硬件设计思路，能效收益来自省掉取指、译码、调度等控制开销。
- **Speculative execution（推测执行）**：事务系统"推测操作能成功完成，若被其他线程修改则中止重来"的执行方式，是 CAS 思路向多地址操作的推广。
- **Speedup（加速比）**：使用 P 个处理器相对 1 个处理器的执行时间缩减倍数，`speedup(P) = T(1) / T(P)`。
- **Spinlock（自旋锁）**：等待者通过原子 test-and-set 循环忙等抢锁的锁实现；低竞争时延迟低，高竞争时产生大量一致性通信与 CPU 浪费。
- **SPMD（Single Program, Multiple Data）**：只定义一个函数，但并行运行该函数的多个实例，各实例处理不同数据。
- **Stall（停顿）**：后续指令依赖未完成的指令（典型如访存）时处理器无法推进的状态；访存是停顿的主要来源。
- **Start-Flag Representation（起始标志表示）**：用与数据平行的 flag 序列（1 表示分段起点）表示嵌套序列的紧凑编码。
- **Starvation（饥饿）**：系统整体在推进但某些进程始终得不到资源的状态，通常不是永久状态，本质是公平性问题而非正确性问题。
- **Static Assignment（静态分配）**：不依赖运行时动态行为的分配；只要工作量与 worker 数已知即可预先确定。
- **STM barrier（软件事务屏障 / 插桩）**：编译器为事务内每个内存访问插入的 STM 运行时调用（tmRead/tmWr 等），负责簿记并需要函数克隆或动态翻译。
- **Streaming dataflow（流式数据流）**：数据以流水方式流经各计算/存储单元的执行模式，常与 kernel fusion、pipelining 配合，避免中间结果落回 off-chip 内存。
- **Strong atomicity / Weak atomicity（强原子性 / 弱原子性）**：是否要求非事务代码的访问也参与事务原子性语义；纯软件提供强原子性代价高昂。
- **Sub-core（子核）**：SM 内的一个执行分区，含独立取指/译码、warp selector 与一组 SIMD 功能单元（V100 每 sub-core 16 个 fp32 ALU），可在 16 个 warp 间交织调度。
- **Super-Linear Speedup（超线性加速比）**：加速比超过处理器数的现象，通常因工作集变小后装进 cache（或从磁盘换页变为装入内存）所致。
- **Superscalar Execution（超标量执行）**：处理器自动找出指令流中的独立指令，在多个执行单元（ALU）上并行执行（如每时钟 2 条），由乱序控制逻辑调度。
- **SWMR Invariant（单写多读不变量）**：任意时刻每个地址要么处于"单写者 epoch"（只有一个处理器可写），要么处于"只读 epoch"（多个处理器只读），是 coherence 的两条不变量之一。
- **Systolic array（脉动阵列）**：大量同构处理单元（PE）组成的阵列，数据像脉搏一样在相邻 PE 间逐周期流动（数据驱动/wavefront）；权重驻留 PE、输入与部分和沿阵列传播，通信只发生在相邻 PE 之间。
- **Tag（消息标识）**：send/receive 中可选的消息标识符，用于区分不同消息。
- **Task Granularity（任务粒度）**：单个任务包含的工作量；细粒度利于均衡但同步开销高，粗粒度反之。
- **Task Parallelism（任务并行）**：把问题分解为相互独立（或带依赖）的子任务、由调度器分配给 worker 执行的并行方式，强调"不同的工作单元"。
- **TCC（Transactional Coherence and Consistency）**：把事务当作一致性机制本身（所有事务所有时间）的 HTM 方案，lazy + optimistic，每执行步至多一个提交。
- **tcgen05**：B100 上编程 tensor core 的指令族（alloc 分配 TMEM、mma batch + commit 启动异步 MMA、fence 排序回收），标志"Not your father's CUDA"。
- **Temporal Locality（时间局部性）**：反复访问同一地址导致命中。
- **Tensor Core（张量核）**：SM 内专为矩阵乘设计的 SIMD 单元（如低精度 FP16 4×4×4 乘加），提供远超普通 fp32 ALU 的吞吐；需要大量并行工作（大 batch/大矩阵）才能喂饱。
- **Tensor core（张量核）**：GPU 上的专用矩阵乘加单元，一条 MMA 指令完成整个矩阵块乘加（如 8×4 × 4×8，fp16 输入、fp32 累加）；H100 的 tensor core 提供 989 TFLOPS fp16，占芯片算力绝大部分。
- **Tensor Parallelism（TP，张量并行）**：按 hidden 维切分权重/激活，多卡协作计算同一层；通信模式为 ReduceScatter+AllGather 或 AllReduce。
- **Test-and-Set（测试并置位）**：原子指令，读旧值并置位，返回旧值；`std::atomic_flag::test_and_set`、x86 `lock xchg` 等是其实现。
- **Test-and-set（测试并置位）**：原子指令，读回内存旧值并在旧值为 0 时置 1，是构建自旋锁的最小硬件原语。
- **Test-and-test-and-set lock（测试-测试并置位锁）**：先普通读自旋、观察到锁释放后才执行一次原子 test-and-set 的锁，大幅降低 coherence 流量（每次释放 O(P) 次失效）。
- **Test-and-Test-and-Set（TTAS）**：先普通读测试锁状态、空闲才执行 test-and-set 的改进自旋锁，把每轮测试的写降为仅锁释放时一次失效。
- **Thread Block Scheduler（线程块调度器）**：GPU 上的硬件工作调度器，把 block 按动态调度策略映射到 SM，需满足线程上下文数与 shared memory 等资源约束；block 完成即释放资源。
- **Thread Block（线程块）**：若干 CUDA 线程组成的组，块内线程并发运行、可经 shared memory 通信并用 `__syncthreads()` 同步；一个 block 整体驻留在同一 SM 上。
- **Thread-Level Parallelism（TLP，线程级并行）**：多个线程（指令流）同时在多个核上执行；软件通过线程 API（如 std::thread）向硬件暴露。
- **Throughput / Bandwidth（吞吐 / 带宽）**：单位时间完成的操作数或提供的数据量（如高速公路每小时车辆数、内存 GB/s）。
- **Throughput Computing（吞吐计算）**：以提升系统总吞吐为目标的设计哲学——可能延长单个线程的完成时间以换取多线程整体吞吐。
- **Thunderkittens**：CUDA 瓦片（tile）编程原语库——异步瓦片加载/存储、高级内存布局支持，面向高级开发者编写分块 kernel（含 FlashAttention 示例）。
- **ThunderKittens（TK）**：Embedded CUDA DSL template library，设计三原则：16×16 tile 为基本数据类型、处处异步、高层 GPU 协调模式（producer-consumer）。
- **Ticket lock（取号锁）**：通过"取号 + 等叫号"实现 FIFO 公平的锁，取号需原子自增、等号阶段只需普通读，每次释放仅一次失效。
- **Ticket Lock（票锁）**：取号（fetch_add 自己的票号）+ 等叫号（只读 now_serving）的锁；等待只读无失效、每释放仅一条失效（O(P) 通信），且 FIFO 公平。
- **Tiled tensor / tiled programming（分块张量 / 分块编程）**：以 16×16、32×32 等形状的 tensor tile 为编程与搬运单位（如 CUTLASS、Triton、Thunderkittens），目标是 GEMM 最大 TFLOPS 与低指令开销。
- **TMA（Tensor Memory Accelerator，张量内存加速器）**：H100 起的专用数据搬运单元，用拷贝描述符描述区域、单线程发起、硬件生成地址并异步搬运张量分块，配合 mbarrier 同步。
- **TMEM（Tensor Memory）**：B100 引入的专供 tensor core 数据存放的片上存储；配合 tcgen05 系列指令，单个线程即可执行 MMA（不再有 warp 概念）。
- **torch.compile**：PyTorch 的编译器，自动对关键 DNN 操作做融合与调度。
- **TPU（Tensor Processing Unit，张量处理单元）**：Google 的领域专用 DNN 加速器，核心是脉动阵列执行 matrix_multiply/convolve，指令集极小（read/write host memory、read weights、matrix_multiply、activate）。
- **Transaction descriptor（事务描述符）**：每线程一份的事务状态结构，含读集、写集、undo log 或 write buffer。
- **Transaction record（事务记录）**：每个数据项关联的指针大小记录，shared 态用版本号/共享读锁、exclusive 态用指向属主事务的写锁。
- **Transactional memory（TM，事务内存）**：以数据库事务为灵感、把一段内存访问序列声明为原子且隔离执行的同步抽象，`atomic { }` 即其编程形态。
- **Triton**：面向 GPU 的 DSL——支持把数据"块"加载进 shared memory 并对其做数据并行操作（自带两层分块的矩阵乘参考实现）。
- **True Sharing（真共享）**：多个处理器访问同一地址、通信由程序语义必然引起的共享；与伪共享相对。
- **TSO（Total Store Order）**：写缓冲引入的模型——只放松 W→R（自己的读可越过自己的写），W→W 仍保持；x86 使用一种未完全定义的 TSO。
- **TSV（Through-Silicon-Via，硅通孔）**：穿过 DRAM 芯片的垂直互连通孔，为 3D 堆叠提供高并行度的层间连接；堆叠底层为"逻辑层"（内存控制器）。
- **uniform（类型修饰符）**：变量在所有程序实例中值相同、只有一份存储；使用它纯属优化，与正确性无关。
- **Uniform（统一类型）**：ISPC 类型修饰符，表示所有 program instance 取值相同；使用它纯粹是优化，不影响正确性。
- **Varying（可变类型）**：ISPC 默认类型，每个 program instance 各有一份独立的取值。
- **varying（类型修饰符）**：变量在每个程序实例中各有一份（ISPC 默认），对应 SIMD 向量中的一个 lane；varying 值不能赋给 uniform 变量。
- **Victim（受害者）**：工作窃取中被空闲线程选中、被偷取工作的线程。
- **VLIW（Very-Long Instruction Word，超长指令字）**：一条指令同时指定多个不同运算（与 SIMD 的"一条指令对多数据做同运算"相对），由编译器静态安排并行。
- **Warp Selector（warp 选择器）**：sub-core 中每时钟从可运行 warp 里选一个取指执行的硬件单元。
- **Warpgroup / PTX**：Warpgroup 是 128 个连续线程的组合（跨 4 个 warp）；PTX（Parallel Thread Execution）是 NVIDIA 的虚拟指令集架构。
- **Warp（线程束）**：32 个连续 CUDA 线程组成的一组，是 GPU 硬件调度/执行的基本单位；不属于 CUDA 编程模型，而是 NVIDIA GPU 的重要实现细节（block 内 0-31 为 warp 0，32-63 为 warp 1……）。
- **Work Queue（工作队列）**：存放待执行任务的列表，worker 从中取任务、产生新任务时再推入。
- **Work Stealing（工作窃取）**：每个 worker 有本地队列，空闲时从随机选中的 victim 线程队列偷取工作的动态负载均衡调度。
- **Work-Efficient Scan（工作高效 scan，Blelloch 算法）**：up-sweep（自底向上建树）+ down-sweep（自顶向下修正）两阶段，Work = O(N)、Span = O(log N)，但指令数与常数因子较大。
- **Workload Balance（负载均衡）**：让所有处理器在程序执行期间持续计算且同时完成各自工作的目标状态。
- **Work（工作量）**：并行算法执行的总操作数；与 Span 一起评价并行算法。
- **Write Buffer（写缓冲）**：处理器把写暂存其中即可继续执行后续指令的硬件缓冲，用于隐藏写延迟；是现代 x86/ARM/RISC-V 处理器的标配，也是 TSO 行为的来源。
- **Write-Back Cache（写回缓存）**：写只改缓存行并置 dirty bit，行被换出时才写回内存；吸收写流量，但 dirty 意味着独占所有权，需要更复杂的一致性协议。
- **Write-Through Cache（写直达缓存）**：每次写立即穿透到内存，实现简单但带宽需求极高（每个写都占用内存带宽）。
- **Write-write conflict（写写冲突）**：两个事务都处于未提交状态且都写了地址 X。


---

## 附录：资料与作业说明

### 已获取的公开资料清单

所有资料均下载自课程公开网站，保存在本工作目录 `cs149_fall25/` 下：

| 资料 | 位置 | 说明 |
|---|---|---|
| 课程主页 / 课程信息页 / 讲座索引页 | `pages/*.html` | 主页含完整日程表、作业链接；课程信息页含评分、政策、FAQ |
| 全部 18 讲幻灯片 PDF | `slides/*.pdf` | 与讲座页面上链接一致 |
| 幻灯片全文提取文本 | `raw/*.txt` | 每讲一文件，含页码标记 |
| 书面作业 1–4 PDF 与文本 | `pdfs/written_asst*.pdf`、`raw/written_asst*.txt` | 全部公开可下载 |
| 编程作业 README | `pdfs/asst*_README.md` | 来自 GitHub 公开仓库 stanford-cs149/asst1…asst5-kernels |
| 结构化数据记录 | `lectures_data.json` | 每讲的编号、日期、主题、材料链接、文本路径 |
| 最终学习笔记 | `cs149_fall2025_learning_notes.md` | 本文档 |

### 讲座幻灯片 PDF 链接（公开）

| 讲次 | 标题 | 幻灯片 PDF |
|---|---|---|
| 1 | Why Parallelism? Why Efficiency? | https://gfxcourses.stanford.edu/cs149/fall25content/media/efficiency/01_efficiency_hyF1AJq.pdf |
| 2 | A Modern Multi-Core Processor (Part I) | https://gfxcourses.stanford.edu/cs149/fall25content/media/multicore1/02_basicarch.pdf |
| 3 | Multi-Core Architecture (Part II) + ISPC | https://gfxcourses.stanford.edu/cs149/fall25content/media/multicore2/03_multicore2-ispc_WueDBzT.pdf |
| 4 | Parallelizing Code: An Example Thought Process | https://gfxcourses.stanford.edu/cs149/fall25content/media/thoughtprocess/04_progbasics.pdf |
| 5 | Program Optimization 1: Work Distribution and Scheduling | https://gfxcourses.stanford.edu/cs149/fall25content/media/perfopt1/05_progperf1.pdf |
| 6 | Program Optimization 2: Locality and Communication | https://gfxcourses.stanford.edu/cs149/fall25content/media/perfopt2/06_progperf2.pdf |
| 7 | GPU Architecture and CUDA Programming | https://gfxcourses.stanford.edu/cs149/fall25content/media/gpuarch/07_gpuarch.pdf |
| 8 | Data-Parallel Thinking | https://gfxcourses.stanford.edu/cs149/fall25content/media/dataparallel/08_dataparallel.pdf |
| 9 | Efficiently Evaluating DNNs on GPUs | https://gfxcourses.stanford.edu/cs149/fall25content/media/dnninference/09_dnneval.pdf |
| 10 | Hardware Specialization | https://gfxcourses.stanford.edu/cs149/fall25content/media/accelerators/10_Specialized.pdf |
| 11 | Programming Systems for Specialized Hardware | https://gfxcourses.stanford.edu/cs149/fall25content/media/proghardware/11_SpecializedHardwareProgramming.pdf |
| 12 | Mapping AI Applications to the Datacenter Computer | https://gfxcourses.stanford.edu/cs149/fall25content/media/aidatacenter/12_AI_DatacenterMapping.pdf |
| 13 | Domain-Specific Programming Systems and AI-Driven Performance Optimization | https://gfxcourses.stanford.edu/cs149/fall25content/media/aiperfoptimization/13_autooptimize.pdf |
| 14 | Cache Coherence | https://gfxcourses.stanford.edu/cs149/fall25content/media/cachecoherence/14_coherence.pdf |
| 15 | Implementing Synchronization + Memory Consistency | https://gfxcourses.stanford.edu/cs149/fall25content/media/sync_consistency/15_consistency.pdf |
| 16 | Fine-Grained Locking and Lock-Free Programming | https://gfxcourses.stanford.edu/cs149/fall25content/media/finegrainedsync/16_finegrainedlock.pdf |
| 17 | Transactional Memory (Part I) | https://gfxcourses.stanford.edu/cs149/fall25content/media/transactions/17_transactionalmem.pdf |
| 18 | Transactional Memory (Part II) + AMA | https://gfxcourses.stanford.edu/cs149/fall25content/media/wrapup/18_transactionalmem_A4wu1Q8.pdf |

### 无法访问的内容（已记录，未下载）

- **2025 年讲座视频**：课程主页明确说明 "We cannot distribute lecture videos to the public this year"。替代资源：2023 年版公开播放列表 https://www.youtube.com/playlist?list=PLoROMvodv4rMp7MTFr4hQsDEcX7Bx6Odp
- **Ed Discussion 论坛 / Canvas 内容**：需校内账号登录，非公开。
- **作业内部文件**（如测试用例、手写提纲图片、AWS 配置说明等 GitHub 仓库内非 README 文件）：编程作业主体（README）公开，其余 starter code 需注册学生身份使用。

### 作业说明汇总

- **Programming Assignment 1**（10 月 6 日截止，100 分 + 6 分附加）：在 myth 机器的四核 Intel Core i7（4.2GHz、AVX2、每核 2 硬件线程）上，分析 SIMD（AVX2 8 宽单精度）与多核并行、超线程对程序性能的影响；练习测量与推理并行程序性能。编程量小，分析为主。对应第 1–4 讲（处理器架构、ISPC、并行化思路）。
- **Programming Assignment 2**（10 月 16 日截止，100 分）：从零实现一个 C++ 任务执行库：先支持批量数据并行任务启动（类似 ISPC task launch），再扩展为支持带依赖的任务图调度；使用线程池、互斥锁、条件变量。对应第 5 讲（工作分配与调度）与第 15 讲（同步原语）。
- **Programming Assignment 3**（10 月 30 日截止，100 分）：在 AWS GPU 上用 CUDA 编写并行圆形渲染器；重点是可并行构建的数据结构设计。对应第 7–8 讲（CUDA、数据并行思维）。
- **Programming Assignment 4**（11 月 13 日截止，100 分）：在 AWS Trainium2 加速器上编写并优化 kernel：第 1 部分向量加法与矩阵转置，第 2 部分融合卷积+最大池化层；体会循环分块（blocking）与循环融合（fusion）的局部性价值。对应第 9–11 讲（DNN 评估、硬件特化、专用硬件编程）。
- **Programming Assignment 5**（12 月 4 日截止，100 分，**不可用晚交天数**）：开放课题，从若干 AI kernel 中选择，在 H100 GPU 上用 CUDA / Triton / TileLang 优化，允许使用 LLM 辅助，有班级排行榜；成绩依据优化工作日志评定。对应第 13 讲（AI 驱动性能优化）及全课程性能工程能力。
- **Written Assignment 1**（10 月 9 日）：Hardware Basics——多核/SIMD 峰值吞吐计算、处理器基础。
- **Written Assignment 2**（10 月 21 日）：To Fuse or Not to Fuse——循环融合与并行化分析。
- **Written Assignment 3**（11 月 6 日）：Improving Locality on Specialized Hardware——图像模糊两遍算法的局部性优化与专用硬件设计。
- **Written Assignment 4**（12 月 3 日）：MSI Coherence Protocol Warmup——MSI 状态机、CAS 与一致性交互分析。

> 注：以上全部作业描述均已公开获取；不存在"作业描述未公开"的情况。若个别内部文件（如测试用例）需要课程账号，本文档已在"无法访问的内容"一节说明。

{% endraw %}
