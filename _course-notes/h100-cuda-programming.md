---
title: "H100 CUDA 编程 · 完整教学合集（Lesson 1–10）"
excerpt: "NVIDIA H100 GPU 的 CUDA 编程完整教学笔记合集（Lesson 1–10），涵盖 Hopper 架构、TMA 与 cuTensorMap、异步与屏障、内联 PTX、WGMMA 与集群。"
collection: course-notes
permalink: /course-notes/h100-cuda-programming
toc: true
toc_sticky: true
---
{% raw %}
> 由 12 份 H100 单课笔记按课程顺序（Lesson 1 → Lesson 10）原文合并，未修改内容；各讲之间以分隔线隔开。

# H100 架构介绍（Introduction to H100）

> 本文档基于课程讲义《1. Introduction to H100.pdf》（Lesson 1）整理，系统介绍 NVIDIA H100 GPU 的
> 体系结构，作为学习后续 CUDA 专题（TMA、异步执行、Tensor Core、WGMMA、流并发等）的架构基础。
> 建议先完成 `02-CUDA-并行编程与-GPU-体系结构.md` 中的同步模型与内存层级基础，再阅读本文。

---

## 目录

1. [概述与规格](#1-概述与规格)
2. [核心变革：全面异步化](#2-核心变革全面异步化)
3. [TMA（Tensor Memory Accelerator）](#3-tmatensor-memory-accelerator)
4. [同步拷贝 vs 异步拷贝](#4-同步拷贝-vs-异步拷贝)
5. [第四代 Tensor Core 与 WGMMA](#5-第四代-tensor-core-与-wgmma)
6. [其他关键特性](#6-其他关键特性)
7. [GPU 整体组织结构](#7-gpu-整体组织结构)
8. [GigaThread Engine](#8-gigathread-engine)
9. [存储层级：HBM3 → L2 → Shared → Registers](#9-存储层级hbm3--l2--shared--registers)
10. [SM 详解](#10-sm-详解)
11. [Warp 与指令调度](#11-warp-与指令调度)
12. [线程块在 SM/SMSP 间的分布](#12-线程块在-smsmsp-间的分布)
13. [PCIe 5.0 主机接口](#13-pcie-50-主机接口)
14. [总结与学习衔接](#14-总结与学习衔接)

---

## 1. 概述与规格

H100 是 NVIDIA 基于 **Hopper 架构** 的旗舰 GPU，专为深度学习与高性能计算设计。

| 规格项 | 数值 | 说明 |
|--------|------|------|
| 发布时间 | 2022 年 9 月 | 基于 TSMC 4N 工艺 |
| 架构 | Hopper | 继 Ampere (A100) 之后 |
| 形态 | PCIe（300W）/ SXM（700W） | 两种封装 |
| 显存 | 80 GB HBM3 | 带宽 3.35 TB/s |
| SM 数量 | 132 个 | 启用的 SM |
| Tensor Core | 528 个 | 每 SM 4 个 |
| 晶体管数 | 800 亿（80B） | 定制 4N 节点 |

> **核心定位**：H100 相比上一代**最大的改进是引入了异步（Asynchronous）特性**——这是贯穿整个
> Hopper 架构学习的主题（TMA、异步拷贝、异步屏障、WGMMA、流并发等都在此基础之上）。

---

## 2. 核心变革：全面异步化

传统 GPU 编程（同步模型）里，数据搬运和计算是**串行**的：

```
同步模型：拷贝 → 计算 → 拷贝 → 计算 → ...（步骤串行，延迟无法隐藏）
```

H100 的核心变革是把数据搬运交给**专用硬件在后台完成**，让 SM **专注计算**：

```
异步模型：SM 专注计算，TMA 在后台搬运数据，二者重叠执行
```

> 这就是 `02` 文档里"同步执行模型 vs 异步执行模型"一节的硬件基础——Hopper 把"流水线重叠"这一
> 软件优化思想，做成了**架构级的一等公民**。

---

## 3. TMA（Tensor Memory Accelerator）

**TMA 是 Hopper 引入的、每个 SM 都配备的专用数据搬运单元**，用于把"张量拷贝"从 SM 上卸载出去。

### 3.1 为什么需要 TMA

- **目的**：加速 **memory-bound（访存受限）** 操作。
- **过去**：global memory ↔ shared memory 的搬运要靠**每个线程**做地址计算、循环遍历、发多条指令。
- **现在**：TMA 通过**描述符（descriptor）** 异步地发起搬运，**单个线程就能发起整个数据搬移**，
  硬件在后台负责 stride/offset/bounds 与数据搬移。

### 3.2 对比

| | 传统（逐线程拷贝） | TMA |
|---|---|---|
| 发起者 | 每个线程各搬一段 | 单个线程发起整块搬移 |
| 地址/边界处理 | 软件逐条指令算 | 硬件自动处理 stride/offset/bounds |
| 是否占用 SM 资源 | 占用大量线程与指令槽 | 卸载到专用单元，SM 专注计算 |
| 与计算重叠 | 需手工流水 | 天然后台异步 |

> TMA 与 `cp.async.bulk`（后续专题）配合，是 Hopper 上高性能 GEMM/注意力 kernel 的核心搬运动力。

---

## 4. 同步拷贝 vs 异步拷贝

这是理解 Hopper 编程模型的**最关键概念之一**：

- **同步拷贝（Synchronous copy）**：发起拷贝的线程（或 warp）**必须等拷贝完成后才能继续**。
  心智模型简单——控制权回到线程时，数据一定已就位。
- **异步拷贝（Asynchronous copy）**：发起只**启动**传输，发起者**可以立刻做别的事**，
  等到真正要用数据前再 `wait/check` 完成。

### 4.1 异步拷贝的价值

```
能够重叠执行 → 用计算隐藏内存延迟
```

典型场景是**流水线 kernel（GEMM/attention）**：

```
计算 tile i 的同时，预取 tile i+1
```

### 4.2 代价

异步的唯一代价是**簿记（bookkeeping）复杂度上升**——你要自己跟踪"哪些数据已到位、何时能安全使用"。
这正是后续专题（异步屏障 `cuda::barrier`、`cuda::pipeline`、TMA）要解决的问题。

---

## 5. 第四代 Tensor Core 与 WGMMA

每个 SM 有 **4 个 Tensor Core**（全 GPU 共 528 个）。第四代 Tensor Core 的关键特性：

### 5.1 WGMMA（Warp Group Matrix Multiply Accumulate）

- **4 个 warp（= 128 线程）协作完成一个 MMA**（矩阵乘加）。
- 与传统 `wmma`（单 warp）相比，WGMMA 通过 warp 组获得更大的矩阵 tile 和更高的吞吐。

### 5.2 稀疏 Tensor Core

- 支持**结构化稀疏**（structured sparsity，数据以预定义布局置零）的快速 MMA。

### 5.3 FP8 支持

- 提供针对 **FP8** 的专用指令，可达 **1,979 TFLOPS**（相对上一代巨大提升）。
- **吞吐**：FP16 密集为 **1024 FLOPS/cycle**，FP8 为 **2048 FLOPS/cycle**。

### 5.4 WGMMA 寄存器布局（重要约束）

| 矩阵 | 存放位置 |
|------|---------|
| A | shared memory **或** 寄存器 |
| B | **必须**在 shared memory（SMEM），**不能**放寄存器 |
| C | 分布在 **4 个 warp（128 线程）的寄存器**中 |

> FP8/FP16 时，WGMMA 期望数据**打包进寄存器**（例如 2 个 FP16 装进 1 个 32 位寄存器）。
> 这些约束是后续学习 `wgmma` 专题时必须记住的硬性要求。

---

## 6. 其他关键特性

- **Transformer Engine**：面向 Transformer 的混合精度（FP8/FP16）加速。
- **4th-Gen NVLink**：GPU 间高速互连（详见 §7.5）。
- **DPX 指令**：动态规划加速（如 Smith-Waterman 序列比对）。
- **50 MB L2 Cache**：大容量统一二级缓存（详见 §9.3）。

---

## 7. GPU 整体组织结构

整个 GPU die 划分为：

- NVIDIA **GigaThread Engine**（线程块分发）
- **8 个 GPC**（Graphics Processing Cluster）
- **HBM3 堆栈 + 内存控制器**
- **PCIe 5.0 主机接口**
- **NVLink 交换机 / 端口 / Hub**
- **L2 缓存切片**

---

## 8. GigaThread Engine

**GigaThread Engine 是把一次 kernel 启动分发成线程块（CTA）的硬件**：

- 跟踪哪些 CTA **未启动 / 运行中 / 已完成**。
- 当某个 SM 有容量容纳下一个 CTA 时，GigaThread Engine（及相关前端逻辑）就把下一个 CTA 分给它。
- 强制**占用率（occupancy）限制**；在 Hopper 上还**理解 cluster（线程块簇）**（后续专题）。

> 简单理解：它是"线程块调度器"，决定哪个 block 去哪个 SM。

---

## 9. 存储层级：HBM3 → L2 → Shared → Registers

H100 的数据流遵循经典的**越近越快、容量越小**层级：

```
HBM3（大容量、高带宽）→ L2 Cache（统一 50MB）→ Shared Memory（低延迟）→ Registers（最快）
```

### 9.1 HBM3（片外显存）

- 80 GB，3.35 TB/s，**5 个堆栈**（实际 6 个，1 个因良率关闭）。
- **5120 位总线宽度** → 使 TMA 能**一次搬 128 字节**。
- 通过 **10 个独立的 512-bit 内存控制器**连接。
- 数据路径：`SM → (L1D/coalescer) → L2 → memory partition/crossbar → memory controller → HBM 堆栈`。
- 内存控制器负责 DRAM 协议与调度（activate/precharge/读写时序、重排以最大化行命中、地址 → channel/bank/row/col 映射）。

### 9.2 GPC 与 TPC

- **GPC = 一组 SM 的分组**（讲义记为每 GPC 18 个 SM；H100 共 8 个 GPC、132 个启用 SM）。
- 每个 GPC 连接**自己专属的 L2 分片**；取数到 shared memory 时走 `HBM → L2 → L1`。
- GPC 内部支持 **DSMEM（分布式共享内存，distributed shared memory）**：同一 GPC 内的 SM 可访问彼此的 shared memory；**GPC 之外没有 DSMEM**。
- **TPC 内放 2 个 SM**，作用就是让这 2 个 SM 之间的 DSMEM 通信**特别快**。

### 9.3 L2 Cache

- **50 MB**，分成 **25 MB 分区**。
- L2 是**分区的**：同一 GPC 内的 SM 到"直连的那个 L2 分区"路径更近——访问大多命中邻近分区时，有效延迟/带宽更好。
- **128 字节 cache line，32 字节 sector**：一次内存请求可触及单条 128B line 的 1–4 个 sector。
  访存"不合并"会导致 sector/request 爆炸。
- L2 吸收并合并来自 SM 的**零碎小写**，转成**干净的大块高效写**到 HBM。

### 9.4 统一 Shared Memory + L1（每 SM）

- 总共 **256 KB**、**33 TB/s 带宽**，分成 **32 个 bank**（每 bank 4 字节）。
- **可配置的 shared memory 上限 = 228 KB/SM**；**每 block 上限 = 227 KB**（CUDA 保留 1 KB）。
- 无冲突时 load/store 达 **128 B/cycle**；TMA 异步拷贝能逼近峰值带宽并与计算重叠。
- bank 按**连续的 4 字节字**交错。
- L1 cache 充当**合并缓冲（coalescing buffer）**：收集 warp 请求的数据并高效交付。
- cache line = 128 B，sector = 32 B，可逐 sector 填充。

> **shared memory 与 L1 此消彼长**：调大 shared，留给 L1 缓存的空间就变小，反之亦然。
> 因此 tile 大小需要权衡（见下）。

### 9.5 关于 shared memory 的更多细节

- **静态 shared memory**（编译期数组）**架构上限 48 KB**（为兼容旧 compute capability）。
  要超过它必须用**动态 shared memory**：`extern __shared__`。
- GEMM 的最优 tile 通常 **64–128 KB/block**；**过大的传输会损害延迟隐藏**。

### 9.6 Registers（寄存器）

- 每线程私有的片上寄存器：**最快带宽、最低延迟**，每 SM **256 KB**，最多 **128 读/写每 cycle**。
- **寄存器经常是限制因子**：若 kernel 每线程用 R 个寄存器，最大驻留线程数 ≈ `floor(65536 / R)`。
  例如 128 regs/thread → 每 block 最多 512 线程。
- **32 位是寄存器基本单位**；用 FP16/FP8 时需用**打包类型**把 2/4 个元素塞进 1 个寄存器。
- **功耗效率**：寄存器比 shared memory 高效 30–50×，比 HBM3 高效 **1000×+**。
- **寄存器用量在编译期确定**，不是运行时。
- 寄存器不够时**溢出（spill）到 local memory（很慢）**；**CUDA 13.0 新增"先溢出到 shared memory、
  shared 不够才回落 local memory"**。
- **粒度陷阱**：即使 63 → 65 regs/thread 的小变化，也可能因内部粒度取整而**丢掉驻留 warp**，导致性能下降。

---

## 10. SM 详解

SM 是 GPU 的**基本执行单元**，执行 CUDA kernel 的线程块。其组件包括：

- FP32 CUDA Core、INT/FP64 单元
- 第四代 Tensor Core
- Shared Memory / L1 cache
- L1 指令缓存
- Warp Scheduler（warp 调度器）
- Dispatch Unit（分发单元）
- Registers（寄存器）
- L0 指令缓存

### 10.1 Quadrant / SMSP（四个子分区）

每个 SM 划分为 **4 个完全相同的子分区**，叫 **Quadrant 或 SMSP**。每个 quadrant 含：

| 单元 | 每 quadrant | 每 SM |
|------|-----------|-------|
| FP64 单元 | 16 | 64 |
| FP32 CUDA Core | 32 | 128 |
| INT32 单元 | 16 | 64 |
| Tensor Core | 1 | 4 |
| Load/Store 单元（LSU） | 8 | 32 |
| SFU（特殊函数单元） | 4 | 16 |

- 每个 quadrant 还有**自己的 warp 调度器、L0 缓存、分发单元、寄存器文件**。
- 每个 quadrant **每个 cycle 都能向本地单元发指令**，最多同时调度 **16 个 warp**。

### 10.2 SFU（Special Function Unit，特殊函数单元）

- 每 SM **16 个 SFU**（每 SMSP 4 个），每个每 cycle 发 1 条指令 → 每 SM **16 ops/cycle**。
- 负责**复杂数学函数**：sin、cos、log、exp、sqrt、倒数等（在 CUDA Core 上算会很贵）。
- 用**多项式近似 + 查找表 + 插值**，**牺牲一点精度换巨大吞吐**。
- **注意**：若所有线程同时调用数学函数，SFU 会**停顿（stall）**。

### 10.3 LSU（Load/Store Unit）

- 执行**每线程的内存指令**：load、store、atomic。
- 每 quadrant 8 个，共 **32 个/SM**，直连 L1、L2。
- warp 执行 ld/st/atom 时，LSU **合并 32 个线程的地址**，形成 cache-line/sector 请求并查询 L1：
  命中则快速回寄存器；未命中则继续到 L2/DRAM，并可能回填 L1。
- **合并访存**时，LSU 把 warp 的 32 个地址合并成**最少的 cache line**，减少请求/重放/停顿，提升有效带宽。

### 10.4 INT / FP32 / FP64 单元

| 单元 | 每 SM | 全 GPU（SXM5） | 用途 |
|------|-------|---------------|------|
| INT32 | 64 | 8,448 | 内存寻址、循环控制、通用整数运算 |
| FP32 CUDA Core | 128 | 16,896（132 SM） | 通用单精度计算（"默认"精度） |
| FP64 | 64 | — | 科学计算（独立于 FP32 的专用核） |

- INT 单元可与浮点数据通路**并行执行**：一边算地址（INT）、一边算数据（FP），互不阻塞。

### 10.5 L0 与 L1 指令缓存

- **L0 I-Cache**：指令流的**微缓冲**，只装很少的指令（通常几个紧凑循环）。
  - 作用：以硬件速度喂指令，避免饱和 shared memory 带宽；**只求速度、不求容量**。
  - **激进循环展开 / 内联可能超出 L0 容量**。
  - **SMSP 0 的代码不能用 SMSP 1 的 L0**。
- **L1 I-Cache**：缓冲 SASS 机器码，让 warp 调度器始终有指令可发；解耦执行单元与存储层级
  （否则每次取指令要几百周期，造成大量停顿）。

---

## 11. Warp 与指令调度

### 11.1 Warp（线程束）

- 一组 **32 个线程**：
  - 从线程块一起创建（线程 0–31 = warp 0，32–63 = warp 1，……）。
  - 每 cycle 共享一个 warp 调度器的调度。
  - **寄存器私有，但指令流（基本）共享**。
- 为什么是 32：在"每个线程的控制粒度"与"每条指令的工作量"之间取平衡；32 线程也让常见访存模式天然对齐缓存/总线粒度、易于合并。

### 11.2 Warp Scheduler（warp 调度器）

负责指令发射：

- 每 cycle 从就绪 warp 中选一个发射指令。
- 处理 warp 级分支与**分歧（divergence）**。
- 维护 **scoreboard**，跟踪哪些 warp 真正就绪。
- 确保源寄存器在发射前已从寄存器文件读出或由流水线前递。
- 处理结构限制（如一个 warp 同时能有多少长延迟操作在飞、执行管道的可用性等）。

### 11.3 Warp 如何执行

```
warp 调度器扫描分配给该 SMSP 的 16 个 warp → 挑 1 个就绪的
→ 取指令 → 把选中的 warp ID + 程序计数器发给分发单元 → 发射到对应执行管道
→ 等待分发反馈
```

- warp 调度器总是尽量发射指令、让空闲 warp 忙起来。
- **实践建议**：让 block 维度是 **4 个 warp 的倍数（如 128 线程）**，以便工作均匀分布在 4 个子分区上。

### 11.4 Dispatch Unit 与 Dispatch Port

- **Dispatch Unit**：物理上把 warp 的操作**发往对应功能单元**（每个 cycle 发到执行管道）。
- **Dispatch Port**：分发单元连接具体执行管道（Tensor Core / CUDA Core 等）的**物理连接点**。
  - 多个执行管道**竞争同一个 port** → 不能同时接收指令。
  - 每 cycle 一个共享 port 只能发 1 条指令 → 分发单元需串行化发射。
  - **port 只在"发射指令"时需要，不占整个执行周期**。

---

## 12. 线程块在 SM/SMSP 间的分布

- **GigaThread Engine 选择有足够资源的 SM** 来容纳线程块。
- 硬件**不会把整个 block 塞进单一 SMSP**（除非 block 特别小）；而是把 block 切成 warp，
  **轮转（round-robin）分配到各 SMSP**。
- `__syncthreads()` 由 **shared memory 中的屏障逻辑**处理（所有 SMSP 共享）：
  其它 warp 进入休眠，直到所有 warp 都完成执行。

---

## 13. PCIe 5.0 主机接口

- 主 **128 GB/s** 数据通道，连接 H100 与主机 CPU / 系统内存。
- 连接 GPU 与**网卡（NIC）**的关键物理桥梁，支持 **GPUDirect RDMA**：网卡直接读写 GPU 内存，**不占用 CPU**。

---

## 14. 总结与学习衔接

### 14.1 H100 的本质（PDF 结论）

1. **决定性特征是"全面异步执行"**：从简单同步拷贝，转向用 **TMA** 在后台处理数据搬移，让 SM 专注计算。
2. **专用化（Specialization）**：针对每个瓶颈用专用单元——
   - **TMA** → 内存带宽；
   - **第四代 Tensor Core** → 矩阵运算；
   - **SFU** → 复杂数学函数。
3. **高效的数据流**：`HBM3（高带宽）→ L2（50MB 统一）→ Shared Memory（低延迟）→ Registers（最高速）`。

### 14.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 / 文档 |
|---------|-------------------|
| TMA、异步拷贝 | 《4. cuTensorMap》《5. cp.async.bulk》《3. Asynchronicity and barriers》 |
| WGMMA / Tensor Core | 《6. WGMMA-1》《7. Wgmma part 2》 |
| Cluster / DSMEM | 《2. Clusters, Data types, inline PTX, State Spaces》 |
| SM / warp / 合并访存 | `02-CUDA-并行编程与-GPU-体系结构.md` 及 `code/02-cuda-gpu-architecture/` |
| 存储层级 → tiled GEMM | `03-线性代数与矩阵计算优化.md` 及 `code/03-linear-algebra-matmul/` |
| compute/memory-bound | `04-深度学习模型基础.md` |

### 14.3 一句话记忆

> **H100 = 全面异步化 + 专用单元（TMA 搬数据 / Tensor Core 算矩阵 / SFU 算函数）+ 分级存储。**
> 学习 Hopper 编程，核心就是学会"让 TMA 在后台搬数据、让 Tensor Core 在前台算矩阵、让 SM 专注计算"。

---

> 参考来源：`1. Introduction to H100.pdf`（Lesson 1 - Introduction to H100，35 页）。

---

# H100 · Clusters、数据类型、内联 PTX、状态空间

> 本文档基于课程讲义《2. Clusters, Data types, inline PTX, State Spaces.pdf》（Lesson 2，47 页）整理，
> 覆盖四个主题：**线程块簇（Thread Block Clusters）**、**数据类型（Data Types）**、
> **内联 PTX（Inline PTX）**、**状态空间与指针（State Spaces & Pointers）**。
> 前置阅读：`H100-架构介绍.md`（尤其 GPC、DSMEM、Tensor Core 相关内容）。

---

## 目录

1. [线程块簇（Thread Block Clusters）](#1-线程块簇thread-block-clusters)
2. [分布式共享内存（DSMEM）](#2-分布式共享内存dsmem)
3. [创建与使用线程块簇](#3-创建与使用线程块簇)
4. [PTX 与内联 PTX](#4-ptx-与内联-ptx)
5. [PTX 状态空间（State Spaces）](#5-ptx-状态空间state-spaces)
6. [数据类型（Data Types）](#6-数据类型data-types)
7. [内存地址与 Shared Memory Bank](#7-内存地址与-shared-memory-bank)
8. [指针与状态空间](#8-指针与状态空间)
9. [总结与学习衔接](#9-总结与学习衔接)

---

## 1. 线程块簇（Thread Block Clusters）

### 1.1 什么是线程块簇

**Cluster 是"最多 16 个线程块"的集合，保证被共同调度到同一 GPC 内相邻的 SM 上并发执行。**

它把传统的 CUDA 编程层级从**三级**扩展为**四级**：

```
线程（thread）→ 线程块（thread block）→ 线程块簇（thread block cluster）→ 网格（grid）
```

- 簇内所有 block 在**物理上相邻的 SM** 上并发运行，从而支持**跨 SM 的高效协作**。
- 使用 cluster 的**最主要动机**：大部分时间是为了使用**分布式共享内存（DSMEM）**。

### 1.2 为什么需要 cluster

- 每个线程块受**单个 SM 上有限的共享内存与算力**约束。
- 线程若想访问更多数据，就得去**全局内存**取——昂贵。
- 对 **top-k、矩阵乘法**这类需要"访问其它 SM 上的线程块数据"的算法，cluster 提供了解决方案。
- 它让算法可以**用少量内存带宽换取跨多个 SM 的、显著扩大的数据可访问性**。

### 1.3 共享内存"池化"的直觉

- 把若干线程块的共享内存**汇聚**起来，不同 block 的线程就能访问所需数据，而无需走全局内存。
- 但**不是真的倒进一个池子**：每个 block **仍然拥有自己那部分共享内存**，区别只在于——
  现在线程可以**访问其它 block 的共享内存**。
- 这个"池"通常**很小，且被限制在一个 GPC 内**。

### 1.4 几个要点

- **线程块仍然运行在单个 SM 上**——这个一对一关系没有变。
- SM 可以同时运行多个线程块——以前可以，现在依然可以。
- **Cluster 是软件概念，GPC 是硬件概念**。
- **没有自动分配**：你必须在代码里显式定义 cluster；调度器不会自动把 block 分组。

### 1.5 关于 cluster 大小（重要权衡）

| cluster 大小 | 效果 |
|-------------|------|
| 8 个 block | 每个 GPC 可容纳 2 个 cluster，高效利用约 16 个 SM |
| > 8 | 需显式设置 `cudaFuncAttributeNonPortableClusterSizeAllowed`；≤8 时代码保持可移植 |
| 16 个 block | 每个 GPC 只能容纳 1 个 cluster，每 GPC 约闲置 1–2 个 SM（全 GPU 约闲置 18 个 SM） |
| 2 个 block | **实践发现最优**——更大的 cluster 有隐藏同步开销（4+ 为 87 周期，16 为 150 周期） |

> **结论**：cluster 不是越大越好；**size=2 往往最优**，因为更大的 cluster 带来同步开销且浪费 SM。

---

## 2. 分布式共享内存（DSMEM）

### 2.1 定义

**DSMEM 是 H100 的特性，允许簇内不同 SM 之间直接访问共享内存。**

- H100 为 cluster 实现了**专用的 SM-to-SM 网络**，提供对远端共享内存的**快速、低延迟**访问。
- 使得一个 SM 能对**其它 SM 的共享内存**执行 load / store / atomic 操作。
- DSMEM 可与 L2 cache 访问**同时使用**——应用在 SM 间通信时可**叠加两条通路的总带宽**。

### 2.2 Multicast 与 TMA + DSMEM

- 可以把 **TMA 用于 DSMEM 的异步拷贝**。
- DSMEM 最重要的特性之一是 **multicast（多播）**：把数据**同时投递到多个 SM 的共享内存**。
- **TMA multicast 绕开了 SM-to-SM 网络的瓶颈**：与其让线程显式跨 DSMEM 读写（造成拥塞与同步开销），
  不如让**每个 cluster 只需一个线程发起一次 TMA multicast**，把数据一次性分发给所有 SM。

---

## 3. 创建与使用线程块簇

### 3.1 编译期定义

在 kernel 声明里用 `__cluster_dims__` 属性定义 cluster 维度：

```cuda
// 编译期定义 cluster 维度（例如 2×1×1）
__global__ void __cluster_dims__(2, 1, 1) my_kernel(...) { ... }
```

- 定义后可以正常启动 kernel，但 **grid 维度必须是 cluster 大小的整数倍**。

### 3.2 Cluster 句柄（cooperative_groups）

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

cg::cluster_group cluster = cg::this_cluster();          // 当前线程所属的 cluster 句柄
int* remote_smem = cluster.map_shared_rank(smem, target_block_rank);  // 映射到目标 block 的共享内存
remote_smem[idx] = value;                                // 跨 SM 写共享内存
unsigned int cluster_size = cluster.num_blocks();        // cluster 中 block 数
unsigned int cluster_rank = cluster.block_rank();        // 本 block 在 cluster 中的 rank
```

> 讲义提到：实际开发中**不常用** `cooperative_groups` 这套 API，**更常用 `mapa`**（见 §8.4）
> 把共享内存地址转换成 cluster 内地址。

### 3.3 PTX 特殊寄存器（cluster 相关）

这些寄存器提供 cluster 内线程块的信息：

| 寄存器 | 含义 |
|--------|------|
| `%cluster_ctaid` | cluster 内的 CTA ID |
| `%cluster_nctaid` | cluster 的维度 |
| `%cluster_ctarank` | CTA 在 cluster 中的**线性化 rank** |
| `%cluster_nctarank` | cluster 中 CTA 的总数 |
| `%is_explicit_cluster` | 区分显式 vs 隐式（1×1×1）cluster 启动 |

---

## 4. PTX 与内联 PTX

### 4.1 什么是 PTX

**PTX 是 NVIDIA GPU 的汇编语言，是 CUDA 生态里的指令集架构（ISA）。**

编译链：

```
CUDA C++ 源码 → 编译 → PTX → JIT 编译 → SASS → 汇编成二进制
```

**为什么要学 PTX**：它能直接访问 **CUDA C/C++ 未直接暴露的底层 GPU 特性**，用于手工优化性能关键段。
很多知名仓库（CUTLASS 等）大量使用 PTX，学会它就能读懂高度优化库里的各种优化技巧。

### 4.2 内联 PTX（Inline PTX）

**内联 PTX = 用 `asm` 关键字把 PTX 汇编直接嵌入 CUDA C/C++。**

- 好处：省去**从零手写整个 PTX 文件**的重活，直接在 C++ 里写 PTX 指令。
- 常配合**模板**：把**编译期常量参数化后直接嵌入 GPU 指令**。

### 4.3 PTX 指令的格式

```cuda
asm("ptx 指令字符串" : 输出操作数 : 输入操作数 : clobbers);
```

- `asm()`：把 PTX 代码插入 CUDA 程序。
- **`volatile` 关键字**（`asm volatile(...)`）：防止编译器**删除或移动**你的 PTX 指令。
- 若不想让编译器看到你的内存访问方式，在 clobbers 里用 **`"memory"`**。

### 4.4 输入与输出操作数

操作数是 **C++ 变量与汇编指令之间的桥梁**，用**约束（constraint）语法**告诉编译器如何把变量映射到寄存器或内存。

- 顺序：**先输出操作数 → 再输入操作数 → 最后 clobbers**。
- 操作数按**输出在前、输入在后**统一顺序编号。
- 无输出时可留空：`asm("string" :: 输入操作数)`。

### 4.5 约束修饰符（Constraint Modifiers）

| 修饰符 | 含义 |
|--------|------|
| `=` | 只写（write-only） |
| `+` | 读写（read-write） |
| `&` | **Early clobber**：防止编译器把输出与后续输入用同一寄存器；当输出在"所有输入被消费完之前"就写入时**至关重要** |

### 4.6 约束（Constraints）

| 约束 | 含义 |
|------|------|
| `h` | 16 位无符号整数 |
| `r` | 32 位无符号整数（32 位地址/无符号整数） |
| `l` | 64 位无符号整数（64 位地址） |
| `f` | 32 位浮点数 |
| `d` | 64 位浮点数 |
| `n` | 立即数整数（编译期常量） |

### 4.7 操作数的表示

- 操作数按**文本顺序**用 `%0, %1, %2` 表示（`%0` = 指令字符串后第一个变量）。
- 用约束修饰符 + 约束表示变量：输出 `"+r"`、`"=f"`、`"+l"`，输入 `"r"`、`"l"`、`"f"`。
- **花括号 `{}` 用于局部寄存器作用域**。

### 4.8 指令字符串 / 输出 / 输入 / clobbers

- **指令字符串**：真正要在 GPU 上执行的指令；可单行或多行（用换行符）；用 `%0/%1` 占位。
  ```cuda
  asm volatile("wgmma.mma_async {%0, %1, %2, ...}");
  ```
- **输出操作数**：在指令字符串之后；多个用逗号分隔；无输出则留空（`::`）。
- **输入操作数**：在输出之后、以冒号分隔；只读（`"n"`/`"r"`/`"f"`/`"l"`）或读写（`"+n"`/`"+r"`/`"+f"`），**不能只写**；`"+n"` 可拿到编译期常量。
- **Clobbers**：告诉编译器"除了输出操作数之外，还有哪些资源可能被修改"，
  防止编译器做错误假设而改坏代码。

---

## 5. PTX 状态空间（State Spaces）

### 5.1 什么是状态空间

**状态空间 = GPU 线程可访问的不同内存区域，各自针对特定用途优化。**

- 写 PTX 时必须指定指令作用于哪个状态空间。
- 它们不只是底层 PTX 细节——正是因为"内存放置是一等公民"，才能写出高性能代码。

### 5.2 各状态空间

| 状态空间 | 含义 | 说明 |
|---------|------|------|
| `.reg` | 寄存器内存 | 需要临时寄存器（输入/输出约束系统覆盖不到）时显式声明 `.reg`；寄存器超限会自动 spill 到 local memory |
| `.global` | 全局内存 | 用全局内存数据的各种操作 |
| `.local` | 每线程私有内存 | 存寄存器放不下的数据；**很少用** |
| `.param` | 参数空间 | 双用途：kernel 入参（只读、per-grid）+ 设备函数参数（读写、per-thread）；不走通用寄存器文件；**也不常用** |
| `.shared` | 共享内存 | 通过子限定符 `::cta` 或 `::cluster` 可被同一 cluster 内其它 CTA 访问 |

---

## 6. 数据类型（Data Types）

### 6.1 基础类型

| 类别 | 类型 |
|------|------|
| 无符号整数 | `.u8, .u16, .u32, .u64` |
| 有符号整数 | `.s8, .s16, .s32, .s64` |
| 浮点 | `.f16, .f32, .f64` |
| 原始位模式 | `.b8, .b16, .b32, .b64, .b128` |
| 谓词 | 存 TRUE/FALSE |

### 6.2 深度学习中重要的数据类型

| 类型 | 存储 | 说明 |
|------|------|------|
| **BF16** | `.b16` | 8 位指数 + 7 位尾数（共 16 位） |
| **e4m3 / e5m2（FP8）** | `.b8` | 8 位浮点：4 位指数+3 位尾数，或 5 位指数+2 位尾数 |
| **TF32** | `.b32` | 范围与 f32 相同、精度降低（尾数 ≥10 位）的 32 位格式 |
| **e2m1（4-bit Float）** | — | 超紧凑：2 位指数 + 1 位尾数（共 4 位） |

### 6.3 打包数据类型（Packed Types）

把多个值打包进一个寄存器以并行操作：

| 类型 | 含义 |
|------|------|
| `.u16x2, .s16x2` | 一个 32 位寄存器装 2 个 16 位整数 |
| `.f16x2` | 2 个 `.f16` |
| `.bf16x2` | 2 个 BF16 |
| `.e4m3x2` | 2 个 e4m3 |
| `.e5m2x2` | 2 个 e5m2 |

> 这正是 `H100-架构介绍.md` 里"32 位是寄存器基本单位，FP16/FP8 需打包"的 PTX 层面体现。

---

## 7. 内存地址与 Shared Memory Bank

### 7.1 内存地址的本质

- 把 GPU 全部内存想象成**一把无限长的尺子**；可寻址的最小单位是**字节（8 位）**，也叫 **atom**。
- **地址 = 距尺子起点的字节距离**；字节地址就是尺子上的原始整数索引。
- CUDA 里很少直接操作原始字节，而是操作类型（float/int/bf16）。写 `ptr + 1` 时，编译器移动的是
  **1 个元素**，而不是 1 个字节。

### 7.2 Shared Memory 与 Bank

为了让 warp 的 32 个线程同时访问内存，shared memory 被分成 **32 个 bank**，编号 0–31，每个 bank **4 字节宽**。

- GPU 根据 **4 字节字索引**确定地址属于哪个 bank：
  - 字节 0–3 → Bank 0；字节 4–7 → Bank 1；字节 8–11 → Bank 2；字节 12–15 → Bank 3，以此类推。

### 7.3 Bank Conflict

- 硬件**每个 bank 每 cycle 只能服务 1 个唯一地址**。
  - 2 个线程冲突 → 访问串行化（2 倍慢）；**最坏 32 个线程冲突 → 32 倍慢**。
- 多个线程读**完全相同的地址**时**无冲突**——硬件做 multicast/broadcast，1 个 cycle 服务所有线程。
- **Bank Conflict = 同一 warp 内多个线程访问"映射到同一 bank 的不同地址"**。
- H100 内存控制器按 **128 字节事务**处理请求。
- 若 warp 请求总量超过 128 字节（如每线程加载 16 字节 `float4`），请求会被拆成**多个事务（wave）**。

### 7.4 Swizzling（打散）的由来

- H100 依赖 Tensor Core（读取 **64×64 tile**），**标准线性寻址常造成大规模 bank conflict（跨步访问）**。
- 例：一个 64×64 的 bf16 tile：
  - 行大小 = 64 元素 × 2 字节 = 128 字节；
  - 每行 bank 容量 = 32 bank × 4 字节 = 128 字节。
  - → 矩阵的**每一列都完美地落在同一个 bank**（`(128/4) mod 32 = 0`）。
- **我们希望 `Matrix[0][0]` 与 `Matrix[1][0]` 落在不同 bank**，即使它们的步长恰好是 bank 宽度的整数倍——
  这就是 **swizzle（地址打散）** 的动机，后续 WGMMA 专题会用到。

---

## 8. 指针与状态空间

### 8.1 UVA（统一虚拟寻址）地址映射

UVA 把各状态空间映射到**互不重叠**的区域：

| 地址范围 | 状态空间 |
|---------|---------|
| `0x0000_0000_0000_0000 → 0x0000_FFFF_FFFF_FFFF` | Reserved（保留） |
| `0x0001_... → 0x0001_FFFF_...` | Global memory（全局内存） |
| `0x0002_... → 0x0002_FFFF_...` | Local memory（每上下文） |
| `0x0003_... → 0x0003_FFFF_...` | Shared memory（每 block） |
| `0x0004_... → 0x0004_FFFF_...` | Constant memory（常量内存） |

### 8.2 通用指针（Generic Pointers）

- 现代 CUDA（Compute Capability ≥ 3.5）里，所有指针默认是**通用指针**——即统一的 64 位虚拟地址空间，横跨所有状态空间。
- **没有运行时检查**确定当前处于哪个状态空间。
- 可以像 CPU 指针一样做普通算术。
- 解引用时，硬件**检查高位**把请求路由到正确内存子系统，代价是**损失几个周期**做空间检测。

### 8.3 状态空间指针（State Space Pointers）

- 写内联 PTX（或编译器生成 PTX）时，指针可用其状态空间限定。
- **不是普通指针**，只是对特定硬件指令有意义的原始位。
- **不能在 C++ 中解引用**（`*(float*)shared_bits` 会崩溃或得到垃圾值）。
- **无运行时空间检测**——硬件立刻知道它是 shared memory。
- 支持需要**显式空间限定**的专用指令。

### 8.4 `cvta` 与 `mapa`

- **`cvta`（Convert To Address）**：PTX 指令，在"通用地址"与"特定内存空间地址"之间转换指针。
  CUDA C++ 对应 API：
  ```cuda
  __cvta_generic_to_global / __cvta_generic_to_shared
  __cvta_generic_to_local  / __cvta_generic_to_constant
  ```
- **`mapa`**：用于**把当前 CTA 的共享内存地址转换为同一 cluster 内另一个 CTA 的共享内存地址**（即分布式共享内存地址转换）。
  - **关键点：`mapa` 接受的是 `rank`，不是 block ID！**

### 8.5 Rank vs Block ID

| | 含义 | 范围 |
|---|------|------|
| **Block ID（`%ctaid`）** | 在 grid 中的全局位置 | 0 到 N-1（跨所有 cluster） |
| **Rank（`%cluster_ctarank`）** | 在 cluster 内的位置 | 0 到 cluster_size-1 |

### 8.6 什么时候需要这些（cvta / 状态空间指针）

- **Tensor Core 的 `ldmatrix`** 需要显式的 shared memory 地址。
- **TMA** 需要显式空间限定。
- **Thread Block Cluster** 跨 SM 共享数据需要精确的空间控制（用 cluster 专用地址）。
- **MIG（多实例 GPU）** 配置下，显式地址空间转换有助于保证各 GPU 实例之间的内存隔离。

---

## 9. 总结与学习衔接

### 9.1 四个主题速记

| 主题 | 核心要点 |
|------|---------|
| **Thread Block Clusters** | 最多 16 个 block 同调度到相邻 SM；四级层级；主用途是 DSMEM；size=2 常最优 |
| **DSMEM** | 跨 SM 访问共享内存；TMA multicast 绕开 SM-to-SM 瓶颈 |
| **Inline PTX** | `asm` 内嵌 PTX；操作数/约束/clobbers；`volatile` 防优化 |
| **State Spaces** | `.reg/.global/.local/.param/.shared`；UVA 地址映射；generic vs 空间指针；`cvta`/`mapa` |
| **Data Types** | bf16/e4m3/e5m2/tf32/e2m1 + 打包类型；swizzle 打散 bank |

### 9.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| Cluster / DSMEM / multicast | 《9. Multi GPU》《10. Multi GPU Part 2》（跨 GPU 扩展的基础） |
| TMA + cluster | 《4. cuTensorMap》《5. cp.async.bulk》 |
| Inline PTX / `wgmma` / ldmatrix | 《6. WGMMA-1》《7. Wgmma part 2》 |
| 状态空间指针 / mapa | 《4. cuTensorMap》（TMA 描述符需要显式空间地址） |
| swizzle / bank | 《6. WGMMA-1》（Tensor Core 共享内存布局） |

### 9.3 一句话记忆

> **Cluster 让"跨 SM 共享内存"成为可能（软件概念、限 GPC 内、size=2 常用）；**
> **内联 PTX 让你直接指挥底层指令（asm + 约束 + clobbers）；**
> **状态空间与指针（UVA/generic/cvta/mapa）决定了数据放在哪、怎么寻址——这是写高性能 kernel 的底层根基。**

---

> 参考来源：`2. Clusters, Data types, inline PTX, State Spaces.pdf`（Lesson 2，47 页）。

---

# H100 · 异步与屏障（Asynchronicity and barriers）

> 本文档基于课程讲义《3. Asynchronicity and barriers.pdf》（Lesson 3，50 页）整理，
> 系统介绍 H100 的**异步执行模型**与**同步原语（mbarrier）**。
> 前置阅读：`H100-架构介绍.md`（TMA、异步拷贝、Tensor Core）与
> `H100-Clusters-数据类型-内联PTX-状态空间.md`（PTX、状态空间、mapa）。

---

## 目录

1. [同步世界与异步的动机](#1-同步世界与异步的动机)
2. [阻塞 vs 非阻塞操作](#2-阻塞-vs-非阻塞操作)
3. [GPU 中的异步：H100 的真正天赋](#3-gpu-中的异步h100-的真正天赋)
4. [异步操作的三阶段模式](#4-异步操作的三阶段模式)
5. [两大问题与 mbarrier 的引入](#5-两大问题与-mbarrier-的引入)
6. [mbarrier：硬件加速的同步原语](#6-mbarrier硬件加速的同步原语)
7. [Proxy（代理）模型](#7-proxy代理模型)
8. [冒险（Hazard）：RAW 与 WAR](#8-冒险hazardraw-与-war)
9. [fence（栅栏）](#9-fence栅栏)
10. [mbarrier 完整流水线](#10-mbarrier-完整流水线)
11. [phase 与 expected transaction count](#11-phase-与-expected-transaction-count)
12. [mbarrier 初始化与关键注意点](#12-mbarrier-初始化与关键注意点)
13. [同步：try_wait / test_wait / acquire](#13-同步try_wait--test_wait--acquire)
14. [producer / consumer warp 分工](#14-producer--consumer-warp-分工)
15. [mbarrier 复用模式](#15-mbarrier-复用模式)
16. [mbarrier 不止用于异步拷贝（WAR 防护）](#16-mbarrier-不止用于异步拷贝war-防护)
17. [arrive 的变体与 release 语义](#17-arrive-的变体与-release-语义)
18. [barrier.cluster（跨块屏障）](#18-barriercluster跨块屏障)
19. [异步组（bulk_group / commit_group / wait_group）](#19-异步组bulk_group--commit_group--wait_group)
20. [namedbarriers（命名屏障）](#20-namedbarriers命名屏障)
21. [总结与学习衔接](#21-总结与学习衔接)

---

## 1. 同步世界与异步的动机

任何系统都有两类主要操作：

- **Doing（做）**：处理信息、解方程、做饭……
- **Fetching（取）**：取数据、读题、备料……

**同步世界**里二者串行：你被"阻塞"直到当前任务完成——做饭时不做别的，做完才做下一件。

**异步（Asynchronicity）把"请求（Request）"与"结果（Result）"解耦**：
做饭时去做别的事，过一会儿收到信号再回来用做好的饭。→ 在"做一件事的时间里完成了两件事"，
这就是**延迟隐藏（Latency Hiding）**。

> **本质**：如果"取数据"的代价高于"处理数据"的时间，就必须让**取下一项**与**处理当前项**重叠。
> 延迟隐藏的核心是：**不被某条指令阻塞，能把当前任务放到后台、去干别的活。**

---

## 2. 阻塞 vs 非阻塞操作

| | 阻塞 / 同步 | 非阻塞 / 异步 |
|---|---|---|
| 行为 | 暂停发起线程直到操作**完全完成** | **立即**把控制权还给发起线程（操作尚未完成） |
| 控制权 | 不返回，无法执行下一行 | 立即返回，继续执行 |
| 同步点 | 隐式同步点，保证任务完成后才继续 | 无隐式同步，需后续显式等待 |

---

## 3. GPU 中的异步：H100 的真正天赋

- 现代计算里，"做"（算力）极快，"取"（访存）**慢得痛苦**——取数据的时间远高于计算它的时间。
- 因此高性能架构的目标**不只是让数学更快，而是确保数学永不停止**。
- **H100 常被夸"算力强（FLOPS）"，但它真正的天才在于异步架构**：确保巨大的 Tensor Core **永不等数据**。
- 实现方式：为 **Tensor Core 和 TMA** 使用**非阻塞指令**。

---

## 4. 异步操作的三阶段模式

| 阶段 | 内容 |
|------|------|
| **Stage 1 初始化** | 一个线程发起异步操作，立刻去执行下一条指令 |
| **Stage 2 跟踪与并行执行** | 系统某部分跟踪该操作（TMA 用 **mbarrier**，wgmma 用内部硬件 **scoreboard**）；同时其它单元并行工作 |
| **Stage 3 同步** | 后台操作完成后，进行同步 |

### 4.1 H100 里的同步步骤

- warp 发指令极快（纳秒级）。
- 异步指令到达执行单元后，**立刻尝试读内存**。
- 瓶颈是**内存带宽**：从 HBM 搬到 shared/registers 是**几百纳秒**的高延迟操作（相对逻辑核心）。

### 4.2 由此产生的两大问题

1. **发令者（~ns）比搬数据者（~几百 ns）快** → 若拷贝与计算同时发指令，Tensor Core 要等几百 ns 等数据到达
   → 存在**延迟隐藏的空间**。
2. 大 kernel 里指令队列塞满"数据还没到"的命令时，若无同步，Tensor Core 可能在**数据到达前就执行**了操作。

> 所以我们需要：**① 保证对的数据上做对的操作；② 一个能实现延迟隐藏的系统。** 答案就是 mbarrier。

---

## 5. 两大问题与 mbarrier 的引入

mbarrier 解决了上面的两个问题：

1. 让**慢操作独立于快操作发起**——等快单元启动时数据已就位。
2. **防止快执行单元在错误的数据上操作**。

---

## 6. mbarrier：硬件加速的同步原语

**mbarrier 是驻留在 Shared Memory 中的、硬件加速的同步原语，用于跟踪异步内存事务的完成。**

- 与传统 barrier（等"线程"到达）不同，mbarrier 是**等"数据"到达**。
- 它把 **Producer（发起拷贝的一方）** 与 **Consumer（等数据的一方）** 解耦，
  实现**分阶段（split-phase）、"发完即忘（fire-and-forget）"的内存流水线**。
- 用法：Producer 在 barrier 里**设定期望的传输大小**，硬件在后台干活，事务完成后 barrier 就打开。

> 一句话：**mbarrier 让"快的发令者"尊重"慢的物理硬件"**——用硬件加速的计数来协调"数据到了没"。

---

## 7. Proxy（代理）模型

在 H100 / CUDA PTX 内存模型里，**Proxy 用于区分"谁在执行内存操作"**：

- 两个内存操作发生在**同一个 proxy**（如 Generic Proxy）→ 硬件保证它们（大体）**安全且有序**。
- 操作 A 在 Proxy 1、操作 B 在 Proxy 2 → 硬件**停止检查**，认为二者完全无关，让它们**乱序、并行地跑**。

### 7.1 Generic Proxy vs Async Proxy

| | Generic Proxy（通用代理） | Async Proxy（异步代理） |
|---|---|---|
| 谁 | 写 kernel 的**线程**执行顺序 load/store | **硬件机制**做 bulk 异步拷贝 / Tensor Core 运算 |
| 举例 | 普通线程读写 | `cp.async.bulk`、`wgmma` |
| 关系 | 线程只是"踢一脚"命令就继续 | 完全独立运行，不知道线程在干嘛，线程也不知道它何时完成 |

### 7.2 两条不同的内存通路（关键！）

- **Generic Proxy 走 SM 的 L1 cache**。
- **Async Proxy 绕过 L1，直接与 L2 / HBM 交互**。
- Generic Proxy 的 store 先停留在**本地 store buffer 或 L1**，**不会立即对系统其余部分（含 TMA）可见**。

---

## 8. 冒险（Hazard）：RAW 与 WAR

由于两条 proxy 走不同通路，产生两类经典数据冒险：

### 8.1 RAW（Read After Write，读后写冒险）

- 线程（Generic Proxy）写了地址 X → 立刻让 Async Proxy 去读同一地址 X。
- 但新数据还卡在 Generic Proxy 的 L1 里 → Async Proxy 会从 L2/DRAM 读到**旧数据（stale）**。

### 8.2 WAR（Write After Read，写后读冒险）

- 线程（或 CTA）用 Generic Proxy 读了 X，把旧版本"锚"在 L1。
- 之后 Async Proxy 通过 L2/HBM 更新了 X，但**没有更新/失效 Generic Proxy 的 L1 视图**。
- 两种失败模式：
  1. 之后的 Generic store/writeback/eviction 把 Async Proxy 写的**新值覆盖**（旧行胜出）；
  2. 之后的 Generic load 一直返回**旧值**。

> 结论：跨 proxy 访问同一地址时，**必须用 proxy fence 同步**（见 §9）。

---

## 9. fence（栅栏）

**fence 是显式的"排序/可见性"点**，约束内存效果何时变得可观察——尤其当异步操作不会像你假设的那样
与普通 load/store 自动排序时。

- NVIDIA 明确指出：**跨 proxy 需要 proxy fence 才能正确排序**。
- fence 是**有作用域的**（`.cta` / `.cluster` / `.gpu` / `.sys`），作用域决定"谁必须看到该排序"，
  对应层级中的一致性点（如 L1 vs L2）。
- 两种类型：**普通 fence** 与 **跨 proxy fence**。

### 9.1 Release fence（生产者侧）

单向规则：**所有在 release 之前（程序顺序）的内存操作（尤其是写），对"与之同步的其它线程"而言，
都可见于 release 之后出现的任何操作之前。** 它防止先前的写被延迟/重排到 release 点之后。

### 9.2 Acquire fence（消费者侧）

相反的单向规则：**acquire 之后（程序顺序）的内存操作（尤其是读）不允许被观察到发生在它之前。**
acquire 之后，线程保证能观察到"由匹配的 release 变为可见"的那些写。

### 9.3 Cross-proxy fence

- 跨多个 proxy 访问同一地址时，需要**跨 proxy fence**。
- 对 async proxy 用 **`fence.proxy.async`** 同步 generic 与 async proxy 之间的内存。
- 作用：排空/排序 generic-proxy 的 shared-memory 写可见性，让 async proxy 不读到旧视图。
- **它不是"集体 flush"**，而是**每线程排序**——所以仍需一次 block 同步（如 `__syncthreads()`），
  确保所有写者都做完，再由被选中的线程启动 TMA。

---

## 10. mbarrier 完整流水线

| 步骤 | 操作 |
|------|------|
| Step 1 | 在 shared memory 中初始化 mbarrier，用 `mbarrier.init` 创建带**期望线程到达数**的 barrier |
| Step 2 | 用 `mbarrier.arrive` 为即将发起的异步操作**设定期望事务数（expect_tx）**；每次调用都记录一次指令、递减到达数 |
| Step 3 | 发起异步操作（如 `cp.async.bulk`） |
| Step 4 | 发起其它操作并等待（线程休眠，直到"线程到达数"和"expected_tx"都归零） |
| Step 5 | **翻转 phase**、重置到达数，进行下一次操作 |

---

## 11. phase 与 expected transaction count

- **phase**：barrier 当前可复用的状态/周期。它是**单个 bit**，每完成一个周期就翻转一次。
- **transaction count**：你正在执行的异步操作的大小。
- **expected transaction count**：异步操作"还差多少工作"。
  - 因为是异步操作，硬件会在异步操作推进时**自动递减事务计数**（把 barrier 挂到异步操作上就为这个行为）。
- **复用 barrier**：在操作末尾翻转 phase、并增大事务计数；线程到达数会**按初始化时设定的值自动重置**。

---

## 12. mbarrier 初始化与关键注意点

**`mbarrier.init` 是"几乎所有 bug 的源头"——这里错了，其它都没意义。**

```cuda
mbarrier.init.shared::cta.b64 [addr], count;
```

- `addr`：mbarrier 在状态空间中的内存地址（shared memory）。
- `count`：期望的**线程到达数**——也是复用 barrier（翻转 phase）时 barrier 重置到的值。

### 12.1 必须牢记的点

- 作用域通常用 **`shared::cta`**：只有同一线程块内的线程能"看到"该 barrier。
- 地址是 **shared memory 指针**，用 **`__cvta`** 创建。
- 用 **`shared::cluster`** 可让 barrier 对整个 cluster 的所有线程可见；此时地址需用 **`mapa`** PTX 指令获取。
- **mbarrier 对象必须 64 位对齐**；非对齐访问可能造成**静默数据损坏**。

### 12.2 设置期望事务数（`mbarrier.arrive.expect_tx`）

- 把到达数减 1。
- 把 `tx_count` 字节加到 barrier 的待处理事务数上。
- **累加语义**：若调用两次 `arrive.expect_tx` 各带 4096，则 barrier 总共期望 8192 字节。

### 12.3 占位符 `_`

- 指令里的 `_` 是 `phase_out` 的占位符——本可拿到一个含"当前 phase 与事务数"的 64 位编码 token。
- 它用于某些复杂同步场景；我们**丢弃 phase_out**，因为把信息写进 producer 线程的寄存器里更省
  （**寄存器压力昂贵**，翻转 1 位整数比写 64 位 token 便宜）。

---

## 13. 同步：try_wait / test_wait / acquire

发起异步拷贝后，怎么知道操作完成了？

- **老方法 `barrier.sync`**：阻塞线程直到操作完成——**破坏异步性**。
- **Hopper 的方法**：用一个能返回"完成 true / 未完成 false"的指令 → 实现**延迟隐藏**：
  线程检查 barrier、发现没好就重复，直到操作结束、phase 翻转。这正是 **`mbarrier.try_wait.parity`** 的工作。

### 13.1 `mbarrier.try_wait.parity`

```cuda
mbarrier.try_wait.parity.shared::cta.b64 waitComplete, [addr], phaseParity, suspendTimeHint;
```

- **phaseParity**：传入 0 或 1，代表"必须确认已完成的那个 phase"。
- 硬件把你的 parity 与 barrier 当前内部 parity 比较：
  - **相等** → 该 phase 仍在处理 → 返回 **false（继续等）**；
  - **不等** → barrier 已进入下一 phase → 返回 **true（继续）**。
- 成功返回意味着 barrier parity 已"翻转"，即你请求的 parity 现在指的是**刚完成的前一个 phase**。
- **suspendTimeHint**：可选立即数，告诉调度器"条件不满足时让线程让出多久"。

### 13.2 `mbarrier.test_wait.parity`

- 与 try_wait 类似，用 phase parity 位（0/1）而非原始整数计数跟踪进度。
- `waitComplete`（目标）：1 位谓词寄存器——barrier 完成 → 1，仍在忙 → 0。

### 13.3 `mbarrier.wait` 带 `.acquire`

- 做 try_wait 的一切，但有一个关键区别：**若当前同步轮次已完成，`.acquire` 语义会建立严格的 memory fence**，
  保证 async proxy 或其它线程的**所有数据写**在继续前**一定可见**。
- 例：从 shared memory 加载数据到寄存器（手动喂 Tensor Core）时，`.acquire` 确保这些 LD 指令
  **不会在数据有效前发射**——否则寄存器里是垃圾。
- 即使 `wgmma` 直接读 shared memory，指令本身也需要描述符与内存状态一致；`.acquire` 保证依赖链被遵守。

---

## 14. producer / consumer warp 分工

- **Producer warp**：只有**单个线程**在执行 `mbarrier.init`、`mbarrier.arrive.expect_tx`、`cp.async.bulk` 等指令；
  warp 里其它线程只是闲着。
- **Consumer warp**：执行 **wait** 操作，等 TMA 拷贝完成后才能处理数据。

---

## 15. mbarrier 复用模式

- **phase 声明在寄存器里，不是 shared memory！** 每个线程各自维护一个 `int phase`。
- 循环的最后一步做 **`phase ^= 1`**（翻转）。
- 通过翻转 phase，同一个 mbarrier 对象可被反复复用（下一轮 tile）。

---

## 16. mbarrier 不止用于异步拷贝（WAR 防护）

mbarrier 不只是为 `cp.async.bulk` 服务的。考虑：

```
Producer 写 shared memory → Consumer 用 WGMMA 读 shared memory
→ Producer 想用下一个 tile 覆盖同一块 shared memory
```

- 若 Producer 在 WGMMA 还在读 `sA` 时就覆盖它 → **WAR 冒险，数学结果就是垃圾**。

### 16.1 如何防 WAR

1. 创建带"期望线程到达数"的 barrier。
2. Consumer 忙着读数据，Producer 在 barrier 上**自旋（spin）**。
3. Consumer 执行完最后一条读指令。
4. Consumer 执行递减线程到达数的指令。
5. barrier 状态翻转 → Producer 现在可以写新数据。

> 递减到达数的指令是 **`mbarrier.arrive`**（见 §17）。

---

## 17. arrive 的变体与 release 语义

### 17.1 `mbarrier.arrive`

- 只是把 barrier 的**待处理线程到达数减 count**。

### 17.2 编译器重排序（为什么需要 release）

编译器常为效率重排操作：

```c
a = 1; b = 2; c = a + b;   // 编译器可能先做 b=2 再做 a=1，认为不影响 c
```

- **多线程代码里重排会破坏正确性**：不同线程会观察到不同顺序。
  ```c
  // 线程 a           // 线程 b
  d[0][0] = 42.0f;    while (flag == 0);
  flag = 1;           float x = d[0][0];
  ```
  若 `flag=1` 被重排到 `d[0][0]=42` 之前，线程 b 会读到未初始化的 `d[0][0]`。

### 17.3 `mbarrier.arrive.release`

- **`.release` 后缀提供 release 语义**：单向硬件 fence——**它上方的内存写不能重排到它下方**。
- 用法：线程完成"产出将被其它线程消费的数据"之后，调用 `mbarrier.arrive.release`。

### 17.4 `mbarrier.arrive_drop`（及其后缀）

- 行为类似标准 arrive（递减当前 phase 的待处理到达数）。
- **但更重要**：它**永久递减 barrier 的期望到达数**——若之后翻转 phase 复用 barrier，
  "期望到达数"将是"原值 − 已发生的 drop 数"。
- 后缀组合：
  - `arrive_drop.expect_tx`：设定期望事务数，并**永久 + 临时**地把到达数减 1。
  - `.sem`（sem 可为 `.release` 或 `.relaxed`）：
    - `.relaxed`：不强制任何内存排序；
    - `.release`：保证该线程在到达前做的所有写，对任何等待该 barrier 的线程可见。
  - `.noComplete`：让硬件执行到达（递减 pending/expected 计数）但**不触发 phase 完成**，
    即使"pending 计数归零"的完成条件已满足。

### 17.5 销毁 barrier（`mbarrier.inval`）

- 正式使 mbarrier 对象（64 位同步原语）失效，抹掉硬件对该 barrier 状态的跟踪。
- 释放该 shared memory 地址，使其可被安全覆盖或改作他用。
- 它把通用指针转换成 **32 位 shared memory 偏移**，确保 GPU 硬件寻址到正确的本地内存 bank。

### 17.6 完整流水线回顾

```
1. mbarrier.init          ：在 shared memory 创建 barrier，设期望线程数，起始 phase=0
2. mbarrier.arrive.expect_tx：producer 告诉 barrier 还要期待即将到来的异步操作的字节数
3. cp.async.bulk          ：发起拷贝，然后 mbarrier.try_wait.parity 0 自旋，直到 phase 0 完成
4. Flip                   ：所有期望字节（和线程）到达 → barrier 自动翻到 phase 1
5. 开始计算
6. 计算结束 → 手动翻转 phase → 开始下一个 k tile（复用 barrier）
```

---

## 18. barrier.cluster（跨块屏障）

- 标准 barrier（如 `bar.sync` / `__syncthreads`）只能同步**单 block 内**的线程，无法跨 cluster 协调 producer/consumer 块。
- **`barrier.cluster`** 用于同步**同一 cluster 内共调度的不同线程块**，并保证 barrier 之前对 DSMEM 的写在 barrier 之后对所有块可见。
- **关键用例：TMA Multicast**。初始时 block 0 可能在跑而 block 1 还没跑；
  若 block 0 直接写 block 1 会崩溃。启动 `barrier.cluster` 就能确保每个 block **物理上都已就位**。

### 18.1 两个重要指令

- **`barrier.cluster.arrive`**：线程发出"到达"信号，**不停止**，继续执行不依赖其它块数据的独立指令（数学、本地寄存器操作）。
- **`barrier.cluster.wait`**：**阻塞**指令，线程停在此处，直到 cluster 内其它所有线程/块都发出了 arrive。
  一旦解除阻塞，就保证其它块写的所有数据**现在可安全读取**。

> H100 上，因为 Block A 能写 Block B 的内存，就需要 barrier 防止 Block B 在 A 写完之前读。

---

## 19. 异步组（bulk_group / commit_group / wait_group）

### 19.1 bulk_group

- 把 `bulk_group` 挂到 `cp.async.bulk` 指令上，就能用 `cp.async.bulk.commit_group` 和 `cp.async.bulk.wait_group`。
- 用途：**批量发起一堆指令，再统一等待**。

### 19.2 `cp.async.bulk.commit_group`

- 用 `cp.async.bulk` 发起一批操作（配 bulk_group）。
- 之前没有 commit_group → 这些拷贝是"未提交"的；commit 即"把它们打包成一个组"。
- **打包成组的目的**：之后能让其它单元**等到这 N 个单元执行完**。
- 该指令之后发起的指令属于**下一组**（或只是普通指令）。
- 可以有**多个组**，每组多个 cp.async.bulk 操作。

### 19.3 `cp.async.bulk.wait_group<N>`

- 一批操作已发起并提交后，让要用其输出的其它单元**等待**。
- **`wait_group<N>` 等到"最多还剩 N 个已提交的组处于 pending"**：
  - `wait_group<0>`：等所有组完成；
  - `wait_group<2>`：等到你发起的所有操作里**只剩最近 2 个仍 pending**。
- **计数指"仍 pending 的组"，不是"已完成的组"**。

### 19.4 `cp.async.bulk.wait_group.read`

- **既停顿执行，又强制一个 Acquire Fence**。
- 对 **read-after-write** 场景至关重要（否则可能读到旧数据）。

---

## 20. namedbarriers（命名屏障）

- `__syncthreads()` 是**全 block 的汇合点**。
- **命名屏障**让你在一个 block 内创建**多个独立的同步点**，让**不同 warp 子集**各自协调，不必拖住无关 warp。
- PTX 明确允许不同 warp 用**不同的操作**使用同一个命名屏障，例如混合 `.arrive` 与 `.sync` 来构建 producer/consumer 流水线。

```cuda
bar.sync a, b;   // a = 屏障 ID，b = 参与线程数
```

---

## 21. 总结与学习衔接

### 21.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **异步动机** | 取数据远慢于算数据 → 让"取下一块"与"算当前块"重叠 |
| **Proxy** | Generic（走 L1）vs Async（绕过 L1 直连 L2/HBM），二者互不自动有序 |
| **Hazard** | RAW（async 读到 L1 里的旧数据）、WAR（L1 旧值覆盖/读旧） |
| **fence** | release（生产者）/ acquire（消费者）/ `fence.proxy.async`（跨 proxy） |
| **mbarrier** | 硬件原语，**等数据到达**而非等线程到达 |
| **phase** | 1 bit，每完成一轮翻转；复用即翻转 phase |
| **try_wait.parity** | 非阻塞轮询，返回 true/false，保留延迟隐藏 |
| **wait.acquire** | 完成后强制 acquire fence，保证数据可见 |
| **arrive_drop** | 永久递减期望到达数 |
| **barrier.cluster** | 跨块同步，TMA Multicast 的前置 |
| **commit/wait_group** | 批量异步拷贝的分组与等待（wait_group<0> 全等） |
| **namedbarriers** | block 内多个独立同步点 |

### 21.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| TMA 异步拷贝 + mbarrier | 《4. cuTensorMap》《5. cp.async.bulk》 |
| wgmma scoreboard / 异步组 | 《6. WGMMA-1》《7. Wgmma part 2》 |
| cluster / DSMEM / barrier.cluster | 《2. Clusters…》（已整理）、《9. Multi GPU》 |
| 异步流水线（double buffering） | 《8. Kernel Design》（软件流水 / GEMM 优化） |

### 21.3 一句话记忆

> **H100 的异步 = "发令者（快）"与"搬数者（慢）"解耦；**
> **mbarrier 是协调二者的硬件原语（等数据、翻 phase、可复用、跨 proxy 需 fence）；**
> **把拷贝（cp.async.bulk）与计算（wgmma）都变成后台流水线，让 Tensor Core 永不等数据。**

---

> 参考来源：`3. Asynchronicity and barriers.pdf`（Lesson 3，50 页）。

---

# H100 · cuTensorMap（TMA 描述符）

> 本文档基于课程讲义《4. cuTensorMap.pdf》（Lesson 4 / TMA-1，41 页）整理，
> 系统介绍 **TMA（Tensor Memory Accelerator）** 与其核心抽象 **cuTensorMap（描述符）**。
> 前置阅读：`H100-架构介绍.md`（TMA 概述）、`H100-异步与屏障.md`（mbarrier、异步拷贝）。

---

## 目录

1. [什么是 TMA](#1-什么是-tma)
2. [H100 如何做到"完美异步"](#2-h100-如何做到完美异步)
3. [为什么需要描述符](#3-为什么需要描述符)
4. [TMA 拷贝的三步流程](#4-tma-拷贝的三步流程)
5. [cuTensorMap：描述符的内容](#5-cutensormap描述符的内容)
6. [创建与编码：cuTensorMapEncodeTiled](#6-创建与编码cutensormapencodetiled)
7. [数据类型与 FTZ](#7-数据类型与-ftz)
8. [张量秩、全局地址、维度与步长](#8-张量秩全局地址维度与步长)
9. [boxDim 与 elementStrides](#9-boxdim-与-elementstrides)
10. [Shared Memory、Bank 与 Bank Conflict](#10-shared-memorybank-与-bank-conflict)
11. [Swizzling（地址打散）](#11-swizzling地址打散)
12. [Interleaving（交错布局）](#12-interleaving交错布局)
13. [L2 Promotion（L2 提升）](#13-l2-promotionl2-提升)
14. [OOB Fill（越界填充）](#14-oob-fill越界填充)
15. [总结与学习衔接](#15-总结与学习衔接)

---

## 1. 什么是 TMA

**TMA 是 H100 新增的、用最少的线程参与实现"全异步"数据搬移的单元。**

- **单个线程**发起 TMA 指令后**立即继续执行**，整个操作由硬件在后台完成。
- 不再做地址计算，而是用**描述符（descriptor）**。
- TMA 能**同时把数据搬到多个 SM 的 shared memory**（multicast）。
- 能处理 **1D–5D 张量**。

---

## 2. H100 如何做到"完美异步"

- 目标：**让所有单元始终忙碌**。
- 手段：**多个 buffer 用 TMA 加载，同时 Tensor Core 做运算**（多缓冲流水线）。
- **关键点**：你**不需要大量复杂工程**——**描述符**把"数据、布局、优化"全都封装好，
  把重活交给硬件。

---

## 3. 为什么需要描述符

描述符是 H100 异步能力的"关键拼图"：

| 时代 | 取数方式 | 问题 |
|------|---------|------|
| 早期 | 线程算地址 → 取数 | 每线程都要算地址 |
| Ampere | 拷贝指令非阻塞 | 线程仍需**每 16 字节算一次地址**，SM 仍被拷贝引擎拴住 |
| **Hopper** | **描述符** | 把**整个传输状态**封装进内存里一个 **128 字节**对象，硬件在后台独立完成，SM 不再参与 |

> 因为完成传输所需的全部信息都在那个 128 字节的描述符里，硬件就能在无 SM 参与的情况下后台执行。

---

## 4. TMA 拷贝的三步流程

1. 用 CUDA API 创建 **cuTensorMap**。
2. 用所需信息**编码（encode）**它。
3. **发起操作**（如 `cp.async.bulk.tensor`）。

---

## 5. cuTensorMap：描述符的内容

cuTensorMap 是一个描述符，存储以下信息：

- **内存基指针**（device 地址）
- **张量形状**（每维元素数）
- **步长**（字节）
- **数据类型**
- **对齐与 swizzle（打散）**
- **内存空间**（device / host / unified）
- **阶（order）与秩（rank）**（最多 32 维）
- 可选：**tiling（分块）或 interleaved（交错）布局**

---

## 6. 创建与编码：cuTensorMapEncodeTiled

```cuda
CUtensorMap tma_desc;   // 在 host 内存创建为局部变量
```

- 大小 **128 字节**，**必须 128B 对齐**。
- **不是普通指针**，而是 NVIDIA 定义的特定数据结构。
- 用 **`cuTensorMapEncodeTiled()`** 把值编码进 tensormap。

`cuTensorMapEncodeTiled` 的主要参数（本讲义展开的）：

| 参数 | 含义 |
|------|------|
| `CUtensorMap` | 待填充的描述符对象 |
| `dataType` | 从 HBM 拷贝的数据类型（TMA 用它自动得到内存对齐与传输大小） |
| `tensorRank` | 张量维数 |
| `globalAddress` | HBM 中张量的地址（**须 16 字节对齐**） |
| `globalDim[]` | 每维大小（元素数） |
| `globalStrides[]` | 每维之间的字节步长 |
| `boxDim[]` | 每次拷贝的 tile 大小 |
| `elementStrides[]` | 拷贝时每维跳过的元素数 |
| `swizzle` | 打散模式（NONE/32B/64B/128B） |
| `interleave` | 交错模式（NONE/16B/32B） |
| `l2Promotion` | L2 提升策略 |
| `oobFill` | 越界填充模式 |

---

## 7. 数据类型与 FTZ

`dataType` 决定实际从 HBM 拷贝的数据类型，TMA 引擎据此**自动得到内存对齐与传输大小**。

支持的类型（`CU_TENSOR_MAP_DATA_TYPE_*`）：

| 类型 | 说明 |
|------|------|
| `UINT8` / `UINT16` / `UINT32` / `UINT64` | 无符号整数 |
| `INT32` / `INT64` | 有符号整数 |
| `FLOAT16` / `FLOAT32` / `FLOAT64` | 半/单/双精度浮点 |
| `BFLOAT16` | 16 位 brain float |
| `FLOAT32_FTZ` | 带 flush-to-zero 的 32 位浮点 |
| `TFLOAT32` / `TFLOAT32_FTZ` | TensorFloat-32 格式（可选 FTZ） |

### 7.1 Flush-to-Zero（FTZ）

**FTZ 是一种浮点优化**：把**非规格化数（denormal/subnormal）直接置为 0**，而不是正常计算它们。

---

## 8. 张量秩、全局地址、维度与步长

### 8.1 Tensor Rank 与 Global Address

- **Tensor Rank = 张量的维数**（不是线性代数里的"矩阵秩"）。
- **Global Address = 张量在 HBM 中的地址**；**须 16 字节对齐**以获得高效访存。

### 8.2 Global Dimension（`globalDim`）

- 指定张量**每维的大小（元素数，不是字节）**。
- 对于 r 维数组，数据排列约定：
  - `globalDim[0]` = **最内层维**
  - `globalDim[r-1]` = **最外层维**
- 每个元素范围 0 到 2³²（约 40 亿）。

### 8.3 Global Strides（`globalStrides`）

- 每维之间的**字节步长**：硬件沿某维从一坐标移到下一坐标要跳过多少字节。
- **最内层维的步长是隐式的**（由元素大小决定），因此该数组大小为 **rank − 1**。
- 对应关系：
  - `globalStrides[0]` = 第二内层维（`globalDim[1]`）的字节步长；
  - `globalStrides[rank-2]` = 最外层维（`globalDim[rank-1]`）的字节步长。

---

## 9. boxDim 与 elementStrides

### 9.1 Element Strides（`elementStrides`）

- 做异步拷贝时，**每维想跳过的元素数**。
- 大小 = rank 的数组，要求：
  - **必须非零**；
  - **步长值 ≤ 8**；
  - 数组大小 = rank。
- 以**元素数**计，不是字节。

### 9.2 Box Dim（`boxDim`）

- **tile 大小**：大小 = rank 的数组，每维一个条目。
- 指定每次 TMA 操作从 global memory 搬到 shared memory 的**遍历盒子（traversal box）的大小**。
- 以**元素数**计（对应所传输的数据类型）。
- **最内层维必须 ≤ swizzle 大小**。

---

## 10. Shared Memory、Bank 与 Bank Conflict

- **Shared memory**：挂在 SM 上的片上暂存区；每个 block 分到一块，块内所有线程可读写其中任意地址，程序上是一个平坦地址空间。
- 硬件底层把它分成 **32 个 bank**——每个 bank 是可独立服务的"lane"。为什么是 32？因为一个 warp 有 32 个线程，理想情况是**一线程一 bank**。
- **关键**：bank 不是 32 个手动索引的数组——你仍只算一个地址，**bank 由地址决定**。

### 10.1 Bank Conflict

- 32 个 bank 可**一个 cycle 同时访问**。一个 warp 发一条逻辑 load/store，但 SMEM 子系统可能分**多轮**执行：
  - 第 1 轮：每个 bank 各取一部分请求；第 2 轮：剩余的冲突请求……以此类推。
- 多个线程访问同一 bank → 访问串行化 → bank conflict。常见的**跨步访问**就会造成串行化。
- 冲突程度：1 请求 = 无冲突，2 = 2-way，4 = 4-way，8 = 8-way，32 = 最坏。

---

## 11. Swizzling（地址打散）

### 11.1 原理

**Swizzling 是地址映射函数**：把逻辑地址（程序以为写到的 GmemAddress）"打散"位，
生成物理地址（数据真正落地的 SmemAddress）。

目的：把**顺序访问模式分散到所有 bank**，避免冲突。

三种模式：**32B、64B、128B**——它们定义打散的**跨度/chunk 大小**，从而决定内存事务大小。
**选对布局是按应用访存模式做性能调优的关键一步。**

### 11.2 Bank 方程

```
bank = (a' / 4) % 32    其中 a' 是硬件实际用的 shared-memory 字节地址
```

- 即 **bank ID 来自地址位 `a'[6:2]`**；每 `32 × 4 = 128 字节`重复一次。

### 11.3 Swizzle 公式

```
a' = a ^ ((a & Y_mask) >> 3)
```

- `a` = 字节地址，`Y_mask = 1 << 7`，低位 4 位 `a[3:0]` 从不改变。
- 三种模式的位变换：
  - **32B**：`a[4]' = a[4] ^ a[7]`
  - **64B**：`a[5:4]' = a[5:4] ^ a[8:7]`
  - **128B**：`a[6:4]' = a[6:4] ^ a[9:7]`

> 被 XOR 的位是"shared memory 中 swizzle 模式行索引的低位"，**本质上不一定是数学矩阵的行**；
> 只有当你的布局把矩阵行映射到那些 shared-memory 模式行时，它们才等于矩阵行位。

### 11.4 Swizzle Span 与 Atom

对 SM90，完整 swizzle 模式的重复周期：

| 模式 | 重复周期 | 说明 |
|------|---------|------|
| 32B | 256 字节 | 每 128B 行拆成 8 个 16B 单元，两两 16B 单元隔行交换；每 2 行重复 |
| 64B | 512 字节 | 4 个 16B 单元一组，`x' = x ^ (y & 3)`；每 4 行重复 |
| 128B | 1024 字节 | 8 个 16B 单元，`a[6:4]' = a[6:4] ^ a[9:7]`；每 8 行重复 |

- 每行 128 字节，拆成 **8 个 16 字节单元**；swizzle 只**置换这 8 个单元**（从一行到下一行）。
- **swizzle row ≠ 你的矩阵行**——它是 128 字节的 shared memory 行。
- swizzle **只以 16B 单元为粒度置换**，单元内部的小段连续数据保持原样，不随机打乱。

### 11.5 三种模式的细节

| 模式 | span | atom | 操作 |
|------|------|------|------|
| **128B** | 128B | 16B | 每行 8 个 16B cell（x=0..7），`a[6:4]' = a[6:4] ^ a[9:7]`，行号 mod 8 选置换，每 8 行重复（8×128B=1024B） |
| **64B** | 64B | 16B | 每行 4 个 16B cell（x=0..3），`x' = x ^ (y & 3)`，左/右各 64B 半边相同置换，每 4 行重复（4×128B=512B） |
| **32B** | 32B | 16B | 每行 2 个 16B cell（x=0..1），`x' = x ^ (y & 1)`，隔行交换两个 16B cell，每 2 行重复（2×128B=256B） |

---

## 12. Interleaving（交错布局）

### 12.1 为什么需要

- 全局内存里的数据不总是线性的。**cuDNN 等库主要为卷积等操作使用 NCHW 内存布局**。
- `CUtensorMapInterleave` 参数告诉 TMA **如何解码地址空间**：把 HBM 中物理上交错排列的数据，
  在 shared memory 里重建成逻辑上**线性**的布局。

### 12.2 两种模式（针对 NC/xHWCx 布局）

| 模式 | 布局 | 数学 | 行为 |
|------|------|------|------|
| `CU_TENSOR_MAP_INTERLEAVE_16B` | NC/8HWC8（8 通道向量） | 8 通道 × 2 字节(FP16) = 16 字节 | 按 16 字节交错块访问 |
| `CU_TENSOR_MAP_INTERLEAVE_32B` | NC/16HWC16（16 通道向量） | 16 通道 × 2 字节 = 32 字节 | 按 32 字节交错块访问 |

> 若通道数不是 slice 的整数倍，**最后一个 slice 需零填充**以维持交错粒度。

### 12.3 Interleaving 与 Swizzling 的耦合

- **二者不独立**。文档规定：
  > 当 `interleave = CU_TENSOR_MAP_INTERLEAVE_32B` 时，`swizzle` 必须设为 `CU_TENSOR_MAP_SWIZZLE_32B`。
- 原因：从 global memory 解交错的 32B 块，**直接喂给 shared memory 的 32B atom swizzle 逻辑**，
  二者步调一致地把复杂全局布局映射成无 bank conflict 的 shared 布局。

### 12.4 关键注意点

- 用 32B interleaving → 全局地址应 **32B 对齐**；用 16B → **16B 对齐**。
- 用 16B interleaving → 全局步长应为 **16 的倍数**；用 32B → **32 的倍数**。
- 用 interleaving 时，**维度必须 ≥ 3**。

---

## 13. L2 Promotion（L2 提升）

- 从 HBM 取数远慢于从 L2 取数。**预取（prefetch）** 就是发非阻塞的 HBM→L2 传输，提前准备未来数据。
- **不做提升（`NONE`）**：内存控制器用默认 cache line 抓取（通常 32 字节）——但 L2 实际按 128 字节行管理数据，故低效。
- 把抓取尺寸提升到 128B 或 256B，一次 TMA 请求就能**一次抓进更大块的连续数据**，最大化带宽效率，
  确保计算单元需要时数据已在 L2。

### 13.1 各档位与硬件含义

| 档位 | 行为 | 适用 |
|------|------|------|
| `L2_PROMOTION_NONE` | 默认 32B sector 抓取 | 稀疏张量/大步长（避免"过度抓取"浪费带宽） |
| `L2_PROMOTION_L2_64B` | 一次抓 2 个相邻 32B sector（共 64B） | 特定 FP16 形状（内层维恰好 64 字节 = 32 元素） |
| `L2_PROMOTION_L2_128B` | 一次抓满 128B cache line，**DRAM 命令开销降 75%**（1 命令而非 4） | **密集 FP16/BF16 GEMM 的标准默认** |
| `L2_PROMOTION_L2_256B` | 一次抓 2 条 cache line（256B） | 高数据密度、需立即消费；**数据不用则最高风险污染缓存** |

### 13.2 经验法则

- **密集 GEMM/Conv**：总是用 `L2_128B` 或 `L2_256B`，最大化总线效率。
- **Embedding 查表 / 稀疏**：用 `NONE`（32B），不抓你不会碰的数据。
- **调优变量**：取决于 `Tensor_Inner_Dim_Bytes`——若内层维 < 64B，`L2_128B` 可能浪费；让提升尺寸匹配你的连续数据宽度。

---

## 14. OOB Fill（越界填充）

- **`oobFill` 处理 tensor 拷贝时的越界情况**：
  - 自动在**目标（shared memory）**把越界区域填成 0 或 NaN，**不是在 HBM 里填**。
  - **源数据在 HBM 中全程不变**。
  - 免去 kernel 里的手工边界检查，提升代码清晰度与性能。
- 对**分块操作**尤其有用（tile 超出张量边缘时）。
- 填充选择取决于需求：**zero 用于 padding，NaN 用于调试**。

```cuda
CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;  // 用 NaN 填充，但 FMA 运算时按 0 处理
```

> 第二条用 NaN 填充，但**在 FMA 运算时它被当作 0**。

---

## 15. 总结与学习衔接

### 15.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **TMA** | 单线程发起、硬件后台搬数、1D–5D 张量、可 multicast |
| **描述符** | 128 字节、128B 对齐的对象，封装传输的全部状态 |
| **cuTensorMapEncodeTiled** | 编码 dtype/rank/地址/维度/步长/boxDim/elementStrides/swizzle/interleave/L2/OOB |
| **boxDim** | tile 大小（元素），内层维 ≤ swizzle 大小 |
| **elementStrides** | 每维跳过元素数，非零且 ≤ 8 |
| **swizzle** | 32B/64B/128B 地址打散，`a' = a ^ ((a & Y_mask)>>3)`，防 bank conflict |
| **interleave** | 16B/32B 交错（NCHW），32B 必须配 32B swizzle |
| **L2 promotion** | NONE/64B/128B/256B；密集 GEMM 用 128B，稀疏用 NONE |
| **OOB fill** | 越界填 0/NaN（在目标，不在源） |

### 15.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| TMA 异步拷贝 | 《5. cp.async.bulk》（下一课，直接用 cuTensorMap 发起拷贝） |
| swizzle / bank | 《6. WGMMA-1》《7. Wgmma part 2》（Tensor Core 共享内存布局） |
| mbarrier 配合 | 《3. Asynchronicity and barriers》（已整理） |
| L2 promotion / 流水线 | 《8. Kernel Design》（GEMM 优化） |

### 15.3 一句话记忆

> **cuTensorMap = 一张"传输说明书"（128 字节描述符）**，把"数据在哪、什么类型、什么形状、怎么打散、
> 怎么交错、L2 怎么预取、越界怎么填"全部编码进去；**TMA 读它、在后台搬数，SM 专注算矩阵。**

---

> 参考来源：`4. cuTensorMap.pdf`（Lesson 4 / TMA-1，41 页）。

---

# H100 · cp.async.bulk（异步批量拷贝指令族）

> 本文档基于课程讲义《5. cp.async.bulk.pdf》（Lesson 5，40 页）整理，
> 系统介绍 Hopper H100 的 **`cp.async.bulk` 指令族**：张量拷贝、原始拷贝、multicast、prefetch、reduce。
> 前置阅读：`H100-架构介绍.md`（TMA）、`H100-cuTensorMap.md`（描述符）、`H100-异步与屏障.md`（mbarrier）。

---

## 目录

1. [cp.async.bulk 概述](#1-cpasyncbulk-概述)
2. [cp.async.bulk vs cp.async（Ampere）](#2-cpasyncbulk-vs-cpasyncampere)
3. [两种布局：tensor vs raw](#3-两种布局tensor-vs-raw)
4. [tensor 布局详解](#4-tensor-布局详解)
5. [完成机制：mbarrier vs bulk_group](#5-完成机制mbarrier-vs-bulk_group)
6. [L2 缓存提示（Cache Hinting）](#6-l2-缓存提示cache-hinting)
7. [结构化 vs 非结构化拷贝](#7-结构化-vs-非结构化拷贝)
8. [Multicast（多播）](#8-multicast多播)
9. [Multicast 与 mbarrier 的正确用法](#9-multicast-与-mbarrier-的正确用法)
10. [动态共享内存布局](#10-动态共享内存布局)
11. [Multicast 完整流程](#11-multicast-完整流程)
12. [结构化拷贝的操作数](#12-结构化拷贝的操作数)
13. [cp.async.bulk.prefetch](#13-cpasyncbulkprefetch)
14. [cp.reduce.async（归约异步）](#14-cpreduceasync归约异步)
15. [总结与学习衔接](#15-总结与学习衔接)

---

## 1. cp.async.bulk 概述

**`cp.async.bulk` 是 Hopper H100 上用于硬件加速、异步批量内存传输的一族 PTX 指令。**

- 这些操作被**卸载到专用硬件（TMA）**，与 SM 的计算流水线**独立执行**——数据传输在后台进行，计算可继续。
- 能高效处理**大块、多维张量传输**：1D–5D 张量、数百到数千字节。
- **需要 barrier 对象做协调**，保证异步传输与计算之间的正确顺序。

---

## 2. cp.async.bulk vs cp.async（Ampere）

| | Ampere 的 `cp.async` | Hopper 的 `cp.async.bulk` |
|---|---|---|
| 硬件 | **Load/Store Unit（LSU）** | **Tensor Memory Accelerator（TMA）** |
| 地址计算 | 线程发出指令后继续，但**每 16 字节**仍要自己算地址、发命令 | **单线程发一条指令**拷贝整个 tile，TMA 在后台处理所有地址计算、循环展开与搬移 |
| 拷贝 4KB tile | warp 里**每个线程都要循环**发多条 `cp.async`，烧寄存器与指令缓存 | **单线程**发起整个 block 的传输，其余 31（或 127）个线程不做拷贝发起相关的事 |
| 完成跟踪 | `cp.async.commit_group` / `wait_group` | 用 **mbarrier**，TMA 硬件随字节到达**自动更新 barrier 的事务计数** |

> 核心升级：**从"每线程每 16 字节算地址" → "单线程一条指令 + 描述符 + TMA 后台搬"**。

---

## 3. 两种布局：tensor vs raw

### 3.1 Tensor 布局（结构化）

当你有 TMA 描述符（Tensor Map）、想拷贝一个特定的多维 tile 时使用：

```
cp.async.bulk.tensor.<ndim>.<dst>.<src>.<barrier_type>{.cache_hint}{.multicast}
    dst_addr, tensor_map, coordinate_array, mbarrier_addr;
```

### 3.2 Raw 布局（线性 / 非结构化）

用于简单、连续的字节拷贝：

```
cp.async.bulk.<dst>.<src>.<barrier_type>{.cache_hint}
    dst_addr, src_addr, size, mbarrier_addr;
```

---

## 4. tensor 布局详解

### 4.1 `.tensor`

- 加 `.tensor` 表示操作**张量感知**——工作在**多维张量数据**上，而非扁平数组。
- 它开启了一堆重要选项，也是**启用 cuTensorMap 的前提**。

### 4.2 `.{1d,2d,3d,4d,5d}`

- 告诉硬件：需要从你这里读**多少个索引（数）**来定位 tile。
- 一旦拿到定位 tile 所需的 n 个索引，就按 cuTensorMap 里的信息抓取 tile。
- `Nd` = 张量有 N 维；**最多 5 维**。

### 4.3 源/目标状态空间

```
cp.async.bulk.tensor.<dim>.<space1>.<space2>
```

- `space1` = **源张量**的状态空间；`space2` = **目标张量**的状态空间。
- 可为：`global`、`shared::cta`、`shared::cluster` 等。

### 4.4 Load Mode（`.tile` / `.im2col`）

- 这个修饰符**至关重要**：告诉 TMA 硬件**如何解释你给的坐标、如何即时变换数据**。
- **`.tile`**：TMA 用 `tensorCoords` 提供的坐标算出基地址，按 tensorMap 中的步长抓取一个**稠密连续的多维盒子（tile）**。
- **`.im2col`**：抓取时**硬件加速完成 im2col 变换**。
  - 过去要写 kernel 把像素从图像布局（N,C,H,W）拷成列布局（矩阵），浪费带宽与寄存器。
  - 现在只需给卷积窗口的左上角坐标，TMA 就抓取滤波所需像素、展开它们、像矩阵的一列那样放进 L2 → 加速卷积。

### 4.5 Completion Mechanism（完成机制）

```
cp.async.bulk.tensor.{}.{}.{}.<completion_mechanism>
```

两种完成机制：

- **`mbarrier::complete_tx::bytes`**：见 §5.1。
- **`.bulk_group`**：见 §5.2。

---

## 5. 完成机制：mbarrier vs bulk_group

### 5.1 `mbarrier::complete_tx::bytes`

- `cp.async.bulk` 调用带上 mbarrier 句柄（通常在 `[mbar]` 参数里）与完成机制说明。
- 硬件跟踪该操作，随数据搬移**自动更新 barrier 的 tx-count**。
- 当 **tx-count 归零**，barrier phase 翻转、等待的线程被释放。
- 操作数需要：**mbarrier 指针 + 拷贝操作的大小**。

### 5.2 `.bulk_group`

- **比 mbarrier 更简单、更轻量的替代方案**。
- 不再跟踪 tx_count，而是：用 `bulk_group` 发起一堆拷贝 → `commit_group` 批量打包 → `wait_group<N>`
  等到"最多 N 个（或更少）最近的 bulk 异步组仍 pending"。

---

## 6. L2 缓存提示（Cache Hinting）

```
cp.async.bulk.tensor.1d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint
```

- 异步拷贝时若让数据**淹没 L2**，会踢掉你反复在用的其它重要缓存数据。为防"一次性数据"挤占 L2，用 L2 hinting。

### 6.1 三种策略

| 策略 | 含义 |
|------|------|
| `evict_first` | 告诉硬件**立即丢弃**该数据，省缓存空间 |
| `evict_last` | 标记数据为**持久（persistent）**，尽量久留 |
| `evict_normal` | 默认行为 |

### 6.2 Loads（global → shared/dsmem）的用法

- 数据按 **write-back 缓存策略**加载并缓存进 L2。
- **复用权重/数据（最佳）**：用 `evict_last`（标记持久，尽量久留）。
- **流式/一次性数据**：用 `evict_first`（用一次就扔）。

### 6.3 Stores（shared → global）的用法

- 结果写回 global 时通常不需要立刻再读。
- **最终输出（最佳）**：用 `evict_first`（写进 HBM、多半不会再在本 SM 读回），最小化缓存污染——数据进 L2（为服务写）但立刻标记为首要淘汰对象，保持缓存干净给输入用。

### 6.4 创建缓存策略描述符（`createpolicy`）

```cuda
createpolicy.fractional.L2::evict_last.b64  policy_reg, 1.0;
createpolicy.fractional.L2::evict_first.b64 policy_reg, 1.0;
```

- 生成一个 **64 位缓存策略描述符**，编码特定访存模式的淘汰优先级。
- `dest`（`policy_reg`）：存放描述符的 64 位寄存器。
- `fraction(1.0)`：策略应用到**多大比例的数据**。

---

## 7. 结构化 vs 非结构化拷贝

| | `cp.async.bulk`（非结构化） | `cp.async.bulk.tensor`（结构化） |
|---|---|---|
| 本质 | 硬件加速的 **memcpy** | 基于**描述符**的智能拷贝 |
| 对数据的理解 | 当作**线性字节流**，不知道是矩阵/3D/tile | **"理解"**维度、步长、边界（靠 `CUtensorMap`） |
| 用法 | `cp.async.bulk.shared` / `cp.async.bulk.global` | `cp.async.bulk.tensor`（cuTensorMap 出场） |

### 7.1 非结构化拷贝的操作数

- **shared**：`dstmem`（shared 目标地址）、`srcMem`（shared 源）、`size`（字节）、`mbar`（mbarrier 指针）、`cache_policy`（64 位策略描述符指针）。
- **global**：`Dst`（global 目标）、`Src`（源）、`Size`、`Cache_policy`、`Mask`（**掩码写**：指定写目标的哪些字节）。
- **dsmem**：`dst`（DSMEM 目标）、`src`（shared 源）、`size`、`completion_mechanism`（mbarrier）。
  > 关键：这里不是搬到"分给其它 block"的 shared memory，而是搬到**所有 block 共享的分布式共享内存（DSMEM）池**。

---

## 8. Multicast（多播）

### 8.1 起源

- **算数据比搬数据快**，尤其从 HBM 搬数据既耗时又耗能。
- 深度学习大量用 GEMM，而 GEMM 里**很多不同 block 都要读同一块 Matrix A**，去和各自的 Matrix B 块相乘。
  例如 SM0、SM1、SM2、SM3 都需要 "Tile A0"。
- 旧架构：4 个 SM 各自从 global memory 请求同一份数据（如权重矩阵）——**非常昂贵**。
- **为什么不从 HBM 只取一次，流经总线时同时拷给 4 个 SM？** → 这就是 multicast。

### 8.2 操作拆解

1. TMA 从 global memory（srcMem）读 `size` 字节进 **L2 cache**；cache hint 让 L2 持久化该行，优化后续 wave 的带宽。
2. L2 控制器**读一次数据，通过 cluster crossbar 广播**，同时写入 mask 里每个 block 的 SMEM bank。
3. 完成时，TMA 用**多播编码的 mbar 指针**，**原子地**给每个参与 block 的 mbarrier 事务计数**加 size 字节**。
4. **单个 leader 线程**发起这条非阻塞指令；TMA 硬件独立管理整个"取数-广播-发信号"流水线，所有 block 的线程可边等边算或休眠。

---

## 9. Multicast 与 mbarrier 的正确用法

这里**很容易踩坑**，要理解 cluster 内特定线程块使用 mbarrier 的规则：

- **mbarrier 对象在每个参与 CTA 的 shared memory 中，以相同的相对偏移被复制**。
- producer 发起 TMA 指令时，硬件把数据广播给 `ctaMask` 里的所有 CTA，并自动给**每个目标 CTA 中那个特定地址的 mbarrier**发信号。
- 硬件从指令的 mask **立刻知道组成员**；接收方 CTA **不需要"arrive"来组队**，只需在本地 mbarrier 实例上等待数据落地。

### 9.1 参与方如何协作

1. cluster 内**所有 block 的第一个线程**调用 `expect_tx`（带上它们期望的数据量），各自指向**自己 shared memory 里的 mbarrier**；`cp.async` 通过**偏移**找到每个 block 的 barrier。
2. **整个 block 的单个线程**调用带 multicast + mask 的 `cp.async.bulk`，指向**自己的 barrier**。
3. 想只传给 block 0、3 而不传 1、2 → **block 1、2 不要调用 mbarrier，也不要放进 mask**。
4. 所有 block 的 consumer warp 都在**本地 barrier** 上 `mbarrier.try_wait` 自旋。
5. **到达数由 TMA 自己管理**（对 mask 内的 block）：TMA 自动向指定偏移的 mbarrier 发"远程到达（remote arrival）"信号。

---

## 10. 动态共享内存布局

用动态共享内存时，**要手动管理布局**：

```cuda
// 不要：extern __shared__ char smem[]; 然后直接往后面塞 mbarrier
// 要：
uint64_t* bar_ptr = reinterpret_cast<uint64_t*>(smem);
int tma_alignment = 128;
int data_offset = (sizeof(uint64_t) + tma_alignment - 1) & ~(tma_alignment - 1);
half* tile_ptr = reinterpret_cast<half*>(smem + data_offset);
```

> 即：先放 mbarrier（64 位），再按 **128 字节对齐**计算数据区偏移，避免 mbarrier 与 tile 数据重叠/错位。

---

## 11. Multicast 完整流程

以 16 个 CTA（在 cluster 内 16 个不同 SM 上）为例：

1. **建立 multicast 组**：每个参与 CTA 在 shared memory 分配**相同的 mbarrier 对象**，调用 `mbarrier.init.shared.b64`（带到达数），然后对该 mbarrier 执行 `arrive()`。
   - 这个 arrive **不只是同步，而是硬件注册**——内存子系统现在知道这 16 个 CTA 组成一个"接收相同数据"的逻辑组。
2. **每个 CTA 创建完全相同的 `CUtensorMap` 描述符**：host 上调用
   `make_tma_copy(SM90_TMA_LOAD_MULTICAST{}, gmem_tensor, smem_layout, cluster_size)`，
   编码张量几何、数据类型、swizzle 模式，以及**关键的 cluster 维度**。
   - 描述符传给 kernel（标 `__grid_constant__`），告诉 TMA 取哪块 global、放到每个 CTA shared memory 哪里。
   - **所有 16 个 CTA 必须用完全相同的描述符**——任何偏差都破坏"要同一份数据"的契约。
3. **每个 CTA（通常由单个被选中的线程）发 TMA multicast 指令**：
   ```
   cp.async.bulk.tensor.shared.cluster.global.mbarrier.multicast
   ```
   - 注意 `.cluster` 作用域与 `.multicast` 限定符——它们向硬件表达意图。
   - 操作数：shared 目标地址、tensorMap 指针、张量坐标、mbarrier 指针，以及**关键的 `ctaMask`**。
   - `ctaMask` 是 **16 位位掩码**（对 size=16 的 cluster），bit i 表示 CTA i 是否参与；`0xFFFF` = 全部 16 个 CTA 接收。
4. **L2 层的魔法**：L2 控制器收到 16 个"看似独立"的、对同一 tile 的请求，都带相同的 mbarrier group ID。硬件识别出该模式，**提升其中一个请求为 leader**。
   - leader 触发**一次 HBM 读**（比如 1MB 权重）；数据流入 L2 时，控制器**不只发给一个 SM，而是多播给全部 16 个 SM**的 L1 并直接进它们的 shared memory。
   - **你付 1MB 的 HBM 带宽，却向 SM 交付了 16MB 的数据**。
5. TMA 硬件随数据到达，**自动递减每个 CTA 的 mbarrier 事务字节数（tx-count）**，跟踪完成进度。
6. 每个 CTA 执行 `wait_barrier(tma_load_mbar, phase)`（或等价的 `mbarrier.try_wait`），阻塞直到 tx-count 归零——即所有期望字节都已送达该 CTA 的 shared memory。
7. 所有 CTA 越过 barrier 后，保证 shared memory 已填好数据，可开始计算。

### 11.1 Multicast 指令操作数

`Dst`（目标地址）、`Src`（global 指针）、`Size`（操作大小）、`Mbar`（mbarrier 指针）、
`Ctamask`（16 位多播掩码）、`Cache-policy`（缓存策略）。

---

## 12. 结构化拷贝的操作数

### 12.1 global → CTA（shared）

`dstMem`（shared 指针）、`tensorMap, tensorCoord`（tensorMap 地址 + box 坐标数组）、
`srcMem`（global 地址指针）、`cache-policy`。

### 12.2 global → DSMEM（多播）

`dstMem`（DSMEM 指针）、`tensorMap, tensorCoord`、`mbar`（mbarrier 指针）、
`ctaMask`（选择要拷贝的 block 的掩码）、`cache-policy`。

### 12.3 TMA Stores（用 bulk_group）

`tensorMap, tensorCoords`（cuTensorMap 的 64 位指针 + box 坐标数组）、
`srcMem`（数据来源）、`Cache-policy`。

---

## 13. cp.async.bulk.prefetch

- 可**预取数据到 L2** 以降低延迟：用 `cp.async.bulk.prefetch.tensor`。
- **几个要点**：
  - `cp.async.bulk.prefetch` 是**性能提示**：若发完 prefetch 立刻发 `cp.async.bulk.tensor` 而数据还没缓存，
    设备**仍会从 HBM 取数据**。
  - 即使 main 拷贝与 prefetch 用**同一个 tensorMap**，**L2 promotion 与 swizzling 对"数据如何缓存在 L2"没有影响**。

---

## 14. cp.reduce.async（归约异步）

- **把"整个数据 tile 的原子累加"从 SM 卸载到 TMA**。
- 两个版本：
  ```
  cp.reduce.async.bulk.dst.src.<completion_mechanism>{.level::cache_hint}.<redOp>.<type>
      [dstMem], [srcMem], size{, cache-policy}

  cp.reduce.async.bulk.tensor.dim.dst.src.<redOp>{.load_mode}.<completion_mechanism>{.level::cache_hint}
      [tensorMap, tensorCoords], [srcMem]{, cache-policy}
  ```

### 14.1 允许的 redOp 与数据类型

| redOp | 允许的数据类型 |
|-------|---------------|
| `.add` | `.f16` |
| `.min` / `.max` | `.bf16` |
| `.inc` / `.dec` | `.b32` |
| `.and` | `.u32` |
| `.or` | `.s32` |
| `.xor` | `.b64` |
| | `.u64` / `.s64` / `.f32` / `.f64` |

---

## 15. 总结与学习衔接

### 15.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **cp.async.bulk** | TMA 驱动的异步批量拷贝，1D–5D，需 barrier 协调 |
| **vs Ampere** | 从"每线程每 16B 算地址" → "单线程一条指令 + 描述符" |
| **两种布局** | tensor（描述符）vs raw（线性 memcpy） |
| **load mode** | `.tile`（稠密 box）/ `.im2col`（硬件卷积展开） |
| **完成机制** | `mbarrier::complete_tx::bytes` 或 `bulk_group`（commit/wait_group） |
| **L2 hinting** | `evict_first/last/normal`；复用用 `last`，流式/输出用 `first`；`createpolicy` 造 64 位描述符 |
| **multicast** | 一次 HBM 读、多播给 N 个 SM（付 1MB 带宽交付 N×MB） |
| **prefetch** | 预取到 L2（性能提示） |
| **reduce.async** | 把 tile 的原子归约卸载给 TMA |

### 15.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| cp.async.bulk + mbarrier + 多缓冲流水线 | 《8. Kernel Design》《8.1 Stream-K》（GEMM 软件流水） |
| swizzle / multicast 的共享内存布局 | 《6. WGMMA-1》《7. Wgmma part 2》 |
| multicast / cluster / DSMEM | 《9. Multi GPU》《10. Multi GPU Part 2》 |

### 15.3 一句话记忆

> **`cp.async.bulk` = 让 TMA 在后台搬数的一族指令**：用 cuTensorMap 描述"搬什么、搬到哪、怎么打散"，
> 用 mbarrier/bulk_group 同步"搬完了没"，用 multicast 一份数据喂多个 SM，
> 用 prefetch 提前暖 L2、用 reduce.async 把归约也卸载给 TMA——**把 SM 从"搬数"里彻底解放出来。**

---

> 参考来源：`5. cp.async.bulk.pdf`（Lesson 5，40 页）。

---

# H100 · WGMMA 入门（Warp Group Matrix Multiply Accumulate）

> 本文档基于课程讲义《6. WGMMA-1.pdf》（Lesson 6，51 页）整理，
> 系统介绍 Hopper 的 **WGMMA（Warp Group MMA）**：第四代 Tensor Core 的异步矩阵乘加指令。
> 前置阅读：`H100-架构介绍.md`（Tensor Core）、`H100-异步与屏障.md`（异步）、
> `H100-cuTensorMap.md`（swizzle）、`H100-内联PTX` 相关知识。

---

## 目录

1. [Hopper MMA 的四大创新](#1-hopper-mma-的四大创新)
2. [范式转变：warp → warp group](#2-范式转变warp--warp-group)
3. [WGMMA 流水线](#3-wgmma-流水线)
4. [wgmma.fence.sync.aligned](#4-wgmmafencesyncaligned)
5. [wgmma.mma_async 指令](#5-wgmmamma_async-指令)
6. [形状 m64nXkY：M 为何固定为 64](#6-形状-m64nxky-m-为何固定为-64)
7. [操作数位置与 scale 参数](#7-操作数位置与-scale-参数)
8. [A 的位置：寄存器 vs 共享内存](#8-a-的位置寄存器-vs-共享内存)
9. [ldmatrix：把 A 装进寄存器](#9-ldmatrix把-a-装进寄存器)
10. [16×16 tile 的装载：Split-Warp 策略](#10-16×16-tile-的装载split-warp-策略)
11. [Swizzle 地址计算](#11-swizzle-地址计算)
12. [BF16 打包（Packing）](#12-bf16-打包packing)
13. [WGMMA 描述符（Descriptor）](#13-wgmma-描述符descriptor)
14. [wgmma.mma_async 的结果写回](#14-wgmmamma_async-的结果写回)
15. [总结与学习衔接](#15-总结与学习衔接)

---

## 1. Hopper MMA 的四大创新

除了更快的 Tensor Core，Hopper 的 MMA 还有四个重要创新：

1. **完全异步**：Tensor Core 运算现在是**非阻塞**的——支持**多个在飞（in-flight）操作**，
   让 Tensor Core 活跃更久。
2. **warp → warp group**：从单 warp 转到 **warp group**，可使用**大得多的 tile**。
3. **直接从 shared memory / 寄存器异步取数**：Ampere 里线程要等数据进寄存器才能发 `mma.sync`；
   而 WGMMA **可以跳过加载寄存器，直接发起数学运算**。
4. **FP8 与稀疏的专用硬件支持**。

---

## 2. 范式转变：warp → warp group

- Hopper 把延续十年的"**warp 是执行单元**"改成"**warp group 是执行单元**"。
- 调 `wgmma` 时，你**把 4 个 warp 融合成一个在 Tensor Core 上运算的计算实体**——
  最大化 Tensor Core 利用率，省去流水化多条 `mma.sync` 的麻烦。
- 4 个 warp **一起**发起 wgmma 指令；warp 调度器检查所有 warp 是否在该指令（同一 PC）**汇聚**，
  然后调度器把它们融合、把数学命令派发给 Tensor Core。
- 命令一旦交给 Tensor Core，线程就**继续执行下一条指令**（异步）。

---

## 3. WGMMA 流水线

```
1. 异步加载 tile 数据（必要时预加载下一个 buffer）
     - A 装寄存器 → 用 ldmatrix + 一堆复杂寻址；B 用 wgmma 描述符加载
     - 或（A/B 都放 shared）→ 只为 A、B 各建一个描述符
2. 发起 wgmma 运算
3. 从寄存器收集结果 → 用 wgmma.fence.sync.align 让寄存器写可见 → 复杂地址计算 → 搬到 shared memory
```

---

## 4. wgmma.fence.sync.aligned

- **`wgmma.fence` 是 `wgmma.mma_async` 的"寄存器排序屏障"，不是"完成屏障"**。
- 作用：在后续 `wgmma.mma_async` 复用同样寄存器之前，让**先前的 warpgroup 寄存器访问以正确顺序可见**。
- 两个主要场景需要它：
  1. warpgroup 中**第一次** `wgmma.mma_async` 之前；
  2. 任何时候某线程访问过"稍后的 `wgmma.mma_async` 将复用为累加器或 A 片段输入"的寄存器。
- **它不负责**排序 `wgmma.mma_async` 消费的矩阵描述符/数据的 shared-memory 写；
  那需要 **async proxy fence（`fence.proxy.async`）** 来排序"先前的 shared 写"与"后续 wgmma 读"。

---

## 5. wgmma.mma_async 指令

核心异步指令，让 Tensor Core 做矩阵乘加：

- **A 直接从 shared memory / 寄存器读**；**B 直接从 shared memory 读**；**C、D 在寄存器里**。

```cuda
wgmma.mma_async.sync.aligned.<shape>.<dtypeD>.<dtypeA>.<dtypeB>
    d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b;
```

### 5.1 `.sync` 与 `.aligned`

- **`.sync`**：SM 级屏障，要求 warpgroup 内所有参与线程（通常 128 线程 = 4 warp）
  **都到达该指令**后，任何一个才能继续。
- **`.aligned`**：断言 warpgroup 内所有线程在该指令地址（PC）处**已汇聚（lockstep）**，
  warp group 之间没有线程分歧。

### 5.2 指令限定符

- **Shape**：tile 形状，用 `m64, n{}, k{}` 表示。
- **dtypeD / dtypeA / dtypeB**：D、A、B 张量的数据类型。

---

## 6. 形状 m64nXkY：M 为何固定为 64

对所有 wgmma 指令：

- **M 维固定为 64**。
- **K 维**（内积维）**严格由 A、B 的精度决定**。
- **N 维最灵活**：决定处理 Matrix B（和 C）的多少列。
  - N 必须是底层内存分配块大小的倍数（通常为 8 或 16 的倍数，视类型而定）。
  - **合法 N 值范围：8 到 256**。

### 6.1 固定 M 的原因

- 硬件设计为**每线程 4 个寄存器** → 每个 warpgroup 可持 **128 × 4 = 512 个寄存器**。
- 若用最小 tile `m=64, n=8`，则 `64 × 8 = 512` 个元素，**恰好等于一个 warpgroup 的容量**。
- 硬件把 A 的数据**物理上静态映射**到 Tensor Core 输入 B；M=64 时有**完美的静态映射，运行时零决策逻辑**。
- 若 M 可配置，就需要一个**巨大的 Crossbar Switch（复杂多路复用器）**在寄存器与数学单元之间
  按请求动态重路由——**昂贵**。

---

## 7. 操作数位置与 scale 参数

### 7.1 操作数位置

| 矩阵 | 位置 |
|------|------|
| A | 寄存器 **或** shared memory |
| B | **只能** shared memory |
| C/D | **只能** 寄存器 |

### 7.2 `scale_d`

- 指定是**累加**（`D = A × B + C`）还是**覆盖**（`D = A × B`）：
  - `scale_d = 1` → 累加；
  - `scale_d = 0` → 覆盖。

### 7.3 `scale_a` / `scale_b`

- A / B 元素的缩放因子：**+1 或 −1**。

### 7.4 寄存器操作数 `d`

- 输出存放的寄存器；在 PTX 指令串里是**第一个操作数**，必须声明为花括号 `{}` 包裹的**向量（元组）**。
- 输出操作数个数 = **输出总元素数 ÷ 总线程数**。

---

## 8. A 的位置：寄存器 vs 共享内存

A 的 tile 可放 shared memory 或寄存器，因此告诉 wgmma A 在哪有两种方式：

- **寄存器**：把寄存器直接传给 wgmma 指令。
- **共享内存**：传 **wgmma 描述符**。

### 8.1 A 放寄存器（何时用）

- **复用数据**时是好选择：从 shared 同时读 A 和 B 会**加倍 bank 压力**、多次写 shared 代价高，不如从寄存器读。
- **从 shared 搬数贵，从寄存器搬数便宜**——把可复用数据放在能更快访问的地方。
- 寄存器数据来自 shared memory：**`ldmatrix`** 取未 swizzle 的地址，把数据装进寄存器。
- **不复用数据时**，把 A 放寄存器没意义（有指令开销 + 寄存器压力）。

---

## 9. ldmatrix：把 A 装进寄存器

**`ldmatrix` 是 WGMMA 流水线里把 Matrix A 数据装进寄存器的首要机制。**

- 与标准 `ld.shared`（加载线性数据）不同，`ldmatrix` 以**不透明的寄存器模式**加载数据，
  物理上与 Tensor Core 的输入 lane 对齐。
- **提供指针的线程，有时不是最终拿到数据的线程**。
- `.m8n8`：从 shared memory 装进 warp 寄存器的矩阵 tile 的几何形状。
- `.x1` / `.x2` / `.x4`：**向量化宽度**，以及每条指令加载的矩阵片段数。

### 9.1 `.x{1,2,4}` 的含义

- 指一次能把多少个 **8×8 核心矩阵**搬进寄存器：
  - **`.x4`**：每线程 4 个寄存器 = 4 个 8×8 核心矩阵。**最常用**——搬 `16×16×4 = 1024 字节`，
    **打满寄存器带宽**。
  - **`.x2`**：每线程 2 个寄存器 = 512 字节，用于较小 tile。
  - **`.x1`**：每线程 1 个寄存器，主要用于**边界处理**等。

### 9.2 `.sync`、`.aligned`、`.trans`

- **`.sync`**：迷你屏障——硬件要保证 T16 准备好让 T0 请求的数据覆盖其寄存器；
  若 T16 还在忙别的，硬件写者会破坏 T16 的状态。
- **`.aligned`**：所有线程必须一起执行它，硬件才知道 warp 已就绪。
- **`.trans`**：加载时是否**转置**矩阵。因为线程的寄存器对他人不可见，这整条指令由硬件内部完成。

---

## 10. 16×16 tile 的装载：Split-Warp 策略

整个操作分两个阶段：

- **Address Phase（谁提供指针？= Gather）**
- **Register Phase（谁持有结果？= Destination）**

`.x4` 下我们处理的是 **4 个 8×8 子块**。

### 10.1 Split-Warp（拆 warp）

- 地址**不是线性算的**。因为 `ldmatrix` 按 **8 列块**加载，把 warp 拆成两条"竖直条带"：
  - **左半（列 0–7）**：由前 16 个线程（lane 0–15）控制。
  - **右半（列 8–15）**：由后 16 个线程（lane 16–31）控制。

### 10.2 Phase 1：地址责任（谁指）

- 硬件看每个线程的地址寄存器，决定从 shared memory 哪里读。warp 被拆成 **4 组 × 8 线程**。

### 10.3 Phase 2：寄存器责任（谁持有）

- 数据取到后，`ldmatrix` 把它**按行条带化**分配到 warp 里（每组 4 线程），准备好喂 Tensor Core。
- 对任意 8×8 矩阵（M0/M1/M2/M3），行分布：
  - 第 0 行 → 线程 0–3；第 1 行 → 线程 4–7；第 2 行 → 线程 8–11；……；第 7 行 → 线程 28–31。

---

## 11. Swizzle 地址计算

tile 在 shared memory 里是 **swizzled 布局**（为避 bank conflict），所以"逻辑行"的字节在物理上不连续。

- 每个线程从逻辑坐标出发（"我要第 r 行、第 c 列块"），用 swizzle 映射（常为 XOR 重映射）算物理 shared 地址：
  ```
  smem_addr = base + (row_offset ^ swizzle_mask) + col_offset
  ```
- 因为指针经 swizzle 映射，访问落到**不同 bank**，warp 命中**极少/零 bank conflict**。
- `ldmatrix` 每线程加载 **16 字节**，并重排成分片布局。

### 11.1 地址位拆解

- **位 [0–1]**：4 字节字内的字节偏移（与 bank 选择无关）。
- **位 [2–6]**：**Bank Index**——这 5 位决定数据落到 32 个 bank（0–31）中的哪一个。
  - 其中 **位 [4–6] 控制 bank index 的高 3 位**。
- **位 [7–9]**：**Row Index**——因 pitch 是 128 字节（2⁷），**位 7 是换行时第一个变化的位**。
- 目标：让 **Bank（位 4–6）随 Row（位 7–9）变化**。

### 11.2 Swizzle 地址公式

```cuda
Physical_Address = (Base_Address + Linear_Offset) ^ Swizzle_Mask;

// 128 字节 = 2^7，取位 [7,8,9] 移到位置 [4,5,6]
uint32_t mask = ((linear_offset >> 7) & 0x7) << 4;
uint32_t smem_ptr = (base_ptr + linear_offset) ^ mask;
```

---

## 12. BF16 打包（Packing）

- NVIDIA GPU **没有 16 位寄存器**，只有 32 位寄存器。所以 bf16 数据在寄存器里**必须成对打包**。
- **容器**：一个 `.b32`（32 位）寄存器装两个 bf16。
- **布局**：
  - 位 0–15（LO）：元素 N（偶数下标）；
  - 位 16–31（HI）：元素 N+1（奇数下标）。
- **搬数据**时用 `.b32` 类型指令；**计算**时用 `.bf16x2`。

### 12.1 ldmatrix 的打包

- 用 `ldmatrix.sync.aligned.m8n8.x1.b16`：每线程收 1 个 32 位寄存器（含一对打包 bf16）。
- 用 `.x2` / `.x4`：得 2 或 4 个寄存器，每个含一对打包 bf16。
- **shared memory 地址须 16 字节（128 位）对齐**。
- `.trans`：加载时把行主序数据**直接转置成列主序**（或反之），无需手动 shuffle。

---

## 13. WGMMA 描述符（Descriptor）

**64 位描述符**，包含数据存储信息、步长与 swizzle 布局。它打包 **5 个字段**：
地址、LBO、SBO、Matrix base offset、swizzle 布局。

### 13.1 蓝图

- 一个 64 位寄存器编码：基地址、K 步长、M/N 步长、矩阵基偏移、swizzle 模式。
- **所有步长都预先除以 16**，以塞进 14 位字段（低 4 位没用，因为 shared 指针 16 字节对齐）；
  硬件运行时再乘 16，保证 16 字节对齐、降低地址延迟。
- **基地址必须 16 字节对齐**（否则预处理步长无法被硬件正确解释）。

### 13.2 各字段

| 字段 | 位 | 含义 |
|------|----|------|
| **Base address** | 前 14 位 | 要加载的 tile 的 shared memory 地址（从地址取 14 位） |
| **LBO（Leading Byte Offset）** | 16–29 | 沿"leading"方向相邻两个 core-matrix 列的字节距离；编码 K 步长（÷16、掩 14 位、左移 16） |
| **SBO（Stride Byte Offset）** | 32–45 | 另一个方向跳"一个 core-matrix 块"的 SMEM 字节距离 |
| **Matrix base offset** | — | swizzle 模式每 128B 重复；起始不在边界时告诉硬件从哪个 128B 块开始 |
| **Swizzle mode** | 61–63 | `00`=无 swizzle，`01`=128B，`10`=64B，`11`=32B |

### 13.3 LBO / SBO 细节

- **LBO**：
  - K-major swizzled 布局**不用 LBO**（硬件假设为 1）。
  - MN-major 时，LBO = 从"前 (swizzle-byte-size/16) 行"到"下一 (swizzle-byte-size/16) 行"的偏移；
    对 128B swizzle，`128/16 = 8`，即"跳 8 行"。
- **SBO**：
  - K-major：从"前 8 行"到"下一 8 行"的偏移；
  - MN-major：从"前 8 列"到"下一 8 列"的偏移。

### 13.4 设置 swizzle 的重要性

- 若 shared memory 数据被 swizzle，**必须正确设置 swizzle 布局**，否则 wgmma 单元会把打散的数据**线性地读错**。

---

## 14. wgmma.mma_async 的结果写回

- `wgmma.mma_async` **异步**把数据从 shared memory / 寄存器搬到 Tensor Core 做运算。
  - consumer warp group 的**所有线程**都调用它。
- Tensor Core 开始算，结果**按 warpId、laneId 写回 consumer warpgroup 的寄存器**——开销低，硬件把数据"倒"到最近的存储。
- 硬件想把结果写给**物理上最靠近负责该块的数学单元**的线程，所以线程被**交错成小块**，
  每个线程在内存里持有的是**非连续**数据。

### 14.1 每个线程拿到多少寄存器

- 例：`wgmma` 用 `m64n256k16` → 每线程累加器有 **128 个元素**。
- **每一条 WGMMA 指令由 warpgroup 里每个线程都发起！**
- 对 128 线程的 warpgroup：16 位累加器需 **16 条 `stmatrix`（.x4）**，32 位累加器需 **32 条 `stmatrix`**。
- **提示**：像 FlashAttention 3 这类 kernel，若用寄存器装 A tile，要小心 softmax-GEMM 流水线带来的**寄存器压力**。

---

## 15. 总结与学习衔接

### 15.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **四大创新** | 全异步、warp→warp group、直接 shared/reg 取数、FP8+稀疏 |
| **warp group** | 4 warp = 128 线程 = 一个计算实体，M 固定 64 |
| **操作数位置** | A 寄存器/shared，B 只能 shared，C/D 只能寄存器 |
| **ldmatrix** | 把 A 从 shared 装进寄存器，`.x4` 打满带宽，split-warp + swizzle 防冲突 |
| **描述符** | 64 位，含 base/LBO/SBO/base-offset/swizzle，步长÷16 塞 14 位 |
| **swizzle** | `addr = (base + linear) ^ mask`，bank 随 row 变 |
| **packing** | bf16 成对打包进 .b32（LO=偶，HI=奇） |
| **写回** | 异步、按 warpId/laneId 写寄存器、线程交错、stmatrix 搬出 |

### 15.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| WGMMA 高级用法 / 寄存器布局 | 《7. Wgmma part 2》 |
| cp.async.bulk 供数 + wgmma 计算 | 《8. Kernel Design》（GEMM 软件流水） |
| Stream-K / FlashAttention | 《8.1 Stream-K》《8.2 Kernel Launch》 |

### 15.3 一句话记忆

> **WGMMA = 4 个 warp 拧成一个 128 线程的"计算实体"，用一条异步指令让 Tensor Core 直接
> 从 shared/寄存器取 A、从 shared 取 B、把 C 累加在寄存器里；**
> **`ldmatrix` 负责把 A 装进寄存器（swizzle 防 bank conflict），64 位描述符告诉硬件 A/B 在 shared 里怎么摆。**

---

> 参考来源：`6. WGMMA-1.pdf`（Lesson 6，51 页）。

---

# H100 · WGMMA Part 2（分组、stmatrix、FP8、稀疏）

> 本文档基于课程讲义《7. Wgmma part 2.pdf》（Lesson 7，43 页）整理，
> 承接 `H100-WGMMA.md`（Part 1），深入 **WGMMA 的分组同步、结果写出（stmatrix）、FP8 与稀疏**。
> 前置阅读：`H100-WGMMA.md`、`H100-异步与屏障.md`、`H100-cuTensorMap.md`。

---

## 目录

1. [分组：commit_group / wait_group](#1-分组commit_group--wait_group)
2. [Commit/Wait 流水线时间线](#2-commitwait-流水线时间线)
3. [wgmma.fence 的双重角色](#3-wgmmafence-的双重角色)
4. [结果写出：stmatrix](#4-结果写出stmatrix)
5. [stmatrix 的协作寻址](#5-stmatrix-的协作寻址)
6. [stmatrix 与 FP32 的兼容性](#6-stmatrix-与-fp32-的兼容性)
7. [Swizzle 的 atom（16B）](#7-swizzle-的-atom16b)
8. [FP8：为何能翻倍算力](#8-fp8为何能翻倍算力)
9. [e4m3 与 e5m2](#9-e4m3-与-e5m2)
10. [饱和（Saturation）](#10-饱和saturation)
11. [缩放因子（Scaling Factors）](#11-缩放因子scaling-factors)
12. [FP8 打包（x2 / x4）](#12-fp8-打包x2--x4)
13. [FP8 转换与量化](#13-fp8-转换与量化)
14. [FP8 WGMMA 指令](#14-fp8-wgmma-指令)
15. [FP8 的 A 在寄存器、K-Major 规则与精度陷阱](#15-fp8-的-a-在寄存器k-major-规则与精度陷阱)
16. [稀疏 WGMMA（Sparse）](#16-稀疏-wgmmasparse)
17. [sp-sel 与 sp-meta 的配置](#17-sp-sel-与-sp-meta-的配置)
18. [总结与学习衔接](#18-总结与学习衔接)

---

## 1. 分组：commit_group / wait_group

- WGMMA 启动是**异步**的：`wgmma.mma_async` 把工作"放进在飞队列"并立即返回。
- 若**逐条跟踪每个 MMA**，要么付沉重的 scoreboard 开销，要么被迫用"等一切"的过度保守屏障。
- **分组（group）给了我们流水线友好的粒度**：
  - **`wgmma.commit_group`** = "关闭当前批次"（使其可被跟踪）。
  - **`wgmma.wait_group N`** = "仅在在飞批次过多时才停顿"。
- 这带来**重叠**：group g 在算时，你可以为 group g+1 做准备（地址计算、TMA、staging 等）。

### 1.1 `wgmma.commit_group.sync.aligned`

- 只是把**所有尚未提交的 `wgmma.mma_async`** 打成包。
- 硬件能同时跟踪多个组；分组可**降低"逐条跟踪每个矩阵乘"的开销**。
- 通常：发一组算出**一个输出 tile**（如 64×64）的 WGMMA 指令，然后把该 tile 提交为一个组。

### 1.2 `wgmma.wait_group.sync.aligned N`

- 同步点："**暂停本线程，直到只剩 N 个已提交组仍在运行**。"
  - **`wait_group 0`**：等**一切**完成——用于"即将读寄存器里的最终结果并写回 global"之前。
  - **`wait_group N`**：让 N 个组在后台继续跑，同时你去准备下一组数据。

---

## 2. Commit/Wait 流水线时间线

不是"先 commit 再 wait"，而是**流水化**：

```
1. 发一批 wgmma.mma_async（这是 1 个输出 tile）
2. wgmma.commit_group.sync.aligned;
3. 开始准备下一个 tile（TMA、指针计算等）
4. wgmma.wait_group.sync.aligned N;   // 保留 N 个组在跑
5. 排空（drain）
6. wgmma.wait_group.sync.aligned 0;   // 等全部完成
7. 现在安全读累加器 D 并写出
```

---

## 3. wgmma.fence 的双重角色

`wgmma.fence.sync.aligned` 是 WGMMA 流水线的 warpgroup fence，承担两个紧密相关的角色：

1. **WGMMA 发射的"到达/组边界"标记**：WGMMA 操作按组发射，fence 用于标记一个组发射窗口的起点。
2. **操作数的寄存器排序 / 冒险控制**：`wgmma.mma_async` 是异步的、用 warpgroup 级流水线；
   fence 防止涉及**累加器寄存器**与**寄存器驻留的操作数片段**（尤其是寄存器来源的 A）的重排/冒险。

> 注意：它**不是跨 proxy fence**。若你读的是"被 TMA 修改过"的寄存器数据，需要跨 proxy fence：
> `fence.proxy.async`（或等价的 acquire/release 发布协议）。

---

## 4. 结果写出：stmatrix

- WGMMA 指令结束时，结果矩阵 C 在**累加器寄存器**里。结果**不是连续矩阵**（行 0、行 1……），
  而是以**高度不透明的方式分散在 128 个线程的寄存器**里。
- 普通 `st.shared`：谁写数据谁给地址——这需要手工管理地址，不行。
- **`stmatrix`**：像 `ldmatrix` 一样，**"数据提供者"与"地址提供者"解耦**。

### 4.1 stmatrix 指令

- 每个线程输入自己从 wgmma 拿到的寄存器数据，并给出写目标指针。
- `.m8n8`：原子单元仍是 8×8 tile。
- `.x4`：每线程写 4 个寄存器（共 16×16 tile）。
- `.x1` / `.x2`：也支持（更小 tile）。
- `.shared`：目标**永远是 shared memory**。
- `.sync.aligned`：所有 warp 必须同步、每个线程都发起该指令。
- `.m16n8`：仅在 Blackwell 的 `.b8` 类型下有效。
- `.b16`：表示所存元素的数据类型与大小。

---

## 5. stmatrix 的协作寻址

- 像 `ldmatrix` 一样，`stmatrix` 是**协作**的：warp 内每个线程（0–31）提供自己的指针（数据所属的 shared 地址）。
- 因为线程持有的数据是**特定的非线性模式**，它们生成的地址也必须**同样非线性**，才能让数据在 shared memory 里落地成**连续、整洁的矩阵**。
- 执行 `stmatrix` 时，指令把这些元素**按行主序（连续行）从 `[addr]` 开始写入 SMEM**。

### 5.1 m8n8 tile 结构

- 指令操作一个 **8×8 的 16 位元素 tile**。
- **`.x1`**：每线程持有 1 个 8×8 的一部分；**`.x4`**：每线程持有 4 个 8×8。
- 与 `ldmatrix` 一样，**特定线程持有特定行**（线程 0–7 持有第 0 行，等等）。

### 5.2 Phase 1：地址责任（谁指哪）

与 `ldmatrix` 一致（16×16 的 x4 布局，四个 8×8 象限）：

| 线程 | 指向 |
|------|------|
| 0–7 | 左上块（行 0–7，列 0–7） |
| 8–15 | 左下块（行 8–15，列 0–7） |
| 16–23 | 右上块（行 0–7，列 8–15） |
| 24–31 | 右下块（行 8–15，列 8–15） |

### 5.3 寄存器来源（谁持有什么）—— "条纹"布局

- 线程 0 是整个 tile 的"第 0、1 列拥有者"：它看到的**不是连续行，而是竖直切片**。
- 线程 0 的寄存器映射（x4）：
  - `r0`：矩阵 A（左上）→ 行 0，列 [0,1]
  - `r1`：矩阵 B（左下）→ 行 8，列 [0,1]
  - `r2`：矩阵 C（右上）→ 行 0，列 [8,9]
  - `r3`：矩阵 D（右下）→ 行 8，列 [8,9]

### 5.4 序列化循环

- `stmatrix.x4` 每次只消费**每线程 4 个寄存器**，但一个输出 tile 的累加器更大。
- 因此要**对累加器的"切片"做循环**：
  ```
  切片 i → 打包进 {r0,r1,r2,r3} → stmatrix(...)
  切片 i+1 → 打包 → stmatrix(...)
  ```
- 这就是为什么 **epilogue（尾声）通常是一个循环，而不是单次 store**。

---

## 6. stmatrix 与 FP32 的兼容性

**最关键的一点：`stmatrix` 不支持 f32，只支持 `.b16`（打包 16 位）和 `.b8`（打包 8 位）。**
这造成了处理 wgmma 输出的分歧：

- **FP32 累加器（`.f32`）**：累加器在**未打包的 `.f32` 寄存器**里，不能直接喂给 `stmatrix`；
  必须**先降精度并打包**（会损失精度）。
- **FP16 累加器（`.f16`）**：累加器在**打包的 `.b32` 寄存器**（当作 `.f16x2`），
  一个 32 位寄存器装两个元素，**无需转换与打包**。

---

## 7. Swizzle 的 atom（16B）

- 我们 swizzle 的 **"atom" 是 16B**。
- 16B 之所以重要：一个 m8n8 bf16 tile 的一行 = 8 元素 = **16 字节**，所以每行天然是一个 16B 块。
- 技巧在于：**我们 swizzle 的是"行 atom"，不是单个元素**。
- 即使 atom 被置换，**每个 16B atom 内部的数据仍是行主序**。
- 正因如此，**TMA 才能"反 swizzle"回整洁的行主序矩阵**。

---

## 8. FP8：为何能翻倍算力

- FP8 Tensor Core 理论上给出 BF16/FP16 **两倍**的 TFLOPS，H100 上约 **3958 TFLOPS**（含稀疏；稠密约一半）。
- **位宽减半** → 降低 global memory 带宽压力，L2 可缓存 **2 倍**大的模型/batch。
- 收益主要在**矩阵乘密集**的部分（线性层、注意力投影、MLP）；许多流水线仍把 softmax/归一化/规约留在 BF16/FP16。

### 8.1 范式转变

- 约 2022 年前，AI 标准是 FP32 或 FP16/BF16。
- FP8 主要因 **H100（Hopper）** 而工业化，能用**一半内存**把训练提速**两倍**。
- 大模型实例：**DeepSeek V3 用 FP8 训练**，**Kimi K2 用 INT4 量化感知训练（QAT）**。
- 与 INT8 不同，FP8 是"浮点"，能表示宽动态范围。8 位太挤，故拆成两个专用格式平衡"精度 vs 范围"：**e4m3 和 e5m2**。

---

## 9. e4m3 与 e5m2

| | **e4m3** | **e5m2** |
|---|---|---|
| 指数/尾数 | 4 位指数 + 3 位尾数 | 5 位指数 + 2 位尾数 |
| 范围 | ≈ ±448（最大正常值 448，最小正常 0.015625） | ≈ ±57344（远大于 e4m3） |
| 用途 | **权重与激活（前向）**——动态范围好，表示 NaN/Inf | **训练梯度与反向传播**——能表示梯度更新需要的很大数值 |

---

## 10. 饱和（Saturation）

- FP32 动态范围巨大（~10³⁸），几乎不会触顶。
- FP8 e4m3 的"天花板"是 **448**：若矩阵乘结果是 450 而不处理，就**饱和**。把 float 降成 FP8 时，硬件要决定怎么处理 450，通常两种模式（由 intrinsic 控制）：
  1. **钳位到最大可表示值（448）**：数据被裁剪（clipping），但数学"稳定"。
  2. **变 Inf（或 e4m3 里变 NaN，因 e4m3 无 Inf 表示）**：立刻毁掉你的训练。

---

## 11. 缩放因子（Scaling Factors）

既然改不了 8 位的物理，就只能"作弊"——用**缩放因子**。这是 H100 上 FP8 的核心课。

- **Tensor-wise**：整个张量一个缩放因子（取 max 值）——最简单、开销最低。
- **Vector-wise**：每行/每列一个缩放因子。
- **Block-wise（MXFP8 / 微缩放）**：把矩阵切成小 tile（如 16×16 或 32×32），每个 tile 一个 scale。
- 其它：SmoothQuant、Delayed Scaling 等。

---

## 12. FP8 打包（x2 / x4）

- FP8 在传输时**不是独立对象**：H100 内存控制器不会为小于 32 字节（一个 sector）的数据醒来。
  若你要 1 字节（一个 FP8 值），GPU 会取 32 字节、给你那 1 个、丢掉其余 31 个。
- 为消除浪费，用**打包类型 `e4m3x2/4` 与 `e5m2x2/4`**（硬件理解的 PTX 类型）。

### 12.1 x4 Pack（寄存器填充型）

- `.e4m3x4` / `.e5m2x4`：装 **4 个 FP8**。
- 总大小 8 位 × 4 = 32 位 → **恰好填满一个标准 GPU 寄存器**。
- 布局（小端）：
  ```
  [Element 3 | Element 2 | Element 1 | Element 0]
   <-- MSB (31)                    LSB (0) -->
  ```
- 这是**存储与传输的主格式**；搬数据或做简单数学（如找 max）时用 x4。

### 12.2 x2 Pack（数学输入型）

- `.e4m3x2` / `.e5m2x2`：装 **2 个 FP8**，共 16 位，**与 FP16 同大小**。
- H100 被设计成帮人从 FP16 迁到 FP8：有专用逻辑把一个装两个 FP16 的 32 位寄存器**直接压缩**成装两个 FP8 的 16 位。
- **Tensor Core 常以 16 位块摄取数据**。

---

## 13. FP8 转换与量化

- 源数据是 FP16/FP32 时，用 **`cvt`**（常用向量形式 `.e4m3x2`/`.e5m2x2`）转 FP8，通常带 **`.satfinite`** 让越界值钳位：
  ```cuda
  cvt.rn.satfinite.e4m3x2.f16x2   // 取 2 个 fp16 → 饱和 → 舍入 → 打包
  ```
- 在喂 WGMMA 前，**先决定量化策略**（舍入模式、是否 satfinite、per-tensor/per-channel 缩放）。
- PTX 描述饱和行为：`|input|` 超过目标最大正常值时，结果变为**保号的 max normal**。
- PTX 为 sm_90+ 提供 `cvt.satfinite.{e4m3x2,e5m2x2}.{f32,f16x2}`。

---

## 14. FP8 WGMMA 指令

```
wgmma.mma_async.sync.aligned.m64nNk32.f32.e4m3.e4m3
    d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b;
```

- **`m64nNk32`**：M 固定 64，**K = 32（对 8 位）**，N 可变（8, 16, ... 256）。
- **`.f32`**：累加器 D 类型（单精度）。
- **`.e4m3.e4m3`**：A、B 的输入类型。FP8 特殊在 A、B **可为不同 FP8 格式**（如 `.e4m3 × .e5m2` 混合）。
- **`scale_*`**：缩放/符号翻转因子（A、B 取 {−1, 1}，scale_d 取 {0, 1}）。
- **K = 32 是关键区别**：一次吞掉 A 的 32 列。

### 14.1 Hopper 第四代 Tensor Core 的 FP8

- 输入 8 位（FP8），但**累加在 32 位（FP32）寄存器**里以保证数值稳定。
- 通过 WGMMA **原生支持 e4m3 与 e5m2**。
- 一条 WGMMA 指令发出巨量数学，有效隐藏延迟。
- 寄存器虽是 FP32，实际**像 FP22（~8 位指数 + ~13 位尾数）**——已知的硬件特性。
- 本质：**把 FP8 当压缩存储格式，在"伪 FP32"空间里做数学**。

---

## 15. FP8 的 A 在寄存器、K-Major 规则与精度陷阱

### 15.1 A 在寄存器（RS）

- FP8 RS WGMMA 中，A 操作数是**直接传给 WGMMA 的寄存器向量表达式**。
- **"显然"的办法（`ldmatrix ... .b8`）在 Hopper 上不可用**（SM90a 的 PTX 说明 `.b8` ldmatrix 仅 sm_100a+ 支持）。
  **Hopper 上不能用 ldmatrix 加载 FP8。**
- 因此大多数 Hopper FP8 WGMMA 实现二选一：
  - **SS 路径**：A、B 都用描述符；
  - **RS 路径**：用普通 shared load（`ld.shared.b32` / 向量化）把 A 装进寄存器，**事先按 WGMMA 想要的打包方式排好**，而不是用 ldmatrix。

### 15.2 严格的 K-Major 规则

- `wgmma.mma_async` **不暴露转置控制（`imm-trans-a/b`）**（不像 FP16/BF16 变体）。
- 无转置时，WGMMA 按默认 **K-major 规范布局**解释 shared 操作数；喂 MN-major 则无法"重解释"，必须自己打包/转置。
- 转置/打包步骤会增加指令、同步与 shared 流量，伤吞吐；MN-major 还会让 FP8 的 swizzle-atom 对齐/整除约束更复杂（128B atom）。
- **结论**：FP8 tile 用 **K-major staging（或离线预打包）**；只有 staging 期间会转置时才用 MN-major。

### 15.3 精度陷阱

- 即便文档说用 FP32 累加器，**实测报告 FP8 Tensor Core 累加像"降精度 FP32"（~8 位指数 + ~13 位尾数，即"共 22 位"）**——最低的 FP32 尾数位在累加时可能实际丢失。
- 大点积（大 K）下，缩小的有效尾数会**增大舍入/截断误差**，且误差随规约深度累积。
- 对策：用 **K-slicing 和/或多阶段累加**（先累加部分和，再以更高精度归约部分和），限制每个 WGMMA "chunk" 的累加深度，改善数值稳定。

---

## 16. 稀疏 WGMMA（Sparse）

- 一种特殊 WGMMA：Tensor Core 做同样的 MMA，区别是 **A 是结构化稀疏（50% 零）**。
- 操作数：`descA`（或 RS 形式的 A 寄存器片段）、`descB`、**`sp-meta`**（含打包索引的 `.b32` 寄存器）、
  **`sp-sel`**（32 位常量，"稀疏选择器"，选择哪些线程贡献某组的 metadata），以及其它操作数。

### 16.1 重要点

- 硬件**严格按"稀疏 A × 稠密 B"**运行：即使 B 数学上稀疏（含零），Tensor Core 也当它是稠密矩阵。
- 需为**每个线程**建一个 `sp_meta` 寄存器；只在硬件会读的 lane 里给 `sp_meta` 赋有意义值，
  其它 lane 通常置 `sp_meta = 0`（或任意值），因为那些 lane 的 metadata 被忽略。

### 16.2 打包与 Metadata（2:4 结构化稀疏）

- NVIDIA 稀疏 Tensor Core 的物理规则是 **2:4 结构化稀疏**：
  - 提供 4 个连续值（叫 **Quartet**），**必须删掉其中恰好 2 个**，只存两个幸存者（在 global 或经 shared 变换）。
  - 省 **50% 带宽与存储**。
  - 但若只给 GPU `[8.5, 3.2]`，它不知道它们原来在哪（8.5 来自索引 0、1 还是 2？）——这就是 **metadata** 的作用。
- 因为从 4 个位置（0,1,2,3）里选 2 个，需要**编码幸存者的位置**。

---

## 17. sp-sel 与 sp-meta 的配置

### 17.1 sp-sel（线程选择器）

- 告诉硬件**谁负责提供 metadata**——即在每组 4 个连续线程（T0–T3）里，哪个线程对是 metadata 贡献者（只在"仅一对贡献"时）。
- 各精度要求：
  - **TF32 稀疏（`.m64nNk16 .tf32`）**：spSel 须为 0（T0,T1）或 1（T2,T3）。
  - **FP16/BF16 稀疏（`.m64nNk32 .f16/.bf16`）**：spSel 须为 0 或 1。
  - **FP8/INT8 稀疏（`.m64nNk64 .e4m3/.e5m2/.s8/.u8`）**：**所有线程都贡献 metadata，spSel 必须为 0**（否则未定义）。

### 17.2 sp-sel 的两种选择规则

- **选 0**：从 T0、T1 读 metadata，忽略 T2、T3。
- **选 1**：从 T2、T3 读 metadata，忽略 T0、T1。
- **规则 1（Replicated）**：若加载器把 metadata 广播给所有线程（常见）→ **始终 sp-sel=0**（最简单）。
- **规则 2（Sharded）**：若拆分 metadata 省寄存器 → **动态让 sp-sel 匹配持有数据的线程**。
- 配置错误 → Tensor Core 从空寄存器读，把块当稠密或清零。

### 17.3 sp-meta（metadata 位域）

- 告诉硬件 **A 的非零在哪**。`spMeta` 是打包位域，规则：
  - **2:4 稀疏（FP16/BF16、FP8、INT8）**：A 每 4 个相邻元素有 2 个非零；只存 2 个非零，其位置（0..3）用**两个 2 位索引**编码进 metadata。
  - **1:2 稀疏（TF32）**：A 每 2 个相邻元素有 1 个非零；metadata 用 4 位索引指示 2 个位置中的哪个，只有两个特定位模式有意义，其它值未定义。

### 17.4 配置 sp-meta（FP16/BF16/INT8）

- 一个寄存器装载的 metadata **恰好够 4 条连续 WGMMA**。
- 规则：**主循环展开 4 次**，让 sp-meta 循环 0,1,2,3（第一个 K-tile 用 0，第二个用 1……）。
- 该索引递增硬件内部指向"寄存器内下一组 2 位索引"的指针。
- **不递增 → 硬件对所有计算重复用第一个 tile 的稀疏模式。**

### 17.5 配置 sp-meta（TF32 的例外）

- TF32 寄存器只够 2 条 WGMMA，策略特殊：**不能用顺序索引（0,1），必须在硬编码位掩码 14 和 4 之间切换**。
  - 第 1 次 K=16 计算：`sp-meta = 0b1110 (14)`（解码低位）。
  - 第 2 次 K=16 计算：`sp-meta = 0b0100 (4)`（解码高位）。
- 这些特定值是让硬件 swizzler 对齐非标准的 19 位/32 位 TF32 数据格式所必需的。
- 用 0 或 1 这类普通索引 → 硬件错位、矩阵结果错误。

---

## 18. 总结与学习衔接

### 18.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **commit/wait_group** | 把异步 WGMMA 分组；`wait_group N` 保留 N 组在跑，实现流水重叠 |
| **wgmma.fence** | 组边界标记 + 寄存器排序（不是跨 proxy fence） |
| **stmatrix** | 协作写回（地址/数据解耦），只支持 .b16/.b8，FP32 要先降精度打包 |
| **swizzle atom** | 16B；swizzle 的是"行 atom"，行内仍行主序，TMA 可反 swizzle |
| **FP8** | e4m3（前向）/e5m2（梯度）；靠缩放因子、打包（x2/x4）、cvt.satfinite |
| **FP8 WGMMA** | K=32，A/B 可混合格式，K-major 强制，累加像 FP22 |
| **稀疏** | 2:4 结构化；sp-sel 选 metadata 线程对；sp-meta 编码非零位置；TF32 用 14/4 特例 |

### 18.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| 流水线（commit/wait + TMA 供数） | 《8. Kernel Design》《8.1 Stream-K》 |
| 内核启动 / 集群协同 | 《8.2 Kernel Launch》 |
| 多 GPU 扩展 | 《9. Multi GPU》《10. Multi GPU Part 2》 |

### 18.3 一句话记忆

> **WGMMA Part 2 = 把异步 MMA"流水线化 + 正确写出 + 玩转低精度与稀疏"：**
> `commit/wait_group` 控制重叠、`stmatrix` 把碎片化的寄存器结果写成整洁 SMEM、
> FP8 靠缩放/打包/饱和翻倍算力、稀疏靠 `sp-meta/sp-sel` 再省一半带宽。

---

> 参考来源：`7. Wgmma part 2.pdf`（Lesson 7，43 页）。

---

# H100 · Kernel Design（kernel 设计的"招式"）

> 本文档基于课程讲义《8. Kernel Design.pdf》（Lesson 8 / The Tricks，98 页）整理，
> 系统总结 Hopper 上高性能 kernel 的设计方法与优化技巧：**warp specialization、流水线、
> ping-pong/cooperative 流水线、调度（persistent/Stream-K）、epilogue 融合**。
> 前置阅读：`H100-异步与屏障.md`、`H100-cp-async.bulk.md`、`H100-WGMMA.md`、`H100-WGMMA-part2.md`。

---

## 目录

1. [CUDA kernel 的两类与拐点](#1-cuda-kernel-的两类与拐点)
2. [Compute-bound kernel 的优化手段](#2-compute-bound-kernel-的优化手段)
3. [Warp Specialization（warp 特化）](#3-warp-specializationwarp-特化)
4. [流水线（Pipelining）与环形缓冲](#4-流水线pipelining与环形缓冲)
5. [两屏障握手与 ABA 问题](#5-两屏障握手与-aba-问题)
6. [Producer / Consumer 流程与三阶段](#6-producer--consumer-流程与三阶段)
7. [两种流水线：Cooperative vs Ping-Pong](#7-两种流水线cooperative-vs-ping-pong)
8. [Tile 尺寸与 RS/SS 选择](#8-tile-尺寸与-rsss-选择)
9. [调度（Scheduling）](#9-调度scheduling)
10. [Static / Grouped / Stream-K 调度器](#10-static--grouped--stream-k-调度器)
11. [Memory-bound kernel 的三类](#11-memory-bound-kernel-的三类)
12. [Epilogue（尾声融合）](#12-epilogue尾声融合)
13. [总结与学习衔接](#13-总结与学习衔接)

---

## 1. CUDA kernel 的两类与拐点

CUDA kernel 分两类：

- **Compute-bound**：受**算术运算速率**限制。
- **Memory-bound**：受**数据搬移速率**限制。

分界点是**算术强度（Arithmetic Intensity）**：`AI = FLOPs / Bytes moved`。

**拐点（Ridge Point）= 峰值 FLOPS / 峰值带宽**：

| 精度 | 拐点 |
|------|------|
| FP16 Tensor 运算 | ≈ **295 FLOP/Byte** |
| FP32 CUDA Core | ≈ **20 FLOP/Byte** |

- `AI < 拐点` → **memory-bound**；
- `AI > 拐点` → **compute-bound**。

### 1.1 Compute-bound kernel 的特征

1. **高算术强度**：每字节读写的 FLOPs 多，复用强（数据留在寄存器/shared，多次使用）。
2. **Producer 空闲**：最显著特征是 Producer/Consumer 失衡——producer warpgroup 在屏障处空等 consumer 赶上。
3. **Occupancy "必要但不充分"**：许多 compute-bound kernel 在**中等占用率**下即可接近峰值，只要发射足够 WGMMA。
   发射计算指令不是问题，**寄存器利用才是**。

---

## 2. Compute-bound kernel 的优化手段

- **Warp Specialization**（warp 特化）
- **Persistent Kernels + Tile Scheduling**（持久 kernel + tile 调度）
- **Circular buffer**（共享内存多级环形缓冲 + 显式同步）
- **Cluster-Level Optimizations**（簇级优化）
- **Register Pressure Management**（寄存器压力管理）
- **Megakernels**（巨型 kernel）
- **Epilogue Fusion**（尾声融合）

---

## 3. Warp Specialization（warp 特化）

Hopper 上一个 SM 能容纳很多活跃 warp，但**每个 cycle 只有少数能发射指令**。

- 若同一 warp 内线程走不同分支，warp 会串行化这些路径（最坏 ~32×）。
- 但若 **warp 0 负责加载数据、warp 1 对数据做计算**，它们是不同 warp、有独立执行上下文，就**避免了 SIMT 惩罚**。

**Warp specialization = 刻意让同一线程块内不同 warp 承担不同工作**：

- **producer warp**：搬/准备数据。
- **consumer warp**：对数据做计算。

### 3.1 为什么 warp specialization 几乎是必须的

1. **资源约束逼的**：你无法在不溢出的情况下把整个活跃状态（寄存器/谓词等）塞进每线程/每 warp，所以把工作拆到不同 warp。
2. **变延迟操作难以静态调度**：内存等变延迟操作让编译器/静态调度难以让所有单元都忙。
3. **阻塞同步会卡住发射**：若某 warp 要等屏障，特化后其它 warp 能立刻跑，SM 不浪费发射槽。

### 3.2 资源约束与 `setmaxnreg`

- WGMMA kernel 的**首要瓶颈是寄存器文件**：要最大化吞吐，线程必须在寄存器里持一大块输出矩阵；
  但每线程分配 ~200+ 寄存器会**大幅减少 SM 能容纳的 warp 数**。
- Warp specialization 把工作拆成两个角色，做**非对称资源分配**：
  - **Producer**：发 `cp.async`，需要**最少寄存器**。
  - **Consumer**：执行 WGMMA，需要**最多寄存器**。
- **`setmaxnreg`** 指令让 warp 在运行时**动态改变自己拥有的寄存器数**：
  - 以低寄存器数启动 kernel → SM 容纳更多活跃 warp → 最大化内存带宽利用。
  - 进入重计算段前，consumer warp 执行 `setmaxnreg` 请求更多寄存器；producer 用它减少寄存器。

### 3.3 Warp group 数量选择

- **Producer**：总是**恰好 1 个 warp group**（单线程就能发一条搬几 GB 数据的 `cp.async`）。
  producer 几乎不做数学，只管理 mbarrier；可用 `set_maxnreg` 把它们限制在 ~32 寄存器。
- **Consumer**：根据寄存器压力选 **1、2 或（罕见）3 个**；数量由累加器 tile 大小决定。
  - 预算：~232–240 寄存器（留出 barrier 等空间）。两种用法：
    - **Pingpong/Basic**：每个 WG 独立算一个完整 tile，EffectiveThreads = 128。
    - **Cooperative**：两个 WG 拆分同一 tile，EffectiveThreads = 256。

---

## 4. 流水线（Pipelining）与环形缓冲

**流水化 = 重叠**：把工作拆成阶段，让不同"项"的不同阶段同时进行，而不是一个项做完再做下一个。

```
无流水（串行）：Load → Wait → Compute → Wait → Load → Wait → Compute → Wait ...
                每个阶段都等上一阶段完成，传输时硬件空闲

有流水（重叠）：
  Load:  L0      L1      L2      L3
  MMA:         M0      M1      M2      M3
  算 tile N 的同时加载 tile N+1 —— 硬件一直忙
```

### 4.1 为什么必须做对

- **延迟鸿沟**：全局内存 ~400–800 周期，WGMMA ~30–60 周期/次。不流水时，**计算要等 20 倍于它运行的时间**。
- 不流水：load（400 周期计算空闲）→ compute（20 周期内存空闲）→ load（又 400 空闲）→ 计算硬件约 **5% 利用率**。
- 流水化：计算嚼 tile N 时，内存已在取 tile N+1；若流水够深，计算**永不等数据**，从 5% → 近 100%。

### 4.2 Circular Buffering（环形缓冲）

- 让 producer 与 consumer 重叠，需要一个二者之间的缓冲：一组**固定数量的 shared-memory "槽位（stage）"**，
  producer 填、consumer 排空。
- **环形缓冲 = 轮转复用这些槽位**：producer 写 stage i、i+1……绕回 0；consumer 读 i、i+1……绕回 0。
- 它让"算 tile N 的同时加载 tile N+1"成为可能。

### 4.3 Stages 与为什么需要多个

- 单缓冲会强制串行（Load→Compute→Store 只有一个阶段在忙）。
- 多缓冲允许多个在飞数据块：stage 1 填 buffer A、stage 2 处理 buffer B、stage 3 排空 buffer C。
- 环形意味着 buffer 走完一圈后**复用**——无需无限内存，只要足够 buffer 让所有阶段都忙。
- **需要多个 buffer 的根因是"所有权 + 重叠"**：一个阶段不能安全覆盖另一个阶段仍在读的数据。

---

## 5. 两屏障握手与 ABA 问题

环形缓冲里每个 stage 有**两个信号**，producer/consumer 永不竞争：

1. **FULL 屏障**：consumer 读之前在此等待。对 TMA 流水线它是事务屏障——producer 调 `mbarrier.arrive`，
   TMA 引擎在字节落到 shared memory 时发完成信号。consumer 无需轮询就知道 stage 确实填好了。
2. **EMPTY 屏障**：producer 覆盖前在此等待；consumer 用完该 stage 后发信号。

```
Producer 等 EMPTY → 声明 FULL 的 expect_tx → 发 TMA → TMA 完成 → FULL 触发
→ Consumer 读/用 → Consumer arrive EMPTY → stage 可复用
```

### 5.1 ABA 歧义

- 环形缓冲跨多轮复用同一 stage 索引。若同步只按 `stage_id ∈ [0..Stages-1]` 索引，
  consumer 看到"stage 0 满"时无法区分是 **第 1 轮 stage 0（旧数据）** 还是 **第 2 轮 stage 0（绕回后的新数据）**。
- 解法：跟踪一个 **phase bit**，每当环形索引从 Stages−1 绕回 0 时翻转。
  每个参与者（producer/consumer）维护状态三元组：`index`（当前槽位）、`phase`（绕回翻转的 1 位纪元）、
  `count`（单调递增迭代计数，用于簿记）。

---

## 6. Producer / Consumer 流程与三阶段

### 6.1 Producer 流程

每个 warpgroup 只有**一个被选中的线程**真正碰 mbarrier 和 TMA。每次迭代做三件事：

1. **获取 stage**：等 EMPTY 屏障，然后在 FULL 屏障上调 `expect_tx`。
2. **发 TMA 拷贝**：DMA 引擎写完 shared 字节后自动给 FULL 发信号（硬件级 producer-consumer 信号）。
3. **前进**：递增流水线状态，绕回时翻转 phase。

所有工作结束时，`mbarrier.try_wait` 等 consumer 释放每个剩余 stage 后才退出（不能有人还在读你的数据就退出）。

### 6.2 Consumer 流程

consumer 维护两个流水线指针：**正在消费的 stage** 与 **可释放回 producer 的 stage**。

- 因 WGMMA 是异步的，不立即完成；**只有确认读已完成后（`warpgroup_wait`）才能释放它读的 buffer**。
- 每次迭代：**等数据（FULL 屏障）→ 发 WGMMA（fence, arrive, gemm, commit）→ 等最老 WGMMA 完成（`wgmma.wait_group<N>`）→ 释放最老 buffer（EMPTY）→ 前进两个指针**。
- **每次 WGMMA 前的跨 proxy fence** 防止编译器把累加器读写重排到 WGMMA 边界之外。

### 6.3 三阶段

- **Prologue（填充）**：consumer 发 N 条 WGMMA 且不释放任何 buffer，让流水线填满，
  稳态时总有在飞 WGMMA 与 TMA load 重叠。第一条 WGMMA 把累加器初始化为零。
- **Steady-State（稳态）**：吞吐最优循环——对每个 k-tile：等数据 → 发 WGMMA → 等最老 → 释放最老。
  TMA 与 WGMMA 完全重叠，producer 领先 consumer 的释放指针 Stages 步。
- **Drain（排空）**：最后一个 k-tile 消费后，`warpgroup_wait<0>()` 冲刷所有在飞 WGMMA，
  再释放剩余 N 个 buffer，让 producer 干净退出。

---

## 7. 两种流水线：Cooperative vs Ping-Pong

两种主流水线都用 warp specialization（producer 加载 + consumer 计算 + 环形缓冲）。
**区别只在 epilogue 阶段**：Ping-Pong 把 wgmma 与 epilogue 重叠，Cooperative 不重叠。

### 7.1 Cooperative Pipeline

- **384 线程，3 个 warp group**：
  - WG0：Producer（发 TMA 加载）
  - WG1：Consumer（WGMMA + epilogue）
  - WG2：Consumer（WGMMA + epilogue，同一输出 tile）
- 输出 tile 128×128 时，两个 consumer 平分几何工作：consumer 0 算上半（M 0–63），consumer 1 算下半（M 64–127）。
  因同时处理同一输出 tile，它们**同一时刻消费同一份 A、B tile**。
- 持久调度：每个 CTA 循环抓取一串 tile，每个 warpgroup 走同一串 tile。
- **Consumer 循环**：维护 read 指针 + release 指针，偏移 N（通常 1）。这个"滞后"是因 WGMMA 异步——
  先等 read 游标 → 发 WGMMA → 前进 read 游标；**只有对应 wgmma 完成后（`wgmma.wait_group`）才能释放更老的 stage** 并前进 release 游标。
- **关键洞察：Cooperative 不归约 WGMMA 结果**——"合作"是共享 smem tile，不是合并累加器。
  每个 WG 独立做所有 K tile 的 wgmma，累加进自己的寄存器 accum（warpgroup 寄存器私有，物理上不可能跨 WG 通信）。
  每个 WG 写自己不相交的 M 区域。
- **收益**：单个更大的 tile 可由两个 WG 一起映射，实现**更大的有效 MMA tile（256×128 而非 128×128）**，同时每个 WG 仍满足寄存器预算。

### 7.2 Cooperative 的缺口（The Gap）

- consumer 累加完所有 k-tile 后要做 epilogue（缩放、bias、激活、写 global）。
- **整个 epilogue 期间 Tensor Core 完全空闲**。
- 对大 epilogue 或小 K 维，这段空闲时间占总运行时间的相当比例——流水线能重叠 load 与 compute，
  却**无法重叠 compute 与 epilogue**（因为只有一个 consumer 两件事都做）。

### 7.3 Ping-Pong（补上缺口）

- **加第二个 consumer**：一个跑 epilogue 时另一个对下一个 tile 做 MMA，二者交替，WGMMA 永不停。
- 384 线程，3 个 WG：WG0 producer，WG1 consumer（C0），WG2 consumer（C1）。
  producer WG **释放自己的寄存器**，把寄存器文件空间让给两个 MMA WG。
- 交替：
  ```
  C0: [MMA T0][Epi T0][MMA T2][Epi T2]
  C1:          [MMA T1][Epi T1][MMA T3][Epi T3]
  ```
- 持久调度下，CTA 被分配一串 tile T0..T5：producer 全部处理（步长 1），
  consumer 0 处理偶数（步长 2，起点 0），consumer 1 处理奇数（步长 2，起点 1）。

### 7.4 Ping-Pong 的 barrier（2×2 网格）

- 3 个 stage 的流水线**只分配恰好 3 个物理 barrier**，跨迭代复用。
- 用 **2×2 的 mbarrier 网格**保证 C0 的 epilogue 与 C1 的 MMA 同时进行而不破坏 shared memory：
  - **行（Stages/Depth）**：0 = MMA 阶段，1 = Epilogue 阶段。
  - **列（Groups）**：0 = Consumer 0，1 = Consumer 1。
- 每个 consumer 有 `group_id`；`arrive()` 给**另一个** consumer 的 barrier（当前深度）发信号，
  `wait()` 在**自己**的 barrier（当前深度）上等。
- 深度循环：MMA → Epilogue → MMA → …。
- 必须保证：① 一次只有一个 consumer 做 MMA；② 一次只有一个做 epilogue；③ 同一 tile 先 MMA 后 epilogue。

### 7.5 一个 warp group 迭代的完整时间线

```
1. ordered_barrier.wait()       ← 等 C1 的上一个 epilogue
2. WGMMA mainloop               ← K 循环：从 smem 读、累加进寄存器
3. ordered_barrier.arrive()     ← "我的 MMA 完了，C1 可开始它的 MMA"
4. mma_tail()                   ← warpgroup_wait<0>，释放最后几个 smem stage
5. ordered_barrier.wait()       ← 等 C1 的 epilogue 完成
6. epilogue.store()             ← 融合 + R→S 拷贝 + TMA store
7. epilogue.store_tail()        ← 等所有 TMA store 落地
8. advance pipeline states      ← 前进 2 步（每个 warp group 一步）
9. ordered_barrier.arrive()     ← "我的 epilogue 完了，C1 可开始它的 epilogue"
10. fetch next tile, loop 回 1
```

### 7.6 warpgroup 内的角色分工

- **Producer warp group（128 线程 = 4 warp）**，每个 warp 一个角色：
  - Mainloop DMA warp：TMA 加载 A、B tile 到 shared；
  - Epilogue DMA warp：TMA 加载 C tile 到 shared（用于残差加）；
  - Scheduler warp：取下一个 tile 坐标；
  - Auxiliary warp：可选的额外加载。
- **Consumer 0 / 1（各 128 线程）**：偶/奇 tile 的数学 + epilogue。

> 注意：数学 warpgroup 会**切换角色**——MMA 时是 mainloop 流水线的 consumer，epilogue 时变成 store 流水线的 producer。

---

## 8. Tile 尺寸与 RS/SS 选择

### 8.1 输出 tile 尺寸

- 输出 tile 决定单个 CTA 计算多大块的 C。
- **目标：让 tile_m、tile_n 越大越好**（越大算术强度越高，因为每载一次 A、B tile 就做 `2×tile_m×tile_n×tile_k` 次运算）。
- **小/中 tile（128×128 或 64×128）**：单个 consumer WG 的寄存器够存累加器 → 用 Base 或 Ping-Pong。
- **大 tile（256×128 或 128×256）**：单个 consumer WG 在 FP32 下装不下 → 用 Cooperative。

### 8.2 内层 tile_K 与流水线 stage 数

- `tile_k`：内层循环每次沿 K 维走多深；大小要"喂饱 Tensor Core 又不撑爆 shared memory"。
- 要让 epilogue 重叠高效，**K 维必须够大**（太小则 C1 的 MMA 会在 C0 的 epilogue 之前做完，又卡住）。
  **计算时间必须 ≥ 内存写时间**。
- WGMMA 原生一次消费 **K=16**（FP16/BF16）或 **K=16**（TF32）。
- 通常要 **3 或 4 个 stage** 隐藏 TMA 加载延迟。

### 8.3 RS vs SS（A 放寄存器 vs 共享内存）

| 输入类型 | 选择 | 原因 |
|---------|------|------|
| f16/bf16（2 字节、等宽、无 scale） | **永远 SS** | 原生 SS 支持广，A、B 都留 shared，寄存器压力最小 |
| 非 2 字节等宽（tf32/f32/fp8/int8） | **AkBk（TN）才 SS，否则 RS** | 这些类型要 K-major 喂入；布局已是 AkBk 则 SS 直喂，否则需 RS 做换位/转置 |
| 混合宽度 | **RS** | 混合宽度 MMA 前要转换/反量化，SS 无 pre-MMA 变换阶段，RS 的 smem→register 拷贝可以做 |
| 带 scale/零点（tuple mixed） | **RS** | scale/零点要在 wgmma 前做逐元素算术，在寄存器里做 |

---

## 9. 调度（Scheduling）

**调度 = 决定"谁在何时以何顺序做什么"的决策逻辑**，即把工作单元（Tile）映射到工作者（线程/warp/CTA/SM）的策略+机制。

好的调度区分"硬件都忙"与"部分 SM 空闲"、区分"良好局部性"与"抖动/停顿"，保证可预测的完成时间而非尾延迟悬崖。

### 9.1 为什么调度重要

- **Occupancy/Utilization**：给 SM 足够多独立 tile 隐藏延迟。
- **Load Balance**：避免"长尾"（少数 CTA 拿到大 tile，其它早完成闲置）。
- **Locality/Bandwidth**：最大化 L2 复用、最小化冗余 global load、促进 multicast。
- **解耦 tile 大小与调度粒度**：持久调度引入**第三分解轴：K 维**。

### 9.2 非持久调度（问题）

- Grid Size = 总工作量 / Block Size，硬件调度器分配 block，完成即退役。
- 三个问题：① 多次启动 kernel 有开销；② **尾效应**（133 block vs 132 SM → 先跑 132 再跑 1）；
  ③ 硬件调度器通常**线性**（Block 0 算 (0,0)，Block 1 算 (0,1)……等扫到下一行时，所需数据早已从 L2 被逐出）。

### 9.3 持久调度（Persistent）

- 启动**固定数量**的线程块（通常 = SM 数），这些 block **常驻 GPU**，循环计算下一个 tile 的索引并处理，直到全部完成。

### 9.4 波量化（Wave Quantization）问题

- 数据并行基线：CTA 轮转取 tile。150 tile / 132 SM = 1.136 tile-units/SM → 实际 2 波 → 利用率 56.8%。
- 消除波量化：把"余下的 8 个 tile"沿 **K 维**平分给所有 132 个 SM（每个 SM 算约 0.06 个 tile 的 K 工作）。
- **本质洞察：波量化不是硬件限制，而是"tile 并行分解"的结果；持久调度让你换一种分解。**

---

## 10. Static / Grouped / Stream-K 调度器

### 10.1 Static Persistent Scheduler

- 标准 GEMM 的默认高吞吐调度器。"Static"指工作到线程的映射是**数学预计算**的，不是原子计数动态认领。
- 把输出矩阵看成 tile 网格，用**光栅化（Rasterization / Swizzling）曲线**（常为 Z 曲线或 U 曲线）给持久 block 分配 tile。
- 持久 block 算完第一个 tile 后，按"启动的总 block 数"跳步（Grid Stride Loop）找下一个 tile。
- 用于标准 GEMM、compute-bound kernel（开销最低、无块间同步、swizzle 最大化缓存命中）。

**Rasterization（光栅化顺序）**：
- 决定 CTA 被分配到输出 tile 的顺序，直接影响 L2 局部性（让共享输入数据的相邻 CTA 在时间上靠近执行）。
- 顺序错了 → 每跳一个新 tile 都要从 DRAM 重载整块矩阵。

**路径策略 AlongN vs AlongM**：
- **Column-major（外=N，内=M）**：固定列 n 扫完一整列再右移 → **B 的列保持热**（B 最大复用），A 行在轮换。
- **Row-major（外=M，内=N）**：固定行 m 扫完一整行再下移 → **A 的行保持热**（A 最大复用）。
- 结论：**想让哪个矩阵留在缓存，就垂直于它的复用维度遍历。**

**Swizzling（打散遍历）**：
- 只沿一个方向复用（如 A 一行保持热）时，B 是零复用流式；扫下一行时又要重载 B。
- 需要**两个维度同时有局部性**。swizzle 把"细线"遍历改成"厚块"遍历。
- swizzle size 可调：size=1 是细光栅，size=2 → 4 个 tile 厚。CUTLASS 常用 1、2、4、8。
- 规则：**内层（快轴）取更长的维度**（更长内循环 = 更多次迭代后才"迈大步"，迈大步是昂贵的缓存上下文切换点）。

**Cluster 与 swizzling**：
- 纯 swizzle 的复用靠"相邻 CTA 时间上靠近"（希望），但物理执行顺序会漂移。
- Cluster 解决：簇内 CTA 在**空间+时间上一起**执行、可同步协作；共享操作数面板可跨 CTA 复用——复用从"偶然"变"有意"。

### 10.2 Grouped Persistent Scheduling

- 为 **Grouped GEMM**（一次 kernel 启动算多个不同 GEMM）设计。
- 把"Group 0 再 Group 1…"概念上**拼接成一条长线性序列**；维护当前 group ID、该 group 起始线性索引、该 group 的 tile 数。
- CTA 的 `linear_idx` 前进时可能跨 group 边界，朴素标量搜索太贵 → 用 **warp-level speculative search**。

**warp-level speculative search**：
- 不在当前 group 时，warp 以 **32 个 group 为一批**扫描：每个 lane 载一个 group 的形状、算它的（cluster 对齐）tile 数。
- 用 warp 原语（`__ballot_sync`、`__ffs`、`__shfl_sync`）选出"范围包含 linear_idx"的 lane 并广播 GroupInfo；没命中则跳 32 个 group 重试。
- 找到所属 group 后：算局部偏移 `k = linear_idx - start`，swizzle k → (cluster_major, cluster_minor)，按光栅顺序转 (M,N)，再加簇内 CTA 偏移。
- 该扫描很便宜（持久化摊薄，主要在 group 边界触发）。

### 10.3 Stream-K Scheduling

**要解决的问题（尾效应）**：标准调度把完整输出 tile 分给线程块；若 tile 总数不是 SM 数的整数倍，最后一"波"只填一部分——活跃 SM 处理最后 tile，其余 SM 全闲置。

- Stream-K 把整个矩阵乘看成**一条连续的 1D "数学迭代带"**，单元是"一定数量的 MMA 运算"（而非"一个完整 tile"）。
- 把总数学量**严格平分**给所有处理单元；一个 block 可能算完整 tile、可能继承半成品、也可能预算用完时停在一半。
- **Hybrid 实现（只打尾）**：把 tile 沿 K 维拆分引入跨块通信开销，对全问题用 Stream-K 常适得其反。
  最优是**混合**：早期波保持纯数据并行（最大吞吐），**只对最后"尾"波用 Stream-K 均衡负载**。
  **50% 启发式**：尾波已较满（>50%）时回退到标准数据并行，避免不必要的归约开销。

**Fixup（点对点归约）**：
- Cluster A 算 Tile X 前 50%、Cluster B 算后 50% → 写 global 前要相加（Fixup）。
- 用一块 global scratchpad：前半写完部分累加器 + 置 flag；后半完成时检查，看到 A 完成就载入 A 的部分结果、加到自己的寄存器、写最终和。

**反向 tile 迭代（Backward Tile Iteration）**：
- 两个 block 共享一个 tile 时，"算 K 尾"的 block 要等"算 K 头"的 block 完成才能写。
- Stream-K 让 worker **按 K 逆序**迭代：越靠后的块越晚算它的共享部分，等它要合并时，前面的块已算完 → 大幅减少等待。

**保持 L2 局部性**：
- Stream-K 的 1D 带天然破坏空间局部性。解法：把 worker 逻辑分组，让同组单元处理**不同输出 tile 的重叠 K 区间**——
  它们同时迭代完全相同的 K 切片，L2 里的输入读完美重叠。

---

## 11. Memory-bound kernel 的三类

1. **Bandwidth-bound**：DRAM/L2 吞吐接近峰值，运行时间由"搬了多少字节 + 效率"决定。
2. **Latency-bound**：性能受"依赖解析多久"（访存、原子、同步、长指令链）限制，带宽未饱和。
3. **Locality-bound**：搬了很多字节但大量浪费（L2 命中低、随机/间接访问、cache/TLB 抖动）。

> bias、激活、缩放、残差加、amax 跟踪——在朴素实现里都是独立 memory-bound kernel；
> 好 kernel 里它们被**折叠进 GEMM 的 epilogue**。**你永远不写独立的 bias+GELU kernel**，
> 而是在 epilogue 里组合；一次 global 往返而非三次。

---

## 12. Epilogue（尾声融合）

**Epilogue = GEMM kernel 的最终处理阶段**：发生在寄存器里的 wgmma 完成之后、写回 global 之前。
它把原始累加结果变换成最终输出格式，并顺便做其它 memory-bound 操作——目标是在做必要数学
（缩放、bias、激活）时**隐藏延迟且不溢出寄存器**。

### 12.1 Epilogue 操作清单

1. `alpha * Acc`（只缩放累加器）、`alpha * Acc + beta * C`（线性组合）
2. Bias 加（每行/每列 bias）
3. 激活（ReLU / GELU / SiLU）
4. 残差式加法路径
5. TopK + Softmax 融合（列向 softmax 变体）
6. Aux 张量操作（aux load / aux store）
7. 规约融合（行/列/标量规约）
8. Absmax/amax 跟踪（FP8 路径常用）
9. 缩放融合（A/B/C/D 的每行/列 scale、alpha-beta 变体）
10. 块缩放因子生成（块缩放工作流）

### 12.2 Epilogue 操作顺序

```
1. 载入 tile 所需全部输入：Acc、可选 C、bias、scales、可选 AuxIn
2. 先建基础表达式：Z = alpha*Acc + beta*C（无 C 则 Z = alpha*Acc）
3. 再加仿射项：行/列 bias、残差/额外线性项
4. 对 Z 施加非线性：ReLU / GELU / SiLU
5. 施加输出缩放/量化（需要时收集 amax）
6. 物化输出：主输出 D、可选 AuxOut（激活前/后）、可选行/列/标量规约
7. 最终类型转换 + store
```

### 12.3 模板

1. **常见前向融合**：`Z = alpha*Acc + beta*C → Z += bias → Y = activation(Z) → D = cast/scale(Y)`
2. **训练式（存 aux）**：`Z = alpha*Acc + beta*C + bias → AuxOut = Z → Y = activation(Z) → D = Y`
3. **反向式**：`dY = alpha*Acc + beta*C → dX = dActivation(dY, AuxIn) → dBias = reduction(dX) → D = dX`
4. **Softmax/TopK 路径**：`S = alpha*Acc + beta*C → rowmax/rowsum/exp 归一化 → 可选 TopK → D`

### 12.4 融合生命周期（六阶段）

SM90 的 epilogue 不是单个函数，而是一个围绕"**分离变化的部分与不变的部分**"设计的六阶段流水线：

| 阶段 | 作用 |
|------|------|
| `begin()` | 一次性全局初始化 |
| `begin_loop()` | 每 tile 设置 |
| `previsit()` | 取操作数（staging bias/scales/residual，TMA 搬到 smem，lane 按需载入寄存器） |
| `visit()` | 算术计算（EVT 遍历：缩放/bias/激活/clamp 直接在累加器上做，全程留在寄存器+ALU） |
| `reduce()` | FP8 的 amax 归约（warp shuffle 先、跨 warp 再 smem staging） |
| `postreduce()` | R→S（stmatrix，含 swizzle + 精度转换），把控制权交给 TMA |
| `tma_store()` | `fence.proxy.async` → `producer_commit` → 单 leader 线程发 `cp.async.bulk.tensor`，TMA 异步 smem→gmem |
| `end_loop()/end()` | 前进流水线 / 最终冲刷清理 |

> **visit() 的主要风险是寄存器过度使用**：融合深度超过寄存器预算时，local memory spill 会抵消融合收益。
> **postreduce() 是"交接边界"**：数据 staged 后，CUDA core 从流水线释放，TMA 硬件独立管理异步写 gmem。

---

## 13. 总结与学习衔接

### 13.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **bound 分类** | AI < 拐点 → memory-bound；FP16 拐点 295、FP32 拐点 20 |
| **Warp specialization** | producer（少寄存器发 TMA）+ consumer（多寄存器跑 WGMMA），`setmaxnreg` 动态调 |
| **流水线** | 环形缓冲 + 两屏障（FULL/EMPTY）+ phase bit 解 ABA |
| **三阶段** | Prologue 填充 / Steady-State 重叠 / Drain 排空 |
| **Cooperative** | 两 consumer 平分一个 tile（大 tile、无 epilogue 重叠） |
| **Ping-Pong** | 两 consumer 交替 MMA/epilogue（WGMMA 永不停），2×2 mbarrier 网格 |
| **调度** | Persistent（Static 光栅化 / Grouped 投机搜索 / Stream-K 打尾） |
| **Epilogue** | 把 memory-bound 算子全折叠进尾声，一次 global 往返 |

### 13.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| Stream-K 调度 | 《8.1 Stream-K》 |
| kernel 启动 / 持久化协同 | 《8.2 Kernel Launch》 |
| 多 GPU 扩展 | 《9. Multi GPU》《10. Multi GPU Part 2》 |

### 13.3 一句话记忆

> **Kernel Design = 把"算得慢"（compute-bound）与"搬得慢"（memory-bound）都变成"永不停"：**
> warp specialization 分 producer/consumer、环形缓冲 + 双屏障把 load 与 WGMMA 重叠、
> ping-pong 让 MMA 与 epilogue 重叠、persistent+Stream-K 调度消除尾效应、epilogue 融合省掉全部多余访存。

---

> 参考来源：`8. Kernel Design.pdf`（Lesson 8 / The Tricks，98 页）。

---

# H100 · Stream-K 调度

> 本文档基于课程讲义《8.1 Stream-K.pdf》（Lesson 8.1，10 页）整理，
> 深入讲解 **Stream-K 调度器**的机制：fixup、split、三重角色、锁、分组与 L2 局部性，
> 以及与 **HyTiS** 的对比。前置阅读：`H100-Kernel-Design.md`（§10.3 Stream-K 概述）。

---

## 目录

1. [第一性原理：fixup 从何而来](#1-第一性原理fixup-从何而来)
2. [工作单元（Unit of Work）](#2-工作单元unit-of-work)
3. [Split（一次拆分的贡献）](#3-split一次拆分的贡献)
4. [跟踪三元组：K_idx / k_tile_count / is_final_split](#4-跟踪三元组k_idx--k_tile_count--is_final_split)
5. [三种角色](#5-三种角色)
6. [锁（The Lock）](#6-锁the-lock)
7. [Groups（分组）与 L2 局部性](#7-groups分组与-l2-局部性)
8. [HyTiS：另一种解决波量化的思路](#8-hytis另一种解决波量化的思路)
9. [总结与学习衔接](#9-总结与学习衔接)

---

## 1. 第一性原理：fixup 从何而来

对单个输出 tile `C_tile`，GEMM 计算：

```
C_tile = Σ_{k-tiles} (A_tile_k · B_tile_k)
```

- **一个 CTA 算完该输出 tile 的所有 K-tile** → **无需跨 CTA 归约**。
- **多个 CTA 各自只算 K-tile 的一个子集** → 每个 CTA 产出一个**部分和**，这些部分和**必须合并**。

在 Stream-K 调度里，这个合并步骤就叫 **fixup**。

---

## 2. 工作单元（Unit of Work）

每个工作单元包含：

- **tile 坐标**：`(m_idx, n_idx, l_idx)`（`l_idx` 用于 batch/分组问题）。
- **`k_idx`**：该工作单元在输出 tile 内**起始的 k-tile**。
- **`k_tile_count`**：该工作单元为这个输出 tile **计算了多少个 k-tile**。

是否需要归约由下面判断（这是 `requires_fixup(...)` 背后的关键谓词）：

| 情况 | 判定 | 归约 |
|------|------|------|
| K 上的完整 tile | `k_tile_count == k_tiles_per_output_tile` | **不需要** |
| K 上的部分 tile | `k_tile_count != k_tiles_per_output_tile` | **需要** |

---

## 3. Split（一次拆分的贡献）

一个 CTA 被分配的 k-tile 迭代区间，可能**正好落在一个输出 tile 的中间**。

- 例：**3 个输出 tile**（每个 90 个 k-tile），**4 个 CTA 单元**。
- **"split" = 一个 CTA 对单个输出 tile 的贡献**。
- 上面的 Unit 1 有**两个 split**：一个是 tile 0 的尾部，一个是 tile 1 的头部。
- 代码一次处理一个 split（这就是 `advance_to_next_work` 里的 `k_tile_remaining` 循环）。

---

## 4. 跟踪三元组：K_idx / k_tile_count / is_final_split

对单个 split（一个 CTA 在某个输出 tile 上的工作）：

- **`K_idx`**：该 split 在输出 tile 的 K 维内**从哪里开始**。
  ```
  K_idx = tile_iter_start - output_tile_iter_start
  ```
  - Unit 0 对 tile 0：`K_idx = 0`；
  - Unit 1 对 tile 0：`K_idx = 67`。
- **`k_tile_count`**：该 split 处理多少个 k-tile。
  - Unit 0 对 tile 0：67；
  - Unit 1 对 tile 0：23（`= 90 − 67`）。
- **`is_final_split()`**：`(K_idx + k_tile_count) == k_tiles_per_output_tile`。
  - 为真 → 该 split 覆盖到 K 维的**末尾**。Unit 1 对 tile 0 就是 final split（`67 + 23 = 90`）。

---

## 5. 三种角色

给定一个输出 tile 可能有 2–4 个 CTA 各算 K 的一段，角色直接由三元组决定：

| 条件 | 角色 | 行为 |
|------|------|------|
| `K_idx == 0` | **first split（首个）** | 算了 `[0, N)`，在你之前没人写 workspace → **直接 store** |
| `is_final_split() == true` 且非"分离归约" | **final split（末尾）+ epilogue 拥有者** | 算了 `[X, 90)`，**等前面所有人**、载入他们的累加结果、加上自己的、跑 epilogue |
| 其它 | **middle split（中间）** | 算了 `[A, B)`（`0 < A < B < 90`），把自己的部分和**归约进 workspace 已有的值** |

---

## 6. 锁（The Lock）

- **物理上**：global memory 里一个**连续的 int 数组**，每个输出 tile 一个（多 warpgroup kernel 再乘 `num_barriers`）。
- 该数组的指针就放在**归约数据缓冲区之后**的同一块分配里；kernel 启动时每个锁都从 0 开始。
- **锁是一个整数**，编码"这个输出 tile 已完成并写进 workspace 的 K 维工作量"。普通（非分离归约）模式下，
  它计数**累计已处理的 k-tile 数**，且**只增不减**。
- 锁在 **K-tile 空间**里编码进度：
  - **确定性模式（deterministic）**：每个 split 等待**恰好等于它起始位置的累计 k-tile 数**，
    强制严格的**从左到右**归约顺序。
  - **非确定性模式（non-deterministic）**：middle split 只需知道 workspace 已被初始化（`lock >= 1`），
    然后**竞争式地原子归约**进去。

---

## 7. Groups（分组）与 L2 局部性

**Groups 是 L2 局部性优化**：把 stream-K 单元划分成 **G 个独立子组**，每组只协作处理自己那部分 stream-K tile。

- 这是 **stream-K 特有的优化**：因为 stream-K 破坏了"波式光栅化"——一个 CTA 可能横跨输出网格不同区域的 tile，破坏局部性。
- **无分组（G=1）**：所有 stream-K 单元共享一个跨所有输出 tile 的大 K-tile 池。Unit 0 可能处理 tile 0 和 tile 1，
  而 Unit 7 处理 tile 5 和 tile 6——**空间位置完全不同，L2 里的数据毫不重叠**。
- **有分组**：每组里的 unit 会计算"与数据并行公式中、按光栅化顺序属于同一波的那些 tile"的**相同 K 区间**。

### 7.1 分组层级

```
Groups（最多 8 个，为了 L2 局部性）
  └── 每个 group 含多个 cluster-tile
      └── 每个 cluster 含多个 CTA（线程块）
          └── 每个 CTA 处理 K-tile
```

- 分组沿**光栅化维度**确定。例：沿 M 光栅化、`problem_blocks_m / cluster_m = 4` → 得 4 个 group。
- group 在输出空间里**交错**，最终 tile id 计算：
  ```
  output_tile_id = (output_tile_id_in_group * num_groups) + group_idx
  ```

---

## 8. HyTiS：另一种解决波量化的思路

- **HyTiS** 通过让**部分波（partial wave）用更细粒度的 tile** 来解决波量化，让更多 SM 保持忙碌。
- 它是**纯空间分解（M×N）+ 异构 tile 大小**，对比 Stream-K 的 **K 维分解 + 同构 tile 大小**。
- HyTiS 在一次 kernel 启动里用**两种 tile 大小**：
  - **大 tile（如 128×256）**给完整波 → **最大吞吐**；
  - **小 tile（如 64×64）**给部分波 → **最小延迟**。
- **无归约、无 workspace、无 barrier、无 fixup**。
- **代价**：当问题在 M、N 上很小而 K 很大时，HyTiS 无能为力（因为它只能分解 M×N 空间）。

### 8.1 Stream-K vs HyTiS

| | Stream-K | HyTiS |
|---|---|---|
| 分解维度 | K（同构 tile） | M×N 空间（异构 tile） |
| 归约/workspace/fixup | 有（fixup + 锁） | **无** |
| 适用 | 各种形状，含"小 M/N 大 K" | M、N 足够大、可分空间 |
| 权衡 | 引入跨 CTA 归约开销 | 小 M/N、大 K 时帮不上 |

---

## 9. 总结与学习衔接

### 9.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **fixup** | 多 CTA 各算一段 K 后，把部分和合并的步骤 |
| **split** | 一个 CTA 对单个输出 tile 的贡献 |
| **三元组** | `K_idx`（起始）、`k_tile_count`（数量）、`is_final_split()`（是否覆盖 K 末尾） |
| **三角色** | first（直接 store）/ final（epilogue 拥有者）/ middle（归约进 workspace） |
| **锁** | global 里每 tile 一个 int，编码累计 K 进度；确定/非确定两种归约模式 |
| **Groups** | 把单元分组以恢复 L2 局部性（最多 8 组） |
| **HyTiS** | 用异构 tile 大小（大 tile 整波 + 小 tile 尾波）免归约地解决波量化 |

### 9.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| Stream-K 的 kernel 启动与协同 | 《8.2 Kernel Launch》 |
| 跨 CTA/跨 SM 归约与多 GPU | 《9. Multi GPU》《10. Multi GPU Part 2》 |

### 9.3 一句话记忆

> **Stream-K = 把"输出 tile"沿 K 维切成工作单元，让所有 SM 平分数学量、一起同时干完；**
> 代价是跨 CTA 的 fixup 归约，用"锁"（累计 K 进度）协调首/中/尾三种角色，
> 用"Groups"找回被 1D 带破坏的 L2 局部性；HyTiS 则走另一条路——用异构 tile 免归约地打散尾波。

---

> 参考来源：`8.1 Stream-K.pdf`（Lesson 8.1，10 页）。

---

# H100 · Kernel Launch（kernel 启动约束与依赖启动协议）

> 本文档基于课程讲义《8.2 Kernel Launch.pdf》（Lesson 8.2 / Constraints，12 页）整理，
> 讲解 kernel 启动相关的重要**约束/属性**，以及**多 kernel 协作**的依赖启动协议
> （`griddepcontrol`）、L2 预取技巧与重叠窗口调优。
> 前置阅读：`H100-Kernel-Design.md`、`H100-Stream-K.md`、`H100-异步与屏障.md`。

---

## 目录

1. [重要的约束与函数](#1-重要的约束与函数)
2. [为什么需要多 kernel 设置](#2-为什么需要多-kernel-设置)
3. [Stream 串行化问题](#3-stream-串行化问题)
4. [硬件信令：GDC 指令](#4-硬件信令gdc-指令)
5. [主机启动授权（Host Launch Authority）](#5-主机启动授权host-launch-authority)
6. [依赖方的墙：wait](#6-依赖方的墙wait)
7. [生产方的绿灯：launch_dependents](#7-生产方的绿灯launch_dependents)
8. [L2 预取技巧（Split DMA）](#8-l2-预取技巧split-dma)
9. [调优重叠窗口](#9-调优重叠窗口)
10. [总结与学习衔接](#10-总结与学习衔接)

---

## 1. 重要的约束与函数

| 约束/函数 | 作用 |
|----------|------|
| `__cluster_dims__(x,y,z)` | **编译期**指定线程块簇（thread-block cluster）形状 |
| `__launch_bounds__(maxThreads[, minBlocksPerSM[, maxBlocksPerCluster]])` | 限制每 block 线程数等；**第 3 个参数是 cluster 专用，在 SM90 上更重要** |
| `__maxnreg__(N)` | **封顶每线程寄存器数** |
| `__grid_constant__` | 只读的、grid 生命周期内的 kernel 参数——**常用于 CUtensorMap / TMA 描述符** |
| `__forceinline__` | 强制 nvcc 在**单个翻译单元内**内联该函数 |
| `__restrict__` | 告诉 nvcc：该指针在其作用域生命周期内，所指内存**不被其它访问同一数据的指针别名** |

---

## 2. 为什么需要多 kernel 设置

- **Epilogue 有硬上限**：超出会毁性能——融合操作与 mainloop 的累加器 tile **竞争寄存器**，造成寄存器压力。
- **跨 tile 依赖的操作（LayerNorm、Softmax）与 GEMM epilogue 的"独立 tile 处理"架构上不兼容**：
  数学上**必须有一个 kernel 边界**才能正确归约。
- 因此**必须接受 kernel 边界与中间的 DRAM 写**。工程挑战**不是硬塞成一个巨型 kernel**，
  而是**让 kernel 之间的交接几乎零成本**。
- 手段：**依赖启动（dependent-launch）协议、L2 预取策略、跨 grid 的 mbarrier**。

---

## 3. Stream 串行化问题

- 通常**同一 stream 上的下一个 grid 必须等上一个 grid 完全退役**才能开始发工作。
  这种严格完成顺序，在第一个 grid 进入**低占用率的尾部**时造成**利用率缺口**。
- 硬件调度器把同一 stream 的 grid 当成**单一有序队列**：**在 Kernel A 报告全局完成之前，不会给 Kernel B 分配线程块槽位**，
  即使某些 SM 已经欠载。
- 当 Kernel A 接近完成时，只有**越来越少的 SM 仍有活跃工作**；其余 SM 因默认 stream 规则下没有下一 grid 的块可发而闲置。
- 交接是**整 grid 退役事件**，不是逐 tile 转移——除非启用显式依赖启动控制，否则所有依赖安全的、可重叠的机会都被忽略。
- **关键代价是"尾延迟放大"**：硅片在场，却在 Kernel A 的最后阶段不发有用指令。

---

## 4. 硬件信令：GDC 指令

**GDC（`griddepcontrol`）指令控制 GPU 执行时间线里的 grid 交接时机。**
物理上它把两件事分开：**"调度器可以启动依赖方"** vs **"依赖方可以读依赖敏感的内存"**。

- **`griddepcontrol.launch_dependents`**：由**正在运行的 warp** 发出的、面向调度器的信号。
  它告诉分发逻辑：**依赖 grid 现在有资格启动**，即使当前 grid 尚未全局退役。
- **`griddepcontrol.wait`**：依赖 grid 里的**硬执行栅栏**。到达该指令的 warp 被**扣住**，
  直到启动依赖条件被满足，防止**过早读取未解析的数据**。

二者合起来是一个**两段式协议**：**提前启动许可 + 内存安全门控**。
调度器可以重叠启动工作，而栅栏为依赖敏感的 load 保持正确性。
它们是**指令流里的物理控制点**，放置位置直接改变重叠时机与硬件争用。

---

## 5. 主机启动授权（Host Launch Authority）

- **只有 CPU 的 launch packet 才能授权"依赖重叠"的宽松 stream 行为**。物理目的是：除非主机软件显式选择加入，否则保持默认 stream 语义不变。
- 启动时，CPU 把属性写进**GPU 命令处理器消费的命令描述符**。其中一个属性**授予"依赖 grid 在前驱完全退役前被调度"的许可**。
- 若该许可位存在 → 调度器状态机把 `launch_dependents` / `wait` 当作依赖控制指令来执行；
  若缺失 → 调度器强制普通串行化，这些指令**无启用效果**。
- 因此**交接授权是"主机发起、硬件强制执行"的**：设备端信令不能覆盖 stream 策略，除非启动元数据授权它。
- 这保护了混合工作负载的正确性（有的 stream 需严格顺序，有的需受控重叠）。

---

## 6. 依赖方的墙：wait

- 该阶段定义依赖 grid 里 **producer warp 的"提前启动停顿点"**。物理目的：让依赖 kernel **尽早预留执行上下文**，同时**阻塞不安全的内存流量**。
- 依赖 grid 可被接纳、在可用 SM 上开始执行设置指令；producer warp 跑到 `griddepcontrol.wait` 时被硬件**停驻（park）**。
- 停驻期间，这些 warp **不能发依赖敏感的全局读**（如前一个 kernel 产出的激活张量），
  防止因读"尚未最终确定"的数据造成缓存填充与内存序违规。
- 一旦依赖信号满足，栅栏释放，这些 warp 恢复发 load——**无需在 producer 完成后再付完整冷启动延迟**。
- **代价是预留资源**：停驻的 warp 仍占调度槽，可能减少其它工作的即时余量。

---

## 7. 生产方的绿灯：launch_dependents

- 该阶段是**活跃 kernel 发出"打开依赖调度"的精确信号**。物理目的：在"最后一个正确性安全的时刻"触发重叠，仍能暴露启动延迟隐藏。
- 活跃 grid 的 producer warp 在**剩余写已经足够推进、依赖方可安全启动时**执行 `griddepcontrol.launch_dependents`。
  该信号面向的是**调度器资格**，不是"立即可读内存"的许可。
- 在**持久 kernel** 里，正确放置点与**最后一个被调度工作 tile 的生命周期**绑定：
  信号应对齐**真实的流水线完成**，而不是代码区域的词法结尾。
- 信号之后，依赖块可被分发，同时活跃 grid 排空剩余工作；依赖 grid 的 wait 栅栏仍守护依赖敏感读，直到条件满足。
- **发太晚 = 浪费重叠；发太早 = 增加并发压力、可能降低净吞吐。**

---

## 8. L2 预取技巧（Split DMA）

- 该阶段通过**在依赖 kernel 内把角色分到不同 warp** 来实现重叠。物理目的：在依赖受缚的 producer 被栅栏挡住时，让内存结构忙于**无依赖的传输**。
- **一个 producer warp** 到达依赖屏障、在激活读之前暂停；
  **另一个 prefetch warp** 继续跑，对**不依赖 producer 完成的静态权重**发 DMA 式请求。
- 这些请求**提前填充 L2**，减少后续 miss 惩罚；内存系统因此在本来停顿的依赖时间里**预热有用的缓存行**。
- 交接是**非对称但安全**的：激活流量在栅栏后等待，权重预取不用。
  栅栏释放时，producer 因部分工作集已在缓存而看到更好的有效延迟。
- **前提**：预取的数据要有**高复用**且能在 L2 存活到被消费。

---

## 9. 调优重叠窗口

- **重叠比 = 依赖启动信号发射 与 真正依赖就绪 之间的时间差**。更大的差 = 潜在隐藏更多，但**前提是资源未拥塞**。
- **信号太早** → 两个 kernel 争 SM 发射槽、寄存器文件容量、共享内存分配、内存端口，可能把每个 kernel 拖慢到总时间反而增加。
- **预取太激进** → L2 行在被用之前就被颠掉，DRAM 流量因回填而飙升，带宽花在把被逐出的数据搬回来，抵消预取重叠的收益。
- **实用调优靠硬件计数器**：在**最晚安全点**启动、**只预取稳定/复用的张量**、
  以 **L2 命中率 + 内存队列压力** 作为主要护栏。

---

## 10. 总结与学习衔接

### 10.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **约束** | `__cluster_dims__`/`__launch_bounds__`/`__maxnreg__`/`__grid_constant__`/`__restrict__` |
| **多 kernel 必要** | epilogue 有上限、跨 tile 依赖（LayerNorm/Softmax）必须 kernel 边界归约 |
| **串行化问题** | 同 stream grid 必须整 grid 退役 → 尾延迟放大、SM 闲置 |
| **GDC 协议** | `launch_dependents`（提前启动许可）+ `wait`（内存安全栅栏）两段式 |
| **主机授权** | CPU launch packet 授权依赖重叠，设备端信令不能越权 |
| **Split DMA** | producer warp 被栅栏挡、prefetch warp 预取静态权重暖 L2 |
| **调优** | 最晚安全点发信号、只预取复用张量、看 L2 命中 + 队列压力 |

### 10.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| 跨 grid 协同 / 依赖启动 / mbarrier | 《9. Multi GPU》《10. Multi GPU Part 2》 |
| 持久化调度 + 依赖启动 | 《8. Kernel Design》《8.1 Stream-K》 |

### 10.3 一句话记忆

> **Kernel Launch = 用约束/属性（cluster、maxnreg、grid_constant、restrict）给编译器交底，**
> **再用"主机授权 + GDC 双指令（launch_dependents / wait）+ Split DMA 预取"把 kernel 之间的交接做成
> 几乎零成本——既让依赖 grid 提前启动隐藏尾延迟，又用 wait 栅栏守住内存正确性。**

---

> 参考来源：`8.2 Kernel Launch.pdf`（Lesson 8.2 / Constraints，12 页）。

---

# H100 · 多 GPU 系统（Multi GPU）

> 本文档基于课程讲义《9. Multi GPU.pdf》（Lesson 9，27 页）整理，
> 系统介绍 H100 的多 GPU 互联体系：**NVLink / NVSwitch / ConnectX-7 / SuperPOD / Rail 网络 / P2P**。
> 前置阅读：`H100-架构介绍.md`（NVLink、GPC）、`H100-Kernel-Design.md`（调度）。

---

## 目录

1. [为什么需要多 GPU](#1-为什么需要多-gpu)
2. [NVLink 与 NVSwitch](#2-nvlink-与-nvswitch)
3. [H100 DGX 节点与 SuperPOD](#3-h100-dgx-节点与-superpod)
4. [NVLink 4.0 细节](#4-nvlink-40-细节)
5. [NVSwitch 3 与 ConnectX-7](#5-nvswitch-3-与-connectx-7)
6. [OSFP 笼与存储网络](#6-osfp-笼与存储网络)
7. [Rail 对齐系统](#7-rail-对齐系统)
8. [三层网络：Leaf / Spine / Core](#8-三层网络leaf--spine--core)
9. [Quantum-2 QM9700 交换机](#9-quantum-2-qm9700-交换机)
10. [自适应路由与 SHIELD](#10-自适应路由与-shield)
11. [P2P 通信与 UVA](#11-p2p-通信与-uva)
12. [总结与学习衔接](#12-总结与学习衔接)

---

## 1. 为什么需要多 GPU

**问题**：用 H100 训练一个 **1T 参数的模型、10T token**：

- 经验观测：每 token 每参数约需 **6 次运算（FLOPs）**（前向 + 反向 + 更新）。
- 总 FLOPs = `6 × 10^12 × 10^13 = 6 × 10^25`。
- 假设每秒 1000 TFLOPS → 需 **约 1900 年**！

**这就是为什么需要多 GPU**：把多块 GPU 连起来，获得更高吞吐、跑更大模型。两项技术支撑：

- **NVLink Gen 4**：绕过 CPU 的 GPU 直连，**900 GB/s 双向带宽**，比 PCIe Gen 5 快 **7 倍**。
- **NVSwitch**：让节点内每块 GPU 都能以全速与其它任何 GPU 通信。

---

## 2. NVLink 与 NVSwitch

- 传统计算里，GPU 是离散单元，靠较慢的 PCIe 通道通信。
- H100 范式转变：创建一张 **"mesh"**，让每块 GPU 都能以极高速度访问其它 GPU 的内存。
- 8 GPU 集群可表现为**单一的内存+算力池**。
- 架构可从单服务器 8 GPU 扩展到 **256 GPU 集群**（NVLink Switch System），以**原生芯片速度**通信。

---

## 3. H100 DGX 节点与 SuperPOD

### 3.1 DGX H100 节点

- 8 块 **SXM5** 形态的 H100。
- 板上含 **4 个第三代 NVSwitch**。
- GPU 与交换机经**第四代 NVLink** 连接，每 GPU **900 GB/s** 带宽。
- **8 个 ConnectX-7 网卡（NIC）**：专用于把网络任务从主 CPU 卸载出去的处理器。
- **4 个 OSFP 笼**（Octal Small Form-factor Pluggable）。

### 3.2 H100 SuperPOD

- 约 **127–128 块 GPU**，组织成 **4 个 SU（Scalable Unit，可扩展单元）**，每个 SU 有 **32 个 DGX H100 节点**。
- "32" 这个数能用标准交换机端口数（通常每交换机 64 端口）做出**完美平衡的 Level 1 网络**。
- 设计基础：单个 DGX H100 节点有 **8 个独立的 ConnectX-7 计算网络接口（HCA）**，每 GPU 一个。

---

## 4. NVLink 4.0 细节

- 每块 H100 有 **900 GB/s 双向带宽**，约比 PCIe Gen5 快 7 倍；让 GPU 在 HBM 之间通信、跳过 CPU 路径。
- H100 用 **18 条独立 NVLink "link"** 达到该速度。
- 与上代不同，第四代 NVLink **更注重密度**：每条 link 只用 **2 对高速差分对**（从 4 对降下来），
  能在同样空间塞更多 link。

---

## 5. NVSwitch 3 与 ConnectX-7

### 5.1 NVSwitch 3

- 标准 8-GPU HGX 板上用 **4 块 NVSwitch** 连接全部 8 块 H100；**每块交换机接 4–5 个 NVLink 端口**。
- 交换机构成**完全无阻塞的互联**：每块 GPU 能**同时**以全 900 GB/s 与任何 GPU 通信，无需排队等通道清空。
- NVIDIA 把 NVSwitch 装进外部托盘（**NVLink Switch System**），可把最多 **256 块 H100（32 个机架）**连成一个 SuperPOD。
- **NVSwitch 自带 ALU**，可在交换机内做**加法/归约**——这是带来巨大性能提升的最重要特性之一。

### 5.2 ConnectX-7

- 服务器箱内 GPU 靠 NVLink（900 GB/s）通信；数据一旦要出箱到另一台服务器，就撞上 **ConnectX-7**。
- 一个 DGX 里有 **8 个计算网络接口（ConnectX-7）**。
- **这是瓶颈**：训练速度取决于你如何高效应对这个 **18 倍的带宽骤降**。ConnectX-7 的存在就是为了最小化"出箱"的代价。
- ConnectX-7 让网络**在 CPU 不知情的情况下读写 GPU 内存**。
- 网卡与网络交换机协作，**在飞行中（in flight）对梯度求和**——把网络流量从 **O(N)**（随集群规模线性增长）变成 **O(1)**（恒定）。
  **这是大规模训练能线性扩展的唯一原因。**

---

## 6. OSFP 笼与存储网络

### 6.1 OSFP 笼（计算网络）

- **4 个物理笼**提供 **8 条独立的 400Gb/s 网络链路**（共 3.2 Tb/s），靠 **"Twin-port" 双端口技术**实现。
- 它们内部连到 **8 块 ConnectX-7 网卡**。
- 支持 **GPUDirect RDMA**：让不同 DGX 节点的 GPU 互聊，不占用系统 CPU。
- 这 4 个 OSFP 笼**专用于计算网络（Compute Fabric）**——存储流量不走这些线缆。

### 6.2 存储网络（Storage Fabric）

- 用于把海量数据集载入 GPU、保存 checkpoint。
- 用**单独的 PCIe 卡**（装在机箱后部标准 PCIe 槽），**不是 OSFP 笼**。
- 通常是 **Fat Tree 或标准 leaf-spine**；不同于 "Rail" 网络，**任何 DGX 节点都要能连到任何存储阵列**。

---

## 7. Rail 对齐系统

- 标准网络里，一台服务器可能用一条线缆承载它所有组件的流量。DGX SuperPOD 里，网络被**物理拆成 8 条并行、隔离的网络**——这就是 **"Rails"**。
- 节点内有 8 块 GPU（编号 0–7）：**Rail 1 连接集群里每台节点的 GPU 0** 到同一组交换机，**Rail 2 连接每台节点的 GPU 1** 到另一组交换机，以此类推。
- **Rail 1 的流量在叶交换机层绝不干扰 Rail 2** —— 在整个集群上形成 8 个独立连接平面。
- Rail 系统利用 **NVIDIA SHARP** 技术，把数据操作卸载到网络交换机本身。

---

## 8. 三层网络：Leaf / Spine / Core

SuperPOD 用 **3 层**（Leaf、Spine、Core）把 SU 连起来：

| 层 | 作用 |
|----|------|
| **Layer 1: Leaf（叶）** | 连接节点 |
| **Layer 2: Spine（脊）** | 连接 SU 之间 |
| **Layer 3: Core / Super-Spine（核/超脊）** | 最大规模扩展 |

这些层就是一堆 **Quantum-2 InfiniBand 交换机**连在一起。

### 8.1 Leaf 层

- 线缆从 DGX H100 服务器后部物理引出。
- **Rail 隔离**：8 个独立交换机"平面"——Rail 1 交换机只连每节点的第 1 张网卡（GPU 0），Rail 8 只连第 8 张（GPU 7）。
- Leaf 交换机处理**本地 SU 内**流量：若 Node 1 的 GPU 0 要跟（同 SU 的）Node 2 的 GPU 0 通信，
  流量走 `Node → Leaf → Node`，**无需上更高层**。
- QM9700 用 **SHARPv3**（比上代强 32 倍），可处理复杂 AI 数学。

### 8.2 Spine 层

- 连接 Leaf 交换机，让 SU-1 的节点与 SU-2 的节点通信。
- **Spine Groups（脊组）**：脊交换机也按 rail 对齐分组——Spine Group 1 只连"处理 Rail 1"的 Leaf 交换机。
- **隔离**：保证 Rail 1 的流量绝不"泄漏"到 Rail 2 的线缆（否则拥堵）。
- 例：SU-1 的 GPU 0 → SU-2 的 GPU 0：`Node(SU1) → Leaf(Rail1) → Spine(Group1) → Leaf(Rail1) → Node(SU2)`。

### 8.3 Core / Super-Spine 层

- 超大规模集群（如 ≥127 节点）时加第三层。
- 连接多个 **Pod**（SU 的集群）。
- 这一层常改用 **800G 光模块**减少线缆数，再逻辑拆回两条 400G 链路。

---

## 9. Quantum-2 QM9700 交换机

- 提供超高带宽、低延迟的 GPU-GPU 互联。
- **64 个 400Gb/s 端口**。
- 前面板 **32 个 OSFP 笼**；每个 OSFP 笼实际承载**两条独立 400Gb/s 链路** → `32 笼 × 2 链路 = 64 逻辑端口`。
- 64 端口恰好匹配 **32 节点的 SU**：
  - **32 端口（下行）**：连 SU 内 32 个节点（如 Rail 1 连所有 32 节点的 GPU 0）；
  - **32 端口（上行）**：连"上"到 Spine（Layer 2）以访问其它 SU。

---

## 10. 自适应路由与 SHIELD

### 10.1 Leaf 层的自适应路由

- 上游流量（去其它 SU）时，自适应路由**至关重要**。
- 数据包从 GPU 到 leaf、需去另一 SU 时，要上 spine；非阻塞/超订阅 fat-tree 里有多个可选 spine。
- 不用静态 hash（固定把某流送到同一 spine，可能碰撞），Quantum-2 硬件**监控所有上行端口的队列深度与拥塞**，
  按**每包/每消息**动态把包送到**最不拥塞**的 spine 链路。
- 即便去 spine 的某条路堵了，流量也能从其它路顺畅走。

### 10.2 Spine 层的自适应路由

- 标准 2 层 fat-tree 里，从某 spine 到某 leaf 通常只有一条"下行"路径。
- 但自适应路由对**故障处理**和**并行链路多路径**仍重要。
- 若朝某 leaf 的缓冲满了，spine 传递反压；Quantum-2 的自适应特性（如 **SHIELD**）帮助**隔离**该拥塞，
  避免扩散到 spine 里其它未受影响的流量。

### 10.3 SHIELD（硬件故障切换）

- 传统 InfiniBand：线缆断/链路抖动时，软件控制器要 **5–30 秒**算新路由表并推给所有交换机——**直接毁掉整轮训练**。
- **SHIELD 把恢复逻辑直接搬进交换机硬件 ASIC**：
  - 链路 down 时，交换机不丢包，而是**立刻在本地硬件表里找替代有效路径**；
  - 找到就更新自己的表避开坏节点、走健康邻居；
  - 找不到就**向邻居发硬件信号**，让其以后不再把流量发过来。

---

## 11. P2P 通信与 UVA

### 11.1 P2P（Peer-to-Peer）

- P2P CUDA 机制让两块 GPU **不经 CPU** 通信。
- 可显式 P2P 内存拷贝，或 P2P 直接访问；整个传输走 NVLink。
- 这一切由 **UVA（统一虚拟寻址）** 支撑。

### 11.2 UVA

- UVA 让 CPU 与 GPU **共享单一虚拟地址空间**。
  - **UVA 之前**：CPU、GPU 各有自己的指针，需手工管理哪个指针指向哪块物理内存。
  - **UVA 之后**：系统**仅凭指针值**就能确定数据物理上在哪（系统 RAM 还是 H100 的 HBM3）。
- 必须用 `cudaDeviceEnablePeerAccess` 启用 Peer 访问，否则这功能无法工作。

### 11.3 `cudaDeviceEnablePeerAccess` 与 `cudaMemcpyPeer`

- **`cudaDeviceEnablePeerAccess`**：
  - **单向**：若需双向拷贝或两侧 kernel 互访内存，必须**在两块设备上都调用**。
  - 不启用 → 可能回退到更慢的 PCIe 路径。
  - H100 服务器上，若可用会**自动用 NVLink（900 GB/s）**，否则用 PCIe Gen5。
- **`cudaMemcpyPeer`**：让两块独立 GPU 之间**直接传输数据、不经过主机（CPU）内存**。
  - 用 NVLink 拷贝前必须启用 `cudaDeviceEnablePeerAccess`。

### 11.4 原始 P2P 的真相

- H100 HGX 板上不是 8 块 GPU 连成一条线，而是经 NVSwitch 连成**复杂 mesh**。
- 纯 P2P 要**手工管理走哪条链路**：GPU 0 跟 GPU 7 通信是直连还是经 GPU 3 跳？
- 虽然 H100 有 900 GB/s 双向 NVLink，但若从 SM 直接发普通 LD/ST 跨 NVLink 结构，
  **很可能只用一条或几条 NVLink lane，很难打满带宽**——"用吸管喝消防栓的水"。
- **这就是为什么需要 NCCL 这类库来管理多 GPU 连接。**

---

## 12. 总结与学习衔接

### 12.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **为什么多 GPU** | 1T 参数 × 10T token = 6×10²⁵ FLOPs，单卡 1900 年 |
| **NVLink 4.0** | 900 GB/s 双向、7× PCIe、18 link、2 差分对 |
| **NVSwitch 3** | 无阻塞全互联、内置 ALU 做 in-switch 归约 |
| **ConnectX-7** | 出箱瓶颈（18× 带宽降）、GPUDirect、in-flight 梯度求和 O(N)→O(1) |
| **SuperPOD** | 4 SU × 32 节点、127–128 GPU、Rail 对齐 8 平面 |
| **三层网络** | Leaf（连节点）/ Spine（连 SU）/ Core（连 Pod） |
| **SHARP / SHIELD** | 交换机内做归约 / 硬件级故障切换（5–30s → 即时） |
| **P2P / NCCL** | UVA + peer access + memcpyPeer；原始 LD/ST 打不满带宽，需 NCCL |

### 12.2 与本课程其他内容的衔接

| 本文概念 | 对应后续专题 |
|---------|-------------|
| 多 GPU 通信原语 / 集合通信 | 《10. Multi GPU Part 2》 |

### 12.3 一句话记忆

> **单卡算力不够 → 用 NVLink/NVSwitch 把 GPU 连成"单一内存+算力池"（节点内 900 GB/s），**
> **出箱靠 ConnectX-7 + Rail 三层网络 + SHARP 在交换机里做归约、SHIELD 在硬件里做故障切换；**
> **编程上靠 UVA + P2P（peer access/memcpyPeer），但真正打满带宽要用 NCCL。**

---

> 参考来源：`9. Multi GPU.pdf`（Lesson 9，27 页）。

---

# H100 · 多 GPU 编程（Multi GPU Part 2）

> 本文档基于课程讲义《10. Multi GPU Part 2.pdf》（Lesson 10，60 页）整理，
> 系统介绍多 GPU 分布式训练的**作业调度（Slurm/PMIx）、集合通信（NCCL）、四种并行策略**。
> 前置阅读：`H100-Multi-GPU.md`（硬件互联体系）。

---

## 目录

1. [Slurm 与 PMIx（作业调度与进程发现）](#1-slurm-与-pmix作业调度与进程发现)
2. [设备隔离与 GPU 绑定](#2-设备隔离与-gpu-绑定)
3. [NCCL 概述与初始化](#3-nccl-概述与初始化)
4. [NCCL 六大集合通信原语](#4-nccl-六大集合通信原语)
5. [AI 的四种并行策略](#5-ai-的四种并行策略)
6. [NCCL 各操作详解](#6-nccl-各操作详解)
7. [点对点与分组：ncclSend/Recv 与 ncclGroup](#7-点对点与分组ncclsendrecv-与-ncclgroup)
8. [总结与学习衔接](#8-总结与学习衔接)

---

## 1. Slurm 与 PMIx（作业调度与进程发现）

### 1.1 Slurm

- **Slurm 是开源的作业管理器/调度器**，面向 Linux 高性能计算集群。
- 它分配 GPU、CPU、内存；**知道你的作业在哪跑，但不一定知道你的应用如何跨节点自通信**。
- 过去 `mpirun` 用 SSH 连每台节点、核对主机名、交换密钥；大集群（64+ 节点）这个"握手"可能要好几分钟。
- 现在用 **`srun --mpi=pmix`**：Slurm **同时在所有节点上启动进程**。

### 1.2 PMIx（Exascale 进程管理接口）

- Slurm 启动 1000 个进程后，rank 5 知道自己活着，但**不知道 rank 0 的 IP、也不知道怎么联系 rank 999**——它们是孤岛。
- 训练开始前，进程需要交换技术细节（IP、GPU 句柄）。**PMIx 提供一个临时数据库**：
  - **`PMIx_Put`**：进程把元数据（主机 IP、端口、CUDA IPC 句柄）推到本地 PMIx server。
  - **`PMIx_Commit`**：把本地数据推到全局命名空间。
  - **`PMIx_Get`**：其它进程查询这些数据以发现对等方。

### 1.3 为什么 Slurm + PMIx 一起重要

- Slurm 单独负责资源分配（节点/GPU/CPU）与任务放置，但历史上依赖旧 PMI（扩展性难超 ~1 万 GPU）。
- PMIx 是现代、面向 exascale 的进程管理接口（取代 PMI-1/PMI-2）。
- 二者集成带来：许多情况下**免 mpirun 直接启动**、**大规模高效交换作业信息**（rank/节点列表/端点）、
  通过 **NCCL-PMIx 插件**与 NCCL 紧密集成。

### 1.4 Slurm 脚本与术语

- 脚本（`.sh`/`.sbatch`）告诉 Slurm 两件事：**要什么资源**（时间/GPU/CPU/内存）、**拿到后做什么**（跑 Python、编译等）。
- 用 `sbatch script.sh` 提交；Slurm 登录计算节点、设置环境、运行脚本命令。

| 术语 | 含义 |
|------|------|
| **Job** | 最高层工作单元，代表一次资源分配（`sbatch`/`salloc` 创建） |
| **Task** | 应用的一个进程（每节点/每 GPU 的进程数） |
| **Rank** | 任务的 ID；默认指**全局 rank**（整个作业中的 rank） |
| **Local rank** | 任务在**特定节点内**的唯一 ID（0 到该节点任务数） |
| **Namespace** | 作业内每个 rank 的 jobID，作业内唯一 |

---

## 2. 设备隔离与 GPU 绑定

### 2.1 设备隔离（两个机制）

若让 Slurm 每个 task 用一个 GPU，Slurm 不是"礼貌地请你用一块 GPU"，而是在操作系统层**强制**：

1. **`CUDA_VISIBLE_DEVICES`**：Slurm 在进程内设置该环境变量，让 CUDA **只看到列表里的设备**（单节点有效）。
2. **Linux cgroups 的 device allowlist**：为 Task 0 建一个"沙箱"。

> **调试陷阱**：若每个 GPU 上跑不同 task，看日志时**每个错误都会指向 GPU 0**（即使实际不是 GPU 0），
> 因为每个进程看到的可见设备都从 0 重新编号。

### 2.2 启动与绑定步骤

`Slurm` 用 PMIx 编排启动后，应用侧初始化 PMIx 的步骤：

```
1. 连接 Slurm daemon
2. 初始化 PMIx，拿到 namespace 和全局 rank
3. 拿到 local rank
4. 用 local rank 选择具体的 GPU
5. 结束进程
```

- `CUDA_VISIBLE_DEVICES` 让进程看到**过滤后、从 0 重编号**的 GPU 集合。
- **`PMIX_LOCAL_RANK`** 是节点内 rank，是"把任务分配到该节点 GPU"的自然索引。
- 用 `cudaGetDeviceCount` 得知过滤后可见多少设备；需要原始物理索引时可解析 `CUDA_VISIBLE_DEVICES`。
- 深度调试用 `cudaDeviceGetPCIBusId`，把"可见设备 0"对应到 Slurm 实际分配的硬件 GPU。
- 绑定到本地 GPU 后启动 kernel；结束用 `PMIx_Finalize(NULL, 0)` 告诉 Slurm 完成。

---

## 3. NCCL 概述与初始化

### 3.1 什么是 NCCL

**NVIDIA Collective Communications Library** —— 面向 GPU 的库，让 GPU 间的**集合通信**又快又省心。

- 分布式训练中 GPU 持续交换张量（梯度/参数），NCCL 提供高度优化的原语：
  **AllReduce、AllGather、ReduceScatter** 等。
- 在 **CUDA stream + 指针**层面集成：你交给 NCCL 设备指针 + 一个 `cudaStream_t`，
  NCCL 自己在该 stream 上调度 GPU 工作（kernel + memcpy + 网络操作），与你的 kernel 通过普通 CUDA stream 顺序组合。
- 做 AllReduce 时，CPU 只是把一个 NCCL kernel 发到 GPU stream 上。

### 3.2 初始化：NCCL Unique ID

- 通信前，NCCL 需要一个"所有进程找到共同汇合点"的方式，它由 **NCCL Unique ID** 表示。
- **只有一个进程**能生成该 ID；想加入组的每个进程都必须持有**完全相同**的 ID。
- 流程：Rank 0 让 NCCL 创建 Unique ID → 放进 PMIx 的 Key-Value Store（KVS）供他人读 →
  Rank 0 `commit` 推送出去 → fence 保证所有人都能读到 → 各 follower 查 KVS 并调 NCCL 初始化。

### 3.3 `ncclGetUniqueId` 与 `ncclCommonInitRank`

- **`ncclGetUniqueId`**：起始时各 GPU 不知周围其它 GPU 及如何经 NVLink/InfiniBand 到达它们。
  为通信，NCCL 要建一个 **communicator**。`ncclGetUniqueId` 创建 `ncclUniqueId` 结构并记下第一块 GPU 的 IP，
  之后每块 GPU 进来把自己的 IP 记进该结构。
- **`ncclCommonInitRank`**：**设置阶段最贵、最复杂的函数**——ID → 硬件连接：
  - 每个 rank 从 `ncclUniqueId` 提取 Rank 0 的 IP:Port；
  - 每个 rank 建一个标准 TCP socket 连 Rank 0；
  - Rank 0 等到收满 `nranks` 个连接——**这是一个 barrier，Rank 799 慢则所有人等**；
  - TCP 建立后不发数据，先发"自己的 rank、连接类型、设备"等元数据。

### 3.4 Rank 0 作为"架构师"：路由

- Rank 0 分析全局图，为 AllReduce 等操作找**最高带宽、最低延迟**路径：
  - **节点内**优先用 **NVSwitch**；**跨节点**选特定 **InfiniBand NIC**；
  - 决定用 **Ring**（延迟优化）还是 **Tree**（带宽优化）。
- Rank 0 把"路由表"（发给谁、从谁收）发回每个 rank；节点内 rank 互相映射内存、
  配置 NVSwitch 允许 GPU 直访；跨节点 rank 做 **RDMA 握手**，确保能不经 CPU 直接写对方内存缓冲。

### 3.5 `ncclCommDestroy`

- 拆除之前建立的硬件路径：把 communicator 标记为对后续 kernel 启动无效。
- 若 GPU 正在执行 NCCL kernel（如 stream 里的 AllReduce），**不会立即抽走内存**，靠**内部引用计数**。
- 释放：Init 时在 HBM 分配的 4–8MB scratchpad、CPU RAM 里用于 PCIe 的 staging 区、NVLink 映射与 InfiniBand Queue Pair。

---

## 4. NCCL 六大集合通信原语

在 H100 集群上，我们不按"发送/接收"想，而按**跨 NVLink 结构的数据操作模式**想。六大原语：

| 原语 | 行为 |
|------|------|
| **Broadcast** | 一块 GPU 把数据拷给所有 GPU |
| **Reduce** | 所有 GPU 合并数据，结果落在**一块** GPU |
| **AllReduce** | 所有 GPU 都拿到所有人数据的**归约和** |
| **AllGather** | 各 GPU 从片段开始，最终都拿到完整合并缓冲 |
| **ReduceScatter** | 先归约，再拆分输出，每 GPU 拿到不同块 |
| **All-to-All** | 每 GPU 给其它每 GPU 发一块、也收一块 |

---

## 5. AI 的四种并行策略

| 并行 | 一句话 | 主要通信 |
|------|--------|---------|
| **Data Parallelism（数据并行）** | 模型全复制、数据切分 | **AllReduce**（梯度）+ Broadcast（初始化） |
| **Tensor Parallelism（张量并行）** | 把单层矩阵切分到多 GPU | Broadcast / AllGather / AllReduce / ReduceScatter |
| **Pipeline Parallelism（流水线并行）** | 把层堆纵向切分 | 点对点 send/recv |
| **Expert Parallelism（专家并行）** | MoE 把专家分布到多 GPU | **All-to-All** |

### 5.1 Data Parallelism（数据并行）

- 最常见策略：**模型在每个 GPU 上各有一份完整副本**，全局 batch 切成 mini-batch，每 GPU 拿自己那份。
- 每 GPU 对各自数据做前向+反向；优化器更新前，**必须保证所有副本一致** → 聚合（通常平均）各 GPU 的梯度。
- **AllReduce 是数据并行最重要操作**（每次反向结束发生）：GPU i 有梯度 gi，需每 GPU 都拿到平均梯度 `(1/N)Σgi`。
- **Broadcast** 用于训练开始或 checkpoint 恢复：Rank 0 初始化参数并广播给所有 rank，保证第 0 步所有副本数学上一致。

### 5.2 Tensor Parallelism（张量并行）

- 模型大到单 GPU 装不下时（训练时内存需求再 ×3–4：权重 + 梯度 + 优化器状态 + 激活），把矩阵切成块分到不同 GPU 算，算完再合起来。
- 切分方式：
  - **列切分（Column-wise）**：用 **Broadcast** 把完整输入拷给各 worker，乘完用 **AllGather** 合并结果。
  - **行切分（Row-wise）**：乘完用 **AllReduce** 求和得到最终结果。
  - 一个标准 Transformer block 的 TP 意味着 **2 次 AllReduce**（注意力子层 1 次 + FFN 子层 1 次）。

### 5.3 Sequence Parallelism（序列并行，SP）

- 标准 TP 只切重的矩阵乘（Linear 层），**不切** Linear 之间的操作（LayerNorm、Dropout、GELU/SiLU）。
- 标准 TP 里，Linear 后 AllReduce 的输出是**复制**的——8 GPU 每块都存一份完整激活矩阵 `[SeqLen, HiddenDim]` 只为做 LayerNorm/Dropout。
- 但 LayerNorm 只作用于**单个 token 的 Hidden 维**，在 Sequence 维上独立——**无需每块 GPU 复制完整序列，可把序列切分**。
- **打破 AllReduce**：数学上 `AllReduce = ReduceScatter + AllGather`。SP 把 LayerNorm/Dropout 注入通信循环：
  **先 ReduceScatter → 停 → 在分片数据上做 LayerNorm → 需要完整数据做下一次矩阵乘时才 AllGather**。

### 5.4 Pipeline Parallelism（流水线并行）

- 若 TP 是"横向切一层"，PP 就是**纵向切模型**（切层堆）。
- 每 GPU 只存自己那几层的参数与优化器状态。
- 一次传大 batch 会大量 GPU 空等 → 把大 batch 切成**微小 batch（micro-batch）**：GPU 1 算完第一个 micro-batch 就传给 GPU 2、立刻算第二个。
- PP 只和流水线"邻居"通信 → **只用点对点 send/recv**。

### 5.5 Expert Parallelism（专家并行，MoE）

- 与每个输入都用全部参数的"稠密"模型不同，**MoE 只激活一小部分参数（专家）**。
- 好处：参数量可 ×100（加更多专家），但每 token 只选 top-2 专家，算力开销仍较低。
- 专家分布在不同 GPU（GPU 1 持专家 A/B，GPU 2 持 C/D……）；token 需要别的 GPU 上的专家时经网络发过去 →
  最终**每 GPU 都要给其它每 GPU 发数据** = **all-to-all 通信**。
- 专用库（如 **DeepEP**，底层 NVSHMEM）擅长此场景：**先跨节点高效打包发送，再用更快的 NVLink 在本地快速分发**。

---

## 6. NCCL 各操作详解

### 6.1 NCCL 调用的通用格式

```
sendbuff, recvbuff, count, datatype  → 告诉 GPU 搬多少个元素、如何解释
op                                    → 执行什么数学操作
comm                                  → 通信句柄（存节点与集群的元数据）
stream                                → 指定异步行为
```

### 6.2 `ncclBroadcast`

- **Root GPU** 把一个张量发给 communicator 里其它所有 GPU。
- H100 DGX 上：Root GPU 把数据包**发一次**给 NVSwitch，**NVSwitch 自己把它物理复制到其它 7 个端口（同时）**。
- 因交换机负责复制，**广播给 8 GPU 的时间 ≈ 发给 1 GPU 的时间**。
- `datatype` 传 `ncclFloat8e4m3` / `ncclFloat8e5m2` 时，NCCL 切换到 Hopper 专用 intrinsic 指令。

### 6.3 `ncclReduce` 与 Reduction

- **Reduction**：把每块 GPU 的数据用数学算子（sum/max）合并，结果存到某块（或所有）GPU。
- 旧系统要"水桶接力"式传数据求和；**DGX H100 里 NVSwitch 自己做数学，把工作从 GPU 卸载**。
- 通用 `ncclReduce` 通常用 **NVLSTREE**（NCCL 2.18+ 引入）处理"归约到单 root"的流（尤其跨多节点扩展时）。
- 用途：数据消费者集中（通常是 Rank 0）时用，如**推理最后一层只有 Rank 0 需要完整 logits 做 argmax/top-k**。
- `count` 是每 GPU 贡献的元素数，root 收到恰好 `count` 个元素。

### 6.4 `ncclAllReduce`

- 训练中最重要操作之一：N 块 GPU 合并 → 得全局结果 → 分发给 N 块 GPU。
- 8 块 GPU 同时把本地梯度从 HBM 推上 NVLink 通道、指向 NVSwitch → **交换机做 in-flight 归约** →
  立即把结果**多播**回 8 块 GPU → 结果直接落到各 GPU 的 HBM3 目标缓冲。

**`ncclRedOp_t`（归约算子）**：

| 算子 | H100 上的实现 |
|------|--------------|
| `ncclSum` | **唯一对所有硬件路径都全优化的算子**，全部卸载给 NVSwitch |
| `ncclMin` / `ncclMax` | 也从硬件卸载（FP8） |
| `ncclProd` | NVSwitch 不能做浮点乘法 → **卸载给 Tensor Core** |
| `ncclAvg` | 两段：sum 在 NVSwitch 做、除法在 GPU 做 |

### 6.5 `ncclReduceScatter`

- 每 GPU 从完整梯度缓冲开始，**跨 GPU 求和后再打散**，每 GPU 只拿到最终和向量的一块。
- 输入：每 GPU 有 `[A,B,C,D]`；输出：GPU 0 持 Sum(A)、GPU 1 持 Sum(B)……
- `recvcount` 是**输出缓冲**的元素数（不是总输入），= 总元素 / GPU 数；`Total Elements Reduced = recvcount × nranks`。
- **不支持 "jagged" 数组**：总向量 100、3 GPU 时 `100/3 = 33.33` 分不均——必须**填充到 GPU 数的整数倍**。

### 6.6 `ncclAllGather`

- 每 GPU 从不同片段开始，变换后每 GPU 拿到**完整副本**。
- 8 GPU 同时把梯度推进 NVSwitch 结构，NVSwitch 合并数据并路由到所有 GPU。
- 总总线利用率最大化，更接近理论 900 GB/s 聚合吞吐（不用等 ring 转一圈）。
- `sendcount` = 本 rank 贡献的元素数（本地切片大小）；每 rank 最终在 `recvbuff` 拿到所有 rank 的切片。

### 6.7 `ncclAllToAll` 与 MoE 问题

- 每 GPU 持有"要给其它每块 GPU"的数据并分别传输。
- 与常用复杂 Ring/Tree + SHARP 卸载的 AllReduce 不同，AllToAll **物理上更简单但带宽密集**。
- 逻辑上是 `N×(N−1)` 次点对点（加自身），但作为一个集合操作协调优化。
- **MoE 问题**：NCCL 标准 AllToAll 是为**静态、批量同步**通信设计的，而 MoE 路由本质是**动态、稀疏、延迟敏感**的：
  - NCCL 是 CPU 驱动（Host API）、数据大小静态、要求连续块、且是阻塞的。
  - 专用库 **DeepEP（底层 NVSHMEM）** 更优：**设备发起、细粒度**数据搬移，能处理专家路由的不规则流量而不卡 GPU。

### 6.8 集合通信在 NVSwitch 里的优势小结

- Broadcast：交换机复制，8 GPU 广播 ≈ 1 GPU。
- Reduce/AllReduce：交换机做 in-flight 归约 + 多播回写。
- AllGather：交换机合并 + 路由，总线利用率最大化。

---

## 7. 点对点与分组：ncclSend/Recv 与 ncclGroup

### 7.1 `ncclGroupStart` / `ncclGroupEnd`

- 分布式系统常一个 CPU 线程管理多块 GPU；CPU 发 NCCL 指令时，若是阻塞调用会被卡住。
  多数 NCCL 集合调用（如 `ncclAllReduce`）对 CPU 是**异步**的，更阻塞的往往是 communicator 初始化（`ncclCommInitRank`）。
- 若一个 host 线程逐条给多 GPU 发 NCCL 调用，可能造成**部分参与方已开始、其它还没发出**的中间态 → **死锁**。
- `ncclGroupStart()/ncclGroupEnd()` 让你**批量**一组 NCCL 调用：组内收集、`ncclGroupEnd()` 时一起提交，避免部分启动问题。

### 7.2 `ncclSend` / `ncclRecv`

- **`ncclSend`**：非阻塞地把数据从某 GPU 发到另一 GPU，几乎总与目标设备的 `ncclRecv` 配对。
  - `peer`：目标 GPU 的 rank。
  - NCCL 自动找两 GPU 间最快物理路径（NVLink / PCIe / IB RDMA）。
- **`ncclRecv`**：告诉某 GPU 分配空间、等待指定 `peer` 的数据。**异步**（入队后 CPU 立刻恢复）。
  - 一个 `ncclRecv` 必须有匹配的 `ncclSend`；多个 P2P 传输须在各 GPU 上**按兼容顺序入队**，避免循环依赖。

---

## 8. 总结与学习衔接

### 8.1 核心脉络速记

| 概念 | 一句话 |
|------|--------|
| **Slurm + PMIx** | 资源分配 + exascale 进程发现（Put/Commit/Get，KVS 交换 NCCL ID） |
| **设备隔离** | `CUDA_VISIBLE_DEVICES` + cgroups；调试时错误都指向 GPU 0 |
| **NCCL 初始化** | Unique ID → TCP 连 Rank 0（barrier）→ Rank 0 算 Ring/Tree 路由 → RDMA 握手 |
| **六大原语** | Broadcast / Reduce / AllReduce / AllGather / ReduceScatter / All-to-All |
| **四种并行** | 数据（AllReduce）、张量（Broadcast/AllGather/AllReduce）、流水线（P2P）、专家（All-to-All） |
| **SP 打破 AllReduce** | `AllReduce = ReduceScatter + AllGather`，中间插 LayerNorm/Dropout |
| **NVSwitch 卸载** | Broadcast 复制、Reduce/AllReduce in-flight 归约、ncclAvg 两段 |
| **MoE / P2P** | AllToAll 不适用 → DeepEP/NVSHMEM；ncclGroup 批量、ncclSend/Recv 点对点 |

### 8.2 全课程收束

本文档是 H100 系列（Lesson 1–10）的**最后一课**，把前 9 课的成果串成完整图景：

```
单卡 H100（TMA / Tensor Core / WGMMA 异步化）  →  L1–L7
  → 高性能单 kernel 设计（warp 特化 / 流水线 / 调度 / epilogue）  →  L8
  → 多 kernel 交接（依赖启动）  →  L8.2
  → 多 GPU 硬件互联（NVLink/NVSwitch/Rail 网络）  →  L9
  → 多 GPU 编程（Slurm/PMIx/NCCL/四种并行）  →  L10
```

### 8.3 一句话记忆

> **多 GPU 训练 = Slurm+PMIx 把进程铺满集群 → NCCL 用 NVSwitch 做 in-flight 集合通信 →
> 按模型形态选并行策略（数据/张量/流水线/专家）→ 把 AllReduce 拆成 ReduceScatter+AllGather
> 并在中间融合算子 → 用 NCCL 的六大原语在 900 GB/s 的 mesh 上最大化吞吐。**

---

> 参考来源：`10. Multi GPU Part 2.pdf`（Lesson 10，60 页）。

{% endraw %}
