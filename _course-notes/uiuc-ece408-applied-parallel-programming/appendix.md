---
title: "附录：CUDA 并行模式与优化速查表"
collection: course-notes
chapter: true
permalink: /course-notes/uiuc-ece408-applied-parallel-programming/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/uiuc-ece408-applied-parallel-programming/) · [← l12](/course-notes/uiuc-ece408-applied-parallel-programming/l12)

{% raw %}
# 附录：CUDA 并行模式与优化速查表

> 按类别汇总全书涉及的 CUDA API、性能公式、优化技巧与诊断方法。
> 所有性能数字以 **NVIDIA A100 (sm_80)** 为基准：19.5 TFLOPS FP32、1555 GB/s、机器平衡点 12.5 FLOP/Byte。

---

## 目录
1. [内存优化](#1-内存优化)
2. [线程组织](#2-线程组织)
3. [同步与通信](#3-同步与通信)
4. [warp 级原语](#4-warp-级原语)
5. [七种并行模式速查](#5-七种并行模式速查)
6. [性能分析公式](#6-性能分析公式)
7. [性能诊断流程](#7-性能诊断流程)
8. [CUDA API 速查](#8-cuda-api-速查)
9. [编译器与构建选项](#9-编译器与构建选项)
10. [硬件参数速查](#10-硬件参数速查)
11. [CUDA → WebGPU/WGSL 对照](#11-cuda--webgpuwgsl-对照)
12. [常见陷阱速查](#12-常见陷阱速查)

---

## 1. 内存优化

### 1.1 内存空间对照表

| 空间 | 声明 | 作用域 | 生命周期 | 延迟 | 位置 | 典型容量 |
|---|---|---|---|---|---|---|
| 寄存器 | 自动变量（无修饰） | 单线程 | 线程 | ~1 cycle | SM 寄存器文件 | 255 regs/线程，65536/SM |
| 局部内存 | 自动变量（溢出/动态索引数组） | 单线程 | 线程 | 400–800 cycles | **DRAM**（被 L1/L2 缓存） | 受显存限制 |
| 共享内存 | `__shared__` | block | block | ~20–30 cycles | SM 片上（与 L1 同体） | 48–228 KB/SM |
| 全局内存 | `__device__` / `cudaMalloc` | 全 grid | 应用 | 400–800 cycles | DRAM | 显存容量 |
| 常量内存 | `__constant__` | 全 grid | 应用 | 命中缓存 ~数周期；未命中 ~400+ | DRAM + 常量缓存（8 KB/ SM） | **64 KB 总计** |
| 纹理内存 | `texture<>` / `__ldg` | 全 grid | 应用 | 命中 ~数十周期 | 通过纹理/L1 缓存 | 受显存限制 |

### 1.2 全局内存：合并访问（coalescing）

**核心规则**：一个 warp 的 32 个线程若访问**连续的 4 字节地址**，硬件合并为 **1 次 128 字节事务**。

```text
✅ 合并（coalesced）—— 1 次 128B 事务
warp 线程:   t0   t1   t2   t3  ...  t31
地址:       +0   +4   +8   +12 ...  +124
             └──────────── 128 字节 ────────────┘
                    = 1 transaction

❌ 跨步（strided, stride = N*4 字节）—— 32 次事务
warp 线程:   t0    t1    t2   ...   t31
地址:       +0   +4N  +8N   ...  +124N
             ↓     ↓     ↓           ↓
           [128B][128B][128B] ... [128B]  = 32 transactions (32× 浪费)
```

**访存效率公式**：

```text
访存效率 = (warp 实际需要的字节数) / (warp 实际搬运的字节数)
```

| 访问模式 | warp 请求字节 | 实际搬运字节 | 效率 | 事务数 |
|---|---|---|---|---|
| `A[tid]`（连续 float） | 128 B | 128 B | 100% | 1 |
| `A[tid*2]`（stride 2） | 128 B | 256 B | 50% | 2 |
| `A[tid*32]`（stride 32） | 128 B | 4096 B | 3.1% | 32 |
| `A[tid]`（连续 double） | 256 B | 256 B | 100% | 2（每事务 128B） |

**两条铁律**：
1. **让 `threadIdx.x` 映射到内存中最快的维度**（行主序时即"列"，即最后一个下标）。
2. **二维/三维数据展平后按一维连续寻址**，不要用嵌套索引跳着访问。

### 1.3 共享内存：bank 与 bank conflict

**硬件结构**：共享内存被划分为 **32 个 bank**，每个 bank 宽 **4 字节**。

```text
共享内存地址（字节）:  0   4   8   12  ...  124 | 128  132 ...
bank 编号:             b0  b1  b2  b3  ...  b31 | b0   b1  ...
                        └──── 128 字节 = 一轮 ────┘

warp 内 32 个线程同时访问：
  若 32 个线程落在 32 个不同 bank  → 无冲突，1 个周期完成
  若 k 个线程落在同一个 bank      → k-way conflict，串行 k 个周期
  若 k 个线程访问同一 bank 的同一地址 → 广播（broadcast），仍 1 个周期 ✅
```

**bank 编号公式**：`bank = (字节地址 / 4) % 32`

**经典冲突场景与修复**：

| 场景 | 访问模式 | 冲突 | 修复 |
|---|---|---|---|
| 矩阵乘 `subTileN[k][tx]` | 不同 k、同 tx → 地址 = k*(TILE+pad) + tx | 无（跨行时地址差是 TILE 的倍数，若 TILE=32 则同 bank） | **padding**：`[TILE][TILE+1]` |
| 列访问 `A[i][j]` 固定 j 变 i | 地址 = i*32 + j → 同 bank j | **32-way** ❌ | padding 或改用行访问 |
| 归约顺序寻址 `s[tid] += s[tid+h]` | 地址连续、步长 h | 无 ✅ | — |
| 归约交错寻址 `s[tid] += s[tid+1]` | 线程 0,2,4… → bank 0,2,4…（2-way） | 2-way 起，逐轮恶化 | 改顺序寻址 |

**padding 原理**（TILE_WIDTH = 32 时列访问的经典例子）：

```cpp
__shared__ float A[32][32];      // ❌ 列访问 A[i][0] 全部落 bank 0 → 32-way conflict
__shared__ float A[32][32 + 1];  // ✅ 地址 = i*33 + 0，bank = (i*33)%32 = i → 无冲突
```

**共享内存容量速查**：

|  GPU  | 每 SM 共享内存 | 每 block 最大（需 opt-in） | 默认每 block 上限 |
|---|---|---|---|
| A100 (`sm_80`) | 164 KB | 163 KB | 48 KB |
| H100 (`sm_90`) | 228 KB | 227 KB | 48 KB |
| RTX 4090 (`sm_89`) | 100 KB | 99 KB | 48 KB |
| RTX 2080 Ti (`sm_75`) | 64 KB | 64 KB | 48 KB |

```cpp
// 超过 48 KB 的静态共享内存必须动态分配 + opt-in
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
kernel<<<grid, block, 96*1024>>>(...);          // 第三个参数 = 动态共享内存字节数
extern __shared__ float smem[];                 // kernel 内声明
```

### 1.4 常量内存

```cpp
__constant__ float mask[9];                       // 声明（文件作用域）
cudaMemcpyToSymbol(mask, h_mask, 9*sizeof(float));// 主机端初始化
// kernel 内直接按 mask[j] 访问
```

**常量缓存广播机制**：当 warp 内 32 个线程访问**同一个地址**时，1 个周期完成广播；
若访问**不同地址**，则退化为串行（32 个不同地址 = 32 个周期）❌。
→ **只适合"所有线程读同一个标量"的场景**，典型就是卷积的 mask 系数。

### 1.5 内存优化技巧清单

| 技巧 | 原理 | 收益量级 |
|---|---|---|
| **合并访问** | 32 次访问 → 1 次 128B 事务 | 最多 32× |
| **共享内存 tiling** | 全局访问减少 TILE 倍 | 10–20× |
| **padding 消除 bank conflict** | 32-way → 无冲突 | 最多 32×（共享内存侧） |
| **`__restrict__`** | 告知编译器指针不重叠，允许寄存器缓存与重排 | 5–20% |
| **`const __restrict__`** | 配合上一条，启用只读数据路径（LDG/纹理） | 5–15% |
| **结构体数组 → 数组结构（SoA）** | 避免 warp 内跨步访问结构体成员 | 2–10× |
| **向量化 `float4`** | 每线程 1 次 16B 访存 = 128B/8 线程 | 2–4× |
| **共享内存缓存 + 私有化** | 把全局原子竞争转为片上竞争 | 10–100× |
| **`cp.async`（Ampere+）** | 计算当前 tile 时预取下一个 tile | 10–30% |
| **统一内存 `cudaMemPrefetchAsync`** | 提前迁移页，避免按需分页抖动 | 视场景 |
| **固定内存 `cudaMallocHost`** | 真 DMA 传输，避免 bounce buffer | 传输带宽 2–3× |

---

## 2. 线程组织

### 2.1 线程层次与索引

```text
Grid（整个 kernel 的线程集合）
 ├── Block 0        ├── Block 1        ...  ├── Block gridDim.x-1
 │    ├── Warp 0 (32 threads)                每 block 最多 1024 线程
 │    ├── Warp 1                             每 SM 最多 64 warps (A100/H100)
 │    └── ...                                / 48 warps (Ada) / 32 (Turing)
 └── 每 block 有独立的共享内存

索引（三维）：
  threadIdx.{x,y,z}   ∈ [0, blockDim.{x,y,z})
  blockIdx.{x,y,z}    ∈ [0, gridDim.{x,y,z})
```

**全局线程编号推导**（一维）：
```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
**二维**：
```cpp
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
int idx = row * width + col;              // 行主序展平
```
**三维**：
```cpp
int x = blockIdx.x*blockDim.x + threadIdx.x;
int y = blockIdx.y*blockDim.y + threadIdx.y;
int z = blockIdx.z*blockDim.z + threadIdx.z;
long long idx = (long long)x + (long long)y*W + (long long)z*W*H;  // 注意 64 位！
```

**网格尺寸（向上取整）**：
```cpp
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);            // 1D
dim3 grid2((W + 15)/16, (H + 15)/16);              // 2D，block 16×16
```
⚠️ **整数除法陷阱**：`N / blockDim.x` 会漏掉尾部元素，必须向上取整并在 kernel 内做边界检查。

### 2.2 Block 大小选择

| Block 大小 | 评价 |
|---|---|
| 32 | 仅 1 个 warp/block，SM 上 block 数上限（通常 24–32）会限制并发 warp 数 ❌ |
| 64 / 96 | 可用，但粒度过细，调度开销占比高 |
| **128 / 256** | **最常用**，占用率与调度粒度平衡 ✅ |
| 512 | 适合计算密集型 kernel |
| 1024 | 只有当 kernel 资源消耗很低（寄存器少、无共享内存）时才用；否则占用率骤降 |

**选择准则**：
1. block 大小必须是 **32 的整数倍**（否则最后一个 warp 有闲置 lane）；
2. 让**每个 SM 至少能驻留 8–16 个 block**（便于调度器换出）；
3. 用 `cudaOccupancyMaxPotentialBlockSize()` 自动求解：

```cpp
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
```

### 2.3 线程粗化（thread coarsening）

**定义**：让每个线程计算**多个**输出元素，而非一个。

```text
粗化前（coarsening factor = 1）       粗化后（coarsening factor = 4）
thread t  → output[t]                thread t → output[4t], output[4t+1],
                                                         output[4t+2], output[4t+3]

线程数 = N                            线程数 = N/4
```

**收益来源**：
1. **寄存器级数据复用**：4 个输出共享同一批加载的输入（尤其矩阵乘、卷积）；
2. **减少索引计算与循环开销**；
3. **提高 ILP**：4 条独立累加链可填满流水线，减少对 TLP 的依赖 → 允许更低占用率仍能隐藏延迟。

**代价**：线程数减少 → 若总线程数不足以填满 GPU，则并行度不足；寄存器压力上升可能降低占用率。

### 2.4 网格-步长循环（grid-stride loop）

```cpp
__global__ void addGridStride(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C, int N) {
    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
```

**优点**：
- 线程总数与问题规模**解耦**（grid 可固定为"刚好填满 GPU"的大小）；
- 自动获得粗化收益（每线程多次迭代复用索引计算与寄存器）；
- 天然支持任意 N，且便于做多 GPU 拆分与负载均衡；
- 对**未知规模**的输入特别友好（无需重算 grid）。

---

## 3. 同步与通信

### 3.1 同步原语速查

| API | 作用域 | 语义 | 典型用途 |
|---|---|---|---|
| `__syncthreads()` | block | 屏障：所有线程到达后继续；同时保证共享/全局内存可见性 | tiled matmul 的两次屏障 |
| `__syncwarp(mask)` | warp | warp 内线程同步 + 内存栅栏 | Volta+ 上 warp 内通信前 |
| `__threadfence()` | 设备 | 保证本线程之前的写对全设备可见（不阻塞） | 全局内存生产者-消费者 |
| `__threadfence_block()` | block | 同上，block 范围 | — |
| `cudaDeviceSynchronize()` | host↔device | 阻塞主机直到所有先前工作完成 | kernel 后计时/取回结果 |
| `cudaStreamSynchronize(s)` | host↔stream | 只等某个流 | 多流流水线 |
| `cudaEventSynchronize(e)` | host↔event | 等到某事件 | 精细计时 |
| `cudaStreamWaitEvent(s,e)` | stream↔stream | 流 s 等事件 e | 跨流依赖 |

### 3.2 `__syncthreads()` 三条铁律

1. **必须被 block 内所有线程执行**——不能放在 `if (tid < n)` 之类的分支里，否则死锁或未定义行为。
   ```cpp
   // ❌ 错误
   if (tid < 128) { doWork(); __syncthreads(); }
   // ✅ 正确
   if (tid < 128) { doWork(); }
   __syncthreads();
   ```
2. **不要放在可能提前 return 的路径之后**（除非所有线程都走同一路径）。
3. **它只同步不互斥**——不能用来保护临界区，只保证"所有线程都到齐 + 内存可见"。

### 3.3 共享内存数据竞争的三种形态

在 tiled 循环中，每个 tile 迭代必须用**两次**屏障夹住：

```cpp
for (int m = 0; m < numTiles; ++m) {
    As[ty][tx] = A[...];          // 写共享内存
    Bs[ty][tx] = B[...];
    __syncthreads();              // 屏障 1：等所有人写完
    for (int k = 0; k < TILE; ++k) sum += As[ty][k] * Bs[k][tx];
    __syncthreads();              // 屏障 2：等所有人读完，才能覆盖
}
```

| 省略的屏障 | 后果 |
|---|---|
| 屏障 1 | **RAW 竞争**：读到别人还没写入的旧值 → 结果错误 |
| 屏障 2 | **WAR 竞争**：下一轮写入覆盖了别人还没读完的数据 → 结果错误（且随机） |

> 记忆法：屏障数 = 2 ×（tile 迭代数），**少一个都会算错**。

### 3.4 原子操作

| API | 语义 | 支持的精度 |
|---|---|---|
| `atomicAdd(addr, val)` | `*addr += val` | int, uint, ull, float, double, half2 |
| `atomicSub` / `atomicExch` | 减 / 交换 | int, uint, ull, float |
| `atomicMin` / `atomicMax` | 最小 / 最大 | int, uint, ull, longlong |
| `atomicInc` / `atomicDec` | 环回加减 | uint |
| `atomicCAS(addr, cmp, val)` | 比较并交换 | 全部（用来实现任意原子操作） |
| `atomicAnd/Or/Xor` | 位运算 | int, uint, ull |

**性能特征**：同一地址的原子操作被**硬件串行化**。
`N` 个线程对同一地址做 `atomicAdd` → `N` 个串行周期量级。

**私有化（privatization）模式**（直方图/归约的标准手法）：

```cpp
__global__ void histogramPrivatized(const unsigned char* img, unsigned int* hist, int n) {
    __shared__ unsigned int localHist[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) localHist[i] = 0;
    __syncthreads();

    int stride = gridDim.x * blockDim.x;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride)
        atomicAdd(&localHist[img[i]], 1u);        // 竞争被限制在片上，快 ~100×

    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        atomicAdd(&hist[i], localHist[i]);        // 全局原子次数 = 256 × #blocks（极少）
}
```

---

## 4. warp 级原语

### 4.1 shuffle 指令（`sm_30+`，Volta 后必须带 mask）

```cpp
// 从 lane (laneID + delta) 取值；若越界则返回本线程自己的值
__shfl_down_sync(mask, var, delta, width=32)
// 从 lane (laneID - delta) 取值
__shfl_up_sync(mask, var, delta, width=32)
// 从任意 lane srcLane 取值
__shfl_sync(mask, var, srcLane, width=32)
// 蝴蝶交换：laneID ^ laneMask
__shfl_xor_sync(mask, var, laneMask, width=32)
```

| 原语 | 作用 |
|---|---|
| `__ballot_sync(mask, pred)` | 返回 32 位掩码，第 i 位 = lane i 的谓词 |
| `__any_sync` / `__all_sync` | warp 内任一 / 全部为真 |
| `__activemask()` | 当前活跃 lane 的掩码（**谨慎使用**，语义微妙） |
| `__match_any_sync` / `__match_all_sync` | 找出值相等的 lane（`sm_70+`） |
| `__reduce_add_sync` / `__reduce_min_sync` / `__reduce_max_sync` | **硬件归约**（`sm_80+`，int/uint only） |

### 4.2 warp 内归约的标准写法

```cpp
// 前 5 轮（32→1）用 shuffle，无需 __syncthreads()，无需共享内存
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;              // lane 0 持有 warp 总和
}

// block 级：每 warp 归约后由 lane 0 写共享内存（每 warp 只写 1 个元素 → 无 bank conflict）
__device__ float blockReduceSum(float val) {
    __shared__ float warpSums[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) warpSums[wid] = val;
    __syncthreads();

    int nWarps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < nWarps) ? warpSums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}
```

**为什么快**：
- shuffle 不访问内存，1 个周期完成一次交换，**完全不占共享内存带宽**；
- 每 warp 只写 1 个值到共享内存 → **零 bank conflict**；
- 相比"全程用共享内存"版本，共享内存访问次数减少约 `log2(32) = 5` 倍。

⚠️ **mask 正确性**：mask 必须精确描述参与该操作的 lane 集合；在 Volta+ 上
independent thread scheduling 使得"隐式 warp 同步"不再成立，必须用 `_sync` 变体。

---

## 5. 七种并行模式速查

| 模式 | 核心思想 | 关键 CUDA 机制 | 算术强度 | 瓶颈类型 | 首要优化 |
|---|---|---|---|---|---|
| **Element-wise**（向量加法） | 1 线程 1 元素 | 一维映射 + 边界检查 | 0.125 | 带宽 | 合并访问、`float4`、grid-stride |
| **Dense matmul（朴素）** | 1 线程 1 输出，直接读全局 | 二维映射 | 0.25 | 带宽（跨步访存） | 转置 B / tiling |
| **Tiled matmul** | tile 载入共享内存复用 | `__shared__` + 2×`__syncthreads()` | TILE/4 = 4–8 | 从带宽转计算 | 大 TILE、padding、粗化 |
| **Reduction** | 树形两两合并 | 共享内存树 / shuffle | ~0.06 | 带宽 | 顺序寻址、warp shuffle、粗化 |
| **Scan** | 双缓冲对数步长前缀和 | 双缓冲 + 多轮屏障 | ~0.1 | 带宽 | 层次化、work-efficient 权衡 |
| **Convolution（tiled）** | 共享内存 halo 复用 | `__constant__` + halo 加载 | ~1（朴素）→ TILE/8 | 带宽/计算混合 | 常量内存、halo、粗化 |
| **SpMV（CSR）** | 每行一个线程 | 压缩格式 + gather | ~0.167 | 带宽 + 不规则 | ELL/向量化、x 入共享内存 |

### 5.1 各模式的"全局访存减少量"

| 模式 | 朴素版每输出元素访存 | 优化版 | 减少倍数 |
|---|---|---|---|
| Tiled matmul (N×N) | `2N` | `2N / TILE_WIDTH` | **TILE_WIDTH×**（16 或 32） |
| Tiled convolution (1D, mask r) | `2r+1` | `(TILE+2r) / TILE` | 约 **TILE/(2r+1)** |
| Reduction | `N` 次读（1 次/元素） | 同（无法减少） | 1×（本就最优，靠延迟隐藏） |
| SpMV | — | — | 靠格式规则化而非减少字节数 |

### 5.2 tiled 矩阵乘法：访存减少的定量推导

```text
朴素版：输出 C 的每个元素需要
  - 读 A 的一整行：K 次
  - 读 B 的一整列：K 次
  合计 2K 次全局访存 / 输出元素
  → 总量 = 2·M·N·K 次

分块版（TILE_WIDTH = T）：
  - C 被划分为 (M/T)·(N/T) 个 tile，每个 tile 由 1 个 block 计算
  - 每个 block 需要载入 A 的 M/T... 具体为 (M/T)·(K/T) 个 A-tile + (K/T)·(N/T) 个 B-tile
  - 每个 tile 载入一次，被 block 内 T² 个线程各使用 T 次
  → 每个输出元素的全局访存 = 2K / T
  → 总量 = 2·M·N·K / T

算术强度 = FLOPs / Bytes
        = 2·M·N·K / (4·(2·M·N·K/T))     [4 字节/float]
        = T / 4  FLOP/Byte

  T = 16 → 4.0 FLOP/Byte   （低于 A100 平衡点 12.5 → 仍带宽受限）
  T = 32 → 8.0 FLOP/Byte   （接近平衡点）
  T = 64 → 16.0 FLOP/Byte  （超过平衡点 → 计算受限，但共享内存不够）
```

### 5.3 归约：warp 发散对比表

设 block = 256 线程，用共享内存归约，`tid` = 线程在 block 内的编号。

| 轮次 | 交错寻址 `s += s[tid + stride]`（stride = 2^r） | 顺序寻址 `s += s[tid + h]`（h = blockDim/2^r） |
|---|---|---|
| | 活跃线程（`tid % (2*stride) == 0`） | 活跃线程（`tid < h`） |
| 1 | 128（warp 0–3 活跃，但**每 warp 内部只有一半 lane 活跃 → 50% 发散**） | 128（warp 4–7 全部空闲，warp 0–3 **每 lane 都活跃 → 无发散**） |
| 2 | 64（发散 75%） | 64 |
| 3 | 32（发散 87.5%） | 32 |
| 4 | 16（只剩 1/2 warp，浪费 31/32 执行槽） | 16 |
| 5 | 8 | 8 |
| 6 | 4 | 4 |
| 7 | 2 | 2 |
| 8 | 1 | 1 |

**结论**：顺序寻址让"每轮被砍掉的是一整个 warp"，保留的 warp 内部完全整齐 → **无分歧、访存合并**。
加上 warp shuffle 后，前 5 轮完全不访问内存。

### 5.4 扫描：Hillis-Steele vs Blelloch

| 维度 | Hillis-Steele (Kogge-Stone) | Blelloch (work-efficient) |
|---|---|---|
| 步数（深度） | `log2(N)` | `2·log2(N)`（up-sweep + down-sweep） |
| 工作量 | `O(N log N)` | `O(N)` |
| 并行度 | 每步 `N/2` 个活跃操作 | up-sweep 递减，down-sweep 递增 |
| 输出 | inclusive scan（直接得） | exclusive scan（天然） |
| 共享内存 | 需**双缓冲**避免 RAW/WAR | 原地可做（up/down 两阶段） |
| GPU 上偏好 | ✅ **通常更快**（ALU 富余，带宽是瓶颈） | 理论更优雅，但步数多、同步多 |
| 适用 | 单 block、数据量适中 | 大 N、需 work efficiency 的场合 |

**多 block 层次化扫描（三步法）**：
```text
Step 1: 每个 block 独立扫描自己的 chunk，把 chunk 总和写入 blockSums[b]
Step 2: 对 blockSums 做一次扫描（单个 block 完成，或递归）
Step 3: 每个 block 把 blockSums 的前缀和（exclusive）加到自己的结果上
```
总访存 = 读 2 遍 + 写 2 遍 = `4N×4` 字节（N 个 float）→ 理论时间下限 `≈ 16N 字节 / 1555 GB/s`。

---

## 6. 性能分析公式

### 6.1 核心公式集

```text
【算术强度】
  Arithmetic Intensity (AI) = FLOPs executed / Bytes transferred
                            [FLOP/Byte]

【机器平衡点】
  Machine Balance = Peak FLOPS / Peak Bandwidth        [FLOP/Byte]
  A100: 19.5e12 / 1555e9 = 12.5 FLOP/Byte

【Roofline 性能上限】
  Attainable FLOPS = min( Peak FLOPS ,  AI × Peak Bandwidth )

  AI <  Machine Balance  →  带宽受限（memory bound）
  AI >  Machine Balance  →  计算受限（compute bound）

【有效带宽】
  Effective BW = (Bytes actually transferred) / (kernel time)   [GB/s]
  % of peak   = Effective BW / Peak Bandwidth × 100%

【加速比】
  Speedup = T_serial / T_parallel
  Amdahl: S(n) = 1 / ( (1-p) + p/n )     p = 可并行比例

【占用率】
  Occupancy = (active warps per SM) / (max warps per SM) × 100%
  限制因素取以下最小值：
    regs:      floor(65536 / (regs_per_thread × blockDim)) × blockDim / 2048
    smem:      floor(smem_per_SM / smem_per_block) × blockDim / 2048
    blocks:    max_blocks_per_SM × blockDim / 2048
    warps:     max_warps_per_SM / 2048 × 2048   （A100 = 64 warps = 2048 threads）

【Little's Law（延迟隐藏所需并发度）】
  Required Concurrency = Latency × Throughput
  例：DRAM 延迟 500 cycles，SM 每周期可发 1 次访存
      → 需要约 500 个在途访存才能打满带宽
      → 若每线程 1 个在途访存，需要 ≈ 500 线程 ≈ 16 warps 常驻
```

### 6.2 Roofline 图（ASCII，基准机 A100）

```text
Performance
(GFLOP/s, log)
 100000 ┤
        │                                          ╱ 计算屋顶 = 19500 GFLOP/s
  19500 ┤────────────────────────────────────────╱────────────────────
        │                                    ╱          (FP32 峰值)
        │                                ╱
        │                            ╱     ← 拐点 AI = 12.5
        │                        ╱
   1000 ┤                    ╱   tiled matmul T=32 (AI=8)  ● 
        │                ╱     
        │            ╱   tiled matmul T=16 (AI=4)  ● 
    389 ┤────────╱─── naive matmul (AI=0.25)  ● 
        │      ╱ ╱   conv naive (AI≈1)  ●
    260 ┤    ╱  ╱    SpMV CSR (AI≈0.167)  ●
    194 ┤  ╱  ╱      vecAdd (AI=0.125)  ●
        │╱  ╱        reduction (AI≈0.06)  ●
     10 ┤  ╱
        └──┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴────→ AI (FLOP/Byte)
          0.06  0.125  0.25   1    4    8   12.5  32    64
           └──────── 带宽受限区 ────────┘└── 计算受限区 ──┘

  内存屋顶 = 1555 GB/s（斜率 = 带宽）
  带宽受限区内的上限 = AI × 1555 GFLOP/s
```

**读图要点**：本课程几乎所有的 kernel 都落在**带宽受限区**（左侧斜坡）。
这就是为什么全书的优化主线是"减少字节数、让字节规律化"，而不是"减少浮点运算次数"。

### 6.3 各模式的理论性能上限（A100）

| Kernel | AI (FLOP/Byte) | 理论上限 (GFLOP/s) | 占 FP32 峰值 |
|---|---|---|---|
| Reduction | 0.06 | 93 | 0.5% |
| vecAdd | 0.125 | 194 | 1.0% |
| SpMV (CSR) | 0.167 | 260 | 1.3% |
| Naive matmul | 0.25 | 389 | 2.0% |
| Convolution (naive) | ~1.0 | 1555 | 8.0% |
| Tiled matmul T=16 | 4.0 | 6220 | 32% |
| Tiled matmul T=32 | 8.0 | 12440 | 64% |
| Tiled matmul T=64 | 16.0 | 19500（计算受限） | 100% |

> ⚠️ 上表是**理想上限**（假设 100% 带宽利用、无冲突、无延迟损失）。实际达到 60–80% 已属优秀。
> 注意：**低 AI 不代表"这个 kernel 没价值"**——它只说明性能天花板低，
> 而优化目标应定为"尽可能逼近自己的天花板"（见 6.4 的达成率指标）。

### 6.4 衡量优化好坏的三个指标

```text
1. 带宽达成率 = 实测带宽 / 峰值带宽      ← 带宽受限 kernel 看这个
2. 算力达成率 = 实测 GFLOP/s / 峰值      ← 计算受限 kernel 看这个
3. Roofline 达成率 = 实测 / min(峰值, AI×带宽)  ← 通用，最推荐
   目标：> 60% 为良好，> 80% 为优秀
```

---

## 7. 性能诊断流程

### 7.1 六步诊断法

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1  建立正确基线                                                │
│   · CPU golden reference + 容差比对（相对误差 < 1e-4 常用）          │
│   · 先跑通，再跑快。正确性不过关，性能数字毫无意义。                  │
├─────────────────────────────────────────────────────────────────────┤
│ Step 2  测量（务必先预热 + 多次取平均）                              │
│   · cudaEvent 计时，排除首次启动开销与 H2D/D2H 拷贝                  │
│   · 记录：耗时、有效带宽 GB/s、GFLOP/s                              │
├─────────────────────────────────────────────────────────────────────┤
│ Step 3  计算算术强度 → 定位 Roofline 区域                            │
│   · AI = FLOPs / Bytes，与 12.5 FLOP/Byte 比较                      │
│   · 得出"天花板 GFLOP/s"和"当前达成率"                              │
├─────────────────────────────────────────────────────────────────────┤
│ Step 4  检查资源使用                                                 │
│   · nvcc --ptxas-options=-v  → 寄存器数、共享内存、spill             │
│   · cudaOccupancyMaxActiveBlocksPerMultiprocessor → 理论占用率       │
│   · 寄存器 > 64/线程 或 共享内存吃满 → 占用率是嫌疑犯                │
├─────────────────────────────────────────────────────────────────────┤
│ Step 5  Profiler 定位 stall 原因                                     │
│   · ncu --set full ./prog                                           │
│   · 关键指标：dram__throughput.avg.pct_of_peak_sustained_elapsed     │
│              l1tex__data_bank_conflicts_pipe_lsu_mem_shared          │
│              smsp__average_warps_issue_stalled_{long_scoreboard,     │
│                short_scoreboard, barrier, branch_resolving}          │
│                                                                      │
│   stall 原因 → 对策：                                                │
│     long_scoreboard  (等全局内存) → 提高占用率/预取/改善合并         │
│     short_scoreboard (等共享内存) → 消除 bank conflict               │
│     barrier         (等屏障)       → 平衡 warp 工作量、减少同步次数   │
│     branch_resolving(分支发散)     → 消除分歧、重构数据布局           │
│     not_selected    (调度器没选它) → 说明已饱和，可接受              │
├─────────────────────────────────────────────────────────────────────┤
│ Step 6  一次只改一个变量 → 回到 Step 2，记录到优化日志               │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 瓶颈类型 → 优化手段映射

| 瓶颈判定依据 | 瓶颈类型 | 首选优化手段 |
|---|---|---|
| 带宽达成率 > 70%，AI < 平衡点 | **带宽受限** | 减少字节数（tiling/复算）、改善合并、压缩数据格式（FP16/量化）、向量化 |
| 算力达成率 > 70%，AI > 平衡点 | **计算受限** | 减少冗余运算、使用更快的指令（FMA/`__fmaf_rn`）、降低精度、Tensor Core |
| 带宽/算力达成率都低，stall = long_scoreboard | **延迟受限** | 提高占用率、增加 ILP/粗化、软件预取（`cp.async`）、增加在途访存 |
| bank conflict 指标高 | **共享内存冲突** | padding、改变访问模式 |
| barrier stall 高 | **同步开销** | 减少 `__syncthreads()` 次数、warp shuffle 替代、平衡负载 |
| 理论占用率低（< 25%） | **资源限制** | `__launch_bounds__` 限寄存器、减小 tile、`-maxrregcount` |
| 占用率高但仍慢 | **ILP 不足 / 访存不规则** | 线程粗化增加寄存器复用、改数据布局（SoA）、格式规则化（CSR→ELL） |

### 7.3 优化日志模板（最终项目报告的核心素材）

| # | 优化内容 | 实测耗时 | GFLOP/s | 带宽达成率 | 相对上一版加速比 | 瓶颈判断 | 结论 |
|---|---|---|---|---|---|---|---|
| 0 | 基线（CPU 串行参考） | 1250 ms | 0.7 | — | 1.00× | 计算受限 | 建立正确性基准 |
| 1 | 朴素 GPU kernel | 18.5 ms | 46 | 12% | 67.6× | 带宽（跨步访存） | 跨步访问是主因 |
| 2 | + 转置 B 使访问合并 | 4.2 ms | 203 | 52% | 4.4× | 带宽 | 合并访存收益显著 |
| 3 | + 共享内存 tiling T=16 | 1.9 ms | 449 | — | 2.2× | 带宽（AI 提升到 4） | tiling 有效 |
| 4 | + padding 消 bank conflict | 1.5 ms | 569 | — | 1.27× | 共享内存冲突 | 冲突确实存在 |
| 5 | + TILE=32 且粗化因子 4 | 0.85 ms | 1004 | — | 1.76× | 转向计算受限 | 寄存器复用生效 |
| 6 | + `__restrict__` + `-O3` | 0.79 ms | 1080 | — | 1.08× | — | 边际收益 |
| 7 | 尝试 `cp.async` 预取 | 0.81 ms | 1053 | — | 0.98× | — | **无提升，回退**（诚实记录） |

> 最后一行正是官方目标 C17（identify limitations）与 C16（justify the final decision）所要求的：
> **失败的尝试也要如实记录**，它证明了设计空间探索的真实性。

---

## 8. CUDA API 速查

### 8.1 内存管理

```cpp
// 设备内存
cudaMalloc(&d_ptr, bytes);
cudaFree(d_ptr);
cudaMemset(d_ptr, 0, bytes);

// 主机内存
cudaMallocHost(&h_ptr, bytes);        // 固定内存（page-locked），支持真 DMA，带宽 2-3×
cudaFreeHost(h_ptr);

// 统一内存（按需分页）
cudaMallocManaged(&ptr, bytes);
cudaMemPrefetchAsync(ptr, bytes, deviceId, stream);   // 提前迁移，避免页错误抖动

// 二维/三维
cudaMallocPitch(&d_ptr, &pitch, widthBytes, height);
cudaMalloc3D(&d_ptr, make_cudaExtent(w, h, d));

// 符号（__constant__ / __device__ 变量）
cudaMemcpyToSymbol(symbol, h_src, bytes);
cudaMemcpyFromSymbol(h_dst, symbol, bytes);
```

### 8.2 数据传输

```cpp
cudaMemcpy(d_dst, h_src, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_dst, d_src, bytes, cudaMemcpyDeviceToHost);
cudaMemcpy(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice);
cudaMemcpyAsync(d_dst, h_src, bytes, cudaMemcpyHostToDevice, stream);  // 需固定内存才真异步
cudaMemcpy2D(...); cudaMemcpy3D(...);
```

⚠️ **方向写错的代价**：`cudaMemcpyHostToDevice` 写成 `DeviceToHost` 不会报错但结果全错
（在统一寻址的 GPU 上尤其隐蔽）。**永远检查返回值**。

### 8.3 Kernel 执行与查询

```cpp
kernel<<<grid, block, sharedBytes, stream>>>(args...);

cudaGetLastError();                        // 检查启动参数错误（同步，开销小）
cudaPeekAtLastError();                     // 不重置错误
cudaDeviceSynchronize();                   // 等待全部完成
cudaStreamSynchronize(stream);
cudaGetErrorString(err);

// 设备查询
cudaGetDeviceCount(&count);
cudaGetDeviceProperties(&prop, 0);
//   prop.name, prop.major/.minor, prop.multiProcessorCount,
//   prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock,
//   prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlock,
//   prop.regsPerMultiprocessor, prop.warpSize,
//   prop.memoryClockRate (kHz), prop.memoryBusWidth (bits),
//   prop.totalGlobalMem, prop.l2CacheSize, prop.concurrentKernels

// 理论峰值带宽计算
double bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;  // GB/s

// kernel 资源
cudaFuncAttributes attr; cudaFuncGetAttributes(&attr, myKernel);
//   attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes, attr.maxThreadsPerBlock

// 占用率
int numBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, myKernel, blockSize, dynamicSmem);
int minGrid, blockSizeOpt;
cudaOccupancyMaxPotentialBlockSize(&minGrid, &blockSizeOpt, myKernel, dynamicSmemFn, 0);
```

### 8.4 流与事件

```cpp
cudaStream_t s; cudaStreamCreate(&s);
cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
cudaStreamDestroy(s);

cudaEvent_t e0, e1;
cudaEventCreate(&e0); cudaEventCreate(&e1);
cudaEventRecord(e0, s);
kernel<<<g,b,0,s>>>(...);
cudaEventRecord(e1, s);
cudaEventSynchronize(e1);
float ms; cudaEventElapsedTime(&ms, e0, e1);   // 毫秒

cudaStreamWaitEvent(s2, e1, 0);                // 流间依赖
```

### 8.5 标准错误检查宏

```cpp
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                              \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// kernel 启动后立即检查（异步错误在下次同步时才会浮现）
kernel<<<g, b>>>(...);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());
```

---

## 9. 编译器与构建选项

### 9.1 常用 nvcc 选项

| 选项 | 作用 |
|---|---|
| `-O3` | 主机端优化（默认 `-O3` 对设备代码生效） |
| `-arch=sm_80` | 只生成 sm_80 的 SASS（编译快，只能在该架构跑） |
| `-gencode arch=compute_80,code=sm_80` | 精确控制 PTX/SASS 生成 |
| `-arch=native` | 自动匹配本机 GPU（CUDA 11.5+） |
| `--ptxas-options=-v` | **打印寄存器数、共享内存、spill** —— 诊断必用 |
| `-Xptxas -dlcm=cg` | 全局访存绕过 L1（只走 L2） |
| `-Xptxas -dlcm=ca` | 全局访存经 L1 缓存 |
| `--maxrregcount=N` | 限制每线程寄存器数（可能引入 spill） |
| `-lineinfo` | 生成行号信息，供 profiler 使用 |
| `-G` | 设备端调试信息（**严重降低性能，仅调试用**） |
| `--use_fast_math` | 用低精度快速数学函数（**会损失精度，谨慎**） |
| `-std=c++17` | C++ 标准 |
| `--generate-line-info -lineinfo` | Profiler 建议 |
| `-Xcompiler -fopenmp` | 传递选项给主机编译器 |
| `-ccbin g++-12` | 指定主机编译器 |

### 9.2 `__launch_bounds__`

```cpp
// 告诉编译器：block 最多 256 线程，且希望每 SM 至少驻留 4 个 block
__global__ void __launch_bounds__(256, 4) myKernel(...);

// 作用：编译器据此约束寄存器分配（256×4 = 1024 线程 → 最多 64 regs/线程）
// 代价：可能产生寄存器 spill（用 -Xptxas -v 检查）
```

### 9.3 性能相关编译标志速查

```bash
nvcc -O3 -arch=sm_80 -lineinfo --ptxas-options=-v kernel.cu -o kernel
ncu --set full --launch-count 1 ./kernel          # Nsight Compute 全指标
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./kernel
nsys profile --stats=true ./kernel                # 时间线分析（含 H2D/D2H）
```

---

## 10. 硬件参数速查

### 10.1 计算能力（compute capability）

| CC | 架构 | 代表 GPU | 每 SM warp 上限 | 每 SM 线程上限 | 每 block 线程上限 | 共享内存/SM | 关键特性 |
|---|---|---|---|---|---|---|---|
| 7.5 | Turing | RTX 2080 Ti, T4 | 32 | 1024 | 1024 | 64 KB | 独立线程调度 |
| 8.0 | Ampere (GA100) | A100 | 64 | 2048 | 1024 | 164 KB | `cp.async`, TF32, `__reduce_*_sync` |
| 8.6 | Ampere (GA10x) | RTX 3090, A10 | 48 | 1536 | 1024 | 100 KB | 同上（少部分差异） |
| 8.9 | Ada | RTX 4090, L40 | 48 | 1536 | 1024 | 100 KB | FP8 |
| 9.0 | Hopper | H100 | 64 | 2048 | 1024 | 228 KB | TMA, thread block cluster, DPX |

### 10.2 内存层次数量级（A100）

| 层次 | 容量 | 延迟 | 带宽 |
|---|---|---|---|
| 寄存器文件 | 108 SM × 256 KB = 27 MB | 1 cycle | — |
| 共享内存/L1 | 108 × 164 KB ≈ 17.7 MB | 20–30 cycles | ~19 TB/s 聚合 |
| L2 | 40 MB | ~200 cycles | ~5–7 TB/s |
| HBM2e (80GB) | 80 GB | 400–800 cycles | 1555 GB/s（实测 ~1.4 TB/s 可达） |

### 10.3 事务与粒度

| 项目 | 数值 |
|---|---|
| 全局内存事务粒度 | 128 字节（一个 warp 的 32×4B 连续访问） |
| L2 cache line | 128 字节 |
| DRAM burst | 32 字节（这是"访问 4 字节却搬 32 字节"浪费的根源） |
| 共享内存 bank 宽度 | 4 字节 × 32 banks = 128 字节/轮 |
| 常量缓存 | 8 KB/SM，广播 1 周期 |

---

## 11. CUDA → WebGPU/WGSL 对照

ECE408 近年引入 **WebGPU**（Lab）与 **RAI**（最终项目）作为 CUDA 之外的编程路径。
以下是逐项对照，便于有 CUDA 基础者迁移。

| 概念 | CUDA | WebGPU / WGSL |
|---|---|---|
| 计算程序 | `__global__ void kernel(...)` | `@compute @workgroup_size(x,y,z) fn main(...)` |
| 线程层次 | grid → block → thread | dispatch → workgroup → invocation |
| 线程索引 | `blockIdx`, `threadIdx` | `@builtin(global_invocation_id)`, `@builtin(local_invocation_id)` |
| 工作组索引 | `blockIdx` | `@builtin(workgroup_id)` |
| 组内线程数 | `blockDim.x`（≤1024） | `@workgroup_size`（≤256 常见） |
| 调度 | `kernel<<<grid, block>>>()` | `computePass.dispatchWorkgroups(gx, gy, gz)` |
| warp / SIMT | warp = 32 线程 | subgroup（大小由实现决定，常为 32/64） |
| 片上内存 | `__shared__ float s[N];` | `var<workgroup> s: array<f32, N>;` |
| 组内同步 | `__syncthreads()` | `workgroupBarrier()` |
| 全局内存 | `cudaMalloc` + raw pointer | `device.createBuffer()` + `GPUBuffer` |
| 读写缓冲 | 指针解引用 `A[i]` | `var<storage, read_write> A: array<f32>;` |
| 常量/参数 | `__constant__` / kernel 参数 | `var<uniform>` / `var<storage, read>` |
| 数据上传 | `cudaMemcpy` | `device.queue.writeBuffer()` |
| 数据回读 | `cudaMemcpy` D2H | `GPUBuffer` + `mapAsync` |
| 异步并行 | stream + `cudaMemcpyAsync` | queue + 多个 command encoder |
| 原子操作 | `atomicAdd(&x, v)` | `atomicAdd(&x, v)` |
| 计时 | `cudaEvent` | `performance.now()` + `queue.onSubmittedWorkDone()` |
| 错误检查 | `cudaGetErrorString` | `device.pushErrorScope()` / `popErrorScope()` |
| 着色器语言 | CUDA C++ | **WGSL**（WebGPU Shading Language） |

**WGSL 向量加法 compute shader**（与 CUDA 版本逐行对照）：

```wgsl
// 文件: vector_add.wgsl
@group(0) @binding(0) var<storage, read>       A : array<f32>;   // 对应 const float* A
@group(0) @binding(1) var<storage, read>       B : array<f32>;   // 对应 const float* B
@group(0) @binding(2) var<storage, read_write> C : array<f32>;   // 对应 float* C

// 对应 CUDA 的 kernel 参数 N（统一变量）
struct Params { N : u32 };
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(256)                       // 对应 <<<grid, 256>>>
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {   // 对应 blockIdx*blockDim + threadIdx
    let i = gid.x;                                  // 全局线程编号
    if (i >= params.N) { return; }                  // 边界检查（与 CUDA 完全一致）
    C[i] = A[i] + B[i];                             // 合并访存（同样关键）
}
```

**关键差异**（迁移时必须注意）：
1. **WGSL 是强类型且无指针**——用 `array<f32>` + 索引，不能做指针算术；
2. **没有 `__syncthreads()` 的等价全局屏障**——`workgroupBarrier()` 只同步工作组内；
3. **绑定（binding）是显式的**——每个 buffer 必须在 `@group(n) @binding(m)` 中声明，
   并在主机端用 `GPUBindGroupLayout` 精确匹配；
4. **workgroup 大小上限通常为 256**（比 CUDA 的 1024 小）；
5. **没有 `cudaMemcpy` 的同步语义**——所有传输都通过 queue 命令，天然异步，
   必须用 `mapAsync` 才能回读结果；
6. **整数除零/越界行为不同**——WGSL 有明确的边界检查语义，越界索引返回 0/NULL 而非未定义行为。

---

## 12. 常见陷阱速查

| # | 陷阱 | 症状 | 正确做法 |
|---|---|---|---|
| 1 | 忘记 `cudaDeviceSynchronize()` / 事件同步 | 计时错误（测得接近 0）或读到未完成的结果 | kernel 后 `CUDA_CHECK(cudaGetLastError())` + 计时时同步 |
| 2 | 未检查 CUDA API 返回值 | 静默失败，结果全 0 或全 NaN | 全程使用 `CUDA_CHECK` 宏 |
| 3 | 忘记 `__syncthreads()` | 结果随机错误 | tiled 循环中每个 tile 两次屏障 |
| 4 | `__syncthreads()` 放在分支内 | 挂死或未定义行为 | 移到分支外，让所有线程都执行 |
| 5 | 共享内存 bank conflict | 性能仅为预期 1/2 – 1/32 | padding `[N][N+1]`，或改访问模式 |
| 6 | 全局内存跨步访问（未合并） | 带宽达成率 < 15% | 让 `threadIdx.x` 对应最内层维度 |
| 7 | 整数除法导致网格覆盖不足 | 结果尾部元素未计算（常为 0） | `(N + block - 1) / block` |
| 8 | 缺少边界检查 | 段错误或结果错乱 | kernel 内 `if (idx < N)` |
| 9 | `cudaMemcpy` 方向写错 | 无报错但结果全错 | 核对 `cudaMemcpyHostToDevice` / `DeviceToHost` |
| 10 | host 指针传给 kernel | 段错误或 `invalid device pointer` | 只传 `cudaMalloc` 得到的设备指针 |
| 11 | 共享内存超限（>48KB 静态） | 启动失败 `invalid argument` | 动态共享内存 + `cudaFuncSetAttribute` opt-in |
| 12 | 占用率过低（寄存器爆表） | 延迟无法隐藏，性能差 | `-Xptxas -v` 查看，用 `__launch_bounds__` 约束 |
| 13 | 占用率盲目求高 | 性能反而下降（ILP 被牺牲） | 30–50% 常已足够；优先保 ILP |
| 14 | warp 发散 | 分支两侧串行执行 | 重构数据布局，或用谓词化（predication） |
| 15 | 原子操作同地址竞争 | 性能随 N 线性变差 | **私有化**：先片上累加再合并 |
| 16 | 浮点累加顺序不同 | 结果与 CPU 参考不一致 | 用容差比对（相对误差 1e-4），而非精确相等 |
| 17 | `-G` 调试编译用于性能测试 | 性能差 10–100× | 性能测试必须去掉 `-G` 和 `-lineinfo` 之外的调试选项 |
| 18 | 三维索引未用 64 位 | 大数组索引溢出为负 → 越界 | `long long idx = x + (long long)y*W + ...` |
| 19 | 计时把 H2D/D2H 算进去 | 低估 kernel 性能 | 只对 kernel 前后打 event |
| 20 | 首次启动开销未排除 | 大 kernel 尚可，小 kernel 严重失真 | 预热 1–3 次后再计时 |
| 21 | `--use_fast_math` 悄悄降精度 | 容差测试失败 | 正确性验证时关闭；确认可接受后再开 |
| 22 | 网格-步长循环里忘了 `+= stride` 写成 `+= 1` | 重复计算，性能与结果皆错 | 严格检查循环增量 |
| 23 | 共享内存数组未初始化就累加 | 垃圾值参与运算 | 显式清零 + `__syncthreads()` |
| 24 | 多 stream 时误用默认流 | 隐式同步，重叠失效 | 用 `cudaStreamNonBlocking` 非默认流 |
| 25 | 异步拷贝用了普通页内存 | 退化为同步拷贝，无重叠 | `cudaMallocHost` 固定内存 |

---

## 13. 一页纸终极速查（考前复习）

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ 【判断瓶颈】                                                              │
│   AI = FLOPs / Bytes                                                     │
│   AI < 12.5 (A100)  → 带宽受限：减少字节数、合并访问、tiling、压缩格式      │
│   AI > 12.5         → 计算受限：减少运算、FMA、Tensor Core、降精度         │
│   两者都低 + stall 高 → 延迟受限：提高占用率、增加 ILP/粗化                │
├──────────────────────────────────────────────────────────────────────────┤
│ 【优化优先级】（性价比从高到低）                                           │
│   1. 合并访存（最多 32×）                                                 │
│   2. 共享内存 tiling（10–20×）                                            │
│   3. 消除 bank conflict（最多 32×，共享内存侧）                            │
│   4. 线程粗化 / 寄存器复用（1.5–3×）                                       │
│   5. 原子操作私有化（10–100×）                                             │
│   6. 向量化 float4（2–4×）                                                │
│   7. `__restrict__`（5–20%）                                             │
│   8. 流重叠传输与计算（最高 2×）                                           │
├──────────────────────────────────────────────────────────────────────────┤
│ 【必背公式】                                                              │
│   i = blockIdx.x * blockDim.x + threadIdx.x                             │
│   grid = (N + block - 1) / block                                        │
│   bank = (byteAddr / 4) % 32                                            │
│   合并事务 = 128 字节 = 32 threads × 4 bytes                             │
│   算术强度(tiled matmul) = TILE_WIDTH / 4                                │
│   全局访存减少(tiled matmul) = TILE_WIDTH 倍                             │
│   性能上限 = min(PeakFLOPS, AI × PeakBW)                                │
│   所需并发度 = 延迟 × 吞吐率（Little's Law）                              │
│   加速比 = T_serial / T_parallel；Amdahl: 1/((1-p)+p/n)                  │
├──────────────────────────────────────────────────────────────────────────┤
│ 【三条黄金准则】                                                          │
│   ① 正确性优先：先有 golden reference 和容差比对，再谈性能                 │
│   ② 测量驱动：一次只改一个变量，每次都记录到优化日志                        │
│   ③ 认清天花板：先用 Roofline 算出上限，再问"我离上限多远"                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*速查表基于 UIUC ECE408/CS483 公开课程资料（官方课程描述、17 条 Instructional Objectives、
Lab Projects 列表、Prof. Steven S. Lumetta 公开的全部 22 个 Slide Deck）与 CUDA 官方文档整理。*

{% endraw %}
