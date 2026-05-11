---
title: "TensorRT-LLM 推理优化"
excerpt: "计算图优化、内核融合、INT8/FP8量化、张量并行/流水线并行"
collection: llm-libs
permalink: /llm-libs/13-tensorrt-llm
category: inference
---


## 1. 库简介与在LLM开发中的作用

TensorRT-LLM 是 NVIDIA 推出的大语言模型推理优化库，基于 TensorRT 深度学习编译器构建，专门针对 LLM 推理场景进行了深度优化。它提供了一套 Python API，使开发者能够高效地定义、优化和运行大语言模型。

### 核心定位

TensorRT-LLM 在 LLM 开发中的核心作用是**推理加速**。与训练框架（如 PyTorch）不同，TensorRT-LLM 专注于将已训练好的模型转化为高性能推理引擎，其优势体现在：

- **极致推理性能**：通过计算图优化、内核融合、量化等技术，显著降低推理延迟、提升吞吐量
- **显存高效利用**：通过 KV Cache 管理、Paged Attention 等机制，最大化 GPU 显存利用率
- **多GPU并行支持**：原生支持张量并行和流水线并行，轻松扩展到多卡、多节点部署
- **动态批处理**：Inflight Batching 机制可动态调度请求，大幅提升服务吞吐

### 与相关库的关系

| 组件 | 角色 |
|------|------|
| PyTorch / HF Transformers | 模型训练与权重导出 |
| TensorRT-LLM | 模型优化编译与高性能推理 |
| Triton Inference Server | 模型服务化部署（可选） |

---

## 2. 安装方式

### 2.1 通过 pip 安装（推荐）

```bash
# 基础安装
pip install tensorrt-llm

# 指定版本安装
pip install tensorrt-llm==0.12.0
```

### 2.2 通过 Docker 安装

```bash
# 拉取官方镜像
docker pull nvcr.io/nvidia/tensorrt-llm:0.12.0-py3

# 启动容器
docker run --gpus all -it --rm \
  -v /path/to/models:/models \
  nvcr.io/nvidia/tensorrt-llm:0.12.0-py3 \
  /bin/bash
```

### 2.3 从源码构建

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
make -C docker release_build
```

### 环境要求

- GPU：NVIDIA GPU，计算能力 >= 8.0（Ampere及以上架构，如 A100、L40、H100）
- CUDA：>= 11.8
- Python：>= 3.10
- cuDNN：>= 8.9

---

## 3. 核心类/函数/工具详细说明

### 3.1 模型构建 API

TensorRT-LLM 的底层构建 API 直接映射到 TensorRT 的核心概念。

#### 3.1.1 Builder

`Builder` 是引擎构建的入口，负责创建 Network 和配置构建参数。

```python
import tensorrt as trt

# 创建 Logger
logger = trt.Logger(trt.Logger.WARNING)

# 创建 Builder
builder = trt.Builder(logger)

# 关键属性
# builder.max_batch_size  -- 最大批次大小（已弃用，建议使用动态维度）
# builder.max_workspace_size  -- 最大工作空间大小（已弃用）
```

**关键参数：**
- `logger`：日志记录器，控制日志输出级别（`VERBOSE`/`INFO`/`WARNING`/`ERROR`）

#### 3.1.2 Network

`Network` 定义计算图结构，包含输入、层、输出等节点。

```python
# 创建网络定义，flags=1 表示启用显式批次维度
network = builder.create_network(1)

# 添加输入
input_tensor = network.add_input(
    name="input_ids",           # 张量名称
    dtype=trt.int32,            # 数据类型
    shape=(1, -1)               # 形状，-1 表示动态维度
)

# 添加全连接层
fc_layer = network.add_multiply(input_tensor, weight_tensor)

# 标记输出
network.mark_output(fc_layer.get_output(0))
```

**关键参数：**
- `flags`：网络创建标志，`1` 表示启用显式批次（Explicit Batch），现代用法必须启用

#### 3.1.3 Parser

Parser 负责将外部模型格式解析为 TensorRT Network。

```python
# ONNX Parser
parser = trt.OnnxParser(network, logger)
success = parser.parse_from_file("model.onnx")

# 检查解析结果
if not success:
    for i in range(parser.num_errors):
        print(parser.get_error(i))
```

### 3.2 TRT-LLM 高层 API

TensorRT-LLM 提供了更友好的高层 API，屏蔽了底层 TensorRT 的复杂性。

#### 3.2.1 from_hf_face

从 HuggingFace 模型加载权重，是最常用的模型导入方式。

```python
from tensorrt_llm import LLM, ModelConfig

# 方式一：使用 LLM 高层 API（推荐，0.12+ 版本）
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",   # HuggingFace 模型ID或本地路径
    tensor_parallel_size=2,               # 张量并行度
    pipeline_parallel_size=1,             # 流水线并行度
    quantization=None,                    # 量化配置，如 "int8-weight-only"
    max_batch_size=32,                    # 最大批次大小
    max_seq_len=4096,                     # 最大序列长度
    kv_cache_type="paged",               # KV Cache 类型: "paged" | "contiguous"
    trust_remote_code=True,              # 是否信任远程代码
)

# 方式二：使用 ModelConfig + 从权重构建
config = ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,
    max_batch_size=32,
    max_seq_len=4096,
)
```

**关键参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | str | HuggingFace 模型 ID 或本地路径 |
| `tensor_parallel_size` | int | 张量并行度（GPU数） |
| `pipeline_parallel_size` | int | 流水线并行度 |
| `quantization` | str/None | 量化策略：`None`、`"int8-weight-only"`、`"int8-smoothquant"`、`"fp8"` |
| `max_batch_size` | int | 运行时最大批次大小 |
| `max_seq_len` | int | 最大输入+输出序列长度 |
| `kv_cache_type` | str | KV Cache 管理策略 |
| `trust_remote_code` | bool | 是否执行模型仓库中的自定义代码 |

#### 3.2.2 build_engine

将模型编译为 TensorRT 引擎（.engine 文件），这是推理前的必要步骤。

```python
from tensorrt_llm.commands import build

# 使用命令行构建引擎
# trtllm-build --model_dir ./llama-7b-hf \
#              --dtype float16 \
#              --tp_size 2 \
#              --output_dir ./engine_output \
#              --max_batch_size 32 \
#              --max_input_len 1024 \
#              --max_seq_len 4096

# 使用 Python API 构建引擎
from tensorrt_llm.builder import Builder, BuildConfig

builder = Builder()
build_config = BuildConfig(
    max_batch_size=32,       # 最大批次大小
    max_input_len=1024,      # 最大输入长度
    max_seq_len=4096,        # 最大总序列长度
    max_num_tokens=4096,     # 单次最大token数（用于 Inflight Batching）
    max_beam_width=1,        # 最大beam宽度
    tp_size=2,               # 张量并行度
    pp_size=1,               # 流水线并行度
    dtype="float16",         # 计算精度
)

# 构建并保存引擎
engine = builder.build(model_config, build_config)
engine.save("./engine_output")
```

**BuildConfig 关键参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `max_batch_size` | int | 最大批次大小，影响 KV Cache 预分配 |
| `max_input_len` | int | 最大输入序列长度 |
| `max_seq_len` | int | 最大总序列长度（输入+输出） |
| `max_num_tokens` | int | 单次迭代最大 token 数（Inflight Batching 场景） |
| `max_beam_width` | int | Beam Search 宽度，1 表示仅贪心采样 |
| `tp_size` | int | 张量并行度 |
| `pp_size` | int | 流水线并行度 |
| `dtype` | str | 计算精度：`"float16"`、`"bfloat16"`、`"float32"` |

### 3.3 运行时 API

#### 3.3.1 GenerationSession

`GenerationSession` 是推理运行时的核心类，负责管理生成过程。

```python
from tensorrt_llm.runtime import GenerationSession, ModelConfig

# 创建模型配置
model_config = ModelConfig(
    max_batch_size=32,
    max_seq_len=4096,
    num_heads=32,
    num_kv_heads=32,        # GQA 时可小于 num_heads
    hidden_size=4096,
    vocab_size=32000,
    num_layers=32,
    dtype="float16",
)

# 创建生成会话
session = GenerationSession(
    model_config=model_config,
    engine_path="./engine_output",       # 引擎路径
    max_batch_size=32,
)

# 执行推理
output = session.generate(
    input_ids=input_tensor,              # 输入 token IDs，形状 [batch, seq_len]
    sampling_config=sampling_config,     # 采样配置
    max_new_tokens=256,                  # 最大生成 token 数
)
```

#### 3.3.2 SamplingConfig

`SamplingConfig` 控制生成策略（贪心、top-k、top-p 等）。

```python
from tensorrt_llm.runtime import SamplingConfig

sampling_config = SamplingConfig(
    end_id=2,                    # EOS token ID
    pad_id=0,                    # PAD token ID
    num_beams=1,                 # Beam Search 宽度（1=贪心）
    temperature=0.7,             # 采样温度，越高越随机
    top_k=50,                    # Top-K 采样：仅从概率最高的K个token中采样
    top_p=0.9,                   # Top-P (nucleus) 采样：从累积概率<=P的token中采样
    length_penalty=1.0,          # 长度惩罚（仅 Beam Search）
    repetition_penalty=1.1,      # 重复惩罚
    min_tokens=1,                # 最小生成长度
)

# 不同采样策略的典型配置
greedy_config = SamplingConfig(end_id=2, pad_id=0, num_beams=1)
creative_config = SamplingConfig(end_id=2, pad_id=0, temperature=0.9, top_p=0.95)
beam_config = SamplingConfig(end_id=2, pad_id=0, num_beams=4, length_penalty=0.6)
```

**关键参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `end_id` | int | - | 停止 token ID |
| `pad_id` | int | - | 填充 token ID |
| `num_beams` | int | 1 | Beam Search 宽度 |
| `temperature` | float | 1.0 | 采样温度，>1 更随机，<1 更确定 |
| `top_k` | int | 0 | Top-K，0 表示不启用 |
| `top_p` | float | 0.0 | Top-P，0.0 表示不启用 |
| `length_penalty` | float | 0.0 | 长度惩罚因子 |
| `repetition_penalty` | float | 1.0 | 重复惩罚，>1 减少重复 |
| `min_tokens` | int | 0 | 最小生成 token 数 |

---

## 4. 典型使用场景与代码示例

### 4.1 场景一：从 HuggingFace 模型构建引擎并推理

这是最完整的使用流程，涵盖从模型加载到推理输出的全链路。

```python
import torch
from tensorrt_llm import LLM, SamplingParams

# 第一步：初始化 LLM（自动下载模型 + 构建引擎）
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",   # HuggingFace 模型 ID
    tensor_parallel_size=1,               # 单卡推理
    max_seq_len=4096,                     # 最大序列长度
    kv_cache_type="paged",               # 使用 Paged KV Cache
)

# 第二步：配置采样参数
sampling_params = SamplingParams(
    max_tokens=512,           # 最大生成 token 数
    temperature=0.7,          # 采样温度
    top_p=0.9,               # Top-P 采样
    stop_token_ids=[151645],  # 停止 token（Qwen2 的 <|im_end|>）
)

# 第三步：执行推理
prompts = [
    "请解释什么是TensorRT-LLM？",
    "用Python写一个快速排序算法。",
]
outputs = llm.generate(prompts, sampling_params)

# 第四步：查看结果
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")
    print("---")

# 第五步：释放资源
del llm
torch.cuda.empty_cache()
```

### 4.2 场景二：离线构建引擎 + 在线加载推理

适用于生产环境：离线构建引擎文件，线上服务直接加载。

```python
# === 离线阶段：构建引擎 ===
from tensorrt_llm.commands import build

# 方式一：命令行（推荐生产环境使用）
# trtllm-build \
#   --model_dir /models/llama-7b-hf \
#   --dtype float16 \
#   --tp_size 2 \
#   --max_batch_size 64 \
#   --max_input_len 2048 \
#   --max_seq_len 4096 \
#   --output_dir /engines/llama-7b-tp2-fp16

# 方式二：Python API
from tensorrt_llm import LLM

llm = LLM(
    model="/models/llama-7b-hf",
    tensor_parallel_size=2,
    max_batch_size=64,
    max_seq_len=4096,
)
llm.save("/engines/llama-7b-tp2-fp16")
del llm

# === 在线阶段：加载引擎并推理 ===
from tensorrt_llm import LLM, SamplingParams

# 直接加载预构建的引擎
llm = LLM(
    model="/engines/llama-7b-tp2-fp16",   # 引擎目录路径
    tensor_parallel_size=2,
)

sampling_params = SamplingParams(max_tokens=256, temperature=0.8)
output = llm.generate("Hello, world!", sampling_params)
print(output[0].outputs[0].text)
```

### 4.3 场景三：INT8 量化推理

量化可以大幅降低显存占用和提升推理速度。

```python
from tensorrt_llm import LLM, SamplingParams

# Weight-Only INT8 量化
llm_wo_int8 = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="int8-weight-only",    # 仅权重量化为 INT8
    tensor_parallel_size=1,
    max_seq_len=4096,
)
# 权重从 FP16 (2 bytes/param) 压缩到 INT8 (1 byte/param)，显存减半

# SmoothQuant INT8 量化（权重+激活均量化）
llm_sq_int8 = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="int8-smoothquant",    # SmoothQuant: 权重+激活 INT8
    tensor_parallel_size=1,
    max_seq_len=4096,
)
# 激活也量化为 INT8，计算吞吐更高

# FP8 量化（需要 H100/H200 等 Hopper 架构 GPU）
llm_fp8 = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="fp8",                 # FP8 E4M3 格式
    tensor_parallel_size=1,
    max_seq_len=4096,
)

# 推理
sampling_params = SamplingParams(max_tokens=128, temperature=0.7)
output = llm_wo_int8.generate("解释量子计算的基本原理", sampling_params)
print(output[0].outputs[0].text)
```

### 4.4 场景四：多 GPU 张量并行推理

```python
from tensorrt_llm import LLM, SamplingParams

# 2-GPU 张量并行
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,              # 4 卡张量并行
    max_seq_len=4096,
    kv_cache_type="paged",
)

# 推理（自动分配到 4 张 GPU）
sampling_params = SamplingParams(max_tokens=256, temperature=0.8)
outputs = llm.generate(["你好", "Hello"], sampling_params)

for out in outputs:
    print(out.outputs[0].text)
```

### 4.5 场景五：自定义模型构建（低层 API）

对于 TensorRT-LLM 尚未内置支持的模型，可以使用低层 API 手动构建计算图。

```python
import tensorrt as trt
from tensorrt_llm import Builder, Network
from tensorrt_llm.layers import Attention, MLP
from tensorrt_llm.module import Module

class CustomTransformerBlock(Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # QKV 投影层（融合为一个 GEMM）
        self.qkv_proj = ColumnLinear(hidden_size, hidden_size * 3, bias=False)
        # 输出投影层
        self.o_proj = RowLinear(hidden_size, hidden_size, bias=False)
        # MLP 层
        self.mlp = MLP(hidden_size, intermediate_size, bias=False)
        # Layer Norm
        self.input_layernorm = LayerNorm(hidden_size)
        self.post_layernorm = LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask, kv_cache):
        # Pre-LN Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # QKV 融合投影
        qkv = self.qkv_proj(hidden_states)  # [batch, seq, 3*hidden]

        # 分离 Q, K, V
        q, k, v = split(qkv, self.hidden_size, dim=-1)

        # Attention 计算
        attn_output = attention(q, k, v, attention_mask, kv_cache)

        # 输出投影 + 残差连接
        hidden_states = self.o_proj(attn_output) + residual

        # Pre-LN MLP
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states) + residual

        return hidden_states
```

---

## 5. 数学原理

### 5.1 计算图优化：算子融合减少内存读写和 Kernel Launch 开销

计算图优化的核心目标是减少 GPU 全局内存（HBM）的读写次数和 kernel launch 的开销。GPU 计算的关键瓶颈往往不是算力（FLOPS），而是内存带宽（Memory Bandwidth）。

#### 5.1.1 QKV Projection 融合

标准 Transformer 中，Q、K、V 三个投影是独立的矩阵乘法：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

这需要 3 次 GEMM（通用矩阵乘法）调用，每次都需要从 HBM 读取输入 $X$。

融合后，将 $W_Q, W_K, W_V$ 拼接为一个权重矩阵：

$$[Q, K, V] = X \cdot [W_Q | W_K | W_V]$$

**优化效果：**
- 3 次 GEMM → 1 次 GEMM
- 输入 $X$ 仅从 HBM 读取 1 次（原来 3 次）
- 减少 2 次 kernel launch 开销
- 内存读取量从 $3 \times \text{read}(X)$ 减少为 $1 \times \text{read}(X)$

#### 5.1.2 MLP 融合

标准 MLP 包含两次线性投影和一次激活函数：

$$\text{MLP}(X) = \text{act}(XW_1)W_2$$

未融合时需要 3 个 kernel：GEMM(W1) → Activation → GEMM(W2)，中间结果需要写回 HBM 再读取。

融合后将 GEMM + Activation + GEMM 合并为一个 fused kernel：
- 中间结果 $\text{act}(XW_1)$ 保留在寄存器/共享内存中
- 避免了中间结果的 HBM 写入和读取
- 内存带宽节省约 $\frac{1}{3}$（省去一次中间结果的写入+读取）

#### 5.1.3 Attention 融合

标准 Multi-Head Attention 的计算流程：

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

TensorRT-LLM 将 Q×K^T、Scale、Softmax、×V 融合为一个 Flash Attention kernel：
- Q、K、V 从 HBM 加载后，所有中间计算在 SRAM（共享内存）中完成
- 避免 $QK^T$ 大矩阵的 HBM 写入/读取（这是经典内存瓶颈）
- 这正是 Flash Attention 算法的核心思想

### 5.2 INT8 量化

#### 5.2.1 Weight-Only INT8 量化

仅将权重从 FP16/BF16 量化为 INT8，激活值仍保持高精度。

**量化公式：**

$$W_{\text{int8}} = \text{round}\left(\frac{W_{\text{fp16}}}{s}\right), \quad s = \frac{\max(|W_{\text{fp16}}|)}{127}$$

**反量化（推理时）：**

$$W_{\text{fp16}} \approx W_{\text{int8}} \times s$$

**特点：**
- 量化误差仅来自权重的舍入，激活值保持 FP16 精度
- 显存占用减半（权重从 2 bytes/param → 1 byte/param）
- 推理时需要先反量化权重再计算，计算吞吐提升有限
- 适合对精度敏感的场景

#### 5.2.2 SmoothQuant INT8 量化

SmoothQuant 是一种同时量化权重和激活的方法，通过数学等价变换将激活的量化难度转移到权重上。

**核心思想：** 激活值中存在异常大的离群值（outlier），直接量化精度损失大。SmoothQuant 通过逐通道缩放，平滑激活值分布。

**平滑变换：**

$$x_{\text{smooth}} = x \cdot \text{diag}(s)^{-1}, \quad W_{\text{smooth}} = W \cdot \text{diag}(s)$$

其中缩放因子 $s$ 的计算方式为：

$$s_j = \frac{\max(|x_j|)^\alpha}{\max(|w_j|)^{1-\alpha}}$$

- $x_j$：激活值第 $j$ 个通道的值
- $w_j$：权重第 $j$ 个通道的值
- $\alpha$：平滑因子，通常取 0.5（平衡权重和激活的量化难度）

**数学等价性：**

$$x \cdot W = (x \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = x_{\text{smooth}} \cdot W_{\text{smooth}}$$

变换后，激活值中的离群值被抑制，权重吸收了缩放因子，两者都更容易量化为 INT8。

**特点：**
- 权重和激活均为 INT8，可使用 INT8 Tensor Core 加速
- 计算吞吐显著提升（INT8 Tensor Core 吞吐是 FP16 的 2 倍）
- 精度损失通常在可接受范围内
- 需要校准数据集确定缩放因子 $s$

### 5.3 FP8（E4M3 / E5M2）

FP8 是 8 位浮点格式，TensorRT-LLM 支持两种 FP8 编码：

#### E4M3 格式

| 组成 | 位数 | 说明 |
|------|------|------|
| 符号位 | 1 bit | 0 正 1 负 |
| 指数位 | 4 bit | 偏移量 7 |
| 尾数位 | 3 bit | 隐含前导 1 |

- 表示范围：$[-448, 448]$
- 精度：约 3-4 位有效数字
- **用途：前向传播中的权重和激活存储**（需要更高精度）

#### E5M2 格式

| 组成 | 位数 | 说明 |
|------|------|------|
| 符号位 | 1 bit | 0 正 1 负 |
| 指数位 | 5 bit | 偏移量 15 |
| 尾数位 | 2 bit | 隐含前导 1 |

- 表示范围：$[-57344, 57344]$
- 精度：约 2-3 位有效数字
- **用途：反向传播中的梯度存储**（需要更大动态范围）

**FP8 vs INT8 的优势：**
- 浮点格式天然保留了动态范围，不需要逐通道缩放
- E4M3 的 3 位尾数提供了比 INT8 更好的精度
- 不需要校准步骤，可直接训练或推理

---

## 6. 代码原理/架构原理

### 6.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                   用户应用层                          │
│  (Triton Inference Server / 自定义服务)               │
├─────────────────────────────────────────────────────┤
│               TensorRT-LLM Python API                │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   LLM    │  │ ModelConfig  │  │ SamplingParams│  │
│  └──────────┘  └──────────────┘  └───────────────┘  │
├─────────────────────────────────────────────────────┤
│              TensorRT-LLM Runtime                    │
│  ┌──────────────┐  ┌────────────────────────────┐   │
│  │ Generation   │  │     KV Cache Manager       │   │
│  │ Session      │  │  (Paged/Contiguous)        │   │
│  └──────────────┘  └────────────────────────────┘   │
│  ┌──────────────┐  ┌────────────────────────────┐   │
│  │ Inflight     │  │   Sampling & Beam Search   │   │
│  │ Batching     │  │                            │   │
│  └──────────────┘  └────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│             TensorRT Engine (编译优化后)              │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│  │Fused   │ │Fused   │ │Fused   │ │Quantized│       │
│  │Attention│ │MLP    │ │QKV    │ │GEMM    │       │
│  └────────┘ └────────┘ └────────┘ └────────┘       │
├─────────────────────────────────────────────────────┤
│                   CUDA / cuDNN                       │
└─────────────────────────────────────────────────────┘
```

### 6.2 引擎构建流程

引擎构建是一个**编译过程**，将高层模型描述转化为优化的 GPU 可执行代码：

```
HuggingFace 模型权重 (.bin/.safetensors)
        │
        ▼
   权重加载 & 模型构建（TensorRT-LLM Python API）
        │
        ▼
   TensorRT Network 定义（计算图）
        │
        ▼
   ┌──────────────────────────────────┐
   │  TensorRT 优化 Pass:             │
   │  1. 算子融合 (Layer Fusion)      │
   │  2. 精度校准 (Precision Calib)   │
   │  3. 内存规划 (Memory Planning)   │
   │  4. Kernel 选择 (Kernel Select)  │
   └──────────────────────────────────┘
        │
        ▼
   优化后的 TensorRT Engine (.engine)
        │
        ▼
   序列化保存到磁盘
```

### 6.3 张量并行（Tensor Parallelism）

张量并行将模型权重按维度切分到多个 GPU 上，每个 GPU 只保存部分权重并计算部分结果。

**以线性层 $Y = XW$ 为例：**

- **列并行（Column Parallel）**：将 $W$ 按列切分为 $[W_1, W_2]$，每个 GPU 计算部分输出
  - GPU 0: $Y_1 = XW_1$
  - GPU 1: $Y_2 = XW_2$
  - 需要 AllReduce 拼接 $[Y_1, Y_2]$

- **行并行（Row Parallel）**：将 $W$ 按行切分为 $\begin{bmatrix}W_1 \\ W_2\end{bmatrix}$，每个 GPU 计算部分结果
  - GPU 0: $Y_1 = X_1 W_1$（$X$ 也按列切分）
  - GPU 1: $Y_2 = X_2 W_2$
  - 需要 AllReduce 求和 $Y = Y_1 + Y_2$

**TensorRT-LLM 中的典型并行策略：**

```
Transformer Block:
  QKV Proj  → Column Parallel（按注意力头切分）
  O Proj    → Row Parallel
  MLP Up    → Column Parallel
  MLP Down  → Row Parallel
```

这样每个 Attention/MLP 块只需要一次 AllReduce，通信开销最小。

### 6.4 流水线并行（Pipeline Parallelism）

流水线并行将模型按层切分到不同 GPU 上，每个 GPU 负责连续的若干层。

```
GPU 0: Layer 0-7    →  GPU 1: Layer 8-15  →  GPU 2: Layer 16-23  →  GPU 3: Layer 24-31
```

**微批次（Micro-batching）技术：** 将一个批次拆分为多个微批次，流水线式地在各 GPU 间传递，减少 GPU 空闲时间。

```
时间 →  t0    t1    t2    t3    t4    t5    t6    t7
GPU0: [M0]  [M1]  [M2]  [M3]  -     -     -     -
GPU1:  -    [M0]  [M1]  [M2]  [M3]  -     -     -
GPU2:  -     -    [M0]  [M1]  [M2]  [M3]  -     -
GPU3:  -     -     -    [M0]  [M1]  [M2]  [M3]  -
```

### 6.5 KV Cache 管理：PagedKVCache

KV Cache 是自回归生成中最关键的显存管理问题。传统方法为每个序列预分配最大长度的连续内存，导致大量浪费。

**PagedKVCache 借鉴操作系统虚拟内存的分页思想：**

- 将 KV Cache 划分为固定大小的"页"（Block），每页存储若干 token 的 Key/Value
- 维护一个页表（Page Table），记录每个序列的 KV Cache 页映射
- 序列增长时按需分配新页，不需要连续内存
- 序列结束时释放页，供其他序列复用

```
传统方式（预分配连续内存）:
  Seq1: [████████████░░░░░░░░░░]  使用 8/20 blocks，浪费 60%
  Seq2: [████████░░░░░░░░░░░░░░]  使用 8/20 blocks，浪费 60%

PagedKVCache（按需分配）:
  Block Pool: [B1][B2][B3][B4][B5][B6][B7][B8]...
  Seq1 页表:  B1 → B3 → B5 → B7   (逻辑连续，物理不连续)
  Seq2 页表:  B2 → B4 → B6        (按需分配)
```

**优势：**
- 显存利用率接近 100%（几乎无浪费）
- 支持更长的序列和更大的批次
- 为 Inflight Batching 提供基础

### 6.6 批处理：Inflight Batching

传统静态批处理需要等待批次中所有序列完成才能处理下一批，短序列完成后 GPU 空闲。

**Inflight Batching（也称 Continuous Batching）** 动态管理批次：

1. **迭代级调度**：每次解码迭代都可以加入新请求或移除已完成的请求
2. **早退机制**：某个序列生成 EOS 后立即从批次中移除，释放资源
3. **动态加入**：新请求在任意迭代步骤加入批次

```
传统 Static Batching:
  时间 →  t0        t1        t2        t3        t4
  Seq1:  [decode]  [decode]  [decode]  [EOS]     [idle] ← 空等
  Seq2:  [decode]  [decode]  [decode]  [decode]  [decode]

Inflight Batching:
  时间 →  t0        t1        t2        t3        t4
  Seq1:  [decode]  [decode]  [decode]  [EOS]     -
  Seq2:  [decode]  [decode]  [decode]  [decode]  [decode]
  Seq3:  -         -         -         [prefill]  [decode] ← 新请求立即加入
```

---

## 7. 常见注意事项与最佳实践

### 7.1 引擎构建注意事项

1. **构建参数与运行时参数必须匹配**：`max_batch_size`、`max_seq_len` 等参数在构建时确定，运行时不可修改。预估不足会导致 OOM 或无法运行，预估过大会浪费显存。

2. **构建时间较长**：大模型的引擎构建可能需要数分钟到数十分钟，建议离线构建并保存引擎文件，线上直接加载。

3. **GPU 架构绑定**：构建的引擎与 GPU 架构绑定，在 A100 上构建的引擎不能直接在 H100 上使用。跨架构部署需要重新构建。

4. **版本兼容性**：不同版本的 TensorRT-LLM 构建的引擎不兼容，升级版本后需要重新构建。

### 7.2 量化选择建议

| 场景 | 推荐量化方式 | 原因 |
|------|-------------|------|
| 精度敏感（如代码生成） | Weight-Only INT8 | 激活保持 FP16，精度损失最小 |
| 吞吐优先（如聊天服务） | SmoothQuant INT8 | 权重+激活 INT8，计算吞吐翻倍 |
| H100/H200 部署 | FP8 E4M3 | 硬件原生支持，兼顾精度与性能 |
| 模型太大放不下 | Weight-Only INT8 | 显存减半，最低代价 |

### 7.3 并行策略选择

| 模型大小 | GPU 配置 | 推荐策略 |
|----------|----------|----------|
| < 7B | 1× A100 | 无需并行 |
| 7B-13B | 1× A100 80GB | 无需并行（FP16/INT8 均可） |
| 30B-70B | 2-4× A100 | 张量并行（tp=2-4） |
| 70B+ | 4-8× A100 | 张量并行（tp=4-8） |
| 跨节点 | 2+ 节点 | 张量并行 + 流水线并行 |

**原则：** 优先使用张量并行，仅在单节点 GPU 数不够时才引入流水线并行（流水线并行有流水线气泡开销）。

### 7.4 KV Cache 与内存规划

```python
# 计算模型 + KV Cache 显存需求
# 示例：Llama-2-7B, FP16, max_seq_len=4096, max_batch_size=32

# 模型权重: 7B × 2 bytes = 14 GB
# KV Cache:  2 × num_layers × num_kv_heads × head_dim × max_seq_len × max_batch_size × 2 bytes
#          = 2 × 32 × 32 × 128 × 4096 × 32 × 2 bytes ≈ 68.7 GB

# 这说明 KV Cache 可能比模型权重本身占用更多显存！
# 使用 PagedKVCache + INT8 量化 KV Cache 可大幅降低显存需求
```

**最佳实践：**
- 使用 `kv_cache_type="paged"` 而非 `"contiguous"`
- 开启 KV Cache 量化（`--kv_cache_dtype int8`）可将 KV Cache 显存减半
- `max_seq_len` 不要设置过大，根据实际业务需求设置
- 监控 KV Cache 利用率，避免 OOM

### 7.5 Inflight Batching 调优

- `max_num_tokens`：控制单次迭代的最大 token 数，影响批处理粒度。推荐设置为 `max_batch_size × 平均输入长度` 的 1-2 倍。
- 合理设置 `max_batch_size`：过大会导致 KV Cache 不足，过小浪费吞吐。
- 使用 Triton Inference Server 部署时，Inflight Batching 由 Triton 自动管理。

### 7.6 常见问题与排错

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| OOM during build | max_seq_len 过大 | 减小 max_seq_len 或开启量化 |
| OOM during inference | KV Cache 占满 | 减小 max_batch_size 或启用 PagedKVCache |
| 精度下降明显 | 量化精度不够 | 从 SmoothQuant 降级为 Weight-Only |
| 引擎加载失败 | 版本不兼容 | 使用相同版本的 TensorRT-LLM 重新构建 |
| 多卡通信超时 | NCCL 配置 | 检查 `NCCL_SOCKET_IFNAME` 等环境变量 |
| 推理速度慢于预期 | 未使用 fused kernel | 确认引擎构建时启用了层融合 |

### 7.7 性能调优 Checklist

- [ ] 使用 FP16/BF16 而非 FP32 作为基础精度
- [ ] 开启 PagedKVCache（`kv_cache_type="paged"`）
- [ ] 启用 Inflight Batching
- [ ] 根据场景选择合适的量化策略
- [ ] 合理设置 `max_num_tokens` 以优化批处理效率
- [ ] 使用 `nvidia-smi` 和 `nsys` 分析 GPU 利用率
- [ ] 确保使用足够的张量并行度来填满 GPU 内存带宽
- [ ] 预构建引擎文件，避免线上实时编译

---

## 参考资源

- [TensorRT-LLM GitHub 仓库](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT-LLM 官方文档](https://nvidia.github.io/TensorRT-LLM/)
- [SmoothQuant 论文](https://arxiv.org/abs/2211.10438)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [FP8 Formats for Deep Learning 论文](https://arxiv.org/abs/2209.05433)
