---
title: "NumPy 科学计算库"
excerpt: "ndarray、广播机制、矩阵运算、线性代数(SVD/特征值)、FFT、在LLM中的数据预处理"
collection: llm-libs
permalink: /llm-libs/02-numpy
category: core
toc: true
---


## 1. 简介

NumPy（Numerical Python）是 Python 生态中最基础的科学计算库，提供了高性能的多维数组对象（ndarray）和丰富的数学函数。NumPy 是几乎所有 Python 数据科学和机器学习库的底层依赖，包括 PyTorch、TensorFlow、Pandas、SciPy 等。

### NumPy 在 LLM 开发中的核心角色

- **数据预处理**：文本数据的数值化、分词后的索引操作、Batch 的拼接与填充
- **嵌入计算**：词嵌入矩阵的构建与操作
- **注意力矩阵运算**：缩放点积注意力中的矩阵乘法、Softmax 数值稳定计算
- **线性代数运算**：SVD 降维、PCA 分析、特征值分解
- **数值验证**：为 PyTorch 自定义算子提供参考实现，验证正确性

## 2. 安装

```bash
# 标准安装
pip install numpy

# 指定版本
pip install numpy==1.26.4

# 验证安装
python -c "import numpy as np; print(np.__version__)"
```

## 3. 核心模块详解

### 3.1 ndarray：多维数组

ndarray（N-dimensional array）是 NumPy 的核心数据结构，提供高效的同类数据多维容器。

#### 3.1.1 创建数组

```python
import numpy as np

# 从列表创建
a = np.array([1, 2, 3])                     # 1D 数组
b = np.array([[1, 2], [3, 4]], dtype=np.float32)  # 2D 数组，指定类型

# 工厂函数
zeros = np.zeros((3, 4))          # 全零数组，形状 (3, 4)
ones = np.ones((2, 3))            # 全一数组
empty = np.empty((2, 3))          # 未初始化数组（值不确定，但速度最快）
full = np.full((2, 3), 7.0)      # 用指定值填充

# 序列生成
arange = np.arange(0, 10, 2)     # 等差序列：[0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)  # 等间隔序列：[0, 0.25, 0.5, 0.75, 1.0]
logspace = np.logspace(0, 3, 4)  # 对数间隔：[1, 10, 100, 1000]

# 随机数组
rng = np.random.default_rng(42)  # 推荐方式：使用 Generator
rand = rng.random((2, 3))        # 均匀分布 [0, 1)
randn = rng.standard_normal((2, 3))  # 标准正态分布 N(0,1)
ints = rng.integers(0, 10, size=(2, 3))  # 整数 [0, 10)

# 特殊矩阵
eye = np.eye(3)                  # 3×3 单位矩阵
diag = np.diag([1, 2, 3])       # 对角矩阵

# 从 PyTorch 张量转换
import torch
t = torch.randn(2, 3)
arr = t.numpy()                  # CPU 张量 → NumPy（共享内存）
```

#### 3.1.2 数组属性

```python
a = np.random.randn(2, 3, 4)

a.ndim          # 3 — 维度数（轴数）
a.shape         # (2, 3, 4) — 各维度大小
a.size          # 24 — 元素总数
a.dtype         # float64 — 数据类型
a.itemsize      # 8 — 每个元素的字节数
a.nbytes        # 192 = 24 × 8 — 总字节数
a.strides       # (96, 32, 8) — 各维度步长（字节）
```

#### 3.1.3 数据类型

```python
# NumPy 的数据类型体系
np.int8, np.int16, np.int32, np.int64      # 有符号整数
np.uint8, np.uint16, np.uint32, np.uint64  # 无符号整数
np.float16, np.float32, np.float64         # 浮点数
np.bool_                                    # 布尔型
np.complex64, np.complex128                 # 复数

# 类型转换
a = np.array([1.5, 2.7, 3.9])
b = a.astype(np.int32)       # [1, 2, 3] — 截断而非四舍五入
c = a.astype(np.float32)     # 转为单精度

# LLM 中的常用类型
# 模型权重通常用 float32 或 float16
# Token ID 用 int32 或 int64
# 注意力掩码用 bool_
```

### 3.2 广播机制

广播（Broadcasting）是 NumPy 处理不同形状数组间运算的强大机制，无需显式复制数据。

#### 3.2.1 广播规则

1. 两个数组从最右边的维度开始对齐
2. 两个维度大小相同，或其中一个为1，则兼容
3. 兼容的维度中，大小为1的维度被"广播"（沿该维度复制）
4. 不兼容的维度会导致错误

```python
# 例1：(3, 1) + (1, 4) → (3, 4)
a = np.ones((3, 1))    # shape: (3, 1)
b = np.ones((1, 4))    # shape: (1, 4)
c = a + b              # shape: (3, 4)

# 例2：(5, 3, 4) + (3, 4) → (5, 3, 4)
a = np.ones((5, 3, 4))
b = np.ones((3, 4))
c = a + b              # b 被广播到 (5, 3, 4)

# 例3：(5, 3, 4) + (5, 1, 4) → (5, 3, 4)
a = np.ones((5, 3, 4))
b = np.ones((5, 1, 4))
c = a + b              # b 的第1维从1广播到3

# 不兼容的情况
a = np.ones((3, 4))
b = np.ones((3, 5))
# a + b → ValueError: operands could not be broadcast together
```

#### 3.2.2 LLM 中的广播应用

```python
# 场景1：Layer Normalization
# hidden: (batch, seq_len, hidden_dim), gamma/beta: (hidden_dim,)
hidden = np.random.randn(4, 128, 768)
gamma = np.ones(768)
beta = np.zeros(768)

mean = hidden.mean(axis=-1, keepdims=True)  # (4, 128, 1)
var = hidden.var(axis=-1, keepdims=True)     # (4, 128, 1)
normalized = (hidden - mean) / np.sqrt(var + 1e-5)  # 广播减法
output = gamma * normalized + beta  # gamma/beta 广播到 (4, 128, 768)

# 场景2：RoPE 位置编码
# freqs: (seq_len, head_dim/2) → (1, seq_len, 1, head_dim/2)
freqs = np.random.randn(128, 64)
freqs = freqs[np.newaxis, :, np.newaxis, :]  # 广播到 batch 和 num_heads 维度
```

### 3.3 矩阵运算

#### 3.3.1 基本矩阵运算

```python
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# 矩阵乘法
C = np.dot(A, B)          # (3, 4) @ (4, 5) → (3, 5)
C = A @ B                 # Python 3.5+ 推荐写法
C = np.matmul(A, B)       # 与 @ 运算符等价

# 逐元素乘法（不是矩阵乘法！）
D = np.array([1, 2, 3])
E = np.array([4, 5, 6])
F = D * E                 # [4, 10, 18] — 逐元素相乘

# 转置
AT = A.T                  # (4, 3) — 转置
AT = np.transpose(A)      # 等价写法

# 高维转置
X = np.random.randn(2, 3, 4, 5)
XT = np.transpose(X, (0, 2, 1, 3))  # 交换第1和第2维：(2, 4, 3, 5)
```

#### 3.3.2 批量矩阵乘法（LLM 注意力中的核心运算）

```python
# 批量矩阵乘法：支持同时计算多个矩阵乘法
# Q: (batch, heads, seq_len, head_dim)
# K: (batch, heads, head_dim, seq_len)
Q = np.random.randn(4, 8, 128, 64)   # batch=4, heads=8
K = np.random.randn(4, 8, 128, 64)

# 注意力分数：Q @ K^T
attn_scores = Q @ np.transpose(K, (0, 1, 3, 2))  # (4, 8, 128, 128)

# 数值稳定的 Softmax
def stable_softmax(x, axis=-1):
    """避免数值溢出的 Softmax 实现"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)  # 减去最大值防止溢出
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

attn_weights = stable_softmax(attn_scores / np.sqrt(64))  # 缩放点积注意力
```

#### 3.3.3 行列式与逆矩阵

```python
A = np.random.randn(3, 3)

# 行列式
det = np.linalg.det(A)       # 标量

# 逆矩阵
A_inv = np.linalg.inv(A)     # (3, 3)
# 验证：A @ A_inv ≈ I
print(np.allclose(A @ A_inv, np.eye(3)))  # True

# 伪逆（非方阵也可用）
B = np.random.randn(3, 4)
B_pinv = np.linalg.pinv(B)   # (4, 3)
```

### 3.4 随机数生成

#### 3.4.1 Generator（推荐方式）

```python
# 使用 new Generator API（NumPy 1.17+）
rng = np.random.default_rng(seed=42)  # 可选种子，确保可复现

# 基本分布
rng.random((2, 3))              # 均匀分布 [0, 1)
rng.standard_normal((2, 3))     # 标准正态 N(0, 1)
rng.integers(0, 100, size=10)   # 整数 [0, 100)

# LLM 相关分布
rng.normal(loc=0, scale=1, size=(768, 768))   # 正态分布 N(μ, σ²)
rng.uniform(low=-1, high=1, size=1000)          # 均匀分布 U(a, b)
rng.multivariate_normal(mean=[0, 0], cov=[[1, 0.5], [0.5, 1]], size=1000)  # 多元正态

# Dirichlet 分布（用于主题模型、混合权重）
alpha = np.array([1.0, 2.0, 3.0])
samples = rng.dirichlet(alpha, size=5)  # (5, 3)，每行和为1
```

#### 3.4.2 LLM 中的随机数应用

```python
# 权重初始化（Kaiming/He 初始化）
def kaiming_init(fan_in, shape, rng=None):
    """适用于 ReLU 激活的权重初始化"""
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0, std, size=shape)

weights = kaiming_init(768, (768, 3072))  # Transformer FFN 层初始化

# Xavier/Glorot 初始化
def xavier_init(fan_in, fan_out, rng=None):
    """适用于 Sigmoid/Tanh 激活的权重初始化"""
    if rng is None:
        rng = np.random.default_rng()
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0, std, size=(fan_in, fan_out))

# Dropout 实现
def dropout(x, p=0.1, rng=None):
    """NumPy 实现 Dropout"""
    if rng is None:
        rng = np.random.default_rng()
    mask = rng.random(x.shape) > p  # 以概率 p 置零
    return x * mask / (1 - p)        # 缩放以保持期望值不变
```

### 3.5 线性代数 (numpy.linalg)

#### 3.5.1 奇异值分解 (SVD)

```python
A = np.random.randn(4, 3)  # 4×3 矩阵

# 完整 SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)
# U: (4, 4), S: (3,), Vt: (3, 3)
# A = U @ diag(S) @ Vt

# 截断 SVD（节省计算）
U, S, Vt = np.linalg.svd(A, full_matrices=False)
# U: (4, 3), S: (3,), Vt: (3, 3)

# 验证重构
A_reconstructed = U @ np.diag(S) @ Vt
print(np.allclose(A, A_reconstructed))  # True

# LLM 应用：低秩近似与降维
# 嵌入矩阵降维
embeddings = np.random.randn(50000, 768)  # 词表大小 50000，维度 768
U, S, Vt = np.linalg.svd(embeddings, full_matrices=False)

# 保留前 k 个奇异值，实现降维
k = 128
embeddings_low_rank = U[:, :k] @ np.diag(S[:k])  # (50000, 128)
print(f"原始大小: {embeddings.nbytes / 1e6:.1f} MB")
print(f"降维后: {embeddings_low_rank.nbytes / 1e6:.1f} MB")

# 计算有效秩（奇异值衰减分析）
total_energy = np.sum(S ** 2)
cumulative_energy = np.cumsum(S ** 2) / total_energy
effective_rank = np.searchsorted(cumulative_energy, 0.95) + 1  # 保留95%能量
print(f"有效秩（95%能量）: {effective_rank}")
```

#### 3.5.2 特征值分解

```python
# 对称矩阵的特征值分解
A = np.random.randn(3, 3)
A_sym = A @ A.T  # 构造对称矩阵

eigenvalues, eigenvectors = np.linalg.eigh(A_sym)  # 对称矩阵推荐用 eigh
# eigenvalues: (3,) — 特征值（升序排列）
# eigenvectors: (3, 3) — 每列是对应的特征向量

# 验证：A @ v = λ * v
for i in range(3):
    v = eigenvectors[:, i]
    lambda_i = eigenvalues[i]
    print(np.allclose(A_sym @ v, lambda_i * v))  # True

# 非对称矩阵用 eig
eigenvalues, eigenvectors = np.linalg.eig(A)
# 注意：特征值可能是复数

# LLM 应用：协方差矩阵分析（PCA）
data = np.random.randn(1000, 50)  # 1000 个样本，50 维
cov_matrix = np.cov(data.T)        # (50, 50) 协方差矩阵
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# 按特征值降序排列
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# 投影到前 k 个主成分
k = 10
data_pca = data @ eigenvectors[:, :k]  # (1000, 10)
```

#### 3.5.3 解线性方程组

```python
# 求解 Ax = b
A = np.array([[3, 1], [1, 2]], dtype=np.float64)
b = np.array([9, 8], dtype=np.float64)

x = np.linalg.solve(A, b)       # 解向量
print(np.allclose(A @ x, b))    # True

# 最小二乘解（超定方程组）
A = np.random.randn(10, 3)      # 10个方程，3个未知数
b = np.random.randn(10)
x, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
# x: 最小二乘解
# residuals: 残差平方和
# rank: 矩阵的秩
# sv: 奇异值
```

#### 3.5.4 范数

```python
A = np.random.randn(3, 4)

# 向量范数
v = np.array([3.0, 4.0])
np.linalg.norm(v, ord=1)     # L1 范数: 7.0
np.linalg.norm(v, ord=2)     # L2 范数: 5.0
np.linalg.norm(v, ord=np.inf)  # 无穷范数: 4.0

# 矩阵范数
np.linalg.norm(A, ord='fro')  # Frobenius 范数：所有元素平方和的平方根
np.linalg.norm(A, ord=2)      # 谱范数（最大奇异值）
np.linalg.norm(A, ord='nuc')  # 核范数（奇异值之和）

# LLM 应用：梯度裁剪中计算梯度范数
gradients = [np.random.randn(100, 200), np.random.randn(200, 50)]
total_norm = np.sqrt(sum(np.linalg.norm(g) ** 2 for g in gradients))
max_norm = 1.0
if total_norm > max_norm:
    scale = max_norm / total_norm
    gradients = [g * scale for g in gradients]
```

### 3.6 FFT：快速傅里叶变换

```python
# 一维 FFT
signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 128))  # 5Hz 正弦波
fft_result = np.fft.fft(signal)       # 复数频谱
frequencies = np.fft.fftfreq(128, d=1/128)  # 频率轴
magnitudes = np.abs(fft_result)        # 幅度谱
phases = np.angle(fft_result)          # 相位谱

# 逆 FFT
reconstructed = np.fft.ifft(fft_result)
print(np.allclose(signal, reconstructed.real))  # True

# 二维 FFT（可用于图像处理、注意力模式分析）
image = np.random.randn(64, 64)
fft2d = np.fft.fft2(image)
fft2d_shifted = np.fft.fftshift(fft2d)  # 将零频移到中心

# LLM 应用：用 FFT 加速循环矩阵乘法（某些高效注意力机制）
# 循环矩阵乘法可通过 FFT 在 O(n log n) 实现，而非 O(n²)
def circulant_matmul(c, v):
    """利用 FFT 计算循环矩阵与向量的乘法"""
    n = len(v)
    fft_c = np.fft.fft(c, n)
    fft_v = np.fft.fft(v, n)
    return np.fft.ifft(fft_c * fft_v).real

# 验证
c = np.array([1, 2, 3, 4])
v = np.array([5, 6, 7, 8])
from scipy.linalg import circulant
C = circulant(c)
print(np.allclose(circulant_matmul(c, v), C @ v))  # True
```

## 4. 数学原理

### 4.1 SVD 分解：A = UΣV^T

奇异值分解（Singular Value Decomposition）是线性代数中最重要的矩阵分解之一，任意实矩阵 A（m×n）都可以分解为：

$$A = U \Sigma V^T$$

其中：
- **U**：m×m 正交矩阵，称为左奇异向量矩阵，U^T U = I
- **Σ**：m×n 对角矩阵，对角线上的元素 σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 称为奇异值
- **V^T**：n×n 正交矩阵，称为右奇异向量矩阵，V^T V = I
- **r** = rank(A)，矩阵的秩

**关键性质**：

1. **低秩近似**：保留前 k 个最大奇异值，得到 A 的最优 rank-k 近似（Eckart-Young 定理）：
   $$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$
   这是在 Frobenius 范数和谱范数下的最优低秩近似。

2. **与特征值的关系**：A^T A 的特征值 = σᵢ²，即奇异值的平方。

3. **条件数**：cond(A) = σ_max / σ_min，条件数越大矩阵越病态。

**LLM 中的应用**：
- LoRA 的低秩分解本质上是 SVD 思想的应用
- 嵌入矩阵压缩
- 模型权重分析（有效秩衡量模型冗余度）

### 4.2 矩阵乘法结合律

矩阵乘法满足结合律：(AB)C = A(BC)，这在 LLM 计算中有重要优化意义：

**计算量分析**：

对于 A: (m, p), B: (p, q), C: (q, n)：
- (AB)C 的乘法次数：mpq + mqn
- A(BC) 的乘法次数：pqn + mpn

选择不同的结合顺序可以显著减少计算量：

```python
# 示例：注意力计算中的优化
# Q: (batch, heads, seq_len, head_dim)
# K^T: (batch, heads, head_dim, seq_len)
# V: (batch, heads, seq_len, head_dim)

# 标准顺序：先计算 QK^T 再乘 V
# Q @ K^T: (batch, heads, seq_len, seq_len) — O(seq_len² × head_dim)
# (QK^T) @ V: (batch, heads, seq_len, head_dim) — O(seq_len² × head_dim)
# 总计：O(seq_len² × head_dim)

# Flash Attention 利用结合律优化：
# 通过分块计算，避免显式存储 (seq_len, seq_len) 的注意力矩阵
# 从而将显存从 O(seq_len²) 降为 O(seq_len)
```

### 4.3 傅里叶变换公式

离散傅里叶变换（DFT）将时域信号转换为频域表示：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i 2\pi kn / N}, \quad k = 0, 1, ..., N-1$$

逆变换（IDFT）：

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] \cdot e^{i 2\pi kn / N}, \quad n = 0, 1, ..., N-1$$

**FFT（快速傅里叶变换）** 是 DFT 的高效算法，利用分治策略将 O(N²) 的计算量降为 O(N log N)：

1. 将 N 点 DFT 分解为两个 N/2 点 DFT（奇偶分组）
2. 递归分解直到基情形
3. 利用旋转因子 e^{-i2πk/N} 的对称性减少重复计算

**LLM 中的应用**：
- 循环/卷积注意力（如 Linear Attention）可用 FFT 加速
- 位置编码的频域分析
- 信号处理式的特征提取

## 5. 在 LLM 开发中的典型使用场景

### 5.1 文本数据预处理

```python
import numpy as np

# Token ID 序列的 Padding 和 Truncation
def pad_sequences(sequences, max_len=None, pad_value=0):
    """将变长序列填充到相同长度"""
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    batch_size = len(sequences)
    padded = np.full((batch_size, max_len), pad_value, dtype=np.int64)
    attention_mask = np.zeros((batch_size, max_len), dtype=np.bool_)

    for i, seq in enumerate(sequences):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]
        attention_mask[i, :length] = True

    return padded, attention_mask

# 使用示例
sequences = [
    [101, 2023, 2003, 1037, 3231, 102],
    [101, 2023, 2003, 102],
    [101, 1045, 2066, 18435, 102],
]
padded, mask = pad_sequences(sequences, max_len=8)
print("Token IDs:\n", padded)
print("Attention Mask:\n", mask.astype(int))

# Attention Mask:
# [[1 1 1 1 1 1 0 0]
#  [1 1 1 1 0 0 0 0]
#  [1 1 1 1 1 0 0 0]]
```

### 5.2 注意力机制的 NumPy 实现

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力的 NumPy 实现

    参数:
        Q: (batch, heads, seq_len, head_dim) — 查询矩阵
        K: (batch, heads, seq_len, head_dim) — 键矩阵
        V: (batch, heads, seq_len, head_dim) — 值矩阵
        mask: (batch, heads, seq_len, seq_len) 或 (1, 1, seq_len, seq_len) — 注意力掩码

    返回:
        output: (batch, heads, seq_len, head_dim) — 注意力输出
        attn_weights: (batch, heads, seq_len, seq_len) — 注意力权重
    """
    head_dim = Q.shape[-1]
    scale = np.sqrt(head_dim)

    # QK^T / sqrt(d_k)
    scores = Q @ np.transpose(K, (0, 1, 3, 2)) / scale  # (batch, heads, seq_len, seq_len)

    # 应用掩码（如因果掩码）
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 加权求和
    output = attn_weights @ V  # (batch, heads, seq_len, head_dim)
    return output, attn_weights

# 测试
batch, heads, seq_len, head_dim = 2, 4, 8, 64
Q = np.random.randn(batch, heads, seq_len, head_dim)
K = np.random.randn(batch, heads, seq_len, head_dim)
V = np.random.randn(batch, heads, seq_len, head_dim)

# 因果掩码
causal_mask = np.tril(np.ones((seq_len, seq_len)))[np.newaxis, np.newaxis, :, :]

output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
print(f"Output shape: {output.shape}")     # (2, 4, 8, 64)
print(f"Weights shape: {weights.shape}")   # (2, 4, 8, 8)
```

### 5.3 嵌入矩阵操作

```python
# 词嵌入查找与操作
vocab_size = 50000
hidden_dim = 768
embedding_matrix = np.random.randn(vocab_size, hidden_dim) * 0.02  # 小随机初始化

# 查找 Token 嵌入
token_ids = np.array([101, 2023, 2003, 102])  # "It is a [SEP]"
embeddings = embedding_matrix[token_ids]       # (4, 768) — 高级索引

# 余弦相似度计算
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 找到与目标词最相似的词
target_embedding = embedding_matrix[2023]  # "is" 的嵌入
similarities = embedding_matrix @ target_embedding / (
    np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(target_embedding)
)
top_k = 5
top_indices = np.argsort(similarities)[-top_k:][::-1]
print(f"与 'is' 最相似的 {top_k} 个词的索引: {top_indices}")
```

### 5.4 模型权重分析与压缩

```python
# 分析模型权重的奇异值分布
weight_matrix = np.random.randn(768, 3072)  # FFN 层权重
U, S, Vt = np.linalg.svd(weight_matrix, full_matrices=False)

# 计算信息保留率
total_energy = np.sum(S ** 2)
cumulative_ratio = np.cumsum(S ** 2) / total_energy

# 95% 能量保留需要多少奇异值
k_95 = np.searchsorted(cumulative_ratio, 0.95) + 1
print(f"保留 95% 能量需要 {k_95}/{len(S)} 个奇异值")
print(f"压缩比: {768 * 3072 / (768 * k_95 + k_95 + 3072 * k_95):.2f}x")

# 基于核范数的正则化（模型压缩）
def nuclear_norm_reg(weights, lambda_=0.01):
    """核范数正则化：鼓励低秩结构"""
    _, S, _ = np.linalg.svd(weights, full_matrices=False)
    return lambda_ * np.sum(S)

# 计算模型的有效秩
def effective_rank(weights, threshold=0.99):
    """计算有效秩：保留 threshold 比例能量所需的奇异值数"""
    _, S, _ = np.linalg.svd(weights, full_matrices=False)
    total = np.sum(S ** 2)
    cumulative = np.cumsum(S ** 2) / total
    return np.searchsorted(cumulative, threshold) + 1

print(f"FFN 权重有效秩: {effective_rank(weight_matrix)}")
```

## 6. 代码原理 / 架构原理

### 6.1 ndarray 内存布局

NumPy 数组在内存中是连续存储的，通过 strides（步长）机制实现多维索引：

```
一个 shape=(3, 4) 的 float64 数组：
内存: [a₀₀, a₀₁, a₀₂, a₀₃, a₁₀, a₁₁, a₁₂, a₁₃, a₂₀, a₂₁, a₂₂, a₂₃]
       ↑                                                  ↑
     offset=0                                         offset=88

strides = (32, 8)  表示：
  - 沿 axis=0 移动一步 = 跳过 32 字节（4 个 float64）
  - 沿 axis=1 移动一步 = 跳过 8 字节（1 个 float64）

a[i, j] 的内存地址 = base + i * strides[0] + j * strides[1]
```

**视图（View）与拷贝（Copy）**：
- 切片操作返回视图（共享内存）：`a[1:3]`、`a.T`、`a.reshape()`
- 高级索引返回拷贝：`a[[0, 2]]`、`a[a > 0]`
- `np.shares_memory(a, b)` 可检测两个数组是否共享内存

### 6.2 广播的实现原理

广播不实际复制数据，而是通过修改 strides 实现逻辑上的扩展：

```
a: shape=(3, 1), strides=(24, 0)  ← 注意 strides[1]=0，表示沿该维度"不移动"
b: shape=(1, 4), strides=(0, 8)   ← strides[0]=0

a + b 的结果：
- shape=(3, 4)
- a 的元素沿 axis=1 逻辑复制（stride=0）
- b 的元素沿 axis=0 逻辑复制（stride=0）
- 实际不复制任何数据，只在计算时"假装"数据被复制
```

### 6.3 NumPy 与 PyTorch 的互操作

```
NumPy ndarray ←→ PyTorch Tensor
        ↓                ↑
   .numpy()        torch.from_numpy()
        ↓                ↑
   共享内存条件：CPU 上的 Tensor + 不需要梯度
```

```python
import torch

# NumPy → PyTorch（共享内存）
arr = np.array([1, 2, 3])
tensor = torch.from_numpy(arr)  # 共享内存
arr[0] = 999
print(tensor[0])  # tensor(999) — 修改同步

# PyTorch → NumPy（共享内存）
tensor = torch.tensor([4, 5, 6])
arr = tensor.numpy()  # 共享内存
tensor[0] = 999
print(arr[0])  # 999 — 修改同步

# GPU 张量需要先转到 CPU
gpu_tensor = torch.randn(2, 3).cuda()
arr = gpu_tensor.cpu().numpy()  # 必须先 .cpu()，此步产生拷贝

# 带梯度的张量需要先 detach
x = torch.tensor([1.0, 2.0], requires_grad=True)
arr = x.detach().numpy()  # 必须 .detach()
```

## 7. 常见注意事项和最佳实践

### 7.1 性能优化

```python
# 1. 避免循环，使用向量化操作
# 差：Python 循环
result = np.empty(1000000)
for i in range(1000000):
    result[i] = np.sin(i) * np.cos(i)

# 好：向量化
x = np.arange(1000000)
result = np.sin(x) * np.cos(x)  # 快 100 倍以上

# 2. 预分配数组
# 差：逐步扩展
result = np.array([])
for i in range(1000):
    result = np.append(result, i)  # 每次都重新分配内存

# 好：预分配
result = np.empty(1000)
for i in range(1000):
    result[i] = i
# 更好：直接构造
result = np.arange(1000)

# 3. 使用原地操作减少内存分配
a = np.random.randn(1000, 1000)
np.add(a, 1, out=a)       # 原地加法，不创建新数组
np.multiply(a, 2, out=a)  # 原地乘法

# 4. 选择合适的数据类型
# float64 → float32 可节省一半内存
data = np.random.randn(10000, 10000).astype(np.float32)  # 400MB
# vs float64: 800MB
```

### 7.2 数值稳定性

```python
# 1. 数值稳定的 Softmax
def stable_softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 2. 数值稳定的 LogSumExp
def logsumexp(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    return x_max.squeeze(axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))

# 3. 避免大数相减（精度损失）
# 差
a, b = 1e10, 1e10 + 1
diff = a - b  # 可能丢失精度

# 好：使用 np.subtract 或重排计算

# 4. 使用 np.finfo 检查精度
print(np.finfo(np.float32).eps)   # 1.19e-07 — float32 最小精度
print(np.finfo(np.float16).eps)   # 9.77e-04 — float16 最小精度
print(np.finfo(np.float16).max)   # 65500.0 — float16 最大值
```

### 7.3 常见陷阱

```python
# 1. 视图 vs 拷贝混淆
a = np.array([1, 2, 3, 4, 5])
b = a[1:3]       # 视图（共享内存）
b[0] = 999
print(a)         # [1, 999, 3, 4, 5] — a 也被修改了！

c = a[1:3].copy()  # 显式拷贝，不影响原数组

# 2. 整数除法与浮点除法
# Python 3 中 / 是浮点除法，NumPy 中也如此
# 但 dtype=int 的数组运算仍为整数
a = np.array([1, 2, 3])
print(a / 2)      # [0.5, 1. , 1.5] — 浮点
print(a // 2)     # [0, 1, 1] — 整数除法

# 3. 布尔索引返回拷贝
a = np.array([1, 2, 3, 4, 5])
a[a > 2] = 0     # 这可以工作
# 但：
b = a[a > 0]
b[0] = 999       # 不影响 a，因为布尔索引返回拷贝

# 4. 广播导致的意外结果
a = np.array([[1, 2, 3]])   # (1, 3)
b = np.array([[1], [2]])    # (2, 1)
c = a + b                    # (2, 3) — 可能不是期望的结果
# 理解广播规则很重要

# 5. NaN 传播
a = np.array([1, 2, np.nan, 4])
print(np.mean(a))        # nan — NaN 会传播
print(np.nanmean(a))     # 2.333... — 忽略 NaN 的均值
print(np.nansum(a))      # 7.0 — 忽略 NaN 的求和
```

### 7.4 与 PyTorch 协同工作的最佳实践

```python
# 1. 数据预处理用 NumPy，训练用 PyTorch
# NumPy 适合：文本处理、特征工程、统计分析
# PyTorch 适合：模型训练、GPU 计算、自动求导

# 2. 注意内存共享
arr = np.random.randn(100, 768)
tensor = torch.from_numpy(arr)
# arr 和 tensor 共享内存，修改一个会影响另一个
# 如果需要独立副本：
tensor = torch.from_numpy(arr.copy())
# 或
tensor = torch.tensor(arr)  # 总是创建拷贝

# 3. 批量转换
# 多个 NumPy 数组 → PyTorch DataLoader
from torch.utils.data import TensorDataset, DataLoader

features = np.random.randn(1000, 768).astype(np.float32)
labels = np.random.randint(0, 10, size=1000)

dataset = TensorDataset(
    torch.from_numpy(features),
    torch.from_numpy(labels)
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```
