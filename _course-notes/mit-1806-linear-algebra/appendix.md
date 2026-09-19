---
title: "矩阵分解速查表"
collection: course-notes
chapter: true
permalink: /course-notes/mit-1806-linear-algebra/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/mit-1806-linear-algebra/) · [← l30](/course-notes/mit-1806-linear-algebra/l30)

{% raw %}
## 矩阵分解速查表

Strang 说："线性代数的全部故事，就是把矩阵拆开。"下表汇总本课程出现的五种核心分解。**记住"形式—条件—揭示什么—用来干什么"这四件事**，你就能在遇到问题时迅速定位工具。

### 总览表

| 分解 | 形式 | 存在条件 | 唯一性 | 揭示的结构 | 典型应用 |
|:---|:---|:---|:---|:---|:---|
| **A = LU** | $A = LU$（$L$ 单位下三角，$U$ 上三角） | 消元过程中不需要换行（所有主元非零，即各阶顺序主子式非零） | 唯一 | 主元、秩、行列式 $\\det A = \\prod u_{ii}$ | 解 $A\\mathbf{x}=\\mathbf{b}$（两次三角回代）；一次分解多次求解 |
| **PA = LU** | $PA = LU$（$P$ 置换） | **任何**可逆方阵都存在 | $P$ 与 $L,U$ 唯一（在给定选主元策略下） | 行交换的必要性；主元位置 | 数值稳定地解方程（部分选主元） |
| **A = QR** | $A = QR$（$Q$ 列标准正交，$R$ 上三角） | $A$ 的列线性无关（$m\\ge n$） | 当 $R$ 对角元为正时唯一 | 列空间的标准正交基；$\\lvert\\det A\\rvert = \\prod \\lvert r_{ii}\\rvert$ | 最小二乘（$R\\hat{\\mathbf{x}}=Q^{\\mathsf T}\\mathbf{b}$）；特征值的 QR 算法 |
| **A = SΛS⁻¹** | $A = S\\Lambda S^{-1}$（$S$ 为特征向量矩阵，$\\Lambda$ 对角） | $A$ 有 $n$ 个线性无关的特征向量（可对角化） | $\\Lambda$ 唯一（不计顺序），$S$ 不唯一 | 特征值、$A^k$、稳定性 | $A^k=S\\Lambda^kS^{-1}$；差分方程、马尔可夫链、Fibonacci |
| **A = QΛQᵀ** | $A = Q\\Lambda Q^{\\mathsf T}$（$Q$ 正交，$\\Lambda$ 实对角） | $A$ **对称**（$A=A^{\\mathsf T}$） | 特征值唯一，$Q$ 在特征值重复时不唯一 | 实特征值、正交特征向量、$A=\\sum\\lambda_i\\mathbf{q}_i\\mathbf{q}_i^{\\mathsf T}$ | 二次型、正定性、主轴、谱定理 |
| **A = UΣVᵀ** | $A = U\\Sigma V^{\\mathsf T}$（$U,V$ 正交，$\\Sigma$ 对角非负） | **任何** $m\\times n$ 矩阵都存在 | $\\Sigma$ 唯一；奇异向量在不重复时唯一（差符号） | 奇异值、秩、四个子空间的最优正交基、条件数 | 低秩逼近/图像压缩、PCA 降维、伪逆、病态分析 |
| **A = CR** | $A = CR$（$C$ 为主元列 $m\\times r$，$R$ 为 RREF 非零行 $r\\times n$） | **任何**矩阵（$r=\\operatorname{rank}A$） | $C,R$ 由选定的主元列唯一确定 | 秩的显式构造：秩 $r$ = $r$ 个独立列 × $r$ 个独立行 | 理解秩；$A=\\sum_{i=1}^{r}\\mathbf{c}_i\\mathbf{r}_i$（秩 1 之和） |

### 各有何用：一图定位

```
问题：解 A x = b
├─ A 可逆 / 超定？
│   ├─ 方阵、一般情形  ────────────→  PA = LU    （消元 + 回代，代价 n^3/3）
│   └─ 超定、需最小二乘 ───────────→  A  = QR     （正规方程变 Rx = Q^T b，更稳定）
│
问题：A 反复作用于向量（迭代 / 幂 / 演化）
├─ 可对角化 ──────────────────────→  A  = S Λ S^-1  （A^k = S Λ^k S^-1）
├─ 对称 ──────────────────────────→  A  = Q Λ Q^T   （实特征值、正交基）
└─ A 任意（含非方阵）─────────────→  A  = U Σ V^T   （σ_i 决定增长/衰减）

问题：A 的结构 / 压缩 / 降维
├─ 想知道"秩到底是多少"──────────→  A  = C R
├─ 想用最少信息近似 A ───────────→  A  = Σ σ_i u_i v_i^T ，截断前 k 项
└─ 想解病态或欠定系统 ───────────→  A^+ = V Σ^+ U^T（伪逆）
```

### 五种分解之间的联系

1. **$A=LU$ 与行列式**：$\\det A = \\prod_{i} u_{ii}$，也是 $\\prod \\sigma_i$ 的绝对值来源。
2. **$A=QR$ 与 $A^{\\mathsf T}A$**：$A=QR \\Rightarrow A^{\\mathsf T}A = R^{\\mathsf T}R$ —— 这正是 Cholesky 分解，也说明 $A^{\\mathsf T}A$ 正定（$R$ 可逆时）。
3. **$A=S\\Lambda S^{-1}$ 与 $A=U\\Sigma V^{\\mathsf T}$**：对称正定时两者重合，此时 $S=Q=U=V$ 且 $\\sigma_i=\\lambda_i$（奇异值就是特征值）。
4. **$A=CR$ 与 SVD**：$A=CR$ 是"秩 $r$"的朴素表达，$A=\\sum\\sigma_i\\mathbf{u}_i\\mathbf{v}_i^{\\mathsf T}$ 是它的**最优正交版本**。
5. **$A=U\\Sigma V^{\\mathsf T}$ 与四个基本子空间**（课程的顶点）：
   - $\\mathbf{u}_1,\\dots,\\mathbf{u}_r$：$C(A)$ 的标准正交基
   - $\\mathbf{u}_{r+1},\\dots,\\mathbf{u}_m$：$N(A^{\\mathsf T})$ 的标准正交基
   - $\\mathbf{v}_1,\\dots,\\mathbf{v}_r$：$C(A^{\\mathsf T})$ 的标准正交基
   - $\\mathbf{v}_{r+1},\\dots,\\mathbf{v}_n$：$N(A)$ 的标准正交基

```
        A = U Σ V^T  的四个子空间结构
  R^n :  v_1 ... v_r | v_{r+1} ... v_n        R^m :  u_1 ... u_r | u_{r+1} ... u_m
         └─ C(A^T) ──┘ └─── N(A) ───┘               └── C(A) ──┘ └─ N(A^T) ──┘
                  ↓ A                                    ↑
              σ_1 ... σ_r  (伸缩)                 其余方向 → 0
```

### 关键数值关系速记

| 关系 | 公式 | 出处 |
|:---|:---|:---|
| 秩与子空间维数 | $\\dim C(A)=\\dim C(A^{\\mathsf T})=r$，$\\dim N(A)=n-r$，$\\dim N(A^{\\mathsf T})=m-r$ | Lecture 10 |
| 特征值之和/积 | $\\sum\\lambda_i=\\operatorname{tr}(A)$，$\\prod\\lambda_i=\\det(A)$ | Lecture 21 |
| 奇异值与特征值 | $\\sigma_i=\\sqrt{\\lambda_i(A^{\\mathsf T}A)}=\\sqrt{\\lambda_i(AA^{\\mathsf T})}$ | Lecture 29 |
| 行列式与奇异值 | $\\lvert\\det A\\rvert=\\prod_i\\sigma_i$（方阵） | Lecture 20, 29 |
| 谱范数 | $\\lVert A\\rVert_2=\\sigma_1$ | Lecture 29 |
| 条件数 | $\\kappa(A)=\\sigma_1/\\sigma_r$ | Lecture 29 |
| 投影矩阵 | $P=A(A^{\\mathsf T}A)^{-1}A^{\\mathsf T}$，$P^{\\mathsf T}=P$，$P^2=P$ | Lecture 15-16 |
| 最小二乘 | $A^{\\mathsf T}A\\hat{\\mathbf{x}}=A^{\\mathsf T}\\mathbf{b}$ | Lecture 16 |
| 伪逆 | $A^{+}=V\\Sigma^{+}U^{\\mathsf T}$，$\\hat{\\mathbf{x}}=A^{+}\\mathbf{b}$ 为最小范数最小二乘解 | Lecture 33 |
| 幂的收敛 | $A^k\\to 0 \\iff$ 所有 $\\lvert\\lambda_i\\rvert<1$ | Lecture 22 |
| 微分方程稳定 | $\\mathbf{u}(t)\\to\\mathbf{0} \\iff$ 所有 $\\operatorname{Re}(\\lambda_i)<0$ | Lecture 23 |

### 选分解的决策清单

1. **只需要解一次方程？** → 直接用消元（$PA=LU$），不必显式求逆。
2. **要解很多个不同的 $\\mathbf{b}$（同一个 $A$）？** → 先做一次 $A=LU$，之后每个 $\\mathbf{b}$ 只需 $O(n^2)$。
3. **方程个数多于未知数（超定）？** → 最小二乘 + $A=QR$。**永远不要**显式算 $(A^{\\mathsf T}A)^{-1}$。
4. **要算 $A^{100}$ 或迭代 $A^k\\mathbf{x}$？** → 对角化 $A=S\\Lambda S^{-1}$，看 $\\lvert\\lambda\\rvert$ 判断收敛。
5. **矩阵对称且问"是否正定/是否有极小值"？** → $A=Q\\Lambda Q^{\\mathsf T}$，看 $\\lambda_i$ 的符号。
6. **矩阵非方阵、或不可对角化、或问"最重要的方向是什么"？** → SVD $A=U\\Sigma V^{\\mathsf T}$，看 $\\sigma_1 \\ge \\sigma_2 \\ge \\cdots$。
7. **只想说清"秩是多少"并给出基？** → $A=CR$。

---

## 结语：一张图记住 18.06

```
                       A x = b
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   何时有解？                           解有多少个？
   b ∈ C(A) ?                          N(A) 的维数 = n - r
        │                                   │
        └─────────────────┬─────────────────┘
                          │
                 四个基本子空间（Lecture 10）
                          │
      ┌───────────┬───────┴────────┬──────────────┐
      │           │                │              │
   A = LU      A = QR        A = SΛS⁻¹      A = UΣVᵀ
  （消元）    （正交化）      （特征值）      （奇异值）
      │           │                │              │
   解方程      最小二乘        幂与稳定性      一切矩阵的
   与主元      与拟合          与微分方程      最优正交结构
```

**最后一句话**：线性代数的核心不是计算，而是**看穿矩阵背后正在发生的几何**——哪些方向被保留、哪些被压扁、哪些被旋转。掌握四个基本子空间与五种分解，你就掌握了这幅图的全部门票。

{% endraw %}
