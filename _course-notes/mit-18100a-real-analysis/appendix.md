---
title: "核心定义与定理速查表"
collection: course-notes
chapter: true
permalink: /course-notes/mit-18100a-real-analysis/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/mit-18100a-real-analysis/) · [← l25](/course-notes/mit-18100a-real-analysis/l25)

{% raw %}
## 核心定义与定理速查表

> **使用说明**：本表覆盖 Lecture 1–25 的全部官方编号（`Definition` / `Theorem` / `Axiom` / `Corollary` / `Lemma` / `Example` / `Remark` / `Negation` / `Notation` / `Recall` / `Question`）。编号与 OCW 官方 *Complete Lecture Notes* 逐条对应；凡官方笔记未给出编号的陈述，本表标注「（见 Lecture N）」，**不编造编号**。
>
> **阅读顺序建议**：先读第十节的「证明技巧总表」，再回读第二、三节的「实数公理」与「序列极限」——这两节是全课的发动机。

---

### 一、语言与逻辑（Lecture 1–3）

#### 1.1 课程目标与集合语言

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Remark 1 | 课程两大目标 | (1) 获得证明的经验；(2) 证明关于实数、函数与极限的命题。 | L1 |
| Definition 2 | 集合关系 | $A\subset B$ 当且仅当 $a\in A\Rightarrow a\in B$；集合由元素完全确定。 | L1 |
| Problem 3 | 如何描述 $\mathbb{R}$？ | 在 Lecture 3、4 回答；此前先建立集合与证明方法。 | L1 |
| Theorem 4 | De Morgan 律 | $(B\cup C)^c=B^c\cap C^c$，$(B\cap C)^c=B^c\cup C^c$。 | L1 |
| Remark 7 | 反证法 | 假设结论为假，推出假命题，故原命题为真。 | L1 |
| Remark 17 | 分情况讨论 | 若每一分支结论都成立，则总结论成立。 | L3 |

**量词的否定规则（全课最常考的操作，见 Lecture 1–2 与 Final Assignment 第 1 题）**：

```
   ┌────────────────────────┬────────────────────────┐
   │  原命题                 │  否定                   │
   ├────────────────────────┼────────────────────────┤
   │  ∀x, P(x)              │  ∃x, ¬P(x)              │
   │  ∃x, P(x)              │  ∀x, ¬P(x)              │
   │  ∀ε>0 ∃δ>0 ∀x (... )   │  ∃ε>0 ∀δ>0 ∃x (¬... )   │
   │  P ⇒ Q                 │  P 且 ¬Q                │
   │  ∀n≥M, |aₙ-x|<ε        │  ∃ε₀>0 ∀M ∃n≥M, |aₙ-x|≥ε₀│
   └────────────────────────┴────────────────────────┘

   口诀：∀ ↔ ∃ 互换，最后一个断言取反，"且"与"或"互换。
```

#### 1.2 良序原理、归纳法、Bernoulli

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Axiom 5 | 良序原理 | 若 $S\subset\mathbb{N}$ 非空，则 $\exists x\in S$ 使 $x\le y$ 对一切 $y\in S$；即 $\mathbb{N}$ 的任何非空子集有最小元。 | L1 |
| Theorem 6 | 数学归纳法 | 设 $P(n)$ 依赖于 $n\in\mathbb{N}$。若 (1) $P(1)$ 真；(2) $P(m)$ 真 $\Rightarrow P(m+1)$ 真；则 $P(n)$ 对一切 $n\in\mathbb{N}$ 真。 | L1 |
| Theorem 8 | 有限几何和 | $\displaystyle 1+c+c^2+\cdots+c^n=\frac{1-c^{n+1}}{1-c}$，对一切 $c\ne 1$，$n\in\mathbb{N}$。 | L1 |
| Theorem 9 | Bernoulli 不等式 | 对一切 $c\ge -1$ 与 $n\in\mathbb{N}$，$(1+c)^n\ge 1+nc$。 | L1 |
| Corollary 18 | $n<2^n$ | 对一切 $n\in\mathbb{N}\cup\{0\}$，$n<2^n$。 | L3 |
| Remark 19 | 另证 | $n<2^n$ 亦可用归纳法证明（见 Assignment 1）。 | L3 |

$$\text{Bernoulli 的用途：}\ (1+c)^n\ge 1+nc\ \Longrightarrow\ \text{取 }c=\tfrac1n\ \text{得}\ \left(1+\tfrac1n\right)^n\ge 2,\quad \text{以及}\ \sqrt[n]{1+nc}\le 1+c.$$

**归纳法三条路线（证明 $P(n)$ 时的选择）**：

| 形式 | 归纳步假设 | 典型用途 |
|:--|:--|:--|
| 普通归纳 | $P(m)$ | 绝大多数「对一切 $n$」命题 |
| 强归纳 | $P(1),\dots,P(m)$ 全部 | 需要多个前项（如因式分解、递推界） |
| 良序原理反证 | 取反例集 $S\ne\varnothing$，取其最小元 | 当归纳步难以「从 $m$ 到 $m+1$」直接走通 |

#### 1.3 基数：可数与不可数

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Question 10 | 何时两集合「一样大」？ | Cantor：当且仅当能把一个集合的元素与另一个集合的元素一一配对。 | L2 |
| Definition 11 | 基数 | $|A|=|B|$ 当且仅当存在双射 $f:A\to B$；$|A|\le|B|$ 当且仅当存在单射 $A\to B$。 | L2 |
| Theorem 12 | Cantor–Schröder–Bernstein | 若 $|A|\le|B|$ 且 $|B|\le|A|$，则 $|A|=|B|$。若 $|A|=|\mathbb{N}|$ 则 $A$ 可数无穷；有限或可数无穷统称**可数**；否则称**不可数**。 | L2 |
| Example 13 | 具体等势 | $|\{2n\mid n\in\mathbb{N}\}|=|\mathbb{N}|$；$|\{2n-1\mid n\in\mathbb{N}\}|=|\mathbb{N}|$。 | L2 |
| Question 14 | 有比 $\mathbb{N}$ 更大的集合吗？ | 幂集 $P(A)=\{B\mid B\subset A\}$；$P(\varnothing)=\{\varnothing\}$，$P(\{1\})=\{\varnothing,\{1\}\}$。 | L3 |
| Theorem 15 | Cantor 定理 | 对任何集合 $A$，$|A|<|P(A)|$。 | L3 |
| Remark 16 | 无穷的无穷 | $\mathbb{N}<|P(\mathbb{N})|<|P(P(\mathbb{N}))|<\cdots$，故有无穷多个不同大小的无穷集。 | L3 |
| Theorem 22 | 实数的刻画 | 存在唯一的含 $\mathbb{Q}$ 的有序域具有最小上界性质，记作 $\mathbb{R}$。 | L3 |
| Definition 55 | 十进制展开 | $x\in(0,1]$ 由数字 $\{d_{-j}\}$ 表示：$x=\sup\{10^{-1}d_{-1}+\cdots+10^{-n}d_{-n}\mid n\in\mathbb{N}\}$。 | L6 |
| Theorem 56 | 数字表示唯一性 | 每个 $x\in(0,1]$ 有唯一数字序列使 $0\bullet d_{-1}\dots d_{-n}<x\le 0\bullet d_{-1}\dots d_{-n}+10^{-n}$。 | L6 |
| Theorem 57 | Cantor | $(0,1]$ 不可数。 | L6 |
| Corollary 58 | $\mathbb{R}$ 不可数 | 实数集 $\mathbb{R}$ 不可数。 | L6 |
| Recall 54 | 可数的复习 | $A$ 可数 $\iff$ $A$ 有限或 $|A|=|\mathbb{N}|$。 | L6 |

**对角线论证的骨架（Theorem 57 证明机制）**：

```
   假设 (0,1] 可数 ⇒ 存在双射 x: ℕ → (0,1]，把元素排成表：

      x₁ = 0 . d₁₁ d₁₂ d₁₃ d₁₄ ...
      x₂ = 0 . d₂₁ d₂₂ d₂₃ d₂₄ ...
      x₃ = 0 . d₃₁ d₃₂ d₃₃ d₃₄ ...
      x₄ = 0 . d₄₁ d₄₂ d₄₃ d₄₄ ...
                 ⋮

   构造 y = 0 . e₁ e₂ e₃ e₄ ...   其中 eₙ ≠ dₙₙ（并避开 9 以避免进位歧义）
   ⇒ 对每个 n，y ≠ xₙ（第 n 位数字不同）
   ⇒ y ∈ (0,1] 但不在表中 ⇒ 与「双射」矛盾 ⇒ (0,1] 不可数。

   关键：对角线保证 y 与每一个 xₙ 都在「专属的那一位」上不同。
```

**可数性工具箱（见 Lecture 2–3 与 Assignment 1）**：

- $\mathbb{Z}$、$\mathbb{Q}$ 可数；可数个可数集的并仍可数。
- $|A|=|\mathbb{N}|$ 的常用手段：把 $A$ 的元素**排成一个序列**（而不是列举成集合）。
- Remark 62（L6）：**序列不是集合**，例如 $-1,1,-1,1,\dots=\{(-1)^n\}_{n=1}^\infty$ 与集合 $\{-1,1\}$ 完全不同。

---

### 二、实数系公理（Lecture 4–5）

#### 2.1 有序集、上/下界、LUB

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Remark 20 | 实数是什么 | 在某种意义上，$\mathbb{R}$ 是「具有 $\mathbb{Q}$ 的全部代数与序性质，但没有洞」的唯一集合。 | L3 |
| Problem 21 | 精确刻画 $\mathbb{R}$ | 先陈述最终结论，再推导。 | L3 |
| Definition 23 | 有序集 | 集合 $S$ 带关系 $<$： (1) $\forall x,y\in S$ 恰有 $x<y$、$y<x$、$x=y$ 之一；(2) $x<y$ 且 $y<z$ $\Rightarrow x<z$。 | L3 |
| Definition 24 | 上界/下界 | $E$ 有上界：$\exists b\in S$，$\forall x\in E$，$x\le b$（$b$ 为上界）；下界类似。 | L3 |
| Definition 26 | 最小上界性质 (LUB) | 有序集 $S$ 具有 LUB 性质，若每个非空且有上界的 $E\subset S$ 在 $S$ 中有上确界 $\sup E$。 | L3 |
| Example 25 | sup/inf 例 | $S=\mathbb{Z}$、$E=\{-2,-1,0,1,2\}$：$\inf E=-2$，$\sup E=2$。注意 sup/inf 不必属于 $E$。 | L3 |
| Theorem 27 | $\mathbb{Q}$ 的洞（一） | 若 $x=\sup\{q\in\mathbb{Q}\mid q>0,\ q^2<2\}$，则 $x>0$ 且 $x^2=2$。 | L3 |
| Theorem 28 | $\mathbb{Q}$ 无 LUB | $E=\{q\in\mathbb{Q}\mid q>0,\ q^2<2\}$ 在 $\mathbb{Q}$ 中没有上确界。 | L3 |
| Definition 30 | 域 | 集合 $F$ 带 $+,\cdot$ 满足 A1)–A4)（加法封闭、交换、结合、零元）与 M1)–M4)（乘法封闭、交换、结合、单位元）、D) 分配律，以及加法与乘法逆元。 | L4 |
| Example 31 | 有限域 | $\mathbb{Z}_2=\{0,1\}$（$1+1=0$）；$\mathbb{Z}_3=\{0,1,2\}$（模 3）。 | L4 |
| Theorem 32 | $0\cdot x=0$ | 若 $x\in F$（$F$ 为域），则 $0x=0$。 | L4 |
| Definition 33 | 有序域 | 域 $F$ 兼有序集： (i) $x<y\Rightarrow x+z<y+z$；(ii) $x>0$ 且 $y>0$ $\Rightarrow xy>0$。 | L4 |
| Example 34 | 有序域例与非例 | $\mathbb{Q}$ 是有序域；$\mathbb{Z}_2$ **不是**（设 $0<1$ 则 $1<0$，矛盾）。 | L4 |
| Theorem 35 | 取负变号 | 若 $x>0$ 则 $-x<0$（反之亦然）。 | L4 |
| Theorem 36 | 符号法则 | $x>0,y<0$ 或 $x<0,y>0$ $\Rightarrow xy<0$。 | L4 |
| Question 37 | 有最大下界性质吗？ | 有序域若有 LUB 性质，则也有最大下界性质。 | L4 |
| Theorem 38 | 下确界存在 | 若 $F$ 是有 LUB 性质的有序域，$A\subset F$ 非空且有下界，则 $\inf A$ 在 $F$ 中存在。 | L4 |
| Theorem 39 | $\mathbb{R}$ 的唯一性 | 存在「唯一」的有序域 $\mathbb{R}\supset\mathbb{Q}$ 具有 LUB 性质（可由 Dedekind 分割或等价类构造）。 | L4 |
| Theorem 40 | $\sqrt2\in\mathbb{R}$ | 存在唯一 $r\in\mathbb{R}$ 使 $r>0$ 且 $r^2=2$；即 $\sqrt2\in\mathbb{R}$ 但 $\sqrt2\notin\mathbb{Q}$。 | L4 |
| Remark 41 | 更高次根 | Assignment 2 Exercise 7 将证 $\sqrt[3]{2}\in\mathbb{R}$。 | L4 |

#### 2.2 Archimedes、稠密性、绝对值、sup 的 $\epsilon$ 刻画

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Question 42 | 有理数能插进去吗？ | 求 $r\in\mathbb{Q}$ 使 $x<r<y$？答案：能。 | L5 |
| Theorem 43 | Archimedes 性质 + $\mathbb{Q}$ 稠密 | (i) 若 $x,y\in\mathbb{R}$，$x>0$，则 $\exists n\in\mathbb{N}$，$nx>y$；(ii) 若 $x,y\in\mathbb{R}$，$x<y$，则 $\exists r\in\mathbb{Q}$，$x<r<y$。 | L5 |
| Theorem 44 | 另一极限上限 | $\displaystyle 1=\sup\left\{1-\frac1n\ \middle|\ n\in\mathbb{N}\right\}$（见 Lecture 5）。 | L5 |
| Theorem 45 | sup 的 $\epsilon$ 刻画 | 设 $S\subset\mathbb{R}$ 非空有上界。则 $x=\sup S$ 当且仅当 (1) $x$ 是 $S$ 的上界；(2) $\forall\epsilon>0$，$\exists y\in S$ 使 $x-\epsilon<y\le x$。 | L5 |
| Theorem 47 | 平移与 sup | 若 $x\in\mathbb{R}$ 且 $A$ 有上界，则 $x+A$ 有上界且 $\sup(x+A)=x+\sup A$。 | L5 |
| Theorem 48 | sup 与 inf 的分离 | 设 $A,B\subset\mathbb{R}$ 且 $\forall x\in A,\forall y\in B$，$x\le y$。则 $\sup A\le\inf B$。 | L5 |
| Definition 49 | 绝对值 | $\displaystyle |x|:=\begin{cases}x,&x\ge 0\\ -x,&x<0\end{cases}$ | L5 |
| Theorem 50 | 绝对值六条 | (1) $|x|\ge0$ 且 $|x|=0\iff x=0$；(2) $|-x|=|x|$；(3) $|xy|=|x||y|$；(4) $x\le|x|$，$-x\le|x|$；(5) $|x|\le c\iff -c\le x\le c$（$c\ge0$）；(6) $-|x|\le x\le|x|$。 | L5 |
| Theorem 51 | 三角不等式 | $\forall x,y\in\mathbb{R}$，$|x+y|\le|x|+|y|$。 | L6 |
| Remark 52 | $\Delta$-不等式 | 三角不等式简称 $\Delta$-inequality。 | L6 |

$$\boxed{\ \bigl||x|-|y|\bigr|\le|x-y|,\qquad |x+y+z|\ge|x|-|y|-|z|\ (\text{Thm 208, L18})\ }$$

#### 2.3 LUB → Archimedes → 稠密性 的依赖关系（**本节重点**）

```
                     ┌──────────────────────────────────────────┐
                     │  ℝ 是有序域（Def 30 + Def 33）           │
                     └────────────────────┬─────────────────────┘
                                          │
                     ┌────────────────────▼─────────────────────┐
                     │  LUB 性质（Def 26 / Thm 22 / Thm 39）     │
                     │  非空有上界 ⇒ sup 存在                    │
                     └────────────────────┬─────────────────────┘
                                          │  单向依赖（不可逆！）
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
   ┌────────────────────┐   ┌────────────────────────┐   ┌──────────────────────┐
   │ Archimedes 性质     │   │ 下确界存在（Thm 38）    │   │ sup 的 ε 刻画        │
   │ Thm 43 (i)         │   │ 由 -A 的 sup 取负得到   │   │ Thm 45               │
   │ ∀x>0 ∃n: nx>y      │   └───────────┬────────────┘   └──────────┬───────────┘
   └─────────┬──────────┘               │                           │
             │                          │                           │
             └──────────────┬───────────┴───────────────────────────┘
                            ▼
             ┌──────────────────────────────────────────┐
             │ ℚ 稠密性  Thm 43 (ii)                     │
             │ x<y ⇒ ∃r∈ℚ, x<r<y                        │
             │ 证明：取 n 使 n(y-x)>1（Archimedes），     │
             │ 再取 m=⌊nx⌋+1，则 r=m/n 满足要求           │
             └────────────────────┬─────────────────────┘
                                  ▼
             ┌──────────────────────────────────────────┐
             │ 无理数稠密性（见 Lecture 5 习题）          │
             │ x<y ⇒ ∃无理数 ξ, x<ξ<y                    │
             │ 证明：对 x/√2<y/√2 用 ℚ 稠密得 r，再取 ξ=r√2│
             └────────────────────┬─────────────────────┘
                                  ▼
             ┌──────────────────────────────────────────┐
             │ 推论：任意开区间含无穷多有理数与无理数     │
             │ 推论：∀ε>0 ∃n∈ℕ, 1/n<ε                    │
             └──────────────────────────────────────────┘

   反面：ℚ 具有有序域的全部性质 + Archimedes 性质，
        但 ℚ 没有 LUB 性质（Thm 28：E={q>0: q²<2} 无 sup）。
        ⇒ LUB 是 ℝ 与 ℚ 之间唯一的分界线。
        ⇒ Thm 43(i) 的证明必须用 LUB（不能只用有序域公理）。
```

---

### 三、序列极限（Lecture 6–9）

#### 3.1 定义、否定、唯一性、有界性

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Remark 59 | 分析是什么 | 分析就是研究极限的学问。 | L6 |
| Definition 60 | 实数列 | 实数列是函数 $x:\mathbb{N}\to\mathbb{R}$，记 $x(n)=x_n$，$\{x_n\}_{n=1}^\infty$。 | L6 |
| Definition 61 | 有界 | $\{x_n\}$ 有界，若 $\exists B\ge0$，$\forall n$，$|x_n|\le B$。 | L6 |
| Remark 62 | 序列 ≠ 集合 | $-1,1,-1,1,\dots$ 作为序列与集合 $\{-1,1\}$ 不同。 | L6 |
| Definition 63 | 收敛（$\epsilon$-$N$） | $\{x_n\}$ 收敛到 $x$，若 $\forall\epsilon>0$，$\exists M\in\mathbb{N}$，$\forall n\ge M$，$|x_n-x|<\epsilon$。收敛者称**收敛**，否则称**发散**。 | L6 |
| Negation 64 | 不收敛 | $\exists\epsilon_0>0$，$\forall M\in\mathbb{N}$，$\exists n\ge M$，$|x_n-x|\ge\epsilon_0$。 | L6 |
| Theorem 65 | 极限唯一 | 若 $\{x_n\}$ 收敛到 $x$ 又收敛到 $y$，则 $x=y$。 | L6 |
| Theorem 66 | $\epsilon$ 任意小则相等 | 若 $\forall\epsilon>0$，$|x-y|<\epsilon$，则 $x=y$。 | L6 |
| Example 68 | 常数列 | $x_n=c$ $\forall n$ $\Rightarrow$ $\lim_{n\to\infty}x_n=c$。 | L6 |
| Example 69 | $1/n\to0$ | $\lim_{n\to\infty}\frac1n=0$。 | L6 |
| Example 70 | 稍复杂的分式 | $\lim_{n\to\infty}\frac{1}{n^2+2n+100}=0$。 | L7 |
| Example 71 | $(-1)^n$ 发散 | $x_n=(-1)^n$ 发散（$\epsilon_0=1$ 即可截住）。 | L7 |
| Theorem 72 | 收敛 ⇒ 有界 | 若 $\{x_n\}$ 收敛，则 $\{x_n\}$ 有界。 | L7 |
| Notation 81 | DNC | 「序列不收敛」记作 DNC (does not converge)。 | L7 |

$$\text{收敛：}\ \forall\epsilon>0\ \exists M\in\mathbb{N}\ \forall n\ge M,\ |x_n-x|<\epsilon. \qquad \text{发散：}\ \exists\epsilon_0>0\ \forall M\in\mathbb{N}\ \exists n\ge M,\ |x_n-x|\ge\epsilon_0.$$

**$\epsilon$-$N$ 证明的标准四步（见 Lecture 6–7 全部例题）**：

1. 写「令 $\epsilon>0$」；2. 把 $|x_n-x|$ **放大成只含 $n$ 的表达式**（常用三角不等式、有理化、$|x_n|\le B$）；3. 令放大后的式子 $<\epsilon$ 反解 $M$（取 $M=\max\{\dots\}$）；4. 回代验证 $n\ge M$ 时原式 $<\epsilon$。

#### 3.2 单调有界、子序列

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Definition 73 | 单调 | 单调递增：$\forall n$，$x_n\le x_{n+1}$；单调递减：$\forall n$，$x_n\ge x_{n+1}$。 | L7 |
| Example 74 | 单调例 | $x_n=1/n$ 单调（递减）；$y_n=-1/n$ 单调递增；$(-1)^n$ 既不增也不减。 | L7 |
| Theorem 75 | 单调有界（增） | $\{x_n\}$ 单调递增 $\Rightarrow$ $\{x_n\}$ 收敛 $\iff$ $\{x_n\}$ 有界；且 $\lim_{n\to\infty}x_n=\sup\{x_n\mid n\in\mathbb{N}\}$。 | L7 |
| Theorem 76 | 单调有界（减） | $\{x_n\}$ 单调递减 $\Rightarrow$ $\{x_n\}$ 收敛 $\iff$ $\{x_n\}$ 有界；且 $\lim_{n\to\infty}x_n=\inf\{x_n\mid n\in\mathbb{N}\}$。 | L7 |
| Definition 77 | 子序列 | $\{n_k\}$ 严格递增，则 $\{x_{n_k}\}$ 为 $\{x_n\}$ 的子序列。 | L7 |
| Question 78 | 哪些不是子序列 | $1,1,1,1,\dots$；$x_1,x_1,x_2,x_2,\dots$ 等**不是**子序列。 | L7 |
| Theorem 79 | 子序列同极限 | 若 $x_n\to x$，则 $x_{n_k}\to x$。**特别地 $n_k\ge k$**，这是所有子序列估计的起点。 | L7 |
| Remark 80 | 推论 | 由此可知 $\{(-1)^n\}$ 发散（两条子序列极限不同）。 | L7 |

$$\text{Thm 75 的机制：}\ \lim x_n=\sup\{x_n\}\ \text{——把「极限存在」化归为「集合有 sup」，这就是 LUB 的直接消费点。}$$

#### 3.3 极限的运算（**全部公式**）

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Theorem 82 | 夹逼定理 (ST) | 若 $\forall n$，$a_n\le x_n\le b_n$，且 $a_n\to x$、$b_n\to x$，则 $x_n\to x$。 | L8 |
| Remark 83 | 缩写 | Squeeze Theorem 缩写 ST。 | L8 |
| Theorem 84 | 检验收敛的另一写法 | 给出判定 $x_n\to x$ 的另一种等价表述（见 Lecture 8 完整陈述）。 | L8 |
| Example 85 | 夹逼例 | 见 Lecture 8（用 $0\le\dots\le$ 可忽略项夹逼）。 | L8 |
| Theorem 87 | 保序 | (1) $\{x_n\},\{y_n\}$ 收敛且 $\forall n$，$x_n\le y_n$ $\Rightarrow$ $\lim x_n\le\lim y_n$；(2) 由此得「$\forall n\ x_n\le c$ $\Rightarrow$ $\lim x_n\le c$」。 | L8 |
| Theorem 89 | 四则运算 | 若 $x_n\to x$，$y_n\to y$，则 (1) $x_n+y_n\to x+y$；(2) $\forall c\in\mathbb{R}$，$cx_n\to cx$。 | L8 |
| Remark 90 | 幂 | 由归纳法，$\lim_{n\to\infty}(x_n)^k=x^k$。 | L8 |
| Theorem 91 | 平方根 | 若 $\{x_n\}$ 收敛且 $\forall n$，$x_n\ge0$，则 $\sqrt{x_n}\to\sqrt x$。 | L8 |
| Remark 92 | 为何要分情况 | 证明中需分 $x=0$ 与 $x>0$（前者用 $\sqrt{x_n}<\epsilon\iff x_n<\epsilon^2$）。 | L8 |
| Theorem 93 | 绝对值 | 若 $x_n\to x$，则 $|x_n|\to|x|$。 | L8 |
| Theorem 94 | 几何序列 | 若 $c\in(0,1)$ 则 $\lim_{n\to\infty}c^n=0$；若 $c>1$ 则 $\{c^n\}$ 无界。 | L8 |

$$\lim(x_n+y_n)=x+y,\quad \lim(cx_n)=cx,\quad \lim(x_n-y_n)=x-y,\quad \lim(x_n^k)=x^k,$$
$$\lim_{n\to\infty}\frac1{x_n}=\frac1x\ (x\ne0,\ x_n\ne0),\qquad \lim_{n\to\infty}\sqrt{x_n}=\sqrt x,\qquad \lim_{n\to\infty}|x_n|=|x|.$$

**乘积极限的证明技巧（Lecture 8，官方证明的核心）**：

$$x_ny_n-xy=x_n(y_n-y)+y(x_n-x),\qquad |x_ny_n-xy|\le \underbrace{|x_n|}_{\le B}|y_n-y|+\underbrace{|y|}_{|y|+1}|x_n-x|,$$

**加一项减一项**，再用收敛序列有界（Thm 72）把 $|x_n|$ 换成常数 $B=\max\{|x_1|,\dots,|x_{M-1}|,|x|+1\}$。

#### 3.4 limsup / liminf、BW 定理

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Theorem 95 | 特殊序列 | (1) $p>0$ $\Rightarrow$ $n^{-p}\to0$；(2) 含 $p^{1/n}$ 的极限（见 Lecture 9）；(3)、(4) 为开方型特殊序列（完整列表见 Lecture 9）。 | L9 |
| Question 96 | 有界序列必有收敛子列吗？ | 见 Theorem 101/102。 | L9 |
| Definition 97 | limsup / liminf | 对有界 $\{x_n\}$：$\displaystyle\limsup_{n\to\infty}x_n=\lim_{n\to\infty}\bigl(\sup\{x_k\mid k\ge n\}\bigr)$，$\displaystyle\liminf_{n\to\infty}x_n=\lim_{n\to\infty}\bigl(\inf\{x_k\mid k\ge n\}\bigr)$，若极限存在。 | L9 |
| Theorem 98 | 等价的单调刻画 | 令 $a_n=\sup\{x_k\mid k\ge n\}$，$b_n=\inf\{x_k\mid k\ge n\}$，则 $\{a_n\}$ 单调递减、$\{b_n\}$ 单调递增，且 $\limsup x_n=\lim a_n$、$\liminf x_n=\lim b_n$（对一切有界序列都存在）。 | L9 |
| Example 99 | $(-1)^n$ | $\limsup=1$，$\liminf=-1$。 | L9 |
| Example 100 | $1/n$ | $\limsup=\liminf=0$。 | L9 |
| Theorem 101 | 子列的极限实现 | 有界 $\{x_n\}$ 存在子列 $\{x_{n_k}\}$ 使 $\lim_k x_{n_k}=\limsup_n x_n$，且存在子列 $\{x_{m_k}\}$ 使 $\lim_k x_{m_k}=\liminf_n x_n$。 | L9 |
| Theorem 102 | Bolzano–Weierstrass | 任何有界序列都有收敛子序列。 | L9 |
| Remark 103 | 缩写 | Bolzano–Weierstrass 缩写 B-W。 | L9 |
| Notation 104 | 简写 | 上下文清楚时可省略上下极限的下标。 | L9 |
| Theorem 105 | 收敛 $\iff$ liminf = limsup | 有界 $\{x_n\}$ 收敛 $\iff$ $\liminf x_n=\limsup x_n$（此时公共值即为极限）。 | L9 |

$$\inf\{x_k\mid k\ge n\}\le x_n\le\sup\{x_k\mid k\ge n\},\qquad \liminf x_n\le\limsup x_n\ \text{恒成立}.$$

```
   limsup / liminf 与收敛的关系（Thm 105）

        有界序列 {xₙ}
              │
              ├── liminf xₙ = limsup xₙ ──► {xₙ} 收敛，极限 = 公共值
              │        （"上下包夹收成一点"）
              │
              └── liminf xₙ < limsup xₙ ──► {xₙ} 发散（振荡）
                       │                       例：(-1)ⁿ：-1 < 1
                       │
                       └── 但仍有收敛子列（Thm 101/102）：
                           取到 limsup 的子列 + 取到 liminf 的子列

   对比：limsup 永远存在（对任何有界序列）；lim 可能不存在。
        limsup 是「最终上包络的极限」，不是「最大值」。
```

---

### 四、级数（Lecture 10–12）

#### 4.1 Cauchy 序列、完备性、级数的基本判据

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Definition 106 | Cauchy 序列 | $\forall\epsilon>0$，$\exists M\in\mathbb{N}$，$\forall n,k\ge M$，$|x_n-x_k|<\epsilon$。 | L10 |
| Example 107 | $1/n$ 是 Cauchy | $x_n=1/n$ 是 Cauchy 序列。 | L10 |
| Negation 108 | 非 Cauchy | $\exists\epsilon_0>0$，$\forall M\in\mathbb{N}$，$\exists n,k\ge M$，$|x_n-x_k|\ge\epsilon_0$。 | L10 |
| Example 109 | $(-1)^n$ 非 Cauchy | 取 $\epsilon_0=1$，$n=M$，$k=M+1$：$|(-1)^n-(-1)^k|=2\ge1$。 | L10 |
| Theorem 110 | Cauchy ⇒ 有界 | 若 $\{x_n\}$ 是 Cauchy，则 $\{x_n\}$ 有界。 | L10 |
| Theorem 111 | Cauchy + 收敛子列 ⇒ 收敛 | 若 $\{x_n\}$ 是 Cauchy 且有子列收敛，则 $\{x_n\}$ 收敛。 | L10 |
| Theorem 112 | 完备性 | 实数列 $\{x_n\}$ 是 Cauchy $\iff$ $\{x_n\}$ 收敛。 | L10 |
| Remark 113 | 历史 | 级数是最初促使分析诞生的动机。 | L10 |
| Definition 114 | 级数 | 给定 $\{x_n\}$，符号 $\sum_{n=1}^\infty x_n$ 为关联级数；$\sum x_n$ 收敛，若部分和序列 $\{s_m\}$（$s_m=\sum_{n=1}^m x_n$）收敛。 | L10 |
| Remark 115 | 起点无关 | 级数不必从 $n=1$ 起。 | L10 |
| Example 116 | 望远镜级数 | $\sum_{n=1}^\infty\frac{1}{n(n+1)}$ 收敛（$=1$，裂项 $\frac1n-\frac1{n+1}$）。 | L10 |
| Theorem 117 | 几何级数 | 若 $|r|<1$ 则 $\sum_{n=0}^\infty r^n$ 收敛且 $\displaystyle\sum_{n=0}^\infty r^n=\frac{1}{1-r}$。 | L10 |
| Remark 118 | 几何级数定义 | 形如 $\sum\alpha r^n$（$\alpha,r\in\mathbb{R}$）者称几何级数。 | L10 |
| Theorem 119 | 有限个起点无关 | $\sum_{n=1}^\infty x_n$ 收敛 $\iff$ $\sum_{n=M}^\infty x_n$ 收敛。 | L10 |
| Definition 120 | 级数 Cauchy | $\sum x_n$ 是 Cauchy，若其部分和序列是 Cauchy。 | L10 |
| Theorem 121 | Cauchy $\iff$ 收敛 | $\sum x_n$ Cauchy $\iff$ $\sum x_n$ 收敛。 | L10 |
| Theorem 122 | 级数 Cauchy 判据 | $\sum x_n$ Cauchy $\iff$ $\forall\epsilon>0$，$\exists M\in\mathbb{N}$，$\forall m\ge M$，$\forall\ell>m$，$\bigl|\sum_{n=m+1}^{\ell}x_n\bigr|<\epsilon$。 | L10 |
| Theorem 123 | 必要条件 | 若 $\sum x_n$ 收敛，则 $\lim_{n\to\infty}x_n=0$。 | L10 |
| Theorem 124 | 几何级数发散 | 若 $|r|\ge1$，则 $\sum_{n=0}^\infty r^n$ 发散。 | L10 |
| Corollary 125 | 几何级数判据 | $\sum_{n=0}^\infty\alpha r^n$ 收敛 $\iff$ $|r|<1$。 | L10 |

$$\text{Thm 122 是级数一切判别法的母版：}\ \left|\sum_{n=m+1}^{\ell}x_n\right|<\epsilon\ \Longleftrightarrow\ \text{"尾段任意短"}.$$

#### 4.2 正项级数、绝对收敛、比较、$p$-级数

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Recall 126 | 复习 | 若 $\sum x_n$ 收敛则 $\lim x_n=0$。 | L11 |
| Question 127 | 逆定理成立吗？ | $\lim x_n=0\Rightarrow\sum x_n$ 收敛？**不成立**（调和级数，Thm 128）。 | L11 |
| Theorem 128 | 调和级数发散 | $\sum_{n=1}^\infty\frac1n$ 发散。 | L11 |
| Remark 129 | 名称 | $\sum\frac1n$ 称**调和级数**。 | L11 |
| Theorem 130 | 线性 | $\sum x_n,\sum y_n$ 收敛 $\Rightarrow$ $\sum(\alpha x_n+y_n)$ 收敛且 $\sum(\alpha x_n+y_n)=\alpha\sum x_n+\sum y_n$。 | L11 |
| Theorem 131 | 正项级数判据 | 若 $\forall n$，$x_n\ge0$，则 $\sum x_n$ 收敛 $\iff$ 部分和 $\{s_m\}$ 有界。 | L11 |
| Definition 132 | 绝对收敛 | $\sum x_n$ 绝对收敛，若 $\sum|x_n|$ 收敛。 | L11 |
| Theorem 133 | 绝对收敛 ⇒ 收敛 | 若 $\sum x_n$ 绝对收敛，则 $\sum x_n$ 收敛。 | L11 |
| Remark 134 | 反例预告 | $\sum_{n=1}^\infty\frac{(-1)^n}{n}$ 收敛但非绝对收敛。 | L11 |
| Theorem 135 | 比较判别法 | 若 $\forall n$，$0\le x_n\le y_n$，则 (1) $\sum y_n$ 收敛 $\Rightarrow$ $\sum x_n$ 收敛；(2) $\sum x_n$ 发散 $\Rightarrow$ $\sum y_n$ 发散。 | L11 |
| Remark 136 | 万能源头 | 几何级数 + 比较判别法推出一切（比值/根值判别法都源于此）。 | L11 |
| Theorem 137 | $p$-级数 | 对 $p\in\mathbb{R}$，$\sum_{n=1}^\infty\frac{1}{n^p}$ 收敛 $\iff$ $p>1$。 | L11 |

$$\text{调和级数发散（Thm 128 的机制，分组法）：}\ \sum_{n=1}^{2^k}\frac1n\ \ge\ 1+\frac{k}{2}\ \longrightarrow\ \infty.$$

#### 4.3 比值、根值、交错、重排

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Theorem 138 | 比值判别法 | 设 $x_n\ne0$ $\forall n$，$\displaystyle L=\lim_{n\to\infty}\left|\frac{x_{n+1}}{x_n}\right|$。(1) $L<1$ $\Rightarrow$ $\sum x_n$ 绝对收敛；(2) $L>1$ $\Rightarrow$ $\sum x_n$ 发散；(3) $L=1$ 时判别法失效。 | L12 |
| Example 139 | 交错比值例 | 用比值法证 $\sum_{n=1}^\infty\frac{(-1)^n}{\text{(含阶乘的项)}}$ 收敛（见 Lecture 12）。 | L12 |
| Example 140 | 指数级数 | $\forall x\in\mathbb{R}$，$\sum_{n=0}^\infty\frac{x^n}{n!}$ 收敛（比值 $L=0<1$）。 | L12 |
| Remark 141 | 使用提示 | 出现 $(-1)^n$ 或阶乘时比值法最好用；$L=1$ 则判别法不适用。 | L12 |
| Theorem 142 | 根值判别法 | 设 $\displaystyle L=\lim_{n\to\infty}|x_n|^{1/n}$。(1) $L<1$ $\Rightarrow$ $\sum x_n$ 绝对收敛；(2) $L>1$ $\Rightarrow$ $\sum x_n$ 发散；(3) $L=1$ 时判别法失效。 | L12 |
| Remark 143 | 提示 | 同样，$L=1$ 时判别法不适用。 | L12 |
| Theorem 144 | 交错级数判别法 | 若 $\{x_n\}$ 单调递减且 $x_n\to0$，则 $\sum(-1)^nx_n$ 收敛。 | L12 |
| Corollary 145 | 条件收敛例 | $\sum\frac{(-1)^n}{n}$ 收敛但**不**绝对收敛。 | L12 |
| Theorem 146 | 重排定理 | 若 $\sum x_n$ 绝对收敛且 $\sum x_n=x$，$\sigma:\mathbb{N}\to\mathbb{N}$ 为双射，则 $\sum x_{\sigma(n)}$ 绝对收敛且 $\sum x_{\sigma(n)}=x$。即**绝对收敛级数可以任意重排。** | L12 |

#### 4.4 级数判别法决策树（**本节重点**）

```
   ┌─────────────────────────────────────────────────────────────────┐
   │ 给定 Σ xₙ，问：收敛还是发散？                                     │
   └───────────────────────────────┬─────────────────────────────────┘
                                   ▼
              ┌────────────────────────────────────────┐
              │ 第 0 步：检验必要条件                    │
              │ xₙ → 0 ？                              │
              └───────┬────────────────────┬───────────┘
                      │ 否（或 limsup|xₙ|>0）│ 是
                      ▼                     ▼
              ┌──────────────┐   ┌──────────────────────────────┐
              │ 发散         │   │ 第 1 步：含 (-1)ⁿ 或符号交替？ │
              │ (Thm 123)    │   └──────┬──────────────┬─────────┘
              └──────────────┘          │ 是           │ 否
                                        ▼              ▼
                    ┌───────────────────────┐  ┌──────────────────┐
                    │ 交错级数判别法         │  │ Σ|xₙ| 是否收敛？  │
                    │ xₙ 单调递减且 →0       │  │ （先试绝对值）     │
                    │ ⇒ 收敛 (Thm 144)      │  └────┬─────────┬───┘
                    └───────────┬───────────┘       │ 收敛    │ 不确定/发散
                                ▼                   ▼         ▼
                    ┌───────────────────────┐ ┌───────────┐ ┌──────────────┐
                    │ 再查 Σ|xₙ|：(不)绝对？ │ │ 绝对收敛   │ │ 正项可用：    │
                    └───────────────────────┘ │ ⇒ 收敛     │ │ 比较/比值/根值│
                                              │ (Thm 133)  │ └──────┬───────┘
                                              └───────────┘        │
                                                                   ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │ 正项级数 Σxₙ（xₙ≥0）的下一层选择                                  │
        ├──────────────────────────────────────────────────────────────────┤
        │                                                                  │
        │  含阶乘 / 含 (-1)ⁿ / 含 n 的幂之比 ────►  比值判别法  Thm 138      │
        │        L = lim |xₙ₊₁/xₙ|  <1 收敛, >1 发散, =1 失效              │
        │                                                                  │
        │  含 n 次幂 / 整体是 (… )ⁿ 形式 ────────►  根值判别法  Thm 142      │
        │        L = lim |xₙ|^(1/n) <1 收敛, >1 发散, =1 失效              │
        │                                                                  │
        │  能与 1/n^p 或 rⁿ 逐项比较 ────────────►  比较判别法  Thm 135      │
        │        （几何级数 Thm 117 + p-级数 Thm 137 是两大标尺）           │
        │                                                                  │
        │  以上全部失效（L=1）时的兜底：                                    │
        │        Σ 有界部分和? Thm 131   ｜  Cauchy 判据 Thm 122           │
        │        ｜ 望远镜求和 Example 116 ｜ 分组法 Thm 128               │
        └──────────────────────────────────────────────────────────────────┘

   两个标尺（一切比较的基准）：
       几何级数  Σ rⁿ  收敛 ⇔ |r|<1          (Thm 117 / Cor 125)
       p-级数    Σ 1/nᵖ 收敛 ⇔ p>1           (Thm 137)
       注意 p>1 ⇔ 可用分组比较：nᵖ 的增长快于 log，慢于一切指数
```

---

### 五、函数极限与连续（Lecture 13–16）

#### 5.1 聚点、函数极限、序列刻画

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Remark 147 | 直观 | 连续函数是「输出的可容忍变化伴随输入的充分小差异」。 | L13 |
| Definition 148 | 聚点 | $x\in\mathbb{R}$ 是 $S$ 的聚点，若 $\forall\delta>0$，$(x-\delta,x+\delta)\cap(S\setminus\{x\})\ne\varnothing$。 | L13 |
| Theorem 149 | 聚点的序列刻画 | $x$ 是 $S$ 的聚点 $\iff$ 存在 $\{x_n\}\subset S\setminus\{x\}$ 使 $x_n\to x$。 | L13 |
| Definition 150 | 函数极限 | $c$ 是 $S$ 的聚点，$f:S\to\mathbb{R}$。$f(x)\to L$ 于 $c$：$\forall\epsilon>0$，$\exists\delta>0$，若 $x\in S$ 且 $0<|x-c|<\delta$ 则 $|f(x)-L|<\epsilon$。 | L13 |
| Notation 151 | 记号 | 记 $\lim_{x\to c}f(x)=L$。 | L13 |
| Theorem 152 | 极限唯一 | 若 $f(x)\to L_1$ 与 $f(x)\to L_2$（$x\to c$），则 $L_1=L_2$。 | L13 |
| Example 153 | 线性函数 | $f(x)=ax+b$，则 $\forall c$，$\lim_{x\to c}f(x)=ac+b$（取 $\delta=\frac{\epsilon}{1+|a|}$）。 | L13 |
| Example 154 | 平方根 | $f(x)=\sqrt x$，则 $\forall c>0$，$\lim_{x\to c}\sqrt x=\sqrt c$（取 $\delta=\epsilon\sqrt c$）。 | L13 |
| Example 155 | 分段函数 | 见 Lecture 13（分段函数的极限）。 | L13 |
| Theorem 157 | 序列刻画 | $\lim_{x\to c}f(x)=L$ $\iff$ 对每个 $S\setminus\{c\}$ 中的序列 $x_n\to c$，都有 $f(x_n)\to L$。 | L13 |
| Theorem 158 | $x^2$ | $\forall c\in\mathbb{R}$，$\lim_{x\to c}x^2=c^2$。 | L14 |
| Theorem 159 | 振荡例 | (1) $\lim_{x\to0}\sin(1/x)$ 不存在；(2) $\lim_{x\to0}x\sin(1/x)=0$。 | L14 |
| Theorem 160 | 保序（函数版） | 若 $\forall x\in S$，$f(x)\le g(x)$ 且两极限存在，则 $\lim_{x\to c}f(x)\le\lim_{x\to c}g(x)$。 | L14 |

$$\forall\epsilon>0\ \exists\delta>0\ \forall x\in S,\ 0<|x-c|<\delta\Rightarrow|f(x)-L|<\epsilon.$$

**函数极限四则运算**（官方笔记中与序列版 Thm 89 平行，见 Lecture 14）：$\lim(f+g)=\lim f+\lim g$，$\lim(fg)=\lim f\cdot\lim g$，$x\to c$ 时若 $\lim g\ne0$ 则 $\lim(f/g)=\lim f/\lim g$。

#### 5.2 单侧极限与连续性

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Definition 161 | 左极限 | $c$ 是 $S\cap(-\infty,c)$ 的聚点，$f(x)\to L$（$x\to c^-$）：$\forall\epsilon>0$，$\exists\delta>0$，若 $x\in S$ 且 $c-\delta<x<c$ 则 $|f(x)-L|<\epsilon$。 | L14 |
| Definition 163 | 右极限 | $c$ 是 $S\cap(c,\infty)$ 的聚点，$f(x)\to L$（$x\to c^+$）：$\forall\epsilon>0$，$\exists\delta>0$，若 $x\in S$ 且 $c<x<c+\delta$ 则 $|f(x)-L|<\epsilon$。 | L14 |
| Example 165 | 单侧例 | 见 Lecture 14（分段函数在跳跃点处单侧极限）。 | L14 |
| Theorem 166 | 双侧 = 两侧 | 若 $c$ 同时是 $S\cap(-\infty,c)$ 与 $S\cap(c,\infty)$ 的聚点，则 $c$ 是 $S$ 的聚点，且 $\lim_{x\to c}f(x)=L$ $\iff$ $\lim_{x\to c^-}f(x)=\lim_{x\to c^+}f(x)=L$。 | L14 |
| Definition 167 | 连续 | $f$ 在 $c\in S$ 连续：$\forall\epsilon>0$，$\exists\delta>0$，若 $x\in S$ 且 $|x-c|<\delta$ 则 $|f(x)-f(c)|<\epsilon$。$f$ 在 $U\subset S$ 上连续：在每点连续。 | L14 |
| Example 168 | 线性函数连续 | $f(x)=ax+b$ 在 $\mathbb{R}$ 上连续（取 $\delta=\frac{\epsilon}{1+|a|}$）。 | L14 |
| Example 169 | 分段连续例 | 见 Lecture 14。 | L14 |
| Negation 170 | 不连续 | $\exists\epsilon_0$，$\forall\delta>0$，$\exists x\in S$ 使 $|x-c|<\delta$ 且 $|f(x)-f(c)|\ge\epsilon_0$。 | L14 |
| Theorem 171 | 连续的三等价 | (1) 若 $c$ 不是 $S$ 的聚点，则 $f$ 在 $c$ 连续；(2) 若 $c$ 是 $S$ 的聚点，则 $f$ 在 $c$ 连续 $\iff$ $\lim_{x\to c}f(x)=f(c)$。 | L15 |
| Theorem 172 | $\sin,\cos$ | $f(x)=\sin x$、$g(x)=\cos x$ 在 $\mathbb{R}$ 上连续。 | L15 |
| Theorem 173 | 多项式 | $f(x)=a_dx^d+\cdots+a_1x+a_0$ 在 $\mathbb{R}$ 上连续。 | L15 |
| Theorem 174 | 连续函数的代数 | $f,g$ 在 $c$ 连续 $\Rightarrow$ (1) $f+g$ 连续；(2) $f\cdot g$ 连续；(3) 若 $g(c)\ne0$ 则 $f/g$ 连续。 | L15 |
| Theorem 175 | 复合 | $g:A\to B$ 在 $c$ 连续，$f:B\to\mathbb{R}$ 在 $g(c)$ 连续 $\Rightarrow$ $f\circ g$ 在 $c$ 连续。 | L15 |
| Example 176 | 免 $\epsilon$-$\delta$ | $1/x^2$ 在 $(0,\infty)$ 连续（由 $x^2$ 连续 + Thm 174/175 立即得到）。 | L15 |
| Question 177 | 处处不连续？ | 是否存在 $\mathbb{R}\to\mathbb{R}$ 在每点都不连续的函数？ | L15 |
| Theorem 178 | Dirichlet 函数 | 见 Lecture 15（$f(x)=1$ 于 $\mathbb{Q}$、$=0$ 于无理数，处处不连续）。 | L15 |

**连续的三种等价刻画（Thm 171 + Thm 157）**：

```
   f 在 c 连续
        │
        ├── (ε-δ)  ∀ε>0 ∃δ>0 ∀x∈S, |x-c|<δ ⇒ |f(x)-f(c)|<ε
        │
        ├── (极限)  lim_{x→c} f(x) = f(c)      （c 为聚点时）
        │
        └── (序列)  ∀{xₙ}⊂S, xₙ→c ⇒ f(xₙ)→f(c)
                     ← 这个刻画在证明「不连续」时最好用：
                       只需找出一个序列 xₙ→c 但 f(xₙ)↛f(c)
```

#### 5.3 极值定理、介值定理、二分法

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Definition 179 | 有界函数 | $f:S\to\mathbb{R}$ 有界，若 $\exists B\ge0$，$\forall x\in S$，$|f(x)|\le B$。 | L16 |
| Theorem 180 | 闭区间连续 ⇒ 有界 | 若 $f:[a,b]\to\mathbb{R}$ 连续，则 $f$ 有界。 | L16 |
| Definition 181 | 绝对最小/最大 | $f$ 在 $c$ 取得绝对最小：$\forall x\in S$，$f(x)\ge f(c)$；在 $d$ 取得绝对最大：$\forall x$，$f(x)\le f(d)$。 | L16 |
| Theorem 182 | 极值定理 (Min-Max) | 若 $f:[a,b]\to\mathbb{R}$ 连续，则 $f$ 取得绝对最大值与绝对最小值。 | L16 |
| Remark 183 | EVT | 亦名 Extreme Value Theorem；为与 Lebl 教材一致，本课称 Min-Max 定理。 | L16 |
| Remark 184 | 假设的必要性 | 去掉连续性则 Min-Max 定理**不成立**（见 Lecture 16 反例）。 | L16 |
| Question 185 | 取遍中间值吗？ | 由 Bolzano IVT，$f$ 取遍 $[f(c),f(d)]$ 中一切值。 | L16 |
| Theorem 186 | 零点定理（二分法） | 若 $f:[a,b]\to\mathbb{R}$ 连续，$f(a)<0$，$f(b)>0$，则 $\exists c\in(a,b)$ 使 $f(c)=0$。 | L16 |
| Theorem 187 | Bolzano 介值定理 | $f:[a,b]\to\mathbb{R}$ 连续。若 $f(a)<f(b)$ 且 $y\in(f(a),f(b))$，则 $\exists c\in(a,b)$ 使 $f(c)=y$；$f(b)<f(a)$ 时类似。 | L16 |
| Remark 188 | IVT | 亦名 Intermediate Value Theorem。 | L16 |
| Theorem 189 | 连续像 = 闭区间 | $f:[a,b]\to\mathbb{R}$ 连续，$c,d$ 分别为最小、最大点，则 $f([a,b])=[f(c),f(d)]$。 | L16 |
| Theorem 190 | 具体应用 | $f(x)=x^{2021}+x^{2020}+9.03x+1$ 至少有一个实根（$f(0)=1>0$，$f(-1)=-8.03<0$）。 | L16 |

**二分法的收敛机制（Thm 186 证明骨架）**：

```
   设 a₁=a, b₁=b。归纳地：若 f((aₙ+bₙ)/2) < 0 则 (aₙ₊₁,bₙ₊₁)=((aₙ+bₙ)/2, bₙ)
                        若 f((aₙ+bₙ)/2) > 0 则 (aₙ₊₁,bₙ₊₁)=(aₙ, (aₙ+bₙ)/2)

   ⇒ {aₙ} 单调递增有上界 ⇒ 收敛（Thm 75）
   ⇒ {bₙ} 单调递减有下界 ⇒ 收敛（Thm 76）
   ⇒ bₙ - aₙ = (b-a)/2^{n-1} → 0  ⇒ 两极限相等，记为 c
   ⇒ 连续性 ⇒ f(c) = lim f(aₙ) ≤ 0 且 f(c) = lim f(bₙ) ≥ 0  ⇒ f(c)=0

   关键：二分法把「存在性」化归为「单调有界 ⇒ 收敛」，
        而后者又化归为 LUB 性质。
```

**利用 $[a,b]$ 紧性（闭 + 有界）的统一套路**：

- Thm 180：假设无界 $\Rightarrow$ 取 $\{x_n\}$ 使 $|f(x_n)|>n$ $\Rightarrow$ BW 取收敛子列 $x_{n_k}\to x\in[a,b]$ $\Rightarrow$ 连续性给 $f(x_{n_k})\to f(x)$ 有界，矛盾。
- Thm 182：由 Thm 180 得 $M=\sup f([a,b])$ 有限，取 $\{x_n\}$ 使 $f(x_n)\to M$，BW 取收敛子列，连续性给 $M=f(x)\in f([a,b])$。

---

### 六、微分（Lecture 17–20）

#### 6.1 一致连续、Lipschitz

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Recall 191 | 连续（显式依赖） | $f:S\to\mathbb{R}$ 在 $S$ 上连续：$\forall c\in S$，$\forall\epsilon>0$，$\exists\delta=\delta(\epsilon,c)>0$，$\forall x\in S$，$|x-c|<\delta\Rightarrow|f(x)-f(c)|<\epsilon$。 | L17 |
| Example 192 | $1/x$ 在 $(0,1)$ | $f(x)=1/x$ 在 $(0,1)$ 上连续（取 $\delta=\min\{\dots\}$）。 | L17 |
| Definition 193 | 一致连续 | $f:S\to\mathbb{R}$ 一致连续：$\forall\epsilon>0$，$\exists\delta=\delta(\epsilon)>0$，$\forall x,c\in S$，$|x-c|<\delta\Rightarrow|f(x)-f(c)|<\epsilon$。 | L17 |
| Remark 194 | 关键差别 | 一致连续中 $\delta$ **只依赖 $\epsilon$**，不依赖 $c$。 | L17 |
| Example 195 | $x^2$ 在 $[0,1]$ | $f(x)=x^2$ 在 $[0,1]$ 上一致连续（取 $\delta=\epsilon/2$）。 | L17 |
| Negation 196 | 非一致连续 | $\exists\epsilon_0>0$，$\forall\delta>0$，$\exists x,c\in S$ 使 $|x-c|<\delta$ 且 $|f(x)-f(c)|\ge\epsilon_0$。 | L17 |
| Theorem 197 | 紧区间上等价 | $f:[a,b]\to\mathbb{R}$：$f$ 连续 $\iff$ $f$ 一致连续。 | L17 |
| Definition 198 | 可导 | $I$ 为区间，$f:I\to\mathbb{R}$，$c\in I$。$f$ 在 $c$ 可导：$\displaystyle\lim_{x\to c}\frac{f(x)-f(c)}{x-c}$ 存在，记为 $f'(c)$。 | L17 |
| Example 200 | 线性 | $f(x)=ax+b$ $\Rightarrow$ $\forall c$，$f'(c)=a$。 | L17 |
| Example 201 | 幂法则 | $\forall n\in\mathbb{N}$，$f(x)=\alpha x^n$ $\Rightarrow$ $\forall c$，$f'(c)=\alpha nc^{n-1}$。 | L17 |

$$\text{连续：}\ \forall\epsilon>0\ \forall c\in S\ \exists\delta(\epsilon,c)>0\ \dots \qquad\text{一致连续：}\ \forall\epsilon>0\ \exists\delta(\epsilon)>0\ \forall c\in S\ \dots$$

```
   连续 vs 一致连续：量词顺序（见 Def 167 / Def 193）

      连续：      ∀ε ∀c ∃δ ∀x   ( |x-c|<δ ⇒ |f(x)-f(c)|<ε )
                         ↑
                     δ 在 c 之后 ⇒ δ 可依赖 c

      一致连续：  ∀ε ∃δ ∀c ∀x   ( |x-c|<δ ⇒ |f(x)-f(c)|<ε )
                      ↑
                  δ 在 c 之前 ⇒ δ 只依赖 ε

   ⇒ 一致连续 ⟹ 连续（Thm 265 的同类结构）
   ⇒ 连续 ⇏ 一致连续（反例：1/x 在 (0,1)，x² 在 ℝ）
   ⇒ 但在紧区间 [a,b] 上二者等价（Thm 197）
```

#### 6.2 求导法则全表、Rolle、MVT

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Theorem 202 | 可导 ⇒ 连续 | 若 $f:I\to\mathbb{R}$ 在 $c\in I$ 可导，则 $f$ 在 $c$ 连续。 | L18 |
| Question 203 | 逆成立吗？ | 连续 $\Rightarrow$ 可导？**不成立**。 | L18 |
| Example 204 | $|x|$ | $f(x)=|x|$ 在 $0$ 不可导。 | L18 |
| Question 205 | 至少一点可导？ | 连续函数 $\mathbb{R}\to\mathbb{R}$ 必有某点可导？**不成立**（Weierstrass）。 | L18 |
| Remark 206 | 编号说明 | 后续定理编号延后以便引用。 | L18 |
| Theorem 207 | Theorem I | (1) $\forall x,y\in\mathbb{R}$，$|\cos x-\cos y|\le|x-y|$；(2) $\forall c\in\mathbb{R}$，$\forall K\in\mathbb{N}$，$\exists y\in(c+\pi/K,c+3\pi/K)$ 使某一差分下界成立（完整陈述见 Lecture 18）。 | L18 |
| Theorem 208 | Theorem II | $\forall a,b,c\in\mathbb{R}$，$|a+b+c|\ge|a|-|b|-|c|$。 | L18 |
| Theorem 209 | Theorem III | 级数型恒等式与控制估计（见 Lecture 18 完整陈述）。 | L18 |
| Theorem 210 | Weierstrass 反例 | $f(x)=\sum_{k=0}^\infty\frac{\cos(160^kx)}{\text{(振幅因子见讲义)}}$ 连续但处处不可导。 | L18 |
| Remark 211 | 意义 | 存在连续但处处不可微的函数。 | L18 |
| Theorem 212 | 线性 + 乘积法则 | (1) $\forall\alpha\in\mathbb{R}$，$(\alpha f+g)'(c)=\alpha f'(c)+g'(c)$；(2) $(fg)'(c)=f'(c)g(c)+f(c)g'(c)$。 | L19 |
| Theorem 213 | 链式法则 | $g:I_1\to I_2$ 在 $c$ 可导，$f:I_2\to\mathbb{R}$ 在 $g(c)$ 可导，则 $(f\circ g)'(c)=f'(g(c))g'(c)$。 | L19 |
| Definition 214 | 相对极值 | $f$ 在 $c$ 有相对最大：$\exists\delta>0$，$\forall x\in S$，$|x-c|<\delta\Rightarrow f(x)\le f(c)$；相对最小类似。 | L19 |
| Theorem 215 | Fermat 驻点定理 | 若 $f:[a,b]\to\mathbb{R}$ 在 $c\in(a,b)$ 有相对极值且 $f$ 在 $c$ 可导，则 $f'(c)=0$。 | L19 |
| Theorem 216 | Rolle 定理 | $f:[a,b]\to\mathbb{R}$ 连续、在 $(a,b)$ 可导，且 $f(a)=f(b)$，则 $\exists c\in(a,b)$ 使 $f'(c)=0$。 | L19 |
| Remark 217 | 假设必要性 | 三个假设是否都必要，留给读者。 | L19 |
| Theorem 218 | 中值定理 (MVT) | $f:[a,b]\to\mathbb{R}$ 连续、在 $(a,b)$ 可导，则 $\exists c\in(a,b)$ 使 $f(b)-f(a)=f'(c)(b-a)$。 | L19 |
| Remark 219 | 缩写 | MVT = Mean Value Theorem。 | L19 |
| Theorem 220 | 导数为零 ⇒ 常数 | 若 $f:I\to\mathbb{R}$ 可导且 $\forall x\in I$，$f'(x)=0$，则 $f$ 为常数。 | L19 |
| Theorem 221 | 单调性判别 | $f:I\to\mathbb{R}$ 可导。则 (1) $f$ 递增 $\iff$ $\forall x$，$f'(x)\ge0$；(2) $f$ 递减 $\iff$ $\forall x$，$f'(x)\le0$。 | L19 |

**求导法则全表**：

| 法则 | 公式 | 编号 |
|:--|:--|:--|
| 线性 | $(\alpha f+g)'(c)=\alpha f'(c)+g'(c)$ | Thm 212(1) |
| 乘积 | $(fg)'(c)=f'(c)g(c)+f(c)g'(c)$ | Thm 212(2) |
| 商 | $\left(\dfrac fg\right)'(c)=\dfrac{f'(c)g(c)-f(c)g'(c)}{g(c)^2}$（$g(c)\ne0$） | 见 Lecture 19 |
| 链式 | $(f\circ g)'(c)=f'(g(c))\,g'(c)$ | Thm 213 |
| 幂法则 | $(\alpha x^n)'=\alpha nx^{n-1}$ | Example 201 |
| 反函数 | $(f^{-1})'(y_0)=\dfrac{1}{f'(x_0)}$，$y_0=f(x_0)$ | 见 Lecture 19 |

$$\text{三个中值定理的关系：}\quad \text{Fermat (Thm 215)}\ \Longrightarrow\ \text{Rolle (Thm 216)}\ \Longrightarrow\ \text{MVT (Thm 218)}.$$

$$f(b)-f(a)=f'(c)(b-a),\qquad \text{Cauchy MVT（见 Lecture 19）:}\ \frac{f(b)-f(a)}{g(b)-g(a)}=\frac{f'(c)}{g'(c)}.$$

#### 6.3 Taylor 定理与二阶导判别

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Remark 222 | 定位 | Taylor 定理本质上是高阶导数的中值定理。 | L20 |
| Definition 223 | $n$ 次可导 | $f:I\to\mathbb{R}$ 在 $J\subset I$ 上 $n$ 次可导，若 $f',f'',\dots,f^{(n)}$ 在 $J$ 上每点存在。 | L20 |
| Notation 224 | 记号 | 第 $n$ 阶导数记 $f^{(n)}$。 | L20 |
| Theorem 225 | Taylor 定理 | $f:[a,b]\to\mathbb{R}$ 连续、在 $[a,b]$ 上有 $n$ 个连续导数、$f^{(n+1)}$ 在 $(a,b)$ 存在。给定 $x_0,x\in[a,b]$，则 $\exists c\in(x_0,x)$ 使 $\displaystyle f(x)=\sum_{k=0}^{n}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k+\frac{f^{(n+1)}(c)}{(n+1)!}(x-x_0)^{n+1}$。 | L20 |
| Definition 226 | Taylor 多项式与余项 | $P_n(x)=\sum_{k=0}^n\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k$ 称 $n$ 阶 Taylor 多项式；$R_n(x)$ 称 $n$ 阶余项。 | L20 |
| Theorem 227 | 二阶导判别 | $f:(a,b)\to\mathbb{R}$ 有二阶连续导数，$x_0\in(a,b)$，$f'(x_0)=0$，$f''(x_0)>0$，则 $f$ 在 $x_0$ 有严格相对最小。 | L20 |
| Remark 228 | 定位 | Riemann 积分是第一个与经验一致（矩形/三角形/圆面积）的严格「面积」理论，是微分的逆运算，但**不是**完整的面积理论（见 Lebesgue 积分）。 | L20 |

$$f(x)=\underbrace{\sum_{k=0}^{n}\frac{f^{(k)}(x_0)}{k!}(x-x_0)^k}_{P_n(x)}+\underbrace{\frac{f^{(n+1)}(c)}{(n+1)!}(x-x_0)^{n+1}}_{R_n(x)},\qquad c\in(x_0,x).$$

---

### 七、黎曼积分（Lecture 20–22）

#### 7.1 分割、标签、黎曼和、可积性

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Definition 229 | 连续函数类 | $C([a,b]):=\{f:[a,b]\to\mathbb{R}\mid f\ \text{连续}\}$。 | L20 |
| Definition 230 | 分割与网格 | 分割 $x=\{a=x_0<x_1<\cdots<x_n=b\}$；网格 $\|x\|=\max_{j}(x_j-x_{j-1})$。 | L20 |
| Definition 231 | 标签 | 标签 $\xi=\{\xi_1,\dots,\xi_n\}$ 满足 $a=x_0\le\xi_1\le x_1\le\cdots\le x_{n-1}\le\xi_n\le x_n=b$；$(x,\xi)$ 称带标签分割。 | L20 |
| Example 232 | 计算网格 | $(x,\xi)=(\{1,3/2,2,3\},\{5/4,7/4,5/2\})$，$\|x\|=\max\{1/2,1/2,1\}=1$。 | L20 |
| Definition 233 | 黎曼和 | $\displaystyle S_f(x,\xi):=\sum_{j=1}^{n}f(\xi_j)(x_j-x_{j-1})$。 | L20 |
| Theorem 235 | 黎曼积分存在唯一 | 若 $f\in C([a,b])$，则存在唯一数 $\int_a^b f(x)\,dx\in\mathbb{R}$，使对任意 $\epsilon>0$，$\exists\delta>0$，任何 $\|x\|<\delta$ 的带标签分割都满足 $|S_f(x,\xi)-\int_a^b f|<\epsilon$。 | L21 |
| Remark 236 | 唯一性来源 | 唯一性来自实数列极限的唯一性（Thm 65）；只需证存在性。 | L21 |
| Definition 237 | 连续模 | $f\in C([a,b])$，$\eta>0$：$\displaystyle w_f(\eta)=\sup\{|f(x)-f(y)|\mid |x-y|\le\eta\}$。 | L21 |
| Theorem 239 | Theorem I | $\forall f\in C([a,b])$，$\lim_{\eta\to0}w_f(\eta)=0$（即 $\forall\epsilon>0$，$\exists\delta>0$，$\forall\eta<\delta$，$w_f(\eta)<\epsilon$）。 | L21 |
| Theorem 240 | Theorem II | 若 $(x,\xi)$、$(x',\xi')$ 是 $[a,b]$ 的带标签分割且 $x\subset x'$，则 $|S_f(x,\xi)-S_f(x',\xi')|\le w_f(\|x\|)(b-a)$。 | L21 |
| Definition 241 | 加细 | 若 $x\subset x'$，称 $x'$ 是 $x$ 的加细。 | L21 |
| Remark 242 | 加细的构造 | $x$ 的加细通过增加分割点得到。 | L21 |
| Theorem 243 | Theorem III | 任意两个带标签分割 $(x,\xi)$、$(x',\xi')$ 与 $f\in C([a,b])$：$|S_f(x,\xi)-S_f(x',\xi')|\le(w_f(\|x\|)+w_f(\|x'\|))(b-a)$。 | L21 |
| Theorem 245 | 线性 | $f,g\in C([a,b])$，$\alpha\in\mathbb{R}$ $\Rightarrow$ $\int_a^b(\alpha f+g)=\alpha\int_a^b f+\int_a^b g$。 | L21 |

$$S_f(x,\xi)=\sum_{j=1}^{n}f(\xi_j)(x_j-x_{j-1}),\qquad \left|S_f(x,\xi)-\int_a^bf\right|\le w_f(\|x\|)(b-a)\ \xrightarrow[\|x\|\to0]{}\ 0.$$

**可积性证明的统一结构（公共加细技巧）**：

```
   任意两个分割 (x,ξ), (x',ξ')
            │
            └──► 取公共加细  x'' = x ∪ x'   （Definition 241 / Remark 242）
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
   |S_f(x,ξ) - S_f(x'',ξ'')|      |S_f(x',ξ') - S_f(x'',ξ'')|
        ≤ w_f(‖x‖)(b-a)                ≤ w_f(‖x'‖)(b-a)
        (Thm 240)                      (Thm 240)
        └───────────────┬───────────────┘
                        ▼
        |S_f(x,ξ) - S_f(x',ξ')| ≤ (w_f(‖x‖)+w_f(‖x'‖))(b-a)   (Thm 243)
                        │
                        ▼  令 ‖x‖,‖x'‖ → 0，用 Thm 239（w_f(η)→0）
                        ▼
        所有黎曼和收敛到同一个数 ⇒ 积分唯一存在（Thm 235）
```

#### 7.2 积分性质、FTC、分部、换元、Riemann–Lebesgue

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Theorem 246 | 区间可加性 | $f\in C([a,b])$，$a<c<b$ $\Rightarrow$ $\int_a^b f=\int_a^c f+\int_c^b f$。 | L22 |
| Theorem 247 | 上下确界界 | $m_f=\inf\{f(x)\mid x\in[a,b]\}$，$M_f=\sup\{f(x)\mid x\in[a,b]\}$，则 $m_f(b-a)\le\int_a^bf\le M_f(b-a)$。 | L22 |
| Theorem 248 | 单调性 | $f,g\in C([a,b])$。若 $\forall x$，$f(x)\le g(x)$，则 $\int_a^b f\le\int_a^b g$。 | L22 |
| Remark 249 | 积分约定 | $\int_a^a f:=0$；与 $\lim_{b\to a}$ 一致。 | L22 |
| Theorem 250 | 微积分基本定理 (FTC) | $f\in C([a,b])$。(1) 若 $F:[a,b]\to\mathbb{R}$ 可导且 $F'=f$，则 $\int_a^b f=F(b)-F(a)$；(2) 若定义 $F(x)=\int_a^x f$，则 $F$ 可导且 $F'=f$。 | L22 |
| Remark 251 | 缩写 | FTC = Fundamental Theorem of Calculus。 | L22 |
| Theorem 252 | 分部积分 | $f,g\in C([a,b])$ 且 $f',g'\in C([a,b])$，则 $\displaystyle\int_a^b fg'=f(b)g(b)-f(a)g(a)-\int_a^b f'g$。 | L22 |
| Remark 253 | 缩写 | IBP = Integration By Parts。 | L22 |
| Lemma 254 | Riemann–Lebesgue | 设 $f\in C([-\pi,\pi])$，$f'\in C([-\pi,\pi])$，$f$ 为 $2\pi$-周期且 $f(-\pi)=f(\pi)$。对 $n\in\mathbb{N}\cup\{0\}$ 定义 $a_n=\frac1\pi\int_{-\pi}^{\pi}f(x)\cos(nx)\,dx$，$b_n=\frac1\pi\int_{-\pi}^{\pi}f(x)\sin(nx)\,dx$，则 $a_n\to0$、$b_n\to0$。 | L22 |
| Definition 255 | Fourier 系数 | Lemma 254 中的 $a_n,b_n$ 称 $f$ 的 Fourier 系数。 | L22 |
| Theorem 256 | 换元 | $\phi:[a,b]\to[c,d]$ 连续可微，$\phi'>0$ 于 $[a,b]$，$\phi(a)=c$，$\phi(b)=d$，则 $\int_c^d f(u)\,du=\int_a^b f(\phi(t))\phi'(t)\,dt$。 | L22 |

$$\int_a^b f=F(b)-F(a)\ \ (F'=f),\qquad \frac{d}{dx}\int_a^x f=f(x),\qquad \int_a^b fg'=\bigl[fg\bigr]_a^b-\int_a^b f'g,$$
$$\int_c^d f(u)\,du=\int_a^b f(\phi(t))\phi'(t)\,dt,\qquad \int_a^a f=0,\qquad \int_a^b f=-\int_b^a f.$$

---

### 八、函数列与一致收敛（Lecture 23–25）

#### 8.1 逐点 vs 一致（量词逐字对照）

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Remark 257 | 动机 | 幂级数驱动了函数序列的一般讨论。 | L23 |
| Definition 258 | 幂级数 | 关于 $x_0$ 的幂级数：$\sum_{m=0}^\infty a_m(x-x_0)^m$。 | L23 |
| Theorem 259 | 收敛半径公式 | 设 $\displaystyle R=\lim_{m\to\infty}\left|\frac{a_m}{a_{m+1}}\right|$（或 $\lim |a_m|^{-1/m}$），则级数在 $|x-x_0|<R$ 绝对收敛、在 $|x-x_0|>R$ 发散。 | L23 |
| Definition 260 | 收敛半径 | 上述定理中的 $R$ 称收敛半径。 | L23 |
| Example 261 | 几何级数展开 | $\displaystyle f(x)=\frac{1}{1-x}=\sum_{m=0}^\infty x^m$（$|x|<1$）。 | L23 |
| Question 262 | 三个问题 | (1) $f$ 连续吗？(2) $f$ 可导且 $f'=\lim f_n'$ 吗？(3) $\lim\int f_n=\int\lim f_n$ 吗？ | L23 |
| Definition 263 | 逐点收敛 | $\{f_n\}$ 逐点收敛到 $f$：$\forall x\in S$，$\lim_{n\to\infty}f_n(x)=f(x)$。 | L23 |
| Definition 264 | 一致收敛 | $\forall\epsilon>0$，$\exists M\in\mathbb{N}$，$\forall n\ge M$，$\forall x\in S$，$|f_n(x)-f(x)|<\epsilon$。 | L23 |
| Theorem 265 | 一致 ⇒ 逐点 | 若 $f_n\to f$ 一致，则 $f_n\to f$ 逐点。 | L23 |
| Theorem 266 | 反例 | $f_n(x)=x^n$ 在 $[0,1]$ 上逐点收敛于分段函数但不一致收敛（见 Lecture 24）。 | L24 |
| Negation 267 | 非一致收敛 | $\exists\epsilon_0>0$，$\forall M\in\mathbb{N}$，$\exists n\ge M$ 与 $\exists x\in S$ 使 $|f_n(x)-f(x)|\ge\epsilon_0$。 | L24 |

**逐点 vs 一致：量词逐字对照表**

| 项目 | 逐点收敛 (Def 263) | 一致收敛 (Def 264) |
|:--|:--|:--|
| 量词串 | $\forall x\in S\ \ \forall\epsilon>0\ \ \exists M\in\mathbb{N}$ | $\forall\epsilon>0\ \ \exists M\in\mathbb{N}\ \ \forall x\in S$ |
| $\epsilon$ 与 $x$ 的顺序 | $x$ 在前，$\epsilon$ 在后 | $\epsilon$ 在前，$x$ 在后 |
| $M$ 的依赖 | $M=M(\epsilon,x)$，可依赖 $x$ | $M=M(\epsilon)$，**不依赖 $x$** |
| 每个 $x$ 的收敛速度 | 各点可以任意慢 | 全区间**统一**速度 |
| 否定式 | $\exists x\ \exists\epsilon_0>0\ \forall M\ \exists n\ge M,\ |f_n(x)-f(x)|\ge\epsilon_0$ | $\exists\epsilon_0>0\ \forall M\ \exists n\ge M\ \exists x\in S,\ |f_n(x)-f(x)|\ge\epsilon_0$ |
| 几何含义 | 竖切面上逐点看 | $\sup_{x\in S}|f_n(x)-f(x)|\to0$（带状夹逼） |
| 强度 | 弱 | 强（一致 ⟹ 逐点，Thm 265；反之不成立） |

```
   一致收敛的几何图像（"ε 带状"）：

      f(x)+ε  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                                          ／
      f(x)    ────────────────────────／─────────
                                     ／
      f(x)-ε  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄／┄┄┄┄┄┄┄┄┄┄┄
                                    ↑
      存在 M(ε)：n ≥ M 之后，整条曲线 fₙ 都落在 ε 带内（对全部 x 同时成立）

   逐点收敛的图像：
      固定 x 后看数列 f₁(x), f₂(x), … → f(x)；不同 x 可以「进度不同」。
      反例 xⁿ 在 [0,1]：每个 x<1 很快就进入带内，但 x=1-1/n 附近永远拖后腿
      ⇒ sup_{x∈[0,1]}|xⁿ-0| = 1 不趋于 0 ⇒ 不一致收敛。
```

#### 8.2 Weierstrass M-test 与三个交换定理

| 编号 | 名称 | 陈述 | 讲次 |
|:--|:--|:--|:--|
| Theorem 268 | Weierstrass M-test | 设 $f_j:S\to\mathbb{R}$，存在 $M_j>0$ 使 (a) $\forall x\in S$，$|f_j(x)|\le M_j$；(b) $\sum_{j=0}^\infty M_j$ 收敛；则 $\sum_{j=0}^\infty f_j$ 在 $S$ 上一致收敛。 | L24 |
| Remark 269 | 一般情形 | 一般说来极限**不能**交换顺序。 | L24 |
| Example 270 | 反例 | 见 Lecture 24（逐点收敛但 $\lim\int\ne\int\lim$ 的例子）。 | L24 |
| Question 271 | 三个交换问题 | 连续性、可微性（$f'=\lim f_n'$）、可积性（$\lim\int=\int\lim$）能否传递？ | L24 |
| Question 272 | 逐点情形 | 若只有逐点收敛，上述三问的答案**全是否定的**。 | L24 |
| Question 273 | 重复提问 | 同 Question 271/272 的完整表述。 | L25 |
| Theorem 274 | 一致收敛保连续 | 若 $f_n:S\to\mathbb{R}$ 对每个 $n$ 连续，$f:S\to\mathbb{R}$，$f_n\to f$ 一致，则 $f$ 连续。 | L25 |
| Theorem 275 | 一致收敛保积分 | 若 $f_n:[a,b]\to\mathbb{R}$ 连续 $\forall n$，$f:[a,b]\to\mathbb{R}$，$f_n\to f$ 一致，则 $\displaystyle\lim_{n\to\infty}\int_a^b f_n=\int_a^b f$。 | L25 |
| Remark 276 | 记号 | 该结论写作 $\lim_{n\to\infty}\int_a^b f_n=\int_a^b\lim_{n\to\infty}f_n$。 | L25 |
| Theorem 277 | 一致收敛保导数 | 若 $f_n:[a,b]\to\mathbb{R}$ 连续可微，$f,g:[a,b]\to\mathbb{R}$，$f_n\to f$ 逐点，$f_n'\to g$ **一致**，则 $f$ 可导且 $f'=g$。 | L25 |
| Theorem 278 | 幂级数内闭一致 | $\sum_{j=0}^\infty a_j(x-x_0)^j$ 收敛半径 $p\in(0,\infty]$。则 $\forall r\in(0,p)$，$\sum_j a_j(x-x_0)^j$ 在 $[x_0-r,x_0+r]$ 上一致收敛（M-test，取 $M_j=|a_j|r^j$）。 | L25 |
| Theorem 279 | 逐项求导/积分 | 设幂级数有收敛半径 $p\in(0,\infty]$。则 (1) $\forall c\in(x_0-p,x_0+p)$，$\sum_j a_j(x-x_0)^j$ 在 $c$ 可导且可逐项求导；(2) 可逐项积分。 | L25 |
| Remark 280 | 半径不变 | 由 $\lim_{j\to\infty}((j+1)|a_{j+1}|)^{1/j}=\lim_{j\to\infty}|a_j|^{1/j}$，逐项求导后收敛半径不变。 | L25 |
| Remark 281 | 直觉 | 本节的定理实质上是说：「$[a,b]$ 上的每个连续函数几乎都是多项式。」 | L25 |
| Theorem 282 | Weierstrass 逼近定理 | 若 $f\in C([a,b])$，则存在多项式序列 $\{P_n\}$ 使 $P_n\to f$ 在 $[a,b]$ 上一致。 | L25 |
| Theorem 283 | 逼近多项式的显式构造 | 令 $\displaystyle c_n:=\left(\int_{-1}^{1}(1-x^2)^n\,dx\right)^{-1}>0$，再以 $(1-x^2)^n$ 的平移伸缩作卷积型多项式核（完整构造见 Lecture 25）。 | L25 |

$$\text{M-test：}\ |f_j(x)|\le M_j\ \forall x,\ \sum M_j<\infty\ \Longrightarrow\ \sum f_j\ \text{一致收敛}.$$

**三个交换定理的对照**：

| 交换内容 | 结论 | 需要的收敛条件 | 编号 |
|:--|:--|:--|:--|
| $\lim_n\lim_{x\to c}f_n(x)=\lim_{x\to c}\lim_n f_n(x)$ | $f$ 连续 | $f_n$ 连续 + $f_n\to f$ **一致** | Thm 274 |
| $\lim_n\int_a^b f_n=\int_a^b\lim_n f_n$ | 积分可交换 | $f_n$ 连续 + $f_n\to f$ **一致** | Thm 275 |
| $\frac{d}{dx}\lim_n f_n=\lim_n f_n'$ | $f'=g$ | $f_n\to f$ 逐点 + $f_n'\to g$ **一致** | Thm 277 |

**一致 Cauchy 判据（见 Lecture 23–24 与 Final Assignment）**：$\{f_n\}$ 在 $S$ 上一致收敛 $\iff$ $\forall\epsilon>0$，$\exists M$，$\forall n,m\ge M$，$\forall x\in S$，$|f_n(x)-f_m(x)|<\epsilon$。**无需预先知道极限函数 $f$**，这是它比 Definition 264 好用的地方。

---

### 九、全课程核心依赖链

#### 9.1 主依赖链

```
   ════════════════════════════════════════════════════════════════════════════════
                             MIT 18.100A 全课程依赖图
   ════════════════════════════════════════════════════════════════════════════════

    [逻辑层]  Axiom 5 良序原理
                   │
                   ▼
              Theorem 6 数学归纳法 ──► Theorem 8 有限几何和、Theorem 9 Bernoulli
                   │
                   ▼
    [集合层]  Definition 11 基数 ──► Theorem 12 CSB ──► Theorem 15 Cantor 定理
                                                        └► Theorem 57 (0,1] 不可数
                                                            └► Corollary 58 ℝ 不可数
                   │
                   ▼
    [实数层]  Theorem 27/28 ℚ 无 sup（ℚ 有洞）
                   │
                   ▼
              Definition 26 / Theorem 22 / Theorem 39
              ★ LUB 性质：ℝ 是唯一具 LUB 的有序域 ★   ◄── 全课程的发动机
                   │
       ┌───────────┼──────────────────────────┬──────────────────────────┐
       │           │                          │                          │
       ▼           ▼                          ▼                          ▼
  Thm 38 下确界  Thm 43(i) Archimedes    Thm 44/45 sup 的 ε 刻画    Thm 50/51 绝对值
   存在          │                          │                       +三角不等式
                 ▼                          │                          │
            Thm 43(ii) ℚ 稠密               │                          │
            Thm 49–51 绝对值                │                          │
                 │                          │                          │
                 └──────────────┬───────────┴──────────────────────────┘
                                ▼
    [序列层]  Def 63 ε-N 收敛 ──► Thm 65 唯一 ──► Thm 66 (∀ε 则相等)
                   │
                   ├──► Thm 72 收敛 ⇒ 有界
                   │
                   ▼
              Thm 75/76 单调有界定理 ◄── 直接消费 LUB（极限 = sup）
                   │
                   ├──────────────────────────────┐
                   ▼                              ▼
              Thm 79 子序列同极限            Theorem 82 夹逼定理
                   │                              │
                   ▼                              ▼
              Thm 87 保序、Thm 89 四则、Thm 91 根、Thm 93 绝对值、Thm 94 几何
                   │
                   ▼
              Theorem 101 子列实现 limsup/liminf
                   │
                   ▼
              Theorem 102 ★ Bolzano–Weierstrass ★
                   │
                   ├──────────────────┬──────────────────┬──────────────────┐
                   ▼                  ▼                  ▼                  ▼
        Thm 105 收敛 ⟺        Thm 110–112 Cauchy    Thm 160 函数极限     Thm 180 闭区间
        liminf=limsup          完备性               保序                 连续 ⇒ 有界
                   │                  │                  │                  │
                   │                  ▼                  │                  ▼
                   │           Definition 114 级数        │           Theorem 182 极值定理
                   │                  │                  │                  │
                   │      ┌───────────┼──────────┐       │                  ▼
                   │      ▼           ▼          ▼       │           Theorem 186/187
                   │  Thm 122     Thm 117     Thm 128    │           零点定理 / 介值定理
                   │  级数Cauchy  几何级数    调和级数   │                  │
                   │  判据       Cor 125     发散       │                  ▼
                   │      │           │          │       │           Theorem 189
                   │      ▼           ▼          ▼       │           连续像 = 闭区间
                   │  Thm 121     Thm 135 比较  Thm 137  │                  │
                   │  Cauchy⟺收敛  判别法      p-级数    │                  │
                   │      │           │          │       │                  │
                   │      ▼           ▼          ▼       │                  │
                   │  Thm 123     Thm 138/142 Thm 131   │                  │
                   │  必要条     比值/根值   正项判据   │                  │
                   │  件         判别法                │                  │
                   │      │           │                  │                  │
                   │      ▼           ▼                  │                  │
                   │  Thm 132/133  Thm 144 交错         │                  │
                   │  绝对收敛     判别法               │                  │
                   │      │           │                  │                  │
                   │      ▼           ▼                  │                  │
                   │  Thm 146 重排  Cor 145 条件收敛     │                  │
                   │                                     │                  │
                   └──────────────┬──────────────────────┘                  │
                                  ▼                                         │
    [函数层]  Def 148 聚点 ──► Thm 149 序列刻画 ──► Def 150 ε-δ 函数极限      │
                                  │                                         │
                                  ▼                                         │
                          Thm 152 唯一、Thm 157 序列刻画、Thm 160 保序       │
                                  │                                         │
                                  ▼                                         │
                          Def 161/163 单侧极限 ──► Thm 166 双侧 = 两侧       │
                                  │                                         │
                                  ▼                                         │
                          Def 167 连续、Neg 170 不连续                       │
                                  │                                         │
                                  ▼                                         │
                          Thm 171 三等价、Thm 172–175 代数与复合            │
                                  │                                         │
                    ┌─────────────┼─────────────┬────────────────────┐      │
                    ▼             ▼             ▼                    ▼      │
             Thm 197 紧区间    Def 193      Def 198 可导       Thm 180/182 ◄─┘
             连续⟺一致连续    一致连续            │            极值定理
                    │                          ▼                    │
                    │                    Thm 202 可导⇒连续          │
                    │                          │                    │
                    │                          ▼                    │
                    │                    Thm 212 线性/乘积          │
                    │                    Thm 213 链式                │
                    │                          │                    │
                    │                          ▼                    │
                    │                    Thm 215 Fermat ──► Thm 216 Rolle
                    │                          │                    │
                    │                          ▼                    │
                    │                    Thm 218 ★ MVT ★ ◄──────────┘
                    │                          │
                    │              ┌───────────┼───────────┬──────────────┐
                    │              ▼           ▼           ▼              ▼
                    │        Thm 220      Thm 221     Thm 225       Thm 250 FTC
                    │        导数为0      单调性      Taylor 定理    (用 MVT 证)
                    │        ⇒ 常数      判别              │
                    │                                 Thm 227 二阶导判别
                    │
                    ▼
    [积分层]  Def 229–233 连续类/分割/标签/黎曼和
                    │
                    ▼
              Def 237 w_f 连续模 ──► Thm 239 w_f(η)→0
                    │
                    ▼
              Thm 240/243 公共加细误差估计
                    │
                    ▼
              Theorem 235 ★ 黎曼积分存在且唯一（f ∈ C([a,b])）★
                    │
       ┌────────────┼────────────┬────────────┬─────────────┬──────────────┐
       ▼            ▼            ▼            ▼             ▼              ▼
   Thm 245     Thm 246      Thm 247     Thm 248       Thm 250 FTC    Thm 252 IBP
   线性        可加性       上下确界    单调性        (逆运算)       (用乘积法则)
       │                                                                    │
       │                                                                    ▼
       │                                                          Lemma 254 Riemann–Lebesgue
       │                                                            Def 255 Fourier 系数
       ▼
   Thm 256 换元
                    │
                    ▼
    [函数列层]  Def 258 幂级数 ──► Thm 259/Def 260 收敛半径
                    │
                    ▼
              Def 263 逐点 vs Def 264 一致 ──► Thm 265 一致⇒逐点、Thm 266 反例
                    │
                    ▼
              Theorem 268 ★ Weierstrass M-test ★
                    │
       ┌────────────┼────────────┬─────────────────────┐
       ▼            ▼            ▼                     ▼
   Thm 274      Thm 275      Thm 277               Thm 278 幂级数内闭一致
   保连续       保积分       保导数(需 f'ₙ 一致)        │
                                                        ▼
                                                Thm 279 逐项求导/积分
                                                   Remark 280 半径不变
                                                        │
                                                        ▼
                                                Theorem 282 ★ Weierstrass 逼近 ★
                                                   Theorem 283 显式多项式构造

   ════════════════════════════════════════════════════════════════════════════════
   一句话总结：良序 → 归纳 → (ℚ 有洞) → LUB → 单调有界 → BW → 极值/介值
              → MVT → FTC → M-test → Weierstrass 逼近。每一环都靠上一环。
   ════════════════════════════════════════════════════════════════════════════════
```

#### 9.2 哪些定理在 $\mathbb{Q}$ 里会失效

| 定理 / 结论 | 在 $\mathbb{Q}$ 中的情形 | 失效原因 | 编号 |
|:--|:--|:--|:--|
| 最小上界性质 (LUB) | **失效** | $E=\{q\in\mathbb{Q}\mid q>0,q^2<2\}$ 非空有上界（如 $2$）但无有理上确界 | Definition 26 / Theorem 28 |
| $\sqrt2$ 的存在性 | **失效** | 无 $q\in\mathbb{Q}$ 使 $q^2=2$；$\mathbb{R}$ 中由 LUB 凑出唯一 $r=\sqrt2$ | Theorem 40 / Theorem 27 |
| 单调有界定理 | **失效** | 单调递增有上界的**有理**数列可以不收敛（如 $1,1.4,1.41,1.414,\dots$ 逼近 $\sqrt2$）；定理 75 的证明用 $\sup$ 作极限 | Theorem 75 / Theorem 76 |
| Bolzano–Weierstrass | **失效** | 用上述逼近 $\sqrt2$ 的有理数列，任何子列在 $\mathbb{Q}$ 中都不收敛；证明依赖 Thm 75 | Theorem 102 / Theorem 101 |
| Cauchy 完备性 | **失效** | 同上数列在 $\mathbb{Q}$ 中是 Cauchy 但无有理极限；$\mathbb{Q}$ 不完备 | Theorem 112 |
| 极值定理 (Min-Max) | **失效** | $f(x)=x^2$ 在 $\mathbb{Q}\cap[0,2]$ 上无最大值（$\sup$ 处是 $\sqrt2\notin\mathbb{Q}$）；证明依赖 Thm 180/182 的 sup | Theorem 180 / Theorem 182 |
| 介值定理的完整形式 | **失效** | $f(x)=x^2-2$ 在 $\mathbb{Q}\cap[0,2]$ 上变号但取不到 $0$；证明依赖 $\mathbb{R}$ 的完备性 | Theorem 186 / Theorem 187 |
| 黎曼积分的存在性 | **失效** | 在 $\mathbb{Q}$ 上「连续」函数未必一致连续，$w_f(\eta)\to0$ 可能不成立，Thm 235 的存在性证明断裂 | Theorem 235 / Theorem 239 |
| Archimedes 性质 | **在 $\mathbb{Q}$ 中仍成立** | $\mathbb{Q}$ 是有序域且可用整数部分构造；它的证明在 $\mathbb{Q}$ 内即可完成（但练习中通常用 LUB） | Theorem 43(i) |
| $\mathbb{Q}$ 的稠密性 | **在 $\mathbb{Q}$ 中仍成立** | 有理数之间仍有有理数（$r=(x+y)/2$） | Theorem 43(ii) |
| 绝对值六条、三角不等式 | **在 $\mathbb{Q}$ 中仍成立** | 纯代数与序性质，不需要 LUB | Theorem 50 / Theorem 51 |

$$\boxed{\ \text{分界线只有一条：}\quad \mathbb{Q}\ \text{满足有序域全部公理} + \text{Archimedes} + \text{稠密性，唯独缺少 LUB}.\ }$$

---

### 十、证明技巧总表

| # | 技巧名 | 形式 | 用在哪 | 为什么有效（一句话） |
|:--|:--|:--|:--|:--|
| 1 | $\epsilon/2$ 技巧 | 给定 $\epsilon$，分别控制两项在 $\epsilon/2$ 内，相加得 $\epsilon$ | Thm 65 极限唯一、Thm 66、乘积极限 | 两条独立误差各自可任意小，其和仍可任意小 |
| 2 | $\epsilon/3$ 技巧 | 三段误差各 $<\epsilon/3$，用三角不等式合成 $<\epsilon$ | Thm 274 保连续、Thm 277 保导数 | 三点相扣（$f(x)-f_n(x)+f_n(x)-f_n(c)+f_n(c)-f(c)$），每段一个控制源 |
| 3 | $\min\{1,\delta_0\}$ 技巧 | 取 $\delta=\min\{1,\delta_0\}$ | Example 153、Example 192、乘积极限 | $1$ 先给出**上界**（把 $|x-c|<1$ 变成 $|x|<|c|+1$），再让另一部分任意小 |
| 4 | $M=\max\{M_1,\dots,M_k\}$ | 有限多个门槛取最大 | Thm 82 夹逼、Thm 89 四则、Thm 87 保序 | $n\ge M$ 同时满足全部 $k$ 个条件，无需约定哪一个更大 |
| 5 | 取 $\epsilon=1$ 得界 | 令 $\epsilon=1$，得 $M$ 使 $n\ge M$ 时 $|x_n-x|<1$ | Thm 72 收敛⇒有界、Thm 110 Cauchy⇒有界 | 得到 $|x_n|\le|x|+1$（尾部），有限多项头部取 max 即可 |
| 6 | 加一项减一项 | $x_ny_n-xy=x_n(y_n-y)+y(x_n-x)$ | Thm 89 乘积、Thm 212 乘积法则、$x^n-c^n$ | 把交叉项拆成两个「单变量小量 × 常数」 |
| 7 | 有理化共轭 | $\sqrt a-\sqrt b=\dfrac{a-b}{\sqrt a+\sqrt b}$ | Thm 91 平方根、Example 154 | 分母有正下界时把差的估计变成常数倍的 $|a-b|$ |
| 8 | $\epsilon/(|c|+1)$ | 取 $\delta=\dfrac{\epsilon}{|a|+1}$ | Example 153 线性函数、Example 168 | 分母加 $1$ 避免 $a=0$，且保证 $|a|\delta<\epsilon$ |
| 9 | $b=\min\{|y_1|,\dots,|y_M|,|y|/2\}$ | 商极限：取分母正下界 $b>0$ | 商法则、$\lim 1/x_n=1/x$ | 有限多项的最小值与 $|y|/2$ 取 min，保证 $|y_n|\ge b>0$ |
| 10 | 反证取中点（二分法） | 每次把区间对半，保留「坏」的一半 | Thm 186 零点定理、Thm 40 | 区间长度 $\to0$，单调有界 + 连续性锁定唯一候选点 |
| 11 | 对角线论证 | 第 $n$ 位取与 $d_{nn}$ 不同的数字 | Thm 15、Thm 57（(0,1] 不可数） | 造出的对象与表中**每一个**元素在某一位上不同 |
| 12 | $n_k\ge k$（子列指标界） | 子列 $\{x_{n_k}\}$ 满足 $n_k\ge k$ | Thm 79、Thm 101、Thm 111 | 把「$n_k\ge M$」化归为「$k\ge M$」，让 $\epsilon$-$N$ 门槛可以直接对齐 |
| 13 | M-test | $|f_j(x)|\le M_j$ 且 $\sum M_j<\infty$ $\Rightarrow$ $\sum f_j$ 一致 | Thm 268、Thm 278 幂级数 | 用数项级数 $\sum M_j$ 的尾部统一控制所有 $x$，末项余项 $\sum_{j>N}M_j<\epsilon$ |
| 14 | 公共加细 | $x''=x\cup x'$ | Thm 243 黎曼和误差、Thm 275 | 两个分割同时加细到同一分割，三角不等式即可比较 |
| 15 | 望远镜求和 | $\dfrac{1}{n(n+1)}=\dfrac1n-\dfrac1{n+1}$ | Example 116、Thm 128 调和级数分组 | 相邻项抵消，部分和 $s_m=1-\frac1{m+1}$ 有显式表达式 |
| 16 | Bernoulli 不等式 | $(1+c)^n\ge1+nc$ | Thm 94 几何序列、Thm 95 $n^{1/n}\to1$ | 给「$n$ 次幂」一个**线性下界**，从而把平方根/开方估计变成 $1+$ 小量 |
| 17 | 夹逼（Stolz 式包夹） | $a_n\le x_n\le b_n$ 且 $a_n,b_n\to x$ | Thm 82、Thm 105、$\lim n^{1/n}=1$ | 上下包夹把未知序列转化为两个已知序列 |
| 18 | $\sup$ 的 $\epsilon$ 刻画 | $x=\sup S$ $\iff$ 上界 + $\forall\epsilon>0\exists y\in S:y>x-\epsilon$ | Thm 45、Thm 75、Thm 182 | 把「最小上界」翻译成可操作的「逼近」语言，便于构造序列 |
| 19 | 单调性传递逼近 | 若 $a\le x\le b$ 则 $\lim a\le\lim x\le\lim b$ | Thm 87 保序、Thm 105、Thm 160 | 极限保序（非严格），允许用「$\le$」传递双向不等式 |
| 20 | 一致 Cauchy 判据 | $\forall\epsilon\exists M\forall n,m\ge M\forall x\ |f_n(x)-f_m(x)|<\epsilon$ | Thm 268、Thm 282、一致收敛的验证 | 不需要预先知道极限函数 $f$，把「收敛」化为「内部互相靠拢」 |
| 21 | 序列刻画（否定式） | 用序列 $x_n\to c$ 但 $f(x_n)\nrightarrow f(c)$ 证不连续 | Thm 157、Thm 171、Example 204 | 「不连续/极限不存在」的证明只需**构造一个坏序列** |
| 22 | 分段估计（$\delta\le1$） | 先限制 $|x-c|<1$，再在 $[c-1,c+1]$ 上用有界性 | Thm 158 $x^2$、Example 195 | 紧区间上连续 ⇒ 有界，给出可用的常数上界 |
| 23 | Taylor 反复用 MVT | 对余项反复应用 MVT $n+1$ 次 | Thm 225 Taylor 定理 | 高阶余项 = 低阶余项差商的 MVT，逐层降阶到 $f^{(n+1)}(c)$ |
| 24 | 存在性 $\Rightarrow$ 取序列 $\Rightarrow$ 用 BW | 假设无界则取 $x_n$ 使 $|f(x_n)|>n$，BW 取子列 | Thm 180、Thm 182 极值定理 | 紧区间 $[a,b]$ 的有界序列必有收敛子列，且极限仍在 $[a,b]$ |

**技巧与证明的对应速查（哪道题用哪招）**：

```
   ┌──────────────────────────────┬──────────────────────────────────────────┐
   │ 命题                          │ 首选技巧                                 │
   ├──────────────────────────────┼──────────────────────────────────────────┤
   │ 极限存在且唯一                │ ①ε/2 + 三角不等式                        │
   │ 收敛 ⇒ 有界                   │ ⑤取 ε=1 + 有限项取 max                   │
   │ 乘积/商的极限                 │ ⑥加一项减一项 + ⑨分母正下界 + ③min{1,δ₀}│
   │ 平方根/无理式极限              │ ⑦有理化共轭 + ⑧ε/(|c|+1)                 │
   │ 单调有界定理                  │ ⑱sup 的 ε 刻画（这是 LUB 的接口）        │
   │ BW 定理                       │ ⑩反证取中点 / 区间对半 + ⑱              │
   │ limsup = liminf ⇒ 收敛        │ ⑰夹逼 + 极限保序 ⑲                       │
   │ 级数收敛（正项）               │ 比较（几何/ p-级数）+ ⑮望远镜            │
   │ 级数收敛（交替）               │ Thm 144 单调递减 → 0 + 部分和奇偶配对     │
   │ 一致收敛                      │ ⑬M-test 或 ⑳一致 Cauchy 判据             │
   │ 极限函数连续/可积              │ ②ε/3 技巧                                │
   │ 极限函数可导                  │ fₙ 逐点 + fₙ' 一致（Thm 277），再用 MVT   │
   │ 连续函数可积                  │ ⑭公共加细 + w_f(η)→0（不一致但可控）      │
   │ 零点存在                      │ ⑩二分法 + 单调有界定理                    │
   │ 极值存在                      │ ㉔取序列 + BW + 连续性                   │
   │ 不可数性                      │ ⑪对角线论证                              │
   │ 连续但处处不可导              │ ⑬M-test 造级数 + 二分震荡估计            │
   │ 连续函数被多项式逼近            │ ⑳一致 Cauchy + 卷积型多项式核             │
   └──────────────────────────────┴──────────────────────────────────────────┘
```

---

### 十一、最容易记混的 20 个对比

| # | A | B | 关键区别 |
|:--|:--|:--|:--|
| 1 | $\limsup_{n\to\infty}x_n$ | $\lim_{n\to\infty}x_n$ | limsup 对**任何有界序列**都存在（Thm 98）；lim 可能不存在。二者相等 $\iff$ 序列收敛（Thm 105） |
| 2 | $\limsup_{n\to\infty}x_n$ | $\max\{x_n\}$ | limsup 是「最终上包络的极限」，可能取不到（如 $x_n=1/n$ 时 limsup $=0$ 不是任何项）；max 要求取到 |
| 3 | 逐点收敛（Def 263） | 一致收敛（Def 264） | $M$ 是否依赖 $x$：$M(\epsilon,x)$ vs $M(\epsilon)$。一致 ⟹ 逐点（Thm 265），反之不成立（Thm 266） |
| 4 | 连续（Def 167） | 一致连续（Def 193） | $\delta$ 是否依赖 $c$：$\delta(\epsilon,c)$ vs $\delta(\epsilon)$。紧区间上等价（Thm 197） |
| 5 | 连续（Def 167） | Lipschitz | Lipschitz 是**线性模**控制：$|f(x)-f(y)|\le L|x-y|$。Lipschitz ⟹ 一致连续 ⟹ 连续；反之均不成立 |
| 6 | 有界序列（Def 61） | 收敛序列（Def 63） | 收敛 ⟹ 有界（Thm 72）；有界 ⇏ 收敛（$(-1)^n$） |
| 7 | Cauchy 序列（Def 106） | 收敛序列（Def 63） | 在 $\mathbb{R}$ 中二者等价（Thm 112）；在 $\mathbb{Q}$ 中 Cauchy 严格弱于收敛 |
| 8 | $\sup E$（Def 26） | $\max E$ | $\sup E$ 可能不属于 $E$（如 $E=(0,1)$ 时 $\sup E=1\notin E$）；$\max E$ 存在时必须属于 $E$；有 max 必有 sup 且相等 |
| 9 | $A\subset B$（Def 2） | $A\subsetneq B$ | $\subset$ 在官方笔记中即「子集」（允许相等）；真子集须另写 $\subsetneq$ 并附加 $A\ne B$ |
| 10 | 绝对收敛（Def 132） | 收敛（Def 114） | 绝对收敛 ⟹ 收敛（Thm 133）；收敛 ⇏ 绝对收敛（$\sum(-1)^n/n$，Cor 145）。绝对收敛可重排（Thm 146） |
| 11 | 绝对收敛 | 一致收敛（Def 264） | 完全不同的轴：前者是**数项级数**的和可交换；后者是**函数列**的收敛速率对 $x$ 一致。二者不可互相推出 |
| 12 | 可导（Def 198） | 连续（Def 167） | 可导 ⟹ 连续（Thm 202）；连续 ⇏ 可导（$|x|$ 在 $0$，Example 204） |
| 13 | $C^1$ | 可导（Def 198） | $C^1$ = $f'$ 存在**且连续**；可导只要求 $f'$ 存在。$f(x)=x^2\sin(1/x)$（$f(0)=0$）在 $0$ 可导但 $f'$ 不连续 |
| 14 | 开集的原像 | 开集的像 | 连续只保证 $f^{-1}(\text{开})$ 是开（Thm 171 + 拓扑刻画）；像不必是开（$f(x)=x^2$ 把 $(-1,1)$ 映成 $[0,1)$） |
| 15 | $\delta$ 依赖 $\epsilon$ 与 $x$（连续） | $\delta$ 只依赖 $\epsilon$（一致连续） | 量词顺序：$\forall\epsilon\forall x\exists\delta$ vs $\forall\epsilon\exists\delta\forall x$ |
| 16 | Riemann 可积 | 连续（Def 167） | 本课在 $C([a,b])$ 上证明可积（Thm 235）；可积函数类**严格大于**连续函数类（阶梯函数可积但不连续） |
| 17 | $f'\in C([a,b])$ | $f'$ 存在 | $f'$ 存在不保证 $f'$ 连续；IBP（Thm 252）与换元（Thm 256）都**额外假设** $f',g'\in C$ |
| 18 | 幂级数在内部一致收敛 | 幂级数在端点一致收敛 | Thm 278 只保证在 $[x_0-r,x_0+r]\subset(x_0-p,x_0+p)$ 上一致；端点 $x_0\pm p$ 处需另行判断 |
| 19 | 积分 $\int_a^bf$ | 黎曼和 $S_f(x,\xi)$ | 积分是**唯一确定的数**（Thm 235）；黎曼和**依赖分割与标签**，只在 $\|x\|\to0$ 时趋近积分（误差 $\le w_f(\|x\|)(b-a)$） |
| 20 | 序列极限（Def 63） | 函数极限（Def 150） | 函数极限要求 $c$ 是**聚点**且 $0<|x-c|$（挖心邻域，不看 $x=c$）；序列极限没有「挖心」概念。Thm 157 把二者桥接 |

**额外易混对（补充，共 8 组）**：

| # | A | B | 关键区别 |
|:--|:--|:--|:--|
| 21 | 单调有界定理（Thm 75） | 一致收敛（Def 264） | 前者是**单个数列**的收敛判据，后者是**函数列**整体的收敛模式，仅共享「单调/一致」字面相似 |
| 22 | 上界（Def 24） | 上确界（Def 26） | 上界有无穷多个（任一更大的数都是上界）；上确界是**最小**的那个 |
| 23 | 相对极值（Def 214） | 绝对极值（Def 181） | 相对极值只需要在某邻域内最大；绝对极值要求在整个定义域上最大。Thm 182 给出绝对极值的存在性 |
| 24 | 一致连续（Def 193） | Lipschitz 条件 | 一致连续的模可以是 $\omega(\delta)$（如 $\sqrt x$，$\omega=\sqrt\delta$）；Lipschitz 要求模是 $L\delta$，严格更强 |
| 25 | Weierstrass M-test（Thm 268） | 比较判别法（Thm 135） | M-test 是**函数级数一致收敛**的判别法；比较判别法是**数项级数收敛**的判别法。M-test 的 (b) 正是对 $\sum M_j$ 用比较的思想 |
| 26 | Bolzano–Weierstrass（Thm 102） | Cauchy 完备性（Thm 112） | 二者在 $\mathbb{R}$ 中互相等价（是 LUB 的两种面貌）；但 BW 谈**子列**，Cauchy 谈**内部互靠**，后者不需要预知极限 |
| 27 | Fermat 定理（Thm 215） | Rolle 定理（Thm 216） | Fermat 只需要**内点相对极值 + 可导**；Rolle 需要**闭区间连续 + 开区间可导 + 端点相等**，且结论是存在驻点（而非给出驻点处的极值性质） |
| 28 | MVT（Thm 218） | Taylor 定理（Thm 225） | MVT 是 $n=0$ 的 Taylor（$\frac{f'(c)}{1!}(x-x_0)^1$ 项）；Taylor 把 MVT 升到 $n$ 阶并给出余项 $R_n$ |

---

### 附：全课程编号索引（用到时快速定位）

| 编号范围 | 内容块 | 讲次 |
|:--|:--|:--|
| 1–9 | 集合、De Morgan、良序、归纳、几何和、Bernoulli | L1 |
| 10–13 | 基数、CSB、可数例 | L2 |
| 14–28 | Cantor 定理、$n<2^n$、实数刻画、有序集、LUB、$\mathbb{Q}$ 无 sup | L3 |
| 29–41 | 域、有序域、$\mathbb{R}$ 唯一性、$\sqrt2$ | L4 |
| 42–50 | Archimedes、稠密性、sup 的 $\epsilon$ 刻画、绝对值 | L5 |
| 51–58 | 三角不等式、十进制、不可数 | L6 |
| 59–71 | 序列、收敛、唯一性、$\epsilon$-$N$ 例题 | L6–L7 |
| 72–81 | 有界性、单调、子序列、DNC | L7 |
| 82–94 | 夹逼、保序、四则、几何序列 | L8 |
| 95–105 | 特殊序列、limsup/liminf、BW | L9 |
| 106–125 | Cauchy、完备性、级数、几何级数 | L10 |
| 126–137 | 调和级数、绝对收敛、比较、$p$-级数 | L11 |
| 138–146 | 比值、根值、交错、重排 | L12 |
| 147–160 | 聚点、函数极限、序列刻画 | L13–L14 |
| 161–178 | 单侧极限、连续、代数与复合 | L14–L15 |
| 179–190 | 有界函数、极值定理、介值定理、二分法 | L16 |
| 191–201 | 一致连续、可导、幂法则 | L17 |
| 202–211 | 可导⇒连续、Weierstrass 反例 | L18 |
| 212–221 | 求导法则、Rolle、MVT、单调性 | L19 |
| 222–228 | Taylor、二阶导判别、Riemann 积分导言 | L20 |
| 229–245 | 分割、标签、黎曼和、连续模、可积性 | L20–L21 |
| 246–256 | 可加性、单调性、FTC、IBP、Riemann–Lebesgue、换元 | L22 |
| 257–267 | 幂级数、收敛半径、逐点/一致、反例 | L23–L24 |
| 268–277 | M-test、保连续/可积/可导 | L24–L25 |
| 278–283 | 幂级数逐项运算、Weierstrass 逼近 | L25 |

{% endraw %}
