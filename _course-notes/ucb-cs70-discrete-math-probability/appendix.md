---
title: "核心定理与证明技巧速查表"
collection: course-notes
chapter: true
permalink: /course-notes/ucb-cs70-discrete-math-probability/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/ucb-cs70-discrete-math-probability/) · [← l27](/course-notes/ucb-cs70-discrete-math-probability/l27)

{% raw %}
## 核心定理与证明技巧速查表

> 本表按**类别**汇总全课程的关键定义、定理、公式与证明策略。建议在考前用它做"回忆式自测"：遮住右列，先自己复述定理的**条件**（最容易失分的地方）与**证明的关键一步**，再对照。

---

### 一、证明技巧（Lecture 0–3）

#### 1.1 五类证明策略的适用场景

| 策略 | 目标形式 | 做法 | 什么时候用 |
|:---|:---|:---|:---|
| **直接证明** (Direct Proof) | $P \Rightarrow Q$ | 假设 $P$，逐步推导出 $Q$ | 从假设出发能自然地"算出"结论时 |
| **逆否证明** (Contraposition) | $P \Rightarrow Q$ | 假设 $\neg Q$，推导出 $\neg P$ | 假设 $P$ 太弱、$\neg Q$ 信息更丰富时（如"若 $n^2$ 是偶数则 $n$ 是偶数"） |
| **反证法** (Contradiction) | $P$（任意命题） | 假设 $\neg P$，推出 $R \wedge \neg R$ | 结论是"不存在""不可能""无理数"等**否定性**命题时 |
| **分情形** (Proof by Cases) | $P$ | 把 $P$ 的假设域划分为**穷尽且互斥**的若干情形，逐一证明 | 假设信息不足以直接推导时（如带绝对值、按奇偶/模 $k$ 分类） |
| **构造性证明** (Constructive) | $\exists x\, P(x)$ | 显式给出一个 $x$ 并验证 $P(x)$ | 存在性命题，且能找到显式对象时 |
| **非构造性存在性** | $\exists x\, P(x)$ | 证明"若不存在则矛盾"，但不给出 $x$ | 找不到显式对象，或不需要知道具体对象时（如 $\sqrt2^{\sqrt2}$ 的有理性） |

#### 1.2 逻辑等价式（真值表可验证）

| 名称 | 等价式 |
|:---|:---|
| **逆否等价** | $P \Rightarrow Q \;\equiv\; \neg Q \Rightarrow \neg P$ |
| **蕴涵的析取形式** | $P \Rightarrow Q \;\equiv\; \neg P \vee Q$ |
| **双条件分解** | $P \Leftrightarrow Q \;\equiv\; (P \Rightarrow Q) \wedge (Q \Rightarrow P)$ |
| **De Morgan（合取）** | $\neg(P \wedge Q) \;\equiv\; \neg P \vee \neg Q$ |
| **De Morgan（析取）** | $\neg(P \vee Q) \;\equiv\; \neg P \wedge \neg Q$ |
| **分配律** | $P \wedge (Q \vee R) \equiv (P\wedge Q)\vee(P\wedge R)$；$P \vee (Q\wedge R) \equiv (P\vee Q)\wedge(P\vee R)$ |
| **排中律 / 矛盾律** | $P \vee \neg P \equiv \mathbf{T}$；$P \wedge \neg P \equiv \mathbf{F}$ |

**⚠️ 不等价**：$P\Rightarrow Q$ 与**逆命题** $Q \Rightarrow P$ 不等价；与**否命题** $\neg P \Rightarrow \neg Q$ 不等价。（后两者彼此等价，但与原命题无关。）

#### 1.3 量词及其否定

$$\neg \forall x\, P(x) \equiv \exists x\, \neg P(x), \qquad \neg \exists x\, P(x) \equiv \forall x\, \neg P(x)$$

**量词顺序不可交换**：$\forall x \exists y\, P(x,y) \not\equiv \exists y \forall x\, P(x,y)$。强的是后者。

**⚠️ 带蕴涵的量词否定**：$\neg \forall x\,(P(x) \Rightarrow Q(x)) \equiv \exists x\,(P(x) \wedge \neg Q(x))$（"所有质数都是奇数"的否定是"存在一个偶质数"，不是"所有质数都不是奇数"）。

#### 1.4 归纳法

**数学归纳法（简单归纳）**：若
1. **基础情形**：$P(0)$（或 $P(1)$）成立；
2. **归纳步骤**：$\forall k \ge 0$，$P(k) \Rightarrow P(k+1)$；

则 $\forall n \ge 0$，$P(n)$ 成立。

**强归纳法**：若
1. $P(0)$ 成立；
2. $\forall k \ge 0$，$\big(P(0) \wedge P(1) \wedge \cdots \wedge P(k)\big) \Rightarrow P(k+1)$；

则 $\forall n \ge 0$，$P(n)$ 成立。

**合法性的根源**：自然数的**良序原理**（Well-Ordering Principle）—— 任何非空自然数子集有最小元。若命题有反例，取**最小**反例 $n$，则 $P(n-1)$ 成立，与归纳步骤给出的 $P(n)$ 成立矛盾。

**归纳法三件套（写作模板）**：
```text
定理：∀n ≥ n₀, P(n)。
证明（对 n 归纳）：
  基础情形（n = n₀）：[验证 P(n₀)，写出具体计算]
  归纳步骤：假设 P(k) 成立（归纳假设）。要证 P(k+1)。
      P(k+1) 的左端
    = [把 P(k+1) 拆成 P(k) 的部分 + 新增项]   ← 关键第一步
    = [代入归纳假设]
    = [代数化简到 P(k+1) 的形式]              ← 目标明确
  故 P(k) ⇒ P(k+1)。
  由归纳法，∀n ≥ n₀, P(n)。 ∎
```

**何时必须用强归纳**：当 $P(k+1)$ 的证明需要用到 $P(j)$（$j$ 远小于 $k$）时。典型场景：
- **整数分解为质数之积**（因子 $a, b < k+1$）
- **$n \ge 12$ 可写成 $4x+5y$**（需要 $P(n-4)$）
- **图论中删点后图不连通**（需要强归纳或在每个连通分量上归纳）

**经典陷阱：归纳法伪证**。"所有马颜色相同"的伪证中，归纳步骤从 $k$ 到 $k+1$ 时把 $k+1$ 匹马分成两个大小为 $k$ 的子集，声称它们共享一匹马从而颜色相同——但 $k=1$ 时两个子集各只有 1 匹马，**交集为空**，无法传递。教训：**归纳步骤必须对所有 $k \ge$ 基础情形都成立**。

---

### 二、数论与密码（Lecture 4–6）

#### 2.1 同余与模运算

| 概念 | 定义 / 性质 |
|:---|:---|
| **同余** | $a \equiv b \pmod m \iff m \mid (a-b)$ |
| **等价关系** | 同余满足自反、对称、传递；$\mathbb{Z}$ 被划分为 $m$ 个等价类 |
| **加减乘保同余** | $a\equiv c$ 且 $b \equiv d \pmod m$ $\Rightarrow$ $a+b \equiv c+d$，$ab \equiv cd \pmod m$ |
| **除法（危险）** | $ab \equiv ac \pmod m$ **不能**直接推出 $b \equiv c$ |
| **消去律** | 若 $\gcd(c,m)=1$ 且 $ca \equiv cb \pmod m$，则 $a \equiv b \pmod m$ |
| **乘法逆元** | $x^{-1} \bmod m$ 满足 $x\cdot x^{-1} \equiv 1 \pmod m$ |

**逆元存在定理**：$x$ 在模 $m$ 下有乘法逆元 $\iff \gcd(x,m) = 1$。

**证明策略（逆元存在性）**：考察 $x\cdot 0, x\cdot 1, \dots, x\cdot(m-1)$ 这 $m$ 个数。若 $\gcd(x,m)=1$，它们在模 $m$ 下**两两不同**（否则 $x(i-j)\equiv 0$ 而 $\gcd(x,m)=1$ 迫使 $m \mid (i-j)$，但 $|i-j|<m$，只能 $i=j$）。$m$ 个不同的值填满 $m$ 个余数类，其中必有一个是 $1$，故逆元存在。

**快速幂（反复平方）**：把指数写成二进制，$x^n = \prod_{b_i=1} x^{2^i}$。复杂度 $O(\log n)$ 次模乘（朴素做法需 $O(n)$ 次）。

#### 2.2 欧几里得算法

$$\gcd(x,y) = \gcd(y,\; x \bmod y), \qquad \gcd(x,0) = x.$$

**证明策略**：证明两个集合 $\{d : d \mid x,\ d\mid y\}$ 与 $\{d : d \mid y,\ d \mid x\bmod y\}$ **完全相等**（互相包含），因此最大元相同。

**复杂度**：每**两次**递归调用第一个参数至少减半（因为 $x \bmod y < y$ 且 $x \bmod y \le x - y$），故 $O(\log x)$ 次调用。

**扩展欧几里得（Bézout）**：求出 $d = \gcd(x,y)$ 及整数 $a,b$ 使
$$ax + by = d.$$
**求逆元**：若 $\gcd(x,m)=1$，由 $ax + bm = 1$ 得 $x^{-1} \equiv a \pmod m$。

#### 2.3 费马小定理与中国剩余定理

**费马小定理（FLT）**：对**质数** $p$ 与任意 $a$ 满足 $\gcd(a,p)=1$（即 $a \not\equiv 0$）：
$$a^{p-1} \equiv 1 \pmod p.$$

**证明策略**：关键引理 —— $a\cdot 1, a\cdot 2, \dots, a\cdot(p-1) \pmod p$ 是 $\{1,2,\dots,p-1\}$ 的一个**排列**。于是两组数的乘积同余：
$$a^{p-1}(p-1)! \equiv (p-1)! \pmod p \;\Longrightarrow\; a^{p-1}\equiv 1 \pmod p.$$
（引理成立的理由：若 $a i \equiv a j \pmod p$，由 $p$ 是质数且 $a\not\equiv0$ 可消去 $a$，得 $i\equiv j$；$p-1$ 个不同的非零余数正好填满全部非零余数类。）

**⚠️ 条件不可省**：FLT 对**合数模**不成立（$m=4,a=2$：$2^3 = 8 \equiv 0 \not\equiv 1$）；且要求 $a \not\equiv 0 \pmod p$。

**FLT 的推论（求逆元）**：$a^{-1} \equiv a^{p-2} \pmod p$。

**中国剩余定理（CRT）**：若 $m_1,\dots,m_k$ **两两互质**，$M = m_1m_2\cdots m_k$，则同余方程组
$$x \equiv a_1 \pmod{m_1},\ \dots,\ x \equiv a_k \pmod{m_k}$$
在模 $M$ 下有**唯一解**。

**构造性求解**：令 $M_i = M/m_i$，$y_i \equiv M_i^{-1} \pmod{m_i}$，则
$$x \equiv \sum_{i=1}^k a_i M_i y_i \pmod M.$$
**验证机制**：模 $m_j$ 时，和式中除第 $j$ 项外都含因子 $M_i$（$i\ne j$）从而 $\equiv 0$，而第 $j$ 项 $\equiv a_j M_j y_j \equiv a_j \pmod{m_j}$。

**唯一性**：若 $x, x'$ 都是解，则每个 $m_i \mid (x-x')$；两两互质 $\Rightarrow M \mid (x-x')$。

#### 2.4 RSA

**密钥生成**：
1. 选两个大质数 $p \ne q$，令 $N = pq$；
2. $\phi(N) = (p-1)(q-1)$；
3. 选 $e$ 满足 $\gcd(e, \phi(N)) = 1$；
4. 计算 $d \equiv e^{-1} \pmod{\phi(N)}$。

**公钥** $(N,e)$，**私钥** $(N,d)$（$p,q$ 必须销毁）。

**加解密**：$E(x) = x^e \bmod N$；$D(y) = y^d \bmod N$。

**正确性定理**：$D(E(x)) = (x^e)^d \equiv x \pmod N$ 对所有 $x \in \{0,\dots,N-1\}$ 成立。

**证明策略**：由 $ed \equiv 1 \pmod{\phi(N)}$ 得 $ed = 1 + k(p-1)(q-1)$。分两种情形：
- **$\gcd(x,N)=1$**：由欧拉定理 $x^{\phi(N)} \equiv 1$，故 $x^{ed} = x\cdot(x^{\phi(N)})^k \equiv x$。
- **$\gcd(x,N) \ne 1$**：此时 $x$ 是 $p$ 或 $q$ 的倍数（$N=pq$）。分别验证模 $p$ 与模 $q$ 同余于 $x$（用 FLT 处理非零的那个模），再由 CRT 得模 $N=pq$ 同余于 $x$。

**安全性基础**：破解需要分解 $N$；已知最好的分解算法是亚指数时间，无多项式时间算法。

**教科书 RSA 的缺陷（必须加 padding）**：
- **确定性**：相同明文恒得相同密文，可被字典攻击。
- **可乘性**：$E(x_1)E(x_2) \equiv E(x_1x_2) \pmod N$，可被选择明文攻击伪造。
- **小指数攻击**、共模攻击等。

**质数生成**：由**素数定理** $\pi(n) \approx n/\ln n$，$n$ 附近的质数密度约为 $1/\ln n$。随机取奇数 + 素性测试（Miller–Rabin）即可高效生成。

---

### 三、代数与编码（Lecture 7–8）

#### 3.1 多项式

| 定理 | 内容 |
|:---|:---|
| **带余除法 / 因子定理** | $(x-r) \mid P(x) \iff P(r) = 0$ |
| **根的上界** | 次数为 $d$ 的**非零**多项式最多有 $d$ 个根 |
| **插值定理** | 给定 $d+1$ 个 $x_i$ 两两不同的点，**存在唯一**次数 $\le d$ 的多项式经过它们 |
| **Schwartz–Zippel** | 两个不同的次数 $\le d$ 多项式在随机点 $r \in \mathbb{F}_p$ 上取值相等的概率 $\le d/p$ |

**根定理的证明机制**：对次数归纳。若 $P(r)=0$，由带余除法 $P(x) = (x-r)Q(x) + c$，代入 $x=r$ 得 $c=P(r)=0$，故 $P(x)=(x-r)Q(x)$，$\deg Q = d-1$。任何其他根 $r' \ne r$ 必满足 $Q(r')=0$，由归纳假设 $Q$ 最多 $d-1$ 个根。

**插值的存在性（Lagrange 构造）**：
$$\Delta_i(x) = \prod_{j \ne i} \frac{x - x_j}{x_i - x_j}, \qquad \Delta_i(x_j) = \delta_{ij}, \qquad P(x) = \sum_{i=1}^{d+1} y_i \Delta_i(x).$$
**唯一性（反证）**：若 $P_1 \ne P_2$ 都经过这 $d+1$ 个点，则 $P_1 - P_2$ 是次数 $\le d$ 的非零多项式却有 $d+1$ 个根，违反根定理。

**⚠️ 有限域上的重要区别**：在 $\mathbb{Z}_p$ 上，**不同多项式可以定义同一个函数**（如 $x^p \equiv x$，由 FLT）。因此"多项式相等"（系数逐项相等）与"作为函数相等"不同。**但次数 $\le p-1$ 的多项式由函数值唯一确定**（否则差多项式次数 $\le p-1$ 却在 $p$ 个点上全为 0，超过根的上界）。

#### 3.2 秘密共享与纠错码

**(n,k) 门限秘密共享（Shamir 方案）**：选随机多项式 $P$，$\deg P = k-1$，$P(0) = s$（秘密）；第 $i$ 人得 $(i, P(i))$，$i=1,\dots,n$。

- **正确性**：任意 $k$ 个份额由插值定理唯一确定 $P$，取 $P(0)$ 即得 $s$。
- **安全性**：任意 $k-1$ 个份额**不泄露 $s$ 的任何信息**。论证：对任意猜测 $s'$，都存在一个经过这 $k-1$ 个点且常数项为 $s'$ 的次数 $\le k-1$ 多项式（$k$ 个约束、$k$ 个系数，可解）。故这 $k-1$ 个份额与 $s$ 的取值**在信息论上独立**。

**汉明距离与纠错能力**：码的最小距离 $d$ 满足
$$\text{可纠正} \left\lfloor \frac{d-1}{2} \right\rfloor \text{ 个错误}, \qquad \text{可检测 } d-1 \text{ 个错误}.$$
**证明机制**：若收到的串 $y$ 与两个不同码字 $c_1,c_2$ 的距离都 $\le t$，则 $d(c_1,c_2) \le 2t < d$，矛盾（三角不等式）。

**Reed–Solomon 码**：消息 $m_0,\dots,m_{k-1}$ 视为次数 $<k$ 的多项式 $P$ 的系数，码字为 $(P(\alpha_1),\dots,P(\alpha_n))$（$\alpha_i$ 两两不同）。

**最小距离**：两个不同的次数 $<k$ 多项式最多在 $k-1$ 个点相等，故码字至少在 $n-k+1$ 个坐标不同，即 $d = n-k+1$。

**Berlekamp–Welch 译码**（$n \ge k + 2e$，最多 $e$ 个错误）：
1. 引入**错误位置多项式** $E(x) = \prod_{i \in \text{error}}(x - \alpha_i)$，$\deg E \le e$（取首一）。
2. 令 $Q(x) = P(x)E(x)$，$\deg Q \le k-1+e$。
3. 对每个 $i$：$Q(\alpha_i) = y_i E(\alpha_i)$，这是关于 $Q,E$ 的**系数**的线性方程（在正确的点上成立；在错误的点上不一定成立）。
4. 未知量个数 $\le (k+e) + (e+1)$，方程个数 $= n \ge k+2e+1$，解出 $Q,E$。
5. $P(x) = Q(x)/E(x)$（多项式除法）。

**⚠️ 条件**：$n < k+2e$ 时无法保证唯一译码。

---

### 四、图论与匹配（Lecture 9–11）

#### 4.1 图论基本事实

| 概念 / 定理 | 内容 |
|:---|:---|
| **握手引理** | $\sum_{v \in V} \deg(v) = 2\lvert E \rvert$。推论：**奇度顶点个数为偶数** |
| **完全图 $K_n$** | $\lvert V \rvert = n$，$\lvert E \rvert = \binom n2$，$\deg(v) = n-1$ |
| **欧拉回路判定** | 连通图有欧拉回路 $\iff$ **每个顶点度为偶** |
| **欧拉路径判定** | 连通图有欧拉路径 $\iff$ 恰有 **0 或 2** 个奇度顶点 |
| **欧拉公式（平面图）** | 连通平面图：$V - E + F = 2$（$F$ 含外部面） |
| **平面图边数上界** | 简单平面图 $V \ge 3$：$E \le 3V - 6$；若还二分：$E \le 2V - 4$ |
| **非平面判定** | $K_5$：$10 > 3\cdot5-6 = 9$ ✓；$K_{3,3}$：$9 > 2\cdot6-4 = 8$ ✓ |

**握手引理的证明（双重计数）**：按顶点数"顶点—边"关联对，得 $\sum_v \deg v$；按边数同一些对，每边贡献 2，得 $2|E|$。同一集合两种数法，故相等。

**欧拉回路充分性的构造**：从任一点出发，沿未用边走到不能走为止（因每点剩余度为偶，必然回到起点，形成一个回路）。若仍有未用边，由图的**连通性**，存在回路外的边与已走回路共享顶点，从共享点出发走一个新的回路，再把两个回路在该点"拼接"成一个大回路。重复直到用尽所有边。

**欧拉公式的证明（对边数归纳）**：若图是树，$E = V-1$，$F=1$，得 $V-E+F = V-(V-1)+1 = 2$ ✓。若图含环，删去环上一条边：$E$ 减 1，$F$ 减 1（两个相邻面合并），$V-(E-1)+(F-1) = V-E+F$ 不变。反复删边直到成树。

**平面图边数上界的证明**：设每个面的边界长度为 $\ell_i \ge 3$，则 $\sum_i \ell_i = 2E$（每条边被两个面共享），故 $3F \le 2E$。代入 $V - E + F = 2$（即 $F = 2 - V + E$）：$3(2-V+E) \le 2E \Rightarrow E \le 3V - 6$。二分图无三角形，每个面 $\ge 4$ 条边，同理得 $E \le 2V-4$。

**⚠️ 方向性**：$E \le 3V-6$ 是平面的**必要**条件，**不充分**（满足不等式仍可能非平面）。

#### 4.2 树

**树的核心等价刻画**（$V$ 个顶点的图）：
```text
(i)   G 是树（连通且无环）
(ii)  G 连通且有 V-1 条边
(iii) G 无环且有 V-1 条边
(iv)  任意两顶点间恰好有一条简单路径
(v)   G 连通，但删去任一条边后不连通（极小连通）
(vi)  G 无环，但添加任一条边都产生环（极大无环）
```

**证明链条（关键步骤）**：
- $(i) \Rightarrow (ii)$：对顶点数归纳。树有叶子（$\deg = 1$，证明：若所有度 $\ge 2$ 则 $\sum \deg \ge 2V$，由握手引理 $E \ge V$，但连通图 $E \ge V-1$ 且无环图 $E \le V-1$，故 $E = V-1$，此时若所有度 $\ge 2$ 则 $\sum\deg = 2V-2$ 要求恰有两个度 1 顶点）。删去一个叶子得 $V-1$ 个顶点的树，由归纳假设它有 $(V-1)-1$ 条边，加回删掉的边得 $V-1$ 条。
- $(ii) \Rightarrow (iii)$：若连通且有 $V-1$ 条边但含环，删去环上一边仍连通且有 $V-2$ 条边，但连通图至少 $V-1$ 条边，矛盾。
- $(iii) \Rightarrow (i)$：若无环有 $V-1$ 条边，设它有 $c$ 个连通分量，每个分量是树，总边数 $= V - c = V-1$，故 $c=1$，连通。

#### 4.3 稳定匹配（Gale–Shapley）

**不稳定对（blocking pair）**：$(m,w)$ 未配对，但 $m$ 更喜欢 $w$ 且 $w$ 更喜欢 $m$。
**稳定匹配**：不存在不稳定对的完美匹配。

**Gale–Shapley 提出-拒绝算法**：
```text
初始化：所有男女均未匹配，每个男士的"下一个追求对象"指针指向其偏好表首位。
while 存在未匹配且未向所有女士提过议的男士 m:
    w ← m 尚未提过议的最偏好的女士
    m 向 w 提出
    if w 未匹配:
        w 接受 m
    else:
        m' ← w 当前的伴侣
        if w 更喜欢 m 而不是 m':
            w 接受 m，拒绝 m'（m' 变回未匹配）
        else:
            w 拒绝 m（m 保持未匹配）
```

**三个正确性定理**：
1. **终止性**：每轮至少一次拒绝；总提议次数 $\le n^2$，有限。
2. **完美匹配**：结束时不可能有单身者。若男士 $m$ 单身，他已向所有女士提过议且全被拒；每个拒绝他的女士最终都匹配给了她更喜欢的男士，故需要 $n$ 个不同的男士与之配对，但除去 $m$ 只有 $n-1$ 个男士，矛盾。
3. **稳定性**（反证）：设 $(m,w)$ 是不稳定对，$m$ 的配偶是 $w'$，$w$ 的配偶是 $m'$。因 $m$ 更喜欢 $w$ 而 $w'$ 是最终配偶，$m$ 必然曾向 $w$ 提过议并被 $w$ 拒绝。$w$ 只在她已有更喜欢的伴侣时才拒绝，故 $w$ 更喜欢 $m'$ 而不是 $m$——与 $(m,w)$ 是不稳定对（$w$ 更喜欢 $m$）矛盾。

**最优性定理（Gale–Shapley）**：算法产出的是**对提出方最优**的稳定匹配——每个男士得到他在任何稳定匹配中可能得到的最好配偶；相应地，每个女士得到最差的。

**关键引理（最优性的全部基础）**：**若女士 $w$ 曾拒绝男士 $m$，则 $w$ 在任何稳定匹配中都不会与 $m$ 配对。**
*证明思路*：$w$ 拒绝 $m$ 时她已有一个更喜欢的 $m'$。在该稳定匹配中：若 $w$ 与 $m$ 配对，则 $m'$ 必与某个 $w'$ 配对。需证 $(m',w)$ 构成不稳定对：$w$ 更喜欢 $m'$（已知），且 $m'$ 更喜欢 $w$ 而非 $w'$——这需要归纳论证"$m'$ 只在被更喜欢的女士拒绝时才退而求其次"。由此得矛盾。

**推论**：**所有稳定匹配中每个男士的配偶相同**（男士最优稳定匹配唯一）；女士侧同理。**但不同的稳定匹配可以给女士不同结果**。

**⚠️ 重要事实**：稳定匹配**不必唯一**；算法的输出取决于谁是提出方（提出方占优）。这在现实中（住院医师匹配）有伦理含义。

---

### 五、可数性与可计算性（Lecture 12–13）

#### 5.1 可数性

| 集合 | 可数？ | 依据 |
|:---|:---|:---|
| $\mathbb{N}$, $\mathbb{Z}$, $\mathbb{Q}$ | **可数** | 存在显式双向枚举 |
| $\mathbb{N} \times \mathbb{N}$ | **可数** | 对角线枚举 |
| 有限个/可数个数可数集的并 | **可数** | 归约到 $\mathbb{N}\times\mathbb{N}$ |
| 所有有限字符串 $\{0,1\}^*$ | **可数** | 按长度、再按字典序 |
| 所有程序（按任意语言） | **可数** | 程序是有限字符串 |
| $\mathbb{R}$ | **不可数** | 对角线论证 |
| $\mathcal{P}(\mathbb{N})$ | **不可数** | Cantor 定理 |
| 所有函数 $f:\{0,1\}^\infty \to \{0,1\}$ | **不可数** | 与 $\mathcal{P}(\mathbb{N})$ 等势 |

**定义**：$A$ 可数 $\iff$ $A$ 有限，或存在双射 $f: \mathbb{N} \to A$（即 $A$ 可被排成一个序列 $a_0,a_1,a_2,\dots$）。

**关键双射**：
- $\mathbb{Z}$：$f(n) = \begin{cases} n/2 & n \text{ 偶} \\ -(n+1)/2 & n \text{ 奇}\end{cases}$ 给出 $0, -1, 1, -2, 2, \dots$
- $\mathbb{N}\times\mathbb{N}$：按 $i+j$ 递增的对角线枚举；配对函数 $\pi(i,j) = \frac{(i+j)(i+j+1)}{2} + j$。

**Cantor 定理**：对任意集合 $A$，$|A| < |\mathcal{P}(A)|$。

*证明（对角线）*：显然 $|A| \le |\mathcal{P}(A)|$（$a \mapsto \{a\}$ 单射）。假设存在满射 $f: A \to \mathcal{P}(A)$，构造
$$D = \{a \in A : a \notin f(a)\}.$$
$D \subseteq A$ 故 $D \in \mathcal{P}(A)$。由 $f$ 满射，存在 $d$ 使 $f(d) = D$。问：$d \in D$？
- 若 $d \in D$，由 $D$ 定义 $d \notin f(d) = D$，矛盾。
- 若 $d \notin D$，则 $d \in f(d) = D$，矛盾。
故满射不存在，$|A| \ne |\mathcal{P}(A)|$。∎

**$\mathbb{R}$ 不可数（对角线）**：假设 $\mathbb{R}$ 可数，则 $[0,1)$ 可数，列出其小数展开 $x_1, x_2, x_3, \dots$。构造 $y$，其第 $n$ 位小数 $= (x_n$ 的第 $n$ 位小数 $+ 1) \bmod 10$。则 $y$ 与每个 $x_n$ 至少在第 $n$ 位不同，故 $y$ 不在列表中，矛盾。（技术细节：避开以全 9 结尾的展开以避免 $0.4999\ldots = 0.5$ 的歧义，这不会影响论证。）

**超越数存在性（非构造性）**：代数数（整系数多项式的根）可数（每个多项式由有限个整数系数确定，可数多个多项式，每个有限个根）；而 $\mathbb{R}$ 不可数。故存在**不是**代数数的实数（超越数）。注意这个证明**不告诉我们任何具体的超越数**。

**通往可计算性的桥梁**：程序可数（有限字符串），而函数不可数，故**绝大多数函数不可计算**——这是"用计数论证不可能性"的范式。

#### 5.2 可计算性与停机问题

**停机问题**：给定程序 $P$ 与输入 $x$，判定 $P(x)$ 是否终止。

**定理**：停机问题**不可判定（undecidable）**。

*证明（反证 + 自指）*：假设存在判定器 `Halt(P, x)`，总能在有限时间内正确返回"停机"或"不停机"。构造
```text
Turing(P):
    if Halt(P, P) == "停机":
        while true: pass      # 故意死循环
    else:
        return                # 停机
```
考虑 `Turing(Turing)`：
- 若 `Halt(Turing, Turing)` 返回"停机"，则 `Turing(Turing)` 进入死循环——**不停机**，与判定器输出矛盾。
- 若返回"不停机"，则 `Turing(Turing)` 立即返回——**停机**，同样矛盾。

两分支皆矛盾，故 `Halt` 不存在。∎

**关键洞察**：程序本身是字符串，可以作为另一个程序的输入——**自指由此构造出来**。这与 Cantor 对角线论证、$\mathbb{R}$ 不可数论证是**同一个技巧**：

| 对角化实例 | "行" | "列" | 构造的"反例" |
|:---|:---|:---|:---|
| Cantor 定理 | 元素 $a$ | 子集 $f(a)$ | $D = \{a : a \notin f(a)\}$ |
| $\mathbb{R}$ 不可数 | 列表中的实数 $x_n$ | 第 $n$ 位小数 | $y$ 的第 $n$ 位 $= x_n$ 第 $n$ 位 $+1$ |
| 停机问题 | 程序 $P$ | 输入 $P$ | `Turing(Turing)` |

**可判定 vs 可识别**：
- **可判定（decidable）**：存在程序对**所有**输入都终止，且正确输出 yes/no。
- **可识别（recognizable / semi-decidable）**：存在程序在答案为 yes 时终止并输出 yes；答案为 no 时可能永远运行。

**定理**：停机问题是**可识别但不可判定**的。
- *可识别*：模拟器 `Sim(P,x)` 直接运行 $P(x)$；若终止则输出 yes。若 $P(x)$ 不停机，模拟器也不停机——这符合可识别的定义。
- *不可判定*：已证。

**补集论证（通用技巧）**：若 $L$ 与 $\bar L$ 都可识别，则 $L$ 可判定。
*证明*：设 $M_1$ 识别 $L$，$M_2$ 识别 $\bar L$。构造 $M$：在输入 $x$ 上**并行交替**模拟 $M_1(x)$ 与 $M_2(x)$ 的步骤。$x$ 必属于 $L$ 或 $\bar L$ 之一，故其中一个必然终止并给出答案，$M$ 据此输出。$M$ 对所有输入终止，故 $L$ 可判定。∎

**推论**：停机问题的补集**不可识别**（否则停机问题就可判定了）。

**归约（Reduction）模式**：要证问题 $A$ 不可判定，证明"若能判定 $A$，就能判定停机问题"。典型例子：判定"程序 $P$ 在输入 $x$ 上是否输出 0"、判定"两个程序是否对所有输入行为相同"。

**⚠️ 常见误解**："停机问题不可判定"**不**意味着我们无法判断任何具体程序是否停机——很多程序显然停机。不可判定指的是**不存在一个统一的算法**处理所有程序-输入对。

---

### 六、计数与概率基础（Lecture 14–18）

#### 6.1 计数法则

| 法则 | 公式 |
|:---|:---|
| **乘法法则（第一法则）** | 依次有 $n_1,\dots,n_k$ 种选择 $\Rightarrow$ $n_1 n_2 \cdots n_k$ 种 |
| **和法则（第二法则）** | 集合划分为**不交**部分 $\Rightarrow$ 总数 $=$ 各部分之和 |
| **互补法则** | $\lvert A \rvert = \lvert U \rvert - \lvert \bar A \rvert$ |
| **排列** | $P(n,k) = \dfrac{n!}{(n-k)!}$ |
| **组合** | $\dbinom nk = \dfrac{n!}{k!(n-k)!}$ |
| **星棒法（非负解）** | $x_1+\cdots+x_k = n$，$x_i \ge 0$ 的解数 $= \dbinom{n+k-1}{k-1}$ |
| **星棒法（正解）** | $x_i \ge 1$ 的解数 $= \dbinom{n-1}{k-1}$ |
| **容斥（2 集）** | $\lvert A\cup B\rvert = \lvert A\rvert + \lvert B\rvert - \lvert A\cap B\rvert$ |
| **容斥（3 集）** | $\lvert A\cup B\cup C\rvert = \lvert A\rvert+\lvert B\rvert+\lvert C\rvert - \lvert A\cap B\rvert-\lvert A\cap C\rvert-\lvert B\cap C\rvert + \lvert A\cap B\cap C\rvert$ |

**组合数的关键论证**：$\binom nk$ 的分子 $n(n-1)\cdots(n-k+1)$ 数的是**有序排列**；每个 $k$ 元子集被数了 $k!$ 次（其内部的排列数），故除以 $k!$ 得子集数。

**星棒法的双射**：把解 $(x_1,\dots,x_k)$ 映射到"$x_1$ 个星、$1$ 个棒、$x_2$ 个星、$1$ 个棒、……、$x_k$ 个星"的序列。这是双射，序列总数 $= \binom{n+k-1}{k-1}$。

**容斥的一般形式（指示函数证明）**：
$$\mathbb{1}_{A_1 \cup \cdots \cup A_n} = 1 - \prod_{i=1}^n (1 - \mathbb{1}_{A_i}) = \sum_{\emptyset \ne S \subseteq [n]} (-1)^{|S|+1} \mathbb{1}_{\bigcap_{i\in S} A_i}.$$
两边取期望（即求和），即得容斥公式。

#### 6.2 组合恒等式

$$\binom nk = \binom{n}{n-k}$$
$$\binom nk = \binom{n-1}{k-1} + \binom{n-1}{k} \quad \text{(Pascal 法则)}$$
$$\sum_{k=0}^n \binom nk = 2^n, \qquad \sum_{k=0}^n (-1)^k\binom nk = 0$$
$$\sum_{k=0}^n \binom nk^2 = \binom{2n}{n}$$
$$\sum_{k=0}^n k\binom nk = n 2^{n-1}$$
$$\sum_{k} \binom mk \binom{n}{r-k} = \binom{m+n}{r} \quad \text{(Vandermonde)}$$
$$(x+y)^n = \sum_{k=0}^n \binom nk x^k y^{n-k} \quad \text{(二项式定理)}$$

**证明策略一览**：

| 恒等式 | 策略 | 要数的对象 |
|:---|:---|:---|
| $\binom nk = \binom n{n-k}$ | 双射 | $k$ 元子集 $\leftrightarrow$ 其补集（$(n-k)$ 元子集） |
| Pascal 法则 | 双计数 | $n$ 元集的 $k$ 元子集：分"含特定元素 $a$"与"不含 $a$"两类 |
| $\sum_k \binom nk = 2^n$ | 双射 | 子集 $\leftrightarrow$ 长度 $n$ 的 0/1 串 |
| $\sum_k \binom nk^2 = \binom{2n}{n}$ | 双计数 | $n$ 男 $n$ 女中选 $n$ 人：直接数 vs 按男生个数分类 |
| $\sum_k k\binom nk = n2^{n-1}$ | 双计数 | "（子集，子集内一个元素）"这样的对 |
| Vandermonde | 双计数 | $m$ 个红球 $n$ 个蓝球中选 $r$ 个：按红球个数分类 |
| 二项式定理 | 双计数 | $(x+y)^n$ 展开时每项选 $x$ 或 $y$ 的方式数 |

**组合证明的方法论**：
1. 识别等式两边**各自在数什么**；
2. **创造性地定义**要数的集合（常常是"对"或"带标记的对象"）；
3. 验证两种数法**无重、无漏**；
4. 若是双射，必须同时验证**单射**与**满射**。

#### 6.3 概率公理与基本性质

**概率空间 $(\Omega, \mathcal{F}, \Pr)$**：$\Omega$ 样本空间（所有结果），事件 $A \subseteq \Omega$。

**Kolmogorov 三公理**：
1. **非负性**：$\Pr[A] \ge 0$；
2. **归一性**：$\Pr[\Omega] = 1$；
3. **可数可加性**：两两不交的事件 $\Rightarrow \Pr\!\left[\bigcup_i A_i\right] = \sum_i \Pr[A_i]$。

**由公理推出的性质（全部可证）**：

| 性质 | 公式 | 证明要点 |
|:---|:---|:---|
| 空集概率 | $\Pr[\emptyset] = 0$ | $\Omega$ 与 $\emptyset$ 不交且并集为 $\Omega$ |
| 补集 | $\Pr[\bar A] = 1 - \Pr[A]$ | $A$ 与 $\bar A$ 不交且并为 $\Omega$ |
| 单调性 | $A \subseteq B \Rightarrow \Pr[A] \le \Pr[B]$ | $B = A \cup (B\setminus A)$，不交并 |
| 并（容斥起点） | $\Pr[A\cup B] = \Pr[A]+\Pr[B]-\Pr[A\cap B]$ | 拆成三个不交部分 |
| **Union Bound** | $\Pr\!\left[\bigcup_i A_i\right] \le \sum_i \Pr[A_i]$ | 对 $n$ 归纳；**不需要独立/互斥** |

**⚠️ 最容易失分的点**：只有 **互斥（disjoint）** 时才有 $\Pr[A\cup B] = \Pr[A]+\Pr[B]$。

**均匀概率空间（古典概型）**：$|\Omega| = n$，每个结果等可能 $\Rightarrow \Pr[A] = |A|/n$。

**⚠️ 陷阱**："掷两枚硬币"的结果**不是** {0 个正面, 1 个正面, 2 个正面} 三种等可能——"1 个正面"含 HT 与 TH 两个结果。正确的样本空间是 $\{HH, HT, TH, TT\}$，故 $\Pr[1\text{ 正面}] = 1/2$ 而非 $1/3$。

#### 6.4 条件概率与贝叶斯

$$\Pr[A \mid B] = \frac{\Pr[A \cap B]}{\Pr[B]} \quad (\Pr[B] > 0)$$

**直观**：已知 $B$ 发生，样本空间从 $\Omega$ **缩减**到 $B$，重新归一化。

**乘法法则**：$\Pr[A \cap B] = \Pr[A\mid B]\Pr[B] = \Pr[B\mid A]\Pr[A]$
**链式法则**：$\Pr[A_1\cap\cdots\cap A_n] = \Pr[A_1]\Pr[A_2\mid A_1]\cdots\Pr[A_n \mid A_1\cap\cdots\cap A_{n-1}]$

**全概率法则**：若 $B_1,\dots,B_n$ 是 $\Omega$ 的划分，则
$$\Pr[A] = \sum_{i=1}^n \Pr[A \mid B_i]\Pr[B_i].$$

**贝叶斯法则**：
$$\Pr[B \mid A] = \frac{\Pr[A \mid B]\Pr[B]}{\Pr[A]} = \frac{\Pr[A\mid B]\Pr[B]}{\sum_i \Pr[A\mid B_i]\Pr[B_i]}$$
- $\Pr[B]$ = **先验（prior）**
- $\Pr[A \mid B]$ = **似然（likelihood）**
- $\Pr[B \mid A]$ = **后验（posterior）**

**疾病检测算例（必背结论）**：患病率 $1/1000$，灵敏度 $99\%$，特异度 $99\%$：
$$\Pr[\text{病} \mid +] = \frac{0.99 \times 0.001}{0.99\times0.001 + 0.01\times0.999} = \frac{0.00099}{0.01098} \approx 9\%.$$
**为什么这么低**：假阳性（$0.01 \times 999 \approx 10$ 人）远远多于真阳性（$\approx 1$ 人）。

**MAP vs MLE**：$\hat\theta_{\text{MAP}} = \arg\max_\theta \Pr[\text{data}\mid\theta]\Pr[\theta]$；$\hat\theta_{\text{MLE}} = \arg\max_\theta \Pr[\text{data}\mid\theta]$。先验均匀时二者一致。

**⚠️ 高频错误**：混淆 $\Pr[A\mid B]$ 与 $\Pr[B\mid A]$（**检察官谬误**）。上面的算例中 $\Pr[+ \mid \text{病}] = 99\%$ 而 $\Pr[\text{病} \mid +] \approx 9\%$，差了十倍以上。

#### 6.5 独立性与事件组合

**独立**：$A \perp B \iff \Pr[A \cap B] = \Pr[A]\Pr[B] \iff \Pr[A\mid B] = \Pr[A]$。

**相互独立**：$A_1,\dots,A_n$ 相互独立要求**所有**子集 $S \subseteq [n]$ 满足 $\Pr\!\left[\bigcap_{i\in S} A_i\right] = \prod_{i\in S}\Pr[A_i]$。

**⚠️ 两两独立 $\not\Rightarrow$ 相互独立**（经典反例）：掷两枚硬币，$A$ = 第一枚正面，$B$ = 第二枚正面，$C$ = 两枚相同。
- $\Pr[A]=\Pr[B]=\Pr[C]=1/2$；
- 两两独立：$\Pr[A\cap B]=1/4$，$\Pr[A\cap C]=1/4$，$\Pr[B\cap C]=1/4$ ✓；
- 但 $\Pr[A\cap B\cap C] = \Pr[HH] = 1/4 \ne 1/8 = \Pr[A]\Pr[B]\Pr[C]$ ✗。

**⚠️ 不交 vs 独立**（最经典的混淆）：
- **不交**：$A \cap B = \emptyset$（集合层面）。
- **独立**：$\Pr[A\cap B]=\Pr[A]\Pr[B]$（概率层面）。
- **若 $A,B$ 不交且 $\Pr[A],\Pr[B] > 0$，则它们一定不独立**（$0 \ne \Pr[A]\Pr[B] > 0$）。

**条件独立**：$\Pr[A\cap B \mid C] = \Pr[A\mid C]\Pr[B\mid C]$。**条件独立不蕴含独立，独立也不蕴含条件独立**。

**互补技巧（求"至少一个"）**：
$$\Pr[\text{至少一个 } A_i] = 1 - \Pr[\text{全不发生}] \overset{\text{独立}}{=} 1 - \prod_i (1 - \Pr[A_i]).$$

---

### 七、随机变量、期望、方差与极限定理（Lecture 19–26）

#### 7.1 离散分布速查

| 分布 | PMF | 参数 | $\mathbb{E}[X]$ | $\operatorname{Var}(X)$ | 典型场景 |
|:---|:---|:---|:---|:---|:---|
| **伯努利** $\mathrm{Ber}(p)$ | $\Pr[X=1]=p$，$\Pr[X=0]=1-p$ | $p\in[0,1]$ | $p$ | $p(1-p)$ | 单次成败（指示变量） |
| **二项** $\mathrm{Bin}(n,p)$ | $\dbinom nk p^k(1-p)^{n-k}$ | $n\in\mathbb{N},p$ | $np$ | $np(1-p)$ | $n$ 次独立试验的成功数 |
| **几何** $\mathrm{Geo}(p)$ | $(1-p)^{k-1}p$，$k\ge1$ | $p\in(0,1]$ | $\dfrac 1p$ | $\dfrac{1-p}{p^2}$ | 首次成功所需试验次数 |
| **泊松** $\mathrm{Pois}(\lambda)$ | $\dfrac{\lambda^k}{k!}e^{-\lambda}$，$k\ge0$ | $\lambda>0$ | $\lambda$ | $\lambda$ | 单位时间稀有事件数 |
| **离散均匀** | $\Pr[X=i]=\frac1n$，$i\in\{1..n\}$ | $n$ | $\dfrac{n+1}{2}$ | $\dfrac{n^2-1}{12}$ | 掷骰子 |

**泊松是二项分布的极限**：$n\to\infty$，$p\to 0$，$np \to \lambda$ 时 $\mathrm{Bin}(n,p) \to \mathrm{Pois}(\lambda)$。

**几何分布的无记忆性**：$\Pr[X > n+m \mid X > n] = \Pr[X > m]$。

#### 7.2 连续分布速查

| 分布 | PDF | 参数 | $\mathbb{E}[X]$ | $\operatorname{Var}(X)$ | CDF |
|:---|:---|:---|:---|:---|:---|
| **连续均匀** $U[a,b]$ | $\dfrac{1}{b-a}$，$x\in[a,b]$ | $a<b$ | $\dfrac{a+b}{2}$ | $\dfrac{(b-a)^2}{12}$ | $\dfrac{x-a}{b-a}$ |
| **指数** $\mathrm{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$，$x\ge0$ | $\lambda>0$ | $\dfrac1\lambda$ | $\dfrac{1}{\lambda^2}$ | $1-e^{-\lambda x}$ |
| **正态** $\mathcal N(\mu,\sigma^2)$ | $\dfrac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu\in\mathbb R,\sigma>0$ | $\mu$ | $\sigma^2$ | 无初等闭式 |

**连续情形的关键观念**：
- $\Pr[X = x] = 0$ 对每个单点成立（**概率为 0 不等于不可能**）。
- $f(x)$ **不是概率**；$\Pr[X\in[x,x+dx]] \approx f(x)\,dx$。$f(x)$ **可以大于 1**。
- $\mathbb{E}[X] = \int x f(x)\,dx$；$\operatorname{Var}(X) = \int (x-\mu)^2 f(x)dx = \mathbb{E}[X^2] - \mu^2$。

**指数分布的无记忆性**（几何分布的连续类比）：
$$\Pr[X > s+t \mid X > s] = e^{-\lambda t} = \Pr[X > t].$$

**正态分布的 68–95–99.7 法则**：$\Pr[|X-\mu| \le \sigma] \approx 68\%$，$\le 2\sigma \approx 95\%$，$\le 3\sigma \approx 99.7\%$。

**独立正态之和**：$X\sim\mathcal N(\mu_1,\sigma_1^2)$，$Y\sim\mathcal N(\mu_2,\sigma_2^2)$ 独立 $\Rightarrow X+Y \sim \mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$。

#### 7.3 期望与方差的核心工具

**期望的定义**：$\mathbb{E}[X] = \sum_a a\Pr[X=a]$（离散）；$\mathbb{E}[X]=\int x f(x)dx$（连续）。

**期望的线性性（最重要的工具）**：
$$\mathbb{E}\left[\sum_i X_i\right] = \sum_i \mathbb{E}[X_i].$$
**⚠️ 不需要独立性！** 这是全课程最有用的"免费午餐"。

**LOTUS（无意识统计学家定律）**：$\mathbb{E}[g(X)] = \sum_x g(x)\Pr[X=x]$。

**乘积的期望**：$X \perp Y \Rightarrow \mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$。（**反向不成立**。）

**指示变量的威力**：$\mathbb{E}[\mathbb{1}_A] = \Pr[A]$。把复杂的"全局量"分解为简单的"局部指示变量之和"，是线性性应用的核心招式。

**三大经典应用**：

| 应用 | 分解方式 | 结果 |
|:---|:---|:---|
| **二项分布期望** | $X = \sum_{i=1}^n X_i$，$X_i \sim \mathrm{Ber}(p)$ | $\mathbb{E}[X] = np$ |
| **优惠券收集** | $T = \sum_{i=1}^n T_i$，$T_i \sim \mathrm{Geo}\!\left(\frac{n-i+1}{n}\right)$ | $\mathbb{E}[T] = nH_n \approx n\ln n$ |
| **哈希碰撞 / 空箱** | 对每对/每箱用指示变量 | 碰撞数 $\binom n2\frac1n$；空箱数 $n(1-\frac1n)^n \approx \frac ne$ |

**方差的定义与等价形式**：
$$\operatorname{Var}(X) = \mathbb{E}[(X-\mathbb{E}X)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2.$$

**方差的性质**：
$$\operatorname{Var}(aX+b) = a^2\operatorname{Var}(X), \qquad \operatorname{Var}(X+Y) \overset{X\perp Y}{=} \operatorname{Var}(X)+\operatorname{Var}(Y).$$

**⚠️ 对比记忆**：期望线性性**不需要**独立；方差可加性**需要**独立（或至少不相关）。

**协方差**：
$$\operatorname{Cov}(X,Y) = \mathbb{E}[(X-\mathbb{E}X)(Y-\mathbb{E}Y)] = \mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y].$$
- **双线性**：$\operatorname{Cov}(aX+bY,Z) = a\operatorname{Cov}(X,Z)+b\operatorname{Cov}(Y,Z)$；
- $\operatorname{Cov}(X,X) = \operatorname{Var}(X)$；
- **独立 $\Rightarrow \operatorname{Cov}=0$，但反之不成立**（反例：$X$ 均匀取 $\{-1,0,1\}$，$Y=X^2$；$\mathbb{E}[X]=0$，$\mathbb{E}[XY]=\mathbb{E}[X^3]=0$，故 $\operatorname{Cov}=0$，但 $Y$ 完全由 $X$ 决定）。

**一般方差公式**：
$$\operatorname{Var}\left(\sum_i X_i\right) = \sum_i \operatorname{Var}(X_i) + 2\sum_{i<j}\operatorname{Cov}(X_i,X_j).$$

**相关系数**：$\rho_{X,Y} = \dfrac{\operatorname{Cov}(X,Y)}{\sigma_X\sigma_Y} \in [-1,1]$，度量**线性**相关程度。

#### 7.4 联合分布与独立性

**联合 PMF**：$p_{X,Y}(a,b) = \Pr[X=a, Y=b]$；$\sum_{a,b}p_{X,Y}(a,b) = 1$。
**边缘 PMF**：$p_X(a) = \sum_b p_{X,Y}(a,b)$。
**⚠️** 边缘分布**不能**还原联合分布（信息丢失）。

**随机变量独立**：$X \perp Y \iff \Pr[X=a,Y=b]=\Pr[X=a]\Pr[Y=b]$ 对所有 $a,b$。

**独立和的卷积**：$Z = X+Y$ 时 $p_Z(z) = \sum_x p_X(x)p_Y(z-x)$。

**闭包性质**：
- $\mathrm{Bin}(n,p) + \mathrm{Bin}(m,p) \sim \mathrm{Bin}(n+m,p)$；
- $\mathrm{Pois}(\lambda_1)+\mathrm{Pois}(\lambda_2) \sim \mathrm{Pois}(\lambda_1+\lambda_2)$；
- $\mathcal N(\mu_1,\sigma_1^2)+\mathcal N(\mu_2,\sigma_2^2) \sim \mathcal N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$。

#### 7.5 集中不等式

**马尔可夫不等式**：$X \ge 0$，$t>0$ $\Rightarrow$
$$\Pr[X \ge t] \le \frac{\mathbb{E}[X]}{t}.$$
*证明*：对每个 $\omega$，$t\,\mathbb{1}_{X\ge t}(\omega) \le X(\omega)$；取期望得 $t\Pr[X\ge t] \le \mathbb{E}[X]$。

**切比雪夫不等式**：
$$\Pr[|X - \mathbb{E}[X]| \ge t] \le \frac{\operatorname{Var}(X)}{t^2}.$$
*证明*：对 $(X-\mu)^2 \ge 0$ 应用马尔可夫不等式，取 $t^2$。

**切尔诺夫界**（$X = \sum_{i=1}^n X_i$，独立伯努利，$\mu=\mathbb{E}[X]$）：
$$\Pr[X \ge (1+\delta)\mu] \le \left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^{\mu} \le e^{-\mu\delta^2/3} \quad (0<\delta<1),$$
$$\Pr[X \le (1-\delta)\mu] \le e^{-\mu\delta^2/2} \quad (0<\delta<1).$$
*证明机制*：对任意 $t>0$，$\Pr[X\ge a] = \Pr[e^{tX} \ge e^{ta}] \le e^{-ta}\mathbb{E}[e^{tX}]$（马尔可夫）；独立性使 $\mathbb{E}[e^{tX}] = \prod_i\mathbb{E}[e^{tX_i}]$；最后对 $t$ 优化。

**三个界的强度对比**（$n=1000$ 次公平硬币，$\Pr[X \ge 600]$，即 $\mu=500$、$\delta=0.2$）：

| 方法 | 上界数值 | 说明 |
|:---|:---|:---|
| **真实值** | $1.36 \times 10^{-10}$ | 用二项分布精确计算 |
| **切比雪夫** | $\dfrac{\operatorname{Var}(X)}{t^2} = \dfrac{250}{100^2} = 0.025$ | 用了方差，比马尔可夫好但仍很松 |
| **切尔诺夫（紧形式）** | $\left(\frac{e^{0.2}}{1.2^{1.2}}\right)^{500} \approx 8.3\times 10^{-5}$ | 指数级衰减 |
| **切尔诺夫（简化形式）** | $e^{-500 \cdot 0.04/3} \approx 1.3\times 10^{-3}$ | 更松但形式简单，常用 |
| **马尔可夫** | $\dfrac{500}{600} \approx 0.83$ | 只用了期望，几乎无用 |

定性结论：**切尔诺夫 $\gg$ 切比雪夫 $\gg$ 马尔可夫**——用的信息越多（期望 → 方差 → 全部分布结构），界越紧，衰减从多项式变成指数。

> **⚠️ 不要背成"切尔诺夫任何时候都最强"**：上表是 $n=1000$ 的大样本情形。当 $n$ 较小而 $\delta$ 不小（如 $n=100,\delta=0.2$）时，切比雪夫给出 $0.25$，反而**优于**切尔诺夫的 $0.39$。切尔诺夫的优势是**渐近的**：它让上界随 $n$ **指数**衰减，而切比雪夫只让上界按 $1/n$ 多项式衰减。选界时要看具体参数，而非套用排名。

**弱大数定律（WLLN）**：$X_1,\dots,X_n$ 独立同分布，$\mathbb{E}[X_i]=\mu$，$\operatorname{Var}(X_i)=\sigma^2<\infty$，则对任意 $\epsilon>0$：
$$\Pr\left[\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| \ge \epsilon\right] \le \frac{\sigma^2}{n\epsilon^2} \xrightarrow{n\to\infty} 0.$$
*证明*：$\mathbb{E}[\bar X_n] = \mu$（线性性），$\operatorname{Var}(\bar X_n) = \sigma^2/n$（独立性 + 方差可加），代入切比雪夫。

**样本复杂度（估计硬币偏差）**：为使 $\Pr[|\hat p - p| \ge \epsilon] \le \delta$，切比雪夫给出 $n \ge \frac{1}{4\epsilon^2\delta}$；切尔诺夫给出 $n = O\!\left(\frac{\log(1/\delta)}{\epsilon^2}\right)$（对 $\delta$ 是指数级更优）。

#### 7.6 中心极限定理

**CLT**：$X_1,X_2,\dots$ 独立同分布，$\mathbb{E}[X_i]=\mu$，$\operatorname{Var}(X_i)=\sigma^2<\infty$，则
$$\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt n} \xrightarrow{\ d\ } \mathcal N(0,1),$$
即 $\bar X_n \approx \mathcal N\!\left(\mu, \frac{\sigma^2}{n}\right)$（$n$ 充分大时）。

**与 LLN 的对比**：LLN 说"均值收敛到 $\mu$"；CLT 进一步说"波动的**形状**是正态的，尺度是 $\sigma/\sqrt n$"。

**CLT 的适用条件与陷阱**：
- 需要**有限方差**（Cauchy 分布无方差，CLT 失效）；
- 需要**独立同分布**（或满足一定条件的弱相关），强相关时失效；
- 是**渐进**结论，$n$ 不够大时近似很差（原始分布偏斜时尤甚）；
- 二项的正态近似经验判据：$np \ge 10$ 且 $n(1-p)\ge10$；
- **不能**把 CLT 用于"数据本身的分布"——趋近正态的是**均值的分布**。

**置信区间**：$\bar X \pm z_{\alpha/2}\frac{\sigma}{\sqrt n}$；95% 对应 $z \approx 1.96$。正确解读：**重复抽样时，约 95% 的区间会覆盖真值**（而不是"真值有 95% 概率落在本次区间内"）。

#### 7.7 马尔可夫链与条件期望

**马尔可夫性质（无记忆性）**：
$$\Pr[X_{t+1}=j \mid X_t=i, X_{t-1},\dots,X_0] = \Pr[X_{t+1}=j\mid X_t=i] = P_{ij}.$$

**转移矩阵 $P$**：$P_{ij} \ge 0$，$\sum_j P_{ij} = 1$（行随机）。$t$ 步转移：$P^{(t)} = P^t$。

**状态分类**：
- **可达 / 互通 / 不可约（irreducible）**：所有状态两两互通；
- **常返（recurrent）vs 瞬态（transient）**；
- **周期（periodic）vs 非周期（aperiodic）**。

**平稳分布（stationary distribution）**：$\pi P = \pi$，$\sum_i \pi_i = 1$。
**存在唯一性**：有限、不可约、非周期的链有唯一平稳分布，且从任意初始分布收敛到它。

**求解示例（2 状态）**：$P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6\end{pmatrix}$，由 $\pi_1 = 0.7\pi_1 + 0.4\pi_2$ 与 $\pi_1+\pi_2=1$ 解得 $\pi = \left(\frac47, \frac37\right)$。

**首步分析（first-step analysis）**：设 $h_i$ 为从状态 $i$ 出发的某目标概率（或期望量），则
$$h_i = \sum_j P_{ij} h_j \quad (\text{配边界条件}), \qquad t_i = 1 + \sum_j P_{ij}t_j \quad (\text{期望步数}).$$

**赌徒破产问题**：每次以概率 $p$ 赢 1 元、$q=1-p$ 输 1 元；起始 $i$ 元，目标 $N$ 元。最终赢到 $N$ 元的概率
$$h_i = \begin{cases} \dfrac{1 - (q/p)^i}{1 - (q/p)^N} & p \ne \frac12 \\[6pt] \dfrac{i}{N} & p = \frac12 \end{cases}$$
（递推 $h_i = p h_{i+1} + q h_{i-1}$，边界 $h_0 = 0$，$h_N = 1$。）

**条件期望**：$\mathbb{E}[X\mid Y=y] = \sum_x x\Pr[X=x\mid Y=y]$；$\mathbb{E}[X\mid Y]$ 是**随机变量**。

**全期望公式（重期望 / 塔性质）**：
$$\mathbb{E}[X] = \sum_y \mathbb{E}[X\mid Y=y]\Pr[Y=y] = \mathbb{E}\big[\mathbb{E}[X\mid Y]\big].$$
*证明*：
$$\sum_y \mathbb{E}[X\mid Y=y]\Pr[Y=y] = \sum_y \sum_x x\frac{\Pr[X=x,Y=y]}{\Pr[Y=y]}\Pr[Y=y] = \sum_x x\sum_y \Pr[X=x,Y=y] = \mathbb{E}[X].$$

**条件期望的性质**：线性性；$\mathbb{E}[g(Y)X \mid Y] = g(Y)\mathbb{E}[X\mid Y]$（"把已知的提出来"）。

**应用**：条件期望是计算复杂期望的"分而治之"利器——先按某个变量分层，逐层计算，再按各层概率加权。典型用于随机和、几何分布期望、优惠券收集、首步分析。

---

### 八、跨讲次的高频陷阱总清单

1. **逆命题 ≠ 原命题**（L0–L1）：$P\Rightarrow Q$ 与 $Q \Rightarrow P$ 无关。**逆否**才是等价的。
2. **归纳法漏掉基础情形**（L3）：只有归纳步骤不能推出任何结论。
3. **归纳步骤的隐蔽漏洞**（L3）：如"所有马同色"伪证中两个子集交集为空。
4. **模运算中乱用除法**（L4）：$ab \equiv ac \pmod m \not\Rightarrow b\equiv c$，除非 $\gcd(a,m)=1$。
5. **逆元存在条件**（L4–L5）：$x$ 模 $m$ 可逆 $\iff \gcd(x,m)=1$；$0$ 永远不可逆。
6. **FLT 的条件**（L5）：只对**质数** $p$ 且 $a \not\equiv 0 \pmod p$ 成立。
7. **多项式"函数相等"≠"多项式相等"**（L7）：有限域上不同多项式可定义同一函数（$x^p$ vs $x$）。
8. **$E \le 3V-6$ 不充分**（L10）：满足不等式不保证平面。
9. **真子集可以等势**（L12）：$\mathbb{N} \subsetneq \mathbb{Z}$ 但 $|\mathbb{N}|=|\mathbb{Z}|$。"可数"不等于"有限"。
10. **$\Pr[A\cup B] = \Pr[A]+\Pr[B]$ 需互斥**（L15）：不互斥时必须减去 $\Pr[A\cap B]$。
11. **结果不等可能时不能用 $|A|/|\Omega|$**（L15）：掷两枚硬币的"1 个正面"。
12. **混淆 $\Pr[A\mid B]$ 与 $\Pr[B\mid A]$**（L17）：疾病检测算例差十倍以上（检察官谬误）。
13. **不交 ≠ 独立**（L18）：不交且概率为正 $\Rightarrow$ **不**独立。
14. **两两独立 ≠ 相互独立**（L18）：两枚硬币 + "两枚相同"的经典反例。
15. **独立 ≠ 条件独立**（L18）：两个方向都不成立。
16. **期望线性性不需要独立，方差可加需要独立**（L20–L22）：最常见的对比记忆点。
17. **$\operatorname{Var}(cX) = c^2\operatorname{Var}(X)$**（L22）：不是 $c$。
18. **$\operatorname{Cov}=0 \not\Rightarrow$ 独立**（L22）：$Y=X^2$ 但 $X$ 对称。
19. **$\mathbb{E}[XY]=\mathbb{E}[X]\mathbb{E}[Y] \not\Rightarrow$ 独立**（L20）。
20. **连续情形 PDF 不是概率**（L24）：$f(x)$ 可以大于 1；$\Pr[X=x]=0$ 不代表不可能。
21. **CLT 需要有限方差与（近似）独立同分布**（L25）：Cauchy 分布下失效；且 CLT 说的是**均值**趋近正态，不是数据本身。
22. **首步分析别忘边界条件**（L26）：没有边界条件的递推方程有无穷多解。
23. **平稳分布存在 ≠ 唯一**（L26）：可约链可能有无穷多平稳分布；周期链不收敛到平稳分布。
24. **停机问题不可判定 ≠ 无法判断任何程序**（L13）：不存在**统一算法**，具体程序常常很容易判断。

{% endraw %}
