---
title: "RL 算法速查表"
collection: course-notes
chapter: true
permalink: /course-notes/stanford-cs234-reinforcement-learning/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/stanford-cs234-reinforcement-learning/) · [← l16](/course-notes/stanford-cs234-reinforcement-learning/l16)

{% raw %}
## RL 算法速查表

> 本表汇总 CS234 Winter 2026 全课程出现的关键算法。**上标 Lx 表示该算法在正文中的讲次**。
> 记号：$S=\\vert \\mathcal{S}\\vert $，$A=\\vert \\mathcal{A}\\vert $，$T$ 为步数/回合数，$\\epsilon$ 为精度，$\\delta$ 为失败概率，$\\gamma$ 为折扣因子，$H$ 为时域。

---

## 一、规划 / 动态规划（有模型，已知 $P$ 与 $R$）

| 算法 | 讲次 | 核心更新 | 计算复杂度（每轮） | 收敛性 / 保证 | 适用场景 |
|:--|:--|:--|:--|:--|:--|
| **策略评估（迭代版）** | L2 | $V_{k+1}(s)=\\sum_a \\pi(a\\vert s)\\big[R(s,a)+\\gamma\\sum_{s^{\\prime}}P(s^{\\prime}\\vert s,a)V_k(s^{\\prime})\\big]$ | $O(S^2A)$ | $V_k\\to V^\\pi$ 几何收敛，速率 $\\gamma$ | 已知模型、给定策略 |
| **策略评估（解析解）** | L2 | $V^\\pi=(I-\\gamma P^\\pi)^{-1}R^\\pi$ | $O(S^3)$ | 精确解（$S$ 小时可用） | 小型 MDP、验证迭代结果 |
| **策略迭代（PI）** | L2 | 评估到收敛 → 贪婪改进 $\\pi^{\\prime}(s)=\\arg\\max_a Q^\\pi(s,a)$ | $O(S^2A)$ / 轮 | **单调改进**；有限 MDP 中**有限步**收敛到 $\\pi^*$ | 已知模型、$S$ 不大 |
| **值迭代（VI）** | L2 | $V_{k+1}(s)=\\max_a\\big[R(s,a)+\\gamma\\sum_{s^{\\prime}}P(s^{\\prime}\\vert s,a)V_k(s^{\\prime})\\big]$ | $O(S^2A)$ | Bellman 算子是 $\\gamma$-压缩 ⇒ 唯一不动点 $V^*$；迭代次数 $O\\!\\big(\\frac{\\log(1/(\\epsilon(1-\\gamma)))}{1-\\gamma}\\big)$ 量级 | 已知模型、需要最优值函数 |
| **有限时域 VI（H 步）** | L2 | 从 $t=H$ 倒推，$V_H(s)=0$，$V_t(s)=\\max_a[R+\\gamma\\sum P V_{t+1}]$ | $O(HS^2A)$ | 精确最优（非平稳策略） | 有限时域问题 |

**黄金法则**：VI 与 PI 都收敛到同一个 $V^*$（在表格型、$\\gamma<1$ 或适当条件下），但 **PI 通常迭代次数更少而每次迭代更贵**；VI 每次迭代便宜但可能需要更多次。$\\vert A\\vert \\vert S\\vert $ 是 VI 迭代次数的常见上界量级。

---

## 二、值函数方法（无模型）

### 2.1 策略评估（预测）

| 算法 | 讲次 | 更新规则 | 偏差 | 方差 | 收敛性 |
|:--|:--|:--|:--|:--|:--|
| **首次/每次访问 MC** | L3 | $V(s_t)\\leftarrow V(s_t)+\\alpha\\big(G_t-V(s_t)\\big)$ | **无偏** | **高**（整条轨迹的随机性） | $\\alpha$ 满足 Robbins-Monro 时 a.s. 收敛到 $V^\\pi$ |
| **TD(0)** | L3 | $V(s_t)\\leftarrow V(s_t)+\\alpha\\big(r_t+\\gamma V(s_{t+1})-V(s_t)\\big)$ | 有偏（自举） | **低** | 步长条件下 a.s. 收敛到 $V^\\pi$ |
| **n-step TD** | L3/L5 | 目标 $=r_t+\\gamma r_{t+1}+\\dots+\\gamma^{n-1}r_{t+n-1}+\\gamma^n V(s_{t+n})$ | $n$ 越大偏差越小 | $n$ 越大方差越大 | 同 TD(0) |
| **TD($\\lambda$) / 资格迹** | L3 | 目标 $=G_t^\\lambda=(1-\\lambda)\\sum_{n\\ge1}\\lambda^{n-1}G_t^{(n)}$ | $\\lambda\\in[0,1]$ 连续插值 | 同左 | 收敛到 $V^\\pi$ |
| **批处理 MC** | L3 | 最小化 $\\sum_k\\sum_t(G_t^k-V(s_t^k))^2$ | — | — | 收敛到**训练数据上的最小二乘解** |
| **批处理 TD** | L3 | 最小化 $\\sum_k\\sum_t(r+\\gamma V(s^{\\prime})-V(s))^2$ | — | — | 收敛到**确定性等价（certainty-equivalence）** MDP 的解（即 ML/MRP 的 MLE 解） |

> **AB 例子（SB Ex. 6.4）**：8 条轨迹下，MC 给出 $\\hat V(A)=0$，TD 给出 $\\hat V(A)=0.75$。TD 利用了 Markov 结构，MC 没有——这是"TD 在批处理下更高效"的经典证据。

### 2.2 控制（无模型）

| 算法 | 讲次 | on/off-policy | 更新规则 | 收敛条件 | 备注 |
|:--|:--|:--|:--|:--|:--|
| **GLIE MC 控制** | L4 | on-policy | 用 $G_t$ 更新 $Q$，$\\epsilon$-贪婪改进 | GLIE：$\\epsilon_t\\to0$ 且每个 $(s,a)$ 访问无限次 + 步长条件 | 高方差，需完整回合 |
| **SARSA** | L4 | **on-policy** | $Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma Q(s^{\\prime},a^{\\prime})-Q(s,a)\\big]$ | GLIE + 步长条件 | 学到的是**当前 $\\epsilon$-贪婪策略**的价值；悬崖行走中更保守 |
| **Q-learning** | L4 | **off-policy** | $Q(s,a)\\leftarrow Q(s,a)+\\alpha\\big[r+\\gamma\\max_{a^{\\prime}}Q(s^{\\prime},a^{\\prime})-Q(s,a)\\big]$ | 所有 $(s,a)$ 无限次访问 + $\\sum\\alpha=\\infty,\\sum\\alpha^2<\\infty$ | 直接学 $Q^*$；悬崖行走中走最优路径 |
| **Expected SARSA** | L4 | off-policy 变体 | 目标 $=r+\\gamma\\sum_{a^{\\prime}}\\pi(a^{\\prime}\\vert s^{\\prime})Q(s^{\\prime},a^{\\prime})$ | 同 SARSA | 方差低于 SARSA，可利用行为策略数据 |
| **Double Q-learning** | L4 | off-policy | 用两个 $Q$ 表，一个选动作、一个评估 | 同上 | 修正 $\\max$ 带来的**最大化偏差** |

**黄金法则**：`max` 与 `Q(s',a')` 的差别就是 **off-policy 与 on-policy 的分水岭**。

### 2.3 函数逼近与深度 RL

| 算法 | 讲次 | 目标 / 损失 | 稳定性机制 | 理论保证 |
|:--|:--|:--|:--|:--|
| **VFA 策略评估（MC 目标）** | L4 | $\\Delta w=\\alpha\\big(G_t-\\hat V(s_t;w)\\big)\\nabla_w\\hat V(s_t;w)$ | 真梯度（对参数） | 线性情形下收敛到局部最优 |
| **VFA 策略评估（TD 目标）** | L4 | $\\Delta w=\\alpha\\big(r+\\gamma\\hat V(s_{t+1};w)-\\hat V(s_t;w)\\big)\\nabla_w\\hat V(s_t;w)$ | **半梯度** | 线性 + on-policy 下收敛；off-policy/非线性可能发散 |
| **SARSA + VFA** | L4 | $\\Delta w=\\alpha\\big(r+\\gamma\\hat Q(s^{\\prime},a^{\\prime};w)-\\hat Q(s,a;w)\\big)\\nabla_w\\hat Q$ | 半梯度 | 同上 |
| **Q-learning + VFA** | L4 | $\\Delta w=\\alpha\\big(r+\\gamma\\max_{a^{\\prime}}\\hat Q(s^{\\prime},a^{\\prime};w)-\\hat Q(s,a;w)\\big)\\nabla_w\\hat Q$ | 半梯度 | **致命三要素**下可能发散 |
| **DQN** | L4 | $L(w)=\\mathbb{E}\\big[\\big(r+\\gamma\\max_{a^{\\prime}}\\hat Q(s^{\\prime},a^{\\prime};w^-)-\\hat Q(s,a;w)\\big)^2\\big]$ | ① **经验回放** ② **固定 Q 目标网络** $w^-$ | 无一般收敛保证；实践中在 Atari 上成功 |
| **Double DQN** | L4 | 目标中 $a^*=\\arg\\max_{a^{\\prime}}\\hat Q(s^{\\prime},a^{\\prime};w)$，用 $\\hat Q(s^{\\prime},a^*;w^-)$ 评估 | 解耦选择与评估 | 减轻过估计 |
| **优先经验回放（PER）** | L4 | 按 TD 误差的优先级采样 | 更频繁重放"惊讶"的样本 | — |
| **Dueling 网络** | L4 | $\\hat Q(s,a)=V(s)+A(s,a)-\\frac{1}{\\vert A\\vert }\\sum_{a^{\\prime}}A(s,a^{\\prime})$ | 分离状态价值与优势 | — |

> **致命三要素（Deadly Triad）** = **函数逼近 + 自举 + off-policy**。三者同时出现时，价值估计可能发散。DQN 的两个机制正是为了拆解这个三角。

---

## 三、策略梯度方法

| 算法 | 讲次 | 梯度/目标 | Critic | 偏差-方差 | 备注 |
|:--|:--|:--|:--|:--|:--|
| **REINFORCE（MC PG）** | L5 | $\\nabla_\\theta J=\\mathbb{E}\\big[\\sum_t\\nabla_\\theta\\log\\pi_\\theta(a_t\\vert s_t)G_t\\big]$ | 无 | 无偏、**高方差** | 需完整回合；步长敏感 |
| **REINFORCE + 基线** | L5/L6 | 用 $G_t-b(s_t)$ 代替 $G_t$ | $b(s)$ 常取 $\\hat V(s)$ | 仍无偏，方差**降低** | $b$ 只依赖状态即可保无偏 |
| **Actor-Critic** | L5/L6 | $\\nabla_\\theta J\\approx\\mathbb{E}\\big[\\nabla_\\theta\\log\\pi_\\theta(a\\vert s)\\hat Q_w(s,a)\\big]$ | 用参数化 $\\hat Q_w$ / $\\hat V_w$ | 引入偏差、**大幅降方差** | 可在线单步更新 |
| **A2C / A3C** | L5/L6 | 用优势 $\\hat A=\\hat Q_w-\\hat V_w$（或 TD 残差） | $\\hat V_w$ | 平衡偏差方差 | A3C 用异步并行 |
| **GAE** | L7 | $\\hat A_t^{GAE(\\gamma,\\lambda)}=\\sum_{l\\ge0}(\\gamma\\lambda)^l\\delta_{t+l}$，$\\delta_t=r_t+\\gamma V(s_{t+1})-V(s_t)$ | $\\hat V_w$ | $\\lambda\\in[0,1]$ 连续插值（$\\lambda=0$：低方差高偏差；$\\lambda=1$：MC） | PPO 的标配优势估计 |
| **TRPO** | L7 | $\\max_\\theta L_{\\theta_k}(\\theta)$ s.t. $\\bar D_{KL}(\\theta\\vert \\theta_k)\\le\\delta$ | $\\hat V_w$ 或 $\\hat A_w$ | — | 单调改进有理论保证；需 Fisher 矩阵逆（自然梯度），实现昂贵 |
| **PPO（KL 惩罚版）** | L6 | $\\max_\\theta L_{\\theta_k}(\\theta)-\\beta_k\\bar D_{KL}(\\theta\\vert \\theta_k)$ | $\\hat V_w$ | — | $\\beta_k$ 自适应调整 |
| **PPO（裁剪版）** | L6 | $L^{CLIP}=\\mathbb{E}\\big[\\min\\big(r_t(\\theta)\\hat A_t,\\ \\mathrm{clip}(r_t(\\theta),1-\\epsilon,1+\\epsilon)\\hat A_t\\big)\\big]$，$r_t=\\frac{\\pi_\\theta}{\\pi_{\\theta_{old}}}$ | $\\hat V_w$ | — | **实践中最常用的 RL 策略优化算法**；实现简单、稳定 |

**策略梯度定理**（L5）：

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\Big[\sum_{t=0}^{\infty}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\,Q^{\pi_\theta}(s_t,a_t)\Big]$$

**性能差异引理 / 单调改进**（L7）：

$$J(\theta^{\prime})-J(\theta)=\frac{1}{1-\gamma}\,\mathbb{E}_{s\sim d^{\pi_{\theta^{\prime}}}}\Big[\mathbb{E}_{a\sim\pi_{\theta^{\prime}}}\big[A^{\pi_\theta}(s,a)\big]\Big]$$

$$J(\theta^{\prime})\ \ge\ L_\theta(\theta^{\prime})-\frac{2\gamma\epsilon}{(1-\gamma)^2}\,\bar D_{KL}(\theta^{\prime}\vert \theta),\qquad \epsilon=\max_s\big\vert \mathbb{E}_{a\sim\pi_{\theta^{\prime}}}[A^{\pi_\theta}(s,a)]\big\vert $$

---

## 四、数据高效 RL 与探索

### 4.1 多臂老虎机（MAB）

| 算法 | 讲次 | 选择规则 | Regret 量级 | 框架 |
|:--|:--|:--|:--|:--|
| **贪心（greedy）** | L9 | $a_t=\\arg\\max_a\\hat Q_t(a)$ | **线性** $O(T)$（确定性/并列时甚至更糟） | 频率派 regret |
| **$\\epsilon$-贪婪** | L9 | 以 $1-\\epsilon$ 贪婪、以 $\\epsilon$ 随机 | **线性** $O(\\epsilon T)$（$\\epsilon$ 固定时） | 频率派 regret |
| **UCB1** | L9/L10 | $a_t=\\arg\\max_a\\big(\\hat Q_t(a)+\\sqrt{2\\ln t/n_t(a)}\\big)$ | $O(\\sqrt{AT\\ln T})$ | 频率派 regret |
| **Thompson 采样（TS）** | L11 | 从每个臂的后验采样 $\\tilde\\theta_a$，取 $\\arg\\max_a\\tilde\\theta_a$ | 贝叶斯 regret 界；实践中常优于 UCB | **贝叶斯 regret** |
| **Gittins index** | L11 | 为每个臂算 Gittins 指数，选最大者 | 贝叶斯 bandit 的**最优**策略 | 贝叶斯（需已知先验 + 特定结构） |

**Regret 定义**：$\\displaystyle R(T)=\\sum_{t=1}^T\\big(\\mu^*-\\mu_{a_t}\\big)=\\sum_{a}\\Delta_a\\,n_T(a)$，其中 $\\Delta_a=\\mu^*-\\mu_a$ 是 gap。

**Hoeffding 不等式**（UCB 的基础）：$P\\big(\\big\\vert \\hat\\mu_a-\\mu_a\\big\\vert \\ge\\epsilon\\big)\\le 2e^{-2n\\epsilon^2}$。

### 4.2 MDP 中的探索（样本高效 RL）

| 算法 | 讲次 | 核心思想 | 保证形式 | 复杂度量级 |
|:--|:--|:--|:--|:--|
| **RMax** | L11/L12 | 未见过的 $(s,a)$ 赋乐观奖励 $R_{max}$；采样 $m$ 次后建经验模型求最优策略 | **PAC-MDP** | 关于 $S,A,1/\\epsilon,1/\\delta$ 多项式 |
| **MBIE-EB** | L12 | 用置信区间 + 探索奖励 $R^+(s,a)=\\beta/\\sqrt{n(s,a)}$ | **PAC-MDP** | 多项式 |
| **PSRL（后验采样）** | L12 | 每回合从 MDP 后验采样一个模型，对其求最优策略并执行整回合 | **贝叶斯 regret** | $\\tilde O(HS\\sqrt{AT})$ 量级 |
| **Delayed Q-learning** | L12 | 用延迟更新实现乐观 | **PAC-MDP** | 多项式 |
| **计数奖励 / 伪计数** | L12 | $R^+(s,a)=\\beta/n(s,a)$；用密度模型生成伪计数 | 无一般界 | 实践有效（Montezuma's Revenge） |
| **内在动机 / 信息增益** | L12 | 用预测误差、新奇性、信息增益作为内在奖励 | 部分有界 | 实践有效 |
| **元学习探索（DREAM / DPT）** | L12 | 在多任务分布上学习"如何探索" | 经验性 | 跨任务迁移 |

**PAC-MDP 定义**：对给定 $\\epsilon,\\delta$，算法 $\\mathcal{A}$ 是 PAC 的，如果以概率 $\\ge1-\\delta$，其产生**非 $\\epsilon$-最优**动作的步数被关于 $S,A,1/\\epsilon,1/\\delta$ 的多项式所界。

> **Simulation Lemma**：若 $\\hat P,\\hat R$ 与真值的误差为 $\\epsilon_P,\\epsilon_R$，则对任意策略 $\\pi$，
> $$\big\vert V^\pi_{\hat M}-V^\pi_M\big\vert _\infty\ \le\ \frac{\epsilon_R}{1-\gamma}+\frac{\gamma R_{max}\,\epsilon_P}{2(1-\gamma)^2}$$
> 这是把"模型误差"翻译成"价值误差"的关键工具。

---

## 五、模仿学习与偏好学习

| 算法 | 讲次 | 目标 | 数据需求 | 保证 |
|:--|:--|:--|:--|:--|
| **行为克隆（BC）** | L7/L8 | $\\min_\\theta\\mathbb{E}_{(s,a)\\sim\\mathcal{D}}\\big[-\\log\\pi_\\theta(a\\vert s)\\big]$ | 专家示范，**离线** | 复合误差 $O(\\epsilon T^2)$ |
| **DAgger** | L7/L8 | 迭代：执行 $\\pi_\\theta$ → 请专家标注访问到的状态 → 聚合数据集 → 重训 | 可**交互**询问专家 | $O(\\epsilon T)$（无 regret 的形式更优） |
| **线性特征 IRL** | L7 | 从示范恢复 $R(s)=\\phi(s)^\\top w$；特征匹配 $\\mu_{\\pi_E}=\\mu_{\\pi}$ | 专家示范 | 存在**歧义性**（多个奖励对应同一策略） |
| **MaxEnt IRL** | L7 | $\\max_w\\ \\mathbb{E}_{\\tau\\sim\\mathcal{D}}\\big[\\log P(\\tau\\vert w)\\big]$，$P(\\tau\\vert w)\\propto e^{\\sum_t\\phi(s_t,a_t)^\\top w}$ | 专家示范 | 解决歧义；梯度 = 特征期望之差 |
| **RLHF（奖励模型）** | L8 | Bradley–Terry：$P(y_w\\succ y_l)=\\sigma\\big(r_\\phi(x,y_w)-r_\\phi(x,y_l)\\big)$ | **成对偏好比较** | 需假设偏好由标量奖励生成 |
| **RLHF（PPO 微调）** | L8 | $\\max_\\theta\\mathbb{E}\\big[r_\\phi(x,y)-\\beta\\,\\mathrm{KL}(\\pi_\\theta\\vert \\pi_{ref})\\big]$ | 偏好数据 + 在线采样 | 实践中有效；流水线复杂 |
| **DPO** | L8 | $-\\mathbb{E}\\big[\\log\\sigma\\big(\\beta\\log\\frac{\\pi_\\theta(y_w)}{\\pi_{ref}(y_w)}-\\beta\\log\\frac{\\pi_\\theta(y_l)}{\\pi_{ref}(y_l)}\\big)\\big]$ | **只需偏好数据** | 与 RLHF 目标**等价**（在 Bradley–Terry + 最优策略闭式解下）；无需 RM、无在线采样 |

**DPO 的核心推导链**：KL 约束最优策略闭式解 $\\pi^*(y\\vert x)=\\frac{1}{Z(x)}\\pi_{ref}(y\\vert x)e^{r(x,y)/\\beta}$ → 反解 $r(x,y)=\\beta\\log\\frac{\\pi_\\theta(y\\vert x)}{\\pi_{ref}(y\\vert x)}+\\beta\\log Z(x)$ → 代入 Bradley–Terry，**配分函数 $Z(x)$ 配对抵消** → 得到 DPO 损失。

---

## 六、基于模拟的搜索

| 算法 | 讲次 | 核心机制 | 复杂度 | 保证 |
|:--|:--|:--|:--|:--|
| **简单 MC 搜索** | L13 | 每个候选动作 roll-out $K$ 次，取平均回报最大者 | $O(K\\cdot A\\cdot \\text{depth})$ | 估计无偏；动作多时昂贵 |
| **MCTS（UCT）** | L13 | 四阶段：选择（UCT）→ 扩展 → 模拟 → 回传；$a^*=\\arg\\max_a\\big(Q(s,a)+c\\sqrt{\\frac{\\ln N(s)}{N(s,a)}}\\big)$ | anytime，可并行 | 访问次数 $\\to\\infty$ 时 $Q(s,a)\\to q_*(s,a)$（minimax 最优） |
| **AlphaGo / AlphaZero** | L13/L14 | PUCT：$a^*=\\arg\\max_a\\big(Q(s,a)+c\\,P(s,a)\\frac{\\sqrt{N(s)}}{1+N(s,a)}\\big)$ + 策略/价值网络 + 自我对弈 | 大规模并行 | 经验上超人类；MCTS 作为策略改进算子 |
| **MuZero** | L14 | 学习隐式动力学模型 + 预测奖励/价值/策略，在隐空间做 MCTS | — | 无需已知规则 |

**MCTS 的五大优势**（讲义原文）：高度选择性的最佳优先搜索；动态评估状态；用采样打破维数灾难；适用于黑箱模型（只需采样）；计算高效、随时可停（anytime）、可并行。

---

## 七、按「评估框架」分类速查

| 框架 | 衡量什么 | 典型算法 | 保证形式 |
|:--|:--|:--|:--|
| **经验性能** | 多随机种子上的平均回报/学习曲线 | 全部 | 无理论保证（但最贴近实践） |
| **渐近收敛** | $V_t\\to V^*$ 或 $Q_t\\to Q^*$ 几乎必然 | TD(0)、SARSA、Q-learning（表格型） | a.s. 收敛（需步长 + 遍历性条件） |
| **Regret（频率派）** | 相对于最优策略的累积机会损失 | UCB1、MCTS/UCT | $O(\\sqrt{AT\\ln T})$；下界 $\\Omega(\\sqrt{AT})$ |
| **贝叶斯 Regret** | Regret 对**先验**取期望 | Thompson 采样、PSRL、Gittins | $\\tilde O(\\sqrt{T})$ 量级；可利用先验 |
| **PAC / 样本复杂度** | 达到 $\\epsilon$-最优所需步数（高概率） | RMax、MBIE-EB、Delayed Q-learning | 关于 $S,A,1/\\epsilon,1/\\delta$ 的多项式 |
| **计算复杂度** | 每步/每轮的计算开销 | VI/PI（$O(S^2A)$）、策略梯度 | 与样本复杂度往往此消彼长 |

---

## 八、超参数速查与调参直觉

| 超参数 | 作用 | 太小 | 太大 |
|:--|:--|:--|:--|
| $\\alpha$（学习率） | 更新步长 | 收敛慢 | 震荡/发散 |
| $\\gamma$（折扣因子） | 未来奖励的权重 | 短视，只顾眼前 | 有效时域长，方差大，收敛慢（$1/(1-\\gamma)$ 爆炸） |
| $\\epsilon$（探索率） | $\\epsilon$-贪婪的探索概率 | 陷入次优（未探索的动作永不更新） | 退化为随机策略 |
| $\\lambda$（GAE/资格迹） | 偏差-方差插值 | $\\lambda=0$：低方差高偏差（TD 残差） | $\\lambda=1$：无偏高方差（MC） |
| $\\epsilon_{clip}$（PPO 裁剪） | 单次策略更新幅度上限 | 学习过慢 | 失去"近端"约束的意义，可能崩溃 |
| $\\delta$（TRPO/KL 约束） | 信任域半径 | 学习过慢 | 破坏单调改进保证 |
| $c$（UCT 探索常数） | 搜索中探索 vs 利用 | 陷入局部（反复访问同一分支） | 退化为近似均匀搜索 |
| $m$（RMax 采样阈值） | 判定"已知"的样本数 | 模型不可靠 → 策略次优 | 探索代价过高 |
| $\\beta$（KL 惩罚/DPO） | 相对参考策略的偏离惩罚 | 分布漂移、reward hacking | 几乎不偏离参考策略，学不到新行为 |

---

## 九、方法选择决策树

```
问题是否可形式化为 MDP？
├─ 否 → 考虑上下文老虎机（单步决策 + 上下文）或 bandit（无状态）
└─ 是
   ├─ 已知 P 和 R（有模型）？
   │   ├─ 是，且 S、A 小 → 值迭代 / 策略迭代（L2）
   │   └─ 是，且 S 大 / 需要在线决策 → MCTS（L13/L14）
   └─ 否（无模型）
       ├─ 动作空间离散且小？
       │   ├─ 是 → Q-learning / SARSA（表格型，L4）
       │   │        └─ S 大或连续 → DQN 族（L4）
       │   └─ 否（连续/高维动作）
       │       └─ 策略梯度：REINFORCE → Actor-Critic → GAE + PPO（L5–L7）
       ├─ 样本效率是关键约束？
       │   └─ 是 → 探索算法：UCB（L9/L10）、Thompson 采样（L11）、RMax/PSRL（L11/L12）
       ├─ 有专家示范但没有奖励函数？
       │   ├─ 可以交互问专家 → DAgger（L7）
       │   ├─ 只能离线 → 行为克隆（L7）或 MaxEnt IRL（L7）
       │   └─ 有人类偏好比较 → RLHF / DPO（L8）
       └─ 有环境模型可以学习 → 世界模型 + 想象中训练（L15）
```

**最后的黄金法则**：**没有免费的午餐**。表格型方法的收敛保证在函数逼近下失效；off-policy 的数据效率换来稳定性风险；探索带来更好的长期回报但牺牲短期收益；贝叶斯方法利用先验但依赖先验的正确性。**选择算法就是选择你愿意接受哪一组假设。**

{% endraw %}
