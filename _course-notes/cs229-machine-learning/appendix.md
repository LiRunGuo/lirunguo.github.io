---
title: "6. 核心数学与算法速查表（Quick Reference）"
collection: course-notes
chapter: true
permalink: /course-notes/cs229-machine-learning/appendix
toc: true
toc_sticky: true
---
> [目录](/course-notes/cs229-machine-learning/) · [← l20](/course-notes/cs229-machine-learning/l20)

{% raw %}
## 6. 核心数学与算法速查表（Quick Reference）

### 6.1 监督学习：核心公式

| 模型 | 假设/损失 | 关键公式 |
|---|---|---|
| 线性回归 | 高斯噪声 + MLE | $h_\\theta(x)=\\theta^T x$；$J(\\theta)=\\frac{1}{2m}\\sum_i(h_\\theta(x^{(i)})-y^{(i)})^2$；$\\theta=(X^TX)^{-1}X^Ty$ |
| 梯度下降 | — | $\\theta_j := \\theta_j - \\alpha \\frac{1}{m}\\sum_i (h_\\theta(x^{(i)})-y^{(i)})x_j^{(i)}$（批量）；单样本版为 SGD |
| 局部加权回归 | 加权 MSE | $J(\\theta)=\\frac12\\sum_i w^{(i)}(y^{(i)}-\\theta^Tx^{(i)})^2$，$w^{(i)}=\\exp(-\\frac{(x^{(i)}-x)^2}{2\\tau^2})$ |
| 逻辑回归 | Bernoulli + MLE | $h_\\theta(x)=\\frac{1}{1+e^{-\\theta^Tx}}$；$\\ell=\\sum_i[y^{(i)}\\log h+(1-y^{(i)})\\log(1-h)]$；$\\theta_j:=\\theta_j+\\alpha\\sum_i(y^{(i)}-h_\\theta(x^{(i)}))x_j^{(i)}$ |
| 牛顿法 | 二阶优化 | $\\theta := \\theta - H^{-1}\\nabla_\\theta \\ell(\\theta)$，$H_{jk}=\\partial^2\\ell/\\partial\\theta_j\\partial\\theta_k$ |
| 指数族/GLM | $p(y;\\eta)=b(y)e^{\\eta^T T(y)-a(\\eta)}$ | $h_\\theta(x)=E[T(y)\\vert x]=a^{\\prime}(\\theta^Tx)$；特例：高斯→线性回归、伯努利→逻辑回归、多项→Softmax |
| Softmax 回归 | 多项分布 | $h_\\theta(x)=\\frac{1}{\\sum_j e^{\\theta_j^Tx}}[e^{\\theta_1^Tx},\\dots,e^{\\theta_k^Tx}]^T$ |
| GDA | 高斯 + 贝叶斯 | $\\phi,\\mu_0,\\mu_1,\\Sigma$ 由 MLE 估计；共享 $\\Sigma$ ⇒ 线性决策边界 |
| 朴素贝叶斯 | 条件独立 | $P(y\\vert x)\\propto P(y)\\prod_j P(x_j\\vert y)$；拉普拉斯平滑 $\\phi_{j\\vert y}=\\frac{\\#+1}{\\#+2}$ |
| 岭回归 ($L_2$) | 高斯先验/MAP | $\\theta=(X^TX+\\lambda I)^{-1}X^Ty$ |
| Lasso ($L_1$) | 拉普拉斯先验 | $\\min \\frac{1}{2m}\\vert X\\theta-y\\vert ^2+\\lambda\\vert \\theta\\vert _1$ ⇒ 稀疏解（特征选择） |
| 偏差-方差分解 | — | $\\mathbb{E}[(y-\\hat f)^2]=\\sigma^2+\\text{Bias}^2+\\text{Var}$ |

### 6.2 SVM 与核方法

| 概念 | 公式 |
|---|---|
| 核 | $K(x,z)=\\langle\\phi(x),\\phi(z)\\rangle$；高斯核 $K=\\exp(-\\frac{\\vert x-z\\vert ^2}{2\\sigma^2})$ |
| SVM 原始问题 | $\\min_{w,b}\\frac12\\vert w\\vert ^2$ s.t. $y^{(i)}(w^Tx^{(i)}+b)\\ge 1$ |
| 软间隔 | $\\min \\frac12\\vert w\\vert ^2+C\\sum_i\\xi_i$ s.t. $y^{(i)}(w^Tx^{(i)}+b)\\ge 1-\\xi_i,\\ \\xi_i\\ge0$ |
| 对偶决策 | $h(x)=\\text{sign}(\\sum_{i\\in SV}\\alpha_i y^{(i)}K(x^{(i)},x)+b)$ |

### 6.3 无监督学习

| 算法 | 核心步骤/公式 |
|---|---|
| K-Means | 指派 $c^{(i)}=\\arg\\min_j\\vert x^{(i)}-\\mu_j\\vert ^2$；更新 $\\mu_j=\\frac{1}{\\vert S_j\\vert }\\sum_{i\\in S_j}x^{(i)}$；最小化 $J=\\sum_i\\vert x^{(i)}-\\mu_{c^{(i)}}\\vert ^2$ |
| GMM | $p(x)=\\sum_j\\phi_j\\mathcal{N}(x;\\mu_j,\\Sigma_j)$；E 步 $w_j^{(i)}\\propto\\phi_j\\mathcal{N}(x^{(i)};\\mu_j,\\Sigma_j)$；M 步加权 MLE |
| EM | E 步：$Q_i(z)=P(z\\vert x^{(i)};\\theta)$；M 步：$\\theta=\\arg\\max_\\theta\\sum_i\\sum_z Q_i(z)\\log\\frac{p(x^{(i)},z;\\theta)}{Q_i(z)}$；Jensen 保证单调 |
| PCA | $\\Sigma=\\frac1mX^TX$；特征分解取前 $k$ 特征向量 $U_k$；$z=U_k^T\\tilde x$ |
| ICA | $x=As$，找 $W=A^{-1}$ 使 $Wx$ 分量独立（最大化非高斯性） |

### 6.4 决策树与 Boosting

| 算法 | 核心公式 |
|---|---|
| 熵/信息增益 | $H(S)=-\\sum_c p_c\\log_2 p_c$；$\\text{Gain}=H(S)-\\sum_v\\frac{\\vert S_v\\vert }{\\vert S\\vert }H(S_v)$ |
| Gini | $\\text{Gini}(S)=1-\\sum_c p_c^2$ |
| AdaBoost | $\\epsilon_t=\\sum_{i: h_t(x^{(i)})\\ne y^{(i)}}D^{(i)}$；$\\alpha_t=\\frac12\\ln\\frac{1-\\epsilon_t}{\\epsilon_t}$；$D^{(i)}\\propto D^{(i)}e^{-\\alpha_t y^{(i)}h_t(x^{(i)})}$；$H(x)=\\text{sign}(\\sum_t\\alpha_t h_t(x))$ |

### 6.5 神经网络

| 概念 | 公式 |
|---|---|
| 前向传播 | $z^{[l]}=W^{[l]}a^{[l-1]}+b^{[l]}$；$a^{[l]}=g^{[l]}(z^{[l]})$ |
| 反向传播 | $\\delta^{[L]}=\\partial J/\\partial z^{[L]}$（Softmax+CE: $a^{[L]}-y$）；$\\delta^{[l]}=(W^{[l+1]T}\\delta^{[l+1]})\\odot g^{\\prime}(z^{[l]})$；$\\partial J/\\partial W^{[l]}=\\delta^{[l]}a^{[l-1]T}$ |
| 激活导数 | Sigmoid: $g(1-g)$；tanh: $1-g^2$；ReLU: $\\mathbf{1}\\{z>0\\}$ |
| 梯度下降 | $\\theta:=\\theta-\\alpha\\nabla_\\theta J$（Adam 为自适应步长变体） |

### 6.6 强化学习

| 概念 | 公式 |
|---|---|
| 价值函数 | $V^\\pi(s)=\\mathbb{E}[\\sum_t\\gamma^t R(s_t)\\vert s_0=s,\\pi]$ |
| 贝尔曼最优 | $V^*(s)=R(s)+\\gamma\\max_a\\sum_{s^{\\prime}}P_{sa}(s^{\\prime})V^*(s^{\\prime})$ |
| 值迭代 | 反复应用上式直至收敛；$\\pi^*(s)=\\arg\\max_a\\sum_{s^{\\prime}}P_{sa}(s^{\\prime})V^*(s^{\\prime})$ |
| 策略迭代 | 评估 $V^\\pi$（解线性方程）→ 改进 $\\pi^{\\prime}(s)=\\arg\\max_a\\sum_{s^{\\prime}}P_{sa}(s^{\\prime})V^\\pi(s^{\\prime})$ |
| Q-learning | $Q(s,a)\\mathrel{+}=\\alpha[r+\\gamma\\max_{a^{\\prime}}Q(s^{\\prime},a^{\\prime})-Q(s,a)]$（无模型，off-policy） |
| REINFORCE | $\\theta:=\\theta+\\alpha\\sum_t R_t\\nabla_\\theta\\log\\pi_\\theta(a_t\\vert s_t)$（似然比技巧） |

### 6.7 现代主题（LLM / 公平 / 隐私）

| 概念 | 公式/要点 |
|---|---|
| 语言建模 | $\\mathcal{L}=-\\frac1T\\sum_t\\log P_\\theta(x_t\\vert x_{<t})$；PPL $=\\exp(\\mathcal{L})$ |
| 注意力 | $\\text{Attn}(Q,K,V)=\\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V$ |
| 余弦相似度（RAG 检索） | $\\cos(u,v)=\\frac{u\\cdot v}{\\vert u\\vert \\vert v\\vert }$ |
| LoRA | $W^{\\prime}=W+BA$，$B\\in\\mathbb{R}^{d\\times r},A\\in\\mathbb{R}^{r\\times d}$，$r\\ll d$ |
| 公平指标 | 统计均等：$P(\\hat Y=1\\vert A=a)=P(\\hat Y=1\\vert A=b)$；均等化几率：TPR/FPR 组间相等 |
| 差分隐私 | $P(\\mathcal{A}(D)\\in S)\\le e^\\epsilon P(\\mathcal{A}(D^{\\prime})\\in S)$；机制：裁剪梯度 + 加噪（DP-SGD） |

---

## 7. 参考资料与链接

**课程官方（公开部分）**
- 课程主页：https://cs229.stanford.edu/
- Syllabus（Fall 2021，含完整日程与讲义链接）：https://cs229.stanford.edu/syllabus-fall2021.html
- Syllabus（Fall 2020 / Spring 2021）：https://cs229.stanford.edu/syllabus-fall2020.html
- Course Logistics & FAQ（Summer 2026，公开 Google Doc）：https://docs.google.com/document/d/1PbQxBQTpp4K5hTzOJB9r8etKgsAuNPWs/

**公开讲义 PDF（本次笔记的数学内容主要依据）**
- cs229-notes1（监督学习/线性回归/逻辑回归/GLM）：http://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf
- cs229-notes2（生成学习/GDA/朴素贝叶斯）：http://cs229.stanford.edu/notes2021fall/cs229-notes2.pdf
- cs229-notes3（核方法/SVM）：http://cs229.stanford.edu/notes2021fall/cs229-notes3.pdf
- cs229-notes5（正则化与模型选择）：http://cs229.stanford.edu/notes2021fall/cs229-notes5.pdf
- cs229-notes7a（K-Means）/ notes7b（GMM 与 EM）：http://cs229.stanford.edu/notes2021fall/cs229-notes7a.pdf
- cs229-notes8（EM 算法）、notes9（因子分析）、notes10（PCA）、notes11（ICA）、notes12（强化学习）
- deep_learning_notes（神经网络与反向传播）：http://cs229.stanford.edu/notes2021fall/deep_learning_notes.pdf
- ML-advice（调试机器学习系统）：http://cs229.stanford.edu/materials/ML-advice.pdf
- 决策树/Boosting/GMM/EM 幻灯片（Fall 2021）：http://cs229.stanford.edu/notes2021fall/lecture11-decision-trees.pdf 等

**受限资源（需 Stanford 账号，本笔记未使用）**：当季 Syllabus 与讲义（Google Drive）、Canvas（录播/日历）、Ed 论坛、Problem Sets、期中/期末试卷、最终项目。

> **免责声明**：本笔记为学习用途整理，基于公开 Syllabus、公开讲义与通用机器学习知识撰写；数学符号以 CS229 讲义为准。如用于正式学习，请以课程官方当季材料为准。

*—— 完 ——*

{% endraw %}
