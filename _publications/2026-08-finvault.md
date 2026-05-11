---
title: "FinVault: Benchmarking Financial Agent Safety in Execution-Grounded Environments"
collection: publications
category: conferences
permalink: /publication/2026-08-finvault
excerpt: 'The first execution-grounded security benchmark for LLM-based financial agents — 31 regulatory sandbox scenarios, 107 real-world vulnerabilities, 963 test cases. Submitted to EMNLP 2026.'
date: 2026-01-09
venue: 'Submitted to EMNLP 2026'
paperurl: 'https://arxiv.org/abs/2601.07853'
citation: 'Zhi Yang, Runguo Li, Qiqi Qiang, Jiashun Wang, Fangqi Lou, et al. (2026). &quot;FinVault: Benchmarking Financial Agent Safety in Execution-Grounded Environments.&quot; <i>arXiv preprint arXiv:2601.07853</i>. Submitted to EMNLP 2026.'
---

**Status:** Submitted to **EMNLP 2026** · **arXiv:** [2601.07853](https://arxiv.org/abs/2601.07853)

### Authors

Zhi Yang, **Runguo Li**, Qiqi Qiang, Jiashun Wang, Fangqi Lou, Mengping Li, Dongpo Cheng, Rui Xu, Heng Lian, Shuo Zhang, Xiaolong Liang, Xiaoming Huang, Zheng Wei, Zhaowei Liu, Xin Guo, Huacan Wang, Ronghao Chen, Liwen Zhang.

### Abstract

Financial agents powered by large language models (LLMs) are increasingly deployed for investment analysis, risk assessment, and automated decision-making, where their abilities to plan, invoke tools, and manipulate mutable state introduce new security risks in high-stakes and highly regulated financial environments. However, existing safety evaluations largely focus on language-model-level content compliance or abstract agent settings, failing to capture **execution-grounded risks** arising from real operational workflows and state-changing actions.

To bridge this gap, we propose **FinVault**, the first execution-grounded security benchmark for financial agents, comprising:

- **31 regulatory case-driven sandbox scenarios** with state-writable databases and explicit compliance constraints,
- **107 real-world vulnerabilities**, and
- **963 test cases** systematically covering prompt injection, jailbreaking, financially adapted attacks, and benign inputs for false-positive evaluation.

### Key Findings

Experimental results reveal that existing defense mechanisms remain ineffective in realistic financial agent settings, with **average attack success rates (ASR) reaching up to 50.0%** on state-of-the-art models and **remaining non-negligible even for the most robust systems (ASR 6.7%)** — highlighting the limited transferability of current safety designs and the need for stronger financial-specific defenses.

### Categories

Cryptography and Security (cs.CR); Artificial Intelligence (cs.AI).
