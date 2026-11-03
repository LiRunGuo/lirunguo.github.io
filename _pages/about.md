---
permalink: /
title: "Hi, I'm Runguo Li 👋"
author_profile: true
toc: false
redirect_from: 
  - /about/
  - /about.html
---

I am an **M.S. student in Information Management** at the **University of Illinois Urbana-Champaign (UIUC)**.

I work on **ML systems for large-scale models**: inference engines and expert streaming for MoE models, distributed training and rollout infrastructure, RL post-training systems, and GPU compiler and kernel-level correctness. I like to sit between the machine and the model — how weights are read from storage, how collectives and thread teams are sized, and whether a resource plan and the runtime agree about what will actually be built. I keep a research background in LLM reasoning, multimodal learning, and agent safety, and I bring an upstream-first habit to all of it: reproduce failures in real production stacks, then contribute fixes with regression tests and measured before/after numbers.

My research experience includes research at **UIUC** on efficient ML systems, the **Tencent Youtu AI Lab** on content-safety and multimodal research, the **SUFE FinAI Center** (advised by Prof. **Liwen Zhang**) on financial reasoning and agent safety, **Shanghai Jiao Tong University** (advised by Prof. **Xiaodong Gu**) on **LLM for Code**, and the **Head Office of ICBC (Private Banking Department)** on scientist-discovery agents.

🎓 **I plan to apply to Ph.D. programs in ML systems for Fall 2028.** I am happy to hear from faculty and students working on efficient inference, training infrastructure, or GPU kernels.

Research Interests
======
- **ML Systems for Large Models** — inference engines and expert streaming for MoE models, KV-cache and memory budgeting, attention and quantization kernels, performance profiling
- **Distributed Training & Rollout** — ZeRO-3, FSDP, Megatron-LM, Hybrid Engine, collective synchronization and deadlock debugging, RL rollout systems (verl, TRL)
- **GPU Compiler & Kernel Correctness** — MLIR passes, Triton/CUDA kernel-level debugging, reproducible regression tests

Selected Publications
======
- 📝 **GPTQ under random rotation** — [*How Much Does Random Rotation Help GPTQ? Sharp Worst-Case Rate Gaps in the High-Rate Regime*](/publication/2026-10-gptq-rotation) — single author, under review. [arXiv:2611.01669](https://arxiv.org/abs/2611.01669) · [PDF](/files/gptq-random-rotation.pdf)
- 📝 **Recency Is the Wrong Axis** — [*Recency Is the Wrong Axis: A Negative Result on Windowed Deepest-Plane KV Caching in Looped Language Models*](/publication/2026-10-recency-wrong-axis) — **arXiv preprint**, co-author. [arXiv:2610.34436](https://arxiv.org/abs/2610.34436)
- 📝 **ScaleRLT** — [*ScaleRLT: Exact Log-Probabilities and Fast Deferred Rollouts for Reinforcement Learning on Looped Transformers*](/publication/2026-10-scalerlt) — **arXiv preprint**, co-author. [arXiv:2610.34427](https://arxiv.org/abs/2610.34427)
- 📝 **Certified Sparse Attention** — [*Certified Sparse Attention via Perturbed Retrieval*](/publication/2026-10-certified-sparse-attention) — **arXiv preprint**, single author. [arXiv:2610.24991](https://arxiv.org/abs/2610.24991)
- 📝 **BreadthKV** — [*Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning*](/publication/2026-10-breadthkv) — **arXiv preprint**, single author. [arXiv:2610.05685](https://arxiv.org/abs/2610.05685)
- 📝 **FinVault** — [*Benchmarking Financial Agent Safety in Execution-Grounded Environments*](/publication/2026-01-finvault) — **arXiv preprint**, co-first author. [arXiv:2601.07853](https://arxiv.org/abs/2601.07853)

[All publications](/publications/)

Open-Source Contributions (ML Systems)
======
22 public merged pull requests across 14 public upstream projects. Top five projects by GitHub stars: [vLLM](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), [colibri](https://github.com/JustVugg/colibri/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), [LLVM](https://github.com/llvm/llvm-project/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), and [SGLang](https://github.com/sgl-project/sglang/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged). [View all contribution links](/portfolio/).

Get in Touch
======
- ✉️  Email: [runguol2@illinois.edu](mailto:runguol2@illinois.edu)
- 💬 WeChat: **RunguoLi2004**
- 🐙 GitHub: [github.com/LiRunGuo](https://github.com/LiRunGuo)
- 🎓 Google Scholar: [Runguo Li](https://scholar.google.com/citations?user=t82uTeYAAAAJ&hl=en)
- 🆔 ORCID: [0009-0002-0832-6227](https://orcid.org/0009-0002-0832-6227)
- 📄 [Curriculum Vitae](/files/RunguoLi-ML-Systems-Resume.pdf)
