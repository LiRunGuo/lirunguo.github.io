---
permalink: /
title: "Hi, I'm Runguo Li 👋"
author_profile: true
toc: false
redirect_from: 
  - /about/
  - /about.html
---

I am an **M.S. student in Information Science** at the **University of Illinois Urbana-Champaign (UIUC)**.

I work on **ML systems for large-scale models**: inference engines and expert streaming for MoE models, distributed training and rollout infrastructure, RL post-training systems, and GPU compiler and kernel-level correctness. I like to sit between the machine and the model — how weights are read from storage, how collectives and thread teams are sized, and whether a resource plan and the runtime agree about what will actually be built. I keep a research background in LLM reasoning, multimodal learning, and agent safety, and I bring an upstream-first habit to all of it: reproduce failures in real production stacks, then contribute fixes with regression tests and measured before/after numbers.

My research experience includes research at **UIUC** on efficient ML systems, the **Tencent Youtu AI Lab** on content-safety and multimodal research, the **SUFE FinAI Center** (advised by Prof. **Liwen Zhang**) on financial reasoning and agent safety, **Shanghai Jiao Tong University** (advised by Prof. **Xiaodong Gu**) on **LLM for Code**, and the **Head Office of ICBC (Private Banking Department)** on scientist-discovery agents.

🎓 **I plan to apply to Ph.D. programs in ML systems for Fall 2028.** I am happy to hear from faculty and students working on efficient inference, training infrastructure, or GPU kernels.

Research Interests
======
- **ML Systems for Large Models** — inference engines and expert streaming for MoE models, KV-cache and memory budgeting, attention and quantization kernels, performance profiling
- **Distributed Training & Rollout** — ZeRO-3, FSDP, Megatron-LM, Hybrid Engine, collective synchronization and deadlock debugging, RL rollout systems (verl, TRL)
- **GPU Compiler & Kernel Correctness** — MLIR passes, Triton/CUDA kernel-level debugging, reproducible regression tests

Earlier work covered LLM reasoning and multimodal learning ([RVCFT](/publication/2026-07-rvcft)), LLM for code ([VeriBRT](/publication/2026-07-veribrt)), and financial agent safety ([FinVault](/publication/2026-01-finvault)).

Selected Publications
======
- 📝 **BreadthKV** — [*Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning*](/publication/2026-10-breadthkv) — **arXiv preprint**, single author. [arXiv:2610.05685](https://arxiv.org/abs/2610.05685)
- 📝 **FinVault** — [*Benchmarking Financial Agent Safety in Execution-Grounded Environments*](/publication/2026-01-finvault) — **arXiv preprint**, co-first author. [arXiv:2601.07853](https://arxiv.org/abs/2601.07853)
- 📝 **RVCFT** — [*Reasoning-Visual Critical Token Fine-Tuning for Multimodal Reasoning*](/publication/2026-07-rvcft) — co-first author, under review at AAAI 2027.
- 📝 **VeriBRT** — [*Plan-Guided, Evidence-Based Automated Bug Reproduction*](/publication/2026-07-veribrt) — co-first author, under review at ICSE 2027.

[All publications](/publications/)

Open-Source Contributions (ML Systems)
======
17 merged pull requests across 12 upstream projects, including PyTorch, vLLM, DeepSpeed, SGLang, JAX, FlashAttention, and Triton, plus an MLflow bug report whose fix landed upstream. A selection, ordered by the upstream project's star count:

- **PyTorch — CUDA kernels** — fixed an int32 overflow in the `cdist` backward kernel, where indices wrapped negative past 2³¹ and slipped past the bounds check into an illegal memory access.
- **vLLM — KV cache quantization** — let the int4 per-token-head KV cache run on head sizes that are not powers of two (e.g. Phi-3's 96), where a power-of-two-only Hadamard rotation aborted engine initialization instead of quantizing.
- **DeepSpeed — distributed training** — fixed a ZeRO-3 rollout deadlock caused by unsynchronized generation stopping, and blocked partial Hybrid Engine policy injection for unsupported architectures (validated on H200 and MI250 GPUs).
- **colibri — pure-C MoE inference engine** — multi-drive expert streaming for the 510 GB DeepSeek-V4.1 container, physical-core OpenMP team sizing in four engines that were **18.7× slower** without it, and a plan/runtime mismatch that silently allocated **6.25 GiB** of KV cache against a 0.02 GiB budget.
- **FlashAttention — CuTe kernels** — fitted the SM90 forward tile to the block-sparse block size (block sparsity on Hopper went from 8 of 40 head-dim/block-size combinations working to 32), and stopped the causal forward from re-applying an all-true mask on unmasked KV blocks, cutting its instruction count from 580 to 477 per loop.
- **Triton — GPU compiler** — stopped nested-loop fusion from trusting an `llvm.assume` outside the loop, which removed a zero-trip guard and could cause incorrect memory writes (MLIR regression test and H200 reproducer).

[Browse the full portfolio](/portfolio/) — every entry comes with a reproducer or regression test attached.

I also maintain **ARH (AI Research Helper)**, an open-source CLI-first research assistant agent with tool use, plan/execute safety, three-layer memory, skill self-learning and multi-LLM fallback. [github.com/LiRunGuo/Arhelper](https://github.com/LiRunGuo/Arhelper)

News
======
- **2026.10** — *BreadthKV* (single-author) released as an arXiv preprint: [2610.05685](https://arxiv.org/abs/2610.05685). New merged fixes in **vLLM** (int4 KV cache) and **SGLang** (server startup).
- **2026.09** — Open-source work merged across **PyTorch**, **DeepSpeed**, **colibri**, **JAX**, **FlashAttention**, **verl**, **Triton**, **Apache TVM**, **LMCache**, and **vLLM-Omni**, plus a bug report fixed upstream in **MLflow**.
- **2026.08** — Started the M.S. in Information Science program at the University of Illinois Urbana-Champaign.
- **2026.07** — Completed research internships at the SUFE FinAI Center and Shanghai Jiao Tong University.
- **2026.05** — Launched this personal homepage at [runguoli.com](https://runguoli.com). 🎉
- **2026.03** — Joined Shanghai Jiao Tong University as a research intern (LLM for Code, advised by Prof. Xiaodong Gu).
- **2026.01** — *FinVault* released as an arXiv preprint; started working with the Head Office of ICBC on scientist-discovery agents.
- **2025.10** — Joined Tencent Youtu AI Lab as a research intern.

Get in Touch
======
- ✉️  Email: [runguo.ai@gmail.com](mailto:runguo.ai@gmail.com)
- 🐙 GitHub: [github.com/LiRunGuo](https://github.com/LiRunGuo)
- 🎓 Google Scholar: [Runguo Li](https://scholar.google.com/citations?user=t82uTeYAAAAJ&hl=en)
- 🆔 ORCID: [0009-0002-0832-6227](https://orcid.org/0009-0002-0832-6227)
- 📄 [Curriculum Vitae](/files/RunguoLi-ML-Systems-Resume.pdf)
