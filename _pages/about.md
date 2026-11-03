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
- 📝 **Interrupt Budgets** — [*Interrupt Budgets for Pre-Execution AI Control Monitors*](/publication/2026-11-interrupt-budgets) — single author, under review. [PDF](/files/interrupt-budgets-ai-control.pdf)
- 📝 **Trust but Verify** — [*Trust but Verify: Diagnosing Causal Prior-Data Fitted Networks and Testing When to Defer*](/publication/2026-11-causal-pfn-defer) — single author, under review. [PDF](/files/causal-pfn-diagnosis-defer.pdf)
- 📝 **AXR** — [*AXR: Reusing Intermediate Results for LLM Agents in PostgreSQL*](/publication/2026-10-axr) — single author, under review. [PDF](/files/axr-agent-result-reuse.pdf)
- 📝 **Group-Conditional Coverage** — [*What Do Group-Conditional Coverage Metrics Measure?*](/publication/2026-11-group-coverage) — single author, under review. [PDF](/files/group-conditional-coverage.pdf)
- 📝 **Supernode Specificity** — [*Is That Supernode Special?*](/publication/2026-11-supernode-specificity) — single author, under review. [PDF](/files/supernode-specificity.pdf)
- 📝 **Search-Agent Answer State** — [*Do Search Agents Know When They Know?*](/publication/2026-11-search-agent-answer-state) — single author, under review. [PDF](/files/search-agent-answer-state.pdf)
- 📝 **FP4 Long Reasoning** — [*What FP4 Costs Long Reasoning on Blackwell, Measured Above the Noise*](/publication/2026-11-fp4-long-reasoning) — single author, under review. [PDF](/files/fp4-long-reasoning.pdf)
- 📝 **Tessera** — [*Predict, Don't Just Pair: Lossless Storage for Families of Fine-Tuned Models*](/publication/2026-11-tessera-lossless-storage) — single author, under review. [PDF](/files/tessera-lossless-storage.pdf)
- 📝 **What Survives Distillation?** — [*What Survives Distillation? A Pre-registered Study of Training-Free Acceleration*](/publication/2026-11-gmsa-distillation) — single author, under review. [PDF](/files/gmsa-distillation-acceleration.pdf)
- 📝 **Horizon-Aware Losses** — [*What Horizon-Aware Losses Do in Multi-Step Forecasting*](/publication/2026-11-horizon-aware-losses) — single author, under review. [PDF](/files/horizon-aware-losses.pdf)
- 📝 **GPTQ under random rotation** — [*How Much Does Random Rotation Help GPTQ? Sharp Worst-Case Rate Gaps in the High-Rate Regime*](/publication/2026-10-gptq-rotation) — single author, under review. [arXiv:2611.01669](https://arxiv.org/abs/2611.01669) · [PDF](/files/gptq-random-rotation.pdf)
- 📝 **Recency Is the Wrong Axis** — [*Recency Is the Wrong Axis: A Negative Result on Windowed Deepest-Plane KV Caching in Looped Language Models*](/publication/2026-10-recency-wrong-axis) — **arXiv preprint**, co-author. [arXiv:2610.34436](https://arxiv.org/abs/2610.34436)
- 📝 **ScaleRLT** — [*ScaleRLT: Exact Log-Probabilities and Fast Deferred Rollouts for Reinforcement Learning on Looped Transformers*](/publication/2026-10-scalerlt) — **arXiv preprint**, co-author. [arXiv:2610.34427](https://arxiv.org/abs/2610.34427)
- 📝 **Certified Sparse Attention** — [*Certified Sparse Attention via Perturbed Retrieval*](/publication/2026-10-certified-sparse-attention) — **arXiv preprint**, single author. [arXiv:2610.24991](https://arxiv.org/abs/2610.24991)
- 📝 **BreadthKV** — [*Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning*](/publication/2026-10-breadthkv) — **arXiv preprint**, single author. [arXiv:2610.05685](https://arxiv.org/abs/2610.05685)
- 📝 **FinVault** — [*Benchmarking Financial Agent Safety in Execution-Grounded Environments*](/publication/2026-01-finvault) — **arXiv preprint**, co-first author. [arXiv:2601.07853](https://arxiv.org/abs/2601.07853)
- 📝 **RVCFT** — [*Reasoning-Visual Critical Token Fine-Tuning for Multimodal Reasoning*](/publication/2026-07-rvcft) — co-first author, under review.
- 📝 **VeriBRT** — [*Plan-Guided, Evidence-Based Automated Bug Reproduction*](/publication/2026-07-veribrt) — co-first author, under review.

[All publications](/publications/)

Open-Source Contributions (ML Systems)
======
22 public merged pull requests across 14 public upstream projects (private repositories excluded), including vLLM, colibri, DeepSpeed, LLVM, SGLang, JAX, FlashAttention, Triton, Megatron-LM, and FlashInfer. The count includes merged pull requests only; separately filed MLflow bug reports whose fixes landed upstream are not counted. A selection, ordered by the upstream project's star count:

- **vLLM — KV cache quantization** — let the int4 per-token-head KV cache run on head sizes that are not powers of two (e.g. Phi-3's 96), where a power-of-two-only Hadamard rotation aborted engine initialization instead of quantizing.
- **colibri — pure-C MoE inference engine** — multi-drive expert streaming for the 510 GB DeepSeek-V4.1 container, physical-core OpenMP team sizing in four engines that were **18.7× slower** without it, and a plan/runtime mismatch that silently allocated **6.25 GiB** of KV cache against a 0.02 GiB budget.
- **DeepSpeed — distributed training** — fixed a ZeRO-3 rollout deadlock caused by unsynchronized generation stopping, and blocked partial Hybrid Engine policy injection for unsupported architectures (validated on H200 and MI250 GPUs).
- **FlashAttention — CuTe kernels** — fitted the SM90 forward tile to the block-sparse block size (block sparsity on Hopper went from 8 of 40 head-dim/block-size combinations working to 32), and stopped the causal forward from re-applying an all-true mask on unmasked KV blocks, cutting its instruction count from 580 to 477 per loop.
- **Triton — GPU compiler** — stopped nested-loop fusion from trusting an `llvm.assume` outside the loop, which removed a zero-trip guard and could cause incorrect memory writes (MLIR regression test and H200 reproducer).
- **FlashInfer — attention planning** — made strided `indptr`, `indices`, and `last_page_len` metadata contiguous before planning and kernel use, preventing illegal memory accesses and wrong outputs across ragged, paged, and block-sparse attention backends.

[Browse the full portfolio](/portfolio/) — every entry comes with a reproducer or regression test attached.

I also maintain **ARH (AI Research Helper)**, an open-source CLI-first research assistant agent with tool use, plan/execute safety, three-layer memory, skill self-learning and multi-LLM fallback. [github.com/LiRunGuo/Arhelper](https://github.com/LiRunGuo/Arhelper)

News
======
- **2026.11** — Public contribution audit refreshed: **22 merged PRs across 14 public upstream projects**, now including [FlashInfer #5385](https://github.com/flashinfer-ai/flashinfer/pull/5385). Unmerged pull requests and private repositories are excluded.
- **2026.11** — Two co-authored arXiv preprints are now available: *Recency Is the Wrong Axis* ([2610.34436](https://arxiv.org/abs/2610.34436)) and *ScaleRLT* ([2610.34427](https://arxiv.org/abs/2610.34427)).
- **2026.11** — Added ten top-ranked working papers to the publications page, each marked as under review. The GPTQ random-rotation paper is now also available as [arXiv:2611.01669](https://arxiv.org/abs/2611.01669).
- **2026.10** — Two new single-author manuscripts under review: *AXR* (intermediate-result reuse for LLM agents in PostgreSQL) and *How Much Does Random Rotation Help GPTQ?*.
- **2026.10** — *Certified Sparse Attention via Perturbed Retrieval* (single-author) released as an arXiv preprint: [2610.24991](https://arxiv.org/abs/2610.24991). Merged in **Megatron-LM**: MoE checkpoint shards now take the DCP checkpointable path (165× faster state-dict conversion, 3.4× shorter async-save blocking on 8×H200).
- **2026.10** — *BreadthKV* (single-author) released as an arXiv preprint: [2610.05685](https://arxiv.org/abs/2610.05685). New merged fixes in **vLLM** (int4 KV cache), **SGLang** (server startup), **LLVM** (Attributor pointer tracking and ValueTracking range implications), **Triton** (fp64 atomics on AMD), and **vLLM-Omni** (block-sparse Token2Wav attention, up to 3.8× faster DiT).
- **2026.09** — Open-source work merged across **DeepSpeed**, **colibri**, **JAX**, **FlashAttention**, **verl**, **Triton**, **Apache TVM**, **LMCache**, and **vLLM-Omni**, plus bug reports fixed upstream in **MLflow**.
- **2026.08** — Started the M.S. in Information Management program at the University of Illinois Urbana-Champaign.
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
