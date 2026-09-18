---
permalink: /
title: "Hi, I'm Runguo Li (李润果) 👋"
author_profile: true
toc: false
redirect_from: 
  - /about/
  - /about.html
---

I am an **M.S. student in Information Science** at the **University of Illinois Urbana-Champaign (UIUC)** and a graduate of **Shanghai University of Finance and Economics (SUFE)** with a B.S. in Business Analytics (GPA 3.71/4.0, rank 8/128).

I work on **ML systems for large-scale models**: inference engines and expert streaming for MoE models, distributed training and rollout infrastructure, RL post-training systems, and GPU compiler and kernel-level correctness. I like to sit between the machine and the model — how weights are read from storage, how collectives and thread teams are sized, and whether a resource plan and the runtime agree about what will actually be built. I keep a research background in LLM reasoning, multimodal learning, and agent safety, and I bring an upstream-first habit to all of it: reproduce failures in real production stacks, then contribute fixes with regression tests and measured before/after numbers.

My research experience includes the **UIUC** graduate research group of Prof. **Minjia Zhang** on efficient ML systems, the **Tencent Youtu AI Lab** on content-safety and multimodal research, the **SUFE FinAI Center** (advised by Prof. **Liwen Zhang, 张立文**) on financial reasoning and agent safety, **Shanghai Jiao Tong University** (advised by Prof. **Xiaodong Gu, 顾小东**) on **LLM for Code**, and the **Head Office of ICBC (Private Banking Department)** on scientist-discovery agents.

Research Interests
======
- **ML Systems for Large Models** — inference engines and expert streaming for MoE models, KV-cache and memory budgeting, attention and quantization kernels, performance profiling
- **Distributed Training & Rollout** — ZeRO-3, FSDP, Megatron-LM, Hybrid Engine, collective synchronization and deadlock debugging, RL rollout systems (verl, TRL)
- **GPU Compiler & Kernel Correctness** — MLIR passes, Triton/CUDA kernel-level debugging, reproducible regression tests
- **LLM Reasoning & Multimodal Learning** — chain-of-thought supervision, selective fine-tuning, retrieval and fusion
- **Agents & Security** — agentic planning and tool use, RAG and long-term memory, execution-grounded evaluation, runtime safety gates

Selected Highlights
======
- 📝 **FinVault** — *Benchmarking Financial Agent Safety in Execution-Grounded Environments* — **arXiv preprint**, co-first author. [arXiv:2601.07853](https://arxiv.org/abs/2601.07853)
- 📝 **RVCFT** — [*Reasoning-Visual Critical Token Fine-Tuning for Multimodal Reasoning*](/publication/2026-07-rvcft) — co-first author, under review at AAAI 2027.
- 📝 **VeriBRT** — [*Plan-Guided, Evidence-Based Automated Bug Reproduction*](/publication/2026-07-veribrt) — co-first author, under review at ICSE 2027.
- 🛠️ **Open-source ML systems** — merged fixes in **colibri**, **DeepSpeed**, **Triton**, **verl**, and **LMCache**, plus open pull requests across PyTorch, vLLM, SGLang, Megatron-LM, TensorRT-LLM, CUTLASS and more. [See the full portfolio](/portfolio/)
- 🛠️ **ARH (AI Research Helper)** — open-source CLI-first research assistant agent with tool use, plan/execute safety, three-layer memory, skill self-learning and multi-LLM fallback. [github.com/LiRunGuo/Arhelper](https://github.com/LiRunGuo/Arhelper)

Open-Source Contributions (ML Systems)
======
Merged fixes to a pure-C MoE inference engine, DeepSpeed, Triton, verl, and LMCache, plus open pull requests across the wider AI infrastructure stack.

- **colibri — pure-C MoE inference engine** — multi-drive expert streaming for the 510 GB DeepSeek-V4.1 container, physical-core OpenMP team sizing in four engines that were **18.7× slower** without it, and a plan/runtime mismatch that silently allocated **6.25 GiB** of unbudgeted KV cache.
- **DeepSpeed — distributed training** — fixed a ZeRO-3 rollout deadlock caused by unsynchronized generation stopping, and blocked partial Hybrid Engine policy injection for unsupported architectures (validated on H200 and MI250 GPUs).
- **Triton — GPU compiler** — stopped nested-loop fusion from trusting an `llvm.assume` outside the loop, which removed a zero-trip guard and could cause incorrect memory writes (MLIR regression test and H200 reproducer).
- **verl — RL post-training** — restored FSDP value-head critic loading after TRL relocated its value-head model classes.
- **LMCache — KV cache management** — enforced lazy `%`-format logging (ruff G004) so new f-string logging can no longer land in already-migrated files.

[Browse the full portfolio](/portfolio/) — including open pull requests to PyTorch, vLLM, SGLang, Megatron-LM, TensorRT-LLM, CUTLASS, LMCache, Miles and Cordis.

Technical Skills
======
- **ML Systems and Inference** — MoE expert streaming and disk-resident inference, KV-cache and memory budgeting, attention and quantization kernels, serving engines (vLLM, SGLang, TensorRT-LLM), CUDA/Triton kernel-level debugging, GPU compiler passes (MLIR), performance profiling
- **Training and Post-Training Infrastructure** — distributed data/tensor/pipeline parallelism, ZeRO-3 and FSDP, Megatron-LM, DeepSpeed Hybrid Engine, collective synchronization and deadlock debugging, RL rollout systems (verl, TRL), SFT and preference optimization (DPO/GRPO/PPO), distillation, LoRA/PEFT
- **Languages and Tooling** — Python, C, CUDA/Triton, SQL, Bash, LaTeX; PyTorch, Transformers, FlashAttention, FAISS; Git-based upstream contribution, regression testing, reproducible benchmarking, Linux, Docker
- **Systems for Agents** — agentic planning and tool use, RAG and long-term memory, execution-grounded evaluation, runtime safety gates, multi-channel gateways (FastAPI)

News
======
- **2026.09** — Open-source work merged across **colibri**, **DeepSpeed**, **Triton**, **verl**, and **LMCache**; portfolio now tracks 15 open pull requests across the AI infrastructure stack.
- **2026.08** — Started the M.S. in Information Science program at the University of Illinois Urbana-Champaign, advised by Prof. Minjia Zhang.
- **2026.07** — Completed research internships at the SUFE FinAI Center and Shanghai Jiao Tong University.
- **2026.05** — Launched this personal homepage at [runguoli.com](https://runguoli.com). 🎉
- **2026.03** — Joined Shanghai Jiao Tong University as a research intern (LLM for Code, advised by Prof. Xiaodong Gu).
- **2026.01** — *FinVault* released as an arXiv preprint; started working with the Head Office of ICBC on scientist-discovery agents.
- **2025.10** — Joined Tencent Youtu AI Lab as a research intern.

Get in Touch
======
- ✉️  Email: [runguol2@illinois.edu](mailto:runguol2@illinois.edu) · [li19107254665@gmail.com](mailto:li19107254665@gmail.com)
- 🐙 GitHub: [github.com/LiRunGuo](https://github.com/LiRunGuo)
- 📄 [Curriculum Vitae](/files/RunguoLi-ML-Systems-Resume.pdf)
