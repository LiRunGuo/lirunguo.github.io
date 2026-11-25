---
permalink: /
title: "Hi, I'm Runguo Li 👋"
author_profile: true
toc: false
seo_title: "Runguo Li — ML Systems, UIUC"
redirect_from: 
  - /about/
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
[Google Scholar profile](https://scholar.google.com/citations?user=t82uTeYAAAAJ&hl=en)

{% assign selected_papers = site.data.scholar_publications | where: "group", "selected" %}
{% for paper in selected_papers %}
- [**{{ paper.title }}**]({{ paper.scholar }}) — {{ paper.authors }} <em>arXiv:{{ paper.arxiv }}, {{ paper.year }}</em>. [arXiv](https://arxiv.org/abs/{{ paper.arxiv }})
{% endfor %}

Additional Publications
======
{% assign additional_papers = site.data.scholar_publications | where: "group", "additional" %}
{% for paper in additional_papers %}
- [**{{ paper.title }}**]({{ paper.scholar }}) — {{ paper.authors }} <em>arXiv:{{ paper.arxiv }}, {{ paper.year }}</em>. [arXiv](https://arxiv.org/abs/{{ paper.arxiv }})
{% endfor %}

[All publications](/publications/)

Open-Source Contributions (ML Systems)
======
30 public merged pull requests across 17 public upstream projects, plus 8 fixes adopted into PyTorch and NVIDIA NCCL mainlines and 7 reported issues resolved upstream by other changes. Top five projects by GitHub stars: [Kubernetes](https://github.com/kubernetes/kubernetes/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), [vLLM](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), [colibri](https://github.com/JustVugg/colibri/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged), and [LLVM](https://github.com/llvm/llvm-project/pulls?q=is%3Apr+author%3ALiRunGuo+is%3Amerged). [View all contribution links](/portfolio/).

Get in Touch
======
- ✉️  Email: [runguol2@illinois.edu](mailto:runguol2@illinois.edu)
- 💬 WeChat: **RunguoLi2004**
- 🐙 GitHub: [github.com/LiRunGuo](https://github.com/LiRunGuo)
- 🐦 X: [@Runguo_Li](https://x.com/Runguo_Li)
- 🎓 Google Scholar: [Runguo Li](https://scholar.google.com/citations?user=t82uTeYAAAAJ&hl=en)
- 🆔 ORCID: [0009-0002-0832-6227](https://orcid.org/0009-0002-0832-6227)
- 📄 [Curriculum Vitae](/files/RunguoLi-ML-Systems-Resume.pdf)
