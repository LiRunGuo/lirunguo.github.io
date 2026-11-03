---
title: "ScaleRLT: Exact Log-Probabilities and Fast Deferred Rollouts for Reinforcement Learning on Looped Transformers"
collection: publications
category: preprints
permalink: /publication/2026-10-scalerlt
excerpt: 'ScaleRLT makes deferred rollouts on looped transformers trainable by replaying the rollout schedule exactly for the learner log-probability, while retaining the dense learner gradient. Co-authored work.'
date: 2026-10-25
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2610.34427'
bibtexurl: '/files/bibtex/lu2026scalerlt.bib'
citation: 'Dezhen Lu, Zixiang Ni, and Runguo Li. (2026). &quot;ScaleRLT: Exact Log-Probabilities and Fast Deferred Rollouts for Reinforcement Learning on Looped Transformers.&quot; arXiv:2610.34427.'
---

**Status:** arXiv preprint · **arXiv:** [2610.34427](https://arxiv.org/abs/2610.34427) · **PDF:** [Download](/files/scalerlt-looped-transformers.pdf)

### Authors

Dezhen Lu, Zixiang Ni, and Runguo Li.

### Abstract

Looped language models such as Ouro reuse one block of layers several times per token. A rollout engine can batch the deeper loops of several response positions and run them only when a deep readout is due. This deferred schedule is fast, but the learner of an on-policy RL algorithm recomputes the sampled tokens' log-probabilities with a dense pass, and the two computations differ in floating point: in FP32 at 4096 tokens about 18% of the sampled tokens miss a $10^{-5}$ same-weight gate, and with bf16 weights nearly every token does. We use a hybrid learner log-probability, $\ell=\ell_{\mathrm{dense}}+\mathrm{sg}(\ell_{\mathrm{sched}}-\ell_{\mathrm{dense}})$, where $\mathrm{sg}$ denotes stop-gradient. Its value is a no-grad replay through the rollout's own schedule, and its gradient is the dense learner's. The replay reproduces the rollout bit for bit in all 167 FP32 and 28 bf16 real-weight cells we ran on three platforms, including rollouts whose batch shrinks at stop tokens, and the gradient has the same FP32 error as other dense implementations in all 102 pre-registered comparisons from 96 to 4096 tokens; the full GRPO loss of a training step matches a float64 reference to $3.1\times10^{-4}$. On one GPU, deferral raises rollout throughput by $1.47$ to $1.56\times$, and by $1.18$ to $1.54\times$ once every step runs from CUDA graphs, bit for bit. In a two-node GRPO loop, one-wave KV admission and CUDA graphs shorten an update by $2.71\times$ with the gate bitwise at every update, and four nodes keep the update time within 2% at the same per-GPU load. In 100-update runs held-out GSM8K accuracy rose from 23% to 82% with the mixed-depth readout that deferral relies on and from 34% to 90% with the native readout, which reached a higher accuracy in less time. For looped models beyond one node we analyse a ring mapping of layers to GPUs and measure its communication paths.

### Key Findings

- A **hybrid learner log-probability** uses the rollout's scheduled computation for its value and the dense computation for its gradient, preserving bitwise rollout replay without changing the optimization target.
- Replay is bit-identical in **167 FP32 and 28 bf16 real-weight cells** across three platforms, including variable-width rollouts that shrink at stop tokens.
- In **102 pre-registered comparisons**, gradient error matches other dense implementations; the full GRPO step loss is within $3.1\times10^{-4}$ of a float64 reference.
- Deferral improves single-GPU rollout throughput by **1.47–1.56×**, or **1.18–1.54×** with every step in CUDA graphs; a two-node GRPO update is **2.71× faster**, and four nodes stay within 2% at equal per-GPU load.
- Over 100 updates, held-out GSM8K accuracy rises from **23% to 82%** with mixed-depth readout and from **34% to 90%** with native readout.

**Full text:** [Download the PDF](/files/scalerlt-looped-transformers.pdf) · [arXiv:2610.34427](https://arxiv.org/abs/2610.34427)
