---
title: "Recency Is the Wrong Axis: A Negative Result on Windowed Deepest-Plane KV Caching in Looped Language Models"
collection: publications
category: preprints
permalink: /publication/2026-10-recency-wrong-axis
excerpt: 'A negative result on windowed deepest-plane KV caching in looped language models: the cost of sharing is governed by late-loop convergence and which positions share, not by recency. Co-authored work.'
date: 2026-10-25
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2610.34436'
bibtexurl: '/files/bibtex/lu2026recency.bib'
citation: 'Dezhen Lu, Runguo Li, Zixiang Ni, Yuchen Tang, Xingquan Kong, Jiaxing Yang, and Jiarui Guo. (2026). &quot;Recency Is the Wrong Axis: A Negative Result on Windowed Deepest-Plane KV Caching in Looped Language Models.&quot; arXiv:2610.34436.'
---

**Status:** arXiv preprint · **arXiv:** [2610.34436](https://arxiv.org/abs/2610.34436) · **PDF:** [Download](/files/recency-wrong-axis-kv-caching.pdf)

### Authors

Dezhen Lu, Runguo Li, Zixiang Ni, Yuchen Tang, Xingquan Kong, Jiaxing Yang, and Jiarui Guo.

### Abstract

Looped language models reuse one block of layers over *R* recurrent steps but store a key–value (KV) plane for each step. Keeping only the deepest plane shrinks the cache up to *R*-fold. A natural repair, which we call RIPE, keeps every plane for the *W* most recent positions and shares the deepest plane elsewhere. This repair acts on the wrong axis: across three looped models, the cost of sharing is set by how far the late loops' planes have converged (ripeness) and by which positions share, not by recency. On Raven-Llama the deep planes sit within the bf16 rounding floor, so no window is needed. On Huginn-0125 a 64-position window removes most of a concentrated replay-error tail (9 to 1 positions over 48 prompts), yet in a real self-speculative decoder the policy is slower and larger than the deepest plane alone and not measurably closer to full-cache decoding. On Ouro-1.4B the prompt lies outside any 16–64-position window: on 24 GSM8K questions with a 2048-token budget, the full cache and a FlashLoop baseline finish 23 answers and RIPE at most 2, although physical storage matches the emulated policy bitwise. Sensitivity and ripening models fitted on Huginn and Raven explain their contrast, and Ouro's own report that prompt-side sharing is harmful explains the third. Separately, bf16 execution-shape effects are as large as the cache effects being measured.

### Key Findings

- Recent-window repair targets the wrong axis: sharing costs are governed by **late-loop convergence (ripeness)** and by **which positions share**, not by recency.
- The result differs sharply across looped models: Raven-Llama needs no window; Huginn-0125 reduces a concentrated replay-error tail from 9 to 1 positions with a 64-position window but is slower and larger than the deepest-plane policy; Ouro-1.4B completes 23 of 24 GSM8K questions with full caching versus at most 2 with RIPE.
- **bf16 execution-shape effects are as large as the cache effects** being measured, so evaluations need to account for them.
- Recent-window repair of shared-plane caches should not be adopted without a **per-model ripeness test**; the paper releases ten pre-registered batches with every failed prediction.

**Full text:** [Download the PDF](/files/recency-wrong-axis-kv-caching.pdf) · [arXiv:2610.34436](https://arxiv.org/abs/2610.34436)
