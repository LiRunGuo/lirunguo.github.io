---
title: "Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning"
collection: publications
category: preprints
permalink: /publication/2026-10-breadthkv
excerpt: 'BreadthKV splits a fixed decode-time KV byte budget between how many tokens to keep and at what precision, beating eviction alone in 17 of 18 settings across three reasoning models and four benchmarks. Single-author work.'
date: 2026-10-05
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2610.05685'
bibtexurl: '/files/bibtex/li2026breadthkv.bib'
citation: 'Runguo Li. (2026). &quot;Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning.&quot; arXiv:2610.05685.'
---

**Status:** arXiv preprint · **arXiv:** [2610.05685](https://arxiv.org/abs/2610.05685) · **PDF:** [Download](/files/breadthkv-kv-compression.pdf)

### Author

Runguo Li (single author).

### Abstract

Reasoning models write most of their KV cache while decoding long chains of thought (CoT), so the cache has to be compressed online under a fixed memory budget. Decode-time methods mostly decide which tokens to evict. We ask how a fixed byte budget should be split between the number of cached tokens and their precision.

**BreadthKV** spends the bytes on more tokens at low precision, combining quantization with eviction, and picks the bit-width for each model and budget with a 60-problem end-to-end calibration, since offline attention error does not predict it reliably.

### Key Findings

- On three reasoning models and four math and science benchmarks, BreadthKV scores above eviction alone in **17 of 18 settings** and produces shorter outputs.
- Much of what eviction loses comes from **derailed runs** — samples that keep reasoning until the length cap without reaching an answer. On Qwen3-8B at the tightest budget, eviction sends **91%** of AIME samples to the cap and BreadthKV **40%**.
- Under the same protocol, BreadthKV is statistically indistinguishable from a joint rate–distortion allocator (RDKV) that uses **27% more KV memory-time**, and it outperforms our re-implementation of ThinKV.

**Full text:** [Download the PDF](/files/breadthkv-kv-compression.pdf)
