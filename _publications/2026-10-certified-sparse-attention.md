---
title: "Certified Sparse Attention via Perturbed Retrieval"
collection: publications
category: preprints
permalink: /publication/2026-10-certified-sparse-attention
excerpt: 'Sparse attention with a per-head error certificate: perturbing key scores turns threshold retrieval into weighted sampling, so each head stops reading keys once its bound meets the target error. No certificate is violated on three LLMs at 16K–128K tokens. Single-author work.'
date: 2026-10-18
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2610.24991'
bibtexurl: '/files/bibtex/li2026certified.bib'
citation: 'Runguo Li. (2026). &quot;Certified Sparse Attention via Perturbed Retrieval.&quot; arXiv:2610.24991.'
---

**Status:** arXiv preprint · **arXiv:** [2610.24991](https://arxiv.org/abs/2610.24991) · **PDF:** [Download](/files/certified-sparse-attention.pdf)

### Author

Runguo Li (single author).

### Abstract

Sparse attention speeds up long-context decoding by reading few keys of the KV cache, but existing methods do not control the error of a query, and on real KV caches their errors have heavy tails. We show that perturbing the score of every key with an independent Exp(1) variable turns threshold retrieval into sampling keys with probability proportional to their attention weights. Lowering the threshold along a geometric grid gives nested samples, a Horvitz–Thompson estimate of the output at every level, and confidence bounds that hold simultaneously over all levels, so each head stops at the first level whose certificate meets the target error ε. With fresh perturbations at every decoding step, each certificate fails with probability at most δ during generation. At a suitable threshold, the estimator reads O(ε⁻² log(1/δ)) values regardless of the context length, and no adaptive method reads fewer, up to constants. A 4-bit scan with deterministic bounds, shared across a grouped-query-attention group, generates the candidates exactly but reads every key.

### Key Findings

- On three LLMs and ten settings at 16K–128K tokens, **no certificate is violated**.
- At equal keys read, the 90th-percentile error is **4.6–7.9× lower** than that of the best of 15 baseline families. At equal bytes, the best baselines have 0.6–1.6 times its 90th-percentile error and at least 4.9 times its largest.
- End to end, it stays **within one point of dense attention** on tasks and **within 0.1% in PG-19 perplexity**.
- On CPU, decoding steps at 32K–128K tokens run **1.1–1.4× faster** than dense attention on the same 16-bit cache.

**Full text:** [Download the PDF](/files/certified-sparse-attention.pdf)
