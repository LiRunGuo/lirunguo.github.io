---
title: "Predict, Don't Just Pair: Lossless Storage for Families of Fine-Tuned Models"
collection: publications
category: preprints
permalink: /publication/2026-11-tessera-lossless-storage
excerpt: 'Tessera treats lossless checkpoint storage as prediction and coding of residuals, not only base selection, storing 25.3% of deduplicated bytes across 130 fine-tunes. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/tessera-lossless-storage.pdf'
bibtexurl: '/files/bibtex/li2026tessera.bib'
citation: 'Runguo Li. (2026). &quot;Predict, Don''t Just Pair: Lossless Storage for Families of Fine-Tuned Models.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/tessera-lossless-storage.pdf)

### Author

Runguo Li (single author).

### Abstract

Model hubs are versioned stores of millions of near-identical fine-tuned checkpoints. Existing lossless systems largely treat storage as a pairing problem, deciding which stored tensor each new tensor should be delta-encoded against. This paper revisits that view using exhaustive ground truth from 130 fine-tunes of four model families, comprising 817 GB after deduplication, and compresses every candidate tensor pair to compute the exact optimum of base selection. Most remaining redundancy lies in how the residual is predicted and coded. Tessera combines exponent-conditioned entropy coding, bit-reproducible low-rank prediction for merged adapters, and popularity-aware delta-chain planning.

### Key Findings

- Near-optimal pairing is easy to reach; most remaining storage redundancy is in residual prediction and coding rather than base selection.
- Tessera stores **25.3%** of deduplicated bytes, versus 47.6% for a TensorDex reimplementation and 36.3% for the original ZipLLM.
- Its **74.7% reduction** is 14 points above the exact optimum achievable by pairing alone.
- Replayed online on three model families, storage stays within **0.2–1.4 points** of the offline plan and ingests about 1.8 GB/s from local disk.
- A 6 GB model reconstructs in **1.5–3.7 s on CPU** and **0.05–0.12 s on GPU**.

**Full text:** [Download the PDF](/files/tessera-lossless-storage.pdf)
