---
title: "How Much Does Random Rotation Help GPTQ? Sharp Worst-Case Rate Gaps in the High-Rate Regime"
collection: publications
category: preprints
permalink: /publication/2026-10-gptq-rotation
excerpt: 'Viewing GPTQ as Babai''s nearest-plane algorithm, this paper gives the sharp worst-case rate gap between randomly rotated GPTQ and WaterSIC, answering an open question of Ordentlich and Polyanskiy. Single-author work under review at ICML 2027.'
date: 2026-10-27
venue: 'Under review — ICML 2027'
paperurl: '/files/gptq-random-rotation.pdf'
bibtexurl: '/files/bibtex/li2026gptqrotation.bib'
citation: 'Runguo Li. (2026). &quot;How Much Does Random Rotation Help GPTQ? Sharp Worst-Case Rate Gaps in the High-Rate Regime.&quot; Manuscript under review.'
---

**Status:** Manuscript under review at ICML 2027 · **PDF:** [Download](/files/gptq-random-rotation.pdf)

### Author

Runguo Li (single author).

### Abstract

GPTQ is Babai's nearest-plane algorithm on the lattice of the Cholesky factor of the layer's input second moment Σ. In the high-rate, entropy-coded regime it needs ½ log₂(AM/GM) of its Cholesky pivots more bits than WaterSIC, which is within 0.255 bit of the rate-distortion limit for every Σ. Random rotations close most of this gap in practice, and Ordentlich and Polyanskiy asked for the worst-case gap between rotated GPTQ and WaterSIC.

We show that after a Haar rotation the *k*-th pivot converges almost surely, uniformly in *k* ≥ δ*n* and over spectra of bounded condition number, to a classical formula from the theory of Lyapunov exponents, and use it to answer the question.

### Key Findings

- Over spectra with condition number κ, the worst rotated gap is ½ log₂ κ − log₂ ln κ + ½ − 1/ln 2 + o(1) bits, attained only by two-point spectra — just ½ log₂ ln κ + O(1) bits below the worst case without rotation, so **rotation cannot replace waterfilling in general**.
- It nearly can on real layers: rotation cuts the cost of an η-fraction of outlier eigenvalues of size μ from about ημ to **η²μ/2** to leading order, but cannot remove spectral spread on a logarithmic scale; for non-Gaussian weights it also adds a weight-side rate cost, characterized exactly under a standard high-rate assumption.
- On **96 Hessians** of Qwen3 and Llama-3.1, the rotated gap has median **0.04 bit** and maximum 0.36 bit, against 2–5 bits in the worst case for the same condition numbers, and the limit matches it to within 0.009 bit.

**Full text:** [Download the PDF](/files/gptq-random-rotation.pdf)
