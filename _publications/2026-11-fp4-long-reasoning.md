---
title: "What FP4 Costs Long Reasoning on Blackwell, Measured Above the Noise"
collection: publications
published: false
category: preprints
permalink: /publication/2026-11-fp4-long-reasoning
excerpt: 'Repeated calibration draws, paired tests, and accuracy-budget curves show that FP4 mainly changes reasoning length and memory rather than delivering stable accuracy gains. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/fp4-long-reasoning.pdf'
bibtexurl: '/files/bibtex/li2026fp4reasoning.bib'
citation: 'Runguo Li. (2026). &quot;What FP4 Costs Long Reasoning on Blackwell, Measured Above the Noise.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/fp4-long-reasoning.pdf)

### Author

Runguo Li (single author).

### Abstract

Blackwell GPUs run NVFP4 natively, but reasoning models must remain accurate over tens of thousands of generated tokens. The single-checkpoint, single-benchmark comparisons commonly used to rank quantization methods are often noisier than the gaps they report. This paper measures Qwen3-1.7B, Qwen3-8B, and Qwen3-14B on the kernels deployed by vLLM on an RTX 5090 and a B200, using repeated calibration draws, paired tests, and exact accuracy-versus-budget curves. The effects that survive statistical control concern length and memory more than accuracy.

### Key Findings

- Calibrating GPTQ on reasoning traces removes most of the additional generation length caused by chat calibration on Qwen3.
- No published NVFP4 recipe beats trace-calibrated GPTQ, and few accuracy differences among 8B FP4 checkpoints survive multiple-comparison correction.
- For trace-calibrated 8B weights, activation error is approximately uniform over tokens, and an FP8 KV cache is nearly equivalent to BF16 within measurement resolution.
- NVFP4 KV caching collapses the 1.7B model, matching the failure predicted by its teacher-forced error.
- Long-trace serving on a 32 GB card is capacity-bound; a simple KV-capacity model predicts serving rate and transfers from 8B to 14B.

**Full text:** [Download the PDF](/files/fp4-long-reasoning.pdf)
