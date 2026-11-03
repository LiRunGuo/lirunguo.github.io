---
title: "What Survives Distillation? A Pre-registered Study of Training-Free Acceleration for Few-Step and Autoregressive Video Diffusion"
collection: publications
category: preprints
permalink: /publication/2026-11-gmsa-distillation
excerpt: 'A pre-registered study of why training-free video-diffusion accelerators calibrated on 50-step teachers mislead on distilled few-step and autoregressive students. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/gmsa-distillation-acceleration.pdf'
bibtexurl: '/files/bibtex/li2026gmsa.bib'
citation: 'Runguo Li. (2026). &quot;What Survives Distillation? A Pre-registered Study of Training-Free Acceleration for Few-Step and Autoregressive Video Diffusion.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/gmsa-distillation-acceleration.pdf)

### Author

Runguo Li (single author).

### Abstract

Video diffusion models are deployed as distilled few-step and autoregressive students, yet the training-free accelerators paired with them were designed on 50-step teachers. This paper measures what survives distillation on six released students in three model families, each compared with its own teacher, and tests pre-registered predictions on a held-out family, scale, prompts, and GPU. The results show that distillation changes the sampling grid and the surrounding system, not the features: teacher-calibrated cache and sparsity signals can mis-rank where a student can skip. A latency probe and a small prompt sweep measured directly on the student provide a better baseline.

### Key Findings

- At matched noise level, a student's attention is as concentrated as its teacher's, and its features are closer to its teacher's than to its own features one step away.
- Wider steps account for lost redundancy but amplify early errors; the VAE decoder and attention kernel cap end-to-end gains.
- Teacher-calibrated cache signals mis-rank student skips, and block-sparse attention loses to dropping a step at equal speed.
- Two calls on the grid, a small decoder, and compilation reach **2.5–3.5×** end-to-end speedup on four students, within 1% of the probe's prediction.
- The probe, protocol, and registration record are released.

**Full text:** [Download the PDF](/files/gmsa-distillation-acceleration.pdf)
