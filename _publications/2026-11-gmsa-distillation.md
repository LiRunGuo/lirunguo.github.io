---
title: "StudentProbe: What Survives Distillation in Training-Free Video Diffusion Acceleration"
collection: publications
category: preprints
unlisted: true
permalink: /publication/2026-11-gmsa-distillation
excerpt: 'StudentProbe measures distilled few-step and autoregressive video students against their own teachers: students keep their teachers&apos; features, while the few wide steps and the decoder decide which training-free accelerators still pay off. Single-author work.'
date: 2026-11-02
venue: 'Preprint'
paperurl: '/files/gmsa-distillation-acceleration.pdf'
bibtexurl: '/files/bibtex/li2026gmsa.bib'
citation: 'Runguo Li. (2026). &quot;StudentProbe: What Survives Distillation in Training-Free Video Diffusion Acceleration.&quot; Preprint.'
---

**Status:** Preprint · **PDF:** [Download](/files/gmsa-distillation-acceleration.pdf) · **Project page:** [StudentProbe](/projects/2026-11-gmsa-distillation/)

### Author

Runguo Li (single author).

### Abstract

Video generation is increasingly served by distilled students—few-step and autoregressive models that produce a clip in a handful of network calls—yet the training-free accelerators deployed on them, from feature caches to sparse attention, were designed and calibrated on 30–50-step teachers. Whether these accelerators survive distillation has so far been assumed rather than measured. We introduce StudentProbe, to our knowledge the first protocol that measures each student against its own teacher under controlled conditions: at matched noise level, on the states the student actually visits and on one GPU type, with predictions registered before held-out tests on a new family, scale, prompt set and GPU. Recent work concludes that distillation leaves little redundancy or sparsity to exploit; StudentProbe shows the opposite at the level of the network: students keep their teachers' attention concentration and features. What changes is the sampling grid, whose few wide steps remove step-to-step redundancy and amplify early errors, and the system, where the decoder and the attention kernel now cap the gain. This single picture explains why cache signals calibrated on teachers mis-rank the steps a student can skip, and why block-sparse attention over position blocks, even with oracle selection or compensation, drifts further than running one step fewer at equal latency. StudentProbe replaces teacher-side calibration with a few minutes of measurement on the student that predicts the speedup and the lowest-drift skip positions; guided by it, released components set a reference point of 2.5–3.5× end to end on four students, at a measured cost in motion (all four) and aesthetics (three) on untouched prompts. We will release StudentProbe and its full registration record.

### Key Findings

- At matched noise level, a student's attention is as concentrated as its teacher's, and on the same states its features are closer to its teacher's than to its own features one call later.
- The few wide steps amplify early errors, and the VAE decoder and the attention kernel cap end-to-end gains.
- Teacher-calibrated cache signals mis-rank where a student can skip: tuned on the student, FBCache still drifts 13× as far on the held-out LTX-Video student and lowers aesthetics on AnyFlow and rCM.
- Block-sparse attention over position blocks drifts further than one call fewer at equal latency, even with oracle selection or compensation.
- Two calls, a lightweight decoder and compilation reach **2.5–3.5×** end to end on four students, within 1% of the probe's prediction.

**Full text:** [Download the PDF](/files/gmsa-distillation-acceleration.pdf) (paper and supplementary material)
