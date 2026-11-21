---
title: "ComposeOPD: Composing Diffusion Teachers and Published Students, Screened Before Training"
collection: publications
category: preprints
permalink: /publication/2026-11-composeopd
excerpt: 'ComposeOPD distills a composed target built from every model on a diffusion backbone, reward experts and published students alike, and screens the composition weights by sampling before any training.'
date: 2026-11-16
venue: 'Under review'
citation: '&quot;ComposeOPD: Composing Diffusion Teachers and Published Students, Screened Before Training.&quot; Manuscript under review, 2026.'
---

**Status:** Manuscript under review · **Project page:** [ComposeOPD](/projects/2026-11-composeopd/)

### Abstract

Reward fine-tuning has produced a growing zoo of diffusion models on a shared backbone, from single-reward experts to
published distilled students, each strongest on a different kind of prompt. On-policy distillation (OPD) can merge such
models into one student, but it usually routes each prompt to a single teacher, so every model's strengths stay confined
to its own domain. ComposeOPD distills a composed target instead: the velocity of a receiver model plus weighted velocity
deltas of the other models on the backbone. Because an OPD student learns to reproduce its target, sampling the composed
target previews the student, so the weights are screened at inference before any training. The previews track the trained
students with rank correlations of 0.85 across 49 comparisons and 0.98 on compositions trained after the screen was fixed.

### Key Findings

- One composed target covers routing, task arithmetic, convex mixing and teacher extrapolation, and admits any model on the
  backbone as a teacher, published students included, without their data or rewards.
- Additive composition is exact only in special cases; the safe weights depend on the teacher set and its guidance scale.
- An inference-only probe, about one GPU-hour per donor, previews the trained student's image quality (Spearman 0.94).
- Reward experts composed this way raise image quality across domains at unchanged accuracy.
- A student composed from Self-OPD and the DiffusionOPD teachers matches or exceeds four published OPD students on GenEval,
  OCR and PickScore, and a VLM judge prefers it to each.
