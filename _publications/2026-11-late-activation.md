---
title: "Late Activation: A Better Reward–Diversity Trade-off for RL-Tuned Flow Models"
collection: publications
category: preprints
permalink: /publication/2026-11-late-activation
unlisted: true
excerpt: 'RL tuning collapses the variety of text-to-image flow models in the first, high-noise sampling steps, where the reward is largely redundant; letting the base model take those steps keeps 96% of the reward gain with 25% higher CLIP diversity, at no extra cost.'
date: 2026-11-16
venue: 'Manuscript'
---

**Status:** Manuscript

### Abstract

Online reinforcement learning (RL) has become a standard way to align text-to-image flow models with human preferences, but it erodes a property users value as much as quality: variety. After RL tuning, the images for one prompt converge on one composition; on SD3.5-Medium tuned with DiffusionNFT, held-out CLIP diversity falls to 38% of the base model's. Switching the adapter off by noise level shows that the collapse happens in the first few high-noise steps, while the reward added there is largely redundant. Training needs these steps; sampling does not. **Late activation** lets the base model compose the image and hands over to the adapter below a noise level τ, with no retraining and no extra network evaluations. It keeps 96% of NFT's reward gain with 25% higher CLIP diversity and 23% less drift, beats early stopping and velocity or LoRA-weight scaling at matched reward, matches annealed guidance at half its sampling time, and holds across three rewards and four RL methods. Paired with a stronger training tilt, it improves on default NFT in reward and diversity at once, over ten training seeds per arm.

[Project page](/projects/2026-11-late-activation/)
