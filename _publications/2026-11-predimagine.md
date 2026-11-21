---
title: "PredImagine: Latent Visual Imagination by Clean-Target Prediction, and What Flow Matching Adds"
collection: publications
category: preprints
permalink: /publication/2026-11-predimagine
excerpt: 'A controlled protocol for latent visual imagination in VLMs exposes two hidden failures of flow-matching producers; PredImagine, a clean-target recipe, repairs both and shows that generation adds robustness to the noise draw, not accuracy. Under review.'
date: 2026-11-16
venue: 'Under review'
---

**Status:** Manuscript under review · **Project page:** [PredImagine](/projects/2026-11-predimagine/)

### Abstract

Vision-language models can reason about space by imagining an intermediate visual state as latent tokens before they answer, and flow-matching generators of these latents are credited with large gains in accuracy. Yet the generative objective has not been separated from the architecture that carries it, so it remains unclear what actually makes latent imagination work. We isolate it in WM-VLM with a controlled protocol, which we will release: only the producer of the latent changes, every producer is read by one consumer trained on oracle latents, and new rotation tasks fix how many intermediate states are valid. The protocol exposes two hidden failures. The released model behaves as a fixed-noise regressor, whose accuracy falls from 0.87 to chance when only its noise seed changes, and a default rectified flow largely ignores the question. From the flow's one-step optimum we derive PredImagine, a simple recipe that trains the producer to predict the clean intermediate state, concentrating training where the question decides the target. With no new parameters and no inference cost, PredImagine turns the failing flow into a working imaginer, lifts our 7B reproduction to the release's accuracy and, as a flow, holds it under every noise draw we tested, where the release collapses. Its regression and flow variants then show what generation contributes: robustness to the noise draw, but neither accuracy nor the ability to commit to one of several valid states. Our results suggest that latent visual imagination needs prediction more than generation.

### Key Findings

- The released WM-VLM-7B reads **0.870** on Tetris-2D at noise seed 0 and **0.233** at seed 1: it behaves as a fixed-noise regressor.
- The default rectified flow stays near chance; at no flow time does its output depend on the question by more than **2.4%**.
- PredImagine's flow variant keeps its accuracy at a fresh noise draw on every task (7B Tetris-2D: **0.798 → 0.810**), where every fixed-noise regressor falls to chance.
- At its training draw the regression variant is as accurate in distribution; with several valid intermediates, the flow's noise does not select one.
