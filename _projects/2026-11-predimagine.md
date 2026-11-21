---
title: "PredImagine: Latent Visual Imagination by Clean-Target Prediction, and What Flow Matching Adds"
collection: projects
permalink: /projects/2026-11-predimagine/
date: 2026-11-16
status: "Under review"
venue: "Under review"
excerpt: "A controlled protocol for latent visual imagination in VLMs exposes two hidden failures of flow-matching producers; PredImagine, a clean-target recipe derived from the flow's one-step optimum, repairs both and shows that generation adds robustness to the noise draw, not accuracy."
teaser: "/images/projects/2026-11-predimagine/teaser.png"
tags: ["vision-language models", "latent reasoning", "flow matching"]
publication: "/publication/2026-11-predimagine"
# paperurl, pdfurl, codeurl, bibtexurl: add when the paper and code are public
---

## Summary

Vision-language models can reason about space by imagining an intermediate visual state as latent tokens before they answer, and flow-matching generators of these latents are credited with large gains. We isolate the generative objective in WM-VLM with a controlled protocol: only the producer of the latent changes, every producer is read by one consumer trained on oracle latents, and new rotation tasks fix how many intermediate states are valid. The protocol exposes two hidden failures: the released model behaves as a fixed-noise regressor whose accuracy falls from 0.87 to chance when only its noise seed changes, and a default rectified flow largely ignores the question. **PredImagine** trains the producer to predict the clean intermediate state, with training concentrated where the question decides the target. It repairs both failures, lifts our 7B reproduction to the release's accuracy, and as a flow holds that accuracy under every noise draw we tested; its regression and flow variants show that generation contributes robustness to the noise draw, but neither accuracy nor the ability to commit to one of several valid states.

## Key results

![Figure 1: PredImagine at a glance.](/images/projects/2026-11-predimagine/fig1-overview.png){: .align-center}

**Figure 1.** PredImagine changes only the producer's stage-1 training, and every producer is read by one shared consumer. At a fresh noise draw every fixed-noise regressor, the release included, falls to chance, while the flow variant of PredImagine keeps its accuracy on every task (b). With several equally valid intermediates, every producer falls toward chance (c).

![Figure 2: Tasks and their gold intermediates.](/images/projects/2026-11-predimagine/fig2-tasks.png){: .align-center}

**Figure 2.** All tasks are rotation analogies with four options. ImagineBench-rotation sets the number of valid intermediates: one clockwise strip (IB-cw), either of two strips (IB-both), or the answer shape at any of *k* placements (IB-pos-*k*).

**Table 2.** The released WM-VLM-7B on Tetris-2D (accuracy ID / OOD).

| Released WM-VLM-7B, Tetris-2D | K1 | K8 |
|---|---|---|
| Noise seed 0 | 0.870 / 0.736 | 0.738 / 0.620 |
| Noise seed 1 | 0.233 / 0.216 | 0.220 / 0.204 |
| Oracle latents | 0.860 / 0.884 | |
| Zeroed latents | 0.150 / 0.166 | |

The release reproduces its reported accuracy at noise seed 0 and falls near chance at seed 1, the behaviour of a network trained on one fixed noise tensor.

![Figure 3: What the default flow lacks and what PredImagine changes.](/images/projects/2026-11-predimagine/fig3-cond-gap.png){: .align-center}

**Figure 3.** The default flow's output barely depends on the question: at no flow time does matching the question lower its error by more than 2.4%. Single ingredients of PredImagine raise this gap only to 16–59%; a producer becomes readable only when it predicts the clean target and trains half or all of its blocks at *t* = 0, which brings its *t* = 0 error near the regressor's.

**Table 4.** Same consumer for every producer of a task; training draw (K1 s0), mean over trainer seeds, ID / OOD.

| Shared consumer, K1 s0 | *x*-pred. regression | *t* = 0 flow | Repaired flow | Velocity regression |
|---|---|---|---|---|
| Tetris-2D | 0.578 / 0.524 | 0.542 / 0.458 | 0.537 / 0.528 | 0.562 / 0.482 |
| IB-pos1 (*k* = 1) | 0.653 / 0.725 | 0.590 / 0.675 | 0.593 / 0.656 | 0.354 / 0.415 |
| IB-cw (*k* = 1) | 0.768 / 0.791 | 0.628 / 0.655 | 0.698 / 0.765 | 0.721 / 0.775 |
| IB-both (*k* = 2) | 0.598 / 0.642 | 0.468 / 0.518 | 0.483 / 0.550 | 0.649 / 0.672 |

At its training draw, the regression variant of PredImagine is as accurate as the flow variants or more in distribution; the flows' advantage is robustness to the noise draw, which fresh training noise alone provides.

![Figure 5: At k = 2 the flow's noise does not choose a placement.](/images/projects/2026-11-predimagine/fig5-valid-states.png){: .align-center}

**Figure 5.** With two valid placements, two noise draws of the flow land on the same placement for 95% of samples, and its latents carry the answer's rotation at about chance rate. Hypothesis codes with as many codes as valid states each commit to one placement.

## Interactive demo

Switch the noise draw, build each PredImagine variant from its two training choices, and see how many intermediate states each task accepts. All numbers are from the paper.

<iframe src="/files/projects/2026-11-predimagine/demo/index.html" width="100%" height="760" loading="lazy" title="PredImagine interactive demo" style="border:1px solid #e2e1dc;border-radius:8px"></iframe>

## Links and citation

- **Status:** under review.
- **Code:** the protocol, the ImagineBench-rotation generator, the stage-1 configurations of PredImagine and every arm, and the evaluation code will be released.
- **Citation:** to be added with the author list once the paper is public.
