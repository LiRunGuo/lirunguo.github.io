---
title: "StepShare: Diagnosing and Rebalancing the Generation Gap in Native Unified Multimodal Models"
collection: projects
permalink: /projects/2026-11-stepshare/
date: 2026-11-16
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "At equal loss weights, generation in a native unified model is starved rather than conflicted: it gets under 1% of each AdamW step. StepShare measures and rebalances that share; StepShare-Adam closes 91% of the FID gap with no hidden probe cost and no added parameters."
teaser: "/images/projects/2026-11-stepshare/teaser.png"
tags: ["unified multimodal models", "multi-task optimisation", "flow matching", "AdamW"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-stepshare"
---

Unified multimodal models read and generate images with one backbone, and the generation gap that joint training opens is usually blamed on parameter sharing. **StepShare** splits every AdamW step exactly into an understanding part and a generation part, by depth, and shows that at equal loss weights generation is *starved rather than conflicted*: the two gradients are nearly orthogonal, but the understanding gradient is about 100 times larger, so generation receives under 1% of the update. Restoring the share closes most of the FID gap without added parameters, more than task-specific routing, yet quietly lowers intermediate-layer probe accuracy by up to 7 points. **StepShare-Adam**, which restores the share with per-task Adam and allocates it by depth, closes 91% of the gap with essentially no probe cost.

## Key results

![Figure 1: StepShare at equal loss weights, restoring the share, and the hidden cost by layer.](/images/projects/2026-11-stepshare/fig1-teaser.png){: .align-center}

**Figure 1.** At equal weights the two gradients are nearly orthogonal and generation gets 0.9% of the AdamW step. Lowering the understanding weight restores the share and closes most of the FID gap, more than task routing does at equal weights, but costs up to 5.4 probe points in intermediate layers that the last layer and caption NLL do not show. StepShare-Adam closes the gap without that cost.

![Figure 2: StepShare measures the share of each AdamW step and rebalances it.](/images/projects/2026-11-stepshare/fig2-stepshare.png){: .align-center style="max-width: 560px; width: 100%;"}

**Figure 2.** Step 1 splits each AdamW step exactly into Δ<sub>u</sub> + Δ<sub>g</sub>, in the shallow and deep halves. Step 2 restores generation's share and then allocates it by depth: StepShare-Adam gives generation a large share of the deep step and a small one of the shallow step, which keeps the FID gain and removes the probe drop.

| Arm (0.6B, w = understanding weight) | Seeds | FID gap closed, guidance 1 | FID gap closed, guidance 4 | Worst COCO probe layer | Worst IN-100 probe layer |
|---|---|---|---|---|---|
| Dense, w = 1 | 5 | 0% (reference) | 0% (reference) | −0.5 | −0.3 |
| Dense, w = 0.03 | 5 | 82 ± 12% | 78 ± 10% | −5.4 | −5.1 |
| Task routing, w = 0.03 | 5 | 93 ± 5% | 96 ± 5% | −2.9 | −2.8 |
| Per-task Adam, w = 1 | 3 | 92 ± 9% | 97 ± 5% | −3.6 | −3.2 |
| **StepShare-Adam**, w = 1 | 5 | **91 ± 14%** | **89 ± 8%** | **−0.5** | **−0.4** |

**Table 2 (excerpt).** Reweighting and per-task Adam close most of the gap but cost 3 to 5 points in intermediate layers; StepShare-Adam keeps the gain, removes the cost, and adds no parameters. Recovery is mean ± 95% CI over seeds, paired by seed; probe columns are points against understanding-only training of the same seed.

![Figure 3: Gradient ratio during training and the trade-offs of restoring the share.](/images/projects/2026-11-stepshare/fig3-budget.png){: .align-center}

**Figure 3.** At equal weights the gradient ratio rises from 11 to 60 as the generation loss falls, and FID recovery follows the training-average ratio across loss weights and clipping schemes. Along the weight sweep, more recovery comes with a deeper probe deficit; depth allocation and StepShare-Adam leave that trade-off.

![Figure 4: Samples from the same captions and initial noise.](/images/projects/2026-11-stepshare/fig4-samples.webp){: .align-center}

**Figure 4.** Every row uses the same captions and initial noise. At equal weights the dense model produces saturated high-frequency texture; restoring generation's share with StepShare-Adam gives layouts close to generation-only training.

## Interactive demo

Pick an arm to see how much of the FID gap it closes and in which layers its probe cost sits; *Step split* shows StepShare's measurements, and *Samples* compares sample grids generated from the same noise.

<iframe src="/files/projects/2026-11-stepshare/demo/index.html" width="100%" height="900" loading="lazy" title="StepShare interactive explorer" style="border: 0;"></iframe>

## Links and citation

- **Paper:** under review; the PDF and BibTeX will be linked here when they are public.
- **Code:** will be released with the paper.
