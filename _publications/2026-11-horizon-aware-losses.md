---
title: "What Horizon-Aware Losses Do in Multi-Step Forecasting: The L1 Norm, Not Decorrelation"
collection: publications
published: false
category: preprints
permalink: /publication/2026-11-horizon-aware-losses
excerpt: 'A controlled decomposition shows that recent horizon-aware forecasting losses gain from the L1 norm, not from decorrelating forecast horizons. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/horizon-aware-losses.pdf'
bibtexurl: '/files/bibtex/li2026horizon.bib'
citation: 'Runguo Li. (2026). &quot;What Horizon-Aware Losses Do in Multi-Step Forecasting: The L1 Norm, Not Decorrelation.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/horizon-aware-losses.pdf)

### Author

Runguo Li (single author).

### Abstract

Several recent losses for direct multi-step forecasting, including FreDF, Time-o1, and QDF, are motivated by the claim that autocorrelation among future labels biases mean-squared-error training. This paper shows that the claim does not hold for quadratic or orthogonally transformed losses, which share the population minimizer of MSE and, for least squares on a shared design, its solution. Re-weighting output directions only redistributes regularization. A controlled decomposition of the official FreDF and Time-o1 code shows that their empirical gains come from the L1 norm: L2 versions in the same bases match or trail MSE. The size of the gain depends mainly on whether the model normalizes each input window.

### Key Findings

- Autocorrelation among future labels does not bias the population minimizer of MSE or orthogonally transformed losses.
- FreDF and Time-o1 are effectively L1 losses in rotated bases; replacing the L1 term with L2 removes most of the reported gain.
- Without input-window normalization, trimming the largest-error windows from MSE recovers **42% to all** of the gain, whereas clipping the global gradient norm recovers little.
- With normalization, a smaller gain remains on iTransformer, mostly in forecast level; its mechanism is left open.
- Under rolling origins, whether a frequency basis helps depends on the backbone, and validation-period loss is an unreliable way to choose between MSE and a robust loss.

**Full text:** [Download the PDF](/files/horizon-aware-losses.pdf)
