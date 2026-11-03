---
title: "Trust but Verify: Diagnosing Causal Prior-Data Fitted Networks and Testing When to Defer"
collection: publications
category: preprints
permalink: /publication/2026-11-causal-pfn-defer
excerpt: 'A diagnostic study of three released causal PFNs and a retraining-free wrapper that defers to classical estimators only where the PFN shows certified excess risk. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/causal-pfn-diagnosis-defer.pdf'
bibtexurl: '/files/bibtex/li2026causalpfn.bib'
citation: 'Runguo Li. (2026). &quot;Trust but Verify: Diagnosing Causal Prior-Data Fitted Networks and Testing When to Defer.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/causal-pfn-diagnosis-defer.pdf)

### Author

Runguo Li (single author).

### Abstract

Causal prior-data fitted networks estimate conditional average treatment effects in one forward pass, but their outputs do not reveal whether the data fit the pretraining prior or satisfy the identifying assumptions behind their guarantees. This paper diagnoses the three released causal PFNs on a 50-cell simulation platform and four benchmarks. A generic TabPFN inside a doubly robust learner is more accurate than all three causal PFNs on most designs. It also proposes a retraining-free wrapper: in-context cross-fitting yields a paired doubly robust estimate of the PFN's excess CATE risk over classical references, and a region-wise one-sided Holm test replaces the PFN only where it is significantly worse. A certificate covers only the tested comparison; it does not establish that the PFN is accurate.

### Key Findings

- A generic **TabPFN-DR** learner is more accurate than all three released causal PFNs on most designs, providing a necessary baseline for causal pretraining.
- CausalPFN's uncertainty is largely aleatoric: its 90% CATE intervals cover **1.00 in every cell**, while its 95% ATE intervals cover 0.55 in the base design.
- No native signal reliably flags violated overlap, and a confounding bias in CausalPFN and CausalFM does not vanish as sample size grows.
- Under a protocol frozen before the wrapper was built, the result is mixed: the method is non-inferior on all four benchmarks and significantly more accurate on ACIC-2016 and the held-out suite, but misses the pre-registered detection-AUROC target.
- Post-verdict analysis traces much of the harmful deferral to doubly robust bias when overlap fails.

**Full text:** [Download the PDF](/files/causal-pfn-diagnosis-defer.pdf)
