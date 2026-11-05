---
title: "What Do Group-Conditional Coverage Metrics Measure? Identification, Inference and Auditing for Conformal Evaluation"
collection: publications
published: false
category: preprints
permalink: /publication/2026-11-group-coverage
excerpt: 'Separating expected coverage, calibration-draw realized coverage, and test noise shows which group-conditional conformal metrics are identifiable and how to audit them. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/group-conditional-coverage.pdf'
bibtexurl: '/files/bibtex/li2026groupcoverage.bib'
citation: 'Runguo Li. (2026). &quot;What Do Group-Conditional Coverage Metrics Measure? Identification, Inference and Auditing for Conformal Evaluation.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/group-conditional-coverage.pdf)

### Author

Runguo Li (single author).

### Abstract

Conformal methods for group-conditional or class-conditional coverage are commonly compared using empirical coverage gaps, the fraction of under-covered groups, or worst-group coverage. When groups are many and small, much of what these metrics report is binomial test noise. This paper separates a method's expected coverage, the coverage realized by a single calibration draw, and test noise, then studies what the data identify about realized coverage. It gives closed-form noise floors, sharp identified sets for under-covered fractions and worst groups, and a debiased estimator with valid inference for the root-mean-square coverage gap. A pre-registered evaluation on seven datasets, seven methods, and 200 data splits quantifies how often reported gaps are noise or calibration variance.

### Key Findings

- The fraction of under-covered groups and the worst-group coverage are **not identified** when groups contain few test points; the paper gives their sharp identified sets.
- The RMS coverage gap is point-identified, enabling a debiased estimator, rate-optimal intervals, finite-sample bounds, and paired method comparisons.
- Exact binomial tests with Benjamini–Hochberg control can flag individual groups with finite-sample false-discovery-rate guarantees.
- In the pre-registered evaluation, test noise accounts for a median **41%** of the naive mean-squared coverage gap, ranging from 5% to nearly all of it.
- A method with exact group coverage would still show a median **62%** of the reported coverage gap.

**Full text:** [Download the PDF](/files/group-conditional-coverage.pdf)
