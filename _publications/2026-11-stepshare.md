---
title: "StepShare: Diagnosing and Rebalancing the Generation Gap in Native Unified Multimodal Models"
collection: publications
category: preprints
unlisted: true
permalink: /publication/2026-11-stepshare
excerpt: "At equal loss weights, generation in a native unified model is starved rather than conflicted: it gets under 1% of each AdamW step. StepShare measures and rebalances that share; StepShare-Adam closes 91% of the FID gap with no hidden probe cost and no added parameters."
date: 2026-11-16
venue: 'Preprint'
---

**Status:** Preprint · **Project page:** [StepShare](/projects/2026-11-stepshare/)

### Author

Runguo Li

### Abstract

Unified multimodal models increasingly read and generate images with a single backbone, yet joint training often degrades generation, and recent controlled studies answer with architecture: give the two tasks separate parameters. These comparisons are made at a single loss weighting, so they cannot tell whether generation suffers from sharing parameters or from how the optimiser divides each update between the two tasks. We present StepShare, a simple tool that makes this division visible and correctable: it splits every AdamW step exactly into each task's share, by depth, and rebalances that share. Applied to a native unified model on raw pixels (Qwen3-0.6B and 1.7B, five seeds for the key arms, criteria fixed in advance), StepShare reveals that the generation gap is largely an optimisation artefact rather than a cost of sharing. The two gradients are nearly orthogonal, but the understanding gradient is about a hundred times larger, so at equal loss weights generation receives under 1% of the update's squared norm: it is starved rather than conflicted. Restoring its share closes most of the gap with no additional parameters, exceeding what task-specific routing achieves, on top of which routing adds nothing, and a generation-only model given an equally small update reproduces much of the damage. StepShare also uncovers a hidden cost that standard understanding metrics miss, a drop of up to 7 points in intermediate-layer probe accuracy, and StepShare-Adam, which restores the share with a per-task Adam state and allocates it by depth, closes 91% of the gap with essentially no probe cost and no additional parameters. Because the share depends on the training recipe, comparisons at one loss weight can substantially over-credit separation, whose added value is small after rebalancing, though it grows with longer training; a gradient imbalance of the same order appears in BAGEL-7B-MoT. We therefore propose StepShare as a standard companion to any comparison of shared and separated designs for unified models.
