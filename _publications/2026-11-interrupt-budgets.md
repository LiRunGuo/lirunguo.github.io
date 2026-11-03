---
title: "Interrupt Budgets for Pre-Execution AI Control Monitors"
collection: publications
category: preprints
permalink: /publication/2026-11-interrupt-budgets
excerpt: 'Replacing a per-step false-positive target with a trajectory-level false-interruption budget exposes both the cost and the limits of pre-execution AI control monitors. Single-author work under review.'
date: 2026-11-02
venue: 'Under review'
paperurl: '/files/interrupt-budgets-ai-control.pdf'
bibtexurl: '/files/bibtex/li2026interruptbudgets.bib'
citation: 'Runguo Li. (2026). &quot;Interrupt Budgets for Pre-Execution AI Control Monitors.&quot; Manuscript under review.'
---

**Status:** Manuscript under review · **PDF:** [Download](/files/interrupt-budgets-ai-control.pdf)

### Author

Runguo Li (single author).

### Abstract

LLM judges deployed as pre-execution monitors read each proposed action of an AI agent and can block it, but they are usually tuned to a per-step false-positive rate. A deployer experiences something else: alarms compound over a trajectory, so on LinuxArena a 1% per-step threshold interrupts 14–37% of benign runs across 12 monitor models. This paper instead bounds the trajectory-level false interruption rate (FIR). Monitors calibrated to that budget catch less on hard attacks: at a 5% budget, the catch rate of Claude Opus 4.6 on human-crafted sabotage falls from 0.79 to 0.59. Conformal or PAC thresholds calibrated on the deployed agent guarantee the budget, while thresholds calibrated on another agent or environment can exceed it. No statistic tested reliably beats the running maximum.

### Key Findings

- A per-step 1% threshold produces **14–37%** false interruption rates across 12 monitor models on LinuxArena.
- At a 5% trajectory-level budget, the Claude Opus 4.6 catch rate on human-crafted sabotage drops from **0.79 to 0.59**.
- Conformal and PAC calibration can certify the budget when calibrated on the deployed agent, but calibration does not transfer reliably across agents or environments.
- Likelihood-ratio statistics hurt several monitors, provide no gain once threshold uncertainty is counted, and collapse when attack evidence is concentrated in one action.
- When attacks are caught, the median alarm often occurs only at the harmful action, leaving no time for non-blocking review.

**Full text:** [Download the PDF](/files/interrupt-budgets-ai-control.pdf)
