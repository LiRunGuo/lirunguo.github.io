---
title: "RefAudit: Re-evaluating Diffusion Step Caching as Step Scheduling Against Converged References"
collection: publications
category: preprints
unlisted: true
permalink: /publication/2026-11-refaudit
excerpt: "The standard protocol for scoring training-free step caching rewards samplers that share the 50-step reference's error; RefAudit separates accuracy from shared error, and under it optimized static schedules with UniPC match or beat every cache we ran on FLUX.1-dev and SD3.5."
date: 2026-11-16
venue: 'Preprint'
citation: 'Runguo Li. (2026). &quot;RefAudit: Re-evaluating Diffusion Step Caching as Step Scheduling Against Converged References.&quot; Preprint.'
---

**Status:** Preprint · **Project page:** [RefAudit](/projects/2026-11-refaudit/) · **Full text:** the PDF and BibTeX will be linked here when the preprint is public.

### Author

Runguo Li (single author).

### Abstract

Training-free step caching, which skips the transformer at some sampling steps and reuses its last result, is a popular way to accelerate diffusion transformers. Methods such as TeaCache and MagCache are judged by their fidelity to the default 50-step output and compared with sampling at fewer uniform steps. We show that this protocol favors caching, for two reasons. First, the 50-step output is itself a coarse ODE solve, and fidelity to it rewards samplers that share its error rather than samplers that are accurate. Second, under Euler, reusing the last velocity is exactly Euler on the computed steps, and reusing residuals nearly so: step caching is step scheduling, and its fair baseline is an optimized timestep schedule, which the recent caching papers we examined leave out. We propose RefAudit, an evaluation protocol that (i) splits the standard score exactly into the error to a converged reference, a constant and a cross term, the error shared with the reference, (ii) accepts a reference only after checking it against a solver of another family, and (iii) compares methods at equal NFE under the same solver. RefAudit changes the conclusions. Swapping only the solver family of the 50-step reference shifts the gap between two fixed samplers by up to 18 dB and can reverse its sign. On FLUX.1-dev and SD3.5, optimized static schedules with UniPC match or beat every cache we ran, with no run-time decisions or extra memory, and on Wan2.1 they do so on a grid with a tuned time shift. RefAudit also shows that PSNR alone can select a grid with a visible artifact. Its script needs only saved outputs, so any fast sampler can be audited against its reference in minutes.

### Key Findings

- Swapping only the solver family of the 50-step reference shifts the gap between two fixed samplers by up to 18 dB (18.44 dB on Wan2.1), and can reverse its sign.
- Under Euler, reusing the last velocity is exactly Euler on the computed steps, so the fair baseline for a cache is an optimized step schedule.
- On FLUX.1-dev and SD3.5, optimized static schedules with UniPC are 1.0–3.1 dB closer to the converged reference than every cache as shipped, and move from 4th to 1st in the ranking at FLUX-15 and SD3.5-14.
- PSNR alone can select a grid with a visible periodic artifact, which the RefAudit check exposes.
