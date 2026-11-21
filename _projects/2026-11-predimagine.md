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

<style>
/* Page-scoped styles for this project page only (prefix pi-) */
.pi-hero { text-align: center; padding: 26px 18px 22px; margin: 0 0 1.6em; border-radius: 18px;
  border: 1px solid var(--global-border-color);
  background: radial-gradient(120% 140% at 0% 0%, rgba(42,120,214,.10), transparent 55%),
              radial-gradient(120% 140% at 100% 100%, rgba(235,104,52,.10), transparent 55%); }
.pi-hero .pi-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; opacity: .65; margin: 0 0 .6em; }
.pi-hero .pi-tagline { font-size: 1.3em; font-weight: 700; line-height: 1.3; margin: 0 auto .9em; max-width: 24em; letter-spacing: -.01em; }
.pi-pred { color: #2a78d6; }
.pi-gen { color: #eb6834; }
.pi-name { background: linear-gradient(90deg, #2a78d6, #eb6834); -webkit-background-clip: text; background-clip: text; color: transparent; }
.page__content .pi-btns { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin: 0; }
.pi-btn, .pi-btn:visited { display: inline-flex; align-items: center; gap: 7px; padding: 6px 15px; border-radius: 999px; font-size: .74em; font-weight: 600;
  background: #1f2328; color: #fff !important; text-decoration: none !important; border: 1px solid #1f2328; transition: transform .15s, box-shadow .15s; }
.pi-btn:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(0,0,0,.18); }
.pi-btn.pi-off { background: transparent; color: inherit !important; border-color: var(--global-border-color); opacity: .7; cursor: default; }
.pi-btn.pi-off:hover { transform: none; box-shadow: none; }
.pi-btn small { font-weight: 500; opacity: .75; }
.pi-tldr { position: relative; padding: .9em 1.1em .9em 1.3em; border-radius: 12px; margin: 0 0 1.4em; font-size: .92em;
  background: rgba(127,127,127,.08); }
.pi-tldr::before { content: ""; position: absolute; left: 0; top: 12px; bottom: 12px; width: 4px; border-radius: 4px; background: linear-gradient(#2a78d6, #eb6834); }
.pi-tldr b:first-child { letter-spacing: .06em; }
.pi-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 0 0 1.8em; }
.pi-stat { border: 1px solid var(--global-border-color); border-radius: 14px; padding: 12px 12px 10px; }
.pi-stat .v { font-size: 1.25em; font-weight: 750; line-height: 1.15; font-variant-numeric: tabular-nums; letter-spacing: -.01em; }
.pi-stat .v .arr { opacity: .5; font-weight: 500; margin: 0 .15em; }
.pi-stat .l { font-size: .68em; line-height: 1.35; opacity: .75; margin-top: 6px; }
.pi-stat .k { display: inline-block; width: 22px; height: 3px; border-radius: 2px; margin-bottom: 8px; }
.page__content img.pi-shot { display: block; margin: .6em auto .4em; padding: 10px; border-radius: 14px; background: #fff;
  border: 1px solid var(--global-border-color); box-shadow: 0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.06); }
.page__content img.pi-narrow { max-width: 74%; }
.page__content p.pi-cap { font-size: .84em; opacity: .9; margin: .2em 0 2em; }
.pi-cap strong:first-child { color: #2a78d6; }
.pi-kicker { display: inline-block; font-size: .62em; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
  border: 1px solid var(--global-border-color); margin: 1.2em 0 0; opacity: .8; }
.page__content table.pi-table { font-size: .74em; border-radius: 12px; }
.page__content table.pi-table td, .page__content table.pi-table th { white-space: nowrap; }
.pi-demo { border-radius: 16px; overflow: hidden; border: 1px solid var(--global-border-color); box-shadow: 0 1px 2px rgba(0,0,0,.05), 0 10px 30px rgba(0,0,0,.08); }
.pi-demo iframe { display: block; width: 100%; border: 0; }
@media (max-width: 700px) {
  .pi-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .page__content img.pi-narrow { max-width: 100%; }
  .pi-hero .pi-tagline { font-size: 1.12em; }
}
</style>

<div class="pi-hero">
  <p class="pi-eyebrow">Vision-language models · latent visual reasoning · flow matching</p>
  <p class="pi-tagline">Latent visual imagination needs <span class="pi-pred">prediction</span> more than <span class="pi-gen">generation</span>.</p>
  <p class="pi-btns">
    <a class="pi-btn" href="#interactive-demo"><i class="fa-solid fa-play" aria-hidden="true"></i> Interactive demo</a>
    <a class="pi-btn" href="#key-results"><i class="fa-solid fa-chart-column" aria-hidden="true"></i> Key results</a>
    <span class="pi-btn pi-off"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i> Paper <small>· soon</small></span>
    <span class="pi-btn pi-off"><i class="fa-brands fa-github" aria-hidden="true"></i> Code <small>· soon</small></span>
  </p>
</div>

<div class="pi-tldr"><b>TL;DR</b> &nbsp;<span class="pi-name"><b>PredImagine</b></span> trains the latent producer of a VLM to predict the clean intermediate state, with training focused where the question decides the target. It repairs two hidden failures of flow-matching imagination, and its own variants show that generation adds robustness to the noise draw, not accuracy.</div>

<div class="pi-stats">
  <div class="pi-stat"><span class="k" style="background:#8a8984"></span><div class="v">.870<span class="arr">→</span>.233</div><div class="l">Released WM-VLM-7B on Tetris-2D when only the noise seed changes</div></div>
  <div class="pi-stat"><span class="k" style="background:#8a8984"></span><div class="v">≤ 2.4%</div><div class="l">How much the default flow's output depends on the question, at any flow time</div></div>
  <div class="pi-stat"><span class="k" style="background:#eb6834"></span><div class="v">.798<span class="arr">→</span>.810</div><div class="l">PredImagine's flow variant at 7B, from the training draw to a fresh one</div></div>
  <div class="pi-stat"><span class="k" style="background:#eb6834"></span><div class="v">95%</div><div class="l">Samples where two noise draws of the flow land on the same placement (k = 2)</div></div>
</div>

## Summary

Vision-language models can reason about space by imagining an intermediate visual state as latent tokens before they answer, and flow-matching generators of these latents are credited with large gains. We isolate the generative objective in WM-VLM with a controlled protocol: only the producer of the latent changes, every producer is read by one consumer trained on oracle latents, and new rotation tasks fix how many intermediate states are valid. The protocol exposes two hidden failures: the released model behaves as a fixed-noise regressor whose accuracy falls from 0.87 to chance when only its noise seed changes, and a default rectified flow largely ignores the question. **PredImagine** trains the producer to predict the clean intermediate state, with training concentrated where the question decides the target. It repairs both failures, lifts our 7B reproduction to the release's accuracy, and as a flow holds that accuracy under every noise draw we tested; its regression and flow variants show that generation contributes robustness to the noise draw, but neither accuracy nor the ability to commit to one of several valid states.

## Key results

<span class="pi-kicker">Overview</span>

![Figure 1: PredImagine at a glance.](/images/projects/2026-11-predimagine/fig1-overview.png){: .pi-shot}

**Figure 1.** PredImagine changes only the producer's stage-1 training, and every producer is read by one shared consumer. At a fresh noise draw every fixed-noise regressor, the release included, falls to chance, while the flow variant of PredImagine keeps its accuracy on every task (b). With several equally valid intermediates, every producer falls toward chance (c).
{: .pi-cap}

<span class="pi-kicker">Tasks</span>

![Figure 2: Tasks and their gold intermediates.](/images/projects/2026-11-predimagine/fig2-tasks.png){: .pi-shot}

**Figure 2.** All tasks are rotation analogies with four options. ImagineBench-rotation sets the number of valid intermediates: one clockwise strip (IB-cw), either of two strips (IB-both), or the answer shape at any of *k* placements (IB-pos-*k*).
{: .pi-cap}

<span class="pi-kicker">Failure 1 · the release depends on its noise seed</span>

| Released WM-VLM-7B, Tetris-2D | K1 | K8 |
|---|---|---|
| Noise seed 0 | 0.870 / 0.736 | 0.738 / 0.620 |
| Noise seed 1 | 0.233 / 0.216 | 0.220 / 0.204 |
| Oracle latents | 0.860 / 0.884 | |
| Zeroed latents | 0.150 / 0.166 | |
{: .pi-table}

**Table 2.** Accuracy ID / OOD. The release reproduces its reported accuracy at noise seed 0 and falls near chance at seed 1, the behaviour of a network trained on one fixed noise tensor.
{: .pi-cap}

<span class="pi-kicker">Failure 2 · the default flow ignores the question</span>

![Figure 3: What the default flow lacks and what PredImagine changes.](/images/projects/2026-11-predimagine/fig3-cond-gap.png){: .pi-shot .pi-narrow}

**Figure 3.** At no flow time does matching the question lower the default flow's error by more than 2.4%. Single ingredients of PredImagine raise this gap only to 16–59%; a producer becomes readable only when it predicts the clean target and trains half or all of its blocks at *t* = 0, which brings its *t* = 0 error near the regressor's.
{: .pi-cap}

<span class="pi-kicker">What generation adds · robustness, not accuracy</span>

| Shared consumer, K1 s0 | *x*-pred. regression | *t* = 0 flow | Repaired flow | Velocity regression |
|---|---|---|---|---|
| Tetris-2D | 0.578 / 0.524 | 0.542 / 0.458 | 0.537 / 0.528 | 0.562 / 0.482 |
| IB-pos1 (*k* = 1) | 0.653 / 0.725 | 0.590 / 0.675 | 0.593 / 0.656 | 0.354 / 0.415 |
| IB-cw (*k* = 1) | 0.768 / 0.791 | 0.628 / 0.655 | 0.698 / 0.765 | 0.721 / 0.775 |
| IB-both (*k* = 2) | 0.598 / 0.642 | 0.468 / 0.518 | 0.483 / 0.550 | 0.649 / 0.672 |
{: .pi-table}

**Table 4.** Same consumer for every producer of a task; training draw, mean over trainer seeds, ID / OOD. At its training draw the regression variant of PredImagine is as accurate as the flow variants or more in distribution; the flows' advantage is robustness to the noise draw, which fresh training noise alone provides.
{: .pi-cap}

<span class="pi-kicker">Several valid states · the noise does not choose</span>

![Figure 5: At k = 2 the flow's noise does not choose a placement.](/images/projects/2026-11-predimagine/fig5-valid-states.png){: .pi-shot}

**Figure 5.** With two valid placements, two noise draws of the flow land on the same placement for 95% of samples, and its latents carry the answer's rotation at about chance rate. Hypothesis codes with as many codes as valid states each commit to one placement.
{: .pi-cap}

## Interactive demo

Draw fresh noise and watch which producers keep answering, build each PredImagine variant from its two training choices, and see how many intermediate states each task accepts. All numbers are from the paper.

<div class="pi-demo"><iframe src="/files/projects/2026-11-predimagine/demo/index.html" height="900" loading="lazy" title="PredImagine interactive demo"></iframe></div>

## Links and citation

<p class="pi-btns" style="justify-content:flex-start">
  <span class="pi-btn pi-off"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i> Paper <small>· under review</small></span>
  <span class="pi-btn pi-off"><i class="fa-brands fa-github" aria-hidden="true"></i> Code <small>· will be released</small></span>
</p>

The protocol, the ImagineBench-rotation generator, the stage-1 configurations of PredImagine and every arm, and the evaluation code will be released. The citation will be added with the author list once the paper is public.
