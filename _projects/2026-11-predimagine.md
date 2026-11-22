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
published: false
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
/* inline interactive figures: same frame as the figure cards above */
.pi-w { --r: #2a78d6; --f: #eb6834; --c: #1baf7a; --b: #8a8984; --g: #b9b8b2; --soft: rgba(127,127,127,.09);
  margin: .6em 0 .4em; padding: 14px 16px; border-radius: 14px; border: 1px solid var(--global-border-color);
  box-shadow: 0 1px 2px rgba(0,0,0,.05), 0 8px 24px rgba(0,0,0,.06); font-size: .8em; line-height: 1.45; }
html[data-theme="dark"] .pi-w { --r: #3987e5; --f: #d95926; --c: #199e70; --b: #8f8e88; --g: #5b5a55; }
.pi-w button { font: inherit; }
.pi-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 0 0 10px; }
.pi-lab { font-size: .82em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin-right: 2px; }
.pi-chips { display: inline-flex; flex-wrap: wrap; gap: 5px; vertical-align: middle; }
.pi-chip { appearance: none; cursor: pointer; border: 1px solid var(--global-border-color); background: transparent; color: inherit;
  border-radius: 999px; padding: 2px 11px; transition: background .15s, color .15s; }
.pi-chip[aria-pressed="true"] { background: var(--global-text-color); border-color: var(--global-text-color); color: var(--global-bg-color); }
.pi-noise { width: 34px; height: 34px; border-radius: 7px; image-rendering: pixelated; vertical-align: middle; }
.pi-bars { position: relative; }
.pi-br { display: grid; grid-template-columns: 138px 1fr; align-items: center; height: 34px; }
.pi-br .n { text-align: right; padding-right: 10px; line-height: 1.15; }
.pi-tr { position: relative; height: 100%; }
.pi-bar { position: absolute; left: 0; top: 9px; height: 16px; border-radius: 0 5px 5px 0; transition: width .6s cubic-bezier(.2,.8,.2,1); }
.pi-gh { position: absolute; top: 5px; width: 2px; height: 24px; background: var(--g); transition: left .6s cubic-bezier(.2,.8,.2,1); }
.pi-val { position: absolute; top: 7px; white-space: nowrap; opacity: .85; font-variant-numeric: tabular-nums; transition: left .6s cubic-bezier(.2,.8,.2,1); }
.pi-dl { font-size: .9em; font-weight: 700; margin-left: 5px; padding: 0 5px; border-radius: 5px; background: var(--soft); }
.pi-na { position: absolute; left: 4px; top: 7px; font-style: italic; opacity: .6; }
.pi-ax { position: relative; height: 20px; border-top: 1px solid var(--global-border-color); }
.pi-ax span { position: absolute; top: 2px; transform: translateX(-50%); font-size: .88em; opacity: .6; }
.pi-vl { position: absolute; top: 0; bottom: 20px; border-left: 1px solid var(--global-border-color); opacity: .6; pointer-events: none; }
.pi-vl.ch { border-left: 1.5px dashed currentColor; opacity: .45; }
.pi-leg { display: flex; flex-wrap: wrap; gap: 2px 14px; margin-top: 8px; opacity: .85; }
.pi-leg i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.pi-grid2 { display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(0, 1fr); gap: 14px 18px; }
.pi-board { display: grid; grid-template-columns: 74px repeat(2, minmax(0, 1fr)); gap: 6px; }
.pi-board .h { font-size: .9em; opacity: .6; align-self: end; line-height: 1.25; }
.pi-board .rh { font-weight: 600; opacity: .8; align-self: center; line-height: 1.2; }
.pi-cell { appearance: none; cursor: pointer; text-align: left; border: 1px solid var(--global-border-color); background: transparent; color: inherit;
  border-radius: 10px; padding: 7px 9px; line-height: 1.25; }
.pi-cell b { display: flex; align-items: center; gap: 6px; }
.pi-cell b i { width: 9px; height: 9px; border-radius: 3px; flex: none; }
.pi-cell small { display: block; opacity: .6; margin-top: 2px; }
.pi-cell[aria-pressed="true"] { box-shadow: 0 0 0 2px var(--sel); border-color: transparent; }
.pi-cell:disabled { cursor: not-allowed; opacity: .5; }
.pi-sub { font-size: .82em; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .6; margin: 12px 0 6px; }
.pi-strip { display: flex; gap: 5px; align-items: center; flex-wrap: wrap; }
.pi-strip canvas { width: 30px; height: 30px; border-radius: 5px; image-rendering: pixelated; }
.pi-w svg text { fill: currentColor; font-size: 11px; opacity: .75; }
.pi-pr { display: grid; grid-template-columns: 70px 1fr; align-items: center; height: 30px; }
.pi-pr .n { text-align: right; padding-right: 8px; }
.pi-pr .pi-bar { top: 5px; height: 8px; } .pi-pr .pi-bar.b2 { top: 16px; background: var(--b); }
.pi-pr .pi-val { top: 0; font-size: .9em; } .pi-pr .pi-val.v2 { top: 11px; }
.pi-verdict { display: inline-block; font-weight: 700; padding: 1px 9px; border-radius: 999px; background: var(--soft); margin-top: 10px; }
.pi-thumbs { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 6px; margin-bottom: 12px; }
.pi-thumb { appearance: none; cursor: pointer; border: 1px solid var(--global-border-color); background: transparent; color: inherit; border-radius: 10px; padding: 4px; text-align: center; }
.pi-thumb img { display: block; width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 7px; margin: 0; }
.pi-thumb span { display: block; font-weight: 600; margin-top: 3px; } .pi-thumb em { display: block; font-style: normal; opacity: .6; }
.pi-thumb[aria-pressed="true"] { box-shadow: 0 0 0 2px var(--r); border-color: transparent; }
.pi-gal { display: flex; flex-wrap: wrap; gap: 10px; align-items: flex-start; }
.pi-w figure { display: block; margin: 0; text-align: center; }
.pi-w figcaption { font-family: inherit; }
.pi-gal img { display: block; width: 96px; border-radius: 7px; margin: 0; } .pi-gal img.q { width: 124px; }
.pi-gal figcaption { font-size: .9em; opacity: .65; margin-top: 3px; max-width: 130px; margin-left: auto; margin-right: auto; }
.pi-valid { display: flex; flex-wrap: wrap; gap: 6px; padding: 6px; border-radius: 10px; background: var(--soft); }
.pi-valid img { width: 74px; }
.pi-tip { position: fixed; z-index: 50; pointer-events: none; background: var(--global-text-color); color: var(--global-bg-color); font-size: 12px;
  padding: 6px 8px; border-radius: 7px; max-width: 260px; opacity: 0; transition: opacity .1s; }
@media (max-width: 700px) {
  .pi-grid2 { grid-template-columns: 1fr; }
  .pi-br { grid-template-columns: 100px 1fr; }
  .pi-thumbs { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .pi-gal img { width: 80px; } .pi-gal img.q { width: 110px; } .pi-valid img { width: 62px; }
}
@media (max-width: 700px) {
  .pi-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .page__content img.pi-narrow { max-width: 100%; }
  .pi-hero .pi-tagline { font-size: 1.12em; }
}
.pi-stat .t { font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; opacity: .6; margin-bottom: 4px; }
.page__content a > img.pi-shot { cursor: zoom-in; }
.pi-defs { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin: .4em 0 1.6em; }
.pi-defs > div { border: 1px solid var(--global-border-color); border-radius: 12px; padding: 10px 12px; font-size: .8em; line-height: 1.45; }
.pi-defs b { display: block; margin-bottom: 2px; }
.pi-eq { margin: .6em 0 1em; padding: .5em 1em; border-radius: 12px; background: rgba(127,127,127,.08); overflow-x: auto; font-size: .92em; }
.pi-steps { counter-reset: pi; list-style: none; margin: .4em 0 1em; padding: 0; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.page__content .pi-steps li { counter-increment: pi; position: relative; margin: 0; border: 1px solid var(--global-border-color); border-radius: 12px; padding: 10px 12px 10px 44px; font-size: .84em; line-height: 1.45; }
.pi-steps li::before { content: counter(pi); position: absolute; left: 12px; top: 11px; width: 22px; height: 22px; border-radius: 50%; display: grid; place-items: center;
  font-size: 12px; font-weight: 700; color: #fff; background: linear-gradient(135deg, #2a78d6, #eb6834); }
.pi-take { list-style: none; margin: .4em 0 1.4em; padding: 0; display: grid; gap: 10px; }
.page__content .pi-take li { margin: 0; border: 1px solid var(--global-border-color); border-radius: 12px; padding: 10px 14px; font-size: .86em; line-height: 1.5; border-left: 4px solid var(--kc, #8a8984); }
.page__content p.pi-scope { font-size: .8em; opacity: .8; }
@media (max-width: 700px) { .pi-defs, .pi-steps { grid-template-columns: 1fr; }
  .page__content table.pi-table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; } }
</style>

<div class="pi-hero">
  <p class="pi-eyebrow">Vision-language models · latent visual reasoning · flow matching</p>
  <p class="pi-tagline">Latent visual imagination needs <span class="pi-pred">prediction</span> more than <span class="pi-gen">generation</span>.</p>
  <p class="pi-btns">
    <a class="pi-btn" href="#pi-try"><i class="fa-solid fa-play" aria-hidden="true"></i> Try it</a>
    <a class="pi-btn" href="#method"><i class="fa-solid fa-diagram-project" aria-hidden="true"></i> Method</a>
    <a class="pi-btn" href="#key-results"><i class="fa-solid fa-chart-column" aria-hidden="true"></i> Key results</a>
    <span class="pi-btn pi-off"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i> Paper <small>· soon</small></span>
    <span class="pi-btn pi-off"><i class="fa-brands fa-github" aria-hidden="true"></i> Code <small>· soon</small></span>
  </p>
</div>

<div class="pi-tldr"><b>TL;DR</b> &nbsp;<span class="pi-name"><b>PredImagine</b></span> trains the latent producer of a VLM to predict the clean intermediate state, with training focused where the question decides the target. It repairs two hidden failures of flow-matching imagination, and its own variants show that generation adds robustness to the noise draw, not accuracy.</div>

<div class="pi-stats">
  <div class="pi-stat"><div class="t">Failure 1</div><div class="v">.870<span class="arr">→</span>.233</div><div class="l">Accuracy of the released WM-VLM-7B on Tetris-2D when only its noise seed changes (chance is .25)</div></div>
  <div class="pi-stat"><div class="t">Failure 2</div><div class="v">≤ 2.4%</div><div class="l">How much a default flow's imagined latent depends on the question, at any flow time</div></div>
  <div class="pi-stat"><div class="t" style="color:#eb6834;opacity:1">Repair</div><div class="v">.798<span class="arr">→</span>.810</div><div class="l">PredImagine's flow variant at 7B, from the training noise to a fresh draw: it holds</div></div>
  <div class="pi-stat"><div class="t">Still open</div><div class="v">95%</div><div class="l">Questions where two noise draws of the flow imagine the same state although two are valid</div></div>
</div>

## Summary

Vision-language models can answer spatial questions by first *imagining* an intermediate image as latent tokens, and flow-matching generators of these latents are credited with large accuracy gains. We test where the gains come from with a controlled protocol in WM-VLM and find two hidden failures: the released model is in effect a fixed-noise regressor that collapses when its noise seed changes, and a default flow barely uses the question. **PredImagine** repairs both by training the producer to predict the clean intermediate state where the question decides it. Its regression and flow variants then show what generation adds: robustness to the noise draw, but neither accuracy nor the ability to commit to one of several valid states.

## Background

WM-VLM answers a spatial question in two parts. A **producer**, a small generation branch inside the VLM, imagines the hidden intermediate image as 361 latent tokens. A **consumer**, the rest of the VLM, reads those tokens and answers. The producer is a flow model: it starts from Gaussian noise and takes *K* Euler steps toward the latent of the intermediate image, so every answer depends on a noise draw.

[![Figure 1a: the producer imagines the hidden intermediate image, the consumer reads it and answers.](/images/projects/2026-11-predimagine/fig1a-pipeline.png){: .pi-shot}](/images/projects/2026-11-predimagine/fig1a-pipeline.png)

**Figure 1a.** The producer imagines the hidden intermediate image as latent tokens for the consumer to read. PredImagine changes only how the producer is trained (stage 1). To compare producers fairly, each is read by one *shared* consumer trained on oracle latents, the encodings of the true intermediate image.
{: .pi-cap}

<div class="pi-defs">
  <div><b>Noise draw: K and s</b>A read takes K Euler steps from noise seed s. <i>K1 s0</i> is the setting the release reports and the draw every fixed-noise regressor was trained on. <i>K8 s1</i> is a fresh draw.</div>
  <div><b>Shared and own consumers</b>The shared consumer never saw any producer's output, so accuracy on it reflects the latents alone. An own consumer is trained together with its producer.</div>
  <div><b>Oracle and zeroed latents</b>Two controls: the consumer reads the encoding of the true intermediate image, or all zeros. They show whether the answer goes through the latents.</div>
  <div><b>ID, OOD and chance</b>Accuracy in distribution and on held-out shapes, written ID / OOD. Every question has four options, so chance is 0.25.</div>
</div>

[![Figure 2: Tasks and their gold intermediates.](/images/projects/2026-11-predimagine/fig2-tasks.png){: .pi-shot}](/images/projects/2026-11-predimagine/fig2-tasks.png)

**Figure 2.** All tasks are rotation analogies: apply the rotation from A to B to C and pick one of four options. Tetris-2D is WM-VLM's dataset. Our ImagineBench-rotation sets the number *k* of valid intermediates: one clockwise strip (IB-cw), either of two strips (IB-both), or the answer shape at any of *k* placements (IB-pos-*k*).
{: .pi-cap}

## Method

Both failures concern *where* the producer learns the question. At flow time *t* = 0 the producer sees only noise and the question, so only the question can lower its error. There the squared-error optimum is the conditional mean of the target given the question, and one Euler step returns it for any noise (Proposition 1 of the paper). A default flow samples *t* uniformly and spends most of its training denoising the state instead.

PredImagine therefore makes two changes to the producer's training:

<ol class="pi-steps">
  <li><b>Predict the clean target.</b> The producer outputs the intermediate latent itself, not the velocity, which stays well conditioned in high dimension.</li>
  <li><b>Train at <i>t</i> = 0.</b> A share ρ of the training blocks sits exactly at <i>t</i> = 0, where the question decides the target; the rest is shifted toward noise.</li>
</ol>

<div class="pi-eq">\[ \min_\theta\ \mathbb{E}\,\lVert f_\theta(z_t, c, t) - z \rVert^2, \qquad z_t = (1-t)\,\epsilon + t\,z, \qquad t \sim \rho\,\delta_0 + (1-\rho)\,\pi_s \]</div>

Here *z* is the latent of the intermediate image, *c* the question, ε the noise, and π<sub>s</sub> uniform *t* under the time shift *s* = 3. PredImagine adds no parameters and leaves inference unchanged. The noise and the share ρ give three variants, and neighbouring variants differ in one choice, which separates what fresh noise and training over *t* each add.

<span class="pi-kicker">The three variants · click a cell</span>

<div class="pi-w" id="piB">
  <div class="pi-grid2">
    <div>
      <div class="pi-board" id="piB-board"></div>
      <div class="pi-sub">Noise it trains on</div>
      <div class="pi-strip" id="piB-strip"></div>
      <div class="pi-sub">Training timesteps</div>
      <svg id="piB-t" viewBox="0 0 260 104" width="100%" style="max-width:300px;display:block" role="img" aria-label="training timestep distribution"></svg>
    </div>
    <div>
      <div class="pi-row" style="justify-content:space-between;margin-bottom:6px"><span class="pi-lab">Accuracy at the training draw</span><span class="pi-chips" id="piB-split"></span></div>
      <div class="pi-bars" id="piB-bars"></div>
      <div class="pi-leg" id="piB-leg"></div>
      <div class="pi-verdict" id="piB-verdict"></div>
    </div>
  </div>
</div>
<p class="pi-cap"><strong>Interactive B.</strong> <span id="piB-note"></span> Bars: shared consumer, K1 s0, mean over trainer seeds (Table 4). The timestep plot applies the time shift <i>s</i> = 3 to uniform <i>t</i>.</p>

## Key results

<span class="pi-kicker">Failure 1 · the release depends on its noise seed</span>

| Released WM-VLM-7B, Tetris-2D | K1 | K8 |
|---|---|---|
| Noise seed 0 | 0.870 / 0.736 | 0.738 / 0.620 |
| Noise seed 1 | **0.233 / 0.216** | **0.220 / 0.204** |
| Oracle latents | 0.860 / 0.884 | |
| Zeroed latents | 0.150 / 0.166 | |
{: .pi-table}

**Table 2.** Accuracy ID / OOD. The release reproduces its reported accuracy at noise seed 0 and falls near chance at seed 1, as a network trained on one fixed noise tensor would; the public reproduction script defaults to exactly that. Its gain at seed 0 is real and goes through the latents (zeroed: 0.150), but it is a regressor's gain, not generation's.
{: .pi-cap}

<span class="pi-kicker">Failure 2 · the default flow ignores the question</span>

[![Figure 3: What the default flow lacks and what PredImagine changes.](/images/projects/2026-11-predimagine/fig3-cond-gap.png){: .pi-shot .pi-narrow}](/images/projects/2026-11-predimagine/fig3-cond-gap.png)

**Figure 3.** The conditional gap Δ(*t*) measures how much matching the question to its target lowers the producer's error at flow time *t*; Δ = 0 means the question adds nothing. For the default flow Δ never exceeds 2.4%, and it stays near chance on all four 3B datasets. Single ingredients of PredImagine raise the gap only to 16–59%. A producer becomes readable only when it predicts the clean target and trains half or all of its blocks at *t* = 0, which brings its *t* = 0 error near the regressor's (b).
{: .pi-cap}

<span class="pi-kicker">PredImagine repairs both · switch the noise draw</span>

<div class="pi-w" id="piA"><a id="pi-try"></a>
  <div class="pi-row">
    <span><canvas class="pi-noise" id="piA-noise" width="16" height="16" aria-label="noise tensor"></canvas></span>
    <span><span class="pi-lab">Noise</span> <span class="pi-chips" id="piA-draw"></span></span>
  </div>
  <div class="pi-row">
    <span><span class="pi-lab">Task</span> <span class="pi-chips" id="piA-task"></span></span>
    <span><span class="pi-lab">Split</span> <span class="pi-chips" id="piA-split"></span></span>
  </div>
  <div class="pi-bars" id="piA-bars"></div>
  <div class="pi-leg"><span><i style="background:var(--r)"></i>PredImagine, regression variant</span><span><i style="background:var(--f)"></i>PredImagine, flow variant</span><span><i style="background:var(--b)"></i>baselines</span><span><i style="background:var(--g);width:2px"></i>value at the other draw</span></div>
</div>
<p class="pi-cap"><strong>Interactive A.</strong> <span id="piA-note"></span> Every producer is read by its task's shared consumer (trainer seed 42). Gray ticks mark the value at the other draw. Values from Figure 1b and Table S5.</p>

[![Figure 4: What each producer imagines.](/images/projects/2026-11-predimagine/fig4-qualitative.png){: .pi-shot}](/images/projects/2026-11-predimagine/fig4-qualitative.png)

**Figure 4.** What each producer imagines on four random IB-pos1 questions, read in one step at the regressors' training noise (s0) and at a new one (s1). Latents are not decoded: each cell shows the candidate, the answer shape at one of four rotations, that is nearest to the generated latent (✓ marks the answer's rotation). What the regressors imagine changes with the noise seed, and the share with the right rotation falls from 0.59 and 0.79 to 0.23, near chance; what the flow variant of PredImagine imagines does not change (0.88 and 0.87).
{: .pi-cap}

| 7B, Tetris-2D, own consumer | Training draw (K1 s0) | Fresh draw (K8 s1) |
|---|---|---|
| WM-VLM release | 0.858 / 0.732 | 0.220 / 0.204 |
| Velocity regression (our reproduction) | 0.502 / 0.438 | 0.275 / 0.258 |
| Default flow | 0.278 / 0.298 | 0.305 / 0.248 |
| PredImagine, *x*-prediction regression | 0.880 / 0.724 | 0.230 / 0.208 |
| PredImagine, repaired flow | 0.920 / 0.726 | **0.917 / 0.692** |
{: .pi-table}

**Table 3 (excerpt).** Each producer with the consumer trained on it, trainer seed 42, ID / OOD. PredImagine lifts our 7B pipeline from 0.502 (velocity regression) to the release's level (0.880 and 0.920 against 0.858); on the shared consumer the release and both PredImagine variants are statistically level in distribution. Only the flow variant keeps its accuracy at a fresh draw.
{: .pi-cap}

<span class="pi-kicker">What generation adds · robustness, not accuracy</span>

| Shared consumer, K1 s0 | *x*-pred. regression | *t* = 0 flow | Repaired flow | Velocity regression |
|---|---|---|---|---|
| Tetris-2D | 0.578 / 0.524 | 0.542 / 0.458 | 0.537 / 0.528 | 0.562 / 0.482 |
| IB-pos1 (*k* = 1) | 0.653 / 0.725 | 0.590 / 0.675 | 0.593 / 0.656 | 0.354 / 0.415 |
| IB-cw (*k* = 1) | 0.768 / 0.791 | 0.628 / 0.655 | 0.698 / 0.765 | 0.721 / 0.775 |
| IB-both (*k* = 2) | 0.598 / 0.642 | 0.468 / 0.518 | 0.483 / 0.550 | 0.649 / 0.672 |
{: .pi-table}

**Table 4.** Same consumer for every producer of a task; training draw, mean over trainer seeds, ID / OOD. The first three columns are PredImagine's variants. At its training draw the regression variant is as accurate as the flow variants or more in distribution, so the flows' advantage is robustness to the noise draw, which fresh training noise alone provides.
{: .pi-cap}

<span class="pi-kicker">Several valid states · the noise does not choose</span>

[![Figure 5: At k = 2 the flow's noise does not choose a placement.](/images/projects/2026-11-predimagine/fig5-valid-states.png){: .pi-shot}](/images/projects/2026-11-predimagine/fig5-valid-states.png)

**Figure 5.** A generator could, in principle, commit to one of several valid intermediates. Ours does not: with two valid placements, two noise draws of the flow land on the same placement for 95% of samples, and its latents carry the answer's rotation at about chance rate. Hypothesis codes with as many codes as valid states each commit to one placement.
{: .pi-cap}

<div class="pi-w" id="piC">
  <div class="pi-thumbs" id="piC-thumbs"></div>
  <div class="pi-grid2">
    <div class="pi-gal" id="piC-gal"></div>
    <div>
      <svg id="piC-k" viewBox="0 0 300 180" width="100%" style="display:block" role="img" aria-label="accuracy against the number of valid placements k"></svg>
      <div class="pi-row" style="margin:4px 0 0;justify-content:space-between"><span class="pi-leg" style="margin:0"><span><i style="background:var(--r)"></i>PI regr.</span><span><i style="background:var(--f)"></i>PI flow</span><span><i style="background:var(--b)"></i>velocity regr.</span><span><i style="background:var(--c)"></i>codes, M = k</span></span><span class="pi-chips" id="piC-split"></span></div>
    </div>
  </div>
</div>
<p class="pi-cap"><strong>Interactive C.</strong> <span id="piC-note"></span> Click a task to see its valid intermediates. The example is drawn with the ImagineBench-rotation generator (90° clockwise); pixel averages are an illustration, since the paper's mean is over latent encodings. Curve: IB-pos4 shared consumer, K1 s0 (Figure 1c, Table S7).</p>

<div class="pi-tip" id="pi-tip" role="tooltip"></div>
<script>
(function () {
  var IMG = "/images/projects/2026-11-predimagine/samples/";
  var KIND = { release: "b", vel: "b", dflow: "b", regr: "r", flow: "f" };
  var NAMES = { release: "WM-VLM release", dflow: "Default flow", vel: "Velocity regression", regr: "PredImagine regression", flow: "PredImagine flow" };
  var DRAWS = {
    "Tetris-2D": { dflow: [[.270,.298],[.265,.232]], vel: [[.540,.464],[.260,.234]], regr: [[.540,.546],[.302,.232]], flow: [[.520,.514],[.555,.564]] },
    "IB-pos1": { vel: [[.427,.535], null], regr: [[.650,.703],[.253,.212]], flow: [[.635,.715],[.630,.705]] },
    "IB-cw": { vel: [[.690,.740],[.225,.265]], regr: [[.748,.787],[.273,.242]], flow: [[.723,.792],[.728,.780]] },
    "IB-both": { vel: [[.667,.715],[.237,.273]], regr: [[.625,.637],[.230,.282]], flow: [[.480,.515],[.472,.565]] },
    "Tetris-2D 7B": { release: [[.785,.686],[.215,.250]], dflow: [[.250,.256],[.300,.274]], vel: [[.428,.426],[.195,.242]], regr: [[.820,.662],[.255,.264]], flow: [[.798,.716],[.810,.646]] }
  };
  var PARAM = { tasks: ["Tetris-2D", "IB-pos1", "IB-cw", "IB-both"],
    regr: [[.578,.524],[.653,.725],[.768,.791],[.598,.642]], t0: [[.542,.458],[.590,.675],[.628,.655],[.468,.518]],
    flow: [[.537,.528],[.593,.656],[.698,.765],[.483,.550]], vel: [[.562,.482],[.354,.415],[.721,.775],[.649,.672]] };
  var KC = { regr: [[.650,.703],[.403,.485],[.347,.395]], flow: [[.635,.715],[.258,.235],[.242,.240]],
    vel: [[.427,.535],[.280,.287],[.263,.230]], codes: [null,[.428,.520],[.375,.439]] };
  var fmt = function (v) { return v == null ? "–" : v.toFixed(3).replace(/^0/, ""); };
  var tick = function (t) { return t === 0 ? "0" : t === 1 ? "1" : t.toFixed(2).replace(/^0/, ""); };
  var $ = function (id) { return document.getElementById(id); };
  var tip = $("pi-tip");
  function showTip(e, h) { tip.innerHTML = h; tip.style.opacity = 1; tip.style.left = Math.min(e.clientX + 12, window.innerWidth - 270) + "px"; tip.style.top = (e.clientY + 14) + "px"; }
  function hideTip() { tip.style.opacity = 0; }
  function chips(el, opts, val, cb) {
    el.innerHTML = "";
    opts.forEach(function (o) {
      var b = document.createElement("button"); b.type = "button"; b.className = "pi-chip"; b.textContent = o[0];
      b.setAttribute("aria-pressed", String(o[1] === val));
      b.onclick = function () { el.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); }); cb(o[1]); };
      el.appendChild(b);
    });
  }
  function frame(host, left) {
    [.25, .5, .75].forEach(function (t) { var v = document.createElement("div"); v.className = "pi-vl" + (t === .25 ? " ch" : ""); v.style.left = "calc(" + left + "px + (100% - " + left + "px) * " + t + ")"; host.appendChild(v); });
    var ax = document.createElement("div"); ax.className = "pi-ax"; ax.style.marginLeft = left + "px";
    [0, .25, .5, .75, 1].forEach(function (t) { var s = document.createElement("span"); s.style.left = t * 100 + "%"; s.textContent = t === .25 ? ".25 chance" : tick(t); ax.appendChild(s); });
    return ax;
  }
  function rng(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; var t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  function noise(c, seed) {
    var x = c.getContext("2d"), n = c.width, im = x.createImageData(n, n), r = rng(seed * 7919 + 17);
    for (var i = 0; i < n * n; i++) { var g = Math.sqrt(-2 * Math.log(r() + 1e-9)) * Math.cos(2 * Math.PI * r()); var v = Math.max(0, Math.min(255, 128 + 52 * g)); im.data.set([v * .92, v * .95, v, 255], i * 4); }
    x.putImageData(im, 0, 0);
  }
  var narrow = function () { return window.innerWidth <= 700; };
  /* ---- A: re-draw the noise */
  var A = { task: "Tetris-2D 7B", split: 0, draw: 0 }, builtA = "";
  function renderA() {
    var host = $("piA-bars"), d = DRAWS[A.task], keys = ["release", "dflow", "vel", "regr", "flow"].filter(function (k) { return d[k]; });
    var lw = narrow() ? 100 : 138, sig = A.task + lw;
    if (builtA !== sig) {
      builtA = sig; host.innerHTML = ""; var ax = frame(host, lw);
      keys.forEach(function (k) {
        var row = document.createElement("div"); row.className = "pi-br";
        row.innerHTML = '<div class="n">' + (KIND[k] === "b" ? NAMES[k] : "<b>" + NAMES[k] + "</b>") + '</div><div class="pi-tr"><div class="pi-bar" style="background:var(--' + KIND[k] + ')"></div><div class="pi-val"></div><div class="pi-gh"></div><span class="pi-na">not read at a fresh draw</span></div>';
        row.onmousemove = function (e) { var q = DRAWS[A.task][k]; showTip(e, "<b>" + NAMES[k] + "</b> · " + A.task + " " + ["ID", "OOD"][A.split] + "<br>training draw " + fmt(q[0][A.split]) + " → fresh draw " + (q[1] ? fmt(q[1][A.split]) : "not read")); };
        row.onmouseleave = hideTip; host.appendChild(row);
      });
      host.appendChild(ax);
    }
    host.querySelectorAll(".pi-tr").forEach(function (tr, i) {
      var k = keys[i], cur = d[k][A.draw], oth = d[k][1 - A.draw];
      tr.querySelector(".pi-na").style.display = cur ? "none" : "";
      var val = tr.querySelector(".pi-val"), gh = tr.querySelector(".pi-gh");
      val.style.display = cur ? "" : "none"; tr.querySelector(".pi-bar").style.width = cur ? cur[A.split] * 100 + "%" : "0";
      if (cur) {
        var dl = A.draw ? cur[A.split] - d[k][0][A.split] : null;
        val.innerHTML = fmt(cur[A.split]) + (dl == null ? "" : '<span class="pi-dl">' + (dl >= 0 ? "+" : "−") + Math.abs(dl * 100).toFixed(1) + "</span>");
        val.style.left = "calc(" + cur[A.split] * 100 + "% + 6px)";
      }
      gh.style.display = oth ? "" : "none"; if (oth) gh.style.left = oth[A.split] * 100 + "%";
    });
    $("piA-note").innerHTML = A.draw === 0
      ? "At the training draw each regressor reads the noise tensor it was trained on, as the release reports. " + (A.split === 0
        ? "In distribution the regression variant of PredImagine is as accurate as its flow variant or more."
        : "Out of distribution the two PredImagine variants trade places from task to task.")
      : (A.task === "Tetris-2D 7B"
        ? "At a fresh draw the release (0.785 → 0.215 ID) and both regressors fall to chance; the flow variant of PredImagine keeps its accuracy (0.798 → 0.810)."
        : "At a fresh draw every fixed-noise regressor falls to chance" + (d.dflow ? " and the default flow stays there" : "") + "; only the flow variant of PredImagine keeps its accuracy.");
  }
  function setDraw(v) {
    A.draw = v; var c = $("piA-noise");
    if (!matchMedia("(prefers-reduced-motion: reduce)").matches) {
      var i = 0, id = setInterval(function () { noise(c, 1000 + i); if (++i > 6) { clearInterval(id); noise(c, v); renderA(); } }, 45);
    } else { noise(c, v); renderA(); }
  }
  chips($("piA-draw"), [["Training draw · K1 s0", 0], ["Fresh draw · K8 s1", 1]], 0, setDraw);
  chips($("piA-task"), Object.keys(DRAWS).map(function (t) { return [t, t]; }), A.task, function (v) { A.task = v; renderA(); });
  chips($("piA-split"), [["ID", 0], ["OOD", 1]], 0, function (v) { A.split = v; renderA(); });
  /* ---- B: build a variant */
  var B = { cell: "fresh-0.5", split: 0 };
  var V = {
    "fixed-1": { key: "regr", k: "r", name: "x-prediction regression", robust: false,
      note: "One fixed noise draw with every block at t = 0 gives a deterministic map from question to state. It is the most accurate variant at its training draw (ahead of the repaired flow in distribution by 4.2 to 11.4 points) and falls to chance at a fresh draw." },
    "fresh-1": { key: "t0", k: "f", name: "t = 0 flow", robust: true,
      note: "Fresh noise with every block at t = 0 differs from regression only in the noise. It keeps its accuracy at a fresh draw (within 4.1 points on Tetris-2D, IB-cw and IB-both) and trails x-prediction regression by 6 to 15 points ID on ImagineBench: fresh noise alone brings the robustness and costs accuracy." },
    "fresh-0.5": { key: "flow", k: "f", name: "Repaired flow", robust: true,
      note: "Fresh noise, half of the blocks at t = 0 and the rest shifted toward noise. It keeps its accuracy at a fresh draw on every task (7B: 0.798 → 0.810) and, while training twice as long, wins back part of the t = 0 flow's cost only on IB-cw." }
  };
  function renderB() {
    var cell = function (id, l, s, k, dis) { return '<button type="button" class="pi-cell" data-id="' + id + '" ' + (dis ? "disabled" : "") + ' aria-pressed="' + (B.cell === id) + '" style="--sel:var(--' + (k || "b") + ')"><b>' + (k ? '<i style="background:var(--' + k + ')"></i>' : "") + l + "</b><small>" + s + "</small></button>"; };
    var bd = $("piB-board");
    bd.innerHTML = '<div></div><div class="h">all at <i>t</i>&nbsp;=&nbsp;0 (ρ&nbsp;=&nbsp;1)</div><div class="h">half at <i>t</i>&nbsp;=&nbsp;0 (ρ&nbsp;=&nbsp;½)</div>' +
      '<div class="rh">one fixed noise draw</div>' + cell("fixed-1", "x-pred. regression", "regression variant", "r") + cell("fixed-0.5", "not an arm", "a fixed draw also fixes t = 0", null, true) +
      '<div class="rh">fresh noise</div>' + cell("fresh-1", "t = 0 flow", "flow variant", "f") + cell("fresh-0.5", "Repaired flow", "flow variant", "f");
    bd.querySelectorAll(".pi-cell:not(:disabled)").forEach(function (c) { c.onclick = function () { B.cell = c.dataset.id; renderB(); }; });
    var v = V[B.cell], fixed = B.cell.indexOf("fixed") === 0, rho = /-1$/.test(B.cell) ? 1 : .5;
    var st = $("piB-strip"); st.innerHTML = "";
    for (var i = 0; i < 4; i++) { var c = document.createElement("canvas"); c.width = c.height = 10; noise(c, fixed ? 0 : 40 + i); st.appendChild(c); }
    var lab = document.createElement("span"); lab.style.opacity = .65; lab.textContent = fixed ? "the same tensor every update" : "a new tensor every update"; st.appendChild(lab);
    var W = 260, H = 104, L = 6, R = 6, T = 6, Bm = 20, X = function (t) { return L + (W - L - R) * t; }, Y = function (y) { return T + (H - T - Bm) * (1 - y / 3.2); };
    var g = '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + (H - Bm) + '" y2="' + (H - Bm) + '" stroke="currentColor" stroke-opacity=".25"/>';
    [0, .5, 1].forEach(function (t) { g += '<text x="' + X(t) + '" y="' + (H - 5) + '" text-anchor="' + (t === 0 ? "start" : t === 1 ? "end" : "middle") + '">t = ' + t + "</text>"; });
    if (rho < 1) {
      var p = "M " + X(0) + " " + Y(0);
      for (var j = 0; j <= 60; j++) { var t = j / 60; p += " L " + X(t) + " " + Y((1 - rho) * 3 / Math.pow(1 + 2 * t, 2)); }
      g += '<path d="' + p + " L " + X(1) + " " + Y(0) + ' Z" fill="var(--f)" fill-opacity=".2" stroke="var(--f)" stroke-width="1.5"/>';
      g += '<text x="' + X(.3) + '" y="' + Y(.9) + '">other ½: shifted toward noise</text>';
    }
    var top = Y(rho === 1 ? 3 : 2.2);
    g += '<rect x="' + X(0) + '" y="' + top + '" width="8" height="' + (Y(0) - top) + '" rx="2" fill="var(--' + v.k + ')"/>';
    g += '<text x="' + (X(0) + 14) + '" y="' + (top + 10) + '">' + (rho === 1 ? "every block at t = 0" : "½ of blocks at t = 0") + "</text>";
    $("piB-t").innerHTML = g;
    var host = $("piB-bars"); host.innerHTML = ""; var ax = frame(host, 70);
    PARAM.tasks.forEach(function (t, i) {
      var a = PARAM[v.key][i][B.split], b = PARAM.vel[i][B.split], row = document.createElement("div"); row.className = "pi-pr";
      row.innerHTML = '<div class="n">' + t + '</div><div class="pi-tr"><div class="pi-bar" style="width:' + a * 100 + "%;background:var(--" + v.k + ')"></div><div class="pi-bar b2" style="width:' + b * 100 + '%"></div><div class="pi-val" style="left:calc(' + a * 100 + '% + 5px)">' + fmt(a) + '</div><div class="pi-val v2" style="left:calc(' + b * 100 + '% + 5px)">' + fmt(b) + "</div></div>";
      row.onmousemove = function (e) { showTip(e, "<b>" + t + "</b>, " + ["ID", "OOD"][B.split] + "<br>" + v.name + " " + fmt(a) + " · velocity regression " + fmt(b)); };
      row.onmouseleave = hideTip; host.appendChild(row);
    });
    host.appendChild(ax);
    $("piB-leg").innerHTML = '<span><i style="background:var(--' + v.k + ')"></i>' + v.name + '</span><span><i style="background:var(--b)"></i>velocity regression</span>';
    $("piB-verdict").textContent = v.robust ? "✓ keeps its accuracy at a fresh draw" : "✕ falls to chance at a fresh draw";
    $("piB-note").innerHTML = "<b>" + v.name + ".</b> " + v.note;
  }
  chips($("piB-split"), [["ID", 0], ["OOD", 1]], 0, function (v) { B.split = v; renderB(); });
  /* ---- C: several valid answers */
  var C = { task: "pos2", split: 0 };
  var T = {
    cw: ["IB-cw", "k = 1", "strip-cw", null, [["strip-cw", "clockwise strip"]], null,
      "With one valid intermediate (the clockwise strip), the squared-error optimum is that state, so a regressor has nothing to average."],
    both: ["IB-both", "k = 2", "strip-ccw", null, [["strip-cw", "clockwise"], ["strip-ccw", "counter-clockwise"]], "mean-both",
      "Two valid strips turn opposite ways. The flow's 16 draws per question stay together (cosine 0.93 at K8) and reach neither strip (0.82), while the consumer reads the exact mean as well as either strip: the deficit lies in the producer."],
    pos1: ["IB-pos-1", "k = 1", "pos-0", 1, [["pos-0", "centre"]], null,
      "At k = 1 the answer shape is always at the centre; all three PredImagine variants lead velocity regression by over 20 points."],
    pos2: ["IB-pos-2", "k = 2", "mean-pos2", 2, [["pos-0", "centre"], ["pos-1", "top-left"]], "mean-pos2",
      "At k = 2 every producer drops and the flow drops furthest, to chance: two noise seeds at K8 land on the same placement for 95% / 94% of samples. Two hypothesis codes (M = k) each hold one placement."],
    pos4: ["IB-pos-4", "k = 4", "mean-pos4", 4, [["pos-0", "centre"], ["pos-1", "top-left"], ["pos-2", "top-right"], ["pos-3", "bottom-left"]], "mean-pos4",
      "At k = 4 the flow's latents sit near the placement mean at K1 and K8 and, at K128, leave it without approaching a placement; the noise never selects a valid state."]
  };
  function renderC() {
    var th = $("piC-thumbs");
    th.innerHTML = Object.keys(T).map(function (id) { var t = T[id]; return '<button type="button" class="pi-thumb" data-id="' + id + '" aria-pressed="' + (C.task === id) + '"><img src="' + IMG + t[2] + '.png" alt=""><span>' + t[0] + "</span><em>" + t[1] + "</em></button>"; }).join("");
    th.querySelectorAll(".pi-thumb").forEach(function (b) { b.onclick = function () { C.task = b.dataset.id; renderC(); }; });
    var t = T[C.task];
    $("piC-gal").innerHTML = '<figure><img class="q" src="' + IMG + 'question.png" alt="question"><figcaption>question</figcaption></figure>' +
      '<figure><div class="pi-valid">' + t[4].map(function (v) { return '<figure><img src="' + IMG + v[0] + '.png" alt="' + v[1] + '"><figcaption>' + v[1] + "</figcaption></figure>"; }).join("") + "</div><figcaption style=\"max-width:none\">" + (t[4].length === 1 ? "the valid intermediate" : "any of these is valid") + "</figcaption></figure>" +
      (t[5] ? '<figure><img src="' + IMG + t[5] + '.png" alt="pixel average"><figcaption>their pixel average encodes no valid state</figcaption></figure>' : "");
    $("piC-note").textContent = t[6];
    drawK();
  }
  function drawK() {
    var sp = C.split, hk = T[C.task][3], W = 300, H = 180, L = 32, R = 10, Tp = 8, Bm = 24, xs = [1, 2, 4];
    var X = function (i) { return L + (W - L - R) * (i / 2); }, Y = function (v) { return Tp + (H - Tp - Bm) * (1 - (v - .2) / .55); };
    var g = "";
    if (hk) { var i0 = xs.indexOf(hk); g += '<rect x="' + (X(i0) - 16) + '" y="' + Tp + '" width="32" height="' + (H - Tp - Bm) + '" rx="6" fill="currentColor" fill-opacity=".07"/>'; }
    [.25, .4, .55, .7].forEach(function (v) {
      g += '<line x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="currentColor" stroke-opacity="' + (v === .25 ? ".45" : ".12") + '"' + (v === .25 ? ' stroke-dasharray="4 3"' : "") + "/>";
      g += '<text x="' + (L - 5) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + fmt(v) + "</text>";
    });
    xs.forEach(function (k, i) { g += '<text x="' + X(i) + '" y="' + (H - 6) + '" text-anchor="middle">k = ' + k + "</text>"; });
    g += '<text x="' + (L + 4) + '" y="' + (Y(.25) + 13) + '">chance</text>';
    [["vel", "b"], ["codes", "c"], ["regr", "r"], ["flow", "f"]].forEach(function (s) {
      var pts = KC[s[0]].map(function (q, i) { return q ? [X(i), Y(q[sp])] : null; }).filter(Boolean);
      g += '<polyline fill="none" stroke="var(--' + s[1] + ')" stroke-width="2" points="' + pts.map(function (p) { return p.join(","); }).join(" ") + '"/>';
      pts.forEach(function (p) { g += '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="4" fill="var(--' + s[1] + ')" stroke="var(--global-bg-color)" stroke-width="2"/>'; });
    });
    xs.forEach(function (k, i) { g += '<rect class="hit" data-i="' + i + '" x="' + (X(i) - 20) + '" y="' + Tp + '" width="40" height="' + (H - Tp - Bm) + '" fill="transparent"/>'; });
    var svg = $("piC-k"); svg.innerHTML = g;
    svg.querySelectorAll(".hit").forEach(function (r) {
      r.onmousemove = function (e) { var i = +r.dataset.i, f = function (p) { return p ? fmt(p[sp]) : "–"; };
        showTip(e, "<b>k = " + xs[i] + "</b>, " + ["ID", "OOD"][sp] + "<br>PredImagine regr. " + f(KC.regr[i]) + "<br>PredImagine flow " + f(KC.flow[i]) + "<br>velocity regr. " + f(KC.vel[i]) + "<br>codes, M = k " + f(KC.codes[i])); };
      r.onmouseleave = hideTip;
    });
  }
  chips($("piC-split"), [["ID", 0], ["OOD", 1]], 0, function (v) { C.split = v; drawK(); });
  noise($("piA-noise"), 0); renderA(); renderB(); renderC();
  window.addEventListener("resize", function () { builtA = ""; renderA(); });
})();</script>


## Takeaways

<ul class="pi-take">
  <li style="--kc:#2a78d6"><b>For accuracy, predict.</b> In this architecture and on these tasks, the accuracy that latent imagination adds comes with training the producer to predict the intermediate state. At its training draw, a regressor with the same target is as accurate as the flow or more in distribution.</li>
  <li style="--kc:#eb6834"><b>For robustness to noise, train on fresh noise.</b> A fixed seed avoids the collapse at deployment, but any use that draws new noise, such as search or policy gradients over imagined states, needs the flow variant.</li>
  <li><b>Sampling distinct valid states is still open.</b> None of our flows selects among valid states, and winner-take-all codes commit to them only when their number equals the number of valid states.</li>
  <li><b>How to evaluate generated latents.</b> We suggest one consumer trained on oracle latents for all producers, oracle and zeroed reads beside own-consumer accuracy, reads at fresh seeds and step counts, a regressor with the flow's target at equal steps, a set number of valid intermediates before claiming sampling, and multiplicity-corrected paired tests.</li>
</ul>

The study covers one architecture (WM-VLM) on one backbone family and rotation analogies; most cells have one to three training seeds, and small differences are settled by paired tests in the paper.
{: .pi-scope}

## Links and citation

<p class="pi-btns" style="justify-content:flex-start">
  <span class="pi-btn pi-off"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i> Paper <small>· under review</small></span>
  <span class="pi-btn pi-off"><i class="fa-brands fa-github" aria-hidden="true"></i> Code <small>· will be released</small></span>
</p>

The protocol, the ImagineBench-rotation generator, the stage-1 configurations of PredImagine and every arm, and the evaluation code will be released. The citation will be added with the author list once the paper is public.
