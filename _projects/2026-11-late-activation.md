---
title: "Late Activation: A Better Reward–Diversity Trade-off for RL-Tuned Flow Models"
collection: projects
permalink: /projects/2026-11-late-activation/
date: 2026-11-16
status: "Manuscript"
venue: "Manuscript"
excerpt: "Let the base model take the first sampling steps of an RL-tuned flow model: 96% of the reward gain, 25% more CLIP diversity, no retraining and no extra compute."
teaser: "/images/projects/2026-11-late-activation/teaser.webp"
tags: ["diffusion models", "reinforcement learning", "sample diversity"]
publication: "/publication/2026-11-late-activation"
toc: false   # the floating table of contents covers the long title; the page has its own section links
published: false
---

<style>
.la { --la-accent: #D55E00; --la-accent-soft: rgba(213, 94, 0, .10); --la-nft: #0072B2; --la-base: #8c8c8c;
  --la-card: #ffffff; --la-soft: #f6f7f9; --la-line: #e3e6ea; --la-muted: #5f6672;
  --la-shadow: 0 1px 2px rgba(16, 24, 40, .06), 0 4px 14px rgba(16, 24, 40, .06); }
html[data-theme="dark"] .la { --la-accent: #ff8a3d; --la-accent-soft: rgba(255, 138, 61, .14); --la-nft: #4aa3df;
  --la-base: #9aa0a6; --la-card: #23262c; --la-soft: #1d2025; --la-line: #363a42; --la-muted: #a7aeb8;
  --la-shadow: 0 1px 2px rgba(0, 0, 0, .3), 0 6px 18px rgba(0, 0, 0, .25); }
.la { margin-top: .5em; }
.la h2 { margin-top: 2.2em; border-bottom: 0; font-size: 1.45em; }
.la h2 .la-num { color: var(--la-accent); margin-right: .35em; font-variant-numeric: tabular-nums; }
.la p { line-height: 1.7; }
.la-kicker { display: flex; flex-wrap: wrap; gap: .5em; margin: 0 0 .9em; }
.la-tag { font-size: .72em; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; padding: .28em .75em;
  border-radius: 999px; background: var(--la-soft); border: 1px solid var(--la-line); color: var(--la-muted); }
.la-tag.is-accent { background: var(--la-accent-soft); border-color: transparent; color: var(--la-accent); }
.la p.la-tagline { font-size: 1.22em; line-height: 1.55; margin: 0 0 1.1em; }
.la-tagline b { color: var(--la-accent); }
.la-buttons { display: flex; flex-wrap: wrap; gap: .6em; margin-bottom: 1.6em; }
.la-btn { display: inline-flex; align-items: center; gap: .5em; padding: .55em 1.15em; border-radius: 999px;
  font-size: .85em; font-weight: 600; text-decoration: none !important; background: #24292f; color: #fff !important;
  border: 1px solid #24292f; transition: transform .15s ease, box-shadow .15s ease; }
.la-btn:hover { transform: translateY(-1px); box-shadow: var(--la-shadow); }
.la-btn svg { width: 1.05em; height: 1.05em; fill: currentColor; }
.la-btn.is-primary { background: var(--la-accent); border-color: var(--la-accent); }
.la-btn.is-off { background: transparent; color: var(--la-muted) !important; border-color: var(--la-line); cursor: default; }
.la-btn.is-off:hover { transform: none; box-shadow: none; }
html[data-theme="dark"] .la-btn:not(.is-primary):not(.is-off) { background: #e8eaed; color: #16181c !important; border-color: #e8eaed; }
.la-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: .8em; margin: 0 0 1.8em; }
.la-stat { background: var(--la-card); border: 1px solid var(--la-line); border-radius: 14px; padding: 1em 1.1em; box-shadow: var(--la-shadow); }
.la-stat .v { font-size: 1.9em; font-weight: 800; line-height: 1.1; color: var(--la-accent); font-variant-numeric: tabular-nums; }
.la-stat .k { font-size: .8em; color: var(--la-muted); margin-top: .35em; line-height: 1.4; }
.la-card { background: var(--la-card); border: 1px solid var(--la-line); border-radius: 16px; box-shadow: var(--la-shadow);
  padding: 1.1em; margin: 1.2em 0; }
.la-compare-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: .6em; margin-bottom: .8em; }
.la-compare-head .t { font-weight: 700; }
.la-seg { display: inline-flex; flex-wrap: wrap; background: var(--la-soft); border: 1px solid var(--la-line); border-radius: 999px; padding: 3px; }
.la-seg button { border: 0; background: transparent; color: var(--la-muted); font: inherit; font-size: .8em; font-weight: 600;
  padding: .35em .9em; border-radius: 999px; cursor: pointer; }
.la-seg button[aria-pressed="true"] { background: var(--la-card); color: var(--la-accent); box-shadow: var(--la-shadow); }
.la-slider { position: relative; border-radius: 10px; overflow: hidden; user-select: none; touch-action: pan-y;
  aspect-ratio: 642 / 157; background: var(--la-soft); cursor: ew-resize; outline: none; }
.la-slider img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; margin: 0; max-width: none; }
.la-slider .after { clip-path: inset(0 0 0 var(--pos, 50%)); }
.la-slider .bar { position: absolute; top: 0; bottom: 0; left: var(--pos, 50%); width: 3px; margin-left: -1.5px; background: #fff;
  box-shadow: 0 0 0 1px rgba(0, 0, 0, .25); pointer-events: none; }
.la-slider .knob { position: absolute; top: 50%; left: var(--pos, 50%); width: 34px; height: 34px; margin: -17px 0 0 -17px;
  border-radius: 50%; background: #fff; box-shadow: 0 2px 8px rgba(0, 0, 0, .35); pointer-events: none;
  display: grid; place-items: center; color: #333; font-size: 14px; font-weight: 700; }
.la-slider:focus-visible .knob { box-shadow: 0 0 0 3px var(--la-accent); }
.la-slider .lbl { position: absolute; top: 8px; padding: .2em .6em; border-radius: 6px; font-size: .72em; font-weight: 700;
  color: #fff; pointer-events: none; }
.la-slider .lbl.l { left: 8px; background: var(--la-nft); }
.la-slider .lbl.r { right: 8px; background: var(--la-accent); }
.la p.la-caption { font-size: .85em; color: var(--la-muted); margin: .7em 0 0; line-height: 1.55; }
.la-tldr { border-left: 4px solid var(--la-accent); background: var(--la-accent-soft); border-radius: 0 12px 12px 0;
  padding: .9em 1.2em; margin: 1.4em 0; }
.la-tldr b:first-child { color: var(--la-accent); letter-spacing: .04em; margin-right: .4em; }
.la-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: .8em; margin-top: 1em; }
.la-step { background: var(--la-soft); border: 1px solid var(--la-line); border-radius: 12px; padding: .8em 1em; font-size: .88em; line-height: 1.55; }
.la-step b { display: block; margin-bottom: .2em; }
.la-step .n { display: inline-grid; place-items: center; width: 1.6em; height: 1.6em; border-radius: 50%; background: var(--la-accent);
  color: #fff; font-size: .8em; font-weight: 700; margin-right: .4em; }
.la-schematic { width: 100%; height: auto; display: block; }
.la-schematic text { font-family: inherit; }
.la-fig { margin: 1.4em 0; }
.la-fig img { display: block; width: 100%; border-radius: 10px; background: #fff; }
.la .la-fig figcaption { font-size: .88em; line-height: 1.6; margin-top: .8em; color: inherit; font-family: inherit; font-style: normal; text-align: left; }
.la-fig figcaption .badge { display: inline-block; font-size: .78em; font-weight: 700; padding: .12em .55em; border-radius: 6px;
  background: var(--la-accent-soft); color: var(--la-accent); margin-right: .5em; }
.la-fig.is-narrow img { max-width: 620px; margin: 0 auto; }
.la-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .86em; display: table; margin: 0; }
.la-table th, .la-table td { padding: .55em .7em; border-bottom: 1px solid var(--la-line); text-align: right; white-space: nowrap; }
.la-table th:first-child, .la-table td:first-child { text-align: left; white-space: normal; }
.la-table thead th { font-size: .82em; text-transform: uppercase; letter-spacing: .03em; color: var(--la-muted); background: transparent; }
.la-table tr.ours td { background: var(--la-accent-soft); font-weight: 700; }
.la-table tr.ours td:first-child { border-left: 3px solid var(--la-accent); }
.la-table tr.sep td { border-top: 2px solid var(--la-line); }
.la-table .best { color: var(--la-accent); }
.la-table-wrap { overflow-x: auto; }
.la table.la-table, .la table.la-table thead, .la table.la-table tbody, .la table.la-table tr { background: transparent !important; border: 0 !important; }
.la table.la-table th, .la table.la-table td { border: 0 !important; border-bottom: 1px solid var(--la-line) !important; background: transparent; }
.la table.la-table thead th { background: transparent !important; color: var(--la-muted) !important; border-bottom: 2px solid var(--la-line) !important; }
.la table.la-table tr.ours td { background: var(--la-accent-soft) !important; }
.la table.la-table tr.ours td:first-child { border-left: 3px solid var(--la-accent) !important; }
.la table.la-table tr.sep td { border-top: 2px solid var(--la-line) !important; }
.la table.la-table td:first-child, .la table.la-table th:first-child { min-width: 9.5em; }
.la table.la-table { font-size: .82em; }
.la table.la-table th, .la table.la-table td { padding: .5em .45em; }
.la table.la-table thead th { white-space: normal; vertical-align: bottom; line-height: 1.3; }
.la { --la-aigu: #CC79A7; --la-weight: #56B4E9; --la-early: #4d4d4d; }
html[data-theme="dark"] .la { --la-aigu: #e09cc4; --la-weight: #7cc6ef; --la-early: #c4c7cc; }
.la-card-title { font-weight: 700; margin: 0 0 .25em; }
.la-lede { display: grid; gap: .7em; }
.la-lede > div { display: grid; grid-template-columns: 7.2em 1fr; gap: 1em; align-items: baseline; }
.la-lede .h { font-size: .74em; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--la-accent); }
.la-lede p { margin: 0; }
.la-terms { display: grid; grid-template-columns: repeat(2, 1fr); gap: .1em 1.6em; margin: 0; }
.la-terms div { padding: .55em 0; border-bottom: 1px solid var(--la-line); font-size: .88em; line-height: 1.5; }
.la-terms b { display: block; }
.la-terms span { color: var(--la-muted); }
.la p.la-hint { font-size: .88em; margin: .8em 0 0; padding: .55em .8em; border-radius: 8px; background: var(--la-accent-soft); line-height: 1.5; }
.la-list { margin: .6em 0 0; padding-left: 1.2em; font-size: .88em; line-height: 1.6; color: var(--la-muted); }
.la-list b { color: inherit; }
.la-list li { margin: .15em 0; }
.la pre.la-code { margin: .8em 0 0; padding: .9em 1em; border-radius: 10px; background: var(--la-soft); border: 1px solid var(--la-line);
  font-size: .82em; line-height: 1.55; overflow-x: auto; color: inherit; }
.la pre.la-code .c { color: var(--la-muted); }
.la pre.la-code .k { color: var(--la-accent); font-weight: 600; }
.la-two { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2em; }

.la-controls { display: flex; flex-wrap: wrap; align-items: center; gap: .6em 1em; margin: .8em 0; }
.la-controls label { font-size: .85em; color: var(--la-muted); }
.la-controls input[type=range] { width: min(320px, 100%); accent-color: var(--la-accent); }
.la-canvas { display: block; width: 100%; height: auto; border-radius: 10px; background: var(--la-soft); border: 1px solid var(--la-line); }
.la-mini { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .6em; margin-top: .8em; }
.la-mini > div { background: var(--la-soft); border: 1px solid var(--la-line); border-radius: 10px; padding: .55em .8em; }
.la-mini .v { font-size: 1.35em; font-weight: 800; font-variant-numeric: tabular-nums; }
.la-mini .v.up { color: var(--la-accent); }
.la-mini .k { font-size: .75em; color: var(--la-muted); line-height: 1.35; }
.la-chart { width: 100%; height: auto; display: block; margin-top: .6em; }
.la-legend { display: flex; flex-wrap: wrap; gap: .3em 1em; font-size: .78em; color: var(--la-muted); margin-top: .3em; }
.la-legend span::before { content: ""; display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 5px;
  vertical-align: middle; background: var(--c); }
.la-chips { display: flex; flex-wrap: wrap; gap: .4em; margin: .2em 0 .6em; }
.la-chips button { border: 1px solid var(--la-line); background: var(--la-soft); color: inherit; font: inherit; font-size: .8em;
  font-weight: 600; padding: .3em .85em; border-radius: 999px; cursor: pointer; }
.la-chips button[aria-pressed="true"] { border-color: var(--la-accent); background: var(--la-accent-soft); color: var(--la-accent); }
.la-gallery { display: grid; grid-template-columns: 9.5em 1fr; gap: .5em .9em; align-items: center; }
.la-gallery .lab { font-size: .82em; font-weight: 700; border-left: 4px solid; padding-left: .6em; line-height: 1.3; }
.la-gallery img { width: 100%; height: auto; border-radius: 8px; display: block; margin: 0; }
.la p.la-prompt { font-style: italic; color: var(--la-muted); margin: 0 0 .8em; font-size: .92em; }
.la p.la-cite { font-size: .9em; color: var(--la-muted); }
@media (max-width: 720px) {
  .la-stats { grid-template-columns: repeat(2, 1fr); }
  .la-steps { grid-template-columns: 1fr; }
  .la-slider { aspect-ratio: 321 / 157; }
  .la-slider img { object-position: left center; }
  .la-gallery { grid-template-columns: 1fr; }
  .la-terms, .la-two { grid-template-columns: 1fr; }
  .la-lede > div { grid-template-columns: 1fr; gap: .1em; }
  .la-gallery .lab { margin-top: .4em; }
}
</style>

<div class="la">

<div class="la-kicker">
  <span class="la-tag is-accent">Manuscript · 2026</span>
  <span class="la-tag">Text-to-image</span>
  <span class="la-tag">RL fine-tuning</span>
  <span class="la-tag">Sample diversity</span>
</div>

<p class="la-tagline">Text-to-image flow models tuned with reinforcement learning draw the same picture for every seed. Let the base model take the first few
sampling steps and the variety comes back: <b>almost the same reward, more diverse images, no retraining and no extra
compute.</b></p>

<div class="la-buttons">
  <a class="la-btn is-primary" href="#samples"><svg viewBox="0 0 24 24"><path d="M4 5h16v14H4zm2 2v8l4-4 3 3 2-2 3 3V7z"/></svg>Explore samples</a>
  <a class="la-btn" href="#results"><svg viewBox="0 0 24 24"><path d="M5 9h3v10H5zm5.5-4h3v14h-3zM16 13h3v6h-3z"/></svg>Key results</a>
  <span class="la-btn is-off"><svg viewBox="0 0 24 24"><path d="M6 2h9l5 5v15H6zm8 1.5V8h4.5z"/></svg>Paper · coming soon</span>
  <span class="la-btn is-off"><svg viewBox="0 0 24 24"><path d="M9.4 16.6 4.8 12l4.6-4.6L8 6l-6 6 6 6zm5.2 0 4.6-4.6-4.6-4.6L16 6l6 6-6 6z"/></svg>Code · coming soon</span>
</div>

<div class="la-stats">
  <div class="la-stat"><div class="v">96%</div><div class="k">of the RL reward gain kept (DiffusionNFT, τ = 0.96)</div></div>
  <div class="la-stat"><div class="v">+25%</div><div class="k">CLIP diversity over the RL-tuned model</div></div>
  <div class="la-stat"><div class="v">−23%</div><div class="k">drift from the base model's image</div></div>
  <div class="la-stat"><div class="v">0</div><div class="k">extra network evaluations, no retraining</div></div>
</div>

<div class="la-card">
  <div class="la-compare-head">
    <span class="t">Drag to compare: four seeds, the same noise</span>
    <span class="la-seg" id="la-prompt" role="group" aria-label="prompt">
      <button aria-pressed="true" data-v="cats">Two cats</button><button aria-pressed="false" data-v="universe">Spheroid universe</button>
    </span>
    <span class="la-seg" id="la-right" role="group" aria-label="schedule on the right">
      <button aria-pressed="true" data-v="tau096">τ = 0.96</button><button aria-pressed="false" data-v="tau085">τ = 0.85</button><button aria-pressed="false" data-v="base">Base model</button>
    </span>
  </div>
  <div class="la-slider" id="la-slider" style="--pos: 50%" role="slider" aria-label="comparison divider" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
    <img class="before" id="la-before" src="/images/projects/2026-11-late-activation/compare-cats-nft.webp" alt="NFT samples for four seeds">
    <img class="after" id="la-after" src="/images/projects/2026-11-late-activation/compare-cats-tau096.webp" alt="Late activation samples for the same four seeds">
    <span class="lbl l">NFT, whole path</span><span class="lbl r" id="la-rlabel">Late activation, τ = 0.96</span>
    <div class="bar"></div><div class="knob">⇆</div>
  </div>
  <p class="la-caption">SD3.5-Medium tuned with DiffusionNFT on PickScore. Left of the handle: the adapter on the whole
  sampling path. Right: the same adapter switched on only below τ. NFT gives nearly one image for every seed; τ = 0.96
  brings back the base model's colors and backgrounds, τ = 0.85 its layouts.</p>
</div>

<div class="la-tldr"><b>TL;DR</b> The diversity that RL tuning destroys is lost in the first, high-noise sampling steps,
where the reward it adds is largely redundant. Training needs those steps; sampling does not.</div>

<h2 id="summary"><span class="la-num">01</span>Summary</h2>

<div class="la-lede">
  <div><span class="h">Problem</span><p>Online RL, here DiffusionNFT (NFT), aligns text-to-image flow models with a reward such
  as PickScore, a learned model of human preference. It also erodes variety: on SD3.5-Medium, the images for one prompt
  converge on one composition, and held-out CLIP diversity falls to 38% of the base model's.</p></div>
  <div><span class="h">Finding</span><p>The RL-tuned adapter can be switched off for part of the sampling path. Doing so by
  noise level shows that the collapse comes from the first, high-noise steps, where the adapter steers every seed toward
  the same layout, while the reward it adds there is largely recovered by the later steps.</p></div>
  <div><span class="h">Method</span><p><b>Late activation</b> samples the first steps with the base model and switches the
  adapter on below a noise level τ. Nothing is retrained, and each step still runs the network once.</p></div>
  <div><span class="h">Result</span><p>At τ = 0.96 it keeps 96% of the reward gain with 25% higher CLIP diversity and 23%
  less drift. At matched reward it beats early stopping and velocity or LoRA-weight scaling, and matches annealed guidance
  (AIG) at half its sampling time. It holds across three rewards and four RL methods.</p></div>
</div>

<div class="la-card">
  <p class="la-card-title">Reading the numbers</p>
  <div class="la-terms">
    <div><b>Noise level t and threshold τ</b><span>Sampling runs from pure noise (t = 1) to the image (t = 0). The adapter is on
    only for t ≤ τ; τ = 0.96 leaves the first 5 of 40 steps to the base model.</span></div>
    <div><b>Reward gain</b><span>Held-out PickScore of the tuned model minus the base model's. "Kept" is the share of NFT's
    gain that remains under a sampling schedule.</span></div>
    <div><b>CLIP diversity</b><span>Mean pairwise distance between the CLIP embeddings of the images drawn for one prompt:
    how different they look.</span></div>
    <div><b>Latent diversity</b><span>The same distance on the final latents: how different the images are in layout and
    pixels.</span></div>
    <div><b>Drift</b><span>Distance between the tuned model's image and the base model's image from the same noise: how far RL
    tuning moved each sample.</span></div>
    <div><b>Passes per step</b><span>Network evaluations per sampling step. Schedules that mix two models, such as AIG and
    velocity scaling, need two.</span></div>
  </div>
</div>

<h2 id="method"><span class="la-num">02</span>How it works</h2>

<div class="la-card">
<svg class="la-schematic" viewBox="0 0 760 150" role="img" aria-label="Sampling path: base model above tau, adapter below">
  <text x="20" y="28" font-size="13" fill="currentColor" opacity=".65">noise  t = 1</text>
  <text x="740" y="28" font-size="13" fill="currentColor" opacity=".65" text-anchor="end">t = 0  image</text>
  <rect x="20" y="44" width="150" height="34" rx="8" fill="#8c8c8c"/>
  <rect x="176" y="44" width="564" height="34" rx="8" fill="#D55E00"/>
  <text x="95" y="66" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">base model</text>
  <text x="458" y="66" font-size="13" font-weight="700" fill="#fff" text-anchor="middle">RL-tuned adapter</text>
  <line x1="173" x2="173" y1="36" y2="92" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3"/>
  <text x="173" y="108" font-size="14" font-weight="700" fill="currentColor" text-anchor="middle">τ</text>
  <text x="95" y="132" font-size="12" fill="currentColor" opacity=".7" text-anchor="middle">composes the layout</text>
  <text x="458" y="132" font-size="12" fill="currentColor" opacity=".7" text-anchor="middle">refines it toward the reward, one pass per step</text>
</svg>
<div class="la-steps">
  <div class="la-step"><span class="n">1</span><b>Locate the collapse</b>Switch the adapter off by noise level on a finished
  checkpoint: alone above t = 0.85 it reproduces 97% of NFT's loss of latent diversity.</div>
  <div class="la-step"><span class="n">2</span><b>Find the redundancy</b>Alone below t = 0.85, starting from the base
  model's composition, the adapter still reaches 79% of the reward gain.</div>
  <div class="la-step"><span class="n">3</span><b>Switch late</b>Sample above τ with the base model and below it with
  the adapter: a one-line change to the sampler, the same number of network evaluations.</div>
</div>
</div>

<div class="la-card">
  <p class="la-card-title">Try the switch on a 1-d toy</p>
  <p class="la-caption" style="margin-top:0">Exact probability-flow paths of a four-mode base distribution. The reward-tilted
  flow, in the role of the RL-tuned adapter, prefers the upper side of every mode and slightly the top one. Paths follow
  the base flow above τ (gray) and the tilted flow below it (orange). Each line is one sample; the four bumps on the right
  are the modes, standing in for different compositions.</p>
  <div class="la-controls">
    <label for="la-toy-tau">switch at τ = <b id="la-toy-val">0.65</b></label>
    <input id="la-toy-tau" type="range" min="0" max="1" step="0.01" value="0.65" aria-label="toy tau">
    <span class="la-seg" id="la-toy-preset" role="group" aria-label="presets">
      <button aria-pressed="false" data-v="0">Base</button><button aria-pressed="true" data-v="0.65">Late activation</button><button aria-pressed="false" data-v="1">Tilted path</button>
    </span>
  </div>
  <canvas class="la-canvas" id="la-toy" width="760" height="300" aria-label="toy sampling paths"></canvas>
  <div class="la-mini" id="la-toy-stats"></div>
  <p class="la-hint" id="la-toy-hint"></p>
</div>

<h2 id="results"><span class="la-num">03</span>Key results</h2>

<figure class="la-fig la-card">
  <img src="/images/projects/2026-11-late-activation/fig4-frontier.png" alt="Figure 4: held-out PickScore gain against CLIP and latent diversity">
  <figcaption><span class="badge">Figure 4</span>On the same adapter, late activation traces the best reward–diversity
  trade-off. Early stopping and velocity or LoRA-weight scaling weaken the adapter everywhere and give up reward where it
  costs little diversity; AIG comes closest but needs two network evaluations per step. Each curve is one way to trade
  reward for diversity on the same NFT adapter (star); up and to the right is better, and the numbers on the orange curve
  are τ.</figcaption>
</figure>

<div class="la-card">
  <p class="la-card-title">Move τ along the trade-off</p>
  <p class="la-caption" style="margin-top:0">Held-out results on the same adapter, relative to NFT on the whole path
  (256 prompts × 4 seeds, two training seeds). Lower τ hands more of the first steps to the base model.</p>
  <div class="la-controls">
    <label for="la-tau">τ = <b id="la-tau-val">0.96</b></label>
    <input id="la-tau" type="range" min="0" max="6" step="1" value="2" aria-label="tau">
    <span class="la-seg" id="la-metric" role="group" aria-label="diversity measure">
      <button aria-pressed="true" data-v="clip">CLIP diversity</button><button aria-pressed="false" data-v="lat">Latent diversity</button>
    </span>
  </div>
  <div class="la-mini" id="la-tau-stats"></div>
  <svg class="la-chart" id="la-chart" viewBox="0 0 640 300" role="img" aria-label="reward kept against diversity gain"></svg>
  <div class="la-legend" id="la-legend"></div>
  <ul class="la-list">
    <li><b>Velocity weight</b> mixes the base and tuned velocities with a constant weight at every step.</li>
    <li><b>LoRA scale</b> shrinks the adapter's weights by a constant factor.</li>
    <li><b>AIG-u</b> (annealed importance guidance) mixes the two velocities with a weight that grows as the noise falls.</li>
    <li><b>Early stopping</b> uses checkpoints from earlier in RL training.</li>
  </ul>
</div>

<div class="la-card">
<div class="la-table-wrap">
<table class="la-table">
  <thead><tr><th>Sampling (16 seeds per prompt)</th><th>Passes / step</th><th>ΔPickScore</th><th>Latent div.</th><th>CLIP div.</th><th>DINO Vendi</th></tr></thead>
  <tbody>
    <tr><td>Base model</td><td>1</td><td>0.00</td><td>1.073</td><td>0.275</td><td>8.24</td></tr>
    <tr><td>NFT, whole path</td><td>1</td><td>3.38</td><td>0.729</td><td>0.097</td><td>2.48</td></tr>
    <tr class="ours sep"><td>Late activation, τ = 0.96</td><td class="best">1</td><td>3.22</td><td class="best">0.881</td><td class="best">0.123</td><td>3.04</td></tr>
    <tr><td>AIG-u, p = 0.25</td><td>2</td><td>3.20</td><td>0.817</td><td class="best">0.123</td><td class="best">3.05</td></tr>
    <tr><td>Velocity weight 0.75</td><td>2</td><td>3.23</td><td>0.748</td><td>0.115</td><td>2.81</td></tr>
    <tr class="ours sep"><td>Late activation, τ = 0.85</td><td>1</td><td>2.56</td><td>1.009</td><td>0.182</td><td>4.48</td></tr>
  </tbody>
</table>
</div>
<p class="la-caption"><b>Table 2 (excerpt).</b> At about the same reward, late activation leads on latent and CLIP
diversity with half the network passes of AIG-u and velocity scaling; for more variety, τ = 0.85 nearly doubles DINO
Vendi over NFT. ΔPickScore is the gain over the base model; DINO Vendi is the effective number of distinct images among
the 16 drawn per prompt.</p>
</div>

<div class="la-card">
<div class="la-table-wrap">
<table class="la-table">
  <thead><tr><th>RL method, reward</th><th>Reward gain kept</th><th>CLIP diversity</th><th>Drift</th></tr></thead>
  <tbody>
    <tr><td>DiffusionNFT, PickScore</td><td>96%</td><td>+25%</td><td>−23%</td></tr>
    <tr><td>DiffusionNFT, HPSv2.1</td><td>95%</td><td>+30%</td><td>−22%</td></tr>
    <tr><td>DiffusionNFT, OCR</td><td>99%</td><td>+42% (latent)</td><td>n/a</td></tr>
    <tr><td>DiffusionART, PickScore</td><td>99%</td><td>+22%</td><td>−16%</td></tr>
    <tr><td>Flow-GRPO-Fast, PickScore</td><td>93%</td><td>+21%</td><td>−26%</td></tr>
    <tr><td>Flow-GRPO, released adapter (trained with guidance)</td><td>81%</td><td>+20%</td><td>−36%</td></tr>
  </tbody>
</table>
</div>
<p class="la-caption"><b>Table 3 (excerpt).</b> Late activation at τ = 0.96 against each adapter on its whole path. It keeps
93–99% of the gain for the adapters we trained, across three rewards and several RL methods. The adapter released by
Flow-GRPO's authors, trained with guidance, relies more on the first steps: τ = 0.98 keeps 92% of its gain.</p>
</div>

<figure class="la-fig la-card">
  <img src="/images/projects/2026-11-late-activation/fig2-endpoint-predictions.webp" alt="Figure 2: endpoint predictions along the sampling path">
  <figcaption><span class="badge">Figure 2</span>NFT's very first prediction already shows the layout of its final
  sample, for every seed. Under late activation the base model's prediction sets the layout and the adapter refines it.
  Each column shows the image the model predicts at that noise level, for one seed of two prompts; the frame color marks
  which model made the prediction (gray: base, blue: NFT, orange: adapter under late activation).</figcaption>
</figure>

<figure class="la-fig la-card is-narrow">
  <img src="/images/projects/2026-11-late-activation/fig5-reward-signal.png" alt="Figure 5: where the reward signal enters NFT">
  <figcaption><span class="badge">Figure 5</span>Training cannot skip the high-noise steps: the reward tilt above
  t = 0.85 alone nearly matches NFT, while the tilt below it alone ends below the base model. High noise is where the
  adapter learns, not where it should act. Each row is a training variant: the bar shows where along the noise level the
  loss pulls toward the reward, and the dot the reward it reaches after 50 rollouts.</figcaption>
</figure>

<figure class="la-fig la-card is-narrow">
  <img src="/images/projects/2026-11-late-activation/fig6-recipe.png" alt="Figure 6: a stronger training tilt with late activation">
  <figcaption><span class="badge">Figure 6</span>Over ten training seeds per arm, NFT with β = 0.5 and late activation at
  τ = 0.98 beats default NFT in reward (+0.11), CLIP diversity (+5%) and drift (−12%) at the same training and sampling
  cost. β sets how strongly NFT pulls toward the reward in training (lower is stronger): a stronger pull raises
  reward and costs diversity, and late activation gives the diversity back.</figcaption>
</figure>

<h2 id="samples"><span class="la-num">04</span>More samples</h2>

<div class="la-card">
  <div class="la-chips" id="la-prompts" role="group" aria-label="prompt"></div>
  <p class="la-prompt" id="la-prompt-text"></p>
  <div class="la-gallery" id="la-gallery"></div>
  <p class="la-caption">Held-out prompts, four seeds each, from the same noise in every row. Compare the columns within a
  row: NFT's four images share one composition, τ = 0.96 varies colors and backgrounds again, and τ = 0.85 returns to
  the base model's layouts with NFT's rendering. Prompts showing real people or named characters are left out.</p>
</div>

<h2 id="usage"><span class="la-num">05</span>Using it</h2>

<div class="la-card">
<div class="la-two">
<div>
  <p class="la-card-title">A gate around the network call</p>
  <p class="la-caption" style="margin-top:0">Above τ the sampler calls the network with the adapter disabled. The checkpoint and
  the sampler stay as they are, and τ can change per request.</p>
<pre class="la-code"><span class="k">def</span> velocity(x, t, prompt, tau=0.96):
    <span class="k">if</span> t > tau:                    <span class="c"># first, high-noise steps</span>
        <span class="k">with</span> adapter_disabled():   <span class="c"># base model composes</span>
            <span class="k">return</span> model(x, t, prompt)
    <span class="k">return</span> model(x, t, prompt)     <span class="c"># RL-tuned adapter refines</span></pre>
</div>
<div>
  <p class="la-card-title">Choosing τ</p>
  <ul class="la-list" style="margin-top:.2em">
    <li><b>τ = 0.96</b> is a good default: it gives up about 5% of the reward gain for NFT on PickScore and HPSv2.1.</li>
    <li><b>τ = 0.98</b> suits adapters that rely on the first steps, such as those trained with guidance.</li>
    <li><b>τ = 0.85</b> gives more variety: a third to a half of NFT's perceptual loss back for a fifth to a quarter of the gain.</li>
    <li>τ is a noise level, so it means the same for any number of steps. Checking one value takes a 15–20 GPU-minute
    held-out evaluation.</li>
  </ul>
</div>
</div>
<p class="la-caption"><b>Scope.</b> All results are on SD3.5-Medium at 512 × 512 with LoRA adapters; other base models such as
FLUX remain to be tested.</p>
</div>

<h2 id="citation"><span class="la-num">06</span>Paper and citation</h2>

<div class="la-card"><p class="la-cite" style="margin:0">The paper is not yet public. The PDF, code and BibTeX will be
linked here once they are public.</p></div>

</div>

<script>
(function () {
  var base = "/images/projects/2026-11-late-activation/compare-";
  var slider = document.getElementById("la-slider"), before = document.getElementById("la-before"),
      after = document.getElementById("la-after"), rlabel = document.getElementById("la-rlabel");
  var state = {prompt: "cats", right: "tau096"};
  var names = {tau096: "Late activation, τ = 0.96", tau085: "Late activation, τ = 0.85", base: "Base model"};
  function render() {
    before.src = base + state.prompt + "-nft.webp";
    after.src = base + state.prompt + "-" + state.right + ".webp";
    rlabel.textContent = names[state.right];
  }
  function bind(id, key) {
    var group = document.getElementById(id);
    group.addEventListener("click", function (e) {
      var b = e.target.closest("button"); if (!b) return;
      [].forEach.call(group.children, function (c) { c.setAttribute("aria-pressed", String(c === b)); });
      state[key] = b.dataset.v; render();
    });
  }
  bind("la-prompt", "prompt"); bind("la-right", "right");
  function set(p) {
    p = Math.max(0, Math.min(100, p));
    slider.style.setProperty("--pos", p + "%"); slider.setAttribute("aria-valuenow", Math.round(p));
  }
  function move(x) { var r = slider.getBoundingClientRect(); set((x - r.left) / r.width * 100); }
  var dragging = false;
  slider.addEventListener("pointerdown", function (e) { dragging = true; slider.setPointerCapture(e.pointerId); move(e.clientX); });
  slider.addEventListener("pointermove", function (e) { if (dragging) move(e.clientX); });
  slider.addEventListener("pointerup", function () { dragging = false; });
  slider.tabIndex = 0;
  slider.addEventListener("keydown", function (e) {
    var p = parseFloat(slider.style.getPropertyValue("--pos")) || 50;
    if (e.key === "ArrowLeft") { set(p - 5); e.preventDefault(); }
    if (e.key === "ArrowRight") { set(p + 5); e.preventDefault(); }
  });
  var root = document.querySelector(".la");
  function cssVar(n) { return getComputedStyle(root).getPropertyValue(n).trim(); }
  function segment(id, fn) {
    var g = document.getElementById(id);
    g.addEventListener("click", function (e) {
      var b = e.target.closest("button"); if (!b) return;
      [].forEach.call(g.children, function (c) { c.setAttribute("aria-pressed", String(c === b)); });
      fn(b.dataset.v);
    });
    return g;
  }

  /* toy: exact rectified-flow paths, v(x, t) = (x - E[x0 | x_t = x]) / t, base prior above tau, tilted prior below */
  var MU = [-7.5, -2.5, 2.5, 7.5], SD = 0.7, ETA = 2.5, NG = 1201, NS = 24, STEPS = 240, X0 = [], LB = [], LT = [];
  for (var i = 0; i < NG; i++) X0.push(-11 + 22 * i / (NG - 1));
  function nearest(x) { return MU.reduce(function (b, m) { return Math.abs(x - m) < Math.abs(x - b) ? m : b; }, MU[0]); }
  function reward(x) { return 0.15 * x + 0.5 * Math.max(-1.5, Math.min(1.5, (x - nearest(x)) / SD)); }
  X0.forEach(function (x) {
    var a = MU.map(function (m) { return -0.5 * Math.pow((x - m) / SD, 2); }), mx = Math.max.apply(null, a);
    var lb = mx + Math.log(a.reduce(function (s, v) { return s + Math.exp(v - mx); }, 0));
    LB.push(lb); LT.push(lb + ETA * reward(x));
  });
  function invNorm(p) {  /* Acklam's approximation of the standard normal quantile */
    var a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239],
        b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572],
        c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783],
        d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416], q = Math.min(p, 1 - p), x, r;
    if (q < 0.02425) { r = Math.sqrt(-2 * Math.log(q));
      x = (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1); return p < 0.5 ? x : -x; }
    r = q - 0.5; var s2 = r * r;
    x = (((((a[0]*s2+a[1])*s2+a[2])*s2+a[3])*s2+a[4])*s2+a[5])*r / (((((b[0]*s2+b[1])*s2+b[2])*s2+b[3])*s2+b[4])*s2+1);
    return p < 0.5 ? x : -x;
  }
  var START = [], TS = [];
  for (i = 0; i < NS; i++) START.push(invNorm((i + 0.5) / NS));
  for (i = 0; i < STEPS; i++) TS.push(1 - 0.998 * i / (STEPS - 1));
  function paths(tau) {
    var out = START.map(function (x) { return [x]; }), xs = START.slice(), w = new Float64Array(NG);
    for (var k = 0; k < STEPS - 1; k++) {
      var t0 = TS[k], t1 = TS[k + 1], lp = t0 <= tau ? LT : LB, inv = 1 / (2 * t0 * t0);
      for (var s = 0; s < NS; s++) {
        var x = xs[s], mx = -Infinity, z = 0, e = 0, j, dd;
        for (j = 0; j < NG; j++) { dd = x - (1 - t0) * X0[j]; w[j] = lp[j] - dd * dd * inv; if (w[j] > mx) mx = w[j]; }
        for (j = 0; j < NG; j++) { var v = Math.exp(w[j] - mx); z += v; e += v * X0[j]; }
        xs[s] = x - (t0 - t1) * (x - e / z) / t0; out[s].push(xs[s]);
      }
    }
    return out;
  }
  function summary(ps) {
    var f = ps.map(function (p) { return p[p.length - 1]; }), r = f.reduce(function (s, x) { return s + reward(x); }, 0) / f.length;
    var modes = {}; f.forEach(function (x) { modes[nearest(x)] = 1; });
    return {modes: Object.keys(modes).length, reward: Math.abs(r) < 0.005 ? 0 : r};
  }
  var REF = {base: summary(paths(0)), tilt: summary(paths(1))}, toyIn = document.getElementById("la-toy-tau"), toyFrame = 0;
  function drawToy() {
    var tau = +toyIn.value, cv = document.getElementById("la-toy"), g = cv.getContext("2d"), dpr = window.devicePixelRatio || 1;
    cv.width = 760 * dpr; cv.height = 300 * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    var W = 760, H = 300, L = 18, R = W - 130, T = 12, B = H - 26;
    function X(t) { return L + (1 - t) * (R - L); }
    function Y(x) { return T + (10.5 - x) / 21 * (B - T); }
    var base = cssVar("--la-base"), ours = cssVar("--la-accent"), nft = cssVar("--la-nft"), muted = cssVar("--la-muted"),
        ink = getComputedStyle(root).color, ps = paths(tau);
    g.lineWidth = 1.1;
    ps.forEach(function (p) {
      for (var k = 0; k < STEPS - 1; k++) {
        g.strokeStyle = tau >= 1 ? nft : TS[k] > tau ? base : ours;
        g.beginPath(); g.moveTo(X(TS[k]), Y(p[k])); g.lineTo(X(TS[k + 1]), Y(p[k + 1])); g.stroke();
      }
    });
    if (tau > 0 && tau < 1) { g.setLineDash([3, 3]); g.strokeStyle = muted; g.beginPath(); g.moveTo(X(tau), T); g.lineTo(X(tau), B); g.stroke(); g.setLineDash([]); }
    var mx = Math.max.apply(null, LB);
    g.fillStyle = base; g.globalAlpha = 0.3; g.beginPath(); g.moveTo(R + 10, Y(-10.5));
    X0.forEach(function (x, i) { g.lineTo(R + 10 + 50 * Math.exp(LB[i] - mx), Y(x)); }); g.lineTo(R + 10, Y(10.5)); g.fill();
    g.globalAlpha = 1; g.strokeStyle = ink; g.lineWidth = 1; g.beginPath();
    X0.forEach(function (x, i) { var px = R + 82 + 20 * reward(x) / 2.2; if (i) g.lineTo(px, Y(x)); else g.moveTo(px, Y(x)); }); g.stroke();
    g.fillStyle = muted; g.font = "12px sans-serif";
    g.fillText("t = 1 (noise)", L, H - 8); g.fillText("t = 0 (image)", R - 78, H - 8); g.fillText("base density", R + 8, H - 8);
    g.fillText("reward", R + 76, T + 10);
    if (tau > 0 && tau < 1) g.fillText("τ", X(tau) + 4, B - 4);
    var s = summary(ps);
    document.getElementById("la-toy-stats").innerHTML =
      '<div><div class="v up">' + s.modes + ' of 4</div><div class="k">modes reached (base ' + REF.base.modes + ', tilted path ' + REF.tilt.modes + ')</div></div>' +
      '<div><div class="v up">' + s.reward.toFixed(2) + '</div><div class="k">mean reward (base ' + REF.base.reward.toFixed(2) + ', tilted path ' + REF.tilt.reward.toFixed(2) + ')</div></div>';
    document.getElementById("la-toy-hint").textContent = tau <= 0.02 ? "Base flow only: every mode is reached, and the reward stays at the base level."
      : s.modes < REF.base.modes ? "Tilted from high noise: samples leave their modes for the higher-reward ones, so the reward is highest and a mode is lost. This is the collapse."
      : "Late activation: each sample stays in the mode the base flow chose and moves to that mode's high-reward side. All modes survive, with part of the reward.";
  }
  var toyPreset = segment("la-toy-preset", function (v) { setToy(v, true); });
  function setToy(v, fromPreset) {
    toyIn.value = v; document.getElementById("la-toy-val").textContent = (+v).toFixed(2);
    if (!fromPreset) [].forEach.call(toyPreset.children, function (c) { c.setAttribute("aria-pressed", String(+c.dataset.v === +v)); });
    cancelAnimationFrame(toyFrame); toyFrame = requestAnimationFrame(drawToy);
  }
  toyIn.addEventListener("input", function () { setToy(toyIn.value); });

  /* trade-off explorer: two-seed means relative to NFT on the whole path */
  var TAU = [
    {tau: 1.0, off: 0, kept: 100.0, clip: 0.0, lat: 0.0, drift: 0.0}, {tau: 0.98, off: 3, kept: 98.5, clip: 14.1, lat: 12.8, drift: -13.8},
    {tau: 0.96, off: 5, kept: 96.5, clip: 23.9, lat: 20.3, drift: -22.7}, {tau: 0.93, off: 8, kept: 91.7, clip: 41.7, lat: 28.5, drift: -34.0},
    {tau: 0.90, off: 10, kept: 87.7, clip: 53.5, lat: 32.2, drift: -40.6}, {tau: 0.85, off: 14, kept: 78.3, clip: 79.4, lat: 37.6, drift: -52.5},
    {tau: 0.70, off: 23, kept: 50.6, clip: 130.4, lat: 43.3, drift: -74.8}];
  var FULL = {kept: 100, clip: 0, lat: 0};
  var BASELINES = [
    ["AIG-u (2 passes)", "--la-aigu", false, [{kept: 89.2, clip: 47.3, lat: 19.8}, {kept: 96.0, clip: 23.6, lat: 11.7}, FULL]],
    ["Velocity weight (2 passes)", "--la-weight", false, [{kept: 84.6, clip: 49.6, lat: 8.4}, {kept: 96.1, clip: 16.3, lat: 2.4}, FULL]],
    ["LoRA scale (1 pass)", "--la-weight", true, [{kept: 83.4, clip: 40.4, lat: 15.4}, {kept: 93.9, clip: 15.1, lat: 6.2}, {kept: 98.1, clip: 4.2, lat: 2.1}, FULL]],
    ["Early stopping", "--la-early", false, [{kept: 85.0, clip: 13.6, lat: 8.1}, {kept: 90.7, clip: 5.1, lat: 4.0}, {kept: 94.0, clip: 2.9, lat: 2.8},
      {kept: 95.6, clip: 2.2, lat: 2.2}, {kept: 97.7, clip: 1.0, lat: 1.0}, {kept: 99.2, clip: 1.0, lat: 0.3}, FULL]]];
  var tauIn = document.getElementById("la-tau"), metric = "clip";
  function fmt(v) { return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1) + "%"; }
  function drawChart() {
    var p = TAU[+tauIn.value], m = metric, svg = document.getElementById("la-chart");
    document.getElementById("la-tau-val").textContent = p.tau.toFixed(2);
    document.getElementById("la-tau-stats").innerHTML = [
      ["reward gain kept", p.kept.toFixed(1) + "%", false], ["CLIP diversity vs NFT", fmt(p.clip), p.off > 0],
      ["latent diversity vs NFT", fmt(p.lat), p.off > 0], ["drift from base vs NFT", fmt(p.drift), p.off > 0],
      ["steps on the base model", p.off + " of 40", false]].map(function (c) {
        return '<div><div class="v' + (c[2] ? " up" : "") + '">' + c[1] + '</div><div class="k">' + c[0] + '</div></div>'; }).join("");
    var ymax = m === "clip" ? 140 : 50, y0 = m === "clip" ? 5 : 2, L = 64, R = 624, T = 12, B = 256;
    function X(k) { return L + (k - 48) / (106 - 48) * (R - L); }
    function Y(v) { return B - (v + y0) / (ymax + y0) * (B - T); }
    var s = "", k, v;
    for (k = 50; k <= 100; k += 10) s += '<line x1="' + X(k) + '" x2="' + X(k) + '" y1="' + T + '" y2="' + B + '" stroke="var(--la-line)"/>' +
      '<text x="' + X(k) + '" y="' + (B + 16) + '" font-size="12" fill="var(--la-muted)" text-anchor="middle">' + k + '%</text>';
    for (v = 0; v <= ymax; v += m === "clip" ? 20 : 10) s += '<line x1="' + L + '" x2="' + R + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="var(--la-line)"/>' +
      '<text x="' + (L - 6) + '" y="' + (Y(v) + 4) + '" font-size="12" fill="var(--la-muted)" text-anchor="end">+' + v + '%</text>';
    s += '<text x="' + ((L + R) / 2) + '" y="' + (B + 36) + '" font-size="12.5" fill="var(--la-muted)" text-anchor="middle">PickScore gain kept (%)</text>';
    s += '<text transform="translate(14 ' + ((T + B) / 2) + ') rotate(-90)" font-size="12.5" fill="var(--la-muted)" text-anchor="middle">' +
      (m === "clip" ? "CLIP" : "latent") + ' diversity over NFT</text>';
    function line(pts, color, dashed, w) {
      return '<polyline fill="none" stroke="var(' + color + ')" stroke-width="' + w + '"' + (dashed ? ' stroke-dasharray="5 3"' : "") +
        ' points="' + pts.map(function (q) { return X(q.kept) + "," + Y(q[m]); }).join(" ") + '"/>' +
        pts.map(function (q) { return '<circle cx="' + X(q.kept) + '" cy="' + Y(q[m]) + '" r="' + (w > 2 ? 3.5 : 2.6) + '" fill="' +
          (dashed ? "var(--la-card)" : "var(" + color + ")") + '" stroke="var(' + color + ')"/>'; }).join("");
    }
    BASELINES.forEach(function (b) { s += line(b[3], b[1], b[2], 1.5); });
    s += line(TAU, "--la-accent", false, 3);
    s += '<circle cx="' + X(p.kept) + '" cy="' + Y(p[m]) + '" r="8" fill="none" stroke="var(--la-accent)" stroke-width="2"/>' +
      '<text x="' + (X(p.kept) + 13) + '" y="' + (Y(p[m]) - 11) + '" font-size="12.5" font-weight="700" fill="var(--la-accent)" text-anchor="start">τ = ' + p.tau.toFixed(2) + '</text>' +
      '<text x="' + (X(100) + 9) + '" y="' + (Y(0) + 4) + '" font-size="12" font-weight="700" fill="var(--la-nft)" text-anchor="start">NFT</text>';
    svg.innerHTML = s;
    document.getElementById("la-legend").innerHTML = [["Late activation (1 pass)", "--la-accent"]].concat(BASELINES.map(function (b) { return [b[0], b[1]]; }))
      .map(function (c) { return '<span style="--c:var(' + c[1] + ')">' + c[0] + '</span>'; }).join("");
  }
  tauIn.addEventListener("input", drawChart);
  segment("la-metric", function (v) { metric = v; drawChart(); });

  /* gallery */
  var PROMPTS = [[1119, "Two adorable cats. One black and white cat and one orange tabby"],
    [1425, "The universe is a spheroid region 705 meters in diameter by Clara Peeters"],
    [913, "a painting of a big tree on the style of vincent Van gogh"], [2022, "a black Rover75 car being driven through a river"],
    [562, "old crypt, vintage, hyperrealistic, glowing, abandoned"],
    [2013, "cans stacked on top of each other in a pyramic shape at the grocery store"], [327, "vector logo of a viper snake"]];
  var ROWS = [["base", "Base model", "--la-base"], ["nft", "NFT, whole path", "--la-nft"],
    ["tau096", "Late activation, τ = 0.96", "--la-accent"], ["tau085", "Late activation, τ = 0.85", "--la-accent"]];
  var chips = document.getElementById("la-prompts");
  PROMPTS.forEach(function (pr, i) {
    var b = document.createElement("button"); b.textContent = pr[1].split(/[ ,.]+/).slice(0, 3).join(" "); b.title = pr[1];
    b.setAttribute("aria-pressed", "false"); b.onclick = function () { showPrompt(i); }; chips.appendChild(b);
  });
  function showPrompt(i) {
    var pr = PROMPTS[i];
    [].forEach.call(chips.children, function (c, j) { c.setAttribute("aria-pressed", String(j === i)); });
    document.getElementById("la-prompt-text").textContent = "“" + pr[1] + "”";
    document.getElementById("la-gallery").innerHTML = ROWS.map(function (r) {
      return '<div class="lab" style="border-color:var(' + r[2] + ')">' + r[1] + '</div><img src="/images/projects/2026-11-late-activation/samples/' +
        pr[0] + '-' + r[0] + '.webp" alt="' + r[1] + ': four seeds" loading="lazy">'; }).join("");
  }

  showPrompt(0); drawChart(); setToy(0.65);
  new MutationObserver(function () { drawChart(); drawToy(); }).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
})();
</script>
