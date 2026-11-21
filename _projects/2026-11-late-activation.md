---
title: "Late Activation: A Better Reward–Diversity Trade-off for RL-Tuned Flow Models"
collection: projects
permalink: /projects/2026-11-late-activation/
date: 2026-11-16
status: "Under review"
venue: "Under review"
excerpt: "Let the base model take the first sampling steps of an RL-tuned flow model: 96% of the reward gain, 25% more CLIP diversity, no retraining and no extra compute."
teaser: "/images/projects/2026-11-late-activation/teaser.webp"
tags: ["diffusion models", "reinforcement learning", "sample diversity"]
publication: "/publication/2026-11-late-activation"
toc: false   # the floating table of contents covers the long title; the page has its own section links
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
.la .la-fig figcaption { font-size: .88em; line-height: 1.6; margin-top: .8em; color: inherit; }
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
.la-demo { padding: 0; overflow: hidden; }
.la-demo iframe { display: block; width: 100%; border: 0; min-height: 600px; }
.la p.la-cite { font-size: .9em; color: var(--la-muted); }
@media (max-width: 720px) {
  .la-stats { grid-template-columns: repeat(2, 1fr); }
  .la-steps { grid-template-columns: 1fr; }
  .la-slider { aspect-ratio: 321 / 157; }
  .la-slider img { object-position: left center; }
}
</style>

<div class="la">

<div class="la-kicker">
  <span class="la-tag is-accent">Under review · 2026</span>
  <span class="la-tag">Text-to-image</span>
  <span class="la-tag">RL fine-tuning</span>
  <span class="la-tag">Sample diversity</span>
</div>

<p class="la-tagline">RL-tuned flow models draw the same picture for every seed. Let the base model take the first few
sampling steps and the variety comes back: <b>almost the same reward, more diverse images, no retraining and no extra
compute.</b></p>

<div class="la-buttons">
  <a class="la-btn is-primary" href="#demo"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>Interactive demo</a>
  <a class="la-btn" href="#results"><svg viewBox="0 0 24 24"><path d="M5 9h3v10H5zm5.5-4h3v14h-3zM16 13h3v6h-3z"/></svg>Key results</a>
  <span class="la-btn is-off"><svg viewBox="0 0 24 24"><path d="M6 2h9l5 5v15H6zm8 1.5V8h4.5z"/></svg>Paper · coming soon</span>
  <span class="la-btn is-off"><svg viewBox="0 0 24 24"><path d="M9.4 16.6 4.8 12l4.6-4.6L8 6l-6 6 6 6zm5.2 0 4.6-4.6-4.6-4.6L16 6l6 6-6 6z"/></svg>Code · coming soon</span>
</div>

<div class="la-stats">
  <div class="la-stat"><div class="v">96%</div><div class="k">of NFT's reward gain kept (τ = 0.96)</div></div>
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

<p>Online RL has become a standard way to align text-to-image flow models with human preferences, but it erodes
variety: after tuning SD3.5-Medium with DiffusionNFT, held-out CLIP diversity falls to 38% of the base model's.
Switching the RL-tuned adapter off by noise level shows where this happens. The collapse comes from the first, high-noise
steps, where the adapter steers every seed toward the same layout, while the reward it adds there is largely recovered
by the later steps. <b>Late activation</b> therefore samples the first steps with the base model and switches the
adapter on below a noise level τ. It keeps 96% of the reward gain with 25% higher CLIP diversity and 23% less drift,
beats early stopping and velocity or LoRA-weight scaling at matched reward, matches annealed guidance at half its
sampling time, and holds across three rewards and four RL methods.</p>

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

<h2 id="results"><span class="la-num">03</span>Key results</h2>

<figure class="la-fig la-card">
  <img src="/images/projects/2026-11-late-activation/fig4-frontier.png" alt="Figure 4: held-out PickScore gain against CLIP and latent diversity">
  <figcaption><span class="badge">Figure 4</span>On the same adapter, late activation traces the best reward–diversity
  trade-off. Early stopping and velocity or LoRA-weight scaling weaken the adapter everywhere and give up reward where it
  costs little diversity; AIG comes closest but needs two network evaluations per step.</figcaption>
</figure>

<div class="la-card">
<div class="la-table-wrap">
<table class="la-table">
  <thead><tr><th>Sampling, 16 seeds per prompt</th><th>Passes / step</th><th>ΔPickScore</th><th>Latent div.</th><th>CLIP div.</th><th>DINO Vendi</th></tr></thead>
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
Vendi over NFT.</p>
</div>

<figure class="la-fig la-card">
  <img src="/images/projects/2026-11-late-activation/fig2-endpoint-predictions.webp" alt="Figure 2: endpoint predictions along the sampling path">
  <figcaption><span class="badge">Figure 2</span>NFT's very first prediction already shows the layout of its final
  sample, for every seed. Under late activation the base model's prediction sets the layout and the adapter refines it.</figcaption>
</figure>

<figure class="la-fig la-card is-narrow">
  <img src="/images/projects/2026-11-late-activation/fig5-reward-signal.png" alt="Figure 5: where the reward signal enters NFT">
  <figcaption><span class="badge">Figure 5</span>Training cannot skip the high-noise steps: the reward tilt above
  t = 0.85 alone nearly matches NFT, while the tilt below it alone ends below the base model. High noise is where the
  adapter learns, not where it should act.</figcaption>
</figure>

<figure class="la-fig la-card is-narrow">
  <img src="/images/projects/2026-11-late-activation/fig6-recipe.png" alt="Figure 6: a stronger training tilt with late activation">
  <figcaption><span class="badge">Figure 6</span>Over ten training seeds per arm, NFT with β = 0.5 and late activation at
  τ = 0.98 beats default NFT in reward (+0.11), CLIP diversity (+5%) and drift (−12%) at the same training and sampling
  cost.</figcaption>
</figure>

<h2 id="demo"><span class="la-num">04</span>Interactive demo</h2>

<p>Drag τ on a 1-d toy with exact flows, browse samples from the same noise under each schedule, and follow reward and
diversity along the trade-off.</p>

<div class="la-card la-demo">
  <iframe id="la-demo-frame" src="/files/projects/2026-11-late-activation/demo/index.html?embed=1" height="1500" loading="lazy" title="Late activation interactive demo"></iframe>
</div>

<h2 id="citation"><span class="la-num">05</span>Paper and citation</h2>

<div class="la-card"><p class="la-cite" style="margin:0">The paper is under review. The PDF, code and BibTeX will be
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
  // the demo reports its height and follows the site's light or dark theme
  var frame = document.getElementById("la-demo-frame");
  window.addEventListener("message", function (e) {
    if (e.source === frame.contentWindow && e.data && e.data.laDemoHeight) frame.style.height = e.data.laDemoHeight + "px";
  });
  function sendTheme() {
    if (frame.contentWindow) frame.contentWindow.postMessage({laTheme: document.documentElement.getAttribute("data-theme") || "light"}, "*");
  }
  frame.addEventListener("load", sendTheme);
  new MutationObserver(sendTheme).observe(document.documentElement, {attributes: true, attributeFilter: ["data-theme"]});
})();
</script>
