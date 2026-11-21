---
title: "StepShare: Diagnosing and Rebalancing the Generation Gap in Native Unified Multimodal Models"
collection: projects
permalink: /projects/2026-11-stepshare/
date: 2026-11-16
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "At equal loss weights, generation in a native unified model is starved rather than conflicted: it gets under 1% of each AdamW step. StepShare measures and rebalances that share; StepShare-Adam closes 91% of the FID gap with no hidden probe cost and no added parameters."
teaser: "/images/projects/2026-11-stepshare/teaser.png"
tags: ["unified multimodal models", "multi-task optimisation", "flow matching", "AdamW"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-stepshare"
---

<style>
/* Page-scoped styles: everything is under .ss so no other page changes. */
.ss { --c-und: #2a78d6; --c-gen: #eb6834; --c-wgt: #6250d6; --c-rte: #1baf7a; --c-spl: #008300; --c-opt: #2b2b29;
  --ink: #1d1d1b; --ink2: #52514e; --muted: #898781; --line: rgba(11, 11, 11, 0.09); --card: #ffffff; --soft: #f6f6f3;
  --cost: #b9b8b1; --shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 10px 30px rgba(11, 11, 11, 0.06); color: var(--ink); }
html[data-theme="dark"] .ss { --c-und: #3987e5; --c-gen: #d95926; --c-wgt: #9085e9; --c-rte: #1aa5a0; --c-spl: #2a9d2a; --c-opt: #e6e5e0;
  --ink: #f3f3f1; --ink2: #d0cfc8; --muted: #a9a8a1; --line: rgba(255, 255, 255, 0.10); --card: #3b3b3b; --soft: #404040;
  --cost: #6f6e69; --shadow: 0 1px 2px rgba(0, 0, 0, 0.25), 0 10px 30px rgba(0, 0, 0, 0.2); }
.ss p { margin: 0 0 0.9em; }
.ss-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 28px 28px 24px; margin: 0.4em 0 1.4em; border: 1px solid var(--line);
  background: radial-gradient(70% 120% at 0% 0%, rgba(42, 120, 214, 0.13), transparent 60%), radial-gradient(60% 120% at 100% 0%, rgba(235, 104, 52, 0.13), transparent 60%), var(--card);
  box-shadow: var(--shadow); }
.ss-hero::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: linear-gradient(90deg, var(--c-und) 0%, var(--c-und) 45%, var(--c-gen) 100%); }
.ss-kicker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.72em; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin-bottom: 10px; }
.ss-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; background: var(--ink); color: var(--card); font-weight: 700; letter-spacing: 0.04em; }
.ss .ss-name { font-size: 2.6em; line-height: 1.05; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 6px; }
.ss .ss-name span { background: linear-gradient(90deg, var(--c-und), var(--c-gen)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.ss .ss-tagline { font-size: 1.25em; line-height: 1.35; font-weight: 600; margin: 0 0 14px; }
.ss .ss-tagline em { font-style: normal; color: var(--c-gen); }
.ss .ss-lede { font-size: 0.95em; color: var(--ink2); max-width: 46em; }
.ss-btns { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 20px; }
.ss .ss-btn { display: inline-flex; align-items: center; gap: 7px; padding: 7px 15px; border-radius: 999px; font-size: 0.85em; font-weight: 600; text-decoration: none; border: 1px solid var(--ink); }
.ss .ss-btn.primary { background: var(--ink); color: var(--card); }
.ss .ss-btn.ghost { background: transparent; color: var(--ink); }
.ss .ss-btn.soon { background: transparent; color: var(--muted); border-color: var(--line); cursor: default; }
.ss .ss-btn svg { flex: none; }
.ss-split { display: grid; gap: 9px; padding: 14px 16px 12px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); }
.ss-split .row { display: grid; grid-template-columns: 11.5em 1fr 3.6em; align-items: center; gap: 10px; font-size: 0.82em; }
.ss-split .row b { font-weight: 600; }
.ss-split .bar { display: flex; gap: 2px; height: 26px; }
.ss-split .bar i { display: flex; align-items: center; padding-left: 8px; border-radius: 5px; color: #fff; font-style: normal; font-size: 0.86em; white-space: nowrap; overflow: hidden; }
.ss-split .val { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }
.ss-split .cap { font-size: 0.75em; color: var(--muted); }
.ss-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 1.6em; }
.ss-stat { position: relative; padding: 14px 14px 12px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }
.ss-stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.ss-stat .num { font-size: 1.9em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.ss-stat .lab { font-size: 0.78em; color: var(--ink2); line-height: 1.35; margin-top: 4px; }
.ss .ss-k { font-size: 0.7em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 1.8em 0 0.1em; }
.ss h3.ss-h { font-size: 1.1em; margin: 0 0 0.55em; }
.ss-fig { margin: 0 0 1.2em; padding: 14px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); }
.ss-fig .img { background: #fff; border-radius: 10px; padding: 6px; }
.ss-fig img { display: block; width: 100%; height: auto; margin: 0 auto; }
.ss .ss-fig figcaption { font-family: inherit; font-size: 0.82em; color: var(--ink2); line-height: 1.5; margin: 10px 4px 0; }
.ss-fig figcaption strong { color: var(--ink); }
.ss-fig .take { display: block; margin-top: 6px; padding: 7px 10px; border-radius: 8px; background: var(--soft); color: var(--ink); border-left: 3px solid var(--accent, var(--ink2)); }
.ss-method { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 16px; align-items: center; }
.ss-steps { display: grid; gap: 10px; }
.ss-step { padding: 12px 14px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); font-size: 0.86em; line-height: 1.5; }
.ss-step .t { font-weight: 700; display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.ss-step .t span { display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.ss .ss-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.82em; margin: 0; display: table; border: 0; background: transparent; }
.ss .ss-table thead, .ss .ss-table thead tr, .ss .ss-table tbody tr { background: transparent; }
.ss .ss-table th, .ss .ss-table td { border: 0; border-bottom: 1px solid var(--line); background: transparent; }
.ss .ss-table th { text-align: left; font-weight: 600; color: var(--ink2); padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; background: transparent; }
.ss .ss-table td { padding: 9px 10px; border-bottom: 1px solid var(--line); vertical-align: middle; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ss .ss-table tr.hl td { background: rgba(0, 131, 0, 0.07); font-weight: 600; }
.ss-table tr.hl td:first-child { box-shadow: inset 3px 0 0 var(--c-spl); }
.ss-table .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; vertical-align: 0; }
.ss-table .cell { display: flex; align-items: center; gap: 8px; }
.ss-table .bar { flex: none; width: 70px; height: 7px; border-radius: 4px; background: var(--soft); overflow: hidden; }
.ss-table .bar i { display: block; height: 100%; border-radius: 4px; }
.ss-table .ci { color: var(--muted); font-weight: 400; }
.ss-tablewrap { overflow-x: auto; }
.ss-demo { margin: 0.6em 0 1.2em; padding: 16px; border-radius: 18px; border: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(42, 120, 214, 0.06), rgba(235, 104, 52, 0.05)), var(--soft); }
.ss-demo .intro { margin: 2px 4px 12px; font-size: 0.88em; color: var(--ink2); }
.ss-demo iframe { display: block; width: 100%; border: 0; background: transparent; }
.ss-links { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.ss-link { padding: 14px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); font-size: 0.86em; }
.ss-link .h { font-weight: 700; margin-bottom: 2px; }
.ss-link .s { color: var(--muted); }
@media (max-width: 720px) {
  .ss-hero { padding: 22px 18px 18px; }
  .ss .ss-name { font-size: 2.1em; }
  .ss-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ss-method { grid-template-columns: minmax(0, 1fr); }
  .ss-split .row { grid-template-columns: 1fr 3.2em; }
  .ss-split .row b { grid-column: 1 / -1; }
  .ss-links { grid-template-columns: minmax(0, 1fr); }
}
</style>

<div class="ss">

<section class="ss-hero" aria-label="Overview">
  <div class="ss-kicker"><span class="ss-badge">Under review</span><span>Unified multimodal models · Multi-task optimisation</span></div>
  <p class="ss-name"><span>StepShare</span></p>
  <p class="ss-tagline">In a native unified model, generation is <em>starved</em>, not conflicted.</p>
  <p class="ss-lede">Unified multimodal models read and generate images with one backbone, and the generation gap that joint training opens is usually blamed on parameter sharing. <strong>StepShare</strong> splits every AdamW step exactly into an understanding part and a generation part, by depth: at equal loss weights the two gradients are nearly orthogonal, but the understanding gradient is about 100 times larger, so generation receives under 1% of the update. Restoring the share closes most of the FID gap without added parameters, more than task-specific routing, yet quietly lowers intermediate-layer probe accuracy by up to 7 points. <strong>StepShare-Adam</strong>, which restores the share with per-task Adam and allocates it by depth, closes 91% of the gap with essentially no probe cost.</p>
  <div class="ss-btns">
    <a class="ss-btn primary" href="#interactive-demo"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M3 1.5v10l8.5-5z" fill="currentColor"/></svg>Interactive demo</a>
    <a class="ss-btn ghost" href="/files/projects/2026-11-stepshare/demo/index.html" target="_blank" rel="noopener"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M5 2H2v9h9V8M7.5 1.5h4v4M11.5 1.5 6 7" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>Demo full screen</a>
    <span class="ss-btn soon" aria-disabled="true"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M3 1h5l3 3v8H3z M8 1v3h3" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>Paper · coming soon</span>
    <span class="ss-btn soon" aria-disabled="true"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M4.5 3 1.5 6.5l3 3.5M8.5 3l3 3.5-3 3.5" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>Code · coming soon</span>
  </div>
  <div class="ss-split" role="img" aria-label="Generation's share of the AdamW step: 0.9 percent at equal weights, 37.2 percent in the deep half under StepShare-Adam">
    <div class="row"><b>Equal weights, one AdamW</b><div class="bar"><i style="width:99.1%;background:var(--c-und)">understanding</i><i style="width:0.9%;min-width:3px;background:var(--c-gen);padding:0"></i></div><span class="val">0.9%</span></div>
    <div class="row"><b>StepShare-Adam, deep half</b><div class="bar"><i style="width:62.8%;background:var(--c-und)">understanding</i><i style="width:37.2%;background:var(--c-gen)">generation</i></div><span class="val">37.2%</span></div>
    <div class="cap">Generation's share of the AdamW step's squared norm, as measured by StepShare (mean over training).</div>
  </div>
</section>

<div class="ss-stats">
  <div class="ss-stat" style="--accent:var(--c-und)"><div class="num">≈100×</div><div class="lab">larger understanding gradient, nearly orthogonal to generation's (|cos| ≤ 0.05)</div></div>
  <div class="ss-stat" style="--accent:var(--c-gen)"><div class="num">0.9%</div><div class="lab">of each AdamW step goes to generation at equal loss weights</div></div>
  <div class="ss-stat" style="--accent:var(--c-wgt)"><div class="num">7 pts</div><div class="lab">hidden drop in intermediate-layer probes after naive reweighting</div></div>
  <div class="ss-stat" style="--accent:var(--c-spl)"><div class="num">91%</div><div class="lab">of the FID gap closed by StepShare-Adam, worst probe layer −0.5, no added parameters</div></div>
</div>

<h2 id="key-results">Key results</h2>

<p class="ss-k">Finding</p>
<h3 class="ss-h" id="starved-not-conflicted">Starved, not conflicted</h3>
<figure class="ss-fig" style="--accent:var(--c-gen)">
  <div class="img"><img src="/images/projects/2026-11-stepshare/fig1-teaser.png" alt="Figure 1: StepShare at equal loss weights, restoring the share, and the hidden cost by layer." loading="lazy"></div>
  <figcaption><strong>Figure 1.</strong> StepShare: at equal loss weights generation is starved, and restoring its share closes most of the gap.
    <span class="take">At equal weights the two gradients are nearly orthogonal and generation gets 0.9% of the AdamW step. Lowering the understanding weight restores the share and closes most of the FID gap, more than task routing does at equal weights, but costs up to 5.4 probe points in intermediate layers that the last layer and caption NLL do not show.</span></figcaption>
</figure>

<p class="ss-k">Method</p>
<h3 class="ss-h" id="measure-then-rebalance">Measure the share, then rebalance it</h3>
<figure class="ss-fig" style="--accent:var(--c-und)">
  <div class="ss-method">
    <div class="img"><img src="/images/projects/2026-11-stepshare/fig2-stepshare.png" alt="Figure 2: StepShare measures the share of each AdamW step and rebalances it." loading="lazy"></div>
    <div class="ss-steps">
      <div class="ss-step"><div class="t"><span>1</span>Measure</div>With one AdamW, the first moment is a sum of one running average per task over a shared denominator, so every step splits exactly into Δ<sub>u</sub> + Δ<sub>g</sub>, logged by depth from the two backward passes joint training already runs.</div>
      <div class="ss-step"><div class="t"><span>2</span>Restore</div>Lower the understanding weight or give each task its own Adam state: generation's share rises to 14–38% of the step, and the FID gap closes.</div>
      <div class="ss-step"><div class="t"><span>3</span>Allocate by depth</div>StepShare-Adam scales generation's step by β = 0.3 in the shallow half, where the probe cost sits, and keeps it in the deep half, where most of the FID gain comes from.</div>
    </div>
  </div>
  <figcaption><strong>Figure 2.</strong> StepShare. Step 1 splits every AdamW step exactly into an understanding part and a generation part, by depth; Step 2 restores the share and then allocates it by depth.</figcaption>
</figure>

<p class="ss-k">Results</p>
<h3 class="ss-h" id="closing-the-gap">Closing the gap without the hidden cost</h3>
<figure class="ss-fig" style="--accent:var(--c-spl)">
  <div class="ss-tablewrap">
  <table class="ss-table">
    <thead><tr><th>Arm (0.6B, w = understanding weight)</th><th>Seeds</th><th>FID gap closed, guidance 1</th><th>Guidance 4</th><th>Worst COCO probe layer</th><th>Worst IN-100</th></tr></thead>
    <tbody>
      <tr><td><span class="dot" style="background:var(--c-wgt)"></span>Dense, w = 1</td><td>5</td><td><div class="cell"><span class="bar"><i style="width:0%"></i></span>0% <span class="ci">reference</span></div></td><td>0%</td><td><div class="cell"><span class="bar"><i style="width:8%;background:var(--cost)"></i></span>−0.5</div></td><td>−0.3</td></tr>
      <tr><td><span class="dot" style="background:var(--c-wgt)"></span>Dense, w = 0.03</td><td>5</td><td><div class="cell"><span class="bar"><i style="width:82%;background:var(--c-wgt)"></i></span>82% <span class="ci">± 12</span></div></td><td>78% <span class="ci">± 10</span></td><td><div class="cell"><span class="bar"><i style="width:90%;background:var(--cost)"></i></span>−5.4</div></td><td>−5.1</td></tr>
      <tr><td><span class="dot" style="background:var(--c-rte)"></span>Task routing, w = 0.03</td><td>5</td><td><div class="cell"><span class="bar"><i style="width:93%;background:var(--c-rte)"></i></span>93% <span class="ci">± 5</span></div></td><td>96% <span class="ci">± 5</span></td><td><div class="cell"><span class="bar"><i style="width:48%;background:var(--cost)"></i></span>−2.9</div></td><td>−2.8</td></tr>
      <tr><td><span class="dot" style="background:var(--c-opt)"></span>Per-task Adam, w = 1</td><td>3</td><td><div class="cell"><span class="bar"><i style="width:92%;background:var(--c-opt)"></i></span>92% <span class="ci">± 9</span></div></td><td>97% <span class="ci">± 5</span></td><td><div class="cell"><span class="bar"><i style="width:60%;background:var(--cost)"></i></span>−3.6</div></td><td>−3.2</td></tr>
      <tr class="hl"><td><span class="dot" style="background:var(--c-spl)"></span>StepShare-Adam, w = 1</td><td>5</td><td><div class="cell"><span class="bar"><i style="width:91%;background:var(--c-spl)"></i></span>91% <span class="ci">± 14</span></div></td><td>89% <span class="ci">± 8</span></td><td><div class="cell"><span class="bar"><i style="width:8%;background:var(--cost)"></i></span>−0.5</div></td><td>−0.4</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Table 2 (excerpt).</strong> Restoring generation's share: loss weight, routing and optimiser changes. Recovery is mean ± 95% CI over seeds, paired by seed; probe columns are points against understanding-only training of the same seed (grey bars: size of the deficit, full bar = 6 points).
    <span class="take">Reweighting and per-task Adam close most of the gap but cost 3 to 5 points in intermediate layers; StepShare-Adam keeps the gain, removes the cost, and adds no parameters.</span></figcaption>
</figure>

<p class="ss-k">Analysis</p>
<h3 class="ss-h" id="share-follows-ratio">The share follows the gradient ratio</h3>
<figure class="ss-fig" style="--accent:var(--c-wgt)">
  <div class="img"><img src="/images/projects/2026-11-stepshare/fig3-budget.png" alt="Figure 3: Gradient ratio during training and the trade-offs of restoring the share." loading="lazy"></div>
  <figcaption><strong>Figure 3.</strong> StepShare's gradient ratio and the trade-offs of restoring the share, 0.6B unless marked.
    <span class="take">At equal weights the gradient ratio rises from 11 to 60 as the generation loss falls, and FID recovery follows the training-average ratio across loss weights and clipping schemes. Along the weight sweep, more recovery comes with a deeper probe deficit; depth allocation and StepShare-Adam leave that trade-off.</span></figcaption>
</figure>

<p class="ss-k">Samples</p>
<h3 class="ss-h" id="same-noise-samples">Same captions, same noise</h3>
<figure class="ss-fig" style="--accent:var(--c-spl)">
  <div class="img"><img src="/images/projects/2026-11-stepshare/fig4-samples.webp" alt="Figure 4: Samples from the same captions and initial noise for four arms." loading="lazy"></div>
  <figcaption><strong>Figure 4.</strong> Samples from the same noise.
    <span class="take">At equal weights the dense model produces saturated high-frequency texture; restoring generation's share with StepShare-Adam gives layouts close to generation-only training.</span></figcaption>
</figure>

<h2 id="interactive-demo">Interactive demo</h2>

<div class="ss-demo">
  <p class="intro">Follow the guided tour or pick any arm: see how much of the FID gap it closes and in which layers its probe cost sits. <em>Step split</em> shows StepShare's measurements; <em>Samples</em> compares grids from the same noise.</p>
  <iframe src="/files/projects/2026-11-stepshare/demo/index.html" width="100%" height="900" loading="lazy" title="StepShare interactive explorer"></iframe>
</div>

<h2 id="links-and-citation">Links and citation</h2>

<div class="ss-links">
  <div class="ss-link"><div class="h">Paper</div><div class="s">Under review; the PDF and BibTeX will be linked here when they are public.</div></div>
  <div class="ss-link"><div class="h">Code</div><div class="s">Will be released with the paper.</div></div>
  <div class="ss-link"><div class="h">Demo</div><div class="s"><a href="/files/projects/2026-11-stepshare/demo/index.html" target="_blank" rel="noopener">Open the explorer full screen</a>; all numbers are the paper's logged results.</div></div>
</div>

</div>
