---
title: "How Much Does Random Rotation Help GPTQ? Sharp Worst-Case Rate Gaps in the High-Rate Regime"
collection: projects
permalink: /projects/2026-10-gptq-rotation/
date: 2026-10-27
status: "arXiv preprint, under review"
venue: "arXiv preprint, under review"
excerpt: "GPTQ after a random rotation, compared exactly with WaterSIC at high rate: the pivots have an explicit limit, the worst-case gap is ½ log₂ κ − log₂ ln κ + O(1) bits, so rotation cannot replace waterfilling in general, and on 96 real LLM Hessians the gap is 0.04 bit at the median."
teaser: "/images/projects/2026-10-gptq-rotation/teaser.png"
tags: ["LLM quantization", "GPTQ", "random rotations", "rate-distortion theory", "free probability"]
paperurl: "https://arxiv.org/abs/2611.01669"
pdfurl: "/files/gptq-random-rotation.pdf"
bibtexurl: "/files/bibtex/li2026gptqrotation.bib"
publication: "/publication/2026-10-gptq-rotation"
citation: 'Runguo Li. (2026). &quot;How Much Does Random Rotation Help GPTQ? Sharp Worst-Case Rate Gaps in the High-Rate Regime.&quot; arXiv:2611.01669. Manuscript under review.'
published: false
---

<style>
/* Page-scoped styles: everything is under .gr so no other page changes. */
.gr { --c-rot: #2a78d6; --c-lim: #eb6834; --c-none: #898781;
  --ink: #1d1d1b; --ink2: #52514e; --muted: #898781; --line: rgba(11, 11, 11, 0.09); --grid: #e6e5df; --card: #ffffff; --soft: #f6f6f3;
  --shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 10px 30px rgba(11, 11, 11, 0.06); color: var(--ink); }
html[data-theme="dark"] .gr { --c-rot: #3987e5; --c-lim: #d95926; --c-none: #a9a8a1;
  --ink: #f3f3f1; --ink2: #d0cfc8; --muted: #a9a8a1; --line: rgba(255, 255, 255, 0.10); --grid: #4a4a48; --card: #3b3b3b; --soft: #404040;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.25), 0 10px 30px rgba(0, 0, 0, 0.2); }
/* the layout's page title must not run under the fixed table of contents */
@media screen and (min-width: 1200px) { .page__title { max-width: calc(100% - 250px); } }
.gr p { margin: 0 0 0.9em; }
.gr-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 28px 28px 22px; margin: 0.4em 0 1.4em; border: 1px solid var(--line);
  background: radial-gradient(70% 120% at 0% 0%, rgba(42, 120, 214, 0.12), transparent 60%), radial-gradient(60% 120% at 100% 0%, rgba(235, 104, 52, 0.10), transparent 60%), var(--card);
  box-shadow: var(--shadow); }
.gr-hero::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: linear-gradient(90deg, var(--c-rot) 0%, var(--c-rot) 55%, var(--c-lim) 100%); }
.gr-kicker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.72em; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin-bottom: 12px; }
.gr-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; background: var(--ink); color: var(--card); font-weight: 700; letter-spacing: 0.04em; }
.gr .gr-tagline { font-size: 1.65em; line-height: 1.2; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 12px; max-width: 24em; }
.gr .gr-tagline em { font-style: normal; color: var(--c-rot); }
.gr .gr-lede { font-size: 0.95em; line-height: 1.6; color: var(--ink2); max-width: 48em; }
.gr .gr-lede strong { color: var(--ink); }
.gr .gr-byline { font-size: 0.85em; color: var(--ink2); margin: 0 0 4px; }
.gr .gr-byline strong { color: var(--ink); }
.gr-btns { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 20px; }
.gr .gr-btn { display: inline-flex; align-items: center; gap: 7px; padding: 7px 15px; border-radius: 999px; font-size: 0.85em; font-weight: 600; text-decoration: none; border: 1px solid var(--ink); }
.gr .gr-btn.primary { background: var(--ink); color: var(--card); }
.gr .gr-btn.ghost { background: transparent; color: var(--ink); }
.gr-ladder { display: grid; gap: 10px; padding: 14px 16px 12px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); }
.gr-ladder .row { display: grid; grid-template-columns: 17em 1fr; align-items: center; gap: 12px; font-size: 0.82em; }
.gr-ladder .row b { font-weight: 600; }
.gr-ladder .row small { display: block; font-size: 0.92em; color: var(--muted); font-weight: 400; }
.gr-ladder .track { display: flex; align-items: center; gap: 8px; }
.gr-ladder .track i { display: block; height: 18px; border-radius: 0 4px 4px 0; min-width: 3px; }
.gr-ladder .track span { font-weight: 700; white-space: nowrap; }
.gr-ladder .cap { font-size: 0.75em; color: var(--muted); }
.gr-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 1.6em; }
.gr-stat { position: relative; padding: 14px 14px 12px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }
.gr-stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.gr-stat .num { font-size: 1.5em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; }
.gr-stat .lab { font-size: 0.78em; color: var(--ink2); line-height: 1.4; margin-top: 5px; }
.gr .gr-k { font-size: 0.7em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 1.8em 0 0.1em; }
.gr h3.gr-h { font-size: 1.1em; margin: 0 0 0.55em; }
.gr .gr-lead { font-size: 0.95em; line-height: 1.6; color: var(--ink2); margin: 0 0 0.9em; max-width: 52em; }
.gr .gr-lead strong { color: var(--ink); }
.gr .gr-fig { display: block; margin: 0 0 1.2em; padding: 14px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); }
.gr-fig .img { background: #fff; border-radius: 10px; padding: 8px; }
.gr-fig img { display: block; width: 100%; height: auto; margin: 0 auto; }
.gr .gr-fig figcaption { font-family: inherit; font-size: 0.82em; color: var(--ink2); line-height: 1.5; margin: 10px 4px 0; }
.gr-fig figcaption strong { color: var(--ink); }
.gr-fig .take { display: block; margin-top: 6px; padding: 7px 10px; border-radius: 8px; background: var(--soft); color: var(--ink); border-left: 3px solid var(--accent, var(--ink2)); }
.gr .gr-eq { margin: 0.2em 0 1.1em; padding: 12px 14px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); font-size: 0.95em; color: var(--ink); overflow-x: auto; text-align: center; }
.gr .gr-eq small { display: block; color: var(--ink2); font-size: 0.84em; margin-top: 6px; text-align: center; }
.gr-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.gr-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.gr-card { position: relative; padding: 14px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.gr-card .h { font-weight: 700; color: var(--ink); margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.gr-card .h i { flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--accent, var(--ink2)); }
.gr-card .h span { flex: none; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.gr-card .num { font-size: 1.6em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; color: var(--ink); margin-bottom: 2px; }
.gr-card.ev { padding-left: 18px; overflow: hidden; }
.gr-card.ev::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.gr .gr-terms { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 28px; margin: 0.2em 0 1.4em; padding: 8px 18px; border-radius: 14px; background: var(--soft); border: 1px solid var(--line); font-size: 0.84em; line-height: 1.5; }
.gr .gr-terms > div { padding: 8px 0; border-bottom: 1px solid var(--line); }
.gr .gr-terms > div:nth-last-child(-n+2) { border-bottom: 0; }
.gr .gr-terms dt { display: inline; font-size: 1em; font-weight: 700; color: var(--ink); margin: 0; }
.gr .gr-terms dd { display: inline; font-size: 1em; margin: 0; color: var(--ink2); }
.gr blockquote.gr-quote { margin: 0 0 1.1em; padding: 10px 16px; border-left: 3px solid var(--c-lim); background: var(--soft); border-radius: 0 10px 10px 0; font-size: 0.92em; font-style: normal; color: var(--ink); }
.gr blockquote.gr-quote cite { display: block; font-size: 0.86em; font-style: normal; color: var(--ink2); margin-top: 4px; }
.gr .gr-list { font-size: 0.92em; line-height: 1.6; color: var(--ink2); margin: 0 0 1.1em 1.2em; max-width: 52em; }
.gr .gr-list li { margin: 0.25em 0; }
.gr .gr-list strong { color: var(--ink); }
/* controls */
.gr-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 0 2px 10px; font-size: 0.82em; color: var(--ink2); }
.gr-ctl label { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0; }
.gr-seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.gr .gr-seg button { border: 0; border-radius: 0; background: transparent; color: var(--ink2); font: inherit; font-size: 1em; padding: 4px 10px; cursor: pointer; margin: 0; white-space: nowrap; }
.gr .gr-seg button + button { border-left: 1px solid var(--line); }
.gr .gr-seg button[aria-pressed="true"] { background: var(--soft); color: var(--ink); font-weight: 600; }
.gr .gr-act { border: 1px solid var(--line); border-radius: 8px; background: var(--card); color: var(--ink); font: inherit; font-size: 1em; padding: 4px 11px; cursor: pointer; margin: 0; }
.gr .gr-act:hover, .gr .gr-seg button:hover { background: var(--soft); }
.gr-sliders { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px 22px; margin: 0 2px 12px; font-size: 0.82em; color: var(--ink2); }
.gr-sliders > div { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 4px 10px; }
.gr-sliders > div[hidden] { display: none; }
.gr-sliders .nm { font-weight: 600; color: var(--ink); white-space: nowrap; }
.gr-sliders output { grid-column: 1 / -1; font-size: 0.94em; color: var(--ink2); font-variant-numeric: tabular-nums; }
.gr .gr-sliders input[type="range"] { width: 100%; min-width: 0; margin: 0; padding: 0; accent-color: var(--c-rot); }
.gr-ro { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 12px 0 0; }
.gr-ro > div { padding: 9px 12px; border-radius: 10px; background: var(--soft); border: 1px solid var(--line); }
.gr-ro .v { font-size: 1.3em; font-weight: 800; letter-spacing: -0.01em; line-height: 1.15; }
.gr-ro .v small { font-size: 0.55em; font-weight: 600; color: var(--ink2); letter-spacing: 0; }
.gr-ro .l { font-size: 0.72em; color: var(--ink2); line-height: 1.35; margin-top: 3px; }
/* charts */
.gr-chart svg { display: block; overflow: visible; touch-action: pan-y; }
.gr-chart text { font-family: inherit; }
.gr-chart .tk { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.gr-chart .al { fill: var(--ink2); font-size: 11.5px; }
.gr-chart .lb { fill: var(--ink2); font-size: 11px; }
.gr-chart .mu { fill: var(--muted); font-size: 11px; }
.gr-chart .gl { stroke: var(--grid); stroke-width: 1; }
.gr-chart .ax { stroke: var(--muted); stroke-opacity: 0.7; stroke-width: 1; }
.gr-chart .ref { stroke: var(--muted); stroke-width: 1; }
.gr-chart .ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.gr-chart .ln.none { stroke: var(--c-none); }
.gr-chart .ln.rot { stroke: var(--c-rot); }
.gr-chart .ln.lim { stroke: var(--c-lim); }
.gr-chart .ln.dash { stroke-dasharray: 5 5; stroke-linecap: butt; }
.gr-chart .lv { stroke-width: 1; fill: none; }
.gr-chart .lv.none { stroke: var(--c-none); }
.gr-chart .lv.rot { stroke: var(--c-rot); }
.gr-chart .lv.gm { stroke: var(--ink); }
.gr-chart .pt.rot { fill: var(--c-rot); }
.gr-chart .pt.o { fill: var(--c-lim); }
.gr-chart .pt.open { fill: var(--card); stroke: var(--c-none); stroke-width: 1.5; }
.gr-chart .pt.ringed { stroke: var(--card); stroke-width: 1.5; }
.gr-chart .cross { stroke: var(--muted); stroke-width: 1; }
.gr-chart .ring { fill: none; stroke: var(--ink); stroke-width: 1.5; }
.gr-legend { display: flex; flex-wrap: wrap; gap: 4px 16px; font-size: 0.76em; color: var(--ink2); margin: 6px 2px 0; }
.gr-legend .k { display: inline-flex; align-items: center; gap: 6px; }
.gr-legend svg { flex: none; overflow: visible; }
.gr-tip { position: fixed; z-index: 50; pointer-events: none; background: var(--card); color: var(--ink); border: 1px solid var(--line); box-shadow: var(--shadow); border-radius: 8px; padding: 7px 10px; font-size: 12px; line-height: 1.5; min-width: 150px; max-width: 320px; }
.gr-tip[hidden] { display: none; }
.gr-tip .tt { color: var(--ink2); margin-bottom: 3px; }
.gr-tip .tr { display: flex; align-items: center; gap: 7px; white-space: nowrap; }
.gr-tip .tr span { color: var(--ink2); flex: 1; }
.gr-tip .tr b { font-variant-numeric: tabular-nums; margin-left: 10px; }
.gr-tip .tr i { flex: none; display: inline-block; width: 12px; height: 0; border-top: 2px solid var(--c-none); }
.gr-tip .tr i.k-rotl { border-top-color: var(--c-rot); }
.gr-tip .tr i.k-lim { border-top-color: var(--c-lim); }
.gr-tip .tr i.dash { border-top-style: dashed; }
.gr-tip .tr i.k-rot, .gr-tip .tr i.k-o, .gr-tip .tr i.k-open { width: 9px; height: 9px; border: 0; border-radius: 50%; margin: 0 1.5px; background: var(--c-rot); }
.gr-tip .tr i.k-o { background: var(--c-lim); }
.gr-tip .tr i.k-open { background: transparent; border: 1.5px solid var(--c-none); box-sizing: border-box; }
/* tables */
.gr-tablewrap { overflow-x: auto; }
.gr .gr-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.82em; margin: 0; display: table; border: 0; background: transparent; }
.gr .gr-table thead, .gr .gr-table thead tr, .gr .gr-table tbody tr { background: transparent; }
.gr .gr-table th { text-align: left; font-weight: 600; color: var(--ink2); padding: 8px 10px; border: 0; border-bottom: 1px solid var(--line); white-space: nowrap; background: transparent; }
.gr .gr-table td { padding: 8px 10px; border: 0; border-bottom: 1px solid var(--line); vertical-align: middle; font-variant-numeric: tabular-nums; white-space: nowrap; background: transparent; }
.gr .gr-table .num, .gr .gr-table th.num { text-align: right; }
.gr .gr-table td.hl { font-weight: 700; background: rgba(42, 120, 214, 0.08); }
.gr .gr-table .sd { color: var(--muted); font-weight: 400; }
.gr details.gr-more { margin: 10px 4px 0; font-size: 0.86em; color: var(--ink2); }
.gr details.gr-more summary { cursor: pointer; font-weight: 600; color: var(--ink); }
.gr details.gr-more .cap { margin: 6px 0 2px; font-size: 0.92em; }
.gr .gr-scope { font-size: 0.8em; color: var(--muted); margin: 0 0 1.2em; }
.gr-bib { position: relative; margin: 0.6em 0 1em; }
.gr .gr-bib pre { margin: 0; border-radius: 12px; font-size: 0.74em; max-width: 100%; overflow-x: auto; padding: 14px 16px; background: var(--soft); border: 1px solid var(--line); color: var(--ink); }
.gr .gr-bib pre code { background: transparent; color: inherit; padding: 0; border: 0; font-size: 1em; }
.gr .gr-copy { position: absolute; top: 8px; right: 8px; font: inherit; font-size: 0.72em; padding: 3px 11px; border-radius: 999px; border: 1px solid var(--line); background: var(--card); color: var(--ink); cursor: pointer; margin: 0; }
.gr-links { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 1em; }
@media (max-width: 720px) {
  .gr-grid3, .gr-grid2 { grid-template-columns: minmax(0, 1fr); }
  .gr .gr-terms { grid-template-columns: minmax(0, 1fr); }
  .gr .gr-terms > div:nth-last-child(2) { border-bottom: 1px solid var(--line); }
  .gr-hero { padding: 22px 18px 18px; }
  .gr .gr-tagline { font-size: 1.35em; }
  .gr-stats, .gr-ro { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .gr-ladder .row { grid-template-columns: minmax(0, 1fr); gap: 4px; }
  .gr-sliders { grid-template-columns: minmax(0, 1fr); }
  .gr-sliders > div { grid-template-columns: 1fr auto; }
  .gr-sliders .nm { grid-column: 1 / -1; }
}
</style>

<div class="gr">

<section class="gr-hero" aria-label="Overview">
  <div class="gr-kicker"><span class="gr-badge">arXiv preprint</span><span>Under review · cs.IT / cs.LG · November 2026</span></div>
  <p class="gr-tagline">In the worst case a random rotation cannot replace waterfilling. On real LLM layers <em>it nearly does</em>.</p>
  <p class="gr-lede">GPTQ quantizes a layer with one step size for all columns; WaterSIC gives every column its own and is within 0.255 bit of the rate-distortion limit. A random rotation before GPTQ closes most of the difference in practice, and Ordentlich and Polyanskiy asked how large it can be in the worst case. This paper gives the exact limit of rotated GPTQ, the <strong>sharp worst-case gap</strong>, and the reason real layers sit far below it.</p>
  <p class="gr-byline"><strong>Runguo Li</strong> · University of Illinois Urbana-Champaign</p>
  <div class="gr-btns">
    <a class="gr-btn primary" href="https://arxiv.org/abs/2611.01669"><i class="ai ai-arxiv" aria-hidden="true"></i>arXiv 2611.01669</a>
    <a class="gr-btn ghost" href="/files/gptq-random-rotation.pdf"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i>Paper (PDF)</a>
    <a class="gr-btn ghost" href="#rotate-a-spectrum"><i class="fa-solid fa-rotate" aria-hidden="true"></i>Rotate a spectrum</a>
    <a class="gr-btn ghost" href="#links-and-citation"><i class="fa-solid fa-quote-right" aria-hidden="true"></i>BibTeX</a>
  </div>
  <div class="gr-ladder" role="img" aria-label="Extra bits per weight that GPTQ needs over WaterSIC: 4.32 in the worst case without rotation at condition number ten thousand, 2.56 in the worst case with a random rotation, and 0.04 at the median of 96 real layers after rotation">
    <div class="row"><b>Worst case, no rotation<small>over spectra with κ = 10⁴</small></b><div class="track"><i style="width:86%;background:var(--c-none)"></i><span>4.32 bit</span></div></div>
    <div class="row"><b>Worst case, random rotation<small>over spectra with κ = 10⁴</small></b><div class="track"><i style="width:51%;background:var(--c-rot)"></i><span>2.56 bit</span></div></div>
    <div class="row"><b>Real layers, random rotation<small>median of 96 LLM Hessians</small></b><div class="track"><i style="width:0.8%;background:var(--c-rot)"></i><span>0.04 bit</span></div></div>
    <div class="cap">Extra bits per weight that GPTQ needs over WaterSIC at equal distortion (high rate, entropy coded). The real layers have κ between 4×10³ and 10⁶ at the default damping.</div>
  </div>
</section>

<div class="gr-stats">
  <div class="gr-stat" style="--accent:var(--c-lim)"><div class="num">½ log₂ ln κ + O(1)</div><div class="lab">bits are all a random rotation saves in the worst case; the leading term ½ log₂ κ stays</div></div>
  <div class="gr-stat" style="--accent:var(--c-rot)"><div class="num">ημ → η²μ/2</div><div class="lab">cost of an η-fraction of outlier eigenvalues of size μ, before and after rotation, to leading order</div></div>
  <div class="gr-stat" style="--accent:var(--c-rot)"><div class="num">0.04 bit</div><div class="lab">median rotated gap on 96 Hessians of Qwen3 and Llama-3.1 (maximum 0.36), against 2–5 bits in the worst case at the same κ</div></div>
  <div class="gr-stat" style="--accent:var(--c-none)"><div class="num">≤ 0.009 bit</div><div class="lab">between the limit formula and the measured gap after a Haar rotation, over 270 layer and damping pairs</div></div>
</div>

<h2 id="the-question">The question</h2>

<p class="gr-lead">Layerwise post-training quantization replaces a weight matrix \(W\) by \(\hat W\) so that the layer's output changes as little as possible on calibration inputs \(x\), whose second moment is \(\Sigma=\mathbb E[xx^\top]\). Two families of methods improve on plain GPTQ, and at high rate they can be compared exactly.</p>

<div class="gr-grid3">
  <div class="gr-card" style="--accent:var(--c-none)"><div class="h"><i></i>GPTQ</div>Quantizes one input column at a time and feeds the rounding error forward. It is exactly Babai's nearest-plane algorithm on the lattice of the Cholesky factor of \(\Sigma\) (<a href="https://arxiv.org/abs/2507.18553">Chen et al.</a>; <a href="https://arxiv.org/abs/2508.01077">Birnick</a>), so its quality is set by the Cholesky pivots \(\pi_1,\dots,\pi_n\).</div>
  <div class="gr-card" style="--accent:var(--c-rot)"><div class="h"><i></i>Rotation</div>Apply an orthogonal transform to weights and activations, then run GPTQ. The layer is unchanged and \(\Sigma\) becomes \(V^\top\Sigma V\). QuIP, QuIP# and QuaRot use random rotations; SpinQuant learns one. One step size for all columns is kept.</div>
  <div class="gr-card" style="--accent:var(--c-lim)"><div class="h"><i></i>Waterfilling</div><a href="https://arxiv.org/abs/2603.04956">WaterSIC</a> keeps the basis and gives every column its own step size, \(\alpha_k\propto\pi_k^{-1/2}\). It is within \(\tfrac12\log_2\frac{2\pi e}{12}\approx0.255\) bit of the rate-distortion limit for every \(\Sigma\).</div>
</div>

<p class="gr-lead">With per-column entropy coding and at high rate, GPTQ needs this many more bits per weight than WaterSIC at equal distortion (Lifar et al., Theorem 3.3):</p>

<div class="gr-eq">\[ \Delta(\Sigma)=\frac12\log_2\frac{\frac1n\sum_k\pi_k}{\big(\prod_k\pi_k\big)^{1/n}} ,\qquad \pi_k=\frac{\det\Sigma^{(k)}}{\det\Sigma^{(k-1)}} . \]
<small>Half the log of the arithmetic mean of the pivots over their geometric mean. The geometric mean is \((\det\Sigma)^{1/n}\) in every basis; only the arithmetic mean depends on the basis.</small></div>

<p class="gr-lead">The gap is unbounded over \(\Sigma\), yet on Llama-3-8B a random rotation brings it to about 0.1 bit. Whether rotation can stand in for waterfilling in general was left open:</p>

<blockquote class="gr-quote">“It is an interesting open problem to estimate worst possible gap (over possible spectra λ<sub>j</sub> ≥ ε) between GPTQ with rotation and WaterSIC.”<cite>Ordentlich and Polyanskiy, <a href="https://arxiv.org/abs/2605.13768">High-Rate Quantized Matrix Multiplication II</a>, §III-E</cite></blockquote>

<p class="gr-lead">The paper answers it in the setting in which it was posed: high rate, per-column entropy coding, and Haar (uniformly random) rotations.</p>

<p class="gr-k">Terms used below</p>
<dl class="gr-terms">
  <div><dt>Pivot \(\pi_k\).</dt> <dd>The squared \(k\)-th diagonal entry of the Cholesky factor of \(\Sigma\): what is left of coordinate \(k\) after linear prediction from the coordinates before it.</dd></div>
  <div><dt>Rate gap \(\Delta\).</dt> <dd>Extra bits per weight that GPTQ needs over WaterSIC at the same distortion. Zero means one step size is as good as per-column steps.</dd></div>
  <div><dt>Spectrum \(\nu\).</dt> <dd>The distribution of the eigenvalues of \(\Sigma\). AM, GM and HM are its arithmetic, geometric and harmonic means.</dd></div>
  <div><dt>Condition number \(\kappa\).</dt> <dd>Largest eigenvalue over smallest. The worst case is taken over all spectra with a given \(\kappa\).</dd></div>
  <div><dt>Haar rotation.</dt> <dd>An orthogonal matrix drawn uniformly at random. Randomized Hadamard transforms are the fast substitute used in practice.</dd></div>
  <div><dt>Worst basis.</dt> <dd>The eigenbasis (PCA basis) of \(\Sigma\). There the pivots are the eigenvalues and \(\Delta=\tfrac12\log_2(\mathrm{AM}/\mathrm{GM})\).</dd></div>
  <div><dt>High rate.</dt> <dd>The limit of small quantization steps with entropy-coded integers. Fixed-rate formats such as INT4 are outside it.</dd></div>
  <div><dt>Damping.</dt> <dd>GPTQ adds a multiple of the mean diagonal of \(\Sigma\) to its diagonal; 10⁻² is the default and sets \(\kappa\) on real layers.</dd></div>
</dl>

<h2 id="the-limit-of-rotated-gptq">The limit of rotated GPTQ</h2>

<p class="gr-k">Theorem 1</p>
<h3 class="gr-h" id="pivot-profile">After a Haar rotation, the pivots follow one curve</h3>
<p class="gr-lead">For a spectrum \(\nu\), let \(s_t\) solve \(\int\frac{x}{s+x}\,d\nu(x)=t\). The <strong>pivot profile</strong> and its mean are</p>

<div class="gr-eq">\[ h_\nu(t)=\frac{t\,s_t}{1-t}=\frac{1}{S_\nu(-t)},\qquad D(\nu)=\int_0^1h_\nu(t)\,dt,\qquad \Delta(\nu)=\frac12\log_2\frac{D(\nu)}{\mathrm{GM}(\nu)} . \]
<small>\(S_\nu\) is Voiculescu's S-transform. \(h_\nu\) decreases from AM at \(t=0\) to HM at \(t=1\).</small></div>

<p class="gr-lead">Theorem 1 says that after a Haar rotation the \(k\)-th pivot converges almost surely to \(h_\nu(k/n)\), uniformly in \(k\ge\delta n\) and uniformly over all spectra with bounded condition number, with probability of failure at most \(Cne^{-cn}\). Hence the gap converges to \(\Delta(\nu)\), which takes one eigendecomposition and a scalar root-finding to compute.</p>

<ul class="gr-list">
  <li><strong>Strictly between the two extremes.</strong> \(\mathrm{GM}(\nu)\le D(\nu)\le\mathrm{AM}(\nu)\), with equality only when all eigenvalues are equal. A Haar rotation is strictly better than the worst basis and strictly worse than WaterSIC.</li>
  <li><strong>What is new and what is not.</strong> The formula for \(h_\nu\) is classical in the theory of Lyapunov exponents (Newman 1986; Kargin 2008) and appears as the per-stream SINR of successive cancellation with Haar spreading (Tulino and Verdú 2004). The paper adds its identification with GPTQ, a finite-\(n\), per-pivot version that is uniform in the spectrum, the functional \(D\) that enters the rate, and an elementary proof that avoids free probability.</li>
</ul>

<p class="gr-k">Interactive</p>
<h3 class="gr-h" id="rotate-a-spectrum">Rotate a spectrum in your browser</h3>
<p class="gr-lead">Pick a spectrum. The figure draws one Haar rotation, computes the \(n\) pivots of the rotated matrix, and plots them against the limit \(h_\nu(k/n)\). The gap is half the distance, in powers of two, between the mean pivot and the geometric mean, so the three horizontal lines show what rotation saves and what it leaves.</p>

<figure class="gr-fig" id="gr-piv" style="--accent:var(--c-rot)">
  <div class="gr-ctl">
    <label>Spectrum <span class="gr-seg" id="gr-fam" role="group" aria-label="Spectrum"><button type="button" data-fam="two" aria-pressed="true">Two-point</button><button type="button" data-fam="logu" aria-pressed="false">Log-uniform</button><button type="button" data-fam="spk" aria-pressed="false">Bulk + spikes</button></span></label>
    <label>n <span class="gr-seg" id="gr-n" role="group" aria-label="Dimension"><button type="button" data-n="64" aria-pressed="false">64</button><button type="button" data-n="128" aria-pressed="false">128</button><button type="button" data-n="256" aria-pressed="true">256</button></span></label>
    <button type="button" class="gr-act" id="gr-redraw">↻ New rotation</button>
  </div>
  <div class="gr-sliders">
    <div data-fam-only="two logu"><span class="nm">Condition number κ</span><input type="range" id="gr-kappa" min="1" max="8" step="0.25" value="4" aria-label="Condition number, as a power of ten"><span></span><output id="gr-kappa-out" for="gr-kappa"></output></div>
    <div data-fam-only="two"><span class="nm">Share q of large eigenvalues</span><input type="range" id="gr-q" min="0.02" max="0.9" step="0.005" value="0.225" aria-label="Share of eigenvalues equal to one"><button type="button" class="gr-act" id="gr-qstar">Worst q</button><output id="gr-q-out" for="gr-q"></output></div>
    <div data-fam-only="spk" hidden><span class="nm">Number of spikes r</span><input type="range" id="gr-r" min="0" max="16" step="1" value="4" aria-label="Number of spikes"><span></span><output id="gr-r-out" for="gr-r"></output></div>
    <div data-fam-only="spk" hidden><span class="nm">Spike size μ</span><input type="range" id="gr-mu" min="0" max="5" step="0.25" value="2" aria-label="Spike size, as a power of ten"><span></span><output id="gr-mu-out" for="gr-mu"></output></div>
  </div>
  <div class="gr-chart" id="gr-piv-chart"></div>
  <div class="gr-legend">
    <span class="k"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3.5" fill="var(--c-rot)"/></svg><span>pivots after one Haar rotation</span></span>
    <span class="k"><svg width="20" height="10" aria-hidden="true"><line x1="1" y1="5" x2="19" y2="5" stroke="var(--c-lim)" stroke-width="2.5"/></svg><span>limit h<sub>ν</sub>(k/n) of Theorem 1</span></span>
    <span class="k"><svg width="20" height="10" aria-hidden="true"><line x1="1" y1="5" x2="19" y2="5" stroke="var(--c-none)" stroke-width="2.5"/></svg><span>pivots without rotation (the eigenvalues, largest first)</span></span>
    <span class="k"><svg width="20" height="10" aria-hidden="true"><line x1="1" y1="5" x2="19" y2="5" stroke="var(--ink)" stroke-width="1"/></svg><span>thin lines: mean pivot without rotation, mean pivot after rotation, geometric mean</span></span>
  </div>
  <div class="gr-ro" aria-live="polite">
    <div><div class="v"><span id="gr-ro-none">–</span> <small>bit</small></div><div class="l">gap without rotation (eigenbasis, the worst basis)</div></div>
    <div><div class="v"><span id="gr-ro-sim">–</span> <small>bit</small></div><div class="l" id="gr-ro-sim-lab">after one Haar rotation (this draw)</div></div>
    <div><div class="v"><span id="gr-ro-lim">–</span> <small>bit</small></div><div class="l">limit Δ(ν) of Theorem 1 for this spectrum</div></div>
    <div><div class="v"><span id="gr-ro-worst">–</span> <small>bit</small></div><div class="l" id="gr-ro-worst-lab">worst case over all spectra with this κ (Theorem 2)</div></div>
  </div>
  <figcaption><span id="gr-piv-note">The figure needs JavaScript. Without it, see Figure 1 of the paper.</span>
    <span class="take">The numbers in this figure are computed in your browser, not taken from the paper's experiments. At n = 256 one draw differs from the limit by a few hundredths of a bit; the paper reports errors of at most 0.017 bit at n = 256 and 0.0004 bit at n = 8192, averaged over seeds, on five synthetic spectra.</span></figcaption>
</figure>

<h2 id="the-worst-case">The worst case</h2>

<p class="gr-k">Theorem 2</p>
<h3 class="gr-h" id="sharp-worst-case">Rotation cannot replace waterfilling in general</h3>
<p class="gr-lead">Take the supremum of the limit \(\Delta(\nu)\) over all spectra with condition number at most \(\kappa\). It is attained, and <strong>every maximizer is a two-point spectrum</strong>: a share \(q^*\approx2/\ln\kappa\) of the eigenvalues at the top and the rest at the bottom. As \(\kappa\to\infty\),</p>

<div class="gr-eq">\[ \Delta^*(\kappa)=\tfrac12\log_2\kappa-\log_2\ln\kappa+\tfrac12-\tfrac{1}{\ln2}+o(1), \qquad \Delta^*_{\text{no rotation}}(\kappa)=\tfrac12\log_2\kappa-\tfrac12\log_2\ln\kappa-\tfrac{1}{2\ln2}+o(1). \]
<small>In the worst case a random rotation saves \(\tfrac12\log_2\ln\kappa-\tfrac12+\tfrac{1}{2\ln2}+o(1)\) bits. The leading term \(\tfrac12\log_2\kappa\) is unchanged.</small></div>

<p class="gr-lead">In the extremal spectrum a small share of the directions carries almost all the energy. After rotation these directions fill the first \(\approx qn\) coordinates, whose pivots stay far above the rest. One step size cannot allocate rate to them; per-column steps can. The proof rests on a log-convexity property of the first variation of \(D\), which pushes any maximizer to the two endpoints of the spectrum. The same conclusion holds when the constraint is \(\lambda_j\ge\epsilon\cdot\mathrm{AM}\) in place of a condition number (Theorem 3).</p>

<figure class="gr-fig" style="--accent:var(--c-lim)">
  <div class="gr-tablewrap">
  <table class="gr-table">
    <thead><tr><th>κ</th><th class="num">Worst case, no rotation</th><th class="num">Worst case, rotation</th><th class="num">Asymptotic formula</th><th class="num">Worst share q*</th><th class="num">Rotation saves</th></tr></thead>
    <tbody>
      <tr><td>10²</td><td class="num">1.53</td><td class="num hl">0.46</td><td class="num">0.18</td><td class="num">0.39</td><td class="num">1.06</td></tr>
      <tr><td>10⁴</td><td class="num">4.32</td><td class="num hl">2.56</td><td class="num">2.50</td><td class="num">0.225</td><td class="num">1.76</td></tr>
      <tr><td>10⁶</td><td class="num">7.35</td><td class="num hl">5.27</td><td class="num">5.24</td><td class="num">0.149</td><td class="num">2.08</td></tr>
      <tr><td>10¹²</td><td class="num">16.82</td><td class="num hl">14.22</td><td class="num">14.20</td><td class="num">0.073</td><td class="num">2.60</td></tr>
      <tr><td>10⁴⁸</td><td class="num">75.61</td><td class="num hl">72.00</td><td class="num">72.00</td><td class="num">0.018</td><td class="num">3.61</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Table 1.</strong> Worst-case gaps in bits, exact, by one-dimensional maximization over q.
    <span class="take">From κ = 10² to 10⁴⁸ the worst case grows from 0.46 to 72 bits, while what rotation saves grows only from 1.06 to 3.61 bits.</span></figcaption>
</figure>

<p class="gr-k">Figure 1 of the paper, interactive</p>
<h3 class="gr-h" id="worst-case-vs-real-layers">Worst case against real layers</h3>
<p class="gr-lead">The two solid curves are the worst cases with and without rotation; on this logarithmic axis they nearly coincide for large κ. The dashed curves are one fixed spectrum, log-uniform, which also keeps the leading term. The dots are 96 real Hessians, each shown twice: in its worst basis and after a Haar rotation. Hover a dot for the layer, or anywhere else for the curves at that κ.</p>

<figure class="gr-fig" style="--accent:var(--c-rot)">
  <div class="gr-chart" id="gr-worst-chart"></div>
  <div class="gr-legend">
    <span class="k"><svg width="20" height="10" aria-hidden="true"><line x1="1" y1="5" x2="19" y2="5" stroke="var(--c-none)" stroke-width="2.5"/></svg><span>worst case, no rotation</span></span>
    <span class="k"><svg width="20" height="10" aria-hidden="true"><line x1="1" y1="5" x2="19" y2="5" stroke="var(--c-rot)" stroke-width="2.5"/></svg><span>worst case, random rotation (Theorem 2)</span></span>
    <span class="k"><svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="var(--c-none)" stroke-width="2.5" stroke-dasharray="5 4"/></svg><span>log-uniform spectrum, no rotation</span></span>
    <span class="k"><svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="var(--c-rot)" stroke-width="2.5" stroke-dasharray="5 4"/></svg><span>log-uniform spectrum, rotation</span></span>
    <span class="k"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3.3" fill="none" stroke="var(--c-none)" stroke-width="1.5"/></svg><span>real layers, worst (PCA) basis</span></span>
    <span class="k"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3.5" fill="var(--c-rot)"/></svg><span>real layers, Haar rotation</span></span>
  </div>
  <figcaption><strong>Figure 1.</strong> High-rate gap of GPTQ to WaterSIC against the condition number. Dots: 96 Hessians of Qwen3-1.7B, Qwen3-8B-Base and Llama-3.1-8B-Instruct at damping 10⁻², in the worst basis and after a Haar rotation.
    <span class="take">For the condition numbers of real layers the worst case after rotation is 2–5 bits. The layers themselves sit at a median of 0.04 bit, about 1% of it.</span></figcaption>
</figure>

<h2 id="why-real-layers-are-benign">Why real layers are benign</h2>

<p class="gr-lead">Real Hessian spectra are a narrow bulk plus a few very large eigenvalues: the largest is 10³ to 10⁵ times the median. Rotation treats the two parts very differently, and a third kind of spectrum shows what it cannot do.</p>

<div class="gr-grid3">
  <div class="gr-card ev" style="--accent:var(--c-rot)"><div class="num">η²μ / 2</div><div class="h">Outliers become second order</div>An η-fraction of outliers of size μ raises the mean pivot by about ημ in the worst basis. After rotation the first ≈ ηn coordinates absorb the outlier subspace one after another, and the cost is η²μ/2 to leading order (Proposition 2). With the bulk on [½, 1] and μ = 5×10⁵, the rotated gap is 0.003 and 0.21 bit for η = 10⁻⁴ and 10⁻³, against 3.05 and 4.70 bit without rotation.</div>
  <div class="gr-card ev" style="--accent:var(--c-rot)"><div class="num">o(n²)</div><div class="h">A few spikes of any size</div>If Σ has r spikes μ<sub>j</sub> with (r + ln n)·Σμ<sub>j</sub> = o(n²), the rotated gap is that of the bulk alone, minus a small term from the geometric mean (Proposition 3). The condition is sharp up to the factor r + ln n. On the 96 real layers this quantity, divided by n², is at most 0.03.</div>
  <div class="gr-card ev" style="--accent:var(--c-lim)"><div class="num">¼ log₂ κ</div><div class="h">Spread on a log scale stays</div>For eigenvalues spread on a logarithmic scale, the leading term of the gap is the same with and without rotation, and rotation again saves only ½ log₂ ln κ + O(1) (Proposition 4). For the log-uniform spectrum the leading term is ¼ log₂ κ and the profile has a closed form.</div>
</div>

<p class="gr-lead">So the rotated gap is small when the bulk is narrow on a logarithmic scale and the outliers satisfy \(\eta^2\mu\ll D(\nu)\), and it grows like \(\tfrac12\log_2\kappa\) when the spectrum is spread on a log scale or when a share \(\gtrsim1/\ln\kappa\) of the directions carries the energy. Real layers are of the first kind.</p>

<h2 id="experiments">On 96 real Hessians</h2>

<p class="gr-lead">From 8 evenly spaced blocks of Qwen3-1.7B, Qwen3-8B-Base and Llama-3.1-8B-Instruct we take the Hessians of the four distinct linear-layer inputs: 96 matrices with n from 2048 to 14336, computed in float64 from 128 WikiText-2 sequences of 2048 tokens. The gap is computed exactly from the pivots of each rotated matrix and compared with the limit computed from its spectrum alone.</p>

<p class="gr-k">Figure 2 of the paper, interactive</p>
<h3 class="gr-h" id="limit-vs-measured">The limit matches real layers</h3>
<p class="gr-lead">Each dot is one layer. Switch between a Haar rotation and the randomized Hadamard transforms used in practice, and switch the vertical axis to the deviation from the limit to see where they differ.</p>

<figure class="gr-fig" style="--accent:var(--c-rot)">
  <div class="gr-ctl">
    <label>Rotation <span class="gr-seg" id="gr-kind" role="group" aria-label="Rotation"><button type="button" data-kind="4" aria-pressed="true">Haar</button><button type="button" data-kind="5" aria-pressed="false">Hadamard, one round</button><button type="button" data-kind="6" aria-pressed="false">Hadamard, two rounds</button></span></label>
    <label>Show <span class="gr-seg" id="gr-view" role="group" aria-label="Vertical axis"><button type="button" data-view="gap" aria-pressed="true">Gap</button><button type="button" data-view="dev" aria-pressed="false">Deviation from the limit</button></span></label>
  </div>
  <div class="gr-chart" id="gr-real-chart"></div>
  <div class="gr-legend">
    <span class="k"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="4" fill="var(--c-lim)"/></svg><span>o_proj input (24 layers)</span></span>
    <span class="k"><svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="4" fill="var(--c-rot)"/></svg><span>q/k/v, gate/up and down_proj inputs (72 layers)</span></span>
    <span class="k"><svg width="20" height="10" aria-hidden="true"><line x1="1" y1="5" x2="19" y2="5" stroke="var(--muted)" stroke-width="1"/></svg><span>measured = limit</span></span>
  </div>
  <figcaption><span id="gr-real-readout"><strong>Figure 2.</strong> Finite-n gap of rotated GPTQ against the limit of Theorem 1.</span>
    <span class="take">96 real Hessians at damping 10⁻², mean over 3 seeds. With a Haar rotation every deviation is positive and at most 0.006 bit at this damping (0.009 bit over all three dampings). Randomized Hadamard deviates mainly on o_proj inputs, whose covariance is blocked by attention head.</span></figcaption>
  <details class="gr-more"><summary>Table view by input type</summary><p class="cap" id="gr-real-table-cap"></p><div class="gr-tablewrap"><table class="gr-table" id="gr-real-table"></table></div></details>
</figure>

<div class="gr-grid2">
  <div class="gr-card ev" style="--accent:var(--c-rot)"><div class="num">0.041 bit</div><div class="h">Typical against worst case</div>After a Haar rotation the median gap is 0.041 bit, 85% of the layers are below 0.1 bit, and the maximum is 0.356 bit (Llama block 0, q/k/v input). Rotation saves a median of 0.15 bit over unrotated GPTQ and 0.85 bit over the worst basis.</div>
  <div class="gr-card ev" style="--accent:var(--c-lim)"><div class="num">93% and 99%</div><div class="h">Randomized Hadamard</div>One round is within 0.01 bit of the limit on 93% of the layers and two rounds on 99%. On o_proj inputs one round deviates by −0.013 to +0.028 bit, and two rounds fall up to 0.026 bit below the limit; a random permutation of the input coordinates removes that advantage.</div>
  <div class="gr-card ev" style="--accent:var(--c-none)"><div class="num">27%</div><div class="h">One effect outside the theory</div>Unrotated GPTQ with act-order beats a random rotation on 27% of the layers, including 19 of the 24 gate/up inputs. Axis-aligned outliers do not explain it: all these layers still win after the largest 1% of eigenvalues are clipped.</div>
  <div class="gr-card ev" style="--accent:var(--c-none)"><div class="num">n<sup>−1/2</sup></div><div class="h">Ordering after rotation</div>After a Haar rotation the column order is asymptotically irrelevant: the saving from a greedy min-pivot ordering decays like n<sup>−1/2</sup>. Lattice reduction changes little too: on real Hessians, full LLL after a greedy ordering lowers the sum of the pivots by at most 0.37%.</div>
</div>

<h2 id="weight-side-cost">Rotation also costs on the weight side</h2>

<p class="gr-lead">The WaterSIC analysis models weights as i.i.d. Gaussian, for which a rotation leaves the weight entropy unchanged. Real weights are not Gaussian, and under per-column entropy coding a rotation changes the code length of the weights themselves. Under a standard high-rate assumption (Proposition 5), the gap of rotated GPTQ to WaterSIC in the original basis is a Hessian term plus a weight term, and both can be computed before quantizing anything:</p>

<div class="gr-eq">\[ R_{\rm GPTQ}(WQ)-R_{\rm WaterSIC}(W)\ \to\ \Delta(Q^\top\Sigma Q)+C_{\rm w}(Q),\qquad C_{\rm w}(Q)=\bar h_{\rm col}(WQ)-\bar h_{\rm col}(W) . \]
<small>\(\bar h_{\rm col}\) is the mean differential entropy of the columns. \(C_{\rm w}\) is the transform-coding gain and the non-Gaussianity of the native columns that a variance-equalizing rotation gives up. WaterSIC here is the plain variant, PlainWaterSIC.</small></div>

<figure class="gr-fig" style="--accent:var(--c-rot)">
  <div class="img"><img src="/images/projects/2026-10-gptq-rotation/fig3-rate-and-perplexity.png" alt="Figure 3: measured rate gap against the high-rate prediction without rotation and in the Hadamard basis, and WikiText-2 perplexity against bits per weight." loading="lazy"></div>
  <figcaption><strong>Figure 3.</strong> Left and middle: measured equal-distortion rate gap of entropy-coded GPTQ to PlainWaterSIC against the high-rate prediction, without rotation (left) and in the Hadamard basis (middle), on 64 layers at R = 2, 3, 5; the dashed line is y = x. Right: WikiText-2 perplexity at equal entropy-coded rate (lines; blue Qwen3-1.7B, orange Llama-3.1-8B-Instruct; seed 0) and of fixed-rate INT-b with group-128 scales, with and without Hadamard (crosses). Dotted lines: bf16.
    <span class="take">At R = 4 the prediction has a median error of at most 0.003 bit and a maximum error of 0.015 bit, on all but one of 64 layers (0.07 bit on the remaining one). The weight-side cost has median 0.005–0.036 bit per layer type and is predicted from the weights alone with median error 0.0008 bit. On Llama block 0's q/k/v input it is 1.36 bit, and there Hadamard makes entropy-coded GPTQ worse (1.31 → 1.68 bit).</span></figcaption>
</figure>

<p class="gr-k">End to end</p>
<h3 class="gr-h" id="perplexity">Perplexity at equal entropy-coded rate</h3>
<p class="gr-lead">Every linear layer of all decoder blocks is quantized with sequential GPTQ or PlainWaterSIC. As the layerwise gaps predict, GPTQ without rotation is far worse at 2 to 3 bits, while GPTQ with Hadamard and PlainWaterSIC without rotation are close.</p>

<figure class="gr-fig" style="--accent:var(--c-rot)">
  <div class="gr-tablewrap">
  <table class="gr-table">
    <thead><tr><th>Model</th><th class="num">Bits</th><th class="num">GPTQ</th><th class="num">GPTQ + Hadamard</th><th class="num">PlainWaterSIC</th><th class="num">bf16</th></tr></thead>
    <tbody>
      <tr><td>Qwen3-1.7B</td><td class="num">2</td><td class="num">152.95</td><td class="num">26.93</td><td class="num">26.06</td><td class="num">16.72</td></tr>
      <tr><td></td><td class="num">3</td><td class="num">23.53</td><td class="num">18.80 <span class="sd">± 1.14</span></td><td class="num">17.33 <span class="sd">± 0.02</span></td><td class="num">16.72</td></tr>
      <tr><td>Llama-3.1-8B-Instruct</td><td class="num">2</td><td class="num">41.37</td><td class="num">11.18</td><td class="num">10.37</td><td class="num">7.22</td></tr>
      <tr><td></td><td class="num">3</td><td class="num">12.81</td><td class="num">7.85</td><td class="num">7.76</td><td class="num">7.22</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Table 2.</strong> WikiText-2 perplexity (sequence length 2048) at equal entropy-coded rate. Entries with ±: mean ± s.d. over 3 calibration seeds; other entries: seed 0 only.
    <span class="take">In the one setting with three calibration seeds (Qwen, 3 bits), PlainWaterSIC was better for each seed and much more stable. This is an observation: in the single-seed settings the differences are within the seed-to-seed spread of GPTQ with Hadamard.</span></figcaption>
</figure>

<h2 id="when-is-rotation-enough">When is rotation enough?</h2>

<p class="gr-lead">Deciding per layer needs no quantization run, and the rule itself does not use the theorems.</p>

<div class="gr-grid3">
  <div class="gr-card"><div class="h"><span>1</span>Compute the gap of each basis</div>For each candidate basis (native order, act-order, and the rotation actually used), compute the exact gap from one Cholesky factorization of the transformed Σ.</div>
  <div class="gr-card"><div class="h"><span>2</span>Add the weight-side cost</div>Add C<sub>w</sub> for the rotation, estimated from the weights alone. It is zero for a permutation. Keep the cheapest basis.</div>
  <div class="gr-card"><div class="h"><span>3</span>Fall back to per-column steps</div>If even the cheapest exceeds the overhead you tolerate, use per-column steps. This is rarely needed, but without damping o_proj inputs reach 0.63 bit after rotation.</div>
</div>

<h2 id="scope-and-limitations">Scope and limitations</h2>

<ul class="gr-list">
  <li><strong>Haar rotations, not Hadamard.</strong> The theorems are for Haar rotations. The worst case of the randomized Hadamard transforms used in practice remains open. A rotation that makes Σ asymptotically free from the coordinate projections, such as a Hadamard matrix conjugated by a random signed permutation (Anderson and Farrell 2014), has gap at least Δ(ν) in the limit, so its worst case is at least that of Theorem 2 (Proposition 7). Sign-only transforms are not known to be of this kind.</li>
  <li><strong>High rate and entropy coding.</strong> The gap is a high-rate, entropy-coded quantity, verified down to 3 bits; at 2 bits the measured gaps are 0.02–0.03 bit smaller than predicted. With fixed-rate INT-b and group scales the theory does not apply: there the format itself costs 0.8–1.0 bit relative to entropy coding, and rotation helps mainly through clipping.</li>
  <li><strong>Constants.</strong> The constants of Theorem 1 degrade as κ grows, so its bound is not quantitative for real layers with κ around 10⁴ to 10⁶. The prediction is accurate there all the same, which Propositions 2 and 3 explain.</li>
  <li><strong>What is assumed or not proved.</strong> The weight-side result relies on a high-rate assumption that is proved only for i.i.d. Gaussian rows; the experiments support it to about 0.001 bit. The uniqueness of the worst share q* is numerical. The analysis is layerwise.</li>
</ul>

<h2 id="links-and-citation">Links and citation</h2>

<div class="gr-links">
  <a class="gr-btn primary" href="https://arxiv.org/abs/2611.01669"><i class="ai ai-arxiv" aria-hidden="true"></i>arXiv 2611.01669</a>
  <a class="gr-btn ghost" href="/files/gptq-random-rotation.pdf"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i>Paper (PDF, 28 pages)</a>
  <a class="gr-btn ghost" href="/files/bibtex/li2026gptqrotation.bib"><i class="fa-solid fa-quote-right" aria-hidden="true"></i>BibTeX file</a>
  <a class="gr-btn ghost" href="/publication/2026-10-gptq-rotation"><i class="fa-solid fa-book-open" aria-hidden="true"></i>Publication page</a>
</div>

<div class="gr-bib">
<button class="gr-copy" type="button" data-target="gr-bibtex">Copy</button>
<pre class="no-mathjax"><code id="gr-bibtex">@unpublished{li2026gptqrotation,
  title         = {How Much Does Random Rotation Help {GPTQ}? Sharp Worst-Case Rate Gaps in the High-Rate Regime},
  author        = {Li, Runguo},
  year          = {2026},
  eprint        = {2611.01669},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2611.01669},
  note          = {Manuscript under review}
}</code></pre>
</div>

</div>

<script src="/files/projects/2026-10-gptq-rotation/explore.js" defer></script>
