---
title: "StepShare: Diagnosing and Rebalancing the Generation Gap in Native Unified Multimodal Models"
collection: projects
permalink: /projects/2026-11-stepshare/
date: 2026-11-16
unlisted: true
# No venue is named on this page (double-blind review); author list, paper, code and BibTeX links to be added later.
status: "Preprint"
venue: ""
excerpt: "At equal loss weights, generation in a native unified model is starved rather than conflicted: it gets under 1% of each AdamW step. StepShare measures and rebalances that share; StepShare-Adam closes 91% of the FID gap with no hidden probe cost and no added parameters."
teaser: "/images/projects/2026-11-stepshare/teaser.png"
tags: ["unified multimodal models", "multi-task optimisation", "flow matching", "AdamW"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-stepshare"
published: false
---

<style>
/* Page-scoped styles: everything is under .ss so no other page changes. */
.ss { --c-und: #2a78d6; --c-gen: #eb6834; --c-wgt: #6250d6; --c-rte: #1baf7a; --c-spl: #008300; --c-opt: #2b2b29;
  --ink: #1d1d1b; --ink2: #52514e; --muted: #898781; --line: rgba(11, 11, 11, 0.09); --card: #ffffff; --soft: #f6f6f3;
  --cost: #b9b8b1; --shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 10px 30px rgba(11, 11, 11, 0.06); color: var(--ink); }
html[data-theme="dark"] .ss { --c-und: #3987e5; --c-gen: #d95926; --c-wgt: #9085e9; --c-rte: #169aa0; --c-spl: #2a9d2a; --c-opt: #e6e5e0;
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
.ss-split .row { display: grid; grid-template-columns: 13.5em 1fr 3.6em; align-items: center; gap: 10px; font-size: 0.82em; }
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
.ss .ss-fig { display: block; margin: 0 0 1.2em; padding: 14px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); }
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
.ss-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 0 2px 10px; font-size: 0.82em; color: var(--ink2); }
.ss-ctl label { display: inline-flex; align-items: center; gap: 6px; margin: 0; }
.ss .ss-ctl select { font: inherit; font-size: 1em; padding: 4px 8px; border-radius: 8px; border: 1px solid var(--line); background: var(--card); color: var(--ink); width: auto; height: auto; margin: 0; }
.ss-seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.ss .ss-seg button { border: 0; background: transparent; color: var(--ink2); font: inherit; font-size: 1em; padding: 4px 10px; cursor: pointer; margin: 0; }
.ss .ss-seg button + button { border-left: 1px solid var(--line); }
.ss .ss-seg button[aria-pressed="true"] { background: var(--soft); color: var(--ink); font-weight: 600; }
.ss-charts { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; }
.ss-ct { font-size: 0.82em; font-weight: 600; margin: 0 0 2px; }
.ss-ct span { font-weight: 400; color: var(--ink2); }
.ss-chart svg { display: block; overflow: visible; }
.ss-chart text { font-family: inherit; }
.ss-chart .tk { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.ss-chart .al { fill: var(--ink2); font-size: 11.5px; }
.ss-chart .lb { fill: var(--ink); font-size: 11.5px; }
.ss-chart .mu { fill: var(--muted); font-size: 11px; }
.ss-chart .halo { paint-order: stroke; stroke: var(--card); stroke-width: 3px; stroke-linejoin: round; }
.ss-chart .gl { stroke: var(--line); stroke-width: 1; }
.ss-chart .ax { stroke: var(--muted); stroke-opacity: 0.6; stroke-width: 1; }
.ss-chart .shade { fill: var(--soft); }
.ss-chart .hit { cursor: pointer; outline: none; }
.ss-chart .hit:focus-visible { stroke: var(--c-und); stroke-width: 2; }
.ss-legend { display: flex; flex-wrap: wrap; gap: 3px 12px; font-size: 0.74em; color: var(--ink2); margin: 2px 0 0; }
.ss-legend .k { display: inline-flex; align-items: center; gap: 5px; }
.ss-legend svg { overflow: visible; }
.ss-tip { position: fixed; z-index: 50; pointer-events: none; background: var(--card); color: var(--ink); border: 1px solid var(--line); box-shadow: var(--shadow); border-radius: 8px; padding: 7px 9px; font-size: 0.76em; min-width: 130px; }
.ss-tip .tt { color: var(--ink2); margin-bottom: 3px; }
.ss-tip .tr { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.ss-tip .tr i { display: inline-block; width: 12px; height: 2px; border-radius: 1px; }
.ss-tip .tr b { font-variant-numeric: tabular-nums; }
.ss-tip .tr span { color: var(--ink2); }
.ss details.ss-more { margin: 10px 4px 0; font-size: 0.8em; color: var(--ink2); }
.ss details.ss-more summary { cursor: pointer; font-weight: 600; color: var(--ink); }
.ss .ss-all { width: 100%; border-collapse: collapse; margin: 8px 0 0; font-size: 1em; display: table; border: 0; }
.ss .ss-all th, .ss .ss-all td { padding: 4px 8px; border: 0; border-bottom: 1px solid var(--line); background: transparent; text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
.ss .ss-all th:first-child, .ss .ss-all td:first-child { text-align: left; }
.ss .ss-all thead, .ss .ss-all tr { background: transparent; }
.ss-wipe { position: relative; width: 100%; max-width: 560px; aspect-ratio: 1 / 1; margin: 0 auto; border-radius: 10px; overflow: hidden; user-select: none; touch-action: none; background: var(--soft); }
.ss .ss-wipe img { position: absolute; inset: 0; width: 100%; height: 100%; display: block; margin: 0; }
.ss-wipe .bar { position: absolute; top: 0; bottom: 0; width: 2px; background: #fff; box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.35); transform: translateX(-1px); pointer-events: none; }
.ss-wipe .knob { position: absolute; top: 50%; width: 30px; height: 30px; margin: -15px 0 0 -15px; border-radius: 50%; background: #fff; color: #52514e; box-shadow: 0 1px 6px rgba(0, 0, 0, 0.35); pointer-events: none; display: grid; place-items: center; font-size: 13px; }
.ss-wipe .tag { position: absolute; top: 8px; padding: 3px 8px; border-radius: 6px; background: rgba(0, 0, 0, 0.62); color: #fff; font-size: 0.74em; pointer-events: none; }
.ss-wipe .tag.l { left: 8px; }
.ss-wipe .tag.r { right: 8px; }
.ss .ss-wipe input[type="range"] { position: absolute; inset: 0; width: 100%; height: 100%; margin: 0; opacity: 0; cursor: ew-resize; }
.ss-links { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.ss-link { padding: 14px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); font-size: 0.86em; }
.ss-link .h { font-weight: 700; margin-bottom: 2px; }
.ss-link .s { color: var(--muted); }
/* the layout's page title must not run under the fixed table of contents */
@media screen and (min-width: 1200px) { .page__title { max-width: calc(100% - 250px); } }
.ss .ss-lead { font-size: 0.95em; line-height: 1.6; color: var(--ink2); margin: 0 0 0.9em; max-width: 52em; }
.ss .ss-lead strong { color: var(--ink); }
.ss-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.ss-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.ss-card { position: relative; padding: 14px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.ss-card .h { font-weight: 700; color: var(--ink); margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.ss-card .h i { flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--accent, var(--ink2)); }
.ss-card .h span { flex: none; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.ss-card .num { font-size: 1.75em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; color: var(--ink); }
.ss-card.ev { padding-left: 18px; overflow: hidden; }
.ss-card.ev::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.ss-card.ev .h { margin-top: 4px; }
.ss .ss-terms { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 28px; margin: 0.2em 0 1.4em; padding: 8px 18px; border-radius: 14px; background: var(--soft); border: 1px solid var(--line); font-size: 0.84em; line-height: 1.5; }
.ss .ss-terms > div { padding: 8px 0; border-bottom: 1px solid var(--line); }
.ss .ss-terms > div:nth-last-child(-n+2) { border-bottom: 0; }
.ss .ss-terms dt { display: inline; font-size: 1em; font-weight: 700; color: var(--ink); margin: 0; }
.ss .ss-terms dd { display: inline; font-size: 1em; margin: 0; color: var(--ink2); }
.ss .ss-eq { text-align: center; margin: 12px 0 0; padding: 10px 12px; border-radius: 10px; background: var(--soft); font-size: 0.9em; color: var(--ink); }
.ss .ss-eq small { display: block; color: var(--ink2); font-size: 0.88em; margin-top: 3px; }
.ss pre.ss-code { margin: 10px 0 0; padding: 12px 14px; border-radius: 10px; border: 0; background: #1f1f1e; color: #e8e7e2; font-size: 0.76em; line-height: 1.6; overflow-x: auto; white-space: pre; }
.ss pre.ss-code code { background: transparent; color: inherit; padding: 0; border: 0; font-size: 1em; }
.ss pre.ss-code .c { color: #9a9993; }
.ss pre.ss-code .k { color: #8fb8f0; }
.ss .ss-sub { font-size: 0.8em; font-weight: 700; color: var(--ink); margin: 14px 4px 0; }
.ss-chips { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.ss .ss-chips button { border: 1px solid var(--line); background: var(--card); color: var(--ink2); font: inherit; font-size: 1em; padding: 3px 10px; border-radius: 999px; cursor: pointer; margin: 0; }
.ss .ss-chips button[aria-pressed="true"] { border-color: var(--ink); color: var(--ink); background: var(--soft); font-weight: 600; }
.ss .ss-scope { font-size: 0.8em; color: var(--muted); margin: 0 0 1.2em; }
@media (max-width: 720px) {
  .ss-grid3, .ss-grid2 { grid-template-columns: minmax(0, 1fr); }
  .ss .ss-terms { grid-template-columns: minmax(0, 1fr); }
  .ss .ss-terms > div:nth-last-child(2) { border-bottom: 1px solid var(--line); }
  .ss-hero { padding: 22px 18px 18px; }
  .ss .ss-name { font-size: 2.1em; }
  .ss-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ss-method { grid-template-columns: minmax(0, 1fr); }
  .ss-charts { grid-template-columns: minmax(0, 1fr); }
  .ss-split .row { grid-template-columns: 1fr 3.2em; }
  .ss-split .row b { grid-column: 1 / -1; }
  .ss-links { grid-template-columns: minmax(0, 1fr); }
}
</style>

<div class="ss">

<section class="ss-hero" aria-label="Overview">
  <div class="ss-kicker"><span class="ss-badge">Preprint</span><span>Unified multimodal models · Multi-task optimisation</span></div>
  <p class="ss-name"><span>StepShare</span></p>
  <p class="ss-tagline">In a native unified model, generation is <em>starved</em>, not conflicted.</p>
  <p class="ss-lede">One network that both captions and generates images usually generates worse than a generation-only twin, and the usual fix is to give the two tasks separate parameters. <strong>StepShare</strong> shows where the gap comes from instead: it splits every AdamW step exactly into the part each task contributes, and finds that generation receives under 1% of the update. Rebalancing that share, with no added parameters, closes most of the gap, and <strong>StepShare-Adam</strong> closes 91% of it without the hidden cost that plain reweighting leaves in intermediate layers.</p>
  <div class="ss-btns">
    <a class="ss-btn primary" href="#explore-every-arm"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><circle cx="3" cy="9.5" r="1.7" fill="currentColor"/><circle cx="6.5" cy="5.5" r="1.7" fill="currentColor"/><circle cx="10.5" cy="3" r="1.7" fill="currentColor"/></svg>Explore the results</a>
    <a class="ss-btn ghost" href="#compare-samples"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><rect x="1" y="1.5" width="11" height="10" rx="2" fill="none" stroke="currentColor" stroke-width="1.3"/><path d="M6.5 1.5v10" stroke="currentColor" stroke-width="1.3"/></svg>Compare samples</a>
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

<h2 id="the-problem">The problem</h2>

<p class="ss-lead">A <strong>native unified model</strong> uses one transformer for image understanding (here, captioning an image) and image generation (text to image), with every weight shared between the two. Trained on both tasks at equal loss weights, our model captions as well as an understanding-only model, but its FID rises from 152 to 238, which is 57% worse than the same model trained on generation alone. This is the <strong>generation gap</strong>.</p>

<p class="ss-lead">Recent controlled studies answer the gap with architecture: give the two tasks separate parameters. Those comparisons are made at a single loss weighting, so they cannot tell whether generation suffers from <em>sharing parameters</em> or from <em>how the optimiser divides each update</em> between the two tasks. StepShare measures that division directly, and then corrects it.</p>

<div class="ss-grid3">
  <div class="ss-card" style="--accent:var(--c-und)"><div class="h"><i></i>Model</div>Qwen3-0.6B (and 1.7B) reading raw 256×256 pixels as 16-pixel patches. Understanding is trained with caption likelihood, generation with flow matching, on the same image tokens through the same layers.</div>
  <div class="ss-card" style="--accent:var(--c-gen)"><div class="h"><i></i>Training</div>100k COCO images with five captions each, 9000 steps, AdamW at 10<sup>−4</sup>, batch 32. For a given seed, every variant sees the same batches, captions and sampling noise.</div>
  <div class="ss-card" style="--accent:var(--c-spl)"><div class="h"><i></i>Evaluation</div>FID over 5000 samples for generation; caption likelihood and linear probes at every second layer (COCO, ImageNet-100) for understanding. Three to five seeds per variant, with the criteria fixed before the runs.</div>
</div>

<p class="ss-k">Terms used below</p>
<dl class="ss-terms">
  <div><dt>Dense.</dt> <dd>Every parameter is shared by both tasks; there are no task-specific weights.</dd></div>
  <div><dt>w.</dt> <dd>The weight on the understanding loss. The generation loss has weight 1, so w = 1 means equal weights.</dd></div>
  <div><dt>FID gap closed.</dt> <dd>How far a variant moves from dense training at w = 1 (0%) to generation-only training (100%). Higher is better.</dd></div>
  <div><dt>Probe change.</dt> <dd>Accuracy of a linear classifier on one layer's frozen image features, minus the same for an understanding-only model. Negative means the features got worse; the "worst layer" is the lowest value over layers 2 to 28.</dd></div>
  <div><dt>Task routing.</dt> <dd>Generation image tokens get their own feed-forward block in every layer. This is the "separate the parameters" baseline.</dd></div>
  <div><dt>Per-task Adam.</dt> <dd>One AdamW state per task over the shared parameters, so each task's step is normalised by its own statistics.</dd></div>
  <div><dt>Gradient ratio ρ.</dt> <dd>The norm of the weighted understanding gradient over the norm of the generation gradient, w‖g<sub>u</sub>‖ / ‖g<sub>g</sub>‖.</dd></div>
  <div><dt>Guidance 1 and 4.</dt> <dd>The classifier-free guidance scale used when sampling images for FID.</dd></div>
</dl>

<h2 id="key-results">Key results</h2>

<p class="ss-k">Finding</p>
<h3 class="ss-h" id="starved-not-conflicted">Starved, not conflicted</h3>
<p class="ss-lead">If the two tasks pulled the shared weights in opposite directions, their gradients would point against each other. They do not: the gradients are nearly orthogonal (a). The understanding gradient is simply about 100 times larger, and after clipping and Adam's normalisation generation is left with 0.9% of each step. Giving generation a larger share closes most of the FID gap (b), but lowers the quality of understanding features in intermediate layers, which the usual metrics miss (c).</p>
<figure class="ss-fig" style="--accent:var(--c-gen)">
  <div class="img"><img src="/images/projects/2026-11-stepshare/fig1-teaser.png" alt="Figure 1: StepShare at equal loss weights, restoring the share, and the hidden cost by layer." loading="lazy"></div>
  <figcaption><strong>Figure 1.</strong> StepShare: at equal loss weights generation is starved, and restoring its share closes most of the gap.
    <span class="take">At equal weights the two gradients are nearly orthogonal and generation gets 0.9% of the AdamW step. Lowering the understanding weight restores the share and closes most of the FID gap, more than task routing does at equal weights, but costs up to 5.4 probe points in intermediate layers that the last layer and caption NLL do not show.</span></figcaption>
</figure>

<p class="ss-k">Method</p>
<h3 class="ss-h" id="measure-then-rebalance">Measure the share, then rebalance it</h3>
<p class="ss-lead">StepShare has two parts. The first is a measurement that can be logged in any joint model whose two losses are backpropagated separately. The second uses the measurement to give generation its share back in the layers where it helps and to hold it back in the layers where it costs.</p>
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
  <p class="ss-eq">generation's share = ‖Δ<sub>g</sub>‖² / (‖Δ<sub>u</sub>‖² + ‖Δ<sub>g</sub>‖²)
    <small>averaged over training steps, for the shallow half (patch embedding and layers 0–13) and the deep half (layers 14–27)</small></p>
  <p class="ss-sub">StepShare-Adam in a training loop (PyTorch-style sketch)</p>
<pre class="ss-code"><code><span class="c"># One AdamW per task over the same parameters; generation's learning rate is scaled by beta in the shallow half.</span>
opt_und = AdamW(params, lr=lr)
opt_gen = AdamW([{"params": shallow, "lr": beta * lr},   <span class="c"># patch embedding + lower half of the layers, beta = 0.3</span>
                 {"params": deep,    "lr": lr}])

<span class="k">for</span> batch <span class="k">in</span> loader:
    g_und = grad(loss_und(batch), params)                <span class="c"># the two backward passes joint training already runs</span>
    g_gen = grad(loss_gen(batch), params)
    apply(opt_und, clip_norm(g_und, 1.0))                <span class="c"># each task's gradient is clipped to norm 1 ...</span>
    apply(opt_gen, clip_norm(g_gen, 1.0))                <span class="c"># ... and stepped by its own optimiser</span></code></pre>
</figure>

<p class="ss-k">Results</p>
<h3 class="ss-h" id="closing-the-gap">Closing the gap without the hidden cost</h3>
<p class="ss-lead">The table compares the main ways of restoring generation's share. A good method has a long coloured bar (most of the FID gap closed) and a short grey bar (little damage to understanding features). Reweighting, routing and per-task Adam each give the first; only StepShare-Adam gives both without adding parameters.</p>
<figure class="ss-fig" style="--accent:var(--c-spl)">
  <div class="ss-tablewrap">
  <table class="ss-table">
    <thead><tr><th>Training variant (0.6B)</th><th>Seeds</th><th>FID gap closed, guidance 1</th><th>Guidance 4</th><th>Worst COCO probe layer</th><th>Worst IN-100</th></tr></thead>
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

<p class="ss-k">Interactive</p>
<h3 class="ss-h" id="explore-every-arm">Explore every training variant</h3>
<p class="ss-lead">Each point on the left is one of the 14 training variants of the paper's main table. Higher means more of the FID gap closed; further right means a smaller cost to understanding features, so the best place is the top-right corner. Select a variant to see, on the right, in which layers its probe cost sits.</p>
<figure class="ss-fig" id="ss-explore" style="--accent:var(--c-spl)">
  <div class="ss-ctl">
    <span class="ss-chips" role="group" aria-label="Quick picks"><span>Try</span><button type="button" data-pick="w1">Equal weights</button><button type="button" data-pick="w003">w = 0.03</button><button type="button" data-pick="adam">Per-task Adam</button><button type="button" data-pick="ss">StepShare-Adam</button></span>
    <label>or <select id="ss-arm" aria-label="Training variant"></select></label>
    <label>Probe <span class="ss-seg" role="group" aria-label="Probe set"><button type="button" data-probe="coco" aria-pressed="true">COCO</button><button type="button" data-probe="in" aria-pressed="false">ImageNet-100</button></span></label>
  </div>
  <div class="ss-charts">
    <div><div class="ss-ct">FID gap closed vs. hidden cost <span>· click a point to select it</span></div><div class="ss-chart" id="ss-scatter"></div><div class="ss-legend" id="ss-scatter-legend"></div></div>
    <div><div class="ss-ct">Probe change by layer <span id="ss-profile-sub">· COCO, vs. und.-only</span></div><div class="ss-chart" id="ss-profile"></div><div class="ss-legend" id="ss-profile-legend"></div></div>
  </div>
  <figcaption><span id="ss-readout"><strong>StepShare-Adam.</strong> 91% ± 14 of the FID gap closed at guidance 1; worst COCO layer −0.5 points.</span>
    <span class="take" id="ss-note">Per-task Adam with generation's step scaled by β = 0.3 in the shallow half closes 91% of the gap with a worst COCO layer of −0.5 and no added parameters.</span></figcaption>
  <details class="ss-more"><summary>All 14 variants of Table 2</summary><table class="ss-all" id="ss-all"></table></details>
</figure>

<p class="ss-k">Analysis</p>
<h3 class="ss-h" id="share-follows-ratio">The share follows the gradient ratio</h3>
<p class="ss-lead">The gradient ratio ρ is cheap to log and tells in advance how a recipe will behave. During equal-weight training it grows as the generation loss falls (a); across loss weights and clipping schemes, the lower the average ratio, the more of the gap is closed (b); and along the weight sweep each gain in FID is paid for in probe accuracy, a trade-off that allocating the share by depth leaves (c).</p>
<figure class="ss-fig" style="--accent:var(--c-wgt)">
  <div class="img"><img src="/images/projects/2026-11-stepshare/fig3-budget.png" alt="Figure 3: Gradient ratio during training and the trade-offs of restoring the share." loading="lazy"></div>
  <figcaption><strong>Figure 3.</strong> StepShare's gradient ratio and the trade-offs of restoring the share, 0.6B unless marked.
    <span class="take">At equal weights the gradient ratio rises from 11 to 60 as the generation loss falls, and FID recovery follows the training-average ratio across loss weights and clipping schemes. Along the weight sweep, more recovery comes with a deeper probe deficit; depth allocation and StepShare-Adam leave that trade-off.</span></figcaption>
</figure>

<p class="ss-k">Samples</p>
<h3 class="ss-h" id="same-noise-samples">Same captions, same noise</h3>
<p class="ss-lead">The models here are small and trained for only 9000 steps, so every sample is coarse. What matters is the difference between rows: each column uses the same caption and the same initial noise, so a row differs from the one above only in how training divided the update.</p>
<figure class="ss-fig" style="--accent:var(--c-spl)">
  <div class="img"><img src="/images/projects/2026-11-stepshare/fig4-samples.webp" alt="Figure 4: Samples from the same captions and initial noise for four arms." loading="lazy"></div>
  <figcaption><strong>Figure 4.</strong> Samples from the same noise.
    <span class="take">At equal weights the dense model produces saturated high-frequency texture; restoring generation's share with StepShare-Adam gives layouts close to generation-only training.</span></figcaption>
</figure>

<figure class="ss-fig" id="ss-compare" data-base="/images/projects/2026-11-stepshare/samples" style="--accent:var(--c-spl)">
  <div class="ss-ctl" id="compare-samples">
    <label>Left <select id="ss-wa" aria-label="Left grid"></select></label>
    <label>Right <select id="ss-wb" aria-label="Right grid"></select></label>
    <span>Drag across the grid to compare.</span>
  </div>
  <div class="ss-wipe">
    <img id="ss-ia" src="/images/projects/2026-11-stepshare/samples/dense.webp" alt="36 samples, Dense, w = 1">
    <img id="ss-ib" src="/images/projects/2026-11-stepshare/samples/ssadam.webp" alt="36 samples, StepShare-Adam" style="clip-path: inset(0 0 0 50%)">
    <div class="bar" style="left:50%"></div><div class="knob" style="left:50%" aria-hidden="true">⇆</div>
    <div class="tag l" id="ss-ta">Dense, w = 1 · FID 340</div><div class="tag r" id="ss-tb">StepShare-Adam · FID 142</div>
    <input type="range" id="ss-wpos" min="0" max="100" value="50" step="0.5" aria-label="Comparison divider position">
  </div>
  <figcaption><strong>Compare any two variants.</strong> The first 36 FID samples of each variant without selection (seed 0, guidance scale 4); every cell uses the same caption and initial noise in all grids, so cells compare one to one. FID is that seed's.</figcaption>
</figure>

<h2 id="more-evidence">More evidence</h2>

<p class="ss-lead">Four further checks from the paper support the reading that the gap comes from the division of the update.</p>

<div class="ss-grid2">
  <div class="ss-card ev" style="--accent:var(--c-gen)"><div class="num">62–85%</div><div class="h">A small update alone reproduces the gap</div>A generation-only model trained at learning rate 10<sup>−5</sup>, where its update is about as small as generation's part of the joint update, reproduces 62% and 85% of the FID increase (guidance 1 and 4). No second task is needed to cause the damage.</div>
  <div class="ss-card ev" style="--accent:var(--c-rte)"><div class="num">−2 points</div><div class="h">Routing's gain is the optimiser state</div>Under one shared AdamW, task routing adds 67 and 51 points of gap closed (two seeds). On top of per-task Adam it adds −2 and −2: at equal weights, separating the Adam statistics accounts for all of routing's gain.</div>
  <div class="ss-card ev" style="--accent:var(--c-spl)"><div class="num">83%</div><div class="h">It transfers to a larger model</div>With the 0.6B settings unchanged, StepShare-Adam closes 83% of the gap at both guidance scales on Qwen3-1.7B (two seeds).</div>
  <div class="ss-card ev" style="--accent:var(--c-und)"><div class="num">262×</div><div class="h">The same imbalance in a released 7B model</div>In BAGEL-7B-MoT at its published 0.25:1 weighting, the cross-entropy gradient on the understanding expert is 262 times the MSE gradient on the generation expert, and 434 times the MSE gradient that reaches the understanding expert through attention.</div>
</div>

<h2 id="using-stepshare">Using StepShare</h2>

<p class="ss-lead">StepShare gives an order of checks to run before adding task-specific parameters to a unified model.</p>

<div class="ss-grid3">
  <div class="ss-card"><div class="h"><span>1</span>Log the gradient ratio</div>It costs little when the two losses are already backpropagated separately. In our runs a training-average ratio below about 1.5 went with more than 80% of the gap closed, and one above about 4 with less than 70%.</div>
  <div class="ss-card"><div class="h"><span>2</span>Restore the share, and probe inside</div>If the ratio is large, lower w or use per-task Adam. Then check linear probes at intermediate layers as well as the last one: the cost does not show in caption likelihood, retrieval or a VQA probe.</div>
  <div class="ss-card"><div class="h"><span>3</span>Allocate by depth</div>Where a deficit appears, use StepShare-Adam. It keeps a second set of AdamW moments (about 5 GB at 0.6B) and adds no measured training time. Judge separate parameters by what they add after rebalancing.</div>
</div>

<p class="ss-scope">Scope: Qwen3-0.6B and 1.7B trained for 9000 steps, one 90000-step run, AdamW only.</p>

<h2 id="links-and-citation">Links and citation</h2>

<div class="ss-links">
  <div class="ss-link"><div class="h">Paper</div><div class="s">The PDF and BibTeX will be linked here when they are public.</div></div>
  <div class="ss-link"><div class="h">Code</div><div class="s">Will be released with the paper.</div></div>
</div>

</div>

<script src="/files/projects/2026-11-stepshare/explore.js" defer></script>
