---
title: "What Horizon-Aware Losses Do in Multi-Step Forecasting: Robust Norms and Instance Normalization, Not Decorrelation"
collection: projects
permalink: /projects/2026-11-horizon-aware-losses/
date: 2026-11-22
# No venue is named on this page; author list, paper, code and BibTeX links to be added when the paper is public.
status: "In preparation"
venue: "Manuscript in preparation"
excerpt: "FreDF, Time-o1 and QDF are motivated by a decorrelation bias of the MSE. Taking their official code apart shows that L2 versions in the same bases match or trail the MSE, that the gains come from the L1 norm, and that their size depends on instance normalization. All 34 predictions recorded before the runs are listed with their outcomes."
teaser: "/images/projects/2026-11-horizon-aware-losses/teaser.png"
tags: ["time-series forecasting", "training losses", "robust estimation", "instance normalization", "recorded predictions"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-horizon-aware-losses"
published: false
---

<style>
/* Page-scoped styles: everything is under .hl so no other page changes. */
.hl { --c-b: #2a78d6; --c-w: #eb6834; --c-good: #0ca30c; --c-bad: #d03b3b;
  --ink: #1d1d1b; --ink2: #52514e; --muted: #898781; --line: rgba(11, 11, 11, 0.09); --card: #ffffff; --soft: #f6f6f3;
  --band2: #efefec; --band1: #e3eefb; --axis: #c3c2b7; --grid: #e1e0d9;
  --shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 10px 30px rgba(11, 11, 11, 0.06); color: var(--ink); }
html[data-theme="dark"] .hl { --c-b: #3987e5; --c-w: #d95926; --c-good: #0ca30c; --c-bad: #e66767;
  --ink: #f3f3f1; --ink2: #d0cfc8; --muted: #a9a8a1; --line: rgba(255, 255, 255, 0.10); --card: #3b3b3b; --soft: #404040;
  --band2: #444444; --band1: #364559; --axis: #6f6e69; --grid: #4a4a48;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.25), 0 10px 30px rgba(0, 0, 0, 0.2); }
.hl p { margin: 0 0 0.9em; }
.hl-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 28px 28px 22px; margin: 0.4em 0 1.4em; border: 1px solid var(--line);
  background: radial-gradient(70% 120% at 0% 0%, rgba(42, 120, 214, 0.12), transparent 60%), var(--card); box-shadow: var(--shadow); }
.hl-hero::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: linear-gradient(90deg, var(--muted) 0%, var(--muted) 34%, var(--c-b) 34%, var(--c-b) 100%); }
.hl-kicker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.72em; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin-bottom: 10px; }
.hl-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; background: var(--ink); color: var(--card); font-weight: 700; letter-spacing: 0.04em; }
.hl .hl-name { font-size: 2.3em; line-height: 1.08; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 10px; }
.hl .hl-name s { text-decoration-thickness: 3px; text-decoration-color: var(--c-w); color: var(--muted); }
.hl .hl-tagline { font-size: 1.2em; line-height: 1.35; font-weight: 600; margin: 0 0 14px; }
.hl .hl-lede { font-size: 0.95em; color: var(--ink2); max-width: 46em; }
.hl-btns { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 20px; }
.hl .hl-btn { display: inline-flex; align-items: center; gap: 7px; padding: 7px 15px; border-radius: 999px; font-size: 0.85em; font-weight: 600; text-decoration: none; border: 1px solid var(--ink); }
.hl .hl-btn.primary { background: var(--ink); color: var(--card); }
.hl .hl-btn.ghost { background: transparent; color: var(--ink); }
.hl .hl-btn.soon { background: transparent; color: var(--muted); border-color: var(--line); cursor: default; }
.hl .hl-btn svg { flex: none; }
/* transform x norm matrix */
.hl-mx { display: grid; grid-template-columns: 4.6em repeat(3, minmax(0, 1fr)); gap: 6px; font-size: 0.84em; }
.hl-mx .ch { font-size: 0.86em; font-weight: 700; color: var(--ink2); padding: 0 10px 2px; align-self: end; }
.hl-mx .ch small { display: block; font-weight: 400; color: var(--muted); }
.hl-mx .rh { display: grid; place-items: center; text-align: center; border-radius: 10px; font-weight: 800; line-height: 1.15; padding: 6px 4px; }
.hl-mx .rh small { display: block; font-weight: 500; font-size: 0.78em; color: var(--ink2); }
.hl-mx .r2 { background: var(--band2); }
.hl-mx .r1 { background: var(--band1); }
.hl-mx .cell { border-radius: 10px; padding: 9px 10px; display: grid; gap: 5px; align-content: start; }
.hl-mx .cell.pub { box-shadow: inset 0 0 0 2px var(--ink); }
.hl-mx .it { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 8px; }
.hl-mx .it b { font-weight: 600; }
.hl-mx .it span { display: inline-flex; align-items: center; gap: 5px; font-variant-numeric: tabular-nums; color: var(--ink2); white-space: nowrap; }
.hl-mx .it i { flex: none; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; }
.hl-mx .it i.dn { border-top: 8px solid var(--c-b); }
.hl-mx .it i.up { border-bottom: 8px solid var(--c-w); }
.hl-mx .it i.eq { width: 9px; height: 3px; border: 0; background: var(--muted); border-radius: 2px; }
.hl-mx .note { grid-column: 2 / -1; font-size: 0.86em; color: var(--ink2); padding: 0 10px 4px; }
.hl .hl-mxcap { font-size: 0.75em; color: var(--muted); margin: 10px 2px 0; line-height: 1.5; }
.hl .hl-mxcap i { display: inline-block; width: 0; height: 0; border-left: 5px solid transparent; border-right: 5px solid transparent; margin: 0 3px 0 6px; }
.hl .hl-mxcap i.dn { border-top: 8px solid var(--c-b); }
.hl .hl-mxcap i.up { border-bottom: 8px solid var(--c-w); }
.hl-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 1.6em; }
.hl-stat { position: relative; padding: 14px 14px 12px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }
.hl-stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.hl-stat .num { font-size: 1.55em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; white-space: nowrap; }
.hl-stat .num small { font-size: 0.6em; font-weight: 700; color: var(--ink2); }
.hl-stat .lab { font-size: 0.78em; color: var(--ink2); line-height: 1.35; margin-top: 4px; }
.hl .hl-k { font-size: 0.7em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 1.8em 0 0.1em; }
.hl h3.hl-h { font-size: 1.1em; margin: 0 0 0.55em; }
.hl .hl-lead { font-size: 0.95em; line-height: 1.6; color: var(--ink2); margin: 0 0 0.9em; max-width: 52em; }
.hl .hl-lead strong { color: var(--ink); }
.hl .hl-fig { display: block; margin: 0 0 1.2em; padding: 14px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); }
.hl-fig .img { background: #fff; border-radius: 10px; padding: 8px; }
.hl-fig img { display: block; width: 100%; height: auto; margin: 0 auto; }
.hl .hl-fig figcaption { font-family: inherit; font-size: 0.82em; color: var(--ink2); line-height: 1.5; margin: 10px 4px 0; }
.hl-fig figcaption strong { color: var(--ink); }
.hl-fig .take { display: block; margin-top: 6px; padding: 7px 10px; border-radius: 8px; background: var(--soft); color: var(--ink); border-left: 3px solid var(--accent, var(--ink2)); }
.hl-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.hl-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.hl-card { position: relative; padding: 14px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.hl-card .h { font-weight: 700; color: var(--ink); margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.hl-card .h i { flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--accent, var(--ink2)); }
.hl-card .h span { flex: none; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.hl-card .num { font-size: 1.6em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; color: var(--ink); }
.hl-card.ev { padding-left: 18px; overflow: hidden; }
.hl-card.ev::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.hl-card.ev .h { margin-top: 4px; }
.hl-card .eq { margin: 8px 0 6px; padding: 8px 4px; font-size: 0.88em; border-radius: 8px; background: var(--soft); color: var(--ink); text-align: center; overflow-x: auto; overflow-y: hidden; }
.hl-card .ref { display: block; margin-top: 6px; font-size: 0.9em; color: var(--muted); }
.hl .hl-terms { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 28px; margin: 0.2em 0 1.4em; padding: 8px 18px; border-radius: 14px; background: var(--soft); border: 1px solid var(--line); font-size: 0.84em; line-height: 1.5; }
.hl .hl-terms > div { padding: 8px 0; border-bottom: 1px solid var(--line); }
.hl .hl-terms > div:nth-last-child(-n+2) { border-bottom: 0; }
.hl .hl-terms dt { display: inline; font-size: 1em; font-weight: 700; color: var(--ink); margin: 0; }
.hl .hl-terms dd { display: inline; font-size: 1em; margin: 0; color: var(--ink2); }
/* controls */
.hl-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 0 2px 10px; font-size: 0.82em; color: var(--ink2); }
.hl-ctl label { display: inline-flex; align-items: center; gap: 6px; margin: 0; cursor: pointer; }
.hl .hl-ctl input[type="checkbox"] { margin: 0; width: auto; }
.hl-seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.hl .hl-seg button { border: 0; background: transparent; color: var(--ink2); font: inherit; font-size: 1em; padding: 4px 10px; cursor: pointer; margin: 0; }
.hl .hl-seg button + button { border-left: 1px solid var(--line); }
.hl .hl-seg button[aria-pressed="true"] { background: var(--soft); color: var(--ink); font-weight: 600; }
.hl-chips { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.hl .hl-chips button { border: 1px solid var(--line); background: var(--card); color: var(--ink2); font: inherit; font-size: 1em; padding: 3px 10px; border-radius: 999px; cursor: pointer; margin: 0; }
.hl .hl-chips button[aria-pressed="true"] { border-color: var(--ink); color: var(--ink); background: var(--soft); font-weight: 600; }
/* charts */
.hl-scroll { overflow-x: auto; }
.hl-chart svg { display: block; overflow: visible; width: 100%; height: auto; }
.hl-chart text { font-family: inherit; }
.hl-chart .tk { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.hl-chart .al { fill: var(--ink2); font-size: 11.5px; }
.hl-chart .lb { fill: var(--ink); font-size: 11.5px; }
.hl-chart .lbb { fill: var(--ink); font-size: 11.5px; font-weight: 700; }
.hl-chart .mu { fill: var(--muted); font-size: 10.5px; }
.hl-chart .vl { fill: var(--ink); font-size: 11.5px; font-weight: 600; font-variant-numeric: tabular-nums; }
.hl-chart .halo { paint-order: stroke; stroke: var(--card); stroke-width: 3px; stroke-linejoin: round; }
.hl-chart .gl { stroke: var(--grid); stroke-width: 1; }
.hl-chart .ax { stroke: var(--axis); stroke-width: 1; }
.hl-chart .hit { fill: transparent; cursor: default; outline: none; }
.hl-chart .hit:hover, .hl-chart .hit:focus-visible { fill: var(--ink); fill-opacity: 0.05; }
.hl-legend { display: flex; flex-wrap: wrap; gap: 3px 14px; font-size: 0.76em; color: var(--ink2); margin: 0 2px 6px; }
.hl-legend .k { display: inline-flex; align-items: center; gap: 5px; }
.hl-legend svg { overflow: visible; }
.hl-tip { position: fixed; z-index: 50; pointer-events: none; background: var(--card); color: var(--ink); border: 1px solid var(--line); box-shadow: var(--shadow); border-radius: 8px; padding: 7px 9px; font-size: 0.76em; min-width: 150px; max-width: 280px; }
.hl-tip .tt { font-weight: 700; margin-bottom: 3px; }
.hl-tip .tr { display: flex; justify-content: space-between; gap: 14px; white-space: nowrap; }
.hl-tip .tr span { color: var(--ink2); }
.hl-tip .tr b { font-variant-numeric: tabular-nums; }
.hl-tip .tn { color: var(--ink2); margin-top: 4px; white-space: normal; }
/* toy calculation */
.hl-toy { display: grid; gap: 10px; padding: 14px 16px 12px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); font-size: 0.84em; }
.hl-toy .sl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
.hl .hl-toy input[type="range"] { flex: 1 1 180px; min-width: 150px; margin: 0; accent-color: var(--c-b); }
.hl-toy .row { display: grid; grid-template-columns: 9.5em 1fr; align-items: center; gap: 10px; }
.hl-toy .bar { display: flex; gap: 2px; height: 28px; }
.hl-toy .bar i { display: flex; align-items: center; padding-left: 8px; border-radius: 4px; color: #fff; font-style: normal; font-weight: 700; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; min-width: 0; }
.hl-toy .bar i.rest { background: var(--axis); color: var(--ink); font-weight: 400; }
.hl-toy .cap { font-size: 0.9em; color: var(--ink2); line-height: 1.5; }
/* tables */
.hl-tablewrap { overflow-x: auto; }
.hl .hl-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.82em; margin: 0; display: table; border: 0; background: transparent; }
.hl .hl-table thead, .hl .hl-table thead tr, .hl .hl-table tbody tr { background: transparent; }
.hl .hl-table th, .hl .hl-table td { border: 0; border-bottom: 1px solid var(--line); background: transparent; padding: 8px 10px; vertical-align: top; }
.hl .hl-table th { text-align: left; font-weight: 600; color: var(--ink2); white-space: nowrap; }
.hl .hl-heat td.v { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; vertical-align: middle; background: color-mix(in srgb, var(--hc, transparent) var(--ha, 0%), transparent); }
.hl .hl-heat th.v { text-align: right; }
.hl .hl-heat td.v small { display: block; color: var(--ink2); font-size: 0.82em; }
.hl .hl-heat td.g { font-weight: 700; color: var(--ink); white-space: nowrap; vertical-align: middle; }
.hl .hl-heat tr.sep td { border-bottom: 2px solid var(--line); }
.hl .hl-heat td.avg, .hl .hl-heat th.avg { border-left: 1px solid var(--line); font-weight: 700; }
.hl-pill { display: inline-flex; align-items: center; gap: 5px; padding: 1px 8px 1px 6px; border-radius: 999px; border: 1px solid var(--line); font-size: 0.92em; font-weight: 600; white-space: nowrap; color: var(--ink); }
.hl-pill i { flex: none; display: inline-grid; place-items: center; width: 14px; height: 14px; border-radius: 50%; color: #fff; font-style: normal; font-size: 9px; font-weight: 800; }
.hl-pill.held i { background: var(--c-good); }
.hl-pill.fail i { background: var(--c-bad); }
.hl-pill.part i { background: var(--muted); }
.hl .hl-pred td:nth-child(1), .hl .hl-pred td:nth-child(2) { white-space: nowrap; font-variant-numeric: tabular-nums; }
.hl .hl-pred td.o { color: var(--ink2); }
.hl .hl-pred { min-width: 640px; }
.hl-score { display: flex; gap: 2px; height: 28px; margin: 0 0 10px; }
.hl-score i { display: flex; align-items: center; padding-left: 9px; border-radius: 4px; color: #fff; font-style: normal; font-size: 0.8em; font-weight: 700; white-space: nowrap; overflow: hidden; }
.hl .hl-scope { font-size: 0.8em; color: var(--muted); margin: 0 0 1.2em; line-height: 1.55; }
.hl-links { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.hl-link { padding: 14px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); font-size: 0.86em; }
.hl-link .h { font-weight: 700; margin-bottom: 2px; }
.hl-link .s { color: var(--muted); }
/* the layout's page title must not run under the fixed table of contents */
@media screen and (min-width: 1200px) { .page__title { max-width: calc(100% - 250px); } }
@media (max-width: 720px) {
  .hl-grid3, .hl-grid2 { grid-template-columns: minmax(0, 1fr); }
  .hl .hl-terms { grid-template-columns: minmax(0, 1fr); }
  .hl .hl-terms > div:nth-last-child(2) { border-bottom: 1px solid var(--line); }
  .hl-hero { padding: 22px 16px 18px; }
  .hl .hl-name { font-size: 1.8em; }
  .hl-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .hl-links { grid-template-columns: minmax(0, 1fr); }
  .hl-mx { grid-template-columns: minmax(0, 1fr); }
  .hl-mx .ch { display: none; }
  .hl-mx .rh { grid-auto-flow: column; justify-content: start; gap: 8px; padding: 6px 10px; margin-top: 4px; }
  .hl-mx .cell::before { content: attr(data-t); font-size: 0.8em; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .hl-mx .note { grid-column: 1; }
  .hl-toy .row { grid-template-columns: minmax(0, 1fr); gap: 3px; }
  .hl-stat .num { white-space: normal; font-size: 1.3em; }
  .hl-fig .img { overflow-x: auto; }
  .hl-fig .img img { min-width: 620px; }
  /* predictions: one stacked block per record, so the status stays visible without sideways scrolling */
  .hl .hl-pred { min-width: 0; display: block; }
  .hl .hl-pred thead { display: none; }
  .hl .hl-pred tbody { display: block; }
  .hl .hl-pred tr { display: grid; grid-template-columns: auto auto minmax(0, 1fr) auto; gap: 3px 8px; align-items: center; padding: 9px 2px; border-bottom: 1px solid var(--line); }
  .hl .hl-pred td { border: 0; padding: 0; }
  .hl .hl-pred td:nth-child(1) { grid-area: 1 / 1; font-weight: 700; }
  .hl .hl-pred td:nth-child(2) { grid-area: 1 / 2; color: var(--muted); }
  .hl .hl-pred td:nth-child(2)::before { content: "round "; }
  .hl .hl-pred td:nth-child(3) { grid-area: 1 / 3; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .hl .hl-pred td:nth-child(6) { grid-area: 1 / 4; }
  .hl .hl-pred td:nth-child(4) { grid-area: 2 / 1 / 3 / -1; }
  .hl .hl-pred td:nth-child(5) { grid-area: 3 / 1 / 4 / -1; }
}
</style>

<div class="hl">

<section class="hl-hero" aria-label="Overview">
  <div class="hl-kicker"><span class="hl-badge">In preparation</span><span>Time-series forecasting · Training losses · Analysis</span></div>
  <p class="hl-name">The L1 norm, not <s>decorrelation</s></p>
  <p class="hl-tagline">Horizon-aware forecasting losses help because of the norm they use, not because of the basis they move to.</p>
  <p class="hl-lede">FreDF, Time-o1 and QDF replace the mean squared error (MSE) in direct multi-step forecasting, on the argument that correlation among the future values biases it. We re-ran their official code and wrote each loss as a <strong>transform</strong> of the forecast error followed by a <strong>norm</strong>. With an L2 norm, every transform we tried matches or trails the MSE. With an L1 norm, the gains appear in every basis, and at a lookback of 96 they are much larger, on most datasets, for models that do not normalize their input windows.</p>
  <div class="hl-btns">
    <a class="hl-btn primary" href="#l2-is-the-mse"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><circle cx="3" cy="9.5" r="1.7" fill="currentColor"/><circle cx="6.5" cy="5.5" r="1.7" fill="currentColor"/><circle cx="10.5" cy="3" r="1.7" fill="currentColor"/></svg>Explore the results</a>
    <a class="hl-btn ghost" href="#predictions"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M2 6.8l3 3 6-6.6" fill="none" stroke="currentColor" stroke-width="1.6"/></svg>Predictions and outcomes</a>
    <span class="hl-btn soon" aria-disabled="true"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M3 1h5l3 3v8H3z M8 1v3h3" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>Paper · coming soon</span>
    <span class="hl-btn soon" aria-disabled="true"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M4.5 3 1.5 6.5l3 3.5M8.5 3l3 3.5-3 3.5" fill="none" stroke="currentColor" stroke-width="1.4"/></svg>Code · coming soon</span>
  </div>
  <div class="hl-mx" role="table" aria-label="Losses as a transform followed by a norm, with their measured change in test MSE">
    <div></div>
    <div class="ch">Time domain<small>no transform</small></div>
    <div class="ch">Fourier<small>real FFT of the error</small></div>
    <div class="ch">Label PCA<small>principal components of the labels</small></div>
    <div class="rh r2">L2<small>norm</small></div>
    <div class="cell r2" data-t="Time domain"><div class="it"><b>MSE</b><span>reference</span></div></div>
    <div class="cell r2" data-t="Fourier"><div class="it"><b>FFT L2</b><span><i class="eq"></i>0.0%</span></div><div class="it"><b>rFFT L2</b><span><i class="up"></i>+0.4 to +3.9%</span></div></div>
    <div class="cell r2" data-t="Label PCA"><div class="it"><b>PCA L2</b><span><i class="up"></i>+0.1 to +1.7%</span></div></div>
    <div class="rh r1">L1<small>norm</small></div>
    <div class="cell r1" data-t="Time domain"><div class="it"><b>Time L1</b><span><i class="dn"></i>−2.0 to −5.4%</span></div><div class="it"><b>Huber</b><span><i class="dn"></i>−1.5 to −5.3%</span></div><div class="it"><b>Window L<sub>2,1</sub></b><span><i class="dn"></i>−1.2 to −4.7%</span></div></div>
    <div class="cell r1 pub" data-t="Fourier"><div class="it"><b>Frequency L1</b><span><i class="dn"></i>−4.0 to −4.6%</span></div><div class="it"><b>FreDF ★</b><span><i class="dn"></i>−2.7 to −4.5%</span></div></div>
    <div class="cell r1 pub" data-t="Label PCA"><div class="it"><b>PCA L1</b><span><i class="dn"></i>−0.6 to −4.1%</span></div><div class="it"><b>Time-o1 ★</b><span><i class="dn"></i>−0.6 to −4.2%</span></div></div>
  </div>
  <p class="hl-mxcap">Change in test MSE against the same official script trained with the MSE, as the range over ETTh1, ETTh2, ETTm1 and Weather (standard split, 4 horizons × 3 seeds; FFT L2 on ETTh1 and ETTh2 only). <i class="dn"></i>lower error on every dataset <i class="up"></i>higher error on every dataset. ★ Published losses, framed: in their official settings both are L1 losses in a rotated basis.</p>
</section>

<div class="hl-stats">
  <div class="hl-stat" style="--accent:var(--muted)"><div class="num">0 <small>of 3</small></div><div class="lab">L2 losses in a Fourier or PCA basis improve on the MSE (0.0% to +3.9%)</div></div>
  <div class="hl-stat" style="--accent:var(--c-b)"><div class="num">−4.3%</div><div class="lab">test MSE with frequency L1 alone, mean of four datasets; FreDF itself reaches −3.9%</div></div>
  <div class="hl-stat" style="--accent:var(--c-w)"><div class="num">−14.3 → −0.5%</div><div class="lab">time-L1 gain of DLinear on ETTh2 before and after adding RevIN</div></div>
  <div class="hl-stat" style="--accent:var(--c-good)"><div class="num">34</div><div class="lab">predictions recorded before their runs: 14 held, 8 partly, 12 failed</div></div>
</div>

<h2 id="the-question">The question</h2>

<p class="hl-lead">A <strong>direct multi-step forecaster</strong> predicts the next H values of a series from a lookback window in one pass, and is usually trained with the MSE over the horizon. A recent line of work argues that this objective is biased because it ignores the autocorrelation among the H future values. <strong>FreDF</strong> aligns forecasts and labels in the frequency domain, <strong>Time-o1</strong> aligns them in a principal-component basis of the labels, and <strong>QDF</strong> learns a quadratic weight matrix. All three report consistent gains over the MSE.</p>

<p class="hl-lead">The motivation conflicts with a basic property of the squared error: its minimizer is the conditional mean, however the coordinates of the target are correlated. Yet the gains are real. When we re-run the official code, FreDF improves on the same script trained with the MSE by 4.1–4.5% on the standard split of ETTh1, ETTh2 and ETTm1, and by 2.0–5.4% on average over five rolling origins of the four ETT datasets. So what do these losses change, and when do they help?</p>

<div class="hl-grid3">
  <div class="hl-card" style="--accent:var(--c-b)"><div class="h"><i></i>Official code, one change at a time</div>The FreDF, Time-o1 and QDF repositories at pinned commits, with their scripts and hyper-parameters. Our patches only add options: loss variants, rolling borders and normalization switches. Every change is reported against the same script trained with the MSE.</div>
  <div class="hl-card" style="--accent:var(--c-w)"><div class="h"><i></i>Models and data</div>iTransformer and DLinear at a lookback of 96 on ETTh1/2, ETTm1/2, Weather, electricity, traffic and (DLinear) exchange rates; Fredformer and FreTS for Time-o1, TQNet for QDF, and PatchTST and a lookback of 336 as checks. Horizons 96 to 720, three to five seeds, about 1,180 GPU job-hours.</div>
  <div class="hl-card" style="--accent:var(--c-good)"><div class="h"><i></i>Evaluation</div>Five rolling origins per dataset, model selection on validation MSE only, and Diebold–Mariano tests on per-window errors whose size we checked by simulation (5–7.5% at the nominal 5%), with Holm correction.</div>
</div>

<p class="hl-k">Terms used below</p>
<dl class="hl-terms">
  <div><dt>Change vs. MSE.</dt> <dd>Paired relative change in test MSE against the same script trained with the MSE, averaged over horizons and seeds. Negative is better.</dd></div>
  <div><dt>Transform and norm.</dt> <dd>A loss applies a linear map T to the error e = ŷ − y and then a norm. The MSE is T = I with a squared L2 norm.</dd></div>
  <div><dt>Frequency L1.</dt> <dd>The L1 norm of the real FFT of the error. FreDF is this loss plus a small MSE term (weight 0.1 to 0.4).</dd></div>
  <div><dt>Instance normalization.</dt> <dd>Subtracting each input window's own mean (and scale) before the model and adding it back to the forecast: RevIN, the built-in normalization of iTransformer, or centring a linear model.</dd></div>
  <div><dt>Forecast level (DC).</dt> <dd>The mean of the forecast over the horizon, which is the zero-frequency coefficient of its FFT.</dd></div>
  <div><dt>Level offset.</dt> <dd>How far the mean of a period lies from the training mean, in training standard deviations, averaged over channels.</dd></div>
  <div><dt>Rolling origins.</dt> <dd>Five versions of the train/validation/test borders, each shifted by a further 30 days on ETT, so results do not rest on one test period. The origins overlap and are not independent replicates.</dd></div>
  <div><dt>Recorded prediction.</dt> <dd>An expected result written into the version-control history before the corresponding runs started.</dd></div>
</dl>

<h2 id="what-the-theory-says">What the theory says</h2>

<p class="hl-lead">That quadratic losses share the minimizer of the MSE is classical, and An et al. (2026) already called FreDF's decorrelation argument inaccurate. We extend the equivalence to weighted losses such as QDF and to misspecified models, and it leaves exactly one ingredient that can matter.</p>

<div class="hl-grid3">
  <div class="hl-card"><div class="h"><span>1</span>Weighting cannot remove a bias</div>For any positive definite weight matrix Ω, the risk of a forecast f splits into a noise term and a distance to the conditional mean m:
    <div class="eq">\(\mathbb{E}\lVert y-f\rVert_\Omega^2=\mathbb{E}\lVert\varepsilon\rVert_\Omega^2+\mathbb{E}\lVert m-f\rVert_\Omega^2\)</div>
    Correlated labels change the first term, not the minimizer. If every horizon step uses the same function class, the solution is the same for every Ω, also when the model is misspecified.<span class="ref">Propositions 1–2</span></div>
  <div class="hl-card"><div class="h"><span>2</span>A rotation changes nothing under L2</div>For an orthonormal transform T, such as the full FFT or a full-rank PCA,
    <div class="eq">\(\lVert Te\rVert_2^2=\lVert e\rVert_2^2\)</div>
    so these losses <em>are</em> the MSE. With a shared design, weighted least squares returns the ordinary least-squares solution for every Ω, and weighting only acts as a separate ridge penalty per output direction, with signal-to-noise ratios as the optimal weights.<span class="ref">Theorem 1, Corollary 1</span></div>
  <div class="hl-card" style="--accent:var(--c-b)"><div class="h"><span>3</span>What remains is the norm</div>An L1 norm is not invariant to rotation:
    <div class="eq">\(\lVert Te\rVert_1\neq\lVert e\rVert_1\)</div>
    Under symmetric noise it still targets the conditional mean, and in finite samples it bounds the influence of large residuals. The transforms of FreDF and Time-o1 can therefore matter only through re-weighting or through the L1 norm.<span class="ref">Proposition 3</span></div>
</div>

<h2 id="findings">Findings</h2>

<p class="hl-k">Finding 1</p>
<h3 class="hl-h" id="l2-is-the-mse">L2 in any basis is the MSE; the L1 norm carries the gain</h3>
<p class="hl-lead">We swap the norm while keeping the transform, inside the official scripts. The full-spectrum <strong>FFT L2 coincides with the MSE</strong>. PCA L2 and the half-spectrum rFFT L2 used in the FreDF code are slightly <em>worse</em>, although their minimizer is unchanged. Replacing the L2 norm by an L1 norm recovers the full gain: frequency L1 alone matches FreDF, and PCA L1 without Time-o1's standardization matches Time-o1. Plain time-domain L1 and Huber help too.</p>
<figure class="hl-fig" id="hl-dec" style="--accent:var(--c-b)">
  <div class="hl-ctl"><label><input type="checkbox" id="hl-dec-more"> Add electricity and traffic (iTransformer, six losses)</label></div>
  <div class="hl-legend" id="hl-dec-legend"></div>
  <div class="hl-scroll"><div class="hl-chart" id="hl-dec-chart" style="min-width:600px"></div></div>
  <figcaption><strong>Decomposition of FreDF and Time-o1</strong> (paper Figure 2; standard split, mean ± SEM over 4 horizons × 3 seeds). Bars and the bottom row give the mean over ETTh1, ETTh2, ETTm1 and Weather. PCA losses and Time-o1 are relative to the Time-o1 script (Fredformer on ETT, FreTS on Weather), all others to the FreDF script (iTransformer). Hover a column for the values.
    <span class="take">On these four datasets no L2 variant improves on the MSE and every L1-type loss does. On electricity and traffic FFT L2 stays within 0.2% of the MSE, while the L1 gain shrinks on electricity (−2.0% and −0.6% for frequency and time L1) and turns into a loss on traffic (+0.4% and +2.0%).</span></figcaption>
</figure>
<div class="hl-grid2">
  <div class="hl-card ev" style="--accent:var(--c-w)"><div class="num">+0.3 to +3.9%</div><div class="h">Why the rFFT L2 is worse</div>The half spectrum weights the DC direction, the mean level of the forecast, twice. Most of its extra error lies along that direction, and doubling only the DC weight of the MSE raises the test MSE by 0.3–3.9%.</div>
  <div class="hl-card ev" style="--accent:var(--c-b)"><div class="num">−2.9% after tuning</div><div class="h">A tuned MSE does not catch up</div>With the same validation-based budget for both losses, tuned FreDF still beats tuned MSE by 2.9% and 2.8% on ETTh1 and ETTh2, at every rolling origin.</div>
</div>

<p class="hl-k">Finding 2</p>
<h3 class="hl-h" id="normalization">The size of the gain depends on instance normalization</h3>
<p class="hl-lead">Under rolling origins, DLinear gains far more from L1 than iTransformer on ETTh2 and ETTm2. DLinear does not normalize its inputs; iTransformer does. So we varied only the normalization and kept each model otherwise fixed. <strong>Adding RevIN to DLinear almost removes its large gains</strong>, and MSE-trained DLinear with RevIN matches L1-trained DLinear without it (test MSE 0.386 vs. 0.388 on ETTh2). Switching off iTransformer's built-in normalization does the opposite.</p>
<figure class="hl-fig" id="hl-norm" style="--accent:var(--c-w)">
  <div class="hl-ctl"><span>Loss</span><span class="hl-seg" id="hl-norm-seg"></span></div>
  <div class="hl-legend" id="hl-norm-legend"></div>
  <div class="hl-scroll"><div class="hl-chart" id="hl-norm-chart" style="min-width:600px"></div></div>
  <figcaption><strong>L1 gain without and with instance normalization</strong> (paper Figure 3; mean over 5 origins × 4 horizons × seeds, lookback 96). The linear model is the same model with or without lookback-mean centring; DLinear is official vs. RevIN; iTransformer is its built-in normalization switched off vs. official.
    <span class="take">Without normalization the gain reaches 6–18% on ETTh2 and ETTm2 in all three models; with it, at most 5.5%. At a lookback of 336 DLinear behaves the same, but iTransformer gains 4–15% with its normalization, so for iTransformer this holds at a lookback of 96 only.</span></figcaption>
</figure>

<p class="hl-k">Finding 3</p>
<h3 class="hl-h" id="largest-errors">Without normalization, the L1 norm discounts a few windows with large level errors</h3>
<p class="hl-lead">A squared loss lets the windows with the largest errors dominate the fit; an L1 loss bounds their influence. The toy calculation below shows the size of that difference when a tenth of the windows have errors several times larger than the rest.</p>
<figure class="hl-fig" id="hl-toy" style="--accent:var(--c-b)">
  <div class="hl-toy">
    <div class="sl"><span>Errors of the worst 10% of windows are</span><input type="range" id="hl-toy-c" min="1" max="6" step="0.1" value="3.1" aria-label="Error ratio of the worst 10% of windows"><b id="hl-toy-v">3.1×</b><span>the rest</span></div>
    <div class="row"><span>Squared loss (MSE)</span><div class="bar" id="hl-toy-l2"></div></div>
    <div class="row"><span>Absolute loss (L1)</span><div class="bar" id="hl-toy-l1"></div></div>
    <div class="cap" id="hl-toy-cap"></div>
  </div>
  <figcaption><strong>Toy calculation, not an experiment.</strong> Share of the training loss held by the worst 10% of windows if their errors are c times those of the other 90%: 0.1c² / (0.1c² + 0.9) for the squared loss and 0.1c / (0.1c + 0.9) for the absolute loss.
    <span class="take">Measured: on ETTh2 and ETTm2 the 10% of training windows with the largest errors hold 51% and 56% of DLinear's squared training error. A concentrated error is not sufficient, though. The share is as high with RevIN (52% and 59%), where the L1 norm gains almost nothing, and the tails of the training level errors do not predict the gain across settings (ρ = −0.19).</span></figcaption>
</figure>
<p class="hl-lead">What decides is which windows these are. On ETTh2 and ETTm2, MSE-trained DLinear is <em>better</em> than its L1 counterpart on the 10% of test windows with the largest level offset (−11.6% and −22.4%) but worse on the remaining 90% (+22.5% and +19.1%). On the training windows the L1 fit is worse where the offset is largest (+16% and +21%) and about equal elsewhere, so the L1 advantage appears only out of sample. Interventions on the training loss single out the same windows:</p>
<figure class="hl-fig" id="hl-itv" style="--accent:var(--c-b)">
  <div class="hl-ctl"><span>Model</span><span class="hl-seg" id="hl-itv-model"></span><span class="hl-chips" id="hl-itv-ds"></span></div>
  <div class="hl-scroll"><div class="hl-chart" id="hl-itv-chart" style="min-width:560px"></div></div>
  <figcaption><strong>Which change to the squared loss reproduces the L1 gain?</strong> Change in test MSE against the plain MSE (rolling origins; Weather on the standard split). The vertical line marks time L1. Linear model: DLinear-like, without centring. iTransformer keeps its built-in normalization.
    <span class="take" id="hl-itv-take"></span></figcaption>
</figure>
<div class="hl-grid2">
  <div class="hl-card ev" style="--accent:var(--c-b)"><div class="num">85% vs. 20%</div><div class="h">The discounted errors are level errors</div>In the linear model, an L1 norm on the DC coefficient of the horizon alone, with a running-scaled square on the other frequencies, recovers 85% of the gap between a frequency-balanced L2 loss and frequency L1. An L1 norm on every coefficient except the DC recovers 20%.</div>
  <div class="hl-card ev" style="--accent:var(--c-w)"><div class="num">+9.7% with a shift</div><div class="h">A test-period shift alone does not produce the gain</div>In a seasonal AR(1) simulation with week-long level excursions in the training period, time L1 gains 2.9–3.5% when the test period is not shifted and loses up to 9.7% when it is, because the excursion windows it discounts then resemble the test period. Our predictions assumed the opposite and failed, so the correlation between drift and gain across ETT origins (ρ = −0.88) is not causal evidence.</div>
</div>

<p class="hl-k">Finding 4</p>
<h3 class="hl-h" id="forecast-level">With normalization, a smaller gain remains, again in the forecast level</h3>
<p class="hl-lead">L1-type losses still improve iTransformer at every origin, by 1.3–5.5% on average, and the gain survives tuning and oracle stopping. Trimming recovers 82% of the time-L1 gain over 21 dataset × origin cells, but an L2 loss balanced per frequency recovers as much (86%), so those two interventions cannot tell a robust-estimation account from an optimization account. Restricting the L1 norm to the forecast level separates them.</p>
<div class="hl-grid3">
  <div class="hl-card ev" style="--accent:var(--c-b)"><div class="num">57–62%</div><div class="h">of the frequency-L1 gain lies along DC</div>On the standard split, more than half of the gain is a smaller error in the mean level of the forecast, in all four datasets.</div>
  <div class="hl-card ev" style="--accent:var(--c-b)"><div class="num">73–109%</div><div class="h">recovered by L1 on the DC coefficient alone</div>With a running-scaled square on all other frequencies, an L1 norm on the forecast level reaches most of the frequency-L1 gain on the four ETT datasets, as predicted before the runs.</div>
  <div class="hl-card ev" style="--accent:var(--muted)"><div class="num">none</div><div class="h">recovered by L1 everywhere except DC</div>The complementary loss, L1 on every frequency but the DC, recovers nothing. At a lookback of 336 the DC-only loss recovers all of the gap.</div>
</div>

<p class="hl-k">Finding 5</p>
<h3 class="hl-h" id="rolling-origins">Under rolling origins the gains are stable, the best basis is not</h3>
<p class="hl-lead">On iTransformer, all four L1-type losses improve on the MSE at every origin of every ETT dataset, and no origin × horizon cell is significantly worse. The basis gives no consistent advantage: frequency L1 is 0.1–1.5% better than time L1 on iTransformer, but 0.8–4.4% worse on DLinear, on every dataset and at every origin.</p>
<figure class="hl-fig" style="--accent:var(--c-b)">
  <div class="hl-tablewrap"><table class="hl-table hl-heat" id="hl-roll"></table></div>
  <figcaption><strong>Change in test MSE against the MSE, mean over five rolling origins</strong> (paper Table 2; 4 horizons, 3 to 5 seeds, lookback 96). Small print: origins with a negative mean, and origin × horizon cells that are significantly better / worse after Holm correction (of 20 per dataset). The last row of each block compares frequency L1 with time L1.</figcaption>
</figure>
<p class="hl-lead">Can the loss be chosen on validation data instead? On iTransformer the loss with the lowest validation MSE beats the MSE in 80–100% of the origin × horizon cells. On DLinear it does so in only 40% (ETTh1) and 60% (ETTm1) of them. The failure follows the season: validation periods in summer and autumn favour the MSE, the winter test periods favour L1.</p>
<figure class="hl-fig" style="--accent:var(--c-w)">
  <div class="img"><img src="/images/projects/2026-11-horizon-aware-losses/fig4-selection.png" alt="Paper Figure 4: validation versus test change of L1 on ETTh1 across origins, and how often validation-based choices beat the MSE." loading="lazy"></div>
  <figcaption><strong>Selecting the loss on validation data</strong> (paper Figure 4). (a) Validation (dashed) and test (solid) change of L1 against the MSE on ETTh1 across origins. (b, c) How often each rule's choice beats the MSE on test.
    <span class="take">Neither pooling earlier windows nor a same-season validation block fixes the choice. Always using time Huber beats the MSE in 90–100% of the DLinear cells, which, judged on test results, only justifies a default.</span></figcaption>
</figure>

<h2 id="what-did-not-hold">What did not hold</h2>

<p class="hl-lead">The theory predicted the first finding before any run. The roles of normalization, of the largest training errors and of the forecast level emerged from the experiments and were then tested by predictions written down in advance. Several of our own readings were overturned along the way.</p>

<div class="hl-grid2">
  <div class="hl-card ev" style="--accent:var(--c-bad)"><div class="num">6.8% → 1.1%</div><div class="h">A missing-value code inflated the Weather gains</div>The training period of Weather contains the code −9999 in 81 cells of three channels. Interpolating them shrinks iTransformer's time-L1 gain from 6.8% to 1.1%, while the linear models change by less than 0.9 points. iTransformer's Weather gains largely reflect this code.</div>
  <div class="hl-card ev" style="--accent:var(--c-bad)"><div class="num">+1.5 to +35%</div><div class="h">Frequency L1 hurts PatchTST</div>On PatchTST, FFT L2 again coincides with the MSE and time L1 is better on four of five datasets. Frequency L1 is worse on all five, also after equal-budget tuning. The basis that suits the L1 norm depends on the backbone.</div>
  <div class="hl-card ev" style="--accent:var(--c-bad)"><div class="num">−2.2%, not −10%</div><div class="h">More drift did not mean more gain</div>Exchange rates drift more than ETT, and we predicted an L1 gain of 10% or more for DLinear. It gains 2.2% (time L1) and 7.2% (frequency L1), and trimming does not reproduce the gain there. The account of Finding 3 is established on ETT only.</div>
  <div class="hl-card ev" style="--accent:var(--c-bad)"><div class="num">0.4–1.8%</div><div class="h">QDF's learned weighting is not irrelevant</div>We predicted that QDF's learned Ω would do no better than an identity or a random Ω. On TQNet it beats the MSE by 0.4–1.8% while those two do not. An absolute value with Ω = I helps as much on ETT and Weather. The learned weights down-weight the forecast level like GLS weights, where our theory predicts a loss for a linear head, which points to an effect on the learned representation.</div>
</div>

<p class="hl-k" id="predictions">Recorded predictions</p>
<h3 class="hl-h">All 34 predictions and their outcomes</h3>
<p class="hl-lead">Each row was committed to the version-control history before the corresponding runs. Round 0 is the initial analysis; rounds 1 to 5 followed five rounds of revision, each after the earlier results were known. The outcome column is the paper's; the grouping into held, partly and failed is ours for this page.</p>
<figure class="hl-fig" style="--accent:var(--c-good)">
  <div class="hl-score" id="hl-score" role="img" aria-label="14 predictions held, 8 partly held, 12 failed"></div>
  <div class="hl-ctl"><span class="hl-chips" id="hl-pred-chips"></span></div>
  <div class="hl-tablewrap"><table class="hl-table hl-pred" id="hl-pred"></table></div>
  <figcaption><strong>Predictions and outcomes</strong> (paper Appendix, Table of recorded predictions). DL: DLinear without normalization; LM: DLinear-like linear model; iTr: iTransformer with its normalization.</figcaption>
</figure>

<h2 id="in-practice">In practice</h2>

<p class="hl-lead">For iTransformer and DLinear at a lookback of 96, where most of our runs are:</p>

<div class="hl-grid3">
  <div class="hl-card"><div class="h"><span>1</span>Decide the normalization first</div>Whether each input window is normalized or centred matters more than the loss. MSE-trained DLinear with RevIN matches L1-trained DLinear without it.</div>
  <div class="hl-card"><div class="h"><span>2</span>Then consider a robust loss</div>With normalization, robust losses still give modest gains on iTransformer and small or none on DLinear. Without it, a robust loss such as time Huber is the safer default.</div>
  <div class="hl-card"><div class="h"><span>3</span>Check the data and the backbone</div>Look for coded missing values before crediting a loss. The gains do not transfer unchanged to iTransformer at a lookback of 336, to frequency L1 on PatchTST, or to a test period that resembles the discounted training windows.</div>
</div>

<p class="hl-scope">Scope: the ETT family, Weather, electricity and traffic under rolling origins with iTransformer and DLinear; exchange rates with DLinear only; Fredformer and FreTS in the Time-o1 decomposition; PatchTST and a lookback of 336 on the standard split only. The link between L1 gain and level drift is a correlation across overlapping ETT origins. The interventions on deep models ran on ETT and Weather only, and the account of QDF on TQNet is descriptive. Open question: MSE-trained iTransformer stays 0–8% above the exact linear MSE solution on the same windows, while FreDF closes that gap; we do not know why.</p>

<h2 id="links-and-citation">Links and citation</h2>

<div class="hl-links">
  <div class="hl-link"><div class="h">Paper</div><div class="s">Manuscript in preparation; the PDF and BibTeX will be linked here when they are public.</div></div>
  <div class="hl-link"><div class="h">Code and run records</div><div class="s">Patches to the official repositories, configurations, per-run results and the dated prediction records will be released with the paper.</div></div>
</div>

</div>

<script src="/files/projects/2026-11-horizon-aware-losses/explore.js" defer></script>
