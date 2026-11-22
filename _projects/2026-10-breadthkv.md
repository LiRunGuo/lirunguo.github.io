---
title: "Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning"
collection: projects
permalink: /projects/2026-10-breadthkv/
date: 2026-10-05
status: "arXiv preprint"
venue: "arXiv"
excerpt: "Under a fixed KV-cache byte budget for long chain-of-thought decoding, keeping several times as many tokens at 2–4 bits beats evicting tokens at full precision in 17 of 18 settings, mostly because far fewer runs reason until the length cap without answering."
teaser: "/images/projects/2026-10-breadthkv/teaser.png"
tags: ["KV cache compression", "reasoning models", "quantization", "token eviction"]
paperurl: "https://arxiv.org/abs/2610.05685"
pdfurl: "/files/breadthkv-kv-compression.pdf"
bibtexurl: "/files/bibtex/li2026breadthkv.bib"
publication: "/publication/2026-10-breadthkv"
citation: 'Runguo Li. (2026). &quot;Spend Bytes on Breadth: Precision–Count Trade-offs for Decode-Time KV Compression in Long Chain-of-Thought Reasoning.&quot; arXiv:2610.05685.'
---

<style>
/* Page-scoped styles: everything is under .bkv so no other page changes. Blue is eviction, orange is the low-bit hybrid. */
.bkv { --c-ev: #2a78d6; --c-hy: #eb6834; --c-jt: #1baf7a; --c-ref: #6f6e69; --c-win: #2b2b29;
  --ink: #1d1d1b; --ink2: #52514e; --muted: #898781; --line: rgba(11, 11, 11, 0.09); --card: #ffffff; --soft: #f6f6f3; --empty: #e9e8e3;
  --heat: 235, 104, 52; --shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 10px 30px rgba(11, 11, 11, 0.06); color: var(--ink); }
html[data-theme="dark"] .bkv { --c-ev: #3987e5; --c-hy: #d95926; --c-jt: #199e70; --c-ref: #a9a8a1; --c-win: #e6e5e0;
  --ink: #f3f3f1; --ink2: #d0cfc8; --muted: #a9a8a1; --line: rgba(255, 255, 255, 0.10); --card: #3b3b3b; --soft: #404040; --empty: #4d4d4b;
  --heat: 217, 89, 38; --shadow: 0 1px 2px rgba(0, 0, 0, 0.25), 0 10px 30px rgba(0, 0, 0, 0.2); }
/* the layout's page title must not run under the fixed table of contents */
@media screen and (min-width: 1200px) { .page__title { max-width: calc(100% - 250px); } }
.bkv p { margin: 0 0 0.9em; }
.bkv sup.dg { font-size: 0.7em; color: var(--muted); }

.bkv-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 28px 28px 24px; margin: 0.4em 0 1.4em; border: 1px solid var(--line);
  background: radial-gradient(70% 120% at 0% 0%, rgba(42, 120, 214, 0.12), transparent 60%), radial-gradient(60% 120% at 100% 0%, rgba(235, 104, 52, 0.13), transparent 60%), var(--card);
  box-shadow: var(--shadow); }
.bkv-hero::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: linear-gradient(90deg, var(--c-ev) 0%, var(--c-ev) 21%, var(--c-hy) 21%, var(--c-hy) 100%); }
.bkv-kicker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.72em; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin-bottom: 10px; }
.bkv-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; background: var(--ink); color: var(--card); font-weight: 700; letter-spacing: 0.04em; }
.bkv .bkv-name { font-size: 2.6em; line-height: 1.05; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 6px; }
.bkv .bkv-name span { background: linear-gradient(90deg, var(--c-ev), var(--c-hy) 55%); -webkit-background-clip: text; background-clip: text; color: transparent; }
.bkv .bkv-tagline { font-size: 1.25em; line-height: 1.35; font-weight: 600; margin: 0 0 14px; }
.bkv .bkv-tagline em { font-style: normal; color: var(--c-hy); }
.bkv .bkv-lede { font-size: 0.95em; color: var(--ink2); max-width: 46em; }
.bkv .bkv-byline { font-size: 0.85em; color: var(--ink2); margin: 0; }
.bkv-btns { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 20px; }
.bkv .bkv-btn { display: inline-flex; align-items: center; gap: 7px; padding: 7px 15px; border-radius: 999px; font-size: 0.85em; font-weight: 600; text-decoration: none; border: 1px solid var(--ink); }
.bkv .bkv-btn.primary { background: var(--ink); color: var(--card); }
.bkv .bkv-btn.ghost { background: transparent; color: var(--ink); }
.bkv-same { display: grid; gap: 9px; padding: 14px 16px 12px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); }
.bkv-same .row { display: grid; grid-template-columns: 11.5em 1fr; align-items: center; gap: 10px; font-size: 0.82em; }
.bkv-same .row b { font-weight: 600; }
.bkv-same .bar { display: flex; align-items: center; gap: 2px; height: 26px; }
.bkv-same .bar i { display: flex; align-items: center; height: 100%; padding-left: 8px; border-radius: 5px; color: #fff; font-style: normal; font-size: 0.86em; white-space: nowrap; overflow: hidden; }
.bkv-same .bar span { padding-left: 8px; color: var(--ink2); white-space: nowrap; }
.bkv-same .cap { font-size: 0.75em; color: var(--muted); }

.bkv-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 1.6em; }
.bkv-stat { position: relative; padding: 14px 14px 12px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }
.bkv-stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent, var(--c-hy)); }
.bkv-stat .num { font-size: 1.9em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; white-space: nowrap; }
.bkv-stat .num small { font-size: 0.5em; font-weight: 700; letter-spacing: 0; }
.bkv-stat .lab { font-size: 0.78em; color: var(--ink2); line-height: 1.35; margin-top: 4px; }

.bkv .bkv-k { font-size: 0.7em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 1.8em 0 0.1em; }
.bkv h3.bkv-h { font-size: 1.1em; margin: 0 0 0.55em; }
.bkv .bkv-lead { font-size: 0.95em; line-height: 1.6; color: var(--ink2); margin: 0 0 0.9em; max-width: 52em; }
.bkv .bkv-lead strong { color: var(--ink); }
.bkv-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.bkv-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.bkv-card { position: relative; padding: 14px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.bkv-card .h { display: flex; align-items: center; gap: 8px; font-weight: 700; color: var(--ink); margin-bottom: 4px; }
.bkv-card .h i { flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--accent, var(--ink)); }
.bkv-card .h span { flex: none; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.bkv-card .num { font-size: 1.7em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; color: var(--ink); white-space: nowrap; }
.bkv dl.bkv-terms { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 28px; margin: 0.2em 0 1.4em; padding: 8px 18px; border-radius: 14px; background: var(--soft); border: 1px solid var(--line); font-size: 0.84em; line-height: 1.5; }
.bkv-terms > div { padding: 8px 0; border-bottom: 1px solid var(--line); }
.bkv-terms > div:nth-last-child(-n+2) { border-bottom: 0; }
.bkv .bkv-terms dt { display: inline; font-size: 1em; font-weight: 700; color: var(--ink); margin: 0; }
.bkv .bkv-terms dd { display: inline; font-size: 1em; margin: 0; color: var(--ink2); }

.bkv .bkv-fig { display: block; margin: 0 0 1.2em; padding: 14px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); }
.bkv-fig .img { background: #fff; border-radius: 10px; padding: 6px; }
.bkv-fig img { display: block; width: 100%; height: auto; margin: 0 auto; }
.bkv-fig .img a { display: block; }
.bkv-fig.narrow .img img { max-width: 440px; }
.bkv .bkv-fig figcaption { font-family: inherit; font-size: 0.82em; color: var(--ink2); line-height: 1.5; margin: 10px 4px 0; }
.bkv-fig figcaption strong { color: var(--ink); }
.bkv-fig .take { display: block; margin-top: 6px; padding: 7px 10px; border-radius: 8px; background: var(--soft); color: var(--ink); border-left: 3px solid var(--accent, var(--c-hy)); }
.bkv-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
.bkv-step { padding: 12px 14px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.bkv-step .t { font-weight: 700; color: var(--ink); display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.bkv-step .t span { display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.bkv .bkv-eq { margin: 12px 4px 0; text-align: center; font-size: 1em; font-variant-numeric: tabular-nums; }
.bkv .bkv-eq small { display: block; font-size: 0.78em; color: var(--muted); margin-top: 3px; }

/* controls shared by the two interactive figures */
.bkv-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin: 0 2px 12px; font-size: 0.82em; color: var(--ink2); }
.bkv-ctl > span { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 6px; }
.bkv-seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.bkv .bkv-seg button { border: 0; border-radius: 0; background: transparent; color: var(--ink2); font: inherit; font-size: 1em; padding: 4px 10px; cursor: pointer; margin: 0; }
.bkv .bkv-seg button + button { border-left: 1px solid var(--line); }
.bkv .bkv-seg button[aria-pressed="true"] { background: var(--ink); color: var(--card); font-weight: 600; }
.bkv .bkv-seg button:disabled { color: var(--muted); opacity: 0.55; cursor: not-allowed; text-decoration: line-through; }
.bkv .bkv-seg button:focus-visible { outline: 2px solid var(--c-ev); outline-offset: -2px; }

/* split-the-budget figure */
.bkv-split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.05fr); gap: 18px; align-items: start; }
.bkv-ct { font-size: 0.82em; font-weight: 600; margin: 0 0 8px; }
.bkv-ct span { font-weight: 400; color: var(--ink2); }
.bkv-grid { display: grid; grid-template-columns: repeat(32, minmax(0, 1fr)); gap: 2px; }
.bkv-grid i { aspect-ratio: 1 / 1; border-radius: 2px; background: var(--empty); }
.bkv-grid i.e { background: var(--c-ev); }
.bkv-grid i.q { background: var(--c-hy); }
.bkv-grid i.w { background: var(--c-win); }
.bkv .bkv-gridcap { font-size: 0.8em; color: var(--ink2); line-height: 1.5; margin: 8px 0 0; }
.bkv-gridcap b { color: var(--ink); }
.bkv-legend { display: flex; flex-wrap: wrap; gap: 3px 14px; font-size: 0.74em; color: var(--ink2); margin: 8px 0 0; }
.bkv-legend span { display: inline-flex; align-items: center; gap: 5px; }
.bkv-legend i { width: 10px; height: 10px; border-radius: 2px; background: var(--empty); }
.bkv-legend i.e { background: var(--c-ev); } .bkv-legend i.q { background: var(--c-hy); } .bkv-legend i.w { background: var(--c-win); }
.bkv-legend i.tick { width: 3px; height: 13px; border-radius: 1px; background: var(--c-win); }
.bkv-legend i.tick.ev { background: var(--c-ev); }
.bkv-meter { display: grid; grid-template-columns: minmax(0, 1fr) 4.2em; gap: 3px 10px; align-items: center; margin: 0 0 11px; font-size: 0.8em; }
.bkv-meter .lab { grid-column: 1 / -1; color: var(--ink2); }
.bkv-meter .track { position: relative; height: 14px; border-radius: 4px; background: var(--soft); }
.bkv-meter .track i { display: block; height: 100%; border-radius: 0 4px 4px 0; transition: width 0.25s ease; }
.bkv-meter .track i.e { background: var(--c-ev); }
.bkv-meter .track i.q { background: var(--c-hy); }
.bkv-meter .ref { position: absolute; top: -3px; bottom: -3px; width: 3px; margin-left: -1.5px; border-radius: 1px; background: var(--c-win); box-shadow: 0 0 0 1.5px var(--card); }
.bkv-meter .ref.ev { background: var(--c-ev); }
.bkv-meter .val { text-align: right; font-weight: 700; font-variant-numeric: tabular-nums; }

/* tables */
.bkv-tablewrap { overflow-x: auto; }
.bkv .bkv-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.82em; margin: 0; display: table; border: 0; background: transparent; }
.bkv .bkv-table thead, .bkv .bkv-table tr { background: transparent; }
.bkv .bkv-table th, .bkv .bkv-table td { border: 0; border-bottom: 1px solid var(--line); background: transparent; padding: 8px 10px; white-space: nowrap; }
.bkv .bkv-table th { text-align: left; font-weight: 600; color: var(--ink2); }
.bkv .bkv-table td { vertical-align: middle; font-variant-numeric: tabular-nums; }
.bkv .bkv-table .r { text-align: right; }
.bkv .bkv-table .c { text-align: center; }
.bkv .bkv-table tr.hl td { background: rgba(var(--heat), 0.09); font-weight: 600; }
.bkv .bkv-table tr.hl td:first-child { box-shadow: inset 3px 0 0 var(--c-hy); }
.bkv .bkv-table tr.grp td { padding: 12px 10px 4px; font-size: 0.86em; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); border-bottom: 1px solid var(--line); }
.bkv .bkv-table tr.sep td { border-top: 2px solid var(--line); }
.bkv-table .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; background: var(--c-ref); }
.bkv-table .dot.ev { background: var(--c-ev); } .bkv-table .dot.hy { background: var(--c-hy); } .bkv-table .dot.jt { background: var(--c-jt); }
.bkv-table .tag { display: inline-block; margin-left: 6px; padding: 1px 7px; border-radius: 999px; background: var(--c-hy); color: #fff; font-size: 0.82em; font-weight: 700; }
.bkv-table .cell { display: flex; align-items: center; gap: 8px; }
.bkv-table .bar { flex: none; width: 96px; height: 8px; border-radius: 0 4px 4px 0; background: var(--soft); overflow: hidden; }
.bkv-table .bar i { display: block; height: 100%; border-radius: 0 4px 4px 0; background: var(--c-ref); }
.bkv-table .bar i.ev { background: var(--c-ev); } .bkv-table .bar i.hy { background: var(--c-hy); } .bkv-table .bar i.jt { background: var(--c-jt); }
.bkv-table .bar i.cap { background: var(--muted); }
.bkv .bkv-table td.heat { text-align: center; border-left: 2px solid var(--card); }
.bkv-table .ci { color: var(--muted); font-weight: 400; }
.bkv .bkv-note { font-size: 0.78em; color: var(--muted); margin: 8px 4px 0; min-height: 1.2em; }
.bkv-pair { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; align-items: center; }

.bkv .bkv-scope { font-size: 0.8em; color: var(--muted); margin: 0 0 1.2em; }
.bkv-links { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 1em; }
.bkv-bib { position: relative; margin: 0.6em 0 0; }
.bkv .bkv-bib pre { margin: 0; border-radius: 12px; font-size: 0.72em; max-width: 100%; overflow-x: auto; padding: 1em 1.1em; background: var(--soft); border: 1px solid var(--line); color: var(--ink); }
.bkv .bkv-copy { position: absolute; top: 0.55em; right: 0.55em; margin: 0; font: inherit; font-size: 0.72em; padding: 0.25em 0.8em; border-radius: 999px; border: 1px solid var(--line); background: var(--card); color: var(--ink); cursor: pointer; }

@media (max-width: 900px) { .bkv-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } .bkv-grid3, .bkv-steps { grid-template-columns: 1fr; } .bkv-split, .bkv-pair { grid-template-columns: 1fr; } }
@media (max-width: 640px) { .bkv-hero { padding: 22px 16px 18px; } .bkv .bkv-name { font-size: 2.1em; } .bkv-grid2, .bkv dl.bkv-terms { grid-template-columns: 1fr; } .bkv-terms > div:nth-last-child(2) { border-bottom: 1px solid var(--line); }
  .bkv-same .row { grid-template-columns: 1fr; gap: 3px; } .bkv-table .bar { width: 56px; } .bkv-stat .num { font-size: 1.5em; }
  .bkv .bkv-table td:first-child { white-space: normal; } .bkv #bkv-explore .bkv-table td:first-child { min-width: 9.5em; } .bkv #bkv-explore .bkv-table th { white-space: normal; } }
</style>

<div class="bkv">

<section class="bkv-hero" aria-label="Overview">
  <div class="bkv-kicker"><span class="bkv-badge">arXiv preprint</span><span>October 2026 · KV cache compression · Reasoning models</span></div>
  <p class="bkv-name"><span>BreadthKV</span></p>
  <p class="bkv-tagline">In the same bytes, keep <em>more tokens at fewer bits</em>.</p>
  <p class="bkv-lede">Reasoning models write most of their KV cache while they decode, so the cache has to be compressed during generation. Existing methods decide which tokens to evict and keep the rest at full precision. We ask how a fixed byte budget should be split between the number of cached tokens and their precision. On three 7–8B reasoning models, keeping several times as many tokens at 2–4 bits beats eviction alone in 17 of 18 settings, mostly because far fewer runs keep reasoning until the length cap without giving an answer.</p>
  <p class="bkv-byline"><strong>Runguo Li</strong> · University of Illinois Urbana-Champaign</p>
  <div class="bkv-btns">
    <a class="bkv-btn primary" href="https://arxiv.org/abs/2610.05685"><i class="ai ai-arxiv" aria-hidden="true"></i>arXiv</a>
    <a class="bkv-btn primary" href="/files/breadthkv-kv-compression.pdf"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i>Paper (PDF)</a>
    <a class="bkv-btn ghost" href="#split-the-budget"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><rect x="1" y="1" width="4" height="4" rx="1" fill="currentColor"/><rect x="7" y="1" width="2" height="2" fill="currentColor"/><rect x="10" y="1" width="2" height="2" fill="currentColor"/><rect x="7" y="4" width="2" height="2" fill="currentColor"/><rect x="10" y="4" width="2" height="2" fill="currentColor"/><rect x="1" y="8" width="11" height="4" rx="1" fill="none" stroke="currentColor" stroke-width="1.2"/></svg>Split the budget</a>
    <a class="bkv-btn ghost" href="#every-method"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><rect x="1" y="1.5" width="11" height="2.4" rx="1" fill="currentColor"/><rect x="1" y="5.3" width="7" height="2.4" rx="1" fill="currentColor"/><rect x="1" y="9.1" width="9" height="2.4" rx="1" fill="currentColor"/></svg>All results</a>
    <a class="bkv-btn ghost" href="#links-and-citation"><i class="fa-solid fa-quote-right" aria-hidden="true"></i>BibTeX</a>
  </div>
  <div class="bkv-same" role="img" aria-label="At a budget of 512, eviction keeps 512 tokens at 16 bits and the 2-bit hybrid keeps 2,412 tokens in the same bytes">
    <div class="row"><b>Eviction (SnapKV-D)</b><div class="bar"><i style="width:21.2%;background:var(--c-ev);padding:0"></i><span>512 tokens, 16 bits each</span></div></div>
    <div class="row"><b>BreadthKV at 2 bits</b><div class="bar"><i style="width:94.7%;background:var(--c-hy)">2,412 tokens: 2 bits each, plus a 128-token fp16 window</i><i style="width:5.3%;background:var(--c-win);padding:0"></i></div></div>
    <div class="cap">Tokens one KV head can hold at budget B = 512. Both rows occupy the same bytes.</div>
  </div>
</section>

<div class="bkv-stats">
  <div class="bkv-stat"><div class="num">17 <small>of</small> 18</div><div class="lab">held-out settings where the low-bit hybrid beats eviction at the same bytes</div></div>
  <div class="bkv-stat"><div class="num">+17.4 <small>pp</small></div><div class="lab">pooled accuracy gain over SnapKV-D, 95% CI [14.5, 20.3]</div></div>
  <div class="bkv-stat" style="--accent:var(--c-ev)"><div class="num">91% → 40%</div><div class="lab">of Qwen3-8B AIME runs reach the 32K length cap at B = 512, eviction vs. 2 bits</div></div>
  <div class="bkv-stat" style="--accent:var(--c-ref)"><div class="num">0.76×</div><div class="lab">the output length of eviction, which lowers the memory held over time further</div></div>
</div>

<h2 id="the-problem">The problem</h2>

<p class="bkv-lead">On competition math, reasoning models such as the DeepSeek-R1 distillations and Qwen3 often write 10–30K tokens of chain of thought per problem, and the KV cache grows with every token. At 32K tokens, one Qwen3-8B sequence holds 4.8 GB of KV cache, so four sequences already take more memory than the model's 16 GB of weights. The cache has to be compressed while the model is still writing.</p>

<p class="bkv-lead">Decode-time methods so far choose <strong>which</strong> tokens to keep, by attention, redundancy or thought type, and store the survivors as 16-bit floats. An evicted token cannot be recovered, and a long chain of thought keeps returning to its own intermediate results. Under eviction, runs can fall into re-checking loops that last until the length limit, and the extra tokens they write take back part of the memory that was saved.</p>

<p class="bkv-lead">Precision is the other axis. In the bytes of K fp16 tokens, a head can hold about 3K tokens at 4 bits or 6K at 2 bits. A token stored at 2 bits is noisy, but a later reasoning step can still find it. We measure this trade at equal KV bytes, with the eviction scorer and the quantization format fixed and only the split varied.</p>

<div class="bkv-grid3">
  <div class="bkv-card" style="--accent:var(--c-ev)"><div class="h"><i></i>Models</div>R1-Distill-Qwen-7B, R1-Distill-Llama-8B and Qwen3-8B in thinking mode. Sampling at temperature 0.6 and top-p 0.95, up to 32K output tokens (16K on MATH500).</div>
  <div class="bkv-card" style="--accent:var(--c-hy)"><div class="h"><i></i>Budget</div>For each sequence, layer and KV head, decode tokens are limited to the bytes of B fp16 tokens, with B = 512 or 1024. That is 3–8% of an average AIME trace. Every method keeps the prompt in full and compresses every 128 decode steps.</div>
  <div class="bkv-card" style="--accent:var(--c-ref)"><div class="h"><i></i>Evaluation</div>pass@1 on AIME24 and AIME25 (8 samples per problem), MATH500 (1) and GPQA-Diamond (4, one model), 18 held-out settings in all. Confidence intervals come from a paired bootstrap over problems.</div>
</div>

<p class="bkv-k">Terms used below</p>
<dl class="bkv-terms">
  <div><dt>FullKV.</dt> <dd>The uncompressed cache, the reference for accuracy and memory.</dd></div>
  <div><dt>SnapKV-D.</dt> <dd>SnapKV applied during decoding: each head keeps its top-B decode tokens by the attention they receive from the last 8 queries. It is our eviction baseline and the scorer inside every hybrid.</dd></div>
  <div><dt>b-bit hybrid.</dt> <dd>The same eviction, but tokens older than a 128-token fp16 window are stored at b bits in the KIVI format, and the token budget is raised until the bytes match.</dd></div>
  <div><dt>BreadthKV.</dt> <dd>The hybrid whose width b ∈ {2, 3, 4} is chosen for each model and budget on 60 held-out calibration problems.</dd></div>
  <div><dt>Capped or derailed run.</dt> <dd>A sample that reaches the 32K-token length cap without finishing.</dd></div>
  <div><dt>KV memory-time.</dt> <dd>The bytes a cache holds, summed over decode steps. It counts both the budget and the output length.</dd></div>
  <div><dt>pp.</dt> <dd>Percentage points of pass@1 accuracy.</dd></div>
  <div><dt>Held-out regret.</dt> <dd>The gap to the best bit-width on a test setting, for a width chosen without that test set.</dd></div>
</dl>

<h2 id="key-results">Key results</h2>

<p class="bkv-k">Method</p>
<h3 class="bkv-h" id="one-budget-two-splits">One budget, two ways to use it</h3>
<p class="bkv-lead">Both rows of panel (a) occupy the same bytes. Eviction keeps B decode tokens in fp16. The hybrid stores older tokens at b bits and uses the bytes it saves to keep several times as many of them. BreadthKV is this hybrid with b chosen by three short runs on a calibration set.</p>
<figure class="bkv-fig">
  <div class="img"><a href="/images/projects/2026-10-breadthkv/fig1-overview.png" title="Open the figure at full size"><img src="/images/projects/2026-10-breadthkv/fig1-overview.png" alt="Figure 1: overview of BreadthKV: the same bytes split two ways, the calibration of the bit-width, and the share of capped runs by bit-width." loading="lazy"></a></div>
  <div class="bkv-steps">
    <div class="bkv-step"><div class="t"><span>1</span>Quantize</div>Every 128 decode steps, complete 64-token blocks older than the 128-token fp16 window are quantized to b bits: keys per channel and values per token, which costs b + 0.5 bits per element with the scales.</div>
    <div class="bkv-step"><div class="t"><span>2</span>Evict to a larger budget</div>SnapKV-D then brings each head back to B<sub>b</sub> tokens, the number that fits the bytes of B fp16 tokens: 2,412 tokens at 2 bits for B = 512, and 5,689 for B = 1024.</div>
    <div class="bkv-step"><div class="t"><span>3</span>Choose b end to end</div>The 2-, 3- and 4-bit hybrids are run on 60 AIME 2022–23 problems with 4 samples each, and the most accurate width is kept. This takes three runs, no training and no test problems.</div>
  </div>
  <p class="bkv-eq">B<sub>b</sub> = 160 + (B − 160) · 16 / (b + 0.5)
    <small>tokens per head at b bits; 160 approximates the fp16 tail, and memory-time is measured exactly</small></p>
  <figcaption><strong>Figure 1.</strong> Overview of BreadthKV. (a) At the same KV byte budget per head, eviction keeps B full-precision decode tokens, whereas BreadthKV quantizes older tokens to b bits and keeps B<sub>b</sub> of them plus a 128-token fp16 window. (b) The bit-width is chosen for each model and budget on 60 held-out AIME problems. (c) At equal bytes, fewer bits per token lower the share of runs that reach the 32K cap and raise accuracy (FullKV: 10.2% capped, 69.8% accuracy).</figcaption>
</figure>

<p class="bkv-k">Interactive</p>
<h3 class="bkv-h" id="split-the-budget">Split the budget yourself</h3>
<p class="bkv-lead">Choose a model, a budget and a precision. The grid shows how much of an average trace one head can hold in those bytes, and the bars show what happened when we ran that split on AIME25.</p>
<figure class="bkv-fig" id="bkv-split">
  <div class="bkv-ctl">
    <span>Model <span class="bkv-seg" data-role="model" role="group" aria-label="Model"></span></span>
    <span>Budget B <span class="bkv-seg" data-role="budget" role="group" aria-label="Budget"></span></span>
    <span>Precision <span class="bkv-seg" data-role="bits" role="group" aria-label="Precision"></span></span>
  </div>
  <div class="bkv-split">
    <div>
      <div class="bkv-ct">What one KV head can hold <span>· one square = 64 tokens</span></div>
      <div class="bkv-grid" data-role="grid" role="img"></div>
      <p class="bkv-gridcap" data-role="gridcap" aria-live="polite"></p>
      <div class="bkv-legend"><span><i class="e"></i>fp16 tokens</span><span><i class="q"></i>low-bit tokens</span><span><i class="w"></i>128-token fp16 window</span><span><i></i>rest of an average FullKV trace</span></div>
    </div>
    <div>
      <div class="bkv-ct" data-role="outtitle"></div>
      <div data-role="meters"></div>
      <div class="bkv-legend"><span><i class="tick"></i>FullKV</span><span><i class="tick ev"></i>eviction at this budget</span></div>
    </div>
  </div>
  <figcaption><span data-role="readout" aria-live="polite"></span>
    <span class="take">Filled squares show how many tokens a head can hold, not which ones: SnapKV-D chooses them by attention in both cases. A strike-through marks a precision that was not run at that budget. † Budgets of 2,048 and 4,096 were run on AIME25 only.</span></figcaption>
</figure>

<p class="bkv-k">Main result</p>
<h3 class="bkv-h" id="main-result">More tokens at low precision beat fewer tokens in fp16</h3>
<p class="bkv-lead">The table gives BreadthKV's accuracy gain over SnapKV-D at the same budget on every held-out setting. The gain is positive in 17 of 18 and its confidence interval excludes zero in 16. Pooled over all settings it is 17.4 points [14.5, 20.3], or 8.6 on R1-Qwen, 7.9 on R1-Llama and 35.8 on Qwen3-8B. Eviction alone nearly collapses on Qwen3-8B, which scores 5.8% on AIME25 at B = 512 against 65.4% with FullKV.</p>
<figure class="bkv-fig">
  <div class="bkv-tablewrap">
  <table class="bkv-table">
    <thead>
      <tr><th rowspan="2">Benchmark</th><th colspan="2" class="c">R1-Qwen-7B</th><th colspan="2" class="c">R1-Llama-8B</th><th colspan="2" class="c">Qwen3-8B</th></tr>
      <tr><th class="c">B = 512</th><th class="c">1024</th><th class="c">B = 512</th><th class="c">1024</th><th class="c">B = 512</th><th class="c">1024</th></tr>
    </thead>
    <tbody>
      <tr><td>AIME24</td><td class="c" colspan="2"><span class="ci">development set</span></td><td class="heat" style="background:rgba(var(--heat),.36)">+20.8</td><td class="heat" style="background:rgba(var(--heat),.19)">+8.3</td><td class="heat" style="background:rgba(var(--heat),.75)">+49.2</td><td class="heat" style="background:rgba(var(--heat),.76)">+50.0</td></tr>
      <tr><td>AIME25</td><td class="heat" style="background:rgba(var(--heat),.26)">+12.9</td><td class="heat" style="background:rgba(var(--heat),.22)">+10.0</td><td class="heat" style="background:rgba(var(--heat),.17)">+6.7*</td><td class="heat" style="background:rgba(var(--heat),.16)">+5.8</td><td class="heat" style="background:rgba(var(--heat),.56)">+35.4</td><td class="heat" style="background:rgba(var(--heat),.78)">+51.2</td></tr>
      <tr><td>MATH500</td><td class="heat" style="background:rgba(var(--heat),.20)">+8.6</td><td class="heat" style="background:rgba(var(--heat),.16)">+5.8</td><td class="heat" style="background:rgba(var(--heat),.17)">+6.8</td><td class="heat" style="background:var(--soft)">−0.8*</td><td class="heat" style="background:rgba(var(--heat),.32)">+17.4</td><td class="heat" style="background:rgba(var(--heat),.24)">+11.6</td></tr>
      <tr><td>GPQA-Diamond</td><td class="heat" style="background:rgba(var(--heat),.18)">+7.3</td><td class="heat" style="background:rgba(var(--heat),.17)">+6.8</td><td class="c" colspan="2"><span class="ci">not run</span></td><td class="c" colspan="2"><span class="ci">not run</span></td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Table 2.</strong> Accuracy gain (pp) of BreadthKV over SnapKV-D at the same budget. * The 95% CI includes 0. R1-Qwen on AIME24 was our development set and is not reported.
    <span class="take">The one negative cell is R1-Llama on MATH500 at B = 1024, where SnapKV-D is already within 2.4 points of FullKV. Which eviction score is used matters much less than the split: BreadthKV is above the best of SnapKV-D, R-KV, H2O, StreamingLLM, random eviction and VaSE in 17 of 18 settings.</span></figcaption>
</figure>

<p class="bkv-lead">A cache costs the bytes it holds over time, so longer outputs take back part of what a small budget saves. Figure 2 plots accuracy against KV memory-time. Up and to the left is better.</p>
<figure class="bkv-fig" style="--accent:var(--c-ev)">
  <div class="img"><a href="/images/projects/2026-10-breadthkv/fig2-main-aime25.png" title="Open the figure at full size"><img src="/images/projects/2026-10-breadthkv/fig2-main-aime25.png" alt="Figure 2: accuracy against KV memory-time relative to FullKV on AIME25, for three models." loading="lazy"></a></div>
  <figcaption><strong>Figure 2.</strong> Accuracy vs. KV memory-time (relative to FullKV) on AIME25. Each line connects a method's results at B = 512 and 1024. The square is KIVI-4 without eviction, and the dashed line is FullKV.
    <span class="take">SnapKV-D outputs are 1.1–2.0× as long as FullKV's, and BreadthKV's are 0.76× as long as SnapKV-D's. On Qwen3-8B at B = 1024, SnapKV-D holds 0.17× FullKV's memory-time and scores 11.7%, while BreadthKV holds 0.10× and scores 62.9%.</span></figcaption>
</figure>

<p class="bkv-k">Interactive</p>
<h3 class="bkv-h" id="every-method">Every method, setting by setting</h3>
<p class="bkv-lead">This table holds every method we ran on the 18 held-out settings. A good method has a long accuracy bar and a short bar for runs that reach the cap. KV memory-time is relative to FullKV on the same setting.</p>
<figure class="bkv-fig" id="bkv-explore">
  <div class="bkv-ctl">
    <span>Model <span class="bkv-seg" data-role="model" role="group" aria-label="Model"></span></span>
    <span>Benchmark <span class="bkv-seg" data-role="bench" role="group" aria-label="Benchmark"></span></span>
    <span>Budget <span class="bkv-seg" data-role="budget" role="group" aria-label="Budget"></span></span>
  </div>
  <div class="bkv-tablewrap">
  <table class="bkv-table">
    <thead><tr><th>Method</th><th>Accuracy (pass@1)</th><th>Runs that reach the cap</th><th>Output length</th><th>KV memory-time</th></tr></thead>
    <tbody></tbody>
  </table>
  </div>
  <p class="bkv-note" data-role="note"></p>
  <figcaption><span data-role="readout" aria-live="polite"></span>
    <span class="take">The length cap is 32K tokens, and 16K on MATH500. KIVI-4 keeps every token and holds 0.29–0.34× FullKV's memory-time, against 0.08–0.22× for the hybrids. RDKV runs with our fix to its released quantizer, and ThinKV is our re-implementation; both are discussed below.</span></figcaption>
</figure>

<p class="bkv-k">Mechanism</p>
<h3 class="bkv-h" id="derailed-runs">Eviction loses through runs that never finish</h3>
<p class="bkv-lead">We call a run <strong>derailed</strong> if it reaches the 32K cap. Derailed runs are almost never correct, and on R1-Llama and Qwen3-8B, 74–98% of SnapKV-D's wrong answers are derailed. Such a run rarely loops word for word. In one Qwen3-8B run under eviction, the model finds the right answer within the first tenth of its output and then checks it again and again, in slightly different words each time, until it is cut off.</p>
<figure class="bkv-fig" style="--accent:var(--c-ev)">
  <div class="img"><a href="/images/projects/2026-10-breadthkv/fig4-derailment.png" title="Open the figure at full size"><img src="/images/projects/2026-10-breadthkv/fig4-derailment.png" alt="Figure 4: share of samples hitting the 32K length cap on the test AIMEs, by method and budget." loading="lazy"></a></div>
  <div class="bkv-tablewrap" style="margin-top:10px">
  <table class="bkv-table">
    <thead><tr><th>Model, B = 512</th><th>Method</th><th>Accuracy</th><th>Runs that reach the cap</th><th class="r">Wrong answers that are capped</th></tr></thead>
    <tbody>
      <tr><td rowspan="4">R1-Llama-8B</td><td><span class="dot"></span>FullKV</td><td>41.2%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:6%"></i></span>6.0%</div></td><td class="r">10%</td></tr>
      <tr><td><span class="dot ev"></span>SnapKV-D</td><td>29.0%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:62.7%"></i></span>62.7%</div></td><td class="r">88%</td></tr>
      <tr><td><span class="dot hy"></span>4-bit hybrid</td><td>41.9%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:38.8%"></i></span>38.8%</div></td><td class="r">67%</td></tr>
      <tr><td><span class="dot hy"></span>2-bit hybrid</td><td>37.9%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:17.3%"></i></span>17.3%</div></td><td class="r">28%</td></tr>
      <tr class="sep"><td rowspan="4">Qwen3-8B</td><td><span class="dot"></span>FullKV</td><td>69.8%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:10.2%"></i></span>10.2%</div></td><td class="r">32%</td></tr>
      <tr><td><span class="dot ev"></span>SnapKV-D</td><td>9.4%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:91%"></i></span>91.0%</div></td><td class="r">98%</td></tr>
      <tr><td><span class="dot hy"></span>4-bit hybrid</td><td>28.8%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:65.4%"></i></span>65.4%</div></td><td class="r">90%</td></tr>
      <tr><td><span class="dot hy"></span>2-bit hybrid</td><td>51.7%</td><td><div class="cell"><span class="bar"><i class="cap" style="width:40.4%"></i></span>40.4%</div></td><td class="r">82%</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Figure 4 and the derailment table (excerpt).</strong> Share of samples hitting the 32K length cap on the test AIMEs (R1-Qwen: AIME25; others: AIME24 and AIME25), by method and budget. Dashed: FullKV.
    <span class="take">At equal bytes, fewer bits per token mean fewer derailed runs: going from 4 to 3 to 2 bits never raises the capped share. It does not always raise accuracy, because fewer bits add noise to every token that is kept. On R1-Llama at B = 512, 4 bits derails about twice as often as 2 bits and still scores higher.</span></figcaption>
</figure>

<h3 class="bkv-h" id="what-the-cache-holds">What the cache still holds when the model checks again<sup class="dg">†</sup></h3>
<p class="bkv-lead">Our account is that eviction deletes the record of a check along with its result. Once both have gone unattended long enough to be evicted, the model derives the result again, and that derivation is evicted in turn. To test the first half, we replayed all 480 Qwen3-8B AIME runs at B = 512 under the same compression and recorded, each time the model started a new check, how much of its earlier checks the caches still held.</p>
<figure class="bkv-fig" style="--accent:var(--c-ev)">
  <div class="bkv-tablewrap">
  <table class="bkv-table">
    <thead><tr><th>Earlier check ended</th><th><span class="dot ev"></span>SnapKV-D</th><th><span class="dot hy"></span>2-bit hybrid</th></tr></thead>
    <tbody>
      <tr><td>under 1K tokens ago</td><td><div class="cell"><span class="bar"><i class="ev" style="width:41%"></i></span>41%</div></td><td><div class="cell"><span class="bar"><i class="hy" style="width:94%"></i></span>94%</div></td></tr>
      <tr><td>1–2K tokens ago</td><td><div class="cell"><span class="bar"><i class="ev" style="width:3%"></i></span>3%</div></td><td><div class="cell"><span class="bar"><i class="hy" style="width:67%"></i></span>67%</div></td></tr>
      <tr><td>2–4K tokens ago</td><td><div class="cell"><span class="bar"><i class="ev" style="width:1%"></i></span>1%</div></td><td><div class="cell"><span class="bar"><i class="hy" style="width:27%"></i></span>27%</div></td></tr>
      <tr><td>4–8K tokens ago</td><td><div class="cell"><span class="bar"><i class="ev" style="width:1%"></i></span>1%</div></td><td><div class="cell"><span class="bar"><i class="hy" style="width:6%"></i></span>6%</div></td></tr>
      <tr><td>8K or more tokens ago</td><td><div class="cell"><span class="bar"><i class="ev" style="width:0.5%"></i></span>0%</div></td><td><div class="cell"><span class="bar"><i class="hy" style="width:1%"></i></span>1%</div></td></tr>
      <tr class="sep"><td>First statement of the correct answer,<br>when it is restated 1K or more tokens later</td><td><div class="cell"><span class="bar"><i class="ev" style="width:1%"></i></span>1%</div></td><td><div class="cell"><span class="bar"><i class="hy" style="width:23%"></i></span>23%</div></td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Retention replay</strong> (Qwen3-8B, AIME24 and AIME25, B = 512). Share of layer-head caches still holding the tokens of earlier checks when a new check starts, by tokens since the earlier check ended.
    <span class="take">Under SnapKV-D a check is almost entirely evicted within 1K tokens after it ends. Of the runs that state the correct answer at some point, 20% conclude under SnapKV-D, 79% under the 2-bit hybrid and 96% with FullKV. Pinning the first statements of the correct answer in SnapKV-D's cache, an oracle, doubles the share that conclude to 40% and raises accuracy from 9.4% to 14.0%, still far below the hybrid's 51.7%. Whether losing the surrounding checks causes the re-checking is not tested.</span></figcaption>
</figure>

<p class="bkv-k">Choosing the width</p>
<h3 class="bkv-h" id="choosing-the-width">No fixed width is safe for every model</h3>
<p class="bkv-lead">The best width depends on the model and the budget, and the cheap proxies we tried do not find it. Offline attention error at equal bytes gets the preference wrong on two of the three models. For Qwen3-8B it favors 4 bits over 2, the opposite of what happens end to end. BreadthKV measures the preference directly on the calibration set.</p>
<figure class="bkv-fig">
  <div class="bkv-pair">
    <div class="bkv-tablewrap">
    <table class="bkv-table">
      <thead><tr><th>Calibration accuracy (%)</th><th class="r">B</th><th class="r">2-bit</th><th class="r">3-bit</th><th class="r">4-bit</th><th>Chosen</th></tr></thead>
      <tbody>
        <tr><td rowspan="2">R1-Qwen-7B</td><td class="r">512</td><td class="r">28.3</td><td class="r"><b>31.2</b></td><td class="r">28.3</td><td>3 bits</td></tr>
        <tr><td class="r">1024</td><td class="r">33.8</td><td class="r">35.0</td><td class="r"><b>40.0</b></td><td>4 bits</td></tr>
        <tr><td rowspan="2">R1-Llama-8B</td><td class="r">512</td><td class="r">33.8</td><td class="r"><b>36.7</b></td><td class="r">35.8</td><td>3 bits</td></tr>
        <tr><td class="r">1024</td><td class="r"><b>40.0</b></td><td class="r"><b>40.0</b></td><td class="r">39.2</td><td>2 bits</td></tr>
        <tr><td rowspan="2">Qwen3-8B</td><td class="r">512</td><td class="r"><b>43.3</b></td><td class="r">30.4</td><td class="r">20.8</td><td>2 bits</td></tr>
        <tr><td class="r">1024</td><td class="r"><b>62.9</b></td><td class="r">57.1</td><td class="r">50.0</td><td>2 bits</td></tr>
      </tbody>
    </table>
    </div>
    <div class="img"><a href="/images/projects/2026-10-breadthkv/fig3-calibration-regret.png" title="Open the figure at full size"><img src="/images/projects/2026-10-breadthkv/fig3-calibration-regret.png" alt="Figure 3: held-out regret of bit-width selection strategies for each model." loading="lazy"></a></div>
  </div>
  <figcaption><strong>Table 3 and Figure 3.</strong> Calibration accuracy of the 2-, 3- and 4-bit hybrids with the chosen width (ties go to fewer bits), and the held-out regret of each selection strategy: the mean gap (pp) to the best bit-width over each model's six test settings. Lower is better.
    <span class="take">Calibration beats always using 4 bits by 4.5 points [3.2, 5.9] and always using 3 bits by 2.1 [1.1, 3.1]. At these budgets always using 2 bits is statistically as good (+0.4 [−0.3, 1.0]). Each fixed width does badly on some model, 2 bits on R1-Llama (mean regret 3.1 points) and 4 bits on Qwen3-8B (12.7), while calibration stays within 1.3 points on every model.</span></figcaption>
</figure>

<p class="bkv-k">Baselines</p>
<h3 class="bkv-h" id="other-allocators">Against finer allocators</h3>
<p class="bkv-lead">BreadthKV uses one width per model and budget. We ran three finer ways of allocating precision under the same protocol.</p>
<div class="bkv-grid3">
  <div class="bkv-card" style="--accent:var(--c-jt)"><div class="num">−1.0 pp</div><div class="h">RDKV, at 1.27× the memory-time</div>RDKV solves a rate–distortion problem that gives each value token and key channel 0, 2, 4, 8 or 16 bits. We ran the authors' code in its block-wise decode mode, after fixing a zero-point clamp in the released quantizer that made generation collapse on R1-Llama. Pooled over the 18 settings it is 1.0 point below BreadthKV [−2.04, +0.02] while holding 1.27× its KV memory-time. A native decode-time implementation might do better.</div>
  <div class="bkv-card" style="--accent:var(--c-jt)"><div class="num">−12.3 pp</div><div class="h">ThinKV, re-implemented</div>ThinKV has no public code, so we re-implemented it from the paper. Our version scores 12.3 points below BreadthKV [−14.9, −9.7] with 0.86× its memory-time, and its runs reach the cap far more often. In the one setting we can compare with the ThinKV paper, our version reaches 25.0% where 50% is reported<sup class="dg">†</sup>, so this is a gap to our re-implementation and may not hold for ThinKV as published.</div>
  <div class="bkv-card" style="--accent:var(--c-hy)"><div class="num">1 of 18</div><div class="h">Per-head online allocation</div>Choosing the width online for each head lowers offline attention error by 20–42%. End to end it is indistinguishable from fixed widths, with 1 of 18 comparisons significant. Single-step attention error predicts accuracy poorly.</div>
</div>

<h2 id="where-it-stops-helping">Where it stops helping</h2>

<p class="bkv-k">Larger budgets<sup class="dg">†</sup></p>
<h3 class="bkv-h" id="larger-budgets">The lead shrinks as eviction nears the full cache</h3>
<p class="bkv-lead">The main budgets hold 3–8% of an average AIME trace. With 2–8× larger budgets on AIME25, or 12–30% of a trace, eviction comes closer to FullKV and the hybrids' lead shrinks. On R1-Llama it is gone at B = 4096. On R1-Qwen and Qwen3-8B the 4-bit hybrid still leads by 7.9 [+2.5, +13.8] and 16.2 [+7.9, +25.4] points there, and reaches FullKV's accuracy with 0.24× its KV memory-time.</p>
<figure class="bkv-fig" style="--accent:var(--c-ref)">
  <div class="bkv-tablewrap">
  <table class="bkv-table">
    <thead><tr><th>AIME25, pass@1 (%)</th><th>B = 512</th><th>B = 1024</th><th>B = 2048</th><th>B = 4096</th><th class="r">FullKV</th></tr></thead>
    <tbody>
      <tr><td>R1-Qwen-7B</td><td>20.4 → <b>35.4</b> <span class="ci">2-bit</span></td><td>26.2 → <b>37.5</b> <span class="ci">3-bit</span></td><td>31.2 → <b>37.9</b> <span class="ci">4-bit</span></td><td>34.6 → <b>42.5</b> <span class="ci">4-bit</span></td><td class="r">41.2</td></tr>
      <tr><td>R1-Llama-8B</td><td>26.2 → <b>32.9</b> <span class="ci">3-bit</span></td><td>27.1 → <b>34.6</b> <span class="ci">4-bit</span></td><td>30.4 → <b>35.8</b> <span class="ci">4-bit</span></td><td><b>36.7</b> → 35.8 <span class="ci">8-bit</span></td><td class="r">31.2</td></tr>
      <tr><td>Qwen3-8B</td><td>5.8 → <b>41.2</b> <span class="ci">2-bit</span></td><td>11.7 → <b>62.9</b> <span class="ci">2-bit</span></td><td>37.1 → <b>66.2</b> <span class="ci">2-bit</span></td><td>52.1 → <b>68.3</b> <span class="ci">4-bit</span></td><td class="r">65.4</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>SnapKV-D → the most accurate byte-matched hybrid</strong> at each budget on AIME25, with that hybrid's width. The width here is the best on the test set, not the calibrated one.
    <span class="take">The best width rises with the budget, from 2–3 bits at B = 512 to 4 or 8 bits at B = 4096.</span></figcaption>
</figure>

<p class="bkv-k">Cost</p>
<h3 class="bkv-h" id="decode-speed">Decode steps are slower</h3>
<p class="bkv-lead">Holding several times as many tokens makes attention more expensive, even at the same bytes. We wrote a Triton decode kernel that reads packed low-bit KV directly and timed whole decode steps with it on an A800. The accuracy gain does not come with faster decoding.</p>
<div class="bkv-grid2">
  <div class="bkv-card" style="--accent:var(--c-hy)"><div class="num">1.2–3.0×</div><div class="h">Time per decode step<sup class="dg">†</sup></div>A 2-bit hybrid step takes 1.2–2.0× as long as a SnapKV-D step at B = 512 and 1.3–3.0× at B = 1024, more at larger batch sizes. At equal context, the 4-bit kernel is close to fp16 FlashAttention at 27% of the memory.</div>
  <div class="bkv-card" style="--accent:var(--c-hy)"><div class="num">0.89–1.9×</div><div class="h">Decode time per sample<sup class="dg">†</sup></div>The hybrid writes shorter outputs, so a sample needs fewer steps. Counting correct AIME answers per GPU-hour, it is ahead of SnapKV-D by 2.0–6.2× on Qwen3-8B but only 0.8–1.7× on R1-Qwen-7B.</div>
  <div class="bkv-card" style="--accent:var(--c-ev)"><div class="num">40%</div><div class="h">Runs that still derail</div>On Qwen3-8B at our tightest budget, 40% of 2-bit runs still reach the cap. Cutting capped runs where they start to repeat and forcing a conclusion recovers 15% of them on Qwen3-8B and almost none on the R1 models.</div>
  <div class="bkv-card" style="--accent:var(--c-ref)"><div class="num">≈ 2 bits</div><div class="h">The floor of breadth</div>With the codecs we tested, going below 2 bits stops helping. A 1.25-bit codec collapses end to end, and a 2.25-bit one is 2.6 points below the plain 2-bit hybrid although its offline error is the same.</div>
</div>

<h2 id="using-these-results">Using these results</h2>

<div class="bkv-grid3">
  <div class="bkv-card"><div class="h"><span>1</span>Start from 2 bits at tight budgets</div>When the budget holds under a tenth of the trace, quantize older tokens to 2 bits and raise the token budget with the formula above until the bytes match.</div>
  <div class="bkv-card"><div class="h"><span>2</span>Check the width end to end</div>In our runs, three calibration runs on 60 problems avoided the width that fails on each model. Without labels, picking the width with the shortest calibration outputs does as well as always using 2 bits.</div>
  <div class="bkv-card"><div class="h"><span>3</span>Raise the bits with the budget, and watch the cap</div>At 12–30% of a trace, 4 or 8 bits were best on the R1 models. Report the share of capped runs and KV memory-time along with accuracy, since longer outputs take back memory.</div>
</div>

<p class="bkv-scope">Scope: three open 7–8B reasoning models on English math and science benchmarks. Quantization is simulated in the engine with exact byte accounting. The main runs use one sampling seed, with a three-seed check on one setting per model.</p>

<p class="bkv-scope">† Added after arXiv v2. This page follows the current revision of the paper, whose pooled confidence intervals resample each benchmark's problems jointly across the settings that share them, so a few intervals differ from v2.</p>

<h2 id="links-and-citation">Links and citation</h2>

<div class="bkv-links">
  <a class="bkv-btn primary" href="https://arxiv.org/abs/2610.05685"><i class="ai ai-arxiv" aria-hidden="true"></i>arXiv 2610.05685</a>
  <a class="bkv-btn ghost" href="/files/breadthkv-kv-compression.pdf"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i>Paper (PDF)</a>
  <a class="bkv-btn ghost" href="/files/bibtex/li2026breadthkv.bib"><i class="fa-solid fa-quote-right" aria-hidden="true"></i>BibTeX file</a>
  <a class="bkv-btn ghost" href="/publication/2026-10-breadthkv"><i class="fa-solid fa-book-open" aria-hidden="true"></i>Publication page</a>
</div>

<p class="bkv-scope">The code, all configurations and the per-sample generations of every run will be released with the paper.</p>

<div class="bkv-bib">
<button class="bkv-copy" type="button" data-target="bkv-bibtex">Copy</button>
<pre><code id="bkv-bibtex">@misc{li2026breadthkv,
  title         = {Spend Bytes on Breadth: Precision--Count Trade-offs for Decode-Time {KV} Compression in Long Chain-of-Thought Reasoning},
  author        = {Li, Runguo},
  year          = {2026},
  eprint        = {2610.05685},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2610.05685}
}</code></pre>
</div>

</div>

<script src="/files/projects/2026-10-breadthkv/explore.js" defer></script>
