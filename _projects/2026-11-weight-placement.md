---
title: "Weight Placement Is Not Enough: Synchronization in Multi-Node CPU Decoding of Ternary LLMs"
collection: projects
permalink: /projects/2026-11-weight-placement/
date: 2026-11-22
# No venue under review is named on this page. Paper, code and publication links: add when they are public.
status: "Under review"
venue: "Under review"
excerpt: "Once every weight of a ternary LLM is read from its own NUMA node, threads on dual-socket Xeon servers still wait for 20–48% of each decoded token at synchronization points. A column-split decode engine on ik_llama.cpp's kernels cuts that wait and decodes 1.5–2.1× faster where decoding is DRAM-bound."
teaser: "/images/projects/2026-11-weight-placement/teaser.png"
tags: ["LLM inference", "CPU", "NUMA", "ternary LLMs", "synchronization"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-weight-placement"
---

<style>
/* Page-scoped styles: everything is under .wp so no other page changes. */
.wp { --c-ik: #8e8e8e; --c-ikp: #56b4e9; --c-step: #a9d6f2; --c-row: #e69f00; --c-col: #d55e00; --c-wait: #6b6a66; --c-ok: #008300;
  --ink: #1d1d1b; --ink2: #52514e; --muted: #898781; --line: rgba(11, 11, 11, 0.09); --card: #ffffff; --soft: #f6f6f3;
  --shadow: 0 1px 2px rgba(11, 11, 11, 0.04), 0 10px 30px rgba(11, 11, 11, 0.06); color: var(--ink); }
html[data-theme="dark"] .wp { --c-ik: #9a9a96; --c-ikp: #56b4e9; --c-step: #3f7fa6; --c-row: #e69f00; --c-col: #f07a2a; --c-wait: #b5b4ae; --c-ok: #2a9d2a;
  --ink: #f3f3f1; --ink2: #d0cfc8; --muted: #a9a8a1; --line: rgba(255, 255, 255, 0.10); --card: #3b3b3b; --soft: #404040;
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.25), 0 10px 30px rgba(0, 0, 0, 0.2); }
.wp p { margin: 0 0 0.9em; }
/* the layout's page title must not run under the fixed table of contents */
@media screen and (min-width: 1200px) { .page__title { max-width: calc(100% - 250px); } }
.wp-hero { position: relative; overflow: hidden; border-radius: 18px; padding: 28px 28px 22px; margin: 0.4em 0 1.4em; border: 1px solid var(--line);
  background: radial-gradient(70% 120% at 0% 0%, rgba(86, 180, 233, 0.14), transparent 60%), radial-gradient(60% 120% at 100% 0%, rgba(213, 94, 0, 0.12), transparent 60%), var(--card);
  box-shadow: var(--shadow); }
.wp-hero::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: linear-gradient(90deg, var(--c-ik) 0%, var(--c-ikp) 35%, var(--c-row) 68%, var(--c-col) 100%); }
.wp-kicker { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 0.72em; letter-spacing: 0.06em; text-transform: uppercase; color: var(--ink2); margin-bottom: 10px; }
.wp-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; background: var(--ink); color: var(--card); font-weight: 700; letter-spacing: 0.04em; }
.wp .wp-name { font-size: 2.3em; line-height: 1.08; font-weight: 800; letter-spacing: -0.03em; margin: 0 0 8px; }
.wp .wp-name span { background: linear-gradient(90deg, var(--c-ikp), var(--c-col)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.wp .wp-tagline { font-size: 1.2em; line-height: 1.35; font-weight: 600; margin: 0 0 10px; }
.wp .wp-tagline em { font-style: normal; color: var(--c-col); }
.wp .wp-byline { font-size: 0.88em; color: var(--ink2); margin: 0 0 12px; }
.wp .wp-byline strong { color: var(--ink); }
.wp .wp-lede { font-size: 0.95em; color: var(--ink2); max-width: 48em; }
.wp-btns { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 18px; }
.wp .wp-btn { display: inline-flex; align-items: center; gap: 7px; padding: 7px 15px; border-radius: 999px; font-size: 0.85em; font-weight: 600; text-decoration: none; border: 1px solid var(--ink); }
.wp .wp-btn.primary { background: var(--ink); color: var(--card); }
.wp .wp-btn.ghost { background: transparent; color: var(--ink); }
.wp .wp-btn.soon { background: transparent; color: var(--muted); border-color: var(--line); cursor: default; }
.wp .wp-btn svg { flex: none; }
/* chips */
.wp-ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 0 2px 10px; font-size: 0.82em; color: var(--ink2); }
.wp-seg { display: inline-flex; flex-wrap: wrap; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--card); }
.wp .wp-seg button { border: 0; background: transparent; color: var(--ink2); font: inherit; font-size: 1em; padding: 4px 10px; cursor: pointer; margin: 0; }
.wp .wp-seg button + button { border-left: 1px solid var(--line); }
.wp .wp-seg button[aria-pressed="true"] { background: var(--ink); color: var(--card); font-weight: 600; }
.wp .wp-again { border: 1px solid var(--line); background: var(--card); color: var(--ink); font: inherit; font-size: 1em; padding: 4px 12px; border-radius: 999px; cursor: pointer; margin: 0; }
/* replay of the measured rates */
.wp-race { padding: 14px 16px 12px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); }
.wp-race .lane { display: grid; grid-template-columns: 11.5em 1fr 8.2em; align-items: center; gap: 10px; font-size: 0.82em; padding: 5px 0; }
.wp-race .who b { display: block; font-weight: 700; }
.wp-race .who span { display: block; color: var(--muted); font-size: 0.9em; line-height: 1.25; }
.wp-race .track { position: relative; height: 26px; border-radius: 6px; background: var(--card); border: 1px solid var(--line); overflow: hidden; }
.wp-race .track i { position: absolute; left: 0; top: 0; bottom: 0; width: 0; background: var(--c); border-radius: 5px 0 0 5px; }
.wp-race .track .cnt { position: absolute; right: 8px; top: 0; line-height: 24px; font-size: 0.88em; color: var(--ink2); font-variant-numeric: tabular-nums; }
.wp-race .lane.done .track .cnt { color: var(--ink); font-weight: 700; }
.wp-race .rate { text-align: right; font-variant-numeric: tabular-nums; line-height: 1.2; white-space: nowrap; }
.wp-race .rate b { font-size: 1.25em; font-weight: 800; }
.wp-race .rate em { display: block; font-style: normal; color: var(--muted); font-size: 0.9em; }
.wp-race .cap { font-size: 0.75em; color: var(--muted); margin-top: 6px; line-height: 1.45; }
.wp-race .cap b { color: var(--ink2); font-weight: 600; }
/* stat tiles */
.wp-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 1.6em; }
.wp-stat { position: relative; padding: 14px 14px 12px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }
.wp-stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.wp-stat .num { font-size: 1.8em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; }
.wp-stat .lab { font-size: 0.78em; color: var(--ink2); line-height: 1.35; margin-top: 4px; }
/* text */
.wp .wp-k { font-size: 0.7em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 1.8em 0 0.1em; }
.wp h3.wp-h { font-size: 1.1em; margin: 0 0 0.55em; }
.wp .wp-lead { font-size: 0.95em; line-height: 1.6; color: var(--ink2); margin: 0 0 0.9em; max-width: 52em; }
.wp .wp-lead strong { color: var(--ink); }
.wp .wp-scope { font-size: 0.8em; color: var(--muted); margin: 0 0 1.2em; line-height: 1.55; }
/* figures */
.wp .wp-fig { display: block; margin: 0 0 1.2em; padding: 14px; border-radius: 16px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); }
.wp-fig .img { background: #fff; border-radius: 10px; padding: 8px; }
.wp-fig .img.narrow { max-width: 520px; margin: 0 auto; }
.wp-fig img { display: block; width: 100%; height: auto; margin: 0 auto; }
.wp .wp-fig figcaption { font-family: inherit; font-size: 0.82em; color: var(--ink2); line-height: 1.5; margin: 10px 4px 0; }
.wp-fig figcaption strong { color: var(--ink); }
.wp-fig .take { display: block; margin-top: 6px; padding: 7px 10px; border-radius: 8px; background: var(--soft); color: var(--ink); border-left: 3px solid var(--accent, var(--ink2)); }
.wp details.wp-more { margin: 10px 4px 0; font-size: 0.8em; color: var(--ink2); }
.wp details.wp-more summary { cursor: pointer; font-weight: 600; color: var(--ink); }
.wp details.wp-more .img { margin-top: 8px; }
/* trace */
.wp-trace { background: #fff; border-radius: 10px; padding: 8px 8px 4px; }
.wp-trace canvas { display: block; width: 100%; touch-action: pan-y; cursor: crosshair; }
.wp-keys { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 0.76em; color: var(--ink2); margin: 8px 4px 0; }
.wp-keys span { display: inline-flex; align-items: center; gap: 5px; }
.wp-keys i { display: inline-block; width: 12px; height: 10px; border-radius: 2px; border: 1px solid rgba(128, 128, 128, 0.55); }
html[data-theme="dark"] .wp-split .bar i:last-child { box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.4); }
.wp .wp-out { font-size: 0.84em; color: var(--ink2); margin: 8px 4px 0; min-height: 1.5em; font-variant-numeric: tabular-nums; }
.wp .wp-out b { color: var(--ink); }
.wp-tip { position: fixed; z-index: 50; pointer-events: none; background: #fff; color: #1d1d1b; border: 1px solid rgba(11, 11, 11, 0.12); box-shadow: 0 6px 20px rgba(0, 0, 0, 0.18); border-radius: 8px; padding: 7px 9px; font-size: 12px; line-height: 1.45; }
.wp-tip .tt { color: #52514e; }
.wp-tip .tr { display: flex; align-items: center; gap: 6px; white-space: nowrap; }
.wp-tip .tr i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; border: 1px solid rgba(0, 0, 0, 0.15); }
.wp-tip .tr span { color: #52514e; }
.wp-split { display: grid; gap: 6px; margin: 14px 4px 0; padding: 12px 14px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); }
.wp-split .g { font-size: 0.74em; font-weight: 700; letter-spacing: 0.04em; color: var(--muted); text-transform: uppercase; margin-top: 2px; }
.wp-split .row { display: grid; grid-template-columns: 2.6em 1fr 11.5em; align-items: center; gap: 10px; font-size: 0.82em; }
.wp-split .bar { display: flex; height: 18px; }
.wp-split .bar i { display: block; height: 100%; }
.wp-split .bar i:first-child { border-radius: 4px 0 0 4px; }
.wp-split .bar i:last-child { border-radius: 0 4px 4px 0; }
.wp-split .val { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--ink2); }
.wp-split .val b { color: var(--ink); }
/* cards, terms */
.wp-grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.wp-grid3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.wp-grid2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin: 0.2em 0 1.2em; }
.wp-card { position: relative; padding: 14px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); box-shadow: var(--shadow); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.wp-card .h { font-weight: 700; color: var(--ink); margin-bottom: 4px; display: flex; align-items: center; gap: 8px; }
.wp-card .h i { flex: none; width: 9px; height: 9px; border-radius: 50%; background: var(--accent, var(--ink2)); }
.wp-card .h span { flex: none; display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
.wp-card .num { font-size: 1.7em; font-weight: 800; letter-spacing: -0.02em; line-height: 1.1; color: var(--ink); }
.wp-card.ev { padding-left: 18px; overflow: hidden; }
.wp-card.ev::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); }
.wp-card.ev .h { margin-top: 4px; }
.wp .wp-terms { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 28px; margin: 0.2em 0 1.4em; padding: 8px 18px; border-radius: 14px; background: var(--soft); border: 1px solid var(--line); font-size: 0.84em; line-height: 1.5; }
.wp .wp-terms > div { padding: 8px 0; border-bottom: 1px solid var(--line); }
.wp .wp-terms > div:nth-last-child(-n+2) { border-bottom: 0; }
.wp .wp-terms dt { display: inline; font-size: 1em; font-weight: 700; color: var(--ink); margin: 0; }
.wp .wp-terms dd { display: inline; font-size: 1em; margin: 0; color: var(--ink2); }
/* dataflow diagram */
.wp-flowwrap { overflow-x: auto; }
.wp-flow { min-width: 600px; padding: 14px 14px 10px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); font-size: 0.78em; }
.wp-flow .lane { display: grid; grid-template-columns: 3em 1fr; align-items: center; gap: 8px; }
.wp-flow .lane > b { font-size: 1.15em; }
.wp-flow .st { display: flex; align-items: stretch; height: 38px; }
.wp-flow .st span { display: flex; align-items: center; justify-content: center; border: 1px solid var(--muted); background: var(--card); border-radius: 4px; color: var(--ink); white-space: nowrap; min-width: 0; padding: 0 2px; }
.wp-flow .st span.sum { background: rgba(213, 94, 0, 0.16); border-color: var(--c-col); font-weight: 700; }
.wp-flow .st u { flex: none; width: 0; margin: -4px 5px; border-left: 4px solid var(--ink); text-decoration: none; }
.wp-flow .st s { flex: none; width: 0; margin: -1px 5px; border-left: 2px dashed var(--muted); text-decoration: none; }
.wp-flow .hint { color: var(--ink2); margin: 5px 0 12px calc(3em + 8px); }
.wp-flow .st sub { font-size: 0.78em; position: relative; top: 0.35em; margin-left: 1px; }
.wp :not(pre) > code { white-space: nowrap; font-size: 0.88em; }
.wp-flow .key { display: flex; flex-wrap: wrap; gap: 4px 18px; color: var(--ink2); margin-top: 2px; padding-top: 8px; border-top: 1px solid var(--line); }
.wp-flow .key span { display: inline-flex; align-items: center; gap: 7px; }
.wp-flow .key u { display: inline-block; height: 16px; border-left: 4px solid var(--ink); }
.wp-flow .key s { display: inline-block; height: 14px; border-left: 2px dashed var(--muted); }
.wp-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }
.wp-step { padding: 12px 14px; border-radius: 12px; background: var(--soft); border: 1px solid var(--line); font-size: 0.86em; line-height: 1.5; color: var(--ink2); }
.wp-step .t { font-weight: 700; color: var(--ink); display: flex; align-items: center; gap: 8px; margin-bottom: 3px; }
.wp-step .t span { display: inline-grid; place-items: center; width: 22px; height: 22px; border-radius: 50%; background: var(--ink); color: var(--card); font-size: 0.8em; }
/* tables */
.wp-tablewrap { overflow-x: auto; }
.wp .wp-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.82em; margin: 0; display: table; border: 0; background: transparent; }
.wp .wp-table thead, .wp .wp-table thead tr, .wp .wp-table tbody tr { background: transparent; }
.wp .wp-table th, .wp .wp-table td { border: 0; border-bottom: 1px solid var(--line); background: transparent; }
.wp .wp-table th { text-align: right; font-weight: 600; color: var(--ink2); padding: 7px 10px; white-space: nowrap; }
.wp .wp-table th:first-child, .wp .wp-table td:first-child { text-align: left; }
.wp .wp-table th.grp { text-align: center; }
.wp .wp-table td { padding: 8px 10px; text-align: right; vertical-align: middle; font-variant-numeric: tabular-nums; white-space: nowrap; }
.wp .wp-table tr.hl td { background: rgba(213, 94, 0, 0.09); font-weight: 700; }
.wp .wp-table tr.hl td:first-child { box-shadow: inset 3px 0 0 var(--c-col); }
.wp .wp-table tr.sep td { border-top: 2px solid var(--line); color: var(--ink2); }
.wp-table .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 7px; }
/* explorer */
.wp-charts { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 18px 24px; margin-bottom: 18px; }
.wp-charts.even { grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); margin-bottom: 0; }
.wp-ct { font-size: 0.82em; font-weight: 700; margin: 0 0 6px; }
.wp-ct span { font-weight: 400; color: var(--ink2); }
.wp-bars { display: grid; gap: 5px; }
.wp-bars .r { display: grid; grid-template-columns: 13.2em 1fr 8.6em; align-items: center; gap: 8px; font-size: 0.8em; }
.wp-bars.short .r { grid-template-columns: 3em 1fr 3.6em; }
.wp-bars .n { color: var(--ink2); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.wp-bars .n b { color: var(--ink); font-weight: 600; }
.wp-bars .n small { color: var(--muted); font-size: 0.92em; }
.wp-bars .b { position: relative; height: 16px; }
.wp-bars .b.full { background: var(--soft); border-radius: 4px; }
.wp-bars .b i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; transition: width 0.35s ease; }
.wp-bars .b u { position: absolute; top: -3px; bottom: -3px; width: 0; border-left: 2px solid var(--ink); }
.wp-bars .v { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; color: var(--ink2); }
.wp-bars .v b { color: var(--ink); }
.wp-bars .v em { font-style: normal; display: inline-block; min-width: 3.4em; color: var(--ink2); }
.wp-bars .r.hl .v em { color: var(--c-col); font-weight: 700; }
.wp-bars .note { font-size: 0.76em; color: var(--muted); line-height: 1.45; margin-top: 4px; }
.wp-line svg { display: block; overflow: visible; }
.wp-line text { font-family: inherit; }
.wp-line .tk { fill: var(--muted); font-size: 10.5px; font-variant-numeric: tabular-nums; }
.wp-line .al { fill: var(--ink2); font-size: 11px; }
.wp-line .gl { stroke: var(--line); stroke-width: 1; }
.wp-line .lg { display: flex; flex-wrap: wrap; gap: 2px 12px; font-size: 0.74em; color: var(--ink2); margin-top: 2px; }
.wp-line .lg span { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
.wp-line .lg b { color: var(--ink); font-variant-numeric: tabular-nums; }
/* abstract, links, bibtex */
.wp details.wp-abs { margin: 0 0 1.2em; padding: 12px 16px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); font-size: 0.9em; }
.wp details.wp-abs summary { cursor: pointer; font-weight: 700; }
.wp details.wp-abs p { margin: 10px 0 2px; color: var(--ink2); line-height: 1.6; }
.wp-links { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; }
.wp-link { padding: 14px; border-radius: 14px; background: var(--card); border: 1px solid var(--line); font-size: 0.86em; }
.wp-link .h { font-weight: 700; margin-bottom: 2px; }
.wp-link .s { color: var(--muted); }
.wp-bib { position: relative; }
.wp .wp-bib pre { margin: 0; padding: 14px 16px; border-radius: 12px; border: 0; background: #1f1f1e; color: #e8e7e2; font-size: 0.76em; line-height: 1.6; overflow-x: auto; white-space: pre; }
.wp .wp-bib pre code { background: transparent; color: inherit; padding: 0; border: 0; font-size: 1em; white-space: pre; }
.wp .wp-copy { position: absolute; right: 10px; top: 10px; border: 1px solid rgba(255, 255, 255, 0.25); background: transparent; color: #e8e7e2; font: inherit; font-size: 0.72em; padding: 3px 10px; border-radius: 999px; cursor: pointer; margin: 0; }
@media (max-width: 860px) {
  .wp-charts, .wp-charts.even { grid-template-columns: minmax(0, 1fr); }
  .wp-grid4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 720px) {
  .wp-grid3, .wp-grid2, .wp-steps, .wp-links { grid-template-columns: minmax(0, 1fr); }
  .wp .wp-terms { grid-template-columns: minmax(0, 1fr); }
  .wp .wp-terms > div:nth-last-child(2) { border-bottom: 1px solid var(--line); }
  .wp-hero { padding: 22px 16px 16px; }
  .wp .wp-name { font-size: 1.8em; }
  .wp-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .wp-race .lane { grid-template-columns: 1fr 7.6em; gap: 2px 10px; }
  .wp-race .who { grid-column: 1 / -1; }
  .wp-race .who span { display: none; }
  .wp-split .row { grid-template-columns: 2.6em 1fr; }
  .wp-split .val { grid-column: 1 / -1; text-align: left; }
  .wp-bars .r { grid-template-columns: 1fr 8.6em; gap: 1px 8px; }
  .wp-bars .r .n { grid-column: 1 / -1; }
  .wp-bars.short .r { grid-template-columns: 3em 1fr 3.6em; }
  .wp-bars.short .r .n { grid-column: auto; }
}
</style>

<div class="wp">

<section class="wp-hero" aria-label="Overview">
  <div class="wp-kicker"><span class="wp-badge">Under review</span><span>LLM inference · CPU servers · NUMA</span></div>
  <p class="wp-name"><span>Weight placement is not enough</span></p>
  <p class="wp-tagline">With every weight on the right NUMA node, ternary LLM decoding still spends <em>20–48% of each token waiting</em> at synchronization points.</p>
  <p class="wp-byline"><strong>Runguo Li</strong><sup>*</sup> · University of Illinois Urbana-Champaign &nbsp;&nbsp; <strong>Dezhen Lu</strong><sup>*</sup> · University of Tübingen &nbsp;&nbsp; <span><sup>*</sup>equal contribution</span></p>
  <p class="wp-lede">A ternary LLM reads only about 2 GB of weights per generated token, so a dual-socket CPU server with hundreds of GB/s of memory bandwidth should decode it quickly. The fastest ternary runtime we found, ik_llama.cpp, uses less than half of that bandwidth. We fix weight placement, measure what the threads do afterwards, and build a decode engine on the same kernels whose threads meet only a few times per layer. Where decoding is bound by DRAM, it is <strong>1.5–2.1× faster</strong> than ik_llama.cpp at the same accuracy.</p>
  <div class="wp-btns">
    <a class="wp-btn primary" href="#one-layer"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M3 1.5v10l8.5-5z" fill="currentColor"/></svg>Watch one decoder layer</a>
    <a class="wp-btn ghost" href="#explore"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><circle cx="3" cy="9.5" r="1.7" fill="currentColor"/><circle cx="6.5" cy="5.5" r="1.7" fill="currentColor"/><circle cx="10.5" cy="3" r="1.7" fill="currentColor"/></svg>Explore the results</a>
    <span class="wp-btn soon" aria-disabled="true"><svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true"><path d="M3 1h5l3 3v8H3z M8 1v3h3" fill="none" stroke="currentColor" stroke-width="1.3"/></svg>Paper · under review</span>
  </div>
  <div class="wp-ctl">
    <span class="wp-seg" role="group" aria-label="Machine"><button type="button" data-wp-plat="icl">Ice Lake</button><button type="button" data-wp-plat="emr">Emerald Rapids</button></span>
    <span class="wp-seg" role="group" aria-label="Model"><button type="button" data-wp-model="llama3_8b">Llama3-8B</button><button type="button" data-wp-model="falcon3_10b">Falcon3-10B</button><button type="button" data-wp-model="falcon3_1b">Falcon3-1B</button></span>
    <button type="button" class="wp-again" id="wp-race-again">↻ Replay</button>
  </div>
  <div class="wp-race" id="wp-race" aria-label="Replay of the measured decode speeds of four implementations">
    <div class="lane" data-v="orig" style="--c:var(--c-ik)"><div class="who"><b>ik</b><span>ik_llama.cpp, all cores</span></div><div class="track"><i></i><span class="cnt"></span></div><div class="rate"><b>–</b> tok/s<em></em></div></div>
    <div class="lane" data-v="a2" style="--c:var(--c-ikp)"><div class="who"><b>ik-P</b><span>ik + our four patches</span></div><div class="track"><i></i><span class="cnt"></span></div><div class="rate"><b>–</b> tok/s<em></em></div></div>
    <div class="lane" data-v="a3" style="--c:var(--c-row)"><div class="who"><b>Row</b><span>our engine, row-owned</span></div><div class="track"><i></i><span class="cnt"></span></div><div class="rate"><b>–</b> tok/s<em></em></div></div>
    <div class="lane" data-v="a3col" style="--c:var(--c-col)"><div class="who"><b>Col</b><span>our engine, column-split</span></div><div class="track"><i></i><span class="cnt"></span></div><div class="rate"><b>–</b> tok/s<em></em></div></div>
    <div class="cap"><b id="wp-race-cap"></b><br>A real-time replay of the measured decode speeds (medians of 6 interleaved rounds, context depth 0). No model runs in your browser.</div>
  </div>
</section>

<div class="wp-stats">
  <div class="wp-stat" style="--accent:var(--c-wait)"><div class="num">20–48%</div><div class="lab">of each token is spent waiting at synchronization points, with node-local weights and two global barriers per layer</div></div>
  <div class="wp-stat" style="--accent:var(--c-col)"><div class="num">1.5–2.1×</div><div class="lab">faster than ik_llama.cpp with all cores where decoding is DRAM-bound; 3.2× for a small model that runs from the L3 cache</div></div>
  <div class="wp-stat" style="--accent:var(--c-ikp)"><div class="num">63–78%</div><div class="lab">of the machine's read bandwidth is used by Col, against 24–46% for ik_llama.cpp</div></div>
  <div class="wp-stat" style="--accent:var(--c-ok)"><div class="num">29–64%</div><div class="lab">less energy per token than ik_llama.cpp, at the same accuracy</div></div>
</div>

<details class="wp-abs"><summary>Abstract</summary><p>Ternary LLMs read only a few gigabytes per decoded token, yet on dual-socket servers with four NUMA nodes the fastest ternary runtime we found uses less than half of the memory bandwidth. Placing each weight on the node of the thread that reads it is necessary but not enough: in a decode engine with node-local weights that reuses ik_llama.cpp's kernels, threads on two generations of Xeon servers still spend 20–48% of each token waiting at synchronization points. Per-thread barrier traces show that most of the wait is for the slowest threads of each node; the barriers' own latency (microseconds each, but up to 12% of the token for the smallest model) and serial per-node work are smaller parts. On identical kernels, a Megatron-style column split beats a row-owned split without cross-node reductions in every configuration we measured, and isolating the number of barriers shows that most of its gain comes from its threads waiting less for each other and for their node leaders, and only a minority from having fewer barriers. Where decoding is DRAM-bound, the resulting engine is 1.5–2.1× faster than ik_llama.cpp with all cores and 1.5–1.9× faster than ik_llama.cpp at its fastest thread count (3.2× and 2.8× for a small model that runs from the L3 cache), at the same accuracy and with less energy per token.</p></details>

<h2 id="the-problem">The problem</h2>

<p class="wp-lead"><strong>Ternary models</strong> such as BitNet b1.58 restrict their weights to −1, 0 and +1 and store them in about two bits each. An eight-billion-parameter model then reads about 2 GB per generated token, and since each token reads all weights once, decoding speed is set by how fast memory can be read. On <strong>one NUMA node</strong> this works well: the best runtimes use 72–78% of the node's bandwidth.</p>

<p class="wp-lead">Using the <strong>whole machine</strong> is harder. A dual-socket server exposes four NUMA nodes, and across them ik_llama.cpp reads DRAM at only 48% of the machine's bandwidth on our Ice Lake server (Llama3-8B). On our Emerald Rapids server it becomes 3–12% <em>slower</em> when given 104 cores instead of 64. The usual diagnosis is weight placement: a thread that reads weights stored on another node reads them slowly. We ask what limits decoding <em>after</em> placement is solved.</p>

<div class="wp-grid3">
  <div class="wp-card" style="--accent:var(--c-ikp)"><div class="h"><i></i>Machines</div>Two dual-socket Intel servers with four NUMA nodes each. <strong>Ice Lake</strong>: 2 × Xeon Platinum 8358, 64 cores, 354 GB/s of local read bandwidth. <strong>Emerald Rapids</strong>: 2 × Xeon Platinum 8555C, 104 cores, 610 GB/s, and 5.4× the L3 cache. Both are shared machines running Linux 6.8.</div>
  <div class="wp-card" style="--accent:var(--c-row)"><div class="h"><i></i>Models</div>Three ternary Llama-architecture models: Llama3-8B-1.58, Falcon3-10B-1.58bit and Falcon3-1B-1.58bit. In the exact 2-bit format IQ2_BN, one token reads 2.19, 2.73 and 0.51 GB. For the format study, the same models also run as Q2_K, Q4_0 and Q4_K_M.</div>
  <div class="wp-card" style="--accent:var(--c-col)"><div class="h"><i></i>Measurement</div>The rate of generating 128 tokens, with one thread per physical core. Everything a table or chart compares was measured together: each of 5–6 rounds runs every variant once in rotating order, and we report medians and the median of the per-round ratios.</div>
</div>

<p class="wp-k">The four implementations compared</p>
<div class="wp-grid4">
  <div class="wp-card ev" style="--accent:var(--c-ik)"><div class="h">ik</div>ik_llama.cpp at a fixed commit, with <code>--numa distribute</code>. It already places 97% of the weight pages on the node of their reader. Our baseline.</div>
  <div class="wp-card ev" style="--accent:var(--c-ikp)"><div class="h">ik-P</div>ik with four patches: page migration, a two-level barrier, attention split over KV chunks, and an AVX-512 activation quantizer.</div>
  <div class="wp-card ev" style="--accent:var(--c-row)"><div class="h">Row</div>Our decode engine with a row-owned dataflow: every matrix is split by output rows, with five global barriers per layer and no cross-node sums.</div>
  <div class="wp-card ev" style="--accent:var(--c-col)"><div class="h">Col</div>Our engine with the Megatron-style column split: each node owns a quarter of the heads and of the FFN, with two global barriers per layer. The main design.</div>
</div>

<p class="wp-k">Terms used below</p>
<dl class="wp-terms">
  <div><dt>NUMA node.</dt> <dd>A group of cores with its own memory controllers. Here each socket appears as two nodes, so a server has four. Reading another node's memory is slower than reading one's own.</dd></div>
  <div><dt>Barrier.</dt> <dd>A point where threads wait until all of them have arrived. A <em>global</em> barrier gathers all threads of the machine, a <em>node</em> barrier only those of one node.</dd></div>
  <div><dt>Node leader.</dt> <dd>One thread per node that normalizes and quantizes the input of each matrix stage alone, while the node's other threads wait for it.</dd></div>
  <div><dt>Waiting for the last thread.</dt> <dd>The time from a thread's own arrival at a barrier until the last thread arrives. It measures how unevenly the threads finish.</dd></div>
  <div><dt>Barrier latency.</dt> <dd>The time from the last arrival until a thread leaves the barrier: the cost of the barrier itself, a few microseconds.</dd></div>
  <div><dt>DRAM-bound.</dt> <dd>The weights do not fit in the caches, so every token streams them from memory. This holds for all models here except Falcon3-1B on Emerald Rapids, which runs from L3.</dd></div>
  <div><dt>Effective bandwidth.</dt> <dd>Decode speed times the bytes read per token. As a share of the machine's measured read bandwidth, it shows how much of the memory system a runtime uses, whatever its kernels.</dd></div>
  <div><dt>Context depth.</dt> <dd>The number of tokens already in the KV cache when the 128 timed tokens are generated. It is 0 unless stated.</dd></div>
</dl>

<h2 id="where-the-time-goes">Where the time goes</h2>

<p class="wp-k">Finding 1</p>
<h3 class="wp-h" id="placement-first">Placement is nearly solved in ik; synchronization is the larger step</h3>
<p class="wp-lead">We add our changes to ik one at a time (the step chart under <a href="#explore">Results</a> shows every step). Migrating the 3% of misplaced weight pages gains only 0–3%, and 6–14% on the small Falcon3-1B. The largest single step is synchronization: replacing ggml's global barrier by a two-level one, where threads first meet within their node and then one thread per node meets globally, gains 10–14% on Ice Lake and 20–44% on Emerald Rapids. Together the four patches give 1.20–1.73×, and further changes inside ggml gain nothing. What remains is its execution model: one parallel region and one global barrier per operator, about 17 per layer.</p>

<p class="wp-k">Finding 2 · Interactive</p>
<h3 class="wp-h" id="one-layer">One decoder layer, thread by thread</h3>
<p class="wp-lead">Our engine keeps ik's kernels and weights but groups a layer into a few stages. It also records when every thread enters and leaves every barrier. Below is one of the 32 layers of one token on Emerald Rapids. Each thin row is one of 104 threads; colour means the thread is computing, grey means it is waiting. <strong>Row</strong> meets globally five times per layer (▼) and <strong>Col</strong> twice. In both, the grey areas are large: threads of the same node read equal shares of node-local weights and still finish at different times.</p>
<figure class="wp-fig" style="--accent:var(--c-wait)">
  <div class="wp-ctl">
    <button type="button" class="wp-again" id="wp-trace-play">▶ Play in slow motion (about 50,000×)</button>
    <span>Move over a thread to see what it is doing.</span>
  </div>
  <div class="wp-trace"><canvas id="wp-trace" role="img" aria-label="Per-thread timeline of one decoder layer for the Row and Col dataflows: 104 threads on 4 nodes, coloured while computing and grey while waiting. Row finishes the layer in 166 microseconds with 5 global barriers, Col in 135 microseconds with 2."></canvas></div>
  <div class="wp-keys" aria-hidden="true">
    <span><i style="background:#56B4E9"></i>QKV + RoPE</span><span><i style="background:#009E73"></i>attention</span><span><i style="background:#F0E442"></i>O projection</span><span><i style="background:#E69F00"></i>gate/up + SiLU</span><span><i style="background:#D55E00"></i>down projection</span><span><i style="background:#CC79A7"></i>norm + quantization (leader)</span>
    <span><i style="background:#D4D4D4"></i>waiting for the last thread</span><span><i style="background:#9C9C9C"></i>waiting for the node leader</span><span><i style="background:#4A4A4A"></i>barrier latency</span><span>▼ global barrier</span>
  </div>
  <p class="wp-out" id="wp-trace-out" aria-live="polite"></p>
  <div class="wp-split" id="wp-split" aria-label="Per-token time of Row and Col for Llama3-8B, split into computing and three kinds of waiting"></div>
  <figcaption><strong>Figure 1 (measured).</strong> Top: layer 16 of the median-duration token of a traced run (Emerald Rapids, Llama3-8B in IQ2_BN, 104 threads on 4 nodes). Bottom: the whole token of Llama3-8B split the same way, as medians of 5 traced rounds (blue: computing; greys: the three kinds of waiting).
    <span class="take">With all weights node-local, a quarter to more than a third of the token is still waiting, most of it for the last thread to arrive.</span></figcaption>
  <details class="wp-more"><summary>Static version (Figure 1 of the paper)</summary><div class="img"><img src="/images/projects/2026-11-weight-placement/fig1-teaser.png" alt="Figure 1 of the paper: per-thread timeline of one decoder layer for Row and Col, and the per-token time split on both machines." loading="lazy"></div></details>
</figure>

<p class="wp-k">Finding 3</p>
<h3 class="wp-h" id="what-the-wait-is">Most of the wait is for the slowest threads of a thread's own node</h3>
<p class="wp-lead">Across models and weight formats, Col's threads wait for 22–35% of each token on Ice Lake and 20–48% on Emerald Rapids. The traces split this wait into three parts. The share grows with the number of threads per barrier: with 13, 20 and 26 cores per node on Emerald Rapids, threads wait for 18–31, 23–42 and 29–48% of the token.</p>
<div class="wp-grid3">
  <div class="wp-card ev" style="--accent:#b9b8b1"><div class="num">65–86%</div><div class="h">Waiting for the last thread</div>This is 17–26% of the token on Ice Lake and 14–31% on Emerald Rapids. More than half of the late arrivals are threads of the waiting thread's own node.</div>
  <div class="wp-card ev" style="--accent:#4a4a4a"><div class="num">1–12%</div><div class="h">Barrier latency</div>Each barrier costs microseconds, but Falcon3-1B has 18 thin layers, and on Emerald Rapids latency alone takes 9–12% of its token.</div>
  <div class="wp-card ev" style="--accent:#9c9c9c"><div class="num">2–5%</div><div class="h">Waiting for the node leader</div>The serial normalization and quantization of each stage's input by one thread per node.</div>
</div>
<figure class="wp-fig" style="--accent:var(--c-wait)">
  <div class="img"><a href="/images/projects/2026-11-weight-placement/fig2-wait-share.png" title="Open the figure at full size"><img src="/images/projects/2026-11-weight-placement/fig2-wait-share.png" alt="Figure 2: waiting at synchronization points as a share of Col's per-token decode time, for three models in up to four weight formats on both machines." loading="lazy"></a></div>
  <figcaption><strong>Figure 2.</strong> Waiting at synchronization points as a share of Col's per-token decode time (mean over threads; median of 5 interleaved rounds, whiskers: min–max), split into waiting for the node leader (dark), for the last thread (medium) and barrier latency (light).
    <span class="take">The model that reads the fewest bytes per token, Falcon3-1B, waits the most: nearly half of each token on Emerald Rapids.</span></figcaption>
</figure>

<p class="wp-k">Finding 4</p>
<h3 class="wp-h" id="fewer-bytes">Fewer bytes per token, more synchronization</h3>
<p class="wp-lead">The same models in larger formats read 1.3–2.1× as many bytes per token. On Emerald Rapids, going from IQ2_BN to Q4_K_M lowers Col's wait share from 34 to 22% for Llama3-8B and from 48 to 39% for Falcon3-1B, and its speedup over ik from 2.05 to 1.53× and from 3.12 to 2.57×. On Ice Lake the relation is weak. Across models the order holds on both machines: the model with the fewest bytes per token has the largest wait share and the largest speedup in every format. The effect is not specific to ternary weights: with 4-bit formats Col is still 1.40–2.57× faster than ik.</p>
<figure class="wp-fig" style="--accent:var(--c-col)">
  <div class="img narrow"><a href="/images/projects/2026-11-weight-placement/fig3-bytes.png" title="Open the figure at full size"><img src="/images/projects/2026-11-weight-placement/fig3-bytes.png" alt="Figure 3: decode speedup of Col over ik against the gigabytes read per token, for three models in four formats on both machines." loading="lazy"></a></div>
  <figcaption><strong>Figure 3.</strong> Decode speedup of Col over ik against the bytes read per token, three models in four formats (median of per-round ratios, 5 rounds).</figcaption>
</figure>

<h2 id="two-dataflows">Two dataflows on the same kernels</h2>

<p class="wp-lead">To find which property of a dataflow matters, the engine runs two of them on ik's matrix–vector kernels, ik's quantizers and the same weight files. They differ in how often all threads meet and in how much data crosses nodes.</p>

<figure class="wp-fig" style="--accent:var(--c-col)">
  <div class="wp-flowwrap">
  <div class="wp-flow" role="img" aria-label="One layer in the two dataflows. Row has five stages, each ending in a global barrier. Col has two global barriers, each followed by a sum of the nodes' partial results, and six node barriers.">
    <div class="lane"><b>Row</b><div class="st"><span style="flex:10">QKV</span><u></u><span style="flex:11">attention</span><u></u><span style="flex:6">comb.</span><s></s><span style="flex:9">O + res.</span><u></u><span style="flex:10">gate/up</span><u></u><span style="flex:11">down + res.</span><u></u></div></div>
    <div class="hint">Every thread writes its output rows into all 4 node copies of the vector.</div>
    <div class="lane"><b>Col</b><div class="st"><span style="flex:8">QKV<sub>h</sub></span><s></s><span style="flex:8">attn.<sub>h</sub></span><s></s><span style="flex:5">comb.</span><s></s><span style="flex:8">O<sub>h</sub></span><u></u><span class="sum" style="flex:3.5">Σ</span><s></s><span style="flex:10">gate/up<sub>f</sub></span><s></s><span style="flex:10">down<sub>f</sub></span><u></u><span class="sum" style="flex:3.5">Σ</span><s></s></div></div>
    <div class="hint">Outputs stay on the node; Σ adds the four nodes' partial sums after a global barrier.</div>
    <div class="key"><span><u></u>global barrier (all threads)</span><span><s></s>node barrier (one node's threads)</span><span><sub>h</sub> the node's quarter of the heads · <sub>f</sub> its quarter of the FFN</span></div>
  </div>
  </div>
  <div class="wp-steps">
    <div class="wp-step"><div class="t"><span>1</span>The engine</div>One thread is pinned to each physical core. At load time every weight matrix is cut into one slice per node and one range of rows per thread, and each slice is copied into memory of its node, as are the KV cache and the buffers. Barriers are two-level.</div>
    <div class="wp-step"><div class="t"><span>2</span>Row: split by output rows</div>All threads of all nodes share every matrix by rows. A layer has five stages, each ending in a global barrier, and no cross-node sums; instead every thread writes its rows into a copy of the vector on every node.</div>
    <div class="wp-step"><div class="t"><span>3</span>Col: split by heads and columns</div>The Megatron-LM decomposition, which ArcLight applies to the NUMA nodes of a CPU. A node computes its heads and its part of the FFN with node barriers only. O and down yield partial sums, which every node adds up after each of two global barriers.</div>
  </div>
  <figcaption><strong>Figure 4.</strong> One layer in the two dataflows. Each matrix stage starts with the node leader's normalization and quantization (not shown). Col's two sums cost each node three remote reads of a hidden-size vector (16 KiB for Llama3-8B).</figcaption>
</figure>

<p class="wp-k">Comparison</p>
<h3 class="wp-h" id="col-beats-row">Col is faster than Row in all 24 configurations</h3>
<p class="wp-lead">The gain is 5–21% on Ice Lake and 11–46% on Emerald Rapids. It is largest for the model with the fewest bytes per token, and it persists with a context of 4096 tokens.</p>
<figure class="wp-fig" style="--accent:var(--c-col)">
  <div class="wp-tablewrap">
  <table class="wp-table">
    <thead>
      <tr><th></th><th class="grp" colspan="2">IQ2_BN (2-bit, ternary)</th><th class="grp" colspan="2">Q4_0 (4-bit)</th></tr>
      <tr><th>Col speed relative to Row</th><th>depth 0</th><th>depth 4096</th><th>depth 0</th><th>depth 4096</th></tr>
    </thead>
    <tbody>
      <tr><td>Ice Lake · Llama3-8B</td><td>1.09×</td><td>1.06×</td><td>1.05×</td><td>1.06×</td></tr>
      <tr><td>Ice Lake · Falcon3-10B</td><td>1.11×</td><td>1.11×</td><td>1.11×</td><td>1.08×</td></tr>
      <tr><td>Ice Lake · Falcon3-1B</td><td>1.21×</td><td>1.19×</td><td>1.15×</td><td>1.16×</td></tr>
      <tr class="sep"><td>Emerald Rapids · Llama3-8B</td><td>1.21×</td><td>1.18×</td><td>1.14×</td><td>1.11×</td></tr>
      <tr><td>Emerald Rapids · Falcon3-10B</td><td>1.20×</td><td>1.24×</td><td>1.11×</td><td>1.20×</td></tr>
      <tr><td>Emerald Rapids · Falcon3-1B</td><td>1.46×</td><td>1.41×</td><td>1.32×</td><td>1.31×</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Table 3.</strong> Decode speed of Col relative to Row (median of per-round ratios over 5 interleaved rounds; 128 tokens generated from depth 0 or 4096).</figcaption>
</figure>

<p class="wp-k">Why</p>
<h3 class="wp-h" id="why-col-wins">The number of barriers explains only a minority of the gap</h3>
<p class="wp-lead">Row and Col differ in the number of global barriers and in where they read and write, so we separate the two inside the engine. Promoting three of Col's node barriers to global ones gives it Row's five global barriers per layer with its own data placement; adding empty global barriers right after an existing one measures the latency of a barrier with no imbalance to absorb. Neither changes the computed values.</p>
<div class="wp-grid4">
  <div class="wp-card ev" style="--accent:var(--c-wait)"><div class="num">5–26%</div><div class="h">of the gap from barrier count</div>Five global barriers per layer instead of two slow Col by this share of the Row–Col difference (Emerald Rapids).</div>
  <div class="wp-card ev" style="--accent:#4a4a4a"><div class="num">1.4–4.4 µs</div><div class="h">per empty global barrier</div>Among 104 threads. Col's two per layer therefore take 1–10% of its token, a small part of its 20–48% wait.</div>
  <div class="wp-card ev" style="--accent:#56b4e9"><div class="num">36–57%</div><div class="h">shorter QKV stage</div>In Col a node's heads, queries and KV cache stay on the node. In Row most threads write K, V and Q across nodes.</div>
  <div class="wp-card ev" style="--accent:#cc79a7"><div class="num">39–48%</div><div class="h">of the leader's work</div>Col's leaders quantize this share of the elements Row's do per layer, since the inputs of O and down are the node's own slices.</div>
</div>
<figure class="wp-fig" style="--accent:var(--c-col)">
  <div class="img"><a href="/images/projects/2026-11-weight-placement/fig5-stages.png" title="Open the figure at full size"><img src="/images/projects/2026-11-weight-placement/fig5-stages.png" alt="Figure 5: per-token decode time of the engine by stage for Row and Col, three models on both machines." loading="lazy"></a></div>
  <figcaption><strong>Figure 5.</strong> Per-token decode time of the engine by stage (mean over threads; IQ2_BN, depth 0; median of 5 rounds). In Col, O and down include the sum of the partial results.
    <span class="take">Two parts shrink from Row to Col: the barrier wait (by 7–23% on Ice Lake and 25–35% on Emerald Rapids) and the QKV stage. The sums add at most 0.22 ms per token, while the token gets 0.38–1.45 ms shorter.</span></figcaption>
</figure>
<p class="wp-lead">The traces attribute most of the remaining gap to Row's longer waits for late threads and for the leaders. Row's cross-node writes of the stage outputs do not explain them: a variant of Row that writes those outputs only to its own node and copies the rest after the barrier, with bit-identical results, is 1–7% slower and waits as long for late threads.</p>

<h2 id="results">Results</h2>

<p class="wp-k">Interactive</p>
<h3 class="wp-h" id="explore">Explore every machine and model</h3>
<p class="wp-lead">Pick a machine and a model. The step chart shows what each change adds, starting from ik. The bandwidth chart shows how much of the memory system each implementation uses. The two line charts show where the advantage comes from (more nodes) and that it holds with long contexts.</p>
<figure class="wp-fig" id="wp-explore" style="--accent:var(--c-col)">
  <div class="wp-ctl">
    <span class="wp-seg" role="group" aria-label="Machine"><button type="button" data-wp-plat="icl">Ice Lake</button><button type="button" data-wp-plat="emr">Emerald Rapids</button></span>
    <span class="wp-seg" role="group" aria-label="Model"><button type="button" data-wp-model="llama3_8b">Llama3-8B</button><button type="button" data-wp-model="falcon3_10b">Falcon3-10B</button><button type="button" data-wp-model="falcon3_1b">Falcon3-1B</button></span>
  </div>
  <div class="wp-charts">
    <div><div class="wp-ct">Decode speed, step by step <span>· tok/s and speed relative to ik</span></div><div class="wp-bars" id="wp-ladder"></div></div>
    <div><div class="wp-ct">Share of the machine's read bandwidth</div><div class="wp-bars short" id="wp-bw"></div></div>
  </div>
  <div class="wp-charts even">
    <div><div class="wp-ct">Scaling over NUMA nodes <span>· tok/s; legend: speedup from 1 to 4 nodes</span></div><div class="wp-line" id="wp-scale"></div></div>
    <div><div class="wp-ct">Long contexts <span>· tok/s; legend: tok/s at depth 8192</span></div><div class="wp-line" id="wp-depth"></div></div>
  </div>
  <figcaption><span id="wp-readout"></span>
    <span class="take">On one node all four implementations are within 17% of each other and reach 70–84% of the node's bandwidth. From one node to four, ik speeds up 1.3–2.4× and Col 3.0–3.7×: the engine's advantage arises across nodes.</span></figcaption>
  <p class="wp-scope" style="margin:8px 4px 0">Data of Table 2 and Figures 6–8 of the paper; 128 generated tokens, IQ2_BN, medians over 5–6 interleaved rounds. Each chart is its own batch, so the same configuration can differ by a few percent between charts. Each added step in the first chart includes the steps above it.</p>
</figure>

<p class="wp-k">Other runtimes</p>
<h3 class="wp-h" id="arclight">Against ArcLight, which uses the same decomposition</h3>
<p class="wp-lead">ArcLight implements the column split on top of llama.cpp's operators and kernels. It supports F16, Q4_0, Q8_0 and Q6_K weights, so the comparison uses Q4_0, the format it was evaluated with. On one node of Emerald Rapids, Col is only 1.05–1.23× faster than ArcLight; on four nodes in the same batch it is 1.81–3.59× faster. The decomposition alone does not deliver the speedup: across nodes, ArcLight executes llama.cpp's operator graph with node-level barriers inside its tensor-parallel subgraphs and global ones at their boundaries, whereas Col runs each stage as one parallel region.</p>
<figure class="wp-fig" style="--accent:var(--c-col)">
  <div class="wp-tablewrap">
  <table class="wp-table">
    <thead>
      <tr><th></th><th class="grp" colspan="3">Ice Lake, 64 threads</th><th class="grp" colspan="3">Emerald Rapids, 104 threads</th></tr>
      <tr><th>Q4_0 weights, tok/s</th><th>Llama3-8B</th><th>Falcon3-10B</th><th>Falcon3-1B</th><th>Llama3-8B</th><th>Falcon3-10B</th><th>Falcon3-1B</th></tr>
    </thead>
    <tbody>
      <tr><td>llama.cpp (<code>distribute</code>)</td><td>17.0</td><td>12.8</td><td>84.9</td><td>60.6</td><td>48.3</td><td>195.0</td></tr>
      <tr><td><span class="dot" style="background:var(--c-ik)"></span>ik</td><td>41.6</td><td>32.7</td><td>162.9</td><td>68.4</td><td>48.9</td><td>225.1</td></tr>
      <tr><td>ArcLight</td><td>38.9</td><td>32.8</td><td>119.6</td><td>56.3</td><td>45.5</td><td>162.3</td></tr>
      <tr><td><span class="dot" style="background:var(--c-ikp)"></span>ik-P</td><td>44.7</td><td>38.3</td><td>193.1</td><td>80.3</td><td>60.9</td><td>319.0</td></tr>
      <tr><td><span class="dot" style="background:var(--c-row)"></span>Row</td><td>55.7</td><td>40.5</td><td>234.6</td><td>88.7</td><td>72.9</td><td>392.3</td></tr>
      <tr class="hl"><td><span class="dot" style="background:var(--c-col)"></span>Col</td><td>59.4</td><td>45.6</td><td>270.5</td><td>101.6</td><td>77.8</td><td>528.4</td></tr>
      <tr class="sep"><td>Col / ArcLight</td><td>1.53×</td><td>1.39×</td><td>2.25×</td><td>1.75×</td><td>1.74×</td><td>3.23×</td></tr>
    </tbody>
  </table>
  </div>
  <figcaption><strong>Table 5.</strong> Decode speed with Q4_0 weights (tok/s, depth 0; median of 5 interleaved rounds). ArcLight was built for AVX2; a build with <code>-march=native</code> was within 3%. The last row is the median of the per-round ratios. llama.cpp is unusually slow on Ice Lake because it repacks Q4_0 weights into a buffer on one node.</figcaption>
</figure>

<p class="wp-k">More measurements</p>
<h3 class="wp-h" id="more-measurements">Thread count, context depth, energy and accuracy</h3>
<div class="wp-grid2">
  <div class="wp-card ev" style="--accent:var(--c-ik)"><div class="num">1.5–1.9×</div><div class="h">against ik at its fastest thread count</div>ik is not always fastest with all cores: on Emerald Rapids it is 6–14% faster with 64 or 80 of its 104 cores. Against the best count per model, Col is 1.53–1.93× faster on Ice Lake and 1.89–2.78× on Emerald Rapids (2.78× is Falcon3-1B, which runs from L3).</div>
  <div class="wp-card ev" style="--accent:var(--c-ikp)"><div class="num">up to 76%</div><div class="h">faster ik with an interleaved KV cache</div>ik allocates its whole KV cache on one node. Interleaving it with <code>numactl</code> helps at depth 8192 and hurts at depth 128. Against the faster of the two at each depth, Col is 1.66–2.49× faster on Ice Lake and 2.02–2.93× on Emerald Rapids at depth 8192.</div>
  <div class="wp-card ev" style="--accent:var(--c-ok)"><div class="num">29–64%</div><div class="h">less energy per token</div>Measured with RAPL over the whole machine (package and DRAM): 29–45% less than ik on Ice Lake, 39–64% less on Emerald Rapids. There Col draws 16–29% more package power, so the saving is smaller than the speedup.</div>
  <div class="wp-card ev" style="--accent:var(--c-col)"><div class="num">same accuracy</div><div class="h">within the spread of ik's own decode paths</div>On 5100 WikiText-2 tokens, ik's two decode paths have KL divergences of 0.0030–0.0045 from an F16 reference with the same ternary weights. Row falls within this spread and Col has the lowest of all paths (0.0020–0.0034).</div>
</div>
<figure class="wp-fig" style="--accent:var(--c-ikp)">
  <div class="img"><a href="/images/projects/2026-11-weight-placement/fig8-depth.png" title="Open the figure at full size"><img src="/images/projects/2026-11-weight-placement/fig8-depth.png" alt="Figure 8: decode speed after contexts of 128 to 8192 tokens for ik with default and interleaved KV cache, ik-P and Col, three models on both machines." loading="lazy"></a></div>
  <figcaption><strong>Figure 8.</strong> Decode speed after a context of the given depth, all six configurations (5 interleaved rounds). ik with its default KV cache (on one node) and with the cache interleaved over all nodes.
    <span class="take">Compared only with ik's default placement, the depth-8192 speedups would have been overstated as 2.0–5.2×. A baseline's KV cache has to be placed as carefully as its weights.</span></figcaption>
</figure>

<h2 id="takeaways">Takeaways for CPU runtimes</h2>

<div class="wp-grid2">
  <div class="wp-card"><div class="h"><span>1</span>Group a layer into a few stages</div>On multi-node machines, one global barrier per operator, with single-row operators on one thread, costs more than misplaced weights. Beyond that, how the work is divided among threads matters more than the barrier count.</div>
  <div class="wp-card"><div class="h"><span>2</span>Balance the work within a node</div>Most of the remaining wait is imbalance among a node's threads, which fixed row ranges cannot absorb. Assigning rows dynamically within a node is the direct remedy to test.</div>
  <div class="wp-card"><div class="h"><span>3</span>Place the KV cache like the weights</div>An interleave alone makes ik up to 76% faster at depth 8192 than its single-node default.</div>
  <div class="wp-card"><div class="h"><span>4</span>Thresholds tuned on one machine break on another</div>ik's rule for splitting single-query attention leaves most threads idle at short context for models with four KV heads and, at 104 threads, also for Llama3-8B.</div>
</div>

<p class="wp-scope"><strong>Scope and limitations.</strong> Col supports Llama-architecture models whose head, KV-head and FFN counts are divisible by the number of nodes, and it decodes a single sequence without prefill or batching; it targets latency-bound single-user settings. Machines with more than four NUMA nodes, AMD systems and Intel systems without sub-NUMA clustering are outside our measurements. Both servers were shared with other users: variants alternate within every round, so comparisons between implementations are affected little, but absolute wait shares on Ice Lake may be a few points high.</p>

<h2 id="links-and-citation">Links and citation</h2>

<div class="wp-links">
  <div class="wp-link"><div class="h">Paper</div><div class="s">Under review. The PDF will be linked here when it is public.</div></div>
  <div class="wp-link"><div class="h">Authors</div><div class="s">Runguo Li and Dezhen Lu contributed equally.</div></div>
</div>

<div class="wp-bib">
<button class="wp-copy" type="button" data-target="wp-bibtex">Copy</button>
<pre><code id="wp-bibtex">@misc{li2026weightplacement,
  title  = {Weight Placement Is Not Enough: Synchronization in Multi-Node {CPU} Decoding of Ternary {LLMs}},
  author = {Li, Runguo and Lu, Dezhen},
  year   = {2026},
  note   = {Manuscript under review}
}</code></pre>
</div>

<div class="wp-tip" id="wp-tip" hidden></div>

</div>

<script src="/files/projects/2026-11-weight-placement/data.js" defer></script>
<script src="/files/projects/2026-11-weight-placement/page.js" defer></script>
