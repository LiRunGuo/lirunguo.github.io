---
title: "What Do Group-Conditional Coverage Metrics Measure? Identification, Inference and Auditing for Conformal Evaluation"
collection: projects
permalink: /projects/2026-11-group-coverage/
date: 2026-11-02
# No venue under review is named on this page; paper, code and BibTeX links to be added once the preprint is public.
status: "Under review"
venue: "Under review"
excerpt: "Group-coverage metrics for conformal methods are computed from a handful of test points per group, so much of a reported gap is binomial test noise. The paper shows which metrics a test set identifies, gives a debiased estimator with intervals and a group audit, and measures the correction on seven held-out datasets."
teaser: "/images/projects/2026-11-group-coverage/teaser.png"
tags: ["conformal prediction", "group-conditional coverage", "evaluation", "pre-registration"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-group-coverage"
published: false
---

<style>
/* Group-coverage project page: styles scoped to .gc / .gc-* so the rest of the site is untouched; colors follow the theme */
.gc{--gc-blue:#1f5fa8;--gc-blue-soft:rgba(31,95,168,.08);--gc-gray:#6f767e;--gc-gray-soft:rgba(127,127,127,.09);--gc-green:#1f8a3b;--gc-red:#d2452d;--gc-red-soft:rgba(214,69,45,.09);--gc-line:rgba(127,127,127,.28);--gc-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .gc{--gc-blue:#78b0f0;--gc-blue-soft:rgba(120,176,240,.14);--gc-gray:#c3c9cf;--gc-gray-soft:rgba(255,255,255,.07);--gc-green:#62cf72;--gc-red:#f47f68;--gc-red-soft:rgba(244,127,104,.13);--gc-line:rgba(255,255,255,.2);--gc-card:rgba(255,255,255,.04)}
.gc-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.gc-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--gc-blue-soft);color:var(--gc-blue)}
.gc-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.gc-tldr b{color:var(--gc-blue)}
.gc-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.gc-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.gc-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.gc-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--gc-line)}
.gc-btn.muted{background:transparent;color:var(--gc-gray)!important;border:1px dashed var(--gc-line);cursor:default}
.gc-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .gc-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.gc-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--gc-line);border-radius:14px;background:var(--gc-card)}
.gc-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.gc-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--gc-blue);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .gc-tag{color:#16202b}
.gc-fig figcaption b.gc-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.gc-lead{font-size:1em;line-height:1.65;margin:0 0 1em}
.gc-ix{font-size:.9em}
.gc-ctl{display:grid;grid-template-columns:auto 1fr;gap:.45em .9em;align-items:center;margin:0 0 .9em}
.gc-ctl .l{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65}
.gc-seg{display:flex;flex-wrap:wrap;gap:.35em}
.gc-seg button{font:inherit;font-size:.86em;line-height:1.3;padding:.3em .85em;border-radius:999px;border:1px solid var(--gc-line);background:transparent;color:var(--global-text-color);cursor:pointer;transition:background .15s,border-color .15s}
.gc-seg button:hover{border-color:var(--gc-gray)}
.gc-seg button[aria-pressed=true]{background:var(--global-text-color);border-color:var(--global-text-color);color:var(--global-bg-color)}
.gc-seg button.go{border-color:var(--gc-blue);color:var(--gc-blue);font-weight:700}
.gc-range{display:flex;align-items:center;gap:.8em}
.gc-range input{flex:1;max-width:340px;accent-color:var(--gc-blue)}
.gc-range output{font-weight:800;font-variant-numeric:tabular-nums;min-width:3.2em}
.gc-read{margin:.8em 0 0;padding:.5em .8em;border-left:3px solid var(--gc-gray);border-radius:4px;background:var(--gc-gray-soft);font-size:.95em;line-height:1.5}
.gc-read.b{border-left-color:var(--gc-blue);background:var(--gc-blue-soft)}
/* histogram of what a group shows */
.gc-hist{position:relative;height:170px;margin:2.5em 5% 0;border-bottom:2px solid var(--gc-line)}
.gc-hist .bar{position:absolute;bottom:0;transform:translateX(-50%);background:var(--gc-gray);border-radius:3px 3px 0 0;opacity:.8;transition:height .25s}
.gc-hist .bar.low{background:var(--gc-red);opacity:.9}
.gc-hist .bar i{position:absolute;left:50%;bottom:100%;transform:translateX(-50%);font-style:normal;font-size:.72em;font-weight:700;white-space:nowrap;padding-bottom:1px}
.gc-hist .vl{position:absolute;top:-18px;bottom:0;border-left:2px solid var(--global-text-color)}
.gc-hist .vl.cut{border-left:2px dashed var(--gc-red)}
.gc-hist .vl span{position:absolute;top:-4px;font-size:.72em;font-weight:700;white-space:nowrap}
.gc-hist .vl.tgt span{left:5px}.gc-hist .vl.tgt.hi span{left:auto;right:-14px;top:-17px}.gc-hist .vl.cut span{right:5px;color:var(--gc-red)}
.gc-haxis{position:relative;height:2.6em;margin:0 5%;font-size:.74em;opacity:.8}
.gc-haxis span{position:absolute;top:.3em;transform:translateX(-50%)}
.gc-haxis em{position:absolute;left:0;right:0;top:1.6em;text-align:center;font-style:normal}
.gc-tiles{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6em;margin:.6em 0 0}
.gc-tile{padding:.7em .8em;border-radius:12px;border:1px solid var(--gc-line);background:var(--global-bg-color)}
.gc-tile .v{font-size:1.5em;font-weight:800;line-height:1.1;color:var(--gc-red);font-variant-numeric:tabular-nums}
.gc-tile .d{font-size:.84em;line-height:1.4;margin-top:.3em}
.gc-tile .t{font-size:.8em;margin-top:.25em;color:var(--gc-green);font-weight:700}
/* definitions, layers, cards */
.gc-defs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em;margin:0 0 1.6em}
.gc-def{padding:.65em .85em;border-radius:12px;border:1px solid var(--gc-line);background:var(--gc-card);font-size:.86em;line-height:1.5}
.gc-def b{color:var(--gc-blue)}
.gc-def>b:first-child{display:inline-block;margin-right:.35em}
.gc-def.wide{grid-column:1/-1}
.gc-layers{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:.5em;align-items:stretch;margin:0 0 1.6em;font-size:.86em;line-height:1.5}
.gc-layer{padding:.8em .9em;border-radius:14px;border:1px solid var(--gc-line);background:var(--gc-card)}
.gc-layer.on{border-color:var(--gc-blue);background:var(--gc-blue-soft)}
.gc-layer.noise{border-color:var(--gc-red);background:var(--gc-red-soft)}
.gc-layer .k{display:block;font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;opacity:.7}
.gc-layer .h{display:block;font-size:1.08em;font-weight:800;margin:.15em 0 .3em}
.gc-layers .arr{align-self:center;color:var(--gc-gray);font-size:1.2em}
.gc-eqline{text-align:center;font-size:1.05em;line-height:1.7;margin:.1em 0 1em;padding:.6em;border-radius:8px;background:var(--gc-gray-soft)}
.gc-eqline .t-n{color:var(--gc-gray);font-weight:700}.gc-eqline .t-b{color:var(--gc-blue);font-weight:700}.gc-eqline .t-r{color:var(--gc-red);font-weight:700}
.gc-ids{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.gc-id{padding:.9em .95em;border-radius:14px;border:1px solid var(--gc-line);background:var(--gc-card);font-size:.86em;line-height:1.55;position:relative;overflow:hidden}
.gc-id:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--gc-c)}
.gc-id .s{display:block;font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:var(--gc-c)}
.gc-id .h{display:block;font-size:1.06em;font-weight:800;margin:.2em 0 .35em}
/* two worlds */
.gc-worlds{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7em}
.gc-world{padding:.6em .7em .5em;border-radius:12px;border:1px solid var(--gc-line);background:var(--global-bg-color)}
.gc-world .wt{font-weight:800;font-size:.95em}
.gc-world .wv{font-size:.9em;margin-top:.2em}
.gc-world .wv b{font-size:1.25em;font-variant-numeric:tabular-nums}
.gc-lolly{position:relative;height:120px;margin:1.5em 6% 0;border-bottom:2px solid var(--gc-line)}
.gc-lolly .st{position:absolute;bottom:0;width:4px;margin-left:-2px;background:var(--gc-blue);border-radius:2px 2px 0 0;transition:height .25s,left .25s}
.gc-lolly .st.low{background:var(--gc-red)}
.gc-lolly .st i{position:absolute;left:50%;bottom:100%;transform:translateX(-50%);font-style:normal;font-size:.76em;font-weight:700;white-space:nowrap;padding-bottom:2px}
.gc-lolly .st:before{content:"";position:absolute;left:50%;top:-5px;width:12px;height:12px;margin-left:-6px;border-radius:50%;background:inherit}
.gc-lolly .st i{padding-bottom:7px}
.gc-lolly .vl{position:absolute;top:-14px;bottom:0;border-left:2px dashed var(--gc-red)}
.gc-same{margin:.7em 0 0;padding:.55em .8em;border-radius:10px;background:var(--gc-gray-soft);font-size:.92em;line-height:1.5;text-align:center}
.gc-same b{font-variant-numeric:tabular-nums}
/* steps */
.gc-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1em}
.gc-step{padding:.9em .95em;border-radius:14px;border:1px solid var(--gc-blue);background:var(--gc-blue-soft);font-size:.86em;line-height:1.55}
.gc-step .n{display:inline-grid;place-items:center;width:1.7em;height:1.7em;border-radius:50%;background:var(--gc-blue);color:#fff;font-weight:800;font-size:.9em;margin-right:.45em}
html[data-theme="dark"] .gc-step .n{color:#16202b}
.gc-step .h{font-size:1.05em}
.gc-step p{margin:.5em 0 0}
.gc-step .eq{display:block;margin:.55em 0 0;padding:.35em .5em;border-radius:6px;background:var(--global-bg-color);font-size:.95em;text-align:center}
.gc-step .was{display:block;margin-top:.55em;font-size:.92em;color:var(--gc-gray)}
/* range rows (simulation) */
.gc-rows{margin:1em 0 .2em}
.gc-row{display:grid;grid-template-columns:150px 1fr 96px;gap:.7em;align-items:center;height:34px}
.gc-row .nm{font-size:.92em;font-weight:700}
.gc-row .tr{position:relative;height:12px;border-radius:999px;background:var(--gc-gray-soft)}
.gc-row .rg{position:absolute;top:0;bottom:0;border-radius:999px;min-width:3px;transition:left .3s,width .3s}
.gc-row .md{position:absolute;top:-5px;bottom:-5px;width:3px;margin-left:-1.5px;border-radius:2px;background:var(--global-text-color);transition:left .3s}
.gc-row .tv{position:absolute;top:-11px;bottom:-11px;border-left:2px dashed var(--gc-green);transition:left .3s}
.gc-row .vl{font-size:.9em;font-weight:700;font-variant-numeric:tabular-nums;text-align:right}
.gc-row.truth .rg{background:var(--gc-green)}.gc-row.naive .rg{background:var(--gc-gray)}.gc-row.deb .rg{background:var(--gc-blue)}
.gc-row.truth .nm{color:var(--gc-green)}.gc-row.deb .nm{color:var(--gc-blue)}
.gc-axis{display:flex;justify-content:space-between;font-size:.72em;opacity:.7;margin:.2em calc(96px + .7em) 0 calc(150px + .7em)}
/* real-data explorer */
.gc-rank{position:relative;margin:1.6em 0 .2em}
.gc-rank .row{position:absolute;left:0;right:0;height:30px;display:grid;grid-template-columns:22px 150px 1fr 118px;gap:.6em;align-items:center;transition:top .5s cubic-bezier(.2,.8,.2,1)}
.gc-rank .no{width:22px;height:22px;border-radius:50%;border:1px solid var(--gc-line);display:grid;place-items:center;font-size:.8em;font-weight:700}
.gc-rank .row.top .no{background:var(--gc-top,var(--gc-gray));border-color:transparent;color:#fff}
html[data-theme="dark"] .gc-rank .row.top .no{color:#16202b}
.gc-rank .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.92em}
.gc-rank .tr{position:relative;height:14px;border-radius:999px;background:var(--gc-gray-soft)}
.gc-rank .fn{position:absolute;left:0;top:0;bottom:0;border-radius:999px;background:var(--gc-gray);opacity:.45;transition:width .5s}
.gc-rank .fe{position:absolute;left:0;top:3px;bottom:3px;border-radius:999px;background:var(--gc-blue);transition:width .5s}
.gc-rank .vl{font-size:.86em;font-variant-numeric:tabular-nums;white-space:nowrap;text-align:right}
.gc-rank .vl b{color:var(--gc-blue)}
.gc-rank .fl{position:absolute;z-index:1;top:-10px;bottom:-2px;border-left:2px dashed var(--gc-red);pointer-events:none;transition:left .5s}
.gc-rank .fl span{position:absolute;top:-17px;transform:translateX(-50%);font-size:.74em;font-weight:700;white-space:nowrap;color:var(--gc-red)}
.gc-legend{display:flex;flex-wrap:wrap;gap:.3em 1.1em;font-size:.82em;margin:.7em 0 0}
.gc-legend i{display:inline-block;width:1.6em;height:.7em;border-radius:999px;margin-right:.4em;vertical-align:baseline}
.gc-legend .ln{background:var(--gc-gray);opacity:.45}.gc-legend .le{background:var(--gc-blue)}
.gc-legend .lf{height:0;border-top:2px dashed var(--gc-red);border-radius:0;vertical-align:.25em}
.gc-legend .lc{width:.8em;height:.8em;border-radius:50%;background:transparent;border:2px solid var(--gc-gray);vertical-align:-.1em}
.gc-legend .ls{width:.8em;height:.8em;border-radius:50%;background:var(--gc-blue);vertical-align:-.1em}
/* Mondrian dot chart */
.gc-dots{margin:1.4em 0 .2em}
.gc-dot{display:grid;grid-template-columns:150px 1fr;gap:.7em;align-items:center;height:26px}
.gc-dot .nm{font-size:.86em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.gc-dot .tr{position:relative;height:100%}
.gc-dot .tr:before{content:"";position:absolute;left:0;right:0;top:50%;border-top:1px solid var(--gc-line)}
.gc-dot .one{position:absolute;top:0;bottom:0;border-left:2px solid var(--global-text-color);opacity:.55}
.gc-dot .lk{position:absolute;top:50%;height:2px;margin-top:-1px;background:var(--gc-gray);opacity:.5}
.gc-dot .pt{position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;box-sizing:border-box}
.gc-dot .pt.c{border:2px solid var(--gc-gray);background:var(--global-bg-color)}
.gc-dot .pt.s{background:var(--gc-blue)}
.gc-dot .pt.s.over{background:var(--gc-red)}
.gc-dax{position:relative;height:2.6em;margin:0 0 0 calc(150px + .7em);font-size:.74em;opacity:.8}
.gc-dax span{position:absolute;top:.2em;transform:translateX(-50%)}
.gc-dax em{position:absolute;left:0;right:0;top:1.5em;text-align:center;font-style:normal}
/* stats, tables, lists */
.gc-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.gc-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--gc-line);background:var(--gc-card);position:relative;overflow:hidden}
.gc-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--gc-c,var(--gc-blue))}
.gc-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;color:var(--gc-c,var(--gc-blue));font-variant-numeric:tabular-nums}
.gc-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.gc-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--gc-line);border-radius:14px;background:var(--gc-card);overflow-x:auto}
.gc-table table{font-size:.8em;margin:.6em 0;min-width:560px}
.gc-table p{font-size:.86em;margin:.2em}
.gc-table thead,.gc-table th{background:transparent;color:inherit}
.gc-table th{border-bottom:2px solid var(--gc-line)}
.gc-table .y{color:var(--gc-green);font-weight:800}.gc-table .x{color:var(--gc-red);font-weight:800}
.gc-how{font-size:.86em;line-height:1.55;margin:.5em .2em 0;padding:.5em .75em;border-radius:8px;background:var(--gc-gray-soft)}
.gc-take{margin:0 0 1.2em;padding:0;list-style:none;display:grid;gap:.55em}
.gc-take li{padding:.65em .9em .65em 1em;border-left:3px solid var(--gc-green);border-radius:4px;background:var(--gc-card);font-size:.92em;line-height:1.55}
.gc-take li.scope{border-left-color:var(--gc-gray)}
.gc-abs{margin:0 0 1.6em;border:1px solid var(--gc-line);border-radius:12px;background:var(--gc-card)}
.gc-abs summary{cursor:pointer;padding:.6em .9em;font-weight:700;font-size:.92em}
.gc-abs p{margin:0;padding:0 .95em .9em;font-size:.9em;line-height:1.65}
/* the theme's fixed table of contents (220px, right) covers a full-width title: keep the header in the text column */
@media screen and (min-width:1200px){.page__inner-wrap>header{max-width:calc(100% - 240px)}}
@media (max-width:900px){.gc-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:760px){.gc-steps,.gc-ids,.gc-defs{grid-template-columns:1fr}.gc-layers{grid-template-columns:1fr}.gc-layers .arr{display:none}}
@media (max-width:640px){.gc-tldr{font-size:1.04em}.gc-ctl{grid-template-columns:1fr}.gc-tiles{grid-template-columns:1fr}.gc-worlds{grid-template-columns:1fr}
  .gc-row{grid-template-columns:96px 1fr 84px;gap:.45em}.gc-axis{margin:.2em calc(84px + .45em) 0 calc(96px + .45em)}.gc-row .nm{font-size:.82em}
  .gc-rank .row{grid-template-columns:20px 112px 1fr 92px;gap:.35em}.gc-rank .nm{font-size:.82em}.gc-rank .vl{font-size:.78em}
  .gc-dot{grid-template-columns:104px 1fr;gap:.45em}.gc-dax{margin-left:calc(104px + .45em)}.gc-dot .nm{font-size:.78em}}
</style>

<div class="gc">
<div class="gc-kicker"><span>Conformal prediction</span><span>Group-conditional coverage</span><span>Evaluation methodology</span></div>
<p class="gc-tldr">Conformal methods that promise coverage for every group or class are ranked by the coverage each group shows on a test set. With many small groups, <b>much of the reported gap is binomial test noise</b>: a method that covers every group exactly 90% of the time, tested on 5 points per group, shows a coverage gap of 0.118 and 41% of its groups "below 85%". The paper works out what a test set can and cannot say about group coverage, and gives estimators, bounds and an audit that account for the noise.</p>
<div class="gc-btns"><a class="gc-btn" href="#the-problem-the-metrics-include-test-noise">The problem</a><a class="gc-btn ghost" href="#what-a-test-set-identifies">What is identified</a><a class="gc-btn ghost" href="#results">Results</a><a class="gc-btn ghost" href="#recommendations">Recommendations</a><span class="gc-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="gc-fig" id="gc-floor"><div class="gc-ix">
<div class="gc-ctl"><span class="l">Target coverage</span><div class="gc-seg" id="gc-f-tau"></div><span class="l">Test points per group</span><div class="gc-range"><input type="range" id="gc-f-m" min="0" max="8" step="1" value="2" aria-label="Test points per group"><output id="gc-f-mv"></output></div></div>
<div class="gc-hist" id="gc-f-hist" role="img" aria-label="Distribution of the coverage a group shows on the test set"></div>
<div class="gc-haxis" id="gc-f-axis"></div>
<div class="gc-tiles" id="gc-f-tiles"></div>
<p class="gc-read" id="gc-f-read"></p>
</div><figcaption><span class="gc-tag">Interactive</span><b class="gc-title">A perfect method still shows a gap.</b>Every group here has true coverage equal to the target. The bars are the coverage a group shows on its test points (the binomial distribution, computed exactly); red bars fall below the usual cut of target minus 5 points and would be counted as under-covered. The three numbers are the paper's noise floors (Proposition 3.2): what the standard metrics report for this method. Move the slider to see how slowly they shrink.</figcaption></figure>
</div>

## Background

<div class="gc">
<p class="gc-lead">Split conformal prediction turns any model into prediction sets that contain the true label with probability 1 − α, on average over the population. Many recent methods aim for more: coverage 1 − α <i>within every group</i> (a state, an airport, a class). They are compared on a test set by computing each group's empirical coverage and summarizing how far the groups are from the target.</p>
<div class="gc-defs">
<div class="gc-def"><b>c<sub>g</sub></b>Realized coverage of group <i>g</i>: the probability that the deployed prediction sets cover a new point from that group, given the calibration set that was drawn. This is what a user of the method gets, and it is unknown.</div>
<div class="gc-def"><b>ĉ<sub>g</sub> = h<sub>g</sub> / m<sub>g</sub></b>Empirical coverage: of the <i>m<sub>g</sub></i> test points in group <i>g</i>, <i>h<sub>g</sub></i> are covered. With independent test points, <i>h<sub>g</sub></i> is binomial with success probability <i>c<sub>g</sub></i>.</div>
<div class="gc-def"><b>CovGap</b>Mean over groups of |<i>c<sub>g</sub></i> − (1 − α)|. Its one-sided version, the <b>shortfall</b>, averages only the under-coverage (1 − α − <i>c<sub>g</sub></i>)<sub>+</sub>. The <b>RMS gap</b> is the root of the mean squared gap.</div>
<div class="gc-def"><b>FracBelow, worst group</b>The share of groups whose coverage is below a cut (here 1 − α − 0.05), and the smallest group coverage. Both depend on the tail of the <i>c<sub>g</sub></i>.</div>
<div class="gc-def wide"><b>Naive metric</b>The same formula with ĉ<sub>g</sub> in place of <i>c<sub>g</sub></i>. This is what papers report. <b>Mondrian</b> conformal prediction, the classical group-conditional method, calibrates each group on its own calibration points.</div>
</div>
<p class="gc-lead">A reported group metric mixes three sources of variation. Conformal guarantees speak about the first, a practitioner cares about the second, and the test set adds the third:</p>
<div class="gc-layers">
<div class="gc-layer"><span class="k">Layer a</span><span class="h">Expected coverage E[<i>c<sub>g</sub></i>]</span>What the method does on average over calibration sets. Guarantees such as Mondrian's are statements about this layer.</div>
<span class="arr">→</span>
<div class="gc-layer on"><span class="k">Layer b · the target</span><span class="h">Realized coverage <i>c<sub>g</sub></i></span>What one calibration draw delivers to each group. The paper asks what the test set identifies about these numbers.</div>
<span class="arr">→</span>
<div class="gc-layer noise"><span class="k">Layer c</span><span class="h">Test noise ĉ<sub>g</sub> − <i>c<sub>g</sub></i></span>Binomial noise from having <i>m<sub>g</sub></i> test points. Large exactly when groups are many and small.</div>
</div>
</div>

## The problem: the metrics include test noise

<div class="gc">
<p class="gc-lead">For the squared gap the contribution of the test noise is exact (Proposition 3.1): the naive value is the true one plus a binomial variance that does not depend on how good the method is.</p>
<div class="gc-eqline">E[ <span class="t-n">(ĉ<sub>g</sub> − τ)²</span> ] = <span class="t-b">(<i>c<sub>g</sub></i> − τ)²</span> + <span class="t-r"><i>c<sub>g</sub></i>(1 − <i>c<sub>g</sub></i>) / <i>m<sub>g</sub></i></span><br><span style="font-size:.82em;opacity:.85"><span class="t-n">what is reported</span> = <span class="t-b">the method's gap</span> + <span class="t-r">test noise</span>, with τ = 1 − α</span></div>
<p class="gc-lead">The naive CovGap and shortfall are biased upward as well. When every group is covered exactly at the target, the true gaps are zero and the naive metrics show only noise. The paper gives these <b>noise floors</b> in closed form, for an exactly valid method and for Mondrian (whose realized coverages follow a Beta law). At τ = 0.9 the CovGap floor is 0.162, 0.118 and 0.024 for 2, 5 and 100 test points per group. The FracBelow floor is not even monotone: with up to 6 test points, one uncovered point puts a group below 85%, so a reported "30% of groups under-covered" on groups of five is <i>less</i> than a perfect method would show.</p>
</div>

## What a test set identifies

<div class="gc">
<p class="gc-lead">Debiasing a metric requires that the test set determine it. With <i>m</i> test points per group, the data depend on the distribution of the <i>c<sub>g</sub></i> only through its first <i>m</i> moments (Proposition 4.1). Metrics that are polynomials of low degree can be recovered; metrics that count groups beyond a threshold cannot.</p>
<div class="gc-ids">
<div class="gc-id" style="--gc-c:var(--gc-red)"><span class="s">Not identified</span><span class="h">FracBelow, quantiles, worst group</span>With few test points per group, different sets of group coverages give the same data. The paper gives the sharp identified sets for <i>m</i> ≤ 2 and moment-LP bounds in general (Theorem 4.2). On synthetic data the LP bounds covered the truth (pooled coverage 1.000), but with a median width of 0.36 to 0.77 for FracBelow.</div>
<div class="gc-id" style="--gc-c:var(--gc-gray)"><span class="s">Partially identified</span><span class="h">CovGap, shortfall</span>Only bounded when <i>m</i> ≤ 2: at μ = τ = 0.9 the CovGap can be anywhere in [2σ², σ]. Both are always at most the RMS gap (Proposition 4.3). The paper estimates them by deconvolution within strata of equal <i>m</i>.</div>
<div class="gc-id" style="--gc-c:var(--gc-blue)"><span class="s">Point-identified</span><span class="h">RMS gap</span>A degree-2 polynomial, so two test points per group suffice. It has an unbiased estimator of its square and intervals that need no model for how the <i>c<sub>g</sub></i> arise.</div>
</div>
<figure class="gc-fig" id="gc-worlds"><div class="gc-ix">
<div class="gc-ctl"><span class="l">Spread of true group coverage (sd)</span><div class="gc-range"><input type="range" id="gc-w-s" min="1" max="7" step="0.5" value="5" aria-label="Standard deviation of true group coverage, in points"><output id="gc-w-sv"></output></div></div>
<div class="gc-worlds">
<div class="gc-world"><div class="wt">World A</div><div class="gc-lolly" id="gc-w-a"></div><div class="gc-haxis" id="gc-w-ax1" style="margin:0 6%"></div><div class="wv" id="gc-w-av"></div></div>
<div class="gc-world"><div class="wt">World B</div><div class="gc-lolly" id="gc-w-b"></div><div class="gc-haxis" id="gc-w-ax2" style="margin:0 6%"></div><div class="wv" id="gc-w-bv"></div></div>
</div>
<p class="gc-same" id="gc-w-same"></p>
<p class="gc-read b" id="gc-w-read"></p>
</div><figcaption><span class="gc-tag">Interactive</span><b class="gc-title">Two worlds that two test points per group cannot tell apart (Theorem 4.2b).</b>Each panel is a possible set of true group coverages: a stem at a coverage value, its height the share of groups there. Both have mean 90% and the spread you choose, so with two test points per group they produce exactly the same data. Yet the share of groups at or below the 85% cut (red, dashed line) differs. The gray box is what the test set shows in either world.</figcaption></figure>
</div>

## What to report instead

<div class="gc">
<div class="gc-steps">
<div class="gc-step"><span class="n">1</span><b class="h">Debias the squared gap</b><p>For each group with at least two test points, subtract an unbiased estimate of its binomial variance, then average over groups. The result is unbiased for the mean squared gap for every set of group coverages (Theorem 5.1).</p><span class="eq">e<sub>g</sub> = (ĉ<sub>g</sub> − τ)² − ĉ<sub>g</sub>(1 − ĉ<sub>g</sub>) / (m<sub>g</sub> − 1)</span><span class="was">Replaces: the naive CovGap.</span></div>
<div class="gc-step"><span class="n">2</span><b class="h">Bound the gap</b><p>An interval from the spread of the e<sub>g</sub> is asymptotically valid, and no valid interval can be shorter than order G<sup>−1/4</sup> near zero. Its upper end can miss with small groups, so two finite-sample upper bounds are added: U<sub>1</sub> (Bernstein) and the tighter U<sub>3</sub> (Chernoff with the exact binomial moment generating function). They hold for every set of group coverages and any number of groups.</p><span class="was">Replaces: a point value with no error bar.</span></div>
<div class="gc-step"><span class="n">3</span><b class="h">Audit single groups</b><p>To list under-covered groups, compute for each group the exact one-sided binomial <i>p</i>-value of "coverage is at least the cut" and apply the Benjamini–Hochberg procedure. The false discovery rate is controlled in finite samples (Proposition 6.1), and the power for a group is at most that of one exact test.</p><span class="was">Replaces: flagging every group with ĉ<sub>g</sub> below the cut.</span></div>
</div>
<p class="gc-lead">Two methods evaluated on the same test points are compared through the differences of their e<sub>g</sub>, which gives an unbiased estimate of the difference in mean squared gap and an asymptotic paired interval. The simulation below applies step 1 to data you choose.</p>
<figure class="gc-fig" id="gc-sim"><div class="gc-ix">
<div class="gc-ctl"><span class="l">Groups</span><div class="gc-seg" id="gc-s-g"></div><span class="l">Test points per group</span><div class="gc-seg" id="gc-s-m"></div><span class="l">True spread of group coverage (sd)</span><div class="gc-seg" id="gc-s-s"></div><span class="l">Test sets</span><div class="gc-seg"><button type="button" class="go" id="gc-s-go">Draw 200 new test sets</button></div></div>
<div class="gc-rows" id="gc-s-rows"></div>
<div class="gc-axis" id="gc-s-axis"></div>
<p class="gc-read b" id="gc-s-read"></p>
</div><figcaption><span class="gc-tag">Interactive</span><b class="gc-title">Naive and debiased RMS gap on simulated test sets.</b>The group coverages are drawn once around a 90% target with the chosen spread and then held fixed, as for one deployed calibration; the green line is their true RMS gap. Each of 200 test sets gives a naive and a debiased value: bars span the 5th to 95th percentile, the tick is the median. The simulation runs in your browser with the estimator of step 1; it illustrates the method and is not a result from the paper.</figcaption></figure>
</div>

## Results

<div class="gc">
<p class="gc-lead">All criteria were pre-registered: estimators, grids, seeds, comparators and pass thresholds were fixed before any number was computed, and a quarter of each real dataset was held out during development and read once. The real-data study covers 7 datasets with pre-specified groups (13 dataset and score combinations), 7 methods and 200 random splits per run.</p>
<div class="gc-stats">
<div class="gc-stat" style="--gc-c:var(--gc-red)"><div class="v">41%</div><div class="d">median share of the naive mean-squared coverage gap that is test noise on held-out real data; from 5% to nearly all of it</div></div>
<div class="gc-stat" style="--gc-c:var(--gc-gray)"><div class="v">62%</div><div class="d">of the reported CovGap would be shown by a method with exact coverage in every group (median; 96% on ACS)</div></div>
<div class="gc-stat"><div class="v">87–90%</div><div class="d">of 1,344 synthetic cells where our CovGap and shortfall estimates beat the better of two comparators</div></div>
<div class="gc-stat" style="--gc-c:var(--gc-green)"><div class="v">24 / 26</div><div class="d">settings where Mondrian, the method with the largest gap, has a smaller estimated shortfall than standard split (post hoc)</div></div>
</div>
<figure class="gc-fig" id="gc-real"><div class="gc-ix">
<div class="gc-ctl"><span class="l">Dataset · score</span><div class="gc-seg" id="gc-r-cells"></div><span class="l">Order by</span><div class="gc-seg" id="gc-r-sort"><button type="button" data-v="naive">naive CovGap</button><button type="button" data-v="est">estimated CovGap</button></div></div>
<div class="gc-rank" id="gc-r-rank"></div>
<div class="gc-legend"><span><i class="ln"></i>naive CovGap (reported)</span><span><i class="le"></i>our estimate of the realized CovGap</span><span><i class="lf"></i>what exact coverage in every group would show</span></div>
<p class="gc-read" id="gc-r-read"></p>
</div><figcaption><span class="gc-tag">Interactive</span><b class="gc-title">Held-out real data: reported and estimated coverage gap, per method.</b>Seven methods on the same splits at α = 0.1: standard split conformal, Mondrian, shrinkage of group thresholds towards the pooled one at three strengths, clustered conformal (Ding et al., 2023) and conditional conformal with group indicators (Gibbs et al., 2025). Values are means over 200 splits. The estimate is a point in an identified set when some groups have at most two test points. Switch the order to see whether the ranking changes.</figcaption></figure>
</div>

<div class="gc"><div class="gc-table" markdown="1">

<span class="gc-tag">Table 3</span> **Real data, primary runs at α = 0.1 (excerpt).** Medians over the 7 methods.

| Dataset (groups) | Score | Groups | With < 20 test points | Test-noise share of naive MSE | Naive / estimated CovGap | Exact-coverage floor / naive CovGap | Rank agreement (Kendall) |
|---|---|---|---|---|---|---|---|
| ACS income (state × race) | CQR | 408 | 35% | 1.01 | 3.20 | 0.96 | 0.90 |
| Airlines (origin airport) | CQR | 352 | 27% | 0.61 | 1.61 | 0.65 | 1.00 |
| Zurich transport (stop) | CQR | 1,528 | 6% | 0.41 | 1.25 | 0.54 | 1.00 |
| DBpedia (class) | LAC | 219 | 0% | 0.05 | 1.03 | 0.20 | 1.00 |
| CIFAR-100 (class) | APS | 100 | 0% | 0.37 | 1.32 | 0.64 | 0.90 |
| House sales (zipcode) | CQR | 70 | 8% | 0.57 | 1.59 | 0.77 | 0.90 |

<p class="gc-how"><b>How to read it.</b> The noise share is one minus the ratio of the debiased to the naive mean squared gap; it needs no model, and exceeds 1 when the debiased value is slightly negative. On ACS, for every method except Mondrian the debiased RMS gap is at most 0.015 against a naive 0.082 to 0.084. On DBpedia with the LAC score the groups are large and the gap is real, so the correction is small.</p>

The size of the gaps changes more than the order of the methods: the median rank correlation between the naive and the estimated CovGap is 0.90, and the method with the smallest gap differs in 6 of 26 primary runs. In those six, the paired test separates the two leaders in at most 4.5% of splits (post hoc).

</div></div>

### Pre-registered criteria, including the ones that failed

<div class="gc"><div class="gc-table" markdown="1">

<span class="gc-tag">Table 2</span> **Criteria and outcomes.** Each synthetic study used a fresh grid of configurations; intervals are judged by pooled coverage and by the number of cells that reject coverage of at least 0.95 after a Holm correction.

| Check | Result | Met |
|---|---|---|
| CovGap estimate vs. the better comparator | better in 87% of cells | <span class="y">yes</span> |
| Shortfall estimate vs. the better comparator | better in 90% of cells | <span class="y">yes</span> |
| Asymptotic RMS interval | coverage 0.975, but 17 cells rejected (minimum 0.874) | <span class="x">no</span> |
| Moment-LP bounds | coverage 1.000, 0 cells rejected | <span class="y">yes</span> |
| Audit: false discovery rate at q = 0.1 | 0.0014, 0 cells rejected | <span class="y">yes</span> |
| Audit: power vs. simultaneous bootstrap audit | at least as high in 73% of cells | <span class="y">yes</span> |
| Interval with finite-sample upper end U<sub>1</sub>, new grid | coverage 0.997, 0 cells rejected | <span class="y">yes</span> |
| U<sub>3</sub> with shared test points | exceeded in 3 of 672,000 replications | <span class="y">yes</span> |
| Paired interval | coverage 0.960, but 182 cells rejected (minimum 0.762) | <span class="x">no</span> |
| Rule of use for the paired interval (f<sub>1</sub> ≤ 0.05) | 3 cells rejected (minimum 0.892) | <span class="x">no</span> |
| Refined rule (f<sub>1</sub> ≤ 0.05 and at least 200 groups) | coverage 0.980, 0 cells rejected | <span class="y">yes</span> |
| Paired test under exact (permutation) nulls | rejects in 2.4%, 0 cells rejected | <span class="y">yes</span> |
| Real data: naive / estimated CovGap at least 1.5 | 3.22, 1.63, 1.59 | <span class="y">yes</span> |
| Real data: overstatement certified by U<sub>1</sub> | 0 of 7 methods | <span class="x">no</span> |

<p class="gc-how"><b>What failed, and why.</b> The asymptotic intervals miss when groups are small and coverage is high: a small group rarely has an uncovered point, a rare miss gives a large e<sub>g</sub>, and the estimated standard error is small exactly when the estimate is low. For the RMS gap the finite-sample bounds repair this at the cost of width. For comparing two methods, the rule that held on a fresh grid keeps 9% of replications and none with many small groups; a finite-sample paired interval never missed but was a median 50 times wider. The finite-sample bounds certified no overstatement on the real test sets: simulated on those test counts, even an exactly valid method is certified by U<sub>3</sub> in at most 17% of test sets.</p>

</div></div>

### A large gap is not under-coverage

<div class="gc">
<p class="gc-lead">Mondrian is the one method whose reported gap is not inflated: its realized coverages do vary across groups, so its naive and estimated CovGap agree (0.084 and 0.084 on ACS). It has the largest gap in most settings. A post hoc analysis of the logged counts shows what the gap consists of: mostly over-coverage of groups with few calibration points.</p>
<figure class="gc-fig" id="gc-one"><div class="gc-ix">
<div class="gc-dots" id="gc-o-dots"></div>
<div class="gc-dax" id="gc-o-ax"></div>
<div class="gc-legend"><span><i class="lc"></i>estimated CovGap, Mondrian / standard split</span><span><i class="ls"></i>estimated shortfall, Mondrian / standard split (red where it is larger)</span></div>
</div><figcaption><span class="gc-tag">Table 15</span><b class="gc-title">Mondrian against standard split conformal: larger gap, smaller shortfall.</b>Ratios of medians over 200 splits, primary runs at α = 0.1, logarithmic axis. Right of the line Mondrian is larger. Its estimated CovGap is up to 5.5 times larger, while its estimated shortfall, the under-coverage part, is smaller in 12 of these 13 runs (24 of 26 when α = 0.05 is included). Over-coverage costs interval length; under-coverage costs a group its guarantee. A symmetric gap cannot tell the two apart.</figcaption></figure>
</div>

### How many groups does a certificate need?

<div class="gc"><div class="gc-table" markdown="1">

<span class="gc-tag">Table 1</span> **Test-set design.** Number of groups at which the upper bound falls below the naive CovGap in 80% of replications, when every group has <i>m</i> test points and exact coverage. Excerpt.

| Test points per group | U<sub>1</sub>, target 0.90 | U<sub>3</sub>, target 0.90 | U<sub>1</sub>, target 0.95 | U<sub>3</sub>, target 0.95 |
|---|---|---|---|---|
| 2 | 741 | 611 | 2,558 | 2,094 |
| 5 | 366 | 180 | 804 | 355 |
| 20 | 826 | 192 | 1,802 | 318 |
| 100 | 2,599 | 163 | 5,199 | 233 |

<p class="gc-how"><b>How to read it.</b> If the bound is below the reported CovGap, the report is certified to overstate the realized gap. With at least five test points per group U<sub>3</sub> needs 160 to 360 groups in the easiest case, exact coverage everywhere. When coverage varies across groups the naive CovGap overstates less and certification needs many more groups. Whether any valid bound can certify at the sizes of current benchmarks is open.</p>

</div></div>

## Recommendations

<div class="gc">
<ul class="gc-take">
<li><b>Report the noise floor next to any naive group metric.</b> It needs only the test counts per group.</li>
<li><b>Report the debiased RMS gap</b>, or an estimated CovGap with its identification caveat, in place of the naive CovGap. For groups with few test points, do not report the fraction below a cut or the worst group, or report their bounds.</li>
<li><b>Compare two methods with the paired interval only when</b> there are at least 200 groups and at most 5% of them have 20 or fewer test points and empirical coverage 1 for either method. With many small groups, no comparison we tried is both valid and informative.</li>
<li><b>Report one-sided gaps such as the shortfall</b> next to symmetric ones.</li>
<li><b>Bound how large the realized gap can be</b> with U<sub>3</sub>, and check before evaluating whether the test counts could certify anything at all.</li>
<li><b>Flag individual groups with BH on exact binomial <i>p</i>-values</b>, and expect few flags for small groups: a group with true coverage 75%, a cut of 85% and 10 test points is flagged with probability at most 0.22. An audit that flags no group is no evidence that small groups are covered.</li>
<li class="scope"><b>Limits.</b> The targets are conditional on one calibration draw; expected coverage (layer a) was estimated on development data only. The CovGap and shortfall estimates have guarantees only within strata of equal test counts and were significantly worse than the better comparator in 7.7% of synthetic cells. The asymptotic intervals under-cover with small groups and high coverage, and the finite-sample bounds pay for validity with width. Everything assumes independent test points within a group, which held approximately in the two datasets where we checked it, and nothing covers distribution shift.</li>
</ul>
<details class="gc-abs"><summary>Abstract</summary><p>Conformal methods for group-conditional or class-conditional coverage are compared using metrics computed from the empirical coverage of each group on a test set, such as the average coverage gap, the fraction of under-covered groups and the worst-group coverage. When groups are many and small, much of what these metrics report is binomial test noise. We separate the method's expected coverage, the coverage realized by one calibration draw, and the test noise, and study what the data identify about the realized coverage. We give closed-form noise floors, the values that a method with exact group coverage would show. The fraction of under-covered groups and the worst group are not identified when groups have few test points; we give their sharp identified sets. The root-mean-square coverage gap is point-identified. For it we give a debiased estimator, an asymptotically valid interval whose length is rate-optimal up to a logarithmic factor, finite-sample upper bounds, and an asymptotic paired comparison of methods. To flag individual groups, BH on exact binomial <i>p</i>-values controls the false discovery rate in finite samples, and its power for a group is at most that of a single exact test. In a pre-registered evaluation on held-out real data (7 datasets, 7 methods, 200 splits), test noise is a median 41% of the naive mean-squared coverage gap, ranging from 5% to nearly all of it, and a method with exact group coverage would show a median 62% of the reported coverage gap. The finite-sample bounds are too conservative to certify this overstatement at these test-set sizes. Post hoc, the method with the largest gap (Mondrian) mostly over-covers: its estimated shortfall is below standard split's in 24 of 26 settings.</p></details>
</div>

## Paper and code

Paper, code, the pre-registration documents and BibTeX will be linked here once the preprint is public.

{% raw %}
<script>
(function () {
  function $(id) { return document.getElementById(id); }
  function seg(el, items, cur, on, fmt) {
    el.innerHTML = "";
    items.forEach(function (k) { var b = document.createElement("button"); b.type = "button"; b.textContent = fmt ? fmt(k) : k;
      b.setAttribute("aria-pressed", k === cur); b.onclick = function () { on(k); }; el.appendChild(b); });
  }
  function pct(x, d) { return (100 * x).toFixed(d === undefined ? 0 : d) + "%"; }

  /* ---------- 1. noise floor: exact binomial law of the coverage a group shows ---------- */
  var MS = [2, 3, 5, 10, 20, 50, 100, 200, 500], LF = [0];
  for (var i = 1; i <= 500; i++) LF[i] = LF[i - 1] + Math.log(i);
  function pmf(m, p) { var out = []; for (var h = 0; h <= m; h++) out.push(Math.exp(LF[m] - LF[h] - LF[m - h] + h * Math.log(p) + (m - h) * Math.log(1 - p))); return out; }
  var fTau = 0.9, fM = 5;
  function drawFloor() {
    seg($("gc-f-tau"), [0.9, 0.95], fTau, function (k) { fTau = k; drawFloor(); }, function (k) { return pct(k); });
    $("gc-f-m").value = MS.indexOf(fM); $("gc-f-mv").textContent = fM;
    var cut = fTau - 0.05, P = pmf(fM, fTau), gap = 0, below = 0, hidden = 0, top = 0;
    P.forEach(function (q, h) { gap += q * Math.abs(h / fM - fTau); if (h < cut * fM - 1e-9) below += q; if (h / fM < 0.5 - 1e-9) hidden += q; else top = Math.max(top, q); });
    var rms = Math.sqrt(fTau * (1 - fTau) / fM), X = function (v) { return (v - 0.5) / 0.5 * 100; };
    var w = Math.min(7, 160 / fM), html = "";
    P.forEach(function (q, h) { var v = h / fM; if (v < 0.5 - 1e-9 || q < 1e-5) return;
      html += '<div class="bar' + (h < cut * fM - 1e-9 ? " low" : "") + '" style="left:' + X(v) + "%;width:" + w + "%;min-width:2px;height:" + Math.max(0.6, 100 * q / top) + '%">' + (fM <= 10 && q >= 0.004 ? "<i>" + pct(q, q < 0.1 ? 1 : 0) + "</i>" : "") + "</div>"; });
    html += '<div class="vl tgt' + (fTau > 0.92 ? " hi" : "") + '" style="left:' + X(fTau) + '%"><span>target ' + pct(fTau) + '</span></div><div class="vl cut" style="left:' + X(cut) + '%"><span>cut ' + pct(cut) + "</span></div>";
    $("gc-f-hist").innerHTML = html;
    $("gc-f-axis").innerHTML = [0.5, 0.6, 0.7, 0.8, 0.9, 1].map(function (v) { return '<span style="left:' + X(v) + '%">' + pct(v) + "</span>"; }).join("") + "<em>coverage a group shows on its " + fM + " test points</em>";
    $("gc-f-tiles").innerHTML =
      '<div class="gc-tile"><div class="v">' + gap.toFixed(3) + '</div><div class="d">CovGap the test set shows</div><div class="t">true CovGap: 0</div></div>' +
      '<div class="gc-tile"><div class="v">' + pct(below, below < 0.1 ? 1 : 0) + '</div><div class="d">of groups show coverage below ' + pct(cut) + '</div><div class="t">true share: 0%</div></div>' +
      '<div class="gc-tile"><div class="v">' + rms.toFixed(3) + '</div><div class="d">RMS gap the test set shows</div><div class="t">true RMS gap: 0</div></div>';
    var step = 100 / fM, s = "<b>Every group has true coverage exactly " + pct(fTau) + ".</b> With " + fM + " test points a group can only show multiples of " + (step >= 1 ? step.toFixed(step % 1 ? 1 : 0) : step.toFixed(1)) + "%";
    s += fM <= 10 ? ", so it cannot land on the target and its distance from it is noise." : ", and its distance from the target is still noise.";
    s += " Averaged over groups, the standard metrics report a coverage gap of " + gap.toFixed(3) + " and " + pct(below, below < 0.1 ? 1 : 0) + " of groups under-covered.";
    if (hidden > 0.001) s += " (" + pct(hidden, 1) + " of groups show less than 50% and are not drawn.)";
    $("gc-f-read").innerHTML = s;
  }
  $("gc-f-m").addEventListener("input", function () { fM = MS[+this.value]; drawFloor(); });

  /* ---------- 2. two worlds with the same mean and variance (identified set for m <= 2) ---------- */
  var MU = 0.9, T = 0.85;
  function lolly(el, atoms) {
    var X = function (v) { return (v - 0.7) / 0.3 * 100; }, mx = Math.max.apply(null, atoms.map(function (a) { return a.w; }));
    el.innerHTML = atoms.map(function (a) {
      return '<div class="st' + (a.x <= T + 1e-9 ? " low" : "") + '" style="left:' + X(a.x) + "%;height:" + (78 * a.w / Math.max(mx, 0.5)) + '%"><i>' + pct(a.w) + " at " + pct(a.x, Math.abs(100 * a.x - Math.round(100 * a.x)) < 0.05 ? 0 : 1) + "</i></div>"; }).join("") +
      '<div class="vl" style="left:' + X(T) + '%"></div>';
  }
  function drawWorlds() {
    var s = +$("gc-w-s").value / 100, v = s * s, d = MU - T;
    $("gc-w-sv").textContent = (100 * s).toFixed(1) + " pts";
    var a = MU - v / (1 - MU), pa = (1 - MU) * (1 - MU) / ((1 - MU) * (1 - MU) + v);
    var w = v / (v + d * d), b = MU + v / d;
    lolly($("gc-w-a"), [{x: a, w: pa}, {x: 1, w: 1 - pa}]);
    lolly($("gc-w-b"), [{x: T, w: w}, {x: b, w: 1 - w}]);
    var ax = [0.7, 0.8, 0.9, 1].map(function (x) { return '<span style="left:' + ((x - 0.7) / 0.3 * 100) + '%">' + pct(x) + "</span>"; }).join("") + "<em>true group coverage</em>";
    $("gc-w-ax1").innerHTML = ax; $("gc-w-ax2").innerHTML = ax;
    $("gc-w-av").innerHTML = "Groups at or below 85%: <b style=\"color:var(--gc-blue)\">0%</b>";
    $("gc-w-bv").innerHTML = "Groups at or below 85%: <b style=\"color:var(--gc-red)\">" + pct(w) + "</b>";
    var p2 = MU * MU + v, p1 = 2 * (MU - p2), p0 = 1 - p1 - p2;
    $("gc-w-same").innerHTML = "Two test points per group, in both worlds: both covered in <b>" + pct(p2, 1) + "</b> of groups, one in <b>" + pct(p1, 1) + "</b>, none in <b>" + pct(p0, 1) + "</b>.";
    $("gc-w-read").innerHTML = "With a spread of " + (100 * s).toFixed(1) + " points, any share of groups at or below 85% between <b>0% and " + pct(w) + "</b> is consistent with the data, and no estimator can narrow it down. The upper end is σ² / (σ² + (μ − t)²). More test points per group pin down more moments and shrink the set.";
  }
  $("gc-w-s").addEventListener("input", drawWorlds);

  /* ---------- 3. simulation: naive vs debiased RMS gap ---------- */
  function randn() { var u = 0; while (u === 0) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); }
  function gamma(k) { var d = k - 1 / 3, c = 1 / Math.sqrt(9 * d); for (;;) { var x, v; do { x = randn(); v = 1 + c * x; } while (v <= 0); v = v * v * v; var u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; } }
  var sG = 400, sM = 5, sS = 0.04, TAU = 0.9, REPS = 200;
  function q(sorted, p) { return sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]; }
  function simulate() {
    var c = [], g, truth = 0;
    if (sS === 0) for (g = 0; g < sG; g++) c.push(TAU);
    else { var k = TAU * (1 - TAU) / (sS * sS) - 1; for (g = 0; g < sG; g++) { var x = gamma(TAU * k), y = gamma((1 - TAU) * k); c.push(x / (x + y)); } }
    for (g = 0; g < sG; g++) truth += (c[g] - TAU) * (c[g] - TAU);
    truth = Math.sqrt(truth / sG);
    var nv = [], db = [], zero = 0;
    for (var r = 0; r < REPS; r++) { var sn = 0, se = 0;
      for (g = 0; g < sG; g++) { var h = 0, cg = c[g]; for (var i = 0; i < sM; i++) if (Math.random() < cg) h++;
        var ch = h / sM, n2 = (ch - TAU) * (ch - TAU); sn += n2; se += n2 - ch * (1 - ch) / (sM - 1); }
      nv.push(Math.sqrt(sn / sG)); var e = se / sG; if (e <= 0) zero++; db.push(Math.sqrt(Math.max(0, e))); }
    nv.sort(function (a, b) { return a - b; }); db.sort(function (a, b) { return a - b; });
    return {truth: truth, n: [q(nv, 0.05), q(nv, 0.5), q(nv, 0.95)], d: [q(db, 0.05), q(db, 0.5), q(db, 0.95)], zero: zero / REPS};
  }
  function drawSim() {
    seg($("gc-s-g"), [100, 400, 1500], sG, function (k) { sG = k; drawSim(); });
    seg($("gc-s-m"), [2, 5, 10, 20, 50], sM, function (k) { sM = k; drawSim(); });
    seg($("gc-s-s"), [0, 0.02, 0.04, 0.06], sS, function (k) { sS = k; drawSim(); }, function (k) { return k === 0 ? "0 (exact coverage)" : (100 * k).toFixed(0) + " pts"; });
    var R = simulate(), max = 1.18 * Math.max(R.n[2], R.d[2], R.truth, 0.02), X = function (v) { return 100 * v / max; };
    function row(cls, name, lo, md, hi, txt) {
      return '<div class="gc-row ' + cls + '"><span class="nm">' + name + '</span><span class="tr"><span class="tv" style="left:' + X(R.truth) + '%"></span>' +
        (lo === null ? "" : '<span class="rg" style="left:' + X(lo) + "%;width:" + (X(hi) - X(lo)) + '%"></span><span class="md" style="left:' + X(md) + '%"></span>') + '</span><span class="vl">' + txt + "</span></div>";
    }
    $("gc-s-rows").innerHTML = row("truth", "True RMS gap", null, 0, 0, R.truth.toFixed(3)) +
      row("naive", "Naive", R.n[0], R.n[1], R.n[2], R.n[1].toFixed(3)) + row("deb", "Debiased", R.d[0], R.d[1], R.d[2], R.d[1].toFixed(3));
    $("gc-s-axis").innerHTML = "<span>0</span><span>RMS coverage gap</span><span>" + max.toFixed(2) + "</span>";
    var t = R.truth < 1e-9
      ? "The true gap is zero. The naive RMS gap still shows <b>" + R.n[1].toFixed(3) + "</b>, the noise floor for " + sM + " test points per group. The debiased estimate is zero in " + pct(R.zero) + " of the test sets and below " + R.d[2].toFixed(3) + " in 95% of them."
      : "The naive RMS gap has median <b>" + R.n[1].toFixed(3) + "</b>, " + (R.n[1] / R.truth).toFixed(1) + " times the true " + R.truth.toFixed(3) + ". The debiased estimate has median <b>" + R.d[1].toFixed(3) + "</b> and scatters around the truth" + (R.zero > 0.02 ? "; it is zero in " + pct(R.zero) + " of the test sets." : ".");
    $("gc-s-read").innerHTML = t;
  }
  $("gc-s-go").onclick = drawSim;

  /* ---------- 4. real data: per-method means over 200 splits (run summarize_p4r/20261013T135857Z-906c785), one-sided ratios (summarize_r4/20261014T232714Z-a110f10) ---------- */
  var D = [{"name":"ACS (state × race)","score":"CQR","G":408,"small":0.35,"noise":1.01,"ratio":3.2,"floorShare":0.96,"rk":0.9,"floor":0.0517,"bestNaive":"standard split","bestEst":"conditional (ridge)","one":[1.57,5.47,0.13,0.42],"m":[{"n":"clustered","k":"clustered_default","naive":0.053374,"est":0.016498,"cov":0.9},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.053149,"est":0.015641,"cov":0.9},{"n":"Mondrian","k":"mondrian","naive":0.08415,"est":0.083779,"cov":0.909},{"n":"shrink κ=10","k":"shrink_k10","naive":0.058588,"est":0.022978,"cov":0.904},{"n":"shrink κ=200","k":"shrink_k200","naive":0.053707,"est":0.01678,"cov":0.903},{"n":"shrink κ=50","k":"shrink_k50","naive":0.055628,"est":0.019524,"cov":0.904},{"n":"standard split","k":"standard","naive":0.053118,"est":0.015675,"cov":0.9}]},{"name":"Airlines (origin)","score":"CQR","G":352,"small":0.27,"noise":0.61,"ratio":1.61,"floorShare":0.65,"rk":1.0,"floor":0.0472,"bestNaive":"standard split","bestEst":"standard split","one":[1.13,1.87,0.1,0.12],"m":[{"n":"clustered","k":"clustered_default","naive":0.071295,"est":0.043014,"cov":0.899},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.071118,"est":0.042802,"cov":0.899},{"n":"Mondrian","k":"mondrian","naive":0.08006,"est":0.079324,"cov":0.91},{"n":"shrink κ=10","k":"shrink_k10","naive":0.077536,"est":0.051763,"cov":0.904},{"n":"shrink κ=200","k":"shrink_k200","naive":0.072386,"est":0.045004,"cov":0.903},{"n":"shrink κ=50","k":"shrink_k50","naive":0.07472,"est":0.048307,"cov":0.904},{"n":"standard split","k":"standard","naive":0.07078,"est":0.042442,"cov":0.9}]},{"name":"Airlines (origin)","score":"abs","G":352,"small":0.27,"noise":0.88,"ratio":1.87,"floorShare":0.76,"rk":0.62,"floor":0.0472,"bestNaive":"shrink κ=200","bestEst":"conditional (ridge)","one":[1.3,2.42,0.13,0.22],"m":[{"n":"clustered","k":"clustered_default","naive":0.06156,"est":0.032751,"cov":0.9},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.062099,"est":0.03182,"cov":0.9},{"n":"Mondrian","k":"mondrian","naive":0.080053,"est":0.07933,"cov":0.91},{"n":"shrink κ=10","k":"shrink_k10","naive":0.065172,"est":0.036838,"cov":0.906},{"n":"shrink κ=200","k":"shrink_k200","naive":0.061367,"est":0.03215,"cov":0.906},{"n":"shrink κ=50","k":"shrink_k50","naive":0.062697,"est":0.033741,"cov":0.907},{"n":"standard split","k":"standard","naive":0.061436,"est":0.032783,"cov":0.9}]},{"name":"Zurich (stop)","score":"CQR","G":1528,"small":0.06,"noise":0.41,"ratio":1.25,"floorShare":0.54,"rk":1.0,"floor":0.0271,"bestNaive":"shrink κ=200","bestEst":"shrink κ=200","one":[1.83,2.3,0.24,0.29],"m":[{"n":"clustered","k":"clustered_default","naive":0.050034,"est":0.039975,"cov":0.902},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.049129,"est":0.038768,"cov":0.899},{"n":"Mondrian","k":"mondrian","naive":0.089862,"est":0.089387,"cov":0.959},{"n":"shrink κ=10","k":"shrink_k10","naive":0.053206,"est":0.044061,"cov":0.913},{"n":"shrink κ=200","k":"shrink_k200","naive":0.049037,"est":0.038765,"cov":0.904},{"n":"shrink κ=50","k":"shrink_k50","naive":0.050201,"est":0.040402,"cov":0.909},{"n":"standard split","k":"standard","naive":0.049227,"est":0.038865,"cov":0.9}]},{"name":"Zurich (stop)","score":"abs","G":1528,"small":0.06,"noise":0.22,"ratio":1.15,"floorShare":0.44,"rk":1.0,"floor":0.0271,"bestNaive":"conditional (ridge)","bestEst":"conditional (ridge)","one":[1.45,1.66,0.19,0.21],"m":[{"n":"clustered","k":"clustered_default","naive":0.061997,"est":0.053937,"cov":0.899},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.059768,"est":0.051478,"cov":0.9},{"n":"Mondrian","k":"mondrian","naive":0.089813,"est":0.089333,"cov":0.959},{"n":"shrink κ=10","k":"shrink_k10","naive":0.061181,"est":0.053133,"cov":0.913},{"n":"shrink κ=200","k":"shrink_k200","naive":0.060651,"est":0.052457,"cov":0.904},{"n":"shrink κ=50","k":"shrink_k50","naive":0.060027,"est":0.051855,"cov":0.909},{"n":"standard split","k":"standard","naive":0.061845,"est":0.053772,"cov":0.9}]},{"name":"DBpedia (L3 class)","score":"APS","G":219,"small":0.0,"noise":0.29,"ratio":1.27,"floorShare":0.57,"rk":1.0,"floor":0.016,"bestNaive":"shrink κ=200","bestEst":"shrink κ=200","one":[2.35,3.04,1.4,1.77],"m":[{"n":"clustered","k":"clustered_default","naive":0.02786,"est":0.021978,"cov":0.904},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.026139,"est":0.019637,"cov":0.9},{"n":"Mondrian","k":"mondrian","naive":0.062045,"est":0.060569,"cov":0.92},{"n":"shrink κ=10","k":"shrink_k10","naive":0.037394,"est":0.033831,"cov":0.911},{"n":"shrink κ=200","k":"shrink_k200","naive":0.025977,"est":0.019242,"cov":0.902},{"n":"shrink κ=50","k":"shrink_k50","naive":0.028674,"est":0.023341,"cov":0.906},{"n":"standard split","k":"standard","naive":0.026298,"est":0.019878,"cov":0.9}]},{"name":"DBpedia (L3 class)","score":"LAC","G":219,"small":0.0,"noise":0.05,"ratio":1.03,"floorShare":0.2,"rk":1.0,"floor":0.016,"bestNaive":"Mondrian","bestEst":"Mondrian","one":[0.68,0.68,0.19,0.19],"m":[{"n":"clustered","k":"clustered_default","naive":0.078467,"est":0.07586,"cov":0.891},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.067321,"est":0.064299,"cov":0.902},{"n":"Mondrian","k":"mondrian","naive":0.062151,"est":0.060721,"cov":0.92},{"n":"shrink κ=10","k":"shrink_k10","naive":0.071897,"est":0.069129,"cov":0.907},{"n":"shrink κ=200","k":"shrink_k200","naive":0.0875,"est":0.085763,"cov":0.902},{"n":"shrink κ=50","k":"shrink_k50","naive":0.080931,"est":0.078625,"cov":0.905},{"n":"standard split","k":"standard","naive":0.090931,"est":0.08924,"cov":0.9}]},{"name":"CIFAR-100 (class)","score":"APS","G":100,"small":0.0,"noise":0.37,"ratio":1.32,"floorShare":0.64,"rk":0.9,"floor":0.0269,"bestNaive":"conditional (ridge)","bestEst":"conditional (ridge)","one":[1.38,1.68,0.81,0.92],"m":[{"n":"clustered","k":"clustered_default","naive":0.042738,"est":0.033303,"cov":0.901},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.040627,"est":0.030346,"cov":0.9},{"n":"Mondrian","k":"mondrian","naive":0.058044,"est":0.053052,"cov":0.923},{"n":"shrink κ=10","k":"shrink_k10","naive":0.044281,"est":0.03684,"cov":0.905},{"n":"shrink κ=200","k":"shrink_k200","naive":0.041265,"est":0.031073,"cov":0.899},{"n":"shrink κ=50","k":"shrink_k50","naive":0.040963,"est":0.03111,"cov":0.9},{"n":"standard split","k":"standard","naive":0.042065,"est":0.03186,"cov":0.899}]},{"name":"CIFAR-100 (class)","score":"LAC","G":100,"small":0.0,"noise":0.3,"ratio":1.21,"floorShare":0.57,"rk":0.71,"floor":0.0269,"bestNaive":"shrink κ=10","bestEst":"shrink κ=50","one":[1.18,1.31,0.71,0.74],"m":[{"n":"clustered","k":"clustered_default","naive":0.047513,"est":0.040364,"cov":0.902},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.045268,"est":0.037456,"cov":0.901},{"n":"Mondrian","k":"mondrian","naive":0.058355,"est":0.053331,"cov":0.923},{"n":"shrink κ=10","k":"shrink_k10","naive":0.044863,"est":0.036953,"cov":0.903},{"n":"shrink κ=200","k":"shrink_k200","naive":0.047556,"est":0.039154,"cov":0.899},{"n":"shrink κ=50","k":"shrink_k50","naive":0.045617,"est":0.036649,"cov":0.898},{"n":"standard split","k":"standard","naive":0.049257,"est":0.040663,"cov":0.9}]},{"name":"Diamonds (color × clarity)","score":"CQR","G":56,"small":0.1,"noise":0.44,"ratio":1.34,"floorShare":0.62,"rk":0.9,"floor":0.0264,"bestNaive":"conditional (ridge)","bestEst":"conditional (ridge)","one":[1.34,1.72,0.37,0.41],"m":[{"n":"clustered","k":"clustered_default","naive":0.042691,"est":0.031749,"cov":0.897},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.041568,"est":0.030524,"cov":0.9},{"n":"Mondrian","k":"mondrian","naive":0.056555,"est":0.053484,"cov":0.912},{"n":"shrink κ=10","k":"shrink_k10","naive":0.053101,"est":0.046242,"cov":0.907},{"n":"shrink κ=200","k":"shrink_k200","naive":0.042006,"est":0.031302,"cov":0.902},{"n":"shrink κ=50","k":"shrink_k50","naive":0.045473,"est":0.036403,"cov":0.905},{"n":"standard split","k":"standard","naive":0.042268,"est":0.031248,"cov":0.9}]},{"name":"Diamonds (color × clarity)","score":"abs","G":56,"small":0.1,"noise":0.26,"ratio":1.11,"floorShare":0.42,"rk":0.9,"floor":0.0263,"bestNaive":"Mondrian","bestEst":"conditional (ridge)","one":[0.89,0.94,0.23,0.23],"m":[{"n":"clustered","k":"clustered_default","naive":0.06404,"est":0.057857,"cov":0.897},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.059935,"est":0.053065,"cov":0.901},{"n":"Mondrian","k":"mondrian","naive":0.056669,"est":0.053543,"cov":0.912},{"n":"shrink κ=10","k":"shrink_k10","naive":0.066895,"est":0.062095,"cov":0.906},{"n":"shrink κ=200","k":"shrink_k200","naive":0.061624,"est":0.055466,"cov":0.903},{"n":"shrink κ=50","k":"shrink_k50","naive":0.062216,"est":0.056243,"cov":0.905},{"n":"standard split","k":"standard","naive":0.063571,"est":0.056851,"cov":0.9}]},{"name":"House sales (zipcode)","score":"CQR","G":70,"small":0.08,"noise":0.57,"ratio":1.59,"floorShare":0.77,"rk":0.9,"floor":0.0365,"bestNaive":"shrink κ=200","bestEst":"shrink κ=200","one":[1.35,1.87,0.63,0.82],"m":[{"n":"clustered","k":"clustered_default","naive":0.047853,"est":0.030526,"cov":0.898},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.046111,"est":0.02751,"cov":0.901},{"n":"Mondrian","k":"mondrian","naive":0.063397,"est":0.055049,"cov":0.922},{"n":"shrink κ=10","k":"shrink_k10","naive":0.052636,"est":0.040334,"cov":0.915},{"n":"shrink κ=200","k":"shrink_k200","naive":0.045634,"est":0.027053,"cov":0.905},{"n":"shrink κ=50","k":"shrink_k50","naive":0.045824,"est":0.028607,"cov":0.911},{"n":"standard split","k":"standard","naive":0.047125,"est":0.029714,"cov":0.9}]},{"name":"House sales (zipcode)","score":"abs","G":70,"small":0.09,"noise":0.55,"ratio":1.27,"floorShare":0.67,"rk":1.0,"floor":0.0365,"bestNaive":"shrink κ=50","bestEst":"shrink κ=50","one":[1.1,1.19,0.5,0.51],"m":[{"n":"clustered","k":"clustered_default","naive":0.058587,"est":0.047955,"cov":0.895},{"n":"conditional (ridge)","k":"gibbs_ridge_cv","naive":0.053337,"est":0.04101,"cov":0.901},{"n":"Mondrian","k":"mondrian","naive":0.063887,"est":0.055718,"cov":0.921},{"n":"shrink κ=10","k":"shrink_k10","naive":0.053581,"est":0.042046,"cov":0.916},{"n":"shrink κ=200","k":"shrink_k200","naive":0.054242,"est":0.042633,"cov":0.904},{"n":"shrink κ=50","k":"shrink_k50","naive":0.050271,"est":0.037464,"cov":0.91},{"n":"standard split","k":"standard","naive":0.058383,"est":0.047827,"cov":0.9}]}];
  function short(c) { return c.name.split(" (")[0] + " · " + c.score; }
  var cell = 0, order = "naive", ROW = 34;
  function drawReal() {
    seg($("gc-r-cells"), D.map(function (c, i) { return i; }), cell, function (k) { cell = k; drawReal(); }, function (k) { return short(D[k]); });
    $("gc-r-sort").querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.v === order); });
    var C = D[cell], runs = C.m, max = 1.08 * Math.max.apply(null, runs.map(function (r) { return r.naive; }).concat([C.floor]));
    var sorted = runs.slice().sort(function (a, b) { return a[order] - b[order]; });
    var box = $("gc-r-rank"); box.style.height = runs.length * ROW + "px";
    box.style.setProperty("--gc-top", order === "naive" ? "var(--gc-gray)" : "var(--gc-blue)");
    var keep = {}; box.querySelectorAll(".row").forEach(function (e) { keep[e.dataset.k] = e; });
    sorted.forEach(function (r, i) {
      var e = keep[r.k];
      if (!e) { e = document.createElement("div"); e.dataset.k = r.k;
        e.innerHTML = '<span class="no"></span><span class="nm"></span><span class="tr"><span class="fn"></span><span class="fe"></span></span><span class="vl"></span>'; box.appendChild(e); }
      e.className = "row" + (i === 0 ? " top" : ""); e.style.top = i * ROW + "px";
      e.querySelector(".no").textContent = i + 1; e.querySelector(".nm").textContent = r.n;
      e.querySelector(".fn").style.width = 100 * r.naive / max + "%"; e.querySelector(".fe").style.width = 100 * r.est / max + "%";
      e.querySelector(".vl").innerHTML = r.naive.toFixed(3) + " → <b>" + r.est.toFixed(3) + "</b>";
    });
    var line = box.querySelector(".fl");
    if (!line) { line = document.createElement("div"); line.className = "fl"; line.innerHTML = "<span></span>"; box.appendChild(line); }
    var tr = box.querySelector(".tr"); line.style.left = tr.offsetLeft + tr.offsetWidth * C.floor / max + "px";
    line.querySelector("span").textContent = "exact-coverage floor " + C.floor.toFixed(3);
    var s = "<b>" + C.name + ", " + C.score + " score:</b> " + C.G.toLocaleString("en-US") + " groups" + (C.small > 0 ? ", " + pct(C.small) + " of them with fewer than 20 test points" : ", all with at least 20 test points") + ". ";
    s += "A method with exact coverage in every group would show a CovGap of " + C.floor.toFixed(3) + " on these test counts, " + pct(C.floorShare) + " of what the methods report (median). ";
    s += "The naive CovGap is " + C.ratio.toFixed(2) + " times our estimate. ";
    s += C.bestNaive === C.bestEst ? "Both metrics rank <b>" + C.bestNaive + "</b> first." : "The naive metric ranks <b>" + C.bestNaive + "</b> first, the estimate <b>" + C.bestEst + "</b>. Across the runs with such a change, the paired test separates the two leaders in at most 4.5% of splits.";
    $("gc-r-read").innerHTML = s;
  }
  $("gc-r-sort").querySelectorAll("button").forEach(function (b) { b.onclick = function () { order = b.dataset.v; drawReal(); }; });
  window.addEventListener("resize", drawReal);

  /* ---------- 5. Mondrian / standard split: estimated CovGap and shortfall ratios (log axis) ---------- */
  function drawOne() {
    var lo = Math.log(0.1), hi = Math.log(8), X = function (v) { return 100 * (Math.log(v) - lo) / (hi - lo); };
    $("gc-o-dots").innerHTML = D.map(function (c) {
      var a = X(c.one[1]), b = X(c.one[3]);
      return '<div class="gc-dot"><span class="nm">' + short(c) + '</span><span class="tr"><span class="one" style="left:' + X(1) + '%"></span><span class="lk" style="left:' + Math.min(a, b) + "%;width:" + Math.abs(a - b) + '%"></span>' +
        '<span class="pt c" style="left:' + a + '%" title="estimated CovGap ratio ' + c.one[1].toFixed(2) + '"></span><span class="pt s' + (c.one[3] > 1 ? " over" : "") + '" style="left:' + b + '%" title="estimated shortfall ratio ' + c.one[3].toFixed(2) + '"></span></span></div>';
    }).join("");
    $("gc-o-ax").innerHTML = [0.1, 0.2, 0.5, 1, 2, 5].map(function (v) { return '<span style="left:' + X(v) + '%">' + v + "×</span>"; }).join("") + "<em>Mondrian relative to standard split conformal</em>";
  }

  drawFloor(); drawWorlds(); drawSim(); drawReal(); drawOne();
})();
</script>
{% endraw %}
