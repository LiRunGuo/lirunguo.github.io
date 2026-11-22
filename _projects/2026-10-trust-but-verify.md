---
title: "Trust but Verify: Diagnosing Causal Prior-Data Fitted Networks and Testing When to Defer"
collection: projects
permalink: /projects/2026-10-trust-but-verify/
date: 2026-10-15
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "A diagnosis of the three released causal PFNs and a test, certify-or-defer, that replaces a PFN only where it is significantly worse than a classical reference. Under a protocol frozen in advance the verdict is mixed, and a generic TabPFN inside a doubly robust learner is the more accurate default."
teaser: "/images/projects/2026-10-trust-but-verify/teaser.png"
tags: ["causal inference", "treatment effects", "tabular foundation models", "prior-data fitted networks", "pre-registration"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-10-trust-but-verify"
---

<style>
/* Trust-but-Verify project page: styles scoped to .tbv / .tbv-*; colors follow the site's theme toggle.
   Series hues (CausalPFN, CausalFM, Do-PFN) are the first three slots of a palette checked for
   color-vision-deficiency separation on both surfaces; identity is also carried by labels and line style. */
.tbv{--tbv-blue:#1f5fa8;--tbv-blue-soft:rgba(31,95,168,.08);--tbv-gray:#6f767e;--tbv-gray-soft:rgba(127,127,127,.09);--tbv-line:rgba(127,127,127,.28);--tbv-card:rgba(127,127,127,.05);--tbv-c1:#2a78d6;--tbv-c2:#eb6834;--tbv-c3:#1baf7a;--tbv-good:#0a8a0a;--tbv-bad:#d03b3b}
html[data-theme="dark"] .tbv{--tbv-blue:#78b0f0;--tbv-blue-soft:rgba(120,176,240,.14);--tbv-gray:#c3c9cf;--tbv-gray-soft:rgba(255,255,255,.07);--tbv-line:rgba(255,255,255,.2);--tbv-card:rgba(255,255,255,.04);--tbv-c1:#3987e5;--tbv-c2:#d95926;--tbv-c3:#199e70;--tbv-good:#3dbb3d;--tbv-bad:#ee7b7b}
.tbv-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.tbv-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--tbv-blue-soft);color:var(--tbv-blue)}
.tbv-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.tbv-tldr b{color:var(--tbv-blue)}
.tbv-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.tbv-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.tbv-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.tbv-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--tbv-line)}
.tbv-btn.muted{background:transparent;color:var(--tbv-gray)!important;border:1px dashed var(--tbv-line);cursor:default}
.tbv-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .tbv-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.tbv-lead{font-size:1em;line-height:1.65;margin:0 0 1em}
.tbv-fig{display:block;position:relative;margin:0 0 1.6em;padding:.8em;border:1px solid var(--tbv-line);border-radius:14px;background:var(--tbv-card)}
.tbv-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.tbv-fig.narrow img{max-width:min(100%,560px);margin:0 auto}
.tbv-fig.wide .scroll{overflow-x:auto;border-radius:8px}
.tbv-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.tbv-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--tbv-blue);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .tbv-tag{color:#16202b}
.tbv-fig figcaption b.tbv-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.tbv-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.tbv-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--tbv-line);background:var(--tbv-card);position:relative;overflow:hidden}
.tbv-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--tbv-c,var(--tbv-blue))}
.tbv-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;font-variant-numeric:tabular-nums}
.tbv-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;opacity:.88}
.tbv-defs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em;margin:0 0 1.6em}
.tbv-def{padding:.65em .85em;border-radius:12px;border:1px solid var(--tbv-line);background:var(--tbv-card);font-size:.86em;line-height:1.5}
.tbv-def b{display:inline-block;margin-right:.35em;color:var(--tbv-blue)}
.tbv-def.wide{grid-column:1/-1}
.tbv-why{font-size:.9em;line-height:1.55;margin:0 0 1.6em;padding:.6em .9em;border-radius:10px;background:var(--tbv-gray-soft)}
.tbv-box{padding:1em 1.2em;border-radius:14px;background:var(--tbv-blue-soft);border:1px solid var(--tbv-line);margin:0 0 1.6em;line-height:1.65;font-size:.95em}
.tbv-box p{margin:0}
.tbv-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--tbv-line);border-radius:14px;background:var(--tbv-card);overflow-x:auto}
.tbv-table table{font-size:.8em;margin:.6em 0;min-width:560px;width:100%}
.tbv-table p{font-size:.86em;margin:.2em}
.tbv-table thead,.tbv-tbl thead{background:var(--tbv-gray-soft)}.tbv-table th,.tbv-tbl th{background:transparent;color:var(--global-text-color)}
.tbv-how{font-size:.86em;line-height:1.55;margin:.5em .2em 0;padding:.5em .75em;border-radius:8px;background:var(--tbv-gray-soft)}
/* interactive controls */
.tbv-ix{font-size:.9em}
.tbv-ctl{display:grid;grid-template-columns:auto 1fr;gap:.35em .9em;align-items:center;margin:0 0 .7em}
.tbv-ctl .l{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65}
.tbv-seg{display:flex;flex-wrap:wrap;gap:.35em}
.tbv-seg button{font:inherit;font-size:.86em;line-height:1.3;padding:.3em .85em;border-radius:999px;border:1px solid var(--tbv-line);background:transparent;color:var(--global-text-color);cursor:pointer;transition:background .15s,border-color .15s}
.tbv-seg button:hover{border-color:var(--tbv-gray)}
.tbv-seg button[aria-pressed=true]{background:var(--global-text-color);border-color:var(--global-text-color);color:var(--global-bg-color)}
.tbv-meta{display:flex;flex-wrap:wrap;gap:.35em .5em;margin:0 0 .8em;font-size:.84em}
.tbv-meta span{padding:.2em .65em;border-radius:8px;background:var(--tbv-gray-soft);font-variant-numeric:tabular-nums}
.tbv-read{margin:.8em 0 0;padding:.5em .8em;border-left:3px solid var(--tbv-blue);border-radius:4px;background:var(--tbv-blue-soft);font-size:.92em;line-height:1.5}
/* ranked bars (final benchmarks) */
.tbv-rank{position:relative;margin:.9em 0 .2em}
.tbv-rank .row{position:absolute;left:0;right:0;height:32px;display:grid;grid-template-columns:210px 1fr 150px;gap:.6em;align-items:center;transition:top .5s cubic-bezier(.2,.8,.2,1)}
.tbv-rank .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.92em}
.tbv-rank .row.ours .nm{font-weight:700}
.tbv-rank .tr{position:relative;height:14px;margin-right:64px}
.tbv-rank .fl{position:absolute;left:0;top:0;bottom:0;border-radius:0 4px 4px 0;min-width:3px;transition:width .5s cubic-bezier(.2,.8,.2,1)}
.tbv-rank .vl{position:absolute;top:-4px;padding-left:7px;font-size:.88em;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;transition:left .5s cubic-bezier(.2,.8,.2,1)}
.tbv-rank .vs{font-size:.78em;white-space:nowrap;opacity:.85}
.tbv-rank .r-pfn .fl{background:transparent;box-shadow:inset 0 0 0 2px var(--tbv-p)}
.tbv-rank .r-ours .fl{background:var(--tbv-p)}
.tbv-rank .r-ref .fl{background:var(--tbv-gray);opacity:.6}
.tbv-rank .r-naive .fl{background:repeating-linear-gradient(45deg,var(--tbv-gray) 0 4px,transparent 4px 8px);box-shadow:inset 0 0 0 1px var(--tbv-gray)}
.tbv-rank .r-tabdr .fl{background:var(--global-text-color)}
.tbv-rank .row:hover .tr,.tbv-rank .row:focus .tr{outline:1px dashed var(--tbv-line);outline-offset:5px}
.tbv-key{display:flex;flex-wrap:wrap;gap:.3em 1.1em;font-size:.8em;margin:1.1em 0 .9em;opacity:.9}
.tbv-key i{display:inline-block;width:22px;height:10px;margin-right:.4em;vertical-align:-1px;border-radius:0 3px 3px 0}
.tbv-key .k-pfn{box-shadow:inset 0 0 0 2px var(--tbv-p)}.tbv-key .k-ours{background:var(--tbv-p)}
.tbv-key .k-ref{background:var(--tbv-gray);opacity:.6}.tbv-key .k-tabdr{background:var(--global-text-color)}
.tbv-key .k-naive{background:repeating-linear-gradient(45deg,var(--tbv-gray) 0 4px,transparent 4px 8px);box-shadow:inset 0 0 0 1px var(--tbv-gray)}
/* line chart (development platform) */
.tbv-legend{display:flex;flex-wrap:wrap;gap:.25em 1.1em;font-size:.84em;margin:.2em 0 .6em}
.tbv-legend svg{vertical-align:-3px;margin-right:.35em}
.tbv-plot{position:relative}
.tbv-plot svg{display:block;width:100%;height:auto;overflow:visible}
.tbv-plot text{fill:currentColor;font-size:12px;font-family:inherit}
.tbv-plot .ax{opacity:.7}.tbv-plot .grid{stroke:var(--tbv-line);stroke-width:1}
.tbv-plot .cross{stroke:var(--tbv-gray);stroke-width:1;stroke-dasharray:3 3;opacity:0}
.tbv-s{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.tbv-s.m-causalpfn,.tbv-d.m-causalpfn{stroke:var(--tbv-c1)}.tbv-d.m-causalpfn{fill:var(--tbv-c1)}
.tbv-s.m-causalfm,.tbv-d.m-causalfm{stroke:var(--tbv-c2)}.tbv-d.m-causalfm{fill:var(--tbv-c2)}
.tbv-s.m-dopfn,.tbv-d.m-dopfn{stroke:var(--tbv-c3)}.tbv-d.m-dopfn{fill:var(--tbv-c3)}
.tbv-s.m-bcf,.tbv-d.m-bcf{stroke:var(--tbv-gray)}.tbv-d.m-bcf{fill:var(--tbv-gray)}
.tbv-s.m-dr_lgbm{stroke:var(--tbv-gray);stroke-dasharray:5 4}.tbv-d.m-dr_lgbm{stroke:var(--tbv-gray);fill:var(--global-bg-color)}
.tbv-s.m-tabpfn_dr{stroke:var(--global-text-color);stroke-width:2.6}.tbv-d.m-tabpfn_dr{fill:var(--global-text-color);stroke:var(--global-text-color)}
.tbv-ring{fill:none;stroke:var(--global-bg-color);stroke-width:2}
.tbv-tip{position:absolute;z-index:5;pointer-events:none;min-width:150px;padding:.5em .7em;border-radius:8px;border:1px solid var(--tbv-line);background:var(--global-bg-color);box-shadow:0 4px 14px rgba(0,0,0,.18);font-size:.82em;line-height:1.5;opacity:0;transition:opacity .12s}
.tbv-tip b.h{display:block;margin-bottom:.15em}
.tbv-tip .r{display:flex;justify-content:space-between;gap:1.2em;font-variant-numeric:tabular-nums}
.tbv-tip .r svg{vertical-align:-2px;margin-right:.35em}
.tbv-tbl{margin:.6em 0 0;font-size:.84em}
.tbv-tbl summary{cursor:pointer;opacity:.8}
.tbv-tbl table{font-size:.92em;margin:.5em 0 0;width:100%;font-variant-numeric:tabular-nums}
/* method flow */
.tbv-flow{display:grid;gap:.25em;margin:0 0 1.2em}
.tbv-flow .lvl{display:grid;gap:.6em}
.tbv-flow .lvl.c3{grid-template-columns:repeat(3,minmax(0,1fr))}
.tbv-flow .lvl.c2{grid-template-columns:repeat(2,minmax(0,1fr))}
.tbv-node{padding:.7em .85em;border-radius:12px;border:1px solid var(--tbv-line);background:var(--tbv-card);font-size:.86em;line-height:1.5}
.tbv-node b.h{display:block;font-size:1.02em;margin-bottom:.15em}
.tbv-node.key{border-color:var(--tbv-blue);background:var(--tbv-blue-soft)}
.tbv-node.keep{border-left:4px solid var(--tbv-c1)}
.tbv-node.defer{border-left:4px solid var(--tbv-gray)}
.tbv-node .eq{display:block;margin:.4em 0 .1em;padding:.35em .5em;border-radius:6px;background:var(--tbv-gray-soft);text-align:center;font-size:1.04em;font-variant-numeric:tabular-nums}
.tbv-arrow{text-align:center;color:var(--tbv-gray);font-size:1.05em;line-height:1.1}
/* verdict */
.tbv-chip{display:inline-flex;align-items:center;gap:.3em;padding:.08em .6em;border-radius:999px;font-size:.92em;font-weight:700;white-space:nowrap;border:1px solid currentColor}
.tbv-chip.pass{color:var(--tbv-good)}.tbv-chip.fail{color:var(--tbv-bad)}
.tbv-verdict td,.tbv-verdict th{vertical-align:top}
.tbv-verdict table{font-size:.86em}.tbv-verdict td small{display:block;font-size:.92em;opacity:.9;line-height:1.5;margin-top:.35em}.tbv-verdict th:first-child{width:40%}
.tbv-time{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.5em;margin:0 0 1.6em;counter-reset:tbv}
.tbv-time div{position:relative;padding:.7em .75em .7em;border-radius:12px;border:1px solid var(--tbv-line);background:var(--tbv-card);font-size:.8em;line-height:1.45}
.tbv-time div:before{counter-increment:tbv;content:counter(tbv);display:inline-grid;place-items:center;width:1.6em;height:1.6em;border-radius:50%;background:var(--tbv-blue);color:#fff;font-weight:800;font-size:.9em;margin-right:.4em}
html[data-theme="dark"] .tbv-time div:before{color:#16202b}
.tbv-time b{font-size:1.04em}
.tbv-time span{display:block;margin-top:.35em;opacity:.9}
.tbv-take{margin:0 0 1.2em;padding:0;list-style:none;display:grid;gap:.55em}
.tbv-take li{padding:.65em .9em .65em 1em;border-left:3px solid var(--tbv-blue);border-radius:4px;background:var(--tbv-card);font-size:.92em;line-height:1.55}
.tbv-take li.scope{border-left-color:var(--tbv-gray)}
.tbv-abs{margin:0 0 1.6em;border:1px solid var(--tbv-line);border-radius:12px;background:var(--tbv-card)}
.tbv-abs summary{cursor:pointer;padding:.6em .9em;font-weight:700;font-size:.92em}
.tbv-abs p{margin:0;padding:0 .95em .9em;font-size:.9em;line-height:1.65}
/* the theme's fixed table of contents (220px, right) covers a full-width title: keep the header in the text column */
@media screen and (min-width:1200px){.page__inner-wrap>header{max-width:calc(100% - 240px)}}
@media (max-width:900px){.tbv-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.tbv-time{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:760px){.tbv-fig.wide img{min-width:760px}.tbv-defs,.tbv-flow .lvl.c3,.tbv-flow .lvl.c2{grid-template-columns:1fr}.tbv-ctl{grid-template-columns:1fr}.tbv-tldr{font-size:1.04em}
 .tbv-rank .row{height:50px;grid-template-columns:1fr auto;grid-template-rows:auto auto;gap:.1em .6em}.tbv-rank .nm{grid-column:1}.tbv-rank .vs{grid-column:2;grid-row:1;text-align:right}.tbv-rank .tr{grid-column:1/-1;grid-row:2}}
</style>

<div class="tbv">
<div class="tbv-kicker"><span>Causal inference</span><span>Tabular foundation models</span><span>Pre-registered evaluation</span></div>
<p class="tbv-tldr">A causal PFN returns treatment effects in one forward pass, but nothing in its output says whether your data fit its prior. We diagnose the three released models and build <b>certify-or-defer</b>, a test that replaces the PFN only where the data show it is worse than a classical estimator. The evaluation was fixed in advance, its verdict is mixed, and we report all of it.</p>
<div class="tbv-btns"><a class="tbv-btn" href="#the-diagnosis">The diagnosis</a><a class="tbv-btn ghost" href="#certify-or-defer">The method</a><a class="tbv-btn ghost" href="#the-pre-registered-verdict">The verdict</a><a class="tbv-btn ghost" href="#takeaways">Takeaways</a><span class="tbv-btn muted">Paper &amp; code: coming soon</span></div>
<div class="tbv-stats">
<div class="tbv-stat" style="--tbv-c:var(--global-text-color)"><div class="v">37 of 50</div><div class="d">simulation designs where a generic TabPFN inside a doubly robust learner beats the best classical method (worse in 2). It also has the best mean rank of all methods.</div></div>
<div class="tbv-stat" style="--tbv-c:var(--tbv-c1)"><div class="v">1.00 and 0.55</div><div class="d">actual coverage of CausalPFN's 90% CATE intervals and 95% ATE intervals in the base design: one far too wide, the other too narrow</div></div>
<div class="tbv-stat" style="--tbv-c:var(--tbv-good)"><div class="v">8 of 8</div><div class="d">non-inferiority checks passed: on four benchmarks and for both PFNs, the wrapper never loses accuracy to the model it wraps</div></div>
<div class="tbv-stat" style="--tbv-c:var(--tbv-bad)"><div class="v">18.9%</div><div class="d">of the wrapper's replacements for CausalPFN made the estimate worse. The bound we registered was 10%, so this claim failed.</div></div>
</div>
<figure class="tbv-fig" id="tbv-final"><div class="tbv-ix"><div class="tbv-ctl"><span class="l">Causal PFN</span><div class="tbv-seg" id="tbv-pfn"></div><span class="l">Benchmark</span><div class="tbv-seg" id="tbv-bench"></div></div><div class="tbv-meta" id="tbv-meta"></div><div class="tbv-rank" id="tbv-rank"></div><div class="tbv-key" id="tbv-key"></div><p class="tbv-read" id="tbv-read"></p></div><figcaption><span class="tbv-tag">Interactive</span><b class="tbv-title">The final benchmarks, run once: who has the lowest error?</b>Mean √PEHE over the benchmark's tasks (lower is better; hover a bar for the standard error). On the right: whether the wrapped PFN is significantly better or worse than that row (paired Wilcoxon test with Holm correction, and a bootstrap interval that excludes zero). The naive argmin picks whichever estimator has the smallest estimated risk, without a test.</figcaption></figure>
</div>

## Background

<div class="tbv">
<p class="tbv-lead">Estimating how a treatment changes an outcome from observational data usually means fitting and tuning several models per dataset. <b>Causal prior-data fitted networks (PFNs)</b> replace this with a transformer pretrained on large collections of simulated datasets: given a table of covariates, treatments and outcomes, it returns effect estimates in a single forward pass. Three have been released: CausalPFN, Do-PFN and CausalFM. Their guarantees are conditional, though. The model approximates a posterior under its training prior, and the effect is identified only if the data have no hidden confounders and enough overlap. The output does not say whether either holds.</p>
<div class="tbv-defs">
<div class="tbv-def"><b>CATE</b>The conditional average treatment effect τ(x): the mean effect of the treatment for units with covariates x. The ATE is its average over the population.</div>
<div class="tbv-def"><b>√PEHE</b>Root mean squared error of the estimated CATE on held-out units, against the true CATE that only simulated or semi-synthetic benchmarks provide. Lower is better.</div>
<div class="tbv-def"><b>Ignorability and overlap</b>The two conditions that identify the CATE from observed data: all confounders are measured, and every kind of unit has some chance of receiving either treatment.</div>
<div class="tbv-def"><b>DR-learner, BCF</b>The classical references. A doubly robust learner with cross-fitted, tuned gradient-boosting models, and Bayesian causal forests.</div>
<div class="tbv-def"><b>TabPFN-DR</b>A generic TabPFN, pretrained for prediction and not for causal inference, used for the regression and propensity models inside a cross-fitted DR-learner.</div>
<div class="tbv-def"><b>Pre-registration</b>Hypotheses, benchmarks, baselines and pass/fail criteria were committed before the corresponding runs. The final benchmarks and a held-out suite were run once, at the end.</div>
</div>
<p class="tbv-lead">We ask three questions. When, and why, do causal PFNs fail on data they were not trained on? Can such failures be detected from the observed data alone? And can detection be turned into a decision that does no harm, keeping the PFN when it is good and replacing it when it is not?</p>
</div>

## The diagnosis

<div class="tbv">
<p class="tbv-lead">The diagnosis uses a simulation platform with a base design (1,000 training units, 10 covariates of which 5 matter) and one-factor-at-a-time sweeps along 15 axes: 50 designs with 20 seeds each. Each hypothesis was registered with a refutation criterion before its runs.</p>
</div>

### 1. Accuracy depends on prior fit, and the most accurate estimator is not a causal PFN

<div class="tbv">
<p class="tbv-lead">Against BCF, the best classical method in 46 of the 50 designs, CausalPFN is on par in the base design and wins clearly only with strong effect heterogeneity or simple outcome surfaces. Do-PFN and CausalFM are 1.6–5× worse than BCF almost everywhere: they shrink the CATE toward a constant, and their error does not fall with more data. <b>TabPFN-DR</b> is significantly better than the best classical method in 37 designs and worse in 2. Explore the sweeps:</p>
<figure class="tbv-fig" id="tbv-plat"><div class="tbv-ix"><div class="tbv-ctl"><span class="l">Vary</span><div class="tbv-seg" id="tbv-axes"></div></div><div class="tbv-legend" id="tbv-legend"></div><div class="tbv-plot" id="tbv-plot"></div><p class="tbv-read" id="tbv-note"></p><details class="tbv-tbl"><summary>Show these values as a table</summary><div id="tbv-tbl"></div></details></div><div class="tbv-tip" id="tbv-tip"></div><figcaption><span class="tbv-tag">Interactive</span><b class="tbv-title">One design factor at a time: √PEHE of six estimators on the development platform.</b>Mean over 20 seeds; log scale, lower is better; the base design is marked. Colored lines are the three causal PFNs, gray lines the classical references, and the thick neutral line TabPFN-DR. Values are those of the paper's per-design accuracy table; the two axes that only rescale the outcome are left out.</figcaption></figure>
</div>

### 2. CausalPFN's uncertainty is aleatoric

<div class="tbv">
<p class="tbv-lead">CausalPFN's 90% CATE intervals cover the truth in 100% of cases in all 50 designs. That is not good calibration: in the base design they are 3.64 wide for an error of 0.214, and BCF covers 0.87 with intervals six times narrower. The reason is in how the model was trained. Its per-arm variance is, to first order, a fixed smoothing floor plus the outcome noise, so the interval describes how much an individual outcome varies, not how uncertain the effect estimate is.</p>
<figure class="tbv-fig narrow"><img src="/images/projects/2026-10-trust-but-verify/fig1-aleatoric-uncertainty.png" alt="Figure 1: CausalPFN's variance decomposition and interval coverage across 50 designs" loading="lazy"><figcaption><span class="tbv-tag">Figure 1</span><b class="tbv-title">The variance CausalPFN reports is mostly outcome noise.</b>(a) Its per-arm variance splits into a smoothing floor (gray) and a remainder (blue); the black tick is floor + σ², and the diamonds are the realized CATE error, one to two orders of magnitude below. (b) Coverage in the 50 designs: CausalPFN's CATE intervals sit at 1.00 while its ATE intervals range from 0 to about 0.95; BCF is closer to nominal on both.</figcaption></figure>
<p class="tbv-lead">The ATE interval fails the other way. It is formed from independent per-unit draws, so any error shared across units is averaged away: nominal 95% intervals cover 0.55 in the base design, 0.10 with 100 or more covariates, and 0.00 under hidden confounding. The package's temperature calibration narrows them further (0.35).</p>
</div>

### 3. Native signals do not flag violated prerequisites; the data partly can

<div class="tbv">
<p class="tbv-lead">None of the PFNs' own signals (interval width, variance, CausalPFN's calibration statistic) detects a hard overlap violation, and under hidden confounding the raw signals move the wrong way: the reported variance falls while the error grows 4.4×. The observed data say more. The share of units with an estimated propensity outside [0.05, 0.95] separates overlap violations almost perfectly (AUROC 0.95 and 0.99), and a cross-fitted doubly robust estimate of the PFN's excess risk over a reference is unbiased per design, though noisy per dataset. Hidden confounding stays undetectable by any statistic of the observed data, as it must.</p>
</div>

### 4. Two silent limits and a bias that does not vanish

<div class="tbv">
<figure class="tbv-fig wide"><div class="scroll"><img src="/images/projects/2026-10-trust-but-verify/fig3-silent-limits.png" alt="Figure 3: dimension cliff, confounding bias against sample size, and the context cap" loading="lazy"></div><figcaption><span class="tbv-tag">Figure 3</span><b class="tbv-title">Two limits of the released CausalPFN wrapper act without warning, and a confounding bias stays.</b>(a) Above 99 covariates the wrapper projects them with an unsupervised SVD: one extra irrelevant covariate raises the error 1.41×. (b) Under strong observed confounding, the ATE bias of CausalPFN stops shrinking after n = 1,000 and CausalFM's never shrinks, while the classical and TabPFN-based estimators keep improving. (c) The context is capped at 2,048 units per arm; with every unit in context, the loss to BCF at n = 8,000 disappears.</figcaption></figure>
<p class="tbv-lead">Both limits could be fixed in software, and the ranking above does not rest on them: on the 46 designs they leave untouched, TabPFN-DR is still better than the best classical method in 33 and better than CausalPFN in 37. The bias is a property of the models. CausalPFN keeps about 9% of the naive bias at n = 8,000 and CausalFM about 24% at every n; TabPFN-DR removes it.</p>
</div>

### 5. Evaluation choices decide the winner

<div class="tbv">
<p class="tbv-lead">Our harness reproduces the published CausalPFN errors on IHDP, ACIC-2016 (10 settings) and both LaLonde samples within one standard error, and CausalFM and Do-PFN on Jobs within 4%. Reproducing them also showed how much the ranking depends on evaluation choices: which ground truth an error is scored against, whether a benchmark provides noiseless or noisy effects, and how many of its settings are used. The benchmark alone can decide the winner: the CausalPFN / BCF error ratio is 0.22 on IHDP and 1.30 on a 10-setting subset of ACIC-2016.</p>
<p class="tbv-lead">The final evaluation therefore uses all 77 settings of ACIC-2016 (770 tasks), states for each benchmark what the error is measured against, and re-runs every baseline in one harness. The paper's appendix documents the benchmark audit in detail.</p>
</div>

## Certify-or-defer

<div class="tbv">
<p class="tbv-lead">The diagnosis leaves a user with a frozen PFN that may be excellent or badly wrong on their data. Certify-or-defer answers a question the user can act on: <i>is the PFN's CATE worse than a classical reference on this dataset, and if so, where?</i> It wraps any causal PFN, needs no retraining, and uses only the observed data.</p>
<div class="tbv-flow">
<div class="tbv-node"><b class="h">Observed data (X, T, Y), split into K = 5 folds</b>Every estimate below is out of fold: made for units the model did not see.</div>
<div class="tbv-arrow">↓</div>
<div class="lvl c3">
<div class="tbv-node"><b class="h">The frozen PFN</b>For each fold, pass the other folds as context and query the held-out units. K forward passes, no refitting: <i>in-context cross-fitting</i>.</div>
<div class="tbv-node"><b class="h">Classical references</b>DR-learner and BCF, fit on the other folds and predicting the held-out one.</div>
<div class="tbv-node"><b class="h">A doubly robust score φ</b>Propensity and outcome models give a pseudo-outcome whose mean, given x, is the true effect when those models are right.</div>
</div>
<div class="tbv-arrow">↓</div>
<div class="tbv-node key"><b class="h">Paired excess-risk statistic</b>For PFN estimate g and reference r on unit i:<span class="eq">δ<sub>i</sub> = (g − r)(X<sub>i</sub>) · [ (g + r)(X<sub>i</sub>) − 2φ<sub>i</sub> ]</span>Its mean estimates how much larger the PFN's CATE risk is than the reference's. The unknown true effect cancels, and the outcome noise enters only where the two estimators disagree, which keeps the variance small enough for a single dataset.</div>
<div class="tbv-arrow">↓</div>
<div class="lvl c2">
<div class="tbv-node"><b class="h">Region with good overlap</b>Units with estimated propensity in [0.1, 0.9]. One-sided test of "the PFN is not worse", against each reference.</div>
<div class="tbv-node"><b class="h">Region with poor overlap</b>The remaining units, tested separately so that their heavy-tailed scores do not drown the first region.</div>
</div>
<div class="tbv-arrow">↓ Holm correction over the four tests, α = 0.05</div>
<div class="lvl c2">
<div class="tbv-node keep"><b class="h">No rejection: keep the PFN (certificate)</b>The data show no excess risk of the PFN over the references in this region.</div>
<div class="tbv-node defer"><b class="h">Rejection: defer in that region</b>Use the reference that the test found significantly better there.</div>
</div>
</div>
<div class="tbv-box"><p><b>What a certificate does and does not say.</b> It says that a test finds no excess risk relative to the references. It is not evidence that the PFN is accurate: if the PFN and every reference are equally poor, the PFN is kept. Hidden confounding biases the score itself, so no certificate can rule it out. Under standard conditions (overlap, bounded outcomes, a product rate for the nuisance models) the deferral rule has asymptotic size α and power 1, and does no harm in expectation beyond α times the worst loss.</p></div>
</div>

## The pre-registered verdict

<div class="tbv">
<div class="tbv-time">
<div><b>Protocol frozen</b><span>Benchmarks, metrics, baselines, tests and the success criterion, before the method was implemented. A held-out suite of six mechanisms was written and frozen first.</span></div>
<div><b>Two pilots</b><span>On simulated development data only. The second pilot's detection criterion failed for CausalPFN; the method went to the final test unchanged.</span></div>
<div><b>Final run, once</b><span>IHDP, the full ACIC-2016 (770 tasks), two LaLonde samples and the held-out suite: 13,517 job records, all completed.</span></div>
<div><b>Post-verdict analyses</b><span>Planned in writing before they were computed, and unable to change any verdict. They locate where the test went wrong.</span></div>
<div><b>New confirmatory run</b><span>The variant those analyses point to, tested under a new frozen protocol on 160 fresh ACIC-2017 tasks.</span></div>
</div>
<div class="tbv-table tbv-verdict">
<p><span class="tbv-tag">Table 1</span> <b>Every pre-registered claim, as it came out.</b> H: held-out suite. A "failure event" is a task where the PFN's error exceeds the DR-learner's by at least 10%.</p>
<table>
<thead><tr><th>Claim and criterion</th><th>CausalPFN</th><th>CausalFM</th></tr></thead>
<tbody>
<tr><td><b>Detection.</b> The test statistic separates failure events with AUROC ≥ 0.80 on H and on ACIC-2016, and beats the PFN's best native signal.</td><td><span class="tbv-chip fail">✕ Fail</span><small>H: 0.944 (native 0.555). ACIC-2016: 0.783, below 0.80 (native 0.591).</small></td><td><span class="tbv-chip fail">✕ Fail</span><small>H: 0.859 (native 0.650). ACIC-2016: 0.788, below 0.80; margin over native not significant.</small></td></tr>
<tr><td><b>Do no harm.</b> Mean error no larger than the PFN's on H and ACIC-2016, significantly lower on one of them, and at most 10% of the replacements make a task worse.</td><td><span class="tbv-chip fail">✕ Fail</span><small>Error significantly lower on both (2.130 → 1.901, 1.424 → 1.220), but 18.9% of replacements were harmful (101 of 535).</small></td><td><span class="tbv-chip pass">✓ Pass</span><small>Error significantly lower on both (3.839 → 2.562, 4.227 → 1.196); 3.3% harmful (30 of 917).</small></td></tr>
<tr><td><b>Non-inferiority.</b> On each of the four benchmarks, the 95% upper bound of the error ratio to the PFN alone is at most 1.02.</td><td><span class="tbv-chip pass">✓ Pass</span><small>Upper bounds 1.000, 0.875, 1.004, 0.999.</small></td><td><span class="tbv-chip pass">✓ Pass</span><small>Upper bounds 0.381, 0.297, 0.743, 0.676.</small></td></tr>
<tr><td><b>Generality.</b> Detection and do-no-harm hold for both PFNs.</td><td colspan="2"><span class="tbv-chip fail">✕ Fail</span></td></tr>
</tbody>
</table>
</div>
<figure class="tbv-fig wide"><div class="scroll"><img src="/images/projects/2026-10-trust-but-verify/fig2-final-benchmarks.png" alt="Figure 2: error of the wrapper and the main baselines relative to the PFN alone, on five benchmarks and for two PFNs" loading="lazy"></div><figcaption><span class="tbv-tag">Figure 2</span><b class="tbv-title">Accuracy on the final benchmarks, relative to the PFN alone.</b>Mean √PEHE of the wrapper (A-V5 in the paper), the DR-learner, BCF, TabPFN-DR and the naive argmin, divided by that of the PFN each wraps (left CausalPFN, right CausalFM; log scale, below 1 is better). Percentages: share of tasks where the wrapper replaced the PFN in some region. B1 IHDP, B2 ACIC-2016, B3 / B4 LaLonde CPS / PSID, H the held-out suite. TabPFN-DR is not run on B3, whose size exceeds TabPFN's limit.</figcaption></figure>
<p class="tbv-lead">Three results outside the registered claims matter as much. <b>TabPFN-DR is more accurate than the wrapper</b> on every benchmark where it runs, for both PFNs, with one exception (IHDP with CausalPFN). <b>The naive argmin has a lower mean error than the wrapper</b> on the held-out suite and ACIC-2016 for CausalPFN: the test cuts the share of harmful switches from 19% to 6% on the held-out suite but not on ACIC-2016 (22% against 21%). And the wrapper <b>costs 8.8–27× the PFN's own run time</b>, most of it for the reference models.</p>
</div>

## Why it failed where it failed

<div class="tbv">
<p class="tbv-lead">The post-verdict analyses use the true CATE of the benchmarks to check each test decision. For CausalPFN on ACIC-2016, 65 of the 97 harmful replacements come from one pattern: the test rejects the PFN only in the poor-overlap region, in favor of the DR-learner. There the test is not conservative but badly oversized:</p>
<div class="tbv-stats">
<div class="tbv-stat" style="--tbv-c:var(--tbv-bad)"><div class="v">109 of 249</div><div class="d">true null hypotheses rejected in that region (the PFN was in fact no worse), where a valid test rejects about 5%</div></div>
<div class="tbv-stat"><div class="v">2 of 249</div><div class="d">rejections left once the bias of the score is removed; removing its noise instead leaves 98</div></div>
<div class="tbv-stat"><div class="v">+2.95 SE</div><div class="d">median bias of the statistic, mostly from clipping the propensity where the true one is below 0.01; it favors the DR-learner</div></div>
<div class="tbv-stat" style="--tbv-c:var(--tbv-good)"><div class="v">31 of 227</div><div class="d">rejected with an R-loss score in place of the doubly robust one (median bias +0.14 SE); pooled harm falls from 18.9% to 9.8%</div></div>
</div>
<p class="tbv-lead">So the harm is a bias, not bad luck: where overlap fails, clipping makes the bias of the doubly robust score first-order, and the score sides with the estimator built like itself. We therefore recommend the <b>R-loss variant</b>, which the development data alone would also have favored, and tested it under a new frozen protocol on data that no method had seen: ACIC-2017, 160 tasks.</p>
<div class="tbv-table" markdown="1">

<span class="tbv-tag">Table 2</span> **The confirmatory run on ACIC-2017.** Mean √PEHE over 160 tasks; in parentheses, harmful replacements out of the tasks with a replacement. Registered criterion for the R-loss variant: error-ratio bound ≤ 1.02 and at most 10% harmful, for both PFNs.

| Estimator | wrapping CausalPFN | wrapping CausalFM |
|---|---|---|
| PFN alone | 0.666 | 1.052 |
| **Certify-or-defer, R-loss score (the tested claim)** | **0.489 (1 of 129)** | **0.388 (0 of 122)** |
| Certify-or-defer, doubly robust score (as in the final run) | 0.427 (0 of 129) | 0.373 (0 of 121) |
| Naive argmin, R-loss | 0.341 (3 of 156) | 0.343 (5 of 142) |
| Always BCF | 0.341 | 0.341 |
| Always DR-learner | 0.561 | 0.561 |

<p class="tbv-how"><b>How to read it.</b> The R-loss variant passed both conditions for both PFNs (ratio bounds 0.768 and 0.401). But the doubly robust version did no harm on these data either and was more accurate, and simply using BCF was better still: both PFNs are weak on ACIC-2017, so deferring is easy. The run shows that the recommended variant does no harm on new data. It cannot show that it is better than the original.</p>

</div>
</div>

## Takeaways

<div class="tbv">
<ul class="tbv-take">
<li><b>Choosing an estimator for tabular CATE.</b> A generic in-context regressor inside an orthogonal estimator (TabPFN-DR) is a better default than a causal PFN, wrapped or not. Causal pretraining should be judged against it.</li>
<li><b>If you keep a causal PFN.</b> Certify-or-defer bounds the loss where the PFN is good (non-inferior on all four benchmarks) and repairs large errors where it is not, which the naive argmin does not guarantee. Use the R-loss score, and budget 8.8–27× the PFN's run time.</li>
<li><b>Reading a causal PFN's intervals.</b> CausalPFN's CATE interval is a predictive interval for an outcome, not a posterior for the effect, and its ATE interval is too narrow. Neither should be used as a confidence statement.</li>
<li><b>Building causal PFNs.</b> Report epistemic uncertainty apart from the spread of the outcome, form the ATE from joint draws, and warn when preprocessing changes the input, the dimension exceeds the pretraining range or overlap is weak.</li>
<li class="scope"><b>Limits.</b> No statistic of the observed data reveals hidden confounding. The guarantees are asymptotic and assume overlap, which a tenth of the units of ACIC-2016 violate. The registration is internal to our repository until its history is released. The R-loss variant was chosen after the verdicts, and its confirmatory run could not separate it from the original.</li>
</ul>
<details class="tbv-abs"><summary>Abstract</summary><p>Causal prior-data fitted networks (PFNs) estimate conditional average treatment effects in one forward pass, but nothing in their output tells a user whether the data fit the prior and satisfy the identifying assumptions that their guarantees rest on. We diagnose the three released causal PFNs on a 50-cell simulation platform and four benchmarks. A generic TabPFN inside a doubly robust learner (TabPFN-DR) is more accurate than all three on most designs, so causal pretraining should be judged against it. CausalPFN's uncertainty is aleatoric: its 90% CATE intervals cover 1.00 in every cell, while its 95% ATE intervals cover 0.55 in the base design. No native signal flags violated overlap, CausalPFN's wrapper has two silent limits, and a confounding bias in CausalPFN and CausalFM does not vanish with n. We also propose certify-or-defer, a wrapper that needs no retraining: in-context cross-fitting yields a paired doubly robust estimate of the PFN's excess CATE risk over classical references, and a region-wise one-sided Holm test replaces the PFN only where it is significantly worse. A certificate states only that this test finds no excess risk relative to the references, not that the PFN is accurate. Under a protocol frozen before the method was built, a confirmatory test on four benchmarks and six held-out mechanisms gives a mixed verdict. Certify-or-defer is non-inferior to the PFN on all four benchmarks and significantly more accurate on ACIC-2016 and the held-out suite, but its detection AUROC on ACIC-2016 misses the pre-registered 0.80 (0.783 and 0.788), and 18.9% of its deferrals for CausalPFN do harm, above the 10% bound. TabPFN-DR remains more accurate than the wrapper on most benchmarks; post-verdict analyses trace most of the harm to a bias of the doubly robust score where overlap fails. A run pre-registered on 160 new ACIC-2017 tasks confirms the R-loss variant we recommend, though there the registered test did no harm either.</p></details>
</div>

## Paper and code

Paper, code, the frozen protocol and the held-out suite will be linked here once the preprint is public.

{% raw %}
<script>
(function () {
  var D = {"final":{"causalpfn|b1":{"rows":[{"name":"CausalPFN alone","role":"pfn","v":0.5795,"se":0.0702,"k":100,"vs":"no difference"},{"name":"CausalPFN + certify-or-defer","role":"ours","v":0.579,"se":0.0703,"k":100},{"name":"naive argmin (no test)","role":"naive","v":0.5825,"se":0.0703,"k":100,"vs":"no difference"},{"name":"DR-learner","role":"ref","v":2.0062,"se":0.2698,"k":100,"vs":"better"},{"name":"BCF","role":"ref","v":2.6657,"se":0.4395,"k":100,"vs":"better"},{"name":"TabPFN-DR","role":"tabdr","v":0.9192,"se":0.1513,"k":100,"vs":"better"}],"defer":0.02,"noisy":false,"ni":0.999,"ni_upper":1.0},"causalfm|b1":{"rows":[{"name":"CausalFM alone","role":"pfn","v":5.8601,"se":0.7284,"k":100,"vs":"better"},{"name":"CausalFM + certify-or-defer","role":"ours","v":2.1046,"se":0.2699,"k":100},{"name":"naive argmin (no test)","role":"naive","v":1.9737,"se":0.2721,"k":100,"vs":"worse"},{"name":"DR-learner","role":"ref","v":2.0062,"se":0.2698,"k":100,"vs":"worse"},{"name":"BCF","role":"ref","v":2.6657,"se":0.4395,"k":100,"vs":"better"},{"name":"TabPFN-DR","role":"tabdr","v":0.9192,"se":0.1513,"k":100,"vs":"worse"}],"defer":1.0,"noisy":false,"ni":0.359,"ni_upper":0.381},"causalpfn|b2":{"rows":[{"name":"CausalPFN alone","role":"pfn","v":1.424,"se":0.0401,"k":770,"vs":"better"},{"name":"CausalPFN + certify-or-defer","role":"ours","v":1.2197,"se":0.0415,"k":770},{"name":"naive argmin (no test)","role":"naive","v":1.1736,"se":0.0411,"k":770,"vs":"worse"},{"name":"DR-learner","role":"ref","v":1.2584,"se":0.0412,"k":770,"vs":"no difference"},{"name":"BCF","role":"ref","v":1.2029,"se":0.0432,"k":770,"vs":"no difference"},{"name":"TabPFN-DR","role":"tabdr","v":0.9945,"se":0.0454,"k":770,"vs":"worse"}],"defer":0.601,"noisy":false,"ni":0.857,"ni_upper":0.875,"harm":0.21,"auroc":0.783,"native":0.591,"n_fail":440},"causalfm|b2":{"rows":[{"name":"CausalFM alone","role":"pfn","v":4.2265,"se":0.0676,"k":770,"vs":"better"},{"name":"CausalFM + certify-or-defer","role":"ours","v":1.1963,"se":0.0406,"k":770},{"name":"naive argmin (no test)","role":"naive","v":1.1908,"se":0.0409,"k":770,"vs":"no difference"},{"name":"DR-learner","role":"ref","v":1.2584,"se":0.0412,"k":770,"vs":"better"},{"name":"BCF","role":"ref","v":1.2029,"se":0.0432,"k":770,"vs":"no difference"},{"name":"TabPFN-DR","role":"tabdr","v":0.9945,"se":0.0454,"k":770,"vs":"worse"}],"defer":1.0,"noisy":false,"ni":0.283,"ni_upper":0.297,"harm":0.014,"auroc":0.788,"native":0.646,"n_fail":746},"causalpfn|b3":{"rows":[{"name":"CausalPFN alone","role":"pfn","v":8954.5131,"se":20.8561,"k":100,"vs":"no difference"},{"name":"CausalPFN + certify-or-defer","role":"ours","v":8975.6631,"se":21.5614,"k":100},{"name":"naive argmin (no test)","role":"naive","v":8997.9926,"se":22.7843,"k":100,"vs":"better"},{"name":"DR-learner","role":"ref","v":9173.1115,"se":25.5245,"k":100,"vs":"better"},{"name":"BCF","role":"ref","v":12765.4089,"se":67.0101,"k":100,"vs":"better"}],"defer":0.12,"noisy":true,"ni":1.002,"ni_upper":1.004},"causalfm|b3":{"rows":[{"name":"CausalFM alone","role":"pfn","v":12400.2607,"se":15.7996,"k":100,"vs":"better"},{"name":"CausalFM + certify-or-defer","role":"ours","v":9166.7183,"se":25.3445,"k":100},{"name":"naive argmin (no test)","role":"naive","v":9173.1115,"se":25.5245,"k":100,"vs":"better"},{"name":"DR-learner","role":"ref","v":9173.1115,"se":25.5245,"k":100,"vs":"better"},{"name":"BCF","role":"ref","v":12765.4089,"se":67.0101,"k":100,"vs":"better"}],"defer":1.0,"noisy":true,"ni":0.739,"ni_upper":0.743},"causalpfn|b4":{"rows":[{"name":"CausalPFN alone","role":"pfn","v":14382.5589,"se":197.1287,"k":100,"vs":"no difference"},{"name":"CausalPFN + certify-or-defer","role":"ours","v":14328.617,"se":196.1434,"k":100},{"name":"naive argmin (no test)","role":"naive","v":14314.6589,"se":199.8075,"k":100,"vs":"no difference"},{"name":"DR-learner","role":"ref","v":14902.4439,"se":278.9421,"k":100,"vs":"better"},{"name":"BCF","role":"ref","v":22553.229,"se":145.763,"k":100,"vs":"better"},{"name":"TabPFN-DR","role":"tabdr","v":13756.9282,"se":181.6939,"k":100,"vs":"worse"}],"defer":0.09,"noisy":true,"ni":0.996,"ni_upper":0.999},"causalfm|b4":{"rows":[{"name":"CausalFM alone","role":"pfn","v":22436.086,"se":130.3011,"k":100,"vs":"better"},{"name":"CausalFM + certify-or-defer","role":"ours","v":14809.9918,"se":250.1356,"k":100},{"name":"naive argmin (no test)","role":"naive","v":14899.7795,"se":254.9809,"k":100,"vs":"better"},{"name":"DR-learner","role":"ref","v":14902.4439,"se":278.9421,"k":100,"vs":"no difference"},{"name":"BCF","role":"ref","v":22553.229,"se":145.763,"k":100,"vs":"better"},{"name":"TabPFN-DR","role":"tabdr","v":13756.9282,"se":181.6939,"k":100,"vs":"worse"}],"defer":0.97,"noisy":true,"ni":0.66,"ni_upper":0.676},"causalpfn|h":{"rows":[{"name":"CausalPFN alone","role":"pfn","v":2.1297,"se":0.1585,"k":240,"vs":"better"},{"name":"CausalPFN + certify-or-defer","role":"ours","v":1.9009,"se":0.1424,"k":240},{"name":"naive argmin (no test)","role":"naive","v":1.6845,"se":0.1309,"k":240,"vs":"worse"},{"name":"DR-learner","role":"ref","v":3.1662,"se":0.2869,"k":240,"vs":"better"},{"name":"BCF","role":"ref","v":1.782,"se":0.1918,"k":240,"vs":"no difference"},{"name":"TabPFN-DR","role":"tabdr","v":1.3181,"se":0.0888,"k":240,"vs":"worse"}],"defer":0.3,"noisy":false,"harm":0.056,"auroc":0.944,"native":0.555,"n_fail":62},"causalfm|h":{"rows":[{"name":"CausalFM alone","role":"pfn","v":3.8389,"se":0.3569,"k":240,"vs":"better"},{"name":"CausalFM + certify-or-defer","role":"ours","v":2.5622,"se":0.2618,"k":240},{"name":"naive argmin (no test)","role":"naive","v":1.8781,"se":0.1972,"k":240,"vs":"worse"},{"name":"DR-learner","role":"ref","v":3.1662,"se":0.2869,"k":240,"vs":"better"},{"name":"BCF","role":"ref","v":1.782,"se":0.1918,"k":240,"vs":"worse"},{"name":"TabPFN-DR","role":"tabdr","v":1.3181,"se":0.0888,"k":240,"vs":"worse"}],"defer":0.613,"noisy":false,"harm":0.129,"auroc":0.859,"native":0.65,"n_fail":132}},"claims":{"generality":false,"causalpfn":{"A1":false,"A2":false,"NI":true,"harm":0.189,"n_harm":101,"n_def":535},"causalfm":{"A1":false,"A2":true,"NI":true,"harm":0.033,"n_harm":30,"n_def":917}},"f02":{"causalpfn":{"always_pfn":0.666,"bcf":0.341,"dr":0.561,"k":160,"pass":true,"V5-R":{"mean":0.489,"ni_upper":0.768,"n_harm":1,"n_def":129},"V5":{"mean":0.427,"ni_upper":0.673,"n_harm":0,"n_def":129},"naive-R":{"mean":0.341,"ni_upper":0.536,"n_harm":3,"n_def":156}},"causalfm":{"always_pfn":1.052,"bcf":0.341,"dr":0.561,"k":160,"pass":true,"V5-R":{"mean":0.388,"ni_upper":0.401,"n_harm":0,"n_def":122},"V5":{"mean":0.373,"ni_upper":0.382,"n_harm":0,"n_def":121},"naive-R":{"mean":0.343,"ni_upper":0.349,"n_harm":5,"n_def":142}},"records":992},"platform":[{"key":"d","label":"covariates d (5 active)","kind":"num","pts":[{"t":"5","base":false,"causalpfn":0.188,"causalfm":0.312,"dopfn":0.394,"dr_lgbm":0.28,"bcf":0.183,"tabpfn_dr":0.157},{"t":"10","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"25","base":false,"causalpfn":0.269,"causalfm":0.453,"dopfn":0.523,"dr_lgbm":0.318,"bcf":0.237,"tabpfn_dr":0.171},{"t":"50","base":false,"causalpfn":0.274,"causalfm":0.48,"dopfn":0.586,"dr_lgbm":0.327,"bcf":0.26,"tabpfn_dr":0.169},{"t":"100","base":false,"causalpfn":0.451,"causalfm":0.496,"dopfn":0.689,"dr_lgbm":0.347,"bcf":0.271,"tabpfn_dr":0.166},{"t":"150","base":false,"causalpfn":0.571,"causalfm":0.51,"dopfn":0.675,"dr_lgbm":0.373,"bcf":0.291,"tabpfn_dr":0.178}]},{"key":"n","label":"training units n","kind":"num","pts":[{"t":"250","base":false,"causalpfn":0.353,"causalfm":0.381,"dopfn":0.474,"dr_lgbm":0.588,"bcf":0.361,"tabpfn_dr":0.333},{"t":"500","base":false,"causalpfn":0.283,"causalfm":0.371,"dopfn":0.456,"dr_lgbm":0.424,"bcf":0.271,"tabpfn_dr":0.244},{"t":"1000","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"2000","base":false,"causalpfn":0.182,"causalfm":0.359,"dopfn":0.433,"dr_lgbm":0.226,"bcf":0.168,"tabpfn_dr":0.134},{"t":"4000","base":false,"causalpfn":0.15,"causalfm":0.353,"dopfn":0.431,"dr_lgbm":0.178,"bcf":0.137,"tabpfn_dr":0.095},{"t":"8000","base":false,"causalpfn":0.143,"causalfm":0.353,"dopfn":0.432,"dr_lgbm":0.163,"bcf":0.113,"tabpfn_dr":0.076}]},{"key":"kappa","label":"propensity scale κ (larger = weaker overlap)","kind":"num","pts":[{"t":"0.25","base":false,"causalpfn":0.177,"causalfm":0.342,"dopfn":0.44,"dr_lgbm":0.243,"bcf":0.179,"tabpfn_dr":0.112},{"t":"0.5","base":false,"causalpfn":0.19,"causalfm":0.35,"dopfn":0.439,"dr_lgbm":0.26,"bcf":0.188,"tabpfn_dr":0.126},{"t":"1","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"2","base":false,"causalpfn":0.273,"causalfm":0.417,"dopfn":0.469,"dr_lgbm":0.352,"bcf":0.226,"tabpfn_dr":0.225},{"t":"3","base":false,"causalpfn":0.322,"causalfm":0.459,"dopfn":0.482,"dr_lgbm":0.476,"bcf":0.24,"tabpfn_dr":0.294}]},{"key":"rho_c","label":"observed confounding ρc","kind":"num","pts":[{"t":"0","base":false,"causalpfn":0.205,"causalfm":0.371,"dopfn":0.464,"dr_lgbm":0.264,"bcf":0.199,"tabpfn_dr":0.15},{"t":"0.3","base":false,"causalpfn":0.204,"causalfm":0.363,"dopfn":0.456,"dr_lgbm":0.279,"bcf":0.197,"tabpfn_dr":0.15},{"t":"0.6","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"0.95","base":false,"causalpfn":0.239,"causalfm":0.406,"dopfn":0.441,"dr_lgbm":0.318,"bcf":0.21,"tabpfn_dr":0.169}]},{"key":"het","label":"effect heterogeneity (s.d. of τ)","kind":"num","pts":[{"t":"0","base":false,"causalpfn":0.167,"causalfm":0.135,"dopfn":0.136,"dr_lgbm":0.254,"bcf":0.078,"tabpfn_dr":0.108},{"t":"0.25","base":false,"causalpfn":0.192,"causalfm":0.221,"dopfn":0.253,"dr_lgbm":0.262,"bcf":0.139,"tabpfn_dr":0.15},{"t":"0.5","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"1","base":false,"causalpfn":0.245,"causalfm":0.698,"dopfn":0.854,"dr_lgbm":0.37,"bcf":0.284,"tabpfn_dr":0.181},{"t":"2","base":false,"causalpfn":0.27,"causalfm":1.425,"dopfn":1.695,"dr_lgbm":0.553,"bcf":0.401,"tabpfn_dr":0.197}]},{"key":"ate","label":"effect size (ATE)","kind":"num","pts":[{"t":"0","base":false,"causalpfn":0.214,"causalfm":0.396,"dopfn":0.444,"dr_lgbm":0.285,"bcf":0.2,"tabpfn_dr":0.163},{"t":"0.25","base":false,"causalpfn":0.214,"causalfm":0.38,"dopfn":0.438,"dr_lgbm":0.284,"bcf":0.199,"tabpfn_dr":0.163},{"t":"0.5","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"1","base":false,"causalpfn":0.214,"causalfm":0.365,"dopfn":0.48,"dr_lgbm":0.285,"bcf":0.197,"tabpfn_dr":0.163},{"t":"2","base":false,"causalpfn":0.214,"causalfm":0.435,"dopfn":0.601,"dr_lgbm":0.285,"bcf":0.194,"tabpfn_dr":0.163},{"t":"4","base":false,"causalpfn":0.214,"causalfm":0.75,"dopfn":0.997,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163}]},{"key":"noise_sd","label":"outcome noise s.d. σ","kind":"num","pts":[{"t":"0.1","base":false,"causalpfn":0.129,"causalfm":0.368,"dopfn":0.437,"dr_lgbm":0.206,"bcf":0.111,"tabpfn_dr":0.078},{"t":"0.5","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"1","base":false,"causalpfn":0.289,"causalfm":0.376,"dopfn":0.448,"dr_lgbm":0.445,"bcf":0.295,"tabpfn_dr":0.257},{"t":"2","base":false,"causalpfn":0.418,"causalfm":0.412,"dopfn":0.461,"dr_lgbm":0.772,"bcf":0.411,"tabpfn_dr":0.416}]},{"key":"frac_det","label":"share of units with deterministic treatment","kind":"num","pts":[{"t":"0","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"0.05","base":false,"causalpfn":0.217,"causalfm":0.363,"dopfn":0.446,"dr_lgbm":0.28,"bcf":0.202,"tabpfn_dr":0.171},{"t":"0.1","base":false,"causalpfn":0.226,"causalfm":0.366,"dopfn":0.446,"dr_lgbm":0.282,"bcf":0.208,"tabpfn_dr":0.187},{"t":"0.2","base":false,"causalpfn":0.246,"causalfm":0.37,"dopfn":0.45,"dr_lgbm":0.308,"bcf":0.229,"tabpfn_dr":0.274}]},{"key":"gamma_u","label":"hidden confounding strength γu","kind":"num","pts":[{"t":"0","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"0.5","base":false,"causalpfn":0.622,"causalfm":0.568,"dopfn":0.571,"dr_lgbm":0.692,"bcf":0.619,"tabpfn_dr":0.597},{"t":"1","base":false,"causalpfn":0.962,"causalfm":0.796,"dopfn":0.795,"dr_lgbm":0.989,"bcf":0.962,"tabpfn_dr":0.95},{"t":"2","base":false,"causalpfn":1.316,"causalfm":1.058,"dopfn":1.036,"dr_lgbm":1.337,"bcf":1.322,"tabpfn_dr":1.312}]},{"key":"outcome_fn","label":"outcome surface","kind":"cat","pts":[{"t":"additive","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"interact","base":false,"causalpfn":0.184,"causalfm":0.368,"dopfn":0.482,"dr_lgbm":0.333,"bcf":0.237,"tabpfn_dr":0.168},{"t":"linear","base":false,"causalpfn":0.12,"causalfm":0.363,"dopfn":0.456,"dr_lgbm":0.254,"bcf":0.206,"tabpfn_dr":0.129},{"t":"rff","base":false,"causalpfn":0.369,"causalfm":0.434,"dopfn":0.428,"dr_lgbm":0.489,"bcf":0.431,"tabpfn_dr":0.373}]},{"key":"effect_fn","label":"effect function","kind":"cat","pts":[{"t":"linear","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"additive","base":false,"causalpfn":0.283,"causalfm":0.418,"dopfn":0.463,"dr_lgbm":0.316,"bcf":0.26,"tabpfn_dr":0.235},{"t":"interact","base":false,"causalpfn":0.297,"causalfm":0.466,"dopfn":0.496,"dr_lgbm":0.418,"bcf":0.392,"tabpfn_dr":0.275},{"t":"rff","base":false,"causalpfn":0.396,"causalfm":0.471,"dopfn":0.488,"dr_lgbm":0.468,"bcf":0.425,"tabpfn_dr":0.389}]},{"key":"covariates","label":"covariate type","kind":"cat","pts":[{"t":"gauss","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"binary","base":false,"causalpfn":0.125,"causalfm":0.368,"dopfn":0.44,"dr_lgbm":0.174,"bcf":0.109,"tabpfn_dr":0.112},{"t":"corr","base":false,"causalpfn":0.203,"causalfm":0.364,"dopfn":0.411,"dr_lgbm":0.306,"bcf":0.212,"tabpfn_dr":0.162},{"t":"mixed","base":false,"causalpfn":0.183,"causalfm":0.397,"dopfn":0.434,"dr_lgbm":0.254,"bcf":0.174,"tabpfn_dr":0.144}]},{"key":"noise","label":"noise distribution","kind":"cat","pts":[{"t":"gauss","base":true,"causalpfn":0.214,"causalfm":0.368,"dopfn":0.445,"dr_lgbm":0.285,"bcf":0.198,"tabpfn_dr":0.163},{"t":"hetero","base":false,"causalpfn":0.205,"causalfm":0.367,"dopfn":0.451,"dr_lgbm":0.298,"bcf":0.196,"tabpfn_dr":0.172},{"t":"laplace","base":false,"causalpfn":0.211,"causalfm":0.362,"dopfn":0.445,"dr_lgbm":0.29,"bcf":0.205,"tabpfn_dr":0.158}]}],"bench":{"b1":"IHDP","b2":"ACIC-2016","b3":"LaLonde-CPS","b4":"LaLonde-PSID","h":"Held-out suite"},"n_records":13517};
  function $(id) { return document.getElementById(id); }
  function seg(el, items, cur, on) {
    el.innerHTML = "";
    items.forEach(function (it) { var b = document.createElement("button"); b.type = "button"; b.textContent = it[1];
      b.setAttribute("aria-pressed", it[0] === cur); b.onclick = function () { on(it[0]); }; el.appendChild(b); });
  }
  function fmt(v) { return v >= 1000 ? Math.round(v).toLocaleString("en-US") : v.toFixed(3); }
  function pct(v) { return Math.round(100 * v) + "%"; }

  /* ---- final benchmarks: ranked bars ---- */
  var PF = { causalpfn: "CausalPFN", causalfm: "CausalFM" }, HUE = { causalpfn: "var(--tbv-c1)", causalfm: "var(--tbv-c2)" };
  var BN = [["b1", "IHDP"], ["b2", "ACIC-2016"], ["b3", "LaLonde-CPS"], ["b4", "LaLonde-PSID"], ["h", "Held-out suite"]];
  var VS = { better: "▲ wrapper better", worse: "▼ wrapper worse", "no difference": "＝ no difference" };
  var pfn = "causalpfn", bench = "b2";
  function drawFinal() {
    seg($("tbv-pfn"), [["causalpfn", "CausalPFN"], ["causalfm", "CausalFM"]], pfn, function (k) { pfn = k; drawFinal(); });
    seg($("tbv-bench"), BN, bench, function (k) { bench = k; drawFinal(); });
    var B = D.final[pfn + "|" + bench], rows = B.rows.slice().sort(function (a, b) { return a.v - b.v; });
    var max = Math.max.apply(null, rows.map(function (r) { return r.v; }));
    var box = $("tbv-rank"), narrow = box.clientWidth < 520, ROW = narrow ? 56 : 36;
    box.style.height = rows.length * ROW + "px"; box.style.setProperty("--tbv-p", HUE[pfn]); $("tbv-key").style.setProperty("--tbv-p", HUE[pfn]);
    var keep = {}; box.querySelectorAll(".row").forEach(function (e) { if (rows.some(function (r) { return r.role + r.name.replace(/^Causal(PFN|FM)/, "") === e.dataset.n; })) keep[e.dataset.n] = e; else e.remove(); });
    rows.forEach(function (r, i) {
      var id = r.role + r.name.replace(/^Causal(PFN|FM)/, ""), e = keep[id];
      if (!e) { e = document.createElement("div"); e.dataset.n = id; e.tabIndex = 0;
        e.innerHTML = '<span class="nm"></span><span class="tr"><span class="fl"></span><span class="vl"></span></span><span class="vs"></span>'; box.appendChild(e); }
      e.className = "row r-" + r.role + (r.role === "ours" ? " ours" : "");
      e.querySelector(".nm").textContent = r.name; e.querySelector(".vs").textContent = r.vs ? VS[r.vs] : (r.role === "ours" ? "the wrapper" : "");
      var w = 100 * r.v / max; e.style.top = i * ROW + "px";
      e.querySelector(".fl").style.width = w + "%"; var v = e.querySelector(".vl"); v.style.left = w + "%"; v.textContent = fmt(r.v);
      e.title = r.name + ": " + fmt(r.v) + " ± " + fmt(r.se) + " (mean ± s.e. over " + r.k + " tasks)";
    });
    var meta = ["<span>" + rows[0].k + " tasks</span>", "<span>wrapper replaced the PFN on <b>" + pct(B.defer) + "</b> of tasks</span>"];
    if (B.harm !== undefined) meta.push("<span><b>" + (100 * B.harm).toFixed(1) + "%</b> of those replacements were harmful</span>");
    if (B.ni !== undefined) meta.push("<span>error ratio to the PFN alone <b>" + B.ni.toFixed(3) + "</b> (95% upper bound " + B.ni_upper.toFixed(3) + ")</span>");
    if (B.auroc !== undefined) meta.push("<span>detection AUROC <b>" + B.auroc.toFixed(3) + "</b> (best native signal " + B.native.toFixed(3) + ")</span>");
    $("tbv-meta").innerHTML = meta.join("");
    $("tbv-key").innerHTML = '<span><i class="k-pfn"></i>' + PF[pfn] + ' alone</span><span><i class="k-ours"></i>' + PF[pfn] + ' + certify-or-defer</span><span><i class="k-naive"></i>naive argmin</span><span><i class="k-ref"></i>classical references</span>' +
      (rows.some(function (r) { return r.role === "tabdr"; }) ? '<span><i class="k-tabdr"></i>TabPFN-DR</span>' : "");
    var p = B.rows.find(function (r) { return r.role === "pfn"; }), t = B.rows.find(function (r) { return r.role === "tabdr"; });
    var s = "Lowest mean error: <b>" + rows[0].name + "</b>. ";
    s += p.vs === "better" ? "The wrapper is significantly more accurate than " + PF[pfn] + " alone" : p.vs === "worse" ? "The wrapper is significantly less accurate than " + PF[pfn] + " alone" : "The wrapper and " + PF[pfn] + " alone do not differ significantly";
    if (t) s += t.vs === "worse" ? ", and TabPFN-DR is significantly more accurate than the wrapper." : t.vs === "better" ? ", and significantly more accurate than TabPFN-DR." : "; it does not differ significantly from TabPFN-DR.";
    else s += ". TabPFN-based methods are not run here: 14.5k training units exceed TabPFN v2's size limit.";
    if (B.noisy) s += " The error on this benchmark is measured against noisy individual effects, in dollars, and has a noise floor.";
    $("tbv-read").innerHTML = s;
  }

  /* ---- development platform: one-factor sweeps ---- */
  var M = [["causalpfn", "CausalPFN"], ["causalfm", "CausalFM"], ["dopfn", "Do-PFN"], ["bcf", "BCF"], ["dr_lgbm", "DR-learner"], ["tabpfn_dr", "TabPFN-DR"]];
  var SHORT = { d: "covariates d", n: "sample size n", kappa: "overlap κ", rho_c: "confounding ρc", het: "heterogeneity", ate: "effect size", noise_sd: "noise s.d.", frac_det: "hard overlap violation",
    gamma_u: "hidden confounding", outcome_fn: "outcome surface", effect_fn: "effect function", covariates: "covariate type", noise: "noise type" };
  var NOTE = {
    d: "CausalPFN's error jumps from 0.274 at d = 50 to 0.451 at d = 100, where its wrapper silently switches to an SVD projection (finding 4). TabPFN-DR stays flat, between 0.157 and 0.178.",
    n: "CausalFM and Do-PFN barely improve with more data (0.381 → 0.353 and 0.474 → 0.432 from n = 250 to 8,000). Every other estimator does.",
    kappa: "Weaker overlap raises every error. At κ = 3 BCF is the most accurate and the two DR-based learners lose their edge.",
    rho_c: "Observed confounding changes the CATE error little. Its effect shows in the ATE bias (finding 4).",
    het: "Without heterogeneity all three causal PFNs lose to BCF. With strong heterogeneity CausalFM and Do-PFN, which shrink the effect toward a constant, fall far behind, and CausalPFN beats BCF.",
    ate: "The error of Do-PFN and CausalFM grows once the effect is large (about twice their base error at ATE = 4). The other estimators are unaffected.",
    noise_sd: "With very noisy outcomes (σ = 2) the four best estimators are tied at 0.41–0.42 and the DR-learner falls behind.",
    frac_det: "Giving up to 20% of the units a deterministic treatment moves the errors little, and no native PFN signal reacts to it (finding 3). TabPFN-DR degrades most at 20%.",
    gamma_u: "Hidden confounding breaks identification: every estimator's error rises to 1.0–1.3 at γu = 2, and no statistic of the observed data can reveal it.",
    outcome_fn: "CausalPFN matches or edges out TabPFN-DR on linear and random-feature (rff) outcome surfaces, the simple and the very rough ends.",
    effect_fn: "TabPFN-DR has the lowest error for every effect function; the margin narrows for random-feature effects.",
    covariates: "With binary covariates BCF and TabPFN-DR are tied (0.109 and 0.112). Otherwise TabPFN-DR leads.",
    noise: "The noise distribution matters little."
  };
  var axis = "d", tip = $("tbv-tip");
  function sw(m) { return '<svg width="26" height="10" aria-hidden="true"><line x1="1" y1="5" x2="25" y2="5" class="tbv-s m-' + m + '"/><circle cx="13" cy="5" r="3.5" class="tbv-d m-' + m + '" stroke-width="1.5"/></svg>'; }
  function drawPlat() {
    seg($("tbv-axes"), D.platform.map(function (a) { return [a.key, SHORT[a.key]]; }), axis, function (k) { axis = k; drawPlat(); });
    $("tbv-legend").innerHTML = M.map(function (m) { return "<span>" + sw(m[0]) + m[1] + "</span>"; }).join("");
    var A = D.platform.find(function (a) { return a.key === axis; }), P = A.pts, host = $("tbv-plot");
    var W = Math.max(300, host.clientWidth), H = 318, L = 44, R = 12, T = 12, Bm = 64, pw = W - L - R, ph = H - T - Bm;
    var all = []; P.forEach(function (p) { M.forEach(function (m) { all.push(p[m[0]]); }); });
    var lo = Math.log(Math.min.apply(null, all) * 0.86), hi = Math.log(Math.max.apply(null, all) * 1.14);
    function y(v) { return T + ph * (1 - (Math.log(v) - lo) / (hi - lo)); }
    function x(i) { return L + pw * (i + 0.5) / P.length; }
    var ticks = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2].filter(function (t) { return Math.log(t) > lo && Math.log(t) < hi; });
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="√PEHE of six estimators as ' + A.label + ' varies">';
    ticks.forEach(function (t) { s += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(t) + '" y2="' + y(t) + '"/><text class="ax" x="' + (L - 7) + '" y="' + (y(t) + 4) + '" text-anchor="end">' + t + "</text>"; });
    P.forEach(function (p, i) { s += '<text class="ax" x="' + x(i) + '" y="' + (H - Bm + 18) + '" text-anchor="middle"' + (p.base ? ' font-weight="700"' : "") + ">" + p.t + "</text>" + (p.base ? '<text class="ax" x="' + x(i) + '" y="' + (H - Bm + 32) + '" text-anchor="middle" font-size="10.5">base</text>' : ""); });
    s += '<text class="ax" x="' + (L + pw / 2) + '" y="' + (H - 6) + '" text-anchor="middle">' + A.label + "</text>";
    s += '<line class="cross" id="tbv-cross" y1="' + T + '" y2="' + (T + ph) + '"/>';
    M.forEach(function (m) {
      if (A.kind === "num") s += '<path class="tbv-s m-' + m[0] + '" d="' + P.map(function (p, i) { return (i ? "L" : "M") + x(i).toFixed(1) + " " + y(p[m[0]]).toFixed(1); }).join("") + '"/>';
    });
    M.forEach(function (m) { P.forEach(function (p, i) { s += '<circle class="tbv-ring" cx="' + x(i) + '" cy="' + y(p[m[0]]) + '" r="5.5"/><circle class="tbv-d m-' + m[0] + '" cx="' + x(i) + '" cy="' + y(p[m[0]]) + '" r="4" stroke-width="1.5"/>'; }); });
    P.forEach(function (p, i) { s += '<rect data-i="' + i + '" x="' + (L + pw * i / P.length) + '" y="' + T + '" width="' + pw / P.length + '" height="' + (ph + 30) + '" fill="transparent" tabindex="0" aria-label="' + A.label + " = " + p.t + '"/>'; });
    host.innerHTML = s + "</svg>";
    var cross = document.getElementById("tbv-cross"), fig = $("tbv-plat");
    function show(i) {
      var p = P[i], rows = M.map(function (m) { return [m, p[m[0]]]; }).sort(function (a, b) { return a[1] - b[1]; });
      tip.innerHTML = '<b class="h">' + SHORT[axis] + ": " + p.t + (p.base ? " (base)" : "") + "</b>" + rows.map(function (r) { return '<div class="r"><span>' + sw(r[0][0]) + r[0][1] + "</span><b>" + r[1].toFixed(3) + "</b></div>"; }).join("");
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.style.opacity = 1;
      var hr = host.getBoundingClientRect(), fr = fig.getBoundingClientRect(), px = hr.left - fr.left + x(i) * hr.width / W;
      var left = px + 14; if (left + 190 > fr.width) left = px - 14 - tip.offsetWidth;
      tip.style.left = Math.max(4, left) + "px"; tip.style.top = (hr.top - fr.top + 8) + "px"; tip.style.opacity = 1;
    }
    function hide() { tip.style.opacity = 0; cross.style.opacity = 0; }
    host.querySelectorAll("rect[data-i]").forEach(function (r) {
      var i = +r.dataset.i; r.addEventListener("pointerenter", function () { show(i); }); r.addEventListener("pointerdown", function () { show(i); });
      r.addEventListener("focus", function () { show(i); }); r.addEventListener("blur", hide);
    });
    host.onpointerleave = hide;
    $("tbv-note").innerHTML = NOTE[axis];
    $("tbv-tbl").innerHTML = "<table><thead><tr><th>" + SHORT[axis] + "</th>" + M.map(function (m) { return "<th>" + m[1] + "</th>"; }).join("") + "</tr></thead><tbody>" +
      P.map(function (p) { return "<tr><td>" + p.t + (p.base ? " (base)" : "") + "</td>" + M.map(function (m) { return "<td>" + p[m[0]].toFixed(3) + "</td>"; }).join("") + "</tr>"; }).join("") + "</tbody></table>";
  }

  drawFinal(); drawPlat();
  var t; window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(function () { drawFinal(); drawPlat(); }, 120); });
})();
</script>
{% endraw %}
