---
title: "Is That Supernode Special? Selection Rules and Null Models Change What Attribution-Graph Interventions Validate"
collection: projects
permalink: /projects/2026-11-supernode-specificity/
date: 2026-11-22
# No venue under review is named on this page; paper, code and BibTeX links to be added when they are public.
status: "Under review"
venue: "Under review"
excerpt: "A matched-null test for attribution-graph intervention claims: 13 of 14 hand-curated claims pass, but for automatically selected supernodes the verdict depends on the selection rule, on the null model and on which features the null may draw."
teaser: "/images/projects/2026-11-supernode-specificity/teaser.png"
tags: ["mechanistic interpretability", "attribution graphs", "circuit tracing", "null models"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-supernode-specificity"
published: false
---

<style>
/* Supernode-specificity project page: styles scoped to .sn / .sn-* so the rest of the site is untouched; colors follow the theme.
   --sn-acc marks the claim under test; --sn-b / --sn-c are rule B / rule C in the shares chart; --sn-null is a null distribution. */
.sn{--sn-acc:#5b4bc4;--sn-acc-soft:rgba(91,75,196,.08);--sn-gray:#6f767e;--sn-gray-soft:rgba(127,127,127,.09);--sn-b:#2a78d6;--sn-c:#eb6834;--sn-null:#8a9199;--sn-line:rgba(127,127,127,.28);--sn-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .sn{--sn-acc:#b9b1f5;--sn-acc-soft:rgba(185,177,245,.14);--sn-gray:#c3c9cf;--sn-gray-soft:rgba(255,255,255,.07);--sn-b:#5c9fee;--sn-c:#ee7a44;--sn-null:#c3c9cf;--sn-line:rgba(255,255,255,.2);--sn-card:rgba(255,255,255,.04)}
.sn-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.sn-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--sn-acc-soft);color:var(--sn-acc)}
.sn-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.sn-tldr b{color:var(--sn-acc)}
.sn-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.sn-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.sn-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.sn-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--sn-line)}
.sn-btn.muted{background:transparent;color:var(--sn-gray)!important;border:1px dashed var(--sn-line);cursor:default}
.sn-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .sn-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.sn-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--sn-line);border-radius:14px;background:var(--sn-card)}
.sn-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.sn-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.sn-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--sn-acc);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .sn-tag{color:#1d1a33}
.sn-fig figcaption b.sn-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.sn-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.sn-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--sn-line);background:var(--sn-card);position:relative;overflow:hidden}
.sn-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--sn-acc)}
.sn-stat .v{font-size:1.14em;white-space:nowrap;font-weight:800;line-height:1.15;font-variant-numeric:tabular-nums}
.sn-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.sn-lead{font-size:1em;line-height:1.65;margin:0 0 1em}
.sn-box{padding:1em 1.2em;border-radius:14px;background:var(--sn-acc-soft);border:1px solid var(--sn-line);margin:0 0 1.6em;line-height:1.65}
.sn-box p{margin:0}
.sn-q{margin:0 0 1.2em;padding:.7em 1em;border-left:3px solid var(--sn-acc);border-radius:4px;background:var(--sn-acc-soft);font-size:1.02em;line-height:1.55;font-weight:500}
.sn-defs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em;margin:0 0 1.6em}
.sn-def{padding:.65em .85em;border-radius:12px;border:1px solid var(--sn-line);background:var(--sn-card);font-size:.86em;line-height:1.5}
.sn-def>b:first-child{display:inline-block;margin-right:.35em;color:var(--sn-acc)}
.sn-def.wide{grid-column:1/-1}
.sn-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1em}
.sn-step{padding:.9em .95em;border-radius:14px;border:1px solid var(--sn-acc);background:var(--sn-acc-soft);font-size:.86em;line-height:1.55}
.sn-step .n{display:inline-grid;place-items:center;width:1.7em;height:1.7em;border-radius:50%;background:var(--sn-acc);color:#fff;font-weight:800;font-size:.9em;margin-right:.45em}
html[data-theme="dark"] .sn-step .n{color:#1d1a33}
.sn-step .h{font-size:1.05em}
.sn-step p{margin:.5em 0 0}
.sn-why{font-size:.9em;margin:0 0 1.6em;padding:.6em .9em;border-radius:10px;background:var(--sn-gray-soft);line-height:1.55}
.sn-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--sn-line);border-radius:14px;background:var(--sn-card);overflow-x:auto}
.sn-table table{font-size:.8em;margin:.6em 0;min-width:600px}
.sn-table p{font-size:.86em;margin:.2em}
.sn-how{font-size:.86em;line-height:1.55;margin:.5em .2em 0;padding:.5em .75em;border-radius:8px;background:var(--sn-gray-soft)}
.sn-take{margin:0 0 1.2em;padding:0;list-style:none;display:grid;gap:.55em;counter-reset:sn}
.sn-take li{padding:.65em .9em .65em 1em;border-left:3px solid var(--sn-acc);border-radius:4px;background:var(--sn-card);font-size:.92em;line-height:1.55}
.sn-take li.scope{border-left-color:var(--sn-gray)}
.sn-code{margin:0 0 1.6em;padding:.8em 1em;border-radius:12px;border:1px solid var(--sn-line);background:var(--sn-gray-soft);font-size:.76em;line-height:1.6;overflow-x:auto;white-space:pre;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--global-text-color)}
.sn-abs{margin:0 0 1.6em;border:1px solid var(--sn-line);border-radius:12px;background:var(--sn-card)}
.sn-abs summary{cursor:pointer;padding:.6em .9em;font-weight:700;font-size:.92em}
.sn-abs p{margin:0;padding:0 .95em .9em;font-size:.9em;line-height:1.65}
/* interactive figures */
.sn-ix{position:relative;font-size:.9em}
.sn-ctl{display:grid;grid-template-columns:auto 1fr;gap:.4em .9em;align-items:center;margin:0 0 .8em}
.sn-ctl .l{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65}
.sn-seg{display:flex;flex-wrap:wrap;gap:.35em}
.sn-seg button{font:inherit;font-size:.86em;line-height:1.3;padding:.3em .85em;border-radius:999px;border:1px solid var(--sn-line);background:transparent;color:var(--global-text-color);cursor:pointer;transition:background .15s,border-color .15s}
.sn-seg button:hover:not(:disabled){border-color:var(--sn-gray)}
.sn-seg button[aria-pressed=true]{background:var(--global-text-color);border-color:var(--global-text-color);color:var(--global-bg-color)}
.sn-seg button:disabled{opacity:.4;cursor:not-allowed}
.sn-seg.claims button{padding:.3em .6em;min-width:2.9em;font-variant-numeric:tabular-nums}
.sn-claim{margin:0 0 .5em;font-size:.95em;line-height:1.5}
.sn-claim .m{opacity:.7}
.sn-plot{display:block;width:100%}
.sn-plot svg{display:block;width:100%;height:auto;overflow:visible}
.sn-plot text{font-family:inherit;fill:var(--global-text-color)}
.sn-plot .ax{font-size:11px;opacity:.75}
.sn-plot .lab{font-size:12px;font-weight:700}
.sn-plot .grid{stroke:var(--sn-line);stroke-width:1}
.sn-plot .base{stroke:var(--global-text-color);stroke-width:1;opacity:.55}
.sn-plot .bar{fill:var(--sn-null)}
.sn-plot .hit{fill:transparent}
.sn-plot .hit:hover{fill:var(--sn-gray-soft)}
.sn-plot .p95{stroke:var(--global-text-color);stroke-width:1;opacity:.6}
.sn-plot .obs{stroke:var(--sn-acc);stroke-width:2.5;stroke-linecap:round}
.sn-plot .obsdot{fill:var(--sn-acc);stroke:var(--global-bg-color);stroke-width:2}
.sn-plot .rug{stroke:var(--global-text-color);stroke-width:1;opacity:.75}
.sn-read{margin:.8em 0 0;padding:.5em .8em;border-left:3px solid var(--sn-gray);border-radius:4px;background:var(--sn-gray-soft);font-size:.95em;line-height:1.55}
.sn-read.acc{border-left-color:var(--sn-acc);background:var(--sn-acc-soft)}
.sn-tip{position:absolute;z-index:5;pointer-events:none;max-width:260px;padding:.35em .6em;border-radius:6px;background:#22262a;color:#fff;font-size:.82em;line-height:1.4;box-shadow:0 4px 14px rgba(0,0,0,.25);opacity:0;transition:opacity .1s}
.sn-tip.on{opacity:1}
.sn-leg{display:flex;flex-wrap:wrap;gap:.3em 1.1em;margin:0 0 .6em;font-size:.86em}
.sn-leg i{display:inline-block;width:.8em;height:.8em;border-radius:3px;margin-right:.4em;vertical-align:-.05em}
.sn-leg .w{display:inline-block;width:1.2em;height:0;border-top:2px solid var(--global-text-color);opacity:.7;margin-right:.4em;vertical-align:.3em}
.sn-sh .grp{font-size:.72em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65;margin:.9em 0 .35em}
.sn-sh .row{display:grid;grid-template-columns:150px 1fr;gap:0 .8em;align-items:center;margin:0 0 .55em}
.sn-sh .nm{font-size:.9em;line-height:1.25}
.sn-sh .nm small{display:block;opacity:.65;font-size:.82em}
.sn-sh .bars{display:grid;gap:2px}
.sn-sh .bar{position:relative;height:14px;margin-right:74px;background:var(--sn-gray-soft);border-radius:0 4px 4px 0;cursor:default;outline-offset:2px}
.sn-sh .fl{position:absolute;left:0;top:0;bottom:0;border-radius:0 4px 4px 0;transition:width .5s cubic-bezier(.2,.8,.2,1)}
.sn-sh .bar.b .fl{background:var(--sn-b)}.sn-sh .bar.c .fl{background:var(--sn-c)}
.sn-sh .ci{position:absolute;top:50%;height:0;border-top:2px solid var(--global-text-color);opacity:.7;transition:left .5s cubic-bezier(.2,.8,.2,1),width .5s cubic-bezier(.2,.8,.2,1)}
.sn-sh .vl{position:absolute;top:-2px;padding-left:7px;font-size:.84em;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;transition:left .5s cubic-bezier(.2,.8,.2,1)}
.sn-sh .vl small{font-weight:400;opacity:.7;margin-right:.35em}
.sn-axis{display:flex;justify-content:space-between;font-size:.72em;opacity:.7;margin:.2em 74px 0 calc(150px + .8em)}
@media (max-width:900px){.sn-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:760px){.sn-steps{grid-template-columns:1fr}.sn-defs{grid-template-columns:1fr}}
@media (max-width:640px){.sn-ctl{grid-template-columns:1fr}.sn-tldr{font-size:1.04em}.sn-sh .row{grid-template-columns:96px 1fr}.sn-axis{margin-left:calc(96px + .8em)}.sn-axis span:nth-child(2){display:none}.sn-sh .nm{font-size:.82em}}
/* the theme's fixed table of contents (220px, right) covers a full-width title: keep the header in the text column */
@media screen and (min-width:1200px){.page__inner-wrap>header{max-width:calc(100% - 240px)}}
</style>

<div class="sn">
<div class="sn-kicker"><span>Mechanistic interpretability</span><span>Attribution graphs</span><span>Statistical testing</span></div>
<p class="sn-tldr">Circuit-tracing studies back their claims with interventions: suppress a labelled group of features, a <i>supernode</i>, and the answer changes. That shows these features matter, not that they are special. We test each claim against random features that the attribution graph ranks as equally important. <b>13 of 14 hand-curated claims pass</b>; for automatically selected supernodes, the verdict depends on the selection rule, on the null model and on which features the null may draw.</p>
<div class="sn-btns"><a class="sn-btn" href="#how-the-test-works">The test</a><a class="sn-btn ghost" href="#results">Results</a><a class="sn-btn ghost" href="#what-to-report">What to report</a><span class="sn-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="sn-fig" id="sn-try"><div class="sn-ix"><div class="sn-ctl"><span class="l">Claim</span><div class="sn-seg claims" id="sn-claims"></div><span class="l">Null</span><div class="sn-seg" id="sn-nulls"><button type="button" data-v="act">activation-matched</button><button type="button" data-v="infl">influence-matched (primary)</button><button type="button" data-v="inject">inject-only</button></div></div><p class="sn-claim" id="sn-claim"></p><div class="sn-plot" id="sn-plot" role="img"></div><p class="sn-read acc" id="sn-read"></p><div class="sn-tip" id="sn-tip1"></div></div><figcaption><span class="sn-tag">Interactive</span><b class="sn-title">Is this supernode special? One published claim against its null.</b>Pick one of the 14 hand-curated claims (D, M: Gemma-2-2B, from the circuit-tracer tutorial; L: Llama-3.2-1B, from its demo). Gray bars: the claimed metric when every labelled feature is replaced by a matched random feature and the same intervention is applied (5,000 draws for the influence-matched and inject-only nulls, 500 for the activation-matched one). The marked line is the published supernode; short ticks on the axis are random draws that reach it. D3 is the “Texas” example of Figure 1, D4 the one claim that fails, and M1 passes although some random draws go further.</figcaption></figure>
</div>

## Background

<div class="sn">
<p class="sn-lead">An <b>attribution graph</b> explains one model output. The model's MLPs are replaced by sparse <i>transcoder</i> features, the output is attributed to those features, and the analyst groups the important ones into labelled supernodes such as “Texas”. The account is then checked by intervention. On the prompt <i>Fact: the capital of the state containing Dallas is</i>, suppressing the “Texas” supernode removes “Austin” from the output, and also injecting “California” features makes the model say “Sacramento”.</p>
<p class="sn-lead">These interventions carry the evidence, but they answer a weaker question than the one they are used for. The last city token may host many active features that contribute to the answer, and suppressing ten of them may remove the answer whichever ten are chosen. Supernode interventions are usually reported without a null model, even as automated pipelines begin to build supernodes at scale.</p>
<p class="sn-q">The question this work makes testable: would any equally important features at “Dallas” do the same?</p>
<div class="sn-defs">
<div class="sn-def"><b>Transcoder</b>Sparse features that replace a model's MLPs so that an output can be attributed to them: one set per layer (per-layer transcoder, <b>PLT</b>) or a single set that writes to all later layers (cross-layer transcoder, <b>CLT</b>).</div>
<div class="sn-def"><b>Supernode</b>A group of transcoder features under one label, chosen by hand or by a rule. Each member is a (layer, position, feature) triple.</div>
<div class="sn-def"><b>Ablation claim</b>Suppress the supernode at −2 times its activation. The metric is the drop in <i>p</i>(answer).</div>
<div class="sn-def"><b>Swap claim</b>Also inject a donor prompt's supernode at +2 at the same position. The metric is <i>p</i>(donor's answer).</div>
<div class="sn-def"><b>Activation-matched null</b>Replace each labelled feature by one of the 20 active features in the same layer and position with the closest activation. This is the familiar random-feature baseline.</div>
<div class="sn-def"><b>Influence-matched null</b>Replace it by one of the 20 features at the same position, in any layer, with the closest attribution-influence rank: features the graph itself treats as equally important. Our primary null.</div>
<div class="sn-def wide"><b>Specific, weak</b>A claim is <i>specific</i> if its effect beats the influence-matched null after multiple-comparison correction (Benjamini–Hochberg at <i>q</i> = 0.05 within a family of claims; Holm at α = 0.01 for the 14 hand-curated ones). A specific claim is <i>weak</i> if its effect is below twice the null's 95th percentile.</div>
</div>
</div>

## How the test works

<div class="sn">
<div class="sn-steps">
<div class="sn-step"><span class="n">1</span><b class="h">Fix the claim</b><p>A prompt, the feature groups with their scales, and one metric chosen before testing. The metric has to be the one the claim is about: suppressing French features looks non-specific under a generic drop in top-1 probability (<i>p</i> = 0.08) and specific under the claimed <i>p</i>(“big”) (<i>p</i> = 0.002), on the same 5,000 draws.</p></div>
<div class="sn-step"><span class="n">2</span><b class="h">Draw matched replacements</b><p>Replace every labelled feature by a random feature at the same position, taken from its 20 best matches. Apply the same intervention, record the metric, and repeat <i>n</i> times: 500 for automatic claims, 5,000 for hand-curated ones. Labelled features are kept out of the pools.</p></div>
<div class="sn-step"><span class="n">3</span><b class="h">Compare and correct</b><p><i>p</i> = (1 + #{null ≥ observed}) / (<i>n</i> + 1), corrected within each family of claims. The effect-to-null ratio is reported as well: a pass by less than twice the null's 95th percentile is marked weak.</p></div>
</div>
<p class="sn-why">Every intervention uses circuit-tracer's <code>feature_intervention</code> unchanged, with attention frozen. Swap claims get a third null, <i>inject-only</i>, which keeps the real suppression and randomises only the injected features.</p>
<figure class="sn-fig"><img src="/images/projects/2026-11-supernode-specificity/fig1-specificity-test.png" alt="Figure 1: testing an intervention claim against matched nulls" loading="lazy"><figcaption><span class="sn-tag">Figure 1</span><b class="sn-title">Testing an intervention claim against matched nulls.</b>The paper's overview. (1) A claim: the “Texas” supernode at the last city token, suppressed at −2, with the drop in <i>p</i>(Austin) as the metric. (2) Each feature of the supernode is replaced by a random feature at the same position, matched on activation (N<sub>act</sub>) or on attribution-influence rank (N<sub>infl</sub>). (3) The observed effect is compared with the null distribution. Inset: shares of specific Gemma per-layer two-hop ablation claims.</figcaption></figure>
<p class="sn-lead"><b>Where the automatic supernodes come from.</b> Besides the 14 hand-curated claims, we test 2,958 claims whose supernodes are picked by a rule at the position where the intermediate variable should be represented: the city token in two-hop recall, the subject noun in agreement.</p>
<div class="sn-defs">
<div class="sn-def"><b>Rule B, contrastive</b>The ten features whose activation on the prompt most exceeds their activation on a donor prompt: a city from another region, or the same sentence with the subject's number flipped.</div>
<div class="sn-def"><b>Rule C, held-out causal</b>On a <i>sibling</i> prompt that shares the intermediate variable (the other city in the same state or country), ablate each active feature alone, keep the ten with the largest effect, and retain those active on the test prompt. The test prompt's output is never used.</div>
<div class="sn-def"><b>Rule C′</b>Rule C held out in the target as well: features are scored on the sibling's one-hop prompt, “Fact: {sibling} is a city in the state of”, by the drop in <i>p</i>(region).</div>
<div class="sn-def"><b>Tasks and models</b>Two-hop recall (“Fact: the capital of the {state|country} containing {city} is”, 200 prompts) and subject–verb agreement (“The {noun} that I saw yesterday”, 200 prompts), on Gemma-2-2B and Llama-3.2-1B with public per-layer and cross-layer transcoders. Only correctly answered prompts are kept.</div>
</div>
<p class="sn-why">The rules are simple heuristics that we constructed. Published annotation pipelines were not tested.</p>
</div>

## Results

<div class="sn"><div class="sn-stats">
<div class="sn-stat"><div class="v">13 / 14</div><div class="d">hand-curated claims from the circuit-tracer tutorial and demo pass the influence-matched null (5,000 draws, Holm at α = 0.01)</div></div>
<div class="sn-stat"><div class="v">89–100% → 10–82%</div><div class="d">contrastive per-layer two-hop claims judged specific under the activation-matched null, and under the influence-matched one</div></div>
<div class="sn-stat"><div class="v">10–31 points</div><div class="d">lower share of specific contrastive claims when the null may also draw features that other rules select (0–56% remain)</div></div>
<div class="sn-stat"><div class="v">55–97%</div><div class="d">of the two-hop ablation claims that pass beat their null's 95th percentile by less than a factor of two</div></div>
</div></div>

### Hand-curated claims pass

<div class="sn">
<p class="sn-lead">Thirteen of the fourteen claims pass, and every swap claim also passes its inject-only null (<i>p</i> ≤ 0.0004), so it matters which donor features are injected. The one failure, D4, is a claim that the tutorial itself describes as largely ineffective. Its effect is 1.3 times its null's 95th percentile, and in bf16 arithmetic its verdict changes with the GPU type. The tutorial's two other unsuccessful interventions fail as negative controls (<i>p</i> = 0.91 and 0.32).</p>
<p class="sn-lead">Passing does not mean that only the labelled features can produce the effect. For M1, some random suppressions exceed the observed effect (largest 0.67 against 0.49). The curated examples may also have been chosen because they worked.</p>
</div>

<div class="sn"><div class="sn-table" markdown="1">

<span class="sn-tag">Table 1</span> **The 14 hand-curated claims under the influence-matched null** (5,000 draws each). The figure at the top of this page shows the draws behind each row.

| Claim | Intervention → claimed metric | Observed | Null 95th pct. | *p* | Holm, α = 0.01 |
|---|---|---|---|---|---|
| D1 | “say a capital” ×(−2) → *p*(Texas) | 0.460 | 0.185 | 0.0002 | pass |
| D2 | “capital” ×(−2) → drop of *p*(Austin) | 0.407 | 0.106 | 0.0002 | pass |
| D3 | “Texas” ×(−2) → drop of *p*(Austin) | 0.423 | 0.083 | 0.0002 | pass |
| D4 | “state” ×(−2) → drop of *p*(Austin) | 0.051 | 0.039 | 0.016 | **fail** |
| D5 | Texas ×(−2) + California ×2 → *p*(Sacramento) | 0.364 | 0.005 | 0.0002 | pass |
| D6 | Texas ×(−2) + China ×2 → *p*(Beijing) | 0.212 | &lt;0.001 | 0.0002 | pass |
| M1 | French ×(−2) → *p*(“big”) | 0.486 | 0.075 | 0.002 | pass |
| M2 | French ×(−2) + Chinese ×2 → *p*(Chinese “big”) | 0.846 | &lt;0.001 | 0.0002 | pass |
| M3 | small ×(−2) + big ×2 → *p*(“petit”) | 0.863 | 0.012 | 0.0002 | pass |
| L1 | China ×0 → drop of *p*(yuan) | 0.381 | 0.031 | 0.0002 | pass |
| L2 | China ×0 + Thailand/SEA ×1 → *p*(“ba”) | 0.638 | 0.015 | 0.0002 | pass |
| L3 | China ×0 + Thailand ×2 at the colon → *p*(“ba”) | 0.275 | 0.002 | 0.0002 | pass |
| L4 | currency ×0 + language ×5 → *p*(Chinese) | 0.164 | 0.027 | 0.0002 | pass |
| L5 | US ×0 + Scandinavia/Denmark ×5 → *p*(Scandinavian languages) | 0.451 | 0.036 | 0.0002 | pass |

<p class="sn-how"><b>How to read it.</b> D and M: Gemma-2-2B, from the circuit-tracer tutorial; L: Llama-3.2-1B, from its demo. Observed: the claimed metric under the published intervention. 0.0002 is the smallest <i>p</i> that 5,000 draws can give. “ba” is the first token of “baht”.</p>

</div></div>

### The null model changes the verdict on two-hop recall

<div class="sn">
<figure class="sn-fig" id="sn-shares"><div class="sn-ix"><div class="sn-ctl"><span class="l">Claims</span><div class="sn-seg" id="sn-kind"><button type="button" data-v="abl">ablation</button><button type="button" data-v="swap">swap</button></div><span class="l">Null</span><div class="sn-seg" id="sn-null2"><button type="button" data-v="act">activation-matched</button><button type="button" data-v="infl">influence-matched</button></div><span class="l">Null pool excludes</span><div class="sn-seg" id="sn-pool"><button type="button" data-v="common">features of every compared rule</button><button type="button" data-v="orig">only the rule's own run</button></div></div><div class="sn-leg"><span><i style="background:var(--sn-b)"></i>rule B, contrastive</span><span><i style="background:var(--sn-c)"></i>rule C, held-out causal</span><span><span class="w"></span>Wilson 95% interval</span></div><div class="sn-sh" id="sn-sh"></div><div class="sn-axis"><span>0%</span><span>share of claims judged specific</span><span>100%</span></div><p class="sn-read" id="sn-read2"></p><div class="sn-tip" id="sn-tip2"></div></div><figcaption><span class="sn-tag">Interactive</span><b class="sn-title">Same claims, two nulls: switch the null and the two-hop shares fall; agreement barely moves.</b>Share of automatically constructed claims judged specific (Benjamini–Hochberg at <i>q</i> = 0.05 within each family), by task, model and transcoder. The values are those of Figure 3 and Table 6 of the paper; hover or focus a bar for the counts. The third switch shows the original runs, in which a feature selected by one rule could be drawn as a replacement for the other rule's claims.</figcaption></figure>
<p class="sn-lead">On two-hop recall, between 11 and 77 claims per family are specific only under the activation-matched null, and at most one only under the influence-matched null (exact one-sided McNemar test, all <i>p</i> ≤ 0.001). On agreement the two nulls differ little: no family reaches <i>p</i> &lt; 0.01. A probe gives partial support for one explanation. At the city position, activation-matched replacements are nearly inert (median single-feature effect near zero, against 0.004 in the influence-matched pool), so beating them is an easy bar.</p>
<p class="sn-lead"><b>What the null may draw matters too.</b> In the original runs, each rule's null pool excluded only the features of the rules run with it. There, 0–56% of contrastive per-layer two-hop claims were specific, against 10–82% once the pools exclude the features of every compared rule, while the observed effects barely changed. Features that other rules select are strong replacements, so absolute shares depend on what the pools exclude.</p>
</div>

### The selection rule changes which claims pass

<div class="sn"><div class="sn-table" markdown="1">

<span class="sn-tag">Table 2</span> **Two-hop recall under the influence-matched null: share of specific claims with rule B and with rule C.** Null pools exclude the features of all compared rules.

| Model · transcoder (prompts) | Ablation, B → C | C-only / B-only | *p* | Swap, B → C | C-only / B-only | *p* |
|---|---|---|---|---|---|---|
| Gemma-2-2B · PLT (163) | 60.1% → **84.0%** | 43 / 4 | 3·10⁻⁹ | 81.6% → 80.4% | 14 / 16 | 0.86 |
| Llama-3.2-1B · PLT (92) | 44.6% → **78.0%** | 40 / 10 | 2·10⁻⁵ | 9.8% → **56.7%** | 44 / 2 | 3·10⁻¹¹ |
| Gemma-2-2B · CLT (163) | 85.9% → 82.2% | 12 / 18 | 0.36 | 85.3% → **70.6%** | 10 / 34 | 4·10⁻⁴ |
| Llama-3.2-1B · CLT (92) | 83.7% → 77.8% | 8 / 13 | 0.38 | 48.9% → **72.2%** | 23 / 2 | 2·10⁻⁵ |

<p class="sn-how"><b>How to read it.</b> C-only / B-only: claims that are specific under one rule and not the other. <i>p</i>: exact two-sided McNemar test on those counts; bold marks a difference at <i>p</i> &lt; 0.05, in either direction (for Gemma's cross-layer swaps, rule C is worse).</p>

</div></div>

<div class="sn">
<p class="sn-lead"><b>Ablation claims.</b> With per-layer transcoders, rule C yields more specific claims than rule B (Gemma 60% → 84%, Llama 45% → 78%). Part of this advantage is built in: rule C selects features by their ablation effect on a prompt with the same answer, and the test asks whether they have a larger ablation effect than matched features. The advantage survives a rule-matched null that draws other features the same rule ranks highly (91% against 73%, and 95% against 52%), and rule C′, which never selects on the answer, behaves like rule C.</p>
<p class="sn-lead"><b>Swap claims</b>, which no rule selects on, are the more informative comparison. Rule C helps for Llama (10% → 57%) and not for Gemma (82% against 80%). With cross-layer transcoders, contrastive supernodes are specific more often (49–86%), and rule C does not improve ablation claims. Across both transcoder types, the evidence that causal selection finds more specific features therefore rests on Llama.</p>
</div>

### Most two-hop ablation claims that pass are weak

<div class="sn">
<p class="sn-lead">On two-hop recall, 55–58% (Gemma) and 92–95% (Llama) of the specific per-layer ablation claims beat their null's 95th percentile by less than a factor of two, and 72–97% of the cross-layer ones. Suppressing ten equally important features at the city position removes a similar share of the answer. Swap claims are rarely weak (0–18%), and on agreement most specific ablation claims are strong (5–16% weak).</p>
<figure class="sn-fig"><img src="/images/projects/2026-11-supernode-specificity/fig5-effect-null-ratio.png" alt="Figure 5: effect divided by the null's 95th percentile under the two nulls" loading="lazy"><figcaption><span class="sn-tag">Figure 5</span><b class="sn-title">How far claims clear their null.</b>Effect divided by the null's 95th percentile (rule B, per-layer transcoders) under the activation-matched null (light) and the influence-matched null (dark). Dashed line: ratio 1. Dotted line: ratio 2, the threshold for a weak claim. On two-hop recall the ratios under the activation-matched null lie well above those under the influence-matched null; on agreement the two overlap. Gray numbers: share of claims below 0.01, which are clipped.</figcaption></figure>
</div>

## What a pass or a failure means

<div class="sn">
<div class="sn-box"><p>The test does not certify a label. It asks whether the labelled set changes the claimed output more than features that the attribution graph treats as equally important at the same position. <b>A pass</b> shows that not any such set would do. <b>A failure</b> shows only that the claim, as stated about these features, is not supported over that alternative: the label may be wrong, or the variable may be encoded redundantly by unlabelled features in the null pool. The test alone cannot tell these two apart.</p></div>
</div>

## What to report

<div class="sn">
<p class="sn-lead">Because selection rules and nulls change verdicts, we suggest that an intervention result on an attribution graph comes with:</p>
<ul class="sn-take">
<li><b>A claim-specific metric, fixed in advance.</b> Prefer swap claims, or metrics such as the answer's share among its alternatives, to a generic drop in <i>p</i>(answer), which any strong suppression produces.</li>
<li><b>The influence-matched <i>p</i>-value, with enough draws for the planned correction.</b> Holm over <i>m</i> claims at level α needs <i>n</i> ≥ <i>m</i>/α − 1 draws. Exclude from the pools every feature that any compared rule selects.</li>
<li><b>The effect-to-null ratio and the upper tail of the null.</b> Describe claims close to the null as borderline.</li>
<li><b>The supernode-selection rule, compared with at least one alternative.</b> Which rule was better depended on the transcoder.</li>
<li><b>Sensitivity to the intervention scale.</b></li>
<li><b>More than one task family</b> before generalising.</li>
</ul>
</div>

## Using the test

<div class="sn">
<p class="sn-lead">The test is a small library on top of circuit-tracer and will be released with the paper. Testing one ablation claim against the influence-matched null:</p>
<div class="sn-code">from spectest import Claim, Group, SpecificityTester

t = SpecificityTester(model)                 # model: a circuit-tracer ReplacementModel
claim = Claim(prompt,
              [Group(S, prompt, -2.0, None)],  # supernode S, taken from this prompt, suppressed at -2
              metric=lambda p: base - p[answer_id].item(),
              exclude=set(S))
t.test(claim, "influence", n=500, k=20, gen=torch.Generator().manual_seed(0))
# -> {"obs", "null_mean", "null_p95", "null_max", "p", "n"}</div>
</div>

## Scope and limitations

<div class="sn">
<ul class="sn-take">
<li class="scope"><b>Small models, simple tasks.</b> Two open models, one template for two-hop recall and two for agreement, one intervention protocol with frozen attention, and public transcoders only. The two nulls may agree on our agreement task because it is simple.</li>
<li class="scope"><b>Our rules are heuristics.</b> We did not test published annotation pipelines, and the published claims we did test, those of the tutorial and demo, mostly pass.</li>
<li class="scope"><b>The null is drawn feature by feature.</b> Set-level properties such as the layer distribution of a supernode are not matched, and absolute shares depend on what the pools exclude.</li>
<li class="scope"><b>Verdicts near the threshold are fragile.</b> They depend on the number of draws, the correction and numerical details: in bf16, D4 fails on one GPU type and passes on another, and in float32 it fails on both.</li>
<li class="scope"><b>Confirmatory and exploratory analyses are marked in the paper.</b> The common exclusion sets and the robustness analyses were specified after the full results were known and are exploratory.</li>
</ul>
<details class="sn-abs"><summary>Abstract</summary><p>Circuit-tracing studies support claims by suppressing or swapping labelled groups of transcoder features (supernodes), but a successful intervention does not show that these features are special. We compare such claims with random replacements matched on activation or attribution influence; a claim is specific if it beats the influence-matched null after multiple-comparison correction. We test 14 hand-curated claims and 2,958 automatically constructed ones from Gemma-2-2B and Llama-3.2-1B, with per-layer and cross-layer transcoders, on two-hop factual recall and subject–verb agreement. Thirteen of the 14 curated claims pass. On two-hop recall, 10–82% of the per-layer claims built by contrasting a prompt with a donor are specific (52–95% on agreement), and 0–56% if the null may draw features that other rules select. Selecting features by their causal effect on a sibling prompt yields more specific ablation claims with per-layer transcoders, partly by construction, and more specific swap claims only for Llama. The usual activation-matched null declares 89–100% of the contrastive per-layer two-hop claims specific; on agreement the two nulls differ little. Most passing two-hop ablation claims beat their null by less than a factor of two. Because selection rules and nulls change verdicts, we give reporting recommendations and provide the test as a library.</p></details>
</div>

## Paper and code

Paper, code and BibTeX will be linked here once the preprint is public.

{% raw %}
<script>
(function () {
  var H = [{"id":"D1","model":"Gemma-2-2B","iv":"“say a capital” ×(−2)","metric":"p(Texas)","kind":"ablation","obs":0.4596,"holm":"pass","lo":0.0,"hi":0.478,"nulls":{"act":{"n":500,"bins":[1,12,71,76,96,103,63,30,21,14,10,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.09079,"max":0.127,"mean":0.05064,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[116,431,445,527,725,601,413,299,226,151,108,119,130,112,99,78,70,64,53,43,42,29,20,14,20,16,15,10,10,7,1,2,2,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.1852,"max":0.3448,"mean":0.07056,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"D2","model":"Gemma-2-2B","iv":"“capital” ×(−2)","metric":"drop of p(Austin)","kind":"ablation","obs":0.407,"holm":"pass","lo":-0.05548,"hi":0.4251,"nulls":{"act":{"n":500,"bins":[0,2,7,21,42,73,71,109,104,41,22,7,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.04598,"max":0.06752,"mean":0.01587,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[1,21,24,115,228,327,436,548,524,543,452,409,360,315,251,171,122,60,36,23,16,11,4,1,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.106,"max":0.1937,"mean":0.0432,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"D3","model":"Gemma-2-2B","iv":"“Texas” ×(−2)","metric":"drop of p(Austin)","kind":"ablation","obs":0.423,"holm":"pass","lo":-0.1694,"hi":0.4462,"nulls":{"act":{"n":500,"bins":[0,0,0,1,2,5,3,8,27,33,28,19,52,92,100,63,31,17,15,3,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.05479,"max":0.1052,"mean":0.001307,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[1,11,30,42,72,130,151,228,285,444,466,457,451,392,428,364,312,254,158,111,72,55,39,14,9,5,4,5,4,3,0,0,0,2,0,0,1,0,0,0,0,0,0,0,0,0,0,0],"p95":0.08266,"max":0.2949,"mean":-0.007854,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"D4","model":"Gemma-2-2B","iv":"“state” ×(−2)","metric":"drop of p(Austin)","kind":"ablation","obs":0.0505,"holm":"fail","lo":-0.02683,"hi":0.05601,"nulls":{"act":{"n":500,"bins":[0,3,1,5,6,0,1,1,9,9,22,14,24,12,25,24,17,13,27,22,33,16,27,29,24,24,28,14,9,7,17,8,4,1,0,7,5,5,1,0,0,3,0,0,2,0,1,0],"p95":0.02981,"max":0.05289,"mean":0.00839,"ge":1,"geq":[0.05289],"p":0.003992015968063872},"infl":{"n":5000,"bins":[25,0,63,25,46,21,47,84,54,92,85,107,189,238,160,194,140,226,209,136,161,318,275,211,245,57,247,85,107,210,174,219,87,16,80,46,40,45,46,23,20,31,0,0,36,80,0,0],"p95":0.03854,"max":0.05251,"mean":0.01073,"ge":80,"geq":[0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05161,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05231,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251,0.05251],"p":0.016196760647870425}}},{"id":"D5","model":"Gemma-2-2B","iv":"Texas ×(−2) + California ×2","metric":"p(Sacramento)","kind":"swap","obs":0.3637,"holm":"pass","lo":0.0,"hi":0.3782,"nulls":{"act":{"n":500,"bins":[471,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.008164,"max":0.01129,"mean":0.005575,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[4983,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.004502,"max":0.0157,"mean":0.001627,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[448,802,1135,1226,857,278,162,73,18,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.04745,"max":0.07682,"mean":0.02473,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"D6","model":"Gemma-2-2B","iv":"Texas ×(−2) + China ×2","metric":"p(Beijing)","kind":"swap","obs":0.2125,"holm":"pass","lo":0.0,"hi":0.221,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":2.1e-05,"max":6.2e-05,"mean":6.428e-06,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[5000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":7.105e-05,"max":0.002689,"mean":2.134e-05,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[5000,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":1.4e-05,"max":0.000111,"mean":3.99e-06,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"M1","model":"Gemma-2-2B","iv":"French ×(−2)","metric":"p(“big”)","kind":"ablation","obs":0.4865,"holm":"pass","lo":0.0,"hi":0.6992,"nulls":{"act":{"n":500,"bins":[414,8,17,10,10,6,4,7,5,6,0,1,2,1,0,1,1,0,1,0,1,0,2,0,0,2,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.1142,"max":0.5145,"mean":0.02016,"ge":1,"geq":[0.5145],"p":0.003992015968063872},"infl":{"n":5000,"bins":[3913,428,194,121,76,51,33,18,24,18,17,15,9,14,10,9,6,6,4,1,1,1,0,5,5,0,3,3,0,4,0,1,1,1,0,2,0,1,0,0,0,1,1,0,1,1,1,0],"p95":0.07531,"max":0.6723,"mean":0.01684,"ge":9,"geq":[0.4904,0.5142,0.5205,0.5393,0.5983,0.6226,0.6467,0.6639,0.6723],"p":0.001999600079984003}}},{"id":"M2","model":"Gemma-2-2B","iv":"French ×(−2) + Chinese ×2","metric":"p(Chinese “big”)","kind":"swap","obs":0.8458,"holm":"pass","lo":0.0,"hi":0.8796,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":2.9e-05,"max":0.000612,"mean":7.79e-06,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[4998,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.0002163,"max":0.02187,"mean":8.815e-05,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[3895,723,245,80,31,18,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.04445,"max":0.1271,"mean":0.01139,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"M3","model":"Gemma-2-2B","iv":"small ×(−2) + big ×2","metric":"p(“petit”)","kind":"swap","obs":0.8632,"holm":"pass","lo":0.0,"hi":0.8977,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.00709,"max":0.008478,"mean":0.005682,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[4916,66,11,2,1,0,1,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.01205,"max":0.3289,"mean":0.005585,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[1303,716,483,319,274,207,196,142,153,132,115,100,71,80,61,72,66,52,44,52,46,33,53,30,32,25,15,21,21,24,15,14,17,10,2,4,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.4229,"max":0.6682,"mean":0.1144,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"L1","model":"Llama-3.2-1B","iv":"China ×0","metric":"drop of p(yuan)","kind":"ablation","obs":0.381,"holm":"pass","lo":-0.07962,"hi":0.3991,"nulls":{"act":{"n":500,"bins":[0,0,0,0,0,0,6,41,106,49,176,90,21,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.04671,"max":0.05832,"mean":0.02049,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[1,8,38,112,264,536,844,911,829,760,411,187,81,16,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.03142,"max":0.0682,"mean":-0.001902,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"L2","model":"Llama-3.2-1B","iv":"China ×0 + Thailand/SEA ×1","metric":"p(“ba”)","kind":"swap","obs":0.638,"holm":"pass","lo":0.0,"hi":0.6635,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.0005711,"max":0.000879,"mean":0.0004325,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[4718,247,31,3,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.01479,"max":0.09305,"mean":0.004287,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[622,1758,891,437,235,193,155,134,94,93,89,64,54,32,26,39,20,12,10,9,11,4,3,9,3,2,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.1619,"max":0.3815,"mean":0.04954,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"L3","model":"Llama-3.2-1B","iv":"China ×0 + Thailand ×2 at the colon","metric":"p(“ba”)","kind":"swap","obs":0.2747,"holm":"pass","lo":0.0,"hi":0.2857,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.0007741,"max":0.001643,"mean":0.0004379,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[4988,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.00205,"max":0.009538,"mean":0.0006082,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[4850,140,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.00513,"max":0.01501,"mean":0.001646,"ge":0,"geq":[],"p":0.0001999600079984003}}},{"id":"L4","model":"Llama-3.2-1B","iv":"currency ×0 + language ×5","metric":"p(Chinese)","kind":"swap","obs":0.164,"holm":"pass","lo":0.0,"hi":0.173,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.001015,"max":0.002986,"mean":0.0004578,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[2198,1200,596,298,185,129,97,77,40,35,29,25,18,12,6,7,8,4,4,10,3,1,3,5,1,4,1,0,0,1,0,0,0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.02718,"max":0.1229,"mean":0.00795,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[855,1072,737,493,401,291,242,146,108,112,84,69,75,71,27,27,30,24,25,16,16,10,15,3,5,13,6,5,3,4,2,2,3,3,2,0,0,0,0,0,1,0,1,0,0,0,1,0],"p95":0.05001,"max":0.1663,"mean":0.01599,"ge":1,"geq":[0.1663],"p":0.0003999200159968006}}},{"id":"L5","model":"Llama-3.2-1B","iv":"US ×0 + Scandinavia/Denmark ×5","metric":"p(Scandinavian languages)","kind":"swap","obs":0.4508,"holm":"pass","lo":0.0,"hi":0.4689,"nulls":{"act":{"n":500,"bins":[500,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.001623,"max":0.002743,"mean":0.0011,"ge":0,"geq":[],"p":0.001996007984031936},"infl":{"n":5000,"bins":[2752,1315,535,207,105,39,19,18,7,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.03575,"max":0.09567,"mean":0.01236,"ge":0,"geq":[],"p":0.0001999600079984003},"inject":{"n":5000,"bins":[1370,1466,919,511,323,167,110,61,29,18,18,6,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"p95":0.05803,"max":0.1278,"mean":0.0219,"ge":0,"geq":[],"p":0.0001999600079984003}}}];
  var SH = {"common":{"2-hop Gemma PLT":{"B":[[148,163],[98,163],[163,163],[133,163]],"C":[[153,163],[137,163],[156,163],[131,163]]},"2-hop Llama PLT":{"B":[[82,92],[41,92],[86,92],[9,92]],"C":[[90,91],[71,91],[83,90],[51,90]]},"2-hop Gemma CLT":{"B":[[160,163],[140,163],[161,163],[139,163]],"C":[[154,163],[134,163],[148,163],[115,163]]},"2-hop Llama CLT":{"B":[[88,92],[77,92],[67,92],[45,92]],"C":[[81,90],[70,90],[82,90],[65,90]]},"SVA Gemma PLT":{"B":[[89,135],[86,135],[128,135],[128,135]],"C":[[120,133],[114,133],[127,131],[126,131]]},"SVA Llama PLT":{"B":[[59,110],[57,110],[106,110],[102,110]],"C":[[67,94],[64,94],[77,77],[77,77]]}},"orig":{"2-hop Gemma PLT":{"B":[[146,163],[48,163],[162,163],[92,163]],"C":[[152,163],[126,163],[154,163],[117,163]]},"2-hop Llama PLT":{"B":[[81,92],[20,92],[85,92],[0,92]],"C":[[90,91],[66,91],[85,90],[40,90]]},"2-hop Gemma CLT":{"B":[[159,163],[135,163],[160,163],[134,163]],"C":[[152,163],[126,163],[147,163],[109,163]]},"2-hop Llama CLT":{"B":[[87,92],[75,92],[67,92],[42,92]],"C":[[76,90],[59,90],[75,90],[51,90]]}}};
  function $(id) { return document.getElementById(id); }
  function press(el, v) { el.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.v === v); }); }
  function num(x) { var a = Math.abs(x); return a === 0 ? "0" : a < 0.001 ? "<0.001" : x.toFixed(a < 0.1 ? 3 : a < 1 ? 3 : 2); }
  function pv(p) { return p < 0.001 ? p.toFixed(4) : p.toFixed(3); }
  function cnt(n) { return n.toLocaleString("en-US"); }
  function tipFor(tip, host) {
    return { show: function (html, ev) {
      tip.innerHTML = html; tip.classList.add("on");
      var r = host.getBoundingClientRect(), x = ev.clientX - r.left + 12, y = ev.clientY - r.top + 14;
      if (x + tip.offsetWidth > r.width) x = Math.max(0, ev.clientX - r.left - tip.offsetWidth - 12);
      tip.style.left = x + "px"; tip.style.top = y + "px";
    }, hide: function () { tip.classList.remove("on"); } };
  }

  /* ---------- one claim against its null (the draws behind Table 1 and Figure 2) ---------- */
  var NULLNAME = { act: "activation-matched null", infl: "influence-matched null", inject: "inject-only null" };
  var NOTE = {
    D4: " The tutorial itself describes this intervention as largely ineffective.",
    M1: " Passing does not mean that only the labelled features can do this."
  };
  var cur = "D3", nul = "infl";
  var plot = $("sn-plot"), tip1 = tipFor($("sn-tip1"), plot.parentNode);
  function claim() { return H.filter(function (c) { return c.id === cur; })[0]; }
  function ticks(lo, hi, m) {
    var raw = (hi - lo) / m, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p, st = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p, out = [];
    for (var v = Math.ceil(lo / st - 1e-9) * st; v <= hi + 1e-9; v += st) out.push(Math.abs(v) < st / 1e6 ? 0 : v);
    return { v: out, st: st };
  }
  function tfmt(v, st) { var d = Math.max(0, -Math.floor(Math.log10(st) + 1e-9)); return v.toFixed(Math.min(d, 4)); }
  function drawClaim() {
    var c = claim();
    if (!c.nulls[nul]) nul = "infl";
    var N = c.nulls[nul];
    $("sn-claims").querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.v === cur); });
    $("sn-nulls").querySelectorAll("button").forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.v === nul);
      if (b.dataset.v === "inject") { b.disabled = !c.nulls.inject; b.title = c.nulls.inject ? "" : "Swap claims only"; }
    });
    $("sn-claim").innerHTML = "<b>" + c.id + "</b> <span class=\"m\">· " + c.model + " · " + c.kind + " claim</span><br>" + c.iv + " → " + c.metric;

    var W = Math.max(280, plot.clientWidth), Ht = W < 480 ? 220 : 250, L = 42, R = 14, T = 40, B = 34, pw = W - L - R, ph = Ht - T - B, yb = T + ph;
    function X(v) { return L + (v - c.lo) / (c.hi - c.lo) * pw; }
    var nb = N.bins.length, bw = pw / nb, mx = Math.max.apply(null, N.bins) / N.n;
    var yt = ticks(0, mx, 2), ymax = Math.max(mx, yt.v[yt.v.length - 1]);
    function Y(s) { return yb - s / ymax * ph; }
    var s = '<svg viewBox="0 0 ' + W + " " + Ht + '" width="' + W + '" height="' + Ht + '" aria-hidden="true">';
    yt.v.forEach(function (v) { if (v > 0) s += '<line class="grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>';
      s += '<text class="ax" x="' + (L - 6) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + Math.round(v * 100) + "%</text>"; });
    N.bins.forEach(function (k, i) {
      if (!k) return;
      var x = L + i * bw + 0.75, w = Math.max(1, bw - 1.5), y = Math.min(Y(k / N.n), yb - 1.5), r = Math.min(2, w / 2, (yb - y));
      s += '<path class="bar" d="M' + x + "," + yb + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y + "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + yb + 'Z"/>';
    });
    s += '<line class="base" x1="' + L + '" x2="' + (W - R) + '" y1="' + yb + '" y2="' + yb + '"/>';
    var xt = ticks(c.lo, c.hi, W < 480 ? 4 : 6);
    xt.v.forEach(function (v) { s += '<text class="ax" x="' + X(v) + '" y="' + (yb + 16) + '" text-anchor="middle">' + tfmt(v, xt.st) + "</text>"; });
    s += '<text class="ax" x="' + (L + pw / 2) + '" y="' + (Ht - 2) + '" text-anchor="middle">' + c.metric + "</text>";
    N.geq.forEach(function (v) { s += '<line class="rug" x1="' + X(v) + '" x2="' + X(v) + '" y1="' + (yb - 9) + '" y2="' + (yb - 1) + '"/>'; });
    var xp = X(N.p95), xo = X(c.obs), right = xo > L + pw * 0.55;
    s += '<line class="p95" x1="' + xp + '" x2="' + xp + '" y1="' + (T - 8) + '" y2="' + yb + '"/>';
    s += '<text class="ax" x="' + (xp + 5) + '" y="' + (T - 10) + '">null 95th pct. ' + num(N.p95) + "</text>";
    s += '<line class="obs" x1="' + xo + '" x2="' + xo + '" y1="' + (T - 22) + '" y2="' + yb + '"/><circle class="obsdot" cx="' + xo + '" cy="' + (T - 22) + '" r="5"/>';
    s += '<text class="lab" x="' + (xo + (right ? -10 : 10)) + '" y="' + (T - 26) + '" text-anchor="' + (right ? "end" : "start") + '">published supernode ' + num(c.obs) + "</text>";
    for (var i = 0; i < nb; i++) s += '<rect class="hit" data-i="' + i + '" x="' + (L + i * bw) + '" y="' + T + '" width="' + bw + '" height="' + ph + '"/>';
    s += "</svg>";
    plot.innerHTML = s;
    plot.setAttribute("aria-label", "Histogram of the " + NULLNAME[nul] + " for claim " + c.id + ": " + cnt(N.n) + " random draws, 95th percentile " + num(N.p95) +
      ", largest " + num(N.max) + "; the published supernode reaches " + num(c.obs) + ".");
    plot.querySelectorAll(".hit").forEach(function (r) {
      r.addEventListener("mousemove", function (ev) {
        var i = +r.dataset.i, a = c.lo + i * (c.hi - c.lo) / nb, b = a + (c.hi - c.lo) / nb, k = N.bins[i];
        tip1.show("<b>" + a.toFixed(3) + " to " + b.toFixed(3) + "</b><br>" + cnt(k) + " of " + cnt(N.n) + " random draws (" + (100 * k / N.n).toFixed(k / N.n < 0.01 && k ? 2 : 1) + "%)", ev);
      });
      r.addEventListener("mouseleave", tip1.hide);
    });
    var ratio = N.p95 > 0 ? c.obs / N.p95 : Infinity;
    var t = "<b>" + NULLNAME[nul].charAt(0).toUpperCase() + NULLNAME[nul].slice(1) + ".</b> The published supernode gives " + num(c.obs) +
      (ratio > 1000 ? ", more than 1,000 times" : ", " + (ratio < 10 ? ratio.toFixed(1) : Math.round(ratio)) + " times") + " the null's 95th percentile. " +
      (N.ge === 0 ? "None" : cnt(N.ge)) + " of " + cnt(N.n) + " random replacements " + (N.ge === 1 ? "reaches" : "reach") + " it" +
      (N.ge ? " (largest " + num(N.max) + ")" : "") + ", so <i>p</i> = " + pv(N.p) + (N.ge === 0 ? ", the smallest value " + cnt(N.n) + " draws can give" : "") + ". ";
    if (nul === "infl") t += c.holm === "pass" ? "<b>Passes</b> Holm correction over the 14 claims at α = 0.01." : "<b>Fails</b> Holm correction over the 14 claims at α = 0.01.";
    else if (nul === "inject") t += "Here the real suppression is kept and only the injected features are random.";
    else t += "The paper's verdicts use the influence-matched null.";
    if (nul === "infl" && NOTE[c.id]) t += NOTE[c.id];
    $("sn-read").innerHTML = t;
  }
  H.forEach(function (c) {
    var b = document.createElement("button"); b.type = "button"; b.textContent = c.id; b.dataset.v = c.id; b.title = c.iv + " → " + c.metric;
    b.onclick = function () { cur = c.id; drawClaim(); }; $("sn-claims").appendChild(b);
  });
  $("sn-nulls").querySelectorAll("button").forEach(function (b) { b.onclick = function () { nul = b.dataset.v; drawClaim(); }; });

  /* ---------- shares of specific claims (Figure 3 / Table 6) ---------- */
  var ROWS = [["Two-hop recall", [["2-hop Gemma PLT", "Gemma-2-2B", "per-layer"], ["2-hop Llama PLT", "Llama-3.2-1B", "per-layer"], ["2-hop Gemma CLT", "Gemma-2-2B", "cross-layer"], ["2-hop Llama CLT", "Llama-3.2-1B", "cross-layer"]]],
              ["Subject–verb agreement", [["SVA Gemma PLT", "Gemma-2-2B", "per-layer"], ["SVA Llama PLT", "Llama-3.2-1B", "per-layer"]]]];
  var RULE = { B: "Rule B, contrastive", C: "Rule C, held-out causal" };
  var kind = "abl", null2 = "act", pool = "common";
  var sh = $("sn-sh"), tip2 = tipFor($("sn-tip2"), sh.parentNode), built = false;
  function cell(key, rule, k, n2, p) { var t = (p === "orig" && SH.orig[key]) ? SH.orig : SH.common; return t[key][rule][(k === "abl" ? 0 : 2) + (n2 === "act" ? 0 : 1)]; }
  function wilson(k, n) { var z = 1.959964, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d; return [Math.max(0, c - h), Math.min(1, c + h)]; }
  function range(keys, rules, n2) {
    var v = []; keys.forEach(function (key) { rules.forEach(function (r) { var c = cell(key, r, kind, n2, pool); v.push(100 * c[0] / c[1]); }); });
    return Math.round(Math.min.apply(null, v)) + "–" + Math.round(Math.max.apply(null, v)) + "%";
  }
  function drawShares() {
    press($("sn-kind"), kind); press($("sn-null2"), null2); press($("sn-pool"), pool);
    if (!built) {
      ROWS.forEach(function (g) {
        var h = document.createElement("div"); h.className = "grp"; h.textContent = g[0]; sh.appendChild(h);
        g[1].forEach(function (r) {
          var row = document.createElement("div"); row.className = "row";
          row.innerHTML = '<span class="nm">' + r[1] + "<small>" + r[2] + ' transcoder</small></span><div class="bars">' + ["B", "C"].map(function (rule) {
            return '<div class="bar ' + rule.toLowerCase() + '" tabindex="0" data-k="' + r[0] + '" data-r="' + rule + '"><span class="fl"></span><span class="ci"></span><span class="vl"></span></div>'; }).join("") + "</div>";
          sh.appendChild(row);
        });
      });
      sh.querySelectorAll(".bar").forEach(function (b) {
        function html() { var c = cell(b.dataset.k, b.dataset.r, kind, null2, pool), w = wilson(c[0], c[1]);
          return "<b>" + RULE[b.dataset.r] + "</b><br>" + c[0] + " of " + c[1] + " claims specific (" + (100 * c[0] / c[1]).toFixed(1) + "%)<br>Wilson 95% interval " + (100 * w[0]).toFixed(1) + "–" + (100 * w[1]).toFixed(1) + "%"; }
        b.addEventListener("mousemove", function (ev) { tip2.show(html(), ev); });
        b.addEventListener("mouseleave", tip2.hide);
        b.addEventListener("focus", function () { var r = b.getBoundingClientRect(); tip2.show(html(), { clientX: r.left + 40, clientY: r.bottom - 8 }); });
        b.addEventListener("blur", tip2.hide);
      });
      built = true;
    }
    sh.querySelectorAll(".bar").forEach(function (b) {
      var c = cell(b.dataset.k, b.dataset.r, kind, null2, pool), p = 100 * c[0] / c[1], w = wilson(c[0], c[1]);
      b.querySelector(".fl").style.width = p + "%";
      var ci = b.querySelector(".ci"); ci.style.left = 100 * w[0] + "%"; ci.style.width = 100 * (w[1] - w[0]) + "%";
      var v = b.querySelector(".vl"); v.style.left = Math.max(p, 100 * w[1]) + "%"; v.innerHTML = "<small>" + b.dataset.r + "</small>" + p.toFixed(1) + "%";
      b.setAttribute("aria-label", RULE[b.dataset.r] + ": " + c[0] + " of " + c[1] + " claims specific, " + p.toFixed(1) + " percent");
    });
    var two = ROWS[0][1].map(function (r) { return r[0]; }), sva = ROWS[1][1].map(function (r) { return r[0]; });
    var what = kind === "abl" ? "ablation" : "swap", r = $("sn-read2");
    r.className = "sn-read" + (null2 === "infl" ? " acc" : "");
    r.innerHTML = (null2 === "act"
      ? "<b>Activation-matched null:</b> on two-hop recall, " + range(two, ["B", "C"], "act") + " of the " + what + " claims count as specific; on agreement, " + range(sva, ["B", "C"], "act") + "."
      : "<b>Influence-matched null:</b> on two-hop recall the share falls to " + range(two, ["B", "C"], "infl") + " (from " + range(two, ["B", "C"], "act") +
        "). On agreement it is " + range(sva, ["B", "C"], "infl") + ", against " + range(sva, ["B", "C"], "act") + " under the activation-matched null.") +
      (pool === "orig" ? " Original runs: another rule's features could be drawn as replacements. The agreement rows are unchanged, because rules B and C were always run together there." : "");
  }
  [["sn-kind", function (v) { kind = v; }], ["sn-null2", function (v) { null2 = v; }], ["sn-pool", function (v) { pool = v; }]].forEach(function (p) {
    $(p[0]).querySelectorAll("button").forEach(function (b) { b.onclick = function () { p[1](b.dataset.v); drawShares(); }; });
  });

  drawClaim(); drawShares();
  var rz; window.addEventListener("resize", function () { clearTimeout(rz); rz = setTimeout(drawClaim, 120); });

  /* show the change once without a click: when the shares figure is first seen, switch to the influence-matched null */
  var touched = false, fig = $("sn-shares");
  fig.addEventListener("pointerdown", function () { touched = true; }, true);
  fig.addEventListener("keydown", function () { touched = true; }, true);
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!calm && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (es) {
      if (!es[0].isIntersecting) return;
      io.disconnect();
      setTimeout(function () { if (!touched && null2 === "act") { null2 = "infl"; drawShares(); } }, 1600);
    }, { threshold: 0.6 });
    io.observe(sh);
  } else if (calm) { null2 = "infl"; drawShares(); }
})();
</script>
{% endraw %}
