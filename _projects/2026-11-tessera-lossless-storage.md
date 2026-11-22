---
title: "Predict, Don't Just Pair: Lossless Storage for Families of Fine-Tuned Models"
collection: projects
permalink: /projects/2026-11-tessera-lossless-storage/
date: 2026-11-02
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "Lossless stores for fine-tuned checkpoints focus on which tensor to delta-encode against which. Exhaustive ground truth on 130 checkpoints shows that pairing is nearly solved and leaves most of the redundancy behind; Tessera predicts and codes the residual from the base and reduces storage by 74.7%."
teaser: "/images/projects/2026-11-tessera-lossless-storage/teaser.png"
tags: ["model hubs", "lossless compression", "delta storage", "entropy coding"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-tessera-lossless-storage"
published: false
---

<style>
/* Tessera project page: styles scoped to .ts / .ts-* so the rest of the site is untouched; colors follow the theme */
.ts{--ts-acc:#1763a6;--ts-acc-soft:rgba(23,99,166,.09);--ts-amber:#b97a00;--ts-amber-fill:#e0a030;--ts-green:#0a7d5c;--ts-red:#c8372d;--ts-red-soft:rgba(200,55,45,.12);--ts-gray:#6f767e;--ts-gray-soft:rgba(127,127,127,.10);--ts-line:rgba(127,127,127,.28);--ts-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .ts{--ts-acc:#78b0f0;--ts-acc-soft:rgba(120,176,240,.14);--ts-amber:#f0b440;--ts-amber-fill:#d9a23a;--ts-green:#4fd1a1;--ts-red:#f26b67;--ts-red-soft:rgba(242,107,103,.18);--ts-gray:#c3c9cf;--ts-gray-soft:rgba(255,255,255,.08);--ts-line:rgba(255,255,255,.2);--ts-card:rgba(255,255,255,.04)}
/* the fixed table of contents on wide screens would otherwise cover the end of the title */
@media screen and (min-width:1200px){.page__inner-wrap>header{padding-right:250px}}
.ts-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.ts-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--ts-acc-soft);color:var(--ts-acc)}
.ts-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.ts-tldr b{color:var(--ts-acc)}
.ts-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.ts-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.ts-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.ts-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--ts-line)}
.ts-btn.muted{background:transparent;color:var(--ts-gray)!important;border:1px dashed var(--ts-line);cursor:default}
.ts-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .ts-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.ts-lead{margin:0 0 1em;line-height:1.65}
.ts-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--ts-line);border-radius:14px;background:var(--ts-card)}
.ts-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.ts-fig.narrow img{max-width:min(100%,560px);margin:0 auto}
.ts-fig a.zoom{display:block;cursor:zoom-in;background:#fff;border-radius:8px}
.ts-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.ts-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--ts-acc);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .ts-tag{color:#10233a}
.ts-fig figcaption b.ts-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.ts-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.ts-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--ts-line);background:var(--ts-card);position:relative;overflow:hidden}
.ts-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--ts-c,var(--ts-acc))}
.ts-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;color:var(--ts-c,var(--ts-acc));font-variant-numeric:tabular-nums}
.ts-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.ts-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.ts-step{padding:.9em 1em;border-radius:14px;border:1px solid var(--ts-line);background:var(--ts-card)}
.ts-step .n{display:inline-grid;place-items:center;width:1.7em;height:1.7em;border-radius:50%;background:var(--ts-acc-soft);color:var(--ts-acc);font-weight:800;font-size:.85em;margin-bottom:.4em}
.ts-step b{display:block;margin-bottom:.25em}
.ts-step p{margin:0;font-size:.86em;line-height:1.55}
.ts-terms{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.1em 1.6em;margin:0 0 1.6em;padding:.9em 1.1em;border-radius:14px;border:1px solid var(--ts-line);background:var(--ts-card);font-size:.86em;line-height:1.55}
.ts-terms div{padding:.35em 0}
.ts-terms b{color:var(--ts-acc)}
.ts-cmp{display:grid;grid-template-columns:auto 1fr auto 1fr .9fr;gap:.55em .6em;align-items:stretch;margin:0 0 1.6em;font-size:.86em}
.ts-cmp .h{font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:0 .2em;align-self:end}
.ts-cmp .h.s{color:var(--ts-gray)}.ts-cmp .h.c,.ts-cmp .h.o{color:var(--ts-acc)}
.ts-cmp .k{font-weight:700;align-self:center;padding-right:.3em}
.ts-cmp .k i{display:inline-grid;place-items:center;width:1.5em;height:1.5em;border-radius:50%;background:var(--ts-acc);color:#fff;font-style:normal;font-size:.82em;margin-right:.4em}
html[data-theme="dark"] .ts-cmp .k i{color:#10233a}
.ts-cmp .s,.ts-cmp .c,.ts-cmp .o{padding:.55em .75em;border-radius:10px;line-height:1.4}
.ts-cmp div.s{background:var(--ts-gray-soft);border:1px solid var(--ts-line);color:var(--ts-gray)}
.ts-cmp div.c{background:var(--ts-acc-soft);border:1px solid var(--ts-acc)}
.ts-cmp div.o{border-left:3px solid var(--ts-acc);border-radius:4px}
.ts-cmp .arr{align-self:center;color:var(--ts-gray);font-size:1.1em}
.ts-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--ts-line);border-radius:14px;background:var(--ts-card);overflow-x:auto}
.ts-table table{font-size:.8em;margin:.6em 0;min-width:520px;font-variant-numeric:tabular-nums}
.ts-table td strong{color:var(--ts-acc)}
.ts-table p{font-size:.86em;margin:.2em}
.ts-cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.ts-cards.three{grid-template-columns:repeat(3,minmax(0,1fr))}
.ts-cardx{padding:.9em 1em;border-radius:14px;border:1px solid var(--ts-line);background:var(--ts-card);border-top:4px solid var(--ts-c,var(--ts-acc))}
.ts-cardx .w{font-size:.72em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--ts-c,var(--ts-acc))}
.ts-cardx b{display:block;margin:.2em 0 .3em;font-size:1.04em}
.ts-cardx p{margin:0;font-size:.86em;line-height:1.55}
.ts p.ts-note{font-size:.88em;line-height:1.55;margin:0 0 1.6em;padding:.55em .8em;border-left:3px solid var(--ts-acc);background:var(--ts-acc-soft);border-radius:4px}
.ts-limits{margin:0 0 1.6em;padding:.9em 1.1em .5em;border-radius:14px;border:1px solid var(--ts-line);background:var(--ts-card);font-size:.9em;line-height:1.6}
.ts-limits ul{margin:.2em 0 .5em 1.1em}
.ts-limits li{margin:.35em 0}
.ts-abs{margin:0 0 1em;padding:.7em 1em;border-radius:12px;border:1px solid var(--ts-line);background:var(--ts-card);font-size:.9em;line-height:1.65}
.ts-abs summary{cursor:pointer;font-weight:700}
.ts-abs p{margin:.7em 0 .2em}
.ts-seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--ts-line);border-radius:8px;overflow:hidden}
.ts .ts-seg button{font:inherit;font-size:.82em;margin:0;border:0;border-right:1px solid var(--ts-line);border-radius:0;background:transparent;color:var(--global-text-color);padding:.3em .8em;cursor:pointer}
.ts .ts-seg button:last-child{border-right:0}
.ts .ts-seg button[aria-pressed="true"]{background:var(--ts-acc);color:#fff;font-weight:700}
html[data-theme="dark"] .ts .ts-seg button[aria-pressed="true"]{color:#10233a}
.ts p.ts-read{font-size:.86em;line-height:1.55;margin:.8em .1em 0;padding:.55em .8em;border-left:3px solid var(--ts-acc);background:var(--ts-acc-soft);border-radius:4px}
/* interactive Figure 1: reduction per method */
.ts-row{display:flex;flex-wrap:wrap;align-items:center;gap:.3em .7em;font-size:.9em}
.ts-row .l{font-size:.76em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.7}
.ts-bars{position:relative;margin:2.1em 0 .2em;--ts-nw:282px;--ts-vw:46px;--ts-ol:calc(var(--ts-nw) + 10px)}
.ts-grp{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--ts-gray);margin:.75em 0 .15em}
.ts-grp:first-child{margin-top:0}
.ts-bar{display:grid;grid-template-columns:var(--ts-nw) 1fr var(--ts-vw);gap:0 10px;align-items:center;min-height:26px;font-size:.84em}
.ts-bar .nm{line-height:1.25}
.ts-bar .tr{position:relative;height:13px;border-radius:999px;background:var(--ts-gray-soft)}
.ts-bar .fl{position:absolute;left:0;top:0;bottom:0;border-radius:999px;background:var(--ts-gray);transition:width .5s cubic-bezier(.2,.8,.2,1)}
.ts-bar .vl{font-weight:700;font-variant-numeric:tabular-nums;text-align:right}
.ts-bar.k-single .fl{opacity:.4}.ts-bar.k-pair .fl{opacity:.8}
.ts-bar.k-sys .fl{background:var(--ts-amber-fill)}
.ts-bar.k-t1 .fl{background:var(--ts-acc);opacity:.45}.ts-bar.k-t2 .fl{background:var(--ts-acc);opacity:.7}.ts-bar.k-t3 .fl{background:var(--ts-acc)}
.ts-bar.k-t3 .nm,.ts-bar.k-t3 .vl{font-weight:800;color:var(--ts-acc)}
.ts-over{position:absolute;top:0;bottom:0;left:var(--ts-ol);right:calc(var(--ts-vw) + 10px);pointer-events:none}
.ts-ceil{position:absolute;top:-6px;bottom:-4px;width:0;border-left:2px dashed var(--ts-red);transition:left .5s cubic-bezier(.2,.8,.2,1)}
.ts-ceil span{position:absolute;top:-1.75em;left:0;transform:translateX(-50%);white-space:nowrap;font-size:.72em;font-weight:700;color:var(--ts-red);background:var(--global-bg-color,#fff);padding:0 .35em;border-radius:4px}
.ts-axis{display:flex;justify-content:space-between;font-size:.72em;opacity:.7;margin:.35em calc(var(--ts-vw) + 10px) 0 var(--ts-ol)}
/* interactive: BF16 residual explorer */
.ts-ctl{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55em 1.4em;margin:0 0 1em;font-size:.82em}
.ts-ctl .ts-row{font-size:1em}
.ts-ctl .full{grid-column:1/-1}
.ts-ctl .lab{font-weight:700;min-width:7.6em}
.ts-ctl input[type=range]{flex:1 1 130px;min-width:0;accent-color:var(--ts-acc);margin:0}
.ts-ctl output{min-width:5.2em;text-align:right;font-variant-numeric:tabular-nums;color:var(--ts-acc);font-weight:700}
.ts-bits{display:grid;grid-template-columns:auto auto 1fr;gap:.4em .9em;align-items:center;font-size:.84em;margin:0 0 .9em}
.ts-bits .bl{font-weight:700;white-space:nowrap}
.ts-bits .bv{font-variant-numeric:tabular-nums;opacity:.85}
.ts-bitrow{display:flex;gap:2px}
.ts .ts-bit{width:clamp(12px,3.9vw,1.5em);height:1.75em;display:grid;place-items:center;border-radius:4px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em;font-weight:600;color:var(--global-text-color);background:var(--ts-gray-soft)}
.ts .ts-bit.e{background:var(--ts-acc-soft)}
.ts .ts-bit.g{margin-left:5px}
.ts .ts-bit.flip{background:var(--ts-red);color:#fff;font-weight:800}
.ts .ts-bit.zero{opacity:.45}
.ts .ts-bit.h{height:1.2em;background:transparent;overflow:visible;white-space:nowrap;place-items:center start;font-family:inherit;opacity:.65}
.ts .ts-bit.h b{font-size:.74em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding-left:.2em}
.ts-out{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7em;margin:0}
.ts-out>div{padding:.7em .9em;border-radius:10px;border:1px solid var(--ts-line);font-size:.84em;line-height:1.55}
.ts-out>div.x{border-color:var(--ts-red);background:var(--ts-red-soft)}
.ts-out>div.o{border-color:var(--ts-acc);background:var(--ts-acc-soft)}
.ts-out .w{display:block;font-size:.78em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;margin-bottom:.2em}
.ts-out .x .w{color:var(--ts-red)}.ts-out .o .w{color:var(--ts-acc)}
.ts-out b.big{font-size:1.35em;font-variant-numeric:tabular-nums}
.ts-out .mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.95em;font-weight:700}
@media (max-width:900px){.ts-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.ts-steps,.ts-cards.three{grid-template-columns:1fr}}
@media (max-width:640px){.ts-terms,.ts-cards,.ts-ctl,.ts-out{grid-template-columns:1fr}.ts-cmp{grid-template-columns:1fr}.ts-cmp .arr,.ts-cmp .h,.ts-cmp span:empty{display:none}.ts-cmp .k{margin-top:.6em}.ts-tldr{font-size:1.04em}.ts-bars{--ts-nw:0px;--ts-ol:0px}.ts-bar{grid-template-columns:1fr var(--ts-vw);min-height:0;margin-bottom:.35em}.ts-bar .nm{grid-column:1/-1}.ts-bits{grid-template-columns:1fr;gap:.15em}.ts-bits .bv{margin-bottom:.5em}.ts-bits .sp{display:none}.ts-ctl .lab{min-width:0;flex:1 1 100%}}
</style>

<div class="ts">
<div class="ts-kicker"><span>Model hubs</span><span>Lossless compression</span><span>Delta storage</span><span>Entropy coding</span></div>
<p class="ts-tldr">Model hubs store millions of fine-tuned checkpoints that are near-copies of a few base models. Lossless stores treat this as a <i>pairing</i> problem: which stored tensor should each new tensor be delta-encoded against? Exhaustive ground truth on 130 checkpoints shows that pairing is nearly solved and still leaves most of the redundancy behind. <b>Tessera</b> predicts and codes the residual from the base instead, and reduces storage by <b>74.7%</b>, bit for bit.</p>
<div class="ts-btns"><a class="ts-btn" href="#explorer">▶&nbsp; Try the residual explorer</a><a class="ts-btn ghost" href="#pairing">How much pairing matters</a><a class="ts-btn ghost" href="#how">How it works</a><a class="ts-btn ghost" href="#results">Results</a><span class="ts-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="ts-fig" id="ts-hero">
<div class="ts-row"><span class="l">Model family</span><div class="ts-seg" id="ts-fams"></div></div>
<div class="ts-bars" id="ts-bars"></div>
<div class="ts-axis"><span>0%</span><span>reduction after deduplication (higher is better)</span><span>90%</span></div>
<p class="ts-read" id="ts-hero-read" aria-live="polite"></p>
<noscript><img src="/images/projects/2026-11-tessera-lossless-storage/teaser.png" alt="Figure 1: pooled reduction after deduplication over the four families"></noscript>
<figcaption><span class="ts-tag">Figure 1 · interactive</span><b class="ts-title">Pairing alone stops near 60%; predicting and coding the residual reaches 74.7%.</b>Lossless reduction after content-hash deduplication, from Table 3 of the paper; pick a family. Gray: single-model coding and pairing with the XOR/zstd codec of prior systems. Amber: existing systems and codecs, run with their authors' code. Blue: Tessera's techniques, added one at a time on the optimal one-hop pairs. The dashed line is the exact optimum of pairing with XOR/zstd when chains may be of any length: no choice of bases gets past it without changing the codec. Every bar is a real encoded size, decoded and checked against content hashes.</figcaption>
</figure>
</div>

## At a glance

<div class="ts"><div class="ts-stats">
<div class="ts-stat"><div class="v">74.7%</div><div class="d">lossless reduction over 130 fine-tunes of four model families: 206.3 GB stored instead of 816.9 GB, every tensor decoded and hash-checked</div></div>
<div class="ts-stat" style="--ts-c:var(--ts-red)"><div class="v">60.6%</div><div class="d">the most that pairing can reach with the XOR/zstd codec of prior systems, with exact optima and chains of any length</div></div>
<div class="ts-stat"><div class="v">0.2–1.4 pts</div><div class="d">between the online store, which plans heuristically as models arrive, and the exact offline plan</div></div>
<div class="ts-stat" style="--ts-c:var(--ts-green)"><div class="v">76.4–80.9%</div><div class="d">online, on random samples of 112 and 303 hub models from eight bases (median fine-tune: 10–14 downloads)</div></div>
</div></div>

## Background: a hub is a versioned store {#background}

<div class="ts">
<div class="ts-steps">
<div class="ts-step"><span class="n">1</span><b>Hubs fill up with near-copies.</b><p>Hugging Face's storage grew from 0.28 PB in 2023 to over 76 PB by the end of 2025, as TensorDex reports. A few base models spawn thousands of fine-tunes, and each one is uploaded as a full copy of the weights although it differs only slightly from its relatives.</p></div>
<div class="ts-step"><span class="n">2</span><b>Delta systems store the difference.</b><p>After exact duplicates are removed by content hash, a tensor is stored as a residual against a similar stored tensor, and the residual is compressed. Coding one model alone saves about a third of the bytes. Deltas roughly double that.</p></div>
<div class="ts-step"><span class="n">3</span><b>The pairing view.</b><p>Recent systems put their effort into choosing the base. TensorDex sketches every tensor, predicts how well each pair compresses and clusters tensors online. Once a tensor has its base, the residual goes to a generic coder such as zstd.</p></div>
</div>
<p class="ts-lead">This work asks how much pairing matters: how far practical planners are from the best possible pairing, and how much redundancy the best pairing still leaves. The terms used on this page:</p>
<div class="ts-terms">
<div><b>BF16.</b> The 16-bit float format of most checkpoints: 1 sign bit, 8 exponent bits, 7 mantissa bits.</div>
<div><b>Tensor group.</b> The tensors with the same name, type and shape across the fine-tunes of one base. A tensor is only encoded against a member of its own group.</div>
<div><b>Base, delta.</b> A base is stored standalone. A delta is stored as a residual against its parent, which must be decoded first.</div>
<div><b>Reduction.</b> 1 − stored / raw, with raw measured after content-hash deduplication. Higher is better.</div>
<div><b>Hop.</b> One delta on the path from a tensor to its base. In a one-hop plan every delta points directly at a base.</div>
<div><b>One-hop optimum.</b> The smallest storage any one-hop plan can reach for a given codec, solved exactly as an integer program per tensor group.</div>
<div><b>Read amplification.</b> Bytes read to rebuild a model over its raw size, weighted by downloads. Below 1, fewer bytes are read than the model holds.</div>
<div><b>XOR/zstd.</b> Bitwise XOR against the base, split into byte planes and compressed with zstd: the codec of TensorDex's TensorX.</div>
<div><b>FlexSplit.</b> TensorDex's online planner. It attaches each new tensor to the best existing base and periodically splits clusters.</div>
<div><b>LoRA merge.</b> A fine-tune trained as a low-rank adapter and merged into the weights before upload. It differs from its parent by a matrix of rank 4–32.</div>
</div>
</div>

## How much does pairing matter? {#pairing}

<div class="ts">
<p class="ts-lead">Evaluations of delta systems usually measure a planner end to end, which cannot tell a bad plan from an inherent limit. We instead compress <b>every</b> ordered pair of same-group tensors with the codec under study, over a million directed pairs, and solve base selection exactly. The size of any plan is then an exact sum of real encoded sizes. The corpus has four families: Qwen2.5-3B (40 checkpoints), Llama-3.2-3B (30), Gemma-2-2B (30) and Qwen2.5-7B (30), 1,053 GB raw and 816.9 GB after deduplication.</p>
<div class="ts-table" markdown="1">

<span class="ts-tag">Table 2</span> **Planners on the Qwen2.5-3B family with the XOR/zstd codec.** Reduction after deduplication (187.5 GB); all numbers are real encoded sizes.

| Planner | Reduction |
|---|---|
| Standalone coding (no deltas) | 28.6% |
| Every model against the pretrained base | 52.6% |
| FlexSplit with sketch-based prediction (our re-implementation) | 56.5% |
| FlexSplit with exact costs | 57.4% |
| Declared `base_model` metadata (multi-hop) | 62.4% |
| Model-level one-hop optimum | 62.4% |
| **FlexSplit, all members as split candidates** | **62.8%** |
| Tensor-level one-hop optimum | 63.0% |
| Greedy multi-hop chains (depth ≤ 6) | 64.7% |

Read it from top to bottom. Deltas more than double the savings of standalone coding, and a planner that may promote any member to a base is 0.2 points from the exact optimum. Deeper chains add less than two points.

</div>
<div class="ts-cards">
<div class="ts-cardx" style="--ts-c:var(--ts-gray)"><span class="w">Finding 1</span><b>Deltas and the choice of base matter.</b><p>Delta coding lifts the Qwen family from 28.6% to 63.0%. Encoding every model against the pretrained base reaches only 52.6%, because many fine-tunes descend from the Instruct model instead.</p></div>
<div class="ts-cardx"><span class="w">Finding 2</span><b>One filter keeps FlexSplit from the optimum.</b><p>FlexSplit, re-implemented as its paper describes it, trails the one-hop optimum by 6.5 points on Qwen and 16.9 on Llama. Its split phase only considers members that fall below the cluster average, so an Instruct model whose fine-tunes sit <i>at</i> the average is never promoted. Letting every member be a candidate closes the gap to 0.2 points in the three smaller families and 1.0 on Qwen-7B, without tuning.</p></div>
<div class="ts-cardx" style="--ts-c:var(--ts-gray)"><span class="w">Finding 3</span><b>Tensor granularity adds little within a family.</b><p>Choosing a base per tensor beats choosing one base model per model by 0.3–1.0 points. For the average Qwen model, 83% of its tensors have their best base in one single other model.</p></div>
<div class="ts-cardx" style="--ts-c:var(--ts-gray)"><span class="w">Finding 5</span><b>Pairing barely depends on the codec.</b><p>The pairs chosen under XOR/zstd stay within 0.06 points of the pairs re-optimized for Tessera's codec. The best partners stay the same; only their costs change.</p></div>
</div>
<figure class="ts-fig narrow"><a class="zoom" href="/images/projects/2026-11-tessera-lossless-storage/figA3-planners.png"><img src="/images/projects/2026-11-tessera-lossless-storage/figA3-planners.png" alt="Figure A3: cumulative reduction of seven planners as Qwen models arrive in creation order" loading="lazy"></a><figcaption><span class="ts-tag">Figure A3</span><b class="ts-title">The fixed planner tracks the optimum as models arrive.</b>Cumulative reduction as the 40 Qwen models arrive in creation order (XOR/zstd codec). FlexSplit as described (orange) falls behind once the Instruct sub-family forms and never recovers. With every member a split candidate (blue) it follows the tensor-level optimum (black) throughout.</figcaption></figure>
<p class="ts-note"><b>What this implies.</b> With the XOR/zstd codec, no one-hop planner can exceed 63.0% (Qwen), 61.6% (Llama), 56.2% (Gemma) or 57.8% (Qwen-7B), and the fixed FlexSplit is already within 0.2–1.0 points of that. The remaining 37–44% of the bytes are residuals of well-chosen pairs. Shrinking them means changing how a residual is <b>predicted</b> and <b>coded</b>.</p>
</div>

## How Tessera works {#how}

<div class="ts">
<p class="ts-lead">Tessera keeps the tensor groups and the one-hop default of FlexSplit and changes four things. The numbers match the blue stages of Figure 2.</p>
<div class="ts-cmp">
<span></span><span class="h s">Prior delta stores</span><span></span><span class="h c">Tessera</span><span class="h o">What it buys</span>
<span class="k"><i>1</i>Choosing a base</span><div class="s">TensorDex: a sketch of each tensor and a fitted regression predict the pair's ratio; only members below the cluster average may become bases</div><span class="arr">→</span><div class="c">entropy of the codec's own symbols on a shared row sample (at most 32K elements); every member may become a base</div><div class="o">10–12× lower regret in base choice than sketch + regression, with no fit to redo when the codec changes</div>
<span class="k"><i>2</i>Coding the residual</span><div class="s">XOR or integer difference, then a generic coder that ignores the base</div><span class="arr">→</span><div class="c">order-preserving integer residual, rANS-coded with one frequency table per exponent of the base weight</div><div class="o">+10.0 to +11.3 points in every family</div>
<span class="k"><i>3</i>Predicting the tensor</span><div class="s">the parent itself</div><span class="arr">→</span><div class="c">for merged LoRA fine-tunes, the parent plus a low-rank update that the decoder recomputes bit for bit</div><div class="o">+2.5 to +6.6 points, following how common LoRA merges are</div>
<span class="k"><i>4</i>Chain depth</span><div class="s">one hop for every model, as in FlexSplit</div><span class="arr">→</span><div class="c">one hop for the more downloaded half, two for the rest, planned exactly per group</div><div class="o">up to +1.2 points at nearly the same download-weighted read cost</div>
</div>
<figure class="ts-fig"><a class="zoom" href="/images/projects/2026-11-tessera-lossless-storage/fig2-overview.png"><img src="/images/projects/2026-11-tessera-lossless-storage/fig2-overview.png" alt="Figure 2: how Tessera stores one tensor, and the store with ingestion, compaction and retrieval" loading="lazy"></a><figcaption><span class="ts-tag">Figure 2</span><b class="ts-title">Storing one tensor, and the store around it.</b>(a) A tensor <i>t</i> is coded against its decoded parent <i>b</i>. If a merged low-rank update is detected, <i>t</i> is predicted as <i>P</i> = bf16(<i>b</i> + <i>US·V</i>), otherwise <i>P</i> = <i>b</i>. The order-preserving integer residual against <i>P</i> is split into a low byte and a high part, and each is rANS-coded with one frequency table per exponent of <i>P</i>. (b) A model is ingested in five phases. Each tensor group is a tree of blobs: tensors of hot models sit at depth at most 1, those of cold models at depth at most 2. Compaction re-plans a group and commits only if the group shrinks; retrieval decodes level by level on a CPU or GPU and verifies every tensor.</figcaption></figure>
<div class="ts-cards three">
<div class="ts-cardx"><span class="w">Codec</span><b>The base is free context.</b><p>Fine-tuning moves a weight by an amount that scales with the weight itself, so the exponent of the base weight predicts how large the residual is. The decoder rebuilds the base first anyway, so conditioning on it costs no stored bytes. Richer contexts (mantissa bits, neighbouring residuals, adaptive tables) add at most 0.03 points.</p></div>
<div class="ts-cardx"><span class="w">Prediction</span><b>Predict LoRA merges, bit for bit.</b><p>A merged LoRA differs from its parent by a dense rank-<i>r</i> matrix: structured, but not small. Tessera detects it from 256 sampled rows, stores the factors and codes what is left. Losslessness now depends on the decoder recomputing the prediction exactly, so its kernel fixes the order of fp32 operations, without fused multiply-adds, and matches a scalar IEEE reference on x86-64, arm64 and a GPU.</p></div>
<div class="ts-cardx"><span class="w">Hops</span><b>Let cold models chain twice.</b><p>In the Qwen corpus the top 20% of models receive 99.7% of the downloads. Models in the more downloaded half stay one hop from a base; the rest may sit two deltas deep. Unbounded chains add 0.01 points beyond two hops and read 1.5× as many bytes.</p></div>
</div>
<p class="ts-lead">A hub receives models one at a time, so the store plans online. Each arriving tensor compares standalone coding with every stored member that its hop limit allows and takes the cheapest option. Compaction later re-plans the base set of a group:</p>
<figure class="ts-fig"><a class="zoom" href="/images/projects/2026-11-tessera-lossless-storage/fig3-parent-choice.png"><img src="/images/projects/2026-11-tessera-lossless-storage/fig3-parent-choice.png" alt="Figure 3: parent choice in one tensor group as five models arrive, followed by one compaction" loading="lazy"></a><figcaption><span class="ts-tag">Figure 3</span><b class="ts-title">Parent choice in one tensor group, then one compaction.</b>Illustrative costs: the numbers are estimated sizes in % of the raw tensor. (1) The first tensor becomes a base. (2) A delta on <i>m</i><sub>1</sub> is cheaper than standalone. (3) <i>m</i><sub>3</sub> belongs to a hot model and must stay within one hop, so its cheapest candidate <i>m</i><sub>2</sub>, already at depth 1, is not offered. (4) <i>m</i><sub>4</sub> is cold and takes <i>m</i><sub>2</sub> at depth 2. (5) No delta beats standalone coding, so <i>m</i><sub>5</sub> opens a new base. (6) Compaction makes <i>m</i><sub>2</sub> a base and re-encodes only <i>m</i><sub>1</sub> and <i>m</i><sub>3</sub>; the group is committed only if its real encoded size shrinks.</figcaption></figure>
</div>

## Try the residual explorer {#explorer}

<div class="ts">
<p class="ts-lead">Prior delta stores take the bitwise XOR of a fine-tuned weight and its base. Tessera maps both BF16 values to integers that keep the order of the floats and stores their difference. The figure below computes both residuals for one weight in your browser, with exact BF16 arithmetic. It illustrates the codec's first stage; it is not a measurement from the paper.</p>
<figure class="ts-fig" id="ts-demo">
<div class="ts-ctl">
<div class="ts-row full"><span class="lab">Example</span><div class="ts-seg" id="ts-presets"><button type="button" data-b="0.0200" data-d="1.5">Small update</button><button type="button" data-b="0.0311" data-d="1.0">Across an exponent boundary</button><button type="button" data-b="0.0200" data-d="8.0">Larger update</button><button type="button" data-b="-0.0620" data-d="1.0">Negative weight</button></div></div>
<div class="ts-row"><label class="lab" for="ts-b">Base weight <i>b</i></label><input type="range" id="ts-b" min="-0.08" max="0.08" step="0.0001"><output id="ts-b-o"></output></div>
<div class="ts-row"><label class="lab" for="ts-d">Fine-tuning change</label><input type="range" id="ts-d" min="-10" max="10" step="0.1"><output id="ts-d-o"></output></div>
</div>
<div class="ts-bits" id="ts-bits"></div>
<div class="ts-out" id="ts-out"></div>
<p class="ts-read" id="ts-read" aria-live="polite"></p>
<figcaption><span class="ts-tag">Interactive</span><b class="ts-title">One weight, two residuals.</b>The three rows show the 16 bits of the base weight, of the fine-tuned weight and of their XOR, with flipped bits in red. When the fine-tuned value crosses a power of two, XOR flips most of the mantissa and part of the exponent although the value has moved by one or two representable steps. The integer residual <i>d</i> = <i>o</i>(<i>t</i>) − <i>o</i>(<i>b</i>) counts exactly those steps. Tessera then codes <i>d</i> with the frequency table of the base weight's exponent, which the decoder knows before it reads a byte.</figcaption>
</figure>
<div class="ts-table" markdown="1">

<span class="ts-tag">Table A1</span> **What each part is worth.** Points of reduction lost when one component is removed from Tessera, with real encoded sizes; the codec and low-rank rows keep the plan fixed, which is an upper bound on their cost.

| Removed | Qwen2.5-3B | Llama-3.2-3B | Gemma-2-2B | Qwen2.5-7B | Pooled |
|---|---|---|---|---|---|
| Exponent context | −3.0 | −4.0 | −3.8 | −3.3 | −3.4 |
| Integer residual (XOR instead) | −1.5 | −1.5 | −1.4 | −2.7 | −2.0 |
| rANS (zstd instead) | −6.4 | −8.1 | −7.6 | −6.3 | −6.8 |
| **Whole codec (XOR/zstd instead)** | **−8.6** | **−10.5** | **−10.0** | **−9.7** | **−9.6** |
| Low-rank prediction | −6.3 | −3.7 | −4.8 | −2.6 | −3.9 |
| Second hop | −1.2 | −0.9 | 0.0 | −1.1 | −0.9 |

The codec is the largest and most even contribution. Low-rank prediction follows the share of LoRA merges in a family, and the second hop helps where lineages are deep; the Gemma family is flat, so it gains nothing there.

</div>
</div>

## Results {#results}

<div class="ts">
<p class="ts-lead">Every storage number below comes from real encodes: each tensor is encoded as planned, decoded root-first along its chain, and compared with its content hash. Baselines run their authors' code and are round-trip checked as well.</p>
</div>

<div class="ts"><div class="ts-table" markdown="1">

<span class="ts-tag">Table 3</span> **Reduction (%) after deduplication, per family and pooled.** An excerpt; the first figure on this page shows more rows.

| Method | Qwen2.5-3B | Llama-3.2-3B | Gemma-2-2B | Qwen2.5-7B | Pooled |
|---|---|---|---|---|---|
| ZipNN (single model) | 33.1 | 33.6 | 33.6 | 33.3 | 33.4 |
| One-hop pairing optimum, XOR/zstd | 63.0 | 61.6 | 56.2 | 57.8 | 59.5 |
| TensorDex released code (bundle planner, FM++) | 61.7 | 61.8 | 60.6 | 58.3 | 60.0 |
| ZipLLM (declared bases, BitX) | 66.0 | 60.8 | 60.8 | 64.4 | 63.7 |
| Optimal pairs, FM-Delta codec | 69.1 | 68.2 | 62.6 | 64.0 | 65.7 |
| **Tessera** | **80.9** | **77.3** | **72.2** | **71.4** | **74.7** |

Pooled, Tessera stores 206.3 of 816.9 GB, TensorDex's released code 326.7 GB and the exact one-hop XOR/zstd optimum 331.2 GB. The strongest existing codec on the optimal pairs, FM-Delta, is 9.0 points below Tessera. The gains are not carried by a few models: the median fine-tune shrinks by 78.3–92.4% per family, against 58.8–65.1% under the optimal XOR/zstd pairing.

</div></div>

<div class="ts">
<p class="ts-lead"><b>Storage versus read cost.</b> Deeper chains save bytes but make a download read more blobs. This is the recreation/storage tradeoff of dataset versioning, and because tensor groups are small, Tessera can solve it exactly instead of heuristically.</p>
<div class="ts-table" markdown="1">

<span class="ts-tag">Table 4</span> **Hop configurations on Qwen2.5-3B** (codec + low-rank, exact plans). Read amplification and blobs per tensor are weighted by downloads.

| Configuration | Reduction | Read amplification | Blobs per tensor |
|---|---|---|---|
| Hot models pinned as bases | 78.70% | 0.662 | 1.00 |
| All one-hop | 79.72% | 0.683 | 1.28 |
| **Top 50% one-hop, rest two-hop (default)** | **80.93%** | **0.686** | **1.25** |
| All two-hop | 81.08% | 0.737 | 1.55 |
| Unbounded chains | 81.09% | 1.096 | 3.93 |

The default saves 1.2 more points than the one-hop design at nearly the same read cost. Unbounded chains add 0.01 points beyond two hops and read 1.5× as many bytes.

</div>
<figure class="ts-fig"><a class="zoom" href="/images/projects/2026-11-tessera-lossless-storage/fig4-storage-vs-read-cost.png"><img src="/images/projects/2026-11-tessera-lossless-storage/fig4-storage-vs-read-cost.png" alt="Figure 4: reduction against read amplification for Tessera's hop-bounded plans and three dataset-versioning heuristics" loading="lazy"></a><figcaption><span class="ts-tag">Figure 4</span><b class="ts-title">Exact hop-bounded plans against dataset-versioning heuristics.</b>Each panel plots reduction against read amplification on the same costs (codec + low-rank); up and to the left is better. Lines are the best plan of each heuristic (LMG, LMG-All, MP) at or below a read cost. At the read cost of Tessera's default plan (dark dot), the best LMG or LMG-All plan saves 2.4, 3.7, 0.9 and 1.1 fewer points on Qwen, Llama, Gemma and Qwen-7B, with chains 5 to 11 deep instead of 2.</figcaption></figure>
<p class="ts-lead"><b>Online operation.</b> The offline results assume that every model is known in advance. We also replayed each family through the store in creation order, with sampled-cost parent choice, the low-rank path and compaction every 10 models, and retrieved and hash-verified every model at the end.</p>
<figure class="ts-fig narrow"><a class="zoom" href="/images/projects/2026-11-tessera-lossless-storage/fig5-online-replay.png"><img src="/images/projects/2026-11-tessera-lossless-storage/fig5-online-replay.png" alt="Figure 5: reduction of the online store as models arrive, for three families" loading="lazy"></a><figcaption><span class="ts-tag">Figure 5</span><b class="ts-title">Compaction repairs the early choices.</b>Reduction of the online store as models arrive; dotted lines are the offline optimum with the same techniques. A model can only attach to models that arrived before it, so the first ones are placed poorly. On Llama the store is at 49.5% after 10 models, and the first compaction lifts it to 60.0%. Later rounds add at most 0.7 points.</figcaption></figure>
<div class="ts-table" markdown="1">

<span class="ts-tag">Table 6</span> **Online replay in creation order against the offline optimum with the same techniques.**

| | Qwen2.5-3B | Llama-3.2-3B | Gemma-2-2B |
|---|---|---|---|
| Offline optimum (hot 1 hop, cold 2 hops) | 80.89% | 77.30% | 72.20% |
| **Online** | **80.68%** | **75.86%** | **71.45%** |
| Gap (points) | 0.20 | 1.44 | 0.75 |
| Compaction, all rounds | 52 s | 65 s | 11 s |
| Rolled back / attempted groups | 21 / 698 | 0 / 413 | 1 / 190 |

The online layout is already optimal in its own cost model. The remaining gap comes from the compaction schedule and from costs that estimate plain deltas, while the offline plan knows the real low-rank sizes. Compacting Llama every 3 models instead of 10 ends at 77.25%.

</div>
<div class="ts-stats">
<div class="ts-stat"><div class="v">1.7 GB/s</div><div class="d">ingestion of the 40 Qwen models (145 s), 14× the serial path, with the same parents and byte-identical blobs</div></div>
<div class="ts-stat"><div class="v">1.55–3.72 s</div><div class="d">to rebuild a 6.17 GB model on CPU from a cold local disk, every tensor verified</div></div>
<div class="ts-stat"><div class="v">0.13–0.20 s</div><div class="d">to decode a model's whole closure on one H200, including the upload of its compressed blobs; disk reads and hash checks excluded</div></div>
<div class="ts-stat" style="--ts-c:var(--ts-green)"><div class="v">33 / 33</div><div class="d">kills at random times during ingestion and compaction recovered, in at most 0.29 s, with exactly the committed blobs</div></div>
</div>
<p class="ts-note"><b>A random, heterogeneous sample.</b> The four families favour popular fine-tunes. On 112 models drawn uniformly from the hub fine-tunes of eight bases, regardless of downloads, Tessera ingested online stores 92.7 of 392 GB, a 76.4% reduction, against 65.1% for ZipLLM and 63.1% for TensorDex's released code with its bundle planner and FM++. A second draw brings the sample to 303 models and the reduction to 80.9%: more fine-tunes per base give each group more candidate parents.</p>
</div>

## What Tessera does not do {#limits}

<div class="ts"><div class="ts-limits">
<ul>
<li><b>It stores BF16 checkpoints only.</b> Of 2,000 fine-tunes sampled from those the hub lists for eight bases, 31.6% are pure BF16 at the base's size, which is 57.3% of the bytes. FP16 would need another context map and FP32 a wider residual split; neither is implemented. Entropy estimates on 58 sampled FP16 fine-tunes give 70.0% against 62.2% for XOR/zstd.</li>
<li><b>A model without relatives gains little.</b> A new base, or a fine-tune that changes the vocabulary, gets only the third that single-model coding saves. Low-rank prediction helps only merged adapters, and hop limits only skewed downloads.</li>
<li><b>It trades CPU for bytes.</b> Tessera encodes 3–6× slower than FM-Delta and ZipNN, and its CPU decoder does not beat a fast network file system. At list prices, ingesting and compacting the Qwen corpus pays for itself in 13–21 days of storage savings. A CPU decode, however, costs as much as storing a decoded copy for a month after 17–41 downloads, so popular models should be served from a decoded cache.</li>
<li><b>The exact planner is an evaluation tool.</b> The integer programs solve our groups in at most 0.34 s each but do not scale to hub-sized groups: the one-hop program does not prove optimality within 30 minutes at 640 versions. The store itself plans with local search on sampled costs.</li>
<li><b>The ground truth covers four families.</b> All-pairs costs are affordable only for 30–40 fine-tunes per family, here of 2–7B-parameter models. The XOR/zstd ground truth of the 112-model random sample reproduces Findings 2 and 3.</li>
<li><b>Bit-exact prediction is tested on three platforms.</b> x86-64 and arm64 CPUs and one GPU model. A scalar IEEE reference defines the format and every tensor is hash-checked, so a deviating kernel costs speed, not data.</li>
</ul>
</div></div>

## Paper and code

<div class="ts">
<details class="ts-abs"><summary>Abstract</summary><p>Model hubs store millions of near-identical fine-tuned checkpoints. Lossless systems for them treat storage mainly as a <i>pairing</i> problem: which stored tensor should each new tensor be delta-encoded against? We revisit this view with exhaustive ground truth. For 130 checkpoints of four model families (817 GB after deduplication), we compress every candidate pair of same-named tensors and solve base selection exactly. Near-optimal pairing turns out to be easy to reach: letting every member be a split candidate in FlexSplit, the online planner described for TensorDex, brings it within 0.2–1.0 points of the one-hop optimum without tuning. Yet even optimal pairing with the XOR/zstd codec of prior systems leaves most of the redundancy unused. We present Tessera, a lossless tensor store that (1) entropy-codes each residual conditioned on the exponent of its base weight, (2) predicts LoRA-merged fine-tunes with a low-rank update whose decoder is bit-reproducible across machines, and (3) lets rarely downloaded models chain two deltas deep. Planned offline, Tessera reduces storage by 74.7%, against 65.7% for the optimal pairs coded with FM-Delta, 60.6% for optimal XOR/zstd pairing with chains of any length, 60.0% for TensorDex's released code, and 63.7% for ZipLLM. At equal download-weighted read cost, its exact plans save 0.9–3.7 more points than dataset-versioning heuristics. Its online store, which plans heuristically, ends within 0.2–1.4 points of the offline plan and ingests about 1.7 GB/s; a GPU decodes a 6 GB model from pinned host memory in 0.13–0.20 s, excluding disk reads and hash checks.</p></details>
</div>

The manuscript is under review. Paper, code and BibTeX will be linked here once the preprint is public.

{% raw %}
<script>
(function () {
  function $(id) { return document.getElementById(id); }

  /* ---------- Figure 1: reduction per method and family (Table 3 of the paper) ---------- */
  var FAMS = ["Pooled", "Qwen2.5-3B", "Llama-3.2-3B", "Gemma-2-2B", "Qwen2.5-7B"];
  /* values in the order of FAMS */
  var ROWS = [
    { g: "Single-model coding" },
    { n: "ZipNN", k: "single", v: [33.4, 33.1, 33.6, 33.6, 33.3] },
    { g: "Pairing with the XOR/zstd codec" },
    { n: "FlexSplit (our re-implementation)", k: "pair", v: [52.4, 56.5, 44.7, 56.2, 52.2] },
    { n: "Exact one-hop optimum", k: "pair", v: [59.5, 63.0, 61.6, 56.2, 57.8] },
    { g: "Existing systems and codecs (authors' code)" },
    { n: "TensorDex released code (bundle, FM++)", k: "sys", v: [60.0, 61.7, 61.8, 60.6, 58.3] },
    { n: "ZipLLM (declared bases, BitX)", k: "sys", v: [63.7, 66.0, 60.8, 60.8, 64.4] },
    { n: "Optimal pairs + FM-Delta codec", k: "sys", v: [65.7, 69.1, 68.2, 62.6, 64.0] },
    { g: "Tessera, one technique at a time on the optimal pairs" },
    { n: "+ base-conditioned codec", k: "t1", v: [69.9, 73.1, 72.9, 67.6, 67.8] },
    { n: "+ low-rank prediction", k: "t2", v: [73.8, 79.7, 76.4, 72.2, 70.3] },
    { n: "+ hot 1-hop / cold 2-hop = Tessera", k: "t3", v: [74.7, 80.9, 77.3, 72.2, 71.4] }
  ];
  /* exact optimum of XOR/zstd pairing with chains of any length; deduplicated raw and Tessera's stored size (GB) */
  var CEIL = [60.6, 64.7, 62.8, 56.8, 58.9];
  var RAW = [816.9, 187.5, 145.8, 108.9, 374.8], STORED = [206.3, 35.8, 33.1, 30.3, 107.1];
  var XMAX = 90, fam = 0;
  var bars = $("ts-bars");
  if (bars) {
    var over = document.createElement("div"); over.className = "ts-over";
    over.innerHTML = '<div class="ts-ceil"><span></span></div>';
    ROWS.forEach(function (r) {
      var e = document.createElement("div");
      if (r.g) { e.className = "ts-grp"; e.textContent = r.g; }
      else { e.className = "ts-bar k-" + r.k; e.innerHTML = '<span class="nm"></span><span class="tr"><span class="fl"></span></span><span class="vl"></span>'; e.querySelector(".nm").textContent = r.n; r.el = e; }
      bars.appendChild(e);
    });
    bars.appendChild(over);
    var seg = $("ts-fams");
    FAMS.forEach(function (f, i) {
      var b = document.createElement("button"); b.type = "button"; b.textContent = f;
      b.onclick = function () { fam = i; drawBars(); }; seg.appendChild(b);
    });
    function drawBars() {
      seg.querySelectorAll("button").forEach(function (b, i) { b.setAttribute("aria-pressed", i === fam); });
      ROWS.forEach(function (r) {
        if (!r.el) return;
        r.el.querySelector(".fl").style.width = (100 * r.v[fam] / XMAX) + "%";
        r.el.querySelector(".vl").textContent = r.v[fam].toFixed(1) + "%";
      });
      var c = over.querySelector(".ts-ceil"); c.style.left = (100 * CEIL[fam] / XMAX) + "%";
      c.querySelector("span").textContent = "pairing ceiling " + CEIL[fam].toFixed(1) + "%";
      var who = fam === 0 ? "Pooled over the four families" : "On the " + FAMS[fam] + " family";
      var t = ROWS[ROWS.length - 1].v[fam], fm = ROWS[8].v[fam];
      $("ts-hero-read").innerHTML = "<b>" + who + ":</b> no pairing with the XOR/zstd codec gets past " + CEIL[fam].toFixed(1) +
        "%, and the best existing codec on the optimal pairs reaches " + fm.toFixed(1) + "%. Tessera reaches <b>" + t.toFixed(1) +
        "%</b>: it stores " + STORED[fam].toFixed(1) + " of " + RAW[fam].toFixed(1) + " GB." +
        (fam === 3 ? " This family is flat (nearly every fine-tune derives directly from one model), so the second hop adds nothing." : "");
    }
    drawBars();
  }

  /* ---------- BF16 residual explorer ---------- */
  var demo = $("ts-demo");
  if (demo) {
    var f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
    /* float -> BF16 bit pattern, round to nearest even on the upper 16 bits of the float32 */
    function toBf16(x) { f32[0] = x; var b = u32[0]; return ((b + 0x7FFF + ((b >>> 16) & 1)) >>> 16) & 0xFFFF; }
    function fromBf16(v) { u32[0] = (v << 16) >>> 0; return f32[0]; }
    /* order-preserving integer map of the codec: a bijection on all 2^16 patterns */
    function ord(v) { return (v & 0x8000) ? -(v & 0x7FFF) - 1 : v; }
    function expo(v) { return (v >> 7) & 0xFF; }
    function pop(v) { var n = 0; while (v) { n += v & 1; v >>>= 1; } return n; }
    function hex(v, w) { var s = v.toString(16).toUpperCase(); while (s.length < w) s = "0" + s; return "0x" + s; }
    function fmt(x) { return x === 0 ? "0" : (Math.abs(x) < 0.001 ? x.toExponential(3) : x.toPrecision(4)); }
    function bitRow(v, mask) {
      var h = '<div class="ts-bitrow">';
      for (var i = 15; i >= 0; i--) {
        var bit = (v >> i) & 1, cls = "ts-bit" + (i < 15 && i >= 7 ? " e" : "") + (i === 14 || i === 6 ? " g" : "");
        if (mask !== undefined) cls += bit ? " flip" : " zero";
        h += '<span class="' + cls + '">' + bit + "</span>";
      }
      return h + "</div>";
    }
    function headRow() {
      var h = '<div class="ts-bitrow h" aria-hidden="true">', lab = { 15: "s", 14: "exponent", 6: "mantissa" };
      for (var i = 15; i >= 0; i--) h += '<span class="ts-bit h' + (i === 14 || i === 6 ? " g" : "") + '">' + (lab[i] ? "<b>" + lab[i] + "</b>" : "") + "</span>";
      return h + "</div>";
    }
    var sb = $("ts-b"), sd = $("ts-d");
    function draw() {
      var bx = parseFloat(sb.value), dp = parseFloat(sd.value);
      var b = toBf16(bx), t = toBf16(bx * (1 + dp / 100)), x = b ^ t;
      var d = ord(t) - ord(b), z = d >= 0 ? 2 * d : -2 * d - 1, n = pop(x);
      $("ts-b-o").textContent = bx.toFixed(4);
      $("ts-d-o").textContent = (dp > 0 ? "+" : "") + dp.toFixed(1) + "%";
      $("ts-bits").innerHTML =
        '<span class="sp"></span>' + headRow() + '<span class="sp"></span>' +
        '<span class="bl">base <i>b</i></span>' + bitRow(b) + '<span class="bv">' + fmt(fromBf16(b)) + " · exponent " + expo(b) + "</span>" +
        '<span class="bl">fine-tuned <i>t</i></span>' + bitRow(t) + '<span class="bv">' + fmt(fromBf16(t)) + " · exponent " + expo(t) + "</span>" +
        '<span class="bl">XOR <i>t</i> ⊕ <i>b</i></span>' + bitRow(x, 1) + '<span class="bv">' + n + (n === 1 ? " bit" : " bits") + " flipped</span>";
      $("ts-out").innerHTML =
        '<div class="x"><span class="w">XOR residual (prior stores)</span><b class="big">' + hex(x, 4) + "</b><br>byte planes: high <span class='mono'>" + hex(x >> 8, 2) + "</span>, low <span class='mono'>" + hex(x & 0xFF, 2) +
        "</span><br>then compressed by a generic coder, without looking at the base</div>" +
        '<div class="o"><span class="w">Order-preserving residual (Tessera)</span><b class="big">d = ' + (d > 0 ? "+" : "") + d + "</b><br>zigzag " + z + ": high part <span class='mono'>" + (z >> 8) + "</span>, low byte <span class='mono'>" + (z & 0xFF) +
        "</span><br>coded with frequency table " + expo(b) + " of 256, chosen by the base's exponent</div>";
      var steps = Math.abs(d) + (Math.abs(d) === 1 ? " representable value" : " representable values");
      var msg;
      if (d === 0) msg = "After rounding to BF16 the weight has not changed: both residuals are zero. Exact duplicates of whole tensors are removed by content hash before any coding.";
      else if (expo(b) !== expo(t)) msg = "<b>The weight crosses an exponent boundary.</b> It moves by " + steps + ", but XOR flips " + n + " bits" + ((x >> 8) && (x & 0xFF) ? " spread over both byte planes" : "") + ". The integer residual stays " + d + ".";
      else if (n >= 4 && Math.abs(d) <= 2) msg = "<b>A mantissa carry.</b> The weight moves by " + steps + ", yet XOR flips " + n + " bits because the carry runs through the low mantissa bits. The integer residual stays " + d + ".";
      else msg = "Same exponent: the weight moves by " + steps + " and XOR flips " + n + (n === 1 ? " bit" : " bits") + ". Here the two residuals carry similar information; they differ at carries and exponent boundaries.";
      $("ts-read").innerHTML = msg;
    }
    function setPreset(el) {
      sb.value = el.dataset.b; sd.value = el.dataset.d;
      $("ts-presets").querySelectorAll("button").forEach(function (q) { q.setAttribute("aria-pressed", q === el); });
      draw();
    }
    $("ts-presets").querySelectorAll("button").forEach(function (p) { p.onclick = function () { setPreset(p); }; });
    function manual() { $("ts-presets").querySelectorAll("button").forEach(function (q) { q.setAttribute("aria-pressed", "false"); }); draw(); }
    sb.addEventListener("input", manual); sd.addEventListener("input", manual);
    setPreset($("ts-presets").querySelectorAll("button")[1]);
  }
})();
</script>
{% endraw %}
