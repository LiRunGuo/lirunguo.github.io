---
title: "Load or Recompute from Shared Tiers: Characterizing and Planning KV Restoration for Agent Serving"
collection: projects
permalink: /projects/2026-10-tiercut/
date: 2026-10-30
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "When several LLM serving instances share one NVMe tier for evicted KV caches, restoring a prefix can be slower than recomputing it. TierCut plans per request how much to load, and a study on five platforms maps when cutting, read priority or bypassing the tier pays."
teaser: "/images/projects/2026-10-tiercut/teaser.png"
tags: ["LLM serving", "KV cache", "tiered storage", "LLM agents"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-10-tiercut"
---

<style>
/* TierCut project page: styles scoped to .tc / .tc-* so the rest of the site is untouched; colors follow the theme */
.tc{--tc-acc:#0a7d5c;--tc-acc-soft:rgba(0,158,115,.10);--tc-blue:#0072b2;--tc-blue-soft:rgba(0,114,178,.09);--tc-nvme:#b97a00;--tc-nvme-fill:#e69f00;--tc-dram:#56b4e9;--tc-gpu:#b85c90;--tc-gray:#6f767e;--tc-gray-soft:rgba(127,127,127,.09);--tc-line:rgba(127,127,127,.28);--tc-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .tc{--tc-acc:#4fd1a1;--tc-acc-soft:rgba(79,209,161,.14);--tc-blue:#78b0f0;--tc-blue-soft:rgba(120,176,240,.14);--tc-nvme:#f0b440;--tc-nvme-fill:#f0b440;--tc-dram:#7cc6f2;--tc-gpu:#e08fbd;--tc-gray:#c3c9cf;--tc-gray-soft:rgba(255,255,255,.07);--tc-line:rgba(255,255,255,.2);--tc-card:rgba(255,255,255,.04)}
/* the fixed table of contents on wide screens would otherwise cover the end of the title */
@media screen and (min-width:1200px){.page__inner-wrap>header{padding-right:250px}}
.tc-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.tc-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--tc-acc-soft);color:var(--tc-acc)}
.tc-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.tc-tldr b{color:var(--tc-acc)}
.tc-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.tc-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.tc-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.tc-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--tc-line)}
.tc-btn.muted{background:transparent;color:var(--tc-gray)!important;border:1px dashed var(--tc-line);cursor:default}
.tc-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .tc-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.tc-lead{margin:0 0 1em;line-height:1.65}
.tc-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--tc-line);border-radius:14px;background:var(--tc-card)}
.tc-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.tc-fig.narrow img{max-width:min(100%,620px);margin:0 auto}
.tc-fig a.zoom{display:block;cursor:zoom-in;background:#fff;border-radius:8px}
.tc-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.tc-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--tc-acc);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .tc-tag{color:#0c2a20}
.tc-fig figcaption b.tc-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.tc-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.tc-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--tc-line);background:var(--tc-card);position:relative;overflow:hidden}
.tc-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--tc-c,var(--tc-acc))}
.tc-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;color:var(--tc-c,var(--tc-acc));font-variant-numeric:tabular-nums}
.tc-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.tc-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.tc-step{padding:.9em 1em;border-radius:14px;border:1px solid var(--tc-line);background:var(--tc-card)}
.tc-step .n{display:inline-grid;place-items:center;width:1.7em;height:1.7em;border-radius:50%;background:var(--tc-acc-soft);color:var(--tc-acc);font-weight:800;font-size:.85em;margin-bottom:.4em}
.tc-step b{display:block;margin-bottom:.25em}
.tc-step p{margin:0;font-size:.86em;line-height:1.55}
.tc-terms{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.1em 1.6em;margin:0 0 1.6em;padding:.9em 1.1em;border-radius:14px;border:1px solid var(--tc-line);background:var(--tc-card);font-size:.86em;line-height:1.55}
.tc-terms div{padding:.35em 0}
.tc-terms b{color:var(--tc-acc)}
.tc-cmp{display:grid;grid-template-columns:auto 1fr auto 1fr 1.05fr;gap:.55em .6em;align-items:stretch;margin:0 0 1.6em;font-size:.88em}
.tc-cmp .h{font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:0 .2em;align-self:end}
.tc-cmp .h.s{color:var(--tc-gray)}.tc-cmp .h.c,.tc-cmp .h.o{color:var(--tc-acc)}
.tc-cmp .k{font-weight:700;align-self:center;padding-right:.3em}
.tc-cmp .s,.tc-cmp .c,.tc-cmp .o{padding:.55em .75em;border-radius:10px;line-height:1.4}
.tc-cmp div.s{background:var(--tc-gray-soft);border:1px solid var(--tc-line);color:var(--tc-gray)}
.tc-cmp div.c{background:var(--tc-acc-soft);border:1px solid var(--tc-acc)}
.tc-cmp div.o{border-left:3px solid var(--tc-acc);border-radius:4px}
.tc-cmp .arr{align-self:center;color:var(--tc-gray);font-size:1.1em}
.tc-eq{margin:0 0 1em;padding:.7em .9em;border-radius:12px;background:var(--tc-gray-soft);overflow-x:auto;text-align:center}
.tc-eqparts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7em;margin:0 0 1.6em;font-size:.86em;line-height:1.55}
.tc-eqparts div{padding:.7em .9em;border-radius:10px;border-left:3px solid var(--tc-c,var(--tc-acc));background:var(--tc-card)}
.tc-eqparts b{color:var(--tc-c,var(--tc-acc))}
.tc-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--tc-line);border-radius:14px;background:var(--tc-card);overflow-x:auto}
.tc-table table{font-size:.8em;margin:.6em 0;min-width:560px;font-variant-numeric:tabular-nums}
.tc-table td strong{color:var(--tc-acc)}
.tc-table p{font-size:.86em;margin:.2em}
.tc-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.tc-cardx{padding:.9em 1em;border-radius:14px;border:1px solid var(--tc-line);background:var(--tc-card);border-top:4px solid var(--tc-c,var(--tc-acc))}
.tc-cardx .w{font-size:.72em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--tc-c,var(--tc-acc))}
.tc-cardx b{display:block;margin:.2em 0 .3em;font-size:1.04em}
.tc-cardx p{margin:0;font-size:.86em;line-height:1.55}
.tc-note{font-size:.86em;line-height:1.55;margin:0 0 1.6em;padding:.55em .8em;border-left:3px solid var(--tc-acc);background:var(--tc-acc-soft);border-radius:4px}
.tc-limits{margin:0 0 1.6em;padding:.9em 1.1em .5em;border-radius:14px;border:1px solid var(--tc-line);background:var(--tc-card);font-size:.9em;line-height:1.6}
.tc-limits ul{margin:.2em 0 .5em 1.1em}
.tc-limits li{margin:.35em 0}
.tc-abs{margin:0 0 1em;padding:.7em 1em;border-radius:12px;border:1px solid var(--tc-line);background:var(--tc-card);font-size:.9em;line-height:1.65}
.tc-abs summary{cursor:pointer;font-weight:700}
.tc-abs p{margin:.7em 0 .2em}
/* interactive planner */
.tc-ctl{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.55em 1.4em;margin:0 0 .8em;font-size:.82em}
.tc-row{display:flex;flex-wrap:wrap;align-items:center;gap:.3em .7em}
.tc-row.full{grid-column:1/-1}
.tc-row .l{font-weight:700;min-width:8.6em}
.tc-row input[type=range]{flex:1 1 130px;min-width:0;accent-color:var(--tc-acc);margin:0}
.tc-row output{min-width:5.4em;text-align:right;font-variant-numeric:tabular-nums;color:var(--tc-acc);font-weight:700}
.tc-seg{display:inline-flex;flex-wrap:wrap;border:1px solid var(--tc-line);border-radius:8px;overflow:hidden}
.tc .tc-seg button{font:inherit;margin:0;border:0;border-right:1px solid var(--tc-line);border-radius:0;background:transparent;color:var(--global-text-color);padding:.3em .8em;cursor:pointer}
.tc .tc-seg button:last-child{border-right:0}
.tc .tc-seg button[aria-pressed="true"]{background:var(--tc-acc);color:#fff;font-weight:700}
html[data-theme="dark"] .tc .tc-seg button[aria-pressed="true"]{color:#0c2a20}
.tc-chart{display:block;width:100%;height:auto;overflow:visible;margin:.2em 0 0}
.tc-chart text{font-size:11px;fill:var(--global-text-color);font-family:inherit}
.tc-chart .ax{stroke:var(--global-text-color);stroke-width:1;opacity:.55}
.tc-chart .grid{stroke:var(--tc-line);stroke-width:1}
.tc-chart .tot{fill:none;stroke:var(--tc-acc);stroke-width:3;stroke-linejoin:round}
.tc-chart .ld{fill:none;stroke:var(--tc-nvme-fill);stroke-width:1.8;stroke-dasharray:5 3}
.tc-chart .rc{fill:none;stroke:var(--tc-gpu);stroke-width:1.8;stroke-dasharray:2 3}
.tc-chart .pick{fill:var(--tc-acc);stroke:var(--global-bg-color,#fff);stroke-width:2}
.tc-chart .pickline{stroke:var(--tc-acc);stroke-width:1;stroke-dasharray:3 3}
.tc-chart .dram{fill:var(--tc-dram)}.tc-chart .nvme{fill:var(--tc-nvme-fill)}
.tc-chart .cutaway{fill:var(--tc-gray);opacity:.45}
.tc-legend{display:flex;flex-wrap:wrap;gap:.2em 1.2em;font-size:.78em;margin:.4em .1em 0;color:var(--global-text-color)}
.tc-legend>span{display:inline-flex;align-items:center;gap:.4em}
.tc-legend svg{display:inline}
.tc-out{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6em;margin:.9em 0 0}
.tc-out div{padding:.6em .8em;border-radius:10px;border:1px solid var(--tc-line);font-size:.8em;line-height:1.45}
.tc-out div.win{border-color:var(--tc-acc);background:var(--tc-acc-soft)}
.tc-out b{display:block;font-size:1.25em;font-variant-numeric:tabular-nums}
.tc-out div.win b{color:var(--tc-acc)}
.tc p.tc-read{font-size:.86em;line-height:1.55;margin:.9em 0 .2em;padding:.55em .8em;border-left:3px solid var(--tc-acc);background:var(--tc-acc-soft);border-radius:4px}
@media (max-width:900px){.tc-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.tc-steps,.tc-cards{grid-template-columns:1fr}}
@media (max-width:640px){.tc-terms,.tc-eqparts,.tc-ctl,.tc-out{grid-template-columns:1fr}.tc-cmp{grid-template-columns:1fr}.tc-cmp .arr,.tc-cmp .h,.tc-cmp span:empty{display:none}.tc-cmp .k{margin-top:.6em}.tc-tldr{font-size:1.04em}.tc-row .l{min-width:0;flex:1 1 100%}.tc-chart text{font-size:16px}}
</style>

<div class="tc">
<div class="tc-kicker"><span>LLM serving</span><span>KV cache</span><span>Tiered storage</span><span>Agent workloads</span></div>
<p class="tc-tldr">Serving stacks keep evicted KV caches in DRAM and on local NVMe so that an agent's next call can restore its prefix instead of recomputing it. On a multi-GPU node every instance shares that NVMe, and restoring soon becomes <b>slower than recomputing</b>. <b>TierCut</b> decides per request how much of a stored prefix to load, and a study on five platforms shows when cutting, read priority or bypassing the tier is the right control.</p>
<div class="tc-btns"><a class="tc-btn" href="#try-the-planner">▶&nbsp; Try the planner</a><a class="tc-btn ghost" href="#how-tiercut-works">How it works</a><a class="tc-btn ghost" href="#results">Results</a><a class="tc-btn ghost" href="#which-control-when">Decision map</a><span class="tc-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="tc-fig narrow"><a class="zoom" href="/images/projects/2026-10-tiercut/fig1-teaser.png"><img src="/images/projects/2026-10-tiercut/fig1-teaser.png" alt="Figure 1: mean TTFT against the number of Llama-3.1-8B instances sharing one NVMe array" loading="lazy"></a><figcaption><span class="tc-tag">Figure 1</span><b class="tc-title">A shared NVMe tier turns from a shortcut into the bottleneck.</b>Mean time to first token (TTFT) of agent calls with 64k-token contexts as more Llama-3.1-8B instances share one NVMe array (8×H200 node). LMCache with retention loads every hit and its TTFT grows from 0.84 s to 6.17 s; above the dashed line it is slower than recomputing every prefix. A planner that assumes the idle disk's bandwidth (Static-<i>B</i>) falls behind as well. TierCut stays between 0.87 and 1.03 s.</figcaption></figure>
</div>

## At a glance

<div class="tc"><div class="tc-stats">
<div class="tc-stat" style="--tc-c:var(--tc-blue)"><div class="v">1.6×</div><div class="d">the mean TTFT of recomputing every prefix, when two Llama-3.1-8B instances on one disk load every hit instead</div></div>
<div class="tc-stat"><div class="v">−44%</div><div class="d">mean TTFT against LMCache with retention in that setting (H100 NVL node, 95% CI −49 to −40)</div></div>
<div class="tc-stat"><div class="v">−31%</div><div class="d">against a planner with a constant bandwidth while background reads on the disk come and go</div></div>
<div class="tc-stat"><div class="v">19–24%</div><div class="d">faster than a backlog threshold and than read priority with four instances; with two, both simple policies match TierCut</div></div>
</div></div>

## Why restoring can lose to recomputing

<div class="tc">
<div class="tc-steps">
<div class="tc-step"><span class="n">1</span><b>Agents leave long prefixes behind.</b><p>An agent session makes tens of LLM calls, and each prompt is the previous context plus a tool result. In the deep-research traces we replay, the final context has a median of 26k tokens. While a session waits for its tool, the GPU evicts its KV cache to serve others.</p></div>
<div class="tc-step"><span class="n">2</span><b>Evicted KV goes to DRAM and NVMe.</b><p>LMCache keeps evicted KV in host DRAM (L1) and on local NVMe (L2) and loads it back on a hit. From DRAM this takes about 0.2 s for a 64k-token context. From NVMe, nearly all of the time is the disk read.</p></div>
<div class="tc-step"><span class="n">3</span><b>Every instance shares the disk.</b><p>Each GPU runs its own instance, and all of them read and write the same devices. With four Llama instances, one prefix read takes up to 20–26 s. Recomputing a 64k-token prefix takes 3.4 s.</p></div>
</div>
<div class="tc-table" markdown="1">

<span class="tc-tag">Table 1</span> **Mean TTFT (ms) of agent calls with 64k-token contexts, by restoration path** (8×H200 node, LMCache default policy).

| Model | All KV in DRAM | Recompute only | NVMe, 1 instance | NVMe, 4 instances sharing |
|---|---|---|---|---|
| Qwen3.5-9B (hybrid, 32 KB of KV per token) | 186–222 | 2891 | 320 | 800–4510 |
| Llama-3.1-8B (dense, 128 KB of KV per token) | 182 | 1416 | 1698 | 4482–7717 |

Read each row from left to right. DRAM is always cheap. For Llama, four instances sharing the array make NVMe restoration 3–5× slower than recomputing. Qwen3.5-9B stores a quarter of the bytes per token, so loading still pays when it runs alone, but not under heavy sharing.

</div>
<p class="tc-lead">Whether loading one token's KV beats recomputing it depends on a single number. Loading costs the KV size <i>s</i> divided by the bandwidth; recomputing costs the marginal prefill time at the token's position <i>p</i>, which grows with <i>p</i> because attention is quadratic. Recomputation wins below the <b>crossover bandwidth</b>:</p>
<div class="tc-eq">\[ B^{*}(p) = \frac{s}{k_1 + k_2\,p} \]</div>
<figure class="tc-fig narrow"><a class="zoom" href="/images/projects/2026-10-tiercut/fig3-crossover.png"><img src="/images/projects/2026-10-tiercut/fig3-crossover.png" alt="Figure 3: crossover bandwidth against token position for four models" loading="lazy"></a><figcaption><span class="tc-tag">Figure 3</span><b class="tc-title">The break-even point depends on the model, the position and the load.</b>Each curve is the bandwidth below which recomputing the token at position <i>p</i> is faster than loading it (prefill measured on one H200). At the same position the four models differ by up to 4.3×. The shaded band is what one instance obtains from the array, from idle (6.3 GB/s) to three instances sharing it (about 2.2 GB/s). Llama's curve crosses the band: the right decision for a long prefix flips with contention, so no fixed rule such as "always load" is right for every model and load.</figcaption></figure>
<p class="tc-lead">The tiered cache adds side effects of its own. By default it reads a session's prefix from NVMe again on every call, and it writes recomputed chunks back to NVMe although they are already there, which adds to the contention. The terms used on this page:</p>
<div class="tc-terms">
<div><b>TTFT.</b> Time to first token: from sending a call to its first output token. Lower is better.</div>
<div><b>Prefill.</b> The GPU pass that computes the KV cache of the prompt. Recomputing a prefix means prefilling it again.</div>
<div><b>L1, L2.</b> LMCache's two tiers: host DRAM and a file system on local NVMe. KV is stored in chunks of 528 tokens.</div>
<div><b>Cut.</b> The point in a stored prefix up to which KV is restored. Everything after it is recomputed.</div>
<div><b>Drain rate.</b> The rate at which one instance's own reads complete. It already reflects what the other instances read and write.</div>
<div><b>Backlog.</b> The bytes an instance has asked the disk for that have not arrived yet. A new read waits behind them.</div>
<div><b>LMCache-retain.</b> LMCache that loads every hit and keeps chunks read from NVMe in DRAM. Our main baseline.</div>
<div><b>Static-<i>B</i>.</b> TierCut with the rate fixed at the idle disk's bandwidth: a stand-in for planners that assume a constant bandwidth.</div>
<div><b>vLLM APC.</b> vLLM's GPU prefix cache alone: every evicted prefix is recomputed.</div>
<div><b>LMCache-DRAM.</b> LMCache with the DRAM tier only, no NVMe.</div>
</div>
</div>

## How TierCut works

<div class="tc">
<p class="tc-lead">Existing restoration planners such as Cake and CacheFlow treat each request's restoration on its own, inside a modified engine. They do not charge a request's reads for the delay they cause other instances on the same storage, or its recomputation for the decodes it stalls. TierCut plans in the storage layer and changes four things:</p>
<div class="tc-cmp">
<span></span><span class="h s">Load every hit</span><span></span><span class="h c">TierCut</span><span class="h o">Why it matters</span>
<span class="k">Decision</span><div class="s">restore the longest stored prefix</div><span class="arr">→</span><div class="c">choose a cut per request, before any disk read is issued</div><div class="o">a read that has been issued cannot be cancelled</div>
<span class="k">Bandwidth</span><div class="s">a constant, measured on the idle disk</div><span class="arr">→</span><div class="c">the drain rate of the instance's own reads, plus its backlog</div><div class="o">the constant underestimates load time by a median factor of 4.9 with four instances</div>
<span class="k">Recompute cost</span><div class="s">the request's own prefill time</div><span class="arr">→</span><div class="c">prefill time × α, with α = 2</div><div class="o">a recompute prefill stalls the decodes of other requests in the same batch</div>
<span class="k">The tier</span><div class="s">re-reads each call, writes recomputed chunks back</div><span class="arr">→</span><div class="c">keep loaded chunks in DRAM, skip writes of chunks already on NVMe</div><div class="o">87% of NVMe writes disappear with three instances</div>
</div>
<figure class="tc-fig"><a class="zoom" href="/images/projects/2026-10-tiercut/fig2-overview.png"><img src="/images/projects/2026-10-tiercut/fig2-overview.png" alt="Figure 2: TierCut overview" loading="lazy"></a><figcaption><span class="tc-tag">Figure 2</span><b class="tc-title">TierCut decides, before any read, how much of a stored prefix to restore.</b>(a) The green boxes are added to each instance's LMCache server. A lookup finds the stored prefix and the tier of every chunk (1); the planner combines it with the estimator's rate and backlog (2) and picks the cut ĉ (3); only chunks up to ĉ are read, and the unchanged vLLM engine recomputes the rest (4); the store filter skips chunks NVMe already holds (5). (b) Illustrative cost of every cut: under contention the cheapest cut is the end of the DRAM part; on an idle disk it is the full prefix. (c) One request's timeline under contention.</figcaption></figure>
<p class="tc-lead">For a stored prefix of <i>K</i> chunks of <i>t</i> tokens, the planner estimates the time until prefill can finish for every cut <i>c</i> and takes the cheapest:</p>
<div class="tc-eq">\[ T(c) \;=\; \underbrace{\mathbf{1}[A_{\mathrm{N}}(c) &gt; 0]\;\frac{Q + A_{\mathrm{N}}(c)}{\hat r}}_{\text{load from NVMe}} \;+\; \underbrace{\alpha\,\bigl(f(Kt) - f(ct)\bigr)}_{\text{recompute the rest}}, \qquad f(x) = k_1 x + \tfrac{1}{2} k_2 x^2 \]</div>
<div class="tc-eqparts">
<div style="--tc-c:var(--tc-nvme)"><b>Load term.</b> <i>A</i><sub>N</sub>(<i>c</i>) is the bytes cut <i>c</i> must read from NVMe; chunks in DRAM cost nothing. They are charged at the measured drain rate r̂ and wait behind the backlog <i>Q</i>.</div>
<div style="--tc-c:var(--tc-gpu)"><b>Recompute term.</b> The prefill time of the dropped tokens from a one-time profile of the GPU (<i>k</i><sub>1</sub>, <i>k</i><sub>2</sub>), weighted by the threshold α for the decode stalls it causes.</div>
</div>
<div class="tc-steps">
<div class="tc-step"><span class="n">1</span><b>Measure the drain rate.</b><p>The estimator divides the bytes of completed reads by the time during which a read was in flight, both decayed over 10 s. A small prior pulls the estimate back up after a pause, so the planner loads again and finds out whether contention has gone.</p></div>
<div class="tc-step"><span class="n">2</span><b>Price a few cuts.</b><p>Only cuts where the whole model can resume are candidates, which covers hybrid models with recurrent state. Within a run of chunks on one tier the cost is concave, so the best cut is at an end of a run. Planning takes under a millisecond.</p></div>
<div class="tc-step"><span class="n">3</span><b>Report a shorter prefix.</b><p>TierCut runs as two LMCache policy plugins, about 350 lines of Python. vLLM is unchanged: the cache simply reports the prefix up to the cut, and the engine recomputes the rest as it would after any partial hit.</p></div>
</div>
<p class="tc-note"><b>One rule falls out of the cost.</b> Without backlog, reading a whole NVMe run is cheaper than recomputing it exactly when r̂ &gt; <i>B</i>*(<i>m</i>) / α, where <i>m</i> is the token position at the middle of the run. For a 64k-token Llama prefix entirely on NVMe and α = 1, that threshold is 2.52 GB/s on an H200: between the idle array and the array shared three ways.</p>
</div>

## Try the planner

<div class="tc">
<p class="tc-lead">The figure below evaluates the cost model above in your browser, with the prefill profiles measured on one H200. It shows what the planner would decide for one request; its numbers are the model's estimates, not measurements from the paper.</p>
<figure class="tc-fig" id="tc-demo">
<div class="tc-ctl">
<div class="tc-row full"><span class="l">Situation</span><div class="tc-seg" id="tc-presets"><button type="button" data-p="idle">Idle array</button><button type="button" data-p="share">Three instances sharing</button><button type="button" data-p="queue">Reads queued</button></div></div>
<div class="tc-row"><span class="l">Model</span><div class="tc-seg" id="tc-model"><button type="button" data-v="llama">Llama-3.1-8B</button><button type="button" data-v="q32">Qwen3-32B</button></div></div>
<div class="tc-row"><span class="l">Threshold α</span><div class="tc-seg" id="tc-alpha"><button type="button" data-v="1">1</button><button type="button" data-v="2">2</button><button type="button" data-v="3">3</button></div></div>
<div class="tc-row"><label class="l" for="tc-r">Drain rate r̂</label><input type="range" id="tc-r" min="0" max="100" step="1"><output id="tc-r-o"></output></div>
<div class="tc-row"><label class="l" for="tc-q">Backlog <i>Q</i></label><input type="range" id="tc-q" min="0" max="12" step="0.5"><output id="tc-q-o"></output></div>
<div class="tc-row"><label class="l" for="tc-len">Stored prefix</label><input type="range" id="tc-len" min="8" max="64" step="4"><output id="tc-len-o"></output></div>
<div class="tc-row"><label class="l" for="tc-dram">Already in DRAM</label><input type="range" id="tc-dram" min="0" max="100" step="5"><output id="tc-dram-o"></output></div>
</div>
<svg class="tc-chart" id="tc-chart" viewBox="0 0 680 330" role="img" aria-label="Estimated cost of every cut: load term, recompute term and their sum, with the chosen cut marked"></svg>
<div class="tc-legend">
<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--tc-acc)" stroke-width="3"/></svg><span>cost <i>T</i>(<i>c</i>)</span></span>
<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--tc-nvme-fill)" stroke-width="2" stroke-dasharray="5 3"/></svg>load from NVMe</span>
<span><svg width="26" height="10"><line x1="1" y1="5" x2="25" y2="5" stroke="var(--tc-gpu)" stroke-width="2" stroke-dasharray="2 3"/></svg>α × recompute</span>
<span><svg width="12" height="10"><rect width="12" height="9" rx="2" fill="var(--tc-dram)"/></svg>chunk in DRAM</span>
<span><svg width="12" height="10"><rect width="12" height="9" rx="2" fill="var(--tc-nvme-fill)"/></svg>chunk only on NVMe</span>
</div>
<div class="tc-out" id="tc-out"></div>
<p class="tc-read" id="tc-read" aria-live="polite"></p>
<figcaption><span class="tc-tag">Interactive</span><b class="tc-title">The cost of every cut for one request.</b>The horizontal axis is the cut: tokens to the left are restored, tokens to the right are recomputed. The bar under the axis shows where each chunk is stored; the first part of the prefix is in DRAM, as after an earlier call of the same session. The dot marks the cut the planner picks. Lower the drain rate or add backlog and the dot jumps from the right end (load everything) to the end of the DRAM part. Both models are dense, with prefill profiles from one H200. "Three instances sharing" uses the 2.2 GB/s of Figure 3; there, set "Already in DRAM" to 0% and α to 1, and the planner recomputes, because 2.2 GB/s is below Llama's threshold of 2.52 GB/s. "Reads queued" is a made-up state with one earlier 4 GB read still in flight.</figcaption>
</figure>
</div>

## Results

<div class="tc">
<p class="tc-lead">We replay agent trajectories with contexts cut at 64k tokens on five single-node platforms, with every instance on its own GPU and all instances sharing one local disk. Systems run in balanced rounds, and we compare systems only within a platform. The main results below are from nodes allocated to us alone, with α = 2 and three rounds per system.</p>
</div>

<div class="tc"><div class="tc-table" markdown="1">

<span class="tc-tag">Table 2</span> **The five platforms.** All run vLLM 0.29.0, LMCache 0.5.6rc5 and PyTorch 2.13. Prefill is the time for 64k Llama-3.1-8B tokens on one GPU; *B*\* is Llama's crossover bandwidth at positions 0 and 32k.

| Node | Shared L2 device | Idle read | Prefill | *B*\* (GB/s) | Allocation |
|---|---|---|---|---|---|
| 8×H200 | 2 NVMe, RAID0 | 6.3 GB/s | 3.42 s | 5.89 / 2.52 | shared with other users |
| 2×H100 NVL | 1 NVMe | 6.3 GB/s | 4.59 s | 3.94 / 1.81 | exclusive |
| 4×H100 SXM | 1 local SSD | 5.1–5.5 GB/s | 3.31 s | 5.71 / 2.60 | exclusive |
| 8×B200 | 2 NVMe, RAID1 | 7.1 GB/s | 1.59 s | 11.9 / 5.43 | exclusive |
| 4×GH200 | RAM disk, reads throttled | 0.5–8 GB/s | 3.84 s | 5.04 / 2.24 | exclusive |

Workload: OpenResearcher deep-research trajectories replayed as sessions of up to 60 calls. Each instance serves 16 sessions (24 for Qwen3.5-9B) that arrive at 0.2 per second, wait a simulated tool latency between calls and generate 32 output tokens per call. GPU memory is set so that the GPU cache cannot hold all sessions, and DRAM (32 GB per instance for Llama, 16 GB for Qwen3.5-9B) can stage a full prefix but holds only a few sessions. Means are over all calls of all instances and rounds, with 95% bootstrap confidence intervals.

</div></div>

<div class="tc">
<figure class="tc-fig"><a class="zoom" href="/images/projects/2026-10-tiercut/fig4-platforms.png"><img src="/images/projects/2026-10-tiercut/fig4-platforms.png" alt="Figure 4: mean TTFT of all systems on the H100 NVL and B200 nodes" loading="lazy"></a><figcaption><span class="tc-tag">Figure 4</span><b class="tc-title">The same planner on a slow-prefill and a fast-prefill node.</b>Mean TTFT with 95% confidence intervals, Llama-3.1-8B. (a) One instance on the H100 NVL node: every system that uses the disk is within noise, so planning costs nothing when the disk is free. (b) Two instances share the disk: LMCache-retain rises to 3.42 s and TierCut is at 1.91 s, also 13% faster than recomputing everything. (c, d) On the B200 node a 64k-token prefill takes 1.59 s instead of 4.59 s. TierCut is still the fastest system that reads the disk, but serving from DRAM alone is fastest overall: here the NVMe tier should not be used at all.</figcaption></figure>
<figure class="tc-fig narrow"><a class="zoom" href="/images/projects/2026-10-tiercut/fig5-dynamic.png"><img src="/images/projects/2026-10-tiercut/fig5-dynamic.png" alt="Figure 5: rate estimate and mean TTFT while background reads come and go" loading="lazy"></a><figcaption><span class="tc-tag">Figure 5</span><b class="tc-title">The estimate follows contention that changes within a run.</b>Two Llama instances on the H100 NVL node while another process reads the same disk in 90 s bursts (grey bands). (a) TierCut's rate estimate falls while the bursts are on and recovers after them; Static-<i>B</i> keeps its 6.4 GB/s. (b) With the bursts off, the three systems are close. With them on, TierCut cuts 62% of its decisions and has a mean TTFT of 2.56 s, against 5.41 s for Static-<i>B</i> and 6.37 s for LMCache-retain. Over whole runs TierCut is 31% faster than Static-<i>B</i> and 42% faster than LMCache-retain.</figcaption></figure>
</div>

<div class="tc"><div class="tc-table" markdown="1">

<span class="tc-tag">Table 3</span> **Simple node-level policies are strong.** Mean TTFT (s), Llama-3.1-8B. Each column is a separate experiment with its own balanced rounds, so compare within a column only. NVL: two instances on the H100 NVL node; SXM: four instances on the H100 SXM node.

| System | NVL, A | NVL, B | NVL, 2× load | NVL, 256 output tokens | SXM, A | SXM, B |
|---|---|---|---|---|---|---|
| vLLM APC (recompute) | – | – | – | – | **1.28** | – |
| LMCache-DRAM | – | – | – | – | 1.37 | – |
| LMCache-retain | 3.09 | – | 14.4 | – | 6.60 | – |
| Deadline fallback | 3.58 | – | – | – | 6.49 | – |
| Write-back cap | 2.81 | – | – | – | 5.05 | – |
| Backlog threshold | 2.07 | 2.05 | 10.4 | 0.49 | 2.05 | 2.07 |
| Read priority | 2.00 | 2.08 | **7.98** | 0.56 | 2.17 | 1.96 |
| Read priority + backlog threshold | – | 2.04 | – | – | – | 1.54 |
| TierCut | **1.97** | 1.89 | 9.64 | 0.54 | 1.66 | **1.38** |
| TierCut + read priority | – | **1.77** | 8.69 | **0.48** | – | 1.42 |

The backlog threshold serves only the DRAM part whenever the instance has disk reads in flight. Read priority lets no instance write back while any instance on the node is reading. With two instances both are within noise of TierCut (column A), and TierCut combined with read priority is the fastest (column B). With four instances an instance's own backlog says less about the node, and TierCut is 19% and 24% faster than the two (SXM, A). At twice the load, read priority alone leads.

</div></div>

<div class="tc">
<figure class="tc-fig narrow"><a class="zoom" href="/images/projects/2026-10-tiercut/fig6-load-sweep.png"><img src="/images/projects/2026-10-tiercut/fig6-load-sweep.png" alt="Figure 6: mean TTFT relative to read priority and throughput against offered load" loading="lazy"></a><figcaption><span class="tc-tag">Figure 6</span><b class="tc-title">Under heavy load the GPU, not the disk, is the bottleneck.</b>Two Llama instances on the H100 NVL node, with session rate and session count scaled together. (a) Mean TTFT relative to read priority; below 1.0 is faster. (b) Completed calls per second. Up to 1.5× the base load, TierCut with read priority is 8–12% faster than read priority alone. At 2× and 3× it is 11% and 7% slower: every cut adds prefill to a GPU that already has 9–15 requests waiting. Letting α grow with that queue brings TierCut back within noise of read priority at 2×, but at 3× read priority alone is the right policy.</figcaption></figure>
<div class="tc-stats">
<div class="tc-stat"><div class="v">−53%</div><div class="d">TTFT for the hybrid Qwen3.5-9B with five instances; with one or three it rarely cuts and matches LMCache-retain, because its KV is small</div></div>
<div class="tc-stat"><div class="v">−65%</div><div class="d">TTFT on OpenSeeker search-agent traces (median final context 68.5k tokens), three Llama instances</div></div>
<div class="tc-stat" style="--tc-c:var(--tc-blue)"><div class="v">&lt; 1 ms</div><div class="d">median planning time per request (0.63–0.94 ms), about 0.1% of TTFT</div></div>
<div class="tc-stat" style="--tc-c:var(--tc-blue)"><div class="v">218 / 218</div><div class="d">forced cuts on Llama give the same first 32 greedy tokens as full recomputation (H100 NVL node)</div></div>
</div>
<p class="tc-note"><b>About the 8×H200 numbers.</b> Figure 1 and the two cards on Qwen3.5-9B and OpenSeeker come from a server shared with other users, whose disk I/O we did not control. Those runs used α = 1 and two rounds per system. Everything else on this page is from exclusively allocated nodes with α = 2 and three rounds.</p>
</div>

## Which control, when

<div class="tc">
<p class="tc-lead">Cutting is one of three controls a node has over its shared tier. The others act on the tier's writes (read priority over write-back) or switch the tier off (bypass). The decision map places every configuration we ran by how fast its GPU prefills and how much bandwidth one instance obtains.</p>
<figure class="tc-fig narrow"><a class="zoom" href="/images/projects/2026-10-tiercut/fig7-decision-map.png"><img src="/images/projects/2026-10-tiercut/fig7-decision-map.png" alt="Figure 7: decision map over prefill time and per-instance bandwidth" loading="lazy"></a><figcaption><span class="tc-tag">Figure 7</span><b class="tc-title">A decision map for Llama-3.1-8B.</b>Each point is one configuration (node and number of instances). Horizontal: prefill time of 64k tokens on that GPU. Vertical: the median bandwidth of an instance's reads when it loads nearly every hit. The two curves are the crossover bandwidth at position 0 and at 32k tokens divided by α = 2. Circles: TierCut within noise of LMCache-retain. Triangles: TierCut faster. Squares: DRAM-only or recompute-only serving is fastest.</figcaption></figure>
<div class="tc-cards">
<div class="tc-cardx" style="--tc-c:var(--tc-gray)"><span class="w">Above the upper curve</span><b>Load every hit.</b><p>The disk is fast enough for every position. TierCut matches LMCache-retain here, so no planning is needed: one instance per disk on both H100 nodes.</p></div>
<div class="tc-cardx"><span class="w">Near and between the curves</span><b>Cut, with read priority.</b><p>The best cut depends on each request's positions and backlog. A per-request plan pays, and read priority removes the write-back that competes with reads.</p></div>
<div class="tc-cardx" style="--tc-c:var(--tc-blue)"><span class="w">Far below</span><b>Bypass the tier.</b><p>Nearly every read would be cut, and the tier still costs its write-back. Serving from DRAM alone or recomputing has the lowest TTFT: the B200 node, and four instances on the H100 SXM node.</p></div>
</div>
</div>

## Using TierCut

<div class="tc">
<p class="tc-lead">TierCut is a research prototype for stacks that already run vLLM with LMCache. It registers one prefetch policy and one store policy through LMCache's plugin interfaces; nothing in vLLM changes. Each instance runs its own planner; the drain rate it measures on its own reads already reflects what the other instances read and write.</p>
<div class="tc-table" markdown="1">

| What you provide | Value in the paper | What it does |
|---|---|---|
| Prefill profile *k*<sub>1</sub>, *k*<sub>2</sub> | one profile per GPU model | Single requests of 1k–64k random tokens on one GPU, fitted once. Gives the recompute cost of any suffix. |
| Benefit threshold α | 2 | Weight on recompute time for the decodes it stalls. α = 1 cut too eagerly and raised end-to-end latency by 11–26% on one uncontended instance. |
| Estimator window τ | 10 s | Decay of the drain-rate estimate. 10 s was 14% faster than 2 s and equal to 30 s. |
| Prior | 3 GB/s for 0.5 s | About half the idle disk's bandwidth. Used before the first reads and after a pause. |
| Tested with | vLLM 0.29.0, LMCache 0.5.6rc5 | LMCache in multi-process mode, chunks of 528 tokens, L2 on local NVMe with O_DIRECT. |

</div>
<p class="tc-note"><b>Pair it with read priority.</b> Read priority acts on the tier's writes and TierCut on its reads, so the two combine: together they were the fastest system with two instances at the base load. Read priority needs the instances of a node to publish their in-flight bytes in shared memory.</p>
</div>

## What TierCut does not do

<div class="tc"><div class="tc-limits">
<ul>
<li><b>It is not always needed.</b> With two instances and steady load, a rule that stops loading while the instance's own reads queue decides nearly as TierCut does. The measured rate matters with four instances, under background bursts, and across models with different break-even points.</li>
<li><b>It does not fix an overloaded GPU.</b> From twice the base load, cutting delays the requests queued for the GPU. A threshold that follows the queue helps at 2×; at 3× read priority alone is faster.</li>
<li><b>It cannot make a slow tier pay.</b> Where prefill is fast or DRAM holds the working set, the NVMe tier should be bypassed. With 64 or 128 GB of DRAM per instance on the H100 SXM node, DRAM-only serving is fastest.</li>
<li><b>It only chooses a prefix length.</b> It does not overlap loading with computation as Cake does, choose layers or devices, or change admission or eviction. A request-level emulation of overlap matches TierCut's mean TTFT at the base load and falls behind at twice the load.</li>
<li><b>The estimator is simple.</b> A constant rate tuned per node can beat it under steady load, and the plugin wraps three internal LMCache methods, which ties it to LMCache 0.5.6.</li>
</ul>
</div></div>

## Paper and code

<div class="tc">
<details class="tc-abs"><summary>Abstract</summary><p>LLM agents issue tens of calls per session over contexts of tens of thousands of tokens, and serving stacks keep evicted KV caches in host DRAM and local NVMe so that later calls can restore them instead of recomputing them. On a multi-GPU node, however, all serving instances share the NVMe tier, and restoring from it soon becomes slower than recomputing: with two Llama-3.1-8B instances on one disk, loading every hit has 1.6× the mean time to first token (TTFT) of recomputing every prefix. We characterize this contention and the tier's side effects, and study the controls a node can apply to it: cutting how much of a prefix to read, giving reads priority over write-back, and bypassing the tier. For the first, we present TierCut, a storage-layer planner that picks per request how much of a stored prefix to restore from the tier of every chunk, an online estimate of the instance's drain rate and backlog, and the GPU's prefill cost weighted by a threshold for the decode stalls that recomputation causes. Implemented as LMCache plugins with vLLM unchanged, TierCut reduces mean TTFT by 44% relative to LMCache with retention when two instances share an H100 node's disk, by 48–83% with 2–4 instances on an H200 node, and by 31% relative to a constant-bandwidth planner under fluctuating I/O. Simple node-level policies are strong: with two instances, a backlog threshold and read priority come within noise of TierCut, while with four instances on an H100 SXM node TierCut is 19–24% faster than both. Under heavy load the GPU, not the tier, becomes the bottleneck and cutting stops paying; a threshold that grows with the GPU's queue keeps TierCut with read priority on par with read priority at twice the load and keeps its gain at the base load, but at three times the load read priority alone is faster. A decision map over prefill speed and per-instance bandwidth summarizes when each control pays, including nodes where serving from DRAM alone or recomputing every prefix is fastest.</p></details>
</div>

The manuscript is under review. Paper, code and BibTeX will be linked here once the preprint is public.

<script>
(function () {
  var root = document.getElementById("tc-demo");
  if (!root) return;
  var CHUNK = 528, GB = 1e9;
  /* prefill profiles on one H200: seconds = k1 x + k2 x^2 / 2 for x tokens; s = KV bytes per token */
  var MODELS = {
    llama: {name: "Llama-3.1-8B", s: 131072, k1: 2.2255e-5, k2: 9.0876e-10},
    q32: {name: "Qwen3-32B", s: 262144, k1: 9.8472e-5, k2: 3.5412e-9}
  };
  var PRESETS = {
    idle: {r: 6.3, q: 0, len: 64, dram: 30},
    share: {r: 2.2, q: 0, len: 64, dram: 30},
    queue: {r: 1.0, q: 4, len: 64, dram: 30}
  };
  var st = {model: "llama", alpha: 2, r: 1.0, q: 4, len: 64, dram: 30, preset: "queue"};
  var RMIN = 0.25, RMAX = 8;
  function $(id) { return document.getElementById(id); }
  function rToPos(r) { return Math.round(100 * Math.log(r / RMIN) / Math.log(RMAX / RMIN)); }
  function posToR(p) { return RMIN * Math.pow(RMAX / RMIN, p / 100); }
  function f(m, x) { return m.k1 * x + m.k2 * x * x / 2; }
  function fmt(x, d) { return x.toFixed(d === undefined ? 2 : d); }
  function ktok(n) { var k = n / 1024; return (k < 9.95 && Math.abs(k - Math.round(k)) > 0.05 ? k.toFixed(1) : String(Math.round(k))) + "k"; }

  function plan() {
    var m = MODELS[st.model], K = Math.max(1, Math.round(st.len * 1024 / CHUNK));
    var d = Math.round(st.dram / 100 * K), r = st.r * GB, Q = st.q * GB, full = f(m, K * CHUNK);
    var pts = [], best = 0;
    for (var c = 0; c <= K; c++) {
      var bytes = Math.max(0, c - d) * CHUNK * m.s;
      var load = bytes > 0 ? (Q + bytes) / r : 0;
      var rec = full - f(m, c * CHUNK);
      pts.push({c: c, load: load, rec: rec, T: load + st.alpha * rec});
      if (pts[c].T <= pts[best].T + 1e-12) best = c;   /* ties go to the longer prefix */
    }
    var mid = (d + K) * CHUNK / 2;
    return {m: m, K: K, d: d, pts: pts, best: best, thr: m.s / (m.k1 + m.k2 * mid) / GB / st.alpha, mid: mid};
  }

  function el(tag, attrs, text) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    return e;
  }
  function niceStep(max) {
    var raw = max / 4, p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), n = raw / p;
    return (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * p;
  }

  function draw(P) {
    var svg = $("tc-chart"), W = 680, H = 330, L = 52, R = 18, T = 16, B = 74;
    var pw = W - L - R, ph = H - T - B, K = P.K;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var ymax = 0;
    P.pts.forEach(function (p) { ymax = Math.max(ymax, p.T, p.load, st.alpha * p.rec); });
    ymax = (ymax || 1) * 1.16; var step = niceStep(ymax); ymax = Math.ceil(ymax / step) * step;
    function X(c) { return L + pw * c / K; }
    function Y(v) { return T + ph * (1 - v / ymax); }
    for (var v = 0; v <= ymax + 1e-9; v += step) {
      svg.appendChild(el("line", {x1: L, x2: W - R, y1: Y(v), y2: Y(v), "class": "grid"}));
      svg.appendChild(el("text", {x: L - 8, y: Y(v) + 4, "text-anchor": "end"}, fmt(v, step < 1 ? 1 : 0)));
    }
    svg.appendChild(el("line", {x1: L, x2: L, y1: T, y2: T + ph, "class": "ax"}));
    svg.appendChild(el("line", {x1: L, x2: W - R, y1: T + ph, y2: T + ph, "class": "ax"}));
    var lab = el("text", {x: 14, y: T + ph / 2, "text-anchor": "middle", transform: "rotate(-90 14 " + (T + ph / 2) + ")"}, "estimated cost (s)");
    svg.appendChild(lab);
    function path(key, cls) {
      var dd = P.pts.map(function (p, i) { return (i ? "L" : "M") + X(p.c).toFixed(1) + " " + Y(key(p)).toFixed(1); }).join(" ");
      svg.appendChild(el("path", {d: dd, "class": cls}));
    }
    path(function (p) { return p.load; }, "ld");
    path(function (p) { return st.alpha * p.rec; }, "rc");
    path(function (p) { return p.T; }, "tot");
    var b = P.pts[P.best];
    svg.appendChild(el("line", {x1: X(b.c), x2: X(b.c), y1: T + 16, y2: T + ph + 30, "class": "pickline"}));
    svg.appendChild(el("circle", {cx: X(b.c), cy: Y(b.T), r: 6.5, "class": "pick"}));
    /* chunk strip: tier of every chunk; the part right of the cut is recomputed */
    var sy = T + ph + 12, sh = 16;
    if (P.d > 0) svg.appendChild(el("rect", {x: X(0), y: sy, width: X(P.d) - X(0), height: sh, "class": "dram"}));
    if (P.d < K) svg.appendChild(el("rect", {x: X(P.d), y: sy, width: X(K) - X(P.d), height: sh, "class": "nvme"}));
    if (P.best < K) svg.appendChild(el("rect", {x: X(P.best), y: sy, width: X(K) - X(P.best), height: sh, "class": "cutaway"}));
    var ticks = [0, 0.25, 0.5, 0.75, 1];
    ticks.forEach(function (fr) {
      var c = fr * K;
      svg.appendChild(el("text", {x: X(c), y: sy + sh + 15, "text-anchor": fr === 0 ? "start" : fr === 1 ? "end" : "middle"}, ktok(c * CHUNK)));
    });
    svg.appendChild(el("text", {x: L + pw / 2, y: H - 6, "text-anchor": "middle"}, "cut: tokens restored (the rest is recomputed)"));
    var tx = X(b.c), anchor = tx > L + pw * 0.8 ? "end" : tx < L + pw * 0.2 ? "start" : "middle";
    svg.appendChild(el("text", {x: tx, y: T + 10, "text-anchor": anchor, style: "font-weight:700;fill:var(--tc-acc)"},
      b.c === 0 ? "chosen cut: recompute everything" : b.c === K ? "chosen cut: restore all " + ktok(K * CHUNK) + " tokens" : "chosen cut: restore " + ktok(b.c * CHUNK) + " tokens"));
  }

  function report(P) {
    var K = P.K, d = P.d, b = P.pts[P.best], all = P.pts[K], dramOnly = P.pts[d];
    function card(title, p, win) {
      return '<div' + (win ? ' class="win"' : '') + '>' + title + '<b>' + fmt(p.T) + ' s</b>read ' + fmt(p.load) + ' s + recompute ' + fmt(p.rec) + ' s' + (st.alpha !== 1 ? ' × ' + st.alpha : '') + '</div>';
    }
    var cards = card("Load everything", all, P.best === K);
    if (d > 0 && d < K) cards += card("Stop at the DRAM part", dramOnly, P.best === d);
    cards += card("Recompute everything", P.pts[0], P.best === 0);
    $("tc-out").innerHTML = cards;
    var msg;
    if (d === K) msg = "The whole prefix is in DRAM, so there is nothing to read from NVMe: the planner restores all of it.";
    else if (P.best === K) msg = "<b>Load everything.</b> Reading the " + ktok((K - d) * CHUNK) + " tokens on NVMe is cheaper than recomputing them.";
    else if (P.best === d && d > 0) msg = "<b>Cut at the end of the DRAM part.</b> The planner restores " + ktok(d * CHUNK) + " tokens from DRAM, issues no disk read and recomputes " + ktok((K - d) * CHUNK) + " tokens.";
    else if (P.best === 0) msg = "<b>Recompute everything.</b> Nothing is in DRAM and the read would take longer than the weighted prefill.";
    else msg = "<b>Cut at " + ktok(P.best * CHUNK) + " tokens.</b>";
    if (d < K) msg += " Without backlog, reading the NVMe part wins when r̂ is above <i>B</i>*(" + ktok(P.mid) + ") / α = <b>" + fmt(P.thr) + " GB/s</b>; r̂ is " + fmt(st.r) + " GB/s" + (st.q > 0 ? " and " + fmt(st.q, 1) + " GB of backlog is charged on top." : ".");
    $("tc-read").innerHTML = msg;
  }

  function sync() {
    $("tc-r").value = rToPos(st.r); $("tc-q").value = st.q; $("tc-len").value = st.len; $("tc-dram").value = st.dram;
    $("tc-r-o").textContent = fmt(st.r) + " GB/s"; $("tc-q-o").textContent = fmt(st.q, 1) + " GB";
    $("tc-len-o").textContent = st.len + "k tokens"; $("tc-dram-o").textContent = st.dram + "%";
    [["tc-model", st.model], ["tc-alpha", String(st.alpha)]].forEach(function (g) {
      [].forEach.call($(g[0]).querySelectorAll("button"), function (bt) { bt.setAttribute("aria-pressed", String(bt.getAttribute("data-v") === g[1])); });
    });
    [].forEach.call($("tc-presets").querySelectorAll("button"), function (bt) { bt.setAttribute("aria-pressed", String(bt.getAttribute("data-p") === st.preset)); });
    var P = plan(); draw(P); report(P);
  }
  $("tc-r").addEventListener("input", function (e) { st.r = posToR(+e.target.value); st.preset = ""; sync(); });
  $("tc-q").addEventListener("input", function (e) { st.q = +e.target.value; st.preset = ""; sync(); });
  $("tc-len").addEventListener("input", function (e) { st.len = +e.target.value; st.preset = ""; sync(); });
  $("tc-dram").addEventListener("input", function (e) { st.dram = +e.target.value; st.preset = ""; sync(); });
  $("tc-model").addEventListener("click", function (e) { var b = e.target.closest("button"); if (b) { st.model = b.getAttribute("data-v"); sync(); } });
  $("tc-alpha").addEventListener("click", function (e) { var b = e.target.closest("button"); if (b) { st.alpha = +b.getAttribute("data-v"); sync(); } });
  $("tc-presets").addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b) return;
    var p = PRESETS[b.getAttribute("data-p")]; st.preset = b.getAttribute("data-p");
    st.r = p.r; st.q = p.q; st.len = p.len; st.dram = p.dram; sync();
  });
  sync();
})();
</script>
