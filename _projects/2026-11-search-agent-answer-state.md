---
title: "Do Search Agents Know When They Know? Answer-State and Evidence Dynamics in RL-Trained Search Agents"
collection: projects
permalink: /projects/2026-11-search-agent-answer-state/
date: 2026-11-02
# No venue under review is named on this page; paper, code, benchmark and BibTeX links to be added once the preprint is public.
status: "Under review"
venue: "Under review"
excerpt: "Forcing an answer at every decision point of a frozen RL-trained search agent shows that agents often hold the right answer and then lose it; their confidence does not reveal when, no stopping method we evaluate gains significantly, and restating the question after every observation recovers part of the loss."
teaser: "/images/projects/2026-11-search-agent-answer-state/fig1-trajectory.png"
tags: ["search agents", "retrieval-augmented QA", "reinforcement learning", "evaluation", "benchmark"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-search-agent-answer-state"
---

<style>
/* Search-agent project page: styles scoped to .sa / .sa-* so the rest of the site is untouched; colors follow the theme */
.sa{--sa-blue:#1f5fa8;--sa-blue-soft:rgba(31,95,168,.08);--sa-gray:#6f767e;--sa-gray-soft:rgba(127,127,127,.09);--sa-green:#1f8a3b;--sa-green-soft:rgba(38,153,58,.1);--sa-red:#d2452d;--sa-red-soft:rgba(214,69,45,.09);--sa-amber:#d99114;--sa-line:rgba(127,127,127,.28);--sa-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .sa{--sa-blue:#78b0f0;--sa-blue-soft:rgba(120,176,240,.14);--sa-gray:#c3c9cf;--sa-gray-soft:rgba(255,255,255,.07);--sa-green:#62cf72;--sa-green-soft:rgba(98,207,114,.13);--sa-red:#f47f68;--sa-red-soft:rgba(244,127,104,.13);--sa-amber:#f0b648;--sa-line:rgba(255,255,255,.2);--sa-card:rgba(255,255,255,.04)}
.sa-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.sa-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--sa-blue-soft);color:var(--sa-blue)}
.sa-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.sa-tldr b{color:var(--sa-blue)}
.sa-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.sa-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.sa-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.sa-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--sa-line)}
.sa-btn.muted{background:transparent;color:var(--sa-gray)!important;border:1px dashed var(--sa-line);cursor:default}
.sa-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .sa-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.sa-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--sa-line);border-radius:14px;background:var(--sa-card)}
.sa-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.sa-fig.narrow img{max-width:min(100%,560px);margin:0 auto}
.sa-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.sa-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--sa-blue);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .sa-tag{color:#16202b}
.sa-fig figcaption b.sa-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.sa-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.sa-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--sa-line);background:var(--sa-card);position:relative;overflow:hidden}
.sa-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--sa-c,var(--sa-blue))}
.sa-stat .v{font-size:1.14em;font-weight:800;line-height:1.15;color:var(--sa-c,var(--sa-blue));font-variant-numeric:tabular-nums}
.sa-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.sa-lead{font-size:1em;line-height:1.65;margin:0 0 1em}
.sa-defs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em;margin:0 0 1.6em}
.sa-def{padding:.65em .85em;border-radius:12px;border:1px solid var(--sa-line);background:var(--sa-card);font-size:.86em;line-height:1.5}
.sa-def b{display:inline-block;margin-right:.35em;color:var(--sa-blue)}
.sa-def.wide{grid-column:1/-1}
.sa-steps{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1em}
.sa-step{padding:.9em .95em;border-radius:14px;border:1px solid var(--sa-blue);background:var(--sa-blue-soft);font-size:.86em;line-height:1.55}
.sa-step.g{border-color:var(--sa-green);background:var(--sa-green-soft)}
.sa-step .n{display:inline-grid;place-items:center;width:1.7em;height:1.7em;border-radius:50%;background:var(--sa-blue);color:#fff;font-weight:800;font-size:.9em;margin-right:.45em}
.sa-step.g .n{background:var(--sa-green)}
html[data-theme="dark"] .sa-step .n{color:#16202b}
.sa-step .h{font-size:1.05em}
.sa-step p{margin:.5em 0 0}
.sa-step .was{display:block;margin-top:.55em;font-size:.92em;color:var(--sa-gray)}
.sa-why{font-size:.9em;line-height:1.6;margin:0 0 1.6em;padding:.6em .9em;border-radius:10px;background:var(--sa-gray-soft)}
.sa-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--sa-line);border-radius:14px;background:var(--sa-card);overflow-x:auto}
.sa-table table{font-size:.8em;margin:.6em 0;min-width:560px}
.sa-table td strong{color:var(--sa-green)}
.sa-table p{font-size:.86em;margin:.2em}
.sa-how{font-size:.86em;line-height:1.55;margin:.5em .2em 0;padding:.5em .75em;border-radius:8px;background:var(--sa-gray-soft)}
.sa-ix{font-size:.9em}
.sa-ctl{display:grid;grid-template-columns:auto 1fr;gap:.35em .9em;align-items:center;margin:0 0 .8em}
.sa-ctl .l{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65}
.sa-seg{display:flex;flex-wrap:wrap;gap:.35em}
.sa-seg button{font:inherit;font-size:.86em;line-height:1.3;padding:.3em .85em;border-radius:999px;border:1px solid var(--sa-line);background:transparent;color:var(--global-text-color);cursor:pointer;transition:background .15s,border-color .15s}
.sa-seg button:hover{border-color:var(--sa-gray)}
.sa-seg button[aria-pressed=true]{background:var(--global-text-color);border-color:var(--global-text-color);color:var(--global-bg-color)}
.sa-read{margin:.8em 0 0;padding:.5em .8em;border-left:3px solid var(--sa-gray);border-radius:4px;background:var(--sa-gray-soft);font-size:.95em;line-height:1.55}
.sa-read.ok{border-left-color:var(--sa-green);background:var(--sa-green-soft)}
.sa-read.no{border-left-color:var(--sa-red);background:var(--sa-red-soft)}
.sa-sub{font-size:.74em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;opacity:.7;margin:1.1em 0 .45em}
/* trajectory stepper */
.sa-q{padding:.6em .8em;border:1px solid var(--sa-line);border-radius:10px;margin:0 0 .8em;line-height:1.5;background:var(--global-bg-color)}
.sa-q .gold{display:inline-block;margin-left:.4em;padding:.05em .55em;border-radius:999px;background:var(--sa-green-soft);color:var(--sa-green);font-weight:700;font-size:.9em;white-space:nowrap}
.sa-lane{display:grid;grid-template-columns:repeat(var(--n,4),minmax(0,1fr));gap:.55em}
.sa-col{display:flex;flex-direction:column;gap:.5em;padding:.6em;border:2px solid var(--sa-line);border-radius:12px;background:var(--global-bg-color);cursor:pointer;transition:opacity .35s,border-color .35s,box-shadow .35s}
.sa-col.future{opacity:.32}
.sa-col.cur{border-color:var(--sa-blue);box-shadow:0 4px 14px rgba(31,95,168,.18)}
.sa-col .hd{display:flex;align-items:center;gap:.45em;font-size:.82em;font-weight:700}
.sa-col .k{display:inline-grid;place-items:center;width:1.75em;height:1.75em;border-radius:50%;border:1.5px solid var(--global-text-color);font-weight:800;flex:none}
.sa-col.cur .k{background:var(--sa-blue);border-color:var(--sa-blue);color:#fff}
html[data-theme="dark"] .sa-col.cur .k{color:#16202b}
.sa-col .sr{flex:1;padding:.5em .6em;border-radius:8px;background:var(--sa-gray-soft);font-size:.82em;line-height:1.45}
.sa-col .sr .lb{display:block;font-size:.82em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;opacity:.6;margin-top:.45em}
.sa-col .sr .lb:first-child{margin-top:0}
.sa-col .frc{padding:.5em .6em;border-radius:8px;border:1px solid var(--sa-blue);background:var(--sa-blue-soft);font-size:.84em;line-height:1.4}
.sa-col .frc .lb{display:block;font-size:.8em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;color:var(--sa-blue)}
.sa-col .frc .a{font-weight:800}
.sa-col .frc.ok .a{color:var(--sa-green)}.sa-col .frc.no .a{color:var(--sa-red)}
.sa-col .frc .cb{display:block;height:5px;border-radius:999px;background:var(--sa-gray-soft);margin-top:.35em;overflow:hidden}
.sa-col .frc .cb span{display:block;height:100%;border-radius:999px;background:var(--sa-blue)}
.sa-col.future .frc .a,.sa-col.future .frc .c{visibility:hidden}
.sa-final{margin:.7em 0 0;padding:.5em .8em;border:1.5px dashed var(--sa-red);border-radius:10px;background:var(--sa-red-soft);font-size:.92em;line-height:1.5}
.sa-final b{color:var(--sa-red)}
/* bar rows */
.sa-bars .row{display:grid;grid-template-columns:minmax(0,40%) 1fr 4.2em;gap:.7em;align-items:center;margin:.38em 0}
.sa-bars .row.sub .nm{padding-left:1.2em;opacity:.85}
.sa-bars .nm{line-height:1.3}
.sa-bars .tr{position:relative;height:12px;border-radius:999px;background:var(--sa-gray-soft)}
.sa-bars .fl{position:absolute;left:0;top:0;bottom:0;border-radius:999px;transition:width .45s cubic-bezier(.2,.8,.2,1),left .45s cubic-bezier(.2,.8,.2,1)}
.sa-bars .v{font-weight:700;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.sa-bars.dv .row{grid-template-columns:minmax(0,26%) 1fr 12.5em}
.sa-bars.dv .tr{border-radius:4px}.sa-bars.dv .fl{border-radius:3px}
.sa-bars.dv .z{position:absolute;top:-4px;bottom:-4px;width:2px;margin-left:-1px;background:var(--global-text-color);opacity:.55}
.sa-bars.dv .v{font-weight:400}.sa-bars.dv .v b{font-weight:800}
.sa-bars.dv .row.grp{margin-top:.95em}.sa-bars.dv .row.grp:first-child{margin-top:.38em}
.sa-bars.dv .row.pl .nm{padding-left:1.2em;opacity:.8}
.sa-bars.dv .row.top{padding-top:.55em;margin-top:.7em;border-top:1px dashed var(--sa-line)}
.sa-axis{display:flex;justify-content:space-between;font-size:.74em;opacity:.7;margin:.3em 0 0}
.sa-stack .bar{display:flex;height:22px;border-radius:8px;overflow:hidden;background:var(--sa-gray-soft)}
.sa-stack .bar span{flex:1 1 0;min-width:0;transition:flex-grow .45s cubic-bezier(.2,.8,.2,1)}
.sa-stack .lg{display:flex;flex-wrap:wrap;gap:.3em 1.1em;margin:.55em 0 0;font-size:.92em;line-height:1.4}
.sa-stack .lg i{display:inline-block;width:.8em;height:.8em;border-radius:3px;margin-right:.4em;vertical-align:-.05em}
.sa-stack .lg b{font-variant-numeric:tabular-nums}.sa-stack .lg em{font-style:normal}
.sa-take{margin:0 0 1.2em;padding:0;list-style:none;display:grid;gap:.55em}
.sa-take li{padding:.65em .9em .65em 1em;border-left:3px solid var(--sa-green);border-radius:4px;background:var(--sa-card);font-size:.92em;line-height:1.55}
.sa-take li.scope{border-left-color:var(--sa-gray)}
.sa-code.tight{margin-bottom:.7em}
.sa-code.wrap{white-space:pre-wrap}
.sa-code{margin:0 0 1.6em;padding:.8em 1em;border-radius:12px;border:1px solid var(--sa-line);background:var(--sa-gray-soft);font-size:.76em;line-height:1.6;overflow-x:auto;white-space:pre;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;color:var(--global-text-color)}
.sa-abs{margin:0 0 1.6em;border:1px solid var(--sa-line);border-radius:12px;background:var(--sa-card)}
.sa-abs summary{cursor:pointer;padding:.6em .9em;font-weight:700;font-size:.92em}
.sa-abs p{margin:0;padding:0 .95em .9em;font-size:.9em;line-height:1.65}
.sa code{font-size:.9em}
/* the theme's fixed table of contents (220px, right) covers a full-width title: keep the header in the text column */
@media screen and (min-width:1200px){.page__inner-wrap>header{max-width:calc(100% - 240px)}}
@media (max-width:980px){.sa-stats{grid-template-columns:repeat(2,minmax(0,1fr))}.sa-steps{grid-template-columns:repeat(2,minmax(0,1fr))}.sa-lane{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.sa-tldr{font-size:1.04em}.sa-steps,.sa-defs,.sa-lane{grid-template-columns:1fr}.sa-ctl{grid-template-columns:1fr}.sa-bars .row,.sa-bars.dv .row{grid-template-columns:1fr auto;gap:.2em .6em}.sa-bars .tr{grid-column:1/-1;grid-row:2}.sa-bars .row{margin:.6em 0}.sa-col.future{display:none}}
</style>

<div class="sa">
<div class="sa-kicker"><span>Search agents</span><span>Reinforcement learning</span><span>Evaluation protocol</span><span>Benchmark</span></div>
<p class="sa-tldr">RL-trained search agents decide on their own when to stop searching. Forcing an answer at every decision point of a frozen agent shows that they <b>often hold the right answer and then lose it</b>. Their confidence does not reveal when this happens, none of the stopping methods we evaluate gains significantly, and a one-line reminder of the question recovers part of the loss.</p>
<div class="sa-btns"><a class="sa-btn" href="#protocol">The protocol</a><a class="sa-btn ghost" href="#findings">Findings</a><a class="sa-btn ghost" href="#reanchoring">Re-anchoring</a><a class="sa-btn ghost" href="#sufficiencybench">SufficiencyBench</a><span class="sa-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="sa-fig" id="sa-traj"><div class="sa-ix"><div class="sa-ctl"><span class="l">Example</span><div class="sa-seg" id="sa-ex"></div></div><div class="sa-q" id="sa-tq"></div><div class="sa-ctl"><span class="l">Stop the agent</span><div class="sa-seg" id="sa-k"></div></div><div class="sa-lane" id="sa-lane"></div><div class="sa-final" id="sa-tfinal"></div><p class="sa-read" id="sa-tread"></p></div><figcaption><span class="sa-tag" id="sa-ttag">Figure 1 · Interactive</span><b class="sa-title" id="sa-ttitle"></b>Real trajectories of Search-R1-7B. Each column is a decision point <i>k</i>: the gray box is what the agent searched and retrieved to get there, and the blue box is the answer we force from the frozen agent at that point, with its confidence. Pick a point to stop the agent there. The first example is Figure 1 of the paper; the second, from the released data, shows the more common case of an answer the agent knew before searching.</figcaption></figure>
<div class="sa-stats">
<div class="sa-stat" style="--sa-c:var(--sa-red)"><div class="v">5–8%</div><div class="d">of questions are answered correctly at some decision point but wrongly at the end, with the agents' own reasoning allowed</div></div>
<div class="sa-stat" style="--sa-c:var(--sa-gray)"><div class="v">84–93% vs. 23–30%</div><div class="d">mean confidence of the RL agents' forced answers against the share that is correct</div></div>
<div class="sa-stat"><div class="v">≤ +0.4 vs. +6.6–9.4</div><div class="d">accuracy points gained by the best stopping rule (not significant) against an oracle that stops at the first correct point</div></div>
<div class="sa-stat" style="--sa-c:var(--sa-green)"><div class="v">+0.5 to +1.7</div><div class="d">accuracy points from restating the question after every observation; a placebo reminder gives −1.3 to +0.6</div></div>
</div>
</div>

## Background {#background}

<div class="sa">
<p class="sa-lead">Search agents such as Search-R1, ZeroSearch and AutoRefine are language models trained with reinforcement learning to answer questions by interleaving reasoning with calls to a search engine. They decide on their own when to stop and answer. Every extra search costs latency and money, so many recent methods try to make them stop earlier, by penalising tool calls in the reward or by attaching a controller that stops a frozen agent once its evidence looks sufficient. These methods assume that the agent's state (its confidence, its attention to retrieved passages, its hidden representations) <b>reveals when the evidence already suffices</b>. For search agents this assumption is rarely tested, and this work tests it.</p>
<div class="sa-defs">
<div class="sa-def"><b>Decision point <i>k</i></b>The state of a run before any search (<i>k</i> = 0) and right after the <i>k</i>-th retrieved observation. At each one the agent chooses between searching again and answering.</div>
<div class="sa-def"><b>Forced answer <i>a<sub>k</sub></i></b>The answer the frozen agent gives when we append its answer tag at decision point <i>k</i>. Decoding is greedy and the prefix is identical, so this is exactly what the agent would output if stopped there.</div>
<div class="sa-def"><b>Reasoning-allowed forcing</b>A second variant that keeps the agent's own reasoning text after point <i>k</i>, up to where it would search or answer, and forces the answer there. Both variants are reported.</div>
<div class="sa-def"><b>Closed-book</b>Correct at <i>k</i> = 0: the agent would have answered correctly from its own knowledge, before any search.</div>
<div class="sa-def"><b>Abandoned question</b>The forced answer is correct at some decision point, but the agent's final answer is wrong. The agent had the answer and lost it.</div>
<div class="sa-def"><b>Oracle stop</b>Stop at the first decision point whose forced answer is correct. No real method can do this; it is the headroom a stopping method could recover.</div>
<div class="sa-def wide"><b>Agents studied</b>Four public RL-trained agents: Search-R1 (Qwen2.5-7B and 3B), ZeroSearch (Qwen2.5-7B) and AutoRefine (Qwen2.5-7B). An untrained reference, Qwen2.5-7B-Instruct with the Search-R1 prompt, shows what is there without RL, and Search-R1-14B serves as a scale check. Each agent keeps its own training-time prompt, loop protocol and search budget.</div>
</div>
</div>

## The diagnostic protocol {#protocol}

<div class="sa">
<p class="sa-lead">The protocol asks two questions at every decision point of a frozen agent: <i>what would the agent answer if it stopped now</i>, and <i>which retrieved passages actually shape that answer</i>. Because the agents decode greedily, both counterfactuals are exact, and everything is computed from one native run per question, with no training.</p>
<div class="sa-steps">
<div class="sa-step"><span class="n">1</span><b class="h">Probe the answer state</b><p>Force an answer at every decision point, with and without the agent's reasoning. Record whether it is correct and how confident the agent is.</p><span class="was">Answers: when is the answer already there, and does the agent know?</span></div>
<div class="sa-step"><span class="n">2</span><b class="h">Trace the evidence</b><p>Remove each newly retrieved passage, one at a time and jointly, and force the answer again. A passage is <i>necessary</i>, <i>harmful</i>, <i>influential</i>, <i>supportive</i> or has no effect.</p><span class="was">Answers: where does retrieved evidence go?</span></div>
<div class="sa-step"><span class="n">3</span><b class="h">Replay stopping policies</b><p>Stopping only truncates a greedy run, so any stopping policy can be replayed exactly on the recorded states, each one charged for the probes it uses.</p><span class="was">Answers: how much headroom do stopping methods recover?</span></div>
<div class="sa-step g"><span class="n">4</span><b class="h">Re-anchor</b><p>An intervention, not a diagnosis: restate the question after every observation, and compare with a placebo reminder that leaves the question out.</p><span class="was">Tests: does the question fade as the agent reads?</span></div>
</div>
<figure class="sa-fig"><img src="/images/projects/2026-11-search-agent-answer-state/fig2-protocol.png" alt="Figure 2: the diagnostic protocol in four panels: answer-state probing, causal fate of evidence, exact replay, and re-anchoring" loading="lazy"><figcaption><span class="sa-tag">Figure 2</span><b class="sa-title">The diagnostic protocol and re-anchoring.</b>The paper's overview. From one greedy run of a frozen agent, we (1) force an answer at every decision point and record its correctness and confidence, (2) remove newly retrieved passages and probe again to label their causal fate, and (3) replay any stopping policy exactly on the recorded states, which we release as SufficiencyBench. (4) Re-anchoring restates the question after every observation. The example values are those of the trajectory in Figure 1.</figcaption></figure>
<p class="sa-why"><b>Setup.</b> 6,125 questions from seven QA benchmarks (NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultihopQA, MuSiQue, Bamboogle), all agents in one retrieval environment: E5-base over the 21M passages of the Wikipedia-2018 dump, top-3 passages per query, as in Search-R1's training. Accuracy is exact match (EM). The findings are checked with a strict LLM judge, a second question sample, top-5 retrieval, sampled trajectories and Search-R1-14B.</p>
</div>

## Findings {#findings}

### 1. Agents hold the right answer and then lose it

<div class="sa">
<p class="sa-lead">For every RL-trained agent, the answer is often available long before the agent stops. Between 17.6% and 23.1% of questions are answered correctly before the first search, yet the agents still search, and 20–27% of all search calls are issued after the forced answer is already correct. On 6.8–9.1% of questions the agent is correct at some decision point and wrong at the end. When it may first write its own reasoning, 5.2–8.2% remain.</p>
<figure class="sa-fig"><div class="sa-ix"><div class="sa-ctl"><span class="l">Agent</span><div class="sa-seg sa-agents"></div></div><div class="sa-sub">Share of the 6,125 questions</div><div class="sa-bars" id="sa-states"></div><p class="sa-read" id="sa-r1"></p><div class="sa-sub">Where the wrong final answer of an abandoned question comes from</div><div class="sa-stack" id="sa-src"></div><p class="sa-read" id="sa-r1b"></p></div><figcaption><span class="sa-tag">Interactive</span><b class="sa-title">When is the answer already there, and how is it lost? (Tables 1 and 9)</b>Top: forced answers under direct forcing unless stated. "Beyond sampling" counts an abandoned question only if none of 8 answers sampled at the last decision point is correct, a conservative rate. Bottom: abandoned questions whose final answer also fails a lenient match, split by where that wrong answer appears.</figcaption></figure>
</div>

<div class="sa"><div class="sa-table" markdown="1">

<span class="sa-tag">Table 1</span> **When is the answer already there?** Excerpt; shares in % of 6,125 questions.

| Agent | EM | Searches | Correct before any search | Abandoned | with reasoning | beyond sampling | Oracle: Δ searches |
|---|---|---|---|---|---|---|---|
| Search-R1-7B | 42.0 | 2.42 | 22.0 | 9.1 | 8.2 | 5.4 | −27% |
| Search-R1-3B | 32.8 | 1.50 | 17.6 | 9.0 | 7.0 | 3.9 | −20% |
| ZeroSearch-7B | 33.8 | 1.06 | 20.4 | 7.9 | 5.2 | 3.6 | −21% |
| AutoRefine-7B | 43.6 | 1.81 | 23.1 | 6.8 | 5.3 | 3.8 | −25% |
| Qwen2.5-7B-Inst. (untrained) | 32.8 | 1.38 | 16.0 | 6.7 | 3.2 | 3.2 | −15% |

<p class="sa-how"><b>How to read it.</b> The oracle stops at the first correct decision point; its accuracy gain equals the abandoned share by construction, so stopping at the right time would raise accuracy by 6.8–9.1 points for the RL agents while cutting searches by 20–27%.</p>

Most abandoned questions were correct before any search (56–71%; 67–90% with reasoning), and in 66–76% of these the wrong final answer appears in a later passage: retrieved text overrides a correct answer the model already had. Answers found by searching and then lost, as in Figure 1, are rarer: 2.3–4.0% of questions, and 0.5–2.7% with reasoning. Sampling at the final state recovers about half of the oracle's gain, which leaves 3–5% of questions lost beyond what sampling recovers.

</div></div>

### 2. Single passages rarely decide the answer

<div class="sa">
<p class="sa-lead">Removing one newly retrieved passage has no marginal effect on the agent's current answer for 70–80% of passages across the four RL agents. Only 2.6–4.2% are necessary (removing them turns a correct answer wrong) and 1.6–3.0% are harmful (removing them turns a wrong answer right). Gold supporting passages behave the same way, because the answer was already settled or another passage carries the same fact.</p>
<figure class="sa-fig"><div class="sa-ix"><div class="sa-ctl"><span class="l">Agent</span><div class="sa-seg sa-agents"></div></div><div class="sa-sub">Effect of removing one newly retrieved passage on the current forced answer</div><div class="sa-stack" id="sa-fate"></div><p class="sa-read" id="sa-r2"></p></div><figcaption><span class="sa-tag">Interactive</span><b class="sa-title">Causal fate of retrieved passages (Table 2).</b>Each passage of the newest observation is replaced by an omission marker and the answer is forced again. <i>Influential</i>: the answer changes without flipping correctness. <i>Supportive</i>: the answer stays but its log-probability drops by more than one nat. Measured on 300 questions per dataset.</figcaption></figure>
<p class="sa-lead">Single removal cannot see redundancy, so we also remove whole observations. That changes the forced answer in 46–64% of observations, and for 20–23% of observations only a joint removal changes it. Passages matter together more than one by one. These measures concern the current answer only: a passage can still steer the next query or the final answer.</p>
</div>

### 3. Confidence does not tell right from wrong

<div class="sa">
<p class="sa-lead">A stopping rule needs the agent's state to reveal whether its current answer is right. Two properties matter. <b>Calibration</b> is poor but repairable: forced answers are 84–93% confident and 23–30% correct, and Platt scaling fitted on other questions lowers the calibration error from 0.54–0.67 to 0.05 or less. <b>Discrimination</b>, which a stopping rule needs and recalibration cannot change, is only moderate.</p>
<figure class="sa-fig"><div class="sa-ix"><div class="sa-ctl"><span class="l">Agent</span><div class="sa-seg sa-agents"></div></div><div class="sa-sub">Forced answers over all decision points</div><div class="sa-bars" id="sa-cal"></div><div class="sa-sub">How well each signal separates right from wrong answers (AUC)</div><div class="sa-bars" id="sa-auc"></div><div class="sa-axis"><span></span><span>bars run from 0.5 (chance) to 1.0 (perfect)</span><span></span></div><p class="sa-read" id="sa-r3"></p></div><figcaption><span class="sa-tag">Interactive</span><b class="sa-title">Do agents know when they know? (Table 3)</b>Confidence is the geometric mean of the answer's token probabilities. Agreement is the share of 8 sampled answers that match the forced one. The probe is a linear classifier on the agent's hidden state, trained on a disjoint pool of questions.</figcaption></figure>
<p class="sa-lead">Sampling does not help either: all eight samples coincide at 40–65% of the RL agents' decision points, including 31–57% of those with a wrong answer. The untrained reference is equally miscalibrated, but its uncertainty is more informative (agreement AUC 0.82 against 0.66–0.72, probe 0.84 against 0.74–0.77). With a single untrained model, this is only suggestive evidence that outcome-reward RL collapses the variability that uncertainty-based stopping would need.</p>
</div>

### 4. No stopping method recovers the headroom

<div class="sa">
<p class="sa-lead">We replay each stopping method exactly on the recorded states. The label-free rules trace nearly the same accuracy–cost curve as a fixed search budget. None improves accuracy by more than 0.4 points and none of the gains is significant, whereas 13 of the 36 comparisons with the native agent are significant losses after Holm correction. The oracle gains 6.6–9.4 points while saving 16–27% of searches.</p>
<figure class="sa-fig"><div class="sa-ix"><div class="sa-ctl"><span class="l">Agent</span><div class="sa-seg sa-agents"></div></div><div class="sa-sub">Change in exact match against the native agent (points)</div><div class="sa-bars dv" id="sa-stop"></div><div class="sa-axis" id="sa-stopaxis"></div><p class="sa-read" id="sa-r4"></p></div><figcaption><span class="sa-tag">Interactive</span><b class="sa-title">Stopping methods replayed on SufficiencyBench (Table 4).</b><b>Budget-2</b>: at most two searches. <b>Conf.+stable</b>: stop when confidence ≥ 0.95 and the forced answer equals the previous one. <b>Agree+stable</b>: all 8 samples agree and the answer is stable. <b>Probe</b>: hidden-state probe ≥ 0.5. <b>Probe (closed-book)</b>: probe ≥ 0.8, also allowed to answer before the first search. <b>HALT</b>: an external verifier attached to the same trajectories; our setting differs from the original, so this does not test HALT as published. Stars: paired exact McNemar test against the native agent, Holm-adjusted (* <i>p</i> &lt; 0.05, ** <i>p</i> &lt; 0.01, *** <i>p</i> &lt; 0.001). 300 questions per dataset.</figcaption></figure>
<figure class="sa-fig narrow"><img src="/images/projects/2026-11-search-agent-answer-state/fig3-pareto-sr1-7b.png" alt="Figure 3: exact-match accuracy against searches per question for Search-R1-7B; threshold sweeps of the stopping signals lie on the fixed-budget curve, HALT below it and the oracle far above" loading="lazy"><figcaption><span class="sa-tag">Figure 3</span><b class="sa-title">Accuracy–cost frontier of stopping methods for Search-R1-7B.</b>Each curve sweeps the threshold of one signal; "stable" requires the same forced answer at the previous decision point. The sweeps stay on the fixed-budget curve next to the native agent, and the oracle sits far above all of them. The best operating points save 15–26% of searches at a loss of at most half a point.</figcaption></figure>
<p class="sa-lead">Supervised baselines do no better. A learned stopping head reaches AUC 0.78–0.86 for "answering now is correct", but under risk control it never stops early, and its best threshold in hindsight gains at most 0.4 points. Search-R1-14B is more accurate than the 7B agent and repeats every pattern: 7.4% of its questions are abandoned, and every stopping method loses accuracy.</p>
</div>

## Re-anchoring: keeping the question in view {#reanchoring}

<div class="sa">
<p class="sa-lead">More than half of the abandoned questions end with an answer taken from a passage retrieved after the agent was already correct, and in Figure 1 the agent ends up answering a question next to the one it was asked. <b>Online anchoring</b> restates the question after every observation and lets the agent continue as usual:</p>
<div class="sa-code wrap">&lt;information&gt; ...retrieved passages... &lt;/information&gt;
(Reminder: the question is "What United Kingdom Prime Minister was succeeded in the 1990s by the Leader of the Conservative Party?")

<span style="opacity:.6"># placebo, same form and position, without the question</span>
(Reminder: read the information carefully before you continue.)</div>
<p class="sa-lead">We use the reminder as a probe of the mechanism behind abandonment, not as a new method. It raises the accuracy of every agent, significantly for five of six. The placebo reminder does not help, so the gain comes from the question itself.</p>
<figure class="sa-fig"><div class="sa-ix"><div class="sa-sub">Change in exact match against the native agent (points, 6,125 questions)</div><div class="sa-bars dv" id="sa-anchor"></div><div class="sa-axis" id="sa-anchoraxis"></div></div><figcaption><span class="sa-tag">Table 5</span><b class="sa-title">The question reminder helps; a placebo reminder does not.</b>Green: reminder with the question. Gray: placebo reminder without it. Stars: paired exact McNemar test against the native agent (* <i>p</i> &lt; 0.05, ** <i>p</i> &lt; 0.01, *** <i>p</i> &lt; 0.001). The reminder beats the placebo for every agent, by 0.6–2.9 points (Holm-adjusted <i>p</i> ≤ 0.041).</figcaption></figure>
</div>

<div class="sa"><div class="sa-table" markdown="1">

<span class="sa-tag">Table 5</span> **Online anchoring on the 6,125 test questions.** Changes against the native agent.

| Agent | Native EM | Reminder: Δ EM | Δ searches | Placebo: Δ EM | Reminder − placebo |
|---|---|---|---|---|---|
| Search-R1-7B | 42.8 | **+1.7**\*\*\* | +5% | +0.3 | +1.4\*\*\* |
| Search-R1-3B | 32.9 | **+1.6**\*\*\* | −3% | +0.6 | +1.1\* |
| ZeroSearch-7B | 33.8 | +0.5 | −2% | −0.3 | +0.8\* |
| AutoRefine-7B | 44.0 | **+0.9**\*\* | −3% | +0.3 | +0.6\* |
| Search-R1-14B | 45.6 | **+1.2**\*\*\* | 0% | +0.2 | +1.0\*\* |
| Qwen2.5-7B-Inst. (untrained) | 33.0 | **+1.6**\*\*\* | +4% | −1.3\*\* | +2.9\*\*\* |

<p class="sa-how"><b>How to read it.</b> Native EM here scores a run that ends without an answer by its last forced answer, so it differs slightly from Table 1. Green marks significant gains. The reminder changes the number of searches by −3% to +5%: it is not a stopping rule.</p>

**No measurable query drift.** If the agents' queries drifted away from the question, the queries of abandoned questions should move away from it and the reminder should change them. Neither happens: after the first correct point, the similarity to the question of the queries of abandoned and retained questions differs by −0.006 to +0.002, every 95% interval covers zero, and anchoring leaves query similarity unchanged. Our query measures cannot detect a changed relation, such as the query for Thatcher's successor in Figure 1, so we conclude only that the queries show no measurable drift, which points to the agents' reasoning over what they have read.

**Distilling the reminder.** Fine-tuning Search-R1-7B on its own reminded trajectories, with the reminder removed, gains 1.0 points with no reminder at test time and beats a self-training control by 0.95 points (<i>p</i> = 0.004). Training against abandonment directly with preference optimization lowered abandonment but taught premature stopping.

</div></div>

## SufficiencyBench {#sufficiencybench}

<div class="sa">
<p class="sa-lead">We release the decision states behind these findings as <b>SufficiencyBench</b>: 97,841 decision points on 6,125 questions from six agents (the four RL agents, HiPRAG and the untrained reference). Each point records the question and gold answers, the query and retrieved passages, the forced answers with and without reasoning and their oracle labels (string match and LLM judge), the outcome and token cost of the native continuation, and the baseline signals studied here.</p>
<div class="sa-defs">
<div class="sa-def"><b>Exact replay</b>The evaluator replays a stop/continue policy in order, charges every forced answer it reads, and reports accuracy, searches, tokens and latency. Under greedy decoding this equals re-running the agent under the policy.</div>
<div class="sa-def"><b>Fixed split</b>Methods are fit on a calibration pool (train), may be tuned on 4,200 questions outside the replay subset (dev), and are evaluated once on the replay subset of 1,925 questions (test).</div>
<div class="sa-def"><b>Cost</b>A new stopping method is compared with the others on identical states in seconds on a CPU, without running an agent or a retriever.</div>
<div class="sa-def"><b>Scope</b>Replay covers policies that truncate the native trajectory. Interventions such as re-anchoring need new rollouts; our native, anchored and placebo runs are released for comparison.</div>
</div>
<p class="sa-lead">One line per decision point. This is the record behind the correct answer in Figure 1 (abridged, values rounded):</p>
<div class="sa-code">{"agent": "sr1_7b", "qid": "hotpotqa-dev_6008", "dataset": "hotpotqa", "k": 2,
 "question": "What United Kingdom Prime Minister was succeeded in the 1990s by the Leader of the Conservative Party?",
 "golds": ["Margaret Thatcher"],
 "query": "Leader of the Conservative Party in the 1990s",
 "docs": [{"id": "6451293", "title": "1990 Conservative Party (UK) leadership election"}, ...],
 "answer_now": "Margaret Thatcher", "correct_now": 1.0,        <span style="opacity:.6"># forced answer at this point and its oracle label</span>
 "answer_final": "John Major", "correct_final": 0.0,           <span style="opacity:.6"># what happens if the agent continues</span>
 "n_search_native": 3, "prefill_so_far": 665, "decode_so_far": 137,
 "signals": {"conf": 0.960, "stable": false, "agree8": 1.0, "probe_p": 0.144},
 "split": "dev", ...}</div>
<p class="sa-lead">A policy sees one decision point at a time, in order, and the first <code>True</code> ends the episode with the forced answer. The evaluator is a single script and will be released with the data:</p>
<div class="sa-code tight">python evaluate.py data/sr1_7b.jsonl.gz --policy budget:2
python evaluate.py data/sr1_7b.jsonl.gz --policy conf_stable:0.95
python evaluate.py data/sr1_7b.jsonl.gz --policy oracle --closed_book --coef latency/7b.json</div>
<div class="sa-code">from evaluate import load, evaluate
states = load("data/sr1_7b.jsonl.gz")
my_policy = lambda s: s["k"] >= 1 and s["signals"]["probe_p"] >= 0.6
print(evaluate(states, my_policy, closed_book=False))</div>
</div>

<div class="sa"><div class="sa-table" markdown="1">

<span class="sa-tag">Contents</span> **Decision states per agent**, with the two reference rows a new method should be compared with (all 6,125 questions).

| Agent | Decision states | Native accuracy | Oracle stop |
|---|---|---|---|
| Search-R1-7B | 20,949 | 42.0 | 51.0 |
| Search-R1-3B | 15,323 | 32.8 | 41.8 |
| ZeroSearch-7B | 12,648 | 33.8 | 41.7 |
| AutoRefine-7B | 17,216 | 43.6 | 50.4 |
| HiPRAG-3B | 17,128 | 51.1 | 60.5 |
| Qwen2.5-7B-Inst. (untrained) | 14,577 | 32.8 | 39.4 |

<p class="sa-how"><b>How to read it.</b> Accuracy is exact match, except for HiPRAG, which answers in free-form sentences and is scored by whether the answer contains a gold answer. The oracle stops at the first correct decision point, before any search if the agent is already correct.</p>

</div></div>

## Takeaways {#takeaways}

<div class="sa">
<ul class="sa-take">
<li><b>Stopping can raise accuracy, not only save searches.</b> Most lost answers were known before the first search, and later searches often bring in the passage whose answer the agent ends with. Deciding whether and when to search is worth up to 9 points for these agents.</li>
<li><b>Confidence needs discrimination, not recalibration.</b> Recalibration fixes the level of the agents' confidence but cannot add the ranking it lacks. We suggest training agents or light heads on the oracle labels of SufficiencyBench to predict whether the answer at the current point is correct, and evaluating them by exact replay.</li>
<li><b>Keep the question in view.</b> A one-line reminder, which is not a stopping rule, recovers up to a quarter of the headroom. It could enter RL rollouts; preference optimization at decision points taught premature stopping or changed little.</li>
<li><b>Prune passages jointly.</b> Removing a single passage rarely changes the current answer, but whole observations matter. Pruning should remove only passages that have no effect also when removed together with others; detecting passages that would overturn an already correct answer is a second target.</li>
<li class="scope"><b>Scope.</b> Short-answer QA over a Wikipedia snapshot, with agents built on Qwen2.5 (3B to 14B); deep-research agents on the live Web may behave differently. Forced answers cannot show the effect of a search the agent did not make, and exact replay requires greedy decoding. Re-anchoring used one wording, and each training run a single seed.</li>
</ul>
<details class="sa-abs"><summary>Abstract</summary><p>Search agents trained with reinforcement learning (RL) answer questions by querying a search engine and reading the returned pages until they decide to stop. Methods that make them stop earlier assume that the agent's own state reveals when its evidence suffices. We test this assumption with an exact diagnostic protocol. At every decision point of a frozen agent, we force the answer it would give if stopped, with or without its own reasoning, measure how retrieved passages change that answer, and replay any stopping policy offline. On seven QA benchmarks, public RL-trained agents often hold the right answer and then lose it: with reasoning allowed, 5–8% of questions are answered correctly at some point but wrongly at the end, two thirds or more of them already correct before any search, and 3–5% beyond what sampling at the final state recovers. Single retrieved passages rarely decide the current answer on their own. The agents' confidence, sample agreement and hidden-state probes separate right from wrong answers only moderately, and none of the stopping methods we evaluate gains significantly. Restating the question after every observation raises accuracy by 0.5–1.7 points (significantly for five of six agents), whereas a placebo reminder does not, and we find no measurable query drift, which points to the agents' reasoning rather than their search. These findings hold under an LLM judge and for a 14B agent. We release SufficiencyBench, 97,841 decision states with oracle labels and an exact replay harness.</p></details>
</div>

## Paper and code {#paper-and-code}

Paper, code, SufficiencyBench and BibTeX will be linked here once the preprint is public.

{% raw %}
<script>
(function () {
  /* numbers are those of the paper's tables */
  var A = [
    {name: "Search-R1-7B", acc: 42.0, cb: 22.0, reached: 49.1, ab: 9.1, abr: 8.2, abs: 5.4, osrch: 27, strict: 7.0, src: [55.8, 9.1, 11.2, 24.0],
     fate: [2.6, 1.7, 15.2, 0.4, 80.1], n: "14,043", obs: 46.5, obsnec: 9.4, joint: 19.9,
     cacc: 28.2, conf: 93.2, ece: 0.649, platt: 0.036, auc: 0.727, agree: 0.656, probe: 0.767, coll: 65.1, collw: 56.7,
     em: 40.2, stop: [[-0.5, 26, ""], [-0.4, 20, ""], [-0.3, 20, ""], [0.3, 15, ""], [-0.3, 15, ""], [-1.7, 13, "**"], [9.4, 27, ""]]},
    {name: "Search-R1-3B", acc: 32.8, cb: 17.6, reached: 38.6, ab: 9.0, abr: 7.0, abs: 3.9, osrch: 20, strict: 7.2, src: [58.6, 18.6, 20.2, 2.5],
     fate: [4.2, 2.7, 21.6, 0.8, 70.7], n: "8,559", obs: 59.9, obsnec: 12.2, joint: 19.8,
     cacc: 23.4, conf: 86.1, ece: 0.627, platt: 0.048, auc: 0.755, agree: 0.720, probe: 0.765, coll: 39.8, collw: 31.3,
     em: 32.3, stop: [[0.0, 3, ""], [-0.6, 4, ""], [-0.4, 5, ""], [-1.3, 5, ""], [-0.4, 6, ""], [-1.2, 4, ""], [8.7, 20, ""]]},
    {name: "ZeroSearch-7B", acc: 33.8, cb: 20.4, reached: 37.0, ab: 7.9, abr: 5.2, abs: 3.6, osrch: 21, strict: 5.8, src: [57.6, 12.4, 29.8, 0.3],
     fate: [3.7, 3.0, 22.4, 0.7, 70.2], n: "6,024", obs: 64.2, obsnec: 12.3, joint: 21.0,
     cacc: 24.6, conf: 91.2, ece: 0.665, platt: 0.046, auc: 0.729, agree: 0.684, probe: 0.741, coll: 55.7, collw: 47.5,
     em: 32.4, stop: [[-1.6, 0, "**"], [-1.3, 3, "*"], [-1.2, 3, "*"], [-1.0, 2, ""], [-1.3, 10, ""], [-1.2, 2, ""], [8.2, 21, ""]]},
    {name: "AutoRefine-7B", acc: 43.6, cb: 23.1, reached: 48.5, ab: 6.8, abr: 5.3, abs: 3.8, osrch: 25, strict: 5.0, src: [54.4, 14.6, 15.9, 15.2],
     fate: [3.6, 1.6, 14.5, 0.3, 79.9], n: "9,870", obs: 52.4, obsnec: 13.7, joint: 22.5,
     cacc: 29.5, conf: 83.9, ece: 0.544, platt: 0.049, auc: 0.783, agree: 0.719, probe: 0.773, coll: 48.8, collw: 37.6,
     em: 43.3, stop: [[-1.4, 7, ""], [-0.7, 10, ""], [-0.6, 9, ""], [0.2, 9, ""], [-1.4, 14, ""], [-2.5, 7, "***"], [7.0, 26, ""]]},
    {name: "Qwen2.5-7B-Inst.", note: "untrained", acc: 32.8, cb: 16.0, reached: 32.8, ab: 6.7, abr: 3.2, abs: 3.2, osrch: 15, strict: 4.2, src: [40.5, 12.4, 35.9, 11.2],
     fate: [3.1, 2.3, 41.4, 1.0, 52.2], n: "7,743", obs: 74.7, obsnec: 9.0, joint: 17.0,
     cacc: 18.0, conf: 85.1, ece: 0.671, platt: 0.029, auc: 0.791, agree: 0.816, probe: 0.838, coll: 31.8, collw: 22.9,
     em: 31.8, stop: [[-1.7, 12, "**"], [0.1, 1, ""], [-0.2, 2, ""], [-1.7, 2, "***"], [-0.5, 6, ""], [-3.0, 5, "***"], [6.6, 16, ""]]}
  ];
  var METHODS = ["Budget-2", "Conf.+stable", "Agree+stable", "Probe", "Probe (closed-book)", "HALT", "Oracle stop"];
  var R = [
    ["Search-R1-7B", 1.7, "***", 0.3, ""], ["Search-R1-3B", 1.6, "***", 0.6, ""], ["ZeroSearch-7B", 0.5, "", -0.3, ""],
    ["AutoRefine-7B", 0.9, "**", 0.3, ""], ["Search-R1-14B", 1.2, "***", 0.2, ""], ["Qwen2.5-7B-Inst. (untrained)", 1.6, "***", -1.3, "**"]
  ];
  var EX = [
    {name: "Found by searching, then lost", tag: "Figure 1 · Interactive", title: "A question that Search-R1-7B answers correctly and then loses.",
     q: "What United Kingdom Prime Minister was succeeded in the 1990s by the Leader of the Conservative Party?", ds: "HotpotQA", gold: "Margaret Thatcher",
     fin: "Left alone, the agent stops after its third search and answers <b>John Major ✗</b>.",
     T: [
       {t: "before any search", a: "John Major", ok: 0, c: "0.99"},
       {t: "after search 1", q: "United Kingdom Prime Minister succeeded in the 1990s by the Leader of the Conservative Party", d: "Conservative Party (UK); John Major (two passages)", a: "John Major", ok: 0, c: "1.00"},
       {t: "after search 2", q: "Leader of the Conservative Party in the 1990s", d: "1990 Conservative Party (UK) leadership election; Conservative Party (UK) (two passages)", a: "Margaret Thatcher", ok: 1, c: "0.96"},
       {t: "after search 3", q: "Who succeeded Margaret Thatcher as Prime Minister", d: "Margaret Thatcher (three passages): “…Thatcher was replaced as Prime Minister and party leader by Chancellor John Major…”", a: "John Major", ok: 0, c: "1.00"}
     ],
     R: [
       "Stopped <b>before any search</b>, the agent answers <b>John Major</b>: wrong, at confidence 0.99.",
       "After the first search the forced answer is still <b>John Major</b>, now at confidence 1.00.",
       "After the second search the forced answer is <b>Margaret Thatcher</b>: correct. An oracle would stop here, one search early. The agent's confidence is 0.96, its lowest of the run.",
       "The agent searches for Thatcher's successor, reads that John Major replaced her, and ends with <b>John Major</b> at confidence 1.00: wrong, and more confident than when it was right."
     ]},
    {name: "Known before searching, then overwritten", tag: "Interactive", title: "A question that Search-R1-7B answers correctly before any search, and then loses to what it retrieves.",
     q: "Which film was released more recently, Make Mine Music or Old Dogs?", ds: "HotpotQA", gold: "Old Dogs",
     fin: "Left alone, the agent stops after its second search and answers <b>Make Mine Music ✗</b>.",
     T: [
       {t: "before any search", a: "Old Dogs", ok: 1, c: "0.954"},
       {t: "after search 1", q: "Make Mine Music", d: "Make Mine Music (three passages): “…a 1946 American animated anthology film produced by Walt Disney…”", a: "Make Mine Music", ok: 0, c: "1.000"},
       {t: "after search 2", q: "Old Dogs", d: "Aging in dogs (two passages); Old Dog Haven", a: "Make Mine Music", ok: 0, c: "0.995"}
     ],
     R: [
       "Stopped <b>before any search</b>, the agent answers <b>Old Dogs</b>: correct, at confidence 0.954. It searches anyway.",
       "The first search returns three passages on <i>Make Mine Music</i> and nothing on <i>Old Dogs</i>. The forced answer flips to <b>Make Mine Music</b> at confidence 1.000. With its own reasoning allowed, the agent would still answer Old Dogs at this point.",
       "The query “Old Dogs” retrieves passages about ageing dogs, not the film. The agent writes “Old Dogs is a film about aging in dogs, not a movie release” and answers <b>Make Mine Music</b>: a correct answer it already had is overwritten by what it retrieved."
     ]}
  ];
  var C = {blue: "var(--sa-blue)", green: "var(--sa-green)", red: "var(--sa-red)", gray: "var(--sa-gray)", amber: "var(--sa-amber)", ink: "var(--global-text-color)"};
  function $(id) { return document.getElementById(id); }
  function sg(v, d) { return (v < 0 ? "−" : "+") + Math.abs(v).toFixed(d === undefined ? 1 : d); }

  /* rows of [label, value, color, sub?]: built once, then updated so the bars animate */
  function bars(el, rows, lo, hi, dec, unit) {
    if (!el.children.length) el.innerHTML = rows.map(function (r) { return '<div class="row' + (r[3] ? " sub" : "") + '"><span class="nm"></span><span class="tr"><span class="fl"></span></span><span class="v"></span></div>'; }).join("");
    rows.forEach(function (r, i) {
      var e = el.children[i], f = e.querySelector(".fl");
      e.querySelector(".nm").textContent = r[0];
      f.style.width = Math.max(1, 100 * (r[1] - lo) / (hi - lo)) + "%"; f.style.background = r[2];
      e.querySelector(".v").textContent = r[1].toFixed(dec) + unit;
    });
  }
  /* parts of [label, value, color] */
  function stack(el, parts) {
    if (!el.children.length) el.innerHTML = '<div class="bar">' + parts.map(function () { return "<span></span>"; }).join("") + '</div><div class="lg">' + parts.map(function () { return "<span><i></i><b></b> <em></em></span>"; }).join("") + "</div>";
    var segs = el.firstChild.children, lg = el.lastChild.children;
    parts.forEach(function (p, i) {
      segs[i].style.flexGrow = p[1]; segs[i].style.background = p[2]; segs[i].title = p[0] + ": " + p[1].toFixed(1) + "%";
      lg[i].querySelector("i").style.background = p[2]; lg[i].querySelector("b").textContent = p[1].toFixed(1) + "%"; lg[i].querySelector("em").textContent = p[0];
    });
  }
  /* rows of [label, value, color, text, class] around a zero line */
  function diverge(el, rows, lo, hi) {
    var z = 100 * (0 - lo) / (hi - lo);
    if (!el.children.length) el.innerHTML = rows.map(function () { return '<div class="row"><span class="nm"></span><span class="tr"><span class="fl"></span><span class="z"></span></span><span class="v"></span></div>'; }).join("");
    rows.forEach(function (r, i) {
      var e = el.children[i], f = e.querySelector(".fl"), w = 100 * Math.abs(r[1]) / (hi - lo);
      e.className = "row" + (r[4] ? " " + r[4] : "");
      e.querySelector(".nm").innerHTML = r[0]; e.querySelector(".z").style.left = z + "%";
      f.style.left = (r[1] < 0 ? z - w : z) + "%"; f.style.width = Math.max(0.5, w) + "%"; f.style.background = r[2];
      e.querySelector(".v").innerHTML = r[3];
    });
  }
  function axis(el, lo, hi) { el.innerHTML = "<span>" + sg(lo) + "</span><span>0 = the native agent</span><span>" + sg(hi) + "</span>"; }

  var cur = 0;
  function label(a) { return a.name + (a.note ? " (" + a.note + ")" : ""); }
  function draw() {
    var a = A[cur], nm = "<b>" + a.name + "</b>" + (a.note ? " (the untrained reference)" : "");
    document.querySelectorAll(".sa-agents").forEach(function (el) {
      if (!el.children.length) A.forEach(function (x, i) { var b = document.createElement("button"); b.type = "button"; b.textContent = label(x); b.onclick = function () { cur = i; draw(); }; el.appendChild(b); });
      Array.prototype.forEach.call(el.children, function (b, i) { b.setAttribute("aria-pressed", i === cur); });
    });

    bars($("sa-states"), [
      ["Correct before any search (closed-book)", a.cb, C.gray],
      ["Correct at some decision point", a.reached, C.blue],
      ["Final answer correct (accuracy)", a.acc, C.ink],
      ["Abandoned: correct at some point, wrong at the end", a.ab, C.red],
      ["with the agent's reasoning allowed", a.abr, C.red, 1],
      ["beyond sampling", a.abs, C.red, 1]
    ], 0, 55, 1, "%");
    $("sa-r1").innerHTML = nm + " answers " + a.cb.toFixed(1) + "% of questions correctly before any search and " + a.acc.toFixed(1) + "% at the end. On <b>" + a.ab.toFixed(1) + "%</b> it held the right answer and lost it (" + a.abr.toFixed(1) + "% with reasoning allowed). An oracle that stops at the first correct point would gain " + a.ab.toFixed(1) + " points with " + a.osrch + "% fewer searches.";
    stack($("sa-src"), [
      ["in a passage retrieved after the agent was already correct", a.src[0], C.red],
      ["only in an earlier passage", a.src[1], C.amber],
      ["in no retrieved passage", a.src[2], C.blue],
      ["the run ends without an answer", a.src[3], C.gray]
    ]);
    $("sa-r1b").innerHTML = "Of " + a.name + "'s abandoned questions under the strict count (" + a.strict.toFixed(1) + "% of all questions), <b>" + a.src[0].toFixed(1) + "%</b> end with a wrong answer that appears verbatim in a passage retrieved <i>after</i> the agent was already correct." +
      (a.note ? " For the untrained reference more losses have no textual support." : "");

    stack($("sa-fate"), [
      ["no marginal effect", a.fate[4], C.gray], ["influential", a.fate[2], C.green], ["necessary", a.fate[0], C.blue],
      ["harmful", a.fate[1], C.red], ["supportive", a.fate[3], C.amber]
    ]);
    $("sa-r2").innerHTML = nm + ", " + a.n + " passages: removing one passage has no marginal effect on the current answer for <b>" + a.fate[4].toFixed(1) + "%</b> of them. Removing the whole observation changes the answer in <b>" + a.obs.toFixed(1) + "%</b> of observations and turns a correct answer wrong in " + a.obsnec.toFixed(1) + "%; in " + a.joint.toFixed(1) + "% only a joint removal changes it.";

    bars($("sa-cal"), [["Mean confidence", a.conf, C.red], ["Share of forced answers that are correct", a.cacc, C.green]], 0, 100, 1, "%");
    bars($("sa-auc"), [["Confidence", a.auc, C.blue], ["Agreement among 8 samples", a.agree, C.blue], ["Hidden-state probe", a.probe, C.blue]], 0.5, 1, 2, "");
    $("sa-r3").innerHTML = nm + ": calibration error " + a.ece.toFixed(2) + ", and " + a.platt.toFixed(2) + " after Platt scaling. Recalibration fixes the level of the confidence but not its ranking of right and wrong answers. All 8 samples coincide at " + a.coll.toFixed(1) + "% of decision points, and at " + a.collw.toFixed(1) + "% of those with a wrong answer.";

    var best = -1;
    a.stop.forEach(function (s, i) { if (i < 6 && (best < 0 || s[0] > a.stop[best][0])) best = i; });
    diverge($("sa-stop"), a.stop.map(function (s, i) {
      var orc = i === 6;
      return [orc ? "<b>" + METHODS[i] + "</b>" : METHODS[i], s[0], orc ? C.blue : s[0] > 0 ? C.green : s[0] < 0 ? C.red : C.gray,
        "<b>" + sg(s[0]) + s[2] + "</b> EM · " + (s[1] ? "−" + s[1] : "0") + "% searches", orc ? "top" : ""];
    }), -3.5, 10);
    axis($("sa-stopaxis"), -3.5, 10);
    var b = a.stop[best], o = a.stop[6];
    $("sa-r4").innerHTML = nm + ", native EM " + a.em.toFixed(1) + " on the replay subset. " +
      (b[0] > 0 ? "The best rule, " + METHODS[best] + ", gains " + sg(b[0]) + " points, which is not significant. " : "No rule gains accuracy. ") +
      "The oracle gains <b>" + sg(o[0]) + "</b> points with " + o[1] + "% fewer searches.";
  }

  /* re-anchoring: reminder and placebo for the six agents */
  var rows = [];
  R.forEach(function (r) {
    rows.push(["<b>" + r[0] + "</b> · reminder", r[1], C.green, "<b>" + sg(r[1]) + r[2] + "</b>", "grp"]);
    rows.push(["placebo", r[3], C.gray, sg(r[3]) + r[4], "pl"]);
  });
  diverge($("sa-anchor"), rows, -1.5, 2);
  axis($("sa-anchoraxis"), -1.5, 2);
  draw();

  /* trajectories: stop the agent at decision point k */
  var ex = 0, k = 0, lane = $("sa-lane"), touched = false, timer = null;
  function buildT() {
    var E = EX[ex], T = E.T;
    $("sa-tq").innerHTML = "<b>Question</b> (" + E.ds + "): <i>" + E.q + '</i> <span class="gold">Gold: ' + E.gold + "</span>";
    $("sa-tfinal").innerHTML = E.fin; $("sa-ttag").textContent = E.tag; $("sa-ttitle").textContent = E.title;
    lane.style.setProperty("--n", T.length);
    lane.innerHTML = T.map(function (s, i) {
      return '<div class="sa-col"><div class="hd"><span class="k">' + i + "</span><span>" + s.t + '</span></div><div class="sr">' +
        (s.q ? '<span class="lb">Query</span><i>' + s.q + '</i><span class="lb">Retrieved</span>' + s.d : '<span class="lb">No search yet</span>The agent answers from its own knowledge.') +
        '</div><div class="frc ' + (s.ok ? "ok" : "no") + '"><span class="lb">Forced answer a<sub>' + i + '</sub></span><span class="a">' + s.a + (s.ok ? " ✓" : " ✗") + '</span><br><span class="c">confidence ' + s.c + '</span><span class="cb"><span style="width:' + 100 * s.c + '%"></span></span></div></div>';
    }).join("");
    Array.prototype.forEach.call(lane.children, function (c, i) { c.onclick = function () { k = i; drawT(); }; });
    var seg = $("sa-k"); seg.innerHTML = "";
    T.forEach(function (s, i) { var b = document.createElement("button"); b.type = "button"; b.textContent = s.t; b.onclick = function () { k = i; drawT(); }; seg.appendChild(b); });
    var se = $("sa-ex");
    if (!se.children.length) EX.forEach(function (x, i) { var b = document.createElement("button"); b.type = "button"; b.textContent = x.name; b.onclick = function () { ex = i; k = EX[i].T.length - 1; buildT(); }; se.appendChild(b); });
    Array.prototype.forEach.call(se.children, function (b, i) { b.setAttribute("aria-pressed", i === ex); });
    drawT();
  }
  function drawT() {
    var E = EX[ex];
    Array.prototype.forEach.call($("sa-k").children, function (b, i) { b.setAttribute("aria-pressed", i === k); });
    Array.prototype.forEach.call(lane.children, function (c, i) { c.className = "sa-col" + (i === k ? " cur" : i > k ? " future" : ""); });
    var r = $("sa-tread"); r.className = "sa-read " + (E.T[k].ok ? "ok" : "no"); r.innerHTML = E.R[k];
  }
  function stop() { touched = true; if (timer) { clearInterval(timer); timer = null; } }
  $("sa-traj").addEventListener("pointerdown", stop, true);
  $("sa-traj").addEventListener("keydown", stop, true);
  /* play the first run once when the figure is first seen; with reduced motion, show the whole run */
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (calm || !("IntersectionObserver" in window)) { k = 3; buildT(); }
  else {
    buildT();
    var io = new IntersectionObserver(function (es) {
      if (!es[0].isIntersecting) return;
      io.disconnect();
      if (touched) return;
      timer = setInterval(function () { if (touched || k >= 3) { clearInterval(timer); timer = null; return; } k += 1; drawT(); }, 1700);
    }, {threshold: 0.5});
    io.observe(lane);
  }
})();
</script>
{% endraw %}
