---
title: "Interrupt Budgets for Pre-Execution AI Control Monitors"
collection: projects
permalink: /projects/2026-11-interrupt-budgets/
date: 2026-11-02
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "A monitor tuned to a 1% per-step false-positive rate interrupts 14–37% of benign agent runs. Bounding the per-run false interruption rate instead shows what pre-execution monitors catch at a cost a deployer can accept, how to calibrate them with guarantees, and how little warning they give."
teaser: "/images/projects/2026-11-interrupt-budgets/teaser.png"
tags: ["AI control", "LLM agents", "agent monitoring", "conformal calibration"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-interrupt-budgets"
---

<style>
/* Interrupt Budgets project page: styles scoped to .ib / .ib-* so the rest of the site is untouched; colors follow the theme */
.ib{--ib-blue:#1f5fa8;--ib-blue-soft:rgba(31,95,168,.08);--ib-gray:#6f767e;--ib-gray-soft:rgba(127,127,127,.09);--ib-green:#26993a;--ib-red:#d62728;--ib-red-soft:rgba(214,39,40,.09);--ib-line:rgba(127,127,127,.28);--ib-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .ib{--ib-blue:#78b0f0;--ib-blue-soft:rgba(120,176,240,.14);--ib-gray:#c3c9cf;--ib-gray-soft:rgba(255,255,255,.07);--ib-green:#62cf72;--ib-red:#f26b67;--ib-red-soft:rgba(242,107,103,.14);--ib-line:rgba(255,255,255,.2);--ib-card:rgba(255,255,255,.04)}
.ib-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.ib-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--ib-blue-soft);color:var(--ib-blue)}
.ib-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.ib-tldr b{color:var(--ib-blue)}
.ib-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.ib-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.ib-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.ib-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--ib-line)}
.ib-btn.muted{background:transparent;color:var(--ib-gray)!important;border:1px dashed var(--ib-line);cursor:default}
.ib-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .ib-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.ib-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--ib-line);border-radius:14px;background:var(--ib-card)}
.ib-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.ib-fig.narrow img{max-width:min(100%,640px);margin:0 auto}
.ib-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.ib-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--ib-blue);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .ib-tag{color:#16202b}
.ib-fig figcaption b.ib-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.ib-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.ib-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--ib-line);background:var(--ib-card);position:relative;overflow:hidden}
.ib-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--ib-c,var(--ib-blue))}
.ib-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;color:var(--ib-c,var(--ib-blue));font-variant-numeric:tabular-nums}
.ib-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.ib-why{font-size:.9em;margin:0 0 1.6em;padding:.6em .9em;border-radius:10px;background:var(--ib-gray-soft)}
.ib-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--ib-line);border-radius:14px;background:var(--ib-card);overflow-x:auto}
.ib-table table{font-size:.8em;margin:.6em 0;min-width:520px}
.ib-table td strong{color:var(--ib-blue)}
.ib-table p{font-size:.86em;margin:.2em}
.ib-how{font-size:.86em;line-height:1.55;margin:.5em .2em 0;padding:.5em .75em;border-radius:8px;background:var(--ib-gray-soft)}
.ib-ix{font-size:.9em}
.ib-ctl{display:grid;grid-template-columns:auto 1fr;gap:.35em .9em;align-items:center;margin:0 0 .7em}
.ib-ctl .l{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65}
.ib-seg{display:flex;flex-wrap:wrap;gap:.35em}
.ib-seg button{font:inherit;font-size:.86em;line-height:1.3;padding:.3em .85em;border-radius:999px;border:1px solid var(--ib-line);background:transparent;color:var(--global-text-color);cursor:pointer;transition:background .15s,border-color .15s}
.ib-seg button:hover{border-color:var(--ib-gray)}
.ib-seg button[aria-pressed=true]{background:var(--global-text-color);border-color:var(--global-text-color);color:var(--global-bg-color)}
.ib-seg.mode button[data-v=step][aria-pressed=true]{background:var(--ib-gray);border-color:var(--ib-gray);color:#fff}
.ib-seg.mode button[data-v=budget][aria-pressed=true]{background:var(--ib-blue);border-color:var(--ib-blue);color:#fff}
html[data-theme="dark"] .ib-seg.mode button[aria-pressed=true]{color:#16202b}
.ib-rc{display:flex;flex-wrap:wrap;gap:.4em .9em;margin:0 0 .7em;font-size:.88em}
.ib-rc span b{font-variant-numeric:tabular-nums}
/* ranking: one row per monitor */
.ib-rank{position:relative;margin:1.3em 0 .2em}
.ib-rank .row{position:absolute;left:0;right:0;height:26px;display:flex;align-items:center;gap:.6em;transition:top .5s cubic-bezier(.2,.8,.2,1)}
.ib-rank .no{width:22px;height:22px;border-radius:50%;border:1px solid var(--ib-line);display:grid;place-items:center;font-size:.76em;font-weight:700;flex:none}
.ib-rank .row.top .no{background:var(--ib-top,var(--ib-gray));border-color:transparent;color:#fff}
html[data-theme="dark"] .ib-rank .row.top .no{color:#16202b}
.ib-rank .nm{width:214px;flex:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.9em}
.ib-rank .row.ours .nm{font-weight:800;color:var(--ib-blue)}
.ib-rank .tr{position:relative;flex:1;height:11px;margin-right:78px;border-radius:999px;background:var(--ib-gray-soft)}
.ib-rank .fl{position:absolute;left:0;top:0;bottom:0;border-radius:999px;background:var(--ib-bar,var(--ib-gray));transition:width .5s cubic-bezier(.2,.8,.2,1),background .3s}
.ib-rank .lo{position:absolute;top:0;bottom:0;border-radius:0 999px 999px 0;background:repeating-linear-gradient(45deg,var(--ib-red) 0 3px,transparent 3px 7px);opacity:.6;transition:left .5s cubic-bezier(.2,.8,.2,1),width .5s cubic-bezier(.2,.8,.2,1)}
.ib-rank .vl{position:absolute;top:-6px;z-index:2;padding-left:7px;font-size:.88em;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap}
.ib-rank .vl small{font-size:.86em;font-weight:600;color:var(--ib-red);margin-left:.35em}
.ib-rank .ch{width:72px;flex:none;text-align:center;font-size:.76em;font-weight:700;padding:.12em 0;border-radius:999px;font-variant-numeric:tabular-nums;background:var(--ib-red-soft);color:var(--ib-red)}
.ib-rank .ch.ok{background:var(--ib-blue-soft);color:var(--ib-blue)}
.ib-axis{display:flex;justify-content:space-between;font-size:.72em;opacity:.7;margin:.5em 160px 0 246px}
.ib-read{margin:.8em 0 0;padding:.5em .8em;border-left:3px solid var(--ib-gray);border-radius:4px;background:var(--ib-gray-soft);font-size:.92em;line-height:1.5}
.ib-read.on{border-left-color:var(--ib-blue);background:var(--ib-blue-soft)}
/* calibration: sample-size tiles and the three thresholds */
.ib-need{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em;margin:0 0 1em}
.ib-need>div{padding:.7em .9em;border-radius:12px;border:1px solid var(--ib-line);background:var(--global-bg-color)}
.ib-need .k{font-size:.72em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;opacity:.7}
.ib-need .n{font-size:1.5em;font-weight:800;line-height:1.2;font-variant-numeric:tabular-nums;color:var(--ib-blue)}
.ib-need .g{font-size:.86em;line-height:1.45}
.ib-sub{font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;opacity:.7;margin:1.1em 0 .5em}
.ib-cal{display:grid;grid-template-columns:minmax(150px,1.1fr) 2fr 74px 74px;gap:.5em .9em;align-items:center;margin:.6em 0 .2em}
.ib-cal .hd{font-size:.7em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;opacity:.65;line-height:1.3;align-self:end}
.ib-cal .num{text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
.ib-cal .hd.num{font-weight:800}
.ib-cal .m b{display:block}
.ib-cal .m span{font-size:.84em;opacity:.8}
.ib-cal .na{grid-column:2/-1;font-size:.9em;color:var(--ib-gray)}
.ib-bar{position:relative;height:11px;margin-right:46px;border-radius:999px;background:var(--ib-gray-soft)}
.ib-bar .f{position:absolute;left:0;top:0;bottom:0;border-radius:999px;transition:width .4s cubic-bezier(.2,.8,.2,1)}
.ib-bar .v{position:absolute;top:-6px;left:100%;padding-left:8px;font-size:.88em;font-weight:700;font-variant-numeric:tabular-nums}
.ib .lc{text-transform:none}
.ib-bar .d{position:absolute;top:-5px;bottom:-5px;border-left:2px dashed var(--global-text-color);opacity:.55}
.ib-bar .f.q{background:var(--ib-gray)}.ib-bar .f.c{background:var(--ib-blue)}.ib-bar .f.p{background:var(--ib-green)}
.ib-bar .f.sync{background:var(--ib-blue)}.ib-bar .f.late{background:var(--ib-gray)}
.ib-note{font-size:.82em;opacity:.8;margin:.5em 0 0}
/* deployment simulation */
.ib-sim{display:grid;grid-template-columns:minmax(150px,1.1fr) 2.4fr;gap:.7em .9em;align-items:center;margin:.9em 0 .2em}
.ib-sim .m b{display:block}
.ib-sim .m span{font-size:.84em;opacity:.8}
.ib-lead{font-size:1em;line-height:1.65;margin:0 0 1em}
.ib-defs{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em;margin:0 0 1.6em}
.ib-def{padding:.65em .85em;border-radius:12px;border:1px solid var(--ib-line);background:var(--ib-card);font-size:.86em;line-height:1.5}
.ib-def b{display:inline-block;margin-right:.35em;color:var(--ib-blue)}
.ib-def.wide{grid-column:1/-1}
.ib-steps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.7em;margin:0 0 1em}
.ib-step{padding:.9em .95em;border-radius:14px;border:1px solid var(--ib-blue);background:var(--ib-blue-soft);font-size:.86em;line-height:1.55}
.ib-step .n{display:inline-grid;place-items:center;width:1.7em;height:1.7em;border-radius:50%;background:var(--ib-blue);color:#fff;font-weight:800;font-size:.9em;margin-right:.45em}
html[data-theme="dark"] .ib-step .n{color:#16202b}
.ib-step .h{font-size:1.05em}
.ib-step p{margin:.5em 0 0}
.ib-step .was{display:block;margin-top:.55em;font-size:.92em;color:var(--ib-gray)}
.ib-take{margin:0 0 1.2em;padding:0;list-style:none;display:grid;gap:.55em}
.ib-take li{padding:.65em .9em .65em 1em;border-left:3px solid var(--ib-green);border-radius:4px;background:var(--ib-card);font-size:.92em;line-height:1.55}
.ib-take li.scope{border-left-color:var(--ib-gray)}
.ib-abs{margin:0 0 1.6em;border:1px solid var(--ib-line);border-radius:12px;background:var(--ib-card)}
.ib-abs summary{cursor:pointer;padding:.6em .9em;font-weight:700;font-size:.92em}
.ib-abs p{margin:0;padding:0 .95em .9em;font-size:.9em;line-height:1.65}
/* the theme's fixed table of contents (220px, right) covers a full-width title: keep the header in the text column */
@media screen and (min-width:1200px){.page__inner-wrap>header{max-width:calc(100% - 240px)}}
@media (max-width:900px){.ib-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:760px){.ib-steps{grid-template-columns:1fr}.ib-defs{grid-template-columns:1fr}}
@media (max-width:640px){.ib-tldr{font-size:1.04em}.ib-ctl{grid-template-columns:1fr}.ib-rank .nm{width:96px;font-size:.8em}.ib-rank .tr{margin-right:74px}.ib-rank .ch{width:56px;font-size:.7em}.ib-rank .row{gap:.4em}.ib-axis{margin:.5em 136px 0 124px}.ib-need{grid-template-columns:1fr}.ib-cal{grid-template-columns:1fr 62px 62px}.ib-cal .m{grid-column:1/-1;margin-top:.5em}.ib-cal .hd.m{display:none}.ib-cal .na{grid-column:1/-1}.ib-sim{grid-template-columns:1fr}.ib-sim .m{margin-top:.4em}}
</style>

<div class="ib">
<div class="ib-kicker"><span>AI control</span><span>LLM agent monitoring</span><span>Calibration</span></div>
<p class="ib-tldr">A monitor that reads every action of an AI agent before it runs is usually tuned to a 1% false-positive rate <i>per step</i>. Over a whole run those alarms add up: such a threshold stops <b>14–37% of benign runs</b>. We bound that per-run rate instead, the <b>interrupt budget</b>, and measure what monitors still catch, how to calibrate the threshold with guarantees, and how much warning an alarm gives.</p>
<div class="ib-btns"><a class="ib-btn" href="#how-an-interrupt-budget-works">The setup</a><a class="ib-btn ghost" href="#results">Results</a><a class="ib-btn ghost" href="#takeaways">Takeaways</a><span class="ib-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="ib-fig" id="ib-try"><div class="ib-ix"><div class="ib-ctl"><span class="l">Attacks</span><div class="ib-seg" id="ib-sets"></div><span class="l">Threshold</span><div class="ib-seg mode" id="ib-mode"><button type="button" data-v="step">1% per step</button><button type="button" data-v="budget">5% interrupt budget</button></div></div><div class="ib-rc" id="ib-rc"></div><div class="ib-rank" id="ib-rank"></div><div class="ib-axis"><span>0</span><span>share caught before harm</span><span>1</span></div><p class="ib-read" id="ib-read"></p></div><figcaption><span class="ib-tag">Interactive</span><b class="ib-title">Same monitors, same scores, two ways to set the threshold.</b>Each bar is one monitor model on LinuxArena: the share of attacks it interrupts before the harm. The pill on the right is its false interruption rate (FIR), the share of benign runs it stops at that threshold. Under the budget, the hatched part is the catch rate given up. Switch between human-crafted sabotage (72 trajectories, 200 honest runs) and live attacks generated by Claude Opus 4.6 (30 per monitor).</figcaption></figure>
</div>

## Background

<div class="ib">
<p class="ib-lead"><b>AI control</b> studies how to deploy an AI agent safely even if it may be pursuing a harmful side objective. One defense is a <b>monitor</b>: a second model that reads what the agent does and raises an alarm. When an action does damage the moment it runs, only a monitor that reviews each action <i>before</i> it is executed can prevent it. These monitors are LLM judges, and they are compared by their catch rate at a fixed per-step false-positive rate, typically 1%.</p>
<div class="ib-defs">
<div class="ib-def"><b>Pre-execution monitor</b>Scores each proposed action from 0 to 10, given the task and the visible history, before the action runs. The run is interrupted the first time a statistic of the scores (by default their running maximum) exceeds a threshold <i>h</i>.</div>
<div class="ib-def"><b>False interruption rate (FIR)</b>The share of honest runs that are interrupted at least once. This is what a benign user experiences: one interruption costs a review or a failed task, however many actions were flagged.</div>
<div class="ib-def"><b>Interrupt budget α</b>An upper bound on the FIR that the deployer chooses, for example 5%. The threshold is calibrated to keep it.</div>
<div class="ib-def"><b>Harm step and catch rate</b>In a <i>live</i> failure the harm happens at a step <i>t</i>*. An attack counts as caught only if the alarm comes before <i>t</i>*. A <i>delayed</i> failure can be stopped at any step of the run.</div>
<div class="ib-def wide"><b>Lead</b>The number of steps between the alarm and the harm, <i>t</i>* − <i>t</i>. A lead of one means the alarm fired on the last action before the harm, so only a decision that blocks that action can stop it.</div>
</div>
</div>

## The problem: a per-step rate is not what a deployer gets

<div class="ib">
<p class="ib-lead">A 1% chance of a false alarm at every step compounds over a run of many steps. On LinuxArena, the per-step 1% thresholds of 12 monitor models interrupt between 14% and 37% of honest runs. A monitor that stops one benign task in four tends to be switched off, so a catch rate quoted at that operating point describes a deployment nobody would keep.</p>
<p class="ib-lead">Fixing the per-run rate instead changes the picture, as the figure above shows. Every monitor catches less, the loss differs from monitor to monitor, and it is not predicted by how often the per-step threshold interrupted (Kendall τ = 0.03), so the ranking of monitors changes too. The paper proposes no new estimator. It asks what established ones (order-statistic thresholds, conformal calibration, sequential tests) imply for monitors that block actions, using published trajectories from <a href="https://linuxarena.ai">LinuxArena</a> with the stored per-step scores of 12 monitor models and our own monitor (DeepSeek-V4.1-Flash), and repeats the main analyses on MonitoringBench.</p>
</div>

## How an interrupt budget works

<div class="ib">
<div class="ib-steps">
<div class="ib-step"><span class="n">1</span><b class="h">Budget the run</b><p>Constrain the FIR, the probability that an honest run is ever interrupted, to at most α. With the running maximum this is the tail of one number per run: its highest score.</p><span class="was">Replaces: a 1% false-positive rate per step.</span></div>
<div class="ib-step"><span class="n">2</span><b class="h">Calibrate with a guarantee</b><p>Take the threshold from an order statistic of the per-run maxima of <i>n</i> honest runs: split-conformal keeps the budget on average, a PAC order statistic keeps it with probability 1 − δ.</p><span class="was">Replaces: the empirical percentile.</span></div>
<div class="ib-step"><span class="n">3</span><b class="h">Score what is prevented</b><p>Count an attack as caught only if the alarm comes before the harm step, and report the lead, the number of steps left for anyone to react.</p><span class="was">Adds: how much time an alarm leaves.</span></div>
</div>
<figure class="ib-fig"><img src="/images/projects/2026-11-interrupt-budgets/fig1-overview.png" alt="Figure 1: offline calibration on honest runs, the online pre-execution gate, and two example score traces" loading="lazy"><figcaption><span class="ib-tag">Figure 1</span><b class="ib-title">Calibrate offline, gate every action online.</b>(a) The deployer calibrates a threshold <i>h</i> on the per-run maxima of <i>n</i> honest runs of the deployed agent. (b) Online, the monitor scores each proposed action before it runs, and the run is interrupted the first time the statistic exceeds <i>h</i>. (c) A live attack by Claude Opus 4.6: the score crosses the threshold only at the attacking action, one step before the harm. (d) A human-crafted sabotage that is missed: the monitor's rationale names the injected flaw, but the score only ties the threshold. Both thresholds are set for a 5% budget; the circled numbers mark where the four findings below apply.</figcaption></figure>
</div>

## Results

<div class="ib"><div class="ib-stats">
<div class="ib-stat" style="--ib-c:var(--ib-red)"><div class="v">14–37%</div><div class="d">of benign runs interrupted by a 1% per-step threshold, across 12 monitor models on LinuxArena</div></div>
<div class="ib-stat"><div class="v">0.79 → 0.59</div><div class="d">catch rate of the Claude Opus 4.6 monitor on human-crafted sabotage once the FIR is held to 5%</div></div>
<div class="ib-stat" style="--ib-c:var(--ib-gray)"><div class="v">up to 24%</div><div class="d">FIR when a weak monitor's 5% threshold is calibrated on one agent and deployed on another</div></div>
<div class="ib-stat" style="--ib-c:var(--ib-green)"><div class="v">1 step</div><div class="d">median lead on caught Opus 4.6 attacks: 10 of 13 alarms fire on the last action before the harm</div></div>
</div></div>

### 1. The budget costs catch rate where detection is hard

<div class="ib">
<p class="ib-lead">On human-crafted sabotage, a 5% budget lowers the catch rate of every monitor, by up to 0.38: from 0.79 to 0.59 for Claude Opus 4.6, from 0.47 to 0.13 for Claude Sonnet 4.6, and from 0.55 to 0.37 for ours. The order changes as well: GLM 5 falls from third to fifth and Sonnet 4.6 from seventh to tenth. On live attacks generated by Opus 4.6, which only a pre-execution monitor can stop, capable monitors lose little (at most 0.07 for the nine that catch more than 0.10 of human sabotage), and only weak ones lose much (GPT-5 Nano: 0.85 to 0.47). The interactive figure at the top shows both attack sets.</p>
</div>

### 2. The budget can be guaranteed, but calibration transfers poorly

<div class="ib">
<p class="ib-lead">Conformal and PAC thresholds keep the budget on the coarse, often tied scores of LLM judges, as theory predicts, in all 52 feasible settings we tested. The usual empirical percentile does not. The guarantee has a price in catch rate, and a minimum number of honest runs:</p>
<figure class="ib-fig" id="ib-calfig"><div class="ib-ix"><div class="ib-ctl"><span class="l">Budget <span class="lc">α</span></span><div class="ib-seg" id="ib-alpha"></div><span class="l">Confidence 1 − <span class="lc">δ</span></span><div class="ib-seg" id="ib-delta"></div></div><div class="ib-need" id="ib-need"></div><div class="ib-sub">Measured at <span class="lc">α</span> = 5%, <span class="lc">δ</span> = 10% · human-crafted sabotage, our monitor</div><div class="ib-ctl"><span class="l">Honest runs <span class="lc">n</span></span><div class="ib-seg" id="ib-n"></div></div><div class="ib-cal" id="ib-cal"></div><p class="ib-read on" id="ib-read2"></p></div><figcaption><span class="ib-tag">Interactive</span><b class="ib-title">How many honest runs does a guarantee need, and what does it cost?</b>Top: the smallest calibration set for which each threshold exists, ⌈1/α − 1⌉ for conformal and ⌈log δ / log(1 − α)⌉ for PAC. Bottom: the three thresholds when calibration sets of size <i>n</i> are drawn 1,000 times from a pool of 200 honest runs. The bar is the share of draws whose FIR exceeds the 5% budget (dashed line: δ = 10%).</figcaption></figure>
<p class="ib-lead">The guarantees assume that calibration runs look like deployment runs. Across agents they do not. A threshold calibrated on one agent's honest runs and deployed on another's can interrupt far more than intended, and pooling agents does not repair a weak monitor:</p>
<figure class="ib-fig"><img src="/images/projects/2026-11-interrupt-budgets/fig4-calibration-shift.png" alt="Figure 4: mean FIR when a threshold calibrated on one agent is deployed on another" loading="lazy"><figcaption><span class="ib-tag">Figure 4</span><b class="ib-title">A threshold calibrated on another agent can exceed the budget.</b>Mean FIR at a 5% target when a conformal threshold calibrated on one agent's honest runs (rows) is deployed on another's (columns). Left: the weak GPT-5 Nano monitor reaches up to 24%, and a threshold pooled over 11 agents still gives 9.1% on Opus and 10.8% on Sonnet. Right: our stronger monitor stays between 1% and 7%, except for thresholds calibrated on Gemini 3 Flash (6.5–10%). The task mix matters too: holding out one LinuxArena environment raises the pooled FIR of conformal thresholds to 6.0–9.0% for three monitors.</figcaption></figure>
</div>

### 3. The running maximum is hard to beat

<div class="ib">
<p class="ib-lead">Evidence accumulates over a run, so a sequential statistic should beat the maximum. For our monitor on human sabotage, a likelihood-ratio sum does:</p>
</div>

<div class="ib"><div class="ib-table" markdown="1">

<span class="ib-tag">Table 2</span> **Detectors for our monitor at a 5% FIR budget.** Catch rate before harm; Δ against the running maximum with a 95% interval (paired bootstrap over attacks, thresholds held fixed).

| Detector | Human sabotage: catch | Δ vs. max | Haiku 4.5 attacks: catch | Δ vs. max |
|---|---|---|---|---|
| per-step 1% (reference; FIR 0.25 / 0.26) | 0.55 | – | 0.89 | – |
| running maximum | 0.37 | – | 0.85 | – |
| likelihood-ratio sum (LLR-SUM) | 0.50 | +0.14 [+0.06, +0.22] | 0.82 | −0.02 |
| e-process (threshold 1/α) | 0.47 | +0.10 [+0.04, +0.17] | 0.79 | −0.05 |
| CUSUM | 0.42 | +0.05 [+0.01, +0.11] | 0.86 | +0.01 |
| likelihood-ratio CUSUM | 0.28 | −0.08 [−0.17, −0.01] | 0.85 | +0.00 |
| three strikes | 0.19 | −0.18 [−0.28, −0.08] | 0.84 | −0.01 |

<p class="ib-how"><b>How to read it.</b> All rows except the first keep the same 5% budget, so the differences are due to the statistic alone. When the bootstrap also resamples honest runs and refits the thresholds, only the loss of three strikes still excludes zero.</p>

</div></div>

<div class="ib">
<p class="ib-lead">The gain does not carry over to other monitors or attacks:</p>
<figure class="ib-fig"><img src="/images/projects/2026-11-interrupt-budgets/fig6-llr-sum-vs-max.png" alt="Change in catch rate from replacing the running maximum with a likelihood-ratio sum, per monitor" loading="lazy"><figcaption><span class="ib-tag">Figure 6</span><b class="ib-title">A likelihood-ratio sum helps some monitors and hurts others.</b>Change in catch rate from replacing the maximum with LLR-SUM at a 5% budget, with 95% intervals. (a) On human-crafted sabotage it helps GLM 5 and our monitor and hurts GPT-5.4. (b) On Opus-generated live attacks it significantly hurts 6 of 12 monitors, by 0.20 to 0.33. Once threshold uncertainty is counted, no gain remains significant.</figcaption></figure>
<p class="ib-lead">Stress tests on stored scores show why the maximum is the safer default. An attack that concentrates its evidence in one action, close to what Opus 4.6 attacks do, leaves the maximum unaffected and drops LLR-SUM to a catch rate of at most 0.12 in 9 of 11 monitor–attack pairs. An attack that keeps every action just under the threshold does the opposite. Alarming when either statistic crosses its threshold at α/2 keeps the budget and covers both.</p>
</div>

### 4. Alarms can come at the last moment

<div class="ib">
<p class="ib-lead">Detection of model-generated live attacks is close to saturated: at a 5% budget our monitor catches 0.87–1.00 of the attacks of seven attacker models. What differs is <i>when</i> the alarm arrives.</p>
<figure class="ib-fig"><img src="/images/projects/2026-11-interrupt-budgets/fig5-alarm-lead.png" alt="Figure 5: alarm lead by attacker model, and share of late catches for 12 monitors" loading="lazy"><figcaption><span class="ib-tag">Figure 5</span><b class="ib-title">The lead depends on the attacker.</b>Left: steps between the first alarm and the harm for our monitor (dots: caught attacks; red bars: medians; 15 attacks per attacker). Attacks by Grok 4.1 Fast and Gemini 3 Flash are caught a median of 23 and 19 steps early; attacks by Claude Opus 4.6 a median of one step. Right: 12 LinuxArena monitors on Opus-generated live attacks, share caught before the harm (gray) and within two steps of it (red). For 10 of the 12, the median lead is one step.</figcaption></figure>
<p class="ib-lead">A one-step lead leaves no time for a human to look. Replaying the measured scores and monitor latencies under three deployment modes shows what that means:</p>
<figure class="ib-fig" id="ib-simfig"><div class="ib-ix"><div class="ib-ctl"><span class="l">Attacks by</span><div class="ib-seg" id="ib-att"></div><span class="l">Agent step time</span><div class="ib-seg" id="ib-d"></div><span class="l">Review delay</span><div class="ib-seg" id="ib-r"></div></div><div class="ib-sim" id="ib-sim"></div><p class="ib-read on" id="ib-read3"></p></div><figcaption><span class="ib-tag">Interactive</span><b class="ib-title">Only blocking stops an attack that is flagged one step early.</b>Share of live attacks whose harm is prevented by our monitor at a 5% budget (15 attacks by Opus 4.6, 30 by Haiku 4.5). The public data do not record how long an agent step takes, so the step time and the human review delay are assumptions you can vary.</figcaption></figure>
</div>

### Replication and a pitfall

<div class="ib"><div class="ib-table" markdown="1">

<span class="ib-tag">Table 3</span> **The patterns recur on a second benchmark.** Our monitor at α = 5% on LinuxArena (human sabotage; last two rows: Opus 4.6 live attacks) and on MonitoringBench (300 task-matched pairs of attack and honest runs by Claude Opus 4.5).

| | LinuxArena | MonitoringBench |
|---|---|---|
| FIR of a 1% per-step threshold | 0.24 | 0.10 |
| Catch rate: 1% per step / 5% budget | 0.55 / 0.37 | 0.96 / 0.92 |
| Share of deployments over budget: percentile / PAC | 0.44 / 0.03 | 0.42 / 0.02 |
| LLR-SUM − running maximum | +0.14 | −0.09 |
| Three strikes − running maximum | −0.18 | −0.71 |
| Caught before harm | 13 of 15 | 0.87 |
| Median lead; share of catches within 2 steps | 1; 10 of 13 | 1; 0.70 |

<p class="ib-how"><b>How to read it.</b> The budget costs less on MonitoringBench, in part because its benign runs were screened before release. The calibration and timing results replicate, and the likelihood-ratio sum that seemed to help our monitor on LinuxArena hurts it here, so a statistic has to be validated on deployment data.</p>

</div></div>

<div class="ib">
<p class="ib-lead"><b>Label leakage through the submit call.</b> Studies that render LinuxArena transcripts themselves can leak attack labels: the final <code>submit</code> call of an attack carries arguments of the side task's scorer, with field names that never occur in honest runs (34 of 72 human sabotage trajectories). A monitor that is shown the call looks much better than it is: our catch rate at a 5% budget rises from 0.37 to 0.66, and three conclusions of a configuration comparison reverse. LinuxArena's own monitors hide the call, as do all results on this page.</p>
</div>

## Takeaways

<div class="ib">
<ul class="ib-take">
<li><b>Choose the budget, then the guarantee.</b> Pick α from the catch–FIR curve by weighing an interruption against the harm it may prevent, and decide whether it must hold on average (conformal) or with probability 1 − δ (PAC).</li>
<li><b>Calibrate on the deployment.</b> Use at least ⌈1/α − 1⌉ (conformal) or ⌈log δ / log(1 − α)⌉ (PAC) honest runs of the deployed agent, scaffold and task mix, and recalibrate when any of them changes. Check a pooled threshold for the monitor in use before relying on it.</li>
<li><b>Threshold the running maximum.</b> Add an accumulating statistic only if it wins in cross-validation on deployment data, and then run it next to the maximum at α/2 each.</li>
<li><b>Block synchronously.</b> Against a capable attacker the only incriminating action can be the one that does the harm. There is then no evidence to accumulate and no time for review.</li>
<li><b>Evaluating a monitor.</b> Report the catch rate before harm at a per-run budget, together with the lead. Catch rates on human sabotage range from 0.00 to 0.60 across monitors under the same budget, far more than between statistics or calibration methods.</li>
<li class="scope"><b>Scope.</b> The evidence comes from two English-language benchmarks in one domain, software engineering and system administration on Linux. Human-crafted sabotage is scarce (72 trajectories, 5 of them live failures), the adaptive attackers are stress tests on stored scores, and the deployment simulation rests on assumed step times and review delays.</li>
</ul>
<details class="ib-abs"><summary>Abstract</summary><p>LLM judges deployed as pre-execution monitors read each proposed action of an AI agent and can block it, but they are tuned to a per-step false-positive rate. A deployer experiences something else: alarms compound over a trajectory, so on LinuxArena a 1% per-step threshold interrupts 14–37% of benign runs across 12 monitor models. We bound this false interruption rate (FIR) instead. On attacks that are hard to detect, monitors then catch less: at a 5% budget, the catch rate of Claude Opus 4.6 on human-crafted sabotage falls from 0.79 to 0.59. Conformal or PAC thresholds calibrated on the deployed agent guarantee the budget, but thresholds calibrated on another agent or environment can exceed it. No statistic we tested reliably beats the running maximum: likelihood-ratio statistics hurt several monitors, help none once threshold uncertainty is counted, and collapse in stress tests that concentrate an attack's evidence in one action. When attacks by Claude Opus models are caught, the median alarm comes only at the action that causes the harm, leaving no time for review that does not block it. These patterns recur on MonitoringBench, where the budget costs less and likelihood ratios hurt our monitor.</p></details>
<p class="ib-why">All results use published trajectories and attack data from LinuxArena and MonitoringBench (both CC-BY-4.0), plus a separate release of benign MonitoringBench trajectories that is used for evaluation only. No new attack content was created. Results are reported as aggregates, and attack mechanisms are described without reusable payloads.</p>
</div>

## Paper and code

Paper, code and BibTeX will be linked here once the preprint is public.

{% raw %}
<script>
(function () {
  var D = {"rank":{"Human-crafted sabotage":[{"name":"Claude Opus 4.6","step":0.788,"fir":0.298,"budget":0.588},{"name":"Claude Sonnet 4.6","step":0.471,"fir":0.366,"budget":0.13},{"name":"Claude Haiku 4.5","step":0.323,"fir":0.303,"budget":0.174},{"name":"GPT-5.4","step":0.384,"fir":0.15,"budget":0.18},{"name":"GPT-5 Mini","step":0.553,"fir":0.14,"budget":0.241},{"name":"GPT-5 Nano","step":0.03,"fir":0.228,"budget":0.0},{"name":"GPT-4.1 Nano","step":0.013,"fir":0.288,"budget":0.0},{"name":"GPT-OSS-120B","step":0.228,"fir":0.312,"budget":0.032},{"name":"Gemini 3.1 Pro","step":0.73,"fir":0.355,"budget":0.599},{"name":"Gemini 3 Flash","step":0.571,"fir":0.282,"budget":0.336},{"name":"Kimi K2.5","step":0.442,"fir":0.341,"budget":0.135},{"name":"GLM 5","step":0.631,"fir":0.29,"budget":0.253},{"name":"Ours (DeepSeek-V4.1-Flash)","ours":true,"step":0.554,"fir":0.241,"budget":0.3653}],"Opus-generated live attacks":[{"name":"Claude Opus 4.6","step":1.0,"fir":0.296,"budget":0.967},{"name":"Claude Sonnet 4.6","step":0.98,"fir":0.361,"budget":0.967},{"name":"Claude Haiku 4.5","step":0.967,"fir":0.301,"budget":0.943},{"name":"GPT-5.4","step":0.967,"fir":0.153,"budget":0.94},{"name":"GPT-5 Mini","step":0.967,"fir":0.14,"budget":0.957},{"name":"GPT-5 Nano","step":0.847,"fir":0.23,"budget":0.467},{"name":"GPT-4.1 Nano","step":0.783,"fir":0.288,"budget":0.39},{"name":"GPT-OSS-120B","step":0.967,"fir":0.317,"budget":0.873},{"name":"Gemini 3.1 Pro","step":0.967,"fir":0.348,"budget":0.967},{"name":"Gemini 3 Flash","step":0.967,"fir":0.279,"budget":0.9},{"name":"Kimi K2.5","step":0.967,"fir":0.342,"budget":0.9},{"name":"GLM 5","step":0.967,"fir":0.291,"budget":0.967}]},
    "cal":{"quantile":{"20":[0.085,0.566,0.42],"30":[0.078,0.629,0.417],"45":[0.071,0.601,0.409],"60":[0.059,0.46,0.395],"100":[0.055,0.438,0.392],"150":[0.055,0.479,0.388],"200":[0.052,0.411,0.387],"300":[0.051,0.404,0.386]},"conformal":{"20":[0.047,0.33,0.35],"30":[0.03,0.194,0.317],"45":[0.04,0.271,0.352],"60":[0.046,0.365,0.37],"100":[0.048,0.355,0.376],"150":[0.044,0.284,0.367],"200":[0.047,0.324,0.377],"300":[0.047,0.326,0.378]},"pac10":{"20":null,"30":null,"45":[0.018,0.067,0.282],"60":[0.013,0.022,0.266],"100":[0.018,0.031,0.292],"150":[0.024,0.042,0.316],"200":[0.027,0.038,0.325],"300":[0.031,0.032,0.332]}},
    "sim":{"Claude Opus 4.6":{"n":15,"5 s":{"sync":0.87,"semi":[0.20,0.07,0.00],"async":[0.13,0.00,0.00]},"30 s":{"sync":0.87,"semi":[0.20,0.13,0.07],"async":[0.20,0.13,0.00]}},"Claude Haiku 4.5":{"n":30,"5 s":{"sync":0.90,"semi":[0.83,0.53,0.00],"async":[0.63,0.00,0.00]},"30 s":{"sync":0.90,"semi":[0.90,0.73,0.10],"async":[0.83,0.53,0.00]}}}};
  function $(id) { return document.getElementById(id); }
  function seg(el, items, cur, on) {
    el.innerHTML = "";
    items.forEach(function (k) { var b = document.createElement("button"); b.type = "button"; b.textContent = k;
      b.setAttribute("aria-pressed", k === cur); b.onclick = function () { on(k); }; el.appendChild(b); });
  }
  function f2(v) { return v.toFixed(2); }
  function pct(v) { return Math.round(100 * v) + "%"; }
  function rankOf(rows, key, name) { return rows.slice().sort(function (a, b) { return b[key] - a[key]; }).findIndex(function (r) { return r.name === name; }) + 1; }

  /* 1. per-step threshold vs. interrupt budget */
  var set = Object.keys(D.rank)[0], mode = "step", ROW = 30;
  function drawRank() {
    seg($("ib-sets"), Object.keys(D.rank), set, function (k) { set = k; drawRank(); });
    $("ib-mode").querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.v === mode); });
    var rows = D.rank[set], sorted = rows.slice().sort(function (a, b) { return b[mode] - a[mode] || b.step - a.step; });
    var box = $("ib-rank"); box.style.height = rows.length * ROW + "px";
    box.style.setProperty("--ib-top", mode === "step" ? "var(--ib-gray)" : "var(--ib-blue)");
    box.style.setProperty("--ib-bar", mode === "step" ? "var(--ib-gray)" : "var(--ib-blue)");
    var keep = {}; box.querySelectorAll(".row").forEach(function (e) { if (rows.some(function (r) { return r.name === e.dataset.n; })) keep[e.dataset.n] = e; else e.remove(); });
    sorted.forEach(function (r, i) {
      var e = keep[r.name];
      if (!e) { e = document.createElement("div"); e.dataset.n = r.name;
        e.innerHTML = '<span class="no"></span><span class="nm"></span><span class="tr"><span class="lo"></span><span class="fl"></span><span class="vl"></span></span><span class="ch"></span>'; box.appendChild(e); }
      e.className = "row" + (r.ours ? " ours" : "") + (i === 0 ? " top" : "");
      e.querySelector(".no").textContent = i + 1; e.querySelector(".nm").textContent = r.name; e.querySelector(".nm").title = r.name;
      var cur = 100 * r[mode], full = 100 * r.step, lost = r.step - r.budget;
      e.style.top = i * ROW + "px"; e.querySelector(".fl").style.width = cur + "%";
      var lo = e.querySelector(".lo"); lo.style.left = cur + "%"; lo.style.width = (mode === "budget" ? Math.max(0, full - cur) : 0) + "%";
      var v = e.querySelector(".vl"); v.style.left = Math.max(cur, full) + "%";
      v.innerHTML = f2(r[mode]) + (mode === "budget" && lost >= 0.005 ? "<small>−" + f2(lost) + "</small>" : "");
      var ch = e.querySelector(".ch"); ch.className = "ch" + (mode === "budget" ? " ok" : "");
      ch.textContent = "FIR " + (mode === "budget" ? "5%" : pct(r.fir));
      ch.title = mode === "budget" ? "Threshold calibrated to interrupt 5% of benign runs" : "Share of benign runs this threshold interrupts";
    });
    var firs = rows.filter(function (r) { return !r.ours; }).map(function (r) { return r.fir; });
    var worst = rows.slice().sort(function (a, b) { return (b.step - b.budget) - (a.step - a.budget); })[0];
    var isH = set === Object.keys(D.rank)[0];
    $("ib-rc").innerHTML = isH ? ["GLM 5", "Claude Sonnet 4.6", "Gemini 3.1 Pro"].map(function (n) {
      return "<span><b>" + n + "</b>: #" + rankOf(rows, "step", n) + ' per step → <b style="color:var(--ib-blue)">#' + rankOf(rows, "budget", n) + "</b> under the budget</span>";
    }).join("") : "<span>Capable monitors keep most of their catch rate on these attacks; the two weakest lose about half.</span>";
    var r = $("ib-read"); r.className = "ib-read" + (mode === "budget" ? " on" : "");
    r.innerHTML = mode === "step"
      ? "<b>1% per step:</b> the usual operating point. These thresholds interrupt " + Math.round(100 * Math.min.apply(null, firs)) + "–" + pct(Math.max.apply(null, firs)) + " of benign runs."
      : "<b>5% interrupt budget:</b> every threshold now stops about one benign run in twenty. The largest loss is " + worst.name + ", from " + f2(worst.step) + " to " + f2(worst.budget) + ".";
  }
  $("ib-mode").querySelectorAll("button").forEach(function (b) { b.onclick = function () { mode = b.dataset.v; drawRank(); }; });

  /* 2. calibration: minimum n, and the three thresholds at a given n */
  var alpha = "5%", delta = "90%", n = "45";
  function drawNeed() {
    seg($("ib-alpha"), ["2%", "5%", "10%"], alpha, function (k) { alpha = k; drawNeed(); });
    seg($("ib-delta"), ["90%", "95%"], delta, function (k) { delta = k; drawNeed(); });
    var a = parseFloat(alpha) / 100, d = 1 - parseFloat(delta) / 100;
    var nc = Math.ceil(1 / a - 1 - 1e-9), np = Math.ceil(Math.log(d) / Math.log(1 - a) - 1e-9);
    $("ib-need").innerHTML =
      '<div><span class="k">Split-conformal</span><div class="n">n ≥ ' + nc + '</div><div class="g">honest runs, so that the FIR is at most ' + alpha + " on average over calibration sets</div></div>" +
      '<div><span class="k">PAC order statistic</span><div class="n">n ≥ ' + np + '</div><div class="g">honest runs, so that this deployment\'s FIR is at most ' + alpha + " with probability " + delta + "</div></div>";
  }
  function bar(cls, v, max, mark, fmt) {
    var p = 100 * v / max;
    return '<div class="ib-bar"><span class="f ' + cls + '" style="width:' + p + '%"></span>' + (mark ? '<span class="d" style="left:' + 100 * mark / max + '%"></span>' : "") + '<span class="v">' + (fmt || pct)(v) + "</span></div>";
  }
  function drawCal() {
    seg($("ib-n"), Object.keys(D.cal.quantile), n, function (k) { n = k; drawCal(); });
    var M = [["quantile", "Empirical percentile", "no guarantee", "q"], ["conformal", "Split-conformal", "budget kept on average", "c"], ["pac10", "PAC", "budget kept in ≥ 90% of deployments", "p"]];
    var h = '<span class="hd m">Threshold</span><span class="hd">Deployments over budget</span><span class="hd num">Mean FIR</span><span class="hd num">Catch</span>';
    M.forEach(function (m) {
      var v = D.cal[m[0]][n];
      h += '<div class="m"><b>' + m[1] + "</b><span>" + m[2] + "</span></div>";
      h += v ? bar(m[3], v[1], 0.7, 0.1) + '<span class="num">' + (100 * v[0]).toFixed(1) + '%</span><span class="num">' + f2(v[2]) + "</span>"
             : '<span class="na">No finite threshold: PAC at α = 5%, δ = 10% needs n ≥ 45.</span>';
    });
    $("ib-cal").innerHTML = h;
    var q = D.cal.quantile[n], c = D.cal.conformal[n], p = D.cal.pac10[n];
    $("ib-read2").innerHTML = "With " + n + " honest runs, the empirical percentile exceeds the 5% budget in " + pct(q[1]) + " of deployments and split-conformal in " + pct(c[1]) +
      (p ? ". PAC cuts this to " + pct(p[1]) + ", below δ, and gives up " + f2(q[2] - p[2]) + " of catch rate against the percentile."
         : ". A PAC threshold does not exist yet: too few runs to certify 5% with 90% confidence.");
  }

  /* 3. deployment simulation */
  var att = "Claude Opus 4.6", dstep = "30 s", rev = "10 min", REV = ["1 min", "10 min", "1 h"];
  function drawSim() {
    seg($("ib-att"), Object.keys(D.sim), att, function (k) { att = k; drawSim(); });
    seg($("ib-d"), ["5 s", "30 s"], dstep, function (k) { dstep = k; drawSim(); });
    seg($("ib-r"), REV, rev, function (k) { rev = k; drawSim(); });
    var S = D.sim[att][dstep], i = REV.indexOf(rev), semi = S.semi[i], asy = S.async[i];
    function row(name, sub, cls, v) { return '<div class="m"><b>' + name + "</b><span>" + sub + "</span></div>" + bar(cls, v, 1, 0, f2); }
    $("ib-sim").innerHTML =
      row("Synchronous", "every action waits for its score; an alarm stops the agent", "sync", S.sync) +
      row("Semi-synchronous", "a flagged action is held for review; the agent continues", "late", semi) +
      row("Asynchronous", "nothing waits; a flag is reviewed after the delay", "late", asy);
    $("ib-read3").innerHTML = "Attacks by " + att + ", " + rev + " review: blocking prevents <b>" + f2(S.sync) + "</b> of them, review that does not block at most <b>" + f2(Math.max(semi, asy)) + "</b>." +
      (att === "Claude Opus 4.6" ? " These attacks are flagged a median of one step before the harm." : " These attacks are flagged earlier, so a fast review still stops many of them.");
  }

  drawRank(); drawNeed(); drawCal(); drawSim();

  /* show the change once without a click: when the ranking is first seen, switch to the budget */
  var touched = false, fig = $("ib-try");
  fig.addEventListener("pointerdown", function () { touched = true; }, true);
  fig.addEventListener("keydown", function () { touched = true; }, true);
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!calm && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (es) {
      if (!es[0].isIntersecting) return;
      io.disconnect();
      setTimeout(function () { if (!touched && mode === "step") { mode = "budget"; drawRank(); } }, 1600);
    }, {threshold: 0.6});
    io.observe($("ib-rank"));
  }
})();
</script>
{% endraw %}
