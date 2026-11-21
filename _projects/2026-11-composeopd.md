---
title: "ComposeOPD: Composing Diffusion Teachers and Published Students, Screened Before Training"
collection: projects
permalink: /projects/2026-11-composeopd/
date: 2026-11-16
toc: false
status: "Under review"
venue: "Under review"
excerpt: "ComposeOPD distills one student from a composition of every model on a diffusion backbone, reward experts and published students alike, and chooses the composition weights by sampling the composed teacher before any training."
teaser: "/images/projects/2026-11-composeopd/fig1-teaser.png"
tags: ["diffusion models", "on-policy distillation", "model composition", "reward fine-tuning"]
publication: "/publication/2026-11-composeopd"
---

<style>
.cpd { --c-blue: #2a78d6; --c-red: #e34948; --c-ink: #1f2328; --c-ink2: #57606a; --c-muted: #8b949e; --c-card: #ffffff;
  --c-line: rgba(31,35,40,0.12); --c-soft: #f6f8fa; --c-hero: linear-gradient(135deg, rgba(42,120,214,0.10), rgba(227,73,72,0.09));
  --c-shadow: 0 1px 2px rgba(31,35,40,0.06), 0 6px 18px rgba(31,35,40,0.06);
  --c-s2: #eb6834; --c-s3: #1baf7a; --c-s4: #eda100; --c-crit: #d03b3b; --c-good: #1a7f37; --c-grid: #e8e8e3; --c-axis: #c3c2b7; }
html[data-theme="dark"] .cpd { --c-blue: #4c94ea; --c-red: #ef6b6a; --c-ink: #e6edf3; --c-ink2: #b1bac4; --c-muted: #8b949e;
  --c-card: #22272e; --c-line: rgba(230,237,243,0.14); --c-soft: #2d333b;
  --c-hero: linear-gradient(135deg, rgba(76,148,234,0.16), rgba(239,107,106,0.12)); --c-shadow: 0 1px 2px rgba(0,0,0,0.3);
  --c-s2: #e0703f; --c-s3: #2bb583; --c-s4: #d49a1a; --c-crit: #f06a6a; --c-good: #3fb950; --c-grid: #353b43; --c-axis: #4a525c; }
.cpd { color: var(--c-ink); }
.cpd .hero { background: var(--c-hero); border: 1px solid var(--c-line); border-radius: 18px; padding: 26px 26px 20px; margin: 4px 0 26px; }
.cpd .tagline { font-size: 1.32em; line-height: 1.35; font-weight: 650; margin: 0 0 10px; letter-spacing: -0.01em; }
.cpd .tagline em { font-style: normal; color: var(--c-red); }
.cpd .badges { display: flex; flex-wrap: wrap; gap: 8px; margin: 0 0 16px; }
.cpd .badge { font-size: 0.72em; font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
  background: var(--c-card); border: 1px solid var(--c-line); color: var(--c-ink2); }
.cpd .pills { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 20px; }
.cpd .pill { display: inline-flex; align-items: center; gap: 7px; padding: 8px 16px; border-radius: 999px; font-size: 0.86em; font-weight: 600;
  background: #24292f; color: #fff !important; text-decoration: none !important; border: 1px solid #24292f; transition: transform .12s, box-shadow .12s; }
.cpd .pill:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.18); }
.cpd .pill.ghost { background: transparent; color: var(--c-ink2) !important; border: 1px dashed var(--c-line); cursor: default; }
.cpd .pill.ghost:hover { transform: none; box-shadow: none; }
.cpd .pill svg { width: 15px; height: 15px; fill: currentColor; }
html[data-theme="dark"] .cpd .pill:not(.ghost) { background: #e6edf3; color: #1f2328 !important; border-color: #e6edf3; }
.cpd .figcard { background: var(--c-card); border: 1px solid var(--c-line); border-radius: 14px; padding: 14px; box-shadow: var(--c-shadow); margin: 0 0 8px; }
.cpd .figcard img { display: block; width: 100%; height: auto; border-radius: 6px; background: #fff; }
.cpd .cap { font-size: 0.86em; color: var(--c-ink2); margin: 8px 2px 24px; line-height: 1.5; }
.cpd .cap b { color: var(--c-ink); }
.cpd .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 28px; }
.cpd .stat { background: var(--c-card); border: 1px solid var(--c-line); border-radius: 14px; padding: 14px 14px 12px; box-shadow: var(--c-shadow); }
.cpd .stat .num { font-size: 1.75em; font-weight: 750; line-height: 1.1; letter-spacing: -0.02em; color: var(--c-blue); }
.cpd .stat .num.red { color: var(--c-red); }
.cpd .stat .lbl { font-size: 0.78em; color: var(--c-ink2); margin-top: 6px; line-height: 1.4; }
.cpd .tldr { border-left: 4px solid var(--c-blue); background: var(--c-soft); border-radius: 0 12px 12px 0; padding: 14px 18px; margin: 0 0 30px; }
.cpd .tldr p { margin: 0; line-height: 1.6; }
.cpd .tldr .k { font-size: 0.72em; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--c-blue); display: block; margin-bottom: 4px; }
.cpd h2.sec { text-align: center; font-size: 1.45em; margin: 38px 0 6px; border: 0; }
.cpd h3.sec3 { text-align: center; font-size: 1.15em; margin: 30px 0 4px; border: 0; }
.cpd .sub { text-align: center; color: var(--c-ink2); margin: 0 auto 20px; max-width: 46em; font-size: 0.95em; }
.cpd .steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 14px 0 26px; }
.cpd .step { background: var(--c-card); border: 1px solid var(--c-line); border-radius: 14px; padding: 14px 16px; box-shadow: var(--c-shadow); }
.cpd .step .n { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%; background: var(--c-blue); color: #fff;
  font-size: 0.8em; font-weight: 700; margin-bottom: 8px; }
.cpd .step:nth-child(3) .n { background: var(--c-red); }
.cpd .step h4 { margin: 0 0 6px; font-size: 1em; }
.cpd .step p { margin: 0; font-size: 0.86em; color: var(--c-ink2); line-height: 1.5; }
.cpd .split { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 22px; align-items: center; margin: 0 0 26px; }
.cpd .split .txt p { margin: 0 0 10px; line-height: 1.6; }
.cpd .tablecard { background: var(--c-card); border: 1px solid var(--c-line); border-radius: 14px; padding: 6px 14px 4px; box-shadow: var(--c-shadow);
  overflow-x: auto; margin: 0 0 6px; }
.cpd .tablecard table { width: 100%; border-collapse: collapse; font-size: 0.86em; font-variant-numeric: tabular-nums; margin: 6px 0; display: table; }
.cpd .tablecard th, .cpd .tablecard td { border: 0; border-bottom: 1px solid var(--c-line); padding: 7px 8px; white-space: nowrap; }
.cpd .tablecard th { color: var(--c-ink2); font-weight: 600; font-size: 0.92em; white-space: normal; vertical-align: bottom; }
.cpd .tablecard tr:last-child td { border-bottom: 0; }
.cpd .tablecard.ours tr:last-child td { background: rgba(227,73,72,0.08); }
.cpd .tabletitle { font-weight: 650; margin: 4px 0 8px; font-size: 0.95em; }
.cpd .cmp { background: var(--c-card); border: 1px solid var(--c-line); border-radius: 14px; padding: 16px; box-shadow: var(--c-shadow); margin: 0 0 8px; }
.cpd .cmp .row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }
.cpd .cmp .lab { font-size: 0.74em; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--c-muted); margin-right: 4px; min-width: 64px; }
.cpd .chip { font: inherit; font-size: 0.82em; padding: 5px 12px; border-radius: 999px; border: 1px solid var(--c-line); background: var(--c-soft);
  color: var(--c-ink); cursor: pointer; }
.cpd .chip[aria-pressed="true"] { background: var(--c-ink); color: var(--c-card); border-color: var(--c-ink); }
.cpd .juxt { position: relative; width: 100%; max-width: 420px; aspect-ratio: 1 / 1; margin: 6px auto 0; border-radius: 10px; overflow: hidden;
  user-select: none; touch-action: none; cursor: ew-resize; background: #000; }
.cpd .juxt img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; border-radius: 0; }
.cpd .juxt .top { clip-path: inset(0 0 0 50%); }
.cpd .juxt .bar { position: absolute; top: 0; bottom: 0; left: 50%; width: 2px; background: #fff; box-shadow: 0 0 6px rgba(0,0,0,0.5); transform: translateX(-1px); }
.cpd .juxt .knob { position: absolute; top: 50%; left: 50%; width: 34px; height: 34px; margin: -17px 0 0 -17px; border-radius: 50%; background: #fff;
  box-shadow: 0 2px 8px rgba(0,0,0,0.35); display: grid; place-items: center; font-size: 13px; color: #24292f; font-weight: 700; }
.cpd .juxt .tag { position: absolute; top: 10px; padding: 3px 9px; border-radius: 999px; font-size: 0.72em; font-weight: 700; color: #fff; background: rgba(0,0,0,0.55); }
.cpd .juxt .tag.l { left: 10px; } .cpd .juxt .tag.r { right: 10px; background: rgba(227,73,72,0.9); }
.cpd .prompt { text-align: center; font-size: 0.84em; color: var(--c-ink2); margin: 10px auto 0; max-width: 40em; font-style: italic; }
.cpd .widget { background: var(--c-card); border: 1px solid var(--c-line); border-radius: 14px; padding: 16px; box-shadow: var(--c-shadow); margin: 0 0 8px; }
.cpd .widget .row { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 8px; }
.cpd .widget .lab { font-size: 0.74em; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--c-muted); margin-right: 4px; min-width: 64px; }
.cpd .widget .two { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 18px; margin-top: 6px; }
.cpd .widget .wnote { margin: 12px 0 0; font-size: 0.86em; color: var(--c-ink2); line-height: 1.5; border-top: 1px solid var(--c-line); padding-top: 10px; }
.cpd .donorhead { font-weight: 650; font-size: 0.9em; margin: 10px 0 2px; }
.cpd .selpill { display: inline-block; margin-left: 6px; padding: 1px 9px; border-radius: 999px; background: var(--c-soft); border: 1px solid var(--c-line); font-size: 0.92em; }
.cpd .xscroll { overflow-x: auto; }
.cpd table.mini { width: 100%; border-collapse: collapse; font-size: 0.82em; font-variant-numeric: tabular-nums; margin: 4px 0; display: table; }
.cpd table.mini th, .cpd table.mini td { border: 0; border-bottom: 1px solid var(--c-line); padding: 5px 7px; text-align: left; white-space: nowrap; }
.cpd table.mini th { color: var(--c-muted); font-weight: 600; }
.cpd table.mini, .cpd table.mini tr, .cpd table.mini thead, .cpd table.mini tbody { border: 0 !important; background: transparent; }
.cpd table.mini .r { text-align: right; } .cpd table.mini .kind { color: var(--c-muted); font-size: 0.92em; }
.cpd table.matrix td, .cpd table.matrix th { text-align: center; } .cpd table.matrix td.diag { color: var(--c-muted); }
.cpd table.matrix td.cur { box-shadow: inset 0 0 0 2px var(--c-blue); border-radius: 4px; font-weight: 700; }
.cpd .ci { display: inline-flex; align-items: center; gap: 7px; } .cpd .ci svg { flex: none; } .cpd .ci.low span { color: var(--c-crit); font-weight: 600; }
.cpd .verdict { font-weight: 600; white-space: normal !important; min-width: 110px; }
.cpd .verdict.ok { color: var(--c-good); } .cpd .verdict.bad { color: var(--c-crit); } .cpd .verdict.skip { color: var(--c-muted); }
.cpd .pvstats { display: flex; flex-wrap: wrap; gap: 18px; font-size: 0.86em; color: var(--c-ink2); margin: 2px 0 6px; } .cpd .pvstats b { color: var(--c-ink); font-size: 1.15em; }
.cpd .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.8em; color: var(--c-ink2); margin: 4px 0 0; }
.cpd .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 5px; vertical-align: 0; } .cpd .legend .m { color: var(--c-muted); }
.cpd svg text { fill: var(--c-ink2); font-size: 12px; } .cpd svg text.tick { fill: var(--c-muted); font-size: 11px; }
.cpd svg text.mlab { fill: var(--c-ink); font-size: 12.5px; } .cpd svg text.mlab.ours { font-weight: 700; }
.cpd .widget .row > span { display: inline-flex; flex-wrap: wrap; gap: 6px; }
.cpd .plotbox { max-width: 680px; margin: 0 auto; }
.cpd select { font: inherit; font-size: 0.85em; padding: 4px 8px; border-radius: 8px; border: 1px solid var(--c-line); background: var(--c-soft); color: var(--c-ink); }
.cpd .fig5 img { display: block; width: 100%; height: auto; }
.cpd-tip { position: fixed; z-index: 50; display: none; pointer-events: none; max-width: 280px; padding: 7px 10px; border-radius: 8px; font-size: 12.5px; line-height: 1.45;
  background: #ffffff; color: #1f2328; border: 1px solid rgba(31,35,40,0.15); box-shadow: 0 6px 18px rgba(31,35,40,0.14); }
.cpd-tip .m { color: #8b949e; }
html[data-theme="dark"] .cpd-tip { background: #2d333b; color: #e6edf3; border-color: rgba(230,237,243,0.16); }
.cpd .cite { position: relative; background: var(--c-soft); border: 1px solid var(--c-line); border-radius: 12px; padding: 14px 16px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.8em; line-height: 1.55; white-space: pre-wrap; }
.cpd .cite button { position: absolute; top: 8px; right: 8px; font: inherit; font-size: 0.85em; padding: 3px 10px; border-radius: 8px; border: 1px solid var(--c-line);
  background: var(--c-card); color: var(--c-ink); cursor: pointer; }
@media (max-width: 760px) {
  .cpd .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } .cpd .steps, .cpd .split { grid-template-columns: minmax(0, 1fr); }
  .cpd .hero { padding: 18px 16px 14px; } .cpd .tagline { font-size: 1.12em; } .cpd .widget .two { grid-template-columns: minmax(0, 1fr); }
}
</style>

<div class="cpd">

<div class="hero">
  <p class="tagline">Distill every model on a diffusion backbone into one student, and <em>pick the mix before training</em>.</p>
  <div class="badges"><span class="badge">Under review · 2026</span><span class="badge">On-policy distillation</span><span class="badge">SD3.5-Medium</span></div>
  <div class="pills">
    <a class="pill" href="#try-the-screen"><svg viewBox="0 0 16 16"><path d="M4 2.5v11l9-5.5z"/></svg>Try the screen</a>
    <a class="pill" href="#visual-comparison"><svg viewBox="0 0 16 16"><path d="M1 3h6v10H1zM9 3h6v10H9z"/></svg>Compare samples</a>
    <a class="pill" href="#results"><svg viewBox="0 0 16 16"><path d="M2 13h12v1H2zM3 8h2v4H3zM7 5h2v7H7zM11 2h2v10h-2z"/></svg>Results</a>
    <span class="pill ghost"><svg viewBox="0 0 16 16"><path d="M3 1h7l3 3v11H3zM9 2v3h3"/></svg>Paper · after review</span>
    <span class="pill ghost"><svg viewBox="0 0 16 16"><path d="M5.5 4 1.5 8l4 4 1-1-3-3 3-3zm5 0-1 1 3 3-3 3 1 1 4-4z"/></svg>Code · after review</span>
  </div>
  <div class="figcard" style="margin:0"><img src="/images/projects/2026-11-composeopd/fig1-teaser.png" alt="Figure 1: Composing teachers and published students." loading="eager"></div>
  <p class="cap" style="margin-bottom:0"><b>Figure 1.</b> (a) The student regresses onto the base velocity plus the receiver's delta plus weighted donor deltas;
  routing is Γ = I. (b) Published OPD students trade accuracy against preference; ours lies above and to the right of all four.
  In parentheses: how often a VLM judge, used neither in training nor in selection, prefers ours.</p>
</div>

<div class="stats">
  <div class="stat"><div class="num">0.85 / 0.98</div><div class="lbl">rank correlation between composed teachers and trained students (49 comparisons / prospective test)</div></div>
  <div class="stat"><div class="num">≈1 GPU-hour</div><div class="lbl">to screen one donor for all three receivers, against ≈8.5 to fine-tune and evaluate one candidate</div></div>
  <div class="stat"><div class="num red">11 / 12</div><div class="lbl">in-domain comparisons with four published OPD students significantly higher; none lower</div></div>
  <div class="stat"><div class="num red">0.59–0.75</div><div class="lbl">VLM-judge win rate of ours against each published student</div></div>
</div>

<div class="tldr"><span class="k">TL;DR</span><p>Reward fine-tuning has left many models on one diffusion backbone, from single-reward experts to published distilled
students, each strongest on a different kind of prompt. On-policy distillation (OPD) usually routes each prompt to one of them, so their
strengths never combine. <b>ComposeOPD</b> distills a composed target instead, the receiver's velocity plus weighted velocity deltas of every
other model, and chooses the weights Γ before any training: a student learns to reproduce its target, so sampling the composed target
previews the student. Composed from Self-OPD and the DiffusionOPD teachers, our student matches or exceeds four published OPD students on all
three in-domain scores.</p></div>

<h2 class="sec" id="method">How it works</h2>
<p class="sub">One composed target, one inference-only screen, then standard on-policy distillation.</p>
<div class="steps">
  <div class="step"><span class="n">1</span><h4>Compose</h4><p>Every model on the backbone contributes a velocity delta, weighted per prompt domain by a receiver-by-donor matrix Γ.
  Routing, task arithmetic, convex mixing and extrapolation are special cases, and published students enter without their data or rewards.</p></div>
  <div class="step"><span class="n">2</span><h4>Screen before training</h4><p>Sample the composed teacher on held-out training prompts and keep a donor weight only if the
  receiver's own reward, HPSv2.1 and UnifiedReward all stay within noise, with every smaller weight passing too.</p></div>
  <div class="step"><span class="n">3</span><h4>Distill</h4><p>Roll out the student on-policy and regress each step mean onto the composed target, with DiffusionOPD's objective;
  only the target changes.</p></div>
</div>
<div class="figcard"><img src="/images/projects/2026-11-composeopd/fig2-method.png" alt="Figure 2: ComposeOPD." loading="lazy"></div>
<p class="cap"><b>Figure 2.</b> ComposeOPD: (a) the composed target, (b) the screen before training, (c) on-policy distillation onto the
target, and (d) the preview of the trained students by the composed teachers.</p>

<h3 class="sec3" id="try-the-screen">Try the screen</h3>
<p class="sub">Algorithm 1 replayed on the paper's probe tables: each donor is added to the receiver's own model at γ = 0.25, 0.5 and 1 on 128 held-out
training prompts; a γ is harmful if a check is significantly lower (90% interval), and kept only if every smaller γ also passed.</p>
<div class="widget">
  <div class="row"><span class="lab">Teachers</span><span id="cpd-scr-set"></span></div>
  <div class="row"><span class="lab">Receiver</span><span id="cpd-scr-rec"></span></div>
  <div class="two"><div id="cpd-scr-body"></div><div><div class="donorhead">Selected Γ</div><div id="cpd-scr-matrix"></div>
    <p class="cap" style="margin:6px 0 0">Rows: receivers; columns: donors.</p></div></div>
  <p class="wnote" id="cpd-scr-note"></p>
</div>

<h2 class="sec" id="results">Results</h2>
<p class="sub">The composed teacher previews the student, composition lifts image quality across domains, and composing published students
combines their strengths.</p>

<h3 class="sec3" id="composition-strength">Composition strength at inference</h3>
<p class="sub">Composed teachers sampled before any training (setting 1, 512 test prompts per domain). A symmetric gate gives every donor the same
weight γ: γ = 0 is routing and γ = 1 the plain additive target.</p>
<div class="widget">
  <div class="row"><span class="lab">Target</span><span id="cpd-inf-target"></span></div>
  <div class="xscroll"><table class="mini" id="cpd-inf-table"></table></div>
  <p class="wnote" id="cpd-inf-note"></p>
</div>

<h3 class="sec3" id="preview">The teacher previews the student</h3>

<div class="widget">
  <div class="row"><span class="lab">Metrics</span><span id="cpd-pv-group"></span></div>
  <div class="row"><span class="lab">Family</span><span id="cpd-pv-fam"></span></div>
  <div class="pvstats" id="cpd-pv-stats"></div>
  <div class="fig5 plotbox" id="cpd-pv-plot"><img src="/images/projects/2026-11-composeopd/fig5-preview.png" alt="Figure 5: Changes measured on composed teachers against the same changes in trained students." loading="lazy"></div>
  <div class="legend" id="cpd-pv-legend"></div>
</div>
<p class="cap"><b>Figure 5, interactive.</b> Each point is one metric of one change of Γ, measured on composed teachers (training prompts, inference
only) against the same change in the trained students (test prompts); hover for details, click the chips to filter. Image quality is previewed
closely (Spearman 0.94 over 213 cells, 0.85 over all 49 comparisons, 0.98 on eight compositions trained after the rule was fixed); accuracy
changes mostly sit within the noise of a 128-prompt probe.</p>

<div class="tabletitle">Table 2. Probe-selected composition minus routing (test prompts, 3 seeds)</div>
<div class="tablecard" markdown="1">

| Setting | GenEval | OCR | PickScore (in-domain) | PickScore, GenEval prompts | PickScore, OCR prompts | Judge win rate |
|---|---|---|---|---|---|---|
| Setting 1, from scratch | +0.002 | +0.004 | **+0.127** | **+0.347** | **+0.975** | **0.678** |
| Setting 2, fine-tuned | +0.002 | +0.003 | −0.006 | **+0.461** | **+0.152** | **0.626** |
| Setting 2, from scratch (300 updates) | +0.003 | −0.002 | **−0.021** | **+0.449** | **+0.065** | **0.627** |
| Setting 2, from scratch (1000 updates) | +0.010 | −0.003 | +0.002 | **+0.500** | **+0.108** | **0.622** |

</div>
<p class="cap">Composition lifts image quality on the other domains' prompts in every setting, and the 95% intervals exclude GenEval or OCR
losses above 1.5 points (bold: the interval excludes 0, or 0.5 for the judge).</p>

<div class="tabletitle">Table 4. All models under one protocol (test prompts, means)</div>
<div class="tablecard ours" markdown="1">

| Model | GenEval | OCR | PickScore | PickScore, OCR prompts | HPSv2.1, DrawBench |
|---|---|---|---|---|---|
| DiffusionOPD student | 0.944 | 0.949 | 23.75 | 22.36 | 0.338 |
| Self-OPD | **0.981** | 0.950 | 22.78 | **24.46** | 0.316 |
| DreOPD | 0.957 | 0.944 | 23.75 | 22.21 | **0.349** |
| Flow-OPD | 0.901 | 0.934 | 22.79 | 23.31 | 0.306 |
| **ComposeOPD (ours)** | 0.973 | **0.962** | **23.79** | 23.97 | 0.338 |

</div>
<p class="cap">Each published student leads somewhere and trails elsewhere; ours is among the top two in every column of the paper's table,
and no published model is (bold: highest).</p>
<div class="widget">
  <div class="row"><span class="lab">x axis</span><select id="cpd-mx" aria-label="x axis"></select><span class="lab" style="margin-left:12px">y axis</span><select id="cpd-my" aria-label="y axis"></select></div>
  <div class="plotbox" id="cpd-m-plot"></div>
  <p class="wnote" id="cpd-m-note"></p>
</div>
<p class="cap">Pick any two metrics of Table 4: published students trade one domain against another. Hover a model for the VLM judge's preference.</p>

<h2 class="sec" id="visual-comparison">Visual comparison</h2>
<p class="sub">Unselected samples: the first prompt of each test set, with the same initial noise for every model. Drag the slider to compare
ours with a published student.</p>
<div class="cmp" id="cpd-cmp">
  <div class="row"><span class="lab">Prompt</span><span id="cpd-prompts"></span></div>
  <div class="row"><span class="lab">Compare</span><span id="cpd-bases"></span></div>
  <div class="juxt" id="cpd-juxt" aria-label="Before and after comparison">
    <img id="cpd-base" alt=""><img id="cpd-ours" class="top" alt="">
    <div class="bar" id="cpd-bar"></div><div class="knob" id="cpd-knob">⇆</div>
    <span class="tag l" id="cpd-tagl"></span><span class="tag r">ComposeOPD (ours)</span>
  </div>
  <p class="prompt" id="cpd-prompt"></p>
</div>
<p class="cap">The requested text on the OCR prompt is partly hidden for the DiffusionOPD student and DreOPD and complete for Flow-OPD,
Self-OPD and ours. Figure 6 in the paper shows all five models side by side:</p>
<div class="figcard"><img src="/images/projects/2026-11-composeopd/fig6-samples.webp" alt="Figure 6: Unselected samples, the first prompt of each test set with the same noise in each row." loading="lazy"></div>
<p class="cap"><b>Figure 6.</b> Columns: the DiffusionOPD student, DreOPD, Flow-OPD, Self-OPD and ours (framed); rows: GenEval, OCR, PickScore
and DrawBench prompts.</p>

<h2 class="sec" id="citation">Citation</h2>
<p class="sub">The paper is under review; the PDF and code will be linked here after review.</p>
<div class="cite" id="cpd-cite"><button type="button" id="cpd-copy">Copy</button>"ComposeOPD: Composing Diffusion Teachers and Published Students, Screened Before Training." Manuscript under review, 2026.</div>

</div>

<script>
(function () {
  var S = "/images/projects/2026-11-composeopd/samples/";
  var prompts = [
    ["geneval", "GenEval", "a photo of two wine glasses"],
    ["ocr", "OCR", "A high-fashion runway with a sleek, modern backdrop displaying \"Spring Collection 2024\"…"],
    ["pickscore", "PickScore", "Beautiful woman in skin tight body suite working out in the gymq"],
    ["drawbench", "DrawBench", "New York Skyline with 'Google Research Pizza Cafe' written with fireworks on the sky."]
  ];
  var bases = [["selfopd", "Self-OPD"], ["released", "DiffusionOPD student"], ["dreopd", "DreOPD"], ["flowopd", "Flow-OPD"]];
  var st = { p: 1, b: 0, x: 50 };
  function chips(id, items, cur, pick) {
    var host = document.getElementById(id); host.innerHTML = "";
    items.forEach(function (it, i) {
      var b = document.createElement("button"); b.type = "button"; b.className = "chip"; b.textContent = it[1];
      b.setAttribute("aria-pressed", i === cur); b.style.marginRight = "6px"; b.onclick = function () { pick(i); }; host.appendChild(b);
    });
  }
  function setX(x) {
    st.x = Math.max(0, Math.min(100, x));
    document.getElementById("cpd-ours").style.clipPath = "inset(0 0 0 " + st.x + "%)";
    document.getElementById("cpd-bar").style.left = st.x + "%"; document.getElementById("cpd-knob").style.left = st.x + "%";
  }
  function render() {
    var p = prompts[st.p], b = bases[st.b];
    chips("cpd-prompts", prompts.map(function (q) { return [q[0], q[1]]; }), st.p, function (i) { st.p = i; render(); });
    chips("cpd-bases", bases, st.b, function (i) { st.b = i; render(); });
    document.getElementById("cpd-base").src = S + p[0] + "-" + b[0] + ".webp";
    document.getElementById("cpd-base").alt = b[1] + " sample";
    document.getElementById("cpd-ours").src = S + p[0] + "-ours.webp";
    document.getElementById("cpd-ours").alt = "ComposeOPD sample";
    document.getElementById("cpd-tagl").textContent = b[1];
    document.getElementById("cpd-prompt").textContent = "“" + p[2] + "”";
    setX(st.x);
  }
  var j = document.getElementById("cpd-juxt"), drag = false;
  function at(e) { var r = j.getBoundingClientRect(); var cx = (e.touches ? e.touches[0].clientX : e.clientX); setX((cx - r.left) / r.width * 100); }
  j.addEventListener("pointerdown", function (e) { drag = true; at(e); });
  window.addEventListener("pointermove", function (e) { if (drag) at(e); });
  window.addEventListener("pointerup", function () { drag = false; });
  j.tabIndex = 0;
  j.addEventListener("keydown", function (e) { if (e.key === "ArrowLeft") setX(st.x - 5); if (e.key === "ArrowRight") setX(st.x + 5); });
  document.getElementById("cpd-copy").onclick = function () {
    var t = document.getElementById("cpd-cite").textContent.replace(/^Copy/, "");
    if (navigator.clipboard) navigator.clipboard.writeText(t);
    this.textContent = "Copied"; var b = this; setTimeout(function () { b.textContent = "Copy"; }, 1500);
  };
  render();
})();
</script>
<script src="/files/projects/2026-11-composeopd/composeopd.js" defer></script>
