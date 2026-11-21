---
title: "RefAudit: Re-evaluating Diffusion Step Caching as Step Scheduling Against Converged References"
collection: projects
permalink: /projects/2026-11-refaudit/
date: 2026-11-16
# No venue under review is named on this page; author list, paper, code and BibTeX links to be added later.
status: "Under review"
venue: "Under review"
excerpt: "The standard protocol for scoring training-free step caching rewards samplers that share the 50-step reference's error; RefAudit separates accuracy from shared error, and under it optimized static schedules with UniPC match or beat every cache we ran on FLUX.1-dev and SD3.5."
teaser: "/images/projects/2026-11-refaudit/teaser.png"
tags: ["diffusion models", "evaluation", "step caching", "ODE solvers"]
# paperurl: ""
# pdfurl: ""
# codeurl: ""
# bibtexurl: ""
# publication: "/publication/2026-11-refaudit"
---

<style>
/* RefAudit project page: styles scoped to .ra / .ra-* so the rest of the site is untouched; colors follow the theme */
.ra{--ra-blue:#1f5fa8;--ra-blue-soft:rgba(31,95,168,.08);--ra-gray:#6f767e;--ra-gray-soft:rgba(127,127,127,.09);--ra-green:#26993a;--ra-red:#d62728;--ra-line:rgba(127,127,127,.28);--ra-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .ra{--ra-blue:#78b0f0;--ra-blue-soft:rgba(120,176,240,.14);--ra-gray:#c3c9cf;--ra-gray-soft:rgba(255,255,255,.07);--ra-green:#62cf72;--ra-red:#f26b67;--ra-line:rgba(255,255,255,.2);--ra-card:rgba(255,255,255,.04)}
.ra-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.ra-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--ra-blue-soft);color:var(--ra-blue)}
.ra-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.ra-tldr b{color:var(--ra-blue)}
.ra-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.ra-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.ra-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.ra-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--ra-line)}
.ra-btn.muted{background:transparent;color:var(--ra-gray)!important;border:1px dashed var(--ra-line);cursor:default}
.ra-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .ra-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.ra-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--ra-line);border-radius:14px;background:var(--ra-card)}
.ra-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.ra-fig.narrow img{max-width:min(100%,520px);margin:0 auto}
.ra-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.ra-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--ra-blue);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .ra-tag{color:#16202b}
.ra-fig figcaption b.ra-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.ra-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.ra-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--ra-line);background:var(--ra-card);position:relative;overflow:hidden}
.ra-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--ra-c,var(--ra-blue))}
.ra-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;color:var(--ra-c,var(--ra-blue));font-variant-numeric:tabular-nums}
.ra-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.ra-cmp{display:grid;grid-template-columns:auto 1fr auto 1fr 1.05fr;gap:.55em .6em;align-items:stretch;margin:0 0 1em;font-size:.88em}
.ra-cmp .h{font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:0 .2em;align-self:end}
.ra-cmp .h.s{color:var(--ra-gray)}.ra-cmp .h.c,.ra-cmp .h.o{color:var(--ra-blue)}
.ra-cmp .k{font-weight:700;align-self:center;padding-right:.3em}
.ra-cmp .s,.ra-cmp .c,.ra-cmp .o{padding:.55em .75em;border-radius:10px;line-height:1.4}
.ra-cmp div.s{background:var(--ra-gray-soft);border:1px solid var(--ra-line);color:var(--ra-gray)}
.ra-cmp div.c{background:var(--ra-blue-soft);border:1px solid var(--ra-blue)}
.ra-cmp div.o{border-left:3px solid var(--ra-blue);border-radius:4px}
.ra-cmp .arr{align-self:center;color:var(--ra-gray);font-size:1.1em}
.ra-why{font-size:.9em;margin:0 0 1.6em;padding:.6em .9em;border-radius:10px;background:var(--ra-gray-soft)}
.ra-box{padding:1em 1.2em;border-radius:14px;background:var(--ra-blue-soft);border:1px solid var(--ra-line);margin:0 0 1.6em;line-height:1.65}
.ra-box p{margin:0}
.ra-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--ra-line);border-radius:14px;background:var(--ra-card)}
.ra-table{overflow-x:auto}.ra-table table{font-size:.8em;margin:.6em 0;min-width:560px}
.ra-table td strong{color:var(--ra-green)}
.ra-table p{font-size:.86em;margin:.2em}
.ra-slider{position:relative;display:block;width:100%;aspect-ratio:5/4;max-width:520px;margin:0 auto;border-radius:10px;overflow:hidden;user-select:none;touch-action:pan-y;cursor:ew-resize;background:#000}
.ra-slider img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;margin:0;max-width:none;image-rendering:pixelated;border-radius:0}
.ra-slider .after{clip-path:inset(0 0 0 var(--pos,50%))}
.ra-slider .line{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:3px;margin-left:-1.5px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.5)}
.ra-slider .knob{position:absolute;top:50%;left:var(--pos,50%);width:36px;height:36px;margin:-18px 0 0 -18px;border-radius:50%;background:#fff;color:#222;display:grid;place-items:center;font-weight:800;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.ra-slider:focus-visible .knob{box-shadow:0 0 0 3px var(--ra-blue)}
.ra-slider .lbl{position:absolute;top:8px;padding:.2em .6em;border-radius:6px;font-size:.72em;font-weight:700;color:#fff}
.ra-slider .lbl.l{left:8px;background:var(--ra-green)}.ra-slider .lbl.r{right:8px;background:#444}
.ra-ix{font-size:.9em}
.ra-ctl{display:grid;grid-template-columns:auto 1fr;gap:.35em .9em;align-items:center;margin:0 0 .7em}
.ra-ctl .l{font-size:.68em;font-weight:800;letter-spacing:.06em;text-transform:uppercase;opacity:.65}
.ra-seg{display:flex;flex-wrap:wrap;gap:.35em}
.ra-seg button{font:inherit;font-size:.86em;line-height:1.3;padding:.3em .85em;border-radius:999px;border:1px solid var(--ra-line);background:transparent;color:var(--global-text-color);cursor:pointer;transition:background .15s,border-color .15s}
.ra-seg button:hover{border-color:var(--ra-gray)}
.ra-seg button[aria-pressed=true]{background:var(--global-text-color);border-color:var(--global-text-color);color:var(--global-bg-color)}
.ra-seg.ref button[data-v=x50][aria-pressed=true]{background:var(--ra-gray);border-color:var(--ra-gray);color:#fff}
.ra-seg.ref button[data-v=conv][aria-pressed=true]{background:var(--ra-blue);border-color:var(--ra-blue);color:#fff}
html[data-theme="dark"] .ra-seg.ref button[aria-pressed=true]{color:#16202b}
.ra-rc{display:flex;flex-wrap:wrap;gap:.4em .9em;margin:0 0 .7em;font-size:.88em}
.ra-rc span b{font-variant-numeric:tabular-nums}
.ra-rank{position:relative;margin:.2em 0 .2em}
.ra-rank .row{position:absolute;left:0;right:0;height:30px;display:flex;align-items:center;gap:.6em;transition:top .5s cubic-bezier(.2,.8,.2,1)}
.ra-rank .no{width:22px;height:22px;border-radius:50%;border:1px solid var(--ra-line);display:grid;place-items:center;font-size:.8em;font-weight:700;flex:none}
.ra-rank .row.top .no{background:var(--ra-top,var(--ra-gray));border-color:transparent;color:#fff}
html[data-theme="dark"] .ra-rank .row.top .no{color:#16202b}
.ra-rank .nm{width:170px;flex:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-size:.92em}
.ra-rank .tr{position:relative;flex:1;height:12px;margin-right:66px;border-radius:999px;background:var(--ra-gray-soft)}
.ra-rank .fl{position:absolute;left:0;top:0;bottom:0;border-radius:999px;transition:width .5s cubic-bezier(.2,.8,.2,1)}
.ra-rank .vl{position:absolute;top:-5px;padding-left:6px;font-size:.88em;font-weight:700;font-variant-numeric:tabular-nums;white-space:nowrap;transition:left .5s cubic-bezier(.2,.8,.2,1)}
.ra-rank .k-cache .fl{background:var(--ra-red)}.ra-rank .k-cache_unipc .fl{box-shadow:inset 0 0 0 2px var(--ra-red)}
.ra-rank .k-oss_unipc .fl{background:var(--ra-green)}.ra-rank .k-oss_euler .fl{box-shadow:inset 0 0 0 2px var(--global-text-color)}
.ra-rank .k-uniform .fl{background:var(--ra-gray);opacity:.55}
.ra-rank .r50{position:absolute;top:-8px;bottom:-4px;border-left:2px dashed var(--ra-gray);pointer-events:none;transition:left .5s}
.ra-rank .r50 span{position:absolute;top:-16px;transform:translateX(-50%);font-size:.72em;white-space:nowrap;opacity:.8}
.ra-axis{display:flex;justify-content:space-between;font-size:.72em;opacity:.7;margin:.5em 66px 0 202px}
.ra-read{margin:.8em 0 0;padding:.5em .8em;border-left:3px solid var(--ra-gray);border-radius:4px;background:var(--ra-gray-soft);font-size:.92em;line-height:1.5}
.ra-read.conv{border-left-color:var(--ra-blue);background:var(--ra-blue-soft)}
.ra-split{display:grid;grid-template-columns:150px 1fr;gap:.9em 1em;align-items:center}
.ra-split .eq{font-size:.82em;opacity:.85;font-variant-numeric:tabular-nums}
.ra-mini{display:flex;align-items:center;gap:.5em;height:17px}
.ra-mini .ml{width:58px;flex:none;font-size:.74em;text-align:right;opacity:.75}
.ra-mini .mt{position:relative;flex:1;height:9px;margin-right:38px;border-radius:999px;background:var(--ra-gray-soft)}
.ra-mini .mf{position:absolute;left:0;top:0;bottom:0;border-radius:999px}
.ra-mini .mv{position:absolute;top:-5px;padding-left:5px;font-size:.78em;font-weight:700;font-variant-numeric:tabular-nums}
.ra-mini .m-conv{background:var(--ra-blue)}.ra-mini .m-std{background:var(--ra-gray)}
.ra-mini .m-c{background:repeating-linear-gradient(45deg,var(--ra-red) 0 5px,transparent 5px 9px);box-shadow:inset 0 0 0 1px var(--ra-red)}
.t-std{color:var(--ra-gray);font-weight:700}.t-conv{color:var(--ra-blue);font-weight:700}.t-c{color:var(--ra-red);font-weight:700}
.ra-cards{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:.55em}
.ra-card{position:relative;margin:0;padding:.3em;border-radius:10px;border:2px solid transparent;background:var(--ra-gray-soft)}
.ra-card img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;margin:0}
.ra-card .t{font-size:.74em;line-height:1.3;margin:.35em .1em 0;min-height:2.6em}
.ra-card .p{font-size:.9em;font-weight:800;font-variant-numeric:tabular-nums;margin:0 .1em}
.ra-card.best{border-color:var(--ra-bc)}
.ra-card .bd{position:absolute;top:.6em;right:.6em;font-size:.62em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.15em .55em;border-radius:999px;color:#fff;background:var(--ra-bc)}
html[data-theme="dark"] .ra-card .bd{color:#16202b}
@media (max-width:900px){.ra-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.ra-rank .nm{width:108px;font-size:.84em}.ra-axis{margin-left:140px}.ra-cards{grid-template-columns:repeat(2,minmax(0,1fr))}.ra-split{grid-template-columns:1fr}.ra-ctl{grid-template-columns:1fr}.ra-cmp{grid-template-columns:1fr}.ra-cmp .arr,.ra-cmp .h,.ra-cmp span:empty{display:none}.ra-cmp .k{margin-top:.6em}.ra-tldr{font-size:1.04em}}
</style>

<div class="ra">
<div class="ra-kicker"><span>Diffusion transformers</span><span>Training-free acceleration</span><span>Evaluation protocol</span></div>
<p class="ra-tldr">Diffusion step caches are scored by their fidelity to the 50-step output, and that score rewards sharing the 50-step sampler's error. <b>RefAudit</b> scores against a converged reference instead, and an optimized static schedule with UniPC then matches or beats every cache we ran on FLUX.1-dev and SD3.5.</p>
<div class="ra-btns"><a class="ra-btn" href="#ra-try">▶&nbsp; Try it: switch the reference</a><a class="ra-btn ghost" href="#key-results">Key results</a><a class="ra-btn ghost" href="#what-refaudit-changes">The protocol</a><span class="ra-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="ra-fig"><img src="/images/projects/2026-11-refaudit/fig1-refaudit-overview.webp" alt="Figure 1: RefAudit overview" loading="lazy"><figcaption><span class="ra-tag">Figure 1</span><b class="ra-title">RefAudit: the 50-step reference favors the cache, which shares its error.</b>On FLUX.1-dev at 15 steps, MagCache is closer than OSS + UniPC to the 50-step output only because more of its error points the same way as the reference's (shared error 43.8 vs 30.1, ×10⁻³ MSE). Against a converged reference the order reverses.</figcaption></figure>
</div>

## At a glance

<div class="ra"><div class="ra-stats">
<div class="ra-stat" style="--ra-c:var(--ra-gray)"><div class="v">up to 18 dB</div><div class="d">change in the gap between two fixed samplers when only the solver family of the 50-step reference changes</div></div>
<div class="ra-stat" style="--ra-c:var(--ra-green)"><div class="v">1.0–3.1 dB</div><div class="d">OSS + UniPC closer to the converged reference than every cache as shipped on FLUX.1-dev and SD3.5, at the same speed</div></div>
<div class="ra-stat"><div class="v">#4 → #1</div><div class="d">rank of OSS + UniPC among four runs, against the 50-step output vs. the converged reference (FLUX-15, SD3.5-14)</div></div>
<div class="ra-stat" style="--ra-c:var(--ra-red)"><div class="v">84 / 100</div><div class="d">prompts on which DreamSim puts OSS-15 + UniPC closer than MagCache-15 to the converged reference (34 against the 50-step output)</div></div>
</div></div>

## What RefAudit changes

<div class="ra"><div class="ra-cmp">
<span></span><span class="h s">Standard protocol</span><span></span><span class="h c">RefAudit</span><span class="h o">What it shows</span>
<span class="k">Score</span><div class="s">PSNR, LPIPS to the 50-step output x₅₀</div><span class="arr">→</span><div class="c">(i) split the score: error to x* + constant − shared error</div><div class="o">MagCache's lead on x₅₀ is shared error, not accuracy</div>
<span class="k">Reference</span><div class="s">x₅₀, taken as given</div><span class="arr">→</span><div class="c">(ii) converged x*, checked across solver families</div><div class="o">the reference's family alone moves a gap by up to 18 dB</div>
<span class="k">Baseline</span><div class="s">fewer uniform steps</div><span class="arr">→</span><div class="c">(iii) optimized schedule (OSS) at equal NFE, same solver</div><div class="o">OSS + UniPC 1.0–3.1 dB closer than every cache</div>
</div>
<p class="ra-why"><b>Why (iii):</b> under Euler, a step cache that reuses the last velocity <i>is</i> Euler on its computed steps, and residual reuse nearly so. Step caching is step scheduling, so its fair competitor is the best schedule with the same number of steps.</p>
<figure class="ra-fig" id="ra-eq"><div class="ra-ix"><div class="ra-ctl"><span class="l">Setting</span><div class="ra-seg" id="ra-settings"></div></div><div class="ra-split" id="ra-split"></div><p class="ra-read conv" id="ra-read2"></p></div><figcaption><span class="ra-tag">Interactive</span><b class="ra-title">Where the 50-step score comes from (Eq. 1).</b><span class="t-std">MSE(x, x₅₀)</span> = <span class="t-conv">MSE(x, x*)</span> + MSE(x₅₀, x*) − <span class="t-c">c(x)</span>, where c is the error a run shares with the 50-step output. For each run: its error to the converged reference (blue), the shared error (red, subtracted) and what remains, the standard score (gray). Pixel MSE ×10⁻³, mean over prompts; pick a model above.</figcaption></figure></div>

## Summary

<div class="ra"><div class="ra-box"><p>Training-free step caching (TeaCache, MagCache and their successors) speeds up diffusion transformers by reusing the transformer's last result at some sampling steps, and it is judged by fidelity to the default 50-step output and compared with sampling at fewer uniform steps. We show that this protocol favors caching: the 50-step output is itself a coarse ODE solve, so fidelity to it rewards samplers that share its error, and under Euler a reuse cache is exactly or nearly Euler on its computed steps, so its fair baseline is an optimized timestep schedule. <b>RefAudit</b> splits the standard score into accuracy and shared error, checks the reference across solver families, and compares methods at equal NFE under the same solver; under it, optimized static schedules with UniPC match or beat every cache we ran on FLUX.1-dev and SD3.5, with no run-time decisions or extra memory.</p></div></div>

## Key results

<div class="ra">
<figure class="ra-fig narrow"><img src="/images/projects/2026-11-refaudit/fig2-reference-family.png" alt="Figure 2: the reference's solver family sets the sign" loading="lazy"><figcaption><span class="ra-tag">Figure 2</span><b class="ra-title">The reference's solver family sets the sign.</b>Against 50-step references, the gap between the same two samplers follows the reference's solver family on every model, by 4–10 dB either way. From 200 steps on, references of both families give the same sign: the cross-family check RefAudit asks of a reference.</figcaption></figure>
<figure class="ra-fig"><img src="/images/projects/2026-11-refaudit/fig3-ranking-and-pareto.png" alt="Figure 3: ranking under each reference and PSNR against computed steps" loading="lazy"><figcaption><span class="ra-tag">Figure 3</span><b class="ra-title">RefAudit reverses the ranking; on the image models, OSS + UniPC is above every cache shown and the tuned grids, at every budget.</b>Ranked against the 50-step output, OSS + UniPC is last of four runs at FLUX-15 and SD3.5-14; ranked against the converged reference it is first and MagCache is last.</figcaption></figure>
<figure class="ra-fig" id="ra-try"><div class="ra-ix"><div class="ra-ctl"><span class="l">Model</span><div class="ra-seg" id="ra-models"></div><span class="l">Budget</span><div class="ra-seg" id="ra-budgets"></div><span class="l">Reference</span><div class="ra-seg ref" id="ra-ref"><button data-v="x50">50-step output x₅₀</button><button data-v="conv">converged x* (RefAudit)</button></div></div><div class="ra-rc" id="ra-rc"></div><div class="ra-rank" id="ra-rank"></div><div class="ra-axis" id="ra-axis"></div><p class="ra-read" id="ra-read"></p></div><figcaption><span class="ra-tag">Interactive</span><b class="ra-title">Switch the reference and watch the ranking change.</b>Mean PSNR at equal computed steps (DrawBench-100 for the image models, 60 VBench prompts for Wan2.1) against the 50-step output or the converged reference x* (UniPC with exact time conditioning, 200 steps). Red: caches as shipped (outline: with UniPC); green: OSS + UniPC; outline: OSS + Euler; gray: tuned uniform grid. The values behind Figure 3.</figcaption></figure>
</div>

<div class="ra"><div class="ra-table" markdown="1">

<span class="ra-tag">Table 3</span> **At equal NFE, OSS matches or beats every cache on FLUX.1-dev and SD3.5.** Paired PSNR differences (dB), OSS + UniPC minus the cache, against the 50-step output (x₅₀) and the converged reference (x\*); speeds are end-to-end on one H100.

| Model | Cache (steps) | Speed: cache / OSS | vs x₅₀ | vs x\* | LPIPS vs x\* | vs best cache under UniPC |
|---|---|---|---|---|---|---|
| FLUX.1-dev | MagCache (15) | 3.22× / 3.19× | −3.00 | **+1.90** | **−0.101** | +0.29 |
| FLUX.1-dev | TeaCache (20) | 2.41× / 2.43× | +1.61 | **+3.72** | **−0.163** | +3.18 |
| SD3.5-Large | MagCache (14) | 3.38× / 3.39× | −3.06 | **+1.43** | **−0.140** | +0.25 (tie) |
| SD3.5-Large | MagCache (18) | 2.66× / 2.67× | −4.46 | **+1.82** | **−0.123** | −0.11 (tie) |
| Wan2.1-1.3B | MagCache (21) | 2.20× / 2.22× | +2.72 | −0.44 | +0.016 | (caches already use UniPC) |

The standard score and RefAudit disagree in sign on most image-model rows; giving the caches the same solver closes most of the gap, and on Wan2.1, whose default sampler is already UniPC, caching keeps a small edge on the default grid.

</div></div>

<div class="ra"><figure class="ra-fig">
<div class="ra-slider" id="ra-slider" style="--pos:50%" tabindex="0" role="slider" aria-label="Drag to compare OSS + UniPC with the PSNR-tuned uniform grid" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50"><img src="/images/projects/2026-11-refaudit/compare-oss-unipc.webp" alt="OSS + UniPC, 15 steps (crop)"><img class="after" src="/images/projects/2026-11-refaudit/compare-uniform-shift24.webp" alt="Uniform UniPC grid at shift 24, 15 steps (crop)"><span class="line"></span><span class="knob">⇆</span><span class="lbl l">OSS + UniPC</span><span class="lbl r">uniform, shift 24 · highest PSNR</span></div>
<img src="/images/projects/2026-11-refaudit/fig4-psnr-artifact.webp" alt="Figure 4: four crops" loading="lazy" style="margin-top:.7em">
<figcaption><span class="ra-tag">Figure 4</span><b class="ra-title">PSNR can select a visible artifact.</b>Drag the divider. A uniform grid tuned for PSNR to the converged reference has the highest PSNR of these four runs but shows a periodic 16-pixel pattern, one transformer token, which distribution metrics register (ΔFD +35.6 against at most +19.0 for any other run). RefAudit therefore reads fidelity together with a distribution metric. FLUX.1-dev, 15 computed steps, one PartiPrompts image.</figcaption>
</figure>
<figure class="ra-fig" id="ra-one"><div class="ra-ix"><div class="ra-ctl"><span class="l">Reference</span><div class="ra-seg ref" id="ra-ref3"><button data-v="x50">50-step output x₅₀</button><button data-v="conv">converged x*</button></div><span class="l">Show</span><div class="ra-seg" id="ra-view3"><button data-v="img">images</button><button data-v="err">|error| to x*</button></div></div><div class="ra-cards" id="ra-cards"></div><p class="ra-read" id="ra-read3"></p></div><figcaption><span class="ra-tag">Interactive</span><b class="ra-title">One prompt, two references.</b>FLUX.1-dev, DrawBench prompt #2 “A red colored dog.”, seed 0, 15 computed steps; PSNR of each run to the selected reference, and the frame marks the 15-step run closest to it. Error maps share one color scale. The same prompt as Figure 1(d).</figcaption></figure></div>

## Links and citation

Paper, code and BibTeX will be linked here once the preprint is public.

<script>
(function () {
  // drag-to-compare slider (pointer and keyboard)
  var s = document.getElementById("ra-slider");
  if (s) {
    var drag = false;
    function set(p) { p = Math.max(0, Math.min(100, p)); s.style.setProperty("--pos", p + "%"); s.setAttribute("aria-valuenow", Math.round(p)); }
    function move(x) { var r = s.getBoundingClientRect(); set((x - r.left) / r.width * 100); }
    s.addEventListener("pointerdown", function (e) { drag = true; s.setPointerCapture(e.pointerId); move(e.clientX); });
    s.addEventListener("pointermove", function (e) { if (drag) move(e.clientX); });
    s.addEventListener("pointerup", function () { drag = false; });
    s.addEventListener("keydown", function (e) {
      var p = parseFloat(s.getAttribute("aria-valuenow"));
      if (e.key === "ArrowLeft") { set(p - 5); e.preventDefault(); } else if (e.key === "ArrowRight") { set(p + 5); e.preventDefault(); }
    });
  }
})();
</script>

{% raw %}
<script>
(function () {
  var D = {"groups":{"FLUX.1-dev":{"10 steps":[{"name":"TaylorSeer","kind":"cache","x50":15.26,"conv":14.13},{"name":"TaylorSeer + UniPC","kind":"cache_unipc","x50":14.52,"conv":14.1},{"name":"OSS + Euler","kind":"oss_euler","x50":20.76,"conv":16.75},{"name":"OSS + UniPC","kind":"oss_unipc","x50":18.06,"conv":17.47},{"name":"tuned uniform + UniPC","kind":"uniform","x50":15.81,"conv":14.0}],"15 steps":[{"name":"MagCache","kind":"cache","x50":22.34,"conv":16.98},{"name":"MagCache + UniPC","kind":"cache_unipc","x50":19.71,"conv":18.25},{"name":"OSS + Euler","kind":"oss_euler","x50":23.46,"conv":17.25},{"name":"OSS + UniPC","kind":"oss_unipc","x50":19.31,"conv":18.84},{"name":"tuned uniform + UniPC","kind":"uniform","x50":19.14,"conv":15.76}],"18\u201320 steps":[{"name":"MagCache (18)","kind":"cache","x50":23.2,"conv":17.07},{"name":"MagCache + UniPC (18)","kind":"cache_unipc","x50":19.98,"conv":18.53},{"name":"TeaCache (20)","kind":"cache","x50":17.77,"conv":15.34},{"name":"TeaCache + UniPC (19.9)","kind":"cache_unipc","x50":18.0,"conv":15.57},{"name":"OSS + Euler (20)","kind":"oss_euler","x50":24.82,"conv":17.4},{"name":"OSS + UniPC (20)","kind":"oss_unipc","x50":19.34,"conv":19.04},{"name":"tuned uniform + UniPC (20)","kind":"uniform","x50":20.35,"conv":17.31}]},"SD3.5-Large":{"10 steps":[{"name":"TeaCache","kind":"cache","x50":16.28,"conv":14.2},{"name":"TeaCache + UniPC","kind":"cache_unipc","x50":15.91,"conv":14.71},{"name":"OSS + Euler","kind":"oss_euler","x50":20.18,"conv":15.59},{"name":"OSS + UniPC","kind":"oss_unipc","x50":16.5,"conv":15.63},{"name":"tuned uniform + UniPC","kind":"uniform","x50":16.88,"conv":14.71}],"14 steps":[{"name":"MagCache","kind":"cache","x50":20.74,"conv":15.84},{"name":"MagCache + UniPC","kind":"cache_unipc","x50":17.67,"conv":16.9},{"name":"OSS + Euler","kind":"oss_euler","x50":23.2,"conv":16.12},{"name":"OSS + UniPC","kind":"oss_unipc","x50":17.51,"conv":17.3}],"18 steps":[{"name":"MagCache","kind":"cache","x50":22.43,"conv":16.13},{"name":"MagCache + UniPC","kind":"cache_unipc","x50":18.22,"conv":17.49},{"name":"OSS + Euler","kind":"oss_euler","x50":25.06,"conv":16.3},{"name":"OSS + UniPC","kind":"oss_unipc","x50":17.97,"conv":17.83}]},"Wan2.1-1.3B":{"10 steps":[{"name":"TaylorSeer","kind":"cache","x50":12.54,"conv":12.03},{"name":"OSS + Euler","kind":"oss_euler","x50":17.87,"conv":16.1},{"name":"OSS + UniPC","kind":"oss_unipc","x50":20.65,"conv":16.9},{"name":"tuned uniform + UniPC","kind":"uniform","x50":18.45,"conv":16.06}],"21 steps":[{"name":"MagCache","kind":"cache","x50":25.43,"conv":17.63},{"name":"OSS + Euler","kind":"oss_euler","x50":18.19,"conv":16.12},{"name":"OSS + UniPC","kind":"oss_unipc","x50":28.15,"conv":17.19},{"name":"tuned uniform + UniPC","kind":"uniform","x50":20.55,"conv":16.62}],"25 steps":[{"name":"MagCache","kind":"cache","x50":27.45,"conv":17.59},{"name":"TeaCache","kind":"cache","x50":23.02,"conv":17.54},{"name":"OSS + Euler","kind":"oss_euler","x50":18.31,"conv":16.18},{"name":"OSS + UniPC","kind":"oss_unipc","x50":30.69,"conv":17.34},{"name":"tuned uniform + UniPC","kind":"uniform","x50":23.46,"conv":17.17}]}},"ref50":{"FLUX.1-dev":17.55,"SD3.5-Large":16.46,"Wan2.1-1.3B":17.37},"split":{"FLUX.1-dev, 15 steps":{"rows":[{"name":"MagCache","std":8.62,"conv":26.9,"c":43.82,"rho":0.79,"n":100},{"name":"OSS + Euler","std":6.68,"conv":26.03,"c":44.89,"rho":0.83,"n":100},{"name":"OSS + UniPC","std":16.73,"conv":21.29,"c":30.1,"rho":0.56,"n":100}],"const":25.54},"SD3.5-Large, 14 steps":{"rows":[{"name":"MagCache","std":10.53,"conv":34.24,"c":57.11,"rho":0.78,"n":100},{"name":"OSS + Euler","std":5.97,"conv":33.26,"c":60.7,"rho":0.86,"n":100},{"name":"OSS + UniPC","std":23.59,"conv":28.95,"c":38.77,"rho":0.57,"n":100}],"const":33.41},"Wan2.1-1.3B, 21 steps":{"rows":[{"name":"MagCache","std":3.86,"conv":27.26,"c":51.85,"rho":0.9,"n":16},{"name":"OSS + Euler","std":17.52,"conv":32.56,"c":43.49,"rho":0.65,"n":16},{"name":"OSS + UniPC","std":2.06,"conv":29.01,"c":55.39,"rho":0.95,"n":16}],"const":28.45}},"images":[{"name":"Converged x*","img":"converged","err":null,"x50":24.8,"conv":null},{"name":"50-step output x50","img":"std50","err":"std50_err","x50":null,"conv":24.8},{"name":"MagCache, 15 steps","img":"magcache","err":"magcache_err","x50":29.4,"conv":23.5},{"name":"UniPC on MagCache's 15 steps","img":"unipc_s","err":"unipc_s_err","x50":24.7,"conv":31.3},{"name":"OSS + UniPC, 15 steps","img":"oss_unipc","err":"oss_unipc_err","x50":24.0,"conv":29.8}],"prompt":"A red colored dog."};
  var IMG = "/files/projects/2026-11-refaudit/demo/img/";
  function $(id) { return document.getElementById(id); }
  function seg(el, items, cur, on) {
    el.innerHTML = "";
    items.forEach(function (k) { var b = document.createElement("button"); b.type = "button"; b.textContent = k;
      b.setAttribute("aria-pressed", k === cur); b.onclick = function () { on(k); }; el.appendChild(b); });
  }
  function press(el, v) { el.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.v === v); }); }
  function rankOf(runs, key, name) { return runs.slice().sort(function (a, b) { return b[key] - a[key]; }).findIndex(function (r) { return r.name === name; }) + 1; }
  var X50 = "x₅₀";

  /* ranking under each reference (Figure 3's values) */
  var model = "FLUX.1-dev", budget = "15 steps", ref = "x50", ROW = 34;
  function drawRank() {
    seg($("ra-models"), Object.keys(D.groups), model, function (k) { model = k; var b = Object.keys(D.groups[k]); budget = b.find(function (x) { return x.indexOf("15") === 0; }) || b[1]; drawRank(); });
    seg($("ra-budgets"), Object.keys(D.groups[model]), budget, function (k) { budget = k; drawRank(); });
    press($("ra-ref"), ref);
    var runs = D.groups[model][budget], vals = runs.map(function (r) { return r[ref]; });
    var lo = Math.floor(Math.min.apply(null, vals) - 1), hi = Math.ceil(Math.max.apply(null, ref === "conv" ? vals.concat([D.ref50[model]]) : vals) + 0.5);
    var sorted = runs.slice().sort(function (a, b) { return b[ref] - a[ref]; });
    var box = $("ra-rank"); box.style.height = runs.length * ROW + "px";
    box.style.setProperty("--ra-top", ref === "x50" ? "var(--ra-gray)" : "var(--ra-blue)");
    var keep = {}; box.querySelectorAll(".row").forEach(function (e) { if (runs.some(function (r) { return r.name === e.dataset.n; })) keep[e.dataset.n] = e; else e.remove(); });
    sorted.forEach(function (r, i) {
      var e = keep[r.name];
      if (!e) { e = document.createElement("div"); e.dataset.n = r.name;
        e.innerHTML = '<span class="no"></span><span class="nm"></span><span class="tr"><span class="fl"></span><span class="vl"></span></span>'; box.appendChild(e); }
      e.className = "row k-" + r.kind + (i === 0 ? " top" : "");
      e.querySelector(".no").textContent = i + 1; e.querySelector(".nm").textContent = r.name;
      var w = Math.max(2, 100 * (r[ref] - lo) / (hi - lo));
      e.style.top = i * ROW + "px"; e.querySelector(".fl").style.width = w + "%";
      var v = e.querySelector(".vl"); v.style.left = w + "%"; v.textContent = r[ref].toFixed(2) + " dB";
    });
    var line = box.querySelector(".r50");
    if (ref === "conv") {
      if (!line) { line = document.createElement("div"); line.className = "r50"; line.innerHTML = "<span>50-step output</span>"; box.appendChild(line); }
      var tr = box.querySelector(".tr"); line.style.left = tr.offsetLeft + tr.offsetWidth * (D.ref50[model] - lo) / (hi - lo) + "px";
    } else if (line) line.remove();
    $("ra-axis").innerHTML = "<span>" + lo + " dB</span><span>PSNR to " + (ref === "x50" ? X50 : "x*") + "</span><span>" + hi + " dB</span>";
    var picks = [runs.find(function (r) { return r.kind === "oss_unipc"; }), runs.find(function (r) { return r.kind === "cache"; })].filter(Boolean);
    $("ra-rc").innerHTML = picks.map(function (r) {
      var c = r.kind === "oss_unipc" ? "var(--ra-green)" : "var(--ra-red)";
      return '<span><b style="color:' + c + '">' + r.name + "</b>: #" + rankOf(runs, "x50", r.name) + " against " + X50 + ' → <b style="color:var(--ra-blue)">#' + rankOf(runs, "conv", r.name) + "</b> against x*</span>";
    }).join("");
    var top = sorted[0], other = ref === "x50" ? "conv" : "x50";
    var r = $("ra-read"); r.className = "ra-read" + (ref === "conv" ? " conv" : "");
    r.innerHTML = ref === "x50"
      ? "<b>Standard protocol:</b> closest to the 50-step output is <b>" + top.name + "</b>, #" + rankOf(runs, "conv", top.name) + " against the converged reference."
      : "<b>RefAudit:</b> closest to the converged reference is <b>" + top.name + "</b>, #" + rankOf(runs, "x50", top.name) + " against the 50-step output." +
        (model === "Wan2.1-1.3B" ? " Wan2.1's 50-step sampler is UniPC, so the 50-step score already favors the UniPC family; on the default grid the caches keep a small edge at 21–25 steps." : "");
  }
  $("ra-ref").querySelectorAll("button").forEach(function (b) { b.onclick = function () { ref = b.dataset.v; drawRank(); }; });
  window.addEventListener("resize", drawRank);

  /* Eq. 1 split */
  var setting = Object.keys(D.split)[0];
  function drawSplit() {
    seg($("ra-settings"), Object.keys(D.split), setting, function (k) { setting = k; drawSplit(); });
    var S = D.split[setting], C = S.const;
    var max = 1.12 * Math.max.apply(null, S.rows.map(function (r) { return Math.max(r.conv, r.c, r.std); }));
    function bar(cls, v, lab) { var p = 100 * v / max;
      return '<div class="ra-mini"><span class="ml">' + lab + '</span><span class="mt"><span class="mf ' + cls + '" style="width:' + p + '%"></span><span class="mv" style="left:' + p + '%">' + v.toFixed(1) + "</span></span></div>"; }
    $("ra-split").innerHTML = S.rows.map(function (r) {
      return '<div><b>' + r.name + '</b><div class="eq"><span class="t-std">' + r.std.toFixed(1) + '</span> = <span class="t-conv">' + r.conv.toFixed(1) + "</span> + " + C.toFixed(1) + ' − <span class="t-c">' + r.c.toFixed(1) + "</span></div></div><div>" +
        bar("m-conv", r.conv, "to x*") + bar("m-c", r.c, "shared c") + bar("m-std", r.std, "to " + X50) + "</div>";
    }).join("");
    function by(k) { return S.rows.slice().sort(function (a, b) { return a[k] - b[k]; })[0]; }
    var b50 = by("std"), bc = by("conv"), big = S.rows.slice().sort(function (a, b) { return b.c - a.c; })[0];
    $("ra-read2").innerHTML = "Closest to " + X50 + ": <b>" + b50.name + "</b>; closest to x*: <b>" + bc.name + "</b>. The largest shared error is " + big.name +
      "'s (c = " + big.c.toFixed(1) + ", cosine " + big.rho.toFixed(2) + " with the 50-step output's error), and the constant " + C.toFixed(1) + " is the same for every run (" + S.rows[0].n + " prompts).";
  }

  /* one prompt */
  var ref3 = "x50", view3 = "img";
  function drawOne() {
    press($("ra-ref3"), ref3); press($("ra-view3"), view3);
    var steps = D.images.filter(function (x) { return x.x50 !== null && x.conv !== null; });
    var best = steps.slice().sort(function (a, b) { return b[ref3] - a[ref3]; })[0];
    var col = ref3 === "x50" ? "var(--ra-gray)" : "var(--ra-blue)";
    $("ra-cards").innerHTML = D.images.map(function (x) {
      var isRef = x[ref3] === null, src = view3 === "err" && x.err ? x.err : x.img;
      var p = isRef ? '<span style="opacity:.65;font-weight:600">reference</span>' : x[ref3].toFixed(1) + " dB";
      return '<figure class="ra-card' + (x === best ? " best" : "") + '" style="--ra-bc:' + col + '"><img src="' + IMG + src + '.webp" alt="' + x.name + '" loading="lazy">' +
        (x === best ? '<span class="bd">closest</span>' : "") + '<div class="t">' + x.name + '</div><div class="p">' + p + "</div></figure>";
    }).join("");
    var r = $("ra-read3"); r.className = "ra-read" + (ref3 === "conv" ? " conv" : "");
    var mag = D.images.find(function (x) { return x.img === "magcache"; });
    r.innerHTML = ref3 === "x50"
      ? "<b>" + best.name + "</b> is closest to the 50-step output (" + best.x50.toFixed(1) + " dB): it shares the 50-step sampler's Euler error."
      : "<b>" + best.name + "</b> is closest to the converged reference (" + best.conv.toFixed(1) + " dB), and MagCache drops to " + mag.conv.toFixed(1) + " dB: same computed steps, a more accurate solver.";
  }
  $("ra-ref3").querySelectorAll("button").forEach(function (b) { b.onclick = function () { ref3 = b.dataset.v; drawOne(); }; });
  $("ra-view3").querySelectorAll("button").forEach(function (b) { b.onclick = function () { view3 = b.dataset.v; drawOne(); }; });

  drawRank(); drawSplit(); drawOne();
})();
</script>
{% endraw %}
