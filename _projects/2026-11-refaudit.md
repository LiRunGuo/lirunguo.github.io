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
.ra-demo{border:1px solid var(--ra-line);border-radius:16px;overflow:hidden;margin:0 0 1em;background:var(--ra-card)}
.ra-demo .bar{display:flex;align-items:center;gap:.45em;padding:.55em .9em;border-bottom:1px solid var(--ra-line);font-size:.8em}
.ra-demo .dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.ra-demo .bar a{margin-left:auto;font-weight:600}
.ra-demo iframe{display:block;width:100%;border:0;min-height:420px}
@media (max-width:900px){.ra-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.ra-cmp{grid-template-columns:1fr}.ra-cmp .arr,.ra-cmp .h,.ra-cmp span:empty{display:none}.ra-cmp .k{margin-top:.6em}.ra-tldr{font-size:1.04em}}
</style>

<div class="ra">
<div class="ra-kicker"><span>Diffusion transformers</span><span>Training-free acceleration</span><span>Evaluation protocol</span></div>
<p class="ra-tldr">Diffusion step caches are scored by their fidelity to the 50-step output, and that score rewards sharing the 50-step sampler's error. <b>RefAudit</b> scores against a converged reference instead, and an optimized static schedule with UniPC then matches or beats every cache we ran on FLUX.1-dev and SD3.5.</p>
<div class="ra-btns"><a class="ra-btn" href="#interactive-demo">▶&nbsp; Interactive demo</a><a class="ra-btn ghost" href="#key-results">Key results</a><a class="ra-btn ghost" href="#what-refaudit-changes">The protocol</a><span class="ra-btn muted">Paper &amp; code: coming soon</span></div>
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
<p class="ra-why"><b>Why (iii):</b> under Euler, a step cache that reuses the last velocity <i>is</i> Euler on its computed steps, and residual reuse nearly so. Step caching is step scheduling, so its fair competitor is the best schedule with the same number of steps.</p></div>

## Summary

<div class="ra"><div class="ra-box"><p>Training-free step caching (TeaCache, MagCache and their successors) speeds up diffusion transformers by reusing the transformer's last result at some sampling steps, and it is judged by fidelity to the default 50-step output and compared with sampling at fewer uniform steps. We show that this protocol favors caching: the 50-step output is itself a coarse ODE solve, so fidelity to it rewards samplers that share its error, and under Euler a reuse cache is exactly or nearly Euler on its computed steps, so its fair baseline is an optimized timestep schedule. <b>RefAudit</b> splits the standard score into accuracy and shared error, checks the reference across solver families, and compares methods at equal NFE under the same solver; under it, optimized static schedules with UniPC match or beat every cache we ran on FLUX.1-dev and SD3.5, with no run-time decisions or extra memory.</p></div></div>

## Key results

<div class="ra">
<figure class="ra-fig narrow"><img src="/images/projects/2026-11-refaudit/fig2-reference-family.png" alt="Figure 2: the reference's solver family sets the sign" loading="lazy"><figcaption><span class="ra-tag">Figure 2</span><b class="ra-title">The reference's solver family sets the sign.</b>Against 50-step references, the gap between the same two samplers follows the reference's solver family on every model, by 4–10 dB either way. From 200 steps on, references of both families give the same sign: the cross-family check RefAudit asks of a reference.</figcaption></figure>
<figure class="ra-fig"><img src="/images/projects/2026-11-refaudit/fig3-ranking-and-pareto.png" alt="Figure 3: ranking under each reference and PSNR against computed steps" loading="lazy"><figcaption><span class="ra-tag">Figure 3</span><b class="ra-title">RefAudit reverses the ranking; on the image models, OSS + UniPC is above every cache shown and the tuned grids, at every budget.</b>Ranked against the 50-step output, OSS + UniPC is last of four runs at FLUX-15 and SD3.5-14; ranked against the converged reference it is first and MagCache is last.</figcaption></figure>
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
</figure></div>

## Interactive demo

Switch the reference and watch the ranking change, see how the shared error builds the 50-step score (Eq. 1), and
compare one prompt's 15-step runs under both references. All numbers are the paper's logged results.

<div class="ra"><div class="ra-demo"><div class="bar"><span class="dot" style="background:#ff5f57"></span><span class="dot" style="background:#febc2e"></span><span class="dot" style="background:#28c840"></span><span style="margin-left:.4em;opacity:.75">RefAudit · interactive demo</span><a href="/files/projects/2026-11-refaudit/demo/index.html" target="_blank" rel="noopener">Open full screen ↗</a></div><iframe id="ra-demo-frame" src="/files/projects/2026-11-refaudit/demo/index.html" height="720" loading="lazy" title="RefAudit interactive demo"></iframe></div></div>

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
  // the demo reports its height, so the frame fits it without an inner scrollbar
  var f = document.getElementById("ra-demo-frame");
  window.addEventListener("message", function (e) {
    if (f && e.source === f.contentWindow && e.data && typeof e.data.raDemoHeight === "number") f.style.height = (e.data.raDemoHeight + 4) + "px";
  });
})();
</script>
