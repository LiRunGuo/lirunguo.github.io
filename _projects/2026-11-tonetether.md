---
title: "ToneTether: Lowering Drift Without Freezing in Long Autoregressive Video Generation"
collection: projects
permalink: /projects/2026-11-tonetether/
date: 2026-11-16
status: "Under review"
venue: "Under review"
excerpt: "Drift scores reward long videos that stop changing. ToneTether corrects only the predicted tone of the drift in the KV cache, lowering colour drift by a quarter to a third without freezing the video."
teaser: "/images/projects/2026-11-tonetether/fig1-teaser.webp"
tags: ["video generation", "autoregressive diffusion", "KV cache", "training-free"]
bibtexurl: "/files/bibtex/li2026tonetether.bib"
publication: "/publication/2026-11-tonetether"
citation: 'Runguo Li. (2026). &quot;ToneTether: Lowering Drift Without Freezing in Long Autoregressive Video Generation.&quot; Manuscript under review.'
---

<style>
/* ToneTether project page: styles scoped to .tt / .tt-* so the rest of the site is untouched; colors follow the theme */
.tt{--tt-acc:#c85400;--tt-acc-soft:rgba(213,94,0,.09);--tt-blue:#1f5fa8;--tt-blue-soft:rgba(31,95,168,.08);--tt-gray:#6f767e;--tt-gray-soft:rgba(127,127,127,.09);--tt-line:rgba(127,127,127,.28);--tt-card:rgba(127,127,127,.05)}
html[data-theme="dark"] .tt{--tt-acc:#ff9a52;--tt-acc-soft:rgba(255,154,82,.14);--tt-blue:#78b0f0;--tt-blue-soft:rgba(120,176,240,.14);--tt-gray:#c3c9cf;--tt-gray-soft:rgba(255,255,255,.07);--tt-line:rgba(255,255,255,.2);--tt-card:rgba(255,255,255,.04)}
.tt-kicker{display:flex;flex-wrap:wrap;gap:.4em;margin:.2em 0 .9em}
.tt-kicker span{font-size:.72em;font-weight:700;letter-spacing:.06em;text-transform:uppercase;padding:.25em .75em;border-radius:999px;background:var(--tt-acc-soft);color:var(--tt-acc)}
.tt-tldr{font-size:1.16em;line-height:1.6;margin:0 0 1em;font-weight:500}
.tt-tldr b{color:var(--tt-acc)}
.tt-btns{display:flex;flex-wrap:wrap;gap:.5em;margin:0 0 1.4em}
.tt-btn{display:inline-flex;align-items:center;gap:.45em;padding:.5em 1.05em;border-radius:999px;font-size:.85em;font-weight:600;text-decoration:none!important;background:#2b2f33;color:#fff!important;border:1px solid #2b2f33;transition:transform .15s,box-shadow .15s}
.tt-btn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.18)}
.tt-btn.ghost{background:transparent;color:var(--global-text-color)!important;border-color:var(--tt-line)}
.tt-btn.muted{background:transparent;color:var(--tt-gray)!important;border:1px dashed var(--tt-line);cursor:default}
.tt-btn.muted:hover{transform:none;box-shadow:none}
html[data-theme="dark"] .tt-btn:not(.ghost):not(.muted){background:#f2f4f6;color:#1c1f22!important;border-color:#f2f4f6}
.tt-fig{display:block;margin:0 0 1.6em;padding:.8em;border:1px solid var(--tt-line);border-radius:14px;background:var(--tt-card)}
.tt-fig img{display:block;width:100%;height:auto;border-radius:8px;background:#fff;margin:0}
.tt-fig figcaption{display:block;width:100%;font-family:inherit;font-size:.86em;line-height:1.55;margin:.75em .2em .1em;color:var(--global-text-color)}
.tt-tag{display:inline-block;font-size:.78em;font-weight:800;letter-spacing:.04em;text-transform:uppercase;padding:.1em .6em;border-radius:999px;background:var(--tt-acc);color:#fff;margin-right:.5em;vertical-align:.1em}
html[data-theme="dark"] .tt-tag{color:#2a1608}
.tt-fig figcaption b.tt-title{display:block;font-size:1.04em;margin:.35em 0 .2em}
.tt-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.7em;margin:0 0 1.6em}
.tt-stat{padding:.95em .9em .85em;border-radius:14px;border:1px solid var(--tt-line);background:var(--tt-card);position:relative;overflow:hidden}
.tt-stat:before{content:"";position:absolute;left:0;top:0;right:0;height:4px;background:var(--tt-c,var(--tt-acc))}
.tt-stat .v{font-size:1.32em;font-weight:800;line-height:1.15;color:var(--tt-c,var(--tt-acc));font-variant-numeric:tabular-nums}
.tt-stat .d{font-size:.78em;line-height:1.45;margin-top:.45em;color:var(--global-text-color);opacity:.88}
.tt-cmp{display:grid;grid-template-columns:auto 1fr auto 1fr 1.05fr;gap:.55em .6em;align-items:stretch;margin:0 0 1.6em;font-size:.88em}
.tt-cmp .h{font-size:.76em;font-weight:800;letter-spacing:.05em;text-transform:uppercase;padding:0 .2em;align-self:end}
.tt-cmp .h.s{color:var(--tt-gray)}.tt-cmp .h.c,.tt-cmp .h.o{color:var(--tt-acc)}
.tt-cmp .k{font-weight:700;align-self:center;padding-right:.3em}
.tt-cmp .s,.tt-cmp .c,.tt-cmp .o{padding:.55em .75em;border-radius:10px;line-height:1.4}
.tt-cmp div.s{background:var(--tt-gray-soft);border:1px solid var(--tt-line);color:var(--tt-gray)}
.tt-cmp div.c{background:var(--tt-acc-soft);border:1px solid var(--tt-acc)}
.tt-cmp div.o{border-left:3px solid var(--tt-acc);border-radius:4px}
.tt-cmp .arr{align-self:center;color:var(--tt-gray);font-size:1.1em}
.tt-box{padding:1em 1.2em;border-radius:14px;background:var(--tt-acc-soft);border:1px solid var(--tt-line);margin:0 0 1.6em;line-height:1.65}
.tt-box p{margin:0}
.tt-table{margin:0 0 1.6em;padding:.8em;border:1px solid var(--tt-line);border-radius:14px;background:var(--tt-card);overflow-x:auto}
.tt-table table{font-size:.8em;margin:.6em 0;min-width:620px}
.tt-table p{font-size:.86em;margin:.2em}
.tt-tabs{display:flex;flex-wrap:wrap;gap:.4em;margin:0 0 .8em}
.tt-tabs button{font:inherit;font-size:.8em;font-weight:600;padding:.35em .9em;border-radius:999px;border:1px solid var(--tt-line);background:transparent;color:var(--global-text-color);cursor:pointer}
.tt-tabs button[aria-pressed="true"]{background:var(--tt-acc);border-color:var(--tt-acc);color:#fff}
html[data-theme="dark"] .tt-tabs button[aria-pressed="true"]{color:#2a1608}
.tt-vgrid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.6em}
.tt-cell video{display:block;width:100%;aspect-ratio:416/240;border-radius:8px;background:#000}
.tt-vlab{display:flex;justify-content:space-between;gap:.5em;flex-wrap:wrap;font-size:.8em;margin:.3em .1em 0}
.tt-vlab b{font-weight:700}.tt-vlab span{color:var(--tt-gray);font-variant-numeric:tabular-nums}
.tt-cell.ours video{box-shadow:0 0 0 3px var(--tt-acc)}.tt-cell.ours .tt-vlab b{color:var(--tt-acc)}
.tt-prompt{font-size:.8em;color:var(--tt-gray);font-style:italic;margin:.7em .2em 0}
.tt-slider{position:relative;width:100%;aspect-ratio:416/240;border-radius:8px;overflow:hidden;user-select:none;touch-action:pan-y;cursor:ew-resize;background:#000}
.tt-slider video{position:absolute;inset:0;width:100%;height:100%;margin:0}
.tt-slider .after{clip-path:inset(0 0 0 var(--pos,50%))}
.tt-slider .line{position:absolute;top:0;bottom:0;left:var(--pos,50%);width:3px;margin-left:-1.5px;background:#fff;box-shadow:0 0 6px rgba(0,0,0,.5)}
.tt-slider .knob{position:absolute;top:50%;left:var(--pos,50%);width:36px;height:36px;margin:-18px 0 0 -18px;border-radius:50%;background:#fff;color:#222;display:grid;place-items:center;font-weight:800;font-size:14px;box-shadow:0 2px 8px rgba(0,0,0,.35)}
.tt-slider:focus-visible .knob{box-shadow:0 0 0 3px var(--tt-acc)}
.tt-slider .lbl{position:absolute;top:8px;padding:.2em .6em;border-radius:6px;font-size:.72em;font-weight:700;color:#fff;background:rgba(0,0,0,.55)}
.tt-slider .lbl.l{left:8px}.tt-slider .lbl.r{right:8px;background:var(--tt-acc)}
@media (max-width:900px){.tt-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.tt-vgrid{grid-template-columns:1fr}.tt-cmp{grid-template-columns:1fr}.tt-cmp .arr,.tt-cmp .h,.tt-cmp span:empty{display:none}.tt-cmp .k{margin-top:.6em}.tt-tldr{font-size:1.04em}}
</style>

<div class="tt">
<div class="tt-kicker"><span>Video generation</span><span>Autoregressive diffusion</span><span>KV cache</span><span>Training-free</span></div>
<p class="tt-tldr">Long-video corrections are judged by drift scores, and a drift score rewards a video that stops changing. <b>ToneTether</b> corrects only the predicted global tone of the drift in the KV cache, and removes 23–34% of TetherCache's colour drift without a measurable loss of motion or new content.</p>
<div class="tt-btns"><a class="tt-btn" href="#videos">▶&nbsp; Videos</a><a class="tt-btn ghost" href="#key-results">Key results</a><a class="tt-btn ghost" href="#what-tonetether-changes">The idea</a><a class="tt-btn ghost" href="/files/bibtex/li2026tonetether.bib">BibTeX</a><span class="tt-btn muted">Paper &amp; code: coming soon</span></div>
<figure class="tt-fig"><img src="/images/projects/2026-11-tonetether/fig1-teaser.webp" alt="Figure 1: correcting drift without freezing" loading="lazy"><figcaption><span class="tt-tag">Figure 1</span><b class="tt-title">Correcting drift without freezing.</b>Left: the lowest-drift baseline repeats nearly the same frame, while TetherCache's sky darkens after about two minutes. Right: tethers on all directions and low-frequency injection lower drift only together with motion or new content; ToneTether lowers it with neither loss.</figcaption></figure>
</div>

## At a glance

<div class="tt"><div class="tt-stats">
<div class="tt-stat"><div class="v">23–34%</div><div class="d">of TetherCache's colour drift removed on Self-Forcing and Causal Forcing at 60 s and 240 s, significant in all eight cases after Holm correction</div></div>
<div class="tt-stat" style="--tt-c:var(--tt-blue)"><div class="v">≤ 0.7</div><div class="d">points between ToneTether's and TetherCache's dynamic degree; the tether on all directions loses 4–7</div></div>
<div class="tt-stat" style="--tt-c:var(--tt-gray)"><div class="v">31–40%</div><div class="d">less new content than TetherCache for FreqForcing, the method with the lowest drift</div></div>
<div class="tt-stat"><div class="v">87%</div><div class="d">of a prompted tone change shown in long videos with a shadow reference, instead of 25–30%</div></div>
</div></div>

## What ToneTether changes

<div class="tt"><div class="tt-cmp">
<span></span><span class="h s">Usual practice</span><span></span><span class="h c">ToneTether</span><span class="h o">What it shows</span>
<span class="k">Score</span><div class="s">drift: the ending against the opening</div><span class="arr">→</span><div class="c">drift and progression: motion, novelty, prompt following, scripted tone changes</div><div class="o">the corrections with the lowest drift are the ones that freeze</div>
<span class="k">Correction</span><div class="s">every direction of the KV statistics, one strength</div><span class="arr">→</span><div class="c">only the five directions per value head that predict the global tone</div><div class="o">drift shares its directions with legitimate change; a uniform strength is optimal only for proportional second moments</div>
<span class="k">Reference</span><div class="s">the opening of the video</div><span class="arr">→</span><div class="c">a 5 s empty-cache shadow rollout per prompt</div><div class="o">a prompted change of tone is no longer pulled back like drift</div>
</div></div>

## Summary

<div class="tt"><div class="tt-box"><p>Long autoregressive video generators drift in colour, texture and structure, and training-free corrections that anchor the KV cache are judged by drift scores that compare a video's ending with its opening. Such scores cannot tell a corrected video from one that has stopped changing, so the corrections with the lowest drift are the ones that freeze. In the KV cache, drift shares its directions with legitimate change. ToneTether therefore removes only the predicted global tone of a Kalman drift estimate, the smallest such edit in units of legitimate variation. Added to TetherCache, it lowers colour drift on two backbones and two horizons while keeping motion and new content, adds 4 s to a 240 s video, and needs one calibration per backbone and no training.</p></div></div>

## Key results

<div class="tt">
<figure class="tt-fig"><img src="/images/projects/2026-11-tonetether/fig2-kv-drift-anatomy.png" alt="Figure 2: drift in KV statistics" loading="lazy"><figcaption><span class="tt-tag">Figure 2</span><b class="tt-title">Drift shares its directions with legitimate change.</b>Drift in the cached statistics is systematic across prompts and exceeds in-shot change by orders of magnitude in its leading directions, yet those directions also carry a large share of legitimate change. Removing a "drift subspace" therefore removes content as well: what matters is which directions to correct.</figcaption></figure>
<figure class="tt-fig"><img src="/images/projects/2026-11-tonetether/fig3-overview.png" alt="Figure 3: ToneTether overview" loading="lazy"><figcaption><span class="tt-tag">Figure 3</span><b class="tt-title">ToneTether removes only the tone of the drift.</b>It edits frames as they enter TetherCache's memory and removes the tone that a Kalman estimate of the drift would produce, the smallest such edit in units of legitimate variation. After a prompt switch, a short empty-cache shadow rollout moves the reference to the new prompt's tone.</figcaption></figure>
</div>

<div class="tt"><div class="tt-table" markdown="1">

<span class="tt-tag">Table 1</span> **ToneTether lowers drift on both backbones and keeps TetherCache's motion and novelty.** 240 s rollouts on 128 prompts (H100). ΔE: CIELAB colour drift; Drop: image-quality drop between the first and last fifth; Dyn: dynamic degree; Nov.: novelty.

| Method | SF ΔE ↓ | SF Drop ↓ | SF Dyn ↑ | SF Nov. ↑ | CF ΔE ↓ | CF Drop ↓ | CF Dyn ↑ | CF Nov. ↑ |
|---|---|---|---|---|---|---|---|---|
| TetherCache | 4.67 | 0.54 | 73.0 | 9.7 | 7.46 | 0.81 | 86.0 | 15.0 |
| FreqForcing | 1.04 | 0.17 | 64.4 | 5.8 | 2.46 | 0.39 | 86.0 | 10.3 |
| Kalman tether, all directions | 3.60 | 1.15 | 67.2 | 8.8 | 4.81 | 1.44 | 79.5 | 14.0 |
| TetherCache + ToneTether | 3.10 | 1.39 | 73.7 | 9.7 | 5.16 | 1.47 | 86.5 | 14.9 |

The method with the lowest drift loses 31–40% of TetherCache's new content and the tether on all directions loses motion; ToneTether lowers drift on Self-Forcing (SF) and Causal Forcing (CF) with TetherCache's motion and novelty.

</div></div>

<div class="tt">
<figure class="tt-fig"><img src="/images/projects/2026-11-tonetether/figs4-color-cast.webp" alt="Figure S4: a colour cast removed while the scene keeps changing" loading="lazy"><figcaption><span class="tt-tag">Figure S4</span><b class="tt-title">A colour cast removed while the scene keeps changing.</b>With TetherCache the water turns brown and the fish takes on a red cast after about 100 s; with ToneTether the water keeps its colour while the scene keeps changing (ΔE 15.5 → 5.1).</figcaption></figure>
<figure class="tt-fig"><img src="/images/projects/2026-11-tonetether/figs3-shadow-target.webp" alt="Figure S3: prompted tone changes with a shadow reference" loading="lazy"><figcaption><span class="tt-tag">Figure S3</span><b class="tt-title">A shadow reference lets prompted change through.</b>A reference built from the opening pulls a prompted change of tone back like drift. Tethered to the shadow clips' tone, the model renders the storm and the sunset, and long videos show 87% of the prompted change instead of 25–30%.</figcaption></figure>
</div>

## Videos

<div class="tt">
<figure class="tt-fig" id="tt-videos">
<div class="tt-tabs" id="tt-tabs"><button aria-pressed="true" data-i="0">Mammoths</button><button aria-pressed="false" data-i="1">Big Sur</button><button aria-pressed="false" data-i="2">Paper airplanes</button><button aria-pressed="false" data-i="3">Tokyo street</button></div>
<div class="tt-vgrid">
<div class="tt-cell"><video muted loop playsinline preload="metadata" data-k="grab"></video><div class="tt-vlab"><b>GRAB</b><span data-m="grab"></span></div></div>
<div class="tt-cell"><video muted loop playsinline preload="metadata" data-k="tc06"></video><div class="tt-vlab"><b>TetherCache</b><span data-m="tc06"></span></div></div>
<div class="tt-cell"><video muted loop playsinline preload="metadata" data-k="tc_kalcal"></video><div class="tt-vlab"><b>Kalman tether, all directions</b><span data-m="tc_kalcal"></span></div></div>
<div class="tt-cell ours"><video muted loop playsinline preload="metadata" data-k="tc_kalcal_col_t1"></video><div class="tt-vlab"><b>TetherCache + ToneTether</b><span data-m="tc_kalcal_col_t1"></span></div></div>
</div>
<p class="tt-prompt" id="tt-prompt"></p>
<figcaption><span class="tt-tag">Videos</span><b class="tt-title">Same prompt, four caches.</b>240 s Self-Forcing rollouts played at 8× speed, with each video's colour drift ΔE between the first and last fifth and its dynamic degree. The four prompts are those with the largest TetherCache colour drift among the first 32 evaluation prompts (RTX 4090 runs). On Big Sur the tether on all directions removes the drift by stopping the waves; ToneTether keeps them moving.</figcaption>
</figure>
<figure class="tt-fig">
<div class="tt-slider" id="tt-slider" style="--pos:50%" tabindex="0" role="slider" aria-label="Drag to compare TetherCache with TetherCache + ToneTether" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">
<video muted loop playsinline preload="metadata" data-k="tc06"></video><video class="after" muted loop playsinline preload="metadata" data-k="tc_kalcal_col_t1"></video>
<span class="line"></span><span class="knob">⇔</span><span class="lbl l">TetherCache</span><span class="lbl r">+ ToneTether</span></div>
<figcaption><span class="tt-tag">Compare</span><b class="tt-title">TetherCache against TetherCache + ToneTether.</b>Drag the divider; the prompt follows the selection above.</figcaption>
</figure>
</div>

<div class="tt"><div class="tt-table" markdown="1">

<span class="tt-tag">Table 2</span> **Which directions, not how strongly.** Design ablations on 128 development prompts (Self-Forcing, 240 s), all added to TetherCache.

| Correction | ΔE ↓ | Dyn ↑ | Flow ↑ |
|---|---|---|---|
| TetherCache | 4.23 | 76.6 | 60.2 |
| Kalman tether, all directions, τ = 0.5 | 3.27 | 71.4 | 56.2 |
| Kalman tether, all directions, τ = 1 | 3.14 | 70.5 | 55.0 |
| Same correction, random 5-dim. subspace | 4.21 | 72.8 | 58.8 |
| Same correction, top-5 drift directions | 4.04 | 76.3 | 62.2 |
| ToneTether, τ = 1 | 3.08 | 76.2 | 59.6 |

Tethers on all directions trade motion for colour drift at every strength; ToneTether removes as much colour drift with TetherCache's motion and flow, which neither a random nor the top-drift subspace achieves.

</div></div>

<script>
(function () {
  var D = "/files/projects/2026-11-tonetether/";
  var P = [
    {id: "0001", p: "Several giant wooly mammoths approach treading through a snowy meadow ...", m: {grab: [13.89, 100], tc06: [15.97, 96], tc_kalcal: [5.48, 87], tc_kalcal_col_t1: [5.21, 94]}},
    {id: "0003", p: "Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach ...", m: {grab: [5.00, 60], tc06: [11.35, 36], tc_kalcal: [1.71, 2], tc_kalcal_col_t1: [5.70, 49]}},
    {id: "0021", p: "A flock of paper airplanes flutters through a dense jungle, weaving around trees as if they were migrating birds.", m: {grab: [24.35, 100], tc06: [9.66, 100], tc_kalcal: [3.41, 100], tc_kalcal_col_t1: [2.72, 100]}},
    {id: "0000", p: "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage ...", m: {grab: [10.34, 100], tc06: [8.22, 100], tc_kalcal: [8.13, 100], tc_kalcal_col_t1: [5.98, 100]}}
  ];
  var vids = [].slice.call(document.querySelectorAll("#tt-videos video, #tt-slider video"));
  var visible = false;
  function playAll() { if (visible) vids.forEach(function (v) { var r = v.play(); if (r) r.catch(function () {}); }); }
  function load(i) {
    var c = P[i];
    vids.forEach(function (v) { var k = v.getAttribute("data-k"); v.poster = D + "posters/" + c.id + "_" + k + ".jpg"; v.src = D + "videos/" + c.id + "_" + k + ".mp4"; });
    [].forEach.call(document.querySelectorAll("#tt-videos [data-m]"), function (el) { var x = c.m[el.getAttribute("data-m")]; el.textContent = "\u0394E " + x[0].toFixed(2) + " \u00b7 dynamic " + x[1] + "%"; });
    document.getElementById("tt-prompt").textContent = "\u201c" + c.p + "\u201d";
    [].forEach.call(document.querySelectorAll("#tt-tabs button"), function (b) { b.setAttribute("aria-pressed", String(+b.getAttribute("data-i") === i)); });
    playAll();
  }
  [].forEach.call(document.querySelectorAll("#tt-tabs button"), function (b) { b.addEventListener("click", function () { load(+b.getAttribute("data-i")); }); });
  // keep the videos in step, and play them only while they are on screen
  setInterval(function () { var t = vids[0].currentTime; vids.forEach(function (v) { if (!v.paused && Math.abs(v.currentTime - t) > 0.2) v.currentTime = t; }); }, 1000);
  var shown = {};
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { shown[e.target.id] = e.isIntersecting; });
    visible = shown["tt-videos"] || shown["tt-slider"];
    if (visible) playAll(); else vids.forEach(function (v) { v.pause(); });
  });
  io.observe(document.getElementById("tt-videos")); io.observe(document.getElementById("tt-slider"));
  load(0);
  // drag-to-compare slider (pointer and keyboard)
  var s = document.getElementById("tt-slider"), drag = false;
  function set(p) { p = Math.max(0, Math.min(100, p)); s.style.setProperty("--pos", p + "%"); s.setAttribute("aria-valuenow", Math.round(p)); }
  function move(x) { var r = s.getBoundingClientRect(); set((x - r.left) / r.width * 100); }
  s.addEventListener("pointerdown", function (e) { drag = true; s.setPointerCapture(e.pointerId); move(e.clientX); });
  s.addEventListener("pointermove", function (e) { if (drag) move(e.clientX); });
  s.addEventListener("pointerup", function () { drag = false; });
  s.addEventListener("keydown", function (e) { var p = parseFloat(s.getAttribute("aria-valuenow")); if (e.key === "ArrowLeft") { set(p - 5); e.preventDefault(); } else if (e.key === "ArrowRight") { set(p + 5); e.preventDefault(); } });
})();
</script>

## Links and citation

- Publication entry: [ToneTether](/publication/2026-11-tonetether)
- BibTeX: [li2026tonetether.bib](/files/bibtex/li2026tonetether.bib)
- Paper and code: coming soon

Runguo Li. (2026). "ToneTether: Lowering Drift Without Freezing in Long Autoregressive Video Generation." Manuscript under review.
