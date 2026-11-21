---
title: "Certified Sparse Attention via Perturbed Retrieval"
collection: projects
permalink: /projects/2026-10-certified-sparse-attention/
date: 2026-10-18
status: "arXiv preprint"
venue: "arXiv"
excerpt: "Sparse attention with a per-head error certificate: perturbing key scores turns threshold retrieval into weighted sampling, so each head stops reading keys once its bound meets the target error."
teaser: "/images/projects/2026-10-certified-sparse-attention/teaser.png"
tags: ["sparse attention", "long-context decoding", "KV cache", "certified estimation"]
paperurl: "https://arxiv.org/abs/2610.24991"
pdfurl: "/files/certified-sparse-attention.pdf"
bibtexurl: "/files/bibtex/li2026certified.bib"
publication: "/publication/2026-10-certified-sparse-attention"
citation: 'Runguo Li. (2026). &quot;Certified Sparse Attention via Perturbed Retrieval.&quot; arXiv:2610.24991.'
---

<style>
  /* Page-scoped styles (prefix csa-). Dark mode follows the site's theme toggle. */
  :root { --csa-accent:#2a78d6; --csa-accent-ink:#1d5fae; --csa-soft:#eef5fd; --csa-card:#ffffff; --csa-surface:#f6f8fb;
          --csa-line:#e2e7ee; --csa-muted:#5b6674; --csa-shadow:0 1px 2px rgba(16,24,40,.05), 0 8px 24px rgba(16,24,40,.06); }
  html[data-theme="dark"] { --csa-accent:#7fb3f2; --csa-accent-ink:#a9cbf7; --csa-soft:rgba(127,179,242,.13); --csa-card:#505050;
          --csa-surface:#4e4e4e; --csa-line:#626262; --csa-muted:#d0d0d0; --csa-shadow:0 1px 2px rgba(0,0,0,.25), 0 8px 24px rgba(0,0,0,.2); }
  .csa-hero { margin:0 0 1.4em; }
  .page__content p.csa-kicker { font-size:.72em; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--csa-accent-ink); margin:0 0 .5em; }
  .page__content p.csa-tldr { font-size:1.08em; line-height:1.5; margin:0 0 .6em; }
  .page__content p.csa-byline { font-size:.85em; color:var(--csa-muted); margin:0 0 1em; }
  .csa-links { display:flex; flex-wrap:wrap; gap:.5em; margin:0 0 1em; }
  .page__content a.csa-btn, a.csa-btn { display:inline-flex; align-items:center; gap:.45em; padding:.45em 1.05em; border-radius:999px;
          font-size:.8em; font-weight:600; text-decoration:none; color:#fff; background:#2b2f36; border:1px solid #2b2f36;
          transition:transform .12s ease, box-shadow .12s ease; }
  html[data-theme="dark"] a.csa-btn { background:#eceef1; color:#22252a; border-color:#eceef1; }
  a.csa-btn:hover { transform:translateY(-1px); box-shadow:var(--csa-shadow); color:#fff; }
  html[data-theme="dark"] a.csa-btn:hover { color:#22252a; }
  .page__content a.csa-btn--accent, a.csa-btn--accent { background:var(--csa-accent); border-color:var(--csa-accent); color:#fff; }
  html[data-theme="dark"] a.csa-btn--accent { background:var(--csa-accent); border-color:var(--csa-accent); color:#16202c; }
  .csa-stats { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:.75em; margin:0 0 1.5em; }
  .csa-stat { background:var(--csa-card); border:1px solid var(--csa-line); border-radius:12px; box-shadow:var(--csa-shadow); padding:.8em .9em; }
  .csa-stat__v { font-size:1.45em; font-weight:800; letter-spacing:-.01em; color:var(--csa-accent); line-height:1.15; white-space:nowrap; }
  .csa-stat__k { font-size:.72em; line-height:1.4; color:var(--csa-muted); margin-top:.35em; }
  .csa-steps { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.75em; margin:.4em 0 1.1em; counter-reset:csa; }
  .csa-step { background:var(--csa-surface); border:1px solid var(--csa-line); border-radius:12px; padding:.8em .9em; font-size:.8em; line-height:1.5; }
  .csa-step b { display:block; font-size:1.08em; margin-bottom:.25em; }
  .csa-step b::before { counter-increment:csa; content:counter(csa); display:inline-grid; place-items:center; width:1.55em; height:1.55em;
          border-radius:50%; background:var(--csa-accent); color:#fff; font-size:.85em; margin-right:.5em; vertical-align:.08em; }
  html[data-theme="dark"] .csa-step b::before { color:#16202c; }
  .csa-fig { margin:.6em 0 1.6em; background:var(--csa-card); border:1px solid var(--csa-line); border-radius:14px; box-shadow:var(--csa-shadow);
          padding:.9em .9em .8em; }
  .csa-fig img { display:block; width:100%; height:auto; background:#fff; border-radius:8px; }
  .page__content .csa-fig figcaption { font-family:inherit; font-size:.78em; line-height:1.55; color:var(--csa-muted); margin:.8em .2em 0; }
  .csa-fig figcaption strong { color:inherit; }
  .csa-table { margin:.4em 0 .4em; overflow-x:auto; }
  .csa-table table { font-size:.76em; margin:0; }
  .csa-table td:nth-child(2) strong { color:var(--csa-accent); }
  .page__content p.csa-takeaway { font-size:.82em; color:var(--csa-muted); border-left:3px solid var(--csa-accent); padding:.1em 0 .1em .8em; margin:.6em 0 1.6em; }
  .csa-callout { background:var(--csa-soft); border:1px solid var(--csa-line); border-radius:12px; padding:.9em 1.1em .5em; margin:.4em 0 1.6em; font-size:.85em; }
  .csa-callout ul { margin:.3em 0 .4em 1.1em; }
  .csa-callout li { margin:.25em 0; }
  .csa-demo { border:1px solid var(--csa-line); border-radius:14px; overflow:hidden; box-shadow:var(--csa-shadow); background:var(--csa-card); margin:.6em 0 .6em; }
  .csa-demo__bar { display:flex; justify-content:space-between; align-items:center; gap:1em; padding:.55em .9em; font-size:.75em; color:var(--csa-muted);
          background:var(--csa-surface); border-bottom:1px solid var(--csa-line); }
  .csa-demo__bar .csa-dot { display:inline-block; width:.55em; height:.55em; border-radius:50%; background:#1baf7a; margin-right:.45em; vertical-align:.05em; }
  .csa-demo iframe { display:block; width:100%; border:0; }
  .csa-bib { position:relative; margin:.6em 0 0; }
  .csa-bib pre { margin:0; border-radius:12px; font-size:.72em; max-width:100%; overflow-x:auto; padding:1em 1.1em;
          background:var(--csa-surface); border:1px solid var(--csa-line); }
  .csa-copy { position:absolute; top:.55em; right:.55em; font-size:.72em; padding:.25em .8em; border-radius:999px; border:1px solid var(--csa-line);
          background:var(--csa-card); color:inherit; cursor:pointer; }
  @media (max-width: 800px) { .csa-stats { grid-template-columns:repeat(2, minmax(0,1fr)); } .csa-steps { grid-template-columns:1fr; } }
</style>

<div class="csa-hero">
  <p class="csa-kicker">arXiv preprint &middot; October 2026 &middot; cs.LG</p>
  <p class="csa-tldr">Sparse attention that <strong>certifies its error</strong> on every head and every decoding step: perturbing key scores
  turns threshold retrieval into weighted sampling, and each head stops reading keys as soon as its error bound meets the target.</p>
  <p class="csa-byline"><strong>Runguo Li</strong> &middot; University of Illinois Urbana-Champaign</p>
  <div class="csa-links">
    <a class="csa-btn" href="https://arxiv.org/abs/2610.24991"><i class="ai ai-arxiv" aria-hidden="true"></i> arXiv</a>
    <a class="csa-btn" href="/files/certified-sparse-attention.pdf"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i> Paper (PDF)</a>
    <a class="csa-btn csa-btn--accent" href="#interactive-demo"><i class="fa-solid fa-play" aria-hidden="true"></i> Interactive demo</a>
    <a class="csa-btn" href="#links-and-citation"><i class="fa-solid fa-quote-right" aria-hidden="true"></i> BibTeX</a>
  </div>
</div>

<div class="csa-stats">
  <div class="csa-stat"><div class="csa-stat__v">0</div><div class="csa-stat__k">errors above &epsilon; across 120 configurations: 3 LLMs, 16K&ndash;128K tokens</div></div>
  <div class="csa-stat"><div class="csa-stat__v">4.6&ndash;7.9&times;</div><div class="csa-stat__k">lower 90th-percentile error than the best of 15 baseline families at equal keys read</div></div>
  <div class="csa-stat"><div class="csa-stat__v">&le; 1 pt</div><div class="csa-stat__k">end-to-end score change versus dense attention; PG-19 perplexity within 0.1%</div></div>
  <div class="csa-stat"><div class="csa-stat__v">1.1&ndash;1.4&times;</div><div class="csa-stat__k">faster whole decoding steps on CPU at 32K&ndash;128K tokens</div></div>
</div>

Sparse attention speeds up long-context decoding by reading few keys of the KV cache, but existing methods do not control
the error of a query, and on real KV caches their errors have heavy tails. Perturbing the score of every key with an
independent Exp(1) variable turns threshold retrieval into sampling keys in proportion to their attention weights:
descending a geometric grid of thresholds gives nested samples, an unbiased Horvitz–Thompson estimate at every level, and
confidence bounds valid at all levels at once, so each head stops at the first level certified to meet the target error ε.
On three LLMs and ten settings at 16K–128K tokens, no certificate is violated, and at equal keys read the 90th-percentile
error is 4.6–7.9× lower than that of the best of 15 baseline families.

## Key results

### How it works: perturb, lower the threshold, certify

<div class="csa-steps">
  <div class="csa-step"><b>Perturb</b>Every key's score gets an independent Exp(1) lift, drawn afresh at each decoding step. At threshold &tau;, key <i>i</i> is
    retrieved with probability min(1, w<sub>i</sub>/&tau;): sampling in proportion to its attention weight.</div>
  <div class="csa-step"><b>Estimate</b>Lowering &tau; along a geometric grid only adds keys. Heavy keys enter exactly and each sampled light key is weighted by
    &tau;, which gives an unbiased Horvitz&ndash;Thompson estimate at every level.</div>
  <div class="csa-step"><b>Certify and stop</b>Confidence bounds hold at all levels at once, so each head stops at the first level whose bound meets &epsilon;.
    At a suitable threshold, O(&epsilon;<sup>&minus;2</sup> log 1/&delta;) values suffice whatever the context length, which is optimal up to constants.</div>
</div>

<figure class="csa-fig">
  <img src="/images/projects/2026-10-certified-sparse-attention/fig1-method.png" alt="Figure 1: perturbed retrieval is PPS sampling, and one query of a GQA group." loading="lazy">
  <figcaption><strong>Figure 1.</strong> (a) A key is in the sample at threshold &tau; exactly when its perturbed score lies above the line of &tau;,
  which happens with probability min(1, w<sub>i</sub>/&tau;); lowering &tau; only adds keys. (b) A 4-bit scan shared by the query heads of a group bounds
  every score, so its candidates contain the sample exactly, and a certified output errs by more than &epsilon;M with probability at most &delta; (Theorem 4).</figcaption>
</figure>

### 4.6–7.9× lower 90th-percentile error at equal keys read

<div class="csa-table" markdown="1">

**Table 1 (excerpt). Error/M at matched keys read (ε = 0.1; baselines read at most 1.1× our keys).**

| Model · data | Ours: keys / p90 / max | Best baseline: keys, p90 / max | Ratio (p90) |
|---|---|---|---:|
| Llama · LongBench 16K | 2.70% / **0.011** / **0.021** | MagicPIG (ref.) 2.75%, 0.059 / 0.83 | 5.3× |
| Llama · RULER 32K | 1.35% / **0.011** / **0.025** | RetroInfer 1.24%, 0.058 / 0.49 | 5.0× |
| Llama · RULER 128K | 0.46% / **0.012** / **0.029** | RetroInfer 0.47%, 0.077 / 0.65 | 6.5× |
| Qwen3 · RULER 32K | 1.50% / **0.011** / **0.025** | RetroInfer 1.37%, 0.073 / 1.62 | 6.9× |
| Qwen2.5 · RULER 32K | 1.73% / **0.011** / **0.027** | RetroInfer 1.22%, 0.090 / 0.98 | 7.9× |

</div>

<p class="csa-takeaway">Over all ten settings the gap is 4.6–7.9×. Charged for bytes instead, since our 4-bit scan reads every key, the best
baselines reach 0.6–1.6× our 90th-percentile error, but their largest errors stay at least 4.9× ours.</p>

<figure class="csa-fig">
  <img src="/images/projects/2026-10-certified-sparse-attention/fig2-error-vs-keys.png" alt="Figure 2: 90th-percentile error versus keys read on four settings." loading="lazy">
  <figcaption><strong>Figure 2.</strong> On every setting shown, certified stopping (blue, &epsilon; from 0.05 to 0.3) lies below the Pareto front of every
  baseline family. The same estimator at one fixed threshold (dashed) already beats most baselines, but at the closest number of keys read its
  90th-percentile error is 2.4&ndash;9.6&times; ours: choosing the stopping level per head carries most of the gain.</figcaption>
</figure>

### No certificate violated, and no heavy tail

<figure class="csa-fig">
  <img src="/images/projects/2026-10-certified-sparse-attention/fig3-error-tails.png" alt="Figure 3: per-query error distributions and the calibration of the certificates." loading="lazy">
  <figcaption><strong>Figure 3.</strong> At matched keys read, the strongest baselines have heavy error tails, with maxima of 0.39&ndash;1.62&nbsp;&times;&nbsp;M across
  the ten settings, while our largest error at &epsilon; = 0.1 is 0.021&ndash;0.029. Across 120 configurations (10 settings &times; 3 error measures &times;
  4 values of &epsilon;), no error exceeds &epsilon;; the largest is 0.31&epsilon; on the value scale and 0.41&epsilon; on the output norm (c).</figcaption>
</figure>

### End to end: within one point of dense attention

<div class="csa-table" markdown="1">

**Table 2 (excerpt). End-to-end score change versus dense attention (ε = 0.1, fresh perturbations at every step), with task-stratified 95% CIs.**

| Model · data (samples) | Keys read | Δ score [95% CI] | Same output as dense |
|---|---:|---|---:|
| Llama · LongBench (300) | 5.26% | −0.29 [−0.63, −0.03] | 0.80 |
| Llama · RULER 32K (200) | 1.50% | −0.65 [−2.05, +0.35] | 0.93 |
| Llama · RULER 128K (48) | 0.49% | 0.00 [0.00, 0.00] | 0.90 |
| Qwen3 · RULER 32K (100) | 1.53% | +1.00 [−0.20, +3.10] | 0.85 |
| Qwen2.5 · LongBench (150) | 5.38% | +0.07 [−0.09, +0.25] | 0.79 |

</div>

<p class="csa-takeaway">Mean scores stay within one point of dense attention and PG-19 perplexity changes by −0.02% to +0.07%. At similar
bytes the strongest baselines score as well, so the end-to-end benefit is the per-call guarantee rather than higher task scores.</p>

### Trade-offs

<div class="csa-callout">
  <ul>
    <li><strong>Bytes, not just keys.</strong> Finding the sample takes a 4-bit scan of every key, a fixed 16.4% of the bytes of dense attention. This floor
      caps the speedup, and at equal bytes the strongest baselines reach our typical error, though not our largest.</li>
    <li><strong>Conservative bounds.</strong> Observed errors are about &epsilon;/10 at the 90th percentile, and the guarantee costs 1.8&ndash;3.5&times; the keys
      of an uncertified central-limit rule.</li>
    <li><strong>Scope.</strong> All measurements are on CPUs. With seven query heads per KV head (Qwen2.5), attention is not faster.</li>
  </ul>
</div>

## Interactive demo

Step one attention head through the threshold levels and watch its certificate tighten until it stops, then compare
certified stopping with a fixed level and with oracle top-k on 200 heads. The demo runs the paper's value-scale certificate
in your browser on synthetic heads with 32,768 keys.

<div class="csa-demo">
  <div class="csa-demo__bar"><span><span class="csa-dot"></span>Live demo &middot; runs locally in your browser, no server</span>
    <a href="/files/projects/2026-10-certified-sparse-attention/demo/index.html" target="_blank" rel="noopener">Open full screen <i class="fa-solid fa-up-right-from-square" aria-hidden="true"></i></a></div>
  <iframe src="/files/projects/2026-10-certified-sparse-attention/demo/index.html" width="100%" height="1180" loading="lazy" title="Certified sparse attention interactive demo"></iframe>
</div>

## Links and citation

<div class="csa-links">
  <a class="csa-btn" href="https://arxiv.org/abs/2610.24991"><i class="ai ai-arxiv" aria-hidden="true"></i> arXiv 2610.24991</a>
  <a class="csa-btn" href="/files/certified-sparse-attention.pdf"><i class="fa-solid fa-file-pdf" aria-hidden="true"></i> Paper (PDF)</a>
  <a class="csa-btn" href="/files/bibtex/li2026certified.bib"><i class="fa-solid fa-quote-right" aria-hidden="true"></i> BibTeX file</a>
  <a class="csa-btn" href="/publication/2026-10-certified-sparse-attention"><i class="fa-solid fa-book-open" aria-hidden="true"></i> Publication page</a>
</div>

<div class="csa-bib">
<button class="csa-copy" type="button" data-target="csa-bibtex">Copy</button>
<pre><code id="csa-bibtex">@misc{li2026certified,
  title         = {Certified Sparse Attention via Perturbed Retrieval},
  author        = {Li, Runguo},
  year          = {2026},
  eprint        = {2610.24991},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.24991}
}</code></pre>
</div>

<script>
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('.csa-copy');
    if (!b) return;
    var text = document.getElementById(b.getAttribute('data-target')).textContent;
    var done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
  });
</script>
