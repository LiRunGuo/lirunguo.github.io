/* Inline interactive figures for the Certified Sparse Attention project page.
   Demo 1 (#csa-demo-call): one call of the certified rule, level by level.
   Demo 2 (#csa-demo-heads): 200 synthetic heads, certified stopping vs. one fixed level vs. oracle top-k.
   Uses core.js (window.CSA), the paper's value-scale certificate with exact scores. Charts are drawn on a white
   plate in both themes, like the paper's figures; the controls follow the site's theme. */
(function () {
  'use strict';
  const C = window.CSA;
  const figA = document.getElementById('csa-demo-call'), figB = document.getElementById('csa-demo-heads');
  if (!C || (!figA && !figB)) return;
  const N = 32768, D = 32, R = 4, EPS = [0.05, 0.1, 0.2, 0.3];
  const COL = { ink: '#1f2933', muted: '#5b6674', grid: '#eceff2', frame: '#8a94a1', soft: '#f1f6fd',
                heavy: '#1d4f91', light: '#5b9bea', unread: '#b7bec7', bound: '#2a78d6', truth: '#6b7280', target: '#1f2933',
                ours: '#2a78d6', fixed: '#eb6834', topk: '#1baf7a', band: '#f3f4f6' };
  const SUP = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  const pow10 = (e) => (e === 0 ? '1' : e === 1 ? '10' : '10' + String(e).split('').map((c) => SUP[c]).join(''));
  const fmtErr = (x) => (!isFinite(x) ? '∞' : x === 0 ? '0' : x < 0.001 ? x.toExponential(1) : x.toPrecision(2));
  const fmtInt = (x) => Math.round(x).toLocaleString('en-US');
  const pct = (k) => (100 * k / N < 10 ? (100 * k / N).toFixed(2) : (100 * k / N).toFixed(1)) + '%';
  const q = (root, role) => root.querySelector('[data-role="' + role + '"]');
  const fontOf = (el) => getComputedStyle(el).fontFamily;

  // ---------- tooltip ----------
  const tip = document.createElement('div');
  tip.className = 'csa-tip'; tip.setAttribute('role', 'tooltip');
  document.body.appendChild(tip);
  function showTip(html, ev) {
    tip.innerHTML = html; tip.style.display = 'block';
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = ev.clientX + 14, y = ev.clientY + 14;
    if (x + w > window.innerWidth - 4) x = ev.clientX - w - 14;
    if (y + h > window.innerHeight - 4) y = ev.clientY - h - 14;
    tip.style.left = Math.max(4, x) + 'px'; tip.style.top = Math.max(4, y) + 'px';
  }
  const hideTip = () => { tip.style.display = 'none'; };

  function seg(el, values, current, label, onPick) {
    el.innerHTML = values.map((v) => `<button type="button" data-v="${v}" aria-pressed="${v === current}">${label(v)}</button>`).join('');
    el.querySelectorAll('button').forEach((b) => b.addEventListener('click', () => {
      el.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', x === b));
      onPick(b.dataset.v);
    }));
  }
  function sizeSvg(svg, h) {
    const w = Math.max(240, Math.round(svg.clientWidth || svg.parentNode.clientWidth));
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('height', h);
    return w;
  }

  let VALS = null;
  const vals = () => VALS || (VALS = C.makeValues(N, D, 11));

  // =====================================================================
  // Demo 1: one call, level by level
  // =====================================================================
  const A = { ready: false, head: 'retrieval', eps: 0.1, pseed: 1, level: 0, res: null, h: null, geo: null, timer: null, cache: {} };
  const PRESETS = {
    retrieval: { label: 'Retrieval', sigma: 1.0, sink: 11, needles: 24, needleLo: 7, needleHi: 10, recency: 1, mu: 2.5, seed: 101 },
    heavy:     { label: 'Heavy-tailed', sigma: 1.8, sink: 8, needles: 8, needleLo: 6, needleHi: 9, recency: 1, mu: 2.5, seed: 202 },
    diffuse:   { label: 'Diffuse', sigma: 0.45, sink: 3, needles: 0, needleLo: 0, needleHi: 0, recency: 0.5, mu: 2.5, seed: 303 },
  };
  const stopA = () => (A.res.stop >= 0 ? A.res.stop : A.res.trace.keys.length - 1);

  function computeA() {
    if (!A.cache[A.head]) {
      const p = PRESETS[A.head];
      A.cache[A.head] = C.prepare(C.makeScores(N, p, p.seed), vals(), C.makeMu(D, p.mu, p.seed + 7));
    }
    const h = A.cache[A.head];
    const E = C.perturb(N, 9973 * A.pseed + PRESETS[A.head].seed);
    const res = C.run(h, E, { eps: A.eps, trace: true, certAfterStop: true, traceAfter: 16 });
    A.h = h; A.res = res;
    const L = res.trace.keys.length;
    q(figA, 'level').max = L - 1;
    const np = h.npool, x = new Float32Array(np), y = new Float32Array(np), xs = [];
    for (let i = 0; i < np; i++) { x[i] = (h.s[i] - h.c) / Math.LN2; y[i] = x[i] + E[i] / Math.LN2; }
    for (let i = 0; i < np; i += 7) xs.push(x[i]);
    xs.sort((a, b) => a - b);
    const xmin = Math.floor(Math.min(xs[Math.floor(0.002 * xs.length)], -(L - 1) / R - 1) / 2) * 2 - 1;
    A.geo = { x, y, je: res.je, xmin, xmax: 1, ymin: xmin, ymax: 6 };
  }
  function setLevel(j) {
    A.level = Math.max(0, Math.min(j, A.res.trace.keys.length - 1));
    q(figA, 'level').value = A.level;
    drawA();
  }
  function drawA() {
    if (!A.ready) return;
    drawScatter(); drawTrace(); drawReadout();
  }

  function drawScatter() {
    const canvas = q(figA, 'scatter'), ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W) return;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const m = { l: 42, r: 6, t: 6, b: 32 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    const g = A.geo, j = A.level, lt = -j / R, font = fontOf(canvas);
    const sx = (v) => m.l + (v - g.xmin) / (g.xmax - g.xmin) * pw;
    const sy = (v) => m.t + (1 - (Math.min(v, g.ymax) - g.ymin) / (g.ymax - g.ymin)) * ph;
    ctx.fillStyle = COL.soft; ctx.fillRect(m.l, m.t, pw, sy(lt) - m.t);
    ctx.font = '11px ' + font; ctx.fillStyle = COL.muted; ctx.strokeStyle = COL.grid; ctx.lineWidth = 1;
    const step = (g.xmax - g.xmin) > 24 ? 8 : 4;
    ctx.textAlign = 'center';
    for (let v = 0; v >= g.xmin; v -= step) {
      ctx.beginPath(); ctx.moveTo(sx(v), m.t); ctx.lineTo(sx(v), m.t + ph); ctx.stroke();
      ctx.fillText(String(v).replace('-', '−'), sx(v), m.t + ph + 13);
    }
    ctx.textAlign = 'right';
    for (let v = 4; v >= g.ymin; v -= step) {
      ctx.beginPath(); ctx.moveTo(m.l, sy(v)); ctx.lineTo(m.l + pw, sy(v)); ctx.stroke();
      ctx.fillText(String(v).replace('-', '−'), m.l - 5, sy(v) + 4);
    }
    ctx.textAlign = 'center';
    ctx.fillText('log₂ wᵢ (weight relative to the maximum)', m.l + pw / 2, H - 3);
    ctx.save(); ctx.translate(10, m.t + ph / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText('perturbed score log₂(wᵢ e^Eᵢ)', 0, 0); ctx.restore();
    ctx.setLineDash([1.5, 3]); ctx.strokeStyle = COL.frame;
    ctx.beginPath(); ctx.moveTo(sx(g.xmin), sy(g.xmin)); ctx.lineTo(sx(0), sy(0)); ctx.stroke(); ctx.setLineDash([]);
    const np = g.x.length, X = g.x, Y = g.y, je = g.je;
    ctx.fillStyle = COL.unread; ctx.globalAlpha = 0.5;
    for (let i = 0; i < np; i++) if (je[i] > j && X[i] >= g.xmin) ctx.fillRect(sx(X[i]) - 0.8, sy(Y[i]) - 0.8, 1.6, 1.6);
    ctx.globalAlpha = 1; ctx.fillStyle = COL.light;
    for (let i = 0; i < np; i++) if (je[i] <= j && X[i] < lt) ctx.fillRect(sx(X[i]) - 1.5, sy(Y[i]) - 1.5, 3, 3);
    ctx.fillStyle = COL.heavy;
    for (let i = 0; i < np; i++) if (X[i] >= lt) ctx.fillRect(sx(X[i]) - 1.7, sy(Y[i]) - 1.7, 3.4, 3.4);
    ctx.strokeStyle = COL.bound; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(m.l, sy(lt)); ctx.lineTo(m.l + pw, sy(lt)); ctx.stroke();
    ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(sx(lt), sy(lt)); ctx.lineTo(sx(lt), m.t + ph); ctx.stroke(); ctx.setLineDash([]);
    // labels: threshold (on a backing so it stays legible over the points), heavy side, E_i = 0
    ctx.font = '600 11px ' + font; ctx.textAlign = 'left';
    const thr = 'log₂ τ = ' + (lt === 0 ? '0' : '−' + (j / R)) + '  ↑ read above the line';
    const tw = ctx.measureText(thr).width, ty = Math.max(m.t + 14, sy(lt) - 6);
    ctx.fillStyle = '#ffffff'; ctx.globalAlpha = 0.88; ctx.fillRect(m.l + 2, ty - 11, tw + 8, 15); ctx.globalAlpha = 1;
    ctx.fillStyle = COL.bound; ctx.fillText(thr, m.l + 6, ty);
    const right = sx(lt) > m.l + pw - 60;
    ctx.textAlign = right ? 'right' : 'left';
    ctx.fillText(right ? '← light' : 'heavy →', sx(lt) + (right ? -5 : 5), m.t + ph - 6);
    ctx.font = '11px ' + font; ctx.fillStyle = COL.muted; ctx.textAlign = 'left';
    const ang = Math.atan2(sy(0) - sy(g.xmin), sx(0) - sx(g.xmin)), dx = g.xmin + 0.35 * (0 - g.xmin);
    ctx.save(); ctx.translate(sx(dx), sy(dx)); ctx.rotate(ang); ctx.fillText('Eᵢ = 0', 4, 13); ctx.restore();
    ctx.strokeStyle = COL.frame; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, m.t + ph); ctx.lineTo(m.l + pw, m.t + ph); ctx.stroke();
  }

  function drawTrace() {
    const svg = q(figA, 'trace'), H = 250, W = sizeSvg(svg, H);
    const t = A.res.trace, L = t.keys.length, m = { l: 44, r: 10, t: 10, b: 32 }, ymin = -4, ymax = 1;
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const X = (j) => m.l + (L > 1 ? j / (L - 1) : 0) * pw;
    const Y = (v) => {
      const lv = !isFinite(v) || v <= 0 ? (v <= 0 ? ymin : ymax) : Math.max(ymin, Math.min(ymax, Math.log10(v)));
      return m.t + (1 - (lv - ymin) / (ymax - ymin)) * ph;
    };
    const tgt = t.tgt.map((v, j, a) => (isFinite(v) ? v : (a.find(isFinite) || A.eps)));
    const path = (arr) => arr.map((v, j) => `${j ? 'L' : 'M'}${X(j).toFixed(1)} ${Y(v).toFixed(1)}`).join('');
    let s = `<path d="${path(tgt)}L${X(L - 1)} ${m.t + ph}L${X(0)} ${m.t + ph}Z" fill="${COL.soft}"/>`;
    for (let e = ymin; e <= ymax; e++) {
      s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${Y(Math.pow(10, e))}" y2="${Y(Math.pow(10, e))}" stroke="${COL.grid}"/>`;
      s += `<text x="${m.l - 6}" y="${Y(Math.pow(10, e)) + 4}" text-anchor="end">${pow10(e)}</text>`;
    }
    const xstep = L > 60 ? 20 : L > 30 ? 10 : 5;
    for (let j = 0; j < L; j += xstep) s += `<text x="${X(j)}" y="${m.t + ph + 14}" text-anchor="middle">${j}</text>`;
    s += `<text x="${m.l + pw / 2}" y="${H - 3}" text-anchor="middle">level j (each level lowers τ by a quarter octave)</text>`;
    s += `<text transform="translate(11 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">error / M</text>`;
    s += `<text x="${m.l + 6}" y="${m.t + ph - 6}" style="fill:${COL.bound}">certified once β is in the shaded zone</text>`;
    s += `<path d="${path(tgt)}" fill="none" stroke="${COL.target}" stroke-width="1.6" stroke-dasharray="1.5 2.5"/>`;
    s += `<path d="${path(t.err)}" fill="none" stroke="${COL.truth}" stroke-width="2" stroke-dasharray="4 3"/>`;
    s += `<path d="${path(t.beta)}" fill="none" stroke="${COL.bound}" stroke-width="2.5" stroke-linejoin="round"/>`;
    const j = A.level, st = A.res.stop;
    s += `<line x1="${X(j)}" x2="${X(j)}" y1="${m.t}" y2="${m.t + ph}" stroke="${COL.muted}" stroke-width="1"/>`;
    s += `<circle cx="${X(j)}" cy="${Y(t.beta[j])}" r="3.5" fill="${COL.bound}"/>`;
    if (st >= 0) {
      s += `<circle cx="${X(st)}" cy="${Y(t.beta[st])}" r="5.5" fill="${COL.bound}" stroke="#fff" stroke-width="2.5"/>`;
      s += `<text x="${X(st)}" y="${Y(t.beta[st]) - 10}" text-anchor="middle" style="fill:${COL.bound};font-weight:700">stop</text>`;
    }
    s += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ph}" stroke="${COL.frame}"/>`;
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${m.t + ph}" y2="${m.t + ph}" stroke="${COL.frame}"/>`;
    s += `<rect data-role="hit" x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="transparent" style="cursor:pointer"/>`;
    svg.innerHTML = s;
    const hit = svg.querySelector('[data-role="hit"]');
    const levelAt = (ev) => {
      const r = svg.getBoundingClientRect(), vx = (ev.clientX - r.left) / r.width * W;
      return Math.max(0, Math.min(L - 1, Math.round((vx - m.l) / pw * (L - 1))));
    };
    hit.addEventListener('mousemove', (ev) => {
      const jj = levelAt(ev);
      showTip(`<b>Level ${jj}</b> &middot; ${fmtInt(t.keys[jj])} keys (${pct(t.keys[jj])})<br>bound &beta;/M ${fmtErr(t.beta[jj])}` +
        `<br>target &epsilon;M<sub>lo</sub>/M ${fmtErr(t.tgt[jj])}<br>true error/M ${fmtErr(t.err[jj])}` +
        (t.ok[jj] ? '<br>&#10003; certified' : '') + '<br><span style="opacity:.7">click to go to this level</span>', ev);
    });
    hit.addEventListener('mouseleave', hideTip);
    hit.addEventListener('click', (ev) => { stopPlay(); setLevel(levelAt(ev)); });
  }

  function drawReadout() {
    const t = A.res.trace, j = A.level, st = A.res.stop;
    q(figA, 'lvl').textContent = 'j = ' + j;
    q(figA, 'tau').textContent = j === 0 ? 'τ = 1, the largest weight' : 'τ = 2^−' + (j / R);
    let verdict;
    if (st < 0) verdict = 'no level up to J is certified, so the head falls back to exact attention';
    else if (j < st) verdict = `not certified yet: the bound is above the target (this head stops at level ${st})`;
    else if (j === st) verdict = '<b class="csa-ok">&#10003; certified: the head stops here</b>';
    else verdict = `past the stop at level ${st}: deeper levels read more keys for a tighter bound`;
    q(figA, 'readout').innerHTML =
      `<b>${fmtInt(t.keys[j])}</b> of 32,768 keys read (${pct(t.keys[j])}: ${fmtInt(t.nB[j])} heavy, ${fmtInt(t.k[j])} sampled light, ` +
      `${A.h.nwin} recent) &middot; true error <b>${fmtErr(t.err[j])}</b> &middot; bound &beta; <b>${fmtErr(t.beta[j])}</b> &middot; ` +
      `target &epsilon;M<sub>lo</sub> <b>${fmtErr(t.tgt[j])}</b> (all &divide; M). ${verdict}.`;
  }

  function stopPlay() {
    if (A.timer) { clearInterval(A.timer); A.timer = null; q(figA, 'play').innerHTML = '&#9654;&#xFE0E; Play'; }
  }
  function initA() {
    if (A.ready || !figA) return;
    seg(q(figA, 'head'), Object.keys(PRESETS), A.head, (v) => PRESETS[v].label, (v) => { stopPlay(); A.head = v; computeA(); setLevel(stopA()); });
    seg(q(figA, 'eps'), EPS.map(String), String(A.eps), (v) => v, (v) => { stopPlay(); A.eps = parseFloat(v); computeA(); setLevel(stopA()); });
    q(figA, 'play').addEventListener('click', () => {
      if (A.timer) { stopPlay(); return; }
      const end = stopA();
      if (A.level >= end) setLevel(0);
      q(figA, 'play').innerHTML = '&#10074;&#10074; Pause';
      A.timer = setInterval(() => { if (A.level >= end) { stopPlay(); return; } setLevel(A.level + 1); }, 150);
    });
    q(figA, 'redraw').addEventListener('click', () => { stopPlay(); A.pseed++; computeA(); setLevel(stopA()); });
    q(figA, 'level').addEventListener('input', (e) => { stopPlay(); setLevel(parseInt(e.target.value, 10)); });
    computeA(); A.ready = true; setLevel(stopA());
  }

  // =====================================================================
  // Demo 2: 200 heads
  // =====================================================================
  const HEADS = 200;
  const B = { ready: false, eps: 0.1, seed: 1, token: 0, res: [], fixed: -1, hover: -1, G: null };
  const PB = { H: 320, m: { l: 48, r: 12, t: 10, b: 34 }, xmin: -1, xmax: 2, ymin: -4, ymax: 0 };
  let PW = 600;
  const px = (k) => PB.m.l + (Math.log10(Math.max(100 * k / N, 1e-9)) - PB.xmin) / (PB.xmax - PB.xmin) * (PW - PB.m.l - PB.m.r);
  const py = (e) => PB.m.t + (1 - (Math.max(PB.ymin, Math.min(PB.ymax, Math.log10(Math.max(e, 1e-12)))) - PB.ymin) / (PB.ymax - PB.ymin)) * (PB.H - PB.m.t - PB.m.b);
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const quant = (a, p) => { const b = Float64Array.from(a).sort(); return b[Math.min(b.length - 1, Math.floor(p * b.length))]; };
  const at = (arr, j) => arr[Math.min(j, arr.length - 1)];

  function runB() {
    const my = ++B.token;
    B.res = []; B.fixed = -1; B.hover = -1;
    if (!B.G) { const r = C.mulberry32(5); B.G = new Float64Array(N); for (let i = 0; i < N; i++) B.G[i] = C.gaussian(r); }
    const r = C.mulberry32(1000003 * B.seed + 17), specs = [];
    for (let k = 0; k < HEADS; k++) {
      specs.push({ p: { sigma: 0.5 + 1.3 * r(), sink: 7 + 8 * r(), needles: Math.floor(48 * r()), needleLo: 4, needleHi: 5 + 8 * r(),
                        recency: 2 * r(), mu: 1 + 4 * r() },
                   s1: 1 + Math.floor(r() * 1e9), s2: 1 + Math.floor(r() * 1e9), s3: 1 + Math.floor(r() * 1e9) });
    }
    q(figB, 'table').innerHTML = '';
    let k = 0;
    const chunk = () => {
      if (my !== B.token) return;
      const end = Math.min(HEADS, k + 10);
      for (; k < end; k++) {
        const sp = specs[k];
        const h = C.prepare(C.makeScores(N, sp.p, sp.s1, B.G), vals(), C.makeMu(D, sp.p.mu, sp.s2));
        const o = C.run(h, C.perturb(N, sp.s3), { eps: B.eps, errAt: true, keysCap: 0.5 * N });
        B.res.push({ keys: o.keys, err: o.err, stop: o.stop, errAt: o.errAt, keysAt: o.keysAt, topk: C.topk(h, o.keys - h.nwin), eff: h.effKeys });
      }
      q(figB, 'progress').textContent = k < HEADS ? `computing ${k} / ${HEADS} heads…` : '';
      if (k >= HEADS) finishB();
      drawB();
      if (k < HEADS) setTimeout(chunk, 0);
    };
    q(figB, 'progress').textContent = 'computing…';
    setTimeout(chunk, 0);
  }

  function finishB() {
    const P = B.res, ours = mean(P.map((x) => x.keys));
    let best = 0, bd = Infinity;
    for (let j = 0; j <= 160; j++) {
      const kj = mean(P.map((x) => at(x.keysAt, j)));
      if (Math.abs(kj - ours) < bd) { bd = Math.abs(kj - ours); best = j; }
    }
    B.fixed = best;
    P.forEach((x) => { x.fkeys = at(x.keysAt, best); x.ferr = at(x.errAt, best); });
    const rows = [
      ['csa-row-ours', 'Certified stopping (ours)', P.map((x) => x.keys), P.map((x) => x.err)],
      ['', `Same estimator, one fixed level (j = ${best})`, P.map((x) => x.fkeys), P.map((x) => x.ferr)],
      ['', 'Oracle top-k, our keys per head', P.map((x) => x.keys), P.map((x) => x.topk)],
    ];
    q(figB, 'table').innerHTML = '<table><thead><tr><th>Rule</th><th>Keys read (mean)</th><th>Error/M p50</th><th>p90</th><th>max</th>' +
      `<th>Heads above &epsilon;</th></tr></thead><tbody>` +
      rows.map(([cls, name, ks, es]) => `<tr class="${cls}"><td>${name}</td><td>${pct(mean(ks))}</td><td>${fmtErr(quant(es, 0.5))}</td>` +
        `<td>${fmtErr(quant(es, 0.9))}</td><td>${fmtErr(Math.max(...es))}</td><td>${es.filter((e) => e > B.eps).length} of ${HEADS}</td></tr>`).join('') +
      '</tbody></table>';
  }

  function drawB() {
    if (!B.ready) return;
    const svg = q(figB, 'plot');
    PW = sizeSvg(svg, PB.H);
    const P = B.res, m = PB.m, pw = PW - m.l - m.r, ph = PB.H - m.t - m.b, ey = py(B.eps);
    let s = `<rect x="${m.l}" y="${m.t}" width="${pw}" height="${ey - m.t}" fill="${COL.band}"/>`;
    s += `<text x="${m.l + 6}" y="${m.t + 15}">error above the target</text>`;
    for (let e = PB.ymin; e <= PB.ymax; e++) {
      s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${py(Math.pow(10, e))}" y2="${py(Math.pow(10, e))}" stroke="${COL.grid}"/>`;
      s += `<text x="${m.l - 6}" y="${py(Math.pow(10, e)) + 4}" text-anchor="end">${pow10(e)}</text>`;
    }
    [[0.1, '0.1%'], [1, '1%'], [10, '10%'], [100, '100%']].forEach(([v, lab]) => {
      const x = px(v * N / 100);
      s += `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${m.t + ph}" stroke="${COL.grid}"/>`;
      s += `<text x="${x}" y="${m.t + ph + 14}" text-anchor="middle">${lab}</text>`;
    });
    s += `<text x="${m.l + pw / 2}" y="${PB.H - 3}" text-anchor="middle">keys read (% of 32,768)</text>`;
    s += `<text transform="translate(11 ${m.t + ph / 2}) rotate(-90)" text-anchor="middle">error / M</text>`;
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${ey}" y2="${ey}" stroke="${COL.target}" stroke-width="1.4" stroke-dasharray="1.5 2.5"/>`;
    s += `<text x="${m.l + pw - 4}" y="${ey - 5}" text-anchor="end" style="fill:${COL.ink};font-weight:600">target &#949; = ${B.eps}</text>`;
    const fixed = B.fixed >= 0;
    P.forEach((x) => { const cx = px(x.keys), cy = py(x.topk); s += `<rect x="${(cx - 3.2).toFixed(1)}" y="${(cy - 3.2).toFixed(1)}" width="6.4" height="6.4" fill="none" stroke="${COL.topk}" stroke-width="1.4"/>`; });
    if (fixed) P.forEach((x) => { const cx = px(x.fkeys), cy = py(x.ferr); s += `<path d="M${(cx - 3.3).toFixed(1)} ${(cy - 3.3).toFixed(1)}l6.6 6.6m0 -6.6l-6.6 6.6" stroke="${COL.fixed}" stroke-width="1.6"/>`; });
    P.forEach((x) => { s += `<circle cx="${px(x.keys).toFixed(1)}" cy="${py(x.err).toFixed(1)}" r="3.6" fill="${COL.ours}" stroke="#fff" stroke-width="0.9"/>`; });
    if (fixed && P.length && pw >= 480) {   // on narrow screens the caption carries these two notes
      const byK = [...P].sort((a, b) => a.fkeys - b.fkeys), n10 = Math.max(1, Math.floor(0.1 * P.length));
      const left = byK.slice(0, n10), right = byK.slice(-n10);
      // left label in the empty band just above the target line, over the peaked heads
      const lx = Math.max(m.l + 4, Math.min(...left.map((x) => px(x.fkeys))) - 6);
      s += `<text x="${lx}" y="${ey - 8}" style="fill:${COL.fixed};font-weight:650">peaked heads: too few keys &#8595;</text>`;
      const ry = Math.min(m.t + ph - 8, Math.max(...right.map((x) => py(x.ferr))) + 18);
      const rx = Math.min(m.l + pw - 4, Math.max(...right.map((x) => px(x.fkeys))));
      s += `<text x="${rx}" y="${ry}" text-anchor="end" style="fill:${COL.fixed};font-weight:650">&#8599; flat heads: too many keys</text>`;
    }
    if (B.hover >= 0 && P[B.hover]) {
      const x = P[B.hover];
      const pts = [[px(x.keys), py(x.err)], [px(x.keys), py(x.topk)]].concat(fixed ? [[px(x.fkeys), py(x.ferr)]] : []);
      pts.forEach(([cx, cy]) => { s += `<circle cx="${cx}" cy="${cy}" r="8.5" fill="none" stroke="${COL.ink}" stroke-width="1.6"/>`; });
    }
    s += `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${m.t + ph}" stroke="${COL.frame}"/>`;
    s += `<line x1="${m.l}" x2="${m.l + pw}" y1="${m.t + ph}" y2="${m.t + ph}" stroke="${COL.frame}"/>`;
    svg.innerHTML = s;
  }

  function initB() {
    if (B.ready || !figB) return;
    B.ready = true;
    seg(q(figB, 'eps'), EPS.map(String), String(B.eps), (v) => v, (v) => { B.eps = parseFloat(v); runB(); });
    q(figB, 'new').addEventListener('click', () => { B.seed++; runB(); });
    const svg = q(figB, 'plot');
    svg.addEventListener('mousemove', (ev) => {
      const P = B.res, r = svg.getBoundingClientRect();
      const vx = (ev.clientX - r.left) / r.width * PW, vy = (ev.clientY - r.top) / r.height * PB.H;
      let best = -1, bd = 12 * 12;
      P.forEach((x, i) => {
        const cand = [[px(x.keys), py(x.err)], [px(x.keys), py(x.topk)]];
        if (B.fixed >= 0) cand.push([px(x.fkeys), py(x.ferr)]);
        cand.forEach(([cx, cy]) => { const d2 = (cx - vx) ** 2 + (cy - vy) ** 2; if (d2 < bd) { bd = d2; best = i; } });
      });
      if (best !== B.hover) { B.hover = best; drawB(); }
      if (best < 0) { hideTip(); return; }
      const x = P[best];
      showTip(`<b>Head ${best + 1}</b> &middot; effective keys ${fmtInt(x.eff)}<br>` +
        `certified: ${pct(x.keys)} keys, error/M ${fmtErr(x.err)}` + (x.stop < 0 ? ' (exact fallback)' : ` (level ${x.stop})`) + '<br>' +
        (B.fixed >= 0 ? `fixed level ${B.fixed}: ${pct(x.fkeys)} keys, error/M ${fmtErr(x.ferr)}<br>` : '') +
        `oracle top-k: ${pct(x.keys)} keys, error/M ${fmtErr(x.topk)}`, ev);
    });
    svg.addEventListener('mouseleave', () => { B.hover = -1; hideTip(); drawB(); });
    runB();
  }

  // ---------- start each figure when it comes near the viewport ----------
  const start = (el) => (el === figA ? initA() : initB());
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((ents) => ents.forEach((e) => { if (e.isIntersecting) { start(e.target); io.unobserve(e.target); } }),
                                        { rootMargin: '300px 0px' });
    [figA, figB].forEach((el) => el && io.observe(el));
  }
  // fallback: start both once the page has loaded and gone idle (Demo 2 computes in small chunks)
  window.addEventListener('load', () => setTimeout(() => { initA(); initB(); }, 1200));
  let rz = null;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { drawA(); drawB(); }, 100); });
})();
