/* Project page "Weight Placement Is Not Enough": speed replay, per-thread trace of one decoder layer, result explorer.
   All numbers come from data.js (window.WPNE_DATA), which is exported from the paper's result files. */
(function () {
  'use strict';
  var D = window.WPNE_DATA;
  if (!D) return;
  var $ = function (id) { return document.getElementById(id); };
  var MODELS = { llama3_8b: 'Llama3-8B', falcon3_10b: 'Falcon3-10B', falcon3_1b: 'Falcon3-1B' };
  var PLATS = { icl: 'Ice Lake', emr: 'Emerald Rapids' };
  var THREADS = { icl: 64, emr: 104 };
  var BW = { icl: 354.2, emr: 610.4 };
  var IMPL = [['orig', 'ik', 'var(--c-ik)'], ['a2', 'ik-P', 'var(--c-ikp)'], ['a3', 'Row', 'var(--c-row)'], ['a3col', 'Col', 'var(--c-col)']];
  var state = { plat: 'emr', model: 'llama3_8b' };
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var fmt1 = function (x) { return x.toFixed(1); };
  var fmt2 = function (x) { return x.toFixed(2); };
  var listeners = [];

  function onView(el, fn) {   // run fn once when el is first on screen
    if (!('IntersectionObserver' in window)) { fn(); return; }
    var io = new IntersectionObserver(function (es) {
      if (es.some(function (e) { return e.isIntersecting; })) { io.disconnect(); fn(); }
    }, { threshold: 0.35 });
    io.observe(el);
  }

  /* ---------- machine / model chips (shared by the replay and the explorer) ---------- */
  function syncChips() {
    document.querySelectorAll('[data-wp-plat]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-wp-plat') === state.plat)); });
    document.querySelectorAll('[data-wp-model]').forEach(function (b) { b.setAttribute('aria-pressed', String(b.getAttribute('data-wp-model') === state.model)); });
  }
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('[data-wp-plat],[data-wp-model]');
    if (!b) return;
    if (b.hasAttribute('data-wp-plat')) state.plat = b.getAttribute('data-wp-plat');
    else state.model = b.getAttribute('data-wp-model');
    syncChips();
    listeners.forEach(function (f) { f(b); });
  });

  /* ---------- 1. replay of the measured decode rates ---------- */
  (function () {
    var box = $('wp-race');
    if (!box) return;
    var lanes = [].slice.call(box.querySelectorAll('.lane'));
    var raf = 0, seen = false;
    function setup() {
      var L = D.ladder[state.plat][state.model];
      var N = Math.max(64, 64 * Math.round(L.tok.orig * 6 / 64));   // ik needs about 6 s
      $('wp-race-cap').textContent = PLATS[state.plat] + ', ' + THREADS[state.plat] + ' threads on 4 NUMA nodes · ' + MODELS[state.model] +
        ' (IQ2_BN) · decoding ' + N + ' tokens';
      lanes.forEach(function (ln) {
        var v = ln.getAttribute('data-v');
        ln.querySelector('.rate b').textContent = fmt1(L.tok[v]);
        ln.querySelector('.rate em').textContent = v === 'orig' ? 'baseline' : fmt2(L.ratio[v]) + '× ik';
      });
      return { L: L, N: N };
    }
    function paint(cfg, t) {
      var done = true;
      lanes.forEach(function (ln) {
        var r = cfg.L.tok[ln.getAttribute('data-v')], n = Math.min(cfg.N, r * t);
        ln.querySelector('.track i').style.width = (100 * n / cfg.N) + '%';
        ln.querySelector('.cnt').textContent = n >= cfg.N ? 'done in ' + fmt2(cfg.N / r) + ' s' : Math.floor(n) + ' / ' + cfg.N;
        ln.classList.toggle('done', n >= cfg.N);
        if (n < cfg.N) done = false;
      });
      return done;
    }
    function run() {
      cancelAnimationFrame(raf);
      var cfg = setup();
      if (reduced) { paint(cfg, 1e9); return; }
      var t0 = performance.now();
      (function frame(now) {
        if (!paint(cfg, (now - t0) / 1000)) raf = requestAnimationFrame(frame);
      })(t0);
    }
    paint(setup(), 0);
    onView(box, function () { seen = true; run(); });
    $('wp-race-again').addEventListener('click', run);
    listeners.push(function () { if (seen) run(); else paint(setup(), 0); });
  })();

  /* ---------- 2. per-thread trace of one decoder layer ---------- */
  (function () {
    var cv = $('wp-trace');
    if (!cv || !cv.getContext) return;
    var ctx = cv.getContext('2d'), T = D.trace, tip = $('wp-tip'), out = $('wp-trace-out');
    var KC = ['#56B4E9', '#009E73', '#F0E442', '#E69F00', '#D55E00', '#CC79A7', '#D4D4D4', '#9C9C9C', '#4A4A4A'];
    var KL = ['computing QKV + RoPE', 'computing attention', 'computing the O projection', 'computing gate/up + SiLU', 'computing the down projection',
      'node leader: norm + quantization', 'waiting for the last thread', 'waiting for its node leader', 'inside the barrier (latency)'];
    var NAMES = { row: 'Row', col: 'Col' }, NG = { row: 5, col: 2 }, VS = ['row', 'col'];
    var XMAX = 178, ML = 50, MR = 8, GAP = 7, TITLE = 20, MARK = 9, BETWEEN = 16, AXIS = 42;
    var G = {}, base = document.createElement('canvas'), head = null, raf = 0;
    var by = {};
    VS.forEach(function (v) {
      by[v] = [];
      for (var t = 0; t < T[v].threads; t++) by[v].push([]);
      T[v].segs.forEach(function (s) { by[v][s[0]].push(s); });
      by[v].forEach(function (a) { a.sort(function (p, q) { return p[2] - q[2]; }); });
    });
    function layout() {
      var dpr = window.devicePixelRatio || 1, W = cv.clientWidth;
      var rh = Math.max(1.5, Math.round(1.6 * dpr) / dpr);
      var nn = T.row.threads / T.row.tpn, ph = T.row.threads * rh + (nn - 1) * GAP;
      G = { dpr: dpr, W: W, rh: rh, ph: ph, sx: (W - ML - MR) / XMAX, y0: {} };
      G.y0.row = TITLE + MARK;
      G.y0.col = G.y0.row + ph + BETWEEN + TITLE + MARK;
      G.H = G.y0.col + ph + AXIS;
      [cv, base].forEach(function (c) { c.width = Math.round(W * dpr); c.height = Math.round(G.H * dpr); });
      cv.style.height = G.H + 'px';
    }
    function yT(v, t) { return G.y0[v] + t * G.rh + Math.floor(t / T[v].tpn) * GAP; }
    function drawBase() {
      var c = base.getContext('2d');
      c.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
      c.fillStyle = '#fff'; c.fillRect(0, 0, G.W, G.H);
      c.textBaseline = 'middle';
      VS.forEach(function (v) {
        var tv = T[v], nn = tv.threads / tv.tpn;
        c.fillStyle = '#1d1d1b'; c.font = '600 12px system-ui, -apple-system, "Segoe UI", sans-serif'; c.textAlign = 'left';
        c.fillText(NAMES[v] + ' · ' + NG[v] + ' global barriers per layer · ' + (G.W < 520 ? '' : 'this layer takes ') + Math.round(tv.end) + ' µs', ML, G.y0[v] - MARK - TITLE / 2 + 1);
        tv.segs.forEach(function (s) {
          c.fillStyle = KC[s[1]];
          c.fillRect(ML + s[2] / 10 * G.sx, yT(v, s[0]), Math.max((s[3] - s[2]) / 10 * G.sx, 0.4), G.rh + 0.05);
        });
        c.fillStyle = '#52514e'; c.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif'; c.textAlign = 'right';
        for (var n = 0; n < nn; n++) c.fillText('node ' + n, ML - 6, yT(v, n * tv.tpn) + tv.tpn * G.rh / 2);
        c.fillStyle = '#1d1d1b';
        tv.global.forEach(function (g) {
          var x = ML + g * G.sx, y = G.y0[v] - 2;
          c.beginPath(); c.moveTo(x - 4, y - 7); c.lineTo(x + 4, y - 7); c.lineTo(x, y); c.closePath(); c.fill();
        });
      });
      var ya = G.y0.col + G.ph + 5;
      c.strokeStyle = '#898781'; c.lineWidth = 1; c.fillStyle = '#52514e'; c.textAlign = 'center';
      c.beginPath(); c.moveTo(ML, ya + 0.5); c.lineTo(ML + XMAX * G.sx, ya + 0.5); c.stroke();
      for (var x = 0; x <= XMAX; x += 25) {
        c.beginPath(); c.moveTo(ML + x * G.sx + 0.5, ya); c.lineTo(ML + x * G.sx + 0.5, ya + 4); c.stroke();
        if (x % 50 === 0 || G.sx > 3) c.fillText(String(x), ML + x * G.sx, ya + 11);
      }
      c.fillText('time within one decoder layer (µs)', ML + XMAX * G.sx / 2, ya + 27);
    }
    function paint() {
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.drawImage(base, 0, 0);
      if (head === null) return;
      ctx.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
      var x = ML + head * G.sx;
      if (head < XMAX) {
        ctx.fillStyle = 'rgba(255,255,255,0.8)';
        VS.forEach(function (v) { ctx.fillRect(x, G.y0[v] - 1, ML + XMAX * G.sx - x + 1, G.ph + 2); });
      }
      ctx.strokeStyle = '#1d1d1b'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x + 0.5, G.y0.row - MARK); ctx.lineTo(x + 0.5, G.y0.col + G.ph + 4); ctx.stroke();
    }
    function at(v, t, us) {   // segment of thread t at time us
      var a = by[v][t], x = us * 10;
      for (var i = 0; i < a.length; i++) if (a[i][2] <= x && x < a[i][3]) return a[i];
      return null;
    }
    function waiting(v, us) {
      var n = 0;
      for (var t = 0; t < T[v].threads; t++) { var s = at(v, t, us); if (s && s[1] >= 6) n++; }
      return n;
    }
    function readout(us) {
      if (us === null) { out.innerHTML = 'Move over the timeline, or press play, to count the waiting threads at one instant.'; return; }
      out.innerHTML = '<b>t = ' + us.toFixed(1) + ' µs</b> &nbsp;' + VS.map(function (v) {
        return us >= T[v].end ? NAMES[v] + ': layer finished at ' + Math.round(T[v].end) + ' µs'
          : NAMES[v] + ': <b>' + waiting(v, us) + '</b> of ' + T[v].threads + ' threads waiting';
      }).join(' &nbsp;·&nbsp; ');
    }
    function redraw() { layout(); drawBase(); paint(); }
    function play() {
      cancelAnimationFrame(raf);
      var t0 = performance.now(), DUR = 9000;
      (function frame(now) {
        var p = Math.min(1, (now - t0) / DUR);
        head = p * XMAX;
        paint(); readout(Math.min(head, XMAX - 0.1));
        if (p < 1) raf = requestAnimationFrame(frame); else { head = null; paint(); }
      })(t0);
    }
    function locate(e) {
      var r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, us = (x - ML) / G.sx, hit = null;
      VS.forEach(function (v) {
        if (y < G.y0[v] || y > G.y0[v] + G.ph) return;
        for (var t = 0; t < T[v].threads; t++) {
          var yy = yT(v, t);
          if (y >= yy && y < yy + G.rh) hit = { v: v, t: t };
        }
      });
      return { us: us, hit: hit, cx: e.clientX, cy: e.clientY };
    }
    function hover(e) {
      cancelAnimationFrame(raf);
      var p = locate(e);
      if (p.us < 0 || p.us > XMAX) { leave(); return; }
      head = null; paint();
      ctx.setTransform(G.dpr, 0, 0, G.dpr, 0, 0);
      ctx.strokeStyle = 'rgba(29,29,27,0.55)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ML + p.us * G.sx + 0.5, G.y0.row - MARK); ctx.lineTo(ML + p.us * G.sx + 0.5, G.y0.col + G.ph + 4); ctx.stroke();
      readout(p.us);
      var s = p.hit && at(p.hit.v, p.hit.t, p.us);
      if (!s) { tip.hidden = true; return; }
      tip.innerHTML = '<div class="tt">' + NAMES[p.hit.v] + ' · node ' + Math.floor(p.hit.t / T[p.hit.v].tpn) + ' · thread ' + p.hit.t + '</div>' +
        '<div class="tr"><i style="background:' + KC[s[1]] + '"></i><b>' + KL[s[1]] + '</b></div>' +
        '<div class="tr"><span>' + (s[2] / 10).toFixed(1) + ' – ' + (s[3] / 10).toFixed(1) + ' µs (' + ((s[3] - s[2]) / 10).toFixed(1) + ' µs)</span></div>';
      tip.hidden = false;
      var w = tip.offsetWidth, left = Math.min(p.cx + 14, window.innerWidth - w - 8);
      tip.style.left = Math.max(8, left) + 'px';
      tip.style.top = (p.cy + 16) + 'px';
    }
    function leave() { tip.hidden = true; paint(); }
    redraw(); readout(null);
    cv.addEventListener('pointermove', hover);
    cv.addEventListener('pointerdown', hover);
    cv.addEventListener('pointerleave', leave);
    window.addEventListener('scroll', function () { tip.hidden = true; }, { passive: true });
    var rt = 0;
    window.addEventListener('resize', function () { clearTimeout(rt); rt = setTimeout(redraw, 120); });
    $('wp-trace-play').addEventListener('click', play);
    if (!reduced) onView(cv, play);

    // per-token split (Fig. 1b): Llama3-8B, both machines
    var sp = $('wp-split');
    if (sp) {
      var max = 0, rows = [];
      ['icl', 'emr'].forEach(function (p) {
        ['row', 'col'].forEach(function (v) {
          var s = D.split[p][v], tot = s.work + s.late + s.leader + s.latency;
          max = Math.max(max, tot);
          rows.push({ p: p, v: v, s: s, tot: tot });
        });
      });
      sp.innerHTML = rows.map(function (r, i) {
        var w = function (x) { return (100 * x / max).toFixed(2) + '%'; }, wait = r.tot - r.s.work;
        return (i % 2 === 0 ? '<div class="g">' + PLATS[r.p] + ', ' + THREADS[r.p] + ' threads</div>' : '') +
          '<div class="row"><b>' + NAMES[r.v] + '</b><div class="bar" title="' + fmt2(r.tot) + ' ms per token">' +
          '<i style="width:' + w(r.s.work) + ';background:#0072B2"></i><i style="width:' + w(r.s.late) + ';background:#D4D4D4"></i>' +
          '<i style="width:' + w(r.s.leader) + ';background:#9C9C9C"></i><i style="width:' + w(r.s.latency) + ';background:#4A4A4A"></i></div>' +
          '<span class="val">' + fmt2(r.tot) + ' ms · <b>' + Math.round(100 * wait / r.tot) + '%</b> waiting</span></div>';
      }).join('');
    }
  })();

  /* ---------- 3. explorer: ladder, bandwidth, scaling over nodes, context depth ---------- */
  (function () {
    var host = $('wp-explore');
    if (!host) return;
    var esc = function (s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;'); };
    var STEPS = [['orig', 'ik', 'all cores', 'var(--c-ik)'], ['mig', '+ page migration', '', 'var(--c-step)'],
      ['hb', '+ two-level barrier', '', 'var(--c-step)'], ['fa', '+ split-KV attention', '', 'var(--c-step)'],
      ['a2', '+ AVX-512 quantizer', '= ik-P', 'var(--c-ikp)'], ['a3', 'Row', 'engine, row-owned', 'var(--c-row)'],
      ['a3col', 'Col', 'engine, column-split', 'var(--c-col)']];
    function ladder() {
      var L = D.ladder[state.plat][state.model], max = L.tok.a3col;
      $('wp-ladder').innerHTML = STEPS.map(function (s) {
        var v = s[0], r = v === 'orig' ? '1.00×' : fmt2(L.ratio[v]) + '×';
        return '<div class="r' + (v === 'a3col' ? ' hl' : '') + '"><span class="n"><b>' + s[1] + '</b>' + (s[2] ? ' <small>' + s[2] + '</small>' : '') + '</span>' +
          '<span class="b"><i style="width:' + (100 * L.tok[v] / max).toFixed(1) + '%;background:' + s[3] + '"></i></span>' +
          '<span class="v"><b>' + fmt1(L.tok[v]) + '</b> tok/s<em>' + r + '</em></span></div>';
      }).join('');
    }
    function bandwidth() {
      var B = D.bw[state.plat][state.model], any = false;
      $('wp-bw').innerHTML = IMPL.map(function (m) {
        var w = B.weights[m[0]], d = B.dram[m[0]];
        if (d !== undefined) any = true;
        return '<div class="r' + (m[0] === 'a3col' ? ' hl' : '') + '"><span class="n"><b>' + m[1] + '</b></span>' +
          '<span class="b full"><i style="width:' + w + '%;background:' + m[2] + '"></i>' +
          (d !== undefined ? '<u style="left:' + d + '%" title="measured DRAM reads: ' + d + '%"></u>' : '') + '</span>' +
          '<span class="v"><b>' + fmt1(w) + '%</b></span></div>';
      }).join('') + '<div class="note">Bar: decode speed × weight and KV bytes per token, as a share of the machine\'s ' + BW[state.plat] +
        ' GB/s local read bandwidth.' + (any ? ' Tick: the same from DRAM reads counted by the memory controllers (separate batch).' : '') +
        (state.plat === 'emr' && state.model === 'falcon3_1b' ? ' This model\'s 482 MiB of weights sit in the 260 MiB-per-socket L3 caches, so almost nothing is read from DRAM.' : '') + '</div>';
    }
    function lineChart(id, o) {
      var W = 340, H = 200, l = 38, r = 10, t = 10, b = 34, max = 0;
      o.series.forEach(function (s) { s.ys.forEach(function (y) { max = Math.max(max, y); }); });
      var step = [1, 2, 5, 10, 20, 50, 100, 200, 250, 500].filter(function (k) { return max / k <= 5; })[0] || 500;
      var top = Math.ceil(max / step) * step;
      var X = function (i) { return l + (W - l - r) * o.pos[i]; }, Y = function (y) { return t + (H - t - b) * (1 - y / top); };
      var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" role="img" aria-label="' + esc(o.label) + '">';
      for (var g = 0; g <= top; g += step) s += '<line class="gl" x1="' + l + '" x2="' + (W - r) + '" y1="' + Y(g) + '" y2="' + Y(g) + '"/><text class="tk" text-anchor="end" x="' + (l - 5) + '" y="' + (Y(g) + 3.5) + '">' + g + '</text>';
      o.xt.forEach(function (x, i) { s += '<text class="tk" text-anchor="middle" x="' + X(i) + '" y="' + (H - b + 14) + '">' + x + '</text>'; });
      s += '<text class="al" text-anchor="middle" x="' + (l + (W - l - r) / 2) + '" y="' + (H - 4) + '">' + esc(o.xlab) + '</text>';
      o.series.forEach(function (q) {
        s += '<polyline fill="none" stroke="' + q.color + '" stroke-width="2"' + (q.dash ? ' stroke-dasharray="5 3"' : '') + ' points="' +
          q.ys.map(function (y, i) { return X(i).toFixed(1) + ',' + Y(y).toFixed(1); }).join(' ') + '"/>';
        q.ys.forEach(function (y, i) {
          s += '<circle cx="' + X(i).toFixed(1) + '" cy="' + Y(y).toFixed(1) + '" r="3.2" fill="' + q.color + '" stroke="var(--card)" stroke-width="1"><title>' +
            esc(q.name + ', ' + o.xt[i] + ' ' + o.unit + ': ' + fmt1(y) + ' tok/s') + '</title></circle>';
        });
      });
      $(id).innerHTML = s + '</svg><div class="lg">' + o.series.map(function (q) {
        return '<span><svg width="20" height="8" aria-hidden="true"><line x1="0" x2="20" y1="4" y2="4" stroke="' + q.color + '" stroke-width="2"' +
          (q.dash ? ' stroke-dasharray="5 3"' : '') + '/></svg>' + esc(q.name) + ' <b>' + q.note + '</b></span>';
      }).join('') + '</div>';
    }
    function scale() {
      var S = D.scale[state.plat][state.model];
      lineChart('wp-scale', { label: 'Decode speed with 1, 2 and 4 NUMA nodes', pos: [0, 1 / 3, 1], xt: ['1', '2', '4'], xlab: 'NUMA nodes used', unit: 'nodes',
        series: IMPL.map(function (m) {
          var ys = S[m[0]].map(function (p) { return p[0]; });
          return { name: m[1], color: m[2], ys: ys, note: fmt1(ys[2] / ys[0]) + '×' };
        }) });
    }
    function depth() {
      var S = D.depth[state.plat][state.model];
      var lines = [['orig', 'ik, default KV', 'var(--c-ik)', 0], ['orig-il', 'ik, interleaved KV', 'var(--c-ik)', 1], ['a2-il', 'ik-P, interleaved KV', 'var(--c-ikp)', 1], ['a3col', 'Col', 'var(--c-col)', 0]];
      lineChart('wp-depth', { label: 'Decode speed after contexts of 128 to 8192 tokens', pos: [0, 0.5, 5 / 6, 1], xt: ['128', '1k', '4k', '8k'], xlab: 'context depth (tokens, log scale)', unit: 'tokens',
        series: lines.map(function (m) { return { name: m[1], color: m[2], dash: m[3], ys: S[m[0]], note: fmt1(S[m[0]][3]) }; }) });
    }
    function readout() {
      var L = D.ladder[state.plat][state.model], B = D.bw[state.plat][state.model];
      $('wp-readout').innerHTML = '<strong>' + PLATS[state.plat] + ' · ' + MODELS[state.model] + '.</strong> Col decodes ' + fmt1(L.tok.a3col) + ' tok/s, ' +
        fmt2(L.ratio.a3col) + '× ik (' + fmt1(L.tok.orig) + ' tok/s), and reads at ' + fmt1(B.weights.a3col) + '% of the machine\'s bandwidth (ik: ' + fmt1(B.weights.orig) +
        '%). Each token reads ' + fmt2(L.gb) + ' GB of weights and KV cache.';
    }
    function all() { ladder(); bandwidth(); scale(); depth(); readout(); }
    all();
    listeners.push(all);
  })();

  syncChips();

  /* copy button of the BibTeX block */
  document.addEventListener('click', function (e) {
    var b = e.target.closest && e.target.closest('.wp-copy');
    if (!b) return;
    var text = $(b.getAttribute('data-target')).textContent;
    var done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1500); };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
  });
})();
