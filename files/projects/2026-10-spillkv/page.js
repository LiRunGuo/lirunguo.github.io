/* SpillKV project page: three interactive figures drawn from the measured runs in data.js
   (8 or 16 RepoQA sessions of 64K tokens on one RTX 5090, three arrival seeds; the paper's Table 1 and Figure 1b).
   Block comments only: the page may be served compressed onto one line. */
(function () {
  "use strict";
  var D = window.SKV_DATA;
  if (!D) { return; }
  var NS = "http://www.w3.org/2000/svg";

  var MODELS = {
    llama: { name: "Llama-3.1-8B", slots: 3, bf16: 1.6, fp8: 3.2 },
    q4b: { name: "Qwen3-4B", slots: 4, bf16: 2.3, fp8: 4.6 }
  };
  var NAMES = {
    bf16: "vLLM, bf16 KV",
    lmc: "vLLM, bf16 KV + LMCache",
    fp8: "vLLM, FP8 KV",
    fp8lmc: "vLLM, FP8 KV + LMCache",
    fp8lmc_ff: "vLLM, FP8 KV + LMCache",
    skv: "SpillKV",
    skv_ff: "SpillKV",
    skv1: "SpillKV, one slot"
  };

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag), k;
    for (k in attrs) { if (Object.prototype.hasOwnProperty.call(attrs, k)) { e.setAttribute(k, attrs[k]); } }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = el("text", { x: x, y: y, "class": cls || "", "text-anchor": anchor || "start" }, parent);
    t.textContent = s;
    return t;
  }
  function h(tag, cls, text, parent) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== null && text !== undefined) { e.textContent = text; }
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) { s += a[i]; } return s / a.length; }
  function median(a) {
    var b = a.slice().sort(function (x, y) { return x - y; }), m = b.length >> 1;
    return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2;
  }
  function f1(v) { return v < 1 ? v.toFixed(2) : v < 100 ? v.toFixed(1) : v.toFixed(0); }
  function fx(v) { return v.toFixed(1) + "×"; }

  /* segmented buttons: options [[value, label], ...] */
  function seg(box, options, value, onChange) {
    clear(box);
    var buttons = options.map(function (o) {
      var b = h("button", null, o[1], box);
      b.type = "button";
      b.setAttribute("aria-pressed", String(o[0] === value));
      b.addEventListener("click", function () {
        buttons.forEach(function (x) { x.setAttribute("aria-pressed", "false"); });
        b.setAttribute("aria-pressed", "true");
        onChange(o[0]);
      });
      return b;
    });
  }

  var tipEl = null;
  function tip(html, ev) {
    if (!tipEl) { tipEl = h("div", "skv-tip", null, document.body); }
    if (html === null) { tipEl.style.display = "none"; return; }
    tipEl.innerHTML = html;
    tipEl.style.display = "block";
    var x = ev.clientX + 14, y = ev.clientY + 14, w = tipEl.offsetWidth, hh = tipEl.offsetHeight;
    if (x + w > window.innerWidth - 8) { x = ev.clientX - w - 14; }
    if (y + hh > window.innerHeight - 8) { y = ev.clientY - hh - 14; }
    tipEl.style.left = x + "px";
    tipEl.style.top = y + "px";
  }

  /* ---------- Figure A: where 8 or 16 sessions live ---------- */
  function capacity() {
    var fig = document.getElementById("skv-cap");
    if (!fig) { return; }
    var st = { m: "llama", n: 8 };
    var box = fig.querySelector("[data-role=rows]");
    function chips(parent, n, on, kind) {
      var wrap = h("div", "skv-chips", null, parent), i, c, f;
      for (i = 0; i < n; i++) {
        c = h("span", "skv-chip", null, wrap);
        f = Math.max(0, Math.min(1, on - i));
        if (f >= 1) { c.className += " is-gpu"; }
        else if (f > 0) { c.className += " is-part " + kind; c.style.setProperty("--f", Math.round(f * 100) + "%"); }
        else { c.className += " " + kind; }
      }
    }
    function draw() {
      var M = MODELS[st.m], n = st.n;
      clear(box);
      [["vLLM, bf16 KV", M.bf16, "is-evict",
        M.bf16 + " of " + n + " contexts fit beside the weights. A follow-up whose context was evicted prefills all of it again."],
       ["vLLM, FP8 KV", M.fp8, "is-evict",
        M.fp8 + " contexts fit. More sessions than that are live, so evictions and recomputes remain."],
       ["SpillKV", M.slots, "is-ssd",
        M.slots + " sessions decode from 4-bit tiers. The other " + (n - M.slots) +
        " are parked on the SSD and hold no GPU memory; a follow-up reloads its context instead of recomputing it."]
      ].forEach(function (r) {
        var row = h("div", "skv-caprow" + (r[0] === "SpillKV" ? " is-ours" : ""), null, box);
        var lab = h("div", "skv-caplab", null, row);
        h("b", null, r[0], lab);
        chips(row, n, r[1], r[2]);
        h("p", "skv-capnote", r[3], row);
      });
    }
    seg(fig.querySelector("[data-role=model]"), [["llama", "Llama-3.1-8B"], ["q4b", "Qwen3-4B"]], st.m,
        function (v) { st.m = v; draw(); });
    seg(fig.querySelector("[data-role=n]"), [[8, "8 sessions"], [16, "16 sessions"]], st.n,
        function (v) { st.n = v; draw(); });
    draw();
  }

  /* ---------- Figure B: replay a measured run ---------- */
  function replay() {
    var fig = document.getElementById("skv-replay");
    if (!fig) { return; }
    var MODES = { arrival: ["bf16", "fp8", "fp8lmc", "skv"], ff: ["fp8lmc_ff", "skv_ff"] };
    var st = { m: "llama", n: 8, s: 0, mode: "arrival", t: 0 };
    var svg = fig.querySelector("[data-role=plot]");
    var slider = fig.querySelector("[data-role=time]");
    var tlab = fig.querySelector("[data-role=tlab]");
    var playBtn = fig.querySelector("[data-role=play]");
    var slotsBox = fig.querySelector("[data-role=slots]");
    var readout = fig.querySelector("[data-role=readout]");
    var W = 900, L = 4, R = 168, span = 1, panels = [], head = null, raf = 0, last = 0;

    function rows(k) { return D.replay[st.m + "/" + st.n + "/" + st.s + "/" + k]; }
    function endOf(r) { return r[2] + r[3] + r[4]; }
    function X(t) { return L + (W - L - R) * t / span; }
    function title(k) {
      var M = MODELS[st.m];
      if (k === "skv" || k === "skv_ff") {
        return "SpillKV, " + M.slots + " slots" + (k === "skv_ff" ? ", follow-ups first" : "") + ", the other sessions parked on the SSD";
      }
      if (k === "bf16") { return "vLLM, bf16 KV: " + M.bf16 + " contexts fit on the GPU"; }
      if (k === "fp8") { return "vLLM, FP8 KV: " + M.fp8 + " contexts fit"; }
      if (k === "fp8lmc") { return "vLLM, FP8 KV + LMCache: evicted contexts reload from the SSD"; }
      return "vLLM, FP8 KV + LMCache, follow-ups first (priority scheduling)";
    }
    function stopPlay() {
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
      playBtn.textContent = "▶︎ Play";
    }

    function build() {
      stopPlay();
      clear(svg);
      span = 0;
      MODES.arrival.concat(MODES.ff).forEach(function (k) {
        rows(k).forEach(function (r) { span = Math.max(span, endOf(r)); });
      });
      span = Math.ceil(span * 1.02 / 50) * 50;
      var keys = MODES[st.mode], rowH = st.n === 8 ? 11 : 7, y = 2, M = MODELS[st.m];
      panels = keys.map(function (k) {
        var data = rows(k), sids = [], p, ends = [], waits = [];
        data.forEach(function (r) { if (sids.indexOf(r[0]) < 0) { sids.push(r[0]); } });
        sids.sort(function (a, b) { return a - b; });
        p = { k: k, data: data, segs: [], y0: y, slots: (k === "skv" || k === "skv_ff") ? M.slots : 0 };
        p.title = txt(svg, L, y + 11, title(k), "skv-ptitle");
        p.status = el("tspan", { "class": "skv-pstatus" }, p.title);
        y += 17;
        sids.forEach(function (sid, i) {
          var mine = data.filter(function (r) { return r[0] === sid; });
          var lo = Math.min.apply(null, mine.map(function (r) { return r[2]; }));
          var hi = Math.max.apply(null, mine.map(endOf));
          el("line", { x1: X(lo), x2: X(hi), y1: y + i * rowH + rowH * 0.4, y2: y + i * rowH + rowH * 0.4, "class": "skv-life" }, svg);
        });
        data.forEach(function (r) {
          var i = sids.indexOf(r[0]), yy = y + i * rowH, a = r[2], ttft = r[3], dec = r[4], wait = r[6], parts = [];
          if (r[1] === 0) { parts.push([a, a + ttft, "first"]); }
          else if (wait >= 0) { parts.push([a, a + wait, "fu"], [a + wait, a + ttft, "load"]); }
          else { parts.push([a, a + ttft, "fu"]); }
          parts.push([a + ttft, a + ttft + dec, "dec"]);
          parts.forEach(function (q) {
            var rect = el("rect", { x: X(q[0]), y: yy, height: rowH * 0.8, width: 0, "class": "skv-b skv-b--" + q[2] }, svg);
            rect.addEventListener("mousemove", function (ev) { tip(describe(k, r), ev); });
            rect.addEventListener("mouseleave", function () { tip(null); });
            p.segs.push({ t0: q[0], t1: q[1], el: rect, min: q[2] === "load" ? 1.5 : 0 });
          });
          ends.push(endOf(r));
          if (r[1] > 0) { waits.push(ttft); }
        });
        p.end = Math.max.apply(null, ends);
        p.wait = mean(waits);
        var yMid = y + (sids.length * rowH) / 2;
        el("line", { x1: X(p.end), x2: X(p.end), y1: y - 2, y2: y + sids.length * rowH, "class": "skv-endline" }, svg);
        txt(svg, X(p.end) + 6, yMid - 3, "mean follow-up wait: " + f1(p.wait) + " s", "skv-ann");
        txt(svg, X(p.end) + 6, yMid + 11, "last session done: " + p.end.toFixed(0) + " s", "skv-ann skv-ann--muted");
        y += sids.length * rowH + 16;
        return p;
      });
      var axisY = y - 6, step = span > 1000 ? 200 : 100, t;
      el("line", { x1: L, x2: X(span), y1: axisY, y2: axisY, "class": "skv-axis" }, svg);
      for (t = 0; t <= span; t += step) {
        el("line", { x1: X(t), x2: X(t), y1: axisY, y2: axisY + 4, "class": "skv-axis" }, svg);
        txt(svg, X(t), axisY + 16, String(t), "skv-tick", "middle");
      }
      txt(svg, X(span / 2), axisY + 31, "time since the first session arrived (s)", "skv-tick", "middle");
      head = el("line", { x1: 0, x2: 0, y1: 0, y2: axisY, "class": "skv-head" }, svg);
      svg.setAttribute("viewBox", "0 0 " + W + " " + (axisY + 38));
      st.t = span;
      slider.value = 1000;
      update();
      summary();
    }

    function describe(k, r) {
      var sys = k.indexOf("skv") === 0 ? "SpillKV" : NAMES[k];
      var s = "<b>" + sys + "</b> · session " + (r[0] + 1) + ", " + (r[1] === 0 ? "first turn" : "follow-up " + r[1]) +
        "<br>arrives at " + r[2].toFixed(1) + " s<br>first token after " + f1(r[3]) + " s";
      if (r[1] === 0) { s += " (prefills its whole context)"; }
      else if (r[6] >= 0) { s += "<br> " + f1(r[6]) + " s waiting for a free slot<br> " + f1(r[3] - r[6]) + " s reload and prefill"; }
      s += "<br>256 tokens in " + r[4].toFixed(1) + " s";
      return s;
    }

    function stateAt(p, t) {
      var dec = 0, wait = 0, done = 0, bySid = {};
      p.data.forEach(function (r) { (bySid[r[0]] = bySid[r[0]] || []).push(r); });
      Object.keys(bySid).forEach(function (sid) {
        var turns = bySid[sid], i, r, e;
        for (i = 0; i < turns.length; i++) {
          r = turns[i]; e = endOf(r);
          if (t >= r[2] && t < e) { if (t < r[2] + r[3]) { wait++; } else { dec++; } return; }
        }
        if (t >= Math.max.apply(null, turns.map(endOf))) { done++; }
      });
      return { dec: dec, wait: wait, done: done, n: Object.keys(bySid).length };
    }

    function slotsAt(p, t) {
      clear(slotsBox);
      if (!p) { slotsBox.style.display = "none"; return; }
      slotsBox.style.display = "";
      h("span", "skv-slots__lab", "SpillKV's GPU slots at " + t.toFixed(0) + " s", slotsBox);
      var inSlot = {}, i, r, j, c, started = {}, done = {}, lastEnd = {};
      p.data.forEach(function (q) { lastEnd[q[0]] = Math.max(lastEnd[q[0]] || 0, endOf(q)); });
      for (i = 0; i < p.slots; i++) {
        c = h("span", "skv-slot", null, slotsBox);
        r = null;
        for (j = 0; j < p.data.length; j++) {
          var q = p.data[j];
          if (q[5] === i && t >= q[2] + q[6] && t < endOf(q)) { r = q; break; }
        }
        if (!r) { c.className += " is-free"; c.textContent = "slot " + (i + 1) + ": free"; continue; }
        inSlot[r[0]] = 1;
        var phase = t < r[2] + r[3] ? (r[1] === 0 ? "prefill" : "reload") : "decoding";
        c.className += " is-" + (phase === "decoding" ? "dec" : phase === "reload" ? "load" : "first");
        c.textContent = "slot " + (i + 1) + ": session " + (r[0] + 1) + ", " + phase;
      }
      p.data.forEach(function (q) {
        if (q[1] === 0 && t >= q[2] + q[3]) { started[q[0]] = 1; }
        if (t >= lastEnd[q[0]]) { done[q[0]] = 1; }
      });
      var parked = Object.keys(started).filter(function (sid) { return !inSlot[sid] && !done[sid]; }).length;
      h("span", "skv-slots__park", parked + (parked === 1 ? " session" : " sessions") + " parked on the SSD", slotsBox);
    }

    function update() {
      var t = st.t;
      panels.forEach(function (p) {
        p.segs.forEach(function (s) {
          var w = Math.max(0, Math.min(t, s.t1) - s.t0);
          var px = w > 0 ? X(s.t0 + w) - X(s.t0) : 0;
          if (s.min && t >= s.t1) { px = Math.max(px, s.min); }
          s.el.setAttribute("width", px);
        });
        var c = stateAt(p, t);
        p.status.textContent = t >= span - 1e-6 ? "" :
          "  ·  at " + t.toFixed(0) + " s: " + c.dec + " decoding, " + c.wait + " waiting, " + c.done + " of " + c.n + " done";
      });
      head.setAttribute("x1", X(t));
      head.setAttribute("x2", X(t));
      head.style.display = t >= span - 1e-6 ? "none" : "";
      tlab.textContent = "t = " + t.toFixed(0) + " s";
      var sp = panels.filter(function (p) { return p.slots > 0; })[0];
      slotsAt(sp, t);
    }

    function summary() {
      var M = MODELS[st.m], p = panels, sk = p.filter(function (q) { return q.slots > 0; })[0], s, fu, w, pre, others;
      fu = sk.data.filter(function (r) { return r[1] > 0; });
      w = fu.reduce(function (a, r) { return a + r[6]; }, 0) / fu.reduce(function (a, r) { return a + r[3]; }, 0);
      pre = median(fu.map(function (r) { return r[3] - r[6]; }));
      others = p.filter(function (q) { return q !== sk; }).map(function (q) {
        return NAMES[q.k] + " " + f1(q.wait) + " s";
      });
      s = M.name + ", " + st.n + " sessions, arrival seed " + st.s + ". SpillKV's follow-ups wait " + f1(sk.wait) +
        " s on average (" + others.join("; ") + "). " + Math.round(w * 100) +
        "% of that wait is spent waiting for a free slot; the reload and the follow-up's prefill take a median of " +
        pre.toFixed(2) + " s.";
      readout.textContent = s;
    }

    function frame(now) {
      var dt = (now - last) / 1000;
      last = now;
      st.t = Math.min(span, st.t + dt * span / 16);
      slider.value = Math.round(st.t / span * 1000);
      update();
      if (st.t < span) { raf = requestAnimationFrame(frame); } else { stopPlay(); }
    }
    playBtn.addEventListener("click", function () {
      if (raf) { stopPlay(); return; }
      if (st.t >= span - 1e-6) { st.t = 0; }
      playBtn.textContent = "❚❚ Pause";
      last = performance.now();
      raf = requestAnimationFrame(frame);
    });
    slider.addEventListener("input", function () { stopPlay(); st.t = slider.value / 1000 * span; update(); });
    svg.addEventListener("click", function (ev) {
      var pt = svg.createSVGPoint(), m = svg.getScreenCTM();
      if (!m) { return; }
      pt.x = ev.clientX; pt.y = ev.clientY;
      var x = pt.matrixTransform(m.inverse()).x;
      if (x < L || x > W - R) { return; }
      stopPlay();
      st.t = (x - L) / (W - L - R) * span;
      slider.value = Math.round(st.t / span * 1000);
      update();
    });

    seg(fig.querySelector("[data-role=model]"), [["llama", "Llama-3.1-8B"], ["q4b", "Qwen3-4B"]], st.m, function (v) { st.m = v; build(); });
    seg(fig.querySelector("[data-role=n]"), [[8, "8"], [16, "16"]], st.n, function (v) { st.n = v; build(); });
    seg(fig.querySelector("[data-role=seed]"), [[0, "0"], [1, "1"], [2, "2"]], st.s, function (v) { st.s = v; build(); });
    seg(fig.querySelector("[data-role=mode]"), [["arrival", "in arrival order"], ["ff", "follow-ups first"]], st.mode,
        function (v) { st.mode = v; build(); });
    build();
  }

  /* ---------- Figure C: every system over three seeds ---------- */
  function results() {
    var fig = document.getElementById("skv-results");
    if (!fig) { return; }
    var st = { m: "llama", n: 8, metric: "fu" };
    var svg = fig.querySelector("[data-role=plot]");
    var readout = fig.querySelector("[data-role=readout]");
    var ROWS = ["bf16", "lmc", "fp8", "fp8lmc", "skv1", "skv", "|", "fp8lmc_ff", "skv_ff"];
    var MET = {
      fu: { i: 0, unit: "s", axis: "mean wait for a follow-up's first token (s)" },
      first: { i: 3, unit: "s", axis: "median wait for a first turn's first token (s)" },
      tpot: { i: 1, unit: "ms", axis: "mean time per output token (ms)" },
      wall: { i: 2, unit: "s", axis: "time until the last session ends (s)" }
    };
    function vals(k) { return D.summary[st.m + "/" + st.n + "/" + k].map(function (r) { return r[MET[st.metric].i]; }); }
    function m(k) { return mean(vals(k)); }
    function label(k) {
      if (k === "skv") { return "SpillKV, " + MODELS[st.m].slots + " slots"; }
      if (k === "skv_ff") { return "SpillKV, " + MODELS[st.m].slots + " slots"; }
      if (k === "fp8lmc_ff") { return "vLLM, FP8 KV + LMCache (priority)"; }
      return NAMES[k];
    }
    function draw() {
      clear(svg);
      var W = 900, LW = 236, RW = 64, rowH = 26, y = 6, max = 0;
      ROWS.forEach(function (k) { if (k !== "|") { max = Math.max(max, Math.max.apply(null, vals(k))); } });
      var X = function (v) { return LW + (W - LW - RW) * v / max; };
      ROWS.forEach(function (k) {
        if (k === "|") {
          y += 8;
          txt(svg, 0, y + 12, "Both admit arrived follow-ups ahead of first turns", "skv-sep");
          el("line", { x1: 0, x2: W - RW, y1: y + 18, y2: y + 18, "class": "skv-sepline" }, svg);
          y += 26;
          return;
        }
        var ours = k.indexOf("skv") === 0, v = vals(k), mu = mean(v);
        txt(svg, LW - 10, y + rowH * 0.62, label(k), ours ? "skv-rl skv-rl--ours" : "skv-rl", "end");
        var bar = el("rect", { x: LW, y: y + 4, width: Math.max(1, X(mu) - LW), height: rowH - 9, rx: 3,
                               "class": ours ? "skv-rb skv-rb--ours" : "skv-rb" }, svg);
        v.forEach(function (sv, i) {
          el("circle", { cx: X(sv), cy: y + rowH / 2 - 0.5, r: 3.2, "class": ours ? "skv-dot skv-dot--ours" : "skv-dot" }, svg);
        });
        txt(svg, W - 4, y + rowH * 0.62, f1(mu), ours ? "skv-rv skv-rv--ours" : "skv-rv", "end");
        bar.addEventListener("mousemove", function (ev) {
          tip("<b>" + label(k) + "</b><br>mean " + f1(mu) + " " + MET[st.metric].unit + "<br>per seed: " +
              v.map(function (x) { return f1(x); }).join(", "), ev);
        });
        bar.addEventListener("mouseleave", function () { tip(null); });
        y += rowH;
      });
      el("line", { x1: LW, x2: W - RW, y1: y + 2, y2: y + 2, "class": "skv-axis" }, svg);
      var step = niceStep(max), t;
      for (t = 0; t <= max + 1e-9; t += step) {
        el("line", { x1: X(t), x2: X(t), y1: y + 2, y2: y + 6, "class": "skv-axis" }, svg);
        txt(svg, X(t), y + 19, String(Math.round(t * 100) / 100), "skv-tick", "middle");
      }
      txt(svg, LW + (W - RW - LW) / 2, y + 36, MET[st.metric].axis, "skv-tick", "middle");
      svg.setAttribute("viewBox", "0 0 " + W + " " + (y + 44));
      readout.textContent = sentence();
    }
    function niceStep(max) {
      var raw = max / 6, p = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10)), n = raw / p;
      return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * p;
    }
    function range(k) { var v = vals(k); return f1(Math.min.apply(null, v)) + "–" + f1(Math.max.apply(null, v)); }
    function sentence() {
      var M = MODELS[st.m];
      if (st.metric === "fu") {
        return "SpillKV's follow-ups wait " + f1(m("skv")) + " s (per seed " + range("skv") + " s), " +
          fx(m("bf16") / m("skv")) + " less than with vLLM in bf16, " + fx(m("fp8") / m("skv")) + " less than with FP8 KV and " +
          fx(m("fp8lmc") / m("skv")) + " less than with FP8 KV and LMCache. When both admit follow-ups first, vLLM with FP8 KV and LMCache waits " +
          fx(m("fp8lmc_ff") / m("skv_ff")) + " as long as SpillKV (" + f1(m("fp8lmc_ff")) + " against " + f1(m("skv_ff")) + " s).";
      }
      if (st.metric === "first") {
        return "Admitting follow-ups first makes first turns wait longer. SpillKV's median first turn goes from " + f1(m("skv")) + " to " +
          f1(m("skv_ff")) + " s, and vLLM's with FP8 KV and LMCache from " + f1(m("fp8lmc")) + " to " + f1(m("fp8lmc_ff")) + " s.";
      }
      if (st.metric === "tpot") {
        return "SpillKV decodes its " + M.slots + " slots in one batch, so a token takes " + f1(m("skv")) + " ms, against " +
          f1(m("bf16")) + " ms for vLLM in bf16 and " + f1(m("fp8lmc")) + " ms with FP8 KV and LMCache. With one slot it takes " +
          f1(m("skv1")) + " ms.";
      }
      return "The last session ends after " + f1(m("skv")) + " s with SpillKV, " + f1(m("bf16")) + " s with vLLM in bf16 and " +
        f1(m("fp8lmc")) + " s with FP8 KV and LMCache.";
    }
    seg(fig.querySelector("[data-role=model]"), [["llama", "Llama-3.1-8B"], ["q4b", "Qwen3-4B"]], st.m, function (v) { st.m = v; draw(); });
    seg(fig.querySelector("[data-role=n]"), [[8, "8"], [16, "16"]], st.n, function (v) { st.n = v; draw(); });
    seg(fig.querySelector("[data-role=metric]"),
        [["fu", "follow-up wait"], ["first", "first-turn wait"], ["tpot", "time per token"], ["wall", "wall time"]], st.metric,
        function (v) { st.metric = v; draw(); });
    draw();
  }

  function init() { capacity(); replay(); results(); }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); } else { init(); }
})();
