/* Interactive figures of the FP4 long-reasoning project page. Every number is the paper's (Figure 1b, Tables 1-3). */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function signed(v) { return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(1); }
  /* a segmented control: built once, so the pressed button keeps keyboard focus */
  function seg(box, items, cur, on) {
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = it.label; b.setAttribute("aria-pressed", it.v === cur);
      b.onclick = function () {
        box.querySelectorAll("button").forEach(function (o) { o.setAttribute("aria-pressed", o === b); });
        on(it.v);
      };
      box.appendChild(b);
    });
  }

  /* ---- Figure 1b, interactive: every Qwen3-8B W4A4 checkpoint against BF16, and the noise band ---- */
  var GROUPS = [
    { name: "chat GPTQ", color: "#2a78d6", shape: "circle" },
    { name: "trace GPTQ", color: "#eb6834", shape: "square" },
    { name: "RTN", color: "#1baf7a", shape: "up" },
    { name: "MR-GPTQ", color: "#b97f00", shape: "diamond" },
    { name: "released checkpoints", color: "#d6558a", shape: "down" }
  ];
  /* g: group, acc: pooled AIME 2024-2026 (%), d [lo, hi]: paired difference to BF16, x: response tokens / BF16 */
  var PTS = [
    { g: 0, name: "chat GPTQ, draw 1", acc: 59.2, d: -7.8, lo: -13.1, hi: -2.8, x: 1.10 },
    { g: 0, name: "chat GPTQ, draw 1, second sampling seed", acc: 66.1, d: -0.8, lo: -5.0, hi: 3.6, x: 1.11 },
    { g: 0, name: "chat GPTQ, draw 2", acc: 65.3, d: -1.7, lo: -6.9, hi: 3.3, x: 1.08, dodge: -1 },
    { g: 0, name: "chat GPTQ, draw 3", acc: 65.3, d: -1.7, lo: -6.9, hi: 3.3, x: 1.08, dodge: 1 },
    { g: 0, name: "chat GPTQ, draw 4", acc: 64.2, d: -2.8, lo: -7.5, hi: 1.7, x: 1.09 },
    { g: 1, name: "trace GPTQ, own traces, draw 1", acc: 66.4, d: -0.6, lo: -4.7, hi: 3.6, x: 1.02 },
    { g: 1, name: "trace GPTQ, own traces, draw 1, second sampling seed", acc: 66.1, d: -0.8, lo: -5.0, hi: 3.3, x: 1.03 },
    { g: 1, name: "trace GPTQ, own traces, draw 2", acc: 65.0, d: -1.9, lo: -6.7, hi: 2.5, x: 1.04 },
    { g: 1, name: "trace GPTQ, public DeepSeek-R1 traces", acc: 66.1, d: -0.8, lo: -5.6, hi: 3.9, x: 1.01 },
    { g: 2, name: "RTN, first sampling seed", acc: 65.3, d: -1.7, lo: -6.9, hi: 3.3, x: 1.12 },
    { g: 2, name: "RTN, second sampling seed", acc: 61.7, d: -5.3, lo: -10.3, hi: -0.6, x: 1.16 },
    { g: 3, name: "MR-GPTQ, FineWeb-Edu, draw 1", acc: 60.8, d: -6.1, lo: -11.7, hi: -0.8, x: 1.04, dodge: 1 },
    { g: 3, name: "MR-GPTQ, FineWeb-Edu, draw 2", acc: 62.2, d: -4.7, lo: -10.0, hi: 0.3, x: 1.04, dodge: 2.6 },
    { g: 3, name: "MR-GPTQ calibrated on our traces", acc: 65.0, d: -1.9, lo: -6.4, hi: 2.2, x: 1.02, dodge: 1 },
    { g: 4, name: "ISTA QAT (released)", acc: 58.9, d: -8.1, lo: -13.6, hi: -2.8, x: 1.05 },
    { g: 4, name: "NVIDIA ModelOpt (released)", acc: 60.0, d: -6.9, lo: -11.9, hi: -2.2, x: 1.15 }
  ];
  /* Table 1: 95% half-width of a paired comparison, by samples per problem n and problems P */
  var HW = { 4: { 30: 8.1, 90: 4.7, 180: 3.3 }, 16: { 30: 4.1, 90: 2.3, 180: 1.7 }, 64: { 30: 2.0, 90: 1.2, 180: 0.8 } };

  function marker(parent, shape, cx, cy, r, color) {
    var a = { fill: color, stroke: "#fff", "stroke-width": 1.2 };
    if (shape === "circle") { a.cx = cx; a.cy = cy; a.r = r; return el("circle", a, parent); }
    if (shape === "square") { a.x = cx - r * 0.9; a.y = cy - r * 0.9; a.width = r * 1.8; a.height = r * 1.8; return el("rect", a, parent); }
    var pts = shape === "up" ? [[0, -1.15], [1.1, 0.85], [-1.1, 0.85]] : shape === "down" ? [[0, 1.15], [1.1, -0.85], [-1.1, -0.85]]
      : [[0, -1.25], [1.1, 0], [0, 1.25], [-1.1, 0]];
    a.points = pts.map(function (p) { return (cx + p[0] * r) + "," + (cy + p[1] * r); }).join(" ");
    return el("polygon", a, parent);
  }

  function noiseFigure(root) {
    var svg = root.querySelector('[data-role="plot"]'), read = root.querySelector('[data-role="readout"]');
    var pick = root.querySelector('[data-role="pick"]');
    var n = 4, P = 90, sel = -1, nodes = [];
    var legend = root.querySelector('[data-role="legend"]');
    GROUPS.forEach(function (g) {
      var s = document.createElement("span"), i = document.createElementNS(NS, "svg");
      i.setAttribute("width", 14); i.setAttribute("height", 14); marker(i, g.shape, 7, 7, 5, g.color);
      s.appendChild(i); s.appendChild(document.createTextNode(g.name)); legend.appendChild(s);
    });
    seg(root.querySelector('[data-role="n"]'), [{ v: 4, label: "4" }, { v: 16, label: "16" }, { v: 64, label: "64" }], n, function (v) { n = v; draw(); });
    seg(root.querySelector('[data-role="P"]'), [{ v: 30, label: "30 (one AIME year)" }, { v: 90, label: "90 (three years)" }, { v: 180, label: "180" }], P, function (v) { P = v; draw(); });
    function draw() {
      var W = Math.max(300, svg.parentNode.clientWidth), narrow = W < 520, H = narrow ? 300 : 360;
      var m = { l: narrow ? 40 : 50, r: 12, t: 12, b: 40 };
      svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("height", H); svg.innerHTML = "";
      var x0 = 0.995, x1 = 1.172, y0 = -15, y1 = 9;
      function X(v) { return m.l + (v - x0) / (x1 - x0) * (W - m.l - m.r); }
      function Y(v) { return m.t + (y1 - v) / (y1 - y0) * (H - m.t - m.b); }
      var hw = HW[n][P];
      el("rect", { x: m.l, y: Y(Math.min(hw, y1)), width: W - m.l - m.r, height: Y(-hw) - Y(Math.min(hw, y1)), fill: "#e9e7e1" }, svg);
      [-15, -10, -5, 0, 5].forEach(function (t) {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), stroke: t === 0 ? "#52514e" : "#eceae5", "stroke-width": 1, "stroke-dasharray": t === 0 ? "5 4" : "" }, svg);
        el("text", { x: m.l - 7, y: Y(t) + 4, "text-anchor": "end" }, svg).textContent = (t > 0 ? "+" : t < 0 ? "−" : "") + Math.abs(t);
      });
      [1.0, 1.05, 1.1, 1.15].forEach(function (t) {
        el("line", { x1: X(t), x2: X(t), y1: H - m.b, y2: H - m.b + 5, stroke: "#8a8a86" }, svg);
        el("text", { x: X(t), y: H - m.b + 18, "text-anchor": "middle" }, svg).textContent = t.toFixed(2) + "×";
      });
      el("line", { x1: X(1), x2: X(1), y1: m.t, y2: H - m.b, stroke: "#52514e", "stroke-width": 1, "stroke-dasharray": "5 4" }, svg);
      el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "#8a8a86" }, svg);
      el("text", { x: (m.l + W - m.r) / 2, y: H - 5, "text-anchor": "middle" }, svg).textContent = "response length relative to BF16";
      var yl = el("text", { x: 12, y: (m.t + H - m.b) / 2, "text-anchor": "middle", transform: "rotate(-90 12 " + (m.t + H - m.b) / 2 + ")" }, svg);
      yl.textContent = narrow ? "AIME accuracy − BF16" : "AIME accuracy − BF16 (points)";
      var bl = el("text", { x: W - m.r - 6, y: Y(Math.min(hw, y1)) + 14, "text-anchor": "end", "class": "fp4-band" }, svg);
      bl.textContent = "noise: ±" + hw.toFixed(1);
      /* one checkpoint, two sampling seeds */
      var a = PTS[0], b = PTS[1];
      el("line", { x1: X(a.x), y1: Y(a.d), x2: X(b.x), y2: Y(b.d), stroke: GROUPS[0].color, "stroke-width": 1.6, "stroke-dasharray": "2 3" }, svg);
      var sl = el("text", { x: X(a.x) + 10, y: Y(-9.4), "class": "fp4-seed" }, svg);
      sl.textContent = narrow ? "+6.9" : "+6.9, new seed only";
      nodes = PTS.map(function (p, i) {
        var g = GROUPS[p.g], cx = X(p.x) + (p.dodge || 0) * 4;
        var grp = el("g", { tabindex: 0, role: "img", "aria-label": p.name + ": " + signed(p.d) + " points, " + p.x.toFixed(2) + " times the BF16 length", "class": "fp4-pt" }, svg);
        el("line", { x1: cx, x2: cx, y1: Y(p.hi), y2: Y(p.lo), stroke: g.color, "stroke-width": 1.3 }, grp);
        marker(grp, g.shape, cx, Y(p.d), 5.5, g.color);
        el("rect", { x: cx - 7, y: Y(p.hi), width: 14, height: Y(p.lo) - Y(p.hi), fill: "transparent" }, grp);
        function choose() { sel = i; mark(); }
        grp.addEventListener("pointerenter", choose); grp.addEventListener("click", choose); grp.addEventListener("focus", choose);
        return grp;
      });
      read.innerHTML = "With <b>" + n + "</b> samples per problem on <b>" + P + "</b> problems, a paired comparison resolves <b>\u00b1" + hw.toFixed(1) +
        " points</b> (95% half-width)." + (n === 4 && P === 90 ? " This is the setting of the runs plotted." :
          " The points and their intervals are the paper\u2019s four-sample runs on 90 problems; the band shows what this setting would resolve.");
      mark();
    }
    /* highlight the chosen checkpoint in place, so keyboard focus survives */
    function mark() {
      nodes.forEach(function (grp, i) {
        grp.setAttribute("opacity", sel < 0 || i === sel ? 1 : 0.4);
        grp.firstChild.setAttribute("stroke-width", i === sel ? 2.6 : 1.3);
      });
      if (sel >= 0) {
        var q = PTS[sel];
        pick.innerHTML = '<b style="color:' + GROUPS[q.g].color + '">' + q.name + "</b>: AIME " + q.acc.toFixed(1) + "%, " + signed(q.d) + " [" + signed(q.lo) + ", " +
          signed(q.hi) + "] points from BF16, " + q.x.toFixed(2) + "\u00d7 its response length.";
      } else pick.textContent = "Hover over a point, or tap it, to see the checkpoint.";
    }
    draw();
    var t; window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(draw, 120); });
  }

  /* ---- Table 3, interactive: memory, concurrent traces and correct answers per minute on the 32 GB RTX 5090 ---- */
  /* w, kv: GiB; K: cache size (k tokens); L: mean response (k tokens); acc: four-sample AIME accuracy on this card (%);
     c: concurrent traces C*; tp: predicted tokens/s; p / mm: predicted / measured correct answers per minute; err: p/mm - 1 (%) */
  var CAP = [
    { m: "Qwen3-8B", name: "BF16", fmt: "bf16", w: 15.3, kv: 12.0, K: 88, L: 16.6, acc: 66.9, c: 10, tp: 418, p: 1.01, mm: 0.97, err: 4 },
    { m: "Qwen3-8B", name: "FP8", fmt: "bf16", w: 9.1, kv: 17.8, K: 129, L: 17.0, acc: 64.4, c: 15, tp: 591, p: 1.34, mm: 1.38, err: -3 },
    { m: "Qwen3-8B", name: "FP8 + FP8 KV", fmt: "fp8", w: 9.1, kv: 17.9, K: 260, L: 16.8, acc: 67.2, c: 30, tp: 1070, p: 2.57, mm: 2.72, err: -5 },
    { m: "Qwen3-8B", name: "W4A16 trace", fmt: "bf16", w: 6.0, kv: 20.8, K: 151, L: 16.9, acc: 65.6, c: 18, tp: 684, p: 1.60, mm: 1.64, err: -3 },
    { m: "Qwen3-8B", name: "W4A4 trace", fmt: "bf16", w: 6.0, kv: 20.8, K: 152, L: 16.9, acc: 66.4, c: 18, tp: 683, p: 1.61, mm: 1.69, err: -5 },
    { m: "Qwen3-8B", name: "W4A4 trace + FP8 KV", fmt: "fp8", w: 6.0, kv: 20.4, K: 297, L: 17.1, acc: 65.0, c: 34, tp: 1188, p: 2.71, mm: 2.95, err: -8 },
    { m: "Qwen3-8B", name: "W4A4 trace + NVFP4 KV", fmt: "nvfp4", w: 6.0, kv: 20.4, K: 529, L: 17.9, acc: 62.5, c: 58, tp: 1733, p: 3.63, mm: 3.22, err: 13 },
    { m: "Qwen3-14B", name: "FP8", fmt: "bf16", w: 15.4, kv: 11.0, K: 72, L: 15.1, acc: 71.4, c: 9, tp: 394, p: 1.12, mm: 1.04, err: 7 },
    { m: "Qwen3-14B", name: "FP8 + FP8 KV", fmt: "fp8", w: 15.4, kv: 10.6, K: 139, L: 15.1, acc: 73.1, c: 18, tp: 713, p: 2.06, mm: 1.92, err: 7 },
    { m: "Qwen3-14B", name: "W4A4 trace", fmt: "bf16", w: 10.0, kv: 16.3, K: 107, L: 15.6, acc: 69.2, c: 13, tp: 547, p: 1.45, mm: 1.42, err: 2 },
    { m: "Qwen3-14B", name: "W4A4 trace + FP8 KV", fmt: "fp8", w: 10.0, kv: 15.9, K: 209, L: 15.4, acc: 71.7, c: 27, tp: 990, p: 2.76, mm: 2.82, err: -2 }
  ];
  var KVNAME = { bf16: "BF16", fp8: "FP8", nvfp4: "NVFP4" };

  function capFigure(root) {
    var list = root.querySelector('[data-role="rows"]'), read = root.querySelector('[data-role="readout"]');
    var sel = 5, CARD = 30, RMAX = 3.9;
    function draw() {
      var html = "", last = "";
      CAP.forEach(function (r, i) {
        if (r.m !== last) { html += '<div class="fp4-cap__model">' + r.m + "</div>"; last = r.m; }
        html += '<button type="button" class="fp4-cap__row' + (i === sel ? " is-on" : "") + '" data-i="' + i + '" aria-pressed="' + (i === sel) + '">' +
          '<span class="fp4-cap__name">' + r.name + "</span>" +
          '<span class="fp4-cap__mem" title="Weights ' + r.w.toFixed(1) + " GiB, KV cache " + r.kv.toFixed(1) + ' GiB">' +
          '<span class="fp4-cap__w" style="width:' + (100 * r.w / CARD) + '%">' + r.w.toFixed(0) + "</span>" +
          '<span class="fp4-cap__kv fp4-kv--' + r.fmt + '" style="width:' + (100 * r.kv / CARD) + '%">' + r.c + " traces</span></span>" +
          '<span class="fp4-cap__rate"><span class="fp4-cap__bar" style="width:' + (100 * r.mm / RMAX) + '%"></span>' +
          '<span class="fp4-cap__tick" style="left:' + (100 * r.p / RMAX) + '%"></span>' +
          '<span class="fp4-cap__val" style="left:' + (100 * Math.max(r.mm, r.p) / RMAX) + '%">' + r.mm.toFixed(2) + "</span></span></button>";
      });
      list.innerHTML = html;
      list.querySelectorAll(".fp4-cap__row").forEach(function (b) { b.onclick = function () { sel = +b.dataset.i; mark(); }; });
    }
    /* mark the chosen row in place, so it keeps keyboard focus */
    function mark() {
      list.querySelectorAll(".fp4-cap__row").forEach(function (b) {
        var on = +b.dataset.i === sel;
        b.classList.toggle("is-on", on); b.setAttribute("aria-pressed", on);
      });
      var r = CAP[sel], base = CAP[0];
      read.innerHTML = "<b>" + r.m + ", " + r.name + ".</b> " + r.w.toFixed(1) + " GiB of weights leave " + r.kv.toFixed(1) + " GiB for " + (r.fmt === "bf16" ? "a " : "an ") + KVNAME[r.fmt] +
        " cache of <b>" + r.K + "k tokens</b>. With responses of " + r.L.toFixed(1) + "k tokens it holds <b>" + r.c + " traces</b> at once. " +
        "The capacity model then gives " + r.tp.toLocaleString("en-US") + " tokens/s and, at " + r.acc.toFixed(1) + "% accuracy, <b>" + r.p.toFixed(2) +
        "</b> correct answers per minute; measured: <b>" + r.mm.toFixed(2) + "</b> (model " + (r.err > 0 ? "+" : "−") + Math.abs(r.err) + "%), " +
        (sel === 0 ? "the reference rate." : (r.mm / base.mm).toFixed(1) + "× the rate of 8B BF16.");
    }
    draw(); mark();
  }

  function start() {
    var a = document.getElementById("fp4-noise"), b = document.getElementById("fp4-capacity");
    if (a) noiseFigure(a);
    if (b) capFigure(b);
    document.addEventListener("click", function (e) {
      var c = e.target.closest && e.target.closest(".fp4-copy");
      if (!c) return;
      var text = document.getElementById(c.getAttribute("data-target")).textContent;
      var done = function () { c.textContent = "Copied"; setTimeout(function () { c.textContent = "Copy"; }, 1500); };
      if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
