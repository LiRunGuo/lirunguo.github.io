/* Interactive figures of the StudentProbe project page: where a student can skip, what bounds the speedup, and
   side-by-side video clips. All numbers come from the paper (Tables 3-4, Fig. 4). */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var BASE = "/files/projects/2026-11-gmsa-distillation/demo/";

  function svgEl(tag, attrs, parent, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    parent.appendChild(n);
    return n;
  }
  function pills(box, items, current, pick) {
    box.innerHTML = "";
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button"; b.className = "sp-pill"; b.textContent = it.label;
      b.setAttribute("aria-pressed", it.key === current);
      b.onclick = function () { pick(it.key); };
      box.appendChild(b);
    });
  }

  /* 1. where a student can skip: development prompts (cost, aesthetic change) and FBCache's signal (Fig. 4); held out: Table 3 */
  var SKIP = {
    anyflow: { label: "AnyFlow", cost: [.210, .078, .074], aes: ["-.04", "-.01", "-.10*"], fb: [.175, .1503, .1728], sp: 2,
      note: "AnyFlow's lowest-drift skip, call 3, is the only one that lowers aesthetics, so StudentProbe's quality check moves it to call 2. FBCache picks call 2 in 57 of 66 held-out videos and still lowers aesthetics (−0.058); TeaCache skips call 1 at almost three times the drift." },
    rcm: { label: "rCM", cost: [.359, .283, .575], aes: ["+.01", "+.08*", "-2.00*"], fb: [.7596, .6682, .5511], sp: 2,
      note: "FBCache's signal is lowest at call 3, rCM's costliest skip: its sampler re-noises before every call, so reusing the last call carries old noise into the video (held out, aesthetics −0.555 vs −0.054 for StudentProbe's call 2)." },
    klein: { label: "FLUX.2-klein", cost: [.225, .106, .092], aes: ["-.04", "+.01", "+.05"], fb: [.7811, .6659, .5983], sp: 3,
      note: "On the image student the cache signal follows the fidelity cost, and FBCache's choice agrees with the measured one." },
    ltx: { label: "LTX-Video (held out)", cost: [.276, .143, .132, .079, .024, .009], aes: ["+.02", "+.01", "+.01", "-.02", "+.09*", "+.08*"],
      fb: [.5084, .3876, .2998, 1.1942, 1.0129, .6367], sp: 6,
      note: "FBCache's signal is largest exactly where skipping costs least. On held-out prompts its pick, call 3, drifts 13× as far as StudentProbe's call 6 (LPIPS 0.117 vs 0.009), which even raises aesthetics." } };
  var skipCur = "ltx";

  function drawSkip() {
    var d = SKIP[skipCur], svg = document.getElementById("sp-skip-svg");
    pills(document.getElementById("sp-skip-pills"), Object.keys(SKIP).map(function (k) { return { key: k, label: SKIP[k].label }; }),
      skipCur, function (k) { skipCur = k; drawSkip(); });
    svg.innerHTML = "";
    var W = svg.getBoundingClientRect().width < 440 ? 400 : 600;
    svg.setAttribute("viewBox", "0 0 " + W + " 284");
    var L = 52, R = W - 6, T1 = 14, H1 = 160, T2 = 196, H2 = 46, n = d.cost.length, bw = (R - L) / n;
    var fbPick = d.fb.indexOf(Math.min.apply(null, d.fb)) + 1;
    var ymax = Math.max.apply(null, d.cost) * 1.2, y = function (v) { return T1 + H1 - v / ymax * H1; };
    var smax = Math.max.apply(null, d.fb) * 1.25, ys = function (v) { return T2 + H2 - v / smax * H2; };
    var step = [0.05, 0.1, 0.2, 0.25, 0.5, 1].filter(function (s) { return ymax / s <= 5; })[0];
    for (var v = 0; v <= ymax + 1e-9; v += step) {
      svgEl("line", { x1: L, x2: R, y1: y(v), y2: y(v), "class": "sp-grid" }, svg);
      svgEl("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end", "class": "sp-tick" }, svg, v.toFixed(step < 0.1 ? 2 : 1));
    }
    svgEl("rect", { x: L, y: T2 - 6, width: R - L, height: H2 + 12, rx: 4, "class": "sp-strip" }, svg);
    d.cost.forEach(function (c, i) {
      var k = i + 1, x = L + i * bw, bwid = Math.min(bw * 0.56, 54);
      var drop = d.aes[i].charAt(0) === "-" && d.aes[i].slice(-1) === "*";
      var r = svgEl("rect", { x: x + (bw - bwid) / 2, y: y(c), width: bwid, height: T1 + H1 - y(c), rx: 3,
        "class": "sp-bar" + (drop ? " sp-drop" : "") }, svg);
      svgEl("title", {}, r, "Skip call " + k + ": cost (LPIPS) " + c.toFixed(3) + ", Δ aesthetics " + d.aes[i].replace("-", "−") +
        (d.aes[i].slice(-1) === "*" ? " (95% CI excludes 0)" : "") + ", FBCache signal " + d.fb[i].toFixed(3));
      svgEl("text", { x: x + bw / 2, y: T2 + H2 + 22, "text-anchor": "middle", "class": "sp-label" }, svg, k);
      var top = y(c) - 6, cx = Math.min(x + bw / 2, R - 40);            /* keep the label inside the plot on narrow screens */
      if (k === d.sp) { svgEl("text", { x: cx, y: top, "text-anchor": "middle", "class": "sp-mark-sp" }, svg, "StudentProbe"); top -= 16; }
      if (k === fbPick) svgEl("text", { x: cx, y: top, "text-anchor": "middle", "class": "sp-mark-fb" }, svg, "FBCache");
    });
    svgEl("text", { x: (L + R) / 2, y: T2 + H2 + 40, "text-anchor": "middle", "class": "sp-tick" }, svg, "skipped call");
    svgEl("text", { x: L + 6, y: T2 + 8, "class": "sp-tick" }, svg, "FBCache signal");
    svgEl("text", { x: 11, y: T1 + H1 / 2, transform: "rotate(-90 11 " + (T1 + H1 / 2) + ")", "text-anchor": "middle", "class": "sp-tick" }, svg, "LPIPS");
    var pts = d.fb.map(function (v, i) { return [L + (i + 0.5) * bw, ys(v)]; });
    svgEl("polyline", { points: pts.map(function (p) { return p.join(","); }).join(" "), "class": "sp-line" }, svg);
    pts.forEach(function (p, i) {
      var dot = svgEl("circle", { cx: p[0], cy: p[1], r: i + 1 === fbPick ? 5.5 : 4, "class": "sp-dot" + (i + 1 === fbPick ? " sp-pick" : "") }, svg);
      svgEl("title", {}, dot, "FBCache signal at call " + (i + 1) + ": " + d.fb[i].toFixed(3));
    });
    document.getElementById("sp-skip-note").textContent = d.note;
  }

  /* 2. what bounds the speedup: measured latency split on one PRO6000 BW, seconds per video (Table 4) */
  var SPEED = {
    wan22: { label: "Wan2.2 A14B", base: [24.746, 4.493, .413], rec: [10.485, .137, .416] },
    anyflow: { label: "AnyFlow", base: [5.624, 2.899, .372], rec: [2.047, .136, .394] },
    rcm: { label: "rCM", base: [4.469, 2.897, .289], rec: [1.862, .137, .279] },
    rcm14: { label: "rCM-14B (held out)", base: [24.308, 2.903, .292], rec: [10.497, .137, .259] } };
  var COL = ["#2a78d6", "#eb6834", "#b9c4d0"], NAMES = ["DiT", "VAE decoder", "text encoder and other"];
  var speedCur = "anyflow";

  function drawSpeed() {
    var d = SPEED[speedCur], s = +document.getElementById("sp-dit").value, light = document.getElementById("sp-light").checked;
    pills(document.getElementById("sp-speed-pills"), Object.keys(SPEED).map(function (k) { return { key: k, label: SPEED[k].label }; }),
      speedCur, function (k) { speedCur = k; drawSpeed(); });
    var b = d.base, e0 = b[0] + b[1] + b[2];
    var now = [b[0] / s, light ? d.rec[1] : b[1], b[2]], e1 = now[0] + now[1] + now[2];
    [[b, "sp-lane-0"], [now, "sp-lane-1"]].forEach(function (row) {
      var lane = document.getElementById(row[1]), parts = lane.querySelectorAll(".sp-stack span");
      row[0].forEach(function (t, j) {
        parts[j].style.width = (t / e0 * 100) + "%"; parts[j].style.background = COL[j];
        parts[j].title = NAMES[j] + ": " + t.toFixed(2) + " s";
      });
      lane.querySelector("b").textContent = (row[0][0] + row[0][1] + row[0][2]).toFixed(2) + " s";
    });
    document.getElementById("sp-dit-v").textContent = s.toFixed(2) + "×";
    var cap = e0 / ((light ? d.rec[1] : b[1]) + b[2]), recE = d.rec[0] + d.rec[1] + d.rec[2];
    document.getElementById("sp-speed-result").innerHTML = "End to end: <strong>" + (e0 / e1).toFixed(2) + "×</strong>. " +
      "Even an infinitely fast DiT would stop at " + cap.toFixed(2) + "×. Measured StudentProbe-guided baseline (two calls, lightweight decoder, compilation): " +
      (e0 / recE).toFixed(2) + "×.";
  }

  /* 3. side-by-side clips: the released student against the guided baseline (held-out prompt with median drift), and
     skipped calls (first prompt of the supplementary video's sections) */
  var VBASE = "/files/projects/2026-11-gmsa-distillation/videos/";
  var CLIPS = {
    vb: [
      { key: "anyflow", label: "AnyFlow · 3.45×", file: "baseline-anyflow", left: "as released · 8.89 s", right: "guided baseline · 2.58 s",
        note: "Prompt: “indoor movie theater”." },
      { key: "rcm", label: "rCM · 3.36×", file: "baseline-rcm", left: "as released · 7.66 s", right: "guided baseline · 2.28 s",
        note: "Prompt: “A tranquil tableau of the phone booth was tucked away in a quiet alley”." },
      { key: "wan22", label: "Wan2.2 A14B · 2.69×", file: "baseline-wan22_4s", left: "as released · 29.65 s", right: "guided baseline · 11.04 s",
        note: "Prompt: “Yoda playing guitar on the stage”." } ],
    vs: [
      { key: "ltx", label: "LTX-Video: skip call 1", file: "skip-ltx", left: "as released", right: "call 1 skipped",
        note: "Skipping LTX-Video's first call gives a different video, far from the original in LPIPS, yet no quality score falls: drift is not damage." },
      { key: "rcm", label: "rCM: skip the last call", file: "skip-rcm", left: "call 2 skipped (StudentProbe)", right: "call 3 skipped (FBCache)",
        note: "rCM re-noises before every call, so reusing its last call carries old noise into the video. The cache signal is lowest exactly there." },
      { key: "anyflow", label: "AnyFlow: call 2 or call 3", file: "skip-anyflow", left: "call 2 skipped (StudentProbe)", right: "call 3 skipped",
        note: "The two look alike, and call 3 even drifts slightly less (LPIPS 0.074 vs 0.078). Over the held-out videos it lowers aesthetics by 0.10–0.15, which only a quality check catches." } ] };
  var clipCur = { vb: "anyflow", vs: "ltx" };

  function drawClips(id) {
    var list = CLIPS[id], c = list.filter(function (x) { return x.key === clipCur[id]; })[0];
    pills(document.getElementById("sp-" + id + "-pills"), list, clipCur[id], function (k) { clipCur[id] = k; drawClips(id); });
    var box = document.getElementById("sp-" + id), v = box.querySelector("video"), labels = box.querySelectorAll(".sp-video-labels span");
    if (v.getAttribute("data-file") !== c.file) {
      v.setAttribute("data-file", c.file);
      v.poster = VBASE + c.file + ".jpg"; v.src = VBASE + c.file + ".mp4";
      var p = v.play(); if (p && p.catch) p.catch(function () {});
    }
    labels[0].textContent = c.left; labels[1].textContent = c.right;
    document.getElementById("sp-" + id + "-note").textContent = c.note;
  }

  function init() {
    if (!document.getElementById("sp-skip-svg")) return;
    drawSkip();
    document.getElementById("sp-dit").oninput = drawSpeed;
    document.getElementById("sp-light").onchange = drawSpeed;
    document.getElementById("sp-preset").onclick = function () {
      var d = SPEED[speedCur];
      document.getElementById("sp-dit").value = (d.base[0] / d.rec[0]).toFixed(2);
      document.getElementById("sp-light").checked = true;
      drawSpeed();
    };
    drawSpeed();
    drawClips("vb"); drawClips("vs");
    var lastW = window.innerWidth, t;
    window.addEventListener("resize", function () {
      if (window.innerWidth === lastW) return;
      lastW = window.innerWidth; clearTimeout(t); t = setTimeout(drawSkip, 150);
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
