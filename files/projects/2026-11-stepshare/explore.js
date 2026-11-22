/* Interactive figures of the StepShare project page: the arm explorer and the sample comparison.
   Numbers are the paper's (Table 2, supplementary Tables S2 and S3); colours and type come from the page's .ss styles. */
(function () {
"use strict";
const D = {"T2":{"Dense, w=1":{"n":5,"r1":null,"c1":null,"r4":null,"c4":null,"nll":0.0,"l28":-0.2,"worst":-0.5,"inw":-0.3},"Dense, 0.1":{"n":3,"r1":68.0,"c1":18.0,"r4":60.0,"c4":31.0,"nll":0.0,"l28":0.1,"worst":-2.8,"inw":-2.6},"Dense, 0.03":{"n":5,"r1":82.0,"c1":12.0,"r4":78.0,"c4":10.0,"nll":0.0,"l28":0.3,"worst":-5.4,"inw":-5.1},"Dense, 0.01":{"n":3,"r1":95.0,"c1":9.0,"r4":90.0,"c4":26.0,"nll":0.0,"l28":-1.4,"worst":-7.1,"inw":-6.4},"Task, w=1":{"n":5,"r1":60.0,"c1":22.0,"r4":29.0,"c4":48.0,"nll":0.0,"l28":0.4,"worst":0.1,"inw":-0.1},"Task, 0.1":{"n":3,"r1":80.0,"c1":17.0,"r4":73.0,"c4":19.0,"nll":0.0,"l28":0.7,"worst":-1.7,"inw":-1.1},"Task, 0.03":{"n":5,"r1":93.0,"c1":5.0,"r4":96.0,"c4":5.0,"nll":-0.01,"l28":-0.2,"worst":-2.9,"inw":-2.8},"Per-loss clipping, w=1":{"n":3,"r1":60.0,"c1":23.0,"r4":51.0,"c4":17.0,"nll":-0.01,"l28":0.2,"worst":-1.9,"inw":-2.1},"Per-task norm. grads, w=1":{"n":3,"r1":90.0,"c1":10.0,"r4":86.0,"c4":12.0,"nll":0.01,"l28":-0.8,"worst":-3.8,"inw":-3.7},"Per-task Adam, w=1":{"n":3,"r1":92.0,"c1":9.0,"r4":97.0,"c4":5.0,"nll":0.0,"l28":0.2,"worst":-3.6,"inw":-3.2},"Depth alloc. 1 / 0.03":{"n":3,"r1":72.0,"c1":28.0,"r4":61.0,"c4":8.0,"nll":-0.02,"l28":-0.2,"worst":-0.3,"inw":0.3},"Depth alloc. 0.3 / 0.03":{"n":3,"r1":84.0,"c1":52.0,"r4":75.0,"c4":6.0,"nll":-0.01,"l28":0.5,"worst":-0.2,"inw":-0.3},"Depth alloc. 0.03 / 1":{"n":3,"r1":61.0,"c1":23.0,"r4":44.0,"c4":45.0,"nll":0.01,"l28":-0.6,"worst":-6.9,"inw":-6.0},"StepShare-Adam, w=1":{"n":5,"r1":91.0,"c1":14.0,"r4":89.0,"c4":8.0,"nll":0.01,"l28":-0.1,"worst":-0.5,"inw":-0.4}},"COCO":{"Dense, w=1":[-0.1,0.3,-0.1,-0.2,-0.2,-0.2,-0.4,-0.5,-0.4,-0.2,-0.4,-0.1,-0.1,-0.2],"Dense, w=0.1":[-1.8,-1.9,-2.1,-2.8,-2.3,-1.9,-1.7,-1.3,-1.0,-0.7,-0.2,0.0,0.1,0.1],"Dense, w=0.03":[-2.4,-3.8,-5.0,-5.4,-4.9,-4.5,-3.7,-2.2,-1.4,-0.7,-0.1,0.2,0.1,0.3],"Dense, w=0.01":[-2.9,-4.8,-6.0,-7.1,-6.9,-6.3,-5.4,-4.1,-3.5,-2.8,-1.8,-1.9,-1.6,-1.4],"Task, w=1":[0.4,0.5,0.3,0.2,0.4,0.3,0.2,0.2,0.1,0.3,0.2,0.4,0.2,0.4],"Task, w=0.1":[-0.8,-0.3,-0.8,-1.4,-1.7,-1.2,-1.1,-0.9,-0.6,-0.2,0.4,0.7,0.6,0.7],"Task, w=0.03":[-2.0,-2.3,-2.6,-2.9,-2.9,-2.6,-2.3,-1.9,-1.4,-0.7,-0.2,-0.2,-0.2,-0.2],"Per-loss clipping":[-0.9,-1.0,-1.4,-1.9,-1.6,-1.3,-1.3,-0.8,-0.5,-0.3,0.0,0.1,0.2,0.2],"Per-task norm.":[-2.3,-3.1,-3.3,-3.6,-3.8,-3.3,-3.0,-2.3,-2.1,-1.8,-1.0,-0.9,-0.8,-0.8],"Per-task Adam":[-1.1,-2.3,-2.9,-3.6,-3.3,-2.5,-2.2,-1.8,-1.3,-0.9,0.0,0.2,0.1,0.2],"Depth alloc. 1 / 0.03":[0.2,1.4,1.2,0.7,0.6,0.7,0.6,0.7,0.1,-0.3,-0.2,-0.2,-0.3,-0.2],"Depth alloc. 0.3 / 0.03":[-0.2,0.2,0.4,0.4,0.3,0.4,0.4,0.5,0.3,0.1,0.3,0.5,0.6,0.5],"Depth alloc. 0.03 / 1":[-2.2,-3.8,-5.7,-6.9,-6.9,-6.2,-5.8,-3.7,-3.0,-1.8,-0.6,-0.4,-0.4,-0.6]},"IN":{"Dense, w=1":[0.1,0.2,0.5,-0.1,0.0,-0.3,-0.2,-0.1,0.2,-0.2,0.3,-0.1,0.4,0.3],"Dense, w=0.1":[-1.8,-2.4,-2.1,-2.6,-1.9,-1.6,-1.1,-1.0,0.4,0.6,1.0,0.5,1.1,1.4],"Dense, w=0.03":[-2.7,-4.5,-5.1,-5.0,-4.4,-3.6,-2.5,-0.6,0.5,0.6,1.0,1.1,1.6,1.4],"Dense, w=0.01":[-2.8,-5.7,-6.2,-6.4,-5.7,-4.8,-3.2,-2.4,-1.2,-1.4,-0.1,-0.8,-0.2,-0.2],"Task, w=1":[0.4,-0.1,0.4,-0.1,0.1,0.3,0.1,0.4,0.5,0.4,0.7,0.2,0.7,0.5],"Task, w=0.1":[-0.2,-1.0,-0.6,-1.1,-0.9,-0.8,-0.6,-0.3,0.5,0.4,1.0,0.4,0.8,0.9],"Task, w=0.03":[-1.9,-2.3,-2.5,-2.8,-2.2,-2.0,-1.3,-0.7,0.5,0.8,1.2,1.0,1.1,1.1],"Per-loss clipping":[-0.4,-2.1,-1.4,-1.7,-1.2,-1.0,-0.3,-0.2,1.1,1.0,1.8,0.5,1.6,1.6],"Per-task norm.":[-2.0,-3.6,-3.1,-3.7,-2.9,-3.0,-1.8,-1.4,-0.5,-0.3,0.5,-0.1,0.4,0.3],"Per-task Adam":[-1.2,-2.3,-3.2,-2.7,-2.4,-1.6,-0.8,-0.7,0.6,1.0,1.4,0.3,0.8,1.3],"Depth alloc. 1 / 0.03":[1.3,0.7,0.6,0.7,0.7,0.7,0.4,0.3,0.7,0.6,1.2,0.7,0.5,0.6],"Depth alloc. 0.3 / 0.03":[-0.3,0.3,0.9,0.3,0.2,0.3,0.3,0.1,1.1,0.6,1.0,0.2,0.8,1.5],"Depth alloc. 0.03 / 1":[-2.3,-4.8,-5.3,-6.0,-4.7,-3.6,-3.0,-1.9,-0.2,0.6,1.6,1.1,0.9,1.4]},"RHO":{"Dense, w=1 (5 seeds)":[11.5,11.4,12.3,15.4,17.2,22.4,29.7,42.9,60.1,20.9],"Dense, w=0.1":[1.6,1.84,2.46,3.01,3.62,4.77,6.32,8.49,10.8,3.92],"Dense, w=0.03 (5 seeds)":[0.58,0.62,0.92,1.14,1.41,1.87,2.42,3.21,3.86,1.46],"Dense, w=0.01":[0.22,0.27,0.35,0.43,0.53,0.68,0.89,1.18,1.46,0.55],"Per-loss clipping, w=1":[1.3,3.31,4.3,5.03,5.66,6.88,8.27,10.8,13.9,5.48],"1.7B, dense, w=1":[10.8,11.3,13.4,16.6,18.8,25.9,35.6,48.8,68.3,23.1],"1.7B, dense, w=0.03":[0.52,0.72,0.93,1.12,1.26,1.73,2.28,2.96,3.44,1.39]},"SS_COCO":[-0.3,-0.3,-0.5,-0.5,-0.3,-0.1,0.0,-0.1,-0.3,-0.2,0.0,0.1,0.1,-0.1],"SS_IN":[-0.2,-0.4,-0.3,-0.2,0.0,0.1,0.0,0.3,0.7,0.3,1.0,0.6,1.0,0.6]};
const LAYERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28];
const root = document.querySelector(".ss");
if (!root) return;

const FAM = { w: ["--c-wgt", "circle"], task: ["--c-rte", "square"], opt: ["--c-opt", null], depth: ["--c-spl", "pentagon"], ours: ["--c-spl", "star"] };
const ARMS = [
  ["w1", "Dense, w=1", "Dense, w=1", "w", "Dense, w = 1", "Equal weights: generation receives under 1% of each AdamW step, and FID rises 57% over generation-only training. This variant defines 0% of the gap closed."],
  ["w01", "Dense, 0.1", "Dense, w=0.1", "w", "Dense, w = 0.1", "Lowering the understanding weight to 0.1 restores part of generation's share and closes 68% of the FID gap, at a small cost in intermediate layers."],
  ["w003", "Dense, 0.03", "Dense, w=0.03", "w", "Dense, w = 0.03", "w = 0.03 closes 82% of the gap with no added parameters, but the COCO probe drops by up to 5.4 points around layer 8, a cost that caption NLL and the last layer do not show."],
  ["w001", "Dense, 0.01", "Dense, w=0.01", "w", "Dense, w = 0.01", "w = 0.01 closes 95% of the gap, and the intermediate-layer COCO deficit deepens to 7.1 points."],
  ["t1", "Task, w=1", "Task, w=1", "task", "Task routing, w = 1", "A separate feed-forward block for generation image tokens closes 60% of the gap at equal weights, with no probe cost."],
  ["t01", "Task, 0.1", "Task, w=0.1", "task", "Task routing, w = 0.1", "Task routing at w = 0.1 closes 80% of the gap; worst COCO layer −1.7."],
  ["t003", "Task, 0.03", "Task, w=0.03", "task", "Task routing, w = 0.03", "On top of w = 0.03, routing adds 11 points of gap closed (93%; not significant after Holm correction) and roughly halves the COCO probe deficit."],
  ["clip", "Per-loss clipping, w=1", "Per-loss clipping", "opt", "Per-loss clipping", "Clipping each loss's gradient separately lowers the gradient ratio to about 5 and closes 60% of the gap.", "square-open"],
  ["norm", "Per-task norm. grads, w=1", "Per-task norm.", "opt", "Per-task normalised gradients", "Normalising each task's gradient (ρ = 1) closes 90% of the gap but costs up to 3.8 COCO probe points in intermediate layers.", "triangle-open"],
  ["adam", "Per-task Adam, w=1", "Per-task Adam", "opt", "Per-task Adam", "One AdamW per task closes 92% of the gap at equal weights: generation gets 29% of the step, but layers 6 to 10 lose up to 3.6 COCO probe points.", "diamond"],
  ["d1", "Depth alloc. 1 / 0.03", "Depth alloc. 1 / 0.03", "depth", "Depth allocation 1 / 0.03", "Lowering the understanding weight only in the deep half closes 72% of the gap with no probe deficit."],
  ["d03", "Depth alloc. 0.3 / 0.03", "Depth alloc. 0.3 / 0.03", "depth", "Depth allocation 0.3 / 0.03", "Weight 0.3 in the shallow half and 0.03 in the deep half closes 84% of the gap; worst COCO layer −0.2."],
  ["d003", "Depth alloc. 0.03 / 1", "Depth alloc. 0.03 / 1", "depth", "Depth allocation 0.03 / 1", "Lowering the weight only in the shallow half closes just 61% and reproduces the COCO deficit (−6.9): the cost comes from the shallow half, the gain from the deep half."],
  ["ss", "StepShare-Adam, w=1", null, "ours", "StepShare-Adam", "Per-task Adam with generation's step scaled by β = 0.3 in the shallow half closes 91% of the gap with a worst COCO layer of −0.5 and no added parameters."]
].map(([id, t2, p, fam, label, note, shape]) => {
  const r = Object.assign({}, D.T2[t2]);
  if (r.r1 === null) { r.r1 = 0; r.r4 = 0; }
  return { id, fam, label, note, r, color: FAM[fam][0], shape: shape || FAM[fam][1],
    coco: p ? D.COCO[p] : D.SS_COCO, in: p ? D.IN[p] : D.SS_IN };
});
const byId = Object.fromEntries(ARMS.map(a => [a.id, a]));
const GROUPS = [["Loss weight w", ["w1", "w01", "w003", "w001"]], ["Task routing + loss weight", ["t1", "t01", "t003"]],
  ["Optimiser-level, w = 1", ["clip", "norm", "adam"]], ["Depth allocation (shallow / deep w)", ["d1", "d03", "d003"]], ["StepShare-Adam", ["ss"]]];
const GRIDS = [["gen", "Generation only", 113], ["dense", "Dense, w = 1", 340], ["w003", "Dense, w = 0.03", 157],
  ["split", "Per-task Adam", 119], ["ssadam", "StepShare-Adam", 142]];

/* ---------- helpers ---------- */
const NS = "http://www.w3.org/2000/svg";
const v = n => `var(${n})`;
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function T(parent, x, y, s, cls, anchor, extra) {
  const t = S("text", Object.assign({ x, y, class: cls, "text-anchor": anchor || "start" }, extra || {}), parent);
  t.textContent = s;
  return t;
}
const lin = (d0, d1, r0, r1) => x => r0 + (x - d0) / (d1 - d0) * (r1 - r0);
function sgn(x, d) {
  if (x === null || x === undefined) return "–";
  const s = Math.abs(x).toFixed(d === undefined ? 1 : d);
  return Number(s) === 0 ? s : (x > 0 ? "+" : "−") + s;
}
function markerPath(shape, x, y, r) {
  const P = pts => "M" + pts.map(p => p[0].toFixed(2) + "," + p[1].toFixed(2)).join("L") + "Z";
  const poly = (n, rad, rot) => Array.from({ length: n }, (_, k) => { const a = (rot + 360 * k / n) * Math.PI / 180; return [x + rad * Math.cos(a), y + rad * Math.sin(a)]; });
  switch (shape) {
    case "square": case "square-open": { const s = r * 0.88; return P([[x - s, y - s], [x + s, y - s], [x + s, y + s], [x - s, y + s]]); }
    case "diamond": { const s = r * 1.25; return P([[x, y - s], [x + s, y], [x, y + s], [x - s, y]]); }
    case "triangle-open": return P(poly(3, r * 1.25, -90));
    case "pentagon": return P(poly(5, r * 1.1, -90));
    case "star": return P(Array.from({ length: 10 }, (_, k) => { const a = (-90 + 36 * k) * Math.PI / 180, rad = k % 2 ? r * 0.72 : r * 1.65; return [x + rad * Math.cos(a), y + rad * Math.sin(a)]; }));
    default: return `M${x - r},${y}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0Z`;
  }
}
function marker(g, shape, x, y, r, color) {
  const open = shape.endsWith("-open");
  return S("path", { d: markerPath(shape, x, y, r), style: open
    ? `fill:var(--card);stroke:${v(color)};stroke-width:1.75` : `fill:${v(color)};stroke:var(--card);stroke-width:2` }, g);
}
function legend(box, items) {
  box.replaceChildren();
  for (const it of items) {
    const k = document.createElement("span"); k.className = "k";
    const s = S("svg", { width: it.kind === "line" ? 18 : 12, height: 12, "aria-hidden": "true" });
    if (it.kind === "line") S("line", { x1: 0, y1: 6, x2: 18, y2: 6, style: `stroke:${v(it.color)};stroke-width:${it.width || 2};stroke-linecap:round;opacity:${it.opacity || 1}` }, s);
    else marker(s, it.shape, 6, 6, 4, it.color);
    k.append(s, document.createTextNode(it.label));
    box.appendChild(k);
  }
}
const tip = document.createElement("div");
tip.className = "ss-tip"; tip.hidden = true; root.appendChild(tip);
function showTip(title, rows, cx, cy) {
  tip.replaceChildren();
  const h = document.createElement("div"); h.className = "tt"; h.textContent = title; tip.appendChild(h);
  for (const r of rows) {
    const d = document.createElement("div"); d.className = "tr";
    if (r.color) { const k = document.createElement("i"); k.style.background = v(r.color); d.appendChild(k); }
    const b = document.createElement("b"); b.textContent = r.value; d.appendChild(b);
    const l = document.createElement("span"); l.textContent = r.label; d.appendChild(l);
    tip.appendChild(d);
  }
  tip.hidden = false;
  const w = tip.offsetWidth, hh = tip.offsetHeight;
  let x = cx + 14, y = cy + 14;
  if (x + w > window.innerWidth - 8) x = cx - w - 14;
  if (y + hh > window.innerHeight - 8) y = cy - hh - 14;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
const hideTip = () => { tip.hidden = true; };
function axes(svg, o) {
  for (const t of o.yt) {
    const yy = o.y(t);
    S("line", { x1: o.m.l, x2: o.W - o.m.r, y1: yy, y2: yy, class: "gl" }, svg);
    T(svg, o.m.l - 6, yy + 3.5, o.yf ? o.yf(t) : String(t), "tk", "end");
  }
  for (const t of o.xt) {
    const xx = o.x(t);
    if (o.xgrid) S("line", { x1: xx, x2: xx, y1: o.m.t, y2: o.H - o.m.b, class: "gl" }, svg);
    T(svg, xx, o.H - o.m.b + 15, o.xf ? o.xf(t) : String(t), "tk", "middle");
  }
  S("line", { x1: o.m.l, x2: o.W - o.m.r, y1: o.H - o.m.b, y2: o.H - o.m.b, class: "ax" }, svg);
  T(svg, (o.m.l + o.W - o.m.r) / 2, o.H - 6, o.xl, "al", "middle");
  const cy = (o.m.t + o.H - o.m.b) / 2;
  T(svg, 12, cy, o.yl, "al", "middle", { transform: `rotate(-90 12 ${cy})` });
}
function newSvg(box, H, label) {
  box.replaceChildren();
  const W = Math.max(260, box.clientWidth || 360);
  return [S("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": label }, box), W];
}
const minus = t => (t < 0 ? "−" + -t : t > 0 ? "+" + t : "0");

/* ---------- arm explorer ---------- */
const ex = document.getElementById("ss-explore");
let sel = "ss", probe = "coco";
function select(id) { sel = id; document.getElementById("ss-arm").value = id; renderExplore(); }
function renderReadout() {
  const a = byId[sel], r = a.r, worst = probe === "coco" ? r.worst : r.inw;
  const out = document.getElementById("ss-readout");
  out.replaceChildren();
  const b = document.createElement("strong"); b.textContent = a.label + ". ";
  const ref = a.id === "w1";
  out.append(b, document.createTextNode(ref
    ? `Reference variant (${r.n} seeds); worst ${probe === "coco" ? "COCO" : "ImageNet-100"} layer ${sgn(worst)} points.`
    : `${r.r1}% ± ${r.c1} of the FID gap closed at guidance 1 and ${r.r4}% ± ${r.c4} at guidance 4; worst ${probe === "coco" ? "COCO" : "ImageNet-100"} layer ${sgn(worst)} points; caption NLL ${sgn(r.nll, 2)}; ${r.n} seeds.`));
  document.getElementById("ss-note").textContent = a.note;
  ex.style.setProperty("--accent", v(a.color));
  for (const b of ex.querySelectorAll("[data-pick]")) b.setAttribute("aria-pressed", String(b.dataset.pick === sel));
}
function renderScatter() {
  const box = document.getElementById("ss-scatter");
  const H = 290, m = { l: 46, r: 14, t: 12, b: 40 };
  const [svg, W] = newSvg(box, H, "FID gap closed against the worst probe layer, one point per arm");
  const x = lin(-8, 1, m.l, W - m.r), y = lin(-8, 108, H - m.b, m.t), key = probe === "coco" ? "worst" : "inw";
  axes(svg, { W, H, m, x, y, xt: [-8, -6, -4, -2, 0], yt: [0, 25, 50, 75, 100], xgrid: true, xf: minus,
    xl: "worst probe layer (points), closer to 0 is better", yl: "FID gap closed (%)" });
  const P = a => [x(a.r[key]), y(a.r.r1)];
  for (const [ids, c] of [[["w1", "w01", "w003", "w001"], "--c-wgt"], [["t1", "t01", "t003"], "--c-rte"]])
    S("path", { d: "M" + ids.map(id => P(byId[id]).map(n => n.toFixed(1)).join(",")).join("L"),
      style: `fill:none;stroke:${v(c)};stroke-width:2;stroke-linejoin:round;stroke-linecap:round;opacity:.85` }, svg);
  const gp = S("g", {}, svg), gl = S("g", {}, svg);
  for (const a of ARMS.filter(q => q.id !== sel).concat([byId[sel]])) {
    const [px, py] = P(a), on = a.id === sel;
    if (on) S("circle", { cx: px, cy: py, r: a.shape === "star" ? 13 : 10, style: "fill:none;stroke:var(--ink);stroke-width:1.5" }, gp);
    marker(gp, a.shape, px, py, on ? 5.5 : 4.5, a.color);
    const hit = S("circle", { cx: px, cy: py, r: 12, fill: "transparent", class: "hit", tabindex: 0, role: "button",
      "aria-label": `${a.label}: ${a.r.r1}% of the gap closed, worst layer ${sgn(a.r[key])} points` }, gp);
    const show = (cx, cy) => showTip(a.label, [{ value: a.r.r1 + "%", label: "gap closed (guidance 1)" }, { value: sgn(a.r[key]) + " pts", label: "worst probe layer" }], cx, cy);
    hit.addEventListener("pointermove", e => show(e.clientX, e.clientY));
    hit.addEventListener("pointerleave", hideTip);
    hit.addEventListener("focus", () => { const b = hit.getBoundingClientRect(); show(b.right, b.top); });
    hit.addEventListener("blur", hideTip);
    hit.addEventListener("click", () => select(a.id));
    hit.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(a.id); } });
  }
  // labels try spots around the point and keep the first clear of markers, the selection ring and placed labels
  const obstacles = ARMS.map(a => { const [px, py] = P(a), r = a.id === sel ? 14 : 8; return { id: a.id, x0: px - r, y0: py - r, x1: px + r, y1: py + r }; });
  const placed = [], hits = (b, c) => !(b.x1 < c.x0 || b.x0 > c.x1 || b.y1 < c.y0 || b.y0 > c.y1);
  const label = (a, txt, required) => {
    const [px, py] = P(a), right = px > (m.l + W - m.r) * 0.62, ring = a.id === sel ? 14 : 8;
    let spots = [[6, -ring - 5, "end"], [-6, -ring - 5, "start"], [ring + 4, 4, "start"], [-ring - 4, 4, "end"], [6, ring + 13, "end"], [-6, ring + 13, "start"], [6, -ring - 19, "end"], [6, ring + 27, "end"]];
    if (!right) spots = spots.map(([dx, dy, an]) => [-dx, dy, an === "end" ? "start" : "end"]);
    for (const [dx, dy, an] of spots) {
      const t = T(gl, px + dx, py + dy, txt, "lb halo", an), b = t.getBBox();
      const box2 = { x0: b.x - 2, y0: b.y - 1, x1: b.x + b.width + 2, y1: b.y + b.height + 1 };
      if (box2.x0 >= m.l && box2.x1 <= W - 2 && box2.y0 >= 0 && box2.y1 <= H - m.b && !obstacles.some(o => o.id !== a.id && hits(box2, o)) && !placed.some(q => hits(box2, q))) { placed.push(box2); return; }
      t.remove();
    }
    if (required) T(gl, px + spots[0][0], py + spots[0][1], txt, "lb halo", spots[0][2]);
  };
  label(byId[sel], byId[sel].label, true);
  if (sel !== "ss") label(byId.ss, "StepShare-Adam", false);
  legend(document.getElementById("ss-scatter-legend"), [
    { kind: "m", color: "--c-wgt", shape: "circle", label: "lower w" }, { kind: "m", color: "--c-rte", shape: "square", label: "task routing" },
    { kind: "m", color: "--c-spl", shape: "pentagon", label: "depth alloc." }, { kind: "m", color: "--c-opt", shape: "diamond", label: "per-task Adam" },
    { kind: "m", color: "--c-opt", shape: "triangle-open", label: "per-task norm." }, { kind: "m", color: "--c-opt", shape: "square-open", label: "per-loss clip" },
    { kind: "m", color: "--c-spl", shape: "star", label: "StepShare-Adam" }]);
}
function renderProfile() {
  const box = document.getElementById("ss-profile");
  const H = 290, m = { l: 40, r: 12, t: 12, b: 40 };
  const [svg, W] = newSvg(box, H, "Probe change by layer for the selected arm and two references");
  const x = lin(1, 29, m.l, W - m.r), y = lin(-8, 2, H - m.b, m.t);
  S("rect", { x: x(1), y: m.t, width: x(13) - x(1), height: H - m.b - m.t, class: "shade" }, svg);
  axes(svg, { W, H, m, x, y, xt: [2, 8, 14, 20, 28], yt: [-8, -6, -4, -2, 0, 2], yf: minus, xl: "layer", yl: "points vs. und.-only" });
  S("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: "ax" }, svg);
  T(svg, x(1) + 6, m.t + 13, "shallow half", "mu");
  const series = ["w003", "ss"].filter(id => id !== sel).map(id => ({ a: byId[id], ref: true })).concat([{ a: byId[sel], ref: false }]);
  const data = s => (probe === "coco" ? s.a.coco : s.a.in);
  for (const s of series)
    S("path", { d: "M" + data(s).map((val, i) => x(LAYERS[i]).toFixed(1) + "," + y(val).toFixed(1)).join("L"),
      style: `fill:none;stroke:${v(s.a.color)};stroke-width:${s.ref ? 1.5 : 2.5};stroke-linejoin:round;stroke-linecap:round;opacity:${s.ref ? 0.5 : 1}` }, svg);
  const main = data(series[series.length - 1]), mi = main.indexOf(Math.min(...main));
  if (main[mi] < -0.9) {
    const mx = x(LAYERS[mi]), my = y(main[mi]), below = my + 18 < H - m.b - 2;
    marker(svg, "circle", mx, my, 4, byId[sel].color);
    T(svg, below ? mx : mx + 8, below ? my + 18 : my + 4, `${sgn(main[mi])} at layer ${LAYERS[mi]}`, "lb halo", below ? "middle" : "start");
  }
  const cross = S("line", { y1: m.t, y2: H - m.b, class: "ax", visibility: "hidden" }, svg), dots = S("g", {}, svg);
  const ov = S("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  ov.addEventListener("pointermove", e => {
    const b = svg.getBoundingClientRect(), px = (e.clientX - b.left) * (W / b.width);
    let i = 0, best = 1e9;
    LAYERS.forEach((l, k) => { const dd = Math.abs(x(l) - px); if (dd < best) { best = dd; i = k; } });
    cross.setAttribute("x1", x(LAYERS[i])); cross.setAttribute("x2", x(LAYERS[i])); cross.setAttribute("visibility", "visible");
    dots.replaceChildren();
    for (const s of series) marker(dots, "circle", x(LAYERS[i]), y(data(s)[i]), 4, s.a.color);
    showTip(`Layer ${LAYERS[i]}`, series.slice().reverse().map(s => ({ value: sgn(data(s)[i]), label: s.a.label, color: s.a.color })), e.clientX, e.clientY);
  });
  ov.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); dots.replaceChildren(); hideTip(); });
  legend(document.getElementById("ss-profile-legend"), series.slice().reverse().map(s => ({ kind: "line", color: s.a.color,
    width: s.ref ? 1.5 : 2.5, opacity: s.ref ? 0.5 : 1, label: s.ref ? s.a.label + " (reference)" : s.a.label })));
  document.getElementById("ss-profile-sub").textContent = `· ${probe === "coco" ? "COCO" : "ImageNet-100"}, vs. und.-only`;
}
function renderExplore() { renderReadout(); renderScatter(); renderProfile(); }
function buildTable() {
  const t = document.getElementById("ss-all");
  const head = ["Arm", "Seeds", "Gap closed, g1 (%)", "Gap closed, g4 (%)", "ΔNLL", "COCO L28", "COCO worst", "IN-100 worst"];
  const tr = document.createElement("tr");
  for (const h of head) { const th = document.createElement("th"); th.scope = "col"; th.textContent = h; tr.appendChild(th); }
  const thead = document.createElement("thead"); thead.appendChild(tr);
  const tb = document.createElement("tbody");
  for (const a of ARMS) {
    const r = a.r, row = document.createElement("tr");
    [a.label, String(r.n), a.id === "w1" ? "0 (reference)" : `${r.r1} ± ${r.c1}`, a.id === "w1" ? "0 (reference)" : `${r.r4} ± ${r.c4}`,
      sgn(r.nll, 2), sgn(r.l28), sgn(r.worst), sgn(r.inw)].forEach((c, i) => {
      const td = document.createElement(i ? "td" : "th"); if (!i) td.scope = "row"; td.textContent = c; row.appendChild(td);
    });
    tb.appendChild(row);
  }
  t.append(thead, tb);
}
if (ex) {
  const s = document.getElementById("ss-arm");
  for (const [name, ids] of GROUPS) {
    const og = document.createElement("optgroup"); og.label = name;
    for (const id of ids) { const o = document.createElement("option"); o.value = id; o.textContent = byId[id].label; og.appendChild(o); }
    s.appendChild(og);
  }
  s.value = sel;
  s.addEventListener("change", () => select(s.value));
  for (const b of ex.querySelectorAll("[data-probe]")) b.addEventListener("click", () => {
    probe = b.dataset.probe;
    for (const o of ex.querySelectorAll("[data-probe]")) o.setAttribute("aria-pressed", String(o === b));
    renderExplore();
  });
  for (const b of ex.querySelectorAll("[data-pick]")) b.addEventListener("click", () => select(b.dataset.pick));
  buildTable();
  renderExplore();
  let lastW = 0;
  new ResizeObserver(() => { const w = ex.clientWidth; if (w !== lastW) { lastW = w; renderScatter(); renderProfile(); } }).observe(ex);
}

/* ---------- sample comparison ---------- */
const cmp = document.getElementById("ss-compare");
if (cmp) {
  const sa = document.getElementById("ss-wa"), sb = document.getElementById("ss-wb"), pos = document.getElementById("ss-wpos");
  for (const s of [sa, sb]) for (const [id, label, fid] of GRIDS) {
    const o = document.createElement("option"); o.value = id; o.textContent = `${label} (FID ${fid})`; s.appendChild(o);
  }
  sa.value = "dense"; sb.value = "ssadam";
  const base = cmp.dataset.base;
  const upd = () => {
    const A = GRIDS.find(g => g[0] === sa.value), B = GRIDS.find(g => g[0] === sb.value), p = Number(pos.value);
    const ia = document.getElementById("ss-ia"), ib = document.getElementById("ss-ib");
    if (ia.dataset.id !== A[0]) { ia.src = `${base}/${A[0]}.webp`; ia.dataset.id = A[0]; ia.alt = `36 samples, ${A[1]}`; }
    if (ib.dataset.id !== B[0]) { ib.src = `${base}/${B[0]}.webp`; ib.dataset.id = B[0]; ib.alt = `36 samples, ${B[1]}`; }
    ib.style.clipPath = `inset(0 0 0 ${p}%)`;
    cmp.querySelector(".bar").style.left = p + "%";
    cmp.querySelector(".knob").style.left = p + "%";
    document.getElementById("ss-ta").textContent = `${A[1]} · FID ${A[2]}`;
    document.getElementById("ss-tb").textContent = `${B[1]} · FID ${B[2]}`;
    pos.setAttribute("aria-valuetext", `${Math.round(p)}% ${A[1]}, ${Math.round(100 - p)}% ${B[1]}`);
  };
  sa.addEventListener("change", upd); sb.addEventListener("change", upd); pos.addEventListener("input", upd);
  upd();
}
})();
