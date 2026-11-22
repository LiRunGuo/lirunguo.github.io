/* Interactive figures of the horizon-aware-losses project page: decomposition, normalization, interventions,
   the toy calculation, the rolling-origin table and the recorded predictions.
   D is extracted from the paper's generated numbers file (the values behind its figures and tables), not typed by hand;
   PRED transcribes the paper's table of recorded predictions. Colours and type come from the page's .hl styles. */
(function () {
"use strict";
const D = {"dec":{"fft_l2":{"ds":{"ETTh1":[0.0,0.0],"ETTh2":[-0.0,0.24]},"mean":-0.0,"n":2},"freq_l2":{"ds":{"ETTh1":[0.41,0.45],"ETTh2":[1.77,0.57],"ETTm1":[0.69,0.38],"Weather":[3.88,0.71]},"mean":1.69,"n":4},"pca_l2":{"ds":{"ETTh1":[0.34,0.31],"ETTh2":[0.66,0.3],"ETTm1":[0.06,0.02],"Weather":[1.75,0.61]},"mean":0.7,"n":4},"freq_l1":{"ds":{"ETTh1":[-4.61,0.59],"ETTh2":[-4.17,0.6],"ETTm1":[-4.01,0.33],"Weather":[-4.49,1.09]},"mean":-4.32,"n":4},"fredf":{"ds":{"ETTh1":[-4.51,0.66],"ETTh2":[-4.14,0.58],"ETTm1":[-4.11,0.33],"Weather":[-2.74,1.45]},"mean":-3.87,"n":4},"pca_l1_nostd":{"ds":{"ETTh1":[-4.14,0.66],"ETTh2":[-2.69,0.77],"ETTm1":[-0.59,0.17],"Weather":[-1.77,0.3]},"mean":-2.3,"n":4},"timeo1":{"ds":{"ETTh1":[-4.17,0.67],"ETTh2":[-2.67,0.73],"ETTm1":[-0.58,0.17],"Weather":[-1.33,0.43]},"mean":-2.19,"n":4},"time_l1":{"ds":{"ETTh1":[-1.95,0.44],"ETTh2":[-2.04,0.59],"ETTm1":[-4.5,0.36],"Weather":[-5.41,1.24]},"mean":-3.48,"n":4},"huber_t":{"ds":{"ETTh1":[-1.51,0.46],"ETTh2":[-1.95,0.53],"ETTm1":[-4.34,0.3],"Weather":[-5.29,1.24]},"mean":-3.27,"n":4},"win_l2":{"ds":{"ETTh1":[-1.23,0.37],"ETTh2":[-2.84,0.62],"ETTm1":[-2.73,0.21],"Weather":[-4.72,1.29]},"mean":-2.88,"n":4}},"fur":{"fft_l2":{"ECL":[-0.11,0.22],"Traffic":[0.02,0.04]},"freq_l2":{"ECL":[1.28,0.33],"Traffic":[0.14,0.05]},"win_l2":{"ECL":[-0.83,0.26],"Traffic":[-0.4,0.1]},"huber_t":{"ECL":[-0.46,0.21],"Traffic":[1.17,0.21]},"freq_l1":{"ECL":[-1.97,0.52],"Traffic":[0.41,0.18]},"time_l1":{"ECL":[-0.62,0.7],"Traffic":[1.98,0.23]}},"norm":{"lin":{"time_l1":{"ETTh1":[-2.55,-0.47],"ETTh2":[-10.95,-1.17],"ETTm2":[-6.26,-0.39]},"freq_l1":{"ETTh1":[-3.25,-1.04],"ETTh2":[-11.64,-1.69],"ETTm2":[-6.84,-1.07]}},"dl":{"time_l1":{"ETTh1":[-2.4,-1.01],"ETTh2":[-14.27,-0.5],"ETTm2":[-9.77,-0.18]},"freq_l1":{"ETTh1":[1.87,2.74],"ETTh2":[-12.92,1.16],"ETTm2":[-8.99,-0.05]}},"itr":{"time_l1":{"ETTh1":[-2.43,-1.56],"ETTh2":[-17.91,-2.26],"ETTm2":[-17.77,-4.18]},"freq_l1":{"ETTh1":[-1.89,-3.04],"ETTh2":[-13.42,-3.77],"ETTm2":[-13.74,-5.5]}}},"roll":{"iTransformer":{"fredf":{"ETTh1":[-2.96,5,6,0],"ETTh2":[-3.71,5,7,0],"ETTm1":[-2.02,5,2,0],"ETTm2":[-5.38,5,14,0],"avg":[-3.52,20,29,0]},"freq_l1":{"ETTh1":[-3.04,5,6,0],"ETTh2":[-3.85,5,6,0],"ETTm1":[-2.0,5,2,0],"ETTm2":[-5.5,5,14,0],"avg":[-3.6,20,28,0]},"time_l1":{"ETTh1":[-1.53,5,0,0],"ETTh2":[-2.44,5,2,0],"ETTm1":[-1.91,5,2,0],"ETTm2":[-4.18,5,11,0],"avg":[-2.52,20,15,0]},"huber_t":{"ETTh1":[-1.34,5,0,0],"ETTh2":[-2.41,5,3,0],"ETTm1":[-1.75,5,3,0],"ETTm2":[-4.07,5,12,0],"avg":[-2.39,20,18,0]},"f_vs_t":{"ETTh1":[-1.52,5],"ETTh2":[-1.44,5],"ETTm1":[-0.08,3],"ETTm2":[-1.36,5],"avg":[-1.1,18]}},"DLinear":{"freq_l1":{"ETTh1":[1.87,0,0,0],"ETTh2":[-12.92,5,2,0],"ETTm1":[0.46,1,0,1],"ETTm2":[-8.99,5,7,0],"avg":[-4.89,11,9,1]},"time_l1":{"ETTh1":[-2.4,5,4,0],"ETTh2":[-14.27,5,3,0],"ETTm1":[-0.96,4,0,0],"ETTm2":[-9.77,5,8,0],"avg":[-6.85,19,15,0]},"huber_t":{"ETTh1":[-2.35,5,4,0],"ETTh2":[-13.84,5,3,0],"ETTm1":[-1.26,5,0,0],"ETTm2":[-10.26,5,12,0],"avg":[-6.93,20,19,0]},"f_vs_t":{"ETTh1":[4.38,0],"ETTh2":[1.47,0],"ETTm1":[1.45,0],"ETTm2":[0.85,0],"avg":[2.04,0]}}},"itv":{"lin":{"time_l1":{"ETTh1":-2.55,"ETTh2":-10.95,"ETTm1":-1.27,"ETTm2":-6.26},"mse_trim":{"ETTh1":-1.08,"ETTh2":-7.19,"ETTm1":-0.86,"ETTm2":-6.77},"mse_offdrop":{"ETTh1":-0.93,"ETTh2":-6.17,"ETTm1":-0.39,"ETTm2":-6.48},"mse_sclip":{"ETTh1":-0.19,"ETTh2":-2.87,"ETTm1":-0.32,"ETTm2":-4.24},"mse_gclip":{"ETTh1":-0.08,"ETTh2":-0.98,"ETTm1":-0.11,"ETTm2":-1.79},"time_bal":{"ETTh1":-0.98,"ETTh2":-6.69,"ETTm1":-0.68,"ETTm2":-5.74},"time_balema":{"ETTh1":-0.69,"ETTh2":-4.86,"ETTm1":-0.41,"ETTm2":-3.8},"freq_bal":{"ETTh1":-2.24,"ETTh2":-3.88,"ETTm1":-0.45,"ETTm2":-4.85},"freq_balema":{"ETTh1":-2.23,"ETTh2":-1.83,"ETTm1":-0.38,"ETTm2":-2.15},"dc2":{"ETTh1":1.03,"ETTh2":0.6,"ETTm1":0.52,"ETTm2":0.61},"nyq2":{"ETTh1":-0.0,"ETTh2":-0.0,"ETTm1":-0.0,"ETTm2":-0.0},"freq_l1":{"ETTh1":-3.25,"ETTh2":-11.64,"ETTm1":-1.26,"ETTm2":-6.84}},"dl":{"time_l1":{"ETTh2":-14.27,"ETTm2":-9.77},"freq_l1":{"ETTh2":-12.92,"ETTm2":-8.99},"mse_trim":{"ETTh2":-9.19,"ETTm2":-9.82},"mse_sclip":{"ETTh2":-3.77,"ETTm2":-6.07},"mse_offdrop":{"ETTh2":-7.16,"ETTm2":-9.55},"mse_gclip":{"ETTh2":-1.29,"ETTm2":-3.78},"time_bal":{"ETTh2":-8.02,"ETTm2":-8.71},"time_balema":{"ETTh2":-5.53,"ETTm2":-5.97},"freq_bal":{"ETTh2":-4.59,"ETTm2":-6.76},"freq_balema":{"ETTh2":-1.72,"ETTm2":-3.33}},"itr":{"time_l1":{"ETTh1":-1.56,"ETTh2":-2.26,"ETTm1":-1.91,"ETTm2":-4.18,"Weather":-5.41,"share":1.0,"neg":21},"freq_l1":{"ETTh1":-3.04,"ETTh2":-3.77,"ETTm1":-2.0,"ETTm2":-5.5,"Weather":-4.49,"share":1.382,"neg":21},"mse_trim":{"ETTh1":-1.45,"ETTh2":-2.26,"ETTm1":-0.88,"ETTm2":-3.6,"Weather":-4.4,"share":0.824,"neg":20},"mse_sclip":{"ETTh1":-0.21,"ETTh2":-0.36,"ETTm1":-0.22,"ETTm2":-2.14,"Weather":-3.98,"share":0.338,"neg":18},"mse_offdrop":{"ETTh1":0.13,"ETTh2":-0.45,"ETTm1":0.55,"ETTm2":-2.11,"Weather":-2.86,"share":0.223,"neg":14},"mse_gclip":{"ETTh1":0.05,"ETTh2":-0.49,"ETTm1":0.04,"ETTm2":-1.27,"Weather":-3.69,"share":0.219,"neg":14},"time_bal":{"ETTh1":-0.55,"ETTh2":-0.61,"ETTm1":-0.53,"ETTm2":-1.97,"Weather":-4.59,"share":0.417,"neg":20},"time_balema":{"ETTh1":-0.3,"ETTh2":-0.09,"ETTm1":-0.03,"ETTm2":-1.3,"Weather":2.97,"share":0.102,"neg":16},"freq_bal":{"ETTh1":-2.08,"ETTh2":-2.04,"ETTm1":-1.43,"ETTm2":-3.62,"Weather":-1.36,"share":0.858,"neg":20},"freq_balema":{"ETTh1":-1.95,"ETTh2":-1.22,"ETTm1":-1.41,"ETTm2":-2.94,"Weather":4.87,"share":0.595,"neg":20}}},"dc":{"freq_l2":{"ETTh1":[0.41,0.21],"ETTh2":[1.77,1.64],"ETTm1":[0.69,0.34],"Weather":[3.88,2.05]},"freq_l1":{"ETTh1":[-4.61,-2.86],"ETTh2":[-4.17,-2.38],"ETTm1":[-4.01,-2.42],"Weather":[-4.49,-2.77]},"fredf":{"ETTh1":[-4.51,-2.75],"ETTh2":[-4.14,-2.42],"ETTm1":[-4.11,-2.52],"Weather":[-2.74,-1.91]},"time_l1":{"ETTh1":[-1.95,-0.54],"ETTh2":[-2.04,-0.79],"ETTm1":[-4.5,-2.34],"Weather":[-5.41,-3.56]}},"qdf":{"qdf":{"ETTh1":[-1.73,0.42],"ETTh2":[-1.79,0.5],"ETTm1":[-0.82,0.23],"ETTm2":[-1.1,0.22],"Weather":[-0.41,0.09]},"qdf_eye":{"ETTh1":[0.37,0.68],"ETTh2":[0.27,0.3],"ETTm1":[-0.23,0.16],"ETTm2":[-0.38,0.25],"Weather":[0.05,0.16]},"qdf_rand":{"ETTh1":[0.44,0.9],"ETTh2":[0.28,0.29],"ETTm1":[-0.06,0.21],"ETTm2":[-0.37,0.23],"Weather":[0.05,0.15]},"qdf_l1":{"ETTh1":[-1.91,0.4],"ETTh2":[-2.09,0.59],"ETTm1":[-0.18,0.46],"ETTm2":[-0.56,0.39],"Weather":[-1.61,0.33]},"qdf_eye_l1":{"ETTh1":[-1.9,0.51],"ETTh2":[-2.17,0.47],"ETTm1":[-0.31,0.36],"ETTm2":[-0.76,0.32],"Weather":[-1.83,0.3]}},"train":{"DLinear":{"ETTh2":[51.2,15.97,1.41,20.15,-17.95],"ETTm2":[55.8,20.78,0.21,29.07,-15.96]},"DLinear + RevIN":{"ETTh2":[52.0,2.46,0.03,0.16,-0.59],"ETTm2":[58.7,5.34,0.15,3.89,-1.24]},"iTransformer, norm. off":{"ETTh2":[46.7,20.4,8.58,-25.09,-16.09],"ETTm2":[46.9,27.19,7.03,-24.75,-15.71]},"iTransformer":{"ETTh2":[47.8,4.43,0.57,-6.7,-1.47],"ETTm2":[54.5,3.05,-5.71,-5.59,-3.74]}}};
const root = document.querySelector(".hl");
if (!root) return;

/* ---------- helpers ---------- */
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
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
function H(tag, cls, text, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined && text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
const lin = (d0, d1, r0, r1) => x => r0 + (x - d0) / (d1 - d0) * (r1 - r0);
function sgn(x, d) {
  const s = Math.abs(x).toFixed(d === undefined ? 1 : d);
  return Number(s) === 0 ? s : (x > 0 ? "+" : "−") + s;
}
const pct = (x, d) => sgn(x, d) + "%";
const colorOf = x => (x < 0 ? "var(--c-b)" : "var(--c-w)");
/* bar from v0 to v1 along one axis, rounded (4px) at the data end only */
function barPath(horizontal, a0, a1, b, w) {
  const r = Math.min(4, Math.abs(a1 - a0), w / 2), s = a1 >= a0 ? 1 : -1;
  if (horizontal) return `M${a0},${b} L${a1 - s * r},${b} Q${a1},${b} ${a1},${b + r} L${a1},${b + w - r} Q${a1},${b + w} ${a1 - s * r},${b + w} L${a0},${b + w} Z`;
  return `M${b},${a0} L${b},${a1 - s * r} Q${b},${a1} ${b + r},${a1} L${b + w - r},${a1} Q${b + w},${a1} ${b + w},${a1 - s * r} L${b + w},${a0} Z`;
}
function marker(parent, shape, x, y, r, attrs) {
  let e;
  if (shape === "circle") e = S("circle", { cx: x, cy: y, r }, parent);
  else if (shape === "square") e = S("rect", { x: x - r * 0.9, y: y - r * 0.9, width: r * 1.8, height: r * 1.8 }, parent);
  else if (shape === "triangle") e = S("path", { d: `M${x},${y - r * 1.15} L${x + r * 1.1},${y + r * 0.85} L${x - r * 1.1},${y + r * 0.85} Z` }, parent);
  else if (shape === "diamond") e = S("path", { d: `M${x},${y - r * 1.25} L${x + r * 1.05},${y} L${x},${y + r * 1.25} L${x - r * 1.05},${y} Z` }, parent);
  else if (shape === "cross") e = S("path", { d: `M${x - r},${y - r} L${x + r},${y + r} M${x + r},${y - r} L${x - r},${y + r}`, "stroke-linecap": "round" }, parent);
  else e = S("path", { d: `M${x - r * 1.1},${y} L${x + r * 1.1},${y} M${x},${y - r * 1.1} L${x},${y + r * 1.1}`, "stroke-linecap": "round" }, parent);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}

/* ---------- tooltip (one for the page) ---------- */
const tip = H("div", "hl-tip", null, root);
tip.hidden = true;
function showTip(ev, title, rows, note) {
  tip.textContent = "";
  H("div", "tt", title, tip);
  (rows || []).forEach(([k, val]) => { const r = H("div", "tr", null, tip); H("span", null, k, r); H("b", null, val, r); });
  if (note) H("div", "tn", note, tip);
  tip.hidden = false;
  let x, y;
  if (ev && ev.clientX !== undefined && ev.type !== "focus") { x = ev.clientX + 14; y = ev.clientY + 14; }
  else { const b = ev.target.getBoundingClientRect(); x = b.left + b.width / 2 + 10; y = b.top + 10; }
  const w = tip.offsetWidth, h = tip.offsetHeight;
  if (x + w > window.innerWidth - 8) x = Math.max(8, x - w - 28);
  if (y + h > window.innerHeight - 8) y = Math.max(8, y - h - 28);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.hidden = true; };
function bindTip(el, fn) {
  el.setAttribute("tabindex", "0");
  ["pointermove", "focus"].forEach(t => el.addEventListener(t, ev => { const a = fn(); showTip(ev, a[0], a[1], a[2]); }));
  ["pointerleave", "blur"].forEach(t => el.addEventListener(t, hideTip));
}
window.addEventListener("scroll", hideTip, { passive: true });
function seg(host, items, current, onPick) {
  host.textContent = "";
  items.forEach(([id, label]) => {
    const b = H("button", null, label, host);
    b.type = "button";
    b.setAttribute("aria-pressed", String(id === current));
    b.addEventListener("click", () => onPick(id));
  });
}

/* ---------- 1. decomposition (paper Figure 2) ---------- */
const DEC = [
  ["fft_l2", ["FFT", "L2"], "FFT L2", "Full-spectrum FFT of the error, squared. Orthonormal, so it equals the MSE."],
  ["freq_l2", ["rFFT", "L2"], "rFFT L2", "Half-spectrum FFT, squared, as in the FreDF code. Weights the DC and Nyquist directions twice."],
  ["pca_l2", ["PCA", "L2"], "PCA L2", "Squared error in the principal-component basis of the labels (Time-o1 script)."],
  ["freq_l1", ["freq.", "L1"], "Frequency L1", "L1 norm of the rFFT of the error: FreDF without its MSE term."],
  ["fredf", ["FreDF", "★"], "FreDF (published)", "Official mixture: λ·MSE + (1 − λ)·frequency L1, λ = 0.1–0.2 on ETT and 0.4 on Weather."],
  ["pca_l1_nostd", ["PCA", "L1"], "PCA L1", "L1 norm in the label-PCA basis, without Time-o1's standardization."],
  ["timeo1", ["Time-o1", "★"], "Time-o1 (published)", "First-release code of Time-o1 with its own scripts."],
  ["time_l1", ["time", "L1"], "Time L1", "Plain absolute error in the time domain."],
  ["huber_t", ["Huber", ""], "Time Huber", "Huber loss in the time domain."],
  ["win_l2", ["win.", "L2,1"], "Window L2,1", "One L2 norm per window, not squared."]
];
const DS = [["ETTh1", "circle"], ["ETTh2", "square"], ["ETTm1", "triangle"], ["Weather", "diamond"]];
const DS2 = [["ECL", "cross", "Electricity"], ["Traffic", "plus", "Traffic"]];
function drawDecomp() {
  const host = $("hl-dec-chart");
  if (!host) return;
  const more = $("hl-dec-more").checked;
  host.textContent = "";
  const W = 720, Hh = 372, L = 44, R = 8, top = 30, bot = 70, gap = 14;
  const svg = S("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Change in test MSE against the MSE for ten losses: L2 losses in a transformed basis are at or above zero, L1-type losses below zero." }, host);
  const bw = (W - L - R - gap) / 10, y = lin(-7, 5, Hh - bot, top);
  const xOf = i => L + i * bw + (i >= 3 ? gap : 0);
  S("rect", { x: xOf(0), y: top - 22, width: 3 * bw, height: Hh - bot - top + 22 + 50, rx: 8, fill: "var(--band2)" }, svg);
  S("rect", { x: xOf(3), y: top - 22, width: 7 * bw, height: Hh - bot - top + 22 + 50, rx: 8, fill: "var(--band1)" }, svg);
  T(svg, xOf(0) + 8, top - 7, "L2 norm in a transformed basis", "lbb");
  T(svg, xOf(3) + 8, top - 7, "L1-type norms", "lbb");
  for (let t = -6; t <= 4; t += 2) {
    S("line", { x1: L, x2: W - R, y1: y(t), y2: y(t), class: t === 0 ? "ax" : "gl" }, svg);
    T(svg, L - 6, y(t) + 4, sgn(t, 0), "tk", "end");
  }
  T(svg, 12, (top + Hh - bot) / 2, "change in test MSE vs. MSE (%)", "al", "middle", { transform: `rotate(-90 12 ${(top + Hh - bot) / 2})` });
  DEC.forEach(([key, lab, name, desc], i) => {
    const d = D.dec[key], cx = xOf(i) + bw / 2;
    if (Math.abs(d.mean) >= 0.05) S("path", { d: barPath(false, y(0), y(d.mean), cx - 11, 22), fill: colorOf(d.mean), "fill-opacity": 0.55 }, svg);
    const pts = DS.filter(([n]) => d.ds[n]).map(([n, sh]) => [n, sh, d.ds[n], n]);
    if (more && D.fur[key]) DS2.forEach(([n, sh, full]) => pts.push([n, sh, D.fur[key][n], full]));
    const step = Math.min(13, (bw - 14) / Math.max(1, pts.length - 1));
    pts.forEach(([n, sh, val], j) => {
      const px = cx + (j - (pts.length - 1) / 2) * step;
      S("line", { x1: px, x2: px, y1: y(val[0] - val[1]), y2: y(val[0] + val[1]), stroke: "var(--ink2)", "stroke-width": 1.2 }, svg);
      const open = sh === "cross" || sh === "plus";
      marker(svg, sh, px, y(val[0]), open ? 3.6 : 4.2, open ? { fill: "none", stroke: "var(--ink)", "stroke-width": 1.8 }
        : { fill: "var(--ink)", stroke: "var(--card)", "stroke-width": 1.5 });
    });
    T(svg, cx, Hh - bot + 16, lab[0], "lb", "middle");
    if (lab[1]) T(svg, cx, Hh - bot + 30, lab[1], "lb", "middle");
    T(svg, cx, Hh - bot + 46, sgn(d.mean), "vl", "middle");
    const hit = S("rect", { x: xOf(i) + 1, y: top, width: bw - 2, height: Hh - top - bot + 50, rx: 6, class: "hit", "aria-label": name + ", mean " + pct(d.mean) }, svg);
    bindTip(hit, () => [name, pts.map(p => [p[3], pct(p[2][0], 2) + " ± " + p[2][1].toFixed(2)])
      .concat([["Mean of " + d.n + " (bar)", pct(d.mean, 2)]]), desc]);
  });
  T(svg, L - 6, Hh - bot + 46, "mean", "mu", "end");
  const lg = $("hl-dec-legend");
  lg.textContent = "";
  DS.map(([n, sh]) => [n, sh, false]).concat(more ? DS2.map(([n, sh, full]) => [full, sh, true]) : []).forEach(([n, sh, open]) => {
    const k = H("span", "k", null, lg), s = S("svg", { width: 12, height: 12, viewBox: "0 0 12 12" });
    marker(s, sh, 6, 6, open ? 3.6 : 4.2, open ? { fill: "none", stroke: "var(--ink)", "stroke-width": 1.8 } : { fill: "var(--ink)" });
    k.appendChild(s); k.appendChild(document.createTextNode(n));
  });
  [["var(--c-b)", "mean lower than the MSE"], ["var(--c-w)", "mean higher than the MSE"]].forEach(([c, t]) => {
    const k = H("span", "k", null, lg), s = S("svg", { width: 12, height: 12, viewBox: "0 0 12 12" });
    S("rect", { x: 1, y: 1, width: 10, height: 10, rx: 2, fill: c, "fill-opacity": 0.55 }, s);
    k.appendChild(s); k.appendChild(document.createTextNode(t));
  });
}
if ($("hl-dec-more")) { $("hl-dec-more").addEventListener("change", drawDecomp); drawDecomp(); }

/* ---------- 2. normalization (paper Figure 3) ---------- */
const NM = [["lin", "Linear model", "no centring → centring"], ["dl", "DLinear", "official → + RevIN"], ["itr", "iTransformer", "norm. off → official"]];
let normLoss = "time_l1";
function drawNorm() {
  const host = $("hl-norm-chart");
  if (!host) return;
  seg($("hl-norm-seg"), [["time_l1", "Time L1"], ["freq_l1", "Frequency L1"]], normLoss, id => { normLoss = id; drawNorm(); });
  host.textContent = "";
  const W = 720, Hh = 330, L = 44, R = 8, top = 34, bot = 34, pg = 12;
  const svg = S("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": "Gain of an L1 loss over the MSE without and with instance normalization, for a linear model, DLinear and iTransformer on ETTh1, ETTh2 and ETTm2." }, host);
  const y = lin(-20, 4, Hh - bot, top), pw = (W - L - R - 2 * pg) / 3;
  for (let t = -20; t <= 0; t += 5) T(svg, L - 6, y(t) + 4, sgn(t, 0), "tk", "end");
  T(svg, 12, (top + Hh - bot) / 2, "change in test MSE vs. MSE (%)", "al", "middle", { transform: `rotate(-90 12 ${(top + Hh - bot) / 2})` });
  S("defs", null, svg).innerHTML = '<marker id="hl-arr" viewBox="0 0 8 8" refX="6.5" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0.5,0.8 L7,4 L0.5,7.2" fill="none" stroke="var(--muted)" stroke-width="1.4"/></marker>';
  NM.forEach(([key, name, sub], p) => {
    const x0 = L + p * (pw + pg);
    S("rect", { x: x0, y: top - 26, width: pw, height: Hh - bot - top + 26 + 26, rx: 8, fill: "var(--soft)" }, svg);
    T(svg, x0 + 10, top - 9, name, "lbb");
    T(svg, x0 + pw - 10, top - 9, sub, "mu", "end");
    for (let t = -20; t <= 0; t += 5) S("line", { x1: x0, x2: x0 + pw, y1: y(t), y2: y(t), class: t === 0 ? "ax" : "gl" }, svg);
    ["ETTh1", "ETTh2", "ETTm2"].forEach((ds, j) => {
      const val = D.norm[key][normLoss][ds], cx = x0 + pw * (j + 0.5) / 3, y0 = y(val[0]), y1 = y(val[1]);
      const dir = y1 < y0 ? -1 : 1;
      if (Math.abs(y1 - y0) > 16) S("line", { x1: cx, x2: cx, y1: y0 + dir * 7, y2: y1 - dir * 8, stroke: "var(--muted)", "stroke-width": 1.4, "marker-end": "url(#hl-arr)" }, svg);
      S("circle", { cx, cy: y0, r: 5.5, fill: "var(--c-w)", stroke: "var(--card)", "stroke-width": 2 }, svg);
      S("circle", { cx, cy: y1, r: 4.5, fill: "var(--card)", stroke: "var(--c-b)", "stroke-width": 2.2 }, svg);
      T(svg, cx - 10, y0 + 4, sgn(val[0]), "vl halo", "end");
      T(svg, cx + 10, y1 + 4, sgn(val[1]), "lb halo");
      T(svg, cx, Hh - bot + 17, ds, "lb", "middle");
      const hit = S("rect", { x: cx - pw / 6 + 2, y: top, width: pw / 3 - 4, height: Hh - top - bot + 22, rx: 6, class: "hit", "aria-label": `${name}, ${ds}: ${pct(val[0])} without, ${pct(val[1])} with normalization` }, svg);
      bindTip(hit, () => [name + " · " + ds, [["Without normalization", pct(val[0], 2)], ["With normalization", pct(val[1], 2)]],
        (normLoss === "time_l1" ? "Time L1" : "Frequency L1") + " against the MSE of the same model."]);
    });
  });
  const lg = $("hl-norm-legend");
  lg.textContent = "";
  [["without instance normalization (value in bold, left)", true], ["with instance normalization (value right)", false]].forEach(([t, filled]) => {
    const k = H("span", "k", null, lg), s = S("svg", { width: 12, height: 12, viewBox: "0 0 12 12" });
    S("circle", { cx: 6, cy: 6, r: filled ? 5 : 4, fill: filled ? "var(--c-w)" : "var(--card)", stroke: filled ? "none" : "var(--c-b)", "stroke-width": 2 }, s);
    k.appendChild(s); k.appendChild(document.createTextNode(t));
  });
  const k = H("span", "k", null, lg), s = S("svg", { width: 12, height: 12, viewBox: "0 0 12 12" });
  S("path", { d: "M6,11 L6,2 M3,5 L6,1.5 L9,5", fill: "none", stroke: "var(--muted)", "stroke-width": 1.4 }, s);
  k.appendChild(s); k.appendChild(document.createTextNode("adding normalization"));
}
drawNorm();

/* ---------- 3. interventions (paper tables of interventions) ---------- */
const IV = [
  ["time_l1", "Time L1", "absolute error", "l1"],
  ["freq_l1", "Frequency L1", "L1 norm of the rFFT of the error", "l1"],
  ["mse_trim", "Trimmed MSE", "drop the 10% largest-error windows per batch", "sq"],
  ["mse_offdrop", "Offset-drop MSE", "drop the 10% windows with the largest level offset", "sq"],
  ["mse_sclip", "Per-window gradient clip", "each window's gradient clipped at the 90% quantile", "sq"],
  ["mse_gclip", "Global gradient clip", "clip the gradient norm at its running median", "sq"],
  ["time_bal", "L2 balanced per time step", "each coordinate scaled by its batch RMS", "sq"],
  ["freq_bal", "L2 balanced per frequency", "each rFFT coefficient scaled by its batch RMS", "sq"]
];
const IVM = [["lin", "Linear model", ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]], ["dl", "DLinear", ["ETTh2", "ETTm2"]], ["itr", "iTransformer", ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"]]];
const IVT = {
  lin: "Without normalization, dropping the largest-error windows recovers 66% of the time-L1 gain on ETTh2 and all of it on ETTm2, and dropping the largest-offset windows 56% and all of it. Clipping the global gradient norm recovers little.",
  dl: "DLinear agrees with the linear model: trimming recovers 64% of the time-L1 gain on ETTh2 and all of it on ETTm2, while clipping the global gradient norm recovers 9% and 39%.",
  itr: "With normalization, trimming recovers 82% of the time-L1 gain over 21 dataset × origin cells, but an L2 loss balanced per frequency recovers 86%, so these two interventions do not separate the accounts (see Finding 4)."
};
let ivModel = "lin", ivDs = "ETTh2";
function drawItv() {
  const host = $("hl-itv-chart");
  if (!host) return;
  const m = IVM.find(a => a[0] === ivModel);
  if (m[2].indexOf(ivDs) < 0) ivDs = m[2].indexOf("ETTh2") >= 0 ? "ETTh2" : m[2][0];
  seg($("hl-itv-model"), IVM.map(a => [a[0], a[1]]), ivModel, id => { ivModel = id; drawItv(); });
  seg($("hl-itv-ds"), m[2].map(d => [d, d]), ivDs, id => { ivDs = id; drawItv(); });
  $("hl-itv-take").textContent = IVT[ivModel];
  host.textContent = "";
  const rows = IV.map(r => [r, D.itv[ivModel][r[0]][ivDs]]);
  const ref = rows[0][1];
  const lo = Math.min(0, ...rows.map(r => r[1])), hi = Math.max(0, ...rows.map(r => r[1]));
  const span = Math.max(hi - lo, 1);
  const W = 720, L = 268, R = 168, top = 26, rh = 34, Hh = top + rows.length * rh + 30;
  const svg = S("svg", { viewBox: `0 0 ${W} ${Hh}`, role: "img", "aria-label": `Change in test MSE against the MSE for two L1 losses and six modified squared losses, ${m[1]} on ${ivDs}.` }, host);
  const padHi = hi > 0 ? 0.16 * span : 0.02 * span;
  const x = lin(lo - 0.02 * span, hi + padHi, L, W - R);
  const stepT = span > 12 ? 5 : span > 5 ? 2 : span > 2 ? 1 : 0.5;
  for (let t = Math.ceil(lo / stepT) * stepT; t <= hi + 1e-9; t += stepT) {
    S("line", { x1: x(t), x2: x(t), y1: top - 4, y2: Hh - 26, class: Math.abs(t) < 1e-9 ? "ax" : "gl" }, svg);
    T(svg, x(t), Hh - 12, sgn(t, stepT < 1 ? 1 : 0), "tk", "middle");
  }
  T(svg, (L + W - R) / 2, Hh + 6, "change in test MSE vs. MSE (%)", "al", "middle");
  S("rect", { x: 0, y: top - 2, width: W, height: 2 * rh, rx: 8, fill: "var(--band1)" }, svg);
  T(svg, 8, top - 8, "L1 losses", "mu");
  T(svg, 8, top + 2 * rh + 12, "Changes to the squared loss", "mu");
  rows.forEach(([r, val], i) => {
    const yy = top + i * rh + (i >= 2 ? 16 : 0), mid = yy + rh / 2;
    T(svg, 8, mid - 1, r[1], i === 0 ? "lbb" : "lb");
    T(svg, 8, mid + 12, r[2], "mu");
    if (Math.abs(val) >= 0.005) S("path", { d: barPath(true, x(0), x(val), mid - 8, 16), fill: colorOf(val), "fill-opacity": r[3] === "l1" ? 1 : 0.6 }, svg);
    const share = ref < 0 && i > 0 ? Math.round(val / ref * 100) : null;
    const lab = pct(val, 2) + (share !== null && r[3] === "sq" ? "  (" + share + "% of time L1)" : "");
    if (val <= 0) T(svg, x(Math.max(val, 0)) + 6, mid + 4, lab, "vl");
    else T(svg, x(val) + 6, mid + 4, pct(val, 2), "vl");
    const hit = S("rect", { x: 0, y: yy + 1, width: W, height: rh - 2, rx: 6, class: "hit", "aria-label": r[1] + " " + pct(val, 2) }, svg);
    bindTip(hit, () => [r[1], [[m[1] + " · " + ivDs, pct(val, 2)]].concat(share !== null ? [["Share of the time-L1 change", share + "%"]] : []), r[2] + "."]);
  });
  if (ref < 0) {
    S("line", { x1: x(ref), x2: x(ref), y1: top - 4, y2: Hh - 26, stroke: "var(--ink)", "stroke-width": 1.2 }, svg);
    T(svg, x(ref), top - 8, "time L1", "lb halo", "middle");
  }
  svg.setAttribute("viewBox", `0 0 ${W} ${Hh + 16}`);
}
drawItv();

/* ---------- 4. toy calculation: share of the loss held by the worst 10% of windows ---------- */
(function () {
  const sl = $("hl-toy-c");
  if (!sl) return;
  function bar(host, share) {
    host.textContent = "";
    const p = Math.round(share * 100);
    const a = H("i", null, p + "%", host);
    a.style.width = p + "%"; a.style.background = "var(--c-b)";
    const b = H("i", "rest", p < 80 ? "the other 90%" : "", host);
    b.style.width = (100 - p) + "%";
    host.setAttribute("role", "img");
    host.setAttribute("aria-label", "Worst 10% of windows hold " + p + "% of the loss");
  }
  function upd() {
    const c = Number(sl.value), l2 = 0.1 * c * c / (0.1 * c * c + 0.9), l1 = 0.1 * c / (0.1 * c + 0.9);
    $("hl-toy-v").textContent = c.toFixed(1) + "×";
    bar($("hl-toy-l2"), l2); bar($("hl-toy-l1"), l1);
    $("hl-toy-cap").textContent = "At " + c.toFixed(1) + "×, a tenth of the windows holds " + Math.round(l2 * 100) + "% of the squared loss but "
      + Math.round(l1 * 100) + "% of the absolute loss. Each of those windows pulls on the fit " + c.toFixed(1)
      + " times as hard under the squared loss, and exactly as hard as any other window under the absolute loss.";
  }
  sl.addEventListener("input", upd);
  upd();
})();

/* ---------- 5. rolling-origin table (paper Table 2) ---------- */
(function () {
  const tb = $("hl-roll");
  if (!tb) return;
  const cols = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "avg"];
  const names = { fredf: "FreDF ★", freq_l1: "Frequency L1", time_l1: "Time L1", huber_t: "Time Huber" };
  const hr = H("tr", null, null, H("thead", null, null, tb));
  H("th", null, "Backbone", hr); H("th", null, "Loss", hr);
  cols.forEach(c => H("th", "v" + (c === "avg" ? " avg" : ""), c === "avg" ? "Average" : c, hr));
  const body = H("tbody", null, null, tb);
  function cell(tr, c, val, small, title) {
    const td = H("td", "v" + (c === "avg" ? " avg" : ""), pct(val), tr);
    td.style.setProperty("--hc", colorOf(val));
    td.style.setProperty("--ha", Math.round(Math.min(Math.abs(val) / 15, 1) * 42 + (Math.abs(val) >= 0.05 ? 6 : 0)) + "%");
    H("small", null, small, td);
    td.title = title;
  }
  [["iTransformer", "built-in normalization", ["fredf", "freq_l1", "time_l1", "huber_t"]], ["DLinear", "no normalization", ["freq_l1", "time_l1", "huber_t"]]].forEach(([bb, sub, losses]) => {
    losses.forEach((l, i) => {
      const tr = H("tr", null, null, body);
      if (i === 0) { const g = H("td", "g", bb, tr); g.rowSpan = losses.length + 1; H("small", null, sub, g).style.cssText = "display:block;font-weight:400;color:var(--ink2)"; }
      H("td", null, names[l], tr);
      cols.forEach(c => {
        const a = D.roll[bb][l][c], n = c === "avg" ? 20 : 5, cells = c === "avg" ? 80 : 20;
        cell(tr, c, a[0], a[1] + "/" + n + " · " + a[2] + "/" + a[3],
          `${a[1]} of ${n} origins with a negative mean; ${a[2]} of ${cells} origin × horizon cells significantly better and ${a[3]} worse (Holm)`);
      });
    });
    const tr = H("tr", "sep", null, body);
    H("td", null, "Frequency L1 vs. time L1", tr).style.color = "var(--ink2)";
    cols.forEach(c => {
      const a = D.roll[bb].f_vs_t[c], n = c === "avg" ? 20 : 5;
      cell(tr, c, a[0], a[1] + "/" + n, `${a[1]} of ${n} origins where frequency L1 is better than time L1`);
    });
  });
})();

/* ---------- 6. recorded predictions (paper appendix table) ---------- */
const PRED = [
  ["S4-P6a", 0, "iTr", "rFFT and FFT L2 ≈ MSE (Parseval)", "held", "h"],
  ["S4-P6b", 0, "iTr", "FreDF's gain comes from the L1 norm, or a tuned MSE catches up", "L1 held; tuned MSE does not catch up", "h"],
  ["E3a-1", 1, "DL", "L1 gain grows with the test offset over all 35 cells / over the 15 new cells", "held / failed (low power)", "p"],
  ["E3a-2", 1, "DL", "Small gains (≲ 3%) for small offsets", "held", "h"],
  ["E3a-3", 1, "DL", "Small changes with RevIN", "held, except electricity", "p"],
  ["E3b-1", 2, "DL", "Exchange rates: L1 gain ≳ 10%, > 3% at every origin", "failed (−2.2%, −7.2%)", "f"],
  ["E3b-2, 3", 2, "DL", "Small changes with RevIN; gain grows with offset over 40 cells", "held", "h"],
  ["R3-1.1", 3, "iTr", "Electricity, traffic: FFT L2 within ±1%, rFFT L2 not better by > 1%", "held", "h"],
  ["R3-1.2", 3, "iTr", "Electricity, traffic: time and frequency L1 better than MSE", "failed (traffic)", "f"],
  ["R3-2.1, 2", 3, "LM", "Trimmed-MSE change follows the time-L1 change across cells", "held", "h"],
  ["R3-2.3", 3, "LM", "Exchange rates: trimming recovers ≥ half the L1 gain", "failed (+1.2% vs −8.7%)", "f"],
  ["E1", 1, "iTr", "Trimming but not balancing reproduces the gain ⇒ robust estimation", "both do; cause open", "p"],
  ["E4", 1, "TQNet", "Learned Ω = identity = random Ω ⇒ Ω irrelevant", "failed; |·| helps as much", "f"],
  ["S9-A.1, 2", 4, "LM", "Synthetic: L1 gain grows with the test shift; smaller without shift", "failed (opposite)", "f"],
  ["S9-A.3", 4, "LM", "Synthetic: no gain without training excursions", "held", "h"],
  ["S9-A.4", 4, "LM", "Synthetic, centred: shift has no effect", "time L1 held; freq. L1 up to 2.4 points", "p"],
  ["S9-A.5", 4, "LM", "Synthetic: trimming recovers ≥ half the L1 gain", "failed (0 of 7)", "f"],
  ["S9-B", 4, "iTr", "L1 on the DC coefficient only recovers ≥ half the gap to frequency L1, L1 except DC less", "held (66% vs. none)", "h"],
  ["S9-C.1", 4, "LM, iTr, DL", "Weather cleaning changes the overall time-L1 effect by ≤ 1.5 points", "LM, DL held; iTr failed (5.8, 2.7)", "p"],
  ["S9-C.2", 4, "iTr", "Cleaned Weather, norm. off: time-L1 change on wv and OT within −30%", "held (+2%, +3%)", "h"],
  ["S9-E.1", 4, "DL", "L = 336: time L1 ≥ 3% better on ETTh2, ETTm2 and ≥ 2× the RevIN change", "held", "h"],
  ["S9-E.2", 4, "iTr", "L = 336: switching off normalization increases the time-L1 gain on ETTh2, ETTm2", "ETTm2 only", "p"],
  ["S9-F.1", 4, "PatchTST", "FFT L2 within ±1%; rFFT L2 not better by > 1%", "held", "h"],
  ["S9-F.2", 4, "PatchTST", "Time and frequency L1 better than MSE on ≥ 4 of 5 datasets", "time L1 held; freq. L1 failed (0 of 5)", "p"],
  ["S9-D", 4, "TQNet", "Learned weights correlate positively with the SNR", "failed (GLS-like)", "f"],
  ["S10-A.1", 5, "iTr, DL (± norm.)", "Heavier tails of the training level errors go with larger time-L1 gains (ρ < −0.3)", "failed (−0.19)", "f"],
  ["S10-A.2", 5, "iTr, DL (± norm.)", "Tails heavier without normalization in ≥ 75% of matched pairs", "failed (7 of 56)", "f"],
  ["S10-B", 5, "PatchTST", "Equal-budget tuning at least halves the frequency-L1 deficit on ≥ 4 of 5 datasets", "failed (0 of 5)", "f"],
  ["S10-C", 5, "TQNet", "Fixed Ω with DC weight 0.25: ≥ 0.3% better on ≥ 4 of 5; ≥ half of QDF's gain", "failed (2 of 5; 8%)", "f"],
  ["S10-D", 5, "iTr", "L = 336: L1 on DC only recovers ≥ half the gap to frequency L1, L1 except DC less", "held (114% vs. −96%)", "h"],
  ["S10-E", 5, "PatchTST", "Cleaned Weather: higher MSE baseline; time-L1 change within 1.5 points", "held", "h"],
  ["S10-F.1", 5, "LM", "Trend or seasonal drift: time L1 ≥ 2% better at δ = 4 without centring", "held (trend, −5.9%)", "h"],
  ["S10-F.2", 5, "LM", "Trimming recovers ≥ half of these gains", "failed (2 of 5)", "f"],
  ["S10-G", 5, "TQNet", "Electricity, traffic: Ω = I with |·| within ±1% of QDF and ≥ 0.5% better than MSE", "traffic held; electricity failed", "p"]
];
(function () {
  const tb = $("hl-pred");
  if (!tb) return;
  const ST = { h: ["held", "Held", "✓", "var(--c-good)"], p: ["part", "Partly", "–", "var(--muted)"], f: ["fail", "Failed", "✕", "var(--c-bad)"] };
  const n = { h: 0, p: 0, f: 0 };
  PRED.forEach(r => { n[r[5]]++; });
  const sc = $("hl-score");
  ["h", "p", "f"].forEach(k => {
    const i = H("i", null, n[k] + " " + ST[k][1].toLowerCase(), sc);
    i.style.width = (n[k] / PRED.length * 100) + "%"; i.style.background = ST[k][3];
  });
  sc.setAttribute("aria-label", `${n.h} predictions held, ${n.p} partly held, ${n.f} failed`);
  let cur = "all";
  function draw() {
    seg($("hl-pred-chips"), [["all", "All " + PRED.length], ["h", "Held " + n.h], ["p", "Partly " + n.p], ["f", "Failed " + n.f]], cur, id => { cur = id; draw(); });
    tb.textContent = "";
    const hr = H("tr", null, null, H("thead", null, null, tb));
    ["Record", "Round", "Model", "Prediction", "Outcome", ""].forEach(h => H("th", null, h, hr));
    const body = H("tbody", null, null, tb);
    PRED.filter(r => cur === "all" || r[5] === cur).forEach(r => {
      const tr = H("tr", null, null, body);
      H("td", null, r[0], tr); H("td", null, String(r[1]), tr); H("td", null, r[2], tr); H("td", null, r[3], tr); H("td", "o", r[4], tr);
      const pill = H("span", "hl-pill " + ST[r[5]][0], null, H("td", null, null, tr));
      H("i", null, ST[r[5]][2], pill); pill.appendChild(document.createTextNode(ST[r[5]][1]));
    });
  }
  draw();
})();
})();
