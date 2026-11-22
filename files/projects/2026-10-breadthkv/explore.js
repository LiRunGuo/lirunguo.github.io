/* Interactive figures of the BreadthKV project page: the budget-split figure and the per-setting results table.
   Numbers are the paper's runs (the per-setting tables of its appendix): per run, [pass@1 %, share of samples at the
   length cap %, mean output tokens, KV memory-time relative to FullKV, samples, pass@1 as printed in the paper].
   Colours and type come from the page's .bkv styles. */
(function () {
"use strict";
const D = {"q3_8b":{"aime24":{"full":{"0":[74.167,6.2,15531,1.0,240,"74.2"]},"h2o":{"1024":[5.0,96.2,32354,0.244,240,"5.0"]},"hyb2":{"1024":[75.0,10.4,16451,0.107,240,"75.0"],"512":[62.083,29.6,19947,0.081,240,"62.1"]},"hyb3":{"1024":[72.083,18.8,17413,0.118,240,"72.1"],"512":[47.5,47.5,23201,0.096,240,"47.5"]},"hyb4":{"1024":[64.167,30.0,19019,0.133,240,"64.2"],"512":[35.833,57.9,25105,0.104,240,"35.8"]},"kivi4":{"0":[72.5,6.2,15539,0.307,240,"72.5"]},"random":{"1024":[36.667,60.4,24611,0.185,240,"36.7"],"512":[17.5,81.2,28597,0.126,240,"17.5"]},"rdkvblk":{"1024":[72.917,15.8,17633,0.157,240,"72.9"],"512":[61.25,32.9,21615,0.115,240,"61.3"]},"rkv":{"1024":[18.75,80.8,29592,0.223,240,"18.8"],"512":[8.333,92.9,31921,0.141,240,"8.3"]},"snapkv":{"1024":[25.0,74.2,27637,0.208,240,"25.0"],"512":[12.917,88.3,30844,0.136,240,"12.9"]},"streaming":{"1024":[24.583,75.8,27263,0.206,240,"24.6"]},"thinkvkc":{"1024":[43.75,50.0,21334,0.095,240,"43.8"],"512":[21.667,78.3,27667,0.079,240,"21.7"]},"vase":{"1024":[26.667,71.7,27283,0.2,240,"26.7"],"512":[11.25,89.2,31370,0.132,240,"11.2"]}},"aime25":{"full":{"0":[65.417,14.2,18450,1.0,240,"65.4"]},"h2o":{"1024":[5.833,97.5,32337,0.188,240,"5.8"]},"hyb2":{"1024":[62.917,19.2,19216,0.101,240,"62.9"],"2048":[66.25,10.8,17529,0.134,240,"66.2"],"4096":[63.75,6.7,17578,0.166,240,"63.7"],"512":[41.25,51.2,23840,0.079,240,"41.2"]},"hyb3":{"1024":[56.25,35.8,20651,0.112,240,"56.2"],"512":[28.75,62.1,25873,0.087,240,"28.7"]},"hyb4":{"1024":[50.417,45.4,22202,0.123,240,"50.4"],"2048":[61.25,27.9,19844,0.179,240,"61.3"],"4096":[68.333,14.2,18007,0.239,240,"68.3"],"512":[21.667,72.9,27759,0.093,240,"21.7"]},"hyb8":{"2048":[55.0,40.4,21527,0.211,240,"55.0"],"4096":[65.0,26.7,19444,0.319,240,"65.0"]},"kivi4":{"0":[67.5,11.7,17870,0.288,240,"67.5"]},"random":{"1024":[22.917,69.2,26293,0.154,240,"22.9"],"512":[14.167,84.6,29232,0.104,240,"14.2"]},"rdkvblk":{"1024":[60.833,25.0,20718,0.156,240,"60.8"],"512":[47.917,49.6,24577,0.11,240,"47.9"]},"rkv":{"1024":[6.667,93.8,31432,0.183,240,"6.7"],"512":[3.333,96.7,32124,0.113,240,"3.3"]},"snapkv":{"1024":[11.667,82.5,29304,0.172,240,"11.7"],"2048":[37.083,56.7,24728,0.253,240,"37.1"],"4096":[52.083,44.2,22011,0.402,240,"52.1"],"512":[5.833,93.8,31439,0.111,240,"5.8"]},"streaming":{"1024":[19.167,78.3,27505,0.161,240,"19.2"]},"thinkvkc":{"1024":[27.5,65.4,24573,0.084,240,"27.5"],"512":[14.583,85.8,28852,0.068,240,"14.6"]},"vase":{"1024":[13.333,83.3,29104,0.166,240,"13.3"],"512":[6.25,95.4,31821,0.108,240,"6.2"]}},"math500":{"full":{"0":[94.8,4.8,5184,1.0,500,"94.8"]},"h2o":{"1024":[59.0,45.4,10625,0.562,500,"59.0"]},"hyb2":{"1024":[93.4,4.6,5356,0.189,500,"93.4"],"512":[90.4,8.8,5876,0.156,500,"90.4"]},"hyb3":{"1024":[91.6,7.2,5458,0.218,500,"91.6"],"512":[89.0,10.2,5792,0.161,500,"89.0"]},"hyb4":{"1024":[92.2,6.8,5359,0.224,500,"92.2"],"512":[84.8,15.0,6180,0.176,500,"84.8"]},"kivi4":{"0":[94.8,3.8,5236,0.336,500,"94.8"]},"random":{"1024":[87.4,12.4,5390,0.275,500,"87.4"],"512":[80.2,19.8,5992,0.186,500,"80.2"]},"rdkvblk":{"1024":[91.6,6.8,5765,0.206,500,"91.6"],"512":[86.4,14.4,6626,0.17,500,"86.4"]},"rkv":{"1024":[73.4,28.0,7796,0.409,500,"73.4"],"512":[61.2,39.0,9385,0.292,500,"61.2"]},"snapkv":{"1024":[81.8,17.2,6585,0.341,500,"81.8"],"512":[73.0,28.0,7661,0.239,500,"73.0"]},"streaming":{"1024":[84.0,15.0,5118,0.26,500,"84.0"]},"thinkvkc":{"1024":[89.6,10.0,5447,0.161,500,"89.6"],"512":[82.6,17.4,5656,0.118,500,"82.6"]},"vase":{"1024":[81.0,18.8,6485,0.328,500,"81.0"],"512":[71.0,31.2,7918,0.236,500,"71.0"]}}},"r1l8b":{"aime24":{"full":{"0":[51.25,5.8,13381,1.0,240,"51.2"]},"h2o":{"1024":[20.417,75.4,27350,0.257,240,"20.4"]},"hyb2":{"1024":[50.0,3.3,13584,0.107,240,"50.0"],"512":[45.833,15.4,16576,0.084,240,"45.8"]},"hyb3":{"1024":[51.25,9.6,15022,0.126,240,"51.2"],"512":[52.5,26.7,18635,0.096,240,"52.5"]},"hyb4":{"1024":[50.417,15.4,16231,0.14,240,"50.4"],"512":[53.333,32.1,18165,0.094,240,"53.3"]},"kivi4":{"0":[45.833,5.0,13399,0.299,240,"45.8"]},"random":{"1024":[42.5,45.0,20952,0.196,240,"42.5"],"512":[28.333,67.1,25298,0.139,240,"28.3"]},"rdkvblk":{"1024":[48.333,4.2,13619,0.168,240,"48.3"],"512":[45.417,19.2,18605,0.12,240,"45.4"]},"rkv":{"1024":[43.333,40.8,21519,0.202,240,"43.3"],"512":[30.0,57.1,24039,0.132,240,"30.0"]},"snapkv":{"1024":[41.667,45.8,21335,0.2,240,"41.7"],"512":[31.667,60.0,23978,0.132,240,"31.7"]},"streaming":{"1024":[32.083,55.8,22863,0.215,240,"32.1"]},"thinkvkc":{"1024":[45.0,40.4,19785,0.109,240,"45.0"],"512":[27.5,74.2,25920,0.085,240,"27.5"]},"vase":{"1024":[39.583,47.5,21329,0.195,240,"39.6"],"512":[27.5,65.4,25475,0.133,240,"27.5"]}},"aime25":{"full":{"0":[31.25,6.2,14325,1.0,240,"31.2"]},"h2o":{"1024":[16.667,78.3,27810,0.246,240,"16.7"]},"hyb2":{"1024":[32.917,5.4,14051,0.108,240,"32.9"],"2048":[33.75,4.6,13278,0.143,240,"33.8"],"4096":[31.25,5.0,13322,0.166,240,"31.2"],"512":[30.0,19.2,17420,0.088,240,"30.0"]},"hyb3":{"1024":[30.833,10.4,15790,0.127,240,"30.8"],"512":[32.917,32.5,18850,0.097,240,"32.9"]},"hyb4":{"1024":[34.583,19.6,16842,0.139,240,"34.6"],"2048":[35.833,5.4,13745,0.177,240,"35.8"],"4096":[33.333,9.6,14208,0.268,240,"33.3"],"512":[30.417,45.4,21221,0.109,240,"30.4"]},"hyb8":{"2048":[34.583,19.2,16945,0.247,240,"34.6"],"4096":[35.833,6.7,14334,0.336,240,"35.8"]},"kivi4":{"0":[34.583,7.5,14206,0.319,240,"34.6"]},"random":{"1024":[28.75,50.4,21839,0.194,240,"28.7"],"512":[22.5,64.2,24626,0.134,240,"22.5"]},"rdkvblk":{"1024":[35.833,4.6,14657,0.177,240,"35.8"],"512":[31.667,25.0,19142,0.128,240,"31.7"]},"rkv":{"1024":[27.083,42.5,20832,0.185,240,"27.1"],"512":[24.583,60.0,24618,0.133,240,"24.6"]},"snapkv":{"1024":[27.083,51.2,22500,0.199,240,"27.1"],"2048":[30.417,35.4,19645,0.303,240,"30.4"],"4096":[36.667,15.0,14921,0.397,240,"36.7"],"512":[26.25,65.4,25395,0.137,240,"26.2"]},"streaming":{"1024":[25.417,59.6,23296,0.206,240,"25.4"]},"thinkvkc":{"1024":[29.167,45.4,20191,0.105,240,"29.2"],"512":[21.25,72.1,25502,0.087,240,"21.2"]},"vase":{"1024":[28.333,54.2,22579,0.196,240,"28.3"],"512":[21.667,67.9,25689,0.133,240,"21.7"]}},"math500":{"full":{"0":[89.4,5.0,4027,1.0,500,"89.4"]},"h2o":{"1024":[75.6,22.2,6344,0.429,500,"75.6"]},"hyb2":{"1024":[86.2,5.6,4294,0.194,500,"86.2"],"512":[88.0,6.0,4183,0.14,500,"88.0"]},"hyb3":{"1024":[87.8,5.0,4190,0.206,500,"87.8"],"512":[89.6,5.6,4217,0.146,500,"89.6"]},"hyb4":{"1024":[89.6,5.4,4089,0.211,500,"89.6"],"512":[88.2,7.8,4514,0.165,500,"88.2"]},"kivi4":{"0":[89.0,3.4,3957,0.317,500,"89.0"]},"random":{"1024":[86.8,8.8,4361,0.284,500,"86.8"],"512":[83.0,11.2,4713,0.189,500,"83.0"]},"rdkvblk":{"1024":[89.4,4.6,4098,0.215,500,"89.4"],"512":[86.4,7.2,4497,0.143,500,"86.4"]},"rkv":{"1024":[86.2,10.2,4664,0.306,500,"86.2"],"512":[83.8,13.0,5110,0.206,500,"83.8"]},"snapkv":{"1024":[87.0,7.6,4388,0.285,500,"87.0"],"512":[82.8,11.4,4774,0.192,500,"82.8"]},"streaming":{"1024":[86.4,8.6,4033,0.26,500,"86.4"]},"thinkvkc":{"1024":[88.8,6.4,4188,0.144,500,"88.8"],"512":[82.8,15.2,4898,0.12,500,"82.8"]},"vase":{"1024":[86.4,9.6,4393,0.279,500,"86.4"],"512":[80.0,15.2,5345,0.206,500,"80.0"]}}},"r1q7b":{"aime25":{"full":{"0":[41.25,6.7,13993,1.0,240,"41.2"]},"h2o":{"1024":[19.583,50.4,22847,0.196,240,"19.6"]},"hyb2":{"1024":[35.417,11.2,15156,0.113,240,"35.4"],"2048":[35.417,7.1,14229,0.152,240,"35.4"],"4096":[36.25,4.2,13346,0.166,240,"36.2"],"512":[35.417,15.0,16526,0.082,240,"35.4"]},"hyb3":{"1024":[37.5,12.5,15821,0.123,240,"37.5"],"512":[33.333,15.0,16466,0.082,240,"33.3"]},"hyb4":{"1024":[36.25,12.5,15635,0.125,240,"36.2"],"2048":[37.917,10.8,15143,0.192,240,"37.9"],"4096":[42.5,7.1,13677,0.245,240,"42.5"],"512":[29.167,22.1,18264,0.092,240,"29.2"]},"hyb8":{"2048":[37.083,12.1,15041,0.21,240,"37.1"],"4096":[42.083,7.9,14361,0.325,240,"42.1"]},"kivi4":{"0":[46.667,8.3,13611,0.294,240,"46.7"]},"random":{"1024":[27.917,25.0,18521,0.16,240,"27.9"],"512":[21.25,35.4,20465,0.108,240,"21.2"]},"rdkvblk":{"1024":[38.333,9.6,14984,0.162,240,"38.3"],"512":[28.333,20.8,18236,0.11,240,"28.3"]},"rkv":{"1024":[19.167,20.0,20094,0.173,240,"19.2"],"512":[8.75,24.2,23193,0.121,240,"8.8"]},"snapkv":{"1024":[26.25,20.0,18561,0.16,240,"26.2"],"2048":[31.25,23.3,17704,0.264,240,"31.2"],"4096":[34.583,13.8,15784,0.409,240,"34.6"],"512":[20.417,22.9,19673,0.104,240,"20.4"]},"streaming":{"1024":[28.333,17.9,15675,0.134,240,"28.3"]},"thinkvkc":{"1024":[30.833,26.2,17770,0.091,240,"30.8"],"512":[23.75,39.6,21678,0.076,240,"23.8"]},"vase":{"1024":[28.333,25.0,17681,0.149,240,"28.3"],"512":[20.0,28.7,19502,0.1,240,"20.0"]}},"gpqa":{"full":{"0":[36.364,4.5,8965,1.0,792,"36.4"]},"h2o":{"1024":[22.475,30.7,18748,0.364,792,"22.5"]},"hyb2":{"1024":[36.869,1.1,7959,0.118,792,"36.9"],"512":[33.965,2.7,9251,0.103,792,"34.0"]},"hyb3":{"1024":[36.742,4.2,9532,0.162,792,"36.7"],"512":[31.061,5.6,10549,0.123,792,"31.1"]},"hyb4":{"1024":[36.237,2.5,9107,0.155,792,"36.2"],"512":[29.672,6.1,11104,0.131,792,"29.7"]},"kivi4":{"0":[39.52,2.7,8693,0.293,792,"39.5"]},"random":{"1024":[28.914,7.6,11993,0.234,792,"28.9"],"512":[23.106,11.0,13769,0.169,792,"23.1"]},"rdkvblk":{"1024":[35.732,2.3,9543,0.177,792,"35.7"],"512":[30.808,4.7,12496,0.152,792,"30.8"]},"rkv":{"1024":[28.03,2.9,12068,0.231,792,"28.0"],"512":[23.359,2.7,13974,0.171,792,"23.4"]},"snapkv":{"1024":[29.419,5.7,11933,0.23,792,"29.4"],"512":[23.737,5.8,13627,0.169,792,"23.7"]},"streaming":{"1024":[29.924,8.7,12312,0.24,792,"29.9"]},"thinkvkc":{"1024":[34.722,3.5,10217,0.118,792,"34.7"],"512":[27.146,15.2,14936,0.119,792,"27.1"]},"vase":{"1024":[30.303,7.4,11471,0.215,792,"30.3"],"512":[24.747,14.5,14313,0.168,792,"24.7"]}},"math500":{"full":{"0":[93.0,4.2,3753,1.0,500,"93.0"]},"h2o":{"1024":[78.4,19.2,5670,0.453,500,"78.4"]},"hyb2":{"1024":[91.2,4.4,3843,0.194,500,"91.2"],"512":[92.0,5.2,3817,0.149,500,"92.0"]},"hyb3":{"1024":[92.0,3.2,3776,0.208,500,"92.0"],"512":[91.0,4.4,3801,0.155,500,"91.0"]},"hyb4":{"1024":[92.6,4.2,3690,0.217,500,"92.6"],"512":[89.0,7.6,4140,0.178,500,"89.0"]},"kivi4":{"0":[94.4,3.0,3692,0.318,500,"94.4"]},"random":{"1024":[88.2,7.2,4174,0.323,500,"88.2"],"512":[82.8,11.2,4622,0.222,500,"82.8"]},"rdkvblk":{"1024":[91.4,4.8,3938,0.212,500,"91.4"],"512":[88.8,5.4,4377,0.158,500,"88.8"]},"rkv":{"1024":[84.8,9.6,4610,0.361,500,"84.8"],"512":[75.8,15.2,5910,0.286,500,"75.8"]},"snapkv":{"1024":[86.8,8.2,4269,0.332,500,"86.8"],"512":[82.4,11.8,4926,0.238,500,"82.4"]},"streaming":{"1024":[88.4,5.4,3742,0.284,500,"88.4"]},"thinkvkc":{"1024":[91.0,5.4,3879,0.156,500,"91.0"],"512":[87.2,10.0,4744,0.141,500,"87.2"]},"vase":{"1024":[88.0,8.8,4239,0.321,500,"88.0"],"512":[80.4,12.8,4980,0.23,500,"80.4"]}}}};
const MODELS = [["r1q7b", "R1-Distill-Qwen-7B", "R1-Qwen-7B"], ["r1l8b", "R1-Distill-Llama-8B", "R1-Llama-8B"], ["q3_8b", "Qwen3-8B", "Qwen3-8B"]];
const BENCH = { aime24: "AIME24", aime25: "AIME25", math500: "MATH500", gpqa: "GPQA-Diamond" };
const BENCH_ORDER = ["aime24", "aime25", "math500", "gpqa"];
/* bit-width chosen by calibration for each model and budget (Table 3) */
const CALIB = { r1q7b: { 512: 3, 1024: 4 }, r1l8b: { 512: 3, 1024: 2 }, q3_8b: { 512: 2, 1024: 2 } };
const root = document.querySelector(".bkv");
if (!root) return;

/* ---------- helpers ---------- */
const $ = (sel, el) => (el || root).querySelector(sel);
const fmt = n => n.toLocaleString("en-US");
const pct = (x, d) => x.toFixed(d === undefined ? 1 : d) + "%";
const acc = r => r[5] + "%";
/* one decimal with ties to even, as the paper's tables are printed */
function fix1(x) {
  const n = Math.round(Math.abs(x) * 1000), rem = n % 100;
  let q = Math.floor(n / 100);
  if (rem > 50 || (rem === 50 && q % 2 === 1)) q += 1;
  return (q / 10).toFixed(1);
}
const sgn = x => (x > 0 ? "+" : x < 0 ? "−" : "") + fix1(x);
/* tokens a head holds at b bits within the bytes of B fp16 tokens (Eq. 1); 16 bits means eviction alone */
const tokensAt = (B, b) => b === 16 ? B : Math.floor(160 + (B - 160) * 16 / (b + 0.5));
const run = (m, bench, meth, B) => { const r = (D[m][bench] || {})[meth]; return r && r[B] ? r[B] : null; };
function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}
/* segmented control: opts = [[value, label, disabledTitle?]] */
function seg(box, opts, value, onPick) {
  box.replaceChildren();
  for (const [v, label, off] of opts) {
    const b = el("button", "", label);
    b.type = "button";
    b.setAttribute("aria-pressed", String(v === value));
    if (off) { b.disabled = true; b.title = off; }
    b.addEventListener("click", () => onPick(v));
    box.appendChild(b);
  }
}

/* ---------- split the budget ---------- */
(function () {
  const fig = $("#bkv-split");
  if (!fig) return;
  const state = { m: "q3_8b", B: 512, b: 2 };
  const BITS = [[16, "16-bit, eviction only"], [8, "8-bit"], [4, "4-bit"], [3, "3-bit"], [2, "2-bit"]];
  const BUDGETS = [512, 1024, 2048, 4096];
  const CELL = 64;
  const methOf = b => b === 16 ? "snapkv" : "hyb" + b;
  const has = (b, B) => !!run(state.m, "aime25", methOf(b), B);

  function meter(box, label, value, text, max, refs, cls) {
    const row = el("div", "bkv-meter");
    row.appendChild(el("div", "lab", label));
    const track = el("div", "track");
    const fill = el("i", cls);
    fill.style.width = Math.max(0.6, Math.min(100, value / max * 100)) + "%";
    track.appendChild(fill);
    for (const [x, c] of refs) {
      const t = el("span", "ref " + c);
      t.style.left = Math.min(100, x / max * 100) + "%";
      track.appendChild(t);
    }
    row.appendChild(track);
    row.appendChild(el("div", "val", text));
    box.appendChild(row);
  }

  function render() {
    if (!has(state.b, state.B)) state.b = has(2, state.B) ? 2 : 16;
    seg($('[data-role="model"]', fig), MODELS.map(x => [x[0], x[2]]), state.m, v => { state.m = v; render(); });
    seg($('[data-role="budget"]', fig), BUDGETS.map(B => [B, fmt(B) + (B > 1024 ? "†" : "")]), state.B, v => { state.B = v; render(); });
    seg($('[data-role="bits"]', fig), BITS.map(([b, l]) => [b, l, has(b, state.B) ? "" : "Not run at this budget"]), state.b, v => { state.b = v; render(); });

    const full = run(state.m, "aime25", "full", "0"), ev = run(state.m, "aime25", "snapkv", state.B);
    const r = run(state.m, "aime25", methOf(state.b), state.B);
    const quant = state.b !== 16, n = tokensAt(state.B, state.b), trace = full[2];

    /* grid: one square per 64 tokens of an average FullKV trace */
    const grid = $('[data-role="grid"]', fig);
    grid.replaceChildren();
    const total = Math.ceil(trace / CELL), held = Math.min(total, Math.round(n / CELL)), win = quant ? 2 : 0;
    for (let i = 0; i < total; i++) grid.appendChild(el("i", i < held - win ? (quant ? "q" : "e") : i < held ? "w" : ""));
    grid.setAttribute("aria-label", `${fmt(Math.min(n, trace))} of ${fmt(trace)} tokens held`);
    const share = n / trace;
    $('[data-role="gridcap"]', fig).innerHTML = `<b>${fmt(n)} tokens</b> per head at ${state.b} bits` +
      (quant ? `, ${(n / state.B).toFixed(1)}× the ${fmt(state.B)} that eviction keeps in the same bytes` : "") +
      (share >= 1 ? ". That is more than an average trace of " + fmt(trace) + " tokens."
        : `, or ${pct(share * 100, 0)} of an average trace of ${fmt(trace)} tokens.`);

    /* outcome on AIME25 */
    const out = $('[data-role="meters"]', fig);
    out.replaceChildren();
    const cls = quant ? "q" : "e";
    meter(out, "Accuracy (pass@1)", r[0], acc(r), 80, [[full[0], "full"], [ev[0], "ev"]], cls);
    meter(out, "Runs that reach the 32K cap", r[1], pct(r[1]), 100, [[full[1], "full"], [ev[1], "ev"]], cls);
    meter(out, "Mean output length", r[2], (r[2] / 1000).toFixed(1) + "K", 32768, [[full[2], "full"], [ev[2], "ev"]], cls);
    meter(out, "KV memory-time / FullKV", r[3], r[3].toFixed(2) + "×", 0.45, [[ev[3], "ev"]], cls);
    const name = MODELS.find(x => x[0] === state.m)[1];
    $('[data-role="outtitle"]', fig).textContent = `${name} on AIME25, B = ${fmt(state.B)}`;
    const d = r[0] - ev[0];
    $('[data-role="readout"]', fig).innerHTML = quant
      ? `<strong>${state.b}-bit hybrid.</strong> ${acc(r)} against ${acc(ev)} for eviction at the same bytes (${sgn(d)} pp) and ${acc(full)} for FullKV. ` +
        `${pct(r[1])} of its runs reach the cap, against ${pct(ev[1])} for eviction.`
      : `<strong>Eviction only (SnapKV-D).</strong> ${acc(r)}, against ${acc(full)} for FullKV; ${pct(r[1])} of its runs reach the cap (FullKV ${pct(full[1])}). ` +
        `Pick a bit-width to keep more tokens in the same bytes.`;
  }
  render();
})();

/* ---------- every method, setting by setting ---------- */
(function () {
  const fig = $("#bkv-explore");
  if (!fig) return;
  const state = { m: "q3_8b", bench: "aime25", B: 512 };
  const FAM = [
    ["ref", "Reference", [["full", "FullKV, no compression"], ["kivi4", "KIVI-4, quantization without eviction"]]],
    ["ev", "Eviction, tokens kept in fp16", [["snapkv", "SnapKV-D"], ["rkv", "R-KV"], ["random", "Random"], ["vase", "VaSE"], ["streaming", "StreamingLLM"], ["h2o", "H2O"]]],
    ["hy", "Byte-matched hybrids with SnapKV-D eviction", [["hyb4", "4-bit"], ["hyb3", "3-bit"], ["hyb2", "2-bit"]]],
    ["jt", "Other precision–count allocators", [["rdkvblk", "RDKV, block-wise decode mode"], ["thinkvkc", "ThinKV, our re-implementation"]]]
  ];
  const bar = (x, cls) => `<span class="bar"><i class="${cls}" style="width:${Math.max(1, x)}%"></i></span>`;

  function render() {
    const benches = BENCH_ORDER.filter(b => D[state.m][b]);
    if (!benches.includes(state.bench)) state.bench = "aime25";
    seg($('[data-role="model"]', fig), MODELS.map(x => [x[0], x[2]]), state.m, v => { state.m = v; render(); });
    seg($('[data-role="bench"]', fig), benches.map(b => [b, BENCH[b]]), state.bench, v => { state.bench = v; render(); });
    seg($('[data-role="budget"]', fig), [512, 1024].map(B => [B, "B = " + fmt(B)]), state.B, v => { state.B = v; render(); });

    const body = $("tbody", fig);
    body.replaceChildren();
    const pick = "hyb" + CALIB[state.m][state.B];
    for (const [fam, title, rows] of FAM) {
      const head = el("tr", "grp");
      head.appendChild(el("td", "", title));
      head.firstChild.colSpan = 5;
      body.appendChild(head);
      for (const [meth, label] of rows) {
        const r = run(state.m, state.bench, meth, fam === "ref" ? "0" : state.B);
        if (!r) continue;
        const tr = el("tr", meth === pick ? "hl" : "");
        tr.title = `${r[4]} samples`;
        tr.innerHTML = `<td><span class="dot ${fam}"></span>${label}${meth === pick ? ' <span class="tag">BreadthKV</span>' : ""}</td>` +
          `<td><div class="cell">${bar(r[0], fam)}${acc(r)}</div></td>` +
          `<td><div class="cell">${bar(r[1], "cap")}${pct(r[1])}</div></td>` +
          `<td>${(r[2] / 1000).toFixed(1)}K</td><td>${r[3].toFixed(2)}×</td>`;
        body.appendChild(tr);
      }
    }
    const b = run(state.m, state.bench, pick, state.B), s = run(state.m, state.bench, "snapkv", state.B), f = run(state.m, state.bench, "full", "0");
    const name = MODELS.find(x => x[0] === state.m)[1];
    $('[data-role="readout"]', fig).innerHTML = `<strong>${name}, ${BENCH[state.bench]}, B = ${fmt(state.B)}.</strong> ` +
      `BreadthKV (${CALIB[state.m][state.B]}-bit, chosen on the calibration set) scores ${acc(b)} against ${acc(s)} for SnapKV-D, ` +
      `a difference of ${sgn(b[0] - s[0])} pp, with ${(b[3] / s[3]).toFixed(2)}× its KV memory-time. FullKV scores ${acc(f)}.`;
    $('[data-role="note"]', fig).textContent = state.B === 512 ? "StreamingLLM and H2O were run at B = 1,024 only." : "";
  }
  render();
})();

/* ---------- copy BibTeX ---------- */
root.addEventListener("click", e => {
  const b = e.target.closest(".bkv-copy");
  if (!b) return;
  const text = document.getElementById(b.dataset.target).textContent;
  const done = () => { b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy"; }, 1500); };
  if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, () => {});
});
})();
