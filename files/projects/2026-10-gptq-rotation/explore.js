/* Interactive figures of the project page "How Much Does Random Rotation Help GPTQ?".
   Figure A runs in the browser: it draws a Haar rotation, computes the pivots, and evaluates the limit of Theorem 1.
   Figures B and C plot the paper's own data (Figures 1 and 2). Colours and type come from the page's .gr styles. */
(function () {
"use strict";
/* Worst-case curves, rows [log10 kappa, worst case with rotation, q*, worst case without rotation, log-uniform with rotation, log-uniform without]. */
const WORST = [[0.5,0.00316,0.49696,0.11738,0.00036,0.03941],[0.75,0.01516,0.4903,0.25846,0.00179,0.08751],[1.0,0.04457,0.47868,0.44654,0.00548,0.15285],[1.25,0.09953,0.46209,0.67423,0.01288,0.23376],[1.5,0.1861,0.44127,0.93417,0.02555,0.32844],[1.75,0.30732,0.41745,1.2197,0.04497,0.4351],[2.0,0.46325,0.392,1.5253,0.07245,0.55209],[2.25,0.65165,0.36618,1.8463,0.10901,0.67793],[2.5,0.86889,0.34099,2.1794,0.15529,0.81133],[2.75,1.1108,0.31712,2.5218,0.2116,0.9512],[3.0,1.3733,0.29497,2.8717,0.27788,1.0966],[3.25,1.6526,0.27472,3.2276,0.35383,1.2468],[3.5,1.9457,0.25639,3.5883,0.43891,1.4011],[3.75,2.25,0.23989,3.9532,0.53247,1.5591],[4.0,2.5636,0.22507,4.3215,0.63379,1.7202],[4.25,2.8848,0.21179,4.6928,0.74215,1.8841],[4.5,3.2124,0.19985,5.0666,0.85685,2.0506],[4.75,3.5456,0.18911,5.4428,0.97722,2.2192],[5.0,3.8834,0.17942,5.821,1.1027,2.3898],[5.25,4.2255,0.17064,6.201,1.2327,2.5622],[5.5,4.5712,0.16266,6.5826,1.3668,2.7363],[5.75,4.9202,0.15539,6.9658,1.5046,2.9119],[6.0,5.2722,0.14873,7.3503,1.6457,3.0888],[6.25,5.6269,0.14262,7.7361,1.7899,3.267],[6.5,5.9841,0.13698,8.1231,1.9368,3.4463],[6.75,6.3435,0.13178,8.5111,2.0862,3.6267],[7.0,6.705,0.12695,8.9001,2.2379,3.8081],[7.25,7.0685,0.12247,9.29,2.3918,3.9904],[7.5,7.4337,0.11829,9.6808,2.5476,4.1735],[7.75,7.8007,0.11439,10.072,2.7053,4.3575],[8.0,8.1692,0.11074,10.465,2.8647,4.5422],[8.25,8.5391,0.10731,10.858,3.0256,4.7276],[8.5,8.9105,0.10409,11.251,3.1881,4.9137],[8.75,9.2831,0.10106,11.646,3.3519,5.1005],[9.0,9.657,0.0982,12.041,3.517,5.2877],[9.25,10.032,0.0955,12.436,3.6834,5.4756],[9.5,10.408,0.09294,12.832,3.8509,5.664],[9.75,10.785,0.09052,13.229,4.0195,5.8529],[10.0,11.163,0.08822,13.626,4.1892,6.0422],[10.25,11.543,0.08603,14.023,4.3598,6.232],[10.5,11.922,0.08395,14.421,4.5313,6.4223],[10.75,12.303,0.08197,14.819,4.7038,6.6129],[11.0,12.685,0.08008,15.218,4.8771,6.804],[11.25,13.067,0.07827,15.617,5.0512,6.9954],[11.5,13.45,0.07655,16.016,5.226,7.1871],[11.75,13.834,0.07489,16.416,5.4016,7.3792],[12.0,14.219,0.07331,16.816,5.5779,7.5717]];
/* 96 real Hessians at damping 1e-2, rows [model, input type, log10 kappa, limit of Theorem 1, Haar, Hadamard x1, Hadamard x2,
   native basis, PCA basis, worst case with rotation at this kappa, worst case without rotation]. Gaps in bit. */
const LAYERS = [[1,1,4.4955,0.035329,0.036333,0.035717,0.036245,0.15929,1.011,3.2065,5.0599],[1,2,4.6552,0.12815,0.12988,0.13842,0.11935,0.38727,1.1698,3.4187,5.2999],[1,3,4.7363,0.1597,0.16126,0.16059,0.16086,0.44619,1.1674,3.5271,5.422],[1,4,3.5935,0.053686,0.05395,0.053323,0.053858,0.2005,0.71965,2.0583,3.7244],[1,1,4.4251,0.027925,0.029735,0.027943,0.028561,0.38235,1.0594,3.1137,4.9544],[1,2,4.161,0.059756,0.060788,0.066056,0.059349,0.20587,0.92036,2.7696,4.5602],[1,3,4.9667,0.018701,0.020193,0.01905,0.019396,0.29046,0.97049,3.8382,5.7705],[1,4,3.9854,0.057058,0.05725,0.056736,0.057228,0.29195,0.65939,2.545,4.2999],[1,1,4.2355,0.029837,0.030933,0.029746,0.030606,0.23025,0.90265,2.8659,4.6711],[1,2,4.2174,0.048725,0.049585,0.051648,0.049069,0.11867,0.96667,2.8425,4.6442],[1,3,4.7843,0.019288,0.019812,0.01941,0.019809,0.11313,0.72274,3.5916,5.4945],[1,4,3.8548,0.041144,0.041358,0.040657,0.041313,0.23144,0.58182,2.3804,4.1071],[1,1,4.9575,0.04851,0.04991,0.048034,0.05036,0.23171,1.2437,3.8257,5.7565],[1,2,4.5024,0.067705,0.069086,0.068732,0.066259,0.14647,1.1887,3.2156,5.0702],[1,3,4.7656,0.02628,0.026971,0.026623,0.026929,0.083829,0.77291,3.5665,5.4663],[1,4,4.3093,0.056985,0.057194,0.056152,0.057234,0.32375,0.76105,2.9619,4.7812],[1,1,4.9306,0.043588,0.045167,0.042971,0.045645,0.24434,1.2335,3.7892,5.7157],[1,2,4.4339,0.093412,0.09493,0.093258,0.089644,0.34931,1.176,3.1252,4.9675],[1,3,4.7392,0.025844,0.02635,0.026052,0.026594,0.097938,0.79708,3.531,5.4264],[1,4,4.5349,0.059487,0.059758,0.058291,0.059711,0.35726,0.79595,3.2586,5.119],[1,1,4.8025,0.063501,0.064371,0.062719,0.065031,0.36233,1.3229,3.6162,5.5221],[1,2,4.5635,0.12497,0.12599,0.13637,0.12029,0.33528,1.2005,3.2965,5.1619],[1,3,4.7376,0.037734,0.038459,0.038057,0.03885,0.1462,0.89803,3.5289,5.4241],[1,4,4.5687,0.10198,0.10242,0.097261,0.10225,0.54713,0.97677,3.3035,5.1698],[1,1,4.8681,0.06699,0.067867,0.066241,0.068925,0.46942,1.3197,3.7046,5.6212],[1,2,4.2179,0.061172,0.062247,0.066813,0.06005,0.18492,0.90243,2.8431,4.6449],[1,3,4.7038,0.02574,0.026307,0.026229,0.026543,0.086867,0.74489,3.4836,5.373],[1,4,4.6465,0.076729,0.076829,0.073721,0.076943,0.52098,0.83911,3.4071,5.2868],[1,1,4.5945,0.019899,0.020418,0.020253,0.020713,0.11182,0.80993,3.3377,5.2085],[1,2,4.5751,0.10795,0.10869,0.094835,0.10329,0.43254,0.92424,3.312,5.1795],[1,3,4.9684,0.021332,0.022148,0.021454,0.022951,0.25682,1.1188,3.8405,5.7731],[1,4,5.6733,0.041968,0.047523,0.044182,0.042772,0.56478,2.3347,4.8128,6.848],[2,1,4.7798,0.086493,0.087051,0.086893,0.087132,0.11097,1.0492,3.5856,5.4877],[2,2,5.0383,0.1029,0.10358,0.1023,0.10211,0.31966,1.2805,3.9356,5.879],[2,3,4.4132,0.11267,0.11313,0.11304,0.11308,0.17033,0.94764,3.0979,4.9365],[2,4,4.153,0.11426,0.11436,0.1112,0.11443,0.46407,0.97208,2.7594,4.5484],[2,1,5.1392,0.04815,0.048988,0.048386,0.04876,0.22894,1.0784,4.0734,6.0323],[2,2,4.6609,0.091661,0.092079,0.095013,0.087608,0.23064,0.98018,3.4263,5.3085],[2,3,5.4703,0.025662,0.027612,0.026542,0.026337,0.13172,1.5678,4.5299,6.5371],[2,4,4.4126,0.073731,0.073826,0.072694,0.073867,0.29635,0.68389,3.0972,4.9356],[2,1,5.3281,0.026518,0.027256,0.026854,0.026993,0.19855,1.0891,4.3331,6.3201],[2,2,4.2271,0.070287,0.070561,0.075952,0.069529,0.21844,0.91439,2.855,4.6586],[2,3,4.9334,0.017577,0.01787,0.017811,0.017759,0.038489,0.55833,3.793,5.7201],[2,4,3.8376,0.043018,0.043128,0.04285,0.043104,0.18799,0.60788,2.359,4.0819],[2,1,5.0636,0.039405,0.040308,0.039866,0.040175,0.29553,1.1924,3.97,5.9174],[2,2,4.4467,0.062396,0.063077,0.062331,0.062597,0.15382,1.0493,3.142,4.9866],[2,3,4.8299,0.018153,0.018396,0.018373,0.018443,0.039753,0.5374,3.653,5.5633],[2,4,3.8721,0.018146,0.018192,0.01819,0.018214,0.094827,0.4208,2.4022,4.1327],[2,1,4.8754,0.03393,0.034577,0.034383,0.034618,0.29536,1.1237,3.7145,5.6323],[2,2,4.3237,0.067839,0.068398,0.070872,0.066271,0.19274,1.0246,2.9807,4.8026],[2,3,4.828,0.016509,0.016751,0.016725,0.01674,0.050492,0.54212,3.6505,5.5606],[2,4,3.5683,0.019814,0.019895,0.019669,0.019885,0.13339,0.43976,2.0278,3.6877],[2,1,4.6931,0.040233,0.040669,0.040389,0.040785,0.29221,1.126,3.4693,5.357],[2,2,4.2288,0.063988,0.064309,0.066658,0.063656,0.17478,0.87628,2.8572,4.6611],[2,3,4.8579,0.02429,0.024491,0.024556,0.024487,0.08232,0.67669,3.6909,5.6058],[2,4,4.0811,0.048151,0.04822,0.047239,0.048249,0.29364,0.64837,2.667,4.4417],[2,1,4.557,0.040589,0.041082,0.040706,0.041006,0.30121,1.0539,3.2879,5.1522],[2,2,4.7077,0.069975,0.070498,0.074534,0.067144,0.20469,0.90854,3.4889,5.379],[2,3,4.7486,0.02242,0.022613,0.022702,0.022593,0.061675,0.62337,3.5437,5.4406],[2,4,4.3031,0.045125,0.045209,0.044103,0.045237,0.32428,0.70067,2.9539,4.772],[2,1,4.7743,0.021477,0.021949,0.021797,0.021894,0.16508,0.98645,3.5782,5.4794],[2,2,5.0023,0.11368,0.11417,0.13758,0.10693,0.41762,1.1877,3.8866,5.8245],[2,3,5.4048,0.017789,0.018357,0.018921,0.018368,0.32744,1.355,4.4392,6.4372],[2,4,6.0713,0.007641,0.0084958,0.0073841,0.0089586,0.52545,3.0333,5.3731,7.4602],[3,1,5.0704,0.35339,0.35606,0.32712,0.35515,1.3764,2.2288,3.9793,5.9278],[3,2,5.1184,0.27548,0.27748,0.3036,0.24916,0.87846,1.8735,4.0449,6.0007],[3,3,4.3849,0.026109,0.026313,0.026432,0.026329,0.042541,0.54132,3.0608,4.8941],[3,4,4.1174,0.046349,0.046467,0.046098,0.046466,0.20558,0.70779,2.7136,4.4956],[3,1,4.9417,0.031663,0.032096,0.032054,0.032052,0.18931,0.8808,3.8043,5.7326],[3,2,4.6635,0.12099,0.12169,0.13327,0.12046,0.29793,1.22,3.4298,5.3125],[3,3,4.4658,0.016041,0.016222,0.016206,0.016234,0.02685,0.44784,3.1673,5.0154],[3,4,3.9626,0.023777,0.023843,0.023767,0.023832,0.10648,0.51112,2.5162,4.2662],[3,1,4.5328,0.026657,0.0271,0.027016,0.026959,0.12587,0.81541,3.2559,5.1159],[3,2,4.6145,0.09098,0.091528,0.089658,0.090635,0.24625,1.2112,3.3644,5.2386],[3,3,3.8963,0.017441,0.017671,0.01753,0.01763,0.028238,0.48908,2.4324,4.1683],[3,4,3.9739,0.023496,0.02355,0.023557,0.023559,0.079366,0.53613,2.5304,4.2828],[3,1,4.5363,0.027529,0.027984,0.02781,0.027877,0.1548,0.85994,3.2604,5.1211],[3,2,4.5775,0.089663,0.090221,0.089001,0.089354,0.17625,1.3112,3.3151,5.183],[3,3,3.9754,0.019321,0.019524,0.01945,0.019565,0.043727,0.52798,2.5324,4.2852],[3,4,4.2216,0.025794,0.025842,0.025782,0.025865,0.091312,0.58919,2.848,4.6505],[3,1,4.6232,0.032451,0.032956,0.032603,0.032841,0.21296,0.83997,3.3759,5.2517],[3,2,4.5139,0.090968,0.091481,0.096684,0.087637,0.22337,0.96716,3.2308,5.0874],[3,3,3.7881,0.020066,0.020282,0.020207,0.020329,0.032461,0.49613,2.2973,4.0091],[3,4,4.0743,0.024956,0.025039,0.024854,0.025021,0.13196,0.55454,2.6583,4.4315],[3,1,4.6122,0.032671,0.03318,0.032822,0.032996,0.24954,0.87994,3.3613,5.2351],[3,2,4.5717,0.073947,0.074387,0.075177,0.072273,0.22531,0.95273,3.3074,5.1743],[3,3,3.8065,0.01797,0.018147,0.018174,0.018145,0.02534,0.46643,2.3202,4.0362],[3,4,3.8802,0.021993,0.022042,0.021862,0.022054,0.14127,0.53491,2.4123,4.1446],[3,1,4.7846,0.023454,0.023855,0.023608,0.023718,0.17102,0.74223,3.5921,5.4951],[3,2,4.9553,0.086584,0.087126,0.089478,0.086535,0.24736,1.2896,3.8228,5.7533],[3,3,3.962,0.012867,0.013042,0.01308,0.013025,0.017032,0.44064,2.5154,4.2653],[3,4,4.4965,0.01547,0.015548,0.015489,0.015568,0.15868,0.66988,3.2078,5.0614],[3,1,4.7507,0.024018,0.024547,0.024165,0.024297,0.11875,0.73079,3.5465,5.4439],[3,2,5.3142,0.12167,0.12213,0.13097,0.11957,0.39415,1.8066,4.3139,6.2988],[3,3,4.6786,0.018222,0.018486,0.018449,0.018471,0.037558,0.55342,3.4499,5.3351],[3,4,5.6768,0.10103,0.10165,0.099662,0.10205,0.74319,1.975,4.8177,6.8534]];
/* ---------- maths ---------- */
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function gaussian(rng) {
  let u = 0;
  while (u === 0) u = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}
/* A spectrum is a list of atoms x with weights w (summing to 1). */
function means(sp) {
  let am = 0, lg = 0, ih = 0, lo = Infinity, hi = 0;
  for (let i = 0; i < sp.x.length; i++) {
    am += sp.w[i] * sp.x[i]; lg += sp.w[i] * Math.log(sp.x[i]); ih += sp.w[i] / sp.x[i];
    if (sp.x[i] < lo) lo = sp.x[i];
    if (sp.x[i] > hi) hi = sp.x[i];
  }
  return { am: am, gm: Math.exp(lg), hm: 1 / ih, lo: lo, hi: hi };
}
/* Pivot profile h_nu(t) = t s_t / (1 - t), where s_t solves T_nu(s) = integral of x / (s + x) = t  (Definition 1 of the paper). */
function profile(sp, m, t) {
  if (t <= 0) return m.am;
  if (t >= 1) return m.hm;
  const x = sp.x, w = sp.w, len = x.length;
  let lo = Math.log(m.lo * (1 - t) / t), hi = Math.log(m.hi * (1 - t) / t);
  for (let it = 0; it < 60 && hi - lo > 1e-13; it++) {
    const mid = 0.5 * (lo + hi), s = Math.exp(mid);
    let T = 0;
    for (let i = 0; i < len; i++) T += w[i] * x[i] / (s + x[i]);
    if (T > t) lo = mid; else hi = mid;
  }
  return t * Math.exp(0.5 * (lo + hi)) / (1 - t);
}
/* D(nu) = integral of h_nu over (0, 1), by the midpoint rule. */
function meanProfile(sp, m, N) {
  let d = 0;
  for (let j = 0; j < N; j++) d += profile(sp, m, (j + 0.5) / N);
  return d / N;
}
function limitGap(sp, N) {
  const m = means(sp);
  return 0.5 * Math.log2(meanProfile(sp, m, N || 2048) / m.gm);
}
/* Pivots of V' diag(lam) V for a Haar orthogonal V: squared Gram-Schmidt lengths of the vectors diag(lam)^(1/2) v_k. */
function haarPivots(lam, seed) {
  const n = lam.length, rng = mulberry32(seed), V = new Float64Array(n * n), piv = new Float64Array(n);
  for (let i = 0; i < n * n; i++) V[i] = gaussian(rng);
  for (let i = 0; i < n; i++) {
    const a = i * n;
    for (let pass = 0; pass < 2; pass++) {
      for (let j = 0; j < i; j++) {
        const b = j * n; let d = 0;
        for (let k = 0; k < n; k++) d += V[a + k] * V[b + k];
        for (let k = 0; k < n; k++) V[a + k] -= d * V[b + k];
      }
    }
    let s = 0;
    for (let k = 0; k < n; k++) s += V[a + k] * V[a + k];
    s = 1 / Math.sqrt(s);
    for (let k = 0; k < n; k++) V[a + k] *= s;
  }
  const sq = new Float64Array(n);
  for (let k = 0; k < n; k++) sq[k] = Math.sqrt(lam[k]);
  for (let i = 0; i < n; i++) { const a = i * n; for (let k = 0; k < n; k++) V[a + k] *= sq[k]; }
  for (let i = 0; i < n; i++) {
    const a = i * n;
    for (let j = 0; j < i; j++) {
      const b = j * n; let d = 0;
      for (let k = 0; k < n; k++) d += V[a + k] * V[b + k];
      for (let k = 0; k < n; k++) V[a + k] -= d * V[b + k];
    }
    let s = 0;
    for (let k = 0; k < n; k++) s += V[a + k] * V[a + k];
    piv[i] = s;
    s = 1 / Math.sqrt(s);
    for (let k = 0; k < n; k++) V[a + k] *= s;
  }
  return piv;
}
function gapOfPivots(piv) {
  let am = 0, lg = 0;
  for (let i = 0; i < piv.length; i++) { am += piv[i]; lg += Math.log(piv[i]); }
  return 0.5 * Math.log2(am / piv.length / Math.exp(lg / piv.length));
}
/* ---------- page ---------- */
const root = typeof document === "undefined" ? null : document.querySelector(".gr");
if (!root) return;
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
function S(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs || {}) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function T(parent, x, y, s, cls, anchor) {
  const t = S("text", { x: x, y: y, class: cls, "text-anchor": anchor || "start" }, parent);
  t.textContent = s;
  return t;
}
const SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const pow10 = e => "10" + String(e).split("").map(c => SUP[c] || c).join("");
function sci(x) {
  if (x === 0) return "0";
  let e = Math.floor(Math.log10(Math.abs(x)) + 1e-9), m = x / Math.pow(10, e);
  if (Math.abs(m - 10) < 0.05) { m = 1; e += 1; }
  if (e >= -2 && e <= 3) return String(Number(x.toPrecision(3)));
  return (Math.abs(m - 1) < 0.05 ? "" : m.toFixed(1) + "×") + pow10(e);
}
function bits(x, signed) {
  const a = Math.abs(x), s = a < 0.00995 ? a.toFixed(4) : a < 0.0995 ? a.toFixed(3) : a.toFixed(2);
  return (x < 0 && Number(s) !== 0 ? "−" : signed && Number(s) !== 0 ? "+" : "") + s;
}
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
function logTicks(lo, hi) {
  const out = [];
  for (let e = Math.ceil(lo - 1e-9); e <= Math.floor(hi + 1e-9); e++) out.push({ v: e, label: pow10(e) });
  if (out.length >= 3) {
    const step = Math.ceil(out.length / 7);
    return out.filter((t, i) => i % step === 0);
  }
  const fine = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e++) {
    for (const m of [1, 2, 5]) {
      const v = e + Math.log10(m);
      if (v >= lo - 1e-9 && v <= hi + 1e-9) fine.push({ v: v, label: sci(m * Math.pow(10, e)) });
    }
  }
  return fine;
}
function niceTicks(lo, hi, target) {
  const raw = (hi - lo) / target, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  const step = (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p, out = [];
  for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9 * step; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return { ticks: out, step: step };
}
const tip = document.createElement("div");
tip.className = "gr-tip";
tip.hidden = true;
root.appendChild(tip);
function showTip(ev, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > window.innerWidth - 8) x = ev.clientX - w - 14;
  if (y + h > window.innerHeight - 8) y = ev.clientY - h - 14;
  tip.style.left = Math.max(4, x) + "px";
  tip.style.top = Math.max(4, y) + "px";
}
function hideTip() { tip.hidden = true; }
const row = (cls, name, val) => '<div class="tr">' + (cls ? '<i class="' + cls + '"></i>' : "") + "<span>" + name + "</span><b>" + val + "</b></div>";
function seg(host, attr, onPick) {
  host.addEventListener("click", ev => {
    const b = ev.target.closest("button");
    if (!b || !host.contains(b)) return;
    host.querySelectorAll("button").forEach(o => o.setAttribute("aria-pressed", o === b ? "true" : "false"));
    onPick(b.getAttribute(attr));
  });
}
function widthOf(host) { return Math.max(280, Math.floor(host.clientWidth || 640)); }
function interpWorst(lk, col) {
  const a = WORST;
  if (lk < a[0][0] || lk > a[a.length - 1][0]) return null;
  for (let i = 1; i < a.length; i++) {
    if (lk <= a[i][0]) {
      const f = (lk - a[i - 1][0]) / (a[i][0] - a[i - 1][0]);
      return a[i - 1][col] + f * (a[i][col] - a[i - 1][col]);
    }
  }
  return null;
}
const MODELS = ["", "Qwen3-1.7B", "Qwen3-8B-Base", "Llama-3.1-8B-Instruct"];
const TYPES = ["", "q/k/v input", "o_proj input", "gate/up input", "down_proj input"];

/* ---------- figure A: rotate a spectrum ---------- */
const A = { fam: "two", lk: 4, q: 0.225, r: 4, lmu: 2, n: 256, seed: 20261101, res: null };
function buildSpectrum() {
  const n = A.n, lam = new Float64Array(n);
  let sp, kappa, m1 = 0;
  if (A.fam === "two") {
    kappa = Math.pow(10, A.lk);
    m1 = clamp(Math.round(A.q * n), 1, n - 1);
    for (let i = 0; i < n; i++) lam[i] = i < m1 ? 1 : 1 / kappa;
    sp = { x: [1 / kappa, 1], w: [1 - m1 / n, m1 / n] };
  } else if (A.fam === "logu") {
    kappa = Math.pow(10, A.lk);
    for (let i = 0; i < n; i++) lam[i] = Math.pow(kappa, -(i + 0.5) / n);
    sp = { x: Array.from(lam), w: Array.from(lam, () => 1 / n) };
  } else {
    const mu = Math.pow(10, A.lmu), r = A.r;
    for (let i = 0; i < n; i++) lam[i] = i < r ? mu : 1 - 0.5 * (i - r + 0.5) / (n - r);
    sp = { x: Array.from(lam), w: Array.from(lam, () => 1 / n) };
    kappa = r > 0 ? Math.max(mu, 1) / 0.5 : 2;
  }
  return { lam: lam, sp: sp, kappa: kappa, m1: m1 };
}
function computeA() {
  const b = buildSpectrum(), n = A.n, m = means(b.sp);
  const piv = haarPivots(b.lam, A.seed), h = new Float64Array(n);
  for (let k = 1; k <= n; k++) h[k - 1] = profile(b.sp, m, k / n);
  let meanPiv = 0;
  for (let i = 0; i < n; i++) meanPiv += piv[i];
  meanPiv /= n;
  const D = meanProfile(b.sp, m, b.sp.x.length > 2 ? 1024 : 4096);
  A.res = { lam: b.lam, kappa: b.kappa, m1: b.m1, m: m, piv: piv, h: h, meanPiv: meanPiv, D: D,
    gNone: 0.5 * Math.log2(m.am / m.gm), gSim: gapOfPivots(piv), gLim: 0.5 * Math.log2(D / m.gm) };
}
function drawA() {
  const R = A.res, host = $("gr-piv-chart");
  if (!R) return;
  const n = A.n, W = widthOf(host), wide = W >= 600, H = wide ? 340 : 300;
  const mL = 46, mR = wide ? 158 : 12, mT = 14, mB = 40, pw = W - mL - mR, ph = H - mT - mB;
  let lo = Infinity, hi = 0;
  for (let i = 0; i < n; i++) {
    lo = Math.min(lo, R.lam[i], R.piv[i]);
    hi = Math.max(hi, R.lam[i], R.piv[i]);
  }
  let l0 = Math.log10(lo), l1 = Math.log10(hi);
  const pad = Math.max(0.06, 0.04 * (l1 - l0));
  l0 -= pad; l1 += pad;
  const X = t => mL + t * pw, Y = v => mT + (l1 - Math.log10(v)) / (l1 - l0) * ph;
  host.textContent = "";
  const svg = S("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
    "aria-label": "Pivots after a Haar rotation against the limit of Theorem 1, with the pivots without rotation" }, host);
  logTicks(l0, l1).forEach(t => {
    const y = mT + (l1 - t.v) / (l1 - l0) * ph;
    S("line", { x1: mL, x2: mL + pw, y1: y, y2: y, class: "gl" }, svg);
    T(svg, mL - 7, y + 4, t.label, "tk", "end");
  });
  [0, 0.25, 0.5, 0.75, 1].forEach(t => T(svg, X(t), mT + ph + 16, String(t), "tk", "middle"));
  S("line", { x1: mL, x2: mL + pw, y1: mT + ph, y2: mT + ph, class: "ax" }, svg);
  T(svg, mL + pw / 2, H - 6, "Gram–Schmidt coordinate k / n", "al", "middle");
  const tl = T(svg, 0, 0, "pivot πₖ (log scale)", "al", "middle");
  tl.setAttribute("transform", "translate(11," + (mT + ph / 2) + ") rotate(-90)");
  /* pivots without rotation: the eigenvalues, largest first */
  const srt = Array.from(R.lam).sort((a, b) => b - a);
  let d = "";
  for (let i = 0; i < n; i++) d += (i ? "L" : "M") + X(i / n).toFixed(1) + "," + Y(srt[i]).toFixed(1) + "L" + X((i + 1) / n).toFixed(1) + "," + Y(srt[i]).toFixed(1);
  S("path", { d: d, class: "ln none" }, svg);
  /* the three levels */
  const levels = [["none", R.m.am, "mean pivot, no rotation"], ["rot", R.meanPiv, "mean pivot, rotated"], ["gm", R.m.gm, "geometric mean (any basis)"]];
  levels.forEach(l => S("line", { x1: mL, x2: mL + pw, y1: Y(l[1]), y2: Y(l[1]), class: "lv " + l[0] }, svg));
  /* measured pivots and the limit */
  const g = S("g", null, svg);
  for (let i = 0; i < n; i++) S("circle", { cx: X((i + 1) / n).toFixed(1), cy: Y(R.piv[i]).toFixed(1), r: n > 128 ? 1.9 : 2.4, class: "pt rot" }, g);
  d = "";
  for (let i = 0; i < n; i++) d += (i ? "L" : "M") + X((i + 1) / n).toFixed(1) + "," + Y(R.h[i]).toFixed(1);
  S("path", { d: d, class: "ln lim" }, svg);
  if (wide) {
    const ys = levels.map(l => Y(l[1]) + 4), order = [0, 1, 2].sort((a, b) => ys[a] - ys[b]);
    for (let i = 1; i < 3; i++) if (ys[order[i]] - ys[order[i - 1]] < 15) ys[order[i]] = ys[order[i - 1]] + 15;
    const over = ys[order[2]] - (mT + ph + 4);
    if (over > 0) for (let i = 2; i >= 0; i--) { ys[order[i]] -= over; if (i > 0 && ys[order[i]] - ys[order[i - 1]] >= 15) break; }
    levels.forEach((l, i) => {
      S("line", { x1: mL + pw, x2: mL + pw + 8, y1: Y(l[1]), y2: ys[i] - 4, class: "lv " + l[0] }, svg);
      T(svg, mL + pw + 11, ys[i], l[2], "lb");
    });
  }
  const cross = S("line", { y1: mT, y2: mT + ph, class: "cross", visibility: "hidden" }, svg);
  const ring = S("circle", { r: 4.5, class: "ring", visibility: "hidden" }, svg);
  const hit = S("rect", { x: mL, y: mT, width: pw, height: ph, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const bx = svg.getBoundingClientRect(), k = clamp(Math.round((ev.clientX - bx.left - mL) / pw * n), 1, n);
    cross.setAttribute("x1", X(k / n)); cross.setAttribute("x2", X(k / n)); cross.setAttribute("visibility", "visible");
    ring.setAttribute("cx", X(k / n)); ring.setAttribute("cy", Y(R.piv[k - 1])); ring.setAttribute("visibility", "visible");
    showTip(ev, '<div class="tt">coordinate k = ' + k + " of " + n + "</div>" + row("k-rot", "after rotation", sci(R.piv[k - 1])) +
      row("k-lim", "limit h(k/n)", sci(R.h[k - 1])) + row("k-none", "no rotation", sci(srt[k - 1])));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); ring.setAttribute("visibility", "hidden"); hideTip(); });
}
function readoutA() {
  const R = A.res, lk = Math.log10(R.kappa), w = interpWorst(lk, 1);
  $("gr-ro-none").textContent = bits(R.gNone);
  $("gr-ro-sim").textContent = bits(R.gSim);
  $("gr-ro-lim").textContent = bits(R.gLim);
  $("gr-ro-worst").textContent = w === null ? "–" : bits(w);
  $("gr-ro-sim-lab").textContent = "after one Haar rotation, n = " + A.n + " (this draw)";
  $("gr-ro-worst-lab").textContent = "worst case over all spectra with κ = " + sci(R.kappa) + " (Theorem 2)";
  $("gr-kappa-out").textContent = sci(Math.pow(10, A.lk));
  $("gr-q-out").textContent = (R.m1 / A.n).toFixed(3) + " (" + R.m1 + " of " + A.n + ")";
  $("gr-r-out").textContent = String(A.r);
  $("gr-mu-out").textContent = sci(Math.pow(10, A.lmu)) + "× the top of the bulk";
  let note;
  if (A.fam === "two") {
    const qs = interpWorst(A.lk, 2);
    note = "<strong>Two-point spectrum.</strong> A share q of the eigenvalues equals 1 and the rest equal 1/κ. After rotation the first q·n coordinates still carry pivots near (q − t)/(1 − t), far above the rest, and one common step size cannot adapt to them. " +
      (qs !== null && Math.abs(A.q - qs) < 0.012 ? "This q is the worst one for this κ: no spectrum with this condition number has a larger rotated gap." : "The worst q for this κ is " + (qs === null ? "–" : qs.toFixed(3)) + ".");
  } else if (A.fam === "logu") {
    note = "<strong>Log-uniform spectrum.</strong> Eigenvalues spread evenly on a logarithmic scale between 1/κ and 1. Rotation lowers the mean pivot, but the gap keeps its leading term ¼ log₂ κ: spread on a log scale is not removed (Proposition 4).";
  } else {
    let sm = 0;
    for (let i = 0; i < A.r; i++) sm += Math.pow(10, A.lmu);
    const c = (A.r + Math.log(A.n)) * sm / (A.n * A.n);
    note = "<strong>Narrow bulk with a few spikes.</strong> The bulk lies in [0.5, 1], as on real layers. Without rotation every spike raises the mean pivot. After rotation the first few coordinates absorb the spikes and the rest follow the bulk. Proposition 3 needs (r + ln n)·Σμ / n² to be small; here it is " +
      (c < 0.001 ? c.toExponential(1).replace("e-", "×10⁻").replace(/⁻(\d)$/, (s, dgt) => "⁻" + SUP[dgt]) : Number(c.toPrecision(2))) + (c > 0.3 ? ", so at this n the spikes still matter and the gap changes from draw to draw." : ".");
  }
  $("gr-piv-note").innerHTML = note;
}
function showParams() {
  root.querySelectorAll("[data-fam-only]").forEach(e => { e.hidden = e.getAttribute("data-fam-only").split(" ").indexOf(A.fam) < 0; });
}
let pendA = 0;
function updateA(now) {
  clearTimeout(pendA);
  const run = () => { computeA(); drawA(); readoutA(); };
  if (now) run(); else pendA = setTimeout(run, 40);
}
function initA() {
  if (!$("gr-piv-chart")) return;
  seg($("gr-fam"), "data-fam", v => { A.fam = v; showParams(); updateA(true); });
  seg($("gr-n"), "data-n", v => { A.n = Number(v); updateA(true); });
  $("gr-kappa").addEventListener("input", ev => { A.lk = Number(ev.target.value); updateA(); });
  $("gr-q").addEventListener("input", ev => { A.q = Number(ev.target.value); updateA(); });
  $("gr-r").addEventListener("input", ev => { A.r = Number(ev.target.value); updateA(); });
  $("gr-mu").addEventListener("input", ev => { A.lmu = Number(ev.target.value); updateA(); });
  $("gr-qstar").addEventListener("click", () => {
    const qs = interpWorst(A.lk, 2);
    if (qs === null) return;
    A.q = Math.round(qs * 200) / 200;
    $("gr-q").value = A.q;
    updateA(true);
  });
  $("gr-redraw").addEventListener("click", () => { A.seed = (A.seed * 1664525 + 1013904223) | 0; updateA(true); });
  showParams();
  updateA(true);
}

/* ---------- figure B: worst case against real layers ---------- */
function drawB() {
  const host = $("gr-worst-chart");
  if (!host) return;
  const W = widthOf(host), H = W >= 600 ? 380 : 320, mL = 46, mR = 14, mT = 12, mB = 40, pw = W - mL - mR, ph = H - mT - mB;
  const x0 = 1, x1 = 12, y0 = Math.log10(0.005), y1 = Math.log10(30);
  const X = v => mL + (v - x0) / (x1 - x0) * pw, Y = v => mT + (y1 - Math.log10(v)) / (y1 - y0) * ph;
  host.textContent = "";
  const svg = S("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
    "aria-label": "Rate gap of GPTQ to WaterSIC against the condition number: worst cases with and without rotation, the log-uniform spectrum, and 96 real layers" }, host);
  [0.01, 0.1, 1, 10].forEach(v => {
    S("line", { x1: mL, x2: mL + pw, y1: Y(v), y2: Y(v), class: "gl" }, svg);
    T(svg, mL - 7, Y(v) + 4, String(v), "tk", "end");
  });
  for (let e = 2; e <= 12; e += 2) T(svg, X(e), mT + ph + 16, pow10(e), "tk", "middle");
  S("line", { x1: mL, x2: mL + pw, y1: mT + ph, y2: mT + ph, class: "ax" }, svg);
  T(svg, mL + pw / 2, H - 6, "condition number κ", "al", "middle");
  const tl = T(svg, 0, 0, "gap to WaterSIC (bit per weight, log scale)", "al", "middle");
  tl.setAttribute("transform", "translate(11," + (mT + ph / 2) + ") rotate(-90)");
  const clip = S("clipPath", { id: "gr-clip-b" }, S("defs", null, svg));
  S("rect", { x: mL, y: mT, width: pw, height: ph }, clip);
  const gl = S("g", { "clip-path": "url(#gr-clip-b)" }, svg);
  [[3, "ln none"], [1, "ln rot"], [5, "ln none dash"], [4, "ln rot dash"]].forEach(c => {
    let d = "", pen = false;
    WORST.forEach(r => {
      if (r[0] < x0 || r[c[0]] <= 0) return;
      d += (pen ? "L" : "M") + X(r[0]).toFixed(1) + "," + Y(Math.max(r[c[0]], 1e-4)).toFixed(1);
      pen = true;
    });
    S("path", { d: d, class: c[1] }, gl);
  });
  const pts = [];
  LAYERS.forEach(r => {
    pts.push({ r: r, x: X(r[2]), y: Y(r[8]), kind: "pca" });
    pts.push({ r: r, x: X(r[2]), y: Y(r[4]), kind: "haar" });
  });
  pts.filter(p => p.kind === "pca").forEach(p => S("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 3.2, class: "pt open" }, svg));
  pts.filter(p => p.kind === "haar").forEach(p => S("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 3.4, class: "pt rot ringed" }, svg));
  const cross = S("line", { y1: mT, y2: mT + ph, class: "cross", visibility: "hidden" }, svg);
  const ring = S("circle", { r: 6, class: "ring", visibility: "hidden" }, svg);
  const hit = S("rect", { x: mL, y: mT, width: pw, height: ph, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const bx = svg.getBoundingClientRect(), px = ev.clientX - bx.left, py = ev.clientY - bx.top;
    let best = null, bd = 16 * 16;
    pts.forEach(p => { const dd = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py); if (dd < bd) { bd = dd; best = p; } });
    if (best) {
      const r = best.r;
      cross.setAttribute("visibility", "hidden");
      ring.setAttribute("cx", best.x); ring.setAttribute("cy", best.y); ring.setAttribute("visibility", "visible");
      showTip(ev, '<div class="tt">' + MODELS[r[0]] + " · " + TYPES[r[1]] + " · κ = " + sci(Math.pow(10, r[2])) + "</div>" +
        row("k-open", "worst (PCA) basis", bits(r[8]) + " bit") + row("k-rot", "after Haar rotation", bits(r[4]) + " bit") +
        row("k-none", "worst case, no rotation", bits(r[10]) + " bit") + row("k-rotl", "worst case, rotation", bits(r[9]) + " bit"));
      return;
    }
    const lk = clamp(x0 + (px - mL) / pw * (x1 - x0), x0, x1);
    ring.setAttribute("visibility", "hidden");
    cross.setAttribute("x1", X(lk)); cross.setAttribute("x2", X(lk)); cross.setAttribute("visibility", "visible");
    const wn = interpWorst(lk, 3), wr = interpWorst(lk, 1);
    showTip(ev, '<div class="tt">κ = ' + sci(Math.pow(10, lk)) + "</div>" + row("k-none", "worst case, no rotation", bits(wn) + " bit") +
      row("k-rotl", "worst case, rotation", bits(wr) + " bit") + row("", "rotation saves", bits(wn - wr) + " bit") +
      row("k-none dash", "log-uniform, no rotation", bits(interpWorst(lk, 5)) + " bit") + row("k-rotl dash", "log-uniform, rotation", bits(interpWorst(lk, 4)) + " bit"));
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); ring.setAttribute("visibility", "hidden"); hideTip(); });
}

/* ---------- figure C: the limit on real Hessians ---------- */
const C = { kind: 4, view: "gap" };
const KINDS = { 4: "Haar", 5: "Hadamard, one round", 6: "Hadamard, two rounds" };
function drawC() {
  const host = $("gr-real-chart");
  if (!host) return;
  const W = widthOf(host), H = W >= 600 ? 380 : 320, mL = 52, mR = 14, mT = 12, mB = 40, pw = W - mL - mR, ph = H - mT - mB;
  const dev = C.view === "dev", x0 = 0, x1 = 0.38;
  let y0 = 0, y1 = 0.38, yt;
  if (dev) {
    let lo = 0, hi = 0;
    LAYERS.forEach(r => { lo = Math.min(lo, r[C.kind] - r[3]); hi = Math.max(hi, r[C.kind] - r[3]); });
    const span = Math.max(hi - lo, 0.004);
    yt = niceTicks(lo - 0.08 * span, hi + 0.08 * span, 5);
    y0 = Math.min(lo - 0.08 * span, yt.ticks[0]); y1 = Math.max(hi + 0.08 * span, yt.ticks[yt.ticks.length - 1]);
  } else {
    yt = { ticks: [0, 0.1, 0.2, 0.3], step: 0.1 };
  }
  const X = v => mL + (v - x0) / (x1 - x0) * pw, Y = v => mT + (y1 - v) / (y1 - y0) * ph;
  host.textContent = "";
  const svg = S("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img",
    "aria-label": "Measured gap after rotation against the limit of Theorem 1 on 96 real Hessians" }, host);
  const dec = yt.step >= 0.1 ? 1 : yt.step >= 0.01 ? 2 : 3;
  yt.ticks.forEach(v => {
    S("line", { x1: mL, x2: mL + pw, y1: Y(v), y2: Y(v), class: "gl" }, svg);
    T(svg, mL - 7, Y(v) + 4, (v < 0 ? "−" : dev && v > 0 ? "+" : "") + Math.abs(v).toFixed(dec), "tk", "end");
  });
  [0, 0.1, 0.2, 0.3].forEach(v => T(svg, X(v), mT + ph + 16, v.toFixed(1), "tk", "middle"));
  S("line", { x1: mL, x2: mL + pw, y1: mT + ph, y2: mT + ph, class: "ax" }, svg);
  T(svg, mL + pw / 2, H - 6, "limit of Theorem 1 from the layer's spectrum (bit)", "al", "middle");
  const tl = T(svg, 0, 0, dev ? "measured gap − limit (bit)" : "measured gap at finite n (bit)", "al", "middle");
  tl.setAttribute("transform", "translate(11," + (mT + ph / 2) + ") rotate(-90)");
  if (dev) S("line", { x1: mL, x2: mL + pw, y1: Y(0), y2: Y(0), class: "ref" }, svg);
  else S("line", { x1: X(0), y1: Y(0), x2: X(0.38), y2: Y(0.38), class: "ref" }, svg);
  const pts = LAYERS.map(r => ({ r: r, x: X(r[3]), y: Y(dev ? r[C.kind] - r[3] : r[C.kind]) }));
  pts.filter(p => p.r[1] !== 2).forEach(p => S("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 4, class: "pt rot ringed" }, svg));
  pts.filter(p => p.r[1] === 2).forEach(p => S("circle", { cx: p.x.toFixed(1), cy: p.y.toFixed(1), r: 4, class: "pt o ringed" }, svg));
  const ring = S("circle", { r: 6.5, class: "ring", visibility: "hidden" }, svg);
  const hit = S("rect", { x: mL, y: mT, width: pw, height: ph, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", ev => {
    const bx = svg.getBoundingClientRect(), px = ev.clientX - bx.left, py = ev.clientY - bx.top;
    let best = null, bd = 18 * 18;
    pts.forEach(p => { const dd = (p.x - px) * (p.x - px) + (p.y - py) * (p.y - py); if (dd < bd) { bd = dd; best = p; } });
    if (!best) { ring.setAttribute("visibility", "hidden"); hideTip(); return; }
    const r = best.r;
    ring.setAttribute("cx", best.x); ring.setAttribute("cy", best.y); ring.setAttribute("visibility", "visible");
    showTip(ev, '<div class="tt">' + MODELS[r[0]] + " · " + TYPES[r[1]] + "</div>" + row("", "limit of Theorem 1", r[3].toFixed(4) + " bit") +
      row(r[1] === 2 ? "k-o" : "k-rot", KINDS[C.kind], r[C.kind].toFixed(4) + " bit") + row("", "measured − limit", bits(r[C.kind] - r[3], true) + " bit"));
  });
  hit.addEventListener("pointerleave", () => { ring.setAttribute("visibility", "hidden"); hideTip(); });
  /* readout and table */
  const all = LAYERS.map(r => r[C.kind] - r[3]), within = all.filter(v => Math.abs(v) <= 0.01).length;
  const mn = Math.min.apply(null, all), mx = Math.max.apply(null, all);
  $("gr-real-readout").innerHTML = "<strong>Figure 2 \u00b7 " + KINDS[C.kind] + ".</strong> Measured gap minus the limit lies between " + bits(mn, true) + " and " + bits(mx, true) +
    " bit over the 96 layers; " + within + " of 96 (" + Math.round(100 * within / 96) + "%) are within 0.01 bit.";
  const med = a => { const s = a.slice().sort((p, q) => p - q), h = s.length >> 1; return s.length % 2 ? s[h] : 0.5 * (s[h - 1] + s[h]); };
  let html = "<thead><tr><th>Input type</th><th>Layers</th><th>Median</th><th>Smallest</th><th>Largest</th></tr></thead><tbody>";
  [1, 2, 3, 4].forEach(t => {
    const v = LAYERS.filter(r => r[1] === t).map(r => r[C.kind] - r[3]);
    html += "<tr><td>" + TYPES[t] + "</td><td>" + v.length + "</td><td>" + bits(med(v), true) + "</td><td>" + bits(Math.min.apply(null, v), true) + "</td><td>" + bits(Math.max.apply(null, v), true) + "</td></tr>";
  });
  $("gr-real-table").innerHTML = html + "</tbody>";
  $("gr-real-table-cap").textContent = "Measured gap minus limit in bit, " + KINDS[C.kind] + ", damping 10⁻², mean over 3 seeds";
}
function initC() {
  if (!$("gr-real-chart")) return;
  seg($("gr-kind"), "data-kind", v => { C.kind = Number(v); drawC(); });
  seg($("gr-view"), "data-view", v => { C.view = v; drawC(); });
  drawC();
}

initA();
drawB();
initC();
let lastW = 0, pendR = 0;
window.addEventListener("resize", () => {
  clearTimeout(pendR);
  pendR = setTimeout(() => {
    if (window.innerWidth === lastW) return;
    lastW = window.innerWidth;
    drawA(); drawB(); drawC();
  }, 120);
});
root.querySelectorAll(".gr-copy").forEach(b => b.addEventListener("click", () => {
  const src = $(b.getAttribute("data-target"));
  if (!src || !navigator.clipboard) return;
  navigator.clipboard.writeText(src.textContent).then(() => { b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy"; }, 1500); }, () => {});
}));
})();
