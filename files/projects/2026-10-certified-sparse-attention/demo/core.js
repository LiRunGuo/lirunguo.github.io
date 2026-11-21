/* Certified sparse attention: the value-scale certificate of the paper (Section 4, Algorithm 1) with
   exact scores, for one query head. Ported from the paper's NumPy reference implementation (value scale):
   pool = complete 128-token blocks ending before the last 64 tokens, the rest is the exact window;
   E_i ~ Exp(1); levels tau_j = 2^(-j/r); Chernoff/KL count interval; centred Pinelis bound with a
   Bernstein upper bound on G_A(c); Bernstein lower bound on Q_A for M_lo; delta_j = delta/((J+1)*4).
   The 4-bit candidate scan is omitted: it returns exactly the same sample (Lemma 2). */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.CSA = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const LN2 = Math.LN2;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // uniform in (0, 1] with 53 random bits, as in the kernel
  function uniform53(r) {
    const hi = Math.floor(r() * 2097152), lo = Math.floor(r() * 4294967296);
    return (hi * 4294967296 + lo + 1) / 9007199254740992;
  }
  function gaussian(r) {
    const u = uniform53(r), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* Shared value directions Z (n x d) with log-normal norms, so a few values are much larger than
     the typical one, as on real KV caches. */
  function makeValues(n, d, seed) {
    const r = mulberry32(seed), Z = new Float64Array(n * d), zz = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const sc = Math.exp(0.35 * gaussian(r));
      let q = 0;
      for (let t = 0; t < d; t++) { const x = gaussian(r) * sc; Z[i * d + t] = x; q += x * x; }
      zz[i] = q;
    }
    return { n, d, Z, zz };
  }

  /* Scores of one synthetic head: a Gaussian bulk (spread sigma), an attention sink at position 0,
     a few high-scoring "needle" keys and a recency bump on the last 512 tokens. */
  function makeScores(n, p, seed, G) {
    const r = mulberry32(seed), s = new Float64Array(n);
    if (G) {                       // shared standard normals, cyclically shifted per head (faster)
      const off = Math.floor(r() * n);
      for (let i = 0; i < n; i++) s[i] = p.sigma * G[(i + off) % n];
    } else {
      for (let i = 0; i < n; i++) s[i] = p.sigma * gaussian(r);
    }
    s[0] = p.sink;
    for (let k = 0; k < p.needles; k++) {
      const i = 1 + Math.floor(r() * (n - 600));
      s[i] = p.needleLo + (p.needleHi - p.needleLo) * r();
    }
    const bump = p.recency || 0;
    for (let i = n - 512; i < n; i++) s[i] += bump * (i - (n - 512)) / 512;
    return s;
  }

  function makeMu(d, norm, seed) {
    const r = mulberry32(seed), mu = new Float64Array(d);
    let q = 0;
    for (let t = 0; t < d; t++) { mu[t] = gaussian(r); q += mu[t] * mu[t]; }
    const f = norm / Math.sqrt(q);
    for (let t = 0; t < d; t++) mu[t] *= f;
    return mu;
  }

  /* Everything about a call that does not depend on the perturbations. v_i = Z_i + mu. */
  function prepare(s, vals, mu, opt) {
    opt = opt || {};
    const n = vals.n, d = vals.d, Z = vals.Z, block = opt.block || 128, window = opt.window == null ? 64 : opt.window;
    const npb = Math.min(Math.floor(n / block), Math.max(0, Math.floor((n - window) / block)));
    const npool = npb * block;
    let c = -Infinity;
    for (let i = 0; i < n; i++) if (s[i] > c) c = s[i];
    const w = new Float64Array(n), vn2 = new Float64Array(n);
    let mu2 = 0;
    for (let t = 0; t < d; t++) mu2 += mu[t] * mu[t];
    let D = 0, Q = 0, DR = 0, QR = 0, vmax2 = 0;
    const N = new Float64Array(d), NR = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      const wi = Math.exp(s[i] - c);
      w[i] = wi;
      let zm = 0;
      const o = i * d;
      for (let t = 0; t < d; t++) zm += Z[o + t] * mu[t];
      const q = vals.zz[i] + 2 * zm + mu2;
      vn2[i] = q;
      D += wi; Q += wi * q;
      for (let t = 0; t < d; t++) N[t] += wi * (Z[o + t] + mu[t]);
      if (i < npool) { if (q > vmax2) vmax2 = q; }
      else {
        DR += wi; QR += wi * q;
        for (let t = 0; t < d; t++) NR[t] += wi * (Z[o + t] + mu[t]);
      }
    }
    const out = new Float64Array(d);
    let on2 = 0;
    for (let t = 0; t < d; t++) { out[t] = N[t] / D; on2 += out[t] * out[t]; }
    // entropy-based effective number of keys, for display (ln a_i = s_i - c - ln D)
    let H = 0;
    const lnD = Math.log(D);
    for (let i = 0; i < n; i++) H -= (w[i] / D) * (s[i] - c - lnD);
    return { n, d, Z, mu, s, c, w, vn2, npool, nwin: n - npool, D, Q, N, o: out, M: Math.sqrt(Q / D),
             onorm: Math.sqrt(on2), DR, QR, NR, Vmax: Math.sqrt(vmax2), effKeys: Math.exp(H) };
  }

  function perturb(n, seed) {
    const r = mulberry32(seed), E = new Float64Array(n);
    for (let i = 0; i < n; i++) E[i] = -Math.log(uniform53(r));
    return E;
  }

  function levelOf(x) { return x <= 0 ? 0 : Math.ceil(x - 1e-12); }

  // Chernoff / KL interval {mu : k ln(k/mu) + mu - k <= L}, as in pps_ref.kl_ci
  function klCI(k, L) {
    const f = (m) => (k > 0 ? k * Math.log(k / m) : 0) + m - k;
    let lo = Math.max(k, 1e-12), hi = k + 2 * L + Math.sqrt(2 * k * L) + 1;
    for (let it = 0; it < 80; it++) { const mid = (lo + hi) / 2; if (f(mid) > L) hi = mid; else lo = mid; }
    const muHi = hi;
    if (k <= 0) return [0, muHi];
    lo = 1e-300; hi = k;
    for (let it = 0; it < 200; it++) { const mid = (lo + hi) / 2; if (f(mid) > L) lo = mid; else hi = mid; }
    return [lo, muHi];
  }
  function bernsteinUCB(est, b, L) {
    const a = Math.sqrt(2 * b * L);
    const x = (a + Math.sqrt(a * a + 4 * (Math.max(est, 0) + b * L / 3))) / 2;
    return x * x;
  }

  /* One call. opt: eps, delta, r, J, trace (record every level), certAfterStop (keep computing the
     bound after the stop level, for display), keysCap (stop tracing once this many keys are read).
     Returns the stop level (-1: fall back to exact attention), keys read and error at the stop, and
     per-level arrays when trace is set. Errors are ||o^ - o|| / M. */
  function run(h, E, opt) {
    const eps = opt.eps, delta = opt.delta == null ? 0.05 : opt.delta, r = opt.r || 4, J = opt.J == null ? 160 : opt.J;
    const d = h.d, Z = h.Z, mu = h.mu, s = h.s, c = h.c, w = h.w, vn2 = h.vn2, npool = h.npool;
    const je = new Int32Array(npool), jb = new Int32Array(npool);
    const cntE = new Int32Array(J + 2), cntB = new Int32Array(J + 2);
    for (let i = 0; i < npool; i++) {
      const a = levelOf(-r * (s[i] - c + E[i]) / LN2), b = levelOf(-r * (s[i] - c) / LN2);
      je[i] = a; jb[i] = b;
      if (a <= J) cntE[a + 1]++;
      if (b <= J) cntB[b + 1]++;
    }
    for (let j = 0; j <= J; j++) { cntE[j + 1] += cntE[j]; cntB[j + 1] += cntB[j]; }
    const ordE = new Int32Array(cntE[J + 1]), ordB = new Int32Array(cntB[J + 1]);
    const posE = cntE.slice(), posB = cntB.slice();
    for (let i = 0; i < npool; i++) {
      if (je[i] <= J) ordE[posE[je[i]]++] = i;
      if (jb[i] <= J) ordB[posB[jb[i]]++] = i;
    }
    const dj = delta / (J + 1) / 4;
    const LD = Math.log(2 / dj), LG = Math.log(1 / dj), LN = Math.log(2 / dj), LQ = Math.log(1 / dj);
    const NB = new Float64Array(d), sv = new Float64Array(d), oh = new Float64Array(d), cc = new Float64Array(d);
    let DB = 0, QB = 0, k = 0, sv2 = 0, nIncl = 0, nB = 0;
    const Vmax = h.Vmax, M = h.M, keysCap = opt.keysCap || Infinity;
    const tr = opt.trace ? { keys: [], err: [], beta: [], tgt: [], k: [], nB: [], ok: [] } : null;
    const errAt = opt.errAt ? [] : null, keysAt = opt.errAt ? [] : null;
    let stop = -1, stopKeys = h.n, stopErr = 0, stopOh = null, stopBeta = NaN, stopTgt = NaN;
    for (let j = 0; j <= J; j++) {
      const tau = Math.pow(2, -j / r);
      for (let p = cntE[j]; p < cntE[j + 1]; p++) {
        const i = ordE[p], o = i * d;
        nIncl++;
        if (jb[i] <= j) {
          const wi = w[i];
          DB += wi; QB += wi * vn2[i]; nB++;
          for (let t = 0; t < d; t++) NB[t] += wi * (Z[o + t] + mu[t]);
        } else {
          k++; sv2 += vn2[i];
          for (let t = 0; t < d; t++) sv[t] += Z[o + t] + mu[t];
        }
      }
      for (let p = cntB[j]; p < cntB[j + 1]; p++) {
        const i = ordB[p];
        if (je[i] < j) {       // sampled earlier as a light key; it is heavy from this level on
          const o = i * d, wi = w[i];
          k--; sv2 -= vn2[i];
          DB += wi; QB += wi * vn2[i]; nB++;
          for (let t = 0; t < d; t++) { const v = Z[o + t] + mu[t]; sv[t] -= v; NB[t] += wi * v; }
        }
      }
      const Dh = h.DR + DB + tau * k;
      let e2 = 0;
      for (let t = 0; t < d; t++) { oh[t] = (h.NR[t] + NB[t] + tau * sv[t]) / Dh; const x = oh[t] - h.o[t]; e2 += x * x; }
      const err = Math.sqrt(e2) / M, keys = h.nwin + nIncl;
      let beta = Infinity, tgt = NaN, ok = false;
      if (stop < 0 || opt.certAfterStop) {
        const ci = klCI(k, LD);
        const dD = tau * Math.max(k - ci[0], ci[1] - k);
        if (Dh - dD > 0) {
          const DRB = h.DR + DB;
          let cn2 = 0, csv = 0, ohc2 = 0;
          for (let t = 0; t < d; t++) {
            cc[t] = (h.NR[t] + NB[t]) / DRB; cn2 += cc[t] * cc[t]; csv += cc[t] * sv[t];
            const x = oh[t] - cc[t]; ohc2 += x * x;
          }
          const GA = tau * (sv2 - 2 * csv + k * cn2);
          const Rc = Vmax + Math.sqrt(cn2);
          const GAhi = bernsteinUCB(GA, tau * Rc * Rc, LG);
          const dZ = Math.sqrt(2 * tau * GAhi * LN) + 2 * tau * Rc * LN / 3;
          beta = (dZ + Math.sqrt(ohc2) * dD) / (Dh - dD);
          const bq = tau * Vmax * Vmax, aq = Math.sqrt(2 * bq * LQ), qa = tau * sv2 - bq * LQ / 3;
          const xa = qa > 0 ? (Math.sqrt(aq * aq + 4 * qa) - aq) / 2 : 0;
          const Mlo = Math.sqrt((h.QR + QB + xa * xa) / (Dh + dD));
          tgt = eps * Mlo;
          ok = beta <= tgt;
        }
      }
      if (tr) {
        tr.keys.push(keys); tr.err.push(err); tr.beta.push(beta / M); tr.tgt.push(tgt / M);
        tr.k.push(k); tr.nB.push(nB); tr.ok.push(ok);
      }
      if (errAt) { errAt.push(err); keysAt.push(keys); }
      if (ok && stop < 0) {
        stop = j; stopKeys = keys; stopErr = err; stopOh = Float64Array.from(oh); stopBeta = beta / M; stopTgt = tgt / M;
      }
      // after the stop: keep going only to trace levels for display, or to record the error of
      // every fixed level (population view) until keysCap keys are read
      const traceEnd = opt.traceAfter == null ? J : stop + opt.traceAfter;
      const wantMore = (tr && j < traceEnd && nIncl < npool) || (errAt && keys < keysCap);
      if (stop >= 0 && !wantMore) break;
    }
    return { stop, keys: stopKeys, err: stopErr, oh: stopOh, beta: stopBeta, tgt: stopTgt, trace: tr, errAt, keysAt,
             je: tr ? je : null };
  }

  // k-th largest element (1-based) by quickselect; reorders a
  function kthLargest(a, k) {
    let lo = 0, hi = a.length - 1;
    const target = a.length - k;
    while (lo < hi) {
      const pivot = a[(lo + hi) >> 1];
      let i = lo, j = hi;
      while (i <= j) {
        while (a[i] < pivot) i++;
        while (a[j] > pivot) j--;
        if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
      }
      if (target <= j) hi = j;
      else if (target >= i) lo = i;
      else break;
    }
    return a[target];
  }

  /* Oracle top-k: the window plus the kPool highest-scoring pool keys, renormalized (the rest of the
     mass is dropped). Returns ||o_topk - o|| / M. */
  function topk(h, kPool) {
    const d = h.d, Z = h.Z, mu = h.mu, s = h.s, w = h.w, npool = h.npool;
    if (kPool >= npool) return 0;
    let thr = Infinity;
    if (kPool > 0) thr = kthLargest(Float64Array.from(s.subarray(0, npool)), kPool);
    let Ds = h.DR;
    const Ns = Float64Array.from(h.NR);
    for (let i = 0; i < npool; i++) {
      if (s[i] >= thr) {
        const wi = w[i], o = i * d;
        Ds += wi;
        for (let t = 0; t < d; t++) Ns[t] += wi * (Z[o + t] + mu[t]);
      }
    }
    let e2 = 0;
    for (let t = 0; t < d; t++) { const x = Ns[t] / Ds - h.o[t]; e2 += x * x; }
    return Math.sqrt(e2) / h.M;
  }

  return { mulberry32, uniform53, gaussian, makeValues, makeScores, makeMu, prepare, perturb, run, topk, kthLargest, klCI, levelOf };
});
