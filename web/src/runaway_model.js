/* Thermal-runaway model, JavaScript port of ups_battery_prediction/thermal_runaway.py (same equations, same
   parameters, same integrator). scripts/verify_web_page.py runs this block in Node.js and compares it with the
   Python package. */
(function (root) {
  "use strict";
  var R = 8.314462618, SB = 5.670374419e-8;
  var SHARED = { A_sei: 1.667e15, E_sei: 1.3508e5, H_sei: 2.57e5, A_ne: 2.5e13, E_ne: 1.3508e5, H_ne: 1.714e6,
    A_e: 5.14e25, E_e: 2.74e5, H_e: 1.55e5, W_c: 610, W_e: 406, c_sei0: 0.15, c_neg0: 0.75, t_sei0: 0.033,
    t_ref: 0.033, alpha0: 0.04, c_e0: 1.0 };
  function chem(name, A_pe, E_pe, H_pe, W_p) {
    var c = { name: name, A_pe: A_pe, E_pe: E_pe, H_pe: H_pe, W_p: W_p };
    for (var k in SHARED) c[k] = SHARED[k];
    return c;
  }
  var CHEM = {
    LCO: chem("LCO", 6.667e13, 1.396e5, 3.14e5, 1300),
    NMC111: chem("NMC111", 2.25e14, 1.54e5, 7.9e5, 1293),
    LFP: chem("LFP", 2.0e8, 1.03e5, 1.947e5, 960)
  };
  var CELL = { radius: 0.009, height: 0.065, rho_cp: 2.789e6, h: 7.17, emissivity: 0.8 };
  CELL.volume = Math.PI * CELL.radius * CELL.radius * CELL.height;
  CELL.area = 2 * Math.PI * CELL.radius * CELL.height + 2 * Math.PI * CELL.radius * CELL.radius;
  CELL.face = CELL.height * 2 * CELL.radius;
  var NS = 6;

  function heatRate(c, T, cs, cn, ts, al, ce) {            // W/m3 and the four rates
    var r1 = c.A_sei * Math.exp(-c.E_sei / (R * T)) * cs;
    var r2 = c.A_ne * Math.exp(-ts / c.t_ref) * Math.exp(-c.E_ne / (R * T)) * cn;
    var r3 = c.A_pe * al * (1 - al) * Math.exp(-c.E_pe / (R * T));
    var r4 = c.A_e * Math.exp(-c.E_e / (R * T)) * ce;
    return [r1, r2, r3, r4, c.H_sei * c.W_c * r1 + c.H_ne * c.W_c * r2 + c.H_pe * c.W_p * r3 + c.H_e * c.W_e * r4];
  }

  /* o: {chem, n, Tamb, h, G, heater(t) -> W on cell 0 (optional)} */
  function makeRhs(o) {
    var c = o.chem, n = o.n || 1, h = o.h == null ? CELL.h : o.h, eps = CELL.emissivity, G = o.G || 0;
    var Ta4 = Math.pow(o.Tamb, 4), cap = CELL.rho_cp * CELL.volume;
    return function (t, y, dy) {
      var i, k;
      for (i = 0; i < n; i++) {
        k = i * NS;
        var T = Math.max(y[k], 1);
        var r = heatRate(c, T, y[k + 1], y[k + 2], y[k + 3], y[k + 4], y[k + 5]);
        dy[k + 1] = -r[0]; dy[k + 2] = -r[1]; dy[k + 3] = r[1]; dy[k + 4] = r[2]; dy[k + 5] = -r[3];
        var Tk = y[k];
        var p = r[4] * CELL.volume - h * CELL.area * (Tk - o.Tamb) - eps * SB * CELL.area * (Math.pow(Tk, 4) - Ta4);
        if (i === 0 && o.heater) p += o.heater(t);
        dy[k] = p;
      }
      for (i = 0; i < n - 1; i++) {
        var a = y[i * NS], b = y[(i + 1) * NS];
        var ex = G * (b - a) + eps * SB * CELL.face * (Math.pow(b, 4) - Math.pow(a, 4));
        dy[i * NS] += ex; dy[(i + 1) * NS] -= ex;
      }
      for (i = 0; i < n; i++) dy[i * NS] /= cap;
      return dy;
    };
  }

  function initial(c, T0, n) {
    var y = new Float64Array((n || 1) * NS);
    for (var i = 0; i < (n || 1); i++) y.set([T0, c.c_sei0, c.c_neg0, c.t_sei0, c.alpha0, c.c_e0], i * NS);
    return y;
  }

  function solveLinear(A, b, n) {                        // Gaussian elimination with partial pivoting, in place
    for (var k = 0; k < n; k++) {
      var p = k, m = Math.abs(A[k * n + k]);
      for (var i = k + 1; i < n; i++) if (Math.abs(A[i * n + k]) > m) { m = Math.abs(A[i * n + k]); p = i; }
      if (m === 0) return false;
      if (p !== k) {
        for (var j = 0; j < n; j++) { var t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; }
        var tb = b[k]; b[k] = b[p]; b[p] = tb;
      }
      for (i = k + 1; i < n; i++) {
        var f = A[i * n + k] / A[k * n + k];
        if (f !== 0) { for (j = k; j < n; j++) A[i * n + j] -= f * A[k * n + j]; b[i] -= f * b[k]; }
      }
    }
    for (k = n - 1; k >= 0; k--) {
      var s = b[k];
      for (j = k + 1; j < n; j++) s -= A[k * n + j] * b[j];
      b[k] = s / A[k * n + k];
    }
    return true;
  }

  /* Adaptive backward Euler with Newton iteration; same algorithm as the Python version. */
  function backwardEuler(f, y0, tEnd, opt) {
    opt = opt || {};
    var rtol = opt.rtol || 1e-4, atol = opt.atol || 1e-8, dtMax = opt.dtMax || 10, n = y0.length;
    var t = 0, y = Float64Array.from(y0), dt = 1e-3, steps = 0;
    var fy = new Float64Array(n), fz = new Float64Array(n), fk = new Float64Array(n);
    var z = new Float64Array(n), pred = new Float64Array(n), g = new Float64Array(n), J = new Float64Array(n * n);
    var ts = [0], ys = [Float64Array.from(y)];
    f(t, y, fy);
    while (t < tEnd && steps < (opt.maxSteps || 400000)) {
      dt = Math.min(dt, tEnd - t, dtMax);
      var t1 = t + dt, i, k, ok = false, err = 0;
      for (i = 0; i < n; i++) { pred[i] = y[i] + dt * fy[i]; z[i] = pred[i]; }
      for (var it = 0; it < 8; it++) {
        f(t1, z, fz);
        for (i = 0; i < n; i++) g[i] = -(z[i] - y[i] - dt * fz[i]);
        for (k = 0; k < n; k++) {
          var hk = 1e-7 * Math.max(Math.abs(z[k]), 1e-3), zk = z[k];
          z[k] = zk + hk; f(t1, z, fk); z[k] = zk;
          for (i = 0; i < n; i++) J[i * n + k] = (i === k ? 1 : 0) - dt * (fk[i] - fz[i]) / hk;
        }
        if (!solveLinear(J, g, n)) break;
        var conv = 0, bad = false;
        for (i = 0; i < n; i++) {
          z[i] += g[i];
          if (!isFinite(z[i])) bad = true;
          conv = Math.max(conv, Math.abs(g[i]) / (atol + rtol * Math.abs(z[i])));
        }
        if (bad) break;
        if (conv < 1e-3) { ok = true; break; }
      }
      if (ok) {
        for (i = 0; i < n; i++) err = Math.max(err, Math.abs(z[i] - pred[i]) / (atol + rtol * Math.max(Math.abs(z[i]), Math.abs(y[i]))));
        err *= 0.5;
      }
      if (!ok || err > 1) {
        dt *= ok ? Math.max(0.2, 0.9 / Math.sqrt(err)) : 0.3;
        if (dt < 1e-12) break;
        continue;
      }
      t = t1; y.set(z); f(t, y, fy);
      ts.push(t); ys.push(Float64Array.from(y)); steps++;
      dt *= Math.min(3, Math.max(0.3, 0.9 / Math.sqrt(Math.max(err, 1e-10))));
    }
    return { t: ts, y: ys, steps: steps };
  }

  function explicitEuler(f, y0, tEnd, dt) {
    var n = y0.length, y = Float64Array.from(y0), dy = new Float64Array(n), ts = [0], ys = [Float64Array.from(y)];
    var N = Math.round(tEnd / dt), blewAt = null;
    for (var s = 0; s < N; s++) {
      f(s * dt, y, dy);
      for (var i = 0; i < n; i++) y[i] += dt * dy[i];
      ts.push((s + 1) * dt); ys.push(Float64Array.from(y));
      if (!isFinite(y[0]) || Math.abs(y[0]) > 1e6) { blewAt = (s + 1) * dt; break; }
    }
    return { t: ts, y: ys, blewAt: blewAt };
  }

  function crossing(ts, ys, idx, level) {
    for (var i = 1; i < ts.length; i++) {
      if (ys[i][idx] > level) {
        var a = ys[i - 1][idx], b = ys[i][idx];
        return ts[i - 1] + (ts[i] - ts[i - 1]) * (level - a) / (b - a);
      }
    }
    return NaN;
  }

  function ovenTest(c, Toven, opt) {
    opt = opt || {};
    var f = makeRhs({ chem: c, n: 1, Tamb: Toven, h: opt.h });
    var y0 = initial(c, opt.T0 || 301.15, 1), tEnd = opt.tEnd || 7200;
    var sol = opt.explicit ? explicitEuler(f, y0, tEnd, opt.dt || 1) : backwardEuler(f, y0, tEnd, { rtol: opt.rtol || 1e-4 });
    var peak = -Infinity;
    for (var i = 0; i < sol.y.length; i++) if (sol.y[i][0] > peak) peak = sol.y[i][0];
    sol.peak = peak; sol.runaway = peak > Toven + 100; sol.tRunaway = sol.runaway ? crossing(sol.t, sol.y, 0, Toven + 100) : NaN;
    return sol;
  }

  function criticalOven(c, opt) {
    var lo = 380, hi = 520;
    while (hi - lo > 0.1) {
      var mid = 0.5 * (lo + hi);
      var o = { tEnd: 14400 }; for (var k in (opt || {})) o[k] = opt[k];
      if (ovenTest(c, mid, o).runaway) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
  }

  /* Row of n cells, cell 0 starts at Ttrigger. */
  function propagation(c, opt) {
    opt = opt || {};
    var n = opt.n || 6, Tamb = opt.Tamb || 298.15;
    var f = makeRhs({ chem: c, n: n, Tamb: Tamb, h: opt.h, G: opt.G == null ? 0.2 : opt.G });
    var y0 = initial(c, Tamb, n); y0[0] = opt.Ttrigger || 523.15;
    var sol = backwardEuler(f, y0, opt.tEnd || 2400, { rtol: opt.rtol || 1e-4, dtMax: 5 });
    sol.tRun = []; sol.nRun = 0;
    for (var i = 0; i < n; i++) {
      var tc = i === 0 && y0[0] > 573.15 ? 0 : crossing(sol.t, sol.y, i * NS, 573.15);
      sol.tRun.push(tc); if (isFinite(tc)) sol.nRun++;
    }
    return sol;
  }

  /* Heat generation and loss against cell temperature for the fresh cell (Semenov diagram). */
  function semenov(c, Tamb, h, Ts) {
    var gen = [], loss = [];
    for (var i = 0; i < Ts.length; i++) {
      var T = Ts[i];
      gen.push(heatRate(c, T, c.c_sei0, c.c_neg0, c.t_sei0, c.alpha0, c.c_e0)[4] * CELL.volume);
      loss.push((h == null ? CELL.h : h) * CELL.area * (T - Tamb) + CELL.emissivity * SB * CELL.area * (Math.pow(T, 4) - Math.pow(Tamb, 4)));
    }
    return { gen: gen, loss: loss };
  }

  /* Heat released by the reactions and heat passed to the surroundings (W) for one cell state. */
  function powers(c, y, Tamb, h) {
    var T = Math.max(y[0], 1);
    var gen = heatRate(c, T, y[1], y[2], y[3], y[4], y[5])[4] * CELL.volume;
    var loss = (h == null ? CELL.h : h) * CELL.area * (y[0] - Tamb) + CELL.emissivity * SB * CELL.area * (Math.pow(y[0], 4) - Math.pow(Tamb, 4));
    return [gen, loss];
  }

  var api = { powers: powers, CHEM: CHEM, CELL: CELL, NS: NS, makeRhs: makeRhs, initial: initial, backwardEuler: backwardEuler,
    explicitEuler: explicitEuler, ovenTest: ovenTest, criticalOven: criticalOven, propagation: propagation,
    semenov: semenov, crossing: crossing };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.TR = api;
})(typeof window !== "undefined" ? window : this);
