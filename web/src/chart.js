/* Minimal SVG line chart with crosshair tooltip. Colors come from CSS variables of the page. */
(function (root) {
  "use strict";
  var NSV = "http://www.w3.org/2000/svg";
  function el(tag, attrs, parent) {
    var e = document.createElementNS(NSV, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, step = Math.pow(10, Math.floor(Math.log10(span / n))), err = span / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var t = [], v = Math.ceil(lo / step - 1e-9) * step;
    for (; v <= hi + 1e-9 * span; v += step) t.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return t;
  }
  function fmt(v) {
    var a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1).replace(/\.0$/, "");
    return String(+v.toPrecision(2));
  }
  /* opt: {series:[{name,color,x,y,dash,width}], xLabel,yLabel,xDomain,yDomain,yLog,hLines:[{y,label}],
           vLines:[{x,label}], fmtX, fmtY, height} */
  function lineChart(host, opt) {
    host.innerHTML = "";
    host.style.position = "relative";
    var W = Math.max(280, host.clientWidth || 600), H = opt.height || 320;
    var m = { l: 56, r: 16, t: 12, b: 42 };
    host.classList.add("plotbox");
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: "100%", height: H, role: "img", "aria-label": opt.title || "chart" });
    host.appendChild(svg);
    var xd = opt.xDomain, yd = opt.yDomain, pw = W - m.l - m.r, ph = H - m.t - m.b;
    var ly = opt.yLog ? function (v) { return Math.log10(Math.max(v, yd[0])); } : function (v) { return v; };
    var y0 = ly(yd[0]), y1 = ly(yd[1]);
    function X(v) { return m.l + (v - xd[0]) / (xd[1] - xd[0]) * pw; }
    function Y(v) { return m.t + (1 - (ly(v) - y0) / (y1 - y0)) * ph; }
    var g = el("g", {}, svg);
    var xt = niceTicks(xd[0], xd[1], Math.max(3, Math.floor(pw / 90))), yt;
    if (opt.yLog) { yt = []; for (var e = Math.ceil(y0 - 1e-9); e <= y1 + 1e-9; e++) yt.push(Math.pow(10, e)); }
    else yt = niceTicks(yd[0], yd[1], Math.max(3, Math.floor(ph / 55)));
    yt.forEach(function (v) {
      el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: "var(--grid)", "stroke-width": 1 }, g);
      var tx = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end", class: "tk" }, g);
      tx.textContent = opt.fmtY ? opt.fmtY(v) : fmt(v);
    });
    xt.forEach(function (v) {
      var tx = el("text", { x: X(v), y: H - m.b + 20, "text-anchor": "middle", class: "tk" }, g);
      tx.textContent = opt.fmtX ? opt.fmtX(v) : fmt(v);
    });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1 }, g);
    var xl = el("text", { x: m.l + pw / 2, y: H - 8, "text-anchor": "middle", class: "lab" }, g); xl.textContent = opt.xLabel || "";
    var yl = el("text", { x: 14, y: m.t + ph / 2, "text-anchor": "middle", class: "lab", transform: "rotate(-90 14 " + (m.t + ph / 2) + ")" }, g);
    yl.textContent = opt.yLabel || "";
    (opt.hLines || []).forEach(function (hl) {
      if (hl.y < yd[0] || hl.y > yd[1]) return;
      el("line", { x1: m.l, x2: W - m.r, y1: Y(hl.y), y2: Y(hl.y), stroke: "var(--muted)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, g);
      var t = el("text", { x: W - m.r - 4, y: Y(hl.y) - 5, "text-anchor": "end", class: "ann" }, g); t.textContent = hl.label;
    });
    (opt.vLines || []).forEach(function (vl) {
      if (!(vl.x >= xd[0] && vl.x <= xd[1])) return;
      el("line", { x1: X(vl.x), x2: X(vl.x), y1: m.t, y2: H - m.b, stroke: "var(--muted)", "stroke-width": 1.2, "stroke-dasharray": "5 4" }, g);
      var t = el("text", { x: X(vl.x) + 5, y: m.t + 12, class: "ann" }, g); t.textContent = vl.label;
    });
    var clipId = "c" + Math.random().toString(36).slice(2);
    var cp = el("clipPath", { id: clipId }, svg); el("rect", { x: m.l, y: m.t - 2, width: pw, height: ph + 4 }, cp);
    opt.series.forEach(function (s) {
      var d = "", n = s.x.length, stride = Math.max(1, Math.floor(n / 4000)), pen = false, idx = [];
      for (var i = 0; i < n; i += stride) idx.push(i);
      if (idx[idx.length - 1] !== n - 1) idx.push(n - 1);
      idx.forEach(function (i) {
        var yv = s.y[i];
        if (!isFinite(yv) || (opt.yLog && yv <= 0)) { pen = false; return; }
        var yy = Math.max(-1e4, Math.min(1e4, Y(yv)));
        d += (pen ? "L" : "M") + X(s.x[i]).toFixed(1) + " " + yy.toFixed(1); pen = true;
      });
      el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.4, "stroke-linejoin": "round",
        "stroke-dasharray": s.dash || "", "clip-path": "url(#" + clipId + ")" }, svg);
    });
    // hover layer
    var cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
    var tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; host.appendChild(tip);
    var hit = el("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" }, svg);
    function move(ev) {
      var r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
      var xv = xd[0] + (px - m.l) / pw * (xd[1] - xd[0]);
      if (xv < xd[0] || xv > xd[1]) return leave();
      var rows = "";
      opt.series.forEach(function (s) {
        if (s.noTip) return;
        var lo = 0, hi = s.x.length - 1;
        if (xv < s.x[0] || xv > s.x[hi]) return;
        while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (s.x[mid] <= xv) lo = mid; else hi = mid; }
        var f = s.x[hi] > s.x[lo] ? (xv - s.x[lo]) / (s.x[hi] - s.x[lo]) : 0, v = s.y[lo] + f * (s.y[hi] - s.y[lo]);
        if (!isFinite(v)) return;
        rows += '<div class="r"><svg width="16" height="8"><line x1="0" y1="4" x2="16" y2="4" stroke="' + s.color +
          '" stroke-width="2.4"' + (s.dash ? ' stroke-dasharray="4 3"' : "") + '/></svg><i>' + s.name + "</i><b>" +
          (opt.fmtTip ? opt.fmtTip(v) : fmt(v)) + "</b></div>";
      });
      if (!rows) return leave();
      cross.setAttribute("x1", X(xv)); cross.setAttribute("x2", X(xv)); cross.setAttribute("visibility", "visible");
      tip.innerHTML = '<div class="t">' + (opt.fmtTipX ? opt.fmtTipX(xv) : fmt(xv)) + "</div>" + rows;
      tip.hidden = false;
      var left = (X(xv) / W) * r.width + 12;
      if (left + tip.offsetWidth > r.width) left = (X(xv) / W) * r.width - tip.offsetWidth - 12;
      tip.style.left = Math.max(0, left) + "px"; tip.style.top = "8px";
    }
    function leave() { cross.setAttribute("visibility", "hidden"); tip.hidden = true; }
    hit.addEventListener("pointermove", move); hit.addEventListener("pointerleave", leave);
    return { X: X, Y: Y };
  }
  root.Chart = { lineChart: lineChart, fmt: fmt };
})(window);
