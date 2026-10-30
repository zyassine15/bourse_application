"use strict";

const SUGGESTIONS = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "MC.PA", "OR.PA", "AIR.PA", "SAP.DE"];
const COMP_LABELS = { trend: "Tendance", momentum: "Momentum", rsi: "RSI", macd: "MACD", risk: "Risque" };
const IND_LABELS = {
  ret_5: "Rendement 5 j", ret_21: "Rendement 1 mois", ret_63: "Rendement 3 mois", ret_126: "Rendement 6 mois",
  rsi_14: "RSI (14)", macd_hist: "Histogramme MACD", px_vs_sma20: "Prix vs MM20", px_vs_sma50: "Prix vs MM50",
  px_vs_sma200: "Prix vs MM200", sma50_vs_sma200: "MM50 vs MM200", vol_21: "Volatilité annualisée",
  drawdown_252: "Repli vs plus haut 1 an", bb_pctb: "Bandes de Bollinger (%B)", volume_ratio: "Volume 5 j / 60 j",
};
const PCT_KEYS = new Set(["ret_5", "ret_21", "ret_63", "ret_126", "px_vs_sma20", "px_vs_sma50", "px_vs_sma200", "sma50_vs_sma200", "vol_21", "drawdown_252", "macd_hist"]);
const SERIES_LABELS = { global: "Global", court: "Court", moyen: "Moyen", long: "Long" };

const $ = (id) => document.getElementById(id);
let current = null;
let currentSeries = "global";

const color = (score) => (score >= 65 ? "var(--good)" : score < 40 ? "var(--bad)" : "var(--mid)");
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmtPct = (v, digits = 1) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(digits)} %`;

function setStatus(msg, isError = false) {
  const s = $("status");
  s.textContent = msg;
  s.className = "status" + (isError ? " error" : "");
  s.hidden = !msg;
}

async function analyze(ticker) {
  setStatus(`Analyse de ${ticker}…`);
  $("result").hidden = true;
  $("search").querySelector("button").disabled = true;
  try {
    const res = await fetch(`/api/analysis/${encodeURIComponent(ticker)}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(typeof body.detail === "string" ? body.detail : `Erreur ${res.status}`);
    }
    current = await res.json();
    localStorage.setItem("lastTicker", current.ticker);
    render();
    setStatus("");
  } catch (e) {
    setStatus(e.message || "Erreur réseau", true);
  } finally {
    $("search").querySelector("button").disabled = false;
  }
}

function render() {
  const d = current;
  $("result").hidden = false;
  $("r-ticker").textContent = d.ticker;
  $("r-price").textContent = d.last_price.toFixed(2);
  const ch = $("r-change");
  ch.textContent = d.change_1d === null ? "" : fmtPct(d.change_1d, 2);
  ch.className = d.change_1d >= 0 ? "up" : "down";
  $("r-date").textContent = `au ${d.as_of}`;

  const g = $("gauge");
  g.style.setProperty("--pct", d.global.score);
  g.style.setProperty("--col", color(d.global.score));
  $("g-score").textContent = Math.round(d.global.score);
  const v = $("g-verdict");
  v.textContent = d.global.verdict;
  v.style.color = color(d.global.score);

  const val = $("validation");
  val.hidden = false;
  val.textContent = d.validation.message;
  val.className = "notice" + (d.validation.status === "validated" ? " ok" : "");

  const box = $("horizons");
  box.replaceChildren();
  for (const h of Object.values(d.horizons)) {
    const c = el("div", "card");
    c.append(el("h3", "", h.label));
    const s = el("div", "h-score", Math.round(h.score));
    s.style.color = color(h.score);
    c.append(s, el("div", "", h.verdict));
    const bar = el("div", "bar");
    const fill = el("i");
    fill.style.width = `${h.score}%`;
    fill.style.background = color(h.score);
    bar.append(fill);
    c.append(bar);
    for (const [k, val] of Object.entries(h.components)) {
      const r = el("div", "comp");
      const b = el("div", "bar");
      const f = el("i");
      f.style.width = `${val}%`;
      f.style.background = color(val);
      b.append(f);
      r.append(el("span", "", COMP_LABELS[k] || k), b, el("span", "", Math.round(val)));
      c.append(r);
    }
    box.append(c);
  }

  const tabs = $("tabs");
  tabs.replaceChildren();
  for (const key of Object.keys(d.history.series)) {
    const b = el("button", "tab", SERIES_LABELS[key] || key);
    b.type = "button";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(key === currentSeries));
    b.onclick = () => { currentSeries = key; render(); };
    tabs.append(b);
  }
  drawChart();

  const tbl = $("indicators");
  tbl.replaceChildren();
  for (const [k, val] of Object.entries(d.indicators)) {
    const tr = el("tr");
    const shown = PCT_KEYS.has(k) ? fmtPct(val, 2) : val.toFixed(2);
    tr.append(el("td", "", IND_LABELS[k] || k), el("td", "", shown));
    tbl.append(tr);
  }
  $("model").textContent = `Modèle : ${d.model_version}`;
  $("disclaimer").textContent = d.disclaimer;
}

function drawChart() {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = 220;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const pad = { l: 28, r: 8, t: 8, b: 20 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const y = (s) => pad.t + ih * (1 - s / 100);

  ctx.fillStyle = "rgba(34,197,94,.12)"; ctx.fillRect(pad.l, y(100), iw, y(65) - y(100));
  ctx.fillStyle = "rgba(239,68,68,.12)"; ctx.fillRect(pad.l, y(40), iw, y(0) - y(40));
  ctx.font = "11px sans-serif"; ctx.fillStyle = "#94a3b8"; ctx.textAlign = "right";
  for (const t of [0, 40, 65, 100]) ctx.fillText(String(t), pad.l - 4, y(t) + 4);

  const data = current.history.series[currentSeries];
  const dates = current.history.dates;
  const n = data.length;
  const x = (i) => pad.l + (iw * i) / Math.max(n - 1, 1);

  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.beginPath();
  let started = false;
  data.forEach((s, i) => {
    if (s === null) return;
    if (!started) { ctx.moveTo(x(i), y(s)); started = true; } else ctx.lineTo(x(i), y(s));
  });
  ctx.stroke();

  const last = data[n - 1];
  if (last !== null) {
    ctx.fillStyle = color(last); ctx.beginPath(); ctx.arc(x(n - 1), y(last), 4, 0, 7); ctx.fill();
  }
  ctx.fillStyle = "#94a3b8"; ctx.textAlign = "left"; ctx.fillText(dates[0], pad.l, h - 5);
  ctx.textAlign = "right"; ctx.fillText(dates[n - 1], w - pad.r, h - 5);
}

function init() {
  const chips = $("chips");
  for (const t of SUGGESTIONS) {
    const b = el("button", "chip", t);
    b.type = "button";
    b.onclick = () => { $("ticker").value = t; analyze(t); };
    chips.append(b);
  }
  $("search").addEventListener("submit", (e) => {
    e.preventDefault();
    const t = $("ticker").value.trim().toUpperCase();
    if (t) analyze(t);
    $("ticker").blur();
  });
  window.addEventListener("resize", () => current && drawChart());
  const last = localStorage.getItem("lastTicker");
  if (last) { $("ticker").value = last; analyze(last); }
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
}

init();
