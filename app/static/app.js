"use strict";

const COMP_LABELS = { trend: "Tendance", momentum: "Momentum", rsi: "RSI", macd: "MACD", risk: "Risque" };
const SERIES_LABELS = { global: "Global", court: "Court", moyen: "Moyen", long: "Long" };

const $ = (id) => document.getElementById(id);
let current = null;
let currentSeries = "global";
let instruments = [];
let markets = [];
let popular = {};
let market = localStorage.getItem("market") || "MA"; // "MA", "INTL" ou "ALL"
const MARKET_ALL = "ALL";
const inMarket = (item) => market === MARKET_ALL || item.market === market;
const marketLabel = (id) => (id === MARKET_ALL ? "Tous les marchés" : (markets.find((m) => m.id === id) || {}).label || id);
let priceRange = 252; // séances affichées (0 = tout l'historique)
let hoverIndex = null;
const RANGES = [["3M", 63], ["6M", 126], ["1A", 252], ["Tout", 0]];

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

let requestSeq = 0;

async function fetchJson(url) {
  const res = await fetch(url);
  const body = await res.json().catch(() => ({}));
  return { res, body };
}

async function analyze(ticker) {
  const seq = ++requestSeq; // seule la dernière demande a le droit d'afficher son résultat
  setStatus(`Analyse de ${ticker}…`);
  $("result").hidden = true;
  $("search").querySelector("button").disabled = true;
  const path = encodeURIComponent(ticker);
  try {
    let { res, body } = await fetchJson(`/api/analysis/${path}`);
    if (res.status === 422 && instruments.some((i) => i.symbol === ticker)) {
      // Historique trop court pour un score (ex. introduction en bourse récente) : on affiche au moins le cours.
      const prices = await fetchJson(`/api/prices/${path}`);
      if (prices.res.ok) {
        if (seq !== requestSeq) return;
        current = { ...prices.body, limited: true };
        localStorage.setItem("lastTicker", current.ticker);
        render();
        setStatus("");
        return;
      }
    }
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Erreur ${res.status}`);
    if (seq !== requestSeq) return;
    current = body;
    localStorage.setItem("lastTicker", current.ticker);
    render();
    setStatus("");
  } catch (e) {
    if (seq === requestSeq) setStatus(e.message || "Erreur réseau", true);
  } finally {
    if (seq === requestSeq) $("search").querySelector("button").disabled = false;
  }
}

function badgesFor(item) {
  const out = [];
  if (item.kind === "ETF") out.push(el("span", "badge etf", "ETF"));
  if (item.leverage > 1) out.push(el("span", "badge lev", `×${item.leverage}`));
  return out;
}

const money = (v, currency) => `${fmtPrice(v)} ${currency}`;

function render() {
  const d = current;
  $("result").hidden = false;
  $("result").classList.toggle("limited", Boolean(d.limited));
  $("r-ticker").textContent = d.ticker;
  $("r-name").textContent = d.name;
  $("r-badges").replaceChildren(...badgesFor(d));
  $("r-price").textContent = money(d.last_price, d.currency);
  const ch = $("r-change");
  ch.textContent = d.change_1d === null ? "" : fmtPct(d.change_1d, 2);
  ch.className = d.change_1d >= 0 ? "up" : "down";
  $("r-date").textContent = `au ${d.as_of}`;

  const note = $("instrument-notice");
  const noteText = [d.reason, d.notice].filter(Boolean).join(" ");
  note.hidden = !noteText;
  note.textContent = noteText;
  note.className = "notice" + (d.limited ? " info" : " bad");

  if (d.limited) {
    // Pas de score : seule la carte « Cours » est affichée.
    hoverIndex = null;
    renderRangeTabs();
    drawPriceChart();
    $("disclaimer").textContent = "Cours journaliers à titre informatif uniquement.";
    loadIndicators(d.ticker);
    return;
  }

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
  val.className = "notice score-only" + (d.validation.status === "validated" ? " ok" : d.validation.status === "inverted" ? " bad" : "");

  const liq = $("liquidity");
  liq.hidden = !d.liquidity.warning;
  liq.textContent = d.liquidity.warning || "";

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
  hoverIndex = null;
  renderRangeTabs();
  drawPriceChart();

  $("model").textContent = `Modèle : ${d.model_version}`;
  $("disclaimer").textContent = d.disclaimer;
  loadIndicators(d.ticker);
}

/* ---------- Indicateurs détaillés ---------- */
let indicatorsSeq = 0;

const compact = (v) => {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toLocaleString("fr-FR", { maximumFractionDigits: 2 })} Md`;
  if (a >= 1e6) return `${(v / 1e6).toLocaleString("fr-FR", { maximumFractionDigits: 1 })} M`;
  return v.toLocaleString("fr-FR", { maximumFractionDigits: 0 });
};

function formatIndicator(v) {
  if (v.status !== "ok") return { text: { not_meaningful: "N/S", not_applicable: "S/O" }[v.status] || "N/D", na: true };
  const n = (x, d = 1) => x.toLocaleString("fr-FR", { minimumFractionDigits: d, maximumFractionDigits: d });
  switch (v.unit) {
    case "ratio": return { text: `${n(v.value, Math.abs(v.value) < 10 ? 2 : 1)}×` };
    case "pct": return { text: `${n(v.value * 100)} %` };
    case "money": return { text: `${compact(v.value)} ${v.currency || ""}`.trim() };
    case "price": return { text: `${fmtPrice(v.value)} ${v.currency || ""}`.trim() };
    case "points": return { text: n(v.value) };
    default: return { text: compact(v.value) };
  }
}

function renderIndicators(data) {
  const box = $("indicators");
  box.replaceChildren();
  for (const group of data.groups) {
    box.append(el("h3", "ind-group", group.label));
    if (group.note) box.append(el("p", "ind-note", group.note));
    for (const it of group.items) {
      const wrap = el("div", "ind");
      wrap.append(el("h4", "", it.label));
      if (it.help) wrap.append(el("p", "ind-help", it.help));
      for (const v of it.values) {
        const row = el("div", "ind-row");
        const f = formatIndicator(v);
        row.append(el("span", "ind-label", v.label), el("span", "ind-value" + (f.na ? " na" : ""), f.text));
        wrap.append(row);
        const sub = [v.period, v.note].filter(Boolean).join(" · ");
        if (sub) wrap.append(el("p", "ind-sub", sub));
      }
      if (it.reading) wrap.append(el("p", "ind-reading", it.reading));
      box.append(wrap);
    }
  }
}

async function loadIndicators(ticker) {
  const seq = ++indicatorsSeq; // ignore les réponses d'une valeur précédente
  const box = $("indicators");
  box.replaceChildren(el("p", "ind-note", "Chargement des indicateurs…"));
  try {
    const { res, body } = await fetchJson(`/api/indicators/${encodeURIComponent(ticker)}`);
    if (seq !== indicatorsSeq) return;
    if (!res.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Erreur ${res.status}`);
    renderIndicators(body);
  } catch (e) {
    if (seq === indicatorsSeq) box.replaceChildren(el("p", "ind-note", "Indicateurs indisponibles pour le moment."));
  }
}

function renderRangeTabs() {
  const tabs = $("range-tabs");
  tabs.replaceChildren();
  for (const [label, n] of RANGES) {
    const b = el("button", "tab", label);
    b.type = "button";
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(n === priceRange));
    b.onclick = () => { priceRange = n; hoverIndex = null; renderRangeTabs(); drawPriceChart(); };
    tabs.append(b);
  }
}

const fmtPrice = (v) => v.toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function drawPriceChart() {
  const p = current.prices;
  const total = p.close.length;
  const start = priceRange ? Math.max(total - priceRange, 0) : 0;
  const dates = p.dates.slice(start), close = p.close.slice(start);
  const sma50 = p.sma50.slice(start), sma200 = p.sma200.slice(start);
  const n = close.length;

  const perf = close[n - 1] / close[0] - 1;
  const perfEl = $("p-perf");
  perfEl.textContent = n > 1 ? fmtPct(perf) : "";
  perfEl.className = "perf " + (perf >= 0 ? "up" : "down");
  const showReadout = (i) => {
    const parts = [dates[i], `${fmtPrice(close[i])} ${current.currency}`];
    if (sma50[i] !== null) parts.push(`MM50 ${fmtPrice(sma50[i])}`);
    if (sma200[i] !== null) parts.push(`MM200 ${fmtPrice(sma200[i])}`);
    $("p-readout").textContent = parts.join(" · ");
  };
  showReadout(hoverIndex === null ? n - 1 : hoverIndex);

  const canvas = $("price-chart");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = 240;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);

  const pad = { l: 8, r: 52, t: 8, b: 20 };
  const iw = w - pad.l - pad.r, ih = h - pad.t - pad.b;
  const visible = [...close, ...sma50, ...sma200].filter((v) => v !== null);
  let lo = Math.min(...close), hi = Math.max(...close);
  for (const v of visible) { if (v < lo * 0.8) continue; lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const margin = (hi - lo) * 0.06 || 1;
  lo -= margin; hi += margin;
  const x = (i) => pad.l + (iw * i) / Math.max(n - 1, 1);
  const y = (v) => pad.t + ih * (1 - (v - lo) / (hi - lo));

  ctx.font = "11px sans-serif"; ctx.textAlign = "left"; ctx.fillStyle = "#94a3b8";
  ctx.strokeStyle = "rgba(148,163,184,.18)"; ctx.lineWidth = 1;
  for (let k = 0; k <= 3; k++) {
    const v = lo + ((hi - lo) * k) / 3;
    ctx.beginPath(); ctx.moveTo(pad.l, y(v)); ctx.lineTo(w - pad.r, y(v)); ctx.stroke();
    ctx.fillText(v.toFixed(v >= 100 ? 0 : 1), w - pad.r + 4, y(v) + 4);
  }

  const line = (series, col, width, dash) => {
    ctx.strokeStyle = col; ctx.lineWidth = width; ctx.setLineDash(dash); ctx.lineJoin = "round";
    ctx.beginPath();
    let started = false;
    series.forEach((v, i) => {
      if (v === null) return;
      if (!started) { ctx.moveTo(x(i), y(v)); started = true; } else ctx.lineTo(x(i), y(v));
    });
    ctx.stroke(); ctx.setLineDash([]);
  };

  const grad = ctx.createLinearGradient(0, pad.t, 0, pad.t + ih);
  grad.addColorStop(0, "rgba(56,189,248,.25)"); grad.addColorStop(1, "rgba(56,189,248,0)");
  ctx.beginPath(); ctx.moveTo(x(0), y(close[0]));
  close.forEach((v, i) => ctx.lineTo(x(i), y(v)));
  ctx.lineTo(x(n - 1), pad.t + ih); ctx.lineTo(x(0), pad.t + ih); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  line(sma200, "#a78bfa", 1.5, [4, 3]);
  line(sma50, "#f59e0b", 1.5, [4, 3]);
  line(close, "#38bdf8", 2, []);

  ctx.fillStyle = "#94a3b8"; ctx.textAlign = "left"; ctx.fillText(dates[0], pad.l, h - 5);
  ctx.textAlign = "right"; ctx.fillText(dates[n - 1], w - pad.r, h - 5);

  const i = hoverIndex === null ? n - 1 : hoverIndex;
  if (hoverIndex !== null) {
    ctx.strokeStyle = "rgba(241,245,249,.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x(i), pad.t); ctx.lineTo(x(i), pad.t + ih); ctx.stroke();
  }
  ctx.fillStyle = "#38bdf8"; ctx.beginPath(); ctx.arc(x(i), y(close[i]), 4, 0, 7); ctx.fill();

  canvas._geom = { n, pad, iw };
}

function bindPriceChart() {
  const canvas = $("price-chart");
  const move = (e) => {
    if (!current || !canvas._geom) return;
    const { n, pad, iw } = canvas._geom;
    const rect = canvas.getBoundingClientRect();
    const ratio = (e.clientX - rect.left - pad.l) / iw;
    hoverIndex = Math.min(Math.max(Math.round(ratio * (n - 1)), 0), n - 1);
    drawPriceChart();
  };
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerdown", move);
  canvas.addEventListener("pointerleave", () => { hoverIndex = null; if (current) drawPriceChart(); });
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


const norm = (t) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

function matches(query, scope = inMarket) {
  const q = norm(query.trim());
  if (!q) return [];
  const starts = [], contains = [];
  for (const i of instruments.filter(scope)) {
    const sym = i.symbol.toLowerCase(), name = norm(i.name);
    if (sym.startsWith(q) || name.startsWith(q)) starts.push(i);
    else if (name.includes(q)) contains.push(i);
  }
  return [...starts, ...contains];
}

function pick(symbol) {
  $("ticker").value = symbol;
  $("suggest").hidden = true;
  analyze(symbol);
}

function showSuggestions() {
  const box = $("suggest");
  const found = matches($("ticker").value).slice(0, 8);
  box.replaceChildren();
  for (const i of found) {
    const li = el("li");
    const b = el("button", "sugg", "");
    b.type = "button";
    const tag = i.kind === "ETF" ? " · ETF" : "";
    b.append(el("strong", "", i.symbol), el("span", "", ` ${i.name}${tag}${i.analyzable ? "" : " · historique insuffisant"}`));
    b.onclick = () => pick(i.symbol);
    li.append(b);
    box.append(li);
  }
  box.hidden = found.length === 0;
}

function renderMarketFilter() {
  const box = $("market-filter");
  box.replaceChildren();
  const options = [...markets.map((m) => [m.id, m.label, m.count]), [MARKET_ALL, "Tous", instruments.length]];
  for (const [id, label, count] of options) {
    const b = el("button", "seg-btn", label);
    b.type = "button";
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", String(id === market));
    b.append(el("small", "", count));
    b.onclick = () => setMarket(id);
    box.append(b);
  }
}

function renderChips() {
  const chips = $("chips");
  chips.replaceChildren();
  const bySymbol = Object.fromEntries(instruments.map((i) => [i.symbol, i]));
  const symbols = market === MARKET_ALL ? [...(popular.MA || []).slice(0, 5), ...(popular.INTL || [])] : popular[market] || [];
  for (const sym of symbols) {
    const b = el("button", "chip", sym.replace(/\.[A-Z]+$/, ""));
    b.type = "button";
    b.title = bySymbol[sym].name;
    b.onclick = () => pick(sym);
    chips.append(b);
  }
}

function setMarket(id) {
  market = id;
  localStorage.setItem("market", id);
  renderMarketFilter();
  renderChips();
  $("suggest").hidden = true;
  if (!current) setStatus(`${instruments.filter(inMarket).length} valeurs · ${marketLabel(market)}. Recherchez un nom ou un symbole.`);
}

async function init() {
  const input = $("ticker");
  input.addEventListener("input", showSuggestions);
  document.addEventListener("click", (e) => { if (!$("search").contains(e.target)) $("suggest").hidden = true; });
  $("search").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = input.value.trim();
    const exact = instruments.find((i) => i.symbol === q.toUpperCase());
    // Dans le marché choisi d'abord ; sinon on bascule sur le marché de la valeur trouvée ailleurs.
    const target = (exact && inMarket(exact) ? exact : null) || matches(q)[0] || exact || matches(q, () => true)[0];
    if (target) {
      if (!inMarket(target)) setMarket(target.market);
      pick(target.symbol);
    } else setStatus("Valeur introuvable.", true);
    input.blur();
  });
  bindPriceChart();
  window.addEventListener("resize", () => {
    if (!current) return;
    if (current.history) drawChart();
    drawPriceChart();
  });
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});

  try {
    const res = await fetch("/api/instruments");
    const data = await res.json();
    instruments = data.instruments;
    markets = data.markets;
    popular = data.popular;
  } catch {
    setStatus("Impossible de charger la liste des valeurs.", true);
    return;
  }
  if (market !== MARKET_ALL && !markets.some((m) => m.id === market)) market = "MA";
  // Ignore un symbole mémorisé qui n'existe plus dans l'univers ; sinon on bascule sur son marché.
  const last = localStorage.getItem("lastTicker");
  const lastInstrument = instruments.find((i) => i.symbol === last);
  if (!lastInstrument) localStorage.removeItem("lastTicker");
  else if (!inMarket(lastInstrument)) market = lastInstrument.market;
  setMarket(market);
  if (lastInstrument) pick(last);
}

init();
