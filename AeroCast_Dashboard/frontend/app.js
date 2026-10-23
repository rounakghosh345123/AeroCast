/* ==========================================================================
   AeroCast dashboard - vanilla JavaScript
   Reads the exported forecast CSVs through the Node.js API and renders them.
   No ML code lives here: every number comes from /api/*.
   ========================================================================== */
'use strict';

(() => {
  /* ------------------------------------------------------------------ constants */
  const POLLUTANT_ORDER = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'NH3'];
  const POLLUTANT_COLORS = {
    'PM2.5': '#22d3ee', PM10: '#60a5fa', NO2: '#a78bfa', SO2: '#fbbf24', CO: '#34d399', O3: '#f472b6', NH3: '#2dd4bf',
  };
  const MONOGRAM = { 'PM2.5': '2.5', PM10: '10', NO2: 'NO\u2082', SO2: 'SO\u2082', CO: 'CO', O3: 'O\u2083', NH3: 'NH\u2083' };
  const VIEWS = {
    overview: { title: 'Overview', sub: 'Latest generated forecast for the selected station' },
    forecast: { title: 'Forecast', sub: 'Predicted pollutant concentrations from +1 h to +72 h' },
    pollutants: { title: 'Pollutants', sub: 'Current observation, +24 h and +72 h forecast for each pollutant' },
    aqi: { title: 'AQI Analysis', sub: 'Indicative CPCB AQI derived from the forecast concentrations' },
    trends: { title: 'Trends', sub: 'How each pollutant is forecast to move between +1 h and +72 h' },
    models: { title: 'Models', sub: 'Inventory of the pretrained XGBoost models behind the forecast' },
  };
  const EXPECTED_MODELS = 42;

  /* ------------------------------------------------------------------ state */
  const state = {
    health: null, summary: null, forecast: null, aqi: null, trends: null, stations: null,
    errors: {},
    station: null,
    selected: new Set(['PM2.5']),
    horizon: null,
    sort: { key: 'Pollutant', dir: 1 },
    view: 'overview',
    d: null, // derived data
  };
  const charts = {};

  /* ------------------------------------------------------------------ tiny helpers */
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const fmt = (v, digits) => (!isNum(v) ? 'N/A' : digits != null ? v.toFixed(digits) : Math.abs(v) < 10 ? v.toFixed(2) : v.toFixed(1));
  const fmtPct = (v) => (!isNum(v) ? 'N/A' : `${v > 0 ? '+' : ''}${v.toFixed(1)}%`);
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const rgba = (hex, a) => {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
    return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
  };
  const dateFmt = new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC' });
  const fmtUTC = (iso) => {
    const d = iso ? new Date(iso) : null;
    return d && !Number.isNaN(d.getTime()) ? `${dateFmt.format(d)} UTC` : 'N/A';
  };
  const hLabel = (h) => `+${h}h`;
  const unitOf = (name) => state.d?.byPollutant.get(name)?.unit || state.d?.trendBy.get(name)?.Unit || '';

  const icon = (id, cls = '') => `<svg class="icon ${cls}" aria-hidden="true"><use href="#i-${id}"/></svg>`;

  function catKey(category) {
    const c = String(category || '').toLowerCase();
    if (c === 'good') return 'good';
    if (c === 'satisfactory') return 'satisfactory';
    if (c.startsWith('moderate')) return 'moderate';
    if (c === 'very poor') return 'verypoor';
    if (c === 'poor') return 'poor';
    if (c === 'severe') return 'severe';
    return 'unknown';
  }
  const catColor = (category) => cssVar(`--aqi-${catKey(category)}`) || '#64748b';
  const catLabel = (category) => (catKey(category) === 'moderate' ? 'Moderate' : category || 'N/A');

  function trendInfo(trend) {
    const t = String(trend || '').toLowerCase();
    if (t.startsWith('increasing')) return { cls: 'up', icon: 'up', label: 'Increasing', key: 'up' };
    if (t.startsWith('decreasing')) return { cls: 'down', icon: 'down', label: 'Decreasing', key: 'down' };
    if (t.startsWith('approx') || t.startsWith('stable')) return { cls: 'flat', icon: 'flat', label: 'Stable', key: 'flat' };
    return { cls: 'na', icon: 'flat', label: 'N/A', key: 'na' };
  }
  const trendBadge = (trend) => {
    const t = trendInfo(trend);
    return `<span class="trend ${t.cls}">${icon(t.icon)}${t.label}</span>`;
  };

  /* ------------------------------------------------------------------ API layer */
  class ApiError extends Error {
    constructor(message, { kind = 'server', status = 0, file = null } = {}) {
      super(message);
      this.kind = kind; // network | missing_file | server
      this.status = status;
      this.file = file;
    }
  }

  async function apiGet(path, { timeout = 12000 } = {}) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeout);
    let res;
    try {
      res = await fetch(path, { headers: { Accept: 'application/json' }, cache: 'no-store', signal: ctrl.signal });
    } catch (err) {
      throw new ApiError('AeroCast backend unavailable', { kind: 'network' });
    } finally {
      clearTimeout(timer);
    }
    let body = null;
    try { body = await res.json(); } catch (_) { /* non-JSON body */ }
    if (!res.ok) {
      throw new ApiError(body?.message || `Request failed (HTTP ${res.status})`, {
        kind: body?.error === 'missing_file' ? 'missing_file' : 'server', status: res.status, file: body?.file || null,
      });
    }
    if (body === null) throw new ApiError('The server returned an invalid response.', { kind: 'server', status: res.status });
    return body;
  }

  const fetchHealth = () => apiGet('/api/health');
  const fetchSummary = () => apiGet('/api/summary');
  const fetchForecast = (station) => apiGet(station ? `/api/forecast?station=${encodeURIComponent(station)}` : '/api/forecast');
  const fetchAQI = () => apiGet('/api/aqi');
  const fetchTrends = () => apiGet('/api/trends');
  const fetchStations = () => apiGet('/api/stations');

  /* ------------------------------------------------------------------ derived data */
  function derive() {
    const rows = Array.isArray(state.forecast) ? state.forecast : [];
    const byPollutant = new Map();
    rows.forEach((r) => {
      if (!byPollutant.has(r.Pollutant)) byPollutant.set(r.Pollutant, { name: r.Pollutant, unit: r.Unit || '', values: {}, times: {} });
      const o = byPollutant.get(r.Pollutant);
      o.values[r.Horizon] = r.Predicted_Concentration;
      o.times[r.Horizon] = r.Forecast_Time;
    });
    const rank = (n) => (POLLUTANT_ORDER.includes(n) ? POLLUTANT_ORDER.indexOf(n) : 99);
    const pollutants = [...byPollutant.keys()].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
    const horizons = [...new Set(rows.map((r) => r.Horizon).filter(isNum))].sort((a, b) => a - b);
    const trendBy = new Map((Array.isArray(state.trends) ? state.trends : []).map((t) => [t.Pollutant, t]));
    state.d = { byPollutant, pollutants, horizons, trendBy };
    if (!horizons.includes(state.horizon)) state.horizon = horizons.includes(24) ? 24 : horizons[0] ?? null;
    // keep the chart selection valid
    state.selected = new Set([...state.selected].filter((p) => byPollutant.has(p)));
    if (!state.selected.size && pollutants.length) state.selected.add(pollutants.includes('PM2.5') ? 'PM2.5' : pollutants[0]);
  }

  const currentOf = (name) => (state.d?.trendBy.get(name)?.Current ?? null);
  const peakOf = (aqiRows) => {
    const valid = (aqiRows || []).filter((r) => isNum(r.AQI));
    return valid.length ? valid.reduce((a, b) => (b.AQI > a.AQI ? b : a)) : null;
  };

  /* ------------------------------------------------------------------ view helpers: skeleton / empty / error */
  const skeletonBlock = (h = 260) => `<div class="skeleton" style="height:${h}px"></div>`;
  const skeletonRows = (n = 6) => Array.from({ length: n }, () => '<div class="skeleton sk-row"></div>').join('');

  function stateBlock(kind, title, text) {
    const ic = kind === 'error' ? 'alert' : 'inbox';
    return `<div class="state ${kind === 'error' ? 'is-error' : ''}">${icon(ic)}<strong>${esc(title)}</strong><p>${esc(text)}</p></div>`;
  }
  function errorFor(key) {
    const e = state.errors[key];
    if (!e) return null;
    if (e.kind === 'missing_file') return stateBlock('error', `${e.file || 'Data file'} not found`, e.message);
    return stateBlock('error', 'Data unavailable', e.message);
  }
  const emptyState = (what) => stateBlock('empty', `No ${what} yet`, 'The exported CSV is empty. Run the AeroCast notebook to generate forecasts, then refresh.');

  function showSkeletons() {
    $('#kpis').innerHTML = Array.from({ length: 4 }, () => '<div class="skeleton sk-kpi"></div>').join('');
    $('#hero').innerHTML = skeletonBlock(150);
    $('#pollutant-cards').innerHTML = Array.from({ length: 7 }, () => '<div class="skeleton sk-card"></div>').join('');
    ['#forecast-chart-box', '#aqi-chart-box', '#trend-chart-box'].forEach((s) => { destroyChartIn(s); $(s).innerHTML = skeletonBlock(250); });
    $('#pollutant-chips').innerHTML = '';
    $('#forecast-table-body').innerHTML = skeletonRows(7);
    $('#trend-table-body').innerHTML = skeletonRows(7);
    $('#snapshot-body').innerHTML = skeletonRows(7);
    $('#dominant-body').innerHTML = skeletonRows(3);
    $('#models-body').innerHTML = skeletonRows(8);
  }

  /* ------------------------------------------------------------------ chart helpers */
  const chartReady = () => typeof window.Chart !== 'undefined';
  function destroyChart(key) {
    if (charts[key]) { charts[key].destroy(); delete charts[key]; }
  }
  function destroyChartIn(sel) {
    const key = { '#forecast-chart-box': 'forecast', '#aqi-chart-box': 'aqi', '#trend-chart-box': 'trend' }[sel];
    if (key) destroyChart(key);
  }
  function mountChart(key, box, config, label) {
    destroyChart(key);
    if (!chartReady()) {
      box.innerHTML = stateBlock('error', 'Charts unavailable', 'Chart.js could not be loaded from the CDN. Check your internet connection and reload.');
      return null;
    }
    box.innerHTML = '<canvas role="img"></canvas>';
    const canvas = box.firstElementChild;
    canvas.setAttribute('aria-label', label);
    charts[key] = new window.Chart(canvas, config);
    return charts[key];
  }
  function applyChartDefaults() {
    if (!chartReady()) return;
    const C = window.Chart;
    C.defaults.font.family = getComputedStyle(document.body).fontFamily;
    C.defaults.font.size = 12;
    C.defaults.color = cssVar('--muted');
    C.defaults.borderColor = 'rgba(148,163,184,0.12)';
  }
  const axisStyle = () => ({
    grid: { color: 'rgba(148,163,184,0.1)' },
    border: { display: false },
    ticks: { color: cssVar('--muted'), padding: 8 },
  });
  const tooltipStyle = () => ({
    backgroundColor: '#0b101a', borderColor: 'rgba(148,163,184,0.25)', borderWidth: 1, padding: 12, titleColor: '#e8edf6',
    bodyColor: '#b4bfd2', cornerRadius: 10, boxPadding: 5, displayColors: true,
  });

  /* ------------------------------------------------------------------ renderers */
  function renderStatus() {
    const el = $('#status');
    const text = $('#status-text');
    const models = state.summary?.models;
    const ready = state.health && !state.errors.forecast && (models ? models.forecasts_available === EXPECTED_MODELS : state.forecast?.length === EXPECTED_MODELS);
    if (!state.health) { el.dataset.state = 'error'; text.textContent = 'Backend offline'; }
    else if (ready) { el.dataset.state = 'ok'; text.textContent = 'Models Ready'; }
    else { el.dataset.state = 'warn'; text.textContent = 'Data incomplete'; }
    const origin = state.summary?.forecast_origin || state.forecast?.[0]?.Forecast_Origin;
    $('#origin-chip-value').textContent = fmtUTC(origin);
    $('#origin-value').textContent = fmtUTC(origin);
  }

  function renderSidebarStats() {
    const d = state.d;
    const combos = state.forecast ? state.forecast.filter((r) => isNum(r.Predicted_Concentration)).length : null;
    const set = (k, v) => { $(`[data-stat="${k}"]`).textContent = v == null ? '-' : v; };
    set('models', d ? combos : null);
    set('pollutants', d ? d.pollutants.length : null);
    set('horizons', d ? d.horizons.length : null);
  }

  function renderNotices() {
    const box = $('#notices');
    const items = [];
    Object.entries(state.errors).forEach(([key, e]) => {
      if (e.kind === 'missing_file') {
        items.push(`<div class="notice">${icon('alert')}<div><strong>Data file missing.</strong> ${esc(e.message)}</div></div>`);
      } else if (e.kind === 'server') {
        items.push(`<div class="notice">${icon('alert')}<div><strong>Could not load ${esc(key)} data.</strong> ${esc(e.message)}</div></div>`);
      }
    });
    if (state.stations && state.stations.length > 1) {
      items.push(`<div class="notice">${icon('info')}<div><strong>Multiple stations exported.</strong> AQI and trend files describe the most recent notebook run only.</div></div>`);
    }
    box.innerHTML = items.join('');
  }

  function renderStations() {
    const sel = $('#station-select');
    const list = state.stations || [];
    if (!list.length) {
      sel.innerHTML = '<option>N/A</option>';
      sel.disabled = true;
      return;
    }
    sel.innerHTML = list.map((s) => `<option value="${esc(s.id)}">${esc(s.name)}${s.name !== s.id ? ` (${esc(s.id)})` : ''}</option>`).join('');
    if (!list.some((s) => s.id === state.station)) state.station = list[0].id;
    sel.value = state.station;
    sel.disabled = list.length < 2;
    sel.title = list.length < 2 ? 'Only one station exists in the exported results' : 'Select a station';
  }

  function renderSummary() {
    // ---- KPI cards
    const kpis = $('#kpis');
    const peak = state.summary?.forecast_aqi?.peak
      ? { AQI: state.summary.forecast_aqi.peak.aqi, Category: state.summary.forecast_aqi.peak.category, Horizon: state.summary.forecast_aqi.peak.horizon, Dominant_Pollutant: state.summary.forecast_aqi.peak.dominant_pollutant }
      : peakOf(state.aqi);
    const dom = state.summary?.dominant_pollutant;
    const horizons = state.d?.horizons || [];
    const combos = state.forecast ? state.forecast.filter((r) => isNum(r.Predicted_Concentration)).length : null;
    const aqiColor = peak ? catColor(peak.Category) : cssVar('--accent');
    const domCount = dom?.most_frequent ? dom.counts?.[dom.most_frequent] : null;
    const totalH = state.aqi?.length || horizons.length;

    const card = (label, ic, value, sub, color) => `
      <article class="kpi fade-in" style="--kpi-color:${color}">
        <div class="kpi-top"><span class="kpi-label">${label}</span><span class="kpi-icon">${icon(ic)}</span></div>
        <div class="kpi-value">${value}</div>
        <div class="kpi-sub">${sub}</div>
        <span class="kpi-bar"></span>
      </article>`;

    kpis.innerHTML = [
      card('Forecast AQI', 'aqi',
        peak ? `${Math.round(peak.AQI)} <span class="pill cat-${catKey(peak.Category)}">${esc(catLabel(peak.Category))}</span>` : 'N/A',
        peak ? `Peak of the next 72 h, at ${esc(peak.Horizon)} (indicative)` : 'AQI forecast not available', aqiColor),
      card('Dominant Pollutant', 'drop',
        peak && peak.Dominant_Pollutant && peak.Dominant_Pollutant !== 'n/a' ? esc(peak.Dominant_Pollutant) : 'N/A',
        dom?.most_frequent && domCount ? `Leads at ${domCount} of ${totalH} horizons` : 'Highest sub-index at the peak', cssVar('--accent')),
      card('Forecast Horizon', 'clock',
        horizons.length ? `${horizons[horizons.length - 1]} <small>hours</small>` : 'N/A',
        horizons.length ? `${horizons.length} horizons: ${horizons.map(hLabel).join(', ')}` : 'No forecast horizons found', cssVar('--accent-2')),
      card('Models', 'models',
        combos != null && state.forecast ? `${combos} <small>XGBoost</small>` : 'N/A',
        state.d ? `${state.d.pollutants.length} pollutants \u00d7 ${horizons.length} horizons \u00b7 pretrained` : 'Model inventory not available', cssVar('--green')),
    ].join('');

    renderHero(peak);
  }

  function renderHero(peak) {
    const hero = $('#hero');
    if (state.errors.aqi) { hero.innerHTML = errorFor('aqi'); hero.style.removeProperty('--hero-color'); return; }
    const rows = state.aqi || [];
    if (!rows.length) { hero.innerHTML = emptyState('AQI forecast'); return; }
    const color = peak ? catColor(peak.Category) : cssVar('--accent');
    hero.style.setProperty('--hero-color', color);
    const dom = peak?.Dominant_Pollutant && peak.Dominant_Pollutant !== 'n/a' ? peak.Dominant_Pollutant : null;
    hero.innerHTML = `
      <div class="fade-in">
        <div class="hero-eyebrow">Forecast AQI <span title="Indicative: CPCB breakpoints applied to forecast hourly concentrations">${icon('info')}</span></div>
        <div class="hero-value">${peak ? Math.round(peak.AQI) : 'N/A'}</div>
        ${peak ? `<span class="pill hero-cat cat-${catKey(peak.Category)}">${esc(catLabel(peak.Category))}</span>` : ''}
        <div class="hero-dom"><span>Dominant pollutant</span><strong>${esc(dom || 'N/A')}</strong>
        ${peak ? `<span class="muted">Peak of the 72 h window, at ${esc(peak.Horizon)}</span>` : ''}</div>
      </div>
      <div class="hero-strip fade-in" role="list">
        ${rows.map((r) => {
          const c = catColor(r.Category);
          const isPeak = peak && r.Horizon === peak.Horizon;
          return `<div class="h-tile ${isPeak ? 'is-peak' : ''}" style="--c:${c}" role="listitem">
            <div class="h-lab">${esc(r.Horizon)}</div>
            <div class="h-val">${isNum(r.AQI) ? Math.round(r.AQI) : 'N/A'}</div>
            <div class="h-dom">${esc(r.Dominant_Pollutant && r.Dominant_Pollutant !== 'n/a' ? r.Dominant_Pollutant : 'N/A')}</div>
            <div class="h-cat">${esc(catLabel(r.Category))}</div>
          </div>`;
        }).join('')}
      </div>`;
  }

  /* ---- forecast line chart -------------------------------------------------- */
  function renderPollutantChips() {
    const box = $('#pollutant-chips');
    const names = state.d?.pollutants || [];
    box.innerHTML = names.map((n) =>
      `<button type="button" class="chip-btn" data-pollutant="${esc(n)}" style="--c:${POLLUTANT_COLORS[n] || '#22d3ee'}" aria-pressed="${state.selected.has(n)}">${esc(n)}</button>`
    ).join('');
  }

  function renderForecastChart() {
    const box = $('#forecast-chart-box');
    const sub = $('#forecast-chart-sub');
    if (state.errors.forecast) { destroyChart('forecast'); box.innerHTML = errorFor('forecast'); return; }
    if (!state.d?.pollutants.length) { destroyChart('forecast'); box.innerHTML = emptyState('forecast'); return; }
    renderPollutantChips();

    const names = [...state.selected];
    const horizons = state.d.horizons;
    const multi = names.length > 1;
    const grid = axisStyle();
    let labels; let datasets; let yTitle;

    if (!multi) {
      const name = names[0];
      const p = state.d.byPollutant.get(name);
      const cur = currentOf(name);
      const withNow = isNum(cur);
      labels = [...(withNow ? ['Now'] : []), ...horizons.map(hLabel)];
      const data = [...(withNow ? [cur] : []), ...horizons.map((h) => p.values[h] ?? null)];
      const color = POLLUTANT_COLORS[name] || '#22d3ee';
      datasets = [{
        label: name, data, borderColor: color, borderWidth: 2.5, tension: 0.35, fill: true, spanGaps: true,
        pointRadius: 4.5, pointHoverRadius: 6.5, pointBackgroundColor: '#0b101a', pointBorderColor: color, pointBorderWidth: 2,
        backgroundColor: (ctx) => {
          const { chart } = ctx; const { ctx: c, chartArea } = chart;
          if (!chartArea) return 'transparent';
          const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          g.addColorStop(0, rgba(color, 0.28)); g.addColorStop(1, rgba(color, 0));
          return g;
        },
        segment: withNow ? { borderDash: (ctx) => (ctx.p0DataIndex === 0 ? [5, 5] : undefined) } : undefined,
      }];
      yTitle = `${name} (${p.unit || unitOf(name)})`;
      sub.textContent = withNow ? 'Latest observation, then predicted concentration at each horizon' : 'Predicted concentration by forecast horizon';
    } else {
      labels = horizons.map(hLabel);
      datasets = names.map((n) => {
        const p = state.d.byPollutant.get(n);
        const base = p.values[horizons[0]];
        const color = POLLUTANT_COLORS[n] || '#22d3ee';
        return {
          label: n, borderColor: color, backgroundColor: color, borderWidth: 2.2, tension: 0.35, pointRadius: 3.5, pointHoverRadius: 5.5,
          data: horizons.map((h) => (isNum(base) && base > 0 && isNum(p.values[h]) ? (p.values[h] / base) * 100 : null)),
          actual: horizons.map((h) => p.values[h] ?? null), unit: p.unit,
        };
      });
      yTitle = `Index (${hLabel(horizons[0])} forecast = 100)`;
      sub.textContent = 'Several pollutants selected: shown as an index because their units differ';
    }

    mountChart('forecast', box, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
        animation: { duration: 500 },
        plugins: {
          legend: { display: multi, labels: { usePointStyle: true, boxWidth: 8, color: cssVar('--text-2') } },
          tooltip: {
            ...tooltipStyle(),
            callbacks: {
              label: (c) => {
                if (c.raw == null) return `${c.dataset.label}: N/A`;
                if (!multi) return `${c.dataset.label}: ${fmt(c.raw)} ${state.d.byPollutant.get(c.dataset.label)?.unit || ''}`;
                const a = c.dataset.actual[c.dataIndex];
                return `${c.dataset.label}: ${fmt(a)} ${c.dataset.unit || ''}  (index ${c.raw.toFixed(0)})`;
              },
            },
          },
        },
        scales: {
          x: { ...grid, grid: { display: false } },
          y: { ...grid, beginAtZero: false, title: { display: true, text: yTitle, color: cssVar('--muted') } },
        },
      },
    }, `Line chart of the forecast for ${names.join(', ')}`);
  }

  /* ---- pollutant cards ----------------------------------------------------- */
  function renderPollutantCards() {
    const box = $('#pollutant-cards');
    if (state.errors.forecast) { box.innerHTML = errorFor('forecast'); return; }
    const names = state.d?.pollutants || [];
    if (!names.length) { box.innerHTML = emptyState('pollutant forecast'); return; }
    box.innerHTML = names.map((n) => {
      const p = state.d.byPollutant.get(n);
      const t = state.d.trendBy.get(n);
      const cur = currentOf(n);
      const unit = p.unit || t?.Unit || '';
      return `<button type="button" class="p-card fade-in" data-detail="${esc(n)}" style="--c:${POLLUTANT_COLORS[n] || '#22d3ee'}" aria-label="${esc(n)} forecast details">
        <div class="p-head">
          <span class="p-mono">${esc(MONOGRAM[n] || n.slice(0, 3))}</span>
          <div><div class="p-name">${esc(n)}</div><div class="p-unit">${esc(unit)}</div></div>
          <span class="p-trend">${t ? trendBadge(t.Trend) : '<span class="trend na">N/A</span>'}</span>
        </div>
        <div class="p-current"><strong>${fmt(cur)}</strong><span>${isNum(cur) ? esc(unit) : ''} latest observed</span></div>
        <div class="p-rows">
          <div><small>+24h</small><b>${fmt(p.values[24])}</b></div>
          <div><small>+72h</small><b>${fmt(p.values[72])}</b></div>
        </div>
      </button>`;
    }).join('');
  }

  /* ---- forecast table ------------------------------------------------------ */
  function sortedPollutants() {
    const { key, dir } = state.sort;
    const names = [...state.d.pollutants];
    const val = (n) => {
      if (key === 'Pollutant') return POLLUTANT_ORDER.includes(n) ? POLLUTANT_ORDER.indexOf(n) : 99;
      if (key === 'Trend') return state.d.trendBy.get(n)?.Change_Percentage ?? -Infinity;
      return state.d.byPollutant.get(n).values[Number(key.slice(2))] ?? -Infinity;
    };
    return names.sort((a, b) => (val(a) - val(b)) * dir || a.localeCompare(b));
  }

  function renderForecastTable() {
    const box = $('#forecast-table-body');
    if (state.errors.forecast) { box.innerHTML = errorFor('forecast'); return; }
    if (!state.d?.pollutants.length) { box.innerHTML = emptyState('forecast'); return; }
    const hs = state.d.horizons;
    const th = (key, label) => {
      const active = state.sort.key === key;
      return `<th scope="col" aria-sort="${active ? (state.sort.dir === 1 ? 'ascending' : 'descending') : 'none'}">
        <button type="button" data-sort="${key}">${label}<span class="sort" aria-hidden="true">${active && state.sort.dir === -1 ? '\u25BC' : '\u25B2'}</span></button></th>`;
    };
    const rows = sortedPollutants().map((n) => {
      const p = state.d.byPollutant.get(n);
      const t = state.d.trendBy.get(n);
      const vals = hs.map((h) => p.values[h]);
      const max = Math.max(...vals.filter(isNum));
      return `<tr class="clickable" tabindex="0" data-detail="${esc(n)}">
        <td class="cell-name">${esc(n)}<span class="u">${esc(p.unit)}</span></td>
        ${hs.map((h) => `<td class="${isNum(p.values[h]) && p.values[h] === max ? 'peak' : ''}">${fmt(p.values[h])}</td>`).join('')}
        <td>${t ? trendBadge(t.Trend) : '<span class="trend na">N/A</span>'}</td>
      </tr>`;
    }).join('');
    box.innerHTML = `<div class="table-wrap fade-in"><table>
      <thead><tr>${th('Pollutant', 'Pollutant')}${hs.map((h) => th(`h:${h}`, hLabel(h))).join('')}${th('Trend', 'Trend')}</tr></thead>
      <tbody>${rows}</tbody></table></div>`;
  }

  /* ---- horizon snapshot (Forecast view) ------------------------------------ */
  function renderHorizonSnapshot() {
    const seg = $('#horizon-select');
    const body = $('#snapshot-body');
    if (state.errors.forecast) { seg.innerHTML = ''; body.innerHTML = errorFor('forecast'); return; }
    const hs = state.d?.horizons || [];
    if (!hs.length) { seg.innerHTML = ''; body.innerHTML = emptyState('forecast'); return; }
    seg.innerHTML = hs.map((h) => `<button type="button" data-horizon="${h}" aria-pressed="${h === state.horizon}">${hLabel(h)}</button>`).join('');
    body.innerHTML = `<div class="fade-in">${state.d.pollutants.map((n) => {
      const p = state.d.byPollutant.get(n);
      const v = p.values[state.horizon];
      const cur = currentOf(n);
      const top = Math.max(...[v, cur].filter(isNum), 0) || 1;
      const delta = isNum(v) && isNum(cur) && cur > 0 ? ((v - cur) / cur) * 100 : null;
      const dcls = delta == null ? 'na' : delta > 5 ? 'up' : delta < -5 ? 'down' : 'flat';
      return `<div class="snap-row">
        <span class="snap-name">${esc(n)}</span>
        <span class="snap-val">${fmt(v)}<small>${esc(p.unit)}</small></span>
        <span class="snap-bar" title="Bar: forecast. Marker: latest observation">
          <i style="left:0;width:${isNum(v) ? (v / top) * 100 : 0}%;background:${POLLUTANT_COLORS[n] || 'var(--accent)'}"></i>
          ${isNum(cur) ? `<b style="position:absolute;top:-2px;bottom:-2px;width:2px;background:#e8edf6;left:${(cur / top) * 100}%"></b>` : ''}
        </span>
        <span class="snap-delta trend ${dcls}">${fmtPct(delta)}</span>
      </div>`;
    }).join('')}<p class="note">${icon('info')}<span>Bar = forecast at ${esc(hLabel(state.horizon))}; white marker = latest observation; percentage = change versus the latest observation.</span></p></div>`;
  }

  /* ---- AQI ------------------------------------------------------------------ */
  function renderAQIChart() {
    const box = $('#aqi-chart-box');
    const basis = $('#aqi-basis');
    basis.innerHTML = '';
    if (state.errors.aqi) { destroyChart('aqi'); box.innerHTML = errorFor('aqi'); return; }
    const rows = state.aqi || [];
    if (!rows.length) { destroyChart('aqi'); box.innerHTML = emptyState('AQI forecast'); return; }
    const guides = [[50, 'Good'], [100, 'Satisfactory'], [200, 'Moderate'], [300, 'Poor'], [400, 'Very Poor']];
    const maxAqi = Math.max(...rows.map((r) => (isNum(r.AQI) ? r.AQI : 0)), 100);
    const yMax = Math.ceil((maxAqi * 1.18) / 50) * 50;
    const wide = window.innerWidth > 760; // category guide labels need room in the right margin

    const plugin = {
      id: 'aqiDecor',
      beforeDatasetsDraw(chart) {
        const { ctx, chartArea: a, scales: { y } } = chart;
        ctx.save();
        ctx.setLineDash([4, 5]); ctx.lineWidth = 1; ctx.font = '10.5px ' + getComputedStyle(document.body).fontFamily; ctx.textAlign = 'left';
        guides.forEach(([v, label]) => {
          if (v >= y.max) return;
          const py = y.getPixelForValue(v);
          ctx.strokeStyle = 'rgba(148,163,184,0.22)'; ctx.beginPath(); ctx.moveTo(a.left, py); ctx.lineTo(a.right, py); ctx.stroke();
          if (wide) { ctx.fillStyle = 'rgba(148,163,184,0.75)'; ctx.fillText(`${label} \u2264${v}`, a.right + 8, py + 3.5); }
        });
        ctx.restore();
      },
      afterDatasetsDraw(chart) {
        const { ctx } = chart;
        const meta = chart.getDatasetMeta(0);
        ctx.save();
        ctx.font = '600 12px ' + getComputedStyle(document.body).fontFamily; ctx.textAlign = 'center'; ctx.fillStyle = '#e8edf6';
        meta.data.forEach((bar, i) => { if (isNum(rows[i].AQI)) ctx.fillText(String(Math.round(rows[i].AQI)), bar.x, bar.y - 8); });
        ctx.restore();
      },
    };

    mountChart('aqi', box, {
      type: 'bar',
      data: {
        labels: rows.map((r) => r.Horizon),
        datasets: [{
          label: 'AQI', data: rows.map((r) => (isNum(r.AQI) ? r.AQI : null)),
          backgroundColor: rows.map((r) => rgba(catColor(r.Category), 0.85)), borderColor: rows.map((r) => catColor(r.Category)),
          borderWidth: 1, borderRadius: 7, maxBarThickness: 54,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
        layout: { padding: { right: wide ? 96 : 0 } },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(),
            callbacks: {
              title: (items) => { const r = rows[items[0].dataIndex]; return `${r.Horizon} \u00b7 ${fmtUTC(r.Forecast_Time)}`; },
              label: (c) => { const r = rows[c.dataIndex]; return [`AQI: ${isNum(r.AQI) ? Math.round(r.AQI) : 'N/A'}`, `Category: ${catLabel(r.Category)}`, `Dominant: ${r.Dominant_Pollutant || 'N/A'}`]; },
              labelColor: (c) => ({ borderColor: catColor(rows[c.dataIndex].Category), backgroundColor: catColor(rows[c.dataIndex].Category), borderRadius: 3 }),
            },
          },
        },
        scales: {
          x: { ...axisStyle(), grid: { display: false } },
          y: { ...axisStyle(), min: 0, max: yMax, title: { display: true, text: 'AQI', color: cssVar('--muted') } },
        },
      },
      plugins: [plugin],
    }, 'Bar chart of forecast AQI by horizon');

    const b = state.summary?.forecast_aqi?.basis || rows.find((r) => r.Basis)?.Basis;
    if (b) basis.innerHTML = `${icon('info')}<span>AQI basis - ${esc(b)}. Official CPCB AQI uses 24 h / 8 h averages, so treat these values as an indicative trend, not a bulletin value.</span>`;
  }

  function renderDominant() {
    const box = $('#dominant-body');
    if (state.errors.aqi) { box.innerHTML = errorFor('aqi'); return; }
    const rows = state.aqi || [];
    if (!rows.length) { box.innerHTML = emptyState('AQI forecast'); return; }
    const counts = {};
    rows.forEach((r) => { if (r.Dominant_Pollutant && r.Dominant_Pollutant !== 'n/a') counts[r.Dominant_Pollutant] = (counts[r.Dominant_Pollutant] || 0) + 1; });
    const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
    box.innerHTML = `<div class="fade-in">
      <div class="dom-grid">${rows.map((r) => `
        <div class="h-tile" style="--c:${catColor(r.Category)}">
          <div class="h-lab">${esc(r.Horizon)}</div>
          <div class="h-val">${esc(r.Dominant_Pollutant && r.Dominant_Pollutant !== 'n/a' ? r.Dominant_Pollutant : 'N/A')}</div>
          <div class="h-dom">AQI ${isNum(r.AQI) ? Math.round(r.AQI) : 'N/A'}</div>
          <div class="h-cat">${esc(catLabel(r.Category))}</div>
        </div>`).join('')}</div>
      <div class="table-wrap"><table>
        <thead><tr><th scope="col">Horizon</th><th scope="col">Forecast time</th><th scope="col">AQI</th><th scope="col">Category</th><th scope="col">Dominant pollutant</th></tr></thead>
        <tbody>${rows.map((r) => `<tr><td class="cell-name">${esc(r.Horizon)}</td><td>${esc(fmtUTC(r.Forecast_Time))}</td>
          <td>${isNum(r.AQI) ? Math.round(r.AQI) : 'N/A'}</td><td><span class="pill cat-${catKey(r.Category)}">${esc(catLabel(r.Category))}</span></td>
          <td>${esc(r.Dominant_Pollutant || 'N/A')}</td></tr>`).join('')}</tbody></table></div>
      ${top ? `<p class="note">${icon('info')}<span>${esc(top[0])} is the dominant pollutant at ${top[1]} of ${rows.length} horizons.</span></p>` : ''}
    </div>`;
  }

  /* ---- trends --------------------------------------------------------------- */
  function renderTrendAnalysis() {
    const tbody = $('#trend-table-body');
    const cbox = $('#trend-chart-box');
    if (state.errors.trends) { tbody.innerHTML = errorFor('trends'); destroyChart('trend'); cbox.innerHTML = errorFor('trends'); return; }
    const rows = [...(state.trends || [])].sort((a, b) => POLLUTANT_ORDER.indexOf(a.Pollutant) - POLLUTANT_ORDER.indexOf(b.Pollutant));
    if (!rows.length) { tbody.innerHTML = emptyState('trend analysis'); destroyChart('trend'); cbox.innerHTML = emptyState('trend analysis'); return; }

    tbody.innerHTML = `<div class="table-wrap compact fade-in"><table>
      <thead><tr><th scope="col">Pollutant</th><th scope="col">Current</th><th scope="col">+1h</th><th scope="col">+72h</th><th scope="col">Change %</th><th scope="col">Trend</th></tr></thead>
      <tbody>${rows.map((r) => `<tr>
        <td class="cell-name">${esc(r.Pollutant)}<span class="u">${esc(r.Unit || '')}</span></td>
        <td>${fmt(r.Current)}</td><td>${fmt(r['+1h'])}</td><td>${fmt(r['+72h'])}</td>
        <td>${fmtPct(r.Change_Percentage)}</td><td>${trendBadge(r.Trend)}</td>
      </tr>`).join('')}</tbody></table></div>`;

    const colorFor = (t) => ({ up: cssVar('--amber'), down: cssVar('--green'), flat: '#64748b', na: '#475569' }[trendInfo(t).key]);
    const plugin = {
      id: 'trendLabels',
      afterDatasetsDraw(chart) {
        const { ctx } = chart; const meta = chart.getDatasetMeta(0);
        ctx.save(); ctx.font = '600 11.5px ' + getComputedStyle(document.body).fontFamily; ctx.fillStyle = '#e8edf6'; ctx.textBaseline = 'middle';
        meta.data.forEach((bar, i) => {
          const v = rows[i].Change_Percentage; if (!isNum(v)) return;
          ctx.textAlign = v >= 0 ? 'left' : 'right';
          ctx.fillText(fmtPct(v), v >= 0 ? bar.x + 8 : bar.x - 8, bar.y);
        });
        ctx.restore();
      },
    };
    const vals = rows.map((r) => r.Change_Percentage).filter(isNum);
    const pad = Math.max(10, (Math.max(...vals.map(Math.abs), 10)) * 0.22);
    const axMin = Math.floor((Math.min(0, ...vals) - pad) / 10) * 10;
    const axMax = Math.ceil((Math.max(0, ...vals) + pad) / 10) * 10;
    mountChart('trend', cbox, {
      type: 'bar',
      data: {
        labels: rows.map((r) => r.Pollutant),
        datasets: [{
          label: 'Change %', data: rows.map((r) => (isNum(r.Change_Percentage) ? r.Change_Percentage : null)),
          backgroundColor: rows.map((r) => rgba(colorFor(r.Trend), 0.8)), borderColor: rows.map((r) => colorFor(r.Trend)), borderWidth: 1, borderRadius: 6, maxBarThickness: 26,
        }],
      },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
        plugins: {
          legend: { display: false },
          tooltip: {
            ...tooltipStyle(),
            callbacks: {
              label: (c) => { const r = rows[c.dataIndex]; return [`Change: ${fmtPct(r.Change_Percentage)}`, `+1h: ${fmt(r['+1h'])} ${r.Unit || ''}`, `+72h: ${fmt(r['+72h'])} ${r.Unit || ''}`, `Trend: ${trendInfo(r.Trend).label}`]; },
            },
          },
        },
        scales: {
          x: { ...axisStyle(), min: axMin, max: axMax, title: { display: true, text: 'Change from +1h to +72h (%)', color: cssVar('--muted') } },
          y: { ...axisStyle(), grid: { display: false }, ticks: { color: cssVar('--text-2'), padding: 8 } },
        },
      },
      plugins: [plugin],
    }, 'Horizontal bar chart of forecast change percentage per pollutant');
  }

  /* ---- models --------------------------------------------------------------- */
  function renderModelStatus() {
    const box = $('#models-body');
    if (state.errors.forecast) { box.innerHTML = errorFor('forecast'); return; }
    if (!state.forecast || !state.forecast.length) { box.innerHTML = emptyState('model outputs'); return; }
    const { pollutants, horizons, byPollutant } = state.d;
    const have = (n, h) => isNum(byPollutant.get(n)?.values[h]);
    const count = pollutants.reduce((s, n) => s + horizons.filter((h) => have(n, h)).length, 0);
    box.innerHTML = `<div class="fade-in">
      <div class="model-hero">
        <div class="model-count">${count}</div>
        <div><strong>Pretrained XGBoost Models</strong>${count === EXPECTED_MODELS ? 'All models produced a forecast in the latest run' : `${count} of ${EXPECTED_MODELS} expected forecasts are present`}</div>
        <div class="facts">
          <div class="fact"><strong>${pollutants.length}</strong><span>Pollutants</span></div>
          <div class="fact"><strong>${horizons.length}</strong><span>Horizons</span></div>
          <div class="fact"><strong>${count}</strong><span>Models</span></div>
        </div>
      </div>
      <div class="model-grid" role="table" aria-label="Model inventory">
        <div class="m-row head" role="row"><span class="m-label" role="columnheader">Pollutant</span>${horizons.map((h) => `<span class="m-cell hd" role="columnheader">${hLabel(h)}</span>`).join('')}</div>
        ${pollutants.map((n) => `<div class="m-row" role="row"><span class="m-label" role="rowheader">${esc(n)}</span>${horizons.map((h) =>
          have(n, h)
            ? `<span class="m-cell ok" role="cell" title="${esc(n)} ${hLabel(h)}: pretrained model">${icon('check')}<small>Pretrained</small></span>`
            : `<span class="m-cell missing" role="cell" title="No forecast for ${esc(n)} ${hLabel(h)} in the latest export">\u2014<small>Missing</small></span>`
        ).join('')}</div>`).join('')}
      </div>
      <p class="note">${icon('info')}<span>This is an inventory view. Each tick means a pretrained model produced a forecast in the latest exported run; the dashboard does not load, run or train models.</span></p>
    </div>`;
  }

  /* ---- detail modal --------------------------------------------------------- */
  function openDetail(name) {
    const p = state.d?.byPollutant.get(name);
    if (!p) return;
    const t = state.d.trendBy.get(name);
    const cur = currentOf(name);
    const unit = p.unit || t?.Unit || '';
    const hs = state.d.horizons;
    const body = $('#detail-body');
    body.innerHTML = `
      <div class="modal-head">
        <span class="p-mono" style="--c:${POLLUTANT_COLORS[name] || '#22d3ee'}">${esc(MONOGRAM[name] || name.slice(0, 3))}</span>
        <h2 id="detail-title">${esc(name)} Forecast</h2>
        <button type="button" class="icon-btn" data-close aria-label="Close">${icon('x')}</button>
      </div>
      <div class="modal-top">
        <div><small>Current (latest observed)</small><div class="big">${fmt(cur)} <span class="muted" style="font-size:14px">${isNum(cur) ? esc(unit) : ''}</span></div></div>
        <div><small>Trend (+1h to +72h)</small>${t ? `${trendBadge(t.Trend)} <span class="muted">${fmtPct(t.Change_Percentage)}</span>` : '<span class="trend na">N/A</span>'}</div>
        ${t && isNum(t['Change_vs_Current_%']) ? `<div><small>+72h vs current</small>${trendBadge(t.Trend_vs_Current)} <span class="muted">${fmtPct(t['Change_vs_Current_%'])}</span></div>` : ''}
      </div>
      <div class="chart-box" id="modal-chart-box" style="height:230px"></div>
      <div class="modal-values">${hs.map((h) => `<div class="mv"><small>${hLabel(h)}</small><b>${fmt(p.values[h])}</b><span>${esc(unit)} \u00b7 ${esc(fmtUTC(p.times[h]))}</span></div>`).join('')}</div>`;

    const color = POLLUTANT_COLORS[name] || '#22d3ee';
    const withNow = isNum(cur);
    $('#detail-modal').showModal();
    mountChart('modal', $('#modal-chart-box'), {
      type: 'line',
      data: {
        labels: [...(withNow ? ['Now'] : []), ...hs.map(hLabel)],
        datasets: [{
          label: name, data: [...(withNow ? [cur] : []), ...hs.map((h) => p.values[h] ?? null)], borderColor: color, borderWidth: 2.5, tension: 0.35, pointRadius: 4, pointBackgroundColor: '#0b101a', pointBorderColor: color, pointBorderWidth: 2,
          backgroundColor: rgba(color, 0.12), fill: true, spanGaps: true,
          segment: withNow ? { borderDash: (ctx) => (ctx.p0DataIndex === 0 ? [5, 5] : undefined) } : undefined,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 400 },
        plugins: { legend: { display: false }, tooltip: { ...tooltipStyle(), callbacks: { label: (c) => `${fmt(c.raw)} ${unit}` } } },
        scales: { x: { ...axisStyle(), grid: { display: false } }, y: { ...axisStyle(), title: { display: true, text: unit, color: cssVar('--muted') } } },
      },
    }, `Line chart of the ${name} forecast`);
  }

  /* ------------------------------------------------------------------ views / navigation */
  function applyView() {
    const v = VIEWS[state.view] ? state.view : 'overview';
    document.body.dataset.view = v;
    $$('[data-views]').forEach((el) => {
      const views = el.dataset.views.split(' ');
      el.hidden = !(views.includes('all') || views.includes(v));
    });
    $$('.nav-link').forEach((a) => {
      if (a.dataset.nav === v) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    $('#charts-row').classList.toggle('single', v !== 'overview');
    $('#page-title').textContent = VIEWS[v].title;
    $('#page-sub').textContent = VIEWS[v].sub;
    document.title = `${VIEWS[v].title} - AeroCast`;
    // charts created while hidden need a resize once visible
    requestAnimationFrame(() => Object.values(charts).forEach((c) => c.resize()));
  }

  function routeFromHash() {
    const v = location.hash.replace('#', '');
    state.view = VIEWS[v] ? v : 'overview';
    applyView();
    closeDrawer();
    window.scrollTo({ top: 0 });
  }

  function openDrawer() {
    $('#sidebar').classList.add('open'); $('#scrim').hidden = false; $('#menu-btn').setAttribute('aria-expanded', 'true');
  }
  function closeDrawer() {
    $('#sidebar').classList.remove('open'); $('#scrim').hidden = true; $('#menu-btn').setAttribute('aria-expanded', 'false');
  }

  let toastTimer;
  function toast(msg) {
    const t = $('#toast');
    t.textContent = msg; t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, 2600);
  }

  /* ------------------------------------------------------------------ data loading */
  function renderFatal() {
    $('#content').classList.add('is-fatal');
    const f = $('#fatal');
    f.hidden = false;
    f.innerHTML = `<div class="card fatal-card">
      <span class="badge">${icon('alert')}</span>
      <h2>AeroCast backend unavailable</h2>
      <p class="muted">Make sure the Node.js server is running.</p>
      <p class="muted">Open a terminal in the <code>backend</code> folder and run <code>npm start</code>, then open <code>http://localhost:3000</code>.</p>
      <button type="button" class="btn" id="retry-btn">${icon('refresh')}<span>Try again</span></button>
    </div>`;
    $('#retry-btn').addEventListener('click', () => loadAll());
  }

  const settleOne = async (key, promise) => {
    try { return { key, ok: true, value: await promise }; } catch (e) { return { key, ok: false, error: e }; }
  };

  async function loadAll({ refresh = false } = {}) {
    const btn = $('#refresh-btn');
    btn.disabled = true; btn.classList.add('is-loading');
    $('#fatal').hidden = true; $('#content').classList.remove('is-fatal');
    $('#status').dataset.state = 'loading'; $('#status-text').textContent = refresh ? 'Refreshing' : 'Connecting';
    showSkeletons();

    // 1. health
    try {
      state.health = await fetchHealth();
    } catch (e) {
      state.health = null; state.errors = {};
      renderStatus(); renderFatal();
      btn.disabled = false; btn.classList.remove('is-loading');
      return;
    }

    // 2-6. summary, forecast, aqi, trends, stations
    const results = await Promise.all([
      settleOne('summary', fetchSummary()),
      settleOne('forecast', fetchForecast(state.station)),
      settleOne('aqi', fetchAQI()),
      settleOne('trends', fetchTrends()),
      settleOne('stations', fetchStations()),
    ]);
    state.errors = {};
    results.forEach((r) => {
      if (r.ok) state[r.key] = r.value; else { state[r.key] = null; state.errors[r.key] = r.error; }
    });
    // every request failing at the network level means the backend vanished
    if (results.every((r) => !r.ok && r.error.kind === 'network')) {
      state.health = null; renderStatus(); renderFatal();
      btn.disabled = false; btn.classList.remove('is-loading');
      return;
    }
    // `summary` and `stations` are conveniences; their failure alone is not shown as a section error
    delete state.errors.summary; delete state.errors.stations;
    renderAll();
    btn.disabled = false; btn.classList.remove('is-loading');
    if (refresh) toast('Data refreshed from the exported CSV files');
  }

  function renderAll() {
    derive();
    renderStatus();
    renderSidebarStats();
    renderNotices();
    renderStations();
    renderSummary();
    renderForecastChart();
    renderPollutantCards();
    renderForecastTable();
    renderHorizonSnapshot();
    renderAQIChart();
    renderDominant();
    renderTrendAnalysis();
    renderModelStatus();
    applyView();
  }

  /* ------------------------------------------------------------------ events */
  function bindEvents() {
    $('#refresh-btn').addEventListener('click', () => loadAll({ refresh: true }));
    $('#menu-btn').addEventListener('click', () => ($('#sidebar').classList.contains('open') ? closeDrawer() : openDrawer()));
    $('#scrim').addEventListener('click', closeDrawer);
    window.addEventListener('hashchange', routeFromHash);
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeDrawer(); });

    $('#station-select').addEventListener('change', async (e) => {
      state.station = e.target.value;
      try { state.forecast = await fetchForecast(state.station); delete state.errors.forecast; } catch (err) { state.errors.forecast = err; }
      derive(); renderSidebarStats(); renderForecastChart(); renderPollutantCards(); renderForecastTable(); renderHorizonSnapshot(); renderModelStatus(); renderSummary(); renderStatus();
    });

    $('#pollutant-chips').addEventListener('click', (e) => {
      const b = e.target.closest('[data-pollutant]'); if (!b) return;
      const n = b.dataset.pollutant;
      if (state.selected.has(n)) { if (state.selected.size > 1) state.selected.delete(n); } else state.selected.add(n);
      renderForecastChart();
    });

    $('#horizon-select').addEventListener('click', (e) => {
      const b = e.target.closest('[data-horizon]'); if (!b) return;
      state.horizon = Number(b.dataset.horizon); renderHorizonSnapshot();
    });

    $('#forecast-table-body').addEventListener('click', (e) => {
      const s = e.target.closest('[data-sort]');
      if (s) {
        const key = s.dataset.sort;
        state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: key === 'Pollutant' ? 1 : -1 };
        renderForecastTable(); return;
      }
      const r = e.target.closest('[data-detail]'); if (r) openDetail(r.dataset.detail);
    });
    $('#forecast-table-body').addEventListener('keydown', (e) => {
      if (e.key !== 'Enter' && e.key !== ' ') return;
      const r = e.target.closest('tr[data-detail]'); if (r) { e.preventDefault(); openDetail(r.dataset.detail); }
    });
    $('#pollutant-cards').addEventListener('click', (e) => {
      const c = e.target.closest('[data-detail]'); if (c) openDetail(c.dataset.detail);
    });

    const modal = $('#detail-modal');
    modal.addEventListener('click', (e) => { if (e.target === modal || e.target.closest('[data-close]')) modal.close(); });
    modal.addEventListener('close', () => destroyChart('modal'));
  }

  /* ------------------------------------------------------------------ init */
  function init() {
    applyChartDefaults();
    bindEvents();
    routeFromHash();
    loadAll();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
})();
