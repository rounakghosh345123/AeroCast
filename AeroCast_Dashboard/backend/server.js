'use strict';

/**
 * AeroCast - lightweight backend
 *
 * Python/Jupyter (42 XGBoost models) --> evaluation/*.csv --> this server --> REST API --> frontend
 *
 * Responsibilities
 *   - serve ../frontend as static files (and ONLY that folder)
 *   - expose the exported forecast CSVs as JSON through a small REST API
 * Non-responsibilities
 *   - no machine learning, no model loading, no Python execution
 *
 * Only three known files are ever read. No API parameter is ever used to build a file path.
 */

const path = require('path');
const fs = require('fs');
const express = require('express');
const { parse } = require('csv-parse/sync');

const PORT = Number(process.env.PORT) || 3000;

const ROOT_DIR = path.resolve(__dirname, '..');
const EVALUATION_DIR = path.resolve(ROOT_DIR, '..', 'evaluation');
const FRONTEND_DIR = path.join(ROOT_DIR, 'frontend');


/* ------------------------------------------------------------------ *
 * Data layer                                                          *
 * ------------------------------------------------------------------ *
 * Everything the API needs goes through `dataProvider`. Today it reads
 * CSV files. To move to a Python inference service later, replace the
 * three get*() methods (and nothing else) with HTTP calls to that
 * service, returning the same row shapes.
 * ------------------------------------------------------------------ */

const SOURCES = {
  forecast: {
    file: 'aerocast_forecast_results.csv',
    required: ['Station_ID', 'Station_Name', 'Pollutant', 'Horizon', 'Forecast_Origin', 'Forecast_Time', 'Predicted_Concentration'],
    numeric: ['Horizon', 'Predicted_Concentration'],
    timestamps: ['Forecast_Origin', 'Forecast_Time'],
  },
  aqi: {
    file: 'aerocast_aqi_forecast.csv',
    required: ['Horizon', 'Forecast_Time', 'AQI', 'Category', 'Dominant_Pollutant'],
    numeric: ['AQI'],
    timestamps: ['Forecast_Time'],
  },
  trends: {
    file: 'aerocast_trend_analysis.csv',
    required: ['Pollutant', 'Current', '+1h', '+72h', 'Change', 'Change_Percentage', 'Trend'],
    numeric: ['Current', '+1h', '+72h', 'Change', 'Change_Percentage', 'Change_vs_Current_%'],
    timestamps: [],
  },
};

class DataError extends Error {
  constructor(code, message, status, file) {
    super(message);
    this.code = code;
    this.status = status;
    this.file = file;
  }
}

function toNumber(value) {
  if (value === undefined || value === null || String(value).trim() === '') return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function toIso(value) {
  if (!value) return null;
  const d = new Date(String(value).trim().replace(' ', 'T'));
  return Number.isNaN(d.getTime()) ? String(value) : d.toISOString();
}

function readCsv(key) {
  const spec = SOURCES[key];
  const filePath = path.join(EVALUATION_DIR, spec.file); // spec.file is a constant, never user input

  let text;
  let stat;
  try {
    stat = fs.statSync(filePath);
    text = fs.readFileSync(filePath, 'utf8'); // read fresh on every request
  } catch (err) {
    if (err.code === 'ENOENT') {
      throw new DataError(
        'missing_file',
        `${spec.file} was not found in the evaluation folder. Run the AeroCast notebook (Section 18) to export it.`,
        404,
        spec.file
      );
    }
    throw new DataError('read_error', `Could not read ${spec.file}: ${err.code || err.message}`, 500, spec.file);
  }

  let records;
  try {
    records = parse(text, { columns: true, skip_empty_lines: true, bom: true, trim: true });
  } catch (err) {
    throw new DataError('parse_error', `${spec.file} is not valid CSV: ${err.message}`, 500, spec.file);
  }

  if (records.length > 0) {
    const missing = spec.required.filter((col) => !(col in records[0]));
    if (missing.length) {
      throw new DataError(
        'schema_mismatch',
        `${spec.file} is missing expected column(s): ${missing.join(', ')}. Re-export it from the notebook.`,
        500,
        spec.file
      );
    }
  }

  const rows = records.map((rec) => {
    const row = { ...rec };
    spec.numeric.forEach((col) => {
      if (col in row) row[col] = toNumber(row[col]);
    });
    spec.timestamps.forEach((col) => {
      if (col in row) row[col] = toIso(row[col]);
    });
    if (key === 'forecast') row.Horizon = row.Horizon === null ? null : Math.round(row.Horizon);
    if (key === 'aqi') {
      const m = /^\+?(\d+)h$/i.exec(String(row.Horizon));
      row.Horizon_h = m ? Number(m[1]) : null;
    }
    return row;
  });

  return { rows, updated: stat.mtime.toISOString(), file: spec.file };
}

const dataProvider = {
  async getForecast() { return readCsv('forecast'); },
  async getAqi() { return readCsv('aqi'); },
  async getTrends() { return readCsv('trends'); },
};

/* ------------------------------------------------------------------ *
 * Derived data                                                        *
 * ------------------------------------------------------------------ */

function uniqueStations(forecastRows) {
  const seen = new Map();
  forecastRows.forEach((r) => {
    if (r.Station_ID && !seen.has(r.Station_ID)) {
      seen.set(r.Station_ID, { id: r.Station_ID, name: r.Station_Name || r.Station_ID });
    }
  });
  return [...seen.values()];
}

function countBy(items, keyFn) {
  return items.reduce((acc, item) => {
    const k = keyFn(item);
    if (k !== null && k !== undefined && k !== '') acc[k] = (acc[k] || 0) + 1;
    return acc;
  }, {});
}

async function settle(promise) {
  try {
    return { ok: true, value: await promise };
  } catch (err) {
    if (err instanceof DataError) return { ok: false, error: err };
    throw err;
  }
}

async function buildSummary() {
  const [fc, aq, tr] = await Promise.all([
    settle(dataProvider.getForecast()),
    settle(dataProvider.getAqi()),
    settle(dataProvider.getTrends()),
  ]);

  const missingFiles = [fc, aq, tr].filter((r) => !r.ok).map((r) => r.error.file);
  const errors = [fc, aq, tr].filter((r) => !r.ok).map((r) => ({ file: r.error.file, error: r.error.code, message: r.error.message }));

  const summary = {
    project: 'AeroCast',
    station: null,
    forecast_origin: null,
    generated_at: null,
    current_values: null,
    forecast_aqi: null,
    dominant_pollutant: null,
    overall_trend: null,
    models: null,
    sources: {},
    missing_files: missingFiles,
    errors,
  };

  if (fc.ok) {
    const rows = fc.value.rows;
    const stations = uniqueStations(rows);
    summary.station = stations[0] || null;
    summary.forecast_origin = rows.find((r) => r.Forecast_Origin)?.Forecast_Origin || null;
    summary.generated_at = fc.value.updated;
    summary.sources.forecast = { file: fc.value.file, rows: rows.length, updated: fc.value.updated };

    const horizons = [...new Set(rows.map((r) => r.Horizon).filter((h) => h !== null))].sort((a, b) => a - b);
    const pollutants = [...new Set(rows.map((r) => r.Pollutant))];
    const combos = new Set(rows.filter((r) => r.Predicted_Concentration !== null).map((r) => `${r.Pollutant}|${r.Horizon}`));
    summary.models = {
      pollutants: pollutants.length,
      horizons,
      forecasts_available: combos.size,
      expected: 42,
      complete: combos.size === 42,
    };
  }

  if (aq.ok) {
    const rows = aq.value.rows;
    summary.sources.aqi = { file: aq.value.file, rows: rows.length, updated: aq.value.updated };
    const byHorizon = rows.map((r) => ({
      horizon: r.Horizon,
      horizon_hours: r.Horizon_h,
      forecast_time: r.Forecast_Time,
      aqi: r.AQI,
      category: r.Category,
      dominant_pollutant: r.Dominant_Pollutant,
    }));
    const valid = byHorizon.filter((r) => r.aqi !== null);
    const peak = valid.length ? valid.reduce((a, b) => (b.aqi > a.aqi ? b : a)) : null;
    summary.forecast_aqi = {
      peak,
      by_horizon: byHorizon,
      basis: rows.find((r) => r.Basis)?.Basis || null,
    };
    const dominantCounts = countBy(rows, (r) => (r.Dominant_Pollutant && r.Dominant_Pollutant !== 'n/a' ? r.Dominant_Pollutant : null));
    const mostFrequent = Object.entries(dominantCounts).sort((a, b) => b[1] - a[1])[0];
    summary.dominant_pollutant = {
      at_peak: peak ? peak.dominant_pollutant : null,
      most_frequent: mostFrequent ? mostFrequent[0] : null,
      counts: dominantCounts,
    };
  }

  if (tr.ok) {
    const rows = tr.value.rows;
    summary.sources.trends = { file: tr.value.file, rows: rows.length, updated: tr.value.updated };
    summary.current_values = {};
    rows.forEach((r) => {
      summary.current_values[r.Pollutant] = { value: r.Current, unit: r.Unit || null };
    });
    const pm25 = rows.find((r) => r.Pollutant === 'PM2.5') || null;
    summary.overall_trend = {
      pollutant: pm25 ? pm25.Pollutant : null,
      trend: pm25 ? pm25.Trend : null,
      change_percentage: pm25 ? pm25.Change_Percentage : null,
      counts: countBy(rows, (r) => r.Trend),
    };
  }

  return summary;
}

/* ------------------------------------------------------------------ *
 * App                                                                 *
 * ------------------------------------------------------------------ */

const app = express();
app.disable('x-powered-by');

// Basic hardening headers. Chart.js is the only third-party script (CDN).
app.use((req, res, next) => {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline'; " +
      "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
  );
  next();
});

// API responses are always read fresh - never cached by the browser.
app.use('/api', (req, res, next) => {
  res.setHeader('Cache-Control', 'no-store');
  next();
});

function handle(fn) {
  return async (req, res) => {
    try {
      await fn(req, res);
    } catch (err) {
      if (err instanceof DataError) {
        res.status(err.status).json({ error: err.code, message: err.message, file: err.file });
      } else {
        console.error(err);
        res.status(500).json({ error: 'internal_error', message: 'Unexpected server error.' });
      }
    }
  };
}

app.get('/api/health', (req, res) => {
  const files = {};
  Object.values(SOURCES).forEach((s) => {
    files[s.file] = fs.existsSync(path.join(EVALUATION_DIR, s.file));
  });
  res.json({ status: 'ok', project: 'AeroCast', backend: 'Node.js + Express', data_files: files });
});

app.get('/api/forecast', handle(async (req, res) => {
  const { rows } = await dataProvider.getForecast();
  const station = typeof req.query.station === 'string' ? req.query.station : null;
  if (!station) return res.json(rows);
  // `station` is only compared against known IDs; it is never used to build a path.
  if (!uniqueStations(rows).some((s) => s.id === station)) {
    return res.status(404).json({ error: 'unknown_station', message: 'No forecast exists for that station.' });
  }
  return res.json(rows.filter((r) => r.Station_ID === station));
}));

app.get('/api/aqi', handle(async (req, res) => {
  res.json((await dataProvider.getAqi()).rows);
}));

app.get('/api/trends', handle(async (req, res) => {
  res.json((await dataProvider.getTrends()).rows);
}));

app.get('/api/summary', handle(async (req, res) => {
  res.json(await buildSummary());
}));

app.get('/api/stations', handle(async (req, res) => {
  res.json(uniqueStations((await dataProvider.getForecast()).rows));
}));

app.use('/api', (req, res) => {
  res.status(404).json({ error: 'not_found', message: 'Unknown API endpoint.' });
});

// Static frontend - the ONLY folder that is served (models/, data/, notebooks/ and evaluation/ are not).
app.use(express.static(FRONTEND_DIR, { dotfiles: 'ignore', index: 'index.html' }));

const server = app.listen(PORT, () => {
  console.log('\n  AeroCast dashboard');
  console.log(`  -> http://localhost:${PORT}\n`);
  console.log(`  Frontend   : ${FRONTEND_DIR}${fs.existsSync(FRONTEND_DIR) ? '' : '  (NOT FOUND)'}`);
  console.log(`  Evaluation : ${EVALUATION_DIR}`);
  Object.values(SOURCES).forEach((s) => {
    const found = fs.existsSync(path.join(EVALUATION_DIR, s.file));
    console.log(`    ${found ? '[found]  ' : '[MISSING]'} ${s.file}`);
  });
  console.log('');
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`Port ${PORT} is already in use. Stop the other process or start with a different port, e.g. PORT=3001 npm start`);
  } else {
    console.error(err);
  }
  process.exit(1);
});
