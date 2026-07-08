"use strict";

// Foundry operator console. Vanilla JS, no dependencies. Talks to the same-origin
// management API under /api/v1/. Model ids are treated as opaque strings to
// preserve 64-bit precision.

const POLL_MS = 4000;
const API = "../api/v1"; // served from /ui/, so ../api/v1 -> /api/v1

function fmtBytes(n) {
  if (n === null || n === undefined) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0, v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${u[i]}`;
}

function pct(used, total) {
  if (!total) return 0;
  return Math.min(100, Math.max(0, (Number(used) / Number(total)) * 100));
}

function barCell(used, total, label) {
  const p = pct(used, total);
  const cls = p >= 90 ? "bad" : p >= 75 ? "warn" : "";
  const td = el("td");
  const bar = el("div", "bar " + cls);
  const span = el("span");
  span.style.width = p.toFixed(1) + "%";
  const lab = el("label");
  lab.textContent = label !== undefined ? label : `${p.toFixed(0)}%`;
  bar.append(span, lab);
  td.append(bar);
  return td;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

async function fetchJSON(path) {
  const r = await fetch(`${API}${path}`, { headers: { "Accept": "application/json" } });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { const b = await r.json(); if (b && b.error) msg = b.error; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}

function panelError(bodyId, err) {
  const body = document.getElementById(bodyId);
  body.replaceChildren(el("p", "error", `Failed to load: ${err.message}`));
}

function toast(msg, kind) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className = "toast " + (kind || "");
  setTimeout(() => { t.className = "toast hidden"; }, 5000);
}

// --- Hardware panel ---

// Stable, theme-fitting palette for per-model memory segments. A model keeps its
// colour across polls because the index is derived deterministically from its id.
const SEG_PALETTE = [
  "#5aa0ff", "#3fb950", "#d29922", "#bc8cff", "#f778ba",
  "#56d4dd", "#e3b341", "#7ee787", "#ff9e64", "#a5a5ff",
];

// colorForModel maps an opaque model id string to a stable palette colour.
function colorForModel(modelId) {
  const s = String(modelId);
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return SEG_PALETTE[h % SEG_PALETTE.length];
}

// stackedBar renders a proportional memory bar. segments is an array of
// { bytes, color, title }; the track background shows the free remainder when the
// segments sum to less than total.
function stackedBar(segments, total) {
  const track = el("div", "stack");
  segments.forEach((s) => {
    if (!s.bytes) return;
    const seg = el("div", "seg");
    seg.style.width = pct(s.bytes, total).toFixed(2) + "%";
    seg.style.background = s.color;
    if (s.title) seg.title = s.title; // title is set via property -> safe text
    track.append(seg);
  });
  return track;
}

// telemetryChips builds the present-only telemetry readouts for a card.
function telemetryChips(t) {
  const wrap = el("div", "chips");
  if (!t) return wrap;
  if (t.busy_percent !== undefined && t.busy_percent !== null) {
    wrap.append(el("span", "chip", `${Number(t.busy_percent)}% busy`));
  }
  if (t.temperature_millicelsius !== undefined && t.temperature_millicelsius !== null) {
    wrap.append(el("span", "chip", `${(Number(t.temperature_millicelsius) / 1000).toFixed(0)}°C`));
  }
  if (t.power_microwatts !== undefined && t.power_microwatts !== null) {
    wrap.append(el("span", "chip", `${(Number(t.power_microwatts) / 1e6).toFixed(1)} W`));
  }
  if (t.clock_mhz !== undefined && t.clock_mhz !== null) {
    wrap.append(el("span", "chip", `${Number(t.clock_mhz)} MHz`));
  }
  return wrap;
}

// legendRow builds one legend entry: colour swatch, label (text), and value.
function legendRow(color, label, value) {
  const row = el("div", "legend-row");
  const sw = el("span", "swatch");
  if (color) sw.style.background = color; else sw.classList.add("free");
  const name = el("span", "legend-name", label); // textContent -> safe
  name.title = label; // full name on hover when the label is truncated (property -> safe)
  const val = el("span", "legend-val", value);
  row.append(sw, name, val);
  return row;
}

function gpuCard(g) {
  const card = el("div", "gpu-card");

  const head = el("div", "gpu-head");
  head.append(el("span", "gpu-name", `card${g.index} · ${g.identity}`));
  head.append(telemetryChips(g.telemetry));
  card.append(head);

  const total = Number(g.vram_total_bytes);
  const procs = g.processes || [];
  const unattributed = Number(g.unattributed_vram_bytes || 0);

  // VRAM header line + stacked bar.
  card.append(el("div", "meter-label",
    `VRAM ${fmtBytes(g.vram_used_bytes)} / ${fmtBytes(g.vram_total_bytes)}`));
  const segments = procs.map((p) => ({
    bytes: Number(p.vram_bytes),
    color: colorForModel(p.model_id),
    title: `${p.display_name || ("#" + p.model_id)} — ${fmtBytes(p.vram_bytes)}`,
  }));
  segments.push({ bytes: unattributed, color: "var(--muted)", title: `other — ${fmtBytes(unattributed)}` });
  card.append(stackedBar(segments, total));

  // Legend: one row per model, then other + free.
  const legend = el("div", "legend");
  procs.forEach((p) => {
    legend.append(legendRow(colorForModel(p.model_id),
      p.display_name || `#${p.model_id}`, fmtBytes(p.vram_bytes)));
  });
  legend.append(legendRow("var(--muted)", "other (unattributed)", fmtBytes(unattributed)));
  const free = total > Number(g.vram_used_bytes) ? total - Number(g.vram_used_bytes) : 0;
  legend.append(legendRow(null, "free", fmtBytes(free)));
  card.append(legend);

  // GTT pool line (APU-relevant). Only shown when the pool is reported.
  if (g.pools && Number(g.pools.gtt_total_bytes) > 0) {
    card.append(el("div", "meter-label",
      `GTT ${fmtBytes(g.pools.gtt_used_bytes)} / ${fmtBytes(g.pools.gtt_total_bytes)}`));
    card.append(stackedBar(
      [{ bytes: Number(g.pools.gtt_used_bytes), color: "var(--accent)" }],
      Number(g.pools.gtt_total_bytes)));
  }

  return card;
}

async function refreshHardware() {
  let hw;
  try { hw = await fetchJSON("/hardware"); }
  catch (e) { panelError("hardware-body", e); return; }

  const wrap = el("div", "gpu-list");
  (hw.gpus || []).forEach((g) => wrap.append(gpuCard(g)));

  const body = document.getElementById("hardware-body");
  body.replaceChildren(wrap);
  body.append(el("p", "muted", `System RAM available: ${fmtBytes(hw.system_ram_available_bytes)}`));
}

// --- Loaded models panel ---

let totalVRAM = 0;

async function refreshStatus() {
  let st;
  try { st = await fetchJSON("/status"); }
  catch (e) { panelError("loaded-body", e); return; }

  totalVRAM = Number(st.memory ? st.memory.total_vram_bytes : 0);
  const body = document.getElementById("loaded-body");
  const models = st.loaded_models || [];
  if (models.length === 0) {
    body.replaceChildren(el("p", "muted", "No models loaded."));
    return;
  }

  const table = el("table");
  const thead = el("tr");
  ["Model", "Context", "Health", "Est. VRAM", "Meas. VRAM", "Meas. % of total"].forEach((h, i) => {
    thead.append(el("th", i >= 3 ? "num" : "", h));
  });
  table.append(thead);

  models.forEach((m) => {
    const tr = el("tr");
    tr.append(el("td", "", m.display_name || `#${m.model_id}`));
    tr.append(el("td", "num", m.context_size ? m.context_size.toLocaleString() : "—"));
    const h = el("td");
    h.append(el("span", "health " + (m.health || ""), m.health || "—"));
    tr.append(h);
    tr.append(el("td", "num", fmtBytes(m.estimated_vram_bytes)));
    // measured_vram_bytes is 0 when unavailable; show a dash rather than "0 B".
    const measured = Number(m.measured_vram_bytes || 0);
    tr.append(el("td", "num", measured > 0 ? fmtBytes(measured) : "—"));
    tr.append(barCell(measured, totalVRAM));
    table.append(tr);
  });
  body.replaceChildren(table);
}

// --- Models panel (load / unload) ---

async function refreshModels() {
  let resp;
  try { resp = await fetchJSON("/models"); }
  catch (e) { panelError("models-body", e); return; }

  const models = (resp.models || []).slice().sort((a, b) => a.key.localeCompare(b.key));
  const table = el("table");
  const thead = el("tr");
  ["Model", "Size", "State", ""].forEach((h, i) => {
    thead.append(el("th", i === 1 ? "num" : "", h));
  });
  table.append(thead);

  models.forEach((m) => {
    const loaded = (m.loaded_instances || []).length > 0;
    const tr = el("tr");
    tr.append(el("td", "", m.key));
    tr.append(el("td", "num", fmtBytes(m.size_bytes)));
    tr.append(el("td", "", loaded ? "loaded" : "—"));

    const actions = el("td");
    if (loaded) {
      const b = el("button", "danger", "Unload");
      b.onclick = () => unloadModel(m.id, m.key, b);
      actions.append(b);
    } else {
      const b = el("button", "", "Load");
      b.onclick = () => loadModel(m.id, m.key, b);
      actions.append(b);
    }
    tr.append(actions);
    table.append(tr);
  });

  document.getElementById("models-body").replaceChildren(table);
}

async function loadModel(id, name, btn) {
  btn.disabled = true;
  // Pre-flight feasibility check so we can warn before attempting.
  try {
    const d = await fetchJSON(`/models/${id}`);
    const feasible = d.native_estimate && d.native_estimate.feasible;
    const maxCtx = d.max_loadable_context || 0;
    let prompt = `Load "${name}"?`;
    if (!feasible && maxCtx === 0) {
      prompt = `"${name}" does not appear to fit in available VRAM at any context size. Attempt anyway?`;
    } else if (!feasible) {
      prompt = `"${name}" will not fit at native max context; Foundry will load it at up to ${maxCtx.toLocaleString()} tokens. Proceed?`;
    } else {
      prompt = `Load "${name}" (up to ${maxCtx.toLocaleString()} tokens)?`;
    }
    if (!confirm(prompt)) { btn.disabled = false; return; }
  } catch (e) {
    if (!confirm(`Could not estimate "${name}" (${e.message}). Attempt load anyway?`)) {
      btn.disabled = false; return;
    }
  }

  try {
    const r = await fetch(`${API}/models/${id}/load`, { method: "POST" });
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { const b = await r.json(); if (b && b.error) msg = b.error; } catch (_) {}
      throw new Error(msg);
    }
    const lm = await r.json();
    toast(`Loaded "${name}" at ${Number(lm.context_size).toLocaleString()} tokens.`, "ok");
  } catch (e) {
    toast(`Load failed: ${e.message}`, "bad");
  } finally {
    btn.disabled = false;
    refreshAll();
  }
}

async function unloadModel(id, name, btn) {
  if (!confirm(`Unload "${name}"?`)) return;
  btn.disabled = true;
  try {
    const r = await fetch(`${API}/models/${id}`, { method: "DELETE" });
    if (!r.ok && r.status !== 204) {
      let msg = `HTTP ${r.status}`;
      try { const b = await r.json(); if (b && b.error) msg = b.error; } catch (_) {}
      throw new Error(msg);
    }
    toast(`Unloaded "${name}".`, "ok");
  } catch (e) {
    toast(`Unload failed: ${e.message}`, "bad");
  } finally {
    btn.disabled = false;
    refreshAll();
  }
}

// --- Polling ---

function setConn(ok) {
  const c = document.getElementById("conn");
  c.textContent = ok ? "live" : "disconnected";
  c.className = "pill " + (ok ? "live" : "stale");
}

async function refreshLive() {
  // Hardware + status poll frequently; models list changes only on load/unload.
  await Promise.allSettled([refreshHardware(), refreshStatus()]);
}

async function refreshAll() {
  await Promise.allSettled([refreshHardware(), refreshStatus(), refreshModels()]);
}

async function tick() {
  try {
    await refreshLive();
    setConn(true);
  } catch (_) {
    setConn(false);
  }
}

refreshAll().then(() => setConn(true)).catch(() => setConn(false));
setInterval(tick, POLL_MS);
