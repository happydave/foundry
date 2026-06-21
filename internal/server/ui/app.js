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

async function refreshHardware() {
  let hw;
  try { hw = await fetchJSON("/hardware"); }
  catch (e) { panelError("hardware-body", e); return; }

  const table = el("table");
  const thead = el("tr");
  ["GPU", "Identity", "VRAM used / total", "Used"].forEach((h, i) => {
    const th = el("th", i === 3 ? "num" : "", h);
    thead.append(th);
  });
  table.append(thead);

  (hw.gpus || []).forEach((g) => {
    const tr = el("tr");
    tr.append(el("td", "", "card" + g.index));
    tr.append(el("td", "", g.identity));
    tr.append(el("td", "", `${fmtBytes(g.vram_used_bytes)} / ${fmtBytes(g.vram_total_bytes)}`));
    tr.append(barCell(g.vram_used_bytes, g.vram_total_bytes));
    table.append(tr);
  });

  const body = document.getElementById("hardware-body");
  body.replaceChildren(table);
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
  ["Model", "Context", "Health", "Est. VRAM", "Est. % of total"].forEach((h, i) => {
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
    tr.append(barCell(m.estimated_vram_bytes, totalVRAM));
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
