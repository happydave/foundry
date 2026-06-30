# Foundry Metrics — Research Notes

Reference for the upcoming Web UI work. Covers what data is available, where it comes from, and performance characteristics.

---

## Hardware Info (System-Level)

All readable directly from the OS, no external tools required.

### CPU
- `/proc/cpuinfo` — model name, core/thread count
- `/proc/stat` — per-core utilization (requires two reads with a diff interval)

### RAM
- `/proc/meminfo` — total, available, used, buffer/cache breakdown

### AMD GPU (sysfs, no rocm-smi required)
- `/sys/class/drm/cardN/device/mem_info_vram_total`
- `/sys/class/drm/cardN/device/mem_info_vram_used`
- `/sys/class/drm/cardN/device/gpu_busy_percent`
- `/sys/class/drm/cardN/device/hwmon/hwmonN/temp1_input` (millidegrees C)
- GPU name derivable from vendor/device PCI ID files in the same directory

### NVIDIA GPU
- `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader`
- Or via NVML Go bindings for in-process reads

---

## llama-server Metrics Endpoints

### `/metrics` (Prometheus format)
Requires `--metrics` flag at llama-server launch. Exposes atomic in-process counters:

| Metric | Description |
|---|---|
| `llamacpp:tokens_per_second` | Live generation speed gauge |
| `llamacpp:prompt_tokens_seconds` | Prefill latency histogram |
| `llamacpp:eval_tokens_seconds` | Decode latency histogram |
| `llamacpp:kv_cache_usage_ratio` | KV cache fill fraction (0–1) |
| `llamacpp:kv_cache_tokens` | Raw token count in KV cache |
| `llamacpp:queue_size` | Waiting requests |
| `llamacpp:requests_processing` | Active inference slots |
| `llamacpp:prompt_tokens_total` | Cumulative prompt tokens (differentiable for rate) |
| `llamacpp:eval_tokens_total` | Cumulative generation tokens (differentiable for rate) |

**Action required:** pass `--metrics` to each llama-server subprocess in the process manager.

### `/slots` (JSON)
Enabled by default (no flag needed). Returns array of slot states:

| Field | Description |
|---|---|
| `state` | 0 = idle, 1 = processing |
| `n_ctx` / `n_past` | Context capacity vs. tokens consumed (KV fill per slot) |
| `timings.predicted_per_second` | Tokens/sec for last completed generation |
| `timings.prompt_per_second` | Prefill speed for last request |
| `timings.prompt_n` / `predicted_n` | Token counts for last request |

---

## Performance Overhead

**sysfs / `/proc` reads:** virtual filesystem, no disk I/O. Microsecond cost. Safe to poll at 10 Hz.

**llama-server `/metrics` and `/slots`:** reads atomic in-process counters over loopback HTTP. A few ms per model per poll at most.

**Only real concern:** llama-server's HTTP handler is single-threaded. A metrics poll during active inference could add a few ms of latency to the poll response itself. At 1s polling this is negligible; avoid sub-100ms polling during heavy streaming workloads.

---

## Recommended Implementation Shape

1. **Background goroutine in Foundry** polls all sources (sysfs + each llama-server's `/metrics`) at ~1s, caches the merged result in memory.
2. **`GET /api/v1/metrics`** returns the cached snapshot — zero fan-out cost per web request.
3. **Optional SSE variant** (`GET /api/v1/metrics/stream`) for push-based chart updates.

---

## Suggested Web UI Widgets

| Widget | Source |
|---|---|
| Tokens/sec over time (line chart) | `llamacpp:tokens_per_second` or `/slots` `timings` |
| KV cache fill per model | `llamacpp:kv_cache_usage_ratio` or `/slots` `n_past/n_ctx` |
| VRAM used / total | AMD sysfs |
| GPU utilization % | `gpu_busy_percent` |
| GPU temperature | hwmon |
| RAM used / available | `/proc/meminfo` |
| Active requests / queue depth | `llamacpp:requests_processing` / `queue_size` |
| Prefill vs decode speed | `/slots` `timings` per completion |

---

## This Machine (ai2 / GTR reference)

- CPU: AMD Ryzen 9 7950X3D, 32 threads
- RAM: ~96 GB
- GPU (primary): NVIDIA GeForce RTX 5070, 12 GB VRAM — card1 in sysfs
- GPU (secondary/AMD): ~512 MB VRAM (integrated or small card) — card2 in sysfs
- `nvidia-smi` available; `rocm-smi` not installed
