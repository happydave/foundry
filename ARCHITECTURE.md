## Version Context

- Go: 1.26.4 (declared in `/go.mod`)
- External dependency: `gopkg.in/yaml.v3 v3.0.1` (config parsing only)
- External runtime dependency: `llama-server` binary (llama.cpp); allowlisted version checked at startup

## Architectural Boundaries

```
Forbidden: /internal/ packages importing from /cmd/
Forbidden: /internal/server/ reading sysfs or /proc/meminfo directly
Forbidden: /internal/processmanager/ accessing GGUF metadata or model registry
Forbidden: /internal/estimator/ calling /internal/processmanager/ (in-use bytes are caller-supplied)
Required: all subprocess execution routes through /internal/processmanager/
Required: all VRAM/RAM queries route through /internal/estimator/QueryResources and QueryVRAMTotal
Required: all session history I/O routes through /internal/history/Store
Required: /internal/registry/ is populated once at startup and treated as read-only thereafter
```

## Component Index

- **main** `/cmd/foundry/`
  - Inputs: CLI flags (`-config`), SIGTERM/SIGINT
  - Outputs: constructed and wired service graph; calls `server.ListenAndServe`; calls `processmanager.UnloadAll` on shutdown

- **Config** `/internal/config/`
  - Inputs: YAML config file path (filesystem read)
  - Outputs: `*Config` struct with validated fields; default-filled `ListenAddress`, `LogLevel`, `KVCacheType` (q8_0), `Parallel` (1)

- **Registry** `/internal/registry/`
  - Inputs: filesystem scan of configured GGUF directories at startup; reads GGUF binary headers via `/internal/registry/gguf.go`
  - Outputs: `*Registry` (in-memory catalog of `Model` records indexed by stable uint64 ID); `List()`, `Get(id)`, `GetByName(name)`
  - Note: ID is an FNV-64a hash of absolute path + file size; mmproj files are associated with text models in the same directory
  - Note: for models with sliding-window (local) attention, `gguf.go` reads `attention.sliding_window` and derives per-block global vs. SWA layer/head counts onto `Model` via one of two paths. **Array path (Gemma 4):** an explicit per-layer `sliding_window_pattern` bool array plus a per-layer `head_count_kv` array (with `key_length_swa` for SWA head dim). **Period path (Gemma 3):** a positional interleave — a layer is global when `il % period == period-1`, where `period` comes from a scalar `sliding_window_pattern` or an architecture default (`gemma3` ⇒ 6, confirmed against llama.cpp); KV heads and head dim are uniform. If neither path applies, the fields are left zero and the model is treated as fully global attention (a warning is logged)

- **Estimator** `/internal/estimator/`
  - Inputs: `ModelSpec` (file size, layer count, KV head count, head dim, max context, plus sliding-window fields: window size, SWA head dim, and global/SWA layer and KV-head counts), context size, KV cache type, `nParallel` (number of parallel slots), in-use VRAM bytes; reads `/sys/class/drm/card*/device/mem_info_vram_{total,used}` and `/proc/meminfo` via `resources.go`
  - Outputs: `ForwardResult` (weight cost, KV cost, total cost, feasibility); `InverseResult` (maximum fitting context size)
  - Note: KV cache cost is multiplied by `nParallel`; must match the `--parallel` value passed to llama-server. AMD GPU only; errors out if no DRM sysfs entries are found
  - Note: when `SlidingWindowSize > 0`, KV cost splits into a context-scaling global-block term plus a fixed SWA-block term bounded by the window size; `Inverse` solves this in two phases (subtract the fixed SWA cost when the answer exceeds the window, else divide the budget by the combined per-token cost). When `SlidingWindowSize == 0` the formula reduces to the original all-layers form

- **ProcessManager** `/internal/processmanager/`
  - Inputs: model path, mmproj path, context size, GPU layers, `ModelLoadOptions` (KV cache type, parallel slot count, extra args)
  - Outputs: `*LoadedModel` (PID, port, context size, GPU layers, parallel slot count, load time, health status); manages `llama-server` subprocess lifecycle with SIGTERM/SIGKILL shutdown
  - Side-effects: spawns `llama-server` subprocess on a free loopback TCP port; polls `GET /health` until ready (up to 120 s); streams subprocess stdout/stderr into the structured logger; monitors for unexpected exits

- **History** `/internal/history/`
  - Inputs: session ID (alphanumeric + `_-`), role, content, timestamp
  - Outputs: `[]Turn` (chronological turn list); JSONL files written to `history_sessions_dir` (one file per session, named `<session-id>.jsonl`)
  - `history.go`: defines `Store` interface and `Turn` type
  - `jsonl.go`: `JSONLStore` — per-session mutex-serialised append; session ID validated against allowlist before any file operation

- **Server** `/internal/server/`
  - Inputs: HTTP requests on configured listen address; `*registry.Registry`, `*processmanager.Manager`, `*estimator.Estimator`, `history.Store`, resolved per-model load options
  - Outputs: HTTP responses; side-effect: proxies inference requests to llama-server subprocesses; side-effect: writes session turns to history store
  - `server.go`: HTTP mux wiring, management API handlers (`/api/v1/`), JSON response types
  - `lms.go`: LM Studio native API response types for `GET /api/v1/models`; `bitsPerWeight` quantization lookup table
  - `inference.go`: OpenAI and Anthropic inference proxy handlers (`/v1/`); `InferenceHook` extension point
  - `history_session.go`: session header parsing, history prepend, response capture, turn recording

## Dependency Chains

```
/cmd/foundry/ depends on all /internal/ packages
/internal/server/ depends on /internal/registry/, /internal/processmanager/, /internal/estimator/, /internal/history/
/internal/server/inference.go depends on /internal/processmanager/ (LoadedModel, HealthStatus)
/internal/server/history_session.go depends on /internal/history/ (Store, Turn, ErrInvalidSessionID)
/internal/config/, /internal/registry/, /internal/estimator/, /internal/processmanager/, /internal/history/ are leaf packages with no internal cross-imports
```

## Linear Data Flow

### Startup

```
1. main.go parses -config flag; calls config.Load() to read and validate foundry.yaml
2. processmanager.CheckBinaryVersion validates llama-server binary against knownLlamaServerVersions allowlist
3. registry.New() scans model_scan_paths; for each .gguf file, gguf.go parses binary header to extract architecture metadata
4. main.go constructs processmanager.Manager, estimator.Estimator, server.Server with resolved per-model options
5. If history_sessions_dir exists, history.NewJSONLStore attached to Server via SetHistoryStore
6. server.ListenAndServe binds TCP listener and begins serving; blocks until SIGTERM/SIGINT
7. On shutdown signal: server drains in-flight requests (30 s timeout); processmanager.UnloadAll sends SIGTERM to each subprocess
```

### Inference request (POST /v1/chat/completions or /v1/completions)

```
1. Request arrives at server.handleInferenceProxy; body read into buffer
2. model field extracted from JSON body
3. registry.GetByName resolves display name to Model record
4. processmanager.Get retrieves LoadedModel; health status checked
5. Optional InferenceHook applied; may rewrite r.Body
6. If X-Foundry-Persist: true — history.Store.Read loads prior turns; messages array rewritten with prior turns prepended
7. Request reverse-proxied (httputil.ReverseProxy) to http://127.0.0.1:<lm.Port>; SSE streaming passed through with FlushInterval=-1
8. If session active — capturingResponseWriter tees response; on success, user and assistant turns appended to history.Store
```

### Model load (POST /api/v1/models/{id}/load)

```
1. Request arrives at server.handleLoadModel; model ID parsed from path
2. registry.Get retrieves Model record
3. estimator.Inverse computes maximum fitting context (or estimator.Forward validates explicit ctx)
4. estimator queries /sys/class/drm/ for VRAM and /proc/meminfo for RAM
5. processmanager.Load acquires model-scoped lock; launches llama-server subprocess via exec.Command
6. processmanager polls GET http://127.0.0.1:<port>/health at 500 ms until 200 OK or 120 s timeout
7. LoadedModel record created; health monitor goroutine started (watches for unexpected exit)
8. LoadedModel returned to client as JSON
```

### Model unload (DELETE /api/v1/models/{id})

```
1. Request arrives at server.handleUnloadModel
2. processmanager.Unload transitions entry to kindUnloading; sends SIGTERM to subprocess
3. If subprocess does not exit within 10 s, escalates to SIGKILL
4. Entry removed from processmanager map; done channel closed
```
