# Foundry

A local LLM inference service designed to run as a system daemon. Foundry manages one or more `llama-server` subprocesses — one per loaded model — and presents a unified inference API over them, supporting both the OpenAI and Anthropic wire formats.

Foundry is for users who need reliable, scriptable model serving integrated into automation pipelines. Its primary surface is the API; an optional, embedded operator console (see [Management UI](#management-ui)) is available for inspecting hardware and loaded models and for loading/unloading models by hand. Foundry is not a chat interface — interactive chat with models is the concern of the separate webChat project.

## How it works

Foundry scans configured directories for GGUF model files at startup, populates a registry, and exposes two API surfaces on a single HTTP port:

- **`/v1/`** — Inference endpoints. Supports the OpenAI wire format (`/v1/chat/completions`, `/v1/completions`) and the Anthropic wire format (`/v1/messages`). Any client that speaks either protocol can target Foundry without a translation shim.
- **`/api/v1/`** — Management API. Load and unload models, query resource usage, and inspect service state.

Each loaded model gets its own `llama-server` subprocess on a private loopback port. Foundry reverse-proxies inference requests to the right subprocess. A crashed subprocess marks that model unavailable but does not affect Foundry or other loaded models.

## Prerequisites

- Go 1.22 or later
- A `llama-server` binary (from [llama.cpp](https://github.com/ggerganov/llama.cpp)) at a version recognized by Foundry. Foundry validates the binary version at startup and refuses to start if the installed binary is not on the known-good allowlist. The allowlist is defined in `cmd/foundry/main.go` (`knownLlamaServerVersions`).
- One or more GGUF model files

## Build

```sh
go build -o foundry ./cmd/foundry
```

## Configuration

Foundry reads a YAML config file. The default path is `foundry.yaml` in the working directory; override with `-config`.

```yaml
# Directories to scan for .gguf model files (required)
model_scan_paths:
  - /models

# Path to the llama-server binary (required)
llama_server_binary: /usr/local/bin/llama-server

# Number of model layers to offload to GPU (required; use 0 for CPU-only)
default_gpu_layers: 99

# KV cache element type for all models: f16, bf16, f32, q8_0 (optional; default: q8_0)
kv_cache_type: q8_0

# Number of parallel inference slots (optional; default: 1).
# Controls llama-server's --parallel flag. 1 maximises context window size for single-user use.
# Increase only if you need concurrent multi-user inference.
parallel: 1

# Directory for persistent session history files (required).
# If the directory does not exist at startup, session history is disabled with a warning.
history_sessions_dir: /var/lib/foundry/sessions

# Listen address (default: 0.0.0.0:8080)
listen_address: 0.0.0.0:8080

# Log level: debug, info, warn, error (default: info)
log_level: info

# Enable the embedded operator console at /ui/ (optional; default: false).
# When false, /ui/ is not served. See the Management UI section below.
enable_ui: false

# Extra flags appended verbatim to every llama-server subprocess invocation (optional)
# Useful for backend selectors and site-specific flags not otherwise exposed by Foundry.
# llama_server_extra_args:
#   - --vulkan

# Per-model overrides, keyed by model DisplayName (GGUF filename without .gguf extension).
# Each entry is optional. Unknown fields are rejected at startup.
# models:
#   my-model-name:
#     # Override the global kv_cache_type for this model only (same values: f16, bf16, f32, q8_0).
#     # kv_cache_type: f16
#
#     # Override the global parallel for this model only (integer >= 1).
#     # parallel: 4
#
#     # Override the Jinja2 chat template embedded in the model's GGUF.
#     # Use chat_template_file to supply the template from a file (recommended for
#     # multi-line templates), or chat_template for a short inline string.
#     # These fields are mutually exclusive; specifying both is a startup error.
#     chat_template_file: /path/to/template.jinja
#     # chat_template: "{{ bos_token }}{% for message in messages %}..."
```

**Gemma-4 chat template:** Gemma-4 models bundled with an outdated template produce a
`common_chat_try_specialized_template` warning at startup. Suppress it by pointing to the
official template file:

```yaml
models:
  gemma-4-31B-it:
    chat_template_file: /home/dave/Documents/jinja/gemma-4-31B-it/chat_template.jinja
  gemma-4-26B-A4B-it:
    chat_template_file: /home/dave/Documents/jinja/gemma-4-26B-A4B-it/chat_template.jinja
```

Template args are passed to llama-server after the standard model flags
(`--model`, `--ctx-size`, etc.) and before `llama_server_extra_args`.

Run:

```sh
./foundry -config foundry.yaml
```

Foundry logs to stdout in JSON format. `llama-server` subprocess output is captured and re-emitted through the same log stream, tagged with the model ID.

## Model identifiers

Each model is identified by its **display name** — the GGUF filename without the `.gguf` extension. For example, `/models/llama-3.2-3b-instruct-q4_k_m.gguf` has display name `llama-3.2-3b-instruct-q4_k_m`.

Use the display name as the `model` field in inference requests. Management API endpoints (`/api/v1/models/{id}/load`, etc.) take a numeric model ID — an FNV-64a fingerprint of the file's absolute path and size. For currently loaded models, the numeric ID is available from `GET /api/v1/status`.

## OpenAI-compatible inference API

### POST /v1/chat/completions

Standard OpenAI chat completions. The `model` field must match a loaded model's display name. Streaming (`"stream": true`) is supported and passed through without buffering.

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Reasoning/thinking models** (e.g. Qwen3-thinking, DeepSeek-R1) spend a large number of tokens on an internal chain-of-thought before emitting a visible response. Set `max_tokens` high enough to cover both the thinking phase and the answer — values below ~500 will frequently truncate before any visible content is produced.

### Session history (opt-in)

Foundry can maintain server-side chat history for automation clients that fire one-shot requests and do not track conversation context themselves. History is disabled by default; clients opt in per request via two headers:

| Header | Description |
|---|---|
| `X-Foundry-Persist: true` | Enable history for this request. Any other value is ignored. |
| `X-Foundry-Session-Id: <id>` | Session identifier (characters: `[a-zA-Z0-9_-]`). If omitted, Foundry generates one. |

When `X-Foundry-Persist: true` is set and no session ID is provided, the generated ID is returned in the `X-Foundry-Session-Id` response header. Subsequent requests with the same session ID and persist flag will have prior turns prepended to the model context automatically.

Session files are stored in `history_sessions_dir` (one JSONL file per session, named `<session-id>.jsonl`). Files accumulate indefinitely — no expiry or rotation is implemented in this phase.

```sh
# First request: Foundry generates and returns a session ID
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Foundry-Persist: true" \
  -D - \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hi"}]}'
# Response header: X-Foundry-Session-Id: <generated-id>

# Subsequent request: prior turns are prepended automatically
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Foundry-Persist: true" \
  -H "X-Foundry-Session-Id: <generated-id>" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "What did I just say?"}]}'
```

### POST /v1/completions

Same as above for legacy completions format.

## Anthropic-compatible inference API

### POST /v1/messages

Anthropic Messages API. The `model` field must match a loaded model's display name. Both streaming and non-streaming responses are supported and passed through without buffering. Tool calling is supported via `llama-server`'s native Anthropic compatibility.

```sh
curl http://localhost:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct-q4_k_m",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**Error responses** use Anthropic error format (`type`, `error.type`, `error.message`):

| Condition | Status |
|---|---|
| Model not in registry or not loaded | 404 (`not_found_error`) |
| Model subprocess has crashed | 503 (`overloaded_error`) |
| Invalid or missing `model` field | 400 (`invalid_request_error`) |

Session history (`X-Foundry-Persist`) is not applied to `/v1/messages` requests.

### GET /v1/models

Returns the list of currently loaded, healthy models in OpenAI list format.

```sh
curl http://localhost:8080/v1/models
```

**Error responses** use OpenAI error format (`error.message`, `error.type`, `error.code`):

| Condition | Status |
|---|---|
| Model not in registry or not loaded | 404 |
| Model subprocess has crashed | 503 |
| Invalid or missing `model` field | 400 |
| `X-Foundry-Session-Id` contains disallowed characters | 400 |

## Management API

All endpoints are under `/api/v1/`.

### Service status

```
GET /api/v1/status
```

Returns service health, currently loaded models, and a summary of VRAM usage.
Each loaded-model entry includes `model_id` and `port` plus `display_name`,
`context_size`, `health`, and `estimated_vram_bytes` — the latter is the
estimator's predicted footprint for that model at its loaded context size (an
estimate, not a measurement).

### Hardware

```
GET /api/v1/hardware
```

Returns per-GPU hardware detail and system memory availability. Each entry in
`gpus` contains `index` (the DRM card number), `identity` (the driver-provided
product name when available, otherwise the PCI `vendor:device` pair), and
`vram_total_bytes` / `vram_used_bytes` / `vram_available_bytes`. The top-level
`system_ram_available_bytes` reports available system RAM. All figures are live
reads. Returns 500 if no AMD DRM sysfs entries are found.

```sh
curl http://localhost:8080/api/v1/hardware
```

### Model discovery

```
GET /api/v1/models
GET /api/v1/models/{id}
```

`GET /api/v1/models` returns all models found at startup in LM Studio-compatible format (`{ "models": [...] }`). Each entry contains `key` (equals `display_name`), `id` (the decimal model fingerprint as a string, for use in management calls), `type`, `architecture`, `size_bytes`, `context_length`, `quantization` (object with `name` and `bits_per_weight`), and `loaded_instances` (non-null array, non-empty when the model is running). This endpoint is compatible with LM Studio clients such as lmstudio-for-copilot. The `id` is a string because the fingerprint is a 64-bit value that would lose precision as a JSON number in some clients (e.g. browsers).

`GET /api/v1/models/{id}` takes a numeric model ID and returns detailed metadata plus a resource estimate at native max context and the estimated maximum loadable context given current VRAM availability.

### Model lifecycle

```
POST /api/v1/models/{id}/load
DELETE /api/v1/models/{id}
```

Load a model. The optional JSON body may specify `{"ctx": N}` to set a context window size; if omitted, Foundry uses the estimated maximum that fits in available VRAM. Returns the loaded model record including the context size selected.

Unload terminates the subprocess with SIGTERM, escalating to SIGKILL after 10 seconds.

### Resource estimation

```
GET /api/v1/models/{id}/estimate?ctx={n}
```

Returns the estimated memory cost for a given model at a given context size and whether it fits in currently available VRAM. Does not load the model.

## Management UI

Foundry ships an optional, embedded operator console — a static, zero-dependency
web page served by the daemon itself. It is **disabled by default**; enable it by
setting `enable_ui: true` in the config. When enabled, browse to:

```
http://<host>:<port>/ui/
```

The console has three panels:

- **Hardware** — each GPU's identity and VRAM used/total (with percentage), plus
  available system RAM. These are measured figures from `GET /api/v1/hardware`.
- **Loaded models** — display name, context size, health, and estimated VRAM for
  each loaded model, from `GET /api/v1/status`. VRAM here is an *estimate* and is
  labelled as such.
- **Models** — every discovered model with size and loaded state, with Load and
  Unload controls. Loading consults the resource estimate first and warns if the
  model will not fit or must use a reduced context.

The console is served same-origin and calls the existing management API; no extra
ports or build tooling are involved. It currently has **no authentication** — like
the rest of Foundry it assumes a trusted, network-local deployment. If that does
not hold for your network, bind Foundry to loopback or front it with an
authenticating proxy.

## Example workflow

```sh
# See what models are available
curl http://localhost:8080/api/v1/models

# Load a model (Foundry picks the largest context that fits)
curl -X POST http://localhost:8080/api/v1/models/12345678901234567/load

# Run inference
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hi"}]}'

# Unload when done
curl -X DELETE http://localhost:8080/api/v1/models/12345678901234567
```

## Development

```sh
go test ./...
go vet ./...
```

## Status

| Component | Status |
|---|---|
| Model registry (GGUF scan) | Implemented |
| Process manager (llama-server lifecycle) | Implemented |
| Resource estimator (VRAM estimation) | Implemented |
| Management API | Implemented |
| OpenAI-compatible inference proxy | Implemented |
| Anthropic-compatible inference proxy (`/v1/messages`) | Implemented |
| Session history store (JSONL backend) | Implemented |
| Hardware API (`/api/v1/hardware`) | Implemented |
| Management UI (operator console at `/ui/`) | Implemented |
