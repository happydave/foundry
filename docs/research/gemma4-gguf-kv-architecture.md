# Gemma 4 GGUF KV Cache Architecture

**Date**: 2026-06-13
**Model inspected**: `Gemma-4-Garnet-V2-31B-it-ultra-uncensored-heretic-Q4_K_M.gguf` (ai2)
**Purpose**: Understand the SWA (sliding-window attention) metadata in Gemma 4 GGUFs, to fix foundry's KV cache estimator (work item 377).

---

## Raw GGUF Metadata (relevant fields)

Extracted via `parseGGUF` / Python inspection of the binary (GGUF v3, 833 tensors, 44 KV pairs):

| Key | Type | Value |
|---|---|---|
| `gemma4.block_count` | uint32 | `60` |
| `gemma4.context_length` | uint32 | `262144` (256K native max) |
| `gemma4.attention.head_count` | uint32 | `32` |
| `gemma4.attention.head_count_kv` | uint32[] | `[16,16,16,16,16,4]` × 10 |
| `gemma4.attention.key_length` | uint32 | `512` (global blocks) |
| `gemma4.attention.value_length` | uint32 | `512` (global blocks) |
| `gemma4.attention.key_length_swa` | uint32 | `256` (SWA blocks) |
| `gemma4.attention.value_length_swa` | uint32 | `256` (SWA blocks) |
| `gemma4.attention.sliding_window` | uint32 | `1024` |
| `gemma4.attention.sliding_window_pattern` | bool[] | `[T,T,T,T,T,F]` × 10 |
| `gemma4.attention.shared_kv_layers` | uint32 | `0` |
| `general.file_type` | uint32 | `15` (Q4_K_M) |

---

## Attention Block Structure

The `sliding_window_pattern` bool array has one entry per block:
- `True` = sliding-window attention (SWA / local attention)
- `False` = global attention (full context)

Pattern `[T,T,T,T,T,F]` repeats 10 times across 60 blocks:

| Type | Count | KV heads | Head dim | Context |
|---|---|---|---|---|
| Global (False) | **10** | 4 | 512 | Full `ctx_len` |
| SWA (True) | **50** | 16 | 256 | `min(ctx_len, 1024)` |

The `head_count_kv` array is aligned with `sliding_window_pattern`: indices where the pattern is `False` (global blocks) have 4 KV heads; indices where the pattern is `True` (SWA blocks) have 16 KV heads.

Note that global and SWA blocks use different head dimensions: 512 vs 256. These are exposed as separate GGUF keys (`key_length` and `key_length_swa`).

---

## KV Cache Cost Formulae

### Current (incorrect) estimator formula

Uses `max(head_count_kv)` = 16 and `key_length` = 512 across all 60 blocks:

```
kvCost = 60 × 16 × 512 × ctx_len × 2 × bytesPerElem
       = 1,044,480 × ctx_len × bytesPerElem
```

For `ctx_len = 45272` (q8_0, bytesPerElem = 34/32 ≈ 1.0625):
`= 1,044,480 × 45,272 × 1.0625 ≈ 50.2 GB`

### Correct formula

**Global blocks** (scale with context):
```
globalCost = 10 × 4 × 512 × ctx_len × 2 × bytesPerElem
           = 43,520 × ctx_len × bytesPerElem
```

**SWA blocks** (fixed once ctx_len > sliding_window):
```
swaCost = 50 × 16 × 256 × min(ctx_len, 1024) × 2 × bytesPerElem
        = 409,600 × min(ctx_len, 1024) × bytesPerElem
```

For `ctx_len > 1024` (constant):
`swaCost = 409,600 × 1024 × 1.0625 ≈ 446 MB`

**Total** for `ctx_len = 262144` (q8_0):
```
globalCost = 43,520 × 262,144 × 1.0625 ≈ 12.11 GB
swaCost    ≈ 0.44 GB
total      ≈ 12.55 GB
```

### Overcounting factor

```
current bytes/token : 1,044,480
correct bytes/token :    43,520  (scaling term only)
ratio               :       ~24×
```

The current estimator overestimates KV cost by ~24×, causing `Inverse` to return ~45K context when the model's native max (262,144) fits comfortably within available memory.

---

## Impact on Estimator

On ai2, observed before fix (parallel=1):

| Metric | Value |
|---|---|
| Available RAM | ~80 GB |
| Conservative budget (÷1.15) | ~69.9 GB |
| Model file size | ~22.6 GB |
| KV budget | ~47.3 GB |
| Context loaded | **45,272** (estimator limit) |
| Context possible (correct formula) | **262,144** (native max) |

With the correct formula, the SWA fixed cost (≈446 MB) is subtracted from the KV budget first, and only the 43,520 bytes/token global scaling term is used to compute `maxCtx`. The result comfortably reaches the 262,144 native max.

---

## Keys Required for the Fix (work item 377)

| GGUF key (suffix after arch prefix) | Purpose |
|---|---|
| `attention.sliding_window` | SWA window size (uint32) |
| `attention.sliding_window_pattern` | Per-block type (bool array; True=SWA) |
| `attention.key_length_swa` | Head dim for SWA blocks (uint32) |
| `attention.head_count_kv` (array) | Already read; correlate with pattern to split global vs SWA head counts |

The `attention.value_length_swa` is also present and equal to `key_length_swa` (256). If key ≠ value dim in future models, both may need to be read; for now a single `SWAHeadDim` field suffices.

`shared_kv_layers = 0` means no cross-layer KV sharing is active; this field can be ignored for the initial fix.

---

## GGUF Parser Notes

The existing `applyMeta` / `ggufReader` infrastructure handles all required types:
- `uint32` scalars: handled by existing `toUint32` + suffix-match switch
- `bool` arrays (`ggufTypeArray` of `ggufTypeBool`): handled by existing `array()` reader; `value(ggufTypeBool)` returns `bool`
- Correlating `head_count_kv` array with `sliding_window_pattern` array requires a post-parse step (both arrays must be read before deriving global/SWA head counts), which can be a `derive()` method called after the KV loop in `parseGGUF`
