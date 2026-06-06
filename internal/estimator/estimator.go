package estimator

import (
	"errors"
	"fmt"
	"strings"
)

// headroomFactor is the conservative buffer applied to cost estimates before
// comparing against available memory. A factor of 1.15 means the reported
// maximum context will fit in ~87% of available budget, leaving a 15% margin
// to absorb allocator overhead and measurement imprecision. This reduces the
// likelihood of OOM at actual load time.
const headroomFactor = 1.15

// Params configures estimation behavior.
type Params struct {
	// KVCacheType is the element type for the KV cache.
	// Supported values: "f16" (default), "bf16", "f32", "q8_0".
	// An empty string or unrecognised value defaults to f16 (2 bytes/element).
	KVCacheType string
}

// ModelSpec holds the model fields required for estimation.
type ModelSpec struct {
	FileSize    int64 // quantized GGUF file size in bytes (weight cost approximation)
	LayerCount  uint32
	KVHeadCount uint32
	HeadDim     uint32
	MaxContext  uint32 // native maximum context length from model metadata
}

// ForwardResult is the output of a forward estimate.
type ForwardResult struct {
	WeightCost uint64 // bytes
	KVCost     uint64 // bytes
	TotalCost  uint64 // bytes (WeightCost + KVCost)
	Feasible   bool   // true if (TotalCost × headroomFactor) ≤ available memory
}

// InverseResult is the output of an inverse estimate.
type InverseResult struct {
	// MaxContext is the largest feasible context size in tokens.
	// Zero means the model's weights alone exceed available memory.
	MaxContext uint32
}

// Estimator computes memory estimates for loading models.
type Estimator struct {
	params       Params
	queryResFunc func() (vramAvail, ramAvail uint64, err error)
}

// New constructs an Estimator with the given parameters.
func New(params Params) *Estimator {
	return &Estimator{
		params:       params,
		queryResFunc: QueryResources,
	}
}

// Forward estimates the memory cost for loading model at ctxLen tokens and
// returns a feasibility verdict against currently available resources.
//
// inUseBytes is the total VRAM currently consumed by loaded llama-server
// instances. The caller (Process Manager) is responsible for providing this
// value; the estimator does not query the Process Manager directly.
//
// If ctxLen exceeds the model's native MaxContext, it is clamped to
// MaxContext before estimation proceeds.
func (e *Estimator) Forward(model ModelSpec, ctxLen uint32, inUseBytes uint64) (ForwardResult, error) {
	if ctxLen == 0 {
		return ForwardResult{}, errors.New("context size must be positive")
	}
	if model.KVHeadCount == 0 {
		return ForwardResult{}, errors.New("model KV head count must be positive")
	}
	if model.HeadDim == 0 {
		return ForwardResult{}, errors.New("model head dimension must be positive")
	}

	// Clamp to native max context.
	if model.MaxContext > 0 && ctxLen > model.MaxContext {
		ctxLen = model.MaxContext
	}

	vramAvail, ramAvail, err := e.queryResFunc()
	if err != nil {
		return ForwardResult{}, fmt.Errorf("querying system resources: %w", err)
	}

	weightCost := uint64(model.FileSize)
	kvCost := kvCacheBytes(model, ctxLen, e.kvBytesPerElement())
	totalCost := weightCost + kvCost

	budget := effectiveBudget(vramAvail, ramAvail, inUseBytes)
	feasible := float64(totalCost)*headroomFactor <= float64(budget)

	return ForwardResult{
		WeightCost: weightCost,
		KVCost:     kvCost,
		TotalCost:  totalCost,
		Feasible:   feasible,
	}, nil
}

// Inverse computes the largest context size that fits in available resources
// for the given model.
//
// inUseBytes is the total VRAM currently consumed by loaded llama-server
// instances.
//
// Returns MaxContext=0 if the model's weights alone exceed the effective memory
// budget (model does not fit at any context size).
func (e *Estimator) Inverse(model ModelSpec, inUseBytes uint64) (InverseResult, error) {
	if model.KVHeadCount == 0 {
		return InverseResult{}, errors.New("model KV head count must be positive")
	}
	if model.HeadDim == 0 {
		return InverseResult{}, errors.New("model head dimension must be positive")
	}

	vramAvail, ramAvail, err := e.queryResFunc()
	if err != nil {
		return InverseResult{}, fmt.Errorf("querying system resources: %w", err)
	}

	budget := effectiveBudget(vramAvail, ramAvail, inUseBytes)
	// Shrink budget by headroom factor to compute the conservative spending limit.
	conservativeBudget := uint64(float64(budget) / headroomFactor)

	weightCost := uint64(model.FileSize)
	if weightCost >= conservativeBudget {
		return InverseResult{MaxContext: 0}, nil
	}

	kvBudget := conservativeBudget - weightCost
	bytesPerCtx := kvCacheBytesPerToken(model, e.kvBytesPerElement())
	if bytesPerCtx == 0 {
		return InverseResult{}, errors.New("KV cache cost per token is zero; cannot compute inverse")
	}

	maxCtx := kvBudget / bytesPerCtx
	if maxCtx == 0 {
		return InverseResult{MaxContext: 0}, nil
	}

	// Clamp to native max context.
	if model.MaxContext > 0 && maxCtx > uint64(model.MaxContext) {
		maxCtx = uint64(model.MaxContext)
	}

	return InverseResult{MaxContext: uint32(maxCtx)}, nil
}

// kvBytesPerElement returns bytes per KV cache element for the configured type.
func (e *Estimator) kvBytesPerElement() uint64 {
	switch strings.ToLower(e.params.KVCacheType) {
	case "f32":
		return 4
	case "q8_0":
		return 1
	case "f16", "bf16", "":
		return 2
	default:
		// Unknown type: conservatively default to f16 (2 bytes).
		return 2
	}
}

// kvCacheBytes computes the KV cache cost in bytes for a given context length.
// Formula: layers × kv_heads × head_dim × ctx_len × bytes_per_element × 2
// The factor of 2 accounts for both key and value caches.
func kvCacheBytes(model ModelSpec, ctxLen uint32, bytesPerElem uint64) uint64 {
	return uint64(model.LayerCount) * uint64(model.KVHeadCount) * uint64(model.HeadDim) * uint64(ctxLen) * bytesPerElem * 2
}

// kvCacheBytesPerToken returns the KV cache cost in bytes for a single token.
// This is used by Inverse to solve for the maximum context size.
func kvCacheBytesPerToken(model ModelSpec, bytesPerElem uint64) uint64 {
	return uint64(model.LayerCount) * uint64(model.KVHeadCount) * uint64(model.HeadDim) * bytesPerElem * 2
}

// effectiveBudget computes the total available memory budget after subtracting
// VRAM already in use by loaded models.
func effectiveBudget(vramAvail, ramAvail, inUseBytes uint64) uint64 {
	adjustedVRAM := uint64(0)
	if vramAvail > inUseBytes {
		adjustedVRAM = vramAvail - inUseBytes
	}
	return adjustedVRAM + ramAvail
}
