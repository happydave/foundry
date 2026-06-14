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

// Params configures estimation behavior. KVCacheType has been moved to a
// per-call parameter on Forward and Inverse; this struct is retained as an
// extension point for future global parameters.
type Params struct{}

// ModelSpec holds the model fields required for estimation.
type ModelSpec struct {
	FileSize    int64 // quantized GGUF file size in bytes (weight cost approximation)
	LayerCount  uint32
	KVHeadCount uint32
	HeadDim     uint32
	MaxContext  uint32 // native maximum context length from model metadata

	// Sliding-window (local) attention fields. All zero for models that use
	// fully global attention, in which case the KV formula reduces to the
	// all-layers form using LayerCount, KVHeadCount, and HeadDim. When
	// SlidingWindowSize > 0, the KV cache cost splits into a context-scaling
	// global term and a fixed sliding-window term.
	SlidingWindowSize uint32
	SWAHeadDim        uint32
	GlobalLayerCount  uint32
	SWALayerCount     uint32
	GlobalKVHeadCount uint32
	SWAKVHeadCount    uint32
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
// kvType is the KV cache element type (e.g. "f16", "q8_0"); unrecognised or
// empty values default to f16.
// nParallel is the number of parallel KV cache slots (must be >= 1).
//
// inUseBytes is the total VRAM currently consumed by loaded llama-server
// instances. The caller (Process Manager) is responsible for providing this
// value; the estimator does not query the Process Manager directly.
//
// If ctxLen exceeds the model's native MaxContext, it is clamped to
// MaxContext before estimation proceeds.
func (e *Estimator) Forward(model ModelSpec, ctxLen uint32, inUseBytes uint64, kvType string, nParallel int) (ForwardResult, error) {
	if ctxLen == 0 {
		return ForwardResult{}, errors.New("context size must be positive")
	}
	if model.KVHeadCount == 0 {
		return ForwardResult{}, errors.New("model KV head count must be positive")
	}
	if model.HeadDim == 0 {
		return ForwardResult{}, errors.New("model head dimension must be positive")
	}
	if nParallel < 1 {
		return ForwardResult{}, fmt.Errorf("nParallel must be >= 1, got %d", nParallel)
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
	kvCost := kvCacheBytes(model, ctxLen, kvBytesPerElement(kvType)) * uint64(nParallel)
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
// kvType is the KV cache element type (e.g. "f16", "q8_0"); unrecognised or
// empty values default to f16.
// nParallel is the number of parallel KV cache slots (must be >= 1).
//
// inUseBytes is the total VRAM currently consumed by loaded llama-server
// instances.
//
// Returns MaxContext=0 if the model's weights alone exceed the effective memory
// budget (model does not fit at any context size).
func (e *Estimator) Inverse(model ModelSpec, inUseBytes uint64, kvType string, nParallel int) (InverseResult, error) {
	if model.KVHeadCount == 0 {
		return InverseResult{}, errors.New("model KV head count must be positive")
	}
	if model.HeadDim == 0 {
		return InverseResult{}, errors.New("model head dimension must be positive")
	}
	if nParallel < 1 {
		return InverseResult{}, fmt.Errorf("nParallel must be >= 1, got %d", nParallel)
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
	bytesPerElem := kvBytesPerElement(kvType)

	var maxCtx uint64
	if model.SlidingWindowSize > 0 {
		mc, err := inverseSWAContext(model, kvBudget, bytesPerElem, nParallel)
		if err != nil {
			return InverseResult{}, err
		}
		maxCtx = mc
	} else {
		bytesPerCtx := kvCacheBytesPerToken(model, bytesPerElem) * uint64(nParallel)
		if bytesPerCtx == 0 {
			return InverseResult{}, errors.New("KV cache cost per token is zero; cannot compute inverse")
		}
		maxCtx = kvBudget / bytesPerCtx
	}

	if maxCtx == 0 {
		return InverseResult{MaxContext: 0}, nil
	}

	// Clamp to native max context.
	if model.MaxContext > 0 && maxCtx > uint64(model.MaxContext) {
		maxCtx = uint64(model.MaxContext)
	}

	return InverseResult{MaxContext: uint32(maxCtx)}, nil
}

// inverseSWAContext solves for the largest context size that fits the KV budget
// for a sliding-window attention model. It proceeds in two phases.
//
// Phase 1 assumes the answer exceeds the sliding window size (the common case for
// large-context models). There, the sliding-window blocks contribute a fixed
// cost and only the global blocks scale with context: subtract the fixed cost,
// then divide the remainder by the global per-token cost. If the result exceeds
// the window size the assumption held and the result is returned.
//
// Phase 2 handles the case where the budget is small enough that the answer is
// at or below the window size. There, both block types scale linearly with
// context, so divide the budget by the combined per-token cost and cap the
// result at the window size.
func inverseSWAContext(model ModelSpec, kvBudget uint64, bytesPerElem float64, nParallel int) (uint64, error) {
	np := uint64(nParallel)

	// Phase 1: answer assumed greater than the sliding window.
	swaFixed := swaFixedBytes(model, bytesPerElem) * np
	perTokenGlobal := kvCacheBytesPerToken(model, bytesPerElem) * np
	if perTokenGlobal > 0 && swaFixed < kvBudget {
		maxCtx := (kvBudget - swaFixed) / perTokenGlobal
		if maxCtx > uint64(model.SlidingWindowSize) {
			return maxCtx, nil
		}
	}

	// Phase 2: answer at or below the sliding window; both terms scale with ctx.
	combinedPerToken := kvCombinedBytesPerToken(model, bytesPerElem) * np
	if combinedPerToken == 0 {
		return 0, errors.New("KV cache cost per token is zero; cannot compute inverse")
	}
	maxCtx := kvBudget / combinedPerToken
	if maxCtx > uint64(model.SlidingWindowSize) {
		maxCtx = uint64(model.SlidingWindowSize)
	}
	return maxCtx, nil
}

// kvBytesPerElement returns bytes per KV cache element for the given type.
// q8_0 uses 34 bytes per 32 elements (block_q8_0: 2-byte delta + 32 int8 values).
// Unrecognised or empty values conservatively default to f16 (2 bytes/element).
func kvBytesPerElement(kvType string) float64 {
	switch strings.ToLower(kvType) {
	case "f32":
		return 4.0
	case "q8_0":
		return 34.0 / 32.0
	case "f16", "bf16", "":
		return 2.0
	default:
		return 2.0
	}
}

// kvCacheBytes computes the KV cache cost in bytes for a given context length.
//
// For fully global attention (SlidingWindowSize == 0) the formula is:
//
//	layers × kv_heads × head_dim × ctx_len × bytes_per_element × 2
//
// For sliding-window attention (SlidingWindowSize > 0) the cost is the sum of a
// context-scaling global term and a sliding-window term bounded by the window
// size:
//
//	global_layers × global_kv_heads × head_dim × ctx_len × 2
//	+ swa_layers × swa_kv_heads × swa_head_dim × min(ctx_len, window) × 2
//
// The factor of 2 accounts for both key and value caches.
func kvCacheBytes(model ModelSpec, ctxLen uint32, bytesPerElem float64) uint64 {
	if model.SlidingWindowSize == 0 {
		count := uint64(model.LayerCount) * uint64(model.KVHeadCount) * uint64(model.HeadDim) * uint64(ctxLen) * 2
		return uint64(float64(count) * bytesPerElem)
	}
	swaCtx := ctxLen
	if swaCtx > model.SlidingWindowSize {
		swaCtx = model.SlidingWindowSize
	}
	globalElems := uint64(model.GlobalLayerCount) * uint64(model.GlobalKVHeadCount) * uint64(model.HeadDim) * uint64(ctxLen) * 2
	swaElems := uint64(model.SWALayerCount) * uint64(model.SWAKVHeadCount) * uint64(model.SWAHeadDim) * uint64(swaCtx) * 2
	return uint64(float64(globalElems+swaElems) * bytesPerElem)
}

// kvCacheBytesPerToken returns the per-token KV cache cost in bytes — the
// marginal cost of one additional context token. This is used by Inverse to
// solve for the maximum context size.
//
// For sliding-window models, only the global blocks scale with context length;
// the sliding-window blocks contribute a constant once ctx_len exceeds the
// window. Inverse accounts for that constant separately. Thus this function
// returns only the global scaling term when SlidingWindowSize > 0.
func kvCacheBytesPerToken(model ModelSpec, bytesPerElem float64) uint64 {
	if model.SlidingWindowSize == 0 {
		count := uint64(model.LayerCount) * uint64(model.KVHeadCount) * uint64(model.HeadDim) * 2
		return uint64(float64(count) * bytesPerElem)
	}
	count := uint64(model.GlobalLayerCount) * uint64(model.GlobalKVHeadCount) * uint64(model.HeadDim) * 2
	return uint64(float64(count) * bytesPerElem)
}

// swaFixedBytes returns the fixed (context-independent) KV cache cost of the
// sliding-window blocks once context length reaches the window size. Zero for
// non-SWA models.
func swaFixedBytes(model ModelSpec, bytesPerElem float64) uint64 {
	if model.SlidingWindowSize == 0 {
		return 0
	}
	elems := uint64(model.SWALayerCount) * uint64(model.SWAKVHeadCount) * uint64(model.SWAHeadDim) * uint64(model.SlidingWindowSize) * 2
	return uint64(float64(elems) * bytesPerElem)
}

// kvCombinedBytesPerToken returns the per-token KV cost when both global and
// sliding-window blocks scale linearly with context — that is, when context
// length is at or below the window size. Used by Inverse's second phase.
func kvCombinedBytesPerToken(model ModelSpec, bytesPerElem float64) uint64 {
	count := (uint64(model.GlobalLayerCount)*uint64(model.GlobalKVHeadCount)*uint64(model.HeadDim) +
		uint64(model.SWALayerCount)*uint64(model.SWAKVHeadCount)*uint64(model.SWAHeadDim)) * 2
	return uint64(float64(count) * bytesPerElem)
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
