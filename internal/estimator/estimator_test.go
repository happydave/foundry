package estimator

import (
	"errors"
	"testing"
)

// testEstimator returns an Estimator with an injected resource querier so tests
// do not depend on actual GPU hardware.
func testEstimator(vramAvail, ramAvail uint64) *Estimator {
	e := New(Params{})
	e.queryResFunc = func() (uint64, uint64, error) {
		return vramAvail, ramAvail, nil
	}
	return e
}

func errEstimator(queryErr error) *Estimator {
	e := New(Params{})
	e.queryResFunc = func() (uint64, uint64, error) {
		return 0, 0, queryErr
	}
	return e
}

// --- model helpers ---

func llamaModel() ModelSpec {
	return ModelSpec{
		FileSize:    1 << 30, // 1 GiB
		LayerCount:  32,
		KVHeadCount: 8,
		HeadDim:     128,
		MaxContext:  4096,
	}
}

// gemma4Model returns a ModelSpec for the Gemma 4 31B model with sliding-window
// attention, using the confirmed GGUF-derived values: 60 blocks split into 10
// global blocks (4 KV heads, 512 head dim) and 50 SWA blocks (16 KV heads, 256
// head dim) with a 1024-token sliding window and 262144 native context.
func gemma4Model() ModelSpec {
	return ModelSpec{
		FileSize:          20 << 30, // ~20 GiB representative weight size
		LayerCount:        60,       // not used by estimator when SlidingWindowSize > 0
		KVHeadCount:       16,       // max of per-layer array; unused on SWA path
		HeadDim:           512,      // global block head dim (key_length)
		MaxContext:        262144,
		SlidingWindowSize: 1024,
		SWAHeadDim:        256,
		GlobalLayerCount:  10,
		SWALayerCount:     50,
		GlobalKVHeadCount: 4,
		SWAKVHeadCount:    16,
	}
}

// gemma3Model returns a ModelSpec for the Gemma 3 27B model with sliding-window
// attention derived positionally (period 6). Unlike Gemma 4, KV heads and head
// dim are uniform across global and SWA blocks: 62 blocks split into 10 global
// and 52 SWA, all with 16 KV heads and 128 head dim, window 1024, native 131072.
func gemma3Model() ModelSpec {
	return ModelSpec{
		FileSize:          16 << 30, // ~16 GiB representative weight size (27B Q4_K_S)
		LayerCount:        62,
		KVHeadCount:       16,
		HeadDim:           128,
		MaxContext:        131072,
		SlidingWindowSize: 1024,
		SWAHeadDim:        128,
		GlobalLayerCount:  10,
		SWALayerCount:     52,
		GlobalKVHeadCount: 16,
		SWAKVHeadCount:    16,
	}
}

// --- Forward tests ---

func TestForward_Formula(t *testing.T) {
	model := llamaModel()
	// KV cost = 32 * 8 * 128 * 4096 * 2 * 2 = 536870912 bytes (512 MiB) for f16, nParallel=1
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 2
	wantWeight := uint64(1 << 30)
	wantTotal := wantWeight + wantKV

	// Give plenty of memory so feasibility is true.
	e := testEstimator(16<<30, 16<<30)
	r, err := e.Forward(model, 4096, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.WeightCost != wantWeight {
		t.Errorf("WeightCost = %d, want %d", r.WeightCost, wantWeight)
	}
	if r.KVCost != wantKV {
		t.Errorf("KVCost = %d, want %d", r.KVCost, wantKV)
	}
	if r.TotalCost != wantTotal {
		t.Errorf("TotalCost = %d, want %d", r.TotalCost, wantTotal)
	}
	if !r.Feasible {
		t.Error("expected Feasible=true with ample memory")
	}
}

func TestForward_Feasible_BoundaryFalse(t *testing.T) {
	model := llamaModel()
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 2
	wantWeight := uint64(1 << 30)
	totalCost := wantWeight + wantKV

	// Set budget just below what headroom requires: cost * 1.15 > budget → not feasible.
	// budget = totalCost * 1.14 → cost * 1.15 > budget
	budget := uint64(float64(totalCost) * 1.14)

	e := testEstimator(budget, 0)
	r, err := e.Forward(model, 4096, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.Feasible {
		t.Errorf("expected Feasible=false when budget < totalCost * headroomFactor")
	}
}

func TestForward_Feasible_BoundaryTrue(t *testing.T) {
	model := llamaModel()
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 2
	wantWeight := uint64(1 << 30)
	totalCost := wantWeight + wantKV

	// budget = totalCost * 1.16 > totalCost * headroomFactor (1.15) → feasible.
	// Add a clear margin above headroomFactor to avoid floating-point truncation issues.
	budget := uint64(float64(totalCost) * 1.16)

	e := testEstimator(budget, 0)
	r, err := e.Forward(model, 4096, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !r.Feasible {
		t.Errorf("expected Feasible=true when budget > totalCost * headroomFactor")
	}
}

func TestForward_InUseBytes_ReducesVRAM(t *testing.T) {
	model := llamaModel()
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 2
	totalCost := uint64(1<<30) + wantKV

	// With 20 GiB VRAM and 5 GiB in use, effective VRAM = 15 GiB — still plenty.
	e := testEstimator(20<<30, 0)
	r, err := e.Forward(model, 4096, 5<<30, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.TotalCost != totalCost {
		t.Errorf("TotalCost = %d, want %d", r.TotalCost, totalCost)
	}
	if !r.Feasible {
		t.Error("expected Feasible=true")
	}
}

func TestForward_InUseBytes_ExceedsVRAM_NoNegative(t *testing.T) {
	model := llamaModel()
	// VRAM available = 1 GiB but 2 GiB in use → adjusted VRAM = 0, budget = RAM only.
	e := testEstimator(1<<30, 0)
	r, err := e.Forward(model, 4096, 2<<30, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// budget = 0, totalCost > 0, so not feasible
	if r.Feasible {
		t.Error("expected Feasible=false when VRAM is fully consumed")
	}
}

func TestForward_CtxClampedToMaxContext(t *testing.T) {
	model := llamaModel() // MaxContext = 4096
	// Request 8192 — should be clamped to 4096.
	kvAt4096 := uint64(32) * 8 * 128 * 4096 * 2 * 2

	e := testEstimator(32<<30, 32<<30)
	r, err := e.Forward(model, 8192, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != kvAt4096 {
		t.Errorf("KVCost = %d, want %d (context clamped to MaxContext=4096)", r.KVCost, kvAt4096)
	}
}

func TestForward_KVType_Q8(t *testing.T) {
	model := llamaModel()
	// q8_0 = 34/32 bytes/elem (block_q8_0: 2-byte delta + 32 int8 values per block).
	// count = 32 * 8 * 128 * 4096 * 2 = 268435456
	// cost  = 268435456 * 34 / 32 = 285212672
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 34 / 32

	e := testEstimator(32<<30, 32<<30)
	r, err := e.Forward(model, 4096, 0, "q8_0", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != wantKV {
		t.Errorf("KVCost (q8_0) = %d, want %d", r.KVCost, wantKV)
	}
}

func TestForward_KVType_F32(t *testing.T) {
	model := llamaModel()
	// f32 = 4 bytes/elem, double f16
	kvF16 := uint64(32) * 8 * 128 * 4096 * 2 * 2
	kvF32 := kvF16 * 2

	e := testEstimator(32<<30, 32<<30)
	r, err := e.Forward(model, 4096, 0, "f32", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != kvF32 {
		t.Errorf("KVCost (f32) = %d, want %d", r.KVCost, kvF32)
	}
}

func TestForward_ZeroCtx_Error(t *testing.T) {
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Forward(llamaModel(), 0, 0, "f16", 1)
	if err == nil {
		t.Error("expected error for zero context size")
	}
}

func TestForward_ZeroKVHeadCount_Error(t *testing.T) {
	model := llamaModel()
	model.KVHeadCount = 0
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Forward(model, 4096, 0, "f16", 1)
	if err == nil {
		t.Error("expected error for zero KV head count")
	}
}

func TestForward_ZeroHeadDim_Error(t *testing.T) {
	model := llamaModel()
	model.HeadDim = 0
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Forward(model, 4096, 0, "f16", 1)
	if err == nil {
		t.Error("expected error for zero head dimension")
	}
}

func TestForward_ResourceQueryError(t *testing.T) {
	e := errEstimator(errors.New("vulkan unavailable"))
	_, err := e.Forward(llamaModel(), 4096, 0, "f16", 1)
	if err == nil {
		t.Error("expected error when resource query fails")
	}
}

// --- Inverse tests ---

func TestInverse_BasicResult(t *testing.T) {
	model := llamaModel() // MaxContext = 4096

	// Give enough memory to load at a non-trivial context.
	e := testEstimator(16<<30, 16<<30)
	r, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.MaxContext == 0 {
		t.Fatal("expected MaxContext > 0 with ample memory")
	}
	if r.MaxContext > model.MaxContext {
		t.Errorf("MaxContext = %d exceeds native max %d", r.MaxContext, model.MaxContext)
	}
}

func TestInverse_ClampedToNativeMax(t *testing.T) {
	model := llamaModel() // MaxContext = 4096

	// Enormous memory — inverse should be clamped to 4096, not higher.
	e := testEstimator(512<<30, 512<<30)
	r, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.MaxContext != model.MaxContext {
		t.Errorf("MaxContext = %d, want %d (clamped to native max)", r.MaxContext, model.MaxContext)
	}
}

func TestInverse_ModelDoesNotFit(t *testing.T) {
	model := llamaModel()
	model.FileSize = 8 << 30 // 8 GiB weights

	// Only 4 GiB available — model does not fit.
	e := testEstimator(4<<30, 0)
	r, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.MaxContext != 0 {
		t.Errorf("MaxContext = %d, want 0 (model does not fit)", r.MaxContext)
	}
}

func TestInverse_ZeroMemoryAfterInUse(t *testing.T) {
	model := llamaModel()
	// All VRAM consumed, no RAM.
	e := testEstimator(4<<30, 0)
	r, err := e.Inverse(model, 4<<30, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.MaxContext != 0 {
		t.Errorf("MaxContext = %d, want 0 (no memory available)", r.MaxContext)
	}
}

func TestInverse_InverseIsConsistentWithForward(t *testing.T) {
	model := llamaModel()
	vram := uint64(12 << 30)
	ram := uint64(4 << 30)

	e := testEstimator(vram, ram)
	inv, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Inverse error: %v", err)
	}
	if inv.MaxContext == 0 {
		t.Skip("no context fits; nothing to verify")
	}

	// Forward at the reported max context must be feasible.
	fwd, err := e.Forward(model, inv.MaxContext, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if !fwd.Feasible {
		t.Errorf("Forward at Inverse.MaxContext=%d is not feasible; estimates are not consistent", inv.MaxContext)
	}
}

func TestInverse_ZeroLayerCount_Error(t *testing.T) {
	model := llamaModel()
	model.LayerCount = 0
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Inverse(model, 0, "f16", 1)
	if err == nil {
		t.Error("expected error when LayerCount=0 produces zero bytes-per-token")
	}
}

func TestInverse_ZeroKVHeadCount_Error(t *testing.T) {
	model := llamaModel()
	model.KVHeadCount = 0
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Inverse(model, 0, "f16", 1)
	if err == nil {
		t.Error("expected error for zero KV head count")
	}
}

func TestInverse_ZeroHeadDim_Error(t *testing.T) {
	model := llamaModel()
	model.HeadDim = 0
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Inverse(model, 0, "f16", 1)
	if err == nil {
		t.Error("expected error for zero head dimension")
	}
}

func TestInverse_ResourceQueryError(t *testing.T) {
	e := errEstimator(errors.New("vulkan unavailable"))
	_, err := e.Inverse(llamaModel(), 0, "f16", 1)
	if err == nil {
		t.Error("expected error when resource query fails")
	}
}

// --- nParallel tests ---

func TestForward_NParallel_ScalesKVCost(t *testing.T) {
	model := llamaModel()
	e := testEstimator(256<<30, 256<<30)

	r1, err := e.Forward(model, 4096, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Forward(nParallel=1): %v", err)
	}
	r4, err := e.Forward(model, 4096, 0, "f16", 4)
	if err != nil {
		t.Fatalf("Forward(nParallel=4): %v", err)
	}

	if r4.KVCost != r1.KVCost*4 {
		t.Errorf("KVCost(nParallel=4) = %d, want %d (4× nParallel=1)", r4.KVCost, r1.KVCost*4)
	}
	if r4.WeightCost != r1.WeightCost {
		t.Errorf("WeightCost should be unaffected by nParallel: got %d, want %d", r4.WeightCost, r1.WeightCost)
	}
}

func TestForward_NParallelZero_Error(t *testing.T) {
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Forward(llamaModel(), 4096, 0, "f16", 0)
	if err == nil {
		t.Error("expected error for nParallel=0")
	}
}

func TestForward_NParallelNegative_Error(t *testing.T) {
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Forward(llamaModel(), 4096, 0, "f16", -1)
	if err == nil {
		t.Error("expected error for nParallel=-1")
	}
}

func TestInverse_NParallel_ReducesMaxContext(t *testing.T) {
	// Use a model with large MaxContext and limited memory so the memory budget —
	// not the native cap — is the binding constraint for both nParallel values.
	// 3 GiB total with 1 GiB weights leaves ~1.6 GiB for KV after headroom:
	//   bytesPerToken (f16, nParallel=1) = 32*8*128*2*2 = 131072
	//   maxCtx(nParallel=1) ≈ 13170   (well below MaxContext=100000)
	//   maxCtx(nParallel=4) ≈  3292   (ratio ≈ 4.0)
	model := llamaModel()
	model.MaxContext = 100000 // large native cap — won't be the binding constraint

	e := testEstimator(3<<30, 0)

	r1, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Inverse(nParallel=1): %v", err)
	}
	r4, err := e.Inverse(model, 0, "f16", 4)
	if err != nil {
		t.Fatalf("Inverse(nParallel=4): %v", err)
	}

	if r4.MaxContext == 0 {
		t.Fatal("MaxContext(nParallel=4) is 0 with 3 GiB memory")
	}
	ratio := float64(r1.MaxContext) / float64(r4.MaxContext)
	if ratio < 3.5 || ratio > 4.5 {
		t.Errorf("MaxContext ratio (nParallel=1 / nParallel=4) = %.2f, want ~4.0", ratio)
	}
	// nParallel=4 must be ≤ MaxContext.
	if r4.MaxContext > model.MaxContext {
		t.Errorf("MaxContext(nParallel=4) = %d exceeds native max %d", r4.MaxContext, model.MaxContext)
	}
}

func TestInverse_NParallel_ConsistentWithForward(t *testing.T) {
	model := llamaModel()
	model.MaxContext = 1 << 20 // avoid native-max clamp masking the test
	vram := uint64(12 << 30)
	ram := uint64(4 << 30)

	for _, nParallel := range []int{1, 2, 4} {
		e := testEstimator(vram, ram)
		inv, err := e.Inverse(model, 0, "f16", nParallel)
		if err != nil {
			t.Fatalf("Inverse(nParallel=%d): %v", nParallel, err)
		}
		if inv.MaxContext == 0 {
			t.Skipf("no context fits at nParallel=%d; skipping consistency check", nParallel)
		}
		fwd, err := e.Forward(model, inv.MaxContext, 0, "f16", nParallel)
		if err != nil {
			t.Fatalf("Forward(nParallel=%d): %v", nParallel, err)
		}
		if !fwd.Feasible {
			t.Errorf("Forward at Inverse.MaxContext=%d (nParallel=%d) is not feasible", inv.MaxContext, nParallel)
		}
	}
}

func TestInverse_NParallelZero_Error(t *testing.T) {
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Inverse(llamaModel(), 0, "f16", 0)
	if err == nil {
		t.Error("expected error for nParallel=0")
	}
}

func TestInverse_NParallelNegative_Error(t *testing.T) {
	e := testEstimator(16<<30, 16<<30)
	_, err := e.Inverse(llamaModel(), 0, "f16", -1)
	if err == nil {
		t.Error("expected error for nParallel=-1")
	}
}

// --- sliding-window attention (SWA) tests ---

func TestForward_SWA_Formula_LargeContext(t *testing.T) {
	model := gemma4Model() // SlidingWindowSize = 1024, MaxContext = 262144
	// At ctx = 262144 (> window), the SWA term is bounded by the window:
	//   global: 10 × 4 × 512 × 262144 × 2 elems
	//   swa:    50 × 16 × 256 × 1024  × 2 elems  (min(ctx, 1024) = 1024)
	// f16 = 2 bytes/elem.
	wantKV := uint64(10*4*512*262144*2+50*16*256*1024*2) * 2

	e := testEstimator(256<<30, 256<<30)
	r, err := e.Forward(model, 262144, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != wantKV {
		t.Errorf("KVCost = %d, want %d (global scaling + fixed SWA window)", r.KVCost, wantKV)
	}
	if !r.Feasible {
		t.Error("expected Feasible=true with ample memory")
	}
}

func TestForward_SWA_Formula_SmallContext(t *testing.T) {
	model := gemma4Model()
	// At ctx = 512 (<= window 1024), the SWA term uses ctx, not the window size:
	//   global: 10 × 4 × 512 × 512 × 2 elems
	//   swa:    50 × 16 × 256 × 512 × 2 elems
	wantKV := uint64(10*4*512*512*2+50*16*256*512*2) * 2

	e := testEstimator(256<<30, 256<<30)
	r, err := e.Forward(model, 512, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != wantKV {
		t.Errorf("KVCost = %d, want %d (both terms scale with ctx below window)", r.KVCost, wantKV)
	}
}

func TestInverse_SWA_ReturnsNativeMax(t *testing.T) {
	model := gemma4Model()
	// 64 GiB comfortably fits the full 262144 context under the corrected formula
	// (~47 GiB needed), but the pre-fix all-layers formula would have stopped well
	// short (~19K context). The corrected Inverse must reach the native maximum.
	e := testEstimator(64<<30, 0)
	r, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.MaxContext != 262144 {
		t.Errorf("MaxContext = %d, want 262144 (native max; pre-fix formula yields ~19K)", r.MaxContext)
	}
}

func TestInverse_SWA_ConsistentWithForward(t *testing.T) {
	model := gemma4Model()
	// Choose a budget that constrains the result below the native cap so the
	// consistency check exercises a memory-bound answer, not the clamp.
	model.MaxContext = 1 << 20 // avoid native-max clamp masking the test
	e := testEstimator(48<<30, 0)

	inv, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Inverse error: %v", err)
	}
	if inv.MaxContext == 0 {
		t.Fatal("expected a non-zero context to fit in 48 GiB")
	}
	fwd, err := e.Forward(model, inv.MaxContext, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if !fwd.Feasible {
		t.Errorf("Forward at Inverse.MaxContext=%d is not feasible; estimates inconsistent", inv.MaxContext)
	}
}

func TestInverse_SWA_Phase2_SmallBudget(t *testing.T) {
	model := gemma4Model()
	model.FileSize = 1 << 30 // 1 GiB weights so a small KV budget is reachable
	// 2 GiB total leaves a KV budget smaller than the fixed SWA window cost, so
	// phase 1 is skipped and phase 2 (both terms scale with ctx) yields a result
	// at or below the sliding window size.
	e := testEstimator(2<<30, 0)

	inv, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Inverse error: %v", err)
	}
	if inv.MaxContext == 0 {
		t.Fatal("expected a non-zero context to fit")
	}
	if inv.MaxContext > model.SlidingWindowSize {
		t.Errorf("MaxContext = %d, want <= SlidingWindowSize=%d (phase 2 path)", inv.MaxContext, model.SlidingWindowSize)
	}
	fwd, err := e.Forward(model, inv.MaxContext, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if !fwd.Feasible {
		t.Errorf("Forward at Inverse.MaxContext=%d is not feasible (phase 2)", inv.MaxContext)
	}
}

func TestForward_SWA_NonSWARegression(t *testing.T) {
	// A model with all SWA fields zero must produce exactly the pre-fix all-layers
	// KV cost. llamaModel() has SlidingWindowSize == 0.
	model := llamaModel()
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 2 // pre-fix formula, f16

	e := testEstimator(16<<30, 16<<30)
	r, err := e.Forward(model, 4096, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != wantKV {
		t.Errorf("KVCost = %d, want %d (non-SWA path unchanged)", r.KVCost, wantKV)
	}
}

func TestForward_SWA_Gemma3_Uniform(t *testing.T) {
	model := gemma3Model()
	// At native max 131072 (> window), uniform heads/dims:
	//   global: 10 × 16 × 128 × 131072 × 2 elems
	//   swa:    52 × 16 × 128 × 1024   × 2 elems  (min(ctx, 1024) = 1024)
	// f16 = 2 bytes/elem.
	wantKV := uint64(10*16*128*131072*2+52*16*128*1024*2) * 2

	e := testEstimator(256<<30, 256<<30)
	r, err := e.Forward(model, 131072, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != wantKV {
		t.Errorf("KVCost = %d, want %d (uniform global scaling + fixed SWA window)", r.KVCost, wantKV)
	}
	if !r.Feasible {
		t.Error("expected Feasible=true with ample memory")
	}
}

func TestInverse_SWA_Gemma3_BeatsAllLayers(t *testing.T) {
	model := gemma3Model()
	// 48 GiB budget: the corrected SWA formula (~11 GiB KV at full context) reaches
	// the native max 131072, whereas the pre-fix all-layers formula
	// (62 × 16 × 128 × ctx) would cap near ~50K. Assert the native max is reached.
	e := testEstimator(48<<30, 0)
	r, err := e.Inverse(model, 0, "f16", 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.MaxContext != 131072 {
		t.Errorf("MaxContext = %d, want 131072 (native max; all-layers formula would cap far lower)", r.MaxContext)
	}
	// Consistency: Forward at the reported max must be feasible.
	fwd, err := e.Forward(model, r.MaxContext, 0, "f16", 1)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if !fwd.Feasible {
		t.Errorf("Forward at Inverse.MaxContext=%d is not feasible", r.MaxContext)
	}
}

// --- kvBytesPerElement tests ---

func TestKVBytesPerElement(t *testing.T) {
	cases := []struct {
		kvType string
		want   float64
	}{
		{"f16", 2.0},
		{"F16", 2.0},
		{"bf16", 2.0},
		{"BF16", 2.0},
		{"", 2.0},
		{"unknown", 2.0},
		{"f32", 4.0},
		{"F32", 4.0},
		// q8_0: block_q8_0 = 2-byte delta + 32 int8 values = 34 bytes / 32 elements
		{"q8_0", 34.0 / 32.0},
		{"Q8_0", 34.0 / 32.0},
	}
	for _, tc := range cases {
		got := kvBytesPerElement(tc.kvType)
		if got != tc.want {
			t.Errorf("kvBytesPerElement(%q) = %v, want %v", tc.kvType, got, tc.want)
		}
	}
}
