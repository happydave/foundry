package estimator

import (
	"errors"
	"testing"
)

// testEstimator returns an Estimator with an injected resource querier so tests
// do not depend on actual GPU hardware.
func testEstimator(vramAvail, ramAvail uint64, kvType string) *Estimator {
	e := New(Params{KVCacheType: kvType})
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

// --- Forward tests ---

func TestForward_Formula(t *testing.T) {
	model := llamaModel()
	// KV cost = 32 * 8 * 128 * 4096 * 2 * 2 = 536870912 bytes (512 MiB)
	wantKV := uint64(32) * 8 * 128 * 4096 * 2 * 2
	wantWeight := uint64(1 << 30)
	wantTotal := wantWeight + wantKV

	// Give plenty of memory so feasibility is true.
	e := testEstimator(16<<30, 16<<30, "f16")
	r, err := e.Forward(model, 4096, 0)
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

	e := testEstimator(budget, 0, "f16")
	r, err := e.Forward(model, 4096, 0)
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

	e := testEstimator(budget, 0, "f16")
	r, err := e.Forward(model, 4096, 0)
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
	e := testEstimator(20<<30, 0, "f16")
	r, err := e.Forward(model, 4096, 5<<30)
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
	e := testEstimator(1<<30, 0, "f16")
	r, err := e.Forward(model, 4096, 2<<30)
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

	e := testEstimator(32<<30, 32<<30, "f16")
	r, err := e.Forward(model, 8192, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != kvAt4096 {
		t.Errorf("KVCost = %d, want %d (context clamped to MaxContext=4096)", r.KVCost, kvAt4096)
	}
}

func TestForward_KVType_Q8(t *testing.T) {
	model := llamaModel()
	// q8_0 = 1 byte/elem, so KV cost is half of f16
	kvF16 := uint64(32) * 8 * 128 * 4096 * 2 * 2
	kvQ8 := kvF16 / 2

	e := testEstimator(32<<30, 32<<30, "q8_0")
	r, err := e.Forward(model, 4096, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != kvQ8 {
		t.Errorf("KVCost (q8_0) = %d, want %d", r.KVCost, kvQ8)
	}
}

func TestForward_KVType_F32(t *testing.T) {
	model := llamaModel()
	// f32 = 4 bytes/elem, double f16
	kvF16 := uint64(32) * 8 * 128 * 4096 * 2 * 2
	kvF32 := kvF16 * 2

	e := testEstimator(32<<30, 32<<30, "f32")
	r, err := e.Forward(model, 4096, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if r.KVCost != kvF32 {
		t.Errorf("KVCost (f32) = %d, want %d", r.KVCost, kvF32)
	}
}

func TestForward_ZeroCtx_Error(t *testing.T) {
	e := testEstimator(16<<30, 16<<30, "f16")
	_, err := e.Forward(llamaModel(), 0, 0)
	if err == nil {
		t.Error("expected error for zero context size")
	}
}

func TestForward_ZeroKVHeadCount_Error(t *testing.T) {
	model := llamaModel()
	model.KVHeadCount = 0
	e := testEstimator(16<<30, 16<<30, "f16")
	_, err := e.Forward(model, 4096, 0)
	if err == nil {
		t.Error("expected error for zero KV head count")
	}
}

func TestForward_ZeroHeadDim_Error(t *testing.T) {
	model := llamaModel()
	model.HeadDim = 0
	e := testEstimator(16<<30, 16<<30, "f16")
	_, err := e.Forward(model, 4096, 0)
	if err == nil {
		t.Error("expected error for zero head dimension")
	}
}

func TestForward_ResourceQueryError(t *testing.T) {
	e := errEstimator(errors.New("vulkan unavailable"))
	_, err := e.Forward(llamaModel(), 4096, 0)
	if err == nil {
		t.Error("expected error when resource query fails")
	}
}

// --- Inverse tests ---

func TestInverse_BasicResult(t *testing.T) {
	model := llamaModel() // MaxContext = 4096

	// Give enough memory to load at a non-trivial context.
	e := testEstimator(16<<30, 16<<30, "f16")
	r, err := e.Inverse(model, 0)
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
	e := testEstimator(512<<30, 512<<30, "f16")
	r, err := e.Inverse(model, 0)
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
	e := testEstimator(4<<30, 0, "f16")
	r, err := e.Inverse(model, 0)
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
	e := testEstimator(4<<30, 0, "f16")
	r, err := e.Inverse(model, 4<<30)
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

	e := testEstimator(vram, ram, "f16")
	inv, err := e.Inverse(model, 0)
	if err != nil {
		t.Fatalf("Inverse error: %v", err)
	}
	if inv.MaxContext == 0 {
		t.Skip("no context fits; nothing to verify")
	}

	// Forward at the reported max context must be feasible.
	fwd, err := e.Forward(model, inv.MaxContext, 0)
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
	e := testEstimator(16<<30, 16<<30, "f16")
	_, err := e.Inverse(model, 0)
	if err == nil {
		t.Error("expected error when LayerCount=0 produces zero bytes-per-token")
	}
}

func TestInverse_ZeroKVHeadCount_Error(t *testing.T) {
	model := llamaModel()
	model.KVHeadCount = 0
	e := testEstimator(16<<30, 16<<30, "f16")
	_, err := e.Inverse(model, 0)
	if err == nil {
		t.Error("expected error for zero KV head count")
	}
}

func TestInverse_ZeroHeadDim_Error(t *testing.T) {
	model := llamaModel()
	model.HeadDim = 0
	e := testEstimator(16<<30, 16<<30, "f16")
	_, err := e.Inverse(model, 0)
	if err == nil {
		t.Error("expected error for zero head dimension")
	}
}

func TestInverse_ResourceQueryError(t *testing.T) {
	e := errEstimator(errors.New("vulkan unavailable"))
	_, err := e.Inverse(llamaModel(), 0)
	if err == nil {
		t.Error("expected error when resource query fails")
	}
}

// --- kvBytesPerElement tests ---

func TestKVBytesPerElement(t *testing.T) {
	cases := []struct {
		kvType string
		want   uint64
	}{
		{"f16", 2},
		{"F16", 2},
		{"bf16", 2},
		{"BF16", 2},
		{"", 2},
		{"unknown", 2},
		{"f32", 4},
		{"F32", 4},
		{"q8_0", 1},
		{"Q8_0", 1},
	}
	for _, tc := range cases {
		e := New(Params{KVCacheType: tc.kvType})
		got := e.kvBytesPerElement()
		if got != tc.want {
			t.Errorf("kvBytesPerElement(%q) = %d, want %d", tc.kvType, got, tc.want)
		}
	}
}
