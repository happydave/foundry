package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/happydave/foundry/internal/estimator"
	"github.com/happydave/foundry/internal/processmanager"
	"github.com/happydave/foundry/internal/registry"
)

// --- test doubles ---

type fakeRegistry struct {
	models []registry.Model
}

func (f *fakeRegistry) List() []registry.Model { return f.models }
func (f *fakeRegistry) Get(id uint64) (registry.Model, bool) {
	for _, m := range f.models {
		if m.ID == id {
			return m, true
		}
	}
	return registry.Model{}, false
}
func (f *fakeRegistry) GetByName(name string) (registry.Model, bool) {
	for _, m := range f.models {
		if m.DisplayName == name {
			return m, true
		}
	}
	return registry.Model{}, false
}

type fakeProcMgr struct {
	loaded map[uint64]*processmanager.LoadedModel
	loadFn func(ctx context.Context, id uint64, path string, ctxSize, gpu int) (*processmanager.LoadedModel, error)
	// lastOpts records the ModelLoadOptions passed to the most recent Load call.
	lastOpts processmanager.ModelLoadOptions
}

func newFakeProcMgr() *fakeProcMgr {
	return &fakeProcMgr{loaded: make(map[uint64]*processmanager.LoadedModel)}
}

func (f *fakeProcMgr) Load(ctx context.Context, modelID uint64, modelPath, mmprojPath string, contextSize, gpuLayers int, opts processmanager.ModelLoadOptions) (*processmanager.LoadedModel, error) {
	f.lastOpts = opts
	if f.loadFn != nil {
		return f.loadFn(ctx, modelID, modelPath, contextSize, gpuLayers)
	}
	lm := &processmanager.LoadedModel{
		ModelID:     modelID,
		PID:         12345,
		Port:        9000,
		ContextSize: contextSize,
		GPULayers:   gpuLayers,
		LoadTime:    time.Now(),
	}
	f.loaded[modelID] = lm
	return lm, nil
}

func (f *fakeProcMgr) Unload(_ context.Context, modelID uint64) error {
	if _, ok := f.loaded[modelID]; !ok {
		return fmt.Errorf("model %d is not loaded", modelID)
	}
	delete(f.loaded, modelID)
	return nil
}

func (f *fakeProcMgr) List() []*processmanager.LoadedModel {
	out := make([]*processmanager.LoadedModel, 0, len(f.loaded))
	for _, lm := range f.loaded {
		out = append(out, lm)
	}
	return out
}

func (f *fakeProcMgr) Get(modelID uint64) (*processmanager.LoadedModel, bool) {
	lm, ok := f.loaded[modelID]
	return lm, ok
}

type fakeEstimator struct {
	forwardFn func(model estimator.ModelSpec, ctxLen uint32, inUse uint64, kvType string, nParallel int) (estimator.ForwardResult, error)
	inverseFn func(model estimator.ModelSpec, inUse uint64, kvType string, nParallel int) (estimator.InverseResult, error)
}

func (f *fakeEstimator) Forward(model estimator.ModelSpec, ctxLen uint32, inUse uint64, kvType string, nParallel int) (estimator.ForwardResult, error) {
	if f.forwardFn != nil {
		return f.forwardFn(model, ctxLen, inUse, kvType, nParallel)
	}
	return estimator.ForwardResult{TotalCost: 1 << 30, Feasible: true}, nil
}

func (f *fakeEstimator) Inverse(model estimator.ModelSpec, inUse uint64, kvType string, nParallel int) (estimator.InverseResult, error) {
	if f.inverseFn != nil {
		return f.inverseFn(model, inUse, kvType, nParallel)
	}
	return estimator.InverseResult{MaxContext: 4096}, nil
}

// --- test server fixture ---

type serverFixture struct {
	reg  *fakeRegistry
	pm   *fakeProcMgr
	est  *fakeEstimator
	srv  *Server
	addr string
}

func newFixture(t *testing.T) *serverFixture {
	t.Helper()
	reg := &fakeRegistry{}
	pm := newFakeProcMgr()
	est := &fakeEstimator{}

	s := newServer("127.0.0.1:0", reg, pm, est, 35, "q8_0", 1, nil, nil)
	s.queryResources = func() (uint64, uint64, error) { return 8 << 30, 8 << 30, nil }
	s.queryVRAMTotal = func() (uint64, error) { return 16 << 30, nil }
	s.queryHardware = func() ([]estimator.GPUInfo, uint64, error) {
		return []estimator.GPUInfo{
			{Index: 0, Identity: "1002:164e", VRAMTotal: 16 << 30, VRAMUsed: 8 << 30, VRAMAvail: 8 << 30},
		}, 32 << 30, nil
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	httpSrv := &http.Server{Handler: s.http.Handler}
	go func() { _ = httpSrv.Serve(ln) }()
	t.Cleanup(func() { _ = httpSrv.Shutdown(context.Background()) })

	return &serverFixture{
		reg:  reg,
		pm:   pm,
		est:  est,
		srv:  s,
		addr: "http://" + ln.Addr().String(),
	}
}

func (f *serverFixture) do(t *testing.T, method, path string, body any) *http.Response {
	t.Helper()
	var buf bytes.Buffer
	var contentLength int64
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatal(err)
		}
		contentLength = int64(buf.Len())
	}
	req, err := http.NewRequest(method, f.addr+path, &buf)
	if err != nil {
		t.Fatal(err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
		req.ContentLength = contentLength
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func decodeBody[T any](t *testing.T, resp *http.Response) T {
	t.Helper()
	defer func() { _ = resp.Body.Close() }()
	var v T
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return v
}

func assertStatus(t *testing.T, resp *http.Response, want int) {
	t.Helper()
	if resp.StatusCode != want {
		_ = resp.Body.Close()
		t.Fatalf("got HTTP %d, want %d", resp.StatusCode, want)
	}
}

// --- test model ---

func testModel(id uint64) registry.Model {
	return registry.Model{
		ID:           id,
		DisplayName:  "test-model",
		Path:         "/models/test.gguf",
		FileSize:     1 << 30,
		Architecture: "llama",
		LayerCount:   32,
		HeadCount:    32,
		KVHeadCount:  8,
		HeadDim:      128,
		MaxContext:   4096,
		Quantization: "Q4_K_M",
	}
}

// --- GET /api/v1/models ---

func TestListModels_Empty(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/api/v1/models", nil)
	assertStatus(t, resp, http.StatusOK)
	mr := decodeBody[lmsModelsResponse](t, resp)
	if mr.Models == nil {
		t.Fatal("expected JSON array, got null")
	}
	if len(mr.Models) != 0 {
		t.Fatalf("expected empty models array, got %d entries", len(mr.Models))
	}
}

func TestListModels_LoadStatus(t *testing.T) {
	f := newFixture(t)
	m1 := testModel(1)
	m1.DisplayName = "model-one"
	m2 := testModel(2)
	m2.DisplayName = "model-two"
	f.reg.models = []registry.Model{m1, m2}
	f.pm.loaded[m1.ID] = &processmanager.LoadedModel{ModelID: m1.ID, ContextSize: 4096}

	resp := f.do(t, http.MethodGet, "/api/v1/models", nil)
	mr := decodeBody[lmsModelsResponse](t, resp)

	if len(mr.Models) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(mr.Models))
	}
	byKey := map[string]lmsModelEntry{}
	for _, e := range mr.Models {
		byKey[e.Key] = e
	}
	if len(byKey["model-one"].LoadedInstances) == 0 {
		t.Error("model-one should have a loaded instance")
	}
	if len(byKey["model-two"].LoadedInstances) != 0 {
		t.Error("model-two should not have loaded instances")
	}
}

func TestListModels_Shape(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodGet, "/api/v1/models", nil)
	assertStatus(t, resp, http.StatusOK)
	mr := decodeBody[lmsModelsResponse](t, resp)

	if len(mr.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(mr.Models))
	}
	e := mr.Models[0]

	if e.Key != m.DisplayName {
		t.Errorf("key = %q, want %q", e.Key, m.DisplayName)
	}
	if e.DisplayName != m.DisplayName {
		t.Errorf("display_name = %q, want %q", e.DisplayName, m.DisplayName)
	}
	if e.Key != e.DisplayName {
		t.Errorf("key %q != display_name %q", e.Key, e.DisplayName)
	}
	if e.Type != "llm" {
		t.Errorf("type = %q, want llm", e.Type)
	}
	if e.Architecture != m.Architecture {
		t.Errorf("architecture = %q, want %q", e.Architecture, m.Architecture)
	}
	if e.SizeBytes != m.FileSize {
		t.Errorf("size_bytes = %d, want %d", e.SizeBytes, m.FileSize)
	}
	if e.ContextLength != m.MaxContext {
		t.Errorf("context_length = %d, want %d", e.ContextLength, m.MaxContext)
	}
	if e.Quantization.Name != m.Quantization {
		t.Errorf("quantization.name = %q, want %q", e.Quantization.Name, m.Quantization)
	}
	if e.LoadedInstances == nil {
		t.Error("loaded_instances must not be null when model is not loaded")
	}
}

func TestListModels_LoadedInstanceFields(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[m.ID] = &processmanager.LoadedModel{ModelID: m.ID, ContextSize: 2048, Parallel: 1}

	resp := f.do(t, http.MethodGet, "/api/v1/models", nil)
	mr := decodeBody[lmsModelsResponse](t, resp)

	if len(mr.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(mr.Models))
	}
	instances := mr.Models[0].LoadedInstances
	if len(instances) != 1 {
		t.Fatalf("expected 1 loaded instance, got %d", len(instances))
	}
	inst := instances[0]
	if inst.ID != m.DisplayName {
		t.Errorf("instance id = %q, want %q", inst.ID, m.DisplayName)
	}
	if inst.Config.ContextLength != 2048 {
		t.Errorf("instance context_length = %d, want 2048", inst.Config.ContextLength)
	}
	if inst.Config.EvalBatchSize != 512 {
		t.Errorf("instance eval_batch_size = %d, want 512", inst.Config.EvalBatchSize)
	}
	if inst.Config.Parallel != 1 {
		t.Errorf("instance parallel = %d, want 1", inst.Config.Parallel)
	}
}

// --- GET /api/v1/models/{id} ---

func TestGetModel_NotFound(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/api/v1/models/999", nil)
	assertStatus(t, resp, http.StatusNotFound)
	e := decodeBody[apiError](t, resp)
	if e.Error == "" {
		t.Error("expected error field in response")
	}
}

func TestGetModel_BadID(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/api/v1/models/abc", nil)
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

func TestGetModel_Found(t *testing.T) {
	f := newFixture(t)
	m := testModel(42)
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodGet, "/api/v1/models/42", nil)
	assertStatus(t, resp, http.StatusOK)
	detail := decodeBody[modelDetailResponse](t, resp)

	if detail.ID != 42 {
		t.Errorf("got ID %d, want 42", detail.ID)
	}
	if detail.MaxLoadableContext != 4096 {
		t.Errorf("got MaxLoadableContext %d, want 4096", detail.MaxLoadableContext)
	}
	if detail.NativeEstimate.CostBytes == 0 {
		t.Error("expected non-zero native estimate cost")
	}
}

// --- POST /api/v1/models/{id}/load ---

func TestLoadModel_NotFound(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodPost, "/api/v1/models/999/load", nil)
	assertStatus(t, resp, http.StatusNotFound)
	_ = resp.Body.Close()
}

func TestLoadModel_NoContext_UsesEstimate(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	var gotCtx int
	f.pm.loadFn = func(_ context.Context, id uint64, _ string, ctxSize, gpu int) (*processmanager.LoadedModel, error) {
		gotCtx = ctxSize
		return &processmanager.LoadedModel{
			ModelID: id, PID: 1, Port: 9000,
			ContextSize: ctxSize, GPULayers: gpu,
			LoadTime: time.Now(),
		}, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", nil)
	assertStatus(t, resp, http.StatusOK)
	lr := decodeBody[loadedModelResponse](t, resp)

	if gotCtx != 4096 {
		t.Errorf("expected estimated max context 4096, got %d", gotCtx)
	}
	if lr.ModelID != 1 {
		t.Errorf("expected model_id 1, got %d", lr.ModelID)
	}
}

func TestLoadModel_WithContext(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	var gotCtx int
	f.pm.loadFn = func(_ context.Context, id uint64, _ string, ctxSize, gpu int) (*processmanager.LoadedModel, error) {
		gotCtx = ctxSize
		return &processmanager.LoadedModel{
			ModelID: id, PID: 1, Port: 9000,
			ContextSize: ctxSize, GPULayers: gpu,
			LoadTime: time.Now(),
		}, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", loadRequest{Ctx: 2048})
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()
	if gotCtx != 2048 {
		t.Errorf("expected ctx 2048, got %d", gotCtx)
	}
}

func TestLoadModel_ContextClampsToNativeMax(t *testing.T) {
	f := newFixture(t)
	m := testModel(1) // MaxContext = 4096
	f.reg.models = []registry.Model{m}

	var gotCtx int
	f.pm.loadFn = func(_ context.Context, id uint64, _ string, ctxSize, gpu int) (*processmanager.LoadedModel, error) {
		gotCtx = ctxSize
		return &processmanager.LoadedModel{
			ModelID: id, PID: 1, Port: 9000,
			ContextSize: ctxSize, GPULayers: gpu,
			LoadTime: time.Now(),
		}, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", loadRequest{Ctx: 99999})
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()
	if gotCtx != 4096 {
		t.Errorf("expected ctx clamped to 4096, got %d", gotCtx)
	}
}

func TestLoadModel_InfeasibleContext_Returns422(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.est.forwardFn = func(_ estimator.ModelSpec, _ uint32, _ uint64, _ string, _ int) (estimator.ForwardResult, error) {
		return estimator.ForwardResult{TotalCost: 100 << 30, Feasible: false}, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", loadRequest{Ctx: 4096})
	assertStatus(t, resp, http.StatusUnprocessableEntity)
	_ = resp.Body.Close()
}

func TestLoadModel_NoMemory_Returns422(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.est.inverseFn = func(_ estimator.ModelSpec, _ uint64, _ string, _ int) (estimator.InverseResult, error) {
		return estimator.InverseResult{MaxContext: 0}, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", nil)
	assertStatus(t, resp, http.StatusUnprocessableEntity)
	_ = resp.Body.Close()
}

func TestLoadModel_MalformedBody_Returns400(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	body := bytes.NewBufferString("{bad json")
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/api/v1/models/1/load", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

func TestLoadModel_UsesConfigGPULayers(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.srv.defaultGPULayers = 99

	var gotGPU int
	f.pm.loadFn = func(_ context.Context, id uint64, _ string, ctxSize, gpu int) (*processmanager.LoadedModel, error) {
		gotGPU = gpu
		return &processmanager.LoadedModel{
			ModelID: id, PID: 1, Port: 9000,
			ContextSize: ctxSize, GPULayers: gpu,
			LoadTime: time.Now(),
		}, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", nil)
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()
	if gotGPU != 99 {
		t.Errorf("expected gpu layers 99, got %d", gotGPU)
	}
}

func TestLoadModel_PerModelArgs_Threaded(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.srv.resolvedModelOpts = map[string]processmanager.ModelLoadOptions{
		"test-model": {Args: []string{"--chat-template-file", "/path/to/template.jinja"}, KVCacheType: "q8_0"},
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", nil)
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	want := []string{"--chat-template-file", "/path/to/template.jinja"}
	if len(f.pm.lastOpts.Args) != len(want) {
		t.Fatalf("opts.Args = %v, want %v", f.pm.lastOpts.Args, want)
	}
	for i, v := range want {
		if f.pm.lastOpts.Args[i] != v {
			t.Errorf("opts.Args[%d] = %q, want %q", i, f.pm.lastOpts.Args[i], v)
		}
	}
}

func TestLoadModel_NoPerModelArgs_NilArgs(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	// resolvedModelOpts is nil — model has no explicit config

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", nil)
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	if f.pm.lastOpts.Args != nil {
		t.Errorf("expected nil Args for model with no config, got %v", f.pm.lastOpts.Args)
	}
}

func TestLoadModel_AlreadyLoaded_ReturnsExisting(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	existing := &processmanager.LoadedModel{
		ModelID: 1, PID: 999, Port: 8080, ContextSize: 2048,
		GPULayers: 35, LoadTime: time.Now(),
	}
	f.pm.loaded[1] = existing
	f.pm.loadFn = func(_ context.Context, _ uint64, _ string, _, _ int) (*processmanager.LoadedModel, error) {
		return existing, nil
	}

	resp := f.do(t, http.MethodPost, "/api/v1/models/1/load", nil)
	assertStatus(t, resp, http.StatusOK)
	lr := decodeBody[loadedModelResponse](t, resp)
	if lr.PID != 999 {
		t.Errorf("expected existing PID 999, got %d", lr.PID)
	}
}

// --- DELETE /api/v1/models/{id} ---

func TestUnloadModel_NotInRegistry_Returns404(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodDelete, "/api/v1/models/999", nil)
	assertStatus(t, resp, http.StatusNotFound)
	_ = resp.Body.Close()
}

func TestUnloadModel_NotLoaded_Returns404(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodDelete, "/api/v1/models/1", nil)
	assertStatus(t, resp, http.StatusNotFound)
	e := decodeBody[apiError](t, resp)
	if e.Error == "" {
		t.Error("expected error field")
	}
}

func TestUnloadModel_Loaded_Returns204(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{ModelID: 1}

	resp := f.do(t, http.MethodDelete, "/api/v1/models/1", nil)
	assertStatus(t, resp, http.StatusNoContent)
	_ = resp.Body.Close()

	if _, ok := f.pm.loaded[1]; ok {
		t.Error("model should have been removed from process manager")
	}
}

// --- GET /api/v1/models/{id}/estimate ---

func TestEstimate_MissingCtx_Returns400(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodGet, "/api/v1/models/1/estimate", nil)
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

func TestEstimate_ZeroCtx_Returns400(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodGet, "/api/v1/models/1/estimate?ctx=0", nil)
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

func TestEstimate_NonIntCtx_Returns400(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodGet, "/api/v1/models/1/estimate?ctx=banana", nil)
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

func TestEstimate_Valid(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.est.forwardFn = func(_ estimator.ModelSpec, ctxLen uint32, _ uint64, _ string, _ int) (estimator.ForwardResult, error) {
		return estimator.ForwardResult{TotalCost: uint64(ctxLen) * 1024, Feasible: true}, nil
	}

	resp := f.do(t, http.MethodGet, "/api/v1/models/1/estimate?ctx=4096", nil)
	assertStatus(t, resp, http.StatusOK)
	er := decodeBody[estimateResponse](t, resp)

	if er.CostBytes != 4096*1024 {
		t.Errorf("got cost %d, want %d", er.CostBytes, 4096*1024)
	}
	if !er.Feasible {
		t.Error("expected feasible=true")
	}
	if er.AvailableBytes == 0 {
		t.Error("expected non-zero available_bytes")
	}
}

func TestEstimate_ModelNotFound_Returns404(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/api/v1/models/42/estimate?ctx=4096", nil)
	assertStatus(t, resp, http.StatusNotFound)
	_ = resp.Body.Close()
}

// --- GET /api/v1/status ---

func TestStatus_NoModels(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/api/v1/status", nil)
	assertStatus(t, resp, http.StatusOK)
	sr := decodeBody[statusResponse](t, resp)

	if sr.Status != "ok" {
		t.Errorf("expected status ok, got %q", sr.Status)
	}
	if sr.LoadedModels == nil {
		t.Error("loaded_models should be a JSON array, not null")
	}
	if len(sr.LoadedModels) != 0 {
		t.Errorf("expected 0 loaded models, got %d", len(sr.LoadedModels))
	}
	if sr.Memory.TotalVRAM != 16<<30 {
		t.Errorf("unexpected total VRAM: %d", sr.Memory.TotalVRAM)
	}
	if sr.Memory.AvailableVRAM != 8<<30 {
		t.Errorf("unexpected available VRAM: %d", sr.Memory.AvailableVRAM)
	}
	if sr.Memory.InUseVRAM != 8<<30 {
		t.Errorf("unexpected in-use VRAM: %d", sr.Memory.InUseVRAM)
	}
}

func TestStatus_WithLoadedModel(t *testing.T) {
	f := newFixture(t)
	m := testModel(5)
	m.DisplayName = "loaded-model"
	f.reg.models = []registry.Model{m}
	f.pm.loaded[5] = &processmanager.LoadedModel{ModelID: 5, Port: 9100, ContextSize: 8192}

	resp := f.do(t, http.MethodGet, "/api/v1/status", nil)
	assertStatus(t, resp, http.StatusOK)
	sr := decodeBody[statusResponse](t, resp)

	if len(sr.LoadedModels) != 1 {
		t.Fatalf("expected 1 loaded model, got %d", len(sr.LoadedModels))
	}
	got := sr.LoadedModels[0]
	// Backward-compatible fields.
	if got.ModelID != 5 || got.Port != 9100 {
		t.Errorf("unexpected base loaded model entry: %+v", got)
	}
	// Enriched fields.
	if got.DisplayName != "loaded-model" {
		t.Errorf("expected display_name loaded-model, got %q", got.DisplayName)
	}
	if got.ContextSize != 8192 {
		t.Errorf("expected context_size 8192, got %d", got.ContextSize)
	}
	if got.Health == "" {
		t.Error("expected non-empty health")
	}
	if got.EstimatedVRAMBytes == 0 {
		t.Error("expected non-zero estimated VRAM for a registry-known model")
	}
}

func TestStatus_LoadedModelMissingFromRegistry(t *testing.T) {
	f := newFixture(t)
	// Loaded model not present in the (empty) registry: status must still 200
	// with the entry present and enrichment defaulted.
	f.pm.loaded[7] = &processmanager.LoadedModel{ModelID: 7, Port: 9200, ContextSize: 4096}

	resp := f.do(t, http.MethodGet, "/api/v1/status", nil)
	assertStatus(t, resp, http.StatusOK)
	sr := decodeBody[statusResponse](t, resp)

	if len(sr.LoadedModels) != 1 {
		t.Fatalf("expected 1 loaded model, got %d", len(sr.LoadedModels))
	}
	got := sr.LoadedModels[0]
	if got.ModelID != 7 || got.Port != 9200 {
		t.Errorf("unexpected entry: %+v", got)
	}
	if got.DisplayName != "" || got.EstimatedVRAMBytes != 0 {
		t.Errorf("expected defaulted enrichment for unknown model, got %+v", got)
	}
	if got.ContextSize != 4096 {
		t.Errorf("expected context_size 4096 from loaded record, got %d", got.ContextSize)
	}
}

// --- Content-Type ---

func TestResponseContentType(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/api/v1/status", nil)
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected Content-Type application/json, got %q", ct)
	}
}

// --- /v1/ unimplemented catch-all ---

func TestV1Route_UnknownPath_Returns501(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/v1/unknown/endpoint", nil)
	assertStatus(t, resp, http.StatusNotImplemented)
	_ = resp.Body.Close()
}

// --- GET /v1/models ---

func TestOAIModels_Empty(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodGet, "/v1/models", nil)
	assertStatus(t, resp, http.StatusOK)
	mr := decodeBody[oaiModelsResponse](t, resp)
	if mr.Object != "list" {
		t.Errorf("expected object=list, got %q", mr.Object)
	}
	if mr.Data == nil || len(mr.Data) != 0 {
		t.Errorf("expected empty Data array, got %v", mr.Data)
	}
}

func TestOAIModels_ListsHealthyLoaded(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{
		ModelID:  1,
		Port:     9000,
		LoadTime: time.Unix(1000, 0),
	}

	resp := f.do(t, http.MethodGet, "/v1/models", nil)
	assertStatus(t, resp, http.StatusOK)
	mr := decodeBody[oaiModelsResponse](t, resp)

	if len(mr.Data) != 1 {
		t.Fatalf("expected 1 model, got %d", len(mr.Data))
	}
	if mr.Data[0].ID != "test-model" {
		t.Errorf("expected id=test-model, got %q", mr.Data[0].ID)
	}
	if mr.Data[0].Object != "model" {
		t.Errorf("expected object=model, got %q", mr.Data[0].Object)
	}
	if mr.Data[0].Created != 1000 {
		t.Errorf("expected created=1000, got %d", mr.Data[0].Created)
	}
}

// --- POST /v1/chat/completions and /v1/completions ---

func TestInferenceProxy_InvalidJSON(t *testing.T) {
	f := newFixture(t)
	body := strings.NewReader("{bad json")
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusBadRequest)
	eb := decodeBody[oaiErrorBody](t, resp)
	if eb.Error.Message == "" {
		t.Error("expected error message")
	}
}

func TestInferenceProxy_EmptyBody(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodPost, "/v1/chat/completions", nil)
	assertStatus(t, resp, http.StatusBadRequest)
	_ = resp.Body.Close()
}

func TestInferenceProxy_MissingModelField(t *testing.T) {
	f := newFixture(t)
	body := strings.NewReader(`{"messages":[]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusBadRequest)
	eb := decodeBody[oaiErrorBody](t, resp)
	if !strings.Contains(eb.Error.Message, "model") {
		t.Errorf("expected error message to mention model field, got %q", eb.Error.Message)
	}
}

func TestInferenceProxy_ModelNotInRegistry(t *testing.T) {
	f := newFixture(t)
	body := strings.NewReader(`{"model":"unknown-model","messages":[]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusNotFound)
	eb := decodeBody[oaiErrorBody](t, resp)
	if eb.Error.Code != "model_not_found" {
		t.Errorf("expected code=model_not_found, got %q", eb.Error.Code)
	}
}

func TestInferenceProxy_ModelNotLoaded(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	// model is in registry but not in process manager

	body := strings.NewReader(`{"model":"test-model","messages":[]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusNotFound)
	eb := decodeBody[oaiErrorBody](t, resp)
	if eb.Error.Code != "model_not_found" {
		t.Errorf("expected code=model_not_found, got %q", eb.Error.Code)
	}
}

func TestInferenceProxy_ProxiesRequest(t *testing.T) {
	// Start a fake llama-server that echoes back what it received.
	var receivedBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"test","choices":[]}`))
	}))
	defer backend.Close()

	// Extract the port from the backend address.
	backendAddr := backend.Listener.Addr().(*net.TCPAddr)

	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{
		ModelID: 1,
		Port:    backendAddr.Port,
	}

	reqBody := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	body := strings.NewReader(reqBody)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/chat/completions", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	if string(receivedBody) != reqBody {
		t.Errorf("backend received %q, want %q", receivedBody, reqBody)
	}
}

func TestInferenceProxy_CompletionsEndpoint(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"test","choices":[]}`))
	}))
	defer backend.Close()

	backendAddr := backend.Listener.Addr().(*net.TCPAddr)

	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{
		ModelID: 1,
		Port:    backendAddr.Port,
	}

	body := strings.NewReader(`{"model":"test-model","prompt":"hello"}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/completions", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()
}

// --- POST /v1/messages ---

func TestMessagesProxy_InvalidJSON(t *testing.T) {
	f := newFixture(t)
	body := strings.NewReader("{bad json")
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/messages", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusBadRequest)
	eb := decodeBody[anthropicErrorBody](t, resp)
	if eb.Type != "error" {
		t.Errorf("expected type=error, got %q", eb.Type)
	}
	if eb.Error.Message == "" {
		t.Error("expected error message")
	}
}

func TestMessagesProxy_EmptyBody(t *testing.T) {
	f := newFixture(t)
	resp := f.do(t, http.MethodPost, "/v1/messages", nil)
	assertStatus(t, resp, http.StatusBadRequest)
	eb := decodeBody[anthropicErrorBody](t, resp)
	if eb.Type != "error" {
		t.Errorf("expected type=error, got %q", eb.Type)
	}
}

func TestMessagesProxy_MissingModelField(t *testing.T) {
	f := newFixture(t)
	body := strings.NewReader(`{"messages":[]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/messages", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusBadRequest)
	eb := decodeBody[anthropicErrorBody](t, resp)
	if !strings.Contains(eb.Error.Message, "model") {
		t.Errorf("expected error message to mention model field, got %q", eb.Error.Message)
	}
}

func TestMessagesProxy_ModelNotInRegistry(t *testing.T) {
	f := newFixture(t)
	body := strings.NewReader(`{"model":"unknown-model","messages":[]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/messages", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusNotFound)
	eb := decodeBody[anthropicErrorBody](t, resp)
	if eb.Error.Type != "not_found_error" {
		t.Errorf("expected error type not_found_error, got %q", eb.Error.Type)
	}
}

func TestMessagesProxy_ModelNotLoaded(t *testing.T) {
	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}

	body := strings.NewReader(`{"model":"test-model","messages":[]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/messages", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusNotFound)
	eb := decodeBody[anthropicErrorBody](t, resp)
	if eb.Error.Type != "not_found_error" {
		t.Errorf("expected error type not_found_error, got %q", eb.Error.Type)
	}
}

func TestMessagesProxy_ProxiesRequest(t *testing.T) {
	var receivedPath string
	var receivedBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"type":"message","content":[]}`))
	}))
	defer backend.Close()

	backendAddr := backend.Listener.Addr().(*net.TCPAddr)

	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{
		ModelID: 1,
		Port:    backendAddr.Port,
	}

	reqBody := `{"model":"test-model","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`
	body := strings.NewReader(reqBody)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/messages", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	if receivedPath != "/v1/messages" {
		t.Errorf("backend received path %q, want /v1/messages", receivedPath)
	}
	if string(receivedBody) != reqBody {
		t.Errorf("backend received %q, want %q", receivedBody, reqBody)
	}
}

func TestMessagesProxy_InferenceHookNotCalled(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"type":"message","content":[]}`))
	}))
	defer backend.Close()

	backendAddr := backend.Listener.Addr().(*net.TCPAddr)

	f := newFixture(t)
	m := testModel(1)
	f.reg.models = []registry.Model{m}
	f.pm.loaded[1] = &processmanager.LoadedModel{
		ModelID: 1,
		Port:    backendAddr.Port,
	}

	hookCalled := false
	f.srv.inferenceHook = func(w http.ResponseWriter, r *http.Request) bool {
		hookCalled = true
		return true
	}

	body := strings.NewReader(`{"model":"test-model","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}`)
	req, _ := http.NewRequest(http.MethodPost, f.addr+"/v1/messages", body)
	req.ContentLength = int64(body.Len())
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	assertStatus(t, resp, http.StatusOK)
	_ = resp.Body.Close()

	if hookCalled {
		t.Error("inference hook must not be called for /v1/messages requests")
	}
}

// --- Graceful shutdown (tests the http.Server directly) ---

func TestServer_GracefulShutdown(t *testing.T) {
	slowDone := make(chan struct{})
	mux := http.NewServeMux()
	mux.HandleFunc("/slow", func(w http.ResponseWriter, r *http.Request) {
		<-slowDone
		w.WriteHeader(http.StatusOK)
	})

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()

	srv := &http.Server{Handler: mux}
	go func() { _ = srv.Serve(ln) }()

	reqDone := make(chan error, 1)
	go func() {
		resp, err := http.Get("http://" + addr + "/slow")
		if err != nil {
			reqDone <- err
			return
		}
		_ = resp.Body.Close()
		reqDone <- nil
	}()

	time.Sleep(50 * time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	shutdownDone := make(chan error, 1)
	go func() {
		shutdownDone <- srv.Shutdown(ctx)
	}()

	time.Sleep(50 * time.Millisecond)
	close(slowDone)

	if err := <-reqDone; err != nil {
		t.Fatalf("in-flight request failed: %v", err)
	}
	if err := <-shutdownDone; err != nil {
		t.Fatalf("shutdown returned error: %v", err)
	}
}

// --- GET /api/v1/hardware ---

func TestHardware_Success(t *testing.T) {
	f := newFixture(t)
	f.srv.queryHardware = func() ([]estimator.GPUInfo, uint64, error) {
		return []estimator.GPUInfo{
			{Index: 0, Identity: "1002:164e", VRAMTotal: 16 << 30, VRAMUsed: 4 << 30, VRAMAvail: 12 << 30},
			{Index: 1, Identity: "Radeon RX 7900", VRAMTotal: 24 << 30, VRAMUsed: 24 << 30, VRAMAvail: 0},
		}, 64 << 30, nil
	}

	resp := f.do(t, http.MethodGet, "/api/v1/hardware", nil)
	assertStatus(t, resp, http.StatusOK)
	hw := decodeBody[hardwareResponse](t, resp)

	if hw.GPUs == nil {
		t.Fatal("expected gpus array, got null")
	}
	if len(hw.GPUs) != 2 {
		t.Fatalf("expected 2 GPUs, got %d", len(hw.GPUs))
	}
	if hw.GPUs[0].Identity != "1002:164e" || hw.GPUs[0].VRAMAvailableBytes != 12<<30 {
		t.Fatalf("unexpected first GPU: %+v", hw.GPUs[0])
	}
	if hw.GPUs[1].Identity == "" {
		t.Fatal("expected non-empty identity for second GPU")
	}
	if hw.SystemRAMAvailableBytes != 64<<30 {
		t.Fatalf("unexpected system RAM: %d", hw.SystemRAMAvailableBytes)
	}
}

func TestHardware_QueryError(t *testing.T) {
	f := newFixture(t)
	f.srv.queryHardware = func() ([]estimator.GPUInfo, uint64, error) {
		return nil, 0, fmt.Errorf("no AMD DRM VRAM sysfs entries found")
	}

	resp := f.do(t, http.MethodGet, "/api/v1/hardware", nil)
	if resp.StatusCode == http.StatusOK {
		t.Fatalf("expected non-200 on query error, got 200")
	}
}

// --- /ui/ operator console ---

func TestUI_DisabledReturns404(t *testing.T) {
	f := newFixture(t) // uiEnabled defaults to false
	resp := f.do(t, http.MethodGet, "/ui/", nil)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 when UI disabled, got %d", resp.StatusCode)
	}
}

func TestUI_EnabledServesIndex(t *testing.T) {
	f := newFixture(t)
	f.srv.SetUIEnabled(true)

	resp := f.do(t, http.MethodGet, "/ui/", nil)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 when UI enabled, got %d", resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(body), "Foundry") {
		t.Fatalf("expected index HTML to mention Foundry, got %q", string(body[:min(80, len(body))]))
	}
}

func TestUI_EnabledServesAsset(t *testing.T) {
	f := newFixture(t)
	f.srv.SetUIEnabled(true)

	resp := f.do(t, http.MethodGet, "/ui/app.js", nil)
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 for /ui/app.js when enabled, got %d", resp.StatusCode)
	}
}

func TestListModels_ExposesStringID(t *testing.T) {
	f := newFixture(t)
	m := testModel(0xDEADBEEFCAFEF00D) // > 2^53; must round-trip as a string
	f.reg.models = []registry.Model{m}

	resp := f.do(t, http.MethodGet, "/api/v1/models", nil)
	assertStatus(t, resp, http.StatusOK)
	mr := decodeBody[lmsModelsResponse](t, resp)
	if len(mr.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(mr.Models))
	}
	want := "16045690984503111693" // strconv of the fingerprint above
	if mr.Models[0].ID != want {
		t.Fatalf("expected string id %q, got %q", want, mr.Models[0].ID)
	}
}
