package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"strconv"
	"time"

	"github.com/happydave/foundry/internal/estimator"
	"github.com/happydave/foundry/internal/history"
	"github.com/happydave/foundry/internal/processmanager"
	"github.com/happydave/foundry/internal/registry"
)

const shutdownTimeout = 30 * time.Second

// modelRegistry is the subset of registry.Registry used by the server.
type modelRegistry interface {
	List() []registry.Model
	Get(id uint64) (registry.Model, bool)
	GetByName(name string) (registry.Model, bool)
}

// processManager is the subset of processmanager.Manager used by the server.
type processManager interface {
	Load(ctx context.Context, modelID uint64, modelPath, mmprojPath string, contextSize, gpuLayers int, opts processmanager.ModelLoadOptions) (*processmanager.LoadedModel, error)
	Unload(ctx context.Context, modelID uint64) error
	List() []*processmanager.LoadedModel
	Get(modelID uint64) (*processmanager.LoadedModel, bool)
}

// resourceEstimator is the subset of estimator.Estimator used by the server.
type resourceEstimator interface {
	Forward(model estimator.ModelSpec, ctxLen uint32, inUseBytes uint64, kvType string, nParallel int) (estimator.ForwardResult, error)
	Inverse(model estimator.ModelSpec, inUseBytes uint64, kvType string, nParallel int) (estimator.InverseResult, error)
}

type Server struct {
	http              *http.Server
	registry          modelRegistry
	procMgr           processManager
	estimator         resourceEstimator
	defaultGPULayers  int
	globalKVCacheType string
	globalParallel    int
	// resolvedModelOpts maps DisplayName to pre-resolved per-model load options
	// (KV cache type already resolved against the global default, Args populated
	// from chat-template config). Models absent from the map use globalKVCacheType
	// and globalParallel.
	resolvedModelOpts map[string]processmanager.ModelLoadOptions
	logger            *slog.Logger

	// inferenceHook, if set, is called before each inference request is proxied.
	inferenceHook InferenceHook

	// historyStore, if set, enables persistent session history for chat completions.
	historyStore history.Store

	// uiEnabled gates serving of the embedded operator console under /ui/.
	uiEnabled bool

	// Injectable for testing; default to the real sysfs implementations.
	queryResources func() (vramAvail, ramAvail uint64, err error)
	queryVRAMTotal func() (uint64, error)
	queryHardware  func() (gpus []estimator.GPUInfo, ramAvail uint64, err error)
}

func New(addr string, reg *registry.Registry, pm *processmanager.Manager, est *estimator.Estimator, defaultGPULayers int, globalKVCacheType string, globalParallel int, resolvedModelOpts map[string]processmanager.ModelLoadOptions, logger *slog.Logger) *Server {
	return newServer(addr, reg, pm, est, defaultGPULayers, globalKVCacheType, globalParallel, resolvedModelOpts, logger)
}

// resolveOpts returns the ModelLoadOptions for the given model DisplayName.
// Models not in resolvedModelOpts fall back to the global KV cache type and parallel.
func (s *Server) resolveOpts(displayName string) processmanager.ModelLoadOptions {
	if opts, ok := s.resolvedModelOpts[displayName]; ok {
		return opts
	}
	return processmanager.ModelLoadOptions{KVCacheType: s.globalKVCacheType, Parallel: s.globalParallel}
}

// SetHistoryStore attaches a history store to the server, enabling persistent
// session history for chat completions. Must be called before ListenAndServe.
func (s *Server) SetHistoryStore(store history.Store) {
	s.historyStore = store
}

func newServer(addr string, reg modelRegistry, pm processManager, est resourceEstimator, defaultGPULayers int, globalKVCacheType string, globalParallel int, resolvedModelOpts map[string]processmanager.ModelLoadOptions, logger *slog.Logger) *Server {
	s := &Server{
		registry:          reg,
		procMgr:           pm,
		estimator:         est,
		defaultGPULayers:  defaultGPULayers,
		globalKVCacheType: globalKVCacheType,
		globalParallel:    globalParallel,
		resolvedModelOpts: resolvedModelOpts,
		logger:            logger,
		queryResources:    estimator.QueryResources,
		queryVRAMTotal:    estimator.QueryVRAMTotal,
		queryHardware:     estimator.QueryHardware,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", s.handleOAIModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleInferenceProxy)
	mux.HandleFunc("POST /v1/completions", s.handleInferenceProxy)
	mux.HandleFunc("POST /v1/messages", s.handleMessagesProxy)
	mux.HandleFunc("/v1/", notImplemented)
	mux.HandleFunc("GET /api/v1/status", s.handleStatus)
	mux.HandleFunc("GET /api/v1/hardware", s.handleHardware)
	mux.HandleFunc("GET /api/v1/models", s.handleListModels)
	mux.HandleFunc("GET /api/v1/models/{id}", s.handleGetModel)
	mux.HandleFunc("POST /api/v1/models/{id}/load", s.handleLoadModel)
	mux.HandleFunc("DELETE /api/v1/models/{id}", s.handleUnloadModel)
	mux.HandleFunc("GET /api/v1/models/{id}/estimate", s.handleEstimate)
	s.registerUI(mux)

	s.http = &http.Server{
		Addr:    addr,
		Handler: mux,
	}
	return s
}

func notImplemented(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
}

// ListenAndServe binds the listener and serves until the context is cancelled,
// then drains in-flight requests and returns.
func (s *Server) ListenAndServe(ctx context.Context) error {
	ln, err := net.Listen("tcp", s.http.Addr)
	if err != nil {
		return err
	}

	serveErr := make(chan error, 1)
	go func() {
		serveErr <- s.http.Serve(ln)
	}()

	select {
	case err := <-serveErr:
		return err
	case <-ctx.Done():
	}

	s.logger.Info("shutdown initiated, draining in-flight requests")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	if err := s.http.Shutdown(shutdownCtx); err != nil {
		return err
	}
	s.logger.Info("shutdown complete")
	return nil
}

// --- JSON response/request types ---

type apiError struct {
	Error string `json:"error"`
}

type estimateResponse struct {
	CostBytes      uint64 `json:"cost_bytes"`
	Feasible       bool   `json:"feasible"`
	AvailableBytes uint64 `json:"available_bytes"`
}

type modelDetailResponse struct {
	ID                 uint64           `json:"id"`
	DisplayName        string           `json:"display_name"`
	Path               string           `json:"path"`
	FileSize           int64            `json:"file_size"`
	Architecture       string           `json:"architecture"`
	LayerCount         uint32           `json:"layer_count"`
	KVHeadCount        uint32           `json:"kv_head_count"`
	HeadDim            uint32           `json:"head_dim"`
	MaxContext         uint32           `json:"max_context"`
	Quantization       string           `json:"quantization"`
	NativeEstimate     estimateResponse `json:"native_estimate"`
	MaxLoadableContext uint32           `json:"max_loadable_context"`
}

type loadedModelResponse struct {
	ModelID     uint64    `json:"model_id"`
	PID         int       `json:"pid"`
	Port        int       `json:"port"`
	ContextSize int       `json:"context_size"`
	GPULayers   int       `json:"gpu_layers"`
	LoadTime    time.Time `json:"load_time"`
	Health      string    `json:"health"`
}

type loadRequest struct {
	Ctx int `json:"ctx"`
}

type loadedModelInfo struct {
	ModelID            uint64 `json:"model_id"`
	Port               int    `json:"port"`
	DisplayName        string `json:"display_name"`
	ContextSize        int    `json:"context_size"`
	Health             string `json:"health"`
	EstimatedVRAMBytes uint64 `json:"estimated_vram_bytes"`
}

type memoryInfo struct {
	TotalVRAM     uint64 `json:"total_vram_bytes"`
	AvailableVRAM uint64 `json:"available_vram_bytes"`
	InUseVRAM     uint64 `json:"in_use_vram_bytes"`
}

type statusResponse struct {
	Status       string            `json:"status"`
	LoadedModels []loadedModelInfo `json:"loaded_models"`
	Memory       memoryInfo        `json:"memory"`
}

type gpuInfoResponse struct {
	Index              int    `json:"index"`
	Identity           string `json:"identity"`
	VRAMTotalBytes     uint64 `json:"vram_total_bytes"`
	VRAMUsedBytes      uint64 `json:"vram_used_bytes"`
	VRAMAvailableBytes uint64 `json:"vram_available_bytes"`
}

type hardwareResponse struct {
	GPUs                    []gpuInfoResponse `json:"gpus"`
	SystemRAMAvailableBytes uint64            `json:"system_ram_available_bytes"`
}

// --- helpers ---

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, apiError{Error: msg})
}

func parseModelID(w http.ResponseWriter, r *http.Request) (uint64, bool) {
	raw := r.PathValue("id")
	id, err := strconv.ParseUint(raw, 10, 64)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Sprintf("invalid model id %q: must be a positive integer", raw))
		return 0, false
	}
	return id, true
}

func modelSpec(m registry.Model) estimator.ModelSpec {
	return estimator.ModelSpec{
		FileSize:    m.FileSize,
		LayerCount:  m.LayerCount,
		KVHeadCount: m.KVHeadCount,
		HeadDim:     m.HeadDim,
		MaxContext:  m.MaxContext,

		SlidingWindowSize: m.SlidingWindowSize,
		SWAHeadDim:        m.SWAHeadDim,
		GlobalLayerCount:  m.GlobalLayerCount,
		SWALayerCount:     m.SWALayerCount,
		GlobalKVHeadCount: m.GlobalKVHeadCount,
		SWAKVHeadCount:    m.SWAKVHeadCount,
	}
}

func healthString(lm *processmanager.LoadedModel) string {
	if lm.Health() == processmanager.HealthStatusHealthy {
		return "healthy"
	}
	return "unavailable"
}

func loadedModelResp(lm *processmanager.LoadedModel) loadedModelResponse {
	return loadedModelResponse{
		ModelID:     lm.ModelID,
		PID:         lm.PID,
		Port:        lm.Port,
		ContextSize: lm.ContextSize,
		GPULayers:   lm.GPULayers,
		LoadTime:    lm.LoadTime,
		Health:      healthString(lm),
	}
}

// --- handlers ---

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	models := s.registry.List()
	entries := make([]lmsModelEntry, len(models))
	for i, m := range models {
		instances := make([]lmsLoadedInstance, 0)
		if lm, loaded := s.procMgr.Get(m.ID); loaded {
			instances = append(instances, lmsLoadedInstance{
				ID: m.DisplayName,
				Config: lmsInstanceConfig{
					ContextLength:  lm.ContextSize,
					EvalBatchSize:  512,
					FlashAttention: false,
					Parallel:       lm.Parallel,
				},
			})
		}
		entries[i] = lmsModelEntry{
			Key:           m.DisplayName,
			ID:            strconv.FormatUint(m.ID, 10),
			Type:          "llm",
			Publisher:     "",
			DisplayName:   m.DisplayName,
			Architecture:  m.Architecture,
			SizeBytes:     m.FileSize,
			ContextLength: m.MaxContext,
			Quantization: lmsQuantization{
				Name:          m.Quantization,
				BitsPerWeight: bitsPerWeight(m.Quantization),
			},
			LoadedInstances: instances,
		}
	}
	writeJSON(w, http.StatusOK, lmsModelsResponse{Models: entries})
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	id, ok := parseModelID(w, r)
	if !ok {
		return
	}

	m, found := s.registry.Get(id)
	if !found {
		writeError(w, http.StatusNotFound, fmt.Sprintf("model %d not found", id))
		return
	}

	opts := s.resolveOpts(m.DisplayName)
	spec := modelSpec(m)
	fwd, err := s.estimator.Forward(spec, m.MaxContext, 0, opts.KVCacheType, opts.Parallel)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource estimation failed: %v", err))
		return
	}

	inv, err := s.estimator.Inverse(spec, 0, opts.KVCacheType, opts.Parallel)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource estimation failed: %v", err))
		return
	}

	vramAvail, _, resErr := s.queryResources()
	if resErr != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource query failed: %v", resErr))
		return
	}

	resp := modelDetailResponse{
		ID:           m.ID,
		DisplayName:  m.DisplayName,
		Path:         m.Path,
		FileSize:     m.FileSize,
		Architecture: m.Architecture,
		LayerCount:   m.LayerCount,
		KVHeadCount:  m.KVHeadCount,
		HeadDim:      m.HeadDim,
		MaxContext:   m.MaxContext,
		Quantization: m.Quantization,
		NativeEstimate: estimateResponse{
			CostBytes:      fwd.TotalCost,
			Feasible:       fwd.Feasible,
			AvailableBytes: vramAvail,
		},
		MaxLoadableContext: inv.MaxContext,
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) handleLoadModel(w http.ResponseWriter, r *http.Request) {
	id, ok := parseModelID(w, r)
	if !ok {
		return
	}

	m, found := s.registry.Get(id)
	if !found {
		writeError(w, http.StatusNotFound, fmt.Sprintf("model %d not found", id))
		return
	}

	var req loadRequest
	if r.ContentLength != 0 {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("malformed request body: %v", err))
			return
		}
	}

	opts := s.resolveOpts(m.DisplayName)
	spec := modelSpec(m)

	ctxSize := req.Ctx
	if ctxSize <= 0 {
		inv, err := s.estimator.Inverse(spec, 0, opts.KVCacheType, opts.Parallel)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource estimation failed: %v", err))
			return
		}
		if inv.MaxContext == 0 {
			writeError(w, http.StatusUnprocessableEntity, "model does not fit in available memory at any context size")
			return
		}
		ctxSize = int(inv.MaxContext)
	} else {
		// Explicit ctx: clamp to native max; check feasibility.
		if m.MaxContext > 0 && ctxSize > int(m.MaxContext) {
			ctxSize = int(m.MaxContext)
		}
		fwd, err := s.estimator.Forward(spec, uint32(ctxSize), 0, opts.KVCacheType, opts.Parallel)
		if err != nil {
			writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource estimation failed: %v", err))
			return
		}
		if !fwd.Feasible {
			writeError(w, http.StatusUnprocessableEntity,
				fmt.Sprintf("model cannot fit in available memory at context size %d (estimated %d bytes required)", ctxSize, fwd.TotalCost))
			return
		}
	}

	lm, err := s.procMgr.Load(r.Context(), id, m.Path, m.MmprojPath, ctxSize, s.defaultGPULayers, opts)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("failed to load model: %v", err))
		return
	}

	writeJSON(w, http.StatusOK, loadedModelResp(lm))
}

func (s *Server) handleUnloadModel(w http.ResponseWriter, r *http.Request) {
	id, ok := parseModelID(w, r)
	if !ok {
		return
	}

	// Distinguish "unknown model" from "model not loaded".
	if _, found := s.registry.Get(id); !found {
		writeError(w, http.StatusNotFound, fmt.Sprintf("model %d not found", id))
		return
	}

	err := s.procMgr.Unload(r.Context(), id)
	if err != nil {
		// Model exists in registry but is not currently loaded (or mid-load/mid-unload).
		// HTTP 404: the "loaded model" resource does not exist.
		writeError(w, http.StatusNotFound, fmt.Sprintf("model %d is not loaded", id))
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) handleEstimate(w http.ResponseWriter, r *http.Request) {
	id, ok := parseModelID(w, r)
	if !ok {
		return
	}

	m, found := s.registry.Get(id)
	if !found {
		writeError(w, http.StatusNotFound, fmt.Sprintf("model %d not found", id))
		return
	}

	ctxStr := r.URL.Query().Get("ctx")
	if ctxStr == "" {
		writeError(w, http.StatusBadRequest, "ctx query parameter is required")
		return
	}
	ctxVal, err := strconv.ParseUint(ctxStr, 10, 32)
	if err != nil || ctxVal == 0 {
		writeError(w, http.StatusBadRequest, "ctx must be a positive integer")
		return
	}

	estOpts := s.resolveOpts(m.DisplayName)
	spec := modelSpec(m)
	fwd, err := s.estimator.Forward(spec, uint32(ctxVal), 0, estOpts.KVCacheType, estOpts.Parallel)
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource estimation failed: %v", err))
		return
	}

	vramAvail, _, resErr := s.queryResources()
	if resErr != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("resource query failed: %v", resErr))
		return
	}

	writeJSON(w, http.StatusOK, estimateResponse{
		CostBytes:      fwd.TotalCost,
		Feasible:       fwd.Feasible,
		AvailableBytes: vramAvail,
	})
}

func (s *Server) handleHardware(w http.ResponseWriter, r *http.Request) {
	gpus, ramAvail, err := s.queryHardware()
	if err != nil {
		writeError(w, http.StatusInternalServerError, fmt.Sprintf("hardware query failed: %v", err))
		return
	}

	entries := make([]gpuInfoResponse, 0, len(gpus))
	for _, g := range gpus {
		entries = append(entries, gpuInfoResponse{
			Index:              g.Index,
			Identity:           g.Identity,
			VRAMTotalBytes:     g.VRAMTotal,
			VRAMUsedBytes:      g.VRAMUsed,
			VRAMAvailableBytes: g.VRAMAvail,
		})
	}

	writeJSON(w, http.StatusOK, hardwareResponse{
		GPUs:                    entries,
		SystemRAMAvailableBytes: ramAvail,
	})
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	loaded := s.procMgr.List()

	infos := make([]loadedModelInfo, len(loaded))
	for i, lm := range loaded {
		info := loadedModelInfo{
			ModelID:     lm.ModelID,
			Port:        lm.Port,
			ContextSize: lm.ContextSize,
			Health:      healthString(lm),
		}
		// Enrich with display name and an estimated VRAM cost when the model is
		// still in the registry. Estimation is best-effort: a lookup miss or
		// estimator error leaves the defaults (empty name, zero estimate) rather
		// than failing the status response.
		if m, found := s.registry.Get(lm.ModelID); found {
			info.DisplayName = m.DisplayName
			opts := s.resolveOpts(m.DisplayName)
			if fwd, err := s.estimator.Forward(modelSpec(m), uint32(lm.ContextSize), 0, opts.KVCacheType, opts.Parallel); err == nil {
				info.EstimatedVRAMBytes = fwd.TotalCost
			}
		}
		infos[i] = info
	}

	vramTotal, _ := s.queryVRAMTotal()
	vramAvail, _, _ := s.queryResources()

	var inUse uint64
	if vramTotal > vramAvail {
		inUse = vramTotal - vramAvail
	}

	writeJSON(w, http.StatusOK, statusResponse{
		Status:       "ok",
		LoadedModels: infos,
		Memory: memoryInfo{
			TotalVRAM:     vramTotal,
			AvailableVRAM: vramAvail,
			InUseVRAM:     inUse,
		},
	})
}
