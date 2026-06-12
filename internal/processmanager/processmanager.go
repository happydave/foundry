package processmanager

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// sigTermTimeout is the duration the process manager waits for a subprocess to exit
// after SIGTERM before escalating to SIGKILL. Declared as a var so tests can shorten it.
var sigTermTimeout = 10 * time.Second

const (
	healthPollInterval = 500 * time.Millisecond
	healthPollTimeout  = 120 * time.Second
)

// HealthStatus describes the current health of a loaded model's subprocess.
type HealthStatus int

const (
	HealthStatusHealthy     HealthStatus = iota
	HealthStatusUnavailable              // subprocess exited unexpectedly
)

// LoadedModel is the runtime record of a successfully loaded model. All fields except
// Health are immutable after creation.
type LoadedModel struct {
	ModelID     uint64
	PID         int
	Port        int
	ContextSize int
	GPULayers   int
	Parallel    int
	LoadTime    time.Time

	mu     sync.Mutex
	health HealthStatus
}

// Health returns the current health status. Safe for concurrent use.
func (r *LoadedModel) Health() HealthStatus {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.health
}

func (r *LoadedModel) setHealth(s HealthStatus) {
	r.mu.Lock()
	r.health = s
	r.mu.Unlock()
}

type entryKind int

const (
	kindLoading entryKind = iota
	kindLoaded
	kindUnloading
	kindFailed // terminal until replaced by a new load attempt
)

type entry struct {
	kind   entryKind
	done   chan struct{} // closed when current operation (load/unload) completes
	record *LoadedModel  // non-nil when kindLoaded or kindUnloading
	cmd    *exec.Cmd     // non-nil when kindLoaded or kindUnloading
	ph     *procHandle   // non-nil when kindLoaded or kindUnloading
	err    error         // set when kind == kindFailed
}

// procHandle owns the subprocess wait. cmd.Wait() is delegated to a goroutine; results
// are signalled via waitDone. Multiple goroutines may wait on waitDone safely.
type procHandle struct {
	waitDone chan struct{} // closed after cmd.Wait() returns and I/O goroutines finish
	waitErr  error         // set before waitDone is closed
}

// ModelLoadOptions holds per-model options for a single Manager.Load call.
// Replacing the former perModelArgs []string parameter, this struct is the
// extension point for future per-model load configuration.
type ModelLoadOptions struct {
	// Args are appended to the subprocess command line after the cache-type
	// flags and before the manager's global extra args (e.g. --chat-template-file).
	Args []string
	// KVCacheType is the resolved KV cache element type for this model
	// (e.g. "f16", "q8_0"). If empty, doLoad treats it as "q8_0".
	KVCacheType string
	// Parallel is the number of parallel KV cache slots to allocate. If 0,
	// doLoad defaults to 1. Must be >= 1 after defaulting.
	Parallel int
}

// CheckBinaryVersion runs binary --version, captures the combined stdout+stderr
// output, and checks whether the output contains any entry in allowedVersions as
// a substring. The process exit code is ignored — some llama.cpp builds exit
// non-zero for --version while still emitting recognisable output.
//
// Returns an error if the binary cannot be executed, if the output is empty, or
// if none of the allowedVersions strings appear in the output.
func CheckBinaryVersion(binary string, allowedVersions []string) error {
	return checkBinaryVersionWithCmd(binary, allowedVersions, exec.Command)
}

// checkBinaryVersionWithCmd is the injectable implementation used by
// CheckBinaryVersion and tests.
func checkBinaryVersionWithCmd(binary string, allowedVersions []string, newCmd func(string, ...string) *exec.Cmd) error {
	cmd := newCmd(binary, "--version")
	out, err := cmd.CombinedOutput()
	if err != nil && len(out) == 0 {
		return fmt.Errorf("failed to run %q --version: %w", binary, err)
	}
	combined := string(out)
	if strings.TrimSpace(combined) == "" {
		return fmt.Errorf("llama-server --version produced no output; cannot verify version")
	}
	for _, v := range allowedVersions {
		if strings.Contains(combined, v) {
			return nil
		}
	}
	return fmt.Errorf("llama-server version not recognised: got %q; known-good versions: %v",
		strings.TrimSpace(combined), allowedVersions)
}

// Manager launches and terminates llama-server subprocesses, one per loaded model.
// It serialises concurrent load requests for the same model and isolates subprocess
// crashes so they do not affect Foundry or other loaded models.
//
// llama-server flags assumed (verify against the binary in use):
//
//	--model <path>              model file path
//	--ctx-size <n>              context window size
//	--n-gpu-layers <n>          number of layers to offload to GPU
//	--port <n>                  TCP port to listen on
//	--host 127.0.0.1            bind to loopback only
//	--cache-type-k <type>       KV cache key element type
//	--cache-type-v <type>       KV cache value element type
//
// Health endpoint assumed: GET /health returns 200 OK when ready for inference.
type Manager struct {
	binary    string
	extraArgs []string
	logger    *slog.Logger

	// newCmd creates exec.Cmd instances; injectable for testing.
	newCmd func(name string, arg ...string) *exec.Cmd

	mu      sync.Mutex
	models  map[uint64]*entry
	closing bool
}

// New creates a Manager that launches subprocesses using the given binary path.
// extraArgs are appended verbatim to every subprocess invocation after the
// standard flags; pass nil or an empty slice for no extra args.
func New(binary string, extraArgs []string, logger *slog.Logger) *Manager {
	return &Manager{
		binary:    binary,
		extraArgs: extraArgs,
		logger:    logger,
		models:    make(map[uint64]*entry),
		newCmd:    exec.Command,
	}
}

// Load launches a llama-server subprocess for the model, polls its health endpoint
// until ready, and returns the loaded model record. opts carries per-model options
// including the resolved KV cache type and any extra args. Behaviour for concurrent
// calls:
//   - Already loaded: returns the existing record immediately (no new subprocess).
//   - Load in progress: blocks until complete and returns the same outcome.
//   - Load failed previously: starts a fresh load attempt.
func (m *Manager) Load(ctx context.Context, modelID uint64, modelPath, mmprojPath string, contextSize, gpuLayers int, opts ModelLoadOptions) (*LoadedModel, error) {
	m.mu.Lock()

	if m.closing {
		m.mu.Unlock()
		return nil, fmt.Errorf("foundry is shutting down")
	}

	if e, ok := m.models[modelID]; ok {
		switch e.kind {
		case kindLoaded:
			r := e.record
			m.mu.Unlock()
			return r, nil

		case kindLoading, kindUnloading:
			done := e.done
			m.mu.Unlock()
			select {
			case <-done:
			case <-ctx.Done():
				return nil, ctx.Err()
			}
			// Re-enter after the in-progress operation completes.
			return m.Load(ctx, modelID, modelPath, mmprojPath, contextSize, gpuLayers, opts)

		case kindFailed:
			// Previous attempt failed; allow retry by falling through.
		}
	}

	// No entry, or kindFailed: register a new loading entry.
	e := &entry{kind: kindLoading, done: make(chan struct{})}
	m.models[modelID] = e
	m.mu.Unlock()

	record, cmd, ph, loadErr := m.doLoad(ctx, modelID, modelPath, mmprojPath, contextSize, gpuLayers, opts)

	m.mu.Lock()
	if loadErr != nil {
		e.kind = kindFailed
		e.err = loadErr
	} else {
		e.kind = kindLoaded
		e.record = record
		e.cmd = cmd
		e.ph = ph
	}
	close(e.done)
	m.mu.Unlock()

	if loadErr != nil {
		return nil, loadErr
	}

	go m.monitor(modelID, record, ph)
	return record, nil
}

// doLoad performs the actual subprocess launch, I/O wiring, and health polling.
func (m *Manager) doLoad(ctx context.Context, modelID uint64, modelPath, mmprojPath string, contextSize, gpuLayers int, opts ModelLoadOptions) (*LoadedModel, *exec.Cmd, *procHandle, error) {
	port, err := freePort()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("model %d: no free port: %w", modelID, err)
	}

	kvType := opts.KVCacheType
	if kvType == "" {
		kvType = "q8_0"
	}
	parallel := opts.Parallel
	if parallel == 0 {
		parallel = 1
	}

	args := []string{
		"--model", modelPath,
		"--ctx-size", strconv.Itoa(contextSize),
		"--n-gpu-layers", strconv.Itoa(gpuLayers),
		"--port", strconv.Itoa(port),
		"--host", "127.0.0.1",
		"--cache-type-k", kvType,
		"--cache-type-v", kvType,
		"--parallel", strconv.Itoa(parallel),
	}
	if mmprojPath != "" {
		args = append(args, "--mmproj", mmprojPath)
	}
	args = append(args, opts.Args...)
	args = append(args, m.extraArgs...)
	cmd := m.newCmd(m.binary, args...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("model %d: stdout pipe: %w", modelID, err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("model %d: stderr pipe: %w", modelID, err)
	}

	if err := cmd.Start(); err != nil {
		return nil, nil, nil, fmt.Errorf("model %d: start subprocess: %w", modelID, err)
	}

	ph := &procHandle{waitDone: make(chan struct{})}

	var ioWg sync.WaitGroup
	ioWg.Add(2)
	go func() {
		defer ioWg.Done()
		s := bufio.NewScanner(stdout)
		for s.Scan() {
			m.logger.Info(s.Text(), slog.Uint64("model_id", modelID))
		}
	}()
	go func() {
		defer ioWg.Done()
		s := bufio.NewScanner(stderr)
		for s.Scan() {
			m.logger.Warn(s.Text(), slog.Uint64("model_id", modelID))
		}
	}()

	go func() {
		ph.waitErr = cmd.Wait()
		ioWg.Wait()
		close(ph.waitDone)
	}()

	if err := waitHealthy(ctx, port, ph); err != nil {
		_ = cmd.Process.Kill()
		<-ph.waitDone
		return nil, nil, nil, fmt.Errorf("model %d: %w", modelID, err)
	}

	record := &LoadedModel{
		ModelID:     modelID,
		PID:         cmd.Process.Pid,
		Port:        port,
		ContextSize: contextSize,
		GPULayers:   gpuLayers,
		Parallel:    parallel,
		LoadTime:    time.Now(),
		health:      HealthStatusHealthy,
	}

	m.logger.Info("model loaded",
		slog.Uint64("model_id", modelID),
		slog.Int("pid", record.PID),
		slog.Int("port", port),
	)

	return record, cmd, ph, nil
}

// Unload sends SIGTERM to the subprocess and waits up to sigTermTimeout (10 s) for it
// to exit, then escalates to SIGKILL. Returns an error if the model is not loaded.
// Two concurrent unload requests for the same model are serialised: only one SIGTERM
// is sent; the second caller receives an error because the model is no longer loaded.
func (m *Manager) Unload(ctx context.Context, modelID uint64) error {
	m.mu.Lock()

	e, ok := m.models[modelID]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("model %d is not loaded", modelID)
	}

	switch e.kind {
	case kindLoading:
		m.mu.Unlock()
		return fmt.Errorf("model %d is still loading", modelID)

	case kindFailed:
		m.mu.Unlock()
		return fmt.Errorf("model %d is not loaded", modelID)

	case kindUnloading:
		// Another unload is in progress; wait for it, then report not loaded.
		done := e.done
		m.mu.Unlock()
		select {
		case <-done:
		case <-ctx.Done():
			return ctx.Err()
		}
		return fmt.Errorf("model %d is not loaded", modelID)

	case kindLoaded:
		e.kind = kindUnloading
		e.done = make(chan struct{}) // new channel for unload completion; load's done is already closed
		m.mu.Unlock()
	}

	_ = e.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-e.ph.waitDone:
	case <-time.After(sigTermTimeout):
		m.logger.Warn("SIGTERM timeout elapsed, escalating to SIGKILL",
			slog.Uint64("model_id", modelID),
		)
		_ = e.cmd.Process.Kill()
		<-e.ph.waitDone
	}

	m.mu.Lock()
	delete(m.models, modelID)
	close(e.done)
	m.mu.Unlock()

	m.logger.Info("model unloaded", slog.Uint64("model_id", modelID))
	return nil
}

// UnloadAll unloads all currently loaded models. Intended for graceful shutdown.
// Sets the closing flag so subsequent Load calls return an error. In-progress loads
// (kindLoading) are waited on before this function returns — if the caller passes a
// cancellable context derived from the shutdown signal, those loads will abort quickly
// via context cancellation and their subprocesses will be cleaned up by doLoad.
func (m *Manager) UnloadAll(ctx context.Context) error {
	m.mu.Lock()
	m.closing = true
	ids := make([]uint64, 0, len(m.models))
	var loadingDones []chan struct{}
	for id, e := range m.models {
		switch e.kind {
		case kindLoaded:
			ids = append(ids, id)
		case kindLoading:
			loadingDones = append(loadingDones, e.done)
		}
	}
	m.mu.Unlock()

	// Wait for any in-progress loads to settle before unloading.
	for _, done := range loadingDones {
		select {
		case <-done:
		case <-ctx.Done():
		}
	}

	var errs []error
	for _, id := range ids {
		if err := m.Unload(ctx, id); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// List returns a snapshot of all currently loaded model records.
func (m *Manager) List() []*LoadedModel {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*LoadedModel, 0, len(m.models))
	for _, e := range m.models {
		if e.kind == kindLoaded {
			out = append(out, e.record)
		}
	}
	return out
}

// Get returns the loaded model record for the given model ID, or false if not loaded.
func (m *Manager) Get(modelID uint64) (*LoadedModel, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.models[modelID]
	if !ok || e.kind != kindLoaded {
		return nil, false
	}
	return e.record, true
}

// monitor waits for the subprocess to exit and, if the exit is unexpected (not due to
// an Unload call), marks the model unavailable and removes it from the registry.
func (m *Manager) monitor(modelID uint64, record *LoadedModel, ph *procHandle) {
	<-ph.waitDone

	m.mu.Lock()
	e, ok := m.models[modelID]
	expectedExit := ok && e.kind == kindUnloading
	m.mu.Unlock()

	if !expectedExit {
		record.setHealth(HealthStatusUnavailable)
		m.logger.Warn("subprocess exited unexpectedly",
			slog.Uint64("model_id", modelID),
			slog.Any("error", ph.waitErr),
		)
		m.mu.Lock()
		if e2, ok2 := m.models[modelID]; ok2 && e2.kind == kindLoaded {
			delete(m.models, modelID)
		}
		m.mu.Unlock()
	}
}

// waitHealthy polls GET /health on 127.0.0.1:<port> at 500 ms intervals until the
// server responds 200 OK, the subprocess exits, the context is cancelled, or
// healthPollTimeout (120 s) elapses.
func waitHealthy(ctx context.Context, port int, ph *procHandle) error {
	client := &http.Client{Timeout: 2 * time.Second}
	url := fmt.Sprintf("http://127.0.0.1:%d/health", port)
	deadline := time.Now().Add(healthPollTimeout)

	for {
		select {
		case <-ph.waitDone:
			return fmt.Errorf("subprocess exited before becoming healthy")
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		if time.Now().After(deadline) {
			return fmt.Errorf("health check timed out after %s", healthPollTimeout)
		}

		if resp, err := client.Get(url); err == nil {
			_, _ = io.Copy(io.Discard, resp.Body)
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		select {
		case <-ph.waitDone:
			return fmt.Errorf("subprocess exited before becoming healthy")
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(healthPollInterval):
		}
	}
}

// freePort asks the OS for a free TCP port on localhost by binding to :0 and immediately
// releasing it. There is a brief TOCTOU window between releasing the port and passing it
// to llama-server; this is acceptable because llama-server requires an explicit port
// argument and no practical alternative avoids this window.
func freePort() (int, error) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	port := ln.Addr().(*net.TCPAddr).Port
	_ = ln.Close()
	return port, nil
}
