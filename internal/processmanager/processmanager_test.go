package processmanager

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"testing"
	"time"
)

// helperEnvKey is the env var that activates fake-server mode in the test binary.
const helperEnvKey = "FOUNDRY_TEST_HELPER"

// TestMain allows this test binary to act as a fake llama-server subprocess.
// When FOUNDRY_TEST_HELPER is set, the binary runs as a helper and exits; otherwise
// it runs the normal test suite.
func TestMain(m *testing.M) {
	if mode := os.Getenv(helperEnvKey); mode != "" {
		runHelper(mode)
		return // runHelper calls os.Exit
	}
	os.Exit(m.Run())
}

// runHelper implements fake llama-server behaviour for tests.
// It parses --port from os.Args (matching llama-server flag conventions).
func runHelper(mode string) {
	port := ""
	for i, arg := range os.Args {
		if arg == "--port" && i+1 < len(os.Args) {
			port = os.Args[i+1]
		}
	}

	switch mode {
	case "crash":
		os.Exit(1)

	case "healthy":
		mux := http.NewServeMux()
		mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		_, _ = fmt.Fprintln(os.Stdout, "fake-server: ready")
		_, _ = fmt.Fprintln(os.Stderr, "fake-server: stderr line")
		srv := &http.Server{Addr: "127.0.0.1:" + port, Handler: mux}
		stop := make(chan os.Signal, 1)
		signal.Notify(stop, syscall.SIGTERM)
		go func() {
			<-stop
			_ = srv.Close()
		}()
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			os.Exit(1)
		}
		os.Exit(0)

	case "print-version":
		fmt.Println("version: 9536 (308f61c31)")
		fmt.Println("built with GNU 11.4.0 for Linux x86_64")
		os.Exit(0)

	case "hang":
		// Starts a health server but ignores SIGTERM to exercise SIGKILL escalation.
		signal.Ignore(syscall.SIGTERM)
		mux := http.NewServeMux()
		mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusOK)
		})
		srv := &http.Server{Addr: "127.0.0.1:" + port, Handler: mux}
		_, _ = fmt.Fprintln(os.Stdout, "fake-server: ready (hang mode)")
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			os.Exit(1)
		}
		select {} // block forever after server closes

	default:
		fmt.Fprintf(os.Stderr, "unknown helper mode: %s\n", mode)
		os.Exit(1)
	}
}

// newTestManager builds a Manager whose subprocess is the current test binary
// running in the given helper mode.
func newTestManager(t *testing.T, mode string) *Manager {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], nil, logger)
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"="+mode)
		return cmd
	}
	return m
}

func TestLoad_Success(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("Load: unexpected error: %v", err)
	}
	if rec.ModelID != 1 {
		t.Errorf("ModelID = %d, want 1", rec.ModelID)
	}
	if rec.Port == 0 {
		t.Error("Port is 0")
	}
	if rec.PID == 0 {
		t.Error("PID is 0")
	}
	if rec.ContextSize != 4096 {
		t.Errorf("ContextSize = %d, want 4096", rec.ContextSize)
	}
	if rec.GPULayers != 32 {
		t.Errorf("GPULayers = %d, want 32", rec.GPULayers)
	}
	if rec.Health() != HealthStatusHealthy {
		t.Error("Health is not HealthStatusHealthy after load")
	}

	listed := m.List()
	if len(listed) != 1 {
		t.Fatalf("List() len = %d, want 1", len(listed))
	}
	got, ok := m.Get(1)
	if !ok || got != rec {
		t.Error("Get(1) did not return the loaded record")
	}

	if err := m.Unload(ctx, 1); err != nil {
		t.Errorf("Unload: %v", err)
	}
}

func TestLoad_AlreadyLoaded_ReturnsSameRecord(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	rec1, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("first Load: %v", err)
	}
	rec2, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("second Load: %v", err)
	}
	if rec1 != rec2 {
		t.Error("second Load returned a different record; expected same pointer")
	}

	_ = m.Unload(ctx, 1)
}

func TestLoad_Failure_SubprocessCrash(t *testing.T) {
	m := newTestManager(t, "crash")
	ctx := context.Background()

	_, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err == nil {
		t.Fatal("Load: expected error for crashing subprocess, got nil")
	}

	if _, ok := m.Get(1); ok {
		t.Error("Get returned true after failed load; model should not be in registry")
	}
}

func TestLoad_ConcurrentSameModel_OneSubprocess(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	const goroutines = 5
	results := make([]*LoadedModel, goroutines)
	errs := make([]error, goroutines)

	var ready sync.WaitGroup
	start := make(chan struct{})
	ready.Add(goroutines)
	var done sync.WaitGroup
	done.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		i := i
		go func() {
			defer done.Done()
			ready.Done()
			<-start
			results[i], errs[i] = m.Load(ctx, 42, "/fake/model.gguf", "", 2048, 16, ModelLoadOptions{})
		}()
	}

	ready.Wait()
	close(start)
	done.Wait()

	for i, err := range errs {
		if err != nil {
			t.Errorf("goroutine %d Load error: %v", i, err)
		}
	}
	for i := 1; i < goroutines; i++ {
		if results[i] != results[0] {
			t.Errorf("goroutine %d got different record than goroutine 0", i)
		}
	}
	listed := m.List()
	if len(listed) != 1 {
		t.Errorf("List() len = %d after concurrent loads, want 1", len(listed))
	}

	_ = m.Unload(ctx, 42)
}

func TestUnload_Clean(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	port := rec.Port

	if err := m.Unload(ctx, 1); err != nil {
		t.Fatalf("Unload: %v", err)
	}

	if _, ok := m.Get(1); ok {
		t.Error("Get returned true after Unload")
	}

	// Port should now be free.
	ln, err := net.Listen("tcp", "127.0.0.1:"+strconv.Itoa(port))
	if err != nil {
		t.Errorf("port %d not freed after Unload: %v", port, err)
	} else {
		_ = ln.Close()
	}
}

func TestUnload_NotLoaded_ReturnsError(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	if err := m.Unload(ctx, 99); err == nil {
		t.Fatal("Unload of non-loaded model: expected error, got nil")
	}
}

func TestUnload_SIGKILLEscalation(t *testing.T) {
	m := newTestManager(t, "hang")
	ctx := context.Background()

	if _, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{}); err != nil {
		t.Fatalf("Load: %v", err)
	}

	// Shorten timeout so the test completes quickly.
	orig := sigTermTimeout
	sigTermTimeout = 250 * time.Millisecond
	defer func() { sigTermTimeout = orig }()

	start := time.Now()
	if err := m.Unload(ctx, 1); err != nil {
		t.Fatalf("Unload: %v", err)
	}
	elapsed := time.Since(start)

	if elapsed > 5*time.Second {
		t.Errorf("Unload took %v; expected SIGKILL escalation within a second", elapsed)
	}
}

func TestCrash_MarksModelUnavailable(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	p, err := os.FindProcess(rec.PID)
	if err != nil {
		t.Fatalf("FindProcess(%d): %v", rec.PID, err)
	}
	if err := p.Signal(syscall.SIGKILL); err != nil {
		t.Fatalf("kill subprocess: %v", err)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		if rec.Health() == HealthStatusUnavailable {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if rec.Health() != HealthStatusUnavailable {
		t.Error("Health is not HealthStatusUnavailable after crash")
	}

	deadline = time.Now().Add(1 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := m.Get(1); !ok {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	if _, ok := m.Get(1); ok {
		t.Error("Get returned true after crash; model should have been removed from registry")
	}
}

func TestShutdown_RejectsNewLoad(t *testing.T) {
	m := newTestManager(t, "healthy")
	ctx := context.Background()

	_ = m.UnloadAll(ctx)

	if _, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{}); err == nil {
		t.Fatal("Load after UnloadAll: expected error, got nil")
	}
}

func TestLogCapture_SubprocessOutputForwarded(t *testing.T) {
	var mu sync.Mutex
	var logged []string
	w := &captureWriter{fn: func(s string) {
		mu.Lock()
		logged = append(logged, s)
		mu.Unlock()
	}}
	logger := slog.New(slog.NewJSONHandler(w, nil))

	m := New(os.Args[0], nil, logger)
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}
	ctx := context.Background()

	if _, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{}); err != nil {
		t.Fatalf("Load: %v", err)
	}
	time.Sleep(100 * time.Millisecond)
	_ = m.Unload(ctx, 1)

	mu.Lock()
	n := len(logged)
	mu.Unlock()
	if n == 0 {
		t.Error("no log entries captured from subprocess output")
	}
}

func TestLoad_ExtraArgsAppended(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], []string{"--custom-flag", "testval"}, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs, args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("Load: unexpected error: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	found := false
	for i, arg := range capturedArgs {
		if arg == "--custom-flag" && i+1 < len(capturedArgs) && capturedArgs[i+1] == "testval" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("extra args not found in subprocess args: %v", capturedArgs)
	}
}

func TestLoad_PerModelArgsOrdering(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], []string{"--global-flag"}, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs[:0], args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	opts := ModelLoadOptions{Args: []string{"--chat-template-file", "/tmpl.jinja"}}
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, opts)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	// Find positions of the per-model flag and the global flag.
	perModelIdx := -1
	globalIdx := -1
	for i, arg := range capturedArgs {
		if arg == "--chat-template-file" {
			perModelIdx = i
		}
		if arg == "--global-flag" {
			globalIdx = i
		}
	}
	if perModelIdx == -1 {
		t.Fatal("--chat-template-file not found in subprocess args")
	}
	if globalIdx == -1 {
		t.Fatal("--global-flag not found in subprocess args")
	}
	if perModelIdx > globalIdx {
		t.Errorf("per-model arg (idx %d) must appear before global extra arg (idx %d)", perModelIdx, globalIdx)
	}
}

func TestLoad_CacheTypeFlags(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], nil, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs[:0], args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	opts := ModelLoadOptions{KVCacheType: "q8_0"}
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, opts)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	findFlag := func(flag, value string) bool {
		for i, arg := range capturedArgs {
			if arg == flag && i+1 < len(capturedArgs) && capturedArgs[i+1] == value {
				return true
			}
		}
		return false
	}
	if !findFlag("--cache-type-k", "q8_0") {
		t.Errorf("--cache-type-k q8_0 not found in subprocess args: %v", capturedArgs)
	}
	if !findFlag("--cache-type-v", "q8_0") {
		t.Errorf("--cache-type-v q8_0 not found in subprocess args: %v", capturedArgs)
	}
}

func TestLoad_CacheTypeFlags_EmptyOptsDefaultsToQ8(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], nil, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs[:0], args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	findFlag := func(flag, value string) bool {
		for i, arg := range capturedArgs {
			if arg == flag && i+1 < len(capturedArgs) && capturedArgs[i+1] == value {
				return true
			}
		}
		return false
	}
	if !findFlag("--cache-type-k", "q8_0") {
		t.Errorf("--cache-type-k q8_0 not found when KVCacheType is empty: %v", capturedArgs)
	}
	if !findFlag("--cache-type-v", "q8_0") {
		t.Errorf("--cache-type-v q8_0 not found when KVCacheType is empty: %v", capturedArgs)
	}
}

// newVersionCheckCmd returns a command factory that runs the test binary in
// "print-version" helper mode, simulating a llama-server --version invocation.
func newVersionCheckCmd(t *testing.T) func(string, ...string) *exec.Cmd {
	t.Helper()
	return func(_ string, _ ...string) *exec.Cmd {
		cmd := exec.Command(os.Args[0])
		cmd.Env = append(os.Environ(), helperEnvKey+"=print-version")
		return cmd
	}
}

func TestCheckBinaryVersion_KnownVersion(t *testing.T) {
	// The print-version helper emits "version: 9536 (308f61c31)" — verify match.
	err := checkBinaryVersionWithCmd(os.Args[0], []string{"version: 9536 (308f61c31)"},
		newVersionCheckCmd(t))
	if err != nil {
		t.Errorf("expected nil for matching version, got: %v", err)
	}
}

func TestCheckBinaryVersion_UnknownVersion(t *testing.T) {
	err := checkBinaryVersionWithCmd(os.Args[0], []string{"version: 9999 (unknown)"},
		newVersionCheckCmd(t))
	if err == nil {
		t.Error("expected error for non-matching version, got nil")
	}
}

func TestCheckBinaryVersion_BinaryNotFound(t *testing.T) {
	err := CheckBinaryVersion("/nonexistent/binary", []string{"anything"})
	if err == nil {
		t.Error("expected error for missing binary, got nil")
	}
}

func TestLoad_ParallelFlag_ExplicitValue(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], nil, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs[:0], args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	opts := ModelLoadOptions{Parallel: 4}
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, opts)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	findFlag := func(flag, value string) bool {
		for i, arg := range capturedArgs {
			if arg == flag && i+1 < len(capturedArgs) && capturedArgs[i+1] == value {
				return true
			}
		}
		return false
	}
	if !findFlag("--parallel", "4") {
		t.Errorf("--parallel 4 not found in subprocess args: %v", capturedArgs)
	}
	if rec.Parallel != 4 {
		t.Errorf("LoadedModel.Parallel = %d, want 4", rec.Parallel)
	}
}

func TestLoad_ParallelFlag_ZeroDefaultsToOne(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], nil, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs[:0], args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, ModelLoadOptions{})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	findFlag := func(flag, value string) bool {
		for i, arg := range capturedArgs {
			if arg == flag && i+1 < len(capturedArgs) && capturedArgs[i+1] == value {
				return true
			}
		}
		return false
	}
	if !findFlag("--parallel", "1") {
		t.Errorf("--parallel 1 not found when Parallel is 0 (default): %v", capturedArgs)
	}
	if rec.Parallel != 1 {
		t.Errorf("LoadedModel.Parallel = %d, want 1 (default)", rec.Parallel)
	}
}

func TestLoad_ParallelFlag_BeforePerModelAndGlobalArgs(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	m := New(os.Args[0], []string{"--global-flag"}, logger)

	var capturedArgs []string
	m.newCmd = func(_ string, args ...string) *exec.Cmd {
		capturedArgs = append(capturedArgs[:0], args...)
		cmd := exec.Command(os.Args[0], args...)
		cmd.Env = append(os.Environ(), helperEnvKey+"=healthy")
		return cmd
	}

	ctx := context.Background()
	opts := ModelLoadOptions{
		Parallel: 2,
		Args:     []string{"--chat-template-file", "/tmpl.jinja"},
	}
	rec, err := m.Load(ctx, 1, "/fake/model.gguf", "", 4096, 32, opts)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	_ = m.Unload(ctx, rec.ModelID)

	parallelIdx, perModelIdx, globalIdx := -1, -1, -1
	for i, arg := range capturedArgs {
		if arg == "--parallel" {
			parallelIdx = i
		}
		if arg == "--chat-template-file" {
			perModelIdx = i
		}
		if arg == "--global-flag" {
			globalIdx = i
		}
	}
	if parallelIdx == -1 {
		t.Fatal("--parallel not found in subprocess args")
	}
	if perModelIdx == -1 {
		t.Fatal("--chat-template-file not found in subprocess args")
	}
	if globalIdx == -1 {
		t.Fatal("--global-flag not found in subprocess args")
	}
	if parallelIdx > perModelIdx {
		t.Errorf("--parallel (idx %d) must appear before per-model arg (idx %d)", parallelIdx, perModelIdx)
	}
	if parallelIdx > globalIdx {
		t.Errorf("--parallel (idx %d) must appear before global extra arg (idx %d)", parallelIdx, globalIdx)
	}
}

type captureWriter struct{ fn func(string) }

func (w *captureWriter) Write(p []byte) (int, error) {
	w.fn(string(p))
	return len(p), nil
}
