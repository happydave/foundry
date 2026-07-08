package server

import (
	"io"
	"log/slog"
	"net/http"
	"testing"
	"time"

	"github.com/happydave/foundry/internal/estimator"
	"github.com/happydave/foundry/internal/processmanager"
	"github.com/happydave/foundry/internal/registry"
)

// seedLoaded registers a model in the fake registry and marks it loaded in the
// fake process manager with the given PID.
func seedLoaded(f *serverFixture, id uint64, name string, pid, ctxSize int) {
	f.reg.models = append(f.reg.models, registry.Model{ID: id, DisplayName: name})
	f.pm.loaded[id] = &processmanager.LoadedModel{
		ModelID:     id,
		PID:         pid,
		Port:        9000,
		ContextSize: ctxSize,
		LoadTime:    time.Now(),
	}
}

func u64(v uint64) *uint64 { return &v }

func TestHardware_AttributionAndPools(t *testing.T) {
	f := newFixture(t)
	seedLoaded(f, 7, "my-model", 4242, 8192)

	f.srv.queryHardware = func() ([]estimator.GPUInfo, uint64, error) {
		return []estimator.GPUInfo{{
			Index: 0, Identity: "gfx1151",
			VRAMTotal: 100 << 30, VRAMUsed: 10 << 30, VRAMAvail: 90 << 30,
			Pools:     estimator.GPUPools{GTTTotal: 16 << 30, GTTUsed: 2 << 30, VisVRAMTotal: 100 << 30, VisVRAMUsed: 10 << 30, PreemptUsed: 0},
			Telemetry: estimator.GPUTelemetry{BusyPercent: u64(42), TemperatureMilliC: u64(30000), PowerMicroW: u64(5085000), ClockMHz: u64(600)},
		}}, 32 << 30, nil
	}
	f.srv.queryProcessGPUMem = func(pids []int) ([]estimator.ProcessGPUMemory, error) {
		return []estimator.ProcessGPUMemory{{
			PID:   4242,
			Cards: []estimator.ProcessCardMemory{{CardIndex: 0, VRAMBytes: 6 << 30, GTTBytes: 1 << 30}},
		}}, nil
	}

	resp := f.do(t, http.MethodGet, "/api/v1/hardware", nil)
	assertStatus(t, resp, http.StatusOK)
	hw := decodeBody[hardwareResponse](t, resp)

	g := hw.GPUs[0]
	// Pools mapped.
	if g.Pools.GTTTotalBytes != 16<<30 || g.Pools.GTTUsedBytes != 2<<30 {
		t.Errorf("pools: %+v", g.Pools)
	}
	// Telemetry mapped.
	if g.Telemetry.BusyPercent == nil || *g.Telemetry.BusyPercent != 42 ||
		g.Telemetry.TemperatureMilliC == nil || *g.Telemetry.TemperatureMilliC != 30000 ||
		g.Telemetry.ClockMHz == nil || *g.Telemetry.ClockMHz != 600 {
		t.Errorf("telemetry: %+v", g.Telemetry)
	}
	// Attribution.
	if len(g.Processes) != 1 {
		t.Fatalf("expected 1 process, got %+v", g.Processes)
	}
	p := g.Processes[0]
	if p.PID != 4242 || p.ModelID != 7 || p.DisplayName != "my-model" || p.VRAMBytes != 6<<30 || p.GTTBytes != 1<<30 {
		t.Errorf("process: %+v", p)
	}
	// Unattributed = used - attributed = 10 - 6 = 4 GiB.
	if g.UnattributedVRAMBytes != 4<<30 {
		t.Errorf("unattributed: got %d, want %d", g.UnattributedVRAMBytes, uint64(4)<<30)
	}
}

func TestHardware_ProcessesEmptyArrayWhenIdle(t *testing.T) {
	f := newFixture(t) // no loaded models
	resp := f.do(t, http.MethodGet, "/api/v1/hardware", nil)
	assertStatus(t, resp, http.StatusOK)
	hw := decodeBody[hardwareResponse](t, resp)
	if hw.GPUs[0].Processes == nil {
		t.Error("processes should be [] not null when idle")
	}
	if len(hw.GPUs[0].Processes) != 0 {
		t.Errorf("expected no processes, got %+v", hw.GPUs[0].Processes)
	}
	// Nothing attributed -> unattributed equals in-use (8 GiB from fixture).
	if hw.GPUs[0].UnattributedVRAMBytes != 8<<30 {
		t.Errorf("unattributed: got %d, want %d", hw.GPUs[0].UnattributedVRAMBytes, uint64(8)<<30)
	}
}

func TestHardware_UnattributedClampedToZero(t *testing.T) {
	f := newFixture(t)
	seedLoaded(f, 7, "m", 4242, 8192)
	f.srv.queryHardware = func() ([]estimator.GPUInfo, uint64, error) {
		return []estimator.GPUInfo{{Index: 0, VRAMTotal: 100 << 30, VRAMUsed: 4 << 30, VRAMAvail: 96 << 30}}, 0, nil
	}
	// Attributed (6) exceeds used (4) due to sampling skew -> clamp to 0.
	f.srv.queryProcessGPUMem = func(pids []int) ([]estimator.ProcessGPUMemory, error) {
		return []estimator.ProcessGPUMemory{{PID: 4242, Cards: []estimator.ProcessCardMemory{{CardIndex: 0, VRAMBytes: 6 << 30}}}}, nil
	}
	resp := f.do(t, http.MethodGet, "/api/v1/hardware", nil)
	hw := decodeBody[hardwareResponse](t, resp)
	if hw.GPUs[0].UnattributedVRAMBytes != 0 {
		t.Errorf("expected clamp to 0, got %d", hw.GPUs[0].UnattributedVRAMBytes)
	}
}

func TestHardware_AttributionErrorIsGraceful(t *testing.T) {
	f := newFixture(t)
	f.srv.logger = slog.New(slog.NewTextHandler(io.Discard, nil)) // helper logs on error
	seedLoaded(f, 7, "m", 4242, 8192)
	f.srv.queryProcessGPUMem = func(pids []int) ([]estimator.ProcessGPUMemory, error) {
		return nil, io.ErrUnexpectedEOF
	}
	resp := f.do(t, http.MethodGet, "/api/v1/hardware", nil)
	assertStatus(t, resp, http.StatusOK) // still succeeds
	hw := decodeBody[hardwareResponse](t, resp)
	if len(hw.GPUs[0].Processes) != 0 {
		t.Errorf("expected no processes on attribution error, got %+v", hw.GPUs[0].Processes)
	}
}

func TestStatus_MeasuredVRAM(t *testing.T) {
	f := newFixture(t)
	seedLoaded(f, 7, "my-model", 4242, 8192)
	f.srv.queryProcessGPUMem = func(pids []int) ([]estimator.ProcessGPUMemory, error) {
		return []estimator.ProcessGPUMemory{{PID: 4242, Cards: []estimator.ProcessCardMemory{
			{CardIndex: 0, VRAMBytes: 3 << 30},
			{CardIndex: 1, VRAMBytes: 1 << 30}, // summed across cards
		}}}, nil
	}
	resp := f.do(t, http.MethodGet, "/api/v1/status", nil)
	assertStatus(t, resp, http.StatusOK)
	st := decodeBody[statusResponse](t, resp)
	if len(st.LoadedModels) != 1 {
		t.Fatalf("expected 1 loaded model, got %+v", st.LoadedModels)
	}
	if st.LoadedModels[0].MeasuredVRAMBytes != 4<<30 {
		t.Errorf("measured: got %d, want %d", st.LoadedModels[0].MeasuredVRAMBytes, uint64(4)<<30)
	}
}
