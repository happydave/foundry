package estimator

import (
	"os"
	"path/filepath"
	"testing"
)

// writeNestedAttr writes a sysfs-style attribute file that may live in a
// subdirectory (e.g. hwmon/hwmon5/temp1_input), creating parents as needed.
// The flat-file helper writeAttr lives in hardware_test.go.
func writeNestedAttr(t *testing.T, dir, name, content string) {
	t.Helper()
	full := filepath.Join(dir, name)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func TestReadCardPools(t *testing.T) {
	dir := t.TempDir()
	writeAttr(t, dir, "mem_info_gtt_total", "16633774080\n")
	writeAttr(t, dir, "mem_info_gtt_used", "1822310400\n")
	writeAttr(t, dir, "mem_info_vis_vram_total", "103079215104\n")
	writeAttr(t, dir, "mem_info_vis_vram_used", "54845370368\n")
	// mem_info_preempt_used intentionally absent -> must read as zero.

	p := readCardPools(dir)
	if p.GTTTotal != 16633774080 || p.GTTUsed != 1822310400 {
		t.Errorf("gtt: got total=%d used=%d", p.GTTTotal, p.GTTUsed)
	}
	if p.VisVRAMTotal != 103079215104 || p.VisVRAMUsed != 54845370368 {
		t.Errorf("vis vram: got total=%d used=%d", p.VisVRAMTotal, p.VisVRAMUsed)
	}
	if p.PreemptUsed != 0 {
		t.Errorf("absent preempt: got %d, want 0", p.PreemptUsed)
	}
}

func TestReadCardPoolsAllAbsent(t *testing.T) {
	// A directory with none of the attributes must yield a zero-value struct,
	// not an error/panic.
	p := readCardPools(t.TempDir())
	if (p != GPUPools{}) {
		t.Errorf("expected zero pools, got %+v", p)
	}
}

func TestReadCardTelemetry(t *testing.T) {
	dir := t.TempDir()
	writeAttr(t, dir, "gpu_busy_percent", "42\n")
	writeNestedAttr(t, dir, "hwmon/hwmon5/temp1_input", "30000\n")
	writeNestedAttr(t, dir, "hwmon/hwmon5/power1_average", "5085000\n")
	writeAttr(t, dir, "pp_dpm_sclk", "0: 600Mhz *\n1: 2900Mhz\n")

	tel := readCardTelemetry(dir)
	if tel.BusyPercent == nil || *tel.BusyPercent != 42 {
		t.Errorf("busy: got %v", tel.BusyPercent)
	}
	if tel.TemperatureMilliC == nil || *tel.TemperatureMilliC != 30000 {
		t.Errorf("temp: got %v", tel.TemperatureMilliC)
	}
	if tel.PowerMicroW == nil || *tel.PowerMicroW != 5085000 {
		t.Errorf("power: got %v", tel.PowerMicroW)
	}
	if tel.ClockMHz == nil || *tel.ClockMHz != 600 {
		t.Errorf("clock: got %v", tel.ClockMHz)
	}
}

func TestReadCardTelemetryAllAbsent(t *testing.T) {
	tel := readCardTelemetry(t.TempDir())
	if tel.BusyPercent != nil || tel.TemperatureMilliC != nil || tel.PowerMicroW != nil || tel.ClockMHz != nil {
		t.Errorf("expected all-nil telemetry, got %+v", tel)
	}
}

func TestCurrentDPMClockMHz(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    uint64
		wantOK  bool
	}{
		{"active first", "0: 600Mhz *\n1: 2900Mhz\n", 600, true},
		{"active last", "0: 600Mhz\n1: 2900Mhz *\n", 2900, true},
		{"case insensitive", "0: 800MHz *\n", 800, true},
		{"no active marker", "0: 600Mhz\n1: 2900Mhz\n", 0, false},
		{"empty", "", 0, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "pp_dpm_sclk")
			if err := os.WriteFile(path, []byte(tc.content), 0o644); err != nil {
				t.Fatal(err)
			}
			got, ok := currentDPMClockMHz(path)
			if ok != tc.wantOK || got != tc.want {
				t.Errorf("got (%d,%v), want (%d,%v)", got, ok, tc.want, tc.wantOK)
			}
		})
	}
}

func TestCurrentDPMClockMHzAbsentFile(t *testing.T) {
	if _, ok := currentDPMClockMHz(filepath.Join(t.TempDir(), "nope")); ok {
		t.Error("expected ok=false for absent file")
	}
}
