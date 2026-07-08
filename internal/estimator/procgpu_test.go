package estimator

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// procDRMFixture builds a fake /proc and /sys/class/drm tree for attribution
// tests. It returns the proc root and drm root. One PCI device is shared by
// card1 and renderD128 (mirroring a single-GPU host).
type fdSpec struct {
	fdName string
	target string // symlink target, e.g. /dev/dri/renderD128
	fdinfo string // fdinfo file contents; empty means create no fdinfo file
	noLink bool   // if true, skip creating the fd symlink (dangling fdinfo)
}

func buildProcDRM(t *testing.T, pidFDs map[int][]fdSpec) (procRoot, drmRoot string) {
	t.Helper()
	base := t.TempDir()
	procRoot = filepath.Join(base, "proc")
	drmRoot = filepath.Join(base, "drm")

	// Shared PCI device dir, plus card1 and renderD128 both pointing at it.
	pciDev := filepath.Join(drmRoot, "pci_0000_c6_00_0")
	if err := os.MkdirAll(pciDev, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, node := range []string{"card1", "renderD128"} {
		nodeDir := filepath.Join(drmRoot, node)
		if err := os.MkdirAll(nodeDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(pciDev, filepath.Join(nodeDir, "device")); err != nil {
			t.Fatal(err)
		}
	}
	// A connector subdir (card1-DP-1) sharing the PCI device must be ignored.
	connDir := filepath.Join(drmRoot, "card1-DP-1")
	if err := os.MkdirAll(connDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(pciDev, filepath.Join(connDir, "device")); err != nil {
		t.Fatal(err)
	}

	for pid, fds := range pidFDs {
		fdDir := filepath.Join(procRoot, strconv.Itoa(pid), "fd")
		fdinfoDir := filepath.Join(procRoot, strconv.Itoa(pid), "fdinfo")
		if err := os.MkdirAll(fdDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(fdinfoDir, 0o755); err != nil {
			t.Fatal(err)
		}
		for _, fd := range fds {
			if !fd.noLink {
				if err := os.Symlink(fd.target, filepath.Join(fdDir, fd.fdName)); err != nil {
					t.Fatal(err)
				}
			}
			if fd.fdinfo != "" {
				if err := os.WriteFile(filepath.Join(fdinfoDir, fd.fdName), []byte(fd.fdinfo), 0o644); err != nil {
					t.Fatal(err)
				}
			}
		}
	}
	return procRoot, drmRoot
}

// amdFdinfo builds an amdgpu fdinfo body.
func amdFdinfo(clientID, vramKiB, gttKiB string) string {
	s := "pos:\t0\ndrm-driver:\tamdgpu\n"
	if clientID != "" {
		s += "drm-client-id:\t" + clientID + "\n"
	}
	s += "drm-memory-vram:\t" + vramKiB + " KiB\n"
	s += "drm-memory-gtt:\t" + gttKiB + " KiB\n"
	s += "drm-engine-gfx:\t123 ns\n"
	return s
}

func findCard(cards []ProcessCardMemory, idx int) (ProcessCardMemory, bool) {
	for _, c := range cards {
		if c.CardIndex == idx {
			return c, true
		}
	}
	return ProcessCardMemory{}, false
}

func TestQueryProcessGPUMemory_SingleClient(t *testing.T) {
	// 1024 KiB = 1048576 bytes; 2048 KiB = 2097152 bytes.
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {{fdName: "3", target: "/dev/dri/renderD128", fdinfo: amdFdinfo("100", "1024", "2048")}},
	})
	res, err := queryProcessGPUMemory(proc, drm, []int{42})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].PID != 42 {
		t.Fatalf("got %+v", res)
	}
	c, ok := findCard(res[0].Cards, 1)
	if !ok {
		t.Fatalf("no card 1 in %+v", res[0].Cards)
	}
	if c.VRAMBytes != 1024*1024 || c.GTTBytes != 2048*1024 {
		t.Errorf("got vram=%d gtt=%d", c.VRAMBytes, c.GTTBytes)
	}
}

func TestQueryProcessGPUMemory_DuplicateFdsSameClientCountedOnce(t *testing.T) {
	body := amdFdinfo("100", "1024", "0")
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {
			{fdName: "3", target: "/dev/dri/renderD128", fdinfo: body},
			{fdName: "4", target: "/dev/dri/renderD128", fdinfo: body}, // same client-id
		},
	})
	res, _ := queryProcessGPUMemory(proc, drm, []int{42})
	c, _ := findCard(res[0].Cards, 1)
	if c.VRAMBytes != 1024*1024 {
		t.Errorf("dup client double-counted: got %d, want %d", c.VRAMBytes, 1024*1024)
	}
}

func TestQueryProcessGPUMemory_DistinctClientsSummed(t *testing.T) {
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {
			{fdName: "3", target: "/dev/dri/renderD128", fdinfo: amdFdinfo("100", "1024", "0")},
			{fdName: "4", target: "/dev/dri/renderD128", fdinfo: amdFdinfo("200", "2048", "0")},
		},
	})
	res, _ := queryProcessGPUMemory(proc, drm, []int{42})
	c, _ := findCard(res[0].Cards, 1)
	if c.VRAMBytes != (1024+2048)*1024 {
		t.Errorf("distinct clients: got %d, want %d", c.VRAMBytes, (1024+2048)*1024)
	}
}

func TestQueryProcessGPUMemory_CardNodeDirect(t *testing.T) {
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {{fdName: "3", target: "/dev/dri/card1", fdinfo: amdFdinfo("100", "512", "0")}},
	})
	res, _ := queryProcessGPUMemory(proc, drm, []int{42})
	if _, ok := findCard(res[0].Cards, 1); !ok {
		t.Errorf("card node not resolved: %+v", res)
	}
}

func TestQueryProcessGPUMemory_NonAMDSkipped(t *testing.T) {
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {{fdName: "3", target: "/dev/dri/renderD128", fdinfo: "drm-driver:\ti915\ndrm-memory-vram:\t1024 KiB\n"}},
	})
	res, _ := queryProcessGPUMemory(proc, drm, []int{42})
	if len(res) != 0 {
		t.Errorf("non-amdgpu should be skipped, got %+v", res)
	}
}

func TestQueryProcessGPUMemory_NonDRIFdIgnored(t *testing.T) {
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {
			{fdName: "1", target: "/dev/null"},
			{fdName: "3", target: "/dev/dri/renderD128", fdinfo: amdFdinfo("100", "1024", "0")},
		},
	})
	res, _ := queryProcessGPUMemory(proc, drm, []int{42})
	if len(res) != 1 {
		t.Fatalf("got %+v", res)
	}
	c, _ := findCard(res[0].Cards, 1)
	if c.VRAMBytes != 1024*1024 {
		t.Errorf("got %d", c.VRAMBytes)
	}
}

func TestQueryProcessGPUMemory_MissingPIDSkipped(t *testing.T) {
	proc, drm := buildProcDRM(t, map[int][]fdSpec{
		42: {{fdName: "3", target: "/dev/dri/renderD128", fdinfo: amdFdinfo("100", "1024", "0")}},
	})
	// 999 has no /proc entry; must be skipped without error.
	res, err := queryProcessGPUMemory(proc, drm, []int{999, 42})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 1 || res[0].PID != 42 {
		t.Errorf("got %+v", res)
	}
}

func TestParseKiB(t *testing.T) {
	tests := []struct {
		in   string
		want uint64
		ok   bool
	}{
		{"45431656 KiB", 45431656 * 1024, true},
		{"1024", 1024 * 1024, true}, // bare number treated as KiB
		{"2 MiB", 2 * 1024 * 1024, true},
		{"512 B", 512, true},
		{"", 0, false},
		{"notanumber KiB", 0, false},
	}
	for _, tc := range tests {
		got, ok := parseKiB(tc.in)
		if ok != tc.ok || got != tc.want {
			t.Errorf("parseKiB(%q) = (%d,%v), want (%d,%v)", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

func TestDriNodeName(t *testing.T) {
	tests := []struct{ in, want string }{
		{"/dev/dri/renderD128", "renderD128"},
		{"/dev/dri/card1", "card1"},
		{"/dev/null", ""},
		{"/dev/dri/", ""},
		{"/dev/dri/subdir/x", ""},
	}
	for _, tc := range tests {
		if got := driNodeName(tc.in); got != tc.want {
			t.Errorf("driNodeName(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
