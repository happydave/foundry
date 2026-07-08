package estimator

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ProcessCardMemory is one process's measured GPU memory footprint on a single
// AMD card, in bytes.
type ProcessCardMemory struct {
	CardIndex int
	VRAMBytes uint64
	GTTBytes  uint64
}

// ProcessGPUMemory is a process's measured GPU memory, potentially spanning more
// than one card. A process with no attributable amdgpu memory is omitted from
// results rather than reported with an empty Cards slice.
type ProcessGPUMemory struct {
	PID   int
	Cards []ProcessCardMemory
}

// QueryProcessGPUMemory reads measured GPU memory for the given process IDs from
// /proc/<pid>/fdinfo, attributing each process's usage to the AMD card its DRM
// file descriptors target. PIDs are caller-supplied (Foundry passes the PIDs of
// its own llama-server subprocesses); this keeps the estimator free of any
// dependency on the process manager and avoids scanning unrelated processes.
//
// Every read is best-effort: a since-exited PID, an unreadable fdinfo, or an
// unresolvable card is skipped rather than failing the whole query. The returned
// error is non-nil only for a programming-level fault, never for absent data.
func QueryProcessGPUMemory(pids []int) ([]ProcessGPUMemory, error) {
	return queryProcessGPUMemory("/proc", "/sys/class/drm", pids)
}

// queryProcessGPUMemory is the root-parameterized implementation, for testing.
func queryProcessGPUMemory(procRoot, drmRoot string, pids []int) ([]ProcessGPUMemory, error) {
	out := make([]ProcessGPUMemory, 0, len(pids))
	nodeCache := map[string]int{} // DRM node name -> card index (or -1)

	for _, pid := range pids {
		cards := readPidGPUMemory(procRoot, drmRoot, pid, nodeCache)
		if len(cards) == 0 {
			continue
		}
		out = append(out, ProcessGPUMemory{PID: pid, Cards: cards})
	}
	return out, nil
}

// fdMem is one DRM client's memory as parsed from a single fdinfo file.
type fdMem struct {
	clientKey string // drm-client-id when present, else "" (caller supplies fallback)
	vram      uint64
	gtt       uint64
}

// readPidGPUMemory reads and de-duplicates one process's amdgpu memory across all
// of its DRM file descriptors, returning per-card totals.
func readPidGPUMemory(procRoot, drmRoot string, pid int, nodeCache map[string]int) []ProcessCardMemory {
	fdDir := filepath.Join(procRoot, strconv.Itoa(pid), "fd")
	entries, err := os.ReadDir(fdDir)
	if err != nil {
		return nil // process gone or unreadable
	}

	// Dedupe: a process may hold several fds to the same DRM client (identical
	// accounting) or several distinct clients. Key each contribution by
	// (card, clientKey) so duplicate fds of one client are counted once while
	// genuinely distinct clients each contribute.
	type key struct {
		card      int
		clientKey string
	}
	seen := map[key]struct{}{}
	totals := map[int]*ProcessCardMemory{}

	for _, e := range entries {
		fdName := e.Name()
		target, err := os.Readlink(filepath.Join(fdDir, fdName))
		if err != nil {
			continue
		}
		node := driNodeName(target)
		if node == "" {
			continue
		}
		cardIdx, ok := resolveNodeCardIndex(drmRoot, node, nodeCache)
		if !ok {
			continue
		}
		m, ok := parseFdinfoMem(filepath.Join(procRoot, strconv.Itoa(pid), "fdinfo", fdName))
		if !ok {
			continue
		}
		clientKey := m.clientKey
		if clientKey == "" {
			clientKey = "fd:" + fdName
		}
		k := key{card: cardIdx, clientKey: clientKey}
		if _, dup := seen[k]; dup {
			continue
		}
		seen[k] = struct{}{}
		t := totals[cardIdx]
		if t == nil {
			t = &ProcessCardMemory{CardIndex: cardIdx}
			totals[cardIdx] = t
		}
		t.VRAMBytes += m.vram
		t.GTTBytes += m.gtt
	}

	if len(totals) == 0 {
		return nil
	}
	// Deterministic order by card index.
	indices := make([]int, 0, len(totals))
	for idx := range totals {
		indices = append(indices, idx)
	}
	sortInts(indices)
	cards := make([]ProcessCardMemory, 0, len(totals))
	for _, idx := range indices {
		cards = append(cards, *totals[idx])
	}
	return cards
}

// driNodeName returns the DRM node basename (e.g. "renderD128" or "card1") for a
// symlink target under /dev/dri, or "" if the target is not a DRM device node.
func driNodeName(target string) string {
	const prefix = "/dev/dri/"
	if !strings.HasPrefix(target, prefix) {
		return ""
	}
	base := strings.TrimPrefix(target, prefix)
	if base == "" || strings.Contains(base, "/") {
		return ""
	}
	return base
}

// resolveNodeCardIndex maps a DRM node name to the sysfs card index that shares
// its PCI device, matching the index QueryHardware reports. Card nodes ("cardN")
// resolve directly; render nodes ("renderDN") resolve by matching PCI device
// directories. Results are cached per node name.
func resolveNodeCardIndex(drmRoot, node string, cache map[string]int) (int, bool) {
	if v, ok := cache[node]; ok {
		return v, v >= 0
	}
	idx := computeNodeCardIndex(drmRoot, node)
	cache[node] = idx
	return idx, idx >= 0
}

func computeNodeCardIndex(drmRoot, node string) int {
	// A card node carries its index directly.
	if strings.HasPrefix(node, "card") {
		if idx := cardIndex(node); idx >= 0 {
			return idx
		}
	}
	// A render node must be matched to a card by shared PCI device.
	nodeDev, err := filepath.EvalSymlinks(filepath.Join(drmRoot, node, "device"))
	if err != nil {
		return -1
	}
	cards, err := filepath.Glob(filepath.Join(drmRoot, "card*", "device"))
	if err != nil {
		return -1
	}
	for _, cardDevLink := range cards {
		cardName := filepath.Base(filepath.Dir(cardDevLink))
		idx := cardIndex(cardName) // connector subdirs (card1-DP-1) yield -1 and are skipped
		if idx < 0 {
			continue
		}
		cardDev, err := filepath.EvalSymlinks(cardDevLink)
		if err != nil {
			continue
		}
		if cardDev == nodeDev {
			return idx
		}
	}
	return -1
}

// parseFdinfoMem parses one /proc/<pid>/fdinfo/<fd> file, returning the amdgpu
// resident VRAM/GTT (converted from KiB to bytes) and the drm-client-id used for
// de-duplication. ok is false when the file is unreadable or not an amdgpu DRM
// client with memory accounting.
func parseFdinfoMem(path string) (fdMem, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return fdMem{}, false
	}
	var (
		m           fdMem
		isAMD       bool
		haveVRAM    bool
		haveGTT     bool
		haveClient  bool
		clientIDStr string
	)
	for _, line := range strings.Split(string(data), "\n") {
		field, value, ok := splitFdinfoLine(line)
		if !ok {
			continue
		}
		switch field {
		case "drm-driver":
			isAMD = value == "amdgpu"
		case "drm-client-id":
			clientIDStr = value
			haveClient = true
		case "drm-memory-vram":
			if b, ok := parseKiB(value); ok {
				m.vram = b
				haveVRAM = true
			}
		case "drm-memory-gtt":
			if b, ok := parseKiB(value); ok {
				m.gtt = b
				haveGTT = true
			}
		}
	}
	if !isAMD || (!haveVRAM && !haveGTT) {
		return fdMem{}, false
	}
	if haveClient {
		m.clientKey = clientIDStr
	}
	return m, true
}

// splitFdinfoLine splits a "key:\tvalue" fdinfo line into trimmed field and
// value. ok is false for lines without a colon separator.
func splitFdinfoLine(line string) (field, value string, ok bool) {
	idx := strings.IndexByte(line, ':')
	if idx < 0 {
		return "", "", false
	}
	return strings.TrimSpace(line[:idx]), strings.TrimSpace(line[idx+1:]), true
}

// parseKiB parses an fdinfo memory value such as "45431656 KiB" (or a bare
// number) and returns the value in bytes.
func parseKiB(value string) (uint64, bool) {
	fields := strings.Fields(value)
	if len(fields) == 0 {
		return 0, false
	}
	n, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0, false
	}
	unit := "KiB"
	if len(fields) > 1 {
		unit = fields[1]
	}
	switch unit {
	case "KiB", "kB", "kib":
		return n * 1024, true
	case "MiB":
		return n * 1024 * 1024, true
	case "B":
		return n, true
	default:
		// Unknown unit: treat the number as KiB (the amdgpu convention).
		return n * 1024, true
	}
}

// sortInts sorts a small int slice ascending (avoids importing sort for one use
// but kept simple; insertion sort is fine for the handful of cards involved).
func sortInts(a []int) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j-1] > a[j]; j-- {
			a[j-1], a[j] = a[j], a[j-1]
		}
	}
}
