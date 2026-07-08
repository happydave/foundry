package estimator

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// GPUInfo describes a single AMD DRM card's identity and VRAM figures at a point
// in time. All VRAM figures are in bytes and are live reads from sysfs.
type GPUInfo struct {
	// Index is the DRM card number parsed from the sysfs card name (e.g. 2 for
	// "card2"). It disambiguates otherwise-identical cards.
	Index int
	// Identity is a non-empty human-meaningful label: the driver-provided
	// product name if available, otherwise the PCI vendor:device hex pair,
	// otherwise the sysfs card name.
	Identity  string
	VRAMTotal uint64
	VRAMUsed  uint64
	VRAMAvail uint64

	// Pools carries the additional AMD memory-pool figures beyond dedicated
	// VRAM. All are best-effort live reads; a pool whose sysfs file is absent
	// reads as zero. On unified-memory APUs the GTT pool is significant.
	Pools GPUPools
	// Telemetry carries best-effort live card telemetry. Each field is a
	// pointer so an absent sysfs/hwmon source is distinguishable from a real
	// zero and can be omitted from API responses.
	Telemetry GPUTelemetry
}

// GPUPools holds the AMD memory-pool figures (bytes) read from the card's sysfs
// mem_info_* attributes. Zero indicates the corresponding attribute was absent
// or unreadable.
type GPUPools struct {
	GTTTotal     uint64
	GTTUsed      uint64
	VisVRAMTotal uint64
	VisVRAMUsed  uint64
	PreemptUsed  uint64
}

// GPUTelemetry holds best-effort live card telemetry. A nil field means the
// underlying source was unavailable on this card/driver.
type GPUTelemetry struct {
	// BusyPercent is the GPU utilization percentage (0-100) from
	// gpu_busy_percent.
	BusyPercent *uint64
	// TemperatureMilliC is the edge temperature in millidegrees Celsius from
	// hwmon temp1_input.
	TemperatureMilliC *uint64
	// PowerMicroW is average power draw in microwatts from hwmon power1_average.
	PowerMicroW *uint64
	// ClockMHz is the current core (sclk) clock in MHz — the active state in
	// pp_dpm_sclk.
	ClockMHz *uint64
}

// QueryHardware returns per-card VRAM detail for every AMD DRM card that exposes
// VRAM info via sysfs, along with currently available system RAM in bytes.
//
// It anchors on the same sysfs entry as the aggregate queries
// (/sys/class/drm/card*/device/mem_info_vram_total), so the set of cards it
// reports matches the cards the aggregate sums. Non-AMD cards and DRM connector
// subdirectories do not expose that entry and are naturally excluded.
//
// An error is returned if no AMD DRM VRAM entries are found or if all card reads
// fail; the caller must not treat an empty result as "zero cards".
func QueryHardware() (gpus []GPUInfo, ramAvail uint64, err error) {
	matches, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err != nil {
		return nil, 0, fmt.Errorf("globbing sysfs: %w", err)
	}
	if len(matches) == 0 {
		return nil, 0, errors.New("no AMD DRM VRAM sysfs entries found; AMD GPU with Vulkan support required")
	}

	gpus = make([]GPUInfo, 0, len(matches))
	for _, totalPath := range matches {
		total, err := readSysfsUint64(totalPath)
		if err != nil {
			continue
		}
		usedPath := strings.Replace(totalPath, "mem_info_vram_total", "mem_info_vram_used", 1)
		used, err := readSysfsUint64(usedPath)
		if err != nil {
			continue
		}
		var avail uint64
		if total > used {
			avail = total - used
		}

		deviceDir := filepath.Dir(totalPath)
		cardName := cardNameFromDevicePath(deviceDir)
		gpus = append(gpus, GPUInfo{
			Index:     cardIndex(cardName),
			Identity:  resolveCardIdentity(deviceDir, cardName),
			VRAMTotal: total,
			VRAMUsed:  used,
			VRAMAvail: avail,
			Pools:     readCardPools(deviceDir),
			Telemetry: readCardTelemetry(deviceDir),
		})
	}

	if len(gpus) == 0 {
		return nil, 0, errors.New("could not read VRAM info from any AMD DRM sysfs entry")
	}

	ramAvail, err = queryRAM()
	if err != nil {
		return nil, 0, fmt.Errorf("querying RAM: %w", err)
	}
	return gpus, ramAvail, nil
}

// cardNameFromDevicePath extracts the "cardN" segment from a device-node path
// such as /sys/class/drm/card2/device. Returns the empty string if not found.
func cardNameFromDevicePath(deviceDir string) string {
	for _, seg := range strings.Split(deviceDir, string(filepath.Separator)) {
		if strings.HasPrefix(seg, "card") {
			return seg
		}
	}
	return ""
}

// cardIndex parses the numeric suffix of a "cardN" name. Returns -1 if absent.
func cardIndex(cardName string) int {
	n, err := strconv.Atoi(strings.TrimPrefix(cardName, "card"))
	if err != nil {
		return -1
	}
	return n
}

// resolveCardIdentity produces a non-empty identity string for a card given its
// sysfs device directory. Precedence: a non-empty product_name attribute, then
// the PCI vendor:device hex pair, then the card name as a last resort.
func resolveCardIdentity(deviceDir, cardName string) string {
	if name := strings.TrimSpace(readSysfsString(filepath.Join(deviceDir, "product_name"))); name != "" {
		return name
	}
	vendor := pciHex(readSysfsString(filepath.Join(deviceDir, "vendor")))
	device := pciHex(readSysfsString(filepath.Join(deviceDir, "device")))
	if vendor != "" && device != "" {
		return vendor + ":" + device
	}
	return cardName
}

// pciHex normalises a sysfs PCI id value ("0x1002\n") to lowercase hex without
// the 0x prefix ("1002"). Returns the empty string for empty input.
func pciHex(raw string) string {
	v := strings.ToLower(strings.TrimSpace(raw))
	v = strings.TrimPrefix(v, "0x")
	return v
}

// readSysfsString reads a sysfs attribute as a string, returning the empty
// string if the file is absent or unreadable. Identity resolution is best-effort
// and never fails the overall query.
func readSysfsString(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(data)
}

// readCardPools reads the AMD memory-pool attributes from a card's sysfs device
// directory. Every read is best-effort: a missing or unreadable attribute
// contributes zero and never fails the query.
func readCardPools(deviceDir string) GPUPools {
	read := func(name string) uint64 {
		v, _ := readOptionalSysfsUint64(filepath.Join(deviceDir, name))
		return v
	}
	return GPUPools{
		GTTTotal:     read("mem_info_gtt_total"),
		GTTUsed:      read("mem_info_gtt_used"),
		VisVRAMTotal: read("mem_info_vis_vram_total"),
		VisVRAMUsed:  read("mem_info_vis_vram_used"),
		PreemptUsed:  read("mem_info_preempt_used"),
	}
}

// readCardTelemetry reads best-effort live telemetry from a card's sysfs device
// directory. Any source that is absent leaves the corresponding pointer nil.
func readCardTelemetry(deviceDir string) GPUTelemetry {
	var t GPUTelemetry
	if v, ok := readOptionalSysfsUint64(filepath.Join(deviceDir, "gpu_busy_percent")); ok {
		t.BusyPercent = &v
	}
	if v, ok := readHwmonUint64(deviceDir, "temp1_input"); ok {
		t.TemperatureMilliC = &v
	}
	if v, ok := readHwmonUint64(deviceDir, "power1_average"); ok {
		t.PowerMicroW = &v
	}
	if v, ok := currentDPMClockMHz(filepath.Join(deviceDir, "pp_dpm_sclk")); ok {
		t.ClockMHz = &v
	}
	return t
}

// readOptionalSysfsUint64 reads a uint64 sysfs attribute, reporting ok=false
// (rather than an error) when the file is absent or unparseable. Used for
// best-effort figures that must never fail the overall query.
func readOptionalSysfsUint64(path string) (uint64, bool) {
	v, err := readSysfsUint64(path)
	if err != nil {
		return 0, false
	}
	return v, true
}

// readHwmonUint64 reads a named hwmon attribute for a card. The hwmon instance
// number is not stable, so the single hwmon directory under the card's device
// path is discovered by glob. Returns ok=false when absent or unreadable.
func readHwmonUint64(deviceDir, attr string) (uint64, bool) {
	matches, err := filepath.Glob(filepath.Join(deviceDir, "hwmon", "hwmon*", attr))
	if err != nil || len(matches) == 0 {
		return 0, false
	}
	return readOptionalSysfsUint64(matches[0])
}

// currentDPMClockMHz parses an AMD pp_dpm_* clock table and returns the active
// state's frequency in MHz. Each line looks like "0: 600Mhz *"; the active line
// is the one marked with a trailing "*". Returns ok=false when the file is
// absent, empty, or no active line is found.
func currentDPMClockMHz(path string) (uint64, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasSuffix(strings.TrimSpace(line), "*") {
			continue
		}
		// Fields: "<index>:" "<freq>Mhz" "*"
		fields := strings.Fields(line)
		for _, f := range fields {
			lower := strings.ToLower(f)
			if !strings.HasSuffix(lower, "mhz") {
				continue
			}
			numStr := strings.TrimSuffix(lower, "mhz")
			if mhz, err := strconv.ParseUint(numStr, 10, 64); err == nil {
				return mhz, true
			}
		}
	}
	return 0, false
}
