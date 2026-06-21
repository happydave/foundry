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
