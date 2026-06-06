package estimator

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// QueryVRAMTotal returns the sum of total VRAM across all AMD DRM cards in bytes.
// This is the hardware capacity before any usage is subtracted.
func QueryVRAMTotal() (uint64, error) {
	matches, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err != nil {
		return 0, fmt.Errorf("globbing sysfs: %w", err)
	}
	if len(matches) == 0 {
		return 0, errors.New("no AMD DRM VRAM sysfs entries found; AMD GPU with Vulkan support required")
	}

	var total uint64
	readOK := false
	for _, path := range matches {
		v, err := readSysfsUint64(path)
		if err != nil {
			continue
		}
		total += v
		readOK = true
	}
	if !readOK {
		return 0, errors.New("could not read VRAM total from any AMD DRM sysfs entry")
	}
	return total, nil
}

// QueryResources returns currently available VRAM and RAM in bytes.
//
// VRAM is read from the AMD sysfs interface at
// /sys/class/drm/cardN/device/mem_info_vram_{total,used}. The function sums
// available VRAM across all detected cards. An error is returned if no AMD DRM
// sysfs entries are found or if all reads fail — the caller must not silently
// assume zero VRAM.
//
// RAM availability is read from /proc/meminfo (MemAvailable field), converted
// from kibibytes to bytes.
func QueryResources() (vramAvail, ramAvail uint64, err error) {
	vramAvail, err = queryVRAM()
	if err != nil {
		return 0, 0, fmt.Errorf("querying VRAM: %w", err)
	}
	ramAvail, err = queryRAM()
	if err != nil {
		return 0, 0, fmt.Errorf("querying RAM: %w", err)
	}
	return vramAvail, ramAvail, nil
}

// queryVRAM sums available VRAM across all AMD DRM cards by reading sysfs.
// Returns an error if no AMD GPU sysfs entries are found or all reads fail.
func queryVRAM() (uint64, error) {
	matches, err := filepath.Glob("/sys/class/drm/card*/device/mem_info_vram_total")
	if err != nil {
		return 0, fmt.Errorf("globbing sysfs: %w", err)
	}
	if len(matches) == 0 {
		return 0, errors.New("no AMD DRM VRAM sysfs entries found; AMD GPU with Vulkan support required")
	}

	var totalAvail uint64
	readOK := false
	for _, totalPath := range matches {
		usedPath := strings.Replace(totalPath, "mem_info_vram_total", "mem_info_vram_used", 1)

		total, err := readSysfsUint64(totalPath)
		if err != nil {
			continue
		}
		used, err := readSysfsUint64(usedPath)
		if err != nil {
			continue
		}
		if total > used {
			totalAvail += total - used
		}
		readOK = true
	}

	if !readOK {
		return 0, errors.New("could not read VRAM info from any AMD DRM sysfs entry")
	}
	return totalAvail, nil
}

// queryRAM reads MemAvailable from /proc/meminfo and returns the value in bytes.
func queryRAM() (uint64, error) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, fmt.Errorf("reading /proc/meminfo: %w", err)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "MemAvailable:") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			return 0, fmt.Errorf("unexpected MemAvailable format: %q", line)
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0, fmt.Errorf("parsing MemAvailable value: %w", err)
		}
		return kb * 1024, nil
	}
	return 0, errors.New("MemAvailable not found in /proc/meminfo")
}

// readSysfsUint64 reads a single uint64 value from a sysfs file.
func readSysfsUint64(path string) (uint64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	return strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
}
