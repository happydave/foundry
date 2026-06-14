package registry

import (
	"fmt"
	"hash/fnv"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

// Model describes a model discovered during the startup directory scan.
// All fields are populated at scan time from GGUF metadata and filesystem info.
type Model struct {
	ID           uint64
	DisplayName  string // filename without the .gguf extension
	Path         string
	FileSize     int64
	Architecture string
	LayerCount   uint32
	HeadCount    uint32
	KVHeadCount  uint32
	HeadDim      uint32
	MaxContext   uint32
	Quantization string
	MmprojPath   string

	// Sliding-window (local) attention fields. Zero for models that use fully
	// global attention. When SlidingWindowSize is non-zero, the KV cache estimate
	// splits into a context-scaling global term and a fixed sliding-window term.
	SlidingWindowSize uint32
	SWAHeadDim        uint32
	GlobalLayerCount  uint32
	SWALayerCount     uint32
	GlobalKVHeadCount uint32
	SWAKVHeadCount    uint32
}

// Registry is the in-process catalogue of models discovered at startup. It is populated
// once and safe for concurrent read access after New returns.
type Registry struct {
	models []Model
	byID   map[uint64]int // model ID → index in models
}

// New scans each directory in scanPaths for top-level GGUF files, parses their metadata,
// and returns a populated Registry. Unreadable paths and unparseable files are logged and
// skipped; neither prevents the registry from being returned.
func New(scanPaths []string, logger *slog.Logger) *Registry {
	reg := &Registry{byID: make(map[uint64]int)}
	total := 0
	for _, dir := range scanPaths {
		total += reg.scanDir(dir, logger)
	}
	logger.Info("model registry ready", "model_count", total)
	return reg
}

// List returns all registered models as a new slice.
func (r *Registry) List() []Model {
	out := make([]Model, len(r.models))
	copy(out, r.models)
	return out
}

// Get returns the model with the given ID. The second return value is false if no model
// with that ID is registered.
func (r *Registry) Get(id uint64) (Model, bool) {
	idx, ok := r.byID[id]
	if !ok {
		return Model{}, false
	}
	return r.models[idx], true
}

// GetByName returns the model whose DisplayName matches name. DisplayName is the GGUF
// filename without the .gguf extension. The second return value is false if not found.
func (r *Registry) GetByName(name string) (Model, bool) {
	for _, m := range r.models {
		if m.DisplayName == name {
			return m, true
		}
	}
	return Model{}, false
}

// scanDir scans dir for top-level GGUF files (non-recursive). Returns the number of models
// successfully added.
func (r *Registry) scanDir(dir string, logger *slog.Logger) int {
	info, err := os.Stat(dir)
	if err != nil {
		logger.Warn("cannot access model scan path", "path", dir, "error", err)
		return 0
	}
	if !info.IsDir() {
		logger.Warn("model scan path is not a directory", "path", dir)
		return 0
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		logger.Warn("cannot read model scan directory", "path", dir, "error", err)
		return 0
	}

	var textModels []Model
	var mmprojPath string

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		if !strings.EqualFold(filepath.Ext(entry.Name()), ".gguf") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		m, isMmproj, ok := r.parseModel(path, logger)
		if !ok {
			continue
		}

		if isMmproj {
			if mmprojPath == "" {
				mmprojPath = path
			}
		} else {
			textModels = append(textModels, m)
		}
	}

	added := 0
	for _, m := range textModels {
		if mmprojPath != "" {
			m.MmprojPath = mmprojPath
		}
		idx := len(r.models)
		r.models = append(r.models, m)
		r.byID[m.ID] = idx
		added++
	}
	return added
}

// parseModel parses the GGUF file at path. Returns the Model, a boolean indicating
// if it is an mmproj file, and a boolean indicating success.
func (r *Registry) parseModel(path string, logger *slog.Logger) (Model, bool, bool) {
	fi, err := os.Stat(path)
	if err != nil {
		logger.Warn("cannot stat model file", "path", path, "error", err)
		return Model{}, false, false
	}

	meta, err := parseGGUF(path)
	if err != nil {
		logger.Warn("cannot parse GGUF file", "path", path, "error", err)
		return Model{}, false, false
	}

	if meta.isMmproj {
		return Model{}, true, true
	}

	arch := meta.architecture
	if !meta.hasArchitecture {
		arch = "<unknown>"
	}

	var missing []string
	if !meta.hasArchitecture {
		missing = append(missing, "general.architecture")
	}
	if !meta.hasLayerCount {
		missing = append(missing, arch+".block_count")
	}
	if !meta.hasHeadCount {
		missing = append(missing, arch+".attention.head_count")
	}
	if !meta.hasKVHeadCount {
		missing = append(missing, arch+".attention.head_count_kv")
	}
	if !meta.hasMaxContext {
		missing = append(missing, arch+".context_length")
	}
	if !meta.hasFileType {
		missing = append(missing, "general.file_type")
	}
	if len(missing) > 0 {
		logger.Warn("skipping model: missing required metadata fields",
			"path", path,
			"missing_fields", strings.Join(missing, ", "),
		)
		return Model{}, false, false
	}

	// Head dimension: prefer the direct field; fall back to embedding_length / head_count.
	headDim := meta.headDim
	if headDim == 0 && meta.headCount > 0 {
		headDim = meta.embedLen / meta.headCount
	}
	if headDim == 0 {
		logger.Warn("skipping model: cannot determine head dimension",
			"path", path,
		)
		return Model{}, false, false
	}

	// A sliding-window key was present but the per-layer derivation could not run
	// (missing pattern, missing per-layer KV head array, or length mismatch). The
	// model is still usable; estimation falls back to the conservative all-layers
	// formula. Surface the condition so it is visible in logs.
	if meta.hasSlidingWindow && meta.slidingWindowSize == 0 {
		logger.Warn("sliding-window attention present but SWA derivation failed; using conservative all-layers KV estimate",
			"path", path,
			"architecture", arch,
		)
	}

	base := filepath.Base(path)
	displayName := strings.TrimSuffix(base, filepath.Ext(base))
	id := fingerprint(path, fi.Size())

	m := Model{
		ID:           id,
		DisplayName:  displayName,
		Path:         path,
		FileSize:     fi.Size(),
		Architecture: meta.architecture,
		LayerCount:   meta.layerCount,
		HeadCount:    meta.headCount,
		KVHeadCount:  meta.kvHeadCount,
		HeadDim:      headDim,
		MaxContext:   meta.maxContext,
		Quantization: fileTypeString(meta.fileType),

		SlidingWindowSize: meta.slidingWindowSize,
		SWAHeadDim:        meta.swaHeadDim,
		GlobalLayerCount:  meta.globalLayerCount,
		SWALayerCount:     meta.swaLayerCount,
		GlobalKVHeadCount: meta.globalKVHeadCount,
		SWAKVHeadCount:    meta.swaKVHeadCount,
	}

	return m, false, true
}

// fingerprint computes a stable uint64 ID for a model from its absolute path and file size.
// The full path (not just the basename) is included so that two files with identical names
// and sizes in different scan directories receive distinct IDs and are treated as separate models.
func fingerprint(absPath string, fileSize int64) uint64 {
	h := fnv.New64a()
	// fnv hash writers never return an error; discard the result explicitly.
	_, _ = fmt.Fprintf(h, "%s\x00%d", absPath, fileSize)
	return h.Sum64()
}
