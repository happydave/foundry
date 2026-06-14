package registry

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// ggufMagic is the little-endian uint32 encoding of the bytes 'G','G','U','F'.
const ggufMagic uint32 = 0x46554747

type ggufType uint32

const (
	ggufTypeUint8   ggufType = 0
	ggufTypeInt8    ggufType = 1
	ggufTypeUint16  ggufType = 2
	ggufTypeInt16   ggufType = 3
	ggufTypeUint32  ggufType = 4
	ggufTypeInt32   ggufType = 5
	ggufTypeFloat32 ggufType = 6
	ggufTypeBool    ggufType = 7
	ggufTypeString  ggufType = 8
	ggufTypeArray   ggufType = 9
	ggufTypeUint64  ggufType = 10
	ggufTypeInt64   ggufType = 11
	ggufTypeFloat64 ggufType = 12
)

// ggufReader reads GGUF binary data with version-aware string/count encoding.
// v1 uses uint32 counts and string lengths; v2/v3 use uint64.
type ggufReader struct {
	r       io.Reader
	version uint32
}

func (g *ggufReader) uint8() (uint8, error) {
	var b [1]byte
	_, err := io.ReadFull(g.r, b[:])
	return b[0], err
}

func (g *ggufReader) uint16() (uint16, error) {
	var buf [2]byte
	if _, err := io.ReadFull(g.r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(buf[:]), nil
}

func (g *ggufReader) uint32() (uint32, error) {
	var buf [4]byte
	if _, err := io.ReadFull(g.r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(buf[:]), nil
}

func (g *ggufReader) uint64() (uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(g.r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

func (g *ggufReader) readCount() (uint64, error) {
	if g.version == 1 {
		n, err := g.uint32()
		return uint64(n), err
	}
	return g.uint64()
}

func (g *ggufReader) string() (string, error) {
	length, err := g.readCount()
	if err != nil {
		return "", err
	}
	if length > 1<<20 {
		return "", fmt.Errorf("string length %d exceeds sanity limit", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(g.r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

// value reads one GGUF value of the given type and returns it as a Go value.
// The concrete type returned is appropriate for the GGUF type:
// uint8, int8, uint16, int16, uint32, int32, uint32 (float32 bits), bool,
// string, uint64, int64, uint64 (float64 bits), or []any for arrays.
func (g *ggufReader) value(t ggufType) (any, error) {
	switch t {
	case ggufTypeUint8:
		v, err := g.uint8()
		return v, err
	case ggufTypeInt8:
		v, err := g.uint8()
		return int8(v), err
	case ggufTypeUint16:
		v, err := g.uint16()
		return v, err
	case ggufTypeInt16:
		v, err := g.uint16()
		return int16(v), err
	case ggufTypeUint32:
		v, err := g.uint32()
		return v, err
	case ggufTypeInt32:
		v, err := g.uint32()
		return int32(v), err
	case ggufTypeFloat32:
		v, err := g.uint32()
		return v, err // raw bits; we never interpret floats from GGUF metadata
	case ggufTypeBool:
		v, err := g.uint8()
		return v != 0, err
	case ggufTypeString:
		return g.string()
	case ggufTypeUint64:
		v, err := g.uint64()
		return v, err
	case ggufTypeInt64:
		v, err := g.uint64()
		return int64(v), err
	case ggufTypeFloat64:
		v, err := g.uint64()
		return v, err // raw bits
	case ggufTypeArray:
		return g.array()
	default:
		return nil, fmt.Errorf("unknown GGUF type %d", t)
	}
}

func (g *ggufReader) array() ([]any, error) {
	itemTypeRaw, err := g.uint32()
	if err != nil {
		return nil, err
	}
	count, err := g.readCount()
	if err != nil {
		return nil, err
	}
	if count > 1<<20 {
		return nil, fmt.Errorf("array count %d exceeds sanity limit", count)
	}
	items := make([]any, count)
	for i := range items {
		v, err := g.value(ggufType(itemTypeRaw))
		if err != nil {
			return nil, fmt.Errorf("array item %d: %w", i, err)
		}
		items[i] = v
	}
	return items, nil
}

// ggufMeta holds the metadata fields extracted from a GGUF file.
type ggufMeta struct {
	architecture string
	layerCount   uint32
	headCount    uint32
	kvHeadCount  uint32
	headDim      uint32 // from {arch}.attention.key_length; 0 if absent
	embedLen     uint32 // from {arch}.embedding_length; used to derive headDim
	maxContext   uint32
	fileType     uint32

	// Sliding-window (local) attention fields. Populated for models such as
	// Gemma 4 that alternate global and sliding-window attention blocks. When
	// slidingWindowSize is zero the model is treated as fully global attention.
	slidingWindowSize uint32 // from {arch}.attention.sliding_window (validated; 0 if derivation failed)
	swaHeadDim        uint32 // from {arch}.attention.key_length_swa
	globalLayerCount  uint32 // derived: count of global (non-SWA) blocks
	swaLayerCount     uint32 // derived: count of SWA blocks
	globalKVHeadCount uint32 // derived: KV head count for global blocks
	swaKVHeadCount    uint32 // derived: KV head count for SWA blocks

	// Scratch fields used only during derivation; not propagated past parseGGUF.
	kvHeadCountArray     []uint32 // per-layer head_count_kv when stored as an array
	slidingWindowPattern []bool   // per-layer flags: true = SWA block, false = global

	hasArchitecture  bool
	hasLayerCount    bool
	hasHeadCount     bool
	hasKVHeadCount   bool
	hasMaxContext    bool
	hasFileType      bool
	hasSlidingWindow bool // true if attention.sliding_window key was present
	isMmproj         bool
}

// parseGGUF reads the metadata section of the GGUF file at path and returns the fields
// required by the registry. It reads only the header and KV pairs, not tensor data.
func parseGGUF(path string) (ggufMeta, error) {
	f, err := os.Open(path)
	if err != nil {
		return ggufMeta{}, err
	}
	defer func() { _ = f.Close() }()

	r := &ggufReader{r: f}

	magic, err := r.uint32()
	if err != nil {
		return ggufMeta{}, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return ggufMeta{}, fmt.Errorf("bad GGUF magic 0x%08x", magic)
	}

	version, err := r.uint32()
	if err != nil {
		return ggufMeta{}, fmt.Errorf("read version: %w", err)
	}
	if version < 1 || version > 3 {
		return ggufMeta{}, fmt.Errorf("unsupported GGUF version %d", version)
	}
	r.version = version

	// Skip tensor count.
	if _, err := r.readCount(); err != nil {
		return ggufMeta{}, fmt.Errorf("read n_tensors: %w", err)
	}
	nKV, err := r.readCount()
	if err != nil {
		return ggufMeta{}, fmt.Errorf("read n_kv: %w", err)
	}

	var meta ggufMeta
	for i := uint64(0); i < nKV; i++ {
		key, err := r.string()
		if err != nil {
			return ggufMeta{}, fmt.Errorf("read KV key %d: %w", i, err)
		}
		vtRaw, err := r.uint32()
		if err != nil {
			return ggufMeta{}, fmt.Errorf("read KV type for %q: %w", key, err)
		}
		val, err := r.value(ggufType(vtRaw))
		if err != nil {
			return ggufMeta{}, fmt.Errorf("read KV value for %q: %w", key, err)
		}
		applyMeta(&meta, key, val)
	}
	meta.deriveSWA()
	return meta, nil
}

// deriveSWA populates the sliding-window attention fields from the per-layer
// head_count_kv array and the sliding_window_pattern, both gathered during the
// KV scan. The derivation requires that both arrays are present and their
// lengths match the block count; otherwise the model is treated as fully global
// attention (slidingWindowSize cleared to zero) so estimation falls back to the
// conservative all-layers formula. hasSlidingWindow is left untouched so callers
// can detect a failed derivation (key present but fields not derived).
func (meta *ggufMeta) deriveSWA() {
	if !meta.hasSlidingWindow {
		return
	}
	n := int(meta.layerCount)
	if n == 0 || len(meta.slidingWindowPattern) != n || len(meta.kvHeadCountArray) != n {
		meta.slidingWindowSize = 0
		return
	}
	var globalCount, swaCount, globalHeads, swaHeads uint32
	for i := 0; i < n; i++ {
		h := meta.kvHeadCountArray[i]
		if meta.slidingWindowPattern[i] {
			swaCount++
			if h > swaHeads {
				swaHeads = h
			}
		} else {
			globalCount++
			if h > globalHeads {
				globalHeads = h
			}
		}
	}
	meta.globalLayerCount = globalCount
	meta.swaLayerCount = swaCount
	meta.globalKVHeadCount = globalHeads
	meta.swaKVHeadCount = swaHeads
}

// applyMeta extracts known fields from a single KV pair and stores them in meta.
func applyMeta(meta *ggufMeta, key string, val any) {
	switch key {
	case "general.architecture":
		if s, ok := val.(string); ok {
			meta.architecture = s
			meta.hasArchitecture = true
			if s == "clip" {
				meta.isMmproj = true
			}
		}
	case "general.file_type":
		if v, ok := toUint32(val); ok {
			meta.fileType = v
			meta.hasFileType = true
		}
	default:
		// Arch-prefixed keys: "arch.field" or "arch.sub.field".
		// Match on the suffix after the first dot segment.
		dot := strings.IndexByte(key, '.')
		if dot < 0 {
			return
		}
		suffix := key[dot+1:]
		switch suffix {
		case "block_count":
			if v, ok := toUint32(val); ok {
				meta.layerCount = v
				meta.hasLayerCount = true
			}
		case "attention.head_count":
			if v, ok := toUint32(val); ok {
				meta.headCount = v
				meta.hasHeadCount = true
			}
		case "attention.head_count_kv":
			if v, ok := toUint32(val); ok {
				meta.kvHeadCount = v
				meta.hasKVHeadCount = true
			} else if arr, ok := val.([]any); ok && len(arr) > 0 {
				// Some architectures (e.g. Gemma 4) store per-layer kv head counts as an
				// array. Use the maximum value for a conservative memory estimate, and
				// retain the full array for sliding-window attention derivation.
				if v, ok := maxUint32FromArray(arr); ok {
					meta.kvHeadCount = v
					meta.hasKVHeadCount = true
				}
				meta.kvHeadCountArray = uint32SliceFromArray(arr)
			}
		case "attention.key_length":
			if v, ok := toUint32(val); ok {
				meta.headDim = v
			}
		case "attention.key_length_swa":
			if v, ok := toUint32(val); ok {
				meta.swaHeadDim = v
			}
		case "attention.sliding_window":
			if v, ok := toUint32(val); ok {
				meta.slidingWindowSize = v
				meta.hasSlidingWindow = true
			}
		case "attention.sliding_window_pattern":
			if arr, ok := val.([]any); ok {
				meta.slidingWindowPattern = boolSliceFromArray(arr)
			}
		case "embedding_length":
			if v, ok := toUint32(val); ok {
				meta.embedLen = v
			}
		case "context_length":
			if v, ok := toUint32(val); ok {
				meta.maxContext = v
				meta.hasMaxContext = true
			}
		}
	}
}

// maxUint32FromArray returns the maximum uint32-coercible value across a []any slice.
func maxUint32FromArray(arr []any) (uint32, bool) {
	var max uint32
	found := false
	for _, item := range arr {
		if v, ok := toUint32(item); ok {
			if !found || v > max {
				max = v
				found = true
			}
		}
	}
	return max, found
}

// uint32SliceFromArray converts a []any of GGUF numeric values into a []uint32.
// Elements that do not coerce to uint32 are stored as zero, preserving positional
// alignment with the sliding-window pattern.
func uint32SliceFromArray(arr []any) []uint32 {
	out := make([]uint32, len(arr))
	for i, item := range arr {
		if v, ok := toUint32(item); ok {
			out[i] = v
		}
	}
	return out
}

// boolSliceFromArray converts a []any of GGUF bool values into a []bool.
// Non-bool elements are stored as false.
func boolSliceFromArray(arr []any) []bool {
	out := make([]bool, len(arr))
	for i, item := range arr {
		if b, ok := item.(bool); ok {
			out[i] = b
		}
	}
	return out
}

// toUint32 coerces a GGUF numeric value to uint32 when it fits.
func toUint32(v any) (uint32, bool) {
	switch x := v.(type) {
	case uint32:
		return x, true
	case uint64:
		if x <= 0xffffffff {
			return uint32(x), true
		}
	case uint8:
		return uint32(x), true
	case uint16:
		return uint32(x), true
	case int32:
		if x >= 0 {
			return uint32(x), true
		}
	}
	return 0, false
}

var ggufFileTypeNames = map[uint32]string{
	0: "F32", 1: "F16",
	2: "Q4_0", 3: "Q4_1",
	7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
	10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
	14: "Q4_K_S", 15: "Q4_K_M",
	16: "Q5_K_S", 17: "Q5_K_M",
	18: "Q6_K",
	19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
	22: "IQ3_XS", 23: "IQ3_XXS",
	24: "IQ1_S", 25: "IQ4_NL",
	26: "IQ3_S", 27: "IQ3_M",
	28: "IQ2_S", 29: "IQ2_M",
	30: "IQ4_XS", 31: "IQ1_M",
	32: "BF16",
	33: "Q4_0_4_4", 34: "Q4_0_4_8", 35: "Q4_0_8_8",
	36: "TQ1_0", 37: "TQ2_0",
}

// fileTypeString converts a GGUF general.file_type value to a human-readable quantization name.
func fileTypeString(ft uint32) string {
	if s, ok := ggufFileTypeNames[ft]; ok {
		return s
	}
	return fmt.Sprintf("unknown(%d)", ft)
}
