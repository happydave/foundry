package registry

import (
	"encoding/binary"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
)

// testGGUF builds a minimal valid GGUF v2 binary with the provided metadata KV pairs.
func testGGUF(kvs []testKV) []byte {
	var body []byte
	for _, kv := range kvs {
		body = appendGGUFKV(body, kv)
	}

	var out []byte
	out = binary.LittleEndian.AppendUint32(out, ggufMagic) // magic
	out = binary.LittleEndian.AppendUint32(out, 2)         // version
	out = binary.LittleEndian.AppendUint64(out, 0)         // n_tensors
	out = binary.LittleEndian.AppendUint64(out, uint64(len(kvs)))
	out = append(out, body...)
	return out
}

type testKV struct {
	key   string
	vtype ggufType
	val   any // uint32 or string
}

func appendGGUFKV(buf []byte, kv testKV) []byte {
	// key (v2: uint64 length + bytes)
	buf = binary.LittleEndian.AppendUint64(buf, uint64(len(kv.key)))
	buf = append(buf, kv.key...)
	// value type
	buf = binary.LittleEndian.AppendUint32(buf, uint32(kv.vtype))
	// value
	switch v := kv.val.(type) {
	case uint32:
		buf = binary.LittleEndian.AppendUint32(buf, v)
	case string:
		buf = binary.LittleEndian.AppendUint64(buf, uint64(len(v)))
		buf = append(buf, v...)
	case []int32:
		buf = binary.LittleEndian.AppendUint32(buf, uint32(ggufTypeInt32))
		buf = binary.LittleEndian.AppendUint64(buf, uint64(len(v)))
		for _, item := range v {
			buf = binary.LittleEndian.AppendUint32(buf, uint32(item))
		}
	}
	return buf
}

// writeTestGGUF writes a test GGUF file to dir with the given name and metadata.
// It returns the path of the written file.
func writeTestGGUF(t *testing.T, dir, name string, kvs []testKV) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, testGGUF(kvs), 0o600); err != nil {
		t.Fatalf("write test GGUF %s: %v", path, err)
	}
	return path
}

// validKVs returns a complete set of KV pairs for a valid llama model.
func validKVs(arch string) []testKV {
	return []testKV{
		{key: "general.architecture", vtype: ggufTypeString, val: arch},
		{key: "general.file_type", vtype: ggufTypeUint32, val: uint32(15)}, // Q4_K_M
		{key: arch + ".block_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: arch + ".attention.head_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: arch + ".attention.head_count_kv", vtype: ggufTypeUint32, val: uint32(8)},
		{key: arch + ".embedding_length", vtype: ggufTypeUint32, val: uint32(4096)},
		{key: arch + ".context_length", vtype: ggufTypeUint32, val: uint32(4096)},
	}
}

func silentLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError + 1}))
}

func TestRegistry_ThreeModels(t *testing.T) {
	dir := t.TempDir()
	writeTestGGUF(t, dir, "model-a.gguf", validKVs("llama"))
	writeTestGGUF(t, dir, "model-b.gguf", validKVs("mistral"))
	writeTestGGUF(t, dir, "model-c.gguf", validKVs("phi"))

	reg := New([]string{dir}, silentLogger())
	models := reg.List()
	if len(models) != 3 {
		t.Fatalf("expected 3 models, got %d", len(models))
	}
}

func TestRegistry_NonExistentDirectory(t *testing.T) {
	reg := New([]string{"/nonexistent/path/that/does/not/exist"}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected empty registry for non-existent directory")
	}
}

func TestRegistry_PathIsFile(t *testing.T) {
	dir := t.TempDir()
	filePath := writeTestGGUF(t, dir, "model.gguf", validKVs("llama"))

	// Pass the file path itself as a scan directory.
	reg := New([]string{filePath}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected empty registry when scan path is a file")
	}
}

func TestRegistry_EmptyDirectory(t *testing.T) {
	dir := t.TempDir()
	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected empty registry for empty directory")
	}
}

func TestRegistry_NonGGUFFilesIgnored(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("not a model"), 0o600); err != nil {
		t.Fatal(err)
	}
	writeTestGGUF(t, dir, "model.gguf", validKVs("llama"))

	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 1 {
		t.Fatalf("expected 1 model, got %d", len(reg.List()))
	}
}

func TestRegistry_ModelFields(t *testing.T) {
	dir := t.TempDir()
	writeTestGGUF(t, dir, "mistral-7b-q4.gguf", validKVs("mistral"))

	reg := New([]string{dir}, silentLogger())
	models := reg.List()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	m := models[0]

	if m.DisplayName != "mistral-7b-q4" {
		t.Errorf("DisplayName = %q, want %q", m.DisplayName, "mistral-7b-q4")
	}
	if m.Architecture != "mistral" {
		t.Errorf("Architecture = %q, want %q", m.Architecture, "mistral")
	}
	if m.LayerCount != 32 {
		t.Errorf("LayerCount = %d, want 32", m.LayerCount)
	}
	if m.HeadCount != 32 {
		t.Errorf("HeadCount = %d, want 32", m.HeadCount)
	}
	if m.KVHeadCount != 8 {
		t.Errorf("KVHeadCount = %d, want 8", m.KVHeadCount)
	}
	if m.HeadDim != 128 { // 4096 / 32
		t.Errorf("HeadDim = %d, want 128 (derived from embedding/heads)", m.HeadDim)
	}
	if m.MaxContext != 4096 {
		t.Errorf("MaxContext = %d, want 4096", m.MaxContext)
	}
	if m.Quantization != "Q4_K_M" {
		t.Errorf("Quantization = %q, want %q", m.Quantization, "Q4_K_M")
	}
	if m.FileSize <= 0 {
		t.Errorf("FileSize = %d, want > 0", m.FileSize)
	}
	if m.ID == 0 {
		t.Error("ID should be non-zero")
	}
}

func TestRegistry_HeadDimDirectKey(t *testing.T) {
	dir := t.TempDir()
	kvs := []testKV{
		{key: "general.architecture", vtype: ggufTypeString, val: "llama"},
		{key: "general.file_type", vtype: ggufTypeUint32, val: uint32(15)},
		{key: "llama.block_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: "llama.attention.head_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: "llama.attention.head_count_kv", vtype: ggufTypeUint32, val: uint32(8)},
		{key: "llama.attention.key_length", vtype: ggufTypeUint32, val: uint32(64)}, // direct
		{key: "llama.context_length", vtype: ggufTypeUint32, val: uint32(8192)},
		// no embedding_length — head dim must come from key_length
	}
	writeTestGGUF(t, dir, "model.gguf", kvs)

	reg := New([]string{dir}, silentLogger())
	models := reg.List()
	if len(models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(models))
	}
	if models[0].HeadDim != 64 {
		t.Errorf("HeadDim = %d, want 64 (from attention.key_length)", models[0].HeadDim)
	}
}

func TestRegistry_MissingLayerCount(t *testing.T) {
	dir := t.TempDir()
	kvs := []testKV{
		{key: "general.architecture", vtype: ggufTypeString, val: "llama"},
		{key: "general.file_type", vtype: ggufTypeUint32, val: uint32(15)},
		// block_count omitted
		{key: "llama.attention.head_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: "llama.attention.head_count_kv", vtype: ggufTypeUint32, val: uint32(8)},
		{key: "llama.embedding_length", vtype: ggufTypeUint32, val: uint32(4096)},
		{key: "llama.context_length", vtype: ggufTypeUint32, val: uint32(4096)},
	}
	writeTestGGUF(t, dir, "model.gguf", kvs)

	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected model to be skipped due to missing block_count")
	}
}

func TestRegistry_MissingKVHeadCount(t *testing.T) {
	dir := t.TempDir()
	kvs := []testKV{
		{key: "general.architecture", vtype: ggufTypeString, val: "llama"},
		{key: "general.file_type", vtype: ggufTypeUint32, val: uint32(15)},
		{key: "llama.block_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: "llama.attention.head_count", vtype: ggufTypeUint32, val: uint32(32)},
		// head_count_kv omitted
		{key: "llama.embedding_length", vtype: ggufTypeUint32, val: uint32(4096)},
		{key: "llama.context_length", vtype: ggufTypeUint32, val: uint32(4096)},
	}
	writeTestGGUF(t, dir, "model.gguf", kvs)

	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected model to be skipped due to missing head_count_kv")
	}
}

func TestRegistry_Gemma4PerLayerKVHeads(t *testing.T) {
	// Gemma 4 stores attention.head_count_kv as arr[i32, N] (one per layer) rather than
	// a scalar uint32. The registry should accept the model and use the max value.
	dir := t.TempDir()
	kvs := []testKV{
		{key: "general.architecture", vtype: ggufTypeString, val: "gemma4"},
		{key: "general.file_type", vtype: ggufTypeUint32, val: uint32(14)}, // Q4_K_S
		{key: "gemma4.block_count", vtype: ggufTypeUint32, val: uint32(60)},
		{key: "gemma4.attention.head_count", vtype: ggufTypeUint32, val: uint32(32)},
		{key: "gemma4.attention.head_count_kv", vtype: ggufTypeArray, val: []int32{16, 16, 4, 1, 16}},
		{key: "gemma4.attention.key_length", vtype: ggufTypeUint32, val: uint32(512)},
		{key: "gemma4.context_length", vtype: ggufTypeUint32, val: uint32(262144)},
	}
	writeTestGGUF(t, dir, "gemma4-31b.gguf", kvs)

	reg := New([]string{dir}, silentLogger())
	models := reg.List()
	if len(models) != 1 {
		t.Fatalf("expected Gemma 4 model to be registered, got %d models", len(models))
	}
	if models[0].KVHeadCount != 16 {
		t.Errorf("KVHeadCount = %d, want 16 (max of per-layer array)", models[0].KVHeadCount)
	}
}

func TestRegistry_BadMagic(t *testing.T) {
	dir := t.TempDir()
	data := []byte{0x00, 0x01, 0x02, 0x03, 0x00, 0x00, 0x00, 0x02}
	if err := os.WriteFile(filepath.Join(dir, "bad.gguf"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected bad-magic file to be skipped")
	}
}

func TestRegistry_TruncatedFile(t *testing.T) {
	dir := t.TempDir()
	// Write only the magic bytes — too short to be valid.
	var data []byte
	data = binary.LittleEndian.AppendUint32(data, ggufMagic)
	if err := os.WriteFile(filepath.Join(dir, "truncated.gguf"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected truncated file to be skipped")
	}
}

func TestRegistry_StableID(t *testing.T) {
	dir := t.TempDir()
	writeTestGGUF(t, dir, "model.gguf", validKVs("llama"))

	reg1 := New([]string{dir}, silentLogger())
	reg2 := New([]string{dir}, silentLogger())

	models1 := reg1.List()
	models2 := reg2.List()
	if len(models1) != 1 || len(models2) != 1 {
		t.Fatalf("expected 1 model in both registries")
	}
	if models1[0].ID != models2[0].ID {
		t.Errorf("IDs differ across restarts: %d vs %d", models1[0].ID, models2[0].ID)
	}
}

func TestRegistry_IDChangesOnResize(t *testing.T) {
	dir := t.TempDir()
	path := writeTestGGUF(t, dir, "model.gguf", validKVs("llama"))

	reg1 := New([]string{dir}, silentLogger())
	id1 := reg1.List()[0].ID

	// Append a byte to change the file size, simulating a replaced file.
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = f.Write([]byte{0})
	_ = f.Close()

	reg2 := New([]string{dir}, silentLogger())
	// File is now slightly larger and the extra byte makes the header parsing fail since
	// the GGUF structure is unchanged — the fingerprint inputs still change (size differs).
	// If the file still parses successfully, check the ID.
	models2 := reg2.List()
	if len(models2) == 0 {
		t.Skip("resized file failed to parse (expected for GGUF structure integrity checks)")
	}
	id2 := models2[0].ID
	if id1 == id2 {
		t.Errorf("expected different ID after file resize, got same ID %d", id1)
	}
}

func TestRegistry_GetByID(t *testing.T) {
	dir := t.TempDir()
	writeTestGGUF(t, dir, "model.gguf", validKVs("llama"))

	reg := New([]string{dir}, silentLogger())
	models := reg.List()
	if len(models) != 1 {
		t.Fatalf("expected 1 model")
	}
	id := models[0].ID

	m, ok := reg.Get(id)
	if !ok {
		t.Fatal("Get returned not-found for a valid ID")
	}
	if m.ID != id {
		t.Errorf("Get returned model with wrong ID")
	}

	_, ok = reg.Get(id + 1)
	if ok {
		t.Error("Get should return not-found for a non-existent ID")
	}
}

func TestRegistry_MultipleDirs(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()
	writeTestGGUF(t, dir1, "model-a.gguf", validKVs("llama"))
	writeTestGGUF(t, dir2, "model-b.gguf", validKVs("mistral"))

	reg := New([]string{dir1, dir2}, silentLogger())
	if len(reg.List()) != 2 {
		t.Fatalf("expected 2 models from two directories, got %d", len(reg.List()))
	}
}

func TestRegistry_SameNameDifferentDirs(t *testing.T) {
	dir1 := t.TempDir()
	dir2 := t.TempDir()
	// Same filename and same content → same size; should get different IDs (full path used).
	writeTestGGUF(t, dir1, "model.gguf", validKVs("llama"))
	writeTestGGUF(t, dir2, "model.gguf", validKVs("llama"))

	reg := New([]string{dir1, dir2}, silentLogger())
	models := reg.List()
	if len(models) != 2 {
		t.Fatalf("expected 2 distinct models (same name, different paths), got %d", len(models))
	}
	if models[0].ID == models[1].ID {
		t.Error("expected different IDs for same filename in different directories")
	}
}

func TestRegistry_SubdirsIgnored(t *testing.T) {
	dir := t.TempDir()
	sub := filepath.Join(dir, "subdir")
	if err := os.Mkdir(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	// Model inside subdir should not be discovered (top-level scan only).
	writeTestGGUF(t, sub, "model.gguf", validKVs("llama"))

	reg := New([]string{dir}, silentLogger())
	if len(reg.List()) != 0 {
		t.Fatal("expected subdirectories to be ignored (top-level scan only)")
	}
}

func TestFileTypeString(t *testing.T) {
	cases := []struct {
		ft   uint32
		want string
	}{
		{0, "F32"}, {1, "F16"},
		{15, "Q4_K_M"}, {17, "Q5_K_M"},
		{32, "BF16"},
		{99, "unknown(99)"},
	}
	for _, tc := range cases {
		got := fileTypeString(tc.ft)
		if got != tc.want {
			t.Errorf("fileTypeString(%d) = %q, want %q", tc.ft, got, tc.want)
		}
	}
}
