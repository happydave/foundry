package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeTemp(t *testing.T, content string) string {
	t.Helper()
	f, err := os.CreateTemp(t.TempDir(), "config-*.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	return f.Name()
}

func TestLoad_Valid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/models]
llama_server_binary: /usr/bin/llama-server
default_gpu_layers: 32
kv_cache_type: f16
history_sessions_dir: /var/foundry/sessions
listen_address: "127.0.0.1:9090"
log_level: debug
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.ListenAddress != "127.0.0.1:9090" {
		t.Errorf("ListenAddress = %q, want 127.0.0.1:9090", cfg.ListenAddress)
	}
	if cfg.LogLevel != "debug" {
		t.Errorf("LogLevel = %q, want debug", cfg.LogLevel)
	}
}

func TestLoad_Defaults(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/models]
llama_server_binary: /usr/bin/llama-server
history_sessions_dir: /var/foundry/sessions
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.ListenAddress != "0.0.0.0:3456" {
		t.Errorf("ListenAddress = %q, want 0.0.0.0:3456", cfg.ListenAddress)
	}
	if cfg.LogLevel != "info" {
		t.Errorf("LogLevel = %q, want info", cfg.LogLevel)
	}
	if cfg.KVCacheType != "q8_0" {
		t.Errorf("KVCacheType = %q, want q8_0 (default)", cfg.KVCacheType)
	}
}

func TestLoad_MissingFile(t *testing.T) {
	_, err := Load(filepath.Join(t.TempDir(), "nonexistent.yaml"))
	if err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestLoad_MissingRequiredFields(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "missing model_scan_paths",
			content: "llama_server_binary: /bin/llama\nhistory_sessions_dir: /s\n",
			want:    "model_scan_paths",
		},
		{
			name:    "missing llama_server_binary",
			content: "model_scan_paths: [/m]\nhistory_sessions_dir: /s\n",
			want:    "llama_server_binary",
		},
		{
			name:    "missing history_sessions_dir",
			content: "model_scan_paths: [/m]\nllama_server_binary: /bin/llama\n",
			want:    "history_sessions_dir",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeTemp(t, tc.content)
			_, err := Load(path)
			if err == nil {
				t.Fatal("expected validation error, got nil")
			}
			if got := err.Error(); !strings.Contains(got, tc.want) {
				t.Errorf("error %q does not mention field %q", got, tc.want)
			}
		})
	}
}

func TestLoad_KVCacheType_Absent_DefaultsToQ8_0(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.KVCacheType != "q8_0" {
		t.Errorf("KVCacheType = %q, want q8_0 when absent", cfg.KVCacheType)
	}
}

func TestLoad_KVCacheType_Invalid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
kv_cache_type: garbage
`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for unsupported kv_cache_type, got nil")
	}
	if got := err.Error(); !strings.Contains(got, "garbage") {
		t.Errorf("error %q does not mention the invalid value %q", got, "garbage")
	}
	if got := err.Error(); !strings.Contains(got, "kv_cache_type") {
		t.Errorf("error %q does not mention kv_cache_type", got)
	}
}

func TestLoad_KVCacheType_NewTypesValid(t *testing.T) {
	for _, kvType := range []string{"q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"} {
		t.Run(kvType, func(t *testing.T) {
			path := writeTemp(t, fmt.Sprintf(`
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
kv_cache_type: %s
`, kvType))
			cfg, err := Load(path)
			if err != nil {
				t.Fatalf("unexpected error for kv_cache_type %q: %v", kvType, err)
			}
			if cfg.KVCacheType != kvType {
				t.Errorf("KVCacheType = %q, want %q", cfg.KVCacheType, kvType)
			}
		})
	}
}

func TestLoad_UnparsableYAML(t *testing.T) {
	path := writeTemp(t, "this: is: not: valid: yaml: {{{{")
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
}

func TestLoad_ModelConfig_MutualExclusion(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    chat_template: "hello"
    chat_template_file: /some/path.jinja
`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for mutually exclusive fields, got nil")
	}
	if got := err.Error(); !strings.Contains(got, "mutually exclusive") {
		t.Errorf("error %q does not mention mutually exclusive", got)
	}
	if got := err.Error(); !strings.Contains(got, "my-model") {
		t.Errorf("error %q does not identify the model name", got)
	}
}

func TestLoad_ModelConfig_OnlyChatTemplate_Valid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    chat_template: "some template"
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	mc, ok := cfg.Models["my-model"]
	if !ok {
		t.Fatal("model config not found")
	}
	if mc.ChatTemplate != "some template" {
		t.Errorf("ChatTemplate = %q, want %q", mc.ChatTemplate, "some template")
	}
}

func TestLoad_ModelConfig_OnlyChatTemplateFile_Valid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    chat_template_file: /path/to/template.jinja
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Models["my-model"].ChatTemplateFile != "/path/to/template.jinja" {
		t.Errorf("ChatTemplateFile = %q", cfg.Models["my-model"].ChatTemplateFile)
	}
}

func TestLoad_ModelConfig_BothEmpty_Valid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    chat_template: "   "
    chat_template_file: "   "
`)
	_, err := Load(path)
	if err != nil {
		t.Fatalf("whitespace-only fields should not trigger mutual exclusion error: %v", err)
	}
}

func TestLoad_ModelConfig_UnknownField_Rejected(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    unknown_field: value
`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for unknown field in model config, got nil")
	}
}

func TestLoad_ModelConfig_KVCacheType_Valid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
kv_cache_type: q8_0
models:
  my-model:
    kv_cache_type: f16
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Models["my-model"].KVCacheType != "f16" {
		t.Errorf("per-model KVCacheType = %q, want f16", cfg.Models["my-model"].KVCacheType)
	}
	if cfg.KVCacheType != "q8_0" {
		t.Errorf("global KVCacheType = %q, want q8_0", cfg.KVCacheType)
	}
}

func TestLoad_ModelConfig_KVCacheType_Invalid(t *testing.T) {
	path := writeTemp(t, `
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    kv_cache_type: garbage
`)
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected error for unsupported per-model kv_cache_type, got nil")
	}
	if got := err.Error(); !strings.Contains(got, "my-model") {
		t.Errorf("error %q does not identify the model name", got)
	}
	if got := err.Error(); !strings.Contains(got, "garbage") {
		t.Errorf("error %q does not mention the invalid value", got)
	}
}

func TestLoad_ModelConfig_KVCacheType_NewTypesValid(t *testing.T) {
	for _, kvType := range []string{"q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"} {
		t.Run(kvType, func(t *testing.T) {
			path := writeTemp(t, fmt.Sprintf(`
model_scan_paths: [/m]
llama_server_binary: /bin/llama
history_sessions_dir: /s
models:
  my-model:
    kv_cache_type: %s
`, kvType))
			cfg, err := Load(path)
			if err != nil {
				t.Fatalf("unexpected error for per-model kv_cache_type %q: %v", kvType, err)
			}
			if cfg.Models["my-model"].KVCacheType != kvType {
				t.Errorf("per-model KVCacheType = %q, want %q", cfg.Models["my-model"].KVCacheType, kvType)
			}
		})
	}
}
