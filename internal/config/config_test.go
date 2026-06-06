package config

import (
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
kv_cache_type: f16
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
			content: "llama_server_binary: /bin/llama\nkv_cache_type: f16\nhistory_sessions_dir: /s\n",
			want:    "model_scan_paths",
		},
		{
			name:    "missing llama_server_binary",
			content: "model_scan_paths: [/m]\nkv_cache_type: f16\nhistory_sessions_dir: /s\n",
			want:    "llama_server_binary",
		},
		{
			name:    "missing kv_cache_type",
			content: "model_scan_paths: [/m]\nllama_server_binary: /bin/llama\nhistory_sessions_dir: /s\n",
			want:    "kv_cache_type",
		},
		{
			name:    "missing history_sessions_dir",
			content: "model_scan_paths: [/m]\nllama_server_binary: /bin/llama\nkv_cache_type: f16\n",
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

func TestLoad_UnparsableYAML(t *testing.T) {
	path := writeTemp(t, "this: is: not: valid: yaml: {{{{")
	_, err := Load(path)
	if err == nil {
		t.Fatal("expected parse error, got nil")
	}
}
