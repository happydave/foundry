package config

import (
	"errors"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// validKVCacheTypes lists the supported KV cache element types.
var validKVCacheTypes = map[string]bool{
	"f32":    true,
	"f16":    true,
	"bf16":   true,
	"q8_0":   true,
	"q4_0":   true,
	"q4_1":   true,
	"iq4_nl": true,
	"q5_0":   true,
	"q5_1":   true,
}

// ModelConfig holds per-model overrides. Keys in the top-level Models map are
// model DisplayNames (GGUF filename without the .gguf extension).
type ModelConfig struct {
	ChatTemplate     string `yaml:"chat_template"`
	ChatTemplateFile string `yaml:"chat_template_file"`
	KVCacheType      string `yaml:"kv_cache_type"`
}

type Config struct {
	ModelScanPaths       []string               `yaml:"model_scan_paths"`
	LlamaServerBinary    string                 `yaml:"llama_server_binary"`
	LlamaServerExtraArgs []string               `yaml:"llama_server_extra_args"`
	DefaultGPULayers     int                    `yaml:"default_gpu_layers"`
	KVCacheType          string                 `yaml:"kv_cache_type"`
	HistorySessionsDir   string                 `yaml:"history_sessions_dir"`
	ListenAddress        string                 `yaml:"listen_address"`
	LogLevel             string                 `yaml:"log_level"`
	Models               map[string]ModelConfig `yaml:"models"`
}

func Load(path string) (*Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cannot open config file %q: %w", path, err)
	}
	defer func() { _ = f.Close() }()

	var cfg Config
	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("cannot parse config file %q: %w", path, err)
	}

	if err := cfg.validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	if cfg.ListenAddress == "" {
		cfg.ListenAddress = "0.0.0.0:3456"
	}
	if cfg.LogLevel == "" {
		cfg.LogLevel = "info"
	}
	if cfg.KVCacheType == "" {
		cfg.KVCacheType = "q8_0"
	}

	return &cfg, nil
}

func (c *Config) validate() error {
	var errs []error
	if len(c.ModelScanPaths) == 0 {
		errs = append(errs, errors.New("model_scan_paths is required"))
	}
	if c.LlamaServerBinary == "" {
		errs = append(errs, errors.New("llama_server_binary is required"))
	}
	if c.HistorySessionsDir == "" {
		errs = append(errs, errors.New("history_sessions_dir is required"))
	}
	if c.KVCacheType != "" && !validKVCacheTypes[c.KVCacheType] {
		errs = append(errs, fmt.Errorf("kv_cache_type %q is not supported; must be one of: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1", c.KVCacheType))
	}
	for name, mc := range c.Models {
		if strings.TrimSpace(mc.ChatTemplate) != "" && strings.TrimSpace(mc.ChatTemplateFile) != "" {
			errs = append(errs, fmt.Errorf("model %q: chat_template and chat_template_file are mutually exclusive", name))
		}
		if mc.KVCacheType != "" && !validKVCacheTypes[mc.KVCacheType] {
			errs = append(errs, fmt.Errorf("model %q: kv_cache_type %q is not supported; must be one of: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1", name, mc.KVCacheType))
		}
	}
	return errors.Join(errs...)
}
