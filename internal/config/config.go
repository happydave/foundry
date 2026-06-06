package config

import (
	"errors"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

type Config struct {
	ModelScanPaths        []string `yaml:"model_scan_paths"`
	LlamaServerBinary     string   `yaml:"llama_server_binary"`
	LlamaServerExtraArgs  []string `yaml:"llama_server_extra_args"`
	DefaultGPULayers      int      `yaml:"default_gpu_layers"`
	KVCacheType           string   `yaml:"kv_cache_type"`
	HistorySessionsDir    string   `yaml:"history_sessions_dir"`
	ListenAddress         string   `yaml:"listen_address"`
	LogLevel              string   `yaml:"log_level"`
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
	if c.KVCacheType == "" {
		errs = append(errs, errors.New("kv_cache_type is required"))
	}
	if c.HistorySessionsDir == "" {
		errs = append(errs, errors.New("history_sessions_dir is required"))
	}
	return errors.Join(errs...)
}
