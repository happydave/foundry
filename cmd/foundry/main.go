package main

import (
	"context"
	"flag"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/happydave/foundry/internal/config"
	"github.com/happydave/foundry/internal/estimator"
	"github.com/happydave/foundry/internal/history"
	"github.com/happydave/foundry/internal/processmanager"
	"github.com/happydave/foundry/internal/registry"
	"github.com/happydave/foundry/internal/server"
)

// knownLlamaServerVersions lists the llama-server --version substrings accepted
// at startup. Add new entries here when upgrading the binary.
var knownLlamaServerVersions = []string{
	"version: 9536 (308f61c31)",
}

func main() {
	configPath := flag.String("config", "foundry.yaml", "path to YAML config file")
	flag.Parse()

	cfg, err := config.Load(*configPath)
	if err != nil {
		slog.New(slog.NewJSONHandler(os.Stdout, nil)).Error("startup failed", "error", err.Error())
		os.Exit(1)
	}

	var level slog.Level
	if err := level.UnmarshalText([]byte(cfg.LogLevel)); err != nil {
		level = slog.LevelInfo
	}
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(logger)

	logger.Info("foundry starting", "listen_address", cfg.ListenAddress, "log_level", cfg.LogLevel)

	if err := processmanager.CheckBinaryVersion(cfg.LlamaServerBinary, knownLlamaServerVersions); err != nil {
		logger.Error("llama-server version check failed", "error", err.Error())
		os.Exit(1)
	}

	reg := registry.New(cfg.ModelScanPaths, logger)
	pm := processmanager.New(cfg.LlamaServerBinary, cfg.LlamaServerExtraArgs, logger)
	est := estimator.New(estimator.Params{})

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, os.Interrupt)
	defer stop()

	resolvedOpts := make(map[string]processmanager.ModelLoadOptions, len(cfg.Models))
	for name, mc := range cfg.Models {
		kvType := mc.KVCacheType
		if kvType == "" {
			kvType = cfg.KVCacheType
		}
		parallel := mc.Parallel
		if parallel == 0 {
			parallel = cfg.Parallel
		}
		opts := processmanager.ModelLoadOptions{KVCacheType: kvType, Parallel: parallel}
		if f := strings.TrimSpace(mc.ChatTemplateFile); f != "" {
			opts.Args = []string{"--chat-template-file", f}
		} else if t := strings.TrimSpace(mc.ChatTemplate); t != "" {
			opts.Args = []string{"--chat-template", t}
		}
		resolvedOpts[name] = opts
	}

	srv := server.New(cfg.ListenAddress, reg, pm, est, cfg.DefaultGPULayers, cfg.KVCacheType, cfg.Parallel, resolvedOpts, logger)

	if fi, err := os.Stat(cfg.HistorySessionsDir); err != nil || !fi.IsDir() {
		logger.Warn("history_sessions_dir does not exist or is not a directory; persistent session history is disabled",
			"path", cfg.HistorySessionsDir)
	} else {
		srv.SetHistoryStore(history.NewJSONLStore(cfg.HistorySessionsDir))
		logger.Info("session history enabled", "dir", cfg.HistorySessionsDir)
	}

	if err := srv.ListenAndServe(ctx); err != nil {
		logger.Error("server error", "error", err.Error())
		os.Exit(1)
	}

	if err := pm.UnloadAll(context.Background()); err != nil {
		logger.Error("error during shutdown unload", "error", err.Error())
	}

	logger.Info("process exiting")
}
