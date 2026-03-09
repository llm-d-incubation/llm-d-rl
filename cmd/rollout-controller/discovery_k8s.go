//go:build k8s

package main

import (
	"context"
	"flag"

	"github.com/llm-d/llm-d-rl/pkg/discovery"
)

type discoveryConfig = discovery.Config
type engineCallbacks = discovery.EngineCallbacks

func newDiscoveryConfig() *discoveryConfig                          { return &discovery.Config{} }
func bindDiscoveryFlags(cfg *discoveryConfig, fs *flag.FlagSet)     { cfg.BindFlags(fs) }
func discoveryEnabled(cfg *discoveryConfig) bool                    { return cfg.Enabled() }
func startDiscovery(ctx context.Context, cfg *discoveryConfig, cb engineCallbacks) error {
	return cfg.Start(ctx, cb)
}
