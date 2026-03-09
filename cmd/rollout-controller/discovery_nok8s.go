//go:build !k8s

package main

import (
	"context"
	"errors"
	"flag"
)

type discoveryConfig struct{}

type engineCallbacks struct {
	OnReady func(id, address string)
	OnGone  func(id string)
}

func newDiscoveryConfig() *discoveryConfig                      { return &discoveryConfig{} }
func bindDiscoveryFlags(_ *discoveryConfig, _ *flag.FlagSet)    {}
func discoveryEnabled(_ *discoveryConfig) bool                  { return false }
func startDiscovery(_ context.Context, _ *discoveryConfig, _ engineCallbacks) error {
	return errors.New("kubernetes discovery requires building with -tags k8s")
}
