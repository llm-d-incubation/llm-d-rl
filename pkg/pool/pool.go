// Package pool provides the unified engine pool abstraction for the llm-d
// rollout controller. A Pool owns both generation routing and weight
// synchronization. PickEngine() returns the router URL when one is configured,
// or picks the first ready engine from the pool otherwise.
package pool

import (
	"context"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
)


// Pool is the unified interface for an engine pool.
// It owns both generation routing and weight synchronization.
type Pool interface {
	// PickEngine returns the base URL to send the next generation request to.
	// If a router URL is configured it always returns that URL.
	// Otherwise it returns the address of the first ready engine in the pool.
	PickEngine() (string, error)

	// TokensIn reports whether this pool expects token-ID arrays as input (true)
	// or text strings (false). Callers use this to build the correct OAI request body.
	TokensIn() bool

	// AddEngine registers a new engine pod. Called by Populators.
	AddEngine(id, address string)

	// RemoveEngine deregisters an engine pod. Called by Populators.
	RemoveEngine(id string)

	// --- Weight sync (identical for all pool configurations) ---

	InitWeightTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit) error
	UpdateWeights(ctx context.Context, req *v1alpha1.WeightUpdateRequest) error
	PauseAll(ctx context.Context, mode v1alpha1.PauseMode) error
	ResumeAll(ctx context.Context) error
	SleepAll(ctx context.Context, level v1alpha1.SleepLevel) error
	WakeUpAll(ctx context.Context, tags []string) error
	WeightVersion() int64

	// --- Observability ---

	Status() v1alpha1.PoolStatus
	SetPhase(phase v1alpha1.PoolPhase)
	StartHealthLoop(ctx context.Context)
}
