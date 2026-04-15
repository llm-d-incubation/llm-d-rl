package weightsync

import (
	"context"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
)

// EngineClient is the interface for communicating with a single vLLM engine.
// This maps to vLLM's HTTP endpoints behind VLLM_SERVER_DEV_MODE.
type EngineClient interface {
	// Pause pauses generation on this engine.
	Pause(ctx context.Context, mode v1alpha1.PauseMode) error

	// Resume resumes generation on this engine.
	Resume(ctx context.Context) error

	// InitWeightTransfer initializes the weight transfer data plane.
	// rankOffset is this engine's rank in the NCCL group (trainer is rank 0).
	InitWeightTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit, rankOffset int32) error

	// UpdateWeights triggers weight reception on this engine.
	UpdateWeights(ctx context.Context, req *v1alpha1.WeightUpdateRequest) error

	// GetWeightVersion returns this engine's current weight version.
	GetWeightVersion(ctx context.Context) (int64, error)

	// Sleep puts the engine to sleep at the specified level.
	Sleep(ctx context.Context, level v1alpha1.SleepLevel) error

	// WakeUp wakes the engine, restoring the specified resource tags.
	WakeUp(ctx context.Context, tags []string) error

	// Health checks if the engine is healthy.
	Health(ctx context.Context) error

	// ResetPrefixCache clears the engine's prefix cache.
	ResetPrefixCache(ctx context.Context) error
}
