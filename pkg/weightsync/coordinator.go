// Package weightsync implements weight synchronization between RL training loops
// and vLLM inference engine pools managed by llm-d.
//
// The coordinator orchestrates the full weight update lifecycle:
//
//  1. Pause generation on all engines (abort, wait, or keep mode)
//  2. Initialize the weight transfer data plane (NCCL or NIXL group)
//  3. Signal the trainer to broadcast weights
//  4. Wait for all engines to confirm receipt
//  5. Reset KV caches (if requested)
//  6. Resume generation
//  7. Update the pool's weight version
//
// The coordinator does NOT proxy weight tensor data — it orchestrates the
// control plane while the trainer and engines communicate directly via
// NCCL/NIXL for the data plane.
package weightsync

import (
	"context"
	"fmt"
	"sync"
	"time"

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
	InitWeightTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit) error

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

// Coordinator orchestrates weight synchronization across an engine pool.
type Coordinator struct {
	mu             sync.RWMutex
	engines        map[string]EngineClient // engineID -> client
	weightVersion  int64
	transferInited bool
}

// NewCoordinator creates a new weight sync coordinator.
func NewCoordinator() *Coordinator {
	return &Coordinator{
		engines: make(map[string]EngineClient),
	}
}

// RegisterEngine adds an engine to the pool.
func (c *Coordinator) RegisterEngine(id string, client EngineClient) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.engines[id] = client
}

// UnregisterEngine removes an engine from the pool.
func (c *Coordinator) UnregisterEngine(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.engines, id)
	c.transferInited = false // Group needs re-initialization
}

// InitTransfer initializes the weight transfer data plane on all engines.
func (c *Coordinator) InitTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	for id, engine := range c.engines {
		if err := engine.InitWeightTransfer(ctx, init); err != nil {
			return fmt.Errorf("init weight transfer on engine %s: %w", id, err)
		}
	}

	c.transferInited = true
	return nil
}

// UpdateWeights orchestrates a full weight update across the engine pool.
//
// The lifecycle is:
//  1. Pause all engines
//  2. Trigger weight reception (trainer broadcasts via NCCL/NIXL data plane)
//  3. Reset KV caches if requested
//  4. Resume all engines
//  5. Update weight version
func (c *Coordinator) UpdateWeights(ctx context.Context, req *v1alpha1.WeightUpdateRequest) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.transferInited {
		return fmt.Errorf("weight transfer not initialized; call InitTransfer first")
	}

	pauseMode := req.PauseMode
	if pauseMode == "" {
		pauseMode = v1alpha1.PauseModeKeep
	}

	// Step 1: Pause all engines
	if err := c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.Pause(ctx, pauseMode)
	}); err != nil {
		return fmt.Errorf("pause engines: %w", err)
	}

	// Step 2: Trigger weight reception on all engines.
	// The actual tensor data flows directly from trainer to engines via NCCL/NIXL.
	// This call tells each engine to start receiving.
	if err := c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.UpdateWeights(ctx, req)
	}); err != nil {
		// Attempt to resume engines on failure
		_ = c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
			return e.Resume(ctx)
		})
		return fmt.Errorf("update weights: %w", err)
	}

	// Step 3: Reset KV caches if requested (stale after weight update)
	if req.ResetKVCache {
		if err := c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
			return e.ResetPrefixCache(ctx)
		}); err != nil {
			return fmt.Errorf("reset prefix cache: %w", err)
		}
	}

	// Step 4: Resume all engines
	if err := c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.Resume(ctx)
	}); err != nil {
		return fmt.Errorf("resume engines: %w", err)
	}

	// Step 5: Update weight version
	c.weightVersion = req.TargetVersion

	return nil
}

// WeightVersion returns the current weight version of the pool.
func (c *Coordinator) WeightVersion() int64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.weightVersion
}

// PauseAll pauses generation on all engines.
func (c *Coordinator) PauseAll(ctx context.Context, mode v1alpha1.PauseMode) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.Pause(ctx, mode)
	})
}

// ResumeAll resumes generation on all engines.
func (c *Coordinator) ResumeAll(ctx context.Context) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.Resume(ctx)
	})
}

// SleepAll puts all engines to sleep at the specified level.
func (c *Coordinator) SleepAll(ctx context.Context, level v1alpha1.SleepLevel) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.Sleep(ctx, level)
	})
}

// WakeUpAll wakes all engines with the specified resource tags.
func (c *Coordinator) WakeUpAll(ctx context.Context, tags []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.forEachEngine(ctx, func(ctx context.Context, id string, e EngineClient) error {
		return e.WakeUp(ctx, tags)
	})
}

// forEachEngine runs fn on all engines concurrently.
func (c *Coordinator) forEachEngine(ctx context.Context, fn func(ctx context.Context, id string, e EngineClient) error) error {
	var (
		wg   sync.WaitGroup
		errs = make([]error, 0)
		mu   sync.Mutex
	)

	for id, engine := range c.engines {
		wg.Add(1)
		go func(id string, e EngineClient) {
			defer wg.Done()
			if err := fn(ctx, id, e); err != nil {
				mu.Lock()
				errs = append(errs, fmt.Errorf("engine %s: %w", id, err))
				mu.Unlock()
			}
		}(id, engine)
	}

	wg.Wait()

	if len(errs) > 0 {
		return fmt.Errorf("%d engine(s) failed: %v", len(errs), errs)
	}
	return nil
}

// VerifyWeightConsistency checks that all engines report the expected weight version.
func (c *Coordinator) VerifyWeightConsistency(ctx context.Context) (bool, map[string]int64, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	versions := make(map[string]int64)
	consistent := true

	for id, engine := range c.engines {
		v, err := engine.GetWeightVersion(ctx)
		if err != nil {
			return false, nil, fmt.Errorf("get weight version from engine %s: %w", id, err)
		}
		versions[id] = v
		if v != c.weightVersion {
			consistent = false
		}
	}

	return consistent, versions, nil
}

// compile-time interface check
var _ = time.Second // used for future timeout support
