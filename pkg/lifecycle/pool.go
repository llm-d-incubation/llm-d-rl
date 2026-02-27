// Package lifecycle manages the lifecycle of vLLM inference engine pools
// for RL rollout workloads. It handles health monitoring, automatic recovery,
// and sleep/wake orchestration for GPU memory sharing with training.
package lifecycle

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
	"github.com/llm-d/llm-d-rl/pkg/weightsync"
)

// EngineInfo holds metadata about a registered engine.
type EngineInfo struct {
	ID      string
	Address string
	Client  weightsync.EngineClient
}

// PoolManager manages the lifecycle of an inference engine pool.
type PoolManager struct {
	mu      sync.RWMutex
	engines map[string]*engineState
	phase   v1alpha1.PoolPhase

	healthCheckInterval time.Duration
	healthCheckTimeout  time.Duration

	// onEngineUnhealthy is called when an engine fails health checks.
	// The implementation should trigger pod replacement via Kubernetes.
	onEngineUnhealthy func(id string, err error)
}

type engineState struct {
	info              EngineInfo
	ready             bool
	consecutiveFailures int
	lastHealthCheck   time.Time
}

// PoolManagerConfig configures the pool manager.
type PoolManagerConfig struct {
	HealthCheckInterval time.Duration
	HealthCheckTimeout  time.Duration
	MaxFailures         int // consecutive failures before marking unhealthy

	OnEngineUnhealthy func(id string, err error)
}

// NewPoolManager creates a new engine pool manager.
func NewPoolManager(cfg PoolManagerConfig) *PoolManager {
	if cfg.HealthCheckInterval == 0 {
		cfg.HealthCheckInterval = 30 * time.Second
	}
	if cfg.HealthCheckTimeout == 0 {
		cfg.HealthCheckTimeout = 5 * time.Second
	}

	return &PoolManager{
		engines:             make(map[string]*engineState),
		phase:               v1alpha1.PoolPhaseServing,
		healthCheckInterval: cfg.HealthCheckInterval,
		healthCheckTimeout:  cfg.HealthCheckTimeout,
		onEngineUnhealthy:   cfg.OnEngineUnhealthy,
	}
}

// AddEngine registers an engine with the pool.
func (pm *PoolManager) AddEngine(info EngineInfo) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.engines[info.ID] = &engineState{
		info:  info,
		ready: false, // Will be set to true after first health check
	}
}

// RemoveEngine unregisters an engine from the pool.
func (pm *PoolManager) RemoveEngine(id string) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	delete(pm.engines, id)
}

// Status returns the current pool status.
func (pm *PoolManager) Status() v1alpha1.PoolStatus {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	status := v1alpha1.PoolStatus{
		Phase:        pm.phase,
		TotalEngines: len(pm.engines),
	}

	for _, es := range pm.engines {
		engineStatus := v1alpha1.EngineStatus{
			ID:      es.info.ID,
			Address: es.info.Address,
			Ready:   es.ready,
		}
		status.Engines = append(status.Engines, engineStatus)
		if es.ready {
			status.ReadyEngines++
		}
	}

	return status
}

// SetPhase updates the pool's operational phase.
func (pm *PoolManager) SetPhase(phase v1alpha1.PoolPhase) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.phase = phase
}

// ReadyEngines returns the clients of all healthy engines.
func (pm *PoolManager) ReadyEngines() []weightsync.EngineClient {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	var clients []weightsync.EngineClient
	for _, es := range pm.engines {
		if es.ready {
			clients = append(clients, es.info.Client)
		}
	}
	return clients
}

// PickReadyEngine returns a ready engine's info for request dispatching.
// Returns an error if no engines are ready.
func (pm *PoolManager) PickReadyEngine() (*EngineInfo, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	for _, es := range pm.engines {
		if es.ready {
			info := es.info
			return &info, nil
		}
	}
	return nil, fmt.Errorf("no ready engines in pool")
}

// RunHealthChecks runs a single round of health checks on all engines.
func (pm *PoolManager) RunHealthChecks(ctx context.Context) {
	// Snapshot engines under read lock so health checks run without holding the lock.
	pm.mu.RLock()
	type entry struct {
		id string
		es *engineState
	}
	entries := make([]entry, 0, len(pm.engines))
	for id, es := range pm.engines {
		entries = append(entries, entry{id, es})
	}
	pm.mu.RUnlock()

	var wg sync.WaitGroup

	for _, e := range entries {
		wg.Add(1)
		go func(id string, es *engineState) {
			defer wg.Done()

			checkCtx, cancel := context.WithTimeout(ctx, pm.healthCheckTimeout)
			defer cancel()

			err := es.info.Client.Health(checkCtx)

			pm.mu.Lock()
			es.lastHealthCheck = time.Now()

			if err != nil {
				es.consecutiveFailures++
				if es.consecutiveFailures >= 3 && es.ready {
					es.ready = false
					if pm.onEngineUnhealthy != nil {
						go pm.onEngineUnhealthy(id, fmt.Errorf("engine %s: %d consecutive health check failures: %w", id, es.consecutiveFailures, err))
					}
				}
			} else {
				es.consecutiveFailures = 0
				es.ready = true
			}
			pm.mu.Unlock()
		}(e.id, e.es)
	}

	wg.Wait()
}

// StartHealthLoop runs health checks periodically until the context is cancelled.
func (pm *PoolManager) StartHealthLoop(ctx context.Context) {
	ticker := time.NewTicker(pm.healthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			pm.RunHealthChecks(ctx)
		}
	}
}
