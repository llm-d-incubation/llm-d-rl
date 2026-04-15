package pool

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
	"github.com/llm-d/llm-d-rl/pkg/weightsync"
)

// ClientFactory creates a weightsync.EngineClient for a given engine address.
// Callers inject this so the pool stays testable (simulated vs real clients).
type ClientFactory func(address string) weightsync.EngineClient

// BaseConfig configures the engine-tracking and health-check behaviour.
type BaseConfig struct {
	ClientFactory       ClientFactory
	HealthCheckInterval time.Duration
	HealthCheckTimeout  time.Duration
	OnEngineUnhealthy   func(id string, err error)
}

// Config holds the full configuration for an EnginePool.
type Config struct {
	BaseConfig
	// RouterURL is the HTTP base URL of the inference router/EPP gateway.
	// When non-empty, PickEngine always returns this URL for generation requests.
	// Leave empty to dispatch generation requests directly to engine pods.
	RouterURL string
	// TokensIn controls the prompt format: true = token-ID arrays, false = text.
	TokensIn bool
}

// engineState tracks the runtime state of a single registered engine.
type engineState struct {
	id                  string
	address             string
	client              weightsync.EngineClient
	ready               bool
	consecutiveFailures int
	lastHealthCheck     time.Time
}

// EnginePool is the single Pool implementation.
//
// Generation routing:
//   - If routerURL is set, PickEngine always returns it (router/EPP path).
//   - Otherwise PickEngine returns the first ready engine from the tracked pool.
//
// Weight sync always targets the tracked engine pods directly, regardless of
// whether a router URL is configured.
type EnginePool struct {
	mu      sync.RWMutex
	engines map[string]*engineState
	phase   v1alpha1.PoolPhase

	factory             ClientFactory
	healthCheckInterval time.Duration
	healthCheckTimeout  time.Duration
	onEngineUnhealthy   func(id string, err error)

	weightVersion  int64
	transferInited bool

	routerURL string
	tokensIn  bool
}

// NewEnginePool creates an EnginePool from the given configuration.
func NewEnginePool(cfg Config) *EnginePool {
	if cfg.HealthCheckInterval == 0 {
		cfg.HealthCheckInterval = 30 * time.Second
	}
	if cfg.HealthCheckTimeout == 0 {
		cfg.HealthCheckTimeout = 5 * time.Second
	}
	return &EnginePool{
		engines:             make(map[string]*engineState),
		phase:               v1alpha1.PoolPhaseServing,
		factory:             cfg.ClientFactory,
		healthCheckInterval: cfg.HealthCheckInterval,
		healthCheckTimeout:  cfg.HealthCheckTimeout,
		onEngineUnhealthy:   cfg.OnEngineUnhealthy,
		routerURL:           cfg.RouterURL,
		tokensIn:            cfg.TokensIn,
	}
}

// --- Pool interface: routing ---

// PickEngine returns routerURL when configured; otherwise picks the first ready engine.
// engineID is empty in router/EPP mode (the selected pod is unknown at this layer).
func (p *EnginePool) PickEngine() (string, string, error) {
	if p.routerURL != "" {
		return p.routerURL, "", nil
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	for _, es := range p.engines {
		if es.ready {
			return es.address, es.id, nil
		}
	}
	return "", "", fmt.Errorf("no ready engines in pool")
}

// TokensIn reports whether this pool expects token-ID arrays as input.
func (p *EnginePool) TokensIn() bool { return p.tokensIn }

// --- Pool interface: engine membership ---

// AddEngine registers a new engine and creates its client via the factory.
func (p *EnginePool) AddEngine(id, address string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.engines[id] = &engineState{
		id:      id,
		address: address,
		client:  p.factory(address),
	}
	if p.transferInited {
		log.Printf("pool: WARNING weight-transfer group invalidated — engine added id=%s addr=%s; call InitWeightTransfer before next UpdateWeights", id, address)
	}
	p.transferInited = false
	log.Printf("pool: registered engine id=%s addr=%s", id, address)
}

// RemoveEngine deregisters an engine.
func (p *EnginePool) RemoveEngine(id string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.transferInited {
		log.Printf("pool: WARNING weight-transfer group invalidated — engine removed id=%s; call InitWeightTransfer before next UpdateWeights", id)
	}
	delete(p.engines, id)
	p.transferInited = false
	log.Printf("pool: deregistered engine id=%s", id)
}

// --- Pool interface: observability ---

// SetPhase updates the pool's operational phase.
func (p *EnginePool) SetPhase(phase v1alpha1.PoolPhase) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.phase = phase
}

// Status returns a snapshot of the pool's current state.
func (p *EnginePool) Status() v1alpha1.PoolStatus {
	p.mu.RLock()
	defer p.mu.RUnlock()
	status := v1alpha1.PoolStatus{
		Phase:        p.phase,
		TotalEngines: len(p.engines),
	}
	for _, es := range p.engines {
		status.Engines = append(status.Engines, v1alpha1.EngineStatus{
			ID:      es.id,
			Address: es.address,
			Ready:   es.ready,
		})
		if es.ready {
			status.ReadyEngines++
		}
	}
	return status
}

// WeightVersion returns the current weight version.
func (p *EnginePool) WeightVersion() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.weightVersion
}

// --- Pool interface: health checks ---

// StartHealthLoop runs periodic health checks until ctx is cancelled.
func (p *EnginePool) StartHealthLoop(ctx context.Context) {
	p.runHealthChecks(ctx)
	ticker := time.NewTicker(p.healthCheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			p.runHealthChecks(ctx)
		}
	}
}

func (p *EnginePool) runHealthChecks(ctx context.Context) {
	p.mu.RLock()
	snapshot := make([]*engineState, 0, len(p.engines))
	for _, es := range p.engines {
		snapshot = append(snapshot, es)
	}
	p.mu.RUnlock()

	var wg sync.WaitGroup
	for _, es := range snapshot {
		wg.Add(1)
		go func(es *engineState) {
			defer wg.Done()
			checkCtx, cancel := context.WithTimeout(ctx, p.healthCheckTimeout)
			defer cancel()
			err := es.client.Health(checkCtx)

			p.mu.Lock()
			es.lastHealthCheck = time.Now()
			if err != nil {
				es.consecutiveFailures++
				if es.consecutiveFailures >= 3 && es.ready {
					es.ready = false
					if p.onEngineUnhealthy != nil {
						id := es.id
						go p.onEngineUnhealthy(id, fmt.Errorf("engine %s: %d consecutive health check failures: %w", id, es.consecutiveFailures, err))
					}
				}
			} else {
				es.consecutiveFailures = 0
				es.ready = true
			}
			p.mu.Unlock()
		}(es)
	}
	wg.Wait()
}

// --- Pool interface: weight sync ---

// InitWeightTransfer initializes the weight transfer NCCL/NIXL group on all engines.
func (p *EnginePool) InitWeightTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	type work struct {
		client weightsync.EngineClient
		rank   int32
	}
	jobs := make([]work, 0, len(p.engines))
	rank := int32(1) // trainer is rank 0, engines start at 1
	for _, es := range p.engines {
		jobs = append(jobs, work{es.client, rank})
		rank++
	}

	errs := fanOut(ctx, len(jobs), func(ctx context.Context, i int) error {
		return jobs[i].client.InitWeightTransfer(ctx, init, jobs[i].rank)
	})
	if len(errs) > 0 {
		return fmt.Errorf("init weight transfer: %d engine(s) failed: %v", len(errs), errs)
	}
	p.transferInited = true
	return nil
}

// UpdateWeights orchestrates a full weight update cycle across all engines.
func (p *EnginePool) UpdateWeights(ctx context.Context, req *v1alpha1.WeightUpdateRequest) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.transferInited {
		return fmt.Errorf("weight transfer not initialized; call InitWeightTransfer first")
	}

	pauseMode := req.PauseMode
	if pauseMode == "" {
		pauseMode = v1alpha1.PauseModeKeep
	}

	clients := p.clientSlice()

	// 1. Pause
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].Pause(ctx, pauseMode)
	}); len(errs) > 0 {
		return fmt.Errorf("pause engines: %v", errs)
	}

	// 2. Update weights (data plane flows trainer->engine via NCCL/NIXL)
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].UpdateWeights(ctx, req)
	}); len(errs) > 0 {
		fanOut(ctx, len(clients), func(ctx context.Context, i int) error { //nolint:errcheck
			return clients[i].Resume(ctx)
		})
		return fmt.Errorf("update weights: %v", errs)
	}

	// 3. Reset KV cache if requested
	if req.ResetKVCache {
		if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
			return clients[i].ResetPrefixCache(ctx)
		}); len(errs) > 0 {
			return fmt.Errorf("reset prefix cache: %v", errs)
		}
	}

	// 4. Resume
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].Resume(ctx)
	}); len(errs) > 0 {
		return fmt.Errorf("resume engines: %v", errs)
	}

	p.weightVersion = req.TargetVersion
	return nil
}

func (p *EnginePool) PauseAll(ctx context.Context, mode v1alpha1.PauseMode) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	clients := p.clientSlice() // capture once; map order is non-deterministic
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].Pause(ctx, mode)
	}); len(errs) > 0 {
		return fmt.Errorf("pause: %v", errs)
	}
	return nil
}

func (p *EnginePool) ResumeAll(ctx context.Context) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	clients := p.clientSlice()
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].Resume(ctx)
	}); len(errs) > 0 {
		return fmt.Errorf("resume: %v", errs)
	}
	return nil
}

func (p *EnginePool) SleepAll(ctx context.Context, level v1alpha1.SleepLevel) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	clients := p.clientSlice()
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].Sleep(ctx, level)
	}); len(errs) > 0 {
		return fmt.Errorf("sleep: %v", errs)
	}
	return nil
}

func (p *EnginePool) WakeUpAll(ctx context.Context, tags []string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	clients := p.clientSlice()
	if errs := fanOut(ctx, len(clients), func(ctx context.Context, i int) error {
		return clients[i].WakeUp(ctx, tags)
	}); len(errs) > 0 {
		return fmt.Errorf("wake up: %v", errs)
	}
	return nil
}

// --- internal helpers ---

// clientSlice returns all engine clients. Caller must hold p.mu.
func (p *EnginePool) clientSlice() []weightsync.EngineClient {
	out := make([]weightsync.EngineClient, 0, len(p.engines))
	for _, es := range p.engines {
		out = append(out, es.client)
	}
	return out
}

// fanOut runs fn(i) for i in [0, n) concurrently and collects errors.
func fanOut(ctx context.Context, n int, fn func(context.Context, int) error) []error {
	var (
		wg   sync.WaitGroup
		mu   sync.Mutex
		errs []error
	)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			if err := fn(ctx, i); err != nil {
				mu.Lock()
				errs = append(errs, err)
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	return errs
}

// compile-time check
var _ Pool = (*EnginePool)(nil)
