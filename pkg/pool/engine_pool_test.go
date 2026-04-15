package pool

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
	"github.com/llm-d/llm-d-rl/pkg/weightsync"
)

// --- test double ---

// mockClient is a test double for weightsync.EngineClient.
// Each method increments its call counter, then delegates to the corresponding
// Fn field if non-nil; otherwise it returns nil.
type mockClient struct {
	pauseCalls      atomic.Int32
	resumeCalls     atomic.Int32
	updateCalls     atomic.Int32
	resetCacheCalls atomic.Int32

	pauseFn         func(context.Context, v1alpha1.PauseMode) error
	resumeFn        func(context.Context) error
	initTransferFn  func(context.Context, *v1alpha1.WeightTransferInit, int32) error
	updateWeightsFn func(context.Context, *v1alpha1.WeightUpdateRequest) error
	resetCacheFn    func(context.Context) error
	healthFn        func(context.Context) error
	sleepFn         func(context.Context, v1alpha1.SleepLevel) error
	wakeUpFn        func(context.Context, []string) error
}

var _ weightsync.EngineClient = (*mockClient)(nil)

func (m *mockClient) Pause(ctx context.Context, mode v1alpha1.PauseMode) error {
	m.pauseCalls.Add(1)
	if m.pauseFn != nil {
		return m.pauseFn(ctx, mode)
	}
	return nil
}

func (m *mockClient) Resume(ctx context.Context) error {
	m.resumeCalls.Add(1)
	if m.resumeFn != nil {
		return m.resumeFn(ctx)
	}
	return nil
}

func (m *mockClient) InitWeightTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit, rankOffset int32) error {
	if m.initTransferFn != nil {
		return m.initTransferFn(ctx, init, rankOffset)
	}
	return nil
}

func (m *mockClient) UpdateWeights(ctx context.Context, req *v1alpha1.WeightUpdateRequest) error {
	m.updateCalls.Add(1)
	if m.updateWeightsFn != nil {
		return m.updateWeightsFn(ctx, req)
	}
	return nil
}

func (m *mockClient) GetWeightVersion(ctx context.Context) (int64, error) { return 0, nil }

func (m *mockClient) Sleep(ctx context.Context, level v1alpha1.SleepLevel) error {
	if m.sleepFn != nil {
		return m.sleepFn(ctx, level)
	}
	return nil
}

func (m *mockClient) WakeUp(ctx context.Context, tags []string) error {
	if m.wakeUpFn != nil {
		return m.wakeUpFn(ctx, tags)
	}
	return nil
}

func (m *mockClient) Health(ctx context.Context) error {
	if m.healthFn != nil {
		return m.healthFn(ctx)
	}
	return nil
}

func (m *mockClient) ResetPrefixCache(ctx context.Context) error {
	m.resetCacheCalls.Add(1)
	if m.resetCacheFn != nil {
		return m.resetCacheFn(ctx)
	}
	return nil
}

// --- helpers ---

// newTestPool builds an EnginePool whose clients come from the provided map.
// Each map key is used as both engine ID and address for simplicity.
func newTestPool(clients map[string]*mockClient) *EnginePool {
	p := NewEnginePool(Config{
		BaseConfig: BaseConfig{
			ClientFactory: func(addr string) weightsync.EngineClient {
				return clients[addr]
			},
		},
	})
	for id := range clients {
		p.AddEngine(id, id)
	}
	return p
}

// mustInitTransfer calls InitWeightTransfer and fatals on error.
func mustInitTransfer(t *testing.T, p *EnginePool) {
	t.Helper()
	err := p.InitWeightTransfer(context.Background(), &v1alpha1.WeightTransferInit{
		Backend: v1alpha1.WeightSyncNCCL,
	})
	if err != nil {
		t.Fatalf("InitWeightTransfer: %v", err)
	}
}

// --- UpdateWeights ---

// TestUpdateWeights covers the full pause→update→[reset]→resume orchestration,
// including the error-path rollback (resume is called when update fails).
func TestUpdateWeights(t *testing.T) {
	someErr := errors.New("engine error")

	tests := []struct {
		name         string
		engineCount  int
		skipInit     bool
		pauseMode    v1alpha1.PauseMode
		resetKVCache bool
		pauseErr     error
		updateErr    error
		resetErr     error
		resumeErr    error
		wantErr      bool
		wantVersion  int64
	}{
		{
			name:        "not initialized returns error",
			engineCount: 1,
			skipInit:    true,
			wantErr:     true,
		},
		{
			name:        "happy path no KV reset",
			engineCount: 1,
			pauseMode:   v1alpha1.PauseModeKeep,
			wantVersion: 42,
		},
		{
			// PauseMode defaults to Keep when empty.
			name:        "empty pause mode defaults to keep",
			engineCount: 1,
			wantVersion: 42,
		},
		{
			name:         "happy path with KV reset",
			engineCount:  1,
			pauseMode:    v1alpha1.PauseModeKeep,
			resetKVCache: true,
			wantVersion:  42,
		},
		{
			name:        "pause fails: update and resume not called",
			engineCount: 1,
			pauseMode:   v1alpha1.PauseModeKeep,
			pauseErr:    someErr,
			wantErr:     true,
		},
		{
			// On update failure the pool must resume engines before returning.
			name:        "update fails: rollback resume called, version unchanged",
			engineCount: 1,
			pauseMode:   v1alpha1.PauseModeKeep,
			updateErr:   someErr,
			wantErr:     true,
		},
		{
			name:         "KV reset fails: error returned",
			engineCount:  1,
			pauseMode:    v1alpha1.PauseModeKeep,
			resetKVCache: true,
			resetErr:     someErr,
			wantErr:      true,
		},
		{
			name:        "resume fails: error, version not set",
			engineCount: 1,
			pauseMode:   v1alpha1.PauseModeKeep,
			resumeErr:   someErr,
			wantErr:     true,
		},
		{
			name:        "empty pool succeeds",
			engineCount: 0,
			wantVersion: 42,
		},
		{
			// Guards against regressions where clientSlice() stops reaching all engines.
			name:        "multi-engine: all engines paused, updated and resumed",
			engineCount: 3,
			pauseMode:   v1alpha1.PauseModeKeep,
			wantVersion: 42,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clients := make(map[string]*mockClient, tc.engineCount)
			for i := range tc.engineCount {
				id := fmt.Sprintf("engine-%d", i)
				mc := &mockClient{}
				if tc.pauseErr != nil {
					mc.pauseFn = func(_ context.Context, _ v1alpha1.PauseMode) error { return tc.pauseErr }
				}
				if tc.updateErr != nil {
					mc.updateWeightsFn = func(_ context.Context, _ *v1alpha1.WeightUpdateRequest) error { return tc.updateErr }
				}
				if tc.resetErr != nil {
					mc.resetCacheFn = func(_ context.Context) error { return tc.resetErr }
				}
				if tc.resumeErr != nil {
					mc.resumeFn = func(_ context.Context) error { return tc.resumeErr }
				}
				clients[id] = mc
			}

			p := newTestPool(clients)
			if !tc.skipInit {
				mustInitTransfer(t, p)
			}

			req := &v1alpha1.WeightUpdateRequest{
				TargetVersion: 42,
				PauseMode:     tc.pauseMode,
				ResetKVCache:  tc.resetKVCache,
			}
			err := p.UpdateWeights(context.Background(), req)

			if (err != nil) != tc.wantErr {
				t.Errorf("error = %v, wantErr = %v", err, tc.wantErr)
			}
			if got := p.WeightVersion(); got != tc.wantVersion {
				t.Errorf("WeightVersion = %d, want %d", got, tc.wantVersion)
			}

			// Pause fails: UpdateWeights and Resume must not be called.
			if tc.pauseErr != nil {
				for id, mc := range clients {
					if got := mc.updateCalls.Load(); got != 0 {
						t.Errorf("engine %s: UpdateWeights called %d times after pause failure, want 0", id, got)
					}
					if got := mc.resumeCalls.Load(); got != 0 {
						t.Errorf("engine %s: Resume called %d times after pause failure, want 0", id, got)
					}
				}
			}

			// Update fails: rollback resume must be called on all engines.
			if tc.updateErr != nil {
				for id, mc := range clients {
					if got := mc.resumeCalls.Load(); got == 0 {
						t.Errorf("engine %s: Resume not called for rollback", id)
					}
				}
			}

			// Multi-engine happy path: every engine must be paused, updated, and resumed.
			if tc.engineCount > 1 && !tc.wantErr {
				for id, mc := range clients {
					if got := mc.pauseCalls.Load(); got == 0 {
						t.Errorf("engine %s: Pause not called", id)
					}
					if got := mc.updateCalls.Load(); got == 0 {
						t.Errorf("engine %s: UpdateWeights not called", id)
					}
					if got := mc.resumeCalls.Load(); got == 0 {
						t.Errorf("engine %s: Resume not called", id)
					}
				}
			}
		})
	}
}

// --- PauseAll / ResumeAll ---

// TestPauseAll verifies error propagation and that every registered engine is
// reached regardless of individual failures (clientSlice regression guard).
func TestPauseAll(t *testing.T) {
	someErr := errors.New("pause error")

	tests := []struct {
		name          string
		engineCount   int
		failedEngines map[string]bool
		wantErr       bool
	}{
		{name: "empty pool", engineCount: 0},
		{name: "1 engine success", engineCount: 1},
		{name: "3 engines all succeed", engineCount: 3},
		{
			name:          "3 engines one fails",
			engineCount:   3,
			failedEngines: map[string]bool{"engine-1": true},
			wantErr:       true,
		},
		{
			name:          "3 engines all fail",
			engineCount:   3,
			failedEngines: map[string]bool{"engine-0": true, "engine-1": true, "engine-2": true},
			wantErr:       true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clients := make(map[string]*mockClient, tc.engineCount)
			for i := range tc.engineCount {
				id := fmt.Sprintf("engine-%d", i)
				mc := &mockClient{}
				if tc.failedEngines[id] {
					mc.pauseFn = func(_ context.Context, _ v1alpha1.PauseMode) error { return someErr }
				}
				clients[id] = mc
			}
			p := newTestPool(clients)

			err := p.PauseAll(context.Background(), v1alpha1.PauseModeKeep)

			if (err != nil) != tc.wantErr {
				t.Errorf("PauseAll() error = %v, wantErr = %v", err, tc.wantErr)
			}
			// All engines must have been reached, even on partial failure.
			for id, mc := range clients {
				if got := mc.pauseCalls.Load(); got == 0 {
					t.Errorf("engine %s: Pause not called", id)
				}
			}
		})
	}
}

func TestResumeAll(t *testing.T) {
	someErr := errors.New("resume error")

	tests := []struct {
		name          string
		engineCount   int
		failedEngines map[string]bool
		wantErr       bool
	}{
		{name: "empty pool", engineCount: 0},
		{name: "1 engine success", engineCount: 1},
		{name: "3 engines all succeed", engineCount: 3},
		{
			name:          "3 engines one fails",
			engineCount:   3,
			failedEngines: map[string]bool{"engine-1": true},
			wantErr:       true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clients := make(map[string]*mockClient, tc.engineCount)
			for i := range tc.engineCount {
				id := fmt.Sprintf("engine-%d", i)
				mc := &mockClient{}
				if tc.failedEngines[id] {
					mc.resumeFn = func(_ context.Context) error { return someErr }
				}
				clients[id] = mc
			}
			p := newTestPool(clients)

			err := p.ResumeAll(context.Background())

			if (err != nil) != tc.wantErr {
				t.Errorf("ResumeAll() error = %v, wantErr = %v", err, tc.wantErr)
			}
			for id, mc := range clients {
				if got := mc.resumeCalls.Load(); got == 0 {
					t.Errorf("engine %s: Resume not called", id)
				}
			}
		})
	}
}

// --- SleepAll / WakeUpAll ---

func TestSleepAll(t *testing.T) {
	someErr := errors.New("sleep error")

	tests := []struct {
		name        string
		engineCount int
		sleepErr    error
		wantErr     bool
	}{
		{name: "empty pool", engineCount: 0},
		{name: "1 engine success", engineCount: 1},
		{name: "1 engine fails", engineCount: 1, sleepErr: someErr, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clients := make(map[string]*mockClient, tc.engineCount)
			for i := range tc.engineCount {
				id := fmt.Sprintf("engine-%d", i)
				mc := &mockClient{}
				if tc.sleepErr != nil {
					mc.sleepFn = func(_ context.Context, _ v1alpha1.SleepLevel) error { return tc.sleepErr }
				}
				clients[id] = mc
			}
			p := newTestPool(clients)

			err := p.SleepAll(context.Background(), v1alpha1.SleepLevel2)
			if (err != nil) != tc.wantErr {
				t.Errorf("SleepAll() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

func TestWakeUpAll(t *testing.T) {
	someErr := errors.New("wake error")

	tests := []struct {
		name        string
		engineCount int
		wakeErr     error
		wantErr     bool
	}{
		{name: "empty pool", engineCount: 0},
		{name: "1 engine success", engineCount: 1},
		{name: "1 engine fails", engineCount: 1, wakeErr: someErr, wantErr: true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clients := make(map[string]*mockClient, tc.engineCount)
			for i := range tc.engineCount {
				id := fmt.Sprintf("engine-%d", i)
				mc := &mockClient{}
				if tc.wakeErr != nil {
					mc.wakeUpFn = func(_ context.Context, _ []string) error { return tc.wakeErr }
				}
				clients[id] = mc
			}
			p := newTestPool(clients)

			err := p.WakeUpAll(context.Background(), []string{"weights"})
			if (err != nil) != tc.wantErr {
				t.Errorf("WakeUpAll() error = %v, wantErr = %v", err, tc.wantErr)
			}
		})
	}
}

// --- Engine membership ---

func TestAddRemoveEngine(t *testing.T) {
	factory := func(addr string) weightsync.EngineClient { return &mockClient{} }

	t.Run("AddEngine invalidates transfer init", func(t *testing.T) {
		p := NewEnginePool(Config{BaseConfig: BaseConfig{ClientFactory: factory}})
		p.AddEngine("e1", "addr1")
		mustInitTransfer(t, p)
		// Adding a new engine must invalidate the group so the caller re-inits.
		p.AddEngine("e2", "addr2")
		err := p.UpdateWeights(context.Background(), &v1alpha1.WeightUpdateRequest{TargetVersion: 1})
		if err == nil {
			t.Fatal("expected error after AddEngine invalidated transfer init, got nil")
		}
	})

	t.Run("RemoveEngine invalidates transfer init", func(t *testing.T) {
		p := NewEnginePool(Config{BaseConfig: BaseConfig{ClientFactory: factory}})
		p.AddEngine("e1", "addr1")
		mustInitTransfer(t, p)
		p.RemoveEngine("e1")
		err := p.UpdateWeights(context.Background(), &v1alpha1.WeightUpdateRequest{TargetVersion: 1})
		if err == nil {
			t.Fatal("expected error after RemoveEngine invalidated transfer init, got nil")
		}
	})

	t.Run("RemoveEngine removes engine from pool", func(t *testing.T) {
		clients := map[string]*mockClient{"e1": {}}
		p := newTestPool(clients)
		if s := p.Status(); s.TotalEngines != 1 {
			t.Fatalf("initial TotalEngines = %d, want 1", s.TotalEngines)
		}
		p.RemoveEngine("e1")
		if s := p.Status(); s.TotalEngines != 0 {
			t.Errorf("TotalEngines after remove = %d, want 0", s.TotalEngines)
		}
	})
}

// --- PickEngine ---

func TestPickEngine(t *testing.T) {
	factory := func(addr string) weightsync.EngineClient { return &mockClient{} }

	t.Run("router URL always returned in EPP mode", func(t *testing.T) {
		p := NewEnginePool(Config{
			BaseConfig: BaseConfig{ClientFactory: factory},
			RouterURL:  "http://router:8080",
		})
		url, id, err := p.PickEngine()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if url != "http://router:8080" {
			t.Errorf("url = %q, want %q", url, "http://router:8080")
		}
		if id != "" {
			t.Errorf("engineID = %q, want empty (router mode)", id)
		}
	})

	t.Run("direct mode picks ready engine", func(t *testing.T) {
		p := NewEnginePool(Config{BaseConfig: BaseConfig{ClientFactory: factory}})
		p.AddEngine("e1", "http://engine:8000")
		// Mark engine ready directly (bypassing health loop).
		p.mu.Lock()
		p.engines["e1"].ready = true
		p.mu.Unlock()

		url, id, err := p.PickEngine()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if url != "http://engine:8000" {
			t.Errorf("url = %q, want %q", url, "http://engine:8000")
		}
		if id != "e1" {
			t.Errorf("engineID = %q, want %q", id, "e1")
		}
	})

	t.Run("direct mode returns error when no ready engines", func(t *testing.T) {
		p := NewEnginePool(Config{BaseConfig: BaseConfig{ClientFactory: factory}})
		p.AddEngine("e1", "http://engine:8000")
		// Engine registered but not marked ready.
		_, _, err := p.PickEngine()
		if err == nil {
			t.Fatal("expected error when no ready engines, got nil")
		}
	})
}
