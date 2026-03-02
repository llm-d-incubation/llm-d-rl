// simulated_client.go implements a no-op EngineClient for GPU-free demos.
// It forwards health checks to a real engine (e.g., llm-d-inference-sim)
// but no-ops all lifecycle operations (pause, resume, sleep, wake, weight
// transfer) that require vLLM's dev-mode endpoints.
package weightsync

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
)

// SimulatedEngineClient implements EngineClient for GPU-free testing.
// Health checks are forwarded to a real HTTP server (e.g., llm-d-inference-sim).
// All lifecycle operations are logged and return nil.
type SimulatedEngineClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewSimulatedEngineClient creates a simulated client that forwards health
// checks to the given base URL but no-ops all lifecycle operations.
func NewSimulatedEngineClient(baseURL string) *SimulatedEngineClient {
	return &SimulatedEngineClient{
		baseURL:    baseURL,
		httpClient: http.DefaultClient,
	}
}

func (c *SimulatedEngineClient) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("GET /health: %w", err)
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	if resp.StatusCode >= 400 {
		return fmt.Errorf("GET /health returned %d", resp.StatusCode)
	}
	return nil
}

func (c *SimulatedEngineClient) Pause(_ context.Context, mode v1alpha1.PauseMode) error {
	log.Printf("[simulated] engine %s: pause (mode=%s)", c.baseURL, mode)
	return nil
}

func (c *SimulatedEngineClient) Resume(_ context.Context) error {
	log.Printf("[simulated] engine %s: resume", c.baseURL)
	return nil
}

func (c *SimulatedEngineClient) InitWeightTransfer(_ context.Context, init *v1alpha1.WeightTransferInit, rankOffset int32) error {
	log.Printf("[simulated] engine %s: init weight transfer (backend=%s, rank=%d)", c.baseURL, init.Backend, rankOffset)
	return nil
}

func (c *SimulatedEngineClient) UpdateWeights(_ context.Context, req *v1alpha1.WeightUpdateRequest) error {
	log.Printf("[simulated] engine %s: update weights (version=%d)", c.baseURL, req.TargetVersion)
	return nil
}

func (c *SimulatedEngineClient) GetWeightVersion(_ context.Context) (int64, error) {
	return 0, nil
}

func (c *SimulatedEngineClient) Sleep(_ context.Context, level v1alpha1.SleepLevel) error {
	log.Printf("[simulated] engine %s: sleep (level=%d)", c.baseURL, level)
	return nil
}

func (c *SimulatedEngineClient) WakeUp(_ context.Context, tags []string) error {
	log.Printf("[simulated] engine %s: wake up (tags=%v)", c.baseURL, tags)
	return nil
}

func (c *SimulatedEngineClient) ResetPrefixCache(_ context.Context) error {
	log.Printf("[simulated] engine %s: reset prefix cache", c.baseURL)
	return nil
}

// Compile-time interface check.
var _ EngineClient = (*SimulatedEngineClient)(nil)
