// vllm_client.go implements the EngineClient interface for vLLM engines
// using vLLM's HTTP API endpoints (behind VLLM_SERVER_DEV_MODE=1).
package weightsync

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
)

// VLLMClient implements EngineClient for a vLLM HTTP server.
type VLLMClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewVLLMClient creates a client for a vLLM engine at the given base URL.
func NewVLLMClient(baseURL string) *VLLMClient {
	return &VLLMClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (c *VLLMClient) Pause(ctx context.Context, mode v1alpha1.PauseMode) error {
	body := map[string]interface{}{
		"mode":        string(mode),
		"clear_cache": false,
	}
	return c.post(ctx, "/pause", body)
}

func (c *VLLMClient) Resume(ctx context.Context) error {
	return c.post(ctx, "/resume", nil)
}

func (c *VLLMClient) InitWeightTransfer(ctx context.Context, init *v1alpha1.WeightTransferInit) error {
	body := map[string]interface{}{
		"init_info": map[string]interface{}{
			"master_address": init.MasterAddress,
			"master_port":    init.MasterPort,
			"rank_offset":    1, // trainer is rank 0, engines start at rank 1
			"world_size":     init.TrainerWorldSize,
		},
	}
	return c.post(ctx, "/init_weight_transfer_engine", body)
}

func (c *VLLMClient) UpdateWeights(ctx context.Context, req *v1alpha1.WeightUpdateRequest) error {
	updateInfo := map[string]interface{}{
		"names":       req.ParamNames,
		"dtype_names": req.ParamDtypes,
		"shapes":      req.ParamShapes,
		"packed":      false,
	}
	body := map[string]interface{}{
		"update_info": updateInfo,
	}
	return c.post(ctx, "/update_weights", body)
}

func (c *VLLMClient) GetWeightVersion(ctx context.Context) (int64, error) {
	resp, err := c.get(ctx, "/is_paused") // placeholder — vLLM doesn't expose version yet
	if err != nil {
		return 0, err
	}
	_ = resp
	// TODO: Parse weight version from response when vLLM exposes it.
	return 0, nil
}

func (c *VLLMClient) Sleep(ctx context.Context, level v1alpha1.SleepLevel) error {
	body := map[string]interface{}{
		"level": int(level),
		"mode":  "keep",
	}
	return c.post(ctx, "/sleep", body)
}

func (c *VLLMClient) WakeUp(ctx context.Context, tags []string) error {
	body := map[string]interface{}{
		"tags": tags,
	}
	return c.post(ctx, "/wake_up", body)
}

func (c *VLLMClient) Health(ctx context.Context) error {
	_, err := c.get(ctx, "/health")
	return err
}

func (c *VLLMClient) ResetPrefixCache(ctx context.Context) error {
	return c.post(ctx, "/reset_prefix_cache", nil)
}

// --- HTTP helpers ---

func (c *VLLMClient) post(ctx context.Context, path string, body interface{}) error {
	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bodyReader)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	if bodyReader != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("POST %s: %w", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("POST %s returned %d: %s", path, resp.StatusCode, string(respBody))
	}

	return nil
}

func (c *VLLMClient) get(ctx context.Context, path string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", path, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("GET %s returned %d: %s", path, resp.StatusCode, string(body))
	}

	return io.ReadAll(resp.Body)
}

// Compile-time interface check.
var _ EngineClient = (*VLLMClient)(nil)
