// Package rollout implements the rollout controller HTTP/gRPC server.
// This is the primary entry point for RL training frameworks to interact
// with the llm-d rollout infrastructure.
package rollout

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/llm-d/llm-d-rl/api/v1alpha1"
	"github.com/llm-d/llm-d-rl/pkg/pool"
)

// Server is the rollout controller HTTP server.
// It exposes the RolloutControl API surface for RL training frameworks.
type Server struct {
	pool pool.Pool

	mu              sync.RWMutex
	weightVersion   int64
	cachedModelName string // lazily populated on first /v1/models query
}

// NewServer creates a new rollout controller server backed by the given pool.
func NewServer(p pool.Pool) *Server {
	return &Server{pool: p}
}

// Handler returns an http.Handler with all rollout API routes registered.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Generation
	mux.HandleFunc("POST /v1/generate", s.handleGenerate)
	mux.HandleFunc("POST /v1/generate/abort", s.handleAbortGeneration)

	// Weight management
	mux.HandleFunc("POST /v1/weights/init", s.handleInitWeightTransfer)
	mux.HandleFunc("POST /v1/weights/update", s.handleUpdateWeights)
	mux.HandleFunc("GET /v1/weights/version", s.handleGetWeightVersion)

	// Engine lifecycle
	mux.HandleFunc("POST /v1/engines/pause", s.handlePause)
	mux.HandleFunc("POST /v1/engines/resume", s.handleResume)
	mux.HandleFunc("POST /v1/engines/sleep", s.handleSleep)
	mux.HandleFunc("POST /v1/engines/wake", s.handleWakeUp)

	// Pool status
	mux.HandleFunc("GET /v1/pool/status", s.handlePoolStatus)
	mux.HandleFunc("GET /v1/health", s.handleHealth)

	return mux
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req v1alpha1.GenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	endpoint, engineID, err := s.pool.PickEngine()
	if err != nil {
		http.Error(w, fmt.Sprintf("no endpoint available: %v", err), http.StatusServiceUnavailable)
		return
	}

	headers := map[string]string{}
	if req.SessionID != "" {
		headers["X-Session-ID"] = req.SessionID
	}

	resp, err := s.postCompletions(r.Context(), endpoint, &req, headers, s.pool.TokensIn())
	if err != nil {
		http.Error(w, fmt.Sprintf("generate: %v", err), http.StatusBadGateway)
		return
	}

	resp.EngineID = engineID // empty in EPP mode; set in direct-dispatch mode
	s.mu.RLock()
	resp.WeightVersion = s.weightVersion
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// discoverModelName queries baseURL/v1/models the first time it is called and
// caches the result. Falls back to "default" if the endpoint is unreachable.
func (s *Server) discoverModelName(ctx context.Context, baseURL string) string {
	s.mu.RLock()
	name := s.cachedModelName
	s.mu.RUnlock()
	if name != "" {
		return name
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/v1/models", nil)
	if err != nil {
		return "default"
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "default"
	}
	defer resp.Body.Close()
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil || len(models.Data) == 0 {
		return "default"
	}

	s.mu.Lock()
	s.cachedModelName = models.Data[0].ID
	s.mu.Unlock()
	return models.Data[0].ID
}

// buildOAIRequest constructs the OpenAI /v1/completions JSON body.
//
// When tokensIn is true, token IDs are sent in "prompt_token_ids".
// When tokensIn is false, the text prompt is sent in "prompt".
func (s *Server) buildOAIRequest(ctx context.Context, baseURL string, req *v1alpha1.GenerateRequest, tokensIn bool) ([]byte, error) {
	oaiReq := map[string]interface{}{
		"model": s.discoverModelName(ctx, baseURL),
	}
	if tokensIn {
		oaiReq["prompt_token_ids"] = req.PromptTokenIDs
	} else {
		oaiReq["prompt"] = req.Prompt
	}
	if req.SamplingParams != nil {
		sp := req.SamplingParams
		if sp.Temperature > 0 {
			oaiReq["temperature"] = sp.Temperature
		}
		if sp.TopP > 0 {
			oaiReq["top_p"] = sp.TopP
		}
		if sp.MaxTokens > 0 {
			oaiReq["max_tokens"] = sp.MaxTokens
		}
		if sp.NSamples > 0 {
			oaiReq["n"] = sp.NSamples
		}
		if len(sp.StopStrings) > 0 {
			oaiReq["stop"] = sp.StopStrings
		}
	}
	if req.ReturnLogprobs {
		oaiReq["logprobs"] = 1
	}
	return json.Marshal(oaiReq)
}

// parseOAIResponse parses an OpenAI /v1/completions response body.
// It always extracts output token IDs from the "token_ids" field.
func parseOAIResponse(respBody []byte) (*v1alpha1.GenerateResponse, error) {
	var oaiResp struct {
		Choices []struct {
			Text         string  `json:"text"`
			TokenIDs     []int32 `json:"token_ids"`
			FinishReason string  `json:"finish_reason"`
			Logprobs     *struct {
				TokenLogprobs []float32 `json:"token_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	resp := &v1alpha1.GenerateResponse{
		OutputTokenIDs: []int32{}, // always an array, never null
	}
	if len(oaiResp.Choices) > 0 {
		choice := oaiResp.Choices[0]
		resp.FinishReason = choice.FinishReason
		if choice.TokenIDs != nil {
			resp.OutputTokenIDs = choice.TokenIDs
		}
		resp.Text = choice.Text
		if choice.Logprobs != nil {
			resp.Logprobs = choice.Logprobs.TokenLogprobs
		}
	}
	return resp, nil
}

// postCompletions builds the OAI request body, POSTs to baseURL/v1/completions,
// applies any caller-supplied extra headers, and parses the response.
func (s *Server) postCompletions(ctx context.Context, baseURL string, req *v1alpha1.GenerateRequest, extraHeaders map[string]string, tokensIn bool) (*v1alpha1.GenerateResponse, error) {
	body, err := s.buildOAIRequest(ctx, baseURL, req, tokensIn)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		baseURL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range extraHeaders {
		httpReq.Header.Set(k, v)
	}

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("POST %s/v1/completions: %w", baseURL, err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	if httpResp.StatusCode >= 400 {
		return nil, fmt.Errorf("POST %s/v1/completions returned %d: %s", baseURL, httpResp.StatusCode, string(respBody))
	}
	return parseOAIResponse(respBody)
}

func (s *Server) handleAbortGeneration(w http.ResponseWriter, r *http.Request) {
	// TODO: Abort in-flight generation requests.
	http.Error(w, "not yet implemented", http.StatusNotImplemented)
}

func (s *Server) handleInitWeightTransfer(w http.ResponseWriter, r *http.Request) {
	var init v1alpha1.WeightTransferInit
	if err := json.NewDecoder(r.Body).Decode(&init); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if err := s.pool.InitWeightTransfer(r.Context(), &init); err != nil {
		http.Error(w, fmt.Sprintf("init weight transfer: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ready"})
}

func (s *Server) handleUpdateWeights(w http.ResponseWriter, r *http.Request) {
	var req v1alpha1.WeightUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	s.pool.SetPhase(v1alpha1.PoolPhaseSyncing)
	defer s.pool.SetPhase(v1alpha1.PoolPhaseServing)

	if err := s.pool.UpdateWeights(r.Context(), &req); err != nil {
		http.Error(w, fmt.Sprintf("update weights: %v", err), http.StatusInternalServerError)
		return
	}

	s.mu.Lock()
	s.weightVersion = req.TargetVersion
	s.mu.Unlock()

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "updated",
		"weight_version": req.TargetVersion,
	})
}

func (s *Server) handleGetWeightVersion(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	version := s.weightVersion
	s.mu.RUnlock()

	json.NewEncoder(w).Encode(map[string]int64{"weight_version": version})
}

func (s *Server) handlePause(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Mode v1alpha1.PauseMode `json:"mode"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if err := s.pool.PauseAll(r.Context(), req.Mode); err != nil {
		http.Error(w, fmt.Sprintf("pause: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "paused"})
}

func (s *Server) handleResume(w http.ResponseWriter, r *http.Request) {
	if err := s.pool.ResumeAll(r.Context()); err != nil {
		http.Error(w, fmt.Sprintf("resume: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "resumed"})
}

func (s *Server) handleSleep(w http.ResponseWriter, r *http.Request) {
	var req v1alpha1.SleepRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	s.pool.SetPhase(v1alpha1.PoolPhaseSleeping)

	if err := s.pool.SleepAll(r.Context(), v1alpha1.SleepLevel(req.Level)); err != nil {
		http.Error(w, fmt.Sprintf("sleep: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "sleeping"})
}

func (s *Server) handleWakeUp(w http.ResponseWriter, r *http.Request) {
	var req v1alpha1.WakeUpRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if err := s.pool.WakeUpAll(r.Context(), req.Tags); err != nil {
		http.Error(w, fmt.Sprintf("wake up: %v", err), http.StatusInternalServerError)
		return
	}

	s.pool.SetPhase(v1alpha1.PoolPhaseServing)

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "awake"})
}

func (s *Server) handlePoolStatus(w http.ResponseWriter, r *http.Request) {
	status := s.pool.Status()
	status.WeightVersion = s.pool.WeightVersion()
	json.NewEncoder(w).Encode(status)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// compile-time check
var _ context.Context
