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
	"github.com/llm-d/llm-d-rl/pkg/lifecycle"
	"github.com/llm-d/llm-d-rl/pkg/weightsync"
)

// Server is the rollout controller HTTP server.
// It exposes the RolloutControl API surface for RL training frameworks.
type Server struct {
	pool        *lifecycle.PoolManager
	coordinator *weightsync.Coordinator

	// routerURL is the HTTP base URL of the inference router gateway (e.g.
	// llm-d inference scheduler behind Envoy/Istio). When set, inference
	// requests are forwarded here for prefix-cache-aware routing.
	// When empty, requests are dispatched directly to an engine from the pool.
	routerURL string

	mu               sync.RWMutex
	weightVersion    int64
	cachedModelName  string // lazily populated on first /v1/models query
}

// NewServer creates a new rollout controller server.
// routerURL is the HTTP base URL of the inference router/gateway endpoint;
// pass "" to use direct engine dispatch (fallback for local development).
func NewServer(pool *lifecycle.PoolManager, coordinator *weightsync.Coordinator, routerURL string) *Server {
	return &Server{
		pool:        pool,
		coordinator: coordinator,
		routerURL:   routerURL,
	}
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

	var resp *v1alpha1.GenerateResponse
	var err error

	if s.routerURL != "" {
		// Route through the inference router gateway for prefix-cache-aware dispatch.
		resp, err = s.forwardToRouter(r.Context(), &req)
		if err != nil {
			http.Error(w, fmt.Sprintf("generate via router: %v", err), http.StatusBadGateway)
			return
		}
	} else {
		// Fallback: direct dispatch to a ready engine from the pool.
		engine, pickErr := s.pool.PickReadyEngine()
		if pickErr != nil {
			http.Error(w, fmt.Sprintf("no engines available: %v", pickErr), http.StatusServiceUnavailable)
			return
		}
		resp, err = s.forwardToEngine(r.Context(), engine, &req)
		if err != nil {
			http.Error(w, fmt.Sprintf("generate: %v", err), http.StatusBadGateway)
			return
		}
		resp.EngineID = engine.ID
	}

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
// The prompt format depends on which field is set in the GenerateRequest:
//
//   - req.Prompt (text string): sent as "prompt" for the inference router
//     path. The llm-d inference scheduler tokenizes this text for
//     prefix-cache-aware routing, then forwards to the engine.
//
//   - req.PromptTokenIDs (int array): sent as "prompt" for the direct
//     engine path. vLLM's /v1/completions endpoint accepts token ID
//     arrays natively.
func (s *Server) buildOAIRequest(ctx context.Context, baseURL string, req *v1alpha1.GenerateRequest) ([]byte, error) {
	oaiReq := map[string]interface{}{
		"model": s.discoverModelName(ctx, baseURL),
	}
	if req.Prompt != "" {
		// Router path: send the actual text prompt so the inference
		// scheduler can tokenize it for prefix-cache routing.
		oaiReq["prompt"] = req.Prompt
	} else {
		// Direct-to-engine: vLLM accepts prompt as an int array natively.
		oaiReq["prompt"] = req.PromptTokenIDs
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
func parseOAIResponse(respBody []byte) (*v1alpha1.GenerateResponse, error) {
	var oaiResp struct {
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
			Logprobs     *struct {
				TokenLogprobs []float32 `json:"token_logprobs"`
			} `json:"logprobs"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(respBody, &oaiResp); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}
	resp := &v1alpha1.GenerateResponse{}
	if len(oaiResp.Choices) > 0 {
		choice := oaiResp.Choices[0]
		resp.FinishReason = choice.FinishReason
		for _, c := range choice.Text {
			resp.OutputTokenIDs = append(resp.OutputTokenIDs, int32(c))
		}
		if choice.Logprobs != nil {
			resp.Logprobs = choice.Logprobs.TokenLogprobs
		}
	}
	return resp, nil
}

// postCompletions is the shared implementation for both router and direct-engine
// dispatch. It builds the OAI request body, POSTs to baseURL/v1/completions,
// applies any caller-supplied extra headers, and parses the response.
func (s *Server) postCompletions(ctx context.Context, baseURL string, req *v1alpha1.GenerateRequest, extraHeaders map[string]string) (*v1alpha1.GenerateResponse, error) {
	body, err := s.buildOAIRequest(ctx, baseURL, req)
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

// forwardToRouter sends a GenerateRequest to the inference router gateway.
// session_id is forwarded as X-Session-ID for KV-cache session affinity.
// The request's Prompt field (text string) is used as the "prompt" value
// so the inference scheduler can tokenize it for prefix-cache routing.
func (s *Server) forwardToRouter(ctx context.Context, req *v1alpha1.GenerateRequest) (*v1alpha1.GenerateResponse, error) {
	headers := map[string]string{}
	if req.SessionID != "" {
		headers["X-Session-ID"] = req.SessionID
	}
	return s.postCompletions(ctx, s.routerURL, req, headers)
}

// forwardToEngine sends a GenerateRequest directly to a vLLM engine.
// The request's PromptTokenIDs are passed directly in "prompt" as an int
// array, which vLLM accepts natively.
func (s *Server) forwardToEngine(ctx context.Context, engine *lifecycle.EngineInfo, req *v1alpha1.GenerateRequest) (*v1alpha1.GenerateResponse, error) {
	return s.postCompletions(ctx, engine.Address, req, nil)
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

	if err := s.coordinator.InitTransfer(r.Context(), &init); err != nil {
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

	if err := s.coordinator.UpdateWeights(r.Context(), &req); err != nil {
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

	if err := s.coordinator.PauseAll(r.Context(), req.Mode); err != nil {
		http.Error(w, fmt.Sprintf("pause: %v", err), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "paused"})
}

func (s *Server) handleResume(w http.ResponseWriter, r *http.Request) {
	if err := s.coordinator.ResumeAll(r.Context()); err != nil {
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

	if err := s.coordinator.SleepAll(r.Context(), v1alpha1.SleepLevel(req.Level)); err != nil {
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

	if err := s.coordinator.WakeUpAll(r.Context(), req.Tags); err != nil {
		http.Error(w, fmt.Sprintf("wake up: %v", err), http.StatusInternalServerError)
		return
	}

	s.pool.SetPhase(v1alpha1.PoolPhaseServing)

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "awake"})
}

func (s *Server) handlePoolStatus(w http.ResponseWriter, r *http.Request) {
	status := s.pool.Status()
	status.WeightVersion = s.coordinator.WeightVersion()
	json.NewEncoder(w).Encode(status)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

// compile-time check
var _ context.Context
