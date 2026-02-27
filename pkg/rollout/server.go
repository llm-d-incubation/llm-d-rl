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

	mu            sync.RWMutex
	weightVersion int64
}

// NewServer creates a new rollout controller server.
func NewServer(pool *lifecycle.PoolManager, coordinator *weightsync.Coordinator) *Server {
	return &Server{
		pool:        pool,
		coordinator: coordinator,
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

	// Pick a ready engine from the pool.
	// TODO: Replace with EPP routing for KV-cache-aware dispatch.
	engine, err := s.pool.PickReadyEngine()
	if err != nil {
		http.Error(w, fmt.Sprintf("no engines available: %v", err), http.StatusServiceUnavailable)
		return
	}

	// Translate to OpenAI completions format and forward.
	resp, err := s.forwardToEngine(r.Context(), engine, &req)
	if err != nil {
		http.Error(w, fmt.Sprintf("generate: %v", err), http.StatusBadGateway)
		return
	}

	s.mu.RLock()
	resp.WeightVersion = s.weightVersion
	s.mu.RUnlock()
	resp.EngineID = engine.ID

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// forwardToEngine translates a GenerateRequest to OpenAI /v1/completions
// format, forwards it to the engine, and translates the response back.
func (s *Server) forwardToEngine(ctx context.Context, engine *lifecycle.EngineInfo, req *v1alpha1.GenerateRequest) (*v1alpha1.GenerateResponse, error) {
	// Build OpenAI completions request.
	oaiReq := map[string]interface{}{
		"model":  "default",
		"prompt": req.PromptTokenIDs,
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

	body, err := json.Marshal(oaiReq)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost,
		engine.Address+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("POST /v1/completions: %w", err)
	}
	defer httpResp.Body.Close()

	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if httpResp.StatusCode >= 400 {
		return nil, fmt.Errorf("engine returned %d: %s", httpResp.StatusCode, string(respBody))
	}

	// Parse OpenAI completions response.
	var oaiResp struct {
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
			Logprobs     *struct {
				TokenLogprobs []float32 `json:"token_logprobs"`
				Tokens        []string  `json:"tokens"`
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
		// Convert output text to token IDs (simple byte-level fallback).
		for _, c := range choice.Text {
			resp.OutputTokenIDs = append(resp.OutputTokenIDs, int32(c))
		}
		if choice.Logprobs != nil {
			resp.Logprobs = choice.Logprobs.TokenLogprobs
		}
	}

	return resp, nil
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
