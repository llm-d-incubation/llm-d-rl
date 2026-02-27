// Package rollout implements the rollout controller HTTP/gRPC server.
// This is the primary entry point for RL training frameworks to interact
// with the llm-d rollout infrastructure.
package rollout

import (
	"context"
	"encoding/json"
	"fmt"
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

	// TODO: Route through inference scheduler (EPP) for KV-cache-aware dispatch.
	// For now, return a placeholder indicating the generation pipeline is not yet wired.
	http.Error(w, "generation routing not yet implemented — will integrate with EPP", http.StatusNotImplemented)
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
	// TODO: Fan out pause to all engines
	http.Error(w, "not yet implemented", http.StatusNotImplemented)
}

func (s *Server) handleResume(w http.ResponseWriter, r *http.Request) {
	// TODO: Fan out resume to all engines
	http.Error(w, "not yet implemented", http.StatusNotImplemented)
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
	var req struct {
		Tags []string `json:"tags"`
	}
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

// SleepRequest is the JSON body for the sleep endpoint.
type SleepRequest = struct {
	Level int `json:"level"`
}

// compile-time check
var _ context.Context
