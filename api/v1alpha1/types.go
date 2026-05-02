// Package v1alpha1 defines the API types for the llm-d RL rollout infrastructure.
//
// These types define the contract between RL training frameworks and the llm-d
// rollout controller. The API is designed to be framework-agnostic — any training
// loop (veRL, OpenRLHF, SkyRL, NeMo-RL, or custom) can consume it via HTTP/gRPC.
package v1alpha1

// PauseMode controls how in-flight requests are handled during weight updates.
type PauseMode string

const (
	// PauseModeAbort aborts all in-flight requests immediately.
	// Fastest, but discards partial work.
	PauseModeAbort PauseMode = "abort"

	// PauseModeWait stops accepting new requests and waits for in-flight
	// requests to complete before proceeding.
	PauseModeWait PauseMode = "wait"

	// PauseModeKeep freezes all in-flight requests in place. After the
	// weight update completes, frozen requests resume from where they left off.
	// Most efficient but requires vLLM v1+ with PauseMode.KEEP support.
	PauseModeKeep PauseMode = "keep"
)

// SleepLevel controls the GPU memory management behavior during sleep.
type SleepLevel int

const (
	// SleepLevel0 pauses scheduling only. No GPU memory changes.
	SleepLevel0 SleepLevel = 0

	// SleepLevel1 offloads model weights to CPU memory. KV cache is discarded.
	// Good for sleeping and waking with the same model.
	SleepLevel1 SleepLevel = 1

	// SleepLevel2 discards all GPU memory (weights + KV cache).
	// Good for weight update scenarios where previous weights are not needed.
	// Preferred for RL colocated deployments (vLLM >= 0.8.5).
	SleepLevel2 SleepLevel = 2
)

// WeightSyncBackend identifies the transport mechanism for weight transfer.
type WeightSyncBackend string

const (
	// WeightSyncNCCL uses NCCL collective broadcast for weight transfer.
	// Works for both colocated and disaggregated deployments.
	WeightSyncNCCL WeightSyncBackend = "nccl"

	// WeightSyncNIXL uses NIXL/RDMA for high-bandwidth cross-node transfer.
	WeightSyncNIXL WeightSyncBackend = "nixl"

	// WeightSyncCheckpoint loads weights from a shared filesystem path.
	WeightSyncCheckpoint WeightSyncBackend = "checkpoint"
)

// PoolPhase represents the current operational phase of the engine pool.
type PoolPhase string

const (
	PoolPhaseServing  PoolPhase = "Serving"
	PoolPhaseSleeping PoolPhase = "Sleeping"
	PoolPhaseSyncing  PoolPhase = "Syncing"
	PoolPhaseRolling  PoolPhase = "Rolling"
)

// --- Request/Response Types ---

// GenerateRequest is a request to generate token sequences from prompts.
//
// Callers set either Prompt (text string) or PromptTokenIDs (pre-tokenized),
// depending on the inference target:
//   - Inference router (llm-d): set Prompt — the router tokenizes the text
//     for prefix-cache routing and forwards to the engine.
//   - Direct engine: set PromptTokenIDs — token IDs are passed directly
//     to the engine's /v1/completions endpoint.
type GenerateRequest struct {
	// Prompt is the raw text prompt. When set, it is sent as the "prompt"
	// string in the OpenAI completions request. Used when routing through
	// the llm-d inference scheduler, which tokenizes the text for
	// prefix-cache-aware routing.
	Prompt string `json:"prompt,omitempty"`

	// PromptTokenIDs is the tokenized input prompt. Used for direct
	// engine dispatch where the engine accepts token IDs natively.
	PromptTokenIDs []int32 `json:"prompt_token_ids,omitempty"`

	// SamplingParams controls generation behavior (temperature, top_p, max_tokens, etc.).
	SamplingParams *SamplingParams `json:"sampling_params,omitempty"`

	// SessionID enables multi-turn session affinity. Requests with the same
	// session ID are routed to the same engine for KV cache reuse.
	SessionID string `json:"session_id,omitempty"`

	// ReturnLogprobs requests per-token log probabilities in the response.
	ReturnLogprobs bool `json:"return_logprobs,omitempty"`

	// ReturnTokenIDs requests generated token IDs in the response when supported
	// by the underlying engine endpoint.
	ReturnTokenIDs bool `json:"return_token_ids,omitempty"`

	// WeightVersion is the expected weight version. If set, the controller
	// validates that engines are running this version before dispatching.
	WeightVersion int64 `json:"weight_version,omitempty"`
}

// GenerateResponse contains the output of a generation request.
type GenerateResponse struct {
	// OutputTokenIDs is the generated token sequence.
	OutputTokenIDs []int32 `json:"output_token_ids"`

	// Logprobs contains per-token log probabilities (if requested).
	Logprobs []float32 `json:"logprobs,omitempty"`

	// WeightVersion is the weight version of the engine that generated this output.
	WeightVersion int64 `json:"weight_version"`

	// EngineID identifies which engine served this request.
	EngineID string `json:"engine_id"`

	// FinishReason indicates why generation stopped.
	FinishReason string `json:"finish_reason"` // "stop", "length", "abort"

	// Text is the generated text returned by the engine or router.
	Text string `json:"text,omitempty"`
}

// SamplingParams controls the generation sampling behavior.
type SamplingParams struct {
	Temperature    float32 `json:"temperature,omitempty"`
	TopP           float32 `json:"top_p,omitempty"`
	TopK           int     `json:"top_k,omitempty"`
	MaxTokens      int     `json:"max_tokens,omitempty"`
	NSamples       int     `json:"n,omitempty"`
	StopStrings    []string `json:"stop,omitempty"`
}

// WeightTransferInit configures the weight transfer data plane.
type WeightTransferInit struct {
	// Backend selects the transport mechanism.
	Backend WeightSyncBackend `json:"backend"`

	// MasterAddress is the trainer's address for NCCL rendezvous.
	MasterAddress string `json:"master_address"`

	// MasterPort is the trainer's port for NCCL rendezvous.
	MasterPort int32 `json:"master_port"`

	// TrainerWorldSize is the number of trainer ranks participating.
	TrainerWorldSize int32 `json:"trainer_world_size"`

	// Packed enables packed tensor transfer (multiple weights per buffer).
	Packed bool `json:"packed,omitempty"`

	// IsCheckpointFormat indicates weights are in checkpoint format (e.g., bf16)
	// and need requantization for the inference kernel (e.g., fp8).
	IsCheckpointFormat bool `json:"is_checkpoint_format,omitempty"`
}

// WeightUpdateRequest triggers a weight update across the engine pool.
type WeightUpdateRequest struct {
	// TargetVersion is the new weight version after the update.
	TargetVersion int64 `json:"target_version"`

	// PauseMode controls how in-flight requests are handled.
	PauseMode PauseMode `json:"pause_mode,omitempty"`

	// ResetKVCache indicates whether to invalidate KV caches after the update.
	// Should be true for policy weight updates (stale KV cache), false for
	// LoRA adapter updates (KV cache remains valid).
	ResetKVCache bool `json:"reset_kv_cache,omitempty"`

	// ParamNames lists the parameter names to update (must match model parameter order).
	ParamNames []string `json:"param_names,omitempty"`

	// ParamDtypes lists the dtype name for each parameter (e.g., "torch.bfloat16").
	ParamDtypes []string `json:"param_dtypes,omitempty"`

	// ParamShapes lists the shape of each parameter as a list of ints.
	ParamShapes [][]int `json:"param_shapes,omitempty"`

	// Packed enables packed tensor broadcasting for efficient NCCL transfer.
	// When true, multiple tensors are batched into buffers before broadcasting.
	Packed bool `json:"packed,omitempty"`

	// PackedBufferSizeBytes is the size of each packed buffer in bytes.
	// Both trainer and vLLM must use the same value. Default is 1GB.
	PackedBufferSizeBytes int64 `json:"packed_buffer_size_bytes,omitempty"`

	// PackedNumBuffers is the number of buffers for double/triple buffering.
	// Both trainer and vLLM must use the same value. Default is 2.
	PackedNumBuffers int `json:"packed_num_buffers,omitempty"`
}

// PoolStatus reports the current state of the engine pool.
type PoolStatus struct {
	// Phase is the current operational phase.
	Phase PoolPhase `json:"phase"`

	// TotalEngines is the total number of engines in the pool.
	TotalEngines int `json:"total_engines"`

	// ReadyEngines is the number of healthy, ready engines.
	ReadyEngines int `json:"ready_engines"`

	// WeightVersion is the current weight version across the pool.
	WeightVersion int64 `json:"weight_version"`

	// LastWeightSync is the timestamp of the last successful weight sync.
	LastWeightSync string `json:"last_weight_sync,omitempty"`

	// Engines contains per-engine status.
	Engines []EngineStatus `json:"engines,omitempty"`
}

// SleepRequest triggers a sleep across the engine pool.
type SleepRequest struct {
	// Level controls GPU memory behavior during sleep.
	// See SleepLevel constants for valid values.
	Level int `json:"level"`
}

// WakeUpRequest triggers a wake-up across the engine pool.
type WakeUpRequest struct {
	// Tags specifies which resources to restore.
	// Valid tags: "weights", "kv_cache"
	Tags []string `json:"tags"`
}

// EngineStatus reports the status of a single inference engine.
type EngineStatus struct {
	// ID uniquely identifies this engine.
	ID string `json:"id"`

	// Address is the engine's network address (host:port).
	Address string `json:"address"`

	// Ready indicates whether the engine is healthy and accepting requests.
	Ready bool `json:"ready"`

	// WeightVersion is this engine's current weight version.
	WeightVersion int64 `json:"weight_version"`

	// QueueDepth is the number of in-flight requests on this engine.
	QueueDepth int `json:"queue_depth"`

	// KVCacheUtilization is the fraction of KV cache in use (0.0–1.0).
	KVCacheUtilization float64 `json:"kv_cache_utilization"`
}
