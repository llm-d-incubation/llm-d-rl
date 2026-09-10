#!/usr/bin/env bash
set -xeuo pipefail

# ============================================================================
# GRPO training on SWE-reBench via the uni-agent harness: llm-d-router vs
# vanilla verl routing.
#
# This workload does NOT go through run_test.sh -- see README.md ("Not run
# through run_test.sh") for why. This script IS the launcher.
#
# Toggle ROUTING_ARM=vanilla|llmd. Every other Hydra override below is
# identical between arms by construction -- the only source of behavioral
# difference is the agent_loop_manager_class swap (+ the three EPP-only
# overrides) in the block right below.
#
#   ROUTING_ARM=vanilla bash train_swe_rebench.sh
#   ROUTING_ARM=llmd    bash train_swe_rebench.sh
# ============================================================================

ROUTING_ARM=${ROUTING_ARM:-vanilla}   # vanilla | llmd

case "$ROUTING_ARM" in
  vanilla) agent_loop_manager_class="uni_agent.framework.entry.AgentFrameworkRolloutAdapter" ;;
  llmd)    agent_loop_manager_class="uni_agent.framework.llmd_bridge.LlmdRouterAgentFramework" ;;
  *) echo "ROUTING_ARM must be 'vanilla' or 'llmd', got: $ROUTING_ARM" >&2; exit 1 ;;
esac

# Default EPP config: llm-d-rl's existing persistent prefix-cache profile
# (integrations/common/.../configs/epp/profiles/persistent.yaml, rendered as
# epp-config-persistent.yaml). See README.md "Routing policy" for the
# session-affinity upgrade (epp-config-session-affinity.yaml, shipped in this
# folder) and its measured effect.
EPP_CONFIG_FILE=${EPP_CONFIG_FILE:-/etc/llmd-configs/epp-config-persistent.yaml}
EPP_ENDPOINTS_FILE=${EPP_ENDPOINTS_FILE:-/tmp/epp-endpoints.yaml}

project_name=${PROJECT_NAME:-"Uni-Agent-Qwen3-4B-fsdp-swe-rebench"}
exp_name=${EXP_NAME:-"${ROUTING_ARM}_$(date +%Y%m%d%H%M)_exp"}

DATA_DIR=${DATA_DIR:-"/tmp/verl"}
RUNTIME_DIR=${RUNTIME_DIR:-"/tmp/verl"}

MODEL_PATH=${MODEL_PATH:-"${DATA_DIR}/models/Qwen3-4B"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/data/uni_agent/swe_rebench_filtered.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_DIR}/data/uni_agent/swe_bench_verified.parquet"}

# Relative to wherever this script is invoked from (staged on the node
# alongside uni_agent -- see README.md §1/§2).
RUNTIME_ENV=${RUNTIME_ENV:-"runtime_env.yaml"}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}

TASK_CONFIG=${TASK_CONFIG:-"task_config_react.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"hermes"}    # Qwen3-4B (plain, non-Coder) chat template
GATEWAY_COUNT=${GATEWAY_COUNT:-4}       # one gateway actor per vLLM replica
CONCURRENCY=${CONCURRENCY:-32}          # max in-flight rollout sessions (DinD is the likelier bottleneck, not GPU)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-False}

rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"}

# Algorithm parameters
adv_estimator=${ADV_ESTIMATOR:-grpo}

use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}

# Response length parameters -- single-node sized (24576 total), NOT the
# multi-node scripts' 8192+131072. SWE issue-description prompts are short;
# 18K response tokens is enough runway for a bounded (max_steps=60) ReAct loop.
max_prompt_length=${MAX_PROMPT_LENGTH:-6144}
max_response_length=${MAX_RESPONSE_LENGTH:-18432}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-2048}
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
loss_mode=${LOSS_MODE:-vanilla}

# Algorithm
temperature=${TEMPERATURE:-1.0}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# Performance / topology -- FSDP (--config-name=ppo_trainer default), no
# Megatron parallelism knobs: a 4B model needs none at this scale.
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
gen_tp=${GEN_TP:-2}
gpu_mem_util=${GPU_MEM_UTIL:-0.6}
ppo_max_token_len=$((max_prompt_length + max_response_length))

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-8}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}     # GRPO group size
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-8}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-5}
total_epochs=${TOTAL_EPOCHS:-3}

# ============================================================================
# Rollout correction is disabled by default for the standard GRPO baseline.
# ============================================================================
bypass_mode=${BYPASS_MODE:-False}
bypass_loss_type=${BYPASS_LOSS_TYPE:-ppo_clip}
rollout_is=${ROLLOUT_IS:-null}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}
rollout_rs=${ROLLOUT_RS:-null}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

# llm-d-only overrides: wired in as extra args so the vanilla arm's command is
# a strict subset of the llmd arm's, never a separately-maintained branch.
# epp_report_completion=true is required for any EPP config that scores by live
# in-flight request state (active-request-scorer) or session/session-affinity
# state keyed off request identity: with it False (the default), the client
# uses EPPGrpcClient.pick() instead of begin()/complete(), which never sends
# the x-request-id header EPP needs to correlate a session's turns or track
# real completion -- silently degrading session-affinity-scorer to an
# effectively random pick and leaving active-request-scorer's in-flight
# producer without accurate completion signals.
EPP_REPORT_COMPLETION=${EPP_REPORT_COMPLETION:-true}
llmd_overrides=()
if [[ "$ROUTING_ARM" == "llmd" ]]; then
  llmd_overrides+=(
    "++actor_rollout_ref.rollout.custom.epp_config_file=${EPP_CONFIG_FILE}"
    "++actor_rollout_ref.rollout.custom.epp_endpoints_file=${EPP_ENDPOINTS_FILE}"
    "++actor_rollout_ref.rollout.custom.epp_report_completion=${EPP_REPORT_COMPLETION}"
  )
fi

ray job submit --no-wait --runtime-env "$RUNTIME_ENV" \
    -- python3 -m verl.trainer.main_ppo \
    --config-name=ppo_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches} \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    algorithm.rollout_correction.loss_type=${bypass_loss_type} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${bypass_mode} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_rs} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${bypass_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=${agent_loop_manager_class} \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE} \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    "${llmd_overrides[@]}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=1200 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=4 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}" \
    "$@"
