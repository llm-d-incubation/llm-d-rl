"""AgentLoopManager replacement that routes generation through llm-d.

Drop-in replacement for veRL's AgentLoopManager.  Instead of spawning
Ray-managed vLLM processes, it sends HTTP requests to the llm-d Go
controller which load-balances across its pool of vLLM pods.

Usage
-----
In your veRL config:

    actor_rollout_ref:
      rollout:
        agent:
          agent_loop_manager_class: "llmd_verl.agent_loop_manager.LlmdAgentLoopManager"
        # sampling params still read from the same place:
        temperature: 1.0
        top_p: 1.0
        top_k: -1
        max_tokens: 512     # read from response_length

DataProto contract
------------------
Input (prompts):
    non_tensor_batch["raw_prompt"]  list of chat message dicts (tokenized internally)
    meta_info["validate"]           bool  (optional) — validation batch, uses val_kwargs

Output (required by fit() downstream):
    batch["prompts"]        [B, prompt_len]                original prompt ids
    batch["responses"]      [B, response_len]              generated ids (right-padded)
    batch["input_ids"]      [B, prompt_len + response_len] full sequence
    batch["attention_mask"] [B, prompt_len + response_len] 1=real, 0=pad
    batch["position_ids"]   [B, prompt_len + response_len] incremental
    batch["response_mask"]  [B, response_len]              1=generated, 0=pad
    meta_info["timing"]     dict  timing metrics
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await

from .client import AsyncRolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# ---------------------------------------------------------------------------
# Batch generator class
# ---------------------------------------------------------------------------

class _BatchGenerator:
    """Handles batch generation for a single generate_sequences call.

    Each sample flows through an independent async pipeline:
    tokenize → HTTP generate → parse → build tensors → launch reward.
    Samples run concurrently via asyncio.gather, so early samples can
    start generating while later ones are still being tokenized.
    """

    def __init__(
        self,
        prompts: DataProto,
        tokenizer: Any,
        config: Any,
        client: AsyncRolloutControllerClient,
        reward_loop_worker_handles: list | None,
    ):
        self.tokenizer = tokenizer
        self.client = client
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.input_non_tensor_batch = prompts.non_tensor_batch

        rollout_cfg = config.actor_rollout_ref.rollout

        self.temperature = rollout_cfg.temperature
        self.top_p = rollout_cfg.top_p
        self.top_k = rollout_cfg.top_k

        if prompts.meta_info.get("validate", False):
            self.temperature = rollout_cfg.val_kwargs.temperature
            self.top_p = rollout_cfg.val_kwargs.top_p
            self.top_k = rollout_cfg.val_kwargs.top_k

        self.raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        self.batch_size = len(self.raw_prompts)

        self.max_response_len = rollout_cfg.response_length
        self.max_prompt_len = rollout_cfg.prompt_length

        self.pad_token_id = prompts.meta_info.get("pad_token_id", tokenizer.pad_token_id)
        self.eos_token_id = prompts.meta_info.get("eos_token_id", tokenizer.eos_token_id)

        self.stop_strings = (
            [tokenizer.decode([self.eos_token_id])] if self.eos_token_id is not None else []
        )

    def _tokenize_one(self, messages) -> tuple[list[int], str]:
        ids = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        text = self.tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        return ids, text

    def _parse_response(self, resp: dict) -> tuple[list[int], list[float]]:
        ids = resp.get("output_token_ids") or []
        if ids:
            ids = [int(t) for t in ids]
        else:
            logger.warning("llm-d response missing output_token_ids, falling back to encoding text")
            text = resp.get("text", "")
            ids = self.tokenizer.encode(text, add_special_tokens=False) if text else []
        raw_lps = resp.get("logprobs") or []
        if raw_lps:
            lps = [list(d.values())[0] if isinstance(d, dict) else float(d) for d in raw_lps]
        else:
            lps = [0.0] * len(ids)
        return ids, lps

    def _build_padded_sample(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_lps: list[float],
    ) -> dict[str, torch.Tensor]:
        prompt_t = torch.full((1, self.max_prompt_len), self.pad_token_id, dtype=torch.long)
        prompt_mask = torch.zeros(1, self.max_prompt_len, dtype=torch.long)
        p_len = len(prompt_ids)
        if p_len > self.max_prompt_len:
            raise ValueError(
                f"Prompt length ({p_len}) exceeds max_prompt_len ({self.max_prompt_len}). "
                "Check your data pipeline or increase prompt_length in config."
            )
        prompt_t[0, :p_len] = torch.tensor(prompt_ids[:p_len], dtype=torch.long)
        prompt_mask[0, :p_len] = 1

        resp_t = torch.zeros(1, self.max_response_len, dtype=torch.long)
        resp_mask = torch.zeros(1, self.max_response_len, dtype=torch.long)
        resp_logprobs = torch.zeros(1, self.max_response_len, dtype=torch.float32)
        r_len = len(response_ids)
        if r_len > self.max_response_len:
            raise ValueError(
                f"Response length ({r_len}) exceeds max_response_len ({self.max_response_len}). "
                "Check your generation config or increase response_length in config."
            )
        if r_len > 0:
            resp_t[0, :r_len] = torch.tensor(response_ids[:r_len], dtype=torch.long)
            resp_mask[0, :r_len] = 1
            lp_len = min(len(response_lps), r_len)
            resp_logprobs[0, :lp_len] = torch.tensor(response_lps[:lp_len], dtype=torch.float32)

        full_ids = torch.cat([prompt_t, resp_t], dim=1)
        full_mask = torch.cat([prompt_mask, resp_mask], dim=1)
        position_ids = compute_position_id_with_mask(full_mask)

        return {
            "prompt_t": prompt_t,
            "resp_t": resp_t,
            "resp_mask": resp_mask,
            "resp_logprobs": resp_logprobs,
            "full_ids": full_ids,
            "full_mask": full_mask,
            "position_ids": position_ids,
        }

    def _launch_reward(self, idx: int, tensors: dict) -> Any:
        item_batch = TensorDict(
            {
                "prompts": tensors["prompt_t"],
                "responses": tensors["resp_t"],
                "input_ids": tensors["full_ids"],
                "attention_mask": tensors["full_mask"],
                "position_ids": tensors["position_ids"],
                "response_mask": tensors["resp_mask"],
            },
            batch_size=[1],
        )
        item_non_tensor = {k: np.array([v[idx]]) for k, v in self.input_non_tensor_batch.items()}
        item_non_tensor["__num_turns__"] = np.array([2], dtype=np.int32)
        item_non_tensor["tool_extra_fields"] = np.array([{}], dtype=object)
        if "multi_modal_inputs" not in item_non_tensor:
            item_non_tensor["multi_modal_inputs"] = np.array([{}], dtype=object)
        item_data = DataProto(batch=item_batch, non_tensor_batch=item_non_tensor)

        handle = random.choice(self.reward_loop_worker_handles)
        return handle.compute_score.remote(item_data)

    async def _process_sample(self, idx: int) -> tuple[int, dict, Any | None, tuple | None]:
        """Per-sample async pipeline: tokenize → generate → parse → tensors → reward."""
        prompt_ids, prompt_text = self._tokenize_one(self.raw_prompts[idx])

        resp = await self.client.generate(
            prompt=prompt_text,
            prompt_token_ids=prompt_ids,
            max_tokens=self.max_response_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self.stop_strings,
        )

        response_ids, response_lps = self._parse_response(resp)
        tensors = self._build_padded_sample(prompt_ids, response_ids, response_lps)

        reward_ref = None
        if self.reward_loop_worker_handles:
            reward_ref = self._launch_reward(idx, tensors)

        debug_info = (idx, resp, prompt_text)
        return idx, tensors, reward_ref, debug_info

    async def generate_and_collect_results(self) -> tuple[list, list | None, tuple | None]:
        """Run all samples concurrently via asyncio.gather.

        Returns:
            sample_tensors: List of per-sample tensor dicts (ordered by index).
            reward_refs: List of Ray ObjectRefs (or None if no reward workers).
            sample_debug: First sample's (idx, resp, prompt_text) for debug logging.
        """
        results = await asyncio.gather(
            *[self._process_sample(i) for i in range(self.batch_size)]
        )

        sample_tensors: list[dict | None] = [None] * self.batch_size
        reward_refs: list | None = [None] * self.batch_size if self.reward_loop_worker_handles else None
        sample_debug: tuple | None = None

        for idx, tensors, reward_ref, debug_info in results:
            sample_tensors[idx] = tensors
            if reward_refs is not None:
                reward_refs[idx] = reward_ref
            if sample_debug is None:
                sample_debug = debug_info

        return sample_tensors, reward_refs, sample_debug

    async def finalize_rewards(
        self,
        reward_refs: list | None,
        full_mask: torch.Tensor,
        t_end: float,
    ) -> tuple[torch.Tensor, list[str], dict[str, np.ndarray], float | None]:
        if reward_refs is None:
            return (
                torch.zeros(self.batch_size, self.max_response_len, dtype=torch.float32),
                [],
                {},
                None,
            )

        reward_results = await asyncio.to_thread(ray.get, reward_refs)
        scores = [r["reward_score"] for r in reward_results]

        valid_resp_lens = full_mask[:, self.max_prompt_len:].sum(dim=1)
        rm_scores = torch.zeros(self.batch_size, self.max_response_len, dtype=torch.float32)
        for i, (score, vlen) in enumerate(zip(scores, valid_resp_lens)):
            if int(vlen.item()) > 0:
                rm_scores[i, int(vlen.item()) - 1] = float(score)

        reward_extra_keys = list(reward_results[0].get("reward_extra_info", {}).keys())
        reward_extra_info = {
            key: np.array([r["reward_extra_info"][key] for r in reward_results])
            for key in reward_extra_keys
        }

        straggler_wait = time.perf_counter() - t_end

        return rm_scores, reward_extra_keys, reward_extra_info, straggler_wait


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _log_sample_debug(sample_debug: tuple | None) -> None:
    """Log debug info for the first completed generation sample.

    Logs prompt head (first 200 chars), response tail (last 200 chars),
    logprobs presence, weight version, and finish reason at DEBUG level.

    Args:
        sample_debug: Tuple of (idx, response_dict, prompt_text) or None.
    """
    if not sample_debug:
        return

    idx, resp, prompt_text = sample_debug
    text = resp.get("text") or ""
    prompt_head = prompt_text[:200]
    resp_tail = text[-200:] if len(text) > 200 else text

    raw_lps = resp.get("logprobs") or []
    weight_version = resp.get("weight_version", "(none)")
    finish_reason = resp.get("finish_reason") or resp.get("stop_reason") or "(none)"

    logger.debug("[LLMD] generate batch sample (first completed) idx=%d", idx)
    logger.debug("  prompt_head (200): %r", prompt_head)
    logger.debug("  response_tail (200): %r", resp_tail)
    logger.debug("  logprobs present: %s", bool(raw_lps))
    if raw_lps:
        first3_lp: list[float] = []
        for d in raw_lps[:3]:
            if isinstance(d, dict):
                first3_lp.append(float(list(d.values())[0]))
            else:
                first3_lp.append(float(d))
        logger.debug("  first 3 logprobs: %s", first3_lp)
    logger.debug("  weight_version: %s", weight_version)
    logger.debug("  finish_reason: %s", finish_reason)


def _extract_llmd_config(verl_config) -> LlmdRolloutConfig:
    """Pull llm-d settings from veRL's rollout config."""
    rollout = verl_config.actor_rollout_ref.rollout
    kw = rollout.get("custom", {}).get("llmd", {})
    return LlmdRolloutConfig(**kw)

# ------------------------------------------------------------------
# veRL interface 
# ------------------------------------------------------------------

class LlmdAgentLoopManager:
    """veRL AgentLoopManager that uses llm-d for sequence generation.

    veRL's ray_trainer.py constructs this via create():

        manager = LlmdAgentLoopManager.create(
            config=config,
            worker_group=actor_rollout_wg,    # ignored — llm-d owns inference GPUs
            rollout_resource_pool=pool,        # ignored
        )

    and calls:
        output = manager.generate_sequences(prompts_dataproto)
        replicas = manager.rollout_replicas   # empty list — no veRL-managed pods
    """

    def __init__(self, config, client: AsyncRolloutControllerClient, tokenizer=None,
                 reward_loop_worker_handles=None):
        self.config = config
        self.client = client
        self.tokenizer = tokenizer
        self.reward_loop_worker_handles = reward_loop_worker_handles
        # Empty — LlmdVerlCheckpointEngineManager ignores replicas
        self.rollout_replicas: list = []

    @classmethod
    def create(
        cls,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
    ):
        """Create manager.  worker_group / resource_pool are ignored."""
        llmd_config = _extract_llmd_config(config)
        client = AsyncRolloutControllerClient(llmd_config)

        # Load tokenizer so we can tokenize raw_prompt from non_tensor_batch.
        from transformers import AutoTokenizer

        model_path = config.actor_rollout_ref.model.path
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                "LlmdAgentLoopManager could not load the tokenizer from "
                f"actor_rollout_ref.model.path ({model_path!r}). "
                "Set a valid, reachable Hugging Face model id or local path."
            ) from exc
        logger.debug("[LLMD] Loaded tokenizer from %s", model_path)

        return cls(config, client, tokenizer,
                   reward_loop_worker_handles=reward_loop_worker_handles)

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate responses for a batch of prompts via llm-d HTTP API.

        Args:
            prompts: DataProto with batch["input_ids"] and batch["attention_mask"].

        Returns:
            DataProto with all fields required by veRL's fit() loop.
        """
        t_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Build batch generator
        # ------------------------------------------------------------------
        if self.tokenizer is None:
            raise RuntimeError(
                "LlmdAgentLoopManager requires a tokenizer. "
                "Use LlmdAgentLoopManager.create() so the tokenizer is loaded from "
                "actor_rollout_ref.model.path, or set a valid model path."
            )

        gen = _BatchGenerator(
            prompts=prompts,
            tokenizer=self.tokenizer,
            config=self.config,
            client=self.client,
            reward_loop_worker_handles=self.reward_loop_worker_handles,
        )

        # ------------------------------------------------------------------
        # Run per-sample async pipelines concurrently
        # ------------------------------------------------------------------
        t_gen_start = time.perf_counter()

        sample_tensors, reward_refs, sample_debug = await gen.generate_and_collect_results()
        t_gen_end = time.perf_counter()

        _log_sample_debug(sample_debug)
        # ------------------------------------------------------------------
        # Stack pre-padded tensors into batch (just torch.cat)
        #
        # All samples already have identical shapes, so this is trivial.
        # ------------------------------------------------------------------
        prompt_ids = torch.cat([s["prompt_t"] for s in sample_tensors], dim=0)
        resp_ids = torch.cat([s["resp_t"] for s in sample_tensors], dim=0)
        resp_mask = torch.cat([s["resp_mask"] for s in sample_tensors], dim=0)
        rollout_log_probs = torch.cat([s["resp_logprobs"] for s in sample_tensors], dim=0)
        full_ids = torch.cat([s["full_ids"] for s in sample_tensors], dim=0)
        full_mask = torch.cat([s["full_mask"] for s in sample_tensors], dim=0)
        position_ids = torch.cat([s["position_ids"] for s in sample_tensors], dim=0)

        # ------------------------------------------------------------------
        # Build output DataProto
        # ------------------------------------------------------------------
        out_batch = TensorDict(
            {
                "prompts":           prompt_ids,              # [B, max_prompt_len]
                "responses":         resp_ids,                # [B, max_response_len]
                "input_ids":         full_ids,                # [B, max_prompt_len + max_response_len]
                "attention_mask":    full_mask,               # [B, max_prompt_len + max_response_len]
                "position_ids":      position_ids,            # [B, max_prompt_len + max_response_len]
                "response_mask":     resp_mask,               # [B, max_response_len]
                "rollout_log_probs": rollout_log_probs,       # [B, max_response_len]
            },
            batch_size=[gen.batch_size],
        )

        t_end = time.perf_counter()
        timing = {
            "generate_sequences": t_end - t_start,
            "llmd/http_generate":  t_gen_end - t_gen_start,
        }

        out_non_tensor = dict(prompts.non_tensor_batch)
        # veRL's training loop unconditionally iterates non_tensor_batch["multi_modal_inputs"]
        # (ray_trainer.py). Add empty-dict entries for text-only batches so it
        # doesn't KeyError and finds nothing to process (same as what the standard
        # AgentLoopWorker does for non-vision samples).
        if "multi_modal_inputs" not in out_non_tensor:
            out_non_tensor["multi_modal_inputs"] = np.array(
                [{} for _ in range(gen.batch_size)], dtype=object
            )

        # ------------------------------------------------------------------
        # Finalize reward scores
        # ------------------------------------------------------------------
        rm_scores, reward_extra_keys, reward_extra_info, straggler_wait = await gen.finalize_rewards(
            reward_refs, full_mask, t_end
        )
        out_batch["rm_scores"] = rm_scores
        out_non_tensor.update(reward_extra_info)
        if straggler_wait is not None:
            timing["llmd/reward_straggler_wait"] = straggler_wait

        output = DataProto(
            batch=out_batch,
            non_tensor_batch=out_non_tensor,
            meta_info={**prompts.meta_info, "timing": timing,
                       "reward_extra_keys": reward_extra_keys},
        )

        logger.info(
            "generate_sequences: B=%d prompt_len=%d resp_len=%d t=%.2fs",
            gen.batch_size,
            gen.max_prompt_len,
            gen.max_response_len,
            t_end - t_start,
        )

        return output

    # ------------------------------------------------------------------
    # stubs for optional methods called in fit()
    # ------------------------------------------------------------------

    @auto_await
    async def clear_kv_cache(self) -> None:
        """No-op: llm-d manages cache reset readiness externally."""
        return

    def start_profile(self, **kwargs) -> None:
        """No-op: profiling not supported via HTTP API."""

    def stop_profile(self) -> None:
        """No-op: profiling not supported via HTTP API."""