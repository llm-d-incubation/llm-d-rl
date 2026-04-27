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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import ray
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await

from .client import RolloutControllerClient
from .config import LlmdRolloutConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Maximum concurrent HTTP requests to the controller.
_MAX_CONCURRENT = 1024


# ---------------------------------------------------------------------------
# Batch generator class
# ---------------------------------------------------------------------------

class _BatchGenerator:
    """Handles batch generation for a single generate_sequences call.

    Encapsulates all state and operations needed to generate responses for a
    batch of prompts: tokenization, HTTP requests to llm-d, tensor building,
    and reward computation.
    """

    def __init__(
        self,
        prompts: DataProto,
        tokenizer: Any,
        config: Any,
        client: RolloutControllerClient,
        reward_loop_worker_handles: list | None,
    ):
        """Build generation context from prompts and config.

        Args:
            prompts: DataProto with non_tensor_batch["raw_prompt"] and meta_info.
            tokenizer: HuggingFace tokenizer with apply_chat_template support.
            config: veRL config with actor_rollout_ref.rollout settings.
            client: HTTP client for llm-d controller.
            reward_loop_worker_handles: Ray actor handles for reward workers (or None).
        """
        self.tokenizer = tokenizer
        self.client = client
        self.reward_loop_worker_handles = reward_loop_worker_handles
        self.input_non_tensor_batch = prompts.non_tensor_batch

        rollout_cfg = config.actor_rollout_ref.rollout

        # Sampling parameters
        self.temperature = rollout_cfg.temperature
        self.top_p = rollout_cfg.top_p
        self.top_k = rollout_cfg.top_k

        # Override for validation batches
        if prompts.meta_info.get("validate", False):
            self.temperature = rollout_cfg.val_kwargs.temperature
            self.top_p = rollout_cfg.val_kwargs.top_p
            self.top_k = rollout_cfg.val_kwargs.top_k

        # Tokenize prompts
        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        self.batch_size = len(raw_prompts)
        self.prompt_token_ids, self.prompt_texts, prompt_lengths = self._tokenize_prompts(raw_prompts)

        # Fixed lengths (config or fallback to actual max)
        self.max_response_len = rollout_cfg.response_length
        self.max_prompt_len = rollout_cfg.prompt_length

        # Token IDs — match veRL's fallback pattern (meta_info → tokenizer)
        self.pad_token_id = prompts.meta_info.get("pad_token_id", tokenizer.pad_token_id)
        self.eos_token_id = prompts.meta_info.get("eos_token_id", tokenizer.eos_token_id)

        # Stop strings for llm-d HTTP API (veRL's vLLM handles EOS internally)
        self.stop_strings = (
            [tokenizer.decode([self.eos_token_id])] if self.eos_token_id is not None else []
        )

    def _tokenize_prompts(
        self,
        raw_prompts: list,
    ) -> tuple[list[list[int]], list[str], list[int]]:
        """Apply chat template to raw prompts and return token IDs + text.

        Args:
            raw_prompts: List of chat message lists from non_tensor_batch["raw_prompt"].

        Returns:
            prompt_token_ids: List of token ID lists (one per prompt).
            prompt_texts: List of formatted prompt strings (one per prompt).
            prompt_lengths: List of token counts (one per prompt).
        """
        prompt_token_ids: list[list[int]] = []
        prompt_texts: list[str] = []
        prompt_lengths: list[int] = []

        for messages in raw_prompts:
            # apply_chat_template with tokenize=True returns token IDs directly;
            # with tokenize=False returns the formatted string. We need both.
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
            prompt_token_ids.append(ids)
            prompt_texts.append(text)
            prompt_lengths.append(len(ids))

        return prompt_token_ids, prompt_texts, prompt_lengths

    def generate_one(self, idx: int, ids: list[int], text: str) -> tuple[int, dict]:
        """Send one prompt to llm-d controller via HTTP POST.

        Args:
            idx: Index of this prompt in the batch.
            ids: Token IDs for the prompt.
            text: Formatted prompt string.

        Returns:
            Tuple of (idx, response_dict) where response_dict contains
            output_token_ids, logprobs, text, weight_version, finish_reason.
        """
        kwargs = dict(
            prompt=text,
            prompt_token_ids=ids,
            max_tokens=self.max_response_len,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop=self.stop_strings,
        )
        resp = self.client.generate(**kwargs)
        return idx, resp

    def parse_response(self, resp: dict) -> tuple[list[int], list[float]]:
        """Extract token IDs and logprobs from llm-d response.

        Falls back to encoding resp["text"] if output_token_ids is missing.
        Returns zeros for logprobs if not present in response.

        Args:
            resp: Response dict from llm-d controller.

        Returns:
            ids: List of output token IDs.
            lps: List of per-token logprobs (same length as ids).
        """
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

    def build_padded_sample(
        self,
        prompt_ids: list[int],
        response_ids: list[int],
        response_lps: list[float],
    ) -> dict[str, torch.Tensor]:
        """Build single-sample tensors padded to fixed config lengths.

        Like veRL's agent_loop.py, we pad prompts to prompt_length and
        responses to response_length so all samples have identical shapes.

        Args:
            prompt_ids: Token IDs for the prompt.
            response_ids: Token IDs for the generated response.
            response_lps: Per-token logprobs for the response.

        Returns:
            Dict with keys: prompt_t, resp_t, resp_mask, resp_logprobs,
            full_ids, full_mask, position_ids. All tensors have shape [1, *].
        """
        # Pad prompt
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

        # Pad response
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

        # Full sequence
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

    def launch_reward(self, idx: int, tensors: dict) -> Any:
        """Launch reward computation for one sample via Ray remote call.

        veRL's reward workers expect one sample per call, wrapped in a DataProto.
        Builds a single-sample DataProto from pre-built tensors and fires a
        non-blocking Ray RPC to a randomly chosen reward worker.

        Args:
            idx: Index of this sample in the batch (for non_tensor_batch slicing).
            tensors: Dict of padded tensors from build_padded_sample.

        Returns:
            Ray ObjectRef for the reward computation result.
        """
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
        # veRL's reward path iterates multi_modal_inputs; add empty dict for text-only
        # samples so it doesn't raise KeyError
        if "multi_modal_inputs" not in item_non_tensor:
            item_non_tensor["multi_modal_inputs"] = np.array([{}], dtype=object)
        item_data = DataProto(batch=item_batch, non_tensor_batch=item_non_tensor)

        # Load-balance across reward workers using random selection (matches veRL's
        # agent_loop._compute_score pattern). Returns ObjectRef immediately; caller
        # collects results later via ray.get().
        handle = random.choice(self.reward_loop_worker_handles)
        return handle.compute_score.remote(item_data)

    def generate_and_collect_results(self) -> tuple[list, list | None, tuple | None]:
        """Generate all samples, building padded tensors and launching rewards as they arrive.

        Uses a ThreadPoolExecutor to fan out HTTP requests concurrently. As each
        response arrives, immediately builds padded tensors and launches reward
        computation (if reward workers are configured).

        Returns:
            sample_tensors: List of per-sample tensor dicts (all same shape).
            reward_refs: List of Ray ObjectRefs (or None if no reward workers).
            sample_debug: First (idx, resp, prompt_text) tuple for debug logging.
        """
        sample_tensors: list[dict | None] = [None] * self.batch_size
        reward_refs: list | None = [None] * self.batch_size if self.reward_loop_worker_handles else None
        sample_debug: tuple | None = None

        max_workers = min(self.batch_size, _MAX_CONCURRENT)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self.generate_one, i, self.prompt_token_ids[i], self.prompt_texts[i]
                ): i
                for i in range(self.batch_size)
            }
            for future in as_completed(futures):
                idx, resp = future.result()
                if sample_debug is None:
                    sample_debug = (idx, resp, self.prompt_texts[idx])

                # Parse response and build padded tensors ONCE
                response_ids, response_lps = self.parse_response(resp)
                tensors = self.build_padded_sample(self.prompt_token_ids[idx], response_ids, response_lps)
                sample_tensors[idx] = tensors

                # Launch reward immediately using the same tensors
                if reward_refs is not None:
                    reward_refs[idx] = self.launch_reward(idx, tensors)

        return sample_tensors, reward_refs, sample_debug

    async def finalize_rewards(
        self,
        reward_refs: list | None,
        full_mask: torch.Tensor,
        t_end: float,
    ) -> tuple[torch.Tensor, list[str], dict[str, np.ndarray], float | None]:
        """Collect reward results and build rm_scores tensor.

        Awaits all reward Ray ObjectRefs, extracts scores, and places each
        scalar score at the last valid response token position.

        Args:
            reward_refs: List of Ray ObjectRefs (or None if no reward workers).
            full_mask: Attention mask tensor [B, max_prompt_len + max_response_len].
            t_end: Timestamp when generation ended (for straggler timing).

        Returns:
            rm_scores: [B, max_response_len] tensor with scores at last valid token.
            reward_extra_keys: List of extra info keys from reward results.
            reward_extra_info: Dict mapping keys to np arrays of extra info.
            straggler_wait: Time spent waiting for stragglers (or None if no rewards).
        """
        # No reward workers configured — return zeros
        if reward_refs is None:
            return (
                torch.zeros(self.batch_size, self.max_response_len, dtype=torch.float32),
                [],
                {},
                None,
            )

        # Collect all reward results (blocks until all workers finish)
        # ray.get is blocking, so run in worker thread to not block event loop
        reward_results = await asyncio.to_thread(ray.get, reward_refs)
        scores = [r["reward_score"] for r in reward_results]

        # Place scalar score at last valid response token
        # PPO/GRPO assigns credit to the final generated token, so we put
        # the reward there. Example: if response has 3 real tokens,
        # rm_scores = [0, 0, 5.2, 0, 0, 0] (score at position 2)
        valid_resp_lens = full_mask[:, self.max_prompt_len:].sum(dim=1)
        rm_scores = torch.zeros(self.batch_size, self.max_response_len, dtype=torch.float32)
        for i, (score, vlen) in enumerate(zip(scores, valid_resp_lens)):
            if int(vlen.item()) > 0:
                rm_scores[i, int(vlen.item()) - 1] = float(score)

        # Extract any extra info from reward results
        reward_extra_keys = list(reward_results[0].get("reward_extra_info", {}).keys())
        reward_extra_info = {
            key: np.array([r["reward_extra_info"][key] for r in reward_results])
            for key in reward_extra_keys
        }

        # Measure how long we waited for the slowest reward worker
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

    def __init__(self, config, client: RolloutControllerClient, tokenizer=None,
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
        client = RolloutControllerClient(llmd_config)

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
        # Send all prompts concurrently to llm-d controller
        #
        # Like veRL, we pad each sample to fixed lengths (prompt_length,
        # response_length) as it arrives, then launch reward immediately.
        # ------------------------------------------------------------------
        t_gen_start = time.perf_counter()

        # Offload blocking threadpool wait from asyncio event loop
        sample_tensors, reward_refs, sample_debug = await asyncio.to_thread(
            gen.generate_and_collect_results
        )
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