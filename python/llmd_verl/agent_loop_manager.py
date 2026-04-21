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
    batch["input_ids"]      [B, max_prompt_len]  padded prompt token IDs
    batch["attention_mask"] [B, max_prompt_len]  1=real token, 0=padding
    meta_info["validate"]   bool  (optional) — use greedy sampling
    meta_info["do_sample"]  bool  (optional, False → greedy)

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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import random

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


def _extract_llmd_config(verl_config) -> LlmdRolloutConfig:
    """Pull llm-d settings from veRL's rollout config."""
    rollout = verl_config.actor_rollout_ref.rollout
    kw = rollout.get("custom", {}).get("llmd", {})
    return LlmdRolloutConfig(**kw)


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
        # Empty — LlmdVerlCheckpointEngineManager ignores replicas anyway
        self.rollout_replicas: list = []

    # ------------------------------------------------------------------
    # veRL interface — construction
    # ------------------------------------------------------------------

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
        # veRL's RLHFDataset puts chat messages in non_tensor_batch["raw_prompt"]
        # rather than pre-tokenized input_ids.
        tokenizer = None
        try:
            from transformers import AutoTokenizer
            model_path = config.actor_rollout_ref.model.path
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            logger.debug("[LLMD] Loaded tokenizer from %s", model_path)
        except Exception as exc:
            logger.warning("[LLMD] Could not load tokenizer: %s", exc)

        return cls(config, client, tokenizer,
                   reward_loop_worker_handles=reward_loop_worker_handles)

    # ------------------------------------------------------------------
    # veRL interface — generation
    # ------------------------------------------------------------------

    @auto_await
    async def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Generate responses for a batch of prompts via llm-d HTTP API.

        Args:
            prompts: DataProto with batch["input_ids"] and batch["attention_mask"].

        Returns:
            DataProto with all fields required by veRL's fit() loop.
        """
        t_start = time.perf_counter()

        rollout_cfg = self.config.actor_rollout_ref.rollout
        is_validate = prompts.meta_info.get("validate", False)
        do_sample = prompts.meta_info.get("do_sample", True)

        # Sampling parameters — mirror veRL's AgentLoopWorker logic
        if is_validate or not do_sample:
            temperature = rollout_cfg.val_kwargs.get("temperature", 0.0) if is_validate else 0.0
            top_p = rollout_cfg.val_kwargs.get("top_p", 1.0) if is_validate else 1.0
            top_k = rollout_cfg.val_kwargs.get("top_k", -1) if is_validate else -1
        else:
            temperature = rollout_cfg.temperature
            top_p = rollout_cfg.top_p
            top_k = rollout_cfg.top_k

        # Fixed lengths from config (like veRL's agent_loop.py)
        max_response_len = rollout_cfg.response_length
        max_prompt_len = getattr(rollout_cfg, "prompt_length", None)

        # ------------------------------------------------------------------
        # 1. Format raw_prompt chat messages for generation requests
        #
        # veRL's RLHFDataset returns chat messages in non_tensor_batch["raw_prompt"]
        # (not pre-tokenized). We apply the chat template to produce formatted
        # prompt text sent to llm-d. We also tokenize once and reuse those token
        # IDs for both request payloads and padded prompt tensors.
        # ------------------------------------------------------------------
        assert self.tokenizer is not None, (
            "LlmdAgentLoopManager requires a tokenizer. "
            "Set actor_rollout_ref.model.path in your config."
        )
        pad_token_id = prompts.meta_info.get("pad_token_id", None)
        eos_token_id = prompts.meta_info.get("eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        raw_prompts = prompts.non_tensor_batch["raw_prompt"]
        B = len(raw_prompts)
        prompt_token_ids: list[list[int]] = []
        prompt_texts: list[str] = []
        prompt_lengths: list[int] = []
        for messages in raw_prompts:
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

        # Use config prompt_length if set, otherwise use actual max
        if max_prompt_len is None:
            max_prompt_len = max(prompt_lengths)

        # ------------------------------------------------------------------
        # 2. Send all prompts concurrently to llm-d controller
        #
        # Like veRL, we pad each sample to fixed lengths (prompt_length,
        # response_length) as it arrives, then launch reward immediately.
        # Final batch assembly is just torch.cat — no redundant tensor work.
        # ------------------------------------------------------------------
        t_gen_start = time.perf_counter()
        input_non_tensor_batch = prompts.non_tensor_batch

        def _generate_one(idx_ids_text):
            idx, ids, text = idx_ids_text
            kwargs = dict(
                prompt=text,
                prompt_token_ids=ids,
                max_tokens=max_response_len,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop=([self.tokenizer.decode([eos_token_id])] if eos_token_id is not None and self.tokenizer is not None else []),
            )
            try:
                resp = self.client.generate(**kwargs)
            except RuntimeError as exc:
                if "502" in str(exc) or "503" in str(exc):
                    logger.warning("[LLMD DEBUG] Generate request failed (%s), retrying once...", exc)
                    resp = self.client.generate(**kwargs)
                else:
                    raise
            return idx, resp

        def _parse_response(resp: dict) -> tuple[list[int], list[float]]:
            """Extract token IDs and logprobs from llm-d response."""
            ids = [int(t) for t in (resp.get("output_token_ids") or [])]
            if not ids:
                text = resp.get("text", "")
                ids = self.tokenizer.encode(text, add_special_tokens=False) if text else []
            raw_lps = resp.get("logprobs") or []
            if raw_lps:
                lps = [list(d.values())[0] if isinstance(d, dict) else float(d) for d in raw_lps]
            else:
                lps = [0.0] * len(ids)
            return ids, lps

        def _build_padded_sample(idx: int, prompt_ids: list[int], response_ids: list[int], response_lps: list[float]):
            """Build single-sample tensors padded to fixed config lengths.
            
            Like veRL's agent_loop.py, we pad prompts to prompt_length and
            responses to response_length so all samples have identical shapes.
            This allows torch.cat in final batch assembly with no extra work.
            
            Returns dict with all tensors needed for reward + final batch.
            """
            # Pad prompt to max_prompt_len
            prompt_t = torch.full((1, max_prompt_len), pad_token_id, dtype=torch.long)
            prompt_mask = torch.zeros(1, max_prompt_len, dtype=torch.long)
            p_len = min(len(prompt_ids), max_prompt_len)
            prompt_t[0, :p_len] = torch.tensor(prompt_ids[:p_len], dtype=torch.long)
            prompt_mask[0, :p_len] = 1

            # Pad response to max_response_len
            resp_t = torch.zeros(1, max_response_len, dtype=torch.long)
            resp_mask = torch.zeros(1, max_response_len, dtype=torch.long)
            resp_logprobs = torch.zeros(1, max_response_len, dtype=torch.float32)
            r_len = min(len(response_ids), max_response_len)
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
                "prompt_t": prompt_t,              # [1, max_prompt_len]
                "resp_t": resp_t,                  # [1, max_response_len]
                "resp_mask": resp_mask,            # [1, max_response_len]
                "resp_logprobs": resp_logprobs,    # [1, max_response_len]
                "full_ids": full_ids,              # [1, max_prompt_len + max_response_len]
                "full_mask": full_mask,            # [1, max_prompt_len + max_response_len]
                "position_ids": position_ids,      # [1, max_prompt_len + max_response_len]
            }

        def _launch_reward(idx: int, tensors: dict):
            """Launch reward computation using pre-built tensors."""
            item_batch = TensorDict(
                {
                    "prompts":        tensors["prompt_t"],
                    "responses":      tensors["resp_t"],
                    "input_ids":      tensors["full_ids"],
                    "attention_mask": tensors["full_mask"],
                    "position_ids":   tensors["position_ids"],
                    "response_mask":  tensors["resp_mask"],
                },
                batch_size=[1],
            )
            item_non_tensor = {k: np.array([v[idx]]) for k, v in input_non_tensor_batch.items()}
            item_non_tensor["__num_turns__"] = np.array([2], dtype=np.int32)
            item_non_tensor["tool_extra_fields"] = np.array([{}], dtype=object)
            if "multi_modal_inputs" not in item_non_tensor:
                item_non_tensor["multi_modal_inputs"] = np.array([{}], dtype=object)
            item_data = DataProto(batch=item_batch, non_tensor_batch=item_non_tensor)
            handle = random.choice(self.reward_loop_worker_handles)
            return handle.compute_score.remote(item_data)

        def _collect_generate_results():
            """Generate all samples, building padded tensors and launching rewards as they arrive.
            
            Returns:
                sample_tensors: list of per-sample tensor dicts (all same shape)
                reward_refs: list of Ray ObjectRefs (or None if no reward workers)
                sample_debug: first (idx, resp, prompt_text) for debug logging
            """
            sample_tensors = [None] * B
            reward_refs = [None] * B if self.reward_loop_worker_handles else None
            sample_debug = None

            max_workers = min(B, _MAX_CONCURRENT)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_generate_one, (i, prompt_token_ids[i], prompt_texts[i])): i
                    for i in range(B)
                }
                for future in as_completed(futures):
                    idx, resp = future.result()
                    if sample_debug is None:
                        sample_debug = (idx, resp, prompt_texts[idx])

                    # Parse response and build padded tensors ONCE
                    response_ids, response_lps = _parse_response(resp)
                    tensors = _build_padded_sample(idx, prompt_token_ids[idx], response_ids, response_lps)
                    sample_tensors[idx] = tensors

                    # Launch reward immediately using the same tensors
                    if reward_refs is not None:
                        reward_refs[idx] = _launch_reward(idx, tensors)

            return sample_tensors, reward_refs, sample_debug

        # Offload blocking threadpool wait from asyncio event loop
        sample_tensors, reward_refs, sample_debug = await asyncio.to_thread(_collect_generate_results)

        if sample_debug:
            idx, resp, _prompt_text = sample_debug
            _text = resp.get("text") or ""
            _prompt_head = _prompt_text[:200]
            _resp_tail = _text[-200:] if len(_text) > 200 else _text

            raw_lps = resp.get("logprobs") or []

            _wv = resp.get("weight_version", "(none)")
            _finish = resp.get("finish_reason") or resp.get("stop_reason") or "(none)"

            logger.debug("[LLMD] generate batch sample (first completed) idx=%d", idx)
            logger.debug("  prompt_head (200): %r", _prompt_head)
            logger.debug("  response_tail (200): %r", _resp_tail)
            logger.debug("  logprobs present: %s", bool(raw_lps))
            if raw_lps:
                _first3_lp: list[float] = []
                for d in raw_lps[:3]:
                    if isinstance(d, dict):
                        _first3_lp.append(float(list(d.values())[0]))
                    else:
                        _first3_lp.append(float(d))
                logger.debug("  first 3 logprobs: %s", _first3_lp)
            logger.debug("  weight_version: %s", _wv)
            logger.debug("  finish_reason: %s", _finish)

        t_gen_end = time.perf_counter()

        # ------------------------------------------------------------------
        # 3. Stack pre-padded tensors into batch (just torch.cat)
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
        # 4. Build output DataProto
        # ------------------------------------------------------------------
        out_batch = TensorDict(
            {
                "prompts":           prompt_ids,          # [B, max_prompt_len]
                "responses":         resp_ids,            # [B, max_response_len]
                "input_ids":         full_ids,            # [B, max_prompt_len + max_response_len]
                "attention_mask":    full_mask,           # [B, max_prompt_len + max_response_len]
                "position_ids":      position_ids,        # [B, max_prompt_len + max_response_len]
                "response_mask":     resp_mask,           # [B, max_response_len]
                "rollout_log_probs": rollout_log_probs,   # [B, max_response_len]
            },
            batch_size=[B],
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
            out_non_tensor["multi_modal_inputs"] = np.array([{} for _ in range(B)], dtype=object)

        # ------------------------------------------------------------------
        # 5. Finalize reward scores
        #
        # Reward RPCs were launched during generation. Now we just collect
        # the results (ray.get) and place scores in the batch.
        # ------------------------------------------------------------------
        if reward_refs is not None:
            # ray.get blocks; run in worker thread
            reward_results = await asyncio.to_thread(ray.get, reward_refs)
            scores = [r["reward_score"] for r in reward_results]

            # Place scalar score at last valid response token
            valid_resp_lens = full_mask[:, max_prompt_len:].sum(dim=1)
            rm_scores = torch.zeros(B, max_response_len, dtype=torch.float32)
            for i, (score, vlen) in enumerate(zip(scores, valid_resp_lens)):
                if int(vlen.item()) > 0:
                    rm_scores[i, int(vlen.item()) - 1] = float(score)
            out_batch["rm_scores"] = rm_scores

            reward_extra_keys = list(reward_results[0].get("reward_extra_info", {}).keys())
            for key in reward_extra_keys:
                out_non_tensor[key] = np.array([r["reward_extra_info"][key] for r in reward_results])
            # Note: reward RPCs are launched during generation (inside _collect_generate_results),
            # so most finish in parallel with it. This timer captures only the straggler wait —
            # the extra time we block after generation for the slowest reward to return.
            # It is NOT the true reward compute cost (which is hidden inside generate_sequences).
            timing["llmd/reward_straggler_wait"] = time.perf_counter() - t_end
        else:
            out_batch["rm_scores"] = torch.zeros(B, max_response_len, dtype=torch.float32)
            reward_extra_keys = []

        output = DataProto(
            batch=out_batch,
            non_tensor_batch=out_non_tensor,
            meta_info={**prompts.meta_info, "timing": timing,
                       "reward_extra_keys": reward_extra_keys},
        )

        logger.info(
            "generate_sequences: B=%d prompt_len=%d resp_len=%d t=%.2fs",
            B,
            max_prompt_len,
            max_response_len,
            t_end - t_start,
        )

        return output

    # ------------------------------------------------------------------
    # veRL interface — stubs for optional methods called in fit()
    # ------------------------------------------------------------------

    @auto_await
    async def clear_kv_cache(self) -> None:
        """No-op: llm-d manages cache reset readiness externally."""
        return

    def start_profile(self, **kwargs) -> None:
        """No-op: profiling not supported via HTTP API."""

    def stop_profile(self) -> None:
        """No-op: profiling not supported via HTTP API."""