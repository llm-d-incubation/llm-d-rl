# py-inference-scheduler integration for llm-d-rl.
#
# This package provides a standalone HTTP routing proxy that sits between the
# llm-d-rl rollout controller and a pool of vLLM engines. It uses the
# py-inference-scheduler library for load-aware, KV-cache-aware routing.
#
# The rollout controller points --router-url at this server. Generation
# requests are scheduled across vLLM pods; weight sync continues to go
# directly to each pod via the rollout controller's engine pool.
