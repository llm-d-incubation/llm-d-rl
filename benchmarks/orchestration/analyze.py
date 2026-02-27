#!/usr/bin/env python3
"""Analyze orchestration benchmark results and generate comparison tables.

Usage:
    python analyze.py --results-dir ./results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_results(results_dir: str) -> list[dict]:
    """Load all JSON result files from the results directory."""
    results = []
    for fname in sorted(Path(results_dir).glob("*.json")):
        with open(fname) as f:
            data = json.load(f)
            data["_filename"] = fname.name
            results.append(data)
    return results


def compute_stats(values: list[float]) -> dict:
    """Compute median, mean, p5, p95, p99 for a list of values."""
    if not values:
        return {"median": 0, "mean": 0, "p5": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
    s = sorted(values)
    n = len(s)
    return {
        "median": s[n // 2],
        "mean": sum(s) / n,
        "p5": s[max(0, int(n * 0.05))],
        "p95": s[min(n - 1, int(n * 0.95))],
        "p99": s[min(n - 1, int(n * 0.99))],
        "min": s[0],
        "max": s[-1],
    }


PHASES = [
    "generate",
    "sleep",
    "train",
    "wake_weights",
    "nccl_broadcast",
    "update_weights_http",
    "wake_kvcache",
    "step_total",
]


def analyze_run(result: dict) -> dict:
    """Analyze a single benchmark run."""
    steps = result.get("steps", [])
    if not steps:
        return {}

    analysis = {
        "system": result["system"],
        "model": result["model"],
        "num_engines": result["num_engines"],
        "num_params": result.get("num_params", 0),
        "measured_steps": len(steps),
        "model_load_time_s": result.get("model_load_time_s", 0),
        "nccl_init_time_s": result.get("nccl_init_time_s", 0),
        "phases": {},
    }

    for phase in PHASES:
        values = [s.get(phase, 0) for s in steps if phase in s]
        if values:
            analysis["phases"][phase] = compute_stats(values)

    # Compute orchestration overhead:
    # overhead = step_total - generate - train - nccl_broadcast
    overheads = []
    for s in steps:
        total = s.get("step_total", 0)
        gen = s.get("generate", 0)
        train = s.get("train", 0)
        nccl = s.get("nccl_broadcast", 0)
        overhead = total - gen - train - nccl
        overheads.append(overhead)
    analysis["phases"]["orchestration_overhead"] = compute_stats(overheads)

    return analysis


def format_ms(seconds: float) -> str:
    """Format seconds as milliseconds with 1 decimal."""
    return f"{seconds * 1000:.1f}"


def print_comparison_table(analyses: list[dict]):
    """Print a markdown comparison table."""
    # Group by num_engines
    by_engines: dict[int, dict[str, dict]] = {}
    for a in analyses:
        n = a["num_engines"]
        sys = a["system"]
        if n not in by_engines:
            by_engines[n] = {}
        by_engines[n][sys] = a

    for n_engines in sorted(by_engines.keys()):
        systems = by_engines[n_engines]
        print(f"\n## {n_engines} Engine(s)\n")

        # Build table
        header_systems = sorted(systems.keys())
        header = "| Phase | " + " | ".join(
            f"{s} (median ms)" for s in header_systems) + " | Delta |"
        sep = "|" + "|".join(["---"] * (len(header_systems) + 2)) + "|"

        print(header)
        print(sep)

        display_phases = [
            ("Generate", "generate"),
            ("Sleep", "sleep"),
            ("Train", "train"),
            ("Wake weights", "wake_weights"),
            ("NCCL broadcast", "nccl_broadcast"),
            ("Update weights HTTP", "update_weights_http"),
            ("Wake KV cache", "wake_kvcache"),
            ("**Step total**", "step_total"),
            ("**Orchestration overhead**", "orchestration_overhead"),
        ]

        for label, phase_key in display_phases:
            values = []
            for sys in header_systems:
                a = systems[sys]
                phase_stats = a.get("phases", {}).get(phase_key, {})
                values.append(phase_stats.get("median", 0))

            delta = ""
            if len(values) == 2 and values[0] > 0:
                diff_ms = (values[1] - values[0]) * 1000
                pct = ((values[1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                delta = f"{diff_ms:+.1f}ms ({pct:+.0f}%)"

            row = f"| {label} | " + " | ".join(
                format_ms(v) for v in values) + f" | {delta} |"
            print(row)

        # Resource cost
        print(f"\n### Resource comparison ({n_engines} engines)\n")
        print("| Resource | llm-d-rl | Ray |")
        print("|---|---|---|")
        print("| Controller CPU | 500m | - |")
        print("| Controller Memory | 256Mi | - |")
        print("| Ray Head CPU | - | ~2 cores |")
        print("| Ray Head Memory | - | ~2Gi |")
        print("| Trainer GPU | 1x H200 | 1x H200 |")
        print(f"| Engine GPUs | {n_engines}x H200 | {n_engines}x H200 |")
        print(f"| **Total GPUs** | **{n_engines + 1}** | **{n_engines + 1}** |")

    # Summary across scales
    print("\n## Summary: Orchestration Overhead Across Scales\n")
    print("| Engines | llm-d-rl overhead (ms) | Ray overhead (ms) | Delta |")
    print("|---|---|---|---|")

    for n_engines in sorted(by_engines.keys()):
        systems = by_engines[n_engines]
        llmd = systems.get("llm-d-rl", {}).get("phases", {}).get(
            "orchestration_overhead", {}).get("median", 0)
        ray_val = systems.get("ray", {}).get("phases", {}).get(
            "orchestration_overhead", {}).get("median", 0)
        delta = ""
        if llmd > 0:
            diff = (ray_val - llmd) * 1000
            pct = ((ray_val - llmd) / llmd) * 100 if llmd != 0 else 0
            delta = f"{diff:+.1f}ms ({pct:+.0f}%)"
        print(f"| {n_engines} | {format_ms(llmd)} | {format_ms(ray_val)} | {delta} |")


def print_detailed_stats(analyses: list[dict]):
    """Print detailed per-phase statistics."""
    for a in analyses:
        print(f"\n### {a['system']} — {a['num_engines']} engine(s)\n")
        print(f"- Model: {a['model']}")
        print(f"- Parameters: {a.get('num_params', 0):,}")
        print(f"- Model load: {a.get('model_load_time_s', 0):.1f}s")
        print(f"- NCCL init: {a.get('nccl_init_time_s', 0):.1f}s")
        print(f"- Measured steps: {a.get('measured_steps', 0)}")
        print()
        print("| Phase | Median | Mean | P5 | P95 | P99 | Min | Max |")
        print("|---|---|---|---|---|---|---|---|")

        for phase_key in PHASES + ["orchestration_overhead"]:
            stats = a.get("phases", {}).get(phase_key, {})
            if not stats:
                continue
            label = phase_key.replace("_", " ").title()
            print(f"| {label} | {format_ms(stats['median'])} | "
                  f"{format_ms(stats['mean'])} | "
                  f"{format_ms(stats['p5'])} | "
                  f"{format_ms(stats['p95'])} | "
                  f"{format_ms(stats['p99'])} | "
                  f"{format_ms(stats['min'])} | "
                  f"{format_ms(stats['max'])} |")


def main():
    parser = argparse.ArgumentParser(description="Analyze orchestration benchmark results")
    parser.add_argument("--results-dir", required=True, help="Directory with JSON result files")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-phase stats")
    parser.add_argument("--output", help="Write output to file instead of stdout")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No JSON files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    analyses = [analyze_run(r) for r in results if r]
    analyses = [a for a in analyses if a]

    if not analyses:
        print("No valid results to analyze", file=sys.stderr)
        sys.exit(1)

    if args.output:
        sys.stdout = open(args.output, "w")

    print("# Orchestration Overhead Benchmark: llm-d-rl vs Ray\n")
    print(f"Model: {analyses[0].get('model', 'unknown')}")
    print(f"Parameters: {analyses[0].get('num_params', 0):,}")
    print(f"GPU: NVIDIA H200 (143GB)")
    print()

    print_comparison_table(analyses)

    if args.detailed:
        print("\n---\n")
        print("# Detailed Statistics\n")
        print_detailed_stats(analyses)

    # Also dump raw analysis as JSON
    json_output = os.path.join(args.results_dir, "analysis.json")
    with open(json_output, "w") as f:
        json.dump(analyses, f, indent=2)
    print(f"\n\nRaw analysis saved to {json_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
