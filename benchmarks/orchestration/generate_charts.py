#!/usr/bin/env python3
"""Generate benchmark visualization charts."""

import json
import os
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"
CHARTS_DIR = Path(__file__).parent / "charts"


def load_all():
    """Load all result files and compute medians."""
    data = {}
    for name in ["llmd_1engines", "llmd_2engines", "llmd_4engines",
                  "ray_1engines", "ray_2engines", "ray_4engines"]:
        fpath = RESULTS_DIR / f"{name}.json"
        with open(fpath) as f:
            raw = json.load(f)
        steps = raw["steps"]
        medians = {}
        for key in ["step_total", "nccl_broadcast", "generate", "sleep",
                     "train", "wake_weights", "update_weights_http", "wake_kvcache"]:
            vals = [s[key] for s in steps if key in s]
            medians[key] = median(vals) if vals else 0.0
        # Compute orchestration overhead per step, then take median
        overheads = []
        for s in steps:
            orch = s["step_total"] - s["generate"] - s["train"] - s["nccl_broadcast"]
            overheads.append(orch)
        medians["orchestration_overhead"] = median(overheads)
        # All step values for distribution plots
        medians["_steps"] = steps
        medians["_system"] = raw["system"]
        medians["_num_engines"] = raw["num_engines"]
        medians["_nccl_init"] = raw.get("nccl_init_time_s", 0)
        data[name] = medians
    return data


def chart1_step_time_scaling(data):
    """Bar chart: median step time at 1/2/4 engines, llm-d-rl vs Ray."""
    engines = [1, 2, 4]
    llmd_times = [data[f"llmd_{n}engines"]["step_total"] for n in engines]
    ray_times = [data[f"ray_{n}engines"]["step_total"] for n in engines]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(engines))
    w = 0.35
    bars1 = ax.bar(x - w/2, llmd_times, w, label="llm-d-rl", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + w/2, ray_times, w, label="Ray", color="#FF9800", edgecolor="white")

    ax.set_xlabel("Number of vLLM Engines", fontsize=12)
    ax.set_ylabel("Median Step Time (seconds)", fontsize=12)
    ax.set_title("RL Training Step Time: llm-d-rl vs Ray", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in engines])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(llmd_times), max(ray_times)) * 1.15)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.15,
                    f"{h:.1f}s", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "step_time_scaling.png", dpi=150)
    plt.close(fig)
    print("  step_time_scaling.png")


def chart2_step_breakdown(data):
    """Stacked bar chart: phase breakdown at each scale."""
    configs = [
        ("llm-d-rl\n1 eng", "llmd_1engines"),
        ("Ray\n1 eng", "ray_1engines"),
        ("llm-d-rl\n2 eng", "llmd_2engines"),
        ("Ray\n2 eng", "ray_2engines"),
        ("llm-d-rl\n4 eng", "llmd_4engines"),
        ("Ray\n4 eng", "ray_4engines"),
    ]

    phases = [
        ("generate", "Generate", "#4CAF50"),
        ("sleep", "Sleep", "#9C27B0"),
        ("train", "Train", "#FF5722"),
        ("nccl_broadcast", "NCCL Broadcast", "#2196F3"),
        ("orchestration_overhead", "Orchestration", "#FFC107"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(configs))

    bottoms = np.zeros(len(configs))
    for phase_key, phase_label, color in phases:
        vals = []
        for _, key in configs:
            d = data[key]
            if phase_key == "orchestration_overhead":
                # overhead = total - gen - train - nccl
                v = d["step_total"] - d["generate"] - d["train"] - d["nccl_broadcast"]
            else:
                v = d.get(phase_key, 0.0)
            vals.append(v)
        ax.bar(x, vals, bottom=bottoms, label=phase_label, color=color, edgecolor="white", width=0.7)
        bottoms += np.array(vals)

    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Step Time Breakdown by Phase", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in configs], fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "step_breakdown.png", dpi=150)
    plt.close(fig)
    print("  step_breakdown.png")


def chart3_orchestration_overhead(data):
    """Line chart: orchestration overhead across scales."""
    engines = [1, 2, 4]
    llmd_orch = [data[f"llmd_{n}engines"]["orchestration_overhead"] * 1000 for n in engines]
    ray_orch = [data[f"ray_{n}engines"]["orchestration_overhead"] * 1000 for n in engines]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(engines, llmd_orch, "o-", color="#2196F3", linewidth=2, markersize=8, label="llm-d-rl")
    ax.plot(engines, ray_orch, "s--", color="#FF9800", linewidth=2, markersize=8, label="Ray")

    ax.set_xlabel("Number of vLLM Engines", fontsize=12)
    ax.set_ylabel("Orchestration Overhead (ms)", fontsize=12)
    ax.set_title("Orchestration Overhead Stays Flat", fontsize=14, fontweight="bold")
    ax.set_xticks(engines)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim(400, 700)

    for i, (l, r) in enumerate(zip(llmd_orch, ray_orch)):
        ax.annotate(f"{l:.0f}ms", (engines[i], l), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, color="#2196F3")
        ax.annotate(f"{r:.0f}ms", (engines[i], r), textcoords="offset points",
                    xytext=(0, -18), ha="center", fontsize=9, color="#FF9800")

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "orchestration_overhead.png", dpi=150)
    plt.close(fig)
    print("  orchestration_overhead.png")


def chart4_nccl_dominance(data):
    """Bar chart showing NCCL % of step time at each scale."""
    engines = [1, 2, 4]
    nccl_pcts = []
    other_pcts = []
    totals = []
    for n in engines:
        d = data[f"llmd_{n}engines"]
        nccl = d["nccl_broadcast"]
        total = d["step_total"]
        nccl_pcts.append(nccl / total * 100)
        other_pcts.append((total - nccl) / total * 100)
        totals.append(total)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(engines))
    w = 0.5
    ax.bar(x, nccl_pcts, w, label="NCCL Broadcast", color="#2196F3", edgecolor="white")
    ax.bar(x, other_pcts, w, bottom=nccl_pcts, label="Other (gen + train + orch)", color="#E0E0E0", edgecolor="white")

    for i, (npct, total) in enumerate(zip(nccl_pcts, totals)):
        ax.text(x[i], npct / 2, f"{npct:.0f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(x[i], npct + (100 - npct) / 2, f"{100 - npct:.0f}%", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#666")
        ax.text(x[i], 102, f"{total:.1f}s total", ha="center", va="bottom", fontsize=9, color="#666")

    ax.set_xlabel("Number of vLLM Engines", fontsize=12)
    ax.set_ylabel("% of Step Time", fontsize=12)
    ax.set_title("NCCL Broadcast Dominates Step Time (llm-d-rl)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in engines])
    ax.set_ylim(0, 115)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "nccl_dominance.png", dpi=150)
    plt.close(fig)
    print("  nccl_dominance.png")


def chart5_step_distribution(data):
    """Box plot: step time distribution for all 6 runs."""
    configs = [
        ("llm-d-rl 1e", "llmd_1engines"),
        ("Ray 1e", "ray_1engines"),
        ("llm-d-rl 2e", "llmd_2engines"),
        ("Ray 2e", "ray_2engines"),
        ("llm-d-rl 4e", "llmd_4engines"),
        ("Ray 4e", "ray_4engines"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = []
    labels = []
    colors = []
    for label, key in configs:
        steps = data[key]["_steps"]
        box_data.append([s["step_total"] for s in steps])
        labels.append(label)
        colors.append("#2196F3" if "llm-d-rl" in label else "#FF9800")

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.6,
                    medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Step Time (seconds)", fontsize=12)
    ax.set_title("Step Time Distribution (20 Measured Steps)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "step_distribution.png", dpi=150)
    plt.close(fig)
    print("  step_distribution.png")


def chart6_nccl_throughput(data):
    """Bar chart: effective NCCL throughput at each scale."""
    model_size_gb = 16.1  # Llama-3.1-8B bf16
    engines = [1, 2, 4]

    llmd_tp = [model_size_gb / data[f"llmd_{n}engines"]["nccl_broadcast"] for n in engines]
    ray_tp = [model_size_gb / data[f"ray_{n}engines"]["nccl_broadcast"] for n in engines]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(engines))
    w = 0.35
    bars1 = ax.bar(x - w/2, llmd_tp, w, label="llm-d-rl", color="#2196F3", edgecolor="white")
    bars2 = ax.bar(x + w/2, ray_tp, w, label="Ray", color="#FF9800", edgecolor="white")

    ax.set_xlabel("Number of vLLM Engines", fontsize=12)
    ax.set_ylabel("Effective Throughput (GB/s)", fontsize=12)
    ax.set_title("NCCL Broadcast Throughput (16.1 GB model, TCP sockets)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in engines])
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(llmd_tp), max(ray_tp)) * 1.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=10)

    # Add IB target annotation inside plot area
    top = max(max(llmd_tp), max(ray_tp)) * 1.3
    ax.annotate("IB target ~25 GB/s  (4-12x headroom)",
                xy=(2.3, top * 0.92), fontsize=10, color="#4CAF50",
                fontweight="bold", ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="#E8F5E9", ec="#4CAF50", alpha=0.8))

    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "nccl_throughput.png", dpi=150)
    plt.close(fig)
    print("  nccl_throughput.png")


def main():
    os.makedirs(CHARTS_DIR, exist_ok=True)
    data = load_all()
    print("Generating charts:")
    chart1_step_time_scaling(data)
    chart2_step_breakdown(data)
    chart3_orchestration_overhead(data)
    chart4_nccl_dominance(data)
    chart5_step_distribution(data)
    chart6_nccl_throughput(data)
    print(f"\nAll charts saved to {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
