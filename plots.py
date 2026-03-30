"""
Generate a boxplot figure for a single kernel/scenario combination.

Usage:
  python plots.py [results_dir] [--script test_se.py] [--test test_1d_regular]
                  [--dtype float32|float64] [--no-jit] [--gpu] [--format svg]

  results_dir  Directory containing benchmark JSON files (default: out)
  --script     Kernel script to plot, e.g. test_se.py  (default: all)
  --test       Scenario to plot, e.g. test_1d_regular  (default: all)
  --format     Output image format: svg, png, pdf, ...  (default: svg)

Output is saved to <results_dir>/<kernel>/<scenario>/plot.<format>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


SCENARIOS = ["1d_regular", "1d_random", "2d_regular", "2d_random"]
KERNELS   = ["se", "matern"]

SCENARIO_LABELS = {
    "1d_regular": "1D Regular Grid",
    "1d_random":  "1D Random",
    "2d_regular": "2D Regular Grid",
    "2d_random":  "2D Random",
}
KERNEL_LABELS = {
    "se":     "Squared Exponential",
    "matern": "Matérn 5/2",
}

LIB_COLOURS = {
    "kernax":   "#4C72B0",
    "gpjax":    "#55A868",
    "gpflow":   "#C44E52",
    "gpytorch": "#8172B2",
    "sklearn":  "#CCB974",
    "gpy":      "#64B5CD",
}
DEFAULT_COLOUR = "#999999"


# ── data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict:
    """
    Returns nested dict:
        data[lib][kernel][scenario] = list[float]  (times in ms)
    """
    json_files = sorted(
        p for p in results_dir.glob("*.json")
        if p.stem != "run_info" and p.stat().st_size > 0
    )
    if not json_files:
        print(f"No JSON files found in {results_dir}. Run `make all` or individual targets first.")
        sys.exit(1)

    data: dict = {}
    for path in json_files:
        lib = path.stem
        with open(path) as f:
            raw = json.load(f)

        data[lib] = {}
        for bench in raw.get("benchmarks", []):
            fullname = bench.get("fullname", "")
            stats    = bench.get("stats", {})

            kernel_m = re.search(r"/test_(se|matern)\.", fullname)
            if not kernel_m:
                continue
            kernel = kernel_m.group(1)

            scenario_m = re.search(r"::(test_\w+)$", fullname)
            if not scenario_m:
                continue
            scenario = scenario_m.group(1).removeprefix("test_")

            times_ms = [t * 1000 for t in stats.get("data", [])]
            if not times_ms:
                times_ms = [stats.get("mean", 0) * 1000]

            data[lib].setdefault(kernel, {})[scenario] = times_ms

    return data


# ── library ordering ──────────────────────────────────────────────────────────

def global_lib_order(data: dict) -> list[str]:
    """Sort libraries by mean time across all benchmarks, slowest first."""
    means: dict[str, float] = {}
    for lib, kernels in data.items():
        all_times: list[float] = []
        for kernel_data in kernels.values():
            for times in kernel_data.values():
                all_times.extend(times)
        if all_times:
            means[lib] = sum(all_times) / len(all_times)
    return sorted(means, key=lambda lib: means[lib], reverse=True)


# ── plotting ──────────────────────────────────────────────────────────────────

def make_figure(data: dict, lib_order: list[str], kernel: str, scenario: str,
                dtype: str, no_jit: bool, gpu: bool, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    jit_str     = "JIT off" if no_jit else "JIT on"
    backend_str = "GPU" if gpu else "CPU"
    fig.suptitle(
        f"{KERNEL_LABELS.get(kernel, kernel)} | {SCENARIO_LABELS.get(scenario, scenario)}\n"
        f"{dtype} | {jit_str} | {backend_str}",
        fontsize=12, fontweight="bold",
    )

    box_data    = []
    positions   = []
    colours     = []
    tick_labels = []

    for pos, lib in enumerate(lib_order, start=1):
        times = data.get(lib, {}).get(kernel, {}).get(scenario)
        if times is None:
            continue
        box_data.append(times)
        positions.append(pos)
        colours.append(LIB_COLOURS.get(lib, DEFAULT_COLOUR))
        tick_labels.append(lib)

    if not box_data:
        print(f"No data for kernel={kernel}, scenario={scenario} — skipping.")
        plt.close(fig)
        return

    bp = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.5),
        showfliers=True,
    )

    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    for flier, colour in zip(bp["fliers"], colours):
        flier.set_markerfacecolor(colour)
        flier.set_markeredgecolor(colour)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Time (ms)", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xlim(0.3, len(lib_order) + 0.7)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default="out",
                   help="Directory containing benchmark JSON files (default: out)")
    p.add_argument("--script", default=None,
                   help="Kernel script to plot, e.g. test_se.py or test_matern.py")
    p.add_argument("--test",   default=None,
                   help="Scenario to plot, e.g. test_1d_regular")
    p.add_argument("--dtype",  default="float32", choices=["float32", "float64"])
    p.add_argument("--no-jit", action="store_true")
    p.add_argument("--gpu",    action="store_true")
    p.add_argument("--format", default="svg", dest="fmt",
                   help="Output image format: svg, png, pdf, ... (default: svg)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    data        = load_results(results_dir)
    lib_order   = global_lib_order(data)

    # Derive filter values from --script / --test
    kernel_filter = None
    if args.script:
        m = re.match(r"test_(\w+)\.py$", args.script)
        kernel_filter = m.group(1) if m else args.script

    scenario_filter = None
    if args.test:
        scenario_filter = args.test.removeprefix("test_")

    kernels   = [kernel_filter]   if kernel_filter   else KERNELS
    scenarios = [scenario_filter] if scenario_filter else SCENARIOS

    for kernel in kernels:
        for scenario in scenarios:
            out_path = results_dir / kernel / scenario / f"plot.{args.fmt}"
            make_figure(data, lib_order, kernel, scenario,
                        args.dtype, args.no_jit, args.gpu, out_path)

    print(f"Library order (slowest → fastest): {' > '.join(lib_order)}")


if __name__ == "__main__":
    main()
