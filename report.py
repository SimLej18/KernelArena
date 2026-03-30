"""
Aggregate benchmark JSON files and print a cross-library comparison table.

Usage:
  python report.py [results_dir] [--script test_se.py] [--test test_1d_regular]

  results_dir  Directory containing benchmark JSON files (default: out)
  --script     Filter to a specific kernel script, e.g. test_se.py
  --test       Filter to a specific scenario, e.g. test_1d_regular

Each table is printed to stdout and saved to
  <results_dir>/<kernel>/<scenario>/report.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pandas as pd
    from tabulate import tabulate
except ImportError:
    print("Missing dependencies: pip install pandas tabulate")
    sys.exit(1)


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    json_files = sorted(
        p for p in results_dir.glob("*.json")
        if p.stem != "run_info" and p.stat().st_size > 0
    )

    if not json_files:
        print(f"No JSON files found in {results_dir}. Run `make all` or individual targets first.")
        sys.exit(1)

    for path in json_files:
        lib = path.stem

        with open(path) as f:
            data = json.load(f)

        for bench in data.get("benchmarks", []):
            fullname = bench.get("fullname", "")
            stats = bench.get("stats", {})

            kernel_match = re.search(r"/test_(se|matern)\.", fullname)
            kernel = kernel_match.group(1) if kernel_match else "unknown"

            scenario_match = re.search(r"::(test_\w+)$", fullname)
            scenario = scenario_match.group(1).removeprefix("test_") if scenario_match else "unknown"

            rows.append({
                "lib":       lib,
                "kernel":    kernel,
                "scenario":  scenario,
                "mean_ms":   round(stats.get("mean",   0) * 1000, 3),
                "stddev_ms": round(stats.get("stddev", 0) * 1000, 3),
                "min_ms":    round(stats.get("min",    0) * 1000, 3),
                "rounds":    stats.get("rounds", 0),
            })

    return rows


def build_report(rows: list[dict], results_dir: Path,
                 kernel_filter: str | None, scenario_filter: str | None) -> None:
    df = pd.DataFrame(rows)

    for (kernel, scenario), group in df.groupby(["kernel", "scenario"]):
        if kernel_filter and kernel != kernel_filter:
            continue
        if scenario_filter and scenario != scenario_filter:
            continue

        header = (
            f"\n{'=' * 60}\n"
            f"  Kernel: {kernel.upper()}   |   Scenario: {scenario}\n"
            f"{'=' * 60}"
        )
        display = group[["lib", "mean_ms", "stddev_ms", "min_ms", "rounds"]].copy()
        display = display.sort_values("mean_ms")
        table = tabulate(
            display.values,
            headers=["lib", "mean (ms)", "stddev (ms)", "min (ms)", "rounds"],
            tablefmt="rounded_outline",
            floatfmt=".3f",
        )

        print(header)
        print(table)

        out_dir = results_dir / kernel / scenario
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "report.txt"
        out_file.write_text(header.lstrip("\n") + "\n" + table + "\n")
        print(f"Saved: {out_file}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", nargs="?", default="out",
                   help="Directory containing benchmark JSON files (default: out)")
    p.add_argument("--script", default=None,
                   help="Kernel script to filter on, e.g. test_se.py or test_matern.py")
    p.add_argument("--test", default=None,
                   help="Scenario to filter on, e.g. test_1d_regular")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    # Derive filter values from --script / --test
    kernel_filter = None
    if args.script:
        m = re.match(r"test_(\w+)\.py$", args.script)
        kernel_filter = m.group(1) if m else args.script

    scenario_filter = None
    if args.test:
        scenario_filter = args.test.removeprefix("test_")

    rows = load_results(results_dir)
    build_report(rows, results_dir, kernel_filter, scenario_filter)
    print(f"\nLoaded {len(rows)} results from {results_dir}")


if __name__ == "__main__":
    main()