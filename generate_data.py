"""
Pre-generate all benchmark input data and save as float64 numpy arrays.

Stores:
  data/1d_regular.npy                  (10000, 1)
  data/2d_regular.npy                  (10000, 2)
  data/1d_random/round_NNN.npy        (10000, 1) × rounds  — used in run()
  data/2d_random/round_NNN.npy        (10000, 2) × rounds  — used in run()
  data/1d_random_setup/round_NNN.npy  (10000, 1) × rounds  — used in setup() warm-up
  data/2d_random_setup/round_NNN.npy  (10000, 2) × rounds  — used in setup() warm-up
  data/lengthscales.npy               (rounds,)  — log-uniform in [0.1, 10]
  data/meta.json                       {"rounds": N, "seed": 42}

Usage:
  uv run --with "jax>=0.9" --with numpy generate_data.py --rounds 20
"""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

# x64 required so jnp arrays are float64 (JAX defaults to float32)
jax.config.update("jax_enable_x64", True)

import jax.random as jr
from input_generators import (
    generate_1d_regular_grid,
    generate_2d_regular_grid,
    generate_random_inputs,
)

DATA_DIR = Path("data")


def main(rounds: int) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "1d_random").mkdir(exist_ok=True)
    (DATA_DIR / "2d_random").mkdir(exist_ok=True)
    (DATA_DIR / "1d_random_setup").mkdir(exist_ok=True)
    (DATA_DIR / "2d_random_setup").mkdir(exist_ok=True)

    key = jr.PRNGKey(42)

    # Regular grids — deterministic, one file each
    np.save(DATA_DIR / "1d_regular.npy", np.asarray(generate_1d_regular_grid()))
    np.save(DATA_DIR / "2d_regular.npy", np.asarray(generate_2d_regular_grid()))
    print("  [OK] regular grids")

    # Random inputs for run() — one file per round
    for i in range(rounds):
        key, sk1, sk2 = jr.split(key, 3)
        x1d = generate_random_inputs(sk1, n_points=10000, n_dims=1, min_val=-500, max_val=500)
        np.save(DATA_DIR / f"1d_random/round_{i:03d}.npy", np.asarray(x1d))
        x2d = generate_random_inputs(sk2, n_points=10000, n_dims=2, min_val=-20, max_val=20)
        np.save(DATA_DIR / f"2d_random/round_{i:03d}.npy", np.asarray(x2d))

    print(f"  [OK] {rounds} rounds of random inputs for run() (1D + 2D)")

    # Separate random inputs for setup() warm-up — different sequence from run()
    for i in range(rounds):
        key, sk1, sk2 = jr.split(key, 3)
        x1d_setup = generate_random_inputs(sk1, n_points=10000, n_dims=1, min_val=-500, max_val=500)
        np.save(DATA_DIR / f"1d_random_setup/round_{i:03d}.npy", np.asarray(x1d_setup))
        x2d_setup = generate_random_inputs(sk2, n_points=10000, n_dims=2, min_val=-20, max_val=20)
        np.save(DATA_DIR / f"2d_random_setup/round_{i:03d}.npy", np.asarray(x2d_setup))

    print(f"  [OK] {rounds} rounds of random inputs for setup() warm-up (1D + 2D)")

    # Lengthscales — log-uniform in [0.1, 10], one per round
    key, sk = jr.split(key)
    log_ls = jr.uniform(sk, (rounds,), minval=np.log(0.1), maxval=np.log(10.0))
    lengthscales = np.exp(np.asarray(log_ls))
    np.save(DATA_DIR / "lengthscales.npy", lengthscales)
    print(f"  [OK] {rounds} lengthscales (log-uniform in [0.1, 10])")

    meta = {"rounds": rounds, "seed": 42}
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Benchmark data ready in {DATA_DIR}/  (meta: {meta})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=20,
                        help="Number of random-input rounds to generate (default: 20)")
    args = parser.parse_args()
    main(args.rounds)
