"""SE/RBF Kernel benchmarks — GPy."""
import numpy as np
from pathlib import Path

import GPy

DATA_DIR = Path(__file__).parent.parent / "data"


def test_1d_regular(benchmark, request):
    """SE/RBF kernel on 1D regular grid (10000 points)."""
    rounds = int(request.config.getoption("--bench-rounds"))
    x = np.load(DATA_DIR / "1d_regular.npy")
    lengthscales = np.load(DATA_DIR / "lengthscales.npy")
    ls_iter = iter(lengthscales)

    def setup():
        ls = float(next(ls_iter))
        kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=ls)
        return (kernel, x, x), {}

    def run(kernel, x1, x2):
        kernel.K(x1, x2)

    benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)


def test_1d_random(benchmark, request):
    """SE/RBF kernel on 1D random inputs (10000 points)."""
    rounds = int(request.config.getoption("--bench-rounds"))
    xs = [
        np.load(DATA_DIR / f"1d_random/round_{i:03d}.npy")
        for i in range(rounds)
    ]
    lengthscales = np.load(DATA_DIR / "lengthscales.npy")
    xs_iter = iter(xs)
    ls_iter = iter(lengthscales)

    def setup():
        x = next(xs_iter)
        ls = float(next(ls_iter))
        kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=ls)
        return (kernel, x, x), {}

    def run(kernel, x1, x2):
        kernel.K(x1, x2)

    benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)


def test_2d_regular(benchmark, request):
    """SE/RBF kernel on 2D regular grid (10000 points)."""
    rounds = int(request.config.getoption("--bench-rounds"))
    x = np.load(DATA_DIR / "2d_regular.npy")
    lengthscales = np.load(DATA_DIR / "lengthscales.npy")
    ls_iter = iter(lengthscales)

    def setup():
        ls = float(next(ls_iter))
        kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=ls)
        return (kernel, x, x), {}

    def run(kernel, x1, x2):
        kernel.K(x1, x2)

    benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)


def test_2d_random(benchmark, request):
    """SE/RBF kernel on 2D random inputs (10000 points)."""
    rounds = int(request.config.getoption("--bench-rounds"))
    xs = [
        np.load(DATA_DIR / f"2d_random/round_{i:03d}.npy")
        for i in range(rounds)
    ]
    lengthscales = np.load(DATA_DIR / "lengthscales.npy")
    xs_iter = iter(xs)
    ls_iter = iter(lengthscales)

    def setup():
        x = next(xs_iter)
        ls = float(next(ls_iter))
        kernel = GPy.kern.RBF(input_dim=2, variance=1.0, lengthscale=ls)
        return (kernel, x, x), {}

    def run(kernel, x1, x2):
        kernel.K(x1, x2)

    benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)