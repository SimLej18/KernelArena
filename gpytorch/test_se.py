"""SE/RBF Kernel benchmarks — GPyTorch."""
import numpy as np
import torch
from pathlib import Path

from gpytorch.kernels import RBFKernel

DATA_DIR = Path(__file__).parent.parent / "data"

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


def test_1d_regular(benchmark, request):
	"""SE/RBF kernel on 1D regular grid (10000 points)."""
	rounds = int(request.config.getoption("--bench-rounds"))
	dtype = np.float32 if request.config.getoption("--dtype") == "float32" else np.float64
	x = torch.tensor(np.load(DATA_DIR / "1d_regular.npy").astype(dtype))
	lengthscales = np.load(DATA_DIR / "lengthscales.npy")
	ls_iter = iter(lengthscales)

	def setup():
		ls = float(next(ls_iter))
		kernel = RBFKernel()
		kernel.initialize(lengthscale=ls)
		return (kernel, x, x), {}

	def run(kernel, x1, x2):
		kernel(x1, x2).evaluate()

	benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)


def test_1d_random(benchmark, request):
	"""SE/RBF kernel on 1D random inputs (10000 points)."""
	rounds = int(request.config.getoption("--bench-rounds"))
	dtype = np.float32 if request.config.getoption("--dtype") == "float32" else np.float64
	xs = [
		torch.tensor(np.load(DATA_DIR / f"1d_random/round_{i:03d}.npy").astype(dtype))
		for i in range(rounds)
	]
	lengthscales = np.load(DATA_DIR / "lengthscales.npy")
	xs_iter = iter(xs)
	ls_iter = iter(lengthscales)

	def setup():
		x = next(xs_iter)
		ls = float(next(ls_iter))
		kernel = RBFKernel()
		kernel.initialize(lengthscale=ls)
		return (kernel, x, x), {}

	def run(kernel, x1, x2):
		kernel(x1, x2).evaluate()

	benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)


def test_2d_regular(benchmark, request):
	"""SE/RBF kernel on 2D regular grid (10000 points)."""
	rounds = int(request.config.getoption("--bench-rounds"))
	dtype = np.float32 if request.config.getoption("--dtype") == "float32" else np.float64
	x = torch.tensor(np.load(DATA_DIR / "2d_regular.npy").astype(dtype))
	lengthscales = np.load(DATA_DIR / "lengthscales.npy")
	ls_iter = iter(lengthscales)

	def setup():
		ls = float(next(ls_iter))
		kernel = RBFKernel()
		kernel.initialize(lengthscale=ls)
		return (kernel, x, x), {}

	def run(kernel, x1, x2):
		kernel(x1, x2).evaluate()

	benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)


def test_2d_random(benchmark, request):
	"""SE/RBF kernel on 2D random inputs (10000 points)."""
	rounds = int(request.config.getoption("--bench-rounds"))
	dtype = np.float32 if request.config.getoption("--dtype") == "float32" else np.float64
	xs = [
		torch.tensor(np.load(DATA_DIR / f"2d_random/round_{i:03d}.npy").astype(dtype))
		for i in range(rounds)
	]
	lengthscales = np.load(DATA_DIR / "lengthscales.npy")
	xs_iter = iter(xs)
	ls_iter = iter(lengthscales)

	def setup():
		x = next(xs_iter)
		ls = float(next(ls_iter))
		kernel = RBFKernel()
		kernel.initialize(lengthscale=ls)
		return (kernel, x, x), {}

	def run(kernel, x1, x2):
		kernel(x1, x2).evaluate()

	benchmark.pedantic(run, setup=setup, rounds=rounds, iterations=1)