def pytest_addoption(parser):
	parser.addoption(
		"--bench-rounds",
		action="store",
		default="20",
		help="Number of rounds per benchmark (default: 20)",
	)
	parser.addoption(
		"--dtype",
		action="store",
		default="float32",
		choices=["float32", "float64"],
		help="Floating point precision for all benchmarks (default: float32)",
	)
	parser.addoption(
		"--no-jit",
		action="store_true",
		default=False,
		help="Disable JAX JIT compilation",
	)
