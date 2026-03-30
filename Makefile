.PHONY: all data kernax sklearn gpytorch gpjax gpflow gpy report reports plot plots submit lock clean help

ROUNDS       ?= 20
DTYPE        ?= float32
NO_JIT       ?= 0
OUT           = out
GPU          ?= 0
SCRIPT       ?=
TEST         ?=
FORMAT       ?= svg

BENCH_OPTS = --benchmark-only -v -s --benchmark-time-unit=ms \
             --bench-rounds=$(ROUNDS) \
             --dtype=$(DTYPE) \
             $(if $(filter 1,$(NO_JIT)),--no-jit,)

# When GPU=0, hide all CUDA devices so libraries that auto-detect (PyTorch, TF)
# don't accidentally use the GPU. When GPU=1, leave the variable unset so CUDA
# is visible. Not applied to JAX libs (kernax/gpjax): they only get a CUDA
# backend if jax[cuda13] is explicitly installed, so auto-detection isn't a risk.
CUDA_ENV = $(if $(filter 1,$(GPU)),,CUDA_VISIBLE_DEVICES="")

# For report/plots: use results/<ID> if ID is provided, otherwise use out/
RESULTS_PATH = $(if $(ID),results/$(ID),out)

FILTER_OPTS  = $(if $(SCRIPT),--script $(SCRIPT),) $(if $(TEST),--test $(TEST),)
PLOT_OPTS    = --dtype=$(DTYPE) $(if $(filter 1,$(NO_JIT)),--no-jit,) $(if $(filter 1,$(GPU)),--gpu,) --format=$(FORMAT)

help:
	@echo "Targets:"
	@echo "  make all       - Run all benchmarks and generate report"
	@echo "  make kernax    - Benchmark kernax only"
	@echo "  make sklearn   - Benchmark scikit-learn only"
	@echo "  make gpytorch  - Benchmark GPyTorch only"
	@echo "  make gpjax     - Benchmark GPJax only"
	@echo "  make gpflow    - Benchmark GPFlow only"
	@echo "  make gpy       - Benchmark GPy only"
	@echo "  make report    - Print/save report (SCRIPT/TEST to filter; default: all)"
	@echo "  make reports   - Print/save all reports"
	@echo "  make plot      - Generate boxplot (SCRIPT/TEST to filter; default: all)"
	@echo "  make plots     - Generate all boxplots"
	@echo "  make submit    - Package out/ into results/<cpu>_<dtype>_<date>/"
	@echo "  make lock      - Regenerate pinned lockfiles (requirements/*.lock)"
	@echo "  make data      - Pre-generate benchmark input data (required before first run)"
	@echo "  make clean     - Remove venvs, output files, and data"
	@echo ""
	@echo "Options (pass as make var=value):"
	@echo "  ROUNDS=N                Number of benchmark rounds (default: 20)"
	@echo "  DTYPE=float32|float64   Floating point precision (default: float32)"
	@echo "  NO_JIT=1                Disable JAX JIT compilation"
	@echo "  GPU=1                   Install JAX CUDA backend"
	@echo "  ID=<submission>         Read results from results/<submission>/ (report/plot)"
	@echo "  SCRIPT=test_se.py       Filter report/plot to a specific kernel script"
	@echo "  TEST=test_1d_regular    Filter report/plot to a specific scenario"
	@echo "  FORMAT=svg|png|pdf      Plot output format (default: svg)"
	@echo ""
	@echo "Requires: uv  ->  https://docs.astral.sh/uv/"

all: data kernax sklearn gpytorch gpjax gpflow gpy reports plots submit

$(OUT):
	mkdir -p $(OUT)

env:
	mkdir -p env

data: data/meta.json

data/meta.json:
	uv run --no-project --with "jax[cpu]>=0.4" --with numpy python generate_data.py --rounds $(ROUNDS)

kernax: data/meta.json | $(OUT) env
	[ -d env/.venv-kernax ] || uv venv env/.venv-kernax
	uv pip install -q --python env/.venv-kernax/bin/python -r requirements/kernax.txt
	[ "$(GPU)" != "1" ] || uv pip install -q --python env/.venv-kernax/bin/python "jax[cuda13]>=0.9"
	env/.venv-kernax/bin/pytest kernax/ $(BENCH_OPTS) --benchmark-json=$(OUT)/kernax.json

sklearn: data/meta.json | $(OUT) env
	[ -d env/.venv-sklearn ] || uv venv env/.venv-sklearn
	uv pip install -q --python env/.venv-sklearn/bin/python -r requirements/sklearn.txt
	env/.venv-sklearn/bin/pytest sklearn/ $(BENCH_OPTS) --benchmark-json=$(OUT)/sklearn.json

gpytorch: data/meta.json | $(OUT) env
	[ -d env/.venv-gpytorch ] || uv venv env/.venv-gpytorch
	uv pip install -q --python env/.venv-gpytorch/bin/python -r requirements/gpytorch.txt
	$(CUDA_ENV) env/.venv-gpytorch/bin/pytest gpytorch/ $(BENCH_OPTS) --benchmark-json=$(OUT)/gpytorch.json

gpjax: data/meta.json | $(OUT) env
	[ -d env/.venv-gpjax ] || uv venv env/.venv-gpjax
	uv pip install -q --python env/.venv-gpjax/bin/python -r requirements/gpjax.txt
	[ "$(GPU)" != "1" ] || uv pip install -q --python env/.venv-gpjax/bin/python "jax[cuda13]>=0.9"
	env/.venv-gpjax/bin/pytest gpjax/ $(BENCH_OPTS) --benchmark-json=$(OUT)/gpjax.json

# GPFlow requires --no-deps due to numpy version conflict between gpflow and tensorflow.
# pytest/pytest-benchmark are installed first (with their deps) to avoid --no-deps stripping them.
# JAX stays CPU-pinned inside gpflow.txt; do NOT use general.txt here (cuda12 vs pinned CPU conflict).
# GPU=1: install tensorflow[and-cuda] to get CUDA-bundled TF (TF 2.13+ packaging).
gpflow: data/meta.json | $(OUT) env
	[ -d env/.venv-gpflow ] || uv venv env/.venv-gpflow
	uv pip install -q --python env/.venv-gpflow/bin/python "pytest>=7.0" "pytest-benchmark>=5.2"
	uv pip install -q --python env/.venv-gpflow/bin/python --no-deps -r requirements/gpflow.txt
	[ "$(GPU)" != "1" ] || uv pip install -q --python env/.venv-gpflow/bin/python "tensorflow[and-cuda]==2.21.0"
	$(CUDA_ENV) env/.venv-gpflow/bin/pytest gpflow/ $(BENCH_OPTS) --benchmark-json=$(OUT)/gpflow.json

# GPy 1.13 requires numpy==1.26 + scipy==1.12, incompatible with JAX (numpy>=2).
# Uses Python 3.10 managed by uv — run `uv python install 3.10` if not already available.
# Loads pre-generated float64 data from data/ — no JAX dependency at runtime.
gpy: data/meta.json | $(OUT) env
	[ -d env/.venv-gpy ] || uv venv --python 3.10 env/.venv-gpy
	uv pip install -q --python env/.venv-gpy/bin/python -r requirements/gpy.txt
	env/.venv-gpy/bin/pytest gpy/ $(BENCH_OPTS) --benchmark-json=$(OUT)/gpy.json

report: | $(OUT) env
	[ -d env/.venv-report ] || uv venv env/.venv-report
	uv pip install -q --python env/.venv-report/bin/python -r requirements/report.txt
	env/.venv-report/bin/python report.py $(RESULTS_PATH) $(FILTER_OPTS)

reports: | $(OUT) env
	[ -d env/.venv-report ] || uv venv env/.venv-report
	uv pip install -q --python env/.venv-report/bin/python -r requirements/report.txt
	env/.venv-report/bin/python report.py $(RESULTS_PATH)

plot: | $(OUT) env
	[ -d env/.venv-report ] || uv venv env/.venv-report
	uv pip install -q --python env/.venv-report/bin/python -r requirements/report.txt
	env/.venv-report/bin/python plots.py $(RESULTS_PATH) $(FILTER_OPTS) $(PLOT_OPTS)

plots: | $(OUT) env
	[ -d env/.venv-report ] || uv venv env/.venv-report
	uv pip install -q --python env/.venv-report/bin/python -r requirements/report.txt
	env/.venv-report/bin/python plots.py $(RESULTS_PATH) $(PLOT_OPTS)

submit:
	python submit.py --dtype $(DTYPE) --gpu $(GPU) --rounds $(ROUNDS)

lock:
	uv pip compile requirements/kernax.txt   -o requirements/kernax.lock
	uv pip compile requirements/sklearn.txt  -o requirements/sklearn.lock
	uv pip compile requirements/gpytorch.txt -o requirements/gpytorch.lock
	uv pip compile requirements/gpjax.txt    -o requirements/gpjax.lock
	uv pip compile requirements/gpflow.txt   -o requirements/gpflow.lock
	uv pip compile requirements/gpy.txt      -o requirements/gpy.lock
	uv pip compile requirements/report.txt   -o requirements/report.lock

clean:
	rm -rf env/ data/
