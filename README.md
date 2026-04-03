# KernelArena

Benchmarks comparing Gaussian Process kernel implementations across 6 Python libraries — **Kernax**, **scikit-learn**, **GPyTorch**, **GPJax**, **GPFlow**, and **GPy** — on Squared Exponential and Matérn 5/2 kernels.

## Requirements & Quickstart

To run the benchmarks on your machine, you need python>=3.10 and the pip packages listed in requirements.txt. 
With conda, the installation looks like this:

```bash
conda create -n KArena python=3.12
conda activate KArena

git clone https://github.com/SimLej18/KernelArena.git
cd KernelArena

pip install -r requirements.txt
```

Then, you can run the benchmarks with the make commands. For example:

```
make all DTYPE=float64 GPU=1
```

If you want to run the benchmarks on cpu, remove the GPU option or set it to 0.

This generates input data, runs all 6 libraries, then saves reports and plots under `out/`.


> [NOTE]
> **For Windows users:** multiple tools (make, uv, mkdir, ...) do not work the same on Windows.
> The easiest way to run the benchmarks on your machine is to use `wsl`. 
> 
> For example: 
> `wsl make all`

## Example result

Matérn 5/2 — 1D random inputs — float32 — CPU (Apple M4 Max)

![Matérn 5/2, 1D random, CPU](assets/matern_1d_random_cpu.svg)

## Common configurations

```bash
# Default run — float32, 20 rounds, CPU
make all

# Higher precision
make all DTYPE=float64

# Single library, fewer rounds
make gpjax ROUNDS=5

# Single report or plot for one experiment
make report SCRIPT=test_matern.py TEST=test_1d_random
make plot   SCRIPT=test_matern.py TEST=test_1d_random FORMAT=png
```

## All options

```
make help
```

## Submitting your results

The `make submit` command (automatically ran with make all) creates a result folder with an id corresponding to your running configuration.

Once you have generated results, open a pull request adding your `results/<id>/` folder. The only required file is `run_info.json` (generated automatically) alongside the benchmark JSON files.



