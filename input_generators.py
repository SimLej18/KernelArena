"""Shared input generation utilities for KernelBenchmarks."""
import jax
import jax.numpy as jnp
import jax.random as jr


def generate_1d_regular_grid(n_points=10000, min_val=-500, max_val=500):
	x = jnp.linspace(min_val, max_val, n_points)
	return x.reshape(-1, 1)


def generate_2d_regular_grid(n_points_per_dim=100, min_val=-20, max_val=20):
	x1 = jnp.linspace(min_val, max_val, n_points_per_dim)
	x2 = jnp.linspace(min_val, max_val, n_points_per_dim)
	x1_grid, x2_grid = jnp.meshgrid(x1, x2, indexing="ij")
	return jnp.stack([x1_grid.ravel(), x2_grid.ravel()], axis=1)


def generate_random_inputs(key, n_points, n_dims, min_val, max_val):
	x = jr.uniform(key, (n_points, n_dims), minval=min_val, maxval=max_val)
	x.block_until_ready()
	return x


def add_missing_values(key, x, missing_rate=0.25):
	n_points = x.shape[0]
	keep_mask = jr.bernoulli(key, 1 - missing_rate, (n_points,))
	x_with_nan = jnp.where(keep_mask[:, None], x, jnp.nan)
	x_with_nan.block_until_ready()
	return x_with_nan
