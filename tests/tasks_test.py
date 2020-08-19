# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests tasks."""

import jax
import numpy as np

import pytest

from renn.metaopt import tasks


class LinearRegressionTest:

  @pytest.mark.parametrize("dim,num_datapoints,seed,step", [
      (2, 10, 1, 3),
      (20, 100, 0, 0),
      (50, 50, 11, 5),
      (100, 50, 32, 13),
  ])
  def test_loss(self, dim, num_datapoints, seed, step):
    """Smoke test for the quadratic loss function."""

    # Build problem.
    problem_fun = tasks.linreg(dim, num_datapoints, 0.1)
    key = jax.random.PRNGKey(seed)
    params, loss_fun = problem_fun(key)

    loss = loss_fun(params, step)  # Compute loss.
    assert loss >= 0.0  # Loss must be non-negative.

  @pytest.mark.parametrize("dim,num_datapoints,num_seeds", [
      (2, 5, 10),
      (20, 50, 10),
  ])
  def test_batch(self, dim, num_datapoints, num_seeds):
    """Tests batching of multiple problems."""

    # Problem builder.
    problem_fun = tasks.linreg(dim, num_datapoints, 0.1)

    # Sample initial parameters.
    key = jax.random.PRNGKey(0)
    key, prng = jax.random.split(key)
    params, _ = problem_fun(prng)

    # Loss to batch over random seeds.
    def batch_loss(params, prng):
      _, loss_fun = problem_fun(prng)
      step = 0  # Loss doesn't depend on the step, so any value here works.
      return loss_fun(params, step)

    batch_loss = jax.vmap(batch_loss, in_axes=(None, 0))

    # Apply the batched outer loss function.
    prngs = jax.random.split(key, num_seeds)
    losses = batch_loss(params, prngs)

    # Ensure that all losses are unique.
    assert len(np.unique(losses)) == num_seeds


class LogisticRegressionTest:

  @pytest.mark.parametrize("dim,num_datapoints,seed,step", [
      (2, 10, 9, 14),
      (10, 50, 1, 3),
      (20, 50, 0, 0),
      (50, 100, 11, 5),
      (100, 50, 32, 13),
  ])
  def test_loss(self, dim, num_datapoints, seed, step):
    """Smoke test for the quadratic loss function."""

    # Build problem.
    l2_reg = 1e-3
    problem_fun = tasks.logreg(dim, num_datapoints, l2_reg)
    key = jax.random.PRNGKey(seed)
    params, loss_fun = problem_fun(key)

    loss = loss_fun(params, step)  # Compute loss.
    assert loss >= 0.0  # Loss must be non-negative.

  @pytest.mark.parametrize("dim,num_datapoints,num_seeds", [
      (2, 5, 10),
      (20, 50, 10),
  ])
  def test_batch(self, dim, num_datapoints, num_seeds):
    """Tests batching of multiple problems."""

    # Problem builder.
    l2_reg = 1e-5
    problem_fun = tasks.logreg(dim, num_datapoints, l2_reg)

    # Sample initial parameters.
    key = jax.random.PRNGKey(0)
    key, prng = jax.random.split(key)
    params, _ = problem_fun(prng)

    # Loss to batch over random seeds.
    def batch_loss(params, prng):
      _, loss_fun = problem_fun(prng)
      step = 0  # Loss doesn't depend on the step, so any value here works.
      return loss_fun(params, step)

    batch_loss = jax.vmap(batch_loss, in_axes=(None, 0))

    # Apply the batched outer loss function.
    prngs = jax.random.split(key, num_seeds)
    losses = batch_loss(params, prngs)

    # Ensure that all losses are unique.
    assert len(np.unique(losses)) == num_seeds
