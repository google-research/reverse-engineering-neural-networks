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
from jax.experimental import stax

import numpy as np

import pytest

from renn.metaopt import tasks


@pytest.mark.parametrize("dim,lambda_min,lambda_max,seed", [
    (2, -3, 1, 3),
    (20, -2, 0, 0),
    (50, -5, -3, 5),
    (100, 0, 1, 13),
])
def test_quadratic(dim, lambda_min, lambda_max, seed):
  """Smoke test for the quadratic loss function."""

  # Build problem.
  problem_fun = tasks.quad(dim, lambda_min, lambda_max)
  key = jax.random.PRNGKey(seed)
  params, loss_fun = problem_fun(key)

  loss = loss_fun(params, 0)  # Compute loss.
  assert loss >= 0.0  # Loss must be non-negative.


@pytest.mark.parametrize("dim,lambda_min,lambda_max,num_seeds", [
    (2, -3, 0, 10),
    (20, 1, 2, 10),
])
def test_batch_loss(dim, lambda_min, lambda_max, num_seeds):
  """Tests batching of multiple problems."""

  # Problem builder.
  problem_fun = tasks.quad(dim, lambda_min, lambda_max)

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


def test_logistic_regression_loss():
  """Smoke test for the logistic loss function."""

  # Random data.
  features = np.random.randn(1024, 2)
  targets = np.random.randint(2, size=1024)

  # Build problem.
  l2_pen = 1e-3
  model = stax.Dense(1)
  problem_fun = tasks.logistic_regression(model, features, targets, l2_pen)
  key = jax.random.PRNGKey(1234)
  params, loss_fun = problem_fun(key)

  loss = loss_fun(params, 0)  # Compute loss.
  assert loss >= 0.0  # Loss must be non-negative.


def test_softmax_regression_loss():
  """Smoke test for the softmax loss function."""

  # Random data.
  num_classes = 10
  features = np.random.randn(1024, 2)
  targets = np.random.randint(num_classes, size=1024)

  # Build problem.
  l2_pen = 1e-3
  model = stax.serial(stax.Dense(1), stax.LogSoftmax)
  problem_fun = tasks.softmax_regression(model, features, targets, l2_pen)
  key = jax.random.PRNGKey(1234)
  params, loss_fun = problem_fun(key)

  loss = loss_fun(params, 0)  # Compute loss.
  assert loss >= 0.0  # Loss must be non-negative.
