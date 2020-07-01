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

"""Tests optimizer models."""

import jax
import jax.numpy as jnp

import numpy as np

import pytest

from renn.metaopt import models
from renn.rnn import cells


def test_append_to_sequence():

  # Random sequence elements to use for testing.
  x = np.random.randn(2, 10)
  y = np.random.randn(2, 10)
  z = np.random.randn(2, 10)

  # append_to_sequence should drop the first element and append the last.
  sequence = models.append_to_sequence([x, y], z)
  assert np.allclose(sequence, np.stack([y, z]))


def test_cwrnn_optimizer():
  """Smoke test for the component-wise RNN optimizer."""

  key = jax.random.PRNGKey(0)
  cell = cells.GRU(32)

  theta, optimizer_fun = models.cwrnn(key, cell, input_scale="raw")
  init_opt, update_opt, get_params = optimizer_fun(theta)

  # Dummy parameters and gradient.
  params = jnp.array([1.,])
  grads = jnp.array([1.,])

  # Smoke test for the optimizer functions.
  opt_state = init_opt(jnp.array([1.]))
  new_state = update_opt(0, grads, opt_state)

  # Test the get params function.
  assert np.allclose(params, get_params(opt_state))

  # Test the shape of new_state.
  assert params.shape == get_params(new_state).shape
