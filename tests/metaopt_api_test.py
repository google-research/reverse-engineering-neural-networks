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
"""Tests meta-optimization."""

import functools

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

import numpy as np
import pytest

from renn import metaopt
from renn import utils

import toolz


@pytest.mark.parametrize("x,value,expected", [
    ([1., 2., 3.], 2.5, [1., 2., 2.5]),
    ([-5., 0., 5.], 1.0, [-1., 0., 1.]),
])
def test_clip(x, value, expected):
  """Tests the clipping function."""

  # Checks default behavior (no clipping).
  assert np.allclose(x, metaopt.clip(x))

  # Checks clipping at the given value.
  x_clipped = metaopt.clip(jnp.array(x), value)
  assert np.allclose(x_clipped, expected)


@pytest.mark.parametrize("optimizer_fun,decorator,num_steps", [
    (metaopt.models.momentum, utils.identity, 3),
    (metaopt.models.momentum, jax.remat, 3),
])
def test_momentum(optimizer_fun, decorator, num_steps):
  """Numerical test for the meta-gradient of a parameterized optimizer."""

  base_key = jax.random.PRNGKey(0)
  keys = jax.random.split(base_key)

  problem_fun = metaopt.tasks.quad(10, -2, 0)
  theta_init, optimizer_fun = optimizer_fun(keys[0])

  # Build the meta-objective.
  metaobj = metaopt.build_metaobj(problem_fun,
                                  optimizer_fun,
                                  num_steps,
                                  decorator=decorator)

  # Bind the prng key.
  metaobj = functools.partial(metaobj, prng_key=keys[1])

  # Build function to extract just the first output (the meta-objective).
  scalar_metaobj = toolz.compose(utils.fst, metaobj)

  # Check the meta-gradient and meta-hessian numerically.
  check_grads(scalar_metaobj, (theta_init,), order=1)
