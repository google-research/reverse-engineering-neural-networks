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

"""Tests pytree serialization."""

import jax
import numpy as np

import pytest
import renn


# pytrees to use for testing.
TREES = (
    (None,),
    ((),),
    ((1, 2),),
    (((1, "foo"), ["bar", (3, None, 7)]),),
    ([3],),
    (np.array([1, 2, 3], dtype=np.int32),),
    (np.array([np.pi], dtype=np.float32),),
    ({"a": 1, "b": 2},),
    (renn.LinearRNN(np.ones(2), np.eye(2), 0.5),),
    (jax.device_get(renn.GRU(4).init(jax.random.PRNGKey(0), (-1, 1))[1]),),
)


def is_equal(x_tree, y_tree):
  """Checks that two pytrees are equal."""
  x_data, x_def = jax.tree_flatten(x_tree)
  y_data, y_def = jax.tree_flatten(y_tree)

  assert x_def == y_def

  for x, y in zip(x_data, y_data):

    if isinstance(x, np.ndarray):
      assert np.allclose(x, y)

    else:
      assert x == y

  return True


@pytest.mark.parametrize("pytree", TREES)
def test_serialization(pytree):
  """Test cyclical consistency when serializing a pytree."""
  assert is_equal(pytree, renn.loads(renn.dumps(pytree)))
