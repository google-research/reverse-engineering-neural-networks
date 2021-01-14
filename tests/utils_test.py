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
"""Tests utilities."""

import jax.numpy as jnp
import numpy as np

import pytest

from renn import utils


def test_fst():
  """Tests function that returns the first element in a list."""
  assert utils.fst([0, 1, 2, 3]) == 0


def test_snd():
  """Tests function that returns the second element in a list."""
  assert utils.snd([0, 1, 2, 3]) == 1


def test_compose():
  """Tests function composition."""
  data = np.arange(6).reshape(2, 3)
  assert utils.compose(utils.fst, utils.fst)(data) == 0
  assert utils.compose(utils.snd, utils.fst)(data) == 1
  assert utils.compose(utils.fst, utils.snd)(data) == 3
  assert utils.compose(utils.snd, utils.snd)(data) == 4
  assert utils.compose(str, np.abs, lambda x: x**3)(-2) == '8'


def test_empty():
  """Tests fst and snd functions on an empty array."""

  with pytest.raises(IndexError):
    utils.fst([])

  with pytest.raises(IndexError):
    utils.snd([])

  with pytest.raises(IndexError):
    utils.snd([
        0,
    ])


def test_array_norm():
  """Tests computing the l2 norm of a single array."""

  # Generate a single array to compute the norm of.
  n = 10
  l2_norm = utils.norm(jnp.ones(n))
  assert np.allclose(jnp.sqrt(n), l2_norm)


def test_dict_norm():
  """Tests the (vectorized) l2 norm of a pytree."""

  # Generate a random pytree (in this case, a dict).
  rs = np.random.RandomState(0)
  pytree = {
      'x': rs.randn(10,),
      'y': rs.randn(3, 5),
  }

  # Test the norm of the vectorized pytree.
  vec = np.hstack((pytree['x'], pytree['y'].ravel()))
  assert np.allclose(np.linalg.norm(vec), float(utils.norm(pytree)), atol=1e-3)


def test_one_hot_array():
  """Tests the one hot conversion with a jax array as input."""

  # Generate labels to test.
  n = 5
  labels = jnp.arange(n)

  # Convert to a one-hot representation.
  one_hot_labels = utils.one_hot(labels, n)
  assert np.allclose(one_hot_labels, jnp.eye(n))


def test_one_hot_list():
  """Tests the one hot conversion with a list as input."""

  # Generate labels to test.
  n = 5
  labels = list(range(n))

  # Convert to a one-hot representation.
  one_hot_labels = utils.one_hot(labels, n)
  assert np.allclose(one_hot_labels, jnp.eye(n))


def test_one_hot_empty():
  """Tests the one hot function with empty input."""
  assert len(utils.one_hot([], 5)) == 0


def test_select():
  """Tests the select function."""
  sequences = np.stack([
      [[0, 1], [2, 3], [4, 5]],
      [[6, 7], [8, 9], [10, 11]],
  ])

  indices = np.array([0, 1])
  assert np.allclose(utils.select(sequences, indices),
                     jnp.array([[0, 1], [8, 9]]))

  indices = np.array([2, 0])
  assert np.allclose(utils.select(sequences, indices),
                     jnp.array([[4, 5], [6, 7]]))


def test_batch_mean():
  """Tests the batch_mean function.

  This function converts a function that operates on a single input to
  one that operates over an array (batch) of inputs, and returns the mean
  over the batch.
  """

  # Generates a batched version of a square function.
  square_fun = utils.batch_mean(lambda x: x**2, (0,))

  # Map the function over a batch of data.
  data = jnp.array([0, 1, 2])
  result = square_fun(data)
  assert result == jnp.mean(data**2)

def test_build_mask():
  """Tests the build_mask function."""

  max_length = 5
  test_function = utils.build_mask(max_length)
  result = test_function(jnp.array([0,1,2,3,4,5,6,7,8,9,10]))
  ideal_result = jnp.array([[0., 0., 0., 0., 0.],
                            [1., 0., 0., 0., 0.],
                            [1., 1., 0., 0., 0.],
                            [1., 1., 1., 0., 0.],
                            [1., 1., 1., 1., 0.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.],
                            [1., 1., 1., 1., 1.]])
  assert np.allclose(result, ideal_result)

