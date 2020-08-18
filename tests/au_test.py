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

"""Tests for analysis utilities."""

import jax.numpy as jnp
import numpy as np

import pytest

from renn import analysis_utils as au

def test_pseudogrid():
  """ Tests for the pseudogrid function """

  # Test shape correctness
  dimensions = [10, 23, 40]
  pts = [[10,50,23], [24,12], [7,8,9,12]]

  for dimension in dimensions:
    for n_pts in pts:
      coordinates = {i: np.linspace(-1, 1, n_pts[i]) for i in range(len(n_pts))}
      pseudogrid_1 = au.pseudogrid(coordinates, dimension)

      assert pseudogrid_1.shape == (np.product(n_pts), dimension)

  # Test particular coordinates
  coordinates = {0: [0, 1, 2],
                 1: [1]}
  dimension = 4
  pseudogrid_2 = au.pseudogrid(coordinates, dimension)

  ideal = np.array([[0,1,0,0],
                               [1,1,0,0],
                               [2,1,0,0]])

  assert np.array_equal(ideal, pseudogrid_2)
