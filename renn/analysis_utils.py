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
"""Utilities for analysis."""

from collections import defaultdict
from itertools import product
import numpy as np

__all__ = ['pseudogrid']


def pseudogrid(coordinates, dimension):
  """Constructs a pseudogrid
    ('pseudo' in that it is not necessarily evenly-spaced)
    of points in 'dimension'-dimension space
    from the specified coordinates.

    Arguments:
    coordinates: a mapping between dimensions and
                  coordinates in those dimensions
    dimension: number of dimensions

    For all dimensions that are not specified, the coordinate
    is taken to be 0.

    Example:
      if coordinates = {0: [0, 1, 2],
                        2: [1]},
      and dimension = 4, the coordinates in dimensions
      1 and 3 will be taken as [0], yielding the effective
      coordinate-dictionary
        coordinates = {0: [0,1,2],
                       1: [0],
                       2: [1],
                       3: [0]}
      Then the resulting pseudogrid will be constructed as:
        [[0,0,1,0], [1,0,1,0], [2,0,1,0]]
  """

  all_coordinates = defaultdict(lambda: np.array(0.0))
  all_coordinates.update(coordinates)

  max_specified_dim = max(coordinates.keys())

  if max_specified_dim > 32:
    raise NotImplementedError('Maximum specified dimension cannot exceed 32')

  points = np.meshgrid(
      *[all_coordinates[i] for i in range(max_specified_dim + 1)])
  points = np.stack(points).reshape(max_specified_dim + 1, -1).T

  padding = np.zeros((points.shape[0], dimension - max_specified_dim - 1))

  return np.concatenate((points, padding), axis=1)
