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
"""Fixed point finding routines."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import distance

from .. import utils

__all__ = ['build_fixed_point_loss', 'find_fixed_points', 'exclude_outliers']


def build_fixed_point_loss(rnn_cell, cell_params):
  """Builds function to compute speed of hidden states.

  Args:
    rnn_cell: an RNNCell instance.
    cell_params: RNN parameters to use when applying the RNN.

  Returns:
    fixed_point_loss_fun: function that takes a batch of hidden states
      and inputs and computes the speed of the corresponding hidden
      states.
  """

  def fixed_point_loss_fun(h, x):
    """Computes the speed of hidden states.

    The speed is defined as the squared l2 distance between
    the current state and the next state, in response to a given
    input:

      Q = (1/2) || h - F(h, x) ||_2^2

    Args:
      h: The current state as a vector.
      x: The current input as a vector.

    Returns:
      fixed_point_loss_fun: A function that computes the fixed point speeds
        for a list or array of states.
    """
    h_next = rnn_cell.batch_apply(cell_params, x, h)
    return 0.5 * jnp.sum((h - h_next)**2, axis=1)

  return fixed_point_loss_fun


def find_fixed_points(fp_loss_fun,
                      initial_states,
                      x_star,
                      optimizer,
                      tolerance,
                      steps=range(1000)):
  """Run fixed point optimization.

  Args:
    fp_loss_fun: Function that computes fixed point speeds.
    initial_states: Initial state seeds.
    x_star: Input at which to compute fixed points.
    optimizer: A jax.experimental.optimizers tuple.
    tolerance: Stopping tolerance threshold.
    steps: Iterator over steps.

  Returns:
    fixed_points: Array of fixed points for each tolerance.
    loss_hist: Array containing fixed point loss curve.
    squared_speeds: Array containing the squared speed of each fixed point.
  """
  loss_hist, fps = utils.optimize(lambda h: jnp.mean(fp_loss_fun(h, x_star)),
                                  initial_states,
                                  optimizer,
                                  steps,
                                  stop_tol=tolerance)

  fixed_points = jax.device_get(fps)
  squared_speeds = jax.device_get(fp_loss_fun(fps, x_star))

  return fixed_points, loss_hist, squared_speeds


def exclude_outliers(points, threshold=np.inf, verbose=False):
  """Remove points that are not within some threshold of another point."""

  # Return all fixed points if tolerance is <= 0
  if np.isinf(threshold):
    return points

  # Return if there are less than two fixed points.
  if points.shape[0] <= 1:
    return points

  # Compute pairwise distances between all fixed points.
  distances = distance.squareform(distance.pdist(points))

  # Find distance to each nearest neighbor.
  nn_distance = np.partition(distances, 1, axis=0)[1]

  # Keep points whose nearest neighbor is within some distance threshold.
  keep_indices = np.where(nn_distance <= threshold)[0]

  # Log how many points were kept.
  if verbose:
    print(f'Keeping {len(keep_indices)} out of {len(points)} points.')

  return points[keep_indices]
