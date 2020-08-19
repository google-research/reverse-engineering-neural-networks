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
"""Functions for computing a scalar objective from a loss curve."""

import jax.numpy as jnp


def nanmin(fs):
  """Computes the NaN-aware minimum over the loss curve."""
  return jnp.nanmin(fs[1:]) / fs[0]


def final(fs):
  """Returns the final loss value."""
  return fs[-1] / fs[0]


def mean(fs):
  """Returns the average over the loss values."""
  return jnp.mean(fs) / fs[0]
