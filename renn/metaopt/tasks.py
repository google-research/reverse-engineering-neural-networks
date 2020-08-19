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
"""Load tasks from the library."""

import jax
import jax.numpy as jnp

from renn import utils
from sklearn.datasets import make_moons

from .task_lib import classification
from .task_lib import quadratic
from .task_lib import regression

__all__ = [
    'quad',
    'two_moons',
]

# Each task below is a function that takes problem parameters as
# arguments and returns a `problem_fun` function. This function takes a
# single argument, a PRNGKey, and returns a (x_init, loss_fun, data) tuple.
# x_init is a pytree initial problem parameters. loss_fun is a function
# that returns a scalar loss given parameters and a batch of data. Finally,
# data is an iterable pytree. All leaves of the tree must have the same first
# dimension, which is the number of steps to optimize for. These slices
# of data will be passed to the loss_fun during optimization.
quad = quadratic.loguniform


def two_moons(model, num_samples=1024, l2_pen=5e-3, seed=0):
  num_classes = 2
  x, y = make_moons(n_samples=num_samples,
                    shuffle=True,
                    noise=0.1,
                    random_state=seed)
  features = jnp.array(x)
  targets = jnp.array(y)

  return logistic_regression(model, features, targets, l2_pen=l2_pen)


def logistic_regression(model, features, targets, l2_pen=0.):
  """Helper function for logistic regression."""

  m, n = features.shape

  def problem_fun(prng_key):
    keys = jax.random.split(prng_key)
    input_shape = (-1, n)
    init_fun, predict_fun = model
    output_shape, x0 = init_fun(keys[0], input_shape)

    def loss_fun(x, step):
      del step
      logits = jnp.squeeze(predict_fun(x, features))
      data_loss = jnp.mean(jnp.log1p(jnp.exp(logits)) - targets * logits)
      reg_loss = l2_pen * utils.norm(x)
      return data_loss + reg_loss

    return x0, loss_fun

  return problem_fun
