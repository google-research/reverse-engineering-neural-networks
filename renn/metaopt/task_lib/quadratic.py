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
"""Defines quadratic loss functions."""

import jax
import jax.numpy as jnp

HIGHEST = jax.lax.Precision.HIGHEST


def quadform(hess, x, precision):
  """Computes a quadratic form (x^T @ H @ x)."""
  u = jnp.dot(hess, x, precision=precision)  # u = Hx
  return jnp.inner(x, u, precision=precision)


def loguniform(n, lambda_min, lambda_max, precision=HIGHEST):
  """Quadratic loss function with loguniform eigenvalues.

  The loss is: f(x) = (1/2) x^T H x + x^T v + b.

  The eigenvalues of the Hessian (H) are sampled uniformly on a
  logarithmic grid from lambda_min to lambda_max.

  Args:
    n: int, Problem dimension (number of parameters).
    lambda_min: float, Minimum eigenvalue of the Hessian.
    lambda_max: float, Maximum eigenvalue of the Hessian.
    precision: Which lax precision to use (default: HIGHEST).

  Returns:
    problem_fun: Function that takes a jax PRNGkey and a precision argument
      and returns an (initial_params, loss_fun) tuple.
  """

  def problem_fun(key):
    """Builds a quadratic loss problem."""
    pkey, ekey, qkey, vkey = jax.random.split(key, 4)

    # Sample eigenvalues.
    log_eigenvalues = jax.random.uniform(ekey,
                                         shape=(n,),
                                         minval=lambda_min,
                                         maxval=lambda_max)
    eigenvalues = 10**log_eigenvalues

    # Build orthonormal basis.
    basis = jax.nn.initializers.orthogonal()(qkey, shape=(n, n))

    # Define hessian.
    hess = jnp.dot(jnp.dot(basis, jnp.diag(eigenvalues), precision=precision),
                   basis.T,
                   precision=precision)

    # Random vector for the linear term in the loss.
    v = jax.random.normal(vkey, shape=(n,))

    # Compute an offset such that the global minimum has a loss of zero.
    xstar = jnp.linalg.solve(hess, -v)
    offset = -0.5 * quadform(hess, xstar, precision=precision) - jnp.inner(v, xstar, precision=precision)  # pylint: disable=line-too-long

    def loss_fun(x, _):
      return 0.5 * quadform(hess, x, precision=precision) + jnp.inner(v, x, precision=precision) + offset  # pylint: disable=line-too-long

    x_init = jax.random.normal(pkey, shape=(n,))
    return x_init, loss_fun

  return problem_fun
