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
"""Meta-optimization framework."""

from collections import defaultdict
import functools

import jax
import jax.numpy as jnp

import numpy as np

from .. import utils
from . import losses


def unroll_scan(initial_params, loss_fun, optimizer, num_steps, decorator):
  """Runs an optimizer on a given problem, using lax.scan.

  Note: this will cache parameters during the unrolled loop, and thus uses a
  lot of device memory, therefore it is not good for simply evaluating
  (testing) an optimizer. Instead, it is useful for when we need to compute
  a _derivative_ of some final loss with respect to the optimizer parameters.

  Args:
    initial_params: Initial parameters.
    loss_fun: A function that takes (params, step) and returns a loss.
    optimizer: A tuple containing an optimizer init function, an update
      function, and a get_params function.
    num_steps: int, number of steps to run the optimizer.
    decorator: callable, Optional decorator function used to wrap the
      apply_fun argument to lax.scan.

  Returns:
    final_params: Problem parameters after running the optimizer.
    fs: Loss at every step of the loop.
  """

  # Gradient of the loss function.
  f_df = jax.jit(jax.value_and_grad(loss_fun))

  # Get optimizer functions.
  opt_init, opt_update, get_params = optimizer

  # Build function that applies a single step of the optimizer.
  @decorator
  def _apply(state, step):
    """Applies one step of the optimizer."""
    params = get_params(state)  # Get inner parameters.
    f, df = f_df(params, step)  # Loss and gradient.
    next_state = opt_update(step, df, state)  # Step the optimizer.
    return next_state, f

  # Initialize and run the optimizer.
  initial_state = opt_init(initial_params)
  steps = jnp.arange(num_steps)
  final_state, fs = jax.lax.scan(_apply, initial_state, steps)

  return get_params(final_state), fs


def unroll_for(initial_params, loss_fun, optimizer, extract_state, steps):
  """Runs an optimizer on a given problem, using a for loop.

  Note: this is slower to compile than unroll_scan, but can be used to store
  intermediate computations (such as the optimizer state or problem
  parameters) at every iteration, for further analysis.

  Args:
    initial_params: Initial parameters.
    loss_fun: A function that takes (params, step) and returns a loss.
    optimizer: A tuple containing an optimizer init function, an update
      function, and a get_params function.
    extract_state: A function that given some optimizer state, returns
      what from that optimizer state to store. Note that each optimizer state
      is different, so this function depends on a particular optimizer.
    steps: A generator that yields integers from (0, num_steps).

  Returns:
    results: Dictionary containing results to save.
  """

  # Gradient of the loss function.
  f_df = jax.jit(jax.value_and_grad(loss_fun))

  # Get optimizer functions.
  opt_init, opt_update, get_params = optimizer
  opt_state = opt_init(initial_params)

  def extract(opt_state):
    """Function to extract state from a packed OptimizerState object."""
    states_flat, _, subtrees = opt_state
    full_states = map(jax.tree_unflatten, subtrees, states_flat)
    return list(map(extract_state, full_states))

  # Data structure to store intermediate computation.
  store = defaultdict(list)

  # Optimize
  for step in steps:

    # Query function to get loss and gradient.
    params = get_params(opt_state)  # Get parameters.
    loss, gradient = f_df(params, step)  # Loss and gradient.

    # Store current loss, parameters, and optimizer state.
    store['loss'].append(loss)
    store['params'].append(params)
    store['state'].append(extract(opt_state))
    store['gradient'].append(gradient)

    # Apply the optimizer.
    opt_state = opt_update(step, gradient, opt_state)

  # Collect results as numpy arrays.
  return {k: jax.device_get(v) for k, v in store.items()}


def build_metaobj(problem_fun,
                  optimizer_fun,
                  num_inner_steps,
                  meta_loss=losses.mean,
                  l2_penalty=0.0,
                  decorator=jax.remat):
  """Builds a meta-objective function.

  Args:
    problem_fun: callable, Takes a PRNGKey argument and returns initial
      parameters and a loss function.
    optimizer_fun: callable, Takes a PRNGKey argument and returns an
      optimizer tuple (as in jax.experimental.optimizers).
    num_inner_steps: int, Number of optimization steps.
    meta_loss: callable, Function to use to compute a scalar meta-loss.
    l2_penalty: float, L2 penalty to apply to the meta-parameters.
    decorator: callable, Optional function to wrap the apply_fun argument to
      lax.scan. By default, this is jax.remat, which will rematerialize the
      forward computation when computing the gradient, trading off computation
      for memory. Using the identity function will turn off remat.

  Returns:
    meta_objective: callable, Function that takes meta-parameters and a
      PRNGKey and returns a scalar meta-objective and the inner loss history.
  """

  def meta_objective(meta_params, prng_key):
    """Computes meta-loss for a given set of meta-parameters.

    Args:
      meta_params: Parameters of the optimizer.
      prng_key: Pseudorandom number generator key, used to sample the
        problem to train on.

    Returns:
      meta_objective_value: A scalar value that is the meta-objective.
      fs: An array of training losses (training curve).
    """

    # Build a problem.
    x0, loss_fun = problem_fun(prng_key)

    # Build an optimizer with these meta-parameters.
    opt = optimizer_fun(meta_params)

    # Run the optimizer for num_inner_steps.
    _, fs = unroll_scan(x0, loss_fun, opt, num_inner_steps, decorator)

    # Final objective with an l2 norm penalty.
    mobj = meta_loss(fs) + l2_penalty * utils.norm(meta_params)

    return mobj, fs

  return meta_objective


def evaluate(opt, problem_fun, num_steps, eval_key, num_repeats=64):
  """Evaluates an optimizer on a given problem.

  Args:
    opt: An optimizer tuple of functions (init_opt, update_opt, get_params)
      to evaluate.
    problem_fun: A function that returns an (initial_params, loss_fun,
      fetch_data) tuple given a PRNGKey.
    num_steps: Number of steps to run the optimizer for.
    eval_key: Base PRNGKey used for evaluation.
    num_repeats: Number of different evaluation seeds to use.

  Returns:
    losses: Array of loss values with shape (num_repeats, num_steps)
      containing the training loss curve for each random seed.
  """

  @jax.jit
  def _run(k):
    return unroll_scan(*problem_fun(k), opt, num_steps, utils.identity)[1]

  keys = jax.random.split(eval_key, num=num_repeats)
  return jax.device_get(jax.vmap(_run)(keys))


def outer_loop(key,
               initial_meta_params,
               meta_objective,
               meta_optimizer,
               steps,
               batch_size=1,
               save_every=None,
               clip_value=np.inf):
  """Meta-trains an optimizer.

  Args:
    key: Jax PRNG key, used for initializing the inner problem.
    initial_meta_params: pytree, Initial meta-parameters.
    meta_objective: function, Computes a (scalar) loss given meta-parameters
      and an array (batch) of random seeds.
    meta_optimizer: tuple of functions, Defines the meta-optimizer to use (for
      example, a jax.experimental.optimizers Optimizer tuple).
    steps: A generator that yields integers from (0, num_steps).
    batch_size: int, Number of problems to train per batch.
    save_every: int, Specifies how often to store auxiliary information. If
      None, then information is never stored (Default: None).
    clip_value: float, Specifies the gradient clipping value (maximum
      gradient norm) (Default: np.inf).

  Returns:
    final_params: Final optimized parameters.
    store: Dict containing saved auxiliary information during optimization.
  """
  # Store quantities during outer-optimization.
  store = defaultdict(list)

  # Build meta-optimizer.
  init_opt, update_opt, get_params = meta_optimizer
  mopt_state = init_opt(initial_meta_params)

  # Function to comppute the meta-gradient and meta-hessian.
  meta_val_and_grad = jax.value_and_grad(meta_objective)

  # Function to clip gradient values.
  clip_fun = functools.partial(clip, value=clip_value)

  @jax.jit
  def outer_step(key, step, state):
    """Single step of meta-optimization."""

    # Refresh random state.
    prng_key = jax.random.fold_in(key, step)
    prng_keys = jnp.stack(jax.random.split(prng_key, batch_size))

    # Get optimizer with the current meta-parameters.
    meta_params = get_params(state)

    # Evaluate the meta-objective and meta-gradient
    mobj, mgrad = meta_val_and_grad(meta_params, prng_keys)

    # Clip gradient values.
    clipped_mgrad = jax.tree_map(clip_fun, mgrad)

    # Update the optimizer
    state = update_opt(step, clipped_mgrad, state)

    return mobj, mgrad, state

  # Run outer optimization.
  for step in steps:
    mobj, mgrad, mopt_state = outer_step(key, step, mopt_state)

    # Optionally store information.
    if (save_every is not None) and (step % save_every) == 0:
      store['step'].append(step)
      store['mobj'].append(mobj)
      store['mgrad'].append(mgrad)

  final_params = get_params(mopt_state)
  store = {k: np.array(v) for k, v in store.items()}
  return final_params, store


def clip(x, value=jnp.inf):
  """Clips elements of x to have magnitude less than or equal to value."""

  # Guard to short circuit if no value is given.
  if value == jnp.inf:
    return x

  mask = (jnp.abs(x) <= value).astype(jnp.float32)
  return x * mask + value * (1. - mask) * jnp.sign(x)
