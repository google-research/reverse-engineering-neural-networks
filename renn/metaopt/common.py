"""Update functions for common optimizers."""

import jax.numpy as jnp


def momentum(alpha, beta):

  def update(g, v):
    """Momentum update.

    Args:
      g: gradient
      v: velocity
    """
    v = beta * v + g
    return -alpha * v

  return update


def nesterov(alpha, beta):

  def update(g, v):
    """Nesterov momentum update.

    Args:
      g: gradient
      v: velocity
    """
    v = beta * v + g
    return -alpha * (beta * v + g)

  return update


def adagrad(alpha, beta):

  def update(g, g_sq, v):
    """Adagrad update.

    Args:
      g: gradient
      g_sq: cumulative squared gradient
      v: velocity
    """
    g_sq += jnp.square(g)
    g_norm = jnp.where(g_sq > 0, g / jnp.sqrt(g_sq), 0.)
    v = (1. - beta) * g_norm + beta * v
    return -alpha * v

  return update


def rmsprop(alpha, beta=0.9, eps=1e-5):

  def update(g, m):
    """RMSProp update.

    Args:
      g: gradient
      m: running average of the second moment
    """
    m = beta * m + jnp.square(g) * (1. - beta)
    g_norm = g / jnp.sqrt(m + eps)
    return -alpha * g_norm

  return update


def adam(alpha, beta1=0.9, beta2=0.999, eps=1e-5):

  def update(g, m, v):
    """Adam update.

    Note: this is uncorrected.

    Args:
      g: gradient
      v: running average of the first moment (momentum)
      m: running average of the second moment (normalization)
    """
    v = (1 - beta1) * g + beta1 * v  # First moment.
    m = (1 - beta2) * jnp.square(g) + beta2 * m  # Second moment.
    return -alpha * v / (jnp.sqrt(m) + eps)

  return update


def cwrnn(cell_apply, readout_apply):

  def update(g, h):
    """Component-wise RNN Optimizer update.

    Args:
      g: gradient
      h: RNN state
    """
    h_next = cell_apply(g, h)
    return readout_apply(h_next)

  return update
