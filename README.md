# Reverse Engineering Neural Networks (RENN)

![build](https://github.com/google-research/reverse-engineering-neural-networks/workflows/build/badge.svg)

`renn` is a collection of python utilities for reverse engineering neural networks. The goal of the package is to be a shared repository of code, notebooks, and ideas for how to crack open the black box of neural networks to understand what they are doing and how they work. Our focus is on research applications.

Currently, the package focuses on understanding recurrent neural networks (RNNs). We provide code to build and train common RNN architectures, as well as code for understanding the dynamics of trained RNNs through dynamical systems analyses. The core tools for this involve finding and analyzing approximate fixed points of the dynamics of a trained RNN.

All of `renn` uses the [JAX](https://github.com/google/jax/) machine learning library for building neural networks and for automatic differentiation. We assume some basic familiarity with JAX in the documentation.

*See the [documentation](https://reverse-engineering-neural-networks.readthedocs.io/en/latest/) for more information.*

Authors:
- Niru Maheswaranathan (nirum@google.com)
- Vinay Ramasesh (ramasesh@google.com)
