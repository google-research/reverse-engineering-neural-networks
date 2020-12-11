.. Reverse Engineering Neural Networks documentation master file, created by
   sphinx-quickstart on Thu Oct  8 14:58:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Reverse Engineering Neural Networks (renn)
===============================================================

Hello! You are at the main documentation for the `renn` python package.

What is RENN?
-------------
`renn` is a collection of python utilities for reverse engineering neural networks. The goal of the package is to be a shared repository of code, notebooks, and ideas for how to crack open the black box of neural networks to understand what they are doing and how they work. Our focus is on research applications.

Currently, the package focuses on understanding recurrent neural networks (RNNs). We provide code to build and train common RNN architectures, as well as code for understanding the dynamics of trained RNNs through dynamical systems analyses. The core tools for this involve finding and analyzing approximate fixed points of the dynamics of a trained RNN. More details on this are below.

All of `renn` uses the `JAX <https://github.com/google/jax/>`_ machine learning library for building neural networks and for automatic differentiation. We assume some basic familiarity with JAX in the documentation.

What can I use this for?
------------------------
Currently, the best use of `renn` is to train RNNs and then analyze the dynamics of those RNNs by studying numerical fixed points.

The best

Tutorials
---------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   notebooks/quickstart
   notebooks/RNN_sentiment_example

API Documentation
-----------------
.. toctree::
  :maxdepth: 3
  :caption: API Documentation
  api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
