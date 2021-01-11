What is RENN?
-------------
`renn` is a collection of python utilities for reverse engineering neural networks. The goal of the package is to be a shared repository of code, notebooks, and ideas for how to crack open the black box of neural networks to understand what they are doing and how they work. Our focus is on research applications.

Currently, the package focuses on understanding recurrent neural networks (RNNs). We provide code to build and train common RNN architectures, as well as code for understanding the dynamics of trained RNNs through dynamical systems analyses. The core tools for this involve finding and analyzing approximate fixed points of the dynamics of a trained RNN.

All of `renn` uses the `JAX <https://github.com/google/jax/>`_ machine learning library for building neural networks and for automatic differentiation. We assume some basic familiarity with JAX in the documentation.

What can I use this for?
------------------------
Currently, the best use of `renn` is to train RNNs and then analyze the dynamics of those RNNs by studying numerical fixed points.

The best examples of this are in the following research papers:

* `Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks, Sussillo and Barak, Neural Computation, 2013. <https://www.mitpressjournals.org/doi/full/10.1162/NECO_a_00409>`_
* `Reverse engineering recurrent networks for sentiment classification reveals line attractor dynamics, Maheswaranathan*, Williams* et al, NeurIPS 2019. <https://arxiv.org/abs/1906.10720>`_
* `Universality and individuality in neural dynamics across large populations of recurrent networks, Maheswaranathan*, Williams* et al, NeurIPS 2019. <https://arxiv.org/abs/1907.08549>`_
* `How recurrent networks implement contextual processing in sentiment analysis, Maheswaranathan* and Sussillo*, ICML 2020. <https://arxiv.org/abs/2004.08013>`_
* `The geometry of integration in text classification RNNs, Aitken*, Ramasesh* et al, arXiv 2020. <https://arxiv.org/abs/2010.15114>`_
* `Reverse engineering learned optimizers reveals known and novel mechanisms, Maheswaranathan et al, arXiv 2020. <https://arxiv.org/abs/2011.02159>`_
