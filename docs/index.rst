TF2JAX Documentation
====================

TF2JAX is an experimental library for converting TensorFlow functions/graphs to JAX functions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Installation
============

You can install the latest released version of TF2JAX from PyPI via:

.. code-block:: bash

   pip install tf2jax

or you can install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/google-deepmind/tf2jax.git

Quick Start
===========

The rest of this document assumes the following imports:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import numpy as np
   import tensorflow as tf  # Assumes this is v2.
   import tf2jax

An example using the `convert` API and the Sonnet v2 MLP:

.. code-block:: python

   import sonnet.v2 as snt

   model = snt.nets.MLP((64, 10,))

   @tf.function
   def forward(x):
     return model(x)

   x = np.random.normal(size=(128, 16)).astype(np.float32)

   # TF -> JAX, jax_params are the network parameters of the MLP
   jax_func, jax_params = tf2jax.convert(forward, np.zeros_like(x))

   # Call JAX, also return updated jax_params (e.g. variable, batchnorm stats)
   jax_outputs, jax_params = jax_func(jax_params, x)

Example with a pure function
============================

If your function is pure, i.e. it does not capture any variables, then you can
drop the parameters from the inputs and outputs of the converted function with
`tf2jax.convert_functional`.

.. code-block:: python

   @tf.function
   def forward(x):
     return tf.sin(tf.cos(x))

   jax_func = tf2jax.convert_functional(forward, np.zeros_like(x))
   jax_outputs = jax_func(x)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
