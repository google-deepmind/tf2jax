Installation
============

Requirements
------------

TF2JAX requires:

- Python 3.11 or higher
- JAX 0.7.1 or higher
- TensorFlow 2.20.0 or higher
- NumPy 1.23.0 or higher

Installation Methods
--------------------

PyPI Installation
~~~~~~~~~~~~~~~~~

The recommended way to install TF2JAX is using pip:

.. code-block:: bash

   pip install tf2jax

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

To install the latest development version from GitHub:

.. code-block:: bash

   pip install git+https://github.com/google-deepmind/tf2jax.git

From Source
~~~~~~~~~~~

To install from source:

.. code-block:: bash

   git clone https://github.com/google-deepmind/tf2jax.git
   cd tf2jax
   pip install -e .

Verification
------------

To verify your installation:

.. code-block:: python

   import tf2jax
   print(f"TF2JAX version: {tf2jax.__version__}")
