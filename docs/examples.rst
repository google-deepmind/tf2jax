Examples
========

This section provides various examples of using TF2JAX.

Basic Conversion
----------------

Converting a simple mathematical function:

.. code-block:: python

   import tensorflow as tf
   import numpy as np
   import tf2jax

   @tf.function
   def simple_function(x, y):
       return tf.sin(x) + tf.cos(y)

   # Convert to JAX
   jax_func = tf2jax.convert_functional(
       simple_function, 
       np.zeros((5,), dtype=np.float32),
       np.zeros((5,), dtype=np.float32)
   )

   # Use with JAX transformations
   x = np.random.randn(5).astype(np.float32)
   y = np.random.randn(5).astype(np.float32)
   
   result = jax_func(x, y)
   gradient = jax.grad(jax_func, argnums=(0, 1))(x, y)

Neural Network Conversion
-------------------------

Converting a TensorFlow neural network:

.. code-block:: python

   import tensorflow as tf
   import numpy as np
   import tf2jax
   import jax.numpy as jnp

   class SimpleNN(tf.Module):
       def __init__(self, input_size, hidden_size, output_size):
           self.w1 = tf.Variable(tf.random.normal([input_size, hidden_size]))
           self.b1 = tf.Variable(tf.zeros([hidden_size]))
           self.w2 = tf.Variable(tf.random.normal([hidden_size, output_size]))
           self.b2 = tf.Variable(tf.zeros([output_size]))
       
       @tf.function
       def __call__(self, x):
           h = tf.nn.relu(tf.matmul(x, self.w1) + self.b1)
           return tf.matmul(h, self.w2) + self.b2

   # Create and convert model
   model = SimpleNN(10, 20, 5)
   x = np.random.randn(3, 10).astype(np.float32)
   
   jax_func, jax_params = tf2jax.convert(model, np.zeros_like(x))
   
   # Use with JAX
   result, _ = jax_func(jax_params, x)
   print(f"Output shape: {result.shape}")

SavedModel Conversion
---------------------

Converting from a SavedModel:

.. code-block:: python

   import tensorflow as tf
   import numpy as np
   import tf2jax

   # Create and save a model
   model = tf.Module()
   model.f = tf.function(lambda x: tf.sin(x))
   model.f(np.zeros((5,), dtype=np.float32))  # Dummy call
   
   tf.saved_model.save(model, "/tmp/simple_model")
   
   # Load and convert
   restored = tf.saved_model.load("/tmp/simple_model")
   jax_func = tf2jax.convert_functional_from_restored(restored.f)
   
   # Use the converted function
   x = np.random.randn(5).astype(np.float32)
   result = jax_func(x)

Custom Gradients
----------------

Working with custom gradients:

.. code-block:: python

   import tensorflow as tf
   import numpy as np
   import tf2jax

   @tf.function
   @tf.custom_gradient
   def custom_function(x):
       def grad(dy):
           return dy * tf.sin(x)  # Custom gradient
       return tf.reduce_sum(tf.exp(x)), grad

   # Enable custom gradient conversion
   with tf2jax.override_config("convert_custom_gradient", True):
       jax_func = tf2jax.convert_functional(
           custom_function, 
           np.zeros((5,), dtype=np.float32)
       )

   # Test gradient
   x = np.random.randn(5).astype(np.float32)
   gradient = jax.grad(jax_func)(x)
   print(f"Custom gradient: {gradient}")
