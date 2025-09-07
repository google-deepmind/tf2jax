# Quick Start Guide

This guide will help you get started with TF2JAX quickly.

## Basic Usage

### Converting a Simple Function

```python
import tensorflow as tf
import numpy as np
import tf2jax

@tf.function
def tf_function(x):
    return tf.sin(tf.cos(x))

# Convert to JAX
jax_function = tf2jax.convert_functional(tf_function, np.zeros((10,), dtype=np.float32))

# Use the JAX function
x = np.random.randn(10).astype(np.float32)
result = jax_function(x)
```

### Converting a Model with Variables

```python
import tensorflow as tf
import numpy as np
import tf2jax

class SimpleModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable(tf.random.normal([10, 5]))
        self.b = tf.Variable(tf.zeros([5]))
    
    @tf.function
    def __call__(self, x):
        return tf.nn.relu(tf.matmul(x, self.w) + self.b)

model = SimpleModel()
x = np.random.randn(3, 10).astype(np.float32)

# Convert to JAX
jax_function, jax_params = tf2jax.convert(model, np.zeros_like(x))

# Use the JAX function
result, updated_params = jax_function(jax_params, x)
```

## Configuration

TF2JAX provides several configuration options:

```python
# Disable strict shape checking
with tf2jax.override_config("strict_shape_check", False):
    jax_func = tf2jax.convert_functional(tf_func, sample_input)

# Enable custom gradient conversion
with tf2jax.override_config("convert_custom_gradient", True):
    jax_func = tf2jax.convert_functional(tf_func, sample_input)
```

## Error Handling

TF2JAX provides informative error messages:

```python
try:
    jax_func = tf2jax.convert_functional(tf_func, sample_input)
except tf2jax.UnsupportedOperationError as e:
    print(f"Unsupported operation: {e.op_name}")
    print(f"Suggestion: {e.suggestion}")
except tf2jax.ShapeMismatchError as e:
    print(f"Shape mismatch: expected {e.expected_shape}, got {e.actual_shape}")
```
