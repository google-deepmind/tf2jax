# TF2JAX

TF2JAX is an experimental library for converting [TensorFlow] functions/graphs
to [JAX] functions.

Specifically, it aims to transform a `tf.function`, e.g.

```python
@tf.function
def tf_fn(x):
  return tf.sin(tf.cos(x))
```

to a python function equivalent to the following JAX code.

```python
def jax_fn(x):
  return jnp.sin(jnp.cos(x))
```

Users are able to apply additional JAX transforms (e.g. `jit`, `grad`, `vmap`,
`make_jaxpr`, etc.) to the converted function as they would any other code
written in JAX.

[TOC]

## Motivations

TF2JAX enables existing TensorFlow functions and models (including
[SavedModel](https://www.tensorflow.org/guide/saved_model) and
[TensorFlow Hub](https://www.tensorflow.org/hub/tf1_hub_module)) to be reused
and/or fine-tuned within JAX codebases. The conversion process is transparent
to the users, which is useful for debugging and introspection.

This also provide a pathway for JAX users to integrate JAX functions serialized
via `jax2tf.convert`, back into their existing JAX codebases.

See [section](#alternatives) at the end for comparison with an alternative
approach provided by `jax2tf.call_tf`.

## Disclaimer

This is experimental code with potentially unstable API, and there are no
guarantees for using it at this point in time. We highly recommend you
thoroughly test the resulting JAX functions to ensure they meet your
requirements.

## Quick start

The rest of this document assumes the following imports:

```python
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf  # Assumes this is v2.
import tf2jax
```

An example using the `convert` API and the Sonnet v2 MLP.

```python
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
```

`tf2jax.convert` has the signature `convert(fn: tf.Function, *args, **kwargs)`,
where `fn(*args, **kwargs)` is used to trace the function `fn` and generates the
corresponding `tf.GraphDef`. The `zeros_like` is not necessary, only used here
to demonstrate the JAX function is not memorizing the outputs.

### Example with a pure function

If your function is pure, i.e. it does not capture any variables, then you can
drop the parameters from the inputs and outputs of the converted function with
`tf2jax.convert_functional`.

```python
@tf.function
def forward(x):
  return tf.sin(tf.cos(x))

jax_func = tf2jax.convert_functional(forward, np.zeros_like(x))
jax_outputs = jax_func(x)
```

## Randomess and PRNG Keys

A TensorFlow function that make use of random ops will be converted to a JAX
function that takes a PRNG key as a keyword-only argument. TF2JAX will
complain loudly if a PRNG key is required but not provided.

```python
jax_outputs, jax_params = jax_func(jax_params, x, rng=jax.random.PRNGKey(42))
```

## Custom Gradient

Custom gradient support is highly experimental and disabled by default, please
report any errors.

```python
@tf.function
@tf.custom_gradient
def forward(x):
  e = tf.exp(x)
  def grad(dy):
    return dy * tf.sin(x) + e  # # This is deliberately the wrong gradient.
  return tf.reduce_sum(e), grad

with tf2jax.override_config("convert_custom_gradient", True):
  jax_func = tf2jax.convert_functional(forward, np.zeros_like(x))

jax_grads = jax.grad(jax_func)(x)
```

## Support for Serialization Formats

### SavedModel

[SavedModel](https://www.tensorflow.org/guide/saved_model) is the preferred
format for serializing TF2 functions.

```python
model = tf.Module()
model.f = forward
model.f(x)  # Dummy call.
tf.saved_model.save(model, "/tmp/blah")

restored = tf.saved_model.load("/tmp/blah")
jax_func, jax_params = tf2jax.convert(restored.f, np.zeros_like(x))
```

If the restored function has an unambiguous signature, i.e. it was only traced
once prior to export. Then TF2JAX can convert the function directly from its
GraphDef without tracing it again.

```python
jax_func, jax_params = tf2jax.convert_from_restored(restored.f)
```

### TF-Hub

The (legacy, TF1) [TF-Hub](https://www.tensorflow.org/hub/tf1_hub_module)
format is supported with minor boilerplate.

```python
import tensorflow_hub as hub

hub_model = hub.load("/tmp/blah")
jax_func, jax_params = tf2jax.convert(tf.function(hub_model))
jax_outputs, jax_params = jax_func(x)
```

## JAX to TensorFlow and Back Again.

`tf2jax.convert_functional` can convert the outputs of `jax2tf.convert` back
into JAX code.

```python
# Some JAX function.
def forward(*inputs):
  ...

# JAX -> TF
tf_func = jax2tf.convert(forward)

# JAX -> TF -> JAX
jax_func = tf2jax.convert_functional(tf.function(tf_func), *tree.map_structure(np.zeros_like, inputs))

# JAX -> TF -> SavedModel -> TF
model = tf.Module()
model.f = tf.function(tf_func)
model.f(*tree.map_structure(tf.zeros_like, inputs))  # Dummy call.
tf.saved_model.save(model, "/tmp/blah")
restored = tf.saved_model.load("/tmp/blah")

# JAX -> TF -> SavedModel -> TF -> JAX
jax_too_func = tf2jax.convert_functional(restored.f, *tree.map_structure(np.zeros_like, inputs))
```

## Additional Configuration

The behaviour of TF2JAX can be configured globally via `tf2jax.update_config`,
or configured locally via the context manager `tf2jax.override_config`.

### Strict shape and dtype checking

By default, TF2JAX will assert that the input shapes to the converted function
are compatible with the input shapes of the original function. This is because
some functions have shape dependent behaviours that will silently return the
incorrect outputs after conversion, e.g. some batchnorm implementation.

```python
jax_func = tf2jax.convert_functional(forward, np.zeros((10, 5), np.float32))

# This will raise an error.
jax_func(np.zeros((20, 5), np.float32))

# This will not.
with tf2jax.override_config("strict_shape_check", False):
  jax_func(np.zeros((20, 5), np.float32))
```

The input dtypes are not currently checked but this may change in the future.

```python
jax_func = tf2jax.convert_functional(forward, np.zeros((10, 5), np.float32))

# This will not raise an error.
jax_func(np.zeros((20, 5), np.int32))

# This will.
with tf2jax.override_config("strict_dtype_check", True):
  jax_func(np.zeros((20, 5), np.int32))
```

### Convert constants to bfloat16

TF2JAX allows users to trace the converted function with parameters and inputs
of different precision than the original function, e.g. `bfloat16` instead of
`float32`. To aid this, the configuration `force_const_float32_to_bfloat16`
and `force_const_float64_to_bfloat16` can be used to force float constants in
the original function into `bfloat16` precision, to avoid accidental type
promotion.

```python
@tf.function
def forward(x):
  return x + tf.constant(3.14, dtype=tf.float32)

with tf2jax.override_config("force_const_float32_to_bfloat16", True):
  jax_func = tf2jax.convert_functional(forward, np.zeros_like(x))
jax_bf16_outputs = jax_func(jnp.asarray(x, jnp.bfloat16))
```

## Limitations

Currently, only a subset of TensorFlow ops are supported, and not necessarily
all functionalities are supported for some ops. The code will fail fast. Support
for additional TensorFlow ops are added on a as-needed basis. Please submit your
requests via Github issues or send in your pull requests.

There will likely to be some cases where the resulting JAX code is not
equivalent to the TensorFlow code, both in terms of performance and numerical
outputs. The goal is to minimise differences in the latter for model endpoints,
ahead of improving performance.

TF2 control flows are supported with some limitations, e.g. for while loops,
the `cond` and `body` functions cannot have side effects such as assigning to
variables.

TF1 control flows are not supported.

## Alternatives (`jax2tf.call_tf`)

`jax2tf` now also offers the experimental `call_tf` function which allows JAX to
call TensorFlow functions. For compiled code, this works by staging out
TensorFlow functions to XLA.

https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax

From their documentation (as of 08.02.2021):

> The experimental function call_tf allows JAX to call TensorFlow functions.
> These functions can be called anywhere in a JAX computation, including in
> jax.jit, jax.pmap, jax.xmap, or inside JAX's control-flow primitives. For now,
> only reverse-mode autodiff is supported for these functions (no forward-mode
> autodiff, nor vmap).

The advantage of `call_tf` is that it implicitly covers all TensorFlow ops and
supports `custom_gradient` by deferring to TensorFlow during eager execution and
to XLA for compiled code.

The disadvantage is that it only supports a limited set of JAX transforms
(`jit`, `grad`, `pmap`, `remat`) and otherwise appears as a "black box" to
JAX (e.g. `vmap` is not supported, nor custom transforms). A TensorFlow function
must be compileable to XLA if it is to be jitted after `call_tf`.

## Citing TF2JAX

This repository is part of the [DeepMind JAX Ecosystem], to cite TF2JAX please
use the [DeepMind JAX Ecosystem citation].

## Contributing

We are happy to receive pull requests that improve our coverage of TensorFlow
ops.

[DeepMind JAX Ecosystem]: https://deepmind.com/blog/article/using-jax-to-accelerate-our-research "DeepMind JAX Ecosystem"
[DeepMind JAX Ecosystem citation]: https://github.com/deepmind/jax/blob/main/deepmind2020jax.txt "Citation"
[JAX]: https://github.com/google/jax "JAX on GitHub"
[TensorFlow]: https://github.com/tensorflow/tensorflow "TensorFlow on GitHub"
