# TF2JAX

![CI status](https://github.com/deepmind/tf2jax/workflows/ci/badge.svg)
![pypi](https://img.shields.io/pypi/v/tf2jax)

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

## Installation

You can install the latest released version of TF2JAX from PyPI via:

```sh
pip install tf2jax
```

or you can install the latest development version from GitHub:

```sh
pip install git+https://github.com/deepmind/tf2jax.git
```

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

## Randomness and PRNG Keys

A TensorFlow function that make use of random ops will be converted to a JAX
function that takes a PRNG key as a keyword-only argument. TF2JAX will
complain loudly if a PRNG key is required but not provided.

```python
jax_outputs, jax_params = jax_func(jax_params, x, rng=jax.random.PRNGKey(42))
```

## Custom Gradient

Custom gradient support is highly experimental, please report any errors.

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
jax_func, jax_params = tf2jax.convert(tf.function(hub_model), tf.zeros_like(x))
jax_outputs, updated_jax_params = jax_func(jax_params, x)
```

## JAX to TensorFlow and back again.

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

### Disable PreventGradient

If `jax2tf.convert(..., with_gradient=False)` is used to produce the initial TF
function (and possibly exported as SavedModel), then TF2JAX will respect the
inserted `tf.raw_ops.PreventGradient` ops and raise `LookupError` when computing
gradients.

This can be disabled by setting the configuration `raise_on_prevent_gradient` to
false (default is true), so that TF2JAX will only log a warning but otherwise
allow the gradient to be computed as though the `PreventGradient` ops were not
present.

```python
@tf.function
def prevent(x):
  return tf.raw_ops.PreventGradient(input=x * x, message="prevented")

jax_func = tf2jax.convert_functional(prevent, 0.0)
jax.grad(jax_func)(3.14)  # Raise LookupError.

with tf2jax.config.override_config("raise_on_prevent_gradient", False):
  jax_func = tf2jax.convert_functional(prevent, 0.0)
g = jax.grad(jax_func)(3.14)  # Returns 6.28.

```

### Infer Cumulative Reductions

If the `infer_cumulative_reduction_from_jax2tf` flag is true (default) then
TF2JAX will attempt to infer `cummax`, `cummin`, `cumprod` and `cumsum`
operations from `reduce_window` operations generated by JAX2TF. This provides
better performance because `reduce_window` implementation of these ops have
O(N^2) complexity on CPU and GPU backends, and can suffer from long compilation
times due to aggressive constant folding.

See [jax2tf_cumulative_reduction] for more context.

## JAX2TF Native Serialization and XlaCallModule.

From JAX v0.4.7 and onward, `jax2tf.convert` preferred mode of operation (soon
to be default) is **native serialization** in which the target function is
lowered to [StableHLO] and wrapped in a single TensorFlow op, `XlaCallModule`.

The new native serialization format will more faithfully reproduce the semantics
of the target function, at the cost of some reduced flexibility for downstream
processing as the computation graph is no longer exposed as a tf.Graph.

`XlaCallModule` is supported by TF2JAX from v0.3.4 and onward.

However as this makes use of a custom JAX primitive that aims to encapsulate
[StableHLO] payload found in `XlaCallModule`, it does not possess JAX rules for
transformations such as (but not limited to) batching and differentiation.

* **Differentiation**: first order derivative of serialized function is
still supported through custom gradients requested at serialization time with
`jax2tf.convert(..., with_gradient=True)`. This is the default behaviour.
* **Batching**: `jax.vmap` will fail, though users may be able to naively
replicate the desired behavior with `jax.lax.map`, albeit with poorer
performance.

### Platform Specificity

Natively serialized JAX programs are platform specific ([link](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#natively-serialized-jax-modules-are-platform-specific)). Executing a natively
serialized program on platforms other than the one for which it was lowered,
would raise a ValueError, e.g.:

```python
ValueError: Unsupported backend: `cpu` not in `('tpu',)`.
```

This matches the behaviour of `XlaCallModule`.

Users can disable this check via a config flag, but the resulting program may
be slower or fail to execute completely.

```python
with tf2jax.override_config("xlacallmodule_strict_checks", False):
  jax_func(np.zeros((20, 5), np.float32))
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

## Alternatives

### `jax2tf.call_tf`

`jax2tf` now also offers the experimental `call_tf` function which allows JAX to
call TensorFlow functions. For compiled code, this works by staging out
TensorFlow functions to XLA.

From the [jax2tf documentation], as of 2022-07-22:

> The function `call_tf` allows JAX functions to call TensorFlow functions.
> These functions can be called anywhere in a JAX computation, including in
> staging contexts `jax.jit`, `jax.pmap`, `jax.xmap`, or inside JAX's
> control-flow primitives. In non-staging contexts, the TensorFlow function is
> called in eager mode. For now, only reverse-mode autodiff is supported for
> these functions (no forward-mode autodiff, nor `vmap`).

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
[jax2tf documentation]: https://github.com/google/jax/blob/master/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax "jax2tf documentation"
[jax2tf_cumulative_reduction]: https://github.com/google/jax/blob/main/jax/experimental/jax2tf/jax2tf.py#L2172
[StableHLO]: https://github.com/openxla/stablehlo
