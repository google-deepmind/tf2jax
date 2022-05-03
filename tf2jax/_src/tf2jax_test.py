# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests tf2jax."""

from absl.testing import parameterized

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

import sonnet as snt
import tensorflow as tf
from tf2jax._src import tf2jax
import tree


class ModelsTest(tf.test.TestCase, parameterized.TestCase):

  def _test_convert(self, tf_func, inputs):
    jax_func, jax_params = tf2jax.convert(
        tf.function(tf_func), np.zeros_like(inputs))
    jax_func = self.variant(jax_func)

    jax_results, jax_params = jax_func(jax_params, inputs)
    tf_results, tf_params = tf_func(inputs)

    # Check outputs
    for tf_res, jax_res in zip(
        tree.flatten(tf_results), tree.flatten(jax_results)):
      self.assertAllClose(tf_res.numpy(), jax_res, atol=1e-4)

    # Check params (batchnorm stats changed).
    for tf_var in tree.flatten(tf_params):
      jax_var = jax_params[tf_var.name.split(":")[0]]
      self.assertAllClose(tf_var.numpy(), jax_var, atol=1e-5)

  @chex.variants(with_jit=True, without_jit=True)
  def test_mlp(self):
    tf.random.set_seed(42)

    inputs = np.linspace(-1., 1., 128 * 16, dtype=np.float32).reshape((128, 16))

    model = snt.nets.MLP((64, 10,))
    def tf_func(x):
      outputs = model(x)
      return outputs, model.variables

    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("inference", False), ("training", True)),
          named=True,
      ))
  def test_resnet(self, training):
    tf.random.set_seed(42)

    inputs = np.linspace(
        -1., 1., 2 * 64 * 64 * 3, dtype=np.float32).reshape((2, 64, 64, 3))

    model = snt.nets.ResNet([1, 1, 1, 1], 10)
    def tf_func(x):
      outputs = model(x, is_training=training)
      return outputs, model.variables

    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("inference", False), ("training", True)),
          named=True,
      ))
  def test_vqvae(self, training):
    tf.random.set_seed(42)

    inputs = np.linspace(
        -1., 1., 2 * 64 * 64 * 3, dtype=np.float32).reshape((2, 64, 64, 3))

    model = snt.nets.VectorQuantizer(3, 100, 0.1)
    def tf_func(x):
      return model(x, is_training=training), model.variables

    self._test_convert(tf_func, inputs)


class FeaturesTest(tf.test.TestCase, parameterized.TestCase):

  def _setup_saved_model(self, *inputs):
    def tf_func(x):
      return tf.exp(x)

    # Save.
    model = tf.Module()
    model.f = tf.function(tf_func)
    for inp in inputs:
      if isinstance(inp, tf.TensorSpec):
        model.f.get_concrete_function(inp)  # Dummy call.
      else:
        model.f(np.zeros_like(inp))  # Dummy call.
    tmp_dir = self.create_tempdir()
    tf.saved_model.save(model, tmp_dir.full_path)

    return tf_func, tf.saved_model.load(tmp_dir.full_path)

  @chex.variants(with_jit=True, without_jit=True)
  def test_saved_model(self):
    dummy_inputs = np.zeros([10, 5], dtype=np.float32)
    tf_func, restored = self._setup_saved_model(dummy_inputs)

    # Convert to Jax (with and without explicit inputs).
    for inp in [(), (dummy_inputs,)]:
      if inp:
        jax_func, _ = tf2jax.convert(restored.f, *inp)
      else:
        jax_func, _ = tf2jax.convert_from_restored(restored.f)
      jax_func = self.variant(jax_func)

      test_inputs = np.ones([20, 5], dtype=np.float32)
      expected_outputs = tf_func(test_inputs)
      with tf2jax.override_config("strict_shape_check", False):
        actual_outputs, _ = jax_func({}, test_inputs)
      self.assertAllClose(expected_outputs, actual_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_saved_model_ambiguous(self):
    dummy_one = np.zeros([10, 5], dtype=np.float32)
    dummy_two = np.zeros([10, 5, 3], dtype=np.float32)
    tf_func, restored = self._setup_saved_model(dummy_one, dummy_two)

    with self.assertRaisesRegex(ValueError, "Found 2 concrete functions"):
      jax_func, _ = tf2jax.convert_from_restored(restored.f)
    jax_func, _ = tf2jax.convert(restored.f, dummy_two)
    jax_func = self.variant(jax_func)

    test_inputs = np.ones([20, 7, 2], dtype=np.float32)
    expected_outputs = tf_func(test_inputs)
    with tf2jax.override_config("strict_shape_check", False):
      actual_outputs, _ = jax_func({}, test_inputs)
    self.assertAllClose(expected_outputs, actual_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_saved_model_functional(self):
    dummy_inputs = np.zeros([10, 5], dtype=np.float32)
    tf_func, restored = self._setup_saved_model(dummy_inputs)

    jax_func = tf2jax.convert_functional_from_restored(restored.f)
    jax_func = self.variant(jax_func)

    test_inputs = np.ones([20, 5], dtype=np.float32)
    expected_outputs = tf_func(test_inputs)
    with tf2jax.override_config("strict_shape_check", False):
      actual_outputs = jax_func(test_inputs)
    self.assertAllClose(expected_outputs, actual_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_saved_model_polymorphic_shape(self):
    dummy_input_spec = tf.TensorSpec((None, None, 2), dtype=tf.float32)
    tf_func, restored = self._setup_saved_model(dummy_input_spec)

    jax_func = tf2jax.convert_functional_from_restored(restored.f)
    jax_func = self.variant(jax_func)

    test_inputs = np.ones([10, 5, 2], dtype=np.float32)
    expected_outputs = tf_func(test_inputs)
    actual_outputs = jax_func(test_inputs)
    self.assertAllClose(expected_outputs, actual_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_strict_check(self):
    def tf_func(x):
      return x.shape

    orig_inputs = np.zeros((10, 5), dtype=np.float32)

    model = tf.Module()
    model.f = tf.function(tf_func)
    model.f(orig_inputs)
    tmp_dir = self.create_tempdir()
    tf.saved_model.save(model, tmp_dir.full_path)
    restored = tf.saved_model.load(tmp_dir.full_path)

    jax_func = tf2jax.convert_functional_from_restored(restored.f)
    jax_func = self.variant(jax_func)

    with self.subTest("valid_input_shape"):
      expected_outputs = tf_func(orig_inputs)
      with tf2jax.override_config("strict_shape_check", True):
        actual_outputs = jax_func(orig_inputs)
        self.assertAllClose(expected_outputs, actual_outputs)
      with tf2jax.override_config("strict_dtype_check", True):
        actual_outputs = jax_func(orig_inputs)
        self.assertAllClose(expected_outputs, actual_outputs)

    test_inputs = np.ones([10, 5, 2], dtype=np.float32)
    with self.subTest("invalid_input_shape"):
      expected_outputs = tf_func(test_inputs)
      with tf2jax.override_config("strict_shape_check", True):
        with self.assertRaisesRegex(ValueError, "incompatible input shape"):
          actual_outputs = jax_func(test_inputs)
      with tf2jax.override_config("strict_shape_check", False):
        actual_outputs = jax_func(test_inputs)
        self.assertNotAllClose(expected_outputs, actual_outputs)

    test_inputs = np.ones([10, 5], dtype=np.int32)
    with self.subTest("invalid_input_dtype"):
      expected_outputs = tf_func(test_inputs)
      with tf2jax.override_config("strict_dtype_check", True):
        with self.assertRaisesRegex(ValueError, "incompatible input dtype"):
          actual_outputs = jax_func(test_inputs)
      with tf2jax.override_config("strict_dtype_check", False):
        actual_outputs = jax_func(test_inputs)
        self.assertAllClose(expected_outputs, actual_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("float32", tf.float32), ("float64", tf.float64)),
          named=True,
      ))
  def test_force_bf16_consts(self, tf_dtype):
    @tf.function
    def tf_func(x):
      return x + tf.constant(3.14, dtype=x.dtype)

    np_inputs = np.array(42.0, tf_dtype.as_numpy_dtype())
    tf_outputs = tf_func(np_inputs)

    dtype_config_name = ("force_const_float32_to_bfloat16"
                         if tf_dtype == tf.float32 else
                         "force_const_float64_to_bfloat16")

    # This is the default.
    with tf2jax.override_config(dtype_config_name, False):
      jax_func = tf2jax.convert_functional(tf_func, np_inputs)
    jax_func = self.variant(jax_func)
    orig_jax_outputs = jax_func(np_inputs)
    self.assertEqual(
        jnp.array(np_inputs).dtype,
        jnp.array(orig_jax_outputs).dtype)
    self.assertAllClose(tf_outputs, orig_jax_outputs)

    with tf2jax.override_config(dtype_config_name, True):
      jax_func = tf2jax.convert_functional(tf_func, np_inputs)
    jax_func = self.variant(jax_func)
    forced_jax_outputs = jax_func(jnp.asarray(np_inputs, jnp.bfloat16))
    self.assertEqual(jnp.bfloat16, forced_jax_outputs.dtype)
    self.assertAllClose(tf.cast(tf_outputs, tf.bfloat16), forced_jax_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_force_bf16_consts_for_leaks(self):
    @tf.function
    def tf_func(x):
      return x + tf.constant(3.14, dtype=tf.float32)

    np_inputs = np.array(42.0, np.float32)
    tf_outputs = tf_func(np_inputs)

    class CachedFn(hk.Module):

      cache = []

      def __init__(self):
        super().__init__(name=None)
        if not self.cache:
          with tf2jax.override_config("force_const_float32_to_bfloat16", True):
            self.cache.append(jax.jit(
                tf2jax.convert_functional(tf_func, np_inputs)))

      def __call__(self, x):
        return self.cache[0](x)

    def call_cached_fn(x):
      return CachedFn()(x)

    cached_fn = hk.without_apply_rng(hk.transform(call_cached_fn))
    params = self.variant(cached_fn.init)(jax.random.PRNGKey(42), np_inputs)
    jax_inputs = jnp.asarray(np_inputs, jnp.bfloat16)
    jax_outputs = self.variant(cached_fn.apply)(params, jax_inputs)

    self.assertEqual(jnp.bfloat16, jax_outputs.dtype)
    self.assertAllClose(tf.cast(tf_outputs, tf.bfloat16), jax_outputs)

  def test_static_argnums(self):
    @tf.function
    def tf_func(x):
      return tf.zeros((x,))

    with self.subTest("literal_input"):
      jax_func = tf2jax.convert_functional(tf_func, 10)
      self.assertEqual(jax_func(10).shape, (10,))
      with self.assertRaisesRegex(ValueError, "Found unexpected literal value"):
        jax_func(13)
      with self.assertRaisesRegex(ValueError, "Found unexpected tracer"):
        jax.jit(jax_func)(13)
      with self.assertRaisesRegex(ValueError, "Found unexpected literal value"):
        jax.jit(jax_func, static_argnums=0)(13)

    with self.subTest("array_input"):
      jax_func = tf2jax.convert_functional(tf_func, np.array(10))
      self.assertEqual(jax_func(10).shape, (10,))
      self.assertEqual(jax_func(13).shape, (13,))
      with self.assertRaisesRegex(
          TypeError,
          "Shapes must be 1D sequences of concrete values of integer type"):
        jax.jit(jax_func)(13)
      self.assertEqual(jax.jit(jax_func, static_argnums=0)(13).shape, (13,))

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("with_custom_gradient", True), ("without_custom_gradient", False)),
          named=True,
      ))
  def test_custom_gradient(self, use_custom_gradient):
    @tf.function
    @tf.custom_gradient
    def tf_func(x):
      e = tf.exp(x)
      def grad(dy):
        # This is deliberately the wrong gradient.
        return dy * (1 - 1 / (1 + e)) * tf.sin(x) + 0.42
      return tf.reduce_sum(tf.math.log(1 + e)), grad

    np_inputs = np.array(range(6), dtype=np.float32).reshape(3, 2)
    tf_inputs = tf.constant(np_inputs)
    with tf.GradientTape() as tape:
      tape.watch(tf_inputs)
      tf_outputs = tf_func(tf_inputs)
      tf_grads = tape.gradient(tf_outputs, tf_inputs)

    with tf2jax.override_config("convert_custom_gradient", use_custom_gradient):
      jax_func = tf2jax.convert_functional(tf_func, np.zeros_like(np_inputs))
    jax_func = self.variant(jax_func)
    jax_outputs = jax_func(np_inputs)
    jax_grads = jax.grad(jax_func)(np_inputs)

    self.assertAllClose(tf_outputs, jax_outputs)
    if use_custom_gradient:
      self.assertAllClose(tf_grads, jax_grads)
    else:
      self.assertNotAllClose(tf_grads, jax_grads)

  @chex.variants(with_jit=True, without_jit=True)
  def test_trainable(self):
    can_train = tf.Variable(3.14, trainable=True, name="can_train")
    not_train = tf.Variable(42., trainable=False, name="not_train")

    @tf.function
    def tf_func(x):
      return x + can_train + not_train

    np_inputs = np.array(range(6), dtype=np.float32).reshape(3, 2)
    jax_func, jax_params = tf2jax.convert(tf_func, np.zeros_like(np_inputs))
    self.assertTrue(jax_params["can_train"].trainable)
    self.assertFalse(jax_params["not_train"].trainable)

    tf_outputs = tf_func(tf.constant(np_inputs))
    jax_outputs, _ = self.variant(jax_func)(jax_params, np_inputs)
    self.assertAllClose(tf_outputs, jax_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              ("positional", ((3.14, 42.0), {})),
              ("positional_and_keyword", ((3.14,), dict(y=42.0))),
              ("keyword", ((), dict(x=3.14, y=42.0))),
          ),
          named=True,
      ))
  def test_positional_and_keyword_args(self, all_args):
    @tf.function
    def tf_func(x, y):
      return x + y

    fn_args, fn_kwargs = tree.map_structure(np.array, all_args)
    tf_outputs = tf_func(*fn_args, **fn_kwargs)

    zero_args, zero_kwargs = tree.map_structure(np.zeros_like, all_args)
    jax_func = tf2jax.convert_functional(tf_func, *zero_args, **zero_kwargs)
    jax_outputs = self.variant(jax_func)(*fn_args, **fn_kwargs)

    self.assertAllClose(tf_outputs, jax_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              ("no_defaults", lambda *, kw0, kw1: kw0 + kw1),
              ("some_defaults", lambda *, kw0=3.14, kw1: kw0 + kw1),
              ("all_defaults", lambda *, kw0=3.14, kw1=42.0: kw0 + kw1),
              ("all_defaults_some_used",
               lambda *, kw0=3.14, kw1=42.0, kw2=2.8: kw0 + kw1 + kw2),
          ),
          named=True,
      ))
  def test_keyword_only(self, fn):
    tf_func = tf.function(fn)

    inputs = (np.array(3.5, dtype=np.float32), np.array(7.2, dtype=np.float32))
    tf_outputs = tf_func(kw0=inputs[0], kw1=inputs[1])

    jax_func = tf2jax.convert_functional(
        tf_func, kw0=np.zeros_like(inputs[0]), kw1=np.zeros_like(inputs[1]))
    jax_outputs = self.variant(jax_func)(kw0=inputs[0], kw1=inputs[1])

    self.assertAllClose(tf_outputs, jax_outputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_jit_nesting(self):
    @tf.function
    def tf_fn(x):
      return tf.cos(x)

    @jax.jit
    def tf2jax_fn(x):
      jax_fn = tf2jax.convert_functional(tf_fn, x)
      return jnp.sin(self.variant(jax_fn)(x))

    inputs = np.linspace(-1., 1., 6, dtype=np.float32).reshape((2, 3))
    self.assertAllClose(tf.sin(tf_fn(inputs)), tf2jax_fn(inputs))


if __name__ == "__main__":
  tf.test.main()
