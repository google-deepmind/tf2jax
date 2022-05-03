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
"""Tests for JAX -> TF -> JAX."""

from absl.testing import absltest
from absl.testing import parameterized

import chex
import haiku as hk
import jax
from jax.experimental import jax2tf
import jax.numpy as jnp

import numpy as np
import tensorflow as tf
from tf2jax._src import tf2jax
import tree

from tensorflow.compiler.xla import xla_data_pb2  # pytype: disable=import-error

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


def _compute_gradients(func, *inputs):
  def fn(*args):
    return jax.tree_leaves(func(*args))[0]
  return jax.grad(lambda *args: jnp.sum(fn(*args)))(*inputs)


class Jax2TfTest(parameterized.TestCase, tf.test.TestCase):

  def _test_convert(
      self,
      jax_func,
      inputs,
      with_grad,
      enable_xla,
      with_custom_grad=False,
      grad_tols=None,
  ):
    grad_tols = grad_tols or {}
    def assert_grad_all_close(*args):
      return self.assertAllClose(*args, **grad_tols)

    # Jax
    jax_func = self.variant(jax_func)
    jax_outputs = jax_func(*inputs)
    if with_grad:
      jax_grads = _compute_gradients(jax_func, *inputs)

    # Jax -> TF
    tf_func = jax2tf.convert(jax_func, with_gradient=with_grad,
                             enable_xla=enable_xla)
    tf_outputs = tf_func(*inputs)
    jax.tree_map(self.assertAllClose, jax_outputs, tf_outputs)

    # Jax -> TF -> Jax
    with tf2jax.override_config("convert_custom_gradient", with_custom_grad):
      rejax_func = tf2jax.convert_functional(
          tf.function(tf_func), *tree.map_structure(np.zeros_like, inputs))
    rejax_func = self.variant(rejax_func)
    rejax_outputs = rejax_func(*inputs)
    jax.tree_map(self.assertAllClose, rejax_outputs, tf_outputs)
    if with_grad:
      rejax_grads = _compute_gradients(rejax_func, *inputs)
      jax.tree_map(assert_grad_all_close, jax_grads, rejax_grads)

    # Jax -> TF -> SavedModel -> TF
    model = tf.Module()
    model.f = tf.function(tf_func)
    model.f(*tree.map_structure(tf.zeros_like, inputs))  # Dummy call.
    tmp_dir = self.create_tempdir()
    tf.saved_model.save(model, tmp_dir.full_path)
    del model
    restored = tf.saved_model.load(tmp_dir.full_path)
    restored_tf_outputs = restored.f(*inputs)
    jax.tree_map(self.assertAllClose, jax_outputs, restored_tf_outputs)

    # Jax -> TF -> SavedModel -> TF -> Jax
    with tf2jax.override_config("convert_custom_gradient", with_custom_grad):
      rejax_too_func = tf2jax.convert_functional(
          restored.f, *tree.map_structure(np.zeros_like, inputs))
    rejax_too_func = self.variant(rejax_too_func)
    rejax_too_outputs = rejax_too_func(*inputs)
    jax.tree_map(self.assertAllClose, rejax_too_outputs, tf_outputs)
    if with_grad:
      rejax_too_grads = _compute_gradients(rejax_too_func, *inputs)
      jax.tree_map(assert_grad_all_close, jax_grads, rejax_too_grads)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          (("without_custom_gradient", False), ("with_custom_gradient", True)),
          named=True,
      ))
  def test_simple(self, with_grad, enable_xla, with_custom_grad):
    np.random.seed(42)

    def forward(x):
      return jnp.sin(jnp.cos(x))

    inputs = np.random.normal((3, 2))
    self._test_convert(
        forward, [inputs],
        with_grad=with_grad,
        enable_xla=enable_xla,
        with_custom_grad=with_custom_grad)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          (("without_custom_gradient", False), ("with_custom_gradient", True)),
          named=True,
      ))
  def test_mlp(self, with_grad, enable_xla, with_custom_grad):
    np.random.seed(42)

    def forward(x):
      mlp = hk.nets.MLP([300, 100, 10])
      return mlp(x)

    inputs = np.random.normal(size=(8, 28 * 28))
    forward = hk.transform(forward)
    variables = forward.init(jax.random.PRNGKey(42), jnp.zeros_like(inputs))
    variables = hk.data_structures.to_mutable_dict(variables)
    jax_fn = hk.without_apply_rng(forward).apply
    self._test_convert(
        jax_fn, [variables, inputs],
        with_grad=with_grad,
        enable_xla=enable_xla,
        with_custom_grad=with_custom_grad)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          (("without_custom_gradient", False), ("with_custom_gradient", True)),
          named=True,
      ))
  def test_batch_norm(self, with_grad, enable_xla, with_custom_grad):
    np.random.seed(42)

    def forward(x):
      batch_norm = hk.BatchNorm(True, True, 0.9, eps=0.1)
      return batch_norm(x, is_training=True)

    inputs = np.random.normal(size=(8, 17))
    forward = hk.transform_with_state(forward)
    variables, states = forward.init(
        jax.random.PRNGKey(42), jnp.zeros_like(inputs))
    variables = hk.data_structures.to_mutable_dict(variables)
    states = hk.data_structures.to_mutable_dict(states)
    def jax_fn(params, states, x):
      outputs, states = hk.without_apply_rng(forward).apply(params, states, x)
      return outputs, hk.data_structures.to_mutable_dict(states)

    # Perturb variables and states.
    variables = tree.map_structure(
        lambda x: x + np.random.uniform(size=x.shape), variables)
    states = tree.map_structure(
        lambda x: x + np.random.normal(size=x.shape), states)
    self._test_convert(
        jax_fn, [variables, states, inputs],
        with_grad=with_grad,
        enable_xla=enable_xla,
        with_custom_grad=with_custom_grad)

  # Conv2D uses jax.lax.conv_general_dilated which is translated to XlaConv.
  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_conv2d(self, with_grad, enable_xla):
    np.random.seed(42)

    tols = dict(rtol=1e-5) if jax.default_backend().lower() == "gpu" else {}

    def forward(x):
      conv = hk.Conv2D(
          output_channels=7, kernel_shape=3, stride=1, padding="SAME")
      return conv(x)

    inputs = np.random.normal(size=(8, 28, 28, 3))
    forward = hk.transform(forward)
    variables = forward.init(jax.random.PRNGKey(42), jnp.zeros_like(inputs))
    variables = hk.data_structures.to_mutable_dict(variables)
    jax_fn = hk.without_apply_rng(forward).apply
    self._test_convert(
        jax_fn, [variables, inputs],
        with_grad=with_grad,
        enable_xla=enable_xla,
        grad_tols=tols)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          (
              ("default_group_counts", dict()),
              ("feature_group_count", dict(feature_group_count=3)),
              ("batch_group_count", dict(batch_group_count=2)),
          ),
          named=True,
      ))
  def test_xla_conv(self, with_grad, enable_xla, group_counts):
    np.random.seed(42)

    kernels = np.random.normal(size=(3, 3, 3, 12))
    dimension_numbers = jax.lax.ConvDimensionNumbers(
        lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2))

    def forward(x):
      return jax.lax.conv_general_dilated(
          x,
          rhs=kernels,
          window_strides=(1, 1),
          padding="SAME",
          lhs_dilation=(1, 1),
          rhs_dilation=(1, 1),
          dimension_numbers=dimension_numbers,
          **group_counts)

    feature_dim = 3 * group_counts.get("feature_group_count", 1)
    inputs = np.random.normal(size=(8, 28, 28, feature_dim))
    forward = hk.transform(forward)
    variables = forward.init(jax.random.PRNGKey(42), jnp.zeros_like(inputs))
    variables = hk.data_structures.to_mutable_dict(variables)
    jax_fn = hk.without_apply_rng(forward).apply
    self._test_convert(
        jax_fn, [variables, inputs], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          named=True,
      ))
  def test_dot(self, with_grad, enable_xla):
    def forward(lhs, rhs):
      return jax.lax.dot(lhs, rhs)

    inputs = (
        np.linspace(0, 1, 10 * 5).reshape(10, 5),
        np.linspace(-1, 0, 5 * 3).reshape(5, 3),
    )
    self._test_convert(
        forward, inputs, with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          named=True,
      ))
  def test_dot_general(self, with_grad, enable_xla):
    dimension_numbers = (((2,), (1,)), ((0,), (0,)))

    def forward(lhs, rhs):
      return jax.lax.dot_general(lhs, rhs, dimension_numbers)

    inputs = (
        np.linspace(0, 1, 2 * 10 * 5).reshape((2, 10, 5)),
        np.linspace(-1, 0, 2 * 5 * 3).reshape((2, 5, 3)),
    )
    self._test_convert(
        forward, inputs, with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_dynamic_slice(self, with_grad, enable_xla):
    def forward(x):
      return jax.lax.dynamic_slice(x, (1, 1), (2, 3))

    inputs = np.linspace(0, 1, 12).reshape(3, 4)
    self._test_convert(
        forward, [inputs], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_dynamic_update_slice(self, with_grad, enable_xla):
    def forward(x, y):
      return jax.lax.dynamic_update_slice(x, y, (1, 2))

    inputs = [
        np.linspace(0, 1, 12).reshape(3, 4),
        1.0 - np.linspace(0, 1, 12).reshape(3, 4),
    ]
    self._test_convert(
        forward, inputs, with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          named=True,
      ))
  def test_gather(self, with_grad, enable_xla):

    dimension_numbers = jax.lax.GatherDimensionNumbers((1,), (0,), (0, 1))
    slice_sizes = (1, 3)

    def forward(operand, indices):
      return jax.lax.gather(operand, indices, dimension_numbers, slice_sizes)

    inputs = (
        np.linspace(0, 1, 10 * 5).reshape(10, 5),
        np.array([[4, 2], [3, 2]]),
    )
    self._test_convert(
        forward, inputs, with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          named=True,
      ))
  def test_pad(self, with_grad, enable_xla):
    padding_config = [(1, 2, 1), (0, 1, 0)]

    def forward(operand, padding_value):
      return jax.lax.pad(operand, padding_value, padding_config)

    inputs = (
        np.linspace(0, 1, 2 * 3).reshape(2, 3),
        np.array(0.42),
    )
    self._test_convert(
        forward, inputs, with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          (
              ("min", jax.lax.min, jnp.inf),
              ("max", jax.lax.max, -jnp.inf),
              ("add", jax.lax.add, 0.0),
          ),
          named=True,
      ))
  def test_reduce(self, with_grad, enable_xla, reduce_fn, init_value):

    def forward(x):
      dimensions = [1, 2]
      return jax.lax.reduce(x, init_value, reduce_fn, dimensions)

    inputs = np.linspace(0, 1, 2 * 5 * 5 * 3).reshape((2, 5, 5, 3))
    self._test_convert(
        forward, [inputs], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          named=True,
      ))
  def test_reduce_variadic(self, with_grad, enable_xla):

    def forward(args):
      return jax.lax.reduce(args, (0.0, 1.0), lambda xs, ys: xs, [1, 2])

    inputs = (
        np.linspace(0, 1, 2 * 5 * 5 * 3).reshape((2, 5, 5, 3)),
        np.linspace(2, 3, 2 * 5 * 5 * 3).reshape((2, 5, 5, 3)),
    )
    self._test_convert(
        forward, [inputs], with_grad=with_grad, enable_xla=enable_xla)

  # jax.lax.reduce_window is translated to XlaReduceWindow.
  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          (
              ("min", jax.lax.min, jnp.inf),
              ("max", jax.lax.max, -jnp.inf),
              ("add", jax.lax.add, 0.0),
          ),
          named=True,
      ))
  def test_reduce_window(self, with_grad, enable_xla, reduce_fn, init_value):
    np.random.seed(42)

    def forward(x):
      window_shape = [1, 2, 2, 1]
      return jax.lax.reduce_window(x, init_value, reduce_fn, window_shape,
                                   window_shape, "SAME")

    inputs = np.random.normal(size=(8, 28, 28, 3))
    self._test_convert(
        forward, [inputs], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False),),
          (("enable_xla", True),),
          (("uint32", np.uint32),),
          (
              ("default", xla_data_pb2.RandomAlgorithm.RNG_DEFAULT),
              ("threefry", xla_data_pb2.RandomAlgorithm.RNG_THREE_FRY),
              ("philox", xla_data_pb2.RandomAlgorithm.RNG_PHILOX),
          ),
          named=True,
      ))
  def test_rng_bit_generator(self, with_grad, enable_xla, dtype, algorithm):
    def forward(key):
      return jax.lax.rng_bit_generator(
          key, shape=(10, 5), dtype=dtype, algorithm=algorithm)

    if dtype == np.uint32:
      key = np.array([6, 7, 8, 9], dtype=np.uint32)
    else:
      raise ValueError(f"Unsupported dtype={dtype}")

    self._test_convert(
        forward, [key], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              ("scatter", jax.lax.scatter),
              ("scatter_add", jax.lax.scatter_add),
              ("scatter_mul", jax.lax.scatter_mul),
              ("scatter_min", jax.lax.scatter_min),
              ("scatter_max", jax.lax.scatter_max),
          ),
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          (("unique_indices", True), ("non_unique_indices", False)),
          named=True,
      ))
  def test_scatter(self, scatter_fn, with_grad, enable_xla, unique_indices):
    if scatter_fn is jax.lax.scatter_mul and with_grad and not unique_indices:
      self.skipTest("Gradient is disallowed for jax.lax.scatter_mul if "
                    "unique_indices=False")

    dimension_numbers = jax.lax.ScatterDimensionNumbers((1,), (0,), (0,))

    def forward(operand, indices, updates):
      return scatter_fn(
          operand,
          indices,
          updates,
          dimension_numbers,
          unique_indices=unique_indices)

    inputs = (
        np.linspace(0, 1, 10*5).reshape(10, 5),
        np.array([[1], [8], [4]]),
        np.linspace(0, 9, 9).reshape(3, 3),
    )
    self._test_convert(
        forward, inputs, with_grad=with_grad, enable_xla=enable_xla)

  # Derivative of jax.lax.reduce_window uses XlaSelectAndScatter.
  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False),),
          (("enable_xla", True),),
          (
              ("min", jax.lax.min, jnp.inf),
              ("max", jax.lax.max, -jnp.inf),
              ("add", jax.lax.add, 0.0),
          ),
          named=True,
      ))
  def test_select_and_scatter(self, with_grad, enable_xla, reduce_fn,
                              init_value):
    np.random.seed(42)

    def forward(x):
      window_shape = [1, 2, 2, 1]
      return jax.lax.reduce_window(x, init_value, reduce_fn, window_shape,
                                   window_shape, "SAME")

    inputs = np.random.normal(size=(8, 5, 5, 3))
    jax_fn = jax.jacrev(forward)
    self._test_convert(
        jax_fn, [inputs], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("enable_xla", True),),
          (("2nd_last_dim", -2), ("last_dim", -1)),
          (("not_stable", False), ("is_stable", True)),
          (("one_keys", 1), ("two_keys", 2), ("three_keys", 3)),
          named=True,
      ))
  def test_sort_variadic(self, with_grad, enable_xla, dim, is_stable, num_keys):

    def forward(args):
      return jax.lax.sort(
          args, dimension=dim, is_stable=is_stable, num_keys=num_keys)

    inputs = (
        np.array([[6., 2.], [4., 2.], [4., 1.]], np.float32),
        np.array([[1., 2.], [3., 4.], [5., 6.]], np.float32),
        np.array([[6., 5.], [4., 3.], [2., 1.]], np.float32),
    )
    self._test_convert(
        forward, [inputs], with_grad=with_grad, enable_xla=enable_xla)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_polymorphic_shape(self, with_grad, enable_xla):
    inputs = np.array(range(36), dtype=np.float32).reshape(9, 4)

    # TF
    @tf.function
    def forward(x):
      shape = tf.shape(x)
      # Manipulate polymoprhic shape.
      new_shape = tf.stack([tf.pow(shape[0], 1), shape[1]], axis=0)
      outputs = tf.broadcast_to(tf.sin(x), new_shape)
      outputs = tf.concat([outputs] * 2, axis=0)  # Stack along unknown dim
      outputs = tf.concat([outputs] * 2, axis=1)  # Stack along knonwn dim
      return outputs / tf.cast(shape[0], tf.float32)  # Divide by unknown dim
    tf_outputs = forward(inputs)

    # TF -> JAX
    jax_func = tf2jax.convert_functional(
        tf.function(forward), tf.TensorSpec((None, 4)))
    jax_func = self.variant(jax_func)
    jax_outputs = jax_func(inputs)
    self.assertAllClose(tf_outputs, jax_outputs)

    # TF -> JAX -> TF
    new_tf_forward = jax2tf.convert(
        jax_func,
        polymorphic_shapes=["(b, _)"],
        with_gradient=with_grad,
        enable_xla=enable_xla)
    new_tf_forward = tf.function(new_tf_forward)
    concrete_new_tf_forward = new_tf_forward.get_concrete_function(
        tf.TensorSpec(shape=(None, 4)))
    self.assertEqual(concrete_new_tf_forward.structured_outputs.shape.as_list(),
                     [None, 8])
    new_tf_outputs = concrete_new_tf_forward(inputs)
    self.assertAllClose(new_tf_outputs, jax_outputs)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("with_gradient", True),),
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_custom_gradient(self, with_grad, enable_xla):
    inputs = np.array(range(6), dtype=np.float32).reshape(3, 2)

    # JAX
    @jax.custom_gradient
    def forward(x):
      e = jnp.exp(x)
      def grad(dy):
        # This is deliberately the wrong gradient.
        return dy * (1 - 1 / (1 + e)) * jnp.sin(x) + 0.42
      return jnp.sum(jnp.log(1 + e)), grad
    forward = self.variant(forward)
    expected_outputs = forward(inputs)
    expected_grads = jax.grad(forward)(inputs)

    # JAX -> TF
    tf_forward = jax2tf.convert(
        forward, with_gradient=with_grad, enable_xla=enable_xla)
    tf_forward = tf.function(tf_forward)

    # JAX -> TF -> JAX
    with tf2jax.override_config("convert_custom_gradient", True):
      jax_forward = tf2jax.convert_functional(tf_forward,
                                              tf.zeros_like(inputs))
    jax_forward = self.variant(jax_forward)
    jax_outputs = jax_forward(inputs)
    jax_grads = jax.grad(jax_forward)(inputs)
    self.assertAllClose(expected_outputs, jax_outputs)
    self.assertAllClose(expected_grads, jax_grads)

    # Jax -> TF -> SavedModel -> TF
    model = tf.Module()
    model.f = tf_forward
    tmp_dir = self.create_tempdir()
    tf.saved_model.save(model, tmp_dir.full_path)
    del model
    restored = tf.saved_model.load(tmp_dir.full_path)

    # Jax -> TF -> SavedModel -> TF -> Jax
    with tf2jax.override_config("convert_custom_gradient", True):
      re_jax_forward = tf2jax.convert_functional(restored.f,
                                                 tf.zeros_like(inputs))
    re_jax_forward = self.variant(re_jax_forward)
    re_jax_outputs = re_jax_forward(inputs)
    re_jax_grads = jax.grad(re_jax_forward)(inputs)
    self.assertAllClose(expected_outputs, re_jax_outputs)
    self.assertAllClose(expected_grads, re_jax_grads)

  @chex.variants(with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("with_gradient", True),),
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_custom_gradient_nested(self, with_grad, enable_xla):
    inputs = np.array(range(6), dtype=np.float32).reshape(3, 2)

    # JAX
    @jax.custom_gradient
    def forward(x):
      e = jnp.exp(x)
      def grad(dy):
        # This is deliberately the wrong gradient.
        return dy * (1 - 1 / (1 + e)) * jnp.sin(x) + 0.42
      return jnp.sum(jnp.log(1 + e)), grad

    forward = self.variant(forward)
    expected_outputs = forward(inputs)
    expected_grads = jax.grad(forward)(inputs)

    # JAX -> TF
    tf_fn = jax2tf.convert(
        forward, with_gradient=with_grad, enable_xla=enable_xla)
    tf_fn = tf.function(tf_fn)

    # JAX -> TF -> CALL_TF -> TF.
    # This creates dependencies between custom gradients.
    call_tf_fn = jax2tf.call_tf(tf_fn)
    tf_fn_too = jax2tf.convert(call_tf_fn, with_gradient=with_grad)
    tf_fn_too = tf.function(tf_fn_too)

    # JAX -> TF -> CALL_TF -> TF -> JAX
    with tf2jax.override_config("convert_custom_gradient", True):
      jax_fn_too = tf2jax.convert_functional(tf_fn_too, np.zeros_like(inputs))

    jax_outputs = jax_fn_too(inputs)
    jax_grads = jax.grad(jax_fn_too)(inputs)
    self.assertAllClose(expected_outputs, jax_outputs)
    self.assertAllClose(expected_grads, jax_grads)

  @chex.variants(without_jit=True, with_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_gradient", False), ("with_gradient", True)),
          (("disable_xla", False), ("enable_xla", True)),
          (("without_custom_gradient", False), ("with_custom_gradient", True)),
          named=True,
      ))
  def test_relu(self, with_grad, enable_xla, with_custom_grad):
    inputs = np.array([-1.0, 0.0, 1.0], np.float32)
    self._test_convert(
        jax.nn.relu, [inputs],
        with_grad=with_grad,
        enable_xla=enable_xla,
        with_custom_grad=with_custom_grad)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("disable_xla", False), ("enable_xla", True)),
          named=True,
      ))
  def test_empty_return(self, enable_xla):
    np.random.seed(42)

    def forward(x):
      unused_y = jnp.cos(x)
      return ()

    # JAX
    forward = self.variant(forward)
    inputs = np.random.normal((3, 2))
    with self.assertRaisesRegex(
        TypeError,
        "Gradient only defined for scalar-output functions. Output was ()."):
      jax.grad(forward)(inputs)

    # JAX -> TF
    tf_forward = jax2tf.convert(
        forward, with_gradient=True, enable_xla=enable_xla)
    tf_forward = tf.function(tf_forward)

    # JAX -> TF -> JAX
    with tf2jax.override_config("convert_custom_gradient", True):
      jax_forward = tf2jax.convert_functional(tf_forward,
                                              tf.zeros_like(inputs))
      jax_forward = self.variant(jax_forward)

    with self.assertRaisesRegex(
        TypeError,
        "Gradient only defined for scalar-output functions. Output was ()."):
      jax.grad(jax_forward)(inputs)

    # Jax -> TF -> SavedModel -> TF
    model = tf.Module()
    model.f = tf_forward
    tmp_dir = self.create_tempdir()
    tf.saved_model.save(model, tmp_dir.full_path)
    del model
    restored = tf.saved_model.load(tmp_dir.full_path)

    # Jax -> TF -> SavedModel -> TF -> Jax
    with tf2jax.override_config("convert_custom_gradient", True):
      re_jax_forward = tf2jax.convert_functional(restored.f,
                                                 tf.zeros_like(inputs))
      re_jax_forward = self.variant(re_jax_forward)

    with self.assertRaisesRegex(
        TypeError,
        "Gradient only defined for scalar-output functions. Output was ()."):
      jax.grad(re_jax_forward)(inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_jax2tf_nesting(self):
    @tf.function
    def tf_fn(x):
      return tf.cos(x)

    def tf2jax_fn(x):
      jax_fn = tf2jax.convert_functional(tf_fn, x)
      return jnp.sin(self.variant(jax_fn)(x))

    tf2jax2tf_fn = jax2tf.convert(tf2jax_fn)

    inputs = np.linspace(-1., 1., 6, dtype=np.float32).reshape((2, 3))
    self.assertAllClose(tf.sin(tf_fn(inputs)), tf2jax2tf_fn(inputs))


if __name__ == "__main__":
  absltest.main()
