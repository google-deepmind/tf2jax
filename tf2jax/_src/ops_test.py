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
"""Tests for tf2jax."""

import contextlib

from absl.testing import parameterized

import chex
import jax
import numpy as np

import tensorflow as tf
from tf2jax._src import tf2jax
import tree


def _reorder(vals, inds):
  return [vals[idx] for idx in inds]


@contextlib.contextmanager
def _nullcontext(enter_result=None):
  yield enter_result


class OpsTest(tf.test.TestCase, parameterized.TestCase):

  def _assert_if_jitted(self, err):
    jitted = self.variant.type == chex.ChexVariantType.WITH_JIT
    return self.assertRaises(err) if jitted else _nullcontext()

  def _test_convert(self,
                    tf_func,
                    inputs,
                    check_shape_only=False,
                    functional=True):
    if not isinstance(inputs, (list, tuple)):
      inputs = (inputs,)

    jax_func, jax_params = tf2jax.convert(
        tf.function(tf_func), *tree.map_structure(np.zeros_like, inputs))
    if functional:
      self.assertEmpty(jax_params, "Expected no parameters for pure Ops.")

    jax_func = self.variant(jax_func)

    rng = jax.random.PRNGKey(42)
    jax_results, new_jax_params = jax_func(jax_params, *inputs, rng=rng)
    tf_results = tf_func(*inputs)

    # Check outputs
    for tf_res, jax_res in zip(
        tree.flatten(tf_results), tree.flatten(jax_results)):
      self.assertEqual(tf_res.shape, jax_res.shape)
      if not check_shape_only:
        self.assertAllClose(np.asarray(tf_res), jax_res, atol=1e-5)

    return jax_results, new_jax_params

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("log_softmax", "sigmoid", "softmax", "softplus",
                            "tanh", "relu", "relu6", "elu")
  def test_activations(self, op_name):
    np.random.seed(42)
    inputs = [np.random.normal(size=(10, 5)).astype(np.float32)]
    self._test_convert(getattr(tf.nn, op_name), inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_assert(self):

    def assert_fn(cond, data):
      tf.Assert(condition=cond, data=data)
      return data

    self._test_convert(assert_fn, [np.array(5) > 0, [np.array(6)]])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("Abs", "Acosh", "Asinh", "Atanh", "Ceil", "Cos",
                            "Cosh", "Digamma", "Erf", "Erfc", "Erfinv", "Exp",
                            "Expm1", "Floor", "Lgamma", "Log", "Log1p", "Neg",
                            "Reciprocal", "Round", "Rsqrt", "Sign", "Sin",
                            "Sinh", "Sqrt", "Square", "Tan")
  def test_unary_numerics(self, op_name):
    np.random.seed(42)
    if op_name == "Erfinv":
      inputs = np.random.uniform(size=(10, 5)).astype(np.float32)
    else:
      inputs = np.random.normal(size=(10, 5)).astype(np.float32)

    def tf_func(x):
      return getattr(tf.raw_ops, op_name)(x=x)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("Atan2", "Atan2")
  def test_binary_numerics(self, op_name):
    np.random.seed(42)
    inputs = (
        np.random.normal(size=(10, 5)).astype(np.float32),
        np.random.normal(size=(10, 5)).astype(np.float32),
    )

    def tf_func(x, y):
      return getattr(tf.raw_ops, op_name)(x=x, y=y)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("Add", "AddV2", "Div", "FloorDiv", "FloorMod",
                            "Mul", "Pow", "RealDiv", "Sub")
  def test_binary_numerics_with_static(self, op_name):
    np.random.seed(42)
    inputs = (
        np.random.normal(size=(10, 5)).astype(np.float32),
        np.random.normal(size=(10, 5)).astype(np.float32),
    )

    def tf_func(x, y):
      return getattr(tf.raw_ops, op_name)(x=x, y=y)
    self._test_convert(tf_func, inputs)

    # Check static inputs result in static outputs.
    def tf_static():
      vals = tf_func(x=np.array([4.0, 3.0]), y=np.array([1.0, 2.0]))
      return tf.zeros(tf.cast(vals, tf.int32))
    self._test_convert(tf_static, [])

  @chex.variants(with_jit=False, without_jit=True)
  def test_add_n(self):
    np.random.seed(42)
    inputs = [
        [np.random.normal(size=(10, 5)).astype(np.float32) for _ in range(4)]
    ]

    def tf_func(x):
      return tf.raw_ops.AddN(inputs=x)
    self._test_convert(tf_func, inputs)

    # Check static inputs result in static outputs.
    def tf_static():
      vals = tf_func(
          [np.array([4.0, 3.0]), np.array([1.0, 2.0]), np.array([2.0, 1.0])])
      return tf.zeros(tf.cast(vals, tf.int32))
    self._test_convert(tf_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("BitwiseAnd", "BitwiseOr", "BitwiseXor", "Invert")
  def test_bitwise(self, op_name):
    np.random.seed(42)
    inputs = (np.random.randint(1000000, size=(10, 5), dtype=np.int32),
              np.random.randint(1000000, size=(10, 5), dtype=np.int32))

    def tf_func(x, y):
      kwargs = dict(x=x) if op_name == "Invert" else dict(x=x, y=y)
      return getattr(tf.raw_ops, op_name)(**kwargs)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("All", "Any")
  def test_logical_reduction(self, op_name):
    inputs = np.array([[[False, False], [False, True], [True, True]]])

    def tf_func(x):
      return getattr(tf.raw_ops, op_name)(input=x, axis=(0, 2), keep_dims=True)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("LogicalAnd", "LogicalNot", "LogicalOr")
  def test_logical(self, op_name):
    np.random.seed(42)
    inputs = (
        np.random.normal(size=(10, 5)) > 0.,
        np.random.normal(size=(10, 5)) > 0.,
    )

    def tf_func(x, y):
      kwargs = dict(x=x) if op_name == "LogicalNot" else dict(x=x, y=y)
      return getattr(tf.raw_ops, op_name)(**kwargs)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("LeftShift", "RightShift")
  def test_shift(self, op_name):
    np.random.seed(42)
    inputs = (np.random.randint(1000000, size=(3, 2), dtype=np.int32),
              np.random.randint(32, size=(3, 2), dtype=np.int32))

    def tf_func(x, y):
      return getattr(tf.raw_ops, op_name)(x=x, y=y)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("Equal", "Greater", "GreaterEqual", "Less",
                            "LessEqual", "NotEqual")
  def test_binary_comparison(self, op_name):
    np.random.seed(42)
    inputs = (np.random.randint(10, size=(10, 5)),
              np.random.randint(10, size=(10, 5)))

    def tf_func(x, y):
      return tf.cast(getattr(tf.raw_ops, op_name)(x=x, y=y), tf.int32)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
      ("Add", [[4, 3], [1, 2]]),
      ("AddV2", [[4, 3], [1, 2]]),
      ("Div", [[9, 6], [3, 3]]),
      ("Mul", [[4, 3], [1, 2]]),
      ("Neg", [[-4, -3]]),
      ("Sub", [[4, 3], [1, 2]]),
      ("Equal", [[4, 3, 2], [2, 3, 4]]),
      ("Greater", [[4, 3, 2], [2, 3, 4]]),
      ("GreaterEqual", [[4, 3, 2], [2, 3, 4]]),
      ("Less", [[4, 3, 2], [2, 3, 4]]),
      ("LessEqual", [[4, 3, 2], [2, 3, 4]]),
      ("NotEqual", [[4, 3, 2], [2, 3, 4]]),
  )
  def test_jitting_static(self, op_name, inputs):
    def tf_func():
      if len(inputs) == 1:
        shape = getattr(tf.raw_ops, op_name)(x=inputs[0])
      else:
        shape = getattr(tf.raw_ops, op_name)(x=inputs[0], y=inputs[1])
      return tf.zeros(tf.cast(shape, tf.int32) + 1)

    self._test_convert(tf_func, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_argmax(self):
    inputs = np.array(np.reshape(range(60), (5, 4, 3)), dtype=np.int32)

    def argmax_fn(xs):
      return tf.raw_ops.ArgMax(input=xs, dimension=1)
    self._test_convert(argmax_fn, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_argmin(self):
    inputs = np.array(np.reshape(range(60), (5, 4, 3)), dtype=np.int32)

    def argmin_fn(xs):
      return tf.raw_ops.ArgMin(input=xs, dimension=1)
    self._test_convert(argmin_fn, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("SAME", "SAME"), ("VALID", "VALID")),
          (("NHWC", "NHWC"), ("NCHW", "NCHW")),
          named=True,
      ))
  def test_avg_pool(self, padding, data_format):
    np.random.seed(42)

    inputs = np.random.normal(size=(10, 32, 16, 8)).astype(np.float32)
    ksize = (1, 2, 3, 1)
    strides = (1, 3, 2, 1)

    if data_format == "NCHW":
      if jax.default_backend().lower() == "cpu":
        self.skipTest("TensorFlow AvgPool does not support NCHW on CPU.")
      inputs = np.transpose(inputs, [0, 3, 1, 2])
      ksize = _reorder(ksize, [0, 3, 1, 2])
      strides = _reorder(strides, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def pool(x):
      return tf.raw_ops.AvgPool(
          value=x,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format)
    self._test_convert(pool, [inputs])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("NHWC", "NHWC"), ("NCHW", "NCHW")),
          named=True,
      ))
  def test_bias_add(self, data_format):
    np.random.seed(42)

    inputs = np.random.normal(size=(10, 32, 16, 8)).astype(np.float32)
    bias = np.linspace(-1., 1., 8).astype(np.float32)

    if data_format == "NCHW":
      inputs = np.transpose(inputs, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def pool(x, b):
      return tf.raw_ops.BiasAdd(value=x, bias=b, data_format=data_format)

    self._test_convert(pool, [inputs, bias])

  @chex.variants(with_jit=True, without_jit=True)
  def test_bitcast(self):
    inputs = np.array(0xffffffff, dtype=np.uint32)

    def tf_func(x):
      return tf.bitcast(x, type=tf.float32)
    self._test_convert(tf_func, inputs)

    def raw_func(x):
      return tf.raw_ops.Bitcast(input=x, type=tf.float32)
    self._test_convert(raw_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_broadcast_to(self):
    inputs, shape = np.array([1, 2, 3]), (3, 3)

    def broadcast_to(xs):
      return tf.raw_ops.BroadcastTo(input=xs, shape=shape)
    self._test_convert(broadcast_to, inputs)

    # Check static inputs result in static outputs.
    def broadcast_to_static():
      return tf.zeros(broadcast_to(inputs)[0])
    self._test_convert(broadcast_to_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_complex(self):
    reals = np.array([1.2, -2.3, 3.4], np.float32)
    imags = np.array([-4.5, 5.6, -6.7], np.float32)

    self._test_convert(lambda x, y: tf.raw_ops.Complex(real=x, imag=y),
                       [reals, imags])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("ComplexAbs", "Conj", "Imag", "Real")
  def test_complex_ops(self, op_name):
    reals = np.array([1.2, -2.3, 3.4], np.float32)
    imags = np.array([-4.5, 5.6, -6.7], np.float32)

    def forward(x):
      if op_name == "ComplexAbs":
        return getattr(tf.raw_ops, op_name)(x=x)
      else:
        return getattr(tf.raw_ops, op_name)(input=x)

    self._test_convert(forward, [reals + 1j * imags])

  @chex.variants(with_jit=True, without_jit=True)
  def test_concat(self):
    inputs = [np.zeros((10, 5)), np.zeros((7, 5))]

    def concat(xs):
      return tf.raw_ops.ConcatV2(values=xs, axis=0)
    self._test_convert(concat, [inputs])

    # Check static inputs result in static outputs.
    def concat_static():
      return tf.zeros(concat([[10], [5]]))
    self._test_convert(concat_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_conjugate_transpose(self):
    reals = np.array(range(24), np.float32).reshape((2, 3, 4)) - 6
    imags = np.array(range(24), np.float32).reshape((2, 3, 4)) - 18

    self._test_convert(
        lambda x: tf.raw_ops.ConjugateTranspose(x=x, perm=[2, 0, 1]),
        [reals + 1j * imags])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("one", [3, 5, 5, 7], [3, 3, 7, 11], [1, 2, 2, 1], "SAME", "NHWC"),
      ("two", [2, 1, 16, 24], [1, 8, 3, 48], [1, 1, 1, 1], "VALID", "NHWC"),
      ("three", [3, 5, 5, 7], [3, 3, 7, 11], [1, 2, 2, 1], "SAME", "NCHW"),
      ("four", [2, 1, 16, 24], [1, 8, 3, 48], [1, 1, 1, 1], "VALID", "NCHW"),
  )
  def test_conv2d(self, input_shape, filter_shape, strides, padding,
                  data_format):
    np.random.seed(42)

    dilations = [1, 1, 1, 1]
    filters = np.random.normal(size=filter_shape).astype(np.float32)
    inputs = np.random.normal(size=input_shape).astype(np.float32)
    if data_format == "NCHW":
      if jax.default_backend().lower() == "cpu":
        self.skipTest("TensorFlow Conv2D does not support NCHW on CPU.")
      inputs = np.transpose(inputs, [0, 3, 1, 2])
      strides = _reorder(strides, [0, 3, 1, 2])
      dilations = _reorder(dilations, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def tf_func(x):
      return tf.nn.conv2d(
          x,
          filters=filters,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
    self._test_convert(tf_func, inputs)

    def raw_func(x):
      return tf.raw_ops.Conv2D(
          input=x,
          filter=filters,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
    self._test_convert(raw_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("NHWC", "NHWC"), ("NCHW", "NCHW"),),
          named=True,
      ))
  def test_conv2d_transpose(self, data_format):
    np.random.seed(42)

    output_shape = [3, 8, 8, 128]
    padding = "SAME"
    strides = [1, 2, 2, 1]
    dilations = [1, 1, 1, 1]
    filters = np.random.normal(size=[7, 7, 128, 13]).astype(np.float32)
    inputs = np.random.normal(size=[3, 4, 4, 13]).astype(np.float32)
    if data_format == "NCHW":
      if jax.default_backend().lower() == "cpu":
        self.skipTest(
            "TensorFlow Conv2DBackpropInput does not support NCHW on CPU.")
      inputs = np.transpose(inputs, [0, 3, 1, 2])
      strides = _reorder(strides, [0, 3, 1, 2])
      dilations = _reorder(dilations, [0, 3, 1, 2])
      output_shape = _reorder(output_shape, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def tf_func(x):
      return tf.nn.conv2d_transpose(
          x,
          filters=filters,
          output_shape=output_shape,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
    self._test_convert(tf_func, inputs)

    def raw_func(x):
      return tf.raw_ops.Conv2DBackpropInput(
          input_sizes=output_shape,
          filter=filters,
          out_backprop=x,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations)
    self._test_convert(raw_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("exclusive", True), ("not_exclusive", False),),
          (("reverse", True), ("forward", False)),
          named=True,
      ))
  def test_cumsum(self, exclusive, reverse):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.int32)

    def cumsum_fn(xs):
      return tf.raw_ops.Cumsum(
          x=xs, axis=1, exclusive=exclusive, reverse=reverse)
    self._test_convert(cumsum_fn, inputs)

    # Check static inputs result in static outputs.
    def cumsum_static():
      return tf.zeros(cumsum_fn(inputs)[0, -1])
    self._test_convert(cumsum_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("without_explicit_paddings", False),
           ("with_explicit_paddings", True)),
          (("NHWC", "NHWC"), ("NCHW", "NCHW"),),
          named=True,
      ))
  def test_depthwise_conv2d(self, use_explicit_paddings, data_format):
    np.random.seed(42)

    strides = [1, 2, 2, 1]
    filters = np.random.normal(size=[3, 3, 7, 11]).astype(np.float32)
    inputs = np.random.normal(size=[3, 5, 5, 7]).astype(np.float32)
    explicit_paddings = ([[0, 0], [8, 8], [8, 8], [0, 0]]
                         if use_explicit_paddings else [[]])
    dilations = [1, 1, 1, 1]

    if data_format == "NCHW":
      if jax.default_backend().lower() == "cpu":
        self.skipTest(
            "TensorFlow DepthwiseConv2dNative does not support NCHW on CPU.")
      inputs = np.transpose(inputs, [0, 3, 1, 2])
      strides = _reorder(strides, [0, 3, 1, 2])
      dilations = _reorder(dilations, [0, 3, 1, 2])
      if use_explicit_paddings:
        explicit_paddings = _reorder(explicit_paddings, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def tf_func(x):
      return tf.nn.depthwise_conv2d(
          x,
          filter=filters,
          strides=strides,
          padding=explicit_paddings if use_explicit_paddings else "SAME",
          data_format=data_format,
          dilations=dilations[2:] if data_format == "NCHW" else dilations[1:-1])
    self._test_convert(tf_func, inputs)

    def raw_func(x):
      return tf.raw_ops.DepthwiseConv2dNative(
          input=x,
          filter=filters,
          strides=strides,
          padding="EXPLICIT" if use_explicit_paddings else "SAME",
          explicit_paddings=sum(explicit_paddings, []),
          data_format=data_format,
          dilations=dilations)
    self._test_convert(raw_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_einsum(self):
    np.random.seed(42)

    inputs = [
        np.random.normal(size=[10, 2]).astype(np.float32),
        np.random.normal(size=[2, 5]).astype(np.float32),
    ]

    equation_tf = "ij,jk"
    def einsum_tf(inputs):
      return tf.einsum(equation_tf, *inputs)
    self._test_convert(einsum_tf, [inputs])

    equation_raw = "ij,jk->ik"
    def einsum_raw(inputs):
      return tf.raw_ops.Einsum(inputs=inputs, equation=equation_raw)
    self._test_convert(einsum_raw, [inputs])

  @chex.variants(with_jit=True, without_jit=True)
  def test_expand_dims(self):
    inputs, dims = 42, 0

    def expand_dims(x):
      return tf.raw_ops.ExpandDims(input=x, axis=dims)
    self._test_convert(expand_dims, inputs)

    # Check static inputs result in static outputs.
    def expand_dims_static():
      return tf.zeros(expand_dims(inputs))
    self._test_convert(expand_dims_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_empty(self):
    shape = (2,)

    def empty():
      return tf.raw_ops.Empty(shape=shape, dtype=tf.int32, init=True)
    self._test_convert(empty, [])

    # Check static inputs result in static outputs.
    def empty_static():
      return tf.zeros(empty())
    self._test_convert(empty_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_fill(self):
    dims, value = (np.int32(2),), np.int32(3)

    def fill(x):
      return tf.raw_ops.Fill(dims=dims, value=x)
    self._test_convert(fill, value)

    # Check static inputs result in static outputs.
    def fill_static():
      return tf.zeros(fill(value))
    self._test_convert(fill_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("inference", False), ("training", True)),
          (
              ("FusedBatchNormV2", "FusedBatchNormV2"),
              ("FusedBatchNormV3", "FusedBatchNormV3"),
          ),
          (("avg_zero", 0.0), ("avg_half", 0.5), ("avg_one", 1.0)),
          (("NHWC", "NHWC"), ("NCHW", "NCHW")),
          named=True,
      ))
  def test_fused_batch_norm(self, training, op_name, avg_factor, data_format):
    np.random.seed(42)

    ndim = 5
    inputs = np.random.normal(size=(3, 16, 16, ndim)).astype(np.float32)
    if data_format == "NCHW":
      inputs = np.transpose(inputs, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def raw_func(x):
      outputs = getattr(tf.raw_ops, op_name)(
          x=x,
          scale=[3.0] * ndim,
          offset=[2.0] * ndim,
          mean=np.array(range(ndim), np.float32),
          variance=np.array(range(ndim), np.float32) * 0.1,
          exponential_avg_factor=avg_factor,
          data_format=data_format,
          is_training=training)
      return outputs[:3]

    self._test_convert(raw_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("1_0", 1, 0), ("2_0", 2, 0), ("-1_0", -1, 0), ("-2_0", -2, 0),
           ("1_1", 1, 1), ("2_1", 2, 1), ("-1_1", -1, 1), ("-2_1", -2, 1),
           ("2_2", 2, 2), ("3_2", 3, 2), ("-1_2", -1, 2), ("-2_2", -2, 2)),
          named=True,
      ))
  def test_gather(self, axis, batch_dims):
    values = np.array(range(1, 2 * 4 * 5 * 3 + 1)).reshape((2, 4, 5, 3))
    indices = np.array(range(2 * 4 * 2)).reshape((2, 4, 2)) % 3

    def gather_fn(vals, idx):
      return tf.raw_ops.GatherV2(
          params=vals, indices=idx, axis=axis, batch_dims=batch_dims)

    self._test_convert(gather_fn, [values, indices])

    # Check static inputs result in static outputs.
    def gather_static():
      zeros_shape = gather_fn(values, indices)
      while zeros_shape.shape.ndims > 0:
        zeros_shape = zeros_shape[1]
      return tf.zeros(zeros_shape)
    self._test_convert(gather_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              (
                  "index",
                  [[1, 2], [3, 4]],
                  [[0, 0], [1, 1]],
              ),
              (
                  "slice",
                  [[1, 2], [3, 4]],
                  [[1], [0]],
              ),
              (
                  "index_3_1",
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[1]],
              ),
              (
                  "index_3_2",
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[0, 1], [1, 0]],
              ),
              (
                  "index_3_3",
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[0, 0, 1], [1, 0, 1]],
              ),
              (
                  "batch_index",
                  [[1, 2], [3, 4]],
                  [[[0, 0]], [[0, 1]]],
              ),
              (
                  "batch_slice",
                  [[1, 2], [3, 4]],
                  [[[1]], [[0]]],
              ),
              (
                  "batch_3d_1",
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[1]], [[0]]],
              ),
              (
                  "batch_3d_2",
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
              ),
              (
                  "batch_3d_3",
                  [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                  [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]],
              ),
          ),
          named=True,
      ))
  def test_gather_nd(self, values, indices):
    def gather_nd_fn(vals, idx):
      return tf.raw_ops.GatherNd(params=vals, indices=idx)

    self._test_convert(gather_nd_fn, [values, indices])

    # Check static inputs result in static outputs.
    def gather_nd_static():
      zeros_shape = gather_nd_fn(values, indices)
      while zeros_shape.shape.ndims > 0:
        zeros_shape = zeros_shape[-1]
      return tf.zeros(zeros_shape)
    self._test_convert(gather_nd_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_cond(self):
    inputs = [np.array(2), np.array(5)]

    def cond_fn(x, y):
      f1 = lambda: tf.multiply(x, 17)
      f2 = lambda: tf.add(y, 23)
      return tf.cond(tf.less(x, y), f1, f2)
    self._test_convert(cond_fn, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_igamma_ops(self):
    np.random.seed(42)
    inputs = (
        np.random.uniform(size=(10, 5)).astype(np.float32) * 10,
        np.random.uniform(size=(10, 5)).astype(np.float32) * 10,
    )

    self._test_convert(lambda a, x: tf.raw_ops.Igamma(a=a, x=x), inputs)
    self._test_convert(lambda a, x: tf.raw_ops.Igammac(a=a, x=x), inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_inplace_add(self):
    np.random.seed(42)

    def inplace_add(x, idx, val):
      return tf.raw_ops.InplaceAdd(x=x, i=idx, v=val)

    inputs = [
        np.random.normal(size=[10, 2]).astype(np.float32),
        [2, 5],
        [[1, 2], [3, 4]],
    ]
    self._test_convert(inplace_add, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_inplace_update(self):
    np.random.seed(42)

    def inplace_update(x, idx, val):
      return tf.raw_ops.InplaceUpdate(x=x, i=idx, v=val)

    inputs = [
        np.random.normal(size=[10, 2]).astype(np.float32),
        [2, 5],
        [[1, 2], [3, 4]],
    ]
    self._test_convert(inplace_update, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              ("unbatched", ([10, 2], [2, 5])),
              ("batched", ([7, 10, 2], [7, 2, 5])),
          ),
          (
              ("transpose_x", True),
              ("not_transpose_x", False),
          ),
          (
              ("transpose_y", True),
              ("not_transpose_y", False),
          ),
          named=True,
      ))
  def test_matmul(self, shapes, transpose_x, transpose_y):
    np.random.seed(42)

    inputs = [
        np.random.normal(size=shapes[0]).astype(np.float32),
        np.random.normal(size=shapes[1]).astype(np.float32),
    ]
    if transpose_x:
      inputs[0] = np.swapaxes(inputs[0], -1, -2)
    if transpose_y:
      inputs[1] = np.swapaxes(inputs[1], -1, -2)

    def matmul_tf(x, y):
      return tf.matmul(x, y, transpose_a=transpose_x, transpose_b=transpose_y)
    self._test_convert(matmul_tf, inputs)

    def matmul_raw(x, y):
      return tf.raw_ops.BatchMatMulV2(
          x=x, y=y, adj_x=transpose_x, adj_y=transpose_y)

    self._test_convert(matmul_raw, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters((2, 3), (-2, 3), (2, -3))
  def test_matix_band_part(self, lower, upper):
    inputs = np.arange(900).reshape((2, 3, 10, 15))

    def band_part(x):
      return tf.raw_ops.MatrixBandPart(
          input=x, num_lower=lower, num_upper=upper)
    self._test_convert(band_part, [inputs])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(np.float32, np.int32, np.bool_)
  def test_matrix_diag(self, dtype):
    np.random.seed(42)

    if dtype == np.float32:
      inputs = np.random.normal(size=[10, 3, 4]).astype(dtype)
      padding = dtype(47)
    elif dtype == np.int32:
      inputs = np.random.randint(low=0, high=10, size=[10, 3, 4], dtype=dtype)
      padding = dtype(47)
    elif dtype == np.bool_:
      inputs = np.random.normal(size=[10, 3, 4]) > 0.0
      padding = np.bool_(False)
    else:
      raise ValueError(f"Unsupported dtype={dtype}")

    def raw_func(x):
      return tf.raw_ops.MatrixDiagV3(
          diagonal=x, k=-2, num_rows=-1, num_cols=-1, padding_value=padding)
    self._test_convert(raw_func, inputs)

    def tf_func(x):
      return tf.linalg.diag(x, k=-2, padding_value=padding)
    self._test_convert(tf_func, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_max(self):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.int32)

    def max_fn(xs):
      return tf.raw_ops.Max(input=xs, axis=[0, 2, 1])
    self._test_convert(max_fn, inputs)

    # Check static inputs result in static outputs.
    def max_static():
      return tf.zeros(max_fn(inputs))
    self._test_convert(max_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("SAME", "SAME"), ("VALID", "VALID")),
          (("NHWC", "NHWC"), ("NCHW", "NCHW")),
          named=True,
      ))
  def test_max_pool(self, padding, data_format):
    np.random.seed(42)

    inputs = np.random.normal(size=(10, 32, 16, 8)).astype(np.float32)
    ksize = (1, 2, 3, 1)
    strides = (1, 3, 2, 1)

    if data_format == "NCHW":
      if jax.default_backend().lower() == "cpu":
        self.skipTest("TensorFlow MaxPool does not support NCHW on CPU.")
      inputs = np.transpose(inputs, [0, 3, 1, 2])
      ksize = _reorder(ksize, [0, 3, 1, 2])
      strides = _reorder(strides, [0, 3, 1, 2])
    else:
      assert data_format == "NHWC"

    def pool(x):
      return tf.raw_ops.MaxPool(
          input=x,
          ksize=ksize,
          strides=strides,
          padding=padding,
          data_format=data_format)
    self._test_convert(pool, [inputs])

  @chex.variants(with_jit=True, without_jit=True)
  def test_maximum(self):
    x_in, y_in = [1, 4], [2, 3]
    def maximum(x, y):
      return tf.raw_ops.Maximum(x=x, y=y)
    self._test_convert(maximum, [x_in, y_in])

    # Check static inputs result in static outputs.
    def maximum_static():
      return tf.zeros(maximum(x=x_in, y=y_in))
    self._test_convert(maximum_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_min(self):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.int32)

    def min_fn(xs):
      return tf.raw_ops.Min(input=xs, axis=[0, 2, 1])
    self._test_convert(min_fn, inputs)

    # Check static inputs result in static outputs.
    def min_static():
      return tf.zeros(min_fn(inputs))
    self._test_convert(min_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_minimum(self):
    x_in, y_in = [1, 4], [2, 3]
    def minimum(x, y):
      return tf.raw_ops.Minimum(x=x, y=y)
    self._test_convert(minimum, [x_in, y_in])

    # Check static inputs result in static outputs.
    def minimum_static():
      return tf.zeros(minimum(x=x_in, y=y_in))
    self._test_convert(minimum_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_one_hot(self):
    inputs = [[[1, 2], [3, 4], [5, 6]], 42, 47]

    def make_one_hot(axis):
      def one_hot(inds, on, off):
        return tf.raw_ops.OneHot(
            indices=inds, depth=10, on_value=on, off_value=off, axis=axis)
      return one_hot

    self._test_convert(make_one_hot(-1), inputs)
    self._test_convert(make_one_hot(2), inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_pack(self):
    def pack(inputs):
      return tf.raw_ops.Pack(values=inputs)
    self._test_convert(pack, [[np.zeros((10,)), np.zeros((10,))]])

    # Check static inputs result in static outputs.
    def pack_static():
      return tf.zeros(pack([10, 10]))
    self._test_convert(pack_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_pow(self):
    def power(x, y):
      return tf.raw_ops.Pow(x=x, y=y)
    self._test_convert(power, [2, 3])

    # Check static inputs result in static outputs.
    def power_static():
      return tf.zeros((power(2, 3),))
    self._test_convert(power_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_prevent_gradient(self):
    error_message = "Gradient not allowed."

    @tf.function
    def prevent(x):
      return tf.raw_ops.PreventGradient(input=x, message=error_message)

    np_x = np.array(3.14, dtype=np.float32)
    tf_x = tf.constant(np_x)
    tf_y = prevent(tf_x)

    with tf.GradientTape() as g:
      g.watch(tf_x)
      with self.assertRaisesRegex(LookupError, error_message):
        prevent(tf_x)

    jax_func = tf2jax.convert_functional(prevent, np.zeros_like(np_x))
    jax_func = self.variant(jax_func)
    jax_y = jax_func(np_x)

    with self.assertRaisesRegex(LookupError, error_message):
      jax.grad(jax_func)(np_x)

    self.assertAllClose(tf_y, jax_y)

  @chex.variants(with_jit=True, without_jit=True)
  def test_pad(self):
    inputs, pads, values = np.ones((10, 5)), [[1, 2], [3, 4]], 42.0

    def pad(xs):
      return tf.raw_ops.Pad(input=xs, paddings=pads)
    self._test_convert(pad, [inputs])

    def pad_v2(xs, value):
      return tf.raw_ops.PadV2(input=xs, paddings=pads, constant_values=value)
    self._test_convert(pad_v2, [inputs, values])

  @chex.variants(with_jit=True, without_jit=True)
  def test_prod(self):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.int32)

    def prod(xs):
      return tf.raw_ops.Prod(input=xs, axis=[0, 2, 1])
    self._test_convert(prod, inputs)

    # Check static inputs result in static outputs.
    def prod_static():
      return tf.zeros(prod(inputs))
    self._test_convert(prod_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_squeeze(self):
    inputs, dims = np.array([[[42], [47]]]), (0, 2)

    def squeeze(x):
      return tf.raw_ops.Squeeze(input=x, axis=dims)
    self._test_convert(squeeze, inputs)

    # Check static inputs result in static outputs.
    def squeeze_static():
      return tf.zeros(squeeze(inputs))
    self._test_convert(squeeze_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_rand_normal(self):
    tf.random.set_seed(42)  # Not sure how to translate this to Jax.

    def tf_func():
      return tf.random.normal(shape=(16, 7), dtype=tf.float32)
    self._test_convert(tf_func, (), check_shape_only=True)

    def raw_func():
      return tf.raw_ops.RandomStandardNormal(
          shape=(1000000, 7), dtype=tf.float32)
    self._test_convert(raw_func, (), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(tf.function(raw_func))
      jax_func = self.variant(jax_func)
      samples = jax_func(rng=jax.random.PRNGKey(42))
      self.assertAllClose(np.mean(samples), 0.0, atol=1e-3)
      self.assertAllClose(np.std(samples), 1.0, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(rng=jax.random.PRNGKey(42))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(rng=jax.random.PRNGKey(47))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_rand_uniform(self):
    tf.random.set_seed(42)  # Not sure how to translate this to Jax.

    def tf_func():
      return tf.random.uniform(shape=(16, 7), dtype=tf.float32)
    self._test_convert(tf_func, (), check_shape_only=True)

    def raw_func():
      return tf.raw_ops.RandomUniform(shape=(1000000, 7), dtype=tf.float32)
    self._test_convert(raw_func, (), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(tf.function(raw_func))
      jax_func = self.variant(jax_func)
      samples = jax_func(rng=jax.random.PRNGKey(42))
      for expected in np.linspace(0.1, 1, 10):
        actual = np.mean(samples < expected)
        self.assertAllClose(actual, expected, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(rng=jax.random.PRNGKey(42))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(rng=jax.random.PRNGKey(47))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_rand_uniform_int(self):
    tf.random.set_seed(42)  # Not sure how to translate this to Jax.

    def tf_func():
      return tf.random.uniform(
          shape=(16, 7), minval=0, maxval=10, dtype=tf.int32)
    self._test_convert(tf_func, (), check_shape_only=True)

    def raw_func():
      return tf.raw_ops.RandomUniformInt(
          shape=(1000000, 7), minval=0, maxval=10)
    self._test_convert(raw_func, (), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(tf.function(raw_func))
      jax_func = self.variant(jax_func)
      samples = jax_func(rng=jax.random.PRNGKey(42))
      for val in range(0, 10):
        actual = np.mean(samples == val)
        self.assertAllClose(actual, 0.1, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(rng=jax.random.PRNGKey(42))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(rng=jax.random.PRNGKey(47))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_range(self):
    inputs = [np.int32(1), np.int32(5), np.int32(2)]

    def range_fn(start, limit, delta):
      return tf.raw_ops.Range(start=start, limit=limit, delta=delta)
    # jnp.range cannot be jitted.
    with self._assert_if_jitted(jax.core.ConcretizationTypeError):
      self._test_convert(range_fn, inputs)

    # Check static inputs result in static outputs.
    def range_static():
      return tf.zeros(range_fn(*inputs))
    self._test_convert(range_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("upper_sample", (10, 10)), ("down_sample", (5, 5))),
          (
              ("not_align_and_half", (False, True)),
              ("not_align_and_not_half", (False, False)),
              ("align_and_not_half", (True, False)),
          ),
          named=True,
      ))
  def test_resize_bilinear(self, size, align_and_half):
    align, half_pixel = align_and_half

    def resize_bilinear(imgs):
      return tf.raw_ops.ResizeBilinear(
          images=imgs,
          size=size,
          align_corners=align,
          half_pixel_centers=half_pixel)

    images = np.random.normal(size=(4, 7, 7, 3)).astype(np.float32)
    self._test_convert(resize_bilinear, [images])

  @parameterized.named_parameters(
      chex.params_product(
          (
              ("align_and_half", (True, True)),
          ),
          named=True,
      ))
  def test_resize_bilinear_invalid(self, align_and_half):
    align, half_pixel = align_and_half

    def resize_bilinear(imgs):
      return tf.raw_ops.ResizeBilinear(
          images=imgs,
          size=(10, 10),
          align_corners=align,
          half_pixel_centers=half_pixel)

    images = np.random.normal(size=(4, 7, 7, 3)).astype(np.float32)
    with self.assertRaisesRegex(
        ValueError, "align_corners=True and half_pixel_centers=True"):
      _ = tf2jax.convert_functional(tf.function(resize_bilinear), images)

  @chex.variants(with_jit=True, without_jit=True)
  def test_reverse(self):
    axis = (1, 0)

    def reverse(x):
      return tf.raw_ops.ReverseV2(tensor=x, axis=axis)
    self._test_convert(reverse, [[[1, 2], [3, 4]]])

    def reverse_static():
      return tf.zeros(reverse([[1, 2], [3, 4]])[0])
    self._test_convert(reverse_static, ())

  @chex.variants(with_jit=True, without_jit=True)
  def test_scatter_nd(self):
    idxs = np.array([[2, 3], [5, 1]], dtype=np.int32)
    vals = np.array([[1, 2], [3, 4]], dtype=np.int32)
    shape = np.array([10, 5, 2], dtype=np.int32)

    def scatter_nd(idx, val):
      return tf.raw_ops.ScatterNd(indices=idx, updates=val, shape=shape)

    self._test_convert(scatter_nd, [idxs, vals])

  @chex.variants(with_jit=True, without_jit=True)
  def test_select(self):
    inputs = [
        np.array([True, False]),
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
    ]

    def select(cond, x, y):
      return tf.raw_ops.Select(condition=cond, x=x, y=y)
    self._test_convert(select, inputs)

    def select_static():
      return tf.zeros(select([True, False], [1, 2], [3, 4]))
    self._test_convert(select_static, ())

  @chex.variants(with_jit=True, without_jit=True)
  def test_size(self):
    inputs = np.array((10, 5))

    def size(x):
      return tf.raw_ops.Size(input=x)
    self._test_convert(size, inputs)

    def size_static():
      return tf.zeros(size(inputs))
    self._test_convert(size_static, ())

  @chex.variants(with_jit=True, without_jit=True)
  def test_slice(self):
    inputs, begins, sizes = [np.array([[1, 2], [3, 4], [5, 6]]), [1, 1], [2, 1]]

    def slice_fn(xs):
      return tf.raw_ops.Slice(input=xs, begin=begins, size=sizes)
    self._test_convert(slice_fn, inputs)

    # Check static inputs result in static outputs.
    def slice_static():
      return tf.zeros(slice_fn(inputs)[:, 0])
    self._test_convert(slice_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_sparse_softmax_cross_entropy_with_logits(self):
    features = np.array([np.arange(5), np.arange(5, 0, -1)], dtype=np.float32)
    labels = np.array([2, 3])

    def cross_entropy_fn(xs, ys):
      return tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits(
          features=xs, labels=ys)
    self._test_convert(cross_entropy_fn, (features, labels))

  @chex.variants(with_jit=True, without_jit=True)
  def test_split(self):
    inputs = np.array([[1, 2], [3, 4], [5, 6]])

    def split(inputs):
      return tf.raw_ops.Split(axis=0, value=inputs, num_split=len(inputs))
    self._test_convert(split, inputs)

    # Check static inputs result in static outputs.
    def split_static():
      return [tf.zeros(s[0]) for s in split(inputs)]
    self._test_convert(split_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              ("split1", (2, 4)),
              ("split2", (-1, 4)),
              ("split3", (2, -1)),
          ),
          (
              ("axis0", 0),
              ("axis1", 1),
          ),
          named=True,
      ))
  def test_splitv(self, splits, axis):
    inputs = np.arange(36).reshape(6, 6)

    def splitv(inputs):
      return tf.raw_ops.SplitV(
          value=inputs,
          size_splits=np.array(splits),
          axis=axis,
          num_split=len(splits))
    self._test_convert(splitv, inputs)

    # Check static inputs result in static outputs.
    def split_static():
      return [tf.zeros(s[0]) for s in splitv(inputs)]
    self._test_convert(split_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_normal(self):
    def tf_func(seed):
      return tf.random.stateless_normal(
          shape=(16, 7), seed=seed, dtype=tf.float32)
    self._test_convert(tf_func, (np.array([0, 42]),), check_shape_only=True)

    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomNormalV2(
          shape=(1000000, 7), key=key, counter=counter, alg=3, dtype=tf.float32)
    self._test_convert(raw_func, (np.array([0, 42]),), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func), np.array([0, 42]))
      jax_func = self.variant(jax_func)
      samples = jax_func(np.array([0, 42]))
      self.assertAllClose(np.mean(samples), 0.0, atol=1e-3)
      self.assertAllClose(np.std(samples), 1.0, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(np.array([0, 42]))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(np.array([0, 47]))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_uniform(self):
    def tf_func(seed):
      return tf.random.stateless_uniform(
          shape=(16, 7), seed=seed, dtype=tf.float32)
    self._test_convert(tf_func, (np.array([0, 42]),), check_shape_only=True)

    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomUniformV2(
          shape=(1000000, 7), key=key, counter=counter, alg=3, dtype=tf.float32)
    self._test_convert(raw_func, (np.array([0, 42]),), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func), np.array([0, 42]))
      jax_func = self.variant(jax_func)
      samples = jax_func(np.array([0, 42]))
      for expected in np.linspace(0.1, 1, 10):
        actual = np.mean(samples < expected)
        self.assertAllClose(actual, expected, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(np.array([0, 42]))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(np.array([0, 47]))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_uniform_int(self):
    def tf_func(seed):
      return tf.random.stateless_uniform(
          shape=(16, 7), seed=seed, minval=0, maxval=10, dtype=tf.int32)
    self._test_convert(tf_func, (np.array([0, 42]),), check_shape_only=True)

    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomUniformIntV2(
          shape=(1000000, 7),
          key=key,
          counter=counter,
          alg=3,
          minval=0,
          maxval=10)
    self._test_convert(raw_func, (np.array([0, 42]),), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func), np.array([0, 42]))
      jax_func = self.variant(jax_func)
      samples = jax_func(np.array([0, 42]))
      for val in range(0, 10):
        actual = np.mean(samples == val)
        self.assertAllClose(actual, 0.1, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(np.array([0, 42]))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(np.array([0, 47]))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_uniform_full_int(self):
    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomUniformFullIntV2(
          shape=(1000000, 7), key=key, counter=counter, alg=3, dtype=tf.int32)
    self._test_convert(raw_func, (np.array([0, 42]),), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func), np.array([0, 42]))
      jax_func = self.variant(jax_func)
      samples = jax_func(np.array([0, 42]))
      int32_min = np.iinfo(np.int32).min
      int32_max = np.iinfo(np.int32).max
      for val in range(int32_min, int32_max, 200_000_000):
        actual = np.mean(samples < val)
        expected = (val - int32_min) / (int32_max - int32_min)
        self.assertAllClose(actual, expected, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(np.array([0, 42]))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(np.array([0, 47]))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_multinomial(self):
    inputs = np.array([np.arange(5), np.arange(5, 0, -1)], dtype=np.float32)

    def raw_func(logits, seed):
      return tf.raw_ops.StatelessMultinomial(
          logits=logits, num_samples=1000000, seed=seed, output_dtype=tf.int32)

    self._test_convert(
        raw_func, (inputs, np.array([0, 42])), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func), inputs, np.array([0, 42]))
      jax_func = self.variant(jax_func)
      samples = jax_func(inputs, np.array([0, 42]))
      for batch in range(inputs.shape[0]):
        probs = jax.nn.softmax(inputs[batch])
        samps = samples[batch]
        for idx in range(inputs.shape[1]):
          self.assertAllClose(probs[idx], np.mean(samps == idx), atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(inputs, np.array([0, 42]))
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      diff_samples = jax_func(inputs, np.array([0, 47]))
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("slice0", lambda x: x[2:, :, :, :3]),
      ("slice1", lambda x: x[1:-2, ..., :3:2]),
      ("slice2", lambda x: x[..., 1:-2, :3:2]),
      ("slice3", lambda x: x[1:-2, :-2:2, ...]),
      ("newaxis0", lambda x: x[tf.newaxis, ...]),
      ("newaxis1", lambda x: x[..., tf.newaxis, 2:5]),
      ("newaxis2", lambda x: x[:4, tf.newaxis, ..., :2]),
      ("shrink0", lambda x: x[..., 2, 3]),
      ("shrink1", lambda x: x[2:, 3, :, 5]),
      ("mixed0", lambda x: x[..., 4:1:-1, tf.newaxis, 3]),
  )
  def test_strided_slice(self, slice_fn):
    inputs = np.linspace(0., 1., 4 * 5 * 6 * 7).astype(np.float32)
    inputs = inputs.reshape((4, 5, 6, 7))
    self._test_convert(slice_fn, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_sum(self):
    inputs = np.array(np.reshape(range(120), (5, 4, 3, 2)), dtype=np.int32)

    def sum_fn(xs):
      return tf.raw_ops.Sum(input=xs, axis=[2, 1])
    self._test_convert(sum_fn, inputs)

    # Check static inputs result in static outputs.
    def sum_static():
      return tf.zeros(sum_fn(inputs)[0])
    self._test_convert(sum_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  def test_switch_case(self):
    f1 = lambda: tf.constant(17)
    f2 = lambda: tf.constant(31)
    f3 = lambda: tf.constant(-1)

    def switch_fn(x):
      return tf.switch_case(
          tf.convert_to_tensor(x),
          branch_fns={
              0: f1,
              1: f2,
          },
          default=f3,
      )
    self._test_convert(switch_fn, [np.array(1, dtype=np.int32)])

  @chex.variants(with_jit=True, without_jit=True)
  def test_unpack(self):
    inputs = np.array([[1, 2], [3, 4], [5, 6]])

    def unpack(inputs):
      return tf.raw_ops.Unpack(value=inputs, num=len(inputs))
    self._test_convert(unpack, inputs)

    # Check static inputs result in static outputs.
    def unpack_static():
      return [tf.zeros(s) for s in unpack(inputs)]
    self._test_convert(unpack_static, [])

  @chex.variants(without_jit=True)
  def test_where(self):
    inputs = [np.array([True, False])]

    def where(cond):
      return tf.raw_ops.Where(condition=cond)
    self._test_convert(where, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_while_loop(self):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.float32)
    # If offset is a tensor, then the raw versions will fail to capture them as
    # inputs. tf.while_loop correctly handles it.
    offset = np.array(3.14, dtype=np.float32)
    cond = lambda v, i: tf.less(i, 10)
    body = lambda v, i: (v + tf.cast(i, tf.float32) + offset, tf.add(i, 1))
    step = tf.constant(0)

    def while_loop(x):
      return tf.while_loop(cond, body, [x, step])
    self._test_convert(while_loop, inputs)

    if jax.default_backend().lower() == "gpu":
      self.skipTest("Skip remaining tests on GPU due to CUDA errors.")

    def raw_stateless_while(x):
      loop_vars = [x, step]
      return tf.raw_ops.StatelessWhile(
          input=loop_vars,
          cond=tf.function(cond).get_concrete_function(*loop_vars),
          body=tf.function(body).get_concrete_function(*loop_vars))
    self._test_convert(raw_stateless_while, inputs)

    def raw_while(x):
      loop_vars = [x, step]
      return tf.raw_ops.While(
          input=loop_vars,
          cond=tf.function(cond).get_concrete_function(*loop_vars),
          body=tf.function(body).get_concrete_function(*loop_vars))
    self._test_convert(raw_while, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_while_loop_with_random(self):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.float32)
    cond = lambda v, i: tf.less(i, 10)
    body = lambda v, i: (v + tf.random.uniform(v.shape), tf.add(i, 1))
    step = tf.constant(0)

    def while_loop(x):
      return tf.while_loop(cond, body, [x, step])
    self._test_convert(while_loop, inputs, check_shape_only=True)

  @chex.variants(with_jit=True, without_jit=True)
  def test_while_loop_side_effect(self):
    inputs = np.array([42, 47], dtype=np.float32)
    acc_var = tf.Variable(initial_value=[3, 14], dtype=tf.float32)
    cond = lambda v, i: tf.less(i, 2)
    body = lambda v, i: (acc_var.assign_add(v), tf.add(i, 1))
    step = tf.constant(0)

    def while_loop(x):
      return tf.while_loop(cond, body, [x, step]), [acc_var]

    with self.assertRaisesRegex(ValueError, "Some updated parameters are lost"):
      self._test_convert(while_loop, inputs, functional=False)

  @chex.variants(with_jit=True, without_jit=True)
  def test_assign_side_effect(self):
    inputs = np.array([42, 47], dtype=np.float32)
    acc_var = tf.Variable(initial_value=[3, 14], dtype=tf.float32, name="blah")

    def tf_fn(x):
      unused_y = tf.sin(x)
      acc_var.assign_add(tf.cos(x))
      return ()

    jax_result, jax_params = self._test_convert(tf_fn, inputs, functional=False)
    self.assertEqual(jax_result, ())
    self.assertAllClose(jax_params["blah"], acc_var)

  @chex.variants(with_jit=True, without_jit=True)
  def test_top_k(self):
    inputs = np.array([range(10), range(10)[::-1]], dtype=np.float32)
    k = tf.constant(5, dtype=tf.int32)

    def top_k(x):
      return tf.raw_ops.TopKV2(input=x, k=k, sorted=True)
    self._test_convert(top_k, inputs)


if __name__ == "__main__":
  tf.test.main()
