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
from jax.experimental import checkify
import numpy as np

import tensorflow as tf
from tf2jax._src import config
from tf2jax._src import ops
from tf2jax._src import test_util
from tf2jax._src import tf2jax
import tree


def _reorder(vals, inds):
  return [vals[idx] for idx in inds]


class OpsTest(test_util.TestCase):

  def test_get_unsupported(self):
    unsupported = ops.get_unsupported_operations(
        ["Add", "Relu", "NotAnOp", "Blah", "Relu"])
    self.assertEqual(unsupported, {"NotAnOp", "Blah"})

  def _assert_if_jitted(self, err):
    jitted = self.variant.type == chex.ChexVariantType.WITH_JIT
    return self.assertRaises(err) if jitted else contextlib.nullcontext()

  def _test_convert(self,
                    tf_func,
                    inputs,
                    *,
                    check_shape_only=False,
                    functional=True,
                    atol=1e-5):
    if not isinstance(inputs, (list, tuple)):
      inputs = (inputs,)

    if not hasattr(tf_func, "get_concrete_function"):
      tf_func = tf.function(tf_func, jit_compile=True)

    jax_func, jax_params = tf2jax.convert(
        tf_func, *tree.map_structure(np.zeros_like, inputs))
    if functional:
      self.assertEmpty(jax_params, "Expected no parameters for pure Ops.")

    jax_func = self.variant(jax_func)

    rng = jax.random.PRNGKey(42)
    jax_results, new_jax_params = jax_func(jax_params, *inputs, rng=rng)
    tf_results = tf_func(*inputs)

    # Check outputs
    for tf_res, jax_res in jax.util.safe_zip(
        tree.flatten(tf_results), tree.flatten(jax_results)
    ):
      self.assertEqual(tf_res.shape, jax_res.shape)
      if not check_shape_only:
        self.assertAllClose(np.asarray(tf_res), jax_res, atol=atol)

    return jax_results, new_jax_params

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("log_softmax", "sigmoid", "softmax", "softplus",
                            "tanh", "relu", "relu6", "elu", "leaky_relu")
  def test_activations(self, op_name):
    np.random.seed(42)
    inputs = [np.random.normal(size=(10, 5)).astype(np.float32)]

    if jax.default_backend().lower() == "tpu" and op_name in ("softplus",):
      tols = dict(atol=1e-4)
    else:
      tols = {}

    self._test_convert(getattr(tf.nn, op_name), inputs, **tols)

  @chex.variants(with_jit=True, without_jit=True)
  def test_assert(self):

    def assert_fn(cond, data):
      tf.Assert(condition=cond, data=data)
      return data

    self._test_convert(assert_fn, [np.array(5) > 0, [np.array(6)]])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
      "Abs",
      "Acosh",
      "Asinh",
      "Atanh",
      "BesselI0e",
      "BesselI1e",
      "Ceil",
      "Cos",
      "Cosh",
      "Digamma",
      "Erf",
      "Erfc",
      "Erfinv",
      "Exp",
      "Expm1",
      "Floor",
      "Lgamma",
      "Log",
      "Log1p",
      "Neg",
      "Reciprocal",
      "Round",
      "Rsqrt",
      "Sign",
      "Sin",
      "Sinh",
      "Sqrt",
      "Square",
      "Tan",
  )
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

    if jax.default_backend().lower() == "tpu" and op_name in ("Pow",):
      tols = dict(atol=1e-4)
    else:
      tols = {}

    self._test_convert(tf_func, inputs, **tols)

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
      (
          "case0",
          [
              [[[1]]],
              [[[2]]],
              [[[3]]],
              [[[4]]],
          ],
          (2, 2),
          [[0, 0], [0, 0]],
      ),
      (
          "case1",
          [
              [[[1, 2, 3]]],
              [[[4, 5, 6]]],
              [[[7, 8, 9]]],
              [[[10, 11, 12]]],
          ],
          (2, 2),
          [[0, 0], [0, 0]],
      ),
      (
          "case2",
          [
              [[[1], [3]], [[9], [11]]],
              [[[2], [4]], [[10], [12]]],
              [[[5], [7]], [[13], [15]]],
              [[[6], [8]], [[14], [16]]],
          ],
          (2, 2),
          [[0, 0], [0, 0]],
      ),
      (
          "case3",
          [
              [[[0], [1], [3]]],
              [[[0], [9], [11]]],
              [[[0], [2], [4]]],
              [[[0], [10], [12]]],
              [[[0], [5], [7]]],
              [[[0], [13], [15]]],
              [[[0], [6], [8]]],
              [[[0], [14], [16]]],
          ],
          (2, 2),
          [[0, 0], [2, 0]],
      ),
  )
  def test_batch_to_space_nd(self, inputs, block_shape, crops):
    inputs = np.array(inputs)
    block_shape = np.array(block_shape)
    crops = np.array(crops)

    def batch_to_space(x):
      return tf.raw_ops.BatchToSpaceND(
          input=x, block_shape=block_shape, crops=crops
      )

    self._test_convert(batch_to_space, [inputs])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      (
          "case0",
          [
              [[[1], [2]], [[3], [4]]],
          ],
          (2, 2),
          [[0, 0], [0, 0]],
      ),
      (
          "case1",
          [
              [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
          ],
          (2, 2),
          [[0, 0], [0, 0]],
      ),
      (
          "case2",
          [[
              [[1], [2], [3], [4]],
              [[5], [6], [7], [8]],
              [[9], [10], [11], [12]],
              [[13], [14], [15], [16]],
          ]],
          (2, 2),
          [[0, 0], [0, 0]],
      ),
      (
          "case3",
          [
              [[[1], [2], [3], [4]], [[5], [6], [7], [8]]],
              [[[9], [10], [11], [12]], [[13], [14], [15], [16]]],
          ],
          (2, 2),
          [[0, 0], [2, 0]],
      ),
  )
  def test_space_to_batch_nd(self, inputs, block_shape, paddings):
    inputs = np.array(inputs)
    block_shape = np.array(block_shape)
    paddings = np.array(paddings)

    def space_to_batch(x):
      return tf.raw_ops.SpaceToBatchND(
          input=x, block_shape=block_shape, paddings=paddings
      )

    self._test_convert(space_to_batch, [inputs])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (("NHWC", "NHWC"), ("NCHW", "NCHW")),
          (
              ("three_dims", (32, 16, 8)),
              ("four_dims", (3, 32, 16, 8)),
              ("five_dims", (2, 3, 32, 16, 8)),
          ),
          named=True,
      ))
  def test_bias_add(self, data_format, input_shape):
    np.random.seed(42)

    if data_format == "NCHW":
      nchannels = input_shape[1]
    elif data_format == "NHWC":
      nchannels = input_shape[-1]
    else:
      raise ValueError(f"Unsupported format {data_format}")

    inputs = np.random.normal(size=input_shape).astype(np.float32)
    bias = bias = np.linspace(-1., 1., nchannels).astype(np.float32)

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
  @parameterized.named_parameters(
      ("valid", np.array((3.14, 42.0), dtype=np.float32), None),
      ("nan", np.array((np.nan, 42.0), dtype=np.float32), "Found NaN values"),
      ("inf", np.array((3.14, np.inf), dtype=np.float32), "Found Inf values"),
  )
  def test_check_numerics(self, inputs, expected_message):

    def check_numerics(x):
      return tf.raw_ops.CheckNumerics(tensor=x, message="Checking")

    with config.override_config("enable_checkify_for_asserts", False):
      jax_fn = tf2jax.convert_functional(tf.function(check_numerics), inputs)
      jax_fn = self.variant(jax_fn)
      outputs = jax_fn(inputs)
      if not expected_message:
        self.assertAllClose(outputs, inputs)

      checked_fn = checkify.checkify(jax_fn)
      err, checked_outputs = checked_fn(inputs)
      err.throw()  # Nothing to throw.
      if not expected_message:
        self.assertAllClose(checked_outputs, inputs)

    with config.override_config("enable_checkify_for_asserts", True):
      jax_fn = tf2jax.convert_functional(tf.function(check_numerics), inputs)
      jax_fn = self.variant(jax_fn)

      checked_fn = checkify.checkify(jax_fn)
      err, checked_outputs = checked_fn(inputs)
      if not expected_message:
        self.assertAllClose(checked_outputs, inputs)
      else:
        with self.assertRaisesRegex(checkify.JaxRuntimeError,
                                    f"Checking : {expected_message}"):
          err.throw()

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
  @parameterized.named_parameters(
      ("exact", (10, 5), (10, 5), True),
      ("partial0", (10, None), (10, 5), True),
      ("partial1", (None, 5), (10, 5), True),
      ("partial2", (None, None), (10, 5), True),
      ("too_small", (10,), (10,), False),
      ("too_large", (10, 5, 3), (10, 5, 3), False),
      ("incompatible0", (20, 5), (20, 5), False),
      ("incompatible1", (10, 7), (10, 7), False),
      ("incompatible2", (20, None), (20, 5), False),
      ("incompatible3", (None, 7), (10, 7), False),
  )
  def test_ensure_shape(self, expected_shape, example_shape, valid):

    def ensure_shape(x):
      return tf.raw_ops.EnsureShape(
          input=x, shape=tf.TensorShape(expected_shape))

    valid_inputs = np.array(
        range(np.prod(example_shape)), dtype=np.float32).reshape(example_shape)
    self._test_convert(ensure_shape, [valid_inputs])

    jax_fn = tf2jax.convert_functional(tf.function(ensure_shape), valid_inputs)
    jax_fn = self.variant(jax_fn)

    with config.override_config("strict_shape_check", False):
      actual_inputs = np.array(range(50), dtype=np.float32).reshape((10, 5))
      if valid:
        outputs = jax_fn(actual_inputs)
        self.assertAllClose(outputs, actual_inputs)
      else:
        with self.assertRaisesRegex(ValueError, "Expected shape="):
          jax_fn(actual_inputs)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("FFT", tf.raw_ops.FFT, (10,)),
      ("FFT2D", tf.raw_ops.FFT2D, (10, 8)),
      ("FFT3D", tf.raw_ops.FFT3D, (10, 8, 6)),
      ("IFFT", tf.raw_ops.IFFT, (10,)),
      ("IFFT2D", tf.raw_ops.IFFT2D, (10, 8)),
      ("IFFT3D", tf.raw_ops.IFFT3D, (10, 8, 6)),
  )
  def test_fft_ifft(self, fft_op, shape):
    np.random.seed(42)
    inputs = np.random.normal(size=(5,) + shape).astype(np.complex64)
    tols = dict(atol=1e-4) if jax.default_backend().lower() == "cpu" else {}
    self._test_convert(lambda x: fft_op(input=x), inputs, **tols)

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("RFFT", tf.raw_ops.RFFT, (10,), np.float32),
      ("RFFT2D", tf.raw_ops.RFFT2D, (10, 8), np.float32),
      ("RFFT3D", tf.raw_ops.RFFT3D, (10, 8, 6), np.float32),
      ("IRFFT", tf.raw_ops.IRFFT, (10,), np.complex64),
      ("IRFFT2D", tf.raw_ops.IRFFT2D, (10, 8), np.complex64),
      ("IRFFT3D", tf.raw_ops.IRFFT3D, (10, 8, 6), np.complex64),
  )
  def test_rfft_irfft(self, fft_op, shape, dtype):
    np.random.seed(42)
    inputs = np.random.normal(size=(5,) + shape).astype(dtype)
    self._test_convert(
        lambda x: fft_op(input=x, fft_length=[6] * len(shape)), inputs
    )

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
              ("FusedBatchNorm", "FusedBatchNorm"),
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
    if test_util.parse_version(tf.version.VERSION) >= test_util.parse_version(
        "2.14.0"
    ):
      self.skipTest(
          f"Requires earlier than tf 2.14.0, found {tf.version.VERSION}."
      )

    np.random.seed(42)

    @tf.function
    def inplace_add(x, idx, val):
      return tf.raw_ops.InplaceAdd(x=x, i=idx, v=val)

    inputs = [
        np.random.normal(size=[10, 2]).astype(np.float32),
        [2, 5],
        [[1, 2], [3, 4]],
    ]

    try:
      self._test_convert(inplace_add, inputs)
    except tf.errors.InvalidArgumentError as e:
      if jax.default_backend().lower() == "tpu":
        self.assertIn("index must be Rank 1 and size 1", str(e))  # pylint: disable=g-assert-in-except
        tpu_inputs = [
            np.random.normal(size=[10, 2]).astype(np.float32),
            [5],
            [[3, 4]],
        ]
        self._test_convert(inplace_add, tpu_inputs)
      else:
        raise e

  @chex.variants(with_jit=True, without_jit=True)
  def test_inplace_update(self):
    if test_util.parse_version(tf.version.VERSION) >= test_util.parse_version(
        "2.14.0"
    ):
      self.skipTest(
          f"Requires earlier than tf 2.14.0, found {tf.version.VERSION}."
      )

    np.random.seed(42)

    @tf.function
    def inplace_update(x, idx, val):
      return tf.raw_ops.InplaceUpdate(x=x, i=idx, v=val)

    inputs = [
        np.random.normal(size=[10, 2]).astype(np.float32),
        [2, 5],
        [[1, 2], [3, 4]],
    ]
    self._test_convert(inplace_update, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_invert_permutation(self):
    np.random.seed(42)

    def invert(x):
      return tf.raw_ops.InvertPermutation(x=x)

    inputs = np.array([2, 4, 3, 0, 1])
    self._test_convert(invert, inputs)

    def invert_static():
      return tf.zeros(invert(inputs))
    self._test_convert(invert_static, [])

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
  @parameterized.parameters(np.float32, np.int32, np.bool_)
  def test_matrix_set_diag(self, dtype):
    np.random.seed(42)

    diagonal = np.array([[1, 2, 3], [4, 5, 6]]).astype(dtype)

    if dtype == np.float32:
      inputs = np.random.normal(size=[2, 3, 4]).astype(dtype)
    elif dtype == np.int32:
      inputs = np.random.randint(low=0, high=10, size=[2, 3, 4], dtype=dtype)
    elif dtype == np.bool_:
      inputs = np.random.normal(size=[2, 3, 4]) > 0.0
      diagonal = diagonal > 3
    else:
      raise ValueError(f"Unsupported dtype={dtype}")

    def raw_func(x, y):
      return tf.raw_ops.MatrixSetDiagV3(input=x, diagonal=y, k=1)
    self._test_convert(raw_func, [inputs, diagonal])

    def tf_func(x, y):
      return tf.linalg.set_diag(x, y, k=1)
    self._test_convert(tf_func, [inputs, diagonal])

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
  @parameterized.parameters(
      np.int8,
      np.int16,
      np.int32,
      np.int64,
      np.uint8,
      np.uint16,
      np.uint32,
      np.uint64,
  )
  def test_population_count(self, dtype):
    def population_count(x):
      return tf.raw_ops.PopulationCount(x=x)

    inputs = np.arange(1000, dtype=dtype)
    self._test_convert(population_count, inputs)

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
      return tf.raw_ops.PreventGradient(input=x * x, message=error_message)

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

    self.assertAllClose(tf_y, jax_y)
    with self.assertRaisesRegex(LookupError, error_message):
      jax.grad(jax_func)(np_x)

    with tf2jax.config.override_config("raise_on_prevent_gradient", False):
      jax_func = tf2jax.convert_functional(prevent, np.zeros_like(np_x))
      jax_func = self.variant(jax_func)
      jax_y = jax_func(np_x)
      self.assertAllClose(tf_y, jax_y)

      jax_grad = jax.grad(jax_func)(np_x)
      self.assertAllClose(np_x * 2, jax_grad)

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
  def test_rank(self):
    inputs = np.array(np.reshape(range(24), (4, 3, 2)), dtype=np.int32)

    def rank(xs):
      return tf.raw_ops.Rank(input=xs)
    self._test_convert(rank, inputs)

    # Check static inputs result in static outputs.
    def rank_static():
      return tf.zeros(rank(inputs))
    self._test_convert(rank_static, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("basic0", 2, 0),
      ("basic1", -2, 1),
      ("basic2", 0, 1),
      ("single", (2,), (1,)),
      ("double0", (2, -1), (0, 1)),
      ("double1", (2, -1), (1, 0)),
      ("dupe_axis", (2, 1), (1, 1)),
  )
  def test_roll(self, shift, axis):
    inputs = np.array(range(1, 51)).reshape((10, 5))

    def roll(x):
      return tf.raw_ops.Roll(input=x, shift=shift, axis=axis)
    self._test_convert(roll, inputs)

    # Check static inputs result in static outputs.
    def roll_static():
      return tf.zeros(roll(inputs)[0])
    self._test_convert(roll_static, [])

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
    np.random.seed(42)
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
    np.random.seed(42)
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
  @parameterized.named_parameters(
      chex.params_product(
          (("upper_sample", (10, 10)), ("down_sample", (5, 5))),
          (
              ("not_align_and_half", (False, True)),
          ),
          named=True,
      ))
  def test_resize_nearest_neighbor(self, size, align_and_half):
    np.random.seed(42)
    align, half_pixel = align_and_half

    def resize_nearest_neighbor(imgs):
      return tf.raw_ops.ResizeNearestNeighbor(
          images=imgs,
          size=size,
          align_corners=align,
          half_pixel_centers=half_pixel)

    images = np.random.normal(size=(4, 7, 7, 3)).astype(np.float32)
    self._test_convert(resize_nearest_neighbor, [images])

  @parameterized.named_parameters(
      chex.params_product(
          (
              ("align_and_half", (True, True)),
              ("align_and_not_half", (True, False)),
              ("not_align_and_not_half", (False, False)),
          ),
          named=True,
      ))
  def test_resize_nearest_neighbor_invalid(self, align_and_half):
    np.random.seed(42)
    align, half_pixel = align_and_half

    def resize_nearest_neighbor(imgs):
      return tf.raw_ops.ResizeNearestNeighbor(
          images=imgs,
          size=(10, 10),
          align_corners=align,
          half_pixel_centers=half_pixel)

    images = np.random.normal(size=(4, 7, 7, 3)).astype(np.float32)
    with self.assertRaisesRegex(
        ValueError, "align_corners=False and half_pixel_centers=True"):
      _ = tf2jax.convert_functional(tf.function(resize_nearest_neighbor),
                                    images)

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

    def scatter_nd_static():
      return tf.zeros((tf.reduce_sum(scatter_nd(idxs, vals),)))
    self._test_convert(scatter_nd_static, ())

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
  def test_stateless_random_get_alg(self):

    @tf.function
    def tf_func():
      return tf.raw_ops.StatelessRandomGetAlg()

    jax_func = tf2jax.convert_functional(tf_func)
    jax_alg = self.variant(jax_func)()

    self.assertEqual(jax_alg, tf.random.Algorithm.AUTO_SELECT.value)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_normal(self):
    def tf_func(seed):
      return tf.random.stateless_normal(
          shape=(16, 7), seed=seed, dtype=tf.float32)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(tf_func, (seed,), check_shape_only=True)

    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomNormalV2(
          shape=(1000000, 7), key=key, counter=counter, alg=3, dtype=tf.float32)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(raw_func, (seed,), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func, jit_compile=True), seed)
      jax_func = self.variant(jax_func)
      samples = jax_func(seed)
      self.assertAllClose(np.mean(samples), 0.0, atol=1e-3)
      self.assertAllClose(np.std(samples), 1.0, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(seed)
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      another_seed = np.array([0, 47], dtype=np.int32)
      diff_samples = jax_func(another_seed)
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_uniform(self):
    def tf_func(seed):
      return tf.random.stateless_uniform(
          shape=(16, 7), seed=seed, dtype=tf.float32)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(tf_func, (seed,), check_shape_only=True)

    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomUniformV2(
          shape=(1000000, 7), key=key, counter=counter, alg=3, dtype=tf.float32)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(raw_func, (seed,), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func, jit_compile=True), seed)
      jax_func = self.variant(jax_func)
      samples = jax_func(seed)
      for expected in np.linspace(0.1, 1, 10):
        actual = np.mean(samples < expected)
        self.assertAllClose(actual, expected, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(seed)
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      another_seed = np.array([0, 47], dtype=np.int32)
      diff_samples = jax_func(another_seed)
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_uniform_int(self):
    def tf_func(seed):
      return tf.random.stateless_uniform(
          shape=(16, 7), seed=seed, minval=0, maxval=10, dtype=tf.int32)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(tf_func, (seed,), check_shape_only=True)

    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomUniformIntV2(
          shape=(1000000, 7),
          key=key,
          counter=counter,
          alg=3,
          minval=0,
          maxval=10)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(raw_func, (seed,), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func, jit_compile=True), seed)
      jax_func = self.variant(jax_func)
      samples = jax_func(seed)
      for val in range(0, 10):
        actual = np.mean(samples == val)
        self.assertAllClose(actual, 0.1, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(seed)
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      another_seed = np.array([0, 47], dtype=np.int32)
      diff_samples = jax_func(another_seed)
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_random_uniform_full_int(self):
    def raw_func(seed):
      key, counter = tf.raw_ops.StatelessRandomGetKeyCounter(seed=seed)
      return tf.raw_ops.StatelessRandomUniformFullIntV2(
          shape=(1000000, 7), key=key, counter=counter, alg=3, dtype=tf.int32)

    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(raw_func, (seed,), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func, jit_compile=True), seed)
      jax_func = self.variant(jax_func)
      samples = jax_func(seed)
      int32_min = np.iinfo(np.int32).min
      int32_max = np.iinfo(np.int32).max
      for val in range(int32_min, int32_max, 200_000_000):
        actual = np.mean(samples < val)
        expected = (val - int32_min) / (int32_max - int32_min)
        self.assertAllClose(actual, expected, atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(seed)
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      another_seed = np.array([0, 47], dtype=np.int32)
      diff_samples = jax_func(another_seed)
      self.assertNotAllClose(samples, diff_samples)

  @chex.variants(with_jit=True, without_jit=True)
  def test_stateless_multinomial(self):
    def raw_func(logits, seed):
      return tf.raw_ops.StatelessMultinomial(
          logits=logits, num_samples=1000000, seed=seed, output_dtype=tf.int32)

    inputs = np.array([np.arange(5), np.arange(5, 0, -1)], dtype=np.float32)
    seed = np.array([0, 42], dtype=np.int32)
    self._test_convert(raw_func, (inputs, seed), check_shape_only=True)

    with self.subTest("check_statistics"):
      jax_func = tf2jax.convert_functional(
          tf.function(raw_func, jit_compile=True), inputs, seed)
      jax_func = self.variant(jax_func)
      samples = jax_func(inputs, seed)
      for batch in range(inputs.shape[0]):
        probs = jax.nn.softmax(inputs[batch])
        samps = samples[batch]
        for idx in range(inputs.shape[1]):
          self.assertAllClose(probs[idx], np.mean(samps == idx), atol=1e-3)

    with self.subTest("check_same_seed"):
      same_samples = jax_func(inputs, seed)
      self.assertAllClose(samples, same_samples)

    with self.subTest("check_diff_seed"):
      another_seed = np.array([0, 47], dtype=np.int32)
      diff_samples = jax_func(inputs, another_seed)
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
      ("zero_slice", lambda x: x[:, :0, ...]),
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
  @parameterized.named_parameters(
      ("defined", (3, 2)),
      ("partial0", (-1, 2)),
      ("partial1", (3, -1)),
      ("undefined", -1)
  )
  def test_tensor_list_get_item(self, init_shape):
    @tf.function(jit_compile=False)
    def tensor_list_fn():
      handle = tf.raw_ops.TensorListReserve(
          element_shape=init_shape, num_elements=3, element_dtype=tf.int32)
      return tf.raw_ops.TensorListGetItem(
          input_handle=handle,
          index=1,
          element_shape=(3, 2),
          element_dtype=tf.int32)

    self._test_convert(tensor_list_fn, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      ("defined", dict(fn_output_signature=tf.TensorSpec((5, 4), tf.int32))),
      ("undefined", {})
  )
  def test_tensor_list(self, output_signature_kwargs):
    def tensor_list_fn():
      return tf.map_fn(
          fn=lambda t: tf.ones((5, 4), dtype=tf.int32) * t,
          elems=tf.constant([3, 5, 2]),
          **output_signature_kwargs,
      )

    self._test_convert(tensor_list_fn, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.named_parameters(
      chex.params_product(
          (
              (
                  "1D",
                  np.ones([8], dtype=np.float32),
                  np.array([[4], [3], [1], [7]], dtype=np.int32),
                  np.array([9, 10, 11, 12], dtype=np.float32),
              ),
              (
                  "2D_scalar",
                  np.ones([8, 2], dtype=np.float32),
                  np.array([[4, 0], [3, 1], [1, 0], [7, 1]], dtype=np.int32),
                  np.array([9, 10, 11, 12], dtype=np.float32),
              ),
              (
                  "2D_slice",
                  np.ones([8, 2], dtype=np.float32),
                  np.array([[4], [3], [1], [7]], dtype=np.int32),
                  np.array([[9, 90], [10, 100], [11, 110], [12, 120]],
                           dtype=np.float32),
              ),
              (
                  "3D_scalar",
                  np.ones([8, 3, 2], dtype=np.float32),
                  np.array([[4, 0, 0], [3, 1, 1], [1, 2, 0], [7, 0, 1]],
                           dtype=np.int32),
                  np.array([9, 10, 11, 12], dtype=np.float32),
              ),
              (
                  "3D_slice",
                  np.ones([8, 3, 2], dtype=np.float32),
                  np.array([[4, 0], [3, 1], [1, 2], [7, 0]], dtype=np.int32),
                  np.array([[9, 90], [10, 100], [11, 110], [12, 120]],
                           dtype=np.float32),
              ),
              (
                  "3D_block",
                  np.ones([8, 3, 2], dtype=np.float32),
                  np.array([[4], [3], [1], [7]], dtype=np.int32),
                  np.array([
                      [[9, 90], [91, 92], [93, 94]],
                      [[10, 100], [101, 102], [103, 104]],
                      [[11, 110], [111, 112], [113, 114]],
                      [[12, 120], [121, 122], [123, 124]],
                  ],
                           dtype=np.float32),
              ),
          ),
          named=True,
      ))
  def test_tensor_scatter_update(self, tensor, indices, updates):
    def scatter(x, inds, ups):
      return tf.raw_ops.TensorScatterUpdate(tensor=x, indices=inds, updates=ups)

    self._test_convert(scatter, [tensor, indices, updates])

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

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters("UnsortedSegmentSum", "UnsortedSegmentMax",
                            "UnsortedSegmentMin", "UnsortedSegmentProd")
  def test_unsorted_segment(self, op_name):
    def segment_reduce(x, ids):
      return getattr(tf.raw_ops, op_name)(
          data=x, segment_ids=ids, num_segments=2)

    data = np.array([5, 1, 7, 2, 3, 4], np.float32)
    segment_ids = np.array([0, 0, 1, 1, 0, 1], np.int32)
    self._test_convert(segment_reduce, [data, segment_ids])

    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [4, 3, 2, 1]], np.float32)
    segment_ids = np.array([0, 1, 0], np.int32)
    self._test_convert(segment_reduce, [data, segment_ids])

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
  def test_while_captured_static_args(self):
    # This can still fail if output_size is an argument to cond and body.
    output_size = tf.constant([10, 5])
    cond = lambda v, i: tf.less(i, 10)
    body = lambda v, i: (v + tf.ones(output_size, tf.float32), tf.add(i, 1))
    step = tf.constant(0)

    @tf.function
    def while_loop(x):
      return tf.while_loop(cond, body, [x, step])

    if jax.default_backend().lower() == "tpu":
      tpu_context = self.assertRaisesRegex(
          tf.errors.InvalidArgumentError,
          ("Input 0 to node `while/ones` with op Fill must be a compile-time "
           "constant."))
    else:
      tpu_context = contextlib.nullcontext()

    with tpu_context:
      self._test_convert(while_loop, np.zeros(output_size, np.float32))

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

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters((2., 2.), (2., 0.), (0., 0.), (0., 2.))
  def test_div_no_nan(self, x, y):

    @tf.function
    def div_no_nan(inputs):
      x, y = inputs
      return tf.math.divide_no_nan(x=x, y=y)

    x = np.array(x)
    y = np.array(y)
    tf_x = tf.convert_to_tensor(x)
    tf_y = tf.convert_to_tensor(y)

    with tf.GradientTape() as g:
      g.watch(tf_x)
      g.watch(tf_y)
      output = div_no_nan([tf_x, tf_y])
    tf_gradient = g.gradient(output, [tf_x, tf_y])

    jax_func = tf2jax.convert_functional(div_no_nan, [x, y])
    jax_func = self.variant(jax_func)
    jax_gradient = jax.grad(jax_func)([x, y])

    with self.subTest("check_forward_pass"):
      self.assertAllClose(jax_func([x, y]), np.asarray(output))
    with self.subTest("check_backward_pass"):
      self.assertAllClose(jax_gradient, np.asarray(tf_gradient))

  @chex.variants(with_jit=True, without_jit=True)
  def test_angle(self):
    inputs = np.array([-2.25 + 4.75j, 3.25 + 5.75j], dtype=np.csingle)

    def angle(x):
      return tf.raw_ops.Angle(input=x)
    self._test_convert(angle, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_rfft(self):
    inputs = np.array([2.25, 3.25] * 2, dtype=np.single)

    def rfft(x):
      return tf.raw_ops.RFFT(input=x, fft_length=[len(x)])
    self._test_convert(rfft, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_irfft(self):
    inputs = np.array([-2.25 + 4.75j, 3.25 + 5.75j] * 2, dtype=np.csingle)

    def irfft(x):
      return tf.raw_ops.IRFFT(input=x, fft_length=[len(x)])
    self._test_convert(irfft, inputs)

  @chex.variants(with_jit=True, without_jit=True)
  def test_var_handle(self):
    def var_handle():
      return tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=(3, 5), name="blah")

    with self.assertRaisesRegex(
        ValueError, "VarHandleOp `blah` cannot be evaluated"
    ):
      self._test_convert(var_handle, [])

  @chex.variants(with_jit=True, without_jit=True)
  @parameterized.parameters(
      "LowerBound",
      "UpperBound",
      "SearchSorted",
  )
  def test_lower_upper_bound(self, op_name):
    np.random.seed(42)
    inputs = (
        np.array([[0, 1, 2, 3, 4], [-4, -3, -2, -1, 0]], dtype=np.float32),
        np.array(
            [[3.5, 0, 1.5, 10, -1], [-3.5, 0, -1.5, -10, 1]], dtype=np.float32)
    )
    if op_name == "SearchSorted":
      tf_func = lambda x, y: tf.searchsorted(x, y, out_type=tf.int32)
    else:
      # LowerBound and UpperBound expect keyword arguments.
      def tf_func(x, y):
        return getattr(tf.raw_ops, op_name)(sorted_inputs=x, values=y)
    self._test_convert(tf_func, inputs)


if __name__ == "__main__":
  tf.test.main()
