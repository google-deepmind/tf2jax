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
"""Experimental functions for converting TF graphs to Jax functions."""

import collections
import contextlib
import dataclasses
import functools
import inspect
import itertools
from typing import Any, Callable, Iterator, Optional, Mapping, NamedTuple, Sequence, Set, Tuple, Union

from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tf2jax._src import numpy_compat as anp
from tf2jax._src import utils
from tf2jax._src import xla_utils
import tree

# Import usage logging here.

from tensorflow.python.framework import op_def_registry  # pylint: disable=no-name-in-module
from tensorflow.python.framework import ops as tf_ops  # pylint: disable=no-name-in-module


# NoOp inserted to trigger side effects in function with no return values.
_EMPTY_RETURN_OP_NAME = "__NO_RETURN__"
_EMPTY_RETURN_VALUE = object()

_UNUSED_INPUT = object()

_config = dict(
    strict_shape_check=True,
    strict_dtype_check=False,
    force_const_float32_to_bfloat16=False,
    force_const_float64_to_bfloat16=False,
    convert_custom_gradient=True,
    infer_relu_from_jax2tf=True,
)


def get_config(name: str):
  return _config[name]


def update_config(name: str, value: Any):
  if name in _config:
    _config[name] = value
  else:
    raise ValueError(
        f"Parameter named {name} not found in config={_config}")


@contextlib.contextmanager
def override_config(name: str, value: Any):
  old_value = get_config(name)
  update_config(name, value)
  try:
    yield
  finally:
    update_config(name, old_value)


def _check_attrs(proto, expected: Set[str]):
  unexpected = []
  for k, v in proto.attr.items():
    # Ignore attributes with "_" prefix, as they appear to be undocumented.
    if k not in expected and not k.startswith("_"):
      unexpected.append("  `" + f"{k}={v}".strip() + "`")
  if unexpected:
    raise ValueError("\n".join(
        [f"Unexpected attr(s) when parsing {proto.op}: {proto.name}"] +
        unexpected))


def _get_jax_op(
    jax_op: Callable[..., Any],
    expected_attrs: Set[str],
) -> Callable[..., Any]:
  """For wrapping simple ops with no optional parameters."""

  def wrapped(proto):
    _check_attrs(proto, expected_attrs)
    return jax_op

  return wrapped


def _fix_jax_poly_shape(shape: Tuple[Any, ...]) -> Tuple[Any, ...]:
  good_shape = []
  for dim in shape:
    try:
      # This catches _DimPolynomial from jax2tf.
      tf.compat.v1.Dimension(dim)
      good_shape.append(dim)
    except TypeError:
      good_shape.append(None)
  return tuple(good_shape)


_jax_ops = {
    "Abs": _get_jax_op(jnp.abs, {"T"}),
    "Add": _get_jax_op(anp.add, {"T"}),
    "AddN": _get_jax_op(
        lambda *args: anp.sum_(anp.stack(args, axis=0), axis=0, keepdims=False),
        {"T", "N"}),
    "AddV2": _get_jax_op(anp.add, {"T"}),
    "ArgMax": _get_jax_op(jnp.argmax, {"T", "Tidx", "output_type"}),
    "ArgMin": _get_jax_op(jnp.argmin, {"T", "Tidx", "output_type"}),
    "Acosh": _get_jax_op(jnp.arccosh, {"T"}),
    "Asinh": _get_jax_op(jnp.arcsinh, {"T"}),
    "Atanh": _get_jax_op(jnp.arctanh, {"T"}),
    "Atan2": _get_jax_op(jnp.arctan2, {"T"}),
    "BitwiseAnd": _get_jax_op(jnp.bitwise_and, {"T"}),
    "BitwiseOr": _get_jax_op(jnp.bitwise_or, {"T"}),
    "BitwiseXor": _get_jax_op(jnp.bitwise_xor, {"T"}),
    "BroadcastTo": _get_jax_op(anp.broadcast_to, {"T", "Tidx"}),
    "Ceil": _get_jax_op(jnp.ceil, {"T"}),
    "Complex": _get_jax_op(jax.lax.complex, {"T", "Tout"}),
    "ComplexAbs": _get_jax_op(jax.lax.abs, {"T", "Tout"}),
    "Conj": _get_jax_op(jax.lax.conj, {"T", "Tout"}),
    "Cos": _get_jax_op(jnp.cos, {"T"}),
    "Cosh": _get_jax_op(jnp.cosh, {"T"}),
    "Digamma": _get_jax_op(jax.lax.digamma, {"T"}),
    "Div": _get_jax_op(anp.divide, {"T"}),
    "Elu": _get_jax_op(jax.nn.elu, {"T"}),
    "Equal": _get_jax_op(anp.equal, {"T", "incompatible_shape_error"}),
    "Erf": _get_jax_op(jax.lax.erf, {"T"}),
    "Erfc": _get_jax_op(jax.lax.erfc, {"T"}),
    "Erfinv": _get_jax_op(jax.lax.erf_inv, {"T"}),
    "Exp": _get_jax_op(jnp.exp, {"T"}),
    "Expm1": _get_jax_op(jnp.expm1, {"T"}),
    "ExpandDims": _get_jax_op(anp.expand_dims, {"T", "Tdim"}),
    "Floor": _get_jax_op(jnp.floor, {"T"}),
    "FloorMod": _get_jax_op(anp.mod, {"T"}),
    "FloorDiv": _get_jax_op(anp.floor_divide, {"T"}),
    "Greater": _get_jax_op(anp.greater, {"T"}),
    "GreaterEqual": _get_jax_op(anp.greater_equal, {"T"}),
    "Identity": _get_jax_op(lambda x: x, {"T"}),
    "Igamma": _get_jax_op(jax.lax.igamma, {"T"}),
    "Igammac": _get_jax_op(jax.lax.igammac, {"T"}),
    "Imag": _get_jax_op(jax.lax.imag, {"T", "Tout"}),
    "IsFinite": _get_jax_op(jnp.isfinite, {"T"}),
    "Invert": _get_jax_op(jnp.bitwise_not, {"T"}),
    "L2Loss": _get_jax_op(lambda x: 0.5 * jnp.sum(jnp.square(x)), {"T"}),
    "LeftShift": _get_jax_op(jnp.left_shift, {"T"}),
    "Less": _get_jax_op(anp.less, {"T", "incompatible_shape_error"}),
    "LessEqual": _get_jax_op(anp.less_equal, {"T", "incompatible_shape_error"}),
    "Lgamma": _get_jax_op(jax.lax.lgamma, {"T"}),
    "Log": _get_jax_op(jnp.log, {"T"}),
    "Log1p": _get_jax_op(jnp.log1p, {"T"}),
    "LogicalAnd": _get_jax_op(jnp.logical_and, {"T"}),
    "LogicalNot": _get_jax_op(jnp.logical_not, {"T"}),
    "LogicalOr": _get_jax_op(jnp.logical_or, {"T"}),
    "Minimum": _get_jax_op(anp.minimum, {"T"}),
    "Maximum": _get_jax_op(anp.maximum, {"T"}),
    "Mul": _get_jax_op(anp.multiply, {"T"}),
    "Neg": _get_jax_op(anp.negative, {"T"}),
    "NoOp": _get_jax_op(lambda: _EMPTY_RETURN_VALUE, set({})),
    "NotEqual": _get_jax_op(anp.not_equal, {"T", "incompatible_shape_error"}),
    "OnesLike": _get_jax_op(jnp.ones_like, {"T"}),
    "Pow": _get_jax_op(anp.power, {"T"}),
    "Real": _get_jax_op(jax.lax.real, {"T", "Tout"}),
    "ReadVariableOp": _get_jax_op(lambda x: x, {"dtype"}),
    "RealDiv": _get_jax_op(anp.true_divide, {"T"}),
    "Reciprocal": _get_jax_op(anp.reciprocal, {"T"}),
    "Relu": _get_jax_op(jax.nn.relu, {"T"}),
    "Relu6": _get_jax_op(jax.nn.relu6, {"T"}),
    "ReverseV2": _get_jax_op(anp.flip, {"T", "Tidx"}),
    "RightShift": _get_jax_op(jnp.right_shift, {"T"}),
    "Round": _get_jax_op(jnp.round, {"T"}),
    "Rsqrt": _get_jax_op(jax.lax.rsqrt, {"T"}),
    "Shape": _get_jax_op(lambda x: np.array(jnp.shape(x)), {"T", "out_type"}),
    "Sigmoid": _get_jax_op(jax.nn.sigmoid, {"T"}),
    "Sign": _get_jax_op(jnp.sign, {"T"}),
    "Sin": _get_jax_op(jnp.sin, {"T"}),
    "Sinh": _get_jax_op(jnp.sinh, {"T"}),
    "Size": _get_jax_op(lambda x: np.prod(jnp.shape(x), dtype=np.int32),
                        {"T", "out_type"}),
    "Softplus": _get_jax_op(jax.nn.softplus, {"T"}),
    "Sqrt": _get_jax_op(jnp.sqrt, {"T"}),
    "Square": _get_jax_op(jnp.square, {"T"}),
    "StopGradient": _get_jax_op(jax.lax.stop_gradient, {"T"}),
    "Sub": _get_jax_op(anp.subtract, {"T"}),
    "Tan": _get_jax_op(jnp.tan, {"T"}),
    "Tanh": _get_jax_op(jnp.tanh, {"T"}),
    "Tile": _get_jax_op(anp.tile, {"T", "Tmultiples"}),
    "Where": _get_jax_op(jnp.argwhere, {"T"}),
    "ZerosLike": _get_jax_op(jnp.zeros_like, {"T"}),
    # The assignment logic is handled in _OpNode and convert().
    "AssignAddVariableOp": _get_jax_op(jnp.add, {"dtype"}),
    "AssignSubVariableOp": _get_jax_op(jnp.subtract, {"dtype"}),
    "AssignVariableOp": _get_jax_op(
        lambda var, x: x, {"dtype", "validate_shape"}),
}


def _register(func: Callable[..., Any], op_name: str):
  curr_func = _jax_ops.get(op_name, None)
  if curr_func is None:
    _jax_ops[op_name] = func
  else:
    if curr_func != func:
      raise ValueError(
          f"{op_name} is already registered as {curr_func}, received {func}.")

  return func


def register_operation(op_name):
  return functools.partial(_register, op_name=op_name)


@dataclasses.dataclass
class _HigherOrderFunction:
  """Base class for higher order ops."""
  inner_fn_names: Mapping[str, str]

  def get_inner_functions(
      self, library_functions: Mapping[str, Callable[..., Any]]
  ) -> Mapping[str, Callable[..., Any]]:
    return {k: library_functions[v] for k, v in self.inner_fn_names.items()}


@register_operation("All")
def _all(proto):
  _check_attrs(proto, {"Tidx", "keep_dims"})
  keep_dims = proto.attr["keep_dims"].b
  return lambda x, axis: anp.all_(x, axis=axis.tolist(), keepdims=keep_dims)


@register_operation("Any")
def _any(proto):
  _check_attrs(proto, {"Tidx", "keep_dims"})
  keep_dims = proto.attr["keep_dims"].b
  return lambda x, axis: anp.any_(x, axis=axis.tolist(), keepdims=keep_dims)


@register_operation("Assert")
def _assert(proto):
  _check_attrs(proto, {"T", "summarize"})

  logging.warning("Assert has no effect and will just return the data.")

  return lambda cond, data: data


@register_operation("AvgPool")
def _avg_pool(proto):
  """Parse a AvgPool Op."""
  _check_attrs(
      proto,
      {"T", "padding", "explicit_paddings", "ksize", "strides", "data_format"})

  explicit_paddings = tuple(proto.attr["explicit_paddings"].list.i)
  if explicit_paddings:
    raise ValueError("explicit_padding in AvgPool not yet supported.")

  padding = str(proto.attr["padding"].s, "utf-8")
  ksize = tuple(proto.attr["ksize"].list.i)
  strides = tuple(proto.attr["strides"].list.i)
  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format not in ("NHWC", "NCHW"):
    raise ValueError(f"Found unsupported data format {data_format}.")

  reduce_window_args = dict(
      init_value=0.,
      computation=jax.lax.add,
      window_dimensions=ksize,
      window_strides=strides,
      padding=padding)

  def _func(x: jnp.ndarray) -> jnp.ndarray:
    pooled = jax.lax.reduce_window(x, **reduce_window_args)
    if padding == "VALID":
      window_counts = np.prod(ksize)
    else:
      window_counts = jax.lax.reduce_window(
          jnp.ones_like(x), **reduce_window_args)
    return pooled / window_counts

  return _func


@register_operation("BiasAdd")
def _bias_add(proto):
  """Parse a BiasAdd Op."""
  _check_attrs(proto, {"T", "data_format"})

  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format == "NHWC":
    reduce_axis = (0, 1, 2)
  elif data_format == "NCHW":
    reduce_axis = (0, 2, 3)
  else:
    raise ValueError(f"Found unsupported data format {data_format}.")

  def _func(value: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    if bias.ndim != 1:
      raise ValueError(
          f"Expected `bias` as a 1D array, found array with {bias.ndim} dims.")
    bias = anp.expand_dims(bias, axis=reduce_axis)
    return anp.add(value, bias)

  return _func


@register_operation("Bitcast")
def _bit_cast(proto):
  _check_attrs(proto, {"T", "type"})
  dst_type = tf.as_dtype(proto.attr["type"].type)
  return lambda x: jax.lax.bitcast_convert_type(x, anp.get_jax_dtype(dst_type))


@register_operation("BroadcastArgs")
def _broadcast_args(proto):
  _check_attrs(proto, {"T"})
  return lambda s0, s1: np.array(np.broadcast(np.zeros(s0), np.zeros(s1)).shape)


class _CaseOp(_HigherOrderFunction):
  """Represents a Case Op."""

  def __call__(self, branch_index, *operand, **branch_fns):
    def create_branch(fn):
      return lambda args: fn(*args)
    branches = [create_branch(fn) for _, fn in sorted(branch_fns.items())]
    return jax.lax.switch(branch_index, branches=branches, operand=operand)


@register_operation("StatelessCase")
@register_operation("Case")
def _case(proto):
  """Parse a Case op."""
  _check_attrs(proto, {"Tin", "Tout", "output_shapes", "branches"})

  branches = [f.name for f in proto.attr["branches"].list.func]
  output_shapes = [
      [d.size for d in xs.dim] for xs in proto.attr["output_shapes"].list.shape
  ]
  del output_shapes

  return _CaseOp({f"fn_{k:06}": v for k, v in enumerate(branches)})


@register_operation("Cast")
def _cast(proto):
  """Parse a Cast Op."""
  _check_attrs(proto, {"SrcT", "DstT", "Truncate"})

  src_type = tf.as_dtype(proto.attr["SrcT"].type)
  dst_type = tf.as_dtype(proto.attr["DstT"].type)
  truncate = proto.attr["Truncate"].b
  del src_type

  if truncate:
    raise ValueError(f"Cast does not support truncate={truncate}.")

  def _func(x: jnp.ndarray) -> jnp.ndarray:
    return anp.asarray(x, dst_type)

  return _func


@register_operation("ConjugateTranspose")
def _conjugate_transpose(proto):
  _check_attrs(proto, {"T", "Tperm"})
  return lambda x, axes: jax.lax.conj(jnp.transpose(x, axes=axes))


@register_operation("ConcatV2")
def _concatenate(proto):
  """Parse a ConcatV2 Op."""
  _check_attrs(proto, {"T", "Tidx", "N"})

  num_arrays = proto.attr["N"].i

  def _func(*args) -> jnp.ndarray:
    if len(args) != num_arrays + 1:
      raise ValueError(
          f"Concatenate expects {num_arrays} args, received {len(args)}.")

    *inputs, axis = args
    return anp.concatenate(inputs, axis=axis)

  return _func


@register_operation("Const")
def _const(proto):
  """Parse a Const Op."""
  _check_attrs(proto, {"dtype", "value"})
  value = tf.make_ndarray(proto.attr["value"].tensor)
  dtype = value.dtype

  force_float32 = get_config("force_const_float32_to_bfloat16")
  force_float64 = get_config("force_const_float64_to_bfloat16")
  if ((force_float32 and dtype == np.float32) or
      (force_float64 and dtype == np.float64)):
    # NOTE: `jnp.asarray` (rather than the `np` version) cannot be used here;
    # using it in a jitted context can produce runtime `UnexpectedTracerError`s.
    bf16_value = np.asarray(value, dtype=jnp.bfloat16)
    logging.warning("Converting float consts to bfloat16, from %s, to %s.",
                    value, bf16_value)
    value = bf16_value

  return lambda: value


@register_operation("Conv2D")
def _conv2d(proto):
  """Parse a Conv2D Op."""
  _check_attrs(
      proto, {
          "T", "padding", "explicit_paddings", "dilations", "strides",
          "data_format", "use_cudnn_on_gpu"
      })

  explicit_paddings = tuple(proto.attr["explicit_paddings"].list.i)
  if explicit_paddings:
    raise ValueError("explicit_padding in Conv2D not yet supported.")

  padding = str(proto.attr["padding"].s, "utf-8")
  dilations = tuple(proto.attr["dilations"].list.i)
  strides = tuple(proto.attr["strides"].list.i)
  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format == "NHWC":
    dimension_numbers = ("NHWC", "HWIO", "NHWC")
    strides = xla_utils.get_conv_sequence(strides, ndim=2, channel_index=-1)
    dilations = xla_utils.get_conv_sequence(dilations, ndim=2, channel_index=-1)
    feature_group_count_fn = lambda lhs, rhs: lhs.shape[3] // rhs.shape[2]
  elif data_format == "NCHW":
    dimension_numbers = ("NCHW", "HWIO", "NCHW")
    strides = xla_utils.get_conv_sequence(strides, ndim=2, channel_index=1)
    dilations = xla_utils.get_conv_sequence(dilations, ndim=2, channel_index=1)
    feature_group_count_fn = lambda lhs, rhs: lhs.shape[1] // rhs.shape[2]
  else:
    raise ValueError(f"Found unsupported data format {data_format}.")

  _ = proto.attr["use_cudnn_on_gpu"].b

  def _func(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    feature_group_count = feature_group_count_fn(lhs, rhs)
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
        rhs_dilation=dilations,
        feature_group_count=feature_group_count)

  return _func


@register_operation("Conv2DBackpropInput")
def _conv2d_backprop_input(proto):
  """Parse a Conv2DBackpropInput Op."""
  _check_attrs(
      proto, {
          "T", "padding", "explicit_paddings", "dilations", "strides",
          "data_format", "use_cudnn_on_gpu"
      })

  explicit_paddings = tuple(proto.attr["explicit_paddings"].list.i)
  if explicit_paddings:
    raise ValueError(
        "explicit_padding in Conv2DBackpropInput not yet supported.")

  padding = str(proto.attr["padding"].s, "utf-8")
  dilations = tuple(proto.attr["dilations"].list.i)
  strides = tuple(proto.attr["strides"].list.i)
  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format == "NHWC":
    dimension_numbers = ("NHWC", "HWIO", "NHWC")
    strides = xla_utils.get_conv_sequence(strides, ndim=2, channel_index=-1)
    dilations = xla_utils.get_conv_sequence(dilations, ndim=2, channel_index=-1)
  elif data_format == "NCHW":
    dimension_numbers = ("NCHW", "HWIO", "NCHW")
    strides = xla_utils.get_conv_sequence(strides, ndim=2, channel_index=1)
    dilations = xla_utils.get_conv_sequence(dilations, ndim=2, channel_index=1)
  else:
    raise ValueError(f"Found unsupported data format {data_format}.")

  _ = proto.attr["use_cudnn_on_gpu"].b

  def _func(
      input_sizes: jnp.ndarray,
      filters: jnp.ndarray,
      out_backprop: jnp.ndarray,
  ) -> jnp.ndarray:
    del input_sizes
    return jax.lax.conv_transpose(
        out_backprop,
        filters,
        strides=strides,
        padding=padding,
        rhs_dilation=dilations,
        transpose_kernel=True,
        dimension_numbers=dimension_numbers)

  return _func


@register_operation("Cumsum")
def _cumsum(proto):
  """Parse a Cumsum Op."""
  _check_attrs(proto, {"T", "Tidx", "exclusive", "reverse"})

  exclusive = proto.attr["exclusive"].b
  reverse = proto.attr["reverse"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    axis: int = axis.tolist()
    if reverse:
      x = anp.flip(x, axis=axis)
    if exclusive:
      pad_shape = list(x.shape)
      pad_shape[axis] = 1
      x = anp.concatenate([np.zeros(pad_shape, dtype=x.dtype), x], axis=axis)
      x = x[(slice(None),) * axis + (slice(0, -1), Ellipsis)]
    res = anp.cumsum(x, axis=axis)
    if reverse:
      res = anp.flip(res, axis=axis)
    return res

  return _func


@register_operation("DepthwiseConv2dNative")
def _depthwise_conv2d(proto):
  """Parse a DepthwiseConv2d Op."""
  _check_attrs(proto, {
      "T", "strides", "dilations", "padding", "data_format", "explicit_paddings"
  })

  explicit_paddings = tuple(proto.attr["explicit_paddings"].list.i)
  if explicit_paddings:
    explicit_paddings = [
        tuple(x) for x in np.array(explicit_paddings).reshape(4, 2).tolist()
    ]

  padding = explicit_paddings or str(proto.attr["padding"].s, "utf-8")
  dilations = tuple(proto.attr["dilations"].list.i)
  strides = tuple(proto.attr["strides"].list.i)
  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format == "NHWC":
    if explicit_paddings:
      padding = padding[1:3]
    dimension_numbers = ("NHWC", "HWIO", "NHWC")
    strides = xla_utils.get_conv_sequence(strides, ndim=2, channel_index=-1)
    dilations = xla_utils.get_conv_sequence(dilations, ndim=2, channel_index=-1)
    channel_index = -1
  elif data_format == "NCHW":
    if explicit_paddings:
      padding = padding[2:]
    dimension_numbers = ("NCHW", "HWIO", "NCHW")
    strides = xla_utils.get_conv_sequence(strides, ndim=2, channel_index=1)
    dilations = xla_utils.get_conv_sequence(dilations, ndim=2, channel_index=1)
    channel_index = 1
  else:
    raise ValueError(f"Found unsupported data format {data_format}.")

  def _func(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    output_dim = rhs.shape[2] * rhs.shape[3]
    return jax.lax.conv_general_dilated(
        lhs,
        jnp.reshape(rhs, rhs.shape[:2] + (1, output_dim)),
        window_strides=strides,
        padding=padding,
        dimension_numbers=dimension_numbers,
        rhs_dilation=dilations,
        feature_group_count=lhs.shape[channel_index])

  return _func


@register_operation("Einsum")
def _einsum(proto):
  """Parse an Einsum Op."""
  _check_attrs(proto, {"T", "N", "equation"})

  num_inputs = proto.attr["N"].i
  equation = str(proto.attr["equation"].s, "utf-8")

  def _func(*operands):
    if len(operands) != num_inputs:
      raise ValueError(
          f"Expected {num_inputs} input arrays, found {len(operands)}")
    return jnp.einsum(equation, *operands)

  return _func


@register_operation("Empty")
def _empty(proto):
  """Parse an Empty op."""
  _check_attrs(proto, {"dtype", "init"})

  dtype = tf.as_dtype(proto.attr["dtype"].type)
  init = proto.attr["init"].b

  def _func(shape: jnp.ndarray) -> jnp.ndarray:
    return anp.empty(shape=shape, dtype=dtype, init=init)

  return _func


@register_operation("Fill")
def _fill(proto):
  """Parse an Fill op."""
  _check_attrs(proto, {"T", "index_type"})

  dtype = tf.as_dtype(proto.attr["T"].type)

  def _func(shape: jnp.ndarray, fill_value: jnp.ndarray) -> jnp.ndarray:
    return anp.full(shape=shape, fill_value=fill_value, dtype=dtype)

  return _func


@register_operation("FusedBatchNormV3")
@register_operation("FusedBatchNormV2")
def _fused_batch_norm(proto):
  """Parse a FusedBatchNorm Op."""
  _check_attrs(proto, {
      "T", "U", "data_format", "epsilon", "exponential_avg_factor",
      "is_training"
  })

  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format == "NHWC":
    reduce_axis = (0, 1, 2)
    channel_dim = 3
  elif data_format == "NCHW":
    reduce_axis = (0, 2, 3)
    channel_dim = 1
  else:
    raise ValueError(f"Found unsupported data format {data_format}.")

  epsilon = proto.attr["epsilon"].f
  exponential_avg_factor = proto.attr["exponential_avg_factor"].f
  one_minus_factor = 1. - exponential_avg_factor
  is_training = proto.attr["is_training"].b

  def _func(
      x: jnp.ndarray,
      scale: jnp.ndarray,
      offset: jnp.ndarray,
      running_mean: jnp.ndarray,
      running_var: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    batch_mean = jnp.mean(x, axis=reduce_axis)
    batch_var = jnp.var(x, axis=reduce_axis)
    est_mean = batch_mean if is_training else running_mean
    est_var = batch_var if is_training else running_var

    # Prep for broadcasting.
    scale = jnp.expand_dims(scale, axis=reduce_axis)
    offset = jnp.expand_dims(offset, axis=reduce_axis)
    est_mean = jnp.expand_dims(est_mean, axis=reduce_axis)
    est_var = jnp.expand_dims(est_var, axis=reduce_axis)

    inv = scale * jax.lax.rsqrt(est_var + epsilon)
    norm_x = jnp.asarray((x - est_mean) * inv + offset, x.dtype)

    if is_training:
      # Apply Bessel's correction and additional smoothing.
      ndata = x.size / x.shape[channel_dim]
      correction = ndata / jnp.maximum(ndata - 1.0, 1.0)
      running_var = running_var if running_var.size else 0
      running_mean = running_mean if running_mean.size else 0
      new_var = (
          one_minus_factor * running_var +
          exponential_avg_factor * batch_var * correction)
      new_mean = (
          one_minus_factor * running_mean +
          exponential_avg_factor * batch_mean)
      return norm_x, new_mean, new_var
    else:
      return norm_x, running_mean, running_var

  return _func


@register_operation("GatherNd")
def _gather_nd(proto):
  """Parse a GatherNd Op."""
  _check_attrs(proto, {"Tindices", "Tparams"})

  def _func(params: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return params[tuple(anp.moveaxis(indices, -1, 0))]

  return _func


@register_operation("GatherV2")
def _gather(proto):
  """Parse a GatherV2 Op."""
  _check_attrs(proto, {"Taxis", "Tindices", "Tparams", "batch_dims"})

  batch_dims = proto.attr["batch_dims"].i
  if batch_dims < 0:
    raise ValueError(f"batch_dims={batch_dims} must be non-negative.")

  def _func(
      params: jnp.ndarray,
      indices: jnp.ndarray,
      axis: jnp.ndarray,
  ) -> jnp.ndarray:
    return anp.gather(
        params, indices, axis=axis.tolist(), batch_dims=batch_dims)

  return _func


@dataclasses.dataclass
class _IdentityN(_HigherOrderFunction):
  """Represents a IdentityN Op."""

  gradient_op_type: str  # For debug, custom_gradient is handled by _Subgraph.

  def __call__(self, *args):
    return args


@register_operation("IdentityN")
def _identity_n(proto):
  """Parse a IdentityN Op."""
  _check_attrs(proto, {"T"})

  gradient_op_type = str(proto.attr["_gradient_op_type"].s, "utf-8")
  if gradient_op_type:
    logging.info("Found custom gradient %s", gradient_op_type)

  return _IdentityN({}, gradient_op_type=gradient_op_type)


class _IfOp(_HigherOrderFunction):
  """Represents a If Op."""

  def __call__(self, pred, *operand, then_fun, else_fun):
    true_fun = lambda args: then_fun(*args)
    false_fun = lambda args: else_fun(*args)
    return jax.lax.cond(
        pred, true_fun=true_fun, false_fun=false_fun, operand=operand)


@register_operation("StatelessIf")
@register_operation("If")
def _if(proto):
  """Parse a If op."""
  _check_attrs(proto, {
      "Tcond", "Tin", "Tout", "output_shapes", "then_branch", "else_branch"
  })

  then_name = proto.attr["then_branch"].func.name
  else_name = proto.attr["else_branch"].func.name
  output_shapes = [
      [d.size for d in xs.dim] for xs in proto.attr["output_shapes"].list.shape
  ]
  del output_shapes

  return _IfOp(dict(then_fun=then_name, else_fun=else_name))


@register_operation("InplaceAdd")
def _inplace_add(proto):
  """Parse a InplaceAdd op."""
  _check_attrs(proto, {"T"})

  def _func(
      inputs: jnp.ndarray,
      indices: jnp.ndarray,
      updates: jnp.ndarray,
  ) -> jnp.ndarray:
    return jnp.asarray(inputs).at[indices].add(updates)

  return _func


@register_operation("InplaceUpdate")
def _inplace_update(proto):
  """Parse a InplaceUpdate op."""
  _check_attrs(proto, {"T"})

  def _func(
      inputs: jnp.ndarray,
      indices: jnp.ndarray,
      updates: jnp.ndarray,
  ) -> jnp.ndarray:
    return jnp.asarray(inputs).at[indices].set(updates)

  return _func


@register_operation("LogSoftmax")
def _log_softmax(proto):
  _check_attrs(proto, {"T"})
  return lambda x: jax.nn.log_softmax(x, axis=-1)


@register_operation("MatMul")
def _matmul(proto):
  """Parse a MatMul Op."""
  _check_attrs(proto, {"T", "transpose_a", "transpose_b"})

  transpose_a = proto.attr["transpose_a"].b
  transpose_b = proto.attr["transpose_b"].b

  def _func(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    if transpose_a:
      a = jnp.transpose(a)
    if transpose_b:
      b = jnp.transpose(b)

    return jnp.matmul(a, b)

  return _func


@register_operation("BatchMatMulV2")
def _batch_matmul(proto):
  """Parse a BatchMatMul Op."""
  _check_attrs(proto, {"T", "adj_x", "adj_y"})

  adj_x = proto.attr["adj_x"].b
  adj_y = proto.attr["adj_y"].b

  # TODO(shaobohou) Add test for arrays with complex values.
  def _func(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    if adj_x:
      x = jnp.conjugate(jnp.swapaxes(x, -1, -2))
    if adj_y:
      y = jnp.conjugate(jnp.swapaxes(y, -1, -2))

    return jnp.matmul(x, y)

  return _func


@register_operation("MatrixDiagV3")
def _matrix_diag(proto):
  """Parse a MatrixDiagV3 op."""
  _check_attrs(proto, {"T", "align"})

  align = str(proto.attr["align"].s, "utf-8")
  if align != "RIGHT_LEFT":
    raise ValueError(f"MatrixDiagV3 does not support `align={align}` yet.")

  def _func(
      diagonals: jnp.ndarray,
      k: jnp.ndarray,
      num_rows: jnp.ndarray,
      num_cols: jnp.ndarray,
      padding_value: jnp.ndarray,
  ) -> jnp.ndarray:
    if num_rows != -1 or num_cols != -1:
      raise ValueError(f"MatrixDiagV3 does not yet support num_rows={num_rows} "
                       f"or num_cols={num_cols}.")

    diag_fn = lambda inputs: jnp.diagflat(inputs, k=k)
    for _ in range(len(diagonals.shape) - 1):
      diag_fn = jax.vmap(diag_fn)

    outputs = diag_fn(diagonals)
    mask = diag_fn(jnp.ones_like(diagonals, dtype=jnp.bool_))
    return jnp.where(mask, outputs, padding_value)

  return _func


@register_operation("MatrixBandPart")
def _matrix_band_part(proto):
  """Parse a MatrixBandPart op."""
  _check_attrs(proto, {"T", "Tindex"})

  def _func(
      x: jnp.ndarray,
      lower: jnp.ndarray,
      upper: jnp.ndarray,
  ) -> jnp.ndarray:
    if len(x.shape) < 2:
      raise ValueError(
          f"Expected input of at least rank 2, found {len(x.shape)}")
    mask_shape = x.shape[-2:]
    lower = lower.tolist() + 1 if lower.tolist() >= 0 else max(mask_shape)
    mask_lower = jnp.tril(jnp.ones(mask_shape, jnp.int32), -lower)
    upper = upper.tolist() + 1 if upper.tolist() >= 0 else max(mask_shape)
    mask_upper = jnp.triu(jnp.ones(mask_shape, jnp.int32), upper)
    return jnp.where((mask_lower + mask_upper) == 0, x, 0)

  return _func


@register_operation("Max")
def _max(proto):
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.max_(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("MaxPool")
def _max_pool(proto):
  """Parse a MaxPool Op."""
  _check_attrs(
      proto,
      {"T", "padding", "explicit_paddings", "ksize", "strides", "data_format"})

  explicit_paddings = tuple(proto.attr["explicit_paddings"].list.i)
  if explicit_paddings:
    raise ValueError("explicit_padding in MaxPool not yet supported.")

  padding = str(proto.attr["padding"].s, "utf-8")
  ksize = tuple(proto.attr["ksize"].list.i)
  strides = tuple(proto.attr["strides"].list.i)
  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format not in ("NHWC", "NCHW"):
    raise ValueError(f"Found unsupported data format {data_format}.")

  def _func(x: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=jax.lax.max,
        window_dimensions=ksize,
        window_strides=strides,
        padding=padding)

  return _func


@register_operation("Mean")
def _mean(proto):
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("Min")
def _min(proto):
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.min_(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("OneHot")
def _one_hot(proto):
  """Parse a OneHot Op."""
  _check_attrs(proto, {"T", "TI", "axis"})

  axis = proto.attr["axis"].i

  def _func(
      indices: jnp.ndarray,
      depth: jnp.ndarray,
      on_value: jnp.ndarray,
      off_value: jnp.ndarray,
  ) -> jnp.ndarray:
    if axis != -1 and axis != len(indices.shape):
      raise ValueError(f"OneHot does not support axis={axis} yet, "
                       f"indices.shape={indices.shape}.")

    mask = jax.nn.one_hot(indices, num_classes=depth, dtype=jnp.int32)
    return mask * on_value + (1 - mask) * off_value

  return _func


@register_operation("Pack")
def _pack(proto):
  """Parse a Pack op."""
  _check_attrs(proto, {"T", "axis", "N"})

  num_arrays = proto.attr["N"].i
  axis = proto.attr["axis"].i

  def _func(*args) -> jnp.ndarray:
    if len(args) != num_arrays:
      raise ValueError(
          f"Pack expects {num_arrays} args, received {len(args)}.")
    return anp.stack(args, axis=axis)

  return _func


@register_operation("Pad")
def _pad(proto):
  _check_attrs(proto, {"T", "Tpaddings"})
  return lambda x, paddings: jnp.pad(x, pad_width=paddings)


@register_operation("PadV2")
def _pad_v2(proto):
  """Parse a PadV2 op."""
  _check_attrs(proto, {"T", "Tpaddings"})

  def _func(
      inputs: jnp.ndarray,
      padding: jnp.ndarray,
      constant_values: jnp.ndarray,
  ) -> jnp.ndarray:
    return jnp.pad(inputs, pad_width=padding, constant_values=constant_values)

  return _func


class _PartitionedCall(_HigherOrderFunction):
  """Represents a PartitionedCall Op."""

  def __call__(self, *args, inner_fn, rng=None):
    return inner_fn(*args, rng=rng)


# TODO(shaobohou) Add test for StatefulPartitionedCall.
@register_operation("StatefulPartitionedCall")
@register_operation("PartitionedCall")
def _partitioned_call(proto):
  """Parse a PartitionedCall op."""
  _check_attrs(proto,
               {"f", "Tin", "Tout", "config", "config_proto", "executor_type"})

  inner_fn = proto.attr["f"].func.name
  config = str(proto.attr["config"].s, "utf-8")
  config_proto = proto.attr["config_proto"].s  # TODO(shaobohou) decode this?
  executor_type = str(proto.attr["executor_type"].s, "utf-8")
  del config, config_proto, executor_type

  return _PartitionedCall(dict(inner_fn=inner_fn))


@register_operation("Placeholder")
def _placeholder(proto):
  _check_attrs(proto, {"dtype", "shape"})

  name = proto.name

  def _func():
    raise ValueError(f"Placeholder `{name}` cannot be evaluated.")

  return _func


@register_operation("PreventGradient")
def _prevent_gradient(proto):
  """Parse a PreventGradient op."""
  _check_attrs(proto, {"T", "message"})

  message = str(proto.attr["message"].s, "utf-8")
  jax_message = (
      f"Gradient explicitly prevented on node {proto.name}. Reason: {message}")

  @jax.custom_gradient
  def _func(operand: jnp.ndarray) -> jnp.ndarray:
    def grad_fn(_):
      raise LookupError(jax_message)
    return operand, grad_fn

  return _func


@register_operation("Prod")
def _prod(proto):
  """Parse a Prod op."""
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.prod(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("RandomStandardNormal")
def _random_standard_normal(proto):
  """Parse a RandomStandardNormal op."""
  _check_attrs(proto, {"T", "dtype", "seed", "seed2"})

  seed = proto.attr["seed"].i
  seed2 = proto.attr["seed2"].i
  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  if seed != 0 or seed2 != 0:
    logging.warning(
        "RandomStandardNormal does not yet support non-zero seeds, found "
        "seed=%s and seed2=%s.", seed, seed2)

  return lambda shape, *, rng: jax.random.normal(rng, shape, dtype=jax_dtype)


@register_operation("RandomUniform")
def _random_uniform(proto):
  """Parse a RandomUniform op."""
  _check_attrs(proto, {"T", "dtype", "seed", "seed2"})

  seed = proto.attr["seed"].i
  seed2 = proto.attr["seed2"].i
  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  if seed != 0 or seed2 != 0:
    logging.warning(
        "RandomUniform does not yet support non-zero seeds, found "
        "seed=%s and seed2=%s.", seed, seed2)

  return lambda shape, *, rng: jax.random.uniform(rng, shape, dtype=jax_dtype)


@register_operation("RandomUniformInt")
def _random_uniform_int(proto):
  """Parse a RandomUniformInt op."""
  _check_attrs(proto, {"T", "Tout", "seed", "seed2"})

  seed = proto.attr["seed"].i
  seed2 = proto.attr["seed2"].i
  dtype = tf.as_dtype(proto.attr["Tout"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  if seed != 0 or seed2 != 0:
    logging.warning(
        "RandomUniformInt does not yet support non-zero seeds, found "
        "seed=%s and seed2=%s.", seed, seed2)

  def _func(shape, minval, maxval, *, rng):
    return jax.random.randint(
        rng, shape, minval=minval, maxval=maxval, dtype=jax_dtype)

  return _func


@register_operation("Range")
def _range(proto):
  """Parse a Range op."""
  _check_attrs(proto, {"Tidx"})

  dtype = tf.as_dtype(proto.attr["Tidx"].type)

  def _func(
      start: jnp.ndarray,
      limit: jnp.ndarray,
      delta: jnp.ndarray,
  ) -> jnp.ndarray:
    return anp.arange(start, stop=limit, step=delta, dtype=dtype)

  return _func


@register_operation("Reshape")
def _reshape(proto):
  _check_attrs(proto, {"T", "Tshape"})
  return lambda x, shape: jnp.reshape(x, newshape=shape)


@register_operation("ResizeBilinear")
def _resize_linear(proto):
  """Parse a ResizeBilinear op."""
  _check_attrs(proto, {"T", "align_corners", "half_pixel_centers"})

  align_corners = proto.attr["align_corners"].b
  half_pixel_centers = proto.attr["half_pixel_centers"].b
  if align_corners and half_pixel_centers:
    # Not supported by tf.raw_ops.ResizeBilinear.
    raise ValueError(
        "align_corners=True and half_pixel_centers=True are not supported. ")

  def _func(images: jnp.ndarray, size: jnp.ndarray) -> jnp.ndarray:
    if len(images.shape) != 4:
      raise ValueError(
          "Expected A 4D tensor with shape [batch, height, width, channels], "
          f"found {images.shape}")

    inp_batch, inp_height, inp_width, inp_channels = images.shape
    out_height, out_width = size.tolist()

    height_scale = out_height / inp_height
    width_scale = out_width / inp_width
    if align_corners:
      if out_height > 1:
        height_scale = (out_height - 1) / (inp_height - 1)
      if out_width > 1:
        width_scale = (out_width - 1) / (inp_width - 1)
    scale = np.array((height_scale, width_scale))

    translation = np.array(([0.0] * 2))
    if not half_pixel_centers:
      translation = translation - scale * 0.5 + 0.5

    return jax.image.scale_and_translate(
        images,
        shape=(inp_batch, out_height, out_width, inp_channels),
        spatial_dims=(1, 2),
        scale=scale,
        translation=translation,
        method="linear",
        antialias=False,
        precision=None,
    )

  return _func


@register_operation("ScatterNd")
def _scatter_nd(proto):
  """Parse a ScatterNd op."""
  _check_attrs(proto, {"T", "Tindices"})

  def _func(
      indices: jnp.ndarray,
      updates: jnp.ndarray,
      shape: jnp.ndarray,
  ) -> jnp.ndarray:
    zeros = jnp.zeros(shape, updates.dtype)
    key = tuple(jnp.moveaxis(indices, -1, 0))
    return zeros.at[key].set(updates)

  return _func


@register_operation("SelectV2")
@register_operation("Select")
def _select(proto):
  """Parse a Select op."""
  _check_attrs(proto, {"T"})

  def _func(
      conds: jnp.ndarray,
      x: jnp.ndarray,
      y: jnp.ndarray,
  ) -> jnp.ndarray:
    conds = anp.expand_dims(conds, axis=tuple(range(conds.ndim, x.ndim)))
    return anp.where(conds, x, y)

  return _func


@register_operation("Slice")
def _slice(proto):
  """Parse a Slice Op."""
  _check_attrs(proto, {"T", "Index"})

  def _func(
      x: jnp.ndarray,
      begins: jnp.ndarray,
      sizes: jnp.ndarray,
  ) -> jnp.ndarray:
    """`begins` and `sizes` must be concrete arrays."""
    slices = [slice(b, b + s) for b, s in zip(begins, sizes)]
    return x[tuple(slices)]

  return _func


@register_operation("Softmax")
def _softmax(proto):
  _check_attrs(proto, {"T"})
  return lambda x: jax.nn.softmax(x, axis=-1)


@register_operation("SparseSoftmaxCrossEntropyWithLogits")
def _sparse_softmax_cross_entropy_with_logits(proto):
  """Parse a SparseSoftmaxCrossEntropyWithLogits Op."""
  _check_attrs(proto, {"T", "Tlabels"})

  def cross_entropy(xs, ys):
    return -1.0 * jnp.take_along_axis(
        jax.nn.log_softmax(xs, axis=-1), ys[:, jnp.newaxis], axis=-1)[:, 0]

  def _func(features: jnp.ndarray,
            labels: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:

    loss = cross_entropy(features, labels)
    vjp_outputs, vjp_fn = jax.vjp(cross_entropy, features, labels)
    grads = vjp_fn(jnp.ones(vjp_outputs.shape))[0]
    return loss, grads

  return _func


@register_operation("Split")
def _split(proto):
  """Parse a Split op."""
  _check_attrs(proto, {"T", "num_split"})
  num_split = proto.attr["num_split"].i
  return lambda axis, inputs: anp.split(inputs, num_split, axis=axis)


@register_operation("SplitV")
def _splitv(proto):
  """Parse a SplitV op."""
  _check_attrs(proto, {"T", "Tlen", "num_split"})
  num_split = proto.attr["num_split"].i

  def _func(
      value: jnp.ndarray,
      size_splits: jnp.ndarray,
      axis: jnp.ndarray,
  ) -> jnp.ndarray:
    assert size_splits.shape[0] == num_split, (size_splits.shape[0], num_split)
    splits = size_splits.tolist()
    axis = axis.tolist()
    defined_size = sum([x for x in splits if x >= 0])
    splits = [x if x >= 0 else value.shape[axis] - defined_size for x in splits]
    indices = np.cumsum(np.array(splits), axis=0)
    return anp.split(value, indices, axis=axis)

  return _func


@register_operation("SquaredDifference")
def _squared_difference(proto):
  _check_attrs(proto, {"T"})
  return lambda x1, x2: jnp.square(x1 - x2)


@register_operation("Squeeze")
def _squeeze(proto):
  _check_attrs(proto, {"T", "squeeze_dims"})

  axis = tuple(proto.attr["squeeze_dims"].list.i)

  return lambda x: anp.squeeze(x, axis=axis)


@register_operation("StatelessRandomGetKeyCounter")
def _stateless_random_get_key_counter(proto):
  _check_attrs(proto, {"T", "Tseed"})

  def _func(seed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    assert seed.shape == (2,), seed.shape
    seed = jnp.sum(seed)  # An arbitrary choice.
    return jax.random.PRNGKey(seed), jnp.array(0, dtype=jnp.int32)

  return _func


@register_operation("StatelessMultinomial")
def _stateless_multinomial(proto):
  """Parse a StatelessMultinomial op."""
  _check_attrs(proto, {"T", "Tseed", "output_dtype"})

  dtype = tf.as_dtype(proto.attr["output_dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  def _func(
      logits: jnp.ndarray,
      num_samples: jnp.ndarray,
      seed: jnp.ndarray,
  ) -> jnp.ndarray:
    assert seed.shape == (2,), seed.shape
    seed = seed.astype(jnp.uint32)
    shape = (num_samples, logits.shape[0])
    samples = jax.random.categorical(seed, logits, shape=shape)
    return samples.astype(jax_dtype).transpose([1, 0])

  return _func


@register_operation("StatelessRandomNormalV2")
def _stateless_random_normal_v2(proto):
  """Parse a StatelessRandomNormalV2 op."""
  _check_attrs(proto, {"T", "Tshape", "dtype"})

  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  def _func(
      shape: jnp.ndarray,
      key: jnp.ndarray,
      counter: jnp.ndarray,
      alg: jnp.ndarray,
  ) -> jnp.ndarray:
    del counter, alg  # TODO(shaobohou) combine key and counter?
    return jax.random.normal(key=key, shape=shape, dtype=jax_dtype)

  return _func


@register_operation("StatelessRandomUniformV2")
def _stateless_random_uniform_v2(proto):
  """Parse a StatelessRandomNormalV2 op."""
  _check_attrs(proto, {"T", "Tshape", "dtype"})

  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  def _func(
      shape: jnp.ndarray,
      key: jnp.ndarray,
      counter: jnp.ndarray,
      alg: jnp.ndarray,
  ) -> jnp.ndarray:
    del counter, alg  # TODO(shaobohou) combine key and counter?
    return jax.random.uniform(key=key, shape=shape, dtype=jax_dtype)

  return _func


@register_operation("StatelessRandomUniformFullIntV2")
@register_operation("StatelessRandomUniformIntV2")
def _stateless_random_uniform_int_v2(proto):
  """Parse a StatelessRandomUniformIntV2 op."""
  _check_attrs(proto, {"T", "Tshape", "dtype"})

  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  def _func(
      shape: jnp.ndarray,
      key: jnp.ndarray,
      counter: jnp.ndarray,
      alg: jnp.ndarray,
      minval: jnp.ndarray = jnp.iinfo(jax_dtype).min,
      maxval: jnp.ndarray = jnp.iinfo(jax_dtype).max,
  ) -> jnp.ndarray:
    del counter, alg  # TODO(shaobohou) combine key and counter?
    return jax.random.randint(
        key=key, shape=shape, minval=minval, maxval=maxval, dtype=jax_dtype,)

  return _func


class _StatelessWhile(_HigherOrderFunction):
  """Represents a StatelessWhile Op."""

  def __call__(self, *args, cond_fun, body_fun, rng=None):
    def real_cond_fun(args):
      *cond_args, rng = args
      _, cond_key, _ = [None] * 3 if rng is None else jax.random.split(rng, 3)
      outputs = cond_fun(*cond_args, rng=cond_key)
      if len(outputs) != 1:
        raise ValueError(
            f"Expected cond_fun to return a single value, found {outputs}")
      return outputs[0]

    def real_body_fun(args):
      *body_args, rng = args
      key, _, body_key = [None] * 3 if rng is None else jax.random.split(rng, 3)
      outputs = tuple(body_fun(*body_args, rng=body_key))
      return outputs + (key,)

    outputs = jax.lax.while_loop(real_cond_fun, real_body_fun, args + (rng,))
    return outputs


@register_operation("StatelessWhile")
@register_operation("While")
def _stateless_while(proto):
  """Parse a StatelessWhile op."""
  _check_attrs(proto,
               {"T", "body", "cond", "parallel_iterations", "output_shapes"})
  # TODO(shaobohou) Check proto.arg_attr?

  body_name = proto.attr["body"].func.name
  cond_name = proto.attr["cond"].func.name
  parallel_iterations = proto.attr["parallel_iterations"].i
  output_shapes = [
      [d.size for d in xs.dim] for xs in proto.attr["output_shapes"].list.shape
  ]
  del parallel_iterations, output_shapes

  return _StatelessWhile(dict(cond_fun=cond_name, body_fun=body_name))


@register_operation("StridedSlice")
def _strided_slice(proto):
  """Parse a StridedSlice Op."""
  _check_attrs(
      proto, {
          "T", "Index", "begin_mask", "ellipsis_mask", "end_mask",
          "new_axis_mask", "shrink_axis_mask"
      })

  def unpack(x):
    return [(1 if(x & (1 << v)) else 0) for v in range(32)]

  begin_mask = unpack(proto.attr["begin_mask"].i)
  ellipsis_mask = unpack(proto.attr["ellipsis_mask"].i)
  end_mask = unpack(proto.attr["end_mask"].i)
  new_axis_mask = unpack(proto.attr["new_axis_mask"].i)
  shrink_axis_mask = unpack(proto.attr["shrink_axis_mask"].i)

  def _func(
      x: jnp.ndarray,
      begin: jnp.ndarray,
      end: jnp.ndarray,
      strides: jnp.ndarray,
  ) -> jnp.ndarray:
    """`begin`, `end` and `strides` must be concrete arrays."""

    num_specs = len(begin)
    inserted = (
        len(x.shape) + sum(new_axis_mask) - (num_specs - sum(ellipsis_mask)))

    # Rebuild slices.
    dim = 0
    slices = []
    for idx in range(num_specs):
      if new_axis_mask[idx] == 1:
        slices.append(jnp.newaxis)
      elif ellipsis_mask[idx] == 1:
        slices.append(Ellipsis)
        dim += inserted
      else:
        if shrink_axis_mask[idx] == 1:
          slices.append(begin[idx])
        else:
          beg_dim = begin[idx] if begin_mask[idx] == 0 else None
          end_dim = end[idx] if end_mask[idx] == 0 else None
          stride = strides[idx]
          x_dim = x.shape[dim]
          if (stride == 1 and
              jax.core.symbolic_equal_dim(beg_dim or 0, 0) and
              jax.core.symbolic_equal_dim(end_dim or x_dim, x_dim)):
            slices.append(slice(None, None, None))
          else:
            slices.append(slice(beg_dim, end_dim, stride))
        dim += 1

    # TODO(shaobohou) Handle stride=1 slicing along polymoprhic dimensions.
    return x[tuple(slices)]

  return _func


@register_operation("Sum")
def _sum(proto):
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.sum_(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("TopKV2")
def _top_k(proto):
  _check_attrs(proto, {"T", "sorted"})
  sorted_arg = proto.attr["sorted"].b
  if not sorted_arg:
    raise ValueError("sorted=False in TopKV2 is not yet supported.")

  return jax.lax.top_k


@register_operation("Transpose")
def _transpose(proto):
  _check_attrs(proto, {"T", "Tperm"})
  return lambda x, axes: jnp.transpose(x, axes=axes)


@register_operation("Unpack")
def _unpack(proto):
  """Parse a Unpack op."""
  _check_attrs(proto, {"T", "axis", "num"})

  axis = proto.attr["axis"].i
  num = proto.attr["num"].i

  def _func(x: jnp.ndarray) -> jnp.ndarray:
    if x.shape[axis] != num:
      raise ValueError("Unpack expects dimension of {num} for axis={axis}, "
                       "found {x.shape[axis]}, shape={x.shape}")
    return [anp.squeeze(v, axis=axis) for v in anp.split(x, num, axis=axis)]

  return _func


@register_operation("XlaConvV2")
@register_operation("XlaConv")
def _xla_conv(proto):
  """Parse a XlaConv op."""
  _check_attrs(
      proto, {
          "T", "LhsT", "RhsT", "Tindices", "dimension_numbers",
          "precision_config", "preferred_element_type", "batch_group_count"
      })

  dimension_numbers = xla_utils.convolution_dimension_numbers_from_proto(
      proto.attr["dimension_numbers"].s)
  precision_config = xla_utils.precision_config_from_proto(
      proto.attr["precision_config"].s)
  batch_group_count = proto.attr["batch_group_count"].i
  if "preferred_element_type" in proto.attr:
    dst_dtype = tf.as_dtype(proto.attr["preferred_element_type"].type)
    dst_dtype = anp.get_jax_dtype(dst_dtype)
  else:
    dst_dtype = None

  def _func(
      lhs: jnp.ndarray,
      rhs: jnp.ndarray,
      strides: jnp.ndarray,
      padding: jnp.ndarray,
      lhs_dilation: jnp.ndarray,
      rhs_dilation: jnp.ndarray,
      feature_group_count: jnp.ndarray,
  ) -> jnp.ndarray:
    return jax.lax.conv_general_dilated(
        lhs,
        rhs,
        window_strides=strides.tolist(),
        padding=[tuple(v) for v in padding],
        lhs_dilation=lhs_dilation.tolist(),
        rhs_dilation=rhs_dilation.tolist(),
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count.tolist(),  # Should be int.
        batch_group_count=batch_group_count or 1,
        precision=precision_config,
        preferred_element_type=dst_dtype)

  return _func


@register_operation("XlaDotV2")
@register_operation("XlaDot")
def _xla_dot(proto):
  """Parse a XlaDot op."""
  _check_attrs(
      proto, {
          "T", "LhsT", "RhsT", "dimension_numbers", "precision_config",
          "preferred_element_type"
      })

  dimension_numbers = xla_utils.dot_dimension_numbers_from_proto(
      proto.attr["dimension_numbers"].s)
  precision_config = xla_utils.precision_config_from_proto(
      proto.attr["precision_config"].s)
  if "preferred_element_type" in proto.attr:
    dst_dtype = tf.as_dtype(proto.attr["preferred_element_type"].type)
    dst_dtype = anp.get_jax_dtype(dst_dtype)
  else:
    dst_dtype = None

  def _func(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision_config,
        preferred_element_type=dst_dtype)

  return _func


@register_operation("XlaDynamicSlice")
def _xla_dynamic_slice(proto):
  """Parse a XlaDynamicSlice op."""
  _check_attrs(proto, {"T", "Tindices"})
  return jax.lax.dynamic_slice


@register_operation("XlaDynamicUpdateSlice")
def _xla_dynamic_update_slice(proto):
  """Parse a XlaDynamicUpdateSlice op."""
  _check_attrs(proto, {"T", "Tindices"})
  return jax.lax.dynamic_update_slice


@register_operation("XlaGather")
def _xla_gather(proto):
  """Parse a XlaGather op."""
  _check_attrs(proto,
               {"T", "Tindices", "dimension_numbers", "indices_are_sorted"})

  dimension_numbers = xla_utils.gather_dimension_numbers_from_proto(
      proto.attr["dimension_numbers"].s)
  # This should exist on the XLA op, even though it's not exposed by JAX.
  indices_are_sorted = proto.attr["indices_are_sorted"].b
  del indices_are_sorted

  def _func(
      operand: jnp.ndarray,
      start_indices: jnp.ndarray,
      slice_indices: jnp.ndarray,
  ) -> jnp.ndarray:
    return jax.lax.gather(operand, start_indices, dimension_numbers,
                          slice_indices)

  return _func


@register_operation("XlaPad")
def _xla_pad(proto):
  """Parse a XlaPad op."""
  _check_attrs(proto, {"T", "Tindices"})

  def _func(
      operand: jnp.ndarray,
      padding_value: jnp.ndarray,
      padding_low: jnp.ndarray,
      padding_high: jnp.ndarray,
      padding_interior: jnp.ndarray,) -> jnp.ndarray:
    padding_config = np.stack([padding_low, padding_high, padding_interior],
                              axis=0)
    padding_config = [tuple(x) for x in padding_config.transpose().tolist()]
    return jax.lax.pad(operand, padding_value, padding_config)

  return _func


def _maybe_get_jaxpreqn(
    jaxpr: jax.core.ClosedJaxpr) -> Optional[jax.core.JaxprEqn]:
  def is_all_vars(vs):
    return all([isinstance(v, jax.core.Var) for v in vs])

  if (len(jaxpr.eqns) == 1 and
      is_all_vars(jaxpr.jaxpr.invars) and is_all_vars(jaxpr.jaxpr.outvars) and
      is_all_vars(jaxpr.eqns[0].invars) and is_all_vars(jaxpr.eqns[0].outvars)):
    return jaxpr.eqns[0]
  return None


@dataclasses.dataclass
class _XlaVariadicReduce(_HigherOrderFunction):
  """Represents a XlaVariadicReduce Op."""

  dimensions: Sequence[int]

  def __call__(self, *args: jnp.ndarray, reducer: Callable[..., Any]):
    num_args = len(args)
    operands = args[:(num_args//2)]
    init_values = args[(num_args//2):]
    assert len(operands) == len(init_values)
    reducer_fn = lambda xs, ys: reducer(*xs, *ys)
    return jax.lax.reduce(operands, init_values, reducer_fn, self.dimensions)


@register_operation("XlaVariadicReduceV2")
def _xla_variadic_reduce(proto):
  """Parse a XlaVariadicReduceV2 op."""
  _check_attrs(proto, {"T", "reducer", "dimensions_to_reduce"})

  reducer = proto.attr["reducer"].func.name
  dimensions = tuple(proto.attr["dimensions_to_reduce"].list.i)

  return _XlaVariadicReduce(dict(reducer=reducer), dimensions=dimensions)


@dataclasses.dataclass
class _XlaVariadicSort(_HigherOrderFunction):
  """Represents a XlaVariadicSort Op."""

  is_stable: bool

  def _compute_num_keys(
      self,
      dtypes: Sequence[jnp.dtype],
      comparator: Callable[..., Any],
  ) -> int:
    """Infer num_keys from the comparator and operands."""
    def get_operands():
      return sum([[jnp.array(0, dtype)] * 2 for dtype in dtypes], [])

    for idx in range(len(dtypes)):
      operands = get_operands()
      is_eq, = comparator(*operands)

      operands[idx * 2 + 1] = jnp.array(1, dtypes[idx])
      is_lt, = comparator(*operands)

      if idx == 0 and (not is_lt or is_eq):
        raise ValueError(
            "Only less-than comparator is supported for XlaVariadicSort.")

      if is_lt:
        num_keys = idx + 1
      else:
        break

    return num_keys

  def __call__(self, *args: jnp.ndarray, comparator: Callable[..., Any]):
    operands = args[:-1]
    dimension = args[-1].tolist()

    with jax.ensure_compile_time_eval():
      dtypes = [x.dtype for x in operands]
      num_keys = self._compute_num_keys(dtypes, comparator)

    return jax.lax.sort(
        operands,
        dimension=dimension,
        is_stable=self.is_stable,
        num_keys=num_keys)


@register_operation("XlaVariadicSort")
def _xla_variadic_sort(proto):
  """Parse a XlaVariadicSort op."""
  _check_attrs(proto, {"T", "comparator", "is_stable"})

  comparator = proto.attr["comparator"].func.name
  is_stable = proto.attr["is_stable"].b

  logging.warning(
      "Support for XlaVariadicSort is limited, current implementation assumes "
      "the op is generated by `jax2tf.convert(jax.lax.sort)` and does not "
      "support arbitrary comparators")

  return _XlaVariadicSort(dict(comparator=comparator), is_stable=is_stable)


class _XlaReduceWindow(_HigherOrderFunction):
  """Represents a XlaReduceWindow Op."""

  def __call__(
      self,
      operand: jnp.ndarray,
      init_value: jnp.ndarray,
      window_dimensions: jnp.ndarray,
      window_strides: jnp.ndarray,
      base_dilation: jnp.ndarray,
      window_dilation: jnp.ndarray,
      padding: jnp.ndarray,
      *,
      computation: Callable[..., Any],
  ):
    # Pattern matching computations that can be specialized.
    primitives = {
        jax.lax.max_p: jax.lax.max,
        jax.lax.min_p: jax.lax.min,
        jax.lax.add_p: jax.lax.add,
    }
    computation_jaxpr = jax.make_jaxpr(computation)(init_value, init_value)
    computation_eqn = _maybe_get_jaxpreqn(computation_jaxpr)
    if computation_eqn is not None and computation_eqn.primitive in primitives:
      computation_fn = primitives[computation_eqn.primitive]
    else:
      computation_fn = lambda *args: computation(*args)[0]
      logging.info("Calling reduce_window with the following computation:\n%s",
                   computation_jaxpr)

    return jax.lax.reduce_window(
        operand,
        init_value,
        computation=computation_fn,
        window_dimensions=window_dimensions.tolist(),
        window_strides=window_strides.tolist(),
        padding=[tuple(v) for v in padding.tolist()],
        base_dilation=base_dilation.tolist(),
        window_dilation=window_dilation.tolist())


@register_operation("XlaReduceWindow")
def _xla_reduce_window(proto):
  """Parse a XlaReduceWindow op."""
  _check_attrs(proto, {"T", "Tindices", "computation"})

  computation = proto.attr["computation"].func.name

  return _XlaReduceWindow(dict(computation=computation))


@register_operation("XlaRngBitGenerator")
def _xla_rng_bit_generator(proto):
  """Parse a XlaRngBitGenerator op."""
  _check_attrs(proto, {"Tshape", "dtype"})

  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)
  if jax_dtype != jnp.uint32:
    raise ValueError(
        f"XlaRngBitGenerator currently only supports uint32, found{jax_dtype}.")

  def _func(
      algorithm: jnp.ndarray,
      key: jnp.ndarray,
      shape: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jax.lax.rng_bit_generator(
        key=key.reshape(-1),
        shape=shape,
        dtype=jax_dtype,
        # See tensorflow/compiler/tf2xla/ops/xla_ops.cc#L812
        algorithm=xla_utils.get_random_algorithm_from_tf(algorithm.tolist()),
    )

  return _func


@dataclasses.dataclass
class _XlaScatter(_HigherOrderFunction):
  """Represents a XlaScatter Op."""

  dimension_numbers: jax.lax.ScatterDimensionNumbers
  indices_are_sorted: bool

  def __call__(
      self,
      operand: jnp.ndarray,
      indices: jnp.ndarray,
      updates: jnp.ndarray,
      *,
      update_computation: Callable[..., Any],
  ) -> jnp.ndarray:
    dummy_zero = jnp.array(0).astype(operand.dtype)
    jaxpr = jax.make_jaxpr(update_computation)(dummy_zero, dummy_zero)
    if not jaxpr.eqns:
      scatter_fn = jax.lax.scatter
    elif len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive == jax.lax.add_p:
      scatter_fn = jax.lax.scatter_add
    elif len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive == jax.lax.mul_p:
      scatter_fn = jax.lax.scatter_mul
    elif len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive == jax.lax.min_p:
      scatter_fn = jax.lax.scatter_min
    elif len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive == jax.lax.max_p:
      scatter_fn = jax.lax.scatter_max
    else:
      raise ValueError(
          "Reducer not supported as `update_computation`, found {jaxpr}")

    return scatter_fn(
        operand,
        indices,
        updates,
        dimension_numbers=self.dimension_numbers,
        indices_are_sorted=self.indices_are_sorted)


@register_operation("XlaScatter")
def _xla_scatter(proto):
  """Parse a XlaScatter op."""
  _check_attrs(
      proto, {
          "T", "Tindices", "dimension_numbers", "indices_are_sorted",
          "update_computation"
      })

  dimension_numbers = xla_utils.scatter_dimension_numbers_from_proto(
      proto.attr["dimension_numbers"].s)
  update_computation = proto.attr["update_computation"].func.name
  indices_are_sorted = proto.attr["indices_are_sorted"].b

  return _XlaScatter(
      dict(update_computation=update_computation), dimension_numbers,
      indices_are_sorted)


class _XlaSelectAndScatter(_HigherOrderFunction):
  """Represents a XlaSelectAndScatter Op."""

  def __call__(
      self,
      operand: jnp.ndarray,
      window_dimensions: jnp.ndarray,
      window_strides: jnp.ndarray,
      padding: jnp.ndarray,
      source: jnp.ndarray,
      inner_init_value: jnp.ndarray,
      *,
      scatter: Callable[..., Any],
      select: Callable[..., Any],
  ) -> jnp.ndarray:
    # Because jax.lax._select_and_scatter is not part of the JAX public api, we
    # are using a crude pattern matching to determine the reducer used in the
    # original reduce_window call.

    scatter_jaxpr = jax.make_jaxpr(scatter)(inner_init_value, inner_init_value)
    scatter_eqn = _maybe_get_jaxpreqn(scatter_jaxpr)
    if scatter_eqn is not None and scatter_eqn.primitive is not jax.lax.add_p:
      raise ValueError(
          f"Only Add is supported as scatter function, found {scatter_jaxpr}.")

    # TODO(shaobohou) Support jax.lax.add for AvgPool.
    select_primitives = {
        jax.lax.ge_p: (-jnp.inf, jax.lax.max),
        jax.lax.le_p: (jnp.inf, jax.lax.min),
    }
    select_jaxpr = jax.make_jaxpr(select)(inner_init_value, inner_init_value)
    select_eqn = _maybe_get_jaxpreqn(select_jaxpr)
    if select_eqn is not None and select_eqn.primitive in select_primitives:
      init_value, computation = select_primitives[select_eqn.primitive]
    else:
      raise ValueError("Only greater_equal (Max) and less_equal (Min) are "
                       f"supported as select function, found {select_jaxpr}")

    def reduce_window(x):
      return jax.lax.reduce_window(
          x,
          init_value,
          computation=computation,
          window_dimensions=tuple(window_dimensions.tolist()),
          window_strides=tuple(window_strides.tolist()),
          padding=[tuple(v) for v in padding.tolist()])

    _, f_vjp = jax.vjp(reduce_window, operand)
    return f_vjp(source)


@register_operation("XlaSelectAndScatter")
def _xla_select_and_scatter(proto):
  """Parse a XlaSelectAndScatter op."""
  _check_attrs(proto, {"T", "Tindices", "scatter", "select"})

  scatter = proto.attr["scatter"].func.name
  select = proto.attr["select"].func.name

  return _XlaSelectAndScatter(dict(scatter=scatter, select=select))


class _TensorEdge(NamedTuple):
  """Represents an input/output Tensor."""

  op_name: str
  idx: int = 0
  is_control: bool = False

  @classmethod
  def from_string(
      cls,
      op_str: str,
      node_map: Optional[Mapping[str, Any]] = None,
  ):
    """Parse a NodeDef input string."""
    node_map = node_map or {}

    is_control = op_str.startswith("^")
    if is_control:
      op_str = op_str[1:]

    args = op_str.split(":")

    if len(args) == 1:
      op_name, = args
      output_name = ""
      idx = 0
    elif len(args) == 2:
      op_name, idx = args
      output_name = ""
      idx = int(idx)
    elif len(args) == 3:
      op_name, output_name, idx = args
      op_def = op_def_registry.get(node_map[op_name].op)
      assert op_def.output_arg

      if len(op_def.output_arg) == 1:
        idx = int(idx)
      else:
        found_sequence_args = any([
            arg.number_attr or arg.type_list_attr for arg in op_def.output_arg
        ])
        if idx != "0" or found_sequence_args:
          raise ValueError(
              "Only zero index and single tensors are supported for multiple "
              "output args.\n"
              f"op_str={op_str}\n"
              f"op_def={op_def}")
        idx = [arg.name for arg in op_def.output_arg].index(output_name)
    else:
      raise ValueError(f"Invalid input spec string: `{op_str}`")

    return cls(op_name=op_name, idx=idx, is_control=is_control)


_tf_assign_ops = {
    "AssignAddVariableOp", "AssignSubVariableOp", "AssignVariableOp"
}

_tf_random_ops = {"RandomStandardNormal", "RandomUniform", "RandomUniformInt"}


class _LibraryFunction(NamedTuple):
  """A library function."""
  fn: Callable[..., Any]
  require_rng: bool
  # Optional fields (mainly) used by gradient functions.
  params: Optional[Mapping[str, jnp.ndarray]] = None
  input_specs: Optional[Tuple[tf.TensorSpec, ...]] = None
  output_specs: Optional[Tuple[tf.TensorSpec, ...]] = None
  # If fn is a gradient function, this is the output specs for the original fn.
  orig_fn_output_specs: Optional[Tuple[tf.TensorSpec, ...]] = None

  def __call__(self, *args, **kwargs):
    if self.params:
      return self.fn(self.params, *args, **kwargs)
    else:
      # Ignore parameters in inputs and outputs.
      return self.fn({}, *args, **kwargs)[0]


def _unbox_named_args(
    named_args: Sequence[Tuple[str, jnp.ndarray]],
    inputs: Tuple[_TensorEdge, ...],
) -> Tuple[jnp.ndarray, ...]:
  """Extract inputs from named arguments."""
  if len(named_args) != len(inputs):
    raise ValueError(
        f"Expected {len(inputs)} arguments, received {len(named_args)}")
  for (arg_name, _), inp in zip(named_args, inputs):
    if arg_name != inp.op_name:
      raise ValueError(f"Expected inputs {inp.op_name}, found {arg_name}")

  unboxed_args = [
      arg[inp.idx] if isinstance(arg, (list, tuple)) else arg
      for arg, inp in zip([x for _, x in named_args], inputs)
  ]
  return tuple(unboxed_args)


class _OpNode:
  """Represents an Op."""

  def __init__(self, proto, library: Mapping[str, _LibraryFunction],
               node_map: Mapping[str, Any]):
    self.jax_func = _jax_ops[proto.op](proto)
    self.op = proto.op
    self.name = proto.name

    self.inner_fns = dict()
    if isinstance(self.jax_func, _HigherOrderFunction):
      self.inner_fns = self.jax_func.get_inner_functions(library)

    inputs = [_TensorEdge.from_string(inp, node_map) for inp in proto.input]
    self.control_inputs = tuple([inp for inp in inputs if inp.is_control])
    self.inputs = tuple([inp for inp in inputs if not inp.is_control])

  @property
  def all_inputs(self) -> Tuple[_TensorEdge, ...]:
    return self.inputs + self.control_inputs

  @property
  def require_rng(self) -> bool:
    inner_require_rngs = any([fn.require_rng for fn in self.inner_fns.values()])
    return self.op in _tf_random_ops or inner_require_rngs

  def __call__(
      self,
      named_args: Sequence[Tuple[str, jnp.ndarray]],
      *,
      rng: jnp.ndarray,
  ) -> Tuple[Tuple[jnp.ndarray, ...], Mapping[str, jnp.ndarray]]:
    unboxed_args = _unbox_named_args(named_args, self.inputs)
    extras_dict = dict(rng=rng) if self.require_rng else dict()
    extras_dict.update(self.inner_fns)
    outputs = self.jax_func(*unboxed_args, **extras_dict)

    # Return updated variables.
    if self.op in _tf_assign_ops:
      # TODO(shaobohou) This assumes the assign op returns the updated value.
      updated_params = {named_args[0][0]: outputs}
    else:
      updated_params = {}

    return outputs, updated_params

  def __repr__(self) -> str:
    message = f"_OpNode(name={repr(self.name)}, op={repr(self.op)}"
    if self.inputs:
      message += f", inputs={repr([x.op_name for x in self.inputs])}"
    if self.control_inputs:
      message += f", controls={repr([x.op_name for x in self.control_inputs])}"
    message += ")"
    return message


def _parse_input(op_str: str) -> str:
  # Parses an input name in tf.NodeDef. This extracts the node name, removes
  # the control character and the output tensor index e.g. ^Sin:0 -> Sin
  return op_str.split(":")[0].split("^")[-1]


def _toposort(
    nodes: Mapping[str, tf.compat.v1.NodeDef],
    end_node_names: Tuple[str, ...],
):
  """Topological sorting of nodes."""

  child_counts = {}
  stack = list(end_node_names)
  while stack:
    node_name = stack.pop()
    if node_name in child_counts:
      child_counts[node_name] += 1
    else:
      child_counts[node_name] = 1
      node = nodes[node_name]
      stack.extend([_parse_input(v) for v in node.input])
  for node_name in end_node_names:
    child_counts[node_name] -= 1

  sorted_nodes = []
  childless_nodes = [
      node_name for node_name in end_node_names if child_counts[node_name] == 0
  ]
  if not childless_nodes:
    raise ValueError("No childless nodes found.")

  while childless_nodes:
    node_name = childless_nodes.pop()
    node = nodes[node_name]
    sorted_nodes.append(node)
    for parent in [_parse_input(v) for v in node.input]:
      if child_counts[parent] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[parent] -= 1

  return sorted_nodes[::-1]


class Variable(np.ndarray):
  """Array subclass with additional metadaa for representing variables."""

  def __new__(cls, arr: np.ndarray, trainable: bool, name: str):
    obj = np.asarray(arr).view(cls)
    obj.trainable = trainable
    obj.name = name
    return obj

  def assign(self, arr: np.ndarray) -> "Variable":
    return Variable(arr, trainable=self.trainable, name=self.name)

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.trainable = getattr(obj, "trainable", None)
    self.name = getattr(obj, "name", None)

  def __repr__(self) -> str:
    message = f"Variable(name={repr(self.name)}, "
    message += f"trainable={repr(self.trainable)}, "
    message += f"numpy={np.array_repr(self)}"
    message += ")"
    return message


def _make_functional(
    jax_func: Callable[..., Any],
    jax_params: Mapping[str, Variable],
) -> Callable[..., Any]:
  if jax_params:
    raise ValueError(
        f"Expected function to have no captured variables, found {jax_params}")

  return lambda *args, **kwargs: jax_func({}, *args, **kwargs)[0]


def convert_functional(tf_func: Any, *args, **kwargs) -> Callable[..., Any]:
  return _make_functional(*convert(tf_func, *args, **kwargs))


def convert_functional_from_restored(tf_func: Any) -> Callable[..., Any]:
  return _make_functional(*convert_from_restored(tf_func))


def convert_from_restored(
    tf_func) -> Tuple[Callable[..., Any], Mapping[str, Variable]]:
  """Converts a RestoredFunction (from a SaveModel) if it is unambiguous."""
  concrete_functions = getattr(tf_func, "concrete_functions", ())
  if len(concrete_functions) != 1:
    err_message = (
        f"Found {len(concrete_functions)} concrete functions, use "
        "tf2jax.convert or tf2jax.convert_functional with *args and **kwargs "
        "to select one of the options above.")
    saved_model_err_message = ""
    try:
      # Deliberately trigger pretty error message from SavedModel.
      tf_func(None)
    except ValueError as e:
      saved_model_err_message = str(e)
    raise ValueError(saved_model_err_message + "\n\n" + err_message)

  args, kwargs = concrete_functions[0].structured_input_signature
  return convert(tf_func, *args, **kwargs)


def _fix_tfhub_specs(
    structured_specs,
    flat_tensors,
    expected_count: Optional[int] = None,
):
  """Fix names used by TF-Hub TensorSpecs."""
  flat_specs = tree.flatten(structured_specs)
  tensor_count = 0
  for idx, val in enumerate(flat_specs):
    if isinstance(val, tf.TensorSpec):
      flat_specs[idx] = tf.TensorSpec(
          val.shape, val.dtype, name=flat_tensors[tensor_count].op.name)
      tensor_count += 1
  if expected_count is not None:
    assert tensor_count == expected_count

  structured_specs = tree.unflatten_as(structured_specs, flat_specs)
  return structured_specs


def _maybe_tracer_to_tf_spec(val: Any) -> Any:
  if isinstance(val, jax.core.Tracer):
    return tf.TensorSpec(shape=val.shape, dtype=val.dtype)
  else:
    return val


def convert(
    tf_func: Any,  # tensorflow.python.eager.function is not visible.
    *args,
    **kwargs,
) -> Tuple[Callable[..., Any], Mapping[str, Variable]]:
  """Convert a tf.Function to a Jax function for some concrete inputs.

  The concrete inputs are used to instantiate a tf.ConcreteFunction, the
  graphdef of which is then parsed and used to generate the corresponding Jax
  code.

  Args:
    tf_func: a tf.Function
    *args: positional arguments.
    **kwargs: keyword arguments.

  Returns:
    A tuple: the first is a Jax functions that takes a flat parameter dict and a
    variable number of arrays as inputs, and returns a nested structure of
    outputs and an updated parameter dict; the second is the flat parameter dict
    correpondings to all TF variables used by the concrete function.
  """
# Log usage here.

  args, kwargs = tree.map_structure(_maybe_tracer_to_tf_spec, (args, kwargs))
  try:
    concrete_func = tf_func.get_concrete_function(*args, **kwargs)
  except AttributeError:
    logging.error("Expected `tf.Function`, received %s", tf_func)
    raise

  graph = concrete_func.graph
  graphdef = graph.as_graph_def()

  num_flat_args = len(concrete_func.inputs) - len(concrete_func.captured_inputs)
  captures = list(
      zip([inp.op.name for inp in concrete_func.inputs[num_flat_args:]],
          [inp for inp in concrete_func.captured_inputs]))
  func_variables = {v.handle.ref(): v for v in concrete_func.variables}
  variable_map = {
      k: func_variables[v.ref()] for k, v in captures if v.dtype == tf.resource
  }
  constants = {k: v.numpy() for k, v in captures if v.dtype != tf.resource}

  captured_input_names = tuple([
      v.op.name for v in concrete_func.inputs[num_flat_args:]
  ])

  def maybe_tensor_to_spec(v):
    if v is None or isinstance(v, tf.TensorSpec):
      return v
    else:
      return tf.TensorSpec.from_tensor(v)

  num_inputs = len(concrete_func.inputs) - len(concrete_func.captured_inputs)
  structured_inputs = concrete_func.structured_input_signature
  structured_inputs = _fix_tfhub_specs(structured_inputs, concrete_func.inputs,
                                       num_inputs)
  flat_outputs = tree.flatten(concrete_func.outputs)
  structured_outputs = tree.map_structure(maybe_tensor_to_spec,
                                          concrete_func.structured_outputs)
  # TODO(shaobohou) We do not check the number of output tensors because they do
  # not match for custom gradients functions.
  structured_outputs = _fix_tfhub_specs(structured_outputs, flat_outputs)
  try:
    fullargspec = tf_func.function_spec.fullargspec
  except AttributeError:
    logging.warning("No fullargspec found on %s.", tf_func)
    fullargspec = None

  if fullargspec is not None:
    signature = utils.fullargspec_to_signature(fullargspec)
  else:
    exp_args, exp_kwargs = structured_inputs
    if exp_args:
      raise ValueError("If function_spec is None then only keyword arguments "
                       f"are expectd, found args={exp_args} in structure.")
    parameters = tuple([
        # TODO(shaobohou) Remove temporary fix for TF-Hub.
        inspect.Parameter(
            k.replace("$", "___"), kind=inspect.Parameter.KEYWORD_ONLY)
        for k in tree.flatten(exp_kwargs.keys())
    ])
    signature = inspect.Signature(parameters=parameters)

  # Extract custom_gradient functions from the registry.
  if get_config("convert_custom_gradient"):
    library = _convert_all_gradient_functions(graph, {})
  else:
    library = {}

  return _convert(
      graphdef,
      signature=signature,
      structured_inputs=structured_inputs,
      structured_outputs=structured_outputs,
      captured_input_names=captured_input_names,
      variable_map=variable_map,
      constants=constants,
      library=library,
  )


class _EvaluationCache:
  """Cache holding intermediate outputs."""

  def __init__(
      self,
      nodes: Sequence[Any],
      inputs: Sequence[Tuple[str, Any]],
      outputs: Sequence[_TensorEdge],
  ):
    self.outputs = dict(inputs)
    # Reference counting.
    self._counts = collections.Counter()
    for node in nodes:
      for inp in node.inputs + node.control_inputs:
        self._counts[inp.op_name] += 1
    for inp_op_name, _ in inputs:
      self._counts[inp_op_name] += 1
    for out in outputs:
      self._counts[out.op_name] += 1

  def free_inputs(self, node):
    for inp in node.inputs + node.control_inputs:
      self._counts[inp.op_name] -= 1
      # Free up buffers.
      if not self._counts[inp.op_name]:
        del self.outputs[inp.op_name]


def _unique_everseen(seq):
  seen = set()
  return [x for x in seq if not (x in seen or seen.add(x))]


class _Subgraph(NamedTuple):
  """A subgraph of computations with a custom_gradient."""

  subgraph: Tuple[_OpNode, ...]
  captures: Tuple[_TensorEdge, ...]
  outputs: Tuple[_TensorEdge, ...]
  output_node: _OpNode
  grad_fn: _LibraryFunction

  @property
  def name(self) -> str:
    return self.output_node.name  # pytype: disable=attribute-error

  @property
  def inputs(self) -> Tuple[_TensorEdge, ...]:
    return self.function_inputs + self.captures

  @property
  def function_inputs(self) -> Tuple[_TensorEdge, ...]:
    """Inputs for the custom_gradient wrapped function."""
    return tuple(self.output_node.inputs[len(self.outputs):])  # pytype: disable=attribute-error

  @property
  def unique_inputs(self) -> Tuple[_TensorEdge, ...]:
    unique_captures = tuple([
        x for x in _unique_everseen(self.captures)
        if x not in set(self.function_inputs)
    ])
    return self.function_inputs + unique_captures

  @property
  def control_inputs(self) -> Tuple[_TensorEdge, ...]:
    return ()

  @property
  def require_rng(self) -> bool:
    return any([n.require_rng for n in self.subgraph])

  def __call__(
      self,
      named_args: Sequence[Tuple[str, jnp.ndarray]],
      *,
      rng: jnp.ndarray,
  ) -> Tuple[Tuple[jnp.ndarray, ...], Mapping[str, jnp.ndarray]]:
    grad_inputs = tuple([_TensorEdge(v.name) for v in self.grad_fn.input_specs])

    @jax.custom_gradient
    def fn(*args):
      eval_cache = _EvaluationCache(
          self.subgraph, named_args, self.output_node.inputs + grad_inputs)
      assert len(args) == len(self.unique_inputs)
      for inp, val in zip(self.unique_inputs, args):
        if isinstance(eval_cache.outputs[inp.op_name], list):
          eval_cache.outputs[inp.op_name][inp.idx] = val
        elif isinstance(eval_cache.outputs[inp.op_name], tuple):
          old_outputs = eval_cache.outputs[inp.op_name]
          eval_cache.outputs[inp.op_name] = (
              old_outputs[:inp.idx] + (val,) + old_outputs[inp.idx + 1:])
        else:
          eval_cache.outputs[inp.op_name] = val

      num_rng_required = sum([node.require_rng for node in self.subgraph])
      if num_rng_required:
        rng_keys = list(jax.random.split(rng, num_rng_required))
      else:
        rng_keys = []

      for node in self.subgraph + (self.output_node,):
        collected_inputs = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in node.inputs
        ]
        sub_rng = rng_keys.pop() if node.require_rng else None
        eval_cache.outputs[node.name], updated_params = node(
            collected_inputs, rng=sub_rng)
        if updated_params:
          raise ValueError(
              "Variable assignments not supported in custom_gradient subgraph, "
              f"found {tuple(updated_params.keys())}.")

        eval_cache.free_inputs(node)

      # dy is the gradients for fn(*args) + args, i.e. inputs to IdentityN
      def grad_fn(dy):
        assert len(dy) == len(self.output_node.inputs)

        captured_inputs = tuple(grad_inputs[len(dy):])
        named_captures = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in captured_inputs
        ]
        captures = _unbox_named_args(named_captures, captured_inputs)
        grad_args = dy + captures

        dx = self.grad_fn(*grad_args)
        assert len(dx) == len(self.unique_inputs)

        return dx

      return eval_cache.outputs[self.output_node.name], grad_fn

    args_map = dict(named_args)
    fn_named_args = [
        (v.op_name, args_map[v.op_name]) for v in self.unique_inputs
    ]
    unboxed_args = _unbox_named_args(fn_named_args, self.unique_inputs)
    outputs = fn(*unboxed_args)
    return outputs, {}

  def rewrite(
      self,
      nodes: Sequence[_OpNode]) -> Tuple[Union["_Subgraph", _OpNode], ...]:
    """Remove subgraph from the main graph."""
    new_nodes = []
    processed = []
    subgraph = self.subgraph + (self.output_node,)
    subgraph_map = {v.name: v for v in subgraph}
    for node in nodes:
      if node.name in subgraph_map:
        processed.append(node)
        if node == self.output_node:
          new_nodes.append(self)
      else:
        new_nodes.append(node)

    assert len(processed) == len(subgraph)
    return tuple(new_nodes)


def _contains_custom_gradient(node: tf.compat.v1.NodeDef) -> bool:
  # IdentityN are used to override gradients.
  return node.op == "IdentityN"


# Extract subgraphs for custom_gradient.
def _extract_subgraphs(graphdef, nodes, library):
  """Extract all subgraphs with their own custom_gradients."""

  op_map = {n.name: (idx, n) for idx, n in enumerate(nodes)}
  if _EMPTY_RETURN_OP_NAME in op_map:
    logging.info("Skip subgraph extraction for function with no return values.")
    return {}

  subgraphs = {}
  for node in graphdef.node:
    if _contains_custom_gradient(node):
      grad_fn_name = str(node.attr["_gradient_op_type"].s, "utf-8")
      grad_fn = library[grad_fn_name]

      output_node = op_map[node.name][1]
      assert len(node.input) == len(output_node.inputs)

      # Inputs to the gradient function are fn(*args) + args + captured_args
      num_outputs = len(grad_fn.orig_fn_output_specs)
      all_specs = [_TensorEdge(x.name) for x in grad_fn.input_specs]
      outputs = list(output_node.inputs[:num_outputs])
      inputs = list(output_node.inputs[num_outputs:len(node.input)])
      captured_inputs = all_specs[len(node.input):]

      # Traverse and extract subgraph.
      subgraph = set([x.op_name for x in inputs + captured_inputs])
      unexpanded = [x.op_name for x in outputs]
      while unexpanded:
        top_node, *unexpanded = unexpanded
        if top_node not in subgraph:
          subgraph.add(top_node)
          unvisited = [x.op_name for x in op_map[top_node][1].inputs]
          unexpanded += unvisited

      # Separate internal and external captures. Internal captures are found in
      # the subgraph. External captures are found in outer graph.
      unused_captures = []
      internal_captures = []
      external_captures = []
      used_inputs = set(
          sum([op_map[n][1].inputs for n in subgraph], ()) + output_node.inputs)
      for inp in captured_inputs:
        is_internal = all(
            [x.op_name in subgraph for x in op_map[inp.op_name][1].inputs])
        if inp not in used_inputs:
          unused_captures.append(inp)
        elif is_internal:
          internal_captures.append(inp)
        else:
          external_captures.append(inp)

      excluded = inputs + unused_captures + external_captures
      subgraph = subgraph.difference(set([x.op_name for x in excluded]))
      sub_nodes = [op_map[x] for x in subgraph]
      sub_nodes = [x for _, x in sorted(sub_nodes)]
      subgraphs[grad_fn_name] = _Subgraph(
          subgraph=tuple(sub_nodes),
          captures=tuple(external_captures),
          outputs=tuple(outputs),
          output_node=output_node,
          grad_fn=grad_fn,
      )

  num_nodes = sum([len(g.subgraph) for g in subgraphs.values()])
  num_unique_nodes = len(
      set(sum([[x.name for x in g.subgraph] for g in subgraphs.values()], [])))
  if num_nodes != num_unique_nodes:
    raise ValueError("Overlapping subgraphs are not yet supported.")

  return subgraphs


def _infer_relu_from_jax2tf(nodes):
  """Detect max(x, 0) and replace with jax.nn.relu."""
  found_jax2tf = any(["jax2tf_out" in n.name for n in nodes])
  node_map = {n.name: n for n in nodes}
  for node in nodes:
    if node.op == "Maximum":
      cast_arg = node_map[node.inputs[1].op_name]
      if cast_arg.op == "Cast":
        const_arg = node_map[cast_arg.inputs[0].op_name]
        if const_arg.op == "Const" and const_arg((), rng=None)[0].tolist() == 0:
          # Replace the Maximum op with a Relu op, but keep the node name.
          # The Cast and Const ops may now be redundant but are kept anyway.
          node.op = "Relu"
          node.inputs = node.inputs[:1]
          node.jax_func = _jax_ops["Relu"](_NodeDef("Relu", node.name, (), {}))
          if not found_jax2tf:
            logging.warning("Replaced max(x, 0) with jax.nn.relu but did not "
                            "find jax2tf_out.")


_FunctionDef = Any


def _get_function_protos(
    graphdef: tf.compat.v1.GraphDef,) -> Iterator[Tuple[str, _FunctionDef]]:
  """Get all library function protos from a tf.GraphDef."""

  func_protos = {
      func.signature.name: func for func in graphdef.library.function
  }
  processed = set()

  def process_func(func) -> Iterator[Tuple[str, _FunctionDef]]:
    for node in func.node_def:
      yield from process_node(node)

  def process_node(node) -> Iterator[Tuple[str, _FunctionDef]]:
    for attr in node.attr.values():
      for func_name in [attr.func.name] + [f.name for f in attr.list.func]:
        if func_name and func_name not in processed:
          yield from process_func(func_protos[func_name])
          yield func_name, func_protos[func_name]
          processed.add(func_name)

  for node in graphdef.node:
    yield from process_node(node)


# Partially mimics tf.compat.v1.NodeDef
class _NodeDef(NamedTuple):
  op: str
  name: str
  input: Tuple[str, ...]
  attr: Optional[Mapping[str, tf.compat.v1.AttrValue]] = None


class _GraphDef(NamedTuple):
  node: Tuple[Union[tf.compat.v1.NodeDef, _NodeDef], ...]


def _convert(
    graphdef: Union[tf.compat.v1.GraphDef, _GraphDef],
    signature: inspect.Signature,
    structured_inputs,
    structured_outputs,
    captured_input_names: Optional[Tuple[str, ...]] = None,
    variable_map: Optional[Mapping[str, tf.Variable]] = None,
    constants: Optional[Mapping[str, jnp.ndarray]] = None,
    library: Optional[Mapping[str, _LibraryFunction]] = None,
) -> Tuple[Callable[..., Any], Mapping[str, Variable]]:
  """Convert a GraphDef to a Jax function.

  Args:
    graphdef: a tf.GraphDef or GraphDef like object.
    signature: An inspect.Signature representing a call signature.
    structured_inputs: Structured input spec for the function, follows the
      format of inspect.BoundArguments, i.e. a tuple of a list of position args
      and a dict of keyword-only args.
    structured_outputs: Structured output spec for the function.
    captured_input_names: Names of other input tensors that are not part of
      `structured_inputs`, e.g. variables captured by tf.Function.
    variable_map: A mapping from tensor names to tf.Variables. The keys are a
      subset of captured_input_names.
    constants: A mapping from tensor names to constant values. The keys are a
      subset of captured_input_names.
    library: A mapping from function names to Callable. This is non-empty on
      recurisve calls if the FunctionDefLibrary in the GraphDef is non-empty.

  Returns:
    A tuple: the first is a Jax functions that takes a flat parameter dict and a
    variable number of arrays as inputs, and returns a nested structure of
    outputs and an updated parameter dict; the second is the flat parameter dict
    correpondings to all TF variables used by the graph.
  """
  # Recursively convert function protos in FunctionDefLibrary.
  library = dict((library or {}).items())
  if hasattr(graphdef, "library"):
    for func_name, func_proto in _get_function_protos(graphdef):
      if func_name not in library:
        logging.info("Converting library function %s", func_name)
        library[func_name] = _convert_library_function(func_proto, library)

  captured_input_names = captured_input_names or ()
  variable_map = variable_map or {}
  constants = constants or {}

  # Canonicalise inputs and outputs.
  input_path_to_specs = [(p, v)
                         for p, v in tree.flatten_with_path(structured_inputs)
                         if v is not _UNUSED_INPUT]
  # TODO(shaobohou) Remove temporary fix for TF-Hub.
  replace_fn = (lambda k: k.replace("$", "___") if isinstance(k, str) else k)
  input_path_to_specs = [
      (tree.map_structure(replace_fn, p), v) for p, v in input_path_to_specs
  ]

  # Extract input and output tensor names.
  input_names = []
  static_arg_num = 0
  for (_, spec) in input_path_to_specs:
    if isinstance(spec, tf.TensorSpec):
      input_names.append(spec.name)
    else:
      input_names.append(f"__static_arg_{static_arg_num}")
      static_arg_num += 1
  input_names = tuple(input_names)
  assert len(input_names) == len(set(input_names))
  output_names = tuple([v.name for v in tree.flatten(structured_outputs)])

  unsupported = {node.op for node in graphdef.node if node.op not in _jax_ops}
  if unsupported:
    raise ValueError(f"Unsupported operations in graph: {list(unsupported)}\n"
                     "Support for additional TensorFlow ops are added on an "
                     "as-needed basis, please contact the library owner(s).")

  # Extract variables.
  variables_tf = {_parse_input(v.name): v for _, v in variable_map.items()}
  if tf.executing_eagerly():
    variables = {
        k: Variable(v.numpy(), v.trainable, v.name)
        for k, v in variables_tf.items()
    }
  else:
    with tf.compat.v1.Session() as sess:
      sess.run(tf.compat.v1.initialize_variables(variables_tf.values()))
      variables_np = sess.run(variables_tf)
      variables = tree.map_structure(
          lambda var, arr: Variable(arr, var.trainable, var.name),
          variables_tf, variables_np)

  var_by_node = {k: _parse_input(v.name) for k, v in variable_map.items()}
  node_by_var = {v: k for k, v in var_by_node.items()}

  node_map = {n.name: n for n in graphdef.node}
  if not output_names:
    assert _EMPTY_RETURN_OP_NAME not in node_map
    assign_nodes = tuple(
        f"^{n.name}" for n in graphdef.node if n.op in _tf_assign_ops)
    node_map[_EMPTY_RETURN_OP_NAME] = _NodeDef(
        "NoOp", _EMPTY_RETURN_OP_NAME, assign_nodes, {})
    output_names = [_EMPTY_RETURN_OP_NAME]
    logging.warning(
        "No output nodes found, inserted NoOp to trigger side effects: %s",
        assign_nodes)

  nodes = _toposort(node_map, output_names)
  nodes = [_OpNode(node, library, node_map) for node in nodes]
  output_args = [_TensorEdge.from_string(v, node_map) for v in output_names]
  num_rng_required = sum([node.require_rng for node in nodes])

  if get_config("infer_relu_from_jax2tf"):
    _infer_relu_from_jax2tf(nodes)

  if get_config("convert_custom_gradient"):
    subgraphs = _extract_subgraphs(graphdef, nodes, library)
    for _, subgraph in subgraphs.items():
      nodes = subgraph.rewrite(nodes)

  # TODO(shaobohou) restore some sort of signature
  def jax_func(
      params: Mapping[str, jnp.ndarray],
      *func_args,
      rng: Optional[jnp.ndarray] = None,
      **func_kwargs,
  ) -> Tuple[Any, Mapping[str, jnp.ndarray]]:
    if num_rng_required and rng is None:
      raise ValueError(f"PRNG key required for random ops, found rng={rng}.")

    bound_args = signature.bind(*func_args, **func_kwargs)
    bound_args.apply_defaults()
    inputs = dict(
        tree.flatten_with_path((bound_args.args, bound_args.kwargs)))
    inputs = tuple([inputs[p] for p, _ in input_path_to_specs])

    if len(inputs) != len(input_names):
      raise ValueError(
          f"Expected {len(input_names)} args, found {len(inputs)}.")

    inputs = tree.map_structure(
        lambda x: x if isinstance(x, jnp.ndarray) else np.array(x), inputs)
    all_params = {**params, **constants}

    for inp, (path, spec) in zip(inputs, input_path_to_specs):
      if not isinstance(spec, tf.TensorSpec):
        # A `None` spec denotes a Tensor argument that was unused in deepfunc.
        is_tracer = isinstance(inp, jax.core.Tracer)
        if spec is not None and (is_tracer or spec != inp):
          inp_type = "tracer" if is_tracer else "literal value"
          raise ValueError(
              f"Found unexpected {inp_type} `{inp}`, expected `{spec}` for "
              f"argument at {path}. This may be because the original "
              "TensorFlow function was traced with a literal value instead of "
              "a Tensor or Array.")
        else:
          continue
      if (get_config("strict_shape_check") and
          not spec.shape.is_compatible_with(_fix_jax_poly_shape(inp.shape))):
        raise ValueError(
            f"Found incompatible input shape: {inp.shape}, expected "
            f"{spec.shape} for argument at {path}.\n"
            "This can be a problem if the outputs depend on input shapes, e.g. "
            "BatchNorm during training.\n"
            "If this is not an issue for you, the error can be disabled using "
            "either tf2jax.update_config('strict_shape_check', False) or the "
            "context manager "
            "tf2jax.override_config('strict_shape_check', False)")
      if (get_config("strict_dtype_check") and
          not spec.dtype.is_compatible_with(inp.dtype)):
        raise ValueError(
            f"Found incompatible input dtype: {inp.dtype}, expected "
            f"{spec.dtype} for argument at {path}.\n"
            "If this is not an issue for you, the error can be disabled "
            "either tf2jax.update_config('strict_dtype_check', False) or the "
            "context manager "
            "tf2jax.override_config('strict_dtype_check', False)")

    full_inputs = inputs + tuple(
        [all_params[var_by_node.get(v, v)] for v in captured_input_names])
    full_inputs = list(zip(input_names + captured_input_names, full_inputs))

    if num_rng_required:
      rng_keys = list(jax.random.split(rng, num_rng_required))
    else:
      rng_keys = []

    updated_param_names = set()
    eval_cache = _EvaluationCache(nodes, full_inputs, output_args)
    for node in nodes:
      if node.name not in eval_cache.outputs:
        # Double-check control inputs.
        for inp in node.control_inputs:
          if inp.op_name not in eval_cache.outputs:
            raise ValueError(
                f"Control dependency {inp} not executed for node `{node.name}`")
        collected_inputs = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in node.inputs
        ]
        sub_rng = rng_keys.pop() if node.require_rng else None
        eval_cache.outputs[node.name], updated_params = node(
            collected_inputs, rng=sub_rng)
        # Assign variables.
        for var_name, var_val in updated_params.items():
          eval_cache.outputs[var_name] = var_val
          updated_param_names.add(var_name)

        eval_cache.free_inputs(node)

    flat_outputs = tuple([eval_cache.outputs[k.op_name] for k in output_args])
    flat_outputs = [v for v in flat_outputs if v is not _EMPTY_RETURN_VALUE]
    collected_outputs = tree.unflatten_as(structured_outputs, flat_outputs)

    # Parameters after any assignment.
    new_params = {
        var_name: eval_cache.outputs[node_by_var[var_name]]
        for var_name in params.keys()
    }

    lost_params = [v for v in updated_param_names if v not in var_by_node]
    if lost_params:
      raise ValueError(f"Some updated parameters are lost, {lost_params}.")

    return collected_outputs, new_params

  return jax_func, variables


def _convert_library_function(
    proto,
    library: Optional[Mapping[str, _LibraryFunction]],
) -> _LibraryFunction:
  """Convert a FunctionDef."""
  input_nodes = []
  for arg in proto.signature.input_arg:
    input_nodes.append(
        _NodeDef("Placeholder", arg.name, (), {"dtype": tf.as_dtype(arg.type)}))
  output_nodes = []
  for arg in proto.signature.output_arg:
    output_nodes.append(
        _NodeDef(
            "Identity",
            arg.name,
            tuple([proto.ret[arg.name]] + ["^" + v for v in proto.control_ret]),
            {"T": tf.as_dtype(arg.type)},
        ))
  graphdef = _GraphDef(tuple(input_nodes + list(proto.node_def) + output_nodes))

  params = [
      inspect.Parameter(arg.name, inspect.Parameter.POSITIONAL_ONLY)
      for arg in proto.signature.input_arg
  ]
  signature = inspect.Signature(params)

  structured_inputs = tuple([
      tf.TensorSpec(None, dtype=tf.as_dtype(arg.type), name=arg.name)
      for arg in proto.signature.input_arg
  ])
  structured_outputs = tuple([
      tf.TensorSpec(None, dtype=tf.as_dtype(arg.type), name=arg.name)
      for arg in proto.signature.output_arg
  ])

  jax_func, jax_params = _convert(
      graphdef,
      signature,
      [structured_inputs, {}],
      structured_outputs,
      library=library)
  if jax_params:
    raise ValueError(
        f"Library function should be stateless, found variables {jax_params}")

  # Does any of the ops or inner functions require RNG?
  require_rng = any([n.op in _tf_random_ops for n in proto.node_def])
  for node in proto.node_def:
    for attr in node.attr.values():
      if attr.func.name:
        require_rng = require_rng or library[attr.func.name].require_rng

  return _LibraryFunction(jax_func, require_rng, None, structured_inputs,
                          structured_outputs)


def _filter_nodes(
    predicate: Callable[[tf.compat.v1.NodeDef], bool],
    graph: Any,
) -> Iterator[Tuple[tf.compat.v1.NodeDef, Any]]:
  """Filter nodes in a tf.Graph with a predicate."""
  graph_def = graph.as_graph_def()
  for node in graph_def.node:
    if predicate(node):
      yield node, graph
  if hasattr(graph_def, "library"):
    for func in graph_def.library.function:
      # TODO(shaobohou) Use the public API once it is available.
      library_fn = graph._functions[func.signature.name]  # pylint: disable=protected-access
      for node in func.node_def:
        if predicate(node):
          yield node, library_fn.graph


def _convert_all_gradient_functions(
    graph: Any,
    library: Mapping[str, _LibraryFunction],
) -> Mapping[str, _LibraryFunction]:
  """Recursively convert all custom gradients in a tf.Graph."""
  grad_lib = {}
  for node, graph in _filter_nodes(_contains_custom_gradient, graph):
    # Note that dict(**a, **b) will raise TypeError on dupliates, unlike {}.
    grad_lib.update(
        _convert_gradient_function(node, graph, dict(**library, **grad_lib)))
  return grad_lib


def _convert_gradient_function(
    proto: tf.compat.v1.NodeDef,
    graph: Any,
    library: Mapping[str, _LibraryFunction],
) -> Mapping[str, _LibraryFunction]:
  """Convert a custom_gradient function."""
  op = graph.as_graph_element(proto.name)
  input_specs = tuple([tf.TensorSpec.from_tensor(v) for v in op.inputs])
  grad_fn_name = str(proto.attr["_gradient_op_type"].s, "utf-8")
  if grad_fn_name in library:
    return {}

  @tf.function
  def tf_grad_fn(*grad_args, **grad_kwargs):
    fn = tf_ops.gradient_registry.lookup(grad_fn_name)
    return fn(None, *grad_args, **grad_kwargs)

  concrete_tf_grad_fn = tf_grad_fn.get_concrete_function(*input_specs)
  grad_lib = _convert_all_gradient_functions(concrete_tf_grad_fn.graph, library)

  logging.info("Converting gradient function %s", grad_fn_name)
  grad_inputs = concrete_tf_grad_fn.inputs
  grad_captured_inputs = concrete_tf_grad_fn.captured_inputs
  num_flat_args = len(grad_inputs) - len(grad_captured_inputs)
  func_variables = {v.handle.ref(): v for v in concrete_tf_grad_fn.variables}

  # Gradient function can capture tensors in the outer function. Move them
  # into the arguments of the gradient function for conversion to JAX.
  variable_map = {}
  constant_map = {}
  external_capture_specs = []
  internal_capture_names = []
  for inp, cap in zip(grad_inputs[num_flat_args:], grad_captured_inputs):
    if cap.dtype == tf.resource:
      variable_map[inp.op.name] = func_variables[cap.ref()]
      internal_capture_names.append(inp.op.name)
    elif hasattr(cap, "numpy"):
      constant_map[inp.op.name] = cap.numpy()
      internal_capture_names.append(inp.op.name)
    else:
      external_capture_specs.append(tf.TensorSpec.from_tensor(cap))

  structured_grad_input_specs = tree.map_structure(tf.TensorSpec.from_tensor,
                                                   concrete_tf_grad_fn.inputs)
  structured_grad_input_specs = (structured_grad_input_specs, {})
  grad_input_specs = input_specs + tuple(external_capture_specs)
  grad_structured_outputs = tuple(
      itertools.dropwhile(lambda x: x is None,
                          concrete_tf_grad_fn.structured_outputs))
  grad_output_specs = tuple([
      tf.TensorSpec.from_tensor(x) for x in grad_structured_outputs
  ])
  # Nones correspond to the outputs of the original function.
  num_fn_outputs = (
      len(concrete_tf_grad_fn.structured_outputs) -
      len(grad_structured_outputs))
  signature = inspect.Signature(
      (inspect.Parameter("grad_args", inspect.Parameter.VAR_POSITIONAL),))

  jax_grad_fn, jax_grad_params = _convert(
      concrete_tf_grad_fn.graph.as_graph_def(),
      signature,
      structured_grad_input_specs,
      grad_output_specs,
      captured_input_names=tuple(internal_capture_names),
      variable_map=variable_map,
      constants=constant_map,
      # Note that dict(**a, **b) will raise TypeError on dupliates, unlike {}.
      library=dict(**library, **grad_lib),
  )
  grad_fn = _LibraryFunction(jax_grad_fn, False, jax_grad_params,
                             grad_input_specs, grad_output_specs,
                             grad_output_specs[:num_fn_outputs])
  return dict(**grad_lib, **{grad_fn_name: grad_fn})
