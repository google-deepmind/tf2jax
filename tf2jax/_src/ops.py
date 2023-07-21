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

import dataclasses
import functools
from typing import Any, Callable, List, Optional, Mapping, Protocol, Sequence, Set, Tuple, Union

from absl import logging

import jax
from jax._src.lax import control_flow as lax_control_flow
from jax.experimental import checkify
from jax.lib import xla_client
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tf2jax._src import config
from tf2jax._src import numpy_compat as anp
from tf2jax._src import xla_utils


ArrayLike = Union[np.ndarray, jnp.ndarray]

# NoOp inserted to trigger side effects in function with no return values.
_EMPTY_RETURN_OP_NAME = "__NO_RETURN__"
_EMPTY_RETURN_VALUE = object()

safe_zip = jax.util.safe_zip


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
    "Angle": _get_jax_op(jnp.angle, {"T", "Tout"}),
    "Asinh": _get_jax_op(jnp.arcsinh, {"T"}),
    "Atanh": _get_jax_op(jnp.arctanh, {"T"}),
    "Atan2": _get_jax_op(jnp.arctan2, {"T"}),
    "BesselI0e": _get_jax_op(jax.lax.bessel_i0e, {"T"}),
    "BesselI1e": _get_jax_op(jax.lax.bessel_i1e, {"T"}),
    "BitwiseAnd": _get_jax_op(jnp.bitwise_and, {"T"}),
    "BitwiseOr": _get_jax_op(jnp.bitwise_or, {"T"}),
    "BitwiseXor": _get_jax_op(jnp.bitwise_xor, {"T"}),
    "BroadcastTo": _get_jax_op(anp.broadcast_to, {"T", "Tidx"}),
    "Ceil": _get_jax_op(jnp.ceil, {"T"}),
    "Cholesky": _get_jax_op(jnp.linalg.cholesky, {"T"}),
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
    "FFT": _get_jax_op(
        functools.partial(jnp.fft.fftn, axes=(-1,)), {"Tcomplex"}),
    "FFT2D": _get_jax_op(
        functools.partial(jnp.fft.fftn, axes=(-2, -1,)), {"Tcomplex"}),
    "FFT3D": _get_jax_op(
        functools.partial(jnp.fft.fftn, axes=(-3, -2, -1,)), {"Tcomplex"}),
    "Floor": _get_jax_op(jnp.floor, {"T"}),
    "FloorMod": _get_jax_op(anp.mod, {"T"}),
    "FloorDiv": _get_jax_op(anp.floor_divide, {"T"}),
    "Greater": _get_jax_op(anp.greater, {"T"}),
    "GreaterEqual": _get_jax_op(anp.greater_equal, {"T"}),
    "Identity": _get_jax_op(lambda x: x, {"T"}),
    "IFFT": _get_jax_op(
        functools.partial(jnp.fft.ifftn, axes=(-1,)), {"Tcomplex"}),
    "IFFT2D": _get_jax_op(
        functools.partial(jnp.fft.ifftn, axes=(-2, -1,)), {"Tcomplex"}),
    "IFFT3D": _get_jax_op(
        functools.partial(jnp.fft.ifftn, axes=(-3, -2, -1,)), {"Tcomplex"}),
    "IRFFT": _get_jax_op(
        functools.partial(jnp.fft.irfftn, axes=(-1,)), {"Tcomplex", "Treal"}),
    "IRFFT2D": _get_jax_op(
        functools.partial(
            jnp.fft.irfftn, axes=(-2, -1,)), {"Tcomplex", "Treal"}),
    "IRFFT3D": _get_jax_op(
        functools.partial(
            jnp.fft.irfftn, axes=(-3, -2, -1,)), {"Tcomplex", "Treal"}),
    "Igamma": _get_jax_op(jax.lax.igamma, {"T"}),
    "Igammac": _get_jax_op(jax.lax.igammac, {"T"}),
    "Imag": _get_jax_op(jax.lax.imag, {"T", "Tout"}),
    "IsFinite": _get_jax_op(jnp.isfinite, {"T"}),
    "Invert": _get_jax_op(jnp.bitwise_not, {"T"}),
    "InvertPermutation": _get_jax_op(anp.invert_permutation, {"T"}),
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
    "PopulationCount": _get_jax_op(jax.lax.population_count, {"T"}),
    "Pow": _get_jax_op(anp.power, {"T"}),
    "Rank": _get_jax_op(lambda x: np.array(jnp.ndim(x)), {"T"}),
    "Real": _get_jax_op(jax.lax.real, {"T", "Tout"}),
    "ReadVariableOp": _get_jax_op(lambda x: x, {"dtype"}),
    "RealDiv": _get_jax_op(anp.true_divide, {"T"}),
    "Reciprocal": _get_jax_op(anp.reciprocal, {"T"}),
    "Relu": _get_jax_op(jax.nn.relu, {"T"}),
    "Relu6": _get_jax_op(jax.nn.relu6, {"T"}),
    "ReverseV2": _get_jax_op(anp.flip, {"T", "Tidx"}),
    "RFFT": _get_jax_op(
        functools.partial(jnp.fft.rfftn, axes=(-1,)), {"Tcomplex", "Treal"}),
    "RFFT2D": _get_jax_op(
        functools.partial(
            jnp.fft.rfftn, axes=(-2, -1,)), {"Tcomplex", "Treal"}),
    "RFFT3D": _get_jax_op(
        functools.partial(
            jnp.fft.rfftn, axes=(-3, -2, -1,)), {"Tcomplex", "Treal"}),
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
    "TensorListFromTensor": _get_jax_op(
        lambda xs, element_shape: xs, {"element_dtype", "shape_type"}),
    "Tile": _get_jax_op(anp.tile, {"T", "Tmultiples"}),
    "UnsortedSegmentMax": _get_jax_op(
        functools.partial(jax.ops.segment_max, indices_are_sorted=False),
        {"T", "Tindices", "Tnumsegments"}),
    "UnsortedSegmentMin": _get_jax_op(
        functools.partial(jax.ops.segment_min, indices_are_sorted=False),
        {"T", "Tindices", "Tnumsegments"}),
    "UnsortedSegmentProd": _get_jax_op(
        functools.partial(jax.ops.segment_prod, indices_are_sorted=False),
        {"T", "Tindices", "Tnumsegments"}),
    "UnsortedSegmentSum": _get_jax_op(
        functools.partial(jax.ops.segment_sum, indices_are_sorted=False),
        {"T", "Tindices", "Tnumsegments"}),
    "Where": _get_jax_op(jnp.argwhere, {"T"}),
    "ZerosLike": _get_jax_op(jnp.zeros_like, {"T"}),
    # The assignment logic is handled in _OpNode and convert().
    "AssignAddVariableOp": _get_jax_op(jnp.add, {"dtype"}),
    "AssignSubVariableOp": _get_jax_op(jnp.subtract, {"dtype"}),
    "AssignVariableOp": _get_jax_op(
        lambda var, x: x, {"dtype", "validate_shape"}),
}


def get_parser(op_name: str) -> Callable[..., Callable[..., Any]]:
  return _jax_ops[op_name]


def get_unsupported_operations(op_names: Sequence[str]) -> Set[str]:
  return {name for name in op_names if name not in _jax_ops}


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


class _LibraryFunction(Protocol):
  # Inputs corresponding to VarHandleOp
  variable_input_specs: Optional[Tuple[tf.TensorSpec, ...]] = None


@dataclasses.dataclass
class _HigherOrderFunction:
  """Base class for higher order ops."""
  inner_fn_names: Mapping[str, str]

  def get_inner_functions(
      self, library_functions: Mapping[str, Callable[..., Any]]
  ) -> Mapping[str, Callable[..., Any]]:
    return {k: library_functions[v] for k, v in self.inner_fn_names.items()}

  def get_additional_inputs(
      self, **inner_functions: _LibraryFunction
  ) -> Tuple[str, ...]:
    """Get additional input names required by the op."""
    del inner_functions
    return ()


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
  """Parse an Assert Op."""
  _check_attrs(proto, {"T", "summarize"})

  logging.warning("Assert has no effect.")

  def _func(cond, *data):
    del cond, data
    # TODO(shaobohou) Use checkify?
    return _EMPTY_RETURN_VALUE

  return _func


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


@register_operation("BatchToSpaceND")
def _batch_to_space_nd(proto):
  """Parse a BatchToSpaceND Op."""
  _check_attrs(proto, {"T", "Tblock_shape", "Tcrops"})

  def _func(
      operand: jnp.ndarray, block_shape: jnp.ndarray, crops: jnp.ndarray
  ) -> jnp.ndarray:
    batch, *other_shape = list(operand.shape)
    block_shape = block_shape.tolist()
    num_spatial = len(block_shape)
    crops = crops.tolist()
    spatial_shape = other_shape[:num_spatial]
    remaining_shape = other_shape[num_spatial:]

    new_shape = (
        block_shape
        + [batch // np.prod(block_shape)]
        + spatial_shape
        + remaining_shape
    )
    reshaped = operand.reshape(new_shape)

    permuted_axes = [num_spatial]
    for idx in range(num_spatial):
      permuted_axes.extend([idx + 1 + num_spatial, idx])
    permuted_axes.extend(list(range(num_spatial * 2 + 1, len(new_shape))))
    permuted = jnp.transpose(reshaped, axes=permuted_axes)

    uncropped_shape = [new_shape[num_spatial]]
    for idx in range(num_spatial):
      uncropped_shape.append(block_shape[idx] * spatial_shape[idx])
    uncropped_shape.extend(remaining_shape)
    uncropped = permuted.reshape(uncropped_shape)

    cropped_slice = [slice(None)]
    for idx in range(num_spatial):
      cropped_slice.append(
          slice(crops[idx][0], uncropped_shape[1 + idx] - crops[idx][1])
      )
    cropped_slice += [slice(None)] * len(remaining_shape)
    cropped = uncropped[tuple(cropped_slice)]

    return cropped

  return _func


@register_operation("SpaceToBatchND")
def _space_to_batch_nd(proto):
  """Parse a SpaceToBatchND Op."""
  _check_attrs(proto, {"T", "Tblock_shape", "Tpaddings"})

  def _func(
      operand: jnp.ndarray, block_shape: jnp.ndarray, paddings: jnp.ndarray
  ) -> jnp.ndarray:
    batch, *other_shape = list(operand.shape)
    block_shape = block_shape.tolist()
    num_spatial = len(block_shape)
    paddings = paddings.tolist()
    remaining_shape = other_shape[num_spatial:]

    paddings = [[0, 0]] + paddings + [[0, 0]] * len(remaining_shape)
    padded = jnp.pad(operand, paddings)
    padded_shape = padded.shape

    new_shape = [batch]
    for idx in range(num_spatial):
      new_shape.extend(
          [padded_shape[idx + 1] // block_shape[idx], block_shape[idx]]
      )
    new_shape.extend(remaining_shape)
    reshaped = padded.reshape(new_shape)

    permuted_axes = []
    for idx in range(num_spatial):
      permuted_axes.append(idx * 2 + 2)
    permuted_axes += [0]
    for idx in range(num_spatial):
      permuted_axes.append(idx * 2 + 1)
    permuted_axes.extend(list(range(num_spatial * 2 + 1, len(new_shape))))
    permuted = jnp.transpose(reshaped, axes=permuted_axes)

    flatten_shape = [batch * np.prod(block_shape)]
    for idx in range(num_spatial):
      flatten_shape.append(padded_shape[idx + 1] // block_shape[idx])
    flatten_shape.extend(remaining_shape)
    flattened = permuted.reshape(flatten_shape)

    return flattened

  return _func


@register_operation("BiasAdd")
def _bias_add(proto):
  """Parse a BiasAdd Op."""
  _check_attrs(proto, {"T", "data_format"})

  data_format = str(proto.attr["data_format"].s, "utf-8")
  if data_format == "NHWC":
    expand_axis_fn = lambda x: [d for d in range(x.ndim) if d != x.ndim - 1]
  elif data_format == "NCHW":
    # TODO(b/266553458) this seems wrong but matches TF behaviour.
    expand_axis_fn = lambda x: [d for d in range(x.ndim) if d != 1]
  else:
    raise ValueError(f"Found unsupported data format {data_format}.")

  def _func(value: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    if bias.ndim != 1:
      raise ValueError(
          f"Expected `bias` as a 1D array, found array with {bias.ndim} dims.")
    bias = anp.expand_dims(bias, axis=expand_axis_fn(value))
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


@register_operation("CheckNumerics")
def _check_numerics(proto):
  """Parse an CheckNumerics Op."""
  _check_attrs(proto, {"T", "message"})

  message = str(proto.attr["message"].s, "utf-8")

  def _func(operand: jnp.ndarray) -> jnp.ndarray:
    if config.get_config("enable_checkify_for_asserts"):
      checkify.check(
          jnp.logical_not(jnp.any(jnp.isnan(operand))),
          f"{message} : Found NaN values.")
      checkify.check(
          jnp.logical_not(jnp.any(jnp.isinf(operand))),
          f"{message} : Found Inf values.")
    return operand

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

  force_float32 = config.get_config("force_const_float32_to_bfloat16")
  force_float64 = config.get_config("force_const_float64_to_bfloat16")
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
    axis = axis.item()
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


def _div_no_nan_forward(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes a safe divide which returns 0 if y (denominator) is zero."""
  div = anp.divide(x, y)
  return jnp.where(y == 0.0, jnp.zeros_like(div), div)


@register_operation("DivNoNan")
def _div_no_nan(proto):
  """Parse a DivNoNan op."""
  _check_attrs(proto, {"T"})

  @jax.custom_gradient
  def _func(
      x: jnp.ndarray, y: jnp.ndarray
  ) -> Tuple[
      jnp.ndarray,
      Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]],
  ]:
    def _grad(g: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
      """Given upstream grad G and a Div op: Z = X/Y, the gradients are.

        dX = G / Y
        dY = -G*X / Y^2 = (X/Y) * -G / Y = -G*Z / Y

      Following tensorflow/tensorflow/c/experimental/gradients/math_grad.cc
      for handling NaNs.

      Args:
        g: Upstream gradient.

      Returns:
        forward value: div_no_nan(x, y)
        grad: Gradient information in TF format.
      """
      output = _div_no_nan_forward(x, y)
      dx = _div_no_nan_forward(g, y)
      dy = _div_no_nan_forward(-g * output, y)
      return dx, dy

    return _div_no_nan_forward(x, y), _grad

  return _func


@register_operation("Eig")
def _eig(proto):
  """Parse an Eig Op."""
  _check_attrs(proto, {"T", "Tout", "compute_v"})

  compute_v = proto.attr["compute_v"].b

  def _func(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if compute_v:
      evals, evecs = jnp.linalg.eig(x)
    else:
      evals = jnp.linalg.eigvals(x)
      evecs = jnp.zeros(shape=(), dtype=evals.dtype)

    # Sorting by eigenvalues to match tf.raw_ops.Eig better.
    sort_fn = lambda vals, inds: vals[..., inds]
    for _ in range(len(x.shape) - 2):
      sort_fn = jax.vmap(sort_fn, in_axes=(0, 0))
    einds = jnp.argsort(jnp.absolute(evals), axis=-1)
    evals = sort_fn(evals, einds)

    if compute_v:
      evecs = sort_fn(evecs, einds)

    return evals, evecs

  return _func


@register_operation("SelfAdjointEigV2")
def _eigh(proto):
  """Parse an SelfAdjointEigV2 Op."""
  _check_attrs(proto, {"T", "compute_v"})

  compute_v = proto.attr["compute_v"].b

  def symmetrize(x):
    return jnp.tril(x) + jnp.swapaxes(jnp.tril(x, -1), -2, -1)

  def _func(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if compute_v:
      evals, evecs = jnp.linalg.eigh(x, symmetrize_input=False)
    else:
      # symmetrize_input does not exist for eigvalsh.
      # See https://github.com/google/jax/issues/9473
      evals, evecs = jnp.linalg.eigvalsh(symmetrize(x)), None

    # Sorting by eigenvalues to tf.raw_ops.Eig better.
    sort_fn = lambda vals, inds: vals[..., inds]
    for _ in range(len(x.shape) - 2):
      sort_fn = jax.vmap(sort_fn, in_axes=(0, 0))
    einds = jnp.argsort(evals, axis=-1)
    evals = sort_fn(evals, einds)
    if compute_v:
      evecs = sort_fn(evecs, einds)
    else:
      evecs = jnp.zeros(shape=(), dtype=evals.dtype)

    return evals, evecs

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


@register_operation("EnsureShape")
def _ensure_shape(proto):
  """Parse an EnsureShape Op."""
  _check_attrs(proto, {"T", "shape"})

  shape = [dim.size for dim in proto.attr["shape"].shape.dim]

  def is_compatible(x, s):
    return x == -1 or s == -1 or x == s

  def _func(x: jnp.ndarray) -> jnp.ndarray:
    if len(x.shape) != len(shape) or not all(
        is_compatible(x, s) for x, s in safe_zip(x.shape, shape)
    ):
      raise ValueError(f"Expected shape={shape}, found {x.shape}.")
    return x

  return _func


@register_operation("ExpandDims")
def _expand_dims(proto):
  """Parse an ExpandDims Op."""
  _check_attrs(proto, {"T", "Tdim"})

  def _func(arr: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.expand_dims(arr, axis=axis.tolist())

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
@register_operation("FusedBatchNorm")
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


@register_operation("ResourceGather")
@register_operation("GatherV2")
def _gather(proto):
  """Parse a GatherV2 Op."""
  _check_attrs(proto, {
      "Taxis", "Tindices", "Tparams", "batch_dims", "dtype", "validate_indices"
  })

  batch_dims = proto.attr["batch_dims"].i
  if batch_dims < 0:
    raise ValueError(f"batch_dims={batch_dims} must be non-negative.")

  # TODO(b/249826984) Add test for ResourceGather, dtype and validate_indices.

  def _func(
      params: jnp.ndarray,
      indices: jnp.ndarray,
      axis: ArrayLike = np.array(0, dtype=np.int32),
  ) -> jnp.ndarray:
    return anp.gather(
        params, indices, axis=axis.item(), batch_dims=batch_dims)

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


@register_operation("LeakyRelu")
def _leaky_relu(proto):
  """Parse a LeakyRelu Op."""
  _check_attrs(proto, {"T", "alpha"})

  alpha = proto.attr["alpha"].f

  def _func(features: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.leaky_relu(features, alpha)

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

  # TODO(b/266553251) Add test for arrays with complex values.
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


@register_operation("MatrixSetDiagV3")
def _matrix_set_diag(proto):
  """Parse a MatrixSetDiagV3 op."""
  _check_attrs(proto, {"T", "align"})

  align = str(proto.attr["align"].s, "utf-8")
  if align != "RIGHT_LEFT":
    raise ValueError(f"MatrixSetDiagV3 does not support `align={align}` yet.")

  def _func(
      inputs: jnp.ndarray,
      diagonals: jnp.ndarray,
      k: jnp.ndarray,
  ) -> jnp.ndarray:
    k = k.item()

    def diag_fn(inps: jnp.ndarray, diag: jnp.ndarray) -> jnp.ndarray:
      assert len(inps.shape) == 2, inps.shape
      assert len(diag.shape) == 1, diag.shape
      if ((diag.shape[0] + k > inps.shape[1]) or
          (diag.shape[0] - k > inps.shape[0])):
        raise ValueError(
            f"Incompatible inputs shape ({inputs.shape}) and diagonals shape "
            f"({diagonals.shape}).")

      return jnp.diagflat(diag, k=k)[:inps.shape[0], :inps.shape[1]]

    for _ in range(len(diagonals.shape) - 1):
      diag_fn = jax.vmap(diag_fn)

    outputs = diag_fn(inputs, diagonals)
    mask = diag_fn(inputs, jnp.ones_like(diagonals, dtype=jnp.bool_))
    return jnp.where(mask, outputs, inputs)

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
    lower = lower.item() + 1 if lower.item() >= 0 else max(mask_shape)
    mask_lower = jnp.tril(jnp.ones(mask_shape, jnp.int32), -lower)
    upper = upper.item() + 1 if upper.item() >= 0 else max(mask_shape)
    mask_upper = jnp.triu(jnp.ones(mask_shape, jnp.int32), upper)
    return jnp.where((mask_lower + mask_upper) == 0, x, 0)

  return _func


@register_operation("MatrixTriangularSolve")
def _matrix_triangular_solve(proto):
  """Parse an MatrixTriangularSolve Op."""
  _check_attrs(proto, {"T", "lower", "adjoint"})

  lower = proto.attr["lower"].b
  adjoint = proto.attr["adjoint"].b

  def _func(arr: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.linalg.triangular_solve(
        arr,
        rhs,
        lower=lower,
        transpose_a=adjoint,
        conjugate_a=adjoint,
        left_side=True)

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

    mask = jax.nn.one_hot(indices, num_classes=depth.item(), dtype=jnp.int32)
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

  def get_additional_inputs(
      self, **inner_functions: _LibraryFunction
  ) -> Tuple[str, ...]:
    """Get additional input names required by the op."""
    return self._get_additional_inputs(**inner_functions)

  def _get_additional_inputs(
      self, inner_fn: _LibraryFunction
  ) -> Tuple[str, ...]:
    return tuple(spec.name for spec in inner_fn.variable_input_specs)


@register_operation("StatefulPartitionedCall")
@register_operation("PartitionedCall")
def _partitioned_call(proto):
  """Parse a PartitionedCall op."""
  _check_attrs(proto,
               {"f", "Tin", "Tout", "config", "config_proto", "executor_type"})

  inner_fn = proto.attr["f"].func.name
  op_config = str(proto.attr["config"].s, "utf-8")
  op_config_proto = proto.attr["config_proto"].s
  executor_type = str(proto.attr["executor_type"].s, "utf-8")
  del op_config, op_config_proto, executor_type

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

  @jax.custom_gradient
  def _raise_func(
      operand: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Callable[..., Any]]:
    def grad_fn(_):
      raise LookupError(f"Gradient explicitly prevented on node {proto.name}. "
                        f"Original reason: {message}")
    return operand, grad_fn

  def _warn_func(operand: jnp.ndarray) -> jnp.ndarray:
    logging.warning("PreventGradient ignored on node %s. Original reason: %s",
                    proto.name, message)
    return operand

  if config.get_config("raise_on_prevent_gradient"):
    return _raise_func
  else:
    return _warn_func


@register_operation("Prod")
def _prod(proto):
  """Parse a Prod op."""
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.prod(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("Qr")
def _qr(proto):
  """Parse an QR Op."""
  _check_attrs(proto, {"T", "full_matrices"})

  full_matrices = proto.attr["full_matrices"].b

  def _func(x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.linalg.qr(x, mode="complete" if full_matrices else "reduced")

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
def _resize_bilinear(proto):
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
        precision=config.get_config("resize_bilinear_precision"),
    )

  return _func


@register_operation("ResizeNearestNeighbor")
def _resize_nearest_neighbor(proto):
  """Parse a ResizeNearestNeighbor op."""
  _check_attrs(proto, {"T", "align_corners", "half_pixel_centers"})

  align_corners = proto.attr["align_corners"].b
  half_pixel_centers = proto.attr["half_pixel_centers"].b

  if align_corners or not half_pixel_centers:
    # Not supported by tf.raw_ops.ResizeNearestNeighbor.
    raise ValueError(
        "Only align_corners=False and half_pixel_centers=True is supported.")

  def _func(images: jnp.ndarray, size: jnp.ndarray) -> jnp.ndarray:
    if len(images.shape) != 4:
      raise ValueError(
          "Expected A 4D tensor with shape [batch, height, width, channels], "
          f"found {images.shape}")

    inp_batch, _, _, inp_channels = images.shape
    out_height, out_width = size.tolist()

    return jax.image.resize(
        images,
        shape=(inp_batch, out_height, out_width, inp_channels),
        method=jax.image.ResizeMethod.NEAREST,
    )

  return _func


@register_operation("Roll")
def _roll(proto):
  """Parse a Roll Op."""
  _check_attrs(proto, {"T", "Tshift", "Taxis"})

  def _func(
      operand: jnp.ndarray,
      shift: jnp.ndarray,
      axis: jnp.ndarray,
  ) -> jnp.ndarray:
    return anp.roll(operand, shift=shift.tolist(), axis=axis.tolist())

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
    return anp.scatter_nd(indices, updates, shape)

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
    slices = [slice(b, b + s) for b, s in safe_zip(begins, sizes)]
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
    axis = axis.item()
    defined_size = sum([x for x in splits if x >= 0])
    splits = [x if x >= 0 else value.shape[axis] - defined_size for x in splits]
    indices = np.cumsum(np.array(splits), axis=0)
    assert indices[-1] == value.shape[axis]
    return anp.split(value, indices[:-1], axis=axis)

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


@register_operation("StatelessRandomGetAlg")
def _stateless_random_get_alg(proto):
  _check_attrs(proto, set({}))
  # Only jax.lax.rng_bit_generator allows users to choose the algorithm.
  # So this will most likely be ignored.
  return lambda: tf.random.Algorithm.AUTO_SELECT.value


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
    shape = (num_samples.item(), logits.shape[0])
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
      shape: Sequence[int],
      key: jnp.ndarray,
      counter: jnp.ndarray,
      alg: jnp.ndarray,
  ) -> jnp.ndarray:
    del counter, alg  # TODO(b/266553394) combine key and counter?
    return jax.random.normal(key=key, shape=shape, dtype=jax_dtype)

  return _func


@register_operation("StatelessRandomUniformV2")
def _stateless_random_uniform_v2(proto):
  """Parse a StatelessRandomNormalV2 op."""
  _check_attrs(proto, {"T", "Tshape", "dtype"})

  dtype = tf.as_dtype(proto.attr["dtype"].type)
  jax_dtype = anp.get_jax_dtype(dtype)

  def _func(
      shape: Sequence[int],
      key: jnp.ndarray,
      counter: jnp.ndarray,
      alg: jnp.ndarray,
  ) -> jnp.ndarray:
    del counter, alg  # TODO(b/266553394) combine key and counter?
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
    del counter, alg  # TODO(b/266553394) combine key and counter?
    return jax.random.randint(
        key=key,
        shape=shape.tolist(),
        minval=minval,
        maxval=maxval,
        dtype=jax_dtype,
    )

  return _func


def _split_args(
    args: Sequence[Any],
    preds: Sequence[bool],
) -> Tuple[Tuple[Tuple[int, Any]], ...]:
  """Split args into two lists based on some predicates."""
  assert len(args) == len(preds), (len(args), len(preds))
  true_args = []
  false_args = []
  for idx, (arg, pred) in enumerate(safe_zip(args, preds)):
    if pred:
      true_args.append((idx, arg))
    else:
      false_args.append((idx, arg))
  return tuple(true_args), tuple(false_args)


def _merge_args(
    const_args: Sequence[Tuple[int, Any]],
    trace_args: Sequence[Tuple[int, Any]],
) -> Tuple[Any]:
  args = [None] * (len(const_args) + len(trace_args))
  for idx, arg in tuple(const_args) + tuple(trace_args):
    args[idx] = arg
  assert all(x is not None for x in args)
  return tuple(args)


class _StatelessWhile(_HigherOrderFunction):
  """Represents a StatelessWhile Op."""

  def __call__(self, *all_args, cond_fun, body_fun, rng=None):
    # Pull captured arguments out of inputs to cond and body.
    const_idxs_args, trace_idxs_args = _split_args(
        all_args, body_fun.output_is_input)
    trace_idxs, trace_args = safe_zip(*trace_idxs_args)

    def real_cond(args):
      *cond_args, rng = args
      cond_idxs_args = tuple(safe_zip(trace_idxs, cond_args))
      cond_args = _merge_args(const_idxs_args, cond_idxs_args)
      _, cond_key, _ = [None] * 3 if rng is None else jax.random.split(rng, 3)
      outputs = cond_fun(*cond_args, rng=cond_key)
      if len(outputs) != 1:
        raise ValueError(
            f"Expected cond_fun to return a single value, found {outputs}")
      return outputs[0]

    def real_body(args):
      *body_args, rng = args
      body_idxs_args = tuple(safe_zip(trace_idxs, body_args))
      body_args = _merge_args(const_idxs_args, body_idxs_args)
      key, _, body_key = [None] * 3 if rng is None else jax.random.split(rng, 3)
      outputs = body_fun(*body_args, rng=body_key)
      outputs = tuple([outputs[idx] for idx in trace_idxs])
      return outputs + (key,)

    def maybe_initialise(arg, spec):
      # TensorList inputs can be uninitialized.
      if isinstance(spec, jax.ShapeDtypeStruct) and arg.size == 0:
        return jnp.zeros(shape=spec.shape, dtype=spec.dtype)
      else:
        return arg

    output_specs = jax.eval_shape(real_body, trace_args + (rng,))
    trace_args = tuple(
        maybe_initialise(arg, spec)
        for arg, spec in safe_zip(trace_args, output_specs[:-1]))

    outputs = jax.lax.while_loop(real_cond, real_body, trace_args + (rng,))
    *outputs, _ = outputs  # Drop rng.
    outputs = _merge_args(
        const_idxs_args, tuple(safe_zip(trace_idxs, outputs))
    )
    assert len(outputs) == len(all_args), (len(outputs), len(all_args))
    return outputs


@register_operation("StatelessWhile")
@register_operation("While")
def _stateless_while(proto):
  """Parse a StatelessWhile op."""
  _check_attrs(proto,
               {"T", "body", "cond", "parallel_iterations", "output_shapes"})
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
              jax.core.symbolic_equal_dim(x_dim if end_dim is None else end_dim,
                                          x_dim)):
            slices.append(slice(None, None, None))
          else:
            slices.append(slice(beg_dim, end_dim, stride))
        dim += 1

    # TODO(b/266553742) Handle stride=1 slicing along polymoprhic dimensions.
    return x[tuple(slices)]

  return _func


@register_operation("Sum")
def _sum(proto):
  _check_attrs(proto, {"T", "Tidx", "keep_dims"})

  keep_dims = proto.attr["keep_dims"].b

  def _func(x: jnp.ndarray, axis: jnp.ndarray) -> jnp.ndarray:
    return anp.sum_(x, axis=axis.tolist(), keepdims=keep_dims)

  return _func


@register_operation("Svd")
def _svd(proto):
  """Parse an Svd Op."""
  _check_attrs(proto, {"T", "compute_uv", "full_matrices"})

  compute_uv = proto.attr["compute_uv"].b
  full_matrices = proto.attr["full_matrices"].b

  def _func(
      x: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:
    res = jnp.linalg.svd(
        x, compute_uv=compute_uv, full_matrices=full_matrices)
    if compute_uv:
      u, s, vh = res
      v = jnp.conjugate(jnp.swapaxes(vh, -1, -2))
    else:
      u, s, v = None, res, None

    return s, u, v

  return _func


@register_operation("TensorListGetItem")
def _tensor_list_get_item(proto):
  """Parse an TensorListGetItem Op."""
  _check_attrs(proto, {"element_dtype"})

  dtype = tf.as_dtype(proto.attr["element_dtype"].type)
  dtype = anp.get_jax_dtype(dtype)

  def _func(
      xs: jnp.ndarray,
      index: jnp.ndarray,
      element_shape: jnp.ndarray,
  ) -> jnp.ndarray:
    assert xs.dtype == dtype
    if xs.size == 0:
      return np.zeros(element_shape, dtype=dtype)
    if isinstance(index, jax.Array):
      xs = jnp.array(xs)
    return xs[index]

  return _func


def _create_tensor_list(
    num_elements: int,
    shape: Union[int, List[int]],
    dtype: Any,
) -> np.ndarray:
  """Create a tensor corresponding to a stacked tensor list."""
  if not isinstance(shape, list):
    shape = [shape]
  # shape can be unspecified as -1.
  shape = [max(0, sz) for sz in shape]
  return np.zeros([num_elements] + shape, dtype=dtype)


@register_operation("TensorListReserve")
def _tensor_list_reserve(proto):
  """Parse an TensorListReserve Op."""
  _check_attrs(proto, {"element_dtype", "shape_type"})

  dtype = tf.as_dtype(proto.attr["element_dtype"].type)
  dtype = anp.get_jax_dtype(dtype)

  def _func(shape: jnp.ndarray, num_elements: jnp.ndarray) -> ArrayLike:
    return _create_tensor_list(
        num_elements.item(), shape.tolist(), dtype=dtype)

  return _func


@register_operation("TensorListSetItem")
def _tensor_list_set_item(proto):
  """Parse an TensorListSetItem Op."""
  _check_attrs(proto, {"element_dtype", "resize_if_index_out_of_bounds"})

  if proto.attr["resize_if_index_out_of_bounds"].b:
    raise ValueError(
        "resize_if_index_out_of_bounds=True in TensorListSetItem is not yet"
        " supported."
    )

  dtype = tf.as_dtype(proto.attr["element_dtype"].type)
  dtype = anp.get_jax_dtype(dtype)

  def _func(
      xs: jnp.ndarray,
      index: jnp.ndarray,
      item: jnp.ndarray,
  ) -> jnp.ndarray:
    assert xs.shape
    if xs.size == 0:
      xs = _create_tensor_list(xs.shape[0], list(item.shape), dtype=dtype)
    if isinstance(index, jax.Array):
      xs = jnp.array(xs)
    return xs.at[index].set(item)

  return _func


@register_operation("TensorListStack")
def _tensor_list_stack(proto):
  """Parse an TensorListStack Op."""
  _check_attrs(proto, {"element_dtype", "num_elements"})

  dtype = tf.as_dtype(proto.attr["element_dtype"].type)
  dtype = anp.get_jax_dtype(dtype)

  def _func(xs: jnp.ndarray, shape: jnp.ndarray) -> ArrayLike:
    if xs.size == 0:
      xs = _create_tensor_list(xs.shape[0], shape.tolist(), dtype=dtype)
    return xs

  return _func


@register_operation("TensorScatterUpdate")
def _tensor_scatter_update(proto):
  """Parse an TensorScatterUpdate Op."""
  _check_attrs(proto, {"T", "Tindices"})

  def _func(
      operand: jnp.ndarray,
      indices: jnp.ndarray,
      updates: jnp.ndarray,
  ) -> jnp.ndarray:
    dimension_numbers = jax.lax.ScatterDimensionNumbers(
        range(1, updates.ndim), range(indices.shape[-1]),
        range(indices.shape[-1]))
    return jax.lax.scatter(
        operand, indices, updates, dimension_numbers=dimension_numbers)

  return _func


@register_operation("TopKV2")
def _top_k(proto):
  _check_attrs(proto, {"T", "sorted", "Tk", "index_type"})
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

  def _func(x: jnp.ndarray) -> List[jnp.ndarray]:
    if x.shape[axis] != num:
      raise ValueError("Unpack expects dimension of {num} for axis={axis}, "
                       "found {x.shape[axis]}, shape={x.shape}")
    return [anp.squeeze(v, axis=axis) for v in anp.split(x, num, axis=axis)]

  return _func


# TODO(b271811043) Add unit test.
@register_operation("VarHandleOp")
def _var_handle(proto):
  _check_attrs(
      proto, {"shared_name", "container", "allowed_devices", "shape", "dtype"})

  def _func():
    raise ValueError(f"VarHandleOp `{proto.name}` cannot be evaluated.")

  return _func


@register_operation("VariableV2")
def _variable_v2(proto):
  _check_attrs(proto, {"container", "dtype", "shared_name", "shape"})

  name = proto.name

  def _func():
    raise ValueError(f"VariableV2 `{name}` cannot be evaluated.")

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
        feature_group_count=feature_group_count.item(),
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


@register_operation("XlaOptimizationBarrier")
def _xla_optimization_barrier(proto):
  """Parse a XlaOptimizationBarrier op."""
  _check_attrs(proto, {"T"})

  def _func(*operands: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    # TODO(b/241584320) Note this does not reproduce the remat transform in the
    # forward pass, which may require some heurstics when parsing the graphdef.
    return lax_control_flow.optimization_barrier_p.bind(*operands)

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


@register_operation("XlaReducePrecision")
def _xla_reduce_precision(proto):
  """Parse a XlaReducePrecision op."""
  _check_attrs(proto, {"T", "exponent_bits", "mantissa_bits"})

  exponent = proto.attr["exponent_bits"].i
  mantissa = proto.attr["mantissa_bits"].i

  def _func(operand: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.reduce_precision(operand, exponent, mantissa)

  return _func


@register_operation("XlaSharding")
def _xla_sharding(proto):
  """Parse a XlaSharding op."""
  _check_attrs(proto, {"T", "sharding", "unspecified_dims"})

  unspecified_dims = tuple(proto.attr["unspecified_dims"].list.i)
  if unspecified_dims:
    raise ValueError(f"{unspecified_dims=} is not yet supported.")

  sharding_str = proto.attr["sharding"].s

  # Return identity if sharding annotation is empty.
  if not sharding_str:
    return lambda x: x

  sharding = xla_client.OpSharding()
  sharding.ParseFromString(sharding_str)
  jax_sharding = jax.sharding.GSPMDSharding(jax.devices(), sharding)

  # TODO(b/235450851) Remove jax.jit once wsc is usable outside of jit.
  @jax.jit
  def _func(x: jnp.ndarray) -> jnp.ndarray:
    return jax.lax.with_sharding_constraint(x, jax_sharding)

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
    dimension = args[-1].item()

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


# Taken from https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L1056
def _get_max_identity(dtype):
  if jax.dtypes.issubdtype(dtype, np.inexact):
    return np.array(-np.inf, dtype)
  elif jax.dtypes.issubdtype(dtype, np.integer):
    return np.array(jax.dtypes.iinfo(dtype).min, dtype)
  elif jax.dtypes.issubdtype(dtype, np.bool_):
    return np.array(False, np.bool_)


# Taken from https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L1064
def _get_min_identity(dtype):
  if jax.dtypes.issubdtype(dtype, np.inexact):
    return np.array(np.inf, dtype)
  elif jax.dtypes.issubdtype(dtype, np.integer):
    return np.array(jax.dtypes.iinfo(dtype).max, dtype)
  elif jax.dtypes.issubdtype(dtype, np.bool_):
    return np.array(True, np.bool_)


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
    window_dimensions = window_dimensions.tolist()
    window_strides = window_strides.tolist()
    padding = padding.tolist()
    base_dilation = base_dilation.tolist()
    window_dilation = window_dilation.tolist()

    # Pattern matching computations that can be specialized.
    primitives = {
        jax.lax.max_p: jax.lax.max,
        jax.lax.min_p: jax.lax.min,
        jax.lax.add_p: jax.lax.add,
        jax.lax.mul_p: jax.lax.mul,
    }
    computation_jaxpr = jax.make_jaxpr(computation)(init_value, init_value)
    computation_eqn = _maybe_get_jaxpreqn(computation_jaxpr)
    if computation_eqn is not None and computation_eqn.primitive in primitives:
      computation_fn = primitives[computation_eqn.primitive]
    else:
      computation_fn = lambda *args: computation(*args)[0]
      logging.info("Calling reduce_window with the following computation:\n%s",
                   computation_jaxpr)

    def infer_cumulative_reduction():
      ndims = len(window_dimensions)
      assert ndims == operand.ndim
      reduce_axis = np.argmax(window_dimensions)
      reduce_dim = operand.shape[reduce_axis]
      dims = [1] * ndims
      dims[reduce_axis] = reduce_dim
      if not (window_dimensions == dims and window_strides == [1] * ndims and
              base_dilation == [1] * ndims and window_dilation == [1] * ndims):
        return (None, None, None)

      # Determine direction of reduction.
      normal_padding = [[0, 0]] * ndims
      normal_padding[reduce_axis] = [reduce_dim - 1, 0]
      reverse_padding = [[0, 0]] * ndims
      reverse_padding[reduce_axis] = [0, reduce_dim - 1]
      reverse = None
      if padding == normal_padding:
        reverse = False
      elif padding == reverse_padding:
        reverse = True
      else:
        return (None, None, None)

      if computation_fn is jax.lax.add and init_value == 0:
        return (jax.lax.cumsum, reduce_axis, reverse)
      elif computation_fn is jax.lax.mul and init_value == 1:
        return (jax.lax.cumprod, reduce_axis, reverse)
      elif (computation_fn is jax.lax.max and
            init_value == _get_max_identity(operand.dtype)):
        return (jax.lax.cummax, reduce_axis, reverse)
      elif (computation_fn is jax.lax.min and
            init_value == _get_min_identity(operand.dtype)):
        return (jax.lax.cummin, reduce_axis, reverse)
      else:
        return (None, None, None)

    reducer, reduce_axis, reverse = infer_cumulative_reduction()
    if reducer and config.get_config("infer_cumulative_reduction_from_jax2tf"):
      return reducer(operand, axis=reduce_axis, reverse=reverse)
    else:
      return jax.lax.reduce_window(
          operand,
          init_value,
          computation=computation_fn,
          window_dimensions=window_dimensions,
          window_strides=window_strides,
          padding=[tuple(v) for v in padding],
          base_dilation=base_dilation,
          window_dilation=window_dilation)


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
        algorithm=xla_utils.get_random_algorithm_from_tf(algorithm.item()),
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

    # TODO(b/266553393) Support jax.lax.add for AvgPool.
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


def _searchsorted(a: jnp.ndarray, v: jnp.ndarray, side: str):
  """Vmapped version of searchsorted to implement LowerBound and UpperBound."""
  return jax.vmap(
      functools.partial(jnp.searchsorted, side=side),
      in_axes=0,
      out_axes=0,
  )(a, v)


def _lower_upper_bound(proto, side: str):
  """Parse a LowerBound or UpperBound op using searchsorted."""
  _check_attrs(proto, {"T", "Tvalues", "out_type"})
  dtype = tf.as_dtype(proto.attr["out_type"].type)
  if dtype != tf.int32:
    raise ValueError(
        f"Return type {dtype} not supported for LowerBound and UpperBound.")
  return lambda a, v: _searchsorted(a, v, side=side)


@register_operation("LowerBound")
def _lower_bound(proto):
  """Parse a LowerBound op."""
  return _lower_upper_bound(proto, side="left")


@register_operation("UpperBound")
def _upper_bound(proto):
  """Parse an UpperBound op."""
  return _lower_upper_bound(proto, side="right")
