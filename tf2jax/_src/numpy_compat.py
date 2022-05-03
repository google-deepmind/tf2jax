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
"""Utililties for wrapping around numpy and jax.numpy."""

import itertools

from typing import Sequence, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


tf_to_np_dtypes = {
    tf.bool: np.bool_,
    tf.uint8: np.uint8,
    tf.uint16: np.uint16,
    tf.uint32: np.uint32,
    tf.uint64: np.uint64,
    tf.int8: np.int8,
    tf.int16: np.int16,
    tf.int32: np.int32,
    tf.int64: np.int64,
    tf.float16: np.float16,
    tf.float32: np.float32,
    tf.float64: np.float64,
    tf.complex64: np.complex64,
    tf.complex128: np.complex128,
}


tf_to_jnp_dtypes = {
    tf.bool: jnp.bool_,
    tf.uint8: jnp.uint8,
    tf.uint16: jnp.uint16,
    tf.uint32: jnp.uint32,
    tf.uint64: jnp.uint64,
    tf.int8: jnp.int8,
    tf.int16: jnp.int16,
    tf.int32: jnp.int32,
    tf.int64: jnp.int64,
    tf.bfloat16: jnp.bfloat16,
    tf.float16: jnp.float16,
    tf.float32: jnp.float32,
    tf.float64: jnp.float64,
    tf.complex64: jnp.complex64,
    tf.complex128: jnp.complex128,
}


_NP_LIKES = (np.ndarray, np.number, np.bool_, bool, int, float)


# We avoids importing jax2tf.shape_poly.is_poly_dim as jax2tf often depends
# on the most recent tensorflow version.
def _is_poly_dim(x):
  cls = type(x)
  return (cls.__module__ == "jax.experimental.jax2tf.shape_poly" and
          cls.__name__ == "_DimPolynomial")


def _get_np(*args):
  """Select numpy backend based on input types."""
  no_jax = all((isinstance(x, _NP_LIKES) or _is_poly_dim(x)) for x in args)
  return np if no_jax else jnp


def _get_dtypes(*args):
  """Select dtypes backend based on input types."""
  no_jax = _get_np(*args) is np
  return tf_to_np_dtypes if no_jax else tf_to_jnp_dtypes


def get_jax_dtype(dtype: tf.DType):
  return tf_to_jnp_dtypes[dtype]


# Simple numeric ops.
add = lambda x, y: _get_np(x, y).add(x, y)
equal = lambda x, y: _get_np(x, y).equal(x, y)
floor_divide = lambda x, y: _get_np(x, y).floor_divide(x, y)
greater = lambda x, y: _get_np(x, y).greater(x, y)
greater_equal = lambda x, y: _get_np(x, y).greater_equal(x, y)
less = lambda x, y: _get_np(x, y).less(x, y)
less_equal = lambda x, y: _get_np(x, y).less_equal(x, y)
maximum = lambda x, y: _get_np(x, y).maximum(x, y)
minimum = lambda x, y: _get_np(x, y).minimum(x, y)
mod = lambda x, y: _get_np(x, y).mod(x, y)
multiply = lambda x, y: _get_np(x, y).multiply(x, y)
power = lambda x, y: _get_np(x, y).power(x, y)
negative = lambda x: _get_np(x).negative(x)
not_equal = lambda x, y: _get_np(x, y).not_equal(x, y)
reciprocal = lambda x: _get_np(x).reciprocal(x)
subtract = lambda x, y: _get_np(x, y).subtract(x, y)
true_divide = lambda x, y: _get_np(x, y).true_divide(x, y)


def divide(x, y):
  np_ = _get_np(x, y)
  dtype = np_.result_type(x.dtype, y.dtype)
  # This is necessary because for integers, divide behaves like true_divide.
  return np_.asarray(np_.divide(x, y), dtype)


def arange(start, stop, step, dtype: tf.DType):
  dtype = _get_dtypes(start, stop, step)[dtype]
  return _get_np(start, stop, step).arange(start, stop, step, dtype=dtype)


def asarray(arr, dtype: tf.DType):
  if _is_poly_dim(arr):
    arr = jax.core.dimension_as_value(arr)
  dtype = _get_dtypes(arr)[dtype]
  return _get_np(arr).asarray(arr, dtype)


def empty(shape, dtype: tf.DType, init: bool):
  del init
  dtype = _get_dtypes(shape)[dtype]
  return _get_np(shape).full(shape, dtype(), dtype=dtype)


def full(shape, fill_value, dtype: tf.DType):
  dtype = _get_dtypes(shape)[dtype]
  return _get_np(shape, fill_value).full(shape, fill_value, dtype=dtype)


def gather(params, indices, axis: int, batch_dims: int):
  """Implement gather in numpy and jax.numpy."""
  if _get_np(params, indices) is np:
    axis = axis if axis >= 0 else params.ndim + axis
    batch_shape = params.shape[:batch_dims]
    take_shape = (
        params.shape[:axis] + indices.shape[batch_dims:] +
        params.shape[axis + 1:])

    if batch_shape:
      res = np.zeros(take_shape, dtype=params.dtype)
      for idx in itertools.product(*[range(v) for v in batch_shape]):
        res[idx] = np.take(params[idx], indices[idx], axis=axis - batch_dims)
    else:
      res = np.take(params, indices, axis=axis)

    return res
  else:
    axis = axis if axis < 0 else -(params.ndim - axis)
    take_fn = lambda p, i: jnp.take(p, i, axis=axis)
    for _ in range(batch_dims):
      take_fn = jax.vmap(take_fn)
    return take_fn(params, indices)


# Reduction ops.
def all_(arr, axis: Union[int, Sequence[int]], keepdims: bool):
  axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
  return _get_np(arr).all(arr, axis=axis, keepdims=keepdims)


def any_(arr, axis: Union[int, Sequence[int]], keepdims: bool):
  axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
  return _get_np(arr).any(arr, axis=axis, keepdims=keepdims)


def cumsum(arr, axis: int):
  return _get_np(arr).cumsum(arr, axis=axis)


def max_(arr, axis: Union[int, Sequence[int]], keepdims: bool):
  axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
  return _get_np(arr).max(arr, axis=axis, keepdims=keepdims)


def min_(arr, axis: Union[int, Sequence[int]], keepdims: bool):
  axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
  return _get_np(arr).min(arr, axis=axis, keepdims=keepdims)


def prod(arr, axis: Union[int, Sequence[int]], keepdims: bool):
  axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
  return  _get_np(arr).prod(arr, axis=axis, keepdims=keepdims)


def sum_(arr, axis: Union[int, Sequence[int]], keepdims: bool):
  axis = tuple(axis) if isinstance(axis, (list, tuple)) else (axis,)
  return  _get_np(arr).sum(arr, axis=axis, keepdims=keepdims)


# Array manipulation ops.
def broadcast_to(arr, shape):
  np_ = jnp if any([_is_poly_dim(x) for x in shape]) else _get_np(arr)
  return np_.broadcast_to(arr, shape)

concatenate = lambda arrs, axis: _get_np(*arrs).concatenate(arrs, axis=axis)
expand_dims = lambda arr, axis: _get_np(arr).expand_dims(arr, axis=axis)
flip = lambda arr, axis: _get_np(arr).flip(arr, axis=axis)
split = lambda arr, sections, axis: _get_np(arr).split(arr, sections, axis=axis)
squeeze = lambda arr, axis: _get_np(arr).squeeze(arr, axis=axis)
stack = lambda arrs, axis: _get_np(*arrs).stack(arrs, axis=axis)
tile = lambda arr, reps: _get_np(arr, reps).tile(arr, reps=reps)
where = lambda cond, x, y: _get_np(cond, x, y).where(cond, x, y)


def moveaxis(
    arr,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
):
  return _get_np(arr).moveaxis(arr, source=source, destination=destination)
