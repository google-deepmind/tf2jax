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
"""JAX util functions."""

from typing import Optional, Sequence, Union

import jax
import tensorflow as tf

# TODO(shaobohou) Is there a non-direct alias?
from tensorflow.compiler.xla import xla_data_pb2  # pytype: disable=import-error


def get_conv_sequence(
    value: Union[int, Sequence[int]],
    ndim: int,
    channel_index: int,
) -> Union[int, Sequence[int]]:
  """Format strides or dilation sequences for conv operations."""
  if isinstance(value, int):
    return [value] * ndim
  elif len(value) == 1:
    return value * ndim
  elif len(value) == ndim:
    return value
  elif len(value) == ndim + 2:
    return value[2:] if channel_index == 1 else value[1:-1]
  else:
    raise ValueError(f"Unexpected sequence={value}.")


def convolution_dimension_numbers_from_proto(
    message) -> jax.lax.ConvDimensionNumbers:
  """Converts a XLA ConvolutionDimensionNumbers to a ConvDimensionNumbers."""
  proto = xla_data_pb2.ConvolutionDimensionNumbers().FromString(message)
  lhs_spec = ((proto.input_batch_dimension, proto.input_feature_dimension) +
              tuple(proto.input_spatial_dimensions))
  rhs_spec = ((proto.kernel_output_feature_dimension,
               proto.kernel_input_feature_dimension) +
              tuple(proto.kernel_spatial_dimensions))
  out_spec = ((proto.output_batch_dimension, proto.output_feature_dimension) +
              tuple(proto.output_spatial_dimensions))
  return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def dot_dimension_numbers_from_proto(message) -> jax.lax.DotDimensionNumbers:
  proto = xla_data_pb2.DotDimensionNumbers().FromString(message)
  contracting_dimensions = (tuple(proto.lhs_contracting_dimensions),
                            tuple(proto.rhs_contracting_dimensions))
  batch_dimensions = (tuple(proto.lhs_batch_dimensions),
                      tuple(proto.rhs_batch_dimensions))
  return (contracting_dimensions, batch_dimensions)


def gather_dimension_numbers_from_proto(
    message) -> jax.lax.GatherDimensionNumbers:
  proto = xla_data_pb2.GatherDimensionNumbers().FromString(message)
  return jax.lax.GatherDimensionNumbers(
      tuple(proto.offset_dims), tuple(proto.collapsed_slice_dims),
      tuple(proto.start_index_map))


def scatter_dimension_numbers_from_proto(
    message) -> jax.lax.ScatterDimensionNumbers:
  proto = xla_data_pb2.ScatterDimensionNumbers().FromString(message)
  return jax.lax.ScatterDimensionNumbers(
      tuple(proto.update_window_dims), tuple(proto.inserted_window_dims),
      tuple(proto.scatter_dims_to_operand_dims))


def precision_config_from_proto(
    message) -> Optional[Union[jax.lax.Precision, Sequence[jax.lax.Precision]]]:
  """Converts a PrecisionConfig proto to jax.lax.Precision."""
  proto = xla_data_pb2.PrecisionConfig().FromString(message)
  precision_config = [
      jax.lax.Precision(v) for v in proto.operand_precision
  ]
  if not precision_config:
    precision_config = None
  elif len(precision_config) == 1:
    precision_config = precision_config[0]
  else:
    precision_config = tuple(precision_config)

  return precision_config


def get_random_algorithm_from_tf(
    algorithm: int) -> xla_data_pb2.RandomAlgorithm:
  """Map tf random algorithm enum to XLA random algorithm enum."""
  algorithm_map = {
      tf.random.Algorithm.PHILOX.value:
          xla_data_pb2.RandomAlgorithm.RNG_PHILOX,
      tf.random.Algorithm.THREEFRY.value:
          xla_data_pb2.RandomAlgorithm.RNG_THREE_FRY,
      tf.random.Algorithm.AUTO_SELECT.value:
          xla_data_pb2.RandomAlgorithm.RNG_DEFAULT,
  }
  return algorithm_map[algorithm]
