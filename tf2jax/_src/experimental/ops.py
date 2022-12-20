# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
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
"""Experimental ops"."""

from typing import Tuple

from jax._src.lib.mlir import ir
from jax.interpreters import mlir
import jax.numpy as jnp

from tf2jax._src import ops
from tf2jax._src.experimental import mhlo


@ops.register_operation("XlaCallModule")
def _xla_call_module(proto):
  """Parse a XlaCallModule op."""
  ops._check_attrs(  # pylint: disable=protected-access
      proto,
      {"Tin", "Sout", "Tout", "dim_args_spec", "module", "version"})

  version = proto.attr["version"].i
  if version != 2:
    raise ValueError(
        f"Only version 2 (StableHLO) is supported, found {version}.")

  dim_args_spec = tuple(proto.attr["dim_args_spec"].list.s)
  if dim_args_spec:
    raise ValueError("Dynamic shapes is not yet supported, found "
                     f"dim_args_spec={dim_args_spec}.")

  with mlir.make_ir_context():
    module = ir.Module.parse(proto.attr["module"].s)
  mhlo_text = mlir.module_to_string(module)

  def _func(*operands: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    return mhlo.mhlo_apply(*operands, mhlo_text=mhlo_text)

  return _func
