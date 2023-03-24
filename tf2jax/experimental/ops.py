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

import jax
from jax.interpreters import mlir
from jax.lib import xla_client as xc
import jax.numpy as jnp
from jaxlib.mlir import ir

from tf2jax._src import ops
from tf2jax.experimental import mhlo


# See canonicalize_platform for reference
# https://github.com/google/jax/blob/main/jax/_src/xla_bridge.py#L344
def _platform_to_alias(platform: str) -> str:
  aliases = {
      "cuda": "gpu",
      "rocm": "gpu",
  }
  return aliases.get(platform, platform)


@ops.register_operation("XlaCallModule")
def _xla_call_module(proto):
  """Parse a XlaCallModule op."""
  ops._check_attrs(  # pylint: disable=protected-access
      proto,
      {
          "Tin",
          "Sout",
          "Tout",
          "dim_args_spec",
          "module",
          "version",
          "platforms",
      },
  )

  version = proto.attr["version"].i
  if version < 2:
    raise ValueError(
        f"Only version 2 (StableHLO) or above is supported, found {version}.")

  dim_args_spec = tuple(proto.attr["dim_args_spec"].list.s)
  if dim_args_spec:
    raise ValueError("Dynamic shapes is not yet supported, found "
                     f"dim_args_spec={dim_args_spec}.")

  jax_device = jax.default_backend().lower()
  platforms = tuple(
      _platform_to_alias(v.decode("utf-8").lower())
      for v in proto.attr["platforms"].list.s
  )
  if platforms and jax_device not in platforms:
    raise ValueError(
        f"Unsupported backend: `{jax_device}` not in `{platforms}`"
    )

  if version >= 4:
    mhlo_text = xc._xla.mlir.deserialize_portable_artifact(  # pylint: disable=protected-access
        proto.attr["module"].s
    )
  else:
    with mlir.make_ir_context():
      module = ir.Module.parse(proto.attr["module"].s)
    mhlo_text = mlir.module_to_string(module)

  def _func(*operands: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    return mhlo.mhlo_apply(*operands, mhlo_text=mhlo_text)

  return _func
