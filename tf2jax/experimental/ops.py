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

import functools
from typing import List, Tuple

from absl import logging

import jax
import jax.extend as jex
from jax.interpreters import mlir
import jax.numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func
from jaxlib.mlir.dialects import stablehlo as hlo

from tf2jax._src import config
from tf2jax._src import ops
from tf2jax._src import utils
from tf2jax.experimental import mhlo


# See canonicalize_platform for reference
# https://github.com/google/jax/blob/main/jax/_src/xla_bridge.py#L344
def _platform_to_alias(platform: str) -> str:
  aliases = {
      "cuda": "gpu",
      "rocm": "gpu",
  }
  return aliases.get(platform, platform)


# Adapted from
# https://github.com/google/jax/commit/ec8b855fa16962b1394716622c8cbc006ce76b1c
@functools.lru_cache(None)
def _refine_with_static_input_shapes(
    module_text: str, operands: Tuple[jax.core.ShapedArray, ...]
) -> tuple[str, list[jax.core.ShapedArray]]:
  """Refine the polymorphic shapes inside a module."""
  # Wrap original main within another function with static input shapes.
  context = mlir.make_ir_context()
  with context, ir.Location.unknown(context):
    module = ir.Module.parse(module_text)
    symbol_table = ir.SymbolTable(module.operation)
    orig_main = symbol_table["main"]
    orig_main.attributes["sym_visibility"] = ir.StringAttr.get("private")
    symbol_table.set_symbol_name(orig_main, "_orig_main")
    orig_main_name = ir.StringAttr(symbol_table.insert(orig_main)).value

    # This help refine polymorphic shapes.
    new_main_input_types = [mlir.aval_to_ir_type(x) for x in operands]
    # Retain the original element type. This is necessary because
    # jax.custom_gradient will replace integer types with the corresponding
    # tangent types, i.e. float0.
    for idx, (ox, nx) in enumerate(
        zip(orig_main.type.inputs, new_main_input_types, strict=True)
    ):
      assert isinstance(nx, ir.RankedTensorType), nx
      if ox.element_type != nx.element_type:
        new_main_input_types[idx] = ir.RankedTensorType.get(
            nx.shape, ox.element_type
        )
    # Final input specs to be returned.
    input_specs = [
        jax.core.ShapedArray(x.shape, mhlo.ir_type_to_dtype(x.element_type))
        for x in new_main_input_types
    ]

    orig_output_types = orig_main.type.results
    new_main_ftype = ir.FunctionType.get(
        new_main_input_types, orig_output_types
    )
    new_main_op = func.FuncOp(
        "main",
        new_main_ftype,
        ip=ir.InsertionPoint.at_block_begin(module.body),
    )

    try:
      new_main_op.attributes["arg_attrs"] = ir.ArrayAttr(orig_main.arg_attrs)
      assert new_main_op.arg_attrs == orig_main.arg_attrs
    except KeyError:
      pass
    try:
      new_main_op.attributes["res_attrs"] = ir.ArrayAttr(orig_main.result_attrs)
      assert new_main_op.result_attrs == orig_main.result_attrs
    except KeyError:
      pass
    new_main_op.attributes["sym_visibility"] = ir.StringAttr.get("public")
    symbol_table.insert(new_main_op)
    entry_block = new_main_op.add_entry_block()

    with ir.InsertionPoint(entry_block):
      orig_main_args: List[ir.Value] = []
      for new_arg, orig_arg_type in utils.safe_zip(
          new_main_op.arguments, orig_main.type.inputs
      ):
        # TODO(shaobohou) Why is the ConvertOp needed?
        if orig_arg_type != new_arg:
          orig_main_args.append(hlo.ConvertOp(orig_arg_type, new_arg).result)
        else:
          orig_main_args.append(new_arg)
      call = func.CallOp(
          orig_output_types,
          ir.FlatSymbolRefAttr.get(orig_main_name),
          orig_main_args,
      )
      func.ReturnOp(call.results)
  symbol_table.set_symbol_name(new_main_op, "main")

  # Refinement passes.
  input_dims = jax.tree_util.tree_flatten([x.shape for x in operands])[0]
  module = mhlo.refine_polymorphic_shapes(
      module,
      validate_static_shapes=all([isinstance(x, int) for x in input_dims]),
  )
  return mlir.module_to_string(module), input_specs


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
          "function_list",
          "module",
          "version",
          "platforms",
          "has_token_input_output",
          "disabled_checks",
          "use_shardy_partitioner",
      },
  )

  # TODO(b/329832868) Fix this properly.
  assume_grad_fn = proto.name.startswith("jax2tf_vjp")

  version = proto.attr["version"].i
  if version < 2:
    raise ValueError(
        f"Only version 2 (StableHLO) or above is supported, found {version}.")

  dim_args_spec = tuple(proto.attr["dim_args_spec"].list.s)
  if dim_args_spec:
    raise ValueError(
        "Dynamic shapes is not yet supported, found "
        f"dim_args_spec={dim_args_spec}.")

  function_list = tuple(proto.attr["function_list"].list.func)
  if function_list:
    raise ValueError(
        "function_list is not yet supported for custom calls, found "
        f"{function_list=}"
    )

  has_token_input_output = proto.attr["has_token_input_output"].b
  if has_token_input_output:
    raise ValueError(
        "has_token_input_output is not yet supported for custom calls, found "
        f"{has_token_input_output=}"
    )

  disabled_checks = tuple(proto.attr["disabled_checks"].list.s)
  if disabled_checks:
    logging.warning(
        "XlaCallModule was serialized with the following disabled checks: %s",
        disabled_checks,
    )

  target_platforms = tuple(
      _platform_to_alias(v.decode("utf-8").lower())
      for v in proto.attr["platforms"].list.s
  )

  def check_platforms():
    jax_backend = (
        jax.config.jax_default_device.platform.lower()
        if jax.config.jax_default_device
        else jax.default_backend().lower()
    )

    if target_platforms and jax_backend not in target_platforms:
      platform_message = (
          f"Unsupported backend: `{jax_backend}` not in `{target_platforms}`."
      )
      if config.get_config("xlacallmodule_strict_checks"):
        error_message = (
            platform_message
            + "\n"
            + "This error can be disabled using either"
            " tf2jax.update_config('xlacallmodule_strict_checks', False) or"
            " the context manager"
            " tf2jax.override_config('xlacallmodule_strict_checks', False)."
            " You may observe poorer performance or outright failure as a"
            " result."
        )
        raise ValueError(error_message)
      else:
        warn_message = (
            platform_message
            + "\n"
            + "Proceeding anyway as"
            " config.get_config('xlacallmodule_strict_checks') is False. You"
            " may observe poorer performance or outright failure."
        )
        logging.warning(warn_message)
      return None
    else:
      return target_platforms.index(jax_backend)

  if version >= 4:
    mhlo_text = jex.mlir.deserialize_portable_artifact(
        proto.attr["module"].s
    )
  else:
    with mlir.make_ir_context():
      module = ir.Module.parse(proto.attr["module"].s)
    mhlo_text = mlir.module_to_string(module)

  def _func(*operands: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    platform_index = check_platforms()
    require_platform_index = (
        platform_index is not None and len(target_platforms) > 1
    )
    if require_platform_index:
      operands = (jnp.array(platform_index),) + operands

    refined_mhlo_text, input_specs = _refine_with_static_input_shapes(
        mhlo_text,
        tuple(jax.core.ShapedArray(x.shape, x.dtype) for x in operands),
    )
    mhlo_module = mhlo.MhloModule(
        module=refined_mhlo_text,
        fun_name=proto.name,
        assume_grad_fn=assume_grad_fn,
        require_platform_index=require_platform_index,
    )
    if assume_grad_fn:
      # The change in _refine_with_static_input_shapes is not enough as
      # depending on whether we are computing gradient via Jax or TF, integer
      # types may or may not be replaced with float0.
      operands = [
          jnp.zeros(x.shape, y.dtype) if x.dtype == jax.dtypes.float0 else x
          for x, y in zip(operands, input_specs, strict=True)
      ]
    return mhlo.mhlo_apply(*operands, module=mhlo_module)

  return _func
