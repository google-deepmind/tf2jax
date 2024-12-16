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
"""MHLO JAX primitive"."""

import dataclasses
from typing import Tuple

import jax
from jax import core
from jax import export
import jax.extend as jex
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_client as xc
from jax.lib import xla_extension
import jax.numpy as jnp

from jaxlib.mlir import ir


safe_zip = jax.util.safe_zip

mhlo_apply_p = jex.core.Primitive("mhlo_apply")
mhlo_apply_p.multiple_results = True


@dataclasses.dataclass(frozen=True, kw_only=True)
class MhloModule:
  module: str  # string representation of the MLIR module.
  fun_name: str
  assume_grad_fn: bool = False
  require_platform_index: bool = False

  def __str__(self):
    return f"MhloModule(fun_name={self.fun_name}, ...)"


def mhlo_apply(*args, module: MhloModule):
  ret = mhlo_apply_p.bind(*args, module=module)
  # TODO(shaobohou) Unpack singleton result?
  if len(ret) == 1:
    return ret[0]
  else:
    return tuple(ret)


def mhlo_apply_impl(*args, module: MhloModule):
  return xla.apply_primitive(mhlo_apply_p, *args, module=module)

mhlo_apply_p.def_impl(mhlo_apply_impl)


# See https://github.com/google/jax/blob/main/jax/_src/interpreters/mlir.py#L115
# for reference
def ir_type_to_dtype(ir_type: ir.Type) -> jnp.dtype:
  """Converts MLIR type to JAX dtype."""
  ir_to_jax = {
      ir.IntegerType.get_signless(1): jnp.bool_,
      ir.IntegerType.get_signless(8): jnp.int8,
      ir.IntegerType.get_signless(16): jnp.int16,
      ir.IntegerType.get_signless(32): jnp.int32,
      ir.IntegerType.get_signless(64): jnp.int64,
      ir.IntegerType.get_unsigned(8): jnp.uint8,
      ir.IntegerType.get_unsigned(16): jnp.uint16,
      ir.IntegerType.get_unsigned(32): jnp.uint32,
      ir.IntegerType.get_unsigned(64): jnp.uint64,
      ir.F16Type.get(): jnp.float16,
      ir.F32Type.get(): jnp.float32,
      ir.F64Type.get(): jnp.float64,
      ir.BF16Type.get(): jnp.bfloat16,
      ir.ComplexType.get(ir.F32Type.get()): jnp.complex64,
      ir.ComplexType.get(ir.F64Type.get()): jnp.complex128,
      ir.Float8E4M3B11FNUZType.get(): jnp.float8_e4m3b11fnuz,
      ir.Float8E4M3FNType.get(): jnp.float8_e4m3fn,
      ir.Float8E5M2Type.get(): jnp.float8_e5m2,
  }
  return ir_to_jax[ir_type]


_UKNOWN_DIM_PREFIX = "tf2jax_unknown_dim"


def mhlo_apply_abstract_eval(
    *in_avals: core.ShapedArray, module: MhloModule
) -> Tuple[core.ShapedArray, ...]:
  """Abstract evaluation rule."""
  with mlir.make_ir_context():
    mhlo_module = ir.Module.parse(module.module)
    symtab = ir.SymbolTable(mhlo_module.operation)

    # Check we are not reusing existing dimension vars.
    dynamic_count = 0
    has_polymorphic = False
    symbolic_scope = None
    for val in in_avals:
      for dim in val.shape:
        if not isinstance(dim, int):
          has_polymorphic = True
          try:
            symbolic_scope = symbolic_scope or dim.scope
          except AttributeError:
            pass
          if any(x.startswith(_UKNOWN_DIM_PREFIX) for x in dim.get_vars()):
            for dim in dim.get_vars():
              if dim.startswith(_UKNOWN_DIM_PREFIX):
                dynamic_count = max(
                    dynamic_count,
                    (int(dim.removeprefix(_UKNOWN_DIM_PREFIX + "_"))),
                )

    # Map each `dynamic`` dimension to a unique dimension variable because we
    # do not have the information from the avals of the original JAX function.
    # In practice, the output shapes may actually be much more constrained, but
    # the information is not available here.
    output_specs = []
    for idx, res in enumerate(symtab["main"].type.results):
      if any(dim == res.get_dynamic_size() for dim in res.shape):
        out_shape = ", ".join(
            f"{_UKNOWN_DIM_PREFIX}_{(dynamic_count := dynamic_count + 1)}"
            if dim == res.get_dynamic_size()
            else str(dim)
            for dim in res.shape
        )

        assert has_polymorphic, has_polymorphic
        if module.assume_grad_fn:
          # TODO(b/329832868) Fix this properly.
          offset = 1 if module.require_platform_index else 0
          out_shape = in_avals[idx + offset].shape
        else:
          out_shape = export.symbolic_shape(
              out_shape, like=res.shape, scope=symbolic_scope
          )
      else:
        out_shape = res.shape
      output_specs.append(
          core.ShapedArray(out_shape, ir_type_to_dtype(res.element_type))
      )
    return tuple(output_specs)


mhlo_apply_p.def_abstract_eval(mhlo_apply_abstract_eval)


# Taken from
# github.com/google/jax/blob/main/jax/experimental/jax2tf/jax_export.py#L859
def refine_polymorphic_shapes(
    module: ir.Module, validate_static_shapes: bool
) -> ir.Module:
  """Refine the polymorphic shapes inside a module.

  Given a module with static input shapes, but using dynamic shapes due to
  shape polymorphism, run shape refinement to resolve all the dynamic shapes.

  Args:
    module: A module with static input shapes but dynamic shapes inside.
    validate_static_shapes: Whether to check all shapes are static after
      refinement.

  Returns:
    The refined module.
  """
  if xc.mlir_api_version >= 53:
    refined_module_str = xla_extension.mlir.refine_polymorphic_shapes(
        mlir.module_to_bytecode(module),
        enable_shape_assertions=validate_static_shapes,
        validate_static_shapes=validate_static_shapes,
    )
  elif xc.mlir_api_version >= 50:
    refined_module_str = xla_extension.mlir.refine_polymorphic_shapes(
        mlir.module_to_bytecode(module)
    )
  else:
    raise NotImplementedError("refine_polymorphic_shapes needs jaxlib 0.4.12")

  context = mlir.make_ir_context()
  with context:
    return ir.Module.parse(refined_module_str)


def mhlo_apply_lowering(
    ctx: mlir.LoweringRuleContext, *args, module: MhloModule
):
  """Lowering rule."""
  program_name = f"_tf2jax_mhlo_apply_fn_{module.fun_name}"
  mhlo_module = ir.Module.parse(module.module)

  if xc.mlir_api_version < 41:
    callee_name = mlir.merge_mhlo_modules(
        dst_module=ctx.module_context.module,
        sym_name=program_name,
        src_module=mhlo_module)  # type: ignore
  else:
    callee_name = mlir.merge_mlir_modules(
        dst_module=ctx.module_context.module,
        sym_name=program_name,
        src_module=mhlo_module)

  symtab = ir.SymbolTable(ctx.module_context.module.operation)
  result_types = symtab[program_name].type.results

  # Paranoid checks.
  assert len(mlir.flatten_lowering_ir_args(args)) == len(args), (
      len(mlir.flatten_lowering_ir_args(args)),
      len(args),
  )

  call = mlir.func_dialect.CallOp(
      result_types,
      ir.FlatSymbolRefAttr.get(callee_name),
      args,
  )
  return tuple(x for x in call.results)


mlir.register_lowering(mhlo_apply_p, mhlo_apply_lowering)
