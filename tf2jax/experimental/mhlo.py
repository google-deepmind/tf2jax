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
import threading
from typing import Dict, Tuple

from jax import core
from jax.interpreters import mlir
from jax.interpreters import xla
from jax.lib import xla_client as xc
import jax.numpy as jnp

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import mhlo

from tf2jax.experimental import util


_program_cache: Dict[str, Tuple[str, xc.XlaComputation]] = {}
_program_lock = threading.Lock()

mhlo_apply_p = core.Primitive("mhlo_apply")
mhlo_apply_p.multiple_results = True


@dataclasses.dataclass(frozen=True)
class MhloModule:
  module: str  # string representation of the MLIR module.
  fun_name: str

  def __str__(self):
    return f"MhloModule(fun_name={self.fun_name}, ...)"


def _get_program(mhlo_text: str) -> Tuple[str, xc.XlaComputation]:
  """Thread-safe cache of MHLO string to XlaComputation."""
  try:
    return _program_cache[mhlo_text]
  except KeyError:
    pass

  with _program_lock:
    program = xc._xla.mlir.mlir_module_to_xla_computation(  # pylint: disable=protected-access
        mlir_module=mhlo_text,
        use_tuple_args=False,
        return_tuple=True)
    named_program = (f"_tf2jax_user_mhlo_fn_{len(_program_cache):06}", program)
    _program_cache[mhlo_text] = named_program
  return named_program


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
def _ir_type_to_dtype(ir_type: ir.Type) -> jnp.dtype:
  """Converts MLIR type to JAX dtype."""
  ir_to_jax = {
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
    has_polymorphic = False
    for val in in_avals:
      for dim in val.shape:
        if not isinstance(dim, int):
          has_polymorphic = True
          if any(x.startswith(_UKNOWN_DIM_PREFIX) for x in dim.get_vars()):
            raise ValueError(
                "Polymorphic variable name that start with"
                f" `{_UKNOWN_DIM_PREFIX}` are reserved for use by tf2jax"
                f" internal for outputs: `{val.shape}`"
            )

    # Map each `dynamic`` dimension to a unique dimension variable because we
    # do not have the information from the avals of the original JAX function.
    # In practice, the output shapes may actually be much more constrained, but
    # the information is not available here.
    dynamic_count = 0
    output_specs = []
    for res in symtab["main"].type.results:
      if any(dim == res.get_dynamic_size() for dim in res.shape):
        out_shape = ", ".join(
            f"{_UKNOWN_DIM_PREFIX}_{(dynamic_count := dynamic_count + 1)}"
            if dim == res.get_dynamic_size()
            else str(dim)
            for dim in res.shape
        )

        assert has_polymorphic, has_polymorphic
        from jax.experimental.jax2tf import shape_poly  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
        out_shape = shape_poly._parse_spec(out_shape, res.shape)  # pylint: disable=protected-access
      else:
        out_shape = res.shape
      output_specs.append(
          core.ShapedArray(out_shape, _ir_type_to_dtype(res.element_type))
      )
    return tuple(output_specs)


mhlo_apply_p.def_abstract_eval(mhlo_apply_abstract_eval)


# Taken from
# github.com/google/jax/blob/main/jax/experimental/jax2tf/jax_export.py#L859
def refine_polymorphic_shapes(module: ir.Module) -> ir.Module:
  """Refine the polymorphic shapes inside a module.

  Given a module with static input shapes, but using dynamic shapes due to
  shape polymorphism, run shape refinement to resolve all the dynamic shapes.

  Args:
    module: A module with static input shapes but dynamic shapes inside.

  Returns:
    The refined module.
  """
  if xc.mlir_api_version < 50:
    raise NotImplementedError("refine_polymorphic_shapes needs jaxlib 0.4.12")

  refined_module_str = xc._xla.mlir.refine_polymorphic_shapes(  # pylint: disable=protected-access
      mlir.module_to_bytecode(module)
  )
  context = mlir.make_ir_context()
  with context:
    return ir.Module.parse(refined_module_str)


def mhlo_apply_lowering(
    ctx: mlir.LoweringRuleContext, *args, module: MhloModule
):
  """Lowering rule."""
  program_name, program = _get_program(module.module)
  assert program.program_shape().result_shape().is_tuple()

  module_ctx = ctx.module_context
  if xc.mlir_api_version < 41:
    mhlo_module = mlir.xla_computation_to_mhlo_module(program)  # type: ignore
    callee_name = mlir.merge_mhlo_modules(
        dst_module=module_ctx.module,
        sym_name=program_name,
        src_module=mhlo_module)  # type: ignore
  else:
    mhlo_module = mlir.xla_computation_to_mlir_module(program)
    callee_name = mlir.merge_mlir_modules(
        dst_module=module_ctx.module,
        sym_name=program_name,
        src_module=mhlo_module)

  output_types = tuple(map(mlir.aval_to_ir_types, ctx.avals_out))
  flat_output_types = util.flatten(output_types)
  output_type = ir.TupleType.get_tuple(flat_output_types)

  call = mlir.func_dialect.CallOp([output_type],
                                  ir.FlatSymbolRefAttr.get(callee_name),
                                  mlir.flatten_lowering_ir_args(args)).result
  flat_results = [mhlo.GetTupleElementOp(call, mlir.i32_attr(i)).result
                  for i in range(len(flat_output_types))]

  return tuple(util.unflatten(flat_results, list(map(len, output_types))))

mlir.register_lowering(mhlo_apply_p, mhlo_apply_lowering)
