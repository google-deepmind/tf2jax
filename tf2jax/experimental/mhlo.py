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


def mhlo_apply_abstract_eval(*args, module: MhloModule):
  del args
  _, program = _get_program(module.module)
  result_spec = program.program_shape().result_shape()
  assert result_spec.is_tuple()
  return tuple([
      core.ShapedArray(spec.dimensions(), spec.element_type())
      for spec in result_spec.tuple_shapes()
  ])

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
