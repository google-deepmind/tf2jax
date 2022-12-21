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

import functools
import threading
from typing import Any, Dict, Tuple

import jax
from jax import core
from jax._src.lib import xla_client as xc
import jax._src.util as util  # pylint: disable=consider-using-from-import
from jax.interpreters import mlir

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import mhlo

import numpy as np

_program_cache: Dict[str, Tuple[str, xc._xla.XlaComputation]] = {}
_program_lock = threading.Lock()

mhlo_apply_p = core.Primitive("mhlo_apply")
mhlo_apply_p.multiple_results = True


def _get_program(mhlo_text: str) -> Tuple[str, xc._xla.XlaComputation]:
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


@functools.lru_cache(None)
def _get_compiled_program(mhlo_text: str, backend: Any) -> Any:
  executable = backend.compile(mhlo_text)
  return executable


def mhlo_apply(*args, mhlo_text: str):
  ret = mhlo_apply_p.bind(*args, mhlo_text=mhlo_text)
  # TODO(shaobohou) Unpack singleton result?
  if len(ret) == 1:
    return ret[0]
  else:
    return tuple(ret)


def mhlo_apply_impl(*args, mhlo_text: str):
  backend = jax.lib.xla_bridge.get_backend()
  executable = _get_compiled_program(mhlo_text, backend)
  return jax.lib.xla_client.execute_with_python_values(
      executable=executable,
      arguments=[np.asarray(x) for x in args],
      backend=backend)

mhlo_apply_p.def_impl(mhlo_apply_impl)


def mhlo_apply_abstract_eval(*args, mhlo_text: str):
  del args
  _, program = _get_program(mhlo_text)
  result_spec = program.program_shape().result_shape()
  assert result_spec.is_tuple()
  return tuple([
      jax.ShapedArray(spec.dimensions(), spec.element_type())
      for spec in result_spec.tuple_shapes()
  ])

mhlo_apply_p.def_abstract_eval(mhlo_apply_abstract_eval)


def mhlo_apply_lowering(ctx: mlir.LoweringRuleContext, *args, mhlo_text: str):
  """Lowering rule."""
  program_name, program = _get_program(mhlo_text)
  assert program.program_shape().result_shape().is_tuple()

  module_ctx = ctx.module_context
  if xc.mlir_api_version < 41:
    mhlo_module = mlir.xla_computation_to_mhlo_module(program)
    callee_name = mlir.merge_mhlo_modules(
        dst_module=module_ctx.module,
        sym_name=program_name,
        src_module=mhlo_module)
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
