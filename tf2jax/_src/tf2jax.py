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

import collections
import functools
import inspect
import itertools
import json
from typing import Any, Callable, Iterable, Iterator, Optional, Mapping, NamedTuple, Sequence, Tuple, Union

from absl import logging

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from tf2jax._src import config
from tf2jax._src import ops
from tf2jax._src import utils
import tree

# Import usage logging here.

from tensorflow.python.framework import op_def_registry  # pylint: disable=no-name-in-module
from tensorflow.python.framework import ops as tf_ops  # pylint: disable=no-name-in-module

try:
  import tf2jax.experimental.ops  # pylint: disable=g-import-not-at-top,unused-import
except ImportError:
  logging.info(
      "Proceeding without support for experimental ops, e.g. XlaCallModule.")


ArrayLike = ops.ArrayLike
SpecTree = Union[tf.TensorSpec, Iterable["SpecTree"], Mapping[str, "SpecTree"]]

safe_zip = jax.util.safe_zip


class AnnotatedFunction:
  """Callable function with additional metadata.

  The metadata includes: structured inputs and outputs (composed of
  tf.TensorSpec) and the original function signature.
  """

  def __init__(
      self,
      # TODO(b/235830619) Add more specific type annotations.
      fn: Callable[..., Any],
      structured_inputs: SpecTree,
      structured_outputs: SpecTree,
      signature: inspect.Signature,
  ):
    self.fn = fn
    self.structured_inputs = structured_inputs
    self.structured_outputs = structured_outputs

    # Attributes that might be expected by downstream usage.
    self.__doc__ = fn.__doc__
    self.__name__ = fn.__name__
    self.__signature__ = signature

  # This should not be called directly.
  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)

  @property
  def signature(self) -> inspect.Signature:
    return self.__signature__


_EMPTY_RETURN_OP_NAME = ops._EMPTY_RETURN_OP_NAME  # pylint: disable=protected-access
_EMPTY_RETURN_VALUE = ops._EMPTY_RETURN_VALUE  # pylint: disable=protected-access


_UNUSED_INPUT = object()
_XLA_CALL_MODULE = "XlaCallModule"


def _fix_jax_poly_shape(shape: Tuple[Any, ...]) -> Tuple[Any, ...]:
  good_shape = []
  for dim in shape:
    try:
      # This catches _DimPolynomial from jax2tf.
      tf.compat.v1.Dimension(dim)
      good_shape.append(dim)
    except TypeError:
      good_shape.append(None)
  return tuple(good_shape)


class _TensorEdge(NamedTuple):
  """Represents an input/output Tensor."""

  op_name: str
  idx: int = 0
  is_control: bool = False

  @classmethod
  def from_string(
      cls,
      op_str: str,
      node_map: Optional[Mapping[str, Any]] = None,
  ):
    """Parse a NodeDef input string."""
    node_map = node_map or {}

    is_control = op_str.startswith("^")
    if is_control:
      op_str = op_str[1:]

    args = op_str.split(":")

    if len(args) == 1:
      op_name, = args
      output_name = ""
      idx = 0
    elif len(args) == 2:
      op_name, idx = args
      output_name = ""
      idx = int(idx)
    elif len(args) == 3:
      op_name, output_name, idx = args
      op_def = op_def_registry.get(node_map[op_name].op)
      assert op_def.output_arg

      if len(op_def.output_arg) == 1:
        idx = int(idx)
      else:
        found_sequence_args = any([
            arg.number_attr or arg.type_list_attr for arg in op_def.output_arg
        ])
        if idx != "0" or found_sequence_args:
          raise ValueError(
              "Only zero index and single tensors are supported for multiple "
              "output args.\n"
              f"op_str={op_str}\n"
              f"op_def={op_def}")
        idx = [arg.name for arg in op_def.output_arg].index(output_name)
    else:
      raise ValueError(f"Invalid input spec string: `{op_str}`")

    return cls(op_name=op_name, idx=idx, is_control=is_control)


_TF_ASSIGN_OPS = (
    "AssignAddVariableOp", "AssignSubVariableOp", "AssignVariableOp"
)

_TF_RANDOM_OPS = ("RandomStandardNormal", "RandomUniform", "RandomUniformInt")


class _LibraryFunction(NamedTuple):
  """A library function."""
  fn: Callable[..., Any]
  require_rng: bool
  # Optional fields (mainly) used by gradient functions.
  params: Optional[Mapping[str, ArrayLike]] = None
  input_specs: Optional[Tuple[tf.TensorSpec, ...]] = None
  output_specs: Optional[Tuple[tf.TensorSpec, ...]] = None
  # If fn is a gradient function, this is the output specs for the original fn.
  orig_fn_output_specs: Optional[Tuple[tf.TensorSpec, ...]] = None
  # Whether an output is unmodified input to the function.
  output_is_input: Optional[Tuple[bool]] = None
  # Inputs corresponding to VarHandleOp
  variable_input_specs: Optional[Tuple[tf.TensorSpec, ...]] = None

  def __call__(self, *args, **kwargs):
    if self.params:
      return self.fn(self.params, *args, **kwargs)
    else:
      # Ignore parameters in inputs and outputs.
      return self.fn({}, *args, **kwargs)[0]


def _unbox_named_args(
    named_args: Sequence[Tuple[str, jnp.ndarray]],
    inputs: Tuple[_TensorEdge, ...],
) -> Tuple[jnp.ndarray, ...]:
  """Extract inputs from named arguments."""
  if len(named_args) != len(inputs):
    raise ValueError(
        f"Expected {len(inputs)} arguments, received {len(named_args)}")
  for (arg_name, _), inp in safe_zip(named_args, inputs):
    if arg_name != inp.op_name:
      raise ValueError(f"Expected inputs {inp.op_name}, found {arg_name}")

  unboxed_args = [
      arg[inp.idx] if isinstance(arg, (list, tuple)) else arg
      for arg, inp in safe_zip([x for _, x in named_args], inputs)
  ]
  return tuple(unboxed_args)


class _OpNode:
  """Represents an Op."""

  def __init__(self, proto, library: Mapping[str, _LibraryFunction],
               node_map: Mapping[str, Any]):
    self.jax_func = ops.get_parser(proto.op)(proto)
    self.op = proto.op
    self.name = proto.name

    inputs = [_TensorEdge.from_string(inp, node_map) for inp in proto.input]

    self.inner_fns = dict()
    if isinstance(self.jax_func, ops._HigherOrderFunction):
      self.inner_fns = self.jax_func.get_inner_functions(library)
      for input_name in self.jax_func.get_additional_inputs(**self.inner_fns):
        inputs.append(_TensorEdge.from_string(input_name, node_map))

    self.control_inputs = tuple([inp for inp in inputs if inp.is_control])
    self.inputs = tuple([inp for inp in inputs if not inp.is_control])

  @property
  def all_inputs(self) -> Tuple[_TensorEdge, ...]:
    return self.inputs + self.control_inputs

  @property
  def require_rng(self) -> bool:
    inner_require_rngs = any([fn.require_rng for fn in self.inner_fns.values()])
    return self.op in _TF_RANDOM_OPS or inner_require_rngs

  def __call__(
      self,
      named_args: Sequence[Tuple[str, jnp.ndarray]],
      *,
      rng: jnp.ndarray,
  ) -> Tuple[Tuple[jnp.ndarray, ...], Mapping[str, jnp.ndarray]]:
    unboxed_args = _unbox_named_args(named_args, self.inputs)
    extras_dict = dict(rng=rng) if self.require_rng else dict()
    extras_dict.update(self.inner_fns)
    outputs = self.jax_func(*unboxed_args, **extras_dict)

    # Return updated variables.
    if self.op in _TF_ASSIGN_OPS:
      # This assumes the assign op returns the updated value.
      updated_params = {named_args[0][0]: outputs}
    else:
      updated_params = {}

    return outputs, updated_params

  def __repr__(self) -> str:
    message = f"_OpNode(name={repr(self.name)}, op={repr(self.op)}"
    if self.inputs:
      message += f", inputs={repr([x.op_name for x in self.inputs])}"
    if self.control_inputs:
      message += f", controls={repr([x.op_name for x in self.control_inputs])}"
    message += ")"
    return message


def _parse_input(op_str: str) -> str:
  # Parses an input name in tf.NodeDef. This extracts the node name, removes
  # the control character and the output tensor index e.g. ^Sin:0 -> Sin
  return op_str.split(":")[0].split("^")[-1]


def _toposort(
    nodes: Mapping[str, tf.compat.v1.NodeDef],
    end_node_names: Tuple[str, ...],
):
  """Topological sorting of nodes."""

  child_counts = {}
  stack = list(end_node_names)
  while stack:
    node_name = stack.pop()
    if node_name in child_counts:
      child_counts[node_name] += 1
    else:
      child_counts[node_name] = 1
      node = nodes[node_name]
      stack.extend([_parse_input(v) for v in node.input])
  for node_name in end_node_names:
    child_counts[node_name] -= 1

  sorted_nodes = []
  childless_nodes = [
      node_name for node_name in end_node_names if child_counts[node_name] == 0
  ]
  if not childless_nodes:
    raise ValueError("No childless nodes found.")

  while childless_nodes:
    node_name = childless_nodes.pop()
    node = nodes[node_name]
    sorted_nodes.append(node)
    for parent in [_parse_input(v) for v in node.input]:
      if child_counts[parent] == 1:
        childless_nodes.append(parent)
      else:
        child_counts[parent] -= 1

  return sorted_nodes[::-1]


class Variable(np.ndarray):
  """Array subclass with additional metadaa for representing variables."""

  def __new__(cls, arr: np.ndarray, trainable: bool, name: str):
    obj = np.asarray(arr).view(cls)
    obj.trainable = trainable
    obj.name = name
    return obj

  def assign(self, arr: np.ndarray) -> "Variable":
    return Variable(arr, trainable=self.trainable, name=self.name)

  def __array_finalize__(self, obj):
    if obj is None:
      return
    self.trainable = getattr(obj, "trainable", None)
    self.name = getattr(obj, "name", None)

  def __repr__(self) -> str:
    message = f"Variable(name={repr(self.name)}, "
    message += f"trainable={repr(self.trainable)}, "
    message += f"numpy={np.array_repr(self)}"
    message += ")"
    return message


def _make_parameterless(
    jax_func: AnnotatedFunction,
    jax_params: Mapping[str, Variable],
) -> AnnotatedFunction:
  """Return an AnnotatedFunction that neither expects nor returns parameters."""
  def assert_empty_params(params):
    if params:
      raise ValueError(
          f"Expected function to have no captured variables, found {params}")

  def parameterless_fn(*args, **kwargs):
    results, params = jax_func.fn({}, *args, **kwargs)
    assert_empty_params(params)
    return results

  assert_empty_params(jax_params)

  return AnnotatedFunction(
      fn=parameterless_fn,
      structured_inputs=jax_func.structured_inputs,
      structured_outputs=jax_func.structured_outputs,
      signature=jax_func.signature,
  )


def convert_functional(tf_func: Any, *args, **kwargs) -> AnnotatedFunction:
  return _make_parameterless(*convert(tf_func, *args, **kwargs))


def convert_functional_from_restored(tf_func: Any) -> AnnotatedFunction:
  return _make_parameterless(*convert_from_restored(tf_func))


def convert_from_restored(
    tf_func) -> Tuple[AnnotatedFunction, Mapping[str, Variable]]:
  """Converts a RestoredFunction (from a SaveModel) if it is unambiguous."""
  if tf_func.input_signature is None:
    concrete_functions = getattr(tf_func, "concrete_functions", ())
    if len(concrete_functions) != 1:
      err_message = (
          f"Found {len(concrete_functions)} concrete functions, use "
          "tf2jax.convert or tf2jax.convert_functional with *args and **kwargs "
          "to select one of the options above.")
      saved_model_err_message = ""
      try:
        # Deliberately trigger pretty error message from SavedModel.
        tf_func(None)
      except ValueError as e:
        saved_model_err_message = str(e)
      raise ValueError(saved_model_err_message + "\n\n" + err_message)

    args, kwargs = concrete_functions[0].structured_input_signature
  else:
    args = tf_func.input_signature
    kwargs = {}

  return convert(tf_func, *args, **kwargs)


def _is_tensorspec_like(v: Any) -> bool:
  return (isinstance(v, tf.TensorSpec) or (
      type(v).__module__.endswith("tensorflow.python.framework.tensor_spec") and
      type(v).__name__ == "VariableSpec"))


def _fix_tfhub_specs(
    structured_specs,
    flat_tensors,
    expected_count: Optional[int] = None,
):
  """Fix names used by TF-Hub TensorSpecs."""
  flat_specs = tree.flatten(structured_specs)
  tensor_count = 0
  for idx, val in enumerate(flat_specs):
    if _is_tensorspec_like(val):
      flat_specs[idx] = tf.TensorSpec(
          val.shape, val.dtype, name=flat_tensors[tensor_count].op.name)
      tensor_count += 1
  if expected_count is not None:
    assert tensor_count == expected_count

  structured_specs = tree.unflatten_as(structured_specs, flat_specs)
  return structured_specs


def _maybe_tracer_to_tf_spec(val: Any) -> Any:
  if isinstance(val, jax.core.Tracer):
    return tf.TensorSpec(shape=val.shape, dtype=val.dtype)
  else:
    return val


def convert(
    tf_func: Any,  # tensorflow.python.eager.function is not visible.
    *args,
    **kwargs,
) -> Tuple[AnnotatedFunction, Mapping[str, Variable]]:
  """Convert a tf.Function to a Jax function for some concrete inputs.

  The concrete inputs are used to instantiate a tf.ConcreteFunction, the
  graphdef of which is then parsed and used to generate the corresponding Jax
  code.

  Args:
    tf_func: a tf.Function
    *args: positional arguments.
    **kwargs: keyword arguments.

  Returns:
    A tuple: the first is a Jax functions that takes a flat parameter dict and a
    variable number of arrays as inputs, and returns a nested structure of
    outputs and an updated parameter dict; the second is the flat parameter dict
    correpondings to all TF variables used by the concrete function.
  """
# Log usage here.

  args, kwargs = tree.map_structure(_maybe_tracer_to_tf_spec, (args, kwargs))
  try:
    concrete_func = tf_func.get_concrete_function(*args, **kwargs)
  except AttributeError:
    logging.error("Expected `tf.Function`, received %s", tf_func)
    raise

  graph = concrete_func.graph
  graphdef = graph.as_graph_def()

  num_flat_args = len(concrete_func.inputs) - len(concrete_func.captured_inputs)
  captures = list(
      safe_zip(
          [inp.op.name for inp in concrete_func.inputs[num_flat_args:]],
          [inp for inp in concrete_func.captured_inputs],
      )
  )
  func_variables = {v.handle.ref(): v for v in concrete_func.variables}
  variable_map = {
      k: func_variables[v.ref()] for k, v in captures if v.dtype == tf.resource
  }
  constants = {k: v.numpy() for k, v in captures if v.dtype != tf.resource}

  captured_input_names = tuple([
      v.op.name for v in concrete_func.inputs[num_flat_args:]
  ])

  def maybe_tensor_to_spec(v):
    if v is None or isinstance(v, tf.TensorSpec):
      return v
    else:
      return tf.TensorSpec.from_tensor(v)

  num_inputs = len(concrete_func.inputs) - len(concrete_func.captured_inputs)
  structured_inputs = concrete_func.structured_input_signature
  structured_inputs = _fix_tfhub_specs(structured_inputs, concrete_func.inputs,
                                       num_inputs)
  flat_outputs = tree.flatten(concrete_func.outputs)
  structured_outputs = tree.map_structure(maybe_tensor_to_spec,
                                          concrete_func.structured_outputs)
  # We do not check the number of output tensors because they do not match for
  # custom gradients functions.
  structured_outputs = _fix_tfhub_specs(structured_outputs, flat_outputs)
  try:
    fullargspec = tf_func.function_spec.fullargspec
  except AttributeError:
    logging.warning("No fullargspec found on %s.", tf_func)
    fullargspec = None

  if fullargspec is not None:
    signature = utils.fullargspec_to_signature(fullargspec)
  else:
    exp_args, exp_kwargs = structured_inputs
    if exp_args:
      raise ValueError("If function_spec is None then only keyword arguments "
                       f"are expectd, found args={exp_args} in structure.")
    parameters = tuple([
        # TODO(b/266552275) Remove temporary fix for TF-Hub.
        inspect.Parameter(
            k.replace("$", "___"), kind=inspect.Parameter.KEYWORD_ONLY)
        for k in tree.flatten(exp_kwargs.keys())
    ])
    signature = inspect.Signature(parameters=parameters)

  # Extract custom_gradient functions from the registry.
  if config.get_config("convert_custom_gradient"):
    library = _convert_all_gradient_functions(graph, {})
  else:
    library = {}

  jax_func, jax_params = _convert(
      graphdef,
      signature=signature,
      structured_inputs=structured_inputs,
      structured_outputs=structured_outputs,
      captured_input_names=captured_input_names,
      variable_map=variable_map,
      constants=constants,
      library=library,
  )
  annotated_fn = AnnotatedFunction(
      fn=jax_func,
      structured_inputs=structured_inputs,
      structured_outputs=structured_outputs,
      signature=signature,
  )
  return annotated_fn, jax_params


class _EvaluationCache:
  """Cache holding intermediate outputs."""

  def __init__(
      self,
      nodes: Sequence[Any],
      inputs: Sequence[Tuple[str, Any]],
      outputs: Sequence[_TensorEdge],
  ):
    self.outputs = dict(inputs)
    # Reference counting.
    self._counts = collections.Counter()
    for node in nodes:
      for inp in node.inputs + node.control_inputs:
        self._counts[inp.op_name] += 1
    for inp_op_name, _ in inputs:
      self._counts[inp_op_name] += 1
    for out in outputs:
      self._counts[out.op_name] += 1

  def free_inputs(self, node):
    for inp in node.inputs + node.control_inputs:
      self._counts[inp.op_name] -= 1
      # Free up buffers.
      if not self._counts[inp.op_name]:
        del self.outputs[inp.op_name]


def _unique_everseen(seq):
  seen = set()
  return [x for x in seq if not (x in seen or seen.add(x))]


class _Subgraph(NamedTuple):
  """A subgraph of computations with a custom_gradient."""

  subgraph: Tuple[_OpNode, ...]
  captures: Tuple[_TensorEdge, ...]
  outputs: Tuple[_TensorEdge, ...]
  output_node: _OpNode
  grad_fn: _LibraryFunction

  @property
  def name(self) -> str:
    return self.output_node.name  # pytype: disable=attribute-error

  @property
  def inputs(self) -> Tuple[_TensorEdge, ...]:
    return self.function_inputs + self.captures

  @property
  def function_inputs(self) -> Tuple[_TensorEdge, ...]:
    """Inputs for the custom_gradient wrapped function."""
    return tuple(self.output_node.inputs[len(self.outputs):])  # pytype: disable=attribute-error

  @property
  def unique_inputs(self) -> Tuple[_TensorEdge, ...]:
    unique_captures = tuple([
        x for x in _unique_everseen(self.captures)
        if x not in set(self.function_inputs)
    ])
    return self.function_inputs + unique_captures

  @property
  def control_inputs(self) -> Tuple[_TensorEdge, ...]:
    return ()

  @property
  def require_rng(self) -> bool:
    return any([n.require_rng for n in self.subgraph])

  def __call__(
      self,
      named_args: Sequence[Tuple[str, jnp.ndarray]],
      *,
      rng: jnp.ndarray,
  ) -> Tuple[Tuple[jnp.ndarray, ...], Mapping[str, jnp.ndarray]]:
    grad_inputs = tuple([_TensorEdge(v.name) for v in self.grad_fn.input_specs])

    @jax.custom_gradient
    def fn(*args):
      eval_cache = _EvaluationCache(
          self.subgraph, named_args, self.output_node.inputs + grad_inputs)
      assert len(args) == len(self.unique_inputs)
      for inp, val in safe_zip(self.unique_inputs, args):
        if isinstance(eval_cache.outputs[inp.op_name], list):
          eval_cache.outputs[inp.op_name][inp.idx] = val
        elif isinstance(eval_cache.outputs[inp.op_name], tuple):
          old_outputs = eval_cache.outputs[inp.op_name]
          eval_cache.outputs[inp.op_name] = (
              old_outputs[:inp.idx] + (val,) + old_outputs[inp.idx + 1:])
        else:
          eval_cache.outputs[inp.op_name] = val

      num_rng_required = sum([node.require_rng for node in self.subgraph])
      if num_rng_required:
        rng_keys = list(jax.random.split(rng, num_rng_required))
      else:
        rng_keys = []

      for node in self.subgraph + (self.output_node,):
        collected_inputs = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in node.inputs
        ]
        sub_rng = rng_keys.pop() if node.require_rng else None
        eval_cache.outputs[node.name], updated_params = node(
            collected_inputs, rng=sub_rng)
        if updated_params:
          raise ValueError(
              "Variable assignments not supported in custom_gradient subgraph, "
              f"found {tuple(updated_params.keys())}.")

        eval_cache.free_inputs(node)

      # dy is the gradients for fn(*args) + args, i.e. inputs to IdentityN
      def grad_fn(dy):
        assert len(dy) == len(self.output_node.inputs)

        captured_inputs = tuple(grad_inputs[len(dy):])
        named_captures = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in captured_inputs
        ]
        captures = _unbox_named_args(named_captures, captured_inputs)
        grad_args = dy + captures

        dx = self.grad_fn(*grad_args)
        assert len(dx) == len(self.unique_inputs)

        return dx

      return eval_cache.outputs[self.output_node.name], grad_fn

    args_map = dict(named_args)
    fn_named_args = [
        (v.op_name, args_map[v.op_name]) for v in self.unique_inputs
    ]
    unboxed_args = _unbox_named_args(fn_named_args, self.unique_inputs)
    outputs = fn(*unboxed_args)
    return outputs, {}

  def rewrite(
      self,
      nodes: Sequence[_OpNode]) -> Tuple[Union["_Subgraph", _OpNode], ...]:
    """Remove subgraph from the main graph."""
    new_nodes = []
    processed = []
    subgraph = self.subgraph + (self.output_node,)
    subgraph_map = {v.name: v for v in subgraph}
    for node in nodes:
      if node.name in subgraph_map:
        processed.append(node)
        if node == self.output_node:
          new_nodes.append(self)
      else:
        # Rewire control inputs to point to the subgraph.
        new_control_inputs = tuple(
            v._replace(op_name=self.name) if v.op_name in subgraph_map else v
            for v in node.control_inputs
        )
        node.control_inputs = new_control_inputs
        new_nodes.append(node)

    assert len(processed) == len(subgraph), (len(processed), len(subgraph))
    return tuple(new_nodes)


def _contains_custom_gradient(node: tf.compat.v1.NodeDef) -> bool:
  # IdentityN are used to override gradients.
  return node.op == "IdentityN"


def _get_consumers_and_producers_fns(nodes):
  """Returns functions to get consumers and producers for a node."""
  op_map = {n.name: (idx, n) for idx, n in enumerate(nodes)}

  # Maps node name to immediate consumers.
  consumers_map = {n.name: [] for n in nodes}
  for node in nodes:
    for inp in node.inputs:
      consumers_map[inp.op_name].append(node.name)

  # Get all consumers for a node.
  @functools.lru_cache(None)
  def get_consumers(node_name: str) -> list[str]:
    if not consumers_map[node_name]:
      return [node_name]
    else:
      return sum([get_consumers(n) for n in consumers_map[node_name]], [])

  # Get all producers for a node.
  @functools.lru_cache(None)
  def get_producers(node_name: str):
    _, node = op_map[node_name]
    if node.inputs:
      return set.union(
          set([node_name]), *[get_producers(x.op_name) for x in node.inputs]
      )
    else:
      return set([node_name])

  return get_consumers, get_producers


# Extract subgraphs for custom_gradient.
def _extract_subgraphs(graphdef, nodes, library):
  """Extract all subgraphs with their own custom_gradients."""

  op_map = {n.name: (idx, n) for idx, n in enumerate(nodes)}
  if _EMPTY_RETURN_OP_NAME in op_map:
    logging.info("Skip subgraph extraction for function with no return values.")
    return {}

  get_consumers, get_producers = _get_consumers_and_producers_fns(nodes)

  subgraphs = {}
  for node in graphdef.node:
    if _contains_custom_gradient(node):
      grad_fn_name = str(node.attr["_gradient_op_type"].s, "utf-8")
      grad_fn = library[grad_fn_name]

      output_node = op_map[node.name][1]
      assert len(node.input) == len(output_node.inputs)

      # Inputs to the gradient function are fn(*args) + args + captured_args
      num_outputs = len(grad_fn.orig_fn_output_specs)
      all_specs = [_TensorEdge(x.name) for x in grad_fn.input_specs]
      outputs = list(output_node.inputs[:num_outputs])
      inputs = list(output_node.inputs[num_outputs:len(node.input)])
      captured_inputs = all_specs[len(node.input):]

      # Traverse and extract subgraph.
      subgraph = set([x.op_name for x in inputs + captured_inputs])
      unexpanded = [x.op_name for x in outputs]
      while unexpanded:
        top_node, *unexpanded = unexpanded
        if top_node not in subgraph:
          subgraph.add(top_node)
          unvisited = [x.op_name for x in op_map[top_node][1].inputs]
          unexpanded += unvisited

      # Separate internal and external captures. Internal captures are found in
      # the subgraph. External captures are found in outer graph.
      unused_captures = []
      internal_captures = []
      external_captures = []
      used_inputs = set(
          sum([op_map[n][1].inputs for n in subgraph], ()) + output_node.inputs)
      for inp in captured_inputs:
        is_internal = all(
            [x.op_name in subgraph for x in op_map[inp.op_name][1].inputs])
        if inp not in used_inputs:
          unused_captures.append(inp)
        elif is_internal:
          internal_captures.append(inp)
        else:
          external_captures.append(inp)

      # Find side-effects, i.e. nodes that depends on the subgraph but do not
      # feed into the subgraph outputs, e.g. shape check asserts from jax2tf.
      side_effects = []
      subgraph_and_output = subgraph | set([output_node.name])
      for node in nodes:
        if node.name not in subgraph_and_output:
          if any(inp.op_name in subgraph for inp in node.inputs):
            side_effects.append(set(get_consumers(node.name)))
      # Gather all dependencies of side-effects, assume they are not depended on
      # by other nodes outside of the subgraph + side-effects. This may only
      # work for shape check assert added by jax2tf.
      side_effects = set.union(set(), *side_effects)
      side_effect_deps = set()
      for x in set.union(set(), *[get_producers(x) for x in side_effects]):
        if op_map[x][1].op != "Placeholder":
          side_effect_deps.add(x)
      # Merge side-effects and dependencies into the subgraph.
      subgraph = subgraph | side_effects | side_effect_deps
      output_node.control_inputs = output_node.control_inputs + tuple(
          _TensorEdge(op_name=x, is_control=True) for x in side_effects
      )

      excluded = inputs + unused_captures + external_captures
      subgraph = subgraph.difference(set([x.op_name for x in excluded]))
      sub_nodes = [op_map[x] for x in subgraph]
      sub_nodes = [x for _, x in sorted(sub_nodes)]
      subgraphs[grad_fn_name] = _Subgraph(
          subgraph=tuple(sub_nodes),
          captures=tuple(external_captures),
          outputs=tuple(outputs),
          output_node=output_node,
          grad_fn=grad_fn,
      )

  num_nodes = sum([len(g.subgraph) for g in subgraphs.values()])
  num_unique_nodes = len(
      set(sum([[x.name for x in g.subgraph] for g in subgraphs.values()], [])))
  if num_nodes != num_unique_nodes:
    raise ValueError("Overlapping subgraphs are not yet supported.")

  return subgraphs


def _infer_relu_from_jax2tf(nodes):
  """Detect max(x, 0) and replace with jax.nn.relu."""
  found_jax2tf = any(["jax2tf_out" in n.name for n in nodes])
  node_map = {n.name: n for n in nodes}
  for node in nodes:
    if node.op == "Maximum" and "jax2tf" in node.name and "_relu_" in node.name:
      cast_or_const_arg = node_map[node.inputs[1].op_name]
      if cast_or_const_arg.op == "Cast":
        const_arg = node_map[cast_or_const_arg.inputs[0].op_name]
      else:
        const_arg = cast_or_const_arg

      if const_arg.op == "Const" and const_arg((), rng=None)[0].tolist() == 0:
        # Replace the Maximum op with a Relu op, but keep the node name.
        # The Cast and Const ops may now be redundant but are kept anyway.
        node.op = "Relu"
        node.inputs = node.inputs[:1]
        node.jax_func = (
            ops.get_parser("Relu")(_NodeDef("Relu", node.name, (), {})))
        if not found_jax2tf:
          logging.warning("Replaced max(x, 0) with jax.nn.relu but did not "
                          "find jax2tf_out.")


_FunctionDef = Any


def _get_function_protos(
    graphdef: tf.compat.v1.GraphDef,) -> Iterator[Tuple[str, _FunctionDef]]:
  """Get all library function protos from a tf.GraphDef."""

  func_protos = {
      func.signature.name: func for func in graphdef.library.function
  }
  processed = set()

  def process_func(func) -> Iterator[Tuple[str, _FunctionDef]]:
    for node in func.node_def:
      yield from process_node(node)

  def process_node(node) -> Iterator[Tuple[str, _FunctionDef]]:
    for attr in node.attr.values():
      for func_name in [attr.func.name] + [f.name for f in attr.list.func]:
        if func_name and func_name not in processed:
          yield from process_func(func_protos[func_name])
          yield func_name, func_protos[func_name]
          processed.add(func_name)

  for node in graphdef.node:
    yield from process_node(node)


# Partially mimics tf.compat.v1.NodeDef
class _NodeDef(NamedTuple):
  op: str
  name: str
  input: Tuple[str, ...]
  attr: Optional[Mapping[str, tf.compat.v1.AttrValue]] = None


class _GraphDef(NamedTuple):
  node: Tuple[Union[tf.compat.v1.NodeDef, _NodeDef], ...]


def _validate_inputs(
    signature: inspect.Signature,
    structured_specs: Any,
    input_map: Mapping[str, Any],
) -> str:
  """Validate inputs against input specs."""

  class _ExpectedArg:
    def __init__(self, path: Tuple[Any, ...], spec: Any):
      self.path = path
      self.spec = spec

  expected_args = [
      _ExpectedArg(p, v) for p, v in tree.flatten_with_path(structured_specs)
  ]
  expected_tree = tree.unflatten_as(structured_specs, expected_args)

  def format_spec(spec: tf.TensorSpec) -> str:
    return "{}[{}]".format(spec.dtype.name, ",".join(map(str, spec.shape)))

  def check_arg(arg: _ExpectedArg):
    arg_desc = "UNUSED" if arg.spec is _UNUSED_INPUT else format_spec(arg.spec)
    return "." if arg.path in input_map else arg_desc

  checked_args, checked_kwargs = tree.map_structure(check_arg, expected_tree)
  bound_checked_args = signature.bind(*checked_args, **checked_kwargs)
  # TODO(b/282901848) Use a better pretty printer than json which does not
  # render namedtuple correctly.
  return json.dumps(bound_checked_args.arguments, indent=4)


class MissingInputError(Exception):
  ...


def _convert(
    graphdef: Union[tf.compat.v1.GraphDef, _GraphDef],
    signature: inspect.Signature,
    structured_inputs,
    structured_outputs,
    captured_input_names: Optional[Tuple[str, ...]] = None,
    variable_map: Optional[Mapping[str, tf.Variable]] = None,
    constants: Optional[Mapping[str, jnp.ndarray]] = None,
    library: Optional[Mapping[str, _LibraryFunction]] = None,
) -> Tuple[Callable[..., Any], Mapping[str, Variable]]:
  """Convert a GraphDef to a Jax function.

  Args:
    graphdef: a tf.GraphDef or GraphDef like object.
    signature: An inspect.Signature representing a call signature.
    structured_inputs: Structured input spec for the function, follows the
      format of inspect.BoundArguments, i.e. a tuple of a list of position args
      and a dict of keyword-only args.
    structured_outputs: Structured output spec for the function.
    captured_input_names: Names of other input tensors that are not part of
      `structured_inputs`, e.g. variables captured by tf.Function.
    variable_map: A mapping from tensor names to tf.Variables. The keys are a
      subset of captured_input_names.
    constants: A mapping from tensor names to constant values. The keys are a
      subset of captured_input_names.
    library: A mapping from function names to Callable. This is non-empty on
      recurisve calls if the FunctionDefLibrary in the GraphDef is non-empty.

  Returns:
    A tuple: the first is a Jax functions that takes a flat parameter dict and a
    variable number of arrays as inputs, and returns a nested structure of
    outputs and an updated parameter dict; the second is the flat parameter dict
    correpondings to all TF variables used by the graph.
  """
  # Recursively convert function protos in FunctionDefLibrary.
  library = dict((library or {}).items())
  if hasattr(graphdef, "library"):
    if graphdef.library.gradient:
      raise ValueError("GradientDef not currently supported, found "
                       f"{graphdef.library.gradient}")

    for func_name, func_proto in _get_function_protos(graphdef):
      if func_name not in library:
        logging.info("Converting library function %s", func_name)
        library[func_name] = _convert_library_function(func_proto, library)

  captured_input_names = captured_input_names or ()
  variable_map = variable_map or {}
  constants = constants or {}

  # Canonicalise inputs and outputs.
  input_path_to_specs = [(p, v)
                         for p, v in tree.flatten_with_path(structured_inputs)
                         if v is not _UNUSED_INPUT]
  # TODO(b/266552275) Remove temporary fix for TF-Hub.
  replace_fn = (lambda k: k.replace("$", "___") if isinstance(k, str) else k)
  input_path_to_specs = [
      (tree.map_structure(replace_fn, p), v) for p, v in input_path_to_specs
  ]

  # Extract input and output tensor names.
  input_names = []
  static_arg_num = 0
  for (_, spec) in input_path_to_specs:
    if isinstance(spec, tf.TensorSpec):
      input_names.append(spec.name)
    else:
      input_names.append(f"__static_arg_{static_arg_num}")
      static_arg_num += 1
  input_names = tuple(input_names)
  assert len(input_names) == len(set(input_names))
  flat_output_specs = tree.flatten(structured_outputs)
  output_names = tuple(
      v.name for v in flat_output_specs if isinstance(v, tf.TensorSpec)
  )

  unsupported = ops.get_unsupported_operations(
      [node.op for node in graphdef.node])
  if unsupported:
    err_message = f"Unsupported operations in graph: {list(unsupported)}"
    if _XLA_CALL_MODULE in unsupported:
      err_message += "\n"
      err_message += (
          f"`{_XLA_CALL_MODULE}` is generated by jax2tf.convert with native "
          "serialization, this is partially supported in tf2jax (check for "
          "import error if you see this message)."
      )
    err_message += "\n"
    err_message += (
        "Support for additional TensorFlow ops are added on an as-needed "
        "basis, please contact the library owner(s)."
    )
    raise ValueError(err_message)

  # Extract variables.
  if tf.executing_eagerly():
    # Uniqueify variables with identical names.
    variables_tf = {}
    var_name_by_ref = {}
    for v in variable_map.values():
      name = _parse_input(v.name)
      suffix = 1
      var_name = name
      while var_name in variables_tf:
        var_name = f"{name}_{suffix}"
        suffix += 1
      variables_tf[var_name] = v
      var_name_by_ref[v.ref()] = var_name

    variables = {
        k: Variable(v.numpy(), v.trainable, v.name)
        for k, v in variables_tf.items()
    }
  else:
    variables_tf = {_parse_input(v.name): v for _, v in variable_map.items()}
    var_name_by_ref = {
        v.ref(): _parse_input(v.name) for v in variable_map.values()
    }
    if variables_tf:
      with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.initialize_variables(variables_tf.values()))
        variables_np = sess.run(variables_tf)
        variables = tree.map_structure(
            lambda var, arr: Variable(arr, var.trainable, var.name),
            variables_tf, variables_np)
    else:
      variables = {}

  assert len(variable_map) == len(variables_tf)
  assert len(variable_map) == len(var_name_by_ref)
  assert len(variable_map) == len(variables)

  var_by_node = {k: var_name_by_ref[v.ref()] for k, v in variable_map.items()}
  node_by_var = {v: k for k, v in var_by_node.items()}
  assert len(var_by_node) == len(node_by_var)

  node_map = {n.name: n for n in graphdef.node}
  if not output_names:
    assert _EMPTY_RETURN_OP_NAME not in node_map
    assign_nodes = tuple(
        f"^{n.name}" for n in graphdef.node if n.op in _TF_ASSIGN_OPS)
    node_map[_EMPTY_RETURN_OP_NAME] = _NodeDef(
        "NoOp", _EMPTY_RETURN_OP_NAME, assign_nodes, {})
    output_names = [_EMPTY_RETURN_OP_NAME]
    logging.warning(
        "No output nodes found, inserted NoOp to trigger side effects: %s",
        assign_nodes)

  nodes = _toposort(node_map, output_names)
  nodes = [_OpNode(node, library, node_map) for node in nodes]
  output_args = [_TensorEdge.from_string(v, node_map) for v in output_names]
  num_rng_required = sum([node.require_rng for node in nodes])

  if config.get_config("infer_relu_from_jax2tf"):
    _infer_relu_from_jax2tf(nodes)

  if config.get_config("convert_custom_gradient"):
    subgraphs = _extract_subgraphs(graphdef, nodes, library)
    for _, subgraph in subgraphs.items():
      nodes = subgraph.rewrite(nodes)

  def jax_func(
      params: Mapping[str, jnp.ndarray],
      *func_args,
      rng: Optional[jnp.ndarray] = None,
      **func_kwargs,
  ) -> Tuple[Any, Mapping[str, jnp.ndarray]]:
    if num_rng_required and rng is None:
      raise ValueError(f"PRNG key required for random ops, found rng={rng}.")

    bound_args = signature.bind(*func_args, **func_kwargs)
    bound_args.apply_defaults()
    inputs = dict(
        tree.flatten_with_path((bound_args.args, bound_args.kwargs)))
    try:
      inputs = tuple([inputs[p] for p, _ in input_path_to_specs])
    except KeyError as e:
      input_err_message = _validate_inputs(signature, structured_inputs, inputs)
      err_message = (
          "\nSome input(s) are missing (entries that map to dtype[shape]"
          f" annotations in the following structure): \n{input_err_message}"
      )
      raise MissingInputError(err_message) from e

    if len(inputs) != len(input_names):
      raise ValueError(
          f"Expected {len(input_names)} args, found {len(inputs)}.")

    inputs = tree.map_structure(
        lambda x: x if isinstance(x, jnp.ndarray) else np.array(x), inputs)
    all_params = dict(**params, **constants)

    for inp, (path, spec) in safe_zip(inputs, input_path_to_specs):
      if not isinstance(spec, tf.TensorSpec):
        # A `None` spec denotes a Tensor argument that was unused in deepfunc.
        is_tracer = isinstance(inp, jax.core.Tracer)
        if spec is not None and (is_tracer or spec != inp):
          inp_type = "tracer" if is_tracer else "literal value"
          raise ValueError(
              f"Found unexpected {inp_type} `{inp}`, expected `{spec}` for "
              f"argument at {path}. This may be because the original "
              "TensorFlow function was traced with a literal value instead of "
              "a Tensor or Array.")
        else:
          continue
      if (config.get_config("strict_shape_check") and
          not spec.shape.is_compatible_with(_fix_jax_poly_shape(inp.shape))):
        raise ValueError(
            f"Found incompatible input shape: {inp.shape}, expected "
            f"{spec.shape} for argument at {path}.\n"
            "This can be a problem if the outputs depend on input shapes, e.g. "
            "BatchNorm during training.\n"
            "If this is not an issue for you, the error can be disabled using "
            "either tf2jax.update_config('strict_shape_check', False) or the "
            "context manager "
            "tf2jax.override_config('strict_shape_check', False)")
      if (config.get_config("strict_dtype_check") and
          not spec.dtype.is_compatible_with(inp.dtype)):
        raise ValueError(
            f"Found incompatible input dtype: {inp.dtype}, expected "
            f"{spec.dtype} for argument at {path}.\n"
            "If this is not an issue for you, the error can be disabled "
            "either tf2jax.update_config('strict_dtype_check', False) or the "
            "context manager "
            "tf2jax.override_config('strict_dtype_check', False)")

    missing_params = [
        var_by_node.get(v, v)
        for v in captured_input_names
        if var_by_node.get(v, v) not in all_params
    ]
    if missing_params:
      raise ValueError(f"Some parameters are missing, {missing_params}.")

    full_inputs = inputs + tuple(
        [all_params[var_by_node.get(v, v)] for v in captured_input_names])
    full_inputs = list(
        safe_zip(input_names + captured_input_names, full_inputs)
    )

    if num_rng_required:
      rng_keys = list(jax.random.split(rng, num_rng_required))
    else:
      rng_keys = []

    updated_param_names = set()
    eval_cache = _EvaluationCache(nodes, full_inputs, output_args)
    for node in nodes:
      if node.name not in eval_cache.outputs:
        # Double-check control inputs.
        for inp in node.control_inputs:
          if inp.op_name not in eval_cache.outputs:
            raise ValueError(
                f"Control dependency {inp} not executed for node `{node.name}`")
        collected_inputs = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in node.inputs
        ]
        sub_rng = rng_keys.pop() if node.require_rng else None
        eval_cache.outputs[node.name], updated_params = node(
            collected_inputs, rng=sub_rng)
        # Assign variables.
        for var_name, var_val in updated_params.items():
          eval_cache.outputs[var_name] = var_val
          updated_param_names.add(var_name)

        eval_cache.free_inputs(node)

    tensor_outputs = tuple([eval_cache.outputs[k.op_name] for k in output_args])
    tensor_outputs = [v for v in tensor_outputs if v is not _EMPTY_RETURN_VALUE]

    # Merge the tensor and non-tensor outputs.
    output_idx = 0
    flat_outputs = []
    for spec in flat_output_specs:
      if isinstance(spec, tf.TensorSpec):
        flat_outputs.append(tensor_outputs[output_idx])
        output_idx += 1
      else:
        flat_outputs.append(spec)
    assert output_idx == len(tensor_outputs)
    collected_outputs = tree.unflatten_as(structured_outputs, flat_outputs)

    # Parameters after any assignment.
    new_params = {
        var_name: eval_cache.outputs[node_by_var[var_name]]
        for var_name in params.keys()
    }

    lost_params = [v for v in updated_param_names if v not in var_by_node]
    if lost_params:
      raise ValueError(f"Some updated parameters are lost, {lost_params}.")

    return collected_outputs, new_params

  return jax_func, variables


def _convert_library_function(
    proto,
    library: Optional[Mapping[str, _LibraryFunction]],
) -> _LibraryFunction:
  """Convert a FunctionDef."""
  input_nodes = []
  for arg in proto.signature.input_arg:
    input_nodes.append(
        _NodeDef("Placeholder", arg.name, (), {"dtype": tf.as_dtype(arg.type)}))

  input_arg_names = [arg.name for arg in proto.signature.input_arg]

  output_nodes = []
  output_is_input = []
  for arg in proto.signature.output_arg:
    # TODO(b/233985145) Keep following the Identity ops through the node_def?
    output_is_input.append(proto.ret[arg.name] in input_arg_names)
    output_nodes.append(
        _NodeDef(
            "Identity",
            arg.name,
            tuple([proto.ret[arg.name]] + ["^" + v for v in proto.control_ret]),
            {"T": tf.as_dtype(arg.type)},
        ))
  graphdef = _GraphDef(tuple(input_nodes + list(proto.node_def) + output_nodes))
  output_is_input = tuple(output_is_input)

  # VarHandleOp correspond to variables in checkpoints.
  var_handle_ops = [n for n in proto.node_def if n.op == "VarHandleOp"]

  params = [
      inspect.Parameter(arg.name, inspect.Parameter.POSITIONAL_ONLY)
      for arg in proto.signature.input_arg
  ] + [
      inspect.Parameter(
          arg.name.replace("/", "___"), inspect.Parameter.POSITIONAL_ONLY
      )
      for arg in var_handle_ops
  ]
  signature = inspect.Signature(params)

  structured_inputs = tuple([
      tf.TensorSpec(None, dtype=tf.as_dtype(arg.type), name=arg.name)
      for arg in proto.signature.input_arg
  ])
  structured_outputs = tuple([
      tf.TensorSpec(None, dtype=tf.as_dtype(arg.type), name=arg.name)
      for arg in proto.signature.output_arg
  ])

  def node_to_spec(v):
    return tf.TensorSpec(
        shape=[dim.size for dim in v.attr["shape"].shape.dim],
        dtype=tf.as_dtype(v.attr["dtype"].type),
        name=v.name,
    )

  var_inputs = tuple(node_to_spec(v) for v in var_handle_ops)
  structured_inputs = structured_inputs + var_inputs

  jax_func, jax_params = _convert(
      graphdef,
      signature,
      [structured_inputs, {}],
      structured_outputs,
      library=library)
  if jax_params:
    raise ValueError(
        f"Library function should be stateless, found variables {jax_params}")

  # Does any of the ops or inner functions require RNG?
  require_rng = any([n.op in _TF_RANDOM_OPS for n in proto.node_def])
  for node in proto.node_def:
    for attr in node.attr.values():
      if attr.func.name:
        require_rng = require_rng or library[attr.func.name].require_rng

  return _LibraryFunction(
      fn=jax_func,
      require_rng=require_rng,
      params=None,
      input_specs=structured_inputs,
      output_specs=structured_outputs,
      output_is_input=output_is_input,
      variable_input_specs=var_inputs,
  )


def _filter_nodes(
    predicate: Callable[[tf.compat.v1.NodeDef], bool],
    graph: Any,
) -> Iterator[Tuple[tf.compat.v1.NodeDef, Any]]:
  """Filter nodes in a tf.Graph with a predicate."""
  graph_def = graph.as_graph_def()
  for node in graph_def.node:
    if predicate(node):
      yield node, graph
  if hasattr(graph_def, "library"):
    for func in graph_def.library.function:
      # TODO(b/266553384) Use the public API once it is available.
      library_fn = graph._functions[func.signature.name]  # pylint: disable=protected-access
      for node in func.node_def:
        if predicate(node):
          yield node, library_fn.graph


def _convert_all_gradient_functions(
    graph: Any,
    library: Mapping[str, _LibraryFunction],
) -> Mapping[str, _LibraryFunction]:
  """Recursively convert all custom gradients in a tf.Graph."""
  grad_lib = {}
  for node, graph in _filter_nodes(_contains_custom_gradient, graph):
    # Note that dict(**a, **b) will raise TypeError on dupliates, unlike {}.
    grad_lib.update(
        _convert_gradient_function(node, graph, dict(**library, **grad_lib)))
  return grad_lib


def _convert_gradient_function(
    proto: tf.compat.v1.NodeDef,
    graph: Any,
    library: Mapping[str, _LibraryFunction],
) -> Mapping[str, _LibraryFunction]:
  """Convert a custom_gradient function."""
  op = graph.as_graph_element(proto.name)
  input_specs = tuple([tf.TensorSpec.from_tensor(v) for v in op.inputs])
  grad_fn_name = str(proto.attr["_gradient_op_type"].s, "utf-8")
  if grad_fn_name in library:
    return {}

  @tf.function
  def tf_grad_fn(*grad_args, **grad_kwargs):
    fn = tf_ops.gradient_registry.lookup(grad_fn_name)
    return fn(None, *grad_args, **grad_kwargs)

  concrete_tf_grad_fn = tf_grad_fn.get_concrete_function(*input_specs)
  grad_lib = _convert_all_gradient_functions(concrete_tf_grad_fn.graph, library)

  logging.info("Converting gradient function %s", grad_fn_name)
  grad_inputs = concrete_tf_grad_fn.inputs
  grad_captured_inputs = concrete_tf_grad_fn.captured_inputs
  num_flat_args = len(grad_inputs) - len(grad_captured_inputs)
  func_variables = {v.handle.ref(): v for v in concrete_tf_grad_fn.variables}

  # Gradient function can capture tensors in the outer function. Move them
  # into the arguments of the gradient function for conversion to JAX.
  variable_map = {}
  constant_map = {}
  external_capture_specs = []
  internal_capture_names = []
  for inp, cap in safe_zip(grad_inputs[num_flat_args:], grad_captured_inputs):
    if cap.dtype == tf.resource:
      variable_map[inp.op.name] = func_variables[cap.ref()]
      internal_capture_names.append(inp.op.name)
    elif hasattr(cap, "numpy"):
      constant_map[inp.op.name] = cap.numpy()
      internal_capture_names.append(inp.op.name)
    else:
      external_capture_specs.append(tf.TensorSpec.from_tensor(cap))

  structured_grad_input_specs = tree.map_structure(tf.TensorSpec.from_tensor,
                                                   concrete_tf_grad_fn.inputs)
  structured_grad_input_specs = (structured_grad_input_specs, {})
  grad_input_specs = input_specs + tuple(external_capture_specs)
  grad_structured_outputs = tuple(
      itertools.dropwhile(lambda x: x is None,
                          concrete_tf_grad_fn.structured_outputs))
  grad_output_specs = tuple([
      tf.TensorSpec.from_tensor(x) for x in grad_structured_outputs
  ])
  # Nones correspond to the outputs of the original function.
  num_fn_outputs = (
      len(concrete_tf_grad_fn.structured_outputs) -
      len(grad_structured_outputs))
  signature = inspect.Signature(
      (inspect.Parameter("grad_args", inspect.Parameter.VAR_POSITIONAL),))

  jax_grad_fn, jax_grad_params = _convert(
      concrete_tf_grad_fn.graph.as_graph_def(),
      signature,
      structured_grad_input_specs,
      grad_output_specs,
      captured_input_names=tuple(internal_capture_names),
      variable_map=variable_map,
      constants=constant_map,
      # Note that dict(**a, **b) will raise TypeError on dupliates, unlike {}.
      library=dict(**library, **grad_lib),
  )
  grad_fn = _LibraryFunction(jax_grad_fn, False, jax_grad_params,
                             grad_input_specs, grad_output_specs,
                             grad_output_specs[:num_fn_outputs])
  return dict(**grad_lib, **{grad_fn_name: grad_fn})
