# Conversion Phases

*The most important mental model: build time and run time are two distinct phases, both implemented inside `_convert()` and the closure it returns.*

**Source:** `tf2jax/_src/tf2jax.py:976–1285`

## Overview

`_convert()` is a function that:
1. Does a lot of work upfront (build time), then
2. Returns a `jax_func` closure that captures all that work (run time).

A reader must always know which phase they are in. Code above the `def jax_func(...)` definition runs once at conversion time. Code inside `jax_func` runs on every call.

## Build time (inside `_convert`, before `def jax_func`)

In order:

1. **Recurse into FunctionDefLibrary** — any library functions referenced by the graph are converted first and stored in `library: dict[str, _LibraryFunction]`. Each is wrapped in a `_CachedBuilder` for lazy JAX compilation.

2. **Canonicalise inputs/outputs** — flat lists of `input_names`, `input_path_to_specs`, `output_names` are extracted from `structured_inputs`/`structured_outputs`. These are the node names used as keys in the evaluation cache at run time.

3. **Extract variables** — `variable_map` (ref → `tf.Variable`) is evaluated eagerly: `v.numpy()` is called and the result is stored as `Variable` (a NumPy subclass). Two index structures are built: `var_by_node` (node name → variable name) and `node_by_var` (variable name → node name). These are needed at run time to connect `AssignVariableOp` outputs back to the parameter dict.

4. **Topological sort** — `_toposort(node_map, output_names)` produces the ordered list of `NodeDef` protos. Only nodes reachable from outputs are included.

5. **Build `_OpNode` wrappers** — each sorted proto becomes an `_OpNode`. The constructor calls `ops.get_parser(proto.op)(proto)` which bakes static attributes (strides, padding, axis, dtype, etc.) into a closure that is stored as `node.jax_func`.

6. **Relu inference heuristic** — if `infer_relu_from_jax2tf` is set, `_infer_relu_from_jax2tf(nodes)` mutates matching `Maximum` nodes into `Relu` nodes in-place.

7. **Custom gradient subgraph extraction** — if `convert_custom_gradient` is set, `_extract_subgraphs(graphdef, nodes, library)` detects `IdentityN` nodes (the TF custom gradient marker), extracts the surrounding computation, and calls `subgraph.rewrite(nodes)` to replace affected `_OpNode`s with a single `_Subgraph` object.

After step 7, `nodes` is a `list[_OpNode | _Subgraph]`. Both implement the same calling protocol (see [entities/_OpNode.md](../entities/_OpNode.md) and [entities/_Subgraph.md](../entities/_Subgraph.md)).

## What the closure captures

The `jax_func` closure closes over these 12 build-time variables:

| Variable | Type | Purpose |
|----------|------|---------|
| `nodes` | `list[_OpNode \| _Subgraph]` | ordered execution sequence |
| `num_rng_required` | `int` | how many random ops need RNG keys |
| `input_path_to_specs` | `list[(path, TensorSpec)]` | maps arg paths → tensor names |
| `input_names` | `tuple[str]` | names for user-provided inputs |
| `captured_input_names` | `tuple[str]` | names for variable/constant captures |
| `constants` | `dict[str, ndarray]` | constant tensors by node name |
| `var_by_node` | `dict[str, str]` | node name → parameter name |
| `node_by_var` | `dict[str, str]` | parameter name → node name |
| `flat_output_specs` | `list` | output structure with TensorSpec leaves |
| `structured_outputs` | nested | for `tree.unflatten_as` at the end |
| `output_args` | `list[_TensorEdge]` | which cache entries are outputs |
| `signature` | `inspect.Signature` | for `signature.bind(*args, **kwargs)` |

> **Refactoring note:** These 12 implicit captures are a major cognitive load. A `_CompiledGraph` class with these as explicit `self.` fields would make the data model visible. See [design/refactoring-recommendations.md](../design/refactoring-recommendations.md).

## Run time (inside `jax_func`)

1. **Bind and flatten args** — `signature.bind(*func_args, **func_kwargs)` + `bound_args.apply_defaults()` → `inputs` (flat tuple of arrays).

2. **Validate inputs** — shape check (`strict_shape_check`) and dtype check (`strict_dtype_check`) against the stored specs.

3. **Assemble full inputs** — user inputs + captured params/constants → `full_inputs: list[(name, value)]`.

4. **Split RNG** — `jax.random.split(rng, num_rng_required)` if any random ops exist.

5. **Evaluation loop** — for each node in topological order, collect its inputs from `eval_cache.outputs`, call the node, store outputs back, free consumed intermediates (reference counting in `_EvaluationCache`). See [concepts/graph-evaluation.md](graph-evaluation.md).

6. **Reconstruct outputs** — tensor outputs are unflattened using `structured_outputs` and non-tensor outputs (pass-through specs) are reinserted.

7. **Return updated params** — for each variable in `params`, the latest value is read from `eval_cache.outputs[node_by_var[var_name]]`.

## Related pages

- [overview.md](../overview.md)
- [concepts/graph-evaluation.md](graph-evaluation.md)
- [concepts/library-functions.md](library-functions.md)
- [concepts/custom-gradients.md](custom-gradients.md)
- [entities/_OpNode.md](../entities/_OpNode.md)
- [design/refactoring-recommendations.md](../design/refactoring-recommendations.md)
