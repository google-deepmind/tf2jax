# Architecture Overview

*tf2jax converts TensorFlow functions to JAX functions by transpiling computational graphs, not by interpreting eager execution.*

## What it does

Given a `tf.Function`, tf2jax returns an equivalent JAX function and a dict of parameters (the captured TF Variables). The JAX function is purely functional: variables become explicit positional parameters, and updates are returned rather than applied in-place.

```python
jax_fn, params = tf2jax.convert(tf_model, *sample_inputs)
outputs, new_params = jax_fn(params, *real_inputs)
```

## Three-layer data model

```
tf.Function
    │  trace with sample inputs
    ▼
tf.ConcreteFunction  (GraphDef + captured variables)
    │  _convert()
    ▼
[_OpNode, _OpNode, _Subgraph, ...]   ← topologically sorted, build time
    │  jax_func closure / CompiledGraph
    ▼
JAX arrays                           ← run time
```

## Two phases: build time vs run time

This is the most important mental model for reading the code.

**Build time** (happens once, inside `convert()` and `_convert()`):
- TF function is traced with sample inputs → `ConcreteFunction`
- `GraphDef` is extracted
- Library functions (`FunctionDefLibrary`) are recursively converted
- Custom gradient functions are traced (TF) and registered
- Nodes are topologically sorted
- `_OpNode` wrappers are built (each calls `ops.get_parser(proto.op)(proto)` to bake in static attributes)
- Custom gradient subgraphs are detected and nodes are rewritten to `_Subgraph` objects
- A `jax_func` closure (or future: `_CompiledGraph`) is returned, capturing all build-time state

**Run time** (every call to `jax_func`):
- User args are bound to the signature
- Input shapes/dtypes are checked against specs
- `_EvaluationCache` is initialized
- Nodes are executed in topological order
- Variable assignments are tracked and returned

## Module map

| Module | Role |
|--------|------|
| `tf2jax/_src/tf2jax.py` | Conversion engine, public API, `_OpNode`, `_Subgraph`, `_LibraryFunction`, `Variable` |
| `tf2jax/_src/ops.py` | Op registry (`_jax_ops`), `register_operation()`, `_HigherOrderFunction`, 150+ op implementations |
| `tf2jax/_src/numpy_compat.py` | Dual NumPy/JAX-NumPy dispatch layer for constant folding during conversion |
| `tf2jax/_src/config.py` | 11 global config flags with context-manager overrides |
| `tf2jax/_src/xla_utils.py` | XLA protobuf helpers (convolution/dot/gather dimension numbers) |
| `tf2jax/_src/utils.py` | `fullargspec_to_signature`, `safe_zip` |
| `tf2jax/experimental/ops.py` | `XlaCallModule` support (native serialization from jax2tf) |
| `tf2jax/experimental/mhlo.py` | MLIR/StableHLO dialect abstractions |

## Public API

See [entities/convert-api.md](entities/convert-api.md).

## Key design decisions

1. **Graph-level transpilation** (not eager execution wrapping): the ConcreteFunction's GraphDef is parsed and each TF op is mapped to a JAX callable.
2. **Variables become parameters**: `tf.Variable` → `Variable` (NumPy subclass) in the returned `params` dict. Functional update semantics.
3. **Two-level op factory**: `_jax_ops[op_name]` holds a *factory* that takes a `NodeDef` proto and returns the actual JAX callable. Static attributes are baked in at build time.
4. **Custom gradients via subgraph extraction**: `IdentityN` marker nodes are detected, the surrounding computation is extracted as a `_Subgraph`, and wrapped with `jax.custom_gradient`.
5. **Lazy library function compilation**: `_CachedBuilder` ensures each library function is compiled at most once, with the config state frozen at compilation time.

## Related pages

- [concepts/conversion-phases.md](concepts/conversion-phases.md)
- [concepts/op-registry.md](concepts/op-registry.md)
- [concepts/graph-evaluation.md](concepts/graph-evaluation.md)
- [concepts/variable-semantics.md](concepts/variable-semantics.md)
- [concepts/custom-gradients.md](concepts/custom-gradients.md)
- [concepts/library-functions.md](concepts/library-functions.md)
