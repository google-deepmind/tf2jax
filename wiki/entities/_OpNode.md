# `_OpNode`

*Wraps a TF `NodeDef` proto, bakes in its static attributes at build time, and provides a uniform call interface for the evaluation loop.*

**Source:** `tf2jax/_src/tf2jax.py:239–295`

## Construction

```python
class _OpNode:
    def __init__(self, proto, library, node_map):
        self.jax_func = ops.get_parser(proto.op)(proto)   # factory → jax callable
        self.op = proto.op
        self.name = proto.name
        
        inputs = [_TensorEdge.from_string(inp, node_map) for inp in proto.input]
        
        self.inner_fns = {}
        if isinstance(self.jax_func, ops._HigherOrderFunction):
            self.inner_fns = self.jax_func.get_inner_functions(library)
            # higher-order ops may need extra input edges (e.g. VarHandleOp variables)
            for input_name in self.jax_func.get_additional_inputs(**self.inner_fns):
                inputs.append(_TensorEdge.from_string(input_name, node_map))
        
        self.control_inputs = tuple(inp for inp in inputs if inp.is_control)
        self.inputs = tuple(inp for inp in inputs if not inp.is_control)
```

`ops.get_parser(proto.op)` retrieves the factory function from `_jax_ops`, then immediately calls it with `proto`. The factory reads all static attributes (strides, padding, dtype, axis, etc.) and returns a JAX callable closed over them. That callable is stored as `self.jax_func`.

## `require_rng`

```python
@property
def require_rng(self) -> bool:
    inner_require_rngs = any(fn.require_rng for fn in self.inner_fns.values())
    return self.op in _TF_RANDOM_OPS or inner_require_rngs
```

`_TF_RANDOM_OPS = ("RandomStandardNormal", "RandomUniform", "RandomUniformInt")`. Higher-order ops propagate the flag from their inner functions.

## `__call__`

```python
def __call__(self, named_args, *, rng):
    unboxed_args = _unbox_named_args(named_args, self.inputs)
    extras = dict(rng=rng) if self.require_rng else {}
    extras.update(self.inner_fns)
    outputs = self.jax_func(*unboxed_args, **extras)
    
    if self.op in _TF_ASSIGN_OPS:
        updated_params = {named_args[0][0]: outputs}
    else:
        updated_params = {}
    
    return outputs, updated_params
```

`named_args` is a `list[(node_name, value)]` collected by the evaluation loop from `eval_cache.outputs`. `_unbox_named_args` validates names and extracts `value[inp.idx]` for tuple outputs.

For assign ops, the first argument is the variable handle (its node name is the variable's cache key), so `named_args[0][0]` gives the correct key for `updated_params`.

## Protocol shared with `_Subgraph`

Both `_OpNode` and `_Subgraph` implement:

| Attribute/Method | Type | Purpose |
|----------|------|---------|
| `.name` | `str` | unique node name, used as cache key |
| `.inputs` | `Tuple[_TensorEdge]` | data input edges |
| `.control_inputs` | `Tuple[_TensorEdge]` | control dependency edges |
| `.require_rng` | `bool` | whether this node needs a PRNG key |
| `.__call__(named_args, *, rng)` | — | executes the node, returns `(outputs, updated_params)` |

There is no formal interface declaration. See [design/refactoring-recommendations.md](../design/refactoring-recommendations.md) for a proposal to add a `_GraphNode` Protocol.

## `__repr__`

```
_OpNode(name='dense/MatMul', op='MatMul', inputs=['x', 'kernel'])
```

## Variable mutation note

`_infer_relu_from_jax2tf` mutates `_OpNode` instances in-place (`.op`, `.inputs`, `.jax_func`). This is the only place in the codebase where nodes are modified after construction.

## Related pages

- [concepts/op-registry.md](../concepts/op-registry.md)
- [concepts/graph-evaluation.md](../concepts/graph-evaluation.md)
- [entities/_TensorEdge.md](_TensorEdge.md)
- [entities/_Subgraph.md](_Subgraph.md)
- [entities/_HigherOrderFunction.md](_HigherOrderFunction.md)
