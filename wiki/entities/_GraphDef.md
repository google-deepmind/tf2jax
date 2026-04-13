# `_GraphDef` and `_NodeDef` — Synthetic Graph Types

*Lightweight NamedTuples that mimic the TF protobuf types, used to represent library functions as graphs so `_convert` can handle both the top-level graph and inner functions uniformly.*

**Source:** `tf2jax/_src/tf2jax.py:929–938`

## The problem

TF has two different graph representations:

- **`tf.compat.v1.GraphDef`** (protobuf): produced by `concrete_func.graph.as_graph_def()`. Contains `node` (list of `NodeDef`) and `library` (a `FunctionDefLibrary` of `FunctionDef`s).
- **`FunctionDef`** (protobuf): the format for library functions. Has `signature.input_arg`, `signature.output_arg`, `node_def`, and `ret` (output mapping) — a different structure from `GraphDef`.

`_convert` is written to operate on graph-like objects with a flat `node` list. Library functions need to be converted by `_convert` too, but their proto structure doesn't match.

## The solution: synthetic wrappers

```python
class _NodeDef(NamedTuple):
    op: str
    name: str
    input: Tuple[str, ...]
    attr: Optional[Mapping[str, tf.compat.v1.AttrValue]] = None

class _GraphDef(NamedTuple):
    node: Tuple[Union[tf.compat.v1.NodeDef, _NodeDef], ...]
```

`_convert_library_function` assembles a `_GraphDef` from a `FunctionDef` by synthesising nodes:

```python
# Each input_arg → Placeholder _NodeDef
input_nodes = [_NodeDef("Placeholder", arg.name, (), {"dtype": ...})
               for arg in proto.signature.input_arg]

# Each output_arg → Identity _NodeDef  
output_nodes = [_NodeDef("Identity", arg.name, (proto.ret[arg.name], ...), {"T": ...})
                for arg in proto.signature.output_arg]

graphdef = _GraphDef(tuple(input_nodes + list(proto.node_def) + output_nodes))
```

This transforms the `FunctionDef`'s `ret` mapping (output_name → node:idx string) into a standard `Identity` node that the existing graph traversal handles normally.

## The Union type in `_convert`

`_convert`'s signature accepts:

```python
def _convert(
    graphdef: Union[tf.compat.v1.GraphDef, _GraphDef],
    ...
)
```

- Top-level calls (from `convert()`) pass a real `tf.compat.v1.GraphDef`
- Library function calls (from `_convert_library_function`) pass a `_GraphDef`

The distinction matters in one place:

```python
if hasattr(graphdef, "library"):   # True for real GraphDef, False for _GraphDef
    # recurse into FunctionDefLibrary
```

Real `GraphDef`s have the `.library` attribute; synthetic `_GraphDef`s do not.

## Cognitive load

This Union type forces every reader of `_convert` to know both representations exist. A proposed fix (see [design/refactoring-recommendations.md](../design/refactoring-recommendations.md) item 5): normalise real `GraphDef`s into `_GraphDef` before calling `_convert`, so the function always receives one type. The `.library` recursion would happen in `convert()` before the call, not inside `_convert`.

## `_NodeDef` vs `tf.compat.v1.NodeDef`

Both have `.op`, `.name`, `.input`, `.attr`. `_NodeDef` is a NamedTuple; the proto version is a protobuf message. They are interchangeable for all purposes inside `_convert` because:
- `_OpNode.__init__` accesses `proto.op`, `proto.name`, `proto.input`, `proto.attr` — all present on both
- `_TensorEdge.from_string` uses `node_map[op_name].op` to look up `OpDef` — works on both

## Related pages

- [concepts/library-functions.md](../concepts/library-functions.md)
- [concepts/conversion-phases.md](../concepts/conversion-phases.md)
- [design/refactoring-recommendations.md](../design/refactoring-recommendations.md)
- [entities/_OpNode.md](_OpNode.md)
