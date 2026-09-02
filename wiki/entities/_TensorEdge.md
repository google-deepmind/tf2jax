# `_TensorEdge`

*Represents a directed edge in the computation graph: a reference to a specific output tensor of a node.*

**Source:** `tf2jax/_src/tf2jax.py:106–157`

## Definition

```python
class _TensorEdge(NamedTuple):
    op_name: str        # the producing node's name (cache key)
    idx: int = 0        # which output of that node (0 for single-output ops)
    is_control: bool = False   # True for control dependencies (^NodeName)
```

## Construction from proto string

TF `NodeDef.input` entries are strings in one of these formats:

| Format | Meaning |
|--------|---------|
| `"NodeName"` | output 0 of NodeName |
| `"NodeName:2"` | output 2 of NodeName |
| `"NodeName:output_name:0"` | named output (multi-output ops) |
| `"^NodeName"` | control dependency on NodeName |

`_TensorEdge.from_string(op_str, node_map)` parses these. For the three-part format, it looks up the `OpDef` in the TF op registry to resolve `output_name` to an integer index.

## Usage

`_TensorEdge` objects are stored in:
- `_OpNode.inputs` and `_OpNode.control_inputs`
- `_Subgraph.captures`, `_Subgraph.outputs`
- `output_args` list in `_convert` (which outputs to read from `eval_cache`)

The evaluation loop collects inputs for a node as:

```python
collected_inputs = [
    (v.op_name, eval_cache.outputs[v.op_name]) for v in node.inputs
]
```

`op_name` is the cache key; `idx` is used in `_unbox_named_args` to index into tuple outputs.

## Related pages

- [entities/_OpNode.md](_OpNode.md)
- [concepts/graph-evaluation.md](../concepts/graph-evaluation.md)
