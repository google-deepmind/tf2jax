# Custom Gradients

*tf2jax supports `@tf.custom_gradient` by detecting TF's `IdentityN` marker nodes, extracting the surrounding subgraph, and re-wrapping it with `jax.custom_gradient`.*

**Source:** `tf2jax/_src/tf2jax.py:739–874` (subgraph extraction), `606–711` (`_Subgraph`), `1395–1532` (gradient function conversion)

## How TF implements `@tf.custom_gradient`

When TF traces a function decorated with `@tf.custom_gradient`, it inserts an `IdentityN` node into the graph. This node:
- Passes its inputs through unchanged (identity)
- Has a `_gradient_op_type` attribute naming the registered gradient function
- Receives as inputs: `[fn_outputs..., fn_inputs...]` — both the outputs of the decorated function and its inputs (so the gradient function can access them)

tf2jax detects these `IdentityN` nodes and reconstructs the original intent.

## Pipeline

### 1. Gradient function conversion (in `convert()`)

```python
if config.get_config("convert_custom_gradient"):
    _convert_all_gradient_functions(graph, library)
```

`_convert_all_gradient_functions` finds all `IdentityN` nodes in the graph (including in library functions), and for each calls `_convert_gradient_function(proto, graph, library)`.

`_convert_gradient_function`:
1. Looks up the registered gradient function: `tf_ops.gradient_registry.lookup(grad_fn_name)`
2. Wraps it in a `@tf.function` and calls `get_concrete_function(*input_specs)` — **this is the expensive TF trace**
3. Extracts metadata: `external_capture_specs`, `internal_capture_names`, output specs
4. Creates a `_LibraryFunction` with a `_CachedBuilder` whose builder calls `_convert()` on the gradient graph — **JAX conversion is lazy**

> **Eagerness issue:** Step 2 (TF tracing) runs at `convert()` time even if the user never requests gradients. See [design/lazy-custom-gradients.md](../design/lazy-custom-gradients.md) for the design to fix this.

### 2. Subgraph extraction (in `_convert()`)

```python
if config.get_config("convert_custom_gradient"):
    subgraphs = _extract_subgraphs(graphdef, nodes, library)
    for _, subgraph in subgraphs.items():
        nodes = subgraph.rewrite(nodes)
```

`_extract_subgraphs` builds `_Subgraph` objects. For each `IdentityN`:

- `num_outputs = len(grad_fn.orig_fn_output_specs)` — split point between forward outputs and function inputs in the `IdentityN` inputs list
- Backward-reachability from `output_node.inputs[:num_outputs]` identifies which nodes form the subgraph
- `external_captures` = inputs to the subgraph that come from outside it
- Side-effect nodes (e.g. shape assert nodes from jax2tf) are absorbed into the subgraph

After extraction, overlapping subgraphs are rejected (not yet supported).

`subgraph.rewrite(nodes)` removes the subgraph's individual `_OpNode`s from the main node list and replaces them with the `_Subgraph` object at the position of `output_node`.

### 3. `_Subgraph.__call__` (at run time)

Every time the subgraph is reached in the evaluation loop:

```python
@jax.custom_gradient
def fn(*args):
    # 1. Run the subgraph nodes forward (same as _OpNode evaluation)
    ...
    
    def grad_fn(dy):
        # 2. Called only during backpropagation
        # dy has length len(output_node.inputs)
        # captured_inputs are extra inputs to the gradient function
        dx = self.grad_fn(*dy, *captures)
        return dx      # gradients w.r.t. unique_inputs
    
    return outputs, grad_fn

outputs = fn(*unboxed_args)
```

`self.grad_fn` is the `_LibraryFunction` for the gradient function. Calling it triggers `_CachedBuilder`, which triggers `_convert()` on the gradient graph — lazily, on first backprop call.

`@jax.custom_gradient` is re-applied on **every call** to `_Subgraph.__call__` because it's defined inside the method body. This is inefficient; see [design/lazy-custom-gradients.md](../design/lazy-custom-gradients.md).

## Gradient function calling convention

The gradient function (registered via `@tf.custom_gradient`) receives:
- The upstream gradients w.r.t. each of the `IdentityN` inputs (`dy`, same length as `output_node.inputs`)
- Any captured inputs that aren't part of the `IdentityN` node (`captures = grad_inputs[len(dy):]`)

It returns one gradient per element of `unique_inputs`.

## Limitations

- Variable assignments inside custom gradient subgraphs are not supported (raises `ValueError`)
- Overlapping subgraphs (two custom gradients sharing computation) are not supported
- If the gradient function is not serialized in the SavedModel, `grad_fn` is set to `None` and the subgraph falls back to the standard automatic differentiation path

## Config flags

- `convert_custom_gradient` (default `True`): master switch. If False, `IdentityN` nodes are treated as plain identity ops.
- `raise_on_prevent_gradient` (default `True`): if a `PreventGradient` op (inserted by `jax2tf.convert(with_gradient=False)`) is hit during differentiation, raise an error.

## Related pages

- [entities/_Subgraph.md](../entities/_Subgraph.md)
- [entities/_LibraryFunction.md](../entities/_LibraryFunction.md)
- [concepts/library-functions.md](library-functions.md)
- [design/lazy-custom-gradients.md](../design/lazy-custom-gradients.md)
