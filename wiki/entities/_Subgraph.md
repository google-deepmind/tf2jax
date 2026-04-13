# `_Subgraph`

*A group of nodes whose forward computation is wrapped with `jax.custom_gradient`, used to implement `@tf.custom_gradient` sites.*

**Source:** `tf2jax/_src/tf2jax.py:606–736`

## Structure

```python
class _Subgraph(NamedTuple):
    subgraph: Tuple[_OpNode, ...]      # nodes in forward computation (excl. output_node)
    captures: Tuple[_TensorEdge, ...]  # inputs from outside the subgraph (external)
    outputs: Tuple[_TensorEdge, ...]   # which IdentityN inputs are forward outputs
    output_node: _OpNode               # the IdentityN _OpNode
    grad_fn: _LibraryFunction          # the converted gradient function
```

### Computed properties

```python
@property
def function_inputs(self):
    # IdentityN inputs beyond the forward outputs
    return tuple(self.output_node.inputs[len(self.outputs):])

@property
def unique_inputs(self):
    # function_inputs + unique external captures not already in function_inputs
    return self.function_inputs + unique_captures

# Implements the _GraphNode protocol:
name = output_node.name
inputs = function_inputs + captures
control_inputs = ()
require_rng = any(n.require_rng for n in subgraph)
```

## Forward execution

On each call, `__call__` defines a new `@jax.custom_gradient`-decorated function `fn`:

```python
@jax.custom_gradient
def fn(*args):  # args = unboxed unique_inputs
    # 1. Set up eval_cache with named_args + grad residuals slots
    eval_cache = _EvaluationCache(
        self.subgraph, named_args, self.output_node.inputs + grad_inputs)
    
    # 2. Populate cache with the unique_inputs values
    for inp, val in zip(self.unique_inputs, args):
        eval_cache.outputs[inp.op_name] = val  # (with tuple splicing for multi-output nodes)
    
    # 3. Execute subgraph + output_node in order
    for node in self.subgraph + (self.output_node,):
        eval_cache.outputs[node.name], _ = node(collected_inputs, rng=sub_rng)
        eval_cache.free_inputs(node)
    
    # 4. Define and return the grad function
    def grad_fn(dy):
        captured_inputs = grad_inputs[len(dy):]
        captures = _unbox_named_args(
            [(v.op_name, eval_cache.outputs[v.op_name]) for v in captured_inputs],
            captured_inputs)
        dx = self.grad_fn(*dy, *captures)
        return dx
    
    return eval_cache.outputs[self.output_node.name], grad_fn
```

Note: `@jax.custom_gradient` is re-applied on every call because it's defined inside `__call__`. This is wasteful but correct.

## `rewrite(nodes)`

Removes the subgraph's individual `_OpNode`s from the main node list and inserts `self` at the position of `output_node`:

```python
def rewrite(self, nodes):
    new_nodes = []
    subgraph_map = {v.name: v for v in self.subgraph + (self.output_node,)}
    for node in nodes:
        if node.name in subgraph_map:
            if node == self.output_node:
                new_nodes.append(self)  # replace with _Subgraph
            # else: drop (it's an internal subgraph node)
        else:
            # Rewire control inputs that pointed into the subgraph
            new_nodes.append(node_with_rewired_control_inputs)
    return tuple(new_nodes)
```

After `rewrite`, `_Subgraph` is callable via the same interface as `_OpNode`.

## Grad function calling convention

`self.grad_fn` is the `_LibraryFunction` for the registered gradient function. It is called with:
- `dy`: upstream gradients, one per `output_node.inputs` (length = `num_outputs + num_function_inputs`)
- `captures`: additional inputs to the gradient function beyond `output_node.inputs`, sourced from `eval_cache` (kept alive by the `eval_cache` not freeing their slots due to the `grad_inputs` output list)

## Limitations

1. **Variable assignments** inside the subgraph raise `ValueError` — the eval loop does not handle `updated_params` inside `_Subgraph`.
2. **Overlapping subgraphs** raise `ValueError` — checked in `_extract_subgraphs` after all subgraphs are found.
3. **Missing gradient function** — if `grad_fn is None` (not serialized in the SavedModel), the `IdentityN` is skipped and falls back to automatic differentiation.

## Future: lazy custom gradients

Currently `@jax.custom_gradient` is reinstated on every forward call, and the gradient function TF tracing (`get_concrete_function`) happens eagerly at `convert()` time. A redesign using `jax.custom_vjp` would make the backward truly lazy. See [design/lazy-custom-gradients.md](../design/lazy-custom-gradients.md).

## Related pages

- [concepts/custom-gradients.md](../concepts/custom-gradients.md)
- [entities/_OpNode.md](_OpNode.md)
- [entities/_LibraryFunction.md](_LibraryFunction.md)
- [design/lazy-custom-gradients.md](../design/lazy-custom-gradients.md)
