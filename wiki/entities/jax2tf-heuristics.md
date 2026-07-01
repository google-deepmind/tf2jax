# jax2tf Pattern-Matching Heuristics

*Two post-processing passes that detect and replace patterns introduced by `jax2tf.convert`, restoring cleaner JAX equivalents.*

**Source:** `tf2jax/_src/tf2jax.py:877–899` (`_infer_relu_from_jax2tf`), `tf2jax/_src/ops.py:2708–2773` (cumulative reduction, inside `_XlaReduceWindow`)

---

## `_infer_relu_from_jax2tf`

**What it detects:** `Maximum(x, Cast(Const(0)))` — jax2tf's lowering of `jax.nn.relu`.

**How it works:**

```python
def _infer_relu_from_jax2tf(nodes):
    node_map = {n.name: n for n in nodes}
    for node in nodes:
        if node.op == "Maximum" and "jax2tf" in node.name and "_relu_" in node.name:
            cast_or_const_arg = node_map[node.inputs[1].op_name]
            if cast_or_const_arg.op == "Cast":
                const_arg = node_map[cast_or_const_arg.inputs[0].op_name]
            else:
                const_arg = cast_or_const_arg
            
            if const_arg.op == "Const" and const_arg((), rng=None)[0].tolist() == 0:
                node.op = "Relu"
                node.inputs = node.inputs[:1]
                node.jax_func = ops.get_parser("Relu")(_NodeDef("Relu", node.name, (), {}))
```

Matching uses both the node op type (`Maximum`) and the node name (`"jax2tf"` and `"_relu_"` substrings), so it only fires on graphs produced by jax2tf, not on arbitrary user-written `max(x, 0)` patterns.

**Why it matters:** `jax.nn.relu` has a well-defined gradient (subgradient at 0). `jnp.maximum(x, 0)` computes the same values but JAX differentiates it via its general maximum rule, which can differ at exactly 0. Using `relu` directly also produces a cleaner JIT-compiled program.

**Side effect:** This is the only place in the codebase where `_OpNode` objects are **mutated after construction** — `.op`, `.inputs`, and `.jax_func` are all overwritten in-place. The `Cast` and `Const` nodes that fed the replaced zero argument become orphaned (their reference counts drop to zero and `_EvaluationCache` frees them).

**Controlled by:** `infer_relu_from_jax2tf` config flag (default `True`).

---

## Cumulative reduction inference (inside `_XlaReduceWindow`)

**What it detects:** `reduce_window` calls with window patterns matching `cumsum`, `cumprod`, `cummax`, or `cummin`.

**Background:** jax2tf lowers `jax.lax.cumsum` as `jax.lax.reduce_window` with a specific window/padding pattern. The resulting `XlaReduceWindow` op in the TF graph is correct but opaque — JAX treats it as a general reduce_window and cannot recognize it as a cumulative reduction for VJP purposes.

**How it works:** Inside the `_XlaReduceWindow` op's runtime function, after converting the abstract `computation` function to a JAX primitive:

```python
def infer_cumulative_reduction():
    # Check: window spans the full reduce axis, stride 1, no dilation
    reduce_axis = np.argmax(window_dimensions)
    reduce_dim = operand.shape[reduce_axis]
    dims = [1] * ndims; dims[reduce_axis] = reduce_dim
    if not (window_dimensions == dims and window_strides == [1]*ndims ...):
        return (None, None, None)
    
    # Check padding: [reduce_dim-1, 0] → forward; [0, reduce_dim-1] → reverse
    if padding == normal_padding:  reverse = False
    elif padding == reverse_padding: reverse = True
    else: return (None, None, None)
    
    # Match computation to a known cumulative reduction
    if computation_fn is jax.lax.add and init_value == 0:
        return (jax.lax.cumsum, reduce_axis, reverse)
    elif computation_fn is jax.lax.mul and init_value == 1:
        return (jax.lax.cumprod, reduce_axis, reverse)
    elif computation_fn is jax.lax.max and init_value == _get_max_identity(dtype):
        return (jax.lax.cummax, reduce_axis, reverse)
    elif computation_fn is jax.lax.min and init_value == _get_min_identity(dtype):
        return (jax.lax.cummin, reduce_axis, reverse)
```

If a match is found and `infer_cumulative_reduction_from_jax2tf` is set, the function returns `jax.lax.cumsum(operand, ...)` directly instead of `jax.lax.reduce_window(...)`.

**Why it matters:** `jax.lax.cumsum` has an efficient parallel scan VJP. A general `reduce_window` does not. Recognising the pattern restores the efficient gradient.

**Controlled by:** `infer_cumulative_reduction_from_jax2tf` config flag (default `True`).

---

## Comparison

| | `_infer_relu_from_jax2tf` | Cumulative reduction inference |
|--|--|--|
| Applied | Build time (after node construction) | Run time (inside op execution) |
| Mechanism | Mutates `_OpNode` in-place | Branch in `_XlaReduceWindow.__call__` |
| Scope | Entire node list | Individual `XlaReduceWindow` call |
| Detection signal | Node name + op type + constant value | Window shape + padding + computation primitive |

## Related pages

- [concepts/jax2tf-roundtrip.md](../concepts/jax2tf-roundtrip.md)
- [entities/_OpNode.md](_OpNode.md)
- [concepts/graph-evaluation.md](../concepts/graph-evaluation.md)
