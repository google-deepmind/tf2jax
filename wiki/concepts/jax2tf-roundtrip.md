# jax2tf Round-Trip

*tf2jax can convert functions that were themselves produced by `jax2tf.convert`, enabling JAX→TF→JAX round-trips. This use case drives several otherwise-puzzling design choices.*

**Source:** `tf2jax/_src/tf2jax.py:877–899` (relu heuristic), `ops.py:2724–2763` (cumulative reduction heuristic), `numpy_compat.py:71–89` (poly dim handling)

## The round-trip pattern

```python
# JAX → TF
tf_fn = jax2tf.convert(jax_model, polymorphic_shapes=["(b, d)"])

# TF → JAX (via tf2jax)
jax_fn, params = tf2jax.convert(tf_fn, tf.TensorSpec((None, 4)))
```

This is used to:
- Load JAX models serialized as TF SavedModels into a JAX environment
- Fine-tune or compose JAX models that went through a TF checkpoint
- Test round-trip numerical fidelity

## What jax2tf does to the graph

`jax2tf.convert` produces TF graphs that contain artifacts from the lowering process. These are patterns that tf2jax must recognise and handle correctly:

### 1. `max(x, 0)` instead of `relu`

jax2tf lowers `jax.nn.relu` as `tf.maximum(x, tf.cast(0, x.dtype))` rather than `tf.nn.relu`. The resulting graph has a `Maximum` node fed by a `Cast`-of-`Const` node. If tf2jax naively converts this, it creates a JAX graph that does the same thing — but loses the opportunity to use `jax.nn.relu` directly, which is more gradient-friendly.

`_infer_relu_from_jax2tf` detects this pattern. See [entities/jax2tf-heuristics.md](../entities/jax2tf-heuristics.md).

### 2. `reduce_window` instead of `cumsum`/`cumprod`

jax2tf lowers `jax.lax.cumsum` as `jax.lax.reduce_window` (a sliding-window reduction). The XlaReduceWindow op that results is functionally correct but opaque — JAX won't recognise it as a cumulative reduction for gradient purposes.

`infer_cumulative_reduction_from_jax2tf` detects this pattern inside `_XlaReduceWindow`. See [entities/jax2tf-heuristics.md](../entities/jax2tf-heuristics.md).

### 3. Polymorphic shapes

`jax2tf.convert` with `polymorphic_shapes` produces graphs where some dimensions are symbolic (`_DimPolynomial`) rather than concrete integers. tf2jax must propagate these through shape computations rather than trying to evaluate them as NumPy scalars.

The `numpy_compat` layer handles this: `is_poly_dim(x)` detects symbolic dimensions and routes to `jnp` instead of `np`. `broadcast_to` and `_fix_jax_poly_shape` have explicit poly-dim handling. See [concepts/numpy-compat.md](numpy-compat.md).

### 4. `PreventGradient` ops

`jax2tf.convert(with_gradient=False)` inserts `tf.raw_ops.PreventGradient` ops to block differentiation of the converted function. If the converted function is later differentiated in JAX, tf2jax raises an error unless `raise_on_prevent_gradient=False` is set in config.

### 5. Shape assertion nodes

jax2tf inserts shape check assert nodes as side effects of certain operations. `_extract_subgraphs` explicitly handles these: they are detected as "side effect nodes" (nodes that depend on the subgraph but don't feed into its outputs) and absorbed into the subgraph rather than left dangling in the main graph.

### 6. `XlaCallModule` (native serialization)

Newer versions of jax2tf with `native_serialization=True` produce `XlaCallModule` ops containing StableHLO bytecode instead of a full TF graph. These require separate handling in `tf2jax/experimental/ops.py`. Platform-specific — a module lowered for TPU may not run on CPU.

## Config flags specific to this use case

| Flag | Default | Purpose |
|------|---------|---------|
| `infer_relu_from_jax2tf` | `True` | Detect `max(x,0)` and replace with `relu` |
| `infer_cumulative_reduction_from_jax2tf` | `True` | Detect `reduce_window` cumsum/cumprod patterns |
| `raise_on_prevent_gradient` | `True` | Error on differentiation barriers |
| `xlacallmodule_strict_checks` | `True` | Validate platform for native serialization |

## Related pages

- [entities/jax2tf-heuristics.md](../entities/jax2tf-heuristics.md)
- [concepts/numpy-compat.md](numpy-compat.md)
- [concepts/custom-gradients.md](custom-gradients.md)
