# Variable Semantics

*TF's mutable variables are mapped to JAX's functional parameter pattern: variables enter as explicit dict arguments, updated values exit alongside outputs.*

**Source:** `tf2jax/_src/tf2jax.py:343–367` (`Variable`), `1070–1125` (extraction), `1273–1283` (return)

## The problem

TF `tf.Variable` objects are mutable state. JAX arrays are immutable. tf2jax bridges this by treating variables as function parameters that must be explicitly threaded through calls.

## `Variable` class

```python
class Variable(np.ndarray):
    trainable: bool
    name: str
```

A thin NumPy subclass that preserves trainability and name metadata. Variables live in the `params` dict returned by `convert()`. They are NumPy arrays — not JAX arrays — so they can be inspected, sliced, and modified outside of JAX transforms.

## Extraction at build time

During `_convert()`:

1. `variable_map: dict[str, tf.Variable]` is built from `concrete_func.variables` — maps the node name of the variable's placeholder in the graph to the live TF Variable.

2. Variables are uniquified by name (duplicates get `_1`, `_2` suffixes).

3. Each variable is evaluated eagerly: `Variable(v.numpy(), v.trainable, v.name)`.

4. Two bidirectional indexes are built:
   - `var_by_node: dict[str, str]` — graph node name → `params` dict key
   - `node_by_var: dict[str, str]` — `params` dict key → graph node name

## At run time: reading variables

Variables flow in as part of `params`. Inside `jax_func`:

```python
all_params = dict(**params, **constants)
full_inputs = inputs + tuple([all_params[var_by_node.get(v, v)] for v in captured_input_names])
```

`captured_input_names` is the list of graph node names for variable/constant captures. Each is looked up via `var_by_node` to find the key in `all_params`, and the value is prepended to the graph inputs.

## At run time: writing variables

Assign ops (`AssignVariableOp`, `AssignAddVariableOp`, `AssignSubVariableOp`) are registered in `_TF_ASSIGN_OPS`. Their JAX implementations are:

```python
"AssignVariableOp":    lambda var, x: x,
"AssignAddVariableOp": jnp.add,
"AssignSubVariableOp": jnp.subtract,
```

When `_OpNode.__call__` detects one of these, it returns the result as `updated_params`. The evaluation loop writes this back into `eval_cache.outputs` under the variable's node name, so any subsequent reads of that variable in the same call see the updated value.

## Return: new params

At the end of `jax_func`:

```python
new_params = {
    var_name: eval_cache.outputs[node_by_var[var_name]]
    for var_name in params.keys()
}
return collected_outputs, new_params
```

`node_by_var` maps back from the `params` key to the node name whose output in `eval_cache` holds the current (possibly updated) value. Even if a variable was never assigned in this call, its original value is still in `eval_cache` and is returned unchanged.

## Usage pattern

```python
jax_fn, params = tf2jax.convert(tf_model, *sample_inputs)

# Inference (params unchanged)
outputs, params = jax_fn(params, *inputs)

# Training (params updated by assign ops inside the model)
outputs, params = jax_fn(params, *inputs)
# ^ params now reflects any variable updates the model made
```

## Gotcha: `skip_variables_evaluation_inside_tf_tracing`

When `convert()` is called inside a `tf.function` (e.g. when exporting a JAX model back to TF via SavedModel), `tf.inside_function()` returns True and `v.numpy()` cannot be called. Setting `skip_variables_evaluation_inside_tf_tracing=True` in config skips variable evaluation, returning an empty `params` dict. The variables must be supplied by another means.

## Related pages

- [overview.md](../overview.md)
- [concepts/graph-evaluation.md](graph-evaluation.md)
- [entities/convert-api.md](../entities/convert-api.md)
- [entities/Variable.md](../entities/Variable.md)
