# Public API: `convert`, `convert_functional`, `convert_from_restored`

*Three entry points covering the main use cases: general conversion, stateless functions, and SavedModel restoration.*

**Source:** `tf2jax/_src/tf2jax.py:461–570` (`convert`), `394–399` (`convert_functional`), `402–425` (`convert_from_restored`)

---

## `convert(tf_func, *args, **kwargs)`

The primary entry point.

```python
jax_fn, params = tf2jax.convert(tf_model, *sample_inputs)
```

**Returns:** `(AnnotatedFunction, dict[str, Variable])`

The `AnnotatedFunction` wraps a `jax_func(params, *args, rng=None, **kwargs)` callable that returns `(outputs, new_params)`.

**What it does:**
1. Calls `tf_func.get_concrete_function(*args, **kwargs)` to get a `ConcreteFunction`
2. Extracts `GraphDef`, `captures` (variables + constants), input/output specs
3. Optionally converts custom gradient functions (`_convert_all_gradient_functions`)
4. Calls `_convert(graphdef, ...)` to build the JAX function

**Args** can be `tf.TensorSpec` (abstract), actual tensors/arrays, or JAX tracers (they are automatically converted to `tf.TensorSpec` via `_maybe_tracer_to_tf_spec`).

---

## `convert_functional(tf_func, *args, **kwargs)`

For functions with no `tf.Variable` captures.

```python
jax_fn = tf2jax.convert_functional(tf_model, *sample_inputs)
outputs = jax_fn(*real_inputs)
```

**Returns:** `AnnotatedFunction` whose `__call__` takes just `(*args, **kwargs)` — no `params` argument, no `new_params` return.

Raises if the function turns out to have captured variables.

---

## `convert_from_restored(tf_func)`

For functions loaded from a `tf.saved_model.load()` (a `RestoredFunction`). Avoids specifying sample inputs explicitly.

```python
model = tf.saved_model.load("/path/to/model")
jax_fn, params = tf2jax.convert_from_restored(model.signatures["serving_default"])
```

**How it determines the concrete function:**
- If `tf_func.input_signature` is set: uses it as `args`
- If not: checks `tf_func.concrete_functions` — if there's exactly one, uses its `structured_input_signature`. If there are multiple, raises with a helpful error showing the options.

Delegates to `convert()` after resolving args.

`convert_functional_from_restored` is the stateless variant.

---

## `AnnotatedFunction`

```python
class AnnotatedFunction:
    fn: Callable               # the actual jax_func
    structured_inputs: SpecTree
    structured_outputs: SpecTree
    __signature__: inspect.Signature   # mirrors the original TF function's signature
    __name__: str
    __doc__: str
    
    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
```

Preserves the original function's signature for debugging and introspection tools. The `structured_inputs`/`structured_outputs` fields carry the `tf.TensorSpec` tree describing the expected I/O.

---

## Config flags affecting conversion

| Flag | Default | Effect |
|------|---------|--------|
| `convert_custom_gradient` | `True` | Convert `@tf.custom_gradient` sites |
| `strict_shape_check` | `True` | Validate input shapes at run time |
| `strict_dtype_check` | `False` | Validate input dtypes at run time |
| `infer_relu_from_jax2tf` | `True` | Detect `max(x,0)` patterns from jax2tf output |
| `infer_cumulative_reduction_from_jax2tf` | `True` | Detect cumsum/cumprod patterns |
| `skip_variables_evaluation_inside_tf_tracing` | `False` | Needed when calling `convert` inside `tf.function` |

Use `tf2jax.override_config(name, value)` as a context manager, or `tf2jax.update_config(name, value)` for permanent changes.

---

## TF-Hub compatibility

TF-Hub modules may use `$` in tensor names and may have `VariableSpec` objects mixed into structured specs. Three workarounds exist:

- `_fix_tfhub_specs`: rewrites tensor names in structured specs to match the concrete function's tensor names
- `_UNUSED_INPUT`: sentinel for tensor arguments that were unused in the traced function
- `k.replace("$", "___")` substitution in parameter names (because `$` is not a valid Python identifier)

## Related pages

- [overview.md](../overview.md)
- [concepts/conversion-phases.md](../concepts/conversion-phases.md)
- [concepts/variable-semantics.md](../concepts/variable-semantics.md)
- [entities/_convert.md](_convert.md)
