# `Variable`

*NumPy `ndarray` subclass that carries TF variable metadata (name, trainability) through the conversion boundary.*

**Source:** `tf2jax/_src/tf2jax.py:343–367`

## Definition

```python
class Variable(np.ndarray):
    trainable: bool
    name: str
    
    def __new__(cls, arr, trainable, name):
        obj = np.asarray(arr).view(cls)
        obj.trainable = trainable
        obj.name = name
        return obj
    
    def assign(self, arr):
        return Variable(arr, trainable=self.trainable, name=self.name)
    
    def __array_finalize__(self, obj):
        self.trainable = getattr(obj, "trainable", None)
        self.name = getattr(obj, "name", None)
```

## Why NumPy, not JAX?

`Variable` subclasses `np.ndarray` rather than `jax.Array` because:
1. Variables are extracted outside of JIT by calling `tf_var.numpy()` at build time
2. Users may want to inspect, modify, or checkpoint variables as plain NumPy before passing them to JAX functions
3. NumPy arrays are accepted everywhere JAX arrays are via implicit conversion

## Lifecycle

1. **Created** in `_convert()` from `tf.Variable.numpy()` values
2. **Returned** in the `params` dict from `convert()`
3. **Passed** as the first argument to `jax_fn(params, *inputs)`
4. **Updated** — if the model contains assign ops, `jax_fn` returns `new_params` with updated values
5. **Inspected** — `params["dense/kernel"].trainable`, `params["dense/kernel"].name`

## `params` dict conventions

Keys are constructed from `tf.Variable.name` with `_parse_input` applied (strips the `:0` suffix) and uniquified if needed:

```
"dense/kernel"    → params["dense/kernel"]
"dense/bias"      → params["dense/bias"]
"dense/kernel:0"  → variable.name (original TF name)
```

## Related pages

- [concepts/variable-semantics.md](../concepts/variable-semantics.md)
- [entities/convert-api.md](convert-api.md)
