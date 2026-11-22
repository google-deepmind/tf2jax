# NumPy Compatibility Layer

*`numpy_compat` (`anp`) provides a dispatch layer that routes to `np` or `jnp` based on input types, enabling constant folding at graph build time.*

**Source:** `tf2jax/_src/numpy_compat.py`

## Why this exists

During graph conversion, some computations run at **build time** with Python scalars and NumPy arrays (e.g. shape computations, `Const` node evaluation, dtype resolution). Others run at **run time** with JAX arrays inside JIT-compiled functions.

If all ops always use `jnp`, then build-time computations become JAX operations, which: (a) can't be used with Python scalars cleanly, (b) create unnecessary JAX traced values during conversion, and (c) break for symbolic dimensions.

The `anp` layer selects the right backend automatically.

## The dispatch mechanism

```python
_NP_LIKES = (np.ndarray, np.number, np.bool_, bool, int, float, complex)

def _is_np(x) -> bool:
    # Special case: 1-D np.array containing poly dims is NOT np-like
    if isinstance(x, np.ndarray) and x.ndim == 1 and any(map(is_poly_dim, x)):
        return False
    return isinstance(x, _NP_LIKES) or is_poly_dim(x)

def _get_np(*args):
    """Returns np if all args are numpy-like, else jnp."""
    return np if all(map(_is_np, args)) else jnp
```

Simple ops are then one-liners:

```python
add = lambda x, y: _get_np(x, y).add(x, y)
multiply = lambda x, y: _get_np(x, y).multiply(x, y)
```

## Polymorphic dimension handling

`is_poly_dim(x)` detects symbolic dimensions from `jax2tf` (type `_DimPolynomial`). These are neither NumPy arrays nor JAX arrays — they represent symbolic shape values. The compatibility layer routes them to JAX:

```python
def is_poly_dim(x) -> bool:
    if jax.__version_info__ >= (0, 4, 30):
        from jax import export
        return export.is_symbolic_dim(x)
    # Fallback: if operator.index() raises TypeError, it's symbolic
    try:
        operator.index(x)
    except TypeError:
        return True
    return False
```

`broadcast_to` has explicit poly-dim handling:

```python
def broadcast_to(arr, shape):
    np_ = jnp if any([is_poly_dim(x) for x in shape]) else _get_np(arr)
    return np_.broadcast_to(arr, shape)
```

## Dtype conversion

Two parallel tables map `tf.DType` → NumPy/JAX dtypes:

```python
tf_to_np_dtypes  = {tf.float32: np.float32, tf.bool: np.bool_, ...}
tf_to_jnp_dtypes = {tf.float32: jnp.float32, tf.bool: jnp.bool_, ...}
```

Selected via `_get_dtypes(*args)` which delegates to `_get_np`. Used by `asarray`, `arange`, `full`, `empty`.

> **Simplification opportunity:** the two tables are structurally identical (same 14 keys). A single table of `(np_dtype, jnp_dtype)` tuples would make divergences explicit. Most entries are identical objects (`np.float32 is jnp.float32`).

## Ops that need special handling

**`gather`** — NumPy and JAX have fundamentally different `take` semantics for batch dims; both implementations are written separately.

**`scatter_nd`** — NumPy uses in-place indexing (`res[key] = updates`); JAX uses `.at[key].set(updates)`.

**`invert_permutation`** — NumPy uses index assignment; JAX uses `.at[arr].set(...)`.

**`squeeze`** — TF squeezes all dims when `axis=()`, but NumPy/JAX squeeze none. Normalised to `None`.

**`divide`** — NumPy `divide` on integer types is true divide; the result dtype must be forced explicitly.

## Related pages

- [concepts/op-registry.md](op-registry.md)
- [concepts/conversion-phases.md](conversion-phases.md)
