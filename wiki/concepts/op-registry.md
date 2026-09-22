# Op Registry

*The op registry maps TensorFlow op names to JAX implementations via a two-level factory pattern.*

**Source:** `tf2jax/_src/ops.py:68–237`

## The registry

```python
_jax_ops: dict[str, Callable[[NodeDef], Callable[..., Any]]]
```

Keys are TensorFlow op names (e.g. `"Conv2D"`, `"MatMul"`, `"Cast"`). Values are **factory functions** — callables that take a `NodeDef` proto and return the actual JAX operation.

This two-level indirection is the core pattern of `ops.py`:

```
op name  →  factory(proto)  →  jax_callable(*arrays)
```

The factory runs **once at build time**, reading static attributes from the proto (strides, padding, dtype, axis, etc.) and closing over them. The returned `jax_callable` runs **at every execution**.

## Simple ops: `_get_jax_op`

For ops with no interesting runtime behaviour, `_get_jax_op` creates a factory that just validates attributes and returns a fixed JAX function:

```python
def _get_jax_op(jax_op, expected_attrs):
    def wrapped(proto):
        _check_attrs(proto, expected_attrs)  # validates no unknown attrs
        return jax_op                        # returns the JAX fn directly
    return wrapped

# Usage:
"Add": _get_jax_op(anp.add, {"T"}),
"MatMul": _get_jax_op(jnp.matmul, {"T", "transpose_a", "transpose_b"}),
```

~120 ops are registered this way.

## Complex ops: custom factories

For ops with meaningful static attributes, the factory reads them and closes over them:

```python
@register_operation("Conv2D")
def _conv2d(proto):
    padding = str(proto.attr["padding"].s, "utf-8")
    strides = tuple(proto.attr["strides"].list.i)
    data_format = str(proto.attr["data_format"].s, "utf-8")
    dimension_numbers = ("NHWC", "HWIO", "NHWC") if data_format == "NHWC" else ...
    strides = xla_utils.get_conv_sequence(strides, ...)

    def _func(lhs, rhs):                    # ← this is what gets called at run time
        feature_group_count = lhs.shape[...] // rhs.shape[...]
        return jax.lax.conv_general_dilated(lhs, rhs, strides, padding,
                                            dimension_numbers, ...)
    return _func
```

The factory pattern means that `padding`, `strides`, and `dimension_numbers` are Python values baked into the closure — they never touch the JAX tracing.

## Registration

```python
def register_operation(op_name):
    return functools.partial(_register, op_name=op_name)

def _register(func, op_name):
    if op_name in _jax_ops and _jax_ops[op_name] != func:
        raise ValueError(f"{op_name} already registered as {_jax_ops[op_name]}")
    _jax_ops[op_name] = func
    return func
```

`register_operation` is a decorator. Duplicate registration of different functions raises; identical re-registration (e.g. from module reloads) is silently accepted.

Multiple op names can share one factory:

```python
@register_operation("FusedBatchNormV3")
@register_operation("FusedBatchNormV2")
@register_operation("FusedBatchNorm")
def _fused_batch_norm(proto): ...
```

## Dispatch

```python
def get_parser(op_name: str) -> Callable:
    return _jax_ops[op_name]   # KeyError if unsupported
```

Called in `_OpNode.__init__`:

```python
self.jax_func = ops.get_parser(proto.op)(proto)
#                               ↑ factory  ↑ invoke factory → get jax callable
```

Unsupported ops raise `ValueError` early (in `_convert`, before any execution), via `ops.get_unsupported_operations`.

## Higher-order ops

Control flow ops (`Case`, `If`, `While`, `MapFn`) implement the `_HigherOrderFunction` dataclass protocol rather than returning a plain callable. See [entities/_HigherOrderFunction.md](../entities/_HigherOrderFunction.md).

## The `numpy_compat` layer

Many simple ops are registered against `anp.xxx` rather than `jnp.xxx`. `anp` is `numpy_compat`, a dispatch layer that routes to `np` when all inputs are static (NumPy arrays) and to `jnp` otherwise. This is necessary because many graph operations (shape computations, constant folding) are evaluated with NumPy arrays at build time. See [concepts/numpy-compat.md](numpy-compat.md).

## Adding a new op

```python
@register_operation("MyNewOp")
def _my_new_op(proto):
    _check_attrs(proto, {"T", "some_attr"})
    some_attr = proto.attr["some_attr"].i
    return lambda x: jnp.something(x, some_attr)
```

## Related pages

- [entities/_OpNode.md](../entities/_OpNode.md)
- [entities/_HigherOrderFunction.md](../entities/_HigherOrderFunction.md)
- [concepts/numpy-compat.md](numpy-compat.md)
- [overview.md](../overview.md)
