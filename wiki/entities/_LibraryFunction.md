# `_LibraryFunction`

*A lazily-compiled JAX function corresponding to a TF `FunctionDef`, with metadata about its inputs, outputs, and RNG requirements.*

**Source:** `tf2jax/_src/tf2jax.py:187–217`

## Definition

```python
class _LibraryFunction(NamedTuple):
    fn_builder: _CachedBuilder          # lazy: build JAX fn on first call
    require_rng: bool                   # does this fn (or any inner fn) use random ops?
    input_specs: Optional[Tuple[tf.TensorSpec, ...]]   # specs of all inputs
    output_specs: Optional[Tuple[tf.TensorSpec, ...]]  # specs of all outputs
    # Gradient function only:
    orig_fn_output_specs: Optional[Tuple[tf.TensorSpec, ...]]  # outputs of original fn
    output_is_input: Optional[Tuple[bool]]  # which outputs are unmodified inputs
    variable_input_specs: Optional[Tuple[tf.TensorSpec, ...]]  # VarHandleOp inputs
```

### Properties

```python
@property
def fn(self):
    fn, _ = self.fn_builder()   # triggers lazy compilation on first access
    return fn

@property
def params(self):
    _, params = self.fn_builder()
    return params

def __call__(self, *args, **kwargs):
    if self.params:
        return self.fn(self.params, *args, **kwargs)
    else:
        return self.fn({}, *args, **kwargs)[0]
```

Calling `fn_builder()` the first time triggers `_convert()` on the library function's synthesised `_GraphDef`. The result is cached by `_CachedBuilder` (via `lru_cache`), so subsequent calls are free.

## Two creation paths

**Library function** (from `_convert_library_function`):
- Created for each `FunctionDef` in `FunctionDefLibrary`
- `require_rng` computed from node ops + inner function flags
- `orig_fn_output_specs` is `None`

**Gradient function** (from `_convert_gradient_function`):
- Created for each `@tf.custom_gradient` gradient function
- `orig_fn_output_specs` = output specs of the original decorated function (needed to split `IdentityN` inputs)
- `input_specs` = specs of all inputs to the `IdentityN` node (used by `_Subgraph`)

## Name collision

`ops.py` also defines a `_LibraryFunction` as a `Protocol`:

```python
class _LibraryFunction(Protocol):
    variable_input_specs: Optional[Tuple[tf.TensorSpec, ...]] = None
```

This is a different concept — it describes "anything with `variable_input_specs`" and is used as a type hint in `_HigherOrderFunction.get_additional_inputs`. The name overlap with the concrete `NamedTuple` in `tf2jax.py` is a known confusability issue.

## Related pages

- [concepts/library-functions.md](../concepts/library-functions.md)
- [entities/_Subgraph.md](_Subgraph.md)
- [entities/_HigherOrderFunction.md](_HigherOrderFunction.md)
