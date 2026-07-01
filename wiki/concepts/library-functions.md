# Library Functions

*TF's `FunctionDefLibrary` stores reusable sub-functions (control flow bodies, gradient functions). tf2jax converts these recursively and caches each one.*

**Source:** `tf2jax/_src/tf2jax.py:167–218` (`_CachedBuilder`, `_LibraryFunction`), `1288–1374` (`_convert_library_function`), `904–926` (`_get_function_protos`)

## Background: what is `FunctionDefLibrary`?

Every `tf.GraphDef` has an embedded `library` of `FunctionDef` protos. These are the inner functions used by:
- `tf.cond`, `tf.switch_case` → each branch is a library function
- `tf.while_loop` → body and condition are library functions
- `tf.map_fn` → the mapped function is a library function
- `@tf.custom_gradient` → the gradient function is a library function

Library functions can reference each other (e.g. a while body may call a helper). tf2jax processes them in dependency order.

## `_get_function_protos`

```python
def _get_function_protos(graphdef) -> Iterator[(str, FunctionDef)]:
```

A recursive generator that yields `(name, proto)` pairs in dependency order — a function's callees are yielded before the function itself. This ensures the `library` dict is always populated before any function that references it.

## `_convert_library_function`

Converts a `FunctionDef` proto into a `_LibraryFunction`. Steps:

1. **Synthesise a `_GraphDef`** from the `FunctionDef`:
   - Each `input_arg` becomes a `Placeholder` node
   - Each `node_def` is included as-is
   - Each `output_arg` becomes an `Identity` node

2. **Handle `VarHandleOp`** — variables embedded in library functions (from checkpoints) are surfaced as additional positional parameters.

3. **Determine `require_rng`** — check if any node op is in `_TF_RANDOM_OPS` or if any inner function requires RNG.

4. **Create a `_CachedBuilder`** that lazily calls `_convert()` on the synthesised `_GraphDef`.

## `_CachedBuilder`

```python
class _CachedBuilder:
    def __init__(self, builder_fn, cached_configs):
        self._builder_fn = functools.lru_cache(builder_fn)
        self._cached_configs = cached_configs
    
    def __call__(self):
        with config.override_configs(self._cached_configs):
            return self._builder_fn()
```

**Purpose:** call-once semantics with frozen config. `functools.lru_cache` on a zero-argument function means the builder runs exactly once; subsequent calls return the cached result. The config at build time is captured via `config.copy_config()` and restored during execution, so library functions compiled under one config setting don't pick up later changes.

> **Clarity note:** Using `lru_cache` on a zero-arg function to get "call once" semantics is idiomatic but unintuitive. A lazy-init pattern (`if self._result is None: ...`) would be more explicit.

## `_LibraryFunction`

```python
class _LibraryFunction(NamedTuple):
    fn_builder: _CachedBuilder
    require_rng: bool
    input_specs: Optional[Tuple[tf.TensorSpec, ...]]
    output_specs: Optional[Tuple[tf.TensorSpec, ...]]
    orig_fn_output_specs: Optional[Tuple[tf.TensorSpec, ...]]  # gradient functions only
    output_is_input: Optional[Tuple[bool]]
    variable_input_specs: Optional[Tuple[tf.TensorSpec, ...]]
    
    @property
    def fn(self): fn, _ = self.fn_builder(); return fn
    @property
    def params(self): _, p = self.fn_builder(); return p
    
    def __call__(self, *args, **kwargs):
        if self.params:
            return self.fn(self.params, *args, **kwargs)
        else:
            return self.fn({}, *args, **kwargs)[0]
```

When `__call__` is first invoked, `fn_builder()` triggers the lazy `_convert()` and caches the result. The `params` check handles the rare case where a library function captures its own variables (e.g. from a checkpoint-embedded `VarHandleOp`).

## Usage in `_OpNode` (higher-order ops)

Higher-order function ops (`Case`, `While`, `If`, `MapFn`) implement `_HigherOrderFunction`, which has:

```python
def get_inner_functions(self, library_functions):
    return {k: library_functions[v] for k, v in self.inner_fn_names.items()}
```

`_OpNode.__init__` calls this and stores the result in `self.inner_fns`. At run time, inner functions are passed as keyword arguments to the op's `jax_func`:

```python
outputs = self.jax_func(*unboxed_args, **self.inner_fns)
```

The op's factory then uses them:

```python
@register_operation("While")
def _while(proto):
    cond_fn_name = proto.attr["cond"].func.name
    body_fn_name = proto.attr["body"].func.name
    
    def _func(*args, cond_fn=None, body_fn=None):
        return jax.lax.while_loop(cond_fn, body_fn, args)
    
    return _While(dict(cond_fn=cond_fn_name, body_fn=body_fn_name))
```

## Name collision note

`ops.py` defines a separate `_LibraryFunction` as a `Protocol` used for type-checking `variable_input_specs` access in `_HigherOrderFunction.get_additional_inputs`. This is a different concept from the `_LibraryFunction` `NamedTuple` in `tf2jax.py`. The shared name is confusing — the Protocol one describes "something that has `variable_input_specs`".

## Related pages

- [entities/_LibraryFunction.md](../entities/_LibraryFunction.md)
- [concepts/custom-gradients.md](custom-gradients.md)
- [entities/_HigherOrderFunction.md](../entities/_HigherOrderFunction.md)
- [concepts/conversion-phases.md](conversion-phases.md)
