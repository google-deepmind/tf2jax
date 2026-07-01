# `_HigherOrderFunction`

*Dataclass protocol for ops whose JAX implementation requires inner functions (control flow bodies, branch functions).*

**Source:** `tf2jax/_src/ops.py:245–261`

## Definition

```python
@dataclasses.dataclass
class _HigherOrderFunction:
    inner_fn_names: Mapping[str, str]   # kwarg name → library function name

    def get_inner_functions(self, library_functions):
        return {k: library_functions[v] for k, v in self.inner_fn_names.items()}

    def get_additional_inputs(self, **inner_functions):
        # Override in subclasses if the op needs extra input edges
        # (e.g. VarHandleOp inputs for functions with checkpoint variables)
        return ()
```

## How it integrates with `_OpNode`

```python
# In _OpNode.__init__:
if isinstance(self.jax_func, ops._HigherOrderFunction):
    self.inner_fns = self.jax_func.get_inner_functions(library)
    for input_name in self.jax_func.get_additional_inputs(**self.inner_fns):
        inputs.append(_TensorEdge.from_string(input_name, node_map))

# In _OpNode.__call__:
extras_dict.update(self.inner_fns)
outputs = self.jax_func(*unboxed_args, **extras_dict)
```

The op factory returns a `_HigherOrderFunction` instance (not a plain callable). `_OpNode` detects this via `isinstance`, fetches the inner `_LibraryFunction`s, and passes them as keyword arguments at run time.

## Implemented by

**`_CaseOp`** — `tf.switch_case` / `tf.raw_ops.Case`:
```python
class _CaseOp(_HigherOrderFunction):
    def __call__(self, branch_index, *operand, **branch_fns):
        branches = [lambda args: fn(*args) for _, fn in sorted(branch_fns.items())]
        return jax.lax.switch(branch_index, branches=branches, operand=operand)
```

**`_WhileOp`** — `tf.while_loop`: maps to `jax.lax.while_loop(cond_fn, body_fn, init_val)`

**`_IfOp`** — `tf.cond`: maps to `jax.lax.cond(pred, true_fn, false_fn, *operands)`

**`_MapFnOp`** — `tf.map_fn`: maps to `jax.vmap` or `jax.lax.map`

**`_XlaReduceWindow`** — uses `computation` inner function for custom reduce windows

## `get_additional_inputs`

Some higher-order functions contain `VarHandleOp` nodes that reference variables from checkpoints. These variables need to be threaded in as explicit inputs. `get_additional_inputs` returns their names so `_OpNode` can add them as extra `_TensorEdge`s.

## Related pages

- [entities/_OpNode.md](_OpNode.md)
- [entities/_LibraryFunction.md](_LibraryFunction.md)
- [concepts/library-functions.md](../concepts/library-functions.md)
- [concepts/op-registry.md](../concepts/op-registry.md)
