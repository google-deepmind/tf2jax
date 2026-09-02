# Refactoring Recommendations

*Structural changes that would reduce cognitive load for readers and developers. Discussed and agreed upon in April 2026.*

**Status:** Proposed, not yet implemented.

---

## 1. Replace the `jax_func` closure with a `_CompiledGraph` class

**The problem:** `_convert()` defines a `jax_func` closure that implicitly captures 12 build-time variables. Readers must mentally inventory all 12 to understand what the function does. The data model is invisible.

**The fix:** A class whose `__init__` accepts the 12 variables as named parameters and stores them as typed `self.` fields, with `__call__` containing the current closure body.

```python
class _CompiledGraph:
    def __init__(self, nodes, input_names, input_path_to_specs,
                 output_args, structured_outputs, flat_output_specs,
                 captured_input_names, constants, var_by_node, node_by_var,
                 signature, num_rng_required):
        ...
    
    def __call__(self, params, *args, rng=None, **kwargs):
        ...  # current jax_func body, unchanged
```

**Benefits:**
- Build time (`__init__`) and run time (`__call__`) are explicitly separated
- All dependencies are declared and typed, not inferred from captures
- The class is independently testable
- `_convert()` becomes shorter and more readable

---

## 2. Define a `_GraphNode` Protocol

**The problem:** After `_extract_subgraphs`, the node list is silently `list[_OpNode | _Subgraph]`. Both implement the same interface but there is no formal declaration. Readers don't know that `node` can be a `_Subgraph`.

**The fix:**

```python
class _GraphNode(Protocol):
    name: str
    inputs: Tuple[_TensorEdge, ...]
    control_inputs: Tuple[_TensorEdge, ...]
    require_rng: bool
    def __call__(self, named_args: Sequence[Tuple[str, Any]], *, rng) -> ...: ...
```

Type the node list as `list[_GraphNode]` everywhere.

---

## 3. Extract custom gradient code to its own module

**The problem:** ~350 lines of custom gradient machinery (`_filter_nodes`, `_contains_custom_gradient`, `_get_consumers_and_producers_fns`, `_extract_subgraphs`, `_convert_all_gradient_functions`, `_convert_gradient_function`, `_Subgraph`) are interspersed throughout `tf2jax.py`. When reading `_convert`, it's unclear which parts are core and which are the custom gradient extension.

**The fix:** Move to `tf2jax/_src/_custom_gradient.py`. The hook in `_convert` stays as-is — one import, one call. Core conversion becomes readable without knowing anything about custom gradients.

---

## 4. Resolve the `_LibraryFunction` name collision

**The problem:** `_LibraryFunction` exists in both `tf2jax.py` (a concrete `NamedTuple` wrapping a built JAX function) and `ops.py` (a `Protocol` for type-checking `variable_input_specs` access). Same name, different concepts.

**The fix:** Rename the `ops.py` Protocol to `_HasVariableInputSpecs`. Or inline the single attribute access and delete the Protocol entirely.

---

## 5. Normalise to `_GraphDef` before `_convert`

**The problem:** `_convert` accepts `Union[tf.compat.v1.GraphDef, _GraphDef]`. The union propagates through the whole function. The `hasattr(graphdef, "library")` check distinguishes the two. Readers must know both representations exist.

**The fix:** Normalise to `_GraphDef` before calling `_convert`. `_convert_library_function` already assembles `_GraphDef` from `FunctionDef`. Do the same for top-level `GraphDef`s so `_convert` always receives one type.

---

## 6. Pass `outputs_dict` directly to nodes (remove `named_args` ceremony)

**The problem:** The evaluation loop wraps values as `list[(name, value)]` → `_unbox_named_args` validates names and strips them → plain values reach the JAX callable. The name-checking is defensive but redundant (topological sort guarantees correctness).

**The fix:**

```python
# In _OpNode.__call__(self, outputs_dict, *, rng):
args = tuple(
    outputs_dict[inp.op_name][inp.idx]
    if isinstance(outputs_dict[inp.op_name], (list, tuple))
    else outputs_dict[inp.op_name]
    for inp in self.inputs
)
```

Removes `_unbox_named_args` entirely and makes the data flow — "look up each input by name" — directly visible.

---

## 7. Drop TF1 Session code path

**The problem:** Lines `1101–1113` contain a dead `else: # We are in TF1 mode` branch using `tf.compat.v1.Session`. In TF2, `tf.executing_eagerly()` is always True, so this is unreachable.

**The fix:** Delete the `else` branch and remove the associated `tf.compat.v1` imports.

---

## Priority order

| # | Change | Risk | Leverage |
|---|--------|------|---------|
| 3 | Extract custom gradient module | Medium | Highest — unblocks reading core |
| 1 | `_CompiledGraph` class | Medium | High — makes data model visible |
| 2 | `_GraphNode` Protocol | Low | Medium — documents existing contract |
| 7 | Drop TF1 code | Low | Low — dead code removal |
| 4 | Rename `_LibraryFunction` collision | Low | Low — terminology |
| 5 | Normalise to `_GraphDef` | Medium | Medium — removes Union type |
| 6 | Remove `named_args` ceremony | Low | Medium — evaluation loop clarity |

## Related pages

- [design/lazy-custom-gradients.md](lazy-custom-gradients.md)
- [concepts/conversion-phases.md](../concepts/conversion-phases.md)
- [concepts/custom-gradients.md](../concepts/custom-gradients.md)
