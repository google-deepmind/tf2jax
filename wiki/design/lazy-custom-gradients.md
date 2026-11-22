# Lazy Custom Gradient Evaluation

*Design for deferring gradient function setup to the first backward pass, avoiding TF tracing and custom_gradient overhead on forward-only calls.*

**Status:** Proposed, not yet implemented. Discussed April 2026.

---

## The problem

Currently, on every call to `convert()`:

1. `_convert_all_gradient_functions(graph, library)` is called **eagerly**, which for each `@tf.custom_gradient` site:
   - Looks up the gradient function in TF's registry
   - Calls `get_concrete_function(*input_specs)` → **full TF graph trace**
   - Extracts metadata from the concrete function

2. In `_Subgraph.__call__`, `@jax.custom_gradient` is **re-applied on every forward call**, creating a new decorated function each time.

A model used purely for inference pays the full cost of gradient function setup.

**What is already lazy:**
- The JAX conversion of the gradient function body — `_CachedBuilder` defers `_convert()` to first `fn_builder()` call, which only occurs inside `grad_fn`, which only runs during backpropagation.

**What is not lazy:**
- TF tracing of the gradient function (`get_concrete_function`) — happens at `convert()` time
- `_extract_subgraphs` reads `grad_fn.orig_fn_output_specs` and `grad_fn.input_specs` — requires the TF trace
- `@jax.custom_gradient` re-decoration — happens on every forward call

---

## The circular dependency

`_extract_subgraphs` needs from `grad_fn`:

- `grad_fn.orig_fn_output_specs` → `num_outputs`: splits `IdentityN.inputs[:num_outputs]` (forward outputs) from `[num_outputs:]` (function inputs)
- `grad_fn.input_specs[len(node.input):]` → `captured_inputs`: gradient function inputs beyond the `IdentityN` node

Both require TF tracing. But `num_outputs` can be derived from **graph topology alone**:

> The `IdentityN` inputs that are produced by nodes *inside the subgraph* are the forward outputs. Those produced by nodes *outside* are the function inputs.

This is computable from `subgraph_node_names` and the graph structure — no gradient function needed.

`captured_inputs` is only needed for the **backward pass** and can be deferred.

---

## The design

### Step 1: Remove `_convert_all_gradient_functions` from `convert()`

Do not build `_LibraryFunction` for gradient functions at conversion time. Instead, store a lazy reference in each `_Subgraph`.

### Step 2: Compute `num_outputs` from graph topology in `_extract_subgraphs`

```python
# Replace: num_outputs = len(grad_fn.orig_fn_output_specs)
# With:
num_outputs = sum(
    1 for inp in output_node.inputs
    if inp.op_name in subgraph_node_names
)
```

`_extract_subgraphs` no longer needs the pre-built `grad_fn`. `_Subgraph` stores `grad_fn_name`, a reference to the TF graph, and a lazy builder instead of a `_LibraryFunction`.

### Step 3: Switch `_Subgraph.__call__` to `jax.custom_vjp` with lazy backward

```python
class _Subgraph:
    _vjp_fn = None   # built once, cached on instance

    def __call__(self, named_args, *, rng):
        if self._vjp_fn is None:
            self._vjp_fn = self._build_vjp_fn()
        ...
        return self._vjp_fn(*unboxed_args), {}

    def _build_vjp_fn(self):
        subgraph = self.subgraph
        output_node = self.output_node

        @jax.custom_vjp
        def fn(*args):
            return _run_subgraph_forward(subgraph, output_node, args)

        def fn_fwd(*args):
            out = fn(*args)
            return out, args   # save inputs as residuals

        def fn_bwd(residuals, g):
            # First time here → trigger TF tracing + JAX conversion
            grad_fn = self._get_or_build_grad_fn()
            captured = self._extract_captures(residuals, grad_fn)
            return grad_fn(*g, *captured)

        fn.defvjp(fn_fwd, fn_bwd)
        return fn

    def _get_or_build_grad_fn(self):
        if self._grad_fn is None:
            _convert_gradient_function(self._proto, self._graph, self._library)
            self._grad_fn = self._library[self._grad_fn_name]
        return self._grad_fn
```

### Key properties

- **Forward pass only**: `fn_bwd` never runs, no TF tracing, no `_CachedBuilder` materialisation
- **First backward**: `fn_bwd` runs → TF tracing triggered → `_CachedBuilder` triggered → gradient function compiled to JAX → result cached
- **Subsequent backwards**: `self._grad_fn` is populated → direct call, no rebuilding
- **`_vjp_fn` built once**: `@jax.custom_vjp` decoration happens once per `_Subgraph` instance, not per call

### Residuals and `captured_inputs`

`fn_fwd` saves `args` (the `unique_inputs` of the subgraph) as residuals. These are the inputs to the subgraph, which includes function inputs and external captures. In `fn_bwd`, `grad_fn.input_specs[len(g):]` (the additional captures) can be looked up from `residuals` because they are a subset of `unique_inputs`.

If `captured_inputs` are not a subset of `unique_inputs` (edge case: captures from outer scopes not part of the subgraph's explicit inputs), they may need to be passed as `nondiff_argnums` in `custom_vjp` or stored as a JAX-untraced closure.

---

## Summary of changes

| Location | Before | After |
|----------|--------|-------|
| `convert()` | calls `_convert_all_gradient_functions` | removed |
| `_extract_subgraphs` | reads `grad_fn.orig_fn_output_specs`, `grad_fn.input_specs` | uses graph topology for `num_outputs`; defers captures |
| `_Subgraph.__init__` | stores pre-built `_LibraryFunction grad_fn` | stores lazy builder + TF graph reference |
| `_Subgraph.__call__` | re-applies `@jax.custom_gradient` on every call | builds `custom_vjp` function once, cached on instance |
| First forward call | TF already traced, setup done | nothing extra |
| First backward call | JAX conversion of grad fn (already lazy) | TF tracing + JAX conversion both happen here |

**Net result:** A model used purely for inference pays zero cost for custom gradient support.

## Related pages

- [concepts/custom-gradients.md](../concepts/custom-gradients.md)
- [entities/_Subgraph.md](../entities/_Subgraph.md)
- [entities/_LibraryFunction.md](../entities/_LibraryFunction.md)
- [design/refactoring-recommendations.md](refactoring-recommendations.md)
