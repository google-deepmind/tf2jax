# Graph Evaluation

*The run-time execution model: a single left-to-right pass over a topologically sorted node list, with reference-counted intermediate storage.*

**Source:** `tf2jax/_src/tf2jax.py:304–340` (toposort), `573–600` (`_EvaluationCache`), `1236–1283` (evaluation loop)

## Topological sort

`_toposort(node_map, end_node_names)` performs a backward-reachability traversal from the output nodes, then reverses to get execution order. Only nodes reachable from outputs are included — dead nodes are silently dropped.

The algorithm works in two passes:
1. DFS from outputs → count how many times each node is a dependency (`child_counts`)
2. Process nodes with no remaining children first; decrement counts as parents are visited

```python
nodes = _toposort(node_map, output_names)   # list[NodeDef]
nodes = [_OpNode(node, library, node_map) for node in nodes]
```

The resulting list always has inputs before their consumers.

## `_EvaluationCache`

```python
class _EvaluationCache:
    outputs: dict[str, Any]   # node name → value (array or tuple of arrays)
    _counts: Counter[str]     # reference count per node name
```

**Initialisation**: pre-populated with `full_inputs` (user args + captured params/constants). Reference counts are set to the number of times each node's output is consumed (by other nodes + by final outputs). Initial inputs get an extra count because they need to survive until used.

**`free_inputs(node)`**: after a node executes, decrement the reference count of each of its inputs. When a count reaches zero, delete the entry from `outputs`. This frees intermediate JAX arrays promptly — important because JAX arrays are immutable references and the Python GC won't free them until all references drop.

## The evaluation loop

```python
eval_cache = _EvaluationCache(nodes, full_inputs, output_args)
for node in nodes:
    if node.name not in eval_cache.outputs:   # skip if already computed (e.g. inputs)
        collected_inputs = [
            (v.op_name, eval_cache.outputs[v.op_name]) for v in node.inputs
        ]
        sub_rng = rng_keys.pop() if node.require_rng else None
        eval_cache.outputs[node.name], updated_params = node(
            collected_inputs, rng=sub_rng)
        
        for var_name, var_val in updated_params.items():
            eval_cache.outputs[var_name] = var_val   # propagate variable updates
            updated_param_names.add(var_name)
        
        eval_cache.free_inputs(node)
```

### Named-arg protocol

Inputs are passed as `list[(name, value)]` rather than just values. `_OpNode.__call__` calls `_unbox_named_args(named_args, self.inputs)` which:
1. Validates that the names match the expected edge op names (defensive consistency check)
2. Extracts indexed outputs: if a node returns a tuple, `arg[inp.idx]` picks the right element

**Why the names are included:** The protocol predates the current strict topological sort. The name check was a defensive guard to catch bugs where the wrong node's output was collected for a given input edge — possible when the collection code was less structured. Today, topological sort + the `[v.op_name for v in node.inputs]` collection loop guarantees correctness, making the name validation redundant.

**Why `(name, value)` tuples rather than just a dict lookup:** The current loop builds `collected_inputs` as an ordered list that matches `node.inputs` positionally, then `_unbox_named_args` validates and strips names. Passing `eval_cache.outputs` directly and indexing by `inp.op_name` would be equivalent and simpler.

> **Refactoring note:** See [design/refactoring-recommendations.md](../design/refactoring-recommendations.md) item 6.

### Variable assignment ops

`AssignVariableOp`, `AssignAddVariableOp`, `AssignSubVariableOp` return updated values and are listed in `_TF_ASSIGN_OPS`. When `_OpNode.__call__` detects these, it includes them in `updated_params`. The loop then writes the updated value back into `eval_cache.outputs` under the *variable node name*, so subsequent reads of that variable see the updated value.

### RNG distribution

Random ops (`RandomStandardNormal`, `RandomUniform`, `RandomUniformInt`) need a JAX PRNGKey. If any are present, `jax.random.split(rng, num_rng_required)` is called once and keys are distributed one-per-random-op. The count `num_rng_required` is computed at build time.

## Control inputs

Some edges are control dependencies (`^NodeName` in the proto) rather than data edges. These are stored separately as `node.control_inputs`. The evaluation loop validates that control dependencies have been executed (their names are in `eval_cache.outputs`) but does not pass their values to the node.

## Output reconstruction

After the loop:

```python
tensor_outputs = tuple([eval_cache.outputs[k.op_name] for k in output_args])
tensor_outputs = [v for v in tensor_outputs if v is not _EMPTY_RETURN_VALUE]
```

`_EMPTY_RETURN_VALUE` is used for ops with no meaningful return (e.g. `NoOp`). Non-tensor outputs (specs that passed through unchanged) are reinserted, and the whole is unflattened into `structured_outputs` shape.

## Related pages

- [concepts/conversion-phases.md](conversion-phases.md)
- [entities/_OpNode.md](../entities/_OpNode.md)
- [entities/_Subgraph.md](../entities/_Subgraph.md)
- [concepts/variable-semantics.md](variable-semantics.md)
