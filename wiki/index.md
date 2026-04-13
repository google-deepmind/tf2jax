# Wiki Index

*Content-oriented catalog of all pages. Read this first when looking for a specific topic.*

---

## Overview

| Page | Summary |
|------|---------|
| [overview.md](overview.md) | Architecture overview: what tf2jax does, the 3-layer data model, build vs run time, module map |

---

## Concepts

Core mechanisms and design patterns.

| Page | Summary |
|------|---------|
| [concepts/conversion-phases.md](concepts/conversion-phases.md) | Build time vs run time: what `_convert()` does before and inside `jax_func`; the 12 captured variables |
| [concepts/op-registry.md](concepts/op-registry.md) | The `_jax_ops` dict, two-level factory pattern, `register_operation()` decorator, dispatch via `get_parser()` |
| [concepts/graph-evaluation.md](concepts/graph-evaluation.md) | Topological sort, `_EvaluationCache` reference counting, the evaluation loop, named-arg protocol, variable assignment handling |
| [concepts/variable-semantics.md](concepts/variable-semantics.md) | TF variables → JAX params dict; extraction, threading, functional update semantics |
| [concepts/custom-gradients.md](concepts/custom-gradients.md) | `IdentityN` detection, subgraph extraction, `jax.custom_gradient` wrapping, gradient function conversion pipeline |
| [concepts/library-functions.md](concepts/library-functions.md) | `FunctionDefLibrary`, `_CachedBuilder` lazy compilation, `_LibraryFunction`, higher-order op integration |
| [concepts/numpy-compat.md](concepts/numpy-compat.md) | Dual `np`/`jnp` dispatch layer, polymorphic dimension handling, dtype tables |
| [concepts/jax2tf-roundtrip.md](concepts/jax2tf-roundtrip.md) | JAX→TF→JAX round-trip use case; what jax2tf does to graphs and how tf2jax handles it |

---

## Entities

Specific classes, functions, and modules.

| Page | Summary |
|------|---------|
| [entities/convert-api.md](entities/convert-api.md) | `convert()`, `convert_functional()`, `convert_from_restored()`, `AnnotatedFunction`, config flags |
| [entities/_OpNode.md](entities/_OpNode.md) | Wraps `NodeDef` proto, bakes static attrs at build time, uniform call interface for evaluation loop |
| [entities/_Subgraph.md](entities/_Subgraph.md) | Group of nodes wrapped with `jax.custom_gradient`; `rewrite()` splices it into the main node list |
| [entities/_TensorEdge.md](entities/_TensorEdge.md) | Edge in the computation graph: `(op_name, idx, is_control)` parsed from proto input strings |
| [entities/_LibraryFunction.md](entities/_LibraryFunction.md) | Lazily-compiled JAX function from a `FunctionDef`; holds specs and `_CachedBuilder` |
| [entities/_HigherOrderFunction.md](entities/_HigherOrderFunction.md) | Dataclass protocol for control flow ops that need inner function kwargs at call time |
| [entities/Variable.md](entities/Variable.md) | NumPy `ndarray` subclass carrying `trainable` and `name`; lives in the `params` dict |
| [entities/_GraphDef.md](entities/_GraphDef.md) | Synthetic `_GraphDef`/`_NodeDef` NamedTuples that let `_convert` handle both real GraphDefs and library functions uniformly |
| [entities/jax2tf-heuristics.md](entities/jax2tf-heuristics.md) | `_infer_relu_from_jax2tf` and cumulative reduction inference: pattern-matching passes that restore clean JAX ops from jax2tf artifacts |

---

## Design

Design proposals, refactoring notes, trade-off records.

| Page | Summary |
|------|---------|
| [design/refactoring-recommendations.md](design/refactoring-recommendations.md) | 7 structural improvements: `_CompiledGraph` class, `_GraphNode` protocol, custom gradient module, name collision fixes |
| [design/lazy-custom-gradients.md](design/lazy-custom-gradients.md) | Full design for deferring gradient function TF tracing to first backward call using `jax.custom_vjp` |

---

## Schema and meta

| Page | Summary |
|------|---------|
| [SCHEMA.md](SCHEMA.md) | Wiki conventions, directory structure, ingest/query/lint workflows |
| [log.md](log.md) | Append-only chronological record of wiki activity |
