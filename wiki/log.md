# Log

*Append-only chronological record of wiki activity.*

Format: `## [YYYY-MM-DD] operation | description`

---

## [2026-04-12] bootstrap | Initial wiki creation

**Operation:** Initial codebase analysis and wiki bootstrap.

**Sources read:**
- Full codebase exploration via Explore agent (thorough analysis)
- Direct reads: `tf2jax.py` (all sections), `ops.py` (all sections), `numpy_compat.py`, `config.py`, `utils.py`, `xla_utils.py`

**Pages created:**
- `SCHEMA.md` — wiki conventions
- `index.md` — content catalog
- `log.md` — this file
- `overview.md` — architecture overview
- `concepts/conversion-phases.md`
- `concepts/op-registry.md`
- `concepts/graph-evaluation.md`
- `concepts/variable-semantics.md`
- `concepts/custom-gradients.md`
- `concepts/library-functions.md`
- `concepts/numpy-compat.md`
- `entities/convert-api.md`
- `entities/_OpNode.md`
- `entities/_Subgraph.md`
- `entities/_TensorEdge.md`
- `entities/_LibraryFunction.md`
- `entities/_HigherOrderFunction.md`
- `entities/Variable.md`
- `design/refactoring-recommendations.md`
- `design/lazy-custom-gradients.md`

**Context:** Extended conversation covering design review, refactoring recommendations (high-level), and the lazy custom gradients design. All findings from that conversation are encoded in the wiki.

**Notable findings:**
- The 12-variable closure in `_convert` is the main cognitive load issue
- Custom gradient code (~350 lines) is entangled with core conversion in `tf2jax.py`
- `_LibraryFunction` name collision between `tf2jax.py` (NamedTuple) and `ops.py` (Protocol)
- TF tracing of gradient functions is eager even for inference-only use; design for lazy fix is in `design/lazy-custom-gradients.md`
- The JAX conversion inside `_CachedBuilder` is already lazy; only TF tracing is not

**Gaps / pages not yet written:**
- `entities/_EvaluationCache.md`
- `entities/config.md` (the 11 config flags)
- `entities/experimental.md` (XlaCallModule, StableHLO)
- `concepts/tfhub-compatibility.md` (the `_UNUSED_INPUT`, `_fix_tfhub_specs` workarounds)

---

## [2026-04-12] ingest | Gap-fill: 4 missing topics from session

**Pages added:**
- `concepts/jax2tf-roundtrip.md` — the JAX→TF→JAX use case and what it drives
- `entities/jax2tf-heuristics.md` — `_infer_relu_from_jax2tf` and cumulative reduction inference
- `entities/_GraphDef.md` — synthetic `_GraphDef`/`_NodeDef` types and the Union type in `_convert`
- Updated `concepts/graph-evaluation.md` — expanded named-arg protocol section with reasoning

**Also updated:** `index.md` entries for new pages; log gaps list trimmed.
