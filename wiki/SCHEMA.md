# tf2jax Wiki Schema

This wiki captures deep knowledge about the tf2jax codebase. It is maintained by Claude Code and read by developers.

## Directory structure

```
wiki/
  SCHEMA.md          ← this file: conventions and workflows
  index.md           ← content-oriented catalog of all pages
  log.md             ← append-only chronological record
  overview.md        ← architecture overview and entry point

  concepts/          ← design concepts, mechanisms, patterns
  entities/          ← specific classes, functions, modules
  design/            ← design proposals, refactoring notes, trade-off records
```

## Page conventions

- Every page starts with a `# Title` heading and a one-line description in italics.
- Cross-links use `[[PageName]]` style anchors rendered as `[PageName](../path/to/page.md)`.
- Every concept/entity page has a **Source** line pointing to the file:line range.
- Code snippets are kept short — illustrative, not exhaustive.
- Pages in `design/` capture intent and reasoning, not just what the code does.

## Operations

**Ingest** (adding new knowledge from the codebase):
1. Read the relevant source file(s).
2. Create or update the entity/concept page.
3. Update `index.md` with a new entry.
4. Add a log entry to `log.md`.
5. Update cross-links on related pages.

**Query** (answering questions):
1. Read `index.md` to find relevant pages.
2. Read the relevant pages.
3. Synthesize an answer. If the answer is generally useful, file it as a new page in `design/`.

**Lint** (periodic health check):
- Look for orphan pages (no inbound links).
- Look for concepts mentioned but lacking their own page.
- Look for stale claims after code changes.
- Suggest new questions and sources.

## Source of truth

The source of truth is always the codebase at `/Users/shaobohou/Downloads/projects/tf2jax/`. Wiki pages may become stale after refactoring — always verify file:line references before acting on them.
