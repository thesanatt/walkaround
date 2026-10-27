# Conformance

`cases/` contains language-neutral inputs and exact normalized parser observations shared by every implementation in this repository. Inputs are either deterministic synthetic examples or privacy-reviewed captured protocol outputs. Captured cases must exclude prompts, source media, source identifiers, credentials, and personal data; every case must cover behavior defined under [`../protocol/`](../protocol/).

Every case is validated against [`case.schema.json`](case.schema.json), a JSON Schema 2020-12 document. `schema_version` is currently `1`. The machine-readable [`compatibility.json`](compatibility.json) maps the current `@meta-sam/parser` and `meta-sam-parser` versions to one composite identity over the checked-in protocol bytes, the parser case schema and its 42-case corpus, and the object-prompt case schema and its 22-case corpus. It is generated, not hand-edited: `node scripts/sync-compatibility` rewrites it from the checked-in sources and `node scripts/validate-conformance` fails when it drifts. TypeScript versioning regenerates it automatically; a Python version change must run the sync command in the same change. The protocol itself is explicitly unversioned; the matrix records its digest without inventing a protocol version. Case filenames must equal `<name>.json`; the case directory is
intentionally flat, and an unsupported file or nested directory fails discovery
instead of being silently ignored. Every implementation must execute every
discovered case; skips and unclassified cases are conformance failures.

## Case envelope

A case defines:

- `media`: `image` or `video`.
- `chunks`: the exact output-text chunks in source order. Chunks may split anywhere, including inside a token or before a newline.
- `events`: a normalized source-event sequence. Each `output_text_delta` references one chunk by zero-based index. Every chunk must be referenced exactly once in order. Text-lane events carry raw official `item_id`, `output_index`, and `content_index` fields under `lane`; these raw fields are intentionally schema-permissive so cases can exercise runtime lane validation, while expected normalized lanes remain strict.
- `source`: an optional deterministic source-failure directive. `iterator` fails while acquiring the iterator, `next` fails after exactly `after_events` events, and `close` fails during source cleanup. Every directive must have a matching normalized source-error expectation, so a runner cannot silently ignore it.
- `expected.snapshots`: every cumulative snapshot emitted while consuming the parser as an async iterator.
- Exactly one terminal expectation: `expected.result` or `expected.error`.

The source vocabulary covers output-text delta and done events, completed and incomplete responses, failed responses, error events, and refusal delta or done events. A terminal event, when present, must be last. Omitting one exercises end-of-source incompletion.

## Normalization

Expected data deliberately does not copy a language implementation's object model:

- Public camel-case fields become snake case, while official source-lane field names remain unchanged.
- Missing frame references normalize to `frame_index: null`.
- Mask identity is the tuple `{media, frame_index, object_id}`, not an implementation-specific key string.
- Canonical mask bounds use half-open `left`, `top`, `right`, and `bottom` coordinates.
- Every accepted mask includes its encoded payload, exact row-major `raster`, exact
  `coco_rle` compressed counts with `[height, width]` size, and exact polygonal
  `svg_path`. The compact `decoded` form remains as a deterministic run-length
  representation: `decoded.length` is the byte count and each `decoded.runs` entry
  is `{value, length}`. Run lengths sum to the byte count, adjacent runs differ,
  and expanding them must equal `raster`.
- Diagnostics include exact severity, code, message, one-based source line, and raw line.
- Terminal errors use stable protocol codes and language-neutral details such as normalized lane identities. Runtime class names and stack traces are excluded.

Snapshots and final results are compared as complete JSON values. Adding a field to actual normalization or omitting expected data therefore fails conformance rather than weakening the comparison.

## Running and extending

Each implementation provides a runner that executes every discovered case through
its Responses API stream adapter. From the repository root:

```sh
node scripts/validate-conformance
```

When adding a case:

1. Use deterministic synthetic text by default. A captured case must satisfy the privacy-review requirements above.
2. Add one JSON file directly under `cases/`; do not add helper files there.
3. Include the exact source sequence and complete expected observations.
4. Run the focused implementation tests and the root conformance entry point.
5. Update every implementation runner in the same change when the schema version or normalized vocabulary changes.

## Object-prompt cases

[`object-prompts/`](object-prompts/) holds language-neutral cases for the object-prompt builder, validated against [`object-prompt.schema.json`](object-prompt.schema.json). Each case gives the builder input — the media `size` and the `objects` with their per-frame boxes and points — and either the exact text every implementation must build or the error code it must report. Input values are deliberately untyped so cases can exercise rejected input. The directory follows the same rules as `cases/`: flat, one `<name>.json` file per case, and every implementation runs every case. Each implementation maps the language-neutral input to its own API; expected data never copies an object model.
