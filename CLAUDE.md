# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, lexical Retrieval-Augmented Generation (RAG) system over a copy of the
vLLM repository. It indexes `data/raw/vllm-0.10.1/`, retrieves ranked source
spans for a question (`file_path`, `first_character_index`,
`last_character_index`), and asks `Qwen/Qwen3-0.6B` to answer only from those
spans. It is a school assignment (`SUBJECT.md`) with an automated grader
(`moulinette`) and a phased build plan (`ROADMAP/`).

Read `ROADMAP/00-OVERVIEW.md` and `ROADMAP/FINAL-ARCHITECTURE.md` first — they
describe the target architecture and the exact function signatures every phase
must produce. Read the current phase file in `ROADMAP/PHASE-0N-*.md` before
implementing anything in that area; each phase file has the full spec,
rationale, and copy-pasteable check commands for its files. Don't jump ahead
of the phase actually being worked on.

## Commands

```bash
uv sync                          # install deps (the reviewer/moulinette only run this)
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "<query>" --k 5
uv run pytest -q                 # all tests
uv run pytest -q tests/test_chunking.py::test_span_reproduces_text   # single test
make install / run / debug / lint / lint-strict / clean / test / index / search
```

`make lint` runs `flake8 .` and
`mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`.
`setup.cfg` excludes `data/` from both — never remove that exclusion, the
corpus alone is thousands of files that aren't ours to lint.

There is no `make` binary in this Git Bash environment — run the `uv run ...`
commands directly, or wrap them, when `make` isn't available.

## Non-negotiable data contract

Two decisions are made once (Phase 1) and never revisited without invalidating
every stored index:

- **`file_path` spelling**: `Path.as_posix()`, relative to the repo root,
  computed in exactly one place (`corpus.read_corpus_file`). Never
  `str(path)` or `os.path.join` — on Windows those produce backslashes, and
  the grader compares paths verbatim (`data/raw/vllm-0.10.1/...`). A wrong
  path fails invisibly: the pipeline runs, prints plausible output, scores
  zero.
- **Offsets**: character offsets into the UTF-8-decoded text with universal
  newline translation disabled (`open(..., newline="")`). `Path.read_text()`
  silently collapses `\r\n` to `\n` and shifts every later offset in that
  file — never use it here. `text[first:last]` (Python slicing, `last`
  exclusive) must always reproduce the original stored text.
- Chunks never exceed `--max_chunk_size` (default 2000, hard cap 2000): the
  moulinette rejects any retrieved source wider than its
  `max_context_length` and invalidates the whole output.

## Architecture (data flow)

```
data/raw → corpus.py → chunking.py → indexer.py → data/processed/
                                             ↓
   CLI query / dataset → retriever.py ──────┘ → MinimalSource[]
                                                    ├─→ datasets.py    → graded search JSON
                                                    ├─→ evaluation.py  → recall@k (own estimate)
                                                    └─→ generator.py + Qwen → answered JSON
```

`src/models.py` is the contract every stage exchanges data through:

- `MinimalSource` — the only thing ever written to graded JSON: `file_path` +
  the two offsets, nothing else (no score, no text).
- `ScoredSource` — `MinimalSource` + `score`, retriever-internal, never
  serialized to a graded file.
- `Chunk` — indexer/retriever-internal only; carries `text` and optional
  `indexed_text` (what actually gets tokenized — lets Phase 3 enrich the
  searchable text with headings/path words without moving the stored offsets).
  `Chunk.to_source()` drops the text down to a `MinimalSource` for persistence.
- `UnansweredQuestion` / `AnsweredQuestion` / `RagDataset` — the ground-truth
  dataset shape. `Union[AnsweredQuestion, UnansweredQuestion]` order matters:
  pydantic tries members left-to-right, so `AnsweredQuestion` (the more
  specific one) must come first or every answered question silently loses its
  `sources`/`answer` and degrades to unanswered.
- `StudentSearchResults` / `StudentSearchResultsAndAnswer` — what `search_dataset`
  and `answer_dataset` write to disk.

Fixed module responsibilities (see `ROADMAP/00-OVERVIEW.md`'s Vocabulary table
for exact signatures before writing a call):

| Module | Owns |
|---|---|
| `corpus.py` | File discovery + the one place paths get formatted (`read_corpus_file`) |
| `chunking.py` | Splitting text into `Chunk`s with exact offsets; `chunk_file` dispatches by file type |
| `analyzer.py` (Phase 3) | Identifier-aware tokenization |
| `indexer.py` | Builds + persists the index (`data/processed/`: `chunks.jsonl`, vectorizer/matrix, `meta.json`) |
| `bm25.py` (Phase 3) | Sparse BM25 weighting/scoring alongside TF-IDF |
| `retriever.py` | Loads the persisted index, ranks a query, returns `ScoredSource[]` |
| `datasets.py` (Phase 2) | Validated JSON I/O for questions/results/answers |
| `evaluation.py` (Phase 2) | Local recall@k — for iteration only, never the source of truth |
| `generator.py` (Phase 4) | Re-reads the exact retrieved spans from disk and prompts Qwen |
| `__main__.py` | The Fire CLI boundary — argument coercion (`str()`/`int()`) happens here, nowhere else |

## Hard constraints from the subject

- Must run via `uv sync` only — no other install path is exercised by the
  grader.
- CLI is Python Fire (`uv run python -m src <command>`), every path/int is a
  real CLI argument with a default, never hardcoded in a function body. Fire
  literal-evaluates arguments, so coerce with `str()`/`int()` at the CLI
  boundary (`search 2000` arrives as the int `2000`, not a string).
- Required commands: `index`, `search`, `search_dataset`, `answer`,
  `answer_dataset`, `evaluate`.
- The CLI must never crash with an unhandled traceback, including on empty
  query, nonsense query, `k=0`, missing files, malformed JSON.
- Never import or call the `moulinette` — `evaluate` is a faithful
  reimplementation for local iteration; the real score at defense comes from
  running `./moulinette` as a separate process.
- Data models must be pydantic; service/orchestration classes (indexer,
  retriever, pipeline) don't have to be.
- All functions need type hints and must pass the mandatory mypy flags
  (`--strict` is optional/bonus).
- Repo layout under `data/` is fixed (`data/raw/`, `data/processed/`,
  `data/datasets/{Answered,Unanswered}Questions/`,
  `data/output/search_results/<Scope>/`,
  `data/output/search_results_and_answer/<Scope>/`) — the reference exam
  scripts assume this exact shape and fail by design if it's off.
- `data/` and the `moulinette` binary are never committed (`.gitignore`
  covers them); they're supplied as attachments and staged locally, not part
  of the repo.

## Docstring/comment style

Keep docstrings and comments simple and short — one or two plain sentences
stating what a function does and why, not exhaustive parameter walkthroughs.
Prefer clarity over completeness; a comment only earns its place if it
explains something non-obvious (why, not what).
