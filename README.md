*This project has been created as part of the 42 curriculum by <your-42-login>.*

# RAG against the machine

## Description

A local, lexical Retrieval-Augmented Generation (RAG) system built over a
snapshot of the [vLLM](https://github.com/vllm-project/vllm) repository
(`vllm-0.10.1`). The goal: given a question about the codebase, retrieve the
source locations that actually answer it — a file path plus a character
range — and, eventually, hand that exact context to a small local model
(`Qwen/Qwen3-0.6B`) so it can answer grounded in real code and docs instead
of guessing.

The project is built in phases (see `ROADMAP/`). **Status: Phase 1 —
indexing and retrieval from the CLI — is complete.** Answer generation
(Qwen) and the recall-quality tuning phases are not implemented yet; the
sections below reflect what actually runs today and are updated as later
phases land.

## System architecture

```
data/raw/vllm-0.10.1/  →  corpus.py  →  chunking.py  →  indexer.py  →  data/processed/
                                                                             │
                                            CLI query  →  retriever.py  ────┘  →  ranked sources
```

- **`corpus.py`** walks `data/raw/` and decodes each indexable file, producing
  the exact `(file_path, text)` pair every other stage relies on.
  `file_path` is always a forward-slash path relative to the repo root
  (`Path.as_posix()`), computed in this one place only, so it matches the
  grader byte-for-byte regardless of OS.
- **`chunking.py`** slides a fixed-size, slightly overlapping window over
  each file's text and records exact character offsets for every chunk.
- **`indexer.py`** chunks the whole corpus, fits a TF-IDF vectorizer over the
  chunk text, and persists three artefacts under `data/processed/`:
  `chunks.jsonl` (chunk metadata only — no text), `tfidf.joblib` (the fitted
  vectorizer + sparse matrix), and `meta.json` (index stats).
- **`retriever.py`** loads those artefacts, transforms a query with the same
  fitted vectorizer, and ranks chunks by cosine similarity (a plain dot
  product, since TF-IDF vectors are L2-normalised).
- **`src/models.py`** defines every pydantic model exchanged between stages
  (`MinimalSource`, `Chunk`, the dataset/answer models from the subject).
- **`src/__main__.py`** is the Python Fire CLI boundary: `index` and
  `search` today; `search_dataset`, `answer`, `answer_dataset`, `evaluate`
  arrive in later phases.

## Chunking strategy

Phase 1 uses a single fixed-size chunker (`chunk_fixed`) for every file
type: a sliding window of `--max_chunk_size` characters (default and hard
cap: 2000) with 15% overlap between consecutive chunks, so a relevant
passage that straddles a chunk boundary is still fully covered by at least
one neighbour. Offsets are computed on the raw decoded text — before any
normalisation — so `text[first:last]` always reproduces the indexed chunk
exactly.

This is deliberately simple: it's plumbing, not a quality feature yet.
Language-aware chunking (splitting Python by function/class, Markdown by
heading) is planned for the recall-tuning phase, once there's a recall
number to measure the improvement against.

## Retrieval method

TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`, `sublinear_tf=True`)
over the chunk text, ranked by cosine similarity. A query with no term in
the vocabulary, an empty query, or `k <= 0` all return no results rather
than an arbitrary set of zero-scored chunks. Retrieval only returns source
locations (`file_path`, `first_character_index`, `last_character_index`) —
never the chunk text itself, which keeps the index small and forces later
stages to re-read the real file by offset.

BM25 and better tokenization (splitting identifiers, weighting headings) are
planned for a later phase and will sit behind the same `Retriever.search`
signature.

## Performance analysis

Not yet measured with the reference datasets — that requires the
`search_dataset` / `evaluate` commands from a later phase. What's verified
so far on the full corpus (2121 indexable files):

| Metric | Observed | Budget (subject) |
|---|---|---|
| Indexing time | a few seconds | ≤ 5 minutes |
| Chunks produced | 12,784 | — |
| Max chunk width | 2000 chars | ≤ `--max_chunk_size` |

Recall@k figures and the 90-second / 200-question retrieval-throughput
check will be added once the dataset-level commands exist.

## Design decisions

- **Path spelling is decided once.** `read_corpus_file` is the only place in
  the codebase that formats a path, always via `Path.as_posix()` relative to
  the repo root. Any other spelling (`str(path)`, `os.path.join`) produces
  backslashes on Windows, which the grader would never match.
- **Offsets index into raw, undecoded-newline text.** Files are opened with
  `newline=""` so `\r\n` sequences are not silently collapsed — that would
  shift every later offset in the file.
- **The index stores metadata only, not chunk text.** `chunks.jsonl` holds
  just `file_path` + the two offsets; the real text is re-read from disk by
  span when it's needed, which also serves as a standing check that offsets
  are correct.
- **Score lives on a separate model.** `MinimalSource` (3 fields) is what
  gets serialized to graded JSON; `ScoredSource` adds `score` for internal/
  debugging use only and is never written to disk.

## Challenges faced

- **Windows path separators.** `str(Path)` and `os.path.join` both produce
  backslash paths on Windows, which would silently fail every grader
  comparison. Solved by centralising path formatting in one function using
  `Path.as_posix()`.
- **mypy failing on a third-party stub, not our code.** `mypy .` under the
  subject's mandatory flags crashed while parsing `numpy`'s bundled type
  stubs (`type X = ...`, a Python 3.12+ syntax form) because the project
  targets `python_version = 3.10`. Fixed by pinning `numpy<2.3` in
  `pyproject.toml`, which ships stubs compatible with the declared target
  version, instead of loosening the mypy target and losing the 3.10
  compatibility check it's there for.
- **`pytest` crawling into the vendored corpus.** With no test-discovery
  scope configured, `pytest -q` walked into `data/raw/vllm-0.10.1/` and
  tried to collect vLLM's own test suite (which imports `torch`, not a
  dependency here). Fixed with `testpaths = ["tests"]` in `pyproject.toml`.

## Instructions

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                              # install dependencies

uv run python -m src index --max_chunk_size 2000     # build the index (run once)

uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
```

Or via the Makefile:

```bash
make install
make index                 # SIZE=2000 to override --max_chunk_size
make search Q="your question here"   # K=5 to override --k
make test                  # uv run pytest -q
make lint                  # flake8 . && mypy . (mandatory flags)
```

`data/raw/vllm-0.10.1/` (the corpus) must be present under `data/raw/`
before indexing — it's supplied as an attachment, not part of this
repository (see `.gitignore`).

## Example usage

```
$ uv run python -m src index --max_chunk_size 2000
Chunking: 100%|███████████████████| 2121/2121 [00:01<00:00, 2186.19file/s]
Vectorizing 12784 chunks ...
Ingestion complete! 12784 chunks. Indices saved under data/processed

$ uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
 1  data/raw/vllm-0.10.1/docs/features/lora.md  [1700-3700]  0.2619
 2  data/raw/vllm-0.10.1/docs/features/lora.md  [13600-15148]  0.2430
 3  data/raw/vllm-0.10.1/docs/features/lora.md  [0-2000]  0.2109
 4  data/raw/vllm-0.10.1/docs/features/lora.md  [3400-5400]  0.2068
 5  data/raw/vllm-0.10.1/docs/features/lora.md  [5100-7100]  0.1986

$ uv run python -m src search "" --k 5
No results.
```

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the original RAG paper.
- [scikit-learn: TfidfVectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) — the ranking function planned for a later phase.
- [vLLM documentation](https://docs.vllm.ai/) — the corpus this project indexes.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — the generation model required by the subject.

**AI usage:** Claude Code (Anthropic) was used throughout this project to
review the codebase against the subject's requirements, write and run unit
tests (`tests/test_chunking.py`, `tests/test_models.py`,
`tests/test_indexer.py`), diagnose and fix tooling issues (a broken mypy/
numpy interaction, a pytest collection crash), and draft this README from
the actual, verified state of the code. All generated code and text were
reviewed and are understood by the author before being committed.

## Recode instructions

Not applicable yet — added if/when specified for this project's evaluation.
