# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

42-school project "RAG against the machine" (`SUBJECT.md` is the assignment). A lexical RAG system over the vLLM 0.10.1 source tree in `data/raw/vllm-0.10.1/`. The grader (`moulinette`, a Linux ELF binary described in `README_M.md`) scores **retrieval**: for each question, does a top-k source share the ground-truth file and overlap its character span. Pass bars: Recall@5 ≥ 80% on the docs dataset, ≥ 50% on the code dataset.

Implemented commands: `index`, `search`, `search_dataset`, `evaluate`, `answer`. Not yet implemented: `answer_dataset` (Phase 13). Chunking has the two required strategies (`chunk_python` on ast def/class boundaries, `chunk_markdown` on headings, fixed window as fallback). Recall on the public datasets: docs@5 0.830, code@5 0.747 (bars: 0.80 / 0.50). BM25 (Phase 11) is skipped under decision D4. Work is planned phase by phase in `ROADMAP/`; `ROADMAP/00-OVERVIEW.md` holds the phase table and status column. That overview was written assuming a fresh project in a `rag_final` folder — ignore that; development happens in this repository.

## Commands

Package manager is `uv`; always run through `uv run`. The Makefile (`install run debug lint lint-strict test index search clean`) is a submission requirement for Linux reviewers; `make` is not installed on this Windows machine, so use the raw commands.

```powershell
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How do I load a LoRA adapter?" --k 5
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

uv run pytest -q
uv run pytest -q tests/test_retriever.py::test_search_many_crosses_a_batch_boundary
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

flake8 must print nothing and mypy must pass with exactly those flags (they are mandated by the subject and duplicated in `setup.cfg` and the Makefile's `MYPY_FLAGS` — keep both in sync). `disallow_untyped_defs` applies to tests too, so every function, including tests and fixtures, needs annotations. Line length is 88.

Shell is Windows PowerShell 5.1: it strips double quotes inside `uv run python -c "..."` arguments and drops empty-string arguments (`search ""` reaches Fire as a missing argument). Put throwaway Python in a script file instead of `-c`.

## Architecture

Single-process CLI with an offline build step and an online query step that never share a process:

```
index:  corpus.py (which files, exact paths) -> chunking.py (spans) -> indexer.py (TF-IDF fit) -> data/processed/
search: retriever.py loads data/processed/ -> vectorizer.transform(query) -> matrix @ query.T -> top-k spans
```

- `src/__main__.py` — `Cli` class handed to Python Fire; each public method is a command. Fire does not enforce type hints (`--k abc` arrives as a string), so commands cast their own arguments. `main()` is the error boundary: it turns `FileNotFoundError` and `ValueError` (which includes pydantic's `ValidationError`) into `error: <msg>` on stderr with exit code 1. Fire usage errors exit 2.
- `src/models.py` — pydantic v2 models. `MinimalSource`, the question/dataset models and the `Student*` result models are the grader's contract from the subject (§VI.4); do not rename or drop their fields. `ScoredSource` (adds `score`) and `Chunk` are internal.
- `src/dataset_io.py` — load `RagDataset` files (accepts both Answered and Unanswered, via the union) and write result JSON.
- `src/retriever.py` — `search` is a thin wrapper over `search_many`, which scores queries in batches of `BATCH_SIZE`; ranking lives only in `_top_k`. Keep single and batch search on one code path.

### Invariants that silently break grading if violated

- **Paths**: `file_path` must match the grader verbatim (`data/raw/vllm-0.10.1/...`, forward slashes). It is formatted in exactly one place, `corpus.read_corpus_file`, via `relative_to(repo_root).as_posix()`. Never build it elsewhere.
- **Offsets**: `text[first_character_index:last_character_index] == chunk.text`, in decoded characters. Corpus files must be opened with `newline=""` (keeps `\r\n` as two characters) and `errors="replace"`; any code that re-reads spans must open files the same way.
- **Width**: no returned source may exceed 2000 characters — one violation invalidates a whole output file. Nothing currently caps `index --max_chunk_size`, so values above 2000 produce invalid output.
- **Row alignment**: line *i* of `chunks.jsonl` and row *i* of the matrix in `tfidf.joblib` are the same chunk, linked only by position. Both are written from `Indexer.chunks` in order; any reordering or filtering must happen before `save`. `meta.json` is written but not read.
- **Graded output carries only the three `MinimalSource` fields**: `search_dataset` converts `ScoredSource` to `MinimalSource` before writing.
- **Stale index**: `search`/`search_dataset` use the vectorizer saved at index time. Any change to `TEXT_SUFFIXES`, chunking, `OVERLAP_RATIO`, vectorizer options or `Chunk.indexed_text` has no effect until `index` is re-run.

### Extension points

- New chunking strategy: branch inside `chunking.chunk_file` (currently always `chunk_fixed`: 2000-char window, step 1700 = 15% overlap).
- Index enriched text without changing the returned span: set `Chunk.indexed_text` (the vectorizer uses `Chunk.search_text`). Not used yet.
- Default tokenisation (`(?u)\b\w\w+\b`, lowercased) keeps `snake_case` identifiers as one token, so "fused batched MoE" does not match `fused_batched_moe` — the planned lever for code-dataset recall.

## Subject constraints to respect

- All input/output paths are CLI arguments, never hard-coded. Output goes under `data/output/search_results/<Scope>/` and `data/output/search_results_and_answer/<Scope>/`, scoped by `UnansweredQuestions`/`AnsweredQuestions` because both scopes use the same file names.
- Degenerate input (empty query, `k<=0`, missing files, malformed JSON) must never produce a traceback; `Retriever.search_many` returns empty lists rather than raising.
- Long operations show `tqdm` progress bars. Retrieval budget: ≤ 90 s for 200 questions; indexing ≤ 5 min.
- The solution must never import or call the moulinette.
- `data/` and `moulinette` are git-ignored; the evaluator rebuilds the index. `pytest` is restricted to `tests/` (`testpaths`) because `data/raw` contains vLLM's own test suite.

## Rules

- Don't use git commands ever.