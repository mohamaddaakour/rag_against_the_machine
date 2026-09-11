# Final Architecture

## Purpose

The completed project is a local, lexical Retrieval-Augmented Generation (RAG)
system for answering questions about the supplied vLLM codebase. It preserves
the corpus's decoded character positions, returns grader-exact source paths and
spans, and asks `Qwen/Qwen3-0.6B` to answer only from the retrieved evidence.

## Complete structure

```text
rag_against_the_machine/
├── pyproject.toml                 # uv dependencies and Python >= 3.10
├── uv.lock
├── Makefile                       # install, run, debug, clean, lint
├── setup.cfg                      # flake8 and mypy policy
├── README.md
├── scripts/run_pipeline.sh
├── src/
│   ├── __main__.py                # Python Fire command boundary
│   ├── models.py                  # Pydantic wire and internal data models
│   ├── corpus.py                  # safe corpus discovery and lossless reads
│   ├── chunking.py                # Python and Markdown/text chunkers
│   ├── analyzer.py                # identifier-aware lexical tokenisation
│   ├── indexer.py                 # builds and persists TF-IDF/BM25 artefacts
│   ├── bm25.py                    # sparse BM25 weighting and scoring
│   ├── retriever.py               # ranked top-k source locations
│   ├── datasets.py                # validated JSON input/output
│   ├── evaluation.py              # local recall@k calculation
│   └── generator.py               # Qwen prompt construction and generation
├── tests/
│   ├── test_chunking.py
│   ├── test_models.py
│   ├── test_datasets.py
│   ├── test_evaluation.py
│   ├── test_generator.py
│   └── test_cli.py
└── data/                           # ignored by Git
    ├── raw/vllm-0.10.1/
    ├── processed/                 # chunks, vocabulary, sparse matrices, metadata
    ├── datasets/{AnsweredQuestions,UnansweredQuestions}/
    └── output/{search_results,search_results_and_answer}/
```

## Component relationships

```text
data/raw → corpus → chunking → indexer → data/processed
                                      ↓
CLI/query or dataset → retriever ────┘ → MinimalSource[]
                                           ├─→ datasets → graded search JSON
                                           ├─→ evaluation + answered dataset → recall@k
                                           └─→ generator + Qwen → answered JSON
```

`models.py` is the contract between all stages. Only `MinimalSource` fields are
written as retrieved sources: the path is relative to the project root with
forward slashes, and `[first_character_index, last_character_index)` addresses
the original decoded file text. Scores and chunk text remain internal.

## Execution flows

### Indexing

`uv run python -m src index --max_chunk_size 1200` walks `data/raw`, skips
binary/oversized files, reads UTF-8 with newline conversion disabled, and emits
bounded chunks. Python chunks follow declarations where possible; Markdown/text
chunks follow headings and paragraphs. The analyzer preserves ordinary words,
identifier components, and identifier originals. The indexer fits TF-IDF and
creates a sparse BM25 representation, then serialises artefacts below
`data/processed`.

### Search and evaluation

`search` and `search_dataset` load the fitted index once, analyse each query,
score chunks, and select the highest positive top-k BM25 results. Dataset output
is validated as `StudentSearchResults` before it is written. `evaluate` loads
the answered reference dataset and reports recall at 1, 3, 5, and 10. A hit
requires the same exact file path and sufficient span overlap; it never invokes
or imports the official moulinette.

### Answer generation

`answer` retrieves first. `generator.py` re-reads only the selected character
ranges from the raw corpus, constructs a source-labelled prompt that instructs
Qwen to use no unsupported facts, applies a conservative context budget, then
removes any reasoning tags from the displayed answer. `answer_dataset` retains
the source list and writes `StudentSearchResultsAndAnswer`.

## Design decisions and trade-offs

- BM25 is the primary retriever because code identifiers and uneven chunk
  lengths make term-frequency saturation and length normalisation useful. TF-IDF
  remains a simple baseline and diagnostic.
- Chunks never exceed the requested size or 2,000 characters. This protects the
  moulinette's context limit and makes each emitted range valid on its own.
- The project stores sparse matrices using joblib for speed and simplicity.
  They are version-sensitive, so an index is rebuilt after dependency changes.
- Generation is intentionally local and sequential. The mandatory model works
  on CPU; batching, vector embeddings, query caching, and HTTP serving remain
  bonus/future work rather than hidden mandatory complexity.

## Security and resilience

No service is exposed and no credentials are required. CLI boundaries validate
paths, k, chunk sizes, JSON shape, and empty input; file and model errors become
clear messages rather than unhandled tracebacks. Corpus reads use context
managers and decode replacement, outputs are created only in caller-configured
directories, and generated data/model caches remain ignored by Git.

## Testing, performance, and production readiness

Unit tests pin offset invariants, chunk limits, Pydantic serialization,
dataset validation, overlap semantics, prompt construction, and hostile CLI
inputs. `make lint` runs flake8 and the required mypy flags. The final acceptance
run records indexing time (<5 minutes), batch retrieval time (<90 seconds for
200 questions), and recall@5 (docs >=80%, code >=50%) against the real grader.

For a reproducible hand-off: run `uv sync`, supply the ignored data attachments,
run `index`, `search_dataset`, the supplied moulinette, and `answer_dataset`.
The shell script automates that sequence; the README documents the measured
configuration and results. For a larger or multi-user system, add the bonus
embedding/hybrid index, cache invalidation, incremental indexing, and a local
HTTP boundary after mandatory behavior remains green.

## Rebuild-from-memory order

1. Define the Pydantic data contract and lossless corpus reader.
2. Implement bounded chunks and a persisted lexical baseline.
3. Add validated dataset I/O and locally measurable recall.
4. Improve chunking/tokenisation and use BM25 until the measured target passes.
5. Add source-grounded Qwen generation.
6. Harden errors, types, tests, documentation, and the fresh-clone workflow.
