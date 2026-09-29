*This project has been created as part of the 42 curriculum by mdaakour.*

# RAG against the machine

## Description

A local Retrieval-Augmented Generation (RAG) system over a snapshot of the
[vLLM](https://github.com/vllm-project/vllm) repository (`vllm-0.10.1`). Given a
question about the codebase, it retrieves the source locations that answer it
(a file path plus a character range, each at most 2000 characters wide) and
hands that exact context to a small local model, `Qwen/Qwen3-0.6B` running on
CPU, which writes a grounded answer.

Retrieval is hybrid: a lexical BM25 index and a semantic embedding index
(`all-MiniLM-L6-v2`, CPU) over structure-aware chunks, fused into one ranking.
There are no external services. Everything runs through a Python Fire CLI:
`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`,
plus `serve` (bonus HTTP API).

## Informations

- a corpus is the collection of documents (or text data) that the system searches through to find relevant information to feed into the LLM's context before it generates an answer.

- Indexing in a RAG system is the process of organizing your corpus into a structure that allows for fast, efficient search — so that when a query comes in, the system can quickly find the most relevant chunks instead of scanning the entire corpus every time.

## Instructions

Requires Python 3.10+ and [`uv`](https://docs.astral.sh/uv/). The vLLM corpus
must be under `data/raw/vllm-0.10.1/` and the datasets under `data/datasets/`
(both are supplied with the subject and git-ignored).

```bash
uv sync                                                   # install dependencies
uv run python -m src index --max_chunk_size 2000          # build the index

uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Makefile targets: `install`, `run` (`ARGS="..."`), `debug`, `lint`,
`lint-strict`, `test`, `index`, `search` (`Q="..."`), `clean`.

## Example usage

```bash
# One question: ranked sources
uv run python -m src search "How do I load a LoRA adapter?" --k 5

# One question: sources, then a grounded answer from Qwen3-0.6B
uv run python -m src answer "Which endpoint loads a LoRA adapter at runtime?" --k 5

# Own recall@k against the answer key (for iteration only)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10
```

Degenerate input never produces a traceback: an empty query prints
`No results.`; `k <= 0` returns no results; a missing file, malformed JSON, an
unbuilt index or `--max_chunk_size` outside 1..2000 print `error: <message>` on
stderr and exit with code 1.

## Bonus features

Four bonuses are implemented: semantic embeddings, hybrid retrieval, caching
and a local HTTP API.

**Semantic embeddings** (`src/embeddings.py`): `index` also encodes every chunk
with `sentence-transformers/all-MiniLM-L6-v2` on CPU (first 128 tokens of each
chunk, L2-normalised) and saves the vectors to `data/processed/embeddings.joblib`
next to the lexical index.

**Hybrid retrieval** (`Retriever.search_hybrid_many`, the default everywhere):
for each query, the BM25 scores and the embedding cosine scores are each divided
by their maximum, then combined as `alpha * bm25 + (1 - alpha) * semantic` with
`alpha = 0.6`. `--mode lexical` and `--mode semantic` run one side alone, and
`--alpha` changes the weight:

```bash
uv run python -m src search "How do I load a LoRA adapter?" --k 5                 # hybrid
uv run python -m src search "How do I load a LoRA adapter?" --k 5 --mode lexical  # BM25 only
uv run python -m src search_dataset --dataset_path <file> --k 10 --alpha 0.5
```

**Local HTTP API** (`src/server.py`, FastAPI + uvicorn, `serve` command;
interactive docs at `http://127.0.0.1:8000/docs`):

```bash
uv run python -m src serve --port 8000          # make serve PORT=8000

curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
    -d '{"query": "How do I load a LoRA adapter?", "k": 5}'
curl -X POST localhost:8000/answer -H 'Content-Type: application/json' \
    -d '{"query": "Which endpoint loads a LoRA adapter?", "k": 3}'
curl localhost:8000/health ; curl localhost:8000/stats
```

The `Content-Type: application/json` header is required: without it curl sends
form data and the server answers `422`. `POST /search` returns the ranked
sources (hybrid ranking), `POST /answer` returns `{"answer": ...}` (the model is
loaded on the first request and kept in memory). Invalid bodies (bad JSON, wrong types) return `422`, an absurd
`max_new_tokens` returns `400`, an unbuilt index returns `503`; the server never
prints a traceback. It binds to `127.0.0.1` by default.

# Moulinette Usage

```shell
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions

./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

**Caching**:

- *Query cache*: `Retriever` keeps an LRU of the last 256 `(query, k)` results
  per mode (hybrid results are also keyed by `alpha`), so a repeated query skips
  vectorising and scoring (about 49 ms cold, 1.7 ms
  cached over HTTP, on the real index).
- *Index cache*: `Retriever.load_cached` keeps the loaded index in memory and
  reloads it automatically when `data/processed/` is rebuilt, so the server does
  not re-read the index per request (loading takes about 0.14 s).
- *Answer cache*: `/answer` keeps the last 64 answers, so a repeated question is
  returned instantly instead of re-running the model (33 s -> 0 s in a test).
- `GET /stats` reports the cache hit and miss counters.

## System architecture

```
index:   corpus.py -> chunking.py -> indexer.py (BM25 + embeddings) -> data/processed/
search:  retriever.py loads data/processed/ -> BM25 + cosine scores -> fuse -> top-k spans
answer:  retrieved spans -> re-read from data/raw by offset -> prompt -> Qwen3-0.6B
```

- `corpus.py` walks `data/raw/`, filters by extension, and produces the exact
  grader path (`data/raw/vllm-0.10.1/...`, forward slashes) in one place only.
- `chunking.py` cuts each file into chunks that carry exact character offsets.
- `analysis.py` is the tokenizer (identifier splitting + stemming) shared by
  the index and every query.
- `indexer.py` builds the BM25 weights and persists `chunks.jsonl` (metadata
  only), `bm25.joblib` (vectorizer + sparse BM25 matrix), `embeddings.joblib`
  (one vector per chunk) and `meta.json`.
- `embeddings.py` loads the sentence-embedding model and encodes text.
- `retriever.py` loads the index and ranks chunks in three modes (lexical,
  semantic, hybrid); each single-query method wraps its batched `*_many`
  version, so single and dataset search share one path.
- `generator.py` re-reads the retrieved spans from disk, builds a prompt under a
  character budget and runs Qwen greedily (`answer_all` does it for a dataset).
- `evaluation.py` implements the subject's rule (same file, IoU >= 0.05).
- `models.py` holds the pydantic models exchanged between stages;
  `dataset_io.py` reads and writes the JSON files; `__main__.py` is the CLI and
  its single error boundary.

## Chunking strategy

Two strategies, chosen by file type, with a fixed window as a fallback:

- **Python (`chunk_python`)**: `ast` finds top-level `def` / `class` boundaries
  (decorators stay with what they decorate). Regions are packed together up to
  the size limit; a region that is too large is split with the fixed window.
  Files that do not parse fall back to the fixed window.
- **Markdown / text (`chunk_markdown`)**: cut on headings, then neighbouring
  sections are joined to fill each chunk as much as the limit allows.
- **Fixed window (`chunk_fixed`)**: `--max_chunk_size` characters with 30%
  overlap; used for oversized regions and unparsable files. 30% was measured
  against 15% and 45%: it gave the best recall.

Offsets are computed on raw decoded text (files opened with `newline=""`), so
`text[first:last]` always equals the chunk text. No chunk exceeds 2000
characters; `index` rejects larger values because one over-long source would
invalidate a whole output file.

## Retrieval method

Hybrid of a lexical and a semantic ranking.

**Lexical: BM25** (`k1 = 1.2`, `b = 0.75`). At index time every chunk's weight
for every term it contains is precomputed (`indexer.bm25_weights`); a query is a
0/1 vector of the terms it contains, so the BM25 scores of all chunks for a
whole batch of queries are one sparse matrix product. The analyzer is
identifier-aware and stemmed:

- It keeps `fused_batched_moe` as a token **and** emits `fused`, `batched`,
  `moe` (also for camelCase), so "fused batched MoE" matches.
- Every token is stemmed (Snowball, English), so "loading" matches "loaded".
- Each chunk is indexed with its own file path tokens prepended
  (`Chunk.indexed_text`), while the returned span stays the raw chunk.

**Semantic**: cosine similarity between the query embedding and each chunk
embedding (a dot product, since vectors are normalised). It catches questions
worded differently from the text they ask about.

**Fusion**: per query, each side is divided by its maximum score, then
`alpha * bm25 + (1 - alpha) * semantic`, `alpha = 0.6`. Unlike lexical-only
search, a query with no known word can still be answered by the semantic side.

**Dedupe**: because chunks overlap, two neighbouring chunks of the same section
often rank side by side. A candidate is skipped when more than 10% of it is
already covered by a better-ranked chunk of the same file, and the next
candidate takes its place.

Chunks with a zero score are dropped, so a nonsense query in lexical mode
returns nothing.

## Performance analysis

Measured with the moulinette on the public datasets (100 docs questions, 99
code questions), default settings (hybrid, `alpha = 0.6`):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|
| docs | 0.670 | 0.880 | **0.930** | 0.940 | 0.80 |
| code | 0.566 | 0.798 | **0.838** | 0.879 | 0.50 |

How recall@5 evolved (docs / code), each step measured with the moulinette:

| Step | Docs | Code |
|---|---|---|
| TF-IDF, 15% chunk overlap | 0.830 | 0.747 |
| 30% chunk overlap | 0.850 | 0.768 |
| + semantic embeddings, hybrid fusion | 0.860 | 0.808 |
| + stemming | 0.880 | 0.828 |
| + 128-token embeddings, dedupe | 0.900 | 0.818 |
| TF-IDF replaced by BM25 (current) | **0.930** | **0.838** |

Each side alone, for comparison: BM25 only 0.870 / 0.818, embeddings only
0.640 / 0.545. The embeddings are weaker alone but catch questions BM25 misses,
which is why the fused ranking beats both.

| Budget | Required | Measured (campus Linux machine, 24 cores, CPU only) |
|---|---|---|
| Indexing (2121 files, 16,593 chunks) | <= 5 min | 3 min 22 s (about 1,730 s of CPU time) |
| Search 199 questions (hybrid) | <= 90 s / 200 questions | 11 s, model loading included |
| Answer generation | not budgeted | ~75-90 s per question on CPU |

Almost all the indexing time is the embedding step: BM25 alone indexes in about
10 seconds. The embedding step is multi-threaded, so indexing time depends on
the number of cores: on a machine with far fewer cores it can exceed 5 minutes.

`evaluate` is our own reimplementation of the metric, for quick iteration; the
numbers above come from the moulinette.

## Design decisions

- **One place formats paths** (`corpus.read_corpus_file`, `as_posix()`); a
  backslash anywhere would score zero.
- **The index stores metadata only.** The text is re-read by offset, which keeps
  `data/processed/` small and makes the offset invariant the single source of
  truth for both retrieval and generation.
- **Row alignment**: line *i* of `chunks.jsonl` is row *i* of the matrix.
- **Graded output carries only the three `MinimalSource` fields**; the retrieval
  score lives on the internal `ScoredSource`.
- **Greedy decoding** (`do_sample=False`) and Qwen3's thinking mode disabled, so
  answers are reproducible and short. The prompt tells the model to use only the
  context, to say so when the context lacks the answer, and to name the file.
- **Context budget** of 6000 characters keeps prompts small enough for CPU.
- **One error boundary** in `main()` turns expected errors into one-line messages.

## Challenges faced

- **Windows path separators**: `str(Path)` yields backslashes; solved by
  centralising `as_posix()` formatting.
- **Code recall**: plain lexical matching treats `fused_batched_moe` as one opaque token;
  solved by identifier splitting and path tokens (0.586 -> 0.747 at @5).
- **Structure-aware chunking is not free**: it helped docs but initially hurt
  code recall, which is why the tokenizer work followed it.
- **mypy crashing on numpy stubs** under the 3.10 target: fixed by pinning
  `numpy<2.3`.
- **pytest crawling into the vendored corpus**: fixed with `testpaths = ["tests"]`.
- **CPU generation speed**: a 0.6B model still needs over a minute per answer,
  so `answer_dataset` on 100 questions takes hours; `--max_new_tokens` trades
  answer length for time.

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Okapi BM25 (Robertson & Zaragoza, The Probabilistic Relevance Framework)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf)
- [Sentence-Transformers: all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Snowball stemmer](https://snowballstem.org/)
- [Introduction to Information Retrieval (Manning et al.)](https://nlp.stanford.edu/IR-book/)
- [vLLM documentation](https://docs.vllm.ai/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python Fire](https://github.com/google/python-fire), [pydantic](https://docs.pydantic.dev/), [uv](https://docs.astral.sh/uv/)

**AI usage:** AI was used to plan the project phase by phase, review code against the subject, and explain the new topics.