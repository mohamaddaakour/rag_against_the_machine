*This project has been created as part of the 42 curriculum by mdaakour.*

# RAG against the machine

## Description

A local Retrieval-Augmented Generation (RAG) system over a snapshot of the
[vLLM](https://github.com/vllm-project/vllm) repository (`vllm-0.10.1`). Given a
question about the codebase, it retrieves the source locations that answer it
(a file path plus a character range, each at most 2000 characters wide) and
hands that exact context to a small local model, `Qwen/Qwen3-0.6B` running on
CPU, which writes a grounded answer.

Retrieval is lexical (TF-IDF over structure-aware chunks); there are no
embeddings and no external services. Everything runs through a Python Fire CLI:
`index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`,
plus `serve` (bonus HTTP API).

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

Two bonuses are implemented.

**Local HTTP API** (`src/server.py`, FastAPI + uvicorn, `serve` command;
interactive docs at `http://127.0.0.1:8000/docs`):

```bash
uv run python -m src serve --port 8000          # make serve PORT=8000

curl -X POST localhost:8000/search -d '{"query": "How do I load a LoRA adapter?", "k": 5}'
curl -X POST localhost:8000/answer -d '{"query": "Which endpoint loads a LoRA adapter?", "k": 3}'
curl localhost:8000/health ; curl localhost:8000/stats
```

`POST /search` returns the ranked sources, `POST /answer` returns the sources
and the Qwen answer (the model is loaded on the first request and kept in
memory). Invalid bodies (bad JSON, wrong types) return `422`, an absurd
`max_new_tokens` returns `400`, an unbuilt index returns `503`; the server never
prints a traceback. It binds to `127.0.0.1` by default.

**Caching**:

- *Query cache*: `Retriever` keeps an LRU of the last 256 `(query, k)` results,
  so a repeated query skips vectorising and scoring (about 49 ms cold, 1.7 ms
  cached over HTTP, on the real index).
- *Index cache*: `Retriever.load_cached` keeps the loaded index in memory and
  reloads it automatically when `data/processed/` is rebuilt, so the server does
  not re-read the index per request (loading takes about 0.14 s).
- *Answer cache*: `/answer` keeps the last 64 answers, so a repeated question is
  returned instantly instead of re-running the model (33 s -> 0 s in a test).
- `GET /stats` reports the cache hit and miss counters.

## System architecture

```
index:   corpus.py -> chunking.py -> indexer.py (TF-IDF fit) -> data/processed/
search:  retriever.py loads data/processed/ -> transform(query) -> top-k spans
answer:  retrieved spans -> re-read from data/raw by offset -> prompt -> Qwen3-0.6B
```

- `corpus.py` walks `data/raw/`, filters by extension, and produces the exact
  grader path (`data/raw/vllm-0.10.1/...`, forward slashes) in one place only.
- `chunking.py` cuts each file into chunks that carry exact character offsets.
- `analysis.py` is the tokenizer shared by the index and every query.
- `indexer.py` fits the TF-IDF vectorizer and persists `chunks.jsonl`
  (metadata only), `tfidf.joblib` (vectorizer + sparse matrix) and `meta.json`.
- `retriever.py` loads the index and ranks chunks; `search` is a thin wrapper
  over the batched `search_many`, so single and dataset search share one path.
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
- **Fixed window (`chunk_fixed`)**: `--max_chunk_size` characters with 15%
  overlap; used for oversized regions and unparsable files.

Offsets are computed on raw decoded text (files opened with `newline=""`), so
`text[first:last]` always equals the chunk text. No chunk exceeds 2000
characters; `index` rejects larger values because one over-long source would
invalidate a whole output file.

## Retrieval method

TF-IDF (`TfidfVectorizer`, `sublinear_tf=True`) ranked by cosine similarity,
computed as one sparse matrix product for all queries in a batch. Two
identifier-aware additions target code questions:

- The custom analyzer keeps `fused_batched_moe` as a token **and** emits
  `fused`, `batched`, `moe` (also for camelCase), so "fused batched MoE" matches.
- Each chunk is indexed with its own file path tokens prepended
  (`Chunk.indexed_text`), while the returned span stays the raw chunk.

Chunks with a zero score are dropped, so a nonsense query returns nothing.
BM25 was not implemented: TF-IDF already clears both thresholds.

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions), one
correct source per question, hit = same file and IoU >= 0.05:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Required @5 |
|---|---|---|---|---|---|
| docs | 0.530 | 0.800 | **0.830** | 0.880 | 0.80 |
| code | 0.465 | 0.687 | **0.747** | 0.848 | 0.50 |

Evolution of recall@5 (docs / code): fixed window 0.820 / 0.626, then
structure-aware chunking 0.850 / 0.586, then identifier-aware tokenisation
0.830 / 0.747.

| Budget | Required | Measured (Windows laptop, CPU) |
|---|---|---|
| Indexing (2121 files, 14,950 chunks, 61,524 features) | <= 5 min | ~27 s |
| Search 99 questions | <= 90 s / 200 questions | ~2 s |
| Answer generation | not budgeted | ~75-90 s per question on CPU |

`evaluate` is our own reimplementation of the metric; the moulinette binary is
Linux-only and was not run on this machine.

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
- **Code recall**: plain TF-IDF treats `fused_batched_moe` as one opaque token;
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
- [scikit-learn: TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Introduction to Information Retrieval (Manning et al.)](https://nlp.stanford.edu/IR-book/)
- [vLLM documentation](https://docs.vllm.ai/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python Fire](https://github.com/google/python-fire), [pydantic](https://docs.pydantic.dev/), [uv](https://docs.astral.sh/uv/)

**AI usage:** Claude was used to plan the project phase by phase, review code against the subject, and explain the new topics.