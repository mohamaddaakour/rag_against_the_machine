# RAG against the machine — Roadmap

## Current repository assessment (10 September 2026)

This is an **existing, partial repository**, not a greenfield project. The
implementation currently consists of a Pydantic model module and a corpus-file
walker; `src/__main__.py` and `src/__init__.py` are empty. `pyproject.toml`
already uses the required `uv`, Python 3.10+, Pydantic, Fire, tqdm, and lexical
retrieval dependencies. The Makefile has the required target names but its
recipes are malformed, and `flake8` currently reports errors in `src/corpus.py`.
`README.md` is empty. There are no tests, indexer, chunker, retriever, dataset
I/O, evaluator, generator, or corpus/dataset attachment directories in this
checkout.

Preserve and repair `src/models.py`, `src/corpus.py`, the existing dependency
manifest, `.gitignore`, `setup.cfg`, and Makefile rather than replacing the
repository wholesale. The roadmap adds the missing modules incrementally. The
vLLM repository and the supplied question datasets must be unpacked into
`data/` before Phase 1; their absence here is intentional and consistent with
the subject's rule against committing large data or generated output.

**The spine:** Point it at `data/raw/`, ask a question, get back the exact file paths and character ranges that answer it — then let Qwen3-0.6B read those ranges and write the answer.

**Stack:** Python 3.10+ · `uv` (mandatory) · Python Fire CLI · pydantic v2 · tqdm · scikit-learn sparse matrices for TF-IDF and a hand-rolled BM25 · `transformers` + `Qwen/Qwen3-0.6B` on CPU · flake8 + mypy + Makefile.

**Assumptions:**
- The vLLM corpus, question datasets, and `moulinette` binary are supplied separately with the assignment. They are not present or tracked in this checkout. Phase 1 copies them into the required ignored `data/` layout. If a learner has previously committed those attachments, Phase 1 includes a conditional untracking check before proceeding.
- You develop on **Windows**, but the moulinette ships as Linux ELF binaries only. Phase 2 gives you both routes: a faithful local re-implementation of the metric (`evaluate`), and WSL for the real grader. Every path written into JSON is a forward-slash POSIX path — that is a Phase-1 concern, not a polish item.
- The five bonuses are out of scope for v1. Nothing here blocks adding them later.

## Phases

| # | Phase | What runs at the end | Est. | File |
|---|-------|----------------------|------|------|
| 1 | Search the corpus from the CLI | `uv run python -m src index` then `search "how do I use LoRA?" --k 5` prints 5 real `path [first-last]` hits | ~4h | [PHASE-01-search-the-corpus-from-the-cli.md](PHASE-01-search-the-corpus-from-the-cli.md) |
| 2 | Measure your own recall | `search_dataset` writes valid JSON; `evaluate` prints `Recall@1/3/5/10` over 100 real questions | ~3h | [PHASE-02-measure-your-own-recall.md](PHASE-02-measure-your-own-recall.md) |
| 3 | Beat the recall thresholds | Same commands, recall@5 ≥ 0.80 docs and ≥ 0.50 code — and you can say why | ~6h | [PHASE-03-beat-the-recall-thresholds.md](PHASE-03-beat-the-recall-thresholds.md) |
| 4 | Answer in English with Qwen3-0.6B | `answer "how do I use LoRA?" --k 5` prints a grounded paragraph; `answer_dataset` does 100 of them | ~5h | [PHASE-04-answer-with-qwen.md](PHASE-04-answer-with-qwen.md) |
| 5 | Survive a hostile reviewer | `make lint` clean; `k=0`, empty query, missing index, malformed JSON all handled; perf budgets met | ~4h | [PHASE-05-survive-a-hostile-reviewer.md](PHASE-05-survive-a-hostile-reviewer.md) |
| 6 | Defense-ready from a fresh clone | `./scripts/run_pipeline.sh` goes clone → sync → index → search → score → answer, and the README explains every choice | ~3h | [PHASE-06-defense-ready.md](PHASE-06-defense-ready.md) |

## Progress

- [ ] Phase 1 — Search the corpus from the CLI
- [ ] Phase 2 — Measure your own recall
- [ ] Phase 3 — Beat the recall thresholds
- [ ] Phase 4 — Answer in English with Qwen3-0.6B
- [ ] Phase 5 — Survive a hostile reviewer
- [ ] Phase 6 — Defense-ready from a fresh clone

## What the graded data actually looks like

Measured from `datasets_public/`, not guessed. Several design decisions fall straight out of these numbers.

| Fact | Value | Consequence |
|---|---|---|
| Questions per dataset | 100 docs, 99 code | 200 questions is the perf budget worst case |
| Ground-truth sources per question | **exactly 1**, in both datasets | Recall@k is simply "did any of my top-k overlap the one true span" |
| Truth file types | 99 `.py` (all under `vllm/`), 97 `.md` (mostly `docs/`), 3 `.txt` (`CMakeLists.txt`) | Never skip `.txt`; `.json` and `.yaml` never appear as truth |
| Truth span width — docs | p10 187 · median 1155 · p75 1744 · max 1997 | Docs answers are whole sections |
| Truth span width — code | p10 440 · median 878 · p75 1001 · max 1578 | Code answers are sub-function-sized, and **not** aligned to `def` boundaries |
| Encoding | plain UTF-8, LF newlines, character (not byte) offsets | Read with `newline=""` so nothing is translated |
| Overlap rule | IoU ≥ 0.05 against the truth span | Best possible IoU for a chunk of width `C` over a truth of width `W` is `min(W,C)/max(W,C)` |

That last row is the one people miss. At `--max_chunk_size 2000` the *best case you can possibly reach* is 0.96 on docs and 0.98 on code, because a very short truth span can never make 5 % IoU against a 2000-character chunk. At 900 it is 0.99 / 0.99. Chunk size is not a taste question; it is a ceiling.

Verified by hand: `docs/features/lora.md[4695:6098]` starts exactly on a `### Using API Endpoints` line. The reference chunker is heading-aware for Markdown. Phase 3 exploits that.

## Where files end up

```
rag_against_the_machine/
├── pyproject.toml            # phase 1 · torch + transformers added in phase 4
├── uv.lock                   # phase 1 (generated by uv, committed)
├── Makefile                  # phase 1 (repaired — it is broken today) · phase 6 adds `pipeline`
├── setup.cfg                 # phase 1 — flake8 + mypy config, excludes data/
├── README.md                 # phase 6
├── benchmarks.md             # phase 2 — one row per experiment · filled in 3 and 5
├── .gitignore                # already present
├── scripts/
│   └── run_pipeline.sh       # phase 6
├── src/
│   ├── __init__.py           # already present
│   ├── __main__.py           # phase 1 — Fire CLI; gains commands in 2 and 4
│   ├── models.py             # phase 1 (repaired — it has a crashing typo today)
│   ├── corpus.py             # phase 1 — file walk + grader-exact paths
│   ├── chunking.py           # phase 1 — fixed-size · phase 3 — python + markdown
│   ├── indexer.py            # phase 1 — TF-IDF · phase 3 — BM25 alongside
│   ├── retriever.py          # phase 1 — TF-IDF search · phase 3 — --retriever switch
│   ├── analyzer.py           # phase 3 — identifier-aware tokenizer
│   ├── bm25.py               # phase 3 — BM25 weighting over a sparse count matrix
│   ├── datasets.py           # phase 2 — dataset I/O · phase 4 adds save_answers
│   ├── evaluation.py         # phase 2 — recall@k, faithful to the moulinette
│   └── generator.py          # phase 4 — Qwen3-0.6B answering
├── tests/
│   ├── test_chunking.py      # phase 1 · extended in phase 3
│   ├── test_models.py        # phase 1
│   ├── test_datasets.py      # phase 2
│   ├── test_evaluation.py    # phase 2
│   ├── test_generator.py     # phase 4
│   └── test_cli.py           # phase 5
├── moulinette                # phase 1 — copied from moulinette/, never committed
└── data/                     # never committed (.gitignore already covers it)
    ├── raw/vllm-0.10.1/                             # phase 1 — moved from ./vllm-0.10.1
    ├── processed/                                   # phase 1 — index artefacts
    ├── datasets/AnsweredQuestions/*.json            # phase 1 — ground truth
    ├── datasets/UnansweredQuestions/*.json          # phase 1 — questions only
    └── output/
        ├── search_results/<Scope>/*.json            # phase 2
        └── search_results_and_answer/<Scope>/*.json # phase 4
```

## Vocabulary

Fixed names. Every phase uses exactly these — check here before writing a call.

| Thing | Signature | Introduced |
|---|---|---|
| `models.MinimalSource` | `MinimalSource(file_path: str, first_character_index: int, last_character_index: int)` | Phase 1 |
| `models.ScoredSource` | `ScoredSource(MinimalSource, score: float = 0.0)` — internal only, never serialized | Phase 1 |
| `models.Chunk` | `Chunk(file_path, first_character_index, last_character_index, text, indexed_text=None)` | Phase 1 |
| `Chunk.search_text` | property `-> str` — `indexed_text` if set, else `text` | Phase 1 |
| `Chunk.to_source` | `to_source() -> MinimalSource` | Phase 1 |
| `corpus.list_corpus_files` | `list_corpus_files(raw_dir: Path) -> list[Path]` | Phase 1 |
| `corpus.read_corpus_file` | `read_corpus_file(path: Path, repo_root: Path) -> tuple[str, str]` — `(posix_relative_path, text)` | Phase 1 |
| `chunking.chunk_fixed` | `chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> list[Chunk]` | Phase 1 |
| `chunking.chunk_file` | `chunk_file(file_path: str, text: str, max_chunk_size: int) -> list[Chunk]` — the dispatcher | Phase 1 |
| `chunking.chunk_python` | same signature as `chunk_fixed` | Phase 3 |
| `chunking.chunk_markdown` | same signature as `chunk_fixed` | Phase 3 |
| `analyzer.analyze` | `analyze(text: str) -> list[str]` | Phase 3 |
| `bm25.bm25_weights` | `bm25_weights(counts: csr_matrix, k1: float = 1.2, b: float = 0.75) -> csc_matrix` | Phase 3 |
| `bm25.bm25_scores` | `bm25_scores(weights: csc_matrix, term_ids: Sequence[int]) -> np.ndarray` | Phase 3 |
| `indexer.Indexer` | `Indexer(max_chunk_size: int = 2000)` · `.build(raw_dir: Path, repo_root: Path) -> None` · `.save(processed_dir: Path) -> None` — the default becomes whatever Phase 3 measures best | Phase 1 |
| `retriever.Retriever.load` | `load(processed_dir: Path) -> Retriever` — gains `retriever: str = "bm25"` in Phase 3 | Phase 1 |
| `retriever.Retriever.search` | `search(query: str, k: int = 10) -> list[ScoredSource]` | Phase 1 |
| `datasets.load_dataset` | `load_dataset(dataset_path: Path) -> RagDataset` | Phase 2 |
| `datasets.load_questions` | `load_questions(dataset_path: Path) -> list[UnansweredQuestion]` | Phase 2 |
| `datasets.to_minimal` | `to_minimal(source: MinimalSource) -> MinimalSource` — strips the score | Phase 2 |
| `datasets.save_search_results` | `save_search_results(results: StudentSearchResults, save_directory: Path, dataset_path: Path) -> Path` | Phase 2 |
| `datasets.load_search_results` | `load_search_results(path: Path) -> StudentSearchResults` | Phase 2 |
| `datasets.save_answers` | `save_answers(results: StudentSearchResultsAndAnswer, save_directory: Path, source_path: Path) -> Path` | Phase 4 |
| `evaluation.span_iou` | `span_iou(a: MinimalSource, b: MinimalSource) -> float` | Phase 2 |
| `evaluation.truth_by_id` | `truth_by_id(dataset_path: Path) -> dict[str, list[MinimalSource]]` | Phase 2 |
| `evaluation.recall_at_k` | `recall_at_k(results: StudentSearchResults, truth: dict[str, list[MinimalSource]], k: int) -> float` | Phase 2 |
| `evaluation.recall_report` | `recall_report(results, truth, ks=(1, 3, 5, 10)) -> dict[int, float]` | Phase 2 |
| `generator.read_source` | `read_source(source: MinimalSource, repo_root: Path) -> str` | Phase 4 |
| `generator.build_prompt` | `build_prompt(question: str, sources: list[MinimalSource], repo_root: Path) -> str` | Phase 4 |
| `generator.strip_thinking` | `strip_thinking(text: str) -> str` | Phase 4 |
| `generator.Generator` | `Generator(model_name: str = "Qwen/Qwen3-0.6B")` · `.answer(question: str, sources: list[MinimalSource]) -> str` | Phase 4 |
| `__main__.Cli` | `index` · `search` (P1) · `search_dataset` · `evaluate` (P2) · `answer` · `answer_dataset` (P4) | Phase 1+ |

## Not in v1

The five bonuses, parked deliberately — the subject grades them only once the mandatory part validates *in full*, so they come after Phase 6, never instead of it:

- Semantic embedding index (`all-MiniLM-L6-v2`)
- Hybrid lexical + semantic fusion
- Incremental re-indexing on file change
- Index / query-result caching
- Local HTTP API

Also out: LLM query rewriting, cross-encoder re-ranking, anything that needs a GPU.

## Tech debt ledger

| Taken in | Shortcut | Bites you when | Paid off in |
|---|---|---|---|
| Phase 1 | One fixed-size character chunker for every file type | Code recall sits near the floor; identifiers get cut in half at chunk boundaries | Phase 3 |
| Phase 1 | TF-IDF with the default word tokenizer | `get_model_config` and `AsyncLLMEngine` never match a paraphrased question | Phase 3 |
| Phase 1 | Index persisted with `joblib` (pickle underneath) | A scikit-learn upgrade makes an old `data/processed/` unloadable | Not planned — `index` rebuilds in under 5 min; documented in the README |
| Phase 1 | The whole chunk list is held in RAM during `build()` | The corpus grows past a few hundred MB of text | Not planned — 21 MB of text today |
| Phase 2 | `evaluate` re-implements the grader metric instead of calling it | You tune against your number and the moulinette disagrees at defense | Not planned — mitigated by re-running the real moulinette at every Phase 3 checkpoint |
| Phase 3 | BM25 parameters and chunk size tuned against the *public* datasets | The defense dataset is drawn differently and a 1-point margin evaporates | Not planned — mitigated by coarse tuning and aiming for a wide margin |
| Phase 4 | Answers generated one at a time, no batching, no cache | `answer_dataset` over 100 questions takes 25–45 min on CPU | Not planned — `--limit` covers the dev loop; caching is bonus #4 |
| Phase 4 | Context assembled by character budget, not by counting tokens | A source full of dense punctuation or CJK overruns the model window | Mitigated in Phase 4 step 5 with `truncation=True`; a real token budget is not planned |

## How to use this roadmap

Open one phase file. Work top to bottom — the steps are ordered so nothing gets built twice. Run every `**Check:**` as you go; they take seconds and catch a mistake one step after you make it instead of at the end. Run the "Verify it's done" block, tick the definition of done, commit, then open the next phase. Do not read ahead: each phase assumes the previous one is finished.

The single most expensive mistake in this project is a `file_path` the grader does not recognise. Phase 1, step 3 is where that is won or lost.
