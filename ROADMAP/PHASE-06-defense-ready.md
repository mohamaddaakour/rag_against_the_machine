# Phase 6 — Defense-ready from a fresh clone

**Goal:** One script takes a stranger from `git clone` to a scored, answered pipeline, and the README explains every choice you made.
**Time:** ~3h · **Difficulty:** ●○○○○
**Depends on:** Phase 5 complete. Document the project you actually have.

## ✅ What you'll have when this is done

A repository someone else can run. `./scripts/run_pipeline.sh` goes install → index → search → score → answer without a single manual step, and `README.md` answers the seven questions the subject requires plus the ones a reviewer will ask anyway.

```bash
$ git clone <your-repo> fresh && cd fresh
$ cp -r ../rag_against_the_machine/data/raw ./data/raw     # corpus is not in git
$ cp -r ../rag_against_the_machine/data/datasets ./data/datasets
$ ./scripts/run_pipeline.sh
==> installing
==> indexing (max_chunk_size=1200)
Chunking: 100%|███████████████████| 2121/2121 [00:24<00:00, 87.1file/s]
Ingestion complete! 21259 chunks. Indices saved under data/processed
==> searching docs
Searched 100 questions in 8.4s
==> scoring docs
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
Student data is valid: True
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
==> searching code
...
==> answering 5 docs questions
Answering: 100%|█████████████████| 5/5 [01:12<00:00, 14.5s/q]
==> done
```

## Where you're starting from

Everything works and is clean. `README.md` is a zero-byte file left from the initial commit, and the only way anyone else reproduces your results is by reading six shell commands out of a chat log.

## Why this phase now

Two reasons, both mechanical. The README is a graded deliverable with a fixed list of required sections, and it is the part people write badly at 2 a.m. on the deadline. And the defense runs the pipeline end to end as a single automated flow — if that flow needs a manual step you forgot to mention, the corresponding checks are "considered failed by design", in the subject's own words.

## Before you start

Find your 42 login — the README's first line is checked verbatim and a wrong login there is a silly way to lose a point.

Have `benchmarks.md` open. Everything in the *Performance analysis* section comes from it, and if a number is not in that file you do not have it.

## Key design decisions

- **What the pipeline script covers.** Retrieval only, or retrieval plus generation. Recommendation: **both, with generation limited by default** (`ANSWER_LIMIT=5`). A reviewer running your script should not be stuck for 40 minutes; someone reproducing your full results sets `ANSWER_LIMIT=0`. Print which mode ran.

- **Whether the script fetches the corpus.** Download it, or require it to be in place. Recommendation: **require it, and fail with a clear message.** The corpus is a project attachment, not a public URL, and a script that silently half-works is worse than one that says `data/raw is missing`.

- **Where the moulinette fits.** Required, or optional. Recommendation: **optional, invoked when `./moulinette` is present and executable.** It only runs on Linux; the script must still work on the machine you develop on, printing your own `evaluate` numbers either way.

- **README length.** Recommendation: **long enough to answer the seven required sections properly and no longer.** Every one of them is a likely defense question — *"why BM25?"*, *"why 1200?"*, *"what was hard?"* — so write the answer you would say out loud, not a paraphrase of the subject.

## Files in this phase

| File | New/Edit | What it holds |
|---|---|---|
| `scripts/run_pipeline.sh` | new | The whole flow, install to answers |
| `Makefile` | edit | A `pipeline` target |
| `README.md` | edit | Every required section, filled from your own numbers |
| `.gitignore` | edit | Confirm `data/`, `moulinette`, caches |

## Steps

### 1. Write the pipeline script

**Why:** This is the artefact that proves the project reproduces. It is also the thing you run before the defense to be sure nothing rotted.

Create `scripts/run_pipeline.sh`. Every tunable is an environment variable with a default, so a reviewer runs it bare and you run it with `ANSWER_LIMIT=0` for the real numbers.

```bash
#!/usr/bin/env bash
# scripts/run_pipeline.sh - index, search, score and answer, end to end.
set -euo pipefail

CHUNK_SIZE="${CHUNK_SIZE:-1200}"
K="${K:-10}"
ANSWER_LIMIT="${ANSWER_LIMIT:-5}"

SCOPE="UnansweredQuestions"
DATASETS="data/datasets"
SEARCH_OUT="data/output/search_results/${SCOPE}"
ANSWER_OUT="data/output/search_results_and_answer/${SCOPE}"

cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || {
  echo "uv is required: https://docs.astral.sh/uv/" >&2
  exit 1
}
[ -d "data/raw" ] || {
  echo "data/raw is missing - unpack the vLLM corpus there first" >&2
  exit 1
}
[ -d "${DATASETS}/${SCOPE}" ] || {
  echo "${DATASETS}/${SCOPE} is missing - copy the question datasets there" >&2
  exit 1
}

echo "==> installing"
uv sync

echo "==> indexing (max_chunk_size=${CHUNK_SIZE})"
uv run python -m src index --max_chunk_size "${CHUNK_SIZE}"

for scope in docs code; do
  echo "==> searching ${scope}"
  uv run python -m src search_dataset \
    --dataset_path "${DATASETS}/${SCOPE}/dataset_${scope}_public.json" \
    --k "${K}" \
    --save_directory "${SEARCH_OUT}"

  echo "==> scoring ${scope}"
  uv run python -m src evaluate \
    --student_search_results_path "${SEARCH_OUT}/dataset_${scope}_public.json" \
    --dataset_path "${DATASETS}/AnsweredQuestions/dataset_${scope}_public.json"

  if [ -x ./moulinette ]; then
    ./moulinette evaluate_student_search_results \
      "${SEARCH_OUT}/dataset_${scope}_public.json" \
      "${DATASETS}/AnsweredQuestions/dataset_${scope}_public.json" \
      --k "${K}" --max_context_length 2000
  else
    echo "    (no executable ./moulinette - official score skipped)"
  fi
done

echo "==> answering ${ANSWER_LIMIT:-all} docs questions"
uv run python -m src answer_dataset \
  --student_search_results_path "${SEARCH_OUT}/dataset_docs_public.json" \
  --save_directory "${ANSWER_OUT}" \
  --limit "${ANSWER_LIMIT}"

echo "==> done"
```

```bash
chmod +x scripts/run_pipeline.sh
git update-index --chmod=+x scripts/run_pipeline.sh   # ← Windows does not track it
```

That second command matters on Windows: the filesystem has no execute bit, so without it the script arrives in the repository non-executable and the defense machine answers `Permission denied`. Verify with `git ls-files -s scripts/run_pipeline.sh` — the mode must read `100755`, not `100644`.

`set -euo pipefail` means the script stops at the first failure instead of cheerfully continuing to "score" a search that never ran.

Then add the target to the `Makefile` (**tab-indented**, like every other recipe), and to `.PHONY`:

```make
pipeline:
	./scripts/run_pipeline.sh
```

**Check:**

```bash
./scripts/run_pipeline.sh
```
runs to `==> done` with no manual step. Then, to prove the guards work:

```bash
mv data/raw /tmp/raw-backup && ./scripts/run_pipeline.sh ; echo "exit: $?"
mv /tmp/raw-backup data/raw
```
prints `data/raw is missing - unpack the vLLM corpus there first` and `exit: 1`.

### 2. Write the README

**Why:** It is graded against a checklist, and it is the document that carries your reasoning when you are not in the room.

Replace the empty `README.md`. Everything in angle brackets is yours to fill from `benchmarks.md` — do not ship a placeholder.

````markdown
<!-- README.md -->
*This project has been created as part of the 42 curriculum by <your_login>.*

# RAG against the machine

Ask a question about the vLLM codebase; get an answer grounded in the exact
lines that support it.

## Description

A Retrieval-Augmented Generation system over the vLLM 0.10.1 repository. It
ingests ~2 100 source and documentation files, splits them into chunks with a
strategy chosen per file type, indexes them lexically, retrieves the spans most
relevant to a question, and hands those spans to `Qwen/Qwen3-0.6B` running
locally on CPU to produce a grounded answer.

Retrieval quality is measured with recall@k against the reference datasets:
**<your docs R@5> recall@5 on docs questions** and **<your code R@5> on code
questions**.

## Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # or: make install
# place the corpus at data/raw/vllm-0.10.1/
# place the datasets at data/datasets/{Answered,Unanswered}Questions/
./scripts/run_pipeline.sh                 # or: make pipeline
```

Individual commands:

```bash
uv run python -m src index --max_chunk_size 1200
uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src answer "how do I serve a model with LoRA adapters?" --k 5
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

`make lint` runs flake8 and mypy; `make test` runs the test suite.

## System architecture

```
data/raw/  ──►  corpus.py  ──►  chunking.py  ──►  indexer.py  ──►  data/processed/
                (walk, decode,   (per file type,   (tokenize once,   chunks.jsonl
                 posix paths)     exact offsets)    tfidf + bm25)    tfidf.joblib
                                                                     bm25.joblib
                                                                          │
question ──►  analyzer.py  ──►  retriever.py  ──►  MinimalSource[] ───────┘
              (identifier        (BM25 scoring,      (file_path + span)
               splitting)         top-k)                    │
                                                            ▼
                                        generator.py  ──►  answer
                                        (re-read spans from disk,
                                         Qwen3-0.6B, greedy decode)
```

Each stage exchanges pydantic models (`src/models.py`), so a malformed
hand-off fails at the boundary rather than three stages later. The retriever
returns only `file_path` plus a character span; `answer_dataset` re-reads the
text from the corpus by that span, which means answering needs no index and
continuously re-validates the offsets.

## Chunking strategy

Two strategies plus a fallback, dispatched on file extension in
`chunking.chunk_file`:

- **Python (`.py`, `.pyi`)** — the file is parsed with `ast` and cut at
  definition boundaries: top-level functions and classes, **and methods inside
  classes**, decorators included in the span. Reference spans in the code
  dataset start on decorated methods, so a module-level-only split would miss
  them systematically. Files that fail to parse fall back to fixed windows.
- **Markdown and text (`.md`, `.rst`, `.txt`)** — cut on `#`..`######` heading
  lines. Verified against the reference data: the ground-truth span for
  `docs/features/lora.md` starts exactly on a `###` line.
- **Everything else** — fixed windows with 15 % overlap.

Segments below a quarter of the cap are merged into their neighbour; segments
above the cap are windowed by the same fixed-size splitter. Chunk text is
stored raw, while an enriched `indexed_text` (path words plus the heading
trail or file name) is what gets tokenized — so the searchable text can carry
context that the stored character span does not.

<Chunk size and its effect on recall: fill from benchmarks.md.>

## Retrieval method

Both TF-IDF and BM25 are implemented behind one interface and selectable with
`--retriever`. The default is **<your winner>**, chosen by measurement.

BM25 (`src/bm25.py`) is implemented directly over a scikit-learn sparse count
matrix with `k1=1.2`, `b=0.75`. Because the weight of a term in a document
does not depend on the query, all weights are precomputed at index time and a
query reduces to selecting its columns and summing rows. IDF uses the
`ln(1 + (N - df + 0.5)/(df + 0.5))` variant, which cannot go negative for very
common terms like `self` or `import`.

Both rankers share `analyzer.analyze`, which emits each identifier **and** its
parts: `get_model_config` becomes `get_model_config`, `get`, `model`,
`config`; `AsyncLLMEngine` becomes `asyncllmengine`, `async`, `llm`, `engine`.
This is what lets a question that quotes an identifier verbatim and one that
paraphrases it reach the same chunk.

## Performance analysis

| Metric | Budget | Measured |
|---|---|---|
| Indexing, whole corpus | ≤ 5 min | <your time> |
| Retrieval, 200 questions | ≤ 90 s | <your time> |
| Recall@5, docs | ≥ 0.80 | <your score> |
| Recall@5, code | ≥ 0.50 | <your score> |
| Answering, 100 questions (CPU) | — | <your time> |

<Paste the benchmark table from benchmarks.md, then two or three sentences on
what moved the numbers most and what chunk size did to recall.>

## Design decisions

- **Character offsets, not line or token offsets** — read with
  `newline=""` and `errors="replace"` so a stored span always slices back to
  the indexed text. Validated against a reference span in the public dataset.
- **Paths spelled in exactly one place** (`corpus.read_corpus_file`, via
  `as_posix()`) — the grader compares them verbatim, and this project is
  developed on Windows.
- **Lexical retrieval only** — the mandatory part requires TF-IDF or BM25;
  embeddings are a bonus and would not have fixed the failures actually
  observed.
- **`MinimalSource` carries exactly the three graded fields**; the retrieval
  score lives on a `ScoredSource` subclass that never reaches a results file.
- **Tokenize once at index time** with `analyzer=identity` on both vectorizers
  — running a Python analyzer twice over the corpus is the difference between
  a two-minute and a six-minute index.
- **Answers are generated greedily** (`do_sample=False`) so a prompt change is
  the only reason an output changes.

<Add any decision a reviewer asked you about; that is what this section is for.>

## Challenges faced

<Three or four real ones, with what you did. Candidates from this build:>

- <The dominant retrieval failure shape you identified in Phase 3, and the fix.>
- <Whatever cost you the most time — chunk boundaries, the IoU ceiling at large
  chunk sizes, Qwen3 thinking mode eating the token budget, running a Linux
  grader binary from Windows.>

## Example usage

```console
$ uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
 1  data/raw/vllm-0.10.1/docs/features/lora.md  [4695-5900]  12.4831
 2  data/raw/vllm-0.10.1/docs/features/lora.md  [3400-4695]  10.2216
 ...

$ uv run python -m src answer "how do I serve a model with LoRA adapters?" --k 5
<paste a real answer from your own run>
```

## Resources

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — Robertson & Zaragoza
- [scikit-learn: text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python `ast` module documentation](https://docs.python.org/3/library/ast.html)
- [vLLM documentation](https://docs.vllm.ai/) — the corpus itself

### Use of AI

<Be specific and honest; a vague sentence here invites questions you will not
enjoy. Name the tasks, and say what you verified yourself. For example:>

AI assistance was used for <planning the phase breakdown / drafting the BM25
weighting code / explaining `ast` line-to-character conversion / reviewing
error handling>. Every generated fragment was read, tested and modified before
being kept; <name the parts you wrote unaided>. The chunking strategy and the
retrieval evaluation were validated by hand against the reference dataset
rather than taken on trust.
````

Write the *Challenges faced* section from what actually happened to you, not from the list above. It is the section a reviewer uses to tell whether you built the project or received it.

**Check:** open `README.md` and confirm the first line is italicised and reads exactly `*This project has been created as part of the 42 curriculum by <login>.*` with your real login, and that no `<angle bracket>` placeholder survives:

```bash
grep -n '<' README.md | grep -v 'http' | grep -v '^.*```'
```
should return nothing but genuine HTML-free lines.

### 3. Rehearse from a fresh clone

**Why:** Everything works on your machine because of something you did three weeks ago and forgot. A clone is the only way to find out what.

```bash
cd /tmp
rm -rf fresh
git clone /c/Users/user/Desktop/my_projects/python/rag_against_the_machine fresh
cd fresh
mkdir -p data
cp -r /c/Users/user/Desktop/my_projects/python/rag_against_the_machine/data/raw data/raw
cp -r /c/Users/user/Desktop/my_projects/python/rag_against_the_machine/data/datasets data/datasets
cp /c/Users/user/Desktop/my_projects/python/rag_against_the_machine/moulinette .
./scripts/run_pipeline.sh
make lint
make test
```

Anything that fails here is a real defect in the repository — a missing file, an uncommitted `uv.lock`, a path that only resolves on your machine. Fix it in the original repo, commit, and re-clone until this block runs clean.

**Check:** the fresh clone reaches `==> done`, `make lint` exits 0, `make test` passes.

### 4. Rehearse the defense itself

**Why:** The subject warns you about a "recode": a small live modification to prove you understand your own code. Knowing where things live turns that from a panic into two minutes.

Answer these out loud, without opening the files:

- Where is a `file_path` created, and why in exactly one place?
- Why is `last_character_index` exclusive, and what would break if it were not?
- Why does the BM25 IDF have a `1 +` in it?
- What does `indexed_text` hold that `text` does not, and why are they separate fields?
- Why is `Generator` constructed outside the answering loop?
- What does chunk size do to the *maximum achievable* recall, and why?

Then do a practice recode against a clock. Realistic asks, all under five minutes if you know the codebase:

| Likely ask | Where it lands |
|---|---|
| "Add a `--min_score` flag to `search`" | `Retriever.search`, one filter; `Cli.search`, one argument |
| "Print the file extension distribution after indexing" | `Indexer.build`, a `collections.Counter` over the paths |
| "Make the analyzer keep single-character tokens" | `analyzer.MIN_PART_LENGTH` |
| "Change the chunk overlap to 25 %" | `chunking.OVERLAP_RATIO`, then re-index |
| "Return the answer as JSON instead of prose" | `Cli.answer`, `json.dumps` on a small dict |

Pick two and actually do them, then `git checkout .`. If either took more than five minutes, that part of the code is not as clear as you think.

**Check:** you completed two recodes inside five minutes each, and reverted them.

### 5. Final commit

```bash
git status --short          # empty
git log --oneline           # one commit per phase, readable
ls pyproject.toml uv.lock Makefile README.md scripts/run_pipeline.sh
git ls-files -s scripts/run_pipeline.sh   # mode 100755
```

**Check:** every required artefact is committed and nothing else is.

## Common pitfalls

| Pitfall | Why it happens | Avoid it by |
|---|---|---|
| The README's first line is not italicised or has the wrong login | It is copied from the subject without editing | Read it once against the subject, character by character |
| `<placeholder>` text left in the README | The template was filled in a hurry | The `grep -n '<'` check in step 2 |
| Performance numbers invented from memory | `benchmarks.md` was not open | Every number in the README traces to a row in `benchmarks.md` |
| `run_pipeline.sh` commits without the execute bit | Windows does not track file modes | `git update-index --chmod=+x`, then verify `100755` |
| CRLF line endings on the shell script | Windows editor defaults | `file scripts/run_pipeline.sh` must not say "CRLF"; a `\r` makes Linux report `bad interpreter` |
| The fresh clone fails on `uv sync` | `uv.lock` was gitignored or never committed | `git ls-files uv.lock` must list it |
| "Use of AI" written as one vague sentence | It feels like a formality | Name the tasks and what you verified; vagueness invites the questions you least want |
| Testing the pipeline only with a warm HuggingFace cache | You downloaded the weights weeks ago | Note the first-run download cost in the README so a reviewer is not surprised by a 5-minute pause |

## Verify it's done

```bash
cd /tmp && rm -rf fresh && git clone <repo path> fresh && cd fresh
mkdir -p data && cp -r <original>/data/raw data/ && cp -r <original>/data/datasets data/
cp <original>/moulinette .
./scripts/run_pipeline.sh
make lint
make test
head -1 README.md
```

Expected:

```
==> installing
==> indexing (max_chunk_size=1200)
Ingestion complete! 21259 chunks. Indices saved under data/processed
==> searching docs
Searched 100 questions in 8.4s
==> scoring docs
Questions scored: 100
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
Student data is valid: True
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
==> searching code
...
==> answering 5 docs questions
==> done
Success: no issues found in 17 source files
47 passed
*This project has been created as part of the 42 curriculum by <your_login>.*
```

## Definition of done

- [ ] `./scripts/run_pipeline.sh` runs clone-to-answers with no manual step, committed as `100755`
- [ ] `make pipeline` runs it
- [ ] The script fails clearly and early when `data/raw` or the datasets are missing
- [ ] `README.md` has all seven required sections plus Description, Instructions and Resources
- [ ] The first line is italicised and carries your real 42 login
- [ ] Every performance number in the README traces to a row in `benchmarks.md`
- [ ] The "Use of AI" paragraph is specific about tasks and about what you verified
- [ ] A fresh clone runs the pipeline, `make lint` and `make test` clean
- [ ] You can answer the six questions in step 4 without opening a file
- [ ] Two practice recodes done in under five minutes each, then reverted
- [ ] Committed

## Deliberately NOT in this phase

- Any change to retrieval, generation or the CLI → all closed; changing code now invalidates the README you just wrote
- The five bonuses → start them *after* this, only once the mandatory part validates in full: semantic embeddings, hybrid retrieval, incremental indexing, caching, a local HTTP API
- CI, Docker packaging, a web UI → not in v1

## Commit

```bash
git add -A
git commit -m "phase 6: pipeline script and full README"
```

## Next

That is the mandatory part, complete and defensible.

If you want the bonuses, take them in this order — each one is a self-contained addition that does not disturb what you have: **caching** (smallest, and it makes every other loop faster), **semantic embeddings** (`all-MiniLM-L6-v2` beside the lexical index), **hybrid retrieval** (fuse the two rankings; your `benchmarks.md` already has the harness to prove it helps), **incremental indexing**, then the **local HTTP API**. Each deserves its own phase file written the same way this one was — and the subject grades none of them until every mandatory box above is ticked.
