# Phase 7 — Batch search over a dataset

**Goal:** add the `search_dataset` command. It reads a question dataset, gets
the top-k sources for every question in one pass, and writes the
`StudentSearchResults` JSON file that the grader reads.

---

## What the system can do after this phase that it cannot do now

Right now you can ask one question and read the ranked spans on screen. The
grader never runs `search`. At the defense, the reference scripts run exactly
this:

```
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

Then they pass the file it wrote to the moulinette. After this phase that file
exists, has the right shape, and is fast enough to produce: N2 allows at most
90 seconds for 200 questions.

It is also what Phase 8 needs. `evaluate` scores a results file against the
answered dataset, so without this file there is nothing to measure.

## Prerequisites

- Phase 6 works: `uv run python -m src search "lora" --k 5` prints ranked spans.
  **Verified on 2026-09-15.**
- `data/processed/` holds a built index (12 784 chunks, 61 127 features).
- `data/datasets/UnansweredQuestions/` and `data/datasets/AnsweredQuestions/`
  each hold `dataset_docs_public.json` and `dataset_code_public.json`.
- `StudentSearchResults`, `MinimalSearchResults` and `RagDataset` already exist
  in `src/models.py`. **Do not change their fields.** They are the contract.

### Phase 6 gaps — closed on 2026-09-15

1. The `flake8` E501 error at `src/__main__.py:55` was fixed; `flake8 .` is clean.
2. `tests/test_retriever.py` now exists with 6 tests of the current
   `Retriever` (load with a missing index, ranking, the `k` limit, sort order,
   the 2000-character width, degenerate inputs). They pass against the
   pre-refactor code, so they are the safety net for Step 2.

Baseline on 2026-09-15: `flake8` clean, `mypy` clean on 12 files, **25 passed**.

(Missing index still prints a raw `FileNotFoundError` traceback. That is known,
and it belongs to Phase 14. Leave it for now.)

---

## Concepts introduced in this phase

### Concept 1 — Batch vectorisation: one matrix product for many queries

**The problem.** The obvious version of `search_dataset` calls
`retriever.search(question, k)` in a loop. Each call runs `vectorizer.transform`
on one string, multiplies one sparse vector against the matrix, and turns one
column into a dense array. That works, but it repeats the Python and SciPy
overhead once per question. The rule is 90 seconds for 200 questions, *and that
includes loading the index*.

**What solves it.** Linear algebra doesn't care how many queries you have.
Your index matrix `M` has shape `(n_chunks, n_features)`. Stack the queries into
`Q` with shape `(n_queries, n_features)`. Then:

```
scores = M @ Q.T          # shape (n_chunks, n_queries)
```

Column `j` of `scores` is exactly the score vector that a single `search` would
have made for query `j`. You get the same numbers with one `transform` and one
multiply instead of 100 of each.

**Tiny concrete example.** Say there are 3 chunks and 2 queries:

```
scores = [[0.9, 0.0],     chunk 0
          [0.1, 0.7],     chunk 1
          [0.4, 0.2]]     chunk 2
            q0   q1
```

Top-2 for q0: column 0 → chunks 0, 2. Top-2 for q1: column 1 → chunks 1, 2.
The ranking code you already have (`argpartition`, then `argsort`, then stop at
score ≤ 0) runs unchanged on each column.

**Why now.** This is the first command that handles many queries, and N2 is
checked against this command.

**What it costs.** Memory. The dense score matrix is
`n_chunks × n_queries × 8 bytes`. For 12 784 chunks and 100 questions that is
about 10 MB, which is nothing. But datasets have no size limit, and Phase 9 will
change the chunk count. So process queries in **fixed-size batches** (64 is a
good size). Memory then stays flat no matter how big the dataset is, and each
batch gives `tqdm` one step to show.

### Concept 2 — Writing to the output contract

**The problem.** The moulinette reads your JSON file with its own copy of the
models. A misspelled key, a missing `k`, a backslash in `file_path`, or a file
saved in the wrong place will not raise an error on your side. It just scores
zero, or the reference script can't find the file.

**What solves it.** Build the output as pydantic objects, never as hand-made
dicts, and serialise it with `model_dump_json`. The file can only contain what
the model allows. Name the output file after the input file, because the
subject's walkthrough expects
`data/output/search_results/UnansweredQuestions/dataset_docs_public.json` for
the input `.../UnansweredQuestions/dataset_docs_public.json`.

**Tiny concrete example.**

```python
results = StudentSearchResults(search_results=[...], k=10)
(save_dir / dataset_path.name).write_text(results.model_dump_json(indent=2))
```

**Why now.** This is the first command whose output a program reads, not a
person.

**What it costs.** One decision you must make on purpose: **what goes in each
source**. The subject allows extra fields, and `ScoredSource` carries `score`.
This roadmap writes **plain `MinimalSource`** objects (only the three graded
fields). Extra fields gain nothing for grading, and they add a small risk with a
strict reader. Keep scores for the terminal output of `search`.

---

## Concepts deliberately deferred

| Concept | Deferred to | Why not now |
|---|---|---|
| Measuring recall against `AnsweredQuestions` | Phase 8 | You need the file from this phase first. |
| Clean messages for a missing dataset, malformed JSON, `k=0`, a missing index | Phase 14 | Phase 14 adds one error boundary for all commands. Here you only avoid making it worse. |
| Changing chunking or tokenisation to improve ranking | Phases 9–10 | Phase 8 must record a baseline first. |

---

## Implementation sequence

### Step 1 — `src/dataset_io.py` (new file)

This module owns reading datasets and writing results, so `__main__.py` stays
thin. (Don't name it `datasets.py`: that name is easy to confuse with the
Hugging Face `datasets` package.)

```python
"""Read question datasets and write search results as JSON."""

from pathlib import Path

from src.models import RagDataset, StudentSearchResults


def load_dataset(dataset_path: Path) -> RagDataset:
    """Parse and validate the dataset JSON file at `dataset_path`.

    Raises:
        FileNotFoundError: If `dataset_path` is not a file.
        pydantic.ValidationError: If the JSON does not match `RagDataset`.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as handle:
        return RagDataset.model_validate_json(handle.read())


def save_search_results(
    results: StudentSearchResults, save_directory: Path, file_name: str
) -> Path:
    """Write `results` to `save_directory/file_name` and return that path."""
    save_directory.mkdir(parents=True, exist_ok=True)
    out_path = save_directory / file_name
    with out_path.open(mode="w", encoding="utf-8", newline="\n") as handle:
        handle.write(results.model_dump_json(indent=2))
    return out_path
```

Notes:

- `model_validate_json` does the JSON parsing *and* the validation. Malformed
  JSON raises `ValidationError`, not `json.JSONDecodeError`. Phase 14 relies on
  that: it only has one exception type to catch here.
- `RagDataset.rag_questions` is a `Union[AnsweredQuestion,
  UnansweredQuestion]`, so this loader accepts **both** folders. That matters,
  because the defense may point `--dataset_path` at either one.
- Both functions use a context manager for the file handle (N7).

### Step 2 — `Retriever.search_many` in `src/retriever.py`

Add a batch method, then make `search` a one-line wrapper around it. Afterwards
there is **one** ranking implementation. If single and batch search had separate
copies, they would slowly drift apart, and `search` would show you results that
`search_dataset` doesn't produce.

Add `BATCH_SIZE` near the imports:

```python
# Queries scored per matrix multiply; bounds the dense score matrix to
# n_chunks x BATCH_SIZE floats regardless of dataset size.
BATCH_SIZE = 64
```

Replace the `search` method with these two methods:

```python
    def search(self, query: str, k: int = 10) -> List[ScoredSource]:
        """Top-*k* sources for *query*, best first."""
        return self.search_many([query], k)[0]

    def search_many(
        self, queries: List[str], k: int = 10
    ) -> List[List[ScoredSource]]:
        """Top-*k* sources for each query, in the same order as *queries*.

        An empty query, a non-positive *k*, or a query with no known
        vocabulary term gets an empty list; the other queries are unaffected.
        """
        results: List[List[ScoredSource]] = [[] for _ in queries]
        if k <= 0:
            return results

        # Positions of the queries worth scoring at all.
        live = [i for i, q in enumerate(queries) if q.strip()]

        for start in tqdm(
            range(0, len(live), BATCH_SIZE),
            desc="Searching",
            unit="batch",
            disable=len(live) <= BATCH_SIZE,
        ):
            batch = live[start:start + BATCH_SIZE]
            vectors = self.vectorizer.transform([queries[i] for i in batch])
            scores = np.asarray((self.matrix @ vectors.T).todense())
            for column, query_index in enumerate(batch):
                if vectors[column].nnz == 0:
                    continue
                results[query_index] = self._top_k(scores[:, column], k)
        return results

    def _top_k(self, scores: Any, k: int) -> List[ScoredSource]:
        """Rank one score column, best first, dropping non-positive scores."""
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        ranked: List[ScoredSource] = []
        for position in top:
            if scores[position] <= 0.0:
                break
            source = self.sources[int(position)]
            ranked.append(
                ScoredSource(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                    score=float(scores[position]),
                )
            )
        return ranked
```

And add the import at the top:

```python
from tqdm import tqdm
```

Why it's written this way:

- `results` is filled in **by position**. Empty queries are skipped, so batch
  positions don't match dataset positions. `live` maps one to the other. The
  output must keep every question, in dataset order, even when its source list
  is empty.
- `disable=len(live) <= BATCH_SIZE` hides the progress bar for single `search`
  calls, so `search` output stays clean. It still shows for a real dataset
  (F13).
- `_top_k` is your old ranking body, moved without changes.

### Step 3 — the `search_dataset` command in `src/__main__.py`

Add the imports:

```python
from src.dataset_io import load_dataset, save_search_results
from src.models import MinimalSearchResults, MinimalSource, StudentSearchResults
```

Add this method to `Cli`, after `search`:

```python
    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Search every question in *dataset_path*; write StudentSearchResults."""
        k = int(k)
        dataset_file = Path(str(dataset_path))
        dataset = load_dataset(dataset_file)
        questions = dataset.rag_questions
        print(f"Loaded {len(questions)} questions from {dataset_file}")

        retriever = Retriever.load(Path(processed_dir))
        ranked = retriever.search_many([q.question for q in questions], k)

        results = StudentSearchResults(
            k=k,
            search_results=[
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=[
                        MinimalSource(
                            file_path=s.file_path,
                            first_character_index=s.first_character_index,
                            last_character_index=s.last_character_index,
                        )
                        for s in sources
                    ],
                )
                for question, sources in zip(questions, ranked)
            ],
        )
        out_path = save_search_results(
            results, Path(str(save_directory)), dataset_file.name
        )
        print(f"Saved student_search_results to {out_path.as_posix()}")
```

Notes:

- `dataset_path` has **no default**. There is no sensible default dataset, and
  F12 forbids hard-coded input paths. `save_directory` does have a default, but
  it is relative, so the evaluator can always override it.
- `k=k` in the output is the `k` that was *requested*, even when a question
  gets fewer sources. The subject defines `k` as "number of results requested".
- The `MinimalSource(...)` conversion is where Concept 2's decision happens:
  the score is dropped here.
- The final print matches the subject's walkthrough text (§VI.7.2 step 2).

### Step 4 — extend `tests/test_retriever.py` with batch tests

`tests/test_retriever.py` already exists (it was added when Phase 6 was closed)
and has a `retriever` fixture: a tiny index built by the real `Indexer`. Run it
**before** Step 2 and see 6 passed. After Step 2, append these three tests to
the end of the file. The six existing tests must still pass without changes.

```python
def test_search_many_matches_search_one_by_one(retriever: Retriever) -> None:
    queries = ["lora adapters", "fp8 quantization", "openai server port"]
    batched = retriever.search_many(queries, k=3)
    assert batched == [retriever.search(q, k=3) for q in queries]


def test_search_many_keeps_order_and_length_with_empty_queries(
    retriever: Retriever,
) -> None:
    queries = ["", "lora adapters", "zzzzunknownzzzz", "fp8 quantization"]
    batched = retriever.search_many(queries, k=3)
    assert len(batched) == 4
    assert batched[0] == [] and batched[2] == []
    assert batched[1][0].file_path == "data/raw/lora.md"
    assert batched[3][0].file_path == "data/raw/quant.md"


def test_search_many_crosses_a_batch_boundary(
    retriever: Retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.retriever.BATCH_SIZE", 2)
    queries = ["lora adapters", "fp8 quantization", "openai server port"] * 3
    batched = retriever.search_many(queries, k=1)
    assert [r[0].file_path for r in batched] == [
        "data/raw/lora.md", "data/raw/quant.md", "data/raw/server.py"
    ] * 3
```

`test_search_many_matches_search_one_by_one` is the proof that batching
changed nothing. `test_search_many_crosses_a_batch_boundary` covers the one bug
that only appears when a dataset is bigger than one batch.

### Step 5 — `tests/test_dataset_io.py` (new file)

```python
"""Unit tests for dataset loading and result writing."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.dataset_io import load_dataset, save_search_results
from src.models import MinimalSearchResults, MinimalSource, StudentSearchResults

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_load_dataset_reads_the_real_public_datasets() -> None:
    for scope in ("UnansweredQuestions", "AnsweredQuestions"):
        path = REPO_ROOT / "data" / "datasets" / scope / "dataset_docs_public.json"
        if not path.is_file():
            pytest.skip(f"{path} not present")
        assert len(load_dataset(path).rag_questions) == 100


def test_load_dataset_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.json")


def test_load_dataset_malformed_json_raises_validation_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_dataset(bad)


def test_save_search_results_round_trips(tmp_path: Path) -> None:
    results = StudentSearchResults(
        k=5,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="How?",
                retrieved_sources=[
                    MinimalSource(
                        file_path="data/raw/a.md",
                        first_character_index=0,
                        last_character_index=10,
                    )
                ],
            )
        ],
    )
    out = save_search_results(results, tmp_path / "out" / "Scope", "d.json")

    assert out == tmp_path / "out" / "Scope" / "d.json"
    assert StudentSearchResults.model_validate_json(out.read_text()) == results
    source = json.loads(out.read_text())["search_results"][0]["retrieved_sources"][0]
    assert set(source) == {
        "file_path", "first_character_index", "last_character_index"
    }
```

The last assertion pins Concept 2's decision: no `score` field ends up in the
graded file.

---

## Resulting project tree (changes only)

```
src/
├── __main__.py          (+ search_dataset, flake8 fix)
├── dataset_io.py        NEW
└── retriever.py         (search → wrapper, + search_many, _top_k, BATCH_SIZE)
tests/
├── test_dataset_io.py   NEW
└── test_retriever.py    (+ 3 batch tests)
```

---

## Commands to run

From the project root, in order:

```powershell
uv run pytest -q tests/test_retriever.py        # before Step 2: 6 passed
uv run pytest -q
uv run python -m src search "How do I load a LoRA adapter?" --k 5
Measure-Command { uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions | Out-Default }
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

---

## Expected output

**`search`** — the same five lines as before the refactor, in the same order
with the same scores. Here is the Phase 6 output recorded on 2026-09-15:

```
 1  data/raw/vllm-0.10.1/vllm/plugins/lora_resolvers/README.md  [0-830]  0.2602
 2  data/raw/vllm-0.10.1/docs/features/lora.md  [3400-5400]  0.2466
 3  data/raw/vllm-0.10.1/vllm/entrypoints/openai/serving_models.py  [10200-11558]  0.2371
 4  data/raw/vllm-0.10.1/vllm/entrypoints/openai/serving_models.py  [8500-10500]  0.2353
 5  data/raw/vllm-0.10.1/docs/features/lora.md  [5100-7100]  0.2325
```

If any line changed, the refactor changed behaviour. Stop and compare `_top_k`
with the old `search` body.

**`search_dataset`** on the docs set:

```
Loaded 100 questions from data\datasets\UnansweredQuestions\dataset_docs_public.json
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

TotalSeconds      : 4.8...
```

100 questions is less than two batches of 64, so you will see a short
`Searching` progress bar with 2 steps. `TotalSeconds` should be far below 45
(half of the 90-second budget for 200 questions). Most of that time is starting
Python and loading the index, not searching.

**`flake8`** prints nothing. **`mypy`** reports success.
**`pytest`** reports **32 passed** (the 25 you have now, plus 3 batch tests in
`test_retriever.py` and 4 in `test_dataset_io.py`).

---

## Manual verification

1. Open `data/output/search_results/UnansweredQuestions/dataset_docs_public.json`.
   Check that the top level has exactly `search_results` and `k`, that
   `search_results` has 100 entries, and that the first `question_id` is the same
   as the first one in the input dataset.
2. Check the whole file for **backslashes** (R7). This should print `0`:
   `(Select-String -Path data\output\search_results\UnansweredQuestions\*.json -Pattern '\\\\' -SimpleMatch).Count`
3. Check the **2000-character rule** (F6) on both files. This should print
   `max width: 2000` or less:
   ```powershell
   uv run python -c "import json,glob; w=[s['last_character_index']-s['first_character_index'] for f in glob.glob('data/output/search_results/UnansweredQuestions/*.json') for r in json.load(open(f))['search_results'] for s in r['retrieved_sources']]; print('max width:', max(w))"
   ```
4. Run `search_dataset` on the **AnsweredQuestions** copy of the docs dataset,
   with `--save_directory data/output/search_results/AnsweredQuestions`. It must
   work (the loader accepts both), and its `retrieved_sources` must be the same
   as in the Unanswered run. The questions are the same, so the ranking must be
   the same.
5. Run it with `--k 3` and check that `"k": 3` is in the file and that no
   question has more than 3 sources.

## Automated tests

`test_retriever.py` protects the ranking. It checks that batch and single
search agree, that order and length hold when some queries are empty, and that
results stay correct across a batch boundary. `test_dataset_io.py` protects the
contract. It checks that the real datasets parse, that bad input raises the
expected exception types, and that the written file has only the three graded
source fields.

Do not add a test that runs `search_dataset` on the full real index. It needs a
built index, takes seconds, and checks nothing that the unit tests and the
manual steps above don't already check.

---

## Most likely errors

**1. The output file has fewer entries than the dataset.**
*Cause:* results were built only for the non-empty queries, or with
`zip(live, ...)` instead of `zip(questions, ranked)`.
*Fix:* `search_many` must return exactly `len(queries)` lists.
`test_search_many_keeps_order_and_length_with_empty_queries` catches this.

**2. Right questions, wrong sources — every result looks shifted.**
*Cause:* inside the batch loop, `results[column] = ...` was written instead of
`results[query_index] = ...`. Everything after the first batch or the first
empty query gets the wrong question's sources.
*Fix:* index `results` with `query_index`.
`test_search_many_crosses_a_batch_boundary` catches this.

**3. `ValueError: kth(=-1) out of bounds` or `kth out of bounds`.**
*Cause:* `_top_k` was called with `k=0`, or the `k <= 0` guard was moved below
the loop.
*Fix:* keep the `if k <= 0: return results` check as the first thing in
`search_many`.

**4. `mypy: Returning Any from function declared to return "RagDataset"`.**
*Cause:* the loader used `json.load` and returned `RagDataset(**data)` through
an untyped dict, or it returned the result of a function typed `Any`.
*Fix:* use `RagDataset.model_validate_json(...)` exactly as in Step 1. It is
typed to return `RagDataset`.

**5. The file lands in `data/output/search_results/dataset_docs_public.json`
without the scope folder.**
*Cause:* `--save_directory` was left out, so the default was used.
*Fix:* always pass the scoped directory (§VI.7.2 step 2). Both public scopes use
the same file names, so an unscoped run overwrites the other scope's result.

---

## Definition of done

- [ ] `uv run flake8 .` prints nothing.
- [ ] The 6 existing tests in `tests/test_retriever.py` still pass **after** the
      refactor, without being edited.
- [ ] `Retriever.search` is a one-line wrapper around `search_many`.
- [ ] `search` output for the LoRA question matches the recorded Phase 6 output.
- [ ] `search_dataset` writes both public datasets into
      `data/output/search_results/UnansweredQuestions/`.
- [ ] Each output file has one entry per input question, in input order, with
      `k` equal to the requested `k`.
- [ ] No `file_path` has a backslash; no source is wider than 2000 characters.
- [ ] Each `search_dataset` run takes well under 45 seconds.
- [ ] `mypy` is clean with the five flags; `pytest -q` reports 32 passed.

**Natural commit point:** "search_dataset: batch retrieval writing
StudentSearchResults".

---

## What Phase 8 adds and why

Phase 8 adds `evaluate`. It loads the file this phase wrote together with the
matching `AnsweredQuestions` dataset. For each question it checks whether any
retrieved source is in the same file as the ground-truth span and overlaps it
with IoU ≥ 0.05. It then reports recall@1/3/5/10 for docs and code. This number
is the **baseline** that Phases 9 and 10 must beat, and it replaces the
moulinette, which cannot run on this Windows machine (R1).
