# Phase 2 — Measure your own recall

**Goal:** Run search over a whole dataset, write the graded JSON, and print your own recall@k — a number you can move.
**Time:** ~3h · **Difficulty:** ●●○○○
**Depends on:** Phase 1 complete.

## ✅ What you'll have when this is done

The measurement loop. `search_dataset` turns 100 questions into a `StudentSearchResults` file the moulinette accepts, and `evaluate` scores that file against the ground truth using the same metric the grader uses. From here every change you make has a number attached to it.

```bash
$ uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
Searching: 100%|██████████████████| 100/100 [00:03<00:00, 31.9q/s]
Searched 100 questions in 3.1s
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Questions scored: 100
Recall@1: 0.290  Recall@3: 0.430  Recall@5: 0.510  Recall@10: 0.620
```

Those numbers are illustrative of what a Phase-1 chunker and a plain TF-IDF index typically produce: well under the 0.80 docs bar. **That is the point of this phase** — you now have a baseline to beat, and Phase 3 is where the beating happens. If your numbers come out higher, good; if lower, also fine. What matters is that the number exists and moves when you change something.

## Where you're starting from

```
rag_against_the_machine/
├── pyproject.toml · uv.lock · setup.cfg · Makefile   # phase 1
├── src/
│   ├── __main__.py     # Cli with index, search
│   ├── models.py · corpus.py · chunking.py
│   ├── indexer.py · retriever.py
├── tests/test_chunking.py · tests/test_models.py
└── data/
    ├── raw/vllm-0.10.1/
    ├── processed/{chunks.jsonl,tfidf.joblib,meta.json}
    └── datasets/{Answered,Unanswered}Questions/*.json
```

`index` and `search` work. You can retrieve, but you have no idea whether what you retrieve is right.

## Why this phase now

Phase 3 is a tuning phase, and tuning without a metric is guessing with extra steps. Building the measurement before the improvements also front-loads the second-riskiest unknown in the project after path correctness: **does the moulinette accept your JSON at all?** A file the grader rejects scores zero regardless of retrieval quality, and you want to discover that today, not the night before the defense.

## Before you start

No new Python dependencies. What you do need is a way to run the moulinette, which ships as Linux ELF binaries while you are on Windows.

**Route A — Docker (recommended).** `wsl --list` shows you already have `docker-desktop` installed, so this needs no new install and no reboot:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd)":/w -w /w ubuntu:22.04 \
  bash -c "chmod +x moulinette && ./moulinette --help"
```

`MSYS_NO_PATHCONV=1` stops Git Bash from rewriting `/w` into a Windows path — without it the mount silently lands somewhere useless. Wrap the invocation in a shell function so you stop retyping it:

```bash
# add to ~/.bashrc
moulinette() {
  MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd)":/w -w /w ubuntu:22.04 \
    bash -c "chmod +x ./moulinette && ./moulinette $*"
}
```

**Route B — WSL.** `wsl --install -d Ubuntu` (needs admin and a reboot), then copy the binary into the Linux filesystem so the exec bit sticks:

```bash
wsl -d Ubuntu -- bash -c "cp /mnt/c/Users/user/Desktop/my_projects/python/rag_against_the_machine/moulinette ~/moulinette && chmod +x ~/moulinette && ~/moulinette --help"
```

**Route C — a campus Linux machine.** Always works, but the feedback loop is minutes instead of seconds, which is why `evaluate` exists.

If none of the three is available today, build `evaluate` anyway and get the moulinette confirmation before you finish Phase 3. Do not let it slip past that.

## Key design decisions

- **Whose metric do you optimise against?** Only the moulinette, only your own `evaluate`, or both. Recommendation: **build `evaluate` as a faithful re-implementation and use it for the fast loop, then confirm with the real moulinette at every checkpoint.** The subject requires an `evaluate` command anyway. The risk of the two disagreeing is real and it is named in the debt ledger; the mitigation is that a checkpoint is one command away.

- **IoU definition.** The subject says "overlaps its character range" with an IoU of 0.05, which leaves the denominator ambiguous — intersection over union of the two spans, or intersection over the truth span. Recommendation: **intersection over union**, the standard reading and the stricter one. If your `evaluate` is stricter than the grader you are pleasantly surprised at defense; the reverse is how people fail.

- **Where the output filename comes from.** A fixed name, a timestamp, or the input dataset's basename. Recommendation: **the input basename**, into a `--save_directory` scoped by dataset kind. The subject's walkthrough shows exactly this, and it is why the scoping matters: `dataset_docs_public.json` exists under both `AnsweredQuestions/` and `UnansweredQuestions/`, so one flat output directory would have the two runs overwriting each other.

- **What `search_dataset` writes for the score field.** Keep the retrieval score in the JSON as an extra field, or strip it. Recommendation: **strip it.** The subject permits extra fields, but "permits" is not "the grader was tested with them", and the score buys you nothing in a file you only feed to a scorer. Keep it in memory, in `ScoredSource`, where it is useful for debugging.

- **How many questions to search at once.** All of them, or a `--limit` for the dev loop. Recommendation: **all of them.** 100 questions take about 3 seconds at Phase 1 index size; a limit flag here would only hide the perf budget from you. Phase 4 introduces `--limit`, where a run takes 30 minutes and it earns its place.

## Debt taken on

Shortcut: `evaluate` re-implements the grader's metric rather than calling it. Bites you when: your number and the moulinette's disagree, and you have spent a day tuning against the wrong one. Paid off in: **not planned** — mitigated by running the real moulinette at every Phase 3 checkpoint and by keeping `span_iou` to the ten obvious lines, where a bug has nowhere to hide.

## Files in this phase

| File | New/Edit | What it holds |
|---|---|---|
| `src/datasets.py` | new | `load_dataset`, `load_questions`, `save_search_results`, `load_search_results` |
| `src/evaluation.py` | new | `span_iou`, `truth_by_id`, `recall_at_k`, `recall_report` |
| `src/__main__.py` | edit | Adds `search_dataset` and `evaluate` |
| `tests/test_datasets.py` | new | The written JSON has exactly the graded shape |
| `tests/test_evaluation.py` | new | IoU boundaries and the recall arithmetic |
| `benchmarks.md` | new | One row per experiment, starting with today's baseline |

## Steps

### 1. Read the datasets and write the graded JSON

**Why:** The output file is the deliverable the grader reads. Everything about it — the filename, the field set, the nesting — is fixed by the subject, so put it behind two functions and never hand-roll a `json.dump` anywhere else.

Create `src/datasets.py`. `load_dataset` validates the raw JSON through `RagDataset`; `load_questions` flattens it to the list `search_dataset` iterates; `to_minimal` is the one-way door that strips a `ScoredSource` back down to the three graded fields; `save_search_results` writes the file and returns where it went.

```python
# src/datasets.py
"""Read question datasets and write the graded result files."""

import json
from pathlib import Path
from typing import List

from src.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)


def load_dataset(dataset_path: Path) -> RagDataset:
    """Parse and validate a RagDataset JSON file.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file is not valid JSON or not a RagDataset.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{dataset_path} is not valid JSON: {exc}") from exc
    return RagDataset.model_validate(payload)


def load_questions(dataset_path: Path) -> List[UnansweredQuestion]:
    """Every question in *dataset_path*, answered ones included."""
    questions: List[UnansweredQuestion] = []
    for question in load_dataset(dataset_path).rag_questions:
        questions.append(question)
    return questions


def to_minimal(source: MinimalSource) -> MinimalSource:
    """Copy *source* keeping only the three fields the grader reads."""
    return MinimalSource(
        file_path=source.file_path,
        first_character_index=source.first_character_index,
        last_character_index=source.last_character_index,
    )


def save_search_results(
    results: StudentSearchResults,
    save_directory: Path,
    dataset_path: Path,
) -> Path:
    """Write *results* as ``<save_directory>/<dataset basename>``.

    Returns:
        The path written.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = save_directory / dataset_path.name
    cleaned = StudentSearchResults(
        k=results.k,
        search_results=[
            MinimalSearchResults(
                question_id=row.question_id,
                question=row.question,
                retrieved_sources=[
                    to_minimal(source) for source in row.retrieved_sources
                ],
            )
            for row in results.search_results
        ],
    )
    output_path.write_text(
        cleaned.model_dump_json(indent=2), encoding="utf-8"
    )
    return output_path


def load_search_results(path: Path) -> StudentSearchResults:
    """Parse a StudentSearchResults file written by search_dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"search results not found: {path}")
    with path.open(encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return StudentSearchResults.model_validate(payload)
```

`to_minimal` looks redundant — pydantic would drop the extra field on serialization anyway. It is here because "would drop it anyway" depends on `model_config`, on the pydantic minor version, and on whether a future field is optional; rebuilding the object makes the guarantee structural rather than incidental. It also means `save_search_results` cannot accidentally leak whatever you attach to a source while debugging Phase 3.

**Check:**

```bash
uv run python -c "
from pathlib import Path
from src.datasets import load_questions
qs = load_questions(Path('data/datasets/UnansweredQuestions/dataset_docs_public.json'))
print(len(qs), qs[0].question_id, qs[0].question[:50])
"
```
prints `100 17526382-7764-4120-b5e8-2d3726b8a4da What HTTP endpoint is used to dynamically load a Lo`.

### 2. Implement the metric

**Why:** This is what turns Phase 3 from opinion into engineering. Ten lines, and every decision you make for the next six hours leans on them.

Create `src/evaluation.py`. `span_iou` is intersection-over-union with a hard file-path gate; `truth_by_id` indexes the ground truth for O(1) lookup; `recall_at_k` averages the per-question share of found sources; `recall_report` runs the four k values the moulinette prints.

```python
# src/evaluation.py
"""recall@k over retrieved source spans, faithful to the moulinette."""

from pathlib import Path
from typing import Dict, List, Sequence

from src.datasets import load_dataset
from src.models import AnsweredQuestion, MinimalSource, StudentSearchResults

#: A retrieved span counts as a hit at or above this IoU with the truth.
IOU_THRESHOLD = 0.05

#: The k values the moulinette reports.
DEFAULT_KS = (1, 3, 5, 10)


def span_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Intersection over union of two spans, 0.0 across different files."""
    if a.file_path != b.file_path:  # ← different file is never a hit
        return 0.0
    overlap = min(a.last_character_index, b.last_character_index) - max(
        a.first_character_index, b.first_character_index
    )
    if overlap <= 0:
        return 0.0
    union = a.width + b.width - overlap
    return overlap / union if union > 0 else 0.0


def truth_by_id(dataset_path: Path) -> Dict[str, List[MinimalSource]]:
    """Map question_id to its reference sources.

    Raises:
        ValueError: If the dataset carries no answered questions, which
            means you pointed this at UnansweredQuestions by mistake.
    """
    truth: Dict[str, List[MinimalSource]] = {}
    for question in load_dataset(dataset_path).rag_questions:
        if isinstance(question, AnsweredQuestion):
            truth[question.question_id] = list(question.sources)
    if not truth:
        raise ValueError(
            f"{dataset_path} has no answered questions - "
            "point --dataset_path at AnsweredQuestions/"
        )
    return truth


def recall_at_k(
    results: StudentSearchResults,
    truth: Dict[str, List[MinimalSource]],
    k: int,
) -> float:
    """Mean over questions of the share of true sources found in top-k."""
    per_question: List[float] = []
    for row in results.search_results:
        expected = truth.get(row.question_id)
        if not expected:
            continue
        retrieved = row.retrieved_sources[:k]
        found = sum(
            1
            for reference in expected
            if any(
                span_iou(reference, got) >= IOU_THRESHOLD for got in retrieved
            )
        )
        per_question.append(found / len(expected))
    if not per_question:
        raise ValueError(
            "no question_id in common between the results and the dataset"
        )
    return sum(per_question) / len(per_question)


def recall_report(
    results: StudentSearchResults,
    truth: Dict[str, List[MinimalSource]],
    ks: Sequence[int] = DEFAULT_KS,
) -> Dict[int, float]:
    """recall@k for each k, in one pass per k."""
    return {k: recall_at_k(results, truth, k) for k in ks}
```

Two things worth saying out loud. The file-path gate is a `!=` on the exact string, which is why Phase 1 spent a whole step on path spelling — this is the line that turns a backslash into a zero. And `found / len(expected)` is per-question averaging, not a global source count; with exactly one source per question in both public datasets the two coincide, but they diverge the moment a hidden dataset carries multi-source questions, and the subject's wording ("the share of its correct sources that you retrieve") says per-question.

**Check:**

```bash
uv run python -c "
from src.evaluation import span_iou
from src.models import MinimalSource
def s(p, a, b): return MinimalSource(file_path=p, first_character_index=a, last_character_index=b)
print(span_iou(s('a', 0, 1000), s('a', 0, 1000)))        # 1.0
print(span_iou(s('a', 0, 1000), s('b', 0, 1000)))        # 0.0 - other file
print(round(span_iou(s('a', 0, 100), s('a', 0, 2000)), 3))  # 0.05 - exactly the bar
print(span_iou(s('a', 0, 100), s('a', 100, 200)))        # 0.0 - touching, not overlapping
"
```
prints `1.0`, `0.0`, `0.05`, `0.0`. That third line is the ceiling calculation from the roadmap README made concrete: a 100-character truth span inside a 2000-character chunk lands exactly on the threshold.

### 3. Add `search_dataset` to the CLI

**Why:** This is the command the automated defense run invokes. Its output file is the entire retrieval grade.

In `src/__main__.py`, add the imports and the `search_dataset` method. The retriever is loaded **once**, outside the loop — that is the whole perf budget in one decision.

```python
# src/__main__.py
# ... existing imports ...
import time

from tqdm import tqdm

from src.datasets import load_questions, save_search_results
from src.evaluation import recall_report, truth_by_id
from src.models import MinimalSearchResults, StudentSearchResults

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = str(REPO_ROOT / "data" / "raw")
DEFAULT_PROCESSED_DIR = str(REPO_ROOT / "data" / "processed")
DEFAULT_SEARCH_OUT = str(REPO_ROOT / "data" / "output" / "search_results")


class Cli:
    # ... index() and search() unchanged ...

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = DEFAULT_SEARCH_OUT,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Search every question in a dataset and write the results file."""
        questions = load_questions(Path(dataset_path))
        retriever = Retriever.load(Path(processed_dir))  # ← once, not per query
        started = time.perf_counter()
        rows = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=list(
                    retriever.search(question.question, int(k))
                ),
            )
            for question in tqdm(questions, desc="Searching", unit="q")
        ]
        elapsed = time.perf_counter() - started
        output_path = save_search_results(
            StudentSearchResults(search_results=rows, k=int(k)),
            Path(save_directory),
            Path(dataset_path),
        )
        print(f"Searched {len(rows)} questions in {elapsed:.1f}s")
        print(f"Saved student_search_results to {output_path}")
```

The elapsed time is printed on purpose, every run. The subject budgets 200 questions in 90 seconds; seeing `3.1s` today and `41.0s` after Phase 3 tells you immediately whether a chunking change just cost you the budget.

**Check:**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 --save_directory data/output/search_results/UnansweredQuestions
uv run python -c "
import json
d = json.load(open('data/output/search_results/UnansweredQuestions/dataset_docs_public.json', encoding='utf-8'))
print(d['k'], len(d['search_results']))
print(sorted(d['search_results'][0]))
print(sorted(d['search_results'][0]['retrieved_sources'][0]))
"
```
prints `10 100`, then `['question', 'question_id', 'retrieved_sources']`, then `['file_path', 'first_character_index', 'last_character_index']`. Exactly three keys on the source — no `score`.

### 4. Add `evaluate` to the CLI

**Why:** The fast half of the loop. You will run this several hundred times in Phase 3.

Add the `evaluate` method below `search_dataset`. It takes the two paths in the order the subject specifies — your results first, the ground truth second — and prints the four recalls on one line, mirroring the moulinette's own output so you can diff them by eye.

```python
# src/__main__.py
# ... inside class Cli, after search_dataset ...

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Report recall@k of your results against a ground-truth dataset."""
        results = load_search_results(Path(student_search_results_path))
        truth = truth_by_id(Path(dataset_path))
        scored = sum(
            1 for row in results.search_results if row.question_id in truth
        )
        report = recall_report(results, truth)
        print(f"Questions scored: {scored}")
        print(
            "  ".join(
                f"Recall@{key}: {value:.3f}" for key, value in report.items()
            )
        )
```

Add `load_search_results` to the `src.datasets` import line at the top of the file.

`Questions scored` is printed before the recalls because a mismatched pair of files is the failure mode here: point `--dataset_path` at the wrong dataset and you get `Questions scored: 0` and a clear `ValueError` instead of a plausible-looking `Recall@5: 0.000` that costs you an hour.

**Check:**

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
prints `Questions scored: 100` and four recalls, monotonically non-decreasing from @1 to @10. If they are not monotonic, `recall_at_k` is slicing wrong.

### 5. Confirm against the real moulinette

**Why:** Your metric agreeing with itself proves nothing. The one thing you cannot afford to discover late is that the grader rejects your file outright.

```bash
# with the moulinette() function from "Before you start"
moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

The first line of its output is the one that matters:

```
Student data is valid: True
```

If it says `False`, the cause is almost always one of three things, in this order of likelihood: a `file_path` that does not match the corpus verbatim, a retrieved source wider than `max_context_length`, or a missing field. Check the widths first because it is a one-line test:

```bash
uv run python -c "
import json
d = json.load(open('data/output/search_results/UnansweredQuestions/dataset_docs_public.json', encoding='utf-8'))
w = [s['last_character_index'] - s['first_character_index']
     for r in d['search_results'] for s in r['retrieved_sources']]
print('max width', max(w), 'over 2000:', sum(1 for x in w if x > 2000))
"
```

If `max width` is exactly 2000 and the moulinette still says `False`, the grader is treating `last_character_index` as inclusive. The fix is one character: change `end = min(start + max_chunk_size, len(text))` to `min(start + max_chunk_size - 1, len(text))` in `chunk_fixed` and re-index. Chances are you will never need this — Phase 3 settles on a chunk size well below 2000 — but knowing the fix is one constant is worth the paragraph.

**Check:** `Student data is valid: True`, and the moulinette's `Recall@5` is within a couple of points of your `evaluate`. Record both numbers; a persistent gap is a bug in `span_iou`, not noise.

### 6. Start the benchmark log

**Why:** Phase 3 is a sequence of experiments. Without a written record you will lose track of which change bought which point, and the README needs this table anyway.

Create `benchmarks.md` at the repo root with today's baseline. One row per experiment, forever.

```markdown
<!-- benchmarks.md -->
# Retrieval benchmarks

All numbers from `uv run python -m src evaluate`, k=10, public datasets.
Moulinette-confirmed rows are marked ✓.

| # | Chunking | Retriever | Analyzer | Size | Docs R@5 | Code R@5 | Index | Search 200q | Note |
|---|----------|-----------|----------|------|----------|----------|-------|-------------|------|
| 0 | fixed | tfidf | default | 2000 | 0.510 ✓ | 0.180 ✓ | 41s | 6.2s | phase 1 baseline |

Target: docs ≥ 0.80, code ≥ 0.50 at k=5.
```

Fill row 0 with **your** numbers, not the ones above. Add a row for every single change in Phase 3 — one change per row, or the table tells you nothing.

**Check:** the file exists and row 0 holds real measured numbers from your own run.

### 7. Test the shape and the arithmetic

**Why:** These two files are the ones a later refactor breaks silently. A wrong `span_iou` does not crash; it just makes every number afterwards a lie.

```python
# tests/test_datasets.py
"""The written results file must have exactly the graded shape."""

import json
from pathlib import Path

import pytest

from src.datasets import load_search_results, save_search_results
from src.models import (
    MinimalSearchResults,
    ScoredSource,
    StudentSearchResults,
)


def _results() -> StudentSearchResults:
    return StudentSearchResults(
        k=2,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="How to configure the OpenAI server?",
                retrieved_sources=[
                    ScoredSource(
                        file_path="data/raw/vllm-0.10.1/docs/a.md",
                        first_character_index=0,
                        last_character_index=120,
                        score=0.93,
                    )
                ],
            )
        ],
    )


def test_written_file_drops_the_score(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset_docs_public.json"
    dataset.write_text("{}", encoding="utf-8")
    out = save_search_results(_results(), tmp_path / "out", dataset)

    assert out.name == "dataset_docs_public.json"
    payload = json.loads(out.read_text(encoding="utf-8"))
    source = payload["search_results"][0]["retrieved_sources"][0]
    assert set(source) == {
        "file_path",
        "first_character_index",
        "last_character_index",
    }
    assert payload["k"] == 2


def test_round_trip_through_load(tmp_path: Path) -> None:
    dataset = tmp_path / "d.json"
    dataset.write_text("{}", encoding="utf-8")
    out = save_search_results(_results(), tmp_path / "out", dataset)
    reloaded = load_search_results(out)
    assert reloaded.k == 2
    assert reloaded.search_results[0].question_id == "q1"


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_search_results(tmp_path / "nope.json")
```

```python
# tests/test_evaluation.py
"""IoU boundaries and the recall arithmetic."""

from typing import Dict, List

from src.evaluation import IOU_THRESHOLD, recall_at_k, span_iou
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def _source(path: str, first: int, last: int) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=first,
        last_character_index=last,
    )


def test_identical_spans_score_one() -> None:
    assert span_iou(_source("a", 0, 100), _source("a", 0, 100)) == 1.0


def test_different_file_never_matches() -> None:
    assert span_iou(_source("a", 0, 100), _source("b", 0, 100)) == 0.0


def test_touching_spans_do_not_overlap() -> None:
    assert span_iou(_source("a", 0, 100), _source("a", 100, 200)) == 0.0


def test_threshold_boundary() -> None:
    # 100 chars of truth inside a 2000-char chunk: IoU is exactly 0.05.
    assert span_iou(_source("a", 0, 100), _source("a", 0, 2000)) == IOU_THRESHOLD


def test_recall_counts_rank_position() -> None:
    truth: Dict[str, List[MinimalSource]] = {"q1": [_source("a", 0, 100)]}
    results = StudentSearchResults(
        k=3,
        search_results=[
            MinimalSearchResults(
                question_id="q1",
                question="?",
                retrieved_sources=[
                    _source("b", 0, 100),   # rank 1: wrong file
                    _source("a", 20, 120),  # rank 2: hit
                ],
            )
        ],
    )
    assert recall_at_k(results, truth, 1) == 0.0
    assert recall_at_k(results, truth, 3) == 1.0
```

**Check:** `uv run pytest -q` prints `17 passed`.

## Common pitfalls

| Pitfall | Why it happens | Avoid it by |
|---|---|---|
| Both datasets written into one output directory | `--save_directory data/output/search_results` looks tidy | Scope it: `.../search_results/UnansweredQuestions`; the two datasets share basenames and silently overwrite |
| `evaluate` pointed at `UnansweredQuestions/` | The two directories contain identically named files | `truth_by_id` raises with an explicit message — read it rather than re-running |
| The retriever reloaded once per question | The natural way to write the loop | Load outside it; this is the difference between 3 seconds and 20 minutes for 200 questions |
| Recall@1 higher than recall@5 | `recall_at_k` slicing the wrong end, or results not sorted by score | Assert monotonicity by eye every run; `Retriever.search` already sorts descending |
| Tuning against `evaluate` for a whole day without a moulinette check | It is faster and the number looks fine | Confirm with the real binary at the end of every work session |
| `Student data is valid: False` and no idea why | The message does not say which record failed | Check span widths first (one-liner in step 5), then paths, then fields |
| `MSYS_NO_PATHCONV` forgotten in Git Bash | Docker mounts silently land on a rewritten path | Use the shell function from *Before you start*, never the raw command |
| The baseline row never gets written | It feels like admin, not progress | Write row 0 before you start Phase 3; a tuning phase without a baseline has no story to tell |

## Verify it's done

```bash
for scope in docs code; do
  uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_${scope}_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
  uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_${scope}_public.json
done
uv run pytest -q
```

Expected — your recalls will differ, the shape must not:

```
Searching: 100%|██████████████████| 100/100 [00:03<00:00, 31.9q/s]
Searched 100 questions in 3.1s
Saved student_search_results to .../UnansweredQuestions/dataset_docs_public.json
Questions scored: 100
Recall@1: 0.290  Recall@3: 0.430  Recall@5: 0.510  Recall@10: 0.620
Searching: 100%|██████████████████| 99/99 [00:03<00:00, 32.4q/s]
Searched 99 questions in 3.0s
Saved student_search_results to .../UnansweredQuestions/dataset_code_public.json
Questions scored: 99
Recall@1: 0.080  Recall@3: 0.140  Recall@5: 0.180  Recall@10: 0.260
17 passed
```

And the grader agrees the file is well-formed:

```bash
moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```
```
Student data is valid: True
```

Bad input, which must not produce a traceback yet — a clear exception message is acceptable in this phase, Phase 5 makes it graceful:

```bash
$ uv run python -m src evaluate --student_search_results_path nope.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
FileNotFoundError: search results not found: nope.json
```

## Definition of done

- [ ] `search_dataset` writes `<save_directory>/<dataset basename>.json`
- [ ] Retrieved sources carry exactly three fields — no `score`
- [ ] The moulinette reports `Student data is valid: True` on both datasets
- [ ] `evaluate` prints `Recall@1/3/5/10`, monotonically non-decreasing
- [ ] Your `evaluate` and the moulinette agree within a couple of points
- [ ] 200 questions search in well under 90 seconds, and the time is printed
- [ ] `benchmarks.md` row 0 holds real measured baseline numbers
- [ ] `uv run pytest -q` passes
- [ ] Committed

## Deliberately NOT in this phase

- Improving the numbers → **Phase 3**. Resist. Measure first, then change one thing at a time.
- Python-AST / Markdown chunking, BM25, the identifier analyzer → **Phase 3**
- Answer generation → **Phase 4**
- `--limit` for partial runs → **Phase 4**, where a run costs 30 minutes
- Graceful CLI errors on every command → **Phase 5**
- All five bonuses → not in v1

## Commit

```bash
git add -A
git commit -m "phase 2: dataset search, graded JSON output and recall@k"
```

## Next

→ **[Phase 3 — Beat the recall thresholds](PHASE-03-beat-the-recall-thresholds.md)**
