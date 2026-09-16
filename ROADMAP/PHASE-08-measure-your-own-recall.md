# Phase 8 — Measure your own recall

**Goal:** add the `evaluate` command. It compares a `search_dataset` output file
against the matching `AnsweredQuestions` file and prints recall@1/3/5/10, so
every later change can be judged by a number instead of by opinion.

---

## What the system can do after this phase that it cannot do now

You have four valid result files, and no idea whether they are any good. The
official score comes from the moulinette, a Linux binary that cannot run on this
Windows machine (risk R1). Until `evaluate` exists, "did that change help?" is
unanswerable.

This is the **measurement gate** of the whole roadmap. Phases 9, 10 and 11 are
each justified by the number this phase produces, and the subject explicitly
provides for it: *"evaluate --student_search_results_path <path> --dataset_path
<path>: Report your own recall@k against a ground-truth dataset, for your own
testing"* **(§VI.6)**.

## Prerequisites

- Phase 7 is done and verified (2026-09-15): `search_dataset` wrote all four
  files under `data/output/search_results/`, 32 tests pass, flake8 and mypy are
  clean.
- `data/datasets/AnsweredQuestions/` holds the two `_public.json` answer keys.

---

## Concepts introduced in this phase

### Concept 1 — recall@k

**The problem.** You need one number that says "how often does my top-k contain
the right answer". It has to be the *same* number the moulinette computes, or
tuning against it makes things worse, not better.

**What solves it.** The subject defines it precisely **(§VII.1.1)**:

> For each question, recall@k is the share of its correct sources that you
> retrieve in your top-k results. A correct source counts as found when one of
> your results is in the same file and overlaps its character range.

So it is computed **per question** and then averaged over questions:

```
recall@k for one question = (correct sources found in your top k) / (its correct sources)
reported recall@k         = mean of that over every question in the dataset
```

**Tiny concrete example.** A question has 2 correct sources. Your top 5 covers
one of them. That question scores 0.5. Another question with 1 correct source
that you find scores 1.0. Reported recall@5 over those two questions is 0.75.

**A fact that simplifies this dataset, but do not hard-code it.** I checked both
public answer keys: **every question has exactly 1 correct source** (min 1,
max 1, across all 199 questions). So here the score per question is only ever
0.0 or 1.0, and recall@k is a hit rate. Write the general formula anyway — the
private datasets at the defense may differ, and the general form costs one extra
line.

**Why now.** Phases 9 and 10 are changes to retrieval quality. Without this,
they cannot be evaluated.

**What it costs.** Nothing at runtime; it is arithmetic over JSON files already
on disk.

### Concept 2 — IoU (Intersection over Union) on character spans

**The problem.** Your chunk boundaries never line up with the reference span. If
"correct" meant "identical span", you would score ~0 forever. So how much
overlap is enough?

**What solves it.** IoU measures how much two ranges share, relative to the
total they cover together:

```
intersection = max(0, min(a_last, b_last) - max(a_first, b_first))
union        = (a_last - a_first) + (b_last - b_first) - intersection
IoU          = intersection / union          (0.0 when union is 0)
```

The subject sets the bar at **IoU ≥ 0.05** **(§VII.1.1)**: *"The overlap bar is
low (an IoU of 0.05), so you do not need to match the reference span exactly."*

**Tiny concrete example.** Reference span `[4695, 6098)` is 1403 characters.
Your chunk `[5100, 7100)` is 2000 characters.
- intersection = `min(6098, 7100) - max(4695, 5100)` = `6098 - 5100` = 998
- union = 1403 + 2000 − 998 = 2405
- IoU = 998 / 2405 = **0.415** — far above 0.05, so this counts as found.

To *fail* the bar with a 2000-character chunk against that reference, the
overlap would have to be under about 160 characters, i.e. the chunk barely
clips the edge of the answer **[arithmetic from the formula above]**.

**Why now.** It is the other half of the definition of "correct".

**What it costs.** One subtlety: spans are **half-open**, `[first, last)`, the
same convention `text[first:last]` uses. Mixing that up with an inclusive
`last` puts an off-by-one in every comparison. It will rarely change a verdict
at a 0.05 bar, but it makes your number differ from the moulinette's for the
wrong reason.

---

## Concepts deliberately deferred

| Concept | Deferred to | Why not now |
|---|---|---|
| Structure-aware chunking (Python `ast`, Markdown headings) | Phase 9 | It is judged by the number this phase produces. |
| Identifier-aware tokenisation | Phase 10 | Same. |
| BM25 | Phase 11 (conditional) | Held in reserve; see the note at the end. |
| Evaluating *answers* rather than sources | Phase 13+ | Nothing generates answers yet. |

---

## Implementation sequence

### Step 1 — `src/evaluation.py` (new file)

Call it `evaluation.py`, not `evaluate.py`: the CLI method is already named
`evaluate`, and keeping the two names distinct avoids confusion when reading
imports.

```python
"""Recall@k of a search-results file against a ground-truth dataset."""

from typing import Dict, List, Sequence

from src.models import AnsweredQuestion, MinimalSource, RagDataset

# The subject's overlap bar (VII.1.1): a retrieved source counts as correct
# when it is in the same file and its IoU with the reference span is >= this.
IOU_THRESHOLD = 0.05

# The values of k reported by the moulinette (VI.7.2).
REPORTED_K = (1, 3, 5, 10)


def iou(first_a: int, last_a: int, first_b: int, last_b: int) -> float:
    """Intersection over union of two half-open character spans."""
    intersection = max(0, min(last_a, last_b) - max(first_a, first_b))
    union = (last_a - first_a) + (last_b - first_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def is_found(
    reference: MinimalSource, retrieved: Sequence[MinimalSource]
) -> bool:
    """True when some retrieved source covers `reference` well enough."""
    return any(
        candidate.file_path == reference.file_path
        and iou(
            reference.first_character_index,
            reference.last_character_index,
            candidate.first_character_index,
            candidate.last_character_index,
        )
        >= IOU_THRESHOLD
        for candidate in retrieved
    )


def question_recall(
    reference_sources: Sequence[MinimalSource],
    retrieved: Sequence[MinimalSource],
    k: int,
) -> float:
    """Share of `reference_sources` found within the first *k* retrieved."""
    if not reference_sources:
        return 0.0
    top_k = retrieved[:k]
    found = sum(1 for ref in reference_sources if is_found(ref, top_k))
    return found / len(reference_sources)
```

Notes:

- `iou` takes four plain ints rather than two models, so it is trivial to test
  and cannot be confused about which span is which.
- `retrieved[:k]` slices **before** matching. Slicing after would let a match at
  rank 30 count towards recall@5.
- `question_recall` divides by the number of **reference** sources, never by
  `k`. Dividing by `k` would cap recall@10 at 0.1 for a one-source question.

### Step 2 — the scoring pass, same file

```python
def recall_at_k(
    results: Dict[str, List[MinimalSource]],
    truth: RagDataset,
    k_values: Sequence[int] = REPORTED_K,
) -> Dict[int, float]:
    """Mean per-question recall at each k in `k_values`.

    Questions in `truth` that are missing from `results` score 0.0, because a
    question you did not answer is a question you did not retrieve.
    """
    answered = [
        q for q in truth.rag_questions if isinstance(q, AnsweredQuestion)
    ]
    if not answered:
        raise ValueError(
            "ground truth has no answered questions - is this the "
            "AnsweredQuestions dataset?"
        )

    scores: Dict[int, float] = {}
    for k in k_values:
        per_question = [
            question_recall(q.sources, results.get(q.question_id, []), k)
            for q in answered
        ]
        scores[k] = sum(per_question) / len(per_question)
    return scores
```

Three decisions worth stating, because they are judgement calls and not in the
subject:

1. **Questions are matched by `question_id`, never by position.** The two files
   happen to be in the same order today, but an id lookup cannot silently
   mis-score if that ever changes.
2. **A question in the key with no entry in your results scores 0.0.** Ignoring
   it instead would let a crashed half-run report a great average.
3. **`isinstance(q, AnsweredQuestion)` filters the key.** Pydantic's union
   parses an entry without `sources`/`answer` as an `UnansweredQuestion`, which
   has no ground truth to compare against. If *nothing* is answered, the user
   pointed `--dataset_path` at the wrong folder, so raise a `ValueError` saying
   exactly that.

### Step 3 — loading the results file

Add to `src/dataset_io.py`:

```python
def load_search_results(results_path: Path) -> StudentSearchResults:
    """Parse and validate a StudentSearchResults JSON file.

    Raises:
        FileNotFoundError: If `results_path` is not a file.
        pydantic.ValidationError: If the JSON does not match the model.
    """
    if not results_path.is_file():
        raise FileNotFoundError(f"search results not found: {results_path}")
    with results_path.open(encoding="utf-8") as handle:
        return StudentSearchResults.model_validate_json(handle.read())
```

It mirrors `load_dataset` exactly, including the exception types that `main()`
already turns into a one-line error. Import `StudentSearchResults` at the top of
the module alongside `RagDataset`.

### Step 4 — the `evaluate` command in `src/__main__.py`

```python
    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Report recall@k of a results file against a ground-truth dataset."""
        k = int(k)
        results_file = Path(str(student_search_results_path))
        truth_file = Path(str(dataset_path))

        student = load_search_results(results_file)
        truth = load_dataset(truth_file)

        by_id = {
            entry.question_id: entry.retrieved_sources
            for entry in student.search_results
        }
        k_values = [value for value in REPORTED_K if value <= k] or [k]
        scores = recall_at_k(by_id, truth, k_values)

        print(f"{results_file.name}: {len(by_id)} questions scored")
        for value, score in scores.items():
            print(f"  Recall@{value}: {score:.3f}")
```

Add the imports:

```python
from src.dataset_io import load_dataset, load_search_results, save_search_results
from src.evaluation import REPORTED_K, recall_at_k
```

The parameter names `--student_search_results_path` and `--dataset_path` are
fixed by the subject **(§VI.6)** — do not shorten them. `k` caps which values
are reported, so `--k 5` prints 1, 3 and 5. A `--k` below 1 falls back to `[k]`
itself, and `question_recall` with `retrieved[:0]` correctly scores 0.0 rather
than crashing.

### Step 5 — `tests/test_evaluation.py` (new file)

```python
"""Unit tests for the IoU rule and recall@k arithmetic."""

import pytest

from src.evaluation import iou, is_found, question_recall, recall_at_k
from src.models import AnsweredQuestion, MinimalSource, RagDataset


def source(path: str, first: int, last: int) -> MinimalSource:
    """Shorthand for building a source in these tests."""
    return MinimalSource(
        file_path=path, first_character_index=first, last_character_index=last
    )


def test_iou_of_identical_spans_is_one() -> None:
    assert iou(0, 100, 0, 100) == 1.0


def test_iou_of_disjoint_spans_is_zero() -> None:
    assert iou(0, 100, 100, 200) == 0.0
    assert iou(0, 100, 500, 600) == 0.0


def test_iou_matches_the_worked_example() -> None:
    assert iou(4695, 6098, 5100, 7100) == pytest.approx(998 / 2405)


def test_iou_of_zero_width_spans_is_zero_not_a_crash() -> None:
    assert iou(10, 10, 10, 10) == 0.0


def test_is_found_requires_the_same_file() -> None:
    reference = source("a.md", 1000, 2000)
    assert not is_found(reference, [source("b.md", 1000, 2000)])
    assert is_found(reference, [source("a.md", 1000, 2000)])


def test_is_found_rejects_overlap_below_the_bar() -> None:
    # 20 characters shared out of a 2980-character union: IoU ~ 0.0067.
    assert not is_found(source("a.md", 0, 1000), [source("a.md", 980, 3000)])


def test_question_recall_is_a_fraction_of_reference_sources() -> None:
    references = [source("a.md", 0, 100), source("b.md", 0, 100)]
    retrieved = [source("a.md", 0, 100)]
    assert question_recall(references, retrieved, k=5) == 0.5


def test_question_recall_ignores_hits_beyond_k() -> None:
    references = [source("a.md", 0, 100)]
    retrieved = [source("z.md", 0, 100)] * 4 + [source("a.md", 0, 100)]
    assert question_recall(references, retrieved, k=5) == 1.0
    assert question_recall(references, retrieved, k=3) == 0.0


def test_recall_at_k_averages_over_questions_and_matches_by_id() -> None:
    truth = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1", question="?", answer="a",
                sources=[source("a.md", 0, 100)],
            ),
            AnsweredQuestion(
                question_id="q2", question="?", answer="a",
                sources=[source("b.md", 0, 100)],
            ),
        ]
    )
    # Deliberately in the wrong order, and q2 is missing entirely.
    results = {"q1": [source("a.md", 50, 150)]}
    assert recall_at_k(results, truth, [5]) == {5: 0.5}


def test_recall_at_k_rejects_an_unanswered_dataset() -> None:
    truth = RagDataset(rag_questions=[])
    with pytest.raises(ValueError):
        recall_at_k({}, truth, [5])
```

`test_question_recall_ignores_hits_beyond_k` is the one that catches the most
damaging bug in this phase: slicing to top-k after matching instead of before,
which silently inflates every number you then tune against.

---

## Resulting project tree (changes only)

```
src/
├── __main__.py          (+ evaluate)
├── dataset_io.py        (+ load_search_results)
└── evaluation.py        NEW
tests/
└── test_evaluation.py   NEW
```

---

## Commands to run

```powershell
uv run pytest -q
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 10
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
```

---

## Expected output

**These are the real numbers.** I computed them on 2026-09-16 with an
independent script (kept outside the project) over your current Phase 7 output
files, using the rule in §VII.1.1. A correct implementation must reproduce
them exactly:

```
dataset_docs_public.json: 100 questions scored
  Recall@1: 0.540
  Recall@3: 0.770
  Recall@5: 0.820
  Recall@10: 0.880

dataset_code_public.json: 99 questions scored
  Recall@1: 0.444
  Recall@3: 0.576
  Recall@5: 0.626
  Recall@10: 0.707
```

If your `evaluate` prints something different, the bug is in your
implementation, not in the retrieval — the same input files produced the numbers
above.

**`pytest -q`** reports **42 passed** (32 now, plus 10 new).

### What these numbers mean

| Dataset | Recall@5 | Threshold (N3) | Margin |
|---|---|---|---|
| Docs | **0.820** | 0.80 | +0.02 — **2 questions** |
| Code | **0.626** | 0.50 | +0.126 |

**Both thresholds are already met by fixed-window TF-IDF.** That is a genuinely
better starting position than the roadmap assumed in risk R2, which predicted
code recall would land near or below the bar.

The docs margin is the thin one: two questions flipping the wrong way on a
private dataset would fail it. For reference, if the span rule were ignored and
only the *file* had to match, docs recall@5 would be 0.910 and code 0.727 — so
roughly 9 points of the docs gap is "right file, wrong part of it", which is
exactly what better chunk boundaries (Phase 9) address.

---

## Manual verification

1. **Score a file against the wrong dataset.** Point `--student_search_results_path`
   at the docs results and `--dataset_path` at the code answers. The ids do not
   intersect, so every question scores 0.0 and you should see `Recall@5: 0.000`.
   That proves questions are matched by id, not by position.
2. **Score the AnsweredQuestions run.** It must give the same numbers as the
   UnansweredQuestions run — Phase 7 already proved the retrieved sources are
   identical.
3. **Re-run `search_dataset` with `--k 3`** into a scratch directory and
   evaluate that file with `--k 3`. Recall@3 must match the value above (0.770
   for docs), because the top 3 of a k=10 run and a k=3 run are the same.
4. **Check `--k 5`** prints exactly three lines (1, 3, 5).
5. **Degenerate inputs:** a missing results file, a missing dataset, and a
   malformed JSON file must each print one `error:` line and exit 1.

## Automated tests

Ten tests, split between the two concepts: the IoU formula (identical, disjoint,
the worked example, zero-width) and the recall arithmetic (same-file rule,
below-bar rejection, fraction of references, the top-k slice, id matching with a
missing question, wrong-dataset rejection). They use hand-built models, so they
run instantly and do not need an index.

Do not add a test that asserts recall equals 0.820 on the real corpus. It would
fail the moment Phase 9 improves chunking — which is the whole point of Phase 9.
Record the number in the README instead.

---

## Most likely errors

**1. Every recall value is identical.**
*Cause:* `retrieved[:k]` used a fixed `k` (or the slice was applied after
matching), so every k scores the same.
*Fix:* slice inside `question_recall`, with the `k` passed in.

**2. Recall@10 is suspiciously near 0.1, recall@5 near 0.2.**
*Cause:* dividing by `k` instead of by `len(reference_sources)`.
*Fix:* recall is a share of the **correct** sources, not of your results.

**3. `AttributeError: 'UnansweredQuestion' object has no attribute 'sources'`.**
*Cause:* `--dataset_path` points at `UnansweredQuestions/`, which has no answer
key, so the union parsed every entry as `UnansweredQuestion`.
*Fix:* point it at `AnsweredQuestions/`. The `isinstance` filter plus the
`ValueError` in Step 2 turns this into a clear message.

**4. Recall is 0.000 everywhere on a file you know is good.**
*Cause:* usually the two paths are swapped (results file passed as dataset), or
ids are being compared against positions.
*Fix:* check the printed question count first — if it is 0, nothing matched.

**5. Numbers that are close to mine but not equal.**
*Cause:* an inclusive `last` (off-by-one) in the intersection, or `>` instead of
`>=` at the 0.05 bar.
*Fix:* spans are half-open; the bar is inclusive.

---

## Definition of done

- [ ] `uv run python -m src evaluate …` reproduces the docs and code numbers
      above exactly.
- [ ] Recall is computed per question as a share of its reference sources, then
      averaged.
- [ ] Questions are matched by `question_id`; a missing question scores 0.0.
- [ ] Pointing `--dataset_path` at `UnansweredQuestions/` gives a clear error,
      not a traceback.
- [ ] `--k 5` reports exactly 1, 3 and 5.
- [ ] `uv run pytest -q` reports 42 passed; flake8 silent; mypy clean.
- [ ] `src/` still never imports or calls the moulinette (F17).

**Natural commit point:** "evaluate: recall@k against the ground-truth dataset".

**Also worth doing now:** record the four Phase 7 output files' numbers in
`README.md`'s "Performance analysis" section, which currently says recall is
"not yet measured". The README also still contains the `<your-42-login>`
placeholder (OI-3).

---

## What Phase 9 adds and why — and a decision you now get to make

Phase 9 replaces the fixed window with two structure-aware chunkers: Python cut
on `ast` `def`/`class` boundaries, Markdown cut on heading boundaries. Note that
this is **not optional polish**: F4 requires two distinct chunking strategies,
and it is the last unmet mandatory requirement in the codebase. It has to happen
whatever the numbers say.

What the numbers change is the *framing* of Phases 10 and 11:

- **Phase 10** (identifier-aware tokenisation) was aimed at dragging code recall
  over 50 %. It is already at 62.6 %, so Phase 10 becomes margin-building, not
  rescue work.
- **Phase 11** (BM25) was the reserve plan for missing the thresholds. Both are
  met, so under decision D4 it stays unbuilt and becomes README future-work —
  unless Phase 9 *drops* a number below its bar.

Run `evaluate` before and after Phase 9 and keep both numbers. The docs figure
sits only two questions above its threshold, so a change that helps code recall
and quietly costs three docs questions would be a net failure — and without this
phase you would not have seen it.

**Still open (OI-2):** these are *our* numbers under *our* reading of §VII.1.1.
Before the defense, run the real moulinette once on Linux against the same files
and confirm it agrees.
