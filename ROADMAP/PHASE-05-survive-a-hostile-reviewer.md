# Phase 5 — Survive a hostile reviewer

**Goal:** Every command handles every degenerate input without a traceback, `make lint` is clean, and the performance budgets are measured rather than assumed.
**Time:** ~4h · **Difficulty:** ●●○○○
**Depends on:** Phase 4 complete. All six commands must exist before you harden them.

## ✅ What you'll have when this is done

A project that does not fall over when someone pokes it. Empty queries, `k=0`, missing files, malformed JSON, a deleted index, a wrong retriever name, Ctrl-C mid-run — each produces one clear line and a sensible exit code. `flake8` and `mypy` both pass. And you have numbers for the three performance budgets instead of a hope.

```bash
$ uv run python -m src search "" --k 5
No results.

$ uv run python -m src search "lora" --k 0
No results.

$ uv run python -m src evaluate --student_search_results_path nope.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
error: search results not found: nope.json

$ uv run python -m src search "lora" --retriever elasticsearch
error: unknown retriever 'elasticsearch', expected one of ('bm25', 'tfidf')

$ make lint
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
Success: no issues found in 17 source files
```

## Where you're starting from

```
src/
├── __main__.py     # all six subject commands
├── analyzer.py · bm25.py · chunking.py · corpus.py · datasets.py
├── evaluation.py · generator.py · indexer.py · models.py · retriever.py
tests/
├── test_chunking.py · test_datasets.py · test_evaluation.py
├── test_generator.py · test_models.py
benchmarks.md
```

The pipeline works end to end on well-formed input. `make lint` almost certainly does not pass, and nobody has yet run a command with deliberately bad arguments.

## Why this phase now

The subject says it in as many words: *"The CLI is tested with such edge cases and must never crash with an unhandled traceback"*, and *"If your program crashes due to unhandled exceptions during the review, it will be considered non-functional."* This is not polish; it is a pass/fail criterion, and it is cheap to satisfy once — and expensive to satisfy while a reviewer watches.

Doing it now rather than in Phase 6 also means the README you write next describes a project that behaves the way you say it does.

## Before you start

Nothing to install. Have a terminal with the index already built — several checks in this phase search against it.

Know your two exit-code conventions before you write them, because being inconsistent is worse than either choice: **1** for a valid command that could not complete (missing file, bad value), **130** for Ctrl-C. Fire handles unknown commands itself and exits **2**.

## Key design decisions

- **Where errors are caught.** Inside each command, or once around `fire.Fire`. Recommendation: **once, in `main()`.** Every command already raises a specific, well-worded exception from the layer that knows what went wrong (`corpus.list_corpus_files`, `datasets.load_search_results`, `Retriever.load`); wrapping the CLI once turns all of them into one clean line without scattering `try/except` through six methods. Per-command handling would also make it far too easy to swallow a genuine bug.

- **Which exceptions to catch.** A bare `except Exception`, or an explicit tuple. Recommendation: **an explicit tuple.** `except Exception` hides your own bugs behind a friendly message, which is exactly how a defect ships. Anything not in the tuple is a real bug and should print a traceback — during development, where you will see it.

- **What a degenerate query returns.** An error, or an empty result. Recommendation: **an empty result and the line `No results.`** An empty query and `k=0` are well-formed requests with an empty answer, not failures. Reserve exit code 1 for cases where something is genuinely wrong.

- **flake8 line length.** 79 (the default) or 88. Recommendation: **88**, set in `setup.cfg` since Phase 1. `flake8 .` reads your configuration, so the run is clean either way; 88 costs you no arguments about wrapped `MinimalSource(...)` constructor calls.

- **Whether tests run the real CLI.** In-process calls to `Cli` methods, or a subprocess. Recommendation: **subprocess.** The claim you are making is "`python -m src ...` never prints a traceback", and only a subprocess actually tests that claim — an in-process call cannot catch a failure in `main()`'s own error handling.

## Files in this phase

| File | New/Edit | What it holds |
|---|---|---|
| `src/__main__.py` | edit | `EXPECTED_ERRORS`, a hardened `main()`, an empty-input guard in `answer_dataset` |
| `tests/test_cli.py` | new | Nine degenerate invocations, each asserting no traceback |
| every `src/*.py` | edit | Whatever flake8 and mypy flag |
| `.gitignore` | edit | `sweep.log` and any other scratch output |

## Steps

### 1. Turn expected failures into one clean line

**Why:** Six commands, one place to get this right.

In `src/__main__.py`, add the imports and replace `main()`.

```python
# src/__main__.py
# ... existing imports ...
import sys

from pydantic import ValidationError

#: Failures that mean "the input was wrong", not "the code is wrong".
#: `json.JSONDecodeError` and pydantic's `ValidationError` are both
#: ValueError subclasses, and are listed for the reader, not the runtime.
EXPECTED_ERRORS = (
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
    ValidationError,
    ValueError,
)


# ... class Cli unchanged ...


def main() -> None:
    """Hand the CLI to Fire, turning expected failures into messages."""
    try:
        fire.Fire(Cli, name="python -m src")
    except EXPECTED_ERRORS as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
```

Everything outside that tuple still produces a traceback, deliberately. A `TypeError` or an `AttributeError` is your bug, and hiding it behind `error: ...` means finding it later, in front of someone.

`KeyboardInterrupt` is separate because it does not derive from `Exception`, and because 130 is the shell convention for "killed by SIGINT" — a reviewer who Ctrl-Cs a 40-minute `answer_dataset` should get a clean line, not fifty lines of `transformers` stack.

**Check:**

```bash
uv run python -m src search "lora" --processed_dir /definitely/not/here
echo "exit: $?"
```
prints `error: no bm25 index in /definitely/not/here - run: python -m src index` and `exit: 1`, with no traceback.

### 2. Close the remaining degenerate paths

**Why:** The wrapper turns exceptions into messages. It does not stop `answer_dataset` from loading 1.5 GB of weights to answer zero questions.

Two small guards. First, in `src/__main__.py`, inside `answer_dataset`, right after the `rows` slice and before the `Generator` is constructed:

```python
# src/__main__.py
# ... inside answer_dataset, after: print(f"Loaded {len(results.search_results)} questions")

        if not rows:
            print("Nothing to answer.")
            return

        generator = Generator(str(model))
        # ... unchanged ...
```

Second, in `src/indexer.py`, validate the chunk size before walking 2121 files rather than after:

```python
# src/indexer.py
# ... inside class Indexer ...

    def __init__(self, max_chunk_size: int = 1200) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be > 0")
        self.max_chunk_size = max_chunk_size
        self.chunks: List[Chunk] = []
```

The default moves from 2000 to whatever Phase 3's sweep chose — 1200 here as the illustration. Change it in `Cli.index`'s signature too, so the two agree.

Now walk the whole matrix by hand. Every one of these must print a line and exit 0 or 1:

```bash
uv run python -m src search "" --k 5
uv run python -m src search "lora" --k 0
uv run python -m src search "lora" --k -3
uv run python -m src search "zzzzqqqq" --k 5
uv run python -m src search "lora" --retriever elasticsearch
uv run python -m src index --max_chunk_size 0
uv run python -m src index --raw_dir /nope
uv run python -m src search_dataset --dataset_path /nope.json
uv run python -m src evaluate --student_search_results_path nope.json --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
uv run python -m src evaluate --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
echo '{"rag_questions": [' > /tmp/broken.json
uv run python -m src search_dataset --dataset_path /tmp/broken.json
echo '{"rag_questions": []}' > /tmp/empty.json
uv run python -m src search_dataset --dataset_path /tmp/empty.json --save_directory /tmp/out
```

The last two are the interesting ones. Truncated JSON must produce `error: /tmp/broken.json is not valid JSON: ...`; an empty but well-formed dataset must write a valid results file with an empty `search_results` list and exit 0 — it is a legitimate request with a legitimately empty answer.

The `evaluate` line pointed at `UnansweredQuestions/` must produce the explicit message `error: ... has no answered questions - point --dataset_path at AnsweredQuestions/`, which is `truth_by_id` doing its job.

**Check:** every line above prints one message. `grep -c Traceback` over the session output is 0.

### 3. Pin it with a subprocess test

**Why:** "It does not crash" is a claim about the process, and only a subprocess can check it. This file is also what stops a later refactor of `main()` from quietly re-introducing tracebacks.

```python
# tests/test_cli.py
"""Every degenerate invocation must produce a message, not a traceback."""

import subprocess
import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS = REPO_ROOT / "data" / "datasets"


def run(args: List[str]) -> "subprocess.CompletedProcess[str]":
    """Invoke the CLI exactly the way a reviewer does."""
    return subprocess.run(
        [sys.executable, "-m", "src", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


DEGENERATE = [
    ["search", "", "--k", "5"],
    ["search", "lora", "--k", "0"],
    ["search", "lora", "--k", "-3"],
    ["search", "zzzzqqqqxxxx", "--k", "5"],
    ["search", "lora", "--retriever", "elasticsearch"],
    ["search", "lora", "--processed_dir", "/definitely/not/here"],
    ["index", "--max_chunk_size", "0"],
    ["index", "--raw_dir", "/definitely/not/here"],
    ["search_dataset", "--dataset_path", "/definitely/not/here.json"],
]


@pytest.mark.parametrize("args", DEGENERATE, ids=lambda a: " ".join(a))
def test_no_traceback(args: List[str]) -> None:
    result = run(args)
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode in (0, 1), result.stderr


def test_malformed_json_is_reported(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text('{"rag_questions": [', encoding="utf-8")
    result = run(["search_dataset", "--dataset_path", str(broken)])
    assert "Traceback" not in result.stderr
    assert "not valid JSON" in result.stderr
    assert result.returncode == 1


def test_empty_dataset_writes_an_empty_result(tmp_path: Path) -> None:
    empty = tmp_path / "dataset_empty.json"
    empty.write_text('{"rag_questions": []}', encoding="utf-8")
    result = run(
        [
            "search_dataset",
            "--dataset_path",
            str(empty),
            "--save_directory",
            str(tmp_path / "out"),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out" / "dataset_empty.json").is_file()


@pytest.mark.skipif(
    not (DATASETS / "UnansweredQuestions").is_dir(),
    reason="datasets not present",
)
def test_evaluate_against_the_wrong_dataset_explains_itself() -> None:
    result = run(
        [
            "evaluate",
            "--student_search_results_path",
            "nope.json",
            "--dataset_path",
            str(DATASETS / "AnsweredQuestions" / "dataset_docs_public.json"),
        ]
    )
    assert "Traceback" not in result.stderr
    assert "not found" in result.stderr
```

Several of these need a built index, which is why this test file arrives in Phase 5 and not earlier. The `skipif` on the last one keeps `pytest` honest on a fresh clone where `data/` has not been populated yet.

**Check:** `uv run pytest -q` prints `47 passed` (or `46 passed, 1 skipped` without the datasets). The subprocess tests take a few seconds each — that is the cost of testing the real thing.

### 4. Make `make lint` pass

**Why:** It is a stated requirement, it is checked mechanically, and it costs an hour once versus an argument at the defense.

```bash
make lint
```

Work through what it reports. The recurring ones in this codebase, and what they mean:

| flake8 code | What it is | Fix |
|---|---|---|
| `E501 line too long` | A long f-string or a wide constructor call | Wrap the call, or split the f-string across adjacent string literals |
| `F401 imported but unused` | Left over from a refactor | Delete it; do not add `# noqa` |
| `W605 invalid escape sequence` | A regex written without `r"..."` | Make it a raw string — this one is a real bug waiting to happen |
| `E731 lambda assignment` | A lambda given a name | Make it a `def`; `analyzer.identity` already is one for this reason |

Then mypy. The ones you will actually hit:

| mypy complaint | Why | Fix |
|---|---|---|
| `Function is missing a type annotation` | `--disallow-untyped-defs` covers `tests/` too | Annotate every test as `-> None`, and its fixtures (`tmp_path: Path`) |
| `Returning Any from function declared to return "X"` | `joblib.load` and scikit-learn return `Any` | Annotate the intermediate (`payload: Dict[str, Any]`) and construct the typed value explicitly |
| `Module has no attribute ...` for scipy/sklearn | No stubs shipped | Already handled by `ignore_missing_imports = True` in `setup.cfg` |
| `Incompatible types in assignment` on a numpy array | `np.asarray(...)` widens | Declare `-> np.ndarray` and let it be |

Resist `# type: ignore`. Each one is a place the type checker was telling you something. If you must use one, `--warn-unused-ignores` will tell you when it stops being needed.

Then try the stricter rule, which the subject recommends:

```bash
make lint-strict
```

`--strict` will demand annotations on the `Any`-typed attributes in `Retriever` and `Generator`. Getting there is worthwhile but optional; if you stop at `make lint`, say so in the README rather than leaving it ambiguous.

**Check:** `make lint` exits 0 and prints `Success: no issues found in N source files`.

### 5. Measure the three performance budgets

**Why:** The subject states them as hard numbers. "It feels fast" is not an answer at a defense, and one of the three is easy to breach without noticing.

```bash
# 1. Indexing: at most 5 minutes
time uv run python -m src index --max_chunk_size 1200

# 2. Retrieval: at most 90 seconds for 200 questions
time ( \
  uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions ; \
  uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions )

# 3. Recall@5: docs >= 0.80, code >= 0.50, per the moulinette
for scope in docs code; do
  moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json \
    data/datasets/AnsweredQuestions/dataset_${scope}_public.json \
    --k 10 --max_context_length 2000
done
```

Note that budget 2 as written includes two process startups and two index loads, which is harsher than the intended measurement — the `Searched N questions in Xs` line your `search_dataset` prints is the honest retrieval-only figure. Record both; the wall-clock number is the one a reviewer will time.

If retrieval is over budget, the causes in order of likelihood are: constructing the `Retriever` inside the loop (Phase 2 step 3 exists to prevent this), a chunk size so small the index has 60 000 rows, or `argsort` over the full score vector instead of `argpartition`.

Write all three numbers into `benchmarks.md` under a `## Performance` heading. Phase 6 lifts them straight into the README.

**Check:** all three budgets met, with the measured numbers written down.

### 6. Tidy the repository

**Why:** A reviewer's first impression is `git status` and `ls`. Stray scratch files read as carelessness, and a committed 1.5 GB of weights reads as worse.

```bash
printf 'sweep.log\n.hypothesis/\n' >> .gitignore
git status --short
git count-objects -vH | grep size-pack
```

`git status --short` must be empty apart from files you mean to commit. `size-pack` must be a few hundred kilobytes — if it is in the hundreds of megabytes, something large got committed at some point and `git log --stat --all | grep -i -E 'joblib|safetensors|\.bin'` will tell you when.

Also confirm the things the subject requires are actually present at the root:

```bash
ls pyproject.toml uv.lock Makefile README.md
ls src/__init__.py src/__main__.py
```

`README.md` is still empty — that is Phase 6. Everything else must exist.

**Check:** `git status --short` is empty and `size-pack` is small.

## Common pitfalls

| Pitfall | Why it happens | Avoid it by |
|---|---|---|
| `except Exception` around `fire.Fire` | It makes every test pass immediately | An explicit tuple; a `TypeError` must still be loud |
| A traceback from Ctrl-C during `answer_dataset` | `KeyboardInterrupt` does not derive from `Exception` | Catch it separately, exit 130 |
| `k=0` treated as an error | It looks like nonsense input | It is a well-formed request for zero results — print `No results.`, exit 0 |
| An empty dataset crashing `search_dataset` | Nobody tests the empty case | The empty-dataset test in step 3 |
| `# noqa` sprinkled to make flake8 pass | It is faster than wrapping the line | Wrap the line; a `noqa` in a graded project invites the question you least want |
| `mypy` clean because it never looked at `src/` | An over-broad `exclude` regex in `setup.cfg` | Check the "no issues found in N source files" count — N must be ~17, not 2 |
| Tests that pass only because the index exists | They were written after a full run | The `skipif` guard, and one `pytest` run from a fresh clone before you call the phase done |
| Timing a run with a warm page cache and calling it the budget | The second run is always faster | Report the first-run number; it is what the reviewer sees |
| Fixing lint by deleting code that mattered | `F401` on an import you actually use indirectly | Read each finding; `F401` on `src/models` imports in `__main__.py` usually means the wiring is wrong, not the import |

## Verify it's done

```bash
uv run python -m src search "" --k 5
uv run python -m src search "lora" --k 0
uv run python -m src search "lora" --retriever elasticsearch; echo "exit: $?"
uv run python -m src index --max_chunk_size 0; echo "exit: $?"
uv run python -m src evaluate --student_search_results_path nope.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json; echo "exit: $?"
make lint
uv run pytest -q
```

Expected:

```
No results.
No results.
error: unknown retriever 'elasticsearch', expected one of ('bm25', 'tfidf')
exit: 1
error: max_chunk_size must be > 0
exit: 1
error: search results not found: nope.json
exit: 1
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
Success: no issues found in 17 source files
47 passed
```

Not one `Traceback` anywhere in that output.

## Definition of done

- [ ] Every command handles empty query, `k=0`, negative `k`, missing file, missing directory, malformed JSON, empty dataset and a bad `--retriever`
- [ ] No degenerate input produces a traceback; exit codes are 0, 1 or 130 consistently
- [ ] Ctrl-C during `answer_dataset` prints `interrupted` and exits 130
- [ ] `make lint` exits 0 over ~17 source files
- [ ] `uv run pytest -q` passes, including the subprocess CLI tests
- [ ] Indexing under 5 minutes, 200 questions under 90 seconds, both bars cleared — all three measured and written into `benchmarks.md`
- [ ] `git status --short` is empty; no weights, no index, no scratch files committed
- [ ] Committed

## Deliberately NOT in this phase

- The README → **Phase 6**
- The one-command pipeline script → **Phase 6**
- `make lint-strict` passing → optional; do it if it is quick, document it either way
- Retrieval or answer-quality improvements → both phases are closed; changing them now invalidates the numbers you just measured
- Structured logging, a `--verbose` flag, colored output → not in v1

## Commit

```bash
git add -A
git commit -m "phase 5: graceful degenerate input handling, clean lint, measured budgets"
```

## Next

→ **[Phase 6 — Defense-ready from a fresh clone](PHASE-06-defense-ready.md)**
