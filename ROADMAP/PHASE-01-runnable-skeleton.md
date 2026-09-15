# Phase 1 — Runnable skeleton

**Goal:** stand up a lint-clean `uv` project whose Fire CLI answers one real
command, `uv run python -m src status`.

---

## What the system can do after this phase that it cannot do now

Right now nothing runs — there is no project. After this phase you have a
dependency-managed Python project that starts, exposes a command-line
interface, reports whether the subject's mandatory directory layout is present,
and passes `flake8`, `mypy` and `pytest` cleanly. Every later phase adds a
command to a CLI that already works, rather than to nothing.

`status` is not a toy. The subject (§VI.7.1) makes the directory layout a
graded requirement, and the defense scripts fail by design if it is wrong. You
will run `status` at the start of Phases 2, 3 and 15 to confirm the layout
before doing anything else.

## Prerequisites

- Python 3.12.4 — already installed and verified.
- uv 0.11.31 — already installed and verified.
- Nothing else. No corpus, no datasets, no network beyond what `uv sync` needs
  to fetch four small pure-Python packages.

---

## Concepts introduced in this phase

Two, and they are introduced together because a Python project that manages its
dependencies but does nothing, and a CLI with no environment to run in, are both
useless halves.

### Concept 1 — `uv` as the project manager

**The problem.** Your project needs `fire`, `pydantic`, `scikit-learn`,
`torch` and more, at versions that work together, installed in an environment
that is *this project's* and not your global Python. The evaluator will run
exactly one command to reproduce that environment: `uv sync`. If it does not
work, nothing else is graded (§V.4).

**What solves it.** `uv` reads `pyproject.toml` (what you *want*), resolves it
into `uv.lock` (what you *get*, pinned exactly), and materialises it into a
local `.venv`. `uv run <cmd>` then executes `<cmd>` inside that environment
without you ever activating anything.

**Tiny concrete example.** You write this in `pyproject.toml`:

```toml
dependencies = ["fire>=0.6"]
```

You run `uv sync`. uv creates `.venv`, installs a specific version such as
`fire==0.7.1`, and records that exact version plus its hash in `uv.lock`. On
any other machine, `uv sync` reads the lock file and installs `fire==0.7.1` —
the same bytes, not merely "something ≥ 0.6". That reproducibility is the whole
point of a lock file, and it is why the subject requires `uv.lock` to be
committed.

**Why now.** `uv sync` must work from the repository root before any code
exists, because every single command in every later phase is prefixed with
`uv run`.

**What it costs.** A `.venv` directory (a few hundred MB once torch arrives in
Phase 12) and the discipline of never running bare `python` in this project.

### Concept 2 — Python Fire turning a class into a CLI

**The problem.** The subject fixes six commands with named options
(`search_dataset --dataset_path <p> --k <int> --save_directory <d>`). Writing
that with `argparse` means a parser, a subparser per command, and an `add_argument`
call per option — roughly forty lines of boilerplate that you must keep in sync
with the functions they call.

**What solves it.** Fire inspects a Python object and *derives* the CLI from it.
Give it a class: each public method becomes a command, each parameter becomes an
option, and each default value becomes the option's default.

**Tiny concrete example.**

```python
class Cli:
    def greet(self, name: str, times: int = 1) -> None:
        for _ in range(times):
            print(f"hello {name}")

fire.Fire(Cli)
```

Run `python -m src greet --name world --times 2` and Fire parses `--name` and
`--times`, converts `2` to the right slot, and calls the method. You wrote no
parser.

**Why now.** F1 and F2 are structural: the shape of the CLI determines how every
later module is called. Fixing it in Phase 1 means Phases 2–15 each add one
method to an existing class.

**What it costs.** Fire is loose about types — it will happily hand you the
string `"5"` where you expected `int 5`, because it infers from the literal on
the command line. That is why every command in this project casts its own
arguments (`int(k)`, `str(query)`). You will see that pattern from Phase 6
onward, and Phase 14 makes it systematic.

---

## Concepts deliberately deferred

| Concept | Deferred to | Why not now |
|---|---|---|
| pydantic models and the JSON contract | Phase 2 | Nothing produces or consumes structured data yet. |
| Reading the corpus, path normalisation | Phase 3 | No corpus is copied in until it is needed. |
| Chunking and character offsets | Phase 4 | — |
| TF-IDF, `tqdm`, index persistence | Phase 5 | These dependencies are not added to `pyproject.toml` until the phase that uses them. |
| Retrieval and ranking | Phase 6 | — |
| recall@k and the IoU rule | Phase 8 | — |
| `transformers`, `torch`, Qwen3-0.6B | Phase 12 | ~2–3 GB of install for code that does not exist yet. |
| The systematic error boundary for degenerate input | Phase 14 | `status` takes no arguments, so there is nothing to abuse yet. |
| `README.md` content | Phase 15 | — |

---

## Setup

Open **PowerShell** and run these three commands.

```powershell
New-Item -ItemType Directory -Force -Path "C:\Users\user\Desktop\my_projects\python\rag_final"
Set-Location "C:\Users\user\Desktop\my_projects\python\rag_final"
New-Item -ItemType Directory -Force -Path src, tests
```

> If you prefer a different project folder, change it here and use your path
> everywhere `rag_final` appears from now on. Tell me and I will update the
> roadmap's assumption A1.

Everything from here on assumes your shell's working directory is
`C:\Users\user\Desktop\my_projects\python\rag_final`.

---

## Implementation sequence

### Step 1 — `pyproject.toml`

Create **`pyproject.toml`** at the project root with exactly this content:

```toml
[project]
name = "rag-against-the-machine"
version = "0.1.0"
description = "Retrieval-Augmented Generation over the vLLM codebase"
requires-python = ">=3.10"
dependencies = [
    "fire>=0.6",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "flake8>=7.0",
    "mypy>=1.10",
]

[tool.uv]
package = false

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`package = false` tells uv this is an application, not a library to build and
install — so `src/` is imported straight from the working directory and you do
not need a build backend. `[dependency-groups] dev` holds tools the graded code
never imports; `uv sync` installs them by default and they stay out of the
runtime dependency list.

There is no `readme = "README.md"` line yet, because `README.md` does not exist
until Phase 15. It gets added there.

### Step 2 — `setup.cfg`

Create **`setup.cfg`** at the project root:

```ini
[flake8]
max-line-length = 88
exclude = .git,__pycache__,.venv,data,build,dist
extend-ignore = E203

[mypy]
python_version = 3.10
exclude = (?x)(^data[/\\]|^\.venv[/\\])
ignore_missing_imports = True
warn_return_any = True
warn_unused_ignores = True
disallow_untyped_defs = True
check_untyped_defs = True
```

This is the file that makes N4 and N5 checkable. The `[mypy]` section mirrors
the exact flags the subject mandates for `make lint`, so running bare
`mypy .` and running the Makefile target give the same verdict. `data` is
excluded from both tools because the vLLM corpus is 2 880 third-party files —
linting them would be meaningless and slow.

### Step 3 — `Makefile`

Create **`Makefile`** at the project root.

> **Every indented line in a Makefile must begin with a real TAB character, not
> spaces.** This is the single most common way this file fails. If your editor
> converts tabs to spaces, turn that off for this file.

```make
UV := uv
ARGS ?=
MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
              --ignore-missing-imports --disallow-untyped-defs \
              --check-untyped-defs

.DEFAULT_GOAL := help
.PHONY: install run debug clean lint lint-strict test help

install:
	$(UV) sync

run:
	$(UV) run python -m src $(ARGS)

debug:
	$(UV) run python -m pdb -m src $(ARGS)

lint:
	$(UV) run flake8 .
	$(UV) run mypy . $(MYPY_FLAGS)

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict

test:
	$(UV) run pytest -q

clean:
	$(UV) run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]"
	$(UV) run python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('.mypy_cache', '.pytest_cache')]"

help:
	@echo "targets: install run debug lint lint-strict test clean"
```

`make` is not installed on this machine (risk R4), so you cannot run these
targets locally — but the file is a hard submission requirement (F15) and a
reviewer on Linux will run them. Every phase gives you the raw `uv run ...`
command as the primary instruction, and names the equivalent target beside it.

`ARGS ?=` lets a reviewer write `make run ARGS="search 'lora' --k 5"`.

### Step 4 — `.gitignore`

Create **`.gitignore`** at the project root:

```gitignore
__pycache__/
*.py[cod]
.venv/
.mypy_cache/
.pytest_cache/
data/
moulinette
```

`data/` is ignored wholesale: the subject says explicitly not to commit large
data files, model weights or generated outputs, and that the evaluator
generates them (§X). The `moulinette` binary is 10 MB and is not yours to
distribute.

### Step 5 — `src/__init__.py`

Create **`src/__init__.py`** as an **empty file** (zero bytes). Its presence is
what makes `src` an importable package, so `python -m src` and
`from src.models import ...` both work.

In PowerShell:

```powershell
New-Item -ItemType File -Path src\__init__.py
```

### Step 6 — `src/__main__.py`

Create **`src/__main__.py`** with exactly this content:

```python
"""Command-line entry point: uv run python -m src <command>."""

from pathlib import Path
from typing import Dict

import fire

# __file__ is src/__main__.py, so two parents up is the project root.
# Every path in this project is derived from here, never from the
# current working directory, so commands work from any directory.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# The layout the subject makes mandatory in VI.7.1.
EXPECTED_DIRS: Dict[str, str] = {
    "data/raw": "corpus to index",
    "data/processed": "generated index",
    "data/datasets/UnansweredQuestions": "questions without answers",
    "data/datasets/AnsweredQuestions": "ground truth for evaluate",
    "data/output/search_results": "search_dataset output",
    "data/output/search_results_and_answer": "answer_dataset output",
}


class Cli:
    """Retrieval-Augmented Generation over the vLLM codebase."""

    def status(self) -> None:
        """Report the project root and which expected directories exist."""
        print(f"project root: {REPO_ROOT}")
        for relative, purpose in EXPECTED_DIRS.items():
            mark = "OK     " if (REPO_ROOT / relative).is_dir() else "MISSING"
            print(f"  [{mark}] {relative:38s} {purpose}")


def main() -> None:
    """Hand the CLI class to Fire."""
    fire.Fire(Cli, name="python -m src")


if __name__ == "__main__":
    main()
```

Three details worth noticing, because later phases depend on them:

- `REPO_ROOT` is derived from `__file__`, never from the current working
  directory. The evaluator may invoke your CLI from anywhere.
- `Cli` is a class, not a module of loose functions. Fire maps methods to
  commands, so adding `search` in Phase 6 means adding one method here.
- The `if __name__ == "__main__":` guard means importing this module in a test
  does not launch the CLI.

### Step 7 — `tests/__init__.py`

Create **`tests/__init__.py`** as an **empty file**:

```powershell
New-Item -ItemType File -Path tests\__init__.py
```

This makes pytest treat `tests` as a package and put the project root on
`sys.path`, which is what lets `from src.__main__ import ...` resolve.

### Step 8 — `tests/test_cli.py`

Create **`tests/test_cli.py`**:

```python
"""Tests for the CLI module's static wiring."""

from src.__main__ import EXPECTED_DIRS, REPO_ROOT


def test_repo_root_points_at_the_project_root() -> None:
    """REPO_ROOT must resolve to the directory holding pyproject.toml."""
    assert (REPO_ROOT / "pyproject.toml").is_file()


def test_expected_dirs_are_relative_forward_slash_paths() -> None:
    """The layout keys must be relative and use forward slashes only."""
    for relative in EXPECTED_DIRS:
        assert not relative.startswith("/")
        assert "\\" not in relative
```

The second test looks trivial today and is not. Risk R7 in the overview is that
a single Windows backslash in a path scores zero on every graded question. This
test pins the forward-slash convention from the first file that has paths in it,
and Phase 3 extends the same idea to the real corpus paths.

Both tests are annotated `-> None` because `disallow_untyped_defs` applies to
the test files too — `mypy .` checks everything.

---

## Resulting project tree

```
C:\Users\user\Desktop\my_projects\python\rag_final\
├── .gitignore
├── Makefile
├── pyproject.toml
├── setup.cfg
├── src/
│   ├── __init__.py          (empty)
│   └── __main__.py
└── tests/
    ├── __init__.py          (empty)
    └── test_cli.py
```

After `uv sync` this also contains `.venv/` and `uv.lock`, both generated.

---

## Commands to run

From `C:\Users\user\Desktop\my_projects\python\rag_final`, in order:

```powershell
uv sync
uv run python -m src status
New-Item -ItemType Directory -Force -Path data\raw, data\processed, data\datasets\UnansweredQuestions, data\datasets\AnsweredQuestions, data\output\search_results, data\output\search_results_and_answer
uv run python -m src status
uv run flake8 .
uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
uv run pytest -q
```

Running `status` **before and after** creating the directories is deliberate:
it shows you both branches of the check, so you know the command is actually
looking at the filesystem rather than printing a fixed list.

---

## Expected output

**`uv sync`** — resolved versions will differ; the shape is what matters:

```
Using CPython 3.12.4
Creating virtual environment at: .venv
Resolved 12 packages in 431ms
Installed 11 packages in 219ms
 + fire==0.7.1
 + flake8==7.1.1
 + mypy==1.13.0
 ...
```

A `uv.lock` file now exists at the project root. That file is graded — do not
delete it.

**First `uv run python -m src status`** (before the directories exist):

```
project root: C:\Users\user\Desktop\my_projects\python\rag_final
  [MISSING] data/raw                               corpus to index
  [MISSING] data/processed                         generated index
  [MISSING] data/datasets/UnansweredQuestions      questions without answers
  [MISSING] data/datasets/AnsweredQuestions        ground truth for evaluate
  [MISSING] data/output/search_results             search_dataset output
  [MISSING] data/output/search_results_and_answer  answer_dataset output
```

**Second `uv run python -m src status`** (after creating them):

```
project root: C:\Users\user\Desktop\my_projects\python\rag_final
  [OK     ] data/raw                               corpus to index
  [OK     ] data/processed                         generated index
  [OK     ] data/datasets/UnansweredQuestions      questions without answers
  [OK     ] data/datasets/AnsweredQuestions        ground truth for evaluate
  [OK     ] data/output/search_results             search_dataset output
  [OK     ] data/output/search_results_and_answer  answer_dataset output
```

**`uv run flake8 .`** — prints **nothing at all** and exits 0. Silence is
success.

**`uv run mypy . ...`**:

```
Success: no issues found in 4 source files
```

**`uv run pytest -q`**:

```
..                                                                       [100%]
2 passed in 0.03s
```

---

## Manual verification

1. `uv run python -m src` with no command prints Fire's generated help, listing
   `status` as an available command. That help text was never written by you —
   it is derived from the class and its docstrings, which is Concept 2 working.
2. `uv run python -m src status --help` shows the docstring
   *"Report the project root and which expected directories exist."*
3. Delete `data\processed`, run `status` again, and confirm that one line flips
   to `MISSING` while the others stay `OK`. Re-create it afterwards. This proves
   the command reads the filesystem per directory.
4. `Set-Location C:\` then
   `uv run --project C:\Users\user\Desktop\my_projects\python\rag_final python -m src status`
   still prints the same project root. This proves `REPO_ROOT` does not depend
   on the working directory — the property the evaluator's scripts rely on.

## Automated tests

The two tests in `tests/test_cli.py` are the whole suite for this phase, and
that is proportionate: there is one function, it prints. The tests protect the
two things that would be expensive to discover later — that `REPO_ROOT`
resolves correctly, and that paths are forward-slash relative (R7).

Do not add a test that asserts on `status`'s printed text. It would pin a
formatting choice with no behaviour behind it, and you would rewrite it every
time the output changes.

---

## Most likely errors

**1. `make: The term 'make' is not recognized`**
*Symptom:* any `make install` / `make lint` fails immediately.
*Cause:* GNU Make is not installed on this machine (verified — risk R4).
*Fix:* none needed. Use the `uv run ...` commands given above; they do exactly
what the targets do. The `Makefile` still ships because it is a submission
requirement and reviewers run it on Linux.

**2. `Makefile:10: *** missing separator. Stop.`**
*Symptom:* a reviewer on Linux runs `make install` and gets this.
*Cause:* the indented recipe lines start with spaces instead of a TAB.
*Fix:* replace the leading whitespace on every indented line in `Makefile` with
one real tab. In VS Code, click `Spaces: 4` in the status bar, choose *Indent
Using Tabs*, then *Convert Indentation to Tabs*, with the Makefile open.

**3. `No module named src`**
*Symptom:* `uv run python -m src status` fails.
*Cause:* either your shell is not in the project root, or `src/__init__.py` was
not created (or was created as a directory).
*Fix:* `Get-Location` to confirm the directory, then
`Get-ChildItem src` and confirm `__init__.py` is listed as a file of size 0.

**4. `error: Function is missing a type annotation` in `tests/test_cli.py`**
*Symptom:* `mypy` fails on the test file even though the source is clean.
*Cause:* `disallow_untyped_defs` applies to every file mypy checks, tests
included, and a test function was written without `-> None`.
*Fix:* add `-> None` to the test's signature. Every function in this project,
including tests and fixtures, carries a return annotation.

---

## Definition of done

- [ ] `C:\Users\user\Desktop\my_projects\python\rag_final` exists and contains
      `pyproject.toml`, `setup.cfg`, `Makefile`, `.gitignore`, `src/`, `tests/`.
- [ ] `uv sync` completes and produces `uv.lock` and `.venv/`.
- [ ] All six `data/` directories from §VI.7.1 exist.
- [ ] `uv run python -m src status` prints the project root and six `[OK     ]`
      lines.
- [ ] `uv run python -m src` with no arguments prints Fire's help without an
      error.
- [ ] `uv run flake8 .` prints nothing.
- [ ] `uv run mypy .` with the five mandated flags reports success on 4 files.
- [ ] `uv run pytest -q` reports 2 passed.
- [ ] `src/__main__.py` contains no hard-coded absolute path — `REPO_ROOT` is
      derived from `__file__`.

**Natural commit point:** everything above, as one commit — "runnable project
skeleton with uv, Fire CLI and lint/type/test gates". You have chosen to leave
git alone for now, so this is a marker, not an instruction.

---

## What Phase 2 adds and why

Phase 2 introduces **pydantic** and defines the models from §VI.4 that every
stage of the pipeline exchanges, then proves them against the real dataset files
by copying `data/datasets/` in and adding a `check_dataset` command that parses
them. It comes before any corpus work because the shape of the data is the one
thing that is fixed by the grader and expensive to change late — getting the
contract right first means Phases 3 through 13 all read and write the same
validated structures.

---

*Tell me when Phase 1 runs clean, or show me your code and I will review it.
I will expand `PHASE-02-the-data-contract.md` when you say you are ready.*
