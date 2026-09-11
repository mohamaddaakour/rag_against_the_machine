# Phase 1 — Search the corpus from the CLI

**Goal:** A CLI that ingests `data/raw/`, persists a lexical index, and returns ranked `(file_path, first_character_index, last_character_index)` triples for any query.
**Time:** ~4h · **Difficulty:** ●●○○○
**Depends on:** nothing — start here.

## ✅ What you'll have when this is done

The whole spine, thin but real: every text file in the vLLM corpus is read, chunked, indexed, persisted to disk, and searchable in milliseconds from a Python Fire CLI. The triples you print are already in the exact shape the grader compares — path format, index semantics, width cap.

Retrieval quality will be **bad**. That is expected and fine. Phase 1 fixes *shape*; Phase 3 fixes *quality*.

```bash
$ uv run python -m src index --max_chunk_size 2000
Chunking: 100%|███████████████████| 2121/2121 [00:09<00:00, 226.4file/s]
Vectorizing 13217 chunks...
Ingestion complete! 13217 chunks. Indices saved under data/processed

$ uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
 1  data/raw/vllm-0.10.1/docs/features/lora.md  [3400-5400]  0.2841
 2  data/raw/vllm-0.10.1/docs/features/lora.md  [5100-7100]  0.2216
 3  data/raw/vllm-0.10.1/vllm/lora/request.py  [0-1655]  0.1907
 4  data/raw/vllm-0.10.1/examples/offline_inference/multilora_inference.py  [0-2000]  0.1744
 5  data/raw/vllm-0.10.1/docs/models/supported_models.md  [40800-42800]  0.1502
```

The file counts and timings above are for the corpus exactly as shipped (2121 indexable files, 20.5 MB of text). The *ranking* will differ from run to run of your own code — what must match is the shape: forward-slash paths starting `data/raw/vllm-0.10.1/`, spans no wider than `--max_chunk_size`, exactly `k` lines.

## Where you're starting from

```
rag_against_the_machine/
├── .gitignore              # already ignores data/ and moulinette
├── Makefile                # present but BROKEN — indented with spaces, not tabs
├── README.md               # empty
├── SUBJECT.md
├── datasets_public/public/{Answered,Unanswered}Questions/*.json
├── moulinette/moulinette-{ubuntu,fedora}
├── vllm-0.10.1/            # the corpus, in the wrong place
└── src/
    ├── __init__.py         # empty
    ├── __main__.py         # empty
    └── models.py           # present, and it crashes — see step 1
```

Nothing runs yet. There is no `pyproject.toml`, so `uv sync` fails; `src/models.py` references a field name that does not exist; and `make install` dies with `missing separator` because the Makefile uses spaces where make demands tabs. This phase fixes all three on the way to a working search.

## Why this phase now

Two of the highest-consequence decisions in the whole project are made here and are painful to change later: **how a `file_path` is spelled**, and **what a character offset points into**. Both are invisible failures — a pipeline with backslashed paths runs perfectly, prints plausible results, and scores exactly zero. Getting them right before any tuning means every number you measure from Phase 2 onward is real.

## Before you start

Put the attachments where the subject says they live. Run this from the repo root in Git Bash (the layout in section VI.7.1 is not negotiable — the reference exam scripts assume it):

```bash
mkdir -p data/raw data/processed data/datasets data/output
mv vllm-0.10.1 data/raw/
cp -r datasets_public/public/AnsweredQuestions data/datasets/
cp -r datasets_public/public/UnansweredQuestions data/datasets/
cp moulinette/moulinette-ubuntu moulinette && chmod +x moulinette
ls data/raw/vllm-0.10.1/docs/features/lora.md   # must exist
```

**If the attachments were ever committed, untrack them before anything else.** In the current checkout they are absent and this check should return no paths. The subject forbids committing large data, model weights, and generated outputs; `.gitignore` cannot remove files that Git already tracks.

```bash
git ls-files vllm-0.10.1 datasets_public moulinette data
# Only if the preceding command printed attachment files:
git rm -r --cached -q vllm-0.10.1 datasets_public moulinette data
```

If the files exist only in an unshared initial commit, amend it so their blobs do
not enter shared history:

```bash
git commit --amend --no-edit
git ls-files | wc -l                    # a handful of real source files
git count-objects -vH | grep size-pack  # kilobytes, not megabytes
```

Amending rewrites the one commit you have. That is safe here because nothing has been pushed or shared. If you have already pushed, do not amend — commit the removal normally and note in the README that the corpus was tracked early on.

Then the toolchain:

```bash
uv --version        # 0.11+ — you have 0.11.31
python --version    # 3.10 or later — you have 3.12.4
```

No API keys, no accounts. Nothing downloads from the network in this phase except Python packages.

## Key design decisions

- **Where the offsets point.** The alternatives are: offsets into the raw bytes, offsets into the text after universal-newline translation, or offsets into a normalised (stripped/lowercased) copy. Recommendation: **character offsets into the UTF-8-decoded text with newline translation disabled** (`open(..., newline="")`). This is verifiable, not a guess — slicing `docs/features/lora.md[4695:6098]` this way reproduces the ground-truth span from `dataset_docs_public.json` exactly, starting on its `### Using API Endpoints` line. Changing this later invalidates every stored index.

- **Path spelling.** `str(path)`, `os.path.join`, or `Path.as_posix()`. Recommendation: **`as_posix()`, relative to the repo root**, computed at exactly one place in the codebase (`corpus.read_corpus_file`). On Windows the other two produce `data\raw\...`, which the grader compares verbatim and never matches. One function means one place to audit.

- **Chunking strategy for Phase 1.** Fixed-size windows vs. going straight to AST/heading-aware splitting. Recommendation: **fixed-size windows with 15 % overlap**. You cannot tell whether a smarter chunker helps until you can measure, and measurement arrives in Phase 2. Building the AST chunker now means tuning blind for a day. It is tracked debt, paid in Phase 3.

- **Retriever for Phase 1.** TF-IDF via scikit-learn vs. hand-rolled BM25. Recommendation: **`TfidfVectorizer`**, because it is four lines and the point of this phase is the plumbing. Phase 3 adds BM25 behind the same `Retriever.search` signature and A/Bs them.

- **Index persistence.** `joblib` pickle vs. `scipy.sparse.save_npz` + a JSON vocabulary vs. SQLite. Recommendation: **`joblib` for the vectorizer and matrix, JSONL for chunk metadata**. Pickle ties you to the scikit-learn version — real debt, but `index` rebuilds in well under the 5-minute budget, so the blast radius is one command. JSONL keeps the metadata human-greppable, which you will want in Phase 3 when a result looks wrong.

- **What goes into `chunks.jsonl`.** Metadata only, or metadata plus the chunk text. Recommendation: **metadata only**. Storing text would put ~20 MB of duplicated corpus on disk and tempt you into serving snippets from a stale index; Phase 4 reads them back from the real file by offset instead, which doubles as a check that the offsets are right.

## Debt taken on

Shortcut: one fixed-size character chunker for every file type. Bites you when: Phase 2 measures code recall and it sits near the floor, because chunks start and end mid-identifier. Paid off in: **Phase 3**.

Shortcut: `TfidfVectorizer` with its default word tokenizer. Bites you when: a question quotes `get_model_config` verbatim and the index has it as one opaque token that no paraphrase can reach. Paid off in: **Phase 3**.

Shortcut: the index is persisted with `joblib`, i.e. pickle. Bites you when: you upgrade scikit-learn and `data/processed/` stops loading. Paid off in: **not planned** — re-running `index` costs under a minute, and the README documents it.

## Files in this phase

| File | New/Edit | What it holds |
|---|---|---|
| `pyproject.toml` | new | Project metadata and the Phase 1–3 dependency set |
| `setup.cfg` | new | flake8 + mypy config — critically, `exclude = data` |
| `Makefile` | edit | Repaired: real tabs, working `clean` |
| `src/models.py` | edit | `MinimalSource`, `ScoredSource`, `Chunk` and the subject models |
| `src/corpus.py` | new | `list_corpus_files`, `read_corpus_file` — the only code that spells a path |
| `src/chunking.py` | new | `chunk_fixed` and the `chunk_file` dispatcher |
| `src/indexer.py` | new | `Indexer.build` / `Indexer.save` |
| `src/retriever.py` | new | `Retriever.load` / `Retriever.search` |
| `src/__main__.py` | edit | The Fire CLI: `index` and `search` |
| `tests/test_chunking.py` | new | The offset invariant — the most important test in the project |
| `tests/test_models.py` | new | Round-trip and span validation |

## Steps

### 1. Fix `src/models.py` so it stops crashing, and lock the data contract

**Why:** Every stage exchanges these models, and the file as it stands raises `AttributeError` on the very first `MinimalSource` you construct — `check_span` reads `self.first_character_inex`, which is not a field. Fix it now, while there is no code depending on the broken behaviour.

Two structural changes beyond the typo. First, `MinimalSource` loses its `score` field: it is the model that gets serialized into the graded JSON, and it must contain exactly the three fields the subject lists. Second, the debugging score moves to a `ScoredSource` subclass that the retriever returns and the writer never emits. Replace the whole file:

```python
# src/models.py
"""Pydantic models exchanged between the stages of the RAG pipeline."""

import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MinimalSource(BaseModel):
    """One source location: a file and a character span inside it.

    The span follows Python slicing: ``text[first:last]`` is the covered
    text, so ``last`` is exclusive.
    """

    model_config = ConfigDict(extra="allow")

    file_path: str
    first_character_index: int
    last_character_index: int

    @property
    def width(self) -> int:
        """Number of characters covered by this source."""
        return self.last_character_index - self.first_character_index

    @model_validator(mode="after")
    def check_span(self) -> "MinimalSource":
        """Reject spans that cannot address any text."""
        if self.first_character_index < 0:
            raise ValueError("first_character_index must be >= 0")
        if self.last_character_index < self.first_character_index:
            raise ValueError("last_character_index must be >= first")
        return self


class ScoredSource(MinimalSource):
    """A retrieved source plus its retrieval score.

    Internal only: never written to a graded JSON file. The writer in
    ``datasets.py`` converts back to :class:`MinimalSource` first.
    """

    score: float = 0.0


class UnansweredQuestion(BaseModel):
    """A question with no reference answer attached."""

    model_config = ConfigDict(extra="allow")

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question shipped with its reference answer and sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources plus the answer generated from them."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of ``search_dataset``."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of ``answer_dataset``."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """One indexed slice of one corpus file.

    Not part of the subject: this is the internal currency between the
    indexer and the retriever.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    indexed_text: Optional[str] = None  # ← what the retriever tokenizes

    @property
    def search_text(self) -> str:
        """Text handed to the vectorizer: enriched if set, raw otherwise."""
        return self.text if self.indexed_text is None else self.indexed_text

    def to_source(self) -> MinimalSource:
        """Drop the text and keep only what the grader compares."""
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )
```

`indexed_text` looks pointless today — nothing sets it. It is here now because Phase 3 prepends a heading trail and path words to the *searchable* text while the stored span must keep pointing at real file content only. Two separate fields from day one is what stops that change from silently shifting every offset.

`Union[AnsweredQuestion, UnansweredQuestion]` order matters: pydantic tries members left to right in smart mode and prefers the more specific match, so a dict carrying `sources` and `answer` becomes an `AnsweredQuestion`. Reversing the order would collapse every ground-truth question into a bare `UnansweredQuestion` and silently lose the sources.

**Check:**

```bash
uv run python -c "
from src.models import MinimalSource
s = MinimalSource(file_path='a.md', first_character_index=0, last_character_index=10)
print(s.model_dump_json(), s.width)
"
```
prints `{"file_path":"a.md","first_character_index":0,"last_character_index":10} 10` — three fields, no `score`. (Run this after step 2 creates `pyproject.toml`.)

### 2. Declare the project so `uv sync` works

**Why:** The reviewer and the moulinette run `uv sync` and nothing else. Until this file exists, no command in this project runs for anyone but you.

Create `pyproject.toml` at the repo root. Note there is **no `[build-system]` table**: without one, uv treats this as a virtual project, installs the dependencies into `.venv`, and does not try to build `src` as a wheel — which is what you want, since `python -m src` finds the package via the working directory.

```toml
# pyproject.toml
[project]
name = "rag-against-the-machine"
version = "0.1.0"
description = "Retrieval-Augmented Generation over the vLLM codebase"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "pydantic>=2.7",
    "fire>=0.6",
    "tqdm>=4.66",
    "numpy>=1.26",
    "scipy>=1.11",
    "scikit-learn>=1.4",
    "joblib>=1.3",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "flake8>=7.0",
    "mypy>=1.10",
]

[tool.uv]
package = false
```

Now `setup.cfg`, which configures both linters in one file. The `exclude` lines are not cosmetic: `flake8 .` and `mypy .` walk the working directory, and after the move in *Before you start* that directory contains 2000 vLLM source files. Without the exclusion `make lint` reports tens of thousands of errors in code you did not write.

```ini
# setup.cfg
[flake8]
max-line-length = 88
exclude = .git,__pycache__,.venv,data,build,dist
extend-ignore = E203

[mypy]
python_version = 3.10
exclude = (?x)(^data/|^\.venv/)
ignore_missing_imports = True
warn_return_any = True
warn_unused_ignores = True
disallow_untyped_defs = True
check_untyped_defs = True
```

`E203` is ignored because flake8 and every Python formatter disagree about whitespace before a slice colon; it is the one default rule that fights you for no benefit.

Install:

```bash
uv sync
```

**Check:** `uv run python -c "import sklearn, fire, tqdm, pydantic; print('ok')"` prints `ok`, and `ls uv.lock` shows the lockfile uv just generated. Commit `uv.lock` — the subject requires it at the root.

### 3. Walk the corpus and produce grader-exact paths

**Why:** This is the single highest-consequence function in the project. Every result you will ever emit carries a path produced here, the grader compares it verbatim, and a wrong one fails invisibly — the pipeline runs, prints plausible output, and scores zero.

Create `src/corpus.py`. It holds the extension allow-list, a size guard, `list_corpus_files(raw_dir)` returning the sorted list of indexable files, and `read_corpus_file(path, repo_root)` returning `(posix_relative_path, text)`.

```python
# src/corpus.py
"""Walk the raw corpus and decode files with grader-exact paths."""

from pathlib import Path
from typing import List, Tuple

#: Extensions worth indexing. `.txt` is in here because three ground-truth
#: sources live in CMakeLists.txt files; dropping it costs real recall.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".md", ".rst", ".txt",
        ".yaml", ".yml", ".toml", ".cfg", ".in", ".sh",
    }
)

#: Directories that never contain answers, only noise.
SKIP_DIRS = frozenset({".git", "__pycache__", ".mypy_cache", "node_modules"})

#: A single file larger than this is a generated blob, not documentation.
MAX_FILE_BYTES = 2_000_000


def is_indexable(path: Path) -> bool:
    """True when *path* is a text file worth putting in the index."""
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    if SKIP_DIRS.intersection(path.parts):
        return False
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def list_corpus_files(raw_dir: Path) -> List[Path]:
    """Every indexable file under *raw_dir*, in a stable order.

    Sorting matters: it makes chunk ids reproducible across runs, which is
    what lets you diff two indexes when a change moves recall.
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {raw_dir}")
    return sorted(p for p in raw_dir.rglob("*") if p.is_file() and is_indexable(p))


def read_corpus_file(path: Path, repo_root: Path) -> Tuple[str, str]:
    """Decode *path* and return ``(relative_posix_path, text)``.

    The path is relative to the repository root, so it starts with
    ``data/raw/`` exactly as the grader expects. ``newline=""`` disables
    universal-newline translation: character offsets must index into the
    file as it is on disk, not into a copy with CRLF collapsed to LF.
    """
    relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        return relative, handle.read()
```

Three details carry the weight. `as_posix()` is the only path formatting in the codebase — audit this one line and you have audited every path you emit. `newline=""` keeps offsets aligned with the bytes on disk; `read_text()` would silently turn `\r\n` into `\n` and shift every subsequent index in that file. `errors="replace"` maps each undecodable byte to one replacement character, so a corrupt file costs you that file's accuracy instead of aborting a 5-minute index.

**Check:**

```bash
uv run python -c "
from pathlib import Path
from src.corpus import list_corpus_files, read_corpus_file
root = Path('.').resolve()
files = list_corpus_files(root / 'data' / 'raw')
print(len(files), 'files')
lora = root / 'data/raw/vllm-0.10.1/docs/features/lora.md'
rel, text = read_corpus_file(lora, root)
print(rel)
print(repr(text[4695:4730]))
"
```
must print `2121 files`, then `data/raw/vllm-0.10.1/docs/features/lora.md` (forward slashes, no `./`, no drive letter), then `'### Using API Endpoints\n\nLoading a LoR'`. That last line is the ground-truth span from the public dataset. If it matches, your offset convention is correct and provably so. If it does not, stop here — nothing downstream can be right.

### 4. Chunk every file with one fixed-size splitter

**Why:** You need chunks to index. You do not yet need *good* chunks, and there is nothing to tune against until Phase 2 gives you a number.

Create `src/chunking.py` with the overlap constant, `chunk_fixed`, and a `chunk_file` dispatcher that today only forwards to `chunk_fixed`. The dispatcher exists now so that Phase 3 adds strategies without touching `indexer.py`.

```python
# src/chunking.py
"""Split corpus files into chunks carrying exact character offsets."""

from typing import List

from src.models import Chunk

#: Fraction of a chunk repeated at the start of the next one.
OVERLAP_RATIO = 0.15


def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Slide a fixed window over *text*, keeping absolute offsets.

    Args:
        file_path: Grader-exact path, stored on every chunk.
        text: The file content, exactly as decoded.
        max_chunk_size: Hard upper bound on chunk width, in characters.

    Returns:
        Chunks in document order. Blank chunks are dropped.

    Raises:
        ValueError: If *max_chunk_size* is not positive.
    """
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be > 0")

    step = max(1, max_chunk_size - int(max_chunk_size * OVERLAP_RATIO))
    chunks: List[Chunk] = []
    for start in range(0, max(len(text), 1), step):
        end = min(start + max_chunk_size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end,  # ← exclusive, Python slicing
                    text=piece,
                )
            )
        if end >= len(text):
            break
    return chunks


def chunk_file(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Dispatch to the right chunking strategy for *file_path*.

    Phase 1 has exactly one strategy. Phase 3 adds Python and Markdown
    strategies here; nothing outside this module changes.
    """
    return chunk_fixed(file_path, text, max_chunk_size)
```

The overlap is what makes a truth span that straddles a boundary still reachable: with 15 % overlap and the metric's 5 % IoU bar, a span sitting across two chunks is caught by whichever chunk holds more of it. `end` is exclusive throughout the project — `text[first:last]` is the covered text — so a chunk's width is exactly `last - first`, which is what the moulinette caps at `max_context_length`.

**Check:**

```bash
uv run python -c "
from src.chunking import chunk_fixed
text = 'abcdefghij' * 30
cs = chunk_fixed('x.md', text, 100)
print(len(cs), [(c.first_character_index, c.last_character_index) for c in cs][:4])
print(all(text[c.first_character_index:c.last_character_index] == c.text for c in cs))
print(max(c.last_character_index - c.first_character_index for c in cs))
"
```
prints `4 [(0, 100), (85, 185), (170, 270), (255, 300)]`, then `True`, then `100`.

### 5. Build and persist the index

**Why:** Search must be instant. The budget is 200 questions in 90 seconds, which is a load-once-query-many budget, not a search-the-files-each-time budget.

Create `src/indexer.py`. `Indexer` holds `max_chunk_size` and the accumulated chunk list; `build` walks and chunks the corpus under a tqdm bar; `save` writes three artefacts into `data/processed/`.

```python
# src/indexer.py
"""Build the lexical index over the chunked corpus and persist it."""

import json
from pathlib import Path
from typing import List

# joblib is used to save Python/ML objects to disk.
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.chunking import chunk_file
from src.corpus import list_corpus_files, read_corpus_file
from src.models import Chunk

CHUNKS_FILE = "chunks.jsonl"
TFIDF_FILE = "tfidf.joblib"
META_FILE = "meta.json"


class Indexer:
    """Turn a corpus directory into a searchable, persisted index."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunks: List[Chunk] = []

    def build(self, raw_dir: Path, repo_root: Path) -> None:
        """Read and chunk every indexable file under *raw_dir*."""
        paths = list_corpus_files(raw_dir)
        self.chunks = []
        for path in tqdm(paths, desc="Chunking", unit="file"):
            try:
                file_path, text = read_corpus_file(path, repo_root)
            except OSError as exc:  # ← one bad file must not kill the run
                tqdm.write(f"skipped {path}: {exc}")
                continue
            self.chunks.extend(chunk_file(file_path, text, self.max_chunk_size))

    def save(self, processed_dir: Path) -> None:
        """Write chunk metadata, the fitted vectorizer and the matrix."""
        if not self.chunks:
            raise ValueError("nothing to save: build() produced no chunks")

        processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = processed_dir / CHUNKS_FILE
        with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.chunks:
                handle.write(chunk.to_source().model_dump_json() + "\n")

        print(f"Vectorizing {len(self.chunks)} chunks...")
        vectorizer = TfidfVectorizer(sublinear_tf=True)
        matrix = vectorizer.fit_transform(c.search_text for c in self.chunks)
        joblib.dump(
            {"vectorizer": vectorizer, "matrix": matrix},
            processed_dir / TFIDF_FILE,
        )

        meta = {
            "max_chunk_size": self.max_chunk_size,
            "n_chunks": len(self.chunks),
            "n_features": int(matrix.shape[1]),
        }
        (processed_dir / META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
```

`chunk.to_source()` is what keeps 20 MB of duplicated corpus text off your disk: the JSONL holds only the three fields a result needs, one compact line per chunk. `sublinear_tf=True` uses `1 + log(tf)` instead of the raw count, which stops a chunk that repeats one word forty times from dominating the ranking — a real pattern in generated vLLM code.

The `matrix` is L2-normalised by `TfidfVectorizer` default, which is why the retriever can use a plain dot product as cosine similarity in the next step.

**Check:**

```bash
uv run python -m src index --max_chunk_size 2000   # after step 7 wires the CLI
ls -la data/processed/
head -1 data/processed/chunks.jsonl
cat data/processed/meta.json
```
`chunks.jsonl` line 1 must be a three-field JSON object with a `data/raw/...` path.

### 6. Load the index and rank a query against it

**Why:** This is the half of the spine that the grader actually reads. Everything from Phase 2 on calls exactly this method.

Create `src/retriever.py`. `Retriever.load` is a classmethod reading `data/processed/`; `search` transforms the query with the *same fitted vectorizer* and returns the top-k `ScoredSource` in descending score order.

```python
# src/retriever.py
"""Load a persisted index and rank chunks against a query."""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from src.indexer import CHUNKS_FILE, TFIDF_FILE
from src.models import MinimalSource, ScoredSource


class Retriever:
    """Rank indexed chunks against a natural-language query."""

    def __init__(
        self,
        sources: List[MinimalSource],
        vectorizer: Any,
        matrix: Any,
    ) -> None:
        self.sources = sources
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def load(cls, processed_dir: Path) -> "Retriever":
        """Read the artefacts written by :meth:`Indexer.save`.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        chunks_path = processed_dir / CHUNKS_FILE
        tfidf_path = processed_dir / TFIDF_FILE
        if not chunks_path.exists() or not tfidf_path.exists():
            raise FileNotFoundError(
                f"no index in {processed_dir} - run: python -m src index"
            )
        sources: List[MinimalSource] = []
        with chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sources.append(MinimalSource.model_validate_json(line))
        payload: Dict[str, Any] = joblib.load(tfidf_path)
        return cls(sources, payload["vectorizer"], payload["matrix"])

    def search(self, query: str, k: int = 10) -> List[ScoredSource]:
        """Top-*k* sources for *query*, best first.

        Returns an empty list for an empty query, a non-positive *k*, or a
        query whose every term is absent from the vocabulary.
        """
        if k <= 0 or not query.strip():
            return []
        vector = self.vectorizer.transform([query])
        if vector.nnz == 0:  # ← no query term is in the vocabulary
            return []
        scores = np.asarray((self.matrix @ vector.T).todense()).ravel()
        k = min(k, scores.size)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        results: List[ScoredSource] = []
        for position in top:
            if scores[position] <= 0.0:
                break
            source = self.sources[int(position)]
            results.append(
                ScoredSource(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                    score=float(scores[position]),
                )
            )
        return results
```

`np.argpartition` finds the top k in O(n) instead of sorting all 13 000 scores; the second line sorts just those k. At this corpus size it is not the difference between passing and failing the perf budget, but at Phase 3 chunk sizes you run this 200 times and it is free to do right.

Dropping results with a zero score matters more than it looks: without it, a nonsense query returns k arbitrary chunks with score 0.0, which reads as a working search and quietly inflates nothing while confusing you for an hour.

**Check:** covered by the CLI check in step 7.

### 7. Wire the Fire CLI

**Why:** The reference exam scripts invoke `uv run python -m src <command>` and nothing else. If that exact invocation does not work, the automated checks fail by design no matter how good your recall is.

Replace the empty `src/__main__.py`. `Cli` exposes one method per subject command — `index` and `search` in this phase — and every path is a real argument with a default, never a hardcoded literal inside a function body.

```python
# src/__main__.py
"""Command-line entry point: uv run python -m src <command>."""

from pathlib import Path

import fire

from src.indexer import Indexer
from src.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = str(REPO_ROOT / "data" / "raw")
DEFAULT_PROCESSED_DIR = str(REPO_ROOT / "data" / "processed")


class Cli:
    """Retrieval-Augmented Generation over the vLLM codebase."""

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_dir: str = DEFAULT_RAW_DIR,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Ingest *raw_dir* and persist the index under *processed_dir*."""
        indexer = Indexer(max_chunk_size=int(max_chunk_size))
        indexer.build(Path(raw_dir), REPO_ROOT)
        indexer.save(Path(processed_dir))
        print(
            f"Ingestion complete! {len(indexer.chunks)} chunks. "
            f"Indices saved under {processed_dir}"
        )

    def search(
        self,
        query: str,
        k: int = 10,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Print the top-*k* sources for a single *query*."""
        retriever = Retriever.load(Path(processed_dir))
        sources = retriever.search(str(query), int(k))  # ← str(): see below
        if not sources:
            print("No results.")
            return
        for rank, source in enumerate(sources, start=1):
            print(
                f"{rank:2d}  {source.file_path}  "
                f"[{source.first_character_index}-"
                f"{source.last_character_index}]  {source.score:.4f}"
            )


def main() -> None:
    """Hand the CLI class to Fire."""
    fire.Fire(Cli, name="python -m src")


if __name__ == "__main__":
    main()
```

`str(query)` and `int(k)` are not defensive noise. Fire literal-evaluates every argument before it reaches your function, so `search 2000` hands you the integer `2000`, and `.strip()` on an int raises `AttributeError` — a traceback, from a CLI the subject says must never produce one. Coercing at the boundary costs two calls and removes the whole class of bug.

`REPO_ROOT` is derived from `__file__`, not from `os.getcwd()`. That is what makes `read_corpus_file`'s `relative_to` produce `data/raw/...` regardless of which directory the reviewer runs from.

**Check:**

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
uv run python -m src search "AsyncLLMEngine" --k 3
uv run python -m src -- --help
```
The first prints a tqdm bar then the completion line; the searches print 5 and 3 lines with `data/raw/vllm-0.10.1/...` paths; `--help` lists `index` and `search`. (The bare `--` before `--help` is Fire's convention for "this flag is for Fire, not for my command".)

### 8. Pin the offset invariant with tests

**Why:** Phase 3 rewrites the chunker completely. This test is the thing that tells you the rewrite did not shift every index by one — a bug that shows up only as recall you cannot explain.

Create `tests/test_chunking.py` and `tests/test_models.py`.

```python
# tests/test_chunking.py
"""The offset invariant: a chunk span must reproduce the chunk text."""

import pytest

from src.chunking import chunk_file, chunk_fixed

SAMPLE = (
    "# Title\n\nSome prose about serving models.\n\n"
    "## Section\n\ndef get_model_config(name: str) -> Config:\n"
    "    return Config(name)\n" * 12
)


def test_span_reproduces_text() -> None:
    for chunk in chunk_fixed("data/raw/x.md", SAMPLE, 200):
        sliced = SAMPLE[
            chunk.first_character_index : chunk.last_character_index
        ]
        assert sliced == chunk.text


def test_no_chunk_exceeds_the_cap() -> None:
    for chunk in chunk_fixed("data/raw/x.md", SAMPLE, 200):
        assert chunk.last_character_index - chunk.first_character_index <= 200


def test_chunks_cover_every_character() -> None:
    chunks = chunk_fixed("data/raw/x.md", SAMPLE, 200)
    assert chunks[0].first_character_index == 0
    assert chunks[-1].last_character_index == len(SAMPLE)


def test_empty_and_blank_files_produce_nothing() -> None:
    assert chunk_file("data/raw/x.md", "", 200) == []
    assert chunk_file("data/raw/x.md", "   \n\n  ", 200) == []


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_fixed("data/raw/x.md", SAMPLE, 0)
```

```python
# tests/test_models.py
"""The graded JSON shape, pinned."""

import json

import pytest
from pydantic import ValidationError

from src.models import Chunk, MinimalSource, ScoredSource


def test_minimal_source_serializes_exactly_three_fields() -> None:
    source = MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/features/lora.md",
        first_character_index=4695,
        last_character_index=6098,
    )
    assert set(json.loads(source.model_dump_json())) == {
        "file_path",
        "first_character_index",
        "last_character_index",
    }
    assert source.width == 1403


def test_scored_source_is_not_a_minimal_source_on_the_wire() -> None:
    scored = ScoredSource(
        file_path="a.md",
        first_character_index=0,
        last_character_index=5,
        score=0.42,
    )
    assert "score" in json.loads(scored.model_dump_json())


def test_reversed_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MinimalSource(
            file_path="a.md",
            first_character_index=10,
            last_character_index=3,
        )


def test_chunk_search_text_defaults_to_text() -> None:
    chunk = Chunk(
        file_path="a.md",
        first_character_index=0,
        last_character_index=3,
        text="abc",
    )
    assert chunk.search_text == "abc"
    enriched = chunk.model_copy(update={"indexed_text": "docs lora abc"})
    assert enriched.search_text == "docs lora abc"
```

**Check:** `uv run pytest -q` prints `9 passed`. Then add `1` to `end` in `chunk_fixed` and run again — `test_span_reproduces_text` must fail. If it still passes, the test is not testing anything; undo the sabotage either way.

### 9. Repair the Makefile

**Why:** `make install` is how the reviewer installs your project, and today it dies on line 12 with `missing separator`. The rules are indented with spaces; make requires a **tab**.

The `clean` recipe is also corrupted mid-string, so it would be a syntax error even with correct indentation. Rewrite the whole file. **Every indented line below must begin with a real tab character** — if your editor is set to expand tabs, turn that off for this file.

```make
# Makefile
UV := uv
ARGS ?=
MYPY_FLAGS := --warn-return-any --warn-unused-ignores \
              --ignore-missing-imports --disallow-untyped-defs \
              --check-untyped-defs

.DEFAULT_GOAL := help
.PHONY: install run debug clean lint lint-strict test index search help

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

index:
	$(UV) run python -m src index --max_chunk_size $(or $(SIZE),2000)

search:
	$(UV) run python -m src search "$(Q)" --k $(or $(K),5)

clean:
	$(UV) run python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts and 'data' not in p.parts]"
	$(UV) run python -c "import shutil; [shutil.rmtree(d, ignore_errors=True) for d in ('.mypy_cache', '.pytest_cache', '.ruff_cache')]"

help:
	@echo "install run debug lint lint-strict test index search clean"
```

The `clean` recipe filters out `.venv` and `data` explicitly. Without that filter it walks into `data/raw/vllm-0.10.1` and deletes 2000 vLLM `__pycache__` directories, which is slow and, the first time it happens, alarming.

**Check:**

```bash
cat -A Makefile | sed -n '11,13p'
```
The recipe line must start with `^I` (a tab), not spaces. Then `make install` and `make test` both run.

## Common pitfalls

| Pitfall | Why it happens | Avoid it by |
|---|---|---|
| Backslashes in `file_path` | Developing on Windows and reaching for `str(path)` or `os.path.join` | `.as_posix()` in `read_corpus_file` and nowhere else; the step 3 check catches it in five seconds |
| Offsets shifted by CRLF translation | `Path.read_text()` silently collapses `\r\n` to `\n` | `open(..., newline="")`, and verify against the known `lora.md[4695:6098]` span |
| Offsets point into cleaned text | Lowercasing or stripping before chunking and forgetting the offsets moved | Chunk the raw text; all normalisation happens inside the vectorizer, downstream of offsets |
| `make install` fails with `missing separator` | Makefile indented with spaces — it already is | `cat -A Makefile` and look for `^I` |
| `flake8 .` reports 40 000 errors | It walks into `data/raw/vllm-0.10.1` | `exclude = ...,data` in `setup.cfg`, before you ever run `make lint` |
| `python -m src` fails with `No module named src` | Running from a subdirectory, or `src/__init__.py` deleted | Always run from the repo root; keep `__init__.py` present and empty |
| `AttributeError: 'int' object has no attribute 'strip'` | Fire literal-evaluated `search 2000` into an integer | `str(query)` / `int(k)` at the CLI boundary |
| Skipping `.txt` files as "not code, not docs" | It looks like noise next to `.py` and `.md` | Three ground-truth sources are `CMakeLists.txt`; keep `.txt` |
| The corpus stays in git despite `.gitignore` | `.gitignore` has no effect on already-tracked files | `git rm -r --cached` first, as in *Before you start*; check with `git ls-files \| wc -l` |
| A `UnicodeDecodeError` two thousand files into the walk | One non-UTF-8 file in a 2121-file corpus | `errors="replace"` plus the per-file `try/except OSError` |

## Verify it's done

```bash
rm -rf data/processed
uv sync
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5
uv run python -m src search "" --k 5
uv run python -m src search "zzzzqqqqxxxx" --k 5
uv run pytest -q
uv run python -c "
import json
from pathlib import Path
lines = Path('data/processed/chunks.jsonl').read_text(encoding='utf-8').splitlines()
rows = [json.loads(line) for line in lines]
assert all(r['file_path'].startswith('data/raw/vllm-0.10.1/') for r in rows)
assert all('\\\\' not in r['file_path'] for r in rows)
widths = [r['last_character_index'] - r['first_character_index'] for r in rows]
print(len(rows), 'chunks, max width', max(widths))
"
```

Expected:

```
Chunking: 100%|███████████████████| 2121/2121 [00:09<00:00, 226.4file/s]
Vectorizing 13217 chunks...
Ingestion complete! 13217 chunks. Indices saved under .../data/processed
 1  data/raw/vllm-0.10.1/docs/features/lora.md  [3400-5400]  0.2841
 2  ... four more lines ...
No results.
No results.
9 passed
13217 chunks, max width 2000
```

The exact chunk count and the ranking depend on your allow-list and are yours; the four things that must hold are: the index finishes in well under 5 minutes, every path starts `data/raw/vllm-0.10.1/` with forward slashes, no chunk exceeds 2000 characters, and both degenerate queries print `No results.` instead of a traceback.

Spot-check the round trip by hand, which is the check that no test can fake:

```bash
uv run python -c "
from pathlib import Path
from src.corpus import read_corpus_file
root = Path('.').resolve()
p = root / 'data/raw/vllm-0.10.1/docs/features/lora.md'
_, text = read_corpus_file(p, root)
print(text[4695:4760])
"
```
must print the ground-truth section header `### Using API Endpoints` and the two lines after it.

## Definition of done

- [ ] `uv sync` works from a clean checkout and `uv.lock` exists at the root
- [ ] `make install`, `make test` and `make lint` all run (lint may still report style errors — that is Phase 5)
- [ ] `index` completes in under 5 minutes and writes `chunks.jsonl`, `tfidf.joblib`, `meta.json`
- [ ] `search` prints exactly `k` lines with verbatim-correct corpus paths
- [ ] `text[first:last]` on any emitted span reproduces the indexed chunk
- [ ] No chunk exceeds `--max_chunk_size`
- [ ] Empty and nonsense queries print `No results.`, not a traceback
- [ ] `uv run pytest -q` passes and `test_span_reproduces_text` fails when sabotaged
- [ ] `data/` is not in `git status`, and `git ls-files` no longer lists any vLLM file
- [ ] Committed

## Deliberately NOT in this phase

- Dataset-level search and the graded JSON output → **Phase 2**
- Any recall number at all → **Phase 2**
- Python-AST and Markdown-heading chunking, BM25, the identifier tokenizer → **Phase 3**
- Anything involving Qwen → **Phase 4**
- Graceful handling of missing files, malformed JSON, `k=0` across every command → **Phase 5** (this phase handles only the two search cases above)
- flake8/mypy cleanliness → **Phase 5** (but do not let it rot: run `make lint` now and look at the count)
- All five bonuses → not in v1

## Commit

```bash
git add -A
git commit -m "phase 1: index the vLLM corpus and search it from the CLI"
```

## Next

→ **[Phase 2 — Measure your own recall](PHASE-02-measure-your-own-recall.md)**
