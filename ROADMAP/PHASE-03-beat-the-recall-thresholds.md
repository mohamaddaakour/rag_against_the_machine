# Phase 3 — Beat the recall thresholds

**Goal:** Cross 80 % recall@5 on docs and 50 % on code with language-aware chunking and a code-aware BM25 retriever — and be able to explain every point of the gain.
**Time:** ~6h · **Difficulty:** ●●●●○
**Depends on:** Phase 2 complete. The measurement loop is not optional here.

## ✅ What you'll have when this is done

The project's actual grade. Python files cut at `def` and `class` boundaries — including methods inside classes, which is where the ground truth actually lives. Markdown cut at headings, which is provably how the reference corpus was split. A tokenizer that turns `get_model_config` into four searchable words. And a BM25 ranker you chose because it measured better, not because a blog post recommended it.

```bash
$ uv run python -m src index --max_chunk_size 1200
Chunking: 100%|███████████████████| 2121/2121 [00:24<00:00, 87.1file/s]
Tokenizing 21259 chunks...
Vectorizing (tfidf + bm25)...
Ingestion complete! 21259 chunks. Indices saved under data/processed

$ uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
Questions scored: 100
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
```

Those are the shape of a passing run, not a promise. The bars are `docs R@5 ≥ 0.80` and `code R@5 ≥ 0.50`; aim to clear them with margin, because the defense dataset is not this one.

## Where you're starting from

```
src/
├── __main__.py     # index, search, search_dataset, evaluate
├── models.py · corpus.py · chunking.py   # chunk_fixed + a pass-through dispatcher
├── indexer.py      # TF-IDF only
├── retriever.py    # TF-IDF only
├── datasets.py · evaluation.py
benchmarks.md       # row 0: the baseline you are about to beat
```

You can retrieve, you can score, and you know your baseline is well short of both bars.

## Why this phase now

Everything before this was plumbing; everything after is presentation. The two thresholds are the only hard numeric bars in the subject, and they are the one part of the project that can fail for reasons an afternoon cannot fix. Do it while you still have runway — and do it with the measurement loop already built, so every hour of work produces a number instead of a feeling.

## Before you start

No new dependencies. `ast` and `re` are standard library, and BM25 is forty lines over the sparse matrix scikit-learn already gives you.

Open `benchmarks.md` beside your editor. You are about to add a row per experiment, and the discipline of one change per row is what makes this phase finish in six hours instead of sixteen.

Two facts to keep in front of you, both measured from the public datasets rather than assumed:

- A docs ground-truth span starts exactly on a `###` heading line (`docs/features/lora.md[4695:6098]`). Heading-aware Markdown chunking is not a heuristic here; it is reconstructing how the reference was built.
- A code ground-truth span starts at `@property\n    def activation_formats(` — **a decorated method inside a class**, not a top-level definition. A Python chunker that only cuts at module level will systematically miss these.

## Key design decisions

- **Python chunking unit.** Fixed characters, top-level AST nodes only, or top-level nodes plus methods inside classes. Recommendation: **top-level nodes plus class members**, decorators included in the span. The evidence above is decisive: cutting only at module level puts a 15 000-character class into one segment, which then gets windowed arbitrarily and lands nowhere near the truth boundary. Fall back to `chunk_fixed` when `ast.parse` raises — vLLM ships files that will not parse under your interpreter, and a `SyntaxError` must degrade, not abort a 2100-file walk.

- **Oversized definitions.** Drop them, or window them. Recommendation: **window them with `chunk_fixed`**, reusing the exact primitive Phase 1 tested. Some vLLM functions run past 2000 characters and they contain real answers; dropping them is throwing away recall to keep the code tidy.

- **Markdown chunking unit.** Paragraphs, fixed windows, or headings. Recommendation: **headings**, with sections below a quarter of the cap merged into their neighbour and oversized sections windowed. Docs questions are answered by one section almost every time.

- **What the enriched `indexed_text` contains.** Nothing, path words only, or path words plus the heading trail / enclosing definition. Recommendation: **path words plus the heading trail.** A question about LoRA should be able to match `docs/features/lora.md` even when the section body never spells the word. Watch this one with the benchmark, though: it also lets an irrelevant chunk match on path words alone, so it is a measured change, not an obviously good one.

- **Tokenization.** scikit-learn's default `\w\w+` pattern, a stemmer, or a custom identifier splitter. Recommendation: **a custom analyzer emitting both the whole identifier and its parts** — `get_model_config` becomes `get_model_config`, `get`, `model`, `config`. Keeping both is what lets a verbatim quote and a paraphrase hit the same chunk. In a code corpus this is usually the single largest jump in code recall.

- **Where tokenization happens.** Inside each vectorizer via `analyzer=analyze`, or once up front. Recommendation: **once up front**, with both vectorizers set to `analyzer=identity`. A Python analyzer callable is the slow part of indexing, and running it twice over 27 MB of text is how you blow the 5-minute budget for nothing.

- **Retriever: keep TF-IDF or switch to BM25.** Recommendation: **implement BM25 behind the same `search` signature and measure both**, defaulting to whichever wins. BM25's length normalisation usually helps on a corpus that mixes 200-character and 1200-character chunks. Keep `--retriever` so the loser stays runnable: the comparison is a README section and a near-certain defense question.

- **Chunk size.** Recommendation: **sweep 2000, 1200 and 800, then pick.** Not taste — a ceiling. Best achievable IoU for a chunk of width `C` against a truth span of width `W` is `min(W,C)/max(W,C)`, so at 2000 the *theoretical maximum* recall is 0.96 docs / 0.98 code, at 900 it is 0.99 / 0.99. Smaller chunks also raise index size and search time, and they shrink the context Phase 4 feeds the model. Measure, then report the effect — the subject explicitly asks for it.

## Debt taken on

Shortcut: BM25 parameters and chunk size tuned against the *public* datasets. Bites you when: the defense dataset is drawn differently and a one-point margin evaporates. Paid off in: **not planned** — mitigated by keeping the tuning coarse (three chunk sizes, stock `k1`/`b` unless a change is worth several points) and by aiming for a comfortable margin over the bars rather than a hair above them.

## Files in this phase

| File | New/Edit | What it holds |
|---|---|---|
| `src/analyzer.py` | new | `analyze`, `split_identifier`, `identity` |
| `src/bm25.py` | new | `bm25_weights`, `bm25_scores` |
| `src/chunking.py` | edit | `chunk_python`, `chunk_markdown`, a real `chunk_file` dispatcher |
| `src/indexer.py` | edit | Tokenize once, build both indices |
| `src/retriever.py` | edit | `--retriever` switch, BM25 scoring path |
| `src/__main__.py` | edit | `--retriever` on `search` and `search_dataset` |
| `tests/test_chunking.py` | edit | The offset invariant, now over all three strategies |
| `benchmarks.md` | edit | One row per experiment |

## Steps

### 1. Write the identifier-aware analyzer

**Why:** Highest leverage change in the phase, and completely invisible until you measure it. `re.findall(r"\w\w+", "get_model_config")` returns one opaque token; no question that says "model config" will ever match it.

Create `src/analyzer.py`. `split_identifier` breaks a token into its `snake_case` and `CamelCase` parts; `analyze` emits the whole token plus every part; `identity` is the pass-through the vectorizers use once the indexer has already tokenized.

```python
# src/analyzer.py
"""Identifier-aware tokenizer shared by both retrievers."""

import re
from typing import List

#: Words and bare numbers. Identifiers stay whole at this stage.
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")

#: One CamelCase / snake_case part. The first branch keeps acronym runs
#: together: "LLMEngine" splits into "LLM" and "Engine", not L-L-M-Engine.
PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

#: English function words only. Never put short code tokens in here:
#: "id", "kv", "gpu", "tp" and "os" all carry real signal in this corpus.
STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "to", "in", "on", "for", "with", "and",
        "or", "is", "are", "was", "be", "been", "do", "does", "did",
        "how", "what", "which", "when", "where", "why", "who", "whom",
        "this", "that", "these", "those", "it", "its", "as", "at", "by",
        "from", "into", "can", "could", "should", "would", "will",
        "you", "your", "i", "we", "they", "there", "then", "than",
    }
)

MIN_PART_LENGTH = 2


def split_identifier(token: str) -> List[str]:
    """Break an identifier into its snake_case / CamelCase parts."""
    return PART_RE.findall(token.replace("_", " "))


def analyze(text: str) -> List[str]:
    """Tokenize *text* into whole identifiers plus their parts.

    ``get_model_config`` yields ``["get_model_config", "get", "model",
    "config"]``; ``AsyncLLMEngine`` yields ``["asyncllmengine", "async",
    "llm", "engine"]``. Keeping both forms is what lets a question that
    quotes an identifier verbatim and a question that paraphrases it hit
    the same chunk.
    """
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(text):
        whole = raw.lower()
        if whole not in STOPWORDS:
            tokens.append(whole)
        parts = split_identifier(raw)
        if len(parts) < 2:  # ← a single-part token is already emitted
            continue
        for part in parts:
            lowered = part.lower()
            if lowered == whole or len(lowered) < MIN_PART_LENGTH:
                continue
            if lowered in STOPWORDS:
                continue
            tokens.append(lowered)
    return tokens


def identity(tokens: List[str]) -> List[str]:
    """Pass-through analyzer for text already tokenized by `analyze`.

    Must be a module-level function, not a lambda: scikit-learn pickles
    the vectorizer by reference and a lambda is not picklable.
    """
    return tokens
```

The stopword list is deliberately English-only and short. A general-purpose list would strip `is`, `in`, `not` and `return` — which are Python keywords appearing in nearly every chunk, so BM25's IDF already flattens them to nothing — while also stripping `id` and `no`, which do carry signal here. Let the ranking function handle high-frequency terms; use the stopword list only for question filler.

**Check:**

```bash
uv run python -c "
from src.analyzer import analyze
print(analyze('get_model_config'))
print(analyze('AsyncLLMEngine'))
print(analyze('How do I use the v1 KV cache?'))
"
```
prints
```
['get_model_config', 'get', 'model', 'config']
['asyncllmengine', 'async', 'llm', 'engine']
['use', 'v1', 'kv', 'cache']
```
Read that last line out loud: every question word is gone — `how`, `do`, `i`, `the` are all in `STOPWORDS` — and `v1` and `kv` survived. If `kv` disappeared, your stopword list is too aggressive.

### 2. Implement BM25 over a sparse count matrix

**Why:** You must justify your ranking choice at the defense, and "I implemented both and measured" is the only answer that survives a follow-up question. Forty lines, no new dependency, and it is the classic lexical ranker the subject names.

Create `src/bm25.py`. `bm25_weights` precomputes the full per-`(document, term)` BM25 weight **at index time**, so a query is just a column selection and a row sum. `bm25_scores` does that selection.

```python
# src/bm25.py
"""BM25 ranking over a sparse term-count matrix.

The weight of term *t* in document *d* does not depend on the query, so
it is computed once at index time. Scoring a query then reduces to
selecting the query's columns and summing across them.
"""

from typing import Sequence

import numpy as np
from scipy import sparse

K1 = 1.2
B = 0.75


def bm25_weights(
    counts: sparse.csr_matrix,
    k1: float = K1,
    b: float = B,
) -> sparse.csc_matrix:
    """Precompute BM25 weights for every stored (document, term) pair.

    Args:
        counts: Raw term counts, documents in rows.
        k1: Term-frequency saturation. Higher means repeats count more.
        b: Length normalisation, 0 = off, 1 = full.

    Returns:
        The same sparsity pattern, values replaced by BM25 weights, in
        CSC layout because scoring slices columns.
    """
    counts = counts.tocsr()
    n_docs = counts.shape[0]
    doc_length = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = float(doc_length.mean()) or 1.0

    doc_freq = np.diff(counts.tocsc().indptr)  # ← nonzeros per column
    idf = np.log(
        1.0 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5)
    ).astype(np.float32)

    rows = np.repeat(np.arange(n_docs), np.diff(counts.indptr))
    freq = counts.data.astype(np.float32)
    norm = k1 * (1.0 - b + b * doc_length[rows] / average_length)
    weights = idf[counts.indices] * freq * (k1 + 1.0) / (freq + norm)

    scored = sparse.csr_matrix(
        (weights.astype(np.float32), counts.indices.copy(), counts.indptr.copy()),
        shape=counts.shape,
    )
    return scored.tocsc()


def bm25_scores(
    weights: sparse.csc_matrix,
    term_ids: Sequence[int],
) -> np.ndarray:
    """Score every document against the query terms in *term_ids*.

    Repeated ids are kept on purpose: a term the query uses twice
    contributes twice, which is what the BM25 query-side weighting does.
    """
    if len(term_ids) == 0:
        return np.zeros(weights.shape[0], dtype=np.float32)
    selected = weights[:, list(term_ids)]
    return np.asarray(selected.sum(axis=1)).ravel()
```

The IDF formula is the "plus one" variant, `ln(1 + (N - df + 0.5)/(df + 0.5))`. The textbook form without the `1 +` goes *negative* for a term appearing in more than half the documents, which in this corpus means `self`, `import` and `def` actively push a chunk down the ranking. The `1 +` floors it at zero and is what every real implementation uses.

`counts.indptr.copy()` is not paranoia — reusing the array would alias the caller's matrix and leave you debugging a matrix that changed under you.

**Check:**

```bash
uv run python -c "
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.bm25 import bm25_scores, bm25_weights
docs = ['lora adapter serving', 'lora lora lora', 'quantization kernels', 'serving models']
cv = CountVectorizer()
w = bm25_weights(cv.fit_transform(docs))
ids = [cv.vocabulary_[t] for t in ['lora']]
print(np.round(bm25_scores(w, ids), 3))
"
```
prints `[0.641 1.044 0.    0.   ]` — document 1 outranks document 0 because it repeats the term, and documents 2 and 3 score exactly zero. If every document scores non-zero, you sliced rows instead of columns.

### 3. Cut Python files at definition boundaries

**Why:** A chunk that starts mid-function and ends mid-docstring matches nothing and answers nothing. And the ground truth proves the boundary that matters is the *method*, not the module-level definition.

Replace `src/chunking.py`. `chunk_fixed` is unchanged from Phase 1 — it is still the windowing primitive and its test still guards it. Everything else is new.

```python
# src/chunking.py
"""Split corpus files into chunks carrying exact character offsets."""

import ast
import re
from typing import List, Sequence, Tuple

from src.models import Chunk

#: Fraction of a chunk repeated at the start of the next one.
OVERLAP_RATIO = 0.15

#: Segments below this fraction of the cap are merged into their
#: neighbour: a three-line `def` is not a useful retrieval unit.
MIN_SEGMENT_RATIO = 0.25

PYTHON_SUFFIXES = (".py", ".pyi")
TEXT_SUFFIXES = (".md", ".rst", ".txt")

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+)$", re.MULTILINE)


def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Slide a fixed window over *text*, keeping absolute offsets."""
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
                    last_character_index=end,
                    text=piece,
                )
            )
        if end >= len(text):
            break
    return chunks


def _path_words(file_path: str) -> str:
    """Path turned into searchable words, minus the constant prefix."""
    tail = file_path.split("/", 3)[-1]
    return re.sub(r"[/_.\-]+", " ", tail).strip()


def _line_offsets(text: str) -> List[int]:
    """Character offset where each line starts; index 0 is line 1."""
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _definition_start(node: ast.stmt, offsets: Sequence[int]) -> int:
    """Character offset of a definition, decorators included."""
    lines = [node.lineno]
    lines.extend(
        decorator.lineno for decorator in getattr(node, "decorator_list", [])
    )
    return offsets[min(lines) - 1]


def _segments(
    bounds: Sequence[int],
    max_chunk_size: int,
) -> List[Tuple[int, int]]:
    """Turn sorted cut positions into (begin, end) pairs, merging tiny ones."""
    minimum = max(1, int(max_chunk_size * MIN_SEGMENT_RATIO))
    kept = [bounds[0]]
    for position in bounds[1:-1]:
        if position - kept[-1] >= minimum:
            kept.append(position)
    kept.append(bounds[-1])
    return list(zip(kept, kept[1:]))


def _emit(
    file_path: str,
    text: str,
    begin: int,
    end: int,
    max_chunk_size: int,
    header: str,
) -> List[Chunk]:
    """One segment becomes one chunk, or several if it exceeds the cap."""
    body = text[begin:end]
    if not body.strip():
        return []
    if len(body) <= max_chunk_size:
        return [
            Chunk(
                file_path=file_path,
                first_character_index=begin,
                last_character_index=end,
                text=body,
                indexed_text=f"{header}\n{body}",
            )
        ]
    chunks: List[Chunk] = []
    for piece in chunk_fixed(file_path, body, max_chunk_size):
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=begin + piece.first_character_index,
                last_character_index=begin + piece.last_character_index,
                text=piece.text,
                indexed_text=f"{header}\n{piece.text}",
            )
        )
    return chunks


def chunk_python(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Cut a Python file at definition boundaries, methods included.

    Falls back to :func:`chunk_fixed` when the file does not parse.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return chunk_fixed(file_path, text, max_chunk_size)

    offsets = _line_offsets(text)
    cuts = {0}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            cuts.add(_definition_start(node, offsets))
            for member in node.body:  # ← the boundary the truth uses
                if isinstance(
                    member,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    cuts.add(_definition_start(member, offsets))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            cuts.add(_definition_start(node, offsets))

    header = _path_words(file_path)
    chunks: List[Chunk] = []
    for begin, end in _segments(sorted(cuts) + [len(text)], max_chunk_size):
        chunks.extend(
            _emit(file_path, text, begin, end, max_chunk_size, header)
        )
    return chunks


def _heading_trail(matches: Sequence["re.Match[str]"], position: int) -> str:
    """The chain of headings in force at *position*, outermost first."""
    stack: List[Tuple[int, str]] = []
    for match in matches:
        if match.start() > position:
            break
        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
    return " ".join(title for _, title in stack)


def chunk_markdown(
    file_path: str,
    text: str,
    max_chunk_size: int,
) -> List[Chunk]:
    """Cut a Markdown or text file at heading boundaries."""
    matches = list(HEADING_RE.finditer(text))
    bounds = sorted({0} | {match.start() for match in matches}) + [len(text)]
    path_words = _path_words(file_path)
    chunks: List[Chunk] = []
    for begin, end in _segments(bounds, max_chunk_size):
        header = f"{path_words} {_heading_trail(matches, begin)}".strip()
        chunks.extend(
            _emit(file_path, text, begin, end, max_chunk_size, header)
        )
    return chunks


def chunk_file(file_path: str, text: str, max_chunk_size: int) -> List[Chunk]:
    """Dispatch to the right chunking strategy for *file_path*."""
    lowered = file_path.lower()
    if lowered.endswith(PYTHON_SUFFIXES):
        return chunk_python(file_path, text, max_chunk_size)
    if lowered.endswith(TEXT_SUFFIXES):
        return chunk_markdown(file_path, text, max_chunk_size)
    return _emit(
        file_path, text, 0, len(text), max_chunk_size, _path_words(file_path)
    )
```

The shape to notice: every strategy produces a list of **cut positions**, hands it to `_segments`, and lets `_emit` do the rest. That is why the offset invariant survives — only `_emit` and `chunk_fixed` ever construct a `Chunk`, and `_emit` shifts the windowed offsets by `begin` in the one place it can be got wrong.

`indexed_text` carries the header; `text` and the stored span never do. Prepending the header to `text` instead would corrupt every offset in the index, which is exactly the bug the two-field split in Phase 1 exists to prevent.

The `TEXT_SUFFIXES` tuple includes `.txt` so `CMakeLists.txt` goes through the heading chunker. It has no `#` headings, so it becomes one segment windowed by `chunk_fixed` — the right outcome, reached without a special case.

**Check:**

```bash
uv run python -c "
from pathlib import Path
from src.chunking import chunk_markdown, chunk_python
from src.corpus import read_corpus_file
root = Path('.').resolve()

rel, text = read_corpus_file(root / 'data/raw/vllm-0.10.1/docs/features/lora.md', root)
cs = chunk_markdown(rel, text, 1200)
print('markdown', len(cs), 'chunks; heads:', [text[c.first_character_index:c.first_character_index+28].split(chr(10))[0] for c in cs[:4]])
print('offsets ok:', all(text[c.first_character_index:c.last_character_index] == c.text for c in cs))

rel, text = read_corpus_file(root / 'data/raw/vllm-0.10.1/vllm/lora/request.py', root)
cs = chunk_python(rel, text, 1200)
print('python', len(cs), 'chunks; offsets ok:', all(text[c.first_character_index:c.last_character_index] == c.text for c in cs))
print('indexed_text carries the header:', cs[0].search_text.split(chr(10))[0])
"
```
Verified against the real corpus, this prints something like:

```
markdown 18 chunks; heads: ['# LoRA Adapters', 'er is the path to the LoRA', '## Serving LoRA Adapters', ' ```bash']
offsets ok: True
python 6 chunks; offsets ok: True
indexed_text carries the header: vllm lora request py
```

Both offset checks must print `True`, and the header must read like `vllm lora request py`. Note that only *some* starts are heading lines: a section longer than the cap is windowed, and its continuation chunks start mid-text. That is correct — what matters is that every section *begins* on a heading, and that the offsets round-trip.

And the fallback must not raise:

```bash
uv run python -c "
from src.chunking import chunk_python
print(len(chunk_python('a.py', 'def broken(:\n  pass\n' * 50, 200)))
"
```
prints a positive number, not a `SyntaxError`.

### 4. Tokenize once, build both indices

**Why:** The analyzer is the slow part of indexing. Running a Python callable twice over 27 MB of chunk text is how a 2-minute index becomes a 6-minute one and blows the budget for no gain.

In `src/indexer.py`, add the imports and replace `save`. `build` is unchanged.

```python
# src/indexer.py
# ... existing imports ...
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.analyzer import analyze, identity
from src.bm25 import bm25_weights

CHUNKS_FILE = "chunks.jsonl"
TFIDF_FILE = "tfidf.joblib"
BM25_FILE = "bm25.joblib"
META_FILE = "meta.json"


class Indexer:
    # ... __init__ and build() unchanged ...

    def save(self, processed_dir: Path) -> None:
        """Write chunk metadata and both persisted indices."""
        if not self.chunks:
            raise ValueError("nothing to save: build() produced no chunks")

        processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = processed_dir / CHUNKS_FILE
        with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk in self.chunks:
                handle.write(chunk.to_source().model_dump_json() + "\n")

        print(f"Tokenizing {len(self.chunks)} chunks...")
        tokenized = [analyze(chunk.search_text) for chunk in self.chunks]

        print("Vectorizing (tfidf + bm25)...")
        tfidf = TfidfVectorizer(analyzer=identity, sublinear_tf=True)
        joblib.dump(
            {"vectorizer": tfidf, "matrix": tfidf.fit_transform(tokenized)},
            processed_dir / TFIDF_FILE,
        )

        counter = CountVectorizer(analyzer=identity)
        counts = counter.fit_transform(tokenized)
        joblib.dump(
            {"vectorizer": counter, "matrix": bm25_weights(counts)},
            processed_dir / BM25_FILE,
        )

        meta = {
            "max_chunk_size": self.max_chunk_size,
            "n_chunks": len(self.chunks),
            "n_features": int(counts.shape[1]),
        }
        (processed_dir / META_FILE).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
```

Both payloads use the same two keys, `vectorizer` and `matrix`, so the retriever loads either one with the same three lines. For BM25 the "matrix" holds precomputed weights rather than counts — the naming is deliberate: from the retriever's point of view it is the thing you score against, and nothing downstream needs to know which.

**Check:**

```bash
uv run python -m src index --max_chunk_size 1200
ls -la data/processed/
cat data/processed/meta.json
```
Both `.joblib` files exist, `n_chunks` is around 21 000 at size 1200, and the whole run finishes in well under five minutes. Time it: `time uv run python -m src index --max_chunk_size 1200`.

### 5. Add the BM25 path to the retriever

**Why:** Same signature, two rankers, one flag. This is what makes the A/B a command-line argument instead of a branch you have to remember to switch back.

In `src/retriever.py`, replace `load` and the scoring block of `search`. The result construction at the end is unchanged.

```python
# src/retriever.py
# ... existing imports ...
from src.analyzer import analyze
from src.bm25 import bm25_scores
from src.indexer import BM25_FILE, CHUNKS_FILE, TFIDF_FILE

RETRIEVERS = ("bm25", "tfidf")


class Retriever:
    def __init__(
        self,
        sources: List[MinimalSource],
        vectorizer: Any,
        matrix: Any,
        kind: str = "bm25",
    ) -> None:
        self.sources = sources
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.kind = kind

    @classmethod
    def load(cls, processed_dir: Path, retriever: str = "bm25") -> "Retriever":
        """Read the artefacts written by :meth:`Indexer.save`.

        Raises:
            ValueError: If *retriever* is not one of ``RETRIEVERS``.
            FileNotFoundError: If that index has not been built yet.
        """
        if retriever not in RETRIEVERS:
            raise ValueError(
                f"unknown retriever {retriever!r}, expected one of {RETRIEVERS}"
            )
        chunks_path = processed_dir / CHUNKS_FILE
        index_path = processed_dir / (
            BM25_FILE if retriever == "bm25" else TFIDF_FILE
        )
        if not chunks_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                f"no {retriever} index in {processed_dir} - "
                "run: python -m src index"
            )
        sources: List[MinimalSource] = []
        with chunks_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    sources.append(MinimalSource.model_validate_json(line))
        payload: Dict[str, Any] = joblib.load(index_path)
        return cls(sources, payload["vectorizer"], payload["matrix"], retriever)

    def _score(self, query: str) -> np.ndarray:
        """Score every chunk against *query*, using the loaded ranker."""
        tokens = analyze(query)
        if self.kind == "bm25":
            vocabulary = self.vectorizer.vocabulary_
            term_ids = [
                vocabulary[token] for token in tokens if token in vocabulary
            ]
            if not term_ids:
                return np.zeros(len(self.sources), dtype=np.float32)
            return bm25_scores(self.matrix, term_ids)
        vector = self.vectorizer.transform([tokens])  # ← analyzer=identity
        if vector.nnz == 0:
            return np.zeros(len(self.sources), dtype=np.float32)
        return np.asarray((self.matrix @ vector.T).todense()).ravel()

    def search(self, query: str, k: int = 10) -> List[ScoredSource]:
        """Top-*k* sources for *query*, best first."""
        if k <= 0 or not query.strip():
            return []
        scores = self._score(query)
        if not scores.any():
            return []
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

`self.vectorizer.transform([tokens])` passes a *list of tokens* as one document, which is correct precisely because the vectorizer was fitted with `analyzer=identity`. Handing it a raw string here would make scikit-learn iterate the string character by character and every query would silently score zero.

Then thread the flag through the CLI. In `src/__main__.py`, add `retriever: str = "bm25"` to both `search` and `search_dataset` and pass it to `Retriever.load`:

```python
# src/__main__.py
# ... inside class Cli ...

    def search(
        self,
        query: str,
        k: int = 10,
        retriever: str = "bm25",
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Print the top-*k* sources for a single *query*."""
        engine = Retriever.load(Path(processed_dir), str(retriever))
        sources = engine.search(str(query), int(k))
        # ... printing unchanged ...

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = DEFAULT_SEARCH_OUT,
        retriever: str = "bm25",
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Search every question in a dataset and write the results file."""
        questions = load_questions(Path(dataset_path))
        engine = Retriever.load(Path(processed_dir), str(retriever))
        # ... rest unchanged, using `engine` instead of `retriever` ...
```

Rename the local variable to `engine`: leaving it as `retriever` would shadow the new parameter and hand a `Retriever` object where a string is expected on the next call.

**Check:**

```bash
uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5 --retriever bm25
uv run python -m src search "how do I serve a model with LoRA adapters?" --k 5 --retriever tfidf
uv run python -m src search "activation formats fused batched MoE" --k 3
uv run python -m src search "x" --retriever nope
```
The first two print five lines each with different rankings; the third should surface `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` in the top three (that is a real ground-truth answer); the fourth raises `ValueError: unknown retriever 'nope'`.

### 6. Extend the offset test to all three strategies

**Why:** You just rewrote the chunker. The invariant test is what tells you the rewrite did not shift every index — and it is worth exactly nothing if it still only covers `chunk_fixed`.

Replace `tests/test_chunking.py` with a parametrised version.

```python
# tests/test_chunking.py
"""The offset invariant, over every chunking strategy."""

from typing import Callable, List

import pytest

from src.chunking import chunk_file, chunk_fixed, chunk_markdown, chunk_python
from src.models import Chunk

SECTION = "Text about adapters and serving models. " * 10

MARKDOWN = (
    "# vLLM\n\n" + SECTION + "\n\n"
    "## Using LoRA\n\n" + SECTION + "\n\n"
    "### API Endpoints\n\nPOST /v1/load_lora_adapter\n"
)

PYTHON = (
    "import os\n\nCONST = 1\n\n\n"
    "class LoRARequest:\n"
    '    """A request."""\n\n'
    "    @property\n"
    "    def adapter_id(self) -> int:\n"
    "        return self._id\n\n"
    "    def resolve(self) -> str:\n"
    "        return os.path.join('a', 'b')\n\n\n"
    "def get_model_config(name: str) -> dict:\n"
    "    return {'name': name}\n"
)

Strategy = Callable[[str, str, int], List[Chunk]]
CASES = [
    (chunk_fixed, MARKDOWN),
    (chunk_markdown, MARKDOWN),
    (chunk_python, PYTHON),
    (chunk_file, MARKDOWN),
]


@pytest.mark.parametrize("strategy,text", CASES)
def test_span_reproduces_text(strategy: Strategy, text: str) -> None:
    for chunk in strategy("data/raw/x.md", text, 300):
        sliced = text[
            chunk.first_character_index : chunk.last_character_index
        ]
        assert sliced == chunk.text


@pytest.mark.parametrize("strategy,text", CASES)
def test_no_chunk_exceeds_the_cap(strategy: Strategy, text: str) -> None:
    for chunk in strategy("data/raw/x.md", text, 300):
        assert chunk.last_character_index - chunk.first_character_index <= 300


@pytest.mark.parametrize("strategy,text", CASES)
def test_chunks_are_ordered_and_non_empty(strategy: Strategy, text: str) -> None:
    chunks = strategy("data/raw/x.md", text, 300)
    assert chunks
    starts = [chunk.first_character_index for chunk in chunks]
    assert starts == sorted(starts)


def test_markdown_cuts_on_headings() -> None:
    # A cap of 500 keeps each ~400-character section whole, so every cut
    # that survives the merge pass must land on a heading.
    chunks = chunk_markdown("data/raw/x.md", MARKDOWN, 500)
    assert len(chunks) == 3
    for chunk in chunks:
        start = chunk.first_character_index
        assert start == 0 or MARKDOWN[start] == "#"


def test_python_cuts_on_definitions() -> None:
    chunks = chunk_python("data/raw/x.py", PYTHON, 200)
    heads = [chunk.text.lstrip().split("\n")[0] for chunk in chunks]
    assert any(head.startswith("@property") for head in heads)
    assert any(head.startswith("def get_model_config") for head in heads)


def test_unparseable_python_falls_back() -> None:
    broken = "def broken(:\n    pass\n" * 30
    chunks = chunk_python("data/raw/x.py", broken, 200)
    assert chunks
    for chunk in chunks:
        assert (
            broken[chunk.first_character_index : chunk.last_character_index]
            == chunk.text
        )


def test_indexed_text_does_not_move_offsets() -> None:
    for chunk in chunk_markdown("data/raw/docs/lora.md", MARKDOWN, 300):
        assert chunk.search_text.endswith(chunk.text)
        assert len(chunk.search_text) > len(chunk.text)


def test_empty_file_produces_nothing() -> None:
    assert chunk_file("data/raw/x.md", "", 200) == []


def test_rejects_non_positive_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_fixed("data/raw/x.md", MARKDOWN, 0)
```

`test_indexed_text_does_not_move_offsets` is the important one. It asserts the enriched text *ends with* the raw chunk text and is strictly longer — that is, the header was prepended and the stored span still describes only the real file content. This is the test that catches the single most damaging mistake in this phase.

**Check:** `uv run pytest -q` prints `28 passed`. Then, in `_emit`, change `first_character_index=begin + piece.first_character_index` to drop the `begin +` and re-run: `test_span_reproduces_text[chunk_python-...]` must fail. Undo it.

### 7. Sweep chunk size and lock the configuration

**Why:** The subject asks you to report the effect of chunk size on recall@k. The sweep answers that in writing, and it usually finds free points.

Run the full cycle at three sizes and both retrievers. This loop takes 20–40 minutes wall clock; start it and go read the Phase 4 file.

```bash
for size in 2000 1200 800; do
  uv run python -m src index --max_chunk_size $size
  for engine in bm25 tfidf; do
    for scope in docs code; do
      uv run python -m src search_dataset \
        --dataset_path data/datasets/UnansweredQuestions/dataset_${scope}_public.json \
        --k 10 --retriever $engine \
        --save_directory data/output/search_results/UnansweredQuestions
      echo "== size=$size engine=$engine scope=$scope"
      uv run python -m src evaluate \
        --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json \
        --dataset_path data/datasets/AnsweredQuestions/dataset_${scope}_public.json
    done
  done
done | tee sweep.log
```

Add one row to `benchmarks.md` per `(size, engine)` pair — twelve rows, six lines of the table. Then pick the winner subject to three constraints, in this order: both bars cleared with margin, indexing under 5 minutes, 200 questions under 90 seconds. Set that chunk size as the `index` default in `src/__main__.py` and that retriever as the `Retriever.load` default.

`sweep.log` is a scratch file — add it to `.gitignore` or delete it; it is not a deliverable.

**Check:** `benchmarks.md` has twelve new rows, the chosen configuration is marked, and re-running the winning configuration reproduces its numbers exactly. If it does not reproduce, something in your pipeline is order-dependent — `list_corpus_files` sorts for exactly this reason, so look at your own additions first.

### 8. Diagnose whatever is still failing

**Why:** If a bar is still unmet, guessing is expensive. The failures are almost never twenty different problems; they are one problem twenty times.

Dump the worst questions with their ground truth and your top-10 side by side, and read them:

```bash
uv run python -c "
import json
from pathlib import Path
from src.datasets import load_search_results
from src.evaluation import IOU_THRESHOLD, span_iou, truth_by_id

results = load_search_results(Path('data/output/search_results/UnansweredQuestions/dataset_code_public.json'))
truth = truth_by_id(Path('data/datasets/AnsweredQuestions/dataset_code_public.json'))
misses = 0
for row in results.search_results:
    expected = truth.get(row.question_id, [])
    hit = any(span_iou(e, g) >= IOU_THRESHOLD
              for e in expected for g in row.retrieved_sources[:5])
    if hit or misses >= 8:
        continue
    misses += 1
    print('Q:', row.question)
    for e in expected:
        print('   truth:', e.file_path, e.first_character_index, e.last_character_index)
    for g in row.retrieved_sources[:3]:
        print('    mine:', g.file_path, g.first_character_index, g.last_character_index)
    print()
"
```

The shapes you will see, and the fix for each:

| What you see | Fix |
|---|---|
| The truth file is one you never indexed | Widen `TEXT_SUFFIXES` in `corpus.py` |
| Right file, wrong region, every time | Chunk size or the merge threshold — the truth is landing between your cuts |
| Right region ranked 8th–15th | Ranking, not retrieval: try boosting the header, or `k1`/`b` |
| The question shares no rare word with the answer | Analyzer, or genuinely out of reach for a lexical method — this is what the bonus embeddings solve |
| Truth in a test file, yours in the implementation (or the reverse) | Nothing to fix; note it and move on |

**Check:** you can name the dominant failure shape in one sentence. Write that sentence down — it is the "Challenges faced" section of your README in Phase 6.

## Common pitfalls

| Pitfall | Why it happens | Avoid it by |
|---|---|---|
| AST `lineno` used directly as a character offset | It looks like a position and the code runs | Convert through `_line_offsets`; the parametrised offset test catches it instantly |
| The header prepended to `text` instead of `indexed_text` | It is one field over and both "work" | `test_indexed_text_does_not_move_offsets` |
| `ast.parse` aborting the whole index | One file in 1763 that will not parse under your interpreter | The `try/except (SyntaxError, ValueError, RecursionError)` fallback |
| Only cutting at module level in Python files | It is the obvious reading of "top-level nodes" | Walk `ClassDef.body` too — the ground truth sits on methods |
| `analyzer=lambda x: x` on the vectorizer | The obvious way to write a pass-through | Module-level `identity`; a lambda makes `joblib.dump` fail at the end of a 3-minute index |
| Passing a raw string to a vectorizer fitted with `analyzer=identity` | The TF-IDF path looks like ordinary scikit-learn | `transform([analyze(query)])` — a string would be iterated per character and score zero |
| The local variable `retriever` shadowing the new `retriever` parameter | The rename is easy to skip | Call the object `engine` |
| Stopwords eating code tokens | Reaching for a stock English list | The curated list in `analyzer.py`; check that `kv` and `id` survive |
| Tuning three things between two measurements | It feels faster | One change, one benchmark row, one commit |
| Recall@5 climbs while recall@10 collapses | An aggressive boost overfitting the top of the ranking | Always record all four k values; a collapsing tail means the gain is noise |
| Chunks so small the answer is fragmented | Chasing recall@5 down to 300 characters | Weigh Phase 4: these same chunks become the model's context |

## Verify it's done

```bash
uv run python -m src index --max_chunk_size 1200
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

Expected — the recalls are yours, the two bars are not:

```
Chunking: 100%|███████████████████| 2121/2121 [00:24<00:00, 87.1file/s]
Tokenizing 21259 chunks...
Vectorizing (tfidf + bm25)...
Ingestion complete! 21259 chunks. Indices saved under .../data/processed
Searched 100 questions in 8.4s
Questions scored: 100
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
Searched 99 questions in 8.1s
Questions scored: 99
Recall@1: 0.310  Recall@3: 0.480  Recall@5: 0.580  Recall@10: 0.680
28 passed
```

Then the grader, which is the only opinion that counts:

```bash
for scope in docs code; do
  moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_${scope}_public.json \
    data/datasets/AnsweredQuestions/dataset_${scope}_public.json \
    --k 10 --max_context_length 2000
done
```
```
Student data is valid: True
Recall@1: 0.620  Recall@3: 0.790  Recall@5: 0.860  Recall@10: 0.910
Student data is valid: True
Recall@1: 0.310  Recall@3: 0.480  Recall@5: 0.580  Recall@10: 0.680
```

`Recall@5` must be at least `0.800` on docs and `0.500` on code, **as reported by the moulinette**, not only by your `evaluate`.

## Definition of done

- [ ] Two distinct chunking strategies plus a fallback, dispatched by file type
- [ ] Docs recall@5 ≥ 0.80 per the moulinette, with margin
- [ ] Code recall@5 ≥ 0.50 per the moulinette, with margin
- [ ] Indexing under 5 minutes; 200 questions searched in under 90 seconds
- [ ] Both retrievers runnable via `--retriever`; the default is the measured winner
- [ ] `benchmarks.md` has a row per experiment and the winner marked
- [ ] The offset invariant test passes for all three strategies and fails when sabotaged
- [ ] You can state the dominant remaining failure shape in one sentence
- [ ] Committed

## Deliberately NOT in this phase

- Semantic embeddings and hybrid fusion → bonus, not in v1 (and not a substitute for lexical recall)
- Answer generation → **Phase 4**
- Query expansion beyond the analyzer → only if a bar is still unmet after step 8
- flake8 / mypy cleanliness → **Phase 5** (run `make lint` now anyway; do not let the count grow)
- Caching the index for faster cold start → bonus #4

## Commit

```bash
git add -A
git commit -m "phase 3: language-aware chunking and identifier-aware BM25 retrieval"
```

## Next

→ **[Phase 4 — Answer in English with Qwen3-0.6B](PHASE-04-answer-with-qwen.md)**
