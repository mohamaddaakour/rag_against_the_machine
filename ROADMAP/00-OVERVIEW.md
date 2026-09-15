# 00 — Overview

**Project:** RAG against the machine — *Will you answer my questions?* (subject v2.0)
**Roadmap written:** 2026-09-14
**Status:** Phase 1 is current and fully detailed. Phases 2–15 are planned and
listed below; each is expanded into its own `PHASE-XX-*.md` only when you say
you are ready for it.

---

## 1. Objective

Build a Retrieval-Augmented Generation system over the vLLM 0.10.1 source tree
that:

1. ingests `data/raw/` into a persisted, searchable index in under 5 minutes,
2. returns, for any question, the top-k source locations (`file_path` +
   character span, each ≤ 2000 characters wide) that contain the answer,
3. generates a natural-language answer from those spans using
   `Qwen/Qwen3-0.6B` running locally on CPU,
4. reaches **≥ 80 % recall@5 on the docs dataset** and **≥ 50 % recall@5 on the
   code dataset**, as measured by the reference moulinette,
5. does all of the above behind a Python Fire CLI (`uv run python -m src
   <command>`) that never crashes with an unhandled traceback.

The project is graded primarily on retrieval quality, grounding, and prompt
strategy — not on the eloquence of the 0.6B model's prose.

---

## 2. Repository assessment

### 2.1 What is in the working directory today

`C:\Users\user\Desktop\test` contains exactly one file: `subject.md`. There is
no source code, no `pyproject.toml`, no data. **This is a new project.**

### 2.2 The git situation (important, and deliberately left alone)

`C:\Users\user\Desktop\test` is *not* its own git repository. Running
`git rev-parse --show-toplevel` from it returns `C:/Users/user` — your entire
Windows home directory is a git repository, with remote
`https://github.com/mohamaddaakour/Modal-Window.git`, currently on branch
`final-solution`. Its working tree shows hundreds of deletions across unrelated
folders.

You chose to leave version control alone for now, so **this roadmap contains no
git commands at all.** Each phase names a natural commit point; acting on it is
your call. One thing to keep in mind for submission day: the subject requires
`src/`, `pyproject.toml`, `uv.lock`, `Makefile` and `README.md` at the *root of
the repository you submit*. The home-directory repo cannot satisfy that, so at
some point before the defense the project folder will need to become its own
repository. That is noted here as a known open item, not scheduled as a phase.

### 2.3 Prior art found on this machine (inspected, not reused wholesale)

`C:\Users\user\Desktop\my_projects\python\` holds several earlier attempts. The
most advanced is `rag_against_the_machine`:

| Component | State in the donor project |
|---|---|
| `src/models.py` | All required pydantic models plus `ScoredSource` and `Chunk`. Sound. |
| `src/corpus.py` | Walks `data/raw`, filters by extension, returns grader-relative POSIX paths. Sound. |
| `src/chunking.py` | **Fixed-size sliding window only.** The subject demands *two distinct* strategies (Python and Markdown). This is an unmet mandatory requirement there. |
| `src/indexer.py` | TF-IDF via scikit-learn, persisted with joblib. Works; index is built (2 880 corpus files). |
| `src/retriever.py` | Sparse cosine similarity, top-k via `argpartition`. Works. |
| `src/__main__.py` | Only `index` and `search`. **Four of the six mandatory commands are missing.** |
| Generation | **Entirely absent.** No `Qwen`, no `transformers`, no `answer`. |
| `tests/` | Three real pytest files (chunking, indexer, models). |
| Tooling | `Makefile`, `pyproject.toml`, `setup.cfg`, `uv.lock` — all usable as reference. |
| `README.md` | Written, but describes a Phase-1-only system and still has an unfilled login placeholder. |

**How this roadmap treats it:** as a *reference and as a data source*, never as
a base to build on. You chose a fresh build in a new dedicated folder, so every
line of code in this roadmap is written for you to type yourself. The donor
project's `data/raw/vllm-0.10.1/`, `data/datasets/`, and `moulinette` are
copied across in later phases so you do not re-download several hundred
megabytes — that is file copying, not code reuse.

### 2.4 Toolchain verified on this machine

| Tool | Result |
|---|---|
| Python | 3.12.4 — satisfies `>=3.10` |
| uv | 0.11.31 |
| git | 2.47.1.windows.1 |
| **make** | **not installed** — see risk R4 |
| Free disk | 487 GB on C: — ample for torch + model weights |
| WSL | present, but the only distro is `docker-desktop` — see risk R1 |

### 2.5 Facts established by reading the actual datasets

These drove several design decisions, so they are recorded here rather than
rediscovered later:

- `data/datasets/AnsweredQuestions/dataset_docs_public.json` — 100 questions.
- `data/datasets/AnsweredQuestions/dataset_code_public.json` — 99 questions.
- **Every question in both datasets has exactly one ground-truth source.**
  Recall@k therefore reduces to a hit rate: for each question you either cover
  its single span within your top-k or you do not.
- Docs ground truth points only at `.md` (97) and `.txt` (3) files. Code ground
  truth points only at `.py` files (99).
- Ground-truth span widths range from 12 to 1578 characters, median ≈ 878.
- The `Unanswered` and `Answered` files share question ids, so the unanswered
  file is the input and the answered file is the key for the same questions.
- The moulinette binary is `ELF 64-bit LSB executable, x86-64, GNU/Linux`.

---

## 3. Functional requirements (from the subject — non-negotiable)

| # | Requirement |
|---|---|
| F1 | Runnable as `uv run python -m src <command>`; CLI built with **Python Fire**. |
| F2 | Commands: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate`. |
| F3 | `index --max_chunk_size <int>` (default 2000) ingests `data/raw/` → `data/processed/`. |
| F4 | **Two distinct chunking strategies**: one for Python code, one for Markdown/text. |
| F5 | At least one of **TF-IDF** or **BM25** implemented as the retrieval method. |
| F6 | No retrieved source may exceed 2000 characters. One over-long source invalidates the whole output file. |
| F7 | `file_path` must match the corpus path **verbatim**, e.g. `data/raw/vllm-0.10.1/docs/features/lora.md`. |
| F8 | Data exchanged between stages validated with **pydantic** models exactly as specified in §VI.4. |
| F9 | Output files conform to `StudentSearchResults` / `StudentSearchResultsAndAnswer`. |
| F10 | Answers generated locally by `Qwen/Qwen3-0.6B`, grounded in the retrieved spans. |
| F11 | Exact directory layout: `src/`, `data/raw/`, `data/processed/`, `data/datasets/{Unanswered,Answered}Questions/`, `data/output/search_results/<Scope>/`, `data/output/search_results_and_answer/<Scope>/`. |
| F12 | All input/output paths are CLI arguments — **never hard-coded**. |
| F13 | `tqdm` progress bars on long-running operations. |
| F14 | Degenerate input (empty query, nonsense query, `k=0`, missing file, malformed JSON) handled gracefully — **never an unhandled traceback**. |
| F15 | `Makefile` with `install`, `run`, `debug`, `clean`, `lint`. |
| F16 | `README.md` with the subject's mandated sections, in English, first line italicised. |
| F17 | The solution must never import or call the moulinette. |

## 4. Non-functional requirements

| # | Requirement | Source |
|---|---|---|
| N1 | Indexing the whole corpus ≤ 5 minutes. | §VII.1.2 |
| N2 | Retrieval ≤ 90 seconds for 200 questions. | §VII.1.2 |
| N3 | Recall@5 ≥ 80 % docs, ≥ 50 % code. | §VII.1.2 |
| N4 | `flake8 .` clean. | §V.1 |
| N5 | `mypy .` clean with the five mandated flags. | §V.1 |
| N6 | Type hints on all functions; PEP 257 docstrings. | §V.1 |
| N7 | Context managers for every file handle. | §V.1 |
| N8 | `uv` is the package manager; `uv sync` must work from the repo root. | §V.4 |
| N9 | Runs CPU-only on a campus machine. | §IX |

---

## 5. Requirements vs. assumptions vs. my implementation decisions

### 5.1 Hard requirements
Everything in §3 and §4 above. None of it is negotiable and none of it is my
choice.

### 5.2 Assumptions I am proceeding under (correct me and I will revise)

| # | Assumption | Consequence if wrong |
|---|---|---|
| A1 | The project root is `C:\Users\user\Desktop\my_projects\python\rag_final`. You said "new dedicated folder" without naming it; I picked a short, unambiguous name. | Cosmetic — change the path in Phase 1 step 1 and every later command follows. |
| A2 | The vLLM corpus, both datasets, and the moulinette may be copied from `rag_against_the_machine` rather than re-downloaded. Verified present and complete. | You would need the original attachment archive. |
| A3 | The public datasets you have are representative of the private ones used at the defense. | Tuning to the public sets could overfit; §6 risk R3 covers this. |
| A4 | You will run the defense pipeline on a Linux campus machine, where the moulinette runs natively. | Locally you rely on our `evaluate`; see R1. |
| A5 | PowerShell is your primary shell. Commands are given in PowerShell form. | Bash equivalents differ only in `mkdir`/path syntax. |
| A6 | Solo project — README credits one login. | Add logins to the first line. |

### 5.3 My implementation decisions (not required by the subject)

These are choices I am making so later phases stay coherent. Each is revisited
only with a stated reason.

| # | Decision | Why |
|---|---|---|
| D1 | **scikit-learn `TfidfVectorizer`** for the lexical index, rather than hand-rolled TF-IDF. | F5 requires the *method*, not a from-scratch implementation. Sparse matrix multiply gives us N2 (90 s / 200 questions) essentially for free. |
| D2 | Chunks store only `file_path` + span on disk; **chunk text is re-read from the corpus by offset** when generation needs it. | Keeps `data/processed/` small and makes the offset invariant the single source of truth. |
| D3 | **Build the plain fixed-window chunker first (Phase 4), then replace it with the two structure-aware strategies (Phase 9).** | Gives a measurable recall baseline in Phase 8, so Phase 9's structural chunking can be justified by a number instead of by faith. F4 is still satisfied, just at Phase 9 rather than Phase 4. |
| D4 | **BM25 is a conditional phase (11).** | If TF-IDF plus structural chunking plus identifier-aware tokenisation already clears 80 %/50 %, BM25 is complexity with no pressure behind it. F5 is satisfied by TF-IDF alone. |
| D5 | Our own `evaluate` implements the subject's IoU ≥ 0.05 same-file overlap rule. | R1 — the moulinette will not run on your machine. This is the metric you iterate against. |
| D6 | Generation runs at temperature 0 / greedy. | Reproducible answers; the subject grades grounding, not creativity. |
| D7 | Corpus file filter is extension-based and deliberately narrow (`.py`, `.md`, `.txt`, `.rst`). | §2.5 shows ground truth lives only in `.py`, `.md` and `.txt`. Every other extension is index noise that costs N1 and dilutes IDF. |
| D8 | A `status` command beyond the six mandatory ones. | §VI.6 explicitly permits extra options/commands, and F11's layout is easier to verify than to remember. |

---

## 6. Major risks and trade-offs

**R1 — The moulinette cannot run on this machine.** It is a Linux ELF binary;
Windows cannot execute it, and your only WSL distro is `docker-desktop`, which
is not a general-purpose environment. *Mitigation:* Phase 8 builds our own
`evaluate` against the subject's stated rule (same file, IoU ≥ 0.05), and it is
the number every later phase iterates against. *Residual risk:* our
interpretation of the rule could differ from the moulinette's. Before the
defense, run the official binary once on a Linux box (campus machine, a real
WSL Ubuntu distro, or a container) and confirm the two numbers agree. Until
that is done, treat our recall as a strong indicator, not as the verdict.

**R2 — The 50 % code-recall threshold is the hard one.** Docs questions
paraphrase prose that shares vocabulary with the prose being searched; code
questions ("What activation formats does the fused batched MoE layer return?")
must match an identifier buried in a `.py` file. Plain TF-IDF over raw source
text tends to land near the threshold, not comfortably above it. This is why
Phases 9 and 10 exist and why Phase 11 is held in reserve.

**R3 — Overfitting to the public datasets.** The defense uses private ones.
*Mitigation:* prefer changes that are principled (chunk on real structural
boundaries; make identifiers tokenisable) over changes that are empirical
(tuning a magic constant until the public number moves). Phase 10 states this
explicitly.

**R4 — `make` is not installed.** The `Makefile` is mandatory for submission
(F15) and will be written in Phase 1, but you cannot execute it locally.
*Mitigation:* every phase gives the raw `uv run ...` command as the primary
instruction, with the `make` target named alongside. Installing GNU Make on
Windows is optional and is not scheduled as work.

**R5 — The deep-learning stack is large.** `torch` + `transformers` + the
Qwen3-0.6B weights total roughly 2–3 GB. *Mitigation:* they are not installed
until Phase 12, so Phases 1–11 stay fast to set up and fast to lint. Disk is
not a constraint (487 GB free).

**R6 — The 2000-character ceiling is a validity cliff, not a quality knob.** A
single returned source wider than 2000 characters invalidates the entire output
file (F6). *Mitigation:* the width invariant is asserted in a unit test from
Phase 4 onward and re-checked in Phase 14, so it can never regress silently.

**R7 — Windows path separators.** `Path` yields backslashes on Windows;
ground-truth comparison is verbatim against forward-slash paths (F7). A single
`\` anywhere in a `file_path` scores zero on every question. *Mitigation:* one
function owns path normalisation (Phase 3), and a test pins it.

---

## 7. Final technology stack

| Layer | Choice | Version constraint | Introduced in |
|---|---|---|---|
| Language | Python | `>=3.10` (3.12.4 local) | Phase 1 |
| Project/package manager | uv | 0.11.31 | Phase 1 |
| CLI | `fire` | `>=0.6` | Phase 1 |
| Validation | `pydantic` | `>=2.7` | Phase 2 |
| Progress bars | `tqdm` | `>=4.66` | Phase 4 |
| Vectorisation | `scikit-learn` | `>=1.4` | Phase 5 |
| Numerics | `numpy`, `scipy` | `>=1.26`, `>=1.11` | Phase 5 |
| Index persistence | `joblib` | `>=1.3` | Phase 5 |
| Generation | `transformers` | `>=4.51` (Qwen3 support) | Phase 12 |
| Tensor runtime | `torch` (CPU wheel) | `>=2.2` | Phase 12 |
| Tests | `pytest` | `>=8.0` | Phase 1 |
| Lint | `flake8` | `>=7.0` | Phase 1 |
| Types | `mypy` | `>=1.10` | Phase 1 |

Nothing is added to `pyproject.toml` before the phase that actually uses it.

---

## 8. Architecture preview (the finished system, one screen)

```
                       uv run python -m src <command>
                                   │
                          ┌────────▼────────┐
                          │  src/__main__   │   Fire CLI, error boundary
                          └────────┬────────┘
        ┌──────────────┬───────────┼────────────┬───────────────┐
        │              │           │            │               │
     index          search   search_dataset   answer      answer_dataset
        │              │           │            │               │        evaluate
        ▼              ▼           ▼            ▼               ▼           │
┌───────────────┐  ┌──────────────────────┐  ┌──────────────────────┐       │
│  src/corpus   │  │   src/retriever      │  │   src/generator      │       │
│  walk + read  │  │  load index, rank    │  │  Qwen3-0.6B, greedy  │       │
│  exact paths  │  │  top-k, cosine       │  │  prompt = spans      │       │
└───────┬───────┘  └──────────┬───────────┘  └──────────┬───────────┘       │
        │                     │                         │                   │
        ▼                     │                    re-reads spans           │
┌───────────────┐             │                    from data/raw            │
│ src/chunking  │             │                    by offset (D2)           │
│  .py → AST    │             │                                             │
│  .md → heads  │             │                                             │
│  offsets kept │             │                                             │
└───────┬───────┘             │                                             │
        ▼                     │                                             │
┌───────────────┐             │                                   ┌─────────▼────────┐
│  src/indexer  │             │                                   │   src/evaluate   │
│  TF-IDF fit   │             │                                   │  recall@k, IoU   │
└───────┬───────┘             │                                   └─────────▲────────┘
        ▼                     │                                             │
  data/processed/ ────────────┘                                             │
   chunks.jsonl                                                             │
   tfidf.joblib          ┌──────────────────────────────┐                   │
   meta.json             │  src/models.py  (pydantic)   │                   │
                         │  the contract every stage    │───────────────────┘
                         │  reads and writes            │
                         └──────────────────────────────┘
                                      │
   data/output/search_results/<Scope>/*.json           (StudentSearchResults)
   data/output/search_results_and_answer/<Scope>/*.json (StudentSearchResultsAndAnswer)
```

Single process, modular monolith, no services, no database, no container. The
pressure that would justify any of those is absent: one user, one machine, a
read-only corpus, and a batch workload.

---

## 9. Phase table

Estimates assume focused work and exclude one-off downloads.

| # | Phase | Goal | New concepts (1–2) | Runnable result | Problem it solves | Est. | Status |
|---|---|---|---|---|---|---|---|
| 1 | Runnable skeleton | A lint-clean uv project that answers a real command | uv as project manager; Python Fire turning a class into a CLI | `uv run python -m src status` prints the mandatory layout and which parts exist | Nothing runs yet; F1/F11/F15/N4/N5/N8 need a floor to stand on | 45–60 min | **Current — see `PHASE-01-runnable-skeleton.md`** |
| 2 | The data contract | Parse the real dataset files into validated objects | pydantic v2 `BaseModel`; parsing a union of answered/unanswered questions | `check_dataset --dataset_path <p>` reports counts and rejects malformed JSON | Every later stage exchanges these structures (F8); getting the contract wrong late is expensive rework | 45–60 min | Pending |
| 3 | Corpus ingestion | Walk `data/raw/` and produce grader-exact paths | `pathlib.rglob` traversal with an extension allowlist; relative-POSIX path normalisation (R7/F7) | `corpus_stats` prints file count, total characters, per-extension breakdown | A single backslash or wrong prefix in `file_path` scores zero on every question | 45–60 min | Pending |
| 4 | Chunking with exact offsets | Cut files into ≤2000-char pieces that know where they came from | The offset invariant `text[first:last] == chunk.text`; overlapping windows | `chunk_stats` prints chunk count and max width; a test proves the invariant on real corpus files | Retrieval returns spans, not text — if offsets drift, every result is wrong in a way that is invisible until scoring (F6) | 60–75 min | Pending |
| 5 | Build and persist the index | Turn chunks into a searchable TF-IDF matrix on disk | TF-IDF vectorisation; persisting a fitted vectoriser + sparse matrix with joblib | `index --max_chunk_size 2000` completes in under 5 minutes and writes `data/processed/` | Re-chunking 2 880 files per query is impossible; F3 and N1 demand a persisted index | 60–75 min | Pending |
| 6 | Single-query retrieval | Rank the index against one question | Cosine similarity as a sparse dot product; top-k via `argpartition` | `search "How do I load a LoRA adapter?" --k 5` prints ranked file spans | The index is inert until something queries it; this is the first observable answer to a real question | 45–60 min | **Done (2026-09-15)** |
| 7 | Batch search over a dataset | Produce the file the grader actually reads | Batch vectorisation (one `transform` for all queries, for N2); writing the `StudentSearchResults` contract | `search_dataset` writes valid JSON under `data/output/search_results/<Scope>/` | F2/F9/F11 — the defense pipeline runs this command, not `search` | 45–60 min | **Current — see `PHASE-07-batch-search-over-a-dataset.md`** |
| 8 | Measure your own recall | Know your number without the moulinette | The IoU ≥ 0.05 same-file overlap rule; recall@k | `evaluate` prints recall@1/3/5/10 for docs and code — **the baseline every later phase is judged against** | R1: the official binary will not run here, and tuning without a metric is guessing | 60–75 min | Pending |
| 9 | Chunk by structure | Stop cutting chunks mid-function and mid-section | Python chunking on `ast` `def`/`class` boundaries; Markdown chunking on heading boundaries — **the two strategies F4 requires** | Re-index, re-`evaluate`, observe the recall delta against Phase 8 | A fixed window splits the answer across two chunks so neither ranks; also the last unmet mandatory requirement from the donor project | 75–90 min | Pending |
| 10 | Make identifiers matchable | Close the gap on code questions | A custom analyzer that splits `snake_case`/`camelCase`; enriching indexed text with path tokens | `evaluate` shows code recall@5 moving toward and past 50 % | R2 — `fused_batched_moe` is one opaque token to the default analyzer, so a question saying "fused batched MoE" cannot match it | 60–75 min | Pending |
| 11 | BM25 ranking *(conditional)* | Reach the thresholds if Phases 9–10 did not | Okapi BM25 scoring and how its length normalisation differs from TF-IDF cosine | `--method bm25` flag; `evaluate` compares both rankings | Only run if N3 is still unmet after Phase 10. If the bar is already cleared, this becomes README future-work (D4) | 60–90 min | Pending (conditional) |
| 12 | Answer one question with Qwen | The first grounded natural-language answer | Loading a local causal LM with `transformers`; building a prompt inside a token budget from spans re-read off disk (D2) | `answer "..." --k 5` prints a grounded answer | Phases 1–11 only ever return spans; F10 requires prose | 60–90 min + one-off ~1.5 GB download | Pending |
| 13 | Answer a whole dataset | Batch generation over every question | No new concept — the same single→batch step as 6→7, over a `tqdm` loop | `answer_dataset` writes `StudentSearchResultsAndAnswer` under the scoped output dir | F2/F9/F11 — step 4 of the subject's end-to-end walkthrough | 45–60 min | Pending |
| 14 | Harden every entry point | Survive a reviewer trying to break it | A single CLI-level error boundary; systematic degenerate-input tests | Empty query, `k=0`, `k=-1`, missing file, malformed JSON, unbuilt index — each prints a clear message, exit code sane, no traceback | F14 is explicitly tested at the defense, and "it crashed" is graded as non-functional (§V.1) | 60–75 min | Pending |
| 15 | README, full pipeline, validation | Ship it | No new concept — documentation and an end-to-end rehearsal | The subject's four-command walkthrough runs start to finish; `README.md` carries all mandated sections | F16, and R1's open item: confirm our recall against the real moulinette on Linux | 75–90 min | Pending |

**Bonuses (semantic embeddings, hybrid retrieval, incremental indexing,
caching, local HTTP API) are out of scope** by your decision. They are graded
only once the mandatory part validates in full (§IX), so they would slot in
after Phase 15. Say the word and I will plan them then.

### Phase dependency chain

```
1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 ─┬→ 9 → 10 →(11 if needed)→ 12 → 13 → 14 → 15
                               └── 8 is the measurement gate: 9, 10 and 11 are
                                   each justified by the number it produces
```

Phases 1–8 are a straight line — each genuinely needs its predecessor. Phase 8
is the hinge: it turns the rest of the retrieval work from opinion into
measurement.

---

## 10. What I will and will not do

I inspect, analyse, architect, explain, plan, document, review your code when
you show it to me, and hand you exact code to type. **You implement.** I will
not create, modify or delete any source, configuration or data file in your
project, and I will not run anything that changes it. The only files I write
are these roadmap documents.

## 11. Open items

- **OI-1** — The project folder must become its own git repository before
  submission (§2.2). Not scheduled; raise it when you want it.
- **OI-2** — Validate our `evaluate` against the real moulinette on a Linux
  environment (R1). Scheduled as part of Phase 15, but do it sooner if you get
  access to a campus machine.
- **OI-3** — The README's first line needs your 42 login (A6).
