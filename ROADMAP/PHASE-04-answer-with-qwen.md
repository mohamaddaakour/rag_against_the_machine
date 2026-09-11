# Phase 4 — Answer in English with Qwen3-0.6B

**Goal:** Turn retrieved spans into a grounded natural-language answer, for one query and for a whole dataset.
**Time:** ~5h · **Difficulty:** ●●●○○
**Depends on:** Phase 3 complete. An answer is only as good as the context it is given.

## ✅ What you'll have when this is done

The G in RAG. A 0.6-billion-parameter model running on your CPU reads the exact character ranges your retriever returned and writes an answer from them, and `answer_dataset` turns a `StudentSearchResults` file into a `StudentSearchResultsAndAnswer` file.

```bash
$ uv run python -m src answer "how do I serve a model with LoRA adapters?" --k 5
Loading Qwen/Qwen3-0.6B on cpu... ready in 6.8s
Start the server with the --enable-lora flag and pass your adapter with
--lora-modules, for example `--lora-modules sql-lora=$HOME/sql-lora`. You can
also load an adapter at runtime by POSTing its name and path to the
/v1/load_lora_adapter endpoint.

[1] data/raw/vllm-0.10.1/docs/features/lora.md [4695-5900]
[2] data/raw/vllm-0.10.1/docs/features/lora.md [3400-4695]
[3] data/raw/vllm-0.10.1/vllm/lora/request.py [0-1180]
[4] data/raw/vllm-0.10.1/docs/models/supported_models.md [40800-41900]
[5] data/raw/vllm-0.10.1/examples/offline_inference/multilora_inference.py [0-1200]
```

The wording will not be that clean every time — Qwen3-0.6B is a small model and the subject says so explicitly. The bar is: coherent, grounded in the sources, on topic. Retrieval quality and prompt strategy are what get graded here, not prose.

## Where you're starting from

```
src/
├── __main__.py     # index, search, search_dataset, evaluate
├── analyzer.py · bm25.py · chunking.py · corpus.py
├── datasets.py · evaluation.py · indexer.py · models.py · retriever.py
benchmarks.md       # the winning configuration, marked
```

Retrieval clears both bars. Nothing in the project has ever loaded a model.

## Why this phase now

Retrieval was the risk; generation is mostly plumbing plus one real decision (how you spend the context window). Doing it after Phase 3 means the model sees good snippets on its first run, so when an answer is wrong you know it is the prompt, not the retrieval. Doing it before would have you debugging two things at once.

## Before you start

Add the deep-learning stack. This pulls roughly 2 GB of wheels — start it and read on:

```bash
uv add "torch>=2.2" "transformers>=4.51" "accelerate>=0.30"
```

`transformers>=4.51` is the floor for the Qwen3 architecture. An older release loads the tokenizer fine and then fails on the model config, which reads as a corrupt download rather than a version problem.

Point the model cache outside the repo so a stray `git add -A` can never pick up 1.5 GB of weights:

```bash
export HF_HOME="$HOME/.cache/huggingface"    # add it to ~/.bashrc too
uv run python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
print('tokenizer ok')
"
```

That last command is the network check. If it fails behind a campus proxy, the model download will fail the same way — find out now, not 40 minutes into an `answer_dataset` run.

Disk and time budget: about 1.5 GB of weights, roughly 7 seconds to load, and 6–20 seconds per answer on a CPU depending on the machine. A 100-question run is 25–45 minutes. Plan around it: `--limit 3` for the dev loop, one full run at the end.

## Key design decisions

- **Where the snippet text comes from.** Store the chunk text in the index and look it up, or re-read the file by offset. Recommendation: **re-read the file.** `answer_dataset` receives only a results JSON — no index — so it must reconstruct text from `file_path` plus offsets anyway. The bonus is that it is a continuous, free audit of your offsets: if `read_source` returns something that does not look like an answer, your spans are wrong and every recall number you have is fiction.

- **How much context to give the model.** Everything retrieved, a fixed character budget, or a real token budget. Recommendation: **a fixed character budget of ~6000**, roughly 1500–1800 tokens, plus `truncation=True` as a hard backstop at the tokenizer. A real token budget means tokenizing every candidate snippet before deciding, which doubles the tokenizer work on the slowest part of the pipeline for an accuracy nobody measures. The failure mode is named in the debt ledger.

- **Prompt shape.** A raw completion prompt, or the model's chat template. Recommendation: **`apply_chat_template` with a system message.** Qwen3 is instruction-tuned; feeding it a bare prompt gets you continuation rather than answering. The system message is where "only use these sources" lives, and that instruction is the difference between RAG and a small model guessing.

- **Thinking mode.** Qwen3 emits `<think>...</think>` reasoning by default. Recommendation: **disable it** with `enable_thinking=False`, and strip the tags defensively anyway. On CPU, thinking spends your entire token budget on reasoning and truncates before the answer starts. Defensive stripping costs one regex and covers the case where the installed chat template ignores the flag.

- **Sampling.** Greedy or sampled. Recommendation: **greedy** (`do_sample=False`). Reproducibility matters more than variety here: when you change the prompt you need to know the output changed because of the prompt.

- **When the retriever returns nothing.** Refuse, or answer from the model's own knowledge. Recommendation: **refuse, in one sentence.** An ungrounded answer is the exact failure mode the subject calls out, and a small model inventing vLLM flags is worse than a clean "the retrieved sources do not answer this question."

- **How many sources to feed.** All k, or fewer. Recommendation: **feed all k but cap by the character budget**, best-scoring first. At `k=5` and ~1200-character chunks that is roughly 6000 characters — the budget and the default line up, which is not a coincidence.

## Debt taken on

Shortcut: answers are generated one at a time, with no batching and no cache. Bites you when: `answer_dataset` over 100 questions takes 25–45 minutes and you need three iterations. Paid off in: **not planned** — `--limit` covers the dev loop, and caching is bonus #4.

Shortcut: context is assembled by counting characters, not tokens. Bites you when: a source dense in punctuation or non-Latin text tokenizes far above the estimate. Paid off in: mitigated here by `truncation=True`, which clips at `MAX_INPUT_TOKENS` rather than erroring; a real token budget is not planned.

## Files in this phase

| File | New/Edit | What it holds |
|---|---|---|
| `src/generator.py` | new | `read_source`, `build_prompt`, `strip_thinking`, `Generator` |
| `src/datasets.py` | edit | `save_answers` |
| `src/__main__.py` | edit | `answer` and `answer_dataset` |
| `tests/test_generator.py` | new | Prompt assembly and source re-reading, with no model loaded |

## Steps

### 1. Re-read a source from disk by its span

**Why:** This is the bridge from "a span the grader compares" to "text a model can read", and it is where a wrong offset finally becomes visible to a human.

Create `src/generator.py` with the constants and `read_source`. It resolves the path against the repo root, reads with exactly the conventions `corpus.read_corpus_file` used, and slices.

```python
# src/generator.py
"""Generate grounded answers from retrieved sources with a local Qwen3."""

import re
from pathlib import Path
from typing import Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.models import MinimalSource

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

#: Roughly 1500-1800 tokens of context. See "Key design decisions".
MAX_CONTEXT_CHARS = 6000
MAX_NEW_TOKENS = 192
MAX_INPUT_TOKENS = 4096

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

NO_ANSWER = "The retrieved sources do not answer this question."

SYSTEM_PROMPT = (
    "You are a documentation assistant for the vLLM codebase. Answer the "
    "question using ONLY the numbered sources given to you. If the sources "
    "do not contain the answer, say so in one sentence. Answer in at most "
    "four sentences. Never invent an API name, a flag or a file path."
)


def read_source(source: MinimalSource, repo_root: Path) -> str:
    """The text *source* points at, read straight from the corpus file.

    Returns an empty string when the file is missing or unreadable: a
    stale results file must not crash a 40-minute answering run.
    """
    path = repo_root / source.file_path
    try:
        with path.open(
            encoding="utf-8", errors="replace", newline=""
        ) as handle:  # ← same conventions as corpus.read_corpus_file
            text = handle.read()
    except OSError:
        return ""
    return text[source.first_character_index : source.last_character_index]
```

`newline=""` and `errors="replace"` are not copy-paste habit. They are the exact pair `read_corpus_file` used in Phase 1, and if the two ever diverge the text you slice here stops being the text you indexed.

**Check:**

```bash
uv run python -c "
from pathlib import Path
from src.generator import read_source
from src.models import MinimalSource
src = MinimalSource(
    file_path='data/raw/vllm-0.10.1/docs/features/lora.md',
    first_character_index=4695, last_character_index=4790)
print(read_source(src, Path('.').resolve()))
"
```
prints the `### Using API Endpoints` section opening — the same text the ground truth points at.

### 2. Assemble the prompt

**Why:** This is the whole "augmenting" stage of RAG, and it is the part you will iterate on. Keeping it a pure function means you can look at the prompt without loading a 1.5 GB model.

Add `build_prompt` and `strip_thinking` to `src/generator.py`.

```python
# src/generator.py
# ... constants and read_source unchanged ...


def build_prompt(
    question: str,
    sources: List[MinimalSource],
    repo_root: Path,
) -> str:
    """Numbered sources plus the question, inside the character budget.

    Sources are consumed best-first and truncated at the budget, so the
    top-ranked snippet is never the one that gets cut.
    """
    blocks: List[str] = []
    used = 0
    for position, source in enumerate(sources, start=1):
        if used >= MAX_CONTEXT_CHARS:
            break
        snippet = read_source(source, repo_root)[: MAX_CONTEXT_CHARS - used]
        if not snippet.strip():
            continue
        used += len(snippet)
        blocks.append(f"[{position}] {source.file_path}\n{snippet}")

    context = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
    return (
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer using only the sources above."
    )


def strip_thinking(text: str) -> str:
    """Remove Qwen3 reasoning tags, complete or truncated."""
    cleaned = THINK_RE.sub("", text)
    return cleaned.replace("<think>", "").replace("</think>", "").strip()
```

The `[1]`, `[2]` numbering and the file path on each block are doing real work: they give the model a way to refer to a source, and they give *you* a way to tell at a glance whether an answer came from the docs or from a test file. Path in the block also nudges a small model toward naming the right file when the question asks where something lives.

`strip_thinking` handles the truncated case on purpose. When generation hits `MAX_NEW_TOKENS` mid-reasoning there is an opening `<think>` and no close, so the regex matches nothing and the two `replace` calls are what save the output from starting with a stray tag.

**Check:**

```bash
uv run python -c "
from pathlib import Path
from src.generator import build_prompt, strip_thinking
from src.models import MinimalSource
src = MinimalSource(file_path='data/raw/vllm-0.10.1/docs/features/lora.md',
                    first_character_index=4695, last_character_index=5200)
print(build_prompt('How do I load a LoRA adapter?', [src], Path('.').resolve())[:400])
print('---')
print(strip_thinking('<think>hmm</think>  The answer.'))
print(strip_thinking('<think>truncated reasoning'))
"
```
The prompt starts `Sources:\n[1] data/raw/vllm-0.10.1/docs/features/lora.md`, and the two strips print `The answer.` and `truncated reasoning`.

### 3. Load the model and generate

**Why:** Everything above is testable without a model. This is the one class that is not, so keep it thin.

Add the `Generator` class to `src/generator.py`.

```python
# src/generator.py
# ... functions unchanged ...


class Generator:
    """A local causal LM answering strictly from retrieved sources."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self.repo_root = Path(__file__).resolve().parent.parent
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()  # ← disables dropout; we never train

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Answer *question* from *sources*, or say it cannot.

        Returns the model's text with reasoning tags removed, falling back
        to a refusal sentence when generation produces nothing usable.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_prompt(question, sources, self.repo_root),
            },
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # ← Qwen3: skip <think> reasoning
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = generated[0][inputs["input_ids"].shape[-1] :]
        decoded = self.tokenizer.decode(completion, skip_special_tokens=True)
        return strip_thinking(decoded) or NO_ANSWER
```

`generated[0][inputs["input_ids"].shape[-1]:]` slices off the prompt. `model.generate` returns prompt **and** completion concatenated, so decoding the whole tensor prints your entire 6000-character context back at the user — the single most common first-run surprise with `transformers`.

`pad_token_id=self.tokenizer.eos_token_id` silences the "attention mask and pad token id were not set" warning and, more usefully, makes batch behaviour correct if you ever add batching for the bonus.

No `torch_dtype` argument: on CPU the float32 default is both the fastest and the only reliably supported option, and the keyword itself was renamed across recent `transformers` releases. Leaving it out is one fewer thing to break.

**Check:**

```bash
uv run python -c "
import time
from src.generator import Generator
from src.models import MinimalSource
started = time.perf_counter()
generator = Generator()
print(f'loaded in {time.perf_counter() - started:.1f}s')
src = MinimalSource(file_path='data/raw/vllm-0.10.1/docs/features/lora.md',
                    first_character_index=4695, last_character_index=6098)
started = time.perf_counter()
print(generator.answer('What endpoint loads a LoRA adapter at runtime?', [src]))
print(f'answered in {time.perf_counter() - started:.1f}s')
"
```
The answer must mention `/v1/load_lora_adapter`. If it does not, the sources are not reaching the model — print the prompt and look at it before touching anything else.

### 4. Write the answers file

**Why:** Same contract as Phase 2, one field wider. Reusing the same stripping discipline keeps the two output formats from drifting apart.

In `src/datasets.py`, add `MinimalAnswer` and `StudentSearchResultsAndAnswer` to the model imports, then add `save_answers` below `save_search_results`:

```python
# src/datasets.py
# ... load_dataset, load_questions, to_minimal, save_search_results unchanged ...


def save_answers(
    results: StudentSearchResultsAndAnswer,
    save_directory: Path,
    source_path: Path,
) -> Path:
    """Write *results* as ``<save_directory>/<source basename>``.

    *source_path* is the StudentSearchResults file the answers came from,
    so the output keeps the dataset name it was derived from.
    """
    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = save_directory / source_path.name
    cleaned = StudentSearchResultsAndAnswer(
        k=results.k,
        search_results=[
            MinimalAnswer(
                question_id=row.question_id,
                question=row.question,
                retrieved_sources=[
                    to_minimal(source) for source in row.retrieved_sources
                ],
                answer=row.answer,
            )
            for row in results.search_results
        ],
    )
    output_path.write_text(
        cleaned.model_dump_json(indent=2), encoding="utf-8"
    )
    return output_path


# ... load_search_results unchanged ...
```

**Check:** covered by the CLI run in step 5.

### 5. Add `answer` and `answer_dataset` to the CLI

**Why:** These are the last two commands the subject requires. `answer_dataset` in particular is invoked by the reference walkthrough with exactly two arguments, so its signature is not yours to redesign.

In `src/__main__.py`, add the imports and both methods.

```python
# src/__main__.py
# ... existing imports ...
from src.datasets import (
    load_questions,
    load_search_results,
    save_answers,
    save_search_results,
)
from src.generator import DEFAULT_MODEL, Generator
from src.models import (
    MinimalAnswer,
    MinimalSearchResults,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)

DEFAULT_ANSWER_OUT = str(
    REPO_ROOT / "data" / "output" / "search_results_and_answer"
)


class Cli:
    # ... index, search, search_dataset, evaluate unchanged ...

    def answer(
        self,
        query: str,
        k: int = 5,
        retriever: str = "bm25",
        model: str = DEFAULT_MODEL,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
    ) -> None:
        """Answer a single *query* from the top-*k* retrieved sources."""
        engine = Retriever.load(Path(processed_dir), str(retriever))
        sources = engine.search(str(query), int(k))
        started = time.perf_counter()
        generator = Generator(str(model))
        print(
            f"Loading {model} on cpu... ready in "
            f"{time.perf_counter() - started:.1f}s"
        )
        print(generator.answer(str(query), list(sources)))
        print()
        for position, source in enumerate(sources, start=1):
            print(
                f"[{position}] {source.file_path} "
                f"[{source.first_character_index}-"
                f"{source.last_character_index}]"
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = DEFAULT_ANSWER_OUT,
        model: str = DEFAULT_MODEL,
        limit: int = 0,
    ) -> None:
        """Generate an answer for every question in a results file.

        Args:
            limit: Answer only the first N questions. 0 means all of them.
        """
        results = load_search_results(Path(student_search_results_path))
        rows = results.search_results
        if int(limit) > 0:
            rows = rows[: int(limit)]
        print(f"Loaded {len(results.search_results)} questions")

        generator = Generator(str(model))
        answered = [
            MinimalAnswer(
                question_id=row.question_id,
                question=row.question,
                retrieved_sources=list(row.retrieved_sources),
                answer=generator.answer(
                    row.question, list(row.retrieved_sources)
                ),
            )
            for row in tqdm(rows, desc="Answering", unit="q")
        ]
        output_path = save_answers(
            StudentSearchResultsAndAnswer(
                search_results=answered, k=results.k
            ),
            Path(save_directory),
            Path(student_search_results_path),
        )
        print(f"Processed {len(answered)} of {len(results.search_results)} questions")
        print(
            "Saved student_search_results_and_answer to "
            f"{output_path}"
        )
```

`Generator(...)` is constructed **once**, before the loop. Constructing it inside would reload 1.5 GB of weights per question and turn a 40-minute run into an overnight one.

`--limit` defaults to 0 meaning "all", rather than to a number, because the default invocation in the subject's walkthrough passes no limit at all and must answer everything.

**Check:**

```bash
uv run python -m src answer "how do I serve a model with LoRA adapters?" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --limit 3
uv run python -c "
import json
d = json.load(open('data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json', encoding='utf-8'))
print(d['k'], len(d['search_results']))
print(sorted(d['search_results'][0]))
print(d['search_results'][0]['answer'][:200])
"
```
The last block prints `10 3`, then `['answer', 'question', 'question_id', 'retrieved_sources']`, then the first answer.

### 6. Test everything that does not need the model

**Why:** The generation call is slow, network-dependent and not worth mocking. Everything around it is fast, deterministic and exactly where the bugs are.

```python
# tests/test_generator.py
"""Prompt assembly and source re-reading, with no model loaded."""

from pathlib import Path

from src.generator import (
    MAX_CONTEXT_CHARS,
    build_prompt,
    read_source,
    strip_thinking,
)
from src.models import MinimalSource


def _write(tmp_path: Path, relative: str, text: str) -> MinimalSource:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="")
    return MinimalSource(
        file_path=relative,
        first_character_index=0,
        last_character_index=len(text),
    )


def test_read_source_slices_the_file(tmp_path: Path) -> None:
    source = _write(tmp_path, "data/raw/a.md", "hello world")
    source.last_character_index = 5
    assert read_source(source, tmp_path) == "hello"


def test_read_source_survives_a_missing_file(tmp_path: Path) -> None:
    source = MinimalSource(
        file_path="data/raw/gone.md",
        first_character_index=0,
        last_character_index=10,
    )
    assert read_source(source, tmp_path) == ""


def test_prompt_numbers_sources_and_names_files(tmp_path: Path) -> None:
    first = _write(tmp_path, "data/raw/a.md", "alpha content")
    second = _write(tmp_path, "data/raw/b.py", "beta content")
    prompt = build_prompt("why?", [first, second], tmp_path)
    assert "[1] data/raw/a.md" in prompt
    assert "[2] data/raw/b.py" in prompt
    assert prompt.rstrip().endswith("Answer using only the sources above.")


def test_prompt_respects_the_character_budget(tmp_path: Path) -> None:
    sources = [
        _write(tmp_path, f"data/raw/f{i}.md", "x" * 4000) for i in range(5)
    ]
    prompt = build_prompt("why?", sources, tmp_path)
    assert prompt.count("x") == MAX_CONTEXT_CHARS


def test_prompt_without_sources_says_so(tmp_path: Path) -> None:
    assert "(no sources retrieved)" in build_prompt("why?", [], tmp_path)


def test_strip_thinking_handles_open_and_closed_tags() -> None:
    assert strip_thinking("<think>hmm</think> Answer.") == "Answer."
    assert strip_thinking("<think>cut off mid") == "cut off mid"
    assert strip_thinking("  plain  ") == "plain"
```

`test_prompt_respects_the_character_budget` is the one that matters: five 4000-character sources are 20 000 characters of candidate context, and the assertion pins that exactly 6000 reach the prompt. Without it, a refactor that moves the `break` one line silently sends 20 000 characters into a 4096-token window and every answer gets truncated garbage.

**Check:** `uv run pytest -q` prints `34 passed`, and it finishes in seconds — no test in this file loads a model.

### 7. Do one full run and read ten answers

**Why:** A grading criterion here is "coherent, grounded, on point", and that is a human judgement. Nobody can make it for you, and it is the thing you will be asked about at the defense.

```bash
time uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Then read ten answers next to their questions:

```bash
uv run python -c "
import json, random
d = json.load(open('data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json', encoding='utf-8'))
for row in random.sample(d['search_results'], 10):
    print('Q:', row['question'])
    print('A:', row['answer'].replace(chr(10), ' ')[:300])
    print()
"
```

Judge each one against three questions: does it answer what was asked, is every claim in it traceable to a retrieved source, and does it invent a flag or a file path that does not exist. If more than two or three fail, the fix is almost always the system prompt or the context budget — not the model, which you are not allowed to change.

Common repairs, in the order worth trying: tighten the system prompt to name the failure you see ("do not describe the sources, answer the question"); drop `MAX_CONTEXT_CHARS` to 4000 so the relevant snippet is not buried; raise `MAX_NEW_TOKENS` if answers are being cut mid-sentence.

**Check:** you have a sentence about answer quality you would be willing to say out loud at the defense. It goes in the README.

## Common pitfalls

| Pitfall | Why it happens | Avoid it by |
|---|---|---|
| The answer starts with your whole prompt | `generate` returns prompt + completion concatenated | Slice with `generated[0][inputs["input_ids"].shape[-1]:]` |
| Every answer is `<think>` reasoning and no answer | Qwen3 thinking mode plus a small `max_new_tokens` | `enable_thinking=False` and `strip_thinking` |
| The model reloads for every question | `Generator()` constructed inside the loop | Construct once, before `tqdm` |
| `answer_dataset` needs the index | Assuming the retriever is required | It only needs `file_path` plus offsets; `read_source` reads the corpus directly |
| Answers reference sources that were never in the prompt | More retrieved sources than the budget fits | The budget truncates best-first; check `build_prompt` output when an answer looks unmoored |
| Model weights committed to git | `git add -A` after the cache lands inside the repo | `export HF_HOME` outside the repo, before the first download |
| `KeyError: 'qwen3'` on model load | `transformers` older than 4.51 | Pin `transformers>=4.51` in `pyproject.toml` |
| A 40-minute run discarded because the output path was wrong | `--save_directory` typo, discovered at the end | `--limit 3` first, always; check the file, then run the full set |
| Answers are fine on docs and nonsense on code | Code snippets read as context without their enclosing definition | Nothing to fix in this phase — it is a chunking observation, and it belongs in the README |

## Verify it's done

```bash
uv run python -m src answer "how do I serve a model with LoRA adapters?" --k 5
uv run python -m src answer "qwertyuiop asdfghjkl" --k 5
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions \
  --limit 5
uv run pytest -q
```

Expected:

```
Loading Qwen/Qwen3-0.6B on cpu... ready in 6.8s
Start the server with the --enable-lora flag and pass your adapter with
--lora-modules... (a grounded paragraph)

[1] data/raw/vllm-0.10.1/docs/features/lora.md [4695-5900]
... four more source lines ...
Loading Qwen/Qwen3-0.6B on cpu... ready in 6.6s
The retrieved sources do not answer this question.

Loaded 100 questions
Answering: 100%|█████████████████| 5/5 [01:12<00:00, 14.5s/q]
Processed 5 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json
34 passed
```

The nonsense query is the important line: `Retriever.search` returns nothing, `build_prompt` emits `(no sources retrieved)`, and the model is instructed to refuse. If it answers confidently about vLLM instead, your system prompt is not being applied — check that `apply_chat_template` is receiving the system message.

And the output file validates against the model the grader reads:

```bash
uv run python -c "
import json
from pathlib import Path
from src.models import StudentSearchResultsAndAnswer
path = Path('data/output/search_results_and_answer/UnansweredQuestions/dataset_docs_public.json')
parsed = StudentSearchResultsAndAnswer.model_validate(json.loads(path.read_text(encoding='utf-8')))
print(len(parsed.search_results), 'answers, k =', parsed.k)
print('all non-empty:', all(row.answer.strip() for row in parsed.search_results))
"
```
prints the count and `all non-empty: True`.

## Definition of done

- [ ] `answer <query> --k 5` prints a grounded paragraph plus its numbered sources
- [ ] A nonsense query refuses instead of inventing vLLM behaviour
- [ ] `answer_dataset` writes a valid `StudentSearchResultsAndAnswer` file, named after its input
- [ ] `--limit N` answers only the first N questions
- [ ] The model loads once per run, not once per question
- [ ] A full 100-question run completes and you have read ten of its answers
- [ ] No answer is empty
- [ ] `uv run pytest -q` passes and no test loads a model
- [ ] Weights are outside the repo and `git status` is clean
- [ ] Committed

## Deliberately NOT in this phase

- Batched or cached generation → bonus #4; `--limit` covers the dev loop
- Streaming output → not in v1
- Any change to retrieval → **Phase 3** is closed; if an answer is bad because the sources are bad, note it and finish the phase
- Answer-quality scoring → out of scope; the subject grades retrieval, grounding and prompt strategy
- Graceful handling of every malformed input → **Phase 5**

## Commit

```bash
git add -A
git commit -m "phase 4: grounded answer generation with Qwen3-0.6B"
```

## Next

→ **[Phase 5 — Survive a hostile reviewer](PHASE-05-survive-a-hostile-reviewer.md)**
