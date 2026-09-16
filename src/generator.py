"""Generate a grounded answer from retrieved spans with a local Qwen model."""

from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.models import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Characters of context handed to the model. Qwen3-0.6B has a large context
# window, but every extra token costs CPU seconds, and the answer is meant to
# come from the top few spans anyway.
MAX_CONTEXT_CHARS = 6000

SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. Use only the provided "
    "context. If the context does not contain the answer, say so. Answer in "
    "at most four sentences, and name the file you used."
)


def read_span(repo_root: Path, source: MinimalSource) -> str:
    """Re-read the text of `source` from the corpus.

    The index stores spans, not text, so the file is the single source of
    truth. It is opened exactly as the indexer opened it, or the offsets
    would not line up.
    """
    path = repo_root / source.file_path
    if not path.is_file():
        return ""
    with path.open(encoding="utf-8", errors="replace", newline="") as handle:
        text = handle.read()
    return text[source.first_character_index:source.last_character_index]


def build_context(
    repo_root: Path,
    sources: Sequence[MinimalSource],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Concatenate the spans of `sources`, best first, within a budget."""
    blocks: List[str] = []
    used = 0
    for source in sources:
        span = read_span(repo_root, source).strip()
        if not span:
            continue
        block = f"--- {source.file_path} ---\n{span}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    """The chat messages sent to the model."""
    if context:
        user = f"Context:\n{context}\n\nQuestion: {question}"
    else:
        user = (
            f"Question: {question}\n\n"
            "No context was retrieved for this question."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


class Generator:
    """Wraps a local causal language model, loaded once and reused."""

    def __init__(self, tokenizer: Any, model: Any) -> None:
        self.tokenizer = tokenizer
        self.model = model

    @classmethod
    def load(cls, model_name: str = MODEL_NAME) -> "Generator":
        """Load `model_name` on CPU.

        Imported here rather than at module import time: transformers and
        torch together take seconds to import, and no other command needs
        them.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        return cls(tokenizer, model)

    def answer(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Greedy, reproducible answer to `question` grounded in `context`."""
        import torch

        messages = build_messages(question, context)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _strip_thinking(text).strip()


def _strip_thinking(text: str) -> str:
    """Drop a Qwen3 <think> block if the template emitted one anyway."""
    if "</think>" in text:
        return text.split("</think>", 1)[1]
    return text
