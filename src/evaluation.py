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


def recall_at_k(
    results: Dict[str, List[MinimalSource]],
    truth: RagDataset,
    k_values: Sequence[int] = REPORTED_K,
) -> Dict[int, float]:
    """Mean per-question recall at each k in `k_values`.

    Questions in `truth` that are missing from `results` score 0.0, because a
    question you did not answer is a question you did not retrieve.

    Raises:
        ValueError: If `truth` carries no answered question to score against.
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
