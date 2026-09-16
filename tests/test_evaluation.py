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
