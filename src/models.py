"""Pydantic models exchanged between the stages of the RAG pipeline."""

import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MinimalSource(BaseModel):
    """a file and a character span inside it, that represent the chunck of data
    from this file that we want to give to the LLM to extract the data it needs
    from it.

    Attributes:
        file_path: forward-slash-relative path to the file.
        first_character_index: Index of the first covered character.
        last_character_index: Index one past the last covered character.
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
    """A retrieved source plus its retrieval score."""

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
    """Output of search_dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer_dataset."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """One indexed slice of one corpus file.

    Attributes:
        file_path: Path to the source file.
        first_character_index: Index of the first covered character.
        last_character_index: Index one past the last covered character.
        text: The raw chunk text, as read from the file.
        indexed_text: Text to search on instead of text, if set.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    indexed_text: Optional[str] = None

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
