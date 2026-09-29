"""Tokenisation shared by the index and every query."""

import re
from typing import Iterator, List

import snowballstemmer

# Reduces inflected words to a common root (e.g. "loading"/"loaded" -> "load")
# so a query and a doc using different word forms still overlap in the
# lexical (BM25) index.
_STEMMER = snowballstemmer.stemmer("english")

# This defines the pattern for finding tokens.
# example: "user_42 hello" will be ['user_42', 'hello']
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")

# Handle camelCase and PascalCase names and split the word.
# example: getName should be (get, Name)
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Characters that separate the parts of a path.
PATH_SEPARATORS = re.compile(r"[/\.\-_]+")

# Very short tokens are ignored.
MIN_TOKEN_LENGTH = 2


def split_identifier(token: str) -> List[str]:
    """Parts of `token`, split on underscores and camelCase humps."""
    parts: List[str] = []

    for piece in token.split("_"):
        if not piece:
            continue
        parts.extend(part for part in CAMEL.split(piece) if part)

    return parts


def analyze(text: str) -> Iterator[str]:
    """function that takes text and returns an iterator
    something you loop over lazily, one value at a time, instead
    of building a full list upfront."""
    # `finditer()` will find every token
    for match in TOKEN.finditer(text):
        # `group(0)` means: Give me the actual text that matched.
        token = match.group(0)

        lowered = token.lower()

        if len(lowered) >= MIN_TOKEN_LENGTH:
            yield lowered

        parts = split_identifier(token)
        if len(parts) > 1:
            for part in parts:
                lowered_part = part.lower()
                if len(lowered_part) >= MIN_TOKEN_LENGTH:
                    yield lowered_part


def analyze_stemmed(text: str) -> List[str]:
    """Like `analyze`, but stems every token to its root form.

    This is the analyzer the lexical index uses, for chunks and queries.
    """
    tokens = list(analyze(text))
    if not tokens:
        return tokens
    stemmed: List[str] = _STEMMER.stemWords(tokens)
    return stemmed


def path_tokens(file_path: str) -> str:
    """Words carried by `file_path` itself, as indexable text."""
    words: List[str] = []

    for piece in PATH_SEPARATORS.split(file_path):
        if not piece:
            continue

        words.append(piece)
        parts = split_identifier(piece)
        if len(parts) > 1:
            words.extend(parts)
    return " ".join(words)
