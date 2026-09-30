"""
rag.core.validation
----------------------
Validates and preprocesses the raw user question BEFORE it is
embedded (query.received -> retrieval.hybrid) or stuffed into the
LLM prompt. Previously the raw input() string went straight into
the chain with only a whitespace .strip().

Checks:
  - empty / whitespace-only
  - max length (MAX_QUESTION_LENGTH, default 2000 chars)
  - strips control / null characters that serve no purpose in a
    question and can break downstream tokenization or logging
"""

import re
from typing import Optional, Tuple

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE_RE = re.compile(r"\s+")


def validate_question(question: str, max_length: int = 2000) -> Tuple[Optional[str], Optional[str]]:
    """Returns (cleaned_question, None) if valid, or (None, reason) if rejected.

    reason is one of: "empty", "too_long"
    """
    if question is None:
        return None, "empty"

    cleaned = _CONTROL_CHARS_RE.sub("", question)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()

    if not cleaned:
        return None, "empty"

    if len(cleaned) > max_length:
        return None, "too_long"

    return cleaned, None
