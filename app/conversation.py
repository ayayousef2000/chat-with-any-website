"""The earlier turns of a conversation, as the browser sends them along with a follow-up question.

The server keeps no conversation: the browser holds the last few questions and answers and sends them with each new
question, so nothing about a conversation is stored here. The turns are text from the visitor's browser and are
treated as untrusted, like the text of a page.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

MAX_TURNS = 3
MAX_QUESTION_CHARS = 300
MAX_ANSWER_CHARS = 600
MAX_STANDALONE_CHARS = 500

_QUOTES = "\"'" + "".join(chr(code) for code in (0x201C, 0x201D, 0x2018, 0x2019)) + " "  # straight and curly quotes
_CITATION = re.compile(r"\s*\[\d+\]")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class Turn:
    """One question of the visitor and the answer that was given to it."""

    question: str
    answer: str


def _clip(text: str, limit: int) -> str:
    text = _SPACES.sub(" ", text).strip()
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}…"


def compact(turns: Sequence[Turn]) -> list[Turn]:
    """Keep the last few turns, short and without citation markers, so that they cost few tokens.

    Args:
        turns: The turns as the browser sent them, oldest first.

    Returns:
        At most ``MAX_TURNS`` turns, each question and answer clipped to its limit. Turns that are empty after
        cleaning are dropped.
    """
    cleaned = [
        Turn(_clip(turn.question, MAX_QUESTION_CHARS), _clip(_CITATION.sub("", turn.answer), MAX_ANSWER_CHARS))
        for turn in turns[-MAX_TURNS:]
    ]
    return [turn for turn in cleaned if turn.question and turn.answer]


def usable_question(original: str, rewritten: str) -> str:
    """Take the rewritten question when it is usable, and the original one otherwise.

    A model can answer with an empty reply, several lines or a long text instead of one question. Only the first
    line is used, quotes around it are removed, and a result that is empty or too long is not trusted.
    """
    lines = [line.strip() for line in rewritten.splitlines() if line.strip()]
    candidate = lines[0].strip(_QUOTES) if lines else ""
    return candidate if candidate and len(candidate) <= MAX_STANDALONE_CHARS else original
