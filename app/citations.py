"""Helpers for the numbered citations (``[1]``, ``[2][3]``) in generated answers."""

import re

# Some models cite in their own style, for example 【1†L1-L3】, which refers to line numbers that do not exist here.
_FULLWIDTH = re.compile(r"【([^】]*)】")
_WITH_SUFFIX = re.compile(r"\[(\d+)†[^\]]*\]")
_ADJACENT = re.compile(r"(\[\d+\])(?:\s*\1)+")
_SPACE_BETWEEN = re.compile(r"(\[\d+\])[ \t]+(?=\[\d+\])")
_CITATION = re.compile(r"\[(\d+)\]")


def _fullwidth_to_plain(match: re.Match[str]) -> str:
    numbers = re.findall(r"\d+", match.group(1).split("†", 1)[0])
    return "".join(f"[{number}]" for number in numbers) or match.group(0)


def normalize_citations(text: str) -> str:
    """Rewrite any citation style to plain ``[n]`` markers.

    Converts ``【1†L1-L3】`` and ``[1†L1-L3]`` to ``[1]``, joins neighbouring citations (``[1] [2]`` becomes
    ``[1][2]``) and collapses repeats (``[1][1]`` becomes ``[1]``).

    Args:
        text: The answer as written by the model.

    Returns:
        The answer with uniform citation markers.
    """
    text = _FULLWIDTH.sub(_fullwidth_to_plain, text)
    text = _WITH_SUFFIX.sub(r"[\1]", text)
    text = _SPACE_BETWEEN.sub(r"\1", text)
    return _ADJACENT.sub(r"\1", text)


def cited_numbers(text: str, available: int) -> list[int]:
    """List the excerpt numbers an answer cites, in order of first appearance.

    Args:
        text: The answer, with ``[n]`` markers.
        available: How many excerpts were given to the model; numbers outside ``1..available`` are ignored.

    Returns:
        Distinct excerpt numbers that exist, in the order they are first cited.
    """
    numbers: list[int] = []
    for match in _CITATION.finditer(text):
        number = int(match.group(1))
        if 1 <= number <= available and number not in numbers:
            numbers.append(number)
    return numbers
