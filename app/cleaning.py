"""Step 3: normalize extracted text before chunking."""

import re
import unicodedata

_FOOTNOTES = re.compile(r"<sup>.*?</sup>", re.DOTALL)
_EDIT_LINKS = re.compile(r"\n?\\\[edit\\\]", re.IGNORECASE)
_INLINE_TAGS = re.compile(r"</?(?:sub|sup|span|br)\b[^>]*>")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​-‏﻿]")
_INLINE_WHITESPACE = re.compile(r"[ 	]+")  # NFKC normalization already maps non-breaking spaces to spaces
_EXTRA_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize unicode and whitespace, and drop repeated paragraphs (menus, cookie banners, etc.)."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _FOOTNOTES.sub("", text)
    text = _EDIT_LINKS.sub("", text)
    text = _INLINE_TAGS.sub("", text)

    lines = [_INLINE_WHITESPACE.sub(" ", line).strip() for line in text.split("\n")]
    text = _EXTRA_NEWLINES.sub("\n\n", "\n".join(lines))

    seen: set[str] = set()
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        key = paragraph.casefold()
        if not paragraph or key in seen:
            continue
        seen.add(key)
        paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)
