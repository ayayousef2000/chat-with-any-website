from app.cleaning import clean_text

NBSP = chr(0xA0)


def test_collapses_whitespace_and_blank_lines() -> None:
    assert clean_text("a \t b" + NBSP * 2 + "c\n\n\n\n\nd") == "a b c\n\nd"


def test_normalizes_line_endings() -> None:
    assert clean_text("one\r\n\r\ntwo\rthree") == "one\n\ntwo\nthree"


def test_removes_zero_width_and_control_characters() -> None:
    assert clean_text("he" + chr(0x200B) + "llo\x00 wor" + chr(0xFEFF) + "ld") == "hello world"


def test_removes_footnote_markers_and_inline_tags() -> None:
    assert clean_text("Fact.<sup>\\[1\\]</sup> More<br>text <span class='x'>here</span>.") == "Fact. Moretext here."


def test_removes_edit_links_from_headings() -> None:
    assert clean_text("## History\n\\[edit\\]\n\nBody.") == "## History\n\nBody."


def test_drops_repeated_paragraphs_case_insensitively() -> None:
    text = "Accept cookies\n\nReal content.\n\naccept COOKIES\n\nMore content."
    assert clean_text(text) == "Accept cookies\n\nReal content.\n\nMore content."


def test_empty_input_stays_empty() -> None:
    assert clean_text("  \n\n  ") == ""
