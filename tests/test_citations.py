import pytest

from app.citations import cited_numbers, normalize_citations


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Introduced in 2020 【1†L1-L3】", "Introduced in 2020 [1]"),
        ("In 2020 【1†L1-L3】 【2†L1-L2】.", "In 2020 [1][2]."),
        ("Plain citation [1] stays.", "Plain citation [1] stays."),
        ("With suffix [3†L5-L9] here.", "With suffix [3] here."),
        ("Full-width without suffix 【4】.", "Full-width without suffix [4]."),
        ("Several in one 【1†L1-L2, 2†L3】.", "Several in one [1]."),
        ("Repeated [1][1] and [2] [2].", "Repeated [1] and [2]."),
        ("Neighbours [1] [2] [3].", "Neighbours [1][2][3]."),
        ("No citations at all.", "No citations at all."),
        ("Array syntax [a, b] and list[0] stay.", "Array syntax [a, b] and list[0] stay."),
    ],
)
def test_normalize_citations(raw: str, expected: str) -> None:
    assert normalize_citations(raw) == expected


def test_normalize_citations_leaves_unrecognised_brackets() -> None:
    assert normalize_citations("Odd 【see below】 marker") == "Odd 【see below】 marker"


def test_cited_numbers_in_order_of_first_appearance_without_duplicates() -> None:
    assert cited_numbers("A [3] B [1] C [3][2].", available=5) == [3, 1, 2]


def test_cited_numbers_ignores_numbers_outside_range() -> None:
    assert cited_numbers("A [0] B [6] C [2].", available=5) == [2]


def test_cited_numbers_without_citations() -> None:
    assert cited_numbers("The page does not seem to cover it.", available=5) == []
