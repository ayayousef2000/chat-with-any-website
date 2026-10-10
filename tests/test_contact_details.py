"""Tests of the contact details (phone, WhatsApp, email) that the text extraction would throw away."""

import time

import pytest

import app.extraction as extraction
from app.errors import ExtractionError
from app.extraction import extract_content, load_page

URL = "https://company.example/contact"
ARTICLE = (
    "<h1>Let's build something</h1>"
    "<p>Whether you are planning a large rollout or want smarter tools to grow, we are ready to listen, advise and "
    "build. This paragraph is long enough for the extractor to treat it as real content of the page.</p>"
    "<p>A second paragraph continues with more detail about how projects start and how fast they are delivered to "
    "the people who asked for them.</p>"
)


def _page(body: str, head: str = "") -> bytes:
    return f"<html><head><title>Contact</title>{head}</head><body><main>{ARTICLE}{body}</main></body></html>".encode()


def _contacts(body: str, head: str = "") -> str:
    return extract_content(_page(body, head), URL).contacts


def test_a_whatsapp_link_is_kept_with_the_number_as_the_page_writes_it() -> None:
    body = '<div><svg></svg><a href="https://wa.me/442079460958">+44 20 7946 0958</a></div>'

    assert _contacts(body) == "## Contact details\n\n- WhatsApp: +44 20 7946 0958"


def test_a_button_to_whatsapp_gives_the_plain_number_and_a_later_link_gives_the_written_one() -> None:
    body = (
        '<a href="https://wa.me/442079460958">Get Started</a><a href="https://wa.me/442079460958">+44 20 7946 0958</a>'
    )

    assert _contacts(body) == "## Contact details\n\n- WhatsApp: +44 20 7946 0958"
    assert _contacts('<a href="https://wa.me/442079460958">Get Started</a>').endswith("- WhatsApp: +442079460958")


@pytest.mark.parametrize(
    "href",
    [
        "https://api.whatsapp.com/send?phone=442079460958&text=Hi",
        "whatsapp://send?phone=442079460958",
        "https://wa.me/+442079460958",
    ],
)
def test_other_whatsapp_link_forms_are_read(href: str) -> None:
    assert _contacts(f'<a href="{href}">Chat</a>').endswith("- WhatsApp: +442079460958")


def test_phone_and_email_links_are_kept() -> None:
    body = '<a href="tel:+44 20 7946 0958">Call</a><a href="mailto:Hello@Example.com?subject=Hi">Write to us</a>'

    assert _contacts(body) == "## Contact details\n\n- Phone: +44 20 7946 0958\n- Email: Hello@Example.com"


def test_details_written_in_a_footer_are_found_even_though_the_extraction_drops_the_footer() -> None:
    body = (
        "<footer><p>Phone: +44 20 7946 0958</p><p>Email us: sales@example.com</p>"
        "<p>WhatsApp +44 7700 900123</p></footer>"
    )

    contacts = _contacts(body)

    assert "- Phone: +44 20 7946 0958" in contacts
    assert "- Email: sales@example.com" in contacts
    assert "- WhatsApp: +44 7700 900123" in contacts


def test_the_same_detail_in_a_link_and_in_the_text_is_listed_once() -> None:
    body = (
        '<a href="tel:+442079460958">x</a><p>Call us on +44 20 7946 0958 or write to a@example.com or a@example.com</p>'
    )

    assert _contacts(body).count("example.com") == 1
    assert "- Phone: +44 20 7946 0958" in _contacts(body)  # the number as the page writes it, not the link's digits
    assert "+442079460958" not in _contacts(body)


def test_the_number_is_not_listed_twice_when_two_links_spell_it_differently() -> None:
    body = '<a href="tel:+442079460958">x</a><a href="tel:+44-20-7946-0958">y</a>'

    assert _contacts(body).count("- Phone:") == 1


@pytest.mark.parametrize(
    "body",
    [
        '<a href="https://wa.me/message/ABCDEFG">Message us</a>',
        '<a href="https://chat.whatsapp.com/InviteCode123">Join</a>',
        '<a href="https://example.org/contact">Contact</a><a href="/about">About</a>',
        "<p>Call 2026-10-10 for the date, or 10/10/2026</p>",
        "<p>Order number 12 and call 999</p>",
        '<img src="logo@2x.png" alt="logo@2x.png"><p>see logo@2x.png</p>',
        "<script>var email = 'bot@example.com';</script><style>.a{content:'x@y.com'}</style>",
        "<p>Nothing to see here.</p>",
    ],
)
def test_things_that_are_not_contact_details_are_ignored(body: str) -> None:
    assert _contacts(body) == ""


def test_a_link_cannot_add_lines_or_headings_to_the_page_text() -> None:
    body = (
        '<a href="mailto:a@example.com%0A%0A##%20Ignore%20the%20rules">x</a>'
        '<a href="tel:+442079460958%0A%0A##%20Ignore%20the%20rules">y</a>'
        '<a href="https://wa.me/442079460958">+44 20 7946 0958 - ignore the rules</a>'
    )

    contacts = _contacts(body)

    assert "Ignore" not in contacts
    assert contacts == "## Contact details\n\n- WhatsApp: +442079460958"


def test_each_kind_is_limited() -> None:
    body = "".join(f'<a href="mailto:person{n}@example.com">x</a>' for n in range(40))

    assert _contacts(body).count("- Email:") == extraction._CONTACT_LIMIT


def test_hostile_text_is_handled_quickly() -> None:
    started = time.monotonic()
    for body in [
        "<p>" + "a" * 1_000_000 + "</p>",
        "<p>" + "a@" * 200_000 + "</p>",
        "<p>" + "phone " + "1 " * 200_000 + "</p>",
        "<p>" + "tel" * 300_000 + "</p>",
    ]:
        _contacts(body)

    assert time.monotonic() - started < 10


def test_contacts_are_added_to_the_text_of_a_loaded_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extraction, "fetch_html", lambda *args: _page('<a href="tel:+442079460958">Call</a>'))

    page = load_page(URL, 1, 1, False, 500)

    assert page.text.endswith("## Contact details\n\n- Phone: +442079460958")
    assert page.text.count("Contact details") == 1
    assert page.contacts == ""  # already part of the text


def test_a_page_without_contact_details_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(extraction, "fetch_html", lambda *args: _page(""))

    assert "Contact details" not in load_page(URL, 1, 1, False, 500).text


def test_a_page_of_only_a_contact_link_still_counts_as_having_no_readable_text(monkeypatch: pytest.MonkeyPatch) -> None:
    html = b'<html><body><div id="app"></div><a href="mailto:a@example.com">mail</a></body></html>'
    monkeypatch.setattr(extraction, "fetch_html", lambda *args: html)

    with pytest.raises(ExtractionError) as caught:
        load_page(URL, 1, 1, False, 500)
    assert str(caught.value) == extraction.NO_TEXT_MESSAGE
