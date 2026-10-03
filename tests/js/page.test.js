"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { installFakeDom, FakeElement } = require("./fake-dom.js");

installFakeDom();

const page = require("../../app/static/app.js");

const noCitations = () => null;

function markdown(text, cite = noCitations) {
  const container = new FakeElement("div");
  page.renderMarkdown(container, text, cite);
  return container.children.map((child) => child.toHTML()).join("");
}

test("paragraphs, with line breaks inside a paragraph", () => {
  assert.equal(markdown("First line\nsecond line\n\nNext paragraph"), "<p>First line<br></br>second line</p><p>Next paragraph</p>");
});

test("bold, italic and inline code", () => {
  assert.equal(
    markdown("A **bold** and *italic* and `code` word"),
    "<p>A <strong>bold</strong> and <em>italic</em> and <code>code</code> word</p>",
  );
});

test("bullet and numbered lists", () => {
  assert.equal(markdown("- one\n- two"), "<ul><li>one</li><li>two</li></ul>");
  assert.equal(markdown("1. first\n2. second"), "<ol><li>first</li><li>second</li></ol>");
});

test("a list ends where the paragraph starts", () => {
  assert.equal(markdown("- one\n\nAfter"), "<ul><li>one</li></ul><p>After</p>");
});

test("fenced code blocks keep their text exactly and are not interpreted", () => {
  assert.equal(
    markdown("Run:\n```python\nprint('**hi**')\nx = [1]\n```\nDone"),
    "<p>Run:</p><pre><code>print('**hi**')\nx = [1]</code></pre><p>Done</p>",
  );
});

test("an unclosed code fence runs to the end", () => {
  assert.equal(markdown("```\nabc"), "<pre><code>abc</code></pre>");
});

test("headings become bold lines", () => {
  assert.equal(markdown("## Title\nText"), '<p class="md-heading">Title</p><p>Text</p>');
});

test("HTML in the text is shown as text, never interpreted", () => {
  const html = markdown('<img src=x onerror="alert(1)"> and <script>alert(1)</script>');
  assert.ok(!html.includes("<img"));
  assert.ok(!html.includes("<script"));
  assert.ok(html.includes("&lt;img src=x"));
});

test("only http and https links become links", () => {
  assert.equal(
    markdown("See [docs](https://example.com/a?b=1) now"),
    '<p>See <a href="https://example.com/a?b=1" rel="noopener noreferrer">docs</a> now</p>',
  );
  assert.ok(!markdown("[x](javascript:alert(1))").includes("<a"));
  assert.ok(!markdown("[x](data:text/html,hi)").includes("<a"));
});

test("citations use the supplied node, also inside bold text", () => {
  const cite = (number) => {
    if (number > 2) return null;
    const button = new FakeElement("button");
    button.className = "cite";
    button.textContent = `[${number}]`;
    return button;
  };
  assert.equal(
    markdown("Fact [1] and **bold [2]** but [7].", cite),
    '<p>Fact <button class="cite">[1]</button> and <strong>bold <button class="cite">[2]</button></strong> but [7].</p>',
  );
});

test("array syntax and list indexing are not mistaken for citations or italics", () => {
  assert.equal(markdown("Use items[0] and [a, b]."), "<p>Use items[0] and [a, b].</p>");
});

test("safeUrl accepts only absolute http(s) addresses", () => {
  assert.equal(page.safeUrl("https://example.com/x"), "https://example.com/x");
  assert.equal(page.safeUrl("http://example.com"), "http://example.com");
  assert.equal(page.safeUrl("javascript:alert(1)"), null);
  assert.equal(page.safeUrl("//example.com"), null);
  assert.equal(page.safeUrl("https://exa mple.com"), null);
});

test("plainText strips Markdown symbols from source excerpts", () => {
  const input = "# Heading\n**Bold** and *italic* and `code` and [link](https://a.b/c).\n\n\n\n- item";
  assert.equal(page.plainText(input), "Heading\nBold and italic and code and link.\n\n- item");
});

test("friendlyError explains network failures and keeps server messages", () => {
  assert.match(page.friendlyError(new TypeError("Failed to fetch")), /Could not reach the server/);
  assert.equal(page.friendlyError(new Error("Wait a minute.")), "Wait a minute.");
  assert.equal(page.friendlyError(null), "Something went wrong. Please try again.");
});

const SOURCES = [
  { number: 1, heading: "## Intro", text: "**first** text" },
  { number: 3, heading: "", text: "third text" },
];

function sourceState(list) {
  const [summary, first, third] = list.details.children;
  return {
    open: list.details.open,
    first: first.hidden ? "hidden" : "shown",
    third: third.hidden ? "hidden" : "shown",
    summary: summary.textContent,
  };
}

test("source list: labels use the answer's numbers and plain text", () => {
  const list = page.buildSources(SOURCES);
  const [, first, third] = list.details.children;
  assert.equal(first.children[0].textContent, "[1] Intro");
  assert.equal(first.children[1].textContent, "first text");
  assert.equal(third.children[0].textContent, "[3]");
  assert.deepEqual([list.has(1), list.has(2), list.has(3)], [true, false, true]);
});

test("source list: a citation shows only its source", () => {
  const list = page.buildSources(SOURCES);
  list.showOnly(1);
  assert.deepEqual(sourceState(list), {
    open: true,
    first: "shown",
    third: "hidden",
    summary: "Sources (2) · showing [1] only",
  });
  list.showOnly(3);
  assert.equal(sourceState(list).first, "hidden");
  assert.equal(sourceState(list).third, "shown");
});

test("source list: the same citation again closes it", () => {
  const list = page.buildSources(SOURCES);
  list.showOnly(3);
  list.showOnly(3);
  assert.deepEqual(sourceState(list), { open: false, first: "shown", third: "shown", summary: "Sources (2)" });
});

test("source list: the Sources line shows all, then closes, then opens again", () => {
  const list = page.buildSources(SOURCES);
  const summary = list.details.children[0];
  list.showOnly(1);
  summary.click();
  assert.deepEqual(sourceState(list), { open: true, first: "shown", third: "shown", summary: "Sources (2)" });
  summary.click();
  assert.equal(list.details.open, false);
  summary.click();
  assert.equal(list.details.open, true);
});

test("source list: showing a source scrolls to it and highlights it", () => {
  const list = page.buildSources(SOURCES);
  list.showOnly(3);
  const third = list.details.children[2];
  assert.equal(third.scrolled, true);
  assert.equal(third.classList.contains("highlight"), true);
});

test("answer: citations to existing sources become buttons that open them", () => {
  const answer = page.buildAnswer("It was 2020 [1][3] [9].", SOURCES);
  const buttons = answer.body.find((element) => element.tag === "button");
  assert.deepEqual(
    buttons.map((button) => button.textContent),
    ["[1]", "[3]"],
  );
  assert.equal(buttons[0].attributes["aria-label"], "Show source 1");
  buttons[1].click();
  assert.equal(answer.details.open, true);
  assert.ok(answer.body.toHTML().includes("[9]"));
});

test("answer: without sources there are no buttons and no source list", () => {
  const answer = page.buildAnswer("The page does not seem to cover it [1].", []);
  assert.equal(answer.details, null);
  assert.equal(answer.body.find((element) => element.tag === "button").length, 0);
  assert.ok(answer.body.toHTML().includes("[1]"));
});
