"use strict";

// Chat With Any Website: front-end behaviour.
// The helpers up to "Page wiring" only use `document` for creating elements, so they can be tested with a
// stand-in DOM (see tests/js). Everything is built from text nodes; no HTML string is ever inserted.

const EXAMPLE_URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation";
const STORAGE_KEY = "chat-with-any-website:page";

function createElement(tag, className) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  return element;
}

// Only absolute http(s) addresses may become links.
function safeUrl(value) {
  return /^https?:\/\/[^\s]+$/i.test(value) ? value : null;
}

// Strips Markdown symbols so source excerpts read as plain text.
function plainText(markdown) {
  return markdown
    .replace(/```\w*\n?/g, "")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/\[([^\]]+)\]\(https?:\/\/[^)]*\)/g, "$1")
    .replace(/(\*\*|__)(.+?)\1/g, "$2")
    .replace(/(^|[\s(])\*([^*\s][^*]*?)\*(?=[\s).,;:!?]|$)/g, "$1$2")
    .replace(/`([^`\n]+)`/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function friendlyError(error) {
  if (error instanceof TypeError) {
    return "Could not reach the server. Check that the app is running and your connection works, then try again.";
  }
  return error && error.message ? error.message : "Something went wrong. Please try again.";
}

const INLINE = /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*)|(\*[^\s*][^*\n]*?\*)|(\[\d+\])|(\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;

// Renders inline Markdown (code, bold, italic, links) and [n] citation markers into `parent`.
// `cite(number)` returns the node to use for a citation, or null to leave the marker as plain text.
function renderInline(parent, text, cite) {
  let last = 0;
  for (const match of text.matchAll(INLINE)) {
    const token = match[0];
    if (match.index > last) parent.append(text.slice(last, match.index));
    if (match[1]) {
      const code = createElement("code");
      code.textContent = token.slice(1, -1);
      parent.append(code);
    } else if (match[2]) {
      const strong = createElement("strong");
      renderInline(strong, token.slice(2, -2), cite);
      parent.append(strong);
    } else if (match[3]) {
      const emphasis = createElement("em");
      renderInline(emphasis, token.slice(1, -1), cite);
      parent.append(emphasis);
    } else if (match[4]) {
      parent.append(cite(Number(token.slice(1, -1))) || token);
    } else {
      const split = token.indexOf("](");
      const href = safeUrl(token.slice(split + 2, -1));
      if (href) {
        const link = createElement("a");
        link.href = href;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = token.slice(1, split);
        parent.append(link);
      } else {
        parent.append(token);
      }
    }
    last = match.index + token.length;
  }
  if (last < text.length) parent.append(text.slice(last));
}

const FENCE = /^\s*```/;
const HEADING = /^#{1,6}\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+/;
const ORDERED = /^\s*\d+[.)]\s+/;

function startsBlock(line) {
  return FENCE.test(line) || HEADING.test(line) || BULLET.test(line) || ORDERED.test(line);
}

// Renders a safe subset of Markdown (paragraphs, headings, lists, code blocks) into `container`.
function renderMarkdown(container, text, cite) {
  const lines = text.replace(/\r\n?/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
    } else if (FENCE.test(line)) {
      const code = [];
      i += 1;
      while (i < lines.length && !FENCE.test(lines[i])) code.push(lines[i++]);
      i += 1; // the closing fence, if there is one
      const pre = createElement("pre");
      const inner = createElement("code");
      inner.textContent = code.join("\n");
      pre.append(inner);
      container.append(pre);
    } else if (HEADING.test(line)) {
      const heading = createElement("p", "md-heading");
      renderInline(heading, line.match(HEADING)[1], cite);
      container.append(heading);
      i += 1;
    } else if (BULLET.test(line) || ORDERED.test(line)) {
      const marker = ORDERED.test(line) ? ORDERED : BULLET;
      const list = createElement(marker === ORDERED ? "ol" : "ul");
      while (i < lines.length && marker.test(lines[i])) {
        const item = createElement("li");
        renderInline(item, lines[i].replace(marker, ""), cite);
        list.append(item);
        i += 1;
      }
      container.append(list);
    } else {
      const paragraph = createElement("p");
      let first = true;
      while (i < lines.length && lines[i].trim() && (first || !startsBlock(lines[i]))) {
        if (!first) paragraph.append(createElement("br"));
        renderInline(paragraph, lines[i], cite);
        first = false;
        i += 1;
      }
      container.append(paragraph);
    }
  }
}

// Builds the collapsible source list. Clicking a citation shows only that source; clicking the
// "Sources" line shows all of them.
function buildSources(sources, options = {}) {
  const details = createElement("details");
  const summary = createElement("summary");
  details.append(summary);

  const items = new Map();
  let only = null; // the source number currently shown alone, or null for all

  function render() {
    summary.textContent =
      only === null ? `Sources (${sources.length})` : `Sources (${sources.length}) · showing [${only}] only`;
    items.forEach((item, number) => {
      item.hidden = only !== null && number !== only;
    });
  }

  sources.forEach((source) => {
    const item = createElement("div", "source");
    const label = createElement("span", "source-label");
    label.textContent = source.heading ? `[${source.number}] ${plainText(source.heading)}` : `[${source.number}]`;
    const body = createElement("div", "source-text");
    body.textContent = plainText(source.text);
    item.append(label, body);
    details.append(item);
    items.set(source.number, item);
  });

  summary.addEventListener("click", (event) => {
    event.preventDefault();
    if (details.open && only === null) {
      details.open = false;
    } else {
      only = null;
      details.open = true;
      render();
    }
  });

  function showOnly(number) {
    if (details.open && only === number) {
      only = null;
      details.open = false;
      render();
      return;
    }
    only = number;
    details.open = true;
    render();
    const item = items.get(number);
    item.scrollIntoView({ block: "nearest", behavior: options.reduceMotion ? "auto" : "smooth" });
    item.classList.add("highlight");
    setTimeout(() => item.classList.remove("highlight"), 1600);
  }

  render();
  return { details, showOnly, has: (number) => items.has(number) };
}

// Builds an assistant answer: Markdown text whose [n] markers are buttons that reveal the matching source.
function buildAnswer(text, sources, options = {}) {
  const body = createElement("div", "md");
  const sourceList = sources.length ? buildSources(sources, options) : null;

  function cite(number) {
    if (!sourceList || !sourceList.has(number)) return null;
    const button = createElement("button", "cite");
    button.type = "button";
    button.textContent = `[${number}]`;
    button.setAttribute("aria-label", `Show source ${number}`);
    button.addEventListener("click", () => sourceList.showOnly(number));
    return button;
  }

  renderMarkdown(body, text, cite);
  return { body, details: sourceList ? sourceList.details : null };
}

// ---------------------------------------------------------------------------------------------------
// Page wiring
// ---------------------------------------------------------------------------------------------------

function init() {
  const ingestForm = document.getElementById("ingest-form");
  const askForm = document.getElementById("ask-form");
  const urlInput = document.getElementById("url");
  const questionInput = document.getElementById("question");
  const ingestButton = document.getElementById("ingest-button");
  const askButton = document.getElementById("ask-button");
  const newChatButton = document.getElementById("new-chat");
  const statusEl = document.getElementById("status");
  const messagesEl = document.getElementById("messages");
  const pageBar = document.getElementById("page-bar");
  const pageLink = document.getElementById("page-link");
  const pageMeta = document.getElementById("page-meta");

  let page = null; // the page the questions are about: { url, title, chunks }
  let busy = false;

  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function setStatus(text, isError = false) {
    statusEl.textContent = text;
    statusEl.classList.toggle("error", isError);
  }

  function updateControls() {
    urlInput.disabled = busy;
    ingestButton.disabled = busy;
    newChatButton.disabled = busy;
    questionInput.disabled = busy || page === null;
    askButton.disabled = busy || page === null;
  }

  function setBusy(value) {
    busy = value;
    updateControls();
  }

  function savePage() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(page));
    } catch {
      // Storage can be unavailable (private windows); the page just will not be remembered.
    }
  }

  function showPage(newPage) {
    page = newPage;
    const href = safeUrl(page.url);
    if (href) pageLink.href = href;
    pageLink.textContent = page.title;
    pageMeta.textContent = `· ${page.chunks} chunks`;
    pageBar.hidden = false;
    savePage();
    updateControls();
  }

  function showEmptyState() {
    messagesEl.replaceChildren();
    const empty = createElement("div", "empty");
    if (page === null) {
      const first = createElement("p");
      first.textContent = "No page loaded yet.";
      const hint = createElement("p");
      const example = createElement("button", "link");
      example.type = "button";
      example.textContent = "try an example page";
      example.addEventListener("click", () => {
        urlInput.value = EXAMPLE_URL;
        urlInput.focus();
      });
      hint.append("Paste an address above, or ", example, ".");
      empty.append(first, hint);
    } else {
      const only = createElement("p");
      only.textContent = "Page loaded. Ask a question about it below.";
      empty.append(only);
    }
    messagesEl.append(empty);
  }

  function clearEmptyState() {
    messagesEl.querySelector(".empty")?.remove();
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // Long answers should be read from the top, so scroll to the start of the new message.
  function scrollToMessage(element) {
    messagesEl.scrollTo({ top: Math.max(element.offsetTop - 12, 0), behavior: reduceMotion() ? "auto" : "smooth" });
  }

  function addMessage(role, text) {
    clearEmptyState();
    const element = createElement("div", `message ${role}`);
    element.textContent = text;
    messagesEl.append(element);
    scrollToBottom();
    return element;
  }

  function addAnswer(text, sources) {
    clearEmptyState();
    const element = createElement("div", "message assistant");
    const answer = buildAnswer(text, sources, { reduceMotion: reduceMotion() });
    element.append(answer.body);
    if (answer.details) element.append(answer.details);
    messagesEl.append(element);
    scrollToMessage(element);
  }

  function addTyping() {
    clearEmptyState();
    const element = createElement("div", "message assistant");
    element.setAttribute("aria-label", "Thinking");
    const dots = createElement("span", "typing");
    dots.append(createElement("span"), createElement("span"), createElement("span"));
    element.append(dots);
    messagesEl.append(element);
    scrollToBottom();
    return element;
  }

  async function postJson(path, payload) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("; ") : data.detail;
      throw new Error(detail || `Request failed with status ${response.status}`);
    }
    return data;
  }

  ingestForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    setBusy(true);
    setStatus("Fetching, cleaning, chunking and embedding the page. Long pages can take up to a minute…");
    try {
      const result = await postJson("/api/ingest", { url: urlInput.value });
      urlInput.value = result.url;
      showPage({ url: result.url, title: result.title, chunks: result.chunk_count });
      showEmptyState();
      setStatus("");
    } catch (error) {
      // The previous page, if any, stays loaded and visible in the bar above the conversation.
      const kept = page ? ` You are still chatting with "${page.title}".` : "";
      setStatus(friendlyError(error) + kept, true);
    } finally {
      setBusy(false);
      if (page) questionInput.focus();
    }
  });

  askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const question = questionInput.value.trim();
    if (!question || !page) return;

    addMessage("user", question);
    questionInput.value = "";
    const typing = addTyping();
    setBusy(true);
    setStatus("Thinking…");
    try {
      const result = await postJson("/api/ask", { url: page.url, question });
      typing.remove();
      addAnswer(result.answer, result.sources);
    } catch (error) {
      typing.remove();
      addMessage("assistant error", friendlyError(error));
      questionInput.value = question; // keep the question so it can be sent again
    } finally {
      setStatus("");
      setBusy(false);
      questionInput.focus();
    }
  });

  newChatButton.addEventListener("click", () => {
    showEmptyState();
    setStatus("");
    questionInput.focus();
  });

  // Restore the page from earlier in this browser tab; its text is still stored on the server.
  try {
    const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
    if (saved && saved.url && saved.title) {
      urlInput.value = saved.url;
      showPage(saved);
    }
  } catch {
    // Ignore unreadable or unavailable storage.
  }
  showEmptyState();
  updateControls();
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { safeUrl, plainText, friendlyError, renderInline, renderMarkdown, buildSources, buildAnswer };
}

if (typeof document !== "undefined" && document.getElementById && document.getElementById("ingest-form")) {
  init();
}
