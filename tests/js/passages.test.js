"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { installFakeDom, FakeElement } = require("./fake-dom.js");

installFakeDom();

const page = require("../../app/static/app.js");

// The info box of the Arabic Wikipedia page for Ibn Khaldun, as it appears in a source.
const INFOBOX = [
  "ابن خلدون",
  "الإمام",
  "ابن خلدون",
  "رسمٌ تخيُّلي لابن خلدون",
  "معلومات شخصية",
  "اسم الولادة  ·  عبد الرحمن بن محمد بن محمد الحضرمي الإشبيلي",
  "الميلاد  ·  1 رمضان 732هـ = 27 مايو 1332م تُونُس، إفريقية، الدولة الحفصية",
  "الوفاة  ·  28 رمضان 808هـ = 17 مارس 1406م (76 سنة) القاهرة، مصر، الدولة المملوكية",
  "الكنية  ·  أبو زيد",
  "اللقب  ·  ولي الدين",
  "العرق  ·  عربي",
  "الديانة  ·  الإسلام",
  "المذهب الفقهي  ·  مالكي",
  "الطائفة  ·  أهل السنة والجماعة",
  "العقيدة  ·  أشعرية",
  "إخوة وأخوات  ·  يحيى بن خلدون",
  "العصر  ·  العباسي المملوكي",
  "المدرسة الأم  ·  جامعة الزيتونة",
  "اللغات  ·  العربية",
  "أعمال بارزة  ·  مقدمة ابن خلدون، وكتاب العبر، ولباب المحصل في أصول الدين",
].join("\n");

function passageFor(text, question, answer) {
  const found = page.findPassage(text, question, answer);
  return found ? found.unit : null;
}

test("normalizeForMatch ignores vowel marks, letter variants and capitals", () => {
  assert.equal(page.normalizeForMatch("ابنُ خَلدون"), "ابن خلدون");
  assert.equal(page.normalizeForMatch("إسلام"), "اسلام");
  assert.equal(page.normalizeForMatch("الديانة"), "الديانه");
  assert.equal(page.normalizeForMatch("Café Éléphant"), "cafe elephant");
});

test("the birth question finds the birth line of the Arabic info box", () => {
  const unit = passageFor(INFOBOX, "متى ولد ابن خلدون", "ولد ابن خلدون في 1 رمضان 732 هـ، أي ما يوافق 27 مايو 1332 م [1]");
  assert.match(unit, /^الميلاد/);
});

test("a different word form still finds the line: religion", () => {
  // The question says "ديانته" and the answer "إسلامه"; the line says "الديانة" and "الإسلام".
  const unit = passageFor(INFOBOX, "ما ديانته", "إسلامه [1]");
  assert.match(unit, /^الديانة/);
});

test("other questions about the same section find their own lines", () => {
  assert.match(passageFor(INFOBOX, "ما مذهبه الفقهي", "مالكي [1]"), /^المذهب الفقهي/);
  assert.match(passageFor(INFOBOX, "اعطني اسم ولادة ابن خلدون", "عبد الرحمن بن محمد بن محمد الحضرمي الإشبيلي [1]"), /^اسم الولادة/);
  assert.match(passageFor(INFOBOX, "اين توفي", "توفي في القاهرة [1]"), /^الوفاة/);
});

const PROSE = [
  "Alexander Graham Bell was born in Edinburgh in 1847.",
  "He was awarded the first US patent for the telephone in 1876.",
  "Bell also worked on aeronautics and hydrofoils in his later years.",
  "He died in 1922 and was buried in Nova Scotia.",
].join(" ");

test("an English question finds the sentence with the answer", () => {
  assert.match(passageFor(PROSE, "When did he get the patent for the telephone?", "In 1876 [1]"), /first US patent/);
  assert.match(passageFor(PROSE, "Where is he buried?", "He was buried in Nova Scotia [1]"), /buried in Nova Scotia/);
});

test("numbers in the answer point to the line that holds them", () => {
  assert.match(passageFor(PROSE, "What year did he die?", "1922 [1]"), /died in 1922/);
});

test("citation markers in the answer are not used for matching", () => {
  const text = "Unrelated line one.\nUnrelated line two.\nThe number 1 appears here.";
  assert.equal(passageFor(text, "zzzz", "[1][2][3]"), null);
});

test("nothing matches: no passage is chosen", () => {
  assert.equal(passageFor(PROSE, "Quantum chromodynamics?", "Gluons [1]"), null);
  assert.equal(page.findPassage("", "q", "a"), null);
});

test("long lines are split into sentences, and a very long sentence at a space", () => {
  const sentence = "This is a fairly long sentence about retrieval and ranking. ";
  const units = page.splitUnits(sentence.repeat(10));
  assert.ok(units.length >= 10);
  assert.ok(units.every((unit) => unit.length <= 320));

  const arabic = page.splitUnits(`${"جملة عربية طويلة عن التاريخ والاجتماع. ".repeat(8)}\nسطر قصير`);
  assert.ok(arabic.length > 8, "Arabic full stops split sentences");
  assert.equal(arabic.at(-1), "سطر قصير");

  const endless = page.splitUnits("word ".repeat(300));
  assert.ok(endless.length > 3);
  assert.ok(endless.every((unit) => unit.length <= 320));
});

test("matching words are highlighted, but not words that fill the whole section", () => {
  const found = page.findPassage(INFOBOX, "متى ولد ابن خلدون", "1 رمضان 732 هـ، 27 مايو 1332 م [1]");
  assert.equal(found.matches("1332م"), true);
  assert.equal(found.matches("مايو"), true);
  assert.equal(found.matches("الحفصية"), false);
  assert.equal(found.matches("خلدون"), false); // appears in many lines, so it says little
});

// --- the source list ---------------------------------------------------------------------------------------

function longSource() {
  return { number: 1, heading: "ابن خلدون", text: `${INFOBOX}\n\n${"فقرة طويلة عن حياته وأعماله. ".repeat(20)}` };
}

const QUESTION = "متى ولد ابن خلدون";
const ANSWER = "ولد في 27 مايو 1332 م [1]";

test("a long source shows the matching passage first, with the rest one click away", () => {
  const list = page.buildSources([longSource()], { question: QUESTION, answer: ANSWER });
  const [, item] = list.details.children;
  const [label, passage, toggle, full] = item.children;

  assert.equal(label.textContent, "[1] ابن خلدون");
  assert.ok(passage.classList.contains("source-passage"));
  assert.match(passage.textContent, /الميلاد/);
  assert.ok(!passage.textContent.includes("الديانة"), "other lines are not shown");
  assert.ok(passage.textContent.startsWith("…"), "text before the passage is hinted with an ellipsis");
  assert.ok(passage.textContent.trimEnd().endsWith("…"));
  assert.ok(passage.find((element) => element.tag === "mark").length >= 2, "matching words are marked");
  assert.equal(full.hidden, true);
  assert.equal(toggle.textContent, "Show full section");
  assert.equal(toggle.attributes["aria-expanded"], "false");
  assert.equal(toggle.attributes["aria-controls"], full.id);
});

test("the toggle shows the whole section with the passage marked, and goes back", () => {
  const list = page.buildSources([longSource()], { question: QUESTION, answer: ANSWER });
  const [, item] = list.details.children;
  const [, passage, toggle, full] = item.children;

  toggle.click();
  assert.equal(full.hidden, false);
  assert.equal(passage.hidden, true);
  assert.equal(toggle.textContent, "Show less");
  assert.equal(toggle.attributes["aria-expanded"], "true");
  assert.ok(full.textContent.includes("الديانة"), "the whole section is there");
  const marked = full.find((element) => element.tag === "mark" && element.classList.contains("passage"));
  assert.equal(marked.length, 1);
  assert.match(marked[0].textContent, /^الميلاد/);

  toggle.click();
  assert.equal(full.hidden, true);
  assert.equal(passage.hidden, false);
  assert.equal(toggle.textContent, "Show full section");
});

test("a long source with no match shows its beginning and the toggle", () => {
  const list = page.buildSources([longSource()], { question: "Quantum chromodynamics?", answer: "Gluons [1]" });
  const [, item] = list.details.children;
  const [, passage, toggle] = item.children;
  assert.ok(passage.textContent.startsWith("ابن خلدون"));
  assert.ok(passage.textContent.length < 400);
  assert.ok(passage.textContent.trimEnd().endsWith("…"));
  assert.equal(toggle.tag, "button");
});

test("a short source is shown whole, without a toggle", () => {
  const list = page.buildSources([{ number: 2, heading: "", text: "A short **source**." }], { question: "q", answer: "a [2]" });
  const [, item] = list.details.children;
  assert.equal(item.children.length, 2);
  assert.equal(item.children[1].textContent, "A short source.");
  assert.equal(item.children[1].hidden, false);
});

test("an answer builds its sources from the question and the answer text", () => {
  const answer = page.buildAnswer(ANSWER, [longSource()], { question: QUESTION });
  const [, item] = answer.details.children;
  assert.match(item.children[1].textContent, /الميلاد/);
});

test("without a question the source still works and shows its beginning", () => {
  const list = page.buildSources([longSource()]);
  const [, item] = list.details.children;
  assert.equal(item.children[1].classList.contains("source-passage"), true);
});

test("a passage never contains markup: words are text, not HTML", () => {
  const text = `${"filler sentence for length. ".repeat(20)}\nThe <img src=x onerror=alert(1)> payload is here.\n${"more filler text. ".repeat(10)}`;
  const list = page.buildSources([{ number: 1, heading: "", text }], { question: "payload", answer: "payload [1]" });
  const [, item] = list.details.children;
  const html = item.children[1].toHTML();
  assert.ok(!html.includes("<img"));
  assert.ok(html.includes("&lt;img"));
});

test("FakeElement find is available for the tests above", () => {
  assert.equal(typeof new FakeElement("div").find, "function");
});

// --- running text, code documentation and escapes ---------------------------------------------------------------

test("plainText turns Markdown escapes into the characters they stand for", () => {
  assert.equal(page.plainText(String.raw`Path.read\_text and \*star\* and a\[1\]`), "Path.read_text and *star* and a[1]");
});

test("splitting into sentences does not cut code, decimals or Chinese text", () => {
  const code = `Call Path('').exists() now and then. Pi is 3.14 exactly, near e.g. this. `.repeat(8);
  const units = page.splitUnits(code);
  assert.ok(units.some((unit) => unit.includes("Path('').exists()")), "code with dots stays whole");
  assert.ok(units.some((unit) => unit.includes("3.14")), "decimals stay whole");
  assert.ok(units.every((unit) => !/^exists\(/.test(unit)));

  const chinese = page.splitUnits("这是一个很长的句子关于历史和文化。".repeat(30));
  assert.ok(chinese.length >= 30, "Chinese full stops split sentences without needing a space");
});

test("codeNames lists the names in backticks in the order they appear", () => {
  const names = [...page.codeNames("Use `Path.read_text()` like `open()` and `str`, not `x`.")];
  assert.deepEqual(names, ["read_text", "open", "str"]);
  assert.deepEqual([...page.codeNames("No code here, just [1] words.")], []);
});

const DOCS = [
  "Path.open(mode='r', buffering=-1, encoding=None)¶",
  "- Open the file pointed to by the path, like the built-in open() function does:",
  ">>> p = Path('setup.py') >>> with p.open() as f: ... f.readline()",
  "Path.read_text(encoding=None, errors=None, newline=None)¶",
  "- Return the decoded contents of the pointed-to file as a string.",
  "Added in version 3.5.",
  "Path.exists(*, follow_symlinks=True)¶",
  "- Return True if the path points to an existing file or directory.",
  "This method normally follows symlinks; to check if a symlink exists, add the argument follow_symlinks=False.",
  ">>> Path('').exists() # The current directory.",
  "Path.mkdir(mode=0o777, parents=False, exist_ok=False)¶",
  "- Create a new directory at this given path.",
  "Path.rmdir()¶",
  "- Remove this directory. The directory must be empty.",
  "Path.unlink(missing_ok=False)¶",
  "- Remove this file or symbolic link.",
].join("\n");

test("code in the answer leads to the definition of that name and the description after it", () => {
  const found = page.findPassage(DOCS, "How do I read the text of a file?", "Use `Path.read_text()`, which works like `open()` [1]");
  assert.match(found.unit, /^Path\.read_text\(/);
  assert.ok(found.unit.includes("Return the decoded contents"), "the description after the signature comes with it");
  assert.ok(!found.unit.includes("Path.open("), "the other method the answer mentions in passing is not chosen");
});

test("the definition is preferred over an example that calls the name", () => {
  const found = page.findPassage(DOCS, "How do I check that a path exists?", "Use `Path.exists()` [1]");
  assert.match(found.unit, /^Path\.exists\(/);
  assert.ok(found.unit.includes("Return True if the path points"));
});

test("if the named code is not in the text, matching falls back to the words", () => {
  const found = page.findPassage(DOCS, "How do I create a new directory?", "Use `Path.makedirs_everything()` to create it [1]");
  assert.match(found.unit, /Create a new directory/);
});

test("in running text a passage is the best sentence with its neighbours, up to a limit", () => {
  const sentences = Array.from({ length: 8 }, (_, i) => `Sentence number ${i} talks about topic${i} and adds several filler words to make it long enough.`);
  sentences[4] = "The quarterly revenue of the company reached nine hundred million dollars in total last year.";
  const found = page.findPassage(sentences.join(" "), "What was the quarterly revenue?", "Nine hundred million dollars [1]");
  assert.ok(found.unit.includes("quarterly revenue"));
  assert.ok(found.unit.includes("topic3") && found.unit.includes("topic5"), "neighbouring sentences give context");
  assert.ok(found.unit.length <= 520, "but the passage stays short");
  assert.ok(found.first > 0 && found.last < found.units.length - 1);
});

test("info box lines get no neighbours, because each line is a separate fact", () => {
  const found = page.findPassage(INFOBOX, "ما ديانته", "إسلامه [1]");
  assert.equal(found.first, found.last);
});
