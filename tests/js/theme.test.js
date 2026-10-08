"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const { FakeElement } = require("./fake-dom.js");
const theme = require("../../app/static/theme.js");

const html = fs.readFileSync(path.join(__dirname, "../../app/static/index.html"), "utf8");
const css = fs.readFileSync(path.join(__dirname, "../../app/static/styles.css"), "utf8");

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    data,
    getItem: (key) => (key in data ? data[key] : null),
    setItem: (key, value) => {
      data[key] = String(value);
    },
    removeItem: (key) => {
      delete data[key];
    },
  };
}

function brokenStorage() {
  const fail = () => {
    throw new Error("storage is blocked");
  };
  return { getItem: fail, setItem: fail, removeItem: fail };
}

function switchElements() {
  const container = new FakeElement("div");
  container.hidden = true;
  const buttons = ["system", "light", "dark"].map((choice) => {
    const button = new FakeElement("button");
    button.dataset = { choice };
    button.setAttribute("aria-pressed", String(choice === "system"));
    return button;
  });
  return { container, buttons };
}

test("only system, light and dark are accepted; anything else means system", () => {
  assert.deepEqual(theme.THEME_CHOICES, ["system", "light", "dark"]);
  for (const choice of theme.THEME_CHOICES) assert.equal(theme.normalizeChoice(choice), choice);
  for (const odd of [null, undefined, "", "blue", "DARK", 3]) assert.equal(theme.normalizeChoice(odd), "system");
});

test("the saved choice is read back, and nothing saved means system", () => {
  assert.equal(theme.loadChoice(memoryStorage({ [theme.THEME_KEY]: "dark" })), "dark");
  assert.equal(theme.loadChoice(memoryStorage()), "system");
  assert.equal(theme.loadChoice(memoryStorage({ [theme.THEME_KEY]: "nonsense" })), "system");
});

test("blocked or missing storage never breaks the page: it follows the device", () => {
  assert.equal(theme.loadChoice(brokenStorage()), "system");
  assert.equal(theme.loadChoice(null), "system");
  assert.doesNotThrow(() => theme.saveChoice(brokenStorage(), "dark"));
});

test("light and dark are saved, and system removes the saved value", () => {
  const storage = memoryStorage();
  theme.saveChoice(storage, "light");
  assert.equal(storage.data[theme.THEME_KEY], "light");
  theme.saveChoice(storage, "system");
  assert.equal(theme.THEME_KEY in storage.data, false);
});

test("the choice is set on the page as data-theme, and system takes it away", () => {
  const root = { dataset: {} };
  theme.applyChoice(root, "dark");
  assert.equal(root.dataset.theme, "dark");
  theme.applyChoice(root, "light");
  assert.equal(root.dataset.theme, "light");
  theme.applyChoice(root, "system");
  assert.equal("theme" in root.dataset, false);
});

test("the switch is shown, marks the saved choice, and each button changes, applies and remembers the theme", () => {
  const { container, buttons } = switchElements();
  const root = { dataset: {} };
  const storage = memoryStorage({ [theme.THEME_KEY]: "dark" });

  theme.wireSwitch(container, buttons, root, storage);

  assert.equal(container.hidden, false);
  assert.deepEqual(buttons.map((b) => b.attributes["aria-pressed"]), ["false", "false", "true"]);

  buttons[1].click();
  assert.equal(root.dataset.theme, "light");
  assert.equal(storage.data[theme.THEME_KEY], "light");
  assert.deepEqual(buttons.map((b) => b.attributes["aria-pressed"]), ["false", "true", "false"]);

  buttons[0].click();
  assert.equal("theme" in root.dataset, false);
  assert.equal(theme.THEME_KEY in storage.data, false);
  assert.deepEqual(buttons.map((b) => b.attributes["aria-pressed"]), ["true", "false", "false"]);
});

test("the switch still works when the storage is blocked", () => {
  const { container, buttons } = switchElements();
  const root = { dataset: {} };

  theme.wireSwitch(container, buttons, root, brokenStorage());
  buttons[2].click();

  assert.equal(root.dataset.theme, "dark");
  assert.equal(buttons[2].attributes["aria-pressed"], "true");
});

test("the page loads theme.js in its head, after the stylesheet, so the saved theme is applied before the first paint", () => {
  const head = html.match(/<head>([\s\S]*?)<\/head>/)[1];
  assert.match(head, /<script src="\/static\/theme\.js"><\/script>/);
  assert.ok(head.indexOf("styles.css") < head.indexOf("theme.js"));
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)[^>]*>/, "no inline script: the content security policy forbids it");
});

test("the switch in the page is a labelled group of three icon buttons that starts hidden", () => {
  assert.match(html, /<div id="theme-switch" class="theme-switch" role="group" aria-label="Color theme" hidden>/);
  const names = { system: "System theme", light: "Light theme", dark: "Dark theme" };
  for (const [choice, name] of Object.entries(names)) {
    // A button with only an icon needs a name for screen readers and a tooltip for everyone else.
    assert.match(
      html,
      new RegExp(`<button type="button" class="secondary" data-choice="${choice}" aria-pressed="(true|false)" aria-label="${name}" title="[^"]+">\\s*<svg viewBox="0 0 24 24" aria-hidden="true">`),
    );
  }
});

test("the switch sits in the top row of the header, beside the title and not under the subtitle", () => {
  const header = html.match(/<header>([\s\S]*?)<\/header>/)[1];
  const top = header.match(/<div class="header-top">([\s\S]*?)<p class="subtitle">/)[1];
  assert.match(top, /<h1>/);
  assert.match(top, /id="theme-switch"/);
  assert.match(css, /\.header-top \{[^}]*justify-content: space-between;/);
});

test("the icons are drawn with the text color and the pressed button is highlighted", () => {
  assert.match(css, /\.theme-switch svg \{[^}]*stroke: currentColor;/);
  assert.match(css, /\.theme-switch button\.secondary\[aria-pressed="true"\] \{[^}]*color: var\(--cite\);[^}]*background: var\(--cite-bg\);/);
});
