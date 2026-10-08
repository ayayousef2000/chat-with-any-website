"use strict";

// Reads the real stylesheet and checks WCAG contrast for every text and background pair, in both themes.
// Translucent surfaces are blended over the page background, also at the strongest point of each aurora glow.

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const css = fs.readFileSync(path.join(__dirname, "../../app/static/styles.css"), "utf8");

function variables(block) {
  const found = {};
  for (const match of block.matchAll(/--([\w-]+):\s*([^;]+);/g)) found[match[1]] = match[2].trim();
  return found;
}

// Splits "a, rgba(1, 2, 3, 0.5)" at the commas that are not inside parentheses.
function splitArguments(text) {
  const parts = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i += 1) {
    if (text[i] === "(") depth += 1;
    else if (text[i] === ")") depth -= 1;
    else if (text[i] === "," && depth === 0) {
      parts.push(text.slice(start, i).trim());
      start = i + 1;
    }
  }
  parts.push(text.slice(start).trim());
  return parts;
}

// Every color is written once as light-dark(light value, dark value); the fonts are plain values.
function pairs() {
  const root = variables(css.match(/:root\s*\{([\s\S]*?)\n\}/)[1]);
  const found = {};
  const plain = [];
  for (const [name, value] of Object.entries(root)) {
    const pair = value.match(/^light-dark\(([\s\S]*)\)$/);
    if (pair) found[name] = splitArguments(pair[1]);
    else plain.push(name);
  }
  return { found, plain };
}

function themes() {
  const { found } = pairs();
  const light = {};
  const dark = {};
  for (const [name, [lightValue, darkValue]] of Object.entries(found)) {
    light[name] = lightValue;
    dark[name] = darkValue;
  }
  return { light, dark };
}

function parseColor(value) {
  const hex = value.match(/^#([0-9a-f]{6})$/i);
  if (hex) {
    const n = parseInt(hex[1], 16);
    return { rgb: [(n >> 16) & 255, (n >> 8) & 255, n & 255], alpha: 1 };
  }
  const rgba = value.match(/^rgba?\(\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\s*\)$/);
  if (rgba) return { rgb: [Number(rgba[1]), Number(rgba[2]), Number(rgba[3])], alpha: rgba[4] === undefined ? 1 : Number(rgba[4]) };
  throw new Error(`Unsupported color: ${value}`);
}

function over(top, below) {
  const t = parseColor(top);
  const b = Array.isArray(below) ? below : parseColor(below).rgb;
  return t.rgb.map((channel, i) => Math.round(channel * t.alpha + b[i] * (1 - t.alpha)));
}

function luminance(rgb) {
  const [r, g, b] = rgb.map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(a, b) {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi + 0.05) / (lo + 0.05);
}

const solid = (value) => parseColor(value).rgb;

for (const [themeName, v] of Object.entries(themes())) {
  // Page backgrounds: calm, and under the strongest part of each glow.
  const pages = {
    calm: solid(v.bg),
    "under glow 1": over(v.glow1, v.bg),
    "under glow 2": over(v.glow2, v.bg),
  };

  for (const [place, page] of Object.entries(pages)) {
    const surface = over(v.surface, page);
    const chip = over(v["cite-bg"], surface);
    const checks = [
      ["body text on the page", solid(v.text), page, 4.5],
      ["body text on panels", solid(v.text), surface, 4.5],
      ["muted text on the page", solid(v.muted), page, 4.5],
      ["muted text and placeholders on panels", solid(v.muted), surface, 4.5],
      ["links and source labels on panels", solid(v.cite), surface, 4.5],
      ["citation text on its chip", solid(v.cite), chip, 4.5],
      ["error text on the page", solid(v.error), page, 4.5],
      ["error text on panels", solid(v.error), surface, 4.5],
      ["highlighted words in sources", solid(v.text), over(v.flash, surface), 4.5],
      // The title is large text, which only needs 3:1, and fades from the text color through emerald to cyan.
      ["title, start of the gradient", solid(v.text), page, 3],
      ["title, middle of the gradient", solid(v.cite), page, 3],
      ["title, end of the gradient", solid(v.g2), page, 3],
    ];
    for (const [label, foreground, background, minimum] of checks) {
      test(`${themeName} theme, ${place}: ${label} reaches ${minimum}:1`, () => {
        const ratio = contrast(foreground, background);
        assert.ok(ratio >= minimum, `${label} is only ${ratio.toFixed(2)}:1`);
      });
    }
  }

  test(`${themeName} theme: the delete button text reaches 4.5:1`, () => {
    const ratio = contrast(solid(v["danger-text"]), solid(v.danger));
    assert.ok(ratio >= 4.5, `delete button text is only ${ratio.toFixed(2)}:1`);
  });

  for (const stop of ["g1", "g2"]) {
    test(`${themeName} theme: button text on gradient stop ${stop} reaches 4.5:1`, () => {
      const ratio = contrast(solid(v["btn-text"]), solid(v[stop]));
      assert.ok(ratio >= 4.5, `button text is only ${ratio.toFixed(2)}:1`);
    });
  }
}

test("the page bar names the active page with a label in the muted color, which the checks above cover", () => {
  const html = fs.readFileSync(path.join(__dirname, "../../app/static/index.html"), "utf8");
  assert.match(html, /<span class="page-label">Chatting about<\/span>/);
  assert.match(css, /\.page-label \{[^}]*color: var\(--muted\);/);
});

test("every color is one light-dark pair with a light and a dark value, and only the fonts are plain", () => {
  const { found, plain } = pairs();
  assert.ok(Object.keys(found).length >= 20);
  for (const [name, values] of Object.entries(found)) {
    assert.equal(values.length, 2, `--${name} must have exactly a light and a dark value`);
    for (const value of values) assert.doesNotThrow(() => parseColor(value), `--${name}: ${value}`);
  }
  assert.deepEqual(plain.sort(), ["mono", "sans", "serif"]);
});

test("the switch can force a theme, and the palette is not written a second time for the device setting", () => {
  assert.match(css, /:root\[data-theme="light"\]\s*\{\s*color-scheme: light;\s*\}/);
  assert.match(css, /:root\[data-theme="dark"\]\s*\{\s*color-scheme: dark;\s*\}/);
  assert.match(css, /:root\s*\{\s*color-scheme: light dark;/);
  assert.doesNotMatch(css, /@media \(prefers-color-scheme/);
});
