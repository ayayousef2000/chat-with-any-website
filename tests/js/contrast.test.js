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

function themes() {
  const light = css.match(/:root\s*\{([\s\S]*?)\n\}/)[1];
  const dark = css.match(/@media \(prefers-color-scheme: dark\)\s*\{\s*:root\s*\{([\s\S]*?)\n\s*\}\s*\}/)[1];
  return { light: { ...variables(light) }, dark: { ...variables(light), ...variables(dark) } };
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

test("the stylesheet defines both themes with the same set of colors", () => {
  const { light, dark } = themes();
  assert.ok(Object.keys(light).length >= 20);
  assert.deepEqual(Object.keys(dark).sort(), Object.keys(light).sort());
});
