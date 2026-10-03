"use strict";

// A tiny stand-in for the browser DOM, just enough to run the page script's rendering helpers.

class FakeText {
  constructor(text) {
    this.text = text;
  }

  toHTML() {
    return String(this.text).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
}

class FakeElement {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attributes = {};
    this.listeners = {};
    this.classes = new Set();
    this.open = false;
    this.hidden = false;
    this.scrolled = false;
    this.href = undefined;
    this.target = undefined;
    this.rel = undefined;
    this.type = undefined;
  }

  set className(value) {
    this.classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  get className() {
    return [...this.classes].join(" ");
  }

  get classList() {
    return {
      add: (name) => this.classes.add(name),
      remove: (name) => this.classes.delete(name),
      contains: (name) => this.classes.has(name),
    };
  }

  set textContent(value) {
    this.children = [new FakeText(value)];
  }

  get textContent() {
    return this.children.map((child) => (child instanceof FakeText ? child.text : child.textContent)).join("");
  }

  append(...nodes) {
    for (const node of nodes) this.children.push(typeof node === "string" ? new FakeText(node) : node);
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  addEventListener(type, listener) {
    this.listeners[type] = listener;
  }

  click() {
    this.listeners.click({ preventDefault() {} });
  }

  scrollIntoView() {
    this.scrolled = true;
  }

  find(predicate, found = []) {
    for (const child of this.children) {
      if (child instanceof FakeElement) {
        if (predicate(child)) found.push(child);
        child.find(predicate, found);
      }
    }
    return found;
  }

  toHTML() {
    const attrs = [
      this.classes.size ? ` class="${this.className}"` : "",
      this.href ? ` href="${this.href}"` : "",
      this.rel ? ` rel="${this.rel}"` : "",
    ].join("");
    return `<${this.tag}${attrs}>${this.children.map((child) => child.toHTML()).join("")}</${this.tag}>`;
  }
}

function installFakeDom() {
  globalThis.document = { createElement: (tag) => new FakeElement(tag) };
  globalThis.setTimeout = () => 0;
}

module.exports = { FakeElement, installFakeDom };
