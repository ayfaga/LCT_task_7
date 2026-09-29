const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const fakeElement = () => ({ files: [], addEventListener() {}, replaceChildren() {}, classList: { add() {}, remove() {} } });
const elements = new Map();
const context = {
  document: { getElementById: (id) => {
    if (!elements.has(id)) elements.set(id, fakeElement());
    return elements.get(id);
  } },
  fetch: () => Promise.reject(new Error('offline test')),
  Option: function Option() {},
  File: class File { constructor(chunks, name) { this.content = chunks.join(''); this.name = name; } },
};
const source = fs.readFileSync(path.join(__dirname, '..', 'gallery.js'), 'utf8');
vm.runInNewContext(`${source}\nglobalThis.testApi = { validateSelection };`, context);

test('gallery UI sends ZIP and full external CSV together for server-side matching', () => {
  elements.get('galleryArchive').files = [{ name: 'gallery_full_frames.zip', size: 6_000_000 }];
  elements.get('galleryManifest').files = [{ name: 'gallery_bbox.csv', size: 680 }];
  assert.equal(context.testApi.validateSelection().archive.name, 'gallery_full_frames.zip');
  elements.get('galleryArchive').files = [];
  elements.get('galleryManifest').files = [];
});
