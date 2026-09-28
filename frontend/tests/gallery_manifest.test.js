const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const fakeElement = () => ({ addEventListener() {}, replaceChildren() {}, classList: { add() {}, remove() {} } });
const context = {
  document: { getElementById: fakeElement },
  fetch: () => Promise.reject(new Error('offline test')),
  Option: function Option() {},
  File: class File { constructor(chunks, name) { this.content = chunks.join(''); this.name = name; } },
};
const source = fs.readFileSync(path.join(__dirname, '..', 'gallery.js'), 'utf8');
vm.runInNewContext(`${source}\nglobalThis.testApi = { normalizeManifest };`, context);

test('organizer image_id CSV becomes backend filename manifest', async () => {
  const file = { text: async () => 'image_id,x,y,w,h\ncar01,10,20,30,40\n' };
  const result = await context.testApi.normalizeManifest(file, [{ name: 'car01.jpg' }], null);
  assert.equal(result.name, 'manifest.csv');
  assert.match(result.content, /"car01.jpg","car01","10","20","30","40"/);
});

test('organizer CSV refuses missing images and ZIP without filenames', async () => {
  const file = { text: async () => 'image_id,x,y,w,h\nother,10,20,30,40\n' };
  await assert.rejects(() => context.testApi.normalizeManifest(file, [{ name: 'car01.jpg' }], null), /сопоставить/);
  await assert.rejects(() => context.testApi.normalizeManifest(file, [], { name: 'data.zip' }), /Для ZIP/);
});
