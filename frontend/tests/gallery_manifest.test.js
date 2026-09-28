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
vm.runInNewContext(`${source}\nglobalThis.testApi = { normalizeManifest, validateSelection };`, context);

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

test('ZIP accepts a separate manifest with exact nested filenames', async () => {
  const file = { text: async () => 'filename,gallery_id,x,y,w,h\nimages/car01.jpg,1,0,0,10,10\n' };
  const result = await context.testApi.normalizeManifest(file, [], { name: 'frames.zip' });
  assert.equal(result, file);
});

test('gallery UI accepts ZIP and separate CSV in one upload', () => {
  elements.get('galleryArchive').files = [{ name: 'gallery_full_frames.zip', size: 6_000_000 }];
  elements.get('galleryManifest').files = [{ name: 'gallery_bbox.csv', size: 680 }];
  assert.equal(context.testApi.validateSelection().archive.name, 'gallery_full_frames.zip');
  elements.get('galleryArchive').files = [];
  elements.get('galleryManifest').files = [];
});
