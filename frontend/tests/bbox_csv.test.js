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
  window: { addEventListener() {} },
  fetch: () => Promise.reject(new Error('offline test')),
  Option: function Option() {},
};
const source = fs.readFileSync(path.join(__dirname, '..', 'many.js'), 'utf8');
vm.runInNewContext(`${source}\nglobalThis.testApi = { parseBboxCsv, matchRows, selectedImages };`, context);
const { parseBboxCsv, matchRows } = context.testApi;

test('reads organizer-style CSV with BOM, quoted name and integer bbox', () => {
  const rows = parseBboxCsv('\uFEFFimage_id,x,y,w,h\r\n"car,01.jpg",1,2,30,40\r\n');
  assert.equal(rows.length, 1);
  assert.equal(rows[0].image_id, 'car,01.jpg');
  assert.equal(rows[0].w, 30);
});

test('rejects missing, fractional, empty and duplicate bbox fields', () => {
  assert.throws(() => parseBboxCsv('image_id,x,y,w,h\na.jpg,0,0,,10'), /целым числом/);
  assert.throws(() => parseBboxCsv('image_id,x,y,w,h\na.jpg,0,0,1.5,10'), /целым числом/);
  assert.throws(() => parseBboxCsv('image_id,x,y,w,h\na.jpg,0,0,10,10\na.jpg,1,1,10,10'), /повторяется/);
});

test('matches selected images while ignoring unrelated full-CSV rows', () => {
  const images = [{ name: 'car1.jpg' }, { name: 'car2.png' }];
  const rows = parseBboxCsv('image_id,x,y,w,h\nother,0,0,1,1\ncar1,0,0,10,10\ncar2.png,2,3,20,30');
  assert.equal(matchRows(images, rows).size, 2);
  assert.equal(matchRows(images.slice(0, 1), rows).size, 1);
  assert.throws(() => matchRows(images, rows.slice(0, 1)), /ровно одна строка/);
});

test('folder path takes priority and ambiguous basenames are rejected', () => {
  const rows = parseBboxCsv('filename,x,y,w,h\nfolder/car.jpg,0,0,10,10\nother/car.jpg,1,1,10,10');
  assert.equal(matchRows([{ name: 'car.jpg', webkitRelativePath: 'folder/car.jpg' }], rows).size, 1);
  assert.equal(matchRows([{ name: 'car.jpg', webkitRelativePath: 'upload/folder/car.jpg' }], rows).size, 1);
  assert.throws(() => matchRows([{ name: 'car.jpg' }], rows), /ровно одна строка/);
});

test('ZIP selection is distinct from folder and individual files', () => {
  context.document.getElementById('manyArchive').files = [{ name: 'queries.zip' }];
  assert.equal(context.testApi.selectedImages().mode, 'zip');
});
