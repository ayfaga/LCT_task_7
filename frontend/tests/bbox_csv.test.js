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
};
const source = fs.readFileSync(path.join(__dirname, '..', 'many.js'), 'utf8');
vm.runInNewContext(`${source}\nglobalThis.testApi = { parseBboxCsv, matchRows };`, context);
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

test('requires one unambiguous csv row for every image', () => {
  const images = [{ name: 'car1.jpg' }, { name: 'car2.png' }];
  const rows = parseBboxCsv('image_id,x,y,w,h\ncar1,0,0,10,10\ncar2.png,2,3,20,30');
  assert.equal(matchRows(images, rows).size, 2);
  assert.throws(() => matchRows(images, rows.slice(0, 1)), /ровно одна строка/);
  assert.throws(() => matchRows(images.slice(0, 1), rows), /без соответствующих/);
});
