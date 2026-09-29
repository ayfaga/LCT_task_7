const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

test('jury export file controls show the chosen names and combine image sources', () => {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) {
      elements.set(id, {
        files: [], textContent: '', value: '', hidden: false,
        listeners: {}, addEventListener(name, callback) { this.listeners[name] = callback; },
      });
    }
    return elements.get(id);
  };
  const source = fs.readFileSync(path.join(__dirname, '..', 'judge_export.js'), 'utf8');
  vm.runInNewContext(source, {
    document: {
      getElementById: element,
      querySelector: () => ({ value: 'components' }),
      querySelectorAll: () => [],
    },
    fetch: () => Promise.reject(new Error('offline test')),
  });

  element('judgeGalleryCsv').files = [{ name: 'gallery.csv' }];
  element('judgeGalleryCsv').listeners.change();
  assert.equal(element('judgeGalleryCsvName').textContent, 'Выбран: gallery.csv');

  const id = 'a'.repeat(32);
  element('judgeGalleryFiles').files = [{ name: 'gallery.zip' }, { name: `${id}.jpg` }];
  element('judgeGalleryFiles').listeners.change();
  assert.match(element('judgeGallerySummary').textContent, /2 выбранных ZIP\/JPEG/);
  element('judgeGalleryFolder').files = [{ name: 'b'.repeat(32) + '.jpeg' }];
  element('judgeGalleryFolder').listeners.change();
  assert.match(element('judgeGallerySummary').textContent, /3 выбранных ZIP\/JPEG/);

  element('judgeArchive').files = [{ name: 'organizer.zip' }];
  element('judgeArchive').listeners.change();
  assert.equal(element('judgeArchiveName').textContent, 'Выбран: organizer.zip');
});
