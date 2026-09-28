const assert = require('node:assert/strict');
const test = require('node:test');
const { soloJson, manyCsv, download } = require('../export.js');

test('solo JSON preserves the full API payload', () => {
  const payload = { status: 'matched', ranked: [{ gallery_id: 'car-1', confidence: 0.731 }], accepted: [] };
  assert.deepEqual(JSON.parse(soloJson(payload)), payload);
});

test('many CSV escapes filenames and distinguishes ranked from accepted', () => {
  const csv = manyCsv([
    { item: { name: 'a,"car.png' }, payload: {
      status: 'matched', ranked: [{ gallery_id: 'g1', confidence: 0.7 }, { gallery_id: 'g2', confidence: 0.6 }],
      accepted: [{ gallery_id: 'g1', confidence: 0.7 }],
    } },
    { item: { name: 'other.png' }, error: new Error('failed') },
  ]);
  assert.match(csv, /^"query_file","status","rank","gallery_id","confidence_cosine","accepted"/);
  assert.match(csv, /"a,""car.png","matched","1","g1","0.7","true"/);
  assert.match(csv, /"a,""car.png","matched","2","g2","0.6","false"/);
  assert.match(csv, /"other.png","error","","","",""/);
});

test('download creates a named anchor and clicks it within the event call', () => {
  const oldDocument = global.document;
  const oldUrl = global.URL;
  const oldTimeout = global.setTimeout;
  let clicked = false;
  let removed = false;
  let appended = false;
  const link = { style: {}, click() { clicked = true; }, remove() { removed = true; } };
  global.document = { createElement: () => link, body: { appendChild() { appended = true; } } };
  global.URL = { createObjectURL: () => 'blob:fixture', revokeObjectURL: () => {} };
  global.setTimeout = () => 0;
  try {
    download('result.json', '{}', 'application/json');
    assert.equal(link.download, 'result.json');
    assert.equal(link.href, 'blob:fixture');
    assert.equal(appended && clicked && removed, true);
  } finally {
    global.document = oldDocument;
    global.URL = oldUrl;
    global.setTimeout = oldTimeout;
  }
});
