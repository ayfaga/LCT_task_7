const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

class Element {
  constructor(tag) { this.tag = tag; this.children = []; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  setAttribute(name, value) { this[name] = value; }
  addEventListener() {}
}

const context = {
  URL,
  window: { location: { origin: 'http://localhost' } },
  document: { createElement: (tag) => new Element(tag) },
};
const source = fs.readFileSync(path.join(__dirname, '..', 'candidate_gallery.js'), 'utf8');
vm.runInNewContext(`${source}\nglobalThis.Candidates = CandidateGallery;`, context);

test('renders all top-10 images, IDs, cosine scores and accepted state', () => {
  const ranked = Array.from({ length: 10 }, (_, index) => ({
    gallery_id: `car-${index}`, similarity: 0.9 - index / 100,
    image_url: `/api/galleries/a/images/key-${index}/crop`,
  }));
  const grid = context.Candidates.render(ranked, [ranked[0]]);
  assert.equal(grid.children.length, 10);
  assert.equal(grid.children[0].className, 'candidate-card is-accepted');
  assert.equal(grid.children[0].children[0].children[0].src, ranked[0].image_url);
  assert.equal(grid.children[9].children[0].children[0].src, ranked[9].image_url);
  assert.equal(grid.children[0].children[1].children[1].textContent, 'car-0');
  assert.equal(grid.children[0].children[1].children[2].textContent, 'cosine 0.900');
});

test('does not render external image URLs as gallery thumbnails', () => {
  const grid = context.Candidates.render([{ gallery_id: 'bad', similarity: 0,
    image_url: 'https://example.com/image.jpg' }]);
  assert.equal(grid.children[0].children[0].className, 'candidate-image-missing');
});
