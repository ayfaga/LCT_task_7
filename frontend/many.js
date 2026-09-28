const imageInput = document.getElementById('manyImages');
const csvInput = document.getElementById('manyCsv');
const imageSummary = document.getElementById('manyImagesSummary');
const tablePreview = document.getElementById('coordTablePreview');
const galleryStatus = document.getElementById('manyGalleryStatus');
const statusBox = document.getElementById('archiveStatus');
const submitBtn = document.getElementById('submitManyBtn');
const resultsPanel = document.getElementById('manyResults');
const resultsGrid = document.getElementById('manyResultsGrid');
let rows = [], latestResults = [], activePreviewUrls = [];
let commonGallery = null;
const escapeHtml = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
const stem = (value) => String(value).replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '').toLowerCase();

function setStatus(message, kind = 'info') { statusBox.textContent = message; statusBox.className = `upload-status ${kind}`; }

async function loadCommonGallery() {
  const response = await fetch('/api/common-gallery');
  if (!response.ok) throw new Error('Не удалось проверить общую галерею.');
  commonGallery = await response.json();
  galleryStatus.textContent = commonGallery.search_ready
    ? `● Общая галерея готова · ${commonGallery.image_count} автомобилей`
    : `○ В общей галерее ${commonGallery.image_count} автомобилей · нужно минимум ${commonGallery.minimum_for_search || 10}`;
}
const gallerySelectionReady = loadCommonGallery().catch((error) => { galleryStatus.textContent = error.message; });
window.addEventListener('focus', () => { loadCommonGallery().catch((error) => { galleryStatus.textContent = error.message; }); });

function splitCsv(text, delimiter) {
  const table = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') { if (quoted && text[i + 1] === '"') { cell += '"'; i += 1; } else quoted = !quoted; }
    else if (!quoted && char === delimiter) { row.push(cell.trim()); cell = ''; }
    else if (!quoted && (char === '\n' || char === '\r')) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(cell.trim()); cell = '';
      if (row.some(Boolean)) table.push(row);
      row = [];
    } else cell += char;
  }
  if (quoted) throw new Error('В CSV не закрыта кавычка.');
  row.push(cell.trim()); if (row.some(Boolean)) table.push(row);
  return table;
}

function parseBboxCsv(content) {
  const text = String(content).replace(/^\uFEFF/, '');
  const header = text.split(/\r?\n/, 1)[0];
  const delimiter = header.includes(';') ? ';' : header.includes('\t') ? '\t' : ',';
  const table = splitCsv(text, delimiter);
  if (table.length < 2) throw new Error('CSV пуст или не содержит строк с BBox.');
  const headers = table.shift().map((column) => column.toLowerCase());
  const required = ['image_id', 'x', 'y', 'w', 'h'];
  if (required.some((name) => !headers.includes(name))) throw new Error(`Нужны колонки: ${required.join(', ')}.`);
  const parsed = table.map((cells, index) => {
    if (cells.length !== headers.length) throw new Error(`Строка ${index + 2}: число колонок не совпадает с заголовком.`);
    const record = Object.fromEntries(headers.map((name, i) => [name, cells[i]]));
    if (!record.image_id) throw new Error(`Строка ${index + 2}: image_id пустой.`);
    const box = {};
    for (const key of ['x', 'y', 'w', 'h']) {
      if (!/^-?\d+$/.test(record[key])) throw new Error(`Строка ${index + 2}: ${key} должен быть целым числом.`);
      box[key] = Number(record[key]);
    }
    if (box.x < 0 || box.y < 0 || box.w <= 0 || box.h <= 0) throw new Error(`Строка ${index + 2}: BBox должен быть положительным и находиться в кадре.`);
    return { image_id: record.image_id, ...box };
  });
  const names = parsed.map((row) => row.image_id.toLowerCase());
  if (new Set(names).size !== names.length) throw new Error('В CSV повторяется image_id.');
  return parsed;
}

function matchRows(images, table) {
  const matched = new Map(), used = new Set();
  for (const file of images) {
    const name = file.name.toLowerCase();
    const exact = table.filter((row) => row.image_id.toLowerCase() === name);
    const candidates = exact.length ? exact : table.filter((row) => stem(row.image_id) === stem(name));
    if (candidates.length !== 1) throw new Error(`Для «${file.name}» нужна ровно одна строка CSV; найдено ${candidates.length}.`);
    if (used.has(candidates[0])) throw new Error('Разные файлы сопоставились одной строке CSV. Уточните image_id.');
    used.add(candidates[0]); matched.set(file, candidates[0]);
  }
  if (used.size !== table.length) throw new Error(`В CSV есть ${table.length - used.size} строк без соответствующих изображений.`);
  return matched;
}

function imageDimensions(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error('Не удалось открыть изображение.'));
    image.src = url;
  });
}

function renderResults(results) {
  latestResults = results;
  resultsGrid.replaceChildren();
  for (const { item, payload, error } of results) {
    const accepted = Array.isArray(payload?.accepted) ? payload.accepted : [];
    const top = payload?.status === 'matched' ? accepted[0] : null;
    const ranked = Array.isArray(payload?.ranked) ? payload.ranked.slice(0, 10) : [];
    const card = document.createElement('article'); card.className = 'many-result-card';
    const image = document.createElement('img'); image.src = item.preview; image.alt = item.name; card.appendChild(image);
    const caption = document.createElement('div'); caption.className = 'many-result-caption';
    caption.innerHTML = `<strong>${error ? 'Ошибка обработки' : top ? escapeHtml(top.gallery_id) : 'Нет уверенного совпадения'}</strong><span>${top ? `cosine ${Number(top.confidence).toFixed(3)}` : ''}</span>`;
    card.appendChild(caption);
    if (error) { const note = document.createElement('p'); note.className = 'result-error'; note.textContent = error.message; card.appendChild(note); }
    if (ranked.length) {
      const details = document.createElement('details');
      details.innerHTML = `<summary>Top‑10</summary><ol>${ranked.map((item) => `<li>${escapeHtml(item.gallery_id)} · cosine ${Number(item.confidence).toFixed(3)}</li>`).join('')}</ol>`;
      card.appendChild(details);
    }
    resultsGrid.appendChild(card);
  }
  resultsPanel.classList.remove('hidden'); resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

imageInput.addEventListener('change', () => {
  const files = [...imageInput.files];
  imageSummary.textContent = files.length ? `${files.length} изображений: ${files.slice(0, 3).map((file) => file.name).join(', ')}${files.length > 3 ? '…' : ''}` : 'Изображения не выбраны.';
});
csvInput.addEventListener('change', async () => {
  rows = []; tablePreview.classList.add('hidden');
  const file = csvInput.files[0]; if (!file) return;
  try {
    if (file.size > 1024 * 1024) throw new Error('CSV больше 1 МиБ.');
    rows = parseBboxCsv(await file.text());
    tablePreview.innerHTML = `<strong>${escapeHtml(file.name)}</strong> · ${rows.length} строк<div class="bbox-table-scroll"><table><thead><tr><th>image_id</th><th>x</th><th>y</th><th>w</th><th>h</th></tr></thead><tbody>${rows.slice(0, 5).map((row) => `<tr><td>${escapeHtml(row.image_id)}</td><td>${row.x}</td><td>${row.y}</td><td>${row.w}</td><td>${row.h}</td></tr>`).join('')}</tbody></table></div>`;
    tablePreview.classList.remove('hidden'); setStatus(`CSV прочитан: ${rows.length} строк.`, 'success');
  } catch (error) { setStatus(error.message, 'error'); }
});
document.getElementById('exportManyBtn').addEventListener('click', () => ReidExport.download('reid-results.csv', ReidExport.manyCsv(latestResults), 'text/csv;charset=utf-8'));

submitBtn.addEventListener('click', async () => {
  await gallerySelectionReady;
  try { await loadCommonGallery(); }
  catch (error) { setStatus(error.message, 'error'); return; }
  if (!commonGallery?.search_ready) { setStatus('Общая галерея ещё не готова. Сначала добавьте минимум 10 автомобилей.', 'error'); return; }
  const files = [...imageInput.files];
  if (!files.length || !rows.length) { setStatus('Нужны и фотографии, и BBox CSV.', 'error'); return; }
  let matched;
  try {
    if (files.some((file) => !/\.(jpe?g|png)$/i.test(file.name) || file.size > 20 * 1024 * 1024)) throw new Error('Каждый файл должен быть JPG/PNG до 20 МиБ.');
    matched = matchRows(files, rows);
  } catch (error) { setStatus(error.message, 'error'); return; }
  submitBtn.disabled = true; resultsPanel.classList.add('hidden');
  activePreviewUrls.forEach((url) => URL.revokeObjectURL(url)); activePreviewUrls = [];
  const results = [];
  try {
    for (const [index, file] of files.entries()) {
      setStatus(`Обрабатываем ${index + 1} из ${files.length}: ${file.name}`);
      const preview = URL.createObjectURL(file); activePreviewUrls.push(preview);
      const item = { name: file.name, preview };
      try {
        const size = await imageDimensions(preview), bbox = matched.get(file);
        if (bbox.x + bbox.w > size.width || bbox.y + bbox.h > size.height) throw new Error(`BBox выходит за пределы кадра ${size.width}×${size.height}.`);
        const body = new FormData(); body.append('image', file, file.name);
        for (const key of ['x', 'y', 'w', 'h']) body.append(key, String(bbox[key]));
        body.append('topk', '10'); body.append('gallery_id', commonGallery.gallery_id);
        const response = await fetch('/api/infer', { method: 'POST', body });
        const payload = await response.json();
        if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : `HTTP ${response.status}`);
        results.push({ item, payload });
      } catch (error) { results.push({ item, error }); }
    }
    renderResults(results);
    const failed = results.filter((item) => item.error).length;
    setStatus(`Готово: ${results.length - failed} обработано, ${failed} ошибок.`, failed ? 'error' : 'success');
  } finally { submitBtn.disabled = false; }
});
