const imageInput = document.getElementById('manyImages');
const folderInput = document.getElementById('manyFolder');
const archiveInput = document.getElementById('manyArchive');
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
const normalized = (value) => String(value).replace(/\\/g, '/').replace(/^\.\//, '').toLowerCase();
const basename = (value) => normalized(value).split('/').pop();
const imageFiles = (input) => [...input.files].filter((file) => /\.(jpe?g|png)$/i.test(file.name));

function selectedImages() {
  if (archiveInput.files.length) return { mode: 'zip', archive: archiveInput.files[0], files: [] };
  if (folderInput.files.length) return { mode: 'folder', files: imageFiles(folderInput) };
  return { mode: 'files', files: [...imageInput.files] };
}

function updateImageSummary() {
  const selection = selectedImages();
  if (selection.archive) {
    imageSummary.textContent = `ZIP: ${selection.archive.name}. Кадры будут обработаны по одному.`;
    return;
  }
  const { files } = selection;
  imageSummary.textContent = files.length
    ? `${files.length} изображений${selection.mode === 'folder' ? ' из папки' : ''}: ${files.slice(0, 3).map((file) => file.name).join(', ')}${files.length > 3 ? '…' : ''}`
    : 'Изображения не выбраны.';
}

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
  const keyColumn = headers.includes('filename') ? 'filename' : 'image_id';
  const required = [keyColumn, 'x', 'y', 'w', 'h'];
  if (required.some((name) => !headers.includes(name))) throw new Error(`Нужны колонки: ${required.join(', ')}.`);
  const parsed = table.map((cells, index) => {
    if (cells.length !== headers.length) throw new Error(`Строка ${index + 2}: число колонок не совпадает с заголовком.`);
    const record = Object.fromEntries(headers.map((name, i) => [name, cells[i]]));
    if (!record[keyColumn]) throw new Error(`Строка ${index + 2}: ${keyColumn} пустой.`);
    const box = {};
    for (const key of ['x', 'y', 'w', 'h']) {
      if (!/^-?\d+$/.test(record[key])) throw new Error(`Строка ${index + 2}: ${key} должен быть целым числом.`);
      box[key] = Number(record[key]);
    }
    if (box.x < 0 || box.y < 0 || box.w <= 0 || box.h <= 0) throw new Error(`Строка ${index + 2}: BBox должен быть положительным и находиться в кадре.`);
    return { image_id: record[keyColumn], ...box };
  });
  const names = parsed.map((row) => row.image_id.toLowerCase());
  if (new Set(names).size !== names.length) throw new Error('В CSV повторяется image_id.');
  return parsed;
}

function matchRows(images, table) {
  const matched = new Map(), used = new Set();
  const indexes = [new Map(), new Map(), new Map()];
  for (const row of table) {
    [normalized(row.image_id), basename(row.image_id), stem(row.image_id)].forEach((key, index) => {
      if (!indexes[index].has(key)) indexes[index].set(key, []);
      indexes[index].get(key).push(row);
    });
  }
  for (const file of images) {
    const path = file.webkitRelativePath || file.name;
    const parts = normalized(path).split('/');
    const pathMatch = parts.slice(1, -1).map((_, index) => indexes[0].get(parts.slice(index + 1).join('/')))
      .find((entries) => entries?.length);
    const candidates = indexes[0].get(normalized(path))
      || pathMatch
      || indexes[0].get(normalized(file.name))
      || indexes[1].get(basename(file.name))
      || indexes[2].get(stem(file.name)) || [];
    if (candidates.length !== 1) throw new Error(`Для «${file.name}» нужна ровно одна строка CSV; найдено ${candidates.length}.`);
    if (used.has(candidates[0])) throw new Error('Разные файлы сопоставились одной строке CSV. Уточните image_id.');
    used.add(candidates[0]); matched.set(file, candidates[0]);
  }
  return matched;
}

function renderResults(results) {
  latestResults = results;
  resultsGrid.replaceChildren();
  for (const { item, payload, error } of results) {
    const accepted = Array.isArray(payload?.accepted) ? payload.accepted : [];
    const top = payload?.status === 'matched' ? accepted[0] : null;
    const ranked = Array.isArray(payload?.ranked) ? payload.ranked.slice(0, 10) : [];
    const card = document.createElement('article'); card.className = 'many-result-card';
    const overview = document.createElement('div'); overview.className = 'many-result-overview';
    if (item.preview) {
      const image = document.createElement('img'); image.src = item.preview; image.alt = item.name; overview.appendChild(image);
    } else {
      const placeholder = document.createElement('div'); placeholder.className = 'many-result-placeholder';
      placeholder.textContent = 'Кадр из ZIP'; overview.appendChild(placeholder);
    }
    const caption = document.createElement('div'); caption.className = 'many-result-caption';
    const ranking = payload?.ranking_algorithm === 'transductive_aqe_k5_alpha025'
      ? `AQE по пакету из ${Number(payload.query_cohort_size) || 0} запросов · ` : '';
    caption.innerHTML = `<small>Запрос: ${escapeHtml(item.name)}</small><strong>${error ? 'Ошибка обработки' : top ? `Принят ID ${escapeHtml(top.gallery_id)}` : 'Нет уверенного совпадения'}</strong><span>${ranking}${top ? `cosine ${Number(top.confidence).toFixed(3)} · не вероятность` : 'Ближайшие кандидаты — ниже; cosine не вероятность'}</span>`;
    overview.appendChild(caption); card.appendChild(overview);
    if (error) { const note = document.createElement('p'); note.className = 'result-error'; note.textContent = error.message; card.appendChild(note); }
    if (ranked.length) {
      const section = document.createElement('div'); section.className = 'many-candidate-section';
      const heading = document.createElement('h3'); heading.textContent = `Все кандидаты top‑${ranked.length}`;
      section.appendChild(heading);
      section.appendChild(CandidateGallery.render(ranked, accepted));
      card.appendChild(section);
    }
    resultsGrid.appendChild(card);
  }
  resultsPanel.classList.remove('hidden'); resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

imageInput.addEventListener('change', () => {
  if (imageInput.files.length) { folderInput.value = ''; archiveInput.value = ''; }
  updateImageSummary();
});
folderInput.addEventListener('change', () => {
  if (folderInput.files.length) { imageInput.value = ''; archiveInput.value = ''; }
  updateImageSummary();
});
archiveInput.addEventListener('change', () => {
  if (archiveInput.files.length) { imageInput.value = ''; folderInput.value = ''; }
  updateImageSummary();
});
csvInput.addEventListener('change', async () => {
  rows = []; tablePreview.classList.add('hidden');
  const file = csvInput.files[0]; if (!file) return;
  try {
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
  const selection = selectedImages();
  const files = selection.files;
  if ((!files.length && !selection.archive) || !rows.length) { setStatus('Нужны фотографии или ZIP и BBox CSV.', 'error'); return; }
  let matched;
  try {
    if (selection.archive && !/\.zip$/i.test(selection.archive.name)) throw new Error('Нужен ZIP-архив.');
    if (files.some((file) => !/\.(jpe?g|png)$/i.test(file.name))) throw new Error('Каждый файл должен быть JPG/PNG.');
    if (!selection.archive) matched = matchRows(files, rows);
  } catch (error) { setStatus(error.message, 'error'); return; }
  submitBtn.disabled = true; resultsPanel.classList.add('hidden');
  activePreviewUrls.forEach((url) => URL.revokeObjectURL(url)); activePreviewUrls = [];
  const results = [];
  try {
    if (selection.archive) {
      setStatus(`Обрабатываем ZIP: ${selection.archive.name}. Это может занять несколько минут.`);
      const body = new FormData(); body.append('archive', selection.archive);
      body.append('manifest', csvInput.files[0]);
      body.append('gallery_id', commonGallery.gallery_id); body.append('topk', '10');
      const response = await fetch('/api/infer-batch', { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
      for (const row of data.results) results.push({ item: { name: row.filename, preview: null },
        payload: row.payload, error: row.error ? new Error(row.error) : null });
    } else {
      setStatus(`Обрабатываем ${files.length} изображений совместно. Это может занять несколько минут.`);
      const body = new FormData();
      for (const file of files) body.append('images', file, file.webkitRelativePath || file.name);
      body.append('manifest', csvInput.files[0]);
      body.append('gallery_id', commonGallery.gallery_id); body.append('topk', '10');
      const response = await fetch('/api/infer-batch', { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`);
      if (!Array.isArray(data.results) || data.results.length !== files.length) throw new Error('Пакетный ответ неполный.');
      for (const [index, row] of data.results.entries()) {
        const preview = URL.createObjectURL(files[index]); activePreviewUrls.push(preview);
        results.push({ item: { name: row.filename, preview }, payload: row.payload,
          error: row.error ? new Error(row.error) : null });
      }
    }
    renderResults(results);
    const failed = results.filter((item) => item.error).length;
    setStatus(`Готово: ${results.length - failed} обработано, ${failed} ошибок.`, failed ? 'error' : 'success');
  } catch (error) { setStatus(error.message, 'error'); }
  finally { submitBtn.disabled = false; }
});
