const imageList = document.getElementById('imageList');
const coordList = document.getElementById('coordList');
const coordTablePreview = document.getElementById('coordTablePreview');
const backHomeBtn = document.getElementById('backHomeBtn');
const renameModal = document.getElementById('renameModal');
const previewModal = document.getElementById('previewModal');
const renameInput = document.getElementById('renameInput');
const extensionTrigger = document.getElementById('extensionTrigger');
const previewContent = document.getElementById('previewContent');
const renameCloseBtn = document.getElementById('renameCloseBtn');
const cancelRenameBtn = document.getElementById('cancelRenameBtn');
const saveRenameBtn = document.getElementById('saveRenameBtn');
const submitManyBtn = document.getElementById('submitManyBtn');
const archiveUploadBtn = document.getElementById('archiveUploadBtn');
const archiveStatus = document.getElementById('archiveStatus');
const manyResults = document.getElementById('manyResults');
const manyResultsGrid = document.getElementById('manyResultsGrid');

const TABLE_COLUMNS = ['image_id', 'x', 'y', 'w', 'h', 'vehicle_id', 'camera_id'];
const state = { activeRename: null, fileStore: { image: [], coord: [] } };

const escapeHtml = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
const fileStem = (name) => String(name || '').replace(/\\/g, '/').split('/').pop().replace(/\.[^.]+$/, '').toLowerCase();

function setArchiveStatus(message, kind = 'info') {
  archiveStatus.className = `upload-status ${kind}`;
  archiveStatus.textContent = message;
}

function splitDelimitedRow(line, delimiter) {
  const values = []; let value = ''; let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') { value += '"'; i += 1; } else quoted = !quoted;
    } else if (char === delimiter && !quoted) { values.push(value.trim()); value = ''; } else value += char;
  }
  values.push(value.trim());
  return values;
}

function parseCoordinateTable(content) {
  const trimmed = String(content || '').replace(/^\uFEFF/, '').trim();
  if (!trimmed) throw new Error('Таблица пуста.');
  let rows;
  if (trimmed.startsWith('[') || trimmed.startsWith('{')) {
    const parsed = JSON.parse(trimmed);
    rows = Array.isArray(parsed) ? parsed : parsed.rows;
    if (!Array.isArray(rows)) throw new Error('В JSON должен быть массив строк или поле rows.');
  } else {
    const lines = trimmed.split(/\r?\n/).filter((line) => line.trim());
    const delimiter = lines[0].includes('\t') ? '\t' : lines[0].includes(';') ? ';' : ',';
    const headers = splitDelimitedRow(lines[0], delimiter).map((header) => header.trim());
    rows = lines.slice(1).map((line) => {
      const values = splitDelimitedRow(line, delimiter); const row = {};
      headers.forEach((header, index) => { row[header] = values[index] ?? ''; });
      return row;
    });
  }
  const columns = new Set(rows.length ? Object.keys(rows[0]) : []);
  const missing = TABLE_COLUMNS.filter((column) => !columns.has(column));
  if (missing.length) throw new Error(`Нужны колонки: ${TABLE_COLUMNS.join(', ')}.`);
  rows.forEach((row, index) => {
    ['x', 'y', 'w', 'h'].forEach((key) => {
      if (!Number.isFinite(Number(row[key]))) throw new Error(`Строка ${index + 2}: «${key}» должно быть числом.`);
    });
  });
  return rows.map((row) => Object.fromEntries(TABLE_COLUMNS.map((key) => [key, String(row[key] ?? '')])));
}

async function readTable(item) {
  if (item.rows) return item.rows;
  const content = item.file ? await item.file.text() : await fetch(item.url).then((response) => {
    if (!response.ok) throw new Error('Не удалось открыть таблицу.');
    return response.text();
  });
  item.rows = parseCoordinateTable(content);
  return item.rows;
}

function makeTableMarkup(name, rows) {
  const visibleRows = rows.slice(0, 5).map((row) => `<tr>${TABLE_COLUMNS.map((key) => `<td>${escapeHtml(row[key])}</td>`).join('')}</tr>`).join('');
  const more = rows.length > 5 ? `<p>Показано 5 из ${rows.length} строк.</p>` : '';
  return `<div class="bbox-table-title">${escapeHtml(name)}</div><div class="bbox-table-scroll"><table><thead><tr>${TABLE_COLUMNS.map((key) => `<th>${key}</th>`).join('')}</tr></thead><tbody>${visibleRows}</tbody></table></div>${more}`;
}

async function refreshTablePreview() {
  const item = state.fileStore.coord[0];
  if (!item) { coordTablePreview.classList.add('hidden'); coordTablePreview.innerHTML = ''; return; }
  coordTablePreview.classList.remove('hidden'); coordTablePreview.textContent = 'Читаем таблицу…';
  try { coordTablePreview.innerHTML = makeTableMarkup(item.name, await readTable(item)); }
  catch (error) { coordTablePreview.textContent = `Ошибка таблицы: ${error.message}`; }
}

async function loadPersistedFiles() {
  try {
    const response = await fetch('/api/replenishment/files');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const files = await response.json();
    state.fileStore.image = files.filter((item) => item.kind === 'image').map((item) => ({ id: item.id, name: item.name, kind: 'image', preview: item.url }));
    state.fileStore.coord = files.filter((item) => item.kind === 'text').map((item) => ({ id: item.id, name: item.name, kind: 'coord', url: item.url }));
    renderFileList('image'); renderFileList('coord'); refreshTablePreview();
  } catch (error) { setArchiveStatus(`Не удалось загрузить файлы: ${error.message}`, 'error'); }
}

function renderFileList(type) {
  const list = type === 'image' ? imageList : coordList;
  const items = state.fileStore[type]; list.innerHTML = '';
  if (!items.length) { list.innerHTML = '<div class="file-name">Файлы не загружены</div>'; return; }
  items.forEach((item, index) => {
    const row = document.createElement('div'); row.className = 'file-row clickable-row';
    row.innerHTML = `<div class="file-name">${escapeHtml(item.name)}</div><div class="file-actions"><button class="file-action-btn" type="button" data-action="preview" title="Открыть">◉</button><button class="file-action-btn" type="button" data-action="rename" title="Переименовать">✎</button><button class="file-action-btn" type="button" data-action="delete" title="Удалить">🗑</button></div>`;
    row.addEventListener('click', (event) => { const action = event.target.closest('[data-action]')?.dataset.action; if (!action) openPreview(item); else fileAction(type, index, action); });
    list.appendChild(row);
  });
}

async function openPreview(item) {
  previewContent.innerHTML = '';
  if (item.kind === 'image') previewContent.innerHTML = `<img src="${item.preview}" alt="${escapeHtml(item.name)}">`;
  else {
    previewContent.textContent = 'Читаем таблицу…';
    try { previewContent.innerHTML = makeTableMarkup(item.name, await readTable(item)); }
    catch (error) { previewContent.textContent = `Ошибка таблицы: ${error.message}`; }
  }
  previewModal.classList.remove('hidden'); previewModal.setAttribute('aria-hidden', 'false');
}

function closePreview() { previewModal.classList.add('hidden'); previewModal.setAttribute('aria-hidden', 'true'); }

function openRenameModal(type, index) {
  state.activeRename = { type, index }; const item = state.fileStore[type][index];
  renameInput.value = item.name.replace(/\.[^.]+$/, ''); extensionTrigger.textContent = item.name.split('.').pop() || '';
  renameModal.classList.remove('hidden'); renameModal.setAttribute('aria-hidden', 'false');
}
function closeRenameModal() { renameModal.classList.add('hidden'); renameModal.setAttribute('aria-hidden', 'true'); state.activeRename = null; }
async function saveRename() {
  if (!state.activeRename) return;
  const { type, index } = state.activeRename; const item = state.fileStore[type][index];
  const base = renameInput.value.trim(); if (!base) return;
  const newName = extensionTrigger.textContent ? `${base}.${extensionTrigger.textContent}` : base;
  try {
    if (item.id) { const body = new FormData(); body.append('name', newName); const response = await fetch(`/api/replenishment/files/${item.id}`, { method: 'PATCH', body }); if (!response.ok) throw new Error('Не удалось переименовать файл.'); }
    item.name = newName; closeRenameModal(); renderFileList(type); refreshTablePreview();
  } catch (error) { setArchiveStatus(error.message, 'error'); }
}
async function deleteFile(type, index) {
  const item = state.fileStore[type][index];
  if (!confirm(`Удалить «${item.name}»?`)) return;
  try {
    if (item.id) { const response = await fetch(`/api/replenishment/files/${item.id}`, { method: 'DELETE' }); if (!response.ok) throw new Error('Не удалось удалить файл.'); }
    state.fileStore[type].splice(index, 1); renderFileList(type); refreshTablePreview();
  } catch (error) { setArchiveStatus(error.message, 'error'); }
}
function fileAction(type, index, action) { if (action === 'preview') openPreview(state.fileStore[type][index]); if (action === 'rename') openRenameModal(type, index); if (action === 'delete') deleteFile(type, index); }

async function addFiles(type, files) {
  for (const file of Array.from(files || [])) {
    if (type === 'image') {
      if (!file.type.startsWith('image/') && !/\.(png|jpe?g)$/i.test(file.name)) { setArchiveStatus(`«${file.name}» не является изображением.`, 'error'); continue; }
      state.fileStore.image.unshift({ name: file.name, kind: 'image', preview: URL.createObjectURL(file), file });
    } else {
      try { const rows = parseCoordinateTable(await file.text()); state.fileStore.coord.unshift({ name: file.name, kind: 'coord', file, rows }); }
      catch (error) { setArchiveStatus(`«${file.name}»: ${error.message}`, 'error'); }
    }
  }
  renderFileList(type); refreshTablePreview();
}

document.querySelectorAll('.add-file-btn').forEach((button) => button.addEventListener('click', () => {
  const type = button.dataset.type === 'image' ? 'image' : 'coord'; const input = document.createElement('input');
  input.type = 'file'; input.multiple = true; input.accept = type === 'image' ? 'image/*,.png,.jpg,.jpeg' : '.csv,.json,.txt,text/csv,application/json,text/plain';
  input.addEventListener('change', (event) => addFiles(type, event.target.files)); input.click();
}));

function normalizedId(value) { return String(value || '').replace(/\\/g, '/').split('/').pop().toLowerCase(); }
function findRowForImage(rows, item) {
  const fullName = normalizedId(item.name); const stem = fileStem(item.name);
  return rows.find((row) => { const id = normalizedId(row.image_id); return id === fullName || fileStem(id) === stem; }) || (rows.length === 1 ? rows[0] : null);
}
function imageDimensions(src) { return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight }); image.onerror = reject; image.src = src; }); }
function fitBounds(row, size) {
  const x = Math.max(0, Math.floor(Number(row.x))); const y = Math.max(0, Math.floor(Number(row.y)));
  const right = Math.min(size.width, Math.ceil(Number(row.x) + Number(row.w))); const bottom = Math.min(size.height, Math.ceil(Number(row.y) + Number(row.h)));
  if (!Number.isFinite(x + y + right + bottom) || right <= x || bottom <= y) throw new Error('координаты выходят за пределы изображения');
  return { x, y, w: right - x, h: bottom - y };
}
function renderManyResults(results) {
  manyResultsGrid.innerHTML = results.map(({ item, payload, error }) => {
    const recognized = !error && payload.status === 'matched' && payload.ranked?.length;
    const label = recognized ? escapeHtml(payload.ranked[0].gallery_id) : 'Не распознано';
    return `<article class="many-result-card"><img src="${item.preview}" alt="${escapeHtml(item.name)}"><div class="many-result-caption"><strong>${label}</strong><span>${recognized ? '№ 1' : ''}</span></div>${error ? `<small>${escapeHtml(error.message || error)}</small>` : ''}</article>`;
  }).join('');
  manyResults.classList.remove('hidden'); manyResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

submitManyBtn.addEventListener('click', async () => {
  const images = state.fileStore.image;
  if (!images.length) { setArchiveStatus('Добавьте хотя бы одно изображение.', 'error'); return; }
  if (!state.fileStore.coord.length) { setArchiveStatus('Загрузите таблицу с указанными ниже колонками.', 'error'); return; }
  submitManyBtn.disabled = true; setArchiveStatus(`Распознаём ${images.length} изображений…`);
  try {
    const tables = await Promise.all(state.fileStore.coord.map(readTable)); const rows = tables.flat(); const results = [];
    for (const item of images) {
      try {
        const [blob, size] = await Promise.all([fetch(item.preview).then((response) => response.blob()), imageDimensions(item.preview)]);
        const row = findRowForImage(rows, item); if (!row) throw new Error('строка с этим image_id не найдена');
        const bbox = fitBounds(row, size); const body = new FormData(); body.append('image', blob, item.name || 'query.png');
        Object.entries(bbox).forEach(([key, value]) => body.append(key, String(value))); body.append('topk', '10');
        const response = await fetch('/api/infer', { method: 'POST', body }); const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'ошибка распознавания'); results.push({ item, payload });
      } catch (error) { results.push({ item, error }); }
    }
    renderManyResults(results); const failed = results.filter((result) => result.error).length;
    setArchiveStatus(failed ? `Готово: ${failed} изображений не обработано. Проверьте строки таблицы.` : 'Распознавание завершено.', failed ? 'error' : 'success');
  } catch (error) { setArchiveStatus(error.message, 'error'); }
  finally { submitManyBtn.disabled = false; }
});

archiveUploadBtn.addEventListener('click', () => {
  const input = document.createElement('input'); input.type = 'file'; input.accept = '.zip,.tar,.tgz,.gz,.bz2,.xz';
  input.onchange = async (event) => { const file = event.target.files?.[0]; if (!file) return; try { setArchiveStatus('Загружаем архив…'); const body = new FormData(); body.append('archive', file); const response = await fetch('/api/replenishment/archive', { method: 'POST', body }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || 'Ошибка архива'); setArchiveStatus(`В базу добавлено ${payload.count} файлов.`, 'success'); await loadPersistedFiles(); } catch (error) { setArchiveStatus(error.message, 'error'); } };
  input.click();
});

backHomeBtn.addEventListener('click', () => { window.location.href = '/'; });
renameCloseBtn.addEventListener('click', closeRenameModal); cancelRenameBtn.addEventListener('click', closeRenameModal); saveRenameBtn.addEventListener('click', saveRename);
previewModal.addEventListener('click', (event) => { if (event.target.dataset.close === 'true') closePreview(); });
renameModal.addEventListener('click', (event) => { if (event.target.dataset.close === 'true') closeRenameModal(); });
loadPersistedFiles();
