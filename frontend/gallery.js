const gallerySelect = document.getElementById('gallerySelect');
const galleryName = document.getElementById('galleryName');
const galleryImages = document.getElementById('galleryImages');
const galleryArchive = document.getElementById('galleryArchive');
const galleryManifest = document.getElementById('galleryManifest');
const gallerySelection = document.getElementById('gallerySelection');
const galleryDetails = document.getElementById('galleryDetails');
const galleryStatus = document.getElementById('galleryStatus');
const createGalleryBtn = document.getElementById('createGalleryBtn');
const retryGalleryBtn = document.getElementById('retryGalleryBtn');
const uploadGalleryBtn = document.getElementById('uploadGalleryBtn');
let activePoll = null;

function setStatus(message, kind = 'info') {
  galleryStatus.textContent = message;
  galleryStatus.className = `upload-status ${kind}`;
}

async function apiJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : `Ошибка HTTP ${response.status}`);
  }
  return data;
}

function renderGallery(info) {
  if (!info) { galleryDetails.textContent = 'Выберите галерею или создайте новую.'; retryGalleryBtn.classList.add('hidden'); return; }
  const message = info.state === 'failed' ? `Ошибка индексации: ${info.error || 'причина неизвестна'}`
    : info.state === 'building' ? `Индексация: ${info.processed || 0} / ${info.pending_count || 0}`
      : info.search_ready ? 'Готова к поиску' : `Для поиска нужно минимум ${info.minimum_for_search || 10} автомобилей`;
  galleryDetails.textContent = `${info.name}: ${info.image_count} изображений. ${message}.`;
  galleryDetails.dataset.state = info.state;
  retryGalleryBtn.classList.toggle('hidden', info.state !== 'failed' || !info.pending_count);
}

async function refreshGalleries(preferredId = gallerySelect.value) {
  const galleries = await apiJson('/api/galleries');
  gallerySelect.replaceChildren(new Option('Выберите галерею', ''));
  galleries.forEach((item) => gallerySelect.add(new Option(`${item.name} · ${item.image_count} авто`, item.gallery_id)));
  if (preferredId && galleries.some((item) => item.gallery_id === preferredId)) gallerySelect.value = preferredId;
  renderGallery(galleries.find((item) => item.gallery_id === gallerySelect.value));
  return galleries;
}

function summarizeSelection() {
  const images = [...galleryImages.files];
  const archive = galleryArchive.files[0];
  const manifest = galleryManifest.files[0];
  gallerySelection.textContent = [images.length ? `${images.length} изображений` : '', archive ? `ZIP: ${archive.name}` : '', manifest ? `BBox: ${manifest.name}` : ''].filter(Boolean).join(' · ') || 'Файлы не выбраны.';
}

function validateSelection() {
  const images = [...galleryImages.files];
  const archive = galleryArchive.files[0];
  const manifest = galleryManifest.files[0];
  if (Boolean(images.length) === Boolean(archive)) throw new Error('Выберите либо фотографии, либо один ZIP-архив.');
  if (images.length > 200) throw new Error('В одной партии не более 200 изображений.');
  if (images.some((file) => !/\.(jpe?g|png)$/i.test(file.name) || file.size > 20 * 1024 * 1024)) throw new Error('Каждое изображение должно быть JPG/PNG размером до 20 МиБ.');
  if (images.reduce((sum, file) => sum + file.size, 0) > 200 * 1024 * 1024) throw new Error('Размер партии не должен превышать 200 МиБ.');
  if (archive && (archive.size > 100 * 1024 * 1024 || !/\.zip$/i.test(archive.name))) throw new Error('Нужен ZIP размером до 100 МиБ.');
  if (manifest && (manifest.size > 1024 * 1024 || !/\.csv$/i.test(manifest.name))) throw new Error('Нужен CSV размером до 1 МиБ.');
  if (images.length && manifest) {
    const names = images.map((file) => file.name);
    if (new Set(names).size !== names.length) throw new Error('Имена изображений в партии не должны повторяться.');
  }
  return { images, archive, manifest };
}

function csvRows(text) {
  const data = String(text).replace(/^\uFEFF/, '');
  const header = data.split(/\r?\n/, 1)[0];
  const delimiter = header.includes(';') ? ';' : header.includes('\t') ? '\t' : ',';
  const result = []; let row = [], cell = '', quoted = false;
  for (let i = 0; i < data.length; i += 1) {
    const char = data[i];
    if (char === '"') { if (quoted && data[i + 1] === '"') { cell += '"'; i += 1; } else quoted = !quoted; }
    else if (!quoted && char === delimiter) { row.push(cell.trim()); cell = ''; }
    else if (!quoted && (char === '\n' || char === '\r')) {
      if (char === '\r' && data[i + 1] === '\n') i += 1;
      row.push(cell.trim()); cell = '';
      if (row.some(Boolean)) result.push(row);
      row = [];
    } else cell += char;
  }
  if (quoted) throw new Error('В CSV не закрыта кавычка.');
  row.push(cell.trim()); if (row.some(Boolean)) result.push(row);
  return result;
}

function quoteCsv(value) { return `"${String(value).replace(/"/g, '""')}"`; }

async function normalizeManifest(file, images, archive) {
  if (!file) return null;
  const table = csvRows(await file.text());
  if (table.length < 2) throw new Error('BBox CSV пуст.');
  const headers = table[0].map((cell) => cell.toLowerCase());
  if (headers.includes('filename')) {
    if (['x', 'y', 'w', 'h'].some((key) => !headers.includes(key))) throw new Error('В manifest CSV нужны координаты x,y,w,h.');
    return file;
  }
  if (!headers.includes('image_id') || ['x', 'y', 'w', 'h'].some((key) => !headers.includes(key))) {
    throw new Error('BBox CSV должен содержать image_id,x,y,w,h или filename,gallery_id,x,y,w,h.');
  }
  if (archive) throw new Error('Для ZIP нужен manifest.csv с колонкой filename. Для CSV организатора выберите отдельные JPG-файлы.');
  const position = Object.fromEntries(headers.map((name, index) => [name, index]));
  const used = new Set();
  const lines = ['filename,gallery_id,x,y,w,h'];
  for (const values of table.slice(1)) {
    if (values.length !== headers.length) throw new Error('Число колонок в CSV не совпадает с заголовком.');
    const id = values[position.image_id];
    const image = images.find((item) => item.name.toLowerCase() === id.toLowerCase()) || images.find((item) => item.name.replace(/\.[^.]+$/, '').toLowerCase() === id.toLowerCase());
    if (!image || used.has(image.name)) throw new Error(`Не удалось однозначно сопоставить image_id «${id}» с загруженным файлом.`);
    used.add(image.name);
    lines.push([image.name, id, ...['x', 'y', 'w', 'h'].map((key) => values[position[key]])].map(quoteCsv).join(','));
  }
  if (used.size !== images.length) throw new Error('Для каждого загруженного кадра нужна строка BBox CSV.');
  return new File([lines.join('\n') + '\n'], 'manifest.csv', { type: 'text/csv' });
}

async function pollImport(id) {
  clearTimeout(activePoll);
  try {
    const info = await apiJson(`/api/galleries/${id}`);
    renderGallery(info);
    if (info.state === 'building') {
      setStatus(`Индексируем: ${info.processed || 0} / ${info.pending_count || 0}. Страницу можно оставить открытой.`);
      activePoll = setTimeout(() => pollImport(id), 2500);
    } else if (info.state === 'failed') {
      setStatus(`Индексация не завершилась: ${info.error || 'неизвестная ошибка'}. Можно повторить попытку.`, 'error');
    } else {
      setStatus(info.search_ready ? `Галерея готова: ${info.image_count} автомобилей. Теперь можно искать.` : `Партия добавлена; сейчас ${info.image_count} автомобилей. Для поиска требуется минимум 10.`, 'success');
      await refreshGalleries(id);
    }
  } catch (error) { setStatus(error.message, 'error'); }
}

createGalleryBtn.addEventListener('click', async () => {
  const name = galleryName.value.trim();
  if (!name) { setStatus('Введите название новой галереи.', 'error'); return; }
  createGalleryBtn.disabled = true;
  try {
    const form = new FormData(); form.append('name', name);
    const info = await apiJson('/api/galleries', { method: 'POST', body: form });
    await refreshGalleries(info.gallery_id);
    galleryName.value = '';
    setStatus(`Галерея «${info.name}» создана. Теперь добавьте фотографии.`, 'success');
  } catch (error) { setStatus(error.message, 'error'); }
  finally { createGalleryBtn.disabled = false; }
});

gallerySelect.addEventListener('change', async () => {
  clearTimeout(activePoll);
  try {
    const id = gallerySelect.value;
    const info = id ? await apiJson(`/api/galleries/${id}`) : null;
    renderGallery(info);
    if (info?.state === 'building') pollImport(id);
  } catch (error) { setStatus(error.message, 'error'); }
});

galleryImages.addEventListener('change', () => { if (galleryImages.files.length) galleryArchive.value = ''; summarizeSelection(); });
galleryArchive.addEventListener('change', () => { if (galleryArchive.files.length) galleryImages.value = ''; summarizeSelection(); });
galleryManifest.addEventListener('change', summarizeSelection);

retryGalleryBtn.addEventListener('click', async () => {
  const id = gallerySelect.value;
  if (!id) return;
  retryGalleryBtn.disabled = true;
  try {
    await apiJson(`/api/galleries/${id}/retry`, { method: 'POST' });
    setStatus('Повторно индексируем галерею…');
    await pollImport(id);
  } catch (error) { setStatus(error.message, 'error'); }
  finally { retryGalleryBtn.disabled = false; }
});

uploadGalleryBtn.addEventListener('click', async () => {
  if (!gallerySelect.value) { setStatus('Сначала выберите или создайте галерею.', 'error'); return; }
  let selection;
  try { selection = validateSelection(); }
  catch (error) { setStatus(error.message, 'error'); return; }
  uploadGalleryBtn.disabled = true;
  setStatus('Передаём фотографии и запускаем индексацию…');
  try {
    const form = new FormData();
    selection.images.forEach((file) => form.append('images', file));
    if (selection.archive) form.append('archive', selection.archive);
    const normalizedManifest = await normalizeManifest(selection.manifest, selection.images, selection.archive);
    if (normalizedManifest) form.append('manifest', normalizedManifest);
    const id = gallerySelect.value;
    await apiJson(`/api/galleries/${id}/images`, { method: 'POST', body: form });
    galleryImages.value = ''; galleryArchive.value = ''; galleryManifest.value = ''; summarizeSelection();
    await pollImport(id);
  } catch (error) { setStatus(error.message, 'error'); }
  finally { uploadGalleryBtn.disabled = false; }
});

refreshGalleries().catch((error) => setStatus(error.message, 'error'));
