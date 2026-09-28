const galleryImages = document.getElementById('galleryImages');
const galleryArchive = document.getElementById('galleryArchive');
const galleryManifest = document.getElementById('galleryManifest');
const gallerySelection = document.getElementById('gallerySelection');
const galleryDetails = document.getElementById('galleryDetails');
const galleryStatus = document.getElementById('galleryStatus');
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
  if (!info) { galleryDetails.textContent = 'Статус общей галереи недоступен.'; retryGalleryBtn.classList.add('hidden'); return; }
  const message = info.state === 'failed' ? `Ошибка индексации: ${info.error || 'причина неизвестна'}`
    : info.state === 'building' ? `Индексация: ${info.processed || 0} / ${info.pending_count || 0}`
      : info.search_ready ? 'Готова к поиску' : `Для поиска нужно минимум ${info.minimum_for_search || 10} автомобилей`;
  galleryDetails.textContent = `${info.image_count} автомобилей · ${message}`;
  galleryDetails.dataset.state = info.state;
  retryGalleryBtn.classList.toggle('hidden', info.state !== 'failed' || !info.pending_count);
}

async function refreshGallery() {
  const info = await apiJson('/api/common-gallery');
  renderGallery(info);
  return info;
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
  if (images.some((file) => !/\.(jpe?g|png)$/i.test(file.name))) throw new Error('Каждое изображение должно быть JPG/PNG.');
  if (archive && !/\.zip$/i.test(archive.name)) throw new Error('Нужен ZIP-архив.');
  if (manifest && !/\.csv$/i.test(manifest.name)) throw new Error('Нужен CSV-файл.');
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
  if (archive) throw new Error('Для ZIP нужен CSV с колонкой filename и путями файлов внутри архива.');
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

async function pollImport() {
  clearTimeout(activePoll);
  try {
    const info = await refreshGallery();
    if (info.state === 'building') {
      setStatus(`Индексируем: ${info.processed || 0} / ${info.pending_count || 0}. Страницу можно оставить открытой.`);
      activePoll = setTimeout(pollImport, 2500);
    } else if (info.state === 'failed') {
      setStatus(`Индексация не завершилась: ${info.error || 'неизвестная ошибка'}. Можно повторить попытку.`, 'error');
    } else {
      setStatus(info.search_ready ? `Галерея готова: ${info.image_count} автомобилей. Теперь можно искать.` : `Партия добавлена; сейчас ${info.image_count} автомобилей. Для поиска требуется минимум 10.`, 'success');
      await refreshGallery();
    }
  } catch (error) { setStatus(error.message, 'error'); }
}

galleryImages.addEventListener('change', () => { if (galleryImages.files.length) galleryArchive.value = ''; summarizeSelection(); });
galleryArchive.addEventListener('change', () => { if (galleryArchive.files.length) galleryImages.value = ''; summarizeSelection(); });
galleryManifest.addEventListener('change', summarizeSelection);

retryGalleryBtn.addEventListener('click', async () => {
  retryGalleryBtn.disabled = true;
  try {
    await apiJson('/api/common-gallery/retry', { method: 'POST' });
    setStatus('Повторно индексируем галерею…');
    await pollImport();
  } catch (error) { setStatus(error.message, 'error'); }
  finally { retryGalleryBtn.disabled = false; }
});

uploadGalleryBtn.addEventListener('click', async () => {
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
    await apiJson('/api/common-gallery/images', { method: 'POST', body: form });
    galleryImages.value = ''; galleryArchive.value = ''; galleryManifest.value = ''; summarizeSelection();
    await pollImport();
  } catch (error) { setStatus(error.message, 'error'); }
  finally { uploadGalleryBtn.disabled = false; }
});

refreshGallery().then((info) => { if (info.state === 'building') pollImport(); })
  .catch((error) => setStatus(error.message, 'error'));
