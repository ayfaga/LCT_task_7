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
    if (selection.manifest) form.append('manifest', selection.manifest);
    await apiJson('/api/common-gallery/images', { method: 'POST', body: form });
    galleryImages.value = ''; galleryArchive.value = ''; galleryManifest.value = ''; summarizeSelection();
    await pollImport();
  } catch (error) { setStatus(error.message, 'error'); }
  finally { uploadGalleryBtn.disabled = false; }
});

refreshGallery().then((info) => { if (info.state === 'building') pollImport(); })
  .catch((error) => setStatus(error.message, 'error'));
