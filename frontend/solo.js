const imageInput = document.getElementById('soloImageInput');
const canvas = document.getElementById('drawingCanvas');
const stage = document.getElementById('stage');
const galleryStatus = document.getElementById('soloGalleryStatus');
const analyzeBtn = document.getElementById('analyzeBtn');
const errorMessage = document.getElementById('errorMessage');
const bboxReadout = document.getElementById('bboxReadout');
let imageFile = null, imageUrl = '', imageSize = null, start = null, box = null;
let commonGallery = null;

const escapeHtml = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
function showError(message) { errorMessage.textContent = message; errorMessage.classList.add('visible'); }
function clearError() { errorMessage.classList.remove('visible'); }

async function loadCommonGallery() {
  const response = await fetch('/api/common-gallery');
  if (!response.ok) throw new Error('Не удалось проверить общую галерею.');
  commonGallery = await response.json();
  galleryStatus.textContent = commonGallery.search_ready
    ? `● Общая галерея готова · ${commonGallery.image_count} автомобилей`
    : `○ В общей галерее ${commonGallery.image_count} автомобилей · нужно минимум ${commonGallery.minimum_for_search || 10}`;
  renderBox();
}
const gallerySelectionReady = loadCommonGallery().catch((error) => { galleryStatus.textContent = error.message; });
window.addEventListener('focus', () => { loadCommonGallery().catch((error) => { galleryStatus.textContent = error.message; }); });

function renderBox() {
  canvas.querySelector('#selectedBox')?.remove();
  if (box) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.id = 'selectedBox';
    for (const key of ['x', 'y', 'w', 'h']) rect.setAttribute(key === 'w' ? 'width' : key === 'h' ? 'height' : key, box[key]);
    rect.setAttribute('fill', 'rgba(76,201,240,.17)'); rect.setAttribute('stroke', '#ffb703');
    rect.setAttribute('stroke-width', '3'); rect.setAttribute('vector-effect', 'non-scaling-stroke');
    canvas.appendChild(rect);
  }
  bboxReadout.textContent = box && box.w >= 2 && box.h >= 2 ? `BBox: x=${box.x}, y=${box.y}, w=${box.w}, h=${box.h} пикселей` : 'Протяните прямоугольник от одного угла автомобиля до противоположного.';
  analyzeBtn.disabled = !commonGallery?.search_ready || !box || box.w < 2 || box.h < 2;
}
function imagePoint(event) {
  const point = canvas.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
  const inverse = canvas.getScreenCTM()?.inverse(); if (!inverse) return null;
  const mapped = point.matrixTransform(inverse);
  return { x: Math.max(0, Math.min(imageSize.width, Math.round(mapped.x))), y: Math.max(0, Math.min(imageSize.height, Math.round(mapped.y))) };
}
function rectangle(a, b) { return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) }; }

function setImage(file) {
  if (!file) return;
  if (!/\.(jpe?g|png)$/i.test(file.name)) { showError('Выберите JPG или PNG.'); return; }
  const nextUrl = URL.createObjectURL(file), image = new Image();
  image.onload = () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    imageFile = file; imageUrl = nextUrl; imageSize = { width: image.naturalWidth, height: image.naturalHeight };
    box = null; start = null; canvas.setAttribute('viewBox', `0 0 ${imageSize.width} ${imageSize.height}`);
    const photo = document.createElementNS('http://www.w3.org/2000/svg', 'image');
    photo.setAttribute('href', imageUrl); photo.setAttribute('width', imageSize.width); photo.setAttribute('height', imageSize.height);
    canvas.replaceChildren(photo); document.getElementById('uploadBox').classList.add('hidden'); stage.classList.remove('hidden');
    clearError(); renderBox();
  };
  image.onerror = () => { URL.revokeObjectURL(nextUrl); showError('Не удалось открыть изображение.'); };
  image.src = nextUrl;
}
canvas.addEventListener('pointerdown', (event) => {
  if (!imageSize || event.button !== 0) return;
  start = imagePoint(event); if (!start) return;
  box = { x: start.x, y: start.y, w: 0, h: 0 };
  canvas.setPointerCapture(event.pointerId); clearError(); renderBox();
});
canvas.addEventListener('pointermove', (event) => { if (start) { const end = imagePoint(event); if (end) { box = rectangle(start, end); renderBox(); } } });
canvas.addEventListener('pointerup', (event) => {
  if (!start) return;
  const end = imagePoint(event); if (end) box = rectangle(start, end);
  start = null;
  if (!box || box.w < 2 || box.h < 2) { box = null; showError('Выделите прямоугольник машины немного крупнее.'); }
  renderBox();
});
canvas.addEventListener('pointercancel', () => { start = null; box = null; renderBox(); });
document.getElementById('clearBoxBtn').addEventListener('click', () => { box = null; clearError(); renderBox(); });
imageInput.addEventListener('change', (event) => setImage(event.target.files[0]));
document.getElementById('replaceImageBtn').addEventListener('click', () => { imageInput.value = ''; imageInput.click(); });
document.getElementById('backHomeBtn').addEventListener('click', () => { window.location.href = '/'; });

function showResult(data) {
  document.getElementById('soloResultModal')?.remove();
  const modal = document.createElement('div'); modal.id = 'soloResultModal'; modal.className = 'modal solo-result-modal';
  const accepted = Array.isArray(data.accepted) ? data.accepted : [];
  const ranked = Array.isArray(data.ranked) ? data.ranked.slice(0, 10) : [];
  const top = data.status === 'matched' ? accepted[0] : null;
  const score = (item) => Number.isFinite(Number(item?.confidence)) ? Number(item.confidence).toFixed(3) : '—';
  const summary = `<div class="solo-result-main"><img class="solo-result-image" src="${imageUrl}" alt="Загруженный исходный кадр"><div class="solo-result-answer"><p>${top ? 'Принятый кандидат' : 'Совпадение не подтверждено'}</p><h2>${top ? escapeHtml(top.gallery_id) : 'Модель отказалась от ответа'}</h2><p>${top ? `Сходство cosine: ${score(top)}` : 'Даже при отказе можно просмотреть ближайшие автомобили.'} · cosine не является вероятностью</p></div></div>`;
  modal.innerHTML = `<div class="modal-backdrop"></div><section class="solo-result-dialog" role="dialog" aria-modal="true" aria-label="Результат поиска"><button class="close-btn result-close" type="button" aria-label="Закрыть">×</button>${summary}<div class="candidate-section-heading"><h3>Все кандидаты top‑10</h3><p>Нажмите на фото, чтобы рассмотреть автомобиль крупнее.</p></div><div class="candidate-grid-slot"></div><div class="solo-result-footer"><span>Показано ${ranked.length} кандидатов</span><button class="ghost-btn result-export" type="button">Скачать результат JSON</button></div></section>`;
  modal.querySelector('.candidate-grid-slot').appendChild(CandidateGallery.render(ranked, accepted));
  modal.addEventListener('click', (event) => { if (event.target.classList.contains('modal-backdrop') || event.target.classList.contains('result-close')) modal.remove(); });
  modal.querySelector('.result-export').addEventListener('click', () => ReidExport.download('reid-result.json', ReidExport.soloJson(data), 'application/json;charset=utf-8'));
  document.body.appendChild(modal);
}
analyzeBtn.addEventListener('click', async () => {
  if (!imageFile || !box || box.w < 2 || box.h < 2) { showError('Сначала выделите автомобиль.'); return; }
  await gallerySelectionReady;
  try { await loadCommonGallery(); }
  catch (error) { showError(error.message); return; }
  if (!commonGallery?.search_ready) { showError('Общая галерея ещё не готова. Добавьте минимум 10 автомобилей.'); return; }
  const body = new FormData(); body.append('image', imageFile, imageFile.name);
  Object.entries(box).forEach(([key, value]) => body.append(key, String(value)));
  body.append('topk', '10'); body.append('gallery_id', commonGallery.gallery_id);
  analyzeBtn.disabled = true; analyzeBtn.textContent = 'Ищем совпадения…'; clearError();
  try {
    const response = await fetch('/api/infer', { method: 'POST', body });
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Ошибка распознавания');
    showResult(data);
  } catch (error) { showError(error.message); }
  finally { analyzeBtn.textContent = 'Найти совпадения'; renderBox(); }
});
