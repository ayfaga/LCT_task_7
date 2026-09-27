const uploadBox = document.getElementById('uploadBox');
const soloImageInput = document.getElementById('soloImageInput');
const stage = document.getElementById('stage');
const drawingCanvas = document.getElementById('drawingCanvas');
const errorMessage = document.getElementById('errorMessage');
const undoBtn = document.getElementById('undoBtn');
const redoBtn = document.getElementById('redoBtn');
const backHomeBtn = document.getElementById('backHomeBtn');
const replaceImageBtn = document.getElementById('replaceImageBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const gallerySelect = document.getElementById('soloGallerySelect');

async function loadGalleryChoices() {
  const [readyResponse, galleriesResponse] = await Promise.all([fetch('/ready'), fetch('/api/galleries')]);
  if (!readyResponse.ok || !galleriesResponse.ok) throw new Error('Не удалось получить список галерей.');
  const ready = await readyResponse.json();
  const galleries = await galleriesResponse.json();
  gallerySelect.replaceChildren();
  if (ready.gallery_ready) gallerySelect.add(new Option(`Основная (${ready.gallery_size})`, ''));
  for (const gallery of galleries.filter((entry) => entry.search_ready)) {
    gallerySelect.add(new Option(`${gallery.name} (${gallery.image_count})`, gallery.gallery_id));
  }
  if (!gallerySelect.options.length) {
    gallerySelect.add(new Option('Нет готовой галереи', ''));
    gallerySelect.disabled = true;
  }
}
const gallerySelectionReady = loadGalleryChoices().catch((error) => {
  gallerySelect.replaceChildren(new Option(error.message, ''));
  gallerySelect.disabled = true;
});

const MAX_POINTS = 4;

let imageUrl = '';
let points = [];
let redoStack = [];
let activeImage = null;
let dragIndex = null;

function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');
}

function closeResultModal() {
  document.getElementById('soloResultModal')?.remove();
}

function cosineLabel(item) {
  const score = Number(item?.confidence);
  return Number.isFinite(score) ? score.toFixed(3) : '—';
}

function downloadResult(data) {
  ReidExport.download('reid-result.json', ReidExport.soloJson(data), 'application/json;charset=utf-8');
}

function showSoloResult(data) {
  closeResultModal();
  const ranked = Array.isArray(data.ranked) ? data.ranked.slice(0, 10) : [];
  const accepted = Array.isArray(data.accepted) ? data.accepted : [];
  const recognized = data.status === 'matched' && accepted.length > 0;
  const modal = document.createElement('div');
  modal.id = 'soloResultModal';
  modal.className = 'modal solo-result-modal';
  if (!recognized) {
    modal.innerHTML = '<div class="modal-backdrop"></div><section class="solo-result-dialog refusal-dialog"><button class="close-btn result-close" type="button">×</button><h2>Совпадение не подтверждено</h2><p>Модель отказалась от ответа. Ранжированный список доступен в экспорте.</p><button class="ghost-btn result-export" type="button">Скачать результат JSON</button></section>';
  } else {
    const top = accepted[0];
    const alternatives = ranked.filter((item) => item.gallery_id !== top.gallery_id).map((item) => `<li><span>${escapeHtml(item.gallery_id)} · cosine ${cosineLabel(item)}</span></li>`).join('');
    modal.innerHTML = `<div class="modal-backdrop"></div><section class="solo-result-dialog"><button class="close-btn result-close" type="button">×</button><div class="solo-result-main"><img class="solo-result-image" src="${imageUrl}" alt="Загруженный автомобиль"><div class="solo-result-answer"><p>Принятый кандидат</p><h2>${escapeHtml(top.gallery_id)}</h2><p>Сходство cosine: ${cosineLabel(top)} (не вероятность)</p></div></div><h3>Остальные в top‑10</h3><ol class="solo-alternatives">${alternatives || '<li><span>Других вариантов не найдено</span></li>'}</ol><button class="ghost-btn result-export" type="button">Скачать результат JSON</button></section>`;
  }
  modal.addEventListener('click', (event) => { if (event.target.classList.contains('modal-backdrop') || event.target.classList.contains('result-close')) closeResultModal(); });
  modal.querySelector('.result-export')?.addEventListener('click', () => downloadResult(data));
  document.body.appendChild(modal);
}

function polygonBounds(polygon, image) {
  const left = Math.max(0, Math.floor(Math.min(...polygon.map(([x]) => x))));
  const top = Math.max(0, Math.floor(Math.min(...polygon.map(([, y]) => y))));
  const right = Math.min(image.width, Math.ceil(Math.max(...polygon.map(([x]) => x))));
  const bottom = Math.min(image.height, Math.ceil(Math.max(...polygon.map(([, y]) => y))));
  if (right <= left || bottom <= top) throw new Error('Выделите область машины ещё раз.');
  return { x: left, y: top, w: right - left, h: bottom - top };
}

function setStatus(message, kind = 'info') {
  const panel = document.querySelector('.result-panel');
  if (panel) {
    panel.remove();
  }
  const statusBox = document.createElement('div');
  statusBox.className = 'result-panel';
  statusBox.textContent = message;
  statusBox.dataset.kind = kind;
  const toolbar = document.querySelector('.action-bar');
  if (toolbar) {
    toolbar.appendChild(statusBox);
  }
}

function clearError() {
  errorMessage.classList.remove('visible');
}

function updateButtons() {
  undoBtn.disabled = points.length === 0;
  redoBtn.disabled = redoStack.length === 0;
}

function renderPoints() {
  const svg = drawingCanvas;
  svg.innerHTML = '';

  if (!imageUrl) return;

  const img = activeImage || new Image();
  if (!activeImage) {
    img.onload = () => {
      activeImage = img;
      renderPoints();
    };
    img.src = imageUrl;
    return;
  }

  const { width, height } = activeImage;
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  const imageNode = document.createElementNS('http://www.w3.org/2000/svg', 'image');
  imageNode.setAttribute('href', imageUrl);
  imageNode.setAttribute('x', '0');
  imageNode.setAttribute('y', '0');
  imageNode.setAttribute('width', String(width));
  imageNode.setAttribute('height', String(height));
  imageNode.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  svg.appendChild(imageNode);

  if (points.length >= 2) {
    const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    const pointString = points.map((point) => `${point.x},${point.y}`).join(' ');
    polygon.setAttribute('points', pointString);
    polygon.setAttribute('fill', 'rgba(40, 120, 201, 0.2)');
    polygon.setAttribute('stroke', '#1f5ba9');
    polygon.setAttribute('stroke-width', '2');
    svg.appendChild(polygon);
  }

  points.forEach((point, index) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', point.x);
    circle.setAttribute('cy', point.y);
    circle.setAttribute('r', '12');
    circle.setAttribute('fill', '#0f4c81');
    circle.setAttribute('stroke', 'white');
    circle.setAttribute('stroke-width', '2');
    svg.appendChild(circle);

    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    label.setAttribute('x', point.x);
    label.setAttribute('y', point.y + 5);
    label.setAttribute('text-anchor', 'middle');
    label.setAttribute('fill', 'white');
    label.setAttribute('font-size', '14');
    label.setAttribute('font-weight', '700');
    label.textContent = String(index + 1);
    svg.appendChild(label);
  });
}

function setImage(file) {
  if (!file) return;
  imageUrl = URL.createObjectURL(file);
  points = [];
  redoStack = [];
  activeImage = null;
  clearError();
  uploadBox.classList.add('hidden');
  stage.classList.remove('hidden');
  renderPoints();
  updateButtons();
}

function getSvgPoint(event) {
  const svg = drawingCanvas;
  const rect = svg.getBoundingClientRect();
  const scaleX = svg.viewBox.baseVal.width / rect.width;
  const scaleY = svg.viewBox.baseVal.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function segmentsIntersect(a, b, c, d) {
  const orientation = (p, q, r) => {
    const val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
    if (Math.abs(val) < 0.0001) return 0;
    return val > 0 ? 1 : 2;
  };

  const onSegment = (p, q, r) => {
    return Math.min(p.x, r.x) <= q.x && q.x <= Math.max(p.x, r.x)
      && Math.min(p.y, r.y) <= q.y && q.y <= Math.max(p.y, r.y);
  };

  const o1 = orientation(a, b, c);
  const o2 = orientation(a, b, d);
  const o3 = orientation(c, d, a);
  const o4 = orientation(c, d, b);

  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(a, c, b)) return true;
  if (o2 === 0 && onSegment(a, d, b)) return true;
  if (o3 === 0 && onSegment(c, a, d)) return true;
  if (o4 === 0 && onSegment(c, b, d)) return true;
  return false;
}

function isPolygonValid(candidatePoints = points) {
  if (candidatePoints.length < 3) return false;

  for (let i = 0; i < candidatePoints.length; i += 1) {
    const a = candidatePoints[i];
    const b = candidatePoints[(i + 1) % candidatePoints.length];

    for (let j = i + 1; j < candidatePoints.length; j += 1) {
      if (j === i || j === (i + 1) % candidatePoints.length || (i === 0 && j === candidatePoints.length - 1)) continue;
      const c = candidatePoints[j];
      const d = candidatePoints[(j + 1) % candidatePoints.length];
      if (segmentsIntersect(a, b, c, d)) {
        return false;
      }
    }
  }

  const area = candidatePoints.reduce((sum, point, index) => {
    const next = candidatePoints[(index + 1) % candidatePoints.length];
    return sum + (point.x * next.y - next.x * point.y);
  }, 0);

  return Math.abs(area) > 0.1;
}

function addPointAt(point) {
  if (!imageUrl || points.length >= MAX_POINTS) {
    if (points.length >= MAX_POINTS) {
      errorMessage.textContent = 'Максимум 4 точки для выделения.';
      errorMessage.classList.add('visible');
    }
    return;
  }

  const last = points[points.length - 1];
  if (last && Math.hypot(point.x - last.x, point.y - last.y) < 8) return;

  const nextPoints = [...points, point];
  if (nextPoints.length >= 3 && !isPolygonValid(nextPoints)) {
    errorMessage.textContent = 'Некорректное выделение границы. Попробуйте еще раз, пожалуйста';
    errorMessage.classList.add('visible');
    return;
  }

  points.push(point);
  redoStack = [];
  clearError();
  renderPoints();
  updateButtons();
}

function addPoint(event) {
  if (!imageUrl) return;
  const point = getSvgPoint(event);
  addPointAt(point);
}

function undoLastPoint() {
  if (!points.length) return;
  redoStack.push(points.pop());
  clearError();
  renderPoints();
  updateButtons();
}

function redoLastPoint() {
  if (!redoStack.length) return;
  points.push(redoStack.pop());
  clearError();
  renderPoints();
  updateButtons();
}

function handlePointerDown(event) {
  if (!imageUrl) return;

  const point = getSvgPoint(event);
  const hitIndex = points.findIndex((item) => Math.hypot(item.x - point.x, item.y - point.y) <= 16);

  if (hitIndex !== -1) {
    dragIndex = hitIndex;
    drawingCanvas.setPointerCapture?.(event.pointerId);
    return;
  }

  addPointAt(point);
}

function handlePointerMove(event) {
  if (dragIndex === null || !imageUrl) return;

  const nextPoint = getSvgPoint(event);
  const previousPoints = [...points];
  points[dragIndex] = nextPoint;

  if (points.length >= 3 && !isPolygonValid()) {
    points = previousPoints;
    errorMessage.textContent = 'Некорректное выделение границы. Попробуйте еще раз, пожалуйста';
    errorMessage.classList.add('visible');
    return;
  }

  clearError();
  renderPoints();
}

function handlePointerUp() {
  dragIndex = null;
}

soloImageInput.addEventListener('change', (event) => {
  setImage(event.target.files[0]);
});

replaceImageBtn.addEventListener('click', () => {
  soloImageInput.value = '';
  soloImageInput.click();
});

backHomeBtn.addEventListener('click', () => {
  window.location.href = '/';
});

analyzeBtn.addEventListener('click', async () => {
  if (points.length < 3) {
    errorMessage.textContent = 'Некорректное выделение границы. Попробуйте еще раз, пожалуйста';
    errorMessage.classList.add('visible');
    return;
  }
  clearError();

  if (!imageUrl) {
    setStatus('Сначала загрузите изображение.', 'error');
    return;
  }

  const raw = await fetch(imageUrl);
  if (!raw.ok) {
    errorMessage.textContent = 'Не удалось прочитать изображение.';
    errorMessage.classList.add('visible');
    return;
  }
  const blob = await raw.blob();
  const formData = new FormData();
  const polygon = points.map((point) => [point.x, point.y]);
  let bounds;
  try {
    bounds = polygonBounds(polygon, activeImage);
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.add('visible');
    return;
  }
  formData.append('image', blob, 'query.png');
  formData.append('x', String(bounds.x));
  formData.append('y', String(bounds.y));
  formData.append('w', String(bounds.w));
  formData.append('h', String(bounds.h));
  formData.append('topk', '10');
  await gallerySelectionReady;
  if (gallerySelect.disabled) {
    errorMessage.textContent = 'Галерея не готова. Загрузите минимум 10 изображений через /docs → /api/galleries.';
    errorMessage.classList.add('visible');
    return;
  }
  if (gallerySelect.value) formData.append('gallery_id', gallerySelect.value);

  try {
    const response = await fetch('/api/infer', {
      method: 'POST',
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || 'Ошибка распознавания');
    }
    showSoloResult(data);
  } catch (error) {
    errorMessage.textContent = error.message;
    errorMessage.classList.add('visible');
  }
});

undoBtn.addEventListener('click', undoLastPoint);
redoBtn.addEventListener('click', redoLastPoint);
drawingCanvas.addEventListener('pointerdown', handlePointerDown);
drawingCanvas.addEventListener('pointermove', handlePointerMove);
drawingCanvas.addEventListener('pointerup', handlePointerUp);
drawingCanvas.addEventListener('pointerleave', handlePointerUp);
updateButtons();
