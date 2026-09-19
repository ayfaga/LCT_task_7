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

let imageUrl = '';
let points = [];
let redoStack = [];
let activeImage = null;

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

function isPolygonValid() {
  if (points.length < 3) return false;

  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];

    for (let j = i + 1; j < points.length; j += 1) {
      if (j === i || j === (i + 1) % points.length || (i === 0 && j === points.length - 1)) continue;
      const c = points[j];
      const d = points[(j + 1) % points.length];
      if (segmentsIntersect(a, b, c, d)) {
        return false;
      }
    }
  }

  const area = points.reduce((sum, point, index) => {
    const next = points[(index + 1) % points.length];
    return sum + (point.x * next.y - next.x * point.y);
  }, 0);

  return Math.abs(area) > 0.1;
}

function addPoint(event) {
  if (!imageUrl) return;
  const point = getSvgPoint(event);
  const last = points[points.length - 1];
  if (last && Math.hypot(point.x - last.x, point.y - last.y) < 8) return;

  const nextPoints = [...points, point];
  if (nextPoints.length >= 4) {
    const tempPoints = [...nextPoints];
    const n = tempPoints.length;
    for (let i = 0; i < n; i += 1) {
      const a = tempPoints[i];
      const b = tempPoints[(i + 1) % n];
      for (let j = i + 1; j < n; j += 1) {
        if (j === i || j === (i + 1) % n || (i === 0 && j === n - 1)) continue;
        const c = tempPoints[j];
        const d = tempPoints[(j + 1) % n];
        if (segmentsIntersect(a, b, c, d)) {
          errorMessage.classList.add('visible');
          return;
        }
      }
    }
  }

  points.push(point);
  redoStack = [];
  updateButtons();

  if (points.length >= 3 && !isPolygonValid()) {
    points.pop();
    errorMessage.classList.add('visible');
    redoStack = [];
    updateButtons();
    return;
  }

  clearError();
  renderPoints();
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

analyzeBtn.addEventListener('click', () => {
  if (points.length < 3) {
    errorMessage.textContent = 'Некорректное выделение границы. Попробуйте еще раз, пожалуйста';
    errorMessage.classList.add('visible');
    return;
  }
  clearError();
});

undoBtn.addEventListener('click', undoLastPoint);
redoBtn.addEventListener('click', redoLastPoint);
drawingCanvas.addEventListener('click', addPoint);
updateButtons();
