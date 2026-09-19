const imageList = document.getElementById('imageList');
const coordList = document.getElementById('coordList');
const backHomeBtn = document.getElementById('backHomeBtn');
const renameModal = document.getElementById('renameModal');
const previewModal = document.getElementById('previewModal');
const renameInput = document.getElementById('renameInput');
const extensionTrigger = document.getElementById('extensionTrigger');
const extensionOptions = document.getElementById('extensionOptions');
const previewContent = document.getElementById('previewContent');
const renameCloseBtn = document.getElementById('renameCloseBtn');
const cancelRenameBtn = document.getElementById('cancelRenameBtn');
const saveRenameBtn = document.getElementById('saveRenameBtn');
const submitManyBtn = document.getElementById('submitManyBtn');

const state = {
  activeRename: null,
  fileStore: {
    image: [],
    coord: [],
  },
};

const extensionMap = {
  png: ['svg', 'jpg'],
  jpg: ['svg', 'png'],
  jpeg: ['svg', 'png'],
  txt: ['csv', 'json'],
  csv: ['txt', 'json'],
  json: ['txt', 'csv'],
};

function fileNameWithoutExt(fileName) {
  return fileName.includes('.') ? fileName.slice(0, fileName.lastIndexOf('.')) : fileName;
}

function fileExt(fileName) {
  return fileName.includes('.') ? fileName.slice(fileName.lastIndexOf('.') + 1).toLowerCase() : '';
}

function renderFileList(type) {
  const list = type === 'image' ? imageList : coordList;
  list.innerHTML = '';
  const items = state.fileStore[type];

  items.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'file-row';
    row.innerHTML = `
      <div class="file-name">${item.name}</div>
      <div class="file-actions">
        <button class="file-action-btn" type="button" data-action="preview" data-type="${type}" data-index="${index}" title="Предпросмотр">◉</button>
        <button class="file-action-btn" type="button" data-action="rename" data-type="${type}" data-index="${index}" title="Переименовать">✎</button>
        <button class="file-action-btn" type="button" data-action="delete" data-type="${type}" data-index="${index}" title="Удалить">🗑</button>
      </div>
    `;
    list.appendChild(row);
  });

  if (!items.length) {
    const empty = document.createElement('div');
    empty.className = 'file-name';
    empty.textContent = 'Файлы не загружены';
    empty.style.color = '#5a6d87';
    list.appendChild(empty);
  }
}

function openPreview(item) {
  previewContent.innerHTML = '';
  if (item.kind === 'image') {
    const img = document.createElement('img');
    img.src = item.preview;
    img.alt = item.name;
    previewContent.appendChild(img);
  } else {
    const pre = document.createElement('pre');
    pre.textContent = item.text || 'Пустой файл';
    previewContent.appendChild(pre);
  }
  previewModal.classList.remove('hidden');
  previewModal.setAttribute('aria-hidden', 'false');
}

function closePreview() {
  previewModal.classList.add('hidden');
  previewModal.setAttribute('aria-hidden', 'true');
}

function openRenameModal(type, index) {
  state.activeRename = { type, index };
  const item = state.fileStore[type][index];
  renameInput.value = fileNameWithoutExt(item.name);
  const ext = fileExt(item.name) || 'txt';
  extensionTrigger.textContent = ext;
  fillExtensionOptions(ext);
  renameModal.classList.remove('hidden');
  renameModal.setAttribute('aria-hidden', 'false');
}

function fillExtensionOptions(currentExt) {
  const options = extensionMap[currentExt] || ['txt', 'csv'];
  extensionOptions.innerHTML = '';
  options.forEach((ext) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = ext;
    btn.addEventListener('click', () => {
      extensionTrigger.textContent = ext;
      extensionOptions.classList.remove('open');
    });
    extensionOptions.appendChild(btn);
  });
}

function closeRenameModal() {
  renameModal.classList.add('hidden');
  renameModal.setAttribute('aria-hidden', 'true');
  state.activeRename = null;
}

function saveRename() {
  if (!state.activeRename) return;
  const { type, index } = state.activeRename;
  const item = state.fileStore[type][index];
  const newBase = renameInput.value.trim() || fileNameWithoutExt(item.name);
  const newExt = extensionTrigger.textContent.trim();
  item.name = `${newBase}.${newExt}`;
  closeRenameModal();
  renderFileList(type);
}

function handleFileInput(type, fileList) {
  Array.from(fileList).forEach((file) => {
    const base = {
      name: file.name,
      kind: type === 'image' ? 'image' : 'txt',
      preview: type === 'image' ? URL.createObjectURL(file) : '',
      text: type === 'image' ? '' : 'Содержимое файла: ' + file.name,
    };
    state.fileStore[type].push(base);
  });
  renderFileList(type);
}

imageList.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;

  const action = button.dataset.action;
  const index = Number(button.dataset.index);
  const item = state.fileStore.image[index];

  if (action === 'preview') openPreview(item);
  if (action === 'rename') openRenameModal('image', index);
  if (action === 'delete') {
    state.fileStore.image.splice(index, 1);
    renderFileList('image');
  }
});

coordList.addEventListener('click', (event) => {
  const button = event.target.closest('button[data-action]');
  if (!button) return;

  const action = button.dataset.action;
  const index = Number(button.dataset.index);
  const item = state.fileStore.coord[index];

  if (action === 'preview') openPreview(item);
  if (action === 'rename') openRenameModal('coord', index);
  if (action === 'delete') {
    state.fileStore.coord.splice(index, 1);
    renderFileList('coord');
  }
});

document.querySelectorAll('.add-file-btn').forEach((button) => {
  button.addEventListener('click', () => {
    const type = button.dataset.type === 'image' ? 'image' : 'coord';
    const input = document.createElement('input');
    input.type = 'file';
    input.multiple = true;
    input.accept = type === 'image' ? 'image/*' : '.txt,.csv,.json';
    input.addEventListener('change', (event) => handleFileInput(type, event.target.files));
    input.click();
  });
});

backHomeBtn.addEventListener('click', () => {
  window.location.href = '/';
});

renameCloseBtn.addEventListener('click', closeRenameModal);
cancelRenameBtn.addEventListener('click', closeRenameModal);
saveRenameBtn.addEventListener('click', saveRename);

extensionTrigger.addEventListener('click', () => {
  extensionOptions.classList.toggle('open');
});

previewModal.addEventListener('click', (event) => {
  if (event.target.dataset.close === 'true') closePreview();
});

renameModal.addEventListener('click', (event) => {
  if (event.target.dataset.close === 'true') closeRenameModal();
});

submitManyBtn.addEventListener('click', () => {
  const hasFiles = state.fileStore.image.length > 0 || state.fileStore.coord.length > 0;
  if (!hasFiles) {
    alert('Загрузите хотя бы один файл для отправки.');
    return;
  }
  alert('Файлы готовы к отправке.');
});

renderFileList('image');
renderFileList('coord');
