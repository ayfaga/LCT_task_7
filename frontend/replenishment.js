const imageUploadList = document.getElementById('imageUploadList');
const textUploadList = document.getElementById('textUploadList');
const uploadStatus = document.getElementById('uploadStatus');
const archiveProgressShell = document.getElementById('archiveProgressShell');
const archiveProgressFill = document.getElementById('archiveProgressFill');
const archiveProgressText = document.getElementById('archiveProgressText');
const backHomeBtn = document.getElementById('backHomeBtn');

const state = { images: [], texts: [] };
const bucket = (kind) => kind === 'image' ? state.images : state.texts;
const escapeHtml = (value) => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#039;');

function setStatus(message, kind = 'info') {
  uploadStatus.className = `upload-status ${kind}`;
  uploadStatus.textContent = message;
}

async function loadExistingEntries() {
  try {
    const response = await fetch('/api/replenishment/files');
    if (!response.ok) throw new Error('Не удалось получить список файлов');
    const files = await response.json();
    state.images = files.filter((file) => file.kind === 'image');
    state.texts = files.filter((file) => file.kind === 'text');
    render();
  } catch (error) { setStatus(error.message, 'error'); }
}

function progressMarkup(item) {
  if (item.status === 'uploading') return '<div class="row-progress"><span style="width:55%"></span></div><small>Загружается в базу данных…</small>';
  if (item.status === 'saved') return '<div class="row-progress complete"><span style="width:100%"></span></div><small>Сохранено в базе данных</small>';
  if (item.status === 'error') return `<div class="row-progress failed"><span style="width:100%"></span></div><small>Ошибка: ${escapeHtml(item.error || 'не удалось загрузить')}</small>`;
  return '<div class="row-progress complete"><span style="width:100%"></span></div><small>В базе данных</small>';
}

function renderList(container, items, kind) {
  container.innerHTML = '';
  if (!items.length) { container.innerHTML = '<div class="replenishment-empty">Файлы ещё не загружены</div>'; return; }
  items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'replenishment-item clickable-row';
    row.innerHTML = `<div class="replenishment-item-main"><div class="replenishment-item-name">${escapeHtml(item.name)}</div>${progressMarkup(item)}</div><div class="file-actions"><button class="file-action-btn" type="button" data-action="rename" title="Переименовать">✎</button><button class="file-action-btn" type="button" data-action="delete" title="Удалить">🗑</button></div>`;
    row.addEventListener('click', (event) => {
      const action = event.target.closest('[data-action]')?.dataset.action;
      if (action === 'rename') openRename(item);
      else if (action === 'delete') deleteFile(item);
      else openPreview(item);
    });
    container.appendChild(row);
  });
}
function render() { renderList(imageUploadList, state.images, 'image'); renderList(textUploadList, state.texts, 'text'); }

async function uploadFile(file, kind) {
  const item = { id: `pending-${Date.now()}-${Math.random()}`, name: file.name, kind, file, url: URL.createObjectURL(file), status: 'uploading' };
  bucket(kind).unshift(item); render();
  try {
    const body = new FormData(); body.append('kind', kind); body.append('file', file, file.name);
    const response = await fetch('/api/replenishment/files', { method: 'POST', body });
    const saved = await response.json(); if (!response.ok) throw new Error(saved.detail || 'Не удалось сохранить файл');
    Object.assign(item, saved, { status: 'saved', file: null }); setStatus(`«${saved.name}» сохранён в базе данных.`, 'success');
  } catch (error) { item.status = 'error'; item.error = error.message; setStatus(`«${file.name}» не загружен: ${error.message}`, 'error'); }
  render();
}
function openPicker(kind) {
  const input = document.createElement('input'); input.type = 'file'; input.multiple = true;
  input.accept = kind === 'image' ? 'image/*,.png,.jpg,.jpeg' : '.txt,.csv,.json,text/plain';
  input.onchange = (event) => Array.from(event.target.files || []).forEach((file) => uploadFile(file, kind)); input.click();
}

function ensureModal() {
  let modal = document.getElementById('fileModal'); if (modal) return modal;
  modal = document.createElement('div'); modal.id = 'fileModal'; modal.className = 'modal hidden';
  modal.innerHTML = '<div class="modal-backdrop"></div><div class="preview-modal-content"><button class="close-btn modal-close" type="button">×</button><div id="fileModalBody"></div></div>';
  document.body.appendChild(modal);
  modal.addEventListener('click', (event) => { if (event.target.classList.contains('modal-backdrop') || event.target.classList.contains('modal-close')) closeModal(); }); return modal;
}
function closeModal() { document.getElementById('fileModal')?.classList.add('hidden'); }
async function openPreview(item) {
  const modal = ensureModal(); const body = document.getElementById('fileModalBody'); body.innerHTML = '<p>Загрузка просмотра…</p>'; modal.classList.remove('hidden');
  if (item.kind === 'image') body.innerHTML = `<h3>${escapeHtml(item.name)}</h3><img src="${item.url}" alt="${escapeHtml(item.name)}">`;
  else try { const content = item.file ? await item.file.text() : await fetch(item.url).then((r) => r.text()); body.innerHTML = `<h3>${escapeHtml(item.name)}</h3><pre>${escapeHtml(content || 'Пустой файл')}</pre>`; } catch { body.innerHTML = '<p>Не удалось открыть текстовый файл.</p>'; }
}
function openRename(item) {
  const modal = ensureModal(); const body = document.getElementById('fileModalBody');
  body.innerHTML = `<h3>Переименовать файл</h3><form id="renameForm"><input id="renameValue" value="${escapeHtml(item.name)}" aria-label="Новое имя"><div class="modal-actions"><button class="ghost-btn modal-close" type="button">Отмена</button><button class="action-btn" type="submit">Сохранить</button></div></form>`; modal.classList.remove('hidden');
  body.querySelector('#renameForm').addEventListener('submit', async (event) => {
    event.preventDefault(); const name = body.querySelector('#renameValue').value.trim(); if (!name) return;
    try { const form = new FormData(); form.append('name', name); const response = await fetch(`/api/replenishment/files/${item.id}`, { method: 'PATCH', body: form }); const changed = await response.json(); if (!response.ok) throw new Error(changed.detail || 'Не удалось переименовать'); item.name = changed.name; closeModal(); render(); setStatus('Имя файла обновлено.', 'success'); } catch (error) { setStatus(error.message, 'error'); }
  });
}
async function deleteFile(item) {
  if (!confirm(`Удалить «${item.name}» из базы данных?`)) return;
  try { const response = await fetch(`/api/replenishment/files/${item.id}`, { method: 'DELETE' }); if (!response.ok) throw new Error('Не удалось удалить файл'); const items = bucket(item.kind); items.splice(items.indexOf(item), 1); render(); setStatus('Файл удалён из базы данных.', 'success'); } catch (error) { setStatus(error.message, 'error'); }
}
function uploadArchive() {
  const input = document.createElement('input'); input.type = 'file'; input.accept = '.zip,.tar,.tgz,.gz,.bz2,.xz';
  input.onchange = async (event) => { const file = event.target.files?.[0]; if (!file) return; archiveProgressShell.classList.remove('hidden'); archiveProgressFill.style.width = '25%'; archiveProgressText.textContent = 'Архив загружается и распаковывается…'; try { const body = new FormData(); body.append('archive', file); const response = await fetch('/api/replenishment/archive', { method: 'POST', body }); const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || 'Ошибка загрузки архива'); archiveProgressFill.style.width = '100%'; archiveProgressText.textContent = `Готово: в базу добавлено ${payload.count} файлов.`; await loadExistingEntries(); setStatus(`Все ${payload.count} подходящих файлов из архива сохранены.`, 'success'); } catch (error) { archiveProgressText.textContent = `Ошибка: ${error.message}`; setStatus(`Архив не загружен: ${error.message}`, 'error'); } };
  input.click();
}
document.querySelectorAll('[data-mode]').forEach((button) => button.addEventListener('click', () => openPicker(button.dataset.mode)));
document.getElementById('archiveUploadBtn').addEventListener('click', uploadArchive);
backHomeBtn.addEventListener('click', () => { window.location.href = '/'; });
loadExistingEntries();
