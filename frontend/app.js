const state = {
  selectedMode: null,
};

const modal = document.getElementById('modeModal');
const startBtn = document.getElementById('startBtn');
const closeModalBtn = document.getElementById('closeModalBtn');
const historyBtn = document.getElementById('historyBtn');
const requestsList = document.getElementById('requestsList');

function openModal() {
  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');
}

function closeModal() {
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
}

function addRequestCard(item) {
  const card = document.createElement('article');
  card.className = 'request-card';
  card.innerHTML = `
    <div class="meta">
      <span>#${item.id}</span>
      <span>${item.status || 'pending'}</span>
    </div>
    <h3>${item.title}</h3>
    <p>${item.description || 'Нет описания'}</p>
    <p>Mode: ${item.image_name ? 'uploaded' : 'n/a'}</p>
    <button type="button" data-delete-id="${item.id}">Удалить</button>
  `;
  requestsList.appendChild(card);
}

async function loadRequests() {
  try {
    const response = await fetch('/api/requests');
    const data = await response.json();
    requestsList.innerHTML = '';
    if (!Array.isArray(data) || !data.length) {
      requestsList.innerHTML = '<p>Пока нет запросов.</p>';
      return;
    }
    data.forEach(addRequestCard);
    bindDeleteButtons();
  } catch (error) {
    requestsList.innerHTML = '<p>Не удалось загрузить запросы.</p>';
    console.error(error);
  }
}

async function deleteRequest(id) {
  try {
    const response = await fetch(`/api/requests/${id}`, { method: 'DELETE' });
    if (!response.ok) {
      throw new Error('Failed to delete');
    }
    await loadRequests();
  } catch (error) {
    console.error(error);
  }
}

function bindDeleteButtons() {
  document.querySelectorAll('[data-delete-id]').forEach((button) => {
    button.addEventListener('click', () => deleteRequest(button.dataset.deleteId));
  });
}

function handleChoice(mode) {
  state.selectedMode = mode;
  closeModal();

  const targetRoute = mode === 'solo' ? '/solo' : '/many';
  window.location.assign(targetRoute);
}

startBtn.addEventListener('click', openModal);
closeModalBtn.addEventListener('click', closeModal);
historyBtn.addEventListener('click', () => {
  document.getElementById('historyPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
});

modal.addEventListener('click', (event) => {
  if (event.target.dataset.close === 'true') {
    closeModal();
  }
});

document.querySelectorAll('.choice-card').forEach((card) => {
  card.addEventListener('click', () => handleChoice(card.dataset.mode));
});

loadRequests();
