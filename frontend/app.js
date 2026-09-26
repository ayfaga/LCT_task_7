const state = {
  selectedMode: null,
};

const requestsList = document.getElementById('requestsList');

const landingCards = document.querySelectorAll('.landing-card');

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
      requestsList.innerHTML = '<p>В разработке!</p>';
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

function handleChoice(route) {
  state.selectedMode = route;
  window.location.assign(route);
}

document.querySelectorAll('.landing-card').forEach((card) => {
  card.addEventListener('click', () => handleChoice(card.dataset.route));
});

loadRequests();
