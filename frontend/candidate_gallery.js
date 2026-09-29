/* Shared top-10 visual results for single and batch search. */
const CandidateGallery = (() => {
  const cosine = (item) => Number.isFinite(Number(item?.similarity))
    ? Number(item.similarity).toFixed(3) : '—';

  function galleryUrl(value) {
    if (typeof value !== 'string' || !value.startsWith('/api/galleries/')) return null;
    const url = new URL(value, window.location.origin);
    return url.origin === window.location.origin ? url.pathname : null;
  }

  function openLightbox(candidates, selected, opener) {
    const viewable = candidates.map((item, index) => ({ item, rank: index + 1 }))
      .filter(({ item }) => galleryUrl(item.image_url));
    let position = viewable.findIndex(({ item }) => item === selected);
    if (position < 0) return;

    document.querySelector('.candidate-lightbox')?.remove();
    const overlay = document.createElement('div');
    overlay.className = 'candidate-lightbox';
    overlay.innerHTML = '<div class="candidate-lightbox-backdrop"></div><section class="candidate-lightbox-dialog" role="dialog" aria-modal="true" aria-label="Просмотр кандидата"><div class="candidate-lightbox-bar"><strong class="candidate-lightbox-title"></strong><button class="candidate-lightbox-close" type="button" aria-label="Закрыть просмотр">×</button></div><img class="candidate-lightbox-image" alt=""><p class="candidate-lightbox-score"></p><div class="candidate-lightbox-actions"><button class="candidate-lightbox-prev" type="button">← Предыдущий</button><a class="candidate-lightbox-source" target="_blank" rel="noopener noreferrer">Исходный кадр ↗</a><button class="candidate-lightbox-next" type="button">Следующий →</button></div></section>';
    const picture = overlay.querySelector('.candidate-lightbox-image');
    const source = overlay.querySelector('.candidate-lightbox-source');
    function update() {
      const { item, rank } = viewable[position];
      overlay.querySelector('.candidate-lightbox-title').textContent = `Кандидат ${rank} · ID ${item.gallery_id}`;
      overlay.querySelector('.candidate-lightbox-score').textContent = `Сходство cosine: ${cosine(item)} · это не вероятность совпадения`;
      picture.src = galleryUrl(item.image_url);
      picture.alt = `Автомобиль-кандидат ${rank}, ID ${item.gallery_id}`;
      const original = galleryUrl(item.source_image_url);
      source.hidden = !original;
      if (original) source.href = original;
      overlay.querySelector('.candidate-lightbox-prev').disabled = position === 0;
      overlay.querySelector('.candidate-lightbox-next').disabled = position === viewable.length - 1;
    }
    function close() {
      document.removeEventListener('keydown', onKeyDown);
      overlay.remove();
      opener?.focus();
    }
    function onKeyDown(event) {
      if (event.key === 'Escape') close();
      if (event.key === 'ArrowLeft' && position > 0) { position -= 1; update(); }
      if (event.key === 'ArrowRight' && position < viewable.length - 1) { position += 1; update(); }
    }
    overlay.querySelector('.candidate-lightbox-backdrop').addEventListener('click', close);
    overlay.querySelector('.candidate-lightbox-close').addEventListener('click', close);
    overlay.querySelector('.candidate-lightbox-prev').addEventListener('click', () => { position -= 1; update(); });
    overlay.querySelector('.candidate-lightbox-next').addEventListener('click', () => { position += 1; update(); });
    document.addEventListener('keydown', onKeyDown);
    document.body.appendChild(overlay);
    update();
    overlay.querySelector('.candidate-lightbox-close').focus();
  }

  function render(ranked, accepted = []) {
    const candidates = Array.isArray(ranked) ? ranked.slice(0, 10) : [];
    const acceptedIds = new Set((Array.isArray(accepted) ? accepted : []).map((item) => item.gallery_id));
    const grid = document.createElement('div');
    grid.className = 'candidate-grid';
    candidates.forEach((item, index) => {
      const card = document.createElement('article');
      card.className = `candidate-card${acceptedIds.has(item.gallery_id) ? ' is-accepted' : ''}`;
      const url = galleryUrl(item.image_url);
      if (url) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'candidate-image-button';
        button.setAttribute('aria-label', `Рассмотреть кандидата ${index + 1}, ID ${item.gallery_id}`);
        const image = document.createElement('img');
        image.src = url;
        image.alt = `Автомобиль-кандидат ${index + 1}`;
        image.loading = 'lazy';
        image.decoding = 'async';
        image.addEventListener('error', () => { button.textContent = 'Изображение недоступно'; });
        button.appendChild(image);
        button.addEventListener('click', () => openLightbox(candidates, item, button));
        card.appendChild(button);
      } else {
        const missing = document.createElement('div');
        missing.className = 'candidate-image-missing';
        missing.textContent = 'Нет фото';
        card.appendChild(missing);
      }
      const caption = document.createElement('div');
      caption.className = 'candidate-caption';
      const rank = document.createElement('span');
      rank.className = 'candidate-rank';
      rank.textContent = `#${index + 1}`;
      const id = document.createElement('strong');
      id.title = String(item.gallery_id);
      id.textContent = String(item.gallery_id);
      const score = document.createElement('span');
      score.className = 'candidate-score';
      score.textContent = `cosine ${cosine(item)}`;
      caption.append(rank, id, score);
      card.appendChild(caption);
      if (acceptedIds.has(item.gallery_id)) {
        const badge = document.createElement('span');
        badge.className = 'candidate-accepted';
        badge.textContent = 'Принят';
        card.appendChild(badge);
      }
      grid.appendChild(card);
    });
    return grid;
  }

  return { render, galleryUrl };
})();
