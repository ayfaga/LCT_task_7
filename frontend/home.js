const galleryStatus = document.getElementById('homeGalleryStatus');
async function showCommonGalleryStatus() {
  try {
    const response = await fetch('/api/common-gallery');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const gallery = await response.json();
    const count = gallery.image_count || 0;
    galleryStatus.className = `home-gallery-state ${gallery.search_ready ? 'is-ready' : 'is-waiting'}`;
    galleryStatus.textContent = gallery.search_ready
      ? `● Готова к поиску · ${count} автомобилей`
      : gallery.state === 'building'
        ? `◌ Индексируем фотографии · ${gallery.processed || 0} из ${gallery.pending_count || 0}`
        : `○ Пока ${count} автомобилей · для поиска нужно минимум ${gallery.minimum_for_search || 10}`;
  } catch (_) {
    galleryStatus.className = 'home-gallery-state is-waiting';
    galleryStatus.textContent = 'Статус галереи временно недоступен';
  }
}
showCommonGalleryStatus();
