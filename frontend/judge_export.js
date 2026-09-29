(() => {
  const archive = document.getElementById("judgeArchive");
  const start = document.getElementById("judgeStart");
  const status = document.getElementById("judgeStatus");
  const details = document.getElementById("judgeDetails");
  const progress = document.getElementById("judgeProgress");
  const downloads = document.getElementById("judgeDownloads");
  if (!archive || !start || !status || !details || !progress || !downloads) return;

  let enabled = false;
  let timer = null;

  function message(value, extra = "") {
    status.textContent = value;
    details.textContent = extra;
  }

  function remember(jobId) {
    try { sessionStorage.setItem("lct-judge-export-job", jobId); } catch (_) {}
  }

  function remembered() {
    try { return sessionStorage.getItem("lct-judge-export-job"); } catch (_) { return null; }
  }

  function showDownloads(jobId) {
    downloads.hidden = false;
    for (const link of downloads.querySelectorAll("[data-file]")) {
      link.href = `/api/judge-export/jobs/${jobId}/files/${link.dataset.file}`;
    }
  }

  async function checkJob(jobId) {
    if (!/^[0-9a-f]{32}$/.test(jobId)) return;
    try {
      const response = await fetch(`/api/judge-export/jobs/${jobId}`, {cache: "no-store"});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      const done = result.processed || {};
      const counts = result.counts || {};
      const total = (counts.gallery || 0) + (counts.query || 0);
      const completed = (done.gallery || 0) + (done.query || 0);
      progress.hidden = !total;
      progress.value = total ? Math.round(100 * completed / total) : 0;
      if (result.state === "completed") {
        message("Готово: три файла прошли внутреннюю проверку.",
                `Обработано ${counts.gallery} кадров галереи и ${counts.query} запросов. Скачайте также manifest.json с SHA256.`);
        showDownloads(jobId);
        start.disabled = !enabled;
        return;
      }
      if (["failed", "interrupted"].includes(result.state)) {
        message("Экспорт не завершён.", result.error || "Проверьте входные данные и журнал сервера.");
        start.disabled = !enabled;
        return;
      }
      message(result.state === "queued" ? "Задача ожидает начала…" : "Модель обрабатывает кадры…",
              total ? `Галерея ${done.gallery || 0}/${counts.gallery}; запросы ${done.query || 0}/${counts.query}.` : "Пожалуйста, подождите.");
      timer = setTimeout(() => checkJob(jobId), 2000);
    } catch (error) {
      message("Не удалось узнать состояние задачи.", error.message);
      timer = setTimeout(() => checkJob(jobId), 5000);
    }
  }

  function upload(file) {
    if (timer) clearTimeout(timer);
    downloads.hidden = true;
    start.disabled = true;
    progress.hidden = false;
    progress.value = 0;
    message("Передаём ZIP на локальный сервер…", "Данные не загружаются во внешние сервисы.");
    const request = new XMLHttpRequest();
    request.open("POST", "/api/judge-export/jobs");
    request.setRequestHeader("Content-Type", "application/zip");
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) progress.value = Math.round(100 * event.loaded / event.total);
    };
    request.onload = () => {
      let result;
      try { result = JSON.parse(request.responseText); } catch (_) { result = {}; }
      if (request.status !== 202 || !/^[0-9a-f]{32}$/.test(result.job_id || "")) {
        message("Не удалось принять архив.", result.detail || `HTTP ${request.status}. Проверьте ZIP и свободное место.`);
        start.disabled = !enabled;
        return;
      }
      remember(result.job_id);
      progress.value = 0;
      checkJob(result.job_id);
    };
    request.onerror = () => { message("Соединение прервалось при передаче архива."); start.disabled = !enabled; };
    request.send(file);
  }

  archive.addEventListener("change", () => { start.disabled = !enabled || !archive.files.length; });
  start.addEventListener("click", () => {
    const file = archive.files[0];
    if (!file || !/\.zip$/i.test(file.name)) {
      message("Выберите ZIP-архив с двумя CSV и исходными изображениями.");
      return;
    }
    upload(file);
  });

  fetch("/api/judge-export/config", {cache: "no-store"})
    .then((response) => response.json())
    .then((config) => {
      enabled = Boolean(config.enabled);
      start.disabled = !enabled || !archive.files.length;
      if (!enabled) {
        message("Режим выключен администратором.",
                "Для локальной проверки установите LCT_JUDGE_EXPORT_ENABLED=1 и перезапустите оба сервиса. Подробности — docs/JUDGE_EXPORT.md.");
        return;
      }
      const last = remembered();
      if (last && /^[0-9a-f]{32}$/.test(last)) checkJob(last);
      else message("Готовы принять архив.", "Выберите ZIP и нажмите кнопку слева.");
    })
    .catch((error) => message("Не удалось проверить режим экспорта.", error.message));
})();
