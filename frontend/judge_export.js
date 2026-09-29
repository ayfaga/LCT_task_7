(() => {
  const archive = document.getElementById("judgeArchive");
  const start = document.getElementById("judgeStart");
  const status = document.getElementById("judgeStatus");
  const details = document.getElementById("judgeDetails");
  const progress = document.getElementById("judgeProgress");
  const downloads = document.getElementById("judgeDownloads");
  const abort = document.getElementById("judgeAbort");
  const combined = document.getElementById("judgeCombined");
  const components = document.getElementById("judgeComponents");
  const partial = document.getElementById("judgePartial");
  const galleryCsv = document.getElementById("judgeGalleryCsv");
  const queryCsv = document.getElementById("judgeQueryCsv");
  const archiveName = document.getElementById("judgeArchiveName");
  const galleryCsvName = document.getElementById("judgeGalleryCsvName");
  const queryCsvName = document.getElementById("judgeQueryCsvName");
  const imageFiles = {gallery: document.getElementById("judgeGalleryFiles"),
                      query: document.getElementById("judgeQueryFiles")};
  const imageFolders = {gallery: document.getElementById("judgeGalleryFolder"),
                        query: document.getElementById("judgeQueryFolder")};
  const summaries = {gallery: document.getElementById("judgeGallerySummary"),
                     query: document.getElementById("judgeQuerySummary")};
  if (!archive || !start || !status || !details || !progress || !downloads ||
      !combined || !components || !galleryCsv || !queryCsv) return;

  const selected = {gallery: [], query: []};
  let enabled = false;
  let timer = null;

  function mode() {
    return document.querySelector('input[name="judgeMode"]:checked')?.value || "combined";
  }

  function message(value, extra = "") {
    status.textContent = value;
    details.textContent = extra;
  }

  function updateButton() {
    start.disabled = !enabled || (mode() === "combined" ? !archive.files.length :
      !galleryCsv.files.length || !queryCsv.files.length ||
      !selected.gallery.length || !selected.query.length);
  }

  function updateFileName(input, target) {
    target.textContent = input.files[0] ? `Выбран: ${input.files[0].name}` : "Файл не выбран.";
  }

  function remember(jobId) {
    try { sessionStorage.setItem("lct-judge-export-job", jobId); } catch (_) {}
  }

  function forget() {
    try { sessionStorage.removeItem("lct-judge-export-job"); } catch (_) {}
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
        abort.hidden = true;
        const isPartial = result.scope?.startsWith("PARTIAL");
        message(isPartial ? "Готова только тестовая подвыборка — это не полная сдача." :
                "Готово: три файла прошли внутреннюю проверку.",
                `Обработано ${counts.gallery} кадров галереи и ${counts.query} запросов. Скачайте также manifest.json с SHA256.`);
        showDownloads(jobId);
        updateButton();
        return;
      }
      if (["failed", "interrupted"].includes(result.state)) {
        abort.hidden = true;
        message("Экспорт не завершён.", result.error || "Проверьте входные данные и журнал сервера.");
        updateButton();
        return;
      }
      if (result.state === "collecting") {
        abort.hidden = false;
        message("Незавершённая загрузка из четырёх полей.",
                "Повторите загрузку после отмены; уже полученные файлы сохранены для диагностики.");
        return;
      }
      abort.hidden = true;
      message(result.state === "queued" ? "Задача ожидает начала…" : "Модель обрабатывает кадры…",
              total ? `Галерея ${done.gallery || 0}/${counts.gallery}; запросы ${done.query || 0}/${counts.query}.` : "Пожалуйста, подождите.");
      timer = setTimeout(() => checkJob(jobId), 2000);
    } catch (error) {
      message("Не удалось узнать состояние задачи.", error.message);
      timer = setTimeout(() => checkJob(jobId), 5000);
    }
  }

  function uploadRaw(method, url, file, contentType, onProgress) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open(method, url);
      request.setRequestHeader("Content-Type", contentType);
      request.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) onProgress(event.loaded);
      };
      request.onload = () => {
        let result;
        try { result = JSON.parse(request.responseText); } catch (_) { result = {}; }
        if (request.status >= 200 && request.status < 300) resolve(result);
        else reject(new Error(result.detail || `HTTP ${request.status}`));
      };
      request.onerror = () => reject(new Error("Соединение прервалось при передаче файла"));
      request.send(file);
    });
  }

  function resetForUpload() {
    if (timer) clearTimeout(timer);
    downloads.hidden = true;
    start.disabled = true;
    progress.hidden = false;
    progress.value = 0;
  }

  async function uploadCombined(file) {
    resetForUpload();
    message("Передаём готовый ZIP на локальный сервер…", "Данные не загружаются во внешние сервисы.");
    try {
      const result = await uploadRaw("POST", "/api/judge-export/jobs", file, "application/zip",
                                     (loaded) => { progress.value = Math.round(100 * loaded / file.size); });
      if (!/^[0-9a-f]{32}$/.test(result.job_id || "")) throw new Error("Сервер не вернул ID задачи");
      remember(result.job_id);
      progress.value = 0;
      checkJob(result.job_id);
    } catch (error) {
      message("Не удалось принять архив.", `${error.message}. Проверьте ZIP и свободное место.`);
      updateButton();
    }
  }

  function sourceKind(file) {
    if (/\.zip$/i.test(file.name)) return "zip";
    if (/^[0-9a-f]{32}$/.test(file.name.slice(0, 32)) &&
        /^\.jpe?g$/i.test(file.name.slice(32))) return "image";
    return null;
  }

  function appendFiles(split, input) {
    const files = Array.from(input.files || []);
    let skipped = 0;
    for (const file of files) {
      if (sourceKind(file)) selected[split].push(file);
      else skipped += 1;
    }
    summaries[split].textContent = `${selected[split].length} выбранных ZIP/JPEG` +
      (skipped ? `; пропущено ${skipped} файлов без имени <image_id>.jpg` : "") +
      ". Можно добавлять ещё ZIP, папки и JPEG.";
    input.value = "";
    updateButton();
  }

  async function uploadComponents() {
    resetForUpload();
    const files = [galleryCsv.files[0], ...selected.gallery, queryCsv.files[0], ...selected.query];
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    let sent = 0;
    let jobId = null;
    try {
      const duplicates = new Set();
      for (const split of ["gallery", "query"]) {
        duplicates.clear();
        for (const file of selected[split]) {
          if (sourceKind(file) === "image") {
            const id = file.name.slice(0, 32);
            if (duplicates.has(id)) throw new Error(`Повтор изображения ${id} в ${split}`);
            duplicates.add(id);
          }
        }
      }
      const created = await fetch("/api/judge-export/components", {method: "POST"});
      const session = await created.json();
      if (!created.ok || !/^[0-9a-f]{32}$/.test(session.job_id || ""))
        throw new Error(session.detail || "Не удалось создать задачу");
      jobId = session.job_id;
      remember(jobId);
      abort.hidden = false;
      const uploadOne = async (method, url, file, contentType, label) => {
        message(`Загружаем ${label}…`, `${Math.round(sent / 1048576)} из ${Math.round(totalBytes / 1048576)} МиБ передано.`);
        await uploadRaw(method, url, file, contentType, (loaded) => {
          progress.value = totalBytes ? Math.round(100 * (sent + loaded) / totalBytes) : 0;
        });
        sent += file.size;
      };
      for (const split of ["gallery", "query"]) {
        const csvFile = split === "gallery" ? galleryCsv.files[0] : queryCsv.files[0];
        await uploadOne("PUT", `/api/judge-export/components/${jobId}/${split}/csv`,
                        csvFile, "text/csv", `CSV ${split}`);
        for (const file of selected[split]) {
          const kind = sourceKind(file);
          const suffix = kind === "zip" ? "zip" : `images/${file.name.slice(0, 32)}`;
          await uploadOne(kind === "zip" ? "POST" : "PUT",
                          `/api/judge-export/components/${jobId}/${split}/${suffix}`,
                          file, kind === "zip" ? "application/zip" : "image/jpeg", file.name);
        }
      }
      const response = await fetch(`/api/judge-export/components/${jobId}/start?allow_partial=${Boolean(partial.checked)}`,
                                   {method: "POST"});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      remember(jobId);
      abort.hidden = true;
      progress.value = 0;
      checkJob(jobId);
    } catch (error) {
      if (jobId) {
        try { await fetch(`/api/judge-export/components/${jobId}/abort`, {method: "POST"}); } catch (_) {}
      }
      abort.hidden = true;
      forget();
      message("Не удалось собрать набор из четырёх полей.", error.message);
      updateButton();
    }
  }

  for (const [input, target] of [[archive, archiveName], [galleryCsv, galleryCsvName],
                                 [queryCsv, queryCsvName]]) {
    input.addEventListener("change", () => {
      updateFileName(input, target);
      updateButton();
    });
  }
  for (const split of ["gallery", "query"]) {
    imageFiles[split].addEventListener("change", () => appendFiles(split, imageFiles[split]));
    imageFolders[split].addEventListener("change", () => appendFiles(split, imageFolders[split]));
  }
  for (const radio of document.querySelectorAll('input[name="judgeMode"]')) {
    radio.addEventListener("change", () => {
      combined.hidden = mode() !== "combined";
      components.hidden = mode() !== "components";
      updateButton();
    });
  }
  start.addEventListener("click", () => {
    if (mode() === "components") {
      uploadComponents();
      return;
    }
    const file = archive.files[0];
    if (!file || !/\.zip$/i.test(file.name)) {
      message("Выберите готовый ZIP с двумя CSV и исходными изображениями.");
      return;
    }
    uploadCombined(file);
  });
  abort.addEventListener("click", async () => {
    const jobId = remembered();
    if (!jobId || !/^[0-9a-f]{32}$/.test(jobId)) return;
    try {
      const response = await fetch(`/api/judge-export/components/${jobId}/abort`, {method: "POST"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      forget();
      abort.hidden = true;
      message("Загрузка остановлена.", "Полученные ранее файлы оставлены на сервере для диагностики.");
      updateButton();
    } catch (error) {
      message("Не удалось остановить загрузку.", error.message);
    }
  });

  fetch("/api/judge-export/config", {cache: "no-store"})
    .then((response) => response.json())
    .then((config) => {
      enabled = Boolean(config.enabled);
      updateButton();
      if (!enabled) {
        message("Режим выключен администратором.",
                "Для локальной проверки установите LCT_JUDGE_EXPORT_ENABLED=1 и перезапустите оба сервиса. Подробности — docs/JUDGE_EXPORT.md.");
        return;
      }
      const last = remembered();
      if (last && /^[0-9a-f]{32}$/.test(last)) checkJob(last);
      else message("Готовы принять данные.", "Выберите готовый ZIP или заполните четыре отдельных поля.");
    })
    .catch((error) => message("Не удалось проверить режим экспорта.", error.message));
})();
