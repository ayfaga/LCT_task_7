# Импорт исходного архива организатора без ручных кропов

Административный путь для большого ZIP, который не помещается в браузерный
upload. Модель, её препроцессинг и frozen refusal policy не меняются.

```text
dataset.zip
 ├─ test_gallery.csv (image_id,x,y,w,h) + images/<image_id>.jpg
 │   → BBox crop → 336×336 → CLS/L2 1024D → test_gallery.npz
 └─ test_query.csv + images/<image_id>.jpg
     → тот же препроцессинг → test_query.npz → поиск/отказ → 3 файла сдачи
```

Архив не распаковывается целиком. Читаются только нужные test JPEG и два CSV;
train/holdout не используются. Проверяются ID, дубликаты, наличие JPEG,
целочисленный BBox и его попадание в изображение, размер/формат JPEG,
SHA модели, размерность/конечность эмбеддингов и порядок экспорта. Отсутствие
BBox — ошибка, **не** разрешение использовать целый кадр вместо автомобиля.
Повреждённый JPEG/CRC также останавливает импорт. Частичные выходы не
считаются готовыми: нужен `import_manifest.json` со `status=completed`.

## Подготовка и запуск

Из корня клона, в окружении со штатными зависимостями backend и полным LFS-весом:

```sh
.venv/bin/python backend/ml/import_organizer_archive.py \
  --archive /absolute/path/to/dataset.zip \
  --artifact-dir model_artifacts/joint_l336 \
  --output /absolute/path/to/new_import --inspect-only

.venv/bin/python backend/ml/import_organizer_archive.py \
  --archive /absolute/path/to/dataset.zip \
  --artifact-dir model_artifacts/joint_l336 \
  --output /absolute/path/to/new_import \
  --device cpu --threads 4 --batch-size 4
```

Для CUDA использовать `--device cuda` и подобрать batch по свободной VRAM.
CPU на Mac работает, но полный проход существенно медленнее GPU.
`--output` должен отсутствовать или быть пустым: чужие выходы не перезаписываем.
`--limit-per-split 10` — только smoke; его файлы **не сдавать** вместо полного
экспорта. `--inspect-only` проверяет таблицы/наличие файлов, но не декодирует
все изображения и не заменяет полный проход.

Выход: два CSV, `test_gallery.npz`, `test_query.npz`, `import_manifest.json`
и `submission/{submission.csv,embeddings.npy,candidates.csv,manifest.json}`.
NPZ содержит ID, FP32 эмбеддинги, model version и SHA, совместимые с runtime.
Это не оценка скрытого качества: test-меток здесь нет.

## Подключение подготовленной галереи

Для local Python перед запуском:

```sh
export LCT_ML_GALLERY_PATH=/absolute/path/to/new_import/test_gallery.npz
.venv/bin/python quick_start.py
```

Для Docker есть отдельный opt-in override:

```sh
export LCT_ORGANIZER_GALLERY_NPZ=/absolute/path/to/new_import/test_gallery.npz
docker compose -f docker-compose.yml -f docker-compose.organizer-gallery.yml config
docker compose -f docker-compose.yml -f docker-compose.organizer-gallery.yml up -d
```

Он монтирует только NPZ read-only, не весь 6.7 ГБ архив. На `/ready` проверять
`gallery_ready=true`, `gallery_size=750` для текущей organizer gallery;
поиск **без поля `gallery_id`** использует её (строку `default` в это поле
не отправлять: оно предназначено для ID пользовательских галерей).
При смене энкодера эту галерею
нужно пересчитать: runtime отвергает несовместимую версию/вес.

**Граница UI:** «Пополнение базы данных» по-прежнему сохраняет файлы в
отдельном хранилище, не превращает этот большой архив в default gallery.
Браузерный gallery-import с `manifest.csv` теперь потоково обрабатывает ZIP и
не имеет фиксированных лимитов 100 MiB/200 изображений, но по-прежнему зависит
от свободного диска, времени индексации и ресурсов точного поиска. Для
многогигабайтного исходного ZIP рекомендуем описанный возобновляемый CLI.
Пустая default gallery остаётся
безопасным поведением свежего запуска, пока администратор явно не подключит NPZ.

## Фактически выполненная сквозная проверка на Mac · 27.09

Архив SHA `a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613`:
полностью обработаны **750 gallery + 1110 query** исходных JPEG+BBox,
без распаковки train. CPU/4 threads/batch4, FP32. Этап после загрузки
энкодера занял 1463.72 с (включая export и SHA архива).

Все ID и 1860 векторов **побайтово совпали** с прежними cached-crop
выходами (max abs diff0.0). Три файла сдачи также совпали по SHA;
round-trip top10/policy/order прошёл, embeddings1860×1024, candidates2434
строки, 33 отказа. Это не новый score: test labels неизвестны.

Полная галерея read-only подключена к существующему локальному ML-контейнеру:
оба `/ready`=200, gallery_size750; интерфейс получает «Основная (750)».
Пользовательская gallery10 сохранена. Шесть исходных test JPEG+BBox прошли
реальный backend→ML путь: top10 и accepted совпали с экспортом, включая
три отказа. Общий архивный проход использовал offline export; **не все
1110 запросов** повторно прогонялись через HTTP.

Машинные доказательства:
[`organizer_archive_validation_mac_20260927.json`](organizer_archive_validation_mac_20260927.json).
Локальная копия готовых NPZ/CSV/экспорта:
`/Users/hissikly/Documents/lct_auto_detector/data/derived/organizer_archive_import_20260927/`.
Данные и сам архив в Git не отправлялись. Интеграционный путь CLI+runtime
проверен; кнопка replenishment, описанная выше, не стала индексатором.

Повторить HTTP parity против собственного полного экспорта:

```sh
.venv/bin/python backend/ml/check_api_export_parity.py \
  --url http://127.0.0.1:8000 \
  --export /absolute/path/to/new_import/submission \
  --organizer-archive /absolute/path/to/dataset.zip \
  --output /absolute/path/to/new_http_parity.json
```
