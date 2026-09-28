# Joint L336: воспроизведение поставки и проверка · 27.09.2026

Это журнал **проверенной** ветки, а не обещание hidden-score. Презентацию по
указанию пользователя здесь не создаём. Текущий вес и политика отказа
заморожены; эксперименты ниже не меняют их и не вскрывают holdout.

## Артефакты и происхождение

| Объект | Версия / SHA256 / размер |
|---|---|
| Организаторский `dataset.zip` | SHA256 `a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613`; не входит в Git/контейнер. |
| `objects.csv` после EDA | SHA256 `c5eabdd4c17e02265f65b90c0568d51ef7907e099b413ba397f694f6f40c661d`; train/dev/calibration/holdout ID: 1001/231/154/155. |
| CityFlow AIC21 Track2 train ZIP | SHA256 `70bfc5bcec322f2b7046281af4039ae4915510c5b4be1e05895fb38193b681d7`; источник и разрешения: `repro/README.md`. |
| Публичный DINOv2 ViT-L/14 pretrain | SHA256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`. |
| Выбранный train checkpoint | SHA256 `e1e3c03476d3b1af8eb9d9323b896d4e1e0859064597aaa66c79ee1b3e6e8d83`; не включён в inference-пакет. |
| `model_inference.pt` | Git LFS; 1 217 580 473 байта; SHA256 `507b4f2e34384e2366ab5364e147331831664493b35b40041507d88634da05f5`. |
| `boosting_policy.json` | SHA256 `02578b926977de01b7cdb109a4a86160653deb22075fda66f95c7d67366691d9`. |

Единственный актуальный `model_version`:
`dinov2-l14-l336-joint-cityflow-organizer-best19-20260925`.
Условия прав на внешние данные зафиксированы в `repro/README.md`; разрешение
на публичную поставку производного веса сообщено командой после согласования
с организатором. Это не самостоятельная юридическая экспертиза.

## Точный inference-контракт

Исходный JPEG/PNG + целочисленный `BBox=(x,y,w,h)` в пикселях полного кадра;
BBox непустой и внутри изображения. RGB-кроп уменьшается до max-side 448
Lanczos, затем дополняется до 336×336 bicubic с цветом `(123,116,103)` и
нормализуется ImageNet mean/std. Препроцессинг version
`organizer-bbox-rgb-max448-lanczos-pad-bicubic-imagenet-v1`, SHA256 реализации
`2e6187104dc6dfa166e50424e9ecb0314e6dce9982ba26708c00d0c32fda9287`.
Один ViT-L/14 forward → CLS 1024D FP32 → L2-норма → exact cosine/IP по
совместимой галерее. `ranked` — top-10 по убыванию cosine; `accepted` и
`candidates` — префикс, прошедший frozen hybrid policy:

```text
accepted_i = cosine_i >= 0.44 AND (rank_i < 3 OR booster_i >= 0.5)
```

В этой записи `rank_i` начинается с нуля; первые три позиции сохраняют
cosine-правило. Policy дополнительно проверяет монотонность и prefix.

Boosting использует только пять числовых признаков отсортированного top-10,
описанных в `backend/app/ml/boosting.py`; текстовые/номерные признаки не
добавлялись. `confidence` равен **сырому cosine, не вероятности**.
`status=no_confident_match` + пустой `accepted` — корректный отказ, не 500.
Ошибка BBox даёт 422; несовместимая версия/отсутствующий вес — `/ready` 503.

`POST /api/identify`, `/api/infer`, `/v1/search` принимают multipart `image`
и `x,y,w,h`, необязательный `gallery_id` и `topk`; возвращают `status`,
`model_version`, `gallery_id`, `ranked`, `accepted`, `candidates`, `threshold`,
`confidence_semantics`. `POST /v1/embeddings` возвращает нормированный
вектор и размерность. Полная схема — `/openapi.json`/`/docs` запущенного API.

Три файла сдачи: `submission.csv` — 10 gallery ID в убывающем cosine для
каждого query; `embeddings.npy` — FP32 `[1110 query, затем 750 gallery] × 1024`
в порядке строк исходных CSV; `candidates.csv` — принятые пары с числовым
cosine либо пустая строка отказа. Официальный scorer организатором не выдаётся:
проверен наш round-trip, не официальный F1. В веб-экспорте solo JSON содержит
полный API-ответ; many CSV содержит query/status/rank/gallery/cosine/accepted.

## Команды чистой поставки и границы проверки

```sh
git lfs install
git lfs pull
shasum -a 256 model_artifacts/joint_l336/model_inference.pt
python3 -m venv .venv
./.venv/bin/python quick_start.py
# другой вариант: docker compose up --build -d
```

Пустая установка стартует **без** default gallery. Для поиска загрузить
≥10 собственных изображений в версионированную галерею через API/браузер;
их изображения, эмбеддинги и JSON-метаданные переживают restart в SQLite и
`gallery_state` volume. Если нужна предвычисленная галерея, смонтировать
совместимый NPZ в ML-контейнер отдельным Compose override; host env var без
volume этого не сделает. Текущий лимит пользовательской галереи — 1000
снимков. Миллионный ANN ниже — **отдельный** сервисный нагрузочный тест.

Большой исходный ZIP организатора обслуживается отдельным административным
streaming-import: `backend/ml/import_organizer_archive.py`. Он читает
`image_id,x,y,w,h`, вычисляет эмбеддинги по исходным JPEG+BBox и проверяет
все три файла сдачи. Инструкция и opt-in Compose override:
[импорт архива](ORGANIZER_ARCHIVE_IMPORT.md). Кнопка «Пополнение базы данных»
по-прежнему только сохраняет файлы, не индексирует этот большой ZIP.

Полный исходный ZIP750+1110 обработан на Mac: все 1860 векторов и три
файла экспорта совпали побайтово с cached-crop поставкой. Default750
подключена к локальному Docker, `/ready`=200; 6 оригинальных JPEG+BBox
через backend→ML совпали с export, включая отказы. Машинный протокол:
`organizer_archive_validation_mac_20260927.json`. Это не mAP на test без меток.

Контрольные версии локального тестового окружения Mac: FastAPI 0.111.0,
PyTorch 2.8.0, NumPy 1.26.4, Pillow 10.4.0, pandas 2.3.2,
scikit-learn 1.7.2, FAISS 1.15.1. Для штатной установки фиксированные
версии берутся из `backend/requirements*.txt` и `repro/requirements.txt`;
локальная `.venv-test` **не** является подтверждением clean-room Linux.

Проверки текущей ветки:

```sh
.venv-test/bin/python -m pytest backend/tests -q
PYTHONPATH=backend .venv-test/bin/python -m pytest backend/ml/tests -q
PYTHONPATH=repro .venv-test/bin/python -m pytest repro/tests/test_joint_error_slices.py -q
node --test frontend/tests/export.test.js
node --check frontend/solo.js
node --check frontend/many.js
```

Совместный Python-прогон перечисленных групп: **33 passed**, JS **3 passed**.

Исторический локальный Docker smoke из **кэшированных** образов, HTTP и
test-export — в `CRITERIA_AUDIT_MAC_20260927.md`. Чистый Docker build раньше
упёрся в Docker Hub metadata; он не подтверждён. Официальное железо/FPS,
оценщик, публичная ссылка прототипа и финальная заявка проверяются отдельно.
Фактическое сохранение JSON/CSV в обычном Chrome/Firefox пока не подтверждено:
JS сериализаторы и click-download протестированы автоматически; встроенный
браузер ранее не отдавал download event.

## Интерпретируемость без причинных заявлений

`backend/ml/visualize_last_attention.py` перехватывает вход последнего
transformer-блока, вычисляет `softmax(Q_cls Kᵀ / √d)` и усредняет головы по
24×24 patch-grid. Проверены три локальных organizer-кропа с текущим весом:
обычный (`/private/tmp/lct_joint_attention_20260927.png`), wrong-accept
(`…_wrong_accept_20260927.png`) и false-reject
(`…_false_reject_20260927.png`). Patch-mass соответственно 0.9616,
0.9628 и 0.9709. Оверлеи остались только локально; исходные изображения
в публичный Git не добавлялись. Высокое attention **не доказывает**
причинного вклада в cosine или корректность ответа. Для разбора ошибок
визуализировать изображения из CSV аудита ниже, не менять ranker.

## Миллионный ANN как сервисный tie-break, не production-переключение

`backend/ml/ann_million_demo.py` строит FAISS IVF-SQ8 по 750 настоящим
эмбеддингам выбранной модели и 1 000 000 детерминированным случайным
нагрузочным векторам, сохраняет `index.faiss` + version-bound manifest и
перезагружает их. `backend/ml/ann_service.py` предоставляет `/ready` и
`/search` с проверкой версии энкодера, SHA индекса и L2-нормы запроса.
Детали запуска и измерений: `ann_million_persisted_mac_20260927.json`.
На Mac: recall@10 0.99 против потокового exact на **10 реальных запросах**,
single-query p95 101.23 мс, peak process RSS 1404.25 MiB; индекс
восстановился после перезапуска и HTTP `/ready` вернул 200. Это не качество
на миллионе реальных машин и не скорость на оборудовании жюри. Штатный API
продолжает exact cosine для малой галереи, так как приближённый индекс может
снижать mAP/F1 и меняет калибровку отказа.

Для повторения на машине с совместимым FAISS:

```sh
.venv-test/bin/pip install -r backend/ml/requirements-ann-demo.txt
.venv-test/bin/python backend/ml/ann_million_demo.py \
  --gallery-features /absolute/path/to/test_gallery_joint.npz \
  --query-features /absolute/path/to/test_query_joint.npz \
  --synthetic-count 1000000 --nlist 64 --nprobe 32 --queries 10 \
  --index-dir /absolute/path/to/ann_index \
  --output /absolute/path/to/ann_metrics.json
.venv-test/bin/python backend/ml/ann_service.py \
  --index-dir /absolute/path/to/ann_index --host 127.0.0.1 --port 18022
```

Индекс около 1 ГБ лежит вне Git и может быть воспроизведён; `POST /search`
принимает JSON `{model_version, embedding: [1024 floats], topk}`. Вектор
должен быть L2-нормирован. Сервис не применяет production refusal policy.

## Ошибки selected joint-модели

`backend/ml/extract_joint_audit.py` извлекает frozen CLS для dev/calibration
без обучения; `repro/audit_joint_errors.py` проверяет SHA источников,
порядок изображений, версию веса и policy, затем считает full-fold
cross-camera mAP/Rank и отдельные open-set ложные совпадения/отказы. Для
каждого среза галерея фиксирована. Размер, яркость и sharpness — измеримые
proxy; аннотаций ракурса и перекрытия нет, поэтому **их качество не заявляем**.
Оба fold извлечены локально в FP32 и разобраны: dev/calibration mAP
73.1528/80.3063%, pair-F1 58.7382/69.9467%, unknown-query TNR
77.5982/77.7003%. Конкретные типы ошибок, знаменатели, 3 визуально
просмотренные случая и ограничения — в
[отчёте об ошибках](JOINT_ERROR_AUDIT_20260927.md). Holdout и test labels
не читались; новых порогов по этому аудиту не подбирали.
Calibration F1/TNR здесь вычислены окончательным fitted booster на его
fit-ID, **не OOF**; dev — многократно использованный exploratory readout.
Это диагностический профиль runtime, не независимый прогноз hidden-score.
