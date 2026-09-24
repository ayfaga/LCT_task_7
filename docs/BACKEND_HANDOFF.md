# Backend handoff: E2 prototype

## Components

- `backend/app/ml/` owns preprocessing, strict DINOv2 E2 loading, exact cosine search and the portable hybrid refusal policy. `app.ml.api` runs it as a separate one-worker HTTP service on port 8001.
- `backend/app/ml_gateway.py` is the backend's only ML client. `app.main` exposes the public `/api/identify` and compatible `/api/infer`, `/v1/search`, `/v1/embeddings` routes. The backend does not import Torch or load weights.
- `docker-compose.yml` runs the backend and ML service separately. SQLite stores the existing request metadata and validated uploads in a Docker volume. The prototype gallery is a read-only, version-checked NPZ. For a production million-object gallery the backend team owns a persistent vector index and benchmark; this prototype does not claim it.

## ML contract

Request: multipart `image` (JPEG/PNG), `x,y,w,h` integer BBox in pixels of the supplied full frame. The ML service crops that BBox exactly, applies the frozen preprocessing and returns a 1024D L2-normalized FP32 vector. It scores the version-matched gallery by exact cosine and takes top-10. The `shallow5` policy accepts candidate `i` when `cosine_i >= 0.394` and (`i <= 3` or `booster_i >= 0.112`). The returned `similarity` and `confidence` fields are the same raw cosine. Do not use either as a probability; do not provide camera/ID/plate-derived features. Invalid input returns 422/413, missing model/gallery 503, confident refusal `200` with empty `accepted`.

`topk` can be 1…100 for inspection; the refusal policy always runs on the internal cosine top-10. For the competition export, use exactly 10. The optional research k-reciprocal reranker is **not** part of this prototype because its scoring and refusal interaction are still unverified.

The 1,217,580,060-byte `model_inference.pt` must have SHA256 `dc910395dda9dc735b1d5571ef7baf75b767964500b6d0a23edeb7cde4d33c52`. The portable policy and preprocessing SHA are checked against their manifests at ML startup; the gallery records model version and weight SHA. After later full-data training, create a *new* artifact directory and gallery, recalibrate refusal, rerun tests and switch volumes/version together. Do not overwrite this prototype's files in place.

## Acceptance for backend team

1. Review this branch against current `master`; keep the ML package and gateway separate from product/UI changes.
2. Mount a verified artifact directory and compatible gallery, run `docker compose up --build -d`, verify both `/ready` endpoints and `/docs`.
3. Call `/api/identify` with a real image/BBox. Check `ranked`, `accepted`, `model_version` and a refusal case. The original `/api/infer` stub now runs the same real path.
4. Integrate the frontend and request persistence as product work. The current frontend still does not call the model.
5. Benchmark on the organizers' hardware. Current exact NumPy gallery is intended for the supplied 750-image test gallery. Add a versioned scalable index only after measuring retrieval fidelity and end-to-end latency.

The offline export remains under the outer research project's `data/derived/final_e2_hybrid/test_export/`; the service does not generate it on each request. It contains unlabeled test predictions, not an official hidden score.

The later train+dev refit was evaluated once on the held-out proxy split but did not pass its predeclared promotion gate. It is not mounted by this prototype; do not reuse the current gallery or refusal thresholds with a future encoder.
