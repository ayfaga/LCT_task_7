# LCT Vehicle ReID prototype

This branch provides a backend API and an isolated ML service for the **prototype** E2 encoder and hybrid refusal policy. The model is scheduled for further training; this version and its thresholds are not the final competition model.

## Run

Place the verified E2 artifact directory and a gallery built by the **same** encoder on the host. In the parent research workspace the default paths already point to:

- `../data/derived/final_e2_hybrid/` — model, manifests and portable boosting policy;
- `../data/derived/final_e2_hybrid/test_gallery_e2.npz` — 750-image test gallery.

For another checkout, set `LCT_ML_ARTIFACT_DIR` and `LCT_ML_GALLERY_PATH` to absolute host paths. Weights and gallery are mounted read-only; they are never stored in Git or the container image.

```sh
LCT_BACKEND_PORT=18000 LCT_ML_PORT=18001 docker compose up --build -d
docker compose ps
curl -fsS http://127.0.0.1:18000/ready
```

The backend API and OpenAPI docs are at `http://127.0.0.1:18000` and `/docs` with the shown port settings. The ML service is also exposed locally on port 18001 for diagnostics. Host ports default to 8000/8001 and can be overridden independently. The browser UI from the original repository remains a mock and is outside this backend prototype.

On Linux with an NVIDIA GPU and the Docker GPU runtime, use `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d`. This optional GPU path has not been measured locally; the measurements below used Docker Desktop CPU.

A real request uses multipart JPEG/PNG and integer BBox `x,y,w,h` in **original-image pixels**:

```sh
curl -F image=@/path/to/car.png -F x=0 -F y=0 -F w=640 -F h=360 \
  http://127.0.0.1:18000/api/identify
```

`/api/identify`, `/api/infer` and `/v1/search` return the same search schema. `ranked` contains the cosine top-10; `accepted` (also exposed as `candidates` for compatibility) is the subset kept by the hybrid refusal policy. `status=no_confident_match` with `accepted=[]` is a successful empty answer. The candidate fields `similarity` and `confidence` are both raw cosine, **not** a calibrated probability. `/v1/embeddings` returns the normalized 1024D vector. `/health` checks the backend process; `/ready` checks database, ML artifacts and gallery.

Architecture, version contract and handoff details: [backend handoff](docs/BACKEND_HANDOFF.md) and [ML package guide](backend/ml/README.md).
