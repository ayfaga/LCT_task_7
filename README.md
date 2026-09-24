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

## Upload a separate gallery

The public backend accepts JPEG/PNG files, a folder sent as individual multipart files, or a ZIP. A gallery can be extended in several imports. Original images and versioned embedding archives live in the persistent `gallery_state` Docker volume; they are not included in Git or the image. Uploading requires no changes to the default 750-image gallery.

```sh
curl -F 'name=My gallery' http://127.0.0.1:18000/api/galleries
# Copy gallery_id from the response:
curl -F images=@/path/to/car1.jpg -F images=@/path/to/car2.png \
  http://127.0.0.1:18000/api/galleries/GALLERY_ID/images
# Or use one ZIP (can contain manifest.csv at its root):
curl -F archive=@/path/to/cars.zip \
  http://127.0.0.1:18000/api/galleries/GALLERY_ID/images
curl http://127.0.0.1:18000/api/galleries/GALLERY_ID
curl -F image=@/path/to/query.jpg -F x=0 -F y=0 -F w=640 -F h=360 \
  -F gallery_id=GALLERY_ID http://127.0.0.1:18000/api/identify
```

`POST /api/galleries/{id}/images` returns 202; poll `GET /api/galleries/{id}` until `state=ready` (or `collecting` if fewer than ten images). Search requires at least ten processed images because the fixed refusal policy uses cosine top-10. `GET /api/galleries` lists galleries, `GET /api/galleries/{id}/images/{image_key}` previews a stored image, and `POST /api/galleries/{id}/retry` retries a failed import. Each upload is limited to 200 images and 200 MiB total; an individual image is limited to 20 MiB and a ZIP to 100 MiB compressed. A gallery holds at most 1000 images.

Without a manifest, each image is treated as an already cropped vehicle and its filename stem becomes its ID. For full frames, send a CSV as the `manifest` field or put `manifest.csv` at the ZIP root. Columns: `filename,gallery_id,x,y,w,h`; `filename` matches the multipart filename or path inside ZIP, and BBox coordinates refer to the original image. Empty BBox means the whole image. Keep gallery IDs unique within a gallery. A private local test page can be served separately from the research workspace; no test UI is packaged in this Git repository.

Architecture, version contract and handoff details: [backend handoff](docs/BACKEND_HANDOFF.md), [backend developer guide](docs/BACKEND_DEVELOPER_GUIDE.md), and [ML package guide](backend/ml/README.md). Full criteria audit: [criteria status](docs/CRITERIA_AUDIT_20260924.md).
