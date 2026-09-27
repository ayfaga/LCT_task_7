# LCT Vehicle ReID prototype

**Current ML default:** joint CityFlow+organizer DINOv2-L/14@336, best epoch 19,
with a separately recalibrated shallow5 refusal policy. The 1.218 GB model
file in `model_artifacts/joint_l336/` is tracked by Git LFS: run `git lfs pull`
after cloning. A fresh installation starts with an **empty persistent gallery**;
upload real images through the API/browser or pass a compatible, precomputed
`LCT_ML_GALLERY_PATH`. No synthetic gallery is searched by default. Full version,
checksums, measured quality and deployment gates are in
[the joint-model handoff](docs/JOINT_L336_DEPLOYMENT_20260925.md).

The older E2 artifacts remain available as a rollback by explicitly setting
`LCT_ML_ARTIFACT_DIR=../data/derived/final_e2_hybrid` and
`LCT_ML_GALLERY_PATH=../data/derived/final_e2_hybrid/test_gallery_e2.npz`.
The rest of this README documents the existing API and user-gallery workflow.

For exact artifact hashes, data provenance, Python dependencies, preprocessing,
API/export contracts, local checks and remaining external gates, see the
[release verification ledger](docs/RELEASE_VERIFICATION_20260927.md).

For the organizer's large original JPEG + `image_id,x,y,w,h` ZIP, use the
[streaming archive importer](docs/ORGANIZER_ARCHIVE_IMPORT.md). It builds
the compatible gallery and all three submission files without unpacking the
whole archive. The general replenishment button stores files only; it does
not index this archive. Docker gallery mounting has a dedicated opt-in override.

This branch provides a backend API and an isolated ML service for the selected
joint encoder and hybrid refusal policy. The model remains a competition
candidate, not a verified hidden-test winner.

## Run

For a local Python run, hydrate the LFS weight first. `quick_start.py` now
checks the weight size/SHA256 **before** installing dependencies;
it fails with a `git lfs pull` hint if the checkout contains only the 135-byte
LFS pointer. Run it from a virtual environment to avoid changing global Python:

```sh
git lfs install
git lfs pull
python3 -m venv .venv
./.venv/bin/python quick_start.py
```

If dependencies are already installed, add `--skip-install`. For a real local
gallery, export `LCT_ML_GALLERY_PATH` to its absolute `.npz` path (and optionally
`LCT_ML_ARTIFACT_DIR` to a separate verified bundle). `--backend-port` and
`--ml-port` override 8000/8001. Startup waits for ML `/ready` before starting
the backend and waits for backend `/ready` before reporting success. It does not
generate a random gallery. `/ready` reports `gallery_ready=false` and
`gallery_size=0` until a default gallery is supplied; individual user galleries
can be uploaded and searched independently. Gallery embeddings and metadata
are stored in SQLite under the persistent `gallery_state` directory. The
included 10-vector gallery is **synthetic demo data** and is never selected
by default; do not use it to claim quality or submit results.

The Docker image starts with the same artifact check. `docker compose up
--build` still requires the LFS weight to be present on the host, but no
precomputed gallery; building the image from the internet is not an offline build. For an
offline judge run, prepare the image and mounted artifacts in advance, then
verify `docker compose up` with networking disabled. This release gate remains
open until a clean-machine rehearsal is recorded.

Place the verified model artifact directory on the host. A precomputed gallery,
if used, must be built by the **same** encoder. In the parent research workspace:

- `./model_artifacts/joint_l336/` — LFS weight, manifests and portable boosting policy;
- `../data/derived/joint_l336_20260925/test_gallery_joint.npz` — optional 750-image test gallery, **not mounted by default**.

For another checkout, set `LCT_ML_ARTIFACT_DIR` to the absolute host path of
the verified bundle. `LCT_ML_GALLERY_PATH` is read directly in a local Python
run; in Docker, add an explicit read-only volume and the container-side path
in a Compose override. Merely setting a host environment variable does not
mount a gallery. The selected weight is stored via Git LFS; gallery images and
embeddings are not stored in Git or baked into the container image.

```sh
LCT_BACKEND_PORT=18000 LCT_ML_PORT=18001 docker compose up --build -d
docker compose ps
curl -fsS http://127.0.0.1:18000/ready
```

The backend API and OpenAPI docs are at `http://127.0.0.1:18000` and `/docs` with the shown port settings. The ML service is also exposed locally on port 18001 for diagnostics. Host ports default to 8000/8001 and can be overridden independently. The existing browser UI uses the same backend endpoints; it now shows ranked IDs and cosine confidence and can export results. Its visual design was not changed.

On Linux with an NVIDIA GPU and the Docker GPU runtime, use `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d`. This optional GPU path has not been measured locally; the measurements below used Docker Desktop CPU.

A real request uses multipart JPEG/PNG and integer BBox `x,y,w,h` in **original-image pixels**:

```sh
curl -F image=@/path/to/car.png -F x=0 -F y=0 -F w=640 -F h=360 \
  http://127.0.0.1:18000/api/identify
```

`/api/identify`, `/api/infer` and `/v1/search` return the same search schema. `ranked` contains the cosine top-10; `accepted` (also exposed as `candidates` for compatibility) is the subset kept by the hybrid refusal policy. `status=no_confident_match` with `accepted=[]` is a successful empty answer. The candidate fields `similarity` and `confidence` are both raw cosine, **not** a calibrated probability. `/v1/embeddings` returns the normalized 1024D vector. `/health` checks the backend process; `/ready` checks database, ML artifacts and gallery.

## Upload a separate gallery

The public backend accepts JPEG/PNG files, a folder sent as individual multipart files, or a ZIP. A gallery can be extended in several imports. Original images and versioned embedding archives live in the persistent `gallery_state` Docker volume; they are not included in Git or the image. A fresh installation has no default gallery; the optional organizer test gallery of 750 images is kept outside Git and is only loaded when its path is supplied explicitly.

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

Architecture, version contract and handoff details: [backend handoff](docs/BACKEND_HANDOFF.md), [backend developer guide](docs/BACKEND_DEVELOPER_GUIDE.md), and [ML package guide](backend/ml/README.md). The selected model's [training source and exact recipe](repro/README.md) are included. For the current evidence-based criteria audit see [Mac criteria audit](docs/CRITERIA_AUDIT_MAC_20260927.md); the earlier [startup diagnosis](docs/LOCAL_STARTUP_AND_CRITERIA_20260927.md) is historical.
