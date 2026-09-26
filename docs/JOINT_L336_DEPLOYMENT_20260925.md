# Joint L336 ML handoff

Status: model and refusal policy are versioned; local Docker/API verification
passed. The user reports explicit organizer approval to publish these derived
weights in this public repository. This is a reported permission, not an
independent legal review or permission for commercial use. Do not present
these proxy metrics as hidden score.

## What changed

The default encoder is DINOv2 ViT-L/14 at 336×336, fine-tuned jointly on
52,717 labeled CityFlow train crops and 6,209 organizer train crops. CLS
embedding is 1,024D FP32, L2-normalized. The inference-only weight is
1,217,580,473 bytes (SHA256
`507b4f2e34384e2366ab5364e147331831664493b35b40041507d88634da05f5`),
tracked by Git LFS under `model_artifacts/joint_l336/model_inference.pt`.
No public pretrain, training checkpoint, CityFlow ZIP or image data is
included in the Git package. The previous E2 bundle remains an explicit
rollback.

The new refusal policy is **not** copied from E2. It was fit on organizer
calibration IDs (grouped 5-fold OOF threshold selection), with cosine
threshold `0.44`, shallow5 booster threshold `0.5` and first three ranks
preserved. For cosine-sorted top-10 scores `s_i` and booster score `p_i`:

```text
accepted_i = (s_i >= 0.44) and (i < 3 or p_i >= 0.5)
```

The returned `confidence` is raw cosine. Dev pair-F1 changed from 0.55545
for calibrated cosine to 0.58738 with the hybrid; TNR remained 0.77598;
query-correct-ID F1 fell from 0.69333 to 0.67556. This is a tradeoff, not
a proven official F1 gain. The official F1 unit is unknown, and dev and
calibration were both consulted in previous research. Existing holdout was
not used in this integration. Portable policy SHA256:
`02578b926977de01b7cdb109a4a86160653deb22075fda66f95c7d67366691d9`.
The portable scorer matched sklearn exactly on 10,360 real dev candidate
scores (max score error 0, decision differences 0).

## Startup and external gallery

Install Git LFS before cloning or run `git lfs pull` in this repository.
Check `git lfs ls-files` and verify the weight with `shasum -a 256`.
The organizer 750-image test gallery is not part of Git and must be mounted
from the outer workspace or regenerated with `backend/ml/build_gallery.py`.
It **must** be embedded by this model and carry this model version/SHA.
The old `test_gallery_e2.npz` is incompatible. Compose defaults to
`../data/derived/joint_l336_20260925/test_gallery_joint.npz`; set
`LCT_ML_GALLERY_PATH` to another absolute host path if needed.

```sh
docker compose up --build -d
docker compose ps
curl -fsS http://127.0.0.1:8000/ready
```

The backend/frontend HTTP contract is unchanged: multipart JPEG/PNG plus
`x,y,w,h` in original-image pixels; `/api/identify` and `/api/infer`
return cosine top-10, accepted candidates or an empty answer. Existing
user-uploaded galleries keep their images but old embedding NPZ generations
must be regenerated with the new encoder before searching them; an old
gallery search is rejected as version-incompatible. The ML-only migration
tool is `backend/ml/migrate_galleries.py`: stop both services, run its
default dry-run against the persistent `/gallery_state`, then run with
`--apply` only after checking the gallery list. It keeps source images and
old NPZ generations, writes a new versioned archive and atomically updates
metadata. On this Mac, six existing galleries (72 images) were migrated and
revalidated; one empty collecting gallery was untouched. A real search
against a migrated 10-image gallery returned HTTP 200 with ten ranked
candidates. This is a functional check, not a quality metric.

```sh
docker compose stop backend ml
docker compose run --rm --no-deps ml python ml/migrate_galleries.py --artifact-dir /model --gallery-state-dir /gallery_state
# Only after reviewing the dry-run output:
docker compose run --rm --no-deps ml python ml/migrate_galleries.py --artifact-dir /model --gallery-state-dir /gallery_state --apply
docker compose up --no-build -d
```

Do not push organizer images, raw CityFlow data, old E2 weights or a second
copy of the training checkpoint into this Git branch. A clean competition
submission contains only the inference weight and separately generated
`submission.csv`, `embeddings.npy` and `candidates.csv`.

## Measurements and limitations

Organizer-only FP32 retrieval proxy: dev mAP 73.1532%, calibration mAP
80.3178%; the calibration mAP gain over E2 has paired-ID CI95
[−0.2079,+4.9744] percentage points and is not statistically settled.
The model-only A100 batch-1 p50 in the existing extractor metadata was
36.55 ms; it excludes preprocessing, search, HTTP and policy. Local CPU
single-crop embedding after startup was about 0.50 s in a seven-run probe.
On local Docker Desktop CPU, both services passed `/ready` for the new model
and 750-image gallery. The 1,110-query test export wrote 1,860×1,024
embeddings and passed exact top-10 round-trip checks. Six actual HTTP
requests (three accepted, three refused) matched the export top-10 and
accepted lists exactly. HTTP benchmark: 2 warmups, 10 sequential and 12
requests from 3 concurrent clients, zero failures; sequential median
1.667 s, p95 1.932 s; concurrent median 4.987 s, p95 5.246 s, wall
20.008 s / 0.600 completed requests/s. Invalid BBox returned 422.
These are Mac CPU observations with one ML worker, not organizer-hardware
FPS or production throughput. Raw results are in
`../data/derived/joint_l336_20260925/benchmark_http_docker.json` and
`../data/derived/joint_l336_20260925/api_export_parity.json` in the
parent research workspace; those files are not in Git.
