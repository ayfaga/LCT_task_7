# E2 Vehicle ReID ML integration

This directory is the ML-owned adapter in the backend repository. It does not
contain model weights, organizer images, or secrets. Product UI, persistence,
deployment operations, and a million-object index remain backend-owned.

## Selected bundle

`dinov2-l14-l336-fresh16-best13-20260923`: DINOv2-L/14 at 336 px, one
forward, normalized 1024-dimensional FP32 embedding. Trained inference weight
`model_inference.pt`: 1,217,580,060 bytes,
SHA256 `dc910395dda9dc735b1d5571ef7baf75b767964500b6d0a23edeb7cde4d33c52`.
No public-pretrain or training checkpoint is required at inference.

The selected refusal rule acts on **exact cosine top-10**:

`accept(rank) = cosine >= 0.394 and (rank <= 3 or booster_probability >= 0.112)`.

The booster uses five numeric score-context features, no camera, vehicle ID,
time, OCR, plate-derived feature, or additional masking. Returned confidence is
**raw cosine**, not booster probability. Empty answer iff top-1 cosine is below
0.394. The original cosine-only signed release is preserved separately for
rollback. This local `e2-boosting-hybrid1-20260924` bundle is a new candidate,
not the original A100-signed bundle.

## Runtime

Install `backend/requirements.txt` and `backend/requirements-ml.txt` in a
Python 3.12 environment. From the `LCT_task_7` checkout, set these
environment variables **before** startup:

```sh
export LCT_ML_ARTIFACT_DIR=/absolute/path/to/final_e2_hybrid
export LCT_ML_GALLERY_PATH=/absolute/path/to/compatible_gallery.npz
export LCT_ML_DEVICE=auto
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The gallery NPZ must carry matching model version/SHA, unique IDs and L2
normalized 1024D vectors. The runtime verifies model, policy, preprocessing,
and gallery identity before serving. A missing/incompatible artifact returns
`/ready` 503. Weights and gallery are external files/volumes, never git assets.
`/v1/embeddings` and `/v1/search` accept multipart image plus integer `x,y,w,h`
(pixel `xywh` in the original image). `/v1/search` additionally accepts
`topk` 1..100. It returns `ranked` exact-cosine results, `accepted` candidates
and `status=matched|no_confident_match`. It computes the refusal policy on
internal top-10 even if the caller requests fewer visible candidates.

Build a small test gallery from existing organizer crops without changing
them (from the outer research project):

```sh
python LCT_task_7/backend/ml/build_gallery.py \
  --artifact-dir data/derived/final_e2_hybrid \
  --organizer-gallery-csv outputs/eda/test_gallery.csv \
  --crop-dir data/derived/crops \
  --output data/derived/final_e2_hybrid/test_gallery_e2.npz \
  --batch-size 4
```

Do **not** use the 10-image smoke gallery for real evaluation. The current
exact NumPy gallery is suitable for the supplied 750-image test set, not a
million-vehicle production index. Backend should add and benchmark its
versioned FAISS/ANN index without changing cosine semantics or the top-10
policy context. No API E2E/FPS or million-gallery quality claim is made.

## Unlabeled submission export

Run `build_gallery.py` separately for `test_query.csv` and `test_gallery.csv`,
then use `export_test_hybrid.py` with both NPZs. It writes `submission.csv`
(exact cosine top-10), `embeddings.npy` (query then gallery, FP32 1024D),
`candidates.csv` (hybrid accepted candidates or blank refusal row), and
`manifest.json` with SHA256 and an independent round-trip. It does not load
identity labels or the sealed holdout. `confidence` is raw cosine.

From the outer research-project root (which contains `LCT_task_7/` and
`data/derived/crops/`), the remaining commands are:

```sh
python LCT_task_7/backend/ml/build_gallery.py \
  --artifact-dir data/derived/final_e2_hybrid \
  --organizer-gallery-csv outputs/eda/test_query.csv \
  --crop-dir data/derived/crops \
  --output data/derived/final_e2_hybrid/test_query_e2.npz \
  --batch-size 4
python LCT_task_7/backend/ml/export_test_hybrid.py \
  --artifact-dir data/derived/final_e2_hybrid \
  --query-csv outputs/eda/test_query.csv \
  --gallery-csv outputs/eda/test_gallery.csv \
  --query-features data/derived/final_e2_hybrid/test_query_e2.npz \
  --gallery-features data/derived/final_e2_hybrid/test_gallery_e2.npz \
  --output data/derived/final_e2_hybrid/test_export
```

Restart the API after replacing a gallery or bundle; its runtime is cached
per process. A mere file overwrite does not constitute a versioned switch.

## Measured evidence and limitations

The historical cross-camera, unseen-ID dev retrieval mAP of this frozen
encoder is 0.704591; selecting the booster does **not** change it. On the
fixed dev refusal protocol, cosine pair-F1/TNR/query-F1 was
0.506414/0.660508/0.668313; selected hybrid was
0.532541/0.660508/0.663374. The exact official F1 unit is unknown and dev
has been consulted repeatedly. These are not hidden-test scores. A local
portable-vs-sklearn parity check covered 5,000 candidate scores with zero
numerical/decision differences; real-image and API smoke checks verify
correct wiring, not generalization or latency SLOs.

See the research-project `LLM_WIKI/CURRENT.md`,
`reports/l336_e2_refusal_boosting_20260924.md`, and the new local bundle
manifests for provenance. The source booster `.joblib` is kept only for
reproducibility; production inference reads portable `boosting_policy.json`
and has no scikit-learn/joblib dependency.
