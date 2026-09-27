# Reproduce the selected joint L336 encoder

This directory contains the **actual research trainer** for the shipped
CityFlow+organizer DINOv2-L/14@336 encoder and its import closure. The source
was copied from the research workspace without changing the training recipe;
only repository-relative vendor/config paths were adapted. No organizer or
CityFlow images, cookies, or research-only local paths are committed.

## Inputs and provenance

1. Obtain the organizer-provided `dataset.zip` under the competition terms.
   The exact original archive is not redistributed here. `eda.py` reads its
   `train.csv`, `test_query.csv`, `test_gallery.csv` and original JPEG/PNG,
   creates `objects.csv` and 448-pixel max-side PNG BBox crops without further
   plate masking. Training consumes only organizer train identities in the
   checked-in `configs/identity_splits.json` (1,001 train; 231 dev; 154
   calibration; 155 holdout). The dev fold selects the checkpoint; calibration
   and holdout do not enter the trainer.
2. Obtain the public AICity 2021 Track 2 ReID archive from
   https://www.aicitychallenge.org/2021-track2-download/ . The audited ZIP is
   2,152,994,934 bytes, SHA256
   `70bfc5bcec322f2b7046281af4039ae4915510c5b4be1e05895fb38193b681d7`.
   The trainer reads its labeled train split directly from ZIP: 52,717 images,
   440 vehicle IDs. No CityFlow query/test images are trained on. Respect the
   dataset's noncommercial/research-use conditions; the team reports specific
   organizer approval for use and publication of the derived competition weights.
3. Obtain the official public DINOv2 ViT-L/14 PyTorch checkpoint from
   https://github.com/facebookresearch/dinov2 . The audited input file is
   SHA256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`.
   The DINOv2 code and Apache-2.0 license are in `backend/third_party/dinov2-main`.

The organizer `objects.csv` used in the recorded training run had SHA256
`c5eabdd4c17e02265f65b90c0568d51ef7907e099b413ba397f694f6f40c661d`.
The checked-in split has the same JSON identities as the research original;
the copied file has one added final newline, so its byte SHA differs from the
original `9de62999082a5dd832fc6df53755b94468bf0c072aa8dffadb23c927aa2f1345`.
This does not change the identity allocation but changes the recorded byte
signature of a fresh run. The exact original file can be recovered by removing
that final newline if byte-for-byte metadata matching is required.

## Commands

From repository root on a CUDA machine with enough disk space:

```sh
python3.12 -m venv .venv-repro
.venv-repro/bin/pip install -r repro/requirements.txt
.venv-repro/bin/python repro/eda.py \
  --archive /absolute/path/to/organizer/dataset.zip \
  --output outputs/eda --crop-cache data/derived/crops
shasum -a 256 /absolute/path/to/AICity21_Track2_ReID.zip /absolute/path/to/dinov2_vitl14_official.pth outputs/eda/objects.csv
.venv-repro/bin/python repro/train_joint_cityflow_organizer.py \
  --zip /absolute/path/to/AICity21_Track2_ReID.zip \
  --weights /absolute/path/to/dinov2_vitl14_official.pth \
  --objects outputs/eda/objects.csv \
  --splits repro/configs/identity_splits.json \
  --crops data/derived/crops \
  --output outputs/joint_l336_reproduction \
  --epochs 20 --steps-per-epoch 335 --p-each 4 --k 4 --size 336 \
  --last-blocks 4 --lr 1e-5 --head-lr 3e-4 --weight-decay 0.04 \
  --warmup 2 --workers 0 --threads 4 --seed 20260918 --research-only
```

The trainer uses 32-image batches (four IDs from each source × four images),
camera/track-aware sampling, mild appearance augmentation and random erasing,
label-smoothed ID cross-entropy plus within-source batch-hard metric loss,
BF16 autocast, AdamW, last four ViT blocks plus final norm trainable, and
selects the best checkpoint by organizer dev cross-camera mAP. It runs 20 ×
335 = 6,700 optimizer steps. `best.pt`, `history.json`, and `status.json` are
written under the chosen output path. The selected original best was epoch 19
with SHA256 `e1e3c03476d3b1af8eb9d9323b896d4e1e0859064597aaa66c79ee1b3e6e8d83`.

To rerun the identity-disjoint dev/calibration readout and fit a **new**
refusal policy from the freshly trained checkpoint:

```sh
for fold in dev calibration; do
  .venv-repro/bin/python repro/extract.py \
    --objects outputs/eda/objects.csv --crops data/derived/crops \
    --weights /absolute/path/to/dinov2_vitl14_official.pth \
    --checkpoint outputs/joint_l336_reproduction/best.pt \
    --arch large --size 336 --geometry pad --device cuda \
    --identity-split "$fold" --splits repro/configs/identity_splits.json \
    --output "outputs/joint_l336_reproduction/features_$fold" --research-only
done
.venv-repro/bin/python repro/audit_joint_refusal.py \
  --objects outputs/eda/objects.csv --splits repro/configs/identity_splits.json \
  --calibration-features outputs/joint_l336_reproduction/features_calibration/features.npz \
  --dev-features outputs/joint_l336_reproduction/features_dev/features.npz \
  --output outputs/joint_l336_reproduction/refusal_audit
```

The historical dev fold was repeatedly inspected during research, so this
is not a new independent final score. A newly fitted policy must not be
silently substituted for the frozen shipped policy. To rebuild the three
unlabeled test files with the **shipped** version-matched inference bundle:

```sh
.venv-repro/bin/python backend/ml/build_gallery.py \
  --artifact-dir model_artifacts/joint_l336 \
  --organizer-gallery-csv /absolute/path/to/test_gallery.csv \
  --crop-dir data/derived/crops \
  --output outputs/joint_l336_reproduction/test_gallery.npz --device cuda
.venv-repro/bin/python backend/ml/build_gallery.py \
  --artifact-dir model_artifacts/joint_l336 \
  --organizer-gallery-csv /absolute/path/to/test_query.csv \
  --crop-dir data/derived/crops \
  --output outputs/joint_l336_reproduction/test_query.npz --device cuda
.venv-repro/bin/python backend/ml/export_test_hybrid.py \
  --artifact-dir model_artifacts/joint_l336 \
  --query-csv /absolute/path/to/test_query.csv \
  --gallery-csv /absolute/path/to/test_gallery.csv \
  --query-features outputs/joint_l336_reproduction/test_query.npz \
  --gallery-features outputs/joint_l336_reproduction/test_gallery.npz \
  --output outputs/joint_l336_reproduction/submission
```

The helper's `--organizer-gallery-csv` accepts both organizer query and
gallery CSVs; each must contain `image_id`. The exporter checks that feature
order matches the CSV and does not require test labels.

`repro/extract.py` extracts FP32 features from `best.pt` for dev/calibration
and `repro/protocol.py` performs the cross-camera metric (all same-camera
gallery images excluded). The release-specific inference conversion, portable
boosting calibration and unlabeled three-file export are documented in
`docs/JOINT_L336_DEPLOYMENT_20260925.md` and implemented in `backend/ml/`.
`submission.csv` is top-10 sorted gallery IDs per query;
`embeddings.npy` is query then gallery order; `candidates.csv` contains numeric
cosine confidence or an empty refusal row. The official evaluator was not
provided, so our scorer is a documented proxy, not an official certification.

The local Mac check on 2026-09-27 verified the trainer CLI/import closure and
the 1,001/231/154/155 identity split. A full 20-epoch rerun from these
public-repo paths has **not** yet been performed, so exact bitwise reproduction
is not claimed. The original A100 run and metrics are separately recorded in
the research report. Do not substitute the currently shipped inference weight
for the public-pretrain initialization or train on the test/holdout folds.
