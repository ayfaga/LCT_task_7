# Strong L336 + batch AQE inference bundle

Only this DINOv2-L/14@336 inference weight is required by the current
checkout. Fetch its Git LFS object with `git lfs pull` after cloning. The
weight is 1,217,580,060 bytes, SHA256
`de955ba6ec3a2cb232708eddd1b1b57400bc449405d6d9e2ab25c791a46e617f`.
The portable refusal policy is bound to the same model version; never mix
it with a previous encoder or gallery embedding. `bundle_manifest.json`
records sizes and SHA256 for every runtime artifact. Verify with:

```sh
python -m backend.ml.verify_bundle --artifact-dir model_artifacts/strong_l336_aqe
```

Single-query search uses exact cosine. Multi-query search applies
transductive AQE k=5, alpha=0.25 to the submitted cohort and gallery,
then ranks by the expanded-vector cosine. Returned `confidence` is the
raw encoder cosine, not a probability. See
`docs/STRONG_AQE_INTEGRATION_20260929.md` for scope and limitations.

The older joint L336 release remains available only in the parent
research workspace as a local rollback; it is intentionally excluded
from this single-weight competition package.
