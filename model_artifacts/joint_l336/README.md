# Joint L336 inference artifact

`model_inference.pt` is a Git LFS object, not a normal Git blob. After clone,
run `git lfs pull`; verify SHA256
`507b4f2e34384e2366ab5364e147331831664493b35b40041507d88634da05f5`.
The 1,217,580,473-byte file is the **only** DINOv2 weight needed at inference.
The training checkpoint and original public pretrain are not shipped.

`boosting_policy.json` (SHA256
`02578b926977de01b7cdb109a4a86160653deb22075fda66f95c7d67366691d9`)
is fitted for exactly this encoder; never use it with the old E2 weight.
Likewise, a gallery must be re-embedded and tagged with this model version/SHA.
See `docs/JOINT_L336_DEPLOYMENT_20260925.md` for setup and limitations.
