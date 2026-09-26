# Vendored DINOv2 inference code

Only the DINOv2 ViT-L/14 inference import closure from the research project's
local `vendor/dinov2-main` is committed here: `dinov2.hub.backbones`, its
utility module, `vision_transformer`, and required layers. The original
Apache-2.0 `LICENSE` is included. Training/configuration, unrelated research
extensions and notebooks from that local checkout are not part of this
backend patch. The trained E2 checkpoint is an external artifact, not a Git
asset, and its SHA256 is checked by the runtime before loading.
