# Evaluation reference sources

`pnp_inversion/matrics_calculator.py` and `pnp_inversion/evaluate.py` are unmodified
files from the official repository for **PnP Inversion: Boosting Diffusion-based
Editing with 3 Lines of Code**, Xuan Ju et al., ICLR 2024:

https://github.com/cure-lab/PnPInversion/tree/07f97f448150e2ca220bebd54c8f687c5c50c67a/evaluation

`reference_manifest.json` records URLs, commits and SHA256 checksums. The adapter
verifies the calculator checksum before loading it. Only metric functions are
used; the upstream command-line evaluation program is not executed.

DINO is loaded from the pinned official commit
`facebookresearch/dino:7c446df5b9f45747937fb0d72314eb9f7b66930a` (Apache-2.0).
Model weights are downloaded at runtime, not bundled.

The small CLIP tokenizer test fixtures come from the pinned SD 1.4 model revision
listed in the manifest. They are only used for token alignment tests. The Google
Prompt-to-Prompt reference files remain under their original Apache-2.0 notice.
