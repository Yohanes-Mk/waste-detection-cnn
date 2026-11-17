# Phase 0 — one-page experimental protocol proposal

**Status: proposed, pending Gate 0 approval. Nothing below has been executed.**

## Domain & focus
- **Domain:** outdoor litter/waste detection, motivated by (not implementing) autonomous cleanup robotics.
- **Research focus:** object detection (bounding boxes) — chosen over classification (doesn't localize) and segmentation (TACO's segmentation masks exist but pixel-level output is more than the assignment needs and harder to evaluate within scope).

## Research question
See `docs/literature_notes.md` for the full assessment. Recommended framing:
> For a lightweight CNN-based detector fine-tuned on TACO, does training-time augmentation improve detection performance disproportionately on small-footprint/rare litter categories compared to common, large-footprint categories?

## Dataset (candidate, not yet downloaded)
- TACO — http://tacodataset.org/ / https://github.com/pedropro/TACO. ~1,500 images, ~4,784 annotations, ~60 raw categories (exact counts to be confirmed on actual download in Phase 1).
- License: images/annotations on Zenodo under CC-BY-4.0; per-image rights can vary — attribution required, no commercial-reuse claims.
- Known limitations: small size, long-tailed class distribution, many single-instance classes, small-object instances.

## Model (candidate, not yet selected/implemented)
- YOLO11n (Ultralytics), pretrained on COCO, fine-tuned on TACO. CNN backbone + detection head, bounding-box regression + classification.
- On-paper alternative to compare against before locking in (PRD §6): a small Faster R-CNN (two-stage, CNN backbone) — slower, more complex to fine-tune on a 16GB M1 Pro in the available time, but a legitimate literature-supported alternative (see literature note source 5).

## Experiment design
- **E0 (baseline, required):** fine-tune on TACO with standard preprocessing (resize/letterbox to 640×640, correct box remapping), no training-time augmentation beyond framework-default normalization.
- **E1 (augmented, required — but scoped as this pass's stretch goal per your instruction to lock in baseline first):** identical seed/split/epochs/eval as E0, plus a specified, versioned augmentation config (brightness/contrast, moderate scale/crop, horizontal flip where valid).
- **Independent variable:** presence/absence of training-time augmentation.
- **Dependent variable:** mAP@0.5, mAP@0.5:0.95, precision/recall — reported both pooled and broken out by object-size bucket / rare-vs-common class.
- **Controlled variables:** seed, train/val/test split, model architecture, epochs, optimizer, learning rate, input resolution.
- **Test set:** locked before any tuning; used once, at the end, for both E0 and E1.
- **Unseen-image check:** a small set of genuinely new images (not in train/val/test) for the qualitative "unseen data" requirement in the professor's step 5.

## Compute
- Local: Apple M1 Pro, 16GB RAM, PyTorch MPS backend. No ML libraries installed yet (Phase 1 will add `ultralytics`, `torch`, etc. to `requirements.txt`). Expected feasible for YOLO11n fine-tuning at this dataset size without needing Colab; Colab remains a fallback if local training is too slow.

## What happens after approval (Gate 0)
Moving to Phase 1: download TACO, audit real counts/dimensions/class distribution, confirm license terms, propose class taxonomy, and create the leak-resistant train/val/test split — all logged in `data/README.md` and `docs/decisions.md`, with no training started until that data-quality gate passes.
