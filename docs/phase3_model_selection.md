# Phase 3 — model selection (on-paper comparison)

Per PRD §6, comparing the candidate (YOLO11n) against one feasible alternative before implementing either.

| | **YOLO11n (Ultralytics)** — candidate | **Faster R-CNN** (torchvision, e.g. ResNet50-FPN or MobileNetV3-FPN backbone) — alternative |
|---|---|---|
| Architecture | Single-stage, anchor-free. CNN backbone (C3k2 blocks) + C2PSA attention + decoupled detection head. | Two-stage: CNN backbone + Region Proposal Network + ROI head. |
| Parameters | ~2.6M (nano) | ~19M (MobileNetV3-FPN) to ~41M (ResNet50-FPN) |
| Pretrained weights | COCO, via `ultralytics` one-line load | COCO, via `torchvision.models.detection`, also one-line load |
| Training loop | Built-in `model.train(...)`: handles dataloader, loss, LR schedule, checkpointing, metrics, plots | No built-in trainer — requires hand-written training loop, COCO-style batching/collation, and a metrics (mAP) implementation or `torchmetrics` dependency |
| Augmentation control | Built-in, toggled via named hyperparameters (`mosaic`, `fliplr`, `hsv_h`, etc.) — exactly what's needed to cleanly separate E0 (off) from E1 (on) with everything else identical | No built-in augmentation pipeline — would need to hand-build one (e.g. with `albumentations`), adding implementation risk to the E0/E1 comparison itself |
| Compute (M1 Pro, 16GB, MPS) | Light — nano model, fast per-epoch on ~1,050 images | Heavier — two-stage inference/training is slower per image; MobileNetV3-FPN backbone would be the only realistic variant on this hardware in the available time |
| License | AGPL-3.0 (Ultralytics) — fine for open academic/research use; a closed commercial product would need an Ultralytics Enterprise license, not relevant here | BSD-3 (torchvision) — more permissive, but irrelevant to the actual decision criteria below |
| Fit to course content | Course covers CNN backbones (VGGNet/GoogLeNet/ResNet) and pretrained-model transfer learning directly (`Week 7 up` lecture set) — YOLO11n's CNN backbone + fine-tuned head is a direct application of exactly that. | Also CNN-backed, but the RPN/two-stage mechanism isn't part of this course's syllabus topic list — a bigger leap from taught material. |
| Literature support for this exact task | Multiple cited papers (`docs/literature_notes.md`) use YOLO variants directly on litter/garbage detection, including on constrained hardware. | One cited source (CalState thesis) uses Faster R-CNN on TACO specifically — a reasonable precedent, but less common in the litter-detection literature than YOLO. |

## Decision

**YOLO11n**, for three concrete reasons: (1) built-in augmentation toggles make the E0-vs-E1 comparison clean without extra implementation risk, (2) it fits a 10-point assignment's time budget on local M1 Pro/MPS hardware, (3) it maps directly onto this course's own taught content (pretrained CNN backbone + transfer learning), rather than introducing a detector family (two-stage, RPN) the course never covers.

- **Backbone:** CNN feature extractor (C3k2 blocks), pretrained on COCO.
- **Detection head:** decoupled, anchor-free — predicts class + bounding box per cell directly.
- **Fine-tuning:** full-network fine-tuning (not a frozen-backbone linear probe) on the 18-class TACO taxonomy, starting from COCO pretrained weights, since COCO's own classes don't include litter-specific categories — a frozen backbone would leave the head undertrained for this domain.
- **License note:** AGPL-3.0 can require a combined work that integrates `ultralytics` to itself be source-available under a compatible license if distributed (Ultralytics sells commercial licenses specifically for closed-source use). This repo's own code is MIT and will be published openly, which avoids the closed-source conflict, but this is not a substitute for reading Ultralytics' actual AGPL FAQ if this code is ever repurposed for anything closed-source or commercial.
