# Data

## Source & provenance

- **Dataset:** TACO — Trash Annotations in Context, v1.0
- **Source:** Zenodo, DOI [10.5281/zenodo.3587843](https://doi.org/10.5281/zenodo.3587843), file `TACO.zip`
- **License:** CC-BY-4.0 (Zenodo record). Images are crowdsourced from varied real-world sources — preserve attribution, no commercial-reuse claims.
- **Downloaded:** 2026-11-16
- **Archive checksum (sha256):** `51cd6b009cac1e94decf058e470b1fef64148f9c89923d96dcd54d4db61ceae9` (`data/raw/TACO.zip`, 2.5GB)
- **Reproduce:** `scripts/download_taco.sh` (downloads + extracts to `data/raw/TACO/`). Raw/processed data are gitignored — this repo tracks code, manifests, and small sample figures, not the dataset itself.

## Real measured statistics (not estimates — see `outputs/dataset_audit.json`, produced by `scripts/audit_dataset.py`)

| Metric | Value |
|---|---|
| Images | 1,500 |
| Annotations (bounding boxes) | 4,784 |
| Raw categories | 60 |
| Supercategories | 28 |
| Missing/corrupt image files | 0 |
| Image width (min / median / max) | 842 / 2,448 / 6,000 px |
| Image height (min / median / max) | 474 / 3,264 / 5,312 px |

These match the PRD's "approximate reference scale" exactly (1,500 images / 4,784 annotations), confirming this download is the same full release the PRD anticipated — not a smaller/different mirror.

**Class imbalance is severe**, as flagged in the literature review: `Cigarette` has 667 instances; 10 categories have ≤5 instances each (`Toilet tube`, `Six pack rings`, `Paper straw`, `Plastic glooves`, `Tupperware`, `Polypropylene bag`, `Pizza box`, `Battery`, `Other plastic cup`, `Carded blister pack`). Full per-category breakdown in `outputs/dataset_audit.json`.

**Object size is skewed large** relative to a 640×640 training resolution: of 4,784 boxes, 383 are "small" (<32×32px), 1,306 "medium" (<96×96px), 3,095 "large." Source images are full-resolution photos (median 2,448×3,264) — this is exactly why letterbox/resize must preserve small-object visibility (PRD §5.2), and it's the dataset property the refined RQ1 (small/rare-class augmentation effect) is built around.

**Annotation sanity check:** 20 random images with drawn bounding-box overlays were visually inspected (`outputs/overlays/`, compressed samples committed at `report/figures/overlay_thumbnails/`) — boxes are tightly and correctly placed around litter objects in every sample reviewed.

## Class taxonomy (approved — Gate 1, implemented in Phase 2)

60 raw categories is unworkable for a lightweight detector trained on 1,500 images (most classes have single-digit instance counts). Approved rule (Option B): keep every **supercategory with ≥50 instances** under its own name; fold everything rarer into a single `other_litter` catch-all.

Recomputing this directly in `scripts/prepare_dataset.py` (not by hand — an earlier hand-count in this file mistakenly said 13 and omitted `Cigarette`, the single largest category; see `docs/decisions.md`) gives **17 classes ≥50 instances + `other_litter` = 18 total**:

`Plastic bag & wrapper (850), Cigarette (667), Unlabeled litter (517), Bottle (439), Bottle cap (289), Can (273), Other plastic (273), Carton (251), Cup (192), Straw (161), Paper (148), Broken glass (138), Styrofoam piece (112), Pop tab (99), Lid (87), Plastic container (72), Aluminium foil (62), other_litter (everything else)`

Full counts in `data/splits/manifest.json`.

## Split (implemented in Phase 2)

- **70/15/15 train/val/test** by random image ID, fixed seed (441): **1,050 / 225 / 225 images**, **3,385 / 712 / 680 boxes**. 7 of 4,784 raw boxes were dropped as invalid (zero-area or out-of-bounds) before splitting.
- **Leakage check:** re-verified by hashing every image file during `prepare_dataset.py` — no duplicate files found among the 1,500 images. Random (not group-aware) splitting is low-risk for this dataset.
- All 18 classes have at least one instance in both train and val (no zero-instance warning was raised).
- Manifest of image IDs per split: `data/splits/manifest.json`. Test split is not touched again until final evaluation (Phase 5).

## Conversion QA (Gate 2)

- `scripts/test_coord_conversion.py`: round-trip tests (synthetic + 25 real annotations) on the COCO→YOLO bbox conversion, degenerate-box rejection, split-overlap, and taxonomy-threshold checks — **all pass**.
- 3 "after" overlays (one per split, boxes decoded back from the written YOLO label files) visually checked against the image — boxes still land correctly on the litter object post-remap. Samples: `report/figures/after_remap_overlays/`.
- Training/val images are symlinked (not copied) from the immutable raw files into `data/processed/{split}/images/` — `data/processed/` is ~6MB (labels + symlinks only), original 2.5GB dataset untouched.
