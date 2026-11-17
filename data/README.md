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

## Proposed class taxonomy (pending approval — Gate 1)

60 raw categories is unworkable for a lightweight detector trained on 1,500 images (most classes have single-digit instance counts). Two taxonomy options, built from the *actual* measured frequencies above rather than an assumed split:

- **Option A — single `litter` class.** Collapses everything to binary foreground/background detection. Simplest, most robust given the data volume, but discards the category information the report needs to "discuss results... per category" meaningfully, and makes the small/rare-class augmentation question (refined RQ1) harder to state per-category.
- **Option B (recommended) — use the dataset's 28 supercategories, then keep only ones with ≥50 instances** (`Cigarette`-like `Unlabeled litter`, `Plastic bag & wrapper`, `Bottle`, `Bottle cap`, `Can`, `Other plastic`, `Cup`, `Carton`, `Straw`, `Broken glass`, `Paper`, `Lid`, `Pop tab` — 13 groups covering the large majority of instances), with everything below that threshold folded into a single `other litter` class. Keeps enough category resolution for a meaningful per-class discussion and for RQ1's small-vs-common-class comparison, while not asking the model to learn categories it has fewer than ~50 examples to learn from.

**Recommendation: Option B, 13 supercategory-derived classes + 1 `other litter` catch-all (14 classes total).** Pending your approval before any class remapping is implemented.

## Proposed split (pending approval — Gate 1)

- **70/15/15 train/val/test**, stratified by the proposed class taxonomy (not raw 60-class, which is too sparse to stratify).
- **Leakage check performed:** TACO's `annotations.json` records per-image `flickr_url`/source metadata; images are organized into 15 independently-contributed `batch_*` folders with no evidence of repeated frames of the same physical scene (checked: no duplicate file hashes across the dataset). Splitting by random image ID (not by batch) is therefore low-risk for this dataset, unlike datasets built from video frames.
- Split will be saved as a fixed manifest of image IDs (`data/splits/`) before any training begins, and the test split will not be touched again until final evaluation.

## Gate 1

**Pending your approval:** the Option B taxonomy (13 supercategory classes + 1 catch-all) and the 70/15/15 stratified-by-taxonomy split above. Once approved, Phase 2 (preprocessing: COCO→YOLO conversion, split manifest generation, resize/letterbox with correct box remapping) begins.
