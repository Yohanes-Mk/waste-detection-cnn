# Vision-Based Waste Detection: Fine-Tuning a Lightweight CNN Detector on TACO

**DRAFT — for your review before submission.** All numbers below are real, logged results from this repository (`docs/phase3_results.md`, `docs/phase5_results.md`, `outputs/`). Rewrite the prose into your own words/voice before turning this in.

CSCI 441/541 Research Assignment · 10 points

---

## 1. Domain

This project's domain is **outdoor litter and waste detection** — identifying and localizing discarded trash (bottles, bags, cans, cigarette butts, and related items) in photographs of real-world outdoor scenes such as streets, parks, and beaches. The motivating long-term application is an autonomous cleanup robot, which would need to visually locate candidate litter objects before it could navigate toward or collect them. This project studies only that visual perception step — no robotic hardware, navigation, or physical pickup is implemented or claimed.

## 2. Research focus and reasoning

The research focus is **object detection** (drawing a bounding box around each litter instance and classifying it), rather than:

- **Classification** — would only say *whether* an image contains litter, without saying where. That's not useful on its own for a robot, or for the kind of per-object analysis this report does.
- **Segmentation** — TACO does provide pixel-level masks, but a full segmentation model is harder to evaluate meaningfully within this assignment's scope and isn't needed for the localization task that matters here (knowing roughly *where* an object is, not its exact pixel boundary).

Object detection is the right middle ground: it localizes each object (unlike classification) without requiring pixel-perfect masks (unlike segmentation), and it's directly supported by the dataset's bounding-box annotations.

A brief literature scan (`docs/literature_notes.md`) confirmed this is a reasonably well-studied but still active problem — TACO's own paper (Proença & Simões, 2020) reports baseline instance segmentation results; more recent work (2024–2025) applies YOLO-family detectors to litter/garbage detection, including on constrained hardware similar to the machine used here.

## 3. Dataset description

**Dataset:** [TACO — Trash Annotations in Context](http://tacodataset.org/), v1.0, downloaded from its official Zenodo archive (DOI [10.5281/zenodo.3587843](https://doi.org/10.5281/zenodo.3587843)), licensed CC-BY-4.0.

**Real, measured statistics** (not estimates — computed directly from the downloaded annotation file by `scripts/audit_dataset.py`; full output in `outputs/dataset_audit.json`):

| Property | Value |
|---|---|
| Images | 1,500 |
| Annotations (bounding boxes) | 4,784 |
| Raw annotated categories | 60 |
| Image width (min / median / max) | 842 / 2,448 / 6,000 px |
| Image height (min / median / max) | 474 / 3,264 / 5,312 px |
| Missing/corrupt image files | 0 |

**Class imbalance is severe**, which shaped this project's preprocessing: the most common raw category, `Cigarette`, has 667 instances, while 10 of the 60 raw categories have 5 or fewer instances each. Training a 60-way classifier head on this data would leave most classes with far too few examples to learn from.

**Taxonomy used for training:** the 60 raw categories were collapsed to their **17 supercategories with ≥50 instances**, plus one `other_litter` catch-all for everything rarer (18 classes total): `Plastic bag & wrapper (850), Cigarette (667), Unlabeled litter (517), Bottle (439), Bottle cap (289), Can (273), Other plastic (273), Carton (251), Cup (192), Straw (161), Paper (148), Broken glass (138), Styrofoam piece (112), Pop tab (99), Lid (87), Plastic container (72), Aluminium foil (62), other_litter`.

**Split:** 70/15/15 train/val/test by random image ID (fixed seed 441) → **1,050 / 225 / 225 images**, **3,385 / 712 / 680 boxes** (7 of 4,784 raw boxes were dropped during conversion for being zero-area or out-of-bounds). The split was checked for leakage by hashing every image file — no duplicates were found, so a random (not group-aware) split was appropriate. The test split was not touched again until final evaluation.

## 4. CNN training technique and code

**Model: YOLO11n** (Ultralytics), a convolutional neural network pretrained on COCO and fine-tuned on the TACO 18-class taxonomy above. It was chosen over a Faster R-CNN alternative on paper (`docs/phase3_model_selection.md`) for three reasons: its built-in augmentation controls, its direct use of a CNN backbone + transfer learning (which this course covers directly), and its fit to the available compute/time budget.

- **Architecture:** a single-stage, anchor-free detector — a CNN backbone (`C3k2` convolutional blocks) extracts image features, a `C2PSA` attention block refines them, and a decoupled detection head predicts a class and bounding box per location directly (no separate region-proposal stage).
- **Pretrained weights:** COCO. Since COCO's own 80 classes don't include litter-specific categories, the **entire network was fine-tuned** (not just a frozen-backbone linear probe) so the detection head could actually learn the new classes.
- **Training configuration:** 50 epochs, batch size 16, image size 640×640, seed 441, trained on an Apple M1 Pro (16GB RAM) using PyTorch's MPS backend. **All training-time augmentation was disabled** (no color jitter, flips, rotation, scaling, mosaic, or mixup) to keep this a clean, unmodified baseline — see the hyperparameters below.

```python
# scripts/train.py (excerpt)
E0_HYPERPARAMS = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    auto_augment=None,
)

model = YOLO("yolo11n.pt")
model.train(
    data="configs/dataset.yaml",
    epochs=50, imgsz=640, batch=16, device="mps",
    seed=441, deterministic=True,
    project="experiments", name="E0_baseline",
    **E0_HYPERPARAMS,
)
```

Training completed in **1.276 hours**. Loss decreased steadily and without instability throughout (full curves: `experiments/E0_baseline/results.png`):

![Training curves](../experiments/E0_baseline/results.png)

Validation metrics on the best checkpoint (selected on validation only): **precision 0.312, recall 0.154, mAP@0.5 0.158, mAP@0.5:0.95 0.122**.

## 5. Results on unseen data

### Locked test-set evaluation (TACO's held-out 225 images, evaluated once)

| Metric | Value |
|---|---:|
| Precision | 0.272 |
| Recall | 0.177 |
| **mAP@0.5** | **0.129** |
| **mAP@0.5:0.95** | **0.097** |

Test performance is modestly below validation, as expected since the checkpoint was selected on validation data. Per-class results (`outputs/test_metrics.json`) show the model learned some categories reasonably well — `Aluminium foil` (0.401 mAP50), `Bottle` (0.277), `Can` (0.287) — while others, most notably `Cigarette` (0.035 mAP50 despite being the single most frequent class, with 75 test instances), were almost entirely missed.

### Genuinely unseen external images (not from TACO)

To go beyond TACO's own held-out split, the trained model was also run on 5 images sourced independently from Wikimedia Commons (full provenance and licenses in `data/unseen_external/README.md`), with no ground-truth labels — a qualitative check only:

| Image | Result |
|---|---|
| Black plastic bag on grass | Correctly detected, 0.91 confidence, tight box |
| Beach scene (heavy litter) | 5 correct detections (bottles, bags, a bottle cap) despite far more clutter than any TACO training photo; clearly more items went undetected |
| Dense pile of bottles (recycling photo) | No detections — a composition unlike anything in TACO's training images |
| Cigarette butt in snow | No detection — consistent with the test-set finding |
| Cluttered ditch scene | One low-confidence, likely incorrect detection — no correct detections of the real debris present |

![Correct bag detection](figures/unseen_external_predictions/TRASH,_BLACK_PLASTIC_BAG_pred.jpg)
*Figure: correctly detected litter bag (0.91 confidence) on a genuinely external, never-before-seen image.*

### Discussion

The most consistent finding across both the locked TACO test set and the external images is a **specific, repeatable blind spot for small, low-contrast objects** — cigarettes are the dataset's single largest class by instance count, yet the model essentially never detects them, on either evaluation. This suggests the failure is about object size/visual salience, not training data volume. A second pattern is **sensitivity to scene composition**: the model performs reasonably on TACO-like scenes (one or a few litter items against a varied background) but fails on compositions it never saw in training — a dense, repetitive pile of objects, or a heavily cluttered/occluded scene.

This project's original research question asked whether training-time augmentation would improve detection of small/rare objects specifically. **That comparison was not run** — only the non-augmented baseline (E0) above was trained; the augmented condition (E1) was treated as a stretch goal and skipped in this pass. The baseline result reported here is real and directly motivates that follow-up (the `Cigarette` failure is exactly the kind of weakness augmentation is hypothesized to help with), but no claim is made here about whether augmentation would actually help — that remains untested.

---

*Figures and full metrics referenced above: `experiments/E0_baseline/` (training), `experiments/E0_baseline_test_eval/` (test evaluation), `report/figures/unseen_external_predictions/` (external images), `outputs/dataset_audit.json` and `outputs/test_metrics.json` (raw numbers).*
