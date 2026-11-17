# Phase 5 — locked test evaluation & unseen-image results

## Locked test-set metrics (TACO, 225 images held out since Phase 2, touched for the first time here)

Full numbers: `outputs/test_metrics.json`. Produced by `model.val(..., split='test')` on `experiments/E0_baseline/weights/best.pt` — the checkpoint selected on *validation* only, never on test.

| Metric | Validation (Phase 3) | **Test (locked, final)** |
|---|---:|---:|
| Precision | 0.312 | 0.272 |
| Recall | 0.154 | 0.177 |
| mAP@0.5 | 0.158 | **0.129** |
| mAP@0.5:0.95 | 0.122 | **0.097** |

Test mAP is somewhat lower than validation, as expected — the checkpoint was selected on val, so val numbers carry a small optimistic bias. This is the number that goes in the report as "the" result, not the validation number.

Per-class breakdown (test set) confirms the same pattern seen in Phase 3: `Aluminium foil` (0.401 mAP50, thin support — 9 instances) and `Bottle`/`Can` (0.28–0.34 mAP50) score reasonably; `Cigarette` (0.035 mAP50 despite 75 test instances — the most frequent class) and `Broken glass` (0.0) remain the weakest. Full table in `outputs/test_metrics.json`.

## Genuinely unseen external images (qualitative, not metric-bearing)

5 images from Wikimedia Commons (provenance + license in `data/unseen_external/README.md`), none related to TACO. Predictions: `report/figures/unseen_external_predictions/`.

| Image | Detections (conf) | Assessment |
|---|---|---|
| `TRASH,_BLACK_PLASTIC_BAG.JPG` | Plastic bag & wrapper (0.91) | **Correct, confident, tight box.** Clean success — visually similar composition to TACO (single item, grass background). |
| `Litter.JPG` (beach) | Bottle (0.67, 0.54), Plastic bag & wrapper (0.49, 0.44), Bottle cap (0.36) | **Partial success.** 5 real objects found in a much more cluttered multi-object scene than any TACO training photo; clearly many more visible litter items went undetected (recall-limited under clutter), but no obvious false positives (e.g., the prominent tires were correctly left unlabeled). |
| `Plastic_Bottles_-_Waste_(...).jpg` | *(none)* | **Clear failure — distribution shift.** A dense, repetitive close-up pile of bottles; nothing like TACO's typical "single/sparse item on ground" composition. Illustrates that this model generalizes to TACO-like scenes, not to arbitrary bottle imagery. |
| `Filthy_Habit_...jpg` (cigarette butt in snow) | *(none)* | **Failure, but a consistent one.** Matches the test-set finding exactly: `Cigarette` is the model's weakest learned class despite being the most frequent in training — small, low-contrast objects are the model's systematic blind spot, not a one-off miss. |
| `Ditch_litter_5622.jpg` | Cup (0.51) | **Likely false positive.** The boxed region does not clearly contain a cup; a cluttered, occluded, non-TACO-typical background (ditch water, debris) appears to have triggered a spurious detection rather than finding any of the real debris visible in the scene. |

## Error analysis — representative failure modes

1. **Small, low-contrast objects are systematically under-detected.** `Cigarette` scores near-zero mAP on both the TACO test set (0.035, despite 75 instances — the most frequent class) and the one external cigarette-butt photo tried. This is the single clearest, most consistent failure mode, and it's exactly the dataset property the refined RQ1 targets — this baseline result is the evidence that motivates testing whether augmentation (E1, not yet run) would help here.
2. **Composition/distribution shift to dense, repetitive scenes.** The bottle-pile image (no TACO analog: many identical overlapping objects, shallow depth of field, no background context) produced zero detections. The model has learned TACO's typical "isolated item against a background" framing, not generic bottle recognition.
3. **Heavy occlusion / unusual background → false positives rather than true detections.** In the ditch scene, the model produced one low-to-moderate-confidence, likely-incorrect box instead of detecting any of the genuinely visible debris — suggesting that outside TACO's typical framing, the model can mis-fire on clutter rather than simply abstaining.
4. **Recall degrades with scene density**, even when precision holds up. The beach scene shows multiple correct detections with no visible false positives, but clearly undercounts the many litter items in frame — consistent with TACO's training images containing far fewer simultaneous objects per frame than this scene.

## Gate 5

Locked test metrics (mAP@0.5=0.129, mAP@0.5:0.95=0.097) and 5 genuinely external qualitative predictions with an honest, specific error analysis are in hand — both real, logged, and consistent with each other (the `Cigarette` weakness shows up identically in both). This supports an evidence-based conclusion for the report: **the baseline detector works on TACO-like single/sparse-object scenes (confirmed on both the held-out TACO test split and one externally sourced image with similar composition), but has a specific, repeatable blind spot for small/low-contrast objects and degrades on scene compositions (dense piles, heavy occlusion) unlike anything in its training distribution.** H1 (does augmentation help) remains untested — E1 was not run (stretch goal, skipped per your instruction) — so the report should state this as a disclosed limitation, not a resolved finding.
