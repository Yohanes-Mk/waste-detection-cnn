# Phase 3 — E0 baseline results

**Real, logged training run** — not estimated or simulated. Config: `experiments/E0_baseline/args.yaml`. Full per-epoch curves: `experiments/E0_baseline/results.csv` / `results.png`. Checkpoints: `experiments/E0_baseline/weights/{best,last}.pt`.

- **Model:** YOLO11n, pretrained on COCO, fine-tuned on the 18-class TACO taxonomy.
- **Config:** 50 epochs, batch 16, imgsz 640, seed 441, all training-time augmentation disabled (hsv/degrees/translate/scale/shear/flip/mosaic/mixup/copy_paste = 0) — see `scripts/train.py` `E0_HYPERPARAMS`.
- **Hardware/runtime:** Apple M1 Pro, 16GB RAM, PyTorch MPS backend. **1.276 hours wall-clock** for all 50 epochs.
- **Single run, single seed** — acknowledged limitation (PRD §6): repeating with multiple seeds was impractical in this assignment's time budget on this hardware.

## Validation metrics (best checkpoint, selected on validation — test set not yet touched)

| Metric | Value |
|---|---|
| Precision | 0.312 |
| Recall | 0.154 |
| mAP@0.5 | 0.158 |
| mAP@0.5:0.95 | 0.122 |

## Per-class mAP@0.5 (best checkpoint)

| Class | Images | Instances | mAP50 |
|---|---:|---:|---:|
| Aluminium foil | 5 | 8 | 0.495 |
| Bottle cap | 39 | 56 | 0.401 |
| Bottle | 42 | 55 | 0.373 |
| Lid | 10 | 11 | 0.340 |
| Plastic bag & wrapper | 69 | 98 | 0.280 |
| Cup | 30 | 37 | 0.276 |
| Carton | 30 | 50 | 0.194 |
| Can | 30 | 47 | 0.149 |
| Plastic container | 8 | 10 | 0.114 |
| Straw | 19 | 22 | 0.087 |
| Styrofoam piece | 7 | 12 | 0.033 |
| other_litter | 14 | 19 | 0.032 |
| Unlabeled litter | 40 | 73 | 0.040 |
| Paper | 13 | 16 | 0.007 |
| **Cigarette** | 30 | **95** | **0.010** |
| Other plastic | 26 | 39 | 0.011 |
| Broken glass | 3 | 53 | 0.000 |
| Pop tab | 10 | 11 | 0.000 |

## Interpretation (preliminary — this is the baseline, not the final discussion)

- **This is a real, informative finding, not just a weak baseline:** `Cigarette` is the single most frequent class in the dataset (667 training instances, 95 in val) and the model still scores essentially zero AP on it. Cigarettes are tiny, low-contrast objects against varied backgrounds — this is direct evidence that raw instance count alone doesn't predict detectability, and it's exactly the kind of result the refined RQ1 (does augmentation help small/rare objects disproportionately) is positioned to speak to once E1 exists for comparison.
- `Broken glass` and `Pop tab` scoring zero despite reasonable instance counts likely reflects low visual distinctiveness against cluttered outdoor backgrounds (both are small, often-transparent or metallic, low-contrast objects) — a plausible qualitative failure mode to expand on in Phase 5's error analysis, not asserted as fact yet.
- `Aluminium foil`'s unexpectedly strong score (0.495) has very thin support (5 images, 8 instances) — flagging this now so it isn't later over-interpreted as a strength; small-support classes are noisy by nature.
- Overall mAP@0.5 of 0.158 after 50 epochs on 1,050 images with no augmentation is a plausible, unremarkable baseline for this dataset size/class count — consistent with TACO's own documented difficulty (`docs/literature_notes.md`), not a sign of a broken pipeline.

## Gate 3

Baseline is reproducible: config, seed, checkpoints, logs, and curves are all saved and committed. Real (not fabricated) validation metrics are in hand. Ready for Phase 4 (E1 — same config, augmentation enabled) per your earlier instruction to treat it as a stretch goal now that the baseline exists.
