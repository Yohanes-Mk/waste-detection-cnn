# Agent progress log

| Phase | Status | Evidence |
|---|---|---|
| 0 — Understand requirements & formulate RQ | **Done — Gate 0 approved 2026-11-16** | Refined RQ1 (small/rare-class augmentation effect) approved by student. |
| 1 — Dataset investigation & acquisition | **Done — Gate 1 approved 2026-11-16** | TACO v1.0 downloaded & verified (sha256 logged), 1,500 images / 4,784 annotations / 60 categories / 0 missing files confirmed by `scripts/audit_dataset.py` → `outputs/dataset_audit.json`. 20 annotation overlays visually checked. |
| 2 — Data preprocessing | **Done — Gate 2 QA passed** | 18-class taxonomy (17 supercategories ≥50 instances + `other_litter`) and 70/15/15 split implemented in `scripts/prepare_dataset.py` → 1,050/225/225 images, 3,385/712/680 boxes, 7 invalid boxes dropped, 0 leakage. `scripts/test_coord_conversion.py` all pass. 3 after-remap overlays visually checked (`report/figures/after_remap_overlays/`). YOLO-format layout at `data/processed/`, Ultralytics `configs/dataset.yaml` written. |
| 3 — Architecture selection & baseline | Not started | — |
| 4 — Controlled augmentation experiment | Not started | — |
| 5 — Locked evaluation & new images | Not started | — |
| 6 — Report package & student review | Not started | — |
| 7 — Portfolio extension (optional) | Not started | — |

No dataset has been downloaded. No model has been trained. No metrics in this repo are real until Phase 3+ produces logged run output.
