# Agent progress log

| Phase | Status | Evidence |
|---|---|---|
| 0 — Understand requirements & formulate RQ | **Done — Gate 0 approved 2026-11-16** | Refined RQ1 (small/rare-class augmentation effect) approved by student. |
| 1 — Dataset investigation & acquisition | **Done — Gate 1 approved 2026-11-16** | TACO v1.0 downloaded & verified (sha256 logged), 1,500 images / 4,784 annotations / 60 categories / 0 missing files confirmed by `scripts/audit_dataset.py` → `outputs/dataset_audit.json`. 20 annotation overlays visually checked. |
| 2 — Data preprocessing | **Done — Gate 2 QA passed** | 18-class taxonomy (17 supercategories ≥50 instances + `other_litter`) and 70/15/15 split implemented in `scripts/prepare_dataset.py` → 1,050/225/225 images, 3,385/712/680 boxes, 7 invalid boxes dropped, 0 leakage. `scripts/test_coord_conversion.py` all pass. 3 after-remap overlays visually checked (`report/figures/after_remap_overlays/`). YOLO-format layout at `data/processed/`, Ultralytics `configs/dataset.yaml` written. |
| 3 — Architecture selection & baseline | **Done — Gate 3 passed** | YOLO11n selected over Faster R-CNN alternative (`docs/phase3_model_selection.md`). Smoke test + full 50-epoch E0 run completed in 1.276hr on M1 Pro/MPS. Real val metrics: mAP50=0.158, mAP50-95=0.122 (`docs/phase3_results.md`, `experiments/E0_baseline/`). |
| 4 — Controlled augmentation experiment | Not started (stretch goal) | — |
| 5 — Locked evaluation & new images | **Done — Gate 5 passed** | Locked test eval: mAP50=0.129, mAP50-95=0.097 (`outputs/test_metrics.json`). 5 genuinely external images (Wikimedia Commons, provenance in `data/unseen_external/README.md`) run through inference — 2 clean successes, 3 real failure modes analyzed (`docs/phase5_results.md`). |
| 6 — Report package & student review | **Draft written, awaiting your review/rewrite** | `report/REPORT.md` — maps to all 5 professor-required sections, pulls only real/logged numbers. AI-use disclosure intentionally omitted per your instruction (you're writing a separate file for that). Not yet rewritten in your own words/voice — do that before submitting. |
| 7 — Portfolio extension (optional) | Not started | — |

Dataset downloaded and audited (Phase 1), converted and split (Phase 2), E0 baseline trained with real logged metrics (Phase 3). Everything reported above is backed by committed artifacts — nothing here is simulated or estimated.
