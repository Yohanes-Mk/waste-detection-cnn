"""Phase 5: final locked evaluation on the TACO test split (touched once, here).

Usage: .venv/bin/python scripts/evaluate.py
Writes outputs/test_metrics.json (precision/recall/mAP50/mAP50-95, overall + per-class).
"""

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "experiments" / "E0_baseline" / "weights" / "best.pt"


def main():
    model = YOLO(str(CHECKPOINT))
    metrics = model.val(
        data=str(ROOT / "configs" / "dataset.yaml"),
        split="test",
        imgsz=640,
        batch=16,
        device="mps",
        project=str(ROOT / "experiments"),
        name="E0_baseline_test_eval",
        exist_ok=True,
        plots=True,
    )

    names = metrics.names
    ap50 = metrics.box.ap50          # per-class AP@0.5
    ap5095 = metrics.box.ap          # per-class AP@0.5:0.95
    cls_idx = metrics.box.ap_class_index  # classes with support in this split

    per_class = {
        names[int(ci)]: {"mAP50": float(ap50[i]), "mAP50_95": float(ap5095[i])}
        for i, ci in enumerate(cls_idx)
    }

    out = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "split": "test",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "per_class": per_class,
    }

    out_path = ROOT / "outputs" / "test_metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
