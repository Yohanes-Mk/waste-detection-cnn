"""Phase 5: run the trained E0 checkpoint on genuinely external, never-before-seen
images (not from TACO at all) and save annotated predictions.

These are qualitative-only (no ground-truth boxes exist for them) — kept
distinct from the metric-bearing locked test-set evaluation in
scripts/evaluate.py / outputs/test_metrics.json.

Usage: .venv/bin/python scripts/predict.py
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "data" / "unseen_external"
OUT_DIR = ROOT / "report" / "figures" / "unseen_external_predictions"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(ROOT / "experiments" / "E0_baseline" / "weights" / "best.pt"))

    images = sorted(p for p in SRC_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    for img_path in images:
        results = model.predict(source=str(img_path), conf=0.15, imgsz=640, device="mps", verbose=False)
        r = results[0]
        out_path = OUT_DIR / f"{img_path.stem}_pred.jpg"
        r.save(filename=str(out_path))
        boxes = r.boxes
        detections = []
        if boxes is not None:
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                detections.append(f"{r.names[int(cls_id)]} ({conf:.2f})")
        print(f"{img_path.name}: {len(detections)} detections -> {detections}")
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
