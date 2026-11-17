"""Lightweight correctness tests for the COCO->YOLO bbox conversion and split
integrity used by prepare_dataset.py. Plain asserts, no test framework needed
(matches the project's lightweight-tooling decision).

Usage: .venv/bin/python scripts/test_coord_conversion.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import coco_bbox_to_yolo, validate_bbox, MIN_SUPERCAT_INSTANCES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def yolo_to_coco_bbox(xc, yc, w, h, width, height):
    """Inverse of coco_bbox_to_yolo, for round-trip testing."""
    bw, bh = w * width, h * height
    x = xc * width - bw / 2
    y = yc * height - bh / 2
    return x, y, bw, bh


def test_round_trip_synthetic():
    cases = [
        ((0, 0, 10, 10), 100, 100),
        ((50, 50, 20, 30), 200, 150),
        ((0, 0, 100, 100), 100, 100),  # full-image box
        ((842 - 1, 474 - 1, 1, 1), 842, 474),  # 1px box at smallest real image size
    ]
    for bbox, width, height in cases:
        xc, yc, w, h = coco_bbox_to_yolo(bbox, width, height)
        assert 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0, f"center out of [0,1] for {bbox}"
        assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0, f"size out of [0,1] for {bbox}"
        x2, y2, w2, h2 = yolo_to_coco_bbox(xc, yc, w, h, width, height)
        orig = bbox
        assert abs(x2 - orig[0]) < 1e-6, (x2, orig)
        assert abs(y2 - orig[1]) < 1e-6, (y2, orig)
        assert abs(w2 - orig[2]) < 1e-6, (w2, orig)
        assert abs(h2 - orig[3]) < 1e-6, (h2, orig)
    print("PASS: synthetic round-trip conversions")


def test_round_trip_real_annotations(n=25):
    with open(ROOT / "data" / "raw" / "TACO" / "data" / "annotations.json") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco["images"]}
    for ann in coco["annotations"][:n]:
        img = images[ann["image_id"]]
        xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
        x2, y2, w2, h2 = yolo_to_coco_bbox(xc, yc, w, h, img["width"], img["height"])
        for a, b in zip(ann["bbox"], (x2, y2, w2, h2)):
            assert abs(a - b) < 0.01, (ann["bbox"], (x2, y2, w2, h2))
    print(f"PASS: round-trip on {n} real annotations")


def test_validate_bbox_rejects_bad_boxes():
    assert validate_bbox([10, 10, 0, 5], 100, 100) is False  # zero width
    assert validate_bbox([10, 10, 5, 0], 100, 100) is False  # zero height
    assert validate_bbox([-5, 10, 5, 5], 100, 100) is False  # negative x
    assert validate_bbox([95, 95, 20, 20], 100, 100) is False  # extends past bounds
    assert validate_bbox([10, 10, 5, 5], 100, 100) is True  # valid
    print("PASS: validate_bbox rejects degenerate/out-of-bounds boxes")


def test_split_manifest_integrity():
    with open(ROOT / "data" / "splits" / "manifest.json") as f:
        manifest = json.load(f)
    ids = {s: set(manifest["splits"][s]["image_ids"]) for s in ("train", "val", "test")}
    assert ids["train"] & ids["val"] == set(), "train/val overlap!"
    assert ids["train"] & ids["test"] == set(), "train/test overlap!"
    assert ids["val"] & ids["test"] == set(), "val/test overlap!"
    total = sum(len(v) for v in ids.values())
    assert total == 1500, f"expected 1500 images across splits, got {total}"
    print(f"PASS: split manifest has no train/val/test overlap ({total} images total)")


def test_class_taxonomy_threshold_applied():
    with open(ROOT / "data" / "splits" / "manifest.json") as f:
        manifest = json.load(f)
    kept_supercats = [s for s, n in manifest["supercategory_instance_counts"].items()
                       if n >= MIN_SUPERCAT_INSTANCES]
    class_names = manifest["class_names"]
    assert set(kept_supercats) <= set(class_names), "a kept supercategory is missing from class_names"
    assert "other_litter" in class_names
    assert len(class_names) == len(kept_supercats) + 1
    print(f"PASS: class taxonomy = {len(kept_supercats)} supercategories (>= {MIN_SUPERCAT_INSTANCES} "
          f"instances) + other_litter = {len(class_names)} classes")


if __name__ == "__main__":
    test_round_trip_synthetic()
    test_round_trip_real_annotations()
    test_validate_bbox_rejects_bad_boxes()
    test_split_manifest_integrity()
    test_class_taxonomy_threshold_applied()
    print("ALL TESTS PASSED")
