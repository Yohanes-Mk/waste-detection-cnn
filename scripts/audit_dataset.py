"""Audit the downloaded TACO dataset: real counts, dimensions, class balance, missing files.

Usage: .venv/bin/python scripts/audit_dataset.py
Reads data/raw/TACO/data/annotations.json, writes outputs/dataset_audit.json
and a handful of annotation-overlay PNGs to outputs/overlays/ for visual spot-checking.
"""

import json
import random
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
TACO_DIR = ROOT / "data" / "raw" / "TACO" / "data"
ANNOTATIONS_PATH = TACO_DIR / "annotations.json"
OUT_DIR = ROOT / "outputs"
OVERLAY_DIR = OUT_DIR / "overlays"


def load_annotations():
    with open(ANNOTATIONS_PATH) as f:
        return json.load(f)


def bbox_bucket(bbox):
    # bbox = [x, y, w, h] in COCO format; bucket by area in pixels.
    area = bbox[2] * bbox[3]
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def main():
    coco = load_annotations()
    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat for cat in coco["categories"]}
    annotations = coco["annotations"]

    # Missing-file check
    missing = []
    for img in images.values():
        if not (TACO_DIR / img["file_name"]).exists():
            missing.append(img["file_name"])

    widths = [img["width"] for img in images.values()]
    heights = [img["height"] for img in images.values()]

    # Instances and images per category
    instances_per_cat = {}
    images_per_cat = {}
    size_bucket_counts = {"small": 0, "medium": 0, "large": 0}
    supercategory_counts = {}

    for ann in annotations:
        cat_id = ann["category_id"]
        cat_name = categories[cat_id]["name"]
        supercat = categories[cat_id].get("supercategory", "unknown")
        instances_per_cat[cat_name] = instances_per_cat.get(cat_name, 0) + 1
        supercategory_counts[supercat] = supercategory_counts.get(supercat, 0) + 1
        size_bucket_counts[bbox_bucket(ann["bbox"])] += 1
        images_per_cat.setdefault(cat_name, set()).add(ann["image_id"])

    images_per_cat_counts = {k: len(v) for k, v in images_per_cat.items()}

    sorted_cats = sorted(instances_per_cat.items(), key=lambda kv: -kv[1])
    rare_classes = [c for c, n in sorted_cats if n <= 5]

    audit = {
        "source": "Zenodo DOI 10.5281/zenodo.3587843, TACO.zip v1.0, CC-BY-4.0",
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories_raw": len(categories),
        "num_supercategories": len(supercategory_counts),
        "missing_files_count": len(missing),
        "missing_files_sample": missing[:10],
        "image_width": {
            "min": min(widths), "median": statistics.median(widths), "max": max(widths),
        },
        "image_height": {
            "min": min(heights), "median": statistics.median(heights), "max": max(heights),
        },
        "instances_per_category": dict(sorted_cats),
        "images_per_category": images_per_cat_counts,
        "supercategory_instance_counts": supercategory_counts,
        "bbox_size_bucket_counts": size_bucket_counts,
        "rare_categories_leq_5_instances": rare_classes,
        "num_rare_categories_leq_5_instances": len(rare_classes),
    }

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "dataset_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    # Save N random overlays for visual spot-check
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(441)
    img_ids_with_anns = list({ann["image_id"] for ann in annotations})
    sample_ids = random.sample(img_ids_with_anns, k=min(20, len(img_ids_with_anns)))

    ann_by_image = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    for img_id in sample_ids:
        img_info = images[img_id]
        img_path = TACO_DIR / img_info["file_name"]
        if not img_path.exists():
            continue
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for ann in ann_by_image[img_id]:
            x, y, w, h = ann["bbox"]
            cat_name = categories[ann["category_id"]]["name"]
            draw.rectangle([x, y, x + w, y + h], outline="red", width=4)
            draw.text((x, max(0, y - 14)), cat_name, fill="red")
        out_name = img_info["file_name"].replace("/", "_")
        im.save(OVERLAY_DIR / out_name)

    print(f"Wrote {OUT_DIR / 'dataset_audit.json'}")
    print(f"Wrote {len(sample_ids)} overlay images to {OVERLAY_DIR}")
    print(f"num_images={audit['num_images']} num_annotations={audit['num_annotations']} "
          f"num_categories_raw={audit['num_categories_raw']} missing_files={audit['missing_files_count']}")


if __name__ == "__main__":
    main()
