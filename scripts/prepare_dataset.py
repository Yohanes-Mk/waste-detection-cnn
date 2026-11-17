"""Phase 2: convert TACO (COCO-format) annotations into a YOLO-format training layout.

Steps:
  1. Build a compact class taxonomy: keep each supercategory with >=50 instances
     under its own name; fold every rarer supercategory into "other_litter".
  2. Validate every annotation (in-bounds, nonzero area) before converting.
  3. Split images 70/15/15 (train/val/test), fixed seed, after checking for
     duplicate-file leakage risk across the split.
  4. Write data/processed/{train,val,test}/images (symlinks to the immutable
     raw files) and .../labels/*.txt in YOLO format, plus dataset.yaml and
     data/splits/manifest.json.

No resizing/letterboxing is done here: Ultralytics' own dataloader letterboxes
at train/val time, and YOLO-format coordinates are already resolution-independent.
Doing it again here would risk double preprocessing (PRD Sec 5.2).

Usage: .venv/bin/python scripts/prepare_dataset.py
"""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TACO_DIR = ROOT / "data" / "raw" / "TACO" / "data"
ANNOTATIONS_PATH = TACO_DIR / "annotations.json"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLITS_DIR = ROOT / "data" / "splits"
MIN_SUPERCAT_INSTANCES = 50
SPLIT_SEED = 441
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def load_coco():
    with open(ANNOTATIONS_PATH) as f:
        return json.load(f)


def build_class_taxonomy(coco):
    """Return (category_id -> class_name, ordered class_name list)."""
    categories = {c["id"]: c for c in coco["categories"]}
    supercat_counts = {}
    for ann in coco["annotations"]:
        supercat = categories[ann["category_id"]]["supercategory"]
        supercat_counts[supercat] = supercat_counts.get(supercat, 0) + 1

    kept = {s: n for s, n in supercat_counts.items() if n >= MIN_SUPERCAT_INSTANCES}
    class_names = sorted(kept, key=lambda s: -kept[s]) + ["other_litter"]

    cat_id_to_class = {}
    for cat_id, cat in categories.items():
        supercat = cat["supercategory"]
        cat_id_to_class[cat_id] = supercat if supercat in kept else "other_litter"

    return cat_id_to_class, class_names, supercat_counts


def validate_bbox(bbox, width, height):
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return False
    if x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
        return False
    return True


def coco_bbox_to_yolo(bbox, width, height):
    x, y, w, h = bbox
    xc = (x + w / 2) / width
    yc = (y + h / 2) / height
    return xc, yc, w / width, h / height


def file_md5(path, block_size=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def check_duplicate_leakage(images):
    """Hash-check a sample for exact-duplicate files (full leakage risk check
    already done in Phase 1 audit; this re-verifies before splitting)."""
    seen = {}
    dupes = []
    for img in images.values():
        path = TACO_DIR / img["file_name"]
        h = file_md5(path)
        if h in seen:
            dupes.append((seen[h], img["file_name"]))
        else:
            seen[h] = img["file_name"]
    return dupes


def main():
    coco = load_coco()
    images = {img["id"]: img for img in coco["images"]}
    cat_id_to_class, class_names, supercat_counts = build_class_taxonomy(coco)

    print(f"Class taxonomy: {len(class_names)} classes -> {class_names}")

    dupes = check_duplicate_leakage(images)
    if dupes:
        raise RuntimeError(f"Found {len(dupes)} duplicate image files — fix before splitting: {dupes[:5]}")
    print(f"Leakage check: no duplicate files among {len(images)} images.")

    # Validate annotations, group by image
    anns_by_image = {}
    dropped = 0
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        if not validate_bbox(ann["bbox"], img["width"], img["height"]):
            dropped += 1
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)
    print(f"Validated annotations: dropped {dropped} invalid boxes out of {len(coco['annotations'])}.")

    # Split by image id
    image_ids = list(images.keys())
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(image_ids)
    n = len(image_ids)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    split_ids = {
        "train": image_ids[:n_train],
        "val": image_ids[n_train:n_train + n_val],
        "test": image_ids[n_train + n_val:],
    }
    assert set(split_ids["train"]) & set(split_ids["val"]) == set()
    assert set(split_ids["train"]) & set(split_ids["test"]) == set()
    assert set(split_ids["val"]) & set(split_ids["test"]) == set()
    assert sum(len(v) for v in split_ids.values()) == n

    class_to_idx = {name: i for i, name in enumerate(class_names)}

    manifest = {
        "seed": SPLIT_SEED,
        "ratios": SPLIT_RATIOS,
        "class_names": class_names,
        "min_supercategory_instances": MIN_SUPERCAT_INSTANCES,
        "supercategory_instance_counts": supercat_counts,
        "splits": {},
    }

    for split, ids in split_ids.items():
        img_dir = PROCESSED_DIR / split / "images"
        lbl_dir = PROCESSED_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        class_instance_counts = {name: 0 for name in class_names}
        n_boxes_written = 0

        for img_id in ids:
            img = images[img_id]
            src_path = (TACO_DIR / img["file_name"]).resolve()
            stem = img["file_name"].replace("/", "_").rsplit(".", 1)[0]
            ext = img["file_name"].rsplit(".", 1)[-1]
            link_path = img_dir / f"{stem}.{ext}"
            if not link_path.exists():
                link_path.symlink_to(src_path)

            lines = []
            for ann in anns_by_image.get(img_id, []):
                class_name = cat_id_to_class[ann["category_id"]]
                class_idx = class_to_idx[class_name]
                xc, yc, w, h = coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
                lines.append(f"{class_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                class_instance_counts[class_name] += 1
                n_boxes_written += 1

            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        manifest["splits"][split] = {
            "num_images": len(ids),
            "num_boxes": n_boxes_written,
            "class_instance_counts": class_instance_counts,
            "image_ids": ids,
        }
        print(f"{split}: {len(ids)} images, {n_boxes_written} boxes written.")

    SPLITS_DIR.mkdir(exist_ok=True)
    with open(SPLITS_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    dataset_yaml = (
        f"path: {PROCESSED_DIR}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names)) + "\n"
    )
    (ROOT / "configs").mkdir(exist_ok=True)
    (ROOT / "configs" / "dataset.yaml").write_text(dataset_yaml)
    print(f"Wrote {ROOT / 'configs' / 'dataset.yaml'} and {SPLITS_DIR / 'manifest.json'}")

    # Flag any class missing entirely from train or val (known limitation to disclose, not silently fix)
    for split in ["train", "val"]:
        missing = [c for c, n in manifest["splits"][split]["class_instance_counts"].items() if n == 0]
        if missing:
            print(f"WARNING: classes with zero instances in {split}: {missing}")


if __name__ == "__main__":
    main()
