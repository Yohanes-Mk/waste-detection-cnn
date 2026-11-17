"""Phase 3/4: fine-tune YOLO11n on the TACO 18-class taxonomy.

Two experiment configs, otherwise identical (same seed, split, epochs, model size):
  E0 (baseline):  all training-time augmentation disabled.
  E1 (augmented): a specified, modest augmentation config enabled.

Usage:
  .venv/bin/python scripts/train.py --exp E0 --epochs 1          # smoke test
  .venv/bin/python scripts/train.py --exp E0 --epochs 60         # real baseline run
  .venv/bin/python scripts/train.py --exp E1 --epochs 60         # augmented run (Phase 4)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
SEED = 441

# E0: everything that introduces train-time variation is zeroed out.
E0_HYPERPARAMS = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    auto_augment=None,
)

# E1: same base config as E0, plus a specified, modest augmentation set
# (brightness/contrast via hsv_v, moderate scale/crop, horizontal flip).
# Deliberately does NOT touch anything that could corrupt small litter objects
# (no heavy rotation/shear/mosaic/mixup/copy_paste).
E1_HYPERPARAMS = dict(
    hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
    degrees=0.0, translate=0.1, scale=0.3, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.5,
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    auto_augment=None,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["E0", "E1"], required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    hyp = E0_HYPERPARAMS if args.exp == "E0" else E1_HYPERPARAMS
    name = f"{args.exp}_{'baseline' if args.exp == 'E0' else 'augmented'}"

    model = YOLO("yolo11n.pt")
    model.train(
        data=str(ROOT / "configs" / "dataset.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=SEED,
        deterministic=True,
        project=str(ROOT / "experiments"),
        name=name,
        exist_ok=True,
        plots=True,
        **hyp,
    )


if __name__ == "__main__":
    main()
