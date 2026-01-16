#!/usr/bin/env python

from __future__ import annotations
from pathlib import Path
import sys
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cadot.utils.path import get_data_path  # type: ignore
from cadot.models.yolo_loss_balance import (# type: ignore
    CADOT_CLASS_NAMES,
    prepare_yolo_dataset,
    train_yolo_balanced,
)

def main() -> None:
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Using device: 0")

    data_root = get_data_path(data_dir_name="merged_dataset")
    print("Data root:", data_root)

    out_dir = PROJECT_ROOT / "yolo_dataset_split"

    data_yaml = prepare_yolo_dataset(out_dir=out_dir, class_names=CADOT_CLASS_NAMES)

    results = train_yolo_balanced(
        data=str(data_yaml),
        model="yolov8s.pt",
        epochs=80,
        imgsz=640,
        batch=16,
        device="mps",   # GPU NVIDIA > device = "0" ; macbook > device="mps"
        val=True,
        classes=CADOT_CLASS_NAMES,
        # pas besoin de passer class_weights en arguments car c'est la valeur par défaut
    )
    return results


if __name__ == "__main__":
    main()