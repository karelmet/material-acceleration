#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys

import torch

# ----------------------------------------------------------------------
# 1) Ajouter src/ au PYTHONPATH à partir de l'emplacement de ce fichier
# ----------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent          # dossier où se trouve train_yolo_baseline.py
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ----------------------------------------------------------------------
# 2) Imports du projet
# ----------------------------------------------------------------------
from cadot.utils.path import get_data_path  # type: ignore
from cadot.models.yolo import (             # type: ignore
    CADOT_CLASS_NAMES,
    prepare_yolo_dataset,
    train_yolo,
)

# ----------------------------------------------------------------------
# 3) Entrainement du modèle sur 2 classes uniquement
# ----------------------------------------------------------------------
def main() -> None:
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("Using device: 0")

    data_root = get_data_path()
    print("Data root:", data_root)

    out_dir = PROJECT_ROOT / "yolo_dataset_split"

    # 1) Préparer le dataset YOLO (train/val + data.yaml)
    data_yaml = prepare_yolo_dataset(out_dir=out_dir, class_names=CADOT_CLASS_NAMES)

    # 2) Entraîner le modèle YOLOv8n
    results = train_yolo(
        data=str(data_yaml),
        model="yolov8s.pt",
        epochs=80,
        imgsz=640,
        batch=16,
        device="0",   # GPU NVIDIA
        val=True,
        classes=[1, 10], # training on "Buildings" and "Small Vehicle" only
    )
    return results


if __name__ == "__main__":
    main()