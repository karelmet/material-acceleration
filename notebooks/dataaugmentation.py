#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path
import sys
from PIL import Image
import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent          # dossier où se trouve train_yolo_baseline.py
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from cadot.data.loading import load_yolo_annotations, get_image_label_pairs
from cadot.data.augmentation import simple_augmentation, random_horizontal_flip, random_brightness, random_contrast
from cadot.utils.path import get_data_path
from cadot.data.visualisation import plot_image_with_annotations, yolo_boxes_to_annotations
from cadot.models.yolo import CADOT_CLASS_NAMES

# ------------------------------------------
# 
# ------------------------------------------

plot_results = False
save_aug = False

pairs = get_image_label_pairs("train")

for img_path, label_path in pairs:
    img = Image.open(img_path)
    annotations = load_yolo_annotations(label_path, img.width, img.height)
    img = img.convert("RGB")
    img_np = np.array(img)
    boxes = np.loadtxt(label_path, dtype=float).reshape(-1, 5)

    img_aug, boxes_aug = simple_augmentation(img_np, boxes)
    
    # Plot
    if plot_results:
        h, w = img_aug.shape[:2]
        annotations_aug = yolo_boxes_to_annotations(boxes_aug, w, h)
        plot_image_with_annotations(img, annotations, class_names=CADOT_CLASS_NAMES)
        plot_image_with_annotations(img_aug, annotations_aug, class_names=CADOT_CLASS_NAMES)
    
    # Save
    if save_aug:
        img_path = Path(img_path)
        label_path = Path(label_path)

        img_dir = img_path.parent
        label_dir = label_path.parent

        # Nouveau nom de base
        stem = img_path.stem 
        aug_img_path = img_dir / "augmentation" / f"{stem}_aug.jpg"
        aug_label_path = label_dir / "augmentation" / f"{stem}_aug.txt"

        # Sauvegarde de l'image
        Image.fromarray(img_aug).save(aug_img_path, quality=95)

        # Sauvegarde des labels au format YOLO
        boxes_to_save = np.asarray(boxes_aug, dtype=float).reshape(-1, 5)

        with open(aug_label_path, "w", encoding="utf-8") as f:
            for cls, xc, yc, w_norm, h_norm in boxes_to_save:
                f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w_norm:.6f} {h_norm:.6f}\n")
