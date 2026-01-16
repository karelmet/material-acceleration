# src/cadot/data/loading.py
import os
from pathlib import Path
from cadot.utils.path import get_data_path
import json
import torch
import torchvision

from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

def load_yolo_annotations(label_path, img_width, img_height):
    """Load YOLO format annotations from txt file."""
    annotations = []
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            # chaque ligne contient 5 éléments : class_id et 2 coordonnées pour la box, width et height
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            
            # Convert from YOLO format (normalized center coords) to pixel coords
            x_center = cx * img_width
            y_center = cy * img_height
            box_width = w * img_width
            box_height = h * img_height
            
            # Convert to corner coordinates
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            
            annotations.append({
                'class_id': class_id,
                'bbox': [x1, y1, box_width, box_height]
            })
    
    return annotations

def load_coco_annotations(coco_dict, image_id):
    """
    Loads COCO format annotations for a given image ID.
    """
    annotations = []

    # toutes les annotations correspondant à cet id d'image
    for ann in coco_dict["annotations"]:
        if ann["image_id"] == image_id:
            annotations.append({
                "class_id": ann["category_id"],
                "bbox": ann["bbox"]  # [x_min, y_min, w, h]
            })

    return annotations


def get_image_label_pairs_yolo(data_dir_name, split="train"):
    """
    Return list of (image_path, label_path) pairs for a given split.
    
    Args:
        split (str): 'train' or 'valid' for the YOLO dataset
    
    Returns:
        List[Tuple[Path, Path]]: aligned paths for images and labels.
    """

    data_root = get_data_path(data_dir_name)

    images_dir = data_root / "images" / split
    labels_dir = data_root / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # toutes les images .jpg dans le split
    image_paths = sorted(images_dir.glob("*.jpg"))
    
    pairs = []

    for img_path in image_paths:
        label_path = labels_dir / (img_path.stem + ".txt")
        
        if not label_path.exists():
            continue

        pairs.append((img_path, label_path))

    return pairs

def get_image_label_pairs_coco(data_dir_name):
    """
    Retourne toutes les images + leur image_id COCO :
       pairs = [(img_path, image_id), ...]
    """

    data_root = get_data_path(data_dir_name)

    images_dir = data_root / "train"
    coco_path = data_root / "train" / "_annotations.coco.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not coco_path.exists():
        raise FileNotFoundError(f"COCO json not found: {coco_path}")

    # charge le coco
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # mapping : filename → id
    name_to_id = {img["file_name"]: img["id"] for img in coco["images"]}

    image_paths = sorted(images_dir.glob("*.jpg"))

    pairs = []
    for img_path in image_paths:
        img_name = img_path.name
        if img_name in name_to_id:
            pairs.append((img_path, name_to_id[img_name]))

    return pairs, coco


class CocoWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        img = F.to_tensor(img)

        boxes = []
        labels = []
        for t in targets:
            boxes.append(t["bbox"])
            labels.append(t["category_id"])

        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            boxes = torchvision.ops.box_convert(boxes, "xywh", "xyxy")
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        return img, target
    
    
