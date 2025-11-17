import os
import cv2
import random
import numpy as np
from pathlib import Path

DATA_ROOT = "DataCadot"
IMG_DIR = os.path.join(DATA_ROOT, "images", "train")
LBL_DIR = os.path.join(DATA_ROOT, "labels", "train")

OUT_ROOT = "dataset_aug"
OUT_IMG_DIR = os.path.join(OUT_ROOT, "images", "train")
OUT_LBL_DIR = os.path.join(OUT_ROOT, "labels", "train")
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

MINORITY_CLASSES = [0,3,4,7,8,11,12,13]          # Classes à augmenter
AUG_PER_OBJECT = 3              # Nombre de fois à augmenter chaque objet
MAX_TRIES = 40                  # Nombre max d'essais pour trouver une image de destination
RESIZE_IF_TOO_BIG = True        # Redimensionner le patch s'il est trop grand pour l'image de destination    

random.seed(42)
np.random.seed(42)

def list_image_files(dirpath):
    exts = (".jpg", ".jpeg", ".png")
    return [f for f in os.listdir(dirpath) if f.lower().endswith(exts)]

def load_labels(label_path):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])         # classe de l'objet
            xc = float(parts[1])        # x centrée normalisée
            yc = float(parts[2])        # y centrée normalisée
            w = float(parts[3])         # largeur normalisée
            h = float(parts[4])         # hauteur normalisée
            boxes.append((cls, xc, yc, w, h))
    return boxes

def write_labels(path, boxes):
    with open(path, "w") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def norm_to_pixels_yolo(box, W, H):
    """Convertit une boîte YOLO normalisée en pixels."""
    cls, xc, yc, w, h = box
    w_px = int(round(w * W))
    h_px = int(round(h * H))
    x_center = xc * W
    y_center = yc * H
    x_min = int(round(x_center - w_px / 2))
    y_min = int(round(y_center - h_px / 2))

    # clamp
    x_min = max(0, min(x_min, W - 1))
    y_min = max(0, min(y_min, H - 1))
    w_px = max(1, min(w_px, W - x_min))
    h_px = max(1, min(h_px, H - y_min))

    return cls, x_min, y_min, w_px, h_px

def pixels_to_norm_yolo(x, y, w, h, W, H):
    """Convertit une boîte en pixels en YOLO normalisée."""
    xc = (x + w / 2) / W
    yc = (y + h / 2) / H
    return xc, yc, w / W, h / H

# Charger les images
image_files = list_image_files(IMG_DIR)
if not image_files:
    raise SystemExit("Aucune image dans: " + IMG_DIR)

# Charger tous les labels en mémoire
label_map = {img: load_labels(os.path.join(LBL_DIR, Path(img).with_suffix(".txt"))) for img in image_files}

print("Total images :", len(image_files))

# MAIN
for src_file in image_files:
    src_img_path = os.path.join(IMG_DIR, src_file)
    src_lbl_path = os.path.join(LBL_DIR, Path(src_file).with_suffix(".txt").name)

    src_img = cv2.imread(src_img_path)
    if src_img is None:
        continue

    Hs, Ws = src_img.shape[:2]
    src_boxes = load_labels(src_lbl_path)

    for box in src_boxes:
        cls = box[0]
        if cls not in MINORITY_CLASSES:
            continue

        # YOLO en pixels
        cls_px, x_px, y_px, w_px, h_px = norm_to_pixels_yolo(box, Ws, Hs)

        patch = src_img[y_px:y_px + h_px, x_px:x_px + w_px].copy()
        if patch.size == 0:
            print("Patch vide --> ignoré")
            continue

        for aug_i in range(AUG_PER_OBJECT):

            aug_patch = patch.copy()

            # Choix destination
            candidates = [f for f in image_files if f != src_file]
            tries = 0
            dst_file = None

            while tries < MAX_TRIES:
                df = random.choice(candidates)
                dst_img = cv2.imread(os.path.join(IMG_DIR, df))
                if dst_img is None:
                    tries += 1
                    continue

                Hd, Wd = dst_img.shape[:2]

                ph, pw = aug_patch.shape[:2]

                if (pw > Wd or ph > Hd) and RESIZE_IF_TOO_BIG:
                    scale = min((Wd - 1) / pw, (Hd - 1) / ph, 0.5)
                    if scale <= 0:
                        tries += 1
                        continue
                    pw2 = max(1, int(pw * scale))
                    ph2 = max(1, int(ph * scale))
                    aug_res = cv2.resize(aug_patch, (pw2, ph2))
                else:
                    aug_res = aug_patch.copy()

                ph2, pw2 = aug_res.shape[:2]

                if pw2 >= Wd or ph2 >= Hd:
                    tries += 1
                    continue

                x_new = random.randint(0, Wd - pw2)
                y_new = random.randint(0, Hd - ph2)

                dst_file = df
                break

            if dst_file is None:
                print("Aucune destination trouvée pour un patch.")
                continue

            # Collage final image
            dst_img = cv2.imread(os.path.join(IMG_DIR, dst_file))
            dst_img[y_new:y_new + ph2, x_new:x_new + pw2] = aug_res

            # Sauvegarde image
            new_name = f"{Path(dst_file).stem}_aug_cls{cls}_from_{Path(src_file).stem}_i{aug_i}.jpg"
            cv2.imwrite(os.path.join(OUT_IMG_DIR, new_name), dst_img)

            # Sauvegarde labels
            old_boxes = label_map[dst_file]
            out_boxes = old_boxes.copy()

            xc_norm, yc_norm, w_norm, h_norm = pixels_to_norm_yolo(x_new, y_new, pw2, ph2, Wd, Hd)
            out_boxes.append((cls, xc_norm, yc_norm, w_norm, h_norm))

            write_labels(os.path.join(OUT_LBL_DIR, Path(new_name).with_suffix(".txt")), out_boxes)

print("Augmentation terminée.")