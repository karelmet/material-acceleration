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

MINORITY_CLASSES = [0]    # classes à augmenter (adapter)
AUG_PER_OBJECT = 3        # nb d'augmentations par objet trouvé
MAX_TRIES_FOR_DST = 50    # essais pour trouver une destination convenable
RESIZE_IF_TOO_BIG = True  # si le patch > dest image, on le redimensionne
RANDOM_SEED = 42


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def list_image_files(dirpath):
    exts = (".jpg", ".jpeg", ".png")
    return [f for f in os.listdir(dirpath) if f.lower().endswith(exts)]

def load_labels(label_path):
    """Retourne une liste de tuples (cls, x_min_norm, y_min_norm, w_norm, h_norm)"""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            x = float(parts[1]); y = float(parts[2]); w = float(parts[3]); h = float(parts[4])
            boxes.append((cls, x, y, w, h))
    return boxes

def write_labels(label_path, boxes):
    """Ecrit une liste de (cls, x_min_norm, y_min_norm, w_norm, h_norm)"""
    with open(label_path, "w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

def augment_patch(patch):
    """Applique aléatoirement une transformation simple ou d'intensité"""
    choice = random.choice(["noise", "rotate", "intensity"])
    if choice == "noise":
        noise = np.random.normal(0, 12, patch.shape).astype(np.int16)
        out = patch.astype(np.int16) + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    if choice == "rotate":
        angle = random.choice([90, 180, 270])
        if angle == 90:
            return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(patch, cv2.ROTATE_180)
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if choice == "intensity":
        out = patch.astype(np.float32)

        contrast = random.uniform(0.7, 1.3)
        out = 128 + contrast * (out - 128)
    
        gamma = random.uniform(0.7, 1.5)
        out = 255.0 * ((out / 255.0) ** gamma)

        return np.clip(out, 0, 255).astype(np.uint8)
    
    return patch


# Préparer listes
image_files = list_image_files(IMG_DIR)
if not image_files:
    raise SystemExit("Aucune image trouvée dans " + IMG_DIR)

# Pré-indexer les labels pour éviter de coller sur images contenant déjà la classe cible
image_labels_map = {}
for imgf in image_files:
    lbl_path = os.path.join(LBL_DIR, Path(imgf).with_suffix(".txt").name)
    boxes = load_labels(lbl_path)
    image_labels_map[imgf] = boxes

# Fonction utilitaire pour convertir normalisé -> pixels (x_min_norm, y_min_norm, w_norm, h_norm)
def norm_to_pixels(box, W, H):
    cls, x_norm, y_norm, w_norm, h_norm = box
    x_px = int(round(x_norm * W))
    y_px = int(round(y_norm * H))
    w_px = int(round(w_norm * W))
    h_px = int(round(h_norm * H))
    # clamp
    x_px = max(0, min(x_px, W-1))
    y_px = max(0, min(y_px, H-1))
    w_px = max(1, min(w_px, W - x_px))
    h_px = max(1, min(h_px, H - y_px))
    return cls, x_px, y_px, w_px, h_px

def pixels_to_norm(x_px, y_px, w_px, h_px, W, H):
    return x_px / W, y_px / H, w_px / W, h_px / H

# Main augment loop
for src_img_file in image_files:
    src_img_path = os.path.join(IMG_DIR, src_img_file)
    src_lbl_path = os.path.join(LBL_DIR, Path(src_img_file).with_suffix(".txt").name)
    src_img = cv2.imread(src_img_path)
    if src_img is None:
        print("Impossible de lire:", src_img_path); continue
    Hs, Ws = src_img.shape[:2]
    src_boxes = load_labels(src_lbl_path)

    for (cls, x_norm, y_norm, w_norm, h_norm) in src_boxes:
        if cls not in MINORITY_CLASSES:
            continue

        # convertir en pixels (COCO normalisé -> x_min, y_min, w, h)
        cls_px, x_px, y_px, w_px, h_px = norm_to_pixels((cls, x_norm, y_norm, w_norm, h_norm), Ws, Hs)

        # extraire patch (crop)
        patch = src_img[y_px:y_px+h_px, x_px:x_px+w_px].copy()
        if patch.size == 0:
            continue

        for aug_i in range(AUG_PER_OBJECT):
            aug_patch = augment_patch(patch)

            # chercher une image destination convenable (préférer celles sans déjà la classe)
            candidates = [f for f in image_files if f != src_img_file]
            # prioriser images qui n'ont pas la classe
            no_class_candidates = [f for f in candidates if all(b[0] != cls for b in image_labels_map.get(f, []))]
            search_pool = no_class_candidates if no_class_candidates else candidates

            dst_file = None
            tries = 0
            while tries < MAX_TRIES_FOR_DST and search_pool:
                dst_file_candidate = random.choice(search_pool)
                dst_img_path = os.path.join(IMG_DIR, dst_file_candidate)
                dst_img = cv2.imread(dst_img_path)
                if dst_img is None:
                    tries += 1
                    continue
                Hd, Wd = dst_img.shape[:2]

                # si patch plus grand que destination, redimensionner si activé
                ph, pw = aug_patch.shape[:2]
                scale = 1.0
                if (pw > Wd or ph > Hd) and RESIZE_IF_TOO_BIG:
                    scale_w = Wd / pw * 0.5  # on laisse place ; on peut ajuster
                    scale_h = Hd / ph * 0.5
                    scale = min(scale_w, scale_h, 1.0)
                    new_pw = max(1, int(pw * scale))
                    new_ph = max(1, int(ph * scale))
                    aug_patch_resized = cv2.resize(aug_patch, (new_pw, new_ph), interpolation=cv2.INTER_AREA)
                else:
                    aug_patch_resized = aug_patch

                ph2, pw2 = aug_patch_resized.shape[:2]
                if pw2 >= Wd or ph2 >= Hd:
                    # patch encore trop grand → essaie autre image
                    tries += 1
                    continue

                # position aléatoire d'insertion
                new_x1 = random.randint(0, Wd - pw2)
                new_y1 = random.randint(0, Hd - ph2)

                # ok cette destination convient
                dst_file = dst_file_candidate
                break

            if dst_file is None:
                # pas de destination trouvée après essais -> sauter
                continue

            # charger dst_img à nouveau (déjà chargé mais on recharge proprement)
            dst_img_path = os.path.join(IMG_DIR, dst_file)
            dst_img = cv2.imread(dst_img_path)
            Hd, Wd = dst_img.shape[:2]
            # coller
            dst_img[new_y1:new_y1+ph2, new_x1:new_x1+pw2] = aug_patch_resized

            # créer nom et sauver
            save_name = Path(dst_file).stem + f"_aug_cls{cls}_src{Path(src_img_file).stem}_i{aug_i}.jpg"
            save_img_path = os.path.join(OUT_IMG_DIR, save_name)
            cv2.imwrite(save_img_path, dst_img)

            # labels : copier anciens labels de la dst image puis ajouter le nouveau (COCO normalisé)
            dst_lbl_orig = image_labels_map.get(dst_file, [])
            # convertir les dst original boxes (déjà normalisés) -> on les garde tels quels
            out_boxes = [(b[0], b[1], b[2], b[3], b[4]) for b in dst_lbl_orig]

            new_x_norm, new_y_norm, new_w_norm, new_h_norm = pixels_to_norm(new_x1, new_y1, pw2, ph2, Wd, Hd)
            out_boxes.append((cls, new_x_norm, new_y_norm, new_w_norm, new_h_norm))

            save_lbl_path = os.path.join(OUT_LBL_DIR, Path(save_name).with_suffix(".txt").name)
            write_labels(save_lbl_path, out_boxes)

print("Augmentation terminée. Images et labels sauvegardés dans:", OUT_ROOT)