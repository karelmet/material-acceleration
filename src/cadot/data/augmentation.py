import os
import cv2
import json
import random
import numpy as np
from pathlib import Path

# LA COMMANDE POUR RUN CE .py EST : python augmentation.py --mode yolo (ou coco)

# ==========================================
#   CONFIGURATION
# ==========================================
AUG_PER_OBJECT = 3
MINORITY_CLASSES = [1]       # classes à augmenter
MAX_TRIES_FOR_DST = 50
RESIZE_IF_TOO_BIG = True
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SCRIPT_DIR = Path(__file__).resolve().parent


# ==========================================
#   PATCH AUGMENTATION
# ==========================================
def augment_patch(patch):
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
        out = 255 * ((out / 255.0) ** gamma)
        return np.clip(out, 0, 255).astype(np.uint8)

    return patch


# ==========================================
#   UTILS
# ==========================================
def coco_bbox_to_xyxy(bbox):
    x, y, w, h = bbox
    return int(x), int(y), int(x + w), int(y + h)


def yolo_to_pixels(x, y, w, h, W, H):
    x_px = int((x - w/2) * W)
    y_px = int((y - h/2) * H)
    w_px = int(w * W)
    h_px = int(h * H)
    return x_px, y_px, w_px, h_px


def pixels_to_yolo(x, y, w, h, W, H):
    cx = (x + w/2) / W
    cy = (y + h/2) / H
    return cx, cy, w / W, h / H


# ==========================================
#   MODE YOLO
# ==========================================
def augment_yolo(train_dir, out_dir):
    IMG_DIR = train_dir
    LBL_DIR = train_dir.replace("images", "labels")  # si jamais labels existent

    OUT_IMG = os.path.join(out_dir, "images")
    OUT_LBL = os.path.join(out_dir, "labels")
    os.makedirs(OUT_IMG, exist_ok=True)
    os.makedirs(OUT_LBL, exist_ok=True)

    imgs = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]

    # Charge labels YOLO
    label_map = {}
    for img in imgs:
        lbl_path = os.path.join(LBL_DIR, Path(img).with_suffix(".txt").name)
        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    p = line.split()
                    if len(p) == 5:
                        boxes.append((int(p[0]), *map(float, p[1:])))
        label_map[img] = boxes

    for img in imgs:
        img_path = os.path.join(IMG_DIR, img)
        src_img = cv2.imread(img_path)
        if src_img is None:
            continue
        H, W = src_img.shape[:2]

        for cls, x, y, w, h in label_map[img]:
            if cls not in MINORITY_CLASSES:
                continue

            x_px, y_px, w_px, h_px = yolo_to_pixels(x, y, w, h, W, H)
            patch = src_img[y_px:y_px+h_px, x_px:x_px+w_px]

            if patch.size == 0:
                continue

            for k in range(AUG_PER_OBJECT):

                aug_patch = augment_patch(patch)

                dst = random.choice(imgs)
                dst_img = cv2.imread(os.path.join(IMG_DIR, dst))
                Hd, Wd = dst_img.shape[:2]

                ph, pw = aug_patch.shape[:2]
                if (pw > Wd or ph > Hd) and RESIZE_IF_TOO_BIG:
                    scale = min(Wd / pw * 0.5, Hd / ph * 0.5)
                    aug_patch = cv2.resize(aug_patch, (int(pw*scale), int(ph*scale)))
                    ph, pw = aug_patch.shape[:2]

                new_x = random.randint(0, Wd - pw)
                new_y = random.randint(0, Hd - ph)

                img_aug = dst_img.copy()
                img_aug[new_y:new_y+ph, new_x:new_x+pw] = aug_patch

                save_name = f"{Path(dst).stem}_aug_cls{cls}_{k}.jpg"
                cv2.imwrite(os.path.join(OUT_IMG, save_name), img_aug)

                # labels
                x_y, y_y, w_y, h_y = pixels_to_yolo(new_x, new_y, pw, ph, Wd, Hd)
                with open(os.path.join(OUT_LBL, Path(save_name).with_suffix(".txt")), "w") as f:
                    f.write(f"{cls} {x_y:.6f} {y_y:.6f} {w_y:.6f} {h_y:.6f}\n")

    print("YOLO augmentation finished.")


# ==========================================
#   MODE COCO
# ==========================================
def augment_coco(train_dir, out_dir):
    ann_path = os.path.join(train_dir, "_annotations.coco.json")

    with open(ann_path, "r") as f:
        coco = json.load(f)

    OUT_IMG = os.path.join(out_dir, "images")
    os.makedirs(OUT_IMG, exist_ok=True)

    images = {img["id"]: img for img in coco["images"]}
    ann_by_img = {}
    for ann in coco["annotations"]:
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    next_img = max(images) + 1
    next_ann = max(a["id"] for a in coco["annotations"]) + 1

    new_imgs = []
    new_anns = []

    for img_id, img_info in images.items():
        src = cv2.imread(os.path.join(train_dir, img_info["file_name"]))
        if src is None:
            continue

        Hs, Ws = src.shape[:2]
        anns = ann_by_img.get(img_id, [])

        for ann in anns:
            if ann["category_id"] not in MINORITY_CLASSES:
                continue

            x1, y1, x2, y2 = coco_bbox_to_xyxy(ann["bbox"])
            patch = src[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            for k in range(AUG_PER_OBJECT):

                aug_patch = augment_patch(patch)

                dst_id = random.choice(list(images.keys()))
                dst_info = images[dst_id]
                dst_img = cv2.imread(os.path.join(train_dir, dst_info["file_name"]))
                Hd, Wd = dst_img.shape[:2]

                ph, pw = aug_patch.shape[:2]
                if (pw > Wd or ph > Hd) and RESIZE_IF_TOO_BIG:
                    scale = min(Wd / pw * 0.5, Hd / ph * 0.5)
                    aug_patch = cv2.resize(aug_patch, (int(pw*scale), int(ph*scale)))
                    ph, pw = aug_patch.shape[:2]

                new_x = random.randint(0, Wd - pw)
                new_y = random.randint(0, Hd - ph)

                img_aug = dst_img.copy()
                img_aug[new_y:new_y+ph, new_x:new_x+pw] = aug_patch

                save_name = f"{Path(dst_info['file_name']).stem}_aug_{k}.jpg"
                cv2.imwrite(os.path.join(OUT_IMG, save_name), img_aug)

                new_imgs.append({
                    "id": next_img,
                    "file_name": save_name,
                    "width": Wd,
                    "height": Hd
                })

                new_anns.append({
                    "id": next_ann,
                    "image_id": next_img,
                    "category_id": ann["category_id"],
                    "bbox": [new_x, new_y, pw, ph],
                    "area": pw * ph,
                    "iscrowd": 0
                })

                next_img += 1
                next_ann += 1

    coco_final = {
        "images": coco["images"] + new_imgs,
        "annotations": coco["annotations"] + new_anns,
        "categories": coco["categories"]
    }

    with open(os.path.join(out_dir, "annotations_aug.coco.json"), "w") as f:
        json.dump(coco_final, f, indent=2)

    print("COCO augmentation finished.")


# ==========================================
#   FONCTION UNIFIÉE
# ==========================================
def augment_dataset(mode="coco",
                   data_root="CADOT_Dataset",
                   split="train",
                   out_root="dataset_aug"):

    SCRIPT_DIR = Path(__file__).resolve().parent
    REPO_ROOT = SCRIPT_DIR.parent.parent.parent

    train_dir = REPO_ROOT / data_root / split
    out_dir   = REPO_ROOT / out_root / split

    train_dir = train_dir.resolve()
    out_dir = out_dir.resolve()

    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Train dir = {train_dir}")
    print(f"[INFO] Out dir   = {out_dir}")

    if mode.lower() == "yolo":
        augment_yolo(str(train_dir), str(out_dir))

    elif mode.lower() == "coco":
        augment_coco(str(train_dir), str(out_dir))

    else:
        raise ValueError("mode must be 'coco' or 'yolo'.")



# ==========================================
#   MAIN
# ==========================================
if __name__ == "__main__":
    augment_dataset(mode="coco")
