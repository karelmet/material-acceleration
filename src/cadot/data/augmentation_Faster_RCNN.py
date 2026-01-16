import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
import shutil
from collections import defaultdict

# On génère 6 versions synthétiques pour chaque objet trouvé.
AUG_PER_OBJECT = 6

# Liste des ID de classes sous-représentées
MINORITY_CLASSES = [1, 4, 5, 8, 9, 12, 13, 14]

# Si l'objet à coller est plus grand que l'image de destination, on le redimensionne.
# Essentiel pour garder une cohérence d'échelle
RESIZE_IF_TOO_BIG = True
RANDOM_SEED = 42
MIN_PATCH_DIM = 2 # On ignore les objets trop petits (bruit ou mauvaises annotations)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def ensure_unique_filename(filename, existing):
    """
    Garantit l'unicité des noms de fichiers.
    Indispensable car on va copier-coller des images et générer des variations.
    Si 'image.jpg' existe déjà, crée 'image_dup1.jpg'.
    """
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = filename
    i = 1
    while candidate in existing:
        candidate = f"{base}_dup{i}{ext}"
        i += 1
    existing.add(candidate)
    return candidate

def coco_bbox_to_xyxy(bbox):
    """Convertit le format COCO (x, y, w, h) en format de coordonnées absolues (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return int(x), int(y), int(x+w), int(y+h)

def augment_patch(patch):
    """
    Applique des transformations aléatoires sur un patch
    on force le modèle  à apprendre des caractéristiques plus robustes
    pour évitter l'overfitting
    """
    choice = random.choice([
        "noise", "rotate", "intensity",
        "flip", "blur", "cutout",
        "scale",
        "none"
    ])

    h, w = patch.shape[:2]

    # bruit gaussien
    if choice == "noise":
        noise = np.random.normal(0, 12, patch.shape).astype(np.int16)
        return np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # rotation
    if choice == "rotate":
        return cv2.rotate(
            patch,
            random.choice([
                cv2.ROTATE_90_CLOCKWISE,
                cv2.ROTATE_180,
                cv2.ROTATE_90_COUNTERCLOCKWISE
            ])
        )

    # changements d'éclairage
    # changements d'éclairage
    if choice == "intensity":
        out = patch.astype(np.float32)
        out = 128 + random.uniform(0.7, 1.3) * (out - 128)
        # On s'assure qu'on est positif avant le gamma
        out = np.clip(out, 0, 255) 
        gamma = random.uniform(0.7, 1.5)
        out = 255 * ((out / 255) ** gamma)
        return np.clip(out, 0, 255).astype(np.uint8)

    # Symétrie
    if choice == "flip":
        flip_code = random.choice([-1, 0, 1])
        return cv2.flip(patch, flip_code)

    # Simulation de flou
    if choice == "blur":
        k = random.choice([3, 5])
        return cv2.GaussianBlur(patch, (k, k), 0)

    # Cutout : On masque une partie de l'objet
    if choice == "cutout":
        out = patch.copy()
        cut_w = random.randint(3, w // 4)
        cut_h = random.randint(3, h // 4)
        x = random.randint(0, max(0, w - cut_w))
        y = random.randint(0, max(0, h - cut_h))
        out[y:y+cut_h, x:x+cut_w] = random.randint(0, 255)
        return out

    # Changement d'échelle (Zoom in/out) pour simuler la distance
    if choice == "scale":
        scale = random.uniform(1.3, 1.7)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if scale > 1.0: # Crop center
            x0 = (new_w - w) // 2
            y0 = (new_h - h) // 2
            return resized[y0:y0+h, x0:x0+w]
        else: # Pad with zeros
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            out = np.zeros_like(patch)
            out[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
            return out

    return patch


# PROCESSUS DE FUSION ET D'AUGMENTATION

def merge_and_augment_direct(orig_json, orig_img_dir, merged_json, merged_img_dir, max_per_class=1000):
    """
    Fonction principale qui :
    1. Copie les données originales.
    2. Génère des données synthétiques par "Copy-Paste" des classes minoritaires.
    3. Équilibre le dataset final en limitant le nombre d'instances par classe.
    """
    os.makedirs(merged_img_dir, exist_ok=True)

    with open(orig_json, "r") as f:
        coco = json.load(f)

    # Indexation des images pour accès rapide
    orig_images = {img["id"]: img for img in coco["images"]}
    ann_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        ann_by_img[ann["image_id"]].append(ann)

    # Structure du nouveau dataset fusionné
    merged = {
        "images": [],
        "annotations": [],
        "categories": coco["categories"]
    }

    # Compteurs pour générer de nouveaux IDs uniques
    next_img = 1
    next_ann = 1

    class_to_ann = defaultdict(list)
    used_filenames = set()

    # ÉTAPE 1 : COPIE DES DONNÉES ORIGINALES
    # On commence par inclure toutes les données réelles dont on dispose.
    old_to_new = {}

    for img_id, img in orig_images.items():
        src = os.path.join(orig_img_dir, img["file_name"])
        if not os.path.exists(src):
            print(f"[WARN] Missing {src}")
            continue

        new_fname = ensure_unique_filename(img["file_name"], used_filenames)
        shutil.copy(src, os.path.join(merged_img_dir, new_fname))

        merged["images"].append({
            "id": next_img,
            "file_name": new_fname,
            "width": img["width"],
            "height": img["height"]
        })

        old_to_new[img_id] = next_img
        next_img += 1

    for ann in coco["annotations"]:
        if ann["image_id"] not in old_to_new:
            continue
        new_ann = {
            "id": next_ann,
            "image_id": old_to_new[ann["image_id"]],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
            "area": ann.get("area", ann["bbox"][2]*ann["bbox"][3]),
            "iscrowd": int(ann.get("iscrowd",0)),
            "is_aug": False # Marqueur pour distinguer données réelles et synthétiques
        }
        merged["annotations"].append(new_ann)
        class_to_ann[new_ann["category_id"]].append(new_ann)
        next_ann += 1

    # ÉTAPE 2 : GÉNÉRATION DE DONNÉES SYNTHÉTIQUES

    for img_id, img_info in orig_images.items():
        # Lecture de l'image source
        src_path = os.path.join(orig_img_dir, img_info["file_name"])
        src = cv2.imread(src_path)
        if src is None: 
            continue
        H, W = src.shape[:2]

        anns = ann_by_img.get(img_id, [])
        for ann in anns:
            # On ne traite que les classes minoritaires définies en config
            if ann["category_id"] not in MINORITY_CLASSES:
                continue

            # Extraction de l'objet
            x1, y1, x2, y2 = coco_bbox_to_xyxy(ann["bbox"])
            patch = src[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # Vérification taille minimale
            ph, pw = patch.shape[:2]
            if min(ph,pw) < MIN_PATCH_DIM:
                scale = int(np.ceil(MIN_PATCH_DIM/min(ph,pw)))
                patch = cv2.resize(patch, (pw*scale, ph*scale))

            # Génération de N variations de cet objet (Augmentation)
            for _ in range(AUG_PER_OBJECT):
                aug_patch = augment_patch(patch)

                # Choix aléatoire d'une image de destination
                # Cela permet de découpler l'objet de son contexte habituel
                dst_img_id = random.choice(list(old_to_new.values()))
                dst_img_info = next(im for im in merged["images"] if im["id"] == dst_img_id)
                dst_path = os.path.join(merged_img_dir, dst_img_info["file_name"])

                dst = cv2.imread(dst_path)
                if dst is None:
                    continue

                Hd, Wd = dst.shape[:2]
                ph, pw = aug_patch.shape[:2]

                # Ajustement de taille si l'objet est plus gros que le fond
                if (pw > Wd or ph > Hd) and RESIZE_IF_TOO_BIG:
                    s = min((Wd*0.5)/pw, (Hd*0.5)/ph)
                    aug_patch = cv2.resize(aug_patch, (int(pw*s), int(ph*s)))
                    ph, pw = aug_patch.shape[:2]

                if Wd - pw <= 0 or Hd - ph <= 0:
                    continue

                # Positionnement aléatoire sur le fond
                x_new = random.randint(0, Wd - pw)
                y_new = random.randint(0, Hd - ph)

                # Création de la nouvelle image synthétique par collage
                new_img = dst.copy()
                new_img[y_new:y_new+ph, x_new:x_new+pw] = aug_patch

                # Sauvegarde physique de la nouvelle image
                new_fname = ensure_unique_filename(
                    f"{Path(dst_img_info['file_name']).stem}__aug.jpg",
                    used_filenames
                )
                cv2.imwrite(os.path.join(merged_img_dir, new_fname), new_img)

                # Ajout de l'image et de l'annotation aux métadonnées
                merged["images"].append({
                    "id": next_img,
                    "file_name": new_fname,
                    "width": Wd,
                    "height": Hd
                })

                new_ann = {
                    "id": next_ann,
                    "image_id": next_img,
                    "category_id": ann["category_id"],
                    "bbox": [x_new, y_new, pw, ph],
                    "area": pw*ph,
                    "iscrowd": 0,
                    "is_aug": True # Marqué comme synthétique
                }

                merged["annotations"].append(new_ann)
                class_to_ann[ann["category_id"]].append(new_ann)

                next_img += 1
                next_ann += 1


    # ÉTAPE 3 : SOUS-ÉCHANTILLONNAGE
    # Pour éviter d'avoir un dataset déséquilibré,
    # on limite chaque classe à un nombre maximum d'exemples (max_per_class).

    # ÉTAPE 3 : SOUS-ÉCHANTILLONNAGE
    final_anns = []
    
    # CORRECTION ICI : on déballe (class_id, anns)
    for class_id, anns_list in class_to_ann.items():
        
        # Si la classe est petite, on garde tout
        if len(anns_list) <= max_per_class:
            final_anns.extend(anns_list)
            continue
        
        # Sinon, on priorise les données réelles
        orig = [a for a in anns_list if not a["is_aug"]]
        aug = [a for a in anns_list if a["is_aug"]]

        keep = orig[:]
        need = max_per_class - len(keep)

        # Si besoin de compléter, on prend les synthétiques
        if need > 0 and len(aug) > 0:
            aug_sorted = sorted(aug, key=lambda a: a["area"], reverse=True)
            keep += aug_sorted[:need]

        final_anns.extend(keep[:max_per_class])

    merged["annotations"] = final_anns

    # Nettoyage : On supprime les références d'images qui n'ont plus d'annotations
    used = {a["image_id"] for a in merged["annotations"]}
    merged["images"] = [img for img in merged["images"] if img["id"] in used]

    with open(merged_json, "w") as f:
        json.dump(merged, f, indent=2)

    print("Augmentation et regroupement des nouvelles données terminé")
    print("Final annotations:", len(merged["annotations"]))
    print("Final images:", len(merged["images"]))
    return merged_json, merged_img_dir


# Main script
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[3]

    ORIG_IMG_DIR = str(ROOT/"CADOT_Dataset"/"train")
    ORIG_JSON     = str(ROOT/"CADOT_Dataset"/"train"/"_annotations.coco.json")

    MERGED_DIR = str(ROOT/"merged_dataset")
    MERGED_IMG = os.path.join(MERGED_DIR, "images")
    MERGED_JSON = os.path.join(MERGED_DIR, "dataset_merged.json")

    os.makedirs(MERGED_IMG, exist_ok=True)

    merge_and_augment_direct(
        ORIG_JSON,
        ORIG_IMG_DIR,
        MERGED_JSON,
        MERGED_IMG,
        max_per_class=3000 # Cible : 3000 annotations max par classe
    )

    # ÉTAPE 4 : CRÉATION DES JEUX TRAIN / VALID (SPLIT)
    # Une fois le dataset gonflé et équilibré, on le sépare pour l'entraînement.
    # Répartition classique : 80% Train, 20% Validation.

    with open(MERGED_JSON, "r") as f:
        merged = json.load(f)

    images = merged["images"]
    annotations = merged["annotations"]

    # Mélange aléatoire pour garantir une distribution homogène
    random.shuffle(images)

    # Calcul du point de coupure
    n_train = int(0.8 * len(images))
    train_imgs = images[:n_train]
    valid_imgs = images[n_train:]

    # Création des ensembles d'IDs pour filtrage rapide
    train_img_ids = {img["id"] for img in train_imgs}
    valid_img_ids = {img["id"] for img in valid_imgs}

    # Répartition des annotations selon l'appartenance de leur image
    train_anns = [a for a in annotations if a["image_id"] in train_img_ids]
    valid_anns = [a for a in annotations if a["image_id"] in valid_img_ids]

    # Création des dossiers physiques
    TRAIN_DIR = Path(MERGED_DIR) / "train"
    VALID_DIR = Path(MERGED_DIR) / "valid"
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALID_DIR.mkdir(parents=True, exist_ok=True)

    # Déplacement physique des fichiers images vers les dossiers respectifs
    for img in train_imgs:
        shutil.copy(
            os.path.join(MERGED_IMG, img["file_name"]),
            TRAIN_DIR / img["file_name"]
        )

    for img in valid_imgs:
        shutil.copy(
            os.path.join(MERGED_IMG, img["file_name"]),
            VALID_DIR / img["file_name"]
        )

    # Création des fichiers JSON COCO pour chaque sous-ensemble
    train_json = {
        "images": train_imgs,
        "annotations": train_anns,
        "categories": merged["categories"]
    }

    valid_json = {
        "images": valid_imgs,
        "annotations": valid_anns,
        "categories": merged["categories"]
    }

    with open(TRAIN_DIR / "_annotations.coco.json", "w") as f:
        json.dump(train_json, f, indent=2)

    with open(VALID_DIR / "_annotations.coco.json", "w") as f:
        json.dump(valid_json, f, indent=2)

    print("Séparation train/test terminée")
    print(f"Train images: {len(train_imgs)} | Annotations: {len(train_anns)}")
    print(f"Valid images: {len(valid_imgs)} | Annotations: {len(valid_anns)}")