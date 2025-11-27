from pathlib import Path
import sys, os
sys.path.append(os.path.abspath("../src"))

from sklearn.model_selection import train_test_split
import shutil
import yaml
import warnings

from cadot.utils.path import get_data_path, add_src_to_sys_path
add_src_to_sys_path()
from cadot.data.loading import get_image_label_pairs
from cadot.models.yolo import train_yolo

class_names = ['Basketball Field', 'Building', 'Crosswalk', 'Football Field',
           'Graveyard', 'Large Vehicle', 'Medium Vehicle', 'Playground',
           'Roundabout', 'Ship', 'Small Vehicle', 'Swimming Pool',
           'Tennis Court', 'Train'
           ]

data_root = get_data_path()

# get image/label pairs
pairs = list(get_image_label_pairs("train"))
if len(pairs) == 0:
    raise RuntimeError(f"No image/label pairs found in {data_root}")

# split into train / val
train_pairs, val_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

# prepare destination dataset structure
out_dir = Path("yolo_dataset_split")
for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
    (out_dir / sub).mkdir(parents=True, exist_ok=True)

# copy files to the new structure
for img_path, lbl_path in train_pairs:
    shutil.copy2(img_path, out_dir / "train" / "images" / Path(img_path).name)
    shutil.copy2(lbl_path, out_dir / "train" / "labels" / Path(lbl_path).name)

for img_path, lbl_path in val_pairs:
    shutil.copy2(img_path, out_dir / "val" / "images" / Path(img_path).name)
    shutil.copy2(lbl_path, out_dir / "val" / "labels" / Path(lbl_path).name)

nc = len(class_names) # number of classes

# get all the class ids in the labels
class_ids = []
for _, lbl_path in pairs:
    for line in open(lbl_path):
        parts = line.split()
        if parts:
            class_ids.append(int(parts[0]))

# check -> not necessary
if not all(0 <= cid < nc for cid in class_ids):
    raise ValueError("Some labels use a different class id (not in [0, 13]).")

print(f"Classes found in the labels: {sorted(set(class_ids))}")

# write the data.yaml file for YOLO
data_yaml = out_dir / "data.yaml"
data_cfg = {
    "train": str((out_dir / "train" / "images").resolve()),
    "val":   str((out_dir / "val"   / "images").resolve()),
    "nc": nc,
    "names": class_names,
}

with open(data_yaml, "w") as f:
    yaml.dump(data_cfg, f, sort_keys=False)

print(f"Fichier de config YOLO écrit dans : {data_yaml}")

train_yolo(data=str(data_yaml), 
           model="yolov8n.pt", 
           epochs=10, 
           imgsz=512, 
           batch=8, 
           device="0", # -> "mps" pour le mac. Mettre "0" si machine avec GPU NVIDIA
           val=False
           ) 