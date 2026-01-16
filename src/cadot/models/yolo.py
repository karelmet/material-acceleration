from pathlib import Path
from typing import Sequence
from ultralytics import YOLO
import shutil
import yaml
from sklearn.model_selection import train_test_split

from cadot.data.loading import get_image_label_pairs 

CADOT_CLASS_NAMES: list[str] = [
    "Basketball Field",
    "Building",
    "Crosswalk",
    "Football Field",
    "Graveyard",
    "Large Vehicle",
    "Medium Vehicle",
    "Playground",
    "Roundabout",
    "Ship",
    "Small Vehicle",
    "Swimming Pool",
    "Tennis Court",
    "Train",
]

def prepare_yolo_dataset(
    out_dir: Path,
    class_names: Sequence[str] = CADOT_CLASS_NAMES,
    split: str = "train",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """
    Prepare the dataset in YOLO format in `out_dir` from the CADOT split given.

    Return:
        Path to data.yaml created.
    """
    pairs = list(get_image_label_pairs(split))
    if len(pairs) == 0:
        raise RuntimeError(f"No image/label pairs found for split '{split}'")

    # split train / val
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=test_size, random_state=random_state
    )

    # create file structure
    for sub in ["train/images", "train/labels", "val/images", "val/labels"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # copy files
    for img_path, lbl_path in train_pairs:
        shutil.copy2(img_path, out_dir / "train" / "images" / Path(img_path).name)
        shutil.copy2(lbl_path, out_dir / "train" / "labels" / Path(lbl_path).name)

    for img_path, lbl_path in val_pairs:
        shutil.copy2(img_path, out_dir / "val" / "images" / Path(img_path).name)
        shutil.copy2(lbl_path, out_dir / "val" / "labels" / Path(lbl_path).name)

    # verify the classes
    nc = len(class_names)
    class_ids: list[int] = []
    for _, lbl_path in pairs:
        with open(lbl_path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    class_ids.append(int(parts[0]))

    if not all(0 <= cid < nc for cid in class_ids):
        raise ValueError("Some labels use a class id outside [0, nc).")

    print(f"Classes found in labels: {sorted(set(class_ids))} (nc={nc})")

    # write data.yaml
    data_yaml = out_dir / "data.yaml"
    data_cfg = {
        "train": str((out_dir / "train" / "images").resolve()),
        "val":   str((out_dir / "val"   / "images").resolve()),
        "nc": nc,
        "names": list(class_names),
    }

    with open(data_yaml, "w") as f:
        yaml.dump(data_cfg, f, sort_keys=False)

    print(f"Wrote YOLO data config to: {data_yaml}")
    return data_yaml



def train_yolo(data: str,
               model: str = "yolov8n.pt",
               epochs: int = 30,
               imgsz: int = 640,
               batch: int = 16,
               device: str | None = None,
               project: str = "runs/train",
               name: str | None = None,
               resume: bool = False,
               hyp: dict | None = None,
               val: bool = True,
               classes: Sequence[int] | None = None,
               ):
    """
    Train a YOLOv8 model using the ultralytics API.

    Args:
        data: Path to data.yaml (str) or dataset config understood by ultralytics.
        model: Pretrained model file or model spec (e.g. "yolov8n.pt").
        epochs: Number of training epochs.
        imgsz: Image size.
        batch: Batch size.
        device: Device to use, e.g. "0" or "cpu". If None, ultralytics will auto-select.
        project: Directory to save results (default "runs/train").
        name: Run name. If None a name will be generated.
        resume: Whether to resume from last run.
        hyp: Hyperparameters dict or path to hyp.yaml.

    Returns:
        The object returned by ultralytics.YOLO.train (training results).
    """
    yolo = YOLO(model)

    train_kwargs = {
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": project,
        "name": name,
        "resume": resume,
        "hyp": hyp,
        "val": val,
        "classes": classes,
        "patience": 20,
    }
    # remove None values so ultralytics uses defaults where applicable
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    print(f"Starting YOLOv8 training: model={model}, data={data}, epochs={epochs}, imgsz={imgsz}, batch={batch}")
    result = yolo.train(**train_kwargs)

    print(f"Training finished.")
    return result