from typing import Sequence
import torch
import yaml
import shutil
from cadot.data.loading import get_image_label_pairs 
from sklearn.model_selection import train_test_split

# Import from your local modified ultralytics
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent  # Go up to project root
sys.path.insert(0, str(project_root / "ultralytics"))

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.models.yolo.detect import DetectionTrainer


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

rare_class_indices = [0, 3, 4, 7, 8, 11, 12, 13]
common_class_indices = [1, 10]

# Create class weights
class_weights = torch.ones(14)  # assuming 14 classes
class_weights[rare_class_indices] = 200.0  # Double weight for rare classes
class_weights[common_class_indices] = 0.2  # Half weight for common classes

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

class WeightedDetectionTrainer(DetectionTrainer):
    """Custom trainer with class weight support."""
    
    def __init__(self, cfg={}, overrides=None, _callbacks=None):
        """Initialize with class weights."""
        super().__init__(cfg, overrides, _callbacks)
        self.class_weights = None  # Will be set later
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and set up custom loss."""
        model = super().get_model(cfg, weights, verbose)
        return model
    
    def set_model_attributes(self):
        """Set model attributes including custom loss with class weights."""
        super().set_model_attributes()
        
        # Override the loss function with class weights
        if hasattr(self, 'class_weights') and self.class_weights is not None:
            self.loss = v8DetectionLoss(
                self.model, 
                tal_topk=10, 
                class_weights=self.class_weights
            )
        
    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return super().get_validator()

def train_yolo_balanced(data: str,
                        model: str = "yolov8m.pt",
                        epochs: int = 80,
                        imgsz: int = 640,
                        batch: int = 16,
                        device: str | None = None,
                        project: str = "runs/train",
                        name: str | None = None,
                        resume: bool = False,
                        hyp: dict | None = None,
                        val: bool = True,
                        classes: Sequence[int] | None = None,
                        class_weights: torch.Tensor = class_weights,
                        ):
    """Train a YOLOv8 model with optional class balancing.
    
    Returns:
        dict: Dictionary containing training results with keys:
            - 'results': Training results/metrics
            - 'model_path': Path to the best saved model
            - 'trainer': The trainer object (only when using class weights)
    """
    
    # Prepare overrides dict
    overrides = {
        "model": model,  # Add model to overrides
        "data": data,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "project": project,
        "patience": 20,
        "val": val,
        "task": "detect",  # Specify task -> useful?
    }
    
    if device is not None:
        overrides["device"] = device
    if name is not None:
        overrides["name"] = name
    if resume:
        overrides["resume"] = resume
    if hyp is not None:
        overrides["hyp"] = hyp
    if classes is not None:
        overrides["classes"] = classes

    # Use custom trainer if class weights are provided
    if class_weights is not None:
        print(f"Using class weights: {class_weights}")
        
        from ultralytics import YOLO
        yolo = YOLO(model)
        
        # Monkey-patch before training
        def init_weighted_loss(trainer):
            trainer.loss = v8DetectionLoss(
                trainer.model,
                tal_topk=10,
                class_weights=class_weights.to(trainer.device)
            )
        
        yolo.add_callback("on_train_start", init_weighted_loss)
        
        train_kwargs = {k: v for k, v in overrides.items() if k not in ['model', 'task', 'classes']}
        result = yolo.train(**train_kwargs)
        
        print(f"Training finished.")
        
        return {
            'results': result,
            'best': result,
            'model_path': yolo.trainer.best if hasattr(yolo, 'trainer') else None,
            'model': yolo
        }
    else:
        # Standard training without class weights
        from ultralytics import YOLO
        yolo = YOLO(model)
        print(f"Starting YOLOv8 training: model={model}, data={data}, epochs={epochs}")
        result = yolo.train(**overrides)
        
        print(f"Training finished.")
        
        return {
            'results': result,
            'best': result,
            'model_path': yolo.trainer.best if hasattr(yolo, 'trainer') else None,
            'model': yolo  # Return YOLO object for further use
        }