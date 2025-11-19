import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from pathlib import Path


# ==========================
#   WRAPPER DU DATASET COCO
# ==========================
class CocoWrapper(CocoDetection):
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        img = functional.to_tensor(img)
        
        boxes = []
        labels = []
        for t in targets:
            boxes.append(t["bbox"])       # COCO format xywh
            labels.append(t["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": torchvision.ops.box_convert(boxes, "xywh", "xyxy"),
            "labels": labels
        }

        return img, target


# ==============
#   PATHS
# ==============
ROOT = Path(__file__).resolve().parents[3]
save_train = ROOT / "src" / "cadot" / "models"

val_dir = ROOT / "CADOT_Dataset" / "valid"
val_json = val_dir / "_annotations.coco.json"


# ===================
#   DATA LOADER
# ===================
valid_dataset = CocoWrapper(str(val_dir), str(val_json))
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)


# ==========================
#   CHARGEMENT DU MODELE
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTANT → ne pas recharger les poids par défaut !
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=15)
model.to(device)

# Charger le .pth
state_dict = torch.load(save_train / "faster_rcnn.pth", map_location=device)
model.load_state_dict(state_dict)

model.eval()


# ==========================
#   TEST : COLLECT PRED/GT
# ==========================
predictions = []
targets = []

with torch.no_grad():
    for imgs, trgts in valid_loader:
        imgs = [img.to(device) for img in imgs]
        outputs = model(imgs)

        for out, tgt in zip(outputs, trgts):
            predictions.append({
                "boxes": out["boxes"].cpu(),
                "labels": out["labels"].cpu(),
                "scores": out["scores"].cpu()
            })
            targets.append({
                "boxes": tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu()
            })


# ==========================
#   SAUVEGARDE
# ==========================
torch.save(
    {"predictions": predictions, "targets": targets},
    save_train / "faster_rcnn_predictions.pt"
)

print("✔ Sauvegardé dans :", save_train / "faster_rcnn_predictions.pt")