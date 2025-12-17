import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


classes = ['Background', 'Basketball Field', 'Building', 'Crosswalk', 'Football Field',
           'Graveyard', 'Large Vehicle', 'Medium Vehicle', 'Playground',
           'Roundabout', 'Ship', 'Small Vehicle', 'Swimming Pool',
           'Tennis Court', 'Train'
           ]



# WRAPPER COCO POUR MANIPULER IMAGES ET TARGETS
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
    

def predictions_to_coco(predictions, image_ids):
    coco_results = []

    for pred, img_id in zip(predictions, image_ids):
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            coco_results.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1])
                ],
                "score": float(score)
            })

    return coco_results


# CHEMINS ET DATALOADERS
ROOT = Path(__file__).resolve().parents[3]
save_train = ROOT / "src" / "cadot" / "models"

val_dir = ROOT / "merged_dataset" / "valid"
val_json = val_dir / "_annotations.coco.json"
valid_dataset = CocoWrapper(str(val_dir), str(val_json))
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=lambda x: tuple(zip(*x))
)


# CHARGEMENT DU MODELE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ne pas recharger les poids par défaut !
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=15)
model.to(device)
# Charger le .pth
state_dict = torch.load(save_train / "faster_rcnn_best.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()


# ÉVALUATION SUR LE JEU DE VALIDATION
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


torch.save(
    {"predictions": predictions, "targets": targets},
    save_train / "faster_rcnn_predictions.pt"
)

print("✔ Sauvegardé dans :", save_train / "faster_rcnn_predictions.pt")



# MATRICE DE CONFUSION
def compute_confusion_matrix(predictions, targets, iou_threshold=0.5, score_threshold=0.05):
    y_true = []
    y_pred = []

    for pred, tgt in zip(predictions, targets):
        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"].tolist()

        # filtrer les predictions faibles
        mask = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]

        matched_gt = set()

        # Matching préd -> GT (one-to-one)
        for pb, pl in zip(pred_boxes, pred_labels):
            ious = torchvision.ops.box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = torch.max(ious, dim=0)

            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                # vraie prédiction
                y_true.append(gt_labels[max_idx])
                y_pred.append(pl.item())
                matched_gt.add(max_idx.item())

        # Tous les GT non matchés = FN (prédit background)
        for i, gl in enumerate(gt_labels):
            if i not in matched_gt:
                y_true.append(gl)
                y_pred.append(0)   # background

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    return cm

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(classes)),
           yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           title='Faster R-CNN Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(save_train / "faster_rcnn_confusion_matrix.png")


def compute_precision_from_cm(cm, background_idx=0):
    num_classes = cm.shape[0]

    TP = 0
    FP = 0

    for c in range(num_classes):
        if c == background_idx:
            continue

        tp_c = cm[c, c]
        fp_c = cm[:, c].sum() - tp_c

        TP += tp_c
        FP += fp_c

    precision_micro = TP / (TP + FP + 1e-8)
    return precision_micro

def precision_per_class(cm, classes, background_idx=0):
    precisions = {}

    for c, name in enumerate(classes):
        if c == background_idx:
            continue

        tp = cm[c, c]
        fp = cm[:, c].sum() - tp

        if tp + fp > 0:
            precisions[name] = tp / (tp + fp)
        else:
            precisions[name] = 0.0

    return precisions



cm = compute_confusion_matrix(
    torch.load(save_train / "faster_rcnn_predictions.pt")["predictions"],
    torch.load(save_train / "faster_rcnn_predictions.pt")["targets"],
    iou_threshold=0.5,
)

plot_confusion_matrix(
    cm,
    classes
)

precision = compute_precision_from_cm(cm)
print(f"Precision (micro, sans background) : {precision:.4f}")

precisions = precision_per_class(cm, classes)

for cls, p in precisions.items():
    print(f"{cls:20s} : {p:.3f}")



# PRÉCISION-RAPPEL ET AP PAR CLASSE
# Charger annotations GT
coco_gt = COCO(str(val_json))

# Patch si le champ "info" est absent (bug classique pycocotools)
if "info" not in coco_gt.dataset:
    coco_gt.dataset["info"] = {
        "description": "custom dataset",
        "version": "1.0",
        "year": 2025
    }

# image_ids dans le même ordre que valid_loader
image_ids = valid_dataset.ids

# Convertir prédictions
coco_preds = predictions_to_coco(predictions, image_ids)

# Charger résultats
coco_dt = coco_gt.loadRes(coco_preds)

# Évaluation COCO
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

print("\n📊 AP par classe :")

for idx, cls_name in enumerate(classes):
    if idx == 0:
        continue  # background

    coco_eval.params.catIds = [idx]
    coco_eval.evaluate()
    coco_eval.accumulate()
    ap = coco_eval.stats[0]  # AP@[.5:.95]

    print(f"{cls_name:15s} : AP = {ap:.3f}")

