import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection.rpn import AnchorGenerator


classes = ['Background', 'Basketball Field', 'Building', 'Crosswalk', 'Football Field',
           'Graveyard', 'Large Vehicle', 'Medium Vehicle', 'Playground',
           'Roundabout', 'Ship', 'Small Vehicle', 'Swimming Pool',
           'Tennis Court', 'Train']

BACKGROUND_IDX = 0      # Indice du background
SCORE_THRESH_CM = 0.3   # Seuil de confiance pour la matrice de confusion
SCORE_THRESH_PR = 0.05  # Seuil minimal pour les courbes Precision-Recall
IOU_THRESH = 0.5        # Seuil d'IoU pour considérer une détection comme valide (0.5 est un recouvrement cible/prédiction moitié)

# GESTION DES DONNÉES (DATASET)

class CocoWrapper(CocoDetection):
    """
    Wrapper autour de torchvision.datasets.CocoDetection.
    Le format natif de COCO retourne les bounding boxes en (x, y, w, h).
    Or, Faster R-CNN dans PyTorch attend impérativement le format (x1, y1, x2, y2).
    Cette classe effectue la conversion à la volée lors du chargement des données.
    """
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        img = functional.to_tensor(img) # Conversion PIL -> Tensor (0-1)
        
        boxes = []
        labels = []
        
        # Extraction des annotations
        for t in targets:
            boxes.append(t["bbox"])       # Format COCO original : (xmin, ymin, width, height)
            labels.append(t["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        # Conversion critique : (x, y, w, h) -> (x1, y1, x2, y2)
        target["boxes"] = torchvision.ops.box_convert(boxes, "xywh", "xyxy")
        target["labels"] = labels

        return img, target

# CONFIGURATION DES CHEMINS
ROOT = Path(__file__).resolve().parents[3]
save_dir = ROOT / "src" / "cadot" / "models"
save_dir.mkdir(exist_ok=True, parents=True)

# Redéfinition des chemins valid, train
val_dir = ROOT / "merged_dataset" / "valid"
val_json = val_dir / "_annotations.coco.json"

dataset = CocoWrapper(str(val_dir), str(val_json))
loader = DataLoader(dataset, batch_size=1, shuffle=False,
                    collate_fn=lambda x: tuple(zip(*x)))

# Calcul sur GPU (ou CPU sinon)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. On redéfinit les ancres comme à l'entraînement
anchor_generator = AnchorGenerator(
    sizes=((16,), (32,), (64,), (128,), (256,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

# Instancier modèle en lui passant ces ancres
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=None, 
    num_classes=15,
    rpn_anchor_generator=anchor_generator,
    min_size=512,
    max_size=1024
)

# Charger les poids
state_dict = torch.load(save_dir / "faster_rcnn_best.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


predictions = []
targets = []
inference_times = []

with torch.no_grad():
    for imgs, trgts in loader:
        imgs = [img.to(device) for img in imgs]

        start = time.time()
        outputs = model(imgs)
        end = time.time()

        # calcul du temps d'inférence
        inference_times.append(end - start)

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

torch.save({"predictions": predictions, "targets": targets}, save_dir / "predictions.pt")


def compute_confusion_matrix(predictions, targets, iou_threshold=0.5, score_threshold=0.3):
    # Calcule la matrice de confusion en associant les prédictions aux vérités terrain
    # basées sur le chevauchement (IoU) et le score de confiance calculé par le modèle
    y_true, y_pred = [], []

    for pred, tgt in zip(predictions, targets):
        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"].tolist()

        mask = pred["scores"] >= score_threshold
        pred_boxes = pred["boxes"][mask]
        pred_labels = pred["labels"][mask]

        matched_gt = set()

        for pb, pl in zip(pred_boxes, pred_labels):
            if len(gt_boxes) == 0:
                continue

            ious = torchvision.ops.box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = torch.max(ious, dim=0)

            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                y_true.append(gt_labels[max_idx])
                y_pred.append(pl.item())
                matched_gt.add(max_idx.item())

        for i, gl in enumerate(gt_labels):
            if i not in matched_gt:
                y_true.append(gl)
                y_pred.append(BACKGROUND_IDX)

    return confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))


cm = compute_confusion_matrix(predictions, targets, IOU_THRESH, SCORE_THRESH_CM)


def plot_confusion_matrix(cm, classes):
    # Génère une figure visualisant la matrice de confusion et sauvegarde l'image.
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png")
    plt.close()

plot_confusion_matrix(cm, classes)


# PRECISION / RECALL / score F1


def compute_prf(cm, classes, background_idx=0):
    # Calcule les métriques de Précision, Rappel et F1-score pour chaque classe
    # à partir des valeurs de la matrice de confusion.
    metrics = {}

    for c, name in enumerate(classes):
        if c == background_idx:
            continue

        tp = cm[c, c]               # vrais positifs
        fp = cm[:, c].sum() - tp    # faux positifs
        fn = cm[c, :].sum() - tp    # faux négatifs

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        metrics[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    return metrics


prf_metrics = compute_prf(cm, classes)


with open(save_dir / "prf_metrics.json", "w") as f:
    json.dump(prf_metrics, f, indent=4)

print("Matrice de confusion ok, scores ok")


# precision rappel + précision moyenne


def compute_pr_curves(predictions, targets, num_classes, iou_threshold=0.5):
    # Prépare les données pour tracer les courbes Précision-Rappel et calcule
    # l'AP pour chaque classe
    curves = {}

    for cls_id in range(1, num_classes):
        y_true = []
        y_scores = []

        for pred, tgt in zip(predictions, targets):
            gt_boxes = tgt["boxes"]
            gt_labels = tgt["labels"]

            pred_boxes = pred["boxes"]
            pred_labels = pred["labels"]
            pred_scores = pred["scores"]

            used_gt = set()

            for pb, pl, ps in zip(pred_boxes, pred_labels, pred_scores):
                if pl != cls_id:
                    continue

                if len(gt_boxes) == 0:
                    y_true.append(0)
                    y_scores.append(ps.item())
                    continue

                ious = torchvision.ops.box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
                max_iou, max_idx = torch.max(ious, dim=0)

                if max_iou >= iou_threshold and gt_labels[max_idx] == cls_id and max_idx.item() not in used_gt:
                    y_true.append(1)
                    used_gt.add(max_idx.item())
                else:
                    y_true.append(0)

                y_scores.append(ps.item())

        if len(y_true) > 0:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            ap = average_precision_score(y_true, y_scores)

            curves[cls_id] = {
                "precision": precision,
                "recall": recall,
                "ap": ap
            }

    return curves


pr_curves = compute_pr_curves(predictions, targets, len(classes), IOU_THRESH)

# Tracer les courbes Rappel précision

plt.figure(figsize=(10, 8))

for cls_id, data in pr_curves.items():
    plt.plot(data["recall"], data["precision"], label=f"{classes[cls_id]} (AP={data['ap']:.2f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(save_dir / "pr_curves.png")
plt.close()


# Diagramme de bars de la précision moyenne par classe

ap_values = []
cls_names = []

for cls_id, data in pr_curves.items():
    cls_names.append(classes[cls_id])
    ap_values.append(data["ap"])

plt.figure(figsize=(12, 6))
plt.bar(cls_names, ap_values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("AP")
plt.title("Average Precision per Class")
plt.tight_layout()
plt.savefig(save_dir / "ap_per_class.png")
plt.close()


# COCO mAP(mean Average Précision) officiel (cf. méthode COCO internet)


def predictions_to_coco(predictions, image_ids):
    # Formate la liste des prédictions pour qu'elle corresponde à la structure
    # JSON attendue par l'outil d'évaluation officiel COCO.
    coco_results = []

    for pred, img_id in zip(predictions, image_ids):
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            coco_results.append({
                "image_id": int(img_id),
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


coco_gt = COCO(str(val_json))

if "info" not in coco_gt.dataset:
    coco_gt.dataset["info"] = {"description": "custom"}

image_ids = dataset.ids
coco_preds = predictions_to_coco(predictions, image_ids)

coco_dt = coco_gt.loadRes(coco_preds)

coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


# Précision moyenne COCO par classes

ap_per_class_coco = {}

for idx, cls_name in enumerate(classes):
    if idx == 0:
        continue

    coco_eval_cls = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval_cls.params.catIds = [idx]
    coco_eval_cls.evaluate()
    coco_eval_cls.accumulate()
    coco_eval_cls.summarize()

    if len(coco_eval_cls.stats) > 0:
        ap = coco_eval_cls.stats[0]
    else:
        ap = float("nan")

    ap_per_class_coco[cls_name] = float(ap)


# Temps inférence

mean_inf_time = float(np.mean(inference_times))

# Retourne un .json pour les métriques souhaitées

report = {
    "mean_inference_time_sec": mean_inf_time,
    "prf_per_class": prf_metrics,
    "ap_sklearn_per_class": {classes[k]: float(v["ap"]) for k, v in pr_curves.items()},
    "ap_coco_per_class": ap_per_class_coco
}

with open(save_dir / "final_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Courbes toutes plot, json prêt avec les métriques")
print(f"\n temps inférence moyen: {mean_inf_time:.4f}s")