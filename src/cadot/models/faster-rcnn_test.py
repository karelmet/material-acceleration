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


# CHEMINS ET DATALOADERS
ROOT = Path(__file__).resolve().parents[3]
save_train = ROOT / "src" / "cadot" / "models"

val_dir = ROOT / "CADOT_Dataset" / "valid"
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
state_dict = torch.load(save_train / "faster_rcnn.pth", map_location=device)
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
def compute_confusion_matrix(predictions, targets, iou_threshold=0.5):
    y_true = []
    y_pred = []

    for pred, tgt in zip(predictions, targets):
        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"].tolist()

        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"].tolist()
        pred_scores = pred["scores"].tolist()

        matched_gt = set()

        for pb, pl, _ in zip(pred_boxes, pred_labels, pred_scores):
            object = torchvision.ops.box_iou(pb.unsqueeze(0), gt_boxes).squeeze(0)
            max_iou, max_idx = torch.max(object, dim=0)

            if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                y_true.append(gt_labels[max_idx.item()])
                y_pred.append(pl)
                matched_gt.add(max_idx.item())
            else:
                y_true.append(0)  # Background
                y_pred.append(pl)

        for i, gl in enumerate(gt_labels):
            if i not in matched_gt:
                y_true.append(gl)
                y_pred.append(0)  # Background

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

    thresh = cm.max() / 2.      # Threshold for text color
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            
    fig.tight_layout()
    plt.savefig(save_train / "faster_rcnn_confusion_matrix.png")


cm = compute_confusion_matrix(
    torch.load(save_train / "faster_rcnn_predictions.pt")["predictions"],
    torch.load(save_train / "faster_rcnn_predictions.pt")["targets"],
    iou_threshold=0.5,
)

plot_confusion_matrix(
    cm,
    classes
)


# PRÉCISION-RAPPEL ET AP PAR CLASSE
def prepare_pr_data(predictions, targets, classes, iou_threshold=0.5):
    """Prépare les données pour les courbes précision-rappel et le calcul de l'AP par classe.
        iou_threshold: Seuil IoU pour considérer une prédiction comme un vrai positif.
        Returns:
        all_targets (list of list): Liste des cibles binaires par classe.
        all_scores (list of list): Liste des scores de confiance par classe."""
    num_classes = len(classes)

    # Une liste par classe
    all_targets = [[] for _ in range(num_classes)]
    all_scores  = [[] for _ in range(num_classes)]

    for pred, tgt in zip(predictions, targets):

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]

        # IoU matrix une seule fois
        ious = torchvision.ops.box_iou(pred_boxes, gt_boxes)

        # Parcours direct des prédictions
        for p_idx in range(len(pred_boxes)):
            cls = pred_labels[p_idx].item()
            score = pred_scores[p_idx].item()

            # Associer score
            all_scores[cls].append(score)

            # Trouver le GT le plus proche
            best_iou, best_gt_idx = torch.max(ious[p_idx], dim=0)

            # Vrai positif : bonne classe + iou suffisant
            if best_iou >= iou_threshold and gt_labels[best_gt_idx] == cls:
                all_targets[cls].append(1)
            else:
                all_targets[cls].append(0)

    return all_targets, all_scores



# TRACER LES COURBES PRÉCISION-RAPPEL
def plot_precision_recall(all_targets, all_scores, class_names):
    plt.figure(figsize=(10, 8))

    for cls_idx, cls_name in enumerate(class_names):
        y_true = np.array(all_targets[cls_idx])
        y_scores = np.array(all_scores[cls_idx])

        if len(np.unique(y_true)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        plt.plot(recall, precision, label=cls_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves per Class")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(save_train / "faster_rcnn_precision_recall.png")



all_targets, all_scores = prepare_pr_data(predictions, targets, classes, iou_threshold=0.5)

plot_precision_recall(all_targets, all_scores, classes)

# CALCULER L'AVERAGE PRECISION (AP) PAR CLASSE
for cls_idx in range(len(classes)):
    y_true = np.array(all_targets[cls_idx])
    y_scores = np.array(all_scores[cls_idx])

    # Skip classe sans positifs
    if y_true.sum() == 0:
        print(f"Class {classes[cls_idx]} - AP: 0.0000 (aucun vrai positif)")
        continue

    # Skip classe sans scores
    if len(y_scores) == 0:
        print(f"Class {classes[cls_idx]} - AP: 0.0000 (aucun score)")
        continue

    ap = average_precision_score(y_true, y_scores)
    print(f"Class {classes[cls_idx]} - AP: {ap:.4f}")

