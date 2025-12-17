import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
import json
import torch
import torchvision


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision.models.detection.rpn import AnchorGenerator



ROOT = Path(__file__).resolve().parents[3]   # racine projet
sys.path.append(str(ROOT / "src"))

num_classes = 15



# Taken from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.epoch = 0
        self.trace_func = trace_func
    def __call__(self, val_loss, epoch, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CocoWrapper(CocoDetection):
    """Module d'extraction des images et targets à l'aide de CocoDetection, pour formater les cibles comme attendu par Faster R-CNN.
    Retourne une image et un dictionnaire avec les clés "boxes" et "labels"
    """
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        img = functional.to_tensor(img)
        
        boxes = []
        labels = []
        for t in targets:
            boxes.append(t["bbox"])       # format COCO (x, y, w, h)
            labels.append(t["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = torchvision.ops.box_convert(boxes, "xywh", "xyxy")
        target["labels"] = labels

        return img, target


def get_model(num_classes):
    # 1) Charger le modèle pré-entraîné COCO (91 classes)
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # 2) Remplacer la tête de classification
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )
    return model



def make_sampler(coco_json_path, dataset_len):
    """Crée un objet d'échantillonage pour équilibrer les classes lors de l'entraînement, à partir de leur fréquence dans les annotations COCO.
    Args:
        coco_json_path: chemin vers le fichier JSON des annotations COCO
        dataset_len: longueur du dataset (d'entraînement)"""
    with open(coco_json_path) as f:
        coco = json.load(f)

    counts = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        counts[cid] = counts.get(cid, 0) + 1

    # poids = inverse fréquence
    class_weights = {cid: 1 / np.sqrt(c) for cid, c in counts.items()}

    # poids pour CHAQUE image
    weights = []
    imgid_to_annotations = {}

    for ann in coco["annotations"]:
        imgid = ann["image_id"]
        imgid_to_annotations.setdefault(imgid, []).append(ann["category_id"])

    for imgid in sorted(imgid_to_annotations.keys()):
        labels = imgid_to_annotations[imgid]
        w = max(class_weights[c] for c in labels)   # image pondérée par sa classe la + rare
        weights.append(w)

    return WeightedRandomSampler(weights, num_samples=dataset_len, replacement=True)


def compute_validation_loss(model, valid_loader):
    """Calcul de la loss sur le jeu de validation"""
    val_losses = []
    model.train()

    with torch.no_grad():  
        for imgs, targets in valid_loader:
            imgs = [i.cuda(non_blocking=True) for i in imgs]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]

            # calcul des losses (mode train obligatoire pour Faster-RCNN)
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values()).item()
            val_losses.append(loss)

    return np.mean(val_losses)



def collate_fn(batch):
    """Fonction de collate pour DataLoader adaptée à Faster R-CNN. remplace l'argument collate_fn du DataLoader.
    Args:
        batch: liste de tuples (img, target)
    Returns:
        tuple de listes: (list of imgs, list of targets)
    """
    return [b[0] for b in batch], [b[1] for b in batch]



# répertoire d'entraînement
train_dir = ROOT / "merged_dataset" / "train"
path_json =  train_dir / "_annotations.coco.json"

# répertoire de validation
val_dir = ROOT / "CADOT_Dataset" / "valid"
val_json = val_dir / "_annotations.coco.json"

# Sauvegarde du modèle
save_train = ROOT / "src" / "cadot" / "models"
save_train.mkdir(parents=True, exist_ok=True)


# l'argument collate_fn permet de regrouper correctement les données dans un batch compatible avec ce qu'attend Faster R-CNN
# Quand on reçoit un batch, x est du genre
# [
#     (img1, target1),
#     (img2, target2),
#     (img3, target3),
#     (img4, target4)
# ]
# et on veut transformer ça en
# (
#     [[img1, img2, img3, img4],
#     [target1, target2, target3, target4]]
# )

num_classes = 15

# Remplacement de la tête COCO (91 classes) par ta tête de classification (15 classes pour notre problème)
# Nécessaire car la modèle préimporté est entraîné sur COCO 91 classes
model = get_model(num_classes=15)
model.train()
model.to("cuda")

# Optimiseur
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# momentum permet de lisser les mises à jour des poids en conservant un historique des gradients précédents
# weight decay est une régulatisation L2 qui pénalise les poids trop grands pour éviter l'overfitting


def train(model, train_loader, valid_loader):
    """Entraînement du modèle Faster R-CNN avec early stopping.
    Args:
        model: modèle Faster R-CNN
        train_loader: DataLoader pour le jeu d'entraînement
        valid_loader: DataLoader pour le jeu de validation
    """
    model.train()
    model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(
        patience=7,
        verbose=True,
        delta=0,
        path=str(save_train / "faster_rcnn_best.pth")
    )

    num_epochs = 20
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        pbar = tqdm(train_loader, desc="Training", ncols=100)

        for imgs, targets in pbar:

            imgs = [i.cuda() for i in imgs]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": loss.item()})

        val_loss = compute_validation_loss(model, valid_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        early_stopping(val_loss, epoch, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Recharger le meilleur modèle
    model.load_state_dict(torch.load(save_train / "faster_rcnn_best.pth"))
    print("Meilleur modèle chargé")


def evaluate(model, valid_loader):
    """Évaluation du modèle sur le jeu de validation et sauvegarde des prédictions"""
    print("Évaluation du modèle...")
    model.eval()

    predictions = []
    targets_list = []

    with torch.no_grad():
        for imgs, trgts in valid_loader:

            imgs = [i.cuda() for i in imgs]
            outputs = model(imgs)

            for out, tgt in zip(outputs, trgts):
                predictions.append({
                    "boxes": out["boxes"].cpu(),
                    "labels": out["labels"].cpu(),
                    "scores": out["scores"].cpu()
                })

                targets_list.append({
                    "boxes": tgt["boxes"].cpu(),
                    "labels": tgt["labels"].cpu()
                })

    torch.save({
        "predictions": predictions,
        "targets": targets_list
    }, save_train / "faster_rcnn_predictions.pt")

    print("Évaluation sauvegardée dans faster_rcnn_predictions.pt")



def main():
    train_dataset = CocoWrapper(str(train_dir), str(path_json))
    valid_dataset = CocoWrapper(str(val_dir), str(val_json))

    sampler = make_sampler(path_json, len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=4, sampler = sampler,
                              shuffle=False, num_workers=8,
                              pin_memory=True, collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset, batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    model = get_model(num_classes=15).cuda()

    print("Début de l'entraînement")
    train(model, train_loader, valid_loader)

    evaluate(model, valid_loader)


if __name__ == "__main__":
    main()