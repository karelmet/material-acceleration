from xml.parsers.expat import model
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


# CONFIGURATION DES CHEMINS
ROOT = Path(__file__).resolve().parents[3]   
sys.path.append(str(ROOT / "src"))

num_classes = 15


# Source: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Arrêt précoce (Early Stopping) pour éviter le sur-apprentissage."""
    def __init__(self, patience=13, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        """
        Args:
            patience (int): Nombre d'époques à attendre après la dernière amélioration de la loss.
            verbose (bool): Si True, affiche les messages de sauvegarde.
            delta (float): Amélioration minimale requise pour être considérée comme significative.
            path (str): Chemin de sauvegarde du modèle.
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
            # Pas d'amélioration significative
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Amélioration trouvée : on reset le compteur et on sauvegarde
            self.best_score = score
            self.epoch = epoch
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Sauvegarde le modèle quand la loss de validation diminue.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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


# DÉFINITION DU MODÈLE

def get_model(num_classes, anchor_generator):
    """
    Charge le modèle Faster R-CNN avec un backbone ResNet-50-FPN.
    ARCHITECTURE :
    1. Backbone : ResNet-50 pré-entraîné sur COCO.
    2. Neck : FPN pour la détection multi-échelle.
    3. Head : Remplacement de la tête de classification finale pour adapter 
              le modèle à notre nombre spécifique de classes (15 au lieu de 91).
    """
    # Chargement du modèle avec poids par défaut (pré-entraîné sur COCO)
    model = fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        rpn_anchor_generator=anchor_generator,
        min_size=512,
        max_size=1024
    )

    # Récupération de la dimension d'entrée de la tête de classification existante
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Remplacement par une nouvelle tête vierge adaptée à nos 15 classes
    # FastRCNNPredictor gère à la fois la classification et la régression des boîtes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Paramètres RPN (ajustés pour améliorer la détection de petits objets si nécessaire)
    model.rpn.fg_iou_thresh = 0.4
    model.rpn.bg_iou_thresh = 0.1
    model.rpn.pre_nms_top_n_train = 8000
    model.rpn.post_nms_top_n_train = 4000
    model.rpn.pre_nms_top_n_test = 4000
    model.rpn.post_nms_top_n_test = 2000

    return model


# STRATÉGIE D'ÉCHANTILLONNAGE
# Associer des poids proportionnels au nombre d'éléments de chaques classes. Ces "poids" agissent comme une probabilité d'être tiré dans un batch lors de l'entraînement.
# On cherche à augmenter les poids des classes sous représentées pour que le modèle en apprenne mieux les caractéristiques.
def make_sampler(coco_json_path, dataset):
    with open(coco_json_path) as f:
        coco = json.load(f)

    # 1) Comptage des classes
    counts = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        counts[cid] = counts.get(cid, 0) + 1

    class_weights = {cid: 1 / np.sqrt(c) for cid, c in counts.items()}

    # 2) image_id -> catégories
    imgid_to_cats = {}
    for ann in coco["annotations"]:
        imgid = ann["image_id"]
        imgid_to_cats.setdefault(imgid, []).append(ann["category_id"])

    # 3) Map image_id -> dataset index
    id_to_index = {img_id: i for i, img_id in enumerate(dataset.ids)}

    weights = [0.0] * len(dataset)

    for imgid, cats in imgid_to_cats.items():
        if imgid in id_to_index:
            idx = id_to_index[imgid]
            weights[idx] = max(class_weights[c] for c in cats)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# FONCTIONS D'ENTRAÎNEMENT ET VALIDATION

def compute_validation_loss(model, valid_loader):
    """
    Calcule la loss sur le jeu de validation.
    Contrairement aux modèles de classification classiques, Faster R-CNN ne retourne les losses 
    que s'il est en mode train(). En mode eval(), il retourne des prédictions (boxes).
    
    Passer en mode train() mais désactiver le calcul des gradients (torch.no_grad)
    pour évaluer la loss sans modifier les poids.
    """
    val_losses = []
    model.train()  # Indispensable pour récupérer le dictionnaire de losses

    with torch.no_grad():  
        for imgs, targets in valid_loader:
            imgs = [i.cuda(non_blocking=True) for i in imgs]
            targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            # La loss totale est la somme des losses du RPN et du RCNN (cf rapport)
            loss = sum(loss_dict.values()).item()
            val_losses.append(loss)

    return np.mean(val_losses)


def collate_fn(batch):
    """
    Fonction de regroupement personnalisée
    Par défaut, PyTorch essaie d'empiler les tenseurs. Or, en détection d'objets,
    les images peuvent avoir des tailles différentes et
    le nombre de bounding boxes varie d'une image à l'autre.
    On ne peut donc pas créer un tenseur unique pour les cibles.
    """
    return [b[0] for b in batch], [b[1] for b in batch]


# CONFIGURATION INITIALE

# Répertoires des données
train_dir = ROOT / "merged_dataset" / "train"
path_json =  train_dir / "_annotations.coco.json"

val_dir = ROOT / "CADOT_Dataset" / "valid"
val_json = val_dir / "_annotations.coco.json"

# Création du dossier de sauvegarde
save_train = ROOT / "src" / "cadot" / "models"
save_train.mkdir(parents=True, exist_ok=True)

# Définition personnelle des ancres pour le RPN afin d'avoir une meilleure modularité. 
# Adapté aux objets de tailles variées (de très petit 8x8 à grand 128x128)
# Sans cela le modèle repère moins bien les petits éléments.
anchor_generator = AnchorGenerator(
    sizes=((16,), (32,), (64,), (128,), (256,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)


def train(model, train_loader, valid_loader):
    """
    Boucle principale d'entraînement.
    Inclus : AMP (Automatic Mixed Precision) et Early Stopping.
    """
    model.train()
    model.cuda()

    # Optimiseur SGD
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0005)
    
    # Scaler pour la précision mixte, accélère le calcul
    # J'AI AUSSI ENTRAINÉ DES MODÈLES SUR MA MACHINE PERSONELLE, D'OU LA NECESSITÉ D'UTILISER LA GPU AMP POUR PLUS DE PUISSANCE
    scaler = torch.cuda.amp.GradScaler()
    
    early_stopping = EarlyStopping(
        patience=13,
        verbose=True,
        delta=0,
        path=str(save_train / "faster_rcnn_best.pth")
    )

    # Nombre d'epochs de l'entraînement!
    num_epochs = 30
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        pbar = tqdm(train_loader, desc="Training", ncols=100)

        for imgs, targets in pbar:
            # Transfert GPU
            imgs = [i.cuda() for i in imgs]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # Forward pass en précision mixte
            with torch.cuda.amp.autocast():
                loss_dict = model(imgs, targets)
                # Somme des composantes de la loss
                loss = sum(loss_dict.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": loss.item()})

        # Validation à la fin de l'époque
        val_loss = compute_validation_loss(model, valid_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Vérification Early Stopping
        early_stopping(val_loss, epoch, model)
        if early_stopping.early_stop:
            print(f"Early stopping déclenché à l'epoch {epoch+1}")
            break

    # Recharger le meilleur modèle pour l'évaluation finale
    model.load_state_dict(torch.load(save_train / "faster_rcnn_best.pth"))
    print("Meilleur modèle chargé")


def evaluate(model, valid_loader):
    """
    Génère les prédictions sur le jeu de validation pour calcul des métriques.
    """
    print("Évaluation du modèle")
    model.eval()

    predictions = []
    targets_list = []

    with torch.no_grad():
        for imgs, trgts in valid_loader:
            imgs = [i.cuda() for i in imgs]
            
            # Forward pass (retourne boxes, labels, scores)
            outputs = model(imgs)

            # Transfert CPU pour sauvegarde
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
    # 1. Préparation des Datasets
    train_dataset = CocoWrapper(str(train_dir), str(path_json))
    valid_dataset = CocoWrapper(str(val_dir), str(val_json))

    # 2. Création du Sampler pondéré
    sampler = make_sampler(path_json, train_dataset)

    # 3. Création des DataLoaders
    # num_workers=8 pour paralléliser le chargement des images
    # pin_memory=True pour accélérer le transfert RAM -> VRAM
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler,
                              shuffle=False, num_workers=8,
                              pin_memory=True, collate_fn=collate_fn)

    valid_loader = DataLoader(valid_dataset, batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    # 4. Instanciation du modèle sur GPU
    model = get_model(num_classes=15, anchor_generator=anchor_generator).cuda()

    print("Début de l'entraînement")
    train(model, train_loader, valid_loader)

    # 5. Évaluation finale
    evaluate(model, valid_loader)

if __name__ == "__main__":
    main()