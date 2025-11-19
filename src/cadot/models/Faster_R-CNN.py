import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


class CocoWrapper(CocoDetection):
    """Wrapper autour de CocoDetection pour formater les cibles comme attendu par Faster R-CNN.
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

if __name__ == "__main__":
    # Chemin absolu pour obtenir la racine du projet
    ROOT = Path(__file__).resolve().parents[3]

    # répertoire d'entraînement
    train_dir = ROOT / "CADOT_Dataset" / "train"
    train_json = train_dir / "_annotations.coco.json"

    train_dataset = CocoWrapper(str(train_dir), str(train_json))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

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
    #     [img1, img2, img3, img4],
    #     [target1, target2, target3, target4]
    # )


    # Modèle
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model.train()
    model.to("cuda")


    # Optimiseur
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    # momentum permet de lisser les mises à jour des poids en conservant un historique des gradients précédents
    # weight decay est une régulatisation L2 qui pénalise les poids trop grands pour éviter l'overfitting


    # Entraînement
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # barre de progression graphique dans le terminal
        pbar = tqdm(train_loader, desc=f"Training", ncols=100)

        for imgs, targets in pbar:
            imgs = [i.cuda() for i in imgs]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

            # backpropagation et mise à jour des poids
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # afficher la loss en live
            pbar.set_postfix({"loss": loss.item()})

        print("Epoch", epoch, "Loss:", loss.item())
        
    # Sauvegarde du modèle
    torch.save(model.state_dict(), ROOT / "material-acceleration" / "cadot" / "models" / "faster_rcnn.pth")
    print("Modèle sauvegardé en .pth et a pour nom faster_rcnn")


    # Evaluation
    model.eval()
    val_dir = ROOT / "CADOT_Dataset" / "valid"
    val_json = val_dir / "_annotations.coco.json"
    valid_dataset = CocoWrapper(str(val_dir), str(val_json))
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    with torch.no_grad():
        for imgs, targets in tqdm(valid_loader, desc="Evaluation", ncols=100):
            imgs = [i.cuda() for i in imgs]
            outputs = model(imgs)           # Ici, outputs contient les prédictions du modèle
            print("Prédictions boxes:", outputs[0]["boxes"][:3])
            print("Scores:", outputs[0]["scores"][:3])
            print("Labels:", outputs[0]["labels"][:3])
            break  # enlever pour tester tout le dataset