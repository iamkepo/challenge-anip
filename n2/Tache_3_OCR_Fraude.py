# Tâche 3 — OCR & Détection de fraude sur documents (script PyTorch)
# Auteur : Christ-Amour Kakpo
# Objectif : OCR & Détection de fraude sur documents
# %%
"""
OCR & Détection de fraude sur documents
--------------------------------------
Usage :
# Entraînement
python Tache_3_OCR_Fraude.py --mode train \
  --data_dir data/dataset_tache_3/dataset_tache_3/train \
  --out_dir outputs/tache3

# Évaluation
python Tache_3_OCR_Fraude.py --mode eval \
  --data_dir data/dataset_tache_3/dataset_tache_3/test \
  --out_dir outputs/tache3 \
  --resume outputs/tache3/checkpoints/ckpt_ep5.pth
"""

import os
from pathlib import Path
import argparse
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd

import easyocr

# ----------------------------- Utils -------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list_images(folder, exts={'.jpg','.jpeg','.png'}):
    folder = Path(folder)
    if not folder.exists():
        return []
    return [str(x) for x in folder.rglob('*') if x.suffix.lower() in exts]

def extract_text_easyocr(image_path, reader=None):
    if reader is None:
        reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    texts = [res[1] for res in results]
    return " ".join(texts)

# ----------------------------- Dataset -------------------------------

class OCRFraudDataset(Dataset):
    """
    Dataset OCR & fraude.
    - Mode train : explore sous-dossiers normal, forgery_1..4 => labels connus
      même si images sont dans des sous-sous-dossiers.
    - Mode test : explore uniquement images => pas de labels
    """
    def __init__(self, root_dir, transform=None, mode="train"):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mode = mode
        self.records = []

        label_map = {'normal':0,'forgery_1':1,'forgery_2':2,'forgery_3':3,'forgery_4':4}

        if self.mode == "train":
            for country in self.root_dir.iterdir():
                if not country.is_dir():
                    continue
                for label in label_map.keys():
                    folder = country / label
                    if folder.exists():
                        # recursively add all images from subfolders
                        for f in folder.rglob("*"):
                            if f.suffix.lower() in {'.jpg','.jpeg','.png'}:
                                self.records.append((str(f), label_map[label]))
        else:  # mode test
            for country in self.root_dir.iterdir():
                if not country.is_dir():
                    continue
                for f in country.rglob("*"):
                    if f.suffix.lower() in {'.jpg','.jpeg','.png'}:
                        self.records.append((str(f), -1))  # -1 => pas de label

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)

# ----------------------------- Model -------------------------------

class OCRFraudNet(nn.Module):
    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(r.children())[:-1])
        in_dim = r.fc.in_features
        self.fc = nn.Linear(in_dim, 256)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.head(x)

# ----------------------------- Training -------------------------------

def train_one_epoch(model, loader, opt, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total = 0
    for imgs, labels, _ in tqdm(loader, desc="Train"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()*imgs.size(0)
        total += imgs.size(0)
    return total_loss/total

def evaluate(model, loader, device, reader):
    model.eval()
    preds_all, files_all, texts_all = [], [], []
    with torch.no_grad():
        for imgs, _, paths in tqdm(loader, desc="Eval"):
            imgs = imgs.to(device)
            logits = model(imgs)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds_all.extend(pred)
            files_all.extend(paths)
            for path in paths:
                text = extract_text_easyocr(path, reader=reader)
                texts_all.append(text)
    return files_all, preds_all, texts_all

# ----------------------------- Main -------------------------------

def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)
    (out_dir / "submissions").mkdir(exist_ok=True)

    train_transform = T.Compose([
        T.Resize((160,160)),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    eval_transform = T.Compose([
        T.Resize((128,128)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    if args.mode == "train":
        train_ds = OCRFraudDataset(args.data_dir, transform=train_transform, mode="train")
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

        # Optional: subset for fast validation during training (not used here but prepared)
        val_subset = None
        if len(train_ds) > 2000:
            val_subset = Subset(train_ds, list(range(2000)))

        model = OCRFraudNet().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs+1):
            loss = train_one_epoch(model, train_loader, opt, device)
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
            ckpt = out_dir / "checkpoints" / f"ckpt_ep{epoch}.pth"
            torch.save({"model": model.state_dict()}, ckpt)
            print(f"Checkpoint saved: {ckpt}")

    elif args.mode == "eval":
        model = OCRFraudNet().to(device)
        if args.resume and Path(args.resume).exists():
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck["model"])
            print("Loaded checkpoint", args.resume)

        test_ds = OCRFraudDataset(args.data_dir, transform=eval_transform, mode="test")
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        reader = easyocr.Reader(['en'])
        files, preds, texts = evaluate(model, test_loader, device, reader)
        idx2label = {0:"normal",1:"forgery_1",2:"forgery_2",3:"forgery_3",4:"forgery_4"}
        out_rows = [{"image": Path(f).name, "class": idx2label[p], "ocr_text": t} for f,p,t in zip(files,preds,texts)]
        out_csv = out_dir / "submissions" / "task3_ocr_fraud_submission.csv"
        pd.DataFrame(out_rows).to_csv(out_csv, index=False)
        print("✅ Saved submission to", out_csv)

# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train","eval"])
    parser.add_argument("--data_dir", type=str, required=True, help="Dataset train/ ou test/")
    parser.add_argument("--out_dir", type=str, default="outputs/tache3")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)