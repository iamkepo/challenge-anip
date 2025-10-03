# Tâche 1 — Reconnaissance faciale (script PyTorch)
# Auteur : Christ-Amour Kakpo
# Objectif : Entraîner un modèle de reconnaissance faciale et produire un CSV de soumission
# Fichier: Tache_1_Reconnaissance_Faciale.py
# Format: script / notebook (cells séparés par # %% pour faciliter l'ouverture dans Jupyter/VSCode)
# Objectif: fournir un pipeline PyTorch minimal et reproductible pour extraire des embeddings
#           entraîner un modèle simple (classification->embeddings) et produire un CSV de matching
# Structure attendue des dossiers (à adapter):
#  - data/
#      - train/  (images d'entraînement nommées XXXX_0.jpg, XXXX_1.jpg etc.)
#      - test/   (images de test img_0001.jpg...)
#  - outputs/
#      - checkpoints/
#      - embeddings/
#      - submissions/

# %%
"""
INSTALL (virtualenv recommandé):
  pip install torch torchvision timm pandas scikit-learn faiss-cpu pillow tqdm
  # si vous avez GPU et voulez faiss-gpu, installez faiss-gpu approprié

Usage rapide:
  python Tache_1_Reconnaissance_Faciale.py --mode train --data_dir data --out_dir outputs
  python Tache_1_Reconnaissance_Faciale.py --mode eval --data_dir data --out_dir outputs

Ce script implémente:
 - Dataset PyTorch pour les images d'entraînement (labels dérivés du nom de fichier)
 - Backbone ResNet50 (timm) pour extraire un embedding L2-normalisé
 - Head classification (optionnel) pour fine-tuning
 - Boucle d'entraînement simplifiée
 - Extraction d'embeddings sur train/test
 - Matching par nearest neighbor (Faiss si présent, sinon sklearn)
 - Génération d'un CSV de sortie mapping test_image -> predicted_ID

Remarques:
 - Pour de meilleures performances: utiliser ArcFace (insightface) ou pytorch-metric-learning
 - Ici on propose une pipeline simple et reproductible.
"""

# %%
import os
import argparse
from pathlib import Path
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

import pandas as pd

# Try to import faiss; fallback to sklearn
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    from sklearn.neighbors import NearestNeighbors
    _HAS_FAISS = False

# %%
# ----------------------------- Utilities ----------------------------------

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder, exts={'.jpg', '.jpeg', '.png'}):
    p = Path(folder)
    if not p.exists():
        return []
    return [str(x) for x in p.rglob('*') if x.suffix.lower() in exts]

def extract_pid(filename):
    # Robust extraction of pid from filename like XXXX_0.jpg or XXXX-0.jpg
    name = Path(filename).stem
    # Try split by '_' then by '-'
    if '_' in name:
        pid = name.split('_')[0]
    elif '-' in name:
        pid = name.split('-')[0]
    else:
        # fallback: whole stem
        pid = name
    return pid

# %%
# ----------------------------- Dataset ------------------------------------

class FaceMatchingDataset(Dataset):
    """Dataset that reads images and extracts label from filename.
    Filename expected format for train: XXXX_0.jpg or XXXX-0.jpg
    Returns: image tensor, label (int) or -1 for test.
    """
    def __init__(self, files, label_map=None, transform=None, is_train=True):
        self.files = []
        self.records = []
        self.transform = transform
        self.is_train = is_train
        self.label_map = label_map or {}

        for f in files:
            lbl = None
            if is_train:
                pid = extract_pid(f)
                if pid not in self.label_map:
                    self.label_map[pid] = len(self.label_map)
                lbl = self.label_map[pid]
            self.records.append((f, lbl))
            self.files.append(f)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        f, lbl = self.records[idx]
        img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if lbl is None:
            lbl = -1
        return img, lbl, f


# %%
# ----------------------------- Model --------------------------------------

class BackboneEmbedding(nn.Module):
    def __init__(self, out_dim=512, pretrained=True):
        super().__init__()
        # use torchvision resnet18
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        # remove fc
        self.features = nn.Sequential(*list(r.children())[:-1])
        in_dim = r.fc.in_features
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.features(x)  # B x C x 1 x 1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x

# Optional classification head for fine-tuning
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim=512, n_classes=1000):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        return self.fc(x)

# %%
# ----------------------------- Training Loop ------------------------------

def train_one_epoch(model, head, loader, opt, device, epoch, scheduler=None):
    model.train()
    if head: head.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for imgs, labels, _ in tqdm(loader, desc=f"Train ep {epoch}"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        emb = model(imgs)
        logits = head(emb)
        loss = loss_fn(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)
    return total_loss / total

# %%
# ----------------------------- Embeddings ---------------------------------

def extract_embeddings(model, loader, device):
    model.eval()
    emb_list = []
    path_list = []
    with torch.no_grad():
        for imgs, _, paths in tqdm(loader, desc='Extract emb'):
            imgs = imgs.to(device)
            embs = model(imgs)
            embs = embs.cpu().numpy()
            emb_list.append(embs)
            path_list.extend(paths)
    if len(emb_list) == 0:
        return np.zeros((0, model.fc.out_features)), []
    emb_arr = np.vstack(emb_list)
    return emb_arr, path_list

# %%
# ----------------------------- Matching -----------------------------------

def build_index(embeddings):
    if _HAS_FAISS:
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype(np.float32))
        return index
    else:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(embeddings)
        return nbrs


def query_index(index, embeddings, k=1):
    if _HAS_FAISS:
        D, I = index.search(embeddings.astype(np.float32), k)
        return I, D
    else:
        D, I = index.kneighbors(embeddings, n_neighbors=k)
        return I, D

# %%
# ----------------------------- Main / CLI ---------------------------------

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Directories
    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'checkpoints').mkdir(exist_ok=True)
    (out_dir / 'embeddings').mkdir(exist_ok=True)
    (out_dir / 'submissions').mkdir(exist_ok=True)

    # List train files: if train/ does not exist, list all images in data_dir
    if train_dir.exists():
        train_files = list_images(train_dir)
    else:
        print(f"Warning: {train_dir} does not exist. Listing all images in {data_dir} for training.")
        train_files = list_images(data_dir)
    test_files = list_images(test_dir)

    print(f"Train files ({len(train_files)}): {train_files[:5]}{'...' if len(train_files)>5 else ''}")
    print(f"Test files ({len(test_files)}): {test_files[:5]}{'...' if len(test_files)>5 else ''}")

    if len(test_files) == 0:
        print("⚠️ Aucun fichier trouvé dans le dossier test/. Impossible de générer un CSV.")
        return

    # Build label map robustly
    label_map = {}
    for f in train_files:
        pid = extract_pid(f)
        if pid not in label_map:
            label_map[pid] = len(label_map)
    print(f"Label map (size {len(label_map)}): {dict(list(label_map.items())[:5])}{'...' if len(label_map)>5 else ''}")

    # If no labels found, skip training and model creation
    if len(label_map) == 0:
        print("⚠️ Aucun label détecté dans les images d'entraînement. Le modèle ne sera pas créé ni entraîné.")
        # But still produce a CSV with test images and predicted_id = 'unknown'
        out_rows = []
        for p in test_files:
            out_rows.append({'image': Path(p).name, 'predicted_id': 'unknown'})
        out_df = pd.DataFrame(out_rows)
        out_csv = out_dir / 'submissions' / 'task1_matching_submission.csv'
        out_df.to_csv(out_csv, index=False)
        print('Saved submission to', out_csv)
        return

    # Transforms
    train_tf = T.Compose([
        T.Resize((256,256)),
        T.RandomResizedCrop(224, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.15,0.15,0.15,0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Build datasets
    train_ds = FaceMatchingDataset(train_files, label_map=label_map, transform=train_tf, is_train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # For embedding extraction we want deterministic transform
    train_emb_ds = FaceMatchingDataset(train_files, label_map=label_map, transform=val_tf, is_train=False)
    train_emb_loader = DataLoader(train_emb_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_ds = FaceMatchingDataset(test_files, label_map=label_map, transform=val_tf, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    n_classes = len(label_map)
    print(f"Detected {n_classes} unique identities in train")

    # Instantiate model
    model = BackboneEmbedding(out_dim=args.embedding_dim, pretrained=args.pretrained).to(device)
    head = ClassificationHead(embedding_dim=args.embedding_dim, n_classes=n_classes).to(device)

    # If resume checkpoint provided
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        head.load_state_dict(ck['head'])
        print('Loaded checkpoint', args.resume)

    if args.mode == 'train':
        params = list(model.parameters()) + list(head.parameters())
        opt = torch.optim.AdamW(params, lr=args.lr)
        for epoch in range(1, args.epochs+1):
            loss = train_one_epoch(model, head, train_loader, opt, device, epoch)
            print(f"Epoch {epoch} loss: {loss:.4f}")
            # save checkpoint
            ckpt_path = out_dir / 'checkpoints' / f'ckpt_ep{epoch}.pth'
            torch.save({'model': model.state_dict(), 'head': head.state_dict(), 'epoch': epoch}, ckpt_path)

    # Always extract embeddings and produce matching CSV
    print('Extracting train embeddings...')
    train_embs, train_paths = extract_embeddings(model, train_emb_loader, device)
    # map train paths to person id (pid)
    train_pids = []
    for p in train_paths:
        pid = extract_pid(p)
        train_pids.append(pid)
    # unique mapping: keep first occurrence index for each pid (or average embeddings by pid)
    pid_to_indices = {}
    for i, pid in enumerate(train_pids):
        pid_to_indices.setdefault(pid, []).append(i)
    # Average embeddings per pid (recommended)
    pid_list = []
    pid_embs = []
    for pid, idxs in pid_to_indices.items():
        emb_mean = train_embs[idxs].mean(axis=0)
        pid_list.append(pid)
        pid_embs.append(emb_mean)
    pid_embs = np.vstack(pid_embs)

    print('Extracting test embeddings...')
    test_embs, test_paths = extract_embeddings(model, test_loader, device)

    print('Building index...')
    index = build_index(pid_embs)
    I, D = query_index(index, test_embs, k=1)

    # I shape: (n_test, 1)
    preds = [pid_list[idx[0]] for idx in I]
    out_rows = []
    for p, pred in zip(test_paths, preds):
        out_rows.append({'image': Path(p).name, 'predicted_id': pred})
    out_df = pd.DataFrame(out_rows)
    out_csv = out_dir / 'submissions' / 'task1_matching_submission.csv'
    out_df.to_csv(out_csv, index=False)
    print('Saved submission to', out_csv)

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train','eval'], help='train or eval')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true', help='force CPU')
    args = parser.parse_args()
    main(args)
