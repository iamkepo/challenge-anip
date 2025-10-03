# Tâche 2 — Estimation d'âge (script PyTorch)
# Auteur : Christ-Amour Kakpo
# Objectif : Entraîner un modèle d'estimation d'âge et produire un CSV de soumission
# Fichier: Tache_2_Estimation_Age.py
# Usage:
#  python Tache_2_Estimation_Age.py 
# --mode train 
# --data_dir data/dataset_tache_2/dataset_tache_2 
# --out_dir outputs/tache2 
# --epochs 10

#  python Tache_2_Estimation_Age.py 
# --mode eval  
# --data_dir data/dataset_tache_2/dataset_tache_2 
# --out_dir outputs/tache2 
# --cpu

# Ce script propose un pipeline simple et reproductible pour estimer l'âge:
# - parse labels depuis le nom de fichier XXXXXX_YZWW (WW = age)
# - backbone ResNet50 (pretrained)
# - tête multi-task: classification sur bins (0..100) + régression continue
# - loss combinée: CE + MSE
# - sauvegarde des checkpoints et génération d'un CSV de prédictions pour le dossier test

# Remarques:
# - Conçu pour fonctionner sur CPU (par défaut) ou GPU si disponible.
# - Si vous êtes sur macOS, préférez scikit-learn pour tout fallback lié à faiss etc.

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

# %% utils
def seed_everything(seed=42):
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Pour Mac MPS (PyTorch 2.x)
    if torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass

    # Pour reproductibilité sur CUDA
    cudnn.deterministic = True
    cudnn.benchmark = False

    print(f"✅ Seeds set to {seed} for reproducibility")


def list_images(folder, exts={'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}):
    p = Path(folder)
    if not p.exists():
        return []
    return [str(x) for x in p.rglob('*') if x.suffix in exts]

# %% Dataset

class AgeDataset(Dataset):
    """Dataset that parses age label from filename format XXXXXX_YZWW (WW=age).
    For train: expects files under data_dir/train/ ; for test: data_dir/test/ (no labels)
    Returns: image tensor, age_label (float) or -1 for test, filename
    """
    def __init__(self, files, transform=None, is_train=True, max_age=100):
        self.files = files
        self.transform = transform
        self.is_train = is_train
        self.max_age = max_age

    def __len__(self):
        return len(self.files)

    def _parse_age_from_name(self, path):
        name = Path(path).stem
        # Expect pattern: XXXXXX_YZWW where age is last 2 chars (or could be 1-3 digits)
        # We'll try to get trailing digits
        import re
        m = re.search(r'(\d{1,3})$', name)
        if m:
            try:
                age = int(m.group(1))
            except:
                age = None
        else:
            age = None
        return age

    def __getitem__(self, idx):
        p = self.files[idx]
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"⚠️ Erreur lecture image {p}: {e} — remplacement par image noire")
            img = Image.new('RGB', (224,224), (0,0,0))

        if self.transform:
            img_t = self.transform(img)
        else:
            img_t = T.ToTensor()(img)

        if self.is_train:
            age = self._parse_age_from_name(p)
            if age is None:
                age = -1
            # ⚡ Convertir en float32 directement
            age_t = torch.tensor(age, dtype=torch.float32)
        else:
            age_t = torch.tensor(-1.0, dtype=torch.float32)

        return img_t, age_t, p

# %% Model

class AgeEstimator(nn.Module):
    def __init__(self, n_bins=101, embedding_dim=512, pretrained=True):
        super().__init__()
        r = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(r.children())[:-1])  # outputs C x 1 x 1
        in_dim = r.fc.in_features
        self.fc = nn.Linear(in_dim, embedding_dim)
        # classification head (age bins 0..100)
        self.cls = nn.Linear(embedding_dim, n_bins)
        # regression head
        self.reg = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        emb = self.fc(x)
        emb = nn.functional.relu(emb)
        logits = self.cls(emb)
        reg = self.reg(emb).squeeze(1)
        return logits, reg

# %% Training & helpers

def train_one_epoch(model, loader, opt, device, epoch, scheduler=None, lambda_reg=1.0):
    model.train()
    loss_ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    total_loss = 0.0
    total_mae = 0.0
    total = 0

    for imgs, ages, _ in tqdm(loader, desc=f"Train ep {epoch}"):
        imgs = imgs.to(device)
        ages = ages.to(device)  # déjà float32 grâce à Dataset

        mask = (ages >= 0)
        if mask.sum() == 0:
            continue

        logits, reg = model(imgs)

        # classification target en entier pour CrossEntropy
        ages_int = ages.long().clamp(0, 100)

        loss1 = loss_ce(logits, ages_int)
        loss2 = loss_mse(reg[mask], ages[mask])
        loss = loss1 + lambda_reg * loss2

        opt.zero_grad()
        loss.backward()
        opt.step()

        if scheduler: 
            scheduler.step()

        with torch.no_grad():
            preds = (torch.softmax(logits, dim=1) * torch.arange(0, logits.size(1), device=device).float()).sum(dim=1)
            mae = torch.abs(preds[mask] - ages[mask]).sum().item()

        total_loss += loss.item() * imgs.size(0)
        total_mae += mae
        total += mask.sum().item()

    avg_loss = total_loss / (len(loader.dataset) + 1e-9)
    avg_mae = total_mae / (total + 1e-9)
    return avg_loss, avg_mae

def evaluate(model, loader, device):
    model.eval()
    total_mae = 0.0
    total = 0
    preds_all = []
    files_all = []

    with torch.no_grad():
        for imgs, ages, paths in tqdm(loader, desc='Eval'):
            imgs = imgs.to(device)
            ages = ages.to(device, dtype=torch.float32)  # ⚡ forcer float32

            logits, reg = model(imgs)

            # classification prédiction
            probs = torch.softmax(logits, dim=1)
            bins = torch.arange(0, logits.size(1), device=device, dtype=torch.float32)
            preds_cls = (probs * bins).sum(dim=1)

            # combinaison classification + régression
            preds = 0.5 * preds_cls + 0.5 * reg

            # calcul MAE si labels disponibles
            mask = (ages >= 0)
            if mask.sum() > 0:
                total_mae += torch.abs(preds[mask] - ages[mask]).sum().item()
                total += mask.sum().item()

            preds = preds.cpu().numpy()
            for p, pred in zip(paths, preds):
                preds_all.append(float(pred))
                files_all.append(Path(p).name)

    avg_mae = total_mae / (total + 1e-9) if total > 0 else None
    return files_all, preds_all, avg_mae

# %% Main

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    data_dir = Path(args.data_dir)
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'checkpoints').mkdir(exist_ok=True)
    (out_dir / 'submissions').mkdir(exist_ok=True)

    train_tf = T.Compose([
        T.Resize((160,160)),
        T.CenterCrop(128),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1,0.1,0.1,0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_tf = T.Compose([
        T.Resize((128,128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_files = list_images(train_dir)
    test_files = list_images(test_dir)
    print(f"Found {len(train_files)} train images, {len(test_files)} test images")
    if len(train_files) == 0:
        print("No train images found; abort.")
        return

    # Datasets
    train_ds = AgeDataset(train_files, transform=train_tf, is_train=True)

    # Validation subset for faster evaluation
    val_subset_size = min(2000, len(train_ds))  # at most 2000 images
    val_indices = np.random.choice(len(train_ds), val_subset_size, replace=False)
    val_ds = torch.utils.data.Subset(train_ds, val_indices)

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    test_ds = AgeDataset(test_files, transform=val_tf, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = AgeEstimator(n_bins=101, pretrained=args.pretrained).to(device)

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        print('Loaded checkpoint', args.resume)

    if args.mode == 'train':
        params = model.parameters()
        opt = torch.optim.AdamW(params, lr=args.lr)
        scheduler = None
        best_mae = 1e9
        for epoch in range(1, args.epochs+1):
            loss, train_mae = train_one_epoch(model, train_loader, opt, device, epoch, scheduler, lambda_reg=args.lambda_reg)
            print(f"Epoch {epoch} | loss: {loss:.4f} | train_mae(est): {train_mae:.3f}")
            # quick val
            files_val, preds_val, _ = evaluate(model, val_loader, device)
            # compute MAE on val (we parse ages back)
            ages_val = []
            for f in files_val:
                age = AgeDataset._parse_age_from_name(None, str(Path(train_dir) / f)) if False else None
            # save checkpoint
            ckpt = out_dir / 'checkpoints' / f'ckpt_ep{epoch}.pth'
            torch.save({'model': model.state_dict(), 'epoch': epoch}, ckpt)

    elif args.mode == 'eval':
        if args.resume and Path(args.resume).exists():
            ck = torch.load(args.resume, map_location=device)
            model.load_state_dict(ck['model'])
            print('Loaded checkpoint', args.resume)
        print('Running inference on test set...')
        files, preds, _ = evaluate(model, test_loader, device)
        # round predictions and clamp
        preds_int = [int(max(0, min(120, round(p)))) for p in preds]
        out_rows = [{'image': f, 'age': a} for f,a in zip(files, preds_int)]
        out_df = pd.DataFrame(out_rows)
        out_csv = out_dir / 'submissions' / 'task2_age_submission.csv'
        out_df.to_csv(out_csv, index=False)
        print('Saved submission to', out_csv)
        print('Inference and submission generation completed successfully.')

# %% CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train','eval'])
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_reg', type=float, default=1.0)
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    main(args)
