"""
BUS-BRA Augmentation Benchmark Pipeline v2
============================================
Configured for actual BUS-BRA structure:
  - CSV: bus_data.csv  (cols: ID, BIRADS, ...)
  - Images: Images/ (flat folder, no subfolders by category)
  - No pre-defined folds → stratified 5-fold CV generated here

Usage:
    python busbra_pipeline_v2.py --data_root /path/to/BUSBRA --aug baseline
    python busbra_pipeline_v2.py --data_root /path/to/BUSBRA --aug geometric
    python busbra_pipeline_v2.py --data_root /path/to/BUSBRA --aug intensity
    python busbra_pipeline_v2.py --data_root /path/to/BUSBRA --aug ultrasound
    python busbra_pipeline_v2.py --data_root /path/to/BUSBRA --aug modern
    python busbra_pipeline_v2.py --data_root /path/to/BUSBRA --aug combined
"""

import os, argparse, random
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── reproducibility ──────────────────────────────────────────────────────
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_everything()

# ── label map ────────────────────────────────────────────────────────────
BIRADS_TO_IDX = {2: 0, 3: 1, 4: 2, 5: 3}
IDX_TO_BIRADS = {0: 2, 1: 3, 2: 4, 3: 5}
NUM_CLASSES   = 4
IMG_SIZE      = 224


# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def build_dataframe(data_root: str) -> pd.DataFrame:
    data_root = Path(data_root)
    csv_path  = data_root / 'bus_data.csv'
    df_csv    = pd.read_csv(csv_path)

    # Image folder — capital I
    img_dir = data_root / 'Images'
    if not img_dir.exists():
        # fallback
        img_dir = data_root / 'Images_clean'
    print(f"[INFO] Using image folder: {img_dir}")

    rows = []
    missing = 0
    for _, row in df_csv.iterrows():
        cat = int(row['BIRADS'])
        if cat not in BIRADS_TO_IDX:
            continue
        img_id   = str(row['ID'])          # e.g. 'bus_0001-l'
        # Try common extensions
        img_path = None
        for ext in ['.png', '.jpg', '.bmp', '.PNG', '.JPG']:
            candidate = img_dir / f"{img_id}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            missing += 1
            continue
        rows.append({
            'path':    str(img_path),
            'label':   BIRADS_TO_IDX[cat],
            'birads':  cat,
            'device':  row.get('Device', 'unknown'),
            'pathology': row.get('Pathology', 'unknown'),
        })

    df = pd.DataFrame(rows)
    if missing > 0:
        print(f"[WARN] {missing} images not found on disk — skipped")

    # Generate stratified 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    df['fold'] = -1
    for fold_idx, (_, val_idx) in enumerate(
            skf.split(df.index, df['label']), start=1):
        df.loc[df.index[val_idx], 'fold'] = fold_idx

    # Summary
    print(f"\n[INFO] Dataset summary:")
    print(f"  Total images: {len(df)}")
    for cat in [2, 3, 4, 5]:
        n = (df['birads'] == cat).sum()
        print(f"  BI-RADS {cat}: {n:4d}  ({100*n/len(df):.1f}%)")
    print(f"\n  Class imbalance note:")
    print(f"  BI-RADS 5 has only {(df['birads']==5).sum()} samples — "
          f"watch F1(cat5) carefully")
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. AUGMENTATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════

def get_augmentation(aug_name: str) -> A.Compose:
    normalize = [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]

    if aug_name == 'baseline':
        return A.Compose(normalize)

    elif aug_name == 'geometric':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=0, p=0.4),
            A.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])

    elif aug_name == 'intensity':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])

    elif aug_name == 'ultrasound':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.GaussNoise(var_limit=(10, 50), p=0.5),      # speckle proxy
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),      # probe smoothing
            A.RandomShadow(num_shadows_lower=1,
                           num_shadows_upper=2,
                           shadow_dimension=4, p=0.3),      # acoustic shadow
            A.CoarseDropout(max_holes=4, max_height=20,
                            max_width=20, p=0.2),
            A.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])

    elif aug_name == 'modern':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.Rotate(limit=20, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(p=1.0),
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(p=1.0),
                A.RandomGamma(p=1.0),
                A.Sharpen(p=1.0),
            ], p=0.5),
            A.CoarseDropout(max_holes=8, max_height=16,
                            max_width=16, fill_value=0, p=0.3),
            A.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])

    elif aug_name == 'combined':
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                               rotate_limit=0, p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussNoise(var_limit=(10, 40), p=0.4),
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1,
                           shadow_dimension=4, p=0.2),
            A.CoarseDropout(max_holes=4, max_height=16,
                            max_width=16, fill_value=0, p=0.2),
            A.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225]),
            ToTensorV2()
        ])

    else:
        raise ValueError(f"Unknown aug: {aug_name}")


def get_val_transform() -> A.Compose:
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]),
        ToTensorV2()
    ])


# ═══════════════════════════════════════════════════════════════════════
# 3. DATASET
# ═══════════════════════════════════════════════════════════════════════

class BUSBRADataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row['path']).convert('RGB'))
        if self.transform:
            img = self.transform(image=img)['image']
        return img, torch.tensor(row['label'], dtype=torch.long)


# ═══════════════════════════════════════════════════════════════════════
# 4. LOSS — Earth Mover Distance (ordinal-aware)
# ═══════════════════════════════════════════════════════════════════════

class OrdinalEMDLoss(nn.Module):
    def forward(self, logits, targets):
        probs    = torch.softmax(logits, dim=1)
        cdf_pred = torch.cumsum(probs, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)
        cdf_true = torch.cumsum(targets_oh, dim=1)
        return torch.mean(torch.abs(cdf_pred - cdf_true))


# ═══════════════════════════════════════════════════════════════════════
# 5. MODEL
# ═══════════════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    model = models.efficientnet_b3(weights='IMAGENET1K_V1')
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model


# ═══════════════════════════════════════════════════════════════════════
# 6. METRICS
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(preds, labels):
    preds  = np.array(preds)
    labels = np.array(labels)
    return {
        'accuracy':  float((preds == labels).mean()),
        'macro_f1':  float(f1_score(labels, preds, average='macro',
                                     zero_division=0)),
        'mae':       float(np.mean(np.abs(preds - labels))),
        'f1_birads2': float(f1_score(labels, preds, labels=[0],
                                      average='macro', zero_division=0)),
        'f1_birads3': float(f1_score(labels, preds, labels=[1],
                                      average='macro', zero_division=0)),
        'f1_birads4': float(f1_score(labels, preds, labels=[2],
                                      average='macro', zero_division=0)),
        'f1_birads5': float(f1_score(labels, preds, labels=[3],
                                      average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(labels, preds, labels=[0,1,2,3])
    }


# ═══════════════════════════════════════════════════════════════════════
# 7. TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * len(imgs)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    m = compute_metrics(all_preds, all_labels)
    m['loss'] = total_loss / len(loader.dataset)
    return m


# ═══════════════════════════════════════════════════════════════════════
# 8. CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def run_cv(df: pd.DataFrame, aug_name: str, args) -> pd.DataFrame:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Device: {device} | Aug: {aug_name}")

    fold_results = []

    for fold_num in sorted(df['fold'].unique()):
        print(f"\n── Fold {fold_num}/5 ──────────────────────────────")
        seed_everything()

        train_df = df[df['fold'] != fold_num]
        val_df   = df[df['fold'] == fold_num]

        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")
        print(f"  Val BI-RADS dist: "
              + " | ".join(f"{IDX_TO_BIRADS[i]}:{(val_df['label']==i).sum()}"
                           for i in range(4)))

        train_ds = BUSBRADataset(train_df, get_augmentation(aug_name))
        val_ds   = BUSBRADataset(val_df,   get_val_transform())

        # Weighted sampler — oversample minority classes (esp. BI-RADS 5)
        class_counts = train_df['label'].value_counts().sort_index()
        weights_per_class = 1.0 / class_counts.values
        sample_weights = train_df['label'].map(
            dict(enumerate(weights_per_class))).values
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float),
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   sampler=sampler, num_workers=0,
                                   pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                   shuffle=False, num_workers=0,
                                   pin_memory=False)

        model     = build_model().to(device)
        criterion = OrdinalEMDLoss()
        optimizer = optim.AdamW(model.parameters(),
                                lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)

        best_f1, best_metrics, patience_counter = 0, {}, 0

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader,
                                      optimizer, criterion, device)
            val_m      = eval_epoch(model, val_loader, criterion, device)
            scheduler.step()

            if val_m['macro_f1'] > best_f1:
                best_f1, best_metrics = val_m['macro_f1'], val_m.copy()
                patience_counter = 0
                torch.save(model.state_dict(),
                           f"best_fold{fold_num}_{aug_name}.pt")
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"  ep{epoch:3d} | "
                      f"tr_loss={train_loss:.4f} | "
                      f"val_loss={val_m['loss']:.4f} | "
                      f"mF1={val_m['macro_f1']:.4f} | "
                      f"F1(3)={val_m['f1_birads3']:.4f} | "
                      f"MAE={val_m['mae']:.4f}")

            if patience_counter >= args.patience:
                print(f"  ↳ Early stop at epoch {epoch}")
                break

        best_metrics.update({'fold': fold_num, 'aug': aug_name})
        fold_results.append(best_metrics)

        cm = best_metrics['confusion_matrix']
        print(f"\n  Best macro-F1: {best_metrics['macro_f1']:.4f} | "
              f"MAE: {best_metrics['mae']:.4f}")
        print(f"  Per-category F1:")
        for i, cat in enumerate([2,3,4,5]):
            key = f'f1_birads{cat}'
            print(f"    BI-RADS {cat}: {best_metrics[key]:.4f}")
        print(f"  Confusion matrix (rows=true, cols=pred, "
              f"labels=2,3,4,5):")
        print(f"  {cm}")

    # ── aggregate results ──
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'confusion_matrix'}
        for r in fold_results
    ])

    print(f"\n{'═'*58}")
    print(f"FINAL RESULTS  |  Augmentation: {aug_name}")
    print(f"{'─'*58}")
    for metric, label in [
        ('accuracy',  'Accuracy '),
        ('macro_f1',  'Macro-F1 '),
        ('mae',       'MAE      '),
        ('f1_birads2','F1(cat 2)'),
        ('f1_birads3','F1(cat 3)  ← key metric'),
        ('f1_birads4','F1(cat 4)'),
        ('f1_birads5','F1(cat 5)'),
    ]:
        vals = results_df[metric]
        print(f"  {label}: {vals.mean():.4f} ± {vals.std():.4f}")
    print(f"{'═'*58}")

    out = f"results_{aug_name}.csv"
    results_df.to_csv(out, index=False)
    print(f"[INFO] Saved → {out}")
    return results_df


# ═══════════════════════════════════════════════════════════════════════
# 9. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root',  type=str, required=True)
    p.add_argument('--aug',        type=str, default='baseline',
                   choices=['baseline','geometric','intensity',
                            'ultrasound','modern','combined'])
    p.add_argument('--epochs',     type=int, default=50)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--patience',   type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("="*58)
    print("BUS-BRA AUGMENTATION BENCHMARK v2")
    print(f"  aug={args.aug} | epochs={args.epochs} | "
          f"bs={args.batch_size} | lr={args.lr}")
    print("="*58)
    df = build_dataframe(args.data_root)
    run_cv(df, args.aug, args)
