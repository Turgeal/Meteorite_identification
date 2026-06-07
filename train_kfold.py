"""
K-Fold Cross-Validation Ensemble Training for meteorite classification.
Trains multiple models on different data splits and saves all for ensemble prediction.

Key features:
  - Focal Loss for hard example mining
  - CutMix / Mixup augmentation
  - Multi-architecture ensemble support (comma-separated model names)
  - Higher resolution (384) support
  - v2 augmentation pipeline with meteorite-specific transforms

Usage:
  python train_kfold.py --data_root . --output_dir outputs
  python train_kfold.py --model_name "convnext_small.fb_in22k_ft_in1k" --img_size 384 --use_focal_loss
  python train_kfold.py --model_name "efficientnet_b4.ra2_in1k" --img_size 384 --use_cutmix
"""
import os
import sys
import argparse
import time
import math
import json
import numpy as np
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.dataset import (
    StoneDataset, get_train_transforms, get_train_transforms_v2,
    get_val_transforms, cutmix_data, mixup_data, mixup_criterion
)
from src.model import create_classifier, save_model


# ─────────────────────────────────────────────
#  Focal Loss
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal Loss with class weights for handling class imbalance."""

    def __init__(self, class_weights, alpha=0.25, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        log_pt = F.cross_entropy(inputs, targets, reduction='none',
                                label_smoothing=self.label_smoothing, weight=self.class_weights)
        pt = torch.exp(-log_pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * log_pt
        return focal_loss.mean()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch,
                    use_cutmix=False, use_mixup=False,
                    mixup_alpha=0.4, cutmix_alpha=1.0, cutmix_prob=0.5):
    model.train()
    running_loss = 0.0
    clean_correct = 0
    clean_total = 0

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        r = np.random.rand()
        if use_cutmix and r < cutmix_prob:
            images, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        elif use_mixup and r < cutmix_prob:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            clean_total += labels.size(0)
            clean_correct += predicted.eq(labels).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*clean_correct/max(clean_total,1):.1f}%'
        })

    train_acc = 100. * clean_correct / max(clean_total, 1)
    return running_loss / len(loader), train_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validating"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(predicted.cpu().numpy().tolist())

    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    val_f1 = f1_score(all_labels, all_preds, average='binary')
    return val_loss, val_acc, val_f1, all_probs, all_labels


def find_best_threshold(probs, labels):
    probs_arr = np.array(probs)
    labels_arr = np.array(labels)
    best_threshold = 0.5
    best_f1 = 0.0
    for t_int in range(15, 86):
        t = t_int / 100.0
        preds = (probs_arr >= t).astype(int)
        f1 = f1_score(labels_arr, preds, average='binary')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    return best_threshold, best_f1


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        improved = (value > self.best_value + self.min_delta) if self.mode == 'max' \
                   else (value < self.best_value - self.min_delta)
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def _short_name(model_name):
    """Short model name for file naming (safe for filenames)."""
    return model_name.replace('.', '_').replace('/', '_')


def _is_classifier_param(name):
    """Check if a parameter name belongs to the classifier head."""
    return 'head' in name or 'classifier' in name


def train_one_fold(fold, train_indices, val_indices, base_dataset, args, device, fold_model_name=None):
    """Train a single fold with support for Focal Loss, CutMix, and v2 augmentation."""
    model_name = fold_model_name or args.model_name
    short_name = _short_name(model_name)
    print(f"\n{'='*60}")
    print(f"FOLD {fold + 1}/{args.n_folds} - Model: {model_name} - Res: {args.img_size}")
    print(f"{'='*60}")

    train_labels = [base_dataset.labels[i] for i in train_indices]
    val_labels = [base_dataset.labels[i] for i in val_indices]
    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)
    print(f"  Train: {len(train_indices)} samples (class 0: {train_dist[0]}, class 1: {train_dist[1]})")
    print(f"  Val:   {len(val_indices)} samples (class 0: {val_dist[0]}, class 1: {val_dist[1]})")

    # Transforms
    if args.use_v2_transforms:
        train_transforms = get_train_transforms_v2(args.img_size, level=args.aug_level)
    else:
        train_transforms = get_train_transforms(args.img_size, args.augmentation_level)
    val_transforms = get_val_transforms(args.img_size)

    train_ds = Subset(
        StoneDataset(root=args.data_root, split="train", transforms=train_transforms,
                     use_nobg=args.use_nobg),
        train_indices
    )
    val_ds = Subset(
        StoneDataset(root=args.data_root, split="train", transforms=val_transforms,
                     use_nobg=args.use_nobg),
        val_indices
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model
    pretrained_dir = os.path.join(args.data_root, 'pretrained_models') if args.pretrained else None
    model = create_classifier(
        num_classes=2, model_name=model_name, pretrained=args.pretrained,
        pretrained_dir=pretrained_dir, dropout=args.dropout, drop_path_rate=args.drop_path_rate
    ).to(device)

    # Class-weighted loss
    total_samples = len(train_labels)
    class_counts = [train_dist[0], train_dist[1]]
    class_weights = torch.tensor(
        [total_samples / (2 * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)

    if args.use_focal_loss:
        print(f"  Using Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
        criterion = FocalLoss(
            class_weights, alpha=args.focal_alpha, gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing
        )
    else:
        print(f"  Using weighted CE Loss (label_smoothing={args.label_smoothing})")
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    print(f"  Class weights: {class_weights.cpu().numpy()}")

    # Freeze backbone initially
    if args.freeze_epochs > 0:
        for name, param in model.named_parameters():
            if not _is_classifier_param(name):
                param.requires_grad = False

    # Optimizer with differential LR
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and _is_classifier_param(n)]
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not _is_classifier_param(n)]

    if backbone_params:
        param_groups = [
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
            {'params': head_params, 'lr': args.lr},
        ]
    else:
        param_groups = [{'params': head_params, 'lr': args.lr}]

    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stop_patience, min_delta=0.001, mode='max')

    # Training loop
    best_f1 = 0.0
    best_acc = 0.0
    best_threshold = 0.5
    fold_dir = os.path.join(args.output_dir, 'checkpoints', short_name)
    os.makedirs(fold_dir, exist_ok=True)
    best_path = os.path.join(fold_dir, f'best_model_fold{fold}.pth')

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            print(f"  [*] Unfreezing backbone at epoch {epoch}")
            for param in model.parameters():
                param.requires_grad = True
            backbone_params = [p for n, p in model.named_parameters() if not _is_classifier_param(n)]
            head_params = [p for n, p in model.named_parameters() if _is_classifier_param(n)]
            del optimizer  # free old optimizer state before creating new one
            torch.cuda.empty_cache()
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},
                {'params': head_params, 'lr': args.lr},
            ], weight_decay=args.weight_decay)
            remaining = (args.epochs - epoch + 1) * len(train_loader)
            warmup_r = min(len(train_loader), remaining)

            def lr_lambda_uf(step, _w=warmup_r, _t=remaining):
                if step < _w:
                    return float(step) / float(max(1, _w))
                progress = float(step - _w) / float(max(1, _t - _w))
                return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

            scheduler = LambdaLR(optimizer, lr_lambda_uf)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch,
            use_cutmix=args.use_cutmix, use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha,
            cutmix_prob=args.cutmix_prob
        )
        val_loss, val_acc, val_f1, all_probs, all_labels = validate(model, val_loader, criterion, device)
        ep_threshold, ep_f1 = find_best_threshold(all_probs, all_labels)

        print(f"  Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.1f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.1f}% F1={val_f1:.4f} | "
              f"Best-thresh F1={ep_f1:.4f}@{ep_threshold:.2f}")

        if ep_f1 > best_f1:
            best_f1 = ep_f1
            best_acc = val_acc
            best_threshold = ep_threshold
            save_model(model, optimizer, epoch, val_loss, best_path,
                       best_acc=best_acc, best_f1=best_f1, best_threshold=best_threshold,
                       model_name=model_name, best_val_f1_fixed=val_f1)
            print(f"  [*] Saved fold {fold} best (Opt-F1={best_f1:.4f}, "
                  f"Threshold={best_threshold:.2f}, Val-F1@0.5={val_f1:.4f})")

        if early_stopping(ep_f1):
            print(f"  [!] Early stopping at epoch {epoch}")
            break

    print(f"\nFold {fold + 1} Results: Best Opt-F1={best_f1:.4f}, "
          f"Acc={best_acc:.1f}%, Threshold={best_threshold:.2f}")
    return best_path, best_f1, best_threshold


def main(args):
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse model names (multi-architecture: "efficientnet_b0.ra_in1k,resnet50.a1_in1k")
    model_names = [m.strip() for m in args.model_name.split(',')]
    print(f"Model(s): {model_names}")

    # Print config
    print(f"\nConfig: img_size={args.img_size}, epochs={args.epochs}, bs={args.batch_size}")
    print(f"  Focal Loss: {args.use_focal_loss}, CutMix: {args.use_cutmix}, Mixup: {args.use_mixup}")
    print(f"  Augmentation: {'v2_' + args.aug_level if args.use_v2_transforms else args.augmentation_level}")

    print("\nLoading full dataset...")
    if args.use_nobg:
        print("  -> use_nobg=True (train_images_nobg/)")
    base_dataset = StoneDataset(root=args.data_root, split="train", transforms=None,
                                use_nobg=args.use_nobg)
    labels = np.array(base_dataset.labels)
    indices = np.arange(len(base_dataset))
    print(f"Total samples: {len(base_dataset)}, Class distribution: {Counter(base_dataset.labels)}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    all_splits = list(skf.split(indices, labels))

    fold_ids = [int(f.strip()) for f in args.fold.split(',')] if args.fold else list(range(args.n_folds))

    fold_results = []
    start_time = time.time()
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')

    for fold in fold_ids:
        train_idx, val_idx = all_splits[fold]
        fold_model = model_names[fold % len(model_names)]
        fold_short = _short_name(fold_model)

        # Check existing checkpoint with per-model subdirectory
        existing_path = os.path.join(checkpoint_dir, fold_short, f'best_model_fold{fold}.pth')
        if args.skip_existing and os.path.exists(existing_path):
            ckpt = torch.load(existing_path, map_location='cpu', weights_only=False)
            ckpt_model = ckpt.get('model_name', model_names[0])
            # Only skip if the checkpoint's model matches the one we want to train
            if ckpt_model == fold_model:
                print(f"\n>>> Fold {fold} already trained ({ckpt_model}), skipping.")
                fold_results.append({
                    'fold': fold, 'model_path': existing_path,
                    'model_name': ckpt_model,
                    'best_f1': ckpt.get('best_f1', 0),
                    'best_threshold': ckpt.get('best_threshold', 0.5)
                })
                continue
            else:
                print(f"\n>>> Fold {fold} has {ckpt_model}, retraining with {fold_model} (overwriting).")

        best_path, best_f1, best_threshold = train_one_fold(
            fold, train_idx.tolist(), val_idx.tolist(), base_dataset, args, device, fold_model
        )
        fold_results.append({
            'fold': fold, 'model_path': best_path, 'model_name': fold_model,
            'best_f1': best_f1, 'best_threshold': best_threshold
        })
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE ({len(fold_results)} folds, {total_time/60:.1f} min)")
    print(f"{'='*60}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: F1={r['best_f1']:.4f}, Model={r.get('model_name','?')}")
    print(f"  Mean F1: {np.mean([r['best_f1'] for r in fold_results]):.4f}")

    info_path = os.path.join(args.output_dir, 'kfold_info.json')
    existing = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            existing = json.load(f)
    all_results = {r['fold']: r for r in existing.get('folds', [])}
    all_results.update({r['fold']: r for r in fold_results})
    merged = sorted(all_results.values(), key=lambda x: x['fold'])
    with open(info_path, 'w') as f:
        json.dump({'n_folds': args.n_folds, 'folds': merged, 'config': {
            'model_name': args.model_name,
            'img_size': args.img_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'use_focal_loss': args.use_focal_loss,
            'use_cutmix': args.use_cutmix,
            'use_mixup': args.use_mixup,
            'per_model_dirs': {
                _short_name(m): os.path.join('checkpoints', _short_name(m))
                for m in model_names
            }
        }}, f, indent=2)
    print(f"Saved to: {info_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='K-Fold Ensemble Training')

    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='./outputs')

    parser.add_argument('--model_name', type=str,
                        default='convnext_small.fb_in22k_ft_in1k',
                        help='Model name(s), comma-separated for multi-arch ensemble')
    parser.add_argument('--pretrained', type=bool, default=True)

    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--fold', type=str, default=None, help='Specific fold(s): --fold 0 or --fold 0,1,2')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--skip_existing', action='store_true', default=True)
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false')

    parser.add_argument('--epochs', type=int, default=35)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Use 16 for 384px, 32 for 224px')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone_lr_scale', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--img_size', type=int, default=384)

    parser.add_argument('--freeze_epochs', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--drop_path_rate', type=float, default=0.2)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # Focal Loss
    parser.add_argument('--use_focal_loss', action='store_true', default=False)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)

    # CutMix / Mixup
    parser.add_argument('--use_cutmix', action='store_true', default=False)
    parser.add_argument('--use_mixup', action='store_true', default=False)
    parser.add_argument('--mixup_alpha', type=float, default=0.4)
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--cutmix_prob', type=float, default=0.5)

    # Augmentation
    parser.add_argument('--augmentation_level', type=str, default='heavy',
                        choices=['light', 'medium', 'heavy'])
    parser.add_argument('--use_v2_transforms', action='store_true', default=True,
                        help='Use v2 augmentation pipeline (recommended)')
    parser.add_argument('--aug_level', type=str, default='heavy',
                        choices=['medium', 'heavy', 'extreme'])

    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_nobg', action='store_true', default=False,
                        help='Use background-removed train images at train_images_nobg/')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
