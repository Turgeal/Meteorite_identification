"""Extract MAE ViT-L frozen features + train LogReg for v4 stacking.

Outputs (under <features_dir>/):
  mae_train.npy         (5280, 1024)
  mae_stage2.npy        (194, 1024)
  mae_logreg_stage2.csv  id, prob      (for stack_ensemble_v4.py to pick up)
  mae_logreg_oof.csv     id, label, prob
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).parent.resolve()
sys.path.insert(0, str(CODE))
from src.dataset import resolve_test_paths, resolve_train_paths


class ImageListDataset(Dataset):
    def __init__(self, paths, ids, transform):
        self.paths = paths
        self.ids = ids
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.ids[i]


def list_train(data_root, use_nobg=False):
    img_dir, csv = resolve_train_paths(data_root, use_nobg=use_nobg)
    df = pd.read_csv(csv)
    paths, ids, labels = [], [], []
    for _, row in df.iterrows():
        p = os.path.join(img_dir, row["id"])
        if not os.path.exists(p):
            for alt in (".jpg", ".jpeg"):
                if os.path.exists(os.path.splitext(p)[0] + alt):
                    p = os.path.splitext(p)[0] + alt
                    break
            else: continue
        paths.append(p)
        ids.append(row["id"])
        labels.append(int(row["label"]))
    return paths, ids, labels


def list_test(data_root, stage):
    img_dir, csv = resolve_test_paths(data_root, stage=stage)
    df = pd.read_csv(csv)
    paths, ids = [], []
    for _, row in df.iterrows():
        p = os.path.join(img_dir, row["id"])
        if not os.path.exists(p):
            for alt in (".jpg", ".jpeg"):
                if os.path.exists(os.path.splitext(p)[0] + alt):
                    p = os.path.splitext(p)[0] + alt
                    break
            else: continue
        paths.append(p)
        ids.append(row["id"])
    return paths, ids


def extract_mae(model, loader, device):
    feats, ids_out = [], []
    model.eval()
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model.forward_features(x)
                # MAE returns (B, N_patches+1, 1024) tensor, CLS is first token
                if isinstance(out, (list, tuple)):
                    out = out[0]
                cls = out[:, 0] if out.ndim == 3 else out
            feats.append(cls.float().cpu().numpy())
            ids_out.extend(idx if isinstance(idx, list) else list(idx))
    return np.concatenate(feats, axis=0), ids_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--mae_model", default="vit_large_patch16_224.mae")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.data_root) / args.features_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    import timm
    model = timm.create_model(args.mae_model, pretrained=True, num_classes=0)
    cfg = timm.data.resolve_model_data_config(model)
    tfm = timm.data.create_transform(**cfg, is_training=False)
    model = model.to(device).eval()
    print(f"Model: {args.mae_model}  feat_dim={model.num_features}")

    # --- Extract features ---
    train_paths, train_ids, train_labels = list_train(args.data_root, use_nobg=False)
    stage2_paths, stage2_ids = list_test(args.data_root, "stage2")

    for split_name, paths, ids in [
        ("train", train_paths, train_ids),
        ("stage2", stage2_paths, stage2_ids),
    ]:
        ds = ImageListDataset(paths, ids, tfm)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
        print(f"  extracting {split_name} ({len(ds)} imgs)...")
        feats, ids_out = extract_mae(model, dl, device)
        np.save(out_dir / f"mae_{split_name}.npy", feats)
        pd.DataFrame({"id": ids_out}).to_csv(
            out_dir / f"mae_{split_name}_ids.csv", index=False
        )
        print(f"    -> mae_{split_name}.npy  shape={feats.shape}")

    # --- Train LogReg ---
    train_feats = np.load(out_dir / "mae_train.npy")
    train_ids_df = pd.read_csv(out_dir / "mae_train_ids.csv")
    label_df = pd.read_csv(out_dir / "train_ids.csv")
    train_ids_df = train_ids_df.merge(label_df, on="id", how="left")
    y = train_ids_df["label"].values.astype(int)
    stage2_feats = np.load(out_dir / "mae_stage2.npy")
    stage2_ids_df = pd.read_csv(out_dir / "mae_stage2_ids.csv")

    print(f"\ntrain: {train_feats.shape}  stage2: {stage2_feats.shape}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(train_feats))
    fold_models = []
    for fold, (tr, va) in enumerate(skf.split(train_feats, y)):
        scaler = StandardScaler().fit(train_feats[tr])
        Xtr = scaler.transform(train_feats[tr])
        Xva = scaler.transform(train_feats[va])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(Xtr, y[tr])
        oof[va] = clf.predict_proba(Xva)[:, 1]
        f = f1_score(y[va], (oof[va] > 0.5).astype(int))
        a = roc_auc_score(y[va], oof[va])
        print(f"  fold {fold}: F1@0.5={f:.4f}  AUC={a:.4f}")
        fold_models.append((scaler, clf))

    overall_auc = roc_auc_score(y, oof)
    overall_f1 = f1_score(y, (oof > 0.5).astype(int))
    print(f"OOF AUC: {overall_auc:.4f}  F1@0.5: {overall_f1:.4f}")

    pd.DataFrame({"id": train_ids_df["id"], "label": y, "prob": oof}).to_csv(
        out_dir / "mae_logreg_oof.csv", index=False
    )

    # Predict on stage2
    probs = np.mean([
        clf.predict_proba(scaler.transform(stage2_feats))[:, 1]
        for scaler, clf in fold_models
    ], axis=0)
    pd.DataFrame({"id": stage2_ids_df["id"], "prob": probs}).to_csv(
        out_dir / "mae_logreg_stage2.csv", index=False
    )
    print(f"-> mae_logreg_stage2.csv  mean={probs.mean():.4f}")

    del model
    torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
