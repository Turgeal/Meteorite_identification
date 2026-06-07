"""Train LogReg on CLIP image embeddings + predict on stage2.

Requires clip_image_train.npy and clip_image_stage2.npy from extract_hetero_features.py.
Outputs clip_logreg_oof.csv and clip_logreg_stage2.csv for v4 stacking.
"""
from __future__ import annotations

import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    fdir = Path(args.data_root) / args.features_dir

    train_feats = np.load(fdir / "clip_image_train.npy")
    label_df = pd.read_csv(fdir / "train_ids.csv")
    train_ids_df = pd.read_csv(fdir / "clip_zeroshot_train.csv")
    train_ids_df = train_ids_df.merge(label_df, on="id", how="left")
    y = train_ids_df["label"].values.astype(int)

    stage2_feats = np.load(fdir / "clip_image_stage2.npy")
    stage2_ids = pd.read_csv(fdir / "clip_zeroshot_stage2.csv")["id"]

    print(f"train: {train_feats.shape}  stage2: {stage2_feats.shape}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(train_feats))
    fold_models = []
    for fold, (tr, va) in enumerate(skf.split(train_feats, y)):
        scaler = StandardScaler().fit(train_feats[tr])
        Xtr, Xva = scaler.transform(train_feats[tr]), scaler.transform(train_feats[va])
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
        fdir / "clip_logreg_oof.csv", index=False
    )

    probs = np.mean([
        clf.predict_proba(scaler.transform(stage2_feats))[:, 1]
        for scaler, clf in fold_models
    ], axis=0)
    pd.DataFrame({"id": stage2_ids, "prob": probs}).to_csv(
        fdir / "clip_logreg_stage2.csv", index=False
    )
    print(f"-> clip_logreg_stage2.csv  mean={probs.mean():.4f}")

if __name__ == "__main__":
    main()
