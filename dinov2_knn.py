"""DINOv2 kNN with domain-weighted train index.

For each test image, find k nearest neighbors in DINOv2 feature space
among the training set, weighted by stage2_likeness so that
museum-like training images contribute less. Output a probability
that mirrors the weighted positive ratio of the neighbors.

Outputs (under outputs/features/):
  dinov2_knn_oof.csv         id, label, prob
  dinov2_knn_stage2.csv      id, prob
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize


def weighted_knn_prob(query, index_feats, index_labels, index_weights, k):
    sims = query @ index_feats.T
    top_idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
    rows = np.arange(len(query))[:, None]
    top_sims = sims[rows, top_idx]
    top_labels = index_labels[top_idx]
    top_w = index_weights[top_idx] * np.maximum(top_sims, 0)
    num = (top_labels * top_w).sum(axis=1)
    den = top_w.sum(axis=1) + 1e-8
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    fdir = Path(args.data_root) / args.features_dir

    train_feats = np.load(fdir / "dinov2_train.npy")
    train_feats = normalize(train_feats, axis=1)

    label_df = pd.read_csv(fdir / "train_ids.csv")
    weight_df = pd.read_csv(fdir / "domain_weights.csv")
    train_ids_df = pd.read_csv(fdir / "dinov2_train_ids.csv")
    df = train_ids_df.merge(label_df, on="id", how="left").merge(
        weight_df[["id", "weight"]], on="id", how="left"
    )
    y = df["label"].values.astype(int)
    w = df["weight"].fillna(1.0).values

    print(f"train: {train_feats.shape}  k={args.k}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(train_feats))
    for fold, (tr, va) in enumerate(skf.split(train_feats, y)):
        oof[va] = weighted_knn_prob(
            train_feats[va], train_feats[tr], y[tr], w[tr], args.k
        )
        f = f1_score(y[va], (oof[va] > 0.5).astype(int))
        a = roc_auc_score(y[va], oof[va])
        print(f"  fold {fold}: F1@0.5={f:.4f}  AUC={a:.4f}")

    print(f"OOF AUC: {roc_auc_score(y, oof):.4f}  F1@0.5: {f1_score(y, (oof > 0.5).astype(int)):.4f}")

    pd.DataFrame({"id": df["id"], "label": y, "prob": oof}).to_csv(
        fdir / "dinov2_knn_oof.csv", index=False
    )

    for split in ["stage2", "stage1"]:
        feats_path = fdir / f"dinov2_{split}.npy"
        ids_path = fdir / f"dinov2_{split}_ids.csv"
        if not feats_path.exists():
            continue
        feats = normalize(np.load(feats_path), axis=1)
        ids = pd.read_csv(ids_path)["id"].tolist()
        probs = weighted_knn_prob(feats, train_feats, y, w, args.k)
        pd.DataFrame({"id": ids, "prob": probs}).to_csv(
            fdir / f"dinov2_knn_{split}.csv", index=False
        )
        print(f"-> dinov2_knn_{split}.csv  (mean={probs.mean():.4f})")


if __name__ == "__main__":
    main()
