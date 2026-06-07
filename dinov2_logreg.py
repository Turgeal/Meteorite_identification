"""DINOv2 frozen features -> 5fold logistic regression OOF + test.

Outputs (under outputs/features/):
  dinov2_logreg_oof.csv        id, label, prob
  dinov2_logreg_stage2.csv     id, prob
  dinov2_logreg_stage1.csv     id, prob   (if available)

Why frozen + linear, not fine-tune: see Kumar et al. 2022
("Fine-Tuning can Distort Pretrained Features"). On OOD tasks
linear probe systematically beats full fine-tune because it preserves
the foundation model's robust pretrained features. Our task is
textbook OOD (museum train -> field stage2), so frozen + LR is the
right tool.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0,
                    help="LogReg inverse regularization (smaller=stronger reg)")
    args = ap.parse_args()

    fdir = Path(args.data_root) / args.features_dir

    train_feats = np.load(fdir / "dinov2_train.npy")
    train_ids_df = pd.read_csv(fdir / "dinov2_train_ids.csv")
    label_df = pd.read_csv(fdir / "train_ids.csv")
    train_ids_df = train_ids_df.merge(label_df, on="id", how="left")
    y = train_ids_df["label"].values.astype(int)

    print(f"train: {train_feats.shape}")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof = np.zeros(len(train_feats))
    fold_models = []
    for fold, (tr, va) in enumerate(skf.split(train_feats, y)):
        scaler = StandardScaler().fit(train_feats[tr])
        Xtr = scaler.transform(train_feats[tr])
        Xva = scaler.transform(train_feats[va])
        clf = LogisticRegression(C=args.C, max_iter=2000, n_jobs=-1)
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
        fdir / "dinov2_logreg_oof.csv", index=False
    )

    for split in ["stage2", "stage1"]:
        feats_path = fdir / f"dinov2_{split}.npy"
        ids_path = fdir / f"dinov2_{split}_ids.csv"
        if not feats_path.exists():
            continue
        feats = np.load(feats_path)
        ids = pd.read_csv(ids_path)["id"].tolist()
        probs = np.zeros((len(feats), len(fold_models)))
        for i, (scaler, clf) in enumerate(fold_models):
            probs[:, i] = clf.predict_proba(scaler.transform(feats))[:, 1]
        mean_prob = probs.mean(axis=1)
        std_prob = probs.std(axis=1)
        pd.DataFrame({"id": ids, "prob": mean_prob, "prob_std": std_prob}).to_csv(
            fdir / f"dinov2_logreg_{split}.csv", index=False
        )
        print(f"-> dinov2_logreg_{split}.csv  (n={len(ids)}, mean={mean_prob.mean():.4f})")


if __name__ == "__main__":
    main()
