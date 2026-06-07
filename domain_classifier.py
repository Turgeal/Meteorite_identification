"""Domain classifier + weighted OOF + frozen holdout.

Trains a LightGBM binary classifier to distinguish train (label=0)
from stage2 (label=1) using DINOv2 features. Output:

  outputs/features/domain_weights.csv     id, label, stage2_likeness, weight, in_holdout
  outputs/features/domain_classifier.txt  short summary

The 'weight' column is clipped to [0.1, 1.0] for use as sample_weight
in OOF F1 evaluation. Top 20% most-stage2-like train samples are
flagged in_holdout=1: never feed those into stacking train, only use
them as a frozen evaluation set.

Why not directly drop museum-like images: they still carry true
positive labels. Down-weighting (Mehta et al.-style importance
weighting) preserves signal while shifting evaluation to the
deployment distribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--holdout_top_pct", type=float, default=0.20)
    ap.add_argument("--weight_floor", type=float, default=0.1)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    fdir = Path(args.data_root) / args.features_dir

    train_feats = np.load(fdir / "dinov2_train.npy")
    stage2_feats = np.load(fdir / "dinov2_stage2.npy")
    train_ids_df = pd.read_csv(fdir / "dinov2_train_ids.csv")
    label_df = pd.read_csv(fdir / "train_ids.csv")
    train_ids_df = train_ids_df.merge(label_df, on="id", how="left")

    print(f"train: {train_feats.shape}  stage2: {stage2_feats.shape}")

    X = np.concatenate([train_feats, stage2_feats], axis=0)
    y = np.concatenate(
        [np.zeros(len(train_feats), dtype=int),
         np.ones(len(stage2_feats), dtype=int)]
    )

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(X))
    aucs = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=31,
            max_depth=-1, min_child_samples=20, reg_lambda=1.0,
            n_jobs=-1, verbose=-1,
        )
        clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
                callbacks=[lgb.early_stopping(30, verbose=False)])
        oof_pred[va] = clf.predict_proba(X[va])[:, 1]
        auc = roc_auc_score(y[va], oof_pred[va])
        aucs.append(auc)
        print(f"  fold {fold}: AUC={auc:.4f}")

    overall_auc = roc_auc_score(y, oof_pred)
    print(f"OOF AUC: {overall_auc:.4f}  (avg fold AUC {np.mean(aucs):.4f})")

    if overall_auc > 0.95:
        print("  --> Domain shift confirmed: train and stage2 are easily separable.")
    elif overall_auc > 0.80:
        print("  --> Moderate domain shift.")
    else:
        print("  --> Mild domain shift; weighting may have limited effect.")

    train_likeness = oof_pred[: len(train_feats)]
    weight = np.clip(train_likeness, args.weight_floor, 1.0)

    n_holdout = int(len(train_feats) * args.holdout_top_pct)
    holdout_threshold = np.sort(train_likeness)[-n_holdout]
    in_holdout = (train_likeness >= holdout_threshold).astype(int)
    print(f"Holdout: top {args.holdout_top_pct*100:.0f}% = {in_holdout.sum()} samples "
          f"(stage2-likeness >= {holdout_threshold:.4f})")

    out = pd.DataFrame({
        "id": train_ids_df["id"].values,
        "label": train_ids_df["label"].values,
        "stage2_likeness": train_likeness,
        "weight": weight,
        "in_holdout": in_holdout,
    })
    out.to_csv(fdir / "domain_weights.csv", index=False)
    print(f"-> {fdir / 'domain_weights.csv'}")

    summary = {
        "overall_oof_auc": float(overall_auc),
        "fold_aucs": [float(a) for a in aucs],
        "n_train": int(len(train_feats)),
        "n_stage2": int(len(stage2_feats)),
        "holdout_size": int(in_holdout.sum()),
        "holdout_threshold": float(holdout_threshold),
        "weight_floor": args.weight_floor,
        "weight_mean": float(weight.mean()),
        "weight_median": float(np.median(weight)),
    }
    (fdir / "domain_classifier.txt").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
