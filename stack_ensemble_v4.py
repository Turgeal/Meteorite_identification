"""Heterogeneous Stacking v4.

Combines features from multiple orthogonal sources and meta-learns a
stage2-aware ranker. Required inputs (paths configurable via CLI):

  Base ensemble (fine-tuned ConvNeXt + Swin on original train images):
    outputs/stacking_v3/oof_predictions_v3.csv
    outputs/stacking_v3/test_predictions_v3.csv

  nobg ensemble (same backbones, retrained on background-removed train):
    outputs_nobg/stacking_v3/oof_predictions_v3.csv     (optional)
    outputs_nobg/stacking_v3/test_predictions_v3.csv    (optional)

  DINOv2 frozen heads (no fine-tune; preserves OOD-robust features):
    outputs/features/dinov2_logreg_oof.csv         id,label,prob
    outputs/features/dinov2_logreg_stage2.csv      id,prob[,prob_std]
    outputs/features/dinov2_knn_oof.csv            id,label,prob
    outputs/features/dinov2_knn_stage2.csv         id,prob

  CLIP zero-shot:
    outputs/features/clip_zeroshot_train.csv       id,score
    outputs/features/clip_zeroshot_stage2.csv      id,score

  Domain weights + holdout flag:
    outputs/features/domain_weights.csv  id,label,stage2_likeness,weight,in_holdout

Evaluation:
  - 5fold CV F1 with sample_weight = domain weight (weighted OOF F1)
  - Frozen holdout F1 on top-20% stage2-like train images
    (excluded from LightGBM training for unbiased final selection)

Outputs (under <stack_dir>/):
  stacking_v4_oof.csv       id,label,prob,prob_std
  stacking_v4_test.csv      id,prob,prob_std
  stacking_v4_features.txt  feature names + final importances
  topk_submissions/submission_top{K}.csv
  blend_submissions/submission_b{w}_top{K}.csv  (blend with base ensemble)
  summary.json              weighted_oof_f1, holdout_f1, ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def _read_oof(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "id" not in df.columns:
        return None
    return df


def _read_test(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _v3_oof_to_per_arch_means(df):
    cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    keep = ["id"]
    if "label" in df.columns:
        keep.append("label")
    return df[keep + cols].copy()


def _rank01(s):
    return pd.Series(s).rank(method="average").values / max(len(s), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--stage", default="stage2")
    ap.add_argument("--base_stack_dir", default="outputs/stage2/stacking_v3")
    ap.add_argument("--nobg_stack_dir", default="outputs_nobg/stage2/stacking_v3")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--ensemble_csv", default="outputs/stage2/ensemble_probs.csv",
                    help="Base ensemble probs (for blended submissions)")
    ap.add_argument("--out_dir", default="outputs/stage2/stacking_v4")
    ap.add_argument("--ks", default="115,120,125,130,135,140,145,150")
    ap.add_argument("--blend_weights", default="0.4,0.5,0.6,0.7")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_oof = _v3_oof_to_per_arch_means(
        pd.read_csv(root / args.base_stack_dir / "oof_predictions_v3.csv")
    ).rename(columns=lambda c: f"base_{c}" if c not in ("id", "label") else c)
    base_test = _v3_oof_to_per_arch_means(
        pd.read_csv(root / args.base_stack_dir / "test_predictions_v3.csv")
    ).rename(columns=lambda c: f"base_{c}" if c not in ("id", "label") else c)

    nobg_oof_df = _read_oof(root / args.nobg_stack_dir / "oof_predictions_v3.csv")
    nobg_test_df = _read_test(root / args.nobg_stack_dir / "test_predictions_v3.csv")
    have_nobg = nobg_oof_df is not None and nobg_test_df is not None
    if have_nobg:
        nobg_oof = _v3_oof_to_per_arch_means(nobg_oof_df).rename(
            columns=lambda c: f"nobg_{c}" if c not in ("id", "label") else c
        )
        nobg_test = _v3_oof_to_per_arch_means(nobg_test_df).rename(
            columns=lambda c: f"nobg_{c}" if c not in ("id", "label") else c
        )
    else:
        print("WARN: nobg stacking dir missing; running base-only stacking v4")

    fdir = root / args.features_dir
    dv_lr_oof = pd.read_csv(fdir / "dinov2_logreg_oof.csv")[["id", "prob"]].rename(
        columns={"prob": "dinov2_lr"}
    )
    dv_lr_te = pd.read_csv(fdir / f"dinov2_logreg_{args.stage}.csv")[["id", "prob"]].rename(
        columns={"prob": "dinov2_lr"}
    )
    dv_knn_oof = pd.read_csv(fdir / "dinov2_knn_oof.csv")[["id", "prob"]].rename(
        columns={"prob": "dinov2_knn"}
    )
    dv_knn_te = pd.read_csv(fdir / f"dinov2_knn_{args.stage}.csv")[["id", "prob"]].rename(
        columns={"prob": "dinov2_knn"}
    )
    clip_tr = pd.read_csv(fdir / "clip_zeroshot_train.csv").rename(columns={"score": "clip"})
    clip_te = pd.read_csv(fdir / f"clip_zeroshot_{args.stage}.csv").rename(columns={"score": "clip"})
    weights = pd.read_csv(fdir / "domain_weights.csv")[
        ["id", "weight", "in_holdout"]
    ]

    labels = pd.read_csv(root / "train_labels.csv")[["id", "label"]]
    labels["id"] = labels["id"].astype(str)
    if not labels["id"].iloc[0].endswith(".jpg"):
        labels["id"] = labels["id"] + ".jpg"

    train = base_oof.copy()
    if "label" not in train.columns:
        train = train.merge(labels, on="id", how="left")
    if have_nobg:
        nobg_cols = [c for c in nobg_oof.columns if c not in ("label",)]
        train = train.merge(nobg_oof[nobg_cols], on="id", how="inner")
    train = train.merge(dv_lr_oof, on="id", how="inner")
    train = train.merge(dv_knn_oof, on="id", how="inner")
    train = train.merge(clip_tr, on="id", how="inner")
    train = train.merge(weights, on="id", how="left")
    train["weight"] = train["weight"].fillna(1.0)
    train["in_holdout"] = train["in_holdout"].fillna(0).astype(int)
    print(f"Train table: {train.shape}  holdout rows: {int(train['in_holdout'].sum())}")

    test = base_test.copy()
    if have_nobg:
        test = test.merge(nobg_test, on="id", how="inner")
    test = test.merge(dv_lr_te, on="id", how="inner")
    test = test.merge(dv_knn_te, on="id", how="inner")
    test = test.merge(clip_te, on="id", how="inner")
    print(f"Test table: {test.shape}")

    feature_cols = [
        c for c in train.columns
        if c not in ("id", "label", "weight", "in_holdout")
    ]
    test_feature_cols = [c for c in test.columns if c != "id"]
    feature_cols = [c for c in feature_cols if c in test_feature_cols]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    train_dev = train[train["in_holdout"] == 0].reset_index(drop=True)
    train_hold = train[train["in_holdout"] == 1].reset_index(drop=True)
    print(f"Dev: {len(train_dev)}  Holdout: {len(train_hold)}")

    X_dev = train_dev[feature_cols].values
    y_dev = train_dev["label"].values.astype(int)
    w_dev = train_dev["weight"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_dev = np.zeros(len(X_dev))
    importances = np.zeros(len(feature_cols))
    fold_models = []
    for fold, (tr, va) in enumerate(skf.split(X_dev, y_dev)):
        clf = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.02, num_leaves=15,
            max_depth=4, min_child_samples=20, reg_lambda=2.0,
            subsample=0.9, colsample_bytree=0.9, n_jobs=-1, verbose=-1,
        )
        clf.fit(
            X_dev[tr], y_dev[tr], sample_weight=w_dev[tr],
            eval_set=[(X_dev[va], y_dev[va])],
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        oof_dev[va] = clf.predict_proba(X_dev[va])[:, 1]
        importances += clf.feature_importances_
        fold_models.append(clf)
        f_unw = f1_score(y_dev[va], (oof_dev[va] > 0.5).astype(int))
        f_w = f1_score(y_dev[va], (oof_dev[va] > 0.5).astype(int),
                       sample_weight=w_dev[va])
        print(f"  fold {fold}: F1@0.5={f_unw:.4f}  weighted_F1@0.5={f_w:.4f}")

    importances /= len(fold_models)
    imp_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    (out_dir / "stacking_v4_features.txt").write_text(
        "\n".join(f"{n}\t{i:.1f}" for n, i in imp_pairs), encoding="utf-8"
    )
    print("Top features:")
    for n, i in imp_pairs[:10]:
        print(f"  {n:30s}  {i:.1f}")

    weighted_oof_f1 = f1_score(
        y_dev, (oof_dev > 0.5).astype(int), sample_weight=w_dev
    )
    print(f"weighted OOF F1@0.5 (dev): {weighted_oof_f1:.4f}")

    if len(train_hold):
        X_hold = train_hold[feature_cols].values
        y_hold = train_hold["label"].values.astype(int)
        hold_probs = np.mean(
            [m.predict_proba(X_hold)[:, 1] for m in fold_models], axis=0
        )
        n_pos = int(y_hold.sum())
        thr_grid = np.linspace(0.1, 0.9, 81)
        best_thr = max(thr_grid, key=lambda t: f1_score(y_hold, (hold_probs > t).astype(int)))
        holdout_f1_thr = f1_score(y_hold, (hold_probs > best_thr).astype(int))
        order = np.argsort(-hold_probs)
        topk_pred = np.zeros(len(y_hold), dtype=int)
        topk_pred[order[:n_pos]] = 1
        holdout_f1_topk = f1_score(y_hold, topk_pred)
        print(f"holdout F1: thr={best_thr:.3f} -> {holdout_f1_thr:.4f}  "
              f"top-{n_pos} -> {holdout_f1_topk:.4f}")
    else:
        holdout_f1_thr = holdout_f1_topk = float("nan")

    X_test = test[feature_cols].values
    test_probs = np.stack(
        [m.predict_proba(X_test)[:, 1] for m in fold_models], axis=1
    )
    test_mean = test_probs.mean(axis=1)
    test_std = test_probs.std(axis=1)

    pd.DataFrame({
        "id": train_dev["id"], "label": y_dev, "prob": oof_dev,
        "weight": w_dev,
    }).to_csv(out_dir / "stacking_v4_oof.csv", index=False)
    pd.DataFrame({"id": test["id"], "prob": test_mean, "prob_std": test_std}).to_csv(
        out_dir / "stacking_v4_test.csv", index=False
    )

    template = pd.read_csv(root / f"sample_submission_{args.stage}.csv")
    ks = [int(k) for k in args.ks.split(",")]
    sub_dir = out_dir / "topk_submissions"
    sub_dir.mkdir(exist_ok=True)
    sorted_test = test[["id"]].assign(score=test_mean).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)
    for k in ks:
        pos = set(sorted_test["id"].head(k))
        s = template.copy()
        s["label"] = s["id"].isin(pos).astype(int)
        s.to_csv(sub_dir / f"submission_top{k}.csv", index=False)

    blend_dir = out_dir / "blend_submissions"
    blend_dir.mkdir(exist_ok=True)
    ens_path = root / args.ensemble_csv
    if ens_path.exists():
        ens = pd.read_csv(ens_path)[["id", "prob"]].rename(columns={"prob": "ens_prob"})
        bdf = test[["id"]].assign(stack_prob=test_mean).merge(ens, on="id", how="inner")
        bdf["stack_rank"] = _rank01(bdf["stack_prob"].values)
        bdf["ens_rank"] = _rank01(bdf["ens_prob"].values)
        weights = [float(w) for w in args.blend_weights.split(",")]
        for w in weights:
            score = w * bdf["stack_rank"] + (1 - w) * bdf["ens_rank"]
            bdf["bscore"] = score
            sb = bdf.sort_values("bscore", ascending=False).reset_index(drop=True)
            for k in ks:
                pos = set(sb["id"].head(k))
                s = template.copy()
                s["label"] = s["id"].isin(pos).astype(int)
                s.to_csv(blend_dir / f"submission_b{w:.1f}_top{k}.csv", index=False)

    summary = {
        "weighted_oof_f1_dev_thr0.5": float(weighted_oof_f1),
        "holdout_f1_best_thr": float(holdout_f1_thr),
        "holdout_f1_topk": float(holdout_f1_topk),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "have_nobg": bool(have_nobg),
        "ks": ks,
        "blend_weights": [float(w) for w in args.blend_weights.split(",")],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
