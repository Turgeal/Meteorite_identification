"""Heterogeneous Stacking v5 — MAE + CLIP-LogReg + 26-dim image features.

Adds to v4:
  - mae_logreg: LogReg on MAE ViT-L frozen features
  - clip_logreg: LogReg on CLIP image embeddings  
  - 26-dim hand-crafted image features (from stack_ensemble_v3 extract_features)

Usage:
  python stack_ensemble_v5.py --data_root . --stage stage2 --features_dir outputs \
      --stack_dir outputs/stage2/stacking_v3 --ks "84,86,88,90,92,94,96,98,100,102,104,106,108,110"
"""
from __future__ import annotations

import argparse, json, os
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--stage", default="stage2")
    ap.add_argument("--stack_dir", default="outputs/stage2/stacking_v3")
    ap.add_argument("--features_dir", default="outputs/features")
    ap.add_argument("--out_dir", default="outputs/stage2/stacking_v5")
    ap.add_argument("--ks", default="84,86,88,90,92,94,96,98,100,102,104,106,108,110")
    ap.add_argument("--ensemble_csv", default="outputs/stage2/ensemble_probs.csv")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fdir = root / args.features_dir
    sdir = root / args.stack_dir

    # ── 1. Base OOF (ConvNeXt) ──
    base_oof = pd.read_csv(sdir / "oof_predictions_v3.csv").copy()
    # Keep only mean/std/n columns
    keep = ["id"] + [c for c in base_oof.columns if c.endswith(("_mean","_std","_n"))]
    base_oof = base_oof[keep].rename(columns=lambda c: f"base_{c}" if c != "id" else c)
    
    base_test = pd.read_csv(sdir / "test_predictions_v3.csv").copy()
    keep_t = ["id"] + [c for c in base_test.columns if c.endswith(("_mean","_std","_n"))]
    base_test = base_test[keep_t].rename(columns=lambda c: f"base_{c}" if c != "id" else c)

    # ── 2. Labels ──
    labels = pd.read_csv(root / "train_labels.csv")[["id", "label"]]
    labels["id"] = labels["id"].astype(str)
    if not labels["id"].iloc[0].endswith(".jpg"):
        labels["id"] = labels["id"] + ".jpg"

    # ── 3. DINOv2 LogReg + KNN ──
    dv_lr_oof = pd.read_csv(fdir / "dinov2_logreg_oof.csv")[["id","prob"]].rename(columns={"prob":"dinov2_lr"})
    dv_lr_te = pd.read_csv(fdir / f"dinov2_logreg_{args.stage}.csv")[["id","prob"]].rename(columns={"prob":"dinov2_lr"})
    dv_knn_oof = pd.read_csv(fdir / "dinov2_knn_oof.csv")[["id","prob"]].rename(columns={"prob":"dinov2_knn"})
    dv_knn_te = pd.read_csv(fdir / f"dinov2_knn_{args.stage}.csv")[["id","prob"]].rename(columns={"prob":"dinov2_knn"})

    # ── 4. CLIP zero-shot ──
    clip_tr = pd.read_csv(fdir / "clip_zeroshot_train.csv").rename(columns={"score":"clip_zs"})
    clip_te = pd.read_csv(fdir / f"clip_zeroshot_{args.stage}.csv").rename(columns={"score":"clip_zs"})

    # ── 5. MAE LogReg (NEW) ──
    mae_avail = (fdir / "mae_logreg_oof.csv").exists()
    if mae_avail:
        mae_oof = pd.read_csv(fdir / "mae_logreg_oof.csv")[["id","prob"]].rename(columns={"prob":"mae_lr"})
        mae_te = pd.read_csv(fdir / f"mae_logreg_{args.stage}.csv")[["id","prob"]].rename(columns={"prob":"mae_lr"})
        print("MAE LogReg: available")
    else:
        print("MAE LogReg: not found, skipped")

    # ── 6. CLIP-LogReg (NEW) ──
    clip_lr_avail = (fdir / "clip_logreg_oof.csv").exists()
    if clip_lr_avail:
        clip_lr_oof = pd.read_csv(fdir / "clip_logreg_oof.csv")[["id","prob"]].rename(columns={"prob":"clip_lr"})
        clip_lr_te = pd.read_csv(fdir / f"clip_logreg_{args.stage}.csv")[["id","prob"]].rename(columns={"prob":"clip_lr"})
        print("CLIP-LogReg: available")
    else:
        print("CLIP-LogReg: not found, skipped")

    # ── 7. 26-dim image features (NEW) ──
    imgf_avail = (sdir / "train_image_features.csv").exists()
    if imgf_avail:
        imgf_oof = pd.read_csv(sdir / "train_image_features.csv")
        imgf_te = pd.read_csv(sdir / "test_image_features.csv")
        # Drop id column for merging
        imgf_oof = imgf_oof.rename(columns=lambda c: f"imgf_{c}" if c != "id" else c)
        imgf_te = imgf_te.rename(columns=lambda c: f"imgf_{c}" if c != "id" else c)
        print(f"Image features: {imgf_oof.shape[1]-1} dims")
    else:
        print("Image features: not found, skipped")

    # ── 8. Domain weights ──
    dw_avail = (fdir / "domain_weights.csv").exists()
    if dw_avail:
        weights = pd.read_csv(fdir / "domain_weights.csv")[["id","weight","in_holdout"]]
    else:
        weights = None

    # ── Build train table ──
    train = base_oof.copy()
    if "label" not in train.columns:
        train = train.merge(labels, on="id", how="left")
    train = train.merge(dv_lr_oof, on="id", how="inner")
    train = train.merge(dv_knn_oof, on="id", how="inner")
    train = train.merge(clip_tr, on="id", how="inner")
    if mae_avail: train = train.merge(mae_oof, on="id", how="inner")
    if clip_lr_avail: train = train.merge(clip_lr_oof, on="id", how="inner")
    if imgf_avail: train = train.merge(imgf_oof, on="id", how="inner")
    if weights is not None:
        train = train.merge(weights, on="id", how="left")
    else:
        train["weight"] = 1.0
        train["in_holdout"] = 0
    train["weight"] = train["weight"].fillna(1.0)
    train["in_holdout"] = train["in_holdout"].fillna(0).astype(int)
    print(f"Train: {train.shape}  holdout: {int(train['in_holdout'].sum())}")

    # ── Build test table ──
    test = base_test.copy()
    test = test.merge(dv_lr_te, on="id", how="inner")
    test = test.merge(dv_knn_te, on="id", how="inner")
    test = test.merge(clip_te, on="id", how="inner")
    if mae_avail: test = test.merge(mae_te, on="id", how="inner")
    if clip_lr_avail: test = test.merge(clip_lr_te, on="id", how="inner")
    if imgf_avail: test = test.merge(imgf_te, on="id", how="inner")
    print(f"Test: {test.shape}")

    # ── Feature columns ──
    feature_cols = [c for c in train.columns if c not in ("id","label","weight","in_holdout")]
    test_feature_cols = [c for c in test.columns if c != "id"]
    feature_cols = [c for c in feature_cols if c in test_feature_cols]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    train_dev = train[train["in_holdout"]==0].reset_index(drop=True)
    train_hold = train[train["in_holdout"]==1].reset_index(drop=True)

    X_dev = train_dev[feature_cols].values
    y_dev = train_dev["label"].values.astype(int)
    w_dev = train_dev["weight"].values

    # ── LightGBM ──
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
        f_w = f1_score(y_dev[va], (oof_dev[va] > 0.5).astype(int), sample_weight=w_dev[va])
        print(f"  fold {fold}: F1@0.5={f_unw:.4f}  weighted={f_w:.4f}")

    importances /= len(fold_models)
    imp_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
    (out_dir / "features.txt").write_text("\n".join(f"{n}\t{i:.1f}" for n,i in imp_pairs))
    print("\nTop features:")
    for n, i in imp_pairs[:12]:
        print(f"  {n:35s}  {i:.1f}")

    weighted_oof_f1 = f1_score(y_dev, (oof_dev>0.5).astype(int), sample_weight=w_dev)
    print(f"\nweighted OOF F1@0.5: {weighted_oof_f1:.4f}")

    # ── Holdout ──
    if len(train_hold):
        X_hold = train_hold[feature_cols].values
        y_hold = train_hold["label"].values.astype(int)
        hold_probs = np.mean([m.predict_proba(X_hold)[:,1] for m in fold_models], axis=0)
        n_pos = int(y_hold.sum())
        thr_grid = np.linspace(0.1, 0.9, 81)
        best_thr = max(thr_grid, key=lambda t: f1_score(y_hold, (hold_probs>t).astype(int)))
        holdout_f1 = f1_score(y_hold, (hold_probs>best_thr).astype(int))
        print(f"holdout F1@best_thr={best_thr:.3f}: {holdout_f1:.4f}")
    else:
        holdout_f1 = float("nan")

    # ── Test prediction ──
    X_test = test[feature_cols].values
    test_probs = np.stack([m.predict_proba(X_test)[:,1] for m in fold_models], axis=1)
    test_mean = test_probs.mean(axis=1)
    test_std = test_probs.std(axis=1)

    pd.DataFrame({"id": test["id"], "prob": test_mean, "prob_std": test_std}).to_csv(
        out_dir / "stacking_v5_test.csv", index=False
    )
    pd.DataFrame({"id": train_dev["id"], "label": y_dev, "prob": oof_dev}).to_csv(
        out_dir / "stacking_v5_oof.csv", index=False
    )

    # ── Generate submissions ──
    ks = [int(k) for k in args.ks.split(",")]
    sub_dir = out_dir / "topk_submissions"
    sub_dir.mkdir(exist_ok=True)
    template = pd.read_csv(root / f"sample_submission_{args.stage}.csv")
    sorted_test = test[["id"]].assign(score=test_mean).sort_values("score", ascending=False).reset_index(drop=True)

    for k in ks:
        pos = set(sorted_test["id"].head(k))
        s = template.copy()
        s["label"] = s["id"].isin(pos).astype(int)
        s.to_csv(sub_dir / f"submission_top{k}.csv", index=False)

    print(f"\nSubmissions saved to {sub_dir}/")
    print(f"Test probs saved to {out_dir / 'stacking_v5_test.csv'}")


if __name__ == "__main__":
    main()
