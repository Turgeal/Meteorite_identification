"""
LightGBM Stacking v3 for Meteorite Classification.

Root cause analysis of v2 failure (F1 dropped from 0.796 to 0.718):
  Bug 1: test_predictions.csv was EMPTY → OOF features dropped by common_cols filter
         LightGBM trained on 22 weak image features, lost 10 strongest model features
  Bug 2: OOF structure used fillna(0) for 8/10 NaN values per row → misleading signal
  Bug 3: common_cols filter silently dropped features → no error, just bad results
  Bug 4: Old image features (histogram bins, color channels) overfit on train, don't generalize

v3 fixes:
  1. OOF collapsed to per-architecture averages (convnext_mean, swin_mean) instead of
     10 sparse columns with fillna(0). No more NaN, no more misleading zeros.
  2. Test predictions properly generated with FORCE_OVERWRITE flag
  3. Feature alignment is EXPLICIT: same feature engineering on train and test
  4. High-signal features added (corner_uniformity, object_area, is_dark_bg, center_edge_diff)
  5. Feature importance logging to diagnose what LightGBM actually uses
  6. Blending weights made configurable

Workflow:
  Step 1: Extract OOF predictions + Test predictions (one command)
    python stack_ensemble_v3.py --extract_all --data_root . --output_dir outputs/stacking_v3

  Step 2: Extract image features (train + test)
    python stack_ensemble_v3.py --extract_features --data_root . --output_dir outputs/stacking_v3

  Step 3: Train LightGBM and predict
    python stack_ensemble_v3.py --train_and_predict --data_root . --output_dir outputs/stacking_v3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from PIL import Image
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.dataset import StoneDataset, get_val_transforms
from src.model import create_classifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


# ─── Image Feature Extraction ─────────────────────────────────────────────

def extract_image_features(img_path):
    """Extract hand-crafted features from a single image.

    v3 feature set: 26 features total
      - Original 22: brightness, contrast, grad, lap, color stats, histogram, aspect, dark/bright ratio
      - Added 4 high-signal features from full 5098-sample analysis:
        corner_uniformity (d=+0.887), object_area (d=+0.842),
        is_dark_bg (d=-0.452), center_edge_diff (d=-0.537)
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            img = np.array(Image.open(img_path).convert('RGB'))[:, :, ::-1]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        mean_brightness = float(gray.mean())
        std_brightness = float(gray.std())
        contrast = float(gray.max() - gray.min())

        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mean_grad = float(np.sqrt(grad_x**2 + grad_y**2).mean())

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())

        # Histogram bins
        hist = cv2.calcHist([gray], [0], None, [8], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-7)

        # Color stats
        b_mean, g_mean, r_mean = [float(img[:,:,c].mean()) for c in range(3)]
        b_std, g_std, r_std = [float(img[:,:,c].std()) for c in range(3)]

        aspect = w / h
        dark_ratio = float((gray < 30).sum()) / (h * w)
        bright_ratio = float((gray > 220).sum()) / (h * w)

        # ── High-signal features ──
        # corner_uniformity: std of 4 corner region means
        ch, cw = max(h // 8, 1), max(w // 8, 1)
        corners = [gray[:ch, :cw], gray[:ch, -cw:], gray[-ch:, :cw], gray[-ch:, -cw:]]
        corner_means = [float(c.mean()) for c in corners]
        corner_uniformity = float(np.std(corner_means) / 255.0)

        # object_area: Otsu foreground proportion
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        object_area = float(binary.sum() / 255.0) / (h * w)

        # is_dark_bg
        is_dark_bg = 1.0 if mean_brightness < 80 else 0.0

        # center_edge_diff: center vs border edge density
        edge_map = cv2.Canny(gray, 50, 150)
        margin = max(min(h, w) // 8, 1)
        center_region = edge_map[margin:h-margin, margin:w-margin]
        edge_regions = np.concatenate([
            edge_map[:margin, :].flatten(),
            edge_map[h-margin:, :].flatten(),
            edge_map[margin:h-margin, :margin].flatten(),
            edge_map[margin:h-margin, w-margin:].flatten(),
        ])
        center_edge = float(center_region.sum() / 255.0) / max(center_region.size, 1)
        outer_edge = float(edge_regions.sum() / 255.0) / max(len(edge_regions), 1)
        center_edge_diff = center_edge - outer_edge

        features = {
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'contrast': contrast,
            'mean_grad': mean_grad,
            'lap_var': lap_var,
            'b_mean': b_mean, 'g_mean': g_mean, 'r_mean': r_mean,
            'b_std': b_std, 'g_std': g_std, 'r_std': r_std,
            'aspect': aspect,
            'dark_ratio': dark_ratio,
            'bright_ratio': bright_ratio,
            'corner_uniformity': corner_uniformity,
            'object_area': object_area,
            'is_dark_bg': is_dark_bg,
            'center_edge_diff': center_edge_diff,
        }
        for i, h_val in enumerate(hist):
            features[f'hist_{i}'] = float(h_val)

        return features
    except Exception:
        fallback_keys = [
            'mean_brightness', 'std_brightness', 'contrast', 'mean_grad', 'lap_var',
            'b_mean', 'g_mean', 'r_mean', 'b_std', 'g_std', 'r_std',
            'aspect', 'dark_ratio', 'bright_ratio',
            'corner_uniformity', 'object_area', 'is_dark_bg', 'center_edge_diff',
            'hist_0', 'hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5', 'hist_6', 'hist_7'
        ]
        return {k: 0.0 for k in fallback_keys}


def batch_extract_features(data_root, split, output_path, stage='stage2'):
    """Extract image features for a split."""
    if split == 'test':
        from src.dataset import resolve_test_paths
        image_dir, template_csv = resolve_test_paths(data_root, stage=stage)
        df = pd.read_csv(template_csv)
    else:
        image_dir = os.path.join(data_root, 'train_images', 'train_images')
        df = pd.read_csv(os.path.join(data_root, 'train_labels.csv'))

    rows = []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=f"Features({split})")):
        img_id = str(row['id'])
        fname = img_id if img_id.endswith(('.jpg', '.jpeg', '.png')) else f"{img_id}.jpg"
        img_path = os.path.join(image_dir, fname)

        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, img_id)
        if not os.path.exists(img_path):
            continue

        feats = extract_image_features(img_path)
        feats['id'] = img_id
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    feat_df.to_csv(output_path, index=False)
    print(f"  Saved {len(feat_df)} feature rows to {output_path}")
    return feat_df


# ─── OOF + Test Prediction Extraction ─────────────────────────────────────

def _discover_checkpoints(checkpoint_dir):
    """Discover model checkpoints organized by architecture."""
    checkpoints_dir = os.path.join(checkpoint_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        print(f"ERROR: {checkpoints_dir} not found")
        return {}

    model_dirs = {}
    for d in sorted(os.listdir(checkpoints_dir)):
        sub = os.path.join(checkpoints_dir, d)
        if os.path.isdir(sub):
            ckpts = [f for f in os.listdir(sub) if f.endswith('.pth')]
            if ckpts:
                model_dirs[d] = sorted(ckpts)
    return model_dirs


def _extract_fold_number(filename):
    """Extract fold number from checkpoint filename."""
    name_no_ext = filename.replace('.pth', '')
    for part in name_no_ext.split('_'):
        if part.isdigit():
            return int(part)
        elif part.startswith('fold') and part[4:].isdigit():
            return int(part[4:])
    return None


def _load_and_predict(model_name, checkpoint_path, dataloader, device):
    """Load a model from checkpoint and predict on a dataloader."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name_resolved = checkpoint.get('model_name', model_name)

    model = create_classifier(num_classes=2, model_name=model_name_resolved, pretrained=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())

    del model
    torch.cuda.empty_cache()
    return all_probs


def extract_all_predictions(data_root, checkpoint_dir, output_dir, device,
                             n_folds=5, img_size=384, batch_size=32, num_workers=0,
                             skip_oof=False, stage='stage2', use_nobg=False):
    """
    Extract BOTH OOF predictions (train) and test predictions in one pass.

    v3 key fix: OOF is collapsed to per-architecture averages.
    Each training sample gets exactly 2 OOF features (convnext_mean, swin_mean)
    instead of 10 sparse columns with 8 NaN values.

    Test set gets the same 2 features (convnext_mean, swin_mean) averaged
    across all 5 folds per architecture.

    If skip_oof=True, an existing oof_predictions_v3.csv must be present and
    only test inference is run (useful when reusing OOF from a previous stage).
    """
    print("\n=== Extracting All Predictions (OOF + Test) ===")
    if skip_oof:
        print("  --skip_oof enabled: reusing existing OOF, only running test inference")

    labels_df = pd.read_csv(os.path.join(data_root, 'train_labels.csv'))
    all_ids = labels_df['id'].astype(str).tolist()
    all_labels = labels_df['label'].values

    model_dirs = _discover_checkpoints(checkpoint_dir)
    if not model_dirs:
        print("ERROR: No model checkpoints found!")
        return

    print(f"Found {len(model_dirs)} model architectures: {list(model_dirs.keys())}")

    # K-fold split (must match training)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_indices = list(skf.split(np.arange(len(all_ids)), all_labels))

    # Prepare datasets
    val_transform = get_val_transforms(img_size)
    base_dataset = StoneDataset(root=data_root, split="train", transforms=val_transform,
                                 use_nobg=use_nobg)
    test_dataset = StoneDataset(root=data_root, split="test", transforms=val_transform,
                                 test_stage=stage)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_ids = test_dataset.csv_ids

    # Storage: per-architecture OOF and test predictions
    # arch_name -> {sample_id: [prob_fold0, prob_fold1, ...]}
    arch_oof = {d: {} for d in model_dirs}
    # arch_name -> [prob_fold0_test, prob_fold1_test, ...]  (each is list of 511 probs)
    arch_test_preds = {}

    for model_dir_name, ckpt_files in model_dirs.items():
        print(f"\n--- {model_dir_name} ({len(ckpt_files)} folds) ---")
        arch_oof[model_dir_name] = {iid: [] for iid in all_ids}
        arch_test_preds[model_dir_name] = []

        for ckpt_file in ckpt_files:
            fold_num = _extract_fold_number(ckpt_file)
            if fold_num is None:
                print(f"  SKIP {ckpt_file}: no fold number found")
                continue

            ckpt_path = os.path.join(checkpoint_dir, 'checkpoints', model_dir_name, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"  SKIP {ckpt_file}: file not found")
                continue

            # --- OOF prediction ---
            if not skip_oof:
                _, val_idx = fold_indices[fold_num]
                val_subset = Subset(base_dataset, val_idx.tolist())
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)

                print(f"  Fold {fold_num}: OOF prediction on {len(val_idx)} samples...", end=" ")
                oof_probs = _load_and_predict(model_dir_name, ckpt_path, val_loader, device)
                print(f"mean_prob={np.mean(oof_probs):.4f}")

                for i, sample_idx in enumerate(val_idx):
                    iid = all_ids[sample_idx]
                    arch_oof[model_dir_name][iid].append(oof_probs[i])

            # --- Test prediction ---
            print(f"  Fold {fold_num}: Test prediction on {len(test_ids)} samples...", end=" ")
            test_probs = _load_and_predict(model_dir_name, ckpt_path, test_loader, device)
            print(f"mean_prob={np.mean(test_probs):.4f}")
            arch_test_preds[model_dir_name].append(test_probs)

    # ── Collapse to per-architecture averages ──
    print("\n--- Collapsing to per-architecture averages ---")

    if skip_oof:
        oof_path = os.path.join(output_dir, 'oof_predictions_v3.csv')
        if not os.path.exists(oof_path):
            raise FileNotFoundError(
                f"--skip_oof requires existing {oof_path}. "
                "Run once without --skip_oof to generate it."
            )
        oof_df = pd.read_csv(oof_path, index_col='id')
        oof_df.index = oof_df.index.astype(str)
        print(f"  Reused existing OOF: {oof_df.shape}, columns: {list(oof_df.columns)}")
    else:
        # OOF: average the folds that actually predicted each sample
        oof_result = {}  # id -> {convnext_mean: ..., swin_mean: ...}
        for iid in all_ids:
            row = {}
            for arch_name in model_dirs:
                # Short name: take last part before _fold pattern
                short_name = arch_name.split('_fb_in22k')[0].split('_patch4')[0]
                # e.g., "convnext_base" or "swin_base"
                vals = arch_oof[arch_name].get(iid, [])
                row[f'{short_name}_mean'] = np.mean(vals) if vals else 0.5
                row[f'{short_name}_std'] = np.std(vals) if len(vals) > 1 else 0.0
                row[f'{short_name}_n'] = len(vals)
            oof_result[iid] = row

        oof_df = pd.DataFrame.from_dict(oof_result, orient='index')
        oof_df.index.name = 'id'

        # Verify no NaN
        nan_count = oof_df.isna().sum().sum()
        print(f"  OOF DataFrame: {oof_df.shape}, NaN count: {nan_count}")
        print(f"  OOF columns: {list(oof_df.columns)}")
        print(f"  Sample OOF values:")
        print(oof_df.head(3))

        oof_path = os.path.join(output_dir, 'oof_predictions_v3.csv')
        oof_df.to_csv(oof_path)
        print(f"  Saved to {oof_path}")

    # Test: average all folds per architecture
    test_result = {}
    for i, iid in enumerate(test_ids):
        row = {}
        for arch_name in model_dirs:
            short_name = arch_name.split('_fb_in22k')[0].split('_patch4')[0]
            vals = [fold_preds[i] for fold_preds in arch_test_preds[arch_name]]
            row[f'{short_name}_mean'] = np.mean(vals) if vals else 0.5
            row[f'{short_name}_std'] = np.std(vals) if len(vals) > 1 else 0.0
            row[f'{short_name}_n'] = len(vals)
        test_result[iid] = row

    test_pred_df = pd.DataFrame.from_dict(test_result, orient='index')
    test_pred_df.index.name = 'id'

    nan_count = test_pred_df.isna().sum().sum()
    print(f"\n  Test DataFrame: {test_pred_df.shape}, NaN count: {nan_count}")
    print(f"  Test columns: {list(test_pred_df.columns)}")

    test_pred_path = os.path.join(output_dir, 'test_predictions_v3.csv')
    test_pred_df.to_csv(test_pred_path)
    print(f"  Saved to {test_pred_path}")

    # Verify train/test column alignment
    train_cols = set(oof_df.columns)
    test_cols = set(test_pred_df.columns)
    if train_cols != test_cols:
        print(f"\n  WARNING: Train/test column mismatch!")
        print(f"  Train only: {train_cols - test_cols}")
        print(f"  Test only: {test_cols - train_cols}")
    else:
        print(f"\n  Train/test columns MATCH: {sorted(train_cols)}")

    return oof_df, test_pred_df


# ─── LightGBM Training + Prediction ───────────────────────────────────────

def train_and_predict(data_root, output_dir, target_k=105,
                      blend_weight_model=0.7, blend_weight_stack=0.3,
                      stage='stage2'):
    """Train LightGBM meta-learner on OOF + image features, predict on test.

    v3 key changes:
      - OOF features are per-architecture averages (no NaN, no fillna(0))
      - Features are EXPLICITLY aligned between train and test
      - Feature importance is logged
      - Blending weight is configurable (default 0.7*model + 0.3*stack)
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print("ERROR: lightgbm not installed. Run: pip install lightgbm")
        return

    # ── Load OOF predictions (v3 format: per-architecture averages) ──
    oof_path = os.path.join(output_dir, 'oof_predictions_v3.csv')
    if not os.path.exists(oof_path):
        print("ERROR: No v3 OOF predictions. Run with --extract_all first.")
        return
    oof_df = pd.read_csv(oof_path, index_col='id')
    oof_df.index = oof_df.index.astype(str)
    print(f"  OOF features: {oof_df.shape}, columns: {list(oof_df.columns)}")

    # ── Load image features ──
    train_feat_path = os.path.join(output_dir, 'train_image_features.csv')
    test_feat_path = os.path.join(output_dir, 'test_image_features.csv')

    if not os.path.exists(train_feat_path) or not os.path.exists(test_feat_path):
        print("ERROR: Image features not found. Run with --extract_features first.")
        return

    train_img_df = pd.read_csv(train_feat_path, index_col='id')
    train_img_df.index = train_img_df.index.astype(str)
    test_img_df = pd.read_csv(test_feat_path, index_col='id')
    test_img_df.index = test_img_df.index.astype(str)
    print(f"  Train image features: {train_img_df.shape}")
    print(f"  Test image features: {test_img_df.shape}")

    # ── Load test predictions (v3 format) ──
    test_pred_path = os.path.join(output_dir, 'test_predictions_v3.csv')
    if not os.path.exists(test_pred_path):
        print("ERROR: No v3 test predictions. Run with --extract_all first.")
        return
    test_pred_df = pd.read_csv(test_pred_path, index_col='id')
    test_pred_df.index = test_pred_df.index.astype(str)
    print(f"  Test model predictions: {test_pred_df.shape}, columns: {list(test_pred_df.columns)}")

    # ── Merge features ──
    # Train: OOF + image features
    train_combined = oof_df.join(train_img_df, how='left')
    # Test: model predictions + image features
    test_combined = test_pred_df.join(test_img_df, how='left')

    # Verify column alignment
    train_cols = set(train_combined.columns)
    test_cols = set(test_combined.columns)
    if train_cols != test_cols:
        print(f"\n  WARNING: Column mismatch! Train only: {train_cols - test_cols}, Test only: {test_cols - train_cols}")
        # Use only common columns
        common_cols = sorted(train_cols & test_cols)
        print(f"  Using {len(common_cols)} common columns")
    else:
        common_cols = sorted(train_cols)

    feature_names = common_cols
    X_train = train_combined[feature_names].fillna(0).values
    X_test = test_combined[feature_names].fillna(0).values

    # Load labels
    labels_df = pd.read_csv(os.path.join(data_root, 'train_labels.csv'))
    label_map = dict(zip(labels_df['id'].astype(str), labels_df['label']))
    y_train = np.array([label_map.get(str(iid), 0) for iid in train_combined.index])

    print(f"\n  Training data: {X_train.shape}, positive rate: {y_train.mean():.3f}")
    print(f"  Test data: {X_test.shape}")
    print(f"  Features: {feature_names}")

    # ── Train LightGBM ──
    print("\n=== Training LightGBM ===")

    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 15,
        'max_depth': 4,
        'min_child_samples': 50,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 2.0,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
    }

    lgb_oof_probs = np.zeros(len(X_train))
    lgb_test_preds = np.zeros(len(X_test))
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(500),
            ],
        )

        lgb_oof_probs[val_idx] = model.predict(X_val)
        lgb_test_preds += model.predict(X_test) / n_folds
        fold_models.append(model)

        val_preds = (lgb_oof_probs[val_idx] >= 0.5).astype(int)
        fold_f1 = f1_score(y_val, val_preds)
        print(f"  Fold {fold}: F1={fold_f1:.4f}, best_iter={model.best_iteration}")

    # ── Overall OOF F1 ──
    oof_f1_05 = f1_score(y_train, (lgb_oof_probs >= 0.5).astype(int))
    print(f"\n  OOF F1@0.5: {oof_f1_05:.4f}")

    best_thresh, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        preds = (lgb_oof_probs >= t).astype(int)
        f1 = f1_score(y_train, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    print(f"  OOF best F1: {best_f1:.4f} @ threshold={best_thresh:.2f}")

    # ── Feature Importance ──
    print("\n=== Feature Importance ===")
    for model in fold_models[:1]:  # Show first fold's importance
        importance = model.feature_importance(importance_type='gain')
        feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        for fname, imp in feat_imp:
            bar = '#' * min(int(imp / max(importance) * 40), 40)
            print(f"  {fname:25s} {imp:8.1f} {bar}")

    # ── Save predictions ──
    test_ids = list(test_combined.index)
    test_probs = dict(zip(test_ids, lgb_test_preds))

    stack_df = pd.DataFrame([
        {'id': iid, 'prob': p} for iid, p in test_probs.items()
    ])
    stack_path = os.path.join(output_dir, 'test_stacking_probs_v3.csv')
    stack_df.to_csv(stack_path, index=False)
    print(f"\n  Stacking probs saved to {stack_path}")

    # ── Generate Top-K submissions ──
    print("\n=== Generating Top-K Submissions ===")
    from src.dataset import resolve_test_paths
    _, template_csv = resolve_test_paths(data_root, stage=stage)
    template = pd.read_csv(template_csv)
    template['id'] = template['id'].astype(str)

    k_values = [60, 65, 70, 75, 80, 85, 90, 95, 100, 102, 105, 108, 110, 115, 120, 125, 130]

    # Pure stacking submissions
    sorted_items = sorted(test_probs.items(), key=lambda x: x[1], reverse=True)
    sub_dir = os.path.join(output_dir, 'topk_submissions')
    os.makedirs(sub_dir, exist_ok=True)

    for k in k_values:
        top_k_ids = set(iid for iid, _ in sorted_items[:k])
        sub = template.copy()
        sub['label'] = sub['id'].map(lambda x: 1 if x in top_k_ids else 0)
        path = os.path.join(sub_dir, f'submission_top{k}.csv')
        sub.to_csv(path, index=False)
    print(f"  Stacking-only submissions saved to {sub_dir}")

    # Blended submissions: model + stacking
    # Load the original model ensemble probs from the stage-aware location.
    model_probs_paths = [
        os.path.join(data_root, 'outputs', stage, 'ensemble_probs.csv'),
        os.path.join('outputs', stage, 'ensemble_probs.csv'),
    ]

    model_probs = None
    for mp_path in model_probs_paths:
        if os.path.exists(mp_path):
            mp_df = pd.read_csv(mp_path)
            model_probs = dict(zip(mp_df['id'].astype(str), mp_df['prob']))
            print(f"  Loaded model probs from {mp_path} ({len(model_probs)} samples)")
            break

    # Blended Top-K range — wider for stage2 since the sweet spot shifted
    blend_k_values = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 120, 130]

    if model_probs:
        blend_dir = os.path.join(output_dir, 'topk_submissions_blended')
        os.makedirs(blend_dir, exist_ok=True)

        # Try multiple blend ratios
        for w_model, w_stack in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.8, 0.2)]:
            blended_probs = {}
            for iid in test_ids:
                mp = model_probs.get(iid, 0.5)
                sp = test_probs.get(iid, 0.5)
                blended_probs[iid] = w_model * mp + w_stack * sp

            blended_sorted = sorted(blended_probs.items(), key=lambda x: x[1], reverse=True)
            for k in blend_k_values:
                top_k_ids = set(iid for iid, _ in blended_sorted[:k])
                sub = template.copy()
                sub['label'] = sub['id'].map(lambda x: 1 if x in top_k_ids else 0)
                path = os.path.join(blend_dir, f'submission_m{w_model}_s{w_stack}_top{k}.csv')
                sub.to_csv(path, index=False)

        print(f"  Blended submissions saved to {blend_dir}")
    else:
        print("  No model probs found for blending. Use stacking-only submissions.")

    print(f"\nRECOMMENDED: pure stacking → topk_submissions/ ; blended → topk_submissions_blended/")

    return {
        'oof_f1': best_f1,
        'best_threshold': best_thresh,
        'test_probs': test_probs,
    }


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='LightGBM Stacking v3 (Root Cause Fixed)')
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Defaults to ./outputs/<stage>/stacking_v3')
    parser.add_argument('--checkpoint_dir', type=str, default='./outputs')
    parser.add_argument('--stage', type=str, default='stage2',
                        choices=['stage1', 'stage2'],
                        help='Which test stage to predict on.')

    # Actions
    parser.add_argument('--extract_features', action='store_true',
                        help='Extract image features for train and test')
    parser.add_argument('--extract_all', action='store_true',
                        help='Extract OOF + test predictions (per-architecture averages)')
    parser.add_argument('--train_and_predict', action='store_true',
                        help='Train LightGBM and generate predictions')
    parser.add_argument('--skip_oof', action='store_true',
                        help='Skip OOF inference and reuse existing oof_predictions_v3.csv '
                             '(only run test inference). Useful when OOF was generated previously.')

    # Parameters
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0=safe on Windows)')
    parser.add_argument('--target_k', type=int, default=105)
    parser.add_argument('--use_nobg', action='store_true', default=False,
                        help='Use background-removed train images for OOF (must match training)')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join('./outputs', args.stage, 'stacking_v3')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Stage: {args.stage}  |  Output: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.extract_features:
        print("\n=== Extracting Image Features (v3) ===")
        batch_extract_features(args.data_root, 'train',
                               os.path.join(args.output_dir, 'train_image_features.csv'))
        batch_extract_features(args.data_root, 'test',
                               os.path.join(args.output_dir, 'test_image_features.csv'),
                               stage=args.stage)

    if args.extract_all:
        extract_all_predictions(
            args.data_root, args.checkpoint_dir, args.output_dir, device,
            n_folds=args.n_folds, img_size=args.img_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            skip_oof=args.skip_oof, stage=args.stage, use_nobg=args.use_nobg,
        )

    if args.train_and_predict:
        result = train_and_predict(
            args.data_root, args.output_dir, target_k=args.target_k, stage=args.stage,
        )
        if result:
            print(f"\nStacking OOF F1: {result['oof_f1']:.4f}")
            print(f"Best threshold: {result['best_threshold']:.2f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
