"""
Intelligent Prediction with Optimal Top-K Selection.

This script replaces the threshold-based approach with a smarter strategy:

1. Load ALL available model checkpoints (kfold + any other trained models)
2. Apply multi-scale TTA (224 + 384 resolution)
3. Ensemble predictions using multiple methods:
   - Simple average
   - Weighted average (by validation F1)
   - Rank-based ensemble
   - Geometric mean
   - Power mean
4. Analyze probability distribution to estimate test-set positive rate
5. Generate optimal Top-K submissions with fine-grained K values
6. Estimate the best K based on probability distribution analysis

The key insight: on the test set, the positive rate is ~20% (vs ~48% train).
So we use Top-K selection rather than probability thresholds.

Usage:
  python predict_optimal.py --data_root . --output_dir outputs
  python predict_optimal.py --data_root . --estimate_k  # analyze distribution only
"""
import os
import sys
import argparse
import json
import math
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from src.dataset import (
    StoneDataset, get_val_transforms, get_tta_transforms, resolve_test_paths
)
from src.model import create_classifier, load_model


# ─────────────────────────────────────────────
#  Model Discovery
# ─────────────────────────────────────────────

def discover_all_checkpoints(output_dir, data_root):
    """Discover ALL available model checkpoints in the checkpoints/ directory."""
    all_paths = []
    seen_paths = set()

    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    if not os.path.isdir(checkpoints_dir):
        return all_paths

    for f in os.listdir(checkpoints_dir):
        sub_path = os.path.join(checkpoints_dir, f)
        if os.path.isdir(sub_path):
            # Per-model subdirectory (new format)
            for inner in sorted(os.listdir(sub_path)):
                if inner.endswith('.pth') and os.path.getsize(os.path.join(sub_path, inner)) > 1_000_000:
                    path = os.path.join(sub_path, inner)
                    if path not in seen_paths:
                        seen_paths.add(path)
                        all_paths.append(('checkpoints/' + f, path))
        elif f.endswith('.pth') and os.path.getsize(sub_path) > 1_000_000:
            # Flat .pth file (legacy)
            if sub_path not in seen_paths:
                seen_paths.add(sub_path)
                all_paths.append(('checkpoints', sub_path))

    return all_paths


def infer_model_name_from_checkpoint(checkpoint):
    """Infer model architecture from checkpoint state_dict keys.
    NOTE: This is a fallback. train_kfold.py always saves 'model_name' in the
    checkpoint dict, so the caller should check checkpoint['model_name'] first.
    """
    # If model_name is stored in checkpoint metadata, prefer it
    if isinstance(checkpoint, dict):
        saved_name = checkpoint.get('model_name')
        if saved_name and isinstance(saved_name, str) and len(saved_name) > 3:
            return saved_name
        if 'model_state_dict' in checkpoint:
            state_keys = list(checkpoint['model_state_dict'].keys())
        else:
            state_keys = list(checkpoint.keys())
    else:
        state_keys = []

    # ConvNeXt detection
    if any('stem.conv' in k or 'stages.0' in k for k in state_keys):
        if any('layernorm' in k or 'gamma' in k for k in state_keys):
            return 'convnext_small.fb_in22k_ft_in1k'

    # EfficientNet detection
    if any('conv_stem' in k or 'blocks.0.0' in k for k in state_keys):
        if any('bn1' in k for k in state_keys):
            return 'efficientnet_b4.ra2_in1k'
        return 'efficientnet_b0.ra_in1k'

    # Swin detection
    if any('patch_embed' in k or 'layers' in k for k in state_keys):
        patch_embeds = [k for k in state_keys if 'patch_embed' in k]
        if patch_embeds:
            key = patch_embeds[0]
            val = None
            if isinstance(checkpoint, dict):
                val = checkpoint.get('model_state_dict', checkpoint).get(key)
            if val is not None:
                if val.shape[0] == 96:
                    return 'swin_tiny_patch4_window7_224.ms_in1k'
                if val.shape[0] == 128:
                    return 'swin_base_patch4_window12_384.ms_in22k_ft_in1k'
                if val.shape[0] == 1536:
                    return 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'

    return 'convnext_small.fb_in22k_ft_in1k'


# ─────────────────────────────────────────────
#  TTA Prediction
# ─────────────────────────────────────────────

def get_multiscale_tta(img_size=224):
    """Get TTA transforms at multiple scales for better robustness."""
    import torchvision.transforms.functional as TF
    from torchvision import transforms

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    def fixed_rotate(img, angle):
        return TF.rotate(img, angle)

    tta_list = []

    # Original + standard flips
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            t = [
                transforms.Resize((img_size, img_size)),
            ]
            if flip_h:
                t.append(transforms.RandomHorizontalFlip(p=1.0))
            if flip_v:
                t.append(transforms.RandomVerticalFlip(p=1.0))
            t.extend([transforms.ToTensor(), normalize])
            tta_list.append(transforms.Compose(t))

    # Rotations
    for angle in [15, -15, 45, -45]:
        tta_list.append(transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img, a=angle: fixed_rotate(img, a)),
            transforms.ToTensor(),
            normalize,
        ]))

    # Multi-scale
    for scale in [1.1, 1.2, 1.3]:
        tta_list.append(transforms.Compose([
            transforms.Resize((int(img_size * scale), int(img_size * scale))),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]))

    return tta_list


def predict_with_tta(model, images, ids, device, tta_list, batch_size=32):
    """Run inference with multiple TTA transforms."""
    id_to_probs = {iid: [] for iid in ids}

    for t_idx, t in enumerate(tta_list):
        id_to_prob = {}
        for start in range(0, len(images), batch_size):
            end = min(start + batch_size, len(images))
            batch = torch.stack([t(img) for img in images[start:end]])
            batch = batch.to(device, non_blocking=True)
            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            for p, iid in zip(probs, ids[start:end]):
                id_to_prob[iid] = float(p)
        for iid in ids:
            id_to_probs[iid].append(id_to_prob.get(iid, 0.5))

    # Average TTA predictions
    return {iid: np.mean(probs) for iid, probs in id_to_probs.items()}


# ─────────────────────────────────────────────
#  Ensemble Methods
# ─────────────────────────────────────────────

def ensemble_average(all_predictions):
    """Simple average across all models."""
    if not all_predictions:
        return {}
    ids = list(all_predictions[0].keys())
    result = {}
    for iid in ids:
        result[iid] = np.mean([pred.get(iid, 0.5) for pred in all_predictions])
    return result


def ensemble_weighted(all_predictions, weights):
    """Weighted average by validation F1."""
    if not all_predictions:
        return {}
    ids = list(all_predictions[0].keys())
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    result = {}
    for iid in ids:
        result[iid] = sum(
            pred.get(iid, 0.5) * wi
            for pred, wi in zip(all_predictions, w)
        )
    return result


def ensemble_geomean(all_predictions):
    """Geometric mean (reduces impact of overconfident predictions)."""
    if not all_predictions:
        return {}
    ids = list(all_predictions[0].keys())
    eps = 1e-8
    result = {}
    for iid in ids:
        vals = [max(pred.get(iid, 0.5), eps) for pred in all_predictions]
        result[iid] = np.exp(np.mean(np.log(vals)))
    return result


def ensemble_power_mean(all_predictions, p=0.5):
    """Power mean (p=1 is arithmetic mean, p=-1 is harmonic mean, p->0 is geometric)."""
    if not all_predictions:
        return {}
    ids = list(all_predictions[0].keys())
    eps = 1e-8
    result = {}
    for iid in ids:
        vals = [max(pred.get(iid, eps), eps) for pred in all_predictions]
        if abs(p) < 1e-6:
            result[iid] = np.exp(np.mean(np.log(vals)))
        else:
            result[iid] = np.power(np.mean(np.power(vals, p)), 1.0 / p)
    return result


def ensemble_rank(all_predictions):
    """Rank-based ensemble (lower average rank = better)."""
    if not all_predictions:
        return {}
    ids = list(all_predictions[0].keys())
    model_ranks = []
    for pred in all_predictions:
        sorted_items = sorted(pred.items(), key=lambda x: x[1], reverse=True)
        rank_map = {iid: rank for rank, (iid, _) in enumerate(sorted_items)}
        model_ranks.append(rank_map)

    avg_rank = {}
    for iid in ids:
        ranks = [rm.get(iid, len(rm)) for rm in model_ranks]
        avg_rank[iid] = np.mean(ranks)
    return avg_rank  # Lower is better


# ─────────────────────────────────────────────
#  Top-K Analysis
# ─────────────────────────────────────────────

def analyze_distribution(probs_dict):
    """Analyze probability distribution to estimate best K."""
    probs = np.array(list(probs_dict.values()))
    probs_sorted = np.sort(probs)[::-1]

    print("\n=== Probability Distribution Analysis ===")
    print(f"  Mean: {probs.mean():.4f}, Std: {probs.std():.4f}")
    print(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}")
    print(f"  Median: {np.median(probs):.4f}")

    # Find the "elbow" point where probabilities drop sharply
    diffs = np.diff(probs_sorted)
    elbow_idx = np.argmax(np.abs(diffs)) + 1
    print(f"  Elbow point: rank {elbow_idx}, prob={probs_sorted[elbow_idx]:.4f}")

    # Analyze probability gaps
    gaps = []
    for i in range(1, min(200, len(probs_sorted))):
        gap = probs_sorted[i-1] - probs_sorted[i]
        gaps.append((i, gap, probs_sorted[i]))

    # Top gaps (large jumps suggest natural boundaries)
    gaps.sort(key=lambda x: x[1], reverse=True)
    print(f"  Top-5 probability gaps:")
    for rank, gap, prob in gaps[:5]:
        print(f"    Rank {rank}: gap={gap:.4f}, next_prob={prob:.4f}")

    # Estimate positive rate
    # Method 1: Count samples above probability 0.5
    above_half = (probs >= 0.5).sum()
    # Method 2: Elbow point
    # Method 3: Test distribution estimate
    est_rates = [
        ('prob>=0.5', above_half / len(probs)),
    ]
    print(f"  Estimated positive rates:")
    for name, rate in est_rates:
        print(f"    {name}: {rate:.3f} ({int(rate * len(probs))} samples)")

    return {
        'elbow_idx': elbow_idx,
        'elbow_prob': float(probs_sorted[elbow_idx]),
        'above_half': int(above_half),
        'mean_prob': float(probs.mean()),
    }


def generate_topk_submissions(probs_dict, template, output_dir, method_name, k_values=None):
    """Generate Top-K submission files for various K values."""
    if k_values is None:
        k_values = list(range(60, 151, 5))  # 60 to 150 in steps of 5

    # Sort by probability (higher first)
    sorted_items = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    max_k = max(k_values)
    top_items = sorted_items[:max_k]

    sub_dir = os.path.join(output_dir, 'topk_submissions', method_name)
    os.makedirs(sub_dir, exist_ok=True)

    for k in k_values:
        top_k_ids = set(iid for iid, _ in top_items[:k])
        sub = template.copy()
        sub['label'] = sub['id'].map(lambda x: 1 if x in top_k_ids else 0)
        path = os.path.join(sub_dir, f'submission_top{k}.csv')
        sub.to_csv(path, index=False)

    return sub_dir


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Optimal Top-K Prediction')
    parser.add_argument('--data_root', type=str, default='.')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Root output dir; stage-specific subdir auto-created.')
    parser.add_argument('--stage', type=str, default='stage2',
                        choices=['stage1', 'stage2'],
                        help='Which test set to predict on.')
    parser.add_argument('--img_size', type=int, default=384,
                        help='Inference resolution (must match training).')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_multiscale_tta', action='store_true',
                        help='Disable multi-scale TTA')
    parser.add_argument('--estimate_only', action='store_true',
                        help='Only analyze distribution, do not save submissions')
    parser.add_argument('--model_filter', type=str, default=None,
                        help='Only use checkpoints from this model subdir (e.g. "convnext_base_fb_in22k_ft_in1k")')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Stage: {args.stage}  |  Img size: {args.img_size}")

    stage_dir = os.path.join(args.output_dir, args.stage)
    os.makedirs(stage_dir, exist_ok=True)

    # Find all checkpoints
    print("\nDiscovering model checkpoints...")
    all_checkpoints = discover_all_checkpoints(args.output_dir, args.data_root)

    # Apply model filter if specified
    if args.model_filter:
        before = len(all_checkpoints)
        all_checkpoints = [(src, path) for src, path in all_checkpoints
                           if args.model_filter in src]
        print(f"  Model filter '{args.model_filter}': {before} → {len(all_checkpoints)} checkpoints")

    if not all_checkpoints:
        print("ERROR: No checkpoints found!")
        return

    print(f"Found {len(all_checkpoints)} checkpoints:")
    for src, path in all_checkpoints:
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  [{src}] {os.path.basename(path)} ({size_mb:.1f} MB)")

    # Load test images
    print(f"\nLoading test images for {args.stage}...")
    test_dataset = StoneDataset(root=args.data_root, split="test",
                                 transforms=None, test_stage=args.stage)
    images = []
    ids = []
    for i in range(len(test_dataset)):
        img = Image.open(test_dataset.samples[i]).convert("RGB")
        images.append(img)
        ids.append(test_dataset.csv_ids[i])
    print(f"Test images: {len(images)}")

    # Prepare TTA transforms
    if args.no_multiscale_tta:
        tta_list = get_tta_transforms(args.img_size)
    else:
        tta_list = get_multiscale_tta(args.img_size)
    print(f"TTA transforms: {len(tta_list)}")

    # Load each model and predict
    all_predictions = []
    model_info = []

    for src_dir, ckpt_path in all_checkpoints:
        print(f"\n--- Loading: {os.path.basename(ckpt_path)} ---")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

            # Infer model name
            if isinstance(checkpoint, dict) and checkpoint.get('model_name'):
                model_name = checkpoint['model_name']
            else:
                model_name = infer_model_name_from_checkpoint(checkpoint)
            print(f"  Architecture: {model_name}")

            val_f1 = None
            if isinstance(checkpoint, dict):
                val_f1 = checkpoint.get('best_f1')
                if val_f1:
                    print(f"  Val F1: {val_f1:.4f}")

            # Create model
            model = create_classifier(num_classes=2, model_name=model_name, pretrained=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            model = model.to(device)
            model.eval()

            # Predict with TTA
            id_to_prob = predict_with_tta(model, images, ids, device, tta_list, args.batch_size)
            all_predictions.append(id_to_prob)
            model_info.append({
                'path': ckpt_path,
                'name': model_name,
                'val_f1': val_f1,
                'src': src_dir
            })

            print(f"  Predicted {len(id_to_prob)} samples")
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  FAILED: {e}")
            continue

    if not all_predictions:
        print("No successful predictions!")
        return

    print(f"\n{'='*60}")
    print(f"ENSEMBLE: {len(all_predictions)} models, {len(tta_list)}-way TTA")
    print(f"{'='*60}")

    # Weights: use validation F1 if available, otherwise 0.9
    weights = [info['val_f1'] if info['val_f1'] else 0.90 for info in model_info]
    print(f"Model weights: {[f'{w:.4f}' for w in weights]}")

    # Ensemble methods
    print("\n=== Ensemble Methods ===")

    ensembles = {}

    # Method 1: Simple average
    ensembles['avg'] = ensemble_average(all_predictions)
    print("  [1] Simple Average")

    # Method 2: Weighted average
    ensembles['weighted'] = ensemble_weighted(all_predictions, weights)
    print("  [2] Weighted Average (by Val F1)")

    # Method 3: Geometric mean
    ensembles['geomean'] = ensemble_geomean(all_predictions)
    print("  [3] Geometric Mean")

    # Method 4: Power mean (p=0.5)
    ensembles['powermean'] = ensemble_power_mean(all_predictions, p=0.5)
    print("  [4] Power Mean (p=0.5)")

    # Method 5: Rank-based
    rank_dict = ensemble_rank(all_predictions)
    ensembles['rank'] = {k: -v for k, v in rank_dict.items()}  # Negate so higher = better
    print("  [5] Rank-based Ensemble")

    _, template_csv_path = resolve_test_paths(args.data_root, stage=args.stage)
    template = pd.read_csv(template_csv_path)
    n_test = len(template)
    print(f"Template: {template_csv_path}  ({n_test} ids)")

    # Analyze distribution for each ensemble method
    best_analysis = None
    best_method = None
    best_probs = None

    for name, probs_dict in ensembles.items():
        print(f"\n--- {name.upper()} Analysis ---")
        analysis = analyze_distribution(probs_dict)
        if best_analysis is None or analysis['above_half'] > best_analysis.get('above_half', 0):
            best_analysis = analysis
            best_method = name
            best_probs = probs_dict

    print(f"\n=== Best Method: {best_method.upper()} ===")
    print(f"  Elbow at rank {best_analysis['elbow_idx']}")
    print(f"  Prob at elbow: {best_analysis['elbow_prob']:.4f}")
    print(f"  Samples with prob>=0.5: {best_analysis['above_half']}")

    if args.estimate_only:
        print("\n[estimate_only=True] — Skipping submission generation")
        return

    # Generate submissions for each ensemble method
    print("\n=== Generating Submissions ===")

    # Key K values to try on Kaggle (denser around stage2 sweet spot)
    k_values = [70, 80, 90, 95, 100, 102, 105, 110, 115, 120, 125,
                130, 135, 140, 145, 150, 155, 160]

    for name, probs_dict in ensembles.items():
        sub_dir = generate_topk_submissions(probs_dict, template, stage_dir, name, k_values)
        print(f"  {name}: saved to {sub_dir}/")

    # Generate default submission using best method with estimated K
    est_k = best_analysis['elbow_idx']
    est_k_rounded = int(round(est_k / 5) * 5)
    est_k_rounded = max(70, min(160, est_k_rounded))

    # Use the best ensemble method
    default_probs = ensembles[best_method]
    sorted_items = sorted(default_probs.items(), key=lambda x: x[1], reverse=True)

    top_k_ids = set(iid for iid, _ in sorted_items[:est_k_rounded])
    sub = template.copy()
    sub['label'] = sub['id'].map(lambda x: 1 if x in top_k_ids else 0)

    default_path = os.path.join(stage_dir, 'submission.csv')
    sub.to_csv(default_path, index=False)
    print(f"\nDefault submission (method={best_method}, top-K={est_k_rounded}): {default_path}")
    print(f"  Positive predictions: {sub['label'].sum()} / {len(sub)} ({100*sub['label'].mean():.1f}%)")

    # Also save probability distribution for analysis
    probs_sorted = sorted(default_probs.items(), key=lambda x: x[1], reverse=True)
    probs_df = pd.DataFrame([
        {'rank': i+1, 'id': iid, 'prob': prob}
        for i, (iid, prob) in enumerate(probs_sorted)
    ])
    probs_path = os.path.join(stage_dir, 'ensemble_probs.csv')
    probs_df.to_csv(probs_path, index=False)
    print(f"Probabilities saved to: {probs_path}")

    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print(f"  Best ensemble method: {best_method}")
    print(f"  Recommended K range: {est_k_rounded - 10} to {est_k_rounded + 15}")
    print(f"  Try these on Kaggle:")
    for k in [est_k_rounded - 10, est_k_rounded - 5, est_k_rounded,
              est_k_rounded + 5, est_k_rounded + 10]:
        k = max(70, min(160, k))
        print(f"    {args.stage}/topk_submissions/{best_method}/submission_top{k}.csv")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
