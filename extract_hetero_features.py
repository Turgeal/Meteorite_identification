"""Extract heterogeneous frozen features from DINOv2 and CLIP.

Outputs (under outputs/features/):
  dinov2_train.npy   (5098, 1024)   DINOv2-ViT-L/14 CLS embedding
  dinov2_stage2.npy  (194,  1024)
  dinov2_stage1.npy  (511,  1024)   optional sanity-check
  clip_image_train.npy   (5098, 768)   CLIP-ViT-L/14 image embedding (normalized)
  clip_image_stage2.npy  (194,  768)
  clip_image_stage1.npy  (511,  768)
  clip_zeroshot_train.csv     id, score      -- 5pos/5neg prompt average (cosine, tau-scaled softmax)
  clip_zeroshot_stage2.csv    id, score
  clip_zeroshot_stage1.csv    id, score
  feature_extraction.log      run summary

Why frozen, not fine-tuned: see DINOv2 paper (Oquab et al. 2023) and
"Fine-Tuning can Distort Pretrained Features" (Kumar et al. 2022).
On OOD splits like ours (museum train -> field stage2), full fine-tune
distorts the pretrained features and underperforms linear probing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CODE = Path(__file__).parent.resolve()
sys.path.insert(0, str(CODE))
from src.dataset import resolve_test_paths, resolve_train_paths


CLIP_PROMPTS_POS = [
    "a photo of a meteorite",
    "a meteorite with fusion crust",
    "an iron meteorite specimen",
    "a chondrite space rock",
    "a meteorite isolated on white background",
    "a meteorite specimen photograph",
]
CLIP_PROMPTS_NEG = [
    "a photo of a terrestrial rock",
    "an ordinary stone or pebble",
    "a piece of basalt or granite",
    "a volcanic rock sample",
    "a rock isolated on white background",
    "a common rock specimen",
]


class ImageListDataset(Dataset):
    def __init__(self, paths, ids, transform):
        self.paths = paths
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.ids[i]


def list_train(data_root, use_nobg=True):
    img_dir, csv = resolve_train_paths(data_root, use_nobg=use_nobg)
    df = pd.read_csv(csv)
    paths, ids, labels = [], [], []
    for _, row in df.iterrows():
        p = os.path.join(img_dir, row["id"])
        if not os.path.exists(p):
            stem, ext = os.path.splitext(p)
            for alt in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                if os.path.exists(stem + alt):
                    p = stem + alt
                    break
            else:
                continue
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
            stem, ext = os.path.splitext(p)
            for alt in (".jpg", ".jpeg", ".JPG", ".JPEG"):
                if os.path.exists(stem + alt):
                    p = stem + alt
                    break
            else:
                continue
        paths.append(p)
        ids.append(row["id"])
    return paths, ids


def extract_dinov2(model, loader, device):
    feats, ids_out = [], []
    model.eval()
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model.forward_features(x)
                if isinstance(out, dict):
                    cls = out.get("x_norm_clstoken", None)
                    if cls is None:
                        cls = out.get("cls_token", None)
                    if cls is None:
                        tokens = out.get("x", None) or out.get("x_norm", None)
                        cls = tokens[:, 0] if tokens is not None else None
                else:
                    cls = out[:, 0] if out.ndim == 3 else out
            feats.append(cls.float().cpu().numpy())
            ids_out.extend(idx if isinstance(idx, list) else list(idx))
    return np.concatenate(feats, axis=0), ids_out


def extract_clip(model, tokenizer, loader, device):
    pos_tokens = tokenizer(CLIP_PROMPTS_POS).to(device)
    neg_tokens = tokenizer(CLIP_PROMPTS_NEG).to(device)
    model.eval()
    with torch.no_grad():
        text_pos = model.encode_text(pos_tokens)
        text_neg = model.encode_text(neg_tokens)
        text_pos = text_pos / text_pos.norm(dim=-1, keepdim=True)
        text_neg = text_neg / text_neg.norm(dim=-1, keepdim=True)
        text_pos = text_pos.mean(dim=0, keepdim=True)
        text_neg = text_neg.mean(dim=0, keepdim=True)
        text_pos = text_pos / text_pos.norm(dim=-1, keepdim=True)
        text_neg = text_neg / text_neg.norm(dim=-1, keepdim=True)

    feats, scores, ids_out = [], [], []
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                img = model.encode_image(x)
                img = img / img.norm(dim=-1, keepdim=True)
                sim_pos = (img @ text_pos.T).squeeze(-1)
                sim_neg = (img @ text_neg.T).squeeze(-1)
                tau = 40.0
                logits = torch.stack([sim_neg * tau, sim_pos * tau], dim=-1)
                prob = logits.softmax(dim=-1)[:, 1]
            feats.append(img.float().cpu().numpy())
            scores.append(prob.float().cpu().numpy())
            ids_out.extend(idx if isinstance(idx, list) else list(idx))
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(scores, axis=0),
        ids_out,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=".")
    ap.add_argument("--output_dir", default="outputs/features")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dinov2_model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--clip_arch", default="ViT-L-14")
    ap.add_argument("--clip_pretrained", default="laion2b_s32b_b82k")
    ap.add_argument("--skip_stage1", action="store_true")
    ap.add_argument("--skip_dinov2", action="store_true")
    ap.add_argument("--skip_clip", action="store_true")
    ap.add_argument("--use_nobg", action="store_true", default=True,
                    help="Read train images from train_images_nobg/ (default True; "
                         "stage2 test set is already background-removed, so train must "
                         "match for frozen-feature train/test consistency)")
    ap.add_argument("--no_use_nobg", dest="use_nobg", action="store_false",
                    help="Override: read train images from original train_images/")
    args = ap.parse_args()

    out_dir = Path(args.data_root) / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Listing images...")
    train_paths, train_ids, train_labels = list_train(args.data_root, use_nobg=args.use_nobg)
    print(f"  use_nobg={args.use_nobg}  (train set is "
          f"{'nobg' if args.use_nobg else 'ORIGINAL'} for train/test consistency with stage2)")
    stage2_paths, stage2_ids = list_test(args.data_root, "stage2")
    stage1_paths, stage1_ids = ([], [])
    if not args.skip_stage1:
        try:
            stage1_paths, stage1_ids = list_test(args.data_root, "stage1")
        except Exception as e:
            print(f"  stage1 unavailable: {e}")
    print(f"  train: {len(train_paths)}  stage2: {len(stage2_paths)}  stage1: {len(stage1_paths)}")

    pd.DataFrame({"id": train_ids, "label": train_labels}).to_csv(
        out_dir / "train_ids.csv", index=False
    )

    if not args.skip_dinov2:
        print(f"\n=== DINOv2 ({args.dinov2_model}) ===")
        import timm
        model = timm.create_model(args.dinov2_model, pretrained=True, num_classes=0)
        cfg = timm.data.resolve_model_data_config(model)
        tfm = timm.data.create_transform(**cfg, is_training=False)
        model = model.to(device).eval()

        for split_name, paths, ids in [
            ("train", train_paths, train_ids),
            ("stage2", stage2_paths, stage2_ids),
            ("stage1", stage1_paths, stage1_ids),
        ]:
            if not paths:
                continue
            ds = ImageListDataset(paths, ids, tfm)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
            print(f"  extracting {split_name} ({len(ds)} imgs)...")
            feats, ids_out = extract_dinov2(model, dl, device)
            np.save(out_dir / f"dinov2_{split_name}.npy", feats)
            pd.DataFrame({"id": ids_out}).to_csv(
                out_dir / f"dinov2_{split_name}_ids.csv", index=False
            )
            print(f"    -> dinov2_{split_name}.npy  shape={feats.shape}")
        del model
        torch.cuda.empty_cache()

    if not args.skip_clip:
        print(f"\n=== CLIP ({args.clip_arch} / {args.clip_pretrained}) ===")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_arch, pretrained=args.clip_pretrained
        )
        tokenizer = open_clip.get_tokenizer(args.clip_arch)
        model = model.to(device).eval()

        for split_name, paths, ids in [
            ("train", train_paths, train_ids),
            ("stage2", stage2_paths, stage2_ids),
            ("stage1", stage1_paths, stage1_ids),
        ]:
            if not paths:
                continue
            ds = ImageListDataset(paths, ids, preprocess)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
            print(f"  extracting {split_name} ({len(ds)} imgs)...")
            feats, scores, ids_out = extract_clip(model, tokenizer, dl, device)
            np.save(out_dir / f"clip_image_{split_name}.npy", feats)
            pd.DataFrame({"id": ids_out, "score": scores}).to_csv(
                out_dir / f"clip_zeroshot_{split_name}.csv", index=False
            )
            print(f"    -> clip_image_{split_name}.npy  shape={feats.shape}")
            print(f"       clip_zeroshot_{split_name}.csv  mean_score={scores.mean():.4f}")
        del model
        torch.cuda.empty_cache()

    summary = {
        "dinov2_model": args.dinov2_model,
        "clip_arch": args.clip_arch,
        "clip_pretrained": args.clip_pretrained,
        "n_train": len(train_paths),
        "n_stage2": len(stage2_paths),
        "n_stage1": len(stage1_paths),
    }
    (out_dir / "feature_extraction.log").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nDone. Features written to: {out_dir}")


if __name__ == "__main__":
    main()
