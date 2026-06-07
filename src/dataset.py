"""
Dataset module for meteorite classification.
Includes advanced data augmentation, CutMix/Mixup, and high-resolution support.
"""
import os
import random
import math

import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps


# ─────────────────────────────────────────────
#  Helper Classes
# ─────────────────────────────────────────────

class GaussianBlur:
    """Gaussian blur transform for PIL images."""

    def __init__(self, sigma_range=(0.1, 2.0)):
        self.sigma_range = sigma_range

    def __call__(self, img):
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def __repr__(self):
        return f'GaussianBlur(sigma_range={self.sigma_range})'


class RandomSharpness:
    """Randomly adjust sharpness of the image (simulates weathering/texture variation)."""

    def __init__(self, factor_range=(0.5, 2.0), p=0.3):
        self.factor_range = factor_range
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            factor = random.uniform(self.factor_range[0], self.factor_range[1])
            return ImageEnhance.Sharpness(img).enhance(factor)
        return img


class RandomAutoContrast:
    """Randomly apply auto contrast (simulates different lighting on meteorite surfaces)."""

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageEnhance.Contrast(img).enhance(
                random.uniform(0.8, 1.3)
            )
        return img


class RandomEqualize:
    """Randomly equalize histogram (helps with varied rock textures)."""

    def __call__(self, img):
        if random.random() < 0.15:
            return ImageOps.equalize(img)
        return img


class Solarize:
    """Solarize transform (inverts pixels above threshold)."""

    def __init__(self, threshold=128, p=0.1):
        self.threshold = threshold
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            from PIL import ImageOps
            return ImageOps.solarize(img, self.threshold)
        return img


# ─────────────────────────────────────────────
#  CutMix / Mixup (in tensor space)
# ─────────────────────────────────────────────

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix: cut a patch from one image and paste it onto another.

    Args:
        x: batch of images [B, C, H, W]
        y: batch of labels [B]
        alpha: beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam


def mixup_data(x, y, alpha=0.4):
    """
    Mixup: linearly interpolate between two images and their labels.

    Args:
        x: batch of images [B, C, H, W]
        y: batch of labels [B]
        alpha: beta distribution parameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss for Mixup/CutMix."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────
#  Transform Pipeline Functions
# ─────────────────────────────────────────────

def get_train_transforms(img_size=224, augmentation_level='medium'):
    """
    Get training data augmentation transforms.

    Uses Resize instead of RandomCrop to match test preprocessing.

    Args:
        img_size: Target image size
        augmentation_level: 'light', 'medium', or 'heavy'

    Returns:
        transforms.Compose
    """
    geometric_transforms = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ]

    if augmentation_level == 'light':
        color_transforms = [
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    elif augmentation_level == 'medium':
        color_transforms = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([GaussianBlur(sigma_range=(0.1, 2.0))], p=0.3),
        ]
    else:  # heavy
        color_transforms = [
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomApply([GaussianBlur(sigma_range=(0.1, 3.0))], p=0.4),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            RandomSharpness(factor_range=(0.5, 2.0), p=0.3),
            RandomAutoContrast(p=0.2),
        ]

    final_transforms = [
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(geometric_transforms + color_transforms + final_transforms)


def get_advanced_train_transforms(img_size=224, use_randaugment=True, use_trivialaugment=False):
    """
    Get advanced training transforms with RandAugment or TrivialAugment.

    Args:
        img_size: Target image size
        use_randaugment: Use RandAugment (recommended)
        use_trivialaugment: Use TrivialAugment instead

    Returns:
        transforms.Compose
    """
    geometric = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
    ]

    auto_augment = []
    if use_randaugment:
        auto_augment = [transforms.RandAugment(num_ops=2, magnitude=9)]
    elif use_trivialaugment:
        auto_augment = [transforms.TrivialAugmentWide()]

    color = [
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomApply([GaussianBlur(sigma_range=(0.1, 2.0))], p=0.3),
    ]

    final = [
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(geometric + auto_augment + color + final)


def get_train_transforms_v2(img_size=224, level='heavy'):
    """
    NEW v2 training transforms with stronger meteorite-specific augmentation.
    Includes Cutout, sharpness variation, and solarization for rock texture diversity.

    Levels: 'medium', 'heavy', 'extreme'
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if level == 'medium':
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.92, 1.08)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomApply([GaussianBlur(sigma_range=(0.1, 2.0))], p=0.3),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=mean, std=std),
        ]
    elif level == 'heavy':
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.25),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.15),
            transforms.RandomApply([GaussianBlur(sigma_range=(0.1, 2.5))], p=0.35),
            RandomSharpness(factor_range=(0.5, 2.0), p=0.3),
            RandomAutoContrast(p=0.2),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.18), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:  # extreme
        t = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.35),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.35),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            transforms.RandomApply([GaussianBlur(sigma_range=(0.1, 3.0))], p=0.4),
            RandomSharpness(factor_range=(0.3, 2.5), p=0.4),
            RandomAutoContrast(p=0.3),
            Solarize(threshold=128, p=0.1),
            transforms.RandomApply([transforms.Grayscale(num_output_channels=3)], p=0.15),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.4, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(t)


def get_val_transforms(img_size=224):
    """Get validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


import torchvision.transforms.functional as TF


def get_tta_transforms(img_size=224):
    """
    Get test-time augmentation (TTA) transforms.
    Returns a list of different transform pipelines for ensemble prediction.

    Uses fixed-angle rotations (no randomness) and multiple scales.

    Args:
        img_size: Target image size

    Returns:
        List of transforms.Compose
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)

    def fixed_rotate(img, angle):
        return TF.rotate(img, angle)

    tta_list = [
        # 0: Original (same as val_transforms)
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 1: Horizontal flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 2: Vertical flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 3: Horizontal + Vertical flip
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            normalize,
        ]),
        # 4: Fixed +15 degree rotation
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: fixed_rotate(img, 15)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 5: Fixed -15 degree rotation
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: fixed_rotate(img, -15)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 6: 90 degree rotation
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: fixed_rotate(img, 90)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 7: 180 degree rotation
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: fixed_rotate(img, 180)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 8: 270 degree rotation
        transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(lambda img: fixed_rotate(img, 270)),
            transforms.ToTensor(),
            normalize,
        ]),
        # 9: Multi-scale: resize to img_size+32 then center-crop
        transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]),
    ]

    return tta_list


def resolve_test_paths(root, stage="stage2"):
    """Locate the test image directory and submission template for a given stage.

    Layout convention:
      <root>/test_images_<stage>/test_images/      (image files)
      <root>/sample_submission_<stage>.csv         (id template)

    Falls back to legacy unprefixed paths (test_images/, sample_submission.csv)
    if the stage-suffixed ones are absent.
    """
    candidates = [
        (
            os.path.join(root, f"test_images_{stage}", "test_images"),
            os.path.join(root, f"sample_submission_{stage}.csv"),
        ),
        (
            os.path.join(root, "test_images", "test_images"),
            os.path.join(root, "sample_submission.csv"),
        ),
    ]
    for image_dir, csv_path in candidates:
        if os.path.isdir(image_dir) and os.path.exists(csv_path):
            return image_dir, csv_path
    # Return the stage-suffixed paths for clearer error messages
    return candidates[0]


def resolve_train_paths(root, use_nobg=False):
    """Locate the train image directory and labels CSV.

    If use_nobg=True, prefer <root>/train_images_nobg/train_images/ when it exists.
    """
    if use_nobg:
        nobg_dir = os.path.join(root, "train_images_nobg", "train_images")
        if os.path.isdir(nobg_dir):
            return nobg_dir, os.path.join(root, "train_labels.csv")
    return (
        os.path.join(root, "train_images", "train_images"),
        os.path.join(root, "train_labels.csv"),
    )


class StoneDataset(Dataset):
    """Dataset class for stone (meteorite/non-meteorite) images."""

    def __init__(self, root, split="train", transforms=None,
                 test_stage="stage2", use_nobg=False):
        """
        Args:
            root: 数据集根目录
            split: 'train' 或 'test'
            transforms: 图像预处理变换
            test_stage: 'stage1' or 'stage2' — selects which test set to use
            use_nobg: if True and split=='train', use background-removed train images
        """
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'test'.")

        self.root = root
        self.split = split
        self.test_stage = test_stage
        self.use_nobg = use_nobg
        self.transforms = transforms
        self.samples = []
        self.labels = []
        self.csv_ids = []  # Store original CSV ids for test samples

        if split == "train":
            self._load_train_samples()
        else:
            self._load_test_samples()

    def _load_train_samples(self):
        """Load training samples from train_images and train_labels.csv."""
        image_dir, csv_path = resolve_train_paths(self.root, use_nobg=self.use_nobg)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_path} not found.")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"{image_dir} not found.")

        df = pd.read_csv(csv_path)
        if "id" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must contain 'id' and 'label' columns.")

        skipped = 0
        alt_ext_count = 0
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row["id"])
            if not os.path.exists(img_path):
                # Try alternate extension: .jpg <-> .jpeg
                base, ext = os.path.splitext(img_path)
                if ext.lower() == '.jpg':
                    alt_path = base + '.jpeg'
                elif ext.lower() == '.jpeg':
                    alt_path = base + '.jpg'
                else:
                    alt_path = None
                if alt_path and os.path.exists(alt_path):
                    img_path = alt_path
                    alt_ext_count += 1
                else:
                    skipped += 1
                    continue
            self.samples.append(img_path)
            self.labels.append(int(row["label"]))

        print(f"Loaded {len(self.samples)} training samples "
              f"({alt_ext_count} loaded via alternate extension, {skipped} skipped).")

    def _load_test_samples(self):
        """Load test samples for the configured stage."""
        image_dir, template_csv_path = resolve_test_paths(self.root, stage=self.test_stage)

        if not os.path.exists(template_csv_path):
            raise FileNotFoundError(f"{template_csv_path} not found.")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"{image_dir} not found.")

        df = pd.read_csv(template_csv_path)
        if "id" not in df.columns:
            raise ValueError("CSV must contain 'id' column.")

        skipped = 0
        alt_ext_count = 0
        for _, row in df.iterrows():
            img_path = os.path.join(image_dir, row["id"])
            if not os.path.exists(img_path):
                # Try alternate extension: .jpg <-> .jpeg
                base, ext = os.path.splitext(img_path)
                if ext.lower() == '.jpg':
                    alt_path = base + '.jpeg'
                elif ext.lower() == '.jpeg':
                    alt_path = base + '.jpg'
                else:
                    alt_path = None
                if alt_path and os.path.exists(alt_path):
                    img_path = alt_path
                    alt_ext_count += 1
                else:
                    skipped += 1
                    continue
            self.samples.append(img_path)
            self.labels.append(None)
            self.csv_ids.append(row["id"])  # Store original CSV id for submission matching

        print(f"Loaded {len(self.samples)} test samples [{self.test_stage}] "
              f"({alt_ext_count} loaded via alternate extension, {skipped} skipped).")

    def __getitem__(self, index):
        img_path = self.samples[index]
        label = self.labels[index]

        image = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)

        if self.split == "test":
            # Return original CSV id for correct submission matching
            csv_id = self.csv_ids[index] if self.csv_ids else os.path.basename(img_path)
            return image, csv_id
        return image, label

    def __len__(self):
        return len(self.samples)


def create_dataloaders(data_root, batch_size=32, img_size=224, num_workers=4,
                       val_split=0.2, augmentation_level='medium', use_randaugment=False):
    """
    Create train and validation dataloaders.

    Args:
        data_root: 数据集根目录
        batch_size: 批次大小
        img_size: 图像大小
        num_workers: 数据加载线程数
        val_split: 验证集比例
        augmentation_level: 数据增强级别 ('light', 'medium', 'heavy')
        use_randaugment: 是否使用RandAugment

    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import random_split, Subset

    # Choose transforms based on settings
    if use_randaugment:
        train_transforms = get_advanced_train_transforms(img_size, use_randaugment=True)
    else:
        train_transforms = get_train_transforms(img_size, augmentation_level)

    val_transforms = get_val_transforms(img_size)

    # Create a base dataset without transforms to get indices and labels
    base_dataset = StoneDataset(
        root=data_root,
        split="train",
        transforms=None
    )

    # Stratified split: ensures train/val have same class ratio as full dataset
    indices = list(range(len(base_dataset)))
    labels = base_dataset.labels

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_indices, val_indices = next(sss.split(indices, labels))
    train_indices = train_indices.tolist()
    val_indices = val_indices.tolist()

    print(f"Stratified split: {len(train_indices)} train, {len(val_indices)} val")

    # Create separate datasets with appropriate transforms
    train_dataset = Subset(
        StoneDataset(root=data_root, split="train", transforms=train_transforms),
        train_indices
    )

    val_dataset = Subset(
        StoneDataset(root=data_root, split="train", transforms=val_transforms),
        val_indices
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_test_dataloader(data_root, batch_size=32, img_size=224, num_workers=4):
    """
    Create test dataloader.

    Args:
        data_root: 数据集根目录
        batch_size: 批次大小
        img_size: 图像大小
        num_workers: 数据加载线程数

    Returns:
        test_loader
    """
    test_dataset = StoneDataset(
        root=data_root,
        split="test",
        transforms=get_val_transforms(img_size)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


if __name__ == "__main__":
    # Test the dataset
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    data_root = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(data_root, "train_labels.csv")):
        data_root = os.path.dirname(data_root)

    # Test different augmentation levels
    print("Testing augmentation transforms...")

    for level in ['light', 'medium', 'heavy']:
        transforms = get_train_transforms(img_size=224, augmentation_level=level)
        dataset = StoneDataset(root=data_root, split="train", transforms=transforms)
        image, label = dataset[0]
        print(f"  {level}: Image shape {image.shape}, Label {label}")

    # Test RandAugment
    transforms = get_advanced_train_transforms(img_size=224, use_randaugment=True)
    dataset = StoneDataset(root=data_root, split="train", transforms=transforms)
    image, label = dataset[0]
    print(f"  RandAugment: Image shape {image.shape}, Label {label}")

    print(f"\nTotal dataset size: {len(dataset)}")