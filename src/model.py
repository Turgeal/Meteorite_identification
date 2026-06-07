"""
Generic model module for image classification.
Supports any timm model (EfficientNet, ResNet, Swin, ConvNeXt, etc.).
Handles loading pretrained weights from local files or online.
"""
import os
import torch
import torch.nn as nn
import timm


def create_classifier(num_classes=2, model_name='efficientnet_b0.ra_in1k',
                      pretrained=True, pretrained_dir=None,
                      dropout=0.3, drop_path_rate=0.1):
    """
    Create any timm-supported classification model.
    Downloads pretrained weights from HuggingFace if not available locally.

    Args:
        num_classes: Number of output classes
        model_name: Any valid timm model name
        pretrained: Whether to use pretrained weights
        pretrained_dir: Local directory containing pretrained weight files (optional)
        dropout: Head dropout rate
        drop_path_rate: Stochastic depth rate

    Returns:
        model: PyTorch model
    """
    print(f"Creating model: {model_name}")
    print(f"  dropout={dropout}, drop_path_rate={drop_path_rate}")

    # Configure HuggingFace mirror for faster download in China
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    # Create model with pretrained weights (timm will download automatically)
    try:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )
        if pretrained:
            print(f"[OK] Loaded pretrained {model_name}")
        return model
    except Exception as e:
        print(f"Warning: Failed to load pretrained model ({str(e)[:100]})")
        if not pretrained:
            raise
        # Fall back: create without pretrained
        print("Retrying without pretrained weights...")
        model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout,
            drop_path_rate=drop_path_rate
        )
        return model


# Backwards compatibility alias
def create_swin_transformer(num_classes=2, model_name=None, pretrained=True,
                            pretrained_dir=None, dropout=0.0, drop_path_rate=0.0):
    """Backwards-compatible wrapper. Use create_classifier() for new code."""
    if model_name is None:
        model_name = 'swin_tiny_patch4_window7_224'
    return create_classifier(num_classes, model_name, pretrained, pretrained_dir, dropout, drop_path_rate)


def save_model(model, optimizer, epoch, loss, save_path, best_acc=None,
               best_f1=None, best_threshold=None, model_name=None,
               best_val_f1_fixed=None):
    """Save model checkpoint with metadata."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if best_acc is not None:
        checkpoint['best_acc'] = best_acc
    if best_f1 is not None:
        checkpoint['best_f1'] = best_f1  # F1 at optimal threshold (primary metric)
    if best_threshold is not None:
        checkpoint['best_threshold'] = best_threshold
    if model_name is not None:
        checkpoint['model_name'] = model_name
    if best_val_f1_fixed is not None:
        checkpoint['best_val_f1_fixed'] = best_val_f1_fixed  # F1 at threshold=0.5

    torch.save(checkpoint, save_path)


def load_model(checkpoint_path, model, device='cuda'):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model


def infer_model_name_from_checkpoint(checkpoint):
    """Infer model architecture from checkpoint state_dict keys."""
    if isinstance(checkpoint, dict):
        saved_name = checkpoint.get('model_name')
        if saved_name and isinstance(saved_name, str) and len(saved_name) > 3:
            return saved_name
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_keys = list(state_dict.keys())
    else:
        state_keys = []
    if any('stem.conv' in k or 'stages.0' in k for k in state_keys):
        if any('layernorm' in k or 'gamma' in k for k in state_keys):
            return 'convnext_small.fb_in22k_ft_in1k'
    if any('conv_stem' in k or 'blocks.0.0' in k for k in state_keys):
        if any('bn1' in k for k in state_keys):
            return 'efficientnet_b4.ra2_in1k'
        return 'efficientnet_b0.ra_in1k'
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
