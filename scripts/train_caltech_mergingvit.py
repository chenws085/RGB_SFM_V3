#!/usr/bin/env python3
"""Train the repository MergingViT on the deterministic Caltech-101 split."""

from __future__ import annotations

import argparse
import json
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataloader.Caltech101 import Caltech101Dataset  # noqa: E402
from models.MergingViT import MergingViT  # noqa: E402


MODEL_ARGS = {
    "img_size": 224,
    "patch_size": 8,
    "in_chans": 3,
    "num_classes": 101,
    "embed_dims": [32, 64, 128, 256],
    "num_heads": [2, 4, 8, 16],
    "depths": [1, 1, 1, 1],
    "merge_size": [(2, 2), (2, 2), (2, 2)],
    "drop_rate": 0.1,
    "drop_path_rate": 0.3,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CANVAS_RGB = tuple(round(value * 255) for value in IMAGENET_MEAN)
CONTENT_MAX_SIZE = 160


class CenterLetterbox:
    """Keep the full source image in a centered 160-in-224 safe canvas."""

    def __init__(self, content_size=160):
        self.content_size = content_size

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = min(self.content_size / width, self.content_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = TF.resize(image, [resized_height, resized_width], antialias=True)
        canvas = Image.new("RGB", (224, 224), CANVAS_RGB)
        canvas.paste(resized, ((224 - resized_width) // 2, (224 - resized_height) // 2))
        return canvas


class AlignedTranslation:
    """Translate the resized canvas before the original training augmentations."""
    def __init__(self, shifts=(8, 16, 32)):
        self.shifts = tuple(shifts)

    def __call__(self, canvas):
        choices = [(0, 0)] + [(dx*s, dy*s) for s in self.shifts
                               for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))]
        dx, dy = choices[random.randrange(len(choices))]
        translated = Image.new("RGB", canvas.size, CANVAS_RGB)
        translated.paste(canvas, (dx,dy))
        return translated


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs/train/mergingvit_caltech101_letterbox160")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aligned-translation", action="store_true")
    parser.add_argument("--translation-shifts", type=int, nargs="+", help="Override augmentation magnitudes, e.g. 1 2 3 4 5 6 7")
    parser.add_argument("--content-size", type=int, default=160)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.aligned_translation and args.translation_shifts is not None:
        parser.error("Use aligned-translation or translation-shifts, not both")
    if args.translation_shifts is not None and (min(args.translation_shifts) < 1 or len(set(args.translation_shifts)) != len(args.translation_shifts)):
        parser.error("translation-shifts must be distinct positive integers")
    max_shift = max(args.translation_shifts or ((8, 16, 32) if args.aligned_translation else (0,)))
    if not 1 <= args.content_size <= 224 - 2 * max_shift:
        parser.error("content-size and translation exceed safe canvas margin")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        CenterLetterbox(args.content_size),
        *([AlignedTranslation(args.translation_shifts or (8, 16, 32))] if args.aligned_translation or args.translation_shifts else []),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),
    ])
    valid_transform = transforms.Compose([
        CenterLetterbox(args.content_size),
        transforms.ToTensor(),
        normalize,
    ])
    train_set = Caltech101Dataset(
        args.data_root, train=True, transform=train_transform, seed=args.seed
    )
    valid_set = Caltech101Dataset(
        args.data_root, train=False, transform=valid_transform, seed=args.seed
    )
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, **common)
    valid_loader = DataLoader(valid_set, shuffle=False, **common)
    return train_loader, valid_loader


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    airplane_correct = 0
    airplane_count = 0
    airplane_idx = loader.dataset.classes.index("airplanes")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * len(images)
        predictions = logits.argmax(1)
        correct += (predictions == labels).sum().item()
        count += len(images)
        mask = labels == airplane_idx
        airplane_correct += ((predictions == labels) & mask).sum().item()
        airplane_count += mask.sum().item()
    return {
        "loss": loss_sum / count,
        "accuracy": correct / count,
        "airplanes_accuracy": airplane_correct / max(airplane_count, 1),
        "samples": count,
        "airplanes_samples": airplane_count,
    }


def main():
    args = parse_args()
    torch.set_num_threads(4)
    set_seed(args.seed)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run_config.json").exists():
        raise FileExistsError(f"Existing run: {output_dir}")
    device = torch.device(args.device)
    train_loader, valid_loader = make_loaders(args)
    model = MergingViT(**MODEL_ARGS).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    best_accuracy = -1.0
    stale_epochs = 0
    history = []

    config = {
        "dataset": "Caltech101",
        "preprocessing": "safe-letterbox, full source resized to content_max_size inside 224 canvas",
        "content_max_size": args.content_size,
        "input_shape": [224, 224],
        "model": {"name": "MergingViT", "args": MODEL_ARGS},
        "train_samples": len(train_loader.dataset),
        "valid_samples": len(valid_loader.dataset),
        "train_indices": train_loader.dataset.indices,
        "valid_indices": valid_loader.dataset.indices,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "translation_choices": [[0,0]] + [[dx*s,dy*s] for s in (args.translation_shifts or (8,16,32)) for dx,dy in ((-1,0),(1,0),(0,-1),(0,1))] if args.aligned_translation or args.translation_shifts else [[0,0]],
        "initialization": "from scratch",
        **vars(args),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(images)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(images)
        scheduler.step()
        validation = evaluate(model, valid_loader, criterion, device)
        row = {
            "epoch": epoch + 1,
            "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"valid_{key}": value for key, value in validation.items()},
        }
        history.append(row)
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(json.dumps(row), flush=True)

        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            stale_epochs = 0
            torch.save({
                "model_weights": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "valid_acc": validation["accuracy"],
                "airplanes_valid_acc": validation["airplanes_accuracy"],
                "model_args": MODEL_ARGS,
                "seed": args.seed,
            }, output_dir / "MergingViT_best.pth")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop epoch={epoch + 1}", flush=True)
                break

    print(f"best_valid_accuracy={best_accuracy:.6f}", flush=True)


if __name__ == "__main__":
    main()
