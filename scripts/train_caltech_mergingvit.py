#!/usr/bin/env python3
"""Train the repository MergingViT on the deterministic Caltech-101 split."""

from __future__ import annotations

import argparse
import json
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

    def __call__(self, image):
        image = image.convert("RGB")
        width, height = image.size
        scale = min(CONTENT_MAX_SIZE / width, CONTENT_MAX_SIZE / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = TF.resize(image, [resized_height, resized_width], antialias=True)
        canvas = Image.new("RGB", (224, 224), CANVAS_RGB)
        canvas.paste(resized, ((224 - resized_width) // 2, (224 - resized_height) // 2))
        return canvas


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
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders(args):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        CenterLetterbox(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.25), ratio=(0.3, 3.3), value="random"),
    ])
    valid_transform = transforms.Compose([
        CenterLetterbox(),
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
    set_seed(args.seed)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        "preprocessing": "centered safe-letterbox, full source resized to <=160 inside 224 canvas",
        "content_max_size": CONTENT_MAX_SIZE,
        "input_shape": [224, 224],
        "model": {"name": "MergingViT", "args": MODEL_ARGS},
        "train_samples": len(train_loader.dataset),
        "valid_samples": len(valid_loader.dataset),
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
