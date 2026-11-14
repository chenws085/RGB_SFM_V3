#!/usr/bin/env python3
"""Evaluate CLIP stage semantics against text labels (Top-1 and Top-5 retrieval)."""
import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ObjectScaleViT import ObjectScaleViT, corresponding_regions
from dataloader.ObjectScalePairs import ObjectScalePairs, feature_grid

def unit(x): return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def main():
    torch.set_num_threads(4)
    folder = ROOT / 'runs/train/object_scale_full_20261031T093902Z/A'
    cfg = json.loads((folder / 'config.json').read_text())
    cp = torch.load(folder / 'best.pth', map_location='cpu', weights_only=False)
    
    model = ObjectScaleViT(**cfg['model_args']).to('cuda:1').eval()
    model.load_state_dict(cp['model'])
    
    # Load canonical image teacher features for fitting the linear mapping
    teacher = torch.load(ROOT / 'datasets/clip_bbox_vitb32_1101.pt', map_location='cpu', weights_only=False)
    
    # Load text features for 101 classes
    text_features = torch.load(ROOT / 'datasets/clip_caltech101_text_features.pt', map_location='cpu', weights_only=False).numpy().astype('float64')
    
    values = {}
    labels_dict = {}
    with torch.inference_mode():
        for split in ['train', 'valid']:
            buckets = [[] for _ in range(4)]
            labels_list = []
            ds = ObjectScalePairs(str(ROOT / 'data'), cfg['splits'][split], training=False)
            for batch in DataLoader(ds, batch_size=32, num_workers=2):
                _, features = model(batch['a'].to('cuda:1'), True)
                for s, f in enumerate(features):
                    r = corresponding_regions(f, feature_grid(batch['points_a'].to('cuda:1'), f, 8 * 2**s), batch['region'].to('cuda:1'))
                    buckets[s].append(r.cpu().numpy())
                labels_list.append(batch['label'].numpy())
            
            values[split] = [np.concatenate(x).astype('float64') for x in buckets]
            labels_dict[split] = np.concatenate(labels_list)
            
    target = teacher['features'][cfg['splits']['train']].numpy().astype('float64')
    valid = teacher['features'][cfg['splits']['valid']].numpy().astype('float64')
    
    rows = []
    print(f"| 層（1起算） | Text Retrieval Top-1 | Text Retrieval Top-5 |")
    print(f"|---|---:|---:|")
    
    for stage in range(4):
        train = values['train'][stage]
        mean = train.mean(0)
        _, _, vt = np.linalg.svd(train - mean, full_matrices=False)
        basis = vt[:32].T
        x = (train - mean) @ basis
        std = np.maximum(x.std(0), 1e-8)
        x = x / std
        xv = ((values['valid'][stage] - mean) @ basis) / std
        
        x = np.c_[x, np.ones(len(x))]
        xv = np.c_[xv, np.ones(len(xv))]
        penalty = np.eye(33) * 10
        penalty[-1, -1] = 0
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ target)
        z = unit(xv @ weights)
        
        # text_features is [101, 512]
        # z is [N, 512]
        # similarities: [N, 101]
        sims = z @ text_features.T
        
        # Get labels: [N]
        valid_labels = labels_dict['valid']
        
        # Top-1 accuracy
        top1_preds = sims.argmax(axis=-1)
        top1_acc = (top1_preds == valid_labels).mean()
        
        # Top-5 accuracy
        top5_preds = np.argsort(sims, axis=-1)[:, -5:] # [N, 5]
        top5_acc = np.any(top5_preds == valid_labels[:, None], axis=-1).mean()
        
        rows.append(dict(stage=stage, top1=float(top1_acc), top5=float(top5_acc)))
        print(f"| {stage+1} | {top1_acc*100:.2f}% | {top5_acc*100:.2f}% |", flush=True)

if __name__ == '__main__':
    main()
