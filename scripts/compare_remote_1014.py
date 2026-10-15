#!/usr/bin/env python3
"""Verify HDD intake and compare remote 10/14 checkpoints with local experiments."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from torchvision.datasets import Caltech101

import evaluate_caltech_scale as scale

ROOT = scale.ROOT
HDD = Path('/mnt/d/RGB_SFM_artifacts')
OUT = ROOT/'plots/kmeans/caltech_remote_1014_comparison'
REMOTE = {'remote_baseline': ('mergingvit_caltech101_letterbox160', 'baseline'),
          'aligned_1014': ('mergingvit_caltech101_aligned_translation_1014', 'aligned_8_16_32')}


def rows(path):
    with path.open(newline='') as stream: return list(csv.DictReader(stream))


def mean(records, key):
    return float(np.mean([float(row[key]) for row in records]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu'); p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(4); faiss.omp_set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    scale.dump(OUT/'status.json', dict(status='verifying_intake', started_utc=datetime.now(timezone.utc).isoformat()))
    train, valid = scale.split_indices(str(ROOT/'data'), 42)
    train, valid = sorted(train), sorted(valid)
    intake = []
    for name, (folder, archived_name) in REMOTE.items():
        source = HDD/'weights/remote140165'/folder
        checkpoint = source/'MergingViT_best.pth'
        archived = json.loads((ROOT/'plots/kmeans/caltech_non_aligned_1015'/archived_name/'run_manifest.json').read_text())
        checksum = scale.sha(checkpoint)
        assert checksum == archived['checkpoint_sha256'], 'Remote checkpoint differs from historical evaluation'
        assert train == sorted(archived['train_indices']) and valid == sorted(archived['valid_indices'])
        config = json.loads((source/'run_config.json').read_text())
        if 'train_indices' in config:
            assert sorted(config['train_indices']) == train and sorted(config['valid_indices']) == valid
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
        intake.append(dict(name=name, path=str(checkpoint), sha256=checksum, epoch=ckpt['epoch'],
            original_validation_accuracy=ckpt['valid_acc'], config_sha256=scale.sha(source/'run_config.json'),
            history_sha256=scale.sha(source/'history.json'), historical_checkpoint_match=True,
            split_evidence='run_config and historical evaluation' if 'train_indices' in config else 'historical evaluation manifest (old config has no indices)'))
    for filename, expected in [('101_ObjectCategories.tar.gz','b224c7392d521a49829488ab0f1120d9'),
                               ('Annotations.tar','6f83eeb1f24d99cab4eb377263132c91')]:
        path = HDD/'datasets/remote140165/caltech101_official_archives'/filename
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        assert actual == expected
        intake.append(dict(dataset_archive=str(path), md5=actual, sha256=scale.sha(path), official_md5_match=True))
    scale.dump(OUT/'intake_manifest.json', intake)
    base = Caltech101(str(ROOT/'data'), target_type=['category','annotation'], download=False)
    config = SimpleNamespace(output_dir=OUT/'scale', device=a.device, workers=a.workers, clusters=31, nredo=50)
    for name, (folder, _) in REMOTE.items():
        scale.dump(OUT/'status.json', dict(status='evaluating_scale', model=name))
        scale.evaluate(name, HDD/'weights/remote140165'/folder/'MergingViT_best.pth', config, base, train, valid)

    models = {name: OUT/'scale'/name for name in REMOTE}
    models.update({name: ROOT/'plots/kmeans/caltech_scale_1015'/name for name in ('S0','S1','T1')})
    combined, per_image = [], {}
    for name, directory in models.items():
        manifest = json.loads((directory/'run_manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert manifest['train_indices'] == train and manifest['valid_indices'] == valid
        # Reuse local results only under the same extraction and fitting protocol.
        assert manifest['runner_sha256'] == scale.sha(scale.__file__)
        assert manifest['K'] == 31 and manifest['nredo'] == 50
        classification = rows(directory/'classification.csv'); summary = rows(directory/'summary.csv')
        original = next(r for r in classification if r['category']=='all' and int(r['content_size'])==160)
        unseen = [r for r in classification if r['category']=='all' and int(r['content_size']) in (112,144,176)]
        region = [r for r in summary if r['stage']=='2' and r['readout']=='bbox' and int(r['content_size']) in (112,144,176)]
        combined.append(dict(model=name, original_accuracy=float(original['accuracy']),
            unseen_scale_accuracy=mean(unseen,'accuracy'), unseen_scale_macro_accuracy=mean(unseen,'macro_accuracy'),
            stage2_bbox_agreement=mean(region,'cluster_agreement'), stage2_bbox_cosine=mean(region,'cosine'),
            stage2_bbox_correspondence_gap=mean(region,'correspondence_gap'),
            used_clusters=mean(region,'used_clusters'), largest_cluster_fraction=mean(region,'largest_cluster_fraction'),
            cluster_entropy=mean(region,'cluster_entropy')))
        aggregated = {i: {'correct':[], 'cluster_agreement':[]} for i in valid}
        with (directory/'paired_metrics.csv').open(newline='') as stream:
            for r in csv.DictReader(stream):
                if r['stage']=='2' and r['readout']=='bbox' and int(r['content_size']) in (112,144,176):
                    for key in ('correct','cluster_agreement'): aggregated[int(r['source_index'])][key].append(float(r[key]))
        assert all(len(v['correct'])==3 for v in aggregated.values())
        per_image[name] = {key: np.array([np.mean(aggregated[i][key]) for i in valid]) for key in ('correct','cluster_agreement')}
    scale.csv_write(OUT/'scale_comparison.csv', combined)
    paired = {}
    for left, right in [('aligned_1014','remote_baseline'),('T1','aligned_1014'),('S1','aligned_1014'),('S1','S0')]:
        paired[f'{left}_minus_{right}'] = {}
        for key in ('correct','cluster_agreement'):
            delta = per_image[left][key]-per_image[right][key]
            paired[f'{left}_minus_{right}'][key] = dict(mean=float(delta.mean()), image_bootstrap_95ci=scale.bootstrap(delta))
    scale.dump(OUT/'paired_comparison.json', paired)

    shift = []
    for name, archive in [('remote_baseline','baseline'),('aligned_1014','aligned_8_16_32'),('T1','augment_1_7')]:
        directory = ROOT/'plots/kmeans/caltech_non_aligned_restart_1015'/archive
        manifest = json.loads((directory/'run_manifest.json').read_text())
        if name in models:
            expected = json.loads((models[name]/'run_manifest.json').read_text())['checkpoint_sha256']
            assert expected == manifest['checkpoint_sha256']
        data = rows(directory/'summary.csv')
        moved = [r for r in data if r['category']=='all_selected' and r['stage']=='0' and int(r['shift_pixels'])>0]
        result = dict(model=name, shifted_accuracy=mean(moved,'accuracy'), prediction_consistency=mean(moved,'prediction_consistency'))
        for stage in range(4):
            values = [r for r in data if r['category']=='all_selected' and int(r['stage'])==stage and int(r['shift_pixels'])>0]
            result[f'stage{stage}_agreement'] = mean(values,'cluster_agreement')
        shift.append(result)
    scale.csv_write(OUT/'translation_comparison.csv', shift)
    lines = ['# 遠端 10/14 模型與本機 10/15 實驗比較', '',
        '已驗證兩份遠端 checkpoint SHA256 與歷史 spherical 評估完全相符，split 亦一致。新收到資料 archives 通過官方 MD5。',
        '遠端 baseline 原 config 未保存逐筆 indices，split 證據來自綁定相同 checkpoint hash 的歷史評估 manifest。',
        '尺度為本機對遠端 checkpoints 新執行的評估；本機 S0/S1/T1 在相同 runner、split 與 spherical protocol 下的已完成結果重用。',
        '平移重用綁定相同 checkpoint SHA256 的 1–7 px spherical 報告；不是把 10/14 Euclidean 指標改名，也不是另一次獨立重複。', '',
        '## 尺度：101 類、1,770 張 validation', '',
        '未見離散倍率 0.7/0.9/1.1；各 stage/readout 以原位 training features fitting 獨立 codebook。', '',
        '| 模型 | 原位 accuracy | 未見倍率 accuracy | macro accuracy | Stage 2 bbox agreement | ROI cosine |',
        '|---|---:|---:|---:|---:|---:|']
    for r in combined:
        lines.append(f"| {r['model']} | {r['original_accuracy']:.2%} | {r['unseen_scale_accuracy']:.2%} | {r['unseen_scale_macro_accuracy']:.2%} | {r['stage2_bbox_agreement']:.2%} | {r['stage2_bbox_cosine']:.4f} |")
    lines += ['', '## 平移：三類 202 張、四方向 1–7 px 平均', '',
        '| 模型 | 平移 accuracy | 預測一致率 | Stage 2 agreement |', '|---|---:|---:|---:|']
    for r in shift:
        lines.append(f"| {r['model']} | {r['shifted_accuracy']:.2%} | {r['prediction_consistency']:.2%} | {r['stage2_agreement']:.2%} |")
    byname = {r['model']:r for r in combined}
    lines += ['', '## 判讀', '',
        f"- S1 相對 aligned_1014 的未見倍率 accuracy 差 {(byname['S1']['unseen_scale_accuracy']-byname['aligned_1014']['unseen_scale_accuracy'])*100:+.2f} pp，Stage 2 bbox agreement 差 {(byname['S1']['stage2_bbox_agreement']-byname['aligned_1014']['stage2_bbox_agreement'])*100:+.2f} pp。",
        '- 先同時檢查原位 accuracy、macro accuracy、群數／最大群占比／熵與同類異圖對照，不把 agreement 單獨視為語意改善。',
        '- 不同模型群 ID 不直接對應；ROI pooling 穩定不等於原始 token 不變。',
        '- 遠端與本機是不同 training runs／執行環境；本機 S0/S1 是主要尺度增強配對，歷史 aligned 模型是額外比較，不是只差一個變因的嚴格消融。',
        '- 單一 seed、validation 曾用於選模。paired_comparison.json 的影像 bootstrap 不包含 training seed 不確定性，四組比較為探索性，未做多重比較校正。', '',
        '## 檔案', '', 'intake_manifest.json：來源／hash；scale_comparison.csv、translation_comparison.csv：比較；paired_comparison.json：配對差值；scale/：新跑的遠端模型 metrics、圖與 cache。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    scale.dump(OUT/'status.json',dict(status='complete',completed_utc=datetime.now(timezone.utc).isoformat()))
    print('COMPLETE: remote 1014 comparison', flush=True)


if __name__ == '__main__': main()
