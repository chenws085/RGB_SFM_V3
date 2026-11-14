#!/usr/bin/env python3
"""Source-paired aggregate effects and historical replication checks."""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    out = Path(sys.argv[1])
    root = Path(__file__).resolve().parents[1]
    result = json.loads((out/'results.json').read_text())
    for row in result['rows']:
        if row['metric']=='within_stage_scale' and row['condition']=='96_center':
            assert row['agreement']==1.0, row
    arrays = np.load(out/'per_source.npz')
    saved = np.load(out/'dictionaries_assignments.npz')
    for key in saved.files:
        if saved[key].ndim == 2:
            assert np.allclose(np.linalg.norm(saved[key], axis=1), 1, atol=1e-4), key
    old = root/'plots/kmeans/clip_roi_1103'
    # Dictionary assignments, not only floating features, must reproduce the baseline.
    for name, oldname in [('CE', 'A'), ('CLIP', 'CLIP')]:
        for stage in range(4):
            cached = np.load(old/f'{oldname}_raw_stage{stage}.npz')
            assert np.array_equal(saved[f'assign_{name}_raw_96_center_{stage}'], cached['cluster_a'])
            assert np.array_equal(saved[f'assign_{name}_raw_80_shift_{stage}'], cached['cluster_b'])
    for key in arrays.files:
        assert np.isfinite(arrays[key]).all(), key
    occupancy = {}
    for key in saved.files:
        if not key.startswith('assign_'):
            continue
        counts = np.bincount(saved[key], minlength=31)
        p = counts/counts.sum()
        occupancy[key] = dict(occupied=int((p>0).sum()), max_fraction=float(p.max()),
            effective_clusters=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())))
    (out/'all_condition_occupancy.json').write_text(json.dumps(occupancy, indent=2))
    rng = np.random.default_rng(42)
    boot = rng.integers(len(arrays['source_indices']), size=(2000, len(arrays['source_indices'])))
    aggregates, scores = {}, {}
    for metric in ['within_stage_scale', 'cross_stage_scale']:
        aggregates[metric] = {}
        for branch in ['CE_raw', 'CE_probe', 'CLIP_raw', 'CLIP_probe']:
            selected = [r for r in result['rows'] if r['branch']==branch and
                        r['metric']==metric and r['condition'] in ['80_center', '112_center']]
            score = np.mean([arrays[r['per_source_key']] for r in selected], axis=0)
            control = np.mean([arrays[r['per_source_key']+'__control'] for r in selected], axis=0)
            scores[(metric, branch)] = (score, control)
            aggregates[metric][branch] = dict(agreement=float(score.mean()),
                same_class_other=float(control.mean()), gap=float((score-control).mean()))
        effects = {}
        for first, second in [('CE_probe','CE_raw'), ('CLIP_probe','CLIP_raw'),
                              ('CLIP_probe','CE_raw'), ('CLIP_raw','CE_raw'), ('CLIP_probe','CE_probe')]:
            a, ac = scores[(metric, first)]
            b, bc = scores[(metric, second)]
            d, g = a-b, (a-ac)-(b-bc)
            effects[first+'-'+second] = dict(delta=float(d.mean()),
                source_bootstrap_95ci=np.quantile(d[boot].mean(1), [.025,.975]).tolist(),
                gap_delta=float(g.mean()), gap_delta_95ci=np.quantile(g[boot].mean(1), [.025,.975]).tolist())
        aggregates[metric]['effects'] = effects
    dest = out/'overall_summary.json'
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(aggregates, indent=2))
    lines = ['', '## 來源影像內先平均後的整體比較（80/112純縮放）', '',
        '同層先平均四層×兩尺度；跨層先平均六配對×兩尺度×兩方向，最後才在1768個來源間取平均／bootstrap，沒有把層或尺度當獨立樣本。', '',
        '| 指標 | CE raw | CE probe | CLIP raw | CLIP probe |', '|---|---:|---:|---:|---:|']
    for metric, label in [('within_stage_scale','同層跨尺度'), ('cross_stage_scale','跨層＋跨尺度')]:
        vals = [aggregates[metric][b]['agreement'] for b in ['CE_raw','CE_probe','CLIP_raw','CLIP_probe']]
        lines.append('| '+label+' | '+' | '.join(f'{v:.2%}' for v in vals)+' |')
    lines += ['', '配對差值，單位百分點：']
    for metric, label in [('within_stage_scale','同層跨尺度'), ('cross_stage_scale','跨層＋跨尺度')]:
        for contrast in ['CLIP_probe-CLIP_raw', 'CLIP_probe-CE_raw']:
            e = aggregates[metric]['effects'][contrast]
            ci = e['source_bootstrap_95ci']
            lines.append(f"- {label} {contrast}: {100*e['delta']:+.2f} pp，95% CI [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]；同物件−同類異圖差距改變 {100*e['gap_delta']:+.2f} pp。")
    lines += ['', '基準驗證：八個raw字典於兩個歷史條件的validation群指派與舊cache逐筆完全一致；所有來源數值finite。跨層群號映射另以已知置換檢查通過。']
    with (out/'REPORT.md').open('a') as f:
        f.write('\n'.join(lines)+'\n')
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
