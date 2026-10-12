#!/usr/bin/env python3
"""Render saved intervention results, aggregating random repeats within each image."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def image_records(frame):
    # Each image, rather than each random repeat, contributes one point.
    numeric = ['logit_drop', 'random_logit_drop', 'effect_gap', 'margin_drop', 'probability_drop']
    grouped = frame.groupby(['group', 'source_index'], sort=False)
    result = grouped[numeric].mean().reset_index()
    return result.merge(grouped[['target', 'label', 'original_prediction', 'intervention_prediction']].first().reset_index(), on=['group', 'source_index'])


def grid_geometry(model_args, stage):
    size = model_args['img_size']
    height, width = (size, size) if isinstance(size, int) else size
    patch = model_args['patch_size']
    ph, pw = (patch, patch) if isinstance(patch, int) else patch
    height, width = height // ph, width // pw
    scale_h, scale_w = ph, pw
    merge = model_args.get('merge_size', 2)
    for index in range(stage):
        if isinstance(merge, int):
            mh, mw = merge, merge
        elif isinstance(merge[0], (list, tuple)):
            mh, mw = merge[index]
        else:
            mh, mw = merge
        height, width = (height + mh - 1) // mh, (width + mw - 1) // mw
        scale_h, scale_w = scale_h * mh, scale_w * mw
    return height, width, scale_h, scale_w


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='plots/concept_intervention/smoke')
    parser.add_argument('--output-dir')
    parser.add_argument('--max-cases', type=int, default=4)
    args = parser.parse_args()
    if args.max_cases < 0:
        parser.error('max-cases must be nonnegative')
    source = ROOT / args.results_dir
    output = ROOT / args.output_dir if args.output_dir else source / 'visualization'
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / 'run_manifest.json').read_text())
    frame = pd.read_csv(source / 'sample_effects.csv')
    if frame.empty:
        parser.error('no supported concepts in results')
    records = image_records(frame)
    summary = pd.read_csv(source / 'concept_summary.csv')
    groups = summary['group'].tolist()
    palette = ['#2563eb', '#f97316', '#16a34a', '#9333ea']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    caption = ('Smoke / exploratory results. Positive drop = target logit decreases; negative = increases.\n'
               'Dots are images, random repeats averaged within image. Controls do not match position or foreground.')
    with PdfPages(output / 'intervention_report.pdf') as pdf:
        def save(fig, name):
            fig.savefig(output / f'{name}.png', dpi=180, bbox_inches='tight')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(max(13, len(groups)*1.5), 5.6))
        x = np.arange(len(groups))
        axes[0].bar(x-.18, summary['mean_logit_drop'], .36, label='Concept replacement', color=palette[0])
        axes[0].bar(x+.18, summary['mean_random_logit_drop'], .36, label='Random tokens', color='#94a3b8')
        axes[0].set_title('Target logit change'); axes[0].set_ylabel('Mean logit drop'); axes[0].legend(fontsize=8)
        axes[1].bar(x, summary['mean_effect_gap'], color=[palette[2] if v >= 0 else palette[1] for v in summary['mean_effect_gap'].fillna(0)])
        for i, group in enumerate(groups):
            values = records.loc[records.group == group, 'effect_gap'].to_numpy()
            axes[1].scatter(i+np.linspace(-.10, .10, len(values)), values, color='#0f172a', s=18, zorder=3)
        axes[1].set_title('Concept drop minus random drop'); axes[1].set_ylabel('Effect gap')
        axes[2].bar(x, summary['supporting_images'], color='#64748b')
        for i, n in enumerate(summary['supporting_images']):
            axes[2].text(i, n+.1, str(int(n)), ha='center')
        axes[2].set_title('Evidence available'); axes[2].set_ylabel('Supporting images'); axes[2].set_ylim(0, max(summary.supporting_images)*1.2+1)
        for ax in axes:
            ax.axhline(0, color='#334155', linewidth=.8)
            ax.set_xticks(x, groups, rotation=45, ha='right')
        fig.suptitle(f"Stage {manifest['arguments']['stage']} concept interventions | {len(manifest['train_indices'])} training / {len(manifest['eval_indices'])} validation images")
        fig.text(.02, .015, caption, fontsize=9)
        fig.tight_layout(rect=[0,.13,1,.94]); save(fig, 'concept_overview')

        fig, ax = plt.subplots(figsize=(8,6))
        for index, group in enumerate(groups):
            subset = records[records.group == group]
            ax.scatter(subset.random_logit_drop, subset.logit_drop, s=45, alpha=.8, label=f'{group} (n={len(subset)})')
        values = records[['logit_drop', 'random_logit_drop']].to_numpy()
        low, high = float(values.min()), float(values.max())
        padding = max((high-low)*.1, .1)
        ax.plot([low-padding, high+padding], [low-padding, high+padding], '--', color='#64748b', label='Equal effect')
        ax.axhline(0, color='#cbd5e1'); ax.axvline(0, color='#cbd5e1')
        ax.set(xlabel='Random-token logit drop (mean across repeats)', ylabel='Concept logit drop', title='Per-image comparison: above diagonal = larger concept drop')
        ax.legend(fontsize=8); fig.text(.02,.015,caption,fontsize=8)
        fig.tight_layout(rect=[0,.10,1,1]); save(fig, 'paired_effects')

        for joint in [g for g in groups if '+' in g]:
            components = joint.split('+')
            selected = records[records.group == joint].source_index.tolist()
            subset = records[records.source_index.isin(selected) & records.group.isin(components+[joint])]
            pivot = subset.pivot(index='source_index', columns='group', values='logit_drop').reindex(columns=components+[joint]).dropna()
            if pivot.empty:
                continue
            interaction = pivot[joint]-pivot[components].sum(axis=1)
            exported = pivot.copy(); exported['joint_minus_sum'] = interaction
            exported.to_csv(output / f'joint_{joint.replace(":", "_").replace("+", "__")}_paired.csv')
            fig, axes = plt.subplots(1,2,figsize=(11,5))
            for source_id, row in pivot.iterrows():
                axes[0].plot(range(len(row)), row.values, marker='o', label=f'Image {source_id}')
            axes[0].set_xticks(range(len(pivot.columns)), pivot.columns, rotation=20)
            axes[0].set(ylabel='Target logit drop',title=f'Joint-present images only (n={len(pivot)})'); axes[0].legend(fontsize=8)
            axes[1].bar(range(len(interaction)), interaction.values,color=palette[3])
            axes[1].set_xticks(range(len(interaction)), interaction.index.astype(str),rotation=30)
            axes[1].set(ylabel='Joint drop - sum of individual drops',xlabel='Source image',title='Descriptive non-additivity')
            for ax in axes: ax.axhline(0,color='#64748b',linewidth=.8)
            fig.text(.02,.015,'Same images and fixed target for A, B and A+B. Non-additivity alone does not establish a causal mechanism.',fontsize=9)
            fig.tight_layout(rect=[0,.07,1,1]); save(fig, 'joint_'+joint.replace(':','_').replace('+','__'))

        if args.max_cases:
            from torchvision.datasets import Caltech101
            from scripts.test_caltech_translation_kmeans import safe_letterbox
            base = Caltech101(str(ROOT / manifest['arguments']['data_root']),target_type=['category','annotation'],download=False)
            height,width,sh,sw = grid_geometry(manifest['model_args'],manifest['arguments']['stage'])
            ranked = records.assign(magnitude=records.effect_gap.abs()).sort_values('magnitude',ascending=False).drop_duplicates('source_index').head(args.max_cases)
            for _, record in ranked.iterrows():
                source_id = int(record.source_index)
                image, (label, annotation) = base[source_id]
                _, metadata, canvas = safe_letterbox(image,annotation)
                fig, axes = plt.subplots(1,2,figsize=(11,5))
                axes[0].imshow(canvas)
                for index, token in enumerate(record.group.split('+')):
                    position, cluster = map(int,token.split(':'))
                    row,col = divmod(position,width)
                    axes[0].add_patch(Rectangle((col*sw,row*sh),sw,sh,fill=False,linewidth=3,edgecolor=palette[index%len(palette)]))
                    axes[0].text(col*sw,row*sh,token,color='white',fontsize=10,bbox=dict(facecolor=palette[index%len(palette)],alpha=.85,pad=2))
                axes[0].set_title(f'Image {source_id} | {base.categories[label]}'); axes[0].axis('off')
                per_image = records[records.source_index == source_id]
                positions = np.arange(len(per_image))
                axes[1].bar(positions-.18,per_image.logit_drop,.36,label='Concept',color=palette[0])
                axes[1].bar(positions+.18,per_image.random_logit_drop,.36,label='Random',color='#94a3b8')
                axes[1].set_xticks(positions,per_image.group,rotation=35,ha='right'); axes[1].axhline(0,color='#64748b')
                axes[1].set(ylabel='Target logit drop',title=f"Target: {base.categories[int(record.target)]}\nOriginal prediction: {base.categories[int(record.original_prediction)]}")
                axes[1].legend(fontsize=8)
                fig.text(.02,.015,f'Boxes mark token grid cells, not full attention receptive fields. Replacement: {manifest["arguments"]["replacement"]}.\nHighlighted group: {record.group}; bars include all evaluated groups present in this image.',fontsize=9)
                fig.tight_layout(rect=[0,.11,1,1]); save(fig,f'image_{source_id}')
    records.to_csv(output / 'per_image_effects.csv',index=False)
    (output / 'README.md').write_text('# Concept intervention visualization\n\n'
        '概念概覽：concept_overview.png；逐影像對照：paired_effects.png；完整圖集：intervention_report.pdf。\n\n'
        '正的 logit drop 表示目標分數下降，負值表示上升。Effect gap 為概念下降量減隨機下降量。黑點為影像，已在影像內平均 random repeats。\n\n'
        '組合圖只在聯合概念存在、且 A/B/A+B 都有結果的相同影像上比較；joint_minus_sum 是描述性非加成量，不能直接視為因果交互作用證明。\n\n'
        '影像框為 token 網格位置，不是完整感受野。此為小樣本功能展示，無信賴區間或显著性宣稱；隨機對照未匹配位置／前景。\n')
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
