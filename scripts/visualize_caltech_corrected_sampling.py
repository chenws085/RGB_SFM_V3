#!/usr/bin/env python3
"""Show five validation images and the corrected sampling token at each stage."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from run_caltech_corrected_center import ROOT, Caltech101, absolute_annotation, safe_letterbox


def main():
    out = ROOT / 'plots/kmeans/caltech_corrected_center_1012/sampling_examples'
    out.mkdir(exist_ok=True)
    run = out.parent
    manifest = json.loads((run / 'run_manifest.json').read_text())
    base = Caltech101(str(ROOT / 'data'), target_type=['category', 'annotation'], download=False)
    categories = ['airplanes', 'butterfly', 'camera', 'car_side', 'elephant']
    ids = [next(i for i in manifest['valid_indices'] if base.categories[base.y[i]] == name)
           for name in categories]
    with (run / 'coordinate_audit.csv').open() as f:
        audit = {(int(r['source_index']), int(r['stage'])): r for r in csv.DictReader(f)
                 if r['split'] == 'validation'}
    fig, axes = plt.subplots(5, 4, figsize=(12, 15))
    rows = []
    for row, source in enumerate(ids):
        image, (_, annotation) = base[source]
        corrected = absolute_annotation(base, source, annotation)
        _, meta, canvas = safe_letterbox(image, corrected)
        cx, cy = meta['center_x'], meta['center_y']
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(canvas, extent=(0, 224, 224, 0))
            ax.plot(cx, cy, '+', color='red', markersize=10, markeredgewidth=1.5)
            if col == 0:
                ax.set_title(f'{categories[row]} | source {source}', fontsize=10)
            else:
                stage, size = col - 1, 8 * 2 ** (col - 1)
                tx, ty = int(cx // size), int(cy // size)
                saved = audit[source, stage]
                assert (tx, ty) == (int(saved['token_x']), int(saved['token_y']))
                assert np.allclose([cx, cy], [float(saved['center_x']), float(saved['center_y'])])
                for edge in range(0, 225, size):
                    ax.axvline(edge, color='white', alpha=.22, lw=.4)
                    ax.axhline(edge, color='white', alpha=.22, lw=.4)
                ax.add_patch(Rectangle((tx * size, ty * size), size, size,
                                       fill=False, edgecolor='#00ffff', linewidth=2))
                ax.set_title(f'Stage {stage}: {size} x {size} px | token ({tx}, {ty})', fontsize=9)
                canvas.crop((tx * size, ty * size, (tx + 1) * size, (ty + 1) * size)).save(
                    out / f'{row+1}_{categories[row]}_stage{stage}_crop.png')
                rows.append(dict(source_index=source, category=categories[row], stage=stage,
                                 center_x=cx, center_y=cy, token_x=tx, token_y=ty, token_size=size))
            ax.set_xlim(0, 224)
            ax.set_ylim(224, 0)
            ax.axis('off')
        canvas.save(out / f'{row+1}_{categories[row]}_canvas.png')
    fig.suptitle('Corrected object-center sampling | same five validation images across stages\n'
                 'Red +: mean contour position | Cyan box: selected token grid footprint (not receptive field)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(out / 'sampling_five_images.png', dpi=160)
    plt.close(fig)
    with (out / 'samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out / 'sampling_five_images.png')


if __name__ == '__main__':
    main()
