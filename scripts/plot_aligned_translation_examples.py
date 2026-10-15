#!/usr/bin/env python3
"""Original/translated airplane examples, with matched token and optional model outputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import run_caltech_corrected_center as corrected
import test_caltech_translation_kmeans as legacy


def save_examples(base, ids, output_dir, rows=None, model_name=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = list(ids[:6])
    lookup = {(r['source_index'],r['stage'],r['shift_pixels'],r['direction']):r
              for r in (rows or []) if r['model']==model_name}
    conditions = {'original':(0,0), **corrected.DIRECTIONS}
    for shift in [8,16,32]:
        # Show the stage whose grid matches this displacement.
        stage = {8:0,16:1,32:2}[shift]
        fig,axes = plt.subplots(len(selected),5,figsize=(17,3.6*len(selected)),squeeze=False)
        for ri,source in enumerate(selected):
            image,(_,annotation)=base[source]
            annotation=corrected.absolute_annotation(base,source,annotation)
            for ci,(direction,(dx,dy)) in enumerate(conditions.items()):
                _,meta,canvas=legacy.safe_letterbox(image,annotation,dx*shift,dy*shift)
                ax=axes[ri,ci]
                ax.imshow(canvas)
                cx,cy=meta['center_x'],meta['center_y']
                tx,ty=int(cx//shift)*shift,int(cy//shift)*shift
                ax.add_patch(Rectangle((tx,ty),shift,shift,fill=False,edgecolor='yellow',linewidth=1.5))
                ax.plot(cx,cy,'r.',markersize=4)
                ax.set_xticks([]);ax.set_yticks([])
                if ri==0: ax.set_title('Original' if direction=='original' else f'{direction.capitalize()} {shift}px')
                if ci==0: ax.set_ylabel(f'Source {source}')
                key=(source,stage,0 if ci==0 else shift,direction)
                if key in lookup:
                    r=lookup[key]
                    pred=base.categories[r['moved_prediction']]
                    text=f"Prediction: {pred}\nCluster: {r['moved_cluster']}"
                    if ci:
                        text+=f" | cosine: {r['cosine']:.3f}\nPrediction {'same' if r['prediction_consistent'] else 'changed'}; cluster {'same' if r['same_cluster'] else 'changed'}"
                    ax.set_xlabel(text,fontsize=9,color='black' if r['prediction_consistent'] else 'firebrick')
        label=model_name or 'Input examples'
        fig.suptitle(f'{label}: airplanes before/after {shift}px translation | Stage {stage} token in yellow',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.975))
        prefix=f'{model_name}_' if model_name else ''
        fig.savefig(out/f'{prefix}airplanes_stage{stage}_{shift}px_montage.png',dpi=160)
        plt.close(fig)


def main():
    base=legacy.Caltech101('data',target_type=['category','annotation'],download=False)
    _,valid=legacy.split_indices('data',42)
    airplane=base.categories.index('airplanes')
    ids=sorted(i for i in valid if base.y[i]==airplane)
    save_examples(base,ids,'plots/kmeans/caltech_aligned_translation_1014/examples')

if __name__=='__main__':
    main()
