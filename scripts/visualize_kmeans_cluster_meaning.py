#!/usr/bin/env python3
"""Render high/boundary/negative silhouette exemplars near annotated centroids."""
import argparse
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat
from threadpoolctl import threadpool_limits
from spherical_kmeans import unit_rows

METHODS = ["sklearn_existing", "spherical", "faiss_native"]
BIN_COLORS = {"high_positive": "#e2f4e8", "near_zero": "#fff2ce", "negative": "#ffe4e4"}


def select_representatives(rows, count, near_zero):
    """Disjoint examples; prioritize negative extremes and positive maxima."""
    negative = sorted((r for r in rows if r["score"] < 0), key=lambda r:(r["score"],r["source_index"]))[:count]
    high = sorted((r for r in rows if r["score"] > 0), key=lambda r:(-r["score"],r["source_index"]))[:count]
    used = {r["source_index"] for r in negative + high}
    boundary = sorted((r for r in rows if abs(r["score"]) <= near_zero and r["source_index"] not in used),
                      key=lambda r:(abs(r["score"]),r["source_index"]))[:count]
    return {"high_positive":high, "near_zero":boundary, "negative":negative}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",default="plots/kmeans/quality_evaluation_1005")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--per-bin",type=int,default=3)
    parser.add_argument("--near-zero",type=float,default=.05)
    args=parser.parse_args()
    if args.per_bin < 1 or args.near_zero < 0:
        raise ValueError("per-bin must be positive; near-zero must be nonnegative")
    threadpool_limits(4)
    root=Path(args.input_dir)
    output=root/f"cluster_meaning_seed{args.seed}"
    output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"run_manifest.json").read_text())
    assert args.seed in manifest["source_manifest"]["seeds"]
    raw=np.load(root/"heldout_features.npz")
    source_ids=raw["source_ids"].astype(int)
    lookup={int(i):j for j,i in enumerate(source_ids)}
    grouped=defaultdict(list)
    with (root/"heldout_sample_metrics.csv").open() as f:
        for row in csv.DictReader(f):
            if int(row["seed"])==args.seed and row["method"] in METHODS:
                grouped[(int(row["stage"]),row["method"],int(row["cluster"]))].append(
                    dict(source_index=int(row["source_index"]),category=int(row["category"]),score=float(row["cosine_silhouette"])))
    from test_caltech_translation_kmeans import ROOT, Caltech101, safe_letterbox, token_box
    import torch
    base=Caltech101(str(ROOT/"data"),target_type=["category","annotation"],download=False)
    checkpoint=torch.load(manifest["source_manifest"]["checkpoint"],map_location="cpu",weights_only=False)
    model_args=checkpoint["model_args"]
    patch=int(model_args["patch_size"])
    assert patch==8 and model_args["merge_size"]==[(2,2),(2,2),(2,2)]
    font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fonts={size:ImageFont.truetype(font_path,size) for size in [13,15,17,20,24]}

    @lru_cache(maxsize=512)
    def canvas_for(source):
        image,(category,annotation)=base[source]
        _,metadata,canvas=safe_letterbox(image,annotation)
        annotation_file=Path(base.root)/"Annotations"/base.annotation_categories[category]/f"annotation_{base.index[source]:04d}.mat"
        box_coord=loadmat(annotation_file)["box_coord"].ravel()
        # Official show_annotation.m adds box offsets; subtract 1 for PIL coordinates.
        absolute=annotation.astype(float)+np.array([[box_coord[2]],[box_coord[0]]])-1
        _,corrected,_=safe_letterbox(image,absolute)
        return metadata,canvas,category,corrected

    def mark(image,cx,cy,box=None,corrected=None):
        result=image.copy()
        draw=ImageDraw.Draw(result)
        if box is not None:
            draw.rectangle(box,outline="#ffd700",width=2)
        draw.line((cx-4,cy,cx+4,cy),fill="#ff1744",width=1)
        draw.line((cx,cy-4,cx,cy+4),fill="#ff1744",width=1)
        if corrected is not None:
            tx,ty=corrected["center_x"],corrected["center_y"]
            draw.line((tx-5,ty,tx+5,ty),fill="#00c853",width=2)
            draw.line((tx,ty-5,tx,ty+5),fill="#00c853",width=2)
        return result

    audit=[]
    for source in source_ids:
        metadata,_,_,corrected=canvas_for(int(source))
        for stage in range(3):
            size=patch*2**stage
            old_box=token_box(metadata,size)
            new_box=token_box(corrected,size)
            audit.append(dict(source_index=int(source),stage=stage,legacy_x=metadata["center_x"],legacy_y=metadata["center_y"],
                corrected_x=corrected["center_x"],corrected_y=corrected["center_y"],
                displacement_canvas_pixels=float(np.hypot(corrected["center_x"]-metadata["center_x"],corrected["center_y"]-metadata["center_y"])),
                same_token=old_box==new_box))
    print("Coordinate audit complete: bounding-box offsets applied for displayed object centers",flush=True)
    selections, summaries, pages=[],[],[]
    for stage in range(3):
        features=unit_rows(raw[f"stage{stage}"].astype(float))
        pairwise=np.maximum(1-features@features.T,0)
        np.fill_diagonal(pairwise,0)
        for method in METHODS:
            clusters=sorted(c for s,m,c in grouped if s==stage and m==method)
            labels=np.empty(len(source_ids),dtype=int)
            for cluster in clusters:
                for row in grouped[(stage,method,cluster)]:
                    assert row["category"]==base.y[row["source_index"]]
                    labels[lookup[row["source_index"]]]=cluster
            means=pairwise@np.eye(max(clusters)+1)[labels]
            means/=np.maximum(np.bincount(labels,minlength=max(clusters)+1),1)[None,:]
            means[np.arange(len(labels)),labels]=np.inf
            alternative=means.argmin(1)
            method_dir=output/f"stage{stage}"/method
            method_dir.mkdir(parents=True,exist_ok=True)
            for cluster in clusters:
                rows=grouped[(stage,method,cluster)]
                chosen=select_representatives(rows,args.per_bin,args.near_zero)
                counts=Counter(base.categories[r["category"]] for r in rows)
                high_classes=Counter(base.categories[r["category"]] for r in chosen["high_positive"])
                neg_count=sum(r["score"]<0 for r in rows)
                summary=dict(stage=stage,seed=args.seed,method=method,cluster=cluster,samples=len(rows),
                    mean_silhouette=float(np.mean([r["score"] for r in rows])),negative_fraction=neg_count/len(rows),
                    distinct_categories=len(counts),top_categories=json.dumps(counts.most_common(5)),
                    selected_high_categories=json.dumps(high_classes.most_common()),
                    high_selected=len(chosen["high_positive"]),near_zero_selected=len(chosen["near_zero"]),negative_selected=len(chosen["negative"]))
                summaries.append(summary)
                column=190
                width=column*args.per_bin*3+30
                sheet=Image.new("RGB",(width,820),"white")
                draw=ImageDraw.Draw(sheet)
                draw.text((15,10),f"Stage {stage} | {method} | Seed {args.seed} | Cluster {cluster:02d}",font=fonts[24],fill="black")
                draw.text((15,44),f"N={len(rows)} | mean silhouette={summary['mean_silhouette']:.4f} | negative={neg_count/len(rows):.1%} | categories={len(counts)}",font=fonts[17],fill="black")
                category_line="Top categories: "+", ".join(f"{name} {n}/{len(rows)}" for name,n in counts.most_common(3))
                draw.text((15,72),category_line,font=fonts[17],fill="black")
                headings={"high_positive":"Highest positive scores", "near_zero":f"Closest to zero (|s| <= {args.near_zero:g})", "negative":"Most negative scores"}
                for band,(kind,items) in enumerate(chosen.items()):
                    band_x=15+band*args.per_bin*column
                    draw.rectangle((band_x,105,band_x+args.per_bin*column-5,141),fill=BIN_COLORS[kind])
                    draw.text((band_x+8,112),headings[kind],font=fonts[17],fill="black")
                    for rank in range(args.per_bin):
                        x=band_x+rank*column
                        if rank>=len(items):
                            draw.text((x+8,185),"No eligible sample",font=fonts[15],fill="#666666")
                            continue
                        row=items[rank]
                        source=row["source_index"]
                        metadata,canvas,category,corrected=canvas_for(source)
                        size=patch*2**stage
                        box=token_box(metadata,size)
                        cx,cy=metadata["center_x"],metadata["center_y"]
                        context=mark(canvas,cx,cy,box,corrected)
                        tx,ty=corrected["center_x"],corrected["center_y"]
                        zoom_box=(int(round(tx))-32,int(round(ty))-32,int(round(tx))+32,int(round(ty))+32)
                        zoom=context.crop(zoom_box)
                        crop=canvas.crop(box)
                        draw.text((x+4,149),f"#{source}   s={row['score']:+.4f}",font=fonts[15],fill="black")
                        for line,text in enumerate(textwrap.wrap(base.categories[category],width=22)[:2]):
                            draw.text((x+4,171+line*18),text,font=fonts[15],fill="black")
                        alt=int(alternative[lookup[source]])
                        draw.text((x+4,210),f"Nearest other group: {alt:02d}",font=fonts[13],fill="#555555")
                        for y,image,title,resampling in [
                            (247,context,"Full image (224px canvas)",Image.Resampling.LANCZOS),
                            (437,zoom,"Corrected center zoom: 64px",Image.Resampling.NEAREST),
                            (627,crop,f"Sampled token: {size}x{size}px",Image.Resampling.NEAREST)]:
                            draw.text((x+4,y-20),title,font=fonts[13],fill="#333333")
                            sheet.paste(image.resize((160,160),resampling),(x+10,y))
                        content_box=(metadata["content_x0"],metadata["content_y0"],metadata["content_x1"],metadata["content_y1"])
                        selections.append(dict(stage=stage,seed=args.seed,method=method,cluster=cluster,bin=kind,rank=rank+1,
                            source_index=source,category=category,class_name=base.categories[category],silhouette=row["score"],
                            nearest_other_cluster=alt,center_x=cx,center_y=cy,token_box=json.dumps(box),
                            corrected_center_x=tx,corrected_center_y=ty,
                            center_displacement_pixels=float(np.hypot(tx-cx,ty-cy)),
                            sampled_token_matches_corrected_token=box==token_box(corrected,size),
                            zoom_box=json.dumps(zoom_box),token_overlaps_letterbox=not(box[0]>=content_box[0] and box[1]>=content_box[1] and box[2]<=content_box[2] and box[3]<=content_box[3])))
                draw.text((15,799),"Red = legacy sampled point; green = corrected contour center; yellow = sampled token grid. Attention can include content outside the grid.",font=fonts[15],fill="#555555")
                image_path=method_dir/f"cluster_{cluster:02d}.png"
                sheet.save(image_path,optimize=True)
                pages.append(dict(stage=stage,method=method,cluster=cluster,image=str(image_path.relative_to(output)),**{k:summary[k] for k in ['samples','mean_silhouette','distinct_categories','high_selected','near_zero_selected','negative_selected']}))
            print(f"Rendered Stage {stage} / {method}: {len(clusters)} groups",flush=True)
    for name,data in [("representative_samples.csv",selections),("cluster_summary.csv",summaries),("coordinate_audit.csv",audit)]:
        with (output/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    gallery='''<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K-means 群解釋性檢視</title>
<style>body{font-family:system-ui;margin:24px;background:#f7f8fa;color:#20242a}select{padding:8px;margin:6px}img{width:100%;height:auto;background:white}p{max-width:1100px;line-height:1.7}a{color:#1663b8}#meta{margin:16px 0}</style>
<h1>K-means 群解釋性檢視</h1><p>每群以高正分、接近 0、最負分各最多 3 張樣本呈現。每欄依序顯示全圖、修正後輪廓中心的 64px 局部內容、原實驗實際取樣的 token。紅十字是原實驗取樣位置，綠十字是補上 bounding-box offset 後的輪廓點平均位置，黃框是原實驗取樣的 token 範圍。</p><p><strong>座標稽核發現：原實驗漏加輪廓座標的 box 偏移，分群與 silhouette 是原取樣位置的結果。這個圖庫修正中心標示以供對照，沒有重新抽取特徵或更新分群；因此不能將既有群直接稱為物體質心的部件群。</strong> Attention 可能使用框外資訊；不同模型的相同群 ID 不代表相同群。</p>
<label>Stage <select id="stage"><option>0</option><option>1</option><option selected>2</option></select></label>
<label>方法 <select id="method"><option value="sklearn_existing">現有 K-means</option><option value="spherical" selected>手寫球面 K-means</option><option value="faiss_native">FAISS 球面 K-means</option></select></label>
<label>Cluster <select id="cluster"></select></label><div id="meta"></div><a id="full" target="_blank">開啟原尺寸圖片</a><p><img id="image"></p>
<script>const pages=__PAGES__;const stage=document.getElementById('stage'),method=document.getElementById('method'),cluster=document.getElementById('cluster');
function updateOptions(){const previous=cluster.value;cluster.replaceChildren();for(const p of pages.filter(p=>p.stage==stage.value&&p.method==method.value)){const o=document.createElement('option');o.value=p.cluster;o.textContent='C'+String(p.cluster).padStart(2,'0');cluster.append(o);}if([...cluster.options].some(o=>o.value==previous))cluster.value=previous;show();}
function show(){const p=pages.find(p=>p.stage==stage.value&&p.method==method.value&&p.cluster==cluster.value);if(!p)return;document.getElementById('image').src=p.image;document.getElementById('full').href=p.image;document.getElementById('meta').textContent=`N=${p.samples}；平均 silhouette=${p.mean_silhouette.toFixed(4)}；影像類別數=${p.distinct_categories}；代表樣本數：高分 ${p.high_selected}／近零 ${p.near_zero_selected}／負分 ${p.negative_selected}`;}
stage.onchange=method.onchange=updateOptions;cluster.onchange=show;updateOptions();</script></html>'''
    (output/"index.html").write_text(gallery.replace('__PAGES__',json.dumps(pages)))
    (output/"selection_manifest.json").write_text(json.dumps(dict(seed=args.seed,per_bin=args.per_bin,near_zero=args.near_zero,
        methods=METHODS,groups=len(pages),selection="most negative and highest positive first; near-zero excludes already selected IDs; no duplicates within group; no out-of-bin fallback",
        source=str(root/"heldout_sample_metrics.csv"),pixel_note="token grid footprint, not full attention receptive field"),indent=2))
    unchanged={stage:np.mean([r['same_token'] for r in audit if r['stage']==stage]) for stage in range(3)}
    lines=["# 1005 K-means 群解釋性：高分、近零、負分代表影像", "",
        f"使用全類別 held-out 原图，每組 seed={args.seed}；3 stages × 3 methods × 31 clusters，共 {len(pages)} 張群代表圖。",
        f"每群最高正分、最接近 0（|s|≤{args.near_zero}）、最負分各最多 {args.per_bin} 張；不足時顯示 No eligible sample。高分是該群正分的相對排名，不表示一定具有高絕對 silhouette。",
        "優先選最負樣本與最高正分，再選尚未列過的近零樣本；每群不重複 source ID。近零樣本可能是正值或負值；沒有符合門檻的樣本時不以其他分數補位。", "",
        f"[互動圖庫](../plots/kmeans/quality_evaluation_1005/cluster_meaning_seed{args.seed}/index.html)", "",
        "## 座標稽核的重要發現", "",
        "Caltech-101 的 obj_contour 是相對 bounding box 的座標。資料集附帶 show_annotation.m 在繪製輪廓時，x 加 box_coord(3)、y 加 box_coord(1)；既有抽特徵流程未加這兩個 offset。",
        "本圖庫以 obj_contour + [box_x_min, box_y_min] − 1 轉成 Python 圖片座標，再取輪廓點平均作中心近似。這是輪廓點的平均位置，不是物體 mask 的面積質心。",
        f"全體 1,770 張 held-out 圖片，修正前後仍落在相同 token 的比例：Stage 0 {unchanged[0]:.1%}、Stage 1 {unchanged[1]:.1%}、Stage 2 {unchanged[2]:.1%}。完整位移與 token 比對存於 coordinate_audit.csv。",
        "原分群與 silhouette 保持原取樣結果。紅十字是原實驗取樣位置；綠十字是修正後輪廓中心；黃框是原實驗取樣的 token 範圍。每欄依序呈現全圖、修正中心附近 64px 內容、原取樣 token 的放大內容。",
        "這使既有樣本可以和實際物體中心對照，但不能用這批結果宣稱『物體質心分群』或部件語義。後續必須修正特徵抽取座標、重新建立 features cache 並重新 fitting；既有座標版本與新版本應分開記錄。", "",
        "每個樣本顯示 source ID、原圖類別、silhouette、平均距離最近的另一群。",
        "最近另一群依 silhouette 的 b(i) 定義，計算該樣本到每個其他群所有 held-out 樣本的平均 cosine distance，選最小者；不是最近 centroid。負分時另一群平均距離小於本群平均距離，可到對應群圖檢視。", "",
        "token 框是空間網格對應範圍；此模型使用 self-attention，不能將小框當作特徵完整感受野。原圖類別僅提供上下文，不能當作局部部件的 ground truth。", "",
        "## 逐群圖片索引", "",
    ]
    for stage in range(3):
        for method in METHODS:
            lines.extend([f"### Stage {stage} / {method}", ""])
            lines.append(" · ".join(f"[C{p['cluster']:02d}](../plots/kmeans/quality_evaluation_1005/cluster_meaning_seed{args.seed}/{p['image']})" for p in pages if p['stage']==stage and p['method']==method))
            lines.append("")
    lines.extend(["## 如何判讀", "",
        "1. 高分樣本的局部顏色、材質、邊緣或部件是否相似？跨影像類別但局部相似，可支持群具有局部視覺意義。",
        "2. 近零樣本是否混合兩種局部模式，或落在物體／背景邊界？",
        "3. 負分樣本是否在局部內容上更像它的 nearest other group？需对照另一群的高分樣本。",
        "4. 若只有全圖類別相似、局部內容不同，不能直接認定為部件群；也可能反映框外 attention 上下文。",
        "5. 每群只列少量、按分數挑選的案例，供人工檢視，不是整群語義品質的統計證明。群 ID 不應跨方法直接配對。", "",
        f"完整群大小與 top categories：`plots/kmeans/quality_evaluation_1005/cluster_meaning_seed{args.seed}/cluster_summary.csv`；每張入選圖片與框座標：`representative_samples.csv`。",
        "重現：`.venv/bin/python scripts/visualize_kmeans_cluster_meaning.py --seed 42`。可用 `--seed 43`、`--seed 44` 產生對應 seeds 圖庫。",
    ])
    Path(f"plan/1005_kmeans_cluster_meaning_seed{args.seed}.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
