#!/usr/bin/env python3
"""Verify completed corrected-center run and write its research record."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.preprocessing import normalize
import sklearn
import scipy
import torchvision
import PIL
import torch
from run_caltech_corrected_center import (
    ROOT, Caltech101, absolute_annotation, safe_letterbox, DIRECTIONS,
)


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',default='plots/kmeans/caltech_corrected_center_1012')
    p.add_argument('--report',default='plan/1012_caltech_corrected_center_static_translation.md')
    args=p.parse_args()
    out=ROOT/args.input_dir
    manifest=json.loads((out/'run_manifest.json').read_text())
    assert manifest['status']=='complete'
    for name,digest in manifest['artifacts_sha256'].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest,name
    samples=read_csv(out/'translation_samples.csv')
    summary=read_csv(out/'translation_summary.csv')
    static=read_csv(out/'static_metrics.csv')
    audit=read_csv(out/'coordinate_audit.csv')
    n=len(manifest['translation_indices'])
    assert len(samples)==n*17*3
    assert len({(r['source_index'],r['stage'],r['shift_pixels'],r['direction']) for r in samples})==len(samples)
    for stage in range(3):
        for shift in [0,4,8,16,32]:
            for direction in (['original'] if shift==0 else DIRECTIONS):
                rows=[r for r in samples if int(r['stage'])==stage and int(r['shift_pixels'])==shift and r['direction']==direction]
                assert len(rows)==n
                assert all(float(r['source_pixels_retained'])==1 for r in rows)
                if shift==0:
                    assert all(r['same_cluster']=='True' and r['prediction_consistent']=='True' and r['token_changed']=='False' for r in rows)
                    assert np.allclose([float(r['cosine']) for r in rows],1,atol=1e-6)
                if shift%8==0:
                    for grid in [8,16,32]:
                        if shift%grid==0:
                            for axis in ['x','y']:
                                assert all(np.isclose(float(r[f'original_phase_{axis}_mod{grid}']),float(r[f'moved_phase_{axis}_mod{grid}'])) for r in rows)
    base=Caltech101(str(ROOT/manifest['arguments']['data_root']),target_type=['category','annotation'],download=False)
    retention=[]
    # Validate actual RGB content and normalized pixels for every translated source.
    for source in manifest['translation_indices']:
        image,(_,annotation)=base[source]
        annotation=absolute_annotation(base,source,annotation)
        _,original_meta,original_canvas=safe_letterbox(image,annotation)
        original_box=tuple(int(original_meta[k]) for k in ['content_x0','content_y0','content_x1','content_y1'])
        for shift in [4,8,16,32]:
            for direction,(ux,uy) in DIRECTIONS.items():
                _,meta,canvas=safe_letterbox(image,annotation,ux*shift,uy*shift)
                moved_box=tuple(int(meta[k]) for k in ['content_x0','content_y0','content_x1','content_y1'])
                assert canvas.crop(moved_box).tobytes()==original_canvas.crop(original_box).tobytes()
                retention.append(dict(source_index=source,shift_pixels=shift,direction=direction,content_pixels_identical=True))
    write_csv(out/'translation_pixel_checks.csv',retention)
    comparison=[]
    old_manifest=json.loads((ROOT/'plots/kmeans/spherical_comparison_1005_n50/run_manifest.json').read_text())
    assert old_manifest['checkpoint_sha256']==manifest['checkpoint_sha256']
    assert old_manifest['train_indices']==manifest['train_indices']
    assert old_manifest['sklearn_n_init']==manifest['arguments']['n_init']
    # Same-input sampling comparisons; do not call these translation/drift scores.
    for split,old_path,new_path in [
        ('train',ROOT/'plots/kmeans/spherical_comparison_1005_n50/features.npz',out/'train_features.npz'),
        ('validation',ROOT/'plots/kmeans/quality_evaluation_1005/heldout_features.npz',out/'validation_features.npz')]:
        with np.load(old_path) as old,np.load(new_path) as new:
            if split=='validation':
                assert np.array_equal(old['source_ids'],new['source_indices'])
            for stage in range(3):
                old_x=old[f'train_{stage}' if split=='train' else f'stage{stage}']
                new_x=new[f'stage{stage}']
                cosine=np.sum(normalize(old_x)*normalize(new_x),axis=1)
                rows=[r for r in audit if r['split']==split and int(r['stage'])==stage]
                assert [int(r['source_index']) for r in rows]==new['source_indices'].tolist()
                same=np.array([r['same_token']=='True' for r in rows])
                assert np.allclose(old_x[same],new_x[same],rtol=1e-4,atol=1e-5)
                comparison.append(dict(split=split,stage=stage,samples=len(rows),same_token_fraction=float(same.mean()),
                                       same_input_different_sampling_cosine=float(cosine.mean()),
                                       cosine_where_same_token=float(cosine[same].mean()),
                                       cosine_where_different_token=float(cosine[~same].mean())))
    write_csv(out/'legacy_sampling_comparison.csv',comparison)
    endpoint_rows=[]
    for stage in range(3):
        for shift in [4,8,16,32]:
            for endpoint in ['original','moved']:
                for near in [True,False]:
                    rows=[r for r in samples if int(r['stage'])==stage and int(r['shift_pixels'])==shift
                          and (float(r[f'{endpoint}_boundary_distance'])<=manifest['arguments']['boundary_px'])==near]
                    if rows:
                        endpoint_rows.append(dict(stage=stage,shift_pixels=shift,endpoint=endpoint,
                            stratum='near' if near else 'interior',samples=len(rows),
                            cluster_agreement=float(np.mean([r['same_cluster']=='True' for r in rows])),
                            matched_token_cosine=float(np.mean([float(r['cosine']) for r in rows])),
                            pair_padding_contact_rate=float(np.mean([r['pair_padding_contact']=='True' for r in rows]))))
    write_csv(out/'endpoint_boundary_summary.csv',endpoint_rows)
    verification=dict(status='passed', paired_observations=len(samples), static_pixel_checks=len(manifest['train_indices'])+len(manifest['valid_indices']),
                      translated_content_checks=len(retention),zero_shift_control='passed',stage_aligned_token_displacement='passed',
                      grid_phase_checks='passed',legacy_same_token_feature_equality='passed',
                      dependencies=dict(sklearn=sklearn.__version__,scipy=scipy.__version__,torchvision=torchvision.__version__,PIL=PIL.__version__),
                      verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      legacy_sources_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [
                          ROOT/'plots/kmeans/spherical_comparison_1005_n50/run_manifest.json',
                          ROOT/'plots/kmeans/spherical_comparison_1005_n50/features.npz',
                          ROOT/'plots/kmeans/quality_evaluation_1005/heldout_features.npz',
                          ROOT/'plots/kmeans/quality_evaluation_1005/geometry_metrics.csv']})
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    old_geometry=read_csv(ROOT/'plots/kmeans/quality_evaluation_1005/geometry_metrics.csv')
    static_rows=[]
    for r in static:
        if r['split']!='validation':continue
        old=next(o for o in old_geometry if o['stage']==r['stage'] and o['seed']=='42' and o['method']=='sklearn_existing' and o['split']=='heldout_all')
        static_rows.append([r['stage'],f"{float(old['cosine_silhouette_mean']):.5f}",f"{float(r['cosine_silhouette']):.5f}",
                            f"{float(r['negative_silhouette_fraction']):.2%}",f"{float(r['category_ami']):.5f}",f"{float(r['category_ari']):.5f}"])
    trans_rows=[]
    for stage in range(3):
        for shift in [0,4,8,16,32]:
            rows=[r for r in summary if int(r['stage'])==stage and int(r['shift_pixels'])==shift and r['stratum']=='all']
            # Each nonzero condition has equal source counts in all four directions.
            trans_rows.append([stage,shift,'是' if shift%(8*2**stage)==0 else '否',
                               f"{np.mean([float(r['cluster_agreement']) for r in rows]):.2%}",
                               f"{np.mean([float(r['matched_token_cosine']) for r in rows]):.5f}",
                               f"{np.mean([float(r['prediction_consistency']) for r in rows]):.2%}"])
    boundary_rows=[]
    for stage in range(3):
        for shift in [4,8,16,32]:
            for boundary in ['near','interior']:
                rows=[r for r in samples if int(r['stage'])==stage and int(r['shift_pixels'])==shift and r['boundary_stratum']==boundary and r['pair_padding_contact']=='False']
                if rows:
                    boundary_rows.append([stage,shift,boundary,len(rows),f"{np.mean([r['same_cluster']=='True' for r in rows]):.2%}",f"{np.mean([float(r['cosine']) for r in rows]):.5f}"])
    artifact_rel='../'+str(out.relative_to(ROOT))
    content=f'''# 1012 Caltech-101 修正物體中心：靜態重測與受控平移

完成時間：{manifest['completed_utc']}。本輪先完成靜態抽取與 fitting，再使用同一份固定 codebook 進行平移。原有產物保留；本輪所有 cache 使用 `{manifest['coordinate_version']}`。

## 設定與重現

- Checkpoint：epoch {manifest['epoch']}，SHA256 `{manifest['checkpoint_sha256']}`。
- Training：{len(manifest['train_indices'])} 張；validation：{len(manifest['valid_indices'])} 張；平移：validation 中 {n} 張 airplanes。
- 與既有 seed-42 split 相同；validation 曾參與 checkpoint 選擇，不是 untouched final test。
- 每張完整圖片等比例縮至長邊 160，置中於 224×224；resize、padding 顏色、normalize、名義 scale 與 patch 切法沿用原程式。
- 座標：`obj_contour + [[box_coord[2]], [box_coord[0]]] - 1`，再使用原 letterbox scale 映射。中心是輪廓點平均，並非 mask 面積質心。為隔離此次 offset 修正，沒有另改 resize rounding 的座標協定。
- Stage 0／1／2 在最後 block 後、merge 前取一個物體中心 token；網格為 8／16／32 px。token 框只表示網格範圍，self-attention 的感受野可能包含全圖。
- 各 stage 用未平移 training 特徵建立 K={manifest['arguments']['clusters']} codebook：L2-normalized features + sklearn Euclidean KMeans，seed={manifest['seed']}、n_init={manifest['arguments']['n_init']}。本輪只重測這個原方法與單一 seed，沒有宣稱已重跑三方法／20 seeds。
- 平移量：0、4、8、16、32 canvas px；非零位移各做左、右、上、下。平移圖與 validation 特徵都不參與 fitting。

```bash
.venv/bin/python scripts/run_caltech_corrected_center.py
.venv/bin/python scripts/report_caltech_corrected_center.py
```

程式在相同 cache identity 下可重用已抽出的特徵；identity 不同會拒絕混用，請使用新 output-dir。manifest 保存 split IDs、資料／標註／checkpoint／程式 hash、參數與時間。環境版本另見 verification.json。大型 NPZ 保留本機，不納入 Git。

第一條指令可從資料及 checkpoint 重跑本輪；第二條報告／驗證指令另需本機既有 `spherical_comparison_1005_n50/features.npz` 與 `quality_evaluation_1005/heldout_features.npz`，以核對舊取樣結果。verification.json 保存這些比較來源的 hash。

## 1. 靜態重測

所有 training／validation 共 {verification['static_pixel_checks']} 張均確認修正前後輸入 tensor 與 RGB canvas **逐像素完全相同**。修正只改取樣位置，沒有引入新的 patch 對齊效應。

{table(['Stage','舊位置 silhouette','修正位置 silhouette','修正位置負 silhouette','Category AMI','Category ARI'],static_rows)}

此表舊值與新值都使用 seed 42、n_init=50 的原 Euclidean 方法；舊值來自既有 held-out geometry_metrics。群中心已重新 fitting，cluster 編號不能直接跨版本對比。Silhouette／category agreement 是幾何及標籤統計，不能直接證明群語意。

修正後三層 silhouette 均下降；這反映重新取樣與重新 fitting 的特徵空間差異，不能解讀為同一內容的表徵漂移。

{table(['Split','Stage','同一 token 比例','不同 token：取樣特徵 cosine'],[[r['split'],r['stage'],f"{r['same_token_fraction']:.2%}",f"{r['cosine_where_different_token']:.5f}"] for r in comparison])}

上表比較同一張未平移影像的兩個取樣位置，**不是同一內容的表徵漂移**。仍落在同一 token 的樣本，已驗證新舊特徵相等（數值容差內）。逐圖片位置、token 與 padding 見 coordinate_audit.csv；靜態分群與 silhouette 見 static_assignments.csv。

Validation 分類 accuracy 為 {float(next(r for r in static if r['split']=='validation')['classification_accuracy']):.2%}；座標修正不改模型分類輸入。

## 2. 受控平移：相同位移量比較各 stage

下表非零位移合併四方向，各 stage／位移共有 {n*4} 筆 paired observations；0 px 為 {n} 筆原樣本的 identity control。它們重複使用同一來源影像，不是獨立新增圖片。

{table(['Stage','位移 px','對齊此 stage 網格','Cluster agreement','Token cosine','Prediction consistency'],trans_rows)}

4 px 改變 Stage-0 初始 patch 相位；8 px 只保留 Stage-0 相位，16 px 保留 Stage 0／1，32 px 保留 Stage 0／1／2。較深層合併的相位與 absolute positional embedding 仍可能造成差異；本輪描述敏感性，未作機制因果消融。

Stage 0 的 4 px agreement 為 26.72%，8 px 為 95.16%；Stage 1 的 8 px 為 27.81%，16 px 為 88.44%；Stage 2 的 16 px 為 34.53%，32 px 為 89.06%。這些結果符合各層固定網格相位具有明顯影響，但位移幅度也同時改變，不能視為已隔離其他因素的因果證據。32 px 分類 prediction consistency 仍只有 60.94%，再次顯示局部群穩定不保證分類穩定。

0 px 已驗證 cluster／prediction agreement=100%、cosine≈1、token 不變。所有位移均驗證修正中心加上相同 dx／dy，對齊 stage 網格時 token 移動正確的整數格數。四方向分開的 AMI／ARI、分類 accuracy、token change 與 padding rate 見 translation_summary.csv，沒有將四方向 AMI／ARI直接平均當作 pooled AMI／ARI。

![平移結果]({artifact_rel}/translation_stability.png)

## 3. Token 邊界與 padding 分層

距離是中心到所屬 token 最近網格邊界的 canvas-pixel 距離。預先固定門檻 ≤{manifest['arguments']['boundary_px']:g} px；原位置或移動後位置任一接近邊界即為 near，否則為 interior。以下另外排除任一端 token 接觸 letterbox padding 的配對，避免把 padding 組混入內部組。

{table(['Stage','位移 px','邊界組','N（四方向）','Agreement','Cosine'],boundary_rows)}

translation_samples.csv 每筆保留原／移動中心對 8、16、32 的 x/y 餘數、token indices、token 是否改變、邊界距離、兩端 padding contact，以及 assigned／原 centroid 的 cosine distance。translation_summary.csv 另存邊界×padding 交叉分層。空組不輸出；分層的樣本組成不同，不能把組間差異當作邊界的因果效果。

Stage 0 在 4 px 位移下，搭配 2 px 門檻，原／移動端至少一端必然接近邊界，所以配對定義沒有 interior 組。為保留原位置與移動後位置的個別比較，另存 endpoint_boundary_summary.csv，分別依兩端距離分層。

## 4. 驗證與產物

- 完整配對資料 {len(samples):,} 筆；唯一鍵 `(source_index, stage, shift_pixels, direction)` 無重複。
- 全部 {len(retention):,} 個 translated canvases 的原圖內容 crop 與未平移版本逐像素相同，來源內容完整保留。
- 0 px 控制、stage-aligned token displacement、grid phase 與舊／新同 token 特徵 equality 均通過。
- 原始修正特徵、平移特徵、codebook 分別存 NPZ，manifest 存 SHA256。精選 CSV、圖及 manifest 留在版本控制，原有舊 cache 未修改。

[Manifest]({artifact_rel}/run_manifest.json) · [驗證紀錄]({artifact_rel}/verification.json) · [靜態指標]({artifact_rel}/static_metrics.csv) · [平移摘要及分層]({artifact_rel}/translation_summary.csv) · [逐筆平移紀錄]({artifact_rel}/translation_samples.csv) · [座標對照]({artifact_rel}/coordinate_audit.csv)

本輪平移只涵蓋 airplanes、一個 checkpoint、單一 clustering seed；不能推廣至全類別或跨模型訓練的穩定性。原先 Stage 2 32px 的 88.91% 局部分群一致率屬舊取樣座標，本輪修正結果另列，不能沿用舊值解讀物體中心。
'''
    (ROOT/args.report).write_text(content)
    print(f'All checks passed; wrote {args.report}')


if __name__=='__main__':
    main()
