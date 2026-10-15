#!/usr/bin/env python3
"""Auditable two-row cluster-change cases for stage-aligned translations."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import run_caltech_corrected_center as c
import test_caltech_translation_kmeans as l

OUT=Path('plots/kmeans/caltech_aligned_translation_1014/case_analysis')
CHECKPOINTS={'baseline':'runs/train/mergingvit_caltech101_letterbox160/MergingViT_best.pth',
             'augmented':'runs/train/mergingvit_caltech101_aligned_translation_1014/MergingViT_best.pth'}


def canvas_crop(base,source,stage,dx=0,dy=0):
    image,(_,anno)=base[int(source)]
    _,meta,canvas=l.safe_letterbox(image,c.absolute_annotation(base,int(source),anno),dx,dy)
    size=8*2**stage
    x,y=c.position(meta,size)
    box=(x*size,y*size,(x+1)*size,(y+1)*size)
    return canvas,canvas.crop(box),box


def plot_case(base,row,reps,path):
    stage=row['stage']; size=8*2**stage
    original,crop0,box0=canvas_crop(base,row['source_index'],stage)
    moved,crop1,box1=canvas_crop(base,row['source_index'],stage,row['dx'],row['dy'])
    assert crop0.tobytes()==crop1.tobytes(),row
    fig,axes=plt.subplots(2,7,figsize=(18,8))
    for ri,(canvas,crop,box,cluster,pred,color) in enumerate([
        (original,crop0,box0,row['original_cluster'],row['original_prediction'],'red'),
        (moved,crop1,box1,row['moved_cluster'],row['moved_prediction'],'lime')]):
        ax=axes[ri,0];ax.imshow(canvas)
        ax.add_patch(Rectangle(box[:2],size,size,fill=False,edgecolor=color,linewidth=2))
        ax.set_title(('Original' if ri==0 else f"{row['direction']} {size}px")+f'\nPred: {base.categories[pred]}',fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        axes[ri,1].imshow(crop,interpolation='nearest');axes[ri,1].set_title(f'Token crop\nCluster {cluster}');axes[ri,1].axis('off')
        for ci,source in enumerate(reps[stage][cluster]):
            _,rcrop,_=canvas_crop(base,source,stage)
            axes[ri,ci+2].imshow(rcrop,interpolation='nearest')
            axes[ri,ci+2].set_title(f'Cluster {cluster} rep {ci+1}\ntrain #{source}',fontsize=8)
            axes[ri,ci+2].axis('off')
    fig.suptitle(f"{row['model']} / {row['category']} / source #{row['source_index']} / Stage {stage}\n"
                 f"{row['group']} | cosine={row['cosine']:.5f} | identical aligned crops",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.9),h_pad=3.5,w_pad=1);fig.savefig(path,dpi=160);plt.close(fig)


def write_analysis(rows,cases):
    lines=['# 案例分析重點','',
           '每模型皆使用同樣的來源影像、四方向及 stage-aligned 位移。每張影像有 3 stages × 4 directions = 12 筆配對，並非 12 張獨立影像。',
           '換群表示同一模型／stage 的離散 codebook 指派改變；不直接等同語意改變。', '',
           '| Model | Class | Pairs | 換群／預測改變 | 不換群／預測改變 | 換群／預測不變 | 不換群／預測不變 |',
           '|---|---|---:|---:|---:|---:|---:|']
    groups=['cluster_changed__prediction_changed','cluster_same__prediction_changed','cluster_changed__prediction_same','cluster_same__prediction_same']
    for model in ['baseline','augmented']:
        for cat in ['airplanes','chandelier','brain']:
            subset=[r for r in rows if r['model']==model and r['category']==cat]
            counts=[sum(r['group']==g for r in subset) for g in groups]
            lines.append(f"| {model} | {cat} | {len(subset)} | "+' | '.join(map(str,counts))+' |')
    lines += ['', '## 值得先看的案例','']
    examples=[('augmented','chandelier',3876,0,'cluster_changed__prediction_same'),
              ('augmented','brain',3198,0,'cluster_same__prediction_changed'),
              ('augmented','chandelier',3911,0,'cluster_changed__prediction_changed'),
              ('augmented','chandelier',3921,0,'cluster_same__prediction_same')]
    for model,cat,source,stage,group in examples:
        r=next((r for r in cases if r['model']==model and r['category']==cat and int(r['source_index'])==source and int(r['stage'])==stage and r['group']==group),None)
        if r:
            lines += [f"- [{cat} #{source}: {group}]({r['path']})：cluster {r['original_cluster']}→{r['moved_cluster']}，cosine {float(r['cosine']):.5f}。"]
    lines += ['', '## 可支持的解讀','',
              '1. 換群不必然改變預測；不換群也不保證預測不變。中心 token 無法單獨代表整張影像的決策。',
              '2. 高 cosine 仍可能跨越 K-means 指派邊界；案例不能只用 cosine 或群編號判斷語意穩定性。',
              '3. 兩組模型的群中心分別 fitting，因此只比較各模型自己的前後指派，不能把不同模型相同編號當成相同概念。',
              '4. 局部 crop 逐像素相同，仍可能因位置編碼、self-attention 與後續處理改變 activation；目前未隔離機制。',
              '5. 另外兩類是依 validation 表現挑選的探索性案例；單一 seed，沒有全類別或 final-test 推論。',
              '', '[全部案例與分 stage 計數](REPORT.md) · [Mermaid 流程](../../../../plan/1014_aligned_translation_workflow.md)']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines))


def main():
    torch.set_num_threads(4);threadpool_limits(4)
    OUT.mkdir(parents=True,exist_ok=True)
    l.CONTENT_MAX_SIZE=160
    base=l.Caltech101('data',target_type=['category','annotation'],download=False)
    train,valid=l.split_indices('data',42);valid=sorted(valid)
    assert not set(train)&set(valid)
    manifest=dict(status='running',code_sha256=c.sha(__file__),models={},train_indices=train,valid_indices=valid,
                  coordinate_version=c.COORDINATE_VERSION,content_max=160,canvas=224,
                  selection='airplanes plus two lowest augmented original-image accuracy classes with >=20 validation images; exploratory',
                  case_selection='one unique source per model/class/stage/group; changed clusters ranked by highest cosine; stable ranked by highest cosine',
                  stages={0:8,1:16,2:32},K=31,n_init=50,seed=42)
    def save_manifest(): (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    originals={};training={};models={};screen=[]
    for name,path in CHECKPOINTS.items():
        ckpt=torch.load(path,map_location='cpu',weights_only=False)
        model=l.MergingViT(**ckpt['model_args']).cuda().eval();model.load_state_dict(ckpt['model_weights'])
        models[name]=model
        manifest['models'][name]={'path':path,'sha256':c.sha(path)};save_manifest()
        def get_result(tag,indices):
            dest=OUT/f'{name}_{tag}.npz'
            identity=dict(checkpoint_sha256=c.sha(path),indices=indices,extractor_sha256=c.sha(l.__file__),
                          coordinate_sha256=c.sha(c.__file__),content_max=160)
            ident=dest.with_suffix('.identity.json')
            if dest.exists():
                assert json.loads(ident.read_text())==identity,'cache mismatch'
                return c.read_result(dest)
            result=l.extract(model,c.CorrectedSubset(base,indices),32,4,torch.device('cuda:0'),8)
            c.save_result(dest,result);ident.write_text(json.dumps(identity))
            return result
        originals[name]=get_result('original',valid)
        training[name]=get_result('training',train)
        for cat,label in enumerate(base.categories):
            idx=np.flatnonzero(originals[name]['labels']==cat)
            screen.append(dict(model=name,category=label,samples=len(idx),accuracy=float(np.mean(originals[name]['predictions'][idx]==cat))))
        print(name,'screened/extracted',flush=True)
    c.write_csv(OUT/'class_screening.csv',screen)
    eligible=sorted([r for r in screen if r['model']=='augmented' and r['samples']>=20 and r['category']!='airplanes'],key=lambda r:(r['accuracy'],-r['samples'],r['category']))
    classes=['airplanes']+[r['category'] for r in eligible[:2]]
    manifest['selected_classes']=classes;save_manifest();print('Selected',classes,flush=True)
    ids=[i for i in valid if base.categories[base.y[i]] in classes]
    offsets=[valid.index(i) for i in ids]
    allrows=[];summaries=[];cases=[];representatives=[]
    for name,model in models.items():
        orig=originals[name];tr=training[name]
        kms=[];reps=[]
        for stage in range(3):
            f=normalize(tr['features'][stage]);km=KMeans(n_clusters=31,n_init=50,random_state=42).fit(f);kms.append(km)
            sr={}
            for cluster in range(31):
                members=np.flatnonzero(km.labels_==cluster)
                distances=np.linalg.norm(f[members]-km.cluster_centers_[cluster],axis=1)
                nearest=members[np.argsort(distances,kind='stable')[:5]]
                sr[cluster]=[train[i] for i in nearest]
                representatives.extend(dict(model=name,stage=stage,cluster=cluster,rank=rank+1,source_index=train[i],distance=float(distances[np.flatnonzero(members==i)[0]])) for rank,i in enumerate(nearest))
            reps.append(sr)
        for stage,shift in enumerate([8,16,32]):
            before=normalize(orig['features'][stage][offsets]);ca=kms[stage].predict(before)
            for direction,(ux,uy) in c.DIRECTIONS.items():
                moved=l.extract(model,c.CorrectedSubset(base,ids,ux*shift,uy*shift),32,4,torch.device('cuda:0'),8)
                after=normalize(moved['features'][stage]);cb=kms[stage].predict(after)
                distances=kms[stage].transform(after);sortedd=np.sort(distances,axis=1)
                for j,source in enumerate(ids):
                    i=offsets[j];pred0=int(orig['predictions'][i]);pred1=int(moved['predictions'][j]);label=base.y[source]
                    same=bool(ca[j]==cb[j]);consistent=pred0==pred1
                    old,new=orig['metadata'][i],moved['metadata'][j]
                    assert np.isclose(new['center_x']-old['center_x'],ux*shift)
                    assert np.isclose(new['center_y']-old['center_y'],uy*shift)
                    group=('cluster_same' if same else 'cluster_changed')+'__'+('prediction_same' if consistent else 'prediction_changed')
                    allrows.append(dict(model=name,category=base.categories[label],source_index=source,stage=stage,shift_pixels=shift,direction=direction,dx=ux*shift,dy=uy*shift,
                                        original_cluster=int(ca[j]),moved_cluster=int(cb[j]),same_cluster=same,cosine=float(before[j]@after[j]),
                                        original_prediction=pred0,moved_prediction=pred1,prediction_consistent=consistent,original_correct=pred0==label,moved_correct=pred1==label,
                                        moved_assignment_margin=float(sortedd[j,1]-sortedd[j,0]),group=group))
            print(name,'stage',stage,'complete',flush=True)
        for cat in classes:
            for stage in range(3):
                subset=[r for r in allrows if r['model']==name and r['category']==cat and r['stage']==stage]
                used=set()
                for group in ['cluster_changed__prediction_changed','cluster_same__prediction_changed','cluster_changed__prediction_same','cluster_same__prediction_same']:
                    group_rows=[r for r in subset if r['group']==group]
                    summaries.append(dict(model=name,category=cat,stage=stage,shift_pixels=8*2**stage,group=group,pairs=len(group_rows),total_pairs=len(subset),unique_sources=len({r['source_index'] for r in group_rows}),mean_cosine=float(np.mean([r['cosine'] for r in group_rows])) if group_rows else None))
                    candidates=sorted(group_rows,key=lambda r:(-r['cosine'],r['source_index'],r['direction']))
                    chosen=next((r for r in candidates if r['source_index'] not in used),None)
                    if chosen is None: continue
                    used.add(chosen['source_index'])
                    dest=OUT/name/cat/f'stage{stage}'/group;dest.mkdir(parents=True,exist_ok=True)
                    path=dest/f"case_01_source_{chosen['source_index']}.png"
                    plot_case(base,chosen,reps,path)
                    cases.append(dict(**chosen,path=str(path.relative_to(OUT)),identical_crop_verified=True))
        c.write_csv(OUT/'paired_metrics.csv',allrows);c.write_csv(OUT/'group_counts.csv',summaries)
        c.write_csv(OUT/'selected_cases.csv',cases);c.write_csv(OUT/'representatives.csv',representatives)
    report=['# 平移換群／不換群案例分析','',
            '原圖與平移圖分列：整圖＋token crop＋該群五個 training 代表樣本。每張案例已驗證內容對齊 crop 逐像素相同。',
            'Stage 0/1/2 分別使用 8/16/32 px，四方向；160-in-224；64 px 暫緩。Codebook 與既有評估相同規格，各模型獨立，以未平移全類別 training 中心 token 擬合 K=31、n_init=50。',
            '中心為修正 annotation offset 後的輪廓點平均；token 框是網格來源範圍，不是完整感受野。群代表是最接近 centroid 的 training token crop。',
            '這是同一個 stage codebook 跨位置比較；不是每個位置獨立 fitting。cluster ID 不跨模型對應。',
            '保留 airplanes，另依增強模型原圖 validation accuracy 最低且 n>=20 選兩類。這是探索性挑選，不作全資料集效能推論。Validation 曾參與選模；單一訓練 seed。',
            '案例優先選 cosine 高的換群樣本；各 model/class/stage 內案例 source 不重複。缺少的組別以完整計數顯示，不能因未展示就推定不存在。',
            '', '| Class | N | Baseline original accuracy | Augmented original accuracy |','|---|---:|---:|---:|']
    for cat in classes:
        a=next(r for r in screen if r['category']==cat and r['model']=='baseline');b=next(r for r in screen if r['category']==cat and r['model']=='augmented')
        report.append(f"| {cat} | {a['samples']} | {a['accuracy']:.2%} | {b['accuracy']:.2%} |")
    report+=['','## 四種組別計數','', '| Model | Class | Stage / px | Cluster | Prediction | Pairs / total |','|---|---|---|---|---|---:|']
    for r in summaries:
        cluster,pred=r['group'].split('__')
        report.append(f"| {r['model']} | {r['category']} | {r['stage']} / {r['shift_pixels']} | {cluster} | {pred} | {r['pairs']} / {r['total_pairs']} |")
    report+=['','同一影像的四方向重複使用；pair 數不是獨立影像數。Prediction same 可包含兩次皆錯，不等於 correct。','', '## 案例圖','']
    for r in cases:
        report.extend([f"### {r['model']} / {r['category']} / Stage {r['stage']} / {r['group']}",'',f"Source {r['source_index']}, {r['direction']} {r['shift_pixels']}px; cosine={r['cosine']:.5f}; cluster {r['original_cluster']}→{r['moved_cluster']}; prediction {base.categories[r['original_prediction']]}→{base.categories[r['moved_prediction']]}",'',f"![case]({r['path']})",''])
    (OUT/'REPORT.md').write_text('\n'.join(report))
    write_analysis(allrows,cases)
    manifest['status']='complete';manifest['artifacts_sha256']={str(f.relative_to(OUT)):c.sha(f) for f in OUT.rglob('*') if f.suffix in ['.png','.csv','.md']};save_manifest()
    print('COMPLETE',len(cases),'cases',flush=True)

if __name__=='__main__':main()
