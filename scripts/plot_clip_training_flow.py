#!/usr/bin/env python3
"""Reproducible actual-image diagrams for the deep-CLIP training pipeline."""
import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from scipy.io import loadmat
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs,object_crop,MEAN,STD
from clip.clip import _transform
OUT=ROOT/'plots/reports/clip_depth_training_flow_1103';OUT.mkdir(parents=True,exist_ok=True)
font_manager.fontManager.addfont(str(OUT/'assets/NotoSansTC-Medium.ttf'))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=OUT/'assets/NotoSansTC-Medium.ttf').get_name(),'font.size':12,'font.weight':'medium','axes.unicode_minus':False,'svg.fonttype':'path','pdf.fonttype':42})
BLUE='#2563a6';ORANGE='#cc6a21';INK='#172b42'

def text(fig,x,y,s,size=12,color=INK,ha='left',weight='medium'):
 fig.text(x,y,s,fontsize=size,color=color,ha=ha,va='top',weight=weight)
def arrow(fig,x1,y1,x2,y2,color=BLUE):
 fig.add_artist(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=15,lw=2,color=color))
def box(fig,x,y,w,h,label,color=BLUE,fill='#eef5fd',size=12):
 fig.add_artist(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009',transform=fig.transFigure,facecolor=fill,edgecolor=color,lw=1.5));fig.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=size,color=INK)
def panel(fig,xy,image,title,caption):
 x,y,w,h=xy;ax=fig.add_axes(xy);ax.imshow(image,cmap='gray' if np.asarray(image).ndim==2 else None);ax.axis('off');text(fig,x,y+h+.035,title,14,weight='bold');text(fig,x,y-.025,caption,11)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,facecolor='white')
 for svg in OUT.glob(name+'.svg'):
  svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
 plt.close(fig)

def main():
 config_path=ROOT/'runs/train/clip_depth_ablation_1103/deep/config.json'
 cfg=(json.loads(config_path.read_text()) if config_path.exists() else
      {'splits':{'train':[json.loads((OUT/'figure_manifest.json').read_text())['source_index']]}})
 ds=ObjectScalePairs(str(ROOT/'data'),cfg['splits']['train'],training=True)
 index=next(i for i in ds.indices if ds.base.categories[ds.base.y[i]]=='airplanes');ds.indices=[index]
 # Select a real deterministic training pair whose scales differ visibly.
 for epoch in range(50):
  ds.epoch=epoch;sample=ds[0]
  if sorted(sample['geometry'][:,[0,1]].max(1).values.tolist())==[64,128]:break
 image,(label,contour)=ds.base[index];image=image.convert('RGB');crop,mask,_,mat=object_crop(ds.base,index)
 b=loadmat(mat)['box_coord'].ravel();bounds=(max(0,int(b[2])-1),max(0,int(b[0])-1),min(image.width,int(b[3])),min(image.height,int(b[1])))
 points=contour+np.array([[b[2]],[b[0]]])-1
 overlay=image.copy();draw=ImageDraw.Draw(overlay);draw.rectangle(bounds,outline='#ed8a21',width=3);draw.line([tuple(p) for p in points.T]+[tuple(points[:,0])],fill='#00bfd1',width=2)
 rgb=np.asarray(crop)/255;alpha=np.asarray(mask)/255;yy,xx=np.indices(alpha.shape);checker=np.where(((xx//12+yy//12)%2)[...,None],.88,.98);cutout=rgb*alpha[...,None]+checker*(1-alpha[...,None])
 views=[((sample[k]*STD+MEAN).permute(1,2,0).numpy()).clip(0,1) for k in ['a','b']]
 teacher_crop=image.crop(bounds);square=Image.new('RGB',(max(teacher_crop.size),)*2,(123,117,104));square.paste(teacher_crop,((square.width-teacher_crop.width)//2,(square.height-teacher_crop.height)//2))
 teacher_tensor=_transform(224)(square);clip_mean=torch.tensor([.48145466,.4578275,.40821073])[:,None,None];clip_std=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]
 teacher_display=(teacher_tensor*clip_std+clip_mean).permute(1,2,0).numpy().clip(0,1)
 fig=plt.figure(figsize=(19,11));text(fig,.035,.965,'本次訓練的圖片處理流程',25,weight='bold');text(fig,.035,.912,'同一張來源圖分成學生與教師兩條路徑；僅物件縮放與平移，背景不跟著變換。',14)
 text(fig,.035,.856,'學生路徑  |  生成兩個完整畫布 view',17,BLUE,weight='bold')
 panel(fig,(.035,.56,.15,.24),overlay,'① 原圖＋標註','橘框：annotation bbox\n青線：物件輪廓')
 panel(fig,(.235,.56,.13,.24),mask,'② 輪廓轉遮罩','填滿輪廓得到 alpha mask\n依 mask 外框裁切物件')
 panel(fig,(.425,.56,.14,.24),cutout,'③ 物件 RGB × alpha','先乘 alpha，再分別 resize\n格紋只表示此處的背景被移除')
 for i,x in enumerate([.64,.82]):
  w,h,ox,oy=sample['geometry'][i].tolist();dx=ox-(224-w)//2;dy=oy-(224-h)//2
  panel(fig,(x,.56,.145,.24),views[i],f'④ View {chr(65+i)}：224 × 224',f'物件長邊 {max(w,h)} px\n位移 ({dx:+d}, {dy:+d}) px')
 arrow(fig,.19,.68,.225,.68);arrow(fig,.375,.68,.415,.68);arrow(fig,.575,.68,.63,.68)
 box(fig,.415,.415,.55,.055,'訓練取樣：長邊 ∈ {64, 96, 128}；dx、dy 各在 −16…+16\n兩個 view 各自取樣，但共用同一背景色與類別標籤',size=12)
 text(fig,.035,.385,'教師路徑  |  原始 bbox 的固定 CLIP 目標',17,ORANGE,weight='bold')
 panel(fig,(.035,.105,.15,.21),teacher_crop,'⑤ 原始 bbox crop','直接從高解析原圖裁切\n框內仍可能包含背景')
 panel(fig,(.245,.105,.15,.21),teacher_display,'⑥ 補正方形＋CLIP 前處理','等比例保留完整 bbox\n實際教師輸入：224 × 224')
 arrow(fig,.19,.21,.235,.21,ORANGE);arrow(fig,.405,.21,.445,.21,ORANGE)
 box(fig,.465,.155,.19,.105,'凍結 CLIP ViT-B/32\n只 encode image',ORANGE,'#fff4e9',14)
 arrow(fig,.665,.21,.715,.21,ORANGE);box(fig,.735,.155,.22,.105,'512 維 L2 正規化向量\n同時監督兩個學生 view',ORANGE,'#fff4e9',14)
 text(fig,.035,.035,'正規化：學生使用 ImageNet mean/std；教師使用 CLIP mean/std。上方 view 顯示正規化前的 RGB，背景色是成對固定、隨來源／epoch 改變。',11)
 save(fig,'01_image_processing')
 fig=plt.figure(figsize=(18,8));text(fig,.04,.95,'圖片進入模型後：CLIP 只直接監督深層',25,weight='bold');text(fig,.04,.885,'主要實驗：第 3、4 層；學生全模型可訓練，CLIP 教師凍結。',14)
 panel(fig,(.035,.53,.105,.25),views[0],'雙 view','A、B 分別通過同一模型')
 xs=[.20,.37,.54,.71]
 for i,x in enumerate(xs):
  active=i>=2;box(fig,x,.57,.115,.12,f'Stage {i+1}',ORANGE if active else BLUE,'#fff4e9' if active else '#eef5fd',15)
  if i==0:arrow(fig,.15,.63,x-.01,.63)
  else:arrow(fig,xs[i-1]+.125,.63,x-.01,.63)
  if active:
   arrow(fig,x+.057,.56,x+.057,.455,ORANGE);box(fig,x-.005,.355,.125,.09,'對應物件 region\n8 × 8 mask 加權讀取',ORANGE,'#fff4e9',11)
   arrow(fig,x+.057,.345,x+.057,.295,ORANGE);box(fig,x-.005,.205,.125,.08,'Linear → 512\nL2 正規化',ORANGE,'#fff4e9',11)
 arrow(fig,.835,.63,.865,.63);box(fig,.875,.57,.095,.12,'分類 head\nCE loss',BLUE,'#eef5fd',13)
 box(fig,.24,.205,.20,.08,'凍結 CLIP 教師向量',ORANGE,'#fff4e9',13);arrow(fig,.45,.225,.53,.10,ORANGE)
 box(fig,.54,.055,.285,.075,'CLIP cosine loss：兩層 × 兩 view 平均\n總損失 = CE + 0.2 × CLIP loss',ORANGE,'#fff4e9',12)
 arrow(fig,.60,.195,.64,.14,ORANGE);arrow(fig,.76,.195,.73,.14,ORANGE)
 text(fig,.04,.43,'Stage 1、2 無直接 CLIP loss\n深層梯度仍會回傳淺層；淺層沒有凍結。',13)
 text(fig,.04,.105,'消融對照\n全四層 0.2／前兩層 0.2\n後兩層 0.2／後兩層 0.1',12)
 text(fig,.865,.43,'推論：完整圖片 → 分類\n不需要 CLIP 教師',11)
 save(fig,'02_deep_supervision')
 metadata=dict(source_index=index,category=ds.base.categories[label],seed=42,epoch_zero_based=epoch,source_bbox=list(bounds),view_geometry=sample['geometry'].tolist(),scripts={str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['dataloader/ObjectScalePairs.py','scripts/cache_clip_bbox.py','scripts/train_object_scale.py','scripts/plot_clip_training_flow.py'] if (ROOT/p).is_file()})
 (OUT/'figure_manifest.json').write_text(json.dumps(metadata,indent=2));print(metadata)
if __name__=='__main__':main()
