#!/usr/bin/env python3
"""Paired object-only scale/translation pilot, with matched B/D training."""
import argparse,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataloader.ObjectScalePairs import ObjectScalePairs,object_crop,feature_grid
from models.ObjectScaleViT import ObjectScaleViT,corresponding_regions,paired_loss
from dataloader.OriginalImagePairs import OriginalImagePairs
from scripts.train_caltech_mergingvit import MODEL_ARGS


def dump(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def run():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--content-merging',action='store_true');p.add_argument('--hybrid-merging',action='store_true');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--per-class',type=int,default=16);p.add_argument('--valid-per-class',type=int,default=4);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--attention-mode',choices=['global','window'],default='global')
    p.add_argument('--window-size',type=int,default=7)
    p.add_argument('--original-images',action='store_true',help='Complete natural-background source images; no object augmentation')
    p.add_argument('--no-translation',action='store_true',help='Center objects in both training and validation views')
    p.add_argument('--clip-stages',type=int,nargs='+',default=[0,1,2,3],help='Zero-based stages receiving direct CLIP loss')
    p.add_argument('--clip-cache',type=Path);p.add_argument('--clip-weight',type=float,default=.2)
    p.add_argument('--pred-weight',type=float,default=.2);p.add_argument('--region-weight',type=float,default=.2)
    args=p.parse_args()
    if not args.clip_stages or len(set(args.clip_stages))!=len(args.clip_stages) or any(s not in range(4) for s in args.clip_stages):p.error('clip-stages must be distinct values in 0..3')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    reference=json.loads((ROOT/'plots/kmeans/caltech_scale_1015/S0/run_manifest.json').read_text())
    dataset=ObjectScalePairs(str(ROOT/'data'),[],seed=args.seed)
    splits={};rejected=[];source_hashes={}
    for name,key,limit in [('train','train_indices',args.per_class),('valid','valid_indices',args.valid_per_class)]:
        selected=[]
        for label in range(101):
            candidates=[i for i in reference[key] if dataset.base.y[i]==label]
            random.Random(args.seed+label).shuffle(candidates)
            for index in candidates:
                try:
                    _,_,_,annotation=object_crop(dataset.base,index)
                except (ValueError,FileNotFoundError) as exc:
                    rejected.append(dict(index=index,reason=str(exc)));continue
                selected.append(index)
                source_hashes[str(index)]=hashlib.sha256(annotation.read_bytes()).hexdigest()
                if sum(dataset.base.y[i]==label for i in selected)>=limit:break
        splits[name]=selected
    assert not set(splits['train'])&set(splits['valid'])
    dataset_class=OriginalImagePairs if args.original_images else ObjectScalePairs
    train=dataset_class(str(ROOT/'data'),splits['train'],seed=args.seed,no_translation=args.no_translation)
    valid=dataset_class(str(ROOT/'data'),splits['valid'],training=False,seed=args.seed,no_translation=args.no_translation)
    # Initialize shared tensors identically in B/D, despite different merger constructors.
    torch.manual_seed(args.seed);baseline=ObjectScaleViT(**MODEL_ARGS)
    torch.manual_seed(args.seed);model=ObjectScaleViT(**MODEL_ARGS,content_merging=args.content_merging,hybrid_merging=args.hybrid_merging,
        attention_mode=args.attention_mode,window_size=args.window_size)
    shared={k:v for k,v in baseline.state_dict().items() if not k.startswith('merges.')}
    if args.content_merging or args.hybrid_merging:
        model.load_state_dict(shared,strict=False)
        if args.hybrid_merging:
            for i,merge in enumerate(model.merges):
                if hasattr(merge,'base'):merge.base.load_state_dict(baseline.merges[i].state_dict())
    else:model.load_state_dict(baseline.state_dict())
    del baseline
    model.to(args.device)
    alignment_heads=None;teacher_features=None
    parameters=list(model.parameters())
    if args.clip_cache:
        cached=torch.load(args.clip_cache,map_location='cpu',weights_only=False)
        assert set(splits['train'])<=set(cached['splits']['train'])
        assert set(splits['valid'])<=set(cached['splits']['valid'])
        teacher_features=cached['features'].to(args.device)
        assert torch.allclose(teacher_features[splits['train']].norm(dim=-1),torch.ones(len(splits['train']),device=args.device),atol=1e-4)
        alignment_heads=torch.nn.ModuleList([torch.nn.Linear(d,512) for d in MODEL_ARGS['embed_dims']]).to(args.device)
        for stage,head in enumerate(alignment_heads):head.requires_grad_(stage in args.clip_stages)
        parameters+=[p for p in alignment_heads.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.05)
    config=vars(args).copy();config['output']=str(args.output);config['clip_cache']=str(args.clip_cache) if args.clip_cache else None;
    if args.clip_cache:config['clip_cache_sha256']=hashlib.sha256(args.clip_cache.read_bytes()).hexdigest()
    config.update(model_args=MODEL_ARGS,splits=splits,rejected=rejected,annotation_sha256=source_hashes,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['models/ObjectScaleViT.py','models/MergingViT.py','models/WindowAttention.py','dataloader/ObjectScalePairs.py','scripts/train_object_scale.py','dataloader/OriginalImagePairs.py']},
        background='complete source with natural background; deterministic 224 letterbox' if args.original_images else 'class-independent uniform color, same background across views',status='pilot; validation exploratory; contour-derived masks, no manual quality approval')
    dump(args.output/'config.json',config)
    for name,expected in config['source_sha256'].items():
        source=ROOT/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
        snapshot=args.output/'source_snapshot'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
    torch.manual_seed(args.seed+100)
    dump(args.output/'status.json',dict(status='running',pid=os.getpid(),epoch=0))
    from PIL import Image
    preview=train[0];imgs=[]
    mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
    for key in ['a','b']:imgs.append(((preview[key]*std+mean).clamp(0,1)*255).byte().permute(1,2,0).numpy())
    Image.fromarray(np.concatenate(imgs,axis=1)).save(args.output/'pair_preview.png')
    history=[];best=-1
    for epoch in range(args.epochs):
        start=time.time();train.epoch=epoch;model.train();total=correct=count=0;clip_total=0.
        loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=2,
                          generator=torch.Generator().manual_seed(args.seed+epoch))
        for batch in loader:
            batch={k:v.to(args.device) for k,v in batch.items()}
            la,fa=model(batch['a'],True)
            lb,fb=(la,fa) if args.original_images else model(batch['b'],True)
            # Stage index 1 has physical stride 16, and no terminal padding yet.
            ra=corresponding_regions(fa[1],feature_grid(batch['points_a'],fa[1],16),batch['region'])
            rb=corresponding_regions(fb[1],feature_grid(batch['points_b'],fb[1],16),batch['region'])
            loss=paired_loss(la,lb,batch['label'],ra,rb,args.pred_weight,args.region_weight)
            if alignment_heads is not None:
                target=teacher_features[batch['index']]
                terms=[]
                for stage,projection in enumerate(alignment_heads):
                    if stage not in args.clip_stages:continue
                    for features,points in [(fa[stage],batch['points_a']),(fb[stage],batch['points_b'])]:
                        region=corresponding_regions(features,feature_grid(points,features,8*2**stage),batch['region'])
                        terms.append((1-torch.nn.functional.cosine_similarity(projection(region),target,dim=-1)).mean())
                clip_loss=torch.stack(terms).mean()
                clip_total+=clip_loss.item()*len(la)
                loss=loss+args.clip_weight*clip_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad();loss.backward()
            if alignment_heads is not None and epoch==0 and count==0:
                assert all(alignment_heads[s].weight.grad is not None and torch.isfinite(alignment_heads[s].weight.grad).all() for s in args.clip_stages)
                assert all(alignment_heads[s].weight.grad is None for s in range(4) if s not in args.clip_stages)
                assert model.patch_embed.proj.weight.grad is not None and torch.isfinite(model.patch_embed.proj.weight.grad).all()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            n=len(la);count+=n;total+=loss.item()*n;correct+=(la.argmax(1)==batch['label']).sum().item()
        model.eval();vc=vb=agree=vn=0
        with torch.no_grad():
            for batch in DataLoader(valid,batch_size=args.batch_size,num_workers=2):
                la=model(batch['a'].to(args.device));lb=la if args.original_images else model(batch['b'].to(args.device));label=batch['label'].to(args.device)
                vc+=(la.argmax(1)==label).sum().item();vb+=(lb.argmax(1)==label).sum().item()
                agree+=(la.argmax(1)==lb.argmax(1)).sum().item();vn+=len(label)
        row=dict(epoch=epoch+1,loss=total/count,train_accuracy=correct/count,valid_reference_accuracy=vc/vn,
                 valid_unseen_joint_accuracy=vb/vn,prediction_agreement=agree/vn,seconds=time.time()-start)
        if alignment_heads is not None:row['train_clip_cosine']=1-clip_total/count
        history.append(row);dump(args.output/'history.json',history);print(json.dumps(row),flush=True)
        checkpoint=dict(alignment_heads=alignment_heads.state_dict() if alignment_heads is not None else None,model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,config=config)
        torch.save(checkpoint,args.output/'last.pth')
        if vb/vn>best:best=vb/vn;torch.save(checkpoint,args.output/'best.pth')
        dump(args.output/'status.json',dict(status='running',pid=os.getpid(),epoch=epoch+1))
    dump(args.output/'status.json',dict(status='done',pid=os.getpid(),epoch=args.epochs))

if __name__=='__main__':
    try:
        run()
    except BaseException as exc:
        if '--output' in sys.argv:
            output=Path(sys.argv[sys.argv.index('--output')+1])
            status=output/'status.json'
            if status.exists():
                previous=json.loads(status.read_text())
                if previous.get('pid')==os.getpid():
                    previous.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=type(exc).__name__+': '+str(exc))
                    dump(status,previous)
        raise
