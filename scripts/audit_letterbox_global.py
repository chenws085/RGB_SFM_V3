import sys,json,hashlib
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_caltech_mergingvit as t
from models.ObjectScaleViT import ObjectScaleViT
p=t.ROOT/'runs/train/mergingvit_caltech101_letterbox160';cp=torch.load(p/'MergingViT_best.pth',map_location='cpu',weights_only=False)
cfg=json.loads((p/'run_config.json').read_text());sys.argv=['audit'];args=t.parse_args();torch.set_num_threads(4);t.set_seed(42);tr,va=t.make_loaders(args)
assert len(tr.dataset)==cfg['train_samples'] and len(va.dataset)==cfg['valid_samples'];assert not set(tr.dataset.indices)&set(va.dataset.indices)
model=t.MergingViT(**cp['model_args']).cuda(1);model.load_state_dict(cp['model_weights'])
metrics=t.evaluate(model,va,t.nn.CrossEntropyLoss(label_smoothing=.1),torch.device('cuda:1'));assert metrics['accuracy']==cp['valid_acc'];assert metrics['airplanes_accuracy']==cp['airplanes_valid_acc']
# Window wrapper reuses all original parameters with exactly the same initialization.
t.set_seed(42);g=t.MergingViT(**t.MODEL_ARGS);t.set_seed(42);w=ObjectScaleViT(**t.MODEL_ARGS,attention_mode='window');assert all(torch.equal(v,w.state_dict()[k]) for k,v in g.state_dict().items())
from test_caltech_translation_kmeans import safe_letterbox
import numpy as np
im,_=va.dataset.base_dataset[va.dataset.indices[0]];a=t.transforms.ToTensor()(t.CenterLetterbox()(im));b=safe_letterbox(im,np.array([[0.],[0.]]))[0];a=t.transforms.Normalize(t.IMAGENET_MEAN,t.IMAGENET_STD)(a);assert torch.equal(a,b)
out=dict(checkpoint=str(p/'MergingViT_best.pth'),checkpoint_sha256=hashlib.sha256((p/'MergingViT_best.pth').read_bytes()).hexdigest(),validation=metrics,train_indices=tr.dataset.indices,valid_indices=va.dataset.indices,initialization_parity=True,training_eval_preprocessing_parity=True,source_sha256={q:hashlib.sha256((t.ROOT/q).read_bytes()).hexdigest() for q in ['scripts/train_caltech_mergingvit.py','models/MergingViT.py','models/WindowAttention.py','dataloader/Caltech101.py']})
(t.ROOT/'overnight_logs/letterbox_window_1109/global_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(metrics),flush=True)
