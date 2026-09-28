"""Camera-balanced ReID fine-tuning. Camera IDs select samples, never model inputs."""
from __future__ import annotations
import argparse
import io
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms as T
from extract import CropDataset,load_encoder,synchronize
from protocol import make_splits,retrieval_metrics,SEED
from run_policy import check_run_policy


class CosFaceHead(nn.Module):
    """Normalized class proxies with an additive target margin."""
    def __init__(self,dimension,classes,margin=.1,scale=16.):
        super().__init__();self.bn=nn.BatchNorm1d(dimension);self.weight=nn.Parameter(torch.empty(classes,dimension))
        self.margin=float(margin);self.scale=float(scale);nn.init.normal_(self.weight,std=.01)
    def forward(self,features,labels):
        cosine=F.normalize(self.bn(features).float(),dim=1)@F.normalize(self.weight.float(),dim=1).T
        target=F.one_hot(labels,num_classes=cosine.shape[1]).to(cosine.dtype)
        return self.scale*(cosine-self.margin*target)

class AppearanceAugmentation:
    def __init__(self,mode):
        self.mode=mode
        self.color=T.ColorJitter(.18,.18,.12,.015) if mode=='mild' else T.ColorJitter(.4,.4,.25,.04)
        self.affine=T.RandomAffine(3,translate=(.025,.025),scale=(.95,1.05),fill=(123,116,103))
    def __call__(self,im):
        if self.mode=='none':return im
        im=self.affine(im)
        if random.random()<.8:im=self.color(im)
        if random.random()<.15:im=im.filter(ImageFilter.GaussianBlur(random.uniform(.2,1.0)))
        if random.random()<.2:
            b=io.BytesIO();im.save(b,format='JPEG',quality=random.randint(55,95));b.seek(0);im=Image.open(b).convert('RGB')
        return im

class TrainDataset(CropDataset):
    def __init__(self,frame,*args,mode='mild',return_camera=False,**kwargs):
        super().__init__(frame,*args,augment=AppearanceAugmentation(mode),**kwargs)
        self.labels={v:i for i,v in enumerate(sorted(frame.vehicle_id.unique()))}
        self.erase=T.RandomErasing(p=.25 if mode=='mild' else .5,scale=(.01,.10 if mode=='mild' else .25),ratio=(.5,2.),value=0)
        self.mode=mode
        self.return_camera=return_camera
    def __getitem__(self,i):
        x=super().__getitem__(i)
        if self.mode!='none':x=self.erase(x)
        row=self.frame.iloc[i]
        label=self.labels[row.vehicle_id]
        return (x,label,int(row.camera_id)) if self.return_camera else (x,label)

class CameraPKSampler(Sampler):
    def __init__(self,frame,p=16,k=4,seed=SEED):
        self.frame=frame.reset_index(drop=True);self.p=p;self.k=k;self.seed=seed;self.epoch=0
        self.groups={v:{c:g.index.to_numpy() for c,g in f.groupby('camera_id')} for v,f in self.frame.groupby('vehicle_id')}
        self.ids=np.asarray(sorted(self.groups));self.steps=max(1,len(frame)//(p*k))
    def __len__(self):return self.steps*self.p*self.k
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch);self.epoch+=1
        for _ in range(self.steps):
            for vid in rng.choice(self.ids,self.p,replace=False):
                groups=self.groups[vid];cams=list(groups);rng.shuffle(cams)
                for j in range(self.k):yield int(rng.choice(groups[cams[j%len(cams)]]))

class HardNegativeCameraPKSampler(CameraPKSampler):
    """PK sampler that probabilistically mixes pre-mined hard ID pairs with random IDs."""
    def __init__(self,frame,neighbors,p=16,k=4,seed=SEED,hard_pairs_per_batch=2,
                 hard_probability_start=1.,hard_probability_end=1.,curriculum_epochs=1):
        super().__init__(frame,p,k,seed)
        if hard_pairs_per_batch<1 or 2*hard_pairs_per_batch>p:
            raise ValueError('hard_pairs_per_batch must use at most P identities')
        if not 0<=hard_probability_start<=1 or not 0<=hard_probability_end<=1:
            raise ValueError('Hard-pair probabilities must be in [0, 1]')
        if curriculum_epochs<1:raise ValueError('curriculum_epochs must be positive')
        allowed=set(map(int,self.ids));clean={}
        for anchor,values in neighbors.items():
            anchor=int(anchor)
            if anchor not in allowed:continue
            kept=sorted({int(value) for value in values if int(value) in allowed and int(value)!=anchor})
            if kept:clean[anchor]=np.asarray(kept,dtype=int)
        if not clean:raise ValueError('Hard-negative graph has no training identities')
        self.neighbors=clean;self.hard_pairs_per_batch=hard_pairs_per_batch
        self.hard_probability_start=float(hard_probability_start)
        self.hard_probability_end=float(hard_probability_end)
        self.curriculum_epochs=int(curriculum_epochs)

    def hard_probability(self,epoch=None):
        epoch=self.epoch if epoch is None else int(epoch)
        progress=min(max(epoch,0),self.curriculum_epochs-1)/max(1,self.curriculum_epochs-1)
        return self.hard_probability_start+progress*(self.hard_probability_end-self.hard_probability_start)

    def __iter__(self):
        epoch=self.epoch;rng=np.random.default_rng(self.seed+epoch);self.epoch+=1
        hard_probability=self.hard_probability(epoch)
        anchors=np.asarray(sorted(self.neighbors),dtype=int)
        for _ in range(self.steps):
            selected=[]
            if rng.random()<hard_probability:
                for _ in range(self.hard_pairs_per_batch):
                    valid=[int(v) for v in anchors if int(v) not in selected and any(int(n) not in selected for n in self.neighbors[int(v)])]
                    if not valid:break
                    anchor=int(rng.choice(valid));options=[int(n) for n in self.neighbors[anchor] if int(n) not in selected]
                    neighbor=int(rng.choice(options));selected.extend([anchor,neighbor])
            remaining=np.asarray([int(v) for v in self.ids if int(v) not in selected],dtype=int)
            needed=self.p-len(selected)
            if needed:
                selected.extend(rng.choice(remaining,needed,replace=False).astype(int).tolist())
            rng.shuffle(selected)
            for vid in selected:
                groups=self.groups[vid];cams=list(groups);rng.shuffle(cams)
                for j in range(self.k):yield int(rng.choice(groups[cams[j%len(cams)]]))

def batch_hard_loss(features,labels):
    z=F.normalize(features.float(),dim=1);distance=1-z@z.T
    same=labels[:,None]==labels[None,:];same.fill_diagonal_(False);other=labels[:,None]!=labels[None,:]
    hard_pos=distance.masked_fill(~same,-1e6).max(1).values
    hard_neg=distance.masked_fill(~other,1e6).min(1).values
    valid=same.any(1)&other.any(1)
    return F.softplus((hard_pos[valid]-hard_neg[valid])/.1).mean() if valid.any() else features.sum()*0


def soft_crosscam_positive_loss(features, labels, cameras, tau=.10):
    """Smooth cross-camera positives, keeping batch-hard negatives unchanged.

    Returns (loss, valid-anchor fraction). An anchor without a cross-camera
    positive contributes to CE but not the metric term.
    """
    if tau <= 0:
        raise ValueError('tau must be positive')
    z=F.normalize(features.float(),dim=1)
    distance=1-z@z.T
    positive=(labels[:,None]==labels[None,:])&(cameras[:,None]!=cameras[None,:])
    negative=labels[:,None]!=labels[None,:]
    valid=positive.any(dim=1)&negative.any(dim=1)
    valid_fraction=valid.float().mean()
    if not valid.any():
        return features.sum()*0,valid_fraction
    count=positive[valid].sum(dim=1)
    soft_positive=tau*(torch.logsumexp((distance[valid]/tau).masked_fill(~positive[valid],-torch.inf),dim=1)-count.float().log())
    hard_negative=distance[valid].masked_fill(~negative[valid],torch.inf).min(dim=1).values
    return F.softplus((soft_positive-hard_negative)/.10).mean(),valid_fraction


def trimmed_identity_hard_loss(features, labels):
    """Use the second nearest *identity*, avoiding one ambiguous nearest ID.

    The nearest negative instance can be a near-duplicate or noisy annotation.
    Grouping by ID stops one ID's many frames from occupying every hard rank.
    This differs from static hard-pair sampling: every PK batch is still sampled
    normally and the uncertain top identity receives no metric gradient.
    """
    z = F.normalize(features.float(), dim=1)
    similarity = z @ z.T
    same = labels[:, None] == labels[None, :]
    same.fill_diagonal_(False)
    positive = similarity.masked_fill(~same, 1e6).min(dim=1).values
    distinct_ids = torch.unique(labels)
    if len(distinct_ids) < 3:
        raise ValueError('trimmed identity hard loss needs at least three IDs per batch')
    by_id = torch.stack([similarity[:, labels == identity].max(dim=1).values
                         for identity in distinct_ids], dim=1)
    own = labels[:, None] == distinct_ids[None, :]
    by_id = by_id.masked_fill(own, -1e6)
    second_negative = by_id.topk(2, dim=1).values[:, 1]
    valid = same.any(dim=1)
    return F.softplus((second_negative[valid] - positive[valid]) / .1).mean()

def main():
    p=argparse.ArgumentParser();p.add_argument('--objects',default='outputs/eda/objects.csv');p.add_argument('--crops',default='data/derived/crops')
    p.add_argument('--weights',default='weights/dinov2_vits14_timm.bin');p.add_argument('--arch',default='small',choices=['small','base'])
    p.add_argument('--splits',default='configs/identity_splits.json');p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=10);p.add_argument('--last-blocks',type=int,default=4)
    p.add_argument('--augment',choices=['none','mild','strong'],default='mild');p.add_argument('--size',type=int,default=224)
    p.add_argument('--p',type=int,default=16);p.add_argument('--k',type=int,default=4);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--seed',type=int,default=SEED)
    p.add_argument('--research-only',action='store_true');a=p.parse_args()
    policy=check_run_policy(a.research_only)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed);torch.set_num_threads(4)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dev=torch.device(a.device);frame=pd.read_csv(a.objects);splits=make_splits(frame,a.splits)
    train=frame[(frame.split=='train')&frame.vehicle_id.isin(splits['train'])].reset_index(drop=True)
    val=frame[(frame.split=='train')&frame.vehicle_id.isin(splits['dev'])].reset_index(drop=True)
    model=load_encoder(a.weights,a.arch)
    for p0 in model.parameters():p0.requires_grad=False
    for block in model.blocks[-a.last_blocks:]:
        for p0 in block.parameters():p0.requires_grad=True
    for p0 in model.norm.parameters():p0.requires_grad=True
    model.to(dev);head=nn.Sequential(nn.BatchNorm1d(model.embed_dim),nn.Linear(model.embed_dim,len(splits['train']),bias=False)).to(dev)
    sampler=CameraPKSampler(train,a.p,a.k,a.seed)
    loader=DataLoader(TrainDataset(train,a.crops,a.size,mode=a.augment),batch_size=a.p*a.k,sampler=sampler,num_workers=a.workers,
        pin_memory=dev.type=='cuda',persistent_workers=a.workers>0)
    vl=DataLoader(CropDataset(val,a.crops,a.size),batch_size=a.p*a.k,num_workers=a.workers,pin_memory=dev.type=='cuda')
    opt=torch.optim.AdamW([{'params':[p0 for p0 in model.parameters() if p0.requires_grad],'lr':a.lr},
                          {'params':head.parameters(),'lr':1e-3}],weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda');out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2));best=-1;history=[];start=time.monotonic()
    (out/'run_policy.json').write_text(json.dumps(policy,indent=2))
    for epoch in range(a.epochs):
        model.train();head.train();losses=[]
        for x,y in loader:
            x=x.to(dev);y=y.to(dev);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):
                f=model(x);idloss=F.cross_entropy(head(f),y,label_smoothing=.1);metric=batch_hard_loss(f,y);loss=idloss+metric
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5.)
            scaler.step(opt);scaler.update();losses.append([float(loss.detach()),float(idloss.detach()),float(metric.detach())])
        sched.step();model.eval();vf=[]
        with torch.inference_mode():
            for x in vl:
                with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):f=model(x.to(dev))
                vf.append(f.float().cpu().numpy())
        metrics,_=retrieval_metrics(np.concatenate(vf),val)
        row={'epoch':epoch+1,'loss':np.mean(losses,axis=0).tolist(),**metrics,'elapsed_s':time.monotonic()-start};history.append(row)
        if metrics['mAP']>best:
            best=metrics['mAP'];torch.save({'encoder':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'epoch':epoch+1,'config':vars(a)},out/'best.pt')
        (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
