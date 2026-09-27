"""Offline DINOv2 feature extraction using uploaded source and pinned local weights."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from run_policy import check_run_policy
from preprocessing import tensor_from_crop

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend/third_party/dinov2-main'))
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14

def load_encoder(weights,arch='small'):
    factories={'small':dinov2_vits14,'base':dinov2_vitb14,'large':dinov2_vitl14}
    if arch not in factories:raise ValueError(f'Unsupported architecture: {arch}')
    model=factories[arch](pretrained=False)
    state=torch.load(weights,map_location='cpu',weights_only=True)
    if 'model' in state:state=state['model']
    # The timm redistribution does not need the unused masked-pretraining token.
    result=model.load_state_dict(state,strict=False)
    if set(result.missing_keys)-{'mask_token'} or result.unexpected_keys:
        raise ValueError(f'Checkpoint architecture mismatch: {result}')
    return model

class CropDataset(Dataset):
    def __init__(self,frame,root,size=224,geometry='pad',augment=None):
        self.frame=frame.reset_index(drop=True);self.root=Path(root);self.size=size;self.geometry=geometry;self.augment=augment
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        im=Image.open(self.root/(self.frame.iloc[i].image_id+'.png')).convert('RGB')
        if self.augment:im=self.augment(im)
        return tensor_from_crop(im,self.size,self.geometry)

def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize()
    elif device.type=='mps':torch.mps.synchronize()

def main():
    p=argparse.ArgumentParser();p.add_argument('--objects',default='outputs/eda/objects.csv')
    p.add_argument('--crops',default='data/derived/crops');p.add_argument('--weights',default='weights/dinov2_vits14_timm.bin')
    p.add_argument('--arch',choices=['small','base','large'],default='small');p.add_argument('--size',type=int,default=224)
    p.add_argument('--geometry',choices=['pad','stretch'],default='pad');p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--output',default='outputs/features/dinos224_pad');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--checkpoint');p.add_argument('--fp16',action='store_true')
    p.add_argument('--identity-split',choices=['train','dev','calibration'],
                   help='Optional identity-disjoint subset; holdout is deliberately unavailable')
    p.add_argument('--splits',default='repro/configs/identity_splits.json')
    p.add_argument('--research-only',action='store_true');a=p.parse_args()
    policy=check_run_policy(a.research_only)
    torch.manual_seed(20260918);torch.set_num_threads(4);dev=torch.device(a.device)
    frame=pd.read_csv(a.objects)
    if a.identity_split:
        identities=json.loads(Path(a.splits).read_text())['identities'][a.identity_split]
        frame=frame[(frame.split=='train')&frame.vehicle_id.isin(identities)]
        if frame.empty:raise ValueError('Selected identity split is empty')
    frame=frame.head(a.limit) if a.limit else frame
    model=load_encoder(a.weights,a.arch);checkpoint_state=None
    if a.checkpoint:
        checkpoint_state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model.load_state_dict(checkpoint_state['encoder'],strict=True)
    model.eval().to(dev);dset=CropDataset(frame,a.crops,a.size,a.geometry)
    loader=DataLoader(dset,batch_size=a.batch_size,num_workers=a.workers,shuffle=False,pin_memory=dev.type=='cuda')
    allcls=[];allmean=[];started=time.monotonic();n=0
    with torch.inference_mode():
        for x in loader:
            x=x.to(dev)
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=a.fp16):
                f=model.forward_features(x)
            allcls.append(f['x_norm_clstoken'].float().cpu().numpy())
            allmean.append(f['x_norm_patchtokens'].mean(1).float().cpu().numpy());n+=len(x)
            if n%(a.batch_size*20)==0:print(json.dumps({'images':n,'total':len(frame),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
    synchronize(dev);elapsed=time.monotonic()-started;out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    arrays={'cls':np.concatenate(allcls),'mean':np.concatenate(allmean),'image_ids':frame.image_id.to_numpy(dtype=str)}
    if a.checkpoint:
        head=checkpoint_state.get('head',{})
        required={'0.weight','0.bias','0.running_mean','0.running_var'}
        if required.issubset(head):
            weight=head['0.weight'].float().numpy();bias=head['0.bias'].float().numpy()
            mean=head['0.running_mean'].float().numpy();variance=head['0.running_var'].float().numpy()
            arrays['bnneck']=(arrays['cls']-mean)/np.sqrt(variance+1e-5)*weight+bias
            if not np.isfinite(arrays['bnneck']).all():raise FloatingPointError('Non-finite BN-neck features')
    np.savez(out/'features.npz',**arrays)
    latencies=[];sample=dset[0].unsqueeze(0).to(dev)
    with torch.inference_mode():
        for i in range(25):
            synchronize(dev);t=time.perf_counter()
            with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=a.fp16):model(sample)
            synchronize(dev)
            if i>=5:latencies.append((time.perf_counter()-t)*1000)
    metadata={**vars(a),'rows':len(frame),'feature_dimensions':int(allcls[0].shape[1]),'elapsed_s':elapsed,
        'extraction_images_per_second':len(frame)/elapsed,'batch1_model_ms_p50':float(np.median(latencies)),
        'batch1_model_ms_p95':float(np.quantile(latencies,.95)),'benchmark_device':str(dev),'torch_version':str(torch.__version__),
        'weights_sha256':hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),'weights_bytes':Path(a.weights).stat().st_size,
        'checkpoint_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest() if a.checkpoint else None,
        'checkpoint_bytes':Path(a.checkpoint).stat().st_size if a.checkpoint else None,
        'available_embeddings':[key for key in ['cls','mean','bnneck'] if key in arrays],
        'source_sha256':{name:hashlib.sha256((ROOT/'repro'/name).read_bytes()).hexdigest() for name in ['extract.py','preprocessing.py']},
        'run_policy':policy,
        'privacy_preprocessing':{'crop_root':str(Path(a.crops)),
            'method':'organizer-provided anonymization, bbox crop, max-side448' if Path(a.crops).name=='crops' else 'derived cache; inspect its preprocessing manifest',
            'coverage_status':'residual content unverified; query risk slices kept separate'},
        'note':'Timing on benchmark_device above, not an end-to-end service benchmark. Extraction FPS includes cached PNG decode/resize/normalization; batch1 latency excludes them.'}
    (out/'run.json').write_text(json.dumps(metadata,indent=2));print(json.dumps({'complete':True,**metadata}),flush=True)

if __name__=='__main__':main()
