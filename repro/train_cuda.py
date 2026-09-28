"""Offline ReID trainer with epoch-boundary resume, AMP and fixed unseen-ID dev."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

# Use stock PyTorch SDPA: no xformers download or binary dependency required.
os.environ.setdefault('XFORMERS_DISABLED', '1')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from extract import CropDataset, load_encoder, synchronize
from protocol import SEED, make_splits, retrieval_metrics
from run_policy import check_run_policy
from train import TrainDataset, CameraPKSampler, HardNegativeCameraPKSampler, CosFaceHead, batch_hard_loss, trimmed_identity_hard_loss, soft_crosscam_positive_loss


def file_sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def save_json(path, value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2));tmp.replace(path)


def save_checkpoint(path, state):
    path=Path(path);tmp=path.with_suffix('.tmp')
    torch.save(state,tmp);tmp.replace(path)


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def rng_state(device=None):
    ns=np.random.get_state()
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'python':random.getstate(),'numpy':[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],
            'mps':torch.mps.get_rng_state() if device is not None and torch.device(device).type=='mps' else None}


def restore_rng(state):
    torch.set_rng_state(state['torch']);random.setstate(state['python'])
    ns=state['numpy'];np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),*ns[2:]))
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])
    if state.get('mps') is not None:
        if not torch.backends.mps.is_available():raise RuntimeError('Checkpoint requires MPS RNG support')
        torch.mps.set_rng_state(state['mps'])


def lr_factor(epoch, epochs, warmup):
    if warmup and epoch<warmup:return (epoch+1)/warmup
    progress=(epoch-warmup)/max(1,epochs-warmup-1)
    return .05+.95*.5*(1+math.cos(math.pi*min(max(progress,0),1)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--objects',default='outputs/eda/objects.csv');p.add_argument('--crops',default='data/derived/crops')
    p.add_argument('--weights',default='weights/dinov2_vits14_timm.bin');p.add_argument('--arch',choices=['small','base','large','clip'],default='small')
    p.add_argument('--splits',default='configs/identity_splits.json');p.add_argument('--output',required=True)
    p.add_argument('--train-folds',choices=['train','train+dev'],default='train',
                   help='Internal labeled folds for encoder training; calibration and holdout are excluded')
    p.add_argument('--fixed-checkpoint-epoch',type=int,default=0,
                   help='Save best.pt at this predetermined epoch without validation (required for train+dev)')
    p.add_argument('--device',default='cuda');p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--last-blocks',type=int,default=12);p.add_argument('--augment',choices=['none','mild','strong'],default='mild')
    p.add_argument('--size',type=int,default=224);p.add_argument('--p',type=int,default=16);p.add_argument('--k',type=int,default=4)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--threads',type=int,default=8)
    p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--head-lr',type=float,default=1e-3)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--weight-decay',type=float,default=.04)
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--amp',choices=['auto','bf16','fp16','off'],default='auto')
    p.add_argument('--init-checkpoint',help='Optional encoder checkpoint for a new run; not optimizer resume')
    p.add_argument('--init-head',action='store_true',help='Also reuse the compatible ID head from --init-checkpoint')
    p.add_argument('--eval-initial',action='store_true',help='Evaluate and preserve epoch0 as a best-checkpoint candidate')
    p.add_argument('--hard-negative-map',help='JSON neighbors mined from train embeddings only')
    p.add_argument('--hard-pairs-per-batch',type=int,default=0)
    p.add_argument('--hard-probability-start',type=float,default=1.)
    p.add_argument('--hard-probability-end',type=float,default=1.)
    p.add_argument('--hard-curriculum-epochs',type=int,default=1)
    p.add_argument('--metric-weight',type=float,default=1.)
    p.add_argument('--metric-loss',choices=['batch_hard','trimmed_identity','soft_crosscam_positive'],default='batch_hard')
    p.add_argument('--anchor-weight',type=float,default=0.,help='Cosine preservation against frozen --init-checkpoint encoder')
    p.add_argument('--id-loss',choices=['ce','cosface'],default='ce')
    p.add_argument('--label-smoothing',type=float,default=.1)
    p.add_argument('--cosface-margin',type=float,default=.1);p.add_argument('--cosface-scale',type=float,default=16.)
    p.add_argument('--gate-epoch',type=int,default=0,help='Stop after this epoch if dev mAP is below --gate-min-map')
    p.add_argument('--gate-min-map',type=float,default=0.)
    p.add_argument('--resume',action='store_true');p.add_argument('--research-only',action='store_true')
    p.add_argument('--max-steps',type=int,default=0,help='Smoke-test cap; stored in config, not a full epoch')
    a=p.parse_args();policy=check_run_policy(a.research_only)
    if a.epochs<1 or a.last_blocks<1 or a.size%14 or a.p<2 or a.k<2:raise ValueError('Invalid epochs/blocks/size/PK')
    no_validation=a.train_folds=='train+dev'
    if no_validation:
        if not 1<=a.fixed_checkpoint_epoch<=a.epochs:raise ValueError('train+dev requires fixed checkpoint epoch in range')
        if a.eval_initial or a.gate_epoch or a.gate_min_map:raise ValueError('No dev evaluation/gate after adding dev to training')
    elif a.fixed_checkpoint_epoch:raise ValueError('Fixed checkpoint epoch is only for train+dev')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
    dev=torch.device(a.device)
    if dev.type=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no silent CPU fallback')
        torch.cuda.manual_seed_all(a.seed);torch.cuda.reset_peak_memory_stats()
    elif dev.type=='mps':
        if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU fallback')
        torch.mps.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    frame=pd.read_csv(a.objects);splits=make_splits(frame,a.splits)
    training_ids=set(splits['train']) | (set(splits['dev']) if no_validation else set())
    if training_ids & (set(splits['calibration']) | set(splits['holdout'])):raise ValueError('Train overlaps calibration or holdout')
    train=frame[(frame.split=='train')&frame.vehicle_id.isin(training_ids)].reset_index(drop=True)
    val=frame[(frame.split=='train')&frame.vehicle_id.isin(splits['dev'])].reset_index(drop=True) if not no_validation else frame.iloc[:0].copy()
    if a.p>train.vehicle_id.nunique():raise ValueError('P exceeds training identities')
    for image_id in pd.concat([train,val]).image_id:
        if not (Path(a.crops)/(image_id+'.png')).is_file():raise FileNotFoundError(image_id)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    signature={k:v for k,v in vars(a).items() if k not in ['resume','output']}
    signature.update(weights_sha256=file_sha(a.weights),objects_sha256=file_sha(a.objects),splits_sha256=file_sha(a.splits))
    if a.init_checkpoint:
        if not Path(a.init_checkpoint).is_file():raise FileNotFoundError(a.init_checkpoint)
        signature['init_checkpoint_sha256']=file_sha(a.init_checkpoint)
    if a.init_head and not a.init_checkpoint:raise ValueError('--init-head requires --init-checkpoint')
    if bool(a.hard_negative_map)!=bool(a.hard_pairs_per_batch):
        raise ValueError('--hard-negative-map and positive --hard-pairs-per-batch must be used together')
    if a.metric_weight<0 or a.anchor_weight<0:raise ValueError('Loss weights must be non-negative')
    if a.metric_loss=='trimmed_identity' and a.p<3:raise ValueError('trimmed_identity requires P>=3')
    if not 0<=a.label_smoothing<1 or a.cosface_margin<0 or a.cosface_scale<=0:raise ValueError('Invalid ID loss config')
    if a.anchor_weight and not a.init_checkpoint:raise ValueError('--anchor-weight requires --init-checkpoint')
    if not 0<=a.hard_probability_start<=1 or not 0<=a.hard_probability_end<=1 or a.hard_curriculum_epochs<1:
        raise ValueError('Invalid hard-negative curriculum')
    if bool(a.gate_epoch)!=bool(a.gate_min_map):raise ValueError('--gate-epoch and --gate-min-map must be used together')
    if a.gate_epoch<0 or a.gate_epoch>a.epochs or not 0<=a.gate_min_map<=1:raise ValueError('Invalid dev gate')
    if a.hard_negative_map:
        if not Path(a.hard_negative_map).is_file():raise FileNotFoundError(a.hard_negative_map)
        signature['hard_negative_map_sha256']=file_sha(a.hard_negative_map)
    code_root=Path(__file__).resolve().parent
    signature['source_sha256']={name:file_sha(code_root/name) for name in ['train_cuda.py','train.py','extract.py','preprocessing.py','protocol.py']}
    if (out/'config.json').exists():
        if not a.resume:raise FileExistsError('Output already used; choose new output or --resume')
        if json.loads((out/'config.json').read_text())!=signature:raise ValueError('Resume config mismatch')
    else:
        if a.resume:raise FileNotFoundError('No previous run to resume')
        save_json(out/'config.json',signature)
    save_json(out/'run_policy.json',policy)
    if a.arch=='clip':
        if a.size!=224:raise ValueError('OpenCLIP checkpoint requires fixed 224px input')
        from clip_model import load_clip_encoder
        model=load_clip_encoder(a.weights)
    else:
        model=load_encoder(a.weights,a.arch)
    if a.init_checkpoint:
        initial=torch.load(a.init_checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(initial['encoder'],strict=True)
    if a.last_blocks>len(model.blocks):raise ValueError('Too many requested blocks')
    for par in model.parameters():par.requires_grad=False
    for block in model.blocks[-a.last_blocks:]:
        for par in block.parameters():par.requires_grad=True
    for par in model.norm.parameters():par.requires_grad=True
    model.to(dev)
    teacher=None
    if a.anchor_weight:
        teacher=load_encoder(a.weights,a.arch)
        teacher.load_state_dict(initial['encoder'],strict=True)
        for par in teacher.parameters():par.requires_grad=False
        teacher.eval().to(dev)
    head=(nn.Sequential(nn.BatchNorm1d(model.embed_dim),nn.Linear(model.embed_dim,len(training_ids),bias=False))
          if a.id_loss=='ce' else CosFaceHead(model.embed_dim,len(training_ids),a.cosface_margin,a.cosface_scale)).to(dev)
    if a.init_head:
        if a.id_loss=='ce':head.load_state_dict(initial['head'],strict=True)
        else:
            bn={key.removeprefix('0.'):value for key,value in initial['head'].items() if key.startswith('0.')}
            head.bn.load_state_dict(bn,strict=True);head.weight.data.copy_(initial['head']['1.weight'])
    optimizer=torch.optim.AdamW([{'params':[par for par in model.parameters() if par.requires_grad],'lr':a.lr},
                                {'params':head.parameters(),'lr':a.head_lr}],weight_decay=a.weight_decay)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:lr_factor(e,a.epochs,a.warmup))
    amp=(('bf16' if torch.cuda.is_bf16_supported() else 'fp16') if dev.type=='cuda' else 'off') if a.amp=='auto' else a.amp
    if dev.type!='cuda' and amp!='off':raise ValueError('AMP modes supported here on CUDA only')
    dtype=torch.bfloat16 if amp=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=amp=='fp16')
    if a.hard_negative_map:
        graph=json.loads(Path(a.hard_negative_map).read_text())
        sampler=HardNegativeCameraPKSampler(train,graph['neighbors'],a.p,a.k,a.seed,a.hard_pairs_per_batch,
                                             a.hard_probability_start,a.hard_probability_end,a.hard_curriculum_epochs)
    else:
        sampler=CameraPKSampler(train,a.p,a.k,a.seed)
    dataset=TrainDataset(train,a.crops,a.size,mode=a.augment,
                         return_camera=a.metric_loss=='soft_crosscam_positive')
    val_loader=(DataLoader(CropDataset(val,a.crops,a.size),batch_size=a.p*a.k,num_workers=a.workers,
                           pin_memory=dev.type=='cuda',worker_init_fn=seed_worker) if not no_validation else None)
    history=[];best=-1.;start_epoch=0;prior_elapsed=0.
    if a.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(ck['encoder']);head.load_state_dict(ck['head']);optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'])
        history=ck['history'];best=ck['best_mAP'];start_epoch=ck['epoch'];prior_elapsed=ck['elapsed_s']
    environment={'torch':str(torch.__version__),'numpy':np.__version__,'device':str(dev),'amp':amp,
                 'gpu':torch.cuda.get_device_name() if dev.type=='cuda' else ('Apple MPS' if dev.type=='mps' else None),
                 'train_images':len(train),'train_identities':len(training_ids),'dev_images':len(val),
                 'validation_mode':'fixed_epoch_no_dev' if no_validation else 'dev_mAP',
                 'fixed_checkpoint_epoch':a.fixed_checkpoint_epoch,
                 'trainable_encoder_parameters':sum(par.numel() for par in model.parameters() if par.requires_grad),
                 'no_internet_required':True,'holdout_opened':False,'resume_scope':'epoch boundary; same config',
                 'strict_deterministic_kernels':False,'seeded_sampling_and_augmentation':True,
                 'init_checkpoint':a.init_checkpoint,
                 'init_checkpoint_sha256':signature.get('init_checkpoint_sha256'),
                 'init_head':a.init_head,'hard_negative_map':a.hard_negative_map,
                 'hard_negative_map_sha256':signature.get('hard_negative_map_sha256'),
                 'hard_pairs_per_batch':a.hard_pairs_per_batch,
                 'hard_probability_start':a.hard_probability_start,'hard_probability_end':a.hard_probability_end,
                 'hard_curriculum_epochs':a.hard_curriculum_epochs,'metric_weight':a.metric_weight,
                 'metric_loss':a.metric_loss,
                 'anchor_weight':a.anchor_weight,'id_loss':a.id_loss,'label_smoothing':a.label_smoothing,
                 'cosface_margin':a.cosface_margin,'cosface_scale':a.cosface_scale,
                 'gate_epoch':a.gate_epoch,'gate_min_map':a.gate_min_map}
    save_json(out/'environment.json',environment);print(json.dumps({'event':'start',**environment}),flush=True)
    started=time.monotonic();sampled_mps_driver_max=0.
    save_json(out/'status.json',{'state':'running','completed_epochs':start_epoch,'best_dev_mAP':best if best>=0 else None,'holdout_opened':False})
    if a.eval_initial and not a.resume:
        initial_rng=rng_state(dev)
        model.eval();initial_features=[]
        with torch.inference_mode():
            for x in val_loader:
                with torch.autocast(device_type=dev.type,dtype=dtype,enabled=amp!='off'):
                    f=model(x.to(dev,non_blocking=True))
                initial_features.append(f.float().cpu().numpy())
        vf=np.concatenate(initial_features);initial_metrics,initial_details=retrieval_metrics(vf,val)
        restore_rng(initial_rng)
        best=initial_metrics['mAP']
        initial_state={'encoder':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                       'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                       'epoch':0,'config':signature,'run_policy':policy}
        save_checkpoint(out/'best.pt',initial_state)
        pd.DataFrame(initial_details).to_csv(out/'best_dev_query_metrics.csv',index=False)
        np.savez(out/'best_dev_features.npz',cls=vf,image_ids=val.image_id.to_numpy(dtype=str))
        save_json(out/'initial_metrics.json',{'epoch':0,**initial_metrics,'holdout_opened':False})
        initial_state.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                             rng=rng_state(dev),history=history,best_mAP=best,elapsed_s=time.monotonic()-started)
        save_checkpoint(out/'last.pt',initial_state)
        print(json.dumps({'event':'initial_evaluation','epoch':0,**initial_metrics}),flush=True)
    for epoch in range(start_epoch,a.epochs):
        sampler.epoch=epoch;generator=torch.Generator().manual_seed(a.seed+epoch)
        # New workers each epoch have seeded RNG: resume does not inherit hidden persistent-worker states.
        loader=DataLoader(dataset,batch_size=a.p*a.k,sampler=sampler,num_workers=a.workers,pin_memory=dev.type=='cuda',
                          worker_init_fn=seed_worker,generator=generator,persistent_workers=False)
        model.train();head.train();losses=[];synchronize(dev);epoch_start=time.monotonic()
        current_lrs=[group['lr'] for group in optimizer.param_groups]
        valid_anchor_fractions=[]
        for step,batch in enumerate(loader,1):
            if a.metric_loss=='soft_crosscam_positive':
                x,y,cameras=batch
                cameras=cameras.to(dev,non_blocking=True)
            else:
                x,y=batch
            x=x.to(dev,non_blocking=True);y=y.to(dev,non_blocking=True);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type,dtype=dtype,enabled=amp!='off'):
                features=model(x);logits=head(features) if a.id_loss=='ce' else head(features,y)
                ce=F.cross_entropy(logits,y,label_smoothing=a.label_smoothing)
                if teacher is not None:
                    with torch.inference_mode():teacher_features=teacher(x)
            # FP32 metric distances outside autocast (important with BF16/FP16 similarities).
            if a.metric_loss=='soft_crosscam_positive':
                metric,valid_anchor_fraction=soft_crosscam_positive_loss(features,y,cameras)
                valid_anchor_fractions.append(float(valid_anchor_fraction.detach()))
            else:
                metric=(trimmed_identity_hard_loss(features,y) if a.metric_loss=='trimmed_identity'
                        else batch_hard_loss(features,y))
            anchor=(1-(F.normalize(features.float(),dim=1)*F.normalize(teacher_features.float(),dim=1)).sum(1)).mean() if teacher is not None else features.sum()*0
            loss=ce+a.metric_weight*metric+a.anchor_weight*anchor
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            gradnorm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),5.,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();losses.append([float(loss.detach()),float(ce.detach()),float(metric.detach()),float(anchor.detach())])
            if dev.type=='mps':sampled_mps_driver_max=max(sampled_mps_driver_max,torch.mps.driver_allocated_memory()/1024**3)
            if step%20==0 or (a.max_steps and step>=a.max_steps):
                progress={'event':'step','epoch':epoch+1,'step':step,'steps_per_epoch':min(sampler.steps,a.max_steps) if a.max_steps else sampler.steps,
                          'loss':losses[-1][0],'grad_norm':float(gradnorm),'epoch_elapsed_s':time.monotonic()-epoch_start,
                          'mps_sampled_max_driver_GiB':sampled_mps_driver_max if dev.type=='mps' else None}
                print(json.dumps(progress),flush=True)
                save_json(out/'status.json',{'state':'training','completed_epochs':epoch,'best_dev_mAP':best if best>=0 else None,'progress':progress,'holdout_opened':False})
            if a.max_steps and step>=a.max_steps:break
        synchronize(dev);train_s=time.monotonic()-epoch_start;model.eval();features=[]
        metrics={}
        if not no_validation:
            save_json(out/'status.json',{'state':'validating','epoch':epoch+1,'completed_epochs':epoch,'best_dev_mAP':best if best>=0 else None,'holdout_opened':False})
            with torch.inference_mode():
                for x in val_loader:
                    with torch.autocast(device_type=dev.type,dtype=dtype,enabled=amp!='off'):f=model(x.to(dev,non_blocking=True))
                    features.append(f.float().cpu().numpy())
            vf=np.concatenate(features);metrics,details=retrieval_metrics(vf,val)
        elapsed=prior_elapsed+time.monotonic()-started
        row={'epoch':epoch+1,'loss':np.mean(losses,axis=0).tolist(),**metrics,'steps':len(losses),
             'lr':current_lrs,'train_s':train_s,'epoch_s':time.monotonic()-epoch_start,'elapsed_s':elapsed,
             'metric_valid_anchor_fraction':float(np.mean(valid_anchor_fractions)) if valid_anchor_fractions else None,
             'hard_batch_probability':sampler.hard_probability(epoch) if isinstance(sampler,HardNegativeCameraPKSampler) else 0.,
             'train_images_per_second':len(losses)*a.p*a.k/train_s,
             'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3 if dev.type=='cuda' else None,
             'mps_sampled_max_driver_GiB':sampled_mps_driver_max if dev.type=='mps' else None}
        history.append(row);scheduler.step()
        state={'encoder':{k:v.detach().cpu() for k,v in model.state_dict().items()},
               'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},'epoch':epoch+1,'config':signature,'run_policy':policy}
        if no_validation and epoch+1==a.fixed_checkpoint_epoch:
            save_checkpoint(out/'best.pt',state)
        elif not no_validation and metrics['mAP']>best:
            best=metrics['mAP'];save_checkpoint(out/'best.pt',state)
            pd.DataFrame(details).to_csv(out/'best_dev_query_metrics.csv',index=False)
            np.savez(out/'best_dev_features.npz',cls=vf,image_ids=val.image_id.to_numpy(dtype=str))
        state.update(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                     rng=rng_state(dev),history=history,best_mAP=best,elapsed_s=elapsed)
        save_checkpoint(out/'last.pt',state);save_json(out/'history.json',history)
        save_json(out/'status.json',{'state':'running','completed_epochs':epoch+1,'best_dev_mAP':best if best>=0 else None,'last':row,'holdout_opened':False})
        print(json.dumps({'event':'epoch',**row,'best_mAP':best if best>=0 else None,
                          'fixed_checkpoint_epoch':a.fixed_checkpoint_epoch if no_validation else None}),flush=True)
        if a.gate_epoch==epoch+1 and metrics['mAP']<a.gate_min_map:
            gate={'state':'gate_stopped','completed_epochs':epoch+1,'best_dev_mAP':best,'last':row,
                  'gate_epoch':a.gate_epoch,'gate_min_map':a.gate_min_map,'holdout_opened':False}
            save_json(out/'status.json',gate);print(json.dumps({'event':'gate_stopped',**gate}),flush=True);return
    save_json(out/'status.json',{'state':'completed','completed_epochs':len(history),'best_dev_mAP':best if best>=0 else None,
                                  'fixed_checkpoint_epoch':a.fixed_checkpoint_epoch if no_validation else None,
                                  'holdout_opened':False})
    print(json.dumps({'event':'completed','best_mAP':best if best>=0 else None,
                      'fixed_checkpoint_epoch':a.fixed_checkpoint_epoch if no_validation else None,
                      'output':str(out)}),flush=True)


if __name__=='__main__':main()
