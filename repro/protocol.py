"""Identity-disjoint validation and explicit, camera-aware retrieval metrics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

SEED=20260918

def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Embeddings must be finite 2D arrays')
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if (norm<1e-12).any():raise ValueError('Zero embeddings are not valid retrieval vectors')
    return x/norm

def cosine_matrix(a,b):
    result=a@b.T
    if not np.isfinite(result).all() or np.abs(result).max()>1.0001:
        raise ValueError('Invalid cosine values; check encoder/numerical runtime')
    return result

def make_splits(frame,path,seed=SEED):
    path=Path(path)
    ids=np.sort(frame.loc[frame.split=='train','vehicle_id'].unique().astype(int))
    rng=np.random.default_rng(seed);n=len(ids)
    parent={int(v):int(v) for v in ids}
    def root(v):
        while parent[v]!=v:
            parent[v]=parent[parent[v]];v=parent[v]
        return v
    # Different vehicles can be annotated in byte-identical full frames. Keep
    # connected IDs together to prevent shared-frame context across folds.
    if 'file_sha256' in frame:
        for _,g in frame[frame.split=='train'].groupby('file_sha256'):
            members=g.vehicle_id.astype(int).unique()
            for v in members[1:]:parent[root(int(v))]=root(int(members[0]))
    components={}
    for v in ids:components.setdefault(root(int(v)),[]).append(int(v))
    groups=list(components.values());rng.shuffle(groups)
    names=['train','dev','calibration','holdout'];targets=[int(n*.65),int(n*.80)-int(n*.65),int(n*.90)-int(n*.80),n]
    splits={name:[] for name in names};part=0
    for group in groups:
        if part<3 and len(splits[names[part]])>=targets[part]:part+=1
        splits[names[part]].extend(group)
    flat=[v for xs in splits.values() for v in xs]
    assert len(flat)==len(set(flat))==n
    payload={'seed':seed,'unit':'vehicle_id; shared full-frame identity components kept together','identities':splits}
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        previous=json.loads(path.read_text())
        if previous!=payload:raise ValueError('Existing split differs; do not silently replace evaluation IDs')
    else:path.write_text(json.dumps(payload,indent=2))
    return splits

def retrieval_metrics(emb,frame,exclude_all_same_camera=True):
    emb=normalize(emb);ids=frame.vehicle_id.to_numpy();cams=frame.camera_id.to_numpy()
    sim=cosine_matrix(emb,emb);aps=[];inps=[];rank1=[];rank5=[];details=[]
    for i in range(len(frame)):
        valid=cams!=cams[i] if exclude_all_same_camera else ~((ids==ids[i])&(cams==cams[i]))
        valid[i]=False;order=np.flatnonzero(valid);order=order[np.argsort(-sim[i,order],kind='stable')]
        hit=ids[order]==ids[i];npos=int(hit.sum())
        if not npos:continue
        ranks=np.flatnonzero(hit)+1;ap=float((np.arange(1,npos+1)/ranks).mean())
        inp=float(npos/ranks[-1])
        aps.append(ap);inps.append(inp);rank1.append(bool(hit[0]));rank5.append(bool(hit[:5].any()))
        details.append({'image_id':frame.iloc[i].image_id,'vehicle_id':int(ids[i]),'camera_id':int(cams[i]),
                        'ap':ap,'inp':inp,'rank1':int(hit[0]),'first_positive_rank':int(ranks[0]),
                        'last_positive_rank':int(ranks[-1])})
    if not aps:raise ValueError('No evaluable cross-camera queries')
    return {'mAP':float(np.mean(aps)),'mINP':float(np.mean(inps)),'Rank1':float(np.mean(rank1)),'Rank5':float(np.mean(rank5)),
        'queries':len(aps),'excluded_queries':len(frame)-len(aps),'camera_protocol':'exclude_all_same_camera' if exclude_all_same_camera else 'exclude_same_id_same_camera'},details

def open_set_protocol(frame,seed=SEED):
    """Gallery: one camera per known ID. Queries: other cameras + absent IDs."""
    rng=np.random.default_rng(seed);ids=np.sort(frame.vehicle_id.unique());rng.shuffle(ids)
    known=set(ids[:int(.7*len(ids))]);q=[];g=[]
    for vid,part in frame.groupby('vehicle_id',sort=True):
        if vid not in known:q.extend(part.index.tolist());continue
        cameras=np.sort(part.camera_id.unique());gc=rng.choice(cameras)
        g.extend(part.index[part.camera_id==gc].tolist());q.extend(part.index[part.camera_id!=gc].tolist())
    return np.asarray(q,dtype=int),np.asarray(g,dtype=int)

def open_set_scores(emb,frame,seed=SEED,topk=10):
    frame=frame.reset_index(drop=True);emb=normalize(emb);q,g=open_set_protocol(frame,seed)
    sim=cosine_matrix(emb[q],emb[g]);k=min(topk,len(g));order=np.argsort(-sim,axis=1,kind='stable')[:,:k]
    scores=np.take_along_axis(sim,order,axis=1);qid=frame.iloc[q].vehicle_id.to_numpy();gid=frame.iloc[g].vehicle_id.to_numpy()
    truth=qid[:,None]==gid[order];all_matches=qid[:,None]==gid[None,:]
    return {'scores':scores,'truth':truth,'known':all_matches.any(axis=1),'total_positives':int(all_matches.sum()),
        'query_indices':q,'gallery_indices':g,'ranking':order,'gallery_size':len(g),'queries':len(q)}

def refusal_metrics(pack,threshold):
    accepted=pack['scores']>=threshold;tp=int((accepted&pack['truth']).sum());fp=int((accepted&~pack['truth']).sum())
    fn=pack['total_positives']-tp;precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1)
    unknown=~pack['known'];reject=~accepted.any(axis=1)
    return {'threshold_cosine':float(threshold),'candidate_micro_F1':float(2*tp/max(2*tp+fp+fn,1)),
        'precision':float(precision),'recall':float(recall),'TNR':float(reject[unknown].mean()) if unknown.any() else None,
        'known_query_acceptance':float((~reject[pack['known']]).mean()),'unknown_queries':int(unknown.sum()),
        'queries':pack['queries'],'gallery_size':pack['gallery_size'],'tp':tp,'fp':fp,'fn':fn}

def calibrate(pack):
    grid=np.unique(np.r_[np.linspace(-1,1,1001),pack['scores'].max()+1e-6])
    results=[refusal_metrics(pack,t) for t in grid]
    best=max(results,key=lambda m:(m['candidate_micro_F1'],m['TNR'] or 0))
    return best,results

def write_submission(emb,frame,out,threshold,model_info):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    q=frame.index[frame.split=='query'].to_numpy();g=frame.index[frame.split=='gallery'].to_numpy()
    if len(g)<10:raise ValueError('Official format requires at least 10 gallery candidates')
    emb=normalize(emb);test=np.concatenate([emb[q],emb[g]],axis=0).astype(np.float32)
    assert np.isfinite(test).all();np.save(out/'embeddings.npy',test)
    sim=cosine_matrix(emb[q],emb[g]);rank=np.argsort(-sim,axis=1,kind='stable')[:,:10]
    qids=frame.iloc[q].image_id.to_numpy();gids=frame.iloc[g].image_id.to_numpy()
    sub=pd.DataFrame({'query_id':qids,**{f'gallery_id_{j+1}':gids[rank[:,j]] for j in range(10)}})
    sub.to_csv(out/'submission.csv',index=False)
    rows=[]
    for i,qid in enumerate(qids):
        picked=[j for j in rank[i] if sim[i,j]>=threshold]
        rows.extend([{'query_id':qid,'gallery_id':gids[j],'confidence':float(sim[i,j])} for j in picked]
                    or [{'query_id':qid,'gallery_id':'','confidence':''}])
    pd.DataFrame(rows).to_csv(out/'candidates.csv',index=False)
    (out/'manifest.json').write_text(json.dumps({'model':model_info,'confidence_semantics':'raw cosine similarity, not probability',
        'threshold':threshold,'test_order':'test_query.csv rows, then test_gallery.csv rows','shape':list(test.shape),
        'dtype':'float32','status':'experimental baseline; official evaluator and privacy review pending'},indent=2))
