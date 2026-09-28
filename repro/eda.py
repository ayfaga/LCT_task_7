"""Reproducible archive/pixel audit; source data is never modified."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import time
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

def quantiles(values):
    x=np.asarray(values,dtype=float)
    return {str(q):float(np.quantile(x,q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)} if len(x) else {}

def read_tables(archive):
    with ZipFile(archive) as z:
        frames=[]
        for split,name in [('train','train.csv'),('query','test_query.csv'),('gallery','test_gallery.csv')]:
            f=pd.read_csv(io.BytesIO(z.read(name)),dtype={'image_id':str})
            f['split']=split;f['source_row']=np.arange(len(f));frames.append(f)
        return pd.concat(frames,ignore_index=True)

def contact_sheet(records,archive,dest,captions,ncols=5):
    width,height=224,192
    sheet=Image.new('RGB',(width*ncols,height*((len(records)+ncols-1)//ncols)),'white')
    draw=ImageDraw.Draw(sheet)
    with ZipFile(archive) as z:
        for i,r in enumerate(records):
            im=Image.open(io.BytesIO(z.read(r['member']))).convert('RGB')
            crop=im.crop((r['x'],r['y'],r['x']+r['w'],r['y']+r['h']))
            crop.thumbnail((218,155));x,y=(i%ncols)*width,(i//ncols)*height
            sheet.paste(crop,(x+(width-crop.width)//2,y))
            draw.text((x+3,y+158),str(captions[i]),fill='black')
    sheet.save(dest)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',default='data/Датасет/dataset.zip')
    ap.add_argument('--output',default='outputs/eda');ap.add_argument('--crop-cache',default='data/derived/crops')
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cache=Path(args.crop_cache);cache.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1);frame=read_tables(args.archive);records=[];errors=[];start=time.monotonic()
    with ZipFile(args.archive) as z:
        members={Path(n).stem:n for n in z.namelist() if n.lower().endswith(('.jpg','.jpeg','.png'))}
        (out/'dataset_README.md').write_bytes(z.read('README.md'))
        for name in ('train.csv','test_query.csv','test_gallery.csv'):(out/name).write_bytes(z.read(name))
        for i,r in frame.iterrows():
            rec=r.to_dict()
            try:
                member=members[r.image_id];payload=z.read(member)
                im=Image.open(io.BytesIO(payload));im.load();im=im.convert('RGB');iw,ih=im.size
                x,y,w,h=map(int,(r.x,r.y,r.w,r.h));x0,y0,x1,y1=max(0,x),max(0,y),min(iw,x+w),min(ih,y+h)
                if x1<=x0 or y1<=y0:raise ValueError('empty clipped crop')
                crop=im.crop((x0,y0,x1,y1));gray=np.asarray(crop.resize((128,128)).convert('L'))
                rgb=np.asarray(crop.resize((64,64)),dtype=np.float32);dh=np.asarray(crop.resize((9,8)).convert('L'))
                dhash=int.from_bytes(np.packbits((dh[:,1:]>dh[:,:-1]).reshape(-1)).tobytes(),'big')
                rec.update(member=member,image_width=iw,image_height=ih,crop_width=x1-x0,crop_height=y1-y0,
                    bbox_clipped=bool(x0!=x or y0!=y or x1!=x+w or y1!=y+h),bbox_area_fraction=w*h/(iw*ih),
                    aspect_ratio=w/h,min_side=min(w,h),touches_boundary=bool(x<=0 or y<=0 or x+w>=iw or y+h>=ih),
                    brightness=float(gray.mean()),contrast=float(gray.std()),sharpness_128=float(cv2.Laplacian(gray,cv2.CV_64F).var()),
                    mean_r=float(rgb[:,:,0].mean()),mean_g=float(rgb[:,:,1].mean()),mean_b=float(rgb[:,:,2].mean()),
                    dhash=f'{dhash:016x}',file_sha256=hashlib.sha256(payload).hexdigest(),
                    crop_pixel_sha256=hashlib.sha256(np.asarray(crop).tobytes()).hexdigest())
                crop.thumbnail((448,448),Image.Resampling.LANCZOS)
                crop.save(cache/f'{r.image_id}.png');records.append(rec)
            except Exception as exc:errors.append({'image_id':r.image_id,'error':str(exc)})
            if (i+1)%500==0:print(json.dumps({'processed':i+1,'total':len(frame),'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
    df=pd.DataFrame(records);df.to_csv(out/'objects.csv',index=False);train=df[df.split=='train']
    sizes=train.groupby('vehicle_id').size();cams=train.groupby('vehicle_id').camera_id.nunique()
    cc=train.groupby('camera_id').size().sort_values(ascending=False);gs=train.groupby(['vehicle_id','camera_id']).size()
    summary={'archive':str(Path(args.archive).resolve()),'archive_bytes':Path(args.archive).stat().st_size,
        'image_members':len(members),'rows':len(frame),'successfully_decoded':len(df),'errors':errors,
        'official_evaluator_present':False,'train_identities':int(train.vehicle_id.nunique()),'train_cameras':int(train.camera_id.nunique()),
        'images_per_identity_hist':{str(k):int(v) for k,v in sizes.value_counts().sort_index().items()},
        'cameras_per_identity_hist':{str(k):int(v) for k,v in cams.value_counts().sort_index().items()},
        'images_per_identity_camera_hist':{str(k):int(v) for k,v in gs.value_counts().sort_index().items()},
        'camera_counts':{str(int(k)):int(v) for k,v in cc.items()},'top2_camera_share':float(cc.iloc[:2].sum()/len(train)),
        'bbox_clipped_count':int(df.bbox_clipped.sum()),'image_dimensions':{str(k):int(v) for k,v in df.groupby(['image_width','image_height']).size().items()},
        'exact_file_duplicate_rows':int(df.file_sha256.duplicated(keep=False).sum()),
        'exact_crop_duplicate_rows':int(df.crop_pixel_sha256.duplicated(keep=False).sum()),'split_profiles':{},'split_image_overlap':{},
        'crop_cache':str(cache.resolve()),'crop_cache_max_side':448,'crop_cache_format':'PNG, aspect-preserving downscale; original anonymization preserved'}
    for split,f in df.groupby('split'):
        summary['split_profiles'][split]={'rows':len(f),'brightness':quantiles(f.brightness),'min_side':quantiles(f.min_side),
            'aspect_ratio':quantiles(f.aspect_ratio),'bbox_area_fraction':quantiles(f.bbox_area_fraction),'sharpness_128':quantiles(f.sharpness_128),
            'boundary_count':int(f.touches_boundary.sum()),'small_under_64':int((f.min_side<64).sum())}
    for a,b in [('train','query'),('train','gallery'),('query','gallery')]:
        summary['split_image_overlap'][a+'_'+b]=len(set(df[df.split==a].image_id)&set(df[df.split==b].image_id))
    df[df.crop_pixel_sha256.duplicated(keep=False)].sort_values('crop_pixel_sha256').to_csv(out/'exact_crop_duplicates.csv',index=False)
    df[df.file_sha256.duplicated(keep=False)].sort_values('file_sha256').to_csv(out/'exact_frame_duplicates.csv',index=False)
    lut=np.array([bin(i).count('1') for i in range(256)],dtype=np.uint8)
    hashes=np.array([int(s,16) for s in df.dhash],dtype=np.uint64);screen=[]
    for a,b in [('train','query'),('train','gallery'),('query','gallery')]:
        ia=df.index[df.split==a].to_numpy();ib=df.index[df.split==b].to_numpy()
        for st in range(0,len(ia),128):
            aa=ia[st:st+128];xor=np.bitwise_xor(hashes[aa,None],hashes[None,ib])
            dist=lut[xor.view(np.uint8).reshape(len(aa),len(ib),8)].sum(axis=-1);rr,ccs=np.where(dist<=4)
            for ii,jj in zip(rr,ccs):screen.append({'split_a':a,'split_b':b,'image_id_a':df.at[aa[ii],'image_id'],
                'image_id_b':df.at[ib[jj],'image_id'],'hamming':int(dist[ii,jj])})
    pd.DataFrame(screen,columns=['split_a','split_b','image_id_a','image_id_b','hamming']).to_csv(out/'cross_split_near_duplicate_candidates.csv',index=False)
    summary['cross_split_dhash_le4_candidates']=dict(Counter(x['split_a']+'_'+x['split_b'] for x in screen))
    groups={'same_camera':[],'different_camera':[]}
    for _,g in train.groupby('vehicle_id'):
        rows=list(g.index)
        for p,i in enumerate(rows):
            for j in rows[p+1:]:
                key='same_camera' if df.at[i,'camera_id']==df.at[j,'camera_id'] else 'different_camera'
                groups[key].append(bin(int(hashes[i])^int(hashes[j])).count('1'))
    summary['within_identity_dhash']={k:{'pairs':len(v),'distance':quantiles(v),'hamming_le4':sum(x<=4 for x in v)} for k,v in groups.items()}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    sample=df.sample(min(25,len(df)),random_state=42)
    contact_sheet(sample.to_dict('records'),args.archive,out/'random_crops.jpg',[f'{r.split} {r.image_id[:8]}\n{r.w}x{r.h}' for r in sample.itertuples()])
    ids=train.vehicle_id.drop_duplicates().sample(5,random_state=17);ex=pd.concat([train[train.vehicle_id==v].head(5) for v in ids])
    contact_sheet(ex.to_dict('records'),args.archive,out/'identities_cameras.jpg',[f'ID {int(r.vehicle_id)} cam {int(r.camera_id)}\n{r.image_id[:10]}' for r in ex.itertuples()])
    for name,part in [('smallest',df.nsmallest(15,'min_side')),('darkest',df.nsmallest(15,'brightness'))]:
        contact_sheet(part.to_dict('records'),args.archive,out/f'{name}_crops.jpg',[f'{r.split} {r.image_id[:8]}\n{r.w}x{r.h} brightness {r.brightness:.0f}' for r in part.itertuples()])
    print(json.dumps({'complete':True,'elapsed_s':round(time.monotonic()-start,1),'decoded':len(df),'errors':len(errors)}),flush=True)

if __name__=='__main__':main()
