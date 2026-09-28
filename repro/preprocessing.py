"""Shared organizer-crop preprocessing for training, extraction and the API core."""
import math
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

VERSION='organizer-bbox-rgb-max448-lanczos-pad-bicubic-imagenet-v1'


def crop_bbox(image, bbox, max_side=448):
    """Strict xywh in original image coordinates; never inspect plate content."""
    if len(bbox)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v!=int(v) for v in bbox):
        raise ValueError('invalid_bbox: expected four finite integer xywh values')
    x,y,w,h=map(int,bbox)
    if x<0 or y<0 or w<=0 or h<=0 or x+w>image.width or y+h>image.height:
        raise ValueError('invalid_bbox: nonempty box must be inside the image')
    if not isinstance(max_side,int) or max_side<1:raise ValueError('invalid max_side')
    crop=image.convert('RGB').crop((x,y,x+w,y+h))
    crop.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    return crop


def tensor_from_crop(image, size=224, geometry='pad'):
    if not isinstance(size,int) or size<1:raise ValueError('invalid input size')
    image=image.convert('RGB')
    if geometry=='pad':image=ImageOps.pad(image,(size,size),method=Image.Resampling.BICUBIC,color=(123,116,103))
    elif geometry=='stretch':image=image.resize((size,size),Image.Resampling.BICUBIC)
    else:raise ValueError('unsupported geometry')
    tensor=TF.to_tensor(image)
    return TF.normalize(tensor,[.485,.456,.406],[.229,.224,.225])
