"""Joint CityFlow + organizer training for the selected DINOv2-L/14@336.

Four vehicle IDs from each source appear in every P8×K4 batch. Validation
uses organizer dev IDs only. The already opened organizer holdout is never
loaded. This is a research experiment, not a release checkpoint.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms as T

from extract import CropDataset, load_encoder, synchronize
from protocol import make_splits, retrieval_metrics
from run_policy import check_run_policy
from train import TrainDataset
from train_cityflow_zip import ZipTrainDataset, read_manifest
from train_cuda import file_sha, lr_factor, save_checkpoint, save_json


class JointDataset(Dataset):
    def __init__(self, zip_path, city_records, organizer_frame, crops, size):
        self.city = ZipTrainDataset(zip_path, city_records, size, 'mild')
        self.organizer = TrainDataset(organizer_frame, crops, size, mode='mild')
        self.city_count = len(self.city)
        self.city_classes = len(self.city.labels)
        self.city_erasing = T.RandomErasing(p=.25, scale=(.01, .10), ratio=(.5, 2.), value=0)

    def __len__(self):
        return self.city_count + len(self.organizer)

    def __getitem__(self, index):
        if index < self.city_count:
            image, label = self.city[index]
            return self.city_erasing(image), label, 0
        image, label = self.organizer[index - self.city_count]
        return image, self.city_classes + label, 1


class JointCameraTrackPKSampler(Sampler):
    """Fixed 4 CityFlow and 4 organizer IDs per batch, K diverse images/ID."""
    def __init__(self, city_records, organizer_frame, p_each, k, steps, seed):
        self.p_each, self.k, self.steps, self.seed = p_each, k, steps, seed
        self.epoch = 0
        self.city_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, (_, vehicle_id, camera_id, track_id) in enumerate(city_records):
            self.city_groups[int(vehicle_id)][camera_id][int(track_id)].append(index)
        self.org_groups = defaultdict(lambda: defaultdict(list))
        offset = len(city_records)
        for index, row in organizer_frame.reset_index(drop=True).iterrows():
            self.org_groups[int(row.vehicle_id)][int(row.camera_id)].append(offset + index)
        self.city_ids = np.asarray(sorted(self.city_groups))
        self.org_ids = np.asarray(sorted(self.org_groups))
        if len(self.city_ids) < p_each or len(self.org_ids) < p_each:
            raise ValueError('Too few identities for balanced PK batches')

    def __len__(self):
        return self.steps * 2 * self.p_each * self.k

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps):
            chosen = ([(0, int(v)) for v in rng.choice(self.city_ids, self.p_each, replace=False)]
                      + [(1, int(v)) for v in rng.choice(self.org_ids, self.p_each, replace=False)])
            rng.shuffle(chosen)
            for source, vehicle_id in chosen:
                groups = self.city_groups[vehicle_id] if source == 0 else self.org_groups[vehicle_id]
                cameras = list(groups)
                rng.shuffle(cameras)
                used_tracks = set()
                for j in range(self.k):
                    camera = cameras[j % len(cameras)]
                    if source == 0:
                        tracks = list(groups[camera])
                        novel = [track for track in tracks if track not in used_tracks]
                        track = int(rng.choice(novel if novel else tracks))
                        used_tracks.add(track)
                        yield int(rng.choice(groups[camera][track]))
                    else:
                        yield int(rng.choice(groups[camera]))


def within_source_batch_hard(features, labels, sources):
    """Hard negatives from the same dataset; all positives share one ID."""
    normalized = F.normalize(features.float(), dim=1)
    distance = 1 - normalized @ normalized.T
    positive = labels[:, None] == labels[None, :]
    positive.fill_diagonal_(False)
    negative = (labels[:, None] != labels[None, :]) & (sources[:, None] == sources[None, :])
    valid = positive.any(1) & negative.any(1)
    if not valid.all():
        raise ValueError('Balanced PK batch lacks a valid positive or within-source negative')
    hard_positive = distance.masked_fill(~positive, -1e6).max(1).values
    hard_negative = distance.masked_fill(~negative, 1e6).min(1).values
    return F.softplus((hard_positive - hard_negative) / .1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--objects', default='outputs/eda/objects.csv')
    parser.add_argument('--splits', default='repro/configs/identity_splits.json')
    parser.add_argument('--crops', default='data/derived/crops')
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--steps-per-epoch', type=int, default=335)
    parser.add_argument('--p-each', type=int, default=4)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--size', type=int, default=336)
    parser.add_argument('--last-blocks', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=.04)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke-only cap; no checkpoint')
    parser.add_argument('--research-only', action='store_true')
    args = parser.parse_args()
    policy = check_run_policy(args.research_only)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    if args.size % 14 or min(args.epochs, args.steps_per_epoch, args.p_each, args.k, args.last_blocks) < 1:
        raise ValueError('Invalid training dimensions')
    if args.workers != 0:
        raise ValueError('This run requires workers=0 after prior DataLoader shared-memory failure')
    zip_path, weights_path, out = Path(args.zip), Path(args.weights), Path(args.output)
    if out.exists():
        raise FileExistsError(f'Output exists: {out}')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    city_records = read_manifest(zip_path)
    frame = pd.read_csv(args.objects)
    splits = make_splits(frame, args.splits)
    train_ids, dev_ids = set(splits['train']), set(splits['dev'])
    if train_ids & (dev_ids | set(splits['calibration']) | set(splits['holdout'])):
        raise ValueError('Organizer training identities overlap evaluation identities')
    train_frame = frame[(frame.split == 'train') & frame.vehicle_id.isin(train_ids)].reset_index(drop=True)
    dev_frame = frame[(frame.split == 'train') & frame.vehicle_id.isin(dev_ids)].reset_index(drop=True)
    if len(train_frame) != 6209 or len(dev_frame) != 1449:
        raise ValueError('Unexpected organizer train/dev cardinality')
    for image_id in pd.concat([train_frame, dev_frame]).image_id:
        if not (Path(args.crops) / (image_id + '.png')).is_file():
            raise FileNotFoundError(image_id)
    signature = {**vars(args), 'zip_sha256': file_sha(zip_path),
                 'weights_sha256': file_sha(weights_path),
                 'objects_sha256': file_sha(args.objects),
                 'splits_sha256': file_sha(args.splits),
                 'city_images': len(city_records), 'city_ids': len({r[1] for r in city_records}),
                 'organizer_train_images': len(train_frame), 'organizer_train_ids': int(train_frame.vehicle_id.nunique()),
                 'organizer_dev_images': len(dev_frame), 'holdout_opened': False,
                 'labels': '(dataset, vehicle_id) distinct across datasets',
                 'sampling': 'each batch P4 CityFlow IDs + P4 organizer IDs, K4 images each; camera and CityFlow track aware',
                 'loss': 'joint ID CE label smoothing .1 + within-source batch-hard metric loss',
                 'checkpoint': 'model-only best dev mAP; no optimizer checkpoint due limited disk'}
    out.mkdir(parents=True)
    save_json(out / 'config.json', signature)
    save_json(out / 'run_policy.json', policy)
    dataset = JointDataset(zip_path, city_records, train_frame, args.crops, args.size)
    sampler = JointCameraTrackPKSampler(city_records, train_frame, args.p_each, args.k,
                                        args.steps_per_epoch, args.seed)
    classes = len(dataset.city.labels) + len(dataset.organizer.labels)
    if classes != 1441:
        raise ValueError('Unexpected combined ID count')
    dev_loader = DataLoader(CropDataset(dev_frame, args.crops, args.size),
                            batch_size=32, num_workers=0, pin_memory=True)
    model = load_encoder(weights_path, 'large')
    if args.last_blocks > len(model.blocks):
        raise ValueError('Too many requested blocks')
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.blocks[-args.last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    model.cuda()
    head = nn.Sequential(nn.BatchNorm1d(model.embed_dim),
                         nn.Linear(model.embed_dim, classes, bias=False)).cuda()
    optimizer = torch.optim.AdamW([
        {'params': [v for v in model.parameters() if v.requires_grad], 'lr': args.lr},
        {'params': head.parameters(), 'lr': args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_factor(epoch, args.epochs, args.warmup))
    best_map, best_epoch = -1., 0
    history = []
    started = time.monotonic()
    print(json.dumps({'event': 'start', **signature, 'amp': 'bf16'}), flush=True)
    save_json(out / 'status.json', {'state': 'running', 'completed_epochs': 0,
                                    'best_dev_mAP': None, 'holdout_opened': False})
    for epoch in range(args.epochs):
        sampler.epoch = epoch
        loader = DataLoader(dataset, batch_size=2 * args.p_each * args.k,
                            sampler=sampler, num_workers=0, pin_memory=True)
        model.train()
        head.train()
        losses = []
        epoch_start = time.monotonic()
        lrs = [group['lr'] for group in optimizer.param_groups]
        for step, (images, labels, sources) in enumerate(loader, 1):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            sources = sources.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                embeddings = model(images)
                ce = F.cross_entropy(head(embeddings), labels, label_smoothing=.1)
            metric = within_source_batch_hard(embeddings, labels, sources)
            loss = ce + metric
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite joint training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(loss.detach()), float(ce.detach()), float(metric.detach())])
            if step % 50 == 0 or (args.max_steps and step >= args.max_steps):
                progress = {'event': 'step', 'epoch': epoch + 1, 'step': step,
                            'steps_per_epoch': min(args.steps_per_epoch, args.max_steps)
                            if args.max_steps else args.steps_per_epoch,
                            'loss': losses[-1][0], 'grad_norm': float(grad_norm)}
                print(json.dumps(progress), flush=True)
                save_json(out / 'status.json', {'state': 'training', 'completed_epochs': epoch,
                                                'best_dev_mAP': best_map if best_map >= 0 else None,
                                                'progress': progress, 'holdout_opened': False})
            if args.max_steps and step >= args.max_steps:
                break
        synchronize(torch.device('cuda'))
        if args.max_steps:
            row = {'epoch': epoch + 1, 'steps': len(losses), 'loss': np.mean(losses, axis=0).tolist(),
                   'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3}
            save_json(out / 'status.json', {'state': 'smoke_completed', 'completed_epochs': 0,
                                            'smoke_steps': len(losses), 'last': row, 'holdout_opened': False})
            print(json.dumps({'event': 'smoke_completed', **row}), flush=True)
            return
        model.eval()
        dev_features = []
        with torch.inference_mode():
            for images in dev_loader:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    embeddings = model(images.cuda(non_blocking=True))
                dev_features.append(embeddings.float().cpu().numpy())
        dev_metrics, details = retrieval_metrics(np.concatenate(dev_features), dev_frame)
        row = {'epoch': epoch + 1, 'steps': len(losses), 'loss': np.mean(losses, axis=0).tolist(),
               **dev_metrics, 'lr': lrs, 'train_s': time.monotonic() - epoch_start,
               'elapsed_s': time.monotonic() - started,
               'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3}
        history.append(row)
        if dev_metrics['mAP'] > best_map:
            if shutil.disk_usage(out).free < 2.5 * 1024**3:
                raise RuntimeError('Less than 2.5 GiB free before model checkpoint; stopping safely')
            best_map, best_epoch = dev_metrics['mAP'], epoch + 1
            state = {'encoder': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                     'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                     'epoch': best_epoch, 'config': signature, 'run_policy': policy}
            save_checkpoint(out / 'best.pt', state)
            pd.DataFrame(details).to_csv(out / 'best_dev_query_metrics.csv', index=False)
            np.savez(out / 'best_dev_features.npz', cls=np.concatenate(dev_features),
                     image_ids=dev_frame.image_id.to_numpy(dtype=str))
        scheduler.step()
        save_json(out / 'history.json', history)
        save_json(out / 'status.json', {'state': 'running', 'completed_epochs': epoch + 1,
                                        'best_dev_mAP': best_map, 'best_epoch': best_epoch,
                                        'last': row, 'holdout_opened': False})
        print(json.dumps({'event': 'epoch', **row, 'best_dev_mAP': best_map,
                          'best_epoch': best_epoch}), flush=True)
    save_json(out / 'status.json', {'state': 'completed', 'completed_epochs': len(history),
                                    'best_dev_mAP': best_map, 'best_epoch': best_epoch,
                                    'holdout_opened': False})
    print(json.dumps({'event': 'completed', 'best_dev_mAP': best_map,
                      'best_epoch': best_epoch, 'output': str(out)}), flush=True)


if __name__ == '__main__':
    main()
