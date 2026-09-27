"""Research-only CityFlow Track 2 adaptation from the intact official ZIP.

The resulting encoder is an initialization for organizer-ID fine tuning, not a
submission model. No ZIP extraction, plate-specific processing or test labels.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from extract import load_encoder, synchronize
from preprocessing import tensor_from_crop
from run_policy import check_run_policy
from train import AppearanceAugmentation, batch_hard_loss
from train_cuda import file_sha, lr_factor, restore_rng, rng_state, save_checkpoint, save_json, seed_worker

PREFIX = 'AIC21_Track2_ReID/'


def read_manifest(zip_path: Path):
    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        label_path = PREFIX + 'train_label.xml'
        track_path = PREFIX + 'train_track.txt'
        if label_path not in names or track_path not in names:
            raise ValueError('CityFlow train metadata missing from ZIP')
        labels = {}
        # The XML declares GB2312; macOS/Python builds may not expose it to
        # expat, so decode with Python's codec before parsing the Unicode text.
        for item in ET.fromstring(archive.read(label_path).decode('gb2312')).iter('Item'):
            name = item.attrib['imageName']
            if name in labels:
                raise ValueError(f'Duplicate image label: {name}')
            labels[name] = (int(item.attrib['vehicleID']), item.attrib['cameraID'])
        tracks = {}
        for track_index, line in enumerate(archive.read(track_path).decode().splitlines()):
            for name in line.split():
                if name in tracks:
                    raise ValueError(f'Duplicate track membership: {name}')
                tracks[name] = track_index
        if labels.keys() != tracks.keys():
            raise ValueError('Train labels and track membership differ')
        required = {PREFIX + 'image_train/' + name for name in labels}
        if not required.issubset(names):
            raise ValueError(f'{len(required - names)} labeled train images absent from ZIP')
        records = [(name, *labels[name], tracks[name]) for name in sorted(labels)]
    if len(records) != 52717 or len({r[1] for r in records}) != 440:
        raise ValueError('Unexpected CityFlow train cardinality')
    return records


class ZipTrainDataset(Dataset):
    def __init__(self, path: Path, records, size: int, augment: str):
        self.path = path
        self.records = records
        self.size = size
        self.augment = AppearanceAugmentation(augment)
        self.archive = None
        self.labels = {v: i for i, v in enumerate(sorted({r[1] for r in records}))}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self.archive is None:
            self.archive = ZipFile(self.path)
        name, vehicle_id, _, _ = self.records[index]
        with Image.open(io.BytesIO(self.archive.read(PREFIX + 'image_train/' + name))) as image:
            image = image.convert('RGB')
        image = self.augment(image)
        return tensor_from_crop(image, self.size), self.labels[vehicle_id]


class TrackCameraPKSampler(Sampler):
    """P identities, K samples, preferring distinct cameras and tracks."""
    def __init__(self, records, p: int, k: int, steps: int, seed: int):
        self.p, self.k, self.steps, self.seed, self.epoch = p, k, steps, seed, 0
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for index, (_, vehicle_id, camera_id, track_id) in enumerate(records):
            groups[vehicle_id][camera_id][track_id].append(index)
        self.groups = groups
        self.identities = np.asarray(sorted(groups))
        if p > len(self.identities):
            raise ValueError('P exceeds the number of CityFlow identities')

    def __len__(self):
        return self.steps * self.p * self.k

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.steps):
            for vehicle_id in rng.choice(self.identities, self.p, replace=False):
                cameras = list(self.groups[int(vehicle_id)])
                rng.shuffle(cameras)
                used_tracks = set()
                for j in range(self.k):
                    camera = cameras[j % len(cameras)]
                    choices = list(self.groups[int(vehicle_id)][camera])
                    novel = [track for track in choices if track not in used_tracks]
                    track = int(rng.choice(novel if novel else choices))
                    used_tracks.add(track)
                    yield int(rng.choice(self.groups[int(vehicle_id)][camera][track]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--steps-per-epoch', type=int, default=1200)
    parser.add_argument('--size', type=int, default=336)
    parser.add_argument('--p', type=int, default=8)
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--last-blocks', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=.04)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--max-steps', type=int, default=0, help='Smoke-only cap; never call a full epoch')
    parser.add_argument('--resume', action='store_true', help='Resume only at an epoch boundary with identical config')
    parser.add_argument('--research-only', action='store_true')
    args = parser.parse_args()
    policy = check_run_policy(args.research_only)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    if args.size % 14 or min(args.epochs, args.steps_per_epoch, args.p, args.k, args.last_blocks) < 1:
        raise ValueError('Invalid training dimensions')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    zip_path, weights_path, out = Path(args.zip), Path(args.weights), Path(args.output)
    if out.exists() and not args.resume:
        raise FileExistsError(f'Refusing to overwrite existing run: {out}')
    if not out.exists() and args.resume:
        raise FileNotFoundError('No previous run to resume')
    records = read_manifest(zip_path)
    signature = {**{k: v for k, v in vars(args).items() if k != 'resume'},
                 'zip_sha256': file_sha(zip_path), 'weights_sha256': file_sha(weights_path),
                 'train_images': len(records), 'train_identities': len({r[1] for r in records}),
                 'train_tracks': len({r[3] for r in records}), 'holdout_opened': False,
                 'sampling': 'P identities x K images, camera and track aware; fixed steps',
                 'checkpoint_selection': 'fixed last epoch, no target-domain selection'}
    if args.resume:
        if json.loads((out / 'config.json').read_text()) != signature:
            raise ValueError('Resume config mismatch')
    else:
        out.mkdir(parents=True)
        save_json(out / 'config.json', signature)
        save_json(out / 'run_policy.json', policy)
    model = load_encoder(weights_path, 'large')
    if args.last_blocks > len(model.blocks):
        raise ValueError('last-blocks exceeds model depth')
    for parameter in model.parameters():
        parameter.requires_grad = False
    for block in model.blocks[-args.last_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.norm.parameters():
        parameter.requires_grad = True
    model.cuda()
    labels = len({r[1] for r in records})
    head = nn.Sequential(nn.BatchNorm1d(model.embed_dim), nn.Linear(model.embed_dim, labels, bias=False)).cuda()
    optimizer = torch.optim.AdamW([
        {'params': [v for v in model.parameters() if v.requires_grad], 'lr': args.lr},
        {'params': head.parameters(), 'lr': args.head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: lr_factor(epoch, args.epochs, args.warmup))
    dataset = ZipTrainDataset(zip_path, records, args.size, 'mild')
    sampler = TrackCameraPKSampler(records, args.p, args.k, args.steps_per_epoch, args.seed)
    history = []
    start_epoch = 0
    prior_elapsed = 0.
    if args.resume:
        checkpoint = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['encoder'], strict=True)
        head.load_state_dict(checkpoint['head'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'])
        history = checkpoint['history']
        start_epoch = checkpoint['epoch']
        prior_elapsed = checkpoint['elapsed_s']
        if start_epoch >= args.epochs:
            raise ValueError('All requested epochs already completed')
    started = time.monotonic()
    print(json.dumps({'event': 'start', **signature, 'amp': 'bf16'}), flush=True)
    for epoch in range(start_epoch, args.epochs):
        sampler.epoch = epoch
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.p * args.k, sampler=sampler,
                            num_workers=args.workers, pin_memory=True,
                            worker_init_fn=seed_worker, generator=generator,
                            persistent_workers=False)
        model.train()
        head.train()
        losses = []
        epoch_started = time.monotonic()
        for step, (images, labels_batch) in enumerate(loader, 1):
            images = images.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = model(images)
                ce = F.cross_entropy(head(features), labels_batch, label_smoothing=.1)
            metric = batch_hard_loss(features, labels_batch)
            loss = ce + metric
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite CityFlow loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 5., error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(loss.detach()), float(ce.detach()), float(metric.detach())])
            if step % 100 == 0 or (args.max_steps and step >= args.max_steps):
                progress = {'event': 'step', 'epoch': epoch + 1, 'step': step,
                            'steps_per_epoch': min(args.steps_per_epoch, args.max_steps) if args.max_steps else args.steps_per_epoch,
                            'loss': losses[-1][0], 'grad_norm': float(grad_norm)}
                print(json.dumps(progress), flush=True)
                save_json(out / 'status.json', {'state': 'training', 'completed_epochs': epoch,
                                                'progress': progress, 'holdout_opened': False})
            if args.max_steps and step >= args.max_steps:
                break
        synchronize(torch.device('cuda'))
        scheduler.step()
        row = {'epoch': epoch + 1, 'steps': len(losses), 'loss': np.mean(losses, axis=0).tolist(),
               'epoch_s': time.monotonic() - epoch_started, 'elapsed_s': prior_elapsed + time.monotonic() - started,
               'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024**3}
        history.append(row)
        if args.max_steps:
            save_json(out / 'history.json', history)
            save_json(out / 'status.json', {'state': 'smoke_completed', 'completed_epochs': 0,
                                            'smoke_steps': len(losses), 'last': row,
                                            'holdout_opened': False})
            print(json.dumps({'event': 'smoke_completed', **row}), flush=True)
            return
        state = {'encoder': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'head': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'rng': rng_state(torch.device('cuda')), 'history': history, 'elapsed_s': row['elapsed_s'],
                 'epoch': epoch + 1, 'config': signature, 'run_policy': policy}
        save_checkpoint(out / 'last.pt', state)
        save_json(out / 'history.json', history)
        save_json(out / 'status.json', {'state': 'completed' if epoch + 1 == args.epochs else 'running',
                                        'completed_epochs': epoch + 1, 'last': row, 'holdout_opened': False})
        print(json.dumps({'event': 'epoch', **row}), flush=True)
    print(json.dumps({'event': 'completed', 'checkpoint': str(out / 'last.pt')}), flush=True)


if __name__ == '__main__':
    main()
