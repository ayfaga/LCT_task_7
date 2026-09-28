"""Paired target-domain retrieval audit; intentionally excludes holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_finalists import paired_bootstrap
from protocol import make_splits, retrieval_metrics


def aligned_features(path: Path, expected: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        ids = payload['image_ids'].astype(str)
        features = payload['cls'].astype(np.float32)
    if len(ids) != len(set(ids)) or features.shape != (len(ids), 1024):
        raise ValueError(f'Invalid feature rows/dimensions: {path}')
    index = {image_id: position for position, image_id in enumerate(ids)}
    missing = set(expected) - set(ids)
    if missing:
        raise ValueError(f'{len(missing)} required images absent from {path}')
    return features[[index[image_id] for image_id in expected]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--objects', default='outputs/eda/objects.csv')
    parser.add_argument('--splits', default='repro/configs/identity_splits.json')
    parser.add_argument('--baseline-features', required=True)
    parser.add_argument('--candidate-features', required=True)
    parser.add_argument('--fold', choices=['dev', 'calibration'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--draws', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20260918)
    args = parser.parse_args()
    frame = pd.read_csv(args.objects)
    splits = make_splits(frame, args.splits)
    sub = frame[(frame.split == 'train') & frame.vehicle_id.isin(splits[args.fold])].reset_index(drop=True)
    expected = sub.image_id.to_numpy(dtype=str)
    baseline = aligned_features(Path(args.baseline_features), expected)
    candidate = aligned_features(Path(args.candidate_features), expected)
    base_metrics, base_details = retrieval_metrics(baseline, sub, exclude_all_same_camera=True)
    cand_metrics, cand_details = retrieval_metrics(candidate, sub, exclude_all_same_camera=True)
    bootstrap, paired = paired_bootstrap(
        pd.DataFrame(cand_details), pd.DataFrame(base_details), args.draws, args.seed)
    if bootstrap['queries'] != len(sub) or bootstrap['identities'] != sub.vehicle_id.nunique():
        raise ValueError('Paired query or vehicle-ID count mismatch')
    observed = bootstrap['delta_a_minus_b']['mAP']['observed_pp']
    direct = 100 * (cand_metrics['mAP'] - base_metrics['mAP'])
    if abs(observed - direct) > 1e-5:
        raise ValueError('Bootstrap and direct mAP deltas disagree')
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite comparison: {out}')
    out.mkdir(parents=True)
    result = {'fold': args.fold, 'protocol': 'cross-camera, all same-camera gallery excluded',
              'baseline': base_metrics, 'candidate': cand_metrics,
              'bootstrap_candidate_minus_baseline': bootstrap,
              'evaluation_images': len(sub), 'evaluation_identities': int(sub.vehicle_id.nunique()),
              'holdout_opened': False, 'baseline_features': args.baseline_features,
              'candidate_features': args.candidate_features,
              'caveat': 'Internal proxy; dev repeatedly used historically; no hidden-test or independent holdout claim'}
    (out / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    paired.to_csv(out / 'paired_queries.csv', index=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
