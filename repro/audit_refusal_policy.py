"""Bounded, CPU-only selection/audit of a deployable refusal policy.

Model and threshold selection use calibration identities only. The already used
dev set is an exploratory external check; holdout is never loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from protocol import calibrate, normalize, open_set_scores
from refusal_policy import FEATURES, accepted, pair_features, policy_scores, query_features


SEED = 20260923
MODEL_VERSION = "dinov2-l14-l336-fresh16-best13-20260923"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logistic():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.2, class_weight=None, max_iter=1000, random_state=SEED))


def pack_with_gallery(qemb, qids, gemb, gids):
    sim = normalize(qemb) @ normalize(gemb).T
    order = np.argsort(-sim, axis=1, kind="stable")[:, :10]
    scores = np.take_along_axis(sim, order, axis=1)
    matches = qids[:, None] == gids[None, :]
    return {"scores": scores, "truth": np.take_along_axis(matches, order, axis=1),
            "known": matches.any(axis=1), "total_positives": int(matches.sum()),
            "positives_per_query": matches.sum(axis=1).astype(int),
            "queries": len(qids), "gallery_size": len(gids),
            "query_vehicle_ids": qids, "gallery_vehicle_ids": gids}


def attach_ids(pack, frame):
    qids = frame.iloc[pack["query_indices"]].vehicle_id.to_numpy()
    gids = frame.iloc[pack["gallery_indices"]].vehicle_id.to_numpy()
    matches = qids[:, None] == gids[None, :]
    result = dict(pack)
    result["positives_per_query"] = matches.sum(axis=1).astype(int)
    result["query_vehicle_ids"] = qids
    result["gallery_vehicle_ids"] = gids
    assert int(result["positives_per_query"].sum()) == pack["total_positives"]
    return result


def metrics(pack, decisions):
    truth = pack["truth"]
    tp = int((decisions & truth).sum())
    fp = int((decisions & ~truth).sum())
    fn = int(pack["total_positives"] - tp)
    unknown = ~pack["known"]
    correct_query = (decisions & truth).any(axis=1)
    nonempty = decisions.any(axis=1)
    query_tp = int(correct_query.sum())
    query_fp = int((nonempty & ~correct_query).sum())
    query_fn = int((pack["known"] & ~correct_query).sum())
    return {
        "candidate_micro_F1": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "TNR": float((~nonempty)[unknown].mean()),
        "query_correct_id_F1": float(2 * query_tp / max(2 * query_tp + query_fp + query_fn, 1)),
        "query_correct_id_recall": float(query_tp / max(int(pack["known"].sum()), 1)),
        "queries": int(pack["queries"]), "unknown_queries": int(unknown.sum()),
        "gallery_size": int(pack["gallery_size"]),
        "tp": tp, "fp": fp, "fn": fn,
        "query_tp": query_tp, "query_fp": query_fp, "query_fn": query_fn,
    }


def pair_xy(pack, kind):
    matrix = pair_features(pack["scores"], kind).reshape(-1, len(FEATURES[kind]))
    return matrix, pack["truth"].reshape(-1).astype(int)


def oof_pair(pack, kind):
    x, y = pair_xy(pack, kind)
    groups = np.repeat(pack["query_vehicle_ids"], 10)
    result = np.full(len(y), np.nan)
    for train, valid in GroupKFold(n_splits=5).split(x, y, groups):
        fitted = logistic().fit(x[train], y[train])
        result[valid] = fitted.predict_proba(x[valid])[:, 1]
    assert np.isfinite(result).all()
    return result.reshape(pack["scores"].shape)


def oof_query(pack):
    x = query_features(pack["scores"])
    y = pack["known"].astype(int)
    result = np.full(len(y), np.nan)
    for train, valid in GroupKFold(n_splits=5).split(x, y, pack["query_vehicle_ids"]):
        fitted = logistic().fit(x[train], y[train])
        result[valid] = fitted.predict_proba(x[valid])[:, 1]
    assert np.isfinite(result).all()
    return result


def choose_threshold(pack, scores, tnr_floor):
    best = None
    for threshold in np.linspace(0.001, 0.999, 999):
        current = metrics(pack, scores >= threshold)
        if current["TNR"] + 1e-12 < tnr_floor:
            continue
        if best is None or (current["candidate_micro_F1"], current["TNR"]) > (
                best["candidate_micro_F1"], best["TNR"]):
            best = {"threshold": float(threshold), **current}
    assert best is not None
    return best


def fitted_model_payload(model):
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": classifier.coef_[0].tolist(), "intercept": float(classifier.intercept_[0])}


def fixed_policy(kind, threshold, pair_model, query_model=None):
    return {"schema_version": 1, "model_version": MODEL_VERSION,
            "topk": 10, "kind": kind, "threshold": float(threshold),
            "score_semantics": "learned acceptance score; not a proven calibrated probability",
            "ranking_semantics": "cosine order unchanged; accepted subset is a prefix",
            "pair_features": list(FEATURES[kind]),
            "pair_model": fitted_model_payload(pair_model),
            "query_features": ["top1_cosine", "top1_minus_top2", "top10_mean", "top10_std"] if query_model else None,
            "query_model": fitted_model_payload(query_model) if query_model else None}


def bootstrap(pack, baseline, candidate, repeats=2000):
    ids = pack["query_vehicle_ids"]
    unique, inverse = np.unique(ids, return_inverse=True)
    rows_by_id = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(SEED)
    deltas = np.empty((repeats, 3))
    for i in range(repeats):
        rows = np.concatenate([rows_by_id[j] for j in rng.integers(len(unique), size=len(unique))])
        sampled = {key: value[rows] for key, value in pack.items() if key in (
            "scores", "truth", "known", "positives_per_query", "query_vehicle_ids")}
        sampled.update(total_positives=int(sampled["positives_per_query"].sum()),
                       queries=len(rows), gallery_size=pack["gallery_size"])
        a = metrics(sampled, baseline[rows])
        b = metrics(sampled, candidate[rows])
        deltas[i] = [b[key] - a[key] for key in (
            "candidate_micro_F1", "TNR", "query_correct_id_F1")]
    return {"unit": "query vehicle_id; fixed gallery and fitted policies", "repetitions": repeats,
            "F1_delta_95ci": np.quantile(deltas[:, 0], [.025, .975]).tolist(),
            "TNR_delta_95ci": np.quantile(deltas[:, 1], [.025, .975]).tolist(),
            "query_F1_delta_95ci": np.quantile(deltas[:, 2], [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser()
    for name in ("objects", "splits", "features", "output", "policy-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.objects)
    identities = json.loads(args.splits.read_text())["identities"]
    selected = (frame.split == "train") & frame.vehicle_id.isin(identities["dev"] + identities["calibration"])
    evaluation = frame.loc[selected].reset_index(drop=True)
    dev_mask = evaluation.vehicle_id.isin(identities["dev"]).to_numpy()
    assert not evaluation.vehicle_id.isin(identities["holdout"]).any()
    with np.load(args.features, allow_pickle=False) as archive:
        if not np.array_equal(archive["image_ids"].astype(str), evaluation.image_id.to_numpy(dtype=str)):
            raise ValueError("E2 feature/image order mismatch")
        embeddings = archive["cls"]
    cal_frame, dev_frame = evaluation.loc[~dev_mask].reset_index(drop=True), evaluation.loc[dev_mask].reset_index(drop=True)
    cal_emb, dev_emb = embeddings[~dev_mask], embeddings[dev_mask]
    cal = attach_ids(open_set_scores(cal_emb, cal_frame), cal_frame)
    dev = attach_ids(open_set_scores(dev_emb, dev_frame), dev_frame)
    baseline_cal, _ = calibrate(cal)
    baseline_threshold = baseline_cal["threshold_cosine"]
    baseline_dev_decision = dev["scores"] >= baseline_threshold
    floor = baseline_cal["TNR"] - .01

    oof = {name: oof_pair(cal, name) for name in ("score_only", "context4", "context5")}
    oof["context4_times_query"] = oof["context4"] * oof_query(cal)[:, None]
    cal_results = {name: choose_threshold(cal, score, floor) for name, score in oof.items()}
    # Selection rule fixed before dev evaluation: F1 with a calibration TNR
    # floor; within 0.002 F1 prefer fewer features/smaller runtime.
    eligible = ("context4", "context5", "context4_times_query")
    peak = max(cal_results[name]["candidate_micro_F1"] for name in eligible)
    selected_name = next(name for name in eligible if cal_results[name]["candidate_micro_F1"] >= peak - .002)
    if peak < baseline_cal["candidate_micro_F1"] + .005:
        selected_name = "cosine"

    pair_models = {}
    for name in ("score_only", "context4", "context5"):
        x, y = pair_xy(cal, name)
        pair_models[name] = logistic().fit(x, y)
    query_model = logistic().fit(query_features(cal["scores"]), cal["known"].astype(int))

    policies = {name: fixed_policy("context4" if name == "context4_times_query" else name,
                                    cal_results[name]["threshold"],
                                    pair_models["context4" if name == "context4_times_query" else name],
                                    query_model if name == "context4_times_query" else None)
                for name in oof}
    policies["cosine"] = {"schema_version": 1, "model_version": MODEL_VERSION,
                          "topk": 10, "kind": "cosine", "threshold": baseline_threshold,
                          "score_semantics": "raw cosine similarity, not probability"}
    for name, policy in policies.items():
        if name == "cosine":
            continue
        pure = policy_scores(cal["scores"], policy)
        expected = pair_models["context4" if name == "context4_times_query" else name].predict_proba(
            pair_features(cal["scores"], policy["kind"]).reshape(-1, len(FEATURES[policy["kind"]])))[:, 1].reshape(cal["scores"].shape)
        if name == "context4_times_query":
            expected *= query_model.predict_proba(query_features(cal["scores"]))[:, 1][:, None]
        if np.max(np.abs(pure - expected)) > 1e-12:
            raise ValueError(f"Serialization parity failed: {name}")

    dev_results = {name: metrics(dev, accepted(dev["scores"], policy)) for name, policy in policies.items()}
    chosen_policy = policies[selected_name]
    chosen_decision = accepted(dev["scores"], chosen_policy)
    assert dev_results["cosine"]["candidate_micro_F1"] == metrics(dev, baseline_dev_decision)["candidate_micro_F1"]

    q = dev["query_indices"]
    g = dev["gallery_indices"]
    qids = dev_frame.iloc[q].vehicle_id.to_numpy()
    gids = dev_frame.iloc[g].vehicle_id.to_numpy()
    rng = np.random.default_rng(SEED)
    sampled_cal = rng.permutation(len(cal_frame))
    stress = {}
    # Smaller gallery keeps one existing gallery image per known identity.
    one_per_id = np.array([np.flatnonzero(gids == vid)[0] for vid in np.unique(gids)], dtype=int)
    scenarios = [("one_per_known_id", dev_emb[g[one_per_id]], gids[one_per_id])]
    for count in (len(cal_frame) // 4, len(cal_frame) // 2, len(cal_frame)):
        extra = sampled_cal[:count]
        scenarios.append((f"plus_{count}_calibration_distractors",
                          np.concatenate([dev_emb[g], cal_emb[extra]]),
                          np.concatenate([gids, cal_frame.iloc[extra].vehicle_id.to_numpy()])))
    for name, gallery_embeddings, gallery_ids in scenarios:
        scenario = pack_with_gallery(dev_emb[q], qids, gallery_embeddings, gallery_ids)
        stress[name] = {"cosine": metrics(scenario, accepted(scenario["scores"], policies["cosine"])),
                        "selected": metrics(scenario, accepted(scenario["scores"], chosen_policy))}

    result = {
        "state": "completed", "seed": SEED, "holdout_evaluated": False,
        "input_sha256": {str(path): sha256(path) for path in (args.objects, args.splits, args.features, Path(__file__), Path(__file__).with_name("refusal_policy.py"))},
        "protocol": "Official PDF specifies F1 and TNR but not exact F1 grain; both candidate-micro and correct-ID query F1 reported.",
        "selection_rule": "On 5-fold query-ID OOF calibration, maximize candidate-micro F1 subject to TNR >= baseline calibration TNR - 0.01; within 0.002 F1 prefer simpler policy; require +0.005 F1 over baseline or retain cosine. No dev tuning.",
        "baseline_calibration": baseline_cal,
        "calibration_tnr_floor": floor,
        "calibration_oof": cal_results,
        "selected_on_calibration": selected_name,
        "dev": dev_results,
        "selected_vs_cosine_bootstrap": bootstrap(dev, baseline_dev_decision, chosen_decision),
        "stress": stress,
        "policy_artifact": str(args.policy_output),
        "limits": "Dev has prior backbone-selection reuse; stress adds calibration IDs as distractors and is not an independent generalization test; fixed-gallery bootstrap ignores model/threshold-selection uncertainty. No hidden-test or official evaluator claim."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.write_text(json.dumps(chosen_policy, ensure_ascii=False, indent=2))
    result["policy_sha256"] = sha256(args.policy_output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"selected": selected_name,
                      "calibration_oof": {k: {"F1": v["candidate_micro_F1"], "TNR": v["TNR"], "threshold": v["threshold"]} for k, v in cal_results.items()},
                      "dev": {k: {"F1": v["candidate_micro_F1"], "TNR": v["TNR"], "query_F1": v["query_correct_id_F1"]} for k, v in dev_results.items()},
                      "stress": {k: {policy: {"F1": m["candidate_micro_F1"], "TNR": m["TNR"]} for policy, m in values.items()} for k, values in stress.items()},
                      "policy_sha256": result["policy_sha256"]}))


if __name__ == "__main__":
    main()
