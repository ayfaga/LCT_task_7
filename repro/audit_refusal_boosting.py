"""CPU audit of a small monotone gradient booster for frozen E2 retrieval.

Only calibration identities fit models and select thresholds. Dev is an
exploratory readout; the sealed holdout and test labels are never loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from audit_refusal_policy import (MODEL_VERSION, SEED, attach_ids, bootstrap,
                                  logistic, metrics, pack_with_gallery, sha256)
from protocol import calibrate, open_set_scores
from refusal_policy import FEATURES, pair_features


EXPECTED_SHA = {
    "objects": "c5eabdd4c17e02265f65b90c0568d51ef7907e099b413ba397f694f6f40c661d",
    "splits": "9de62999082a5dd832fc6df53755b94468bf0c072aa8dffadb23c927aa2f1345",
    "features": "9c187d3ebccf6a1ca0ab6aaee32e982d30da57ffcee21fe17d2b62129b7cb0d1",
}
SPECS = {
    "boost_shallow4": {"kind": "context4", "max_iter": 80, "max_leaf_nodes": 7,
                       "min_samples_leaf": 80, "l2_regularization": 5.0},
    "boost_shallow5": {"kind": "context5", "max_iter": 120, "max_leaf_nodes": 7,
                       "min_samples_leaf": 60, "l2_regularization": 5.0},
}
KEEP_FIRST = 3
MIN_PAIR_GAIN = 0.005


def xy(pack, kind):
    matrix = pair_features(pack["scores"], kind)
    return matrix.reshape(-1, matrix.shape[-1]), pack["truth"].reshape(-1).astype(int)


def make_booster(spec):
    monotonic = [1 if name in ("candidate_cosine", "candidate_minus_top1") else 0
                 for name in FEATURES[spec["kind"]]]
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=spec["max_iter"],
        max_leaf_nodes=spec["max_leaf_nodes"],
        min_samples_leaf=spec["min_samples_leaf"],
        l2_regularization=spec["l2_regularization"],
        monotonic_cst=monotonic,
        early_stopping=False, random_state=SEED,
    )


def predict(model, pack, kind):
    x, _ = xy(pack, kind)
    score = model.predict_proba(x)[:, 1].reshape(pack["scores"].shape)
    if not np.isfinite(score).all() or np.any(score[:, 1:] > score[:, :-1] + 1e-10):
        raise ValueError("Booster must preserve cosine order within every query")
    return score


def oof_predictions(pack, spec):
    x, y = xy(pack, spec["kind"])
    groups = np.repeat(pack["query_vehicle_ids"], 10)
    out = np.full(len(y), np.nan)
    for train, valid in GroupKFold(n_splits=5).split(x, y, groups):
        out[valid] = make_booster(spec).fit(x[train], y[train]).predict_proba(x[valid])[:, 1]
    if not np.isfinite(out).all():
        raise ValueError("Incomplete calibration OOF prediction")
    out = out.reshape(pack["scores"].shape)
    if np.any(out[:, 1:] > out[:, :-1] + 1e-10):
        raise ValueError("OOF booster scores changed cosine rank")
    return out


def hybrid(scores, predicted, threshold, cosine_threshold):
    result = (scores >= cosine_threshold) & (
        (np.arange(scores.shape[1])[None, :] < KEEP_FIRST) | (predicted >= threshold))
    if np.any(result[:, 1:] & ~result[:, :-1]):
        raise ValueError("Accepted candidates must form a cosine prefix")
    return result


def best_threshold(pack, predicted, baseline, cosine_threshold, mode):
    candidates = np.r_[0.0, np.linspace(0.001, 0.999, 999), 1.0]
    feasible = []
    for threshold in candidates:
        decision = (predicted >= threshold) if mode == "pure" else hybrid(
            pack["scores"], predicted, threshold, cosine_threshold)
        current = metrics(pack, decision)
        if mode == "pure":
            ok = current["TNR"] >= baseline["TNR"] - 0.01
        else:
            ok = current["query_correct_id_F1"] >= baseline["query_correct_id_F1"] - 1e-12
            if abs(current["TNR"] - baseline["TNR"]) > 1e-12:
                raise ValueError("Hybrid changed query abstention")
        if ok:
            # For equal validation metrics, favor less pruning.
            feasible.append((current["candidate_micro_F1"],
                             current["query_correct_id_F1"], -float(threshold),
                             float(threshold), current))
    if not feasible:
        raise ValueError("No calibration OOF threshold satisfies guardrails")
    _, _, _, threshold, chosen = max(feasible)
    return {"threshold": threshold, "metrics": chosen}


def main():
    parser = argparse.ArgumentParser()
    for name in ("objects", "splits", "features", "output", "model-output", "policy-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    sources = {name: sha256(getattr(args, name)) for name in EXPECTED_SHA}
    if sources != EXPECTED_SHA:
        raise ValueError(f"Frozen E2 inputs changed: {sources}")
    frame = pd.read_csv(args.objects)
    identities = json.loads(args.splits.read_text())["identities"]
    selected = ((frame.split == "train") &
                frame.vehicle_id.isin(identities["dev"] + identities["calibration"]))
    evaluation = frame.loc[selected].reset_index(drop=True)
    dev_mask = evaluation.vehicle_id.isin(identities["dev"]).to_numpy()
    if evaluation.vehicle_id.isin(identities["holdout"]).any():
        raise ValueError("Holdout identity in boosting input")
    with np.load(args.features, allow_pickle=False) as archive:
        if not np.array_equal(archive["image_ids"].astype(str),
                              evaluation.image_id.to_numpy(dtype=str)):
            raise ValueError("E2 feature/image order mismatch")
        embeddings = archive["cls"]
    cal_frame = evaluation.loc[~dev_mask].reset_index(drop=True)
    dev_frame = evaluation.loc[dev_mask].reset_index(drop=True)
    cal_emb, dev_emb = embeddings[~dev_mask], embeddings[dev_mask]
    cal = attach_ids(open_set_scores(cal_emb, cal_frame), cal_frame)
    dev = attach_ids(open_set_scores(dev_emb, dev_frame), dev_frame)
    if (cal["queries"], cal["gallery_size"], dev["queries"], dev["gallery_size"]) != (652, 277, 1036, 413):
        raise ValueError("Frozen open-set protocol changed")
    cosine_threshold = calibrate(cal)[0]["threshold_cosine"]
    if not np.isclose(cosine_threshold, 0.394, atol=1e-12):
        raise ValueError("Frozen cosine threshold changed")
    base_cal_decision = cal["scores"] >= cosine_threshold
    base_dev_decision = dev["scores"] >= cosine_threshold
    base_cal, base_dev = metrics(cal, base_cal_decision), metrics(dev, base_dev_decision)

    calibration, fitted, dev_predictions = {}, {}, {}
    for name, spec in SPECS.items():
        oof = oof_predictions(cal, spec)
        calibration[name] = {
            "pure": best_threshold(cal, oof, base_cal, cosine_threshold, "pure"),
            "hybrid": best_threshold(cal, oof, base_cal, cosine_threshold, "hybrid"),
        }
        x, y = xy(cal, spec["kind"])
        fitted[name] = make_booster(spec).fit(x, y)
        dev_predictions[name] = predict(fitted[name], dev, spec["kind"])

    # The single proposed booster is selected entirely from calibration OOF.
    selected_name = max(SPECS, key=lambda name: (
        calibration[name]["hybrid"]["metrics"]["candidate_micro_F1"],
        calibration[name]["hybrid"]["metrics"]["query_correct_id_F1"],
        -list(SPECS).index(name)))
    selected_threshold = calibration[selected_name]["hybrid"]["threshold"]
    selected_cal = calibration[selected_name]["hybrid"]["metrics"]
    promote_candidate = (selected_cal["candidate_micro_F1"] >=
                         base_cal["candidate_micro_F1"] + MIN_PAIR_GAIN)
    selected_dev_decision = hybrid(dev["scores"], dev_predictions[selected_name],
                                   selected_threshold, cosine_threshold) if promote_candidate else base_dev_decision

    # Reproduce the prior logistic hybrid as an exact reference, using the
    # previously fixed keep=3 and p>=.228. Its dev numbers are a sanity check.
    logistic_x, logistic_y = xy(cal, "context4")
    old_model = logistic().fit(logistic_x, logistic_y)
    old_scores = old_model.predict_proba(xy(dev, "context4")[0])[:, 1].reshape(dev["scores"].shape)
    old_decision = hybrid(dev["scores"], old_scores, 0.228, cosine_threshold)
    old_metrics = metrics(dev, old_decision)
    if abs(old_metrics["candidate_micro_F1"] - 0.5471464019851117) > 1e-7:
        raise ValueError("Prior logistic hybrid was not reproduced")

    dev_results = {"cosine": base_dev, "logistic_hybrid": old_metrics}
    for name in SPECS:
        for mode in ("pure", "hybrid"):
            threshold = calibration[name][mode]["threshold"]
            decision = (dev_predictions[name] >= threshold) if mode == "pure" else hybrid(
                dev["scores"], dev_predictions[name], threshold, cosine_threshold)
            dev_results[f"{name}_{mode}"] = metrics(dev, decision)
    dev_results["selected_hybrid_or_fallback"] = metrics(dev, selected_dev_decision)

    q, g = dev["query_indices"], dev["gallery_indices"]
    qids = dev_frame.iloc[q].vehicle_id.to_numpy()
    gids = dev_frame.iloc[g].vehicle_id.to_numpy()
    shuffled = np.random.default_rng(SEED).permutation(len(cal_frame))
    stress = {}
    for count in (len(cal_frame) // 4, len(cal_frame) // 2, len(cal_frame)):
        extra = shuffled[:count]
        case = pack_with_gallery(dev_emb[q], qids,
                                 np.concatenate([dev_emb[g], cal_emb[extra]]),
                                 np.concatenate([gids, cal_frame.iloc[extra].vehicle_id.to_numpy()]))
        base_case = case["scores"] >= cosine_threshold
        boosted_case = hybrid(case["scores"], predict(fitted[selected_name], case,
                                                     SPECS[selected_name]["kind"]),
                              selected_threshold, cosine_threshold) if promote_candidate else base_case
        stress[str(case["gallery_size"])] = {"cosine": metrics(case, base_case),
                                              "selected": metrics(case, boosted_case)}

    result = {
        "state": "completed", "model_version": MODEL_VERSION, "seed": SEED,
        "holdout_evaluated": False, "sklearn_version": sklearn.__version__,
        "sources_sha256": {**sources, "script": sha256(Path(__file__))},
        "fixed_specs": SPECS, "selection": {
            "basis": "5-fold query-ID grouped calibration OOF; max hybrid pair-F1 with query-F1 >= baseline and unchanged TNR; fixed first3; require +.005 calibration pair-F1 or cosine fallback",
            "selected_booster": selected_name, "hybrid_threshold": selected_threshold,
            "cosine_threshold": cosine_threshold, "candidate_passed_gate": promote_candidate,
        },
        "calibration_baseline": base_cal, "calibration_oof": calibration,
        "dev": dev_results,
        "selected_vs_cosine_bootstrap": bootstrap(dev, base_dev_decision, selected_dev_decision),
        "selected_vs_logistic_bootstrap": bootstrap(dev, old_decision, selected_dev_decision),
        "stress": stress,
        "limits": "All policies use frozen E2 and calibration-only selection. Dev is repeatedly used exploratory data. Stress adds calibration-ID distractors involved in fitting. Exact official F1 grain and million-scale gallery remain unknown.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.policy_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[selected_name], args.model_output)
    policy = {"schema_version": 1, "research_only": True,
              "model_version": MODEL_VERSION, "kind": "hybrid_hist_gradient_boosting",
              "feature_kind": SPECS[selected_name]["kind"], "topk": 10,
              "preserve_first_n": KEEP_FIRST, "cosine_threshold": cosine_threshold,
              "booster_threshold": selected_threshold, "model_file": args.model_output.name,
              "model_sha256": sha256(args.model_output), "sklearn_version": sklearn.__version__,
              "candidate_passed_calibration_gate": promote_candidate,
              "confidence_semantics": "raw cosine remains output confidence; booster score is an internal acceptance score"}
    args.policy_output.write_text(json.dumps(policy, ensure_ascii=False, indent=2))
    result["model_sha256"] = policy["model_sha256"]
    result["policy_sha256"] = sha256(args.policy_output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({"selected": selected_name, "passed_gate": promote_candidate,
                      "calibration": {k: {m: v[m]["metrics"]["candidate_micro_F1"]
                                           for m in ("pure", "hybrid")} for k, v in calibration.items()},
                      "dev": {k: {m: v[m] for m in ("candidate_micro_F1", "TNR", "query_correct_id_F1")}
                              for k, v in dev_results.items()},
                      "policy_sha256": result["policy_sha256"]}))


if __name__ == "__main__":
    main()
