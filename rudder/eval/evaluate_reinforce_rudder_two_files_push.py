#!/usr/bin/env python
"""
Comprehensive stress tests for reinforce_rudder_combined.pt using two SafeDICE input files.

Implements five diagnostics:
1) Ablation of Intent (factor independence)
2) Saliency Mapping (temporal credit assignment)
3) Zero-Shot Generalization (environment robustness)
4) Rank Correlation (continuous ranking quality)
5) Value Function Consistency (action sensitivity check)

Example:
    python rudder/eval/evaluate_reinforce_rudder_two_files_push.py \
        --checkpoint rudder/models/reinforce_rudder_combined.pt \
        --hr_lc_dataset_path /home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle \
        --lr_hc_dataset_path /home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0.pickle \
        --holdout_dataset rudder/dataset/reinforce_rudder_holdout_push.pkl \
        --output_dir rudder/eval/combined_model_tests_push
"""

import argparse
from datetime import datetime
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
    import pickle

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class CombinedRUDDER(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
        super().__init__()
        self.input_dim = state_dim + action_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        norm_out = self.layer_norm(lstm_out)
        drop_out = self.dropout(norm_out)
        return self.output_layer(drop_out)


class OriginalNoLayerNormRUDDER(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        return self.output_layer(lstm_out)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)


def _mem_available_gib() -> Optional[float]:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kib = float(parts[1])
                        return kib / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def _log_file_size(path: str, label: str) -> None:
    try:
        size_bytes = os.path.getsize(path)
        size_mib = size_bytes / (1024.0 * 1024.0)
        _log("%s size: %.2f MiB" % (label, size_mib))
    except Exception:
        _log("%s size: unknown" % label)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_input_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)

    candidates: List[str] = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.append(os.path.abspath(normalized))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", "models", normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "models", normalized)))
        candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", "dataset", normalized)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError("Input not found: %s. Tried: %s" % (path, ", ".join(candidates)))


def _resolve_output_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(_repo_root(), normalized))


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _load_dataset(dataset_path: str) -> Tuple[List[Dict[str, Any]], str]:
    resolved = _resolve_input_path(dataset_path)
    _log("Loading dataset: %s" % resolved)
    with open(resolved, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "trajectories" not in payload:
        raise ValueError("Expected dataset dict with key 'trajectories'")

    trajectories = payload["trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("No trajectories found in dataset")

    _log("Loaded %d trajectories" % len(trajectories))

    return trajectories, resolved


def _load_holdout_dataset(dataset_path: str) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray], str, Dict[str, Any]]:
    resolved = _resolve_input_path(dataset_path)
    _log("Loading holdout dataset: %s" % resolved)
    with open(resolved, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "trajectories" not in payload:
        raise ValueError("Expected holdout payload dict with key 'trajectories'")

    trajectories = payload["trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("No trajectories found in holdout dataset")

    labels_np: Optional[np.ndarray] = None
    if "labels" in payload:
        labels_np = np.asarray(payload["labels"], dtype=np.float32).reshape(-1)
        if labels_np.shape[0] != len(trajectories):
            raise ValueError("Holdout labels length does not match trajectories length")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    _log("Loaded holdout trajectories: %d" % len(trajectories))
    return trajectories, labels_np, resolved, metadata


def _load_safedice_pickle(path: str) -> Dict[str, np.ndarray]:
    resolved = _resolve_input_path(path)
    with open(resolved, "rb") as f:
        data = pickle.load(f)

    if "observations" in data:
        obs = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
        costs = np.asarray(data["costs"], dtype=np.float32).reshape(-1)
        dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1)
    elif "states" in data and "dones" in data:
        obs = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1)
        costs = np.asarray(data["costs"], dtype=np.float32).reshape(-1)
        dones = np.asarray(data["dones"], dtype=np.float32).reshape(-1)
    else:
        raise KeyError(
            "Unrecognised SafeDICE format in %s. Expected observations/actions/rewards/costs/dones"
            % resolved
        )

    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "costs": costs,
        "dones": dones,
    }


def _segment_into_trajectories(flat: Dict[str, np.ndarray], source_label: int) -> List[Dict[str, Any]]:
    obs = flat["obs"]
    actions = flat["actions"]
    rewards = flat["rewards"]
    costs = flat["costs"]
    dones = flat["dones"]

    n = len(obs)
    episode_end_indices = list(np.where(dones == 1)[0])
    if not episode_end_indices or episode_end_indices[-1] != n - 1:
        episode_end_indices.append(n - 1)

    trajectories: List[Dict[str, Any]] = []
    start = 0
    for end in episode_end_indices:
        end_excl = int(end + 1)
        if end_excl <= start:
            start = end_excl
            continue
        trajectories.append(
            {
                "states": obs[start:end_excl],
                "actions": actions[start:end_excl],
                "rewards": rewards[start:end_excl],
                "costs": costs[start:end_excl],
                "source": int(source_label),
            }
        )
        start = end_excl

    return trajectories


def _load_safedice_trajectories(path: str, source_label: int) -> Tuple[List[Dict[str, Any]], str]:
    resolved = _resolve_input_path(path)
    flat = _load_safedice_pickle(path)
    trajectories = _segment_into_trajectories(flat, source_label=source_label)
    return trajectories, resolved


def _load_combined_safedice(
    hr_lc_path: str,
    lr_hc_path: str,
    low_reward_threshold: float,
    high_reward_threshold: float,
    cost_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    hr_lc_trajs, hr_lc_resolved = _load_safedice_trajectories(hr_lc_path, source_label=1)
    lr_hc_trajs, lr_hc_resolved = _load_safedice_trajectories(lr_hc_path, source_label=0)

    def _filter(trajs: List[Dict[str, Any]], label: int) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for traj in trajs:
            rsum, csum = _trajectory_sums(traj)
            if label == 1:
                if rsum > high_reward_threshold and csum < cost_threshold:
                    kept.append(traj)
            else:
                if rsum < low_reward_threshold and csum > cost_threshold:
                    kept.append(traj)
        return kept

    hr_lc_filtered = _filter(hr_lc_trajs, label=1)
    lr_hc_filtered = _filter(lr_hc_trajs, label=0)

    if len(hr_lc_filtered) == 0:
        raise ValueError("No HR/LC trajectories passed the threshold filter")
    if len(lr_hc_filtered) == 0:
        raise ValueError("No LR/HC trajectories passed the threshold filter")

    combined = lr_hc_filtered + hr_lc_filtered
    metadata: Dict[str, Any] = {
        "mode": "safedice_separate_datasets",
        "hr_lc_path": hr_lc_resolved,
        "lr_hc_path": lr_hc_resolved,
        "hr_lc_raw_count": int(len(hr_lc_trajs)),
        "lr_hc_raw_count": int(len(lr_hc_trajs)),
        "hr_lc_filtered_count": int(len(hr_lc_filtered)),
        "lr_hc_filtered_count": int(len(lr_hc_filtered)),
        "combined_order": "lr_hc_then_hr_lc",
        "low_reward_threshold": float(low_reward_threshold),
        "high_reward_threshold": float(high_reward_threshold),
        "cost_threshold": float(cost_threshold),
    }
    return combined, metadata, (lr_hc_resolved + " + " + hr_lc_resolved)


def _extract_core_arrays(traj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = np.asarray(traj["states"], dtype=np.float32)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    rewards = np.asarray(traj.get("rewards", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
    costs = np.asarray(traj.get("costs", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
    n = int(min(len(states), len(actions), len(rewards), len(costs)))
    if n <= 0:
        raise ValueError("Encountered empty trajectory")
    return states[:n], actions[:n], rewards[:n], costs[:n]


def _pack_sa(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.concatenate([states, actions], axis=-1).astype(np.float32)


def _trajectory_sums(traj: Dict[str, Any]) -> Tuple[float, float]:
    _, _, rewards, costs = _extract_core_arrays(traj)
    return float(np.sum(rewards)), float(np.sum(costs))


def _trajectory_summary(trajs: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(trajs) == 0:
        return {
            "size": 0,
            "reward_min": float("nan"),
            "reward_mean": float("nan"),
            "reward_max": float("nan"),
            "cost_min": float("nan"),
            "cost_mean": float("nan"),
            "cost_max": float("nan"),
        }

    rewards = []
    costs = []
    for traj in trajs:
        reward_sum, cost_sum = _trajectory_sums(traj)
        rewards.append(reward_sum)
        costs.append(cost_sum)

    rewards_np = np.asarray(rewards, dtype=np.float32)
    costs_np = np.asarray(costs, dtype=np.float32)
    return {
        "size": int(len(trajs)),
        "reward_min": float(rewards_np.min()),
        "reward_mean": float(rewards_np.mean()),
        "reward_max": float(rewards_np.max()),
        "cost_min": float(costs_np.min()),
        "cost_mean": float(costs_np.mean()),
        "cost_max": float(costs_np.max()),
    }


def _make_synthetic_coward_variant(base_traj: Dict[str, Any], variant_index: int) -> Dict[str, Any]:
    states, actions, rewards, costs = _extract_core_arrays(base_traj)
    n = len(states)
    synthetic_states = np.repeat(states[:1], n, axis=0).astype(np.float32)
    synthetic_actions = np.zeros_like(actions, dtype=np.float32)
    synthetic_rewards = np.zeros_like(rewards, dtype=np.float32)
    synthetic_costs = np.zeros_like(costs, dtype=np.float32)

    if variant_index > 0:
        rng = np.random.RandomState(variant_index)
        synthetic_states = synthetic_states + rng.normal(0.0, 1e-3, size=synthetic_states.shape).astype(np.float32)
        synthetic_actions = synthetic_actions + rng.normal(0.0, 1e-3, size=synthetic_actions.shape).astype(np.float32)

    synthetic_traj = dict(base_traj)
    synthetic_traj["states"] = synthetic_states
    synthetic_traj["actions"] = synthetic_actions
    synthetic_traj["rewards"] = synthetic_rewards
    synthetic_traj["costs"] = synthetic_costs
    synthetic_traj["synthetic"] = True
    synthetic_traj["synthetic_type"] = "coward"
    synthetic_traj["synthetic_variant"] = int(variant_index)
    return synthetic_traj


def _infer_model(ckpt: Dict[str, Any], device: str) -> Tuple[nn.Module, float, int, int, int, float, float]:
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing model_state_dict")

    model_state = ckpt["model_state_dict"]
    state_dim = int(ckpt["state_dim"])
    action_dim = int(ckpt["action_dim"])
    seq_len = int(ckpt.get("seq_len", 0))
    baseline = float(ckpt.get("baseline", 0.0))
    reward_threshold = float(ckpt.get("reward_threshold", 15.0))
    cost_threshold = float(ckpt.get("cost_threshold", 25.0))

    hidden_dim = int(model_state["lstm.weight_ih_l0"].shape[0] // 4)
    has_layer_norm = ("layer_norm.weight" in model_state) and ("layer_norm.bias" in model_state)

    if has_layer_norm:
        model = CombinedRUDDER(state_dim, action_dim, hidden_dim=hidden_dim, dropout=float(ckpt.get("dropout", 0.2)))
    else:
        model = OriginalNoLayerNormRUDDER(state_dim, action_dim, hidden_dim=hidden_dim)

    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.eval()
    return model, baseline, state_dim, action_dim, seq_len, reward_threshold, cost_threshold


def _predict_sequence(model: nn.Module, baseline: float, sa_seq: np.ndarray, seq_len: int, device: str) -> np.ndarray:
    feature_dim = int(sa_seq.shape[-1])
    x = np.zeros((1, seq_len, feature_dim), dtype=np.float32)
    n = int(min(len(sa_seq), seq_len))
    x[0, :n] = sa_seq[:n]

    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(device))
        g_t = pred[0, :, 0].detach().cpu().numpy() + baseline

    return g_t[:n]


def _predict_final(model: nn.Module, baseline: float, sa_seq: np.ndarray, seq_len: int, device: str) -> float:
    g_t = _predict_sequence(model, baseline, sa_seq, seq_len, device)
    return float(g_t[-1])


def _label_conjunction(reward_sum: float, cost_sum: float, reward_threshold: float, cost_threshold: float) -> float:
    return 1.0 if (reward_sum > reward_threshold and cost_sum < cost_threshold) else 0.0


def _scalar_label(value: Any) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _label_from_trajectory(
    traj: Dict[str, Any],
    reward_threshold: float,
    cost_threshold: float,
    preferred_source_values: List[int],
) -> float:
    if "preference_label" in traj:
        val = _scalar_label(traj["preference_label"])
        if not np.isnan(val):
            return 1.0 if val > 0.5 else 0.0

    if "source" in traj:
        src = int(round(_scalar_label(traj["source"])))
        return 1.0 if src in preferred_source_values else 0.0

    reward_sum, cost_sum = _trajectory_sums(traj)
    return _label_conjunction(reward_sum, cost_sum, reward_threshold, cost_threshold)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_average_ties(x)
    ry = _rankdata_average_ties(y)
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(rx * ry) / denom)


def _batch_predict_final_scores(
    model: nn.Module,
    baseline: float,
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    seq_len: int,
    device: str,
    batch_size: int = 128,
    log_prefix: str = "batch-predict",
) -> np.ndarray:
    n_total = len(trajectories)
    if n_total == 0:
        return np.zeros((0,), dtype=np.float32)

    feature_dim = state_dim + action_dim
    batch_size = int(max(1, batch_size))
    preds = np.zeros((n_total,), dtype=np.float32)

    _log("%s: predicting %d trajectories with batch_size=%d" % (log_prefix, n_total, batch_size))
    n_batches = int(np.ceil(n_total / float(batch_size)))

    for b in range(n_batches):
        start = b * batch_size
        end = min(n_total, start + batch_size)
        x = np.zeros((end - start, seq_len, feature_dim), dtype=np.float32)

        for j, i in enumerate(range(start, end)):
            states, actions, _, _ = _extract_core_arrays(trajectories[i])
            sa = _pack_sa(states, actions)
            n = int(min(len(sa), seq_len))
            x[j, :n] = sa[:n]

        with torch.no_grad():
            batch_pred = model(torch.from_numpy(x).to(device))[:, -1, 0].detach().cpu().numpy() + baseline
        preds[start:end] = batch_pred.astype(np.float32)

        if (b + 1) % max(1, n_batches // 10) == 0 or (b + 1) == n_batches:
            _log("%s: batch %d/%d complete" % (log_prefix, b + 1, n_batches))

    return preds


def ablation_of_intent_test(
    model: nn.Module,
    baseline: float,
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    seq_len: int,
    reward_threshold: float,
    cost_threshold: float,
    device: str,
    predict_batch_size: int,
    min_set_b_size: int,
) -> Dict[str, Any]:
    _log("Ablation: computing reward/cost cohorts")
    reward_sums = []
    cost_sums = []
    for traj in trajectories:
        _, _, rewards, costs = _extract_core_arrays(traj)
        reward_sums.append(float(np.sum(rewards)))
        cost_sums.append(float(np.sum(costs)))

    reward_sums_np = np.asarray(reward_sums, dtype=np.float32)
    cost_sums_np = np.asarray(cost_sums, dtype=np.float32)

    set_a_idx = np.where((reward_sums_np > reward_threshold) & (cost_sums_np >= cost_threshold))[0]
    set_b_idx = np.where((reward_sums_np <= reward_threshold) & (cost_sums_np < cost_threshold))[0]

    set_a = [trajectories[int(i)] for i in set_a_idx.tolist()]
    set_b = [trajectories[int(i)] for i in set_b_idx.tolist()]

    set_a_real = list(set_a)
    set_b_real = list(set_b)

    if len(set_b) < int(min_set_b_size):
        _log(
            "Ablation: set B has only %d real samples; backfilling to %d with synthetic coward trajectories"
            % (len(set_b), int(min_set_b_size))
        )
        coward_sources = list(set_b_real)
        if len(coward_sources) == 0:
            candidate_idx = np.argsort((reward_sums_np * 1000.0) + cost_sums_np)
            coward_sources = [trajectories[int(i)] for i in candidate_idx[: max(1, int(min_set_b_size))].tolist()]

        synthetic_needed = int(min_set_b_size) - len(set_b)
        for variant_index in range(synthetic_needed):
            base_traj = coward_sources[variant_index % len(coward_sources)]
            set_b.append(_make_synthetic_coward_variant(base_traj, variant_index))

    # If set A is unexpectedly small, keep the real cohort and let the summary show the shortage.
    if len(set_a) == 0:
        _log("Ablation: WARNING set A is empty before evaluation")

    _log("Ablation: set A size=%d (real=%d) | set B size=%d (real=%d)" % (len(set_a), len(set_a_real), len(set_b), len(set_b_real)))

    preds_a = _batch_predict_final_scores(
        model,
        baseline,
        set_a,
        state_dim,
        action_dim,
        seq_len,
        device,
        batch_size=predict_batch_size,
        log_prefix="ablation-setA",
    ) if len(set_a) > 0 else np.zeros((0,), dtype=np.float32)
    preds_b = _batch_predict_final_scores(
        model,
        baseline,
        set_b,
        state_dim,
        action_dim,
        seq_len,
        device,
        batch_size=predict_batch_size,
        log_prefix="ablation-setB",
    ) if len(set_b) > 0 else np.zeros((0,), dtype=np.float32)

    class_a = (preds_a >= 0.5).astype(np.float32)
    class_b = (preds_b >= 0.5).astype(np.float32)

    p_nonpreferred_a = float(np.mean(class_a == 0.0)) if len(class_a) > 0 else 0.0
    p_nonpreferred_b = float(np.mean(class_b == 0.0)) if len(class_b) > 0 else 0.0

    set_a_summary = _trajectory_summary(set_a)
    set_b_summary = _trajectory_summary(set_b)

    return {
        "set_a_desc": "Kamikaze: high reward + high cost -> expected label 0",
        "set_b_desc": "Coward: low reward + low cost -> expected label 0",
        "set_a_size": int(len(set_a)),
        "set_a_real_size": int(len(set_a_real)),
        "set_b_size": int(len(set_b)),
        "set_b_real_size": int(len(set_b_real)),
        "set_b_synthetic_size": int(max(0, len(set_b) - len(set_b_real))),
        "set_a_pred_nonpreferred_rate": p_nonpreferred_a,
        "set_b_pred_nonpreferred_rate": p_nonpreferred_b,
        "set_a_safety_blind_rate": float(np.mean(class_a == 1.0)) if len(class_a) > 0 else 0.0,
        "set_a_reward_min": set_a_summary["reward_min"],
        "set_a_reward_mean": set_a_summary["reward_mean"],
        "set_a_reward_max": set_a_summary["reward_max"],
        "set_a_cost_min": set_a_summary["cost_min"],
        "set_a_cost_mean": set_a_summary["cost_mean"],
        "set_a_cost_max": set_a_summary["cost_max"],
        "set_b_reward_min": set_b_summary["reward_min"],
        "set_b_reward_mean": set_b_summary["reward_mean"],
        "set_b_reward_max": set_b_summary["reward_max"],
        "set_b_cost_min": set_b_summary["cost_min"],
        "set_b_cost_mean": set_b_summary["cost_mean"],
        "set_b_cost_max": set_b_summary["cost_max"],
        "success": bool(p_nonpreferred_a >= 0.9 and p_nonpreferred_b >= 0.9 and len(set_a) > 0 and len(set_b) > 0),
        "notes": "Pass indicates conjunction behavior (goal AND safety). Set B is backfilled with synthetic coward trajectories when needed.",
    }


def saliency_mapping_test(
    model: nn.Module,
    baseline: float,
    trajectories: List[Dict[str, Any]],
    seq_len: int,
    reward_threshold: float,
    cost_threshold: float,
    corruption_window: int,
    output_dir: str,
    device: str,
) -> Dict[str, Any]:
    _log("Saliency: scanning for preferred trajectories")
    preferred_candidates: List[Tuple[int, float]] = []

    for i, traj in enumerate(trajectories):
        states, actions, rewards, costs = _extract_core_arrays(traj)
        rsum = float(np.sum(rewards))
        csum = float(np.sum(costs))
        y = _label_conjunction(rsum, csum, reward_threshold, cost_threshold)
        if y < 0.5:
            continue
        sa = _pack_sa(states, actions)
        g_final = _predict_final(model, baseline, sa, seq_len, device)
        preferred_candidates.append((i, g_final))
        if len(preferred_candidates) % 200 == 0:
            _log("Saliency: found %d preferred candidates so far" % len(preferred_candidates))

    if len(preferred_candidates) == 0:
        return {
            "success": False,
            "error": "No preferred trajectory found for corruption test.",
        }

    best_idx = int(sorted(preferred_candidates, key=lambda z: z[1], reverse=True)[0][0])
    _log("Saliency: selected trajectory index %d for corruption" % best_idx)
    states, actions, rewards, costs = _extract_core_arrays(trajectories[best_idx])
    sa_clean = _pack_sa(states, actions)

    n = len(sa_clean)
    if n < 4:
        return {"success": False, "error": "Trajectory too short for saliency analysis."}

    w = int(max(1, min(corruption_window, max(1, n // 3))))
    g_clean = _predict_sequence(model, baseline, sa_clean, seq_len, device)
    action_dim = actions.shape[-1]
    state_noise_scale = float(np.std(states) + 1e-6)

    def _corrupt_window(anchor_start: int, rng_seed: int) -> Dict[str, Any]:
        start = int(max(0, min(anchor_start, n - w)))
        end = int(min(n, start + w))

        sa_corrupt = sa_clean.copy()
        rng = np.random.RandomState(rng_seed)
        sa_corrupt[start:end, -action_dim:] *= -1.0
        sa_corrupt[start:end, :-action_dim] += rng.normal(
            0.0,
            0.25 * state_noise_scale,
            size=sa_corrupt[start:end, :-action_dim].shape,
        ).astype(np.float32)

        g_corrupt = _predict_sequence(model, baseline, sa_corrupt, seq_len, device)

        r_clean = np.zeros_like(g_clean)
        r_corrupt = np.zeros_like(g_corrupt)
        r_clean[0] = g_clean[0]
        r_corrupt[0] = g_corrupt[0]
        if len(g_clean) > 1:
            r_clean[1:] = g_clean[1:] - g_clean[:-1]
            r_corrupt[1:] = g_corrupt[1:] - g_corrupt[:-1]

        delta_r = r_corrupt - r_clean
        window_slice = slice(start, end)
        spike_idx_rel = int(np.argmin(delta_r[window_slice]))
        spike_idx = int(start + spike_idx_rel)
        spike_value = float(delta_r[spike_idx])

        return {
            "start": start,
            "end": end,
            "g_corrupt": g_corrupt,
            "delta_r": delta_r,
            "spike_idx": spike_idx,
            "spike_value": spike_value,
            "clean_final": float(g_clean[-1]),
            "corrupt_final": float(g_corrupt[-1]),
            "success": bool(spike_value < -0.05 and float(g_corrupt[-1]) < float(g_clean[-1])),
        }

    saliency_positions = [
        ("start", 0),
        ("early", max(1, n // 4 - w // 2)),
        ("midpoint", max(1, n // 2 - w // 2)),
        ("ending", max(0, n - w)),
    ]

    saliency_checks: List[Dict[str, Any]] = []
    for idx, (label, anchor_start) in enumerate(saliency_positions):
        result = _corrupt_window(anchor_start, rng_seed=123 + idx)
        result["label"] = label
        saliency_checks.append(result)

    success = bool(any(check["success"] for check in saliency_checks))

    plot_path = ""
    if HAS_MPL:
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "saliency_corruption_plot_push.png")
        fig, axes = plt.subplots(len(saliency_checks) + 1, 1, figsize=(10, 3.5 * (len(saliency_checks) + 1)), sharex=True)

        axes[0].plot(g_clean, label="g_t clean", color="tab:blue")
        for check in saliency_checks:
            axes[0].axvspan(check["start"], check["end"], alpha=0.12, label="%s window" % check["label"])
        axes[0].set_ylabel("g_t")
        axes[0].legend(loc="best")
        axes[0].grid(alpha=0.25)

        for idx, check in enumerate(saliency_checks, start=1):
            axes[idx].plot(check["delta_r"], label="delta redistributed reward", color="tab:purple")
            axes[idx].axvline(check["spike_idx"], color="black", linestyle="--", alpha=0.7, label="min spike")
            axes[idx].axvspan(check["start"], check["end"], color="gray", alpha=0.2)
            axes[idx].set_ylabel("delta r_t")
            axes[idx].set_title(
                "%s corruption | spike=%.4f | final_drop=%.4f"
                % (check["label"], check["spike_value"], check["clean_final"] - check["corrupt_final"])
            )
            axes[idx].legend(loc="best")
            axes[idx].grid(alpha=0.25)

        axes[-1].set_xlabel("timestep")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "trajectory_index": int(best_idx),
        "clean_final_gT": float(g_clean[-1]),
        "saliency_checks": [
            {
                "label": check["label"],
                "corruption_window": [int(check["start"]), int(check["end"])],
                "corrupt_final_gT": float(check["corrupt_final"]),
                "min_delta_rt_index": int(check["spike_idx"]),
                "min_delta_rt_value": float(check["spike_value"]),
                "success": bool(check["success"]),
            }
            for check in saliency_checks
        ],
        "plot_path": plot_path,
        "success": success,
        "notes": "Pass indicates a sharp local negative response around corruption and lower final confidence. Multiple windows are tested: start, early, midpoint.",
    }


def zero_shot_generalization_test(
    model: nn.Module,
    baseline: float,
    zero_shot_trajectories: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    seq_len: int,
    reward_threshold: float,
    cost_threshold: float,
    device: str,
    predict_batch_size: int,
    success_accuracy: float = 0.85,
) -> Dict[str, Any]:
    preds = _batch_predict_final_scores(
        model,
        baseline,
        zero_shot_trajectories,
        state_dim,
        action_dim,
        seq_len,
        device,
        batch_size=predict_batch_size,
        log_prefix="zero-shot",
    )
    pred_cls = (preds >= 0.5).astype(np.float32)

    y_true = np.zeros((len(zero_shot_trajectories),), dtype=np.float32)
    for i, traj in enumerate(zero_shot_trajectories):
        _, _, rewards, costs = _extract_core_arrays(traj)
        y_true[i] = _label_conjunction(float(np.sum(rewards)), float(np.sum(costs)), reward_threshold, cost_threshold)

    acc = float(np.mean(pred_cls == y_true)) if len(y_true) > 0 else 0.0
    return {
        "num_trajectories": int(len(zero_shot_trajectories)),
        "accuracy": acc,
        "success_threshold": float(success_accuracy),
        "success": bool(acc >= success_accuracy),
        "notes": "Accuracy near random (~0.5) suggests map overfitting instead of concept learning.",
    }


def holdout_classifier_accuracy_test(
    model: nn.Module,
    baseline: float,
    holdout_trajectories: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    seq_len: int,
    reward_threshold: float,
    cost_threshold: float,
    preferred_source_values: List[int],
    holdout_labels: Optional[np.ndarray],
    device: str,
    predict_batch_size: int,
) -> Dict[str, Any]:
    preds = _batch_predict_final_scores(
        model,
        baseline,
        holdout_trajectories,
        state_dim,
        action_dim,
        seq_len,
        device,
        batch_size=predict_batch_size,
        log_prefix="holdout-accuracy",
    )
    pred_cls = (preds >= 0.5).astype(np.float32)

    if holdout_labels is not None:
        y_true = np.asarray(holdout_labels, dtype=np.float32).reshape(-1)
    else:
        y_true = np.zeros((len(holdout_trajectories),), dtype=np.float32)
        for i, traj in enumerate(holdout_trajectories):
            y_true[i] = _label_from_trajectory(
                traj=traj,
                reward_threshold=reward_threshold,
                cost_threshold=cost_threshold,
                preferred_source_values=preferred_source_values,
            )

    accuracy = float(np.mean(pred_cls == y_true)) if len(y_true) > 0 else 0.0
    tp = int(np.sum((pred_cls == 1.0) & (y_true == 1.0)))
    tn = int(np.sum((pred_cls == 0.0) & (y_true == 0.0)))
    fp = int(np.sum((pred_cls == 1.0) & (y_true == 0.0)))
    fn = int(np.sum((pred_cls == 0.0) & (y_true == 1.0)))

    return {
        "num_trajectories": int(len(holdout_trajectories)),
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "success": bool(accuracy >= 0.85),
        "notes": "Ground-truth labels use preference_label, then source mapping, then conjunction fallback.",
    }


def _safe_import_sklearn_metrics():
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

        return {
            "roc_auc_score": roc_auc_score,
            "average_precision_score": average_precision_score,
            "balanced_accuracy_score": balanced_accuracy_score,
        }
    except Exception:
        return {}


def _compute_calibration_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_prob) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) == 0:
            continue
        avg_conf = float(np.mean(y_prob[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_prob)) * abs(avg_conf - avg_acc)
    return float(ece)


def construct_near_threshold_holdout(trajectories, reward_threshold, cost_threshold, delta=3.0, max_samples=200, seed=0):
    cand_pref = []
    cand_nonpref = []
    for traj in trajectories:
        _, _, rewards, costs = _extract_core_arrays(traj)
        rsum = float(np.sum(rewards))
        csum = float(np.sum(costs))
        near = (abs(rsum - reward_threshold) <= delta) or (abs(csum - cost_threshold) <= delta)
        if not near:
            continue
        label = 1 if (rsum > reward_threshold and csum < cost_threshold) else 0
        if label == 1:
            cand_pref.append(traj)
        else:
            cand_nonpref.append(traj)

    rng = np.random.RandomState(seed)
    n_pref = min(len(cand_pref), max_samples // 2)
    n_nonpref = min(len(cand_nonpref), max_samples - n_pref)
    if n_pref + n_nonpref == 0:
        return []
    chosen = []
    if n_pref > 0:
        chosen += list(rng.choice(cand_pref, size=n_pref, replace=False))
    if n_nonpref > 0:
        chosen += list(rng.choice(cand_nonpref, size=n_nonpref, replace=False))
    rng.shuffle(chosen)
    return chosen


def apply_state_noise_to_trajectory(traj, sigma=0.05, mask_fraction=0.0, seed=0):
    rng = np.random.RandomState(seed)
    traj_copy = dict(traj)
    states = np.asarray(traj_copy.get("states"), dtype=np.float32).copy()
    noise = rng.normal(scale=sigma, size=states.shape).astype(np.float32)
    states = states + noise
    if mask_fraction > 0.0:
        num_mask = int(np.ceil(states.size * mask_fraction))
        flat_idx = rng.choice(states.size, size=num_mask, replace=False)
        states.flat[flat_idx] = 0.0
    traj_copy["states"] = states
    return traj_copy


def holdout_robust_metrics_test(
    model: nn.Module,
    baseline: float,
    holdout_trajectories: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    seq_len: int,
    reward_threshold: float,
    cost_threshold: float,
    preferred_source_values: List[int],
    holdout_labels: Optional[np.ndarray],
    device: str,
    predict_batch_size: int,
    compute_robust: bool = True,
    noise_sigma: float = 0.05,
    mask_fraction: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    preds = _batch_predict_final_scores(
        model,
        baseline,
        holdout_trajectories,
        state_dim,
        action_dim,
        seq_len,
        device,
        batch_size=predict_batch_size,
        log_prefix="holdout-robust",
    )
    if holdout_labels is not None:
        y_true = np.asarray(holdout_labels, dtype=np.float32).reshape(-1)
    else:
        y_true = np.zeros((len(holdout_trajectories),), dtype=np.float32)
        for i, traj in enumerate(holdout_trajectories):
            y_true[i] = _label_from_trajectory(
                traj=traj,
                reward_threshold=reward_threshold,
                cost_threshold=cost_threshold,
                preferred_source_values=preferred_source_values,
            )

    pred_cls = (preds >= 0.5).astype(np.float32)
    accuracy = float(np.mean(pred_cls == y_true)) if len(y_true) > 0 else 0.0

    metrics = {
        "num_trajectories": int(len(holdout_trajectories)),
        "accuracy": accuracy,
        "tp": int(np.sum((pred_cls == 1.0) & (y_true == 1.0))),
        "tn": int(np.sum((pred_cls == 0.0) & (y_true == 0.0))),
        "fp": int(np.sum((pred_cls == 1.0) & (y_true == 0.0))),
        "fn": int(np.sum((pred_cls == 0.0) & (y_true == 1.0))),
    }

    sklearn_metrics = _safe_import_sklearn_metrics()
    if "roc_auc_score" in sklearn_metrics and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(sklearn_metrics["roc_auc_score"](y_true, preds))
        except Exception:
            metrics["roc_auc"] = float('nan')
    else:
        metrics["roc_auc"] = float('nan')

    if "average_precision_score" in sklearn_metrics and len(np.unique(y_true)) > 1:
        try:
            metrics["pr_auc"] = float(sklearn_metrics["average_precision_score"](y_true, preds))
        except Exception:
            metrics["pr_auc"] = float('nan')
    else:
        metrics["pr_auc"] = float('nan')

    if "balanced_accuracy_score" in sklearn_metrics and len(np.unique(y_true)) > 1:
        try:
            metrics["balanced_accuracy"] = float(sklearn_metrics["balanced_accuracy_score"](y_true, pred_cls))
        except Exception:
            metrics["balanced_accuracy"] = float('nan')
    else:
        metrics["balanced_accuracy"] = float('nan')

    metrics["ece"] = _compute_calibration_ece(y_true, preds, n_bins=10)

    if compute_robust and len(holdout_trajectories) > 0:
        corrupted = [apply_state_noise_to_trajectory(t, sigma=noise_sigma, mask_fraction=mask_fraction, seed=seed + i) for i, t in enumerate(holdout_trajectories)]
        cpreds = _batch_predict_final_scores(
            model,
            baseline,
            corrupted,
            state_dim,
            action_dim,
            seq_len,
            device,
            batch_size=predict_batch_size,
            log_prefix="holdout-corrupted",
        )
        cpred_cls = (cpreds >= 0.5).astype(np.float32)
        metrics["corrupted_accuracy"] = float(np.mean(cpred_cls == y_true))
        metrics["delta_accuracy_corrupted"] = metrics["corrupted_accuracy"] - metrics["accuracy"]
    return metrics


def rank_correlation_test(
    model: nn.Module,
    baseline: float,
    trajectories: List[Dict[str, Any]],
    state_dim: int,
    action_dim: int,
    seq_len: int,
    utility_cost_weight: float,
    max_samples: int,
    device: str,
    predict_batch_size: int,
) -> Dict[str, Any]:
    n = int(min(max_samples, len(trajectories)))
    sample = trajectories[:n]

    preds = _batch_predict_final_scores(
        model,
        baseline,
        sample,
        state_dim,
        action_dim,
        seq_len,
        device,
        batch_size=predict_batch_size,
        log_prefix="rank-correlation",
    )
    utility = np.zeros((n,), dtype=np.float32)

    for i, traj in enumerate(sample):
        _, _, rewards, costs = _extract_core_arrays(traj)
        utility[i] = float(np.sum(rewards) - utility_cost_weight * np.sum(costs))

    rho = _spearmanr(preds, utility)
    return {
        "num_trajectories": int(n),
        "utility_cost_weight": float(utility_cost_weight),
        "spearman_rho": float(rho),
        "success": bool(rho > 0.7),
        "notes": "Higher rho means the model encodes degrees of goodness, not only binary pass/fail.",
    }


def value_function_consistency_test(
    model: nn.Module,
    baseline: float,
    trajectories: List[Dict[str, Any]],
    seq_len: int,
    utility_cost_weight: float,
    max_checks: int,
    neighbor_pool_limit: int,
    device: str,
) -> Dict[str, Any]:
    _log("Bellman check: building local transition pool (limit=%d)" % int(neighbor_pool_limit))
    # Build a reusable pool of (state, action, immediate utility) tuples.
    pool_states: List[np.ndarray] = []
    pool_actions: List[np.ndarray] = []
    pool_u: List[float] = []

    for traj in trajectories:
        states, actions, rewards, costs = _extract_core_arrays(traj)
        imm_u = rewards - utility_cost_weight * costs
        for t in range(len(states)):
            pool_states.append(states[t])
            pool_actions.append(actions[t])
            pool_u.append(float(imm_u[t]))
            if len(pool_states) >= neighbor_pool_limit:
                break
        if len(pool_states) >= neighbor_pool_limit:
            break

    if len(pool_states) < 20:
        return {"success": False, "error": "Insufficient transition pool for action-sensitivity check."}

    S = np.asarray(pool_states, dtype=np.float32)
    A = np.asarray(pool_actions, dtype=np.float32)
    U = np.asarray(pool_u, dtype=np.float32)

    diffs: List[float] = []
    anchor_kinds: List[str] = []
    used = 0
    _log("Bellman check: evaluating up to %d anchor states (start + midpoint per trajectory)" % int(max_checks))

    for traj in trajectories:
        states, actions, rewards, costs = _extract_core_arrays(traj)
        if len(states) < 4:
            continue

        for anchor_kind, t in (("start", 0), ("midpoint", int(min(len(states) - 1, len(states) // 2)))):
            anchor_state = states[t]

            d2 = np.sum((S - anchor_state[None, :]) ** 2, axis=1)
            k = int(min(64, len(d2)))
            nn_idx = np.argpartition(d2, k - 1)[:k]

            local_u = U[nn_idx]
            local_a = A[nn_idx]

            good_idx = int(np.argmax(local_u))
            bad_idx = int(np.argmin(local_u))

            a_good = local_a[good_idx]
            a_bad = local_a[bad_idx]

            sa_good = _pack_sa(states.copy(), actions.copy())
            sa_bad = sa_good.copy()

            action_dim = actions.shape[-1]
            sa_good[t, -action_dim:] = a_good
            sa_bad[t, -action_dim:] = a_bad

            g_good = _predict_sequence(model, baseline, sa_good, seq_len, device)
            g_bad = _predict_sequence(model, baseline, sa_bad, seq_len, device)
            diffs.append(float(g_good[t] - g_bad[t]))
            anchor_kinds.append(anchor_kind)

            used += 1
            if used % 10 == 0:
                _log("Bellman check: completed %d/%d checks" % (used, int(max_checks)))
            if used >= max_checks:
                break

        if used >= max_checks:
            break

    if len(diffs) == 0:
        return {"success": False, "error": "No valid checks produced for action-sensitivity test."}

    diffs_np = np.asarray(diffs, dtype=np.float32)
    mean_diff = float(np.mean(diffs_np))
    frac_positive = float(np.mean(diffs_np > 0.0))

    return {
        "num_checks": int(len(diffs_np)),
        "anchor_kinds": anchor_kinds,
        "mean_g_good_minus_g_bad": mean_diff,
        "fraction_positive": frac_positive,
        "success": bool(mean_diff > 0.02 and frac_positive > 0.65),
        "notes": "Higher g(s,a_good) than g(s,a_bad) indicates action sensitivity consistent with value-like behavior.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reinforce_rudder_combined.pt with five advanced diagnostics (_push)")
    parser.add_argument("--checkpoint", type=str, default="rudder/models/reinforce_rudder_combined.pt")
    parser.add_argument(
        "--hr_lc_dataset_path",
        type=str,
        default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle",
        help="Preferred HR/LC SafeDICE pickle used during training",
    )
    parser.add_argument(
        "--lr_hc_dataset_path",
        type=str,
        default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0.pickle",
        help="Non-preferred LR/HC SafeDICE pickle used during training",
    )
    parser.add_argument(
        "--holdout_dataset",
        type=str,
        default="rudder/dataset/reinforce_rudder_holdout_push.pkl",
        help="Deterministic holdout payload created by create_reinforce_rudder_holdout_push.py",
    )
    parser.add_argument("--output_dir", type=str, default="rudder/eval/combined_model_tests_push")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--low_reward_threshold", type=float, default=2.5)
    parser.add_argument("--high_reward_threshold", type=float, default=5.0)
    parser.add_argument("--cost_threshold", type=float, default=25.0, help="Cost threshold used in training/eval")

    parser.add_argument("--corruption_window", type=int, default=100)
    parser.add_argument("--rank_samples", type=int, default=100)
    parser.add_argument("--utility_cost_weight", type=float, default=1.0)
    parser.add_argument("--bellman_checks", type=int, default=40)
    parser.add_argument("--neighbor_pool_limit", type=int, default=8000)
    parser.add_argument("--predict_batch_size", type=int, default=128)
    parser.add_argument("--min_set_b_size", type=int, default=50)
    parser.add_argument("--hard_delta", type=float, default=3.0, help="Delta around thresholds for near-boundary holdout")
    parser.add_argument("--hard_holdout_size", type=int, default=200, help="Max size for near-threshold hard holdout")
    parser.add_argument("--noise_sigma", type=float, default=0.05, help="Gaussian noise sigma for corrupted holdout")
    parser.add_argument("--mask_fraction", type=float, default=0.0, help="Fraction of state entries to zero for corrupted holdout")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = _resolve_input_path(args.checkpoint)
    _log("Loading checkpoint: %s" % checkpoint_path)
    _log_file_size(checkpoint_path, "Checkpoint")
    ckpt = torch.load(checkpoint_path, map_location=args.device)
    (
        model,
        baseline,
        state_dim,
        action_dim,
        seq_len_ckpt,
        _reward_threshold_ckpt,
        _cost_threshold_ckpt,
    ) = _infer_model(ckpt, args.device)

    hr_lc_resolved = _resolve_input_path(args.hr_lc_dataset_path)
    lr_hc_resolved = _resolve_input_path(args.lr_hc_dataset_path)
    _log_file_size(hr_lc_resolved, "HR/LC dataset")
    _log_file_size(lr_hc_resolved, "LR/HC dataset")
    mem_gib = _mem_available_gib()
    if mem_gib is not None:
        _log("MemAvailable before primary load: %.2f GiB" % mem_gib)
    trajectories, primary_meta, dataset_resolved = _load_combined_safedice(
        hr_lc_path=args.hr_lc_dataset_path,
        lr_hc_path=args.lr_hc_dataset_path,
        low_reward_threshold=float(args.low_reward_threshold),
        high_reward_threshold=float(args.high_reward_threshold),
        cost_threshold=float(args.cost_threshold),
    )

    holdout_resolved_input = _resolve_input_path(args.holdout_dataset)
    _log_file_size(holdout_resolved_input, "Holdout dataset")
    holdout_trajectories, holdout_labels, holdout_path, holdout_metadata = _load_holdout_dataset(holdout_resolved_input)

    reward_threshold = float(args.high_reward_threshold)
    cost_threshold = float(args.cost_threshold)
    preferred_source_values = [int(v) for v in ckpt.get("preferred_source_values", [1, 3])]

    # When checkpoint did not store seq_len, use the max trajectory length from the dataset.
    if seq_len_ckpt <= 0:
        seq_len = int(max(min(len(np.asarray(t["states"])), len(np.asarray(t["actions"]))) for t in trajectories))
    else:
        seq_len = int(seq_len_ckpt)

    out_dir = _resolve_output_path(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("REINFORCE RUDDER COMBINED - ADVANCED EVAL SUITE")
    print("=" * 80)
    _log("Checkpoint: %s" % checkpoint_path)
    _log("Primary dataset (from two files): %s" % dataset_resolved)
    _log("Holdout dataset: %s" % holdout_path)
    _log("state_dim=%d action_dim=%d seq_len=%d baseline=%.6f" % (state_dim, action_dim, seq_len, baseline))
    _log("Conjunction label rule: reward > %.3f AND cost < %.3f" % (reward_threshold, cost_threshold))

    results: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "dataset": dataset_resolved,
        "primary_dataset_metadata": _json_safe(primary_meta),
        "holdout_dataset": holdout_path,
        "holdout_dataset_metadata": _json_safe(holdout_metadata),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "seq_len": seq_len,
        "baseline": baseline,
        "reward_threshold": reward_threshold,
        "cost_threshold": cost_threshold,
        "metric_explanations": {
            "ablation_of_intent": "Compares trajectories with high reward/high cost against low reward/low cost cohorts to check whether the model separates goal achievement from safety cost.",
            "saliency_mapping": "Corrupts a local time window in a selected preferred trajectory and checks whether the model's redistributed reward shows a local negative spike and whether the final score drops.",
            "zero_shot_generalization": "Evaluates the model on the holdout dataset using the reward-and-cost conjunction rule as ground truth to measure out-of-sample concept generalization.",
            "holdout_classifier_accuracy": "Evaluates the model on the holdout dataset using dataset labels when available, falling back to source mapping or the reward-and-cost rule if needed.",
            "rank_correlation": "Computes Spearman correlation between the model's final trajectory score and true utility defined as cumulative reward minus cumulative cost.",
            "holdout_rank_correlation": "Same rank-correlation metric as above, but computed on the holdout trajectories to measure generalization of the score ordering.",
            "value_function_consistency": "Swaps in locally good and bad actions at the same state and checks whether the model assigns a higher score to the better action.",
        },
        "evaluation_scope": {
            "main_dataset": "Used for ablation_of_intent, saliency_mapping, rank_correlation, and value_function_consistency.",
            "holdout_dataset": "Used for zero_shot_generalization, holdout_classifier_accuracy, and holdout_rank_correlation.",
        },
        "tests": {},
    }

    _log("[1/5] Ablation of Intent")
    results["tests"]["ablation_of_intent"] = ablation_of_intent_test(
        model=model,
        baseline=baseline,
        trajectories=trajectories,
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        reward_threshold=reward_threshold,
        cost_threshold=cost_threshold,
        device=args.device,
        predict_batch_size=int(args.predict_batch_size),
        min_set_b_size=int(args.min_set_b_size),
    )
    print(json.dumps(results["tests"]["ablation_of_intent"], indent=2))

    _log("[2/5] Saliency Mapping")
    results["tests"]["saliency_mapping"] = saliency_mapping_test(
        model=model,
        baseline=baseline,
        trajectories=trajectories,
        seq_len=seq_len,
        reward_threshold=reward_threshold,
        cost_threshold=cost_threshold,
        corruption_window=int(args.corruption_window),
        output_dir=out_dir,
        device=args.device,
    )
    print(json.dumps(results["tests"]["saliency_mapping"], indent=2))

    _log("[3/5] Zero-Shot Generalization")
    zero_shot_trajectories: List[Dict[str, Any]] = list(holdout_trajectories)
    results["zero_shot_dataset"] = holdout_path
    results["tests"]["zero_shot_generalization"] = zero_shot_generalization_test(
        model=model,
        baseline=baseline,
        zero_shot_trajectories=zero_shot_trajectories,
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        reward_threshold=reward_threshold,
        cost_threshold=cost_threshold,
        device=args.device,
        predict_batch_size=int(args.predict_batch_size),
        success_accuracy=0.85,
    )
    print(json.dumps(results["tests"]["zero_shot_generalization"], indent=2))

    _log("[3b] Holdout Classifier Accuracy")
    if len(zero_shot_trajectories) > 0:
        results["tests"]["holdout_classifier_accuracy"] = holdout_classifier_accuracy_test(
            model=model,
            baseline=baseline,
            holdout_trajectories=zero_shot_trajectories,
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            reward_threshold=reward_threshold,
            cost_threshold=cost_threshold,
            preferred_source_values=preferred_source_values,
            holdout_labels=holdout_labels,
            device=args.device,
            predict_batch_size=int(args.predict_batch_size),
        )
    else:
        results["tests"]["holdout_classifier_accuracy"] = {
            "success": False,
            "skipped": True,
            "reason": "Holdout set unavailable or empty",
        }
    print(json.dumps(results["tests"]["holdout_classifier_accuracy"], indent=2))

    # Additional harder holdout tests: near-threshold (hard negatives/positives) and corrupted inputs
    if len(zero_shot_trajectories) > 0:
        _log("[3b-1] Construct near-threshold hard holdout (delta=%.2f size=%d)" % (args.hard_delta, int(args.hard_holdout_size)))
        hard_holdout = construct_near_threshold_holdout(
            zero_shot_trajectories,
            reward_threshold,
            cost_threshold,
            delta=float(args.hard_delta),
            max_samples=int(args.hard_holdout_size),
            seed=int(args.seed),
        )
        results["tests"]["holdout_hard_near_threshold"] = {
            "constructed_size": int(len(hard_holdout)),
            "summary": _trajectory_summary(hard_holdout),
        }
        if len(hard_holdout) > 0:
            results["tests"]["holdout_hard_near_threshold"]["metrics"] = holdout_robust_metrics_test(
                model=model,
                baseline=baseline,
                holdout_trajectories=hard_holdout,
                state_dim=state_dim,
                action_dim=action_dim,
                seq_len=seq_len,
                reward_threshold=reward_threshold,
                cost_threshold=cost_threshold,
                preferred_source_values=preferred_source_values,
                holdout_labels=None,
                device=args.device,
                predict_batch_size=int(args.predict_batch_size),
                compute_robust=False,
            )
        print(json.dumps(results["tests"]["holdout_hard_near_threshold"], indent=2))

        _log("[3b-2] Corrupted holdout robustness (noise_sigma=%.3f mask_fraction=%.3f)" % (args.noise_sigma, args.mask_fraction))
        results["tests"]["holdout_corrupted"] = holdout_robust_metrics_test(
            model=model,
            baseline=baseline,
            holdout_trajectories=zero_shot_trajectories,
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            reward_threshold=reward_threshold,
            cost_threshold=cost_threshold,
            preferred_source_values=preferred_source_values,
            holdout_labels=holdout_labels,
            device=args.device,
            predict_batch_size=int(args.predict_batch_size),
            compute_robust=True,
            noise_sigma=float(args.noise_sigma),
            mask_fraction=float(args.mask_fraction),
            seed=int(args.seed),
        )
        print(json.dumps(results["tests"]["holdout_corrupted"], indent=2))

    _log("[3c] Holdout Rank Correlation")
    if len(zero_shot_trajectories) > 0:
        results["tests"]["holdout_rank_correlation"] = rank_correlation_test(
            model=model,
            baseline=baseline,
            trajectories=zero_shot_trajectories,
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            utility_cost_weight=float(args.utility_cost_weight),
            max_samples=int(args.rank_samples),
            device=args.device,
            predict_batch_size=int(args.predict_batch_size),
        )
    else:
        results["tests"]["holdout_rank_correlation"] = {
            "success": False,
            "skipped": True,
            "reason": "Holdout set unavailable or empty",
        }
    print(json.dumps(results["tests"]["holdout_rank_correlation"], indent=2))

    _log("[4/5] Rank Correlation")
    results["tests"]["rank_correlation"] = rank_correlation_test(
        model=model,
        baseline=baseline,
        trajectories=trajectories,
        state_dim=state_dim,
        action_dim=action_dim,
        seq_len=seq_len,
        utility_cost_weight=float(args.utility_cost_weight),
        max_samples=int(args.rank_samples),
        device=args.device,
        predict_batch_size=int(args.predict_batch_size),
    )
    print(json.dumps(results["tests"]["rank_correlation"], indent=2))

    _log("[5/5] Value Function Consistency")
    results["tests"]["value_function_consistency"] = value_function_consistency_test(
        model=model,
        baseline=baseline,
        trajectories=trajectories,
        seq_len=seq_len,
        utility_cost_weight=float(args.utility_cost_weight),
        max_checks=int(args.bellman_checks),
        neighbor_pool_limit=int(args.neighbor_pool_limit),
        device=args.device,
    )
    print(json.dumps(results["tests"]["value_function_consistency"], indent=2))

    summary_path = os.path.join(out_dir, "combined_eval_summary_push.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    _log("Saved summary: %s" % summary_path)
    _log("Done.")


if __name__ == "__main__":
    main()
