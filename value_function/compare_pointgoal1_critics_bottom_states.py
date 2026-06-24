#!/usr/bin/env python
"""
Compare PointGoal1 critic values between:
1) PPO-Lagrangian reward/cost critics from a fixed TF checkpoint epoch
2) SafeDICE critic from a pickle checkpoint

The script loads one shared state set and compares:
- raw value distributions
- rank-order agreement across states
- bottom-k overlap for a few score definitions
"""

import argparse
import csv
import json
import os
import pickle5 as pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
SAFEDICE_PATH = ROOT / "SafeDICE"
sys.path.insert(0, str(SAFEDICE_PATH))

try:
    from algorithms.safedice import SafeDICE as AntiDICE
except ImportError as exc:
    raise ImportError(f"Could not import SafeDICE from {SAFEDICE_PATH}: {exc}")


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_percent_list(text: str) -> List[float]:
    percentages = parse_float_list(text)
    for p in percentages:
        if p <= 0 or p >= 100:
            raise ValueError(f"Percentages must be in (0, 100), got {p}")
    return percentages


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - mean) / std


def rankdata_ordinal(values: np.ndarray) -> np.ndarray:
    """Return 0-based ordinal ranks for a 1D array."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic without SciPy."""
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    combined = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, combined, side="right") / max(1, len(a))
    cdf_b = np.searchsorted(b, combined, side="right") / max(1, len(b))
    return float(np.max(np.abs(cdf_a - cdf_b))) if len(combined) else 0.0


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    """1D Wasserstein distance for equal-weight samples."""
    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(np.mean(np.abs(a[:n] - b[:n])))


def infer_ckpt_prefix(ckpt_data_path: str) -> str:
    suffix = ".data-00000-of-00001"
    if ckpt_data_path.endswith(suffix):
        return ckpt_data_path[:-len(suffix)]
    return ckpt_data_path


def extract_states_from_dataset(data) -> np.ndarray:
    if isinstance(data, dict):
        if "states" in data:
            return np.asarray(data["states"], dtype=np.float32)
        if "observations" in data:
            return np.asarray(data["observations"], dtype=np.float32)
        if "trajectories" in data:
            trajectories = data["trajectories"]
            parts = []
            for traj in trajectories:
                if not isinstance(traj, dict):
                    continue
                if "states" in traj:
                    parts.append(np.asarray(traj["states"], dtype=np.float32))
                elif "observations" in traj:
                    parts.append(np.asarray(traj["observations"], dtype=np.float32))
            if parts:
                return np.concatenate(parts, axis=0)

    if isinstance(data, (list, tuple)):
        parts = []
        for traj in data:
            if not isinstance(traj, dict):
                continue
            if "states" in traj:
                parts.append(np.asarray(traj["states"], dtype=np.float32))
            elif "observations" in traj:
                parts.append(np.asarray(traj["observations"], dtype=np.float32))
        if parts:
            return np.concatenate(parts, axis=0)

    raise ValueError("Could not extract state observations from dataset")


def load_states(dataset_path: str, max_states: int = 0) -> np.ndarray:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    states = extract_states_from_dataset(data)
    if max_states and max_states > 0:
        states = states[:max_states]
    return states.astype(np.float32)


class PPORewardCostCriticLoader:
    """Loads PPO reward critic (vf) and cost critic (vc) from checkpoint weights."""

    def __init__(self, ckpt_prefix: str, obs_dim: int, hidden_sizes: Tuple[int, int] = (256, 256)):
        self.ckpt_prefix = ckpt_prefix
        self.obs_dim = int(obs_dim)
        self.hidden_sizes = hidden_sizes
        self.vf_weights = None
        self.vc_weights = None
        self._load_weights()

    def _load_dense_weights(self, scope: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        dense_blocks = ["dense", "dense_1", "dense_2"]
        layers = []
        for block in dense_blocks:
            kernel_name = f"{scope}/{block}/kernel"
            bias_name = f"{scope}/{block}/bias"
            w = tf.train.load_variable(self.ckpt_prefix, kernel_name)
            b = tf.train.load_variable(self.ckpt_prefix, bias_name)
            layers.append((w, b))
        return layers

    def _load_weights(self):

        index_path = self.ckpt_prefix + ".index"
        meta_path = self.ckpt_prefix + ".meta"
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Checkpoint prefix not valid: {self.ckpt_prefix} (missing .index/.meta)"
            )

        print(f"Loading PPO critic weights from checkpoint prefix: {self.ckpt_prefix}")
        self.vf_weights = self._load_dense_weights("vf")
        self.vc_weights = self._load_dense_weights("vc")

        in_dim = int(self.vf_weights[0][0].shape[0])
        if in_dim != self.obs_dim:
            raise ValueError(f"Checkpoint obs dim={in_dim}, but states have dim={self.obs_dim}")

    @staticmethod
    def _forward_mlp(x: np.ndarray, weights: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        h = x
        for i, (w, b) in enumerate(weights):
            h = h @ w + b
            if i < len(weights) - 1:
                h = np.tanh(h)
        return h.reshape(-1)

    def predict(self, states: np.ndarray, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=np.float32)
        reward_values = []
        cost_values = []

        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            batch = states[start:end]
            v_batch = self._forward_mlp(batch, self.vf_weights)
            vc_batch = self._forward_mlp(batch, self.vc_weights)
            reward_values.append(v_batch)
            cost_values.append(vc_batch)

        return (
            np.concatenate(reward_values, axis=0).reshape(-1),
            np.concatenate(cost_values, axis=0).reshape(-1),
        )

    def close(self):
        return


class SafeDICECriticLoader:
    """Load SafeDICE critic and evaluate values on states."""

    def __init__(self, weights_path: str, config_dict: Dict = None):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"SafeDICE weights not found: {weights_path}")

        self.weights_path = weights_path
        self.model = None
        self.state_dim = None
        self.action_dim = None

        self.config = {
            "hidden_size": 256,
            "critic_lr": 1e-4,
            "actor_lr": 1e-5,
            "gamma": 0.99,
            "alpha": 0.0,
            "grad_reg_coeffs": (10, 1e-6),
            "use_last_layer_bias_cost": False,
            "use_last_layer_bias_critic": False,
            "kernel_initializer": "he_normal",
        }
        if config_dict is not None:
            self.config.update(config_dict)
        else:
            try:
                sys.path.insert(0, str(SAFEDICE_PATH / "config"))
                from safedice_config import hparams
                if hparams:
                    self.config.update(hparams[0])
            except Exception:
                pass

        self._load()

    def _load(self):
        print(f"Loading SafeDICE weights: {self.weights_path}")
        with open(self.weights_path, "rb") as f:
            data = pickle.load(f)

        training_state = data["training_state"]
        critic_params = training_state.get("critic_params", [])
        cost_params = training_state.get("cost_params", [])
        if not critic_params or not cost_params:
            raise ValueError("SafeDICE checkpoint missing critic or cost params")

        state_dim = None
        cost_input_dim = None
        for name, param in critic_params:
            if "mlp/dense/kernel" in name or "mlp/dense" in name:
                state_dim = int(param.shape[0])
                break
        for name, param in cost_params:
            if "mlp/dense/kernel" in name or "mlp/dense" in name:
                cost_input_dim = int(param.shape[0])
                break

        if state_dim is None or cost_input_dim is None:
            raise ValueError("Could not infer state/action dims from SafeDICE weights")

        action_dim = int(cost_input_dim - state_dim)
        if action_dim <= 0:
            raise ValueError(f"Invalid inferred action_dim={action_dim}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        print(f"SafeDICE inferred dims: state_dim={self.state_dim}, action_dim={self.action_dim}")

        self.model = AntiDICE(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mixture_actor=False,
            is_discrete_action=False,
            config=self.config,
        )
        self.model.set_training_state(training_state)

    def get_state_values(self, states: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        values = []
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            batch = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
            v_batch, _ = self.model.critic(batch)
            values.append(v_batch.numpy())
        return np.concatenate(values, axis=0).reshape(-1)


def bottom_k_mask(values: np.ndarray, percentage: float) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    n = len(values)
    k = max(1, int(np.floor((percentage / 100.0) * n)))
    idx = np.argpartition(values, k - 1)[:k]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def top_k_mask(values: np.ndarray, percentage: float) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    n = len(values)
    k = max(1, int(np.floor((percentage / 100.0) * n)))
    idx = np.argpartition(values, n - k)[n - k :]
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask


def compare_distributions(reference_values: np.ndarray, candidate_values: np.ndarray) -> Dict[str, float]:
    reference_values = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    candidate_values = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    n = min(len(reference_values), len(candidate_values))
    ref = reference_values[:n]
    cand = candidate_values[:n]

    ref_ranks = rankdata_ordinal(ref)
    cand_ranks = rankdata_ordinal(cand)

    pearson_raw = float(np.corrcoef(ref, cand)[0, 1]) if n > 1 else 0.0
    pearson_ranks = float(np.corrcoef(ref_ranks, cand_ranks)[0, 1]) if n > 1 else 0.0

    return {
        "n_states": int(n),
        "ref_mean": float(np.mean(ref)),
        "ref_std": float(np.std(ref)),
        "ref_min": float(np.min(ref)),
        "ref_max": float(np.max(ref)),
        "cand_mean": float(np.mean(cand)),
        "cand_std": float(np.std(cand)),
        "cand_min": float(np.min(cand)),
        "cand_max": float(np.max(cand)),
        "pearson_raw": pearson_raw,
        "pearson_ranks": pearson_ranks,
        "ks_statistic": ks_statistic(ref, cand),
        "wasserstein_1d": wasserstein_1d(ref, cand),
    }


def compute_overlap_rows(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    candidate_name: str,
    percentages: Iterable[float],
) -> List[Dict]:
    rows = []
    n = len(reference_values)

    for pct in percentages:
        ref_mask = bottom_k_mask(reference_values, pct)
        cand_mask = bottom_k_mask(candidate_values, pct)

        ref_k = int(np.sum(ref_mask))
        cand_k = int(np.sum(cand_mask))
        overlap = int(np.sum(ref_mask & cand_mask))
        union = int(np.sum(ref_mask | cand_mask))

        rows.append(
            {
                "candidate": candidate_name,
                "percentage": pct,
                "n_states": n,
                "k_ref": ref_k,
                "k_candidate": cand_k,
                "overlap_count": overlap,
                "overlap_pct_of_ref": 100.0 * overlap / max(1, ref_k),
                "overlap_pct_of_candidate": 100.0 * overlap / max(1, cand_k),
                "jaccard_pct": 100.0 * overlap / max(1, union),
            }
        )

    return rows


def compute_rank_rows(
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    candidate_name: str,
    percentages: Iterable[float],
) -> List[Dict]:
    rows = []
    reference_values = np.asarray(reference_values).reshape(-1)
    candidate_values = np.asarray(candidate_values).reshape(-1)
    n = min(len(reference_values), len(candidate_values))
    ref = reference_values[:n]
    cand = candidate_values[:n]

    ref_order = np.argsort(ref)
    cand_order = np.argsort(cand)

    ref_rank = np.empty(n, dtype=np.int64)
    cand_rank = np.empty(n, dtype=np.int64)
    ref_rank[ref_order] = np.arange(n)
    cand_rank[cand_order] = np.arange(n)

    for pct in percentages:
        k = max(1, int(np.floor((pct / 100.0) * n)))
        ref_bottom = set(ref_order[:k].tolist())
        cand_bottom = set(cand_order[:k].tolist())
        ref_top = set(ref_order[-k:].tolist())
        cand_top = set(cand_order[-k:].tolist())

        bottom_overlap = int(len(ref_bottom & cand_bottom))
        top_overlap = int(len(ref_top & cand_top))

        rows.append(
            {
                "candidate": candidate_name,
                "percentage": pct,
                "n_states": n,
                "bottom_overlap_count": bottom_overlap,
                "bottom_overlap_frac": 100.0 * bottom_overlap / max(1, k),
                "top_overlap_count": top_overlap,
                "top_overlap_frac": 100.0 * top_overlap / max(1, k),
                "rank_pearson": float(np.corrcoef(ref_rank, cand_rank)[0, 1]) if n > 1 else 0.0,
            }
        )

    return rows


def print_overlap_summary(rows: List[Dict]):
    print("\n" + "=" * 90)
    print("BOTTOM-K OVERLAP VS SAFEDICE (LOWER VALUE = WORSE STATE)")
    print("=" * 90)
    print(
        f"{'Candidate':34s} {'%':>6s} {'Overlap':>10s} {'Ref%':>10s} {'Cand%':>10s} {'Jaccard%':>10s}"
    )
    print("-" * 90)

    for row in rows:
        overlap = f"{row['overlap_count']}/{row['k_ref']}"
        print(
            f"{row['candidate'][:34]:34s} {row['percentage']:6.1f} "
            f"{overlap:>10s} {row['overlap_pct_of_ref']:10.2f} {row['overlap_pct_of_candidate']:10.2f} "
            f"{row['jaccard_pct']:10.2f}"
        )


def print_distribution_summary(rows: List[Dict]):
    print("\n" + "=" * 90)
    print("DISTRIBUTION AND RANKING SUMMARY")
    print("=" * 90)
    print(
        f"{'Candidate':34s} {'Pearson':>10s} {'RankPearson':>12s} {'KS':>8s} {'Wass':>10s} {'MeanDiff':>10s}"
    )
    print("-" * 90)
    for row in rows:
        mean_diff = abs(row["ref_mean"] - row["cand_mean"])
        print(
            f"{row['candidate'][:34]:34s} {row['pearson_raw']:10.4f} {row['pearson_ranks']:12.4f} "
            f"{row['ks_statistic']:8.4f} {row['wasserstein_1d']:10.4f} {mean_diff:10.4f}"
        )


def save_csv(rows: List[Dict], output_csv: str, fieldnames: List[str] = None):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(payload: Dict, output_json: str):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PPO reward/cost critics with SafeDICE critic on value distributions and state rankings"
    )
    parser.add_argument(
        "--ppo-ckpt-data-path",
        type=str,
        default=(
            "/home/ed21b059/ddp/data_new/ppo_lagrangian_PointPush1/ppo_lagrangian_PointPush1_s0/critic_checkpoints_point_push1_s0/critic_epoch_000280.ckpt.data-00000-of-00001"
        ),
        help="Path to .ckpt.data-00000-of-00001 file for PPO critic epoch",
    )
    parser.add_argument(
        "--ppo-ckpt-prefix",
        type=str,
        default="",
        help="Optional explicit checkpoint prefix. If set, overrides --ppo-ckpt-data-path",
    )
    parser.add_argument(
        "--safedice-weights",
        type=str,
        default=(
            "/home/ed21b059/ddp/SafeDICE/weights/antidice_PointPush1_seed0_20260527_080012_iter1000000_reinforce.pickle"
        ),
        help="Path to SafeDICE weights pickle",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle"
        ),
        help="PointGoal1 dataset used to extract the shared state set",
    )
    parser.add_argument(
        "--percentages",
        type=str,
        default="5,10,20,25,30,35,40",
        help="Comma-separated bottom percentages to evaluate",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.1,0.25,0.5,1.0,2.0",
        help="Comma-separated lambda values for score = reward - lambda * cost",
    )
    parser.add_argument(
        "--z-lambdas",
        type=str,
        default="0.02,0.05,0.1,0.15,0.2,0.25,0.3",
        help="Comma-separated lambda values for score = z(reward) - lambda * z(cost)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for critic inference",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=0,
        help="If >0, evaluate only first N states for faster debugging",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "value_function_analysis_combined"),
        help="Directory to save summary outputs",
    )
    args = parser.parse_args()

    percentages = parse_percent_list(args.percentages)
    lambda_values = parse_float_list(args.lambdas)
    z_lambda_values = parse_float_list(args.z_lambdas)
    ckpt_prefix = args.ppo_ckpt_prefix.strip() or infer_ckpt_prefix(args.ppo_ckpt_data_path)

    print(f"Loading states from dataset: {args.dataset}")
    states = load_states(args.dataset, max_states=args.max_states)
    print(f"Loaded shared state set: {states.shape[0]} states x {states.shape[1]} dims")

    ppo_loader = PPORewardCostCriticLoader(ckpt_prefix=ckpt_prefix, obs_dim=states.shape[1])
    try:
        print("Running PPO reward/cost critics...")
        ppo_reward, ppo_cost = ppo_loader.predict(states, batch_size=args.batch_size)
    finally:
        ppo_loader.close()

    safedice_loader = SafeDICECriticLoader(args.safedice_weights)
    if states.shape[1] != safedice_loader.state_dim:
        raise ValueError(
            f"State dim mismatch: dataset has {states.shape[1]}, "
            f"SafeDICE expects {safedice_loader.state_dim}"
        )

    print("Running SafeDICE critic...")
    safedice_values = safedice_loader.get_state_values(states, batch_size=args.batch_size)

    print("Comparing raw value distributions and overall rank agreement...")
    distribution_rows: List[Dict] = []
    for candidate_name, values in [
        ("ppo_reward", ppo_reward),
        ("ppo_cost", ppo_cost),
        ("ppo_reward_minus_0.1_cost", ppo_reward - 0.1 * ppo_cost),
        ("ppo_reward_minus_1.0_cost", ppo_reward - 1.0 * ppo_cost),
        ("ppo_reward_minus_2.0_cost", ppo_reward - 2.0 * ppo_cost),
        ("ppo_zreward_minus_0.1_zcost", zscore(ppo_reward) - 0.1 * zscore(ppo_cost)),
        ("ppo_zreward_minus_0.2_zcost", zscore(ppo_reward) - 0.2 * zscore(ppo_cost)),
        ("ppo_zreward_minus_0.3_zcost", zscore(ppo_reward) - 0.3 * zscore(ppo_cost)),
    ]:
        summary = compare_distributions(safedice_values, values)
        summary["candidate"] = candidate_name
        distribution_rows.append(summary)

    ppo_reward_z = zscore(ppo_reward)
    ppo_cost_z = zscore(ppo_cost)

    candidate_scores: List[Tuple[str, np.ndarray]] = [
        ("ppo_reward", ppo_reward),
        ("ppo_cost", ppo_cost),
    ]

    for lam in lambda_values:
        score = ppo_reward - lam * ppo_cost
        candidate_scores.append((f"ppo_reward_minus_{lam:g}_cost", score))

    for lam in z_lambda_values:
        z_score = ppo_reward_z - lam * ppo_cost_z
        candidate_scores.append((f"ppo_zreward_minus_{lam:g}_zcost", z_score))

    overlap_rows: List[Dict] = []
    rank_rows: List[Dict] = []
    for name, values in candidate_scores:
        overlap_rows.extend(
            compute_overlap_rows(
                reference_values=safedice_values,
                candidate_values=values,
                candidate_name=name,
                percentages=percentages,
            )
        )
        rank_rows.extend(
            compute_rank_rows(
                reference_values=safedice_values,
                candidate_values=values,
                candidate_name=name,
                percentages=percentages,
            )
        )

    print_distribution_summary(distribution_rows)
    print_overlap_summary(overlap_rows)
    print("\n" + "=" * 90)
    print("RANK CORRELATION BY BOTTOM-K/TOP-K CUTS")
    print("=" * 90)
    print(f"{'Candidate':34s} {'%':>6s} {'RankPearson':>12s} {'Bottom%':>10s} {'Top%':>10s}")
    print("-" * 90)
    for row in rank_rows:
        print(
            f"{row['candidate'][:34]:34s} {row['percentage']:6.1f} {row['rank_pearson']:12.4f} "
            f"{row['bottom_overlap_frac']:10.2f} {row['top_overlap_frac']:10.2f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "pointgoal1_bottom_top_overlap_vs_safedice.csv")
    distribution_csv_path = os.path.join(args.output_dir, "pointgoal1_distribution_summary_vs_safedice.csv")
    rank_csv_path = os.path.join(args.output_dir, "pointgoal1_rank_summary_vs_safedice.csv")
    npz_path = os.path.join(args.output_dir, "pointgoal1_critic_values_vs_safedice.npz")
    json_path = os.path.join(args.output_dir, "pointgoal1_critic_comparison_summary.json")

    save_csv(
        overlap_rows,
        csv_path,
        fieldnames=[
            "candidate",
            "percentage",
            "n_states",
            "k_ref",
            "k_candidate",
            "overlap_count",
            "overlap_pct_of_ref",
            "overlap_pct_of_candidate",
            "jaccard_pct",
        ],
    )
    save_csv(
        distribution_rows,
        distribution_csv_path,
        fieldnames=[
            "candidate",
            "n_states",
            "ref_mean",
            "ref_std",
            "ref_min",
            "ref_max",
            "cand_mean",
            "cand_std",
            "cand_min",
            "cand_max",
            "pearson_raw",
            "pearson_ranks",
            "ks_statistic",
            "wasserstein_1d",
        ],
    )
    save_csv(
        rank_rows,
        rank_csv_path,
        fieldnames=[
            "candidate",
            "percentage",
            "n_states",
            "bottom_overlap_count",
            "bottom_overlap_frac",
            "top_overlap_count",
            "top_overlap_frac",
            "rank_pearson",
        ],
    )
    np.savez(
        npz_path,
        states=states,
        ppo_reward=ppo_reward,
        ppo_cost=ppo_cost,
        ppo_reward_z=ppo_reward_z,
        ppo_cost_z=ppo_cost_z,
        safedice_value=safedice_values,
    )
    save_json(
        {
            "dataset": args.dataset,
            "ppo_checkpoint_prefix": ckpt_prefix,
            "safedice_weights": args.safedice_weights,
            "state_shape": list(states.shape),
            "distribution_rows": distribution_rows,
            "overlap_rows": overlap_rows,
            "rank_rows": rank_rows,
        },
        json_path,
    )

    print(f"\nSaved overlap summary: {csv_path}")
    print(f"Saved distribution summary: {distribution_csv_path}")
    print(f"Saved rank summary: {rank_csv_path}")
    print(f"Saved value arrays: {npz_path}")
    print(f"Saved JSON summary: {json_path}")


if __name__ == "__main__":
    main()
