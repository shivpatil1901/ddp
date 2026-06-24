#!/usr/bin/env python
"""
Inspect the action-dependent SafeDICE critic score for PointGoal1 states.

The SafeDICE checkpoint stores two learned critics:
- `critic`: a state-only value head
- `cost`: an action-conditioned head that acts like a Q function over (state, action)

This script loads the `cost` head and reports, for each state:
- the score of the deterministic policy action
- the score for the zero action
- statistics over random action probes
- the local action gradient of the score

Outputs are written to CSV and NPZ for later inspection.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
SAFEDICE_PATH = ROOT / "SafeDICE"
sys.path.insert(0, str(SAFEDICE_PATH))

try:
    from algorithms.safedice import SafeDICE as AntiDICE
except ImportError as exc:
    raise ImportError(f"Could not import SafeDICE from {SAFEDICE_PATH}: {exc}")


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


class SafeDICEQInspector:
    """Load SafeDICE and inspect the action-conditioned critic head."""

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

    def deterministic_actions(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = []
        for start in range(0, len(states), 4096):
            end = min(start + 4096, len(states))
            batch = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
            batch_actions = self.model.step(batch, deterministic=True)
            actions.append(batch_actions.numpy())
        return np.concatenate(actions, axis=0).astype(np.float32)

    def q_values(self, states: np.ndarray, actions: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if len(states) != len(actions):
            raise ValueError("states and actions must have the same length")

        value_parts = []
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            sa = np.concatenate([states[start:end], actions[start:end]], axis=1)
            sa_tensor = tf.convert_to_tensor(sa, dtype=tf.float32)
            values, _ = self.model.cost(sa_tensor)
            value_parts.append(tf.reshape(values, [-1]).numpy())
        return np.concatenate(value_parts, axis=0).reshape(-1)

    def action_gradients(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        batch_size: int = 1024,
    ) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if len(states) != len(actions):
            raise ValueError("states and actions must have the same length")

        gradient_parts = []
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            state_batch = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
            action_batch = tf.convert_to_tensor(actions[start:end], dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(action_batch)
                sa = tf.concat([state_batch, action_batch], axis=1)
                values, _ = self.model.cost(sa)
                values = tf.reshape(values, [-1])
            grads = tape.gradient(values, action_batch)
            gradient_parts.append(grads.numpy())

        return np.concatenate(gradient_parts, axis=0).astype(np.float32)


def save_csv(rows: List[Dict], output_csv: str):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: List[Dict], action_dim: int):
    print("\n" + "=" * 100)
    print("SAFE DICE ACTION-DEPENDENT Q INSPECTION")
    print("=" * 100)
    print(
        f"{'State':>7s} {'Q(actor)':>12s} {'Q(zero)':>12s} {'Q(best-rand)':>14s} "
        f"{'Q(range)':>12s} {'|grad|':>12s}"
    )
    print("-" * 100)

    for row in rows[: min(len(rows), 20)]:
        q_range = row["q_random_max"] - row["q_random_min"]
        print(
            f"{int(row['state_index']):7d} {row['q_actor']:12.4f} {row['q_zero']:12.4f} "
            f"{row['q_random_max']:14.4f} {q_range:12.4f} {row['grad_norm_actor']:12.4f}"
        )

    if rows:
        ranked = sorted(rows, key=lambda r: r["grad_norm_actor"], reverse=True)[:10]
        print("\nTop states by action sensitivity (gradient norm):")
        for row in ranked:
            grad_vector = ", ".join(f"{row[f'grad_actor_{i}']:.4f}" for i in range(action_dim))
            print(
                f"  state {int(row['state_index'])}: |grad|={row['grad_norm_actor']:.4f}, "
                f"q_actor={row['q_actor']:.4f}, grads=[{grad_vector}]"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the SafeDICE action-conditioned critic on PointGoal1 states"
    )
    parser.add_argument(
        "--safedice-weights",
        type=str,
        default=(
            "/home/ed21b059/ddp/SafeDICE/weights/"
            "antidice_PointGoal1_seed0_20260423_002630_iter1000000_reinforce.pickle"
        ),
        help="Path to the SafeDICE weights pickle",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
        ),
        help="Dataset used to extract states for inspection",
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=200,
        help="Inspect only the first N states for a quick run",
    )
    parser.add_argument(
        "--num-random-actions",
        type=int,
        default=64,
        help="Number of random actions to probe per state",
    )
    parser.add_argument(
        "--random-action-scale",
        type=float,
        default=1.0,
        help="Scale for uniform random action probes in [-scale, scale]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for critic evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "value_function_analysis_combined"),
        help="Directory to save CSV and NPZ outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for action probes",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading states from dataset: {args.dataset}")
    states = load_states(args.dataset, max_states=args.max_states)
    print(f"Loaded states: {states.shape}")

    inspector = SafeDICEQInspector(args.safedice_weights)
    if states.shape[1] != inspector.state_dim:
        raise ValueError(
            f"State dim mismatch: dataset has {states.shape[1]}, SafeDICE expects {inspector.state_dim}"
        )

    print("Computing deterministic policy actions...")
    actor_actions = inspector.deterministic_actions(states)
    zero_actions = np.zeros_like(actor_actions, dtype=np.float32)

    print("Evaluating Q(state, action) for actor and zero actions...")
    q_actor = inspector.q_values(states, actor_actions, batch_size=args.batch_size)
    q_zero = inspector.q_values(states, zero_actions, batch_size=args.batch_size)

    print(f"Sampling {args.num_random_actions} random actions per state...")
    random_actions = rng.uniform(
        low=-args.random_action_scale,
        high=args.random_action_scale,
        size=(len(states), args.num_random_actions, inspector.action_dim),
    ).astype(np.float32)

    q_random_mean = np.zeros(len(states), dtype=np.float32)
    q_random_std = np.zeros(len(states), dtype=np.float32)
    q_random_min = np.zeros(len(states), dtype=np.float32)
    q_random_max = np.zeros(len(states), dtype=np.float32)
    q_random_best = np.zeros(len(states), dtype=np.float32)
    best_random_actions = np.zeros((len(states), inspector.action_dim), dtype=np.float32)

    for i in range(len(states)):
        state_block = np.repeat(states[i : i + 1], args.num_random_actions, axis=0)
        action_block = random_actions[i]
        q_block = inspector.q_values(state_block, action_block, batch_size=args.batch_size)
        q_random_mean[i] = float(np.mean(q_block))
        q_random_std[i] = float(np.std(q_block))
        q_random_min[i] = float(np.min(q_block))
        q_random_max[i] = float(np.max(q_block))
        best_idx = int(np.argmax(q_block))
        q_random_best[i] = float(q_block[best_idx])
        best_random_actions[i] = action_block[best_idx]

    print("Computing local action gradients at actor actions...")
    grad_actor = inspector.action_gradients(states, actor_actions, batch_size=min(1024, args.batch_size))
    grad_norm_actor = np.linalg.norm(grad_actor, axis=1)

    rows: List[Dict] = []
    for i in range(len(states)):
        row = {
            "state_index": i,
            "q_actor": float(q_actor[i]),
            "q_zero": float(q_zero[i]),
            "q_random_mean": float(q_random_mean[i]),
            "q_random_std": float(q_random_std[i]),
            "q_random_min": float(q_random_min[i]),
            "q_random_max": float(q_random_max[i]),
            "q_random_best": float(q_random_best[i]),
            "grad_norm_actor": float(grad_norm_actor[i]),
        }
        for dim in range(inspector.action_dim):
            row[f"actor_action_{dim}"] = float(actor_actions[i, dim])
            row[f"zero_action_{dim}"] = 0.0
            row[f"best_random_action_{dim}"] = float(best_random_actions[i, dim])
            row[f"grad_actor_{dim}"] = float(grad_actor[i, dim])
        rows.append(row)

    summarize_rows(rows, inspector.action_dim)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "safedice_pointgoal1_q_action_importance.csv")
    npz_path = os.path.join(args.output_dir, "safedice_pointgoal1_q_action_importance.npz")

    save_csv(rows, csv_path)
    np.savez(
        npz_path,
        states=states,
        actor_actions=actor_actions,
        zero_actions=zero_actions,
        q_actor=q_actor,
        q_zero=q_zero,
        q_random_mean=q_random_mean,
        q_random_std=q_random_std,
        q_random_min=q_random_min,
        q_random_max=q_random_max,
        q_random_best=q_random_best,
        best_random_actions=best_random_actions,
        grad_actor=grad_actor,
        grad_norm_actor=grad_norm_actor,
    )

    print(f"\nSaved inspection summary: {csv_path}")
    print(f"Saved inspection arrays: {npz_path}")


if __name__ == "__main__":
    main()