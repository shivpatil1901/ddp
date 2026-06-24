#!/usr/bin/env python
"""
Validate SafeDICE Q function against trajectory actions.

Load the PPO Lagrangian PointGoal1 expert trajectories and evaluate the SafeDICE
cost head (acting as Q-function) on the actual expert actions. Compare with:
- Random actions
- Zero actions
- Policy actions (deterministic)

If the Q function is a good proxy for the true Q function, expert trajectory
actions should score higher than random/zero actions on average.
"""

import argparse
import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parent.parent
SAFEDICE_PATH = ROOT / "SafeDICE"
sys.path.insert(0, str(SAFEDICE_PATH))

try:
    from algorithms.safedice import SafeDICE as AntiDICE
except ImportError as exc:
    raise ImportError(f"Could not import SafeDICE from {SAFEDICE_PATH}: {exc}")


def extract_transitions(data) -> Tuple[np.ndarray, np.ndarray]:
    """Extract flat transition data (states, actions) from dataset."""
    if isinstance(data, dict):
        # Check for flat transition format
        if "states" in data and "actions" in data:
            states = np.asarray(data["states"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            if len(states) == len(actions):
                return states, actions
        
        # Check for trajectory format
        if "trajectories" in data:
            trajectories = data["trajectories"]
            all_states = []
            all_actions = []
            for traj in trajectories:
                if not isinstance(traj, dict):
                    continue
                if "states" in traj and "actions" in traj:
                    states_traj = np.asarray(traj["states"], dtype=np.float32)
                    actions_traj = np.asarray(traj["actions"], dtype=np.float32)
                    # Ensure alignment
                    n = min(len(states_traj), len(actions_traj))
                    all_states.append(states_traj[:n])
                    all_actions.append(actions_traj[:n])
            if all_states:
                return np.concatenate(all_states), np.concatenate(all_actions)

    raise ValueError("Could not extract transition data with actions from dataset")


def load_transitions(dataset_path: str, max_transitions: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    states, actions = extract_transitions(data)
    if max_transitions and max_transitions > 0:
        states = states[:max_transitions]
        actions = actions[:max_transitions]
    return states, actions


class SafeDICEQValidator:
    """Load SafeDICE and validate Q function against trajectories."""

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


def compute_action_percentiles(
    traj_states: np.ndarray,
    traj_actions: np.ndarray,
    validator: SafeDICEQValidator,
    num_samples: int = 100,
    action_scale: float = 1.0,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (state, action) pair in trajectory, compute what percentile
    the trajectory action occupies in a distribution of random actions.
    
    Returns:
        q_traj: Q values for trajectory actions
        percentiles: percentile ranks (0-100) of trajectory actions
        q_random_mean: mean Q of random actions per state
        q_random_std: std Q of random actions per state
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    n_states = len(traj_states)
    q_traj = validator.q_values(traj_states, traj_actions, batch_size=4096)
    
    percentiles = np.zeros(n_states, dtype=np.float32)
    q_random_mean = np.zeros(n_states, dtype=np.float32)
    q_random_std = np.zeros(n_states, dtype=np.float32)
    
    print(f"Computing percentiles for {n_states} states...")
    for i in range(n_states):
        # Sample random actions for this state
        random_actions = rng.uniform(
            low=-action_scale,
            high=action_scale,
            size=(num_samples, validator.action_dim),
        ).astype(np.float32)
        
        # Evaluate Q for random actions
        state_reps = np.repeat(traj_states[i:i+1], num_samples, axis=0)
        q_random = validator.q_values(state_reps, random_actions, batch_size=4096)
        
        # Compute percentile
        traj_q = q_traj[i]
        rank = np.sum(q_random <= traj_q) / num_samples
        percentiles[i] = float(rank * 100.0)
        q_random_mean[i] = float(np.mean(q_random))
        q_random_std[i] = float(np.std(q_random))
        
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_states} states processed...")
    
    return q_traj, percentiles, q_random_mean, q_random_std


def main():
    parser = argparse.ArgumentParser(
        description="Validate SafeDICE Q function against expert trajectory actions"
    )
    parser.add_argument(
        "--safedice-weights",
        type=str,
        default=(
            "/home/ed21b059/ddp/SafeDICE/weights/"
            "antidice_PointGoal1_seed0_20260423_002630_iter1000000_reinforce.pickle"
        ),
        help="Path to SafeDICE weights pickle",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle"
        ),
        help="Dataset with expert trajectories",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="If >0, use only first N trajectories",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=0,
        help="If >0, use only first N transitions total",
    )
    parser.add_argument(
        "--num-random-samples",
        type=int,
        default=100,
        help="Number of random actions to sample per state for percentile computation",
    )
    parser.add_argument(
        "--random-action-scale",
        type=float,
        default=1.0,
        help="Scale for random action sampling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "value_function_analysis_combined"),
        help="Output directory for CSV and NPZ",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Loading transitions from: {args.dataset}")
    traj_states, traj_actions = load_transitions(args.dataset, max_transitions=args.max_transitions)
    print(f"Loaded {len(traj_states)} transitions")
    print(f"State dim: {traj_states.shape[1]}, Action dim: {traj_actions.shape[1]}")

    # Create trajectory indices (mark all as from same batch for now)
    traj_indices = np.zeros(len(traj_states), dtype=int)

    # Load validator
    validator = SafeDICEQValidator(args.safedice_weights)
    if traj_states.shape[1] != validator.state_dim:
        raise ValueError(
            f"State dim mismatch: dataset has {traj_states.shape[1]}, "
            f"SafeDICE expects {validator.state_dim}"
        )

    # Get Q values for trajectory actions
    print("Evaluating Q values for trajectory actions...")
    q_traj = validator.q_values(traj_states, traj_actions, batch_size=4096)

    # Get policy deterministic actions and their Q values
    print("Computing policy deterministic actions...")
    policy_actions = validator.deterministic_actions(traj_states)
    q_policy = validator.q_values(traj_states, policy_actions, batch_size=4096)

    # Get zero action Q values
    print("Evaluating zero actions...")
    zero_actions = np.zeros_like(traj_actions, dtype=np.float32)
    q_zero = validator.q_values(traj_states, zero_actions, batch_size=4096)

    # Compute percentiles
    print("Computing action percentiles against random sampling...")
    q_traj_pct, percentiles, q_random_mean, q_random_std = compute_action_percentiles(
        traj_states,
        traj_actions,
        validator,
        num_samples=args.num_random_samples,
        action_scale=args.random_action_scale,
        rng=rng,
    )

    # Build report rows
    rows: List[Dict] = []
    for i in range(len(traj_states)):
        row = {
            "transition_index": i,
            "trajectory_index": int(traj_indices[i]),
            "q_trajectory_action": float(q_traj[i]),
            "q_policy_action": float(q_policy[i]),
            "q_zero_action": float(q_zero[i]),
            "q_random_mean": float(q_random_mean[i]),
            "q_random_std": float(q_random_std[i]),
            "action_percentile": float(percentiles[i]),
            "policy_action_beats_trajectory": 1 if q_policy[i] > q_traj[i] else 0,
            "zero_action_beats_trajectory": 1 if q_zero[i] > q_traj[i] else 0,
        }
        for dim in range(validator.action_dim):
            row[f"trajectory_action_{dim}"] = float(traj_actions[i, dim])
            row[f"policy_action_{dim}"] = float(policy_actions[i, dim])
        rows.append(row)

    # Print summary statistics
    print("\n" + "=" * 100)
    print("TRAJECTORY Q VALIDATION SUMMARY")
    print("=" * 100)
    
    print(f"\nTotal transitions: {len(rows)}")
    print(f"\nQ-value statistics (trajectory actions):")
    print(f"  Mean:   {np.mean(q_traj):10.4f}")
    print(f"  Std:    {np.std(q_traj):10.4f}")
    print(f"  Min:    {np.min(q_traj):10.4f}")
    print(f"  Max:    {np.max(q_traj):10.4f}")
    
    print(f"\nAction percentile (trajectory vs random):")
    print(f"  Mean:   {np.mean(percentiles):10.2f}%")
    print(f"  Std:    {np.std(percentiles):10.2f}%")
    print(f"  Min:    {np.min(percentiles):10.2f}%")
    print(f"  Max:    {np.max(percentiles):10.2f}%")
    
    pct_policy_better = 100.0 * np.mean([r["policy_action_beats_trajectory"] for r in rows])
    pct_zero_better = 100.0 * np.mean([r["zero_action_beats_trajectory"] for r in rows])
    
    print(f"\nComparison to alternatives:")
    print(f"  % transitions where policy action beats trajectory action: {pct_policy_better:.2f}%")
    print(f"  % transitions where zero action beats trajectory action:   {pct_zero_better:.2f}%")
    
    print(f"\nInterpretation:")
    if np.mean(percentiles) > 50:
        print("  ✓ Trajectory actions are above median Q-value (good proxy)")
    else:
        print("  ✗ Trajectory actions are below median Q-value (poor proxy)")
    
    if pct_policy_better < 20:
        print("  ✓ Trajectory actions are generally better than policy actions (expert data)")
    else:
        print("  ⚠ Policy actions often beat trajectory actions (policy may be suboptimal)")
    
    if pct_zero_better < 5:
        print("  ✓ Zero action rarely beats trajectory action")
    else:
        print("  ⚠ Zero action sometimes beats trajectory action")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "q_validation_trajectory_actions.csv")
    npz_path = os.path.join(args.output_dir, "q_validation_trajectory_actions.npz")

    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        npz_path,
        states=traj_states,
        trajectory_actions=traj_actions,
        policy_actions=policy_actions,
        zero_actions=zero_actions,
        q_trajectory=q_traj,
        q_policy=q_policy,
        q_zero=q_zero,
        percentiles=percentiles,
        trajectory_indices=traj_indices,
    )

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved NPZ: {npz_path}")

    # Print sample of high and low percentile actions
    print("\n" + "=" * 100)
    print("SAMPLE TRAJECTORIES")
    print("=" * 100)
    
    sorted_idx = np.argsort(percentiles)
    print("\nLowest percentile actions (trajectory action worse than random):")
    for idx in sorted_idx[:5]:
        row = rows[idx]
        print(f"  Idx {int(row['transition_index'])}: percentile={row['action_percentile']:.1f}%, "
              f"Q_traj={row['q_trajectory_action']:.4f}, Q_rand_mean={row['q_random_mean']:.4f}")
    
    print("\nHighest percentile actions (trajectory action better than random):")
    for idx in sorted_idx[-5:]:
        row = rows[idx]
        print(f"  Idx {int(row['transition_index'])}: percentile={row['action_percentile']:.1f}%, "
              f"Q_traj={row['q_trajectory_action']:.4f}, Q_rand_mean={row['q_random_mean']:.4f}")


if __name__ == "__main__":
    main()
