import argparse
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pickle5 as pickle
except ImportError:
    import pickle


def pick(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)


def as_1d(arr: Any) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized

    # Prefer caller cwd for relative paths, then try repo/rudder conventions.
    candidates = [
        os.path.abspath(normalized),
        os.path.abspath(os.path.join(repo_root(), normalized)),
        os.path.abspath(os.path.join(repo_root(), "rudder", normalized)),
        os.path.abspath(os.path.join(repo_root(), "rudder", "dataset", normalized)),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # For output paths that don't exist yet, keep cwd-relative behavior first.
    return candidates[0]


def load_payload(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    resolved = resolve_path(path)
    _log("Loading dataset: %s" % resolved)
    with open(resolved, "rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "trajectories" not in payload:
        raise ValueError("Expected dataset dict with key 'trajectories'")

    trajectories = payload["trajectories"]
    if not isinstance(trajectories, list) or len(trajectories) == 0:
        raise ValueError("No trajectories found in dataset")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    _log("Loaded %d trajectories" % len(trajectories))
    return trajectories, metadata, resolved


def trajectory_reward(traj: Dict[str, Any]) -> float:
    rewards = pick(traj, ("rewards", "reward", "rews", "r", "env_rewards"))
    if rewards is None:
        raise KeyError("Trajectory missing rewards")
    return float(np.sum(as_1d(rewards).astype(np.float32)))


def trajectory_cost(traj: Dict[str, Any]) -> float:
    costs = pick(traj, ("costs", "cost", "c"))
    if costs is None:
        raise KeyError("Trajectory missing costs")
    return float(np.sum(as_1d(costs).astype(np.float32)))


def trajectory_label(traj: Dict[str, Any], reward_threshold: float, cost_threshold: float) -> int:
    reward_sum = trajectory_reward(traj)
    cost_sum = trajectory_cost(traj)
    return 1 if (reward_sum > reward_threshold and cost_sum < cost_threshold) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Split combined cost/reward dataset into train and 100/100 holdout")
    parser.add_argument("--input", default="rudder/dataset/combined_cost_reward_balanced_1800.pkl")
    parser.add_argument("--train_output", default="rudder/dataset/combined_cost_reward_balanced_1800_train.pkl")
    parser.add_argument("--holdout_output", default="rudder/dataset/combined_cost_reward_holdout_200.pkl")
    parser.add_argument("--reward_threshold", type=float, default=15.0)
    parser.add_argument("--cost_threshold", type=float, default=25.0)
    parser.add_argument("--num_preferred", type=int, default=100)
    parser.add_argument("--num_nonpreferred", type=int, default=100)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _log("Starting holdout split")
    trajectories, metadata, resolved_input = load_payload(args.input)
    rng = np.random.RandomState(args.seed)

    preferred_idx: List[int] = []
    nonpreferred_idx: List[int] = []
    labels = np.zeros((len(trajectories),), dtype=np.int32)
    chunk_size = int(max(1, args.chunk_size))

    for i, traj in enumerate(trajectories):
        label = trajectory_label(traj, args.reward_threshold, args.cost_threshold)
        labels[i] = int(label)
        if label == 1:
            preferred_idx.append(i)
        else:
            nonpreferred_idx.append(i)

        processed = i + 1
        if processed % chunk_size == 0 or processed == len(trajectories):
            _log("Labeled trajectories: %d/%d" % (processed, len(trajectories)))

    _log("Loaded: %s" % resolved_input)
    _log("Total trajectories: %d" % len(trajectories))
    _log("Preferred pool: %d" % len(preferred_idx))
    _log("Non-preferred pool: %d" % len(nonpreferred_idx))

    if len(preferred_idx) < args.num_preferred or len(nonpreferred_idx) < args.num_nonpreferred:
        raise ValueError(
            "Not enough trajectories for the requested holdout. Need preferred=%d nonpreferred=%d, got preferred=%d nonpreferred=%d"
            % (args.num_preferred, args.num_nonpreferred, len(preferred_idx), len(nonpreferred_idx))
        )

    chosen_pref_local = rng.choice(len(preferred_idx), size=args.num_preferred, replace=False)
    chosen_nonpref_local = rng.choice(len(nonpreferred_idx), size=args.num_nonpreferred, replace=False)

    chosen_pref_global = [preferred_idx[int(i)] for i in chosen_pref_local]
    chosen_nonpref_global = [nonpreferred_idx[int(i)] for i in chosen_nonpref_local]
    holdout_indices = np.asarray(chosen_pref_global + chosen_nonpref_global, dtype=np.int64)

    holdout = [trajectories[int(i)] for i in holdout_indices.tolist()]
    rng.shuffle(holdout)
    _log("Sampled holdout trajectories: %d" % len(holdout))

    holdout_set = set(int(i) for i in holdout_indices.tolist())
    remaining = []
    for i, traj in enumerate(trajectories):
        if i not in holdout_set:
            remaining.append(traj)
        processed = i + 1
        if processed % chunk_size == 0 or processed == len(trajectories):
            _log("Built remaining set: %d/%d" % (processed, len(trajectories)))
    _log("Built remaining train trajectories: %d" % len(remaining))

    if len(remaining) + len(holdout) != len(trajectories):
        raise RuntimeError("Holdout removal produced an unexpected trajectory count")

    holdout_labels = labels[holdout_indices]
    remaining_mask = np.ones((len(trajectories),), dtype=bool)
    remaining_mask[holdout_indices] = False
    remaining_labels = labels[remaining_mask]

    holdout_payload = {
        "trajectories": holdout,
        "metadata": {
            "source": resolved_input,
            "reward_threshold": float(args.reward_threshold),
            "cost_threshold": float(args.cost_threshold),
            "num_preferred": int(args.num_preferred),
            "num_nonpreferred": int(args.num_nonpreferred),
            "num_total": int(len(holdout)),
            "preferred_count": int((holdout_labels == 1).sum()),
            "nonpreferred_count": int((holdout_labels == 0).sum()),
            "seed": int(args.seed),
        },
    }

    train_payload = {
        "trajectories": remaining,
        "metadata": dict(metadata, **{
            "source": resolved_input,
            "excluded_holdout": os.path.basename(resolve_path(args.holdout_output)),
            "reward_threshold": float(args.reward_threshold),
            "cost_threshold": float(args.cost_threshold),
            "remaining_total": int(len(remaining)),
            "preferred_count": int((remaining_labels == 1).sum()),
            "nonpreferred_count": int((remaining_labels == 0).sum()),
            "seed": int(args.seed),
        }),
    }

    train_path = resolve_path(args.train_output)
    holdout_path = resolve_path(args.holdout_output)
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(holdout_path), exist_ok=True)

    with open(holdout_path, "wb") as f:
        pickle.dump(holdout_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log("Saved holdout to: %s" % holdout_path)
    with open(train_path, "wb") as f:
        pickle.dump(train_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    _log("Saved reduced train set to: %s" % train_path)

    _log(
        "Holdout preferred/nonpreferred: %d/%d"
        % (int((holdout_labels == 1).sum()), int((holdout_labels == 0).sum()))
    )
    _log(
        "Remaining preferred/nonpreferred: %d/%d"
        % (int((remaining_labels == 1).sum()), int((remaining_labels == 0).sum()))
    )
    _log("Done")


if __name__ == "__main__":
    main()
