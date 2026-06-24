#!/usr/bin/env python
"""
Create deterministic holdout trajectories for reinforce RUDDER from two SafeDICE input files.

This reproduces the training-time data construction exactly:
1) Load HR/LC and LR/HC pickles.
2) Segment by done flags.
3) Apply the same reward/cost threshold filtering.
4) Combine in the same order used during training: LR/HC first, then HR/LC.
5) Build labels the same way and call RudderTrainer._resolve_split_indices(...) with
   the same seed/val_split to obtain val_idx.
6) Save holdout payload containing val trajectories and val labels.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
    import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
    import pickle


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAINER_DIR = os.path.abspath(os.path.join(REPO_ROOT, "rudder", "trainer"))
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

from rudder_train import RudderTrainer  # pylint: disable=import-error


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print("[%s] %s" % (ts, msg), flush=True)


def _resolve_input_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)

    candidates: List[str] = []
    if os.path.isabs(normalized):
        candidates.append(normalized)
    else:
        candidates.append(os.path.abspath(normalized))
        candidates.append(os.path.abspath(os.path.join(REPO_ROOT, normalized)))
        candidates.append(os.path.abspath(os.path.join(REPO_ROOT, "rudder", normalized)))
        candidates.append(os.path.abspath(os.path.join(REPO_ROOT, "rudder", "dataset", normalized)))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError("Input dataset not found: %s. Tried: %s" % (path, ", ".join(candidates)))


def _resolve_output_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(REPO_ROOT, normalized))


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
            "Unrecognized pickle format in %s. Expected keys: observations/actions/rewards/costs/dones"
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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    hr_lc_trajs, hr_lc_resolved = _load_safedice_trajectories(hr_lc_path, source_label=1)
    lr_hc_trajs, lr_hc_resolved = _load_safedice_trajectories(lr_hc_path, source_label=0)

    def _filter(trajs: List[Dict[str, Any]], label: int) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for traj in trajs:
            cr = float(np.sum(traj["rewards"]))
            cc = float(np.sum(traj["costs"]))
            if label == 1:
                if cr > high_reward_threshold and cc < cost_threshold:
                    kept.append(traj)
            else:
                if cr < low_reward_threshold and cc > cost_threshold:
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
    return combined, metadata


def _scalar_label(value: Any) -> float:
    arr = np.asarray(value).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


def _build_labels(
    trajectories: List[Dict[str, Any]],
    low_reward_threshold: float,
    high_reward_threshold: float,
    cost_threshold: float,
    preferred_source_values: List[int],
) -> torch.Tensor:
    y = np.zeros((len(trajectories), 1), dtype=np.float32)

    for i, traj in enumerate(trajectories):
        rewards = np.asarray(traj.get("rewards", []), dtype=np.float32).reshape(-1)
        costs = np.asarray(traj.get("costs", []), dtype=np.float32).reshape(-1)
        if len(rewards) == 0 or len(costs) == 0:
            raise ValueError("Trajectory %d missing rewards/costs" % i)

        c_rew = float(np.sum(rewards))
        c_cost = float(np.sum(costs))

        label_value = None
        if "preference_label" in traj:
            val = _scalar_label(traj["preference_label"])
            if not np.isnan(val):
                label_value = 1.0 if val > 0.5 else 0.0

        if label_value is None and "source" in traj:
            src = int(round(_scalar_label(traj["source"])))
            label_value = 1.0 if src in preferred_source_values else 0.0

        if label_value is None:
            if c_rew > high_reward_threshold and c_cost < cost_threshold:
                label_value = 1.0
            elif c_rew < low_reward_threshold and c_cost > cost_threshold:
                label_value = 0.0
            else:
                label_value = 0.0

        y[i, 0] = float(label_value)

    return torch.from_numpy(y)


def _infer_state_action_dims(trajectories: List[Dict[str, Any]]) -> Tuple[int, int]:
    first = trajectories[0]
    state_dim = int(np.asarray(first["states"]).shape[-1])
    action_dim = int(np.asarray(first["actions"]).shape[-1])
    return state_dim, action_dim


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic reinforce-RUDDER holdout (_push)")
    parser.add_argument(
        "--hr_lc_dataset_path",
        type=str,
        default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle",
        help="Preferred HR/LC SafeDICE pickle",
    )
    parser.add_argument(
        "--lr_hc_dataset_path",
        type=str,
        default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0.pickle",
        help="Non-preferred LR/HC SafeDICE pickle",
    )
    parser.add_argument("--low_reward_threshold", type=float, default=2.5)
    parser.add_argument("--high_reward_threshold", type=float, default=5.0)
    parser.add_argument("--cost_threshold", type=float, default=25.0)
    parser.add_argument("--preferred_source_values", type=str, default="1")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--holdout_output_path",
        type=str,
        default="rudder/dataset/reinforce_rudder_holdout_push.pkl",
        help="Where to save the deterministic validation holdout",
    )
    parser.add_argument(
        "--train_output_path",
        type=str,
        default="rudder/dataset/reinforce_rudder_train_split_push.pkl",
        help="Optional training-side split output for traceability",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    preferred_source_values: List[int] = []
    for token in str(args.preferred_source_values).split(","):
        token = token.strip()
        if token:
            preferred_source_values.append(int(token))

    _log("Loading and filtering HR/LC + LR/HC datasets in training order")
    trajectories, load_meta = _load_combined_safedice(
        hr_lc_path=args.hr_lc_dataset_path,
        lr_hc_path=args.lr_hc_dataset_path,
        low_reward_threshold=float(args.low_reward_threshold),
        high_reward_threshold=float(args.high_reward_threshold),
        cost_threshold=float(args.cost_threshold),
    )

    labels = _build_labels(
        trajectories=trajectories,
        low_reward_threshold=float(args.low_reward_threshold),
        high_reward_threshold=float(args.high_reward_threshold),
        cost_threshold=float(args.cost_threshold),
        preferred_source_values=preferred_source_values,
    )

    state_dim, action_dim = _infer_state_action_dims(trajectories)
    trainer = RudderTrainer(state_dim, action_dim)
    trainer.baseline = torch.mean(labels).item()

    train_idx, val_idx = trainer._resolve_split_indices(
        labels=labels,
        n=len(labels),
        val_split=float(args.val_split),
        seed=int(args.seed),
        train_idx=None,
        val_idx=None,
    )

    val_indices = np.asarray(val_idx, dtype=np.int64)
    train_indices = np.asarray(train_idx, dtype=np.int64)
    val_trajectories = [trajectories[int(i)] for i in val_indices.tolist()]
    train_trajectories = [trajectories[int(i)] for i in train_indices.tolist()]
    val_labels = labels[val_indices].detach().cpu().numpy().astype(np.float32)
    train_labels = labels[train_indices].detach().cpu().numpy().astype(np.float32)

    holdout_payload: Dict[str, Any] = {
        "trajectories": val_trajectories,
        "labels": val_labels,
        "indices": val_indices.tolist(),
        "metadata": {
            "split_type": "deterministic_val_from_training_logic",
            "seed": int(args.seed),
            "val_split": float(args.val_split),
            "baseline": float(trainer.baseline),
            "preferred_source_values": preferred_source_values,
            "num_total": int(len(trajectories)),
            "num_holdout": int(len(val_trajectories)),
            "num_train": int(len(train_trajectories)),
            "holdout_preferred": int((val_labels.reshape(-1) == 1.0).sum()),
            "holdout_nonpreferred": int((val_labels.reshape(-1) == 0.0).sum()),
            "train_preferred": int((train_labels.reshape(-1) == 1.0).sum()),
            "train_nonpreferred": int((train_labels.reshape(-1) == 0.0).sum()),
            **load_meta,
        },
    }

    train_payload: Dict[str, Any] = {
        "trajectories": train_trajectories,
        "labels": train_labels,
        "indices": train_indices.tolist(),
        "metadata": _json_safe(dict(holdout_payload["metadata"], split_partition="train")),
    }

    holdout_out = _resolve_output_path(args.holdout_output_path)
    train_out = _resolve_output_path(args.train_output_path)
    os.makedirs(os.path.dirname(holdout_out), exist_ok=True)
    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    with open(holdout_out, "wb") as f:
        pickle.dump(holdout_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(train_out, "wb") as f:
        pickle.dump(train_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    _log("Saved holdout (_push): %s" % holdout_out)
    _log("Saved train split (_push): %s" % train_out)
    _log(
        "Counts | total=%d train=%d holdout=%d" %
        (int(len(trajectories)), int(len(train_trajectories)), int(len(val_trajectories)))
    )


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""
Create a deterministic validation holdout for reinforce_rudder_train_2_files.py.

This script reproduces the exact split logic used during training:
1) Load LR/HC and HR/LC SafeDICE pickle files in the same order as training.
2) Build labels with the same logic as _build_training_tensors.
3) Call RudderTrainer._resolve_split_indices(...) with the same val_split/seed.
4) Save val_trajectories and val_labels as a holdout pickle.
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch

try:
    import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
    import pickle


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TRAINER_DIR = os.path.join(REPO_ROOT, "rudder", "trainer")
if TRAINER_DIR not in sys.path:
    sys.path.insert(0, TRAINER_DIR)

from reinforce_rudder_train_2_files import _build_training_tensors, _load_combined_safedice
from rudder_train import RudderTrainer


def _resolve_output_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(REPO_ROOT, normalized))


def _parse_int_list(csv: str) -> List[int]:
    values: List[int] = []
    for token in str(csv).split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def _attach_labels_to_trajectories(trajectories: List[Dict[str, Any]], labels: np.ndarray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, traj in enumerate(trajectories):
        item = dict(traj)
        item["preference_label"] = float(labels[i])
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic validation holdout for reinforce RUDDER (_push)")

    parser.add_argument(
        "--hr_lc_dataset_path",
        type=str,
        default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle",
        help="Path to High-Reward / Low-Cost (preferred) SafeDICE pickle",
    )
    parser.add_argument(
        "--lr_hc_dataset_path",
        type=str,
        default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0.pickle",
        help="Path to Low-Reward / High-Cost (non-preferred) SafeDICE pickle",
    )

    parser.add_argument("--low_reward_threshold", type=float, default=2.5)
    parser.add_argument("--high_reward_threshold", type=float, default=5.0)
    parser.add_argument("--cost_threshold", type=float, default=25.0)
    parser.add_argument("--preferred_source_values", type=str, default="1")

    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--holdout_output_path",
        type=str,
        default="rudder/dataset/reinforce_rudder_holdout_push.pkl",
        help="Output path for validation holdout payload",
    )
    parser.add_argument(
        "--train_output_path",
        type=str,
        default="rudder/dataset/reinforce_rudder_train_split_push.pkl",
        help="Optional output path for deterministic train split payload",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    preferred_source_values = _parse_int_list(args.preferred_source_values)

    trajectories, metadata, resolved_dataset = _load_combined_safedice(
        hr_lc_path=args.hr_lc_dataset_path,
        lr_hc_path=args.lr_hc_dataset_path,
        low_reward_threshold=float(args.low_reward_threshold),
        high_reward_threshold=float(args.high_reward_threshold),
        cost_threshold=float(args.cost_threshold),
    )

    (
        labels_t,
        state_dim,
        action_dim,
        _cumulative_rewards,
        _cumulative_costs,
        _target_len,
        _label_source_counts,
    ) = _build_training_tensors(
        trajectories=trajectories,
        low_reward_threshold=float(args.low_reward_threshold),
        high_reward_threshold=float(args.high_reward_threshold),
        cost_threshold=float(args.cost_threshold),
        seq_len=int(args.seq_len),
        preferred_source_values=preferred_source_values,
    )

    trainer = RudderTrainer(state_dim, action_dim)
    trainer.baseline = torch.mean(labels_t.float()).item()

    train_idx, val_idx = trainer._resolve_split_indices(
        labels=labels_t,
        n=len(labels_t),
        val_split=float(args.val_split),
        seed=int(args.seed),
        train_idx=None,
        val_idx=None,
    )

    labels_np = labels_t.detach().cpu().numpy().reshape(-1).astype(np.float32)

    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)

    val_trajectories_raw = [trajectories[int(i)] for i in val_idx.tolist()]
    train_trajectories_raw = [trajectories[int(i)] for i in train_idx.tolist()]
    val_labels = labels_np[val_idx]
    train_labels = labels_np[train_idx]

    val_trajectories = _attach_labels_to_trajectories(val_trajectories_raw, val_labels)
    train_trajectories = _attach_labels_to_trajectories(train_trajectories_raw, train_labels)

    holdout_payload: Dict[str, Any] = {
        "trajectories": val_trajectories,
        "labels": val_labels,
        "metadata": {
            "mode": "deterministic_val_split_from_two_safedice_inputs",
            "resolved_dataset": resolved_dataset,
            "source_metadata": metadata,
            "hr_lc_dataset_path": args.hr_lc_dataset_path,
            "lr_hc_dataset_path": args.lr_hc_dataset_path,
            "low_reward_threshold": float(args.low_reward_threshold),
            "high_reward_threshold": float(args.high_reward_threshold),
            "cost_threshold": float(args.cost_threshold),
            "preferred_source_values": preferred_source_values,
            "val_split": float(args.val_split),
            "seed": int(args.seed),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "num_total": int(len(trajectories)),
            "num_train": int(len(train_idx)),
            "num_val": int(len(val_idx)),
            "val_indices": val_idx,
            "train_indices": train_idx,
            "baseline": float(trainer.baseline),
            "label_counts": {
                "train_preferred": int((train_labels == 1.0).sum()),
                "train_nonpreferred": int((train_labels == 0.0).sum()),
                "val_preferred": int((val_labels == 1.0).sum()),
                "val_nonpreferred": int((val_labels == 0.0).sum()),
            },
        },
    }

    train_payload: Dict[str, Any] = {
        "trajectories": train_trajectories,
        "labels": train_labels,
        "metadata": holdout_payload["metadata"],
    }

    holdout_output_path = _resolve_output_path(args.holdout_output_path)
    train_output_path = _resolve_output_path(args.train_output_path)

    os.makedirs(os.path.dirname(holdout_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)

    with open(holdout_output_path, "wb") as f:
        pickle.dump(holdout_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(train_output_path, "wb") as f:
        pickle.dump(train_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved holdout (_push):", holdout_output_path)
    print("Saved train split (_push):", train_output_path)
    print("Total trajectories:", len(trajectories))
    print("Train/Val:", len(train_idx), "/", len(val_idx))
    print(
        "Val labels | preferred=%d non_preferred=%d"
        % (int((val_labels == 1.0).sum()), int((val_labels == 0.0).sum()))
    )


if __name__ == "__main__":
    main()
