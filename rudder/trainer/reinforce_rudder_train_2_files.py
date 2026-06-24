import argparse
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

try:
	import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
	import pickle

from rudder_train import RudderTrainer


def _repo_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)
	if os.path.isabs(normalized):
		return normalized
	return os.path.abspath(os.path.join(_repo_root(), normalized))


def _resolve_input_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)

	candidates = []
	if os.path.isabs(normalized):
		candidates.append(normalized)
	else:
		candidates.append(os.path.abspath(normalized))
		candidates.append(os.path.abspath(os.path.join(_repo_root(), normalized)))
		candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", normalized)))
		candidates.append(os.path.abspath(os.path.join(_repo_root(), "rudder", "dataset", normalized)))

	for candidate in candidates:
		if os.path.isfile(candidate):
			return candidate

	raise FileNotFoundError("Input dataset not found: %s. Tried: %s" % (path, ", ".join(candidates)))


# ---------------------------------------------------------------------------
# SafeDICE raw pickle loader
# ---------------------------------------------------------------------------

def _load_safedice_pickle(path: str) -> Dict[str, np.ndarray]:
	"""Load a SafeDICE-format pickle file.

	Expected top-level keys (flat arrays, all timesteps concatenated):
	  observations, actions, rewards, costs, dones
	"""
	resolved = _resolve_input_path(path)
	with open(resolved, "rb") as f:
		data = pickle.load(f)

	# Accept both raw SafeDICE format and pre-converted format.
	if "observations" in data:
		obs     = np.asarray(data["observations"], dtype=np.float32)
		actions = np.asarray(data["actions"],      dtype=np.float32)
		rewards = np.asarray(data["rewards"],      dtype=np.float32).reshape(-1)
		costs   = np.asarray(data["costs"],        dtype=np.float32).reshape(-1)
		dones   = np.asarray(data["dones"],        dtype=np.float32).reshape(-1)
	elif "states" in data and "dones" in data:
		# Pre-converted variant
		obs     = np.asarray(data["states"],   dtype=np.float32)
		actions = np.asarray(data["actions"],  dtype=np.float32)
		rewards = np.asarray(data["rewards"],  dtype=np.float32).reshape(-1)
		costs   = np.asarray(data["costs"],    dtype=np.float32).reshape(-1)
		dones   = np.asarray(data["dones"],    dtype=np.float32).reshape(-1)
	else:
		raise KeyError(
			f"Unrecognised pickle format in {path}. "
			"Expected keys: 'observations'/'actions'/'rewards'/'costs'/'dones'."
		)

	return {"obs": obs, "actions": actions, "rewards": rewards, "costs": costs, "dones": dones}


def _segment_into_trajectories(
	flat: Dict[str, np.ndarray],
	source_label: int,
) -> List[Dict[str, Any]]:
	"""Split flat (concatenated) arrays into per-episode trajectory dicts.

	Episode boundaries are determined by `dones == 1`.  The last timestep of
	each episode is the one where done==1; the episode runs from the previous
	boundary (exclusive) to the current one (inclusive).
	"""
	obs     = flat["obs"]
	actions = flat["actions"]
	rewards = flat["rewards"]
	costs   = flat["costs"]
	dones   = flat["dones"]

	n = len(obs)
	episode_end_indices = list(np.where(dones == 1)[0])

	# Handle the case where the last episode does not end with done==1.
	if not episode_end_indices or episode_end_indices[-1] != n - 1:
		episode_end_indices.append(n - 1)

	trajectories: List[Dict[str, Any]] = []
	start = 0
	for end in episode_end_indices:
		end_excl = end + 1  # Python slice end (exclusive)
		if end_excl <= start:
			start = end_excl
		states = np.asarray(obs[start:end_excl], dtype=np.float32)
		action_segment = np.asarray(actions[start:end_excl], dtype=np.float32)
		reward_segment = np.asarray(rewards[start:end_excl], dtype=np.float32)
		cost_segment = np.asarray(costs[start:end_excl], dtype=np.float32)
		traj = {
			"states": states,
			"actions": action_segment,
			"rewards": reward_segment,
			"costs": cost_segment,
			"source": int(source_label),
		}
		trajectories.append(traj)
		start = end_excl

	return trajectories


def _load_safedice_trajectories(path: str, source_label: int) -> Tuple[List[Dict[str, Any]], str]:
	resolved = _resolve_input_path(path)
	flat = _load_safedice_pickle(resolved)
	trajectories = _segment_into_trajectories(flat, source_label=source_label)
	return trajectories, resolved


def _load_combined_safedice(
	hr_lc_path: str,
	lr_hc_path: str,
	low_reward_threshold: float,
	high_reward_threshold: float,
	cost_threshold: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	"""Load HR/LC and LR/HC SafeDICE datasets, filter by thresholds, combine.

	HR/LC trajectories (source=1): cumulative_reward > high_reward_threshold
	                                AND cumulative_cost < cost_threshold
	LR/HC trajectories (source=0): cumulative_reward < low_reward_threshold
	                                AND cumulative_cost > cost_threshold

	Trajectories that do not satisfy either criterion are discarded so that the
	preference signal is unambiguous.
	"""
	hr_lc_trajs, hr_lc_resolved = _load_safedice_trajectories(hr_lc_path, source_label=1)
	lr_hc_trajs, lr_hc_resolved = _load_safedice_trajectories(lr_hc_path, source_label=0)

	print(f"\nRaw episode counts before filtering:")
	print(f"  HR/LC ({hr_lc_resolved}): {len(hr_lc_trajs)} episodes")
	print(f"  LR/HC ({lr_hc_resolved}): {len(lr_hc_trajs)} episodes")

	def _threshold_stats(trajs: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
		reward_sums = np.asarray([float(np.sum(traj["rewards"])) for traj in trajs], dtype=np.float32)
		cost_sums = np.asarray([float(np.sum(traj["costs"])) for traj in trajs], dtype=np.float32)
		return reward_sums, cost_sums

	hr_reward_sums, hr_cost_sums = _threshold_stats(hr_lc_trajs)
	lr_reward_sums, lr_cost_sums = _threshold_stats(lr_hc_trajs)
	hr_mask = (hr_reward_sums > high_reward_threshold) & (hr_cost_sums < cost_threshold)
	lr_mask = (lr_reward_sums < low_reward_threshold) & (lr_cost_sums > cost_threshold)

	print(
		"  HR/LC reward min/mean/max = %.3f / %.3f / %.3f | cost min/mean/max = %.3f / %.3f / %.3f"
		% (
			float(hr_reward_sums.min()),
			float(hr_reward_sums.mean()),
			float(hr_reward_sums.max()),
			float(hr_cost_sums.min()),
			float(hr_cost_sums.mean()),
			float(hr_cost_sums.max()),
		)
	)
	print(
		"  LR/HC reward min/mean/max = %.3f / %.3f / %.3f | cost min/mean/max = %.3f / %.3f / %.3f"
		% (
			float(lr_reward_sums.min()),
			float(lr_reward_sums.mean()),
			float(lr_reward_sums.max()),
			float(lr_cost_sums.min()),
			float(lr_cost_sums.mean()),
			float(lr_cost_sums.max()),
		)
	)
	print(
		"  Threshold matches -> HR/LC: %d / %d | LR/HC: %d / %d"
		% (int(hr_mask.sum()), len(hr_mask), int(lr_mask.sum()), len(lr_mask))
	)

	hr_lc_filtered = [traj for traj, keep in zip(hr_lc_trajs, hr_mask) if keep]
	lr_hc_filtered = [traj for traj, keep in zip(lr_hc_trajs, lr_mask) if keep]

	print(f"\nAfter threshold filtering (R>{high_reward_threshold} & C<{cost_threshold} for HR/LC | "
	      f"R<{low_reward_threshold} & C>{cost_threshold} for LR/HC):")
	print(f"  HR/LC kept: {len(hr_lc_filtered)} / {len(hr_lc_trajs)}")
	print(f"  LR/HC kept: {len(lr_hc_filtered)} / {len(lr_hc_trajs)}")

	if len(hr_lc_filtered) == 0:
		raise ValueError(
			f"No HR/LC trajectories passed the filter "
			f"(R>{high_reward_threshold} & C<{cost_threshold}). "
			"Check your thresholds or dataset."
		)
	if len(lr_hc_filtered) == 0:
		raise ValueError(
			f"No LR/HC trajectories passed the filter "
			f"(R<{low_reward_threshold} & C>{cost_threshold}). "
			"Check your thresholds or dataset."
		)

	combined = lr_hc_filtered + hr_lc_filtered  # non-preferred first, then preferred

	metadata: Dict[str, Any] = {
		"mode": "safedice_separate_datasets",
		"hr_lc_path": hr_lc_resolved,
		"lr_hc_path": lr_hc_resolved,
		"hr_lc_raw_count": len(hr_lc_trajs),
		"lr_hc_raw_count": len(lr_hc_trajs),
		"hr_lc_filtered_count": len(hr_lc_filtered),
		"lr_hc_filtered_count": len(lr_hc_filtered),
		"low_reward_threshold": low_reward_threshold,
		"high_reward_threshold": high_reward_threshold,
		"cost_threshold": cost_threshold,
	}
	resolved_str = f"{hr_lc_resolved} + {lr_hc_resolved}"
	return combined, metadata, resolved_str


def _build_feature_tensor(
	trajectories: List[Dict[str, Any]],
	target_len: int,
	state_dim: int,
	action_dim: int,
) -> torch.Tensor:
	feature_dim = int(state_dim + action_dim)
	x = np.zeros((len(trajectories), int(target_len), feature_dim), dtype=np.float32)

	for i, traj in enumerate(trajectories):
		states = np.asarray(traj["states"], dtype=np.float32)
		actions = np.asarray(traj["actions"], dtype=np.float32)
		n = int(min(len(states), len(actions)))
		if n <= 0:
			raise ValueError(f"Trajectory {i} is empty")

		sa = np.concatenate([states[:n], actions[:n]], axis=-1)
		use_n = min(n, int(target_len))
		x[i, :use_n] = sa[:use_n]

	return torch.from_numpy(x)


# ---------------------------------------------------------------------------
# Legacy combined-pkl loader (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _load_payload(dataset_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	resolved = _resolve_input_path(dataset_path)
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

	return trajectories, metadata, resolved


def _load_trajectory_payload(dataset_path: str, source_label: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	trajectories, metadata, resolved = _load_payload(dataset_path)
	annotated_trajectories = []
	for traj in trajectories:
		item = dict(traj)
		item["source"] = source_label
		annotated_trajectories.append(item)
	return annotated_trajectories, metadata, resolved


def _load_combined_payload(
	dataset_path: str,
	low_reward_dataset_path: str,
	high_reward_dataset_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	combined_trajectories: List[Dict[str, Any]] = []
	combined_metadata: Dict[str, Any] = {
		"mode": "separate_datasets",
		"low_reward_dataset_path": low_reward_dataset_path,
		"high_reward_dataset_path": high_reward_dataset_path,
	}
	resolved_paths: List[str] = []

	low_trajectories, low_metadata, low_resolved = _load_trajectory_payload(low_reward_dataset_path, source_label=0)
	high_trajectories, high_metadata, high_resolved = _load_trajectory_payload(high_reward_dataset_path, source_label=1)
	combined_trajectories.extend(low_trajectories)
	combined_trajectories.extend(high_trajectories)
	combined_metadata["low_reward_metadata"] = low_metadata
	combined_metadata["high_reward_metadata"] = high_metadata
	resolved_paths.extend([low_resolved, high_resolved])

	combined_metadata["requested_dataset_path"] = dataset_path
	combined_metadata["resolved_dataset_paths"] = resolved_paths
	return combined_trajectories, combined_metadata, " + ".join(resolved_paths)


# ---------------------------------------------------------------------------
# Label / tensor building (unchanged from original)
# ---------------------------------------------------------------------------

def _scalar_label(value: Any) -> float:
	arr = np.asarray(value).reshape(-1)
	if arr.size == 0:
		return float("nan")
	return float(arr[0])


def _build_training_tensors(
	trajectories: List[Dict[str, Any]],
	low_reward_threshold: float,
	high_reward_threshold: float,
	cost_threshold: float,
	seq_len: int,
	preferred_source_values: List[int],
) -> Tuple[torch.Tensor, int, int, np.ndarray, np.ndarray, int, Dict[str, int]]:
	first = trajectories[0]
	if "states" not in first or "actions" not in first:
		raise KeyError("Each trajectory must contain states and actions")

	state_dim = int(np.asarray(first["states"]).shape[-1])
	action_dim = int(np.asarray(first["actions"]).shape[-1])

	lengths = [min(len(t["states"]), len(t["actions"])) for t in trajectories]
	target_len = int(max(lengths)) if seq_len <= 0 else int(seq_len)

	y = np.zeros((len(trajectories), 1), dtype=np.float32)
	cumulative_rewards = np.zeros((len(trajectories),), dtype=np.float32)
	cumulative_costs = np.zeros((len(trajectories),), dtype=np.float32)

	label_source_counts = {
		"preference_label": 0,
		"source": 0,
		"threshold_logic": 0,
	}

	for i, traj in enumerate(trajectories):
		states = np.asarray(traj["states"], dtype=np.float32)
		actions = np.asarray(traj["actions"], dtype=np.float32)
		rewards = np.asarray(traj.get("rewards", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
		costs = np.asarray(traj.get("costs", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)

		n = int(min(len(states), len(actions), len(rewards), len(costs)))
		if n <= 0:
			raise ValueError(f"Trajectory {i} is empty")

		c_rew = float(np.sum(rewards[:n]))
		c_cost = float(np.sum(costs[:n]))
		cumulative_rewards[i] = c_rew
		cumulative_costs[i] = c_cost

		label_value = None
		if "preference_label" in traj:
			val = _scalar_label(traj["preference_label"])
			if not np.isnan(val):
				label_value = 1.0 if val > 0.5 else 0.0
				label_source_counts["preference_label"] += 1

		if label_value is None and "source" in traj:
			src = int(round(_scalar_label(traj["source"])))
			label_value = 1.0 if src in preferred_source_values else 0.0
			label_source_counts["source"] += 1

		if label_value is None:
			if c_rew > high_reward_threshold and c_cost < cost_threshold:
				label_value = 1.0
			elif c_rew < low_reward_threshold and c_cost > cost_threshold:
				label_value = 0.0
			else:
				label_value = 0.0
			label_source_counts["threshold_logic"] += 1

		y[i, 0] = label_value

	return (
		torch.from_numpy(y),
		state_dim,
		action_dim,
		cumulative_rewards,
		cumulative_costs,
		target_len,
		label_source_counts,
	)


def _build_stratified_folds(labels: torch.Tensor, num_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
	labels_np = labels.detach().cpu().numpy().reshape(-1)
	if num_folds < 2:
		raise ValueError("num_folds must be at least 2")
	if len(labels_np) < num_folds:
		raise ValueError("Need at least as many trajectories as folds")

	rng = np.random.RandomState(seed)
	pos_idx = np.where(labels_np == 1.0)[0]
	neg_idx = np.where(labels_np == 0.0)[0]
	rng.shuffle(pos_idx)
	rng.shuffle(neg_idx)

	pos_folds = np.array_split(pos_idx, num_folds)
	neg_folds = np.array_split(neg_idx, num_folds)

	folds: List[Tuple[np.ndarray, np.ndarray]] = []
	all_idx = np.arange(len(labels_np), dtype=np.int64)
	for fold_idx in range(num_folds):
		val_idx = np.concatenate([pos_folds[fold_idx], neg_folds[fold_idx]]).astype(np.int64)
		if len(val_idx) == 0:
			raise ValueError("One of the CV folds is empty; reduce num_folds or check label balance")
		train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False).astype(np.int64)
		folds.append((train_idx, val_idx))

	return folds


def _run_five_fold_cross_validation(
	features: torch.Tensor,
	labels: torch.Tensor,
	state_dim: int,
	action_dim: int,
	hidden_dim: int,
	dropout: float,
	lr: float,
	scheduler_step_size: int,
	scheduler_gamma: float,
	epochs: int,
	batch_size: int,
	val_split: float,
	seed: int,
	num_folds: int = 5,
) -> Dict[str, Any]:
	labels_np = labels.detach().cpu().numpy().reshape(-1)
	pos_count = int(np.sum(labels_np == 1.0))
	neg_count = int(np.sum(labels_np == 0.0))
	effective_num_folds = min(int(num_folds), len(labels_np), pos_count, neg_count)
	if effective_num_folds < 2:
		print(
			"Skipping cross validation: need at least 2 trajectories and at least 1 trajectory per class, got "
			f"preferred={pos_count}, non_preferred={neg_count}."
		)
		return {
			"num_folds": 0,
			"folds": [],
			"mean_val_accuracy": float("nan"),
			"std_val_accuracy": float("nan"),
			"min_val_accuracy": float("nan"),
			"max_val_accuracy": float("nan"),
			"skipped": True,
			"skip_reason": (
				f"Insufficient data for stratified CV: preferred={pos_count}, non_preferred={neg_count}, "
				f"requested_folds={num_folds}"
			),
		}

	if effective_num_folds != num_folds:
		print(
			f"Reducing cross-validation folds from {num_folds} to {effective_num_folds} to match the filtered dataset size."
		)

	folds = _build_stratified_folds(labels, num_folds=effective_num_folds, seed=seed)
	fold_stats: List[Dict[str, Any]] = []

	for fold_index, (train_idx, val_idx) in enumerate(folds, start=1):
		print("\n--- Fold %d/%d ---" % (fold_index, effective_num_folds))
		fold_trainer = RudderTrainer(
			state_dim,
			action_dim,
			hidden_dim=hidden_dim,
			dropout=dropout,
			lr=lr,
			scheduler_step_size=scheduler_step_size,
			scheduler_gamma=scheduler_gamma,
		)
		fold_train_stats = fold_trainer.train(
			trajectories=features,
			labels=labels,
			epochs=epochs,
			batch_size=batch_size,
			val_split=val_split,
			seed=seed,
			train_idx=train_idx,
			val_idx=val_idx,
		)
		fold_accuracy = float(fold_train_stats["best_val_acc"])
		fold_loss = float(fold_train_stats["best_val_loss"])
		print("Fold %d accuracy: %.4f" % (fold_index, fold_accuracy))
		fold_stats.append(
			{
				"fold": int(fold_index),
				"train_size": int(len(train_idx)),
				"val_size": int(len(val_idx)),
				"val_accuracy": fold_accuracy,
				"val_loss": fold_loss,
				"best_epoch": int(fold_train_stats["best_epoch"]),
			}
		)

	accuracies = np.asarray([item["val_accuracy"] for item in fold_stats], dtype=np.float32)
	return {
		"num_folds": int(effective_num_folds),
		"folds": fold_stats,
		"mean_val_accuracy": float(accuracies.mean()),
		"std_val_accuracy": float(accuracies.std()),
		"min_val_accuracy": float(accuracies.min()),
		"max_val_accuracy": float(accuracies.max()),
	}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
	parser = argparse.ArgumentParser(description="Train RUDDER on SafeDICE preference-signal trajectories")

	# --- SafeDICE direct dataset paths (new, primary mode) ---
	parser.add_argument(
		"--hr_lc_dataset_path",
		type=str,
		default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle",
		help="Path to the High-Reward / Low-Cost (preferred) SafeDICE pickle",
	)
	parser.add_argument(
		"--lr_hc_dataset_path",
		type=str,
		default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_s0.pickle",
		help="Path to the Low-Reward / High-Cost (non-preferred) SafeDICE pickle",
	)

	# --- Legacy combined-pkl path (kept for backward compatibility) ---
	parser.add_argument(
		"--dataset_path",
		type=str,
		default="rudder/dataset/combined_cost_reward_balanced_1800.pkl",
		help="[Legacy] Path to combined trajectory dataset (ignored when hr_lc/lr_hc paths are set)",
	)
	parser.add_argument(
		"--low_reward_dataset_path",
		type=str,
		default="",
		help="[Legacy] Optional path to the LR/HC trajectory dataset in pre-segmented format",
	)
	parser.add_argument(
		"--high_reward_dataset_path",
		type=str,
		default="",
		help="[Legacy] Optional path to the HR/LC trajectory dataset in pre-segmented format",
	)

	# --- Thresholds ---
	parser.add_argument("--low_reward_threshold",  type=float, default=2.5,  help="Cumulative reward below which a trajectory is non-preferred (LR/HC)")
	parser.add_argument("--high_reward_threshold", type=float, default=5.0,  help="Cumulative reward above which a trajectory is preferred (HR/LC)")
	parser.add_argument("--cost_threshold",        type=float, default=25.0, help="Cumulative cost boundary separating LC from HC trajectories")

	parser.add_argument(
		"--preferred_source_values",
		type=str,
		default="1",
		help="Comma-separated source values interpreted as preferred labels (used for legacy mode)",
	)
	parser.add_argument("--seq_len",            type=int,   default=0,    help="Sequence length (0 = max length in data)")
	parser.add_argument("--hidden_dim",         type=int,   default=64)
	parser.add_argument("--dropout",            type=float, default=0.2)
	parser.add_argument("--lr",                 type=float, default=1e-3)
	parser.add_argument("--scheduler_step_size",type=int,   default=10)
	parser.add_argument("--scheduler_gamma",    type=float, default=0.5)
	parser.add_argument("--epochs",             type=int,   default=30)
	parser.add_argument("--batch_size",         type=int,   default=32)
	parser.add_argument("--val_split",          type=float, default=0.1)
	parser.add_argument("--num_folds",          type=int,   default=5)
	parser.add_argument("--num_workers",        type=int,   default=0)
	parser.add_argument("--pin_memory",         action="store_true")
	parser.add_argument("--seed",               type=int,   default=0)
	parser.add_argument("--save_path",          type=str,   default="rudder/models/reinforce_rudder_combined.pt")
	args = parser.parse_args()

	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	preferred_source_values = []
	for token in str(args.preferred_source_values).split(","):
		token = token.strip()
		if token:
			preferred_source_values.append(int(token))

	# ------------------------------------------------------------------
	# Dataset loading: SafeDICE mode takes priority when hr/lr paths are
	# set; fall back to legacy mode otherwise.
	# ------------------------------------------------------------------
	use_safedice = bool(str(args.hr_lc_dataset_path).strip()) and bool(str(args.lr_hc_dataset_path).strip())
	use_legacy_separate = bool(str(args.low_reward_dataset_path).strip()) or bool(str(args.high_reward_dataset_path).strip())

	if use_safedice:
		print("=== SafeDICE mode: loading HR/LC and LR/HC pickle files directly ===")
		trajectories, metadata, resolved_dataset = _load_combined_safedice(
			hr_lc_path=args.hr_lc_dataset_path,
			lr_hc_path=args.lr_hc_dataset_path,
			low_reward_threshold=float(args.low_reward_threshold),
			high_reward_threshold=float(args.high_reward_threshold),
			cost_threshold=float(args.cost_threshold),
		)
	elif use_legacy_separate:
		if not str(args.low_reward_dataset_path).strip() or not str(args.high_reward_dataset_path).strip():
			raise ValueError("When using separate legacy datasets, both --low_reward_dataset_path and --high_reward_dataset_path are required")
		trajectories, metadata, resolved_dataset = _load_combined_payload(
			args.dataset_path,
			args.low_reward_dataset_path,
			args.high_reward_dataset_path,
		)
	else:
		trajectories, metadata, resolved_dataset = _load_payload(args.dataset_path)

	(
		labels_t,
		state_dim,
		action_dim,
		cumulative_rewards,
		cumulative_costs,
		target_len,
		label_source_counts,
	) = _build_training_tensors(
		trajectories=trajectories,
		low_reward_threshold=float(args.low_reward_threshold),
		high_reward_threshold=float(args.high_reward_threshold),
		cost_threshold=float(args.cost_threshold),
		seq_len=int(args.seq_len),
		preferred_source_values=preferred_source_values,
	)

	n_pref    = int((labels_t.numpy().reshape(-1) == 1.0).sum())
	n_nonpref = int((labels_t.numpy().reshape(-1) == 0.0).sum())
	if n_pref == 0 or n_nonpref == 0:
		raise ValueError(
			"Need both classes for RUDDER preference training; got preferred=%d non_preferred=%d"
			% (n_pref, n_nonpref)
		)

	print("\nLoaded dataset:", resolved_dataset)
	print(f"Trajectories  : {len(trajectories)}")
	print(f"State dim     : {state_dim} | Action dim: {action_dim} | Seq len used: {target_len}")
	print(
		f"Reward/Cost thresholds | low_reward<{args.low_reward_threshold}, "
		f"high_reward>{args.high_reward_threshold}, cost<{args.cost_threshold}"
	)
	print(f"Label counts  | preferred(1): {n_pref} | non_preferred(0): {n_nonpref}")
	print("Label source counts:", label_source_counts)
	print(
		"Cumulative reward stats | min=%.3f avg=%.3f max=%.3f"
		% (float(cumulative_rewards.min()), float(cumulative_rewards.mean()), float(cumulative_rewards.max()))
	)
	print(
		"Cumulative cost stats   | min=%.3f avg=%.3f max=%.3f"
		% (float(cumulative_costs.min()), float(cumulative_costs.mean()), float(cumulative_costs.max()))
	)

	feature_tensor = _build_feature_tensor(
		trajectories=trajectories,
		target_len=target_len,
		state_dim=state_dim,
		action_dim=action_dim,
	)
	print("\n=== 5-Fold Cross Validation ===")
	cv_stats = _run_five_fold_cross_validation(
		features=feature_tensor,
		labels=labels_t,
		state_dim=state_dim,
		action_dim=action_dim,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		lr=args.lr,
		scheduler_step_size=args.scheduler_step_size,
		scheduler_gamma=args.scheduler_gamma,
		epochs=args.epochs,
		batch_size=args.batch_size,
		val_split=args.val_split,
		seed=args.seed,
		num_folds=int(args.num_folds),
	)
	print(
		"CV accuracy | mean=%.4f std=%.4f min=%.4f max=%.4f"
		% (
			float(cv_stats["mean_val_accuracy"]),
			float(cv_stats["std_val_accuracy"]),
			float(cv_stats["min_val_accuracy"]),
			float(cv_stats["max_val_accuracy"]),
		)
	)

	print("\n--- Final Full-Data Fit ---")
	trainer = RudderTrainer(
		state_dim,
		action_dim,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		lr=args.lr,
		scheduler_step_size=args.scheduler_step_size,
		scheduler_gamma=args.scheduler_gamma,
	)
	train_stats = trainer.train(
		trajectories=feature_tensor,
		labels=labels_t,
		epochs=args.epochs,
		batch_size=args.batch_size,
		val_split=args.val_split,
		seed=args.seed,
	)

	save_path = _resolve_path(args.save_path)
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	torch.save(
		{
			"model_state_dict":      trainer.model.state_dict(),
			"best_state_dict":       trainer.best_state_dict,
			"baseline":              float(trainer.baseline),
			"state_dim":             int(state_dim),
			"action_dim":            int(action_dim),
			"seq_len":               int(target_len),
			"low_reward_threshold":  float(args.low_reward_threshold),
			"high_reward_threshold": float(args.high_reward_threshold),
			"cost_threshold":        float(args.cost_threshold),
			"hr_lc_dataset_path":    args.hr_lc_dataset_path,
			"lr_hc_dataset_path":    args.lr_hc_dataset_path,
			# legacy keys preserved for compatibility
			"low_reward_dataset_path":  args.low_reward_dataset_path,
			"high_reward_dataset_path": args.high_reward_dataset_path,
			"preferred_source_values":  preferred_source_values,
			"label_source_counts":      label_source_counts,
			"dataset_path":             resolved_dataset,
			"dataset_metadata":         metadata,
			"cv_stats":                 cv_stats,
			"train_stats":              train_stats,
			"val_split":                float(args.val_split),
			"num_folds":                int(args.num_folds),
			"dropout":                  float(args.dropout),
			"optimizer":                "Adam",
			"scheduler":                "StepLR",
			"scheduler_step_size":      int(args.scheduler_step_size),
			"scheduler_gamma":          float(args.scheduler_gamma),
			"label_convention": (
				"SafeDICE mode — 1 if source==1 (HR/LC, R>high_reward_threshold & C<cost_threshold); "
				"0 if source==0 (LR/HC, R<low_reward_threshold & C>cost_threshold)"
			),
		},
		save_path,
	)
	print("Saved trained reinforce RUDDER to:", save_path)


if __name__ == "__main__":
	main()