#!/usr/bin/env python

import argparse
import hashlib
import os
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import pickle5 as _pickle5  # type: ignore[import-not-found]
except ImportError:
    _pickle5 = None

import pickle

import gym
import safety_gym  # noqa: F401
from safe_rl.utils.load_utils import load_policy


def _resolve_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)
	if os.path.isabs(normalized):
		return normalized
	return os.path.abspath(normalized)


def _normalize_policy_inputs(policy_path: str, itr: int) -> Tuple[str, int]:
	"""
	load_policy expects experiment dir and appends simple_save{itr}.
	If caller passes .../simple_save332, rewrite to parent dir + itr=332.
	"""
	tail = os.path.basename(os.path.normpath(policy_path))
	if tail.startswith("simple_save"):
		suffix = tail[len("simple_save") :]
		if suffix.isdigit():
			inferred_itr = int(suffix)
			parent = os.path.dirname(os.path.normpath(policy_path))
			return parent, inferred_itr
	return policy_path, int(itr)


def _load_raw_pickle(path: str) -> Any:
	try:
		with open(path, "rb") as f:
			return pickle.load(f)
	except ValueError as e:
		msg = str(e)
		if "unsupported pickle protocol" not in msg:
			raise

		if _pickle5 is not None:
			with open(path, "rb") as f:
				return _pickle5.load(f)

		raise RuntimeError(
			"This dataset uses pickle protocol 5, but current Python is %d.%d and pickle5 is not installed. "
			"Use Python >=3.8 or install pickle5 in this environment: pip install pickle5"
			% (sys.version_info.major, sys.version_info.minor)
		) from e


def _split_trajectories_from_arrays(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
	states = np.asarray(payload["states"], dtype=np.float32)
	actions = np.asarray(payload["actions"], dtype=np.float32)
	rewards = np.asarray(payload.get("rewards", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
	costs = np.asarray(payload.get("costs", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)
	dones = np.asarray(payload.get("dones", np.zeros((len(states),), dtype=np.float32)), dtype=np.float32).reshape(-1)

	n = int(min(len(states), len(actions), len(rewards), len(costs), len(dones)))
	if n <= 0:
		raise ValueError("Array-format dataset is empty")

	trajectories: List[Dict[str, np.ndarray]] = []
	start = 0
	for i in range(n):
		done_flag = bool(dones[i] >= 0.5)
		end_of_data = i == (n - 1)
		if done_flag or end_of_data:
			end = i + 1
			if end > start:
				trajectories.append(
					{
						"states": states[start:end].astype(np.float32, copy=False),
						"actions": actions[start:end].astype(np.float32, copy=False),
						"rewards": rewards[start:end].astype(np.float32, copy=False),
						"costs": costs[start:end].astype(np.float32, copy=False),
						"dones": dones[start:end].astype(np.float32, copy=False),
					}
				)
			start = end

	return trajectories


def load_existing_trajectories(dataset_path: str) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, Any], str]:
	resolved = _resolve_path(dataset_path)
	try:
		payload = _load_raw_pickle(resolved)
	except Exception as e:
		msg = str(e)
		fallback_payload = None
		fallback_path = None

		if "pickle data was truncated" in msg and ("/safetygym_original/" in resolved or (os.sep + "safetygym_original" + os.sep) in resolved):
			fallback_path = resolved.replace(os.sep + "safetygym_original" + os.sep, os.sep + "safetygym" + os.sep)
			if os.path.isfile(fallback_path):
				print(
					"[WARN] Existing dataset appears truncated. Falling back to readable dataset: %s"
					% fallback_path
				)
				fallback_payload = _load_raw_pickle(fallback_path)

		if fallback_payload is None:
			raise

		payload = fallback_payload
		resolved = fallback_path

	if isinstance(payload, dict) and "trajectories" in payload:
		raw_trajs = payload["trajectories"]
		metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
	elif isinstance(payload, list):
		raw_trajs = payload
		metadata = {}
	elif isinstance(payload, dict) and all(k in payload for k in ["states", "actions", "rewards", "costs", "dones"]):
		raw_trajs = _split_trajectories_from_arrays(payload)
		metadata = {
			"source_format": "flat_arrays_split_by_dones",
			"source_keys": sorted(list(payload.keys())),
		}
	else:
		raise ValueError(
			"Unsupported dataset format. Expected dict with 'trajectories', list of trajectories, or flat arrays."
		)

	trajectories: List[Dict[str, np.ndarray]] = []
	for i, traj in enumerate(raw_trajs):
		if not isinstance(traj, dict):
			raise TypeError("Trajectory %d is not a dict" % i)
		if "states" not in traj or "actions" not in traj:
			raise KeyError("Trajectory %d missing states/actions" % i)

		states = np.asarray(traj["states"], dtype=np.float32)
		actions = np.asarray(traj["actions"], dtype=np.float32)

		rewards = np.asarray(
			traj.get("rewards", np.zeros((len(states),), dtype=np.float32)),
			dtype=np.float32,
		).reshape(-1)
		costs = np.asarray(
			traj.get("costs", np.zeros((len(states),), dtype=np.float32)),
			dtype=np.float32,
		).reshape(-1)
		dones = np.asarray(
			traj.get("dones", np.zeros((len(states),), dtype=np.float32)),
			dtype=np.float32,
		).reshape(-1)

		n = int(min(len(states), len(actions), len(rewards), len(costs), len(dones)))
		if n <= 0:
			continue

		trajectories.append(
			{
				"states": states[:n],
				"actions": actions[:n],
				"rewards": rewards[:n],
				"costs": costs[:n],
				"dones": dones[:n],
			}
		)

	if len(trajectories) == 0:
		raise ValueError("No valid trajectories found in existing dataset: %s" % resolved)

	return trajectories, metadata, resolved


def _traj_fingerprint(traj: Dict[str, np.ndarray], decimals: int = 3, max_steps: int = 256) -> str:
	states = np.asarray(traj["states"], dtype=np.float32)
	actions = np.asarray(traj["actions"], dtype=np.float32)
	rewards = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)
	costs = np.asarray(traj["costs"], dtype=np.float32).reshape(-1)

	n = int(min(len(states), len(actions), len(rewards), len(costs), max_steps))
	if n <= 0:
		return "empty"

	state_head = np.round(states[:n], decimals=decimals)
	action_head = np.round(actions[:n], decimals=decimals)
	reward_head = np.round(rewards[:n], decimals=decimals)
	cost_head = np.round(costs[:n], decimals=decimals)

	digest = hashlib.sha1()
	digest.update(state_head.tobytes())
	digest.update(action_head.tobytes())
	digest.update(reward_head.tobytes())
	digest.update(cost_head.tobytes())
	digest.update(str(states.shape).encode("utf-8"))
	digest.update(str(actions.shape).encode("utf-8"))
	return digest.hexdigest()


def _run_one_episode(
	env,
	get_action,
	rng: np.random.RandomState,
	noise_std: float,
	random_action_prob: float,
	max_ep_len: int,
) -> Tuple[Dict[str, np.ndarray], float, float, int]:
	obs = env.reset()
	done = False
	ep_len = 0

	states: List[np.ndarray] = []
	actions: List[np.ndarray] = []
	rewards: List[float] = []
	costs: List[float] = []
	dones: List[bool] = []

	while True:
		action = np.asarray(get_action(obs), dtype=np.float32)
		if noise_std > 0.0:
			action = action + rng.normal(loc=0.0, scale=noise_std, size=action.shape).astype(np.float32)

		if random_action_prob > 0.0 and float(rng.rand()) < float(random_action_prob):
			action = rng.uniform(low=env.action_space.low, high=env.action_space.high).astype(np.float32)

		action = np.clip(action, env.action_space.low, env.action_space.high)

		next_obs, reward, done, info = env.step(action)
		step_cost = float(info.get("cost", 0.0))

		states.append(np.asarray(obs, dtype=np.float32))
		actions.append(np.asarray(action, dtype=np.float32))
		rewards.append(float(reward))
		costs.append(step_cost)
		dones.append(bool(done))

		obs = next_obs
		ep_len += 1

		if done or (max_ep_len > 0 and ep_len >= max_ep_len):
			break

	traj = {
		"states": np.asarray(states, dtype=np.float32),
		"actions": np.asarray(actions, dtype=np.float32),
		"rewards": np.asarray(rewards, dtype=np.float32),
		"costs": np.asarray(costs, dtype=np.float32),
		"dones": np.asarray(dones, dtype=np.float32),
	}
	ret_sum = float(np.sum(traj["rewards"]))
	cost_sum = float(np.sum(traj["costs"]))
	return traj, ret_sum, cost_sum, ep_len


def _schedule(attempt: int, target_new: int) -> Tuple[float, float]:
	progress = min(1.0, float(attempt) / max(1.0, float(target_new) * 2.0))
	noise_std = 0.01 + 0.04 * progress
	random_action_prob = 0.0 + 0.02 * progress
	return noise_std, random_action_prob


def generate_diverse_trajectories(
	env,
	get_action,
	existing_trajectories: Sequence[Dict[str, np.ndarray]],
	target_new: int,
	max_ep_len: int,
	max_attempts: int,
	seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
	rng = np.random.RandomState(seed)

	existing_fingerprints = set()
	for traj in existing_trajectories:
		existing_fingerprints.add(_traj_fingerprint(traj))

	new_trajectories: List[Dict[str, np.ndarray]] = []
	new_fingerprints = set()
	duplicate_count = 0

	pbar = tqdm(total=target_new, desc="collect_new_diverse", ncols=110)
	attempts = 0

	while len(new_trajectories) < target_new and attempts < max_attempts:
		attempts += 1
		noise_std, random_action_prob = _schedule(attempt=attempts, target_new=target_new)

		traj, ret_sum, cost_sum, ep_len = _run_one_episode(
			env=env,
			get_action=get_action,
			rng=rng,
			noise_std=noise_std,
			random_action_prob=random_action_prob,
			max_ep_len=max_ep_len,
		)

		key = _traj_fingerprint(traj)
		if key in existing_fingerprints or key in new_fingerprints:
			duplicate_count += 1
		else:
			new_trajectories.append(traj)
			new_fingerprints.add(key)
			pbar.update(1)
			if len(new_trajectories) <= 5 or len(new_trajectories) % 100 == 0:
				print(
					"[KEEP] #%d ret=%.3f cost=%.3f len=%d noise=%.3f rand=%.3f"
					% (len(new_trajectories), ret_sum, cost_sum, ep_len, noise_std, random_action_prob)
				)

		if attempts % 200 == 0:
			print(
				"[INFO] attempts=%d kept=%d duplicates=%d"
				% (attempts, len(new_trajectories), duplicate_count)
			)

	pbar.close()

	if len(new_trajectories) < target_new:
		raise RuntimeError(
			"Could not collect enough unique trajectories: got %d/%d after %d attempts. "
			"Increase --max_attempts or adjust exploration schedule."
			% (len(new_trajectories), target_new, attempts)
		)

	return new_trajectories, attempts, duplicate_count


def _stats(arr: np.ndarray) -> Dict[str, float]:
	return {
		"min": float(np.min(arr)),
		"mean": float(np.mean(arr)),
		"max": float(np.max(arr)),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Append diverse Safety-PointGoal trajectories for Q-network training")
	parser.add_argument(
		"--policy_path",
		type=str,
		default="/home/ed21b059/ddp/data/ppo_lagrangian_PointGoal1/ppo_lagrangian_PointGoal1_s0/simple_save332",
		help="Path to experiment folder or direct simple_save folder",
	)
	parser.add_argument("--itr", type=int, default=332, help="Ignored if policy_path already ends with simple_saveNNN")
	parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy actions")
	parser.add_argument(
		"--existing_dataset",
		type=str,
		default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym_original/ppo_lagrangian_PointGoal1_s0.pickle",
		help="Path to existing pickle with ~1000 trajectories",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="/home/ed21b059/ddp/SafeDICE/dataset/safetygym_original/ppo_lagrangian_PointGoal1_s0_diverse_2000.pickle",
		help="Output pickle path (dict with trajectories + metadata)",
	)
	parser.add_argument("--new_trajectories", type=int, default=1000, help="Number of additional trajectories to generate")
	parser.add_argument("--max_ep_len", type=int, default=0, help="Episode truncation length; 0 means env default")
	parser.add_argument("--max_attempts", type=int, default=120000, help="Max rollout attempts to gather unique episodes")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument(
		"--env_id",
		type=str,
		default="Safexp-PointGoal1-v0",
		help="Fallback env id if policy checkpoint does not store env",
	)
	args = parser.parse_args()

	policy_path_raw = _resolve_path(args.policy_path)
	policy_path, resolved_itr = _normalize_policy_inputs(policy_path_raw, int(args.itr))

	existing_trajectories, existing_metadata, existing_resolved = load_existing_trajectories(args.existing_dataset)
	print("Loaded existing dataset:", existing_resolved)
	print("Existing trajectories:", len(existing_trajectories))

	env, get_action, _sess = load_policy(
		policy_path,
		resolved_itr if resolved_itr >= 0 else "last",
		bool(args.deterministic),
	)
	if env is None:
		env = gym.make(args.env_id)
		print("Policy loaded without env. Created env:", args.env_id)

	new_trajectories, attempts, duplicate_count = generate_diverse_trajectories(
		env=env,
		get_action=get_action,
		existing_trajectories=existing_trajectories,
		target_new=int(args.new_trajectories),
		max_ep_len=int(args.max_ep_len),
		max_attempts=int(args.max_attempts),
		seed=int(args.seed),
	)

	all_trajectories = list(existing_trajectories) + list(new_trajectories)

	reward_sums = np.asarray([np.sum(t["rewards"]) for t in all_trajectories], dtype=np.float32)
	cost_sums = np.asarray([np.sum(t["costs"]) for t in all_trajectories], dtype=np.float32)
	traj_lens = np.asarray([min(len(t["states"]), len(t["actions"])) for t in all_trajectories], dtype=np.int32)

	payload = {
		"trajectories": all_trajectories,
		"metadata": {
			"base_dataset_path": existing_resolved,
			"base_dataset_count": int(len(existing_trajectories)),
			"generated_new_count": int(len(new_trajectories)),
			"total_count": int(len(all_trajectories)),
			"policy_path": policy_path,
			"itr": int(resolved_itr),
			"seed": int(args.seed),
			"max_ep_len": int(args.max_ep_len),
			"attempted_episodes": int(attempts),
			"duplicate_rejections": int(duplicate_count),
			"dedup_fingerprint": {
				"quantization_decimals": 3,
				"max_steps_hashed": 256,
				"hash": "sha1",
			},
			"exploration_schedule": {
				"noise_std": "0.01 -> 0.05 (linear over attempts)",
				"random_action_prob": "0.00 -> 0.02 (linear over attempts)",
			},
			"reward_sum_stats": _stats(reward_sums),
			"cost_sum_stats": _stats(cost_sums),
			"trajectory_length_stats": {
				"min": int(np.min(traj_lens)),
				"mean": float(np.mean(traj_lens)),
				"max": int(np.max(traj_lens)),
			},
			"inherits_existing_metadata": existing_metadata,
		},
	}

	output_path = _resolve_path(args.output)
	out_dir = os.path.dirname(output_path)
	if out_dir:
		os.makedirs(out_dir, exist_ok=True)

	with open(output_path, "wb") as f:
		pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

	print("\nSaved combined diverse dataset to:", output_path)
	print("Total trajectories:", len(all_trajectories))
	print("New unique trajectories:", len(new_trajectories))
	print("Duplicate episodes rejected:", duplicate_count)
	print(
		"Reward sum stats min/mean/max: %.3f / %.3f / %.3f"
		% (payload["metadata"]["reward_sum_stats"]["min"], payload["metadata"]["reward_sum_stats"]["mean"], payload["metadata"]["reward_sum_stats"]["max"])
	)
	print(
		"Cost sum stats   min/mean/max: %.3f / %.3f / %.3f"
		% (payload["metadata"]["cost_sum_stats"]["min"], payload["metadata"]["cost_sum_stats"]["mean"], payload["metadata"]["cost_sum_stats"]["max"])
	)


if __name__ == "__main__":
	main()
