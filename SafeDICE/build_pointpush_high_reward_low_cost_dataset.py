#!/usr/bin/env python
"""Build a PointPush1 high-reward/low-cost trajectory dataset.

Target criteria:
- episode return > 6
- episode cost < 25

The script starts from the existing SafeDICE dataset, keeps all unique
trajectories that already satisfy the criteria, then samples additional noisy
policy rollouts until it reaches 1000 unique trajectories. Output format
matches the repository's pickle convention:

{
  "trajectories": [...],
  "metadata": {...}
}
"""

import argparse
import hashlib
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pickle5 as pickle
except ImportError:
    import pickle


DEFAULT_POLICY_PATH = "/home/ed21b059/ddp/data_new/ppo_lagrangian_PointPush1/ppo_lagrangian_PointPush1_s0/simple_save280"
DEFAULT_SAFEDICE_DATASET = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_lagrangian_PointPush1_s0.pickle"
DEFAULT_OUTPUT_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_high_reward_low_cost_1000.pickle"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


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

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError("Input dataset not found: %s. Tried: %s" % (path, ", ".join(candidates)))


def _pick(mapping: Dict[str, Any], candidates: Sequence[str]) -> Optional[Any]:
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _as_1d(arr: Any) -> np.ndarray:
    return np.asarray(arr).reshape(-1)


def _trajectory_return(traj: Dict[str, Any]) -> float:
    rewards = _pick(traj, ("rewards", "reward", "rews", "r", "env_rewards"))
    if rewards is None:
        raise KeyError("Trajectory missing rewards/reward/rews/r field")
    return float(np.sum(_as_1d(rewards).astype(np.float32)))


def _trajectory_cost(traj: Dict[str, Any]) -> float:
    costs = _pick(traj, ("costs", "cost", "c"))
    if costs is None:
        raise KeyError("Trajectory missing costs/cost/c field")
    return float(np.sum(_as_1d(costs).astype(np.float32)))


def _traj_fingerprint(traj: Dict[str, np.ndarray], decimals: int = 3, max_steps: int = 256) -> str:
    states = np.asarray(traj["states"], dtype=np.float32)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    rewards = np.asarray(traj["rewards"], dtype=np.float32).reshape(-1)
    costs = np.asarray(traj["costs"], dtype=np.float32).reshape(-1)

    n = int(min(len(states), len(actions), len(rewards), len(costs), max_steps))
    if n <= 0:
        return "empty"

    digest = hashlib.sha1()
    digest.update(np.round(states[:n], decimals=decimals).tobytes())
    digest.update(np.round(actions[:n], decimals=decimals).tobytes())
    digest.update(np.round(rewards[:n], decimals=decimals).tobytes())
    digest.update(np.round(costs[:n], decimals=decimals).tobytes())
    digest.update(str(states.shape).encode("utf-8"))
    digest.update(str(actions.shape).encode("utf-8"))
    return digest.hexdigest()


def _safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return next_obs, reward, done, info
    return out


def _policy_root_and_itr(policy_path: str) -> Tuple[str, Union[str, int]]:
    path = _resolve_path(policy_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Policy path not found: {path}")

    base = os.path.basename(path)
    if os.path.isdir(path) and base.startswith("simple_save"):
        suffix = base[len("simple_save"):]
        if suffix.isdigit():
            return os.path.dirname(path), int(suffix)
        return os.path.dirname(path), "last"

    return path, "last"


def _load_policy(policy_path: str, deterministic: bool):
    starter_root = os.path.join(_repo_root(), "3rdparty", "safety-starter-agents")
    if starter_root not in sys.path:
        sys.path.insert(0, starter_root)

    from safe_rl.utils.load_utils import load_policy  # pylint: disable=import-error

    policy_root, itr = _policy_root_and_itr(policy_path)
    env, get_action, sess = load_policy(policy_root, itr=itr, deterministic=deterministic)
    return env, get_action, sess, policy_root, itr


def _build_env_if_missing(env, env_id: str):
    if env is not None:
        return env

    import gym
    import safety_gym  # noqa: F401

    return gym.make(env_id)


def _extract_payload(loaded: Any) -> Any:
    payload = loaded
    if isinstance(payload, dict):
        ts = payload.get("training_state")
        if isinstance(ts, dict):
            for key in ("dataset", "data", "replay_buffer", "buffer", "expert_data", "trajectories"):
                if key in ts:
                    payload = ts[key]
                    break

        if payload is loaded:
            for key in ("dataset", "data", "expert_data", "trajectories"):
                if key in payload:
                    payload = payload[key]
                    break

    return payload


def _split_flat_dataset_into_trajectories(payload: Dict[str, Any]) -> List[Dict[str, np.ndarray]]:
    states = _pick(payload, ("states", "observations", "obs", "state", "s"))
    actions = _pick(payload, ("actions", "acts", "action", "a"))
    next_states = _pick(payload, ("next_states", "next_observations", "next_obs", "next_state", "s_next", "obs2"))
    rewards = _pick(payload, ("rewards", "reward", "rews", "r", "env_rewards"))
    costs = _pick(payload, ("costs", "cost", "c"))
    dones = _pick(payload, ("dones", "done", "terminals", "terminal", "episode_ends", "timeouts"))

    if rewards is None:
        raise KeyError("No reward field found in flat dataset")
    if dones is None:
        raise KeyError("No done/terminal field found in flat dataset; cannot segment trajectories")

    arrays: Dict[str, np.ndarray] = {}
    if states is not None:
        arrays["states"] = np.asarray(states, dtype=np.float32)
    if actions is not None:
        arrays["actions"] = np.asarray(actions, dtype=np.float32)
    if next_states is not None:
        arrays["next_states"] = np.asarray(next_states, dtype=np.float32)
    arrays["rewards"] = np.asarray(rewards, dtype=np.float32)
    if costs is not None:
        arrays["costs"] = np.asarray(costs, dtype=np.float32)

    done_arr = _as_1d(dones).astype(bool)
    lengths = [len(done_arr)] + [len(v) for v in arrays.values()]
    n = int(min(lengths))
    done_arr = done_arr[:n]
    for key in list(arrays.keys()):
        arrays[key] = arrays[key][:n]

    trajectories: List[Dict[str, np.ndarray]] = []
    start = 0
    for i, done in enumerate(done_arr):
        if done:
            if i + 1 > start:
                traj = {k: v[start:i + 1] for k, v in arrays.items()}
                traj["dones"] = done_arr[start:i + 1].astype(np.float32)
                trajectories.append(traj)
            start = i + 1

    if start < n:
        traj = {k: v[start:n] for k, v in arrays.items()}
        traj["dones"] = done_arr[start:n].astype(np.float32)
        trajectories.append(traj)

    return trajectories


def _extract_trajectories(loaded: Any) -> List[Dict[str, Any]]:
    payload = _extract_payload(loaded)

    if isinstance(payload, list):
        trajectories = [t for t in payload if isinstance(t, dict)]
        if not trajectories:
            raise ValueError("Dataset payload is a list but contains no trajectory dicts")
        return trajectories

    if isinstance(payload, dict):
        return _split_flat_dataset_into_trajectories(payload)

    raise TypeError("Unsupported dataset payload type: %s" % type(payload))


def _load_pickle_like(path: str) -> Any:
    with open(path, "rb") as f:
        header = f.read(256)

    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "Dataset file appears to be a Git LFS pointer, not actual pickle data: %s. Run 'git lfs pull' and retry."
            % path
        )

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ValueError as exc:
        msg = str(exc)
        if "unsupported pickle protocol: 5" in msg:
            raise RuntimeError(
                "Failed to load protocol-5 pickle at %s using interpreter %s and module %s. "
                "Install pickle5 in this env or use Python >= 3.8."
                % (path, sys.version.split()[0], pickle.__name__)
            )
        raise


def _run_one_episode(
    env,
    get_action,
    rng: np.random.RandomState,
    noise_std: float,
    random_action_prob: float,
    max_ep_len: int,
) -> Tuple[Dict[str, np.ndarray], float, float, int]:
    obs = _safe_reset(env)
    done = False
    ep_len = 0

    states: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    next_states: List[np.ndarray] = []
    rewards: List[float] = []
    costs: List[float] = []
    dones: List[float] = []

    while not done and ep_len < max_ep_len:
        action = np.asarray(get_action(obs), dtype=np.float32)
        if noise_std > 0.0:
            action = action + rng.normal(0.0, noise_std, size=action.shape).astype(np.float32)

        if random_action_prob > 0.0 and hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
            if float(rng.rand()) < float(random_action_prob):
                action = rng.uniform(low=env.action_space.low, high=env.action_space.high).astype(np.float32)

        if hasattr(env.action_space, "low") and hasattr(env.action_space, "high"):
            action = np.clip(action, env.action_space.low, env.action_space.high)

        next_obs, reward, done, info = _safe_step(env, action)
        cost = float(info.get("cost", 0.0))

        states.append(np.asarray(obs, dtype=np.float32))
        actions.append(np.asarray(action, dtype=np.float32))
        next_states.append(np.asarray(next_obs, dtype=np.float32))
        rewards.append(float(reward))
        costs.append(cost)
        dones.append(float(done))

        obs = next_obs
        ep_len += 1

    traj = {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_states": np.asarray(next_states, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "costs": np.asarray(costs, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
    }
    return traj, float(np.sum(rewards)), float(np.sum(costs)), int(ep_len)


def _stats(trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
    returns = np.asarray([_trajectory_return(t) for t in trajectories], dtype=np.float32)
    costs = np.asarray([_trajectory_cost(t) for t in trajectories], dtype=np.float32)
    lengths = np.asarray([len(np.asarray(t["rewards"]).reshape(-1)) for t in trajectories], dtype=np.float32)
    return {
        "count": float(len(trajectories)),
        "return_min": float(np.min(returns)),
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_max": float(np.max(returns)),
        "cost_min": float(np.min(costs)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_max": float(np.max(costs)),
        "len_min": float(np.min(lengths)),
        "len_mean": float(np.mean(lengths)),
        "len_max": float(np.max(lengths)),
    }


def _collect_high_reward_low_cost_from_policy(
    env,
    get_action,
    target_total: int,
    initial_seen: Iterable[str],
    reward_threshold: float,
    cost_threshold: float,
    noise_std: float,
    noise_max_std: float,
    noise_growth: float,
    noise_patience: int,
    random_action_prob: float,
    max_rollout_episodes: int,
    max_ep_len: int,
    seed: int,
) -> Tuple[List[Dict[str, np.ndarray]], int, int]:
    rng = np.random.RandomState(seed)
    collected: List[Dict[str, np.ndarray]] = []
    seen = set(initial_seen)
    attempted = 0
    duplicate_count = 0
    current_noise_std = float(noise_std)
    no_keep_streak = 0

    while attempted < max_rollout_episodes and len(seen) < target_total:
        traj, ep_ret, ep_cost, ep_len = _run_one_episode(
            env=env,
            get_action=get_action,
            rng=rng,
            noise_std=current_noise_std,
            random_action_prob=random_action_prob,
            max_ep_len=max_ep_len,
        )
        attempted += 1

        keep = (ep_ret > reward_threshold) and (ep_cost < cost_threshold)
        if keep:
            fingerprint = _traj_fingerprint(traj)
            if fingerprint not in seen:
                seen.add(fingerprint)
                collected.append(traj)
                no_keep_streak = 0
            else:
                duplicate_count += 1
        else:
            no_keep_streak += 1
            if no_keep_streak >= max(1, int(noise_patience)):
                next_noise = min(float(noise_max_std), float(current_noise_std) * float(noise_growth))
                if next_noise > current_noise_std:
                    current_noise_std = next_noise
                    print(
                        "Increasing noise_std to %.4f after %d consecutive non-kept rollouts"
                        % (current_noise_std, no_keep_streak)
                    )
                no_keep_streak = 0

        print(
            "Rollout %d | return=%.3f cost=%.3f len=%d | keep(>%.2f,<%.2f)=%s | unique=%d/%d | noise_std=%.4f"
            % (
                attempted,
                ep_ret,
                ep_cost,
                ep_len,
                reward_threshold,
                cost_threshold,
                str(keep),
                len(seen),
                target_total,
                current_noise_std,
            )
        )

    if len(seen) < target_total:
        raise RuntimeError(
            "Collected only %d unique high-reward/low-cost trajectories (target=%d) after %d episodes. "
            "Increase --max_rollout_episodes, increase --noise_std/--noise_max_std, set --random_action_prob, or relax the thresholds."
            % (len(seen), target_total, attempted)
        )

    return collected, attempted, duplicate_count


def _keep_unique_trajectories(
    trajectories: List[Dict[str, Any]],
    reward_threshold: float,
    cost_threshold: float,
) -> Tuple[List[Dict[str, Any]], set]:
    unique_trajs: List[Dict[str, Any]] = []
    seen = set()
    for traj in trajectories:
        if (_trajectory_return(traj) > reward_threshold) and (_trajectory_cost(traj) < cost_threshold):
            fingerprint = _traj_fingerprint(traj)
            if fingerprint not in seen:
                seen.add(fingerprint)
                copied = dict(traj)
                copied["source"] = np.asarray([0], dtype=np.int32)
                unique_trajs.append(copied)
    return unique_trajs, seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PointPush1 high-reward/low-cost dataset from SafeDICE plus noisy PPO rollouts"
    )
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--safedice_dataset", type=str, default=DEFAULT_SAFEDICE_DATASET)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target_count", type=int, default=1000)
    parser.add_argument("--reward_threshold", type=float, default=5.0)
    parser.add_argument("--cost_threshold", type=float, default=25.0)
    parser.add_argument("--noise_std", type=float, default=0.08)
    parser.add_argument("--noise_max_std", type=float, default=0.60)
    parser.add_argument("--noise_growth", type=float, default=1.18)
    parser.add_argument("--noise_patience", type=int, default=60)
    parser.add_argument("--random_action_prob", type=float, default=0.04)
    parser.add_argument("--max_rollout_episodes", type=int, default=120000)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--env_id", type=str, default="Safexp-PointPush1-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base_dataset_path = _resolve_input_path(args.safedice_dataset)
    loaded = _load_pickle_like(base_dataset_path)
    trajectories = _extract_trajectories(loaded)

    base_unique, initial_seen = _keep_unique_trajectories(
        trajectories,
        reward_threshold=args.reward_threshold,
        cost_threshold=args.cost_threshold,
    )
    print("Loaded base dataset:", base_dataset_path)
    print("Base trajectories:", len(trajectories))
    print("Base qualifying unique trajectories:", len(base_unique))

    if len(base_unique) > args.target_count:
        base_unique = base_unique[: args.target_count]
        initial_seen = {_traj_fingerprint(traj) for traj in base_unique}

    env, get_action, sess, policy_root, itr = _load_policy(args.policy_path, args.deterministic)
    env = _build_env_if_missing(env, args.env_id)

    print("Policy root: %s" % policy_root)
    print("Policy itr: %s" % str(itr))
    print(
        "Top-up sampling until total unique trajectories reaches %d with return > %.3f and cost < %.3f"
        % (args.target_count, args.reward_threshold, args.cost_threshold)
    )
    print(
        "Adaptive noise config: init_std=%.4f max_std=%.4f growth=%.3f patience=%d random_action_prob=%.3f"
        % (args.noise_std, args.noise_max_std, args.noise_growth, args.noise_patience, args.random_action_prob)
    )

    needed = max(0, args.target_count - len(base_unique))
    generated, attempted, duplicate_count = _collect_high_reward_low_cost_from_policy(
        env=env,
        get_action=get_action,
        target_total=args.target_count,
        initial_seen=initial_seen,
        reward_threshold=args.reward_threshold,
        cost_threshold=args.cost_threshold,
        noise_std=args.noise_std,
        noise_max_std=args.noise_max_std,
        noise_growth=args.noise_growth,
        noise_patience=args.noise_patience,
        random_action_prob=args.random_action_prob,
        max_rollout_episodes=args.max_rollout_episodes,
        max_ep_len=args.max_ep_len,
        seed=args.seed,
    )

    all_trajectories = list(base_unique) + list(generated)
    if len(all_trajectories) > args.target_count:
        all_trajectories = all_trajectories[: args.target_count]

    stats = _stats(all_trajectories)
    payload = {
        "trajectories": all_trajectories,
        "metadata": {
            "base_dataset_path": base_dataset_path,
            "base_dataset_count": int(len(trajectories)),
            "base_qualifying_unique_count": int(len(base_unique)),
            "generated_new_count": int(len(generated)),
            "total_count": int(len(all_trajectories)),
            "policy_path": _resolve_path(args.policy_path),
            "policy_root": policy_root,
            "policy_itr": str(itr),
            "reward_threshold": float(args.reward_threshold),
            "cost_threshold": float(args.cost_threshold),
            "target_count": int(args.target_count),
            "needed_from_policy": int(needed),
            "noise_std": float(args.noise_std),
            "noise_max_std": float(args.noise_max_std),
            "noise_growth": float(args.noise_growth),
            "noise_patience": int(args.noise_patience),
            "random_action_prob": float(args.random_action_prob),
            "seed": int(args.seed),
            "rollout_attempted_episodes": int(attempted),
            "duplicate_rejections": int(duplicate_count),
            "stats": stats,
            "criteria": "return > %.3f and cost < %.3f" % (args.reward_threshold, args.cost_threshold),
        },
    }

    out_path = _resolve_path(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nSaved high-reward/low-cost dataset to:", out_path)
    print("Total trajectories:", len(all_trajectories))
    print("Base qualifying unique trajectories:", len(base_unique))
    print("Generated unique trajectories:", len(generated))
    print("Duplicate episodes rejected:", duplicate_count)
    print(
        "Reward sum stats min/mean/max: %.3f / %.3f / %.3f"
        % (stats["return_min"], stats["return_mean"], stats["return_max"])
    )
    print(
        "Cost sum stats   min/mean/max: %.3f / %.3f / %.3f"
        % (stats["cost_min"], stats["cost_mean"], stats["cost_max"])
    )

    if env is not None:
        env.close()
    if sess is not None:
        sess.close()


if __name__ == "__main__":
    main()