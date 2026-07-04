#!/usr/bin/env python
"""Build a PointPush1 low-reward/high-cost trajectory dataset.

Target criteria:
- episode return < 2
- episode cost > 25

The script samples rollouts from a PPO policy and progressively increases
action noise until it collects 1000 unique trajectories satisfying the
criteria. Output format matches the repository's pickle convention:

{
  "trajectories": [...],
  "metadata": {...}
}
"""

import argparse
import hashlib
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import pickle5 as pickle
except ImportError:
    import pickle


DEFAULT_POLICY_PATH = "/home/ed21b059/ddp/data_new/ppo_PointPush1/ppo_PointPush1_s0/simple_save300"
DEFAULT_OUTPUT_PATH = "/home/ed21b059/ddp/SafeDICE/dataset/safetygym/ppo_PointPush1_low_reward_high_cost_1000.pickle"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(os.path.join(_repo_root(), normalized))


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
        "source": np.asarray([0], dtype=np.int32),
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


def _collect_unique_low_reward_high_cost(
    env,
    get_action,
    target_count: int,
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
    seen_fingerprints = set()
    attempted = 0
    duplicate_count = 0
    current_noise_std = float(noise_std)
    current_random_action_prob = float(random_action_prob)
    no_keep_streak = 0

    while attempted < max_rollout_episodes and len(collected) < target_count:
        rollout_frac = float(attempted) / max(1.0, float(max_rollout_episodes - 1))
        scheduled_noise_std = min(
            float(noise_max_std),
            float(noise_std) + rollout_frac * (float(noise_max_std) - float(noise_std)),
        )
        scheduled_random_action_prob = min(0.35, float(random_action_prob) + rollout_frac * (0.35 - float(random_action_prob)))
        current_noise_std = max(current_noise_std, scheduled_noise_std)
        current_random_action_prob = max(current_random_action_prob, scheduled_random_action_prob)

        traj, ep_ret, ep_cost, ep_len = _run_one_episode(
            env=env,
            get_action=get_action,
            rng=rng,
            noise_std=current_noise_std,
            random_action_prob=current_random_action_prob,
            max_ep_len=max_ep_len,
        )
        attempted += 1

        keep = (ep_ret < reward_threshold) and (ep_cost > cost_threshold)
        if keep:
            fingerprint = _traj_fingerprint(traj)
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                collected.append(traj)
                no_keep_streak = 0
            else:
                duplicate_count += 1
        else:
            no_keep_streak += 1
            if no_keep_streak >= max(1, int(noise_patience)):
                next_noise = min(
                    float(noise_max_std),
                    max(float(current_noise_std) * float(noise_growth), float(current_noise_std) + 0.20),
                )
                if next_noise > current_noise_std:
                    current_noise_std = next_noise
                current_random_action_prob = min(0.50, max(current_random_action_prob * 1.20, current_random_action_prob + 0.05))
                print(
                    "Increasing exploration to noise_std=%.4f rand_prob=%.3f after %d consecutive non-kept rollouts"
                    % (current_noise_std, current_random_action_prob, no_keep_streak)
                )
                no_keep_streak = 0

        print(
            "Rollout %d | return=%.3f cost=%.3f len=%d | keep(<%.2f,>%.2f)=%s | kept=%d/%d | noise_std=%.4f rand_prob=%.3f"
            % (
                attempted,
                ep_ret,
                ep_cost,
                ep_len,
                reward_threshold,
                cost_threshold,
                str(keep),
                len(collected),
                target_count,
                current_noise_std,
                current_random_action_prob,
            )
        )

    if len(collected) < target_count:
        raise RuntimeError(
            "Collected only %d unique low-reward/high-cost trajectories (target=%d) after %d episodes. "
            "Increase --max_rollout_episodes, increase --noise_std/--noise_max_std, set --random_action_prob, or relax the thresholds."
            % (len(collected), target_count, attempted)
        )

    return collected, attempted, duplicate_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a PointPush1 low-reward/high-cost dataset from noisy PPO rollouts"
    )
    parser.add_argument("--policy_path", type=str, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target_count", type=int, default=1000)
    parser.add_argument("--reward_threshold", type=float, default=2.5)
    parser.add_argument("--cost_threshold", type=float, default=25.0)
    parser.add_argument("--noise_std", type=float, default=0.35)
    parser.add_argument("--noise_max_std", type=float, default=2.50)
    parser.add_argument("--noise_growth", type=float, default=1.35)
    parser.add_argument("--noise_patience", type=int, default=10)
    parser.add_argument("--random_action_prob", type=float, default=0.30)
    parser.add_argument("--max_rollout_episodes", type=int, default=120000)
    parser.add_argument("--max_ep_len", type=int, default=1000)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--env_id", type=str, default="Safexp-PointPush1-v0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = None
    sess = None
    policy_root = ""
    itr: Union[str, int] = ""

    env, get_action, sess, policy_root, itr = _load_policy(args.policy_path, args.deterministic)
    env = _build_env_if_missing(env, args.env_id)

    print("Policy root: %s" % policy_root)
    print("Policy itr: %s" % str(itr))
    print(
        "Collecting %d unique trajectories with return < %.3f and cost > %.3f"
        % (args.target_count, args.reward_threshold, args.cost_threshold)
    )
    print(
        "Adaptive noise config: init_std=%.4f max_std=%.4f growth=%.3f patience=%d random_action_prob=%.3f"
        % (args.noise_std, args.noise_max_std, args.noise_growth, args.noise_patience, args.random_action_prob)
    )

    trajectories, attempted, duplicate_count = _collect_unique_low_reward_high_cost(
        env=env,
        get_action=get_action,
        target_count=args.target_count,
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

    stats = _stats(trajectories)
    payload = {
        "trajectories": trajectories,
        "metadata": {
            "policy_path": _resolve_path(args.policy_path),
            "policy_root": policy_root,
            "policy_itr": str(itr),
            "reward_threshold": float(args.reward_threshold),
            "cost_threshold": float(args.cost_threshold),
            "target_count": int(args.target_count),
            "noise_std": float(args.noise_std),
            "noise_max_std": float(args.noise_max_std),
            "noise_growth": float(args.noise_growth),
            "noise_patience": int(args.noise_patience),
            "random_action_prob": float(args.random_action_prob),
            "seed": int(args.seed),
            "rollout_attempted_episodes": int(attempted),
            "duplicate_rejections": int(duplicate_count),
            "stats": stats,
            "criteria": "return < %.3f and cost > %.3f" % (args.reward_threshold, args.cost_threshold),
        },
    }

    out_path = _resolve_path(args.output)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nSaved low-reward/high-cost dataset to:", out_path)
    print("Total trajectories:", len(trajectories))
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