#!/usr/bin/env python

import argparse
import os

import numpy as np
from tqdm import tqdm

try:
    import pickle5 as pickle
except ImportError:
    import pickle

import gym
import safety_gym
from safe_rl.utils.load_utils import load_policy


DEFAULT_POLICY_PATH = "data/ppo_PointGoal1/ppo_PointGoal1_s0"
DEFAULT_OUTPUT_PATH = "datasets/ppo_PointGoal1_s0_highcost_lowreward_1000.pickle"


def _resolve_path(path):
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(normalized):
        return normalized
    return os.path.abspath(normalized)


def _normalize_policy_inputs(policy_path, itr):
    """
    load_policy expects fpath to experiment dir and appends 'simple_save{itr}'.
    If caller passes .../simple_save332 directly, rewrite to parent dir + itr=332.
    """
    tail = os.path.basename(os.path.normpath(policy_path))
    if tail.startswith("simple_save"):
        suffix = tail[len("simple_save"):]
        if suffix.isdigit():
            inferred_itr = int(suffix)
            parent = os.path.dirname(os.path.normpath(policy_path))
            return parent, inferred_itr
    return policy_path, itr


def _run_one_episode(
    env,
    get_action,
    rng,
    noise_std=0.0,
    random_action_prob=0.0,
    max_ep_len=0,
):
    obs = env.reset()
    done = False
    ep_len = 0

    states = []
    actions = []
    rewards = []
    costs = []
    dones = []

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


def collect_filtered_trajectories(
    env,
    get_action,
    target_count,
    reward_lt,
    cost_gt,
    noise_std=0.35,
    noise_max_std=1.0,
    noise_growth=1.25,
    noise_patience=50,
    random_action_prob=0.10,
    max_ep_len=0,
    max_attempts=200000,
    seed=0,
):
    if env is None:
        raise RuntimeError(
            "Environment not found. Pass --env_id to create one manually, "
            "for example --env_id Safexp-PointGoal1-v0"
        )

    kept = []
    attempts = 0
    current_noise_std = float(noise_std)
    no_keep_streak = 0
    rng = np.random.RandomState(seed)

    pbar = tqdm(total=target_count, desc="kept_trajectories", ncols=100)

    while len(kept) < target_count and attempts < max_attempts:
        attempts += 1

        traj, ret_sum, cost_sum, ep_len = _run_one_episode(
            env,
            get_action,
            rng,
            noise_std=current_noise_std,
            random_action_prob=random_action_prob,
            max_ep_len=max_ep_len,
        )

        keep = (cost_sum > cost_gt) and (ret_sum < reward_lt)
        if keep:
            no_keep_streak = 0
            kept.append(traj)
            pbar.update(1)
            print(
                "[KEEP] #%d | EpRet=%.3f EpCost=%.3f EpLen=%d | noise_std=%.4f"
                % (len(kept), ret_sum, cost_sum, ep_len, current_noise_std)
            )
        else:
            no_keep_streak += 1
            if no_keep_streak >= max(1, int(noise_patience)):
                new_noise_std = min(float(noise_max_std), float(current_noise_std) * float(noise_growth))
                if new_noise_std > current_noise_std:
                    current_noise_std = new_noise_std
                    print(
                        "Increasing noise_std to %.4f after %d consecutive non-kept episodes"
                        % (current_noise_std, no_keep_streak)
                    )
                no_keep_streak = 0

        if attempts % 50 == 0:
            print(
                "[INFO] attempts=%d kept=%d | last EpRet=%.3f EpCost=%.3f EpLen=%d | noise_std=%.4f"
                % (attempts, len(kept), ret_sum, cost_sum, ep_len, current_noise_std)
            )

    pbar.close()

    if len(kept) < target_count:
        raise RuntimeError(
            "Could not collect enough filtered trajectories. "
            "Collected %d/%d after %d attempts. Increase --max_attempts or relax thresholds."
            % (len(kept), target_count, attempts)
        )

    return kept, attempts


def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectories and keep only high-cost and low-reward episodes"
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default=DEFAULT_POLICY_PATH,
        help="Path to experiment folder (e.g. data/.../ppo_PointGoal1_s0) or direct simple_save folder",
    )
    parser.add_argument("--itr", "-i", type=int, default=332)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", "-d", action="store_true")
    parser.add_argument(
        "--env_id",
        "-e",
        type=str,
        default=None,
        help="Gym env id to use if env was not saved in vars*.pkl",
    )
    parser.add_argument(
        "--num_trajectories",
        "-n",
        type=int,
        default=1000,
        help="Number of filtered trajectories to keep",
    )
    parser.add_argument(
        "--reward_lt",
        type=float,
        default=15.0,
        help="Keep only trajectories with cumulative reward < this value",
    )
    parser.add_argument(
        "--cost_gt",
        type=float,
        default=25.0,
        help="Keep only trajectories with cumulative cost > this value",
    )
    parser.add_argument("--noise_std", type=float, default=0.35, help="Initial Gaussian noise std added to actions")
    parser.add_argument("--noise_max_std", type=float, default=1.0, help="Maximum adaptive action noise std")
    parser.add_argument("--noise_growth", type=float, default=1.25, help="Noise multiplier after patience is hit")
    parser.add_argument("--noise_patience", type=int, default=50, help="Consecutive non-kept episodes before increasing noise")
    parser.add_argument("--random_action_prob", type=float, default=0.10, help="Probability of replacing policy action with random env action")
    parser.add_argument("--max_ep_len", "-l", type=int, default=0)
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=200000,
        help="Maximum episodes to roll out while filtering",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Output pickle path",
    )
    args = parser.parse_args()

    policy_path = _resolve_path(args.policy_path)
    policy_path, resolved_itr = _normalize_policy_inputs(policy_path, args.itr)
    output_path = _resolve_path(args.output)

    env, get_action, _sess = load_policy(
        policy_path,
        resolved_itr if resolved_itr >= 0 else "last",
        args.deterministic,
    )

    if env is None and args.env_id is not None:
        env = gym.make(args.env_id)
        print("Loaded policy without saved env. Created env: %s" % args.env_id)

    trajectories, attempts = collect_filtered_trajectories(
        env=env,
        get_action=get_action,
        target_count=int(args.num_trajectories),
        reward_lt=float(args.reward_lt),
        cost_gt=float(args.cost_gt),
        noise_std=float(args.noise_std),
        noise_max_std=float(args.noise_max_std),
        noise_growth=float(args.noise_growth),
        noise_patience=int(args.noise_patience),
        random_action_prob=float(args.random_action_prob),
        max_ep_len=int(args.max_ep_len),
        max_attempts=int(args.max_attempts),
        seed=int(args.seed),
    )

    reward_sums = np.asarray([np.sum(t["rewards"]) for t in trajectories], dtype=np.float32)
    cost_sums = np.asarray([np.sum(t["costs"]) for t in trajectories], dtype=np.float32)

    payload = {
        "trajectories": trajectories,
        "metadata": {
            "policy_path": policy_path,
            "itr": int(resolved_itr),
            "num_trajectories": int(args.num_trajectories),
            "attempted_episodes": int(attempts),
            "seed": int(args.seed),
            "filters": {
                "cumulative_cost_gt": float(args.cost_gt),
                "cumulative_reward_lt": float(args.reward_lt),
            },
            "noise": {
                "noise_std": float(args.noise_std),
                "noise_max_std": float(args.noise_max_std),
                "noise_growth": float(args.noise_growth),
                "noise_patience": int(args.noise_patience),
                "random_action_prob": float(args.random_action_prob),
            },
            "reward_sum_stats": {
                "min": float(reward_sums.min()),
                "mean": float(reward_sums.mean()),
                "max": float(reward_sums.max()),
            },
            "cost_sum_stats": {
                "min": float(cost_sums.min()),
                "mean": float(cost_sums.mean()),
                "max": float(cost_sums.max()),
            },
        },
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved filtered dataset to: %s" % output_path)
    print("Kept trajectories: %d" % len(trajectories))
    print(
        "Filters applied: cumulative_cost > %.3f and cumulative_reward < %.3f"
        % (float(args.cost_gt), float(args.reward_lt))
    )


if __name__ == "__main__":
    main()
