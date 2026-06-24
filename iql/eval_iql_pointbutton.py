#!/usr/bin/env python3
"""Evaluate IQ-Learn PointButton policy and save reward/cost statistics.

Runs N episodes and writes mean/std for reward and cost to a timestamped JSON.
"""
import time
import json
import os
import argparse
import numpy as np
import torch
import importlib
import sys

# Ensure project root is on sys.path so `import iql` works when running
# this script directly (python iql/eval_iql_pointbutton.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from iql.iql_run_pointbutton import (
    SafetyEnvToGymEnvWrapper,
    IQLearnAgent,
    get_preferred_device,
    env_reset,
    env_step,
)

try:
    gym = importlib.import_module("gym")
except Exception:
    gym = None

try:
    safety_gym = importlib.import_module("safety_gym")
except Exception:
    safety_gym = None


def make_safety_env(env_name):
    if gym is None:
        raise ImportError("gym not available. Install gym to use safety_gym environments")
    if safety_gym is None:
        raise ImportError("safety_gym not available. Install safety_gym to use Safety Gym environments")
    env = gym.make(env_name)
    env = SafetyEnvToGymEnvWrapper(env)
    # Wrap with Monitor for consistent logging/behavior
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env)
    return env


def load_agent_from_checkpoint(checkpoint_path, obs_dim, act_dim, device):
    agent = IQLearnAgent(obs_dim=obs_dim, act_dim=act_dim, device=device)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    agent.critic.load_state_dict(ckpt['critic'])
    agent.critic_target.load_state_dict(ckpt.get('critic_target', agent.critic.state_dict()))
    agent.actor.load_state_dict(ckpt['actor'])
    # log_alpha may be a tensor or dict entry
    agent.log_alpha = ckpt.get('log_alpha', agent.log_alpha)
    # Load optimizer states if present (optional)
    try:
        if 'critic_optimizer' in ckpt:
            agent.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        if 'actor_optimizer' in ckpt:
            agent.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        if 'alpha_optimizer' in ckpt:
            agent.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])
    except Exception:
        pass
    return agent


def evaluate(policy_agent, env, n_episodes=500, deterministic=True, verbose=False):
    rewards = []
    costs = []
    for ep in range(n_episodes):
        obs = env_reset(env)
        done = False
        ep_reward = 0.0
        ep_cost = 0.0
        steps = 0
        while not done:
            action = policy_agent.select_action(obs, deterministic=deterministic)
            obs, reward, done, info = env_step(env, action)
            ep_reward += float(reward)
            # cost may be in info under several names
            c = 0.0
            if isinstance(info, dict):
                c = float(info.get('cost', info.get('costs', info.get('cost_value', 0.0))))
            ep_cost += c
            steps += 1
            # safety: avoid extremely long episodes
            if steps > 10000:
                break
        rewards.append(ep_reward)
        costs.append(ep_cost)
        if verbose:
            print(f"Episode {ep+1}/{n_episodes}: reward={ep_reward:.2f}, cost={ep_cost:.2f}, steps={steps}")
    return np.array(rewards), np.array(costs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-name', default='Safexp-PointButton1-v0')
    parser.add_argument('--checkpoint', default='iq_learn_pointbutton_best.pth')
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--save-dir', default='iql/results')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--verbose', action='store_true', help='Print per-episode reward/cost')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = get_preferred_device()
    print(f"Using device: {device}")

    print(f"Creating environment: {args.env_name}")
    env = make_safety_env(args.env_name)

    obs_dim = int(np.prod(env.observation_space.shape))
    try:
        act_dim = int(np.prod(env.action_space.shape))
    except Exception:
        # Discrete action spaces unsupported for this agent
        raise RuntimeError('Unsupported action space for IQL agent')

    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")

    print(f"Loading agent from: {args.checkpoint}")
    agent = load_agent_from_checkpoint(args.checkpoint, obs_dim, act_dim, device)

    print(f"Running {args.episodes} episodes...")
    start = time.time()
    rewards, costs = evaluate(agent, env, n_episodes=args.episodes, deterministic=args.deterministic, verbose=args.verbose)
    elapsed = time.time() - start

    summary = {
        'env': args.env_name,
        'checkpoint': args.checkpoint,
        'episodes': int(args.episodes),
        'reward_mean': float(np.mean(rewards)),
        'reward_std': float(np.std(rewards)),
        'cost_mean': float(np.mean(costs)),
        'cost_std': float(np.std(costs)),
        'time_s': float(elapsed)
    }

    # Also save per-episode arrays for detailed analysis
    summary['per_episode_rewards'] = [float(x) for x in rewards.tolist()]
    summary['per_episode_costs'] = [float(x) for x in costs.tolist()]

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.save_dir, f'iql_eval_pointbutton_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Evaluation complete:')
    print(json.dumps(summary, indent=2))
    print(f'Saved results to: {out_path}')

    env.close()


if __name__ == '__main__':
    main()
