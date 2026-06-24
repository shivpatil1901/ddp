import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import gym
    import safety_gym  # noqa: F401
except Exception as exc:  # pragma: no cover
    gym = None
    _ENV_IMPORT_ERROR = exc


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, fixed_std=True):
        super().__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        if fixed_std:
            self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=False)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs):
        mean = self.mean_net(obs)
        std = torch.exp(self.log_std)
        return mean, std


def load_policy(checkpoint_path, obs_dim, act_dim, device):
    policy = GaussianPolicy(obs_dim, act_dim, fixed_std=True).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_policy(policy, env_name, episodes, seed, device, results_path=None):
    if gym is None:
        raise RuntimeError(
            "gym/safety_gym could not be imported. Original error: %s" % _ENV_IMPORT_ERROR
        )

    env = gym.make(env_name)
    episode_rewards = []
    episode_costs = []
    episode_records = []

    for episode in range(episodes):
        if hasattr(env, "seed"):
            env.seed(seed + episode)
        obs = env.reset()

        done = False
        total_reward = 0.0
        total_cost = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = policy(obs_tensor)
            action = mean.squeeze(0).detach().cpu().numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            total_cost += float(info.get("cost", 0.0))

        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        episode_records.append(
            {
                "episode": episode + 1,
                "reward": total_reward,
                "cost": total_cost,
                "seed": seed + episode,
            }
        )
        print(f"Episode {episode + 1}: reward={total_reward:.2f} cost={total_cost:.2f}")

    env.close()
    reward_mean = float(np.mean(episode_rewards))
    reward_std = float(np.std(episode_rewards))
    cost_mean = float(np.mean(episode_costs))
    cost_std = float(np.std(episode_costs))

    print("\nSummary")
    print(f"Mean reward: {reward_mean:.2f} ± {reward_std:.2f}")
    print(f"Mean cost: {cost_mean:.2f} ± {cost_std:.2f}")

    if results_path is not None:
        results_path = Path(results_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "env": env_name,
            "episodes": episodes,
            "seed": seed,
            "device": str(device),
            "checkpoint": None,
            "episode_results": episode_records,
            "summary": {
                "mean_reward": reward_mean,
                "std_reward": reward_std,
                "mean_cost": cost_mean,
                "std_cost": cost_std,
            },
        }
        with results_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to: {results_path}")

    return reward_mean, cost_mean


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained BC policy in Safety Gym")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ed21b059/ddp/bc/models/bc_gaussian_policy_train_set_button.pth",
        help="Path to the saved BC checkpoint",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Safexp-PointButton1-v0",
        help="Safety Gym environment name",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--obs-dim",
        type=int,
        default=60,
        help="Observation dimension expected by the policy",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=2,
        help="Action dimension expected by the policy",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/home/ed21b059/ddp/bc/results",
        help="Directory where evaluation results will be saved",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initial device choice: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    policy = load_policy(str(checkpoint_path), args.obs_dim, args.act_dim, device)

    if device.type == "cuda":
        try:
            with torch.no_grad():
                probe = torch.zeros(1, args.obs_dim, device=device)
                policy(probe)
        except Exception as exc:
            print(f"CUDA probe failed, falling back to CPU: {exc}")
            device = torch.device("cpu")
            policy = load_policy(str(checkpoint_path), args.obs_dim, args.act_dim, device)

    print(f"Using device: {device}")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.results_dir) / f"bc_eval_{checkpoint_path.stem}_{args.env}_{timestamp}.json"
    evaluate_policy(policy, args.env, args.episodes, args.seed, device, results_path=results_path)


if __name__ == "__main__":
    main()