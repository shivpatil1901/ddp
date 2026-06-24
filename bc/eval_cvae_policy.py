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


class Encoder(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logstd_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        h = self.net(x)
        mean = self.mean_layer(h)
        logstd = self.logstd_layer(h)
        return mean, logstd


class Decoder(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def forward(self, state, latent):
        x = torch.cat([state, latent], dim=-1)
        return self.net(x)


class PriorNetwork(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logstd_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean_layer(h)
        logstd = self.logstd_layer(h)
        return mean, logstd


class CVAE(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.encoder = Encoder(obs_dim, act_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(obs_dim, act_dim, latent_dim, hidden_dim)
        self.prior = PriorNetwork(obs_dim, latent_dim, hidden_dim)
        self.latent_dim = latent_dim

    def reparameterize(self, mean, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mean + std * eps

    def sample_action(self, state, num_samples=1, use_mean=False):
        self.eval()
        with torch.no_grad():
            prior_mean, prior_logstd = self.prior(state)

            if use_mean:
                latent = prior_mean
            else:
                latent = self.reparameterize(prior_mean, prior_logstd)

            if num_samples > 1:
                state_expanded = state.repeat(num_samples, 1)
                latent_samples = self.reparameterize(
                    prior_mean.repeat(num_samples, 1),
                    prior_logstd.repeat(num_samples, 1),
                )
                return self.decoder(state_expanded, latent_samples)

            return self.decoder(state, latent)


def load_stats(stats_path):
    stats = np.load(stats_path)
    return {
        "obs_mean": stats["obs_mean"].astype(np.float32),
        "obs_std": stats["obs_std"].astype(np.float32),
        "action_min": float(np.asarray(stats["action_min"]).reshape(-1)[0]),
        "action_max": float(np.asarray(stats["action_max"]).reshape(-1)[0]),
        "safety_margin": float(np.asarray(stats["safety_margin"]).reshape(-1)[0]),
        "obs_dim": int(np.asarray(stats["obs_dim"]).reshape(-1)[0]),
        "act_dim": int(np.asarray(stats["act_dim"]).reshape(-1)[0]),
        "latent_dim": int(np.asarray(stats["latent_dim"]).reshape(-1)[0]),
        "hidden_dim": int(np.asarray(stats["hidden_dim"]).reshape(-1)[0]) if "hidden_dim" in stats else 512,
    }


def inverse_linear_normalize(normalized_actions, min_val, max_val, safety_margin):
    unshrunk = normalized_actions / (1.0 - safety_margin)
    unscaled_01 = (unshrunk + 1.0) / 2.0
    denom = max_val - min_val + 1e-8
    return unscaled_01 * denom + min_val


def load_policy(checkpoint_path, stats, device):
    policy = CVAE(
        stats["obs_dim"],
        stats["act_dim"],
        latent_dim=stats["latent_dim"],
        hidden_dim=stats["hidden_dim"],
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def evaluate_policy(policy, stats, env_name, episodes, seed, device, use_mean=True, results_path=None):
    if gym is None:
        raise RuntimeError(
            "gym/safety_gym could not be imported. Original error: %s" % _ENV_IMPORT_ERROR
        )

    env = gym.make(env_name)
    episode_rewards = []
    episode_costs = []
    episode_records = []

    obs_mean = stats["obs_mean"]
    obs_std = stats["obs_std"]
    action_min = stats["action_min"]
    action_max = stats["action_max"]
    safety_margin = stats["safety_margin"]

    for episode in range(episodes):
        if hasattr(env, "seed"):
            env.seed(seed + episode)
        obs = env.reset()

        done = False
        total_reward = 0.0
        total_cost = 0.0

        while not done:
            obs_normalized = (obs - obs_mean) / obs_std
            obs_tensor = torch.tensor(obs_normalized, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action_normalized = policy.sample_action(obs_tensor, use_mean=use_mean)
            action_normalized = action_normalized.squeeze(0).detach().cpu().numpy()
            action = inverse_linear_normalize(action_normalized, action_min, action_max, safety_margin)
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
        mode = "mean" if use_mean else "sampled"
        print(f"Episode {episode + 1}: reward={total_reward:.2f} cost={total_cost:.2f} mode={mode}")

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
            "mode": "mean" if use_mean else "sampled",
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
    parser = argparse.ArgumentParser(description="Evaluate a trained CVAE policy in Safety Gym")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/ed21b059/ddp/bc_cvae_policy_button.pth",
        help="Path to the saved CVAE checkpoint",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="/home/ed21b059/ddp/bc_cvae_policy_stats_button.npz",
        help="Path to the saved normalization stats",
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
        "--results-dir",
        type=str,
        default="/home/ed21b059/ddp/bc/results",
        help="Directory where evaluation results will be saved",
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Use sampled latent actions instead of the prior mean",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    stats_path = Path(args.stats)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initial device choice: {device}")
    stats = load_stats(str(stats_path))
    print(f"Loading checkpoint: {checkpoint_path}")
    policy = load_policy(str(checkpoint_path), stats, device)

    if device.type == "cuda":
        try:
            with torch.no_grad():
                probe = torch.zeros(1, stats["obs_dim"], device=device)
                policy.sample_action(probe, use_mean=True)
        except Exception as exc:
            print(f"CUDA probe failed, falling back to CPU: {exc}")
            device = torch.device("cpu")
            policy = load_policy(str(checkpoint_path), stats, device)

    print(f"Using device: {device}")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    results_path = Path(args.results_dir) / f"cvae_eval_{checkpoint_path.stem}_{args.env}_{timestamp}.json"
    evaluate_policy(
        policy,
        stats,
        args.env,
        args.episodes,
        args.seed,
        device,
        use_mean=not args.sampled,
        results_path=results_path,
    )


if __name__ == "__main__":
    main()
