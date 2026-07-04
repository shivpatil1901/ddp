#!/usr/bin/env python
"""Evaluate a trained SafeDICE policy in Safety Gym and save reward/cost stats.

This script loads a SafeDICE checkpoint pickle, reconstructs the policy, runs
evaluation rollouts in the Safety Gym PointGoal environment, and saves a JSON
summary containing per-episode and aggregate reward/cost statistics.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

try:
	import gym
	import safety_gym  # noqa: F401
except Exception as exc:  # pragma: no cover
	gym = None
	_ENV_IMPORT_ERROR = exc


ROOT = Path(__file__).resolve().parent
SAFEDICE_PATH = ROOT
sys.path.insert(0, str(SAFEDICE_PATH))

from algorithms.safedice import SafeDICE as AntiDICE


def _load_pickle_compat(path: str) -> Any:
	try:
		with open(path, "rb") as f:
			return pickle.load(f)
	except ValueError as exc:
		if "unsupported pickle protocol" not in str(exc):
			raise
		try:
			import pickle5 as pickle5  # type: ignore[import-not-found]
		except Exception as import_exc:
			raise RuntimeError(
				"This checkpoint uses pickle protocol 5. Install pickle5 or use Python >= 3.8."
			) from import_exc
		with open(path, "rb") as f:
			return pickle5.load(f)


def _infer_model_dims(training_state: Dict[str, Any]):
	critic_params = training_state.get("critic_params", [])
	cost_params = training_state.get("cost_params", [])
	if not critic_params or not cost_params:
		raise ValueError("Checkpoint missing critic or cost parameters")

	state_dim = None
	cost_input_dim = None

	for name, param in critic_params:
		if "dense/kernel" in name or "mlp/dense" in name:
			state_dim = int(param.shape[0])
			break

	for name, param in cost_params:
		if "dense/kernel" in name or "mlp/dense" in name:
			cost_input_dim = int(param.shape[0])
			break

	if state_dim is None or cost_input_dim is None:
		raise ValueError("Could not infer state/action dims from SafeDICE weights")

	action_dim = int(cost_input_dim - state_dim)
	if action_dim <= 0:
		raise ValueError(f"Invalid inferred action_dim={action_dim}")

	return state_dim, action_dim


def _default_config():
	config = {
		"hidden_size": 256,
		"critic_lr": 1e-4,
		"actor_lr": 1e-5,
		"gamma": 0.99,
		"alpha": 0.0,
		"grad_reg_coeffs": (10, 1e-6),
		"use_last_layer_bias_cost": False,
		"use_last_layer_bias_critic": False,
		"kernel_initializer": "he_normal",
	}

	try:
		from config.safedice_config import hparams

		if hparams:
			config.update(hparams[0])
	except Exception:
		pass

	return config


def load_policy(weights_path: str):
	if not os.path.exists(weights_path):
		raise FileNotFoundError(f"SafeDICE weights not found: {weights_path}")

	print(f"Loading SafeDICE weights: {weights_path}")
	data = _load_pickle_compat(weights_path)
	training_state = data["training_state"]
	state_dim, action_dim = _infer_model_dims(training_state)
	config = _default_config()

	model = AntiDICE(
		state_dim=state_dim,
		action_dim=action_dim,
		mixture_actor=False,
		is_discrete_action=False,
		config=config,
	)
	model.set_training_state(training_state)
	print(f"SafeDICE inferred dims: state_dim={state_dim}, action_dim={action_dim}")
	return model, state_dim, action_dim


def evaluate_policy(model, env_name: str, episodes: int, seed: int, max_steps: int):
	if gym is None:
		raise RuntimeError(
			"gym/safety_gym could not be imported. Original error: %s" % _ENV_IMPORT_ERROR
		)

	env = gym.make(env_name)
	episode_rewards = []
	episode_costs = []
	episode_lengths = []
	episode_records = []

	for episode in range(episodes):
		if hasattr(env, "seed"):
			env.seed(seed + episode)

		obs = env.reset()
		done = False
		total_reward = 0.0
		total_cost = 0.0
		length = 0

		while not done and length < max_steps:
			obs_batch = np.asarray(obs, dtype=np.float32)[None, :]
			action = model.step(tf.convert_to_tensor(obs_batch, dtype=tf.float32), deterministic=True)
			action = action.numpy()[0]
			action = np.clip(action, env.action_space.low, env.action_space.high)

			obs, reward, done, info = env.step(action)
			total_reward += float(reward)
			total_cost += float(info.get("cost", 0.0))
			length += 1

		episode_rewards.append(total_reward)
		episode_costs.append(total_cost)
		episode_lengths.append(length)
		episode_records.append(
			{
				"episode": episode + 1,
				"seed": seed + episode,
				"reward": total_reward,
				"cost": total_cost,
				"length": length,
			}
		)
		print(f"Episode {episode + 1}/{episodes}: reward={total_reward:.2f}, cost={total_cost:.2f}, length={length}")

	env.close()

	reward_mean = float(np.mean(episode_rewards))
	reward_std = float(np.std(episode_rewards))
	cost_mean = float(np.mean(episode_costs))
	cost_std = float(np.std(episode_costs))
	length_mean = float(np.mean(episode_lengths))
	length_std = float(np.std(episode_lengths))

	print("\nSummary")
	print(f"Mean reward: {reward_mean:.2f} ± {reward_std:.2f}")
	print(f"Mean cost: {cost_mean:.2f} ± {cost_std:.2f}")
	print(f"Mean length: {length_mean:.2f} ± {length_std:.2f}")

	return {
		"mean_reward": reward_mean,
		"std_reward": reward_std,
		"mean_cost": cost_mean,
		"std_cost": cost_std,
		"mean_length": length_mean,
		"std_length": length_std,
		"episodes": episode_records,
	}


def parse_args():
	parser = argparse.ArgumentParser(description="Evaluate a trained SafeDICE policy in Safety Gym")
	parser.add_argument(
		"--weights",
		type=str,
		default="/home/ed21b059/ddp/SafeDICE/weights/antidice_PointButton1_seed0_20260424_095015_iter1000000.pickle",
		help="Path to the SafeDICE weights pickle",
	)
	parser.add_argument(
		"--env",
		type=str,
		default="Safexp-PointButton1-v0",
		help="Safety Gym environment name",
	)
	parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
	parser.add_argument(
		"--results-dir",
		type=str,
		default=str(ROOT / "results"),
		help="Directory where evaluation JSON will be saved",
	)
	return parser.parse_args()


def main():
	args = parse_args()
	weights_path = Path(args.weights)
	if not weights_path.exists():
		raise FileNotFoundError(f"Weights file not found: {weights_path}")

	device = tf.config.list_physical_devices("GPU")
	print(f"GPUs available: {len(device)}")
	model, _, _ = load_policy(str(weights_path))
	results = evaluate_policy(model, args.env, args.episodes, args.seed, args.max_steps)

	timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
	results_path = Path(args.results_dir) / f"safedice_eval_{weights_path.stem}_{args.env}_{timestamp}.json"
	results_path.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"timestamp_utc": datetime.utcnow().isoformat() + "Z",
		"weights": str(weights_path),
		"env": args.env,
		"episodes": args.episodes,
		"seed": args.seed,
		"max_steps": args.max_steps,
		"summary": {
			"mean_reward": results["mean_reward"],
			"std_reward": results["std_reward"],
			"mean_cost": results["mean_cost"],
			"std_cost": results["std_cost"],
			"mean_length": results["mean_length"],
			"std_length": results["std_length"],
		},
		"episode_results": results["episodes"],
	}
	with results_path.open("w") as f:
		json.dump(payload, f, indent=2)
	print(f"Saved results to: {results_path}")


if __name__ == "__main__":
	main()
