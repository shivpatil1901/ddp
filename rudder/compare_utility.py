#!/usr/bin/env python
"""Compare reinforce RUDDER signals against reward-cost utility.

This script loads the trained ``reinforce_rudder_combined.pt`` checkpoint and a
holdout trajectory dataset, then checks whether the model's redistributed signal
tracks an environment utility of the form ``reward - lambda * cost``.

The experiment reports:
- trajectory-level Pearson and Spearman correlations for a sweep of candidate
  lambda values
- the best lambda under the sweep
- step-level correlation between redistributed signal and immediate utility
- optional plots and JSON/NPZ summaries

Example:
	python rudder/compare_utility.py \
		--checkpoint /home/ed21b059/ddp/rudder/models/reinforce_rudder_combined.pt \
		--dataset /home/ed21b059/ddp/rudder/dataset/combined_cost_reward_holdout_200.pkl \
		--output_dir rudder/eval/utility_comparison
"""

import argparse
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
	import pickle5 as pickle  # type: ignore[import-not-found]
except ImportError:
	import pickle

try:
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	HAS_MPL = True
except Exception:
	HAS_MPL = False


def _log(msg: str) -> None:
	ts = datetime.now().strftime("%H:%M:%S")
	print("[%s] %s" % (ts, msg), flush=True)


def _repo_root() -> str:
	return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
	raw = os.path.expanduser(os.path.expandvars(path))
	normalized = raw.replace("\\", os.sep).replace("/", os.sep)

	if os.path.isabs(normalized):
		return normalized

	candidates = [
		os.path.abspath(normalized),
		os.path.abspath(os.path.join(_repo_root(), normalized)),
		os.path.abspath(os.path.join(_repo_root(), "rudder", normalized)),
		os.path.abspath(os.path.join(_repo_root(), "rudder", "dataset", normalized)),
		os.path.abspath(os.path.join(_repo_root(), "rudder", "models", normalized)),
	]

	for candidate in candidates:
		if os.path.exists(candidate):
			return candidate

	return candidates[0]


def _resolve_existing_path(path: str) -> str:
	resolved = _resolve_path(path)
	if not os.path.isfile(resolved):
		raise FileNotFoundError("File not found: %s" % resolved)
	return resolved


def _load_payload(path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
	resolved = _resolve_existing_path(path)
	_log("Loading dataset: %s" % resolved)
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


def _pick_first(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
	for key in keys:
		if key in mapping:
			return mapping[key]
	return None


def _as_1d(arr: Any) -> np.ndarray:
	return np.asarray(arr, dtype=np.float32).reshape(-1)


def _scalar_label(value: Any) -> float:
	arr = np.asarray(value).reshape(-1)
	if arr.size == 0:
		return float("nan")
	return float(arr[0])


def _extract_core_arrays(traj: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	states = np.asarray(traj["states"], dtype=np.float32)
	actions = np.asarray(traj["actions"], dtype=np.float32)
	rewards = _as_1d(_pick_first(traj, ("rewards", "reward", "rews", "r", "env_rewards")))
	costs = _as_1d(_pick_first(traj, ("costs", "cost", "c")))

	if rewards.size == 0:
		rewards = np.zeros((len(states),), dtype=np.float32)
	if costs.size == 0:
		costs = np.zeros((len(states),), dtype=np.float32)

	n = int(min(len(states), len(actions), len(rewards), len(costs)))
	if n <= 0:
		raise ValueError("Encountered empty trajectory")

	return states[:n], actions[:n], rewards[:n], costs[:n]


def _trajectory_sums(traj: Dict[str, Any]) -> Tuple[float, float]:
	_, _, rewards, costs = _extract_core_arrays(traj)
	return float(np.sum(rewards)), float(np.sum(costs))


def _trajectory_preference_label(
	traj: Dict[str, Any],
	reward_sum: float,
	cost_sum: float,
	reward_threshold: float,
	cost_threshold: float,
	preferred_source_values: List[int],
) -> int:
	if "preference_label" in traj:
		val = _scalar_label(traj["preference_label"])
		if not np.isnan(val):
			return int(1 if val > 0.5 else 0)

	if "source" in traj:
		src = int(round(_scalar_label(traj["source"])))
		return int(1 if src in preferred_source_values else 0)

	return int(1 if (reward_sum > reward_threshold and cost_sum < cost_threshold) else 0)


def _pack_sa(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
	return np.concatenate([states, actions], axis=-1).astype(np.float32)


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
	x = np.asarray(x, dtype=np.float64).reshape(-1)
	order = np.argsort(x)
	ranks = np.empty_like(order, dtype=np.float64)

	i = 0
	while i < len(order):
		j = i + 1
		while j < len(order) and x[order[j]] == x[order[i]]:
			j += 1
		avg_rank = 0.5 * (i + j - 1) + 1.0
		ranks[order[i:j]] = avg_rank
		i = j
	return ranks


def _pearsonr(x: np.ndarray, y: np.ndarray) -> float:
	x = np.asarray(x, dtype=np.float64).reshape(-1)
	y = np.asarray(y, dtype=np.float64).reshape(-1)
	if len(x) != len(y) or len(x) == 0:
		return 0.0

	x = x - float(np.mean(x))
	y = y - float(np.mean(y))
	denom = float(np.sqrt(np.sum(x ** 2) * np.sum(y ** 2)))
	if denom <= 1e-12:
		return 0.0
	return float(np.sum(x * y) / denom)


def _spearmanr(x: np.ndarray, y: np.ndarray) -> float:
	return _pearsonr(_rankdata_average_ties(x), _rankdata_average_ties(y))


def _kendall_tau_b(x: np.ndarray, y: np.ndarray) -> float:
	x = np.asarray(x, dtype=np.float64).reshape(-1)
	y = np.asarray(y, dtype=np.float64).reshape(-1)
	if len(x) != len(y) or len(x) < 2:
		return 0.0

	concordant = 0.0
	discordant = 0.0
	tie_x = 0.0
	tie_y = 0.0

	for i in range(len(x) - 1):
		dx = x[i + 1:] - x[i]
		dy = y[i + 1:] - y[i]
		prod = dx * dy
		concordant += float(np.sum(prod > 0.0))
		discordant += float(np.sum(prod < 0.0))
		tie_x += float(np.sum((dx == 0.0) & (dy != 0.0)))
		tie_y += float(np.sum((dx != 0.0) & (dy == 0.0)))

	total = concordant + discordant
	denom = float(np.sqrt((total + tie_x) * (total + tie_y)))
	if denom <= 1e-12:
		return 0.0
	return float((concordant - discordant) / denom)


def _pairwise_order_agreement(x: np.ndarray, y: np.ndarray) -> float:
	x = np.asarray(x, dtype=np.float64).reshape(-1)
	y = np.asarray(y, dtype=np.float64).reshape(-1)
	if len(x) != len(y) or len(x) < 2:
		return 0.0

	concordant = 0.0
	discordant = 0.0

	for i in range(len(x) - 1):
		dx = x[i + 1:] - x[i]
		dy = y[i + 1:] - y[i]
		mask = (dx != 0.0) & (dy != 0.0)
		if not np.any(mask):
			continue
		concordant += float(np.sum((dx[mask] > 0.0) == (dy[mask] > 0.0)))
		discordant += float(np.sum((dx[mask] > 0.0) != (dy[mask] > 0.0)))

	total = concordant + discordant
	if total <= 0.0:
		return 0.0
	return float(concordant / total)


def _safe_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
	try:
		sol, *_ = np.linalg.lstsq(x, y, rcond=None)
		return sol
	except np.linalg.LinAlgError:
		return np.zeros((x.shape[1],), dtype=np.float64)


class CombinedRUDDER(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64, dropout: float = 0.2) -> None:
		super().__init__()
		self.input_dim = state_dim + action_dim
		self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
		self.layer_norm = nn.LayerNorm(hidden_dim)
		self.dropout = nn.Dropout(dropout)
		self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		lstm_out, _ = self.lstm(x)
		norm_out = self.layer_norm(lstm_out)
		drop_out = self.dropout(norm_out)
		return self.output_layer(drop_out)


class NoLayerNormRUDDER(nn.Module):
	def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
		super().__init__()
		self.input_dim = state_dim + action_dim
		self.lstm = nn.LSTM(self.input_dim, hidden_dim, batch_first=True)
		self.output_layer = nn.Linear(hidden_dim, 1, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		lstm_out, _ = self.lstm(x)
		return self.output_layer(lstm_out)


def _infer_model(ckpt: Dict[str, Any], device: str) -> Tuple[nn.Module, float, int, int, int]:
	if "model_state_dict" not in ckpt:
		raise KeyError("Checkpoint missing model_state_dict")

	model_state = ckpt["model_state_dict"]
	state_dim = int(ckpt["state_dim"])
	action_dim = int(ckpt["action_dim"])
	seq_len = int(ckpt.get("seq_len", 0))
	baseline = float(ckpt.get("baseline", 0.0))

	hidden_dim = int(model_state["lstm.weight_ih_l0"].shape[0] // 4)
	has_layer_norm = ("layer_norm.weight" in model_state) and ("layer_norm.bias" in model_state)

	if has_layer_norm:
		model = CombinedRUDDER(state_dim, action_dim, hidden_dim=hidden_dim, dropout=float(ckpt.get("dropout", 0.2)))
	else:
		model = NoLayerNormRUDDER(state_dim, action_dim, hidden_dim=hidden_dim)

	model.load_state_dict(model_state, strict=True)
	model.to(device)
	model.eval()
	return model, baseline, state_dim, action_dim, seq_len


def _sequence_signal(
	model: nn.Module,
	sa_seq: np.ndarray,
	seq_len: int,
	device: str,
) -> np.ndarray:
	feature_dim = int(sa_seq.shape[-1])
	x = np.zeros((1, seq_len, feature_dim), dtype=np.float32)
	n = int(min(len(sa_seq), seq_len))
	x[0, :n] = sa_seq[:n]

	with torch.no_grad():
		pred = model(torch.from_numpy(x).to(device))[0, :, 0].detach().cpu().numpy()

	return pred.astype(np.float32)


def _redistribute_signal(step_scores: np.ndarray) -> np.ndarray:
	step_scores = np.asarray(step_scores, dtype=np.float32).reshape(-1)
	if len(step_scores) == 0:
		return step_scores

	redistributed = np.zeros_like(step_scores)
	redistributed[0] = step_scores[0]
	if len(step_scores) > 1:
		redistributed[1:] = step_scores[1:] - step_scores[:-1]
	return redistributed


def _flatten_valid(values: List[np.ndarray]) -> np.ndarray:
	if len(values) == 0:
		return np.zeros((0,), dtype=np.float32)
	return np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in values], axis=0)


def _flatten_selected(values: List[np.ndarray], selected_indices: np.ndarray) -> np.ndarray:
	if len(values) == 0 or len(selected_indices) == 0:
		return np.zeros((0,), dtype=np.float32)
	selected = [np.asarray(values[int(i)], dtype=np.float32).reshape(-1) for i in selected_indices.tolist()]
	if len(selected) == 0:
		return np.zeros((0,), dtype=np.float32)
	return np.concatenate(selected, axis=0)


def _lambda_grid(lambda_min: float, lambda_max: float, num: int, include_zero: bool) -> np.ndarray:
	if num <= 1:
		grid = np.array([float(lambda_min)], dtype=np.float64)
	else:
		if lambda_min > 0.0 and lambda_max > 0.0:
			grid = np.logspace(np.log10(lambda_min), np.log10(lambda_max), num=num, dtype=np.float64)
		else:
			grid = np.linspace(lambda_min, lambda_max, num=num, dtype=np.float64)

	if include_zero and not np.any(np.isclose(grid, 0.0)):
		grid = np.concatenate([np.array([0.0], dtype=np.float64), grid])

	grid = np.unique(grid)
	grid.sort()
	return grid


def _trajectory_summary(trajs: List[Dict[str, Any]]) -> Dict[str, float]:
	if len(trajs) == 0:
		return {
			"count": 0,
			"reward_min": float("nan"),
			"reward_mean": float("nan"),
			"reward_max": float("nan"),
			"cost_min": float("nan"),
			"cost_mean": float("nan"),
			"cost_max": float("nan"),
			"length_min": float("nan"),
			"length_mean": float("nan"),
			"length_max": float("nan"),
		}

	rewards = []
	costs = []
	lengths = []
	for traj in trajs:
		reward_sum, cost_sum = _trajectory_sums(traj)
		states, actions, _, _ = _extract_core_arrays(traj)
		rewards.append(reward_sum)
		costs.append(cost_sum)
		lengths.append(int(min(len(states), len(actions))))

	rewards_np = np.asarray(rewards, dtype=np.float32)
	costs_np = np.asarray(costs, dtype=np.float32)
	lengths_np = np.asarray(lengths, dtype=np.float32)
	return {
		"count": int(len(trajs)),
		"reward_min": float(rewards_np.min()),
		"reward_mean": float(rewards_np.mean()),
		"reward_max": float(rewards_np.max()),
		"cost_min": float(costs_np.min()),
		"cost_mean": float(costs_np.mean()),
		"cost_max": float(costs_np.max()),
		"length_min": float(lengths_np.min()),
		"length_mean": float(lengths_np.mean()),
		"length_max": float(lengths_np.max()),
	}


def _analyze_lambda_sweep(
	model_scores: np.ndarray,
	reward_sums: np.ndarray,
	cost_sums: np.ndarray,
	lambda_values: np.ndarray,
) -> Dict[str, Any]:
	rows: List[Dict[str, float]] = []
	best_idx = 0
	best_rho = float("-inf")

	for idx, lam in enumerate(lambda_values):
		utility = reward_sums - float(lam) * cost_sums
		pearson = _pearsonr(model_scores, utility)
		spearman = _spearmanr(model_scores, utility)
		kendall = _kendall_tau_b(model_scores, utility)
		pairwise = _pairwise_order_agreement(model_scores, utility)
		rows.append(
			{
				"lambda": float(lam),
				"pearson": float(pearson),
				"spearman": float(spearman),
				"kendall_tau_b": float(kendall),
				"pairwise_order_agreement": float(pairwise),
				"utility_min": float(np.min(utility)),
				"utility_mean": float(np.mean(utility)),
				"utility_max": float(np.max(utility)),
			}
		)

		if spearman > best_rho:
			best_rho = float(spearman)
			best_idx = int(idx)

	best = rows[best_idx]
	best_kendall_idx = int(np.argmax([row["kendall_tau_b"] for row in rows]))
	best_kendall = rows[best_kendall_idx]
	return {
		"sweep": rows,
		"best": best,
		"best_index": int(best_idx),
		"best_lambda": float(best["lambda"]),
		"best_spearman": float(best["spearman"]),
		"best_pearson": float(best["pearson"]),
		"best_kendall_index": int(best_kendall_idx),
		"best_kendall_lambda": float(best_kendall["lambda"]),
		"best_kendall_tau_b": float(best_kendall["kendall_tau_b"]),
		"best_kendall_pairwise_order_agreement": float(best_kendall["pairwise_order_agreement"]),
	}


def _analyze_stepwise_alignment(
	step_scores_flat: np.ndarray,
	rewards_flat: np.ndarray,
	costs_flat: np.ndarray,
	lambda_value: float,
) -> Dict[str, float]:
	utility_steps = rewards_flat - float(lambda_value) * costs_flat
	return {
		"lambda": float(lambda_value),
		"pearson": float(_pearsonr(step_scores_flat, utility_steps)),
		"spearman": float(_spearmanr(step_scores_flat, utility_steps)),
		"kendall_tau_b": float(_kendall_tau_b(step_scores_flat, utility_steps)),
		"pairwise_order_agreement": float(_pairwise_order_agreement(step_scores_flat, utility_steps)),
		"step_score_mean": float(np.mean(step_scores_flat)) if len(step_scores_flat) > 0 else float("nan"),
		"utility_step_mean": float(np.mean(utility_steps)) if len(utility_steps) > 0 else float("nan"),
	}


def _estimate_implied_lambda(reward_sums: np.ndarray, cost_sums: np.ndarray, model_scores: np.ndarray) -> Dict[str, float]:
	x = np.stack([reward_sums.astype(np.float64), -cost_sums.astype(np.float64), np.ones_like(reward_sums, dtype=np.float64)], axis=1)
	y = model_scores.astype(np.float64)
	coef = _safe_lstsq(x, y)
	reward_coef = float(coef[0])
	cost_coef = float(coef[1])
	bias = float(coef[2])

	implied_lambda = float("nan")
	if abs(reward_coef) > 1e-12:
		implied_lambda = float(cost_coef / reward_coef)

	return {
		"reward_coef": reward_coef,
		"cost_coef": cost_coef,
		"bias": bias,
		"implied_lambda": implied_lambda,
	}


def _save_plots(out_dir: str, sweep: Dict[str, Any], trajectory_stats: Dict[str, Any], lambda_best: float) -> None:
	if not HAS_MPL:
		return

	lambdas = np.asarray([row["lambda"] for row in sweep["sweep"]], dtype=np.float64)
	pearsons = np.asarray([row["pearson"] for row in sweep["sweep"]], dtype=np.float64)
	spearmans = np.asarray([row["spearman"] for row in sweep["sweep"]], dtype=np.float64)

	fig, axes = plt.subplots(1, 2, figsize=(14, 5))
	fig.suptitle("RUDDER vs Utility Sweep", fontsize=14)

	axes[0].plot(lambdas, pearsons, label="Pearson", linewidth=2)
	axes[0].plot(lambdas, spearmans, label="Spearman", linewidth=2)
	axes[0].axvline(lambda_best, color="black", linestyle="--", linewidth=1, label="best lambda")
	axes[0].set_xlabel("lambda")
	axes[0].set_ylabel("correlation")
	axes[0].set_title("Trajectory-level correlation sweep")
	axes[0].legend()
	axes[0].grid(alpha=0.3)

	rewards = float(trajectory_stats["reward_mean"])
	costs = float(trajectory_stats["cost_mean"])
	axes[1].bar(["mean reward", "mean cost"], [rewards, costs], color=["#2b8cbe", "#de2d26"])
	axes[1].set_title("Holdout summary")
	axes[1].grid(axis="y", alpha=0.3)

	fig.tight_layout()
	fig.savefig(os.path.join(out_dir, "utility_sweep.png"), dpi=200)
	plt.close(fig)


def _save_stepwise_progress_samples(
	out_dir: str,
	step_scores_all: List[np.ndarray],
	reward_steps_all: List[np.ndarray],
	cost_steps_all: List[np.ndarray],
	lambda_best: float,
	preferred_indices: np.ndarray,
	nonpreferred_indices: np.ndarray,
	preferred_count: int,
	nonpreferred_count: int,
	sample_seed: int,
) -> str:
	if not HAS_MPL:
		return ""

	n = int(min(len(step_scores_all), len(reward_steps_all), len(cost_steps_all)))
	if n <= 0:
		return ""

	valid_idx = []
	for i in range(n):
		if len(step_scores_all[i]) > 1 and len(reward_steps_all[i]) > 1 and len(cost_steps_all[i]) > 1:
			valid_idx.append(i)

	if len(valid_idx) == 0:
		return ""

	rng = np.random.RandomState(int(sample_seed))
	valid_set = set(int(i) for i in valid_idx)
	preferred_pool = [int(i) for i in preferred_indices.tolist() if int(i) in valid_set]
	nonpreferred_pool = [int(i) for i in nonpreferred_indices.tolist() if int(i) in valid_set]

	k_pref = int(max(0, min(int(preferred_count), len(preferred_pool))))
	k_nonpref = int(max(0, min(int(nonpreferred_count), len(nonpreferred_pool))))

	chosen_pref = []
	chosen_nonpref = []
	if k_pref > 0:
		chosen_pref = rng.choice(np.asarray(preferred_pool, dtype=np.int64), size=k_pref, replace=False).tolist()
	if k_nonpref > 0:
		chosen_nonpref = rng.choice(np.asarray(nonpreferred_pool, dtype=np.int64), size=k_nonpref, replace=False).tolist()

	chosen_with_group: List[Tuple[int, str]] = []
	for idx in chosen_pref:
		chosen_with_group.append((int(idx), "P"))
	for idx in chosen_nonpref:
		chosen_with_group.append((int(idx), "NP"))

	if len(chosen_with_group) == 0:
		return ""

	n_cols = 2
	k = len(chosen_with_group)
	n_rows = int(math.ceil(float(k) / float(n_cols)))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
	if not isinstance(axes, np.ndarray):
		axes = np.asarray([axes])
	axes_flat = axes.reshape(-1)

	for p, item in enumerate(chosen_with_group):
		traj_idx, group_tag = item
		ax = axes_flat[p]
		model_steps = np.asarray(step_scores_all[int(traj_idx)], dtype=np.float32).reshape(-1)
		env_steps = (
			np.asarray(reward_steps_all[int(traj_idx)], dtype=np.float32).reshape(-1)
			- float(lambda_best) * np.asarray(cost_steps_all[int(traj_idx)], dtype=np.float32).reshape(-1)
		)
		m = int(min(len(model_steps), len(env_steps)))
		if m <= 1:
			ax.set_title("Trajectory %d | insufficient steps" % int(traj_idx))
			ax.axis("off")
			continue

		model_steps = model_steps[:m]
		env_steps = env_steps[:m]
		steps = np.arange(m, dtype=np.int64)
		model_norm = model_steps - float(np.mean(model_steps))
		env_norm = env_steps - float(np.mean(env_steps))
		model_scale = float(np.std(model_norm))
		env_scale = float(np.std(env_norm))
		if model_scale > 1e-12:
			model_norm = model_norm / model_scale
		if env_scale > 1e-12:
			env_norm = env_norm / env_scale

		ax.plot(steps, model_norm, color="#2b8cbe", linewidth=2, label="model redistributed signal")
		ax.plot(steps, env_norm, color="#de2d26", linewidth=2, label="env utility (best lambda)")
		ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
		ax.set_title(
			"Trajectory %d (%s) | steps=%d | step Pearson=%.3f" % (
				int(traj_idx),
				group_tag,
				int(m),
				float(_pearsonr(model_steps, env_steps)),
			)
		)
		ax.set_xlabel("Time step")
		ax.set_ylabel("Normalized step signal")
		ax.grid(alpha=0.25)
		if p == 0:
			ax.legend(fontsize=9)

	for j in range(k, len(axes_flat)):
		axes_flat[j].axis("off")

	fig.suptitle(
		"Step-wise Progression (best lambda=%.4f, sampled trajectories=%d | P=%d, NP=%d)" % (
			float(lambda_best),
			int(k),
			int(len(chosen_pref)),
			int(len(chosen_nonpref)),
		),
		fontsize=13,
	)
	fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
	out_path = os.path.join(out_dir, "stepwise_progress_samples.png")
	fig.savefig(out_path, dpi=200)
	plt.close(fig)
	return out_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare RUDDER signals with reward-cost utility")
	parser.add_argument(
		"--checkpoint",
		type=str,
		default="/home/ed21b059/ddp/rudder/models/reinforce_rudder_combined.pt",
		help="Path to the trained RUDDER checkpoint",
	)
	parser.add_argument(
		"--dataset",
		type=str,
		default="/home/ed21b059/ddp/rudder/dataset/combined_cost_reward_holdout_200.pkl",
		help="Holdout dataset for comparison",
	)
	parser.add_argument("--output_dir", type=str, default="/home/ed21b059/ddp/rudder/eval/utility_comparison")
	parser.add_argument("--device", type=str, default="cpu")
	parser.add_argument("--lambda_min", type=float, default=1e-3)
	parser.add_argument("--lambda_max", type=float, default=1e2)
	parser.add_argument("--lambda_num", type=int, default=80)
	parser.add_argument("--include_zero_lambda", action="store_true", help="Include lambda=0 in the sweep")
	parser.add_argument("--plot_trajectory_count", type=int, default=8, help="Total sample budget for step-wise progression plot (split across preferred/non-preferred if specific counts are not provided)")
	parser.add_argument("--plot_preferred_count", type=int, default=None, help="Preferred trajectories to sample for progression plot")
	parser.add_argument("--plot_nonpreferred_count", type=int, default=None, help="Non-preferred trajectories to sample for progression plot")
	parser.add_argument("--plot_seed", type=int, default=0, help="Random seed for sampled trajectory plots")
	parser.add_argument(
		"--score_mode",
		type=str,
		default="prefix_final",
		choices=("prefix_final", "padded_final"),
		help="Use the final score at the real trajectory end or at the padded sequence end",
	)
	args = parser.parse_args()

	checkpoint_path = _resolve_existing_path(args.checkpoint)
	dataset_path = _resolve_existing_path(args.dataset)
	out_dir = _resolve_path(args.output_dir)
	os.makedirs(out_dir, exist_ok=True)

	_log("Loading checkpoint: %s" % checkpoint_path)
	ckpt = torch.load(checkpoint_path, map_location=args.device)
	model, baseline, state_dim, action_dim, seq_len_ckpt = _infer_model(ckpt, args.device)
	reward_threshold = float(ckpt.get("reward_threshold", 15.0))
	cost_threshold = float(ckpt.get("cost_threshold", 25.0))
	preferred_source_values = [int(v) for v in ckpt.get("preferred_source_values", [1, 3])]

	trajectories, metadata, resolved_dataset = _load_payload(dataset_path)

	if seq_len_ckpt <= 0:
		seq_len = int(max(min(len(np.asarray(t["states"])), len(np.asarray(t["actions"]))) for t in trajectories))
	else:
		seq_len = int(seq_len_ckpt)

	trajectory_stats = _trajectory_summary(trajectories)
	lambda_values = _lambda_grid(float(args.lambda_min), float(args.lambda_max), int(args.lambda_num), bool(args.include_zero_lambda))

	_log("Checkpoint: %s" % checkpoint_path)
	_log("Dataset: %s" % resolved_dataset)
	_log("state_dim=%d action_dim=%d seq_len=%d baseline=%.6f" % (state_dim, action_dim, seq_len, baseline))
	_log("Holdout trajectories: %d" % len(trajectories))

	reward_sums = np.zeros((len(trajectories),), dtype=np.float32)
	cost_sums = np.zeros((len(trajectories),), dtype=np.float32)
	model_scores = np.zeros((len(trajectories),), dtype=np.float32)
	step_scores_all: List[np.ndarray] = []
	reward_steps_all: List[np.ndarray] = []
	cost_steps_all: List[np.ndarray] = []
	preferred_labels = np.zeros((len(trajectories),), dtype=np.int32)
	lengths = []

	for i, traj in enumerate(trajectories):
		states, actions, rewards, costs = _extract_core_arrays(traj)
		sa = _pack_sa(states, actions)
		full_scores = _sequence_signal(model, sa, seq_len=seq_len, device=args.device)
		valid_n = int(min(len(sa), len(full_scores)))
		prefix_scores = full_scores[:valid_n]
		redistributed = _redistribute_signal(prefix_scores)

		reward_sums[i] = float(np.sum(rewards[:valid_n]))
		cost_sums[i] = float(np.sum(costs[:valid_n]))
		preferred_labels[i] = _trajectory_preference_label(
			traj=traj,
			reward_sum=float(reward_sums[i]),
			cost_sum=float(cost_sums[i]),
			reward_threshold=reward_threshold,
			cost_threshold=cost_threshold,
			preferred_source_values=preferred_source_values,
		)
		if args.score_mode == "padded_final":
			model_scores[i] = float(full_scores[-1]) if len(full_scores) > 0 else 0.0
		else:
			model_scores[i] = float(prefix_scores[-1]) if len(prefix_scores) > 0 else 0.0

		step_scores_all.append(redistributed)
		reward_steps_all.append(rewards[: len(redistributed)])
		cost_steps_all.append(costs[: len(redistributed)])
		lengths.append(int(len(redistributed)))

	sweep = _analyze_lambda_sweep(model_scores, reward_sums, cost_sums, lambda_values)
	best_lambda = float(sweep["best_lambda"])
	preferred_indices = np.where(preferred_labels == 1)[0]
	nonpreferred_indices = np.where(preferred_labels == 0)[0]

	if args.plot_preferred_count is None and args.plot_nonpreferred_count is None:
		plot_pref_count = int(max(0, int(args.plot_trajectory_count) // 2))
		plot_nonpref_count = int(max(0, int(args.plot_trajectory_count) - plot_pref_count))
	else:
		plot_pref_count = int(max(0, 0 if args.plot_preferred_count is None else int(args.plot_preferred_count)))
		plot_nonpref_count = int(max(0, 0 if args.plot_nonpreferred_count is None else int(args.plot_nonpreferred_count)))

	step_scores_flat = _flatten_valid(step_scores_all)
	reward_steps_flat = _flatten_valid(reward_steps_all)
	cost_steps_flat = _flatten_valid(cost_steps_all)
	stepwise_best = _analyze_stepwise_alignment(step_scores_flat, reward_steps_flat, cost_steps_flat, best_lambda)

	preferred_step_scores = _flatten_selected(step_scores_all, preferred_indices)
	preferred_reward_steps = _flatten_selected(reward_steps_all, preferred_indices)
	preferred_cost_steps = _flatten_selected(cost_steps_all, preferred_indices)
	nonpreferred_step_scores = _flatten_selected(step_scores_all, nonpreferred_indices)
	nonpreferred_reward_steps = _flatten_selected(reward_steps_all, nonpreferred_indices)
	nonpreferred_cost_steps = _flatten_selected(cost_steps_all, nonpreferred_indices)

	stepwise_preferred = _analyze_stepwise_alignment(
		preferred_step_scores,
		preferred_reward_steps,
		preferred_cost_steps,
		best_lambda,
	)
	stepwise_nonpreferred = _analyze_stepwise_alignment(
		nonpreferred_step_scores,
		nonpreferred_reward_steps,
		nonpreferred_cost_steps,
		best_lambda,
	)
	implied_lambda = _estimate_implied_lambda(reward_sums, cost_sums, model_scores)

	utility_best = reward_sums - best_lambda * cost_sums
	final_pearson = _pearsonr(model_scores, utility_best)
	final_spearman = _spearmanr(model_scores, utility_best)
	final_kendall = _kendall_tau_b(model_scores, utility_best)
	final_pairwise = _pairwise_order_agreement(model_scores, utility_best)
	best_kendall_lambda = float(sweep["best_kendall_lambda"])
	utility_best_kendall = reward_sums - best_kendall_lambda * cost_sums
	final_kendall_at_best = _kendall_tau_b(model_scores, utility_best_kendall)
	final_pairwise_at_best = _pairwise_order_agreement(model_scores, utility_best_kendall)

	results: Dict[str, Any] = {
		"checkpoint": checkpoint_path,
		"dataset": resolved_dataset,
		"dataset_metadata": metadata,
		"state_dim": int(state_dim),
		"action_dim": int(action_dim),
		"seq_len": int(seq_len),
		"baseline": float(baseline),
		"score_mode": args.score_mode,
		"holdout_summary": trajectory_stats,
		"trajectory_level": {
			"best_lambda": best_lambda,
			"best_kendall_lambda": best_kendall_lambda,
			"pearson": float(final_pearson),
			"spearman": float(final_spearman),
			"kendall_tau_b": float(final_kendall),
			"pairwise_order_agreement": float(final_pairwise),
			"kendall_tau_b_at_best_kendall_lambda": float(final_kendall_at_best),
			"pairwise_order_agreement_at_best_kendall_lambda": float(final_pairwise_at_best),
			"model_score_min": float(np.min(model_scores)) if len(model_scores) > 0 else float("nan"),
			"model_score_mean": float(np.mean(model_scores)) if len(model_scores) > 0 else float("nan"),
			"model_score_max": float(np.max(model_scores)) if len(model_scores) > 0 else float("nan"),
		},
		"lambda_sweep": sweep,
		"stepwise_alignment": stepwise_best,
		"stepwise_alignment_by_group": {
			"combined": dict(stepwise_best, num_trajectories=int(len(trajectories)), num_steps=int(len(step_scores_flat))),
			"preferred": dict(stepwise_preferred, num_trajectories=int(len(preferred_indices)), num_steps=int(len(preferred_step_scores))),
			"non_preferred": dict(stepwise_nonpreferred, num_trajectories=int(len(nonpreferred_indices)), num_steps=int(len(nonpreferred_step_scores))),
		},
		"implied_linear_utility": implied_lambda,
		"notes": {
			"interpretation": "Positive trajectory-level correlation means the learned RUDDER signal orders trajectories similarly to reward - lambda * cost.",
			"best_lambda_definition": "Lambda that maximizes Spearman correlation over the sweep.",
			"score_mode": "prefix_final uses the final score at the real trajectory end; padded_final uses the last padded timestep output.",
			"stepwise_groups": "preferred/non-preferred split uses preference_label, then source mapping, then reward-threshold/cost-threshold fallback.",
		},
	}

	summary_path = os.path.join(out_dir, "utility_comparison_summary.json")
	with open(summary_path, "w", encoding="utf-8") as f:
		json.dump(results, f, indent=2)

	np.savez_compressed(
		os.path.join(out_dir, "utility_comparison_arrays.npz"),
		reward_sums=reward_sums,
		cost_sums=cost_sums,
		model_scores=model_scores,
		lambda_values=lambda_values,
		step_scores_flat=step_scores_flat,
		reward_steps_flat=reward_steps_flat,
		cost_steps_flat=cost_steps_flat,
		lengths=np.asarray(lengths, dtype=np.int32),
	)

	_save_plots(out_dir, sweep, trajectory_stats, best_lambda)
	step_dist_plot_path = _save_stepwise_progress_samples(
		out_dir=out_dir,
		step_scores_all=step_scores_all,
		reward_steps_all=reward_steps_all,
		cost_steps_all=cost_steps_all,
		lambda_best=best_lambda,
		preferred_indices=preferred_indices,
		nonpreferred_indices=nonpreferred_indices,
		preferred_count=int(plot_pref_count),
		nonpreferred_count=int(plot_nonpref_count),
		sample_seed=int(args.plot_seed),
	)

	print("=" * 80)
	print("UTILITY COMPARISON SUMMARY")
	print("=" * 80)
	print("Checkpoint:", checkpoint_path)
	print("Dataset:", resolved_dataset)
	print("Holdout trajectories:", len(trajectories))
	print("Score mode:", args.score_mode)
	print("Best lambda by Spearman: %.6f" % best_lambda)
	print("Best lambda by Kendall tau-b: %.6f" % best_kendall_lambda)
	print("Trajectory-level Pearson at best lambda: %.6f" % final_pearson)
	print("Trajectory-level Spearman at best lambda: %.6f" % final_spearman)
	print("Trajectory-level Kendall tau-b at best lambda: %.6f" % final_kendall)
	print("Trajectory-level pairwise order agreement at best lambda: %.6f" % final_pairwise)
	print("Step-level Pearson at best lambda: %.6f" % stepwise_best["pearson"])
	print("Step-level Spearman at best lambda: %.6f" % stepwise_best["spearman"])
	print("Step-level Kendall tau-b at best lambda: %.6f" % stepwise_best["kendall_tau_b"])
	print("Step-level pairwise order agreement at best lambda: %.6f" % stepwise_best["pairwise_order_agreement"])
	print("Preferred trajectories: %d | Non-preferred trajectories: %d" % (int(len(preferred_indices)), int(len(nonpreferred_indices))))
	print(
		"Step-level (preferred) | Pearson: %.6f | Spearman: %.6f | Kendall tau-b: %.6f | Pairwise order: %.6f"
		% (
			float(stepwise_preferred["pearson"]),
			float(stepwise_preferred["spearman"]),
			float(stepwise_preferred["kendall_tau_b"]),
			float(stepwise_preferred["pairwise_order_agreement"]),
		)
	)
	print(
		"Step-level (non-preferred) | Pearson: %.6f | Spearman: %.6f | Kendall tau-b: %.6f | Pairwise order: %.6f"
		% (
			float(stepwise_nonpreferred["pearson"]),
			float(stepwise_nonpreferred["spearman"]),
			float(stepwise_nonpreferred["kendall_tau_b"]),
			float(stepwise_nonpreferred["pairwise_order_agreement"]),
		)
	)
	print("OLS-implied lambda: %.6f" % implied_lambda["implied_lambda"])
	print("Saved summary:", summary_path)
	if HAS_MPL:
		print("Saved plot:", os.path.join(out_dir, "utility_sweep.png"))
		if step_dist_plot_path:
			print("Saved plot:", step_dist_plot_path)


if __name__ == "__main__":
	main()
