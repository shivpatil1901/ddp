#!/usr/bin/env python
"""
Script to load SafeDICE weights and analyze the value function (critic network)
from the trained policy. The value function is used to classify states as:
1. High reward vs Low reward states
2. Safe vs Unsafe states
3. Trajectory quality assessment
"""

import numpy as np
import tensorflow as tf
import pickle
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add SafeDICE to path
safedice_path = Path(__file__).parent / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

from algorithms.safedice import SafeDICE as AntiDICE
import config.safedice_config as antidice_config


class ValueFunctionAnalyzer:
    """Analyzes the value function from a trained SafeDICE policy."""
    
    def __init__(self, weights_path: str, config_dict: Dict = None):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            weights_path: Path to the .pickle weights file
            config_dict: Optional config dictionary. If None, uses safedice_config defaults
        """
        self.weights_path = weights_path
        self.model = None
        self.state_dim = None
        self.action_dim = None
        self.config = dict(config_dict) if config_dict is not None else dict(antidice_config.hparams[0])
        # Backward-compatibility: some safedice configs/checkpoints omit alpha.
        self.config.setdefault('alpha', 0.0)
        
        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # This happens if TF already initialized devices in current process.
                print(f"⚠️  Could not set GPU memory growth (continuing): {e}")
            print(f"✅ GPUs found: {len(gpus)} device(s), memory growth enabled on all")
        else:
            print("⚠️  No GPU found, using CPU")
        
        self._load_weights()
    
    def _load_weights(self):
        """Load the weights from pickle file and reconstruct the model."""
        print(f"Loading weights from: {self.weights_path}")
        
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        # Load pickle file
        try:
            import pickle5 as pickle_lib
        except ImportError:
            import pickle as pickle_lib
        
        with open(self.weights_path, 'rb') as f:
            data = pickle_lib.load(f)
        
        training_state = data['training_state']
        
        # Infer state/action dimensions directly from saved network tensors.
        critic_params = training_state.get('critic_params', [])
        cost_params = training_state.get('cost_params', [])
        
        if not critic_params:
            raise ValueError("No critic parameters found in checkpoint!")
        
        # Extract first-layer input sizes.
        state_dim = None
        cost_input_dim = None
        
        for name, param in critic_params:
            if 'mlp/dense/kernel' in name or 'mlp/dense' in name:
                # First dense layer input dimension = state_dim (since critic takes state only)
                state_dim = param.shape[0]
                break

        for name, param in cost_params:
            if 'mlp/dense/kernel' in name or 'mlp/dense' in name:
                # First dense layer input dimension = state_dim + action_dim
                cost_input_dim = param.shape[0]
                break
        
        if state_dim is None:
            raise ValueError("Could not infer state dimension from weights")

        if cost_input_dim is None:
            raise ValueError("Could not infer cost input dimension from weights")

        action_dim = cost_input_dim - state_dim
        if action_dim <= 0:
            raise ValueError(
                f"Inferred invalid action_dim={action_dim} from cost_input_dim={cost_input_dim} and state_dim={state_dim}"
            )

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        print(f"Reconstructed dimensions: state_dim={state_dim}, action_dim={action_dim}")
        
        # Create model
        self.model = AntiDICE(
            state_dim=state_dim,
            action_dim=action_dim,
            mixture_actor=False,
            is_discrete_action=False,
            config=self.config
        )
        
        # Set weights
        self.model.set_training_state(training_state)
        print("✅ Model loaded and weights set!")
    
    def get_state_values(self, states: np.ndarray, batch_size: int = 65536) -> np.ndarray:
        """
        Get value function estimates for a batch of states.
        
        Args:
            states: Array of shape (N, state_dim)
            
        Returns:
            Array of shape (N,) with value estimates
        """
        states = np.asarray(states)
        n = len(states)
        if n == 0:
            return np.array([], dtype=np.float32)

        batch_size = max(int(batch_size), 1)

        # Evaluate in mini-batches to avoid GPU/CPU OOM on large datasets.
        value_parts = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            state_batch = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
            batch_values, _ = self.model.critic(state_batch)
            value_parts.append(batch_values.numpy())

        return np.concatenate(value_parts, axis=0).reshape(-1)

    def get_cost_values(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        batch_size: int = 65536,
    ) -> np.ndarray:
        """Get cost-head estimates for (state, action) pairs."""
        states = np.asarray(states)
        actions = np.asarray(actions)

        if len(states) != len(actions):
            raise ValueError("states and actions must have the same length")

        if states.ndim == 1:
            states = states.reshape(-1, 1)
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)

        n = len(states)
        if n == 0:
            return np.array([], dtype=np.float32)

        batch_size = max(int(batch_size), 1)
        out_parts = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sa = np.concatenate([states[start:end], actions[start:end]], axis=1)
            sa_tensor = tf.convert_to_tensor(sa, dtype=tf.float32)
            batch_scores, _ = self.model.cost(sa_tensor)
            out_parts.append(batch_scores.numpy())

        return np.concatenate(out_parts, axis=0).reshape(-1)

    @staticmethod
    def _range_stats(arr: np.ndarray) -> Dict:
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            return {
                'count': 0,
                'min': float('nan'),
                'max': float('nan'),
                'mean': float('nan'),
                'median': float('nan'),
                'p10': float('nan'),
                'p90': float('nan'),
            }
        return {
            'count': int(arr.size),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'mean': float(np.mean(arr)),
            'median': float(np.median(arr)),
            'p10': float(np.percentile(arr, 10)),
            'p90': float(np.percentile(arr, 90)),
        }

    @staticmethod
    def _best_threshold_binary(scores: np.ndarray, y_true: np.ndarray) -> Dict:
        """Find threshold/direction with highest balanced accuracy for binary labels."""
        scores = np.asarray(scores).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1).astype(np.int32)

        if scores.size == 0:
            return {
                'threshold': float('nan'),
                'direction': 'unavailable',
                'balanced_accuracy': float('nan'),
                'metrics': {},
            }

        uniq = np.unique(scores)
        if uniq.size > 2048:
            idx = np.linspace(0, uniq.size - 1, 2048).astype(np.int32)
            uniq = uniq[idx]

        best = None
        for direction in ('score_ge_threshold_is_unsafe', 'score_le_threshold_is_unsafe'):
            for t in uniq:
                if direction == 'score_ge_threshold_is_unsafe':
                    y_pred = (scores >= t).astype(np.int32)
                else:
                    y_pred = (scores <= t).astype(np.int32)

                m = ValueFunctionAnalyzer._binary_classification_metrics(y_true, y_pred)
                tpr = m['recall']
                tnr = m['tn'] / max(m['tn'] + m['fp'], 1)
                bal_acc = 0.5 * (tpr + tnr)

                if best is None or bal_acc > best['balanced_accuracy']:
                    best = {
                        'threshold': float(t),
                        'direction': direction,
                        'balanced_accuracy': float(bal_acc),
                        'metrics': m,
                    }

        return best
    
    def classify_states_by_value(
        self, 
        states: np.ndarray,
        method: str = 'percentile',
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify states into high/low value categories.
        
        Args:
            states: Array of shape (N, state_dim)
            method: 'percentile' (default) or 'absolute'
                - 'percentile': classify based on percentile of value distribution
                - 'absolute': classify based on absolute threshold value
            threshold: Threshold for classification
                - With 'percentile': value between 0-1 (0.5 = median)
                - With 'absolute': absolute value threshold
                
        Returns:
            values: Array of shape (N,) with value estimates
            classifications: Array of shape (N,) with binary classification (0 or 1)
        """
        values = self.get_state_values(states)
        
        if method == 'percentile':
            cutoff = np.percentile(values, threshold * 100)
            classifications = (values >= cutoff).astype(int)
        elif method == 'absolute':
            classifications = (values >= threshold).astype(int)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return values, classifications

    @staticmethod
    def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
        """Return correlation while guarding against degenerate inputs."""
        if x.size == 0 or y.size == 0:
            return float('nan')
        if np.allclose(np.std(x), 0.0) or np.allclose(np.std(y), 0.0):
            return float('nan')
        return float(np.corrcoef(x, y)[0, 1])

    @staticmethod
    def _binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute standard binary classification metrics without external deps."""
        y_true = y_true.astype(np.int32)
        y_pred = y_pred.astype(np.int32)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        total = max(len(y_true), 1)
        accuracy = (tp + tn) / total
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def evaluate_state_classification_with_ground_truth(
        self,
        dataset_path: str,
        max_states: int = None,
        eval_batch_size: int = 65536,
    ) -> Dict:
        """
        Evaluate value-function-based classification against environment ground-truth
        rewards/costs stored in the dataset.

        Methodology:
        1) Predict high/low reward and safe/unsafe from value function.
        2) Check mean cost(predicted unsafe) > mean cost(predicted safe).
        3) Check corr(value, reward) > 0 and corr(value, cost) < 0.
        4) Report classification metrics against ground truth labels.
        """
        try:
            import pickle5 as pickle_lib
        except ImportError:
            import pickle as pickle_lib

        with open(dataset_path, 'rb') as f:
            data = pickle_lib.load(f)

        observations, rewards, costs = self._extract_state_reward_cost_arrays(data)

        n = min(len(observations), len(rewards), len(costs))
        observations = observations[:n]
        rewards = rewards[:n]
        costs = costs[:n]

        if max_states is not None:
            n = min(n, int(max_states))
            observations = observations[:n]
            rewards = rewards[:n]
            costs = costs[:n]

        values = self.get_state_values(observations, batch_size=eval_batch_size)

        # Correlation checks requested by user.
        value_reward_corr = self._safe_corrcoef(values, rewards)
        value_cost_corr = self._safe_corrcoef(values, costs)

        # Ground-truth labels.
        reward_threshold = float(np.median(rewards))
        gt_high_reward = (rewards >= reward_threshold).astype(np.int32)

        # Prefer cost > 0 as unsafe for safety-gym style cost signals.
        if np.any(costs > 0):
            gt_unsafe = (costs > 0).astype(np.int32)
        else:
            gt_unsafe = (costs >= np.median(costs)).astype(np.int32)

        # Predicted labels from value function.
        value_reward_threshold = float(np.median(values))
        pred_high_reward = (values >= value_reward_threshold).astype(np.int32)

        unsafe_rate = float(np.mean(gt_unsafe))
        if 0.0 < unsafe_rate < 1.0:
            unsafe_value_threshold = float(np.percentile(values, unsafe_rate * 100.0))
            pred_unsafe = (values <= unsafe_value_threshold).astype(np.int32)
        else:
            unsafe_value_threshold = float(np.median(values))
            pred_unsafe = (values <= unsafe_value_threshold).astype(np.int32)

        # Classification metrics.
        reward_metrics = self._binary_classification_metrics(gt_high_reward, pred_high_reward)
        safety_metrics = self._binary_classification_metrics(gt_unsafe, pred_unsafe)

        # Mean-cost check on predicted safe/unsafe partitions.
        unsafe_mask = pred_unsafe == 1
        safe_mask = pred_unsafe == 0
        mean_cost_pred_unsafe = float(np.mean(costs[unsafe_mask])) if np.any(unsafe_mask) else float('nan')
        mean_cost_pred_safe = float(np.mean(costs[safe_mask])) if np.any(safe_mask) else float('nan')
        unsafe_has_higher_mean_cost = bool(mean_cost_pred_unsafe > mean_cost_pred_safe)

        return {
            'num_states_evaluated': int(n),
            'thresholds': {
                'reward_threshold_median': reward_threshold,
                'value_threshold_for_reward_pred': value_reward_threshold,
                'value_threshold_for_unsafe_pred': unsafe_value_threshold,
                'unsafe_rate_ground_truth': unsafe_rate,
            },
            'correlations': {
                'value_reward': value_reward_corr,
                'value_cost': value_cost_corr,
                'is_value_reward_positive': bool(value_reward_corr > 0) if not np.isnan(value_reward_corr) else False,
                'is_value_cost_negative': bool(value_cost_corr < 0) if not np.isnan(value_cost_corr) else False,
            },
            'cost_partition_check': {
                'mean_cost_predicted_unsafe': mean_cost_pred_unsafe,
                'mean_cost_predicted_safe': mean_cost_pred_safe,
                'unsafe_has_higher_mean_cost': unsafe_has_higher_mean_cost,
            },
            'reward_classification': reward_metrics,
            'safety_classification': safety_metrics,
        }

    def _extract_state_reward_cost_arrays(self, data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract flattened observations/rewards/costs arrays from common dataset formats."""
        # Format A: flat dict of arrays
        if isinstance(data, dict) and ('observations' in data or 'states' in data):
            observations = np.asarray(data.get('observations', data.get('states')))
            rewards = data.get('rewards', None)
            costs = data.get('costs', None)
            if rewards is not None and costs is not None:
                rewards = np.asarray(rewards).reshape(-1)
                costs = np.asarray(costs).reshape(-1)
                if observations.ndim == 1:
                    observations = observations.reshape(-1, 1)
                return observations, rewards, costs

        # Format B: dict with trajectories list
        if isinstance(data, dict) and 'trajectories' in data:
            trajectories = data['trajectories']
        # Format C: list/tuple of trajectory dicts
        elif isinstance(data, (list, tuple)):
            trajectories = data
        else:
            trajectories = None

        if trajectories is None:
            raise ValueError(
                "Unsupported dataset format for ground-truth evaluation. "
                "Expected either dict with observations/states + rewards/costs arrays, "
                "dict with trajectories list, or list of trajectory dicts."
            )

        obs_parts = []
        reward_parts = []
        cost_parts = []

        for traj in trajectories:
            if not isinstance(traj, dict):
                continue

            traj_obs = traj.get('observations', traj.get('states', traj.get('observation', None)))
            traj_rewards = traj.get('rewards', traj.get('reward', None))
            traj_costs = traj.get('costs', traj.get('cost', None))

            if traj_obs is None or traj_rewards is None or traj_costs is None:
                continue

            traj_obs = np.asarray(traj_obs)
            traj_rewards = np.asarray(traj_rewards).reshape(-1)
            traj_costs = np.asarray(traj_costs).reshape(-1)

            n = min(len(traj_obs), len(traj_rewards), len(traj_costs))
            if n <= 0:
                continue

            traj_obs = traj_obs[:n]
            if traj_obs.ndim == 1:
                traj_obs = traj_obs.reshape(-1, 1)

            obs_parts.append(traj_obs)
            reward_parts.append(traj_rewards[:n])
            cost_parts.append(traj_costs[:n])

        if not obs_parts:
            raise ValueError(
                "Could not extract observations/states with rewards/costs from trajectories. "
                "Each trajectory must provide these keys."
            )

        observations = np.concatenate(obs_parts, axis=0)
        rewards = np.concatenate(reward_parts, axis=0)
        costs = np.concatenate(cost_parts, axis=0)

        return observations, rewards, costs

    def _load_dataset_trajectories(self, dataset_path: str, max_trajectories: int = None) -> List[Dict]:
        """Load trajectories from supported dataset formats."""
        try:
            import pickle5 as pickle_lib
        except ImportError:
            import pickle as pickle_lib

        print(f"Loading dataset from: {dataset_path}")
        with open(dataset_path, 'rb') as f:
            data = pickle_lib.load(f)

        if isinstance(data, dict):
            if 'observations' in data or 'states' in data:
                trajectories = self._split_trajectories_from_arrays(data)
            elif 'trajectories' in data:
                trajectories = data['trajectories']
            else:
                trajectories = list(data.values())
        else:
            trajectories = data if isinstance(data, list) else [data]

        if max_trajectories:
            trajectories = trajectories[:max_trajectories]

        return trajectories

    def analyze_initial_state_heads(
        self,
        dataset_path: str,
        max_trajectories: int = None,
        env_cost_limit: float = 25.0,
        eval_batch_size: int = 65536,
    ) -> Dict:
        """
        Analyze trajectory-start signals with independent reward and cost heads.

        Reward ranking uses V_reward(s0). Preferred/non-preferred buckets are taken
        from explicit labels if present, otherwise from median trajectory return.

        Cost calibration uses cumulative environment cost threshold:
          safe if cumulative_cost <= env_cost_limit, else unsafe.
        It then finds a calibrated cost-head threshold in latent space.
        """
        trajectories = self._load_dataset_trajectories(dataset_path, max_trajectories=max_trajectories)

        init_states = []
        init_actions = []
        traj_returns = []
        traj_costs = []
        pref_labels = []
        pref_label_available = []

        for traj in trajectories:
            if not isinstance(traj, dict):
                continue

            obs = traj.get('observations', traj.get('states', None))
            rewards = traj.get('rewards', None)
            costs = traj.get('costs', None)
            actions = traj.get('actions', None)
            if obs is None or rewards is None or costs is None:
                continue

            obs = np.asarray(obs, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            costs = np.asarray(costs, dtype=np.float32).reshape(-1)
            actions_np = None
            if actions is not None:
                actions_np = np.asarray(actions, dtype=np.float32)
                if actions_np.ndim == 1:
                    actions_np = actions_np.reshape(-1, 1)

            if obs.ndim == 1:
                obs = obs.reshape(-1, 1)

            n = min(len(obs), len(rewards), len(costs))
            if actions_np is not None:
                n = min(n, len(actions_np))
            if n <= 0:
                continue

            s0 = obs[0]
            a0 = actions_np[0] if actions_np is not None else None

            init_states.append(s0)
            init_actions.append(a0)
            traj_returns.append(float(np.sum(rewards[:n])))
            traj_costs.append(float(np.sum(costs[:n])))

            pref = None
            for key in ('preferred', 'is_preferred', 'preference', 'label'):
                if key in traj:
                    val = np.asarray(traj[key]).reshape(-1)
                    if val.size > 0:
                        pref = int(val[0] > 0)
                        break
            pref_labels.append(0 if pref is None else int(pref))
            pref_label_available.append(pref is not None)

        if not init_states:
            return {
                'error': 'No valid trajectories with observations/rewards/costs found',
            }

        init_states = np.asarray(init_states, dtype=np.float32)
        traj_returns = np.asarray(traj_returns, dtype=np.float32)
        traj_costs = np.asarray(traj_costs, dtype=np.float32)
        pref_labels = np.asarray(pref_labels, dtype=np.int32)
        pref_label_available = np.asarray(pref_label_available, dtype=bool)

        v_s0 = self.get_state_values(init_states, batch_size=eval_batch_size)

        if np.any(pref_label_available):
            preferred_mask = pref_labels == 1
            preference_source = 'explicit_trajectory_label'
        else:
            ret_thresh = float(np.median(traj_returns))
            preferred_mask = traj_returns >= ret_thresh
            preference_source = f'median_return_threshold={ret_thresh:.6f}'
        non_preferred_mask = ~preferred_mask

        reward_head_summary = {
            'preference_source': preference_source,
            'preferred_v_s0_range': self._range_stats(v_s0[preferred_mask]),
            'non_preferred_v_s0_range': self._range_stats(v_s0[non_preferred_mask]),
            'corr_v_s0_vs_traj_return': self._safe_corrcoef(v_s0, traj_returns),
        }

        safe_mask = traj_costs <= float(env_cost_limit)
        unsafe_mask = ~safe_mask

        with_actions_mask = np.array([a is not None for a in init_actions], dtype=bool)
        cost_head_summary = {
            'env_cost_limit': float(env_cost_limit),
            'num_safe_env': int(np.sum(safe_mask)),
            'num_unsafe_env': int(np.sum(unsafe_mask)),
            'num_with_actions': int(np.sum(with_actions_mask)),
        }

        if np.any(with_actions_mask):
            action_list = [a for a in init_actions if a is not None]
            init_actions_arr = np.asarray(action_list, dtype=np.float32)
            init_states_actions = init_states[with_actions_mask]
            traj_costs_actions = traj_costs[with_actions_mask]

            cost_s0 = self.get_cost_values(
                init_states_actions,
                init_actions_arr,
                batch_size=eval_batch_size,
            )

            safe_actions = traj_costs_actions <= float(env_cost_limit)
            unsafe_actions = ~safe_actions
            y_true_unsafe = unsafe_actions.astype(np.int32)

            best = self._best_threshold_binary(cost_s0, y_true_unsafe)

            cost_head_summary.update({
                'safe_cost_head_s0_range': self._range_stats(cost_s0[safe_actions]),
                'unsafe_cost_head_s0_range': self._range_stats(cost_s0[unsafe_actions]),
                'corr_cost_head_s0_vs_traj_env_cost': self._safe_corrcoef(cost_s0, traj_costs_actions),
                'calibrated_c_limit': best,
            })
        else:
            cost_head_summary.update({
                'warning': 'No trajectory actions found; cost_head(s0,a0) calibration skipped.'
            })

        return {
            'num_trajectories_used': int(len(init_states)),
            'reward_head': reward_head_summary,
            'cost_head': cost_head_summary,
        }
    
    def analyze_trajectory(self, trajectory: Dict) -> Dict:
        """
        Analyze a single trajectory using the value function.
        
        Args:
            trajectory: Dictionary with keys 'observations' and optionally 'rewards', 'costs'
                       observations shape: (T, state_dim)
                       rewards shape: (T,)
                       costs shape: (T,)
        
        Returns:
            Dictionary with analysis results
        """
        states = trajectory.get('observations', trajectory.get('states'))
        if states is None:
            raise ValueError("Trajectory must contain 'observations' or 'states'.")
        rewards = trajectory.get('rewards', None)
        costs = trajectory.get('costs', None)

        # Normalize common dataset shapes (N, 1) -> (N,) for stats/correlation.
        if rewards is not None:
            rewards = np.asarray(rewards).reshape(-1)
        if costs is not None:
            costs = np.asarray(costs).reshape(-1)
        
        values = self.get_state_values(states)
        
        result = {
            'trajectory_length': len(states),
            'value_min': np.min(values),
            'value_max': np.max(values),
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'values': values,
        }
        
        if rewards is not None:
            result.update({
                'reward_min': np.min(rewards),
                'reward_max': np.max(rewards),
                'reward_mean': np.mean(rewards),
                'reward_sum': np.sum(rewards),
                'reward_traj_return': np.sum(rewards),
            })
            # Correlation between value and reward
            result['value_reward_correlation'] = np.corrcoef(values, rewards)[0, 1]
        
        if costs is not None:
            result.update({
                'cost_min': np.min(costs),
                'cost_max': np.max(costs),
                'cost_mean': np.mean(costs),
                'cost_sum': np.sum(costs),
                'cost_accumulation': np.sum(costs),
            })
            result['value_cost_correlation'] = np.corrcoef(values, costs)[0, 1]
        
        # Classify trajectory as high/low quality based on value
        avg_value = np.mean(values)
        result['estimated_trajectory_quality'] = avg_value
        
        return result
    
    def analyze_dataset(
        self, 
        dataset_path: str,
        max_trajectories: int = None
    ) -> Dict:
        """
        Analyze a full dataset of trajectories.
        
        Args:
            dataset_path: Path to pickle file containing trajectories
            max_trajectories: Maximum number of trajectories to analyze
            
        Returns:
            Dictionary with aggregate statistics
        """
        trajectories = self._load_dataset_trajectories(dataset_path, max_trajectories=max_trajectories)
        
        print(f"Found {len(trajectories)} trajectories to analyze...")
        
        analysis_results = []
        value_distributions = []
        error_count = 0
        
        for i, traj in enumerate(tqdm(trajectories, desc="Analyzing trajectories")):
            if isinstance(traj, dict):
                try:
                    result = self.analyze_trajectory(traj)
                    analysis_results.append(result)
                    value_distributions.append(result['values'])
                except Exception as e:
                    error_count += 1
                    if error_count <= 3:  # Show first 3 errors only
                        print(f"  Error analyzing trajectory {i}: {str(e)[:150]}")
                    continue
        
        if error_count > 3:
            print(f"  ...and {error_count - 3} more trajectory errors (suppressed)")
        
        if not analysis_results:
            return {
                'error': 'No valid trajectories analyzed',
                'num_trajectories': 0,
                'avg_trajectory_length': 0.0,
                'value_statistics': {},
                'detailed_results': [],
            }
        
        # Aggregate statistics
        aggregate_stats = {
            'num_trajectories': len(analysis_results),
            'avg_trajectory_length': np.mean([r['trajectory_length'] for r in analysis_results]),
            'value_statistics': {
                'global_min': np.min([r['value_min'] for r in analysis_results]),
                'global_max': np.max([r['value_max'] for r in analysis_results]),
                'mean_of_means': np.mean([r['value_mean'] for r in analysis_results]),
                'std_of_stds': np.mean([r['value_std'] for r in analysis_results]),
            }
        }
        
        # Reward statistics if available
        if any('reward_traj_return' in r for r in analysis_results):
            returns = [r['reward_traj_return'] for r in analysis_results if 'reward_traj_return' in r]
            aggregate_stats['reward_statistics'] = {
                'mean_return': np.mean(returns),
                'std_return': np.std(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
            }
        
        # Cost statistics if available
        if any('cost_accumulation' in r for r in analysis_results):
            costs = [r['cost_accumulation'] for r in analysis_results if 'cost_accumulation' in r]
            aggregate_stats['cost_statistics'] = {
                'mean_cost': np.mean(costs),
                'std_cost': np.std(costs),
                'min_cost': np.min(costs),
                'max_cost': np.max(costs),
            }
        
        # Value-reward correlation
        if any('value_reward_correlation' in r for r in analysis_results):
            corrs = [r['value_reward_correlation'] for r in analysis_results 
                     if not np.isnan(r.get('value_reward_correlation', np.nan))]
            if corrs:
                aggregate_stats['value_reward_correlation_stats'] = {
                    'mean': np.mean(corrs),
                    'std': np.std(corrs),
                    'correlations': corrs,
                }
        
        aggregate_stats['detailed_results'] = analysis_results
        
        return aggregate_stats
    
    def _split_trajectories_from_arrays(self, data: Dict) -> List[Dict]:
        """
        Split trajectory arrays into individual trajectories.
        Handles datasets with or without explicit episode termination flags.
        Uses pattern from build_reward_balanced_trajectories.py for robustness.
        """
        # Extract arrays with flexible key names
        observations = data.get('observations', data.get('states', None))
        rewards = data.get('rewards', data.get('reward', None))
        costs = data.get('costs', data.get('cost', None))
        actions = data.get('actions', data.get('action', None))
        dones = data.get('dones', data.get('terminals', data.get('terminal', None)))
        
        if observations is None:
            return []
        
        # Convert to numpy arrays
        observations = np.asarray(observations, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1) if rewards is not None else None
        costs = np.asarray(costs, dtype=np.float32).reshape(-1) if costs is not None else None
        actions = np.asarray(actions, dtype=np.float32) if actions is not None else None
        dones = np.asarray(dones, dtype=bool).reshape(-1) if dones is not None else None
        
        if observations.size == 0:
            return []
        
        trajectories = []
        n = len(observations)
        
        # Case 1: Explicit episode boundaries with dones/terminals array
        if dones is not None and len(dones) > 0:
            dones = dones[:n]  # Ensure same length as observations
            start_idx = 0
            
            for i, done in enumerate(dones):
                if done:
                    end_idx = i + 1
                    traj = {'observations': observations[start_idx:end_idx]}
                    if rewards is not None:
                        traj['rewards'] = rewards[start_idx:end_idx]
                    if costs is not None:
                        traj['costs'] = costs[start_idx:end_idx]
                    if actions is not None:
                        traj['actions'] = actions[start_idx:end_idx]
                    
                    trajectories.append(traj)
                    start_idx = end_idx
            
            # Handle remainder after last done flag
            if start_idx < n:
                traj = {'observations': observations[start_idx:n]}
                if rewards is not None:
                    traj['rewards'] = rewards[start_idx:n]
                if costs is not None:
                    traj['costs'] = costs[start_idx:n]
                if actions is not None:
                    traj['actions'] = actions[start_idx:n]
                trajectories.append(traj)
        else:
            # Case 2: No explicit episode boundaries
            # Split into fixed-size chunks (common for pre-collected datasets)
            max_trajectory_length = 1000
            
            for start_idx in range(0, n, max_trajectory_length):
                end_idx = min(start_idx + max_trajectory_length, n)
                traj = {'observations': observations[start_idx:end_idx]}
                if rewards is not None:
                    traj['rewards'] = rewards[start_idx:end_idx]
                if costs is not None:
                    traj['costs'] = costs[start_idx:end_idx]
                if actions is not None:
                    traj['actions'] = actions[start_idx:end_idx]
                
                trajectories.append(traj)
        
        return trajectories
    
    def plot_value_distribution(self, states: np.ndarray, labels: str = None, save_path: str = None):
        """
        Plot the distribution of value function estimates.
        
        Args:
            states: Array of shape (N, state_dim)
            labels: Optional array of labels for coloring
            save_path: If provided, save figure to this path
        """
        values = self.get_state_values(states)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(values, bins=50, alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Value Function Estimate')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Value Function Estimates')
        axes[0].grid(alpha=0.3)
        
        # Sorted values
        sorted_values = np.sort(values)
        axes[1].plot(sorted_values, linewidth=2)
        axes[1].set_xlabel('State Index (sorted by value)')
        axes[1].set_ylabel('Value Function Estimate')
        axes[1].set_title('Sorted Value Function Estimates')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        
        plt.show()
        
        return fig
    
    def plot_trajectory_analysis(
        self, 
        trajectory: Dict, 
        save_path: str = None
    ):
        """
        Plot value function along trajectory with rewards/costs if available.
        
        Args:
            trajectory: Dictionary with 'observations' and optional 'rewards', 'costs'
            save_path: If provided, save figure to this path
        """
        states = trajectory['observations']
        values = self.get_state_values(states)
        rewards = trajectory.get('rewards', None)
        costs = trajectory.get('costs', None)
        
        num_plots = 1 + (1 if rewards is not None else 0) + (1 if costs is not None else 0)
        fig, axes = plt.subplots(num_plots, 1, figsize=(14, 4 * num_plots))
        
        if num_plots == 1:
            axes = [axes]
        
        # Value function
        axes[0].plot(values, linewidth=2, label='Value Function', color='blue')
        axes[0].fill_between(range(len(values)), values, alpha=0.3, color='blue')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Value Function')
        axes[0].set_title(f'Value Function Along Trajectory (Length: {len(states)})')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        plot_idx = 1
        
        # Rewards if available
        if rewards is not None:
            axes[plot_idx].plot(rewards, linewidth=2, label='Rewards', color='green')
            axes[plot_idx].fill_between(range(len(rewards)), rewards, alpha=0.3, color='green')
            axes[plot_idx].set_xlabel('Time Step')
            axes[plot_idx].set_ylabel('Reward')
            axes[plot_idx].set_title(f'Rewards Along Trajectory (Cumulative: {np.sum(rewards):.2f})')
            axes[plot_idx].grid(alpha=0.3)
            axes[plot_idx].legend()
            plot_idx += 1
        
        # Costs if available
        if costs is not None:
            axes[plot_idx].plot(costs, linewidth=2, label='Costs', color='red')
            axes[plot_idx].fill_between(range(len(costs)), costs, alpha=0.3, color='red')
            axes[plot_idx].set_xlabel('Time Step')
            axes[plot_idx].set_ylabel('Cost')
            axes[plot_idx].set_title(f'Costs Along Trajectory (Cumulative: {np.sum(costs):.2f})')
            axes[plot_idx].grid(alpha=0.3)
            axes[plot_idx].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to: {save_path}")
        
        plt.show()
        
        return fig


def main():
    parser = argparse.ArgumentParser(
        description='Analyze value function from trained SafeDICE policy'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='SafeDICE/weights/antidice_PointGoal1_seed0_20260330_205404_iter1000000.pickle',
        help='Path to weights pickle file'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
        help='Path to dataset pickle file for analysis'
    )
    parser.add_argument(
        '--max-trajectories',
        type=int,
        default=10,
        help='Maximum number of trajectories to analyze'
    )
    parser.add_argument(
        '--plot-distribution',
        action='store_true',
        help='Plot value distribution'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='value_function_analysis',
        help='Directory to save analysis outputs'
    )
    parser.add_argument(
        '--max-states',
        type=int,
        default=None,
        help='Maximum number of states for ground-truth classification evaluation'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=65536,
        help='Batch size for value-function inference during large-scale evaluation'
    )
    parser.add_argument(
        '--env-cost-limit',
        type=float,
        default=25.0,
        help='Trajectory cumulative environment-cost threshold for safe vs unsafe bucket'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    print("=" * 80)
    print("SafeDICE Value Function Analyzer")
    print("=" * 80)
    
    analyzer = ValueFunctionAnalyzer(args.weights)
    
    # Analyze dataset
    print("\n" + "=" * 80)
    print("Dataset Analysis")
    print("=" * 80)
    
    analysis = analyzer.analyze_dataset(args.dataset, max_trajectories=args.max_trajectories)

    # Ground-truth state classification evaluation requested by user.
    gt_eval = analyzer.evaluate_state_classification_with_ground_truth(
        dataset_path=args.dataset,
        max_states=args.max_states,
        eval_batch_size=args.eval_batch_size,
    )

    initial_head_eval = analyzer.analyze_initial_state_heads(
        dataset_path=args.dataset,
        max_trajectories=args.max_trajectories,
        env_cost_limit=args.env_cost_limit,
        eval_batch_size=args.eval_batch_size,
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    if 'error' in analysis:
        print(f"Dataset analysis warning: {analysis['error']}")

    print(f"Number of trajectories analyzed: {analysis.get('num_trajectories', 0)}")
    print(f"Average trajectory length: {analysis.get('avg_trajectory_length', 0.0):.1f}")
    if analysis.get('value_statistics'):
        print(f"\nValue Function Statistics:")
        for key, val in analysis['value_statistics'].items():
            print(f"  {key}: {val:.4f}")
    
    if 'reward_statistics' in analysis:
        print(f"\nReward Statistics:")
        for key, val in analysis['reward_statistics'].items():
            print(f"  {key}: {val:.4f}")
    
    if 'cost_statistics' in analysis:
        print(f"\nCost Statistics:")
        for key, val in analysis['cost_statistics'].items():
            print(f"  {key}: {val:.4f}")
    
    if 'value_reward_correlation_stats' in analysis:
        print(f"\nValue-Reward Correlation:")
        print(f"  Mean correlation: {analysis['value_reward_correlation_stats']['mean']:.4f}")
        print(f"  Std correlation: {analysis['value_reward_correlation_stats']['std']:.4f}")

    print(f"\nGround-Truth State Classification Evaluation:")
    print(f"  States evaluated: {gt_eval['num_states_evaluated']}")
    print(f"  Corr(value, reward): {gt_eval['correlations']['value_reward']:.4f}")
    print(f"  Corr(value, cost): {gt_eval['correlations']['value_cost']:.4f}")
    print(f"  Corr checks -> reward positive: {gt_eval['correlations']['is_value_reward_positive']}, "
          f"cost negative: {gt_eval['correlations']['is_value_cost_negative']}")
    print(f"  Mean cost predicted unsafe: {gt_eval['cost_partition_check']['mean_cost_predicted_unsafe']:.4f}")
    print(f"  Mean cost predicted safe: {gt_eval['cost_partition_check']['mean_cost_predicted_safe']:.4f}")
    print(f"  Mean-cost check (unsafe > safe): {gt_eval['cost_partition_check']['unsafe_has_higher_mean_cost']}")
    print(f"  Reward classification accuracy: {gt_eval['reward_classification']['accuracy']:.4f}")
    print(f"  Safety classification accuracy: {gt_eval['safety_classification']['accuracy']:.4f}")

    print(f"\nInitial-State Head Analysis (independent heads):")
    if 'error' in initial_head_eval:
        print(f"  Warning: {initial_head_eval['error']}")
    else:
        rh = initial_head_eval['reward_head']
        ch = initial_head_eval['cost_head']
        pref = rh['preferred_v_s0_range']
        non_pref = rh['non_preferred_v_s0_range']
        print(f"  Preference source: {rh['preference_source']}")
        print(f"  Preferred demos V(s0): min={pref['min']:.4f}, max={pref['max']:.4f}, mean={pref['mean']:.4f}")
        print(f"  Non-preferred demos V(s0): min={non_pref['min']:.4f}, max={non_pref['max']:.4f}, mean={non_pref['mean']:.4f}")
        print(f"  Corr(V(s0), traj return): {rh['corr_v_s0_vs_traj_return']:.4f}")
        print(f"  Env safe/unsafe by cumulative cost <= {ch['env_cost_limit']:.2f}: "
              f"{ch['num_safe_env']} safe, {ch['num_unsafe_env']} unsafe")
        if 'calibrated_c_limit' in ch:
            ccal = ch['calibrated_c_limit']
            print(f"  Calibrated C_limit (cost head latent): {ccal['threshold']:.6f}")
            print(f"  Unsafe rule: {ccal['direction']}")
            print(f"  Balanced accuracy at C_limit: {ccal['balanced_accuracy']:.4f}")
        elif 'warning' in ch:
            print(f"  Cost head calibration warning: {ch['warning']}")
    
    # Save detailed results
    output_file = os.path.join(args.output_dir, 'analysis_results.pkl')
    analysis['ground_truth_state_evaluation'] = gt_eval
    analysis['initial_state_head_evaluation'] = initial_head_eval
    with open(output_file, 'wb') as f:
        pickle.dump(analysis, f)
    print(f"\n✅ Saved detailed analysis to: {output_file}")
    
    # Save summary as text
    summary_file = os.path.join(args.output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("SafeDICE Value Function Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Weights file: {args.weights}\n")
        f.write(f"Dataset file: {args.dataset}\n\n")
        if 'error' in analysis:
            f.write(f"Dataset analysis warning: {analysis['error']}\n\n")
        f.write(f"Number of trajectories: {analysis.get('num_trajectories', 0)}\n")
        f.write(f"Avg trajectory length: {analysis.get('avg_trajectory_length', 0.0):.1f}\n\n")
        if analysis.get('value_statistics'):
            f.write("Value Function Statistics:\n")
            for key, val in analysis['value_statistics'].items():
                f.write(f"  {key}: {val:.4f}\n")

        f.write("\nInitial-State Head Analysis:\n")
        if 'error' in initial_head_eval:
            f.write(f"  Warning: {initial_head_eval['error']}\n")
        else:
            rh = initial_head_eval['reward_head']
            ch = initial_head_eval['cost_head']
            pref = rh['preferred_v_s0_range']
            non_pref = rh['non_preferred_v_s0_range']
            f.write(f"  Preference source: {rh['preference_source']}\n")
            f.write(
                f"  Preferred demos V(s0): min={pref['min']:.4f}, max={pref['max']:.4f}, mean={pref['mean']:.4f}\n"
            )
            f.write(
                f"  Non-preferred demos V(s0): min={non_pref['min']:.4f}, max={non_pref['max']:.4f}, mean={non_pref['mean']:.4f}\n"
            )
            f.write(f"  Corr(V(s0), traj return): {rh['corr_v_s0_vs_traj_return']:.4f}\n")
            f.write(
                f"  Env safe/unsafe by cumulative cost <= {ch['env_cost_limit']:.2f}: "
                f"{ch['num_safe_env']} safe, {ch['num_unsafe_env']} unsafe\n"
            )
            if 'calibrated_c_limit' in ch:
                ccal = ch['calibrated_c_limit']
                f.write(f"  Calibrated C_limit (cost head latent): {ccal['threshold']:.6f}\n")
                f.write(f"  Unsafe rule: {ccal['direction']}\n")
                f.write(f"  Balanced accuracy at C_limit: {ccal['balanced_accuracy']:.4f}\n")
            elif 'warning' in ch:
                f.write(f"  Cost head calibration warning: {ch['warning']}\n")
    
    print(f"✅ Saved summary to: {summary_file}")


if __name__ == '__main__':
    main()
