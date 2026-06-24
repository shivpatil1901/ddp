#!/usr/bin/env python
"""
Compare the computed value function (from policy sampling) with the SafeDICE critic network.
Analyzes distribution, overlap in bottom 20% states, and basic statistics.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

import tensorflow as tf
import torch

# Add SafeDICE to path
ROOT = Path(__file__).parent.parent
safedice_path = ROOT / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

try:
    from algorithms.safedice import SafeDICE as AntiDICE
except ImportError as e:
    print(f"Warning: Could not import SafeDICE: {e}")
    AntiDICE = None


class SafeDICECriticLoader:
    """Load and evaluate SafeDICE critic network values."""
    
    def __init__(self, weights_path: str, config_dict: Dict = None):
        """Initialize critic loader from SafeDICE checkpoint."""
        self.weights_path = weights_path
        self.model = None
        self.state_dim = None
        self.action_dim = None
        
        # Configure TensorFlow GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if len(gpus) > 0:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        # Set default config with all required keys
        self.config = {
            'hidden_size': 256,
            'critic_lr': 1e-4,
            'actor_lr': 1e-5,
            'gamma': 0.99,
            'alpha': 0.0,
            'grad_reg_coeffs': (10, 1e-6),
            'use_last_layer_bias_cost': False,
            'kernel_initializer': 'he_normal',
        }
        
        # Try to override with provided or loaded config
        if config_dict is not None:
            self.config.update(config_dict)
        else:
            # Try to load from SafeDICE config file
            try:
                sys.path.insert(0, str(safedice_path / 'config'))
                from safedice_config import hparams
                if hparams:
                    self.config.update(hparams[0])
            except (ImportError, IndexError):
                pass  # Use defaults
        
        self._load_weights()
    
    def _load_weights(self):
        """Load weights from pickle and reconstruct SafeDICE model."""
        print(f"🔄 Loading SafeDICE critic from: {self.weights_path}")
        
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        try:
            import pickle5 as pickle_lib
        except ImportError:
            import pickle as pickle_lib
        
        with open(self.weights_path, 'rb') as f:
            data = pickle_lib.load(f)
        
        training_state = data['training_state']
        critic_params = training_state.get('critic_params', [])
        cost_params = training_state.get('cost_params', [])
        
        if not critic_params:
            raise ValueError("No critic parameters found in checkpoint!")
        
        # Infer dimensions from weights
        state_dim = None
        cost_input_dim = None
        
        for name, param in critic_params:
            if 'mlp/dense/kernel' in name or 'mlp/dense' in name:
                state_dim = param.shape[0]
                break
        
        for name, param in cost_params:
            if 'mlp/dense/kernel' in name or 'mlp/dense' in name:
                cost_input_dim = param.shape[0]
                break
        
        if state_dim is None or cost_input_dim is None:
            raise ValueError("Could not infer state/action dimensions from weights")
        
        action_dim = cost_input_dim - state_dim
        if action_dim <= 0:
            raise ValueError(f"Invalid action_dim: {action_dim}")
        
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        
        print(f"  ✓ State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        # Create and load model
        if AntiDICE is None:
            raise RuntimeError("SafeDICE not available")
        
        self.model = AntiDICE(
            state_dim=state_dim,
            action_dim=action_dim,
            mixture_actor=False,
            is_discrete_action=False,
            config=self.config
        )
        self.model.set_training_state(training_state)
        print(f"  ✓ SafeDICE critic loaded successfully!")
    
    def get_state_values(self, states: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Get critic value estimates for states."""
        states = np.asarray(states, dtype=np.float32)
        n = len(states)
        
        if n == 0:
            return np.array([], dtype=np.float32)
        
        value_parts = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            state_batch = tf.convert_to_tensor(states[start:end], dtype=tf.float32)
            batch_values, _ = self.model.critic(state_batch)
            value_parts.append(batch_values.numpy())
        
        return np.concatenate(value_parts, axis=0).reshape(-1)


class ValueFunctionComparator:
    """Compare Q-network value function with SafeDICE critic."""
    
    def __init__(self, computed_values_path: str, states_path: str, safedice_weights_path: str):
        """
        Initialize comparator.
        
        Args:
            computed_values_path: Path to state_values_policy_traj300.npz
            states_path: Path to state_values_and_labels.npz
            safedice_weights_path: Path to SafeDICE checkpoint
        """
        self.computed_values_path = computed_values_path
        self.states_path = states_path
        
        print("\n" + "="*80)
        print("VALUE FUNCTION COMPARISON: Q-Network vs SafeDICE Critic")
        print("="*80)
        
        # Load computed values
        print(f"\n📂 Loading computed value function from: {computed_values_path}")
        data = np.load(computed_values_path)
        self.V_computed = data['V'].astype(np.float32)
        self.num_computed_states = len(self.V_computed)
        print(f"  ✓ Loaded {self.num_computed_states:,} computed state values")
        
        # Load states and labels
        print(f"\n📂 Loading states and labels from: {states_path}")
        states_data = np.load(states_path)
        self.V_from_labels = states_data['state_values'].astype(np.float32)
        self.labels = states_data['labels'].astype(int)
        self.bad_percentile = float(states_data['bad_percentile'])
        print(f"  ✓ Loaded {len(self.V_from_labels):,} labeled states")
        print(f"  ✓ Bad percentile threshold: {self.bad_percentile}%")
        
        # Verify that both value arrays are consistent (if same size)
        num_states_to_use = min(len(self.V_computed), len(self.V_from_labels))
        if len(self.V_computed) != len(self.V_from_labels):
            print(f"⚠️  Note: V_computed has {len(self.V_computed):,} states, "
                  f"V_from_labels has {len(self.V_from_labels):,} states")
            print(f"   Using first {num_states_to_use:,} states for comparison")
            self.V_computed = self.V_computed[:num_states_to_use]
            self.V_from_labels = self.V_from_labels[:num_states_to_use]
        elif not np.allclose(self.V_computed, self.V_from_labels, rtol=1e-5):
            print("⚠️  Warning: Computed and labeled value arrays differ (this is expected if they come from different runs)")
        
        # Load trajectories to get actual states
        print(f"\n📂 Loading trajectory states...")
        self._load_trajectory_states()
        
        print(f"  ✓ Loaded {len(self.states):,} unique states")
        
        # Load SafeDICE critic
        print(f"\n🔧 Loading SafeDICE critic...")
        self.critic_loader = SafeDICECriticLoader(safedice_weights_path)
        
        # Verify state dimensions match
        if self.states.shape[1] != self.critic_loader.state_dim:
            raise ValueError(
                f"State dimension mismatch: loaded {self.states.shape[1]}, "
                f"SafeDICE expects {self.critic_loader.state_dim}"
            )
    
    def _load_trajectory_states(self):
        """Load actual states from trajectories."""
        # Use the lagrangian dataset that was used for value function computation
        dataset_paths = [
            ROOT / 'SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle',
            ROOT / 'SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1.pickle',
            ROOT / 'SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
            ROOT / 'SafeDICE/dataset/safetygym/ppo_PointGoal1.pickle',
        ]
        
        dataset_path = None
        for path in dataset_paths:
            if path.exists():
                dataset_path = path
                print(f"  Using dataset: {path.name}")
                break
        
        if dataset_path is None:
            raise FileNotFoundError(f"Could not find dataset in {[str(p) for p in dataset_paths]}")
        
        print(f"  Loading from: {dataset_path}")
        
        try:
            import pickle5 as pickle_lib
        except ImportError:
            import pickle as pickle_lib
        
        with open(dataset_path, 'rb') as f:
            data = pickle_lib.load(f)
        
        # Extract states from dataset
        X_states = self._extract_states_from_dataset(data)
        
        # Subset to match computed values
        if len(X_states) > self.num_computed_states:
            print(f"  ⚠️  Dataset has {len(X_states)} states, using first {self.num_computed_states}")
            X_states = X_states[:self.num_computed_states]
        
        self.states = X_states.astype(np.float32)
    
    def _extract_states_from_dataset(self, data) -> np.ndarray:
        """Extract state observations from dataset."""
        # Format A: flat dict with observations/states
        if isinstance(data, dict) and ('observations' in data or 'states' in data):
            obs = data.get('observations', data.get('states'))
            if obs is not None:
                return np.asarray(obs)
        
        # Format B: trajectories list
        if isinstance(data, dict) and 'trajectories' in data:
            trajectories = data['trajectories']
        elif isinstance(data, (list, tuple)):
            trajectories = data
        else:
            trajectories = None
        
        if trajectories is not None:
            obs_parts = []
            for traj in trajectories:
                if isinstance(traj, dict):
                    obs = traj.get('observations', traj.get('states'))
                    if obs is not None:
                        obs_parts.append(np.asarray(obs))
            if obs_parts:
                return np.concatenate(obs_parts, axis=0)
        
        raise ValueError("Could not extract states from dataset")
    
    def compare(self):
        """Run full comparison analysis."""
        print("\n" + "="*80)
        print("COMPUTING SAFEDICE CRITIC VALUES")
        print("="*80)
        
        print(f"\n⏳ Computing critic values for {len(self.states):,} states...")
        self.V_critic = self.critic_loader.get_state_values(self.states, batch_size=2048)
        print(f"  ✓ Computed critic values")
        
        print("\n" + "="*80)
        print("BASIC STATISTICS")
        print("="*80)
        
        self._print_basic_stats()
        
        print("\n" + "="*80)
        print("BOTTOM 20% ANALYSIS")
        print("="*80)
        
        self._analyze_bottom_20_percent()
        
        print("\n" + "="*80)
        print("DISTRIBUTION ANALYSIS")
        print("="*80)
        
        self._analyze_distributions()
        
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        self._analyze_correlation()
        
        return {
            'V_computed': self.V_computed,
            'V_critic': self.V_critic,
            'states': self.states,
            'labels': self.labels,
        }
    
    def _print_basic_stats(self):
        """Print basic statistics for both value functions."""
        print("\n📊 COMPUTED VALUE FUNCTION (Q-Network Policy Sampling):")
        self._print_stats(self.V_computed, "V_computed")
        
        print("\n📊 SAFEDICE CRITIC VALUE FUNCTION:")
        self._print_stats(self.V_critic, "V_critic")
        
        print("\n📊 DIFFERENCE (V_critic - V_computed):")
        diff = self.V_critic - self.V_computed
        self._print_stats(diff, "difference")
    
    def _print_stats(self, values: np.ndarray, name: str):
        """Print statistics for a value array."""
        values = np.asarray(values).flatten()
        
        print(f"  Mean:       {np.mean(values):>10.6f}")
        print(f"  Median:     {np.median(values):>10.6f}")
        print(f"  Std Dev:    {np.std(values):>10.6f}")
        print(f"  Min:        {np.min(values):>10.6f}  (state idx: {np.argmin(values)})")
        print(f"  Max:        {np.max(values):>10.6f}  (state idx: {np.argmax(values)})")
        print(f"  p5:         {np.percentile(values, 5):>10.6f}")
        print(f"  p25:        {np.percentile(values, 25):>10.6f}")
        print(f"  p75:        {np.percentile(values, 75):>10.6f}")
        print(f"  p95:        {np.percentile(values, 95):>10.6f}")
    
    def _analyze_bottom_20_percent(self):
        """Analyze overlap and agreement in bottom 20% states."""
        threshold = np.percentile(self.V_computed, 20)
        bottom_20_computed = self.V_computed <= threshold
        num_bottom_computed = np.sum(bottom_20_computed)
        
        threshold_critic = np.percentile(self.V_critic, 20)
        bottom_20_critic = self.V_critic <= threshold_critic
        num_bottom_critic = np.sum(bottom_20_critic)
        
        overlap = np.sum(bottom_20_computed & bottom_20_critic)
        overlap_pct = 100.0 * overlap / max(1, num_bottom_computed)
        
        print(f"\n📍 Bottom 20% (by computed values):")
        print(f"  Count:              {num_bottom_computed:,} states")
        print(f"  Computed threshold: {threshold:.6f}")
        print(f"  Value range:        [{np.min(self.V_computed[bottom_20_computed]):.6f}, "
              f"{np.max(self.V_computed[bottom_20_computed]):.6f}]")
        
        # Stats for bottom 20% by computed
        print(f"\n  Critic values (for bottom 20% computed states):")
        bottom_critic_vals = self.V_critic[bottom_20_computed]
        self._print_stats(bottom_critic_vals, "critic_bottom20")
        
        print(f"\n📍 Bottom 20% (by critic values):")
        print(f"  Count:             {num_bottom_critic:,} states")
        print(f"  Critic threshold:  {threshold_critic:.6f}")
        print(f"  Value range:       [{np.min(self.V_critic[bottom_20_critic]):.6f}, "
              f"{np.max(self.V_critic[bottom_20_critic]):.6f}]")
        
        # Stats for bottom 20% by critic
        print(f"\n  Computed values (for bottom 20% critic states):")
        bottom_computed_vals = self.V_computed[bottom_20_critic]
        self._print_stats(bottom_computed_vals, "computed_bottom20")
        
        print(f"\n🔄 OVERLAP ANALYSIS:")
        print(f"  States in both bottom 20%: {overlap:,} / {num_bottom_computed:,} ({overlap_pct:.1f}%)")
        
        # Analyze non-overlapping states
        only_computed = bottom_20_computed & ~bottom_20_critic
        only_critic = bottom_20_critic & ~bottom_20_computed
        both = bottom_20_computed & bottom_20_critic
        
        print(f"  Only in computed bottom 20%: {np.sum(only_computed):,}")
        if np.sum(only_computed) > 0:
            print(f"    - Critic values: min={np.min(self.V_critic[only_computed]):.6f}, "
                  f"max={np.max(self.V_critic[only_computed]):.6f}, "
                  f"mean={np.mean(self.V_critic[only_computed]):.6f}")
        
        print(f"  Only in critic bottom 20%: {np.sum(only_critic):,}")
        if np.sum(only_critic) > 0:
            print(f"    - Computed values: min={np.min(self.V_computed[only_critic]):.6f}, "
                  f"max={np.max(self.V_computed[only_critic]):.6f}, "
                  f"mean={np.mean(self.V_computed[only_critic]):.6f}")
        
        print(f"  In both bottom 20%: {np.sum(both):,}")
        
        # Show some example states
        print(f"\n  📌 Sample states in both bottom 20%:")
        both_indices = np.where(both)[0][:5]
        for idx in both_indices:
            print(f"    State {idx}: V_computed={self.V_computed[idx]:.6f}, "
                  f"V_critic={self.V_critic[idx]:.6f}, "
                  f"diff={self.V_critic[idx] - self.V_computed[idx]:.6f}")
        
        print(f"\n  📌 Sample states only in computed bottom 20%:")
        only_comp_indices = np.where(only_computed)[0][:5]
        for idx in only_comp_indices:
            print(f"    State {idx}: V_computed={self.V_computed[idx]:.6f}, "
                  f"V_critic={self.V_critic[idx]:.6f}, "
                  f"diff={self.V_critic[idx] - self.V_computed[idx]:.6f}")
    
    def _analyze_distributions(self):
        """Analyze distribution overlap."""
        print(f"\n📈 Distribution overlap:")
        
        # Compute KL divergence (simplified)
        hist_computed, bins = np.histogram(self.V_computed, bins=50)
        hist_critic, _ = np.histogram(self.V_critic, bins=bins)
        
        # Normalize
        hist_computed = hist_computed / (np.sum(hist_computed) + 1e-10)
        hist_critic = hist_critic / (np.sum(hist_critic) + 1e-10)
        
        # KL divergence
        kl_div = np.sum(hist_computed * (np.log(hist_computed + 1e-10) - np.log(hist_critic + 1e-10)))
        print(f"  KL divergence: {kl_div:.6f}")
        
        # Wasserstein distance (simple)
        cdf_computed = np.cumsum(hist_computed)
        cdf_critic = np.cumsum(hist_critic)
        wasserstein = np.mean(np.abs(cdf_computed - cdf_critic))
        print(f"  Wasserstein distance: {wasserstein:.6f}")
        
        # Percentage overlap in value ranges
        min_v = max(np.min(self.V_computed), np.min(self.V_critic))
        max_v = min(np.max(self.V_computed), np.max(self.V_critic))
        overlap_range = max_v - min_v
        total_range = min(np.max(self.V_computed), np.max(self.V_critic)) - max(np.min(self.V_computed), np.min(self.V_critic))
        overlap_pct = 100.0 * overlap_range / max(total_range, 1e-10)
        print(f"  Value range overlap: {overlap_pct:.1f}%")
    
    def _analyze_correlation(self):
        """Analyze correlation between value functions."""
        corr = np.corrcoef(self.V_computed, self.V_critic)[0, 1]
        print(f"\n🔗 Pearson correlation: {corr:.6f}")
        
        # Spearman rank correlation
        from scipy.stats import spearmanr
        rank_corr, pval = spearmanr(self.V_computed, self.V_critic)
        print(f"  Spearman rank correlation: {rank_corr:.6f} (p-value: {pval:.2e})")
        
        # Mean absolute error
        mae = np.mean(np.abs(self.V_computed - self.V_critic))
        print(f"  Mean absolute error: {mae:.6f}")
        
        # Root mean squared error
        rmse = np.sqrt(np.mean((self.V_computed - self.V_critic) ** 2))
        print(f"  RMSE: {rmse:.6f}")
    
    def plot_comparison(self, output_dir: str = None):
        """Generate comparison plots."""
        if output_dir is None:
            output_dir = str(ROOT / "value_function_analysis_combined")
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Generating comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Q-Network vs SafeDICE Critic Value Function Comparison', fontsize=16, fontweight='bold')
        
        # 1. Histograms
        axes[0, 0].hist(self.V_computed, bins=50, alpha=0.6, label='Q-Network', color='blue', edgecolor='black')
        axes[0, 0].hist(self.V_critic, bins=50, alpha=0.6, label='SafeDICE Critic', color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Value Function Distributions')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Scatter plot
        axes[0, 1].scatter(self.V_computed, self.V_critic, alpha=0.3, s=10)
        min_val = min(np.min(self.V_computed), np.min(self.V_critic))
        max_val = max(np.max(self.V_computed), np.max(self.V_critic))
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity')
        axes[0, 1].set_xlabel('Q-Network')
        axes[0, 1].set_ylabel('SafeDICE Critic')
        axes[0, 1].set_title(f'Value Correlation (r={np.corrcoef(self.V_computed, self.V_critic)[0,1]:.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Difference distribution
        diff = self.V_critic - self.V_computed
        axes[0, 2].hist(diff, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 2].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero diff')
        axes[0, 2].set_xlabel('V_critic - V_computed')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Value Difference Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(alpha=0.3)
        
        # 4. Bottom 20% comparison
        threshold_comp = np.percentile(self.V_computed, 20)
        threshold_crit = np.percentile(self.V_critic, 20)
        bottom_comp = self.V_computed <= threshold_comp
        bottom_crit = self.V_critic <= threshold_crit
        
        categories = ['Both\nBottom 20%', 'Only\nComputed', 'Only\nCritic', 'Neither']
        both = np.sum(bottom_comp & bottom_crit)
        only_comp = np.sum(bottom_comp & ~bottom_crit)
        only_crit = np.sum(~bottom_comp & bottom_crit)
        neither = np.sum(~bottom_comp & ~bottom_crit)
        counts = [both, only_comp, only_crit, neither]
        colors = ['darkgreen', 'lightblue', 'lightcoral', 'lightgray']
        
        axes[1, 0].bar(categories, counts, color=colors, edgecolor='black')
        axes[1, 0].set_ylabel('Number of States')
        axes[1, 0].set_title('Bottom 20% Overlap')
        axes[1, 0].grid(alpha=0.3, axis='y')
        for i, count in enumerate(counts):
            axes[1, 0].text(i, count, str(count), ha='center', va='bottom')
        
        # 5. CDF comparison
        sorted_comp = np.sort(self.V_computed)
        sorted_crit = np.sort(self.V_critic)
        cdf_comp = np.arange(len(sorted_comp)) / len(sorted_comp)
        cdf_crit = np.arange(len(sorted_crit)) / len(sorted_crit)
        
        axes[1, 1].plot(sorted_comp, cdf_comp, label='Q-Network', linewidth=2)
        axes[1, 1].plot(sorted_crit, cdf_crit, label='SafeDICE Critic', linewidth=2)
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Cumulative Probability')
        axes[1, 1].set_title('Cumulative Distribution Functions')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        # 6. Statistics table
        axes[1, 2].axis('off')
        stats = [
            ['Metric', 'Q-Network', 'SafeDICE'],
            ['Mean', f'{np.mean(self.V_computed):.4f}', f'{np.mean(self.V_critic):.4f}'],
            ['Median', f'{np.median(self.V_computed):.4f}', f'{np.median(self.V_critic):.4f}'],
            ['Std Dev', f'{np.std(self.V_computed):.4f}', f'{np.std(self.V_critic):.4f}'],
            ['Min', f'{np.min(self.V_computed):.4f}', f'{np.min(self.V_critic):.4f}'],
            ['Max', f'{np.max(self.V_computed):.4f}', f'{np.max(self.V_critic):.4f}'],
            ['Corr', f'{np.corrcoef(self.V_computed, self.V_critic)[0,1]:.4f}', 'N/A'],
            ['MAE', f'{np.mean(np.abs(self.V_computed - self.V_critic)):.4f}', 'N/A'],
        ]
        
        table = axes[1, 2].table(cellText=stats, cellLoc='center', loc='center',
                                  colWidths=[0.35, 0.325, 0.325])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Header styling
        for i in range(3):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        axes[1, 2].set_title('Comparison Statistics', pad=20, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'value_function_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        
        plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare Q-Network value function with SafeDICE critic'
    )
    parser.add_argument(
        '--computed-values',
        type=str,
        default=str(ROOT / 'state_values_policy_traj300.npz'),
        help='Path to computed value function (state_values_policy_traj300.npz)'
    )
    parser.add_argument(
        '--state-labels',
        type=str,
        default=str(ROOT / 'q_function_models/state_values_and_labels.npz'),
        help='Path to state values and labels'
    )
    parser.add_argument(
        '--safedice-weights',
        type=str,
        default=str(ROOT / 'SafeDICE/weights/antidice_PointGoal1_seed0_20260330_205404_iter1000000.pickle'),
        help='Path to SafeDICE checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(ROOT / 'value_function_analysis_combined'),
        help='Output directory for plots and results'
    )
    
    args = parser.parse_args()
    
    # Run comparison
    comparator = ValueFunctionComparator(
        args.computed_values,
        args.state_labels,
        args.safedice_weights
    )
    
    results = comparator.compare()
    comparator.plot_comparison(args.output_dir)
    
    # Save comparison results
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'comparison_results.npz')
    np.savez(results_file,
             V_computed=results['V_computed'],
             V_critic=results['V_critic'],
             states=results['states'])
    print(f"\n✅ Saved comparison results to: {results_file}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
