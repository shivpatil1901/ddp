#!/usr/bin/env python
"""
Evaluate the value function on a combined dataset of both PPO and PPO-Lagrangian
to avoid class imbalance and get more balanced accuracy metrics.

Combines:
- SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle (high-reward states)
- SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle (cost-aware states)
"""

import numpy as np
import pickle
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple

# Add SafeDICE to path
safedice_path = Path(__file__).parent / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

from analyze_value_function import ValueFunctionAnalyzer


def load_pickle_dataset(path: str) -> Dict:
    """Load a pickle dataset with robustness for pickle5."""
    try:
        import pickle5 as pickle_lib
    except ImportError:
        import pickle as pickle_lib
    
    with open(path, 'rb') as f:
        return pickle_lib.load(f)


def combine_datasets(dataset1_path: str, dataset2_path: str) -> Dict:
    """
    Load and combine two SafetyGym datasets into a single balanced dataset.
    
    Args:
        dataset1_path: Path to first dataset (e.g., ppo_PointGoal1_s0.pickle)
        dataset2_path: Path to second dataset (e.g., ppo_lagrangian_PointGoal1_s0.pickle)
        
    Returns:
        Combined dataset dict with concatenated states, rewards, costs, dones
    """
    print(f"Loading dataset 1: {dataset1_path}")
    data1 = load_pickle_dataset(dataset1_path)
    
    print(f"Loading dataset 2: {dataset2_path}")
    data2 = load_pickle_dataset(dataset2_path)
    
    # Extract arrays from both datasets
    states1 = np.asarray(data1.get('states', data1.get('observations')))
    rewards1 = np.asarray(data1.get('rewards')).reshape(-1)
    costs1 = np.asarray(data1.get('costs')).reshape(-1)
    dones1 = np.asarray(data1.get('dones', data1.get('terminals'))).reshape(-1)
    
    states2 = np.asarray(data2.get('states', data2.get('observations')))
    rewards2 = np.asarray(data2.get('rewards')).reshape(-1)
    costs2 = np.asarray(data2.get('costs')).reshape(-1)
    dones2 = np.asarray(data2.get('dones', data2.get('terminals'))).reshape(-1)
    
    # Align lengths
    n1 = min(len(states1), len(rewards1), len(costs1), len(dones1))
    n2 = min(len(states2), len(rewards2), len(costs2), len(dones2))
    
    states1 = states1[:n1]
    rewards1 = rewards1[:n1]
    costs1 = costs1[:n1]
    dones1 = dones1[:n1]
    
    states2 = states2[:n2]
    rewards2 = rewards2[:n2]
    costs2 = costs2[:n2]
    dones2 = dones2[:n2]
    
    print(f"Dataset 1: {n1} states, reward range [{rewards1.min():.3f}, {rewards1.max():.3f}], "
          f"cost range [{costs1.min():.3f}, {costs1.max():.3f}]")
    print(f"Dataset 2: {n2} states, reward range [{rewards2.min():.3f}, {rewards2.max():.3f}], "
          f"cost range [{costs2.min():.3f}, {costs2.max():.3f}]")
    
    # Concatenate
    combined = {
        'states': np.concatenate([states1, states2], axis=0),
        'rewards': np.concatenate([rewards1, rewards2], axis=0),
        'costs': np.concatenate([costs1, costs2], axis=0),
        'dones': np.concatenate([dones1, dones2], axis=0),
    }
    
    print(f"\nCombined dataset: {len(combined['states'])} states")
    print(f"  Reward: mean={combined['rewards'].mean():.3f}, std={combined['rewards'].std():.3f}, "
          f"range=[{combined['rewards'].min():.3f}, {combined['rewards'].max():.3f}]")
    print(f"  Cost: mean={combined['costs'].mean():.3f}, std={combined['costs'].std():.3f}, "
          f"range=[{combined['costs'].min():.3f}, {combined['costs'].max():.3f}]")
    print(f"  Safe states (cost=0): {(combined['costs'] == 0).sum()}/{len(combined['costs'])} ({100*(combined['costs']==0).mean():.1f}%)")
    print(f"  Unsafe states (cost>0): {(combined['costs'] > 0).sum()}/{len(combined['costs'])} ({100*(combined['costs']>0).mean():.1f}%)")
    
    return combined


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate value function on combined SafetyGym datasets to avoid class imbalance'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='SafeDICE/weights/antidice_PointGoal1_seed0_20260330_205404_iter850000.pickle',
        help='Path to SafeDICE weights checkpoint'
    )
    parser.add_argument(
        '--dataset1',
        type=str,
        default='SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
        help='First dataset (PPO policy, high-reward)'
    )
    parser.add_argument(
        '--dataset2',
        type=str,
        default='SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle',
        help='Second dataset (PPO-Lagrangian policy, cost-aware)'
    )
    parser.add_argument(
        '--max-states',
        type=int,
        default=None,
        help='Maximum total states to evaluate (None = all)'
    )
    parser.add_argument(
        '--sampling-seed',
        type=int,
        default=42,
        help='Random seed for balanced safe/unsafe sampling'
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=65536,
        help='Batch size for value inference'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='value_function_analysis_combined',
        help='Directory to save results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("SafeDICE Value Function Evaluation on Combined Datasets")
    print("=" * 80)
    
    # Load and combine datasets
    print("\nLoading and combining datasets...")
    combined_data = combine_datasets(args.dataset1, args.dataset2)
    
    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = ValueFunctionAnalyzer(args.weights)
    
    # Evaluate on combined dataset
    print("\n" + "=" * 80)
    print("Ground-Truth State Classification Evaluation")
    print("=" * 80)
    
    observations = combined_data['states']
    rewards = combined_data['rewards']
    costs = combined_data['costs']
    
    # Apply max_states limit if specified
    if args.max_states is not None:
        n = min(len(observations), int(args.max_states))
        observations = observations[:n]
        rewards = rewards[:n]
        costs = costs[:n]
        print(f"\nLimited to {n} states")

    # Build balanced safety dataset (equal safe/unsafe counts)
    if np.any(costs > 0):
        gt_unsafe_full = (costs > 0).astype(np.int32)
    else:
        gt_unsafe_full = (costs >= np.median(costs)).astype(np.int32)

    safe_idx = np.where(gt_unsafe_full == 0)[0]
    unsafe_idx = np.where(gt_unsafe_full == 1)[0]

    if len(safe_idx) == 0 or len(unsafe_idx) == 0:
        print("\nWarning: Could not balance classes because only one safety class exists.")
    else:
        target_per_class = min(len(safe_idx), len(unsafe_idx))
        rng = np.random.default_rng(args.sampling_seed)
        sampled_safe = rng.choice(safe_idx, size=target_per_class, replace=False)
        sampled_unsafe = rng.choice(unsafe_idx, size=target_per_class, replace=False)
        sampled_idx = np.concatenate([sampled_safe, sampled_unsafe], axis=0)
        rng.shuffle(sampled_idx)

        observations = observations[sampled_idx]
        rewards = rewards[sampled_idx]
        costs = costs[sampled_idx]

        print(
            f"\nBalanced sampling applied: {target_per_class} safe + {target_per_class} unsafe "
            f"= {2 * target_per_class} states (seed={args.sampling_seed})"
        )
    
    # Get value estimates
    print(f"Evaluating {len(observations)} states...")
    values = analyzer.get_state_values(observations, batch_size=args.eval_batch_size)
    
    print(f"Value estimates: mean={values.mean():.4f}, std={values.std():.4f}, "
          f"range=[{values.min():.4f}, {values.max():.4f}]")
    
    # Compute metrics directly on combined data
    value_reward_corr = analyzer._safe_corrcoef(values, rewards)
    value_cost_corr = analyzer._safe_corrcoef(values, costs)
    
    # Ground-truth labels
    reward_threshold = float(np.median(rewards))
    gt_high_reward = (rewards >= reward_threshold).astype(np.int32)
    
    if np.any(costs > 0):
        gt_unsafe = (costs > 0).astype(np.int32)
    else:
        gt_unsafe = (costs >= np.median(costs)).astype(np.int32)
    
    # Predicted labels
    value_reward_threshold = float(np.median(values))
    pred_high_reward = (values >= value_reward_threshold).astype(np.int32)
    
    unsafe_rate = float(np.mean(gt_unsafe))
    if 0.0 < unsafe_rate < 1.0:
        unsafe_value_threshold = float(np.percentile(values, unsafe_rate * 100.0))
        pred_unsafe = (values <= unsafe_value_threshold).astype(np.int32)
    else:
        unsafe_value_threshold = float(np.median(values))
        pred_unsafe = (values <= unsafe_value_threshold).astype(np.int32)
    
    # Metrics
    reward_metrics = analyzer._binary_classification_metrics(gt_high_reward, pred_high_reward)
    safety_metrics = analyzer._binary_classification_metrics(gt_unsafe, pred_unsafe)
    
    # Mean-cost check
    unsafe_mask = pred_unsafe == 1
    safe_mask = pred_unsafe == 0
    mean_cost_pred_unsafe = float(np.mean(costs[unsafe_mask])) if np.any(unsafe_mask) else float('nan')
    mean_cost_pred_safe = float(np.mean(costs[safe_mask])) if np.any(safe_mask) else float('nan')
    unsafe_has_higher_mean_cost = bool(mean_cost_pred_unsafe > mean_cost_pred_safe)
    
    # Print results
    print("\n" + "=" * 80)
    print("Results on Combined Dataset")
    print("=" * 80)
    print(f"States evaluated: {len(observations)}")
    print(f"Safe/Unsafe split: {(gt_unsafe==0).sum()} safe, {(gt_unsafe==1).sum()} unsafe "
          f"({100*unsafe_rate:.1f}% unsafe)")
    
    print(f"\n📊 Value-Reward Correlation: {value_reward_corr:.4f}")
    print(f"   ✓ Positive: {value_reward_corr > 0}")
    
    print(f"\n📊 Value-Cost Correlation: {value_cost_corr:.4f}")
    print(f"   ✓ Negative (for safety): {value_cost_corr < 0}")
    
    print(f"\n💰 Reward Classification:")
    print(f"   Accuracy: {reward_metrics['accuracy']:.4f}")
    print(f"   Precision: {reward_metrics['precision']:.4f}")
    print(f"   Recall: {reward_metrics['recall']:.4f}")
    print(f"   F1: {reward_metrics['f1']:.4f}")
    
    print(f"\n🛡️  Safety Classification:")
    print(f"   Accuracy: {safety_metrics['accuracy']:.4f}")
    print(f"   Precision: {safety_metrics['precision']:.4f}")
    print(f"   Recall: {safety_metrics['recall']:.4f}")
    print(f"   F1: {safety_metrics['f1']:.4f}")
    
    print(f"\n📈 Mean Cost Partitioning:")
    print(f"   Predicted unsafe mean cost: {mean_cost_pred_unsafe:.4f}")
    print(f"   Predicted safe mean cost: {mean_cost_pred_safe:.4f}")
    print(f"   ✓ Unsafe > Safe: {unsafe_has_higher_mean_cost}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'combined_evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("SafeDICE Value Function Evaluation on Combined Datasets\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Dataset 1: {args.dataset1}\n")
        f.write(f"Dataset 2: {args.dataset2}\n\n")
        f.write(f"Sampling seed: {args.sampling_seed}\n")
        f.write(f"Total states evaluated: {len(observations)}\n")
        f.write(f"Safe states: {(gt_unsafe==0).sum()} ({100*(gt_unsafe==0).mean():.1f}%)\n")
        f.write(f"Unsafe states: {(gt_unsafe==1).sum()} ({100*(gt_unsafe==1).mean():.1f}%)\n\n")
        f.write(f"Value-Reward Correlation: {value_reward_corr:.4f}\n")
        f.write(f"Value-Cost Correlation: {value_cost_corr:.4f}\n\n")
        f.write(f"Reward Classification Accuracy: {reward_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {reward_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {reward_metrics['recall']:.4f}\n")
        f.write(f"  F1: {reward_metrics['f1']:.4f}\n\n")
        f.write(f"Safety Classification Accuracy: {safety_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {safety_metrics['precision']:.4f}\n")
        f.write(f"  Recall: {safety_metrics['recall']:.4f}\n")
        f.write(f"  F1: {safety_metrics['f1']:.4f}\n\n")
        f.write(f"Mean cost (predicted unsafe): {mean_cost_pred_unsafe:.4f}\n")
        f.write(f"Mean cost (predicted safe): {mean_cost_pred_safe:.4f}\n")
        f.write(f"Unsafe has higher mean cost: {unsafe_has_higher_mean_cost}\n")
    
    print(f"\n✅ Saved results to: {summary_file}")


if __name__ == '__main__':
    main()
