#!/usr/bin/env python
"""
Inspect prediction slope: visualize g_t (cumulative value prediction) over timesteps.

If the model has learned features, g_t should be a smooth increasing curve towards
the final cumulative return. If it's mostly flat until the last timestep jumps to 1.0,
the LSTM hasn't learned meaningful features yet.

This version focuses on the combined reward/cost cohorts:
  - High reward / low cost
  - Low reward / high cost

Usage:
    python rudder/eval/inspect_prediction_slope.py \
        --checkpoint rudder/models/reinforce_rudder_combined.pt \
        --dataset rudder/dataset/combined_cost_reward_balanced_1800.pkl \
        --eval_dataset rudder/dataset/combined_cost_reward_holdout_200.pkl \
        --num_trajectories 8 \
        --output_dir rudder/eval/slope_plots
"""

import os
import sys
try:
    import pickle5 as pickle
except ImportError:
    import pickle
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for WSL/headless environments
import matplotlib.pyplot as plt

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from trainer.rudder_train import OriginalRUDDER


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _resolve_input_path(path):
    raw = os.path.expanduser(os.path.expandvars(path))
    normalized = raw.replace('\\', os.sep).replace('/', os.sep)

    if os.path.isabs(normalized):
        return normalized

    cwd_candidate = os.path.abspath(normalized)
    repo_candidate = os.path.abspath(os.path.join(_repo_root(), normalized))
    if os.path.isfile(cwd_candidate):
        return cwd_candidate
    return repo_candidate


def _load_pickle_robust(path):
    """Load pickle with robust error handling."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}\nGenerate via: python rudder/build_reward_balanced_trajectories.py")
    
    if os.path.getsize(path) == 0:
        raise RuntimeError(f"File is empty: {path}")
    
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        if "protocol" in str(e).lower():
            raise RuntimeError(f"Pickle protocol error: {e}\nTry: pip install pickle5 or use Python >= 3.8")
        try:
            with open(path, 'rb') as f:
                return pickle.load(f, encoding='latin1')
        except Exception as e2:
            raise RuntimeError(f"Failed to load pickle {path}: {e2}")


def _load_model_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load metadata
    state_dim = checkpoint.get('state_dim', 55)
    action_dim = checkpoint.get('action_dim', 3)
    baseline = checkpoint.get('baseline', 0.0)
    
    # Reconstruct model
    model = OriginalRUDDER(state_dim, action_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    
    return model, baseline, state_dim, action_dim


def compute_gt_trajectory(model, states, actions, baseline, device='cpu'):
    """
    Compute g_t predictions for all timesteps in a trajectory.
    
    Args:
        model: LSTM model
        states: (seq_len, state_dim) numpy array
        actions: (seq_len, action_dim) numpy array
        baseline: baseline value for centering
        device: torch device
    
    Returns:
        g_t: (seq_len,) numpy array of cumulative predictions
    """
    # Concatenate states and actions
    x = np.concatenate([states, actions], axis=1)  # (seq_len, state_dim + action_dim)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)  # (1, seq_len, state_dim + action_dim)
    
    with torch.no_grad():
        # Get all hidden states + output
        output = model(x)  # (1, seq_len, 1)
        g_t = output[0, :, 0].cpu().numpy() + baseline  # (seq_len,)
    
    return g_t


def _trajectory_reward_cost(traj):
    rewards = np.asarray(traj['rewards'], dtype=np.float32).reshape(-1)
    costs = np.asarray(traj['costs'], dtype=np.float32).reshape(-1)
    return float(np.sum(rewards)), float(np.sum(costs))


def _trajectory_group(reward_sum, cost_sum, reward_threshold, cost_threshold):
    if reward_sum > reward_threshold and cost_sum < cost_threshold:
        return 'High reward / low cost', 1, 'green'
    if reward_sum <= reward_threshold and cost_sum >= cost_threshold:
        return 'Low reward / high cost', 0, 'red'
    return None, None, None


def select_diverse_trajectories(data, num_to_select=8, reward_threshold=15, cost_threshold=25):
    """
    Select trajectories from the two combined reward/cost cohorts.
    
    Returns:
        list of (trajectory_dict, group_label, reward_sum, cost_sum) tuples
    """
    trajectories = data['trajectories']
    selected = []

    high_low_candidates = []
    low_high_candidates = []

    for traj_idx, traj in enumerate(trajectories):
        reward_sum, cost_sum = _trajectory_reward_cost(traj)
        group_label, group_id, _ = _trajectory_group(
            reward_sum,
            cost_sum,
            reward_threshold,
            cost_threshold,
        )
        if group_id == 1:
            high_low_candidates.append((traj_idx, reward_sum, cost_sum))
        elif group_id == 0:
            low_high_candidates.append((traj_idx, reward_sum, cost_sum))

    high_low_candidates.sort(key=lambda item: (item[1] - item[2], item[1]), reverse=True)
    low_high_candidates.sort(key=lambda item: (item[2] - item[1], item[2]), reverse=True)

    num_high_low = num_to_select // 2
    num_low_high = num_to_select - num_high_low

    for traj_idx, reward_sum, cost_sum in high_low_candidates[:num_high_low]:
        selected.append((trajectories[traj_idx], 'High reward / low cost', reward_sum, cost_sum))

    for traj_idx, reward_sum, cost_sum in low_high_candidates[:num_low_high]:
        selected.append((trajectories[traj_idx], 'Low reward / high cost', reward_sum, cost_sum))

    return selected


def plot_trajectory_predictions(model, trajectories, baseline, output_dir, device='cpu', dataset_tag='train'):
    """
    Plot g_t curves for selected trajectories.
    
    Args:
        model: LSTM model
        trajectories: list of (traj_dict, label, cumsum) tuples
        baseline: baseline centering value
        output_dir: directory to save plots
        device: torch device
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual trajectory plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Prediction Slope: $g_t$ Evolution Over Timesteps\n(Should be smooth curve, not flat+jump)', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color='green', lw=2, label='High reward / low cost'),
        Line2D([0], [0], color='red', lw=2, label='Low reward / high cost'),
        Line2D([0], [0], color='blue', lw=1.5, ls='--', label='Target (binary classification)'),
    ]
    
    for idx, (traj, group_label, reward_sum, cost_sum) in enumerate(trajectories[:4]):
        ax = axes[idx]
        
        states = traj['states'].astype(np.float32)
        actions = traj['actions'].astype(np.float32)
        
        # Compute g_t
        g_t = compute_gt_trajectory(model, states, actions, baseline, device)
        seq_len = len(g_t)
        
        curve_color = 'green' if group_label == 'High reward / low cost' else 'red'

        # Plot
        ax.plot(range(seq_len), g_t, color=curve_color, linewidth=2, alpha=0.7, label=f'{group_label} $g_t$')
        ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.5, label='Target (binary classification)')
        ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('$g_t$ (cumulative value)')
        ax.set_title(f'{group_label} | reward={reward_sum:.2f} | cost={cost_sum:.2f} | seq_len={seq_len}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_ylim([-0.1, 1.2])
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'prediction_slopes_sample_{dataset_tag}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved sample plots to {plot_path}")
    plt.close()
    
    # Create a comprehensive grid plot with all trajectories
    num_trajectories = len(trajectories)
    grid_size = int(np.ceil(np.sqrt(num_trajectories)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(16, 14))
    fig.suptitle('All Sampled Trajectories: Prediction Slope Analysis', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, (traj, group_label, reward_sum, cost_sum) in enumerate(trajectories):
        ax = axes[idx]
        
        states = traj['states'].astype(np.float32)
        actions = traj['actions'].astype(np.float32)
        
        # Compute g_t
        g_t = compute_gt_trajectory(model, states, actions, baseline, device)
        seq_len = len(g_t)
        
        color = 'green' if group_label == 'High reward / low cost' else 'red'

        # Plot
        ax.plot(range(seq_len), g_t, color=color, linewidth=1.5, alpha=0.8)
        ax.axhline(y=1.0, color='blue', linestyle='--', alpha=0.3, linewidth=1)
        ax.set_title(f'{group_label} (reward={reward_sum:.1f}, cost={cost_sum:.1f})', fontsize=9, color=color)
        ax.set_xlim([0, seq_len])
        ax.set_ylim([-0.1, 1.2])
        ax.grid(True, alpha=0.2)
        
        # Minimal labels
        if idx % grid_size == 0:
            ax.set_ylabel('$g_t$', fontsize=8)
        if idx >= num_trajectories - grid_size:
            ax.set_xlabel('Step', fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_trajectories, len(axes)):
        axes[idx].set_visible(False)

    fig.legend(handles=legend_handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.99))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'prediction_slopes_all_{dataset_tag}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved all plots to {plot_path}")
    plt.close()


def analyze_slope_quality(model, trajectories, baseline, device='cpu'):
    """
    Quantitatively analyze prediction slopes.
    
    Compute: variance of g_t within trajectory, to detect flat-then-jump patterns.
    """
    print("\n=== Slope Quality Analysis ===")
    print(f"{'Group':<24} {'Reward':<12} {'Cost':<12} {'SeqLen':<10} {'Mean g_t':<12} {'Std g_t':<12} {'Shape Quality':<30}")
    print("-" * 96)

    for idx, (traj, group_label, reward_sum, cost_sum) in enumerate(trajectories):
        states = traj['states'].astype(np.float32)
        actions = traj['actions'].astype(np.float32)
        
        g_t = compute_gt_trajectory(model, states, actions, baseline, device)
        
        mean_g = np.mean(g_t)
        std_g = np.std(g_t)
        seq_len = len(g_t)
        
        # Check if curve is "flat then jump" pattern
        mid_point = seq_len // 2
        early_mean = np.mean(g_t[:mid_point])
        late_mean = np.mean(g_t[mid_point:])
        jump_ratio = late_mean / (early_mean + 1e-6)
        
        # Shape quality: good if smooth progression (low std, moderate progression)
        if std_g > 0.3:
            quality = "✓ GOOD (varied progression)"
        elif jump_ratio > 2.0:
            quality = "✗ POOR (flat then jump)"
        else:
            quality = "~ OK (some progression)"

        print(f"{group_label:<24} {reward_sum:<12.2f} {cost_sum:<12.2f} {seq_len:<10} {mean_g:<12.4f} {std_g:<12.4f} {quality:<30}")


def run_dataset_inspection(model, baseline, dataset_path, num_trajectories, output_dir, device, dataset_tag):
    print(f"\n[{dataset_tag}] Loading dataset from {dataset_path}...")
    data = _load_pickle_robust(dataset_path)
    print(f"  ✓ Loaded {len(data['trajectories'])} trajectories")

    metadata = data.get('metadata', {}) if isinstance(data, dict) else {}
    reward_threshold = float(metadata.get('reward_threshold', 15.0))
    cost_threshold = float(metadata.get('cost_threshold', 25.0))
    print(f"[{dataset_tag}] Using thresholds: reward > {reward_threshold:.2f}, cost < {cost_threshold:.2f}")

    print(f"[{dataset_tag}] Selecting {num_trajectories} diverse trajectories...")
    trajectories = select_diverse_trajectories(
        data,
        num_to_select=num_trajectories,
        reward_threshold=reward_threshold,
        cost_threshold=cost_threshold,
    )
    print(f"  ✓ Selected {len(trajectories)} trajectories (high reward / low cost vs low reward / high cost)")

    print(f"[{dataset_tag}] Analyzing slopes and generating plots...")
    analyze_slope_quality(model, trajectories, baseline, device=device)
    plot_trajectory_predictions(
        model,
        trajectories,
        baseline,
        output_dir,
        device=device,
        dataset_tag=dataset_tag,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=str, default='rudder/models/reward_rudder_1_best.pt',
                        help='Path to best checkpoint')
    parser.add_argument('--dataset', type=str, default='rudder/dataset/reward_balanced_1800.pkl',
                        help='Path to training dataset')
    parser.add_argument('--eval_dataset', type=str, default='rudder/dataset/combined_reward_eval_nonoverlap_200.pkl',
                        help='Path to non-overlap eval dataset')
    parser.add_argument('--num_trajectories', type=int, default=8,
                        help='Number of trajectories to sample and plot')
    parser.add_argument('--output_dir', type=str, default='rudder/eval/slope_plots',
                        help='Output directory for plots')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PREDICTION SLOPE INSPECTION")
    print("=" * 80)
    
    train_dataset_path = _resolve_input_path(args.dataset)
    eval_dataset_path = _resolve_input_path(args.eval_dataset)
    checkpoint_path = _resolve_input_path(args.checkpoint)
    
    # Load model
    print(f"\n[1/3] Loading checkpoint from {checkpoint_path}...")
    model, baseline, state_dim, action_dim = _load_model_checkpoint(checkpoint_path, device=args.device)
    print(f"  ✓ Model: state_dim={state_dim}, action_dim={action_dim}")
    print(f"  ✓ Baseline: {baseline:.4f}")

    print(f"\n[2/3] Inspecting training dataset...")
    run_dataset_inspection(
        model,
        baseline,
        train_dataset_path,
        args.num_trajectories,
        args.output_dir,
        args.device,
        dataset_tag='train',
    )

    print(f"\n[3/3] Inspecting eval dataset...")
    run_dataset_inspection(
        model,
        baseline,
        eval_dataset_path,
        args.num_trajectories,
        args.output_dir,
        args.device,
        dataset_tag='eval',
    )
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("=" * 80)
    print("""
✓ GOOD (model learning features):
  - Shape is smooth curve, gradually increasing towards 1.0
  - g_t values distributed across range, not clustered near 0 or 1
    - High reward / low cost trajs show steeper progression than low reward / high cost trajs
  - Standard deviation fairly high (indicative of varied progression)

✗ POOR (model NOT learning features):
  - Shape is mostly flat near 0, then sudden jump to 1.0 at end
  - All values clustered near 0 until final timestep
    - No discernible difference between the two reward/cost cohorts
  - Standard deviation very low (flat line)
  - AUROC near 0.5 on step-wise positive reward detection

→ If slopes are POOR, the model is essentially ignoring LSTM features and just
  memorizing the final label. This explains why step-wise correlation is near-zero.
  
✗ Common causes:
  - Training gradient not flowing properly (but gradient clipping should help)
  - Baseline normalization removing all signal
  - LSTM learning trivial solution (e.g., constant prediction)
  
→ If slopes are GOOD but step-wise correlation still near-zero, the problem is
  that the identity decomposition (g_t - g_{t-1}) doesn't align with true rewards.
  Solution: auxiliary step-wise loss during training.
    """)
    print("=" * 80)


if __name__ == '__main__':
    main()
