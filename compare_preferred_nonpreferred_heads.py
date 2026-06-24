#!/usr/bin/env python
"""
Compare learned reward and cost heads on preferred vs non-preferred demo datasets.

Preferred:     ppo_lagrangian_PointGoal1_s0.pickle (low cost, safer trajectories)
Non-Preferred: ppo_PointGoal1_s0.pickle (high cost, lower reward)

Shows V_reward(s0) and cost_head(s0, a0) distributions side-by-side to assess
whether the heads have learned to differentiate between the two trajectory types.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

# Add SafeDICE to path
safedice_path = Path(__file__).parent / 'SafeDICE'
sys.path.insert(0, str(safedice_path))

from analyze_value_function import ValueFunctionAnalyzer


def _draw_value_trace_grid(
    analyzer: ValueFunctionAnalyzer,
    preferred_dataset: str,
    non_preferred_dataset: str,
    max_trajectories: int,
    num_sample_trajectories: int,
    sample_seed: int,
    output_path: str,
) -> int:
    """Plot value traces for sampled trajectories from both sets side-by-side."""
    pref_trajs = analyzer._load_dataset_trajectories(preferred_dataset, max_trajectories=max_trajectories)
    non_pref_trajs = analyzer._load_dataset_trajectories(non_preferred_dataset, max_trajectories=max_trajectories)

    if not pref_trajs or not non_pref_trajs:
        return 0

    n_rows = min(num_sample_trajectories, len(pref_trajs), len(non_pref_trajs))
    if n_rows <= 0:
        return 0

    rng = np.random.default_rng(sample_seed)
    pref_idx = rng.choice(len(pref_trajs), size=n_rows, replace=False)
    non_pref_idx = rng.choice(len(non_pref_trajs), size=n_rows, replace=False)

    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3.2 * n_rows), squeeze=False)

    for row in range(n_rows):
        pref_traj = pref_trajs[int(pref_idx[row])]
        non_pref_traj = non_pref_trajs[int(non_pref_idx[row])]

        for col, traj, col_name, color in (
            (0, pref_traj, 'Preferred', 'tab:green'),
            (1, non_pref_traj, 'Non-Preferred', 'tab:red'),
        ):
            ax = axes[row][col]
            states = np.asarray(traj.get('observations', traj.get('states')))
            if states is None or len(states) == 0:
                ax.set_title(f"{col_name} | empty trajectory")
                ax.grid(alpha=0.25)
                continue

            values = analyzer.get_state_values(states)
            rewards = traj.get('rewards', None)
            costs = traj.get('costs', None)

            reward_sum = float(np.sum(rewards)) if rewards is not None else float('nan')
            cost_sum = float(np.sum(costs)) if costs is not None else float('nan')

            ax.plot(values, linewidth=2.0, color=color)
            ax.fill_between(np.arange(len(values)), values, alpha=0.15, color=color)
            ax.set_xlabel('Time step')
            ax.set_ylabel('V(s)')
            ax.set_title(
                f"{col_name} #{row + 1} | len={len(values)} | return={reward_sum:.2f} | cost={cost_sum:.2f}"
            )
            ax.grid(alpha=0.25)

    fig.suptitle('Value-Function Traces on Sampled Trajectories', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return n_rows


def main():
    parser = argparse.ArgumentParser(
        description='Compare reward/cost head outputs on preferred vs non-preferred demos'
    )
    parser.add_argument(
        '--weights',
        type=str,
        default='SafeDICE/weights/antidice_PointGoal1_seed0_20260330_205404_iter900000.pickle',
        help='Path to SafeDICE checkpoint',
    )
    parser.add_argument(
        '--preferred-dataset',
        type=str,
        default='SafeDICE/dataset/safetygym/ppo_lagrangian_PointGoal1_s0.pickle',
        help='Preferred demos (low-cost, safer)',
    )
    parser.add_argument(
        '--non-preferred-dataset',
        type=str,
        default='SafeDICE/dataset/safetygym/ppo_PointGoal1_s0.pickle',
        help='Non-preferred demos (high-cost, lower-reward)',
    )
    parser.add_argument(
        '--max-trajectories',
        type=int,
        default=200,
        help='Max trajectories per dataset',
    )
    parser.add_argument(
        '--env-cost-limit',
        type=float,
        default=25.0,
        help='Cumulative cost threshold for safe/unsafe',
    )
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=65536,
        help='Batch size for inference',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='value_function_analysis_combined',
        help='Output directory',
    )
    parser.add_argument(
        '--num-sample-trajectories',
        type=int,
        default=5,
        help='How many trajectories per set to plot in the side-by-side value-trace figure',
    )
    parser.add_argument(
        '--sample-seed',
        type=int,
        default=42,
        help='Random seed for selecting sample trajectories for plots',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 88)
    print("Comparing Heads on Preferred vs Non-Preferred Demos")
    print("=" * 88)

    # Initialize analyzer once
    print(f"\nLoading checkpoint: {args.weights}")
    analyzer = ValueFunctionAnalyzer(args.weights)

    # Analyze preferred demos
    print(f"\n{'='*88}")
    print("PREFERRED DEMOS (ppo_lagrangian - low cost, safer)")
    print(f"{'='*88}")
    preferred_eval = analyzer.analyze_initial_state_heads(
        dataset_path=args.preferred_dataset,
        max_trajectories=args.max_trajectories,
        env_cost_limit=args.env_cost_limit,
        eval_batch_size=args.eval_batch_size,
    )

    if 'error' not in preferred_eval:
        rh_pref = preferred_eval['reward_head']
        ch_pref = preferred_eval['cost_head']
        pref_stats = rh_pref['preferred_v_s0_range']
        print(f"  Trajectories used: {preferred_eval['num_trajectories_used']}")
        print(f"  V(s0) mean: {pref_stats['mean']:.6f}, std: {pref_stats['count']}")
        print(f"  V(s0) range: [{pref_stats['min']:.6f}, {pref_stats['max']:.6f}]")
        print(f"  Env safe/unsafe: {ch_pref['num_safe_env']} safe, {ch_pref['num_unsafe_env']} unsafe")
        if 'safe_cost_head_s0_range' in ch_pref:
            safe_cost = ch_pref['safe_cost_head_s0_range']
            unsafe_cost = ch_pref['unsafe_cost_head_s0_range']
            print(f"  Cost head on safe demos: mean={safe_cost['mean']:.6f}, std={safe_cost['count']}")
            print(f"  Cost head on unsafe demos: mean={unsafe_cost['mean']:.6f}, std={unsafe_cost['count']}")
    else:
        print(f"  ERROR: {preferred_eval['error']}")

    # Analyze non-preferred demos
    print(f"\n{'='*88}")
    print("NON-PREFERRED DEMOS (ppo - high cost, lower reward)")
    print(f"{'='*88}")
    non_pref_eval = analyzer.analyze_initial_state_heads(
        dataset_path=args.non_preferred_dataset,
        max_trajectories=args.max_trajectories,
        env_cost_limit=args.env_cost_limit,
        eval_batch_size=args.eval_batch_size,
    )

    if 'error' not in non_pref_eval:
        rh_non = non_pref_eval['reward_head']
        ch_non = non_pref_eval['cost_head']
        non_pref_stats = rh_non['preferred_v_s0_range']  # Note: this is still labeled "preferred_v_s0_range" in the dict
        print(f"  Trajectories used: {non_pref_eval['num_trajectories_used']}")
        print(f"  V(s0) mean: {non_pref_stats['mean']:.6f}")
        print(f"  V(s0) range: [{non_pref_stats['min']:.6f}, {non_pref_stats['max']:.6f}]")
        print(f"  Env safe/unsafe: {ch_non['num_safe_env']} safe, {ch_non['num_unsafe_env']} unsafe")
        if 'safe_cost_head_s0_range' in ch_non:
            safe_cost = ch_non['safe_cost_head_s0_range']
            unsafe_cost = ch_non['unsafe_cost_head_s0_range']
            print(f"  Cost head on safe demos: mean={safe_cost['mean']:.6f}")
            print(f"  Cost head on unsafe demos: mean={unsafe_cost['mean']:.6f}")
    else:
        print(f"  ERROR: {non_pref_eval['error']}")

    # Comparison
    print(f"\n{'='*88}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*88}")

    if 'error' not in preferred_eval and 'error' not in non_pref_eval:
        rh_pref = preferred_eval['reward_head']
        rh_non = non_pref_eval['reward_head']
        ch_pref = preferred_eval['cost_head']
        ch_non = non_pref_eval['cost_head']

        pref_v_s0 = rh_pref['preferred_v_s0_range']
        non_v_s0 = rh_non['preferred_v_s0_range']

        v_sep = pref_v_s0['mean'] - non_v_s0['mean']
        print(f"\n📊 REWARD HEAD V(s0) Differentiating Power:")
        print(f"  Preferred mean V(s0):     {pref_v_s0['mean']:9.6f}  (range: {pref_v_s0['min']:.4f} to {pref_v_s0['max']:.4f})")
        print(f"  Non-preferred mean V(s0): {non_v_s0['mean']:9.6f}  (range: {non_v_s0['min']:.4f} to {non_v_s0['max']:.4f})")
        print(f"  Separation (pref - non):  {v_sep:9.6f}")
        if abs(v_sep) > 0.5:
            print(f"  ✓ STRONG differentiation: Preferred demos have {'higher' if v_sep > 0 else 'lower'} V(s0)")
        elif abs(v_sep) > 0.1:
            print(f"  ~ MODERATE differentiation: Some gap but overlapping ranges")
        else:
            print(f"  ✗ WEAK differentiation: Heads treat demos similarly")

        if 'safe_cost_head_s0_range' in ch_pref and 'safe_cost_head_s0_range' in ch_non:
            safe_pref = ch_pref['safe_cost_head_s0_range']
            unsafe_pref = ch_pref['unsafe_cost_head_s0_range']
            safe_non = ch_non['safe_cost_head_s0_range']
            unsafe_non = ch_non['unsafe_cost_head_s0_range']

            print(f"\n🛡️  COST HEAD Differentiating Power (within-dataset safe/unsafe):")
            print(f"  Preferred demos:")
            print(f"    Safe mean cost_head(s0,a0):   {safe_pref['mean']:9.6f}")
            print(f"    Unsafe mean cost_head(s0,a0): {unsafe_pref['mean']:9.6f}")
            print(f"    Separation:                   {safe_pref['mean'] - unsafe_pref['mean']:9.6f}")
            print(f"  Non-preferred demos:")
            print(f"    Safe mean cost_head(s0,a0):   {safe_non['mean']:9.6f}")
            print(f"    Unsafe mean cost_head(s0,a0): {unsafe_non['mean']:9.6f}")
            print(f"    Separation:                   {safe_non['mean'] - unsafe_non['mean']:9.6f}")

        print(f"\n📈 Correlation with trajectory returns (reward head signal quality):")
        print(f"  Preferred corr(V(s0), return): {rh_pref['corr_v_s0_vs_traj_return']:.6f}")
        print(f"  Non-pref corr(V(s0), return):  {rh_non['corr_v_s0_vs_traj_return']:.6f}")

        print(f"\n📈 Correlation with env costs (cost head signal quality):")
        if 'corr_cost_head_s0_vs_traj_env_cost' in ch_pref:
            print(f"  Preferred corr(cost_head, env_cost): {ch_pref['corr_cost_head_s0_vs_traj_env_cost']:.6f}")
        if 'corr_cost_head_s0_vs_traj_env_cost' in ch_non:
            print(f"  Non-pref corr(cost_head, env_cost):  {ch_non['corr_cost_head_s0_vs_traj_env_cost']:.6f}")

        # Summary interpretation
        print(f"\n{'='*88}")
        print("INTERPRETATION")
        print(f"{'='*88}")
        
        if abs(v_sep) > 0.5:
            print("✓ Reward head shows strong preference: valuable for trajectory ranking τ_A ⪰ τ_B")
        else:
            print("✗ Reward head is weak at ranking preferred demos; consider retraining")
        
        if 'safe_cost_head_s0_range' in ch_pref:
            safe_gap_pref = safe_pref['mean'] - unsafe_pref['mean']
            safe_gap_non = safe_non['mean'] - unsafe_non['mean']
            if abs(safe_gap_pref) > 0.2 or abs(safe_gap_non) > 0.2:
                print("✓ Cost head shows signal for safety classification in at least one dataset")
            else:
                print("✗ Cost head is weak; consider using explicit cost signal or retraining")

    if args.num_sample_trajectories > 0:
        sample_plot_path = os.path.join(args.output_dir, 'sample_trajectory_value_traces.png')
        plotted = _draw_value_trace_grid(
            analyzer=analyzer,
            preferred_dataset=args.preferred_dataset,
            non_preferred_dataset=args.non_preferred_dataset,
            max_trajectories=args.max_trajectories,
            num_sample_trajectories=args.num_sample_trajectories,
            sample_seed=args.sample_seed,
            output_path=sample_plot_path,
        )
        if plotted > 0:
            print(f"\n✅ Saved sampled value-trace figure ({plotted} per set) to: {sample_plot_path}")
        else:
            print("\n⚠️  Skipped sampled value-trace figure (no valid trajectories found).")

    # Save results
    save_file = os.path.join(args.output_dir, 'preferred_vs_nonpreferred_headcomparison.txt')
    with open(save_file, 'w') as f:
        f.write("Preferred vs Non-Preferred Head Comparison\n")
        f.write("=" * 88 + "\n\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Preferred dataset: {args.preferred_dataset}\n")
        f.write(f"Non-preferred dataset: {args.non_preferred_dataset}\n")
        f.write(f"Max trajectories per dataset: {args.max_trajectories}\n\n")

        if 'error' not in preferred_eval:
            f.write("PREFERRED RESULTS:\n")
            rh_pref = preferred_eval['reward_head']
            ch_pref = preferred_eval['cost_head']
            pref_stats = rh_pref['preferred_v_s0_range']
            f.write(f"  Trajectories: {preferred_eval['num_trajectories_used']}\n")
            f.write(f"  V(s0) mean/range: {pref_stats['mean']:.6f} [{pref_stats['min']:.4f}, {pref_stats['max']:.4f}]\n")
            f.write(f"  Env safe/unsafe: {ch_pref['num_safe_env']} / {ch_pref['num_unsafe_env']}\n\n")

        if 'error' not in non_pref_eval:
            f.write("NON-PREFERRED RESULTS:\n")
            rh_non = non_pref_eval['reward_head']
            ch_non = non_pref_eval['cost_head']
            non_pref_stats = rh_non['preferred_v_s0_range']
            f.write(f"  Trajectories: {non_pref_eval['num_trajectories_used']}\n")
            f.write(f"  V(s0) mean/range: {non_pref_stats['mean']:.6f} [{non_pref_stats['min']:.4f}, {non_pref_stats['max']:.4f}]\n")
            f.write(f"  Env safe/unsafe: {ch_non['num_safe_env']} / {ch_non['num_unsafe_env']}\n\n")

        if 'error' not in preferred_eval and 'error' not in non_pref_eval:
            v_sep = pref_stats['mean'] - non_pref_stats['mean']
            f.write("COMPARISON:\n")
            f.write(f"  V(s0) separation: {v_sep:.6f}\n")
            if abs(v_sep) > 0.5:
                f.write(f"  Verdict: STRONG differentiation\n")
            elif abs(v_sep) > 0.1:
                f.write(f"  Verdict: MODERATE differentiation\n")
            else:
                f.write(f"  Verdict: WEAK differentiation\n")

    print(f"\n✅ Saved comparison to: {save_file}")


if __name__ == '__main__':
    main()
