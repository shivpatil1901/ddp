#!/usr/bin/env python3
"""
Compare value-function distributions and state rankings between two value files.

Usage:
    python value_function/compare_value_distributions_and_rankings.py \
        --file_a path/to/state_values_a.npz \
        --file_b path/to/state_values_b.npz \
        --output_dir /tmp/vf_compare \
        --top_k 100

The script expects each .npz to contain at least an array named `values` (shape [N])
and optionally `states` (shape [N, ...]) or `labels`. If `states` exist they are
used for diagnostics; otherwise comparisons are done on index-aligned values.

It computes:
 - Distribution comparison (histograms + KS test)
 - Rank correlation (Spearman rho) and Pearson on ranks
 - Pearson correlation on raw values
 - Overlap of bottom-K states (by rank)

Outputs: prints summary and writes plots to `output_dir`.
"""

import os
import argparse
import numpy as np
import math
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _load_npz(path):
    data = np.load(path, allow_pickle=True)
    # prefer key 'values', fall back to 'state_values' or 'y'
    if 'values' in data:
        vals = np.asarray(data['values']).astype(np.float64)
    elif 'state_values' in data:
        vals = np.asarray(data['state_values']).astype(np.float64)
    elif 'y' in data:
        vals = np.asarray(data['y']).astype(np.float64)
    else:
        # take the first 1D array found
        found = None
        for k in data.files:
            a = np.asarray(data[k])
            if a.ndim == 1:
                found = a.astype(np.float64)
                break
        if found is None:
            raise ValueError(f"No 1D value array found in {path}; found keys: {data.files}")
        vals = found
    states = data['states'] if 'states' in data else None
    return vals, states


def plot_distributions(a_vals, b_vals, out_path, title=None):
    plt.figure(figsize=(6,4))
    bins = 100
    plt.hist(a_vals, bins=bins, alpha=0.6, density=True, label='File A')
    plt.hist(b_vals, bins=bins, alpha=0.6, density=True, label='File B')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('Density')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(a_vals, b_vals, out_path, title=None):
    plt.figure(figsize=(5,5))
    plt.scatter(a_vals, b_vals, s=6, alpha=0.4)
    m = np.nanmean(a_vals)
    plt.axline((0,0),(1,1), color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('File A values')
    plt.ylabel('File B values')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def compare_values(a_vals, b_vals, top_k=100, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(a_vals), len(b_vals))
    a = a_vals[:n]
    b = b_vals[:n]

    # Basic stats
    stats_summary = {
        'a_mean': float(np.mean(a)),
        'a_std': float(np.std(a)),
        'b_mean': float(np.mean(b)),
        'b_std': float(np.std(b)),
    }

    # Distribution tests: KS
    ks_res = stats.ks_2samp(a, b)

    # Correlations
    # Pearson on raw
    pearson_r, pearson_p = stats.pearsonr(a, b)
    # Spearman (rank) correlation
    spearman_rho, spearman_p = stats.spearmanr(a, b)

    # Rank lists
    a_rank_idx = np.argsort(a)  # ascending: lowest first
    b_rank_idx = np.argsort(b)

    # Overlap of bottom-k
    k = min(int(top_k), n)
    a_bottom = set(a_rank_idx[:k].tolist())
    b_bottom = set(b_rank_idx[:k].tolist())
    bottom_overlap = len(a_bottom & b_bottom)
    bottom_overlap_frac = bottom_overlap / k

    # Rank correlation of ranks (convert to ranks)
    # ranks: smallest -> rank 0
    a_ranks = np.empty(n, dtype=np.int64)
    b_ranks = np.empty(n, dtype=np.int64)
    a_ranks[a_rank_idx] = np.arange(n)
    b_ranks[b_rank_idx] = np.arange(n)

    # Pearson on ranks
    pearson_ranks, pearson_ranks_p = stats.pearsonr(a_ranks, b_ranks)

    # Spearman between ranks should equal spearmanr(a,b), but compute explicitly
    spearman_ranks, spearman_ranks_p = stats.spearmanr(a_ranks, b_ranks)

    # Write plots
    plot_distributions(a, b, os.path.join(output_dir, 'value_distributions.png'),
                       title='Value distributions')
    plot_scatter(a, b, os.path.join(output_dir, 'value_scatter.png'), title='Value scatter')

    # CDF plot
    plt.figure(figsize=(6,4))
    xa = np.sort(a)
    xb = np.sort(b)
    ya = np.arange(1, len(xa)+1)/len(xa)
    yb = np.arange(1, len(xb)+1)/len(xb)
    plt.plot(xa, ya, label='A')
    plt.plot(xb, yb, label='B')
    plt.legend()
    plt.xlabel('Value')
    plt.ylabel('CDF')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'value_cdf.png'), dpi=150)
    plt.close()

    summary = {
        'n_compared': int(n),
        'ks_statistic': float(ks_res.statistic),
        'ks_pvalue': float(ks_res.pvalue),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'bottom_k': int(k),
        'bottom_overlap': int(bottom_overlap),
        'bottom_overlap_frac': float(bottom_overlap_frac),
        'pearson_ranks': float(pearson_ranks),
        'pearson_ranks_p': float(pearson_ranks_p),
    }
    summary.update(stats_summary)

    # Save summary
    import json
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_a', required=True)
    parser.add_argument('--file_b', required=True)
    parser.add_argument('--output_dir', default='value_function/compare_vf_output')
    parser.add_argument('--top_k', type=int, default=100)
    args = parser.parse_args()

    a_vals, a_states = _load_npz(args.file_a)
    b_vals, b_states = _load_npz(args.file_b)

    print(f"Loaded A: {args.file_a} values={a_vals.shape}")
    print(f"Loaded B: {args.file_b} values={b_vals.shape}")

    summary = compare_values(a_vals, b_vals, top_k=args.top_k, output_dir=args.output_dir)

    print('\n=== SUMMARY ===')
    for k,v in summary.items():
        print(f"{k}: {v}")
    print(f"Plots and summary written to {args.output_dir}")

if __name__ == '__main__':
    main()
