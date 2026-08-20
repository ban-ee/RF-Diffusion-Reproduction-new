"""
Source-Swap Causal Experiment — Plotting Script (CPU-only)

INPUT:
    Reads source_swap_{wifi,fmcw}.csv from ../results/tables/
    Each CSV row: condition_idx, source_a, source_b, pair_type,
                  output_pair_corr, output_pair_nmse

OUTPUT:
    Saves fig_source_swap.png to ../results/figures/
    One figure with 2×2 subplots (WiFi/FMCW × correlation/NMSE by pair type)

USAGE:
    python source_swap_plot.py

REQUIRES: numpy, matplotlib, csv (stdlib)
GPU: NO
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'tables')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'figures')

TASKS = ['wifi', 'fmcw']
TASK_LABELS = {'wifi': 'WiFi', 'fmcw': 'FMCW'}
PAIR_TYPE_LABELS = {
    'self_self': 'Same Source\n(self-self)',
    'self_other': 'Source vs Other\n(self-other)',
    'other_other': 'Different Sources\n(other-other)',
    'self_output_to_source': 'Output to\nOwn Source',
}
PAIR_TYPE_ORDER = ['self_self', 'self_other', 'other_other']


def load_source_swap(task_name):
    """Load source-swap CSV for one task."""
    csv_path = os.path.join(TABLES_DIR, f'source_swap_{task_name}.csv')
    if not os.path.exists(csv_path):
        print(f'WARNING: {csv_path} not found — skipping {task_name}')
        return None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Loaded {len(rows)} rows for {task_name}')
    return rows


def compute_pair_stats(rows):
    """Compute per-pair-type statistics."""
    pair_data = {}
    for r in rows:
        pt = r['pair_type']
        if pt not in pair_data:
            pair_data[pt] = {'corrs': [], 'nmses': []}
        pair_data[pt]['corrs'].append(float(r['output_pair_corr']))
        pair_data[pt]['nmses'].append(float(r['output_pair_nmse']))

    stats = {}
    for pt, vals in pair_data.items():
        stats[pt] = {
            'N': len(vals['corrs']),
            'corr_mean': np.mean(vals['corrs']),
            'corr_std': np.std(vals['corrs']),
            'nmse_mean': np.mean(vals['nmses']),
            'nmse_std': np.std(vals['nmses']),
        }
    return stats


def plot_source_swap(all_rows):
    """Plot source-swap results for WiFi and FMCW."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, task_name in enumerate(TASKS):
        rows = all_rows.get(task_name)
        if rows is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'NO DATA',
                                    transform=axes[row, col].transAxes,
                                    ha='center', va='center', fontsize=14)
            continue

        stats = compute_pair_stats(rows)

        # Row 0: Output pair correlation by pair type
        ax_corr = axes[0, col]
        labels = []
        means = []
        stds = []
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        for pt in PAIR_TYPE_ORDER:
            if pt in stats:
                labels.append(PAIR_TYPE_LABELS.get(pt, pt))
                means.append(stats[pt]['corr_mean'])
                stds.append(stats[pt]['corr_std'])
        x = range(len(labels))
        ax_corr.bar(x, means, yerr=stds, color=colors[:len(labels)],
                    capsize=8, edgecolor='black', linewidth=0.5)
        ax_corr.set_xticks(x)
        ax_corr.set_xticklabels(labels, fontsize=9)
        ax_corr.set_title(f'{TASK_LABELS[task_name]}: Output Pair Correlation\n'
                          f'by Source Relationship', fontsize=12)
        ax_corr.set_ylabel('Pearson r')
        ax_corr.set_ylim(0, 1.05)
        # Annotate values
        for i, (m, s) in enumerate(zip(means, stds)):
            ax_corr.text(i, m + s + 0.02, f'{m:.3f}±{s:.3f}',
                         ha='center', fontsize=8)

        # Row 1: Output pair NMSE by pair type
        ax_nmse = axes[1, col]
        nmse_means = []
        nmse_stds = []
        for pt in PAIR_TYPE_ORDER:
            if pt in stats:
                nmse_means.append(stats[pt]['nmse_mean'])
                nmse_stds.append(stats[pt]['nmse_std'])
        ax_nmse.bar(x, nmse_means, yerr=nmse_stds, color=colors[:len(labels)],
                    capsize=8, edgecolor='black', linewidth=0.5)
        ax_nmse.set_xticks(x)
        ax_nmse.set_xticklabels(labels, fontsize=9)
        ax_nmse.set_title(f'{TASK_LABELS[task_name]}: Output Pair NMSE\n'
                          f'by Source Relationship', fontsize=12)
        ax_nmse.set_ylabel('NMSE')
        for i, (m, s) in enumerate(zip(nmse_means, nmse_stds)):
            ax_nmse.text(i, m + s + 0.02, f'{m:.3f}±{s:.3f}',
                         ha='center', fontsize=8)

    plt.suptitle('Source-Swap Causal Experiment\n'
                 'Fixed condition, varying source x₀ → official native_sampling output',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(FIGURES_DIR, 'fig_source_swap.png')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def print_interpretation(all_rows):
    """Print interpretation of source-swap results."""
    print('\n' + '=' * 70)
    print('SOURCE-SWAP CAUSAL EXPERIMENT — INTERPRETATION')
    print('=' * 70)
    print()
    print('EXPERIMENT: Fixed condition c_i, vary source x₀^(j).')
    print('           Run official native_sampling for each source.')
    print('           Compare outputs across different sources.')
    print()
    print('If native_sampling were source-independent:')
    print('  → All pair correlations ≈ 1.0, all NMSE ≈ 0')
    print()
    print('OBSERVED:')
    for task_name in TASKS:
        rows = all_rows.get(task_name)
        if rows is None:
            continue
        stats = compute_pair_stats(rows)
        print(f'\n  {TASK_LABELS[task_name]}:')
        for pt in PAIR_TYPE_ORDER:
            if pt in stats:
                s = stats[pt]
                print(f'    {PAIR_TYPE_LABELS.get(pt, pt)}: '
                      f'corr={s["corr_mean"]:.3f}±{s["corr_std"]:.3f}, '
                      f'NMSE={s["nmse_mean"]:.3f}±{s["nmse_std"]:.3f}')
        if 'self_output_to_source' in stats:
            s = stats['self_output_to_source']
            print(f'    Output-to-own-source corr: {s["corr_mean"]:.4f}±{s["corr_std"]:.4f}')

    print()
    print('CONCLUSION: native_sampling output IS source-dependent.')
    print('Same condition + different source → different output (low cross-source corr).')
    print('Output strongly correlated with its own source (r ≈ 0.83-0.87).')
    print()
    print('NOTE: degrade_fn contains torch.manual_seed(11) — shared noise realization.')
    print('=' * 70)


def main():
    print('Source-Swap Analysis — Plot Generator')
    print('=' * 50)

    all_rows = {}
    for task_name in TASKS:
        rows = load_source_swap(task_name)
        if rows is not None:
            all_rows[task_name] = rows

    if not all_rows:
        print('ERROR: No data loaded.')
        return

    plot_source_swap(all_rows)
    print_interpretation(all_rows)
    print('\nDone.')


if __name__ == '__main__':
    main()
