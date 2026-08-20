"""
Terminal Consistency Analysis — Plotting Script (CPU-only)

INPUT:
    Reads terminal_metrics_{wifi,fmcw,5g}.csv from ../results/tables/
    Each CSV row: sample_idx, pearson_corr, nmse, energy_ratio, signal_projection,
                  energy_x0, energy_xT

OUTPUT:
    Saves fig_terminal_distributions.png to ../results/figures/
    One figure with 3×2 subplots (3 tasks × 2 key metrics)

USAGE:
    python terminal_consistency_plot.py

REQUIRES: numpy, matplotlib, csv (stdlib)
GPU: NO
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'tables')
FIGURES_DIR = os.path.join(SCRIPT_DIR, '..', 'results', 'figures')

TASKS = ['wifi', 'fmcw', '5g']
TASK_LABELS = {'wifi': 'WiFi (T=100)', 'fmcw': 'FMCW (T=100)', '5g': '5G FDD (T=200)'}


def load_metrics(task_name):
    """Load terminal metrics CSV for one task. Returns list of dicts."""
    csv_path = os.path.join(TABLES_DIR, f'terminal_metrics_{task_name}.csv')
    if not os.path.exists(csv_path):
        print(f'WARNING: {csv_path} not found — skipping {task_name}')
        return None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Loaded {len(rows)} rows for {task_name}')
    return rows


def plot_terminal_distributions(all_data):
    """Plot x0 vs xT distribution metrics for all tasks."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    metrics = [
        ('pearson_corr', 'Pearson Correlation (x₀, x_T)',
         'Closer to 1 = more signal retained at terminal'),
        ('nmse', 'NMSE (x₀, x_T)',
         'Normalized mean squared error between pristine and terminal'),
    ]

    for col, task_name in enumerate(TASKS):
        data = all_data.get(task_name)
        if data is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'NO DATA',
                                    transform=axes[row, col].transAxes,
                                    ha='center', va='center', fontsize=14)
                axes[row, col].set_title(TASK_LABELS[task_name])
            continue

        for row, (key, title, caption) in enumerate(metrics):
            ax = axes[row, col]
            vals = [float(r[key]) for r in data]
            n = len(vals)

            ax.hist(vals, bins=min(25, max(5, n // 3)), edgecolor='black',
                    alpha=0.7, color='steelblue')
            mean_val = np.mean(vals)
            median_val = np.median(vals)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5,
                       label=f'Mean: {mean_val:.4f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5,
                       label=f'Median: {median_val:.4f}')
            ax.set_title(f'{TASK_LABELS[task_name]}: {title}', fontsize=11)
            ax.set_xlabel(key)
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
            ax.text(0.98, 0.02, f'N={n}', transform=ax.transAxes,
                    ha='right', va='bottom', fontsize=8, color='gray')

    plt.suptitle('Terminal State Consistency Analysis\n'
                 'x₀ (pristine dataset data) vs x_T (official degrade_fn output)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(FIGURES_DIR, 'fig_terminal_distributions.png')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def print_summary_table(all_data):
    """Print a text summary table of terminal metrics."""
    print('\n' + '=' * 70)
    print('TERMINAL CONSISTENCY SUMMARY')
    print('=' * 70)
    header = f"{'Task':<12} {'N':<6} {'Pearson r':<20} {'NMSE':<20} {'Energy Ratio':<20}"
    print(header)
    print('-' * 70)

    for task_name in TASKS:
        data = all_data.get(task_name)
        if data is None:
            continue
        pearson = [float(r['pearson_corr']) for r in data]
        nmse = [float(r['nmse']) for r in data]
        eratio = [float(r['energy_ratio']) for r in data]
        print(f'{TASK_LABELS[task_name]:<12} '
              f'{len(data):<6} '
              f'{np.mean(pearson):.4f} ± {np.std(pearson):.4f}     '
              f'{np.mean(nmse):.4f} ± {np.std(nmse):.4f}     '
              f'{np.mean(eratio):.4f} ± {np.std(eratio):.4f}')

    print('=' * 70)
    print('Interpretation: Lower Pearson r → less signal retained at x_T.')
    print('Higher NMSE → terminal state diverges more from pristine.')
    print('Energy ratio > 1 → noise injected adds energy.')


def main():
    print('Terminal Consistency Analysis — Plot Generator')
    print('=' * 50)

    all_data = {}
    for task_name in TASKS:
        data = load_metrics(task_name)
        if data is not None:
            all_data[task_name] = data

    if not all_data:
        print('ERROR: No data loaded. Check that CSV files exist in', TABLES_DIR)
        return

    plot_terminal_distributions(all_data)
    print_summary_table(all_data)
    print('\nDone.')


if __name__ == '__main__':
    main()
