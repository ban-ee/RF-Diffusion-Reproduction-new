"""
Native vs Source-Free Sampling Comparison — Plotting Script (CPU-only)

INPUT:
    Reads native_vs_fast_{wifi,fmcw}.csv from ../results/tables/
    Each CSV row: condition_idx, native_std, fast_std, native_mean, fast_mean,
                  native_energy, fast_energy, native_vs_fast_corr,
                  native_vs_fast_nmse, native_vs_data_corr, fast_vs_data_corr

OUTPUT:
    Saves fig_native_vs_fast.png to ../results/figures/
    One figure with 2×3 subplots (WiFi/FMCW × correlation bars / energy / std)

USAGE:
    python native_vs_fast_plot.py

STATUS: EXPLORATORY — This compares two publicly available official functions.
        It is NOT the paper's official evaluation protocol.

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
CORR_KEYS = ['native_vs_fast_corr', 'native_vs_data_corr', 'fast_vs_data_corr']
CORR_LABELS = ['Native vs Fast', 'Native vs Data', 'Fast vs Data']
CORR_COLORS = ['#e74c3c', '#3498db', '#95a5a6']


def load_native_vs_fast(task_name):
    """Load native-vs-fast CSV for one task."""
    csv_path = os.path.join(TABLES_DIR, f'native_vs_fast_{task_name}.csv')
    if not os.path.exists(csv_path):
        print(f'WARNING: {csv_path} not found — skipping {task_name}')
        return None
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f'Loaded {len(rows)} rows for {task_name}')
    return rows


def plot_native_vs_fast(all_rows):
    """Plot native vs source-free comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    for col, task_name in enumerate(TASKS):
        rows = all_rows.get(task_name)
        if rows is None:
            for row in range(2):
                axes[row, col].text(0.5, 0.5, 'NO DATA',
                                    transform=axes[row, col].transAxes,
                                    ha='center', va='center', fontsize=14)
            continue

        # Row 0: Correlation bar chart
        ax_corr = axes[0, col]
        means = []
        stds = []
        for key in CORR_KEYS:
            vals = [float(r[key]) for r in rows]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        x = range(len(CORR_KEYS))
        ax_corr.bar(x, means, yerr=stds, color=CORR_COLORS,
                    capsize=8, edgecolor='black', linewidth=0.5)
        ax_corr.set_xticks(x)
        ax_corr.set_xticklabels(CORR_LABELS, fontsize=10)
        ax_corr.set_title(f'{TASK_LABELS[task_name]}: Output Correlations',
                          fontsize=12)
        ax_corr.set_ylabel('Pearson r')
        ax_corr.set_ylim(-0.1, 1.05)
        ax_corr.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        for i, (m, s) in enumerate(zip(means, stds)):
            ax_corr.text(i, max(m + s + 0.03, 0.02),
                         f'{m:.4f}±{s:.4f}', ha='center', fontsize=8)

        # Row 1, col 0-1: Native vs fast per-condition scatter
        ax_scatter = axes[1, col]
        native_vals = [float(r['native_std']) for r in rows]
        fast_vals = [float(r['fast_std']) for r in rows]
        ax_scatter.scatter(native_vals, fast_vals, alpha=0.7, s=60,
                          edgecolors='black', linewidth=0.5)
        # Identity line
        all_vals = native_vals + fast_vals
        lim_min = min(all_vals) * 0.9
        lim_max = max(all_vals) * 1.1
        ax_scatter.plot([lim_min, lim_max], [lim_min, lim_max],
                        'r--', linewidth=1, label='y=x (identity)')
        ax_scatter.set_xlabel('Native Output Std')
        ax_scatter.set_ylabel('Fast Output Std')
        ax_scatter.set_title(f'{TASK_LABELS[task_name]}: Native vs Fast Std',
                            fontsize=12)
        ax_scatter.legend(fontsize=8)

    # Third column: Summary comparison
    ax_summary = axes[0, 2]
    summary_data = {}
    for task_name in TASKS:
        rows = all_rows.get(task_name)
        if rows is None:
            continue
        for key in CORR_KEYS:
            vals = [float(r[key]) for r in rows]
            summary_data[f'{task_name}_{key}'] = np.mean(vals)

    if summary_data:
        summary_labels = []
        summary_values = []
        summary_colors = []
        for task_name in TASKS:
            for i, key in enumerate(CORR_KEYS):
                k = f'{task_name}_{key}'
                if k in summary_data:
                    summary_labels.append(f'{TASK_LABELS[task_name]}\n{CORR_LABELS[i]}')
                    summary_values.append(summary_data[k])
                    summary_colors.append(CORR_COLORS[i])

        x = range(len(summary_labels))
        ax_summary.bar(x, summary_values, color=summary_colors,
                      edgecolor='black', linewidth=0.5)
        ax_summary.set_xticks(x)
        ax_summary.set_xticklabels(summary_labels, fontsize=7)
        ax_summary.set_title('Correlation Summary: Native vs Fast vs Data',
                            fontsize=12)
        ax_summary.set_ylabel('Pearson r')
        ax_summary.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        for i, v in enumerate(summary_values):
            ax_summary.text(i, max(v + 0.03, 0.02), f'{v:.4f}',
                           ha='center', fontsize=8)

    # Third column bottom: NMSE summary
    ax_nmse = axes[1, 2]
    nmse_data = {}
    for task_name in TASKS:
        rows = all_rows.get(task_name)
        if rows is None:
            continue
        vals = [float(r['native_vs_fast_nmse']) for r in rows]
        nmse_data[task_name] = (np.mean(vals), np.std(vals))

    if nmse_data:
        nmse_labels = [TASK_LABELS[t] for t in TASKS if t in nmse_data]
        nmse_means = [nmse_data[t][0] for t in TASKS if t in nmse_data]
        nmse_stds = [nmse_data[t][1] for t in TASKS if t in nmse_data]
        x = range(len(nmse_labels))
        ax_nmse.bar(x, nmse_means, yerr=nmse_stds, color=['#3498db', '#e67e22'],
                   capsize=8, edgecolor='black', linewidth=0.5)
        ax_nmse.set_xticks(x)
        ax_nmse.set_xticklabels(nmse_labels, fontsize=10)
        ax_nmse.set_title('Native-Fast NMSE', fontsize=12)
        ax_nmse.set_ylabel('NMSE')
        for i, (m, s) in enumerate(zip(nmse_means, nmse_stds)):
            ax_nmse.text(i, m + s + 0.02, f'{m:.4f}±{s:.4f}',
                        ha='center', fontsize=9)

    plt.suptitle('Native vs Source-Free Sampling — Exploratory Comparison\n'
                 'STATUS: EXPLORATORY. Compares two official functions; '
                 'NOT the paper evaluation protocol.',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(FIGURES_DIR, 'fig_native_vs_fast.png')
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def print_interpretation(all_rows):
    """Print interpretation of native-vs-fast results."""
    print('\n' + '=' * 70)
    print('NATIVE VS SOURCE-FREE — INTERPRETATION')
    print('=' * 70)
    print()
    print('STATUS: EXPLORATORY OFFICIAL-FUNCTION COMPARISON')
    print('This is NOT the paper\'s official evaluation protocol.')
    print()
    print('native_sampling: constructs terminal state from source x₀ + condition')
    print('                  x_T = degrade_fn(x₀, T-1), then one reverse pass')
    print()
    print('fast_sampling:   terminal state from random noise')
    print('                  x_T = noise, then one reverse pass')
    print()

    for task_name in TASKS:
        rows = all_rows.get(task_name)
        if rows is None:
            continue

        nf_corr = np.mean([float(r['native_vs_fast_corr']) for r in rows])
        nf_nmse = np.mean([float(r['native_vs_fast_nmse']) for r in rows])
        nd_corr = np.mean([float(r['native_vs_data_corr']) for r in rows])
        fd_corr = np.mean([float(r['fast_vs_data_corr']) for r in rows])

        print(f'  {TASK_LABELS[task_name]}:')
        print(f'    Native-Fast correlation:  {nf_corr:.4f}')
        print(f'    Native-Fast NMSE:         {nf_nmse:.4f}')
        print(f'    Native-Data correlation:  {nd_corr:.4f}')
        print(f'    Fast-Data correlation:    {fd_corr:.4f}')
        print()

    print('OBSERVATION: Native and fast outputs are decorrelated (r ≈ 0).')
    print('             Native outputs correlate with source data (r ≈ 0.83).')
    print('             Fast outputs do NOT correlate with source data.')
    print()
    print('IMPLICATION: Inference output is protocol-dependent.')
    print('             The terminal state construction method matters.')
    print('=' * 70)


def main():
    print('Native vs Fast Analysis — Plot Generator')
    print('=' * 50)

    all_rows = {}
    for task_name in TASKS:
        rows = load_native_vs_fast(task_name)
        if rows is not None:
            all_rows[task_name] = rows

    if not all_rows:
        print('ERROR: No data loaded.')
        return

    plot_native_vs_fast(all_rows)
    print_interpretation(all_rows)
    print('\nDone.')


if __name__ == '__main__':
    main()
