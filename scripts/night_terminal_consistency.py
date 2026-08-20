"""PHASE 2: Multi-sample terminal consistency analysis.
Uses official SignalDiffusion.degrade_fn on ALL real evaluation samples.
No source modification. No single-sample shortcuts."""
import sys, os, json, csv, torch, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(REPO_ROOT + '/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params, AttrDict
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import from_path_inference, _nested_map

NIGHT = REPO_ROOT
device = torch.device('cuda')
SEED = 42

def compute_terminal_metrics(x0, xT):
    """Compute x0 vs xT metrics. x0 and xT are numpy arrays."""
    x0_flat = x0.ravel()
    xT_flat = xT.ravel()

    # Pearson correlation
    corr = np.corrcoef(x0_flat, xT_flat)[0, 1]

    # NMSE
    mse = np.mean((x0_flat - xT_flat) ** 2)
    norm = np.mean(x0_flat ** 2)
    nmse = mse / norm if norm > 0 else float('inf')

    # Energy ratio
    energy_x0 = np.sum(x0_flat ** 2)
    energy_xT = np.sum(xT_flat ** 2)
    energy_ratio = energy_xT / energy_x0 if energy_x0 > 0 else float('inf')

    # Signal projection (normalized inner product)
    dot = np.dot(x0_flat, xT_flat)
    proj = dot / (np.linalg.norm(x0_flat) * np.linalg.norm(xT_flat)) if (np.linalg.norm(x0_flat) * np.linalg.norm(xT_flat)) > 0 else 0

    return {
        'pearson_corr': float(corr),
        'nmse': float(nmse),
        'energy_ratio': float(energy_ratio),
        'signal_projection': float(proj),
        'energy_x0': float(energy_x0),
        'energy_xT': float(energy_xT)
    }

def run_terminal_analysis(task_idx, task_name, subset_size=None):
    """Run terminal state analysis for one task."""
    torch.manual_seed(SEED)

    params = all_params[task_idx]
    diffusion = SignalDiffusion(params)

    T = params.max_step
    alpha_bar_T = float(diffusion.alpha_bar[-1].item())
    sqrt_alpha_bar_T = float(np.sqrt(alpha_bar_T))

    info_w = diffusion.info_weights[-1].cpu().numpy()
    noise_w = diffusion.noise_weights[-1].cpu().numpy()

    config = {
        'task': task_name,
        'T': T,
        'alpha_bar_T': alpha_bar_T,
        'sqrt_alpha_bar_T': sqrt_alpha_bar_T,
        'blur_schedule': [float(b) for b in params.blur_schedule],
        'noise_schedule': [float(n) for n in params.noise_schedule],
        'info_weights_shape': list(info_w.shape),
        'info_weights_stats': {'mean': float(np.mean(info_w)), 'std': float(np.std(info_w)),
                               'min': float(np.min(info_w)), 'max': float(np.max(info_w))},
        'noise_weights_shape': list(noise_w.shape),
        'noise_weights_stats': {'mean': float(np.mean(noise_w)), 'std': float(np.std(noise_w)),
                                'min': float(np.min(noise_w)), 'max': float(np.max(noise_w))},
    }

    # Load dataset
    dataset = from_path_inference(params)
    all_metrics = []

    sample_idx = 0
    for features in dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        data = features['data']  # This is pristine x0 from dataset

        for b in range(data.shape[0]):
            if subset_size and sample_idx >= subset_size:
                break

            x0 = data[b:b+1]  # Keep batch dim for degrade_fn

            # Apply official degrade_fn at terminal step T-1
            # IMPORTANT: x0 is the pristine data from dataset, NOT degrade_fn(data, 0)
            xT = diffusion.degrade_fn(x0, T - 1, task_id=task_idx)

            # Convert to numpy
            x0_np = x0.cpu().numpy()
            xT_np = xT.cpu().numpy()

            metrics = compute_terminal_metrics(x0_np, xT_np)
            metrics['sample_idx'] = sample_idx
            all_metrics.append(metrics)

            sample_idx += 1

            if sample_idx % 10 == 0:
                print(f'  [{task_name}] Processed {sample_idx} samples...')

        if subset_size and sample_idx >= subset_size:
            break

    N = len(all_metrics)
    print(f'  [{task_name}] Total: {N} samples')

    # Compute statistics
    stats = {}
    for key in ['pearson_corr', 'nmse', 'energy_ratio', 'signal_projection']:
        vals = [m[key] for m in all_metrics]
        stats[key] = {
            'N': N, 'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'median': float(np.median(vals)),
            'p25': float(np.percentile(vals, 25)), 'p75': float(np.percentile(vals, 75)),
            'min': float(np.min(vals)), 'max': float(np.max(vals))
        }

    return config, all_metrics, stats

def save_results(task_name, config, all_metrics, stats):
    """Save CSV, JSON, and plots."""
    base = f'{NIGHT}'

    # CSV
    csv_path = f'{base}/experiments/night_run/terminal_metrics_{task_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f'  CSV: {csv_path}')

    # JSON config + stats
    json_path = f'{base}/artifacts/night_run/terminal_config_{task_name}.json'
    with open(json_path, 'w') as f:
        json.dump({'config': config, 'statistics': stats}, f, indent=2)
    print(f'  JSON: {json_path}')

    # Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics_plot = [
        ('pearson_corr', 'Pearson Correlation (x0 vs xT)', 'corr_distribution.png'),
        ('nmse', 'NMSE (x0 vs xT)', 'nmse_distribution.png'),
        ('energy_ratio', 'Energy Ratio ||xT||^2 / ||x0||^2', 'energy_distribution.png'),
        ('signal_projection', 'Signal Projection', 'proj_distribution.png'),
    ]

    for ax, (key, title, fname) in zip(axes.flat, metrics_plot):
        vals = [m[key] for m in all_metrics]
        ax.hist(vals, bins=min(30, len(vals)//2 + 1), edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(vals), color='red', linestyle='--', label=f'Mean: {np.mean(vals):.4f}')
        ax.axvline(np.median(vals), color='green', linestyle='--', label=f'Median: {np.median(vals):.4f}')
        ax.set_title(f'{task_name}: {title}')
        ax.set_xlabel(key)
        ax.set_ylabel('Count')
        ax.legend()

    plt.tight_layout()
    fig_path = f'{base}/artifacts/night_run/terminal_{task_name}_distributions.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Plot: {fig_path}')

    return stats

# ========== MAIN ==========
print('=' * 60)
print('PHASE 2: MULTI-SAMPLE TERMINAL CONSISTENCY')
print('=' * 60)

all_results = {}

# WiFi: 41 samples
print('\n--- WiFi Terminal Analysis (41 samples) ---')
torch.cuda.empty_cache()
config_w, metrics_w, stats_w = run_terminal_analysis(0, 'wifi')
wifi_stats = save_results('wifi', config_w, metrics_w, stats_w)
all_results['wifi'] = {'config': config_w, 'stats': stats_w}

# FMCW: 31 samples
print('\n--- FMCW Terminal Analysis (31 samples) ---')
torch.cuda.empty_cache()
config_f, metrics_f, stats_f = run_terminal_analysis(1, 'fmcw')
fmcw_stats = save_results('fmcw', config_f, metrics_f, stats_f)
all_results['fmcw'] = {'config': config_f, 'stats': stats_f}

# 5G: 200 samples (or subset=50 if slow)
print('\n--- 5G Terminal Analysis ---')
torch.cuda.empty_cache()
# Try full 200 first; fall back to 50 if too slow
try:
    config_m, metrics_m, stats_m = run_terminal_analysis(2, '5g')
except Exception as e:
    print(f'  Full 200 failed ({e}), falling back to 50 samples with seed=42')
    torch.cuda.empty_cache()
    config_m, metrics_m, stats_m = run_terminal_analysis(2, '5g', subset_size=50)
mimo_stats = save_results('5g', config_m, metrics_m, stats_m)
all_results['5g'] = {'config': config_m, 'stats': stats_m}

# Print summary
print('\n' + '=' * 60)
print('TERMINAL CONSISTENCY SUMMARY')
print('=' * 60)
for name in ['wifi', 'fmcw', '5g']:
    r = all_results[name]
    s = r['stats']
    c = r['config']
    print(f'\n{name.upper()}:')
    print(f'  T={c["T"]}, alpha_bar_T={c["alpha_bar_T"]:.6f}, sqrt(alpha_bar_T)={c["sqrt_alpha_bar_T"]:.6f}')
    for key in ['pearson_corr', 'nmse', 'energy_ratio', 'signal_projection']:
        print(f'  {key}: mean={s[key]["mean"]:.4f} +/- {s[key]["std"]:.4f}, median={s[key]["median"]:.4f}, [{s[key]["p25"]:.4f}, {s[key]["p75"]:.4f}]')

print('\n=== TERMINAL_CONSISTENCY_COMPLETE ===')
