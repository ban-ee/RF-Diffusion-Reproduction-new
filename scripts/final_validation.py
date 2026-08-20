"""
FINAL VALIDATION — RF-Diffusion WiFi Source Dependence & TFD Ablation

Produces:
  A. Formal source dependence (3 metrics, all 41 samples, 3 seeds, both ckpts)
  B. Native vs Fast formal evaluation (distribution-level stats)
  C. True TFD 3-way ablation (Noise Only / Blur Only / Noise+Blur)
  D. alpha_bar vs empirical correlation
  E. All figures (boxplot, comparison, corr_vs_t, nmse_vs_t, spectral_vs_t)

Usage:
  python final_validation.py

Outputs go to experiments/final_validation/
"""
import os, sys, json, csv, time, warnings
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

os.chdir(REPO_ROOT + '/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import WiFiDataset, Collator

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = REPO_ROOT
RESULTS_DIR = f'{BASE_DIR}/experiments/final_validation'
ARTIFACTS_DIR = f'{BASE_DIR}/artifacts/final_validation'
REPORTS_DIR = f'{BASE_DIR}/reports/final_validation'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
N_CROSS_SOURCES = 5  # Number of random alternative sources per condition
TIMESTEPS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
N_ABLATION_SAMPLES = 41  # Use all samples for ablation

device = torch.device('cuda')
print(f'Device: {device}')
print(f'Results: {RESULTS_DIR}')
print(f'Artifacts: {ARTIFACTS_DIR}')

# ============================================================
# MODEL & DATA LOADING
# ============================================================
def load_model(ckpt_path):
    params = all_params[0]  # WiFi
    model = tfdiff_WiFi(params).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    diffusion = SignalDiffusion(params)
    return model, diffusion, params

def load_all_samples():
    params = all_params[0]
    params.data_dir = ['./wifi/cond']
    collator = Collator(params)
    dataset = WiFiDataset(params.data_dir)
    samples = []
    for i in range(len(dataset)):
        raw = dataset[i]
        batch = collator.collate([raw])
        samples.append({
            'data': batch['data'],        # [1, 512, 90, 2] float
            'cond': batch['cond'],        # [1, 6, 2] float
            'idx': i,
        })
    return samples

def ravel(tensor):
    """Flatten tensor to 1D numpy for correlation computation."""
    return tensor.detach().cpu().numpy().ravel()

def compute_metrics(pred, target):
    """Compute correlation, NMSE, energy ratio, spectral distance."""
    p = pred.detach().cpu().numpy().ravel()
    t = target.detach().cpu().numpy().ravel()

    # Pearson correlation
    corr = float(np.corrcoef(p, t)[0, 1]) if len(p) > 1 else 0.0

    # NMSE
    mse = np.mean((p - t) ** 2)
    t_power = np.mean(t ** 2)
    nmse = float(mse / (t_power + 1e-10))

    # Energy ratio
    energy_p = float(np.sum(p ** 2))
    energy_t = float(np.sum(t ** 2))
    energy_ratio = float(energy_p / (energy_t + 1e-10))

    # Spectral distance: normalized L2 between magnitude spectra
    # Reshape to [512*90*2] -> extract complex -> FFT along first dim
    pred_complex = pred.detach().cpu().numpy()
    target_complex = target.detach().cpu().numpy()
    # Compute magnitude spectra (simple: just use absolute value of complex representation)
    p_mag = np.sqrt(pred_complex[..., 0]**2 + pred_complex[..., 1]**2).ravel()
    t_mag = np.sqrt(target_complex[..., 0]**2 + target_complex[..., 1]**2).ravel()
    spectral_dist = float(np.linalg.norm(p_mag - t_mag) / (np.linalg.norm(t_mag) + 1e-10))

    return {
        'pearson_corr': corr,
        'nmse': nmse,
        'energy_ratio': energy_ratio,
        'spectral_distance': spectral_dist,
    }

def bootstrap_ci(values, n_bootstrap=2000, alpha=0.05):
    """Bootstrap 95% CI for mean."""
    values = np.array(values)
    n = len(values)
    means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        means.append(np.mean(values[idx]))
    means = np.array(means)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def stats_summary(values, name, do_bootstrap=True):
    """Compute comprehensive statistics for a list of values."""
    vals = np.array(values)
    s = {
        'N': int(len(vals)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'median': float(np.median(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals)),
    }
    if do_bootstrap and len(vals) >= 10:
        try:
            lo, hi = bootstrap_ci(vals)
            s['ci_95_low'] = lo
            s['ci_95_high'] = hi
        except:
            s['ci_95_low'] = None
            s['ci_95_high'] = None
    return s


# ============================================================
# A. FORMAL SOURCE DEPENDENCE (3 metrics, all 41 samples, 3 seeds)
# ============================================================
def run_source_dependence(model, diffusion, params, samples, tag):
    """Formal source dependence with 3 well-defined metrics.

    For each condition c_i and source x_j:
        y(i,j) = native_sampling(model, x_j, c_i)

    Metric A: Corr(y(i,j), x_j) — output-to-own-source
    Metric B: Corr(y(i,j), x_k) for k != j — output-to-cross-source
    Metric C: Corr(y(i,j), y(i,k)) for j != k — cross-terminal-output similarity
    """
    print(f'\n{"="*60}')
    print(f'SOURCE DEPENDENCE: {tag}')
    print(f'{"="*60}')

    N = len(samples)
    all_results = []

    for seed_idx, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f'\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...')

        for cond_idx in range(N):
            cond_i = samples[cond_idx]['cond'].to(device)

            # Determine source indices for this condition
            # Source j=cond_idx (own source) + N_CROSS_SOURCES random others
            other_indices = [j for j in range(N) if j != cond_idx]
            n_cross = min(N_CROSS_SOURCES, len(other_indices))
            cross_src_indices = list(np.random.choice(other_indices, size=n_cross, replace=False))
            all_src_indices = [cond_idx] + cross_src_indices

            # Generate outputs for all sources
            outputs = {}
            for src_idx in all_src_indices:
                data_j = samples[src_idx]['data'].to(device)
                with torch.no_grad():
                    y = diffusion.native_sampling(model, data_j, cond_i, device)
                outputs[src_idx] = y.cpu()

            # Metric A: output-to-own-source
            own_src = cond_idx
            y_own = outputs[own_src]
            data_own = samples[own_src]['data']
            m_a = compute_metrics(y_own, data_own)
            all_results.append({
                'seed': seed,
                'condition_idx': cond_idx,
                'source_idx': own_src,
                'metric': 'A_output_to_own_source',
                **m_a,
            })

            # Metric B: output-to-cross-source
            for cross_src in cross_src_indices:
                y_cross = outputs[cross_src]
                # Correlate y(i, cross_src) with ITS OWN source
                data_cross = samples[cross_src]['data']
                m_b = compute_metrics(y_cross, data_cross)
                all_results.append({
                    'seed': seed,
                    'condition_idx': cond_idx,
                    'source_idx': cross_src,
                    'metric': 'B_output_to_cross_source',
                    **m_b,
                })

                # Also: Correlate y(i, cross_src) with the CONDITION's source (unrelated)
                m_b2 = compute_metrics(y_cross, data_own)
                all_results.append({
                    'seed': seed,
                    'condition_idx': cond_idx,
                    'source_idx': cross_src,
                    'metric': 'B_output_to_unrelated_source',
                    **m_b2,
                })

            # Metric C: cross-terminal-output similarity
            # Corr(y(i, own_src), y(i, cross_src)) for each cross source
            for cross_src in cross_src_indices:
                y_cross = outputs[cross_src]
                m_c = compute_metrics(y_own, y_cross)
                all_results.append({
                    'seed': seed,
                    'condition_idx': cond_idx,
                    'source_idx': cross_src,
                    'metric': 'C_cross_terminal_output_similarity',
                    **m_c,
                })

        torch.cuda.empty_cache()

    # Save raw CSV
    csv_path = f'{RESULTS_DIR}/source_dependence_{tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f'\n  Raw CSV: {csv_path} ({len(all_results)} rows)')

    # Compute summary stats per metric
    summary = {}
    metric_names = sorted(set(r['metric'] for r in all_results))
    for mname in metric_names:
        m_results = [r for r in all_results if r['metric'] == mname]
        corrs = [r['pearson_corr'] for r in m_results]
        nmses = [r['nmse'] for r in m_results]
        summary[mname] = {
            'N': len(m_results),
            'pearson_corr': stats_summary(corrs, 'pearson_corr'),
            'nmse': stats_summary(nmses, 'nmse'),
        }

        print(f'  {mname}:')
        print(f'    N={len(m_results)}, corr={np.mean(corrs):.4f}±{np.std(corrs):.4f}, '
              f'median={np.median(corrs):.4f}, '
              f'95%CI=[{summary[mname]["pearson_corr"].get("ci_95_low","?"):.4f}, '
              f'{summary[mname]["pearson_corr"].get("ci_95_high","?"):.4f}]')

    # Save summary JSON
    json_path = f'{RESULTS_DIR}/source_dependence_{tag}_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  Summary JSON: {json_path}')

    # Compute Source Dependence Score (SDS)
    a_corrs = [r['pearson_corr'] for r in all_results if r['metric'] == 'A_output_to_own_source']
    b_corrs = [r['pearson_corr'] for r in all_results if r['metric'] == 'B_output_to_unrelated_source']
    if a_corrs and b_corrs:
        sds = np.mean(a_corrs) - np.mean(b_corrs)
        print(f'  Source Dependence Score (SDS): {sds:.4f}')
        print(f'    = mean(A: output-to-own-source) - mean(B: output-to-unrelated-source)')
        print(f'    = {np.mean(a_corrs):.4f} - {np.mean(b_corrs):.4f} = {sds:.4f}')

    return all_results, summary


# ============================================================
# B. NATIVE vs FAST FORMAL EVALUATION
# ============================================================
def run_native_vs_fast(model, diffusion, params, samples, tag):
    """Formal Native vs Fast comparison.

    Reports:
      - Per-sample metrics (correlation, NMSE, spectral distance)
      - Distribution-level statistics (mean, std, energy spectrum)
      - Terminal distribution analysis
    """
    print(f'\n{"="*60}')
    print(f'NATIVE vs FAST: {tag}')
    print(f'{"="*60}')

    N = len(samples)
    results = []

    # Collect all outputs for distribution analysis
    native_outputs = []
    fast_outputs = []
    data_samples = []

    torch.manual_seed(42)
    np.random.seed(42)

    for ci in range(N):
        data_i = samples[ci]['data'].to(device)
        cond_i = samples[ci]['cond'].to(device)

        with torch.no_grad():
            native_out = diffusion.native_sampling(model, data_i, cond_i, device)
            fast_out = diffusion.fast_sampling(model, cond_i, device)

        native_cpu = native_out.cpu()
        fast_cpu = fast_out.cpu()
        data_cpu = data_i.cpu()

        native_outputs.append(native_cpu)
        fast_outputs.append(fast_cpu)
        data_samples.append(data_cpu)

        # Per-sample metrics
        nf = compute_metrics(native_out, fast_out)
        nd = compute_metrics(native_out, data_i)
        fd = compute_metrics(fast_out, data_i)

        results.append({
            'condition_idx': ci,
            'native_fast_corr': nf['pearson_corr'],
            'native_fast_nmse': nf['nmse'],
            'native_fast_spectral': nf['spectral_distance'],
            'native_data_corr': nd['pearson_corr'],
            'native_data_nmse': nd['nmse'],
            'fast_data_corr': fd['pearson_corr'],
            'fast_data_nmse': fd['nmse'],
            'native_energy': float(torch.sum(native_cpu ** 2)),
            'fast_energy': float(torch.sum(fast_cpu ** 2)),
            'data_energy': float(torch.sum(data_cpu ** 2)),
        })

    torch.cuda.empty_cache()

    # Save per-sample CSV
    csv_path = f'{RESULTS_DIR}/native_vs_fast_{tag}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Compute per-sample summary
    per_sample_keys = [k for k in results[0].keys() if k != 'condition_idx']
    per_sample_summary = {}
    for k in per_sample_keys:
        vals = [r[k] for r in results]
        per_sample_summary[k] = stats_summary(vals, k)

    # Distribution-level statistics
    # Aggregate all native outputs, fast outputs, and data
    native_all = torch.cat([n.ravel() for n in native_outputs]).numpy()
    fast_all = torch.cat([f.ravel() for f in fast_outputs]).numpy()
    data_all = torch.cat([d.ravel() for d in data_samples]).numpy()

    dist_stats = {
        'native': {
            'mean': float(np.mean(native_all)),
            'std': float(np.std(native_all)),
            'energy': float(np.sum(native_all ** 2)),
            'N_samples': len(native_all),
        },
        'fast': {
            'mean': float(np.mean(fast_all)),
            'std': float(np.std(fast_all)),
            'energy': float(np.sum(fast_all ** 2)),
            'N_samples': len(fast_all),
        },
        'data': {
            'mean': float(np.mean(data_all)),
            'std': float(np.std(data_all)),
            'energy': float(np.sum(data_all ** 2)),
            'N_samples': len(data_all),
        },
    }
    # Cross-distribution correlations
    dist_stats['native_fast_dist_corr'] = float(np.corrcoef(native_all, fast_all)[0, 1])
    dist_stats['native_data_dist_corr'] = float(np.corrcoef(native_all, data_all)[0, 1])
    dist_stats['fast_data_dist_corr'] = float(np.corrcoef(fast_all, data_all)[0, 1])

    # Save distribution stats
    json_path = f'{RESULTS_DIR}/native_vs_fast_{tag}_summary.json'
    full_summary = {
        'per_sample': per_sample_summary,
        'distribution': dist_stats,
    }
    with open(json_path, 'w') as f:
        json.dump(full_summary, f, indent=2)

    # Print key results
    print(f'\n  Per-sample metrics:')
    print(f'    Native-Fast corr: {per_sample_summary["native_fast_corr"]["mean"]:.4f}±{per_sample_summary["native_fast_corr"]["std"]:.4f}')
    print(f'    Native-Data corr: {per_sample_summary["native_data_corr"]["mean"]:.4f}±{per_sample_summary["native_data_corr"]["std"]:.4f}')
    print(f'    Fast-Data corr:   {per_sample_summary["fast_data_corr"]["mean"]:.4f}±{per_sample_summary["fast_data_corr"]["std"]:.4f}')
    print(f'  Distribution-level correlations:')
    print(f'    Native-Fast dist corr: {dist_stats["native_fast_dist_corr"]:.4f}')
    print(f'    Native-Data dist corr: {dist_stats["native_data_dist_corr"]:.4f}')
    print(f'    Fast-Data dist corr:   {dist_stats["fast_data_dist_corr"]:.4f}')
    print(f'  Energy comparison:')
    print(f'    Native/Data energy: {dist_stats["native"]["energy"]/dist_stats["data"]["energy"]:.4f}')
    print(f'    Fast/Data energy:   {dist_stats["fast"]["energy"]/dist_stats["data"]["energy"]:.4f}')

    return results, full_summary


# ============================================================
# C. TRUE TFD 3-WAY ABLATION
# ============================================================
def build_noise_only_weights(params):
    """Build Noise-Only forward weights (standard DDPM).

    x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1-alpha_bar[t]) * randn()
    """
    beta = np.array(params.noise_schedule)
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    return {
        'alpha_bar': torch.tensor(alpha_bar.astype(np.float32)),
        'info_weights': torch.tensor(np.sqrt(alpha_bar).astype(np.float32)),  # [T]
        'noise_weights': torch.tensor(np.sqrt(1 - alpha_bar).astype(np.float32)),  # [T]
    }

def build_blur_only_weights(params):
    """Build Blur-Only forward weights (frequency blur, no additive noise).

    x_t = gaussian_kernel_bar[t] * x_0  (no noise added)
    """
    max_step = params.max_step
    input_dim = params.sample_rate
    var_blur = np.array(params.blur_schedule).astype(np.float32)
    var_blur_bar = np.cumsum(var_blur)

    # Build gaussian_kernel_bar (same as official get_kernel)
    gaussian_kernel_bar_list = []
    for t in range(max_step):
        var_k = max(input_dim / max(var_blur_bar[t], 1e-30), 1e-10)
        samples = np.arange(0, input_dim)
        kernel = np.exp(-((samples - input_dim // 2) ** 2) / (2 * var_k))
        kernel = kernel / np.sqrt(2 * np.pi * var_k)
        kernel = input_dim * kernel / (np.sum(kernel) + 1e-10)
        gaussian_kernel_bar_list.append(kernel)

    gaussian_kernel_bar = torch.tensor(np.stack(gaussian_kernel_bar_list, axis=0).astype(np.float32))

    return {
        'gaussian_kernel_bar': gaussian_kernel_bar,  # [T, N]
        'info_weights': gaussian_kernel_bar,  # [T, N] — just the blur kernel
        'noise_weights': torch.zeros(max_step, input_dim),  # No noise
    }

def forward_noise_only(data, t, noise_weights, info_weights, seed=11):
    """Noise-only forward: x_t = sqrt(alpha_bar[t]) * x_0 + sqrt(1-alpha_bar[t]) * ε

    info_weights: [T] — sqrt(alpha_bar[t]), scalar per timestep
    noise_weights: [T] — sqrt(1-alpha_bar[t]), scalar per timestep
    """
    B = data.shape[0]

    # Weights are scalars per timestep; broadcast to [B, 1, 1, 1]
    iw = info_weights[t].view(B, 1, 1, 1).to(data.device)
    nw = noise_weights[t].view(B, 1, 1, 1).to(data.device)

    torch.manual_seed(seed)
    noise = nw * torch.randn_like(data)
    x_t = iw * data + noise
    return x_t

def forward_blur_only(data, t, gaussian_kernel_bar, seed=11):
    """Blur-only forward: x_t = gaussian_kernel_bar[t] * x_0 (no noise)

    gaussian_kernel_bar: [T, N] — frequency-domain kernel, varies per timestep and time bin
    """
    B = data.shape[0]
    N_dim = data.shape[1]
    # gaussian_kernel_bar[t] returns [B, N] for batch tensor t
    iw = gaussian_kernel_bar[t].view(B, N_dim, 1, 1).to(data.device)
    x_t = iw * data
    return x_t

def run_tfd_ablation(diffusion, params, samples, tag='official'):
    """True TFD 3-way ablation: Noise Only / Blur Only / Noise+Blur.

    For each sample × timestep × protocol, compute:
      - Pearson(x0, xt)
      - NMSE(x0, xt)
      - Signal power ratio
      - Spectral distance
    """
    print(f'\n{"="*60}')
    print(f'TFD 3-WAY ABLATION')
    print(f'{"="*60}')

    # Report blur schedule statistics
    blur_sched = np.array(params.blur_schedule)
    print(f'\n  WiFi blur_schedule:')
    print(f'    min={blur_sched.min():.2e}, max={blur_sched.max():.2e}, mean={blur_sched.mean():.2e}')
    print(f'    var_blur_bar at T-1: {blur_sched.sum():.2e}')
    if blur_sched.max() < 1e-8:
        print(f'    VERDICT: Blur variance is NEGLIGIBLE (~1e-10 per step)')

    # Build ablation variants
    noise_only_w = build_noise_only_weights(params)
    blur_only_w = build_blur_only_weights(params)

    # Use all samples
    N = min(N_ABLATION_SAMPLES, len(samples))
    test_indices = list(range(N))

    results = []
    t0 = time.time()

    for sample_idx in test_indices:
        data_0 = samples[sample_idx]['data']  # [1, 512, 90, 2] float

        for t_step in TIMESTEPS:
            t_tensor = torch.tensor([t_step], dtype=torch.int64)

            # --- Noise + Blur (official) ---
            x_t_nb = diffusion.degrade_fn(data_0, t_tensor, 0)
            m_nb = compute_metrics(x_t_nb, data_0)

            # --- Noise Only ---
            x_t_n = forward_noise_only(data_0, t_tensor,
                                       noise_only_w['noise_weights'],
                                       noise_only_w['info_weights'])
            m_n = compute_metrics(x_t_n, data_0)

            # --- Blur Only ---
            x_t_b = forward_blur_only(data_0, t_tensor,
                                      blur_only_w['gaussian_kernel_bar'])
            m_b = compute_metrics(x_t_b, data_0)

            # Collect results
            for protocol, metrics in [('Noise+Blur', m_nb), ('Noise_Only', m_n), ('Blur_Only', m_b)]:
                results.append({
                    'sample_idx': int(sample_idx),
                    'timestep': t_step,
                    'protocol': protocol,
                    'alpha_bar': float(diffusion.alpha_bar[t_step].item()),
                    **metrics,
                })

    elapsed = time.time() - t0
    print(f'  Computed {len(results)} rows in {elapsed:.1f}s')

    # Save raw CSV
    csv_path = f'{RESULTS_DIR}/tfd_ablation.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f'  Raw CSV: {csv_path}')

    # Aggregate by timestep × protocol
    agg_data = []
    for protocol in ['Noise_Only', 'Blur_Only', 'Noise+Blur']:
        for ts in TIMESTEPS:
            ts_results = [r for r in results if r['protocol'] == protocol and r['timestep'] == ts]
            corrs = [r['pearson_corr'] for r in ts_results]
            nmses = [r['nmse'] for r in ts_results]
            specs = [r['spectral_distance'] for r in ts_results]
            energies = [r['energy_ratio'] for r in ts_results]

            agg_data.append({
                'protocol': protocol,
                'timestep': ts,
                'N': len(ts_results),
                'corr_mean': float(np.mean(corrs)),
                'corr_std': float(np.std(corrs)),
                'nmse_mean': float(np.mean(nmses)),
                'nmse_std': float(np.std(nmses)),
                'spectral_mean': float(np.mean(specs)),
                'spectral_std': float(np.std(specs)),
                'energy_ratio_mean': float(np.mean(energies)),
                'energy_ratio_std': float(np.std(energies)),
            })

    # Save aggregate CSV
    agg_path = f'{RESULTS_DIR}/tfd_ablation_aggregate.csv'
    with open(agg_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=agg_data[0].keys())
        writer.writeheader()
        writer.writerows(agg_data)

    # Print final timestep comparison
    print(f'\n  At t=T-1=99:')
    for protocol in ['Noise_Only', 'Blur_Only', 'Noise+Blur']:
        ts99 = [r for r in agg_data if r['protocol'] == protocol and r['timestep'] == 99][0]
        print(f'    {protocol}: corr={ts99["corr_mean"]:.4f}±{ts99["corr_std"]:.4f}, '
              f'NMSE={ts99["nmse_mean"]:.4f}, spectral_dist={ts99["spectral_mean"]:.4f}')

    return results, agg_data


# ============================================================
# FIGURE GENERATION
# ============================================================
def generate_figures(src_results_official, src_results_self,
                     nf_results_official, nf_results_self,
                     ablation_agg, diffusion, params):
    """Generate all required figures."""
    print(f'\n{"="*60}')
    print(f'GENERATING FIGURES')
    print(f'{"="*60}')

    # ---- Source Dependence Boxplot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (tag, results) in enumerate([('Official', src_results_official),
                                              ('Self-trained', src_results_self)]):
        if results is None:
            continue
        ax = axes[ax_idx]

        metric_data = {}
        for r in results:
            m = r['metric']
            if m not in metric_data:
                metric_data[m] = []
            metric_data[m].append(r['pearson_corr'])

        labels = []
        data = []
        colors = []
        for mname in ['A_output_to_own_source', 'B_output_to_cross_source',
                       'B_output_to_unrelated_source', 'C_cross_terminal_output_similarity']:
            if mname in metric_data:
                short_name = mname.replace('A_', '').replace('B_', '').replace('C_', '')
                labels.append(short_name)
                data.append(metric_data[mname])
                if 'own_source' in mname:
                    colors.append('#2196F3')
                elif 'unrelated' in mname:
                    colors.append('#FF5722')
                elif 'cross_source' in mname:
                    colors.append('#FF9800')
                else:
                    colors.append('#4CAF50')

        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(f'{tag} Checkpoint\nSource Dependence Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pearson Correlation')
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add mean values
        for i, d in enumerate(data):
            ax.annotate(f'{np.mean(d):.3f}', xy=(i+1, np.mean(d)),
                       fontsize=7, ha='center', va='bottom', color='darkred')

    plt.tight_layout()
    fig_path = f'{ARTIFACTS_DIR}/source_dependence_boxplot.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Boxplot: {fig_path}')

    # ---- Checkpoint Comparison ----
    if src_results_official is not None and src_results_self is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax_idx, mname in enumerate(['A_output_to_own_source',
                                         'B_output_to_unrelated_source',
                                         'C_cross_terminal_output_similarity']):
            ax = axes[ax_idx]

            off_vals = [r['pearson_corr'] for r in src_results_official if r['metric'] == mname]
            self_vals = [r['pearson_corr'] for r in src_results_self if r['metric'] == mname]

            positions = [1, 2]
            bp = ax.boxplot([off_vals, self_vals], labels=['Official', 'Self-trained'],
                          patch_artist=True)
            for patch, color in zip(bp['boxes'], ['#2196F3', '#FF9800']):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

            short_name = mname.split('_', 1)[1] if '_' in mname else mname
            ax.set_title(short_name, fontsize=10, fontweight='bold')
            ax.set_ylabel('Pearson Correlation')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle('Source Dependence: Official vs Self-Trained Checkpoint',
                    fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig_path = f'{ARTIFACTS_DIR}/source_dependence_checkpoint_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Checkpoint comparison: {fig_path}')

    # ---- TFD Ablation: Correlation vs Timestep ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    protocols = ['Noise_Only', 'Blur_Only', 'Noise+Blur']
    colors_p = {'Noise_Only': '#E91E63', 'Blur_Only': '#2196F3', 'Noise+Blur': '#4CAF50'}
    line_styles = {'Noise_Only': '--', 'Blur_Only': ':', 'Noise+Blur': '-'}

    # corr vs t
    ax = axes[0]
    for p in protocols:
        p_data = [r for r in ablation_agg if r['protocol'] == p]
        ts = [r['timestep'] for r in p_data]
        corr_means = [r['corr_mean'] for r in p_data]
        corr_stds = [r['corr_std'] for r in p_data]
        ax.plot(ts, corr_means, line_styles[p], color=colors_p[p], linewidth=2, label=p)
        ax.fill_between(ts,
                        [m-s for m,s in zip(corr_means, corr_stds)],
                        [m+s for m,s in zip(corr_means, corr_stds)],
                        color=colors_p[p], alpha=0.15)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Pearson Corr(x0, xt)')
    ax.set_title('Terminal Correlation vs Timestep')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # NMSE vs t
    ax = axes[1]
    for p in protocols:
        p_data = [r for r in ablation_agg if r['protocol'] == p]
        ts = [r['timestep'] for r in p_data]
        nmse_means = [r['nmse_mean'] for r in p_data]
        nmse_stds = [r['nmse_std'] for r in p_data]
        ax.plot(ts, nmse_means, line_styles[p], color=colors_p[p], linewidth=2, label=p)
        ax.fill_between(ts,
                        [m-s for m,s in zip(nmse_means, nmse_stds)],
                        [m+s for m,s in zip(nmse_means, nmse_stds)],
                        color=colors_p[p], alpha=0.15)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('NMSE(x0, xt)')
    ax.set_title('NMSE vs Timestep')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Spectral distance vs t
    ax = axes[2]
    for p in protocols:
        p_data = [r for r in ablation_agg if r['protocol'] == p]
        ts = [r['timestep'] for r in p_data]
        spec_means = [r['spectral_mean'] for r in p_data]
        spec_stds = [r['spectral_std'] for r in p_data]
        ax.plot(ts, spec_means, line_styles[p], color=colors_p[p], linewidth=2, label=p)
        ax.fill_between(ts,
                        [m-s for m,s in zip(spec_means, spec_stds)],
                        [m+s for m,s in zip(spec_means, spec_stds)],
                        color=colors_p[p], alpha=0.15)
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('Spectral Distance')
    ax.set_title('Spectral Distance vs Timestep')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('TFD 3-Way Mechanism Ablation — WiFi (Released Configuration)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    for suffix in ['corr', 'nmse', 'spectral_distance']:
        fig_path = f'{ARTIFACTS_DIR}/tfd_{suffix}_vs_timestep.png'
        # We save the full combined figure and also individual ones
    fig_path = f'{ARTIFACTS_DIR}/tfd_ablation_combined.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  TFD ablation combined: {fig_path}')

    # Individual TFD figures
    metric_names = [
        ('corr', 'Pearson Corr(x0, xt)', 'corr_mean', 'corr_std'),
        ('nmse', 'NMSE(x0, xt)', 'nmse_mean', 'nmse_std'),
        ('spectral_distance', 'Normalized L2 Spectral Distance', 'spectral_mean', 'spectral_std'),
    ]
    for suffix, ylabel, mean_key, std_key in metric_names:
        fig, ax = plt.subplots(figsize=(10, 6))
        for p in protocols:
            p_data = [r for r in ablation_agg if r['protocol'] == p]
            ts = [r['timestep'] for r in p_data]
            means = [r[mean_key] for r in p_data]
            stds = [r[std_key] for r in p_data]
            ax.plot(ts, means, line_styles[p], color=colors_p[p], linewidth=2, label=p)
            ax.fill_between(ts,
                           [m-s for m,s in zip(means, stds)],
                           [m+s for m,s in zip(means, stds)],
                           color=colors_p[p], alpha=0.15)
        ax.set_xlabel('Timestep t', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'TFD 3-Way Ablation: {ylabel} vs Timestep\nWiFi Released Configuration (blur σ²=1e-10)',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = f'{ARTIFACTS_DIR}/tfd_{suffix}_vs_timestep.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  TFD {suffix}: {fig_path}')

    # ---- alpha_bar vs Empirical Correlation ----
    # Use official Noise+Blur data for this
    nb_data = [r for r in ablation_agg if r['protocol'] == 'Noise+Blur']
    alpha_vals = [diffusion.alpha_bar[r['timestep']].item() for r in nb_data]
    corr_vals = [r['corr_mean'] for r in nb_data]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(alpha_vals, corr_vals, 'o-', color='purple', linewidth=2, markersize=8,
           label='Empirical r(x0, xt) vs ᾱ(t)')
    # Identity line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='y=x (identity)')
    # Annotation
    ax.annotate(f'ᾱ(99)={alpha_vals[-1]:.4f}\nr={corr_vals[-1]:.4f}\nRatio={corr_vals[-1]/alpha_vals[-1]:.4f}',
               xy=(alpha_vals[-1], corr_vals[-1]), xytext=(0.3, 0.15),
               arrowprops=dict(arrowstyle='->', color='gray'),
               fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    ax.set_xlabel('ᾱ(t) = Π(1-β_s)', fontsize=12)
    ax.set_ylabel('Empirical Pearson r(x0, xt)', fontsize=12)
    ax.set_title('ᾱ(t) vs Empirical Signal Correlation\nWiFi Released Configuration',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig_path = f'{ARTIFACTS_DIR}/alpha_bar_vs_empirical_corr.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  alpha_bar vs corr: {fig_path}')


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    t_start = time.time()

    print('='*60)
    print('FINAL VALIDATION — RF-Diffusion WiFi')
    print('='*60)

    # Load data (shared across experiments)
    samples = load_all_samples()
    print(f'\nLoaded {len(samples)} WiFi samples')

    # ---- OFFICIAL CHECKPOINT ----
    print(f'\n{"#"*70}')
    print(f'# OFFICIAL CHECKPOINT')
    print(f'{"#"*70}')

    ckpt_official = f'{BASE_DIR}/official/model/wifi/b32-256-100s/weights.pt'
    model_o, diffusion_o, params_o = load_model(ckpt_official)
    print(f'Loaded official model from {ckpt_official}')

    # Source dependence
    src_results_o, src_summary_o = run_source_dependence(
        model_o, diffusion_o, params_o, samples, 'official')
    torch.cuda.empty_cache()

    # Native vs Fast
    nf_results_o, nf_summary_o = run_native_vs_fast(
        model_o, diffusion_o, params_o, samples, 'official')
    torch.cuda.empty_cache()

    # TFD Ablation (uses official diffusion, shared across checkpoints)
    ablation_results, ablation_agg = run_tfd_ablation(
        diffusion_o, params_o, samples, 'official')

    # Free official model memory
    del model_o
    torch.cuda.empty_cache()

    # ---- SELF-TRAINED CHECKPOINT ----
    print(f'\n{"#"*70}')
    print(f'# SELF-TRAINED CHECKPOINT')
    print(f'{"#"*70}')

    ckpt_self = f'{BASE_DIR}/runs/wifi_train/weights-final.pt'
    model_s, diffusion_s, params_s = load_model(ckpt_self)
    print(f'Loaded self-trained model from {ckpt_self}')

    src_results_s, src_summary_s = run_source_dependence(
        model_s, diffusion_s, params_s, samples, 'selftrained')
    torch.cuda.empty_cache()

    nf_results_s, nf_summary_s = run_native_vs_fast(
        model_s, diffusion_s, params_s, samples, 'selftrained')
    torch.cuda.empty_cache()

    del model_s
    torch.cuda.empty_cache()

    # ---- GENERATE FIGURES ----
    generate_figures(src_results_o, src_results_s,
                    nf_results_o, nf_results_s,
                    ablation_agg, diffusion_o, params_o)

    # ---- FINAL SUMMARY ----
    elapsed = time.time() - t_start
    print(f'\n{"="*60}')
    print(f'FINAL VALIDATION COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print(f'Results: {RESULTS_DIR}/')
    print(f'Artifacts: {ARTIFACTS_DIR}/')
    print(f'{"="*60}')

    # Quick summary for immediate reading
    print(f'\n{"="*60}')
    print('QUICK SUMMARY')
    print(f'{"="*60}')

    # Source dependence
    if src_summary_o:
        a_key = 'A_output_to_own_source'
        b_key = 'B_output_to_unrelated_source'
        c_key = 'C_cross_terminal_output_similarity'

        print(f'\nOfficial Source Dependence:')
        if a_key in src_summary_o:
            s = src_summary_o[a_key]['pearson_corr']
            print(f'  Metric A (output-to-own-source): {s["mean"]:.4f}±{s["std"]:.4f} (N={s["N"]})')
        if b_key in src_summary_o:
            s = src_summary_o[b_key]['pearson_corr']
            print(f'  Metric B (output-to-unrelated-source): {s["mean"]:.4f}±{s["std"]:.4f} (N={s["N"]})')
        if c_key in src_summary_o:
            s = src_summary_o[c_key]['pearson_corr']
            print(f'  Metric C (cross-terminal-output): {s["mean"]:.4f}±{s["std"]:.4f} (N={s["N"]})')

        if a_key in src_summary_o and b_key in src_summary_o:
            sds = src_summary_o[a_key]['pearson_corr']['mean'] - src_summary_o[b_key]['pearson_corr']['mean']
            print(f'  SDS = {sds:.4f}')

    if src_summary_s:
        a_key = 'A_output_to_own_source'
        print(f'\nSelf-trained Source Dependence:')
        if a_key in src_summary_s:
            s = src_summary_s[a_key]['pearson_corr']
            print(f'  Metric A (output-to-own-source): {s["mean"]:.4f}±{s["std"]:.4f} (N={s["N"]})')

    # TFD ablation summary
    print(f'\nTFD Ablation at t=99:')
    for p_data in ablation_agg:
        if p_data['timestep'] == 99:
            print(f'  {p_data["protocol"]}: corr={p_data["corr_mean"]:.4f}±{p_data["corr_std"]:.4f}')

    # Blur assessment
    blur_sched = np.array(params_o.blur_schedule)
    print(f'\nWiFi Blur Schedule: min={blur_sched.min():.2e}, max={blur_sched.max():.2e}')
    if blur_sched.max() < 1e-8:
        print(f'  BLUR CONTRIBUTION: NEGLIGIBLE at released WiFi config')

    print(f'\nDONE. All results in {RESULTS_DIR}/ and {ARTIFACTS_DIR}/')
