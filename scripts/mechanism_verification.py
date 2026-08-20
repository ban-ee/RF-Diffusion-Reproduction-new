"""
FINAL MECHANISM VERIFICATION — RF-Diffusion WiFi

P0: Alpha_bar independent recomputation + monotonicity check
P1: Dump official SignalDiffusion internal tensors
P2: Blur analysis — deterministic attenuation vs indirect noise effects
P3: get_noise_weights() full audit — float32/float64/stable comparison
P4: Redefined forward protocol comparison (4 variants)
P5: Paired SDS from existing data (if available)

All CPU-only except optional paired SDS inference.
"""
import os, sys, json, csv, time, warnings
import numpy as np
import torch
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

os.chdir(REPO_ROOT + '/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params
from tfdiff.diffusion import SignalDiffusion

# ============================================================
BASE_DIR = REPO_ROOT
RESULTS_DIR = f'{BASE_DIR}/experiments/mechanism_verification'
ARTIFACTS_DIR = f'{BASE_DIR}/artifacts/mechanism_verification'
REPORTS_DIR = f'{BASE_DIR}/reports/mechanism_verification'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

params = all_params[0]  # WiFi
TIMESTEPS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
DTYPE_EPS = torch.finfo(torch.float32).eps

print('='*70)
print('FINAL MECHANISM VERIFICATION')
print('='*70)

# ============================================================
# P0: ALPHA_BAR INDEPENDENT RECOMPUTATION
# ============================================================
print('\n' + '='*70)
print('P0: ALPHA_BAR VERIFICATION')
print('='*70)

beta_np = np.array(params.noise_schedule, dtype=np.float64)
alpha_np = 1.0 - beta_np
alpha_bar_np64 = np.cumprod(alpha_np)

# float32 version (to check)
beta_np32 = np.array(params.noise_schedule, dtype=np.float32)
alpha_np32 = 1.0 - beta_np32
alpha_bar_np32 = np.cumprod(alpha_np32)

# Official torch object
diffusion = SignalDiffusion(params)
alpha_bar_official = diffusion.alpha_bar.numpy()

# Check monotonicity
is_monotonic = bool(np.all(np.diff(alpha_bar_np64) <= 0))
print(f'alpha_bar monotonic decreasing: {is_monotonic}')

# Verify official
diff_off = np.abs(alpha_bar_official - alpha_bar_np32)
max_diff = diff_off.max()
print(f'Max |official - numpy32|: {max_diff:.2e}')
print(f'alpha_bar[99] official: {alpha_bar_official[99]:.10f}')
print(f'alpha_bar[99] np64:     {alpha_bar_np64[99]:.10f}')
print(f'alpha_bar[99] np32:     {alpha_bar_np32[99]:.10f}')

# Check previous table: was t=90 alpha_bar=0.436 claimed?
a90 = alpha_bar_official[90]
a99 = alpha_bar_official[99]
print(f'alpha_bar[90]: {a90:.6f}')
print(f'alpha_bar[99]: {a99:.6f}')
print(f'alpha_bar[90] <= alpha_bar[99]? {a90 <= a99} (should be >= due to monotonic)')
if a90 > a99:
    print('  monotonicity OK: alpha_bar[90] > alpha_bar[99]')
else:
    print('  WARNING: non-monotonic!')

# Save alpha_bar reference
csv_path = f'{RESULTS_DIR}/alpha_bar_reference.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t', 'beta', 'alpha', 'alpha_bar_np64', 'alpha_bar_np32',
                     'alpha_bar_official', 'abs_diff', 'monotonic'])
    for t in range(100):
        writer.writerow([
            t, beta_np[t], alpha_np[t],
            f'{alpha_bar_np64[t]:.12f}',
            f'{alpha_bar_np32[t]:.12f}',
            f'{alpha_bar_official[t]:.12f}',
            f'{abs(alpha_bar_official[t] - alpha_bar_np32[t]):.2e}',
            'OK' if t == 0 or alpha_bar_np64[t] <= alpha_bar_np64[t-1] else 'NON-MONOTONIC'
        ])
print(f'Saved: {csv_path}')

# Quick check: all beta > 0?
all_beta_positive = bool(np.all(beta_np > 0))
print(f'All beta > 0: {all_beta_positive}')
if all_beta_positive:
    print('  → alpha_bar MUST be strictly decreasing (confirmed)')

# ============================================================
# P1: DUMP OFFICIAL INTERNAL WEIGHTS
# ============================================================
print('\n' + '='*70)
print('P1: OFFICIAL INTERNAL WEIGHTS')
print('='*70)

gaussian_kernel = diffusion.gaussian_kernel  # [T, N]
gaussian_kernel_bar = diffusion.gaussian_kernel_bar  # [T, N]
info_weights = diffusion.info_weights  # [T, N]
noise_weights = diffusion.noise_weights  # [T, N]
alpha = diffusion.alpha  # [T]
alpha_bar = diffusion.alpha_bar  # [T]
var_blur = diffusion.var_blur  # [T]
var_blur_bar = diffusion.var_blur_bar  # [T]

# Save summary CSV
csv_path = f'{RESULTS_DIR}/official_internal_weights_summary.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t', 'alpha_bar', 'alpha',
                     'gkernel_min', 'gkernel_max', 'gkernel_mean', 'gkernel_std',
                     'gkernel_bar_min', 'gkernel_bar_max', 'gkernel_bar_mean', 'gkernel_bar_std',
                     'info_w_min', 'info_w_max', 'info_w_mean', 'info_w_std',
                     'noise_w_min', 'noise_w_max', 'noise_w_mean', 'noise_w_std', 'noise_w_l2',
                     'var_blur', 'var_blur_bar',
                     'gkernel_deviation_from_one_max'])
    for t in range(100):
        gk = gaussian_kernel[t].numpy()
        gkb = gaussian_kernel_bar[t].numpy()
        iw = info_weights[t].numpy()
        nw = noise_weights[t].numpy()
        writer.writerow([
            t,
            f'{alpha_bar[t].item():.8f}',
            f'{alpha[t].item():.8f}',
            f'{gk.min():.6e}', f'{gk.max():.6e}', f'{gk.mean():.6e}', f'{gk.std():.6e}',
            f'{gkb.min():.6e}', f'{gkb.max():.6e}', f'{gkb.mean():.6e}', f'{gkb.std():.6e}',
            f'{iw.min():.6e}', f'{iw.max():.6e}', f'{iw.mean():.6e}', f'{iw.std():.6e}',
            f'{nw.min():.6e}', f'{nw.max():.6e}', f'{nw.mean():.6e}', f'{nw.std():.6e}',
            f'{np.linalg.norm(nw):.6e}',
            f'{var_blur[t].item():.6e}', f'{var_blur_bar[t].item():.6e}',
            f'{np.abs(gkb - 1.0).max():.6e}',
        ])
print(f'Saved: {csv_path}')

# Key check: gaussian_kernel_bar deviation from 1
for t_name, t_val in [('t=0', 0), ('t=50', 50), ('t=90', 90), ('t=99', 99)]:
    gkb = gaussian_kernel_bar[t_val].numpy()
    dev = np.abs(gkb - 1.0).max()
    print(f'  {t_name}: max|G_bar - 1| = {dev:.6e}')

# Check noise_weights vs sqrt(1-alpha_bar)
print('\n  noise_weights vs sqrt(1-alpha_bar):')
for t_val in [0, 10, 50, 90, 99]:
    nw = noise_weights[t_val].numpy()
    ref = np.sqrt(1 - alpha_bar[t_val].item())
    nw_mean = nw.mean()
    print(f'  t={t_val}: noise_w mean={nw_mean:.6f}, sqrt(1-ᾱ)={ref:.6f}, ratio={nw_mean/ref:.4f}')

# ============================================================
# P2-P3: get_noise_weights() FULL AUDIT
# ============================================================
print('\n' + '='*70)
print('P2-P3: get_noise_weights() FULL AUDIT')
print('='*70)

# Float32 (official)
diffusion_f32 = SignalDiffusion(params)
nw_f32 = diffusion_f32.noise_weights

# Float64 reconstruction
beta64 = torch.tensor(np.array(params.noise_schedule, dtype=np.float64))
alpha64 = 1.0 - beta64
alpha_bar64 = torch.cumprod(alpha64, dim=0)
var_blur64 = torch.tensor(np.array(params.blur_schedule, dtype=np.float64))
var_blur_bar64 = torch.cumsum(var_blur64, dim=0)
input_dim = params.sample_rate

def get_kernel_64(var_kernel):
    samples = torch.arange(0, input_dim, dtype=torch.float64)
    gk = torch.exp(-((samples - input_dim // 2)**2) / (2 * var_kernel)) / torch.sqrt(2 * torch.pi * var_kernel)
    gk = input_dim * gk / torch.sum(gk, dim=1, keepdim=True)
    return gk

var_kernel64 = (input_dim / var_blur64).unsqueeze(1)
var_kernel_bar64 = (input_dim / var_blur_bar64).unsqueeze(1)
gkernel64 = get_kernel_64(var_kernel64)
gkernel_bar64 = get_kernel_64(var_kernel_bar64)

def get_noise_weights_64(alpha_64, var_blur_64, input_dim_64):
    noise_weights_64 = []
    max_step = len(alpha_64)
    for t in range(max_step):
        upper_bound = t + 1
        one_minus_alpha_sqrt = torch.sqrt(1 - alpha_64[:upper_bound])
        rev_one_minus_alpha_sqrt = torch.flipud(one_minus_alpha_sqrt)
        rev_alpha = torch.flipud(alpha_64[:upper_bound])
        rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1])
        rev_var_blur = torch.flipud(var_blur_64[:upper_bound])
        rev_var_blur_bar = torch.cumsum(rev_var_blur, dim=0) - rev_var_blur[-1]

        # Check for negative values
        has_neg = (rev_var_blur_bar < 0).any().item()

        rev_var_kernel_bar = (input_dim_64 / rev_var_blur_bar).unsqueeze(1)
        rev_kernel_bar = get_kernel_64(rev_var_kernel_bar)
        rev_kernel_bar[0, :] = torch.ones(input_dim_64)
        nw = torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1),
                      rev_one_minus_alpha_sqrt)
        noise_weights_64.append(nw)
    return torch.stack(noise_weights_64, dim=0)

nw_f64 = get_noise_weights_64(alpha64, var_blur64, input_dim)

# Stable DDPM reference: sqrt(1-alpha_bar)
nw_stable = torch.sqrt(1 - alpha_bar64).unsqueeze(-1).expand(-1, input_dim)

# Comparison
print('\n  Float32 vs Float64 vs Stable comparison:')
csv_path = f'{RESULTS_DIR}/noise_weights_precision_comparison.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t', 'nw_f32_mean', 'nw_f32_std', 'nw_f32_min', 'nw_f32_max',
                     'nw_f64_mean', 'nw_f64_std', 'nw_f64_min', 'nw_f64_max',
                     'nw_stable_mean',
                     'f32_vs_f64_max_rel_diff', 'f32_vs_stable_ratio',
                     'f64_vs_stable_ratio'])
    for t_val in TIMESTEPS:
        n32 = nw_f32[t_val].numpy()
        n64 = nw_f64[t_val].numpy()
        ns = nw_stable[t_val].numpy()
        rel_diff = np.abs(n32 - n64).max() / (np.abs(n64).mean() + 1e-30)
        r32 = n32.mean() / (ns.mean() + 1e-30)
        r64 = n64.mean() / (ns.mean() + 1e-30)
        writer.writerow([t_val,
                        f'{n32.mean():.6e}', f'{n32.std():.6e}', f'{n32.min():.6e}', f'{n32.max():.6e}',
                        f'{n64.mean():.6e}', f'{n64.std():.6e}', f'{n64.min():.6e}', f'{n64.max():.6e}',
                        f'{ns.mean():.6e}',
                        f'{rel_diff:.6e}', f'{r32:.6f}', f'{r64:.6f}'])
        print(f'  t={t_val}: f32={n32.mean():.6f}, f64={n64.mean():.6f}, '
              f'stable={ns.mean():.6f}, ratio_f32={r32:.4f}, ratio_f64={r64:.4f}')

print(f'Saved: {csv_path}')

# ============================================================
# REV_VAR_BLUR DEBUG — check for negative values
# ============================================================
print('\n' + '='*70)
print('REV_VAR_BLUR DEBUG')
print('='*70)

csv_path = f'{RESULTS_DIR}/rev_var_blur_debug.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['t_step', 'index_in_loop', 'rev_var_blur_val', 'cumsum_val',
                     'subtracted_val', 'rev_var_blur_bar_val', 'is_negative', 'is_zero',
                     'var_kernel_bar_val', 'is_inf'])

    for t_step in [10, 50, 90, 99]:
        upper_bound = t_step + 1
        rev_var_blur = np.flipud(np.array(params.blur_schedule[:upper_bound], dtype=np.float64))
        cumsum_val = np.cumsum(rev_var_blur)
        subtracted = rev_var_blur[-1]
        rev_var_blur_bar = cumsum_val - subtracted

        n_neg = int(np.sum(rev_var_blur_bar < 0))
        n_zero = int(np.sum(np.abs(rev_var_blur_bar) < 1e-30))

        print(f'\n  t={t_step} (upper_bound={upper_bound}):')
        print(f'    rev_var_blur: all={rev_var_blur[0]:.2e}')
        print(f'    subtracted: {subtracted:.2e}')
        print(f'    rev_var_blur_bar: range=[{rev_var_blur_bar.min():.2e}, {rev_var_blur_bar.max():.2e}]')
        print(f'    n_negative: {n_neg}, n_zero: {n_zero}')

        for i in range(len(rev_var_blur)):
            vkb = input_dim / max(rev_var_blur_bar[i], 1e-300)
            writer.writerow([t_step, i,
                           f'{rev_var_blur[i]:.6e}',
                           f'{cumsum_val[i]:.6e}',
                           f'{subtracted:.6e}',
                           f'{rev_var_blur_bar[i]:.6e}',
                           'YES' if rev_var_blur_bar[i] < 0 else 'NO',
                           'YES' if abs(rev_var_blur_bar[i]) < 1e-30 else 'NO',
                           f'{vkb:.6e}',
                           'YES' if vkb > 1e20 else 'NO'])

print(f'Saved: {csv_path}')

# ============================================================
# P4: REDEFINED FORWARD PROTOCOL COMPARISON
# ============================================================
print('\n' + '='*70)
print('P4: FORWARD PROTOCOL COMPARISON')
print('='*70)

# Load WiFi samples (CPU-only forward pass)
from tfdiff.dataset import WiFiDataset, Collator
collator = Collator(params)
params_cp = all_params[0]
params_cp.data_dir = ['./wifi/cond']
dataset = WiFiDataset(params_cp.data_dir)
N_data = len(dataset)
print(f'Loaded {N_data} WiFi samples')

# Pre-load all samples (collated)
all_samples = []
for i in range(N_data):
    raw = dataset[i]
    batch = collator.collate([raw])
    all_samples.append({
        'data': batch['data'],  # [1, 512, 90, 2]
        'idx': i,
    })

# Prepare protocol weights
alpha_bar_t = diffusion.alpha_bar
alpha_bar_t64 = alpha_bar64

# Standard DDPM info/noise weights
info_ddpm = torch.sqrt(alpha_bar_t)  # [T] scalar
noise_ddpm = torch.sqrt(1 - alpha_bar_t)  # [T] scalar

# Official TFD info/noise weights
info_tfd = diffusion.info_weights  # [T, N]
noise_tfd = diffusion.noise_weights  # [T, N]

# TFD_NO_DIRECT_BLUR: info = sqrt(alpha_bar), noise = official
info_no_direct_blur = torch.sqrt(alpha_bar_t).unsqueeze(-1).expand(-1, input_dim)
noise_no_direct_blur = noise_tfd  # Same as official

# Reference-noise TFD: use float64 stable noise, official info
# Only if f64 noise is significantly different from f32
noise_ref = nw_stable  # [T, N] — standard DDPM noise, broadcast to N
info_ref = torch.sqrt(alpha_bar_t).unsqueeze(-1).expand(-1, input_dim)

# Run forward comparison
# IMPORTANT: degrade_fn() hardcodes torch.manual_seed(11).
# Strategy: call OFFICIAL_TFD first, then capture the same epsilon by
# re-setting seed to 11 and regenerating randn_like — this ensures
# all protocols use the SAME epsilon.

results = []

for sample_idx in range(N_data):
    data_0 = all_samples[sample_idx]['data']  # [1, 512, 90, 2]

    for t_step in TIMESTEPS:
        t_tensor = torch.tensor([t_step], dtype=torch.int64)

        # ---- B: OFFICIAL_TFD (call first — it sets seed=11 internally) ----
        x_t_b = diffusion.degrade_fn(data_0, t_tensor, 0)

        # Capture the epsilon that degrade_fn used (seed → 11 → randn_like)
        torch.manual_seed(11)
        eps = torch.randn_like(data_0)

        # ---- A: STANDARD_DDPM_REFERENCE (same eps, DDPM weights) ----
        iw_a = info_ddpm[t_step].view(1, 1, 1, 1)    # scalar broadcast
        nw_a = noise_ddpm[t_step].view(1, 1, 1, 1)
        x_t_a = iw_a * data_0 + nw_a * eps

        # ---- C: TFD_NO_DIRECT_BLUR (info=sqrt(alpha_bar) flat, noise=official) ----
        iw_c = info_no_direct_blur[t_step].view(1, -1, 1, 1)
        nw_c = noise_no_direct_blur[t_step].view(1, -1, 1, 1)
        x_t_c = iw_c * data_0 + nw_c * eps

        # ---- D: TFD_REFERENCE_NOISE (DDPM info + DDPM noise, both flat) ----
        iw_d = info_ref[t_step].view(1, -1, 1, 1)
        nw_d = noise_ref[t_step].view(1, -1, 1, 1)
        x_t_d = iw_d * data_0 + nw_d * eps

        # Compute metrics for each
        for proto_name, x_t in [('STANDARD_DDPM', x_t_a), ('OFFICIAL_TFD', x_t_b),
                                 ('TFD_NO_DIRECT_BLUR', x_t_c), ('TFD_REFERENCE_NOISE', x_t_d)]:
            p = x_t.detach().numpy().ravel()
            t_arr = data_0.detach().numpy().ravel()
            corr = float(np.corrcoef(p, t_arr)[0, 1]) if len(p) > 1 else 0.0
            nmse = float(np.mean((p - t_arr)**2) / (np.mean(t_arr**2) + 1e-30))
            energy_ratio = float(np.sum(p**2) / (np.sum(t_arr**2) + 1e-30))
            # Complex-magnitude distance
            x_t_complex = x_t.detach().numpy()
            data_complex = data_0.detach().numpy()
            p_mag = np.sqrt(x_t_complex[..., 0]**2 + x_t_complex[..., 1]**2).ravel()
            t_mag = np.sqrt(data_complex[..., 0]**2 + data_complex[..., 1]**2).ravel()
            cmag_dist = float(np.linalg.norm(p_mag - t_mag) / (np.linalg.norm(t_mag) + 1e-30))

            results.append({
                'sample_idx': sample_idx,
                'timestep': t_step,
                'protocol': proto_name,
                'alpha_bar': float(alpha_bar_t[t_step].item()),
                'pearson_corr': corr,
                'nmse': nmse,
                'energy_ratio': energy_ratio,
                'complex_magnitude_distance': cmag_dist,
            })

# Save
csv_path = f'{RESULTS_DIR}/forward_protocol_comparison.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f'\nSaved: {csv_path} ({len(results)} rows)')

# Aggregate by protocol × timestep
protocols = sorted(set(r['protocol'] for r in results))
agg = []
for proto in protocols:
    for ts in TIMESTEPS:
        tsr = [r for r in results if r['protocol'] == proto and r['timestep'] == ts]
        corrs = [r['pearson_corr'] for r in tsr]
        nmses = [r['nmse'] for r in tsr]
        cmags = [r['complex_magnitude_distance'] for r in tsr]
        agg.append({
            'protocol': proto, 'timestep': ts, 'N': len(tsr),
            'corr_mean': float(np.mean(corrs)), 'corr_std': float(np.std(corrs)),
            'nmse_mean': float(np.mean(nmses)), 'nmse_std': float(np.std(nmses)),
            'cmag_dist_mean': float(np.mean(cmags)), 'cmag_dist_std': float(np.std(cmags)),
        })

# Print final timestep summary
print('\n  === t=99 SUMMARY ===')
for proto in protocols:
    a = [r for r in agg if r['protocol'] == proto and r['timestep'] == 99][0]
    print(f'  {proto}: corr={a["corr_mean"]:.4f}±{a["corr_std"]:.4f}, '
          f'NMSE={a["nmse_mean"]:.4f}, cmag_dist={a["cmag_dist_mean"]:.4f}')

# ============================================================
# P5: PAIRED SDS FROM EXISTING DATA
# ============================================================
print('\n' + '='*70)
print('P5: PAIRED SOURCE DEPENDENCE SCORE')
print('='*70)

# Try to load existing source dependence CSVs
src_paths = {
    'official': f'{BASE_DIR}/experiments/final_validation/source_dependence_official.csv',
    'selftrained': f'{BASE_DIR}/experiments/final_validation/source_dependence_selftrained.csv',
}

for tag, src_path in src_paths.items():
    if not os.path.exists(src_path):
        print(f'  {tag}: NOT FOUND at {src_path}')
        continue

    print(f'\n  Computing paired SDS for {tag}...')
    all_rows = []
    with open(src_path, 'r') as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    # Group by (seed, condition_idx)
    from collections import defaultdict
    groups = defaultdict(lambda: {'own_source': [], 'unrelated_source': [], 'cross_terminal': []})

    for row in all_rows:
        key = (row['seed'], row['condition_idx'])
        metric = row['metric']
        corr = float(row['pearson_corr'])
        if metric == 'A_output_to_own_source':
            groups[key]['own_source'].append(corr)
        elif metric == 'B_output_to_unrelated_source':
            groups[key]['unrelated_source'].append(corr)
        elif metric == 'C_cross_terminal_output_similarity':
            groups[key]['cross_terminal'].append(corr)

    # Compute paired differences: for each (seed, cond), pair own_source value
    # with each unrelated_source value from the SAME group
    paired_diffs = []
    for key, vals in groups.items():
        for own_val in vals['own_source']:
            for unr_val in vals['unrelated_source']:
                paired_diffs.append(own_val - unr_val)

    if paired_diffs:
        diffs = np.array(paired_diffs)
        # Bootstrap CI
        n_bs = 2000
        bs_means = []
        rng = np.random.RandomState(42)
        n = len(diffs)
        for _ in range(n_bs):
            idx = rng.randint(0, n, n)
            bs_means.append(np.mean(diffs[idx]))
        bs_means = np.array(bs_means)
        ci_lo = float(np.percentile(bs_means, 2.5))
        ci_hi = float(np.percentile(bs_means, 97.5))

        paired_sds = float(np.mean(diffs))
        print(f'    Paired SDS = {paired_sds:.4f}')
        print(f'    N_pairs = {len(diffs)}')
        print(f'    Mean ± Std = {paired_sds:.4f} ± {np.std(diffs):.4f}')
        print(f'    Median = {np.median(diffs):.4f}')
        print(f'    95% bootstrap CI = [{ci_lo:.4f}, {ci_hi:.4f}]')

        # Save
        csv_out = f'{RESULTS_DIR}/paired_sds_{tag}.csv'
        with open(csv_out, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pair_idx', 'own_source_corr', 'unrelated_source_corr', 'delta'])
            for i, (own, unr) in enumerate(zip(
                [v for vals in groups.values() for v in vals['own_source']],
                [v for vals in groups.values() for v in vals['unrelated_source']]
            )):
                if i < len(paired_diffs):
                    writer.writerow([i, f'{own:.6f}', f'{unr:.6f}', f'{own-unr:.6f}'])

        summary = {
            'tag': tag,
            'paired_sds_mean': paired_sds,
            'paired_sds_std': float(np.std(diffs)),
            'paired_sds_median': float(np.median(diffs)),
            'paired_sds_ci95_low': ci_lo,
            'paired_sds_ci95_high': ci_hi,
            'N_pairs': len(diffs),
        }
        json_out = f'{RESULTS_DIR}/paired_sds_summary.json'
        # Append or create
        all_summaries = {}
        if os.path.exists(json_out):
            with open(json_out) as jf:
                all_summaries = json.load(jf)
        all_summaries[tag] = summary
        with open(json_out, 'w') as jf:
            json.dump(all_summaries, jf, indent=2)
        print(f'    Saved: {csv_out}')
    else:
        print(f'    No paired data available for {tag}')

# ============================================================
# FINAL NUMERICAL INSTABILITY VERDICT
# ============================================================
print('\n' + '='*70)
print('NUMERICAL INSTABILITY VERDICT')
print('='*70)

# Check: float32 vs float64 max relative difference
max_rel_diff = 0.0
for t_val in TIMESTEPS:
    n32 = nw_f32[t_val].numpy()
    n64 = nw_f64[t_val].numpy()
    rd = np.abs(n32 - n64).max() / (np.abs(n64).mean() + 1e-30)
    max_rel_diff = max(max_rel_diff, rd)

# Check: float32 noise_weights vs stable
n32_t99 = nw_f32[99].numpy().mean()
nstable_t99 = np.sqrt(1 - alpha_bar_np64[99])
ratio_f32_stable = n32_t99 / nstable_t99 if nstable_t99 > 0 else float('inf')

# Check: float64 noise_weights vs stable
n64_t99 = nw_f64[99].numpy().mean()
ratio_f64_stable = n64_t99 / nstable_t99 if nstable_t99 > 0 else float('inf')

# Check: official terminal corr vs standard DDPM terminal corr
off_corr_t99 = [r for r in agg if r['protocol'] == 'OFFICIAL_TFD' and r['timestep'] == 99][0]
ddpm_corr_t99 = [r for r in agg if r['protocol'] == 'STANDARD_DDPM' and r['timestep'] == 99][0]
no_db_corr_t99 = [r for r in agg if r['protocol'] == 'TFD_NO_DIRECT_BLUR' and r['timestep'] == 99][0]
ref_corr_t99 = [r for r in agg if r['protocol'] == 'TFD_REFERENCE_NOISE' and r['timestep'] == 99][0]

print(f'\n  Noise weights comparison at t=99:')
print(f'    float32 mean:  {n32_t99:.6f}')
print(f'    float64 mean:  {n64_t99:.6f}')
print(f'    stable (DDPM): {nstable_t99:.6f}')
print(f'    f32/stable ratio: {ratio_f32_stable:.4f}')
print(f'    f64/stable ratio: {ratio_f64_stable:.4f}')
print(f'    max f32 vs f64 rel diff: {max_rel_diff:.2e}')

print(f'\n  Terminal correlation at t=99:')
print(f'    STANDARD_DDPM:    {ddpm_corr_t99["corr_mean"]:.4f}')
print(f'    OFFICIAL_TFD:     {off_corr_t99["corr_mean"]:.4f}')
print(f'    TFD_NO_DIRECT_BLUR: {no_db_corr_t99["corr_mean"]:.4f}')
print(f'    TFD_REFERENCE_NOISE: {ref_corr_t99["corr_mean"]:.4f}')

# Verdict
print(f'\n  === VERDICT ===')
if max_rel_diff < 1e-6 and abs(ratio_f32_stable - 1.0) < 0.05:
    print(f'  max f32/f64 rel diff = {max_rel_diff:.2e} < 1e-6 → f32 precision ADEQUATE')
    print(f'  f32/stable ratio = {ratio_f32_stable:.4f} ≈ 1.0 → noise_weights ≈ sqrt(1-ᾱ)')
    print(f'  → NUMERICAL INSTABILITY: NOT SUPPORTED')
    print(f'  → Cause of low terminal corr is NOT numerical precision')
else:
    print(f'  max f32/f64 rel diff = {max_rel_diff:.2e}')
    if max_rel_diff > 1e-3:
        print(f'  → f32 vs f64 DIVERGE → precision issue confirmed')
    if abs(ratio_f32_stable - 1.0) > 0.1:
        print(f'  → noise_weights substantially differ from sqrt(1-ᾱ)')
    print(f'  f32/stable ratio = {ratio_f32_stable:.4f}')

if abs(no_db_corr_t99["corr_mean"] - off_corr_t99["corr_mean"]) < 0.01:
    print(f'  TFD_NO_DIRECT_BLUR ≈ OFFICIAL_TFD → direct blur attenuation NEGLIGIBLE')
else:
    print(f'  TFD_NO_DIRECT_BLUR ≠ OFFICIAL_TFD → direct blur has material effect')

if abs(ref_corr_t99["corr_mean"] - ddpm_corr_t99["corr_mean"]) < 0.01:
    print(f'  TFD_REFERENCE_NOISE ≈ STANDARD_DDPM → noise drives the difference')
else:
    print(f'  TFD_REFERENCE_NOISE corr={ref_corr_t99["corr_mean"]:.4f}')

print(f'\n  ROOT CAUSE ANALYSIS:')
gap_off_ddpm = ddpm_corr_t99["corr_mean"] - off_corr_t99["corr_mean"]
gap_off_nodb = no_db_corr_t99["corr_mean"] - off_corr_t99["corr_mean"]
gap_off_ref = ref_corr_t99["corr_mean"] - off_corr_t99["corr_mean"]
print(f'  Gap(DDPM - Official): {gap_off_ddpm:.4f}')
print(f'  Gap(TFD_NO_DIRECT_BLUR - Official): {gap_off_nodb:.4f} (direct blur effect)')
print(f'  Gap(TFD_REFERENCE_NOISE - Official): {gap_off_ref:.4f} (noise weight effect)')

print(f'\n{"="*70}')
print('MECHANISM VERIFICATION COMPLETE')
print(f'Results: {RESULTS_DIR}/')
print(f'{"="*70}')
