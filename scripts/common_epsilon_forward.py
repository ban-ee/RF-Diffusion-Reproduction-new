"""
COMMON-EPSILON FORWARD COUNTERFACTUAL — RF-Diffusion WiFi

P5-P8: True common-epsilon comparison across 4 forward protocols.
Includes regression check: manual official reconstruction vs official degrade_fn.

Protocols:
  A. STANDARD_DDPM_REFERENCE
  B. OFFICIAL_TFD_MANUAL (reconstructed from official tensors, NOT degrade_fn)
  C. TFD_NO_DIRECT_BLUR (info=sqrt(alpha_bar) flat, noise=official)
  D. TFD_REFERENCE_NOISE (info=official info_weights, noise=sqrt(1-alpha_bar))
"""
import os, sys, json, csv, time, warnings
import numpy as np
import torch
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

BASE_DIR = REPO_ROOT
os.chdir(f'{BASE_DIR}/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import WiFiDataset, Collator

RESULTS_DIR = f'{BASE_DIR}/experiments/final_patch'
ARTIFACTS_DIR = f'{BASE_DIR}/artifacts/final_patch'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
TIMESTEPS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

device = torch.device('cpu')
print(f'Device: {device}')

# ============================================================
# LOAD OFFICIAL TENSORS & DATA
# ============================================================
params = all_params[0]
diffusion = SignalDiffusion(params)

# Extract official tensors (all on CPU, float32)
alpha_bar = diffusion.alpha_bar                           # [T]
info_weights = diffusion.info_weights                      # [T, N]
noise_weights = diffusion.noise_weights                    # [T, N]
gaussian_kernel_bar = diffusion.gaussian_kernel_bar        # [T, N]
input_dim = params.sample_rate  # 512

# Standard DDPM weights
alpha_bar_1d = alpha_bar                                  # [T]
info_ddpm = torch.sqrt(alpha_bar_1d)                      # [T] scalar
noise_ddpm = torch.sqrt(1 - alpha_bar_1d)                 # [T] scalar

print('Official tensors loaded.')
print(f'  alpha_bar[99]: {alpha_bar[99].item():.6f}')
print(f'  info_weights shape: {info_weights.shape}')
print(f'  noise_weights shape: {noise_weights.shape}')

# Load data
params_cp = all_params[0]
params_cp.data_dir = ['./wifi/cond']
collator = Collator(params_cp)
dataset = WiFiDataset(params_cp.data_dir)
N_data = len(dataset)
print(f'Loaded {N_data} WiFi samples')

all_samples = []
for i in range(N_data):
    raw = dataset[i]
    batch = collator.collate([raw])
    all_samples.append({
        'data': batch['data'],
        'cond': batch['cond'],
        'idx': i,
    })

# ============================================================
# MANUAL OFFICIAL FORWARD (reconstructs degrade_fn logic)
# ============================================================
def manual_official_forward(x_0, t, eps, task_id=0):
    """Reconstruct official degrade_fn using pre-extracted tensors and external epsilon.

    This is mathematically identical to SignalDiffusion.degrade_fn(x_0, t, task_id)
    EXCEPT that it uses the provided epsilon instead of generating its own.
    degrade_fn internals:
      noise = noise_weights[t] * randn_like(x_0)  [seed=11]
      x_t = info_weights[t] * x_0 + noise
    """
    t_idx = t.item() if isinstance(t, torch.Tensor) else t

    if task_id in [0, 1]:
        nw = noise_weights[t_idx, :].unsqueeze(-1).unsqueeze(-1)
        iw = info_weights[t_idx, :].unsqueeze(-1).unsqueeze(-1)
    else:
        nw = noise_weights[t_idx, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        iw = info_weights[t_idx, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    noise = nw * eps
    x_t = iw * x_0 + noise
    return x_t


# ============================================================
# P11: REGRESSION CHECK — manual vs official degrade_fn
# ============================================================
print('\n' + '='*60)
print('REGRESSION CHECK: manual_official_forward vs official degrade_fn')
print('='*60)

regression_results = []

for sample_idx in range(N_data):
    data_0 = all_samples[sample_idx]['data']

    for t_step in TIMESTEPS:
        t_tensor = torch.tensor([t_step], dtype=torch.int64)

        # Call official degrade_fn (uses seed=11 internally)
        x_t_official = diffusion.degrade_fn(data_0, t_tensor, task_id=0)

        # Capture the epsilon degrade_fn used
        torch.manual_seed(11)
        eps = torch.randn_like(data_0)

        # Manual reconstruction with same epsilon
        x_t_manual = manual_official_forward(data_0, t_tensor, eps, task_id=0)

        # Compare
        diff = (x_t_official - x_t_manual).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        rel_l2 = float(torch.norm(x_t_official - x_t_manual) /
                       (torch.norm(x_t_official) + 1e-30))
        corr_off_man = float(np.corrcoef(
            x_t_official.numpy().ravel(), x_t_manual.numpy().ravel())[0, 1])

        regression_results.append({
            'sample_idx': sample_idx,
            'timestep': t_step,
            'max_abs_diff': max_abs,
            'mean_abs_diff': mean_abs,
            'relative_l2_error': rel_l2,
            'pearson_corr': corr_off_man,
        })

# Save regression
csv_path = f'{RESULTS_DIR}/official_forward_regression.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=regression_results[0].keys())
    writer.writeheader()
    writer.writerows(regression_results)
print(f'Saved: {csv_path}')

# Regression summary
max_all = max(r['max_abs_diff'] for r in regression_results)
mean_all = np.mean([r['mean_abs_diff'] for r in regression_results])
worst_rel_l2 = max(r['relative_l2_error'] for r in regression_results)
worst_corr = min(r['pearson_corr'] for r in regression_results)
print(f'  max_abs_diff across all: {max_all:.2e}')
print(f'  mean_abs_diff: {mean_all:.2e}')
print(f'  worst relative_l2: {worst_rel_l2:.2e}')
print(f'  worst pearson_corr: {worst_corr:.6f}')

REGRESSION_PASS = max_all < 1e-5
if REGRESSION_PASS:
    print(f'  REGRESSION: PASS (max_abs_diff {max_all:.2e} < 1e-5)')
else:
    print(f'  REGRESSION: FAIL (max_abs_diff {max_all:.2e} >= 1e-5)')
    print(f'  WARNING: manual reconstruction does not exactly match official degrade_fn')
    print(f'  Check broadcast shapes, dtype, task_id, seed handling.')
    print(f'  ABORTING — fix regression before running counterfactual.')
    sys.exit(1)


# ============================================================
# P7-P8: COMMON-EPSILON 4-PROTOCOL COMPARISON
# ============================================================
print('\n' + '='*60)
print('COMMON-EPSILON 4-PROTOCOL FORWARD COMPARISON')
print('='*60)

all_results = []

for seed_idx, seed in enumerate(SEEDS):
    print(f'\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...')

    for sample_idx in range(N_data):
        data_0 = all_samples[sample_idx]['data']

        for t_step in TIMESTEPS:
            t_tensor = torch.tensor([t_step])

            # Generate ONE epsilon for all 4 protocols
            # Deterministic unique seed per (seed, sample, timestep)
            eps_seed = seed * 10000 + sample_idx * 100 + t_step
            torch.manual_seed(eps_seed)
            eps = torch.randn_like(data_0)

            # ---- A: STANDARD_DDPM_REFERENCE ----
            # Scalar weights broadcast over all dimensions
            iw_a = info_ddpm[t_step].view(1, 1, 1, 1)
            nw_a = noise_ddpm[t_step].view(1, 1, 1, 1)
            x_t_a = iw_a * data_0 + nw_a * eps

            # ---- B: OFFICIAL_TFD_MANUAL ----
            x_t_b = manual_official_forward(data_0, t_tensor, eps, task_id=0)

            # ---- C: TFD_NO_DIRECT_BLUR ----
            # info = sqrt(alpha_bar[t]) flat scalar (kernel removed)
            # noise = official per-dimension noise_weights
            iw_c = info_ddpm[t_step].view(1, 1, 1, 1)
            nw_c = noise_weights[t_step, :].unsqueeze(-1).unsqueeze(-1)
            x_t_c = iw_c * data_0 + nw_c * eps

            # ---- D: TFD_REFERENCE_NOISE ----
            # info = official per-dimension info_weights (kernel preserved)
            # noise = sqrt(1-alpha_bar[t]) flat scalar (standard DDPM noise)
            iw_d = info_weights[t_step, :].unsqueeze(-1).unsqueeze(-1)
            nw_d = noise_ddpm[t_step].view(1, 1, 1, 1)
            x_t_d = iw_d * data_0 + nw_d * eps

            # Compute metrics for each protocol
            for proto_name, x_t in [
                ('STANDARD_DDPM', x_t_a),
                ('OFFICIAL_TFD', x_t_b),
                ('TFD_NO_DIRECT_BLUR', x_t_c),
                ('TFD_REFERENCE_NOISE', x_t_d),
            ]:
                p = x_t.detach().numpy().ravel()
                t_arr = data_0.detach().numpy().ravel()
                corr = float(np.corrcoef(p, t_arr)[0, 1]) if len(p) > 1 else 0.0
                nmse = float(np.mean((p - t_arr)**2) / (np.mean(t_arr**2) + 1e-30))
                energy_ratio = float(np.sum(p**2) / (np.sum(t_arr**2) + 1e-30))

                # Complex-magnitude distance (NOT spectral distance)
                xt_np = x_t.detach().numpy()
                d0_np = data_0.detach().numpy()
                p_mag = np.sqrt(xt_np[..., 0]**2 + xt_np[..., 1]**2).ravel()
                t_mag = np.sqrt(d0_np[..., 0]**2 + d0_np[..., 1]**2).ravel()
                cmag_dist = float(np.linalg.norm(p_mag - t_mag) /
                                  (np.linalg.norm(t_mag) + 1e-30))

                all_results.append({
                    'seed': seed,
                    'sample_idx': sample_idx,
                    'timestep': t_step,
                    'protocol': proto_name,
                    'alpha_bar': float(alpha_bar[t_step].item()),
                    'pearson_corr': corr,
                    'nmse': nmse,
                    'energy_ratio': energy_ratio,
                    'complex_magnitude_distance': cmag_dist,
                })

# Save raw CSV
csv_path = f'{RESULTS_DIR}/common_epsilon_forward_raw.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)
print(f'\nSaved: {csv_path} ({len(all_results)} rows)')

# Aggregate by protocol × timestep
protocols = sorted(set(r['protocol'] for r in all_results))
agg = []
for proto in protocols:
    for ts in TIMESTEPS:
        tsr = [r for r in all_results if r['protocol'] == proto and r['timestep'] == ts]
        corrs = [r['pearson_corr'] for r in tsr]
        nmses = [r['nmse'] for r in tsr]
        cmags = [r['complex_magnitude_distance'] for r in tsr]
        energies = [r['energy_ratio'] for r in tsr]
        agg.append({
            'protocol': proto, 'timestep': ts, 'N': len(tsr),
            'corr_mean': float(np.mean(corrs)), 'corr_std': float(np.std(corrs)),
            'nmse_mean': float(np.mean(nmses)), 'nmse_std': float(np.std(nmses)),
            'cmag_dist_mean': float(np.mean(cmags)), 'cmag_dist_std': float(np.std(cmags)),
            'energy_ratio_mean': float(np.mean(energies)),
        })

csv_path = f'{RESULTS_DIR}/common_epsilon_forward_summary.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=agg[0].keys())
    writer.writeheader()
    writer.writerows(agg)
print(f'Saved: {csv_path}')

# Terminal summary
print('\n' + '='*60)
print('t=99 TERMINAL SUMMARY')
print('='*60)
for proto in protocols:
    a = [r for r in agg if r['protocol'] == proto and r['timestep'] == 99][0]
    print(f'  {proto:25s}: corr={a["corr_mean"]:.4f}±{a["corr_std"]:.4f}, '
          f'NMSE={a["nmse_mean"]:.4f}, cmag_dist={a["cmag_dist_mean"]:.4f}')

# Causal decomposition at t=99
off = [r for r in agg if r['protocol'] == 'OFFICIAL_TFD' and r['timestep'] == 99][0]
ddpm = [r for r in agg if r['protocol'] == 'STANDARD_DDPM' and r['timestep'] == 99][0]
nodb = [r for r in agg if r['protocol'] == 'TFD_NO_DIRECT_BLUR' and r['timestep'] == 99][0]
refn = [r for r in agg if r['protocol'] == 'TFD_REFERENCE_NOISE' and r['timestep'] == 99][0]

print('\n  CAUSAL DECOMPOSITION:')
direct_blur_effect = nodb['corr_mean'] - off['corr_mean']
noise_weight_effect = refn['corr_mean'] - off['corr_mean']
ddpm_gap = ddpm['corr_mean'] - off['corr_mean']
print(f'    Direct blur effect   (TFD_NO_DIRECT_BLUR - OFFICIAL_TFD): {direct_blur_effect:+.4f}')
print(f'    Noise-weight effect  (TFD_REFERENCE_NOISE - OFFICIAL_TFD): {noise_weight_effect:+.4f}')
print(f'    DDPM reference gap   (STANDARD_DDPM - OFFICIAL_TFD):      {ddpm_gap:+.4f}')

print(f'\n    Direct blur:    {"NEGLIGIBLE" if abs(direct_blur_effect) < 0.01 else "MATERIAL"}')
print(f'    Noise-weight:   {"MATERIAL" if abs(noise_weight_effect) > 0.01 else "NEGLIGIBLE"}')
print(f'    Regression:     {"PASS" if REGRESSION_PASS else "FAIL"} (max_abs_diff={max_all:.2e})')

print('\nDone.')
