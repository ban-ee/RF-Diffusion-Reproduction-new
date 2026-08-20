"""
Fig 5 data preparation: extract a REAL WiFi CSI sample x_0 and its four
terminal states at t=99 for the terminal-state visualization figure.

This is NOT a new experiment — it loads the already-saved WiFi dataset and the
official SignalDiffusion schedule (no model weights, no training, no checkpoint
changes), and applies the deterministic forward functions already audited in
phase 3/4. It saves raw magnitudes as .npy for local plotting.

Four panels (all from the SAME x_0 and, where noise is involved, the SAME eps):
  1. Original        x_0
  2. Noise-only      sqrt(alpha_bar[99]) * x_0 + sqrt(1 - alpha_bar[99]) * eps   (STANDARD DDPM)
  3. Blur-only       gaussian_kernel_bar[99] * x_0                               (deterministic blur, no noise)
  4. Noise+Blur      info_weights[99] * x_0 + noise_weights[99] * eps             (official TFD)
"""
import os, sys, json, warnings
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

OUT_DIR = f'{BASE_DIR}/experiments/fig5_terminal'
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_IDX = 0
T = 99
EPS_SEED = 42

device = torch.device('cpu')

params = all_params[0]
diffusion = SignalDiffusion(params)

alpha_bar = diffusion.alpha_bar
info_weights = diffusion.info_weights
noise_weights = diffusion.noise_weights
gaussian_kernel_bar = diffusion.gaussian_kernel_bar
input_dim = params.sample_rate  # 512

# Load real data
params_cp = all_params[0]
params_cp.data_dir = ['./wifi/cond']
collator = Collator(params_cp)
dataset = WiFiDataset(params_cp.data_dir)
print(f'Loaded {len(dataset)} WiFi samples')

raw = dataset[SAMPLE_IDX]
batch = collator.collate([raw])
x0 = batch['data'].float()   # [1, 512, 90, 2] complex CSI (real/imag)

# Fixed epsilon shared by both noisy protocols
torch.manual_seed(EPS_SEED)
eps = torch.randn_like(x0)

# --- terminal states (all t=99) ---
# 1. original
x_original = x0.clone()

# 2. noise-only  (STANDARD DDPM scalar weights)
a = torch.sqrt(alpha_bar[T]).item()
x_noise_only = float(a) * x0 + float(np.sqrt(1 - a ** 2)) * eps

# 3. blur-only (deterministic, no noise)
gw = gaussian_kernel_bar[T, :].unsqueeze(-1).unsqueeze(-1)  # [512, 1, 1]
x_blur_only = gw * x0

# 4. noise+blur (official TFD)
nw = noise_weights[T, :].unsqueeze(-1).unsqueeze(-1)
iw = info_weights[T, :].unsqueeze(-1).unsqueeze(-1)
x_noise_blur = iw * x0 + nw * eps


def magnitude(x):
    # x: [1, 512, 90, 2] -> sqrt(re^2 + im^2) -> [512, 90]
    return torch.sqrt(x[0, ..., 0] ** 2 + x[0, ..., 1] ** 2).numpy().astype(np.float32)


mags = {
    'original': magnitude(x_original),
    'noise_only': magnitude(x_noise_only),
    'blur_only': magnitude(x_blur_only),
    'noise_blur': magnitude(x_noise_blur),
}

# Save magnitudes
np.savez_compressed(f'{OUT_DIR}/terminal_magnitudes.npz', **mags)

# Also save raw complex (real/imag) for completeness
np.savez_compressed(
    f'{OUT_DIR}/terminal_signals.npz',
    original=x_original.numpy().astype(np.float32),
    noise_only=x_noise_only.numpy().astype(np.float32),
    blur_only=x_blur_only.numpy().astype(np.float32),
    noise_blur=x_noise_blur.numpy().astype(np.float32),
)

# Energy + correlation diagnostics (for the manifest)
def energy(x):
    return float((x ** 2).sum())


def corr(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


meta = {
    'sample_idx': SAMPLE_IDX,
    'timestep': T,
    'eps_seed': EPS_SEED,
    'input_dim': input_dim,
    'shape': list(x0.shape),
    'alpha_bar_99': float(alpha_bar[T]),
    'sqrt_alpha_bar_99': float(np.sqrt(alpha_bar[T])),
    'sqrt_1_minus_alpha_bar_99': float(np.sqrt(1 - alpha_bar[T])),
    'info_weight_99_mean': float(info_weights[T].mean()),
    'noise_weight_99_mean': float(noise_weights[T].mean()),
    'noise_weight_vs_ddpm_ratio': float(noise_weights[T].mean() / np.sqrt(1 - alpha_bar[T])),
    'gaussian_kernel_bar_deviation_from_1': float((gaussian_kernel_bar[T] - 1).abs().max()),
    'energies': {
        'original': energy(x_original),
        'noise_only': energy(x_noise_only),
        'blur_only': energy(x_blur_only),
        'noise_blur': energy(x_noise_blur),
    },
    'corr_with_x0': {
        'original': corr(x_original, x_original),
        'noise_only': corr(x_noise_only, x_original),
        'blur_only': corr(x_blur_only, x_original),
        'noise_blur': corr(x_noise_blur, x_original),
    },
    'mag_stats': {
        k: {'min': float(v.min()), 'max': float(v.max()), 'mean': float(v.mean()),
            'p99': float(np.percentile(v, 99))}
        for k, v in mags.items()
    },
}

with open(f'{OUT_DIR}/terminal_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print(json.dumps(meta, indent=2))
print(f'\nSaved to {OUT_DIR}/terminal_magnitudes.npz, terminal_signals.npz, terminal_meta.json')
