"""
TASK TWO: Official TFD Core Algorithm Analysis
===============================================

This script ONLY calls official RF-Diffusion source code.
It does NOT re-implement any algorithm.
It reads official tensors, records intermediate states, and generates diagnostics.

Official source used:
  - tfdiff/params.py → all_params
  - tfdiff/diffusion.py → SignalDiffusion, GaussianDiffusion
  - tfdiff/dataset.py → from_path_inference, _nested_map

Purpose:
  A. Understand the TFD forward degradation mechanism
  B. Extract and visualize intermediate states
  C. Verify information/noise weight properties
  D. Compare SignalDiffusion vs GaussianDiffusion
"""
import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT + '/official')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from tfdiff.params import all_params, AttrDict
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion

OUTPUT_DIR = REPO_ROOT + '/artifacts/tfd_core'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def analyze_task(task_id, name):
    """Analyze TFD for one task using official code only."""
    params = all_params[task_id]
    print(f"\n{'='*60}")
    print(f"TFD Analysis: {name} (task_id={task_id})")
    print(f"{'='*60}")

    # ============================================================
    # 1. Initialize official SignalDiffusion
    # ============================================================
    diffusion = SignalDiffusion(params)
    T = params.max_step
    N = params.sample_rate
    print(f"\n  T (max_step): {T}")
    print(f"  N (sample_rate): {N}")
    print(f"  hidden_dim: {params.hidden_dim}")
    print(f"  num_block: {params.num_block}")

    # ============================================================
    # 2. Extract and record official schedule parameters
    # ============================================================
    alpha = diffusion.alpha.numpy()          # α_t [T]
    alpha_bar = diffusion.alpha_bar.numpy()   # ᾱ_t [T]
    beta = 1 - alpha                          # β_t [T]
    var_blur = diffusion.var_blur.numpy()     # blur variance [T]
    var_blur_bar = diffusion.var_blur_bar.numpy()  # cumulative blur var [T]

    np.savez(f'{OUTPUT_DIR}/{name}_schedules.npz',
             alpha=alpha, alpha_bar=alpha_bar, beta=beta,
             var_blur=var_blur, var_blur_bar=var_blur_bar)

    print(f"\n  Schedule ranges:")
    print(f"    β: [{beta[0]:.2e}, {beta[-1]:.2e}]")
    print(f"    ᾱ_T: {alpha_bar[-1]:.6f}")
    print(f"    blur σ² cumsum at T: {var_blur_bar[-1]:.6f}")

    # ============================================================
    # 3. Extract official kernel weights
    # ============================================================
    gaussian_kernel = diffusion.gaussian_kernel.numpy()       # G_t [T, N]
    gaussian_kernel_bar = diffusion.gaussian_kernel_bar.numpy() # Ḡ_t [T, N]
    info_weights = diffusion.info_weights.numpy()             # info [T, N]
    noise_weights = diffusion.noise_weights.numpy()           # noise [T, N]

    np.savez(f'{OUTPUT_DIR}/{name}_weights.npz',
             gaussian_kernel=gaussian_kernel,
             gaussian_kernel_bar=gaussian_kernel_bar,
             info_weights=info_weights,
             noise_weights=noise_weights)

    # Verify info_weights formula: Ḡ_t * √ᾱ_t
    expected_info = gaussian_kernel_bar * np.sqrt(alpha_bar)[:, np.newaxis]
    info_diff = np.abs(info_weights - expected_info).max()
    print(f"\n  info_weights = Ḡ_t * √ᾱ_t verification: max diff = {info_diff:.2e}")
    assert info_diff < 1e-6, "info_weights formula mismatch!"

    # ============================================================
    # 4. Analyze how information degrades over time
    # ============================================================
    # At each t, average info_weight and noise_weight
    # This shows how much original signal vs noise is in x_t
    timesteps = [0, 25, 50, 75, T-1]
    print(f"\n  {'t':>4s}  {'info_w (mean)':>14s}  {'noise_w (mean)':>14s}  {'info/noise ratio':>16s}")
    print(f"  {'-'*4}  {'-'*14}  {'-'*14}  {'-'*16}")
    for t in timesteps:
        iw = info_weights[t].mean()
        nw = noise_weights[t].mean()
        ratio = iw / nw if nw > 0 else float('inf')
        print(f"  {t:4d}  {iw:14.6f}  {nw:14.6f}  {ratio:16.4f}")

    # ============================================================
    # 5. Frequency-domain analysis of blur kernels
    # ============================================================
    # Compute FFT of gaussian_kernel_bar to see frequency-domain effect
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'TFD Frequency-Domain Analysis: {name}', fontsize=14)

    # Plot gaussian kernels in time domain
    for i, t in enumerate([0, 25, 50, 75, 99]):
        ax = axes[0, i % 3] if i < 3 else axes[1, i % 3]
        row_idx = 0 if i < 3 else 1
        col_idx = i % 3
        ax = axes[row_idx, col_idx]
        ax.plot(gaussian_kernel[t], alpha=0.5, label=f'G_{t}')
        ax.plot(gaussian_kernel_bar[t], alpha=0.8, label=f'Ḡ_{t}')
        ax.set_title(f't={t}')
        ax.legend(fontsize=8)
        ax.set_xlabel('Sample index')

    # Plot frequency-domain kernel magnitude
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(f'TFD Blur Kernel Frequency Response: {name}', fontsize=14)
    for i, t in enumerate([0, 25, 50, 75, 99]):
        row_idx = 0 if i < 3 else 1
        col_idx = i % 3
        ax = axes2[row_idx, col_idx]
        g_bar_freq = np.abs(np.fft.fft(gaussian_kernel_bar[t]))
        ax.plot(g_bar_freq[:N//2])
        ax.set_title(f't={t}: Ḡ freq response')
        ax.set_xlabel('Frequency bin')

    plt.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/{name}_kernels.png', dpi=150)
    fig2.savefig(f'{OUTPUT_DIR}/{name}_freq_response.png', dpi=150)
    plt.close('all')
    print(f"\n  Saved kernel visualization to {OUTPUT_DIR}/{name}_kernels.png")
    print(f"  Saved freq response to {OUTPUT_DIR}/{name}_freq_response.png")

    # ============================================================
    # 6. Compare SignalDiffusion vs GaussianDiffusion
    # ============================================================
    gaussian_diff = GaussianDiffusion(params)
    print(f"\n  GaussianDiffusion vs SignalDiffusion comparison:")
    print(f"  {'t':>4s}  {'Sig_info':>10s}  {'Gauss_info':>10s}  {'Sig_noise':>10s}  {'Gauss_noise':>10s}")
    for t in timesteps:
        si = info_weights[t].mean()
        sn = noise_weights[t].mean()
        gi = np.sqrt(alpha_bar[t])
        gn = np.sqrt(1 - alpha_bar[t])
        print(f"  {t:4d}  {si:10.6f}  {gi:10.6f}  {sn:10.6f}  {gn:10.6f}")

    return diffusion


def visualize_degradation(diffusion, task_id, name):
    """Use official degrade_fn to visualize x_t at different timesteps."""
    params = all_params[task_id]
    N = params.sample_rate

    # Create synthetic test signal matching official data shapes
    # Official data shapes from dataset.py Collator:
    #   WiFi/FMCW: [B, N, S, 2] where S = extra_dim[0]
    #   MIMO:      [B, N, S, A, 2] where extra_dim = [S, A]
    torch.manual_seed(42)
    if task_id in [0, 1]:
        S = params.extra_dim[0]
        x_0 = torch.randn(1, N, S, 2).float()
    else:
        extra_dim = params.extra_dim
        x_0 = torch.randn(1, N, extra_dim[0], extra_dim[1], 2).float()

    print(f"\n  Synthetic x_0 shape: {x_0.shape} (matches official data format)")

    # Apply official degrade_fn at different timesteps
    timesteps = [0, 25, 50, 75, 99]
    x_t_list = []
    for t_val in timesteps:
        t = torch.full((1,), t_val, dtype=torch.int64)
        x_t = diffusion.degrade_fn(x_0.clone(), t, task_id)
        x_t_list.append(x_t)

    # Visualize
    fig, axes = plt.subplots(3, len(timesteps), figsize=(20, 12))
    fig.suptitle(f'TFD Degradation Process: {name}', fontsize=14)

    for i, (t_val, x_t) in enumerate(zip(timesteps, x_t_list)):
        # Extract for visualization (official data shape: [B, N, S, 2])
        if task_id in [0, 1]:
            signal = x_t[0, :, 0, :].numpy()  # [N, 2]
        else:
            signal = x_t[0, :, 0, 0, :].numpy()  # [N, 2]

        # Time domain (I/Q)
        ax = axes[0, i]
        ax.plot(signal[:, 0], alpha=0.7, label='I')
        ax.plot(signal[:, 1], alpha=0.7, label='Q')
        ax.set_title(f't={t_val}')
        if i == 0:
            ax.set_ylabel('Amplitude')

        # Magnitude
        ax = axes[1, i]
        mag = np.sqrt(signal[:, 0]**2 + signal[:, 1]**2)
        ax.plot(mag)
        if i == 0:
            ax.set_ylabel('|x|')
        ax.set_xlabel('Sample')

        # FFT
        ax = axes[2, i]
        fft = np.abs(np.fft.fft(signal[:, 0] + 1j * signal[:, 1]))
        ax.plot(fft[:N//2])
        if i == 0:
            ax.set_ylabel('|FFT|')
        ax.set_xlabel('Freq bin')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{name}_degradation.png', dpi=150)
    plt.close()
    print(f"  Saved degradation visualization to {OUTPUT_DIR}/{name}_degradation.png")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK TWO: TFD Core Algorithm Analysis")
    print("Using ONLY official RF-Diffusion source code")
    print("=" * 60)

    for task_id, name in [(0, 'wifi'), (1, 'fmcw')]:
        diffusion = analyze_task(task_id, name)
        visualize_degradation(diffusion, task_id, name)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("TFD analysis complete.")
