"""
TASK TWO: Formal TFD Real-Data Visualization
=============================================

Uses ONLY official code:
  - tfdiff.dataset.from_path_inference()
  - tfdiff.diffusion.SignalDiffusion.degrade_fn()
  - tfdiff.params.all_params

Reads ONE real sample from official dataset.
Generates x_t at diagnostic timesteps.
Records: time-domain I/Q, magnitude, spectrum, x_t vs x_0 correlation.

Output: artifacts/tfd_real_data/
Status: FORMAL CORE ALGORITHM REPRODUCTION (REAL DATA)
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

# CRITICAL: Must run from official/ directory for relative paths to work
os.chdir(REPO_ROOT + '/official')

from tfdiff.params import all_params, AttrDict
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import from_path_inference, _nested_map

OUTPUT_DIR = REPO_ROOT + '/artifacts/tfd_real_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_real_data(task_id, name):
    """Use official dataset loader + official degrade_fn on real samples."""
    params = all_params[task_id]
    T = params.max_step
    N = params.sample_rate
    device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"Real-Data TFD Visualization: {name} (task_id={task_id})")
    print(f"{'='*60}")

    # ============================================================
    # 1. Load ONE real sample using official dataset loader
    # ============================================================
    print(f"\n  Loading real sample via official from_path_inference()...")
    dataset = from_path_inference(params)

    # Get first sample
    for features in dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        data = features['data']      # official shape from Collator
        cond = features['cond']      # official shape from Collator
        break  # only need one sample

    print(f"  Real data shape: {list(data.shape)}")
    print(f"  Real cond shape: {list(cond.shape)}")
    print(f"  Data dtype: {data.dtype}")
    print(f"  Data mean: {data.mean().item():.6f}, std: {data.std().item():.6f}")

    # ============================================================
    # 2. Initialize official SignalDiffusion
    # ============================================================
    diffusion = SignalDiffusion(params)

    # ============================================================
    # 3. Generate x_t at diagnostic timesteps
    # ============================================================
    timesteps = [0, T//4, T//2, 3*T//4, T-1]
    timestep_labels = ['t=0', f't=T/4={T//4}', f't=T/2={T//2}', f't=3T/4={3*T//4}', f't=T-1={T-1}']

    x_t_results = {}
    for t_val, label in zip(timesteps, timestep_labels):
        t = torch.full((data.shape[0],), t_val, dtype=torch.int64)
        x_t = diffusion.degrade_fn(data.clone(), t, task_id)
        x_t_results[t_val] = x_t
        print(f"  {label}: x_t mean={x_t.mean().item():.6f}, std={x_t.std().item():.6f}")

    # ============================================================
    # 4. Correlation analysis: x_T vs x_0
    # ============================================================
    x_0 = x_t_results[0]
    x_T = x_t_results[T-1]

    # Flatten for correlation computation
    x_0_flat = x_0.reshape(-1)
    x_T_flat = x_T.reshape(-1)

    correlation = torch.corrcoef(torch.stack([x_0_flat, x_T_flat]))[0, 1].item()
    print(f"\n  x_0 vs x_T Pearson correlation: {correlation:.6f}")
    print(f"  NOTE: This is a linear correlation of the degraded signal with")
    print(f"  the original under the released configuration.")
    print(f"  It is NOT equivalent to an information-theoretic measure of")
    print(f"  'how much information is preserved'.")

    # Also compute per-frequency-bin correlation
    if task_id in [0, 1]:
        # x shape: [B, N, S, 2], compute correlation across spatial dim
        for freq_bin in [0, N//4, N//2]:
            x0_bin = x_0[0, freq_bin, :, :].reshape(-1)
            xT_bin = x_T[0, freq_bin, :, :].reshape(-1)
            corr_bin = torch.corrcoef(torch.stack([x0_bin, xT_bin]))[0, 1].item()
            print(f"    x_0 vs x_T correlation at sample index {freq_bin}: {corr_bin:.6f}")

    # ============================================================
    # 5. Generate visualizations with real data
    # ============================================================
    fig, axes = plt.subplots(3, len(timesteps), figsize=(20, 12))
    fig.suptitle(f'TFD Degradation Process on Real Data: {name}\n(DIAGNOSTIC TIMESTEPS — NOT PAPER PARAMETERS)', fontsize=14)

    for i, (t_val, label) in enumerate(zip(timesteps, timestep_labels)):
        x_t = x_t_results[t_val]

        if task_id in [0, 1]:
            # Shape: [B, N, S, 2], extract first spatial channel
            signal = x_t[0, :, 0, :].numpy()  # [N, 2]
        elif task_id == 2:
            # Shape: [B, N, S, A, 2], extract first spatial channel
            signal = x_t[0, :, 0, 0, :].numpy()

        # Row 0: Time-domain I/Q
        ax = axes[0, i]
        ax.plot(signal[:, 0], alpha=0.7, label='I', linewidth=0.5)
        ax.plot(signal[:, 1], alpha=0.7, label='Q', linewidth=0.5)
        ax.set_title(label, fontsize=10)
        if i == 0:
            ax.set_ylabel('Amplitude (I/Q)')
            ax.legend(fontsize=7)

        # Row 1: Magnitude
        ax = axes[1, i]
        mag = np.sqrt(signal[:, 0]**2 + signal[:, 1]**2)
        ax.plot(mag, linewidth=0.5)
        if i == 0:
            ax.set_ylabel('|x| (Magnitude)')
        ax.set_xlabel('Sample index')

        # Row 2: FFT magnitude spectrum
        ax = axes[2, i]
        complex_signal = signal[:, 0] + 1j * signal[:, 1]
        fft = np.abs(np.fft.fft(complex_signal))
        ax.plot(fft[:N//2], linewidth=0.5)
        if i == 0:
            ax.set_ylabel('|FFT|')
        ax.set_xlabel('Frequency bin')
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{name}_real_degradation.png', dpi=150)
    plt.close()
    print(f"\n  Saved: {OUTPUT_DIR}/{name}_real_degradation.png")

    # ============================================================
    # 6. Save numerical data for reproducibility
    # ============================================================
    np.savez(f'{OUTPUT_DIR}/{name}_real_x_t.npz',
             x_0=x_t_results[0].numpy(),
             x_T4=x_t_results[T//4].numpy(),
             x_T2=x_t_results[T//2].numpy(),
             x_3T4=x_t_results[3*T//4].numpy(),
             x_T=x_t_results[T-1].numpy(),
             timesteps=np.array(timesteps),
             correlation_x0_xT=correlation)
    print(f"  Saved: {OUTPUT_DIR}/{name}_real_x_t.npz")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK TWO: Formal TFD Real-Data Visualization")
    print("Using OFFICIAL dataset loader + OFFICIAL degrade_fn")
    print("=" * 60)

    for task_id, name in [(0, 'wifi'), (1, 'fmcw')]:
        try:
            visualize_real_data(task_id, name)
        except Exception as e:
            print(f"\n  ERROR for {name}: {e}")
            import traceback
            traceback.print_exc()

    # MIMO (task_id=2) requires different data handling, skip for now
    print(f"\n  MIMO (task_id=2): DEFERRED — requires separate data path handling")

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("STATUS: FORMAL CORE ALGORITHM REPRODUCTION (REAL OFFICIAL DATA)")
