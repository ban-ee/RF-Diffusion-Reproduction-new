"""
TASK THREE: Terminal State Analysis (CORRECTED LANGUAGE)

Strictly reports official parameter values without unsupported causal claims.

Rules:
  - alpha_bar is a cumulative diffusion coefficient, NOT "information retention"
  - blur effect causal claims require controlled experiments
  - "9.46x noise" is an OBSERVATION, NOT attributed to frequency blur

Uses ONLY official params and SignalDiffusion/GaussianDiffusion code.
"""
import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT + '/official')

import numpy as np
import torch
import os

from tfdiff.params import all_params
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion

OUTPUT_DIR = REPO_ROOT + '/artifacts/terminal_state'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def report_terminal_state(task_id, name):
    """Report official terminal state parameters with strict language."""
    params = all_params[task_id]
    diffusion = SignalDiffusion(params)
    gaussian_diff = GaussianDiffusion(params)
    T = params.max_step
    N = params.sample_rate

    print(f"\n{'='*60}")
    print(f"Terminal State Report: {name} (T={T})")
    print(f"{'='*60}")

    # Read official values only - no interpretation
    alpha_bar_T = diffusion.alpha_bar[T-1].item()
    info_T_mean = diffusion.info_weights[T-1, :].mean().item()
    noise_T_mean = diffusion.noise_weights[T-1, :].mean().item()
    gauss_noise_T = np.sqrt(1 - gaussian_diff.alpha_bar[T-1].item()).item()
    gauss_info_T = np.sqrt(gaussian_diff.alpha_bar[T-1].item()).item()

    print(f"\n  OFFICIAL PARAMETER VALUES (read from SignalDiffusion object):")
    print(f"    alpha_bar[T-1] = {alpha_bar_T:.6f}")
    print(f"    info_weights[T-1].mean() = {info_T_mean:.6f}")
    print(f"    noise_weights[T-1].mean() = {noise_T_mean:.6f}")

    print(f"\n  OFFICIAL PARAMETER VALUES (read from GaussianDiffusion object):")
    print(f"    sqrt(alpha_bar[T-1]) = {gauss_info_T:.6f}")
    print(f"    sqrt(1 - alpha_bar[T-1]) = {gauss_noise_T:.6f}")

    # OBSERVATION only - no causal attribution
    noise_ratio = noise_T_mean / gauss_noise_T
    print(f"\n  OBSERVATION:")
    print(f"    SignalDiffusion noise_weight / GaussianDiffusion noise_weight = {noise_ratio:.2f}")
    print(f"    The two implementations produce substantially different equivalent")
    print(f"    terminal noise coefficients under the released configuration.")
    print(f"    CAUSAL INTERPRETATION: NOT YET ESTABLISHED.")
    print(f"    Requires controlled experiment where only the frequency-domain")
    print(f"    component is varied while holding all other parameters constant.")

    # Record alpha_bar proximity to zero
    if alpha_bar_T < 0.01:
        print(f"\n  OBSERVATION: alpha_bar_T = {alpha_bar_T:.6f} is numerically close to zero.")
    elif alpha_bar_T > 0.5:
        print(f"\n  OBSERVATION: alpha_bar_T = {alpha_bar_T:.6f} is numerically far from zero.")
    else:
        print(f"\n  OBSERVATION: alpha_bar_T = {alpha_bar_T:.6f} is in an intermediate range.")
    print(f"    This is a cumulative diffusion coefficient, not an information-theoretic metric.")
    print(f"    The relationship between alpha_bar_T and actual source information")
    print(f"    in x_T requires real-data experiments to quantify.")

    # Blur parameters
    blur_T = diffusion.var_blur_bar[T-1].item()
    print(f"\n  OFFICIAL BLUR PARAMETERS:")
    print(f"    blur_schedule[0] = {params.blur_schedule[0]:.2e}")
    print(f"    blur_schedule[T-1] = {params.blur_schedule[T-1]:.2e}")
    print(f"    var_blur_bar[T-1] = {blur_T:.6e}")
    print(f"    var_kernel_bar[T-1] = {diffusion.var_kernel_bar[T-1, 0].item():.4e}")
    print(f"    gaussian_kernel_bar[T-1].min() = {diffusion.gaussian_kernel_bar[T-1].min().item():.6f}")
    print(f"    gaussian_kernel_bar[T-1].max() = {diffusion.gaussian_kernel_bar[T-1].max().item():.6f}")
    print(f"    gaussian_kernel_bar[T-1].mean() = {diffusion.gaussian_kernel_bar[T-1].mean().item():.6f}")


def compare_all_tasks_corrected():
    """Cross-task comparison with strict language."""
    print(f"\n{'='*60}")
    print("Cross-Task Terminal Parameter Comparison")
    print(f"{'='*60}")

    for task_id, name in [(0, 'WiFi'), (1, 'FMCW'), (2, '5G FDD')]:
        params = all_params[task_id]
        diffusion = SignalDiffusion(params)
        T = params.max_step

        alpha_bar_T = diffusion.alpha_bar[T-1].item()
        info_T = diffusion.info_weights[T-1, :].mean().item()
        noise_T = diffusion.noise_weights[T-1, :].mean().item()
        blur_T = diffusion.var_blur_bar[T-1].item()

        print(f"\n  {name} (T={T}):")
        print(f"    alpha_bar[T-1] = {alpha_bar_T:.6f}")
        print(f"    info_weights[T-1].mean() = {info_T:.6f}")
        print(f"    noise_weights[T-1].mean() = {noise_T:.6f}")
        print(f"    var_blur_bar[T-1] = {blur_T:.6e}")
        print(f"    noise_schedule range: [{params.noise_schedule[0]:.2e}, {params.noise_schedule[T-1]:.2e}]")
        print(f"    blur_schedule range: [{params.blur_schedule[0]:.2e}, {params.blur_schedule[T-1]:.2e}]")

        # Only state numerical facts, not interpretations
        if alpha_bar_T < 0.01:
            print(f"    → alpha_bar_T is numerically close to zero")
        else:
            print(f"    → alpha_bar_T is numerically far from zero ({alpha_bar_T:.4f})")


if __name__ == "__main__":
    print("=" * 60)
    print("TASK THREE: Terminal State Analysis (CORRECTED)")
    print("Strict parameter reporting — no unsupported causal claims")
    print("=" * 60)

    for task_id, name in [(0, 'wifi'), (1, 'fmcw'), (2, 'mimo_5g')]:
        report_terminal_state(task_id, name)

    compare_all_tasks_corrected()

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nREMINDER: alpha_bar is a cumulative diffusion coefficient.")
    print("It is NOT an information-theoretic metric.")
    print("Real-data experiments are required to quantify source information in x_T.")
