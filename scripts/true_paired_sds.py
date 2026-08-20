"""
TRUE PER-OUTPUT PAIRED SDS — RF-Diffusion WiFi

For each exact generated output y(i,j) = native_sampling(model, x_j, c_i):
  r_own(i,j)   = Corr(y(i,j), x_j)
  r_unrel(i,j,k) = Corr(y(i,j), x_k)  for k != j
  delta(i,j,k) = r_own(i,j) - r_unrel(i,j,k)

Paired SDS = mean(delta)

This is a strict per-output computation. Each delta comes from the SAME
generated output compared to its own terminal source and an unrelated source.
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
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import WiFiDataset, Collator

RESULTS_DIR = f'{BASE_DIR}/experiments/final_patch'
ARTIFACTS_DIR = f'{BASE_DIR}/artifacts/final_patch'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456]
N_TERMINAL_SOURCES = 5  # at least 5 terminal sources per condition
N_UNRELATED = 5         # at least 5 unrelated sources per output
TIMESTEPS_FORWARD = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
print(f'CUDA available: {torch.cuda.is_available()}')

# ============================================================
# MODEL & DATA LOADING
# ============================================================
def load_model(ckpt_path):
    params = all_params[0]
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
            'data': batch['data'],
            'cond': batch['cond'],
            'idx': i,
        })
    return samples

def pearson_corr(x, y):
    """Compute Pearson correlation between two flattened tensors."""
    xf = x.detach().cpu().numpy().ravel()
    yf = y.detach().cpu().numpy().ravel()
    return float(np.corrcoef(xf, yf)[0, 1])

# ============================================================
# TRUE PAIRED SDS
# ============================================================
def run_true_paired_sds(model, diffusion, samples, checkpoint_tag):
    """For each seed × condition × terminal_source:
    Generate ONE output y(i,j), then compute r_own and r_unrelated on that exact output.
    """
    print(f'\n{"="*60}')
    print(f'TRUE PAIRED SDS: {checkpoint_tag}')
    print(f'{"="*60}')

    N = len(samples)
    all_deltas = []
    r_own_all = []
    r_unrel_all = []
    n_outputs = 0

    for seed_idx, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f'\n  Seed {seed} ({seed_idx+1}/{len(SEEDS)})...')

        for cond_idx in range(N):
            cond_i = samples[cond_idx]['cond'].to(device)

            # Select terminal sources (j): always include cond_idx, plus random others
            other_indices = [j for j in range(N) if j != cond_idx]
            n_terminal = min(N_TERMINAL_SOURCES - 1, len(other_indices))
            terminal_src_indices = [cond_idx] + list(
                np.random.choice(other_indices, size=n_terminal, replace=False))

            # Generate one output per terminal source
            for src_j in terminal_src_indices:
                data_j = samples[src_j]['data'].to(device)

                with torch.no_grad():
                    y_ij = diffusion.native_sampling(model, data_j, cond_i, device)
                y_ij_cpu = y_ij.cpu()
                n_outputs += 1

                # r_own: Corr(y(i,j), x_j)
                r_own_val = pearson_corr(y_ij_cpu, samples[src_j]['data'])
                r_own_all.append(r_own_val)

                # r_unrelated: Corr(y(i,j), x_k) for k != j
                unrelated_candidates = [k for k in range(N) if k != src_j]
                n_unrel = min(N_UNRELATED, len(unrelated_candidates))
                unrel_indices = list(np.random.choice(
                    unrelated_candidates, size=n_unrel, replace=False))

                for k in unrel_indices:
                    r_unrel_val = pearson_corr(y_ij_cpu, samples[k]['data'])
                    r_unrel_all.append(r_unrel_val)

                    delta = r_own_val - r_unrel_val
                    all_deltas.append({
                        'checkpoint': checkpoint_tag,
                        'seed': seed,
                        'condition_idx': cond_idx,
                        'terminal_source_idx': src_j,
                        'unrelated_source_idx': k,
                        'r_own': r_own_val,
                        'r_unrelated': r_unrel_val,
                        'delta': delta,
                    })
            # Progress reporting
            if (cond_idx + 1) % 10 == 0:
                print(f'    condition {cond_idx+1}/{N} done, '
                      f'{n_outputs} outputs, {len(all_deltas)} deltas')

    # Summary statistics
    deltas_arr = np.array([d['delta'] for d in all_deltas])
    r_own_arr = np.array(r_own_all)
    r_unrel_arr = np.array(r_unrel_all)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    n_bs = 2000
    n_d = len(deltas_arr)
    bs_means = np.array([np.mean(deltas_arr[rng.randint(0, n_d, n_d)]) for _ in range(n_bs)])
    ci_lo = float(np.percentile(bs_means, 2.5))
    ci_hi = float(np.percentile(bs_means, 97.5))

    summary = {
        'checkpoint': checkpoint_tag,
        'N_outputs': n_outputs,
        'N_pairs': len(all_deltas),
        'mean_delta': float(np.mean(deltas_arr)),
        'std_delta': float(np.std(deltas_arr)),
        'median_delta': float(np.median(deltas_arr)),
        'ci95_low': ci_lo,
        'ci95_high': ci_hi,
        'mean_r_own': float(np.mean(r_own_arr)),
        'mean_r_unrelated': float(np.mean(r_unrel_arr)),
        'std_r_own': float(np.std(r_own_arr)),
        'std_r_unrelated': float(np.std(r_unrel_arr)),
    }

    print(f'\n  === {checkpoint_tag} SUMMARY ===')
    for k, v in summary.items():
        if isinstance(v, float):
            print(f'    {k}: {v:.4f}')
        else:
            print(f'    {k}: {v}')

    return all_deltas, summary


# ============================================================
# MAIN
# ============================================================
print('='*60)
print('TRUE PER-OUTPUT PAIRED SDS')
print('='*60)

# Load data once
samples = load_all_samples()
print(f'Loaded {len(samples)} WiFi samples')

all_summaries = {}

# --- Official checkpoint ---
ckpt_path = f'{BASE_DIR}/official/model/wifi/b32-256-100s/weights.pt'
print(f'\nLoading official checkpoint: {ckpt_path}')
model_o, diffusion_o, params_o = load_model(ckpt_path)
print('Official model loaded.')

deltas_o, summary_o = run_true_paired_sds(model_o, diffusion_o, samples, 'official')
all_summaries['official'] = summary_o

# Save official CSV
csv_path = f'{RESULTS_DIR}/paired_sds_official_true.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=deltas_o[0].keys())
    writer.writeheader()
    writer.writerows(deltas_o)
print(f'Saved: {csv_path} ({len(deltas_o)} rows)')

# Free GPU memory
del model_o, diffusion_o
torch.cuda.empty_cache()

# --- Self-trained checkpoint ---
ckpt_path = f'{BASE_DIR}/runs/wifi_train/weights-final.pt'
print(f'\nLoading self-trained checkpoint: {ckpt_path}')
model_s, diffusion_s, params_s = load_model(ckpt_path)
print('Self-trained model loaded.')

deltas_s, summary_s = run_true_paired_sds(model_s, diffusion_s, samples, 'selftrained')
all_summaries['selftrained'] = summary_s

# Save self-trained CSV
csv_path = f'{RESULTS_DIR}/paired_sds_selftrained_true.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=deltas_s[0].keys())
    writer.writeheader()
    writer.writerows(deltas_s)
print(f'Saved: {csv_path} ({len(deltas_s)} rows)')

# Save combined summary
json_path = f'{RESULTS_DIR}/paired_sds_true_summary.json'
with open(json_path, 'w') as f:
    json.dump(all_summaries, f, indent=2)
print(f'Saved: {json_path}')

# Final comparison
print('\n' + '='*60)
print('FINAL COMPARISON')
print('='*60)
for tag in ['official', 'selftrained']:
    s = all_summaries[tag]
    print(f'\n  {tag}:')
    print(f'    True paired SDS = {s["mean_delta"]:.4f} [{s["ci95_low"]:.4f}, {s["ci95_high"]:.4f}]')
    print(f'    mean r_own = {s["mean_r_own"]:.4f}')
    print(f'    mean r_unrelated = {s["mean_r_unrelated"]:.4f}')
    print(f'    N_outputs = {s["N_outputs"]}, N_pairs = {s["N_pairs"]}')

# Compare with old grouped SDS
print('\n  Old grouped SDS:')
print('    official:    0.8275')
print('    selftrained: 0.8215')
print('\n  True paired SDS:')
print(f'    official:    {all_summaries["official"]["mean_delta"]:.4f}')
print(f'    selftrained: {all_summaries["selftrained"]["mean_delta"]:.4f}')

delta_diff_o = abs(all_summaries["official"]["mean_delta"] - 0.8275)
delta_diff_s = abs(all_summaries["selftrained"]["mean_delta"] - 0.8215)
print(f'\n  |true_paired - old_grouped|:')
print(f'    official:    {delta_diff_o:.4f}')
print(f'    selftrained: {delta_diff_s:.4f}')

if delta_diff_o < 0.02 and delta_diff_s < 0.02:
    print('\n  VERDICT: CONSISTENT — true paired SDS matches old grouped SDS')
else:
    print('\n  VERDICT: DIFFERENT — see above for details')

print('\nDone.')
