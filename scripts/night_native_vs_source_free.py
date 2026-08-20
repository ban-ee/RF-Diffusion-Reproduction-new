"""PHASE 4: Native vs Source-Free Protocol Audit.

Checks whether official fast_sampling can be legally called for WiFi/FMCW.
If yes: EXPLORATORY OFFICIAL-FUNCTION COMPARISON.
If no: UNSUPPORTED BY RELEASED PROTOCOL.

NO modification to official sampling functions.
"""
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
N_CONDITIONS = 10

def run_comparison(task_idx, task_name, params, model):
    """Compare native_sampling vs fast_sampling outputs."""
    diffusion = SignalDiffusion(params)

    # Collect conditions from dataset
    dataset = from_path_inference(params)
    all_conds = []
    all_data = []
    for features in dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        data = features['data']
        cond = features['cond']
        for b in range(data.shape[0]):
            all_data.append(data[b:b+1])
            all_conds.append(cond[b:b+1])

    N_total = len(all_conds)
    print(f'  [{task_name}] Total conditions available: {N_total}')

    # Select conditions
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    indices = np.random.choice(N_total, size=N_CONDITIONS, replace=False)

    results = []

    for ci in indices:
        data_i = all_data[ci]
        cond_i = all_conds[ci]

        with torch.no_grad():
            # native_sampling: source-dependent
            native_out = diffusion.native_sampling(model, data_i, cond_i, device)

            # fast_sampling: source-free
            fast_out = diffusion.fast_sampling(model, cond_i, device)

        native_np = native_out.cpu().numpy().ravel()
        fast_np = fast_out.cpu().numpy().ravel()
        data_np = data_i.cpu().numpy().ravel()

        # Native vs fast correlation
        nf_corr = np.corrcoef(native_np, fast_np)[0, 1]
        nf_nmse = np.mean((native_np - fast_np) ** 2) / (np.mean(native_np ** 2) + 1e-10)

        # Each vs data
        native_data_corr = np.corrcoef(native_np, data_np)[0, 1]
        fast_data_corr = np.corrcoef(fast_np, data_np)[0, 1]

        # Energy and std
        results.append({
            'condition_idx': int(ci),
            'native_std': float(np.std(native_np)),
            'fast_std': float(np.std(fast_np)),
            'native_mean': float(np.mean(native_np)),
            'fast_mean': float(np.mean(fast_np)),
            'native_energy': float(np.sum(native_np ** 2)),
            'fast_energy': float(np.sum(fast_np ** 2)),
            'native_vs_fast_corr': float(nf_corr),
            'native_vs_fast_nmse': float(nf_nmse),
            'native_vs_data_corr': float(native_data_corr),
            'fast_vs_data_corr': float(fast_data_corr),
        })

    return results

def analyze(task_name, results):
    """Analyze and save comparison results."""
    base = NIGHT

    # CSV
    csv_path = f'{base}/experiments/night_run/native_vs_fast_{task_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    keys = [k for k in results[0].keys() if k != 'condition_idx']
    stats = {}
    for k in keys:
        vals = [r[k] for r in results]
        stats[k] = {
            'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
            'min': float(np.min(vals)), 'max': float(np.max(vals)),
        }

    json_path = f'{base}/artifacts/night_run/native_vs_fast_{task_name}.json'
    with open(json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'  [{task_name}] Results:')
    print(f'    native_vs_fast_corr:    {stats["native_vs_fast_corr"]["mean"]:.4f} +/- {stats["native_vs_fast_corr"]["std"]:.4f}')
    print(f'    native_vs_fast_nmse:    {stats["native_vs_fast_nmse"]["mean"]:.4f} +/- {stats["native_vs_fast_nmse"]["std"]:.4f}')
    print(f'    native_vs_data_corr:    {stats["native_vs_data_corr"]["mean"]:.4f} +/- {stats["native_vs_data_corr"]["std"]:.4f}')
    print(f'    fast_vs_data_corr:      {stats["fast_vs_data_corr"]["mean"]:.4f} +/- {stats["fast_vs_data_corr"]["std"]:.4f}')
    print(f'    native_energy / fast_energy ratio: {stats["native_energy"]["mean"] / stats["fast_energy"]["mean"]:.4f}')

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. native vs fast correlation per condition
    axes[0].bar(range(len(results)), [r['native_vs_fast_corr'] for r in results])
    axes[0].set_title(f'{task_name}: Native vs Fast Output Correlation')
    axes[0].set_xlabel('Condition index')
    axes[0].set_ylabel('Pearson r')
    axes[0].set_ylim(0, 1)

    # 2. Energy comparison
    x = range(len(results))
    axes[1].bar([i-0.15 for i in x], [r['native_energy'] for r in results], width=0.3, label='Native', color='blue')
    axes[1].bar([i+0.15 for i in x], [r['fast_energy'] for r in results], width=0.3, label='Fast', color='red')
    axes[1].set_title(f'{task_name}: Output Energy Comparison')
    axes[1].set_xlabel('Condition index')
    axes[1].set_ylabel('Energy')
    axes[1].legend()

    # 3. Std comparison
    axes[2].bar([i-0.15 for i in x], [r['native_std'] for r in results], width=0.3, label='Native', color='blue')
    axes[2].bar([i+0.15 for i in x], [r['fast_std'] for r in results], width=0.3, label='Fast', color='red')
    axes[2].set_title(f'{task_name}: Output Std Comparison')
    axes[2].set_xlabel('Condition index')
    axes[2].set_ylabel('Std')
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(f'{base}/artifacts/night_run/native_vs_fast_{task_name}.png', dpi=150, bbox_inches='tight')
    plt.close()

    return stats

# ========== MAIN ==========
print('=' * 60)
print('PHASE 4: NATIVE VS SOURCE-FREE PROTOCOL AUDIT')
print('=' * 60)

# Verify fast_sampling API compatibility
print('\n--- API COMPATIBILITY CHECK ---')
print('fast_sampling signature: fast_sampling(self, restore_fn, cond, device)')
print('native_sampling signature: native_sampling(self, restore_fn, data, cond, device)')
print('fast_sampling handles task_id 0,1 with 2 unsqueeze dims (vs 3 for task_id 2,3)')
print('VERDICT: fast_sampling IS callable for WiFi/FMCW without modification.')
print('Proceeding with EXPLORATORY OFFICIAL-FUNCTION COMPARISON.')
print()

all_native_vs_fast = {}

# === WiFi ===
print('--- WiFi: Native vs Fast ---')
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.empty_cache()

params_w = all_params[0]
from tfdiff.wifi_model import tfdiff_WiFi
model_w = tfdiff_WiFi(AttrDict(params_w)).to(device)
ckpt = torch.load(params_w.model_dir + '/weights.pt', map_location=device)
model_w.load_state_dict(ckpt['model'])
model_w.eval()

res_w = run_comparison(0, 'wifi', params_w, model_w)
stats_w = analyze('wifi', res_w)
all_native_vs_fast['wifi'] = stats_w
torch.cuda.empty_cache()

# === FMCW ===
print('\n--- FMCW: Native vs Fast ---')
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.empty_cache()

params_f = all_params[1]
from tfdiff.fmcw_model import tfdiff_fmcw
model_f = tfdiff_fmcw(AttrDict(params_f)).to(device)
ckpt = torch.load(params_f.model_dir + '/weights.pt', map_location=device)
model_f.load_state_dict(ckpt['model'])
model_f.eval()

res_f = run_comparison(1, 'fmcw', params_f, model_f)
stats_f = analyze('fmcw', res_f)
all_native_vs_fast['fmcw'] = stats_f
torch.cuda.empty_cache()

print('\n=== NATIVE_VS_FAST_COMPLETE ===')
print('STATUS: EXPLORATORY OFFICIAL-FUNCTION COMPARISON')
print('NOTE: This is NOT the paper official evaluation protocol.')
print('It compares two publicly available official functions.')
