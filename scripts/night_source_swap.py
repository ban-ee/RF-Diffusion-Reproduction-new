"""PHASE 3: Source-swap causal experiment.
Tests whether native_sampling output depends on which source x0 is
used to construct the terminal state, under fixed condition.

CRITICAL: Only variable changed is source x0. Everything else is fixed.
Uses official native_sampling without modification.
Records hardcoded torch.manual_seed(11) in degrade_fn as IMPLEMENTATION DETAIL.
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
from tqdm import tqdm

NIGHT = REPO_ROOT
device = torch.device('cuda')
SEED = 42
N_CONDITIONS = 10
N_SOURCES = 5

# SSIM helper (complex-valued, from official pattern)
def gaussian(window_size, sigma):
    g = torch.tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return g / g.sum()

def create_window(height, width):
    h_window = gaussian(height, 1.5).unsqueeze(1)
    w_window = gaussian(width, 1.5).unsqueeze(1)
    win = h_window.mm(w_window.t()).unsqueeze(0).unsqueeze(0)
    return win.expand(1, 1, height, width).contiguous()

def compute_ssim(pred, target, height, width):
    window = create_window(height, width).to(torch.complex64).to(device)
    padding = [height//2, width//2]
    mu_pred = torch.nn.functional.conv2d(pred, window, padding=padding, groups=1)
    mu_data = torch.nn.functional.conv2d(target, window, padding=padding, groups=1)
    mu_pred_pow = mu_pred.pow(2.)
    mu_data_pow = mu_data.pow(2.)
    mu_pred_data = mu_pred * mu_data
    sp = torch.nn.functional.conv2d(pred*pred, window, padding=padding, groups=1) - mu_pred_pow
    sd = torch.nn.functional.conv2d(target*target, window, padding=padding, groups=1) - mu_data_pow
    spd = torch.nn.functional.conv2d(pred*target, window, padding=padding, groups=1) - mu_pred_data
    C1 = 0.01**2; C2 = 0.03**2
    ssim_map = ((2*mu_pred*mu_data+C1)*(2*spd.real+C2))/((mu_pred_pow+mu_data_pow+C1)*(sp+sd+C2))
    return 2*ssim_map.mean().real

def run_source_swap(task_idx, task_name, params, model, dataset):
    """Run source-swap experiment for one task."""
    diffusion = SignalDiffusion(params)

    # Collect all (data, cond) pairs
    all_samples = []
    for features in dataset:
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        data = features['data']
        cond = features['cond']
        for b in range(data.shape[0]):
            all_samples.append((data[b:b+1], cond[b:b+1]))

    N_total = len(all_samples)
    print(f'  [{task_name}] Total samples available: {N_total}')

    # Select condition indices (deterministic)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    cond_indices = np.random.choice(N_total, size=N_CONDITIONS, replace=False)

    # For each condition, select source indices
    source_indices = {}
    for ci in cond_indices:
        available = [j for j in range(N_total) if j != ci]
        chosen = [ci] + list(np.random.choice(available, size=N_SOURCES-1, replace=False))
        source_indices[ci] = chosen

    results = []

    for ci in tqdm(cond_indices, desc=f'{task_name} source-swap'):
        _, cond_i = all_samples[ci]

        outputs_ci = {}
        for sj in source_indices[ci]:
            data_j, _ = all_samples[sj]

            # Execute official native_sampling
            with torch.no_grad():
                output = diffusion.native_sampling(model, data_j, cond_i, device)

            outputs_ci[sj] = output.cpu()

        # Compute metrics between all pairs of outputs
        sources = list(outputs_ci.keys())
        for a_idx, sa in enumerate(sources):
            for b_idx, sb in enumerate(sources):
                if a_idx >= b_idx:
                    continue
                out_a = outputs_ci[sa]
                out_b = outputs_ci[sb]

                oa_flat = out_a.numpy().ravel()
                ob_flat = out_b.numpy().ravel()

                corr = np.corrcoef(oa_flat, ob_flat)[0, 1]
                mse = np.mean((oa_flat - ob_flat) ** 2)
                nmse = mse / (np.mean(oa_flat ** 2) + 1e-10)

                results.append({
                    'condition_idx': int(ci),
                    'source_a': int(sa),
                    'source_b': int(sb),
                    'pair_type': 'self_self' if (sa == ci and sb == ci) else ('self_other' if (sa == ci or sb == ci) else 'other_other'),
                    'output_pair_corr': float(corr),
                    'output_pair_nmse': float(nmse),
                })

        # Output-to-source metrics
        for sj in sources:
            out_j = outputs_ci[sj]
            data_j, _ = all_samples[sj]

            o_flat = out_j.numpy().ravel()
            d_flat = data_j.cpu().numpy().ravel()

            corr = np.corrcoef(o_flat, d_flat)[0, 1]
            nmse = np.mean((o_flat - d_flat) ** 2) / (np.mean(d_flat ** 2) + 1e-10)

            results.append({
                'condition_idx': int(ci),
                'source_a': int(sj),
                'source_b': int(sj),
                'pair_type': 'self_output_to_source',
                'output_pair_corr': float(corr),
                'output_pair_nmse': float(nmse),
            })

    return results, cond_indices, source_indices

def analyze_and_plot(task_name, results, cond_indices, source_indices):
    """Analyze source-swap results and generate plots."""
    base = NIGHT

    # Save CSV
    csv_path = f'{base}/experiments/night_run/source_swap_{task_name}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Analyze by pair type
    pair_types = {}
    for r in results:
        pt = r['pair_type']
        if pt not in pair_types:
            pair_types[pt] = {'corrs': [], 'nmses': []}
        pair_types[pt]['corrs'].append(r['output_pair_corr'])
        pair_types[pt]['nmses'].append(r['output_pair_nmse'])

    stats = {}
    for pt, data in pair_types.items():
        stats[pt] = {
            'N': len(data['corrs']),
            'corr_mean': float(np.mean(data['corrs'])),
            'corr_std': float(np.std(data['corrs'])),
            'nmse_mean': float(np.mean(data['nmses'])),
            'nmse_std': float(np.std(data['nmses'])),
        }

    # Save stats
    json_path = f'{base}/artifacts/night_run/source_swap_{task_name}.json'
    with open(json_path, 'w') as f:
        json.dump({'stats': stats, 'cond_indices': [int(c) for c in cond_indices]}, f, indent=2)

    # Plot: self-vs-swapped metric difference
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Output pair correlation by type
    ax = axes[0]
    labels = []
    values_corr = []
    for pt in ['self_self', 'self_other', 'other_other']:
        if pt in pair_types:
            labels.append(pt)
            values_corr.append(np.mean(pair_types[pt]['corrs']))
    ax.bar(labels, values_corr, color=['green', 'orange', 'red'])
    ax.set_title(f'{task_name}: Output Pair Correlation by Source Relation')
    ax.set_ylabel('Mean Correlation')

    # 2. Output pair NMSE by type
    ax = axes[1]
    values_nmse = []
    for pt in ['self_self', 'self_other', 'other_other']:
        if pt in pair_types:
            values_nmse.append(np.mean(pair_types[pt]['nmses']))
    ax.bar(labels, values_nmse, color=['green', 'orange', 'red'])
    ax.set_title(f'{task_name}: Output Pair NMSE by Source Relation')
    ax.set_ylabel('Mean NMSE')

    # 3. Source effect by condition
    ax = axes[2]
    cond_output_corrs = {}
    for r in results:
        if r['pair_type'] in ['self_other', 'other_other']:
            ci = r['condition_idx']
            if ci not in cond_output_corrs:
                cond_output_corrs[ci] = []
            cond_output_corrs[ci].append(r['output_pair_corr'])

    ci_list = sorted(cond_output_corrs.keys())
    means = [np.mean(cond_output_corrs[ci]) for ci in ci_list]
    stds = [np.std(cond_output_corrs[ci]) for ci in ci_list]
    ax.bar(range(len(ci_list)), means, yerr=stds, capsize=5)
    ax.set_title(f'{task_name}: Cross-Source Output Correlation by Condition')
    ax.set_xlabel('Condition Index')
    ax.set_ylabel('Mean Output Pair Correlation')

    plt.tight_layout()
    fig_path = f'{base}/artifacts/night_run/source_swap_{task_name}.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  [{task_name}] Stats:')
    for pt, s in stats.items():
        print(f'    {pt}: corr={s["corr_mean"]:.4f}+/-{s["corr_std"]:.4f}, nmse={s["nmse_mean"]:.4f}+/-{s["nmse_std"]:.4f}')

    return stats

# ========== MAIN ==========
print('=' * 60)
print('PHASE 3: SOURCE-SWAP CAUSAL EXPERIMENT')
print('=' * 60)
print('OFFICIAL IMPLEMENTATION DETAIL: degrade_fn contains torch.manual_seed(11)')
print('This means all source comparisons share deterministic noise realization.')
print('')

all_swap_results = {}

# === WiFi ===
print('\n--- WiFi Source-Swap ---')
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.empty_cache()

params_w = all_params[0]
from tfdiff.wifi_model import tfdiff_WiFi
model_w = tfdiff_WiFi(AttrDict(params_w)).to(device)
ckpt = torch.load(params_w.model_dir + '/weights.pt', map_location=device)
model_w.load_state_dict(ckpt['model'])
model_w.eval()

dataset_w = from_path_inference(params_w)
swap_w, cond_idx_w, src_idx_w = run_source_swap(0, 'wifi', params_w, model_w, dataset_w)
stats_w = analyze_and_plot('wifi', swap_w, cond_idx_w, src_idx_w)
all_swap_results['wifi'] = stats_w
torch.cuda.empty_cache()

# === FMCW ===
print('\n--- FMCW Source-Swap ---')
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.empty_cache()

params_f = all_params[1]
from tfdiff.fmcw_model import tfdiff_fmcw
model_f = tfdiff_fmcw(AttrDict(params_f)).to(device)
ckpt = torch.load(params_f.model_dir + '/weights.pt', map_location=device)
model_f.load_state_dict(ckpt['model'])
model_f.eval()

dataset_f = from_path_inference(params_f)
swap_f, cond_idx_f, src_idx_f = run_source_swap(1, 'fmcw', params_f, model_f, dataset_f)
stats_f = analyze_and_plot('fmcw', swap_f, cond_idx_f, src_idx_f)
all_swap_results['fmcw'] = stats_f
torch.cuda.empty_cache()

print('\n=== SOURCE_SWAP_COMPLETE ===')
