"""PHASE 1: HDT Runtime Trace — official model + checkpoint + real data."""
import sys, os, json, torch, numpy as np
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT + '/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params, AttrDict
from tfdiff.dataset import from_path_inference, _nested_map

NIGHT = REPO_ROOT
device = torch.device('cuda')
results_all = {}

def trace_model(model, data, cond, timestep, name):
    hooks = []
    trace = []

    def make_hook(layer_name):
        def hook(module, inp, outp):
            info = {'layer': layer_name, 'type': type(module).__name__}
            if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                info['input_shapes'] = [list(t.shape) for t in inp if isinstance(t, torch.Tensor)]
            if isinstance(outp, torch.Tensor):
                info['output_shape'] = list(outp.shape)
            elif isinstance(outp, (tuple, list)):
                info['output_shapes'] = [list(t.shape) if isinstance(t, torch.Tensor) else str(type(t)) for t in outp[:3]]
            trace.append(info)
        return hook

    for child_name, child_module in model.named_children():
        hooks.append(child_module.register_forward_hook(make_hook(child_name)))

    with torch.no_grad():
        try:
            output = model(data, timestep, cond)
            output_shape = list(output.shape)
        except Exception as e:
            output = None
            output_shape = f'ERROR: {e}'

    for h in hooks:
        h.remove()

    params = sum(p.numel() for p in model.parameters())

    result = {
        'model': name,
        'total_params': params,
        'input_shape': list(data.shape),
        'cond_shape': list(cond.shape),
        'timestep': timestep.item() if isinstance(timestep, torch.Tensor) else timestep,
        'output_shape': output_shape,
        'trace': trace
    }

    param_counts = {}
    for n, p in model.named_parameters():
        top = n.split('.')[0]
        param_counts[top] = param_counts.get(top, 0) + p.numel()
    result['param_by_module'] = param_counts

    return result

torch.manual_seed(42)

# === WiFi ===
print('--- WiFi HDT Trace ---')
params_wifi = all_params[0]
from tfdiff.wifi_model import tfdiff_WiFi
model_wifi = tfdiff_WiFi(AttrDict(params_wifi)).to(device)
ckpt = torch.load(params_wifi.model_dir + '/weights.pt', map_location=device)
model_wifi.load_state_dict(ckpt['model'])
model_wifi.eval()
print(f'Loaded: {sum(p.numel() for p in model_wifi.parameters()):,} params')

dataset_wifi = from_path_inference(params_wifi)
features_wifi = next(iter(dataset_wifi))
features_wifi = _nested_map(features_wifi, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
data_w, cond_w = features_wifi['data'], features_wifi['cond']
t_w = torch.tensor([params_wifi.max_step - 1], device=device)
trace_wifi = trace_model(model_wifi, data_w, cond_w, t_w, 'WiFi')
results_all['wifi'] = trace_wifi
torch.cuda.empty_cache()

# === FMCW ===
print('--- FMCW HDT Trace ---')
params_fmcw = all_params[1]
from tfdiff.fmcw_model import tfdiff_fmcw
model_fmcw = tfdiff_fmcw(AttrDict(params_fmcw)).to(device)
ckpt = torch.load(params_fmcw.model_dir + '/weights.pt', map_location=device)
model_fmcw.load_state_dict(ckpt['model'])
model_fmcw.eval()
print(f'Loaded: {sum(p.numel() for p in model_fmcw.parameters()):,} params')

dataset_fmcw = from_path_inference(params_fmcw)
features_fmcw = next(iter(dataset_fmcw))
features_fmcw = _nested_map(features_fmcw, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
data_f, cond_f = features_fmcw['data'], features_fmcw['cond']
t_f = torch.tensor([params_fmcw.max_step - 1], device=device)
trace_fmcw = trace_model(model_fmcw, data_f, cond_f, t_f, 'FMCW')
results_all['fmcw'] = trace_fmcw
torch.cuda.empty_cache()

# === 5G MIMO ===
print('--- 5G MIMO HDT Trace ---')
params_mimo = all_params[2]
from tfdiff.mimo_model import tfdiff_mimo
model_mimo = tfdiff_mimo(AttrDict(params_mimo)).to(device)
ckpt = torch.load(params_mimo.model_dir + '/weights.pt', map_location=device)
model_mimo.load_state_dict(ckpt['model'])
model_mimo.eval()
print(f'Loaded: {sum(p.numel() for p in model_mimo.parameters()):,} params')

dataset_mimo = from_path_inference(params_mimo)
features_mimo = next(iter(dataset_mimo))
features_mimo = _nested_map(features_mimo, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
data_m, cond_m = features_mimo['data'], features_mimo['cond']
t_m = torch.tensor([params_mimo.max_step - 1], device=device)
trace_mimo = trace_model(model_mimo, data_m, cond_m, t_m, '5G_MIMO')
results_all['mimo'] = trace_mimo
torch.cuda.empty_cache()

# Save results
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)

for k, v in results_all.items():
    fpath = f'{NIGHT}/artifacts/night_run/hdt_trace_{k}.json'
    with open(fpath, 'w') as f:
        json.dump(v, f, indent=2, cls=NpEncoder)
    print(f'Saved: {fpath}')

# Print summary
print('')
print('=' * 60)
print('HDT TRACE SUMMARY')
print('=' * 60)
for name, r in results_all.items():
    print(f'\n{name}:')
    print(f'  Input:       {r["input_shape"]}')
    print(f'  Condition:   {r["cond_shape"]}')
    print(f'  Timestep:    {r["timestep"]}')
    print(f'  Output:      {r["output_shape"]}')
    print(f'  Total params: {r["total_params"]:,}')
    print(f'  Hooked layers: {len(r["trace"])}')
    for t in r['trace']:
        shapes = t.get('output_shape') or t.get('output_shapes', '?')
        print(f'    {t["layer"]} ({t["type"]}): out={shapes}')

print('\n=== HDT_TRACE_COMPLETE ===')
