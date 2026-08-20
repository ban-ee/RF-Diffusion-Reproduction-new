"""WiFi 训练链路冒烟测试。

检查数据加载、前向、反向、优化器更新以及 checkpoint 保存和重新加载。
该脚本随机初始化模型，不用于复现官方 checkpoint 指标。
"""
import os, sys, time
import torch
import numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(REPO_ROOT + '/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params, AttrDict
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import Collator, _nested_map, WiFiDataset

params = all_params[0]
# Use cond dir as data dir since we don't have raw/
params.data_dir = ['./wifi/cond']
params.batch_size = 4
params.model_dir = '../runs/wifi_smoke'
params.log_dir = '../runs/wifi_smoke/log'
os.makedirs(params.model_dir, exist_ok=True)
os.makedirs(params.log_dir, exist_ok=True)

device = torch.device('cuda')
print(f'CUDA: {torch.cuda.get_device_name(0)}')

# Use official Collator for proper variable-length handling
collator = Collator(params)
dataset = WiFiDataset(params.data_dir)
print(f'Dataset: {len(dataset)} samples')

# Use DataLoader with official collator
from torch.utils.data import DataLoader
loader = DataLoader(
    dataset,
    batch_size=params.batch_size,
    collate_fn=collator.collate,
    shuffle=True,
    num_workers=0,  # Avoid fork issues
    drop_last=True,
)

# Model
model = tfdiff_WiFi(params).to(device)
diffusion = SignalDiffusion(params)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model params: {n_params:,}')

# Check first batch shape
for batch in loader:
    batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
    data = batch['data']
    cond = batch['cond']
    print(f'Batch data: {data.shape}, cond: {cond.shape}')
    print(f'Data dtype: {data.dtype}, Cond dtype: {cond.dtype}')
    break

# Optimizer + Scheduler (keep original StepLR(1, 0.5))
opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.5)
loss_fn = torch.nn.MSELoss()

# Training loop
MAX_ITER = 100
model.train()
t0 = time.time()
iter_cnt = 0
epoch = 0
losses = []
peak_mem = 0
last_loss = None

while iter_cnt < MAX_ITER:
    for batch in loader:
        if iter_cnt >= MAX_ITER:
            break
        batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        data = batch['data']
        cond = batch['cond']
        B = data.shape[0]

        t = torch.randint(0, diffusion.max_step, [B], dtype=torch.int64)  # CPU for degrade_fn indexing
        xt = diffusion.degrade_fn(data, t, 0)
        pred = model(xt, t, cond)

        last_loss = loss_fn(data, pred)
        opt.zero_grad()
        last_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        opt.step()

        losses.append(last_loss.item())
        iter_cnt += 1
        peak_mem = max(peak_mem, torch.cuda.memory_allocated() / 1e9)

        if iter_cnt % 20 == 0 or iter_cnt <= 3:
            elapsed = time.time() - t0
            lr = opt.param_groups[0]['lr']
            print(f'  iter={iter_cnt:5d}  loss={last_loss.item():.6f}  lr={lr:.2e}  mem={peak_mem:.2f}GB  t={elapsed:.0f}s')

    lr_scheduler.step()
    epoch += 1

elapsed = time.time() - t0
print(f'\nSMOKE COMPLETE: {iter_cnt} iters in {elapsed:.0f}s')
print(f'  final_loss={last_loss.item():.6f}')
print(f'  first_loss={losses[0]:.6f}')
print(f'  loss_trend: {losses[0]:.4f} -> {losses[-1]:.4f}')
print(f'  peak_memory={peak_mem:.2f}GB')
print(f'  sec_per_iter={elapsed/iter_cnt:.2f}')

# Check for NaN/Inf
has_nan = any(np.isnan(l) for l in losses)
has_inf = any(np.isinf(l) for l in losses)
print(f'  NaN: {has_nan}, Inf: {has_inf}')

# Checkpoint
ckpt_path = f'{params.model_dir}/weights.pt'
torch.save({
    'iter': iter_cnt,
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
}, ckpt_path)
ckpt_size = os.path.getsize(ckpt_path) / 1e6
print(f'Checkpoint saved: {ckpt_path} ({ckpt_size:.1f}MB)')

# Verify reload
model2 = tfdiff_WiFi(params).to(device)
ckpt = torch.load(ckpt_path, map_location=device)
model2.load_state_dict(ckpt['model'])
model2.eval()

# Test forward with real batch shape
with torch.no_grad():
    dx = torch.randn(1, 512, 90, 2, device=device)
    tc = torch.zeros(1, dtype=torch.int64, device=device)
    co = torch.randn(1, 6, device=device)
    out = model2(dx, tc, co)
print(f'Reload OK: out shape={list(out.shape)}, mean={out.mean().item():.4f}, std={out.std().item():.4f}')

# GPU memory summary
print(f'GPU memory used: {torch.cuda.max_memory_allocated()/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB')
print('SMOKE_TEST=PASS')
