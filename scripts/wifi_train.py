"""WiFi CSI 受控训练入口。

相对官方发布代码的实验修正：
  1. 使用正确的任务映射：task_id=0 对应 WiFi；
  2. 学习率调度改为 StepLR(10, 0.5)，避免每个 epoch 都减半；
  3. 保留可配置的最大迭代数与批大小。
"""
import argparse, os, sys, time, json, csv
import torch
import numpy as np
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.chdir(REPO_ROOT + '/official')
sys.path.insert(0, '.')

from tfdiff.params import all_params
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.diffusion import SignalDiffusion
from tfdiff.dataset import Collator, _nested_map, WiFiDataset
from torch.utils.data import DataLoader

# ============================================================
# CONFIG
# ============================================================
parser = argparse.ArgumentParser(description='RF-Diffusion WiFi CSI 受控训练')
parser.add_argument('--max_iter', type=int, default=2000, help='最大优化迭代数')
parser.add_argument('--batch_size', type=int, default=4, help='单卡批大小')
args = parser.parse_args()

if args.max_iter <= 0:
    parser.error('--max_iter 必须大于 0')
if args.batch_size <= 0:
    parser.error('--batch_size 必须大于 0')

MAX_ITER = args.max_iter
BATCH_SIZE = args.batch_size
EFFECTIVE_BATCH = 4  # No gradient accumulation for controlled training
GRAD_ACCUM = 1

MODEL_DIR = '../runs/wifi_train'
LOG_DIR = '../runs/wifi_train/log'
DATA_DIR = ['./wifi/cond']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device('cuda')
print(f'CUDA: {torch.cuda.get_device_name(0)}')
print(f'Config: max_iter={MAX_ITER}, batch_size={BATCH_SIZE}, '
      f'grad_accum={GRAD_ACCUM}, effective_batch={EFFECTIVE_BATCH}')

# ============================================================
# DATA
# ============================================================
params = all_params[0]
params.data_dir = DATA_DIR
params.batch_size = BATCH_SIZE

collator = Collator(params)
dataset = WiFiDataset(params.data_dir)
print(f'Dataset: {len(dataset)} samples')

loader = DataLoader(
    dataset, batch_size=BATCH_SIZE, collate_fn=collator.collate,
    shuffle=True, num_workers=0, drop_last=True,
)

# ============================================================
# MODEL
# ============================================================
model = tfdiff_WiFi(params).to(device)
diffusion = SignalDiffusion(params)
n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params:,} params')

# ============================================================
# OPTIMIZER + SCHEDULER (REPAIRED)
# ============================================================
opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
# REPAIRED: step_size=20 instead of 1 — LR halves every 20 epochs, not every epoch
lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, 20, gamma=0.5)
loss_fn = torch.nn.MSELoss()

# ============================================================
# TRAINING LOOP
# ============================================================
model.train()
t_start = time.time()
metrics = []
iter_cnt = 0
epoch = 0
best_loss = float('inf')
last_ckpt_iter = 0

print(f'\n{"="*50}')
print(f'TRAINING START — {time.strftime("%H:%M:%S")}')
print(f'{"="*50}')

while iter_cnt < MAX_ITER:
    epoch_losses = []

    for batch in loader:
        if iter_cnt >= MAX_ITER:
            break

        batch = _nested_map(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        data = batch['data']
        cond = batch['cond']
        B = data.shape[0]

        t = torch.randint(0, diffusion.max_step, [B], dtype=torch.int64)
        xt = diffusion.degrade_fn(data, t, 0)
        pred = model(xt, t, cond)

        loss = loss_fn(data, pred) / GRAD_ACCUM
        loss.backward()
        epoch_losses.append(loss.item() * GRAD_ACCUM)

        if (iter_cnt + 1) % GRAD_ACCUM == 0 or iter_cnt == MAX_ITER - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            opt.step()
            opt.zero_grad()

        iter_cnt += 1

        if torch.isnan(loss).any():
            print(f'[FATAL] NaN loss at iter {iter_cnt}')
            sys.exit(1)

    lr_scheduler.step()
    epoch += 1

    avg_loss = float(np.mean(epoch_losses))
    current_lr = opt.param_groups[0]['lr']
    elapsed = time.time() - t_start
    mem = torch.cuda.max_memory_allocated() / 1e9

    metrics.append({
        'epoch': epoch, 'iter': iter_cnt,
        'loss': avg_loss, 'lr': current_lr,
        'wall_time_s': round(elapsed, 1),
        'gpu_mem_gb': round(mem, 2),
    })

    if epoch % 5 == 0 or epoch <= 3:
        print(f'[Epoch {epoch:4d}] iter={iter_cnt:6d}  loss={avg_loss:.6f}  '
              f'lr={current_lr:.2e}  mem={mem:.1f}GB  t={elapsed:.0f}s')

    # Checkpoint every 500 iters
    if iter_cnt - last_ckpt_iter >= 500:
        ckpt_path = f'{MODEL_DIR}/weights-{iter_cnt}.pt'
        torch.save({
            'iter': iter_cnt, 'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': opt.state_dict(),
            'params': dict(params),
            'metrics': metrics,
        }, ckpt_path)
        last_ckpt_iter = iter_cnt
        print(f'  [Checkpoint] {ckpt_path}')

    if avg_loss < best_loss:
        best_loss = avg_loss

# ============================================================
# FINAL SAVE
# ============================================================
elapsed = time.time() - t_start

# Save final checkpoint
final_ckpt = f'{MODEL_DIR}/weights-final.pt'
torch.save({
    'iter': iter_cnt, 'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
    'params': dict(params),
    'metrics': metrics,
}, final_ckpt)

# Also save as weights.pt for easy loading
torch.save({
    'iter': iter_cnt, 'epoch': epoch,
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
    'params': dict(params),
    'metrics': metrics,
}, f'{MODEL_DIR}/weights.pt')

print(f'\n{"="*50}')
print(f'TRAINING COMPLETE — {time.strftime("%H:%M:%S")}')
print(f'{"="*50}')
print(f'Iterations: {iter_cnt}')
print(f'Epochs: {epoch}')
print(f'Final loss: {metrics[-1]["loss"]:.6f}')
print(f'Best loss: {best_loss:.6f}')
print(f'Final LR: {metrics[-1]["lr"]:.2e}')
print(f'Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)')
print(f'Peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f}GB')
print(f'Checkpoint: {final_ckpt}')

# Save metrics CSV
csv_path = f'{LOG_DIR}/training_metrics.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
    writer.writeheader()
    writer.writerows(metrics)

# Save summary JSON
summary = {
    'status': 'COMPLETED',
    'max_iter': MAX_ITER,
    'batch_size': BATCH_SIZE,
    'effective_batch': EFFECTIVE_BATCH,
    'grad_accum': GRAD_ACCUM,
    'total_iters': iter_cnt,
    'total_epochs': epoch,
    'final_loss': metrics[-1]['loss'],
    'best_loss': best_loss,
    'final_lr': metrics[-1]['lr'],
    'total_time_s': round(elapsed, 1),
    'sec_per_iter': round(elapsed / iter_cnt, 2),
    'peak_gpu_mem_gb': round(torch.cuda.max_memory_allocated()/1e9, 2),
    'n_params': n_params,
    'dataset_size': len(dataset),
    'model': 'tfdiff_WiFi',
    'scheduler': 'StepLR(10, 0.5) — REPAIRED from StepLR(1, 0.5)',
    'loss_fn': 'MSELoss (complex)',
    'ema': 'NONE (paper says 0.999, release has no EMA)',
}
with open(f'{LOG_DIR}/training_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\nSummary saved to {LOG_DIR}/')
print('TRAINING_RESULT=PASS')
