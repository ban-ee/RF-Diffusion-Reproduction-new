# 原论文与官方实现

本仓库是对 **RF-Diffusion: Radio Signal Generation via Time-Frequency Diffusion**
（ACM MobiCom 2024, pp. 77-92）的**批判性复现与机理分析**，作为浙江大学课程评估项目完成。

仓库包含两部分：(a) **未经修改的官方源码**，以及 (b) 从完整实验工程中筛选出的正式训练、
评测、机理分析和绘图脚本。整理过程没有重写算法或实验逻辑；仅将服务器绝对路径替换为
可移植的 `REPO_ROOT`，补充 WiFi 训练命令行参数，并修正冒烟测试中的 checkpoint 保存变量名
（见[说明](#说明)）。

> **原论文：** Guoxuan Chi, Zheng Yang, Chenshu Wu, Jingao Xu, Yuchong Gao,
> Yunhao Liu, Tony Xiao Han, *RF-Diffusion: Radio Signal Generation via
> Time-Frequency Diffusion*, MobiCom 2024.
>
> **官方仓库：** https://github.com/mobicom24/RF-Diffusion
> （上游 commit `eb872b0c4543da65424f5598ae40826e76e7edea`，"Update README.md"，
> 2024-06-17）。`official/` 目录即该源码，**未经修改**，以 **GPL v3** 许可证发布
> （见 `official/LICENSE`）。它**不是**我们自己的实现。

## 仓库结构

```
.
├── README.md                      # 本文件
├── requirements.txt               # 核心 Python 依赖
├── .gitignore                     # 排除数据 / checkpoint / 输出
├── official/                      # A. 未修改的官方 RF-Diffusion 源码（GPL v3）
│   ├── LICENSE
│   ├── README.md                  # 上游 README
│   ├── inference.py               # 官方评测入口
│   ├── train.py                   # 官方训练入口
│   ├── complex/                   # 复数值层
│   └── tfdiff/                    # 扩散 / learner / params / dataset / 模型
├── scripts/                       # B. 正式实验入口与训练链路检查脚本
│   ├── wifi_train.py              # WiFi CSI 受控训练
│   ├── wifi_smoke.py              # WiFi 训练链路冒烟测试
│   ├── night_terminal_consistency.py   # 终端分布分析
│   ├── night_source_swap.py            # 源交换因果实验
│   ├── night_native_vs_source_free.py  # native vs source-free 采样
│   ├── night_hdt_trace.py              # HDT 运行时追踪
│   ├── final_validation.py             # 正式源依赖 + 消融
│   ├── true_paired_sds.py              # 逐输出成对 SDS
│   ├── common_epsilon_forward.py       # 公共 epsilon 反事实
│   ├── mechanism_verification.py       # TFD 机理验证（P0-P5）
│   └── extract_terminal_states.py      # 供图 5 使用的终端张量
├── analysis/                      # C. Layer-B 前向 / 终端分析
│   ├── analyze_official_tfd.py    # 官方 TFD 核心算法分析
│   ├── trace_hdt.py               # HDT 架构追踪
│   ├── tfd_real_data_visualize.py # 真实数据 TFD 可视化
│   └── terminal_state_corrected.py# 终端状态分析（正式修正版）
└── plotting/                      # D. 绘图脚本（纯 CPU，读取 CSV）
    ├── terminal_consistency_plot.py
    ├── source_swap_plot.py
    └── native_vs_fast_plot.py
```

## 环境

报告结果是在如下 AutoDL 实例上产出的：

| 项目 | 值 |
|------|----|
| GPU | NVIDIA RTX 4090 D (24 GB) |
| Python | 3.8.10 (conda, `/root/miniconda3`) |
| PyTorch | 2.0.0+cu118 |
| CUDA | 11.8 |
| OS | Linux (容器) |

安装依赖：

```bash
# 先装 PyTorch（官方 README：`pip3 install torch`）
pip3 install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

## 数据准备

数据集与 checkpoint **不包含**在本仓库中。请从官方 release 下载：

> https://github.com/mobicom24/RF-Diffusion/releases/tag/dataset_model

放置位置需让官方代码（其 `tfdiff/params.py` 使用相对路径 `./dataset/...` 与
`./model/...`）能够找到：

```bash
# 在仓库根目录执行
unzip -q dataset.zip -d official/dataset   # -> official/dataset/{wifi,fmcw,mimo}/...
unzip -q model.zip   -d official           # -> official/model/{wifi,fmcw,mimo}/...

# 自定义 WiFi 训练和验证脚本沿用原实验的 official/wifi/cond 路径。
# 在首次准备数据时建立兼容链接；如果链接已经存在，不要重复执行。
ln -s dataset/wifi official/wifi
ln -s dataset/fmcw official/fmcw
ln -s dataset/mimo official/mimo
```

- 评测数据（`*.mat`）位于 `official/dataset/{wifi,fmcw,mimo}/cond/`。
- 生成的输出写入 `official/dataset/{wifi,fmcw,mimo}/{img,img_matric,output}/`。
- WiFi **训练**脚本从 `official/wifi/cond/` 读取它的 41 个 `.mat` 样本
  （见 `scripts/wifi_train.py`，`DATA_DIR=['./wifi/cond']`）。

## 模型权重准备

官方预训练 checkpoint（来自 `model.zip`）应位于：

| 任务 | 路径（`official/model/...`） |
|------|-----------------------------|
| WiFi | `wifi/b32-256-100s/weights.pt` |
| FMCW | `fmcw/b32-256-100s/weights.pt` |
| 5G MIMO | `mimo/b32-256-200s/weights.pt` |

由 `scripts/wifi_train.py` 训练出的自训练 WiFi checkpoint 会写入
`runs/wifi_train/weights-final.pt`，并被 `final_validation.py` 与
`true_paired_sds.py` 作为"自训练"条件读取。

## 官方模型权重复现

用官方 checkpoint 复现论文报告的指标（这是严格复现中使用的确切命令，需在
`official/` 目录内执行）：

```bash
cd official
ulimit -n 65536            # 避免 DataLoader 文件描述符耗尽（192 workers）
python inference.py --task_id 0    # WiFi
python inference.py --task_id 1    # FMCW
python inference.py --task_id 2    # 5G FDD 信道估计
```

指标（SSIM / FID / SNR）打印到 stdout。FID 使用了代码自带的修正系数
（WiFi 为 x1.9，FMCW 为 x0.9，见 `inference.py:259,261`）。

## WiFi CSI 训练

在 41 个 WiFi CSI 样本上的受控单卡训练（修复了任务编号映射与学习率调度器；详见脚本说明）：

```bash
python scripts/wifi_train.py --max_iter 2000 --batch_size 4
```

快速冒烟测试（随机初始化模型，检查数据加载、前向、反向、更新和 checkpoint 重载）：

```bash
python scripts/wifi_smoke.py
```

## 终端分布分析

在所有真实评测样本上刻画 TFD 前向退化 `x_0 -> x_T`：

```bash
python scripts/night_terminal_consistency.py    # 多样本终端指标
python scripts/extract_terminal_states.py       # 供图 5 使用的终端张量
python analysis/terminal_state_corrected.py     # 采用严格措辞的正式终端分析
```

## 源依赖分析

检验发布的 `native_sampling` 协议是否使输出依赖于终端源样本：

```bash
python scripts/night_source_swap.py             # 源交换因果实验
python scripts/final_validation.py              # 正式：3 项指标、全部 41 个样本、3 个随机种子、两组权重
python scripts/true_paired_sds.py               # 逐输出成对源依赖得分
```

注意：`mechanism_verification.py` 会读取
`experiments/final_validation/source_dependence_{official,selftrained}.csv`，因此请先运行
`final_validation.py` 再运行 `mechanism_verification.py`。

## 机理验证

验证时频扩散机理（噪声/模糊加权、公共 epsilon 反事实、HDT 架构）：

```bash
python scripts/mechanism_verification.py        # P0-P5 机理检查
python scripts/common_epsilon_forward.py        # 公共 epsilon 反事实
python scripts/night_hdt_trace.py               # HDT 运行时追踪
python analysis/analyze_official_tfd.py         # 官方 TFD 核心算法分析
python analysis/trace_hdt.py                    # HDT 架构追踪
python analysis/tfd_real_data_visualize.py      # 真实数据 TFD 可视化
```

## 评测 / 绘图

绘图脚本为纯 CPU，从 `results/tables/` 读取 CSV，把 PNG 写入 `results/figures/`。先把
实验 CSV 复制到该位置：

```bash
mkdir -p results/tables
cp experiments/night_run/terminal_metrics_*.csv   results/tables/   # 来自终端分析
cp experiments/night_run/source_swap_*.csv        results/tables/   # 来自源交换
cp experiments/night_run/native_vs_fast_*.csv     results/tables/   # 来自 native-vs-fast

python plotting/terminal_consistency_plot.py
python plotting/source_swap_plot.py
python plotting/native_vs_fast_plot.py
```

## 实验 → 脚本映射

| 实验（报告） | 脚本 | 命令 | 输入 | 输出 |
|---|---|---|---|---|
| 官方模型权重复现 | `official/inference.py` | `python inference.py --task_id {0,1,2}` | 官方权重 + `official/dataset/*/cond` | SSIM/FID/SNR（终端输出） |
| WiFi CSI 训练 | `scripts/wifi_train.py` | `python scripts/wifi_train.py --max_iter 2000 --batch_size 4` | `official/wifi/cond`（41 个 .mat） | `runs/wifi_train/weights-final.pt` |
| 终端分布 | `scripts/night_terminal_consistency.py` | `python scripts/night_terminal_consistency.py` | 官方权重 + 评测数据 | `experiments/night_run/terminal_metrics_*.csv` |
| 终端状态（图 5） | `scripts/extract_terminal_states.py` | `python scripts/extract_terminal_states.py` | 官方权重 + 评测数据 | `experiments/fig5_terminal/*` |
| 源依赖（因果） | `scripts/night_source_swap.py` | `python scripts/night_source_swap.py` | 官方权重 + 评测数据 | `experiments/night_run/source_swap_*.csv` |
| 源依赖（正式） | `scripts/final_validation.py` | `python scripts/final_validation.py` | 官方权重 + 自训练权重 | `experiments/final_validation/*.csv` |
| 逐输出成对 SDS | `scripts/true_paired_sds.py` | `python scripts/true_paired_sds.py` | 官方权重 + 自训练权重 | `experiments/final_patch/*` |
| 原生采样与无源采样对比 | `scripts/night_native_vs_source_free.py` | `python scripts/night_native_vs_source_free.py` | 官方权重 + 评测数据 | `experiments/night_run/native_vs_fast_*.csv` |
| 机理验证 | `scripts/mechanism_verification.py` | `python scripts/mechanism_verification.py` | `experiments/final_validation/*.csv` | `experiments/mechanism_verification/*` |
| 公共 epsilon 反事实 | `scripts/common_epsilon_forward.py` | `python scripts/common_epsilon_forward.py` | 官方权重 + 评测数据 | `experiments/final_patch/*` |
| HDT 运行时追踪 | `scripts/night_hdt_trace.py` | `python scripts/night_hdt_trace.py` | 官方权重 | `artifacts/night_run/hdt_trace_*.json` |
| 官方 TFD 分析 | `analysis/analyze_official_tfd.py` | `python analysis/analyze_official_tfd.py` | 官方参数与扩散实现 | `artifacts/tfd_core/*` |
| 绘图 | `plotting/*.py` | `python plotting/<name>.py` | `results/tables/*.csv` | `results/figures/*.png` |

## 说明

- **说明文字：** 本 README、依赖说明和忽略规则使用中文。`official/README.md`、GPL 许可证
  以及官方源码注释保持上游原文，便于核验来源；实验脚本中的原始技术注释也不做批量翻译，
  避免产生与实验无关的大范围源码差异。
- **可移植性修正：** `scripts/` 与 `analysis/` 中原有的服务器绝对路径已替换为
  `REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`，使脚本从仓库根目录
  定位 `official/` 和输出目录。`wifi_train.py` 增加了与 README 一致的 `--max_iter`、
  `--batch_size` 参数；`wifi_smoke.py` 仅修正 checkpoint 保存目标变量名。这些修正不改变
  模型、损失函数、采样过程或正式实验设计。
- **正式终端分析：** 仓库只保留 `terminal_state_corrected.py`。早期脚本将
  `alpha_bar` 描述为“信息保留率”，该表述缺少依据，因此不作为正式复现入口。
- **输出：** 实验脚本写入 `experiments/`、`artifacts/`、`runs/` 与 `logs/` 下；绘图脚本
  读取 `results/tables/` 并写入 `results/figures/`。这些目录已被 git 忽略（运行时重新生成）。
- **不提交任何数据 / checkpoint。** 本仓库只包含代码；老师需要先下载 `dataset.zip` 与
  `model.zip`（见“数据准备”和“模型权重准备”）才能重新运行。
- **随机种子：** 官方 `degrade_fn` 硬编码了 `torch.manual_seed(11)`；源交换 / 终端实验
  各自设定自己的种子（`SEED = 42`），详见各脚本。

## 许可证 / 归属

`official/` 源码以 **GPL v3**（`official/LICENSE`）发布，是 RF-Diffusion 作者的
未经修改的原作。我们的实验、分析与绘图脚本仅供复现/评估使用，并非对该算法的重新实现。
