# EEG 6-Words Inner Speech Dataset — 资源归档

论文 **"EEG-based brain-computer interface (BCI) dataset for directional word recognition"**
（*Scientific Data*, vol. 13, article 1195）的 data / code / paper 完整归档。

| 项目 | 内容 |
|---|---|
| 论文 DOI | [10.1038/s41597-026-07809-9](https://doi.org/10.1038/s41597-026-07809-9) |
| 数据 DOI | [10.5281/zenodo.20374418](https://doi.org/10.5281/zenodo.20374418)（CC-BY-4.0） |
| 代码 | [github.com/AvedikEkiz/inner-speech-bci](https://github.com/AvedikEkiz/inner-speech-bci) @ `649ec92` |
| 归档日期 | 2026-08-31 |

## 目录结构

```
20260831_reproduce_20260817_kostulin/
├── paper/                          # 论文
│   └── s41597-026-07809-9.pdf     # 正文 PDF（13 页）
├── data/                           # 数据（Zenodo）
│   ├── Inner_Speech_Dataset.zip    # 原始下载包（4.97 GB，MD5 见下）
│   └── inner_speech_v2/            # 解压后内容
├── code/                           # 分析代码（GitHub clone）
│   └── inner-speech-bci/
├── experiments/                    # 论文复现 + 深度模型对比（本仓库新增）
│   ├── common.py                   # 数据加载、55-65Hz SPM+coherence 特征、统一 CV 划分
│   ├── ml_paper.py                 # 论文三分类器复现：RF / 线性SVM / LDA
│   ├── dl_models.py                # GRU / LSTM / ViT（架构移植自 20260703_Preprocess）
│   ├── dl_paper.py                 # 深度模型训练（与 ML 完全相同的 5 折划分）
│   ├── summarize.py                # 汇总对比 + 作图
│   └── results/                    # ml_results.csv / dl_results.csv / summary.csv/.png
└── README.md
```

## 数据集简介

22 名健康被试（12 名俄语母语者 + 10 名西班牙语母语者）在**外显（overt）与内隐（inner/covert）**
两种条件下说出/想象 6 个方向词（上、下、左、右、前、后）的多导联 EEG 数据。

- **采集**：Neurovisor-BMM-52 (NVX) 系统，500 Hz，38 个 EEG 电极（10-10 系统），部分被试同步采集 EMG（咬肌/喉部）
- **任务**：6 词 × overt/covert 两条件，每词约 110±20 trials，每被试约 70 分钟连续记录
- **预处理**（已随数据提供）：1–70 Hz 带通、50 Hz 陷波、ICA 去伪迹

### 数据内容（`data/inner_speech_v2/`）

```
inner_speech_v2/
├── readme.md                  # 作者提供的数据说明
├── manifest.csv               # 文件清单
├── data_dictionary.md         # 字段字典
├── raw/                       # 原始连续记录
│   ├── russian/sub1..sub12/sub*_session.edf
│   └── spanish/sub0..sub9/sub*_session.edf
├── preprocessed/              # 预处理 epochs
│   ├── russian/sub1..sub12/{sub*_epochs.fif, sub*_events.xlsx}
│   └── spanish/sub0..sub9/{sub*_epochs.fif, sub*_events.xlsx}
└── metadata/subject_metadata.json   # 年龄、母语等被试信息
```

- `.edf`：原始连续 EEG/EMG（含事件标记）
- `.fif`：MNE Epochs（trials × 38 通道 × 750 时间点）
- `.xlsx`：trial 时间戳与条件标签（后缀 `1`=overt，`2`=inner；`GO`/`GZ` 为 block 开始/休息标记）

## 代码使用

```bash
cd code/inner-speech-bci
pip install -r requirements.txt   # mne, numpy, pandas
```

```python
import utils
from pathlib import Path

utils.DEFAULT_BASE = Path(r"D:/repositories/20260831_reproduce_20260817_kostulin/data/inner_speech_v2")
# 详见 example_usage.ipynb 与 utils/loader.py、utils/labels.py
```

## 校验信息

| 文件 | 大小 (bytes) | MD5 |
|---|---|---|
| `data/Inner_Speech_Dataset.zip` | 4,972,969,427 | `5ce82e9b48fd441b136eea141c45f769`（Zenodo 官方值） |

## 复现实验：论文三分类器 + GRU/LSTM/ViT 对比（2026-08-31）

任务：三分类 rest(GZ) / overt(词\*1) / inner(词\*2)，逐被试 CV。运行环境：conda env
`decoder`（mne 1.11 / sklearn 1.7 / torch 2.10+cu128）。特征：55–65 Hz Welch 谱功率
（对数）+ 通道两两相干。epochs 统一为以 marker 为中心的 1 s 窗（不足补零），深度模型
输入 (N, C, 500) 逐通道 z-score；AdamW 1e-3，早停 patience=8。

### 交叉验证：leave-one-session-out（主结果，防泄露）

数据集内相邻 trial 高度相关（同一段内的漂移、伪迹、连续发音），随机 5 折会把
时间相邻的 trial 同时划入训练/测试造成泄露。因此按 marker 时间流做时间分段
（自适应 gap 阈值 60→10 s，段数 6–10；某类只出现在单一时段时拆为 3 子段），
每次留一整段作测试。ML 与 DL 使用完全相同划分。

结果（22 被试平均准确率 / 平衡准确率）：

| 模型 | 准确率 | 平衡准确率 | 论文报告值 |
|---|---|---|---|
| **LDA** | **75.1 ± 14.2%** | 74.1% | 67 ± 6% |
| SVM (linear) | 73.4 ± 12.2% | 72.3% | 74 ± 5% |
| GRU (2 层双向, 0.43M) | 73.2 ± 16.1% | 72.3% | — |
| LSTM (2 层双向, 0.57M) | 70.9 ± 17.7% | 70.1% | — |
| RandomForest | 70.6 ± 11.1% | 66.5% | 78 ± 4% |
| ViT (dim128 depth4, 0.66M) | 60.5 ± 14.3% | 59.2% | — |

与随机 5 折（存在泄露，`results/*_stratified5.csv`）的对比：所有模型下降 10–13 个
百分点（LDA 87.9→75.1、SVM 85.7→73.4、RF 85.3→70.6、GRU 85.2→73.2、LSTM
83.6→70.9、ViT 71.8→60.5），且 LOSO 数值与论文报告值（78/74/67）更接近——说明
论文的评估可能同样带有一定时间泄露，或我们的特征更充分。

结论：消除泄露后浅层与深层方法差距缩小（LDA 75.1 vs GRU 73.2），LDA 仍最优；
深度模型方差更大（±16–18%），小样本下不稳定；ViT 依然最弱。复现步骤：

```bash
cd experiments
python ml_paper.py    # RF / SVM / LDA（leave-one-session-out）
python dl_paper.py    # GRU / LSTM / ViT（GPU，同划分）
python summarize.py   # 汇总 + summary.png
```

## 说明

- **Supplementary**：该论文在 nature.com 上未提供独立的补充材料文件（页面无 Supplementary
  Information 章节及 MOESM 附件），全部图表包含在 `paper/` 的正文 PDF 中，故本仓库无 `supplementary/` 目录。
- 数据与论文均为 CC-BY-4.0 开放许可；引用要求见原文 Rights and permissions。

> 改名记录：2026-10-09 按顶层命名规范 yyyymmdd_reproduce_yyyymmdd_author 自 `20260831_EEG_6_Words` 改名（第二个日期为论文公开发表日，一作 Kostulin（EEG directional word recognition, Sci Data 2026-08-17）），引用路径已全量同步。
