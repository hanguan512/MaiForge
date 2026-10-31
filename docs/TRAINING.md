# P3 训练指南

本文档面向在 GPU 服务器上运行神经音频转谱面模型（P3）的 Agent 或工程师。
**不要在当前工作区训练**：这里只有 CPU，真实训练请放到带 NVIDIA GPU 的服务器。
训练/推理代码位于 `src/mainote/ml/`。

## 1. 已实现内容

- Log-mel 特征提取（numpy）。
- 谱面事件 tokenizer 和语法约束解析器。
- 窗口数据集、按歌划分、SpecAugment。
- 音频编码器：Conv1d 前端 + Transformer encoder。
- 事件解码器：decoder-only Transformer + 对音频的 cross-attention。
- 难度条件：序列前缀 `DIFF_n`、`LEVEL_k`。
- 训练循环：AdamW、warmup + cosine、AMP、梯度累积、checkpoint、resume、
  验证、early stop、JSONL metrics。
- 语法约束的贪婪推理，输出并编译为 `maidata.txt`。
- 通过 `mainote eval` 做音符级评估（onset F1、位置/类型、时长、Slide 终点）。

尚未实现：MERT/PaSST 预训练特征、beam search、动态 BPM 推理、跨窗口
Hold/Slide 活跃状态、分布式训练。

## 2. 服务器环境

```bash
git clone git@github.com:hanguan512/MaiForge.git
cd MaiForge

python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[audio,train]' -i https://pypi.org/simple
```

要求：

- Linux + NVIDIA GPU（小 batch 8GB 可能够用；16GB 更稳）
- CUDA 版 PyTorch
- `ffmpeg` 在 `PATH` 中（音频解码）
- Python 3.11+

检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version | head -1
```

## 3. 数据准备

原始布局：

```text
data/raw/<song_id>/
  audio.mp3 | audio.wav | audio.flac | ...
  maidata.txt
  meta.json          # 可选：{"title": "...", "artist": "..."}
```

完整流水线：

```bash
# 1. 解析、校验、统计、可选对齐
python -m mainote dataset ingest data/raw --out data/processed --align

# 2. 查看每难度统计和对齐质量
python -m mainote dataset report data/processed --out out/dataset_report.json

# 3. 按歌划分（不要按谱面/窗口）
python -m mainote dataset split data/processed \
  --out out/splits.json --train 0.8 --val 0.1 --test 0.1 --seed 42

# 4. 每个 (歌曲, 难度) 一条训练样本
python -m mainote dataset samples data/processed \
  --out data/samples --splits out/splits.json --require-audio

# 5. 预计算 log-mel 特征（每首歌一个文件）
python -m mainote ml prepare-features data/processed --out data/features
```

预期输出：

```text
data/samples/manifest.jsonl       # 每行一个 (song_id, inote_N)
data/samples/<song_id>/inote_N.json
data/features/<song_id>.npz       # mel, fps, sr, n_mels, duration
```

训练前检查：

```bash
python - <<'PY'
import json
rows=[json.loads(l) for l in open("data/samples/manifest.jsonl") if l.strip()]
print("rows", len(rows))
print("splits", {s: sum(r["split"]==s for r in rows) for s in ("train","val","test")})
print("invalid", sum(not r["valid"] for r in rows))
print("low_align", sum((r.get("alignment_confidence") or 0) < 0.5 for r in rows))
PY
```

如果 `val` 为空，训练会退回用 train loss 选择 checkpoint；正式训练前请创建
真正的验证集。

## 4. 训练

```bash
python -m mainote ml train --config configs/p3_base.yaml
```

常用选项：

```bash
# 快速冒烟：构建一个 batch，跑一次 forward
python -m mainote ml train --config configs/p3_base.yaml --dry-run

# 单 batch 调试
python -m mainote ml train --config configs/p3_base.yaml --limit-batches 1

# 恢复训练
python -m mainote ml train --config configs/p3_base.yaml \
  --resume runs/p3_base/last.pt

# 指定 GPU、短跑
python -m mainote ml train --config configs/p3_base.yaml \
  --device cuda --max-steps 2000
```

输出在 `runs/p3_base/`：

```text
config.json        # 解析后的 YAML 快照
metrics.jsonl      # train/val loss、token accuracy、LR
last.pt            # 每个 epoch
best.pt            # 最佳验证 loss
epoch_NNN.pt       # 每 save_every 个 epoch
```

观察重点：

- `train_loss` / `val_loss` 应该下降；train 降、val 升时增加数据或增强正则。
- token accuracy 只是参考，不等于谱面质量；最终要看 `mainote eval` 和试听。
- 规则 baseline 仍然是对照组：只学会密度的模型并不算更好。

## 5. 推理

```bash
python -m mainote ml infer \
  --checkpoint runs/p3_base/best.pt \
  --audio song.mp3 \
  --difficulty master --level 12.9 \
  -o out/p3/maidata.txt
```

可选参数：

- `--timing out/timing.json` — 复用/修正 `mainote analyze` 的 BPM 和 `first`
- `--features data/features/<song_id>.npz` — 复用预计算特征
- `--temperature 0.8 --top-k 20` — 采样而不是贪婪
- `--window-bars 16 --stride-bars 8 --context-bars 2`
- `--device cuda`

推理输出：

```text
out/p3/maidata.txt
out/p3/timing.json
out/p3/p3_ir.json
out/p3/p3_summary.json
```

推理默认使用语法约束的贪婪解码。合并后的谱面会经过校验；如果非法会写
`p3_ir.json` / `p3_validation.json` 并报错，方便排查。

## 6. 评估

```bash
python -m mainote eval out/p3/maidata.txt reference.txt --index 5
```

公平比较的要求：

- 同一音频、同一难度、同一 `first`；
- 只用 test split 的歌曲；
- 报告 onset F1、类型/位置准确率、时长误差和难度统计；
- 同时对比规则 baseline（`mainote generate`）和人工试听。
  只看 token loss 不能判断谱面质量。

## 7. 架构说明

- **时间单位：** 20 ticks（1/48 拍）；4/4 小节 192 单位。
- **Token 语法：** `<BOS> <DIFF_n> <LEVEL_k> <BAR>* <T_n> <TYPE> <POS> ... <END_EVENT> <EOS>`。
- **时长：** 1/48 拍单位，最大 2048（约 42 拍）。
- **Slide：** 支持 single、same-start（`*`）和 connected path；track break/EX 会 token 化。
- **音频：** 80-bin log-mel，1024 FFT，256 hop（约 86 fps），按歌归一化。
- **模型：** Conv1d ×2 + 4 层 Transformer encoder；6 层 decoder + cross-attention；
  token embedding；输出投影与 embedding 绑定。
- **窗口：** 目标窗口 16 小节，stride 8，音频上下文 2 小节。v0 不把窗口开始前
  已经活跃的 Hold/Slide 输入模型。
- **划分：** 只能按歌划分；同一首歌的所有难度必须在同一个 split。

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `No module named torch` | 在 GPU 服务器安装 `.[train]` |
| `training dataset is empty` | 检查 manifest 的 `split`、`valid`、`has_audio` 和 features |
| `missing features for <song>` | 运行 `mainote ml prepare-features` |
| CUDA OOM | 降低 `batch_size`、`window_bars` 或 `max_audio_frames` |
| DataLoader worker 卡住 | 设置 `train.num_workers: 0` |
| val loss 不降 | 增加数据、降低 LR、增加 dropout、检查对齐 |
| 推理校验失败 | 模型欠训练；延长训练或减小窗口，检查 `p3_ir.json` |
| 推理太慢 | 贪婪解码是逐 token 的；用 GPU、缩短窗口，后续可加 beam/cache |

## 9. 数据规模

- 100 首歌 × 4–5 个难度是合理的首个 PoC。
- 300+ 首歌 / 1200+ 张谱是更可用的 v0。
- 比数量更重要的是：音频版本正确、`first`/BPM 正确、谱师/风格多样、
  难度均衡、对齐噪声低。
- `alignment_confidence < 0.5` 的样本应人工检查或剔除。
- 不要按谱面或窗口划分，否则同一段音频会泄漏到 train/test。

## 10. 当前限制

- 推理目前是恒定 BPM（训练特征使用谱面 tempo map）。
- 还没有 MERT/PaSST 预训练音频编码器。
- 还没有 beam search / candidate reranking。
- 跨窗口的 Hold/Slide 活跃状态尚未传入模型，长音符可能被截断。
- 本仓库没有训练好的 `best.pt`；必须在 GPU 服务器上运行训练后才会产生。
