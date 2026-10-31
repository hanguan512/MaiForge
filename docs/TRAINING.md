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
- 难度条件：序列前缀 `DIFF_n`、`LEVEL_k`（`15+` 映射到 bucket 31）。
- 训练循环：AdamW、warmup + cosine、AMP、梯度累积、checkpoint、resume、
  验证、early stop、JSONL metrics。
- 训练目标：按窗口归一化的 token 交叉熵（每张谱面等权），可配置
  `token_class_weights` 提高 type/structure/position 的权重，缓解 86% 词表是
  时长 token 的偏置。键可以是类别（`type`、`structure`、...），也可以是精确
  token（如 `SLIDE`、`TOUCH`、`TOUCH_HOLD`）；精确 token 优先于所属类别。
- 验证指标：总 token accuracy + 按 token 类别（time/type/position/duration/
  structure/flag/condition）分组的 accuracy。
- 谱面级验证：每 N 个 epoch 在少量 val 窗口上真实解码，用 onset F1、位置/类型
  准确率、slide 终点准确率计算 `note_score`，默认用它选择 `best.pt`。
- 语法约束推理：默认采样解码（temperature 0.8 + top-k 20）；`temperature=0`
  仍可贪婪解码。严格模式禁止同 tick 同位置重复、活跃 Hold 内新事件、直线
  slide 起终点相同。
- 手动 BPM/first 校正，以及 timing JSON 中的分段 `bpm_changes` tempo map。
- 特征与谱面 JSON 的 LRU 缓存；超长音频窗口直接报错而不是静默中心裁剪。
- 通过 `mainote eval` 做音符级评估（onset F1、位置/类型、时长、Slide 终点）。

尚未实现：MERT/PaSST 预训练特征、辅助损失（onset / duration / 难度画像）、beam
search 或候选重排、自动 tempo map 估计（可手动提供）、KV cache、跨窗口
Hold/Slide 活跃状态、分布式训练。

> 阶段编号采用仓库现状（P2 = 数据集/评估，P3 = 神经模型），与设计计划早期版本的
> P2/P3 划分不同。

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

如果 `val` 为空，`note_eval` 无法运行；配置里的 `checkpoint_metric: note_score`
会直接报错。正式训练前请创建真正的验证集，或把 `checkpoint_metric` 改为
`val_loss`（此时会退回用 train loss 选择 checkpoint，不建议用于正式训练）。

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
metrics.jsonl      # train/val loss、分类别 token accuracy、note eval、LR
last.pt            # 每个 epoch
best.pt            # checkpoint_metric 最佳（默认 note_score）
epoch_NNN.pt       # 每 save_every 个 epoch
```

观察重点：

- `train_loss` 是按窗口平均的加权 loss；`train_token_ce` 是未加权 token CE，
  用于和旧实验对比。两者都应与 `val_loss` 一起看。
- `val_class_accuracy` 中 `duration` 通常虚高（时长 token 占 86%），重点看
  `type`、`position`、`structure`。
- `note_eval.note_score` 是 onset F1、位置准确率、类型准确率和 slide 终点准确率
  （有 slide 时）的平均值。它比 token accuracy 更接近真实谱面质量。
- `note_eval` 每 `every_n_epochs` 个 epoch 在 `max_songs` 首歌、`max_windows`
  个窗口上解码一次；窗口数越大越慢。关闭时把 `note_eval.enabled` 设为 false，
  并把 `checkpoint_metric` 设回 `val_loss`。
- 如果 `max_steps` 小于 `warmup_steps`，warmup 会自动截断到总步数的 10%，并在
  启动日志中记录 `warmup_clamped_from`；短跑结果不能代表正式训练。
- 规则 baseline 仍然是对照组：只学会密度的模型并不算更好。
- `DIFF_n` / `LEVEL_k` 前缀不能保证目标密度或音符类型比例；请在生成后用
  `mainote stats` 检查，后续应加入 `L_level` 辅助损失。

## 5. 推理

```bash
python -m mainote ml infer \
  --checkpoint runs/p3_base/best.pt \
  --audio song.mp3 \
  --difficulty master --level 12.9 \
  -o out/p3/maidata.txt
```

可选参数：

- `--timing out/timing.json` — 复用/修正 `mainote analyze` 的 BPM 和 `first`；
  JSON 可额外包含 `"bpm_changes": [{"tick": 3840, "bpm": 180}, ...]`
- `--bpm 180 --first 0.32` — 直接覆盖 BPM / 起始时间（会清空 timing 里的
  tempo map）
- `--features data/features/<song_id>.npz` — 复用预计算特征
- `--temperature 0` — 贪婪解码；默认是采样解码（`temperature 0.8`、`top-k 20`）
- `--window-bars 16 --stride-bars 8 --context-bars 2`
- `--device cuda`

推理输出：

```text
out/p3/maidata.txt
out/p3/timing.json
out/p3/p3_ir.json
out/p3/p3_summary.json
```

推理默认使用语法约束的采样解码。严格语法禁止同 tick 同位置重复、活跃 Hold 内
新事件和起终点相同的直线 slide；跨窗口重叠仍由 `_repair_events` 兜底。合并后的
谱面会经过校验；如果非法会写 `p3_ir.json` / `p3_validation.json` 并报错，方便
排查。若某个窗口的 mel 帧数超过 `model.max_audio_frames`，`p3_summary.json` 的
`warnings` 会提示编码器将截断尾部音频。

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
- **模型 v2：** Conv1d 前端（默认总 stride 2，约 43 fps 的 encoder 时间分辨率）
  + 4 层 pre-norm Transformer encoder（末尾 LayerNorm）；6 层 decoder +
  cross-attention（末尾 LayerNorm）；token embedding 乘以 `sqrt(hidden)` 后再加
  正弦位置编码，避免 PE 幅度约为 token 内容的 25 倍；输出投影与 embedding 绑定。
- **兼容性：** v2 与 v1 checkpoint 不兼容（新增 final LayerNorm、前端结构和
  embedding 尺度不同），必须重新训练；加载旧 checkpoint 会给出明确错误。
- **窗口：** 目标窗口 16 小节，stride 8，音频上下文 2 小节；窗口音频超过
  `max_audio_frames` 会报错，不会静默裁剪。v0 仍不把窗口开始前已经活跃的
  Hold/Slide 输入模型。
- **划分：** 只能按歌划分；同一首歌的所有难度必须在同一个 split。

## 8. 故障排查

| 现象 | 处理 |
|---|---|
| `No module named torch` | 在 GPU 服务器安装 `.[train]` |
| `training dataset is empty` | 检查 manifest 的 `split`、`valid`、`has_audio` 和 features |
| `missing features for <song>` | 运行 `mainote ml prepare-features` |
| `exceeding max_audio_frames` | 减小 `window_bars`/`context_bars`，或同时提高 data/model 的 `max_audio_frames` |
| CUDA OOM | `audio_stride: 2` 比 v1 更耗显存；降低 `batch_size`、`window_bars` 或改回 `audio_stride: 4`（需重训） |
| DataLoader worker 卡住 | 设置 `train.num_workers: 0` |
| val loss 不降 | 增加数据、降低 LR、增加 dropout、检查对齐 |
| checkpoint 无法加载 | v1 与 v2 架构不兼容，必须用当前配置重训 |
| `note_eval` 太慢 | 减少 `max_songs`/`max_windows`/`max_new_tokens`，或调大 `every_n_epochs` |
| 推理校验失败 | 模型欠训练；延长训练或减小窗口，检查 `p3_ir.json` |
| 推理太慢 | 解码是逐 token 的（暂无 KV cache）；用 GPU、缩短窗口，或降低 `max_new_tokens` |

## 9. 数据规模

- 100 首歌 × 4–5 个难度是合理的首个 PoC。
- 300+ 首歌 / 1200+ 张谱是更可用的 v0。
- **必须覆盖 13、14、15、15+。** 没有高难度谱面，规则基线和 P3 都学不到高难度
  长什么样；不要用 ≤12.9 的数据声称支持 13–15/15+。
- 把上述规模当作硬性门槛：PoC 数据不达标就不要启动 P3 训练。
- 比数量更重要的是：音频版本正确、`first`/BPM 正确、谱师/风格多样、
  难度均衡、对齐噪声低。
- `alignment_confidence < 0.5` 的样本应人工检查或剔除。
- 不要按谱面或窗口划分，否则同一段音频会泄漏到 train/test。

## 10. 当前限制

- 自动 tempo map 估计尚未实现；默认仍是恒定 BPM，但可以用 `--timing` JSON 的
  `bpm_changes` 或 `--bpm`/`--first` 手动校正。
- 还没有 MERT/PaSST 预训练音频编码器。
- 还没有 beam search / candidate reranking；默认采样，`temperature=0` 为贪婪。
- 解码还没有 KV cache，逐 token 重跑 decoder。
- 跨窗口的 Hold/Slide 活跃状态尚未传入模型；严格语法只在单个窗口内生效，
  跨窗口重叠仍靠 `_repair_events` 丢弃/修补。
- 训练损失仍是 token 交叉熵（已加权、按窗口归一化），还没有 onset / duration /
  `L_level` 辅助损失。
- 输入只有 80-bin log-mel；还没有 onset/beat/downbeat 辅助特征或节拍同步池化。
- 数据增强目前只有 SpecAugment；EQ、噪声、混响、时间拉伸等尚未实现。
- tokenizer 支持到 15+（bucket 31）；16.0 以上会 clamp 到 15+。
- 难度先验还没有用真实数据校准，高等级即使训练后也可能达不到目标密度/复杂度。
- 本仓库没有训练好的 `best.pt`；必须在 GPU 服务器上运行训练后才会产生。

## 11. Go/No-Go 检查清单

在把默认生成器从规则 baseline 切换为 P3 之前，逐项确认：

- [ ] 生成的谱面 100% 可编译、可重新解析。
- [ ] 非法事件为 0；自动修复事件 < 2%。
- [ ] 在留出歌曲上，onset F1、位置/类型准确率不低于规则 baseline。
- [ ] notes/s、最大同时音、类型比例落在目标难度的训练集分位区间。
- [ ] 测试集包含 13 / 14 / 15 / 15+ 谱面。
- [ ] 人工盲评中，P3 相对规则 baseline 的胜率 > 50%。
- [ ] 跨窗口长音符截断率和重复率已经统计并在可接受范围。

任何一项不满足，就继续把 `mainote generate`（规则 baseline）作为默认生成器。
