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
- 特征与谱面 JSON 的 LRU 缓存；构造数据集时过滤超长音频、缺失特征和超出
  `model.max_tokens` 的窗口，并在训练启动日志中报告跳过原因，而不是在
  DataLoader worker 里中途崩溃。
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
python -m pip install -r requirements-train.txt
# 等价于：python -m pip install -e '.[audio,web,train]'
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

也可以直接从 AWMC-MaiArchive（`download.wmc.pub`）批量获取谱面和音频：

```bash
# 先 dry-run 看匹配结果；正式下载默认跳过已存在的 wmc_<shortid>
python -m mainote dataset fetch-wmc -q "Project SEKAI" --limit 50 --dry-run
python -m mainote dataset fetch-wmc -q "Project SEKAI" --limit 50 --align

# 已知 shortid 时精确下载；--include-video 会显著增大体积
python -m mainote dataset fetch-wmc --id 011661
```

每首歌以 `wmc_<shortid>` 入库；zip 缓存在 `out/wmc-cache/`，可用 `--delay` 控制请求间隔。
Web UI 数据集页提供同样的在线检索、多选和后台导入进度。

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

# 只处理选中的歌曲
python -m mainote ml prepare-features data/processed --out data/features \
  --songs song_a,song_b

# 或者一步导出所选歌曲的样本 + 特征
python -m mainote ml prepare-data data/processed \
  --samples-out data/samples --features-out data/features \
  --songs song_a,song_b --require-audio
```

预期输出：

```text
data/samples/manifest.jsonl       # 每行一个 (song_id, inote_N)
data/samples/<song_id>/inote_N.json
data/features/<song_id>.npz       # mel, fps, sr, n_mels, duration
```

`manifest.jsonl` 的 `valid` 按谱面判定：`meta.json` 会汇总一首歌所有难度的校验错误，
但导出样本时会逐难度重新校验，只保留该难度自己的错误。一张坏谱不会让同歌的其他
难度失去训练资格；训练也只使用 `valid` 的行。

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

# 结构化 token 特征实验：从 baseline 权重热启动（只新增零初始化的投影）
python -m mainote ml train --config configs/p3_posdur_warm.yaml \
  --init-from runs/web_train/best.pt

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
- `note_eval.note_score` 是 onset F1、位置准确率、类型准确率、事件类型分布得分
  `type_ratio_score`（有 slide 时再加 slide 终点准确率）的加权平均。权重可用
  `note_eval.type_ratio_weight` 调整（默认 1.0，设为 0 退回旧公式）。
  `type_accuracy` 只看已匹配音符的类型，发现不了“大量 tap 被写成 slide”这类
  分布塌缩，因此 `metrics.jsonl` 同时记录 `type_ratios`、`type_ratio_l1`
  （两个分布的总变差距离 ×2）和 `type_ratio_score`（1 = 分布完全一致）。
  也可以把 `checkpoint_metric` 设为 `type_ratio_score` 单独按分布选模
  （该指标越大越好）。
- `note_eval` 每 `every_n_epochs` 个 epoch 在 `max_songs` 首歌、`max_windows`
  个窗口上解码一次；窗口数越大越慢。默认是 4 首歌 × 8 个窗口，且每个窗口的
  解码预算会按参考 token 数动态收紧（最多 `max_new_tokens`）。关闭时把
  `note_eval.enabled` 设为 false，并把 `checkpoint_metric` 设回 `val_loss`。
- 热启动（`--init-from`）会把 backbone LR 乘以 `warm_start_lr_scale`（默认 0.1），
  并把 `position_proj`/`duration_proj` 的 LR 乘以 `feature_lr_multiplier`
  （默认 1.0；结构化特征配置里设为 3.0）。这样零初始化的新投影能快速学习，
  同时不会在它们还接近零时洗掉预训练模型的类型先验。启动日志的 `lr_groups`
  和每步日志的 `lr`/`lr_feature` 会打印实际学习率。
- 同 tick 的多个事件按 `Event.sort_key()` 排序时 slide 永远排在 tap 前面，
  模型可能把“先 slide 后 tap”当成伪规律。`data.shuffle_same_tick_events: true`
  用每个窗口的稳定种子打乱同 tick 内的顺序，事件集合和 `T_` 时间结构不变。
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
- 不传 `--bpm` 时自动估计 BPM（165 BPM 谱面先验 + 节拍倍频候选）。`bpm_confidence`
  低于 0.2 时推理 summary 会给出 warning 和 `bpm_candidates` 前三名；节拍倍频
  歧义无法只靠音频完全消除，听感或窗口对不齐时仍应显式 `--bpm`。
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

`p3_summary.json` 还会写入 `type_counts` / `type_ratios`。如果生成谱面的 slide
占比超过 30% 且音符数 ≥ 50，`warnings` 会提示 checkpoint 可能发生了类型塌缩；
本地 master 谱面的 slide 先验约为 13.6%，全库约 9.1%。

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

怀疑模型发生类型塌缩（例如输出大量 slide、几乎没有 tap）时，先做
teacher-forcing 诊断，不必等慢速解码：

```bash
PYTHONPATH=src python scripts/diagnose_type_bias.py \
  --checkpoint runs/p3_posdur_warm/best.pt --split val
```

输出 gold / teacher-forced argmax 的类型比例、gold→predicted 混淆矩阵、
type accuracy，以及 `position_proj` / `duration_proj` 增量的平均范数和 token
embedding 范数。如果 teacher-forced argmax 本身就过度预测 slide，问题在
checkpoint 而不是采样温度/top-k。

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
- **结构化 token 特征（实验）：** `model.position_features` /
  `model.duration_features` 会在 decoder 输入侧，为 `P_*`、`S_*` 和 `DUR_*`
  附加固定的几何/时长特征（按钮 1–8 的环形 sin/cos、触控传感器所在环的
  one-hot、时长的 log/线性值），再经零初始化的线性投影加到 token embedding。
  词表、grammar、IR 和旧 checkpoint 都不变；两个开关默认关闭。启用后可用
  `configs/p3_posdur.yaml` 从头训练，或用 `configs/p3_posdur_warm.yaml`
  + `--init-from` 从 baseline 权重热启动。
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
| 输出以 slide 为主、几乎没有 tap | 运行 `scripts/diagnose_type_bias.py`；确认是 checkpoint 塌缩还是采样问题。热启动用 `warm_start_lr_scale`/`feature_lr_multiplier`，并保持 `shuffle_same_tick_events: true` |
| 推理校验失败 | 模型欠训练；延长训练或减小窗口，检查 `p3_ir.json` |
| 推理太慢 | 解码是逐 token 的（暂无 KV cache）；用 GPU、缩短窗口，或降低 `max_new_tokens` |
| `--resume` 报 optimizer 参数组不匹配 | 旧 checkpoint 只有一个参数组；改用 `--init-from` 继续，或重新训练 |

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

## 12. 本地 Web UI（可选）

不想记 CLI 参数时，可以用本地 Web UI 管理数据浏览、推理和训练：

```bash
pip install -e '.[audio,train,web]'
mainote web --no-browser --port 8765
```

- **浏览：** 自动扫描 `data/processed`、`out/demo/processed`、`out/audio-fm/dataset/processed`；
  其他位置用 `MAINOTE_WEB_DATA_ROOTS=/path/a:/path/b` 指定。
- **数据集管理：** 从 AWMC-MaiArchive 在线检索、多选下载并后台入库；也可以批量上传
  zip，一个压缩包视为一首歌，包内需要音频（优先 `track.*`）和 `maidata` 文件。
  解压后写入 `data/raw/<song_id>/`，再通过 `dataset ingest` 生成
  `data/processed/<song_id>/{meta.json,maidata.ir.json}`。
  可选 "分析 BPM / 对齐" 会对每首歌运行音频分析（更慢）。
  已入库列表会显示样本与 log-mel 特征状态；“补全样本与特征”会为所有缺失歌曲
  调用 `ml prepare-data`，也可以点单行的补全图标只处理一首。
- **预览：** 内嵌官方 MajdataView WebGL 构建，自动载入谱面、MP3 与背景图；
  流速和偏移在 MajdataView 自带设置中调整。
- **推理：** 选择 checkpoint/音频/难度/采样参数，等价于
  `mainote ml infer ...`；完成后可直接跳转到预览器。
- **训练：** 训练页可以直接从数据目录勾选歌曲，按歌曲重新划分 train/val（同一首歌
  不会跨集），并用“准备所选数据”调用 `ml prepare-data` 仅导出所选歌曲的样本与
  log-mel 特征。页面还提供超参数覆盖（优化器、数据窗口、模型宽度/层数）、实时
  网络层级图，以及 `dry run`/`max_steps` 快速验证。启动时会生成
  `runs/<run>/training_manifest.jsonl` 和 `runs/<run>/train_config.yaml`，不修改
  `configs/` 下的原文件。若只选一首歌，没有验证集，后端会自动关闭 note-level 评估
  并改用 `val_loss` 选择 checkpoint。
  任务面板会 tail 日志并绘制 `metrics.jsonl` 曲线；历史 run 可查看 checkpoint、
  `config.json` 和指标。

后端只允许启动 `mainote ml train|infer` 和 `ml prepare-data`，同一时间一个任务，
且所有路径必须位于仓库、`runs/` 或 `out/` 内。Web 界面本身不替代 GPU 服务器：训练仍应在有
NVIDIA GPU 的机器上运行。前端开发模式见 `README.md`。
