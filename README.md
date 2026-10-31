# MaiForge

**面向 maimai DX / simai 的音频 → maidata 谱面生成工具。**

MaiForge 会先把音频转换成结构化的事件 IR，再确定性地编译为 `maidata.txt`，从而生成 simai 谱面。仓库目前包含 P0–P2 的基础能力，以及 P3 神经模型的 v0 训练/推理代码：完整的格式工具链、音频时间轴前端、规则基线、数据集流水线、窗口化训练管线和音符级评估。P3 v0 尚未训练（默认采样解码、支持手动 tempo map、无跨窗口 Hold/Slide 状态），训练需要 GPU 服务器和数据。

> **命名说明：** MaiForge 是项目名称。Python 包和命令行工具目前仍名为 `mainote`，后续计划重命名。

## 为什么选择 MaiForge？

直接生成 `maidata` 文本很脆弱：逗号、`{n}` 分割、`[a:b]` 时长、滑动路径以及结尾的 `E` 都很容易写错。MaiForge 把问题拆成两层：

```text
音频 → 时间轴 → 事件 IR → 确定性编译器 → maidata.txt
```

- 生成器只负责决定**是什么音符、在哪个位置、什么时候出现、持续多久**。
- 编译器负责**格式化**：逗号、分割数、时长分数、元数据顺序和 `E`。
- 每份输出都能被重新解析，并通过往返测试校验。

这样一来，即使音乐生成层仍在开发中，格式层也能保持可靠。

## 功能特性

- 支持多难度谱面的 **maidata 解析器 / IR / 编译器 / 校验器**。
- 支持 Tap、Hold、Slide、Touch、Touch Hold、Break、EX、烟花（fireworks）、注释、BPM 变化、按难度设置的 `first`、`*` 滑动链以及连通的路径滑动。
- **往返保证：** IR → maidata → IR 的完整往返由测试覆盖。
- **音频前端：** ffmpeg 解码、numpy 起音检测、BPM/节拍/强拍跟踪，并支持通过 `--bpm` / `--first` 手动修正。
- **规则基线：** 根据难度和等级条件，从音频生成合法的谱面。
- **数据集流水线：** 数据收录、音频与谱面对齐评分、按难度统计、无数据泄漏的歌曲级划分，以及按难度导出样本。
- **P3 v0 训练/推理代码：** log-mel 特征 + 音频编码器 + 事件 Transformer（架构 v2）、窗口数据集、难度条件、严格语法约束的采样/贪婪解码和 checkpoint 推理（尚未训练）。
- **评估：** 起音 F1、位置/类型准确率、时长误差、滑动端点准确率以及难度画像差异。
- **合成演示：** 不依赖任何受版权保护的音频，即可跑通完整的音频 → maidata 流程。

## 当前状态

| 阶段 | 范围 | 状态 |
|---|---|---|
| P0 | 解析器、IR、编译器、校验器、统计、往返测试 | 已完成 |
| P1 | 音频时间轴前端、规则基线、端到端 CLI | 已完成 |
| P2 | 数据集收录/对齐/报告/划分/样本、评估 | 已完成 |
| P3 | 神经音频编码器 + 事件解码器 | v0 训练/推理代码已就绪；未训练 |

本仓库中没有任何已训练的神经模型检查点。当前的默认生成器仍是规则基线：它会跟随检测到的起音/节拍和目标难度画像，但并不理解旋律或谱面风格。P3 代码位于 `src/mainote/ml/`，训练方法见 [`docs/TRAINING.md`](docs/TRAINING.md)。P3 v0 仍缺少辅助损失、beam/候选重排、跨窗口状态和动态 BPM；在留出歌曲上稳定超过规则基线之前，不会设为默认生成器。

## 快速开始

### P0 工具链（无第三方依赖）

在仓库根目录执行：

```bash
PYTHONPATH=src python3 -m mainote parse dataset/xxdw/maidata --json out/sample.ir.json
PYTHONPATH=src python3 -m mainote validate dataset/xxdw/maidata
PYTHONPATH=src python3 -m mainote stats dataset/xxdw/maidata
PYTHONPATH=src python3 -m mainote roundtrip dataset/xxdw/maidata
PYTHONPATH=src python3 -m mainote eval dataset/xxdw/maidata dataset/xxdw/maidata --index 5
```

把 IR 重新编译为 maidata：

```bash
PYTHONPATH=src python3 -m mainote compile out/sample.ir.json -o out/roundtrip.txt
```

### 完整环境

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[audio]'   # numpy + mutagen
# 如果 pip 无法解析依赖，请加上：-i https://pypi.org/simple
```

之后使用 `.venv/bin/python` 运行音频和数据集对齐相关命令。

```bash
# 分析
PYTHONPATH=src .venv/bin/python -m mainote analyze song.mp3 --out out/timing.json

# 生成
PYTHONPATH=src .venv/bin/python -m mainote generate song.mp3 \
  --difficulty master --level 12.9 -o out/maidata.txt
```

密度默认由等级先验决定；可以用 `--density 5.2` 直接指定 notes/s，或用
`--density-scale 1.3` 按倍数调整（例如 13.5 默认偏保守时可放大）。

如果 BPM 或强拍检测有误，可以手动覆盖后重新生成：

```bash
PYTHONPATH=src .venv/bin/python -m mainote analyze song.mp3 \
  --bpm 160 --first 0.0 --out out/timing.json
PYTHONPATH=src .venv/bin/python -m mainote generate song.mp3 \
  --timing out/timing.json --difficulty master --level 12.9 -o out/maidata.txt
```

### 合成演示

无需自备音频文件：

```bash
PYTHONPATH=src .venv/bin/python scripts/make_demo.py out/demo
```

该命令会在 `out/demo/` 下写入点击音轨、时间轴 JSON、生成的谱面、IR 以及一个小型原始数据集。

## CLI 概览

```text
mainote parse <maidata> [--json out.ir.json]
mainote compile <ir.json> -o <maidata.txt>
mainote validate <maidata|ir.json> [--json]
mainote stats <maidata|ir.json> [--json]
mainote roundtrip <maidata> [--out recompiled.txt]
mainote eval <generated> <reference> [--index N] [--tolerance 0.06]
mainote analyze <audio> --out timing.json [--bpm N] [--first S]
mainote generate <audio> -o maidata.txt [--difficulty master] [--level 12.9] [--density N] [--density-scale 1.3] [--timing timing.json]
mainote demo [--out out/demo]
mainote dataset ingest <raw_dir> --out <processed_dir> [--align]
mainote dataset report <processed_dir> [--out report.json] [--csv report.csv]
mainote dataset split <processed_dir> --out splits.json [--train 0.8 --val 0.1 --test 0.1]
mainote dataset samples <processed_dir> --out samples_dir [--splits splits.json] [--require-audio]
mainote ml prepare-features <processed_dir> --out <features_dir>
mainote ml train --config configs/p3_base.yaml [--resume <checkpoint>] [--dry-run]
mainote ml infer --checkpoint <best.pt> --audio <song.mp3> -o <maidata.txt>
```

## 架构

```text
音频
  │
  ├─ 解码 / 单声道 / 重采样                  ffmpeg + numpy
  ├─ 起音、BPM、节拍、强拍                    audio/features.py
  │
  ├─ 时间轴映射（BPM、first、小节、起音）      audio/timing.py
  │
  ├─ 生成器
  │    ├─ P1：规则基线                        generate/rule.py
  │    └─ P3：神经事件解码器 v0               已就绪，未训练
  │
  ├─ 事件 IR                                  maidata/model.py
  │    ├─ tick（每拍 960，4/4 小节 3840）
  │    ├─ 类型 / 位置 / 标志 / 时长
  │    └─ 滑动链（same_start 或 path）
  │
  ├─ 校验                                      maidata/validate.py
  ├─ 确定性编译器                              maidata/compiler.py
  └─ maidata.txt
```

模型面向的表示是事件 IR，而不是原始 maidata 文本。编译器会为每个小节选择能对齐所有事件和 BPM 变化的最小 `{n}`，然后写出逗号、`[a:b]` 时长、`&lv`/`&des` 元数据和 `E`。

## 数据集流水线

原始数据布局：

```text
data/raw/<song_id>/
  audio.mp3 | audio.wav | audio.flac | ...
  maidata.txt
  meta.json            # 可选：{"title": "...", "artist": "..."}
```

收录并检查：

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset ingest data/raw \
  --out data/processed --align
PYTHONPATH=src .venv/bin/python -m mainote dataset report data/processed \
  --out out/dataset_report.json
```

创建歌曲级划分和按难度样本：

```bash
PYTHONPATH=src .venv/bin/python -m mainote dataset split data/processed \
  --out out/splits.json --train 0.8 --val 0.1 --test 0.1 --seed 0

PYTHONPATH=src .venv/bin/python -m mainote dataset samples data/processed \
  --out data/samples --splits out/splits.json --require-audio
```

`dataset samples` 会为每首歌保留一份 IR，同时为每个难度写出一份自包含的训练样本：

```text
data/samples/
  manifest.jsonl          # 每行对应一个 (song_id, inote_N)
  summary.json
  splits.json
  <song_id>/
    inote_1.json ... inote_5.json
```

同一首歌的所有难度必须留在同一个划分中。切勿按谱面或窗口划分，否则同一段音频会泄漏到训练集和测试集。manifest 记录以下字段：`sample_id`、`chart_index`、`difficulty`、`level_num`、`audio`、`ir`、`first`、`initial_bpm`、`event_count`、`stats`、`alignment_confidence`、`valid` 和 `split`。

## P3 神经模型训练

P3 需要 PyTorch 和 GPU。安装训练依赖：

```bash
pip install -e '.[audio,train]' -i https://pypi.org/simple
```

预计算 log-mel 特征，然后训练：

```bash
python -m mainote ml prepare-features data/processed --out data/features
python -m mainote ml train --config configs/p3_base.yaml
```

训练输出在 `runs/p3_base/`，包括 `best.pt`、`last.pt`、`metrics.jsonl`。推理：

```bash
python -m mainote ml infer \
  --checkpoint runs/p3_base/best.pt \
  --audio song.mp3 \
  --difficulty master --level 12.9 \
  -o out/p3/maidata.txt
```

> P3 v0 当前默认采样解码（`temperature=0` 可贪婪）、支持手动 tempo map、无跨窗口 Hold/Slide 状态；训练使用按窗口归一化的加权 token 交叉熵，并用音符级 `note_score` 选择 checkpoint，尚无候选重排和难度画像辅助损失。它还不能替代规则基线作为默认生成器。

完整的数据准备、训练、推理、评估和故障排查见 [`docs/TRAINING.md`](docs/TRAINING.md)。

### 训练数据门槛

不要在仓库自带的单首 fixture 上训练 P3。建议的最低门槛：

- PoC：至少 100 首 × 4–5 个难度；
- v0：300+ 首 / 1200+ 张谱；
- 必须覆盖 13、14、15、15+；否则无法验证高难度生成；
- 音频与谱面必须是同一剪辑版本，对齐置信度 < 0.5 的样本人工检查或排除；
- 按歌曲划分训练/验证/测试，禁止按谱面或窗口划分。

数据不达标时，先做 P2.5 的难度画像校准，而不是启动训练。

## 项目结构

```text
src/mainote/
  maidata/      解析器、IR 模型、编译器、校验器、统计
  audio/        ffmpeg 解码、起音/BPM/节拍前端、TimingInfo
  generate/     基于规则的谱面生成器
  dataset/      收录、对齐、报告、划分、按难度样本
  eval/         音符级评估指标
  ml/           P3 tokenizer、特征、数据集、模型、训练、推理
  cli.py        argparse 命令行入口
configs/        训练配置
docs/           训练和开发文档
tests/          标准库 unittest 测试集
scripts/        演示辅助脚本
dataset/xxdw/   提交到仓库的 maidata 测试样本（4 张谱面，BPM 143，无音频）
```

## 评估

评估器在给定时间容差（默认 60 ms）下，将生成谱面与参考谱面进行比较：

- 起音精确率 / 召回率 / F1
- 位置、类型和完全匹配准确率
- Hold 时长 MAE
- 滑动端点准确率
- 每秒音符数、最大同时音符数以及类型占比差异

```bash
PYTHONPATH=src python3 -m mainote eval generated.txt reference.txt --index 5
```

## 测试

```bash
# 完整测试集（音频测试需要 numpy）
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v

# 仅 P0：会跳过 12 个音频/数据集/模型测试
PYTHONPATH=src python3 -m unittest discover -s tests

# 单个测试
PYTHONPATH=src python3 -m unittest tests.test_parser.TestTokenSyntax.test_path_chain -v
```

测试集目前共有 84 个测试。项目没有单独的 lint/typecheck/CI 配置；`python3 -m compileall -q src tests scripts` 是一个有用的快速检查。

## 路线图

- **P2.5：** 使用真实数据校准难度画像，重点覆盖 13–15/15+；确认 15+ = 15.5 的数值语义（tokenizer 已支持 bucket 31）；改进节拍/强拍跟踪，检查低置信度的对齐结果。
- **P3：** 训练 v0 模型；补齐辅助损失（onset / duration / L_level）、beam search 或候选重排、跨窗口 Hold/Slide 状态和动态 BPM；在留出歌曲上稳定超过规则基线后再设为默认。
- **P4：** 整曲生成、全局段落 memory、重复乐段一致性、动态 BPM、复杂滑动以及人工评估。
- **P5：** 预览/编辑器、手动时间轴修正、局部重新生成以及打包。

## 已知限制

- 当前默认生成器是**规则基线**，不是学习得到的模型。
- P3 v0 训练/推理代码已经就绪，但仓库没有训练检查点；需要 GPU 服务器和数据。
- 时间轴前端假设一首歌使用**恒定 BPM**；P3 推理可通过 `--timing` JSON 的 `bpm_changes` 或 `--bpm`/`--first` 手动校正，但不会自动估计 tempo map。
- 规则基线的等级密度先验在 15.0 封顶，master/remaster 的密度会饱和；13–15/15+ 的复杂度和音符类型比例尚未用真实数据校准。
- P3 tokenizer 已支持 `15+`（bucket 31）；16.0 以上会 clamp 到 `15+`。高难度的密度和复杂度仍需 P2.5 校准。
- P3 v0 默认采样解码（可切换 greedy），训练为加权 token 交叉熵；尚无候选重排、辅助损失和跨窗口状态。
- 节拍/强拍跟踪较为轻量；真实歌曲可能需要 `--bpm`/`--first` 修正。
- 仓库内置的测试样本没有音频；音频需由用户自行提供。
- 神经谱面生成需要大量成对的音频/谱面数据和 GPU。仅凭一首歌不足以训练通用模型。

## 文档

- [`MAIDATA_FORMAT.md`](MAIDATA_FORMAT.md) — maidata/simai 格式参考。
- [`AUDIO_TO_MAIDATA_PLAN.md`](AUDIO_TO_MAIDATA_PLAN.md) — 完整系统的设计计划。部分内容属于愿景；代码和测试才是事实来源。
- [`data/README.md`](data/README.md) — 数据集布局与质量规则。
- [`docs/TRAINING.md`](docs/TRAINING.md) — P3 服务端训练、推理、评估和故障排查。
- [`AGENTS.md`](AGENTS.md) — 面向在本仓库工作的 AI 编码代理的指南。

## 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

Copyright (c) 2026 瀚观
