import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { PlayArrow, Refresh, Visibility } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { api, formatError, postJson } from "../api";
import HyperparamPanel from "../components/HyperparamPanel";
import JobPanel from "../components/JobPanel";
import MetricChart from "../components/MetricChart";
import ModelArchitectureView from "../components/ModelArchitecture";
import PageHeader from "../components/PageHeader";
import StatStrip from "../components/StatStrip";
import Track from "../components/Track";
import TrainingDataPanel from "../components/TrainingDataPanel";
import { buildTrainingSeries, metricLabels } from "../metrics";
import { NUMERIC_FONT, PALETTE } from "../theme";
import {
  DEFAULT_HYPERPARAMS,
  architectureQuery,
  hyperparamsFromConfig,
  hyperparamsPayload,
} from "../trainingForm";
import type { HyperparamForm } from "../trainingForm";
import { useJob } from "../useJob";
import type {
  ModelArchitecture,
  Options,
  RunDetail,
  TrainConfigResponse,
  TrainData,
} from "../types";

interface TrainForm {
  config: string;
  device: string;
  outputDir: string;
  maxSteps: string;
  numWorkers: string;
  resume: string;
  dryRun: boolean;
}

function optionalNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMetric(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
}

export default function TrainingPage() {
  const navigate = useNavigate();
  const [options, setOptions] = useState<Options | null>(null);
  const [form, setForm] = useState<TrainForm>({
    config: "",
    device: "auto",
    outputDir: "web_train",
    maxSteps: "",
    numWorkers: "",
    resume: "",
    dryRun: false,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [hyper, setHyper] = useState<HyperparamForm>(DEFAULT_HYPERPARAMS);
  const [valRatio, setValRatio] = useState("0.1");
  const [trainData, setTrainData] = useState<TrainData | null>(null);
  const [dataRoot, setDataRoot] = useState("");
  const [selectedSongs, setSelectedSongs] = useState<string[]>([]);
  const [overwriteFeatures, setOverwriteFeatures] = useState(false);
  const [dataLoading, setDataLoading] = useState(false);
  const [architecture, setArchitecture] = useState<ModelArchitecture | null>(null);
  const [archLoading, setArchLoading] = useState(false);
  const [archError, setArchError] = useState<string | null>(null);
  const configLoaded = useRef(false);
  const archSeq = useRef(0);
  const { snapshot, error: jobError, refresh } = useJob(1200, ["train", "prepare"]);
  const busy = starting || Boolean(snapshot?.active);

  const loadConfig = async (path: string): Promise<void> => {
    if (!path) return;
    try {
      const data = await api<TrainConfigResponse>(
        `/api/train/config?path=${encodeURIComponent(path)}`,
      );
      setHyper(hyperparamsFromConfig(data.config));
    } catch (caught) {
      setError(formatError(caught));
    }
  };

  const load = async (): Promise<void> => {
    try {
      const data = await api<Options>("/api/options");
      setOptions(data);
      const preferred =
        data.configs.find((item) => item.value === data.defaults.config)?.value ||
        data.configs[0]?.value ||
        "";
      setForm((current) => (current.config ? current : { ...current, config: preferred }));
      if (preferred && !configLoaded.current) {
        configLoaded.current = true;
        void loadConfig(preferred);
      }
      setError(null);
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setLoading(false);
    }
  };

  const loadTrainData = async (root?: string): Promise<void> => {
    setDataLoading(true);
    try {
      const suffix = root ? `?root=${encodeURIComponent(root)}` : "";
      const data = await api<TrainData>(`/api/train/data${suffix}`);
      setTrainData(data);
      setDataRoot(data.root);
      setSelectedSongs((current) => {
        const ids = new Set(data.songs.map((song) => song.id));
        const kept = current.filter((id) => ids.has(id));
        if (kept.length) return kept;
        const ready = data.songs
          .filter((song) => song.trainable)
          .map((song) => song.id);
        return ready.length ? ready : data.songs.map((song) => song.id);
      });
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setDataLoading(false);
    }
  };

  useEffect(() => {
    void load();
    void loadTrainData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (snapshot?.job?.finished_at) {
      void load();
      void loadTrainData(dataRoot || undefined);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot?.job?.finished_at]);

  useEffect(() => {
    const seq = ++archSeq.current;
    const timer = window.setTimeout(() => {
      setArchLoading(true);
      const params = architectureQuery(hyper);
      api<ModelArchitecture>(`/api/model/architecture?${params.toString()}`)
        .then((data) => {
          if (seq !== archSeq.current) return;
          setArchitecture(data);
          setArchError(null);
        })
        .catch((caught) => {
          if (seq !== archSeq.current) return;
          setArchError(formatError(caught));
        })
        .finally(() => {
          if (seq === archSeq.current) setArchLoading(false);
        });
    }, 250);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    hyper.hidden,
    hyper.audioLayers,
    hyper.decoderLayers,
    hyper.heads,
    hyper.ffn,
    hyper.dropout,
  ]);

  const update = <K extends keyof TrainForm>(key: K, value: TrainForm[K]) =>
    setForm((current) => ({ ...current, [key]: value }));

  const updateHyper = <K extends keyof HyperparamForm>(
    key: K,
    value: HyperparamForm[K],
  ) => setHyper((current) => ({ ...current, [key]: value }));

  const changeRoot = (root: string) => {
    setDataRoot(root);
    setSelectedSongs([]);
    void loadTrainData(root);
  };

  const toggleSong = (id: string) =>
    setSelectedSongs((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id],
    );

  const selectReadySongs = () =>
    setSelectedSongs(
      (trainData?.songs ?? [])
        .filter((song) => song.trainable)
        .map((song) => song.id),
    );

  const prepare = async () => {
    if (!selectedSongs.length) {
      setError("请先选择至少一首歌曲");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      await postJson("/api/train/prepare", {
        root: dataRoot,
        song_ids: selectedSongs,
        overwrite_features: overwriteFeatures,
      });
      await refresh();
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setStarting(false);
    }
  };

  const start = async () => {
    if (!form.config) {
      setError("请选择训练配置");
      return;
    }
    if (!selectedSongs.length) {
      setError("请先选择至少一首歌曲");
      return;
    }
    const missing = (trainData?.songs ?? []).filter(
      (song) =>
        selectedSongs.includes(song.id) &&
        (!song.sample_ready || !song.features_ready),
    );
    if (missing.length) {
      setError(
        `所选歌曲中有 ${missing.length} 首缺少样本或特征，请先点击「准备所选数据」`,
      );
      return;
    }
    setStarting(true);
    setError(null);
    try {
      await postJson("/api/train/start", {
        config: form.config,
        device: form.device,
        output_dir: form.outputDir,
        max_steps: optionalNumber(form.maxSteps),
        num_workers: optionalNumber(form.numWorkers),
        resume: form.resume || null,
        dry_run: form.dryRun,
        root: dataRoot || null,
        song_ids: selectedSongs,
        features_dir: trainData?.features_dir ?? null,
        seed: optionalNumber(hyper.seed),
        val_ratio: optionalNumber(valRatio),
        hyperparams: hyperparamsPayload(hyper),
      });
      await refresh();
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setStarting(false);
    }
  };

  const openDetail = async (path: string) => {
    setDetailLoading(true);
    setError(null);
    try {
      const data = await api<RunDetail>(`/api/run?path=${encodeURIComponent(path)}`);
      setDetail(data);
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setDetailLoading(false);
    }
  };

  const detailSeries = useMemo(
    () => (detail ? buildTrainingSeries(detail.metrics) : []),
    [detail],
  );
  const detailLabels = useMemo(
    () => (detail ? metricLabels(detail.metrics) : []),
    [detail],
  );

  return (
    <Stack spacing={3}>
      <PageHeader
        title="训练"
        description="从数据集里选歌、准备样本与特征、调整超参数，再启动 mainote ml train。正式训练请放到 GPU 服务器运行。"
        actions={
          <Button startIcon={<Refresh />} onClick={() => void load()} disabled={loading}>
            刷新
          </Button>
        }
      />

      <StatStrip
        items={[
          {
            label: "计算设备",
            value: options?.gpu ? options.gpu.name : "CPU",
            accent: options?.gpu ? PALETTE.green : PALETTE.gold,
            detail: options?.gpu
              ? `${options.gpu.used_mb} / ${options.gpu.total_mb} MB · ${options.gpu.util}%`
              : "未检测到 NVIDIA GPU",
          },
          { label: "训练配置", value: options?.configs.length ?? "-" },
          { label: "历史 run", value: options?.runs.length ?? "-" },
          { label: "checkpoint", value: options?.checkpoints.length ?? "-" },
          {
            label: "已选歌曲",
            value: `${selectedSongs.length}/${trainData?.songs.length ?? 0}`,
            accent: PALETTE.aqua,
          },
        ]}
      />

      {error ? <Alert severity="error">{error}</Alert> : null}
      {options && !options.gpu ? (
        <Alert severity="info">
          当前机器没有检测到 NVIDIA GPU。正式训练请在 GPU 服务器运行；这里可以先用 dry run
          或很小的 max_steps 验证流程。
        </Alert>
      ) : null}

      <Box>
        <Track
          first
          label="数据"
          hint="按歌曲切分 train / val，同一首歌不会跨集"
        >
          <TrainingDataPanel
            data={trainData}
            loading={dataLoading}
            root={dataRoot}
            selected={selectedSongs}
            busy={busy}
            overwrite={overwriteFeatures}
            onRootChange={changeRoot}
            onToggle={toggleSong}
            onSelectReady={selectReadySongs}
            onClear={() => setSelectedSongs([])}
            onPrepare={() => void prepare()}
            onRefresh={() => void loadTrainData(dataRoot || undefined)}
            onOverwriteChange={setOverwriteFeatures}
          />
        </Track>

        <Track
          label="启动"
          hint="覆盖值写入 runs/<run>/train_config.yaml"
          accent={PALETTE.aqua}
        >
          {loading ? (
            <Box sx={{ py: 5, textAlign: "center" }}>
              <CircularProgress size={22} />
            </Box>
          ) : (
            <Stack spacing={2}>
              <Box
                sx={{
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "1fr",
                    sm: "1fr 1fr",
                    lg: "minmax(220px, 2fr) repeat(3, minmax(120px, 1fr))",
                  },
                  gap: 1.5,
                }}
              >
                <FormControl fullWidth size="small">
                  <InputLabel>训练配置</InputLabel>
                  <Select
                    label="训练配置"
                    value={form.config}
                    onChange={(event) => {
                      update("config", event.target.value);
                      void loadConfig(event.target.value);
                    }}
                  >
                    {options?.configs.map((item) => (
                      <MenuItem key={item.value} value={item.value}>
                        {item.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth size="small">
                  <InputLabel>设备</InputLabel>
                  <Select
                    label="设备"
                    value={form.device}
                    onChange={(event) => update("device", event.target.value)}
                  >
                    <MenuItem value="auto">auto</MenuItem>
                    <MenuItem value="cuda">cuda</MenuItem>
                    <MenuItem value="cpu">cpu</MenuItem>
                  </Select>
                </FormControl>
                <TextField
                  fullWidth
                  label="输出目录"
                  value={form.outputDir}
                  onChange={(event) => update("outputDir", event.target.value)}
                />
                <TextField
                  fullWidth
                  label="max_steps (可选)"
                  value={form.maxSteps}
                  onChange={(event) => update("maxSteps", event.target.value)}
                />
                <TextField
                  fullWidth
                  label="num_workers (可选)"
                  value={form.numWorkers}
                  onChange={(event) => update("numWorkers", event.target.value)}
                />
                <FormControl fullWidth size="small" sx={{ gridColumn: { lg: "span 2" } }}>
                  <InputLabel>恢复 checkpoint (可选)</InputLabel>
                  <Select
                    label="恢复 checkpoint (可选)"
                    value={form.resume}
                    onChange={(event) => update("resume", event.target.value)}
                  >
                    <MenuItem value="">从头开始</MenuItem>
                    {options?.checkpoints.map((item) => (
                      <MenuItem key={item.value} value={item.value}>
                        {item.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Box>

              <Stack
                direction={{ xs: "column", sm: "row" }}
                spacing={2}
                alignItems={{ sm: "center" }}
              >
                <FormControlLabel
                  control={
                    <Switch
                      checked={form.dryRun}
                      onChange={(event) => update("dryRun", event.target.checked)}
                    />
                  }
                  label={<Typography variant="body2">dry run</Typography>}
                />
                <Typography variant="caption" color="text.secondary">
                  只检查数据与配置，不写 checkpoint
                </Typography>
                <Box sx={{ flexGrow: 1 }} />
                <Button
                  variant="contained"
                  size="large"
                  startIcon={<PlayArrow />}
                  disabled={busy || !form.config || !selectedSongs.length}
                  onClick={() => void start()}
                >
                  {snapshot?.active ? "有任务正在运行" : `开始训练 · ${selectedSongs.length} 首`}
                </Button>
              </Stack>
            </Stack>
          )}
        </Track>

        <Track label="参数" hint="留空表示沿用配置文件中的值">
          <HyperparamPanel
            form={hyper}
            valRatio={valRatio}
            onValRatioChange={setValRatio}
            onChange={updateHyper}
            onReset={() => void loadConfig(form.config)}
          />
        </Track>

        <Track label="结构" hint="按超参数实时估算，未训练时不代表已有模型">
          <ModelArchitectureView
            architecture={architecture}
            loading={archLoading}
            error={archError}
          />
        </Track>

        <Track label="任务" hint="同一时间只运行一个后台任务" accent={PALETTE.green}>
          <JobPanel
            snapshot={snapshot}
            error={jobError}
            onStop={() => void postJson("/api/jobs/stop", {}).then(refresh)}
          />
        </Track>

        <Track
          label="记录"
          hint="runs/ 下按时间排列，可查看指标与 checkpoint"
          accent={PALETTE.gold}
        >
          {options?.runs.length ? (
            <Box sx={{ borderTop: 1, borderColor: "divider" }}>
              <Box
                sx={{
                  display: { xs: "none", md: "grid" },
                  gridTemplateColumns: "minmax(0, 1fr) 72px 108px 108px 90px auto",
                  columnGap: 2,
                  alignItems: "center",
                  py: 0.75,
                  borderBottom: 1,
                  borderColor: "divider",
                  color: "text.disabled",
                }}
              >
                <Typography variant="caption">run</Typography>
                <Typography variant="caption" sx={{ textAlign: "right" }}>
                  epoch
                </Typography>
                <Typography variant="caption" sx={{ textAlign: "right" }}>
                  val loss
                </Typography>
                <Typography variant="caption" sx={{ textAlign: "right" }}>
                  note score
                </Typography>
                <Typography variant="caption" sx={{ textAlign: "right" }}>
                  ckpt
                </Typography>
                <Typography variant="caption" sx={{ textAlign: "right" }}>
                  操作
                </Typography>
              </Box>
              {options.runs.map((run) => (
                <Box
                  key={run.path}
                  sx={{
                    display: "grid",
                    gridTemplateColumns: { xs: "minmax(0, 1fr) auto", md: "minmax(0, 1fr) 72px 108px 108px 90px auto" },
                    columnGap: 2,
                    rowGap: 0.75,
                    alignItems: "center",
                    py: 1.1,
                    borderBottom: 1,
                    borderColor: "divider",
                    "&:hover": { bgcolor: "action.hover" },
                  }}
                >
                  <Typography
                    variant="body2"
                    sx={{ fontVariantNumeric: "tabular-nums", wordBreak: "break-all" }}
                  >
                    {run.path}
                  </Typography>
                  {/* Numeric cells are hidden on xs; the row keeps its actions. */}
                  {([
                    ["epoch", run.last_epoch ?? "-"],
                    ["val", formatMetric(run.best_val_loss)],
                    ["note", formatMetric(run.best_note_score)],
                    ["ckpt", run.checkpoint_count],
                  ] as const).map(([key, value]) => (
                    <Typography
                      key={key}
                      sx={{
                        display: { xs: "none", md: "block" },
                        textAlign: "right",
                        fontFamily: NUMERIC_FONT,
                        fontSize: 12.5,
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {value}
                    </Typography>
                  ))}
                  <Stack
                    direction="row"
                    spacing={0.5}
                    justifyContent="flex-end"
                    sx={{ gridColumn: { xs: "1 / -1", md: "auto" } }}
                  >
                    <Button
                      size="small"
                      startIcon={<Visibility sx={{ fontSize: 16 }} />}
                      onClick={() => void openDetail(run.path)}
                    >
                      详情
                    </Button>
                    {run.checkpoints[0] ? (
                      <Button
                        size="small"
                        onClick={() =>
                          navigate(`/infer?checkpoint=${encodeURIComponent(run.checkpoints[0])}`)
                        }
                      >
                        用于推理
                      </Button>
                    ) : null}
                  </Stack>
                </Box>
              ))}
            </Box>
          ) : (
            <Box
              sx={{
                border: 1,
                borderStyle: "dashed",
                borderColor: "divider",
                px: 3,
                py: 4,
                textAlign: "center",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                还没有训练记录。启动一次训练后会出现在这里。
              </Typography>
            </Box>
          )}
        </Track>
      </Box>

      <Dialog
        open={Boolean(detail) || detailLoading}
        onClose={() => setDetail(null)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle sx={{ fontSize: 14, wordBreak: "break-all" }}>
          {detail?.path ?? "加载中…"}
        </DialogTitle>
        <DialogContent dividers>
          {detailLoading ? (
            <Box sx={{ py: 5, textAlign: "center" }}>
              <CircularProgress size={24} />
            </Box>
          ) : detail ? (
            <Stack spacing={2}>
              <MetricChart series={detailSeries} xLabels={detailLabels} />
              <Typography variant="subtitle2">Checkpoints</Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
                {detail.checkpoints.map((checkpoint) => (
                  <Button
                    key={checkpoint}
                    size="small"
                    variant="outlined"
                    onClick={() => navigate(`/infer?checkpoint=${encodeURIComponent(checkpoint)}`)}
                  >
                    {checkpoint}
                  </Button>
                ))}
                {!detail.checkpoints.length ? (
                  <Typography variant="body2" color="text.secondary">
                    无
                  </Typography>
                ) : null}
              </Stack>
              <Typography variant="subtitle2">文件</Typography>
              <Box sx={{ borderTop: 1, borderColor: "divider" }}>
                {detail.files.map((file) => (
                  <Stack
                    key={file.path}
                    direction="row"
                    justifyContent="space-between"
                    sx={{ py: 0.6, borderBottom: 1, borderColor: "divider" }}
                  >
                    <Typography variant="caption" sx={{ wordBreak: "break-all" }}>
                      {file.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ flex: "none", ml: 2 }}>
                      {(file.size / 1024).toFixed(1)} KB
                    </Typography>
                  </Stack>
                ))}
              </Box>
              {detail.config ? (
                <>
                  <Typography variant="subtitle2">config.json</Typography>
                  <Box
                    component="pre"
                    sx={{
                      m: 0,
                      p: 1.5,
                      maxHeight: 220,
                      overflow: "auto",
                      bgcolor: "rgba(0,0,0,0.32)",
                      border: 1,
                      borderColor: "divider",
                      fontSize: 12,
                    }}
                  >
                    {JSON.stringify(detail.config, null, 2)}
                  </Box>
                </>
              ) : null}
            </Stack>
          ) : null}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetail(null)}>关闭</Button>
        </DialogActions>
      </Dialog>
    </Stack>
  );
}
