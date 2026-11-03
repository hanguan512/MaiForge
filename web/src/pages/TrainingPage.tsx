import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
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
import TrainingDataPanel from "../components/TrainingDataPanel";
import { buildTrainingSeries, metricLabels } from "../metrics";
import { NUMERIC_FONT } from "../theme";
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

function Panel({ children }: { children: ReactNode }) {
  return (
    <Box
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: "4px",
        bgcolor: "background.paper",
        p: { xs: 2, md: 2.5 },
      }}
    >
      {children}
    </Box>
  );
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
  const { snapshot, error: jobError, refresh } = useJob();
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
    <Stack spacing={2.5}>
      <PageHeader
        title="训练"
        description="从已下载的数据集中选择歌曲、准备样本与特征、调整超参数，再启动 mainote ml train。正式训练建议放到 GPU 服务器上跑。"
        actions={
          <Button startIcon={<Refresh />} onClick={() => void load()} disabled={loading}>
            刷新
          </Button>
        }
      />

      {error ? <Alert severity="error">{error}</Alert> : null}
      {options && !options.gpu ? (
        <Alert severity="info">
          当前机器没有检测到 NVIDIA GPU。正式训练请在 GPU 服务器运行;这里可以先用 dry run 或很小的 max_steps 验证流程。
        </Alert>
      ) : null}

      <Panel>
        {loading ? (
          <Box sx={{ py: 6, textAlign: "center" }}>
            <CircularProgress size={26} />
          </Box>
        ) : (
          <Stack spacing={2.25}>
            <Stack direction="row" alignItems="baseline" justifyContent="space-between" spacing={2}>
              <Typography variant="h6" sx={{ fontSize: 18 }}>
                启动配置
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {options?.runs.length ?? 0} 条历史 run / {options?.checkpoints.length ?? 0} 个 checkpoint
              </Typography>
            </Stack>

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", md: "repeat(4, 1fr)" },
                gap: 2,
              }}
            >
              <FormControl fullWidth size="small" sx={{ gridColumn: { md: "span 2" } }}>
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
                size="small"
                label="输出目录"
                value={form.outputDir}
                onChange={(event) => update("outputDir", event.target.value)}
              />

              <TextField
                fullWidth
                size="small"
                label="max_steps (可选)"
                value={form.maxSteps}
                onChange={(event) => update("maxSteps", event.target.value)}
              />
              <TextField
                fullWidth
                size="small"
                label="num_workers (可选)"
                value={form.numWorkers}
                onChange={(event) => update("numWorkers", event.target.value)}
              />
              <FormControl fullWidth size="small" sx={{ gridColumn: { md: "span 2" } }}>
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
                只检查数据与配置,不写 checkpoint
              </Typography>
              <Box sx={{ flexGrow: 1 }} />
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrow />}
                disabled={busy || !form.config || !selectedSongs.length}
                onClick={() => void start()}
              >
                {snapshot?.active ? "有任务正在运行" : "开始训练"}
              </Button>
            </Stack>
          </Stack>
        )}
      </Panel>

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

      <HyperparamPanel
        form={hyper}
        valRatio={valRatio}
        onValRatioChange={setValRatio}
        onChange={updateHyper}
        onReset={() => void loadConfig(form.config)}
      />

      <Panel>
        <Stack
          direction={{ xs: "column", sm: "row" }}
          alignItems={{ sm: "baseline" }}
          justifyContent="space-between"
          spacing={1}
        >
          <Box>
            <Typography variant="h6" sx={{ fontSize: 18 }}>
              网络结构
            </Typography>
            <Typography variant="caption" color="text.secondary">
              按「模型结构」超参数实时估算；未训练 checkpoint 时仅表示待构建的模型。
            </Typography>
          </Box>
        </Stack>
        <Box sx={{ mt: 2 }}>
          <ModelArchitectureView
            architecture={architecture}
            loading={archLoading}
            error={archError}
          />
        </Box>
      </Panel>

      <JobPanel
        snapshot={snapshot}
        error={jobError}
        onStop={() => void postJson("/api/jobs/stop", {}).then(refresh)}
      />

      <Box
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: "4px",
          bgcolor: "background.paper",
        }}
      >
        <Box sx={{ px: 2.5, pt: 2, pb: 1.5 }}>
          <Stack direction="row" alignItems="baseline" justifyContent="space-between" spacing={2}>
            <Typography variant="h6" sx={{ fontSize: 18 }}>
              历史训练
            </Typography>
            <Typography variant="caption" color="text.secondary">
              runs/ 下按时间排列
            </Typography>
          </Stack>
        </Box>
        {options?.runs.length ? (
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>run</TableCell>
                  <TableCell align="right">epoch</TableCell>
                  <TableCell align="right">best val loss</TableCell>
                  <TableCell align="right">best note score</TableCell>
                  <TableCell align="right">checkpoints</TableCell>
                  <TableCell align="right">操作</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {options.runs.map((run) => (
                  <TableRow key={run.path} hover>
                    <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{run.path}</TableCell>
                    <TableCell align="right" sx={{ fontFamily: NUMERIC_FONT }}>
                      {run.last_epoch ?? "-"}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: NUMERIC_FONT }}>
                      {formatMetric(run.best_val_loss)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: NUMERIC_FONT }}>
                      {formatMetric(run.best_note_score)}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: NUMERIC_FONT }}>
                      {run.checkpoint_count}
                    </TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                        <Button size="small" startIcon={<Visibility />} onClick={() => void openDetail(run.path)}>
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
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        ) : (
          <Box sx={{ px: 2.5, pb: 2.5 }}>
            <Box
              sx={{
                border: 1,
                borderStyle: "dashed",
                borderColor: "divider",
                borderRadius: "3px",
                p: 3,
                textAlign: "center",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                还没有训练记录。启动一次训练后会出现在这里。
              </Typography>
            </Box>
          </Box>
        )}
      </Box>

      <Dialog open={Boolean(detail) || detailLoading} onClose={() => setDetail(null)} maxWidth="md" fullWidth>
        <DialogTitle sx={{ fontFamily: "monospace", fontSize: 14 }}>
          {detail?.path ?? "加载中..."}
        </DialogTitle>
        <DialogContent dividers>
          {detailLoading ? (
            <Box sx={{ py: 5, textAlign: "center" }}>
              <CircularProgress size={26} />
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
              <Table size="small">
                <TableBody>
                  {detail.files.map((file) => (
                    <TableRow key={file.path}>
                      <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{file.name}</TableCell>
                      <TableCell align="right">{(file.size / 1024).toFixed(1)} KB</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
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
                      borderRadius: "3px",
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
