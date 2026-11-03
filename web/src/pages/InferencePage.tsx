import { useEffect, useState } from "react";
import type { ChangeEvent } from "react";
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  CircularProgress,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { ExpandMore, PlayArrow, Refresh, Upload } from "@mui/icons-material";
import { Link as RouterLink, useSearchParams } from "react-router-dom";
import { api, formatError, postJson } from "../api";
import JobPanel from "../components/JobPanel";
import PageHeader from "../components/PageHeader";
import Track from "../components/Track";
import { useJob } from "../useJob";
import { PALETTE } from "../theme";
import type { Options } from "../types";

interface FormState {
  audio: string;
  checkpoint: string;
  difficulty: string;
  level: string;
  temperature: string;
  topK: string;
  seed: string;
  maxNewTokens: string;
  bpm: string;
  first: string;
  device: string;
  outputDir: string;
  timing: string;
  features: string;
}

const INITIAL: FormState = {
  audio: "",
  checkpoint: "",
  difficulty: "master",
  level: "",
  temperature: "0.8",
  topK: "20",
  seed: "0",
  maxNewTokens: "2048",
  bpm: "",
  first: "",
  device: "auto",
  outputDir: "web_infer",
  timing: "",
  features: "",
};

const DIFFICULTIES = [
  { value: "basic", label: "BASIC" },
  { value: "advanced", label: "ADVANCED" },
  { value: "expert", label: "EXPERT" },
  { value: "master", label: "MASTER" },
  { value: "remaster", label: "Re:MASTER" },
];

function optionalNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

export default function InferencePage() {
  const [params] = useSearchParams();
  const requestedCheckpoint = params.get("checkpoint") ?? "";
  const [options, setOptions] = useState<Options | null>(null);
  const [form, setForm] = useState<FormState>(INITIAL);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [starting, setStarting] = useState(false);
  const { snapshot, error: jobError, refresh } = useJob(1200, ["infer"]);

  const load = async (): Promise<void> => {
    try {
      const data = await api<Options>("/api/options");
      setOptions(data);
      setForm((current) => {
        const requested = data.checkpoints.some((item) => item.value === requestedCheckpoint)
          ? requestedCheckpoint
          : "";
        return {
          ...current,
          checkpoint:
            current.checkpoint ||
            requested ||
            data.checkpoints.find((item) => item.recommended)?.value ||
            data.checkpoints[0]?.value ||
            "",
          audio: current.audio || data.audio[0]?.value || "",
        };
      });
      setError(null);
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  useEffect(() => {
    if (snapshot?.job?.status === "completed") {
      void load();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snapshot?.job?.finished_at]);

  const update = (key: keyof FormState, value: string) =>
    setForm((current) => ({ ...current, [key]: value }));

  const handleUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const body = new FormData();
    body.append("file", file);
    setUploading(true);
    setError(null);
    try {
      const result = await api<{ path: string }>("/api/upload/audio", {
        method: "POST",
        body,
      });
      update("audio", result.path);
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setUploading(false);
    }
  };

  const start = async () => {
    if (!form.audio || !form.checkpoint) {
      setError("请选择音频和 checkpoint");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      await postJson("/api/infer/start", {
        audio: form.audio,
        checkpoint: form.checkpoint,
        difficulty: form.difficulty,
        level: optionalNumber(form.level),
        temperature: optionalNumber(form.temperature),
        top_k: optionalNumber(form.topK),
        seed: optionalNumber(form.seed),
        max_new_tokens: optionalNumber(form.maxNewTokens),
        bpm: optionalNumber(form.bpm),
        first: optionalNumber(form.first),
        device: form.device,
        output_dir: form.outputDir,
        timing: form.timing || null,
        features: form.features || null,
      });
      await refresh();
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setStarting(false);
    }
  };

  const busy = starting || Boolean(snapshot?.active);

  return (
    <Stack spacing={3}>
      <PageHeader
        title="推理"
        description="用 P3 checkpoint 把音频转成 maidata。生成完成后可以直接进入环形预览器检查，也可以继续训练新的模型。"
        actions={
          <>
            <Button component={RouterLink} to="/train">
              去训练
            </Button>
            <Button startIcon={<Refresh />} onClick={() => void load()} disabled={loading}>
              刷新选项
            </Button>
          </>
        }
      />

      {error ? <Alert severity="error">{error}</Alert> : null}
      {options && options.checkpoints.length === 0 ? (
        <Alert severity="warning">
          还没有可用 checkpoint。先在 <RouterLink to="/train">训练</RouterLink> 页面启动一次训练。
        </Alert>
      ) : null}

      <Box>
        <Track
          first
          label="输入"
          hint="音频文件与模型权重"
          accent={PALETTE.aqua}
          action={
            <Typography variant="caption" color="text.secondary">
              {options?.checkpoints.length ?? 0} 个 checkpoint · {options?.audio.length ?? 0} 个音频
            </Typography>
          }
        >
          {loading ? (
            <Box sx={{ py: 5, textAlign: "center" }}>
              <CircularProgress size={22} />
            </Box>
          ) : (
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
                gap: 2,
              }}
            >
              <Stack spacing={1}>
                <FormControl fullWidth size="small">
                  <InputLabel>音频</InputLabel>
                  <Select
                    label="音频"
                    value={form.audio}
                    onChange={(event) => update("audio", event.target.value)}
                  >
                    {options?.audio.map((item) => (
                      <MenuItem key={item.value} value={item.value}>
                        {item.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <Stack direction="row" spacing={1} alignItems="center">
                  <Button
                    component="label"
                    size="small"
                    variant="outlined"
                    startIcon={<Upload sx={{ fontSize: 16 }} />}
                    disabled={uploading}
                  >
                    {uploading ? "上传中…" : "上传音频"}
                    <input
                      type="file"
                      hidden
                      accept="audio/*"
                      onChange={(event) => void handleUpload(event)}
                    />
                  </Button>
                  <Typography
                    variant="caption"
                    color="text.disabled"
                    noWrap
                    title={form.audio}
                    sx={{ minWidth: 0 }}
                  >
                    {form.audio || "未选择"}
                  </Typography>
                </Stack>
              </Stack>

              <FormControl fullWidth size="small">
                <InputLabel>Checkpoint</InputLabel>
                <Select
                  label="Checkpoint"
                  value={form.checkpoint}
                  onChange={(event) => update("checkpoint", event.target.value)}
                >
                  {options?.checkpoints.map((item) => (
                    <MenuItem key={item.value} value={item.value}>
                      {item.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Box>
          )}
        </Track>

        <Track
          label="谱面"
          hint="难度、等级、BPM 与运行位置"
          accent={PALETTE.gold}
          action={
            <Typography variant="caption" color="text.secondary">
              temperature {form.temperature} · top-k {form.topK}
            </Typography>
          }
        >
          <Stack spacing={2}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.75 }}>
                目标难度
              </Typography>
              <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                {DIFFICULTIES.map((item) => {
                  const active = form.difficulty === item.value;
                  return (
                    <Button
                      key={item.value}
                      size="small"
                      variant={active ? "contained" : "outlined"}
                      onClick={() => update("difficulty", item.value)}
                      sx={{
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.04em",
                        minWidth: 0,
                        px: 1.25,
                      }}
                    >
                      {item.label}
                    </Button>
                  );
                })}
              </Stack>
            </Box>

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                gap: 1.5,
              }}
            >
              <TextField
                fullWidth
                label="BPM 覆盖"
                placeholder="留空自动估计"
                helperText="填谱面 BPM；自动估计出现倍频错误时手动指定"
                value={form.bpm}
                onChange={(event) => update("bpm", event.target.value)}
              />
              <TextField
                fullWidth
                label="first / 起始秒"
                placeholder="留空默认为 0"
                helperText="音乐第一拍相对音频开头的秒数"
                value={form.first}
                onChange={(event) => update("first", event.target.value)}
              />
            </Box>

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr 1fr", md: "repeat(4, 1fr)" },
                gap: 1.5,
              }}
            >
              <TextField
                fullWidth
                label="等级 (可选)"
                value={form.level}
                onChange={(event) => update("level", event.target.value)}
              />
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
            </Box>

            <Accordion disableGutters>
              <AccordionSummary
                expandIcon={<ExpandMore sx={{ fontSize: 18 }} />}
                sx={{ borderTop: 1, borderColor: "divider" }}
              >
                <Typography variant="caption" color="text.secondary">
                  采样、timing 与 features
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: { xs: "1fr 1fr", md: "repeat(4, 1fr)" },
                    gap: 1.5,
                  }}
                >
                  {[
                    ["temperature", "temperature"],
                    ["topK", "top-k"],
                    ["seed", "seed"],
                    ["maxNewTokens", "max_new_tokens"],
                  ].map(([key, label]) => (
                    <TextField
                      key={key}
                      fullWidth
                      label={label}
                      value={form[key as keyof FormState]}
                      onChange={(event) => update(key as keyof FormState, event.target.value)}
                    />
                  ))}
                  <FormControl fullWidth size="small">
                    <InputLabel>timing</InputLabel>
                    <Select
                      label="timing"
                      value={form.timing}
                      onChange={(event) => update("timing", event.target.value)}
                    >
                      <MenuItem value="">不使用</MenuItem>
                      {options?.timing.map((item) => (
                        <MenuItem key={item.value} value={item.value}>
                          {item.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  <FormControl fullWidth size="small">
                    <InputLabel>features</InputLabel>
                    <Select
                      label="features"
                      value={form.features}
                      onChange={(event) => update("features", event.target.value)}
                    >
                      <MenuItem value="">自动计算</MenuItem>
                      {options?.features.map((item) => (
                        <MenuItem key={item.value} value={item.value}>
                          {item.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Box>
              </AccordionDetails>
            </Accordion>

            <Stack direction={{ xs: "column", sm: "row" }} spacing={2} alignItems={{ sm: "center" }}>
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrow />}
                disabled={busy || !form.audio || !form.checkpoint}
                onClick={() => void start()}
              >
                {snapshot?.active ? "有任务正在运行" : "开始推理"}
              </Button>
              <Typography variant="caption" color="text.secondary">
                任务在后台以 mainote ml infer 运行，可以留在这个页面看日志。
              </Typography>
            </Stack>
          </Stack>
        </Track>

        <Track label="任务" hint="后台进程、实时日志与生成结果" accent={PALETTE.green}>
          <JobPanel
            snapshot={snapshot}
            error={jobError}
            onStop={() => void postJson("/api/jobs/stop", {}).then(refresh)}
          />
        </Track>
      </Box>
    </Stack>
  );
}
