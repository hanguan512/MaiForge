import { useEffect, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
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
import { useJob } from "../useJob";
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

function optionalNumber(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
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

export default function InferencePage() {
  const [params] = useSearchParams();
  const requestedCheckpoint = params.get("checkpoint") ?? "";
  const [options, setOptions] = useState<Options | null>(null);
  const [form, setForm] = useState<FormState>(INITIAL);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [starting, setStarting] = useState(false);
  const { snapshot, error: jobError, refresh } = useJob();

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

  return (
    <Stack spacing={2.5}>
      <PageHeader
        title="推理"
        description="用 P3 checkpoint 把音频转成 maidata。生成完成后可以直接进入环形预览器检查,也可以继续训练新的模型。"
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

      <Panel>
        {loading ? (
          <Box sx={{ py: 6, textAlign: "center" }}>
            <CircularProgress size={26} />
          </Box>
        ) : (
          <Stack spacing={2.25}>
            <Stack direction="row" alignItems="baseline" justifyContent="space-between" spacing={2}>
              <Typography variant="h6" sx={{ fontSize: 18 }}>
                生成参数
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {options?.checkpoints.length ?? 0} 个 checkpoint / {options?.audio.length ?? 0} 个音频
              </Typography>
            </Stack>

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
                  <Button component="label" size="small" variant="outlined" startIcon={<Upload />} disabled={uploading}>
                    {uploading ? "上传中..." : "上传音频"}
                    <input type="file" hidden accept="audio/*" onChange={(event) => void handleUpload(event)} />
                  </Button>
                  <Typography variant="caption" color="text.secondary" noWrap>
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

            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr 1fr", md: "repeat(4, 1fr)" },
                gap: 2,
              }}
            >
              <FormControl fullWidth size="small">
                <InputLabel>难度</InputLabel>
                <Select
                  label="难度"
                  value={form.difficulty}
                  onChange={(event) => update("difficulty", event.target.value)}
                >
                  {["basic", "advanced", "expert", "master", "remaster"].map((value) => (
                    <MenuItem key={value} value={value}>
                      {value}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
              <TextField
                fullWidth
                size="small"
                label="等级 (可选)"
                value={form.level}
                onChange={(event) => update("level", event.target.value)}
              />
              <FormControl fullWidth size="small">
                <InputLabel>设备</InputLabel>
                <Select label="设备" value={form.device} onChange={(event) => update("device", event.target.value)}>
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
            </Box>

            <Accordion disableGutters>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="body2">高级参数</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Box
                  sx={{
                    display: "grid",
                    gridTemplateColumns: { xs: "1fr 1fr", md: "repeat(4, 1fr)" },
                    gap: 2,
                  }}
                >
                  {[
                    ["temperature", "temperature"],
                    ["topK", "top-k"],
                    ["seed", "seed"],
                    ["maxNewTokens", "max_new_tokens"],
                    ["bpm", "BPM 覆盖"],
                    ["first", "first (秒)"],
                  ].map(([key, label]) => (
                    <TextField
                      key={key}
                      fullWidth
                      size="small"
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

            <Stack direction="row" spacing={2} alignItems="center">
              <Button
                variant="contained"
                size="large"
                startIcon={<PlayArrow />}
                disabled={starting || !form.audio || !form.checkpoint || Boolean(snapshot?.active)}
                onClick={() => void start()}
              >
                {snapshot?.active ? "有任务正在运行" : "开始推理"}
              </Button>
              <Typography variant="caption" color="text.secondary">
                任务在后台以 mainote ml infer 运行,可以留在这个页面看日志。
              </Typography>
            </Stack>
          </Stack>
        )}
      </Panel>

      <JobPanel snapshot={snapshot} error={jobError} onStop={() => void postJson("/api/jobs/stop", {}).then(refresh)} />
    </Stack>
  );
}
