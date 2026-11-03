import { useEffect, useMemo, useRef } from "react";
import { Alert, Box, Button, Divider, Stack, Typography } from "@mui/material";
import { Download, StopCircle, Visibility } from "@mui/icons-material";
import { useTheme } from "@mui/material/styles";
import { useNavigate } from "react-router-dom";
import MetricChart from "./MetricChart";
import { buildTrainingSeries, metricLabels } from "../metrics";
import { accentFor, NUMERIC_FONT, PALETTE } from "../theme";
import type { JobSnapshot } from "../types";

const STATUS: Record<string, { label: string; color: string }> = {
  running: { label: "运行中", color: PALETTE.aqua },
  stopping: { label: "正在停止", color: PALETTE.gold },
  completed: { label: "已完成", color: PALETTE.green },
  failed: { label: "失败", color: PALETTE.red },
  stopped: { label: "已停止", color: PALETTE.muted },
};

function StatusTag({ label, color }: { label: string; color: string }) {
  const theme = useTheme();
  const tone = accentFor(theme.palette.mode, color);
  return (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.75,
        px: 0.9,
        py: 0.35,
        border: 1,
        borderColor: tone,
        color: tone,
        borderRadius: "2px",
        fontSize: 11,
        fontWeight: 700,
        lineHeight: 1.2,
      }}
    >
      <Box sx={{ width: 5, height: 5, borderRadius: "50%", bgcolor: tone }} />
      {label}
    </Box>
  );
}

export default function JobPanel({
  snapshot,
  onStop,
  error,
}: {
  snapshot: JobSnapshot | null;
  onStop?: () => void;
  error?: string | null;
}) {
  const navigate = useNavigate();
  const logRef = useRef<HTMLPreElement>(null);
  const logLines = snapshot?.log ?? [];
  const job = snapshot?.job ?? null;
  const metrics = snapshot?.metrics ?? [];
  const series = useMemo(() => buildTrainingSeries(metrics), [metrics]);
  const xLabels = useMemo(() => metricLabels(metrics), [metrics]);

  useEffect(() => {
    const node = logRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [logLines.length]);

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }
  if (!job) {
    return (
      <Box
        sx={{
          border: 1,
          borderStyle: "dashed",
          borderColor: "divider",
          px: 2.5,
          py: 4,
          textAlign: "center",
        }}
      >
        <Typography variant="body2" color="text.secondary">
          当前没有任务。启动训练或推理后,这里会显示进度、指标和日志。
        </Typography>
      </Box>
    );
  }

  const status = STATUS[job.status] ?? STATUS.stopped;
  const result = snapshot?.result_maidata;

  return (
    <Box>
      <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
        <StatusTag label={status.label} color={status.color} />
        <Typography variant="body2" color="text.secondary">
          {job.kind === "train" ? "训练" : job.kind === "prepare" ? "数据准备" : "推理"} #{job.id}
        </Typography>
        <Typography
          sx={{ fontFamily: NUMERIC_FONT, fontSize: 12, color: "text.secondary", fontVariantNumeric: "tabular-nums" }}
        >
          {job.elapsed.toFixed(1)}s
          {job.returncode !== null ? ` / exit ${job.returncode}` : ""}
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        {snapshot?.active && onStop ? (
          <Button color="warning" startIcon={<StopCircle />} onClick={onStop} size="small">
            停止任务
          </Button>
        ) : null}
      </Stack>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
          gap: 1.5,
          mt: 2,
        }}
      >
        <Box>
          <Typography variant="caption" color="text.secondary">
            输出目录
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 12, wordBreak: "break-all" }}>
            {job.output_dir}
          </Typography>
        </Box>
        <Box>
          <Typography variant="caption" color="text.secondary">
            日志文件
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 12, wordBreak: "break-all" }}>
            {job.log_path}
          </Typography>
        </Box>
        {Object.entries(job.extra)
          .filter(([key]) => key !== "result_maidata")
          .map(([key, value]) => (
            <Box key={key}>
              <Typography variant="caption" color="text.secondary">
                {key}
              </Typography>
              <Typography variant="body2" sx={{ fontFamily: "monospace", fontSize: 12, wordBreak: "break-all" }}>
                {typeof value === "string" || typeof value === "number" || typeof value === "boolean"
                  ? String(value)
                  : JSON.stringify(value)}
              </Typography>
            </Box>
          ))}
      </Box>

      {result ? (
        <Stack direction="row" spacing={1} sx={{ mt: 2 }} flexWrap="wrap" useFlexGap>
          <Button
            variant="contained"
            size="small"
            startIcon={<Visibility />}
            onClick={() => navigate(`/viewer?file=${encodeURIComponent(result)}`)}
          >
            预览生成的谱面
          </Button>
          <Button
            size="small"
            startIcon={<Download />}
            href={`/api/files?path=${encodeURIComponent(result)}&download=1`}
          >
            下载 maidata
          </Button>
        </Stack>
      ) : null}

      {job.kind === "train" && metrics.length > 0 ? (
        <>
          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" sx={{ mb: 1 }}>
            训练指标
          </Typography>
          <MetricChart series={series} xLabels={xLabels} />
        </>
      ) : null}

      <Divider sx={{ my: 2 }} />
      <Typography variant="subtitle2" sx={{ mb: 1 }}>
        实时日志
      </Typography>
      <Box
        component="pre"
        ref={logRef}
        sx={{
          m: 0,
          p: 1.5,
          maxHeight: 300,
          overflow: "auto",
          bgcolor: (theme) =>
            theme.palette.mode === "dark" ? "rgba(0,0,0,0.28)" : "rgba(17,24,39,0.045)",
          border: 1,
          borderColor: "divider",
          fontSize: 11.5,
          lineHeight: 1.6,
          color: "text.secondary",
          fontFamily: "'SFMono-Regular', Consolas, 'Liberation Mono', monospace",
        }}
      >
        {logLines.length ? logLines.join("\n") : "(等待日志输出...)"}
      </Box>
      <Box component="details" sx={{ mt: 1.25 }}>
        <Typography component="summary" variant="caption" color="text.secondary" sx={{ cursor: "pointer" }}>
          查看完整命令
        </Typography>
        <Typography
          variant="caption"
          component="pre"
          sx={{ mt: 1, whiteSpace: "pre-wrap", wordBreak: "break-all", fontFamily: "monospace", color: "text.secondary" }}
        >
          {job.command.join(" ")}
        </Typography>
      </Box>
    </Box>
  );
}
