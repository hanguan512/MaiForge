import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Paper,
  Stack,
  Typography,
} from "@mui/material";
import { ArrowBack, PlayArrow } from "@mui/icons-material";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, formatError } from "../api";
import DifficultyChip from "../components/DifficultyChip";
import Jacket from "../components/Jacket";
import { NUMERIC_FONT, PALETTE } from "../theme";
import type { SongDetail } from "../types";

function formatTime(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function Tag({ children, tone = "default" }: { children: ReactNode; tone?: "default" | "warn" | "ok" }) {
  const color =
    tone === "warn" ? "warning.main" : tone === "ok" ? "success.main" : "text.secondary";
  return (
    <Box
      component="span"
      sx={{
        px: 0.75,
        py: 0.25,
        border: 1,
        borderColor: color,
        color,
        borderRadius: "2px",
        fontSize: 11,
        fontWeight: 600,
        lineHeight: 1.4,
      }}
    >
      {children}
    </Box>
  );
}

function Metric({ value, label }: { value: string | number; label: string }) {
  return (
    <Box sx={{ minWidth: 62, textAlign: "right" }}>
      <Typography
        sx={{
          fontFamily: NUMERIC_FONT,
          fontWeight: 700,
          fontSize: 15,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
    </Box>
  );
}

function DefinitionRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <Stack
      direction="row"
      justifyContent="space-between"
      spacing={2}
      sx={{ py: 0.9, borderBottom: 1, borderColor: "divider", "&:last-of-type": { borderBottom: "none" } }}
    >
      <Typography variant="body2" color="text.secondary" sx={{ flex: "none" }}>
        {label}
      </Typography>
      <Typography variant="body2" sx={{ textAlign: "right", wordBreak: "break-all" }}>
        {value}
      </Typography>
    </Stack>
  );
}

export default function SongPage() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const root = params.get("root") ?? "";
  const id = params.get("id") ?? "";
  const [song, setSong] = useState<SongDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!id) {
      setError("缺少歌曲 ID");
      setLoading(false);
      return;
    }
    let alive = true;
    setLoading(true);
    setError(null);
    (async () => {
      try {
        const data = await api<SongDetail>(
          `/api/song?root=${encodeURIComponent(root)}&id=${encodeURIComponent(id)}`,
        );
        if (alive) setSong(data);
      } catch (caught) {
        if (alive) setError(formatError(caught));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [root, id]);

  if (loading) {
    return (
      <Box sx={{ py: 12, textAlign: "center" }}>
        <CircularProgress size={28} />
      </Box>
    );
  }
  if (error || !song) {
    return (
      <Stack spacing={2}>
        <Alert severity="error">{error ?? "歌曲不存在"}</Alert>
        <Button startIcon={<ArrowBack />} onClick={() => navigate("/")} sx={{ alignSelf: "flex-start" }}>
          返回曲库
        </Button>
      </Stack>
    );
  }

  const confidence = song.alignment.min_confidence;
  const bpm = typeof song.timing?.bpm === "number" ? song.timing.bpm : null;
  const first = typeof song.timing?.first === "number" ? song.timing.first : null;
  const duration = typeof song.timing?.duration === "number" ? song.timing.duration : null;

  return (
    <Stack spacing={3}>
      <Button
        startIcon={<ArrowBack />}
        onClick={() => navigate("/")}
        size="small"
        sx={{ alignSelf: "flex-start" }}
      >
        返回曲库
      </Button>

      <Box
        sx={{
          display: "flex",
          gap: { xs: 2, md: 3 },
          alignItems: { xs: "flex-start", md: "flex-end" },
          flexWrap: "wrap",
          borderBottom: 1,
          borderColor: "divider",
          pb: 2.5,
        }}
      >
        <Jacket title={song.title} id={song.id} src={song.cover_url} size={148} radius={4} />
        <Box sx={{ flexGrow: 1, minWidth: 240 }}>
          <Typography
            variant="h3"
            sx={{ fontSize: { xs: 30, md: 40 }, lineHeight: 1.15, wordBreak: "break-word" }}
          >
            {song.title}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5 }}>
            {song.artist || "未知作者"}
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ mt: 1.5 }}>
            <Tag>{song.root}</Tag>
            {confidence !== null && confidence !== undefined ? (
              <Tag tone={confidence >= 0.5 ? "ok" : "warn"}>
                对齐 {(confidence * 100).toFixed(0)}%
              </Tag>
            ) : null}
            {!song.has_audio ? <Tag tone="warn">无音频</Tag> : null}
            {!song.ir_available ? <Tag tone="warn">缺少 IR</Tag> : null}
          </Stack>
        </Box>
        {song.audio_url ? (
          <Box sx={{ width: { xs: "100%", md: 340 }, flex: "none" }}>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: "block", mb: 0.5 }}
            >
              音频试听
            </Typography>
            <Box component="audio" src={song.audio_url} controls sx={{ width: "100%", height: 40 }} />
          </Box>
        ) : null}
      </Box>

      {song.errors.length ? <Alert severity="error">{song.errors.join("; ")}</Alert> : null}
      {song.warnings.length ? <Alert severity="warning">{song.warnings.join("; ")}</Alert> : null}

      <Box>
        <Stack direction="row" alignItems="baseline" justifyContent="space-between" sx={{ mb: 1 }}>
          <Typography variant="h5" sx={{ fontSize: 24 }}>
            难度谱面
          </Typography>
          <Typography variant="caption" color="text.secondary">
            点击任意一行进入 MajdataView 风格预览
          </Typography>
        </Stack>
        <Box sx={{ borderTop: 1, borderColor: "divider" }}>
          {song.charts.map((chart) => (
            <Box
              key={chart.index}
              role="button"
              tabIndex={0}
              onClick={() =>
                navigate(
                  `/viewer?root=${encodeURIComponent(song.root)}&id=${encodeURIComponent(song.id)}&index=${chart.index}`,
                )
              }
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  navigate(
                    `/viewer?root=${encodeURIComponent(song.root)}&id=${encodeURIComponent(song.id)}&index=${chart.index}`,
                  );
                }
              }}
              sx={{
                display: "flex",
                alignItems: "center",
                gap: { xs: 1.5, md: 2.5 },
                px: { xs: 0.5, md: 1 },
                py: 1.5,
                borderBottom: 1,
                borderColor: "divider",
                cursor: "pointer",
                transition: "background-color 120ms ease",
                "&:hover": { bgcolor: "action.hover" },
                "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: -2 },
              }}
            >
              <DifficultyChip difficulty={chart.difficulty} level={chart.level} size="medium" />
              <Box sx={{ flexGrow: 1, minWidth: 0 }}>
                <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap>
                  {chart.designer || "未署名"}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {chart.notes} 音符
                  {chart.density ? ` / ${chart.density.toFixed(2)} n/s` : ""}
                  {chart.duration ? ` / ${formatTime(chart.duration)}` : ""}
                </Typography>
              </Box>
              <Stack
                direction="row"
                spacing={2.5}
                sx={{ display: { xs: "none", sm: "flex" }, alignItems: "center" }}
              >
                <Metric value={chart.notes} label="音符" />
                <Metric value={chart.density ? chart.density.toFixed(2) : "-"} label="密度" />
                <Metric value={formatTime(chart.duration)} label="时长" />
              </Stack>
              <Button size="small" startIcon={<PlayArrow />} sx={{ flex: "none" }}>
                预览
              </Button>
            </Box>
          ))}
        </Box>
        {!song.charts.length ? (
          <Box sx={{ py: 4 }}>
            <Alert severity="info">
              没有可浏览的难度{song.ir_available ? "" : "(IR 文件缺失)"}。
            </Alert>
          </Box>
        ) : null}
      </Box>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
          gap: 2.5,
        }}
      >
        <Paper sx={{ p: 2.5 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            时间轴
          </Typography>
          <DefinitionRow label="BPM" value={bpm !== null ? bpm.toFixed(2) : "-"} />
          <DefinitionRow label="first" value={first !== null ? `${first.toFixed(3)}s` : "-"} />
          <DefinitionRow label="时长" value={formatTime(duration)} />
          <DefinitionRow label="目录" value={<code>{song.root}</code>} />
        </Paper>
        <Paper sx={{ p: 2.5 }}>
          <Typography variant="subtitle1" sx={{ mb: 1 }}>
            元数据
          </Typography>
          <Box
            component="pre"
            sx={{
              m: 0,
              maxHeight: 240,
              overflow: "auto",
              fontSize: 12,
              lineHeight: 1.7,
              whiteSpace: "pre-wrap",
              color: "text.secondary",
            }}
          >
            {JSON.stringify(song.metadata, null, 2)}
          </Box>
        </Paper>
      </Box>
    </Stack>
  );
}
