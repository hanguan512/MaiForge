import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { Alert, Box, Button, CircularProgress, Stack, Typography } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { ArrowBack, PlayArrow } from "@mui/icons-material";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, formatError } from "../api";
import DifficultyChip from "../components/DifficultyChip";
import Jacket from "../components/Jacket";
import Track from "../components/Track";
import { accentFor, NUMERIC_FONT, PALETTE } from "../theme";
import type { SongDetail } from "../types";

function formatTime(value: number | null | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60);
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function Tag({
  children,
  tone = "default",
}: {
  children: ReactNode;
  tone?: "default" | "warn" | "ok";
}) {
  const theme = useTheme();
  const color =
    tone === "warn"
      ? accentFor(theme.palette.mode, PALETTE.gold)
      : tone === "ok"
        ? accentFor(theme.palette.mode, PALETTE.green)
        : "text.secondary";
  return (
    <Box
      component="span"
      sx={{
        px: 0.75,
        py: 0.15,
        border: 1,
        borderColor: tone === "default" ? "divider" : color,
        color,
        fontSize: 10.5,
        fontWeight: 600,
        lineHeight: 1.5,
        whiteSpace: "nowrap",
      }}
    >
      {children}
    </Box>
  );
}

function Cell({
  label,
  value,
  span,
}: {
  label: string;
  value: ReactNode;
  span?: number;
}) {
  return (
    <Box
      sx={{
        px: 2,
        py: 1.4,
        borderBottom: 1,
        borderColor: "divider",
        gridColumn: span
          ? { xs: `span ${Math.min(span, 2)}`, sm: `span ${span}` }
          : undefined,
        minWidth: 0,
      }}
    >
      <Typography variant="caption" color="text.secondary" sx={{ display: "block" }}>
        {label}
      </Typography>
      <Typography
        sx={{
          fontFamily: NUMERIC_FONT,
          fontWeight: 700,
          fontSize: 16,
          fontVariantNumeric: "tabular-nums",
          mt: 0.25,
          wordBreak: "break-all",
        }}
      >
        {value}
      </Typography>
    </Box>
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
        <CircularProgress size={26} />
      </Box>
    );
  }
  if (error || !song) {
    return (
      <Stack spacing={2}>
        <Alert severity="error">{error ?? "歌曲不存在"}</Alert>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate("/")}
          sx={{ alignSelf: "flex-start" }}
        >
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
        sx={{ alignSelf: "flex-start", ml: -1 }}
      >
        返回曲库
      </Button>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "168px minmax(0, 1fr)" },
          gap: { xs: 2, sm: 3.5 },
          alignItems: "end",
          pb: 3,
          borderBottom: 1,
          borderColor: "divider",
        }}
      >
        <Box sx={{ width: { xs: 132, sm: 168 }, aspectRatio: "1 / 1", maxWidth: "100%" }}>
          <Jacket
            title={song.title}
            id={song.id}
            src={song.cover_url}
            size="100%"
            radius={2}
          />
        </Box>
        <Box sx={{ minWidth: 0 }}>
          <Typography
            sx={{
              fontFamily: (theme) => theme.typography.h2.fontFamily,
              fontSize: { xs: 30, md: 42 },
              lineHeight: 1.15,
              wordBreak: "break-word",
            }}
          >
            {song.title}
          </Typography>
          <Typography variant="body1" color="text.secondary" sx={{ mt: 0.5 }}>
            {song.artist || "未知作者"}
          </Typography>
          <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap sx={{ mt: 1.5 }}>
            <Tag>{song.root}</Tag>
            {confidence !== null && confidence !== undefined ? (
              <Tag tone={confidence >= 0.5 ? "ok" : "warn"}>
                对齐 {(confidence * 100).toFixed(0)}%
              </Tag>
            ) : null}
            {!song.has_audio ? <Tag tone="warn">无音频</Tag> : null}
            {!song.ir_available ? <Tag tone="warn">缺少 IR</Tag> : null}
          </Stack>
          {song.audio_url ? (
            <Box
              component="audio"
              src={song.audio_url}
              controls
              sx={{ mt: 2, width: "100%", maxWidth: 420, height: 36 }}
            />
          ) : null}
        </Box>
      </Box>

      {song.errors.length ? <Alert severity="error">{song.errors.join("; ")}</Alert> : null}
      {song.warnings.length ? <Alert severity="warning">{song.warnings.join("; ")}</Alert> : null}

      <Box>
        <Track
          first
          label="难度"
          hint="点任意一行进入 MajdataView 预览"
          action={
            <Typography variant="caption" color="text.secondary">
              {song.charts.length} 个难度
            </Typography>
          }
        >
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
                  position: "relative",
                  display: "grid",
                  gridTemplateColumns: "auto minmax(0, 1fr) auto",
                  alignItems: "center",
                  gap: { xs: 1.5, md: 2.5 },
                  px: 1,
                  py: 1.5,
                  borderBottom: 1,
                  borderColor: "divider",
                  cursor: "pointer",
                  transition: "background-color 140ms ease",
                  "&::before": {
                    content: '""',
                    position: "absolute",
                    left: 0,
                    top: 9,
                    bottom: 9,
                    width: 2,
                    bgcolor: "primary.main",
                    opacity: 0,
                    transition: "opacity 140ms ease",
                  },
                  "&:hover": { bgcolor: "action.hover" },
                  "&:hover::before": { opacity: 1 },
                  "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: -2 },
                }}
              >
                <DifficultyChip
                  difficulty={chart.difficulty}
                  level={chart.level}
                  size="medium"
                />
                <Box sx={{ minWidth: 0 }}>
                  <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap>
                    {chart.designer && chart.designer !== "-" ? chart.designer : "未署名"}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {chart.notes} 音符
                    <Box component="span" sx={{ display: { xs: "inline", sm: "none" } }}>
                      {chart.density ? ` · ${chart.density.toFixed(2)} n/s` : ""}
                      {chart.duration ? ` · ${formatTime(chart.duration)}` : ""}
                    </Box>
                  </Typography>
                </Box>
                <Stack direction="row" spacing={2} alignItems="center">
                  <Stack
                    direction="row"
                    spacing={2.5}
                    sx={{ display: { xs: "none", sm: "flex" } }}
                  >
                    {[
                      ["密度", chart.density ? chart.density.toFixed(2) : "-"],
                      ["时长", formatTime(chart.duration)],
                    ].map(([label, value]) => (
                      <Box key={label} sx={{ textAlign: "right", minWidth: 52 }}>
                        <Typography
                          sx={{
                            fontFamily: NUMERIC_FONT,
                            fontWeight: 700,
                            fontSize: 14,
                            fontVariantNumeric: "tabular-nums",
                          }}
                        >
                          {value}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {label}
                        </Typography>
                      </Box>
                    ))}
                  </Stack>
                  <PlayArrow sx={{ fontSize: 19, color: "text.secondary" }} />
                </Stack>
              </Box>
            ))}
          </Box>
          {!song.charts.length ? (
            <Alert severity="info" sx={{ mt: 2 }}>
              没有可浏览的难度{song.ir_available ? "" : "（IR 文件缺失）"}。
            </Alert>
          ) : null}
        </Track>

        <Track label="时间轴" hint="来自音频分析结果" accent={PALETTE.gold}>
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: { xs: "1fr 1fr", sm: "repeat(4, 1fr)" },
              borderTop: 1,
              borderLeft: 1,
              borderColor: "divider",
            }}
          >
            <Cell label="BPM" value={bpm !== null ? bpm.toFixed(2) : "-"} />
            <Cell label="first" value={first !== null ? `${first.toFixed(3)} s` : "-"} />
            <Cell label="时长" value={formatTime(duration)} />
            <Cell
              label="对齐"
              value={confidence !== null && confidence !== undefined ? `${(confidence * 100).toFixed(0)}%` : "-"}
            />
            <Cell label="谱面" value={song.charts.length} />
            <Cell label="音频" value={song.has_audio ? "已入库" : "缺失"} />
            <Cell
              label="目录"
              value={<Box component="code" sx={{ fontSize: 11.5 }}>{song.root}</Box>}
              span={2}
            />
          </Box>
        </Track>

        <Track label="元数据" hint="maidata 头部字段" accent={PALETTE.violet}>
          <Box
            component="pre"
            sx={{
              m: 0,
              maxHeight: 260,
              overflow: "auto",
              px: 2,
              py: 1.5,
              border: 1,
              borderColor: "divider",
              bgcolor: (theme) =>
                theme.palette.mode === "dark" ? "rgba(0,0,0,0.24)" : "rgba(17,24,39,0.045)",
              fontSize: 11.5,
              lineHeight: 1.7,
              whiteSpace: "pre-wrap",
              color: "text.secondary",
            }}
          >
            {JSON.stringify(song.metadata, null, 2)}
          </Box>
        </Track>
      </Box>
    </Stack>
  );
}
