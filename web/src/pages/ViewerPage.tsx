import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Divider,
  IconButton,
  Stack,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  ArrowBack,
  ContentCopy,
  Download,
  OpenInNew,
  Refresh,
  WarningAmber,
} from "@mui/icons-material";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, formatError, formatSeconds } from "../api";
import DifficultyChip from "../components/DifficultyChip";
import Jacket from "../components/Jacket";
import MajdataView from "../components/MajdataView";
import { DIFFICULTY_COLORS, NUMERIC_FONT, PALETTE } from "../theme";
import type { ChartPayload } from "../types";

const KIND_LABELS: Record<string, string> = {
  tap: "Tap",
  hold: "Hold",
  slide: "Slide",
  touch: "Touch",
  touch_hold: "Touch Hold",
};

const KIND_META = [
  { key: "tap", label: "Tap", color: PALETTE.pink },
  { key: "hold", label: "Hold", color: PALETTE.aqua },
  { key: "slide", label: "Slide", color: PALETTE.violet },
  { key: "touch", label: "Touch", color: PALETTE.gold },
  { key: "touch_hold", label: "Touch Hold", color: PALETTE.green },
];

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function Readout({ label, value }: { label: string; value: string }) {
  return (
    <Box>
      <Typography
        sx={{
          fontFamily: NUMERIC_FONT,
          fontWeight: 700,
          fontSize: 17,
          lineHeight: 1.2,
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
      spacing={2}
      justifyContent="space-between"
      sx={{ py: 0.75, borderBottom: 1, borderColor: "divider", "&:last-of-type": { borderBottom: "none" } }}
    >
      <Typography variant="body2" color="text.secondary" sx={{ flex: "none" }}>
        {label}
      </Typography>
      <Typography
        variant="body2"
        sx={{ textAlign: "right", wordBreak: "break-all", fontVariantNumeric: "tabular-nums" }}
      >
        {value}
      </Typography>
    </Stack>
  );
}

export default function ViewerPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const root = params.get("root");
  const id = params.get("id");
  const file = params.get("file");
  const index = params.get("index");

  const [chart, setChart] = useState<ChartPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState(0);
  const [copied, setCopied] = useState(false);
  const [viewerNonce, setViewerNonce] = useState(0);
  const [seek, setSeek] = useState(0);

  useEffect(() => {
    // Moving to another song or difficulty always starts at the beginning.
    setSeek(0);
  }, [root, id, file, index]);

  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    const query = new URLSearchParams();
    if (file) {
      query.set("path", file);
      if (index) query.set("index", index);
    } else if (id) {
      query.set("id", id);
      if (root) query.set("root", root);
      query.set("index", index ?? "5");
    }
    (async () => {
      try {
        const data = await api<ChartPayload>(`/api/chart${file ? "/file" : ""}?${query.toString()}`);
        if (!alive) return;
        setChart(data);
      } catch (caught) {
        if (alive) setError(formatError(caught));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [root, id, file, index]);

  // The embedded player is the real MajdataView WebGL build.  The backend
  // compiles the selected chart into the difficulty slot that build selects
  // and hands the player an mp3 track (silent if the song has no audio).
  // Seeking rebuilds the chart/track from the target time and reloads the
  // iframe; the in-player bar owns playback and difficulty switching.
  const viewerSrc = useMemo(() => {
    if (!chart) return "";
    const source = chart.source;
    const timeline = Math.max(
      1,
      numeric(chart.audio_duration_seconds) ?? chart.first + chart.duration_seconds + 2,
    );
    const seekSeconds = Math.min(Math.max(0, seek), Math.max(0, timeline - 0.25));
    const remaining = Math.max(1, timeline - seekSeconds);

    const options = (
      chart.charts?.length
        ? chart.charts
        : [{ index: chart.index, difficulty: chart.difficulty, level: chart.level }]
    ).map((item) => {
      const colors = DIFFICULTY_COLORS[item.difficulty] ?? DIFFICULTY_COLORS.unknown;
      return { ...item, label: colors.label, bg: colors.bg, fg: colors.fg };
    });

    const query = new URLSearchParams();
    query.set("autoload", "1");
    query.set("title", chart.title);
    query.set("charts", JSON.stringify(options));
    query.set("index", String(chart.index));
    query.set("duration", timeline.toFixed(3));
    query.set("seek", seekSeconds.toFixed(3));

    let maidata =
      source.kind === "file" && source.path
        ? `/api/majdataview/maidata?path=${encodeURIComponent(source.path)}&index=${chart.index}`
        : `/api/majdataview/maidata?root=${encodeURIComponent(source.root ?? "")}&id=${encodeURIComponent(source.song_id ?? "")}&index=${chart.index}`;
    if (seekSeconds > 0) {
      maidata += `&seek=${seekSeconds.toFixed(3)}`;
    }
    query.set("maidata", maidata);

    let audio = `/api/majdataview/audio?duration=${Math.ceil(remaining)}`;
    if (chart.audio_url) {
      const url = new URL(chart.audio_url, window.location.origin);
      const search = new URLSearchParams(url.search);
      search.set("start", seekSeconds.toFixed(3));
      search.set("duration", String(Math.ceil(remaining)));
      audio = `/api/majdataview/audio?${search.toString()}`;
    }
    query.set("audio", audio);

    if (source.kind !== "file" && source.root && source.song_id) {
      query.set(
        "bg",
        `/api/majdataview/bg?root=${encodeURIComponent(source.root)}&song=${encodeURIComponent(source.song_id)}`,
      );
    }
    query.set("v", String(viewerNonce));
    return `/majdataview/?${query.toString()}`;
  }, [chart, seek, viewerNonce]);

  const changeDifficulty = (nextIndex: number) => {
    if (!chart || nextIndex === chart.index) return;
    const next = new URLSearchParams(params);
    next.set("index", String(nextIndex));
    setSeek(0);
    setParams(next, { replace: true });
  };

  const seekTo = (seconds: number) => {
    setSeek(Math.max(0, seconds));
    setViewerNonce((value) => value + 1);
  };

  const downloadMaidata = () => {
    if (!chart) return;
    const blob = new Blob([chart.maidata], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${chart.title}_${chart.difficulty}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const copyMaidata = async () => {
    if (!chart) return;
    await navigator.clipboard.writeText(chart.maidata);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  if (loading) {
    return (
      <Box sx={{ py: 14, textAlign: "center" }}>
        <CircularProgress size={28} />
      </Box>
    );
  }
  if (error || !chart) {
    return (
      <Stack spacing={2}>
        <Alert severity="error">{error ?? "谱面不存在"}</Alert>
        <Button startIcon={<ArrowBack />} onClick={() => navigate("/")} sx={{ alignSelf: "flex-start" }}>
          返回曲库
        </Button>
      </Stack>
    );
  }

  const counts = (chart.stats.counts ?? {}) as Record<string, number>;
  const notesTotal = numeric(chart.stats.notes_total) ?? chart.events.length;
  const density = numeric(chart.stats.notes_per_second);
  const peak = numeric(chart.stats.peak_notes_per_second_1s);
  const bars = numeric(chart.stats.bars);
  const chartDuration = chart.first + chart.duration_seconds;
  const composition = KIND_META.map((item) => ({
    ...item,
    count: counts[item.key] ?? 0,
  }));
  const compositionTotal = composition.reduce((sum, item) => sum + item.count, 0) || chart.events.length || 1;

  return (
    <Stack spacing={2}>
      <Stack
        direction={{ xs: "column", md: "row" }}
        spacing={2}
        alignItems={{ md: "center" }}
        sx={{ borderBottom: 1, borderColor: "divider", pb: 2 }}
      >
        <Stack direction="row" spacing={1.5} alignItems="center" sx={{ flexGrow: 1, minWidth: 0 }}>
          <IconButton onClick={() => navigate(-1)} aria-label="返回" sx={{ flex: "none" }}>
            <ArrowBack />
          </IconButton>
          <Jacket title={chart.title} id={chart.source.song_id ?? chart.title} src={chart.cover_url} size={64} />
          <Box sx={{ minWidth: 0 }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 0 }}>
              <Typography
                variant="h4"
                sx={{
                  fontSize: { xs: 22, md: 28 },
                  lineHeight: 1.2,
                  display: "-webkit-box",
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: "vertical",
                  overflow: "hidden",
                  wordBreak: "break-word",
                }}
              >
                {chart.title}
              </Typography>
              <DifficultyChip difficulty={chart.difficulty} level={chart.level} size="medium" />
              {!chart.validation.ok ? (
                <Tooltip title={chart.validation.errors.join("; ")}>
                  <WarningAmber sx={{ color: "error.main", fontSize: 18, flex: "none" }} />
                </Tooltip>
              ) : null}
            </Stack>
            <Typography variant="body2" color="text.secondary" noWrap>
              {chart.artist || "未知作者"} / {chart.designer || "未署名"}
            </Typography>
          </Box>
        </Stack>

        <Stack direction="row" spacing={3} alignItems="center" sx={{ flex: "none", pl: { md: 0, xs: 0.5 } }}>
          <Readout label="BPM" value={chart.initial_bpm.toFixed(2)} />
          <Readout label="音符" value={String(notesTotal)} />
          <Readout label="时长" value={formatSeconds(chartDuration)} />
          <Button
            variant="outlined"
            size="small"
            startIcon={<Download />}
            onClick={downloadMaidata}
            sx={{ flex: "none" }}
          >
            下载
          </Button>
        </Stack>
      </Stack>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", lg: "minmax(0, 1fr) 372px" },
          gap: 2.5,
          alignItems: "start",
        }}
      >
        <Box sx={{ minWidth: 0 }}>
          <MajdataView
            src={viewerSrc}
            title={`${chart.title} · MajdataView`}
            onDifficultyChange={changeDifficulty}
            onSeek={seekTo}
          />

          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={1}
            alignItems={{ sm: "center" }}
            sx={{ mt: 1.25, px: 0.25 }}
          >
            <Button
              size="small"
              variant="outlined"
              startIcon={<Refresh />}
              onClick={() => setViewerNonce((value) => value + 1)}
              sx={{ color: "text.secondary", borderColor: "divider" }}
            >
              重新载入
            </Button>
            <Button
              size="small"
              startIcon={<OpenInNew />}
              component="a"
              href={viewerSrc}
              target="_blank"
              rel="noreferrer"
            >
              新窗口打开
            </Button>
            <Box sx={{ flexGrow: 1 }} />
            <Typography variant="caption" color="text.secondary">
              难度、进度与暂停在播放器底部调整；流速与偏移仍在 MajdataView 设置中
            </Typography>
          </Stack>
        </Box>

        <Box
          sx={{
            borderTop: { xs: 1, lg: 0 },
            borderLeft: { lg: 1 },
            borderColor: "divider",
            overflow: "hidden",
            position: { lg: "sticky" },
            top: { lg: 16 },
            pt: { xs: 2, lg: 0 },
            pl: { lg: 2.5 },
          }}
        >
          <Tabs
            value={tab}
            onChange={(_, value: number) => setTab(value)}
            variant="fullWidth"
            sx={{ borderBottom: 1, borderColor: "divider" }}
          >
            <Tab label="统计" />
            <Tab label="事件" />
            <Tab label="校验" />
            <Tab label="maidata" />
          </Tabs>

          <Box sx={{ p: 2, maxHeight: { lg: "calc(100vh - 220px)" }, overflow: "auto" }}>
            {tab === 0 ? (
              <Stack spacing={2}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    音符构成
                  </Typography>
                  <Box
                    sx={{
                      mt: 0.75,
                      height: 10,
                      display: "flex",
                      gap: "2px",
                      bgcolor: "divider",
                      borderRadius: "2px",
                      overflow: "hidden",
                    }}
                  >
                    {composition.map((item) =>
                      item.count > 0 ? (
                        <Box
                          key={item.key}
                          title={`${item.label}: ${item.count}`}
                          sx={{
                            width: `${(item.count / compositionTotal) * 100}%`,
                            bgcolor: item.color,
                          }}
                        />
                      ) : null,
                    )}
                  </Box>
                  <Box
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "1fr 1fr",
                      columnGap: 2,
                      rowGap: 0.5,
                      mt: 1.25,
                    }}
                  >
                    {composition.map((item) => (
                      <Stack key={item.key} direction="row" justifyContent="space-between" alignItems="center">
                        <Stack direction="row" spacing={0.75} alignItems="center">
                          <Box sx={{ width: 7, height: 7, borderRadius: "50%", bgcolor: item.color }} />
                          <Typography variant="caption" color="text.secondary">
                            {item.label}
                          </Typography>
                        </Stack>
                        <Typography
                          sx={{ fontFamily: NUMERIC_FONT, fontSize: 12, fontVariantNumeric: "tabular-nums" }}
                        >
                          {item.count}
                        </Typography>
                      </Stack>
                    ))}
                  </Box>
                </Box>

                <Divider />

                <Box>
                  <DefinitionRow label="总音符" value={notesTotal} />
                  <DefinitionRow label="密度" value={density !== null ? `${density.toFixed(2)} n/s` : "-"} />
                  <DefinitionRow label="峰值密度" value={peak !== null ? `${peak} n/s (1s)` : "-"} />
                  <DefinitionRow label="小节数" value={bars !== null ? bars.toFixed(1) : "-"} />
                  <DefinitionRow label="谱面时长" value={formatSeconds(chartDuration)} />
                  <DefinitionRow label="BPM 变化" value={`${chart.bpm_changes.length}`} />
                  <DefinitionRow label="设计师" value={chart.designer || "-"} />
                </Box>

                <Divider />

                <Box>
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 0.5 }}>
                    来源
                  </Typography>
                  <Typography
                    sx={{
                      fontFamily: "monospace",
                      fontSize: 11,
                      color: "text.secondary",
                      wordBreak: "break-all",
                    }}
                  >
                    {chart.source.path ?? `${chart.source.root}/${chart.source.song_id}`}
                  </Typography>
                </Box>
              </Stack>
            ) : null}

            {tab === 1 ? (
              <Box>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>时间</TableCell>
                      <TableCell>类型</TableCell>
                      <TableCell>位置</TableCell>
                      <TableCell>细节</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {chart.events.slice(0, 600).map((event, position) => (
                      <TableRow key={`${event.tick}-${position}`} hover>
                        <TableCell
                          sx={{
                            fontFamily: NUMERIC_FONT,
                            fontSize: 12,
                            whiteSpace: "nowrap",
                            fontVariantNumeric: "tabular-nums",
                          }}
                        >
                          {event.seconds.toFixed(3)}s
                        </TableCell>
                        <TableCell sx={{ fontSize: 12 }}>{KIND_LABELS[event.kind] ?? event.kind}</TableCell>
                        <TableCell sx={{ fontSize: 12 }}>{event.position}</TableCell>
                        <TableCell sx={{ fontSize: 11, color: "text.secondary" }}>
                          {event.segments.length
                            ? event.segments.map((segment) => `${segment.shape}${segment.end}`).join(" → ")
                            : event.flags.join(", ")}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {chart.events.length > 600 ? (
                  <Typography variant="caption" color="text.secondary" sx={{ display: "block", pt: 1 }}>
                    仅显示前 600 个事件(共 {chart.events.length})。
                  </Typography>
                ) : null}
              </Box>
            ) : null}

            {tab === 2 ? (
              <Stack spacing={1.5}>
                {chart.validation.ok ? (
                  <Alert severity="success">语法校验通过,没有错误。</Alert>
                ) : (
                  <Alert severity="error">
                    {chart.validation.errors.map((item) => (
                      <div key={item}>{item}</div>
                    ))}
                  </Alert>
                )}
                {chart.validation.warnings.length ? (
                  <Alert severity="warning">
                    {chart.validation.warnings.map((item) => (
                      <div key={item}>{item}</div>
                    ))}
                  </Alert>
                ) : null}
              </Stack>
            ) : null}

            {tab === 3 ? (
              <Box>
                <Stack direction="row" spacing={1} sx={{ mb: 1.5 }}>
                  <Button size="small" startIcon={<ContentCopy />} onClick={() => void copyMaidata()}>
                    {copied ? "已复制" : "复制"}
                  </Button>
                  <Button size="small" startIcon={<Download />} onClick={downloadMaidata}>
                    下载
                  </Button>
                </Stack>
                <Box
                  component="pre"
                  sx={{
                    m: 0,
                    p: 1.5,
                    maxHeight: "56vh",
                    overflow: "auto",
                    fontSize: 12,
                    lineHeight: 1.6,
                    bgcolor: (theme) =>
                theme.palette.mode === "dark" ? "rgba(0,0,0,0.24)" : "rgba(17,24,39,0.045)",
                    border: 1,
                    borderColor: "divider",
                    color: "text.secondary",
                  }}
                >
                  {chart.maidata}
                </Box>
              </Box>
            ) : null}
          </Box>
        </Box>
      </Box>
    </Stack>
  );
}
