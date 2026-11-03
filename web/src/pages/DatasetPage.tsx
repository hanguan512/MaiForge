import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  FormControlLabel,
  IconButton,
  LinearProgress,
  Stack,
  Switch,
  Tab,
  Tabs,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  AutoFixHigh,
  CloudUpload,
  DeleteOutline,
  FolderZip,
  Refresh,
  Search,
} from "@mui/icons-material";
import { alpha, useTheme } from "@mui/material/styles";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, formatError, postJson } from "../api";
import DifficultyChip from "../components/DifficultyChip";
import Jacket from "../components/Jacket";
import OnlineCatalogue from "../components/OnlineCatalogue";
import PageHeader from "../components/PageHeader";
import { ReadyMarks } from "../components/ReadyMark";
import StatStrip from "../components/StatStrip";
import { accentFor, PALETTE } from "../theme";
import type {
  DatasetImportItem,
  DatasetImportResponse,
  DatasetSong,
  DatasetSummary,
  JobSnapshot,
} from "../types";

type LibraryFilter = "all" | "ready" | "missing" | "noaudio";
type TabKey = "library" | "online" | "import";

const FILTERS: Array<{ key: LibraryFilter; label: string }> = [
  { key: "all", label: "全部" },
  { key: "ready", label: "可训练" },
  { key: "missing", label: "缺数据" },
  { key: "noaudio", label: "无音频" },
];

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function isReady(song: DatasetSong): boolean {
  return song.has_audio && song.sample_ready && song.features_ready;
}

function SongRow({
  song,
  canBackfill,
  preparing,
  onBackfill,
  onPreview,
  onDetail,
}: {
  song: DatasetSong;
  canBackfill: boolean;
  preparing: boolean;
  onBackfill: () => void;
  onPreview: () => void;
  onDetail: () => void;
}) {
  return (
    <Box
      sx={{
        position: "relative",
        display: "grid",
        gridTemplateColumns: {
          xs: "minmax(0, 1fr) auto",
          md: "minmax(0, 1.35fr) minmax(190px, 1fr) 132px auto",
        },
        columnGap: { xs: 1.25, md: 2.5 },
        rowGap: 1,
        alignItems: "center",
        py: 1.3,
        borderBottom: 1,
        borderColor: "divider",
        transition: "background-color 140ms ease",
        "&::before": {
          content: '""',
          position: "absolute",
          left: 0,
          top: 10,
          bottom: 10,
          width: 2,
          bgcolor: "primary.main",
          opacity: 0,
          transition: "opacity 140ms ease",
        },
        "&:hover": { bgcolor: alpha(PALETTE.text, 0.028) },
        "&:hover::before": { opacity: 1 },
        "&:focus-within::before": { opacity: 1 },
      }}
    >
      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ minWidth: 0 }}>
        <Jacket title={song.title} id={song.id} src={song.cover_url} size={42} radius={1} />
        <Box sx={{ minWidth: 0 }}>
          <Typography
            sx={{ fontSize: 14, fontWeight: 600, lineHeight: 1.35 }}
            noWrap
            title={song.title}
          >
            {song.title}
          </Typography>
          <Typography variant="caption" color="text.secondary" noWrap sx={{ display: "block" }}>
            {song.artist || "未知作者"}
            <Box component="span" sx={{ color: "text.disabled" }}>
              {"  /  "}
              {song.id}
            </Box>
            {song.has_audio ? "" : "  /  无音频"}
          </Typography>
        </Box>
      </Stack>

      <Box sx={{ display: { xs: "none", md: "flex" }, gap: 0.5, flexWrap: "wrap" }}>
        {song.charts.map((chart) => (
          <DifficultyChip
            key={chart.index}
            difficulty={chart.difficulty}
            level={chart.level}
          />
        ))}
      </Box>

      <Box sx={{ display: { xs: "none", md: "block" } }}>
        <ReadyMarks
          hasAudio={song.has_audio}
          sampleReady={song.sample_ready}
          featuresReady={song.features_ready}
        />
      </Box>

      <Stack
        direction="row"
        spacing={0.25}
        justifyContent="flex-end"
        sx={{ gridColumn: { xs: "1 / -1", md: "auto" }, gridRow: { xs: 2, md: "auto" } }}
      >
        {canBackfill ? (
          <Tooltip title="补全样本与 log-mel 特征">
            <span>
              <IconButton
                size="small"
                aria-label={`补全 ${song.title}`}
                disabled={preparing}
                onClick={onBackfill}
              >
                <AutoFixHigh sx={{ fontSize: 17 }} />
              </IconButton>
            </span>
          </Tooltip>
        ) : null}
        <Button size="small" onClick={onPreview}>
          预览
        </Button>
        <Button size="small" onClick={onDetail}>
          详情
        </Button>
      </Stack>
    </Box>
  );
}

export default function DatasetPage() {
  const theme = useTheme();
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const tab = (params.get("tab") as TabKey) || "library";
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [align, setAlign] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [importing, setImporting] = useState(false);
  const [results, setResults] = useState<DatasetImportItem[] | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [prepareNote, setPrepareNote] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<LibraryFilter>("all");
  const inputRef = useRef<HTMLInputElement | null>(null);
  const prepareTimer = useRef<number | null>(null);

  const load = async (): Promise<void> => {
    try {
      const data = await api<DatasetSummary>("/api/datasets");
      setSummary(data);
      setError(null);
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
    return () => {
      if (prepareTimer.current !== null) {
        window.clearInterval(prepareTimer.current);
        prepareTimer.current = null;
      }
    };
  }, []);

  const setTab = (next: TabKey) => {
    const updated = new URLSearchParams(params);
    if (next === "library") updated.delete("tab");
    else updated.set("tab", next);
    setParams(updated, { replace: true });
  };

  const addFiles = (incoming: FileList | File[] | null) => {
    if (!incoming) return;
    const candidates = Array.from(incoming);
    const zips = candidates.filter((file) => file.name.toLowerCase().endsWith(".zip"));
    if (zips.length !== candidates.length) {
      setError("只接受 .zip 压缩包，其余文件已忽略");
    } else {
      setError(null);
    }
    setFiles((current) => {
      const keys = new Set(current.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
      const merged = [...current];
      for (const file of zips) {
        const key = `${file.name}:${file.size}:${file.lastModified}`;
        if (!keys.has(key)) {
          keys.add(key);
          merged.push(file);
        }
      }
      return merged;
    });
  };

  const removeFile = (index: number) =>
    setFiles((current) => current.filter((_, position) => position !== index));

  const totalSize = useMemo(
    () => files.reduce((sum, file) => sum + file.size, 0),
    [files],
  );

  const songs = summary?.songs ?? [];
  const missingSongs = useMemo(
    () => songs.filter((song) => song.has_audio && (!song.sample_ready || !song.features_ready)),
    [songs],
  );
  const readyCount = songs.filter(isReady).length;
  const chartCount = songs.reduce((total, song) => total + song.chart_count, 0);
  const featureReadyCount = songs.filter((song) => song.features_ready).length;

  const visibleSongs = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return songs.filter((song) => {
      if (filter === "ready" && !isReady(song)) return false;
      if (filter === "missing" && (!song.has_audio || isReady(song))) return false;
      if (filter === "noaudio" && song.has_audio) return false;
      if (!needle) return true;
      return (
        song.title.toLowerCase().includes(needle) ||
        song.artist.toLowerCase().includes(needle) ||
        song.id.toLowerCase().includes(needle)
      );
    });
  }, [songs, filter, query]);

  const counts: Record<LibraryFilter, number> = {
    all: songs.length,
    ready: readyCount,
    missing: missingSongs.length,
    noaudio: songs.filter((song) => !song.has_audio).length,
  };

  const backfill = async (songIds: string[]) => {
    if (!summary || !songIds.length || preparing) return;
    setPreparing(true);
    setError(null);
    setPrepareNote(`正在为 ${songIds.length} 首歌曲生成训练样本与 log-mel 特征…`);
    try {
      const started = await postJson<JobSnapshot>("/api/train/prepare", {
        root: summary.processed_root,
        song_ids: songIds,
      });
      const jobId = started.job?.id;
      if (!jobId) {
        setPreparing(false);
        await load();
        return;
      }
      if (prepareTimer.current !== null) {
        window.clearInterval(prepareTimer.current);
      }
      prepareTimer.current = window.setInterval(() => {
        void (async () => {
          try {
            const current = await api<JobSnapshot>("/api/jobs/current");
            const job = current.job;
            if (job && job.id === jobId && current.active) return;
            if (prepareTimer.current !== null) {
              window.clearInterval(prepareTimer.current);
              prepareTimer.current = null;
            }
            setPreparing(false);
            await load();
            if (job?.id === jobId && job.status === "completed") {
              setPrepareNote(
                `已为 ${songIds.length} 首歌曲补全样本与 log-mel 特征，可以直接开始训练。`,
              );
            } else if (job?.id === jobId && (job.status === "failed" || job.status === "stopped")) {
              setError(
                `数据补全${job.status === "failed" ? "失败" : "已停止"}，请到训练页查看任务日志。`,
              );
            }
          } catch (caught) {
            if (prepareTimer.current !== null) {
              window.clearInterval(prepareTimer.current);
              prepareTimer.current = null;
            }
            setPreparing(false);
            setError(formatError(caught));
          }
        })();
      }, 1500);
    } catch (caught) {
      setPreparing(false);
      setPrepareNote(null);
      setError(formatError(caught));
    }
  };

  const upload = async () => {
    if (!files.length) {
      setError("请先选择至少一个 zip 压缩包");
      return;
    }
    setImporting(true);
    setError(null);
    setResults(null);
    try {
      const body = new FormData();
      files.forEach((file) => body.append("files", file));
      body.append("align", align ? "true" : "false");
      const response = await api<DatasetImportResponse>("/api/datasets/import", {
        method: "POST",
        body,
      });
      setResults(response.results);
      setFiles([]);
      await load();
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setImporting(false);
    }
  };

  const successes = results?.filter((item) => item.ok).length ?? 0;
  const failures = (results?.length ?? 0) - successes;

  return (
    <Stack spacing={3}>
      <PageHeader
        title="数据集"
        description="本地曲库、在线谱库和 zip 导入都在这里。入库后自动出现在 data/processed，样本与 log-mel 特征可一键补全。"
        actions={
          <Button startIcon={<Refresh />} onClick={() => void load()} disabled={loading}>
            刷新
          </Button>
        }
      />

      <StatStrip
        items={[
          { label: "已入库歌曲", value: summary?.processed_count ?? "-", accent: PALETTE.aqua },
          { label: "谱面总数", value: chartCount, accent: PALETTE.gold },
          { label: "可训练", value: readyCount, accent: PALETTE.green },
          { label: "特征就绪", value: `${featureReadyCount}/${songs.length || 0}` },
          { label: "入库位置", value: "processed", detail: summary?.processed_root ?? "data/processed" },
        ]}
      />

      {error ? <Alert severity="error">{error}</Alert> : null}
      {prepareNote ? (
        <Alert severity="info" onClose={() => setPrepareNote(null)}>
          <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
            <span>{prepareNote}</span>
            {preparing ? <CircularProgress size={13} /> : null}
          </Stack>
        </Alert>
      ) : null}

      <Box>
        <Tabs value={tab} onChange={(_, value: TabKey) => setTab(value)}>
          <Tab value="library" label={`本地曲库  ${songs.length}`} />
          <Tab value="online" label="在线谱库" />
          <Tab value="import" label="导入压缩包" />
        </Tabs>
        <Box sx={{ borderBottom: 1, borderColor: "divider" }} />
      </Box>

      {tab === "library" ? (
        <Box>
          <Stack
            direction={{ xs: "column", md: "row" }}
            spacing={{ xs: 1.5, md: 2 }}
            alignItems={{ md: "center" }}
            sx={{ mb: 1.5 }}
          >
            <TextField
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索曲名、作者或 ID"
              sx={{ width: { xs: "100%", md: 300 } }}
              InputProps={{
                startAdornment: <Search sx={{ fontSize: 17, mr: 1, color: "text.disabled" }} />,
              }}
            />
            <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
              {FILTERS.map((item) => {
                const active = filter === item.key;
                return (
                  <Chip
                    key={item.key}
                    clickable
                    label={`${item.label} ${counts[item.key]}`}
                    onClick={() => setFilter(item.key)}
                    sx={{
                      bgcolor: active ? "text.primary" : "transparent",
                      color: active ? "background.default" : "text.secondary",
                      border: 1,
                      borderColor: active ? "text.primary" : "divider",
                      "&:hover": {
                        bgcolor: active ? "text.primary" : alpha(PALETTE.text, 0.06),
                      },
                    }}
                  />
                );
              })}
            </Stack>
            <Box sx={{ flexGrow: 1 }} />
            <Button
              variant="outlined"
              size="small"
              startIcon={
                preparing ? <CircularProgress size={13} /> : <AutoFixHigh sx={{ fontSize: 17 }} />
              }
              disabled={preparing || !missingSongs.length}
              onClick={() => void backfill(missingSongs.map((song) => song.id))}
            >
              {preparing
                ? "正在补全…"
                : missingSongs.length
                  ? `补全样本与特征 (${missingSongs.length})`
                  : "样本与特征已齐全"}
            </Button>
          </Stack>

          {preparing ? <LinearProgress sx={{ mb: 1 }} /> : null}

          <Box
            sx={{
              display: { xs: "none", md: "grid" },
              gridTemplateColumns: "minmax(0, 1.35fr) minmax(190px, 1fr) 132px 128px",
              columnGap: 2.5,
              pt: 1,
              pb: 0.75,
              borderBottom: 1,
              borderColor: "divider",
            }}
          >
            <Typography variant="caption" color="text.disabled">
              歌曲
            </Typography>
            <Typography variant="caption" color="text.disabled">
              难度
            </Typography>
            <Typography variant="caption" color="text.disabled">
              训练数据
            </Typography>
            <Typography
              variant="caption"
              color="text.disabled"
              sx={{ textAlign: "right" }}
            >
              操作
            </Typography>
          </Box>

          {loading ? (
            <Box sx={{ py: 8, textAlign: "center" }}>
              <CircularProgress size={24} />
            </Box>
          ) : visibleSongs.length ? (
            visibleSongs.map((song) => (
              <SongRow
                key={song.id}
                song={song}
                canBackfill={song.has_audio && (!song.sample_ready || !song.features_ready)}
                preparing={preparing}
                onBackfill={() => void backfill([song.id])}
                onPreview={() =>
                  navigate(
                    `/viewer?root=${encodeURIComponent(summary?.processed_root ?? "data/processed")}&id=${encodeURIComponent(song.id)}&index=${song.charts[0]?.index ?? 5}`,
                  )
                }
                onDetail={() =>
                  navigate(
                    `/song?root=${encodeURIComponent(summary?.processed_root ?? "data/processed")}&id=${encodeURIComponent(song.id)}`,
                  )
                }
              />
            ))
          ) : songs.length ? (
            <Box sx={{ py: 6, textAlign: "center" }}>
              <Typography variant="body2" color="text.secondary">
                没有匹配的歌曲。
              </Typography>
              <Button
                size="small"
                sx={{ mt: 1 }}
                onClick={() => {
                  setQuery("");
                  setFilter("all");
                }}
              >
                清除筛选
              </Button>
            </Box>
          ) : (
            <Box
              sx={{
                border: 1,
                borderStyle: "dashed",
                borderColor: "divider",
                px: 3,
                py: 6,
                textAlign: "center",
              }}
            >
              <Typography variant="body2" color="text.secondary">
                还没有导入歌曲。去「在线谱库」挑几首，或在「导入压缩包」里拖入 zip。
              </Typography>
              <Button sx={{ mt: 1.5 }} variant="outlined" onClick={() => setTab("online")}>
                打开在线谱库
              </Button>
            </Box>
          )}
        </Box>
      ) : null}

      {tab === "online" ? <OnlineCatalogue onImported={load} /> : null}

      {tab === "import" ? (
        <Box sx={{ maxWidth: 900 }}>
          <Box
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              addFiles(event.dataTransfer.files);
            }}
            onClick={() => inputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                inputRef.current?.click();
              }
            }}
            sx={{
              border: 1,
              borderStyle: "dashed",
              borderColor: dragging ? "primary.main" : "divider",
              bgcolor: dragging ? alpha(PALETTE.pink, 0.05) : "transparent",
              px: { xs: 2.5, md: 4 },
              py: { xs: 4, md: 5 },
              display: "flex",
              alignItems: "center",
              gap: 2.5,
              cursor: "pointer",
              transition: "border-color 140ms ease, background-color 140ms ease",
              "&:hover": { borderColor: "primary.main" },
              "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: 2 },
            }}
          >
            <CloudUpload sx={{ fontSize: 38, color: "text.secondary", flex: "none" }} />
            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ fontSize: 17, fontWeight: 600 }}>
                把 zip 拖到这里，或点击选择
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.35 }}>
                一个压缩包 = 一首歌，可多选批量导入；同名歌曲会追加 _2、_3，不会覆盖已有数据。
              </Typography>
            </Box>
            <Box sx={{ flexGrow: 1 }} />
            <Button variant="outlined" sx={{ flex: "none", display: { xs: "none", sm: "inline-flex" } }}>
              选择文件
            </Button>
            <input
              ref={inputRef}
              type="file"
              hidden
              multiple
              accept=".zip,application/zip"
              onChange={(event) => {
                addFiles(event.target.files);
                event.target.value = "";
              }}
            />
          </Box>

          {files.length ? (
            <Box sx={{ mt: 2, borderTop: 1, borderColor: "divider" }}>
              {files.map((file, index) => (
                <Stack
                  key={`${file.name}:${file.size}:${file.lastModified}`}
                  direction="row"
                  spacing={1.25}
                  alignItems="center"
                  sx={{ py: 0.9, borderBottom: 1, borderColor: "divider" }}
                >
                  <FolderZip sx={{ fontSize: 17, color: "text.secondary" }} />
                  <Typography variant="body2" sx={{ flexGrow: 1, wordBreak: "break-all" }}>
                    {file.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {formatBytes(file.size)}
                  </Typography>
                  <Tooltip title="移除">
                    <IconButton
                      size="small"
                      onClick={() => removeFile(index)}
                      aria-label={`移除 ${file.name}`}
                    >
                      <DeleteOutline sx={{ fontSize: 18 }} />
                    </IconButton>
                  </Tooltip>
                </Stack>
              ))}
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", pt: 1 }}>
                共 {files.length} 个压缩包 / {formatBytes(totalSize)}
              </Typography>
            </Box>
          ) : null}

          <Stack
            direction={{ xs: "column", sm: "row" }}
            spacing={2}
            alignItems={{ sm: "center" }}
            sx={{ mt: 2 }}
          >
            <FormControlLabel
              control={<Switch checked={align} onChange={(event) => setAlign(event.target.checked)} />}
              label={
                <Box>
                  <Typography variant="body2">导入后分析 BPM / 对齐</Typography>
                  <Typography variant="caption" color="text.secondary">
                    需要 numpy 与 ffmpeg，批量导入会明显变慢
                  </Typography>
                </Box>
              }
            />
            <Box sx={{ flexGrow: 1 }} />
            <Button
              variant="contained"
              size="large"
              startIcon={<CloudUpload />}
              disabled={importing || !files.length}
              onClick={() => void upload()}
            >
              {importing
                ? "正在导入…"
                : files.length
                  ? `导入 ${files.length} 个压缩包`
                  : "导入压缩包"}
            </Button>
          </Stack>

          {importing ? <LinearProgress sx={{ mt: 2 }} /> : null}

          {results ? (
            <Box sx={{ mt: 2.5 }}>
              <Typography variant="caption" color="text.secondary">
                导入结果 · 成功 {successes} 个{failures ? ` · 失败 ${failures} 个` : ""}
              </Typography>
              <Box sx={{ mt: 1, borderTop: 1, borderColor: "divider" }}>
                {results.map((item, index) => {
                  const messages = [...item.errors, ...item.warnings].join("; ");
                  return (
                    <Box
                      key={`${item.file}-${index}`}
                      sx={{
                        display: "grid",
                        gridTemplateColumns: "minmax(0, 1fr) auto",
                        gap: 1.5,
                        py: 1,
                        borderBottom: 1,
                        borderColor: "divider",
                        alignItems: "baseline",
                      }}
                    >
                      <Box sx={{ minWidth: 0 }}>
                        <Typography variant="body2" sx={{ wordBreak: "break-all" }}>
                          {item.file}
                        </Typography>
                        <Typography
                          variant="caption"
                          sx={{
                            color: item.ok ? "text.secondary" : "error.main",
                            display: "block",
                            wordBreak: "break-word",
                          }}
                        >
                          {item.ok
                            ? `${item.title ?? item.song_id} (${item.song_id}) · ${item.chart_count} 张谱面${messages ? ` · ${messages}` : ""}`
                            : messages || "导入失败"}
                        </Typography>
                      </Box>
                      <Box
                        component="span"
                        sx={{
                          fontSize: 11,
                          fontWeight: 700,
                          color: accentFor(
                            theme.palette.mode,
                            item.ok ? PALETTE.green : PALETTE.red,
                          ),
                        }}
                      >
                        {item.ok ? "成功" : "失败"}
                      </Box>
                    </Box>
                  );
                })}
              </Box>
            </Box>
          ) : null}
        </Box>
      ) : null}
    </Stack>
  );
}
