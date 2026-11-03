import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  AutoFixHigh,
  CloudUpload,
  DeleteOutline,
  FolderZip,
  Refresh,
  Visibility,
} from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { api, formatError, postJson } from "../api";
import DifficultyChip from "../components/DifficultyChip";
import Jacket from "../components/Jacket";
import OnlineCatalogue from "../components/OnlineCatalogue";
import PageHeader from "../components/PageHeader";
import StatStrip from "../components/StatStrip";
import { PALETTE } from "../theme";
import type {
  DatasetImportItem,
  DatasetImportResponse,
  DatasetSummary,
  JobSnapshot,
} from "../types";

function formatBytes(value: number): string {
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value: number | null | undefined): string {
  if (!value) return "-";
  return new Date(value * 1000).toLocaleString();
}

function SectionTitle({
  title,
  detail,
  action,
}: {
  title: string;
  detail?: string;
  action?: ReactNode;
}) {
  return (
    <Stack
      direction="row"
      alignItems="center"
      justifyContent="space-between"
      spacing={2}
      sx={{ mb: 1.5 }}
    >
      <Stack direction="row" alignItems="baseline" spacing={1.5} sx={{ minWidth: 0 }}>
        <Typography variant="h6" sx={{ fontSize: 18, whiteSpace: "nowrap" }}>
          {title}
        </Typography>
        {detail ? (
          <Typography variant="caption" color="text.secondary" noWrap>
            {detail}
          </Typography>
        ) : null}
      </Stack>
      {action}
    </Stack>
  );
}

export default function DatasetPage() {
  const navigate = useNavigate();
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

  const addFiles = (incoming: FileList | File[] | null) => {
    if (!incoming) return;
    const candidates = Array.from(incoming);
    const zips = candidates.filter((file) => file.name.toLowerCase().endsWith(".zip"));
    if (zips.length !== candidates.length) {
      setError("只接受 .zip 压缩包,其余文件已忽略");
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

  const missingSongs = useMemo(
    () =>
      (summary?.songs ?? []).filter(
        (song) => song.has_audio && (!song.sample_ready || !song.features_ready),
      ),
    [summary],
  );
  const featureReadyCount = summary?.songs.filter((song) => song.features_ready).length ?? 0;

  const backfill = async (songIds: string[]) => {
    if (!summary || !songIds.length || preparing) return;
    setPreparing(true);
    setError(null);
    setPrepareNote(`正在为 ${songIds.length} 首歌曲生成训练样本与 log-mel 特征...`);
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
    <Stack spacing={2.5}>
      <PageHeader
        title="数据集"
        description="从 AWMC-MaiArchive 在线检索并下载谱面,也可以一次拖入多个本地 zip。入库后自动出现在 data/processed。"
        actions={
          <Button startIcon={<Refresh />} onClick={() => void load()} disabled={loading}>
            刷新
          </Button>
        }
      />

      <StatStrip
        items={[
          { label: "已入库歌曲", value: summary?.processed_count ?? "-", accent: PALETTE.aqua },
          { label: "原始目录", value: summary?.raw_count ?? "-" },
          { label: "入库位置", value: "processed", detail: summary?.processed_root ?? "data/processed" },
          { label: "导入格式", value: "zip", detail: "单个压缩包上限 1 GB" },
        ]}
      />

      {error ? <Alert severity="error">{error}</Alert> : null}
      {prepareNote ? (
        <Alert severity="info" onClose={() => setPrepareNote(null)}>
          {prepareNote}
        </Alert>
      ) : null}

      <OnlineCatalogue onImported={load} />

      <Box
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: "4px",
          bgcolor: "background.paper",
          p: { xs: 2, md: 2.5 },
        }}
      >
        <SectionTitle title="导入歌曲" detail="同名歌曲会自动追加 _2、_3,不会覆盖已有数据" />

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
            borderRadius: "3px",
            px: 3,
            py: { xs: 4, md: 5.5 },
            textAlign: "center",
            cursor: "pointer",
            bgcolor: dragging ? "rgba(255,95,158,0.05)" : "transparent",
            transition: "border-color 120ms ease, background-color 120ms ease",
            "&:hover": { borderColor: "primary.main" },
            "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: 2 },
          }}
        >
          <CloudUpload sx={{ fontSize: 44, color: "text.secondary" }} />
          <Typography variant="h6" sx={{ mt: 1, fontSize: 20 }}>
            把 zip 拖到这里,或点击选择
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
            可多选,支持批量导入
          </Typography>
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
                <FolderZip sx={{ fontSize: 18, color: "text.secondary" }} />
                <Typography variant="body2" sx={{ flexGrow: 1, wordBreak: "break-all" }}>
                  {file.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatBytes(file.size)}
                </Typography>
                <Tooltip title="移除">
                  <IconButton size="small" onClick={() => removeFile(index)} aria-label={`移除 ${file.name}`}>
                    <DeleteOutline fontSize="small" />
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
                  需要 numpy 与 ffmpeg,批量导入会明显变慢
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
              ? "正在导入..."
              : files.length
                ? `导入 ${files.length} 个压缩包`
                : "导入压缩包"}
          </Button>
        </Stack>

        {importing ? <LinearProgress sx={{ mt: 2 }} /> : null}
      </Box>

      {results ? (
        <Box sx={{ border: 1, borderColor: "divider", borderRadius: "4px", bgcolor: "background.paper" }}>
          <Box sx={{ px: 2.5, pt: 2, pb: 1.5 }}>
            <SectionTitle
              title="导入结果"
              detail={`成功 ${successes} 个${failures ? `,失败 ${failures} 个` : ""}`}
            />
          </Box>
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>压缩包</TableCell>
                  <TableCell>状态</TableCell>
                  <TableCell>歌曲</TableCell>
                  <TableCell align="right">谱面数</TableCell>
                  <TableCell>信息</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.map((item, index) => (
                  <TableRow key={`${item.file}-${index}`}>
                    <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>{item.file}</TableCell>
                    <TableCell>
                      <Chip
                        size="small"
                        color={item.ok ? "success" : "error"}
                        label={item.ok ? "成功" : "失败"}
                      />
                    </TableCell>
                    <TableCell>
                      {item.ok ? `${item.title ?? item.song_id} (${item.song_id})` : "-"}
                    </TableCell>
                    <TableCell align="right">{item.ok ? item.chart_count : "-"}</TableCell>
                    <TableCell sx={{ fontSize: 12, color: item.ok ? "text.secondary" : "error.main" }}>
                      {[...item.errors, ...item.warnings].join("; ") || "-"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </Box>
      ) : null}

      <Box sx={{ border: 1, borderColor: "divider", borderRadius: "4px", bgcolor: "background.paper" }}>
        <Box sx={{ px: 2.5, pt: 2, pb: 1.5 }}>
          <SectionTitle
            title="已入库歌曲"
            detail={`${summary?.processed_root ?? "data/processed"} / ${summary?.processed_count ?? 0} 首`}
            action={
              <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
                <Typography variant="caption" color="text.secondary" sx={{ whiteSpace: "nowrap" }}>
                  特征 {featureReadyCount}/{summary?.processed_count ?? 0}
                </Typography>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={
                    preparing ? <CircularProgress size={14} /> : <AutoFixHigh fontSize="small" />
                  }
                  disabled={preparing || !missingSongs.length}
                  onClick={() => void backfill(missingSongs.map((song) => song.id))}
                >
                  {preparing
                    ? "正在补全..."
                    : missingSongs.length
                      ? `补全样本与特征 (${missingSongs.length})`
                      : "样本与特征已齐全"}
                </Button>
              </Stack>
            }
          />
        </Box>
        {loading ? (
          <Box sx={{ py: 6, textAlign: "center" }}>
            <CircularProgress size={26} />
          </Box>
        ) : summary?.songs.length ? (
          <Box sx={{ overflowX: "auto" }}>
            <Table size="small" sx={{ tableLayout: "fixed", minWidth: 900 }}>
              <TableHead>
                <TableRow>
                  <TableCell sx={{ width: "30%" }}>歌曲</TableCell>
                  <TableCell sx={{ width: "18%", whiteSpace: "nowrap" }}>难度</TableCell>
                  <TableCell align="right" sx={{ width: "8%", whiteSpace: "nowrap" }}>
                    谱面数
                  </TableCell>
                  <TableCell sx={{ width: "12%", whiteSpace: "nowrap" }}>训练数据</TableCell>
                  <TableCell sx={{ width: "17%", whiteSpace: "nowrap" }}>更新时间</TableCell>
                  <TableCell align="right" sx={{ width: "15%" }}>
                    操作
                  </TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {summary.songs.map((song) => (
                  <TableRow key={song.id} hover>
                    <TableCell>
                      <Stack direction="row" spacing={1.25} alignItems="center">
                        <Jacket title={song.title} id={song.id} src={song.cover_url} size={36} radius={2} />
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap>
                            {song.title}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" noWrap>
                            {song.artist || "未知作者"} / {song.id}
                            {song.has_audio ? "" : " / 无音频"}
                          </Typography>
                        </Box>
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {song.charts.map((chart) => (
                          <DifficultyChip
                            key={chart.index}
                            difficulty={chart.difficulty}
                            level={chart.level}
                          />
                        ))}
                      </Stack>
                    </TableCell>
                    <TableCell align="right">{song.chart_count}</TableCell>
                    <TableCell>
                      {song.has_audio ? (
                        <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                          <Tooltip
                            title={song.sample_ready ? "训练样本已导出" : "尚未导出训练样本"}
                          >
                            <Chip
                              size="small"
                              label="样本"
                              color={song.sample_ready ? "success" : "default"}
                              variant={song.sample_ready ? "filled" : "outlined"}
                            />
                          </Tooltip>
                          <Tooltip
                            title={
                              song.features_ready
                                ? "log-mel 特征已生成"
                                : "尚未生成 log-mel 特征"
                            }
                          >
                            <Chip
                              size="small"
                              label="特征"
                              color={song.features_ready ? "success" : "default"}
                              variant={song.features_ready ? "filled" : "outlined"}
                            />
                          </Tooltip>
                        </Stack>
                      ) : (
                        <Typography variant="caption" color="text.secondary">
                          无音频
                        </Typography>
                      )}
                    </TableCell>
                    <TableCell sx={{ whiteSpace: "nowrap", fontSize: 12, color: "text.secondary" }}>
                      {formatDate(song.updated_at)}
                    </TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={0.5} justifyContent="flex-end">
                        {song.has_audio && (!song.sample_ready || !song.features_ready) ? (
                          <Tooltip title="补全样本与 log-mel 特征">
                            <span>
                              <IconButton
                                size="small"
                                aria-label={`补全 ${song.title}`}
                                disabled={preparing}
                                onClick={() => void backfill([song.id])}
                              >
                                <AutoFixHigh fontSize="small" />
                              </IconButton>
                            </span>
                          </Tooltip>
                        ) : null}
                        <Button
                          size="small"
                          startIcon={<Visibility />}
                          sx={{ whiteSpace: "nowrap", flexShrink: 0 }}
                          onClick={() =>
                            navigate(
                              `/viewer?root=${encodeURIComponent(summary.processed_root)}&id=${encodeURIComponent(song.id)}&index=${song.charts[0]?.index ?? 5}`,
                            )
                          }
                        >
                          预览
                        </Button>
                        <Button
                          size="small"
                          sx={{ whiteSpace: "nowrap", flexShrink: 0 }}
                          onClick={() =>
                            navigate(
                              `/song?root=${encodeURIComponent(summary.processed_root)}&id=${encodeURIComponent(song.id)}`,
                            )
                          }
                        >
                          详情
                        </Button>
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
                还没有导入歌曲。把包含音频与 maidata 的 zip 拖到上方即可。
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    </Stack>
  );
}
