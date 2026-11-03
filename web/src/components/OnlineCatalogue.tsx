import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  LinearProgress,
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
import { Cancel, CloudDownload, Search } from "@mui/icons-material";
import { api, formatError, postJson } from "../api";
import Jacket from "./Jacket";
import DifficultyChip from "./DifficultyChip";
import { PALETTE } from "../theme";
import type {
  WmcChart,
  WmcImportJob,
  WmcImportResult,
  WmcJobsResponse,
  WmcSearchResponse,
} from "../types";

const LEVEL_NAMES: Record<string, string> = {
  "1": "easy",
  "2": "basic",
  "3": "advanced",
  "4": "expert",
  "5": "master",
  "6": "remaster",
  "7": "original",
};

const PAGE_SIZE = 30;

type Filters = {
  q: string;
  version: string;
  type: string;
  genre: string;
  sort: string;
};

const EMPTY_FILTERS: Filters = { q: "", version: "", type: "", genre: "", sort: "id" };

function SectionTitle({ title, detail }: { title: string; detail?: string }) {
  return (
    <Stack
      direction="row"
      alignItems="baseline"
      justifyContent="space-between"
      spacing={2}
      sx={{ mb: 1.5 }}
    >
      <Typography variant="h6" sx={{ fontSize: 18 }}>
        {title}
      </Typography>
      {detail ? (
        <Typography variant="caption" color="text.secondary">
          {detail}
        </Typography>
      ) : null}
    </Stack>
  );
}

function levelsOf(chart: WmcChart): { index: string; difficulty: string; level: string }[] {
  return Object.entries(chart.levels ?? {})
    .map(([index, value]) => ({
      index,
      difficulty: LEVEL_NAMES[index] ?? "unknown",
      level: String(value?.raw ?? value?.num ?? ""),
    }))
    .filter((item) => item.level)
    .sort((a, b) => Number(a.index) - Number(b.index));
}

function resultStatus(result: WmcImportResult): {
  label: string;
  color: "success" | "warning" | "error";
} {
  if (result.skipped) return { label: "跳过", color: "warning" };
  if (result.ok) return { label: "成功", color: "success" };
  return { label: "失败", color: "error" };
}

export default function OnlineCatalogue({
  onImported,
}: {
  onImported: () => void | Promise<void>;
}) {
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS);
  const [data, setData] = useState<WmcSearchResponse | null>(null);
  const [page, setPage] = useState(1);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Record<string, WmcChart>>({});
  const [align, setAlign] = useState(false);
  const [includeVideo, setIncludeVideo] = useState(false);
  const [force, setForce] = useState(false);
  const [starting, setStarting] = useState(false);
  const [job, setJob] = useState<WmcImportJob | null>(null);
  const [jobError, setJobError] = useState<string | null>(null);

  const runSearch = async (targetPage: number, nextFilters: Filters): Promise<void> => {
    setSearching(true);
    setSearchError(null);
    const params = new URLSearchParams();
    if (nextFilters.q.trim()) params.set("q", nextFilters.q.trim());
    if (nextFilters.version) params.set("version", nextFilters.version);
    if (nextFilters.type) params.set("type", nextFilters.type);
    if (nextFilters.genre) params.set("genre", nextFilters.genre);
    if (nextFilters.sort) params.set("sort", nextFilters.sort);
    params.set("page", String(targetPage));
    params.set("limit", String(PAGE_SIZE));
    try {
      const response = await api<WmcSearchResponse>(`/api/wmc/search?${params.toString()}`);
      setData(response);
      setPage(response.page || targetPage);
    } catch (caught) {
      setSearchError(formatError(caught));
    } finally {
      setSearching(false);
    }
  };

  useEffect(() => {
    void runSearch(1, EMPTY_FILTERS);
    void (async () => {
      try {
        const response = await api<WmcJobsResponse>("/api/wmc/imports");
        const active =
          response.jobs.find((item) => item.status === "running" || item.status === "stopping") ??
          response.jobs[response.jobs.length - 1];
        if (active) setJob(active);
      } catch {
        // The catalogue is still usable without a previous job.
      }
    })();
  }, []);

  useEffect(() => {
    if (!job || (job.status !== "running" && job.status !== "stopping")) return undefined;
    let stopped = false;
    const timer = window.setInterval(() => {
      void (async () => {
        try {
          const next = await api<WmcImportJob>(`/api/wmc/imports/${job.id}`);
          if (stopped) return;
          setJob(next);
          if (next.status === "done" || next.status === "cancelled") {
            window.clearInterval(timer);
            await onImported();
          }
        } catch (caught) {
          if (!stopped) setJobError(formatError(caught));
        }
      })();
    }, 1200);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [job?.id, job?.status]);

  const versions = useMemo(
    () =>
      Object.entries(data?.available?.versions ?? {})
        .map(([id, label]) => ({ id, label }))
        .sort((a, b) => Number(a.id) - Number(b.id)),
    [data],
  );
  const genres = data?.available?.genres ?? [];

  const charts = data?.charts ?? [];
  const pageIds = charts.map((chart) => chart.shortid);
  const selectedOnPage = pageIds.filter((id) => selected[id]).length;
  const allOnPage = pageIds.length > 0 && selectedOnPage === pageIds.length;
  const selectedIds = Object.keys(selected);
  const pages = Math.max(1, data?.pages ?? 1);

  const toggle = (chart: WmcChart, checked: boolean) => {
    setSelected((current) => {
      const next = { ...current };
      if (checked) next[chart.shortid] = chart;
      else delete next[chart.shortid];
      return next;
    });
  };

  const togglePage = (checked: boolean) => {
    setSelected((current) => {
      const next = { ...current };
      for (const chart of charts) {
        if (checked) next[chart.shortid] = chart;
        else delete next[chart.shortid];
      }
      return next;
    });
  };

  const startImport = async (): Promise<void> => {
    if (!selectedIds.length) return;
    setStarting(true);
    setJobError(null);
    try {
      const created = await postJson<WmcImportJob>("/api/wmc/import", {
        ids: selectedIds,
        align,
        include_video: includeVideo,
        force,
      });
      setJob(created);
      setSelected({});
    } catch (caught) {
      setJobError(formatError(caught));
    } finally {
      setStarting(false);
    }
  };

  const cancelImport = async (): Promise<void> => {
    if (!job) return;
    setJobError(null);
    try {
      const next = await postJson<WmcImportJob>(`/api/wmc/imports/${job.id}/cancel`, {});
      setJob(next);
    } catch (caught) {
      setJobError(formatError(caught));
    }
  };

  const jobRunning = job?.status === "running" || job?.status === "stopping";
  const jobResults = job?.results ?? [];
  const succeeded = jobResults.filter((item) => item.ok && !item.skipped).length;
  const skipped = jobResults.filter((item) => item.skipped).length;
  const failed = jobResults.filter((item) => !item.ok).length;

  return (
    <Stack spacing={2}>
      <Box
        sx={{
          border: 1,
          borderColor: "divider",
          borderRadius: "4px",
          bgcolor: "background.paper",
          p: { xs: 2, md: 2.5 },
        }}
      >
        <SectionTitle
          title="在线谱库"
          detail={data?.source ?? "AWMC-MaiArchive · download.wmc.pub"}
        />

        <Box
          component="form"
          onSubmit={(event) => {
            event.preventDefault();
            void runSearch(1, filters);
          }}
        >
          <Stack direction={{ xs: "column", md: "row" }} spacing={1.25} flexWrap="wrap" useFlexGap>
            <TextField
              size="small"
              label="曲名 / 别名 / ID"
              value={filters.q}
              onChange={(event) => setFilters({ ...filters, q: event.target.value })}
              sx={{ minWidth: 220, flexGrow: 2 }}
            />
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel id="wmc-version">版本</InputLabel>
              <Select
                labelId="wmc-version"
                label="版本"
                value={filters.version}
                onChange={(event) => setFilters({ ...filters, version: event.target.value })}
              >
                <MenuItem value="">全部版本</MenuItem>
                {versions.map((version) => (
                  <MenuItem key={version.id} value={version.label}>
                    {version.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 110 }}>
              <InputLabel id="wmc-type">类型</InputLabel>
              <Select
                labelId="wmc-type"
                label="类型"
                value={filters.type}
                onChange={(event) => setFilters({ ...filters, type: event.target.value })}
              >
                <MenuItem value="">全部</MenuItem>
                <MenuItem value="st">ST</MenuItem>
                <MenuItem value="dx">DX</MenuItem>
                <MenuItem value="utg">宴</MenuItem>
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel id="wmc-genre">分类</InputLabel>
              <Select
                labelId="wmc-genre"
                label="分类"
                value={filters.genre}
                onChange={(event) => setFilters({ ...filters, genre: event.target.value })}
              >
                <MenuItem value="">全部分类</MenuItem>
                {genres.map((genre) => (
                  <MenuItem key={genre} value={genre}>
                    {genre}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl size="small" sx={{ minWidth: 110 }}>
              <InputLabel id="wmc-sort">排序</InputLabel>
              <Select
                labelId="wmc-sort"
                label="排序"
                value={filters.sort}
                onChange={(event) => setFilters({ ...filters, sort: event.target.value })}
              >
                <MenuItem value="id">默认</MenuItem>
                <MenuItem value="popular">热度</MenuItem>
                <MenuItem value="title">标题</MenuItem>
                <MenuItem value="version">版本</MenuItem>
              </Select>
            </FormControl>
            <Button
              type="submit"
              variant="contained"
              startIcon={searching ? <CircularProgress size={16} color="inherit" /> : <Search />}
              disabled={searching}
            >
              搜索
            </Button>
          </Stack>
        </Box>

        {searchError ? (
          <Alert severity="error" sx={{ mt: 2 }}>
            {searchError}
          </Alert>
        ) : null}

        <Stack
          direction={{ xs: "column", sm: "row" }}
          alignItems={{ sm: "center" }}
          spacing={1.5}
          sx={{ mt: 2, mb: 1 }}
        >
          <Typography variant="body2" color="text.secondary">
            {data
              ? `匹配 ${data.matched} 首 / 共 ${data.total} 首，第 ${page} / ${pages} 页`
              : "输入关键词或直接翻页浏览谱库"}
          </Typography>
          <Box sx={{ flexGrow: 1 }} />
          <Button
            size="small"
            disabled={searching || page <= 1}
            onClick={() => void runSearch(page - 1, filters)}
          >
            上一页
          </Button>
          <Button
            size="small"
            disabled={searching || page >= pages}
            onClick={() => void runSearch(page + 1, filters)}
          >
            下一页
          </Button>
        </Stack>

        <Box sx={{ overflowX: "auto", borderTop: 1, borderColor: "divider" }}>
          {searching && !data ? (
            <Box sx={{ py: 6, textAlign: "center" }}>
              <CircularProgress size={26} />
            </Box>
          ) : (
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    <Checkbox
                      checked={allOnPage}
                      indeterminate={selectedOnPage > 0 && !allOnPage}
                      onChange={(event) => togglePage(event.target.checked)}
                      inputProps={{ "aria-label": "选择本页全部歌曲" }}
                    />
                  </TableCell>
                  <TableCell>歌曲</TableCell>
                  <TableCell>版本 / 类型</TableCell>
                  <TableCell align="right">BPM</TableCell>
                  <TableCell>难度</TableCell>
                  <TableCell align="right">下载</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {charts.map((chart) => (
                  <TableRow key={chart.shortid} hover selected={Boolean(selected[chart.shortid])}>
                    <TableCell padding="checkbox">
                      <Checkbox
                        checked={Boolean(selected[chart.shortid])}
                        onChange={(event) => toggle(chart, event.target.checked)}
                        inputProps={{ "aria-label": `选择 ${chart.title}` }}
                      />
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1.25} alignItems="center">
                        <Jacket
                          title={chart.title}
                          id={chart.shortid}
                          src={chart.cover_url}
                          size={36}
                          radius={2}
                        />
                        <Box sx={{ minWidth: 0 }}>
                          <Typography variant="body2" sx={{ fontWeight: 600 }} noWrap>
                            {chart.displayTitle || chart.title}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" noWrap>
                            {chart.artist || "未知作者"} / {chart.shortid}
                          </Typography>
                        </Box>
                      </Stack>
                    </TableCell>
                    <TableCell sx={{ whiteSpace: "nowrap" }}>
                      <Stack direction="row" spacing={0.6} alignItems="center" useFlexGap>
                        <Chip
                          size="small"
                          variant="outlined"
                          label={chart.version || "-"}
                          sx={{ height: 22 }}
                        />
                        <Chip
                          size="small"
                          label={chart.type || "-"}
                          sx={{
                            height: 22,
                            bgcolor: chart.type === "DX" ? PALETTE.aqua : undefined,
                            color: chart.type === "DX" ? "#04141c" : undefined,
                          }}
                        />
                      </Stack>
                    </TableCell>
                    <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                      {chart.bpm ?? "-"}
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                        {levelsOf(chart).map((item) => (
                          <DifficultyChip
                            key={item.index}
                            difficulty={item.difficulty}
                            level={item.level}
                          />
                        ))}
                      </Stack>
                    </TableCell>
                    <TableCell align="right" sx={{ fontVariantNumeric: "tabular-nums" }}>
                      {chart.downloads ?? "-"}
                    </TableCell>
                  </TableRow>
                ))}
                {!charts.length && !searching ? (
                  <TableRow>
                    <TableCell colSpan={6}>
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ py: 4, textAlign: "center" }}
                      >
                        没有匹配的曲目。
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : null}
              </TableBody>
            </Table>
          )}
        </Box>

        <Stack
          direction={{ xs: "column", md: "row" }}
          spacing={2}
          alignItems={{ md: "center" }}
          sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: "divider" }}
        >
          <FormControlLabel
            control={
              <Switch checked={align} onChange={(event) => setAlign(event.target.checked)} />
            }
            label="导入后对齐 BPM"
          />
          <FormControlLabel
            control={
              <Switch
                checked={includeVideo}
                onChange={(event) => setIncludeVideo(event.target.checked)}
              />
            }
            label="下载背景视频"
          />
          <FormControlLabel
            control={
              <Switch checked={force} onChange={(event) => setForce(event.target.checked)} />
            }
            label="强制覆盖已有"
          />
          <Box sx={{ flexGrow: 1 }} />
          <Typography variant="caption" color="text.secondary">
            已选 {selectedIds.length} 首（可跨页，单次最多 100 首）
          </Typography>
          <Button
            variant="contained"
            startIcon={starting ? <CircularProgress size={16} color="inherit" /> : <CloudDownload />}
            disabled={starting || jobRunning || selectedIds.length === 0}
            onClick={() => void startImport()}
          >
            {starting ? "提交中..." : `导入 ${selectedIds.length || ""} 首`}
          </Button>
        </Stack>

        {jobError ? (
          <Alert severity="error" sx={{ mt: 2 }}>
            {jobError}
          </Alert>
        ) : null}
      </Box>

      {job ? (
        <Box
          sx={{
            border: 1,
            borderColor: "divider",
            borderRadius: "4px",
            bgcolor: "background.paper",
            p: { xs: 2, md: 2.5 },
          }}
        >
          <SectionTitle
            title="在线导入进度"
            detail={
              job.status === "running"
                ? "正在下载并入库"
                : job.status === "stopping"
                  ? "正在停止..."
                  : job.status === "cancelled"
                    ? "已取消"
                    : "已完成"
            }
          />
          <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1 }}>
            <LinearProgress
              variant="determinate"
              value={job.total ? Math.min(100, (job.completed / job.total) * 100) : 0}
              sx={{ flexGrow: 1 }}
            />
            <Typography variant="body2" sx={{ fontVariantNumeric: "tabular-nums" }}>
              {job.completed} / {job.total}
            </Typography>
            {jobRunning ? (
              <Button
                size="small"
                color="error"
                variant="outlined"
                startIcon={<Cancel />}
                disabled={job.status === "stopping"}
                onClick={() => void cancelImport()}
              >
                停止
              </Button>
            ) : null}
          </Stack>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1.5 }}>
            {job.current ? `当前：${job.current}` : "任务已结束"}
            {jobResults.length
              ? ` · 成功 ${succeeded} · 跳过 ${skipped} · 失败 ${failed}`
              : ""}
          </Typography>

          {jobResults.length ? (
            <Box sx={{ overflowX: "auto", borderTop: 1, borderColor: "divider" }}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>状态</TableCell>
                    <TableCell>歌曲</TableCell>
                    <TableCell align="right">谱面数</TableCell>
                    <TableCell>信息</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {jobResults.map((result, index) => {
                    const status = resultStatus(result);
                    const messages = [...result.errors, ...result.warnings].join("; ");
                    return (
                      <TableRow key={`${result.shortid}-${index}`}>
                        <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>
                          {result.shortid}
                        </TableCell>
                        <TableCell>
                          <Chip size="small" color={status.color} label={status.label} />
                        </TableCell>
                        <TableCell>
                          {result.title ?? result.song_id ?? "-"}
                          {result.song_id && result.title
                            ? ` (${result.song_id})`
                            : ""}
                        </TableCell>
                        <TableCell align="right">{result.chart_count ?? "-"}</TableCell>
                        <TableCell
                          sx={{ fontSize: 12, color: result.ok ? "text.secondary" : "error.main" }}
                        >
                          {messages || "-"}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </Box>
          ) : null}
        </Box>
      ) : null}
    </Stack>
  );
}
