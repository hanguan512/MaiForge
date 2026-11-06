import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { Cancel, CloudDownload, Search } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
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
    <Stack spacing={2.5}>
      <Box
        component="form"
        onSubmit={(event) => {
          event.preventDefault();
          void runSearch(1, filters);
        }}
      >
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr 1fr",
              md: "minmax(200px, 2.2fr) 150px 100px 170px 110px auto",
            },
            gap: 1,
            alignItems: "center",
          }}
        >
          <TextField
            label="曲名 / 别名 / ID"
            value={filters.q}
            onChange={(event) => setFilters({ ...filters, q: event.target.value })}
            sx={{ gridColumn: { xs: "1 / -1", md: "auto" } }}
          />
          <FormControl size="small" fullWidth sx={{ minWidth: 0 }}>
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
          <FormControl size="small" fullWidth sx={{ minWidth: 0 }}>
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
          <FormControl size="small" fullWidth sx={{ minWidth: 0 }}>
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
          <FormControl size="small" fullWidth sx={{ minWidth: 0 }}>
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
            startIcon={searching ? <CircularProgress size={15} color="inherit" /> : <Search />}
            disabled={searching}
            sx={{ height: 40, gridColumn: { xs: "1 / -1", md: "auto" } }}
          >
            搜索
          </Button>
        </Box>
      </Box>

      {searchError ? <Alert severity="error">{searchError}</Alert> : null}

      <Stack
        direction={{ xs: "column", sm: "row" }}
        alignItems={{ sm: "center" }}
        spacing={1.5}
        sx={{ mt: 0.5 }}
      >
        <Typography variant="caption" color="text.secondary">
          {data
            ? `匹配 ${data.matched} 首 / 共 ${data.total} 首 · 第 ${page} / ${pages} 页`
            : "输入关键词或直接翻页浏览谱库"}
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <Stack direction="row" spacing={0.5} alignItems="center">
          <Checkbox
            size="small"
            checked={allOnPage}
            indeterminate={selectedOnPage > 0 && !allOnPage}
            onChange={(event) => togglePage(event.target.checked)}
            inputProps={{ "aria-label": "选择本页全部歌曲" }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mr: 1 }}>
            全选本页
          </Typography>
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
      </Stack>

      <Stack
        direction={{ xs: "column", md: "row" }}
        alignItems={{ md: "center" }}
        spacing={{ xs: 1, md: 1.5 }}
        flexWrap="wrap"
        useFlexGap
        sx={{ borderTop: 1, borderColor: "divider", py: 1 }}
      >
        <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
          <FormControlLabel
            control={<Switch size="small" checked={align} onChange={(event) => setAlign(event.target.checked)} />}
            label={<Typography variant="caption">对齐 BPM</Typography>}
          />
          <FormControlLabel
            control={
              <Switch
                size="small"
                checked={includeVideo}
                onChange={(event) => setIncludeVideo(event.target.checked)}
              />
            }
            label={<Typography variant="caption">背景视频</Typography>}
          />
          <FormControlLabel
            control={
              <Switch size="small" checked={force} onChange={(event) => setForce(event.target.checked)} />
            }
            label={<Typography variant="caption">覆盖已有</Typography>}
          />
        </Stack>
        <Box sx={{ flexGrow: 1 }} />
        <Typography variant="caption" color="text.secondary">
          已选 {selectedIds.length} 首（可跨页，单次最多 100 首）
        </Typography>
        <Button
          variant="contained"
          startIcon={starting ? <CircularProgress size={15} color="inherit" /> : <CloudDownload />}
          disabled={starting || jobRunning || selectedIds.length === 0}
          onClick={() => void startImport()}
        >
          {starting ? "提交中…" : selectedIds.length ? `导入 ${selectedIds.length} 首` : "导入所选"}
        </Button>
      </Stack>

      <Box sx={{ borderTop: 1, borderColor: "divider" }}>
        <Box
          sx={{
            display: { xs: "none", md: "grid" },
            gridTemplateColumns: "24px minmax(0, 1fr) 100px 64px minmax(210px, auto)",
            columnGap: 1.5,
            alignItems: "center",
            py: 0.75,
            borderBottom: 1,
            borderColor: "divider",
            color: "text.disabled",
          }}
        >
          <span />
          <Typography variant="caption">歌曲</Typography>
          <Typography variant="caption">版本 / 类型</Typography>
          <Typography variant="caption" sx={{ textAlign: "right" }}>
            BPM
          </Typography>
          <Typography variant="caption">难度</Typography>
        </Box>

        {searching && !data ? (
          <Box sx={{ py: 8, textAlign: "center" }}>
            <CircularProgress size={24} />
          </Box>
        ) : charts.length ? (
          charts.map((chart) => {
            const active = Boolean(selected[chart.shortid]);
            return (
              <Box
                key={chart.shortid}
                role="checkbox"
                aria-checked={active}
                aria-label={`选择 ${chart.displayTitle || chart.title}`}
                tabIndex={0}
                onClick={() => toggle(chart, !active)}
                onKeyDown={(event) => {
                  if (event.key === " " || event.key === "Enter") {
                    event.preventDefault();
                    toggle(chart, !active);
                  }
                }}
                sx={{
                  position: "relative",
                  display: "grid",
                  gridTemplateColumns: {
                    xs: "24px minmax(0, 1fr)",
                    sm: "24px minmax(0, 1fr) minmax(200px, auto)",
                    md: "24px minmax(0, 1fr) 100px 64px minmax(210px, auto)",
                  },
                  columnGap: 1.5,
                  alignItems: "center",
                  py: 0.9,
                  borderBottom: 1,
                  borderColor: "divider",
                  cursor: "pointer",
                  bgcolor: active ? alpha(PALETTE.pink, 0.055) : "transparent",
                  "&:hover": { bgcolor: active ? alpha(PALETTE.pink, 0.08) : alpha(PALETTE.text, 0.028) },
                  "&:focus-visible": {
                    outline: `2px solid ${PALETTE.aqua}`,
                    outlineOffset: -2,
                  },
                  "&::before": {
                    content: '""',
                    position: "absolute",
                    left: 0,
                    top: 6,
                    bottom: 6,
                    width: 2,
                    bgcolor: "primary.main",
                    opacity: active ? 1 : 0,
                  },
                }}
              >
                <Checkbox
                  size="small"
                  checked={active}
                  onChange={(event) => toggle(chart, event.target.checked)}
                  onClick={(event) => event.stopPropagation()}
                  inputProps={{ "aria-label": `选择 ${chart.title}` }}
                  sx={{ p: 0.25 }}
                />
                <Stack direction="row" spacing={1.25} alignItems="center" sx={{ minWidth: 0 }}>
                  <Jacket
                    title={chart.title}
                    id={chart.shortid}
                    src={chart.cover_url}
                    size={38}
                    radius={1}
                  />
                  <Box sx={{ minWidth: 0 }}>
                    <Typography
                      sx={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.35 }}
                      noWrap
                      title={chart.displayTitle || chart.title}
                    >
                      {chart.displayTitle || chart.title}
                    </Typography>
                    <Typography variant="caption" color="text.secondary" noWrap sx={{ display: "block" }}>
                      {chart.artist || "未知作者"}
                      <Box component="span" sx={{ color: "text.disabled" }}>
                        {"  /  "}
                        {chart.shortid}
                      </Box>
                    </Typography>
                  </Box>
                </Stack>
                <Stack
                  direction="row"
                  spacing={0.5}
                  alignItems="center"
                  sx={{ display: { xs: "none", md: "flex" } }}
                >
                  <Typography
                    variant="caption"
                    sx={{ color: "text.secondary", minWidth: 28 }}
                  >
                    {chart.version || "-"}
                  </Typography>
                  <Box
                    component="span"
                    sx={{
                      fontSize: 10,
                      fontWeight: 700,
                      px: 0.5,
                      py: 0.1,
                      border: 1,
                      borderColor: chart.type === "DX" ? PALETTE.aqua : "divider",
                      color: chart.type === "DX" ? PALETTE.aqua : "text.secondary",
                    }}
                  >
                    {chart.type || "-"}
                  </Box>
                </Stack>
                <Typography
                  variant="caption"
                  sx={{
                    display: { xs: "none", md: "block" },
                    textAlign: "right",
                    fontFamily: "inherit",
                    fontVariantNumeric: "tabular-nums",
                    color: "text.secondary",
                  }}
                >
                  {chart.bpm ?? "-"}
                </Typography>
                <Stack
                  direction="row"
                  spacing={0.5}
                  flexWrap="wrap"
                  useFlexGap
                  sx={{ display: { xs: "none", sm: "flex" } }}
                >
                  {levelsOf(chart).map((item) => (
                    <DifficultyChip
                      key={item.index}
                      difficulty={item.difficulty}
                      level={item.level}
                    />
                  ))}
                </Stack>
              </Box>
            );
          })
        ) : !searching ? (
          <Box sx={{ py: 7, textAlign: "center" }}>
            <Typography variant="body2" color="text.secondary">
              没有匹配的曲目。
            </Typography>
          </Box>
        ) : null}
      </Box>

      {jobError ? <Alert severity="error">{jobError}</Alert> : null}

      {job ? (
        <Box sx={{ pt: 2, borderTop: 1, borderColor: "divider" }}>
          <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 1 }}>
            <Typography sx={{ fontSize: 13, fontWeight: 600 }}>
              {job.status === "running"
                ? "正在下载并入库"
                : job.status === "stopping"
                  ? "正在停止…"
                  : job.status === "cancelled"
                    ? "已取消"
                    : "导入完成"}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={job.total ? Math.min(100, (job.completed / job.total) * 100) : 0}
              sx={{ flexGrow: 1 }}
            />
            <Typography
              variant="caption"
              sx={{ fontVariantNumeric: "tabular-nums", minWidth: 52 }}
            >
              {job.completed} / {job.total}
            </Typography>
            {jobRunning ? (
              <Button
                size="small"
                color="error"
                variant="outlined"
                startIcon={<Cancel sx={{ fontSize: 16 }} />}
                disabled={job.status === "stopping"}
                onClick={() => void cancelImport()}
              >
                停止
              </Button>
            ) : null}
          </Stack>
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", mb: 1 }}>
            {job.current ? `当前：${job.current}` : "任务已结束"}
            {jobResults.length ? ` · 成功 ${succeeded} · 跳过 ${skipped} · 失败 ${failed}` : ""}
          </Typography>

          {jobResults.length ? (
            <Box sx={{ borderTop: 1, borderColor: "divider" }}>
              {jobResults.map((result, index) => {
                const status = resultStatus(result);
                const messages = [...(result.errors ?? []), ...(result.warnings ?? [])].join(
                  "; ",
                );
                return (
                  <Box
                    key={`${result.shortid}-${index}`}
                    sx={{
                      display: "grid",
                      gridTemplateColumns: "64px minmax(0, 1fr) auto",
                      gap: 1.5,
                      alignItems: "baseline",
                      py: 0.75,
                      borderBottom: 1,
                      borderColor: "divider",
                    }}
                  >
                    <Typography
                      variant="caption"
                      sx={{ color: "text.disabled", fontVariantNumeric: "tabular-nums" }}
                    >
                      {result.shortid}
                    </Typography>
                    <Box sx={{ minWidth: 0 }}>
                      <Typography variant="body2" noWrap>
                        {result.title ?? result.song_id ?? "-"}
                      </Typography>
                      {messages ? (
                        <Typography
                          variant="caption"
                          sx={{
                            display: "block",
                            color: result.ok ? "text.secondary" : "error.main",
                            wordBreak: "break-word",
                          }}
                        >
                          {messages}
                        </Typography>
                      ) : null}
                    </Box>
                    <Typography
                      variant="caption"
                      sx={{
                        color:
                          status.color === "success"
                            ? PALETTE.green
                            : status.color === "warning"
                              ? PALETTE.gold
                              : PALETTE.red,
                        fontWeight: 700,
                      }}
                    >
                      {status.label}
                      {result.chart_count ? ` ${result.chart_count}` : ""}
                    </Typography>
                  </Box>
                );
              })}
            </Box>
          ) : null}
        </Box>
      ) : null}
    </Stack>
  );
}
