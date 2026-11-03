import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  Box,
  Button,
  Dialog,
  DialogContent,
  Divider,
  IconButton,
  MenuItem,
  Select,
  Slider,
  Stack,
  TextField,
  Tooltip,
  Typography,
  useMediaQuery,
} from "@mui/material";
import {
  Close,
  FolderOpen,
  Refresh,
  Search,
  Settings,
} from "@mui/icons-material";
import { useNavigate, useSearchParams } from "react-router-dom";
import { api, formatError } from "../api";
import CoverFlow from "../components/CoverFlow";
import { DISPLAY_FONT, NUMERIC_FONT } from "../theme";
import type { Overview, SongSummary } from "../types";

const ALL_ROOTS = "__all__";

const DIM_KEY = "mainote.bgDim";

type RootOption = { value: string; label: string; exists: boolean; song_count: number };

function mod(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function readNumber(key: string, fallback: number, minimum: number, maximum: number): number {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    const value = Number(raw);
    if (!Number.isFinite(value)) return fallback;
    return Math.min(maximum, Math.max(minimum, value));
  } catch {
    return fallback;
  }
}

function writeNumber(key: string, value: number): void {
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    // localStorage may be unavailable; the in-memory setting still applies
  }
}

const glassButton = {
  width: 42,
  height: 42,
  border: "1px solid rgba(255,255,255,0.55)",
  bgcolor: "rgba(255,255,255,0.26)",
  color: "#33204d",
  backdropFilter: "blur(10px)",
  transition: "background-color 140ms ease, transform 140ms ease",
  "&:hover": { bgcolor: "rgba(255,255,255,0.46)" },
  "&:active": { transform: "scale(0.95)" },
  "&:focus-visible": { outline: "2px solid #3fd4e6", outlineOffset: 2 },
};

const lightSlider = {
  color: "#a06cff",
  height: 4,
  py: 1.5,
  "& .MuiSlider-rail": { opacity: 0.22 },
  "& .MuiSlider-track": { border: "none" },
  "& .MuiSlider-thumb": {
    width: 16,
    height: 16,
    bgcolor: "#ffffff",
    border: "2px solid #a06cff",
    boxShadow: "none",
    "&:hover, &.Mui-focusVisible": {
      boxShadow: "0 0 0 6px rgba(160,108,255,0.14)",
    },
  },
};

function SettingRow({
  label,
  value,
  children,
}: {
  label: string;
  value: string;
  children: ReactNode;
}) {
  return (
    <Box sx={{ py: 1.05 }}>
      <Stack direction="row" alignItems="baseline" justifyContent="space-between" spacing={2}>
        <Typography
          sx={{
            fontSize: 11.5,
            fontWeight: 700,
            letterSpacing: "0.1em",
            color: "#6d5a85",
          }}
        >
          {label}
        </Typography>
        <Typography
          sx={{
            fontFamily: NUMERIC_FONT,
            fontSize: 13,
            fontWeight: 800,
            color: "#2a1a3e",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {value}
        </Typography>
      </Stack>
      {children}
    </Box>
  );
}

function Keycap({ children }: { children: ReactNode }) {
  return (
    <Box
      component="span"
      sx={{
        display: "inline-grid",
        placeItems: "center",
        width: 20,
        height: 20,
        borderRadius: "3px",
        border: "1px solid rgba(60,32,90,0.22)",
        bgcolor: "rgba(255,255,255,0.22)",
        color: "rgba(45,25,70,0.72)",
        fontFamily: NUMERIC_FONT,
        fontSize: 10.5,
        fontWeight: 700,
      }}
    >
      {children}
    </Box>
  );
}

export default function BrowsePage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const reducedMotion = useMediaQuery("(prefers-reduced-motion: reduce)");

  const [roots, setRoots] = useState<RootOption[]>([]);
  const [root, setRoot] = useState("");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [songs, setSongs] = useState<SongSummary[]>([]);
  const [cursor, setCursor] = useState(0);
  const [chartIndex, setChartIndex] = useState<number | null>(null);
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [loadingRoots, setLoadingRoots] = useState(true);
  const [loadingSongs, setLoadingSongs] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const [bgDim, setBgDim] = useState(() => readNumber(DIM_KEY, 0.2, 0, 1));

  useEffect(() => writeNumber(DIM_KEY, bgDim), [bgDim]);

  useEffect(() => {
    let alive = true;
    setLoadingRoots(true);
    (async () => {
      try {
        const data = await api<Overview>("/api/overview");
        if (!alive) return;
        setOverview(data);
        const available = data.roots.filter((item) => item.exists);
        setRoots(data.roots);
        if (!data.roots.length) {
          setLoadingSongs(false);
        }
        setRoot((current) => {
          if (current === ALL_ROOTS && available.length) return current;
          if (current && data.roots.some((item) => item.value === current)) return current;
          const requested = params.get("root");
          if (requested === ALL_ROOTS && available.length) return requested;
          if (requested && data.roots.some((item) => item.value === requested)) return requested;
          if (available.length > 1) return ALL_ROOTS;
          return available[0]?.value ?? data.roots[0]?.value ?? "";
        });
      } catch (caught) {
        if (alive) {
          setError(formatError(caught));
          setLoadingSongs(false);
        }
      } finally {
        if (alive) setLoadingRoots(false);
      }
    })();
    return () => {
      alive = false;
    };
    // `params` is only read to restore the last selection; it must not re-trigger
    // the overview request when the URL is updated after a carousel move.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  useEffect(() => {
    if (!root || loadingRoots) return;
    let alive = true;
    setLoadingSongs(true);
    setError(null);
    (async () => {
      try {
        const targets =
          root === ALL_ROOTS
            ? roots.filter((item) => item.exists)
            : roots.filter((item) => item.value === root);
        const requests = targets.length
          ? targets.map((item) =>
              api<{ songs: SongSummary[] }>(`/api/songs?root=${encodeURIComponent(item.value)}`),
            )
          : [api<{ songs: SongSummary[] }>("/api/songs")];
        const results = await Promise.all(requests);
        if (!alive) return;
        const seen = new Set<string>();
        const merged = results
          .flatMap((result) => result.songs)
          .filter((song) => {
            const key = `${song.root}:${song.id}`;
            if (seen.has(key)) return false;
            seen.add(key);
            return true;
          })
          .sort(
            (left, right) =>
              left.title.localeCompare(right.title, "zh-Hans-CN") ||
              left.root.localeCompare(right.root),
          );
        setSongs(merged);
        const wanted = params.get("song");
        const restored = wanted ? merged.findIndex((song) => song.id === wanted) : -1;
        setCursor(restored >= 0 ? restored : 0);
      } catch (caught) {
        if (alive) {
          setSongs([]);
          setError(formatError(caught));
        }
      } finally {
        if (alive) setLoadingSongs(false);
      }
    })();
    return () => {
      alive = false;
    };
    // same note as above: restoring `song` from the URL is a one-shot read
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [root, roots, loadingRoots, refreshKey]);

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return songs;
    return songs.filter((song) =>
      `${song.title} ${song.artist} ${song.id}`.toLowerCase().includes(needle),
    );
  }, [songs, query]);

  useEffect(() => {
    setCursor(0);
  }, [query]);

  const count = filtered.length;
  const activeIndex = count ? mod(cursor, count) : 0;
  const activeSong = filtered[activeIndex] ?? null;

  useEffect(() => {
    setChartIndex(activeSong?.charts.at(-1)?.index ?? null);
    // reset when the selected song changes, not when its chart list is rebuilt
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSong?.root, activeSong?.id]);

  useEffect(() => {
    if (!activeSong || !root) return;
    setParams(
      (previous) => {
        const next = new URLSearchParams(previous);
        next.set("root", root);
        next.set("song", activeSong.id);
        return next;
      },
      { replace: true },
    );
  }, [activeSong?.root, activeSong?.id, root, setParams]);

  const chooseRoot = useCallback(
    (value: string) => {
      setRoot(value);
      setCursor(0);
      setQuery("");
      setSearchOpen(false);
      setParams(
        (previous) => {
          const next = new URLSearchParams(previous);
          next.set("root", value);
          next.delete("song");
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );

  const move = useCallback(
    (delta: number) => {
      if (!count) return;
      setCursor((current) => current + delta);
    },
    [count],
  );

  const view = useCallback(() => {
    if (!activeSong) return;
    const chart =
      activeSong.charts.find((item) => item.index === chartIndex) ??
      activeSong.charts.at(-1);
    if (!chart) return;
    navigate(
      `/viewer?root=${encodeURIComponent(activeSong.root)}&id=${encodeURIComponent(
        activeSong.id,
      )}&index=${chart.index}`,
    );
  }, [activeSong, chartIndex, navigate]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (settingsOpen) return;
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }
      if (event.key === "a" || event.key === "A" || event.key === "ArrowLeft") {
        event.preventDefault();
        move(-1);
      } else if (event.key === "d" || event.key === "D" || event.key === "ArrowRight") {
        event.preventDefault();
        move(1);
      } else if (event.key === "Enter" && !searchOpen) {
        event.preventDefault();
        view();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move, view, settingsOpen, searchOpen]);

  const rootOptions = useMemo(() => {
    const available = roots.filter((item) => item.exists);
    if (available.length > 1) {
      return [
        {
          value: ALL_ROOTS,
          label: "全部曲库",
          exists: true,
          song_count: available.reduce((sum, item) => sum + item.song_count, 0),
        },
        ...roots,
      ];
    }
    return roots;
  }, [roots]);

  const currentRoot = rootOptions.find((item) => item.value === root);
  const chartTotal = filtered.reduce((sum, song) => sum + song.chart_count, 0);
  const busy = loadingRoots || (loadingSongs && Boolean(root));

  return (
    <Box
      sx={{
        position: "fixed",
        left: { xs: 0, md: 232 },
        right: 0,
        top: { xs: "50px", md: 0 },
        bottom: 0,
        overflow: "hidden",
        zIndex: 1,
        bgcolor: "#ded0ff",
        background:
          "linear-gradient(38deg, #ffbafd 0%, #f3c2ff 16%, #ded0ff 48%, #cde1ff 72%, #bde7ff 100%)",
        "@keyframes mainote-spin": {
          from: { transform: "rotate(0deg)" },
          to: { transform: "rotate(360deg)" },
        },
      }}
    >
      <Box
        aria-hidden="true"
        sx={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "radial-gradient(105% 88% at 50% 42%, rgba(255,255,255,0.3) 0%, rgba(255,255,255,0.08) 38%, transparent 66%)",
        }}
      />

      <Typography
        sx={{
          position: "absolute",
          top: { xs: 12, sm: 18 },
          left: "50%",
          transform: "translateX(-50%)",
          zIndex: 220,
          fontFamily: DISPLAY_FONT,
          fontSize: { xs: 24, sm: 31 },
          lineHeight: 1,
          letterSpacing: "0.07em",
          color: "rgba(255,255,255,0.94)",
          textShadow: "0 2px 22px rgba(60,22,96,0.28)",
          pointerEvents: "none",
          whiteSpace: "nowrap",
        }}
      >
        MaiForge
      </Typography>

      <Stack
        direction="row"
        alignItems="center"
        justifyContent="space-between"
        sx={{
          position: "absolute",
          top: { xs: 8, sm: 14 },
          left: 0,
          right: 0,
          px: { xs: 1.5, sm: 3 },
          zIndex: 230,
          pointerEvents: "none",
        }}
      >
        <Box sx={{ pointerEvents: "auto" }}>
          <Tooltip title="切换数据目录">
            <Button
              onClick={() => setSettingsOpen(true)}
              startIcon={<FolderOpen sx={{ fontSize: 18 }} />}
              sx={{
                display: { xs: "none", sm: "inline-flex" },
                height: 42,
                px: 1.75,
                borderRadius: "999px",
                border: "1px solid rgba(255,255,255,0.55)",
                bgcolor: "rgba(255,255,255,0.26)",
                color: "#33204d",
                backdropFilter: "blur(10px)",
                fontSize: 13,
                fontWeight: 600,
                "&:hover": { bgcolor: "rgba(255,255,255,0.46)" },
              }}
            >
              {currentRoot?.label ?? "曲库"}
              <Box
                component="span"
                sx={{ ml: 1, fontFamily: NUMERIC_FONT, fontSize: 11.5, opacity: 0.6 }}
              >
                {songs.length || currentRoot?.song_count || 0} 首
              </Box>
            </Button>
          </Tooltip>
        </Box>

        <Stack direction="row" spacing={1} sx={{ ml: "auto", pointerEvents: "auto" }}>
          <Tooltip title="搜索">
            <IconButton
              aria-label="搜索曲库"
              aria-expanded={searchOpen}
              onClick={() => setSearchOpen((value) => !value)}
              sx={glassButton}
            >
              <Search sx={{ fontSize: 20 }} />
            </IconButton>
          </Tooltip>
          <Tooltip title="设置">
            <IconButton
              aria-label="打开设置"
              onClick={() => setSettingsOpen(true)}
              sx={glassButton}
            >
              <Settings sx={{ fontSize: 20 }} />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      {searchOpen ? (
        <Box
          sx={{
            position: "absolute",
            top: { xs: 52, sm: 68 },
            left: "50%",
            transform: "translateX(-50%)",
            width: "min(440px, calc(100% - 28px))",
            zIndex: 240,
          }}
        >
          <TextField
            autoFocus
            fullWidth
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                setSearchOpen(false);
                setQuery("");
              }
            }}
            placeholder="搜索曲名、作者或 ID"
            slotProps={{ htmlInput: { "aria-label": "搜索曲库" } }}
            InputProps={{
              startAdornment: (
                <Box sx={{ display: "flex", mr: 1, color: "#6d5a85" }}>
                  <Search sx={{ fontSize: 19 }} />
                </Box>
              ),
              endAdornment: (
                <IconButton
                  size="small"
                  aria-label="关闭搜索"
                  onClick={() => {
                    setSearchOpen(false);
                    setQuery("");
                  }}
                  sx={{ color: "#6d5a85" }}
                >
                  <Close sx={{ fontSize: 18 }} />
                </IconButton>
              ),
            }}
            sx={{
              "& .MuiOutlinedInput-root": {
                height: 44,
                borderRadius: "999px",
                bgcolor: "rgba(255,255,255,0.9)",
                color: "#2a1a3e",
                boxShadow: "0 16px 38px rgba(55,24,90,0.22)",
                backdropFilter: "blur(12px)",
                "& fieldset": { borderColor: "rgba(255,255,255,0.7)" },
                "&:hover fieldset": { borderColor: "rgba(255,255,255,0.9)" },
                "&.Mui-focused fieldset": { borderColor: "rgba(160,108,255,0.7)" },
              },
              "& input::placeholder": { color: "rgba(60,40,85,0.5)", opacity: 1 },
            }}
          />
        </Box>
      ) : null}

      {!busy && !error && count > 0 ? (
        <CoverFlow
          songs={filtered}
          cursor={cursor}
          chartIndex={chartIndex}
          dim={bgDim}
          reducedMotion={Boolean(reducedMotion)}
          onMove={move}
          onView={view}
          onChartChange={setChartIndex}
        />
      ) : null}

      {busy ? (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            zIndex: 150,
            pointerEvents: "none",
          }}
        >
          <Stack alignItems="center" spacing={1.4}>
            <Box
              sx={{
                width: 42,
                height: 42,
                borderRadius: "50%",
                border: "3px solid rgba(70,38,110,0.16)",
                borderTopColor: "#6f43cf",
                animation: "mainote-spin 900ms linear infinite",
              }}
            />
            <Typography
              sx={{
                fontFamily: NUMERIC_FONT,
                fontSize: 12.5,
                fontWeight: 700,
                letterSpacing: "0.16em",
                color: "rgba(56,32,84,0.66)",
              }}
            >
              Loading...
            </Typography>
          </Stack>
        </Box>
      ) : null}

      {!busy && error ? (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            px: 3,
            zIndex: 150,
          }}
        >
          <Box
            sx={{
              width: "min(460px, 100%)",
              px: 3,
              py: 3,
              borderRadius: "4px",
              border: "1px solid rgba(255,255,255,0.65)",
              bgcolor: "rgba(255,255,255,0.76)",
              backdropFilter: "blur(16px)",
              boxShadow: "0 26px 60px rgba(58,24,96,0.22)",
              textAlign: "center",
            }}
          >
            <Typography sx={{ fontFamily: DISPLAY_FONT, fontSize: 25, color: "#2a1a3e" }}>
              无法读取曲库
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, color: "rgba(52,32,78,0.7)" }}>
              {error}
            </Typography>
            <Button
              variant="contained"
              startIcon={<Refresh />}
              onClick={() => setRefreshKey((value) => value + 1)}
              sx={{ mt: 2, bgcolor: "#6f43cf", "&:hover": { bgcolor: "#5c34b4" } }}
            >
              重试
            </Button>
          </Box>
        </Box>
      ) : null}

      {!busy && !error && count === 0 ? (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "grid",
            placeItems: "center",
            px: 3,
            zIndex: 150,
          }}
        >
          <Box
            sx={{
              width: "min(480px, 100%)",
              px: 3.5,
              py: 3.5,
              borderRadius: "4px",
              border: "1px solid rgba(255,255,255,0.65)",
              bgcolor: "rgba(255,255,255,0.74)",
              backdropFilter: "blur(16px)",
              boxShadow: "0 26px 60px rgba(58,24,96,0.2)",
              textAlign: "center",
            }}
          >
            <Typography sx={{ fontFamily: DISPLAY_FONT, fontSize: 26, color: "#2a1a3e" }}>
              {songs.length ? "没有匹配的歌曲" : "曲库是空的"}
            </Typography>
            <Typography variant="body2" sx={{ mt: 1, color: "rgba(52,32,78,0.7)" }}>
              {songs.length
                ? "换个关键词，或者清空搜索。"
                : "把包含音频与 maidata 的 zip 压缩包导入数据集，入库后就会出现在这里。"}
            </Typography>
            <Button
              variant="contained"
              onClick={() => {
                if (songs.length) {
                  setQuery("");
                  setSearchOpen(false);
                } else {
                  navigate("/datasets");
                }
              }}
              sx={{ mt: 2.5, bgcolor: "#6f43cf", "&:hover": { bgcolor: "#5c34b4" } }}
            >
              {songs.length ? "清空搜索" : "导入数据集"}
            </Button>
          </Box>
        </Box>
      ) : null}

      <Typography
        sx={{
          position: "absolute",
          left: { sm: 26 },
          bottom: { sm: 26 },
          display: { xs: "none", sm: "block" },
          zIndex: 210,
          fontSize: 11,
          color: "rgba(45,25,70,0.58)",
          letterSpacing: "0.03em",
        }}
      >
        本地曲库 · {filtered.length} 首 / {chartTotal} 谱面
        {overview ? ` · ${overview.runs} 个训练 run` : ""}
      </Typography>

      <Stack
        direction="row"
        spacing={0.75}
        alignItems="center"
        sx={{
          position: "absolute",
          right: 26,
          bottom: 26,
          display: { xs: "none", sm: "flex" },
          zIndex: 210,
          color: "rgba(45,25,70,0.55)",
          fontSize: 11,
        }}
      >
        <Keycap>A</Keycap>
        <Box component="span" sx={{ opacity: 0.5 }}>
          /
        </Box>
        <Keycap>D</Keycap>
        <Box component="span" sx={{ ml: 0.25 }}>
          切换
        </Box>
      </Stack>

      <Dialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        fullWidth
        maxWidth="sm"
        PaperProps={{
          sx: {
            bgcolor: "#f7f3fb",
            color: "#2a1a3e",
            borderRadius: "6px",
            border: "1px solid rgba(80,50,120,0.18)",
            boxShadow: "0 32px 90px rgba(40,12,70,0.36)",
          },
        }}
        slotProps={{
          backdrop: {
            sx: {
              backgroundColor: "rgba(36,18,58,0.4)",
              backdropFilter: "blur(4px)",
            },
          },
        }}
      >
        <DialogContent sx={{ px: { xs: 2.5, sm: 3.5 }, py: { xs: 2.5, sm: 3 } }}>
          <Stack direction="row" alignItems="flex-start" justifyContent="space-between">
            <Box>
              <Typography
                sx={{ fontFamily: DISPLAY_FONT, fontSize: 27, lineHeight: 1.1, color: "#2a1a3e" }}
              >
                Settings
              </Typography>
              <Typography sx={{ mt: 0.35, fontSize: 12, color: "rgba(70,48,98,0.6)" }}>
                播放速度与偏移在播放页的 MajdataView 设置中调整
              </Typography>
            </Box>
            <IconButton
              aria-label="关闭设置"
              onClick={() => setSettingsOpen(false)}
              sx={{ color: "#6d5a85", ml: 2 }}
            >
              <Close />
            </IconButton>
          </Stack>

          <Typography
            sx={{
              mt: 2.5,
              mb: 0.5,
              fontSize: 10.5,
              fontWeight: 800,
              letterSpacing: "0.18em",
              color: "#8a72a6",
            }}
          >
            DISPLAY
          </Typography>

          <SettingRow label="BACKGROUND DIM" value={bgDim.toFixed(1)}>
            <Slider
              min={0}
              max={1}
              step={0.05}
              value={bgDim}
              onChange={(_, value) => setBgDim(value as number)}
              aria-label="背景暗度"
              sx={lightSlider}
            />
          </SettingRow>

          <Divider sx={{ my: 2, borderColor: "rgba(80,50,120,0.14)" }} />

          <Typography
            sx={{
              mb: 1.25,
              fontSize: 10.5,
              fontWeight: 800,
              letterSpacing: "0.18em",
              color: "#8a72a6",
            }}
          >
            LIBRARY
          </Typography>

          <Select
            fullWidth
            value={root}
            onChange={(event) => chooseRoot(event.target.value)}
            aria-label="数据目录"
            sx={{
              bgcolor: "#ffffff",
              color: "#2a1a3e",
              fontSize: 13,
              "& .MuiOutlinedInput-notchedOutline": { borderColor: "rgba(80,50,120,0.2)" },
              "&:hover .MuiOutlinedInput-notchedOutline": {
                borderColor: "rgba(80,50,120,0.34)",
              },
              "&.Mui-focused .MuiOutlinedInput-notchedOutline": {
                borderColor: "rgba(160,108,255,0.7)",
              },
              "& .MuiSvgIcon-root": { color: "#6d5a85" },
            }}
            MenuProps={{
              PaperProps: { sx: { bgcolor: "#ffffff", color: "#2a1a3e" } },
            }}
          >
            {rootOptions.map((item) => (
              <MenuItem key={item.value} value={item.value} sx={{ fontSize: 13 }}>
                {item.label}
                <Box
                  component="span"
                  sx={{ ml: 1, fontFamily: NUMERIC_FONT, fontSize: 11, opacity: 0.55 }}
                >
                  {item.song_count} 首{item.exists ? "" : " · 不存在"}
                </Box>
              </MenuItem>
            ))}
          </Select>

          <Typography sx={{ mt: 1.1, fontSize: 11.5, color: "rgba(70,48,98,0.58)" }}>
            数据目录只在本机读取，设置保存在当前浏览器。
          </Typography>

          <Stack direction="row" justifyContent="flex-end" sx={{ mt: 2 }}>
            <Button
              size="small"
              startIcon={<Refresh />}
              onClick={() => setRefreshKey((value) => value + 1)}
              sx={{ color: "#6d5a85" }}
            >
              重新扫描
            </Button>
          </Stack>
        </DialogContent>
      </Dialog>
    </Box>
  );
}
