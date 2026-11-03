import { useMemo, useState } from "react";
import {
  Box,
  CssBaseline,
  IconButton,
  Stack,
  ThemeProvider,
  Tooltip,
  Typography,
} from "@mui/material";
import {
  AutoAwesome,
  DarkMode,
  FolderZip,
  LibraryMusic,
  LightMode,
  ModelTraining,
} from "@mui/icons-material";
import type { SvgIconComponent } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import {
  Link as RouterLink,
  Navigate,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";
import BrowsePage from "./pages/BrowsePage";
import DatasetPage from "./pages/DatasetPage";
import SongPage from "./pages/SongPage";
import ViewerPage from "./pages/ViewerPage";
import InferencePage from "./pages/InferencePage";
import TrainingPage from "./pages/TrainingPage";
import { DISPLAY_FONT, PALETTE, darkTheme, lightTheme } from "./theme";

const NAV: { label: string; hint: string; path: string; icon: SvgIconComponent }[] = [
  { label: "曲库", hint: "浏览与预览", path: "/", icon: LibraryMusic },
  { label: "数据集", hint: "导入与管理", path: "/datasets", icon: FolderZip },
  { label: "推理", hint: "音频生成谱面", path: "/infer", icon: AutoAwesome },
  { label: "训练", hint: "P3 模型训练", path: "/train", icon: ModelTraining },
];

const RAIL = 232;

function isActive(pathname: string, path: string): boolean {
  if (path === "/") {
    return (
      pathname === "/" ||
      pathname.startsWith("/song") ||
      pathname.startsWith("/viewer")
    );
  }
  return pathname.startsWith(path);
}

function BrandMark({ size = 34 }: { size?: number }) {
  const dots = Array.from({ length: 8 }, (_, index) => {
    const angle = ((67.5 - index * 45) * Math.PI) / 180;
    return {
      x: 20 + Math.cos(angle) * 16,
      y: 20 - Math.sin(angle) * 16,
    };
  });
  return (
    <Box component="svg" viewBox="0 0 40 40" sx={{ width: size, height: size, flex: "none" }}>
      <circle cx="20" cy="20" r="16.5" fill="none" stroke={PALETTE.pink} strokeWidth="1.6" opacity="0.85" />
      <circle cx="20" cy="20" r="9.5" fill="none" stroke={PALETTE.aqua} strokeWidth="1.2" opacity="0.7" />
      {dots.map((dot, index) => (
        <circle
          key={index}
          cx={dot.x}
          cy={dot.y}
          r={index % 2 === 0 ? 2.1 : 1.5}
          fill={index % 4 === 0 ? PALETTE.pink : PALETTE.text}
          opacity={index % 4 === 0 ? 1 : 0.6}
        />
      ))}
      <circle cx="20" cy="20" r="2.6" fill={PALETTE.gold} />
    </Box>
  );
}

function Wordmark() {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography
        sx={{
          fontFamily: DISPLAY_FONT,
          fontSize: 22,
          lineHeight: 1.05,
          color: "text.primary",
          whiteSpace: "nowrap",
        }}
      >
        MaiForge
      </Typography>
      <Typography variant="caption" color="text.secondary" sx={{ letterSpacing: "0.14em" }}>
        谱面工坊
      </Typography>
    </Box>
  );
}

function RailLink({
  item,
  active,
  compact = false,
}: {
  item: (typeof NAV)[number];
  active: boolean;
  compact?: boolean;
}) {
  const Icon = item.icon;
  return (
    <Box
      component={RouterLink}
      to={item.path}
      sx={{
        display: "flex",
        alignItems: "center",
        gap: compact ? 0.75 : 1.5,
        px: compact ? 1 : 2,
        py: compact ? 0.75 : 1.1,
        color: active ? "text.primary" : "text.secondary",
        bgcolor: active ? "rgba(255,95,158,0.08)" : "transparent",
        borderLeft: compact ? "none" : "3px solid",
        borderBottom: compact ? "2px solid" : "none",
        borderColor: active ? "primary.main" : "transparent",
        textDecoration: "none",
        transition: "background-color 120ms ease, color 120ms ease",
        "&:hover": {
          color: "text.primary",
          bgcolor: active ? "rgba(255,95,158,0.12)" : "action.hover",
        },
        "&:focus-visible": {
          outline: `2px solid ${PALETTE.aqua}`,
          outlineOffset: -2,
        },
      }}
    >
      <Icon sx={{ fontSize: compact ? 20 : 19 }} />
      <Box sx={{ minWidth: 0 }}>
        <Typography sx={{ fontSize: compact ? 13 : 14, fontWeight: 600, whiteSpace: "nowrap" }}>
          {item.label}
        </Typography>
        {!compact ? (
          <Typography variant="caption" color="text.secondary" sx={{ display: "block", lineHeight: 1.3 }}>
            {item.hint}
          </Typography>
        ) : null}
      </Box>
    </Box>
  );
}

export default function App() {
  const [mode, setMode] = useState<"dark" | "light">("dark");
  const location = useLocation();
  const theme = useMemo(() => (mode === "dark" ? darkTheme : lightTheme), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: "flex", minHeight: "100vh" }}>
        <Box
          component="nav"
          sx={{
            width: RAIL,
            flex: "none",
            position: "fixed",
            inset: "0 auto 0 0",
            display: { xs: "none", md: "flex" },
            flexDirection: "column",
            bgcolor: "background.paper",
            borderRight: 1,
            borderColor: "divider",
            zIndex: 20,
          }}
        >
          <Stack direction="row" spacing={1.5} alignItems="center" sx={{ px: 2, pt: 2.5, pb: 2 }}>
            <BrandMark />
            <Wordmark />
          </Stack>

          <Box sx={{ mt: 1.5, display: "flex", flexDirection: "column" }}>
            {NAV.map((item) => (
              <RailLink
                key={item.path}
                item={item}
                active={isActive(location.pathname, item.path)}
              />
            ))}
          </Box>

          <Box sx={{ flexGrow: 1 }} />

          <Box sx={{ px: 2, py: 2, borderTop: 1, borderColor: "divider" }}>
            <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1.25 }}>
              <Box
                sx={{
                  width: 7,
                  height: 7,
                  borderRadius: "50%",
                  bgcolor: PALETTE.green,
                  boxShadow: `0 0 0 3px rgba(93,211,158,0.14)`,
                }}
              />
              <Typography variant="caption" color="text.secondary">
                本地服务运行中
              </Typography>
            </Stack>
            <Typography
              variant="caption"
              color="text.disabled"
              sx={{ display: "block", fontFamily: "monospace", fontSize: 11 }}
            >
              127.0.0.1:8765
            </Typography>
            <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mt: 1.5 }}>
              <Typography variant="caption" color="text.disabled">
                数据不离开本机
              </Typography>
              <Tooltip title={mode === "dark" ? "切换到浅色" : "切换到深色"}>
                <IconButton
                  size="small"
                  onClick={() => setMode(mode === "dark" ? "light" : "dark")}
                  aria-label={mode === "dark" ? "切换到浅色主题" : "切换到深色主题"}
                >
                  {mode === "dark" ? <LightMode fontSize="small" /> : <DarkMode fontSize="small" />}
                </IconButton>
              </Tooltip>
            </Stack>
          </Box>
        </Box>

        <Box sx={{ flexGrow: 1, minWidth: 0, ml: { md: `${RAIL}px` } }}>
          <Box
            component="header"
            sx={{
              display: { xs: "flex", md: "none" },
              position: "sticky",
              top: 0,
              zIndex: 20,
              alignItems: "center",
              gap: 1,
              height: 50,
              px: 2,
              py: 0,
              bgcolor: (theme) => alpha(theme.palette.background.paper, 0.94),
              backdropFilter: "blur(10px)",
              borderBottom: 1,
              borderColor: "divider",
              overflowX: "auto",
              scrollbarWidth: "none",
              "&::-webkit-scrollbar": { display: "none" },
            }}
          >
            <BrandMark size={26} />
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
              {NAV.map((item) => (
                <RailLink
                  key={item.path}
                  item={item}
                  active={isActive(location.pathname, item.path)}
                  compact
                />
              ))}
            </Box>
            <Box sx={{ flexGrow: 1 }} />
            <IconButton
              size="small"
              onClick={() => setMode(mode === "dark" ? "light" : "dark")}
              aria-label="切换主题"
            >
              {mode === "dark" ? <LightMode fontSize="small" /> : <DarkMode fontSize="small" />}
            </IconButton>
          </Box>

          <Box
            component="main"
            sx={{
              px: { xs: 2, sm: 3, lg: 4 },
              py: { xs: 2.5, md: 3.5 },
              maxWidth: 1600,
              mx: "auto",
            }}
          >
            <Routes>
              <Route path="/" element={<BrowsePage />} />
              <Route path="/datasets" element={<DatasetPage />} />
              <Route path="/song" element={<SongPage />} />
              <Route path="/viewer" element={<ViewerPage />} />
              <Route path="/infer" element={<InferencePage />} />
              <Route path="/train" element={<TrainingPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
