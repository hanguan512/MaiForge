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

const RAIL = 216;

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
      <circle cx="20" cy="20" r="16.5" fill="none" stroke={PALETTE.pink} strokeWidth="1.4" opacity="0.85" />
      <circle cx="20" cy="20" r="9.5" fill="none" stroke={PALETTE.aqua} strokeWidth="1" opacity="0.65" />
      {dots.map((dot, index) => (
        <circle
          key={index}
          cx={dot.x}
          cy={dot.y}
          r={index % 2 === 0 ? 2.1 : 1.4}
          fill={index % 4 === 0 ? PALETTE.pink : PALETTE.text}
          opacity={index % 4 === 0 ? 1 : 0.55}
        />
      ))}
      <circle cx="20" cy="20" r="2.2" fill={PALETTE.gold} />
    </Box>
  );
}

function Wordmark() {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography
        sx={{
          fontFamily: DISPLAY_FONT,
          fontSize: 21,
          lineHeight: 1.05,
          color: "text.primary",
          whiteSpace: "nowrap",
        }}
      >
        MaiForge
      </Typography>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: "block", mt: 0.35, letterSpacing: "0.16em" }}
      >
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
  if (compact) {
    return (
      <Box
        component={RouterLink}
        to={item.path}
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 0.75,
          px: 1.1,
          py: 0.6,
          flex: "none",
          color: active ? "text.primary" : "text.secondary",
          bgcolor: active ? alpha(PALETTE.pink, 0.1) : "transparent",
          borderBottom: "2px solid",
          borderColor: active ? "primary.main" : "transparent",
          textDecoration: "none",
          "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: -2 },
        }}
      >
        <Icon sx={{ fontSize: 18, color: active ? PALETTE.pink : "inherit" }} />
        <Typography sx={{ fontSize: 13, fontWeight: 600, whiteSpace: "nowrap" }}>
          {item.label}
        </Typography>
      </Box>
    );
  }

  return (
    <Box
      component={RouterLink}
      to={item.path}
      sx={{
        position: "relative",
        display: "flex",
        alignItems: "center",
        gap: 1.5,
        pl: 2.25,
        pr: 1.5,
        py: 1.15,
        color: active ? "text.primary" : "text.secondary",
        textDecoration: "none",
        background: active
          ? `linear-gradient(90deg, ${alpha(PALETTE.pink, 0.1)}, transparent 72%)`
          : "transparent",
        transition: "background-color 140ms ease, color 140ms ease",
        "&::before": {
          content: '""',
          position: "absolute",
          left: -4,
          top: "50%",
          mt: "-3.5px",
          width: 7,
          height: 7,
          borderRadius: "50%",
          bgcolor: active ? PALETTE.pink : "background.paper",
          border: active ? "none" : "1px solid",
          borderColor: "divider",
          boxShadow: active ? `0 0 0 4px ${alpha(PALETTE.pink, 0.1)}` : "none",
          transition: "background-color 140ms ease, box-shadow 140ms ease",
        },
        "&:hover": {
          color: "text.primary",
          bgcolor: active ? alpha(PALETTE.pink, 0.13) : "action.hover",
        },
        "&:focus-visible": {
          outline: `2px solid ${PALETTE.aqua}`,
          outlineOffset: -2,
        },
      }}
    >
      <Icon sx={{ fontSize: 18, color: active ? PALETTE.pink : "inherit", flex: "none" }} />
      <Box sx={{ minWidth: 0 }}>
        <Typography sx={{ fontSize: 13.5, fontWeight: 600, lineHeight: 1.35 }}>
          {item.label}
        </Typography>
        <Typography
          variant="caption"
          color="text.disabled"
          sx={{ display: "block", lineHeight: 1.35 }}
        >
          {item.hint}
        </Typography>
      </Box>
    </Box>
  );
}

export default function App() {
  const [mode, setMode] = useState<"dark" | "light">("dark");
  const location = useLocation();
  const theme = useMemo(() => (mode === "dark" ? darkTheme : lightTheme), [mode]);
  const toggleMode = () => setMode(mode === "dark" ? "light" : "dark");

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
          <Stack
            direction="row"
            spacing={1.4}
            alignItems="center"
            sx={{ px: 2.25, pt: 2.75, pb: 3 }}
          >
            <BrandMark />
            <Wordmark />
          </Stack>

          <Box
            sx={{
              ml: 2.75,
              borderLeft: "1px solid",
              borderColor: "divider",
              display: "flex",
              flexDirection: "column",
            }}
          >
            {NAV.map((item) => (
              <RailLink
                key={item.path}
                item={item}
                active={isActive(location.pathname, item.path)}
              />
            ))}
          </Box>

          <Box sx={{ flexGrow: 1 }} />

          <Box sx={{ px: 2.25, py: 2, borderTop: 1, borderColor: "divider" }}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box
                sx={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  bgcolor: PALETTE.green,
                  boxShadow: `0 0 0 3px ${alpha(PALETTE.green, 0.14)}`,
                }}
              />
              <Typography variant="caption" color="text.secondary">
                本地服务运行中
              </Typography>
            </Stack>
            <Typography
              variant="caption"
              color="text.disabled"
              sx={{ display: "block", mt: 0.75, fontVariantNumeric: "tabular-nums" }}
            >
              127.0.0.1:8765
            </Typography>
            <Stack
              direction="row"
              alignItems="center"
              justifyContent="space-between"
              sx={{ mt: 1.25 }}
            >
              <Typography variant="caption" color="text.disabled">
                数据不离开本机
              </Typography>
              <Tooltip title={mode === "dark" ? "切换到浅色" : "切换到深色"}>
                <IconButton
                  size="small"
                  onClick={toggleMode}
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
              height: 52,
              px: 2,
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
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.25 }}>
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
            <IconButton size="small" onClick={toggleMode} aria-label="切换主题">
              {mode === "dark" ? <LightMode fontSize="small" /> : <DarkMode fontSize="small" />}
            </IconButton>
          </Box>

          <Box
            component="main"
            sx={{
              px: { xs: 2, sm: 3, lg: 4.5 },
              py: { xs: 3, md: 4.5 },
              maxWidth: 1440,
              mx: "auto",
            }}
          >
            <Box
              key={location.pathname}
              sx={{ animation: "mainote-rise 360ms cubic-bezier(0.22, 1, 0.36, 1) both" }}
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
      </Box>
    </ThemeProvider>
  );
}
