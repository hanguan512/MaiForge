import { alpha, createTheme } from "@mui/material/styles";
import type { Theme } from "@mui/material/styles";

export const DISPLAY_FONT = '"ZCOOL KuaiLe", "Noto Sans SC", "PingFang SC", sans-serif';
export const NUMERIC_FONT = '"Zen Maru Gothic", "Noto Sans SC", "PingFang SC", sans-serif';
export const BODY_FONT =
  '"Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif';

export const PALETTE = {
  pink: "#ff5f9e",
  pinkDeep: "#e33f80",
  aqua: "#3fd4e6",
  gold: "#ffd166",
  violet: "#a78bfa",
  green: "#5dd39e",
  red: "#ff5e62",
  ink: "#0d0f18",
  inkRaised: "#121522",
  panel: "#171a29",
  panelRaised: "#1d2133",
  line: "rgba(238, 241, 248, 0.10)",
  muted: "#98a0b8",
  text: "#eef1f8",
};

export const DIFFICULTY_COLORS: Record<string, { bg: string; fg: string; label: string }> = {
  basic: { bg: "#3fbf74", fg: "#07130c", label: "BAS" },
  advanced: { bg: "#f0b84a", fg: "#1a1204", label: "ADV" },
  expert: { bg: "#ef5b5b", fg: "#1c0707", label: "EXP" },
  master: { bg: "#a06cff", fg: "#100720", label: "MAS" },
  remaster: { bg: "#f2f3fb", fg: "#191b26", label: "REM" },
  unknown: { bg: "#5a6076", fg: "#f2f3fb", label: "???" },
};

function build(mode: "light" | "dark"): Theme {
  const dark = mode === "dark";
  const text = dark ? PALETTE.text : "#191c27";
  const muted = dark ? PALETTE.muted : "#5f667a";
  const bg = dark ? PALETTE.ink : "#eceef4";
  const panel = dark ? PALETTE.panel : "#ffffff";
  const line = dark ? PALETTE.line : "rgba(25, 28, 39, 0.13)";
  const primary = dark ? PALETTE.pink : PALETTE.pinkDeep;

  return createTheme({
    palette: {
      mode,
      primary: { main: primary, contrastText: dark ? "#1a0710" : "#ffffff" },
      secondary: { main: dark ? PALETTE.aqua : "#0e8f9e" },
      error: { main: PALETTE.red },
      warning: { main: PALETTE.gold },
      success: { main: PALETTE.green },
      info: { main: dark ? PALETTE.aqua : "#0e8f9e" },
      text: { primary: text, secondary: muted },
      background: { default: bg, paper: panel },
      divider: line,
    },
    shape: { borderRadius: 4 },
    typography: {
      fontFamily: BODY_FONT,
      h1: { fontFamily: DISPLAY_FONT, fontWeight: 400, letterSpacing: "0.01em" },
      h2: { fontFamily: DISPLAY_FONT, fontWeight: 400, letterSpacing: "0.01em" },
      h3: { fontFamily: DISPLAY_FONT, fontWeight: 400, letterSpacing: "0.01em" },
      h4: { fontFamily: DISPLAY_FONT, fontWeight: 400, letterSpacing: "0.01em" },
      h5: { fontFamily: DISPLAY_FONT, fontWeight: 400, letterSpacing: "0.01em" },
      h6: { fontFamily: DISPLAY_FONT, fontWeight: 400, letterSpacing: "0.02em" },
      subtitle1: { fontWeight: 600 },
      subtitle2: { fontWeight: 600 },
      overline: { letterSpacing: "0.12em", fontWeight: 600 },
      button: { textTransform: "none", fontWeight: 600, letterSpacing: 0 },
      body1: { lineHeight: 1.65 },
      body2: { lineHeight: 1.6 },
      caption: { lineHeight: 1.45 },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: bg,
            backgroundImage: "none",
          },
          "::selection": {
            backgroundColor: alpha(PALETTE.pink, 0.32),
          },
          "*": { scrollbarColor: `${alpha(muted, 0.45)} transparent` },
          "*::-webkit-scrollbar": { width: 10, height: 10 },
          "*::-webkit-scrollbar-thumb": {
            backgroundColor: alpha(muted, 0.35),
            borderRadius: 999,
            border: `3px solid ${bg}`,
          },
          "*::-webkit-scrollbar-track": { background: "transparent" },
        },
      },
      MuiPaper: {
        defaultProps: { elevation: 0 },
        styleOverrides: {
          root: {
            backgroundImage: "none",
            border: `1px solid ${line}`,
            boxShadow: "none",
          },
        },
      },
      MuiCard: {
        defaultProps: { elevation: 0 },
        styleOverrides: {
          root: {
            backgroundImage: "none",
            border: `1px solid ${line}`,
            boxShadow: "none",
          },
        },
      },
      MuiButton: {
        defaultProps: { disableElevation: true },
        styleOverrides: {
          root: { borderRadius: 3, paddingInline: 14 },
          outlined: { borderColor: line },
        },
      },
      MuiIconButton: {
        styleOverrides: { root: { borderRadius: 3 } },
      },
      MuiChip: {
        styleOverrides: {
          root: { borderRadius: 3, fontWeight: 600 },
          outlined: { borderColor: line },
        },
      },
      MuiTabs: {
        styleOverrides: {
          root: { minHeight: 40 },
          indicator: { height: 2 },
        },
      },
      MuiTab: {
        styleOverrides: {
          root: {
            minHeight: 40,
            paddingInline: 14,
            fontSize: 13,
            fontWeight: 600,
            letterSpacing: "0.02em",
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: { borderColor: line },
          head: {
            color: muted,
            fontSize: 12,
            fontWeight: 600,
            letterSpacing: "0.02em",
          },
        },
      },
      MuiTableRow: {
        styleOverrides: {
          root: {
            "&.MuiTableRow-hover:hover": { backgroundColor: alpha(text, 0.035) },
          },
        },
      },
      MuiAlert: {
        styleOverrides: {
          root: { borderRadius: 3, border: `1px solid ${line}` },
        },
      },
      MuiTooltip: {
        styleOverrides: {
          tooltip: { borderRadius: 3, fontSize: 12 },
        },
      },
      MuiSlider: {
        styleOverrides: {
          rail: { opacity: 0.3 },
          thumb: { borderRadius: 2, width: 12, height: 18 },
          track: { border: "none" },
        },
      },
      MuiSwitch: {
        styleOverrides: {
          root: {
            width: 38,
            height: 22,
            padding: 0,
            "& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track": {
              backgroundColor: alpha(primary, 0.55),
            },
          },
          switchBase: {
            padding: 3,
            "&.Mui-checked": { transform: "translateX(16px)" },
          },
          thumb: {
            width: 16,
            height: 16,
            borderRadius: "50%",
            backgroundColor: "#f5f7fc",
            boxShadow: "none",
          },
          track: {
            borderRadius: 999,
            opacity: "1 !important",
            backgroundColor: alpha(muted, 0.32),
          },
        },
      },
      MuiTextField: {
        defaultProps: { size: "small" },
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: { borderRadius: 3 },
          notchedOutline: { borderColor: line },
        },
      },
      MuiSelect: {
        styleOverrides: { select: { borderRadius: 3 } },
      },
      MuiDialog: {
        styleOverrides: { paper: { borderRadius: 4 } },
      },
    },
  });
}

export const darkTheme = build("dark");
export const lightTheme = build("light");
