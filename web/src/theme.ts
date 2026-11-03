import { alpha, createTheme } from "@mui/material/styles";
import type { Theme } from "@mui/material/styles";

export const DISPLAY_FONT =
  '"ZCOOL KuaiLe", "Noto Sans CJK JP", "Noto Sans SC", "PingFang SC", sans-serif';
export const NUMERIC_FONT =
  '"Zen Maru Gothic", "Noto Sans CJK JP", "Noto Sans SC", "PingFang SC", sans-serif';
export const BODY_FONT =
  '"Noto Sans SC", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", system-ui, sans-serif';

export const PALETTE = {
  pink: "#ff5f9e",
  pinkDeep: "#e33f80",
  aqua: "#43d9e8",
  gold: "#ffd166",
  violet: "#a78bfa",
  green: "#5dd39e",
  red: "#ff5e62",
  ink: "#070a13",
  inkRaised: "#0b0f19",
  panel: "#0d111c",
  panelRaised: "#121827",
  line: "rgba(226, 232, 245, 0.10)",
  lineStrong: "rgba(226, 232, 245, 0.2)",
  muted: "#939bb2",
  text: "#eef1f8",
};

/**
 * Accent colors are tuned for the dark stage; on the light surface they lose
 * contrast, so darken them for text-sized marks.
 */
export function accentFor(mode: string, color: string): string {
  if (mode !== "light") return color;
  const value = color.replace("#", "");
  if (!/^[0-9a-fA-F]{6}$/.test(value)) return color;
  const n = parseInt(value, 16);
  const channel = (shift: number) =>
    Math.round((((n >> shift) & 0xff) * 0.58 + 0x10 * 0.42) * 0.92);
  return `rgb(${channel(16)}, ${channel(8)}, ${channel(0)})`;
}

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
  const bg = dark ? PALETTE.ink : "#eef0f5";
  const panel = dark ? PALETTE.panel : "#ffffff";
  const line = dark ? PALETTE.line : "rgba(25, 28, 39, 0.13)";
  const lineStrong = dark ? PALETTE.lineStrong : "rgba(25, 28, 39, 0.24)";
  const primary = dark ? PALETTE.pink : PALETTE.pinkDeep;
  const accent = dark ? PALETTE.aqua : "#0e8f9e";

  return createTheme({
    palette: {
      mode,
      primary: { main: primary, contrastText: dark ? "#1a0710" : "#ffffff" },
      secondary: { main: accent },
      error: { main: PALETTE.red },
      warning: { main: PALETTE.gold },
      success: { main: PALETTE.green },
      info: { main: accent },
      text: { primary: text, secondary: muted },
      background: { default: bg, paper: panel },
      divider: line,
    },
    shape: { borderRadius: 2 },
    typography: {
      fontFamily: BODY_FONT,
      h1: {
        fontFamily: DISPLAY_FONT,
        fontWeight: 400,
        fontSize: 48,
        lineHeight: 1.08,
        letterSpacing: "0.01em",
      },
      h2: {
        fontFamily: DISPLAY_FONT,
        fontWeight: 400,
        fontSize: 38,
        lineHeight: 1.1,
        letterSpacing: "0.01em",
      },
      h3: {
        fontFamily: DISPLAY_FONT,
        fontWeight: 400,
        fontSize: 28,
        lineHeight: 1.15,
        letterSpacing: "0.01em",
      },
      h4: {
        fontFamily: DISPLAY_FONT,
        fontWeight: 400,
        fontSize: 22,
        lineHeight: 1.2,
        letterSpacing: "0.01em",
      },
      h5: { fontFamily: DISPLAY_FONT, fontWeight: 400, fontSize: 18, lineHeight: 1.25 },
      h6: { fontFamily: DISPLAY_FONT, fontWeight: 400, fontSize: 15.5, lineHeight: 1.3 },
      subtitle1: { fontWeight: 600, fontSize: 14.5 },
      subtitle2: { fontWeight: 600, fontSize: 13 },
      overline: { letterSpacing: "0.04em", fontWeight: 600, textTransform: "none" },
      button: { textTransform: "none", fontWeight: 600, letterSpacing: 0 },
      body1: { fontSize: 14.5, lineHeight: 1.7 },
      body2: { fontSize: 13.5, lineHeight: 1.6 },
      caption: { fontSize: 11.5, lineHeight: 1.45 },
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            backgroundColor: bg,
            backgroundImage: "none",
            WebkitFontSmoothing: "antialiased",
          },
          "::selection": {
            backgroundColor: alpha(PALETTE.pink, 0.32),
          },
          "*": { scrollbarColor: `${alpha(muted, 0.4)} transparent` },
          "*::-webkit-scrollbar": { width: 9, height: 9 },
          "*::-webkit-scrollbar-thumb": {
            backgroundColor: alpha(muted, 0.3),
            borderRadius: 999,
            border: `3px solid ${bg}`,
          },
          "*::-webkit-scrollbar-track": { background: "transparent" },
          "@keyframes mainote-rise": {
            from: { opacity: 0 },
            to: { opacity: 1 },
          },
          "@keyframes mainote-lane": {
            from: { transform: "scaleY(0.2)", opacity: 0 },
            to: { transform: "scaleY(1)", opacity: 1 },
          },
          "@media (prefers-reduced-motion: reduce)": {
            "*, *::before, *::after": {
              animationDuration: "0.001ms !important",
              animationIterationCount: "1 !important",
              transitionDuration: "0.001ms !important",
              scrollBehavior: "auto !important",
            },
          },
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
          root: {
            borderRadius: 2,
            paddingInline: 14,
            "&.Mui-focusVisible": { outline: `2px solid ${accent}`, outlineOffset: 2 },
          },
          contained: {
            "&:hover": { backgroundColor: dark ? PALETTE.pinkDeep : "#c92f6d" },
          },
          outlined: { borderColor: lineStrong },
          text: {
            "&.MuiButton-textPrimary": { color: text },
            "&:hover": { backgroundColor: alpha(text, 0.06) },
            "&.Mui-disabled": { color: alpha(text, 0.32) },
          },
        },
      },
      MuiIconButton: {
        styleOverrides: {
          root: {
            borderRadius: 2,
            "&.Mui-focusVisible": { outline: `2px solid ${accent}`, outlineOffset: 1 },
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: { borderRadius: 2, fontWeight: 600, height: 22, fontSize: 11 },
          outlined: { borderColor: lineStrong },
        },
      },
      MuiTabs: {
        styleOverrides: {
          root: { minHeight: 44 },
          indicator: { height: 2 },
        },
      },
      MuiTab: {
        styleOverrides: {
          root: {
            minHeight: 44,
            paddingInline: 2,
            marginRight: 20,
            fontSize: 13.5,
            fontWeight: 600,
            color: muted,
            "&.Mui-selected": { color: text },
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: { borderColor: line, padding: "9px 10px" },
          sizeSmall: { padding: "9px 10px" },
          head: {
            color: muted,
            fontSize: 11.5,
            fontWeight: 600,
            letterSpacing: "0.02em",
            borderBottomColor: lineStrong,
          },
        },
      },
      MuiTableRow: {
        styleOverrides: {
          root: {
            "&.MuiTableRow-hover:hover": { backgroundColor: alpha(text, 0.03) },
          },
        },
      },
      MuiAlert: {
        styleOverrides: {
          root: { borderRadius: 2, border: `1px solid ${line}`, fontSize: 13 },
        },
      },
      MuiTooltip: {
        styleOverrides: {
          tooltip: {
            borderRadius: 2,
            fontSize: 11.5,
            backgroundColor: dark ? "#1d2434" : "#2b3040",
            border: `1px solid ${lineStrong}`,
            padding: "6px 9px",
          },
        },
      },
      MuiSlider: {
        styleOverrides: {
          rail: { opacity: 0.3 },
          thumb: { borderRadius: 1, width: 10, height: 16 },
        },
      },
      MuiSwitch: {
        styleOverrides: {
          root: {
            width: 36,
            height: 21,
            padding: 0,
            "& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track": {
              backgroundColor: alpha(primary, 0.6),
            },
          },
          switchBase: {
            padding: 3,
            "&.Mui-checked": { transform: "translateX(15px)" },
          },
          thumb: {
            width: 15,
            height: 15,
            borderRadius: "50%",
            backgroundColor: "#f5f7fc",
            boxShadow: "none",
          },
          track: {
            borderRadius: 999,
            opacity: "1 !important",
            backgroundColor: alpha(muted, 0.3),
          },
        },
      },
      MuiTextField: {
        defaultProps: { size: "small" },
      },
      MuiOutlinedInput: {
        styleOverrides: {
          root: {
            borderRadius: 2,
            backgroundColor: dark ? "rgba(255,255,255,0.018)" : "transparent",
            "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: lineStrong },
          },
          notchedOutline: { borderColor: line },
        },
      },
      MuiSelect: {
        styleOverrides: { select: { borderRadius: 2 } },
      },
      MuiDialog: {
        styleOverrides: {
          paper: {
            borderRadius: 2,
            backgroundColor: dark ? PALETTE.panelRaised : "#ffffff",
            border: `1px solid ${lineStrong}`,
          },
        },
      },
      MuiAccordion: {
        styleOverrides: {
          root: {
            border: "none",
            backgroundImage: "none",
            backgroundColor: "transparent",
            "&::before": { display: "none" },
          },
        },
      },
      MuiAccordionSummary: {
        styleOverrides: { root: { minHeight: 42, paddingInline: 0 } },
      },
      MuiAccordionDetails: { styleOverrides: { root: { padding: "0 0 12px" } } },
      MuiLinearProgress: {
        styleOverrides: {
          root: { height: 3, borderRadius: 0, backgroundColor: alpha(muted, 0.18) },
          bar: { borderRadius: 0 },
        },
      },
      MuiDivider: { styleOverrides: { root: { borderColor: line } } },
    },
  });
}

export const darkTheme = build("dark");
export const lightTheme = build("light");
