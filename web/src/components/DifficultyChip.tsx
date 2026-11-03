import { Box } from "@mui/material";
import { DIFFICULTY_COLORS, NUMERIC_FONT } from "../theme";

const FALLBACK: Record<string, { bg: string; fg: string; label: string }> = {
  easy: { bg: "#6fb7d8", fg: "#04141c", label: "EAS" },
  original: { bg: "#7ad0c0", fg: "#031713", label: "ORI" },
  utage: { bg: "#f38ba8", fg: "#1c0710", label: "宴" },
};

export default function DifficultyChip({
  difficulty,
  level,
  size = "small",
}: {
  difficulty: string;
  level?: string | number;
  size?: "small" | "medium";
}) {
  const key = (difficulty || "unknown").toLowerCase();
  const colors = DIFFICULTY_COLORS[key] ?? FALLBACK[key] ?? DIFFICULTY_COLORS.unknown;
  const medium = size === "medium";
  return (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: medium ? 0.8 : 0.55,
        height: medium ? 26 : 21,
        px: medium ? 1 : 0.7,
        borderRadius: "2px",
        bgcolor: colors.bg,
        color: colors.fg,
        fontSize: medium ? 12 : 10,
        fontWeight: 800,
        letterSpacing: "0.03em",
        lineHeight: 1,
        whiteSpace: "nowrap",
      }}
    >
      {colors.label}
      {level ? (
        <Box
          component="span"
          sx={{
            fontFamily: NUMERIC_FONT,
            fontWeight: 900,
            fontSize: medium ? 15 : 12,
            letterSpacing: 0,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {level}
        </Box>
      ) : null}
    </Box>
  );
}
