import { Box, Tooltip, Typography } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { PALETTE } from "../theme";

/**
 * A note-glyph readiness marker: a filled diamond for ready, a hollow one for
 * missing. Quieter than a chip and never looks like a button.
 */
export default function ReadyMark({
  label,
  ready,
  title,
  warn = false,
}: {
  label: string;
  ready: boolean;
  title?: string;
  warn?: boolean;
}) {
  const color = ready ? PALETTE.green : warn ? PALETTE.gold : undefined;
  const mark = (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.6,
        color: ready ? "text.secondary" : warn ? "text.secondary" : "text.disabled",
        fontSize: 11.5,
        lineHeight: 1.2,
        whiteSpace: "nowrap",
      }}
    >
      <Box
        component="span"
        aria-hidden
        sx={{
          width: 6,
          height: 6,
          transform: "rotate(45deg)",
          bgcolor: ready ? color : warn ? alpha(PALETTE.gold, 0.18) : "transparent",
          border: ready ? "none" : `1px solid ${warn ? PALETTE.gold : "currentColor"}`,
          opacity: warn && !ready ? 0.8 : 1,
        }}
      />
      {label}
    </Box>
  );
  return title ? <Tooltip title={title}>{mark}</Tooltip> : mark;
}

export function ReadyMarks({
  hasAudio = true,
  sampleReady,
  featuresReady,
}: {
  hasAudio?: boolean;
  sampleReady: boolean;
  featuresReady: boolean;
}) {
  if (!hasAudio) {
    return (
      <Typography variant="caption" color="text.disabled">
        无音频
      </Typography>
    );
  }
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, flexWrap: "wrap" }}>
      <ReadyMark label="样本" ready={sampleReady} title="训练样本是否已导出" />
      <ReadyMark label="特征" ready={featuresReady} title="log-mel 特征是否已生成" />
    </Box>
  );
}
