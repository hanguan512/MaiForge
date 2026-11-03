import { Box, Typography } from "@mui/material";
import { useTheme } from "@mui/material/styles";
import { accentFor, BODY_FONT, NUMERIC_FONT } from "../theme";

export interface StatItem {
  label: string;
  value: string | number;
  accent?: string;
  detail?: string;
}

export default function StatStrip({ items }: { items: StatItem[] }) {
  const theme = useTheme();
  const lastRowStart = Math.floor((items.length - 1) / 2) * 2;
  return (
    <Box
      sx={{
        display: { xs: "grid", sm: "flex" },
        gridTemplateColumns: { xs: "1fr 1fr" },
        flexWrap: "wrap",
        alignItems: "stretch",
        borderTop: 1,
        borderBottom: 1,
        borderColor: "divider",
      }}
    >
      {items.map((item, index) => {
        const numeric =
          typeof item.value === "number" || /^-?\d(?:[\d.,%]*)$/.test(String(item.value));
        const accent = item.accent ? accentFor(theme.palette.mode, item.accent) : undefined;
        const spansLastRow = items.length % 2 === 1 && index === items.length - 1;
        return (
          <Box
            key={item.label}
            sx={{
              px: {
                xs: index % 2 === 0 ? 0 : 1.5,
                sm: index === 0 ? 0 : 3,
              },
              py: { xs: 1.25, sm: 1.6 },
              minWidth: 0,
              flex: { sm: "0 0 auto" },
              gridColumn: spansLastRow ? "1 / -1" : "auto",
              borderLeft: index === 0 ? "none" : { xs: "none", sm: 1 },
              borderRight: {
                xs: index % 2 === 0 && !spansLastRow ? 1 : "none",
                sm: 1,
              },
              borderBottom: {
                xs: index < lastRowStart ? 1 : "none",
                sm: 0,
              },
              borderColor: "divider",
              "&:last-of-type": { borderRight: "none" },
            }}
          >
            <Typography
              sx={{
                fontFamily: numeric ? NUMERIC_FONT : BODY_FONT,
                fontWeight: numeric ? 900 : 600,
                fontSize: numeric ? 25 : 14,
                lineHeight: 1.1,
                color: accent ?? "text.primary",
                fontVariantNumeric: "tabular-nums",
                wordBreak: "break-word",
              }}
            >
              {item.value}
            </Typography>
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ display: "block", mt: 0.5 }}
            >
              {item.label}
            </Typography>
            {item.detail ? (
              <Typography
                variant="caption"
                color="text.disabled"
                sx={{ display: "block", mt: 0.15, fontFamily: BODY_FONT }}
              >
                {item.detail}
              </Typography>
            ) : null}
          </Box>
        );
      })}
    </Box>
  );
}
