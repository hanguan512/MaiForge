import { Box, Typography } from "@mui/material";
import { BODY_FONT, NUMERIC_FONT } from "../theme";

export interface StatItem {
  label: string;
  value: string | number;
  accent?: string;
  detail?: string;
}

export default function StatStrip({ items }: { items: StatItem[] }) {
  return (
    <Box
      sx={{
        display: "flex",
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
        return (
          <Box
            key={item.label}
            sx={{
              px: index === 0 ? 0 : { xs: 2, sm: 3 },
              py: 1.5,
              minWidth: { xs: "50%", sm: 0 },
              flex: { sm: "0 0 auto" },
              borderLeft: index === 0 ? "none" : { xs: "none", sm: 1 },
              borderRight: { xs: "none", sm: 1 },
              borderColor: "divider",
              "&:last-of-type": { borderRight: "none" },
            }}
          >
            <Typography
              sx={{
                fontFamily: numeric ? NUMERIC_FONT : BODY_FONT,
                fontWeight: numeric ? 900 : 600,
                fontSize: numeric ? 24 : 15,
                lineHeight: 1.1,
                color: item.accent ?? "text.primary",
                fontVariantNumeric: "tabular-nums",
                wordBreak: "break-word",
              }}
            >
              {item.value}
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.4 }}>
              {item.label}
            </Typography>
            {item.detail ? (
              <Typography variant="caption" color="text.disabled" sx={{ display: "block" }}>
                {item.detail}
              </Typography>
            ) : null}
          </Box>
        );
      })}
    </Box>
  );
}
