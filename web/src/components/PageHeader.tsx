import type { ReactNode } from "react";
import { Box, Stack, Typography } from "@mui/material";

function BeatRuler() {
  return (
    <Box
      aria-hidden
      sx={{
        position: "relative",
        height: 11,
        mt: 2.5,
        borderBottom: 1,
        borderColor: "divider",
        backgroundImage: (theme) =>
          `repeating-linear-gradient(90deg, ${theme.palette.divider} 0 1px, transparent 1px 112px), repeating-linear-gradient(90deg, ${theme.palette.divider} 0 1px, transparent 1px 28px)`,
        backgroundSize: "auto 10px, auto 4px",
        backgroundPosition: "left bottom, left bottom",
        backgroundRepeat: "repeat-x",
      }}
    >
      <Box
        sx={{
          position: "absolute",
          left: 0,
          bottom: -1,
          width: 56,
          height: 2,
          bgcolor: "primary.main",
        }}
      />
    </Box>
  );
}

export default function PageHeader({
  title,
  description,
  actions,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <Box>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        alignItems={{ sm: "flex-end" }}
        justifyContent="space-between"
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography
            variant="h1"
            sx={{ fontSize: { xs: 32, sm: 38, md: 44 }, color: "text.primary" }}
          >
            {title}
          </Typography>
          {description ? (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mt: 1, maxWidth: 760, lineHeight: 1.75 }}
            >
              {description}
            </Typography>
          ) : null}
        </Box>
        {actions ? (
          <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap sx={{ flex: "none" }}>
            {actions}
          </Stack>
        ) : null}
      </Stack>
      <BeatRuler />
    </Box>
  );
}
