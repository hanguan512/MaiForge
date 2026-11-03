import type { ReactNode } from "react";
import { Box, Stack, Typography } from "@mui/material";
import type { SxProps, Theme } from "@mui/material/styles";
import { alpha } from "@mui/material/styles";
import { PALETTE } from "../theme";

/**
 * A page section laid out like a lane in the playfield: a short label sits in
 * a narrow left column with a note marker on the lane line, the content runs
 * to its right. On small screens the lane folds into a normal section header.
 */
export default function Track({
  label,
  hint,
  action,
  children,
  accent = PALETTE.pink,
  first = false,
  contentSx,
  sx,
}: {
  label: ReactNode;
  hint?: ReactNode;
  action?: ReactNode;
  children: ReactNode;
  accent?: string;
  first?: boolean;
  contentSx?: SxProps<Theme>;
  sx?: SxProps<Theme>;
}) {
  return (
    <Box
      component="section"
      sx={[
        {
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "148px minmax(0, 1fr)" },
          borderTop: first ? 0 : 1,
          borderColor: "divider",
          position: "relative",
        },
        ...(Array.isArray(sx) ? sx : [sx]),
      ]}
    >
      <Box
        sx={{
          pt: { xs: 1.75, md: 2.5 },
          pb: { xs: 0.5, md: 2.5 },
          pr: { md: 2.5 },
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-start",
          gap: 0.6,
          minWidth: 0,
        }}
      >
        <Stack direction="row" alignItems="center" spacing={1} sx={{ flex: "none" }}>
          <Box
            aria-hidden
            sx={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              bgcolor: accent,
              flex: "none",
              boxShadow: `0 0 0 3px ${alpha(accent, 0.12)}`,
            }}
          />
          <Typography
            component="h2"
            sx={{
              fontSize: 12.5,
              fontWeight: 700,
              letterSpacing: "0.08em",
              color: "text.primary",
              whiteSpace: "nowrap",
            }}
          >
            {label}
          </Typography>
        </Stack>
        {hint ? (
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ lineHeight: 1.55, maxWidth: { xs: "none", md: 150 } }}
          >
            {hint}
          </Typography>
        ) : null}
      </Box>

      <Box
        sx={[
          {
            minWidth: 0,
            pt: { xs: 1, md: 2.5 },
            pb: { xs: 2.5, md: 2.5 },
            pl: { md: 3 },
            borderLeft: { md: 1 },
            borderColor: "divider",
          },
          ...(Array.isArray(contentSx) ? contentSx : [contentSx]),
        ]}
      >
        {action ? (
          <Stack
            direction="row"
            spacing={1}
            alignItems="center"
            justifyContent="flex-end"
            flexWrap="wrap"
            useFlexGap
            sx={{ mb: 1.5 }}
          >
            {action}
          </Stack>
        ) : null}
        {children}
      </Box>
    </Box>
  );
}
