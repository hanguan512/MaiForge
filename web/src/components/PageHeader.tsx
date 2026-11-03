import type { ReactNode } from "react";
import { Box, Stack, Typography } from "@mui/material";

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
    <Box sx={{ borderBottom: 1, borderColor: "divider", pb: 2 }}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={2}
        alignItems={{ sm: "flex-end" }}
        justifyContent="space-between"
      >
        <Box sx={{ minWidth: 0 }}>
          <Typography
            variant="h3"
            sx={{ fontSize: { xs: 30, md: 38 }, lineHeight: 1.12, color: "text.primary" }}
          >
            {title}
          </Typography>
          {description ? (
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mt: 0.75, maxWidth: 720 }}
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
    </Box>
  );
}
