import { Box, Typography, useTheme } from "@mui/material";

export interface ChartSeries {
  label: string;
  color: string;
  values: (number | null)[];
}

export default function MetricChart({
  series,
  xLabels,
  height = 220,
}: {
  series: ChartSeries[];
  xLabels: string[];
  height?: number;
}) {
  const theme = useTheme();
  const width = 720;
  const padding = { left: 44, right: 12, top: 14, bottom: 26 };
  const allValues = series.flatMap((item) => item.values).filter((value): value is number => value !== null);
  if (!allValues.length) {
    return (
      <Typography variant="body2" color="text.secondary" sx={{ py: 4, textAlign: "center" }}>
        暂无指标数据
      </Typography>
    );
  }
  let min = Math.min(...allValues);
  let max = Math.max(...allValues);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  const pad = (max - min) * 0.08;
  min -= pad;
  max += pad;
  const count = Math.max(2, series[0]?.values.length ?? 2);
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xAt = (index: number) => padding.left + (plotWidth * index) / (count - 1);
  const yAt = (value: number) =>
    padding.top + plotHeight * (1 - (value - min) / (max - min));

  return (
    <Box sx={{ width: "100%", overflow: "hidden" }}>
      <Box component="svg" viewBox={`0 0 ${width} ${height}`} sx={{ width: "100%", height }}>
        {[0, 0.25, 0.5, 0.75, 1].map((fraction) => {
          const y = padding.top + plotHeight * fraction;
          const value = max - (max - min) * fraction;
          return (
            <g key={fraction}>
              <line
                x1={padding.left}
                x2={width - padding.right}
                y1={y}
                y2={y}
                stroke={theme.palette.divider}
                strokeDasharray="3 4"
              />
              <text x={6} y={y + 4} fontSize={11} fill={theme.palette.text.secondary}>
                {value.toFixed(3)}
              </text>
            </g>
          );
        })}
        {series.map((item) => {
          const segments: string[] = [];
          let open = false;
          item.values.forEach((value, index) => {
            if (value === null) {
              open = false;
              return;
            }
            const command = open ? "L" : "M";
            segments.push(`${command}${xAt(index).toFixed(2)},${yAt(value).toFixed(2)}`);
            open = true;
          });
          if (!segments.length) return null;
          return (
            <path
              key={item.label}
              d={segments.join(" ")}
              fill="none"
              stroke={item.color}
              strokeWidth={2.2}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          );
        })}
        {xLabels.map((label, index) => {
          if (index % Math.max(1, Math.ceil(xLabels.length / 8)) !== 0) return null;
          return (
            <text
              key={`${label}-${index}`}
              x={xAt(index)}
              y={height - 7}
              fontSize={11}
              textAnchor="middle"
              fill={theme.palette.text.secondary}
            >
              {label}
            </text>
          );
        })}
      </Box>
      <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap", px: 1, pb: 0.5 }}>
        {series.map((item) => (
          <Box key={item.label} sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
            <Box sx={{ width: 14, height: 3, borderRadius: 1, bgcolor: item.color }} />
            <Typography variant="caption" color="text.secondary">
              {item.label}
            </Typography>
          </Box>
        ))}
      </Box>
    </Box>
  );
}
