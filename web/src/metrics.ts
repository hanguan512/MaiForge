import type { ChartSeries } from "./components/MetricChart";
import type { MetricRow } from "./types";

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function buildTrainingSeries(metrics: MetricRow[]): ChartSeries[] {
  const series: ChartSeries[] = [
    {
      label: "train_loss",
      color: "#3fd4e6",
      values: metrics.map((row) => numberOrNull(row.train_loss)),
    },
    {
      label: "val_loss",
      color: "#ff5f9e",
      values: metrics.map((row) => numberOrNull(row.val_loss)),
    },
    {
      label: "note_score",
      color: "#ffd166",
      values: metrics.map((row) => numberOrNull(row.note_eval?.note_score)),
    },
  ];
  return series.filter((item) => item.values.some((value) => value !== null));
}

export function metricLabels(metrics: MetricRow[]): string[] {
  return metrics.map((row, index) =>
    row.epoch !== undefined ? `e${row.epoch}` : `${row.step ?? index}`,
  );
}
