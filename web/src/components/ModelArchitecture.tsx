import { Box, CircularProgress, Stack, Typography } from "@mui/material";
import { alpha } from "@mui/material/styles";
import { NUMERIC_FONT, PALETTE } from "../theme";
import type { ModelArchitecture, ModelLayer, ModelLayerGroup } from "../types";

const KIND_COLORS: Record<string, string> = {
  input: PALETTE.aqua,
  conv: PALETTE.gold,
  pos: PALETTE.muted,
  encoder: PALETTE.violet,
  norm: PALETTE.green,
  embed: PALETTE.pink,
  decoder: PALETTE.pinkDeep,
  head: PALETTE.green,
};

export function formatParamCount(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  if (value >= 1_000_000_000) return `${(value / 1_000_000_000).toFixed(2)}B`;
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return String(value);
}

function LayerCard({ layer, index }: { layer: ModelLayer; index: number }) {
  const color = KIND_COLORS[layer.kind] ?? PALETTE.muted;
  return (
    <Box
      sx={{
        position: "relative",
        border: 1,
        borderColor: alpha(color, 0.34),
        borderLeft: `2px solid ${color}`,
        borderRadius: "3px",
        bgcolor: alpha(color, 0.055),
        px: 1.15,
        py: 0.8,
      }}
    >
      <Stack direction="row" alignItems="baseline" spacing={1}>
        <Typography
          sx={{
            minWidth: 18,
            color: alpha(color, 0.9),
            fontFamily: NUMERIC_FONT,
            fontSize: 10,
            fontWeight: 700,
          }}
        >
          {String(index + 1).padStart(2, "0")}
        </Typography>
        <Typography sx={{ fontSize: 12, fontWeight: 700, lineHeight: 1.3 }}>
          {layer.name}
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <Typography
          sx={{
            color: "text.secondary",
            fontFamily: NUMERIC_FONT,
            fontSize: 10,
            fontVariantNumeric: "tabular-nums",
            whiteSpace: "nowrap",
          }}
        >
          {layer.params > 0 ? formatParamCount(layer.params) : "—"}
        </Typography>
      </Stack>
      <Typography
        sx={{
          mt: 0.35,
          color: alpha(color, 0.92),
          fontFamily: NUMERIC_FONT,
          fontSize: 10.5,
          letterSpacing: "0.01em",
        }}
      >
        {layer.shape}
      </Typography>
      {layer.detail ? (
        <Typography
          sx={{ mt: 0.2, color: "text.secondary", fontSize: 10, lineHeight: 1.45 }}
        >
          {layer.detail}
        </Typography>
      ) : null}
    </Box>
  );
}

function Group({ group }: { group: ModelLayerGroup }) {
  const color = KIND_COLORS[group.layers[0]?.kind ?? "pos"] ?? PALETTE.muted;
  return (
    <Box>
      <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 0.7 }}>
        <Box
          sx={{
            width: 5,
            height: 5,
            borderRadius: "50%",
            bgcolor: color,
            boxShadow: `0 0 8px ${alpha(color, 0.8)}`,
          }}
        />
        <Typography
          variant="overline"
          sx={{ color: "text.secondary", fontSize: 10, lineHeight: 1.2 }}
        >
          {group.label}
        </Typography>
        <Box sx={{ flexGrow: 1, borderTop: 1, borderColor: "divider" }} />
        <Typography
          sx={{
            color: "text.secondary",
            fontFamily: NUMERIC_FONT,
            fontSize: 10,
          }}
        >
          {group.layers.length} 层
        </Typography>
      </Stack>
      <Stack spacing={0.65}>
        {group.layers.map((layer, index) => (
          <LayerCard key={layer.id} layer={layer} index={index} />
        ))}
      </Stack>
    </Box>
  );
}

function Column({ groups, title }: { groups: ModelLayerGroup[]; title: string }) {
  return (
    <Box sx={{ minWidth: 0 }}>
      <Typography
        variant="subtitle2"
        sx={{ mb: 1.1, fontSize: 12.5, letterSpacing: "0.03em" }}
      >
        {title}
      </Typography>
      <Stack spacing={1.35}>
        {groups.map((group) => (
          <Group key={group.id} group={group} />
        ))}
      </Stack>
    </Box>
  );
}

export default function ModelArchitectureView({
  architecture,
  loading,
  error,
}: {
  architecture: ModelArchitecture | null;
  loading?: boolean;
  error?: string | null;
}) {
  if (error) {
    return (
      <Box sx={{ py: 4, textAlign: "center" }}>
        <Typography variant="body2" color="error.main">
          {error}
        </Typography>
      </Box>
    );
  }
  if (!architecture) {
    return (
      <Box sx={{ py: 6, textAlign: "center" }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  const encoderGroups = architecture.groups.filter(
    (group) => group.column === "encoder",
  );
  const decoderGroups = architecture.groups.filter(
    (group) => group.column !== "encoder",
  );

  return (
    <Stack spacing={1.75}>
      <Stack
        direction="row"
        spacing={2}
        flexWrap="wrap"
        useFlexGap
        alignItems="center"
      >
        <Typography sx={{ fontFamily: NUMERIC_FONT, fontSize: 20, fontWeight: 700 }}>
          {formatParamCount(architecture.total_params)}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          可训练参数 · 词表 {architecture.vocab_size.toLocaleString()} · mel{" "}
          {architecture.mel_fps.toFixed(1)} fps → memory{" "}
          {architecture.encoded_fps.toFixed(1)} fps
        </Typography>
        {loading ? <CircularProgress size={14} /> : null}
      </Stack>

      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" },
          gap: { xs: 2, md: 2.5 },
          alignItems: "start",
        }}
      >
        <Column groups={encoderGroups} title="音频编码器 · AUDIO ENCODER" />
        <Box sx={{ position: "relative" }}>
          <Box
            sx={{
              display: { xs: "none", md: "block" },
              position: "absolute",
              top: 46,
              left: -18,
              bottom: 18,
              borderLeft: `1px dashed ${alpha(PALETTE.violet, 0.45)}`,
            }}
          />
          <Column groups={decoderGroups} title="事件解码器 · EVENT DECODER" />
        </Box>
      </Box>

      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: "block", pt: 0.25 }}
      >
        编码器 memory 通过 cross-attention 进入每个 Decoder Layer；Token
        Embedding 与 LM Head 共享权重，参数只计一次。参数按 PyTorch
        norm_first 层的默认结构估算。
      </Typography>
    </Stack>
  );
}
