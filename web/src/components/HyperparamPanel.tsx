import {
  Box,
  Button,
  FormControlLabel,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { RestartAlt } from "@mui/icons-material";
import { NUMERIC_FONT } from "../theme";
import type { HyperparamForm } from "../trainingForm";

type NumericKey = Exclude<keyof HyperparamForm, "amp">;

const GROUPS: Array<{
  id: string;
  title: string;
  hint: string;
  fields: Array<{ key: NumericKey; label: string; hint?: string }>;
}> = [
  {
    id: "train",
    title: "优化器与训练循环",
    hint: "写入本次 run 的 train_config.yaml，不修改 configs/ 下的原文件。",
    fields: [
      { key: "epochs", label: "epochs" },
      { key: "batchSize", label: "batch_size" },
      { key: "gradAccum", label: "grad_accum" },
      { key: "lr", label: "learning_rate" },
      { key: "weightDecay", label: "weight_decay" },
      { key: "warmupSteps", label: "warmup_steps" },
      { key: "gradClip", label: "grad_clip" },
      { key: "labelSmoothing", label: "label_smoothing" },
      { key: "patience", label: "patience" },
      { key: "saveEvery", label: "save_every" },
      { key: "logEvery", label: "log_every" },
      { key: "seed", label: "seed" },
    ],
  },
  {
    id: "data",
    title: "数据窗口",
    hint: "窗口按小节（bar）切分；验证集比例按歌曲分配，见训练数据面板。",
    fields: [
      { key: "windowBars", label: "window_bars" },
      { key: "strideBars", label: "stride_bars" },
      { key: "contextBars", label: "context_bars" },
      { key: "minEvents", label: "min_events" },
      { key: "maxWindows", label: "max_windows", hint: "每谱面窗口上限；留空 = 不限制" },
      { key: "specProb", label: "spec_augment.prob" },
    ],
  },
  {
    id: "model",
    title: "模型结构",
    hint: "与下方网络结构图联动；修改结构与 resume checkpoint 不兼容。",
    fields: [
      { key: "hidden", label: "hidden" },
      { key: "audioLayers", label: "audio_layers" },
      { key: "decoderLayers", label: "decoder_layers" },
      { key: "heads", label: "heads" },
      { key: "ffn", label: "ffn" },
      { key: "dropout", label: "dropout" },
    ],
  },
];

export default function HyperparamPanel({
  form,
  valRatio,
  onValRatioChange,
  onChange,
  onReset,
}: {
  form: HyperparamForm;
  valRatio: string;
  onValRatioChange: (value: string) => void;
  onChange: <K extends keyof HyperparamForm>(key: K, value: HyperparamForm[K]) => void;
  onReset: () => void;
}) {
  return (
    <Box>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        alignItems={{ sm: "center" }}
        justifyContent="space-between"
        spacing={1}
        sx={{ mb: 1 }}
      >
        <Typography variant="caption" color="text.secondary">
          这些值会覆盖所选配置；留空表示沿用配置文件中的值。
        </Typography>
        <Button size="small" startIcon={<RestartAlt sx={{ fontSize: 16 }} />} onClick={onReset}>
          恢复配置值
        </Button>
      </Stack>

      <Stack>
        {GROUPS.map((group, index) => (
          <Box
            key={group.id}
            sx={{
              pt: index === 0 ? 0.5 : 2,
              mt: index === 0 ? 0 : 2,
              borderTop: index === 0 ? 0 : 1,
              borderColor: "divider",
            }}
          >
            <Stack
              direction={{ xs: "column", sm: "row" }}
              alignItems={{ sm: "baseline" }}
              spacing={{ xs: 0.25, sm: 1.5 }}
              sx={{ mb: 1 }}
            >
              <Typography sx={{ fontSize: 12.5, fontWeight: 700, letterSpacing: "0.04em" }}>
                {group.title}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {group.hint}
              </Typography>
            </Stack>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: {
                  xs: "1fr 1fr",
                  sm: "repeat(3, 1fr)",
                  lg: group.id === "data" ? "repeat(4, 1fr)" : "repeat(6, 1fr)",
                },
                gap: 1.25,
              }}
            >
              {group.fields.map((field) => (
                <TextField
                  key={field.key}
                  label={field.label}
                  value={form[field.key]}
                  onChange={(event) => onChange(field.key, event.target.value)}
                  helperText={field.hint}
                  inputProps={{
                    inputMode: "decimal",
                    style: { fontFamily: NUMERIC_FONT, fontSize: 13 },
                  }}
                />
              ))}
              {group.id === "data" ? (
                <TextField
                  label="val_ratio"
                  value={valRatio}
                  onChange={(event) => onValRatioChange(event.target.value)}
                  helperText="验证歌曲比例"
                  inputProps={{
                    inputMode: "decimal",
                    style: { fontFamily: NUMERIC_FONT, fontSize: 13 },
                  }}
                />
              ) : null}
            </Box>
            {group.id === "train" ? (
              <FormControlLabel
                sx={{ mt: 0.5 }}
                control={
                  <Switch
                    size="small"
                    checked={form.amp}
                    onChange={(event) => onChange("amp", event.target.checked)}
                  />
                }
                label={<Typography variant="caption">AMP 混合精度（CUDA）</Typography>}
              />
            ) : null}
          </Box>
        ))}
      </Stack>
    </Box>
  );
}
