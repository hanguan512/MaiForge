export interface HyperparamForm {
  seed: string;
  epochs: string;
  batchSize: string;
  gradAccum: string;
  lr: string;
  weightDecay: string;
  warmupSteps: string;
  gradClip: string;
  labelSmoothing: string;
  patience: string;
  saveEvery: string;
  logEvery: string;
  amp: boolean;
  windowBars: string;
  strideBars: string;
  contextBars: string;
  minEvents: string;
  maxWindows: string;
  specProb: string;
  hidden: string;
  audioLayers: string;
  decoderLayers: string;
  heads: string;
  ffn: string;
  dropout: string;
}

export const DEFAULT_HYPERPARAMS: HyperparamForm = {
  seed: "42",
  epochs: "30",
  batchSize: "8",
  gradAccum: "2",
  lr: "0.0003",
  weightDecay: "0.01",
  warmupSteps: "500",
  gradClip: "1.0",
  labelSmoothing: "0.0",
  patience: "8",
  saveEvery: "5",
  logEvery: "20",
  amp: true,
  windowBars: "16",
  strideBars: "8",
  contextBars: "2",
  minEvents: "1",
  maxWindows: "",
  specProb: "0.5",
  hidden: "256",
  audioLayers: "4",
  decoderLayers: "6",
  heads: "8",
  ffn: "1024",
  dropout: "0.1",
};

type Section = Record<string, unknown>;

function section(config: Record<string, unknown>, key: string): Section {
  const value = config[key];
  return value && typeof value === "object" ? (value as Section) : {};
}

function text(value: unknown, fallback: string): string {
  if (value === null || value === undefined) return fallback;
  return String(value);
}

export function hyperparamsFromConfig(
  config: Record<string, unknown>,
): HyperparamForm {
  const train = section(config, "train");
  const data = section(config, "data");
  const model = section(config, "model");
  const spec = data.spec_augment && typeof data.spec_augment === "object"
    ? (data.spec_augment as Section)
    : {};
  const pick = (value: unknown, fallback: string) =>
    value === null || value === undefined ? fallback : String(value);
  return {
    seed: pick(config.seed, DEFAULT_HYPERPARAMS.seed),
    epochs: pick(train.epochs, DEFAULT_HYPERPARAMS.epochs),
    batchSize: pick(train.batch_size, DEFAULT_HYPERPARAMS.batchSize),
    gradAccum: pick(train.grad_accum, DEFAULT_HYPERPARAMS.gradAccum),
    lr: pick(train.lr, DEFAULT_HYPERPARAMS.lr),
    weightDecay: pick(train.weight_decay, DEFAULT_HYPERPARAMS.weightDecay),
    warmupSteps: pick(train.warmup_steps, DEFAULT_HYPERPARAMS.warmupSteps),
    gradClip: pick(train.grad_clip, DEFAULT_HYPERPARAMS.gradClip),
    labelSmoothing: pick(
      train.label_smoothing,
      DEFAULT_HYPERPARAMS.labelSmoothing,
    ),
    patience: pick(train.patience, DEFAULT_HYPERPARAMS.patience),
    saveEvery: pick(train.save_every, DEFAULT_HYPERPARAMS.saveEvery),
    logEvery: pick(train.log_every, DEFAULT_HYPERPARAMS.logEvery),
    amp:
      typeof train.amp === "boolean"
        ? train.amp
        : DEFAULT_HYPERPARAMS.amp,
    windowBars: pick(data.window_bars, DEFAULT_HYPERPARAMS.windowBars),
    strideBars: pick(data.stride_bars, DEFAULT_HYPERPARAMS.strideBars),
    contextBars: pick(data.context_bars, DEFAULT_HYPERPARAMS.contextBars),
    minEvents: pick(data.min_events, DEFAULT_HYPERPARAMS.minEvents),
    maxWindows: text(data.max_windows_per_chart, DEFAULT_HYPERPARAMS.maxWindows),
    specProb: pick(spec.prob, DEFAULT_HYPERPARAMS.specProb),
    hidden: pick(model.hidden, DEFAULT_HYPERPARAMS.hidden),
    audioLayers: pick(model.audio_layers, DEFAULT_HYPERPARAMS.audioLayers),
    decoderLayers: pick(
      model.decoder_layers,
      DEFAULT_HYPERPARAMS.decoderLayers,
    ),
    heads: pick(model.heads, DEFAULT_HYPERPARAMS.heads),
    ffn: pick(model.ffn, DEFAULT_HYPERPARAMS.ffn),
    dropout: pick(model.dropout, DEFAULT_HYPERPARAMS.dropout),
  };
}

function numeric(value: string): number | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function assign(
  target: Record<string, unknown>,
  key: string,
  value: string,
): void {
  const parsed = numeric(value);
  if (parsed !== null) target[key] = parsed;
}

export function hyperparamsPayload(form: HyperparamForm): {
  data: Record<string, unknown>;
  model: Record<string, unknown>;
  train: Record<string, unknown>;
} {
  const data: Record<string, unknown> = {};
  const model: Record<string, unknown> = {};
  const train: Record<string, unknown> = {};
  assign(data, "window_bars", form.windowBars);
  assign(data, "stride_bars", form.strideBars);
  assign(data, "context_bars", form.contextBars);
  assign(data, "min_events", form.minEvents);
  data.max_windows_per_chart = numeric(form.maxWindows);
  assign(data, "spec_augment_prob", form.specProb);
  assign(model, "hidden", form.hidden);
  assign(model, "audio_layers", form.audioLayers);
  assign(model, "decoder_layers", form.decoderLayers);
  assign(model, "heads", form.heads);
  assign(model, "ffn", form.ffn);
  assign(model, "dropout", form.dropout);
  assign(train, "epochs", form.epochs);
  assign(train, "batch_size", form.batchSize);
  assign(train, "grad_accum", form.gradAccum);
  assign(train, "lr", form.lr);
  assign(train, "weight_decay", form.weightDecay);
  assign(train, "warmup_steps", form.warmupSteps);
  assign(train, "grad_clip", form.gradClip);
  assign(train, "label_smoothing", form.labelSmoothing);
  assign(train, "patience", form.patience);
  assign(train, "save_every", form.saveEvery);
  assign(train, "log_every", form.logEvery);
  train.amp = form.amp;
  return { data, model, train };
}

export function architectureQuery(form: HyperparamForm): URLSearchParams {
  const params = new URLSearchParams();
  const add = (key: string, value: string, fallback: string) => {
    const parsed = numeric(value);
    params.set(key, String(parsed === null ? fallback : parsed));
  };
  add("hidden", form.hidden, DEFAULT_HYPERPARAMS.hidden);
  add("audio_layers", form.audioLayers, DEFAULT_HYPERPARAMS.audioLayers);
  add("decoder_layers", form.decoderLayers, DEFAULT_HYPERPARAMS.decoderLayers);
  add("heads", form.heads, DEFAULT_HYPERPARAMS.heads);
  add("ffn", form.ffn, DEFAULT_HYPERPARAMS.ffn);
  add("dropout", form.dropout, DEFAULT_HYPERPARAMS.dropout);
  return params;
}
