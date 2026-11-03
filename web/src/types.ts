export interface SongChartBrief {
  index: number;
  difficulty: string;
  level: string;
  designer?: string;
  notes: number;
  density?: number | null;
  duration?: number | null;
  counts?: Record<string, number>;
  bpm_min?: number;
  bpm_max?: number;
  alignment?: { confidence?: number | null } | null;
}

export interface SongSummary {
  root: string;
  id: string;
  title: string;
  artist: string;
  bpm: number | null;
  has_audio: boolean;
  has_ir: boolean;
  cover_url?: string | null;
  chart_count: number;
  charts: SongChartBrief[];
  warnings: string[];
  errors: string[];
}

export interface SongDetail {
  root: string;
  id: string;
  title: string;
  artist: string;
  metadata: Record<string, string>;
  has_audio: boolean;
  audio_url: string | null;
  cover_url?: string | null;
  timing: Record<string, unknown> | null;
  alignment: { min_confidence?: number | null; max_confidence?: number | null };
  charts: SongChartBrief[];
  warnings: string[];
  errors: string[];
  ir_available: boolean;
}

export interface SlideSegment {
  shape: string;
  start: string;
  end: string;
  via: string | null;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
}

export interface ChartEvent {
  tick: number;
  seconds: number;
  end_seconds: number;
  kind: string;
  position: string;
  flags: string[];
  chain_type: string;
  segments: SlideSegment[];
  bar: number;
}

export interface ChartOption {
  index: number;
  difficulty: string;
  level: string;
}

export interface ChartPayload {
  source: {
    kind: "song" | "file";
    root?: string;
    song_id?: string;
    path?: string;
  };
  title: string;
  artist: string;
  index: number;
  difficulty: string;
  level: string;
  designer: string;
  initial_bpm: number;
  first: number;
  end_tick: number;
  duration_seconds: number;
  audio_duration_seconds?: number | null;
  charts: ChartOption[];
  bpm_changes: { tick: number; bpm: number; seconds: number }[];
  events: ChartEvent[];
  stats: Record<string, unknown>;
  validation: { ok: boolean; errors: string[]; warnings: string[] };
  audio_url: string | null;
  cover_url?: string | null;
  maidata: string;
}

export interface RootInfo {
  value: string;
  label: string;
  exists: boolean;
  song_count: number;
}

export interface OptionItem {
  value: string;
  label: string;
  group?: string;
  name?: string;
  recommended?: boolean;
  mtime?: number;
}

export interface RunSummary {
  name: string;
  path: string;
  best_val_loss: number | null;
  best_note_score: number | null;
  last_epoch: number | null;
  last_step: number | null;
  checkpoint_count: number;
  checkpoints: string[];
  mtime: number;
  has_metrics: boolean;
}

export interface RunDetail extends RunSummary {
  metrics: MetricRow[];
  files: { name: string; path: string; size: number; mtime: number }[];
  config: Record<string, unknown> | null;
}

export interface MetricRow {
  [key: string]: unknown;
  epoch?: number;
  step?: number;
  train_loss?: number | null;
  train_token_ce?: number | null;
  val_loss?: number | null;
  train_acc?: number | null;
  val_acc?: number | null;
  note_eval?: {
    note_score?: number;
    onset_f1?: number;
    position_accuracy?: number;
    type_accuracy?: number;
  } | null;
}

export interface JobInfo {
  id: string;
  kind: string;
  status: "running" | "stopping" | "completed" | "failed" | "stopped";
  started_at: number;
  finished_at: number | null;
  elapsed: number;
  returncode: number | null;
  output_dir: string;
  log_path: string;
  command: string[];
  extra: Record<string, unknown>;
}

export interface JobSnapshot {
  active: boolean;
  job: JobInfo | null;
  history?: JobInfo[];
  log?: string[];
  metrics?: MetricRow[];
  result_maidata?: string;
}

export interface Options {
  configs: OptionItem[];
  checkpoints: OptionItem[];
  audio: OptionItem[];
  timing: OptionItem[];
  features: OptionItem[];
  runs: RunSummary[];
  gpu: GpuStatus | null;
  defaults: {
    config: string;
    checkpoint: string;
    train_output: string;
    infer_output: string;
  };
  active_job: JobSnapshot;
}

export interface TrainSongChart {
  index: number | null;
  level: string | null;
  difficulty: string | null;
}

export interface TrainSong {
  id: string;
  title: string;
  artist: string;
  chart_count: number;
  charts: TrainSongChart[];
  has_audio: boolean;
  has_ir: boolean;
  sample_count: number;
  sample_ready: boolean;
  features_ready: boolean;
  trainable: boolean;
}

export interface TrainData {
  root: string;
  roots: RootInfo[];
  songs: TrainSong[];
  manifest: { path: string; exists: boolean; rows: number };
  samples_dir: string;
  features_dir: string;
  summary: {
    songs: number;
    sample_ready: number;
    features_ready: number;
    trainable: number;
  };
}

export interface TrainConfigResponse {
  path: string;
  config: Record<string, unknown>;
}

export interface ModelLayer {
  id: string;
  name: string;
  kind: string;
  shape: string;
  detail?: string;
  params: number;
}

export interface ModelLayerGroup {
  id: string;
  label: string;
  column: "encoder" | "decoder";
  layers: ModelLayer[];
}

export interface ModelArchitecture {
  config: Record<string, number | boolean>;
  vocab_size: number;
  total_params: number;
  mel_fps: number;
  encoded_fps: number;
  groups: ModelLayerGroup[];
}

export interface GpuStatus {
  name: string;
  used_mb: number;
  total_mb: number;
  util: number;
  temp: number;
}

export interface Overview {
  songs: number;
  charts: number;
  roots: RootInfo[];
  runs: number;
  checkpoints: number;
  configs: number;
  gpu: GpuStatus | null;
  active_job: JobSnapshot;
}

export interface DatasetSongChart {
  index: number;
  difficulty: string;
  level: string;
  notes: number;
}

export interface DatasetSong {
  id: string;
  title: string;
  artist: string;
  has_audio: boolean;
  cover_url?: string | null;
  chart_count: number;
  charts: DatasetSongChart[];
  updated_at?: number | null;
  sample_ready: boolean;
  features_ready: boolean;
}

export interface DatasetSummary {
  raw_root: string;
  processed_root: string;
  raw_count: number;
  processed_count: number;
  songs: DatasetSong[];
}

export interface DatasetImportItem {
  file: string;
  ok: boolean;
  song_id: string | null;
  title: string | null;
  artist: string | null;
  chart_count: number;
  has_audio: boolean;
  raw: string | null;
  processed: string | null;
  warnings: string[];
  errors: string[];
}

export interface DatasetImportResponse {
  ok: boolean;
  align: boolean;
  results: DatasetImportItem[];
}

export interface WmcLevel {
  raw: string;
  num: number;
}

export interface WmcChart {
  shortid: string;
  title: string;
  displayTitle?: string;
  artist: string;
  type: string;
  version: string;
  versionid?: number;
  versionFolder?: string;
  folderName?: string;
  genre?: string;
  aliases?: string[];
  bpm: number | null;
  levels: Record<string, WmcLevel>;
  designers?: Record<string, string>;
  hasVideo?: boolean;
  image?: string;
  cover_url?: string | null;
  audio?: string;
  downloads?: number;
  page?: string;
  filesIndex?: string;
}

export interface WmcSearchResponse {
  total: number;
  matched: number;
  page: number;
  pages: number;
  limit: number;
  count: number;
  charts: WmcChart[];
  available?: {
    versions?: Record<string, string>;
    genres?: string[];
  };
  source?: string;
}

export interface WmcImportResult {
  shortid: string;
  ok: boolean;
  skipped: boolean;
  title: string | null;
  song_id?: string | null;
  file?: string;
  chart_count?: number;
  has_audio?: boolean;
  has_background?: boolean;
  errors: string[];
  warnings: string[];
  page?: string;
  version?: string;
  bpm?: number | null;
  levels?: Record<string, WmcLevel>;
}

export type WmcJobStatus = "running" | "stopping" | "done" | "cancelled";

export interface WmcImportJob {
  id: string;
  status: WmcJobStatus;
  total: number;
  completed: number;
  current: string | null;
  include_video: boolean;
  align: boolean;
  force: boolean;
  results: WmcImportResult[];
  error: string | null;
  started_at: number;
  finished_at: number | null;
}

export interface WmcJobsResponse {
  jobs: WmcImportJob[];
}
