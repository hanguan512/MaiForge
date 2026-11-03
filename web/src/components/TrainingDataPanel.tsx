import {
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControl,
  FormControlLabel,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Switch,
  Typography,
} from "@mui/material";
import { Refresh, Tune } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import { NUMERIC_FONT, PALETTE } from "../theme";
import type { TrainData, TrainSong } from "../types";

function ReadyTag({
  label,
  ready,
  detail,
}: {
  label: string;
  ready: boolean;
  detail?: string;
}) {
  return (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.5,
        px: 0.65,
        py: 0.2,
        border: 1,
        borderStyle: ready ? "solid" : "dashed",
        borderColor: ready ? alpha(PALETTE.green, 0.55) : "divider",
        borderRadius: "2px",
        color: ready ? PALETTE.green : "text.secondary",
        fontSize: 10,
        fontWeight: 600,
        lineHeight: 1.3,
        whiteSpace: "nowrap",
      }}
      title={detail}
    >
      <Box
        component="span"
        sx={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          bgcolor: ready ? PALETTE.green : "text.disabled",
        }}
      />
      {label}
    </Box>
  );
}

function SongRow({
  song,
  checked,
  onToggle,
}: {
  song: TrainSong;
  checked: boolean;
  onToggle: () => void;
}) {
  return (
    <Box
      onClick={onToggle}
      sx={{
        display: "grid",
        gridTemplateColumns: "auto minmax(0, 1fr) auto",
        gap: 1,
        alignItems: "center",
        px: 1,
        py: 0.7,
        borderBottom: 1,
        borderColor: "divider",
        cursor: "pointer",
        bgcolor: checked ? alpha(PALETTE.violet, 0.06) : "transparent",
        "&:hover": { bgcolor: alpha(PALETTE.text, 0.035) },
      }}
    >
      <Checkbox
        size="small"
        checked={checked}
        onChange={onToggle}
        sx={{ p: 0.4 }}
        onClick={(event) => event.stopPropagation()}
      />
      <Box sx={{ minWidth: 0 }}>
        <Typography
          variant="body2"
          sx={{
            fontWeight: 600,
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {song.title}
        </Typography>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", lineHeight: 1.4 }}
        >
          <Box
            component="span"
            sx={{ fontFamily: "monospace", fontSize: 11, mr: 0.75 }}
          >
            {song.id}
          </Box>
          {song.chart_count} 个难度
          {song.artist ? ` · ${song.artist}` : ""}
        </Typography>
      </Box>
      <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
        <ReadyTag label="音频" ready={song.has_audio} />
        <ReadyTag
          label="样本"
          ready={song.sample_ready}
          detail={song.sample_ready ? `${song.sample_count} 条` : "尚未导出"}
        />
        <ReadyTag label="特征" ready={song.features_ready} />
      </Stack>
    </Box>
  );
}

export default function TrainingDataPanel({
  data,
  loading,
  root,
  selected,
  busy,
  overwrite,
  onRootChange,
  onToggle,
  onSelectReady,
  onClear,
  onPrepare,
  onRefresh,
  onOverwriteChange,
}: {
  data: TrainData | null;
  loading: boolean;
  root: string;
  selected: string[];
  busy: boolean;
  overwrite: boolean;
  onRootChange: (root: string) => void;
  onToggle: (id: string) => void;
  onSelectReady: () => void;
  onClear: () => void;
  onPrepare: () => void;
  onRefresh: () => void;
  onOverwriteChange: (value: boolean) => void;
}) {
  const selectedSet = new Set(selected);
  const songs = data?.songs ?? [];
  const selectedSongs = songs.filter((song) => selectedSet.has(song.id));
  const missing = selectedSongs.filter(
    (song) => !song.sample_ready || !song.features_ready,
  );
  const readyCount = songs.filter((song) => song.trainable).length;

  return (
    <Box
      sx={{
        border: 1,
        borderColor: "divider",
        borderRadius: "4px",
        bgcolor: "background.paper",
        p: { xs: 2, md: 2.5 },
      }}
    >
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1.5}
        alignItems={{ sm: "center" }}
        justifyContent="space-between"
      >
        <Box>
          <Typography variant="h6" sx={{ fontSize: 18 }}>
            训练数据
          </Typography>
          <Typography variant="caption" color="text.secondary">
            从数据集选择歌曲；按歌曲切分 train/val，同一首歌不会跨集。至少 2 首时按比例留出验证歌，只选 1 首则不做 note-level 验证。所选歌曲需要先导出样本与 log-mel 特征。
          </Typography>
        </Box>
        <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap" useFlexGap>
          <FormControl size="small" sx={{ minWidth: 170 }}>
            <InputLabel>数据目录</InputLabel>
            <Select
              label="数据目录"
              value={root || data?.root || ""}
              onChange={(event) => onRootChange(event.target.value)}
              disabled={busy}
            >
              {(data?.roots ?? []).map((item) => (
                <MenuItem key={item.value} value={item.value} disabled={!item.exists}>
                  {item.label} · {item.song_count} 首
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Button
            size="small"
            startIcon={<Refresh />}
            onClick={onRefresh}
            disabled={loading || busy}
          >
            刷新
          </Button>
        </Stack>
      </Stack>

      <Stack
        direction="row"
        spacing={1.5}
        alignItems="center"
        flexWrap="wrap"
        useFlexGap
        sx={{ mt: 1.75 }}
      >
        <Typography
          sx={{ fontFamily: NUMERIC_FONT, fontSize: 12, fontVariantNumeric: "tabular-nums" }}
        >
          已选 {selected.length} / {songs.length} 首
        </Typography>
        <Typography variant="caption" color="text.secondary">
          可直接训练 {data?.summary.trainable ?? 0} 首 · 共 {readyCount} 首就绪
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <Button size="small" onClick={onSelectReady} disabled={busy || !songs.length}>
          全选已就绪
        </Button>
        <Button size="small" onClick={onClear} disabled={busy || !selected.length}>
          清空
        </Button>
      </Stack>

      {loading ? (
        <Box sx={{ py: 5, textAlign: "center" }}>
          <CircularProgress size={24} />
        </Box>
      ) : songs.length ? (
        <Box
          sx={{
            mt: 1,
            maxHeight: 318,
            overflowY: "auto",
            border: 1,
            borderColor: "divider",
            borderRadius: "3px",
          }}
        >
          {songs.map((song) => (
            <SongRow
              key={song.id}
              song={song}
              checked={selectedSet.has(song.id)}
              onToggle={() => onToggle(song.id)}
            />
          ))}
        </Box>
      ) : (
        <Box
          sx={{
            mt: 1.25,
            border: 1,
            borderStyle: "dashed",
            borderColor: "divider",
            borderRadius: "3px",
            py: 4,
            textAlign: "center",
          }}
        >
          <Typography variant="body2" color="text.secondary">
            当前数据目录还没有歌曲。先在「数据集」页下载或导入谱面。
          </Typography>
        </Box>
      )}

      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1.5}
        alignItems={{ sm: "center" }}
        sx={{ mt: 1.75 }}
      >
        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={overwrite}
              onChange={(event) => onOverwriteChange(event.target.checked)}
              disabled={busy}
            />
          }
          label={<Typography variant="caption">覆盖已有特征</Typography>}
        />
        <Typography variant="caption" color="text.secondary">
          {missing.length
            ? `所选歌曲中有 ${missing.length} 首缺少样本或特征`
            : selected.length
              ? "所选歌曲的数据已就绪"
              : "选择歌曲后可以准备数据"}
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <Button
          variant="outlined"
          size="small"
          startIcon={<Tune />}
          onClick={onPrepare}
          disabled={busy || !selected.length}
        >
          准备所选数据
        </Button>
      </Stack>
    </Box>
  );
}
