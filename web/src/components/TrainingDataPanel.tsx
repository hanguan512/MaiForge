import { useMemo, useState } from "react";
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
  TextField,
  Typography,
} from "@mui/material";
import { Refresh, Search, Tune } from "@mui/icons-material";
import { alpha } from "@mui/material/styles";
import ReadyMark from "./ReadyMark";
import { NUMERIC_FONT, PALETTE } from "../theme";
import type { TrainData, TrainSong } from "../types";

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
        position: "relative",
        display: "grid",
        gridTemplateColumns: "auto minmax(0, 1fr) auto",
        gap: 1.25,
        alignItems: "center",
        px: 1.25,
        py: 0.7,
        borderBottom: 1,
        borderColor: "divider",
        cursor: "pointer",
        bgcolor: checked ? alpha(PALETTE.pink, 0.05) : "transparent",
        "&:hover": { bgcolor: checked ? alpha(PALETTE.pink, 0.08) : alpha(PALETTE.text, 0.03) },
        "&::before": {
          content: '""',
          position: "absolute",
          left: 0,
          top: 5,
          bottom: 5,
          width: 2,
          bgcolor: "primary.main",
          opacity: checked ? 1 : 0,
        },
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
          <Box component="span" sx={{ color: "text.disabled", mr: 0.75 }}>
            {song.id}
          </Box>
          {song.chart_count} 个难度
          {song.artist ? ` · ${song.artist}` : ""}
        </Typography>
      </Box>
      <Stack direction="row" spacing={1.25} flexWrap="wrap" useFlexGap sx={{ justifyContent: "flex-end" }}>
        <ReadyMark label="音频" ready={song.has_audio} />
        <ReadyMark
          label="样本"
          ready={song.sample_ready}
          title={song.sample_ready ? `${song.sample_count} 条` : "尚未导出"}
        />
        <ReadyMark label="特征" ready={song.features_ready} />
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
  const [query, setQuery] = useState("");
  const selectedSet = new Set(selected);
  const songs = data?.songs ?? [];
  const selectedSongs = songs.filter((song) => selectedSet.has(song.id));
  const missing = selectedSongs.filter(
    (song) => !song.sample_ready || !song.features_ready,
  );
  const visibleSongs = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return songs;
    return songs.filter(
      (song) =>
        song.title.toLowerCase().includes(needle) ||
        song.artist.toLowerCase().includes(needle) ||
        song.id.toLowerCase().includes(needle),
    );
  }, [songs, query]);

  return (
    <Stack spacing={1.5}>
      <Stack
        direction={{ xs: "column", sm: "row" }}
        spacing={1.5}
        alignItems={{ sm: "center" }}
      >
        <TextField
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="筛选歌曲、作者或 ID"
          sx={{ width: { xs: "100%", sm: 260 } }}
          InputProps={{
            startAdornment: <Search sx={{ fontSize: 16, mr: 0.75, color: "text.disabled" }} />,
          }}
        />
        <Typography
          sx={{ fontFamily: NUMERIC_FONT, fontSize: 12, fontVariantNumeric: "tabular-nums" }}
        >
          已选 {selected.length} / {songs.length} 首
        </Typography>
        <Typography variant="caption" color="text.secondary">
          可直接训练 {data?.summary.trainable ?? 0} 首
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        <Button size="small" onClick={onSelectReady} disabled={busy || !songs.length}>
          全选已就绪
        </Button>
        <Button size="small" onClick={onClear} disabled={busy || !selected.length}>
          清空
        </Button>
      </Stack>

      <Typography variant="caption" color="text.secondary">
        按歌曲切分 train / val，同一首歌不会跨集。至少 2 首时按比例留出验证歌，只选 1
        首则不做 note-level 验证。所选歌曲需要先导出样本与 log-mel 特征。
      </Typography>

      <Stack direction="row" spacing={1.5} alignItems="center" flexWrap="wrap" useFlexGap>
        <FormControl size="small" sx={{ minWidth: 210 }}>
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
          startIcon={<Refresh sx={{ fontSize: 16 }} />}
          onClick={onRefresh}
          disabled={loading || busy}
        >
          刷新
        </Button>
        <Box sx={{ flexGrow: 1 }} />
        <Typography variant="caption" color="text.secondary">
          {data ? `${data.samples_dir} · ${data.features_dir}` : ""}
        </Typography>
      </Stack>

      {loading ? (
        <Box sx={{ py: 5, textAlign: "center" }}>
          <CircularProgress size={22} />
        </Box>
      ) : visibleSongs.length ? (
        <Box
          sx={{
            maxHeight: 340,
            overflowY: "auto",
            borderTop: 1,
            borderBottom: 1,
            borderColor: "divider",
          }}
        >
          {visibleSongs.map((song) => (
            <SongRow
              key={song.id}
              song={song}
              checked={selectedSet.has(song.id)}
              onToggle={() => onToggle(song.id)}
            />
          ))}
        </Box>
      ) : songs.length ? (
        <Box sx={{ py: 4, textAlign: "center" }}>
          <Typography variant="body2" color="text.secondary">
            没有匹配的歌曲。
          </Typography>
        </Box>
      ) : (
        <Box
          sx={{
            border: 1,
            borderStyle: "dashed",
            borderColor: "divider",
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
        <Typography
          variant="caption"
          color={missing.length ? PALETTE.gold : "text.secondary"}
        >
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
          startIcon={<Tune sx={{ fontSize: 16 }} />}
          onClick={onPrepare}
          disabled={busy || !selected.length}
        >
          准备所选数据
        </Button>
      </Stack>
    </Stack>
  );
}
