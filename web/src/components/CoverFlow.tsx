import { Box, Typography } from "@mui/material";
import { ChevronLeft, ChevronRight } from "@mui/icons-material";
import CoverArt from "./CoverArt";
import DifficultyChip from "./DifficultyChip";
import { DISPLAY_FONT, NUMERIC_FONT, PALETTE } from "../theme";
import type { SongSummary } from "../types";

const OFFSETS = [-4, -3, -2, -1, 0, 1, 2, 3, 4];
// SongSelect.cs: covers are spaced 3.3 units apart, centre scale 10.8, side 7.8.
const CARD_SHIFT = 3.3 / 10.8;
const SIDE_SCALE = 7.8 / 10.8;

function mod(value: number, size: number): number {
  return ((value % size) + size) % size;
}

function difficultySummary(song: SongSummary): string {
  if (!song.charts.length) return "无谱面";
  return song.charts
    .map((chart) => `${chart.difficulty.slice(0, 3).toUpperCase()} ${chart.level}`)
    .join(" · ");
}

export default function CoverFlow({
  songs,
  cursor,
  chartIndex,
  dim,
  reducedMotion,
  onMove,
  onView,
  onChartChange,
}: {
  songs: SongSummary[];
  cursor: number;
  chartIndex: number | null;
  dim: number;
  reducedMotion: boolean;
  onMove: (delta: number) => void;
  onView: () => void;
  onChartChange: (index: number) => void;
}) {
  const count = songs.length;
  if (count === 0) return null;
  const active = songs[mod(cursor, count)];
  const activeChart =
    active.charts.find((chart) => chart.index === chartIndex) ?? active.charts.at(-1) ?? null;
  const duration = !reducedMotion
    ? "transform 500ms cubic-bezier(0.25, 0.46, 0.45, 0.94), opacity 500ms cubic-bezier(0.25, 0.46, 0.45, 0.94), box-shadow 500ms ease"
    : "none";

  return (
    <>
      <Box
        sx={{
          position: "absolute",
          inset: { xs: "50px 0 0", sm: 0 },
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: { xs: 3, sm: 4 },
          pb: { xs: 5, sm: 3 },
          "--cover": "clamp(304px, min(64vh, 96vw), 620px)",
        }}
      >
        <Box
          sx={{
            position: "relative",
            width: "var(--cover)",
            height: "var(--cover)",
            flex: "none",
          }}
        >
        {OFFSETS.map((offset) => {
          const virtual = cursor + offset;
          const song = songs[mod(virtual, count)];
          const center = offset === 0;
          const distance = Math.abs(offset);
          const visible = distance <= 2;
          const card = (
            <Box
              key={virtual}
              role={center ? undefined : "button"}
              tabIndex={center || !visible ? -1 : 0}
              aria-label={center ? undefined : `选择 ${song.title}`}
              aria-current={center ? "true" : undefined}
              onClick={center ? undefined : () => onMove(offset)}
              onKeyDown={
                center
                  ? undefined
                  : (event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onMove(offset);
                      }
                    }
              }
              sx={{
                position: "absolute",
                left: "50%",
                top: "50%",
                width: "var(--cover)",
                height: "var(--cover)",
                transform: `translate(-50%, -50%) translateX(calc(${offset} * var(--cover) * ${CARD_SHIFT})) scale(${
                  center ? 1 : SIDE_SCALE
                })`,
                zIndex: 100 - distance,
                opacity: visible ? 1 : 0,
                cursor: center ? "default" : "pointer",
                borderRadius: center ? "3px" : "2px",
                overflow: "hidden",
                border: "1px solid",
                borderColor: center ? "rgba(255,255,255,0.26)" : "rgba(255,255,255,0.12)",
                boxShadow: center
                  ? "0 38px 90px rgba(43, 15, 72, 0.45), 0 4px 16px rgba(43, 15, 72, 0.28)"
                  : "0 18px 42px rgba(43, 15, 72, 0.28)",
                transition: duration,
                willChange: "transform, opacity",
                outline: "none",
                "&:focus-visible": {
                  outline: `2px solid ${PALETTE.aqua}`,
                  outlineOffset: 3,
                },
                "&:hover .cover-art": { filter: center ? "none" : "brightness(1.08)" },
              }}
            >
              <Box
                className="cover-art"
                sx={{
                  position: "absolute",
                  inset: 0,
                  transition: "filter 180ms ease",
                }}
              >
                <CoverArt title={song.title} id={song.id} src={song.cover_url} />
              </Box>

              <Box
                sx={{
                  position: "absolute",
                  inset: 0,
                  bgcolor: "#0b0714",
                  opacity: Math.min(0.78, dim * 0.8),
                  pointerEvents: "none",
                  transition: "opacity 200ms ease",
                }}
              />
              <Box
                sx={{
                  position: "absolute",
                  inset: 0,
                  bgcolor: "#0b0714",
                  opacity: center ? 0 : Math.min(0.55, 0.16 + distance * 0.13),
                  pointerEvents: "none",
                  transition: duration,
                }}
              />

              <Box
                sx={{
                  position: "absolute",
                  inset: 0,
                  background:
                    "linear-gradient(180deg, rgba(8,5,17,0) 38%, rgba(8,5,17,0.34) 61%, rgba(8,5,17,0.9) 100%)",
                  opacity: center ? 1 : 0,
                  pointerEvents: "none",
                  transition: duration,
                }}
              />

              <Box
                sx={{
                  position: "absolute",
                  inset: "auto 0 0",
                  px: { xs: 2.5, sm: 4 },
                  pb: { xs: 2.5, sm: 3.5 },
                  color: "#ffffff",
                  opacity: center ? 1 : 0,
                  pointerEvents: center ? "auto" : "none",
                  transition: duration,
                }}
                aria-hidden={!center}
              >
                <Typography
                  sx={{
                    fontFamily: DISPLAY_FONT,
                    fontSize: "clamp(23px, calc(var(--cover) * 0.075), 44px)",
                    lineHeight: 1.08,
                    textShadow: "0 2px 20px rgba(12,4,26,0.5)",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                    wordBreak: "break-word",
                  }}
                >
                  {song.title}
                </Typography>
                <Typography
                  sx={{
                    mt: 0.6,
                    fontSize: { xs: 13, sm: 15 },
                    color: "rgba(255,255,255,0.68)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {song.artist || "未知作者"}
                </Typography>

                <Box
                  sx={{
                    display: "flex",
                    alignItems: "center",
                    gap: 1.25,
                    mt: 1.75,
                    flexWrap: "wrap",
                    rowGap: 0.75,
                  }}
                >
                  {active.charts.map((chart) => {
                    const selected = activeChart?.index === chart.index;
                    return (
                      <Box
                        key={chart.index}
                        component="button"
                        type="button"
                        tabIndex={center ? 0 : -1}
                        aria-pressed={selected}
                        aria-label={`${chart.difficulty} ${chart.level}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onChartChange(chart.index);
                        }}
                        sx={{
                          p: 0,
                          border: "1px solid",
                          borderColor: selected ? "rgba(255,255,255,0.9)" : "rgba(255,255,255,0.1)",
                          bgcolor: "transparent",
                          borderRadius: "2px",
                          cursor: "pointer",
                          opacity: selected ? 1 : 0.55,
                          transition: "opacity 140ms ease, border-color 140ms ease",
                          "&:hover": { opacity: 1 },
                          "&:focus-visible": {
                            outline: `2px solid ${PALETTE.aqua}`,
                            outlineOffset: 2,
                          },
                        }}
                      >
                        <DifficultyChip difficulty={chart.difficulty} level={chart.level} />
                      </Box>
                    );
                  })}

                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "baseline",
                      gap: 0.5,
                      ml: { xs: 0, sm: "auto" },
                      color: "rgba(255,255,255,0.62)",
                    }}
                  >
                    <Box
                      component="span"
                      sx={{
                        fontFamily: NUMERIC_FONT,
                        fontWeight: 700,
                        fontSize: 15,
                        color: "#ffffff",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {song.bpm ? song.bpm.toFixed(1) : "--"}
                    </Box>
                    <Box component="span" sx={{ fontSize: 11 }}>
                      BPM
                    </Box>
                  </Box>
                </Box>

                <Typography
                  sx={{
                    mt: 1.1,
                    fontSize: 11,
                    color: "rgba(255,255,255,0.42)",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {activeChart
                    ? `${activeChart.notes} 音符${activeChart.designer ? ` / ${activeChart.designer}` : ""}`
                    : difficultySummary(song)}
                  {!song.has_audio ? " / 无音频" : ""}
                </Typography>
              </Box>
            </Box>
          );
          return card;
        })}
        </Box>

      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: { xs: 1.5, sm: 2.25 },
          zIndex: 200,
        }}
      >
        <Box
          component="button"
          type="button"
          aria-label="上一首"
          onClick={() => onMove(-1)}
          sx={{
            width: { xs: 42, sm: 48 },
            height: { xs: 42, sm: 48 },
            display: "grid",
            placeItems: "center",
            borderRadius: "50%",
            border: "1px solid rgba(255,255,255,0.55)",
            bgcolor: "rgba(255,255,255,0.28)",
            color: "#35224f",
            cursor: "pointer",
            backdropFilter: "blur(10px)",
            transition: "background-color 140ms ease, transform 140ms ease",
            "&:hover": { bgcolor: "rgba(255,255,255,0.48)" },
            "&:active": { transform: "scale(0.94)" },
            "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: 2 },
          }}
        >
          <ChevronLeft sx={{ fontSize: { xs: 24, sm: 28 } }} />
        </Box>

        <Box
          component="button"
          type="button"
          onClick={onView}
          disabled={!activeChart}
          sx={{
            minWidth: { xs: 168, sm: 196 },
            height: { xs: 50, sm: 56 },
            px: 4,
            borderRadius: "999px",
            border: "1px solid rgba(255,255,255,0.28)",
            bgcolor: "#6f43cf",
            color: "#ffffff",
            fontFamily: DISPLAY_FONT,
            fontSize: { xs: 18, sm: 20 },
            letterSpacing: "0.08em",
            cursor: activeChart ? "pointer" : "not-allowed",
            opacity: activeChart ? 1 : 0.45,
            boxShadow: "0 14px 34px rgba(60,24,120,0.36)",
            transition: "background-color 140ms ease, transform 140ms ease, box-shadow 140ms ease",
            "&:hover": {
              bgcolor: activeChart ? "#5c34b4" : "#6f43cf",
              boxShadow: "0 18px 40px rgba(60,24,120,0.44)",
            },
            "&:active": { transform: "translateY(1px)" },
            "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: 3 },
          }}
        >
          View
        </Box>

        <Box
          component="button"
          type="button"
          aria-label="下一首"
          onClick={() => onMove(1)}
          sx={{
            width: { xs: 42, sm: 48 },
            height: { xs: 42, sm: 48 },
            display: "grid",
            placeItems: "center",
            borderRadius: "50%",
            border: "1px solid rgba(255,255,255,0.55)",
            bgcolor: "rgba(255,255,255,0.28)",
            color: "#35224f",
            cursor: "pointer",
            backdropFilter: "blur(10px)",
            transition: "background-color 140ms ease, transform 140ms ease",
            "&:hover": { bgcolor: "rgba(255,255,255,0.48)" },
            "&:active": { transform: "scale(0.94)" },
            "&:focus-visible": { outline: `2px solid ${PALETTE.aqua}`, outlineOffset: 2 },
          }}
        >
          <ChevronRight sx={{ fontSize: { xs: 24, sm: 28 } }} />
        </Box>
      </Box>
      </Box>
    </>
  );
}
