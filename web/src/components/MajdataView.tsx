import { useEffect, useRef, useState } from "react";
import { Alert, Box, CircularProgress } from "@mui/material";

interface MajdataViewProps {
  /** URL of the vendored MajdataView WebGL page. */
  src: string;
  title?: string;
  /** Difficulty chosen from the control bar rendered inside the player. */
  onDifficultyChange?: (index: number) => void;
  /** Seek target in seconds, chosen from the player's progress bar. */
  onSeek?: (seconds: number) => void;
}

interface BridgeMessage {
  type?: string;
  message?: string;
  index?: number;
  seconds?: number;
}

/**
 * Embeds the real MajdataView WebGL build (not a reimplementation).
 *
 * The iframe page fetches the compiled maidata, an mp3 track and a background
 * from the backend, then feeds them into MajdataView's own upload inputs.
 * Playback controls live inside the iframe so clicks stay in the same frame
 * as the Unity audio context (browsers gate audio on user activation).
 * It reports back through postMessage for ready/error and control changes.
 */
export default function MajdataView({
  src,
  title,
  onDifficultyChange,
  onSeek,
}: MajdataViewProps) {
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const callbacks = useRef({ onDifficultyChange, onSeek });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    callbacks.current = { onDifficultyChange, onSeek };
  }, [onDifficultyChange, onSeek]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const onMessage = (event: MessageEvent<BridgeMessage>) => {
      if (event.source !== frameRef.current?.contentWindow) return;
      const data = event.data;
      if (!data || typeof data.type !== "string") return;
      if (data.type === "mainote:majdataview-ready") {
        setLoading(false);
        setError(null);
      } else if (data.type === "mainote:majdataview-error") {
        setLoading(false);
        setError(data.message || "MajdataView 加载失败");
      } else if (
        data.type === "mainote:majdataview-difficulty" &&
        typeof data.index === "number"
      ) {
        callbacks.current.onDifficultyChange?.(data.index);
      } else if (
        data.type === "mainote:majdataview-seek" &&
        typeof data.seconds === "number"
      ) {
        callbacks.current.onSeek?.(data.seconds);
      }
    };
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [src]);

  return (
    <Box
      sx={{
        position: "relative",
        width: "100%",
        aspectRatio: "960 / 600",
        bgcolor: "#07080d",
        border: 1,
        borderColor: "divider",
        borderRadius: "4px",
        overflow: "hidden",
      }}
    >
      <Box
        component="iframe"
        ref={frameRef}
        src={src}
        title={title ?? "MajdataView"}
        allow="autoplay; fullscreen; gamepad; clipboard-write"
        sx={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          border: 0,
          display: "block",
        }}
      />
      {loading && !error ? (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            bgcolor: "rgba(7,8,13,0.5)",
            pointerEvents: "none",
          }}
        >
          <CircularProgress size={26} />
        </Box>
      ) : null}
      {error ? (
        <Box
          sx={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            p: 3,
            bgcolor: "rgba(7,8,13,0.92)",
          }}
        >
          <Alert severity="error">{error}</Alert>
        </Box>
      ) : null}
    </Box>
  );
}
