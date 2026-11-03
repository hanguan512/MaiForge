import { useEffect, useState } from "react";
import { Box } from "@mui/material";
import { DISPLAY_FONT, NUMERIC_FONT } from "../theme";

export const PALETTES = [
  { bg: "#1c1038", a: "#ff5f9e", b: "#ffd166", ink: "#fff0f8" },
  { bg: "#04222c", a: "#3fd4e6", b: "#ff5f9e", ink: "#e9fdff" },
  { bg: "#2b1307", a: "#ffb347", b: "#ff5e62", ink: "#fff4e4" },
  { bg: "#0d1b2e", a: "#a78bfa", b: "#5dd39e", ink: "#eff0ff" },
  { bg: "#1d0a16", a: "#ff7ab8", b: "#3fd4e6", ink: "#fff0f7" },
  { bg: "#0f241b", a: "#5dd39e", b: "#ffd166", ink: "#f0fff7" },
];

export function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function starPoints(cx: number, cy: number, outer: number, inner: number): string {
  const points: string[] = [];
  for (let index = 0; index < 10; index += 1) {
    const radius = index % 2 === 0 ? outer : inner;
    const angle = -Math.PI / 2 + (index * Math.PI) / 5;
    points.push(`${(cx + Math.cos(angle) * radius).toFixed(2)},${(cy + Math.sin(angle) * radius).toFixed(2)}`);
  }
  return points.join(" ");
}

export function Motif({ variant, palette }: { variant: number; palette: (typeof PALETTES)[number] }) {
  if (variant === 0) {
    return (
      <>
        <circle cx="34" cy="38" r="27" fill={palette.a} />
        <circle cx="34" cy="38" r="18" fill={palette.bg} />
        <circle cx="34" cy="38" r="10" fill={palette.b} />
        <path d="M66 0 L100 0 L100 100 L52 100 Z" fill={palette.a} opacity="0.16" />
        <path d="M0 78 L100 46" stroke={palette.ink} strokeWidth="1" opacity="0.24" />
        <path d="M0 88 L100 56" stroke={palette.ink} strokeWidth="1" opacity="0.14" />
      </>
    );
  }
  if (variant === 1) {
    return (
      <>
        <g transform="rotate(-20 50 50)">
          <rect x="-10" y="8" width="120" height="13" fill={palette.a} />
          <rect x="-10" y="27" width="120" height="5" fill={palette.ink} opacity="0.45" />
          <rect x="-10" y="41" width="120" height="26" fill={palette.b} opacity="0.9" />
          <rect x="-10" y="74" width="120" height="5" fill={palette.ink} opacity="0.3" />
          <rect x="-10" y="85" width="120" height="9" fill={palette.a} opacity="0.5" />
        </g>
        <circle cx="74" cy="22" r="9" fill={palette.bg} stroke={palette.ink} strokeWidth="2" />
      </>
    );
  }
  if (variant === 2) {
    return (
      <>
        <circle cx="50" cy="50" r="40" fill="none" stroke={palette.a} strokeWidth="7" opacity="0.85" />
        <circle cx="50" cy="50" r="29" fill="none" stroke={palette.ink} strokeWidth="1" opacity="0.4" />
        <polygon points={starPoints(50, 50, 23, 10)} fill={palette.b} />
        <rect x="0" y="0" width="100" height="12" fill={palette.a} opacity="0.25" />
        <rect x="0" y="88" width="100" height="12" fill={palette.a} opacity="0.25" />
      </>
    );
  }
  return (
    <>
      <circle cx="78" cy="24" r="36" fill={palette.b} />
      <circle cx="78" cy="24" r="25" fill={palette.bg} />
      <circle cx="78" cy="24" r="15" fill={palette.a} />
      <path d="M0 100 L0 58 L38 20 L70 52 L100 22 L100 100 Z" fill={palette.a} opacity="0.22" />
      <path d="M0 100 L0 72 L36 36 L64 64 L100 28 L100 100 Z" fill={palette.bg} opacity="0.75" />
      <path d="M0 100 L0 86 L34 52 L60 78 L100 38 L100 100 Z" fill={palette.a} opacity="0.6" />
    </>
  );
}

export function coverTheme(title: string, id?: string) {
  const seed = hashString(`${title}:${id ?? ""}`);
  const palette = PALETTES[seed % PALETTES.length];
  const variant = Math.floor(seed / 7) % 4;
  const initial = (title.trim()[0] ?? "?").toUpperCase();
  const catalog = (id ?? "").replace(/[^a-z0-9]/gi, "").slice(-4).toUpperCase() || "MAI";
  return { seed, palette, variant, initial, catalog };
}

export default function Jacket({
  title,
  id,
  size = 56,
  radius = 3,
  src,
}: {
  title: string;
  id?: string;
  size?: number | string;
  radius?: number;
  src?: string | null;
}) {
  const { palette, variant, initial, catalog } = coverTheme(title, id);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);

  return (
    <Box
      sx={{
        width: size,
        height: size,
        flex: "none",
        borderRadius: `${radius}px`,
        overflow: "hidden",
        position: "relative",
        border: "1px solid rgba(255,255,255,0.12)",
        bgcolor: palette.bg,
      }}
    >
      <Box component="svg" viewBox="0 0 100 100" sx={{ width: "100%", height: "100%", display: "block" }}>
        <rect width="100" height="100" fill={palette.bg} />
        <Motif variant={variant} palette={palette} />
        <text
          x="7"
          y="86"
          fill={palette.ink}
          fontFamily={DISPLAY_FONT}
          fontSize="34"
          opacity="0.96"
        >
          {initial}
        </text>
        <text
          x="95"
          y="10"
          fill={palette.ink}
          fontFamily={NUMERIC_FONT}
          fontSize="5.4"
          textAnchor="end"
          opacity="0.7"
          letterSpacing="0.6"
        >
          {catalog}
        </text>
      </Box>
      {src && !failed ? (
        <Box
          component="img"
          src={src}
          alt=""
          loading="lazy"
          draggable={false}
          onError={() => setFailed(true)}
          sx={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block",
          }}
        />
      ) : null}
    </Box>
  );
}
