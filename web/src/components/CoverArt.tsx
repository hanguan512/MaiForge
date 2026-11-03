import { useEffect, useId, useState } from "react";
import { Box } from "@mui/material";
import { DISPLAY_FONT, NUMERIC_FONT } from "../theme";
import { Motif, coverTheme } from "./Jacket";

/**
 * Full-bleed album artwork for the song carousel.
 *
 * The local library has no jacket images, so the sleeve is generated from the
 * same deterministic palette/motif system as the small `Jacket` chip. Everything
 * is drawn in a 1080x1080 space so it stays crisp at the centre-cover size.
 */
export default function CoverArt({
  title,
  id,
  src,
}: {
  title: string;
  id?: string;
  src?: string | null;
}) {
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const { seed, palette, variant, initial, catalog } = coverTheme(title, id);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
  }, [src]);
  const barcode = Array.from({ length: 30 }, (_, index) => {
    const value = (seed >>> (index % 24)) ^ Math.imul(index + 3, 2654435761);
    return {
      x: index * 11 + (index % 3),
      width: 2 + (Math.abs(value) % 3) * 1.6,
      height: 38 + (Math.abs(value) % 5) * 9,
    };
  });

  return (
    <Box sx={{ position: "relative", width: "100%", height: "100%" }}>
      <Box
        component="svg"
        viewBox="0 0 1080 1080"
        aria-hidden="true"
        sx={{ display: "block", width: "100%", height: "100%" }}
      >
      <defs>
        <linearGradient id={`${uid}-base`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0%" stopColor={palette.bg} />
          <stop offset="58%" stopColor={palette.bg} />
          <stop offset="100%" stopColor="#05040a" />
        </linearGradient>
        <radialGradient id={`${uid}-glow`} cx="72%" cy="24%" r="78%">
          <stop offset="0%" stopColor={palette.a} stopOpacity="0.62" />
          <stop offset="46%" stopColor={palette.a} stopOpacity="0.14" />
          <stop offset="100%" stopColor={palette.a} stopOpacity="0" />
        </radialGradient>
        <radialGradient id={`${uid}-vignette`} cx="50%" cy="46%" r="72%">
          <stop offset="58%" stopColor="#000000" stopOpacity="0" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0.5" />
        </radialGradient>
        <linearGradient id={`${uid}-seam`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={palette.ink} stopOpacity="0.2" />
          <stop offset="50%" stopColor={palette.ink} stopOpacity="0.04" />
          <stop offset="100%" stopColor={palette.ink} stopOpacity="0.16" />
        </linearGradient>
        <clipPath id={`${uid}-frame`}>
          <rect x="26" y="26" width="1028" height="1028" />
        </clipPath>
      </defs>

      <rect width="1080" height="1080" fill={`url(#${uid}-base)`} />
      <rect width="1080" height="1080" fill={`url(#${uid}-glow)`} />

      <g clipPath={`url(#${uid}-frame)`}>
        <rect x="936" y="26" width="118" height="1028" fill={`url(#${uid}-seam)`} />
        <g transform="scale(10.8)" opacity="0.96">
          <Motif variant={variant} palette={palette} />
        </g>
        <circle
          cx="540"
          cy="540"
          r="396"
          fill="none"
          stroke={palette.ink}
          strokeOpacity="0.1"
          strokeWidth="1.5"
        />
        <circle
          cx="540"
          cy="540"
          r="468"
          fill="none"
          stroke={palette.ink}
          strokeOpacity="0.06"
          strokeWidth="1"
          strokeDasharray="3 14"
        />
        <path
          d="M0 762H1080"
          stroke={palette.ink}
          strokeOpacity="0.16"
          strokeWidth="1"
        />
        <path
          d="M64 808H430"
          stroke={palette.ink}
          strokeOpacity="0.12"
          strokeWidth="1"
        />
      </g>

      <text
        x="1004"
        y="418"
        textAnchor="end"
        fill={palette.ink}
        fillOpacity="0.14"
        fontFamily={DISPLAY_FONT}
        fontSize="392"
      >
        {initial}
      </text>

      <g fill={palette.ink} fillOpacity="0.34">
        {barcode.map((bar, index) => (
          <rect
            key={index}
            x={936 + (index % 15) * 9}
            y={724 + Math.floor(index / 15) * 52}
            width={bar.width}
            height={bar.height * 0.55}
          />
        ))}
      </g>

      <text
        x="64"
        y="92"
        fill={palette.ink}
        fillOpacity="0.62"
        fontFamily={NUMERIC_FONT}
        fontSize="21"
        fontWeight="700"
        letterSpacing="3"
      >
        MAIFORGE / {catalog}
      </text>
      <text
        x="64"
        y="982"
        fill={palette.ink}
        fillOpacity="0.36"
        fontFamily={NUMERIC_FONT}
        fontSize="17"
        letterSpacing="2.4"
      >
        LOCAL ARCHIVE
      </text>

      <rect width="1080" height="1080" fill={`url(#${uid}-vignette)`} />
      <rect
        x="26"
        y="26"
        width="1028"
        height="1028"
        fill="none"
        stroke={palette.ink}
        strokeOpacity="0.16"
        strokeWidth="2"
      />
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
