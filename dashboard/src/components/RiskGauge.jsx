/**
 * Semicircular SVG risk gauge (0-10) with colored zones and an animated needle.
 *
 * Zones: green 0-4, amber 4-6.5, red 6.5-10. The backend risk score is 0-1, so it
 * is scaled by 10 for display. The needle transitions smoothly when the score
 * updates via a CSS transform transition.
 */
import React from "react";
import { TIER_COLOR, TIER_LABEL, colorForGauge } from "../theme";

const CX = 100;
const CY = 100;
const R = 80;

function polar(cx, cy, r, angleDeg) {
  const a = (angleDeg * Math.PI) / 180;
  return { x: cx + r * Math.cos(a), y: cy - r * Math.sin(a) };
}

// Value 0-10 maps to angle 180deg (left) .. 0deg (right).
function valueToAngle(value) {
  return 180 - (Math.max(0, Math.min(10, value)) / 10) * 180;
}

function arcPath(startValue, endValue) {
  const start = polar(CX, CY, R, valueToAngle(startValue));
  const end = polar(CX, CY, R, valueToAngle(endValue));
  const largeArc = 0;
  return `M ${start.x} ${start.y} A ${R} ${R} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

export default function RiskGauge({ score = 0, tier }) {
  const value = Math.max(0, Math.min(10, score * 10));
  const needle = polar(CX, CY, R - 10, valueToAngle(value));
  const displayTier = TIER_LABEL(tier);
  const tierColor = TIER_COLOR[displayTier] || TIER_COLOR.NORMAL;

  return (
    <div className="flex flex-col items-center">
      <h3 className="mb-2 self-start text-sm font-medium text-gray-300">
        Infection Risk
      </h3>
      <svg viewBox="0 0 200 120" className="w-full max-w-xs">
        {/* track */}
        <path d={arcPath(0, 10)} fill="none" stroke="#262b38" strokeWidth="14" strokeLinecap="round" />
        {/* colored zones */}
        <path d={arcPath(0, 4)} fill="none" stroke="#10b981" strokeWidth="14" strokeLinecap="round" />
        <path d={arcPath(4, 6.5)} fill="none" stroke="#f59e0b" strokeWidth="14" />
        <path d={arcPath(6.5, 10)} fill="none" stroke="#ef4444" strokeWidth="14" strokeLinecap="round" />

        {/* needle (animated) */}
        <line
          x1={CX}
          y1={CY}
          x2={needle.x}
          y2={needle.y}
          stroke={colorForGauge(value)}
          strokeWidth="3"
          strokeLinecap="round"
          style={{ transition: "all 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)" }}
        />
        <circle cx={CX} cy={CY} r="6" fill={colorForGauge(value)} />

        <text x={CX} y={CY - 18} textAnchor="middle" className="fill-white" fontSize="26" fontWeight="700">
          {value.toFixed(1)}
        </text>
        <text x={CX} y={CY + 2} textAnchor="middle" fill="#9ca3af" fontSize="9">
          / 10
        </text>
      </svg>

      <span
        className="mt-1 rounded-full px-4 py-1 text-xs font-semibold text-white"
        style={{ backgroundColor: tierColor }}
      >
        {displayTier}
      </span>
    </div>
  );
}
