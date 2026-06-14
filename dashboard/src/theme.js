/** Shared clinical theme helpers — tier colors and archetype styling. */

export const TIER_COLOR = {
  CRITICAL: "#ef4444",
  WARNING: "#f59e0b",
  WATCH: "#3b82f6",
  NORMAL: "#10b981",
  null: "#10b981",
};

export const TIER_LABEL = (tier) => tier || "NORMAL";

/** Map a 0-1 risk score to the matching tier label using the backend thresholds. */
export function tierForScore(score) {
  if (score >= 0.85) return "CRITICAL";
  if (score >= 0.65) return "WARNING";
  if (score >= 0.4) return "WATCH";
  return "NORMAL";
}

/** Color for a 0-1 risk score. */
export function colorForScore(score) {
  return TIER_COLOR[tierForScore(score)];
}

export const ARCHETYPE_BADGE = {
  chemo_nadir: { label: "Chemo Nadir", color: "#a855f7" },
  organ_transplant: { label: "Transplant", color: "#06b6d4" },
  hiv_managed: { label: "HIV Managed", color: "#22c55e" },
};

export function archetypeStyle(archetype) {
  return ARCHETYPE_BADGE[archetype] || { label: archetype, color: "#6b7280" };
}

/** Color for a 0-10 risk score on the gauge. */
export function colorForGauge(value) {
  if (value >= 6.5) return "#ef4444";
  if (value >= 4) return "#f59e0b";
  return "#10b981";
}
