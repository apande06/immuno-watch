/**
 * Per-sensor SHAP contribution bars (pure CSS, no chart library).
 *
 * One horizontal bar per sensor; width is proportional to |SHAP| and color
 * encodes direction: red = pushed risk up (contributed to the alert), green =
 * pulled risk down. This is the at-a-glance "why did the model fire" panel.
 */
import React from "react";

export default function ShapCard({ temp = 0, impedance = 0, hrv = 0 }) {
  const rows = [
    { name: "Temperature", value: temp },
    { name: "Impedance", value: impedance },
    { name: "HRV", value: hrv },
  ];
  const max = Math.max(0.01, ...rows.map((r) => Math.abs(r.value)));

  return (
    <div className="space-y-1.5">
      {rows.map((row) => {
        const pct = (Math.abs(row.value) / max) * 100;
        const positive = row.value >= 0;
        return (
          <div key={row.name} className="flex items-center gap-2 text-[11px]">
            <span className="w-20 shrink-0 text-gray-400">{row.name}</span>
            <div className="relative h-3 flex-1 overflow-hidden rounded bg-background">
              <div
                className="absolute inset-y-0 left-0 rounded"
                style={{
                  width: `${pct}%`,
                  backgroundColor: positive ? "#ef4444" : "#10b981",
                }}
              />
            </div>
            <span
              className="w-12 shrink-0 text-right tabular-nums"
              style={{ color: positive ? "#ef4444" : "#10b981" }}
            >
              {row.value >= 0 ? "+" : ""}
              {row.value.toFixed(2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
