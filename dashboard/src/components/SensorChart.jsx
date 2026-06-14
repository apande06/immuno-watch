/**
 * Single-sensor time-series chart with a personal-baseline band overlay.
 *
 * Used three times (temperature, HRV, impedance). Renders a Recharts
 * ComposedChart: the live reading line, a dashed ReferenceLine at the patient's
 * learned baseline, and a shaded +/-2 sigma ReferenceArea — so a clinician sees
 * instantly when a value leaves *this patient's* normal envelope.
 */
import React, { useEffect, useState } from "react";
import {
  Area,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getBaseline, getReadings } from "../api/client";

const SENSORS = {
  temp: {
    field: "temp_c",
    label: "Core Temperature",
    unit: "°C",
    baselineKey: "baseline_temp",
    sdKey: "baseline_temp_sd",
    color: "#ef4444",
  },
  hrv: {
    field: "hrv_rmssd_ms",
    label: "HRV (RMSSD)",
    unit: "ms",
    baselineKey: "baseline_hrv",
    sdKey: "baseline_hrv_sd",
    color: "#3b82f6",
  },
  impedance: {
    field: "impedance_ohm",
    label: "Bioimpedance",
    unit: "Ω",
    baselineKey: "baseline_impedance",
    sdKey: "baseline_impedance_sd",
    color: "#10b981",
  },
};

const RANGES = { "2h": 120, "6h": 360, "24h": 1440, "7d": 10080 };

export default function SensorChart({ patientId, sensor, tick }) {
  const cfg = SENSORS[sensor];
  const [range, setRange] = useState("6h");
  const [data, setData] = useState([]);
  const [baseline, setBaseline] = useState(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [readings, base] = await Promise.all([
          getReadings(patientId, { limit: RANGES[range] }),
          getBaseline(patientId),
        ]);
        if (!alive) return;
        setBaseline(base);
        setData(
          readings.map((r) => ({
            t: new Date(r.timestamp).getTime(),
            value: r[cfg.field],
          }))
        );
      } catch {
        if (alive) setData([]);
      }
    })();
    return () => {
      alive = false;
    };
  }, [patientId, sensor, range, tick, cfg.field]);

  const base = baseline ? baseline[cfg.baselineKey] : null;
  const sd = baseline ? baseline[cfg.sdKey] : null;
  const fmtTime = (t) =>
    range === "7d"
      ? new Date(t).toLocaleDateString([], { month: "short", day: "numeric" })
      : new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <div className="rounded-xl border border-gray-800 bg-surface p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-200">
          {cfg.label} <span className="text-gray-500">({cfg.unit})</span>
        </h3>
        <div className="flex gap-1 rounded-md bg-background p-0.5">
          {Object.keys(RANGES).map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`rounded px-2 py-0.5 text-[11px] ${
                range === r ? "bg-accent text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              {r}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ComposedChart data={data} margin={{ top: 5, right: 10, bottom: 0, left: -10 }}>
          <XAxis
            dataKey="t"
            tickFormatter={fmtTime}
            stroke="#4b5563"
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            minTickGap={40}
          />
          <YAxis
            domain={["auto", "auto"]}
            stroke="#4b5563"
            tick={{ fontSize: 10, fill: "#9ca3af" }}
            width={44}
          />
          {base != null && sd != null && (
            <ReferenceArea
              y1={base - 2 * sd}
              y2={base + 2 * sd}
              fill={cfg.color}
              fillOpacity={0.08}
              stroke="none"
            />
          )}
          {base != null && (
            <ReferenceLine
              y={base}
              stroke={cfg.color}
              strokeDasharray="4 4"
              strokeOpacity={0.7}
              label={{ value: "baseline", position: "insideTopLeft", fill: "#9ca3af", fontSize: 9 }}
            />
          )}
          <Line
            type="monotone"
            dataKey="value"
            stroke={cfg.color}
            strokeWidth={1.6}
            dot={false}
            isAnimationActive={false}
          />
          <Tooltip
            content={<SensorTooltip unit={cfg.unit} base={base} fmtTime={fmtTime} />}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function SensorTooltip({ active, payload, unit, base, fmtTime }) {
  if (!active || !payload || !payload.length) return null;
  const point = payload[0].payload;
  const delta = base != null ? point.value - base : null;
  return (
    <div className="rounded-md border border-gray-700 bg-background px-3 py-2 text-xs">
      <div className="text-gray-400">{fmtTime(point.t)}</div>
      <div className="font-semibold text-white">
        {point.value?.toFixed(2)} {unit}
      </div>
      {delta != null && (
        <div className={delta >= 0 ? "text-danger" : "text-safe"}>
          {delta >= 0 ? "+" : ""}
          {delta.toFixed(2)} {unit} vs baseline
        </div>
      )}
    </div>
  );
}
