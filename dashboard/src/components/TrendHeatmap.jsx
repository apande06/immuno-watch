/**
 * 7-day risk heatmap: 7 rows (days) x 24 columns (hours). Each cell is colored by
 * its hourly risk score (green -> amber -> red). Backs the "Trends" tab so a
 * clinician can spot the slow build-up of risk that precedes an event.
 */
import React, { useEffect, useMemo, useState } from "react";
import { getTrend } from "../api/client";
import { tierForScore } from "../theme";

const HOURS = Array.from({ length: 24 }, (_, i) => i);

// Interpolate green -> amber -> red across risk 0..1.
function riskColor(risk) {
  if (risk == null) return "#1a1d27";
  const stops =
    risk < 0.5
      ? [[16, 185, 129], [245, 158, 11], risk / 0.5]
      : [[245, 158, 11], [239, 68, 68], (risk - 0.5) / 0.5];
  const [a, b, t] = stops;
  const mix = a.map((c, i) => Math.round(c + (b[i] - c) * t));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

export default function TrendHeatmap({ patientId, tick }) {
  const [points, setPoints] = useState([]);

  useEffect(() => {
    let alive = true;
    getTrend(patientId)
      .then((data) => alive && setPoints(data))
      .catch(() => alive && setPoints([]));
    return () => {
      alive = false;
    };
  }, [patientId, tick]);

  const { rows, days } = useMemo(() => buildGrid(points), [points]);

  return (
    <div className="rounded-xl border border-gray-800 bg-surface p-5">
      <h3 className="mb-4 text-sm font-medium text-gray-200">7-Day Risk Trend</h3>

      {rows.length === 0 ? (
        <p className="py-8 text-center text-sm text-gray-500">No trend data available</p>
      ) : (
        <div className="space-y-1">
          <div className="flex items-center gap-1 pl-16 text-[9px] text-gray-600">
            {HOURS.filter((h) => h % 3 === 0).map((h) => (
              <span key={h} className="w-[27px]">
                {String(h).padStart(2, "0")}
              </span>
            ))}
          </div>
          {rows.map((row, di) => (
            <div key={di} className="flex items-center gap-1">
              <span className="w-14 shrink-0 text-right text-[10px] text-gray-500">
                {days[di]}
              </span>
              <div className="flex gap-0.5">
                {HOURS.map((h) => {
                  const cell = row[h];
                  return (
                    <div
                      key={h}
                      title={
                        cell == null
                          ? `${days[di]} ${String(h).padStart(2, "0")}:00 — no data`
                          : `${days[di]} ${String(h).padStart(2, "0")}:00 — risk ${(cell * 100).toFixed(0)}% (${tierForScore(cell)})`
                      }
                      className="h-5 w-2.5 rounded-sm transition-transform hover:scale-150"
                      style={{ backgroundColor: riskColor(cell) }}
                    />
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}

      <Legend />
    </div>
  );
}

function Legend() {
  return (
    <div className="mt-5 flex items-center gap-2 text-[10px] text-gray-500">
      <span>low</span>
      <div className="flex">
        {Array.from({ length: 20 }, (_, i) => (
          <div key={i} className="h-2 w-3" style={{ backgroundColor: riskColor(i / 19) }} />
        ))}
      </div>
      <span>high risk</span>
    </div>
  );
}

function buildGrid(points) {
  const byDay = new Map();
  for (const p of points) {
    const d = new Date(p.timestamp);
    const key = d.toLocaleDateString([], { month: "short", day: "numeric" });
    if (!byDay.has(key)) byDay.set(key, new Array(24).fill(null));
    byDay.get(key)[d.getHours()] = p.risk_score;
  }
  const days = [...byDay.keys()].slice(-7);
  return { rows: days.map((d) => byDay.get(d)), days };
}
