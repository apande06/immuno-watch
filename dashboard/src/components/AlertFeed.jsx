/**
 * Chronological alert feed (newest first) with SHAP attribution and dual
 * explanations. Each card has a tier-colored left border, the risk score, the
 * ShapCard, a collapsible clinical explanation, and an always-visible
 * patient-facing explanation. Filter chips narrow by tier.
 */
import React, { useEffect, useMemo, useState } from "react";
import { getAlerts } from "../api/client";
import { TIER_COLOR } from "../theme";
import ShapCard from "./ShapCard.jsx";

const FILTERS = ["ALL", "WATCH", "WARNING", "CRITICAL"];

export default function AlertFeed({ patientId, tick }) {
  const [alerts, setAlerts] = useState([]);
  const [filter, setFilter] = useState("ALL");

  useEffect(() => {
    let alive = true;
    getAlerts(patientId, { hours: 168 })
      .then((data) => alive && setAlerts(data))
      .catch(() => alive && setAlerts([]));
    return () => {
      alive = false;
    };
  }, [patientId, tick]);

  const filtered = useMemo(
    () => (filter === "ALL" ? alerts : alerts.filter((a) => a.tier === filter)),
    [alerts, filter]
  );

  return (
    <div className="space-y-4">
      <div className="flex gap-1 rounded-lg bg-surface p-1 w-fit">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-md px-3 py-1 text-xs ${
              filter === f ? "bg-accent text-white" : "text-gray-400 hover:text-white"
            }`}
          >
            {f}
          </button>
        ))}
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-xl border border-gray-800 bg-surface p-8 text-center text-sm text-gray-500">
          No alerts in last 24 hours — patient stable
        </div>
      ) : (
        filtered.map((alert, i) => <AlertCard key={i} alert={alert} />)
      )}
    </div>
  );
}

function AlertCard({ alert }) {
  const [open, setOpen] = useState(false);
  const color = TIER_COLOR[alert.tier] || TIER_COLOR.NORMAL;
  const ts = new Date(alert.timestamp).toLocaleString();

  return (
    <div
      className="rounded-xl border border-gray-800 bg-surface p-4"
      style={{ borderLeft: `4px solid ${color}` }}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="rounded-full px-2.5 py-0.5 text-[11px] font-semibold text-white"
            style={{ backgroundColor: color }}
          >
            {alert.tier}
          </span>
          <span className="text-xs text-gray-500">{ts}</span>
        </div>
        <div className="text-right">
          <div className="text-sm font-semibold text-white">
            risk {(alert.risk_score * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-gray-500">severity {alert.severity.toFixed(1)}/10</div>
        </div>
      </div>

      <div className="mt-3">
        <ShapCard
          temp={alert.shap_temp}
          impedance={alert.shap_impedance}
          hrv={alert.shap_hrv}
        />
      </div>

      <p className="mt-3 rounded-lg bg-background p-3 text-xs text-gray-300">
        {alert.patient_explanation}
      </p>

      <button
        onClick={() => setOpen((o) => !o)}
        className="mt-2 text-[11px] text-accent hover:underline"
      >
        {open ? "Hide" : "Show"} clinical detail
      </button>
      {open && (
        <p className="mt-1 rounded-lg border border-gray-800 bg-background p-3 text-[11px] leading-relaxed text-gray-400">
          {alert.clinical_explanation}
        </p>
      )}
    </div>
  );
}
