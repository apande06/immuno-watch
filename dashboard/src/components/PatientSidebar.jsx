/**
 * Fixed left sidebar: the patient roster.
 *
 * Each card shows the patient id, an archetype badge, a tier-colored risk badge,
 * and the last-updated time. Clicking a card selects that patient for every panel.
 */
import React from "react";
import {
  TIER_COLOR,
  TIER_LABEL,
  archetypeStyle,
  tierForScore,
} from "../theme";

export default function PatientSidebar({ patients, selectedId, onSelect, error }) {
  return (
    <aside className="flex w-72 flex-col border-r border-gray-800 bg-surface">
      <div className="border-b border-gray-800 px-5 py-4">
        <div className="flex items-center gap-2">
          <span className="text-xl">🩺</span>
          <span className="text-lg font-semibold text-white">ImmunoWatch</span>
        </div>
        <p className="mt-1 text-xs text-gray-500">Patient roster</p>
      </div>

      <div className="flex-1 overflow-y-auto p-3">
        {patients.length === 0 && (
          <p className="px-2 py-4 text-xs text-gray-500">
            {error ? "API unavailable" : "No patients yet"}
          </p>
        )}
        {patients.map((p) => (
          <PatientCard
            key={p.patient_id}
            patient={p}
            selected={p.patient_id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </div>

      <div className="border-t border-gray-800 px-5 py-3 text-[10px] text-gray-600">
        ImmunoWatch v1.0 · {patients.length} monitored
      </div>
    </aside>
  );
}

function PatientCard({ patient, selected, onSelect }) {
  const arch = archetypeStyle(patient.archetype);
  const tier = patient.current_tier || tierForScore(patient.current_risk_score);
  const tierColor = TIER_COLOR[tier] || TIER_COLOR.NORMAL;
  const updated = patient.last_updated
    ? new Date(patient.last_updated).toLocaleTimeString()
    : "—";

  return (
    <button
      onClick={() => onSelect(patient.patient_id)}
      className={`mb-2 w-full rounded-lg border p-3 text-left transition ${
        selected
          ? "border-accent bg-background"
          : "border-gray-800 bg-surface hover:border-gray-700"
      }`}
    >
      <div className="flex items-center justify-between">
        <span className="truncate text-sm font-medium text-white">
          {patient.patient_id}
        </span>
        <span
          className="rounded-full px-2 py-0.5 text-[10px] font-semibold text-white"
          style={{ backgroundColor: tierColor }}
        >
          {(patient.current_risk_score * 100).toFixed(0)}%
        </span>
      </div>
      <div className="mt-2 flex items-center justify-between">
        <span
          className="rounded px-1.5 py-0.5 text-[10px] font-medium"
          style={{ backgroundColor: `${arch.color}22`, color: arch.color }}
        >
          {arch.label}
        </span>
        <span className="text-[10px]" style={{ color: tierColor }}>
          {TIER_LABEL(patient.current_tier)}
        </span>
      </div>
      <p className="mt-1 text-[10px] text-gray-500">updated {updated}</p>
    </button>
  );
}
