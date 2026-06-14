/**
 * ImmunoWatch dashboard root.
 *
 * Fixed left sidebar (patient roster) + main content area with three tabs:
 * Monitor (risk gauge + the three live sensor charts), Alerts (the SHAP-annotated
 * alert feed), and Trends (the 7-day risk heatmap). A global polling interval
 * keeps every panel fresh; the SimulateButton can temporarily accelerate it.
 */
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { getPatients } from "./api/client";
import PatientSidebar from "./components/PatientSidebar.jsx";
import SensorChart from "./components/SensorChart.jsx";
import RiskGauge from "./components/RiskGauge.jsx";
import AlertFeed from "./components/AlertFeed.jsx";
import TrendHeatmap from "./components/TrendHeatmap.jsx";
import SimulateButton from "./components/SimulateButton.jsx";

const TABS = [
  { id: "monitor", label: "Monitor" },
  { id: "alerts", label: "Alerts" },
  { id: "trends", label: "Trends" },
];

export default function App() {
  const [patients, setPatients] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [activeTab, setActiveTab] = useState("monitor");
  const [pollMs, setPollMs] = useState(60000);
  const [tick, setTick] = useState(0);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      const data = await getPatients();
      setPatients(data);
      setError(null);
      setSelectedId((prev) => prev || (data[0] && data[0].patient_id) || null);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  // Poll on the active interval; each tick also nudges child panels to refetch.
  useEffect(() => {
    refresh();
    const id = setInterval(() => {
      refresh();
      setTick((t) => t + 1);
    }, pollMs);
    return () => clearInterval(id);
  }, [refresh, pollMs]);

  const selected = useMemo(
    () => patients.find((p) => p.patient_id === selectedId) || null,
    [patients, selectedId]
  );

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background text-gray-200">
      <PatientSidebar
        patients={patients}
        selectedId={selectedId}
        onSelect={setSelectedId}
        error={error}
      />

      <main className="flex flex-1 flex-col overflow-hidden">
        <header className="flex items-center justify-between border-b border-gray-800 bg-surface px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold text-white">
              {selected ? selected.patient_id : "ImmunoWatch"}
            </h1>
            <p className="text-xs text-gray-400">
              Continuous AI monitoring for immunocompromised patients
            </p>
          </div>
          <nav className="flex gap-1 rounded-lg bg-background p-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`rounded-md px-4 py-1.5 text-sm transition ${
                  activeTab === tab.id
                    ? "bg-accent text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </header>

        <section className="flex-1 overflow-y-auto p-6">
          {!selected ? (
            <EmptyState error={error} />
          ) : activeTab === "monitor" ? (
            <MonitorTab patient={selected} tick={tick} />
          ) : activeTab === "alerts" ? (
            <AlertFeed patientId={selected.patient_id} tick={tick} />
          ) : (
            <TrendHeatmap patientId={selected.patient_id} tick={tick} />
          )}
        </section>
      </main>

      {selected && (
        <SimulateButton patientId={selected.patient_id} setPollMs={setPollMs} />
      )}
    </div>
  );
}

function MonitorTab({ patient, tick }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="rounded-xl border border-gray-800 bg-surface p-4 lg:col-span-1">
          <RiskGauge
            score={patient.current_risk_score}
            tier={patient.current_tier}
          />
        </div>
        <div className="grid grid-cols-1 gap-4 lg:col-span-2">
          <SensorChart patientId={patient.patient_id} sensor="temp" tick={tick} />
        </div>
      </div>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <SensorChart patientId={patient.patient_id} sensor="hrv" tick={tick} />
        <SensorChart patientId={patient.patient_id} sensor="impedance" tick={tick} />
      </div>
    </div>
  );
}

function EmptyState({ error }) {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center text-gray-500">
      <div className="text-4xl">🩺</div>
      <p className="mt-3 text-sm">
        {error
          ? `Cannot reach the ImmunoWatch API (${error}).`
          : "Loading patients…"}
      </p>
      {error && (
        <p className="mt-1 text-xs text-gray-600">
          Start the backend with <code>uvicorn api.main:app</code> and refresh.
        </p>
      )}
    </div>
  );
}
