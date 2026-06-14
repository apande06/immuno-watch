/**
 * Floating "Simulate Infection" control (bottom-right, above everything).
 *
 * On click it triggers the backend's 60-minute infection cascade, accelerates the
 * dashboard polling to 10s so the panels visibly react, and surfaces escalating
 * toasts as the alert tier climbs WATCH -> WARNING -> CRITICAL. After five minutes
 * it restores the normal 60s polling and shows a completion toast.
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import { getAlerts, simulateInfection } from "../api/client";
import { TIER_COLOR } from "../theme";

const TOASTS = {
  WATCH: "⚠️ WATCH detected",
  WARNING: "🚨 WARNING detected",
  CRITICAL: "🆘 CRITICAL — Immediate intervention required",
};

export default function SimulateButton({ patientId, setPollMs }) {
  const [running, setRunning] = useState(false);
  const [toasts, setToasts] = useState([]);
  const timers = useRef([]);

  const pushToast = useCallback((text, tier) => {
    const id = Math.random().toString(36).slice(2);
    setToasts((t) => [...t, { id, text, tier }]);
    const timer = setTimeout(
      () => setToasts((t) => t.filter((x) => x.id !== id)),
      6000
    );
    timers.current.push(timer);
  }, []);

  useEffect(
    () => () => timers.current.forEach(clearTimeout),
    []
  );

  const onClick = useCallback(async () => {
    if (running) return;
    setRunning(true);
    setPollMs(10000); // accelerate polling during the simulation
    try {
      await simulateInfection(patientId);
      const alerts = await getAlerts(patientId, { hours: 1 });
      const tiers = new Set(alerts.map((a) => a.tier));
      ["WATCH", "WARNING", "CRITICAL"].forEach((tier, i) => {
        if (tiers.has(tier)) {
          const t = setTimeout(() => pushToast(TOASTS[tier], tier), i * 1500);
          timers.current.push(t);
        }
      });
      if (tiers.size === 0) pushToast("Simulation ran — patient remained stable", "WATCH");
    } catch (e) {
      pushToast(`Simulation failed: ${e.message}`, "CRITICAL");
    }

    const reset = setTimeout(() => {
      setPollMs(60000);
      setRunning(false);
      pushToast("Simulation complete", "WATCH");
    }, 5 * 60 * 1000);
    timers.current.push(reset);
  }, [running, patientId, setPollMs, pushToast]);

  return (
    <>
      <div className="pointer-events-none fixed bottom-6 right-6 z-50 flex flex-col items-end gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="pointer-events-auto rounded-lg px-4 py-2 text-sm font-medium text-white shadow-lg"
            style={{ backgroundColor: TIER_COLOR[t.tier] || TIER_COLOR.WATCH }}
          >
            {t.text}
          </div>
        ))}
        <button
          onClick={onClick}
          disabled={running}
          className={`pointer-events-auto rounded-full px-5 py-3 text-sm font-semibold text-white shadow-xl transition ${
            running ? "cursor-not-allowed bg-gray-600" : "bg-danger animate-immuno-pulse hover:brightness-110"
          }`}
        >
          {running ? "Simulating…" : "🦠 Simulate Infection"}
        </button>
      </div>
    </>
  );
}
