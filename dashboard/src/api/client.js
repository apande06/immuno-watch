/**
 * Axios API client for the ImmunoWatch backend.
 *
 * One configured instance with interceptors for consistent error handling, plus
 * a typed-ish function per backend endpoint so components never build URLs.
 */
import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const api = axios.create({
  baseURL,
  timeout: 15000,
  headers: { "Content-Type": "application/json" },
});

// Request interceptor — a place to attach auth headers in a real deployment.
api.interceptors.request.use((config) => config);

// Response interceptor — normalise errors (incl. RFC 7807 problem+json bodies).
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const problem = error?.response?.data;
    const message =
      problem?.detail || problem?.title || error.message || "Request failed";
    // eslint-disable-next-line no-console
    console.error(`[ImmunoWatch API] ${message}`);
    return Promise.reject(new Error(message));
  }
);

export const getPatients = () => api.get("/patients").then((r) => r.data);

export const getPatientStatus = (id) =>
  api.get(`/patients/${id}/status`).then((r) => r.data);

export const getReadings = (id, { start, end, limit = 360 } = {}) =>
  api
    .get(`/patients/${id}/readings`, { params: { start, end, limit } })
    .then((r) => r.data);

export const getTrend = (id) => api.get(`/patients/${id}/trend`).then((r) => r.data);

export const getBaseline = (id) =>
  api.get(`/patients/${id}/baseline`).then((r) => r.data);

export const getAlerts = (id, { hours = 24, tier } = {}) =>
  api
    .get(`/patients/${id}/alerts`, { params: { hours, tier } })
    .then((r) => r.data);

export const getCriticalAlerts = () =>
  api.get("/alerts/critical").then((r) => r.data);

export const simulateInfection = (id) =>
  api.post(`/admin/simulate/${id}/infection`).then((r) => r.data);

export const postReading = (id, reading) =>
  api.post(`/patients/${id}/readings`, reading).then((r) => r.data);

export const getModelMetrics = () =>
  api.get("/admin/model/metrics").then((r) => r.data);

export default api;
