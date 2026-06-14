import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard runs on :3000 (the origin the FastAPI CORS policy allows).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    host: true,
  },
});
