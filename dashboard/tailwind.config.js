/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Dark clinical theme.
        background: "#0f1117",
        surface: "#1a1d27",
        accent: "#3b82f6",
        warning: "#f59e0b",
        danger: "#ef4444",
        safe: "#10b981",
      },
    },
  },
  plugins: [],
};
