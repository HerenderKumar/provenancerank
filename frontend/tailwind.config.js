/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#0b1020",
        panel: "#11182e",
        edge: "#1f2a44",
        brand: { DEFAULT: "#6d8cff", dim: "#3d5bd6" },
        good: "#34d399",
        warn: "#fbbf24",
        bad: "#f87171",
        muted: "#8aa0c6",
      },
    },
  },
  plugins: [],
};
