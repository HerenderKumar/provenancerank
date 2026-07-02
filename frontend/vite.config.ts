import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies API + SSE to the FastAPI backend so the UI calls same-origin
// paths and there's no CORS dance in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
