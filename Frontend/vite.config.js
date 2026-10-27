import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api/* to the FastAPI backend, so no CORS setup is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8001", changeOrigin: true },
    },
  },
});
