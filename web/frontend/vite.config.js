import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_PROXY_TARGET = "http://127.0.0.1:8000";

const apiProxy = {
  "/predict": API_PROXY_TARGET,
  "/health": API_PROXY_TARGET,
  "/surface": API_PROXY_TARGET,
  "/comparison": API_PROXY_TARGET,
};

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: apiProxy,
  },
  preview: {
    port: 5173,
    proxy: apiProxy,
  },
});
