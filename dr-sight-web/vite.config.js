import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// During `npm run dev` the `/model` path is proxied to the DR-Sight FastAPI
// service, so the browser talks to same-origin and local CORS is a non-issue.
// In a production build there is no proxy: set VITE_MODEL_API_URL to the real
// service URL and have that service allow-list this app's origin.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_MODEL_API_URL || "http://localhost:8000";
  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/model": {
          target,
          changeOrigin: true,
          secure: false,
          rewrite: (p) => p.replace(/^\/model/, ""),
        },
      },
    },
  };
});
