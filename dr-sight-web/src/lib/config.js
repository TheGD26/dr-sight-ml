// Where the browser sends /predict, and the key it attaches.
//
// dev  : leave VITE_MODEL_API_URL unset (or set it) — requests go to `/model/*`
//        which Vite proxies to the FastAPI service (see vite.config.js). No CORS.
// prod : set VITE_MODEL_API_URL to the deployed service URL. That service must
//        allow-list this app's origin (ALLOWED_ORIGIN on the DR-Sight API).
//
// NOTE: with a direct browser -> FastAPI connection the API key ships in the
// client bundle. That is acceptable for a local / demo / hackathon build; for a
// public deployment put a proxy in front (see dr-sight-ml/base44-integration).

const rawUrl = import.meta.env.VITE_MODEL_API_URL?.replace(/\/+$/, "") || "";

export const MODEL_BASE_URL = import.meta.env.DEV
  ? "/model"
  : rawUrl || "/model";

export const MODEL_API_KEY = import.meta.env.VITE_MODEL_API_KEY || "";

// Sample fundus image used by "try with a sample image" — same asset the
// original Retina app used.
export const SAMPLE_IMAGE_URL =
  "https://images.unsplash.com/photo-1620121692029-d088224dd2c4?auto=format&fit=crop&w=900&q=80";
