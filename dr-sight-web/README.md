# Retina — DR-Sight web front-end

A faithful rebuild of the Base44 **Retina** app
(`fast-clear-retina-scan.base44.app`) as a plain Vite + React project, wired
**directly** to the DR-Sight FastAPI model (`POST /predict`, `X-API-Key`)
instead of Base44 backend functions.

Pages: **Home**, **Dashboard**, **New Screening**, **Results**, **History**,
**Learn** — same layout, copy, fonts (Fraunces + Inter) and teal theme as the
original.

## What changed vs. the Base44 version

| Base44 | Here |
|---|---|
| `base44.functions.invoke("analyzeScreening")` | [`src/lib/analyzeScreening.js`](src/lib/analyzeScreening.js) — browser → FastAPI `/predict`, same response mapping as [`base44-integration/analyzeScreening.ts`](../base44-integration/analyzeScreening.ts) |
| `entities.Screening` (server DB) | [`src/lib/store.js`](src/lib/store.js) — `localStorage`, images downscaled to data URLs |
| `integrations.Core.UploadFile` | image kept client-side as a data URL |
| `exportScreeningReport` PDF function | **Download PDF report** = `window.print()` with a print stylesheet |
| Grad-CAM shown as fabricated `affected_regions` hotspots | real `heatmap_base64` overlay with an opacity slider; hotspot mode still supported if the API ever returns `affected_regions` |
| Extra: model `uncertain` / `referable` / `probabilities` surfaced on Results |

## Run

```bash
cd dr-sight-web
npm install
cp .env.example .env      # set VITE_MODEL_API_URL / VITE_MODEL_API_KEY
npm run dev               # http://localhost:5173
```

In dev the app calls `/model/*`, which Vite proxies to `VITE_MODEL_API_URL`
(see [`vite.config.js`](vite.config.js)) — no CORS setup needed. Point it at the
local service (`http://localhost:8000`, run `../scripts/serve-local.sh`)
or the current Cloudflare tunnel.

## Build

```bash
npm run build            # -> dist/
npm run preview
```

A production build has **no proxy**: it calls `VITE_MODEL_API_URL` directly, so
the DR-Sight service must allow-list this app's origin (`ALLOWED_ORIGIN`). The
`X-API-Key` ships in the bundle — fine for a demo; for a public deploy put a
proxy in front (see `../base44-integration/predictDR.js`).

## Model response contract

`analyzeScreening.js` expects the shape documented in
[`../README.md`](../README.md#response-shape): `usable_image`,
`grade`, `label`, `confidence`, `uncertain`, `referable`, `referral_escalated`,
`referral_action`, `heatmap_base64`, `probabilities`, `disclaimer`.
