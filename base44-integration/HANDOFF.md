# Base44 wiring — hand-off (dev tunnel)

For the teammate who owns **https://fast-clear-retina-scan.base44.app**.
Full explanation is in [`README.md`](./README.md); this is just the concrete
values for the current setup.

The DR-Sight API is **not deployed yet** — it runs on Giridharan's laptop and is
exposed through a temporary Cloudflare tunnel. It's live only while he's running
it, and **the URL changes every restart** — ping him for the current one before
a demo. Move to a real deploy (Render, see [`DEPLOY.md`](./DEPLOY.md)) before
anything that needs to stay up.

## Values

| Thing | Value |
|---|---|
| Model API URL (temp) | `https://cabinet-born-newcastle-tribes.trycloudflare.com` |
| `API_KEY` / Base44 Secret `DR_MODEL_API_KEY` | `175615c9ea8da92ca43392be107463536387bd2f3ca12de2` |
| CORS origin already allow-listed on the API | `https://fast-clear-retina-scan.base44.app` |

## Steps in the Base44 editor

1. **Settings → Secrets** → add `DR_MODEL_API_KEY` =
   `175615c9ea8da92ca43392be107463536387bd2f3ca12de2`.
2. **Backend Functions → New** → name it `predictDR` → paste
   [`predictDR.js`](./predictDR.js) (the `MODEL_API_URL` in it is already set to
   the current tunnel; update it when the tunnel URL changes or when we deploy).
3. Wire the **Analyze** button to call it — see README.md **Step 3** (file →
   base64 → `base44.functions.predictDR({ imageBase64 })`) and **Step 4** for
   rendering `grade` / `label` / `confidence` / `referral_action` /
   `heatmap_base64` / `disclaimer`.

## Quick check (from anywhere, while the tunnel is up)

```bash
curl -s https://cabinet-born-newcastle-tribes.trycloudflare.com/health
# {"status":"ok","model_loaded":true,"synthetic_weights":true,"backbone":"edge",...}
```

`synthetic_weights: true` is expected — the model isn't trained on real fundus
images yet, so grades are not meaningful. The contract (fields, Grad-CAM, quality
gate, referral text) is real and final.
