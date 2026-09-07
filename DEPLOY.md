# Deploying the DR-Sight API

Goal: get the container running on a public **HTTPS** URL that the Base44 app can
call. We use **Render** — it builds from your `Dockerfile`, gives you a stable
`https://<name>.onrender.com` URL for free, and needs no cloud background.

> Render's free web service **sleeps after ~15 min idle** and takes ~30–60 s to
> wake. For judging, hit `/health` a few minutes beforehand to warm it, or use
> the cheapest paid instance ($7/mo) to disable sleeping.

---

## 1. Prerequisites

- A GitHub repo containing this `dr-sight-ml/` folder (Render deploys from Git).
- A trained checkpoint committed at `models/best.pt`.
  - Real weights if training finished — see `README.md`.
  - Or the `edge` (MobileNetV3-Small) checkpoint if B3 is too slow on free CPU.
  - `models/best.pt` is **gitignored by default** (`.gitignore` line `models/`).
    Force-add the one file you want to ship:
    ```bash
    git add -f models/best.pt
    git commit -m "ship DR-Sight checkpoint"
    ```
  - If the file is large (>100 MB) use Git LFS, or bake it into the image another
    way (Render also supports a build-time `curl` from a release asset).

## 2. Create the service on Render

1. Sign in at <https://render.com> with GitHub.
2. **New → Web Service** → pick your repo.
3. Settings:
   | Field | Value |
   |---|---|
   | Root Directory | `dr-sight-ml` |
   | Runtime | **Docker** |
   | Dockerfile Path | `dr-sight-ml/Dockerfile` |
   | Instance Type | Free (or Starter $7/mo to avoid cold starts) |
   | Health Check Path | `/health` |
4. Click **Create Web Service**. The first build takes ~5–10 min (PyTorch).

## 3. Set environment variables (Render dashboard → *Environment*)

**Never put these in code or in the Dockerfile.**

| Key | Example | Notes |
|---|---|---|
| `API_KEY` | `dr-sight-9f3c…` (32+ random chars) | The shared secret. Generate with `openssl rand -hex 24`. The Base44 Secret `DR_MODEL_API_KEY` must be set to the **same value**. |
| `ALLOWED_ORIGIN` | `https://your-app.base44.app` | Your teammate's Base44 app domain(s), comma-separated. **Not `*`.** Ask them for the exact domain. |
| `DR_BACKBONE` | `b3` | `b3` (accuracy) or `edge` (fast). Must match the committed checkpoint. |
| `DR_DEVICE` | `cpu` | Render free/standard instances are CPU-only. |
| `DR_UNCERTAINTY_THRESHOLD` | `0.6` | Below this top-1 softmax, results are flagged `"uncertain": true`. |
| `MAX_UPLOAD_BYTES` | `12582912` | 12 MB. Bodies larger than this get `413` before the model runs. |

Redeploy after adding vars (Render does this automatically on save).

## 4. Verify

```bash
BASE=https://your-app.onrender.com
KEY=<the API_KEY you set>

curl -s $BASE/health
# {"status":"ok","model_loaded":true,"using_trained_weights":true,
#  "synthetic_weights":false,"backbone":"b3",...}

curl -s -X POST $BASE/predict \
  -H "X-API-Key: $KEY" \
  -F "file=@sample_fundus.jpg" | python -m json.tool
```

You should get the JSON described in `README.md` (grade, confidence,
`referral_action`, `heatmap_base64`, disclaimer).

Auth check — this must return `401`:
```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE/predict -F "file=@sample_fundus.jpg"
```

## 5. Performance notes

- **EfficientNet-B3 on CPU:** ~1–4 s/image on a Render standard instance.
  Acceptable for a live demo (one image at a time).
- **If it's visibly slow during judging:** switch to the small model.
  1. Train/obtain a `edge` checkpoint: `python -m src.model.train --backbone edge …`
  2. Commit it as `models/best.pt`.
  3. Set `DR_BACKBONE=edge` on Render. Redeploy.
  MobileNetV3-Small is ~5× faster and ~2.5M params.
- `--workers 1` in the Dockerfile keeps memory predictable on a 512 MB free
  instance. Bump to 2 only on a paid instance with ≥2 GB RAM.

## Alternative hosts (same idea, if Render doesn't work out)

- **Railway** (<https://railway.app>): New Project → Deploy from repo → it detects
  the Dockerfile. Set the same env vars under *Variables*. Gives an
  `*.up.railway.app` HTTPS URL. Small monthly usage credit on the free plan.
- **Hugging Face Space (Docker SDK)**: create a Space, SDK = *Docker*, push this
  folder; add `API_KEY` / `ALLOWED_ORIGIN` as *Repository secrets*. URL is
  `https://<user>-<space>.hf.space`. Note the app must listen on port **7860**
  there — change the Dockerfile `CMD`/`EXPOSE` to `7860`.
