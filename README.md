# DR-Sight — ML backend

Diabetic-retinopathy screening model + explainability + REST API for the
**DR-Sight** app (Smart India Hackathon 2026, PS **SIH26038**, Team **Diasight**).

Grades a fundus image **0–4** on the International Clinical DR severity scale,
explains the prediction with **Grad-CAM**, rejects unusable photos with an
**image-quality gate**, and serves it all over a small **FastAPI** service the
Base44 front-end calls through a Backend Function.

> ⚠️ **Prototype, not a medical device.** This tool **screens** and
> **recommends a referral timeframe**. It does **not diagnose**. Every result
> carries a disclaimer and must be confirmed by an ophthalmologist. The
> not-a-device notice lives at the top of
> [`src/inference/pipeline.py`](src/inference/pipeline.py).

---

## What's in here

| Path | What |
|---|---|
| `src/data/prepare_aptos.py` | Download APTOS 2019 (Kaggle API), patient-level stratified split, resize-cache. `--synthetic` fallback. |
| `src/data/dataset.py` | PyTorch `Dataset` + conservative augmentations (rotate / flip / mild brightness-contrast — no colour distortion). |
| `src/quality/image_quality.py` | Blur (variance-of-Laplacian), exposure, field-of-view gate → `{"usable", "reason", "scores"}`. |
| `src/model/architecture.py` | `timm` EfficientNet-B3 (`b3`) and MobileNetV3-Small (`edge`) with a 5-way head. |
| `src/model/train.py` | Two-stage transfer learning, class-weighted loss, checkpoint on best val **quadratic weighted kappa**. |
| `src/model/evaluate.py` | Sensitivity / specificity / precision / F1 / AUROC / QWK + confusion matrix. Headline = **referable-DR (grade ≥ 2) sensitivity**. |
| `src/explain/gradcam.py` | `pytorch-grad-cam` on the last conv block → overlay PNG (base64) + raw CAM array. CLI dumps 5 samples. |
| `src/inference/pipeline.py` | quality gate → preprocess → model → Grad-CAM → referral lookup + uncertainty flag. |
| `src/api/main.py` | `POST /predict`, `GET /health`, `X-API-Key` auth, CORS allow-list, body-size limit. |
| `base44-integration/` | Reference Base44 Backend Function + wiring guide. |
| `Dockerfile`, `DEPLOY.md` | CPU image + step-by-step deploy (Render). |

---

## Setup

```bash
cd dr-sight-ml
python3 -m venv .venv && source .venv/bin/activate      # Python 3.11+ (3.13 tested)
pip install -r requirements.txt
export PYTHONPATH=.
```

> The `Dockerfile` uses `python:3.11-slim` (the version the SIH deck commits
> to). Local dev was done on 3.13 — everything is version-agnostic.

## Data

**Real APTOS 2019** (needs a Kaggle account + accepted competition rules):

```bash
export KAGGLE_USERNAME=... KAGGLE_KEY=...          # or ~/.kaggle/kaggle.json
python -m src.data.prepare_aptos --out data/aptos
```

APTOS 2019 ships **one image per patient** with no `patient_id` column, so the
split keys on `id_code`; if a future data drop adds `patient_id`, the script
groups by it automatically.

**If the competition download 401s** (Kaggle gates the *entire* Competitions
API — even after phone verification, a new account often still needs to
re-accept site Terms in a browser and click *Join Competition*): use a mirror
served through the non-gated **Datasets API** instead —

```bash
python -m src.data.prepare_aptos --out data/aptos --kaggle-dataset mariaherrerot/aptos2019
```

`prepare_aptos.py` locates the label CSV and images regardless of the mirror's
folder layout / column names (`id_code`/`image`, `diagnosis`/`level`). A mirror
is a community re-upload, not the official source — say so in the write-up. Or
download any copy by hand and pass `--raw /path/to/unzipped`.

**Synthetic fallback** (no Kaggle needed — lets you build/test the whole
pipeline; **metrics from it are meaningless** and every artifact says so):

```bash
python -m src.data.prepare_aptos --out data/aptos --synthetic --n 400
```

Follow-up datasets (noted for later, **no pipeline built**):
- **IDRiD** — pixel-level lesion masks; use to check Grad-CAM actually lands on
  real lesions (lesion-overlap score against the raw CAM array).
- **Messidor / DDR** — cross-population generalisation testing.

## Train

```bash
# accuracy backbone (SIH tech-stack slide)
python -m src.model.train --data data/aptos/labels.csv --backbone b3

# small/edge backbone — the one that would later be INT8/TFLite-quantised for
# the offline on-device path (that conversion is out of scope here)
python -m src.model.train --data data/aptos/labels.csv --backbone edge

# quick smoke test (few minutes, CPU)
python -m src.model.train --data data/aptos/labels.csv --backbone edge \
  --epochs-head 1 --epochs-finetune 1 --batch-size 16 --device cpu
```

Stage 1 trains the head with the backbone frozen; stage 2 unfreezes and
fine-tunes at a lower LR. Loss is class-weighted (APTOS grade 0 dominates).
Best checkpoint (by val QWK) → `models/best.pt`; per-epoch log →
`models/train_log.jsonl`.

## Evaluate

```bash
python -m src.model.evaluate --data data/aptos/labels.csv --weights models/best.pt
```

Prints the report + confusion matrix, writes `models/test_metrics.json` and
`models/confusion_matrix.png`. **Headline metric: sensitivity for referable DR
(grade ≥ 2)** — not overall accuracy, which is misleading on this imbalanced
ordinal problem.

## Grad-CAM sanity check

```bash
python -m src.explain.gradcam --data data/aptos/labels.csv --weights models/best.pt \
  --split val --n 5
# -> tests/gradcam_samples/*.png  — eyeball that the hot region sits on retinal
#    structures, not the black border / lens flare.
```

## Run the API locally

```bash
export PYTHONPATH=.
export API_KEY=dev-secret
export ALLOWED_ORIGIN=http://localhost:5173        # your Base44 dev origin
# no trained weights yet? allow an untrained model so the service still starts:
export DR_ALLOW_UNTRAINED=1
uvicorn src.api.main:app --reload --port 8000
```

### Test it with curl

```bash
BASE=http://localhost:8000
KEY=dev-secret

curl -s $BASE/health | python -m json.tool

# multipart upload
curl -s -X POST $BASE/predict -H "X-API-Key: $KEY" \
  -F "file=@sample_fundus.jpg" | python -m json.tool

# base64 JSON (what the Base44 Backend Function sends)
curl -s -X POST $BASE/predict -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i sample_fundus.jpg)\"}" | python -m json.tool

# missing key -> 401
curl -s -o /dev/null -w "%{http_code}\n" -X POST $BASE/predict -F "file=@sample_fundus.jpg"
```

### Response shape

```json
{
  "usable_image": true,
  "rejection_reason": null,
  "grade": 2,
  "label": "Moderate NPDR",
  "confidence": 0.74,
  "uncertain": false,
  "referable": true,
  "referral_action": "Refer to ophthalmologist within ~6 months.",
  "heatmap_base64": "iVBORw0KG...",
  "probabilities": [0.03, 0.08, 0.74, 0.10, 0.05],
  "quality_scores": { "blur_var_laplacian": 210.4, "...": 0 },
  "disclaimer": "AI screening aid only - not a diagnosis. Confirm with an ophthalmologist."
}
```

If `usable_image` is `false`, `rejection_reason` is set and all grading fields
are `null`. If `confidence < DR_UNCERTAINTY_THRESHOLD` (default 0.6),
`"uncertain": true` — the UI must flag it for mandatory human review.

Referral table (from project research, single source of truth in
[`src/config.py`](src/config.py) `REFERRAL_ACTIONS`):

| Grade | Label | Action |
|---|---|---|
| 0 | No DR | No referral. Routine annual screening. |
| 1 | Mild NPDR | No urgent referral. Re-screen in 9–12 months. |
| 2 | Moderate NPDR | Refer to ophthalmologist within ~6 months. |
| 3 | Severe NPDR | Refer within ~1 month — treat as priority. |
| 4 | Proliferative DR | Urgent referral within days. |

## Tests

```bash
PYTHONPATH=. pytest -q
```

`tests/test_api.py` — API contract (auth, multipart + base64, quality reject,
garbage bytes, 413). `tests/test_smoke.py` — quality gate + pipeline contract +
referral/uncertainty logic. All run against an **untrained** model — they check
structure, not accuracy.

## Deploy

See [`DEPLOY.md`](DEPLOY.md) (Render, with Railway / HF Space as alternatives)
and [`base44-integration/README.md`](base44-integration/README.md) for the
Base44 wiring.

---

## Current model performance

**Not yet trained on real data in this environment.** The pipeline has only
been exercised end-to-end on the **synthetic placeholder** dataset, whose
numbers are meaningless by construction (random labels). `models/best.pt`
currently holds a synthetic-trained `edge` checkpoint purely so the API and
Grad-CAM run; `GET /health` reports `"synthetic_weights": true` for it, and
`evaluate.py` prints a warning when the data manifest is synthetic.

**To fill this in:** run `prepare_aptos.py` with Kaggle credentials, then
`train.py --backbone b3` (≈30–60 min on a single modern GPU; longer on CPU),
then `evaluate.py`. Paste the headline numbers here:

> _EfficientNet-B3, APTOS 2019 held-out test split (n = …):_
> **Referable-DR (grade ≥ 2) sensitivity: …% / specificity: …%**,
> quadratic weighted kappa **…**, per-grade AUROC …. Confusion matrix in
> `models/confusion_matrix.png`.

---

## Needs my input

1. **Real dataset access** — Kaggle token is set and valid, but this account's
   **Competitions API returns 401** for everything (list + download), while the
   Datasets API works. Fix: in a browser, log in at kaggle.com, clear any
   "accept updated Terms" banner, open the APTOS 2019 competition and click
   *Join Competition* / *I Understand and Accept*; if it still 401s, regenerate
   the API token (Settings → API → Create New Token). Meanwhile, unblock with
   `--kaggle-dataset mariaherrerot/aptos2019` (Datasets API mirror). Until real
   data is in, everything runs on synthetic data and **no accuracy number is
   real**.
2. **Actual training run** — needs a GPU box / Colab. Nothing here has been
   trained on real fundus images yet.
3. **Base44 app domain for CORS** — I put `ALLOWED_ORIGIN` as an env var with a
   placeholder. Get the real `https://<something>.base44.app` domain from the
   teammate who owns the app and set it on the deploy host (see `DEPLOY.md` §3).
4. **Deployment host** — `DEPLOY.md` is written for **Render** (free tier, Docker,
   stable HTTPS). Switch if you prefer Railway / HF Space — notes included.
5. **`API_KEY` value** — generate one (`openssl rand -hex 24`) and set it in two
   places with the **same value**: the model host env (`API_KEY`) and the Base44
   Secret `DR_MODEL_API_KEY`.
6. **Quality-gate thresholds** — blur / exposure / FOV cut-offs in
   `src/config.py` (`QualityConfig`) are tuned on synthetic + a handful of test
   images. Re-check against real rural-camera photos and adjust the
   `DR_BLUR_MIN_VAR` / `DR_EXPOSURE_*` / `DR_FOV_*` env vars.
7. **Uncertainty threshold** — starts at `0.6` (`DR_UNCERTAINTY_THRESHOLD`).
   Calibrate against a real validation set once trained.
8. **Docker build not run here** — no Docker in the build environment. The
   `Dockerfile` is written and reviewed but unbuilt; do a `docker build .` before
   relying on it for judging.
9. **On-device / INT8 path** — the `edge` backbone trains and serves, but the
   torch → ONNX → TFLite → INT8 quantisation + on-device Grad-CAM work in the SIH
   deck is **out of scope** here and not started.
