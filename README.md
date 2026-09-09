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
| `src/inference/pipeline.py` | **Reference PyTorch pipeline.** quality gate → preprocess → model → Grad-CAM → referral lookup + uncertainty flag. Used only when `SCREENING_BACKEND=python` (local debugging), and for training-time validation. |
| `src/inference/matlab_pipeline.py` | **Default screening backend.** Owns one long-lived MATLAB Engine session, imports `dr_sight.onnx` once, and runs `matlab/screen_image.m` per request. Returns the same dict shape as `PredictionResult.to_dict()`. |
| `matlab/screen_image.m` | The MATLAB orchestrator: `quality_gate.m` → `grade_dr.m` → `explain_gradcam.m`, mirroring all five stages of `DRPipeline.run()`. |
| `src/api/main.py` | `POST /predict`, `GET /health`, `X-API-Key` auth, CORS allow-list, body-size limit. Thin transport — dispatches to the backend named by `SCREENING_BACKEND`. |
| `scripts/compare_backends.py` | Screen sample images through **both** backends and print grade / confidence / P(referable) + max |Δ softmax| — proves the MATLAB path reproduces the validated PyTorch numbers. |
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

## Screening backend (MATLAB by default)

Under PS **SIH26038** the live service does **all** screening computation —
image-quality assessment, DR grading, Grad-CAM explainability — **inside
MATLAB**, via `matlab/screen_image.m` called through the MATLAB Engine API for
Python (`src/inference/matlab_pipeline.py`). FastAPI is a thin HTTP transport
only: it handles auth, CORS and the body-size cap, then hands the image bytes to
one long-lived MATLAB session that has `dr_sight.onnx` imported once. **PyTorch
does not run in the request path** — it remains only as historical training code
and for the one-time `scripts/export_onnx.py`.

`src/api/main.py` reads `SCREENING_BACKEND` at load time:

| `SCREENING_BACKEND` | pipeline |
|---|---|
| `matlab` *(default)* | MATLAB does everything (`screen_image.m`). Required for PS26038. |
| `python` | the reference `src/inference/pipeline.py` PyTorch pipeline — local debugging / comparing against the original numbers only. |

**One-time install** (into the interpreter that runs FastAPI):

```bash
matlab -batch "disp(matlabroot)"                 # find <matlabroot>
cd "<matlabroot>/extern/engines/python"
python -m pip install .                           # R2022b+  (older: python setup.py install)
python -c "import matlab.engine; print('ok')"
```

Also add the **Deep Learning Toolbox Converter for ONNX Model Format** support
package in MATLAB (Add-On Explorer). Details: `requirements.txt` and
[`matlab/README.md`](matlab/README.md).

Check the MATLAB path reproduces the validated 89.7 % / 93.8 % numbers:

```bash
python scripts/compare_backends.py
```

## Run the API locally

```bash
export PYTHONPATH=.
export API_KEY=dev-secret
export ALLOWED_ORIGIN=http://localhost:5173        # your Base44 dev origin
export SCREENING_BACKEND=matlab                    # default; needs the MATLAB Engine API
# local debugging without MATLAB? use the reference PyTorch pipeline instead:
#   export SCREENING_BACKEND=python
#   export DR_ALLOW_UNTRAINED=1                     # start even with no trained weights
uvicorn src.api.main:app --reload --port 8000
```

On boot the app eagerly starts the MATLAB engine and imports the network once
(logged with timings); `GET /health` reports `screening_backend`, whether the
engine started, and whether the ONNX network imported. The `/predict`
request/response contract is **identical** for both backends — the frontend
needs zero changes.

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
  "p_referable": 0.89,
  "referral_escalated": false,
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

**Referable flag (screening-biased).** `grade` is always the model's argmax
point estimate. `referable` is decided separately: `p_referable` (the softmax
mass on grades ≥ 2) ≥ `DR_REFERABLE_THRESHOLD` (default `0.5`). Lower the env
var (e.g. `0.35`) to catch more referable DR at the cost of specificity — the
right trade for a screening tool. When a low threshold makes a grade-0/1 image
referable, `referral_escalated` is `true` and `referral_action` is bumped to
the "refer within ~6 months" tier. `python -m src.model.evaluate` prints a
sensitivity/specificity-vs-threshold sweep to help pick the value.

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

Trained on **real APTOS 2019** — 3 662 fundus images (one per patient),
patient-level stratified split 2 563 / 550 / 549 (train / val / test). The
checkpoint in `models/best.pt` is the **`edge` backbone (MobileNetV3-Small,
~2.5 M params)**, picked here for fast CPU inference; the larger `b3` backbone
should match or beat these numbers but has not been trained in this
environment. `GET /health` reports `"synthetic_weights": false` for this
checkpoint.

Held-out **test** split (n = 549), from `models/test_metrics.json`:

| Metric | Value |
|---|---|
| **Referable-DR (grade ≥ 2) sensitivity** | **89.7%** |
| Referable-DR specificity | 93.8% |
| Referable-DR precision / F1 | 91.0% / 0.90 |
| Quadratic weighted kappa | **0.871** |
| Overall accuracy | 78.1% |
| Per-grade AUROC (grades 0–4) | 0.99 / 0.94 / 0.92 / 0.91 / 0.91 |

Per-grade sensitivity drops on the rare high grades (grade 0 97%, 1 64%,
2 65%, 3 41%, 4 49%) — expected on APTOS's heavy imbalance (grade 0 is ~half
the data), and the reason the headline metric is *referable-DR sensitivity*
rather than accuracy: nearly all grade-3/4 errors fall into grade 2, which is
still a referral (only 3 of 224 referable cases are missed as
non-referable). Full confusion matrix in `models/confusion_matrix.png`.

> _Reproduce:_ `prepare_aptos.py --kaggle-dataset mariaherrerot/aptos2019`
> → `train.py --backbone edge` → `evaluate.py --weights models/best.pt`.

---

## Needs my input

1. **Real dataset access** — *resolved.* The official Competitions API still
   401s on this account, so data was pulled from the non-gated Datasets mirror
   `--kaggle-dataset mariaherrerot/aptos2019` (a community re-upload — flag it
   as such in the write-up). To use the official source, in a browser log in at
   kaggle.com, clear any "accept updated Terms" banner, open the APTOS 2019
   competition and click *Join Competition*; if it still 401s, regenerate the
   API token (Settings → API → Create New Token).
2. **Actual training run** — *done* for the `edge` backbone (see *Current model
   performance*). The `b3` backbone still needs a GPU run for its own numbers.
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
