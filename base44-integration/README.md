# Wiring the DR-Sight model API into the Base44 app

This is for **the teammate who owns the Base44 app**. The ML side is a plain
REST API (deployed per `../DEPLOY.md`); Base44 talks to it through a **Backend
Function** so the API key never reaches the browser.

```
[Base44 UI  "Analyze" button]
      | base44.functions.predictDR({ imageBase64 })
      v
[Base44 Backend Function  predictDR.js]   <-- holds DR_MODEL_API_KEY secret
      | POST /predict   (X-API-Key header)
      v
[DR-Sight model API on Render/Railway/HF]  --> grade + Grad-CAM + referral JSON
```

---

## Step 1 — Add the API key as a Secret

1. Open your app in Base44 → **Settings → Secrets** (a.k.a. Environment
   Variables / Server Secrets, depending on your Base44 version).
2. Add:
   - **Name:** `DR_MODEL_API_KEY`
   - **Value:** the exact string set as `API_KEY` on the deployed model service
     (ask me for it, or generate one together with `openssl rand -hex 24` and we
     both set the same value).
3. Save. Secrets are only readable from Backend Functions, never from the
   client — that's the whole point.

## Step 2 — Create the Backend Function

1. Base44 → **Backend Functions** (or **Functions → New Function**).
2. Name it `predictDR`.
3. Paste the contents of [`predictDR.js`](./predictDR.js).
4. Change the one line:
   ```js
   const MODEL_API_URL = "https://<your-deployed-model-url>";
   ```
   to the real URL from deployment, e.g. `https://dr-sight.onrender.com`
   (no trailing slash).
5. Save / deploy the function.

> If Base44 generated a TypeScript function instead, the body is identical —
> just add the param type `({ imageBase64 }: { imageBase64: string })` and keep
> the rest.

## Step 3 — Call it from the "Analyze" action

Wherever the screening flow gets the fundus image (file input, camera capture),
convert it to base64 and call the function.

```js
// in the component / action behind the "Analyze" button
async function handleAnalyze(file) {
  const imageBase64 = await fileToBase64(file); // "data:image/jpeg;base64,..." is fine

  const result = await base44.functions.predictDR({ imageBase64 });
  // (some Base44 SDK versions: await base44.invokeFunction("predictDR", { imageBase64 }))

  setScreening(result); // -> render per Step 4
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result); // data URI; the API strips the prefix
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}
```

## Step 4 — Render the result

```jsx
{result.usable_image === false ? (
  <Callout tone="warning">
    Image rejected: {result.rejection_reason}
    <br />Please retake the photo.
  </Callout>
) : (
  <div>
    <h3>{result.label} (grade {result.grade})</h3>
    <p>Confidence: {(result.confidence * 100).toFixed(0)}%</p>

    {result.uncertain && (
      <Callout tone="warning">
        Low confidence - mandatory human review before acting on this result.
      </Callout>
    )}

    <p><strong>Recommended action:</strong> {result.referral_action}</p>

    {result.heatmap_base64 && (
      <img
        alt="Grad-CAM heatmap - regions the model focused on"
        src={`data:image/png;base64,${result.heatmap_base64}`}
        style={{ maxWidth: 360, borderRadius: 8 }}
      />
    )}

    <small>{result.disclaimer}</small>
  </div>
)}
```

Field reference:

| Field | Type | Use in UI |
|---|---|---|
| `usable_image` | bool | `false` → show `rejection_reason`, ask for a retake, stop. |
| `rejection_reason` | string \| null | Why the quality gate rejected it (blur / exposure / field of view). |
| `grade` | 0–4 \| null | Severity. |
| `label` | string \| null | "No DR" … "Proliferative DR". |
| `confidence` | number \| null | Top-1 softmax, 0–1. |
| `uncertain` | bool | `true` → visually flag for mandatory human review. |
| `referable` | bool \| null | `grade >= 2`. Handy for a "needs referral" badge. |
| `referral_action` | string \| null | The sentence to show the health worker. |
| `heatmap_base64` | string \| null | PNG; `<img src="data:image/png;base64,…">`. |
| `probabilities` | number[] \| null | Per-grade softmax, if you want a bar chart. |
| `disclaimer` | string | Always display. Screening aid, not a diagnosis. |

## Testing without the UI

```bash
# base64 a local image and call the deployed function's underlying API directly
curl -s -X POST https://<your-deployed-model-url>/predict \
  -H "X-API-Key: <DR_MODEL_API_KEY value>" \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -i sample_fundus.jpg)\"}" | python -m json.tool
```

---

## Alternative: Custom Integration (OpenAPI) — *not built here*

If more than one Base44 app needs the model, you can register the API once at
the **workspace** level as a **Custom Integration** by importing an OpenAPI
spec (FastAPI serves one at `GET /openapi.json` on the deployed service). Base44
then exposes it to every app without a per-app Backend Function.

For a single app the Backend Function above is simpler, so that's the
recommended path — mentioned here only so you know the option exists.
