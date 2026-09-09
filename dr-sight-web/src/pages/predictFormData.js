// dr-sight-web/src/lib/predictFormData.js
//
// Multipart/form-data path for POST /predict — used for the common case where
// the user picked/dropped an actual image File (see Screening.jsx). Returns
// the SAME mapped shape as analyzeScreening() (dr_grade/summary/findings/...)
// by reusing its mapResult(), so callers can swap one call for the other
// without touching anything downstream (Store.create, Results.jsx, etc).
//
// Why this exists alongside analyzeScreening.js rather than replacing it:
//   - analyzeScreening.js also handles the "try with a sample image" flow,
//     which fetches a remote URL and re-encodes it — base64 JSON is the
//     simplest way to shuttle that through the SAME backend contract.
//   - For a real uploaded File, multipart avoids the ~33% base64 size
//     overhead, which matters here specifically: the Simulink workflow model
//     (matlab/simulink/run_workflow_sim.m) identifies rural clinic uplink
//     bandwidth as a real bottleneck for this programme, so shaving upload
//     size on the actual hot path (a phone/camera uploading a fundus photo
//     over a clinic's 5 Mbps line) is a genuine, not cosmetic, improvement.
//
// Backend contract (src/api/main.py, POST /predict):
//   - Accepts EITHER multipart/form-data with a `file` field, OR
//     application/json {"image": "<base64>", "with_heatmap": true}.
//   - Auth: every call needs header `X-API-Key: <key>` (server responds 401
//     without it, 503 if the server itself has no API_KEY configured).
//   - Raw response shape (src/inference/pipeline.py::PredictionResult) is
//     mapped by analyzeScreening.js's mapResult() into the UI schema this
//     app's Store/Results page expect.

import { MODEL_BASE_URL, MODEL_API_KEY } from "./config.js";
import { mapResult } from "./analyzeScreening.js";

/**
 * Screen a fundus image file via multipart/form-data.
 *
 * @param {File} file - the image file the user selected or dropped.
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {Promise<object>} the same mapped shape analyzeScreening() returns
 *   (dr_grade, confidence, summary, findings, affected_regions,
 *   recommendation, urgency, plus the raw model extras).
 * @throws {Error} with a human-readable message on network failure, a non-2xx
 *   response (including the server's `detail` string when present), or an
 *   obviously-wrong input (not a File, or not an image/* MIME type).
 */
export async function predictWithFormData(file, opts = {}) {
  const { signal } = opts;

  if (!(file instanceof File)) {
    throw new Error("predictWithFormData: `file` must be a File instance.");
  }
  if (!file.type.startsWith("image/")) {
    throw new Error(
      `predictWithFormData: expected an image file, got "${file.type || "unknown type"}".`,
    );
  }

  // Build the multipart body. Do NOT set a Content-Type header yourself here —
  // the browser sets `multipart/form-data; boundary=...` automatically when
  // the fetch body is a FormData instance, and overriding it manually is a
  // classic bug (it drops the boundary and the server can't parse the body).
  const form = new FormData();
  form.append("file", file, file.name);
  // NOTE: main.py's multipart branch always runs with heatmaps on today (only
  // the JSON-body path reads `with_heatmap` — see `PredictJSON`). If you need
  // to disable the heatmap on the multipart path too, add a `with_heatmap`
  // form field here AND read `request.form()` for it server-side; not done
  // here to avoid touching the backend for what is currently an unused case.

  let response;
  try {
    response = await fetch(`${MODEL_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        // Only the API key header — never set Content-Type for FormData.
        ...(MODEL_API_KEY ? { "X-API-Key": MODEL_API_KEY } : {}),
      },
      body: form,
      signal,
    });
  } catch (err) {
    // fetch() throws only on network-level failure (DNS, connection refused,
    // CORS preflight rejection, offline) — never on a non-2xx HTTP status.
    if (err.name === "AbortError") throw err; // let callers handle cancellation
    throw new Error(
      `DR model API unreachable — is the FastAPI service running at ${MODEL_BASE_URL}? (${err.message})`,
    );
  }

  if (!response.ok) {
    // FastAPI's HTTPException body is {"detail": "..."} — surface it when present
    // (e.g. 401 bad key, 413 too large, 422 "Could not process image: <MATLAB error>",
    // 503 MATLAB engine/ONNX not ready). Fall back to the bare status otherwise.
    let detail = "";
    try {
      const body = await response.json();
      detail = body?.detail ?? "";
    } catch {
      /* response wasn't JSON — ignore, use bare status below */
    }
    throw new Error(
      `DR model API error ${response.status}${detail ? `: ${detail}` : ""}`,
    );
  }

  return mapResult(await response.json());
}
