// Base44 Backend Function - runs SERVER-SIDE inside Base44, never in the browser.
// This is what keeps the model API key off the client. Do NOT call the DR model
// API directly from a page / component with the key attached - anyone with dev
// tools would see it.
//
// Setup (see base44-integration/README.md for the full walkthrough):
//   1. In your Base44 app settings, add a Secret named  DR_MODEL_API_KEY
//      with the same value as the API_KEY env var on the deployed model service.
//   2. Set MODEL_API_URL below to your deployed URL (e.g. Render).
//   3. Create a Backend Function from this file.
//   4. Call it from the UI:  base44.functions.predictDR({ imageBase64 })
//
// The DR model /predict endpoint accepts JSON: { image: "<base64>", with_heatmap: bool }
// `image` may be a bare base64 string or a full data URI ("data:image/png;base64,...").

// TEMPORARY dev tunnel to Giridharan's laptop (cloudflared quick tunnel).
// It only works while that machine runs `./scripts/serve-local.sh` + the tunnel,
// and the hostname changes every restart. Swap in the permanent Render/Railway
// URL once the API is deployed (see ../DEPLOY.md).
const MODEL_API_URL = "https://cabinet-born-newcastle-tribes.trycloudflare.com";

export default async function predictDR({ imageBase64, withHeatmap = true }) {
  if (!imageBase64 || typeof imageBase64 !== "string") {
    throw new Error("predictDR: `imageBase64` (string) is required.");
  }

  const apiKey = process.env.DR_MODEL_API_KEY;
  if (!apiKey) {
    throw new Error(
      "predictDR: DR_MODEL_API_KEY secret is not set in this Base44 app."
    );
  }

  let response;
  try {
    response = await fetch(`${MODEL_API_URL}/predict`, {
      method: "POST",
      headers: {
        "X-API-Key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image: imageBase64, with_heatmap: withHeatmap }),
    });
  } catch (networkErr) {
    throw new Error(`DR model API unreachable: ${networkErr.message}`);
  }

  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail || "";
    } catch (_) {
      /* non-JSON error body */
    }
    throw new Error(`DR model API error ${response.status}: ${detail}`);
  }

  const result = await response.json();

  // Pass the screening result straight through to the caller. Shape:
  // {
  //   usable_image: boolean,
  //   rejection_reason: string | null,
  //   grade: 0..4 | null,
  //   label: "No DR" | "Mild NPDR" | "Moderate NPDR" | "Severe NPDR" | "Proliferative DR" | null,
  //   confidence: number | null,          // top-1 softmax, 0..1
  //   uncertain: boolean,                 // true => flag for mandatory human review
  //   referable: boolean | null,          // P(grade>=2) >= server threshold
  //   p_referable: number | null,         // softmax mass on grades >= 2
  //   referral_escalated: boolean,        // true => referral driven by threshold, not argmax grade
  //   referral_action: string | null,
  //   heatmap_base64: string | null,      // PNG, render as data:image/png;base64,...
  //   probabilities: number[] | null,
  //   disclaimer: string
  // }
  return result;
}
