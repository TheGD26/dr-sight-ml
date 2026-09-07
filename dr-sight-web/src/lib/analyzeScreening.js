// Client-side port of dr-sight-ml/base44-integration/analyzeScreening.ts.
//
// Sends a fundus image to the DR-Sight FastAPI service (POST /predict,
// X-API-Key auth) and maps the service response onto the schema the Retina UI
// renders (dr_grade / confidence / summary / findings / affected_regions /
// recommendation / urgency), keeping the model's extra fields (heatmap,
// probabilities, uncertainty, referable) alongside.

import { MODEL_BASE_URL, MODEL_API_KEY } from "./config.js";

const GRADE_SEVERITY = {
  0: "none",
  1: "mild",
  2: "moderate",
  3: "severe",
  4: "severe",
};
const GRADE_URGENCY = {
  0: "routine",
  1: "routine",
  2: "follow-up",
  3: "urgent",
  4: "immediate",
};

function toBase64(bytes) {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

async function fileToBase64(file) {
  const buf = await file.arrayBuffer();
  return toBase64(new Uint8Array(buf));
}

async function urlToBase64(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`could not fetch image (${resp.status})`);
  return toBase64(new Uint8Array(await resp.arrayBuffer()));
}

function mapResult(m) {
  // Quality gate rejected the photo — mirror the original "unreadable image"
  // fallback so the UI handles it the same way.
  if (m.usable_image === false) {
    return {
      dr_grade: "No DR",
      confidence: 0.1,
      summary:
        `The image could not be interpreted as a retinal fundus photograph` +
        (m.rejection_reason ? ` (${m.rejection_reason})` : "") +
        `. Please upload a clear, macula-centered fundus photo.`,
      findings: [],
      affected_regions: [],
      recommendation:
        "Re-capture the fundus image with better focus, lighting and framing, then screen again.",
      urgency: "routine",
      usable_image: false,
      rejection_reason: m.rejection_reason ?? null,
      disclaimer: m.disclaimer,
    };
  }

  const grade = m.grade ?? 0;
  const conf = typeof m.confidence === "number" ? m.confidence : 0;
  const pct = Math.round(conf * 100);

  const findings =
    grade > 0
      ? [
          {
            name: m.label,
            severity: GRADE_SEVERITY[grade] ?? "moderate",
            description:
              `Model-assessed diabetic retinopathy severity: ${m.label} ` +
              `(grade ${grade}). ${m.referral_action ?? ""}`.trim(),
          },
        ]
      : [];

  let urgency = GRADE_URGENCY[grade] ?? "routine";
  if (m.referral_escalated && urgency === "routine") urgency = "follow-up";

  const summary =
    `${m.label} (grade ${grade}) identified with ${pct}% model confidence.` +
    (m.uncertain
      ? " Confidence is below the review threshold — mandatory human review required."
      : "") +
    ` ${m.disclaimer ?? ""}`.trimEnd();

  return {
    dr_grade: m.label,
    confidence: conf,
    summary,
    findings,
    affected_regions: Array.isArray(m.affected_regions)
      ? m.affected_regions
      : [],
    recommendation: m.referral_action ?? "Routine annual screening.",
    urgency,
    // model extras kept for the UI
    grade,
    uncertain: !!m.uncertain,
    referable: m.referable ?? null,
    p_referable: m.p_referable ?? null,
    referral_escalated: !!m.referral_escalated,
    probabilities: m.probabilities ?? null,
    heatmap_base64: m.heatmap_base64 ?? null,
    disclaimer: m.disclaimer,
  };
}

/**
 * @param {{ file?: File, imageUrl?: string }} input
 * @returns mapped screening result
 */
export async function analyzeScreening({ file, imageUrl }) {
  let imageBase64;
  if (file) imageBase64 = await fileToBase64(file);
  else if (imageUrl) imageBase64 = await urlToBase64(imageUrl);
  else throw new Error("analyzeScreening: a file or imageUrl is required.");

  let resp;
  try {
    resp = await fetch(`${MODEL_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(MODEL_API_KEY ? { "X-API-Key": MODEL_API_KEY } : {}),
      },
      body: JSON.stringify({ image: imageBase64, with_heatmap: true }),
    });
  } catch (err) {
    throw new Error(
      `DR model API unreachable — is the service running and reachable at its configured URL? (${err.message})`,
    );
  }

  if (!resp.ok) {
    let detail = "";
    try {
      detail = (await resp.json())?.detail ?? "";
    } catch {
      /* non-JSON body */
    }
    throw new Error(`DR model API error ${resp.status}${detail ? `: ${detail}` : ""}`);
  }

  return mapResult(await resp.json());
}
