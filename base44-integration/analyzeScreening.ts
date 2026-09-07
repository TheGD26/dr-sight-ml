// Base44 backend function `analyzeScreening` — DR-Sight model version.
//
// Drop-in replacement for the original InvokeLLM implementation. Same input
// (`{ image_url }`), same response shape (dr_grade / confidence / summary /
// findings / affected_regions / recommendation / urgency), so the frontend and
// the exportScreening* functions need no changes.
//
// It fetches the uploaded fundus image, sends it to the DR-Sight FastAPI
// service (`POST /predict`, X-API-Key auth), and maps that service's response
// onto the schema the UI already renders. `affected_regions` come straight from
// the model's Grad-CAM; nothing clinical is invented here.
//
// Requires a Secret / env var in this Base44 app:
//   DR_MODEL_API_KEY  — same string as API_KEY on the DR-Sight service
//
// Update MODEL_API_URL when the dev tunnel restarts, or to the Render URL once
// the service is deployed.

import { createClientFromRequest } from 'npm:@base44/sdk@0.8.44';

const MODEL_API_URL = "https://cabinet-born-newcastle-tribes.trycloudflare.com";

const GRADE_SEVERITY: Record<number, "none" | "mild" | "moderate" | "severe"> = {
  0: "none", 1: "mild", 2: "moderate", 3: "severe", 4: "severe",
};
const GRADE_URGENCY: Record<number, "routine" | "follow-up" | "urgent" | "immediate"> = {
  0: "routine", 1: "routine", 2: "follow-up", 3: "urgent", 4: "immediate",
};

function toBase64(bytes: Uint8Array): string {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function mapResult(m: any) {
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
      // extra, non-schema fields the UI may ignore
      usable_image: false,
      rejection_reason: m.rejection_reason ?? null,
      disclaimer: m.disclaimer,
    };
  }

  const grade: number = m.grade ?? 0;
  const conf: number = typeof m.confidence === "number" ? m.confidence : 0;
  const pct = Math.round(conf * 100);

  const findings =
    grade > 0
      ? [{
          name: m.label,
          severity: GRADE_SEVERITY[grade] ?? "moderate",
          description:
            `Model-assessed diabetic retinopathy severity: ${m.label} ` +
            `(grade ${grade}). ${m.referral_action ?? ""}`.trim(),
        }]
      : [];

  let urgency = GRADE_URGENCY[grade] ?? "routine";
  // Referral escalated by the screening-biased threshold even though the argmax
  // grade is < 2 — bump routine up to follow-up.
  if (m.referral_escalated && urgency === "routine") urgency = "follow-up";

  const summary =
    `${m.label} (grade ${grade}) identified with ${pct}% model confidence.` +
    (m.uncertain
      ? " Confidence is below the review threshold — mandatory human review required."
      : "") +
    ` ${m.disclaimer ?? ""}`.trimEnd();

  return {
    dr_grade: m.label,                       // already "No DR" … "Proliferative DR"
    confidence: conf,
    summary,
    findings,
    affected_regions: Array.isArray(m.affected_regions) ? m.affected_regions : [],
    recommendation: m.referral_action ?? "Routine annual screening.",
    urgency,
    // extra, non-schema fields (safe to ignore in the UI, handy if you want them)
    grade,
    uncertain: !!m.uncertain,
    referable: m.referable ?? null,
    probabilities: m.probabilities ?? null,
    heatmap_base64: m.heatmap_base64 ?? null,
    disclaimer: m.disclaimer,
  };
}

export default async function (req: Request): Promise<Response> {
  try {
    const base44 = createClientFromRequest(req);
    const user = await base44.auth.me();
    if (!user) return Response.json({ error: 'Unauthorized' }, { status: 401 });

    const body = await req.json();
    const imageUrl = body?.image_url;
    if (!imageUrl) return Response.json({ error: 'image_url is required' }, { status: 400 });

    const apiKey = Deno.env.get('DR_MODEL_API_KEY');
    if (!apiKey) {
      return Response.json(
        { error: 'DR_MODEL_API_KEY secret is not set in this Base44 app.' },
        { status: 500 },
      );
    }

    // 1. pull the uploaded image and base64 it
    const imgResp = await fetch(imageUrl);
    if (!imgResp.ok) {
      return Response.json(
        { error: `could not fetch image_url (${imgResp.status})` },
        { status: 400 },
      );
    }
    const imageBase64 = toBase64(new Uint8Array(await imgResp.arrayBuffer()));

    // 2. call the DR-Sight model service
    const modelResp = await fetch(`${MODEL_API_URL}/predict`, {
      method: 'POST',
      headers: { 'X-API-Key': apiKey, 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64, with_heatmap: true }),
    });

    if (!modelResp.ok) {
      let detail = '';
      try { detail = (await modelResp.json())?.detail ?? ''; } catch (_) { /* non-JSON */ }
      return Response.json(
        { error: `DR model API error ${modelResp.status}: ${detail}` },
        { status: 502 },
      );
    }

    // 3. map to the shape the UI already expects
    return Response.json(mapResult(await modelResp.json()));
  } catch (error) {
    return Response.json({ error: (error as Error).message }, { status: 500 });
  }
}
