// Single source of truth for the DR severity scale, urgency tiers and finding
// severities — mirrors the maps used across the original Retina UI.

export const GRADE_META = {
  "No DR": {
    label: "No DR",
    color: "text-emerald-700 bg-emerald-50 ring-emerald-200",
    dot: "bg-emerald-500",
    blurb: "No abnormalities detected",
    learn: "No abnormalities detected — routine annual rescreen.",
  },
  "Mild NPDR": {
    label: "Mild",
    color: "text-amber-700 bg-amber-50 ring-amber-200",
    dot: "bg-amber-400",
    blurb: "Microaneurysms only",
    learn: "Microaneurysms only — the earliest sign. Rescreen within 12 months.",
  },
  "Moderate NPDR": {
    label: "Moderate",
    color: "text-orange-700 bg-orange-50 ring-orange-200",
    dot: "bg-orange-500",
    blurb: "Hemorrhages, exudates, cotton-wool spots",
    learn:
      "Hemorrhages, hard exudates, cotton-wool spots, venous beading. Refer to ophthalmology.",
  },
  "Severe NPDR": {
    label: "Severe",
    color: "text-red-700 bg-red-50 ring-red-200",
    dot: "bg-red-500",
    blurb: "Extensive hemorrhages, IRMA, venous beading",
    learn:
      "Extensive hemorrhages, IRMA, venous beading (4-2-1 rule). Prompt referral.",
  },
  "Proliferative DR": {
    label: "Proliferative",
    color: "text-rose-800 bg-rose-50 ring-rose-300",
    dot: "bg-rose-600",
    blurb: "Neovascularization, vitreous hemorrhage",
    learn:
      "Neovascularization or vitreous/preretinal hemorrhage. Immediate referral.",
  },
};

export const GRADE_ORDER = Object.keys(GRADE_META);

export const GRADE_HEX = {
  "No DR": "#10b981",
  "Mild NPDR": "#d97706",
  "Moderate NPDR": "#ea580c",
  "Severe NPDR": "#dc2626",
  "Proliferative DR": "#be185d",
};

export const URGENCY_META = {
  routine: {
    label: "Routine rescreen",
    color: "text-emerald-700 bg-emerald-50 ring-emerald-200",
  },
  "follow-up": {
    label: "Follow-up advised",
    color: "text-amber-700 bg-amber-50 ring-amber-200",
  },
  urgent: {
    label: "Urgent referral",
    color: "text-orange-700 bg-orange-50 ring-orange-200",
  },
  immediate: {
    label: "Immediate referral",
    color: "text-rose-700 bg-rose-50 ring-rose-200",
  },
};

export const URGENCY_HEX = {
  routine: "#10b981",
  "follow-up": "#f59e0b",
  urgent: "#f97316",
  immediate: "#e11d48",
};

export const URGENCY_SHORT = {
  routine: "Routine",
  "follow-up": "Follow-up",
  urgent: "Urgent referral",
  immediate: "Immediate referral",
};

export const SEVERITY_CLASS = {
  none: "text-muted-foreground bg-muted",
  mild: "text-amber-700 bg-amber-50",
  moderate: "text-orange-700 bg-orange-50",
  severe: "text-rose-700 bg-rose-50",
};
