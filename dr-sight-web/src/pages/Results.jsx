import { useEffect, useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Eye,
  ClipboardList,
  ClipboardCheck,
  NotebookPen,
  Check,
  Loader2,
  AlertCircle,
  AlertTriangle,
  ChevronLeft,
  Activity,
  Download,
} from "lucide-react";
import GradeBadge from "../components/GradeBadge.jsx";
import HeatmapOverlay from "../components/HeatmapOverlay.jsx";
import { Screening as Store } from "../lib/store.js";
import { URGENCY_META, SEVERITY_CLASS } from "../lib/grades.js";

export default function Results() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [record, setRecord] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [notes, setNotes] = useState("");
  const [savingNotes, setSavingNotes] = useState(false);
  const [savedFlash, setSavedFlash] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const r = await Store.get(id);
        setRecord(r);
        setNotes(r.notes || "");
      } catch (e) {
        setError(e?.message || "Screening not found");
      } finally {
        setLoading(false);
      }
    })();
  }, [id]);

  const saveNotes = async () => {
    try {
      setSavingNotes(true);
      const r = await Store.update(id, { notes });
      setRecord(r);
      setSavedFlash(true);
      setTimeout(() => setSavedFlash(false), 1800);
    } catch (e) {
      setError(e?.message || "Could not save notes");
    } finally {
      setSavingNotes(false);
    }
  };

  if (loading)
    return (
      <div className="py-32 flex flex-col items-center gap-3 text-muted-foreground">
        <Loader2 className="h-7 w-7 text-primary animate-spin" />
        <p className="text-sm">Loading screening…</p>
      </div>
    );

  if (error || !record)
    return (
      <div className="py-32 flex flex-col items-center gap-4 text-center">
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-muted-foreground">{error || "Screening not found."}</p>
        <Link
          to="/history"
          className="text-primary text-sm font-medium hover:underline"
        >
          Back to history
        </Link>
      </div>
    );

  const urgency = URGENCY_META[record.urgency] || URGENCY_META.routine;
  const confPct = Math.round((record.confidence || 0) * 100);
  const regions = record.affected_regions || [];

  return (
    <div className="py-8 max-w-5xl mx-auto">
      <Link
        to="/history"
        className="no-print inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition mb-6"
      >
        <ChevronLeft className="h-4 w-4" /> Back to history
      </Link>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex flex-wrap items-center gap-3">
          <GradeBadge grade={record.dr_grade} />
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset ${urgency.color}`}
          >
            <Activity className="h-3 w-3" /> {urgency.label}
          </span>
          {record.referable ? (
            <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ring-inset text-orange-700 bg-orange-50 ring-orange-200">
              Needs referral
            </span>
          ) : null}
          {record.patient_id && (
            <span className="text-sm text-muted-foreground">
              · {record.patient_id}
            </span>
          )}
        </div>

        <h1 className="mt-4 font-display text-3xl sm:text-4xl font-semibold tracking-tight">
          {record.dr_grade}
        </h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">{record.summary}</p>
      </motion.div>

      {record.uncertain && (
        <div className="mt-6 flex items-start gap-2.5 rounded-xl bg-amber-50 ring-1 ring-amber-200 px-4 py-3 text-sm text-amber-800">
          <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
          <span>
            Model confidence is below the review threshold — this result requires
            mandatory human review before any action is taken.
          </span>
        </div>
      )}

      <div className="mt-8 grid lg:grid-cols-2 gap-8">
        {/* left: explainability */}
        <div>
          <div className="flex items-center gap-2 mb-3">
            <Eye className="h-4 w-4 text-primary" strokeWidth={1.7} />
            <h2 className="font-medium text-sm">Explainability overlay</h2>
            <span className="text-xs text-muted-foreground">
              {record.heatmap_base64 ? "— Grad-CAM" : "— hover regions"}
            </span>
          </div>
          <HeatmapOverlay
            imageUrl={record.image_url}
            heatmap={record.heatmap_base64}
            regions={regions}
          />
          {regions.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {regions.map((r, i) => (
                <span
                  key={i}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs"
                >
                  <span
                    className="h-1.5 w-1.5 rounded-full bg-orange-500"
                    style={{ opacity: r.intensity }}
                  />
                  {r.label}
                </span>
              ))}
            </div>
          )}
          {record.probabilities && (
            <div className="mt-5 rounded-2xl bg-card ring-1 ring-border p-5">
              <h3 className="text-sm font-medium mb-3">Per-grade probability</h3>
              <div className="space-y-2">
                {["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"].map(
                  (label, i) => {
                    const p = record.probabilities[i] ?? 0;
                    return (
                      <div key={label} className="flex items-center gap-3 text-xs">
                        <span className="w-28 shrink-0 text-muted-foreground">
                          {label}
                        </span>
                        <div className="flex-1 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div
                            className="h-full rounded-full bg-primary"
                            style={{ width: `${Math.round(p * 100)}%` }}
                          />
                        </div>
                        <span className="w-9 text-right tabular-nums">
                          {Math.round(p * 100)}%
                        </span>
                      </div>
                    );
                  },
                )}
              </div>
            </div>
          )}
        </div>

        {/* right: details */}
        <div className="space-y-6">
          <div className="rounded-2xl bg-card ring-1 ring-border p-6">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Confidence</span>
              <span className="font-display text-2xl font-semibold">
                {confPct}%
              </span>
            </div>
            <div className="mt-3 h-2 rounded-full bg-muted overflow-hidden">
              <motion.div
                className="h-full rounded-full bg-primary"
                initial={{ width: 0 }}
                animate={{ width: `${confPct}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
              />
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Model confidence in the assigned severity grade.
            </p>
          </div>

          <div className="rounded-2xl bg-card ring-1 ring-border p-6">
            <div className="flex items-center gap-2 mb-4">
              <ClipboardList className="h-4 w-4 text-primary" strokeWidth={1.7} />
              <h2 className="font-medium text-sm">Clinical findings</h2>
            </div>
            {(record.findings || []).length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No significant findings detected.
              </p>
            ) : (
              <ul className="space-y-4">
                {record.findings.map((f, i) => (
                  <li key={i} className="flex gap-3">
                    <span
                      className={`mt-0.5 inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium capitalize ${
                        SEVERITY_CLASS[f.severity] || SEVERITY_CLASS.none
                      }`}
                    >
                      {f.severity}
                    </span>
                    <div>
                      <p className="text-sm font-medium">{f.name}</p>
                      <p className="text-sm text-muted-foreground mt-0.5">
                        {f.description}
                      </p>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="rounded-2xl bg-card ring-1 ring-border p-6">
            <div className="flex items-center gap-2 mb-3">
              <ClipboardCheck className="h-4 w-4 text-primary" strokeWidth={1.7} />
              <h2 className="font-medium text-sm">Recommendation</h2>
            </div>
            <p className="text-sm text-foreground/90 leading-relaxed">
              {record.recommendation}
            </p>
          </div>

          <div className="no-print rounded-2xl bg-card ring-1 ring-border p-6">
            <div className="flex items-center gap-2 mb-3">
              <NotebookPen className="h-4 w-4 text-primary" strokeWidth={1.7} />
              <h2 className="font-medium text-sm">Clinician notes</h2>
            </div>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add your review notes…"
              rows={3}
              className="w-full rounded-xl border border-input bg-background px-3 py-2.5 text-sm outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition resize-none"
            />
            <button
              onClick={saveNotes}
              disabled={savingNotes}
              className="mt-3 inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-4 py-2 text-sm font-medium hover:shadow-md transition disabled:opacity-60"
            >
              {savingNotes ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Check className="h-3.5 w-3.5" />
              )}
              {savingNotes ? "Saving…" : savedFlash ? "Saved" : "Save notes"}
            </button>
          </div>

          {record.disclaimer && (
            <p className="text-xs text-muted-foreground leading-relaxed">
              {record.disclaimer}
            </p>
          )}
        </div>
      </div>

      <div className="no-print mt-10 flex flex-wrap gap-3">
        <Link
          to="/screening"
          className="inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-5 py-2.5 text-sm font-medium hover:shadow-md transition"
        >
          New screening
        </Link>
        <button
          onClick={() => navigate("/history")}
          className="inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium ring-1 ring-border hover:bg-muted transition"
        >
          View all screenings
        </button>
        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded-full bg-card px-5 py-2.5 text-sm font-medium ring-1 ring-primary/30 text-primary hover:bg-primary/5 hover:shadow-md transition"
        >
          <Download className="h-4 w-4" strokeWidth={1.8} />
          Download PDF report
        </button>
      </div>
    </div>
  );
}
