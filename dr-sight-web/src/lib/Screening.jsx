import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  ScanEye,
  UploadCloud,
  Image as ImageIcon,
  Loader2,
  AlertCircle,
  X,
} from "lucide-react";
import { analyzeScreening } from "../lib/analyzeScreening.js";
import { predictWithFormData } from "../lib/predictFormData.js";
import { Screening as Store, downscaleToDataUrl } from "../lib/store.js";
import { SAMPLE_IMAGE_URL } from "../lib/config.js";

export default function Screening() {
  const navigate = useNavigate();
  const fileInput = useRef(null);

  const [previewUrl, setPreviewUrl] = useState(null);
  const [file, setFile] = useState(null);
  const [patientId, setPatientId] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState(null);

  const accept = (f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please upload an image file (JPG, PNG).");
      return;
    }
    setError(null);
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer.files?.[0]);
  };

  const reset = () => {
    setPreviewUrl(null);
    setFile(null);
    setError(null);
  };

  const run = async (useSample = false) => {
    try {
      setBusy(true);
      setError(null);

      let result;
      let storedImageUrl;

      if (useSample) {
        setPreviewUrl(SAMPLE_IMAGE_URL);
        storedImageUrl = SAMPLE_IMAGE_URL;
        result = await analyzeScreening({ imageUrl: SAMPLE_IMAGE_URL });
      } else if (file) {
        storedImageUrl = await downscaleToDataUrl(file);
        // Real uploaded file -> multipart/form-data (smaller payload than
        // base64 JSON; matters on the rural clinic uplinks this app targets).
        result = await predictWithFormData(file);
      } else if (previewUrl) {
        storedImageUrl = previewUrl;
        result = await analyzeScreening({ imageUrl: previewUrl });
      } else {
        setError("Please upload a retinal image first.");
        setBusy(false);
        return;
      }

      const record = await Store.create({
        patient_id: patientId || null,
        image_url: storedImageUrl,
        dr_grade: result.dr_grade,
        confidence: result.confidence,
        findings: result.findings || [],
        affected_regions: result.affected_regions || [],
        recommendation: result.recommendation,
        urgency: result.urgency,
        summary: result.summary,
        heatmap_base64: result.heatmap_base64 || null,
        probabilities: result.probabilities || null,
        grade: result.grade ?? null,
        uncertain: !!result.uncertain,
        referable: result.referable ?? null,
        referral_escalated: !!result.referral_escalated,
        disclaimer: result.disclaimer || null,
        usable_image: result.usable_image !== false,
        rejection_reason: result.rejection_reason || null,
      });

      navigate(`/results/${record.id}`);
    } catch (err) {
      console.error(err);
      setError(err?.message || "Analysis failed. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="py-10 max-w-3xl mx-auto">
      <div className="text-center">
        <span className="inline-flex items-center gap-2 rounded-full bg-primary/[0.08] px-3 py-1 text-xs font-medium text-primary ring-1 ring-primary/15">
          <ScanEye className="h-3.5 w-3.5" /> New screening
        </span>
        <h1 className="mt-5 font-display text-3xl sm:text-4xl font-semibold tracking-tight">
          Upload a retinal scan
        </h1>
        <p className="mt-3 text-muted-foreground max-w-lg mx-auto">
          A macula-centered fundus photograph works best. The AI will grade
          severity and explain its reasoning.
        </p>
      </div>

      <div className="mt-10">
        <label className="block text-sm font-medium mb-2">
          Patient reference{" "}
          <span className="text-muted-foreground font-normal">(optional)</span>
        </label>
        <input
          type="text"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
          placeholder="e.g. PT-04217"
          className="w-full rounded-xl border border-input bg-card px-4 py-2.5 text-sm outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary transition"
        />
      </div>

      <AnimatePresence mode="wait">
        {previewUrl ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6"
          >
            <div className="relative rounded-2xl overflow-hidden ring-1 ring-border bg-muted">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full max-h-[420px] object-contain bg-black/5"
              />
              {!busy && (
                <button
                  onClick={reset}
                  className="absolute top-3 right-3 rounded-full bg-foreground/80 backdrop-blur p-1.5 text-background hover:bg-foreground transition"
                  aria-label="Remove"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
              <AnimatePresence>
                {busy && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="absolute inset-0 bg-background/70 backdrop-blur-sm flex flex-col items-center justify-center gap-4"
                  >
                    <Loader2 className="h-8 w-8 text-primary animate-spin" />
                    <div className="text-center">
                      <p className="font-medium">Analyzing retinal scan…</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Detecting lesions and grading severity
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {error && (
              <div className="mt-4 flex items-start gap-2 rounded-xl bg-destructive/10 px-4 py-3 text-sm text-destructive">
                <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <div className="mt-6 flex flex-wrap gap-3">
              <button
                onClick={() => run(false)}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-full bg-primary text-primary-foreground px-6 py-3 text-sm font-medium shadow-sm hover:shadow-md transition disabled:opacity-60"
              >
                {busy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <ScanEye className="h-4 w-4" strokeWidth={1.8} />
                )}
                {busy ? "Analyzing…" : "Run analysis"}
              </button>
              <button
                onClick={reset}
                disabled={busy}
                className="inline-flex items-center gap-2 rounded-full px-5 py-3 text-sm font-medium ring-1 ring-border hover:bg-muted transition disabled:opacity-60"
              >
                Replace image
              </button>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="drop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => fileInput.current?.click()}
            className={`mt-6 cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-colors ${
              dragging
                ? "border-primary bg-primary/5"
                : "border-border hover:border-primary/40 hover:bg-muted/50"
            }`}
          >
            <input
              ref={fileInput}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => accept(e.target.files?.[0])}
            />
            <div className="mx-auto h-14 w-14 rounded-2xl bg-primary/10 flex items-center justify-center ring-1 ring-primary/15">
              <UploadCloud className="h-6 w-6 text-primary" strokeWidth={1.6} />
            </div>
            <p className="mt-5 font-medium">Drag &amp; drop or click to upload</p>
            <p className="mt-1 text-sm text-muted-foreground">
              JPG or PNG · fundus photograph recommended
            </p>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                run(true);
              }}
              disabled={busy}
              className="mt-6 inline-flex items-center gap-2 text-sm font-medium text-primary hover:underline disabled:opacity-60"
            >
              <ImageIcon className="h-4 w-4" strokeWidth={1.7} />
              or try with a sample image
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      <p className="mt-8 text-xs text-muted-foreground text-center max-w-lg mx-auto leading-relaxed">
        Decision support only — not a medical diagnosis. Always confirm results
        with a qualified eye-care professional.
      </p>
    </div>
  );
}
